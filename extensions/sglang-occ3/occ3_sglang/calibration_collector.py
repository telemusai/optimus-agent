"""Digest-attested collection of the protected DGX-local calibration corpus.

The per-message StateBridge fit degenerates because it fits on the rows it must
map (see statebridge_calibrated.py). The independent-calibration campaign
therefore collects whole source turns once, freezes them on protected
DGX-local storage, and fits on that corpus later. This module is the collector
half; the latent worker owns the capture hook and the real embedding lookup
and calls this minimal surface:

    collector = Collector(provenance=<caller identity strings>)
    collector.add(CaptureRecord(...), embedding_lookup=<runtime callable>)
    summary = collector.write(dirpath)          # corpus files + manifest.json
    partitions = load(dirpath)                  # verified split manifest shape
    manifest = read_manifest(dirpath)           # verified, receipt-safe

Rules carried over from spec sections 6, 7, 14 and 15:

- A turn is the unit. CaptureRecord carries the SELECTED semantic-span rows
  under the predictor convention (the state at absolute position p predicts
  the accepted token at p+1; the constant is PREDICTOR_POSITION_OFFSET = -1
  and the caller supplies the row alignment). The collector never pads and
  never duplicates rows; it refuses anything it cannot attest.
- Reference embeddings come from a runtime-supplied callable (token id ->
  [D] row). This module only consumes it: unknown ids are refused, never
  fabricated, and every refusal is recorded in a bounded list.
- write() splits whole turns through calibration_corpus.split_manifest, so
  families stay atomic, a turn appears in exactly one partition and the
  FIT/VALIDATION/TEST assignment is deterministic in turn ids and families
  alone. Budget selections are computed over the FIT partition only, as
  prefixes of one fixed round-robin sequence, so the M=512 selection is a
  prefix of M=1024 and of M=2048, with a per-turn contribution cap and honest
  (never padded) row counts.
- The manifest carries bounded scalars, ids and digests only: no raw tensor
  content in any JSON. Tensor files are plain torch.save archives whose
  sha256, content digests, partition digests, split digest and corpus digests
  are all verified at load; a tampered corpus is refused. The module never
  touches a network and never resolves a path outside the caller-provided
  root.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .calibration_corpus import (M_ROW_BUDGETS, MIN_TURN_ROWS,
                                 PARTITION_DIGEST_DOMAIN, SPLIT_SCHEMA, TurnBlock,
                                 _byte_order, _canonical, _identifier, _matrix_digest,
                                 split_manifest, turn_order_key)
from .latent_types import HEX, LatentError, bounded_payload, require
from .statebridge_calibrated import (MAX_CORPUS_TURNS, PARTITION_FIELDS,
                                     PREDICTOR_POSITION_OFFSET, corpus_digest)

CORPUS_SCHEMA = "optimus.occ3.calibration-collector-corpus.v1"
PRECISION = "float32"
MANIFEST_NAME = "manifest.json"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_REFUSALS = 64
MAX_PROVENANCE_FIELDS = 32
MAX_PROVENANCE_VALUE = 256
DEFAULT_PER_TURN_ROW_CAP = 256
DEFAULT_MAX_TURNS = MAX_CORPUS_TURNS
DEFAULT_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_BUDGET_PER_TURN_CAP = 128
DEFAULT_FIT_TURNS = 32
DEFAULT_VALIDATION_TURNS = 8
DEFAULT_TEST_TURNS = 16
PARTITION_NAMES = ("fit", "validation", "test")
BUDGET_KEYS = frozenset({"rows", "turns", "rows_per_turn", "digest"})
TURN_ENTRY_KEYS = frozenset({"turn_id", "family_id", "rows", "partition",
                             "span_policy", "span_metadata"})
MANIFEST_KEYS = frozenset({"schema", "byte_order", "precision",
                           "predictor_position_offset", "provenance", "split",
                           "turns", "partitions", "corpus_manifest_digest"})
PARTITION_ENTRY_KEYS = frozenset({"turn_ids", "row_counts", "rows", "hidden_dim",
                                  "manifest_digest", "corpus_digest", "hidden",
                                  "embeddings", "token_ids", "absolute_positions"})


def _torch():
    import torch
    return torch


def _digest(value):
    return type(value) is str and HEX.fullmatch(value) is not None


def _uint(value, minimum, maximum):
    return (type(value) is int and not isinstance(value, bool)
            and minimum <= value <= maximum)


def _int_vector_digest(vector, torch, tag):
    """Content digest of one int64 vector image; mirrors _matrix_digest."""
    require(getattr(vector, "ndim", 0) == 1, "profile")
    flat = vector.detach().to(dtype=torch.int64, device="cpu", copy=True).contiguous()
    try:
        import numpy
    except ImportError:
        import array
        raw = array.array("q", flat.tolist()).tobytes()
    else:
        raw = bytes(memoryview(numpy.ascontiguousarray(flat.numpy())))
    require(len(raw) == int(flat.numel()) * 8, "numerical")
    value = hashlib.sha256()
    value.update(f"{tag}:{int(vector.shape[0])}:int64:".encode("ascii"))
    value.update(raw)
    return value.hexdigest()


def _file_sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _validate_provenance(value):
    """Caller-supplied identity strings; nothing is derived from the corpus."""
    require(value is None or isinstance(value, Mapping), "profile")
    if value is None:
        return {}
    result = {}
    for key, entry in value.items():
        require(type(key) is str and _identifier(key), "profile")
        require(type(entry) is str, "profile")
        require(1 <= len(entry) <= MAX_PROVENANCE_VALUE, "profile")
        require(entry.isascii() and entry.isprintable(), "profile")
        result[key] = entry
    require(len(result) <= MAX_PROVENANCE_FIELDS, "profile")
    return result


def _validate_budgets(budgets):
    values = tuple(budgets)
    require(bool(values), "profile")
    for value in values:
        require(_uint(value, 1, 2 ** 31) and value in M_ROW_BUDGETS, "profile")
    require(all(values[index] < values[index + 1]
                 for index in range(len(values) - 1)), "profile")
    return values


def _partition_digest(name, turn_ids, row_counts, hidden, embeddings, torch):
    """The partition digest body of calibration_corpus._partition over pooled rows.

    The mirror is pinned by the round-trip test: a loaded partition must carry
    exactly the digest split_manifest produced, so any drift here fails loud.
    """
    row_turn_ids = [turn_id for turn_id, count in zip(turn_ids, row_counts)
                    for _ in range(count)]
    body = {"domain": PARTITION_DIGEST_DOMAIN, "partition": name,
            "rows": int(hidden.shape[0]), "hidden_dim": int(hidden.shape[1]),
            "byte_order": _byte_order(), "turn_ids": list(turn_ids),
            "row_turn_ids": row_turn_ids,
            "hidden": _matrix_digest(hidden, torch, "hidden"),
            "embeddings": _matrix_digest(embeddings, torch, "embeddings")}
    return hashlib.sha256(_canonical(body)).hexdigest()


def _split_digest(targets, turns, partition_digests):
    """The corpus-level split digest body of calibration_corpus.split_manifest."""
    body = {"schema": SPLIT_SCHEMA, "targets": dict(targets), "turns": dict(turns),
            "partitions": {name: partition_digests[name]
                            for name in PARTITION_NAMES}}
    return hashlib.sha256(_canonical(body)).hexdigest()


@dataclass(frozen=True)
class CaptureRecord:
    """One accepted source turn, already reduced to its selected span rows.

    hidden holds the SELECTED semantic-span rows in position order; token_ids
    holds, per row, the accepted token that row predicts (the state at
    absolute_positions[i] predicts token_ids[i]); absolute_positions are the
    absolute accepted indices of the rows, strictly increasing and possibly
    non-contiguous. span_policy is the versioned capture policy and
    span_metadata a bounded receipt-safe dict. A turn with fewer than
    MIN_TURN_ROWS rows is refused: it is not a unit the whole-turn split can
    keep.
    """
    turn_id: str
    family_id: str
    hidden: object
    token_ids: object
    absolute_positions: object
    span_policy: str
    span_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        torch = _torch()
        require(_identifier(self.turn_id), "profile")
        require(_identifier(self.family_id), "profile")
        require(_identifier(self.span_policy), "profile")
        require(type(self.span_metadata) is dict, "profile")
        bounded_payload(dict(self.span_metadata))
        hidden = self.hidden
        require(hasattr(hidden, "dtype") and getattr(hidden, "ndim", None) == 2,
                "profile")
        require(hidden.dtype == torch.float32, "profile")
        rows, dim = (int(value) for value in hidden.shape)
        require(rows >= MIN_TURN_ROWS and dim > 2, "profile")
        require(bool(hidden.isfinite().all()), "numerical")
        for name in ("token_ids", "absolute_positions"):
            vector = getattr(self, name)
            require(hasattr(vector, "dtype") and getattr(vector, "ndim", None) == 1,
                    "profile")
            require(not vector.dtype.is_floating_point
                    and not vector.dtype.is_complex
                    and vector.dtype is not torch.bool, "profile")
            require(int(vector.shape[0]) == rows, "profile")
            require(bool((vector >= 0).all()), "profile")
        positions = self.absolute_positions
        require(rows < 2 or bool((positions[1:] > positions[:-1]).all()), "profile")


def _embedding_rows(token_ids, embedding_lookup, dim, torch, *, on_refusal=None):
    """E rows for the record token ids, aligned with the hidden rows.

    The lookup decides which ids exist. A row it refuses (None or a raised
    error), a wrong-width row and a nonfinite row are all refused here:
    nothing is fabricated, zero-filled or skipped. on_refusal, when given,
    records (token id, reason) pairs for a bounded refusal list.
    """
    require(callable(embedding_lookup), "profile")
    collected = []
    for value in token_ids.tolist():
        token_id = int(value)
        try:
            row = embedding_lookup(token_id)
        except Exception as failure:
            if on_refusal is not None:
                on_refusal(token_id, "lookup")
            raise LatentError("identity") from failure
        if row is None:
            if on_refusal is not None:
                on_refusal(token_id, "unknown")
            raise LatentError("identity")
        if hasattr(row, "dtype"):
            vector = row.detach().to(dtype=torch.float32, device="cpu")
        else:
            try:
                vector = torch.tensor(list(row), dtype=torch.float32)
            except Exception as failure:
                if on_refusal is not None:
                    on_refusal(token_id, "shape")
                raise LatentError("profile") from failure
        if vector.ndim != 1 or int(vector.shape[0]) != dim:
            if on_refusal is not None:
                on_refusal(token_id, "shape")
            raise LatentError("profile")
        if not bool(vector.isfinite().all()):
            if on_refusal is not None:
                on_refusal(token_id, "nonfinite")
            raise LatentError("numerical")
        collected.append(vector)
    require(len(collected) == int(token_ids.shape[0]), "profile")
    return torch.stack(collected, 0).contiguous()


class Collector:
    """Bounded store of whole turns that freezes into a verified corpus.

    Bounds are caller-configurable with sane defaults: a per-turn row cap, a
    total-turn cap and a total byte cap over what write() can persist. The
    byte cap is charged at add for the hidden rows plus the reference rows
    they will resolve to, so a resolution can never push the corpus over it.
    """

    def __init__(self, *, per_turn_row_cap=DEFAULT_PER_TURN_ROW_CAP,
                 max_turns=DEFAULT_MAX_TURNS,
                 max_total_bytes=DEFAULT_MAX_TOTAL_BYTES, provenance=None):
        require(_uint(per_turn_row_cap, MIN_TURN_ROWS, 2 ** 31), "profile")
        require(_uint(max_turns, 1, MAX_CORPUS_TURNS), "profile")
        require(_uint(max_total_bytes, 1, 2 ** 63), "profile")
        self._per_turn_row_cap = int(per_turn_row_cap)
        self._max_turns = int(max_turns)
        self._max_total_bytes = int(max_total_bytes)
        self._provenance = _validate_provenance(provenance)
        self._records = {}
        self._bytes = 0
        self._refusals = []

    def add(self, record, *, embedding_lookup=None):
        """Add one whole turn; returns its accepted row count.

        When embedding_lookup is given the reference rows are resolved now,
        and a lookup refusal means the turn is not added at all.
        """
        require(isinstance(record, CaptureRecord), "profile")
        require(record.turn_id not in self._records, "identity")
        rows, dim = (int(value) for value in record.hidden.shape)
        require(rows <= self._per_turn_row_cap, "profile")
        require(len(self._records) < self._max_turns, "quota")
        for entry in self._records.values():
            require(int(entry["hidden"].shape[1]) == dim, "profile")
            break
        cost = rows * dim * 4 * 2 + rows * 8 * 2
        require(self._bytes + cost <= self._max_total_bytes, "quota")
        embeddings = None
        if embedding_lookup is not None:
            embeddings = _embedding_rows(
                record.token_ids, embedding_lookup, dim, _torch(),
                on_refusal=lambda token_id, reason:
                    self._record_refusal(record.turn_id, token_id, reason))
        torch = _torch()
        self._records[record.turn_id] = {
            "family_id": record.family_id,
            "span_policy": record.span_policy,
            "span_metadata": json.loads(_canonical(dict(record.span_metadata))),
            "rows": rows,
            "hidden": record.hidden.detach().to(
                dtype=torch.float32, device="cpu", copy=True).contiguous(),
            "token_ids": record.token_ids.detach().to(
                dtype=torch.int64, device="cpu", copy=True).contiguous(),
            "absolute_positions": record.absolute_positions.detach().to(
                dtype=torch.int64, device="cpu", copy=True).contiguous(),
            "embeddings": embeddings,
        }
        self._bytes += cost
        return rows

    def resolve_embeddings(self, record, embedding_lookup):
        """Resolve and store the E rows of one already-added turn.

        Unknown token ids, wrong-width rows and nonfinite rows are refused
        and recorded; a failed resolution leaves the turn unresolved so the
        caller can retry. A turn is resolved once; re-resolving is refused.
        """
        require(isinstance(record, CaptureRecord), "profile")
        require(record.turn_id in self._records, "identity")
        stored = self._records[record.turn_id]
        require(stored["embeddings"] is None, "identity")
        dim = int(stored["hidden"].shape[1])
        stored["embeddings"] = _embedding_rows(
            record.token_ids, embedding_lookup, dim, _torch(),
            on_refusal=lambda token_id, reason:
                self._record_refusal(record.turn_id, token_id, reason))
        return stored["embeddings"].clone()

    def _record_refusal(self, turn_id, token_id, reason):
        self._refusals.append({"turn_id": turn_id, "token_id": int(token_id),
                                "reason": reason})
        del self._refusals[:-MAX_REFUSALS]

    def refusals(self):
        """The bounded, receipt-safe refusal records of this collector."""
        return tuple(dict(entry) for entry in self._refusals)

    def turn_ids(self):
        """All added turn ids in the deterministic digest order."""
        return tuple(sorted(self._records, key=turn_order_key))

    @property
    def turn_count(self):
        return len(self._records)

    @property
    def row_count(self):
        return sum(entry["rows"] for entry in self._records.values())

    def write(self, dirpath, *, include_embeddings=True,
              fit_turns=DEFAULT_FIT_TURNS,
              validation_turns=DEFAULT_VALIDATION_TURNS,
              test_turns=DEFAULT_TEST_TURNS, budgets=M_ROW_BUDGETS,
              budget_per_turn_cap=DEFAULT_BUDGET_PER_TURN_CAP, torch_module=None):
        """Freeze the corpus under a caller-provided root.

        Partitions come from calibration_corpus.split_manifest over whole
        family-grouped turns. Tensor files are written first and manifest.json
        last, so an interrupted write is never mistaken for a corpus. Returns
        a scalar-only summary whose corpus_manifest_digest pins the manifest.
        """
        torch = torch_module or _torch()
        require(type(include_embeddings) is bool, "profile")
        budget_values = _validate_budgets(budgets)
        require(_uint(budget_per_turn_cap, 1, 2 ** 31), "profile")
        require(self._records, "profile")
        for entry in self._records.values():
            require(entry["embeddings"] is not None, "profile")
        blocks = [TurnBlock(turn_id=turn_id, hidden=entry["hidden"],
                            embeddings=entry["embeddings"],
                            family=entry["family_id"])
                  for turn_id, entry in sorted(self._records.items(),
                                               key=lambda item:
                                                   turn_order_key(item[0]))]
        split = split_manifest(blocks, fit_turns=fit_turns,
                               validation_turns=validation_turns,
                               test_turns=test_turns, torch_module=torch)
        root = Path(dirpath)
        root.mkdir(parents=True, exist_ok=True)
        files = []
        partitions = {}
        for name in PARTITION_NAMES:
            partitions[name] = self._write_partition(
                root, name, split[name], include_embeddings, budget_values,
                budget_per_turn_cap, torch, files)
        assignment = {}
        for name in PARTITION_NAMES:
            for turn_id in partitions[name]["turn_ids"]:
                assignment[turn_id] = name
        turns = []
        for turn_id in sorted(self._records, key=turn_order_key):
            entry = self._records[turn_id]
            turns.append({"turn_id": turn_id, "family_id": entry["family_id"],
                          "rows": entry["rows"], "partition": assignment[turn_id],
                          "span_policy": entry["span_policy"],
                          "span_metadata": entry["span_metadata"]})
        manifest = {"schema": CORPUS_SCHEMA, "byte_order": _byte_order(),
                    "precision": PRECISION,
                    "predictor_position_offset": PREDICTOR_POSITION_OFFSET,
                    "provenance": dict(self._provenance),
                    "split": {"schema": SPLIT_SCHEMA,
                              "targets": dict(split["targets"]),
                              "turns": dict(split["turns"]),
                              "manifest_digest": split["manifest_digest"]},
                    "turns": turns, "partitions": partitions}
        manifest["corpus_manifest_digest"] = hashlib.sha256(
            _canonical(manifest)).hexdigest()
        (root / MANIFEST_NAME).write_bytes(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("ascii"))
        files.append(MANIFEST_NAME)
        return {"schema": CORPUS_SCHEMA, "dirpath": str(dirpath),
                "corpus_manifest_digest": manifest["corpus_manifest_digest"],
                "embeddings_included": include_embeddings,
                "turns": dict(split["turns"]),
                "rows": {name: partitions[name]["rows"]
                         for name in PARTITION_NAMES},
                "budgets": {key: {"rows": value["rows"], "turns": value["turns"]}
                            for key, value in
                            partitions["fit"]["budget_selections"].items()},
                "files": sorted(files)}

    def _write_partition(self, root, name, partition, include_embeddings,
                         budget_values, budget_per_turn_cap, torch, files):
        stored = self._records
        turn_ids = list(partition["turn_ids"])
        hidden = partition["hidden"].contiguous()
        embeddings = partition["embeddings"].contiguous()
        token_ids = torch.cat([stored[turn_id]["token_ids"]
                               for turn_id in turn_ids], 0).contiguous()
        positions = torch.cat([stored[turn_id]["absolute_positions"]
                               for turn_id in turn_ids], 0).contiguous()
        row_counts = [stored[turn_id]["rows"] for turn_id in turn_ids]
        entry = {"turn_ids": turn_ids, "row_counts": row_counts,
                 "rows": int(hidden.shape[0]),
                 "hidden_dim": int(hidden.shape[1]),
                 "manifest_digest": partition["manifest_digest"],
                 "corpus_digest": corpus_digest(hidden, embeddings,
                                                turn_ids=tuple(turn_ids),
                                                torch_module=torch)}
        entry["hidden"] = _save(root, f"{name}_hidden.pt", hidden, torch, files)
        entry["hidden"]["rows"] = int(hidden.shape[0])
        entry["hidden"]["hidden_dim"] = int(hidden.shape[1])
        entry["hidden"]["content_digest"] = _matrix_digest(hidden, torch, "hidden")
        if include_embeddings:
            entry["embeddings"] = _save(root, f"{name}_embeddings.pt", embeddings,
                                        torch, files)
            entry["embeddings"]["content_digest"] = _matrix_digest(
                embeddings, torch, "embeddings")
        else:
            entry["embeddings"] = {
                "included": False,
                "content_digest": _matrix_digest(embeddings, torch, "embeddings")}
        for key, tensor in (("token_ids", token_ids),
                            ("absolute_positions", positions)):
            entry[key] = _save(root, f"{name}_{key}.pt", tensor, torch, files)
            entry[key]["content_digest"] = _int_vector_digest(tensor, torch, key)
        if name == "fit":
            entry["budget_selections"] = budget_selections(
                partition, budgets=budget_values, per_turn_cap=budget_per_turn_cap,
                torch_module=torch)
        return entry


def _save(root, name, tensor, torch, files):
    path = root / name
    torch.save(tensor, path)
    files.append(name)
    return {"file": name, "sha256": _file_sha256(path)}


def budget_selections(fit_partition, *, budgets=M_ROW_BUDGETS,
                      per_turn_cap=DEFAULT_BUDGET_PER_TURN_CAP, torch_module=None):
    """Nested deterministic row selections over the FIT partition only.

    Each turn contributes an evenly spread sample of at most per_turn_cap of
    its rows, so a long reasoning trace cannot dominate. One fixed sequence
    interleaves turns in digest order one row per round, and a budget M takes
    the first M pairs of that sequence, so every smaller budget is a prefix of
    every larger one in the same turn alignment. A corpus smaller than the
    budget is never padded: the actual rows and unique turns are reported.
    """
    torch = torch_module or _torch()
    require(isinstance(fit_partition, Mapping)
            and set(fit_partition) == set(PARTITION_FIELDS), "profile")
    budget_values = _validate_budgets(budgets)
    require(_uint(per_turn_cap, 1, 2 ** 31), "profile")
    turn_ids = tuple(fit_partition["turn_ids"])
    row_turn_ids = tuple(fit_partition["row_turn_ids"])
    hidden = fit_partition["hidden"]
    require(getattr(hidden, "ndim", None) == 2, "profile")
    rows = int(hidden.shape[0])
    require(len(row_turn_ids) == rows, "profile")
    require(all(_identifier(value) for value in turn_ids), "profile")
    require(len(set(turn_ids)) == len(turn_ids)
            and set(turn_ids) == set(row_turn_ids), "identity")
    require(list(turn_ids) == sorted(turn_ids, key=turn_order_key), "identity")
    counts, offsets = {}, {}
    for index, turn_id in enumerate(row_turn_ids):
        if turn_id not in offsets:
            offsets[turn_id] = index
        counts[turn_id] = counts.get(turn_id, 0) + 1
    for turn_id in turn_ids:
        start, count = offsets[turn_id], counts[turn_id]
        require(all(value == turn_id
                     for value in row_turn_ids[start:start + count]), "profile")
    eligible = {}
    for turn_id in turn_ids:
        count = counts[turn_id]
        size = min(count, per_turn_cap)
        eligible[turn_id] = [offsets[turn_id] + (index * count) // size
                             for index in range(size)]
    sequence = []
    for round_index in range(max(len(indices) for indices in eligible.values())):
        for turn_id in turn_ids:
            indices = eligible[turn_id]
            if round_index < len(indices):
                sequence.append((turn_id, indices[round_index]))
    selections = {}
    for budget in budget_values:
        pairs = sequence[:budget]
        rows_per_turn = {}
        for turn_id, index in pairs:
            rows_per_turn.setdefault(turn_id, []).append(index)
        selections[str(budget)] = {
            "rows": len(pairs), "turns": len(rows_per_turn),
            "rows_per_turn": rows_per_turn,
            "digest": hashlib.sha256(_canonical(
                {"budget": budget, "rows": len(pairs),
                 "rows_per_turn": rows_per_turn})).hexdigest()}
    return selections


def read_manifest(dirpath, *, torch_module=None):
    """The fully verified manifest of a written corpus; scalar-safe only."""
    del torch_module
    path = Path(dirpath) / MANIFEST_NAME
    require(path.is_file(), "profile")
    raw = path.read_bytes()
    require(len(raw) <= MAX_MANIFEST_BYTES, "quota")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except ValueError as failure:
        raise LatentError("profile") from failure
    _validate_manifest(manifest)
    body = {key: value for key, value in manifest.items()
            if key != "corpus_manifest_digest"}
    require(hashlib.sha256(_canonical(body)).hexdigest()
            == manifest["corpus_manifest_digest"], "identity")
    return manifest


def load(dirpath, *, embedding_lookup=None, torch_module=None):
    """Read a corpus back as the verified split-manifest partitions.

    Returns the shape calibration_corpus.split_manifest produces; each
    partition holds exactly the contract keys fit_map and evaluate_map
    consume. Every digest in the chain is verified against the loaded bytes:
    the manifest self digest, each file sha256, each tensor content digest,
    each partition digest, the split digest and each corpus digest. A
    corpus written without embedding files rebuilds its reference rows
    through the runtime lookup and must match the recorded digests.
    """
    torch = torch_module or _torch()
    manifest = read_manifest(dirpath)
    root = Path(dirpath)
    partitions, digests = {}, {}
    for name in PARTITION_NAMES:
        entry = manifest["partitions"][name]
        rows, dim = entry["rows"], entry["hidden_dim"]
        hidden = _load_tensor(root / entry["hidden"]["file"],
                              entry["hidden"]["sha256"], torch)
        require(hidden.ndim == 2 and int(hidden.shape[0]) == rows
                and int(hidden.shape[1]) == dim, "profile")
        require(hidden.dtype == torch.float32, "profile")
        require(_matrix_digest(hidden, torch, "hidden")
                == entry["hidden"]["content_digest"], "identity")
        token_ids = _load_tensor(root / entry["token_ids"]["file"],
                                 entry["token_ids"]["sha256"], torch)
        positions = _load_tensor(
            root / entry["absolute_positions"]["file"],
            entry["absolute_positions"]["sha256"], torch)
        for vector in (token_ids, positions):
            require(vector.ndim == 1 and int(vector.shape[0]) == rows, "profile")
            require(vector.dtype == torch.int64, "profile")
        require(_int_vector_digest(token_ids, torch, "token_ids")
                == entry["token_ids"]["content_digest"], "identity")
        require(_int_vector_digest(positions, torch, "absolute_positions")
                == entry["absolute_positions"]["content_digest"], "identity")
        embeddings_block = entry["embeddings"]
        if "file" in embeddings_block:
            embeddings = _load_tensor(root / embeddings_block["file"],
                                      embeddings_block["sha256"], torch)
        else:
            require(callable(embedding_lookup), "profile")
            embeddings = _embedding_rows(token_ids, embedding_lookup, dim, torch)
        require(embeddings.ndim == 2 and tuple(int(x) for x in embeddings.shape)
                == (rows, dim), "profile")
        require(embeddings.dtype == torch.float32, "profile")
        require(_matrix_digest(embeddings, torch, "embeddings")
                == embeddings_block["content_digest"], "identity")
        turn_ids = tuple(entry["turn_ids"])
        row_turn_ids = tuple(turn_id for turn_id, count
                            in zip(turn_ids, entry["row_counts"])
                            for _ in range(count))
        require(len(row_turn_ids) == rows, "profile")
        digest = _partition_digest(name, turn_ids, entry["row_counts"], hidden,
                                   embeddings, torch)
        require(digest == entry["manifest_digest"], "identity")
        require(corpus_digest(hidden, embeddings, turn_ids=turn_ids,
                              torch_module=torch) == entry["corpus_digest"],
                "identity")
        partitions[name] = {"hidden": hidden, "embeddings": embeddings,
                            "row_turn_ids": row_turn_ids, "turn_ids": turn_ids,
                            "manifest_digest": entry["manifest_digest"]}
        digests[name] = digest
    split = manifest["split"]
    require(_split_digest(split["targets"], split["turns"], digests)
            == split["manifest_digest"], "identity")
    return {"schema": SPLIT_SCHEMA, "fit": partitions["fit"],
            "validation": partitions["validation"], "test": partitions["test"],
            "turns": dict(split["turns"]), "targets": dict(split["targets"]),
            "manifest_digest": split["manifest_digest"]}


def _load_tensor(path, expected_sha256, torch):
    """One plain tensor, with the file digest verified before any load."""
    require(path.is_file() and _digest(expected_sha256), "profile")
    require(_file_sha256(path) == expected_sha256, "identity")
    try:
        value = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as failure:
        raise LatentError("profile") from failure
    require(hasattr(value, "dtype"), "profile")
    return value


def _validate_manifest(manifest):
    """Exact manifest structure; a tampered layout is refused before payloads."""
    require(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS,
            "profile")
    require(manifest["schema"] == CORPUS_SCHEMA, "identity")
    require(manifest["byte_order"] == _byte_order()
            and manifest["precision"] == PRECISION, "profile")
    require(manifest["predictor_position_offset"] == PREDICTOR_POSITION_OFFSET,
            "identity")
    require(_digest(manifest["corpus_manifest_digest"]), "identity")
    _validate_provenance(manifest["provenance"])
    split = manifest["split"]
    require(isinstance(split, dict) and set(split)
            == {"schema", "targets", "turns", "manifest_digest"}, "profile")
    require(split["schema"] == SPLIT_SCHEMA, "identity")
    require(_digest(split["manifest_digest"]), "identity")
    for name in ("targets", "turns"):
        block = split[name]
        require(isinstance(block, dict) and set(block) == set(PARTITION_NAMES),
                "profile")
        require(all(_uint(value, 1, MAX_CORPUS_TURNS)
                    for value in block.values()), "profile")
    turns = manifest["turns"]
    require(isinstance(turns, list) and bool(turns), "profile")
    require(len(turns) <= MAX_CORPUS_TURNS, "quota")
    seen, families = {}, {}
    for entry in turns:
        require(isinstance(entry, dict) and set(entry) == TURN_ENTRY_KEYS,
                "profile")
        require(_identifier(entry["turn_id"]) and _identifier(entry["family_id"]),
                "profile")
        require(_identifier(entry["span_policy"]), "profile")
        require(_uint(entry["rows"], MIN_TURN_ROWS, 2 ** 31), "profile")
        require(entry["partition"] in PARTITION_NAMES, "profile")
        require(type(entry["span_metadata"]) is dict, "profile")
        bounded_payload(entry["span_metadata"])
        require(entry["turn_id"] not in seen, "identity")
        seen[entry["turn_id"]] = entry
        families.setdefault(entry["family_id"], set()).add(entry["partition"])
    for members in families.values():
        require(len(members) == 1, "identity")
    keys = [entry["turn_id"] for entry in turns]
    require(keys == sorted(keys, key=turn_order_key), "identity")
    partitions = manifest["partitions"]
    require(isinstance(partitions, dict)
            and set(partitions) == set(PARTITION_NAMES), "profile")
    assigned = {}
    for name in PARTITION_NAMES:
        entry = partitions[name]
        _validate_partition_entry(name, entry)
        for turn_id in entry["turn_ids"]:
            require(turn_id not in assigned, "identity")
            assigned[turn_id] = name
    require(set(assigned) == set(seen), "identity")
    for turn_id, entry in seen.items():
        require(assigned[turn_id] == entry["partition"], "identity")
        order = partitions[entry["partition"]]["turn_ids"].index(turn_id)
        require(entry["rows"]
                == partitions[entry["partition"]]["row_counts"][order], "identity")
    require({name: len(partitions[name]["turn_ids"]) for name in PARTITION_NAMES}
            == split["turns"], "identity")


def _validate_partition_entry(name, entry):
    required = set(PARTITION_ENTRY_KEYS)
    if name == "fit":
        required.add("budget_selections")
    require(isinstance(entry, dict) and set(entry) == required, "profile")
    turn_ids = entry["turn_ids"]
    require(isinstance(turn_ids, list) and bool(turn_ids), "profile")
    require(all(_identifier(value) for value in turn_ids)
            and len(set(turn_ids)) == len(turn_ids), "identity")
    row_counts = entry["row_counts"]
    require(isinstance(row_counts, list) and len(row_counts) == len(turn_ids),
            "profile")
    require(all(_uint(value, MIN_TURN_ROWS, 2 ** 31) for value in row_counts),
            "profile")
    require(_uint(entry["rows"], MIN_TURN_ROWS, 2 ** 31)
            and entry["rows"] == sum(row_counts), "profile")
    require(_uint(entry["hidden_dim"], 3, 2 ** 20), "profile")
    require(_digest(entry["manifest_digest"]) and _digest(entry["corpus_digest"]),
            "identity")
    hidden = entry["hidden"]
    require(isinstance(hidden, dict) and set(hidden)
            == {"file", "sha256", "rows", "hidden_dim", "content_digest"},
            "profile")
    require(hidden["file"] == f"{name}_hidden.pt", "identity")
    require(_digest(hidden["sha256"]) and _digest(hidden["content_digest"]),
            "identity")
    require(hidden["rows"] == entry["rows"]
            and hidden["hidden_dim"] == entry["hidden_dim"], "identity")
    embeddings = entry["embeddings"]
    if isinstance(embeddings, dict) and "file" in embeddings:
        require(set(embeddings) == {"file", "sha256", "content_digest"}, "profile")
        require(embeddings["file"] == f"{name}_embeddings.pt", "identity")
        require(_digest(embeddings["sha256"])
                and _digest(embeddings["content_digest"]), "identity")
    else:
        require(set(embeddings) == {"included", "content_digest"}, "profile")
        require(embeddings["included"] is False, "identity")
        require(_digest(embeddings["content_digest"]), "identity")
    for key in ("token_ids", "absolute_positions"):
        block = entry[key]
        require(isinstance(block, dict)
                and set(block) == {"file", "sha256", "content_digest"}, "profile")
        require(block["file"] == f"{name}_{key}.pt", "identity")
        require(_digest(block["sha256"]) and _digest(block["content_digest"]),
                "identity")
    if name == "fit":
        _validate_budget_selections(entry["budget_selections"], turn_ids,
                                    row_counts)


def _validate_budget_selections(selections, turn_ids, row_counts):
    """FIT-only, in-range, nested and digest-bound row selections."""
    require(isinstance(selections, dict) and bool(selections), "profile")
    keys = list(selections)
    require(all(type(key) is str and key.isdigit() and int(key) in M_ROW_BUDGETS
                for key in keys), "profile")
    require(len(set(keys)) == len(keys), "identity")
    keys = sorted(keys, key=int)
    starts, limits = {}, dict(zip(turn_ids, row_counts))
    position = 0
    for turn_id, count in zip(turn_ids, row_counts):
        starts[turn_id] = position
        position += count
    previous = None
    for key in keys:
        value = selections[key]
        require(isinstance(value, dict) and set(value) == BUDGET_KEYS, "profile")
        rows_per_turn = value["rows_per_turn"]
        require(isinstance(rows_per_turn, dict), "profile")
        require(set(rows_per_turn) <= set(turn_ids), "identity")
        total = 0
        for turn_id, indices in rows_per_turn.items():
            require(isinstance(indices, list), "profile")
            start, limit = starts[turn_id], limits[turn_id]
            require(all(_uint(index, start, start + limit - 1)
                        for index in indices), "identity")
            require(all(indices[index] < indices[index + 1]
                         for index in range(len(indices) - 1)), "identity")
            total += len(indices)
        require(_uint(value["rows"], 1, 2 ** 31) and value["rows"] == total,
                "identity")
        require(_uint(value["turns"], 1, len(turn_ids))
                and value["turns"] == len(rows_per_turn), "identity")
        require(_digest(value["digest"])
                and value["digest"] == hashlib.sha256(_canonical(
                    {"budget": int(key), "rows": value["rows"],
                     "rows_per_turn": rows_per_turn})).hexdigest(), "identity")
        if previous is not None:
            for turn_id, indices in previous.items():
                larger = value["rows_per_turn"].get(turn_id, [])
                require(indices == larger[:len(indices)], "identity")
        previous = rows_per_turn
