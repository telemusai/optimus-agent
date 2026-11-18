#!/usr/bin/env python3
"""Offline calibration-corpus pooling: legacy partitions + supplemental turns.

The supplemental collection (host-backed capture under
``/data/neo-glm53/occ3-calibration-capture/<window>``) lands in its own
corpus directory with its own whole-family 8/8/8 split. The P5 refit runs on
the POOLED corpus, so this tool merges corpora offline:

  pool(legacy_dir, [supplemental_dir, ...], out_dir)

Rules (fixed by the accepted design; nothing here re-derives them):

- The FIRST corpus is the legacy corpus: every one of its turns keeps the
  partition its own manifest assigned it. Each supplemental corpus contributes
  its turns to the SAME partition that corpus assigned them, so every pooled
  partition gains the supplemental families the supplemental split placed
  there (validation sees the retained turns).
- Whole-family invariant: a family must map to exactly one partition across
  the whole pool, exactly as a freshly split corpus guarantees.
- A turn id may appear in more than one corpus only when the records are
  BITWISE IDENTICAL (same family, span policy, canonical span metadata, row
  count and every tensor row); identical duplicates collapse to one turn,
  any divergence is refused. Reference embedding rows merge the same way:
  one token id must resolve to the same float32 row everywhere it is known,
  else the pool is refused.
- Every input is verified through calibration_collector's own digest chain
  (read_manifest + load + load_turns), and every pooled turn re-enters the
  pipeline through CaptureRecord + Collector.add, so the pooled corpus can
  never be looser than a freshly collected one (the retained v3 policy's
  rows == span_content_count alignment rule included).
- The pooled corpus is written by Collector.write_partitions with
  embeddings included, so the P5 runner consumes it exactly like a
  collector-written corpus.

The output summary is scalar-only (digests, counts, ids); no tensor content
is ever printed. This module never touches a network.

Usage (DGX-local CPU, or anywhere with the source and torch):

  python3 calibration_pool.py [--source SRC] --legacy DIR \
      --supplemental DIR [--supplemental DIR ...] --out DIR [--json-out FILE]

``--source`` is the directory holding the ``occ3_sglang`` package (the
attested install inside the serving image, or the staged source root); it
defaults to the environment of the running interpreter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

POOL_SCHEMA = "optimus.occ3.calibration-pool.v1"
PARTITION_NAMES = ("fit", "validation", "test")
#: Provenance fields that must agree across every pooled corpus.
#: method_id is deliberately absent: a supplemental corpus collected under
#: a different latent method is the distribution shift pooling exists to
#: cover, so its id is unioned into POOLED_METHOD_IDS instead of refusing.
SHARED_PROVENANCE_KEYS = ("hidden_dim", "layer_index",
                           "marker_profile_max_len")
#: Manifest field recording every pooled input's method_id, sorted and
#: comma-joined, present only when the ids actually differ (byte-identity
#: for agreeing corpora; the value cap is the manifest provenance quota).
POOLED_METHOD_IDS = "pooled_method_ids"
#: Provenance fields allowed to differ (the artifact a corpus was collected
#: under); distinct values are recorded, comma-joined in sorted order.
MERGED_PROVENANCE_KEYS = ("artifact_digest",)
TENSOR_KEYS = ("hidden", "token_ids", "absolute_positions", "embeddings")


class PoolError(RuntimeError):
    """One fixed refusal code; nothing is written after a refusal."""

    def __init__(self, code, detail=""):
        self.code, self.detail = code, detail
        super().__init__(code if not detail else code + ": " + detail)


def _require(condition, code, detail=""):
    if not condition:
        raise PoolError(code, detail)


def _occ3(source=None):
    """Import the landed occ3_sglang package, optionally from --source."""
    if source:
        source = os.path.abspath(source)
        if source not in sys.path:
            sys.path.insert(0, source)
    import torch  # noqa: F401  (the landed modules need it)
    from occ3_sglang import calibration_collector
    return {"torch": torch, "collector": calibration_collector}


def _canonical_metadata(turn, collector):
    return collector._canonical(dict(turn["span_metadata"]))


def _turns_identical(left, right, torch, collector):
    """Bitwise turn equality: scalars, canonical metadata and every tensor."""
    if (left["turn_id"] != right["turn_id"]
            or left["family_id"] != right["family_id"]
            or left["span_policy"] != right["span_policy"]
            or left["rows"] != right["rows"]
            or _canonical_metadata(left, collector)
            != _canonical_metadata(right, collector)):
        return False
    return all(bool(torch.equal(left[key], right[key])) for key in TENSOR_KEYS)


def _absorb(merged, token_id, row, torch, origin):
    """One token id -> row; conflicts refused unless bitwise identical."""
    if hasattr(row, "dtype"):
        vector = row.detach().to(dtype=torch.float32, device="cpu")
    else:
        vector = torch.tensor(list(row), dtype=torch.float32)
    _require(vector.ndim == 1, "profile", "embedding row not 1-D")
    existing = merged.get(int(token_id))
    if existing is None:
        merged[int(token_id)] = vector.clone()
        return
    _require(bool(torch.equal(existing, vector)), "identity",
             "embedding row conflict for token %d (%s)" % (int(token_id), origin))


def _merged_lookup(merged):
    def lookup(token_id):
        return merged.get(int(token_id))
    return lookup


def _merge_provenance(corpora):
    """Shared identity fields must agree; artifact digests merge sorted."""
    merged = {}
    for key in SHARED_PROVENANCE_KEYS:
        values = [corpus["provenance"].get(key) for corpus in corpora]
        values = [value for value in values if value is not None]
        _require(values, "identity", "missing provenance field %s" % key)
        _require(all(value == values[0] for value in values), "identity",
                 "provenance field %s differs across corpora" % key)
        merged[key] = values[0]
    # method_id keeps the FIRST corpus's value (the legacy corpus is first)
    # so any consumer reading it as one string stays fed; when a pooled
    # corpus was collected under a different latent method, the union is
    # recorded instead of refusing - nothing is silently dropped.
    method_ids = [corpus["provenance"].get("method_id") for corpus in corpora]
    method_ids = [value for value in method_ids if value is not None]
    _require(method_ids, "identity", "missing provenance field method_id")
    merged["method_id"] = method_ids[0]
    if len(set(method_ids)) > 1:
        union = ",".join(sorted(set(method_ids)))
        _require(len(union) <= 256, "quota", "too many pooled method ids")
        merged[POOLED_METHOD_IDS] = union
    for key in MERGED_PROVENANCE_KEYS:
        values = sorted({corpus["provenance"][key] for corpus in corpora
                         if corpus["provenance"].get(key) is not None})
        if values:
            merged[key] = ",".join(values)
    pooled_from = ",".join(corpus["corpus_manifest_digest"]
                           for corpus in corpora)
    _require(len(pooled_from) <= 256, "quota", "too many pooled corpora")
    merged["pooled_from"] = pooled_from
    merged["pool_schema"] = POOL_SCHEMA
    return merged


def pool(legacy_dir, supplemental_dirs, out_dir, *, embedding_lookup=None,
         torch_module=None):
    """Pool corpora into out_dir; returns a scalar-only summary.

    ``legacy_dir`` is the first corpus: its partition assignment is
    preserved. Every corpus in ``supplemental_dirs`` contributes its turns
    to the same partitions its own manifest assigned them. See the module
    docstring for the conflict and identity rules.
    """
    env = _occ3()
    torch = torch_module or env["torch"]
    collector = env["collector"]
    dirs = [legacy_dir] + list(supplemental_dirs)
    _require(len(dirs) >= 2, "profile", "pooling needs a legacy and at least "
             "one supplemental corpus")
    manifests = [collector.read_manifest(path) for path in dirs]

    # Phase A: corpora that carry embedding files load first and seed the
    # merged reference rows; a corpus without them loads through the merged
    # lookup, exactly like calibration_collector.load does.
    order = sorted(range(len(dirs)),
                   key=lambda index: 0 if "file" in
                   manifests[index]["partitions"]["fit"]["embeddings"] else 1)
    merged = {}
    corpora = [None] * len(dirs)
    for index in order:
        manifest = manifests[index]
        included = all("file" in manifest["partitions"][name]["embeddings"]
                       for name in PARTITION_NAMES)
        lookup = None if included else _merged_lookup(merged)
        try:
            turns = collector.load_turns(dirs[index],
                                         embedding_lookup=lookup,
                                         torch_module=torch)
        except Exception as failure:  # a corpus that cannot verify is fatal
            raise PoolError("corpus_load_failed",
                            "%s: %s" % (dirs[index], type(failure).__name__)
                            ) from failure
        if included:
            for turn in turns:
                for token_id, row in zip(turn["token_ids"].tolist(),
                                         turn["embeddings"]):
                    _absorb(merged, token_id, row, torch, dirs[index])
        corpora[index] = {"dirpath": str(dirs[index]),
                          "corpus_manifest_digest":
                              manifest["corpus_manifest_digest"],
                          "provenance": manifest["provenance"],
                          "embeddings_included": included, "turns": turns}

    # The caller lookup (the runtime embedding source) merges last.
    if embedding_lookup is not None:
        _require(callable(embedding_lookup), "profile", "embedding_lookup")
        for key in sorted(merged):
            row = embedding_lookup(key)
            if row is not None:
                _absorb(merged, key, row, torch, "caller")
    lookup = _merged_lookup(merged)

    # Phase B: dedupe bitwise-identical turns; refuse any divergence.
    pooled, deduped = [], []
    seen = {}
    for corpus in corpora:
        for turn in corpus["turns"]:
            previous = seen.get(turn["turn_id"])
            if previous is None:
                seen[turn["turn_id"]] = turn
                pooled.append(turn)
            else:
                _require(_turns_identical(previous, turn, torch, collector),
                         "identity", "turn conflict: %s/%s"
                         % (turn["turn_id"], turn["family_id"]))
                deduped.append(turn["turn_id"])

    # Phase C: the assignment - legacy partitions preserved, supplemental
    # turns land in the partitions their own corpus assigned them.
    assignment, families = {}, {}
    for turn in pooled:
        _require(turn["partition"] in PARTITION_NAMES, "profile",
                 "unknown partition for %s/%s" % (turn["turn_id"],
                                                  turn["family_id"]))
        _require(turn["turn_id"] not in assignment, "identity",
                 "turn assigned twice: %s" % turn["turn_id"])
        assignment[turn["turn_id"]] = turn["partition"]
        families.setdefault(turn["family_id"], set()).add(turn["partition"])
    for family, members in families.items():
        _require(len(members) == 1, "identity",
                 "family %s spans partitions %s" % (family, sorted(members)))
    partitions = {name: sorted((turn_id for turn_id, target
                                in assignment.items() if target == name),
                               key=collector.turn_order_key)
                  for name in PARTITION_NAMES}
    for name in PARTITION_NAMES:
        _require(bool(partitions[name]), "profile",
                 "pooled partition %s would be empty" % name)

    # Phase D: re-enter through the collector itself. A turn a corpus file
    # carried but CaptureRecord refuses (a misaligned retained span, an
    # unknown token id) is a hard refusal: the pool is never looser than a
    # fresh collection.
    store = collector.Collector(provenance=_merge_provenance(corpora))
    for turn in sorted(pooled, key=lambda item:
                       collector.turn_order_key(item["turn_id"])):
        try:
            record = collector.CaptureRecord(
                turn_id=turn["turn_id"], family_id=turn["family_id"],
                hidden=turn["hidden"], token_ids=turn["token_ids"],
                absolute_positions=turn["absolute_positions"],
                span_policy=turn["span_policy"],
                span_metadata=dict(turn["span_metadata"]))
            store.add(record, embedding_lookup=lookup)
        except Exception as failure:  # collector refusals stay fail-closed
            code = getattr(failure, "code", None)
            raise PoolError("record_refused",
                            "%s: %s" % (turn["turn_id"], code
                                        if code is not None
                                        else type(failure).__name__)) from failure
    try:
        summary = store.write_partitions(out_dir, partitions)
    except Exception as failure:
        code = getattr(failure, "code", None)
        raise PoolError("write_refused", str(code if code is not None
                                             else type(failure).__name__)
                        ) from failure
    summary["schema"] = POOL_SCHEMA
    summary["pooled_from"] = [corpus["corpus_manifest_digest"]
                              for corpus in corpora]
    summary["deduped_turns"] = sorted(deduped)
    summary["embedding_rows"] = len(merged)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=None,
                        help="directory holding the occ3_sglang package")
    parser.add_argument("--legacy", required=True,
                        help="legacy corpus directory (partitions preserved)")
    parser.add_argument("--supplemental", action="append", required=True,
                        help="supplemental corpus directory (repeatable)")
    parser.add_argument("--out", required=True, help="pooled output directory")
    parser.add_argument("--json-out", default=None,
                        help="optional path for the scalar summary")
    args = parser.parse_args(argv)
    summary = pool(args.legacy, args.supplemental, args.out)
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    print(payload)
    if args.json_out:
        directory = os.path.dirname(os.path.abspath(args.json_out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
