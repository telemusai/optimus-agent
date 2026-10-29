"""Three-way calibration corpora, frozen split manifests and shared evaluation.

The live window measured the hidden-to-embedding relation inside single turns,
because that was all it captured: 51 fit rows of 6144 dimensions. A pooled
corpus of whole turns has strictly more capacity, and the only honest way to
score it is to hold out whole turns, never rows.

Three rules define this module:

1. A turn is the split unit. split_manifest freezes a FIT, VALIDATION and
   FINAL TEST partition of whole turns, grouped by caller-supplied source
   fact/context families so near-duplicate prompts never cross partitions.
   FIT fits; VALIDATION selects hyperparameters and the serving variant;
   TEST is never touched by fitting or selection.
2. Evaluation records, never gates. evaluate_map scores a frozen artifact on
   an independent partition and returns the complete metric record of spec
   section 12: raw and served metrics kept distinct, per-turn, turn-macro and
   pooled views, negatives recorded, undefined denominators recorded as
   undefined instead of raising.
3. Promotion happens only in validate_for_activation, which consumes a policy
   and a validation record bound to the artifact digest. Nothing in this
   module refuses a measurement because its quality is poor.

Turn order is the ascending sha256 of the turn id, never insertion order, so a
row budget truncates the same turns on every run and on every host.

The controls are the campaign's matched arms as a scoring library, not a live
runner: correct state, wrong-sender state, shuffled state, matched-shape null
state, no attachment, and reference-token-only. They are kept mathematically
distinct so a null or shuffled arm cannot be mistaken for the correct arm.
A metric that was not measured is None, never zero.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .latent_types import require
from .statebridge_calibrated import (MIN_CALIBRATION_ROWS, PARTITION_FIELDS,
                                     serving_transform)

M_ROW_BUDGETS = (512, 1024, 2048)
MIN_TURN_ROWS = 2
TURN_ORDER_DOMAIN = "optimus.occ3.calibration-turn-order.v1"
SPLIT_DOMAIN = "optimus.occ3.calibration-split.v2"
SPLIT_SCHEMA = "optimus.occ3.calibration-split.v2"
PARTITION_DIGEST_DOMAIN = "optimus.occ3.calibration-partition.v1"
RECORD_SCHEMA = "optimus.occ3.evaluation-record.v1"
SELECTION_SCHEMA = "optimus.occ3.calibration-selection.v1"
NORM_FLOOR = 1e-6
RHO_GRID = (0.002, 0.02, 0.2)
SERVING_VARIANTS = ("n0", "n1")

VIEW_KEYS = ("r2_standard", "r2_vs_fit_mean", "sse", "sst_test", "sse_vs_fit_mean",
             "relative_frobenius", "mean_row_cosine", "raw_row_norm_mean",
             "served_r2_standard", "served_r2_vs_fit_mean", "served_relative_frobenius",
             "served_mean_row_cosine", "served_row_norm_mean", "rows")
RECORD_KEYS = frozenset({
    "schema", "method_id", "artifact_digest", "corpus_digest", "fit_manifest_digest",
    "partition_manifest_digest", "serving", "nu", "rows", "turns",
    "output_variance", "output_effective_rank", "served_output_variance",
    "served_output_effective_rank", "per_turn", "turn_macro", "pooled",
    "undefined_denominators",
}) | frozenset(VIEW_KEYS) - {"rows"}

CONTROL_KINDS = ("correct", "wrong-sender", "shuffled", "null",
                 "no-attachment", "reference-token-only")
CONTROL_SCORE_KEYS = ("kind", "source", "rows", "attached", "cosine", "relative_error")
SOURCE_HIDDEN = "hidden"
SOURCE_TOKEN = "token-reference"
SOURCE_NONE = "none"

_IDENTIFIER = re.compile(r"[A-Za-z0-9._:/-]{1,256}\Z")


def _torch():
    import torch
    return torch


def _identifier(value):
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _byte_order():
    require(sys.byteorder == "little", "profile")
    return sys.byteorder


def _matrix_digest(matrix, torch, tag):
    """Content digest of one float32 matrix image; numpy memcpy, array fallback."""
    require(getattr(matrix, "ndim", 0) == 2, "profile")
    flat = matrix.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()
    try:
        import numpy
    except ImportError:
        import array
        raw = array.array("f", flat.view(-1).tolist()).tobytes()
    else:
        raw = bytes(memoryview(numpy.ascontiguousarray(flat.numpy())))
    require(len(raw) == int(flat.numel()) * 4, "numerical")
    value = hashlib.sha256()
    value.update(f"{tag}:{int(matrix.shape[0])}x{int(matrix.shape[1])}:float32:".encode("ascii"))
    value.update(raw)
    return value.hexdigest()


@dataclass(frozen=True)
class TurnBlock:
    """Every row of one captured turn, tagged by its turn id and family.

    reference_embeddings is the token channel of the same span (no hidden
    state). wrong_sender_hidden is a matched-shape hidden block from a
    different sender. family groups turns that share a source fact or context,
    so near-duplicate prompts stay on one side of a split; a turn without a
    family is its own family. The optional controls may be absent, and a
    control that was not captured must read as absent, never as zero.
    """
    turn_id: str
    hidden: object
    embeddings: object
    reference_embeddings: object = None
    wrong_sender_hidden: object = None
    family: object = None

    def __post_init__(self):
        require(_identifier(self.turn_id), "profile")
        require(self.family is None or _identifier(self.family), "profile")
        require(getattr(self.hidden, "ndim", None) == 2
                and getattr(self.embeddings, "ndim", None) == 2, "profile")
        require(tuple(int(x) for x in self.hidden.shape)
                == tuple(int(x) for x in self.embeddings.shape), "profile")
        rows, dim = (int(x) for x in self.hidden.shape)
        require(rows >= MIN_TURN_ROWS and dim > 2, "profile")
        for optional in (self.reference_embeddings, self.wrong_sender_hidden):
            if optional is not None:
                require(getattr(optional, "ndim", None) == 2
                        and tuple(int(x) for x in optional.shape) == (rows, dim), "profile")

    @property
    def rows(self):
        return int(self.hidden.shape[0])


def turn_order_key(turn_id):
    """Stable position of one turn in the digest order; independent of arrival."""
    require(_identifier(turn_id), "profile")
    return hashlib.sha256(f"{TURN_ORDER_DOMAIN}|{turn_id}".encode("utf-8")).hexdigest()


def ordered_blocks(blocks):
    """The corpus in digest order, refusing duplicate turn ids."""
    values = tuple(blocks)
    require(len(values) > 0, "profile")
    require(all(isinstance(value, TurnBlock) for value in values), "profile")
    ids = [value.turn_id for value in values]
    require(len(set(ids)) == len(ids), "identity")
    return tuple(sorted(values, key=lambda value: turn_order_key(value.turn_id)))


def _family_key(family):
    return hashlib.sha256(f"{SPLIT_DOMAIN}|{family}".encode("utf-8")).hexdigest()


def _partition(name, blocks, torch):
    """One contract partition: {turn_ids, hidden, embeddings, row_turn_ids,
    manifest_digest}, rows pooled in digest order and bound by content digest."""
    values = ordered_blocks(blocks)
    require(bool(values), "profile")
    hidden = torch.cat([value.hidden for value in values], dim=0).contiguous()
    embeddings = torch.cat([value.embeddings for value in values], dim=0).contiguous()
    turn_ids = tuple(value.turn_id for value in values)
    row_turn_ids = tuple(turn_id for value in values
                         for turn_id in (value.turn_id,) * value.rows)
    body = {"domain": PARTITION_DIGEST_DOMAIN, "partition": name,
            "rows": int(hidden.shape[0]), "hidden_dim": int(hidden.shape[1]),
            "byte_order": _byte_order(), "turn_ids": list(turn_ids),
            "row_turn_ids": list(row_turn_ids),
            "hidden": _matrix_digest(hidden, torch, "hidden"),
            "embeddings": _matrix_digest(embeddings, torch, "embeddings")}
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    return {"turn_ids": turn_ids, "hidden": hidden, "embeddings": embeddings,
            "row_turn_ids": row_turn_ids, "manifest_digest": digest}


def split_manifest(corpus, *, fit_turns, validation_turns, test_turns,
                   torch_module=None):
    """Freeze the three-way whole-turn split: fit, validation, final test.

    Turns are grouped by family (the turn id itself when no family is given),
    families are ordered by sha256, and each whole family is assigned to the
    partition furthest below its turn target, so no family ever crosses a
    partition and the result depends only on turn ids, families and targets.
    Targets are approximate because families are atomic; every turn is
    assigned, the manifest records the actual counts, and a partition that
    would end up empty is refused.
    """
    torch = torch_module or _torch()
    targets = {}
    for name, count in (("fit", fit_turns), ("validation", validation_turns),
                       ("test", test_turns)):
        require(type(count) is int and not isinstance(count, bool) and count >= 1,
                "profile")
        targets[name] = count
    blocks = ordered_blocks(corpus)
    families = {}
    for block in blocks:
        families.setdefault(block.family if block.family is not None else block.turn_id,
                            []).append(block)
    ordered_families = sorted(families, key=_family_key)
    keys = [_family_key(family) for family in ordered_families]
    require(len(set(keys)) == len(keys), "identity")
    assigned = {name: [] for name in ("fit", "validation", "test")}
    counts = {name: 0 for name in ("fit", "validation", "test")}
    for family in ordered_families:
        members = tuple(families[family])
        target = max(targets, key=lambda name: targets[name] - counts[name])
        assigned[target].extend(members)
        counts[target] += len(members)
    for name in ("fit", "validation", "test"):
        require(bool(assigned[name]), "profile")
    partitions = {name: _partition(name, assigned[name], torch)
                  for name in ("fit", "validation", "test")}
    manifest = {"schema": SPLIT_SCHEMA,
                "fit": partitions["fit"], "validation": partitions["validation"],
                "test": partitions["test"], "turns": counts, "targets": dict(targets),
                "manifest_digest": hashlib.sha256(_canonical({
                    "schema": SPLIT_SCHEMA, "targets": dict(targets), "turns": counts,
                    "partitions": {name: partitions[name]["manifest_digest"]
                                   for name in ("fit", "validation", "test")},
                })).hexdigest()}
    return manifest


def select_calibration_rows(fit_blocks, row_budget):
    """Whole turns from the digest order, as a prefix that fits the budget.

    The first turn that would exceed the budget ends the selection, so the
    result depends only on the ordered turns and on the budget. A corpus
    smaller than the budget is not padded: the caller must report the rows it
    actually got.
    """
    require(type(row_budget) is int and row_budget in M_ROW_BUDGETS, "profile")
    ordered = ordered_blocks(fit_blocks)
    selected, rows = [], 0
    for block in ordered:
        if rows + block.rows > row_budget:
            break
        selected.append(block)
        rows += block.rows
    require(rows >= MIN_CALIBRATION_ROWS, "profile")
    return tuple(selected), rows


def pooled_rows(blocks, *, torch_module=None):
    """One row block over whole turns: hidden rows and their embeddings."""
    torch = torch_module or _torch()
    values = ordered_blocks(blocks)
    hidden = torch.cat([value.hidden for value in values], dim=0)
    embeddings = torch.cat([value.embeddings for value in values], dim=0)
    require(hidden.shape == embeddings.shape and int(hidden.shape[0]) > 0, "profile")
    return hidden.contiguous(), embeddings.contiguous()


def _unit(matrix, floor=NORM_FLOOR):
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(floor)


def _cosine(left, right):
    return float((_unit(left) * _unit(right)).sum(-1).mean())


def _scaled_target(embeddings, target_norm):
    """Per-row reference at the scalar target radius, the production convention."""
    norms = embeddings.norm(dim=-1, keepdim=True).clamp_min(NORM_FLOOR)
    require(math.isfinite(target_norm) and target_norm > 0.0, "numerical")
    return embeddings * (target_norm / norms)


def _relative_error(predicted, target):
    return float((predicted - target).norm() / target.norm().clamp_min(NORM_FLOOR))


def _median(values):
    ordered = sorted(values)
    size = len(ordered)
    require(size > 0, "profile")
    middle = size // 2
    return float(ordered[middle]) if size % 2 else float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _scores(values):
    result = [float(value) for value in values]
    require(len(result) >= 1, "profile")
    require(all(math.isfinite(value) for value in result), "numerical")
    return result


def separation(correct_scores, control_scores):
    """Paired correct-arm minus control-arm separation of per-trial margins.

    A score is the pre-registered margin log P(correct label) - max log P(other
    labels) at the first generation position, so a positive score is a correct
    trial and accuracy is the fraction of positive scores.
    """
    correct = _scores(correct_scores)
    control = _scores(control_scores)
    require(len(correct) == len(control), "profile")
    trials = len(correct)
    accuracy = (sum(1 for value in correct if value > 0) / trials
                - sum(1 for value in control if value > 0) / trials)
    return {"trials": trials,
            "delta_margin": _median([left - right for left, right in zip(correct, control)]),
            "delta_accuracy": accuracy}


def separate_candidates(candidate_scores):
    """Per-control separation against the correct arm; empty when not supplied."""
    if candidate_scores is None:
        return {}
    require(type(candidate_scores) is dict and "correct" in candidate_scores, "profile")
    require(set(candidate_scores) <= set(CONTROL_KINDS), "profile")
    controls = sorted(name for name in candidate_scores if name != "correct")
    require(bool(controls), "profile")
    return {name: separation(candidate_scores["correct"], candidate_scores[name])
            for name in controls}


def control_rows(block, kind, *, torch_module=None):
    """Rows of one matched control arm; (source, rows) with rows None when absent."""
    torch = torch_module or _torch()
    require(isinstance(block, TurnBlock) and kind in CONTROL_KINDS, "profile")
    if kind == "correct":
        return SOURCE_HIDDEN, block.hidden
    if kind == "wrong-sender":
        require(block.wrong_sender_hidden is not None, "profile")
        return SOURCE_HIDDEN, block.wrong_sender_hidden
    if kind == "shuffled":
        # Reversal, not a random permutation: reproducible without a seed.
        return SOURCE_HIDDEN, torch.flip(block.hidden, [0]).contiguous()
    if kind == "null":
        return SOURCE_HIDDEN, torch.zeros_like(block.hidden)
    if kind == "reference-token-only":
        require(block.reference_embeddings is not None, "profile")
        return SOURCE_TOKEN, block.reference_embeddings
    return SOURCE_NONE, None


def score_controls(apply, artifact, block, *, target_norm=None, torch_module=None):
    """Score every control arm of one turn against that turn's embeddings.

    apply is the method's apply_map; the hidden arms go through the frozen map
    and then the N1 serving transform at the matched radius, so every attached
    arm is compared at the same scale. target_norm defaults to the turn's own
    mean row norm because a single turn is a diagnostic; the pooled evaluation
    always passes the calibration mean. Only the attached arms produce a
    cosine: the no-attachment arm reports attached=0 with no score instead of
    a fabricated zero.
    """
    torch = torch_module or _torch()
    require(callable(apply), "profile")
    require(isinstance(block, TurnBlock), "profile")
    with torch.no_grad():
        embeddings = block.embeddings.float()
        require(bool(torch.isfinite(embeddings).all()), "numerical")
        norms = embeddings.norm(dim=-1, keepdim=True).clamp_min(NORM_FLOOR)
        scalar = float(norms.mean()) if target_norm is None else float(target_norm)
        require(math.isfinite(scalar) and scalar > 0.0, "numerical")
        target = _scaled_target(embeddings, scalar)
        result = {}
        for kind in CONTROL_KINDS:
            source, rows = control_rows(block, kind, torch_module=torch)
            if rows is None:
                result[kind] = {"kind": kind, "source": source, "rows": 0, "attached": 0,
                                "cosine": None, "relative_error": None}
                continue
            rows = rows.float()
            require(bool(torch.isfinite(rows).all()), "numerical")
            prefix = (serving_transform({"kind": "n1"},
                                        apply(artifact, rows, torch_module=torch), nu=scalar)
                      if source == SOURCE_HIDDEN else _scaled_target(rows, scalar))
            require(bool(torch.isfinite(prefix).all()), "numerical")
            result[kind] = {"kind": kind, "source": source, "rows": int(prefix.shape[0]),
                            "attached": 1, "cosine": _cosine(prefix, target),
                            "relative_error": _relative_error(prefix, target)}
        for value in result.values():
            require(set(value) == set(CONTROL_SCORE_KEYS), "profile")
        return result


def _require_partition(partition, torch):
    """The contract partition, for evaluation: exactly five keys, whole turns."""
    require(isinstance(partition, Mapping), "profile")
    require(set(partition) == set(PARTITION_FIELDS), "profile")
    hidden, embeddings = partition["hidden"], partition["embeddings"]
    for matrix in (hidden, embeddings):
        require(hasattr(matrix, "dtype") and matrix.dtype.is_floating_point, "profile")
        require(getattr(matrix, "ndim", None) == 2, "profile")
    require(hidden.shape == embeddings.shape, "profile")
    require(hidden.device == embeddings.device, "profile")
    rows = int(hidden.shape[0])
    require(rows >= 2, "profile")
    row_turn_ids = partition["row_turn_ids"]
    turn_ids = partition["turn_ids"]
    require(isinstance(row_turn_ids, (tuple, list)) and isinstance(turn_ids, (tuple, list)),
            "profile")
    row_turn_ids, turn_ids = tuple(row_turn_ids), tuple(turn_ids)
    require(len(row_turn_ids) == rows, "profile")
    require(all(_identifier(value) for value in row_turn_ids), "profile")
    require(all(_identifier(value) for value in turn_ids), "profile")
    require(len(set(turn_ids)) == len(turn_ids) and set(turn_ids) == set(row_turn_ids),
            "profile")
    digest = partition["manifest_digest"]
    require(type(digest) is str and (digest == "" or len(digest) == 64), "profile")
    return hidden, embeddings, row_turn_ids, turn_ids, digest


def _serving_config(serving):
    require(serving is None or (type(serving) is dict and set(serving) == {"kind"}
                                and serving["kind"] == "n1"), "profile")
    return None if serving is None else dict(serving)


def _fit_embedding_mean(artifact, dim):
    for name in ("mean_embedding", "mu_e"):
        value = getattr(artifact, name, None)
        if value is not None:
            require(getattr(value, "ndim", None) in (1, 2)
                    and int(value.numel()) == dim, "profile")
            return value.reshape(1, dim)
    require(False, "profile")


def _row_metrics(predicted, target, mu_fit):
    """Pooled spec-12 block for one row subset; None on a zero denominator."""
    sse = float((predicted - target).pow(2).sum())
    sst = float((target - target.mean(0, keepdim=True)).pow(2).sum())
    sse_mean = float((target - mu_fit).pow(2).sum())
    require(math.isfinite(sse) and math.isfinite(sst) and math.isfinite(sse_mean),
            "numerical")
    return {"sse": sse, "sst_test": sst, "sse_vs_fit_mean": sse_mean,
            "r2_standard": None if sst == 0.0 else 1.0 - sse / sst,
            "r2_vs_fit_mean": None if sse_mean == 0.0 else 1.0 - sse / sse_mean,
            "relative_frobenius": float((predicted - target).norm()
                                         / max(float(target.norm()), NORM_FLOOR)),
            "mean_row_cosine": _cosine(predicted, target)}


def _view(z_raw, z_served, target, mu_fit):
    """One uniform metric view: raw and served kept distinct, plus row count."""
    raw = _row_metrics(z_raw, target, mu_fit)
    served = _row_metrics(z_served, target, mu_fit)
    return {"r2_standard": raw["r2_standard"], "r2_vs_fit_mean": raw["r2_vs_fit_mean"],
            "sse": raw["sse"], "sst_test": raw["sst_test"],
            "sse_vs_fit_mean": raw["sse_vs_fit_mean"],
            "relative_frobenius": raw["relative_frobenius"],
            "mean_row_cosine": raw["mean_row_cosine"],
            "raw_row_norm_mean": float(z_raw.norm(dim=-1).mean()),
            "served_r2_standard": served["r2_standard"],
            "served_r2_vs_fit_mean": served["r2_vs_fit_mean"],
            "served_relative_frobenius": served["relative_frobenius"],
            "served_mean_row_cosine": served["mean_row_cosine"],
            "served_row_norm_mean": float(z_served.norm(dim=-1).mean()),
            "rows": int(target.shape[0])}


def _output_stats(z, torch):
    """Variance and effective rank of the output rows, in float64 for stability."""
    centered = z - z.mean(0, keepdim=True)
    variance = float(centered.pow(2).sum() / centered.shape[0])
    singular = torch.linalg.svdvals(centered.double())
    weights = singular.pow(2)
    total = float(weights.sum())
    require(math.isfinite(variance) and math.isfinite(total), "numerical")
    if total <= 0.0:
        return variance, 0.0
    share = weights / total
    entropy = float(torch.where(share > 0, share * share.log(), torch.zeros(())).sum())
    return variance, float(math.exp(-entropy))


def evaluate_map(apply, artifact, partition, *, serving=None, nu=None,
                 torch_module=None):
    """Score a frozen artifact on an INDEPENDENT partition; record, never gate.

    apply is the method's apply_map. The artifact is scored on the partition's
    rows before any serving transform and again after the optional serving
    transform, and both metric sets are kept distinct. Zero denominators are
    recorded as undefined values plus entries in undefined_denominators;
    negative scores are recorded, never suppressed. mu_E_fit for
    r2_vs_fit_mean comes from the frozen artifact, never from the partition.
    """
    torch = torch_module or _torch()
    require(callable(apply), "profile")
    hidden, embeddings, row_turn_ids, turn_ids, digest = _require_partition(partition, torch)
    config = _serving_config(serving)
    nu_used = None
    if config is not None:
        require(isinstance(nu, (int, float)) and not isinstance(nu, bool)
                and math.isfinite(nu) and nu > 0.0, "profile")
        nu_used = float(nu)
    with torch.no_grad():
        embeddings = embeddings.float()
        require(bool(torch.isfinite(embeddings).all()), "numerical")
        z_raw = apply(artifact, hidden, torch_module=torch)
        require(getattr(z_raw, "ndim", None) == 2 and tuple(
            int(x) for x in z_raw.shape) == tuple(int(x) for x in embeddings.shape),
            "profile")
        z = z_raw.detach().to(dtype=torch.float64)
        served_raw = serving_transform(config, z_raw, nu=nu_used)
        z_served = served_raw.detach().to(dtype=torch.float64)
        target = embeddings.to(dtype=torch.float64)
        mu_fit = _fit_embedding_mean(artifact, target.shape[1]).to(dtype=torch.float64)
        pooled = _view(z, z_served, target, mu_fit)
        raw_variance, raw_rank = _output_stats(z, torch)
        served_variance, served_rank = _output_stats(z_served, torch)
        indices = {}
        for index, turn_id in enumerate(row_turn_ids):
            indices.setdefault(turn_id, []).append(index)
        per_turn = {}
        undefined = []
        for turn_id in turn_ids:
            selection = indices.get(turn_id)
            require(bool(selection), "profile")
            rows = torch.tensor(selection, dtype=torch.long)
            view = _view(z[rows], z_served[rows], target[rows], mu_fit)
            for name in ("r2_standard", "r2_vs_fit_mean", "served_r2_standard",
                         "served_r2_vs_fit_mean"):
                if view[name] is None:
                    undefined.append(f"{name}@{turn_id}")
            per_turn[turn_id] = view
        turn_macro = {}
        for name in VIEW_KEYS[:-1]:
            values = [per_turn[turn_id][name] for turn_id in turn_ids
                      if per_turn[turn_id][name] is not None]
            turn_macro[name] = sum(values) / len(values) if values else None
        for name in ("r2_standard", "r2_vs_fit_mean", "served_r2_standard",
                     "served_r2_vs_fit_mean"):
            if pooled[name] is None:
                undefined.append(name)
        binding = getattr(artifact, "binding", None)
        record = {
            "schema": RECORD_SCHEMA,
            "method_id": getattr(artifact, "method_id", None),
            "artifact_digest": getattr(artifact, "content_digest", None),
            "corpus_digest": getattr(binding, "corpus_digest", None),
            "fit_manifest_digest": getattr(binding, "fit_manifest_digest", None),
            "partition_manifest_digest": digest,
            "serving": config, "nu": nu_used,
            "rows": pooled["rows"], "turns": len(turn_ids),
            "output_variance": raw_variance, "output_effective_rank": raw_rank,
            "served_output_variance": served_variance,
            "served_output_effective_rank": served_rank,
            "per_turn": per_turn, "turn_macro": turn_macro, "pooled": pooled,
            "undefined_denominators": undefined,
        }
        record.update({name: pooled[name] for name in VIEW_KEYS[:-1]})
    return record


def record_digest(record):
    """Digest of one evaluation record; promotion evidence is pinned by it."""
    require(isinstance(record, Mapping), "profile")
    return hashlib.sha256(_canonical(record)).hexdigest()


def validate_metrics_record(record):
    """Fixed key set, exact scalar types, bounded structure. No tensor content."""
    require(isinstance(record, Mapping) and set(record) == set(RECORD_KEYS), "profile")
    require(record["schema"] == RECORD_SCHEMA, "profile")
    require(type(record["method_id"]) is str, "profile")
    for name in ("artifact_digest", "corpus_digest", "fit_manifest_digest",
                 "partition_manifest_digest"):
        value = record[name]
        require(value is None or (type(value) is str and len(value) == 64), "profile")
    require(record["serving"] is None
            or (type(record["serving"]) is dict and set(record["serving"]) == {"kind"}
                and record["serving"]["kind"] == "n1"), "profile")
    require(record["nu"] is None
            or (isinstance(record["nu"], (int, float)) and not isinstance(record["nu"], bool)
                and math.isfinite(record["nu"])), "profile")
    for name in ("rows", "turns"):
        require(type(record[name]) is int and not isinstance(record[name], bool)
                and record[name] >= 0, "profile")
    for name in VIEW_KEYS[:-1]:
        value = record[name]
        require(value is None or (type(value) is float and math.isfinite(value)),
                "numerical")
    for name in ("output_variance", "output_effective_rank", "served_output_variance",
                 "served_output_effective_rank"):
        value = record[name]
        require(type(value) is float and math.isfinite(value), "numerical")
    require(type(record["per_turn"]) is dict, "profile")
    for turn_id, view in record["per_turn"].items():
        require(type(turn_id) is str and type(view) is dict, "profile")
        require(set(view) == set(VIEW_KEYS), "profile")
        require(type(view["rows"]) is int, "profile")
        for name in VIEW_KEYS[:-1]:
            require(view[name] is None
                    or (type(view[name]) is float and math.isfinite(view[name])),
                    "numerical")
    require(type(record["turn_macro"]) is dict
            and set(record["turn_macro"]) == set(VIEW_KEYS) - {"rows"}, "profile")
    for value in record["turn_macro"].values():
        require(value is None or (type(value) is float and math.isfinite(value)),
                "numerical")
    require(type(record["pooled"]) is dict and set(record["pooled"]) == set(VIEW_KEYS),
            "profile")
    for name, value in record["pooled"].items():
        if name == "rows":
            require(type(value) is int, "profile")
        else:
            require(value is None or (type(value) is float and math.isfinite(value)),
                    "numerical")
    require(type(record["undefined_denominators"]) is list, "profile")
    require(all(type(name) is str for name in record["undefined_denominators"]),
            "profile")
    require(len(record["per_turn"]) == record["turns"], "profile")
    require(sum(view["rows"] for view in record["per_turn"].values()) == record["rows"],
            "profile")
    return True


def validate_for_activation(artifact, validation_record, policy):
    """The only promotion gate: allow or a named refusal, never a raised gate.

    The record must be bound to the artifact: either its artifact_digest equals
    the artifact's verified content digest, or the artifact binding pins the
    record by digest (bind_validation_evidence) and the record must still name
    the artifact it was measured against. Policy thresholds are fixed by the
    caller as {"min_<metric>": value} or {"max_<metric>": value} pairs over
    numeric record fields; an undefined metric is a refusal, not a zero.
    """
    if not isinstance(validation_record, Mapping):
        return ("refuse", "record")
    for name in ("schema", "artifact_digest", "rows", "turns", "r2_standard",
                 "r2_vs_fit_mean"):
        if name not in validation_record:
            return ("refuse", "record")
    binding = getattr(artifact, "binding", None)
    if binding is None:
        return ("refuse", "unbound")
    pinned = getattr(binding, "validation_record_digest", "") or ""
    if pinned:
        if record_digest(validation_record) != pinned:
            return ("refuse", "validation_evidence")
        expected = getattr(binding, "validation_artifact_digest", "") or ""
        if expected != validation_record["artifact_digest"]:
            return ("refuse", "artifact_digest")
    else:
        digest = getattr(artifact, "content_digest", None)
        if digest is not None:
            if validation_record["artifact_digest"] != digest:
                return ("refuse", "artifact_digest")
        else:
            for name in ("corpus_digest", "fit_manifest_digest"):
                expected = getattr(binding, name, None)
                if not expected or validation_record.get(name) != expected:
                    return ("refuse", "artifact_digest")
            if validation_record.get("method_id") != getattr(artifact, "method_id", None):
                return ("refuse", "artifact_digest")
    if not isinstance(policy, Mapping) or not policy:
        return ("refuse", "policy")
    for name in sorted(policy):
        threshold = policy[name]
        if not (isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
                and math.isfinite(threshold)):
            return ("refuse", "policy")
        if name.startswith("min_"):
            metric, below = name[4:], True
        elif name.startswith("max_"):
            metric, below = name[4:], False
        else:
            return ("refuse", "policy")
        if metric not in validation_record:
            return ("refuse", "policy")
        value = validation_record[metric]
        if value is None:
            return ("refuse", "undefined:" + metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(value):
            return ("refuse", "policy")
        if below and value < threshold:
            return ("refuse", name)
        if not below and value > threshold:
            return ("refuse", name)
    return ("allow", None)


def bind_validation_evidence(artifact, validation_record, serving):
    """Pin the promotion evidence and the serving choice into the binding.

    Returns a new artifact with identical tensors; the original digest
    changes, so the pinned record keeps the digest it was measured against
    (validation_artifact_digest) and the gate re-checks that pairing. The
    served artifact re-validates against the same record it was made from.
    """
    require(isinstance(validation_record, Mapping)
            and "artifact_digest" in validation_record, "profile")
    binding = getattr(artifact, "binding", None)
    require(binding is not None, "profile")
    digest = getattr(artifact, "content_digest", None)
    require(type(digest) is str and digest == validation_record["artifact_digest"],
            "identity")
    pinned = replace(binding,
                     validation_record_digest=record_digest(validation_record),
                     validation_artifact_digest=validation_record["artifact_digest"],
                     serving=_serving_config(serving))
    return replace(artifact, binding=pinned, content_digest=None)


def select_calibrated_map(fit, apply, fit_partition, validation_partition, *,
                          hyperparams_grid=None, serving_grid=SERVING_VARIANTS,
                          nu=None, torch_module=None):
    """Fit on FIT for every grid point; select hyperparameters and serving on VALIDATION.

    The final test partition is not an argument and can never enter fitting or
    selection. Hyperparameters are chosen by the raw r2_standard the frozen map
    earns on validation; the serving variant is chosen by the served r2 the
    same artifact earns on validation under N0 or N1. Ties keep the earlier
    grid entry, so the selection is deterministic.
    """
    torch = torch_module or _torch()
    require(callable(fit) and callable(apply), "profile")
    grid = tuple(dict(value) for value in
                 (hyperparams_grid if hyperparams_grid is not None
                  else tuple({"rho": rho} for rho in RHO_GRID)))
    require(bool(grid), "profile")
    variants = tuple(serving_grid)
    require(bool(variants) and set(variants) <= {"n0", "n1"}, "profile")
    if "n1" in variants:
        require(isinstance(nu, (int, float)) and not isinstance(nu, bool)
                and math.isfinite(nu) and nu > 0.0, "profile")
        nu = float(nu)

    def raw_score(record):
        value = record["r2_standard"]
        return float("-inf") if value is None else float(value)

    candidates = []
    for hyperparams in grid:
        artifact = fit(fit_partition, hyperparams, torch_module=torch)
        record = evaluate_map(apply, artifact, validation_partition, torch_module=torch)
        candidates.append((hyperparams, artifact, record))
    best_hyper, best_artifact, _ = max(candidates, key=lambda entry: raw_score(entry[2]))
    serving_candidates = []
    for variant in variants:
        config = None if variant == "n0" else {"kind": "n1"}
        record = evaluate_map(apply, best_artifact, validation_partition,
                              serving=config, nu=nu, torch_module=torch)
        serving_candidates.append((variant, config, record))

    def served_score(entry):
        value = entry[2]["served_r2_standard"]
        return float("-inf") if value is None else float(value)

    best_variant, best_config, best_record = max(serving_candidates, key=served_score)
    return {"schema": SELECTION_SCHEMA,
            "hyperparams": dict(best_hyper), "serving": best_config,
            "serving_variant": best_variant, "artifact": best_artifact,
            "validation_record": best_record,
            "candidates": [{"hyperparams": dict(hyperparams),
                            "r2_standard": record["r2_standard"],
                            "artifact_digest": record["artifact_digest"]}
                           for hyperparams, _, record in candidates],
            "serving_candidates": [{"variant": variant,
                                    "served_r2_standard": record["served_r2_standard"]}
                                   for variant, _, record in serving_candidates]}
