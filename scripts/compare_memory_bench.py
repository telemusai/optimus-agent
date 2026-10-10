#!/usr/bin/env python3
"""Compare existing memory_bench runs and memory-bench-scores/1 artifacts offline.

Library: compare_arms([{"name": "baseline", "run": Path(...), "score": Path(...)},
                       {"name": "candidate", "run": Path(...), "score": Path(...)}]).
An optional "variant" selects records from a mixed-variant run. No scorer is run.
A score artifact's optional run_sha256 must match the exact input run bytes.
Older artifacts without it remain accepted with an explicit metadata-only caveat.

All-attempted means use unique selected run qids, including driver failures. They
retain artifact scores for flagged evaluations, NOT a claim of model quality.
Missing scores (including missing_record placeholders) are never imputed: the
full mean is null if coverage is incomplete; available_score_mean is separate.
Judge-success sensitivity uses answered, unflagged evaluations only (including
deterministic evaluators). It reflects recorded flags, not a new judge audit.
Pairs use qids actually answered without driver errors in BOTH runs. Up/down/tie
is right minus left on scored pairs; missing pairs and flags remain explicit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

SCHEMA = "memory-bench-comparison/1"
SCORE_SCHEMA = "memory-bench-scores/1"
DEDUP_POLICIES = ("error", "identical", "first", "last")
JUDGE_TYPES = {"llm_judge", "llm_abstention", "llm_gotchas"}
DRIVER_FLAGS = {"driver_error", "empty_answer"}


class ComparisonError(ValueError):
    """Input evidence is malformed, incompatible, or ambiguous."""


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ComparisonError(f"non-finite JSON number: {value}")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        _nonfinite(value)
    return number


def _parse(text, source):
    try:
        return json.loads(text, object_pairs_hook=_object_pairs,
                          parse_constant=_nonfinite, parse_float=_float)
    except ValueError as exc:
        raise ComparisonError(f"{source}: {exc}") from exc


def _read(path):
    path = Path(path)
    data = path.read_bytes()
    lineage = {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
               "bytes": len(data)}
    return data.decode("utf-8-sig"), lineage


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ComparisonError(f"{label} must be a nonempty string")
    return value


def _records(rows, label):
    for position, row in rows:
        if not isinstance(row, dict):
            raise ComparisonError(f"{label}:{position}: expected an object")
        _text(row.get("qid"), f"{label}:{position}: qid")
        for key in ("variant", "benchmark", "run_id", "env_id"):
            if row.get(key) is not None:
                _text(row[key], f"{label}:{position}: {key}")
    return rows


def _deduplicate(rows, policy, label):
    groups = {}
    for position, row in rows:
        groups.setdefault(row["qid"], []).append((position, row))
    selected, duplicates = {}, []
    for qid, group in groups.items():
        divergent = any(row != group[0][1] for _, row in group[1:])
        if len(group) > 1:
            if policy == "error" or (policy == "identical" and divergent):
                kind = "divergent" if divergent else "identical"
                raise ComparisonError(f"{label}: {kind} duplicate qid {qid!r}; "
                                      "choose an explicit --dedup policy")
            kept = group[-1] if policy == "last" else group[0]
            duplicates.append({"qid": qid, "positions": [p for p, _ in group],
                               "divergent": divergent, "kept_position": kept[0]})
        else:
            kept = group[0]
        selected[qid] = kept[1]
    return selected, duplicates


def _values(rows, key):
    return sorted({row[key] for row in rows if row.get(key) is not None})


def _match(left, right, label):
    if left is not None and right is not None and left != right:
        raise ComparisonError(f"{label} mismatch: {left!r} != {right!r}")


def _status(row):
    error = str(row.get("error") or "").lower()
    stop = str(row.get("stop_reason") or "").lower()
    if row.get("timed_out") or stop in ("timeout", "timed_out") or any(
            word in error for word in ("timeout", "timed out", "timed_out")):
        return "timeout"
    if error or stop in ("error", "aborted", "cancelled", "canceled"):
        return "driver_error"
    answer = row.get("answer_text")
    if answer is not None and not isinstance(answer, str):
        raise ComparisonError(f"run qid {row['qid']!r}: answer_text must be a string or null")
    return "answered" if answer and answer.strip() else "empty_answer"


def _flags(row):
    flags = []
    for key in ("flag", "judge_flag", "judge_error", "eval_error", "error"):
        value = row.get(key)
        if value:
            if key in ("flag", "judge_flag"):
                _text(value, f"score qid {row['qid']!r}: {key}")
                flags.append(value)
            else:
                flags.append(key)
    return sorted(set(flags))


def _metric(qids, values):
    qids = sorted(qids)
    present = [qid for qid in qids if qid in values]
    total = math.fsum(values[qid] for qid in present)
    mean = total / len(present) if present else None
    return {"denominator": len(qids), "scored_n": len(present), "score_sum": total,
            "mean": mean if len(present) == len(qids) else None,
            "available_score_mean": mean, "missing_score_qids": sorted(set(qids) - set(present))}


def _load_arm(spec, dedup):
    name = _text(spec.get("name"), "arm name")
    variant = spec.get("variant")
    if variant is not None:
        _text(variant, f"{name}: selected variant")
    run_text, run_lineage = _read(spec["run"])
    score_text, score_lineage = _read(spec["score"])
    run_rows = _records([(i, _parse(line, f"{name} run:{i}"))
                         for i, line in enumerate(run_text.splitlines(), 1) if line.strip()], "run")
    selected = [(i, row) for i, row in run_rows if variant is None or row.get("variant") == variant]
    if variant is not None and not selected:
        raise ComparisonError(f"{name}: no run records for variant {variant!r}")
    variants = _values([row for _, row in selected], "variant")
    if len(variants) > 1 or (variants and any(row.get("variant") is None for _, row in selected)):
        raise ComparisonError(f"{name}: ambiguous run variants; select --variant NAME VARIANT")
    variant = variant if variant is not None else (variants[0] if variants else None)
    runs, run_duplicates = _deduplicate(selected, dedup, f"{name} run")
    artifact = _parse(score_text, f"{name} score")
    if not isinstance(artifact, dict) or artifact.get("schema") != SCORE_SCHEMA:
        raise ComparisonError(f"{name}: expected {SCORE_SCHEMA}")
    declared_run_sha256 = artifact.get("run_sha256")
    binding = "available metadata only; score artifact has no run-content hash"
    if "run_sha256" in artifact:
        if (not isinstance(declared_run_sha256, str) or len(declared_run_sha256) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in declared_run_sha256)):
            raise ComparisonError(f"{name}: run_sha256 must be exactly 64 hexadecimal characters")
        if declared_run_sha256.lower() != run_lineage["sha256"]:
            raise ComparisonError(f"{name}: run_sha256 mismatch with exact input run bytes")
        binding = "verified run-content SHA-256"
    results = artifact.get("results")
    if not isinstance(results, list):
        raise ComparisonError(f"{name}: score results must be a list")
    score_rows = _records(list(enumerate(results, 1)), "score results")
    for i, row in score_rows:
        value = row.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ComparisonError(f"{name} score results:{i}: score must be finite and in [0, 1]")
        _flags(row)
    scores, score_duplicates = _deduplicate(score_rows, dedup, f"{name} score")
    meta = artifact.get("run")
    meta = {} if meta is None else meta
    if not isinstance(meta, dict):
        raise ComparisonError(f"{name}: score run metadata must be an object")
    for key in ("variant", "benchmark", "run_id"):
        if meta.get(key) is not None:
            _text(meta[key], f"{name}: score run {key}")
    _match(variant, meta.get("variant"), f"{name}: variant")
    benchmarks = _values([row for _, row in selected], "benchmark")
    if len(benchmarks) > 1:
        raise ComparisonError(f"{name}: mixed run benchmarks")
    benchmark = benchmarks[0] if benchmarks else None
    for candidate in (artifact.get("bench"), meta.get("benchmark")):
        if candidate is not None:
            _text(candidate, f"{name}: benchmark")
        _match(benchmark, candidate, f"{name}: benchmark")
        benchmark = benchmark or candidate
    run_ids = _values(list(runs.values()), "run_id")
    if run_ids and meta.get("run_id") is not None and meta["run_id"] not in run_ids:
        raise ComparisonError(f"{name}: score run_id is not present in selected run records")
    if len(_values(results, "variant")) > 1:
        raise ComparisonError(f"{name}: mixed score row variants")
    for _, row in score_rows:
        _match(variant, row.get("variant"), f"{name}: score row variant")
        _match(meta.get("variant"), row.get("variant"), f"{name}: score metadata variant")
        _match(benchmark, row.get("benchmark"), f"{name}: score row benchmark")

    statuses, flags, values, successful = {}, {}, {}, set()
    issues, judge_errors, evaluator_errors = [], [], []
    for qid, record in runs.items():
        statuses[qid] = status = _status(record)
        row = scores.get(qid)
        if row is None:
            continue
        _match(record.get("env_id"), row.get("env_id"), f"{name}: qid {qid!r} env_id")
        _match(record.get("run_id"), row.get("run_id"), f"{name}: qid {qid!r} run_id")
        evaluation = record.get("eval")
        evaluation = {} if evaluation is None else evaluation
        if not isinstance(evaluation, dict):
            raise ComparisonError(f"{name}: qid {qid!r} eval must be an object")
        _match(evaluation.get("type"), row.get("eval_type"), f"{name}: qid {qid!r} eval_type")
        flags[qid] = row_flags = _flags(row)
        if "missing_record" in row_flags:
            issues.append({"qid": qid, "kind": "score_marks_existing_run_record_missing"})
            continue
        flag = row.get("flag")
        if flag == "driver_error" and status not in ("driver_error", "timeout"):
            raise ComparisonError(f"{name}: qid {qid!r} driver_error contradicts run record")
        if flag == "empty_answer" and status != "empty_answer":
            raise ComparisonError(f"{name}: qid {qid!r} empty_answer contradicts run record")
        if status != "answered" and (flag not in DRIVER_FLAGS or row["score"] != 0):
            raise ComparisonError(f"{name}: qid {qid!r} score contradicts failed/empty run record")
        values[qid] = float(row["score"])
        if status == "answered" and not row_flags:
            successful.add(qid)
        elif any(f not in DRIVER_FLAGS for f in row_flags):
            if row.get("eval_type", evaluation.get("type")) in JUDGE_TYPES:
                judge_errors.append(qid)
            else:
                evaluator_errors.append(qid)
    answered = {qid for qid, status in statuses.items() if status == "answered"}
    extra = sorted(set(scores) - set(runs))
    flag_counts = Counter(flag for row_flags in flags.values() for flag in row_flags)
    summary = {
        "name": name, "benchmark": benchmark, "variant": variant,
        "lineage": {"run": run_lineage, "score": score_lineage,
                    "score_declared_run_file": artifact.get("run_file"),
                    "run_ids": run_ids, "score_declared_run_id": meta.get("run_id"),
                    "score_declared_variant": meta.get("variant"),
                    "score_declared_run_sha256": declared_run_sha256,
                    "binding": binding},
        "duplicates": {"run": run_duplicates, "score": score_duplicates},
        "coverage": {
            "run_records": len(run_rows), "selected_run_records": len(selected),
            "excluded_variant_records": len(run_rows) - len(selected),
            "attempted_n": len(runs), "answered_n": len(answered),
            "score_rows": len(results), "score_unique_qids": len(scores),
            "attempted_qids": sorted(runs), "answered_qids": sorted(answered),
            "missing_score_qids": sorted(set(runs) - set(values)),
            "unattempted_score_qids": extra,
            "unattempted_score_flags": dict(sorted(Counter(
                flag for qid in extra for flag in _flags(scores[qid])).items())),
        },
        "driver_status_counts": dict(sorted(Counter(statuses.values()).items())),
        "driver_status_by_qid": dict(sorted(statuses.items())),
        "score_flags": dict(sorted(flag_counts.items())),
        "score_flags_by_qid": {qid: f for qid, f in sorted(flags.items()) if f},
        "judge_flagged_qids": sorted(judge_errors),
        "evaluator_flagged_qids": sorted(evaluator_errors), "coverage_issues": issues,
        "all_attempted": _metric(runs, values),
        "judge_success": dict(_metric(successful, values), qids=sorted(successful)),
    }
    return {"summary": summary, "runs": runs, "values": values,
            "answered": answered, "successful": successful}


def _paired(left, right, qids):
    available = set(qids) & left["values"].keys() & right["values"].keys()
    deltas = [right["values"][qid] - left["values"][qid] for qid in sorted(available)]
    full = len(available) == len(qids)
    delta = math.fsum(deltas) / len(deltas) if deltas else None
    return {"denominator": len(qids), "paired_n": len(available), "qids": sorted(available),
            "missing_score_qids": sorted(set(qids) - available),
            "left": _metric(qids, {qid: left["values"][qid] for qid in available}),
            "right": _metric(qids, {qid: right["values"][qid] for qid in available}),
            "mean_delta": delta if full else None, "available_mean_delta": delta,
            "up": sum(d > 0 for d in deltas), "down": sum(d < 0 for d in deltas),
            "tie": sum(d == 0 for d in deltas)}


def compare_arms(arms, *, dedup="error"):
    """Return JSON-safe evidence. Inputs are explicit name/run/score/variant dicts.

    Raise ComparisonError on ambiguity. first/last refer to physical input order
    in both files, never score size; identical permits only equal full records.
    """
    if dedup not in DEDUP_POLICIES:
        raise ComparisonError(f"unknown dedup policy: {dedup!r}")
    if len(arms) < 2:
        raise ComparisonError("at least two explicit arms are required")
    names = [_text(arm.get("name"), "arm name") for arm in arms]
    if len(set(names)) != len(names):
        raise ComparisonError("arm names must be unique")
    loaded = [_load_arm(arm, dedup) for arm in arms]
    pairs = []
    for left, right in combinations(loaded, 2):
        _match(left["summary"]["benchmark"], right["summary"]["benchmark"], "cross-arm benchmark")
        for qid in left["runs"].keys() & right["runs"].keys():
            for field in ("env_id", "question", "eval", "answer"):
                _match(left["runs"][qid].get(field), right["runs"][qid].get(field),
                       f"cross-arm qid {qid!r} {field}")
        common = left["answered"] & right["answered"]
        sensitivity = common & left["successful"] & right["successful"]
        pairs.append({"left": left["summary"]["name"], "right": right["summary"]["name"],
                      "common_answered_qids": sorted(common),
                      "left_only_answered_qids": sorted(left["answered"] - right["answered"]),
                      "right_only_answered_qids": sorted(right["answered"] - left["answered"]),
                      "all_scored": _paired(left, right, common),
                      "judge_success": _paired(left, right, sensitivity),
                      "judge_success_excluded_qids": sorted(common - sensitivity)})
    common = set.intersection(*(arm["answered"] for arm in loaded))
    return {"schema": SCHEMA, "dedup_policy": dedup,
            "semantics": {
                "all_attempted": "unique selected run qids; artifact scores retained including flags",
                "missing_scores": "never imputed; incomplete full means are null",
                "judge_success": "answered, no recorded score flags/errors; includes deterministic evaluation",
                "paired": "intersection of answered run qids; right minus left; exact ties",
                "caution": "flagged artifact zeros are not evidence of model failure; no causal or confidence claim",
            },
            "common_answered_qids": sorted(common),
            "arms": [arm["summary"] for arm in loaded], "pairs": pairs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", nargs=3, action="append", required=True,
                        metavar=("NAME", "RUN_JSONL", "SCORE_JSON"),
                        help="repeat at least twice; input order sets left/right")
    parser.add_argument("--variant", nargs=2, action="append", default=[], metavar=("NAME", "VARIANT"),
                        help="select this arm's run variant; score metadata must match")
    parser.add_argument("--dedup", choices=DEDUP_POLICIES, default="error",
                        help="duplicate qids: fail (default), equal full rows only, or physical first/last")
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout; cannot overwrite inputs")
    args = parser.parse_args(argv)
    try:
        variants = {}
        names = {name for name, _, _ in args.arm}
        for name, variant in args.variant:
            if name not in names or name in variants:
                raise ComparisonError("--variant must name a declared arm exactly once")
            variants[name] = variant
        arms = [{"name": name, "run": run, "score": score, "variant": variants.get(name)}
                for name, run, score in args.arm]
        if args.out:
            input_paths = [Path(arm[key]) for arm in arms for key in ("run", "score")]
            if any(args.out.resolve() == path.resolve() for path in input_paths):
                raise ComparisonError("output cannot overwrite an input file")
            try:
                args.out.stat()
            except FileNotFoundError:
                pass
            else:
                # Resolved paths do not identify hard links. Inspection errors
                # must reach the CLI error handler rather than permit a write.
                if any(args.out.samefile(path) for path in input_paths):
                    raise ComparisonError("output cannot overwrite an input file")
        result = compare_arms(arms, dedup=args.dedup)
        output = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            args.out.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 0
    except (ComparisonError, OSError, UnicodeError) as exc:
        sys.stderr.write(json.dumps({"schema": SCHEMA, "error": str(exc)}) + "\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
