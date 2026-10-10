"""Validate score bindings, then report noncausal local overlap diagnostics.

Raw outputs are private content-bearing evidence. No flag, missing store, or
legacy unbound score is converted into proof of a memory failure stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.evidence import (IDENTITY_FIELDS, effective_variant, jsonl_records, native_failure,
                            read_questions, run_identity, validate_scores)
from bench.eval.local_scoring import normalized_tokens
from bench.io import atomic_write_text, check_output_path
from bench.protocol import canonical_hash, question_set_hash


def gold_tokens(gold: str) -> set:
    return set(normalized_tokens(gold or ""))


def entry_tokens(entry: dict) -> set:
    text = " ".join(str(entry.get(key) or "") for key in ("title", "content"))
    return set(normalized_tokens(text))


def contains_gold(entry: dict, gtoks: set, threshold: float = 0.6) -> bool:
    return bool(gtoks) and len(gtoks & entry_tokens(entry)) >= max(1, round(threshold * len(gtoks)))


def _environments(raw, required):
    """Join only exact declared identities; run dedup cannot choose a snapshot."""
    if not required:
        return {}
    selected = {}
    for row in jsonl_records(raw):
        identity = run_identity(row)
        if identity not in required:
            continue
        if identity in selected:
            raise ValueError("ambiguous snapshot identity; duplicate exact run/benchmark/variant/env")
        selected[identity] = row
    if required != set(selected):
        raise ValueError("missing exact snapshot run/benchmark/variant/env identity")
    return selected

def attribute(args) -> dict:
    all_questions, question_files = read_questions(Path(args.data_common) / args.bench)
    raw_run = Path(args.runs).read_bytes()
    raw_scores = Path(args.scores).read_bytes()
    scores = json.loads(raw_scores)
    dedup = getattr(args, "dedup", "last")
    questions, runs, score_rows, status = validate_scores(
        scores, raw_run, all_questions, question_files, family=args.bench,
        variant=args.variant, dedup=dedup,
        allow_legacy_unverified=getattr(args, "allow_legacy_unverified", False),
    )
    raw_envs = Path(args.envs).read_bytes()
    selected_variant = effective_variant(runs, args.variant)
    required = set()
    if status != "unverified-legacy" and selected_variant != "nomem":
        for question in questions:
            record, scored = runs.get(question["qid"]), score_rows.get(question["qid"])
            if scored is not None and not scored.get("flag") and not any(native_failure(record).values()):
                required.add(run_identity(record))
    environments = _environments(raw_envs, required)
    per_question = []
    counts, abstention = Counter(), Counter()
    for question in questions:
        qid = question["qid"]
        record, scored = runs.get(qid), score_rows.get(qid)
        row = {"qid": qid, "env_id": question.get("env_id"), "category": question.get("category")}
        if status == "unverified-legacy":
            row.update(stage="unverified-legacy", recorded_score=scored.get("score") if scored else None)
        elif scored is None or scored.get("flag"):
            row.update(stage="score-flagged", flag=scored.get("flag") if scored else "unscored")
        elif any(native_failure(record).values()):
            row.update(stage="native-failure", native_failure=native_failure(record))
        else:
            row.update(stage="scored", score=scored["score"], passed_threshold=scored["score"] >= 0.5)
            if question["eval"]["type"] == "abstain_f1":
                abstention["threshold-met" if row["passed_threshold"] else "threshold-not-met"] += 1
            identity = run_identity(record)
            row["declared_run_identity"] = dict(zip(IDENTITY_FIELDS, identity))
            if selected_variant == "nomem":
                row["diagnostic"] = "memory-off-declared"
            else:
                env = environments[identity]
                snapshot = env.get("store_snapshot")
                if not isinstance(snapshot, dict) or not isinstance(snapshot.get("memory"), dict):
                    raise ValueError("snapshot requires an explicit memory object")
                store = snapshot["memory"]
                target = question.get("answer") or question["eval"].get("params", {}).get("evidence_cue") or ""
                targets = [gold_tokens(part) for part in str(target).splitlines() if part.strip()]
                matches = [key for key, value in store.items()
                           if any(contains_gold(value, tokens) for tokens in targets)]
                recalls = set(record.get("recall_ids") or [])
                search_ids = [item.get("id") for item in record.get("search_ranked", [])]
                aliases = set(matches) | {"project:memory:" + key for key in matches}
                ranks = [index for index, key in enumerate(search_ids, 1) if key in aliases]
                row["diagnostic"] = "heuristic-overlap-only"
                row["snapshot_record_sha256"] = canonical_hash(env)
                row["overlap_ids"] = matches
                row["overlap_recall_ids"] = sorted(aliases & recalls)
                row["best_overlap_search_rank"] = min(ranks) if ranks else None
                row["has_text_target"] = bool(targets)
        counts[row["stage"]] += 1
        per_question.append(row)
    return {
        "schema": "memory-bench-attribution/2", "bench": args.bench, "variant": args.variant,
        "effective_variant": selected_variant,
        "dedup": dedup, "questions": len(questions), "binding_status": status,
        "snapshot_binding_status": ("unverified-legacy" if status == "unverified-legacy" else
                                    "matched-exact-identities" if required else "not-applicable"),
        "binding_claim": "caller-declared local artifact consistency only; not actual presentation or gold proof",
        "interpretation": "text-overlap diagnostics only; not causal capture/retrieval/injection/reasoning proof",
        "input_sha256": {
            "run": hashlib.sha256(raw_run).hexdigest(), "scores": hashlib.sha256(raw_scores).hexdigest(),
            "environments": hashlib.sha256(raw_envs).hexdigest(),
            "question_set": question_set_hash(all_questions),
        },
        "question_files": question_files, "funnel": dict(counts), "abstention": dict(abstention),
        "per_question": per_question,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("runs", "envs", "bench", "out", "scores"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--variant")
    parser.add_argument("--dedup", choices=["error", "first", "last"], default="last")
    parser.add_argument("--allow-legacy-unverified", action="store_true",
                        help="read unbound old scores, labeled unverified; no overlap/stage inference")
    parser.add_argument("--data-common", default=os.environ.get("MEMORY_BENCH_COMMON_ROOT") or
                        str(Path(__file__).resolve().parents[2] / "data" / "common"))
    args = parser.parse_args(argv)
    inputs = [Path(args.runs), Path(args.envs), Path(args.scores),
              *(Path(args.data_common) / args.bench).glob("qs_*.jsonl")]
    try:
        output = check_output_path(args.out, inputs)
        result = attribute(args)
        atomic_write_text(output, json.dumps(result, ensure_ascii=False, indent=1))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"attribution rejected: {type(exc).__name__}; check inputs, bindings, and explicit legacy opt-in\n")
    print(f"[attribute] {result['binding_status']}: {result['questions']} rows -> {output}")


if __name__ == "__main__":
    main()
