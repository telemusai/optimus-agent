"""Evaluate an explicit local run under versioned scoring and question protocols.

No whole-stack upstream parity is claimed. LoCoMo and cue scores use
optimus-local-scoring/1.0.0; permissive LME ports retain separate identities.
Output remains memory-bench-scores/1 with mandatory new scoring provenance.
All loaded questions remain in zero-filled aggregates; flags are not valid
judge evidence. Network is off unless explicit DGX config and opt-in are given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import deterministic as det
from . import judges
from bench.io import atomic_write_text, check_output_path
from bench.evidence import (read_questions, select_run_records, make_provenance, native_failure,
                            validate_driver_question)
from bench.protocol import canonical_hash, evaluator_identity, validate_question_protocol

DEFAULT_DATA_COMMON = Path(os.environ.get("MEMORY_BENCH_COMMON_ROOT") or
                           Path(__file__).resolve().parents[2] / "data" / "common")

FAMILIES = ("longmemeval", "locomo", "lme_v2", "locomo_plus")
JUDGE_TYPES = {"llm_judge", "llm_abstention", "llm_gotchas"}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_questions(bench: str, qs_dir: Optional[Path] = None, full: bool = False) -> List[dict]:
    directory = Path(qs_dir) if qs_dir else DEFAULT_DATA_COMMON / bench
    if full and not qs_dir:
        directory /= "full"
    return read_questions(directory)[0]


def load_run(run_path: Path, variant: Optional[str] = None, *, raw_bytes: Optional[bytes] = None,
             dedup: str = "last") -> tuple[Dict[str, dict], int]:
    """Explicit variant/dedup selection; mixed variants require --variant."""
    return select_run_records(raw_bytes if raw_bytes is not None else run_path.read_bytes(), variant, dedup)


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------

Z_95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z_95):
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def aggregate(rows: List[dict]) -> dict:
    n = len(rows)
    scored = [r for r in rows if not r.get("flag")]
    n_flagged = n - len(scored)
    scores = [float(r["score"]) for r in rows]
    mean_score = sum(scores) / n if n else 0.0
    k = sum(1 for s in scores if s >= 0.5)
    lo, hi = wilson_ci(k, n)
    judge_lat = [r["judge_latency_ms"] for r in rows if r.get("judge_latency_ms") is not None]
    return {
        "n": n,
        "n_flagged": n_flagged,
        "mean_score": round(mean_score, 4),
        "acc_ge_0.5": round(k / n, 4) if n else 0.0,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
        "mean_judge_latency_ms": round(sum(judge_lat) / len(judge_lat), 1) if judge_lat else None,
    }


# ---------------------------------------------------------------------------
# per-question evaluation
# ---------------------------------------------------------------------------


def evaluate_question(q: dict, record: Optional[dict], *, judge_settings=None) -> dict:
    """Score one question. record is the driver output (None => missing)."""
    qid = q.get("qid")
    ev = q.get("eval") or {}
    etype = ev.get("type", "")
    params = ev.get("params") or {}
    out: Dict[str, Any] = {
        "qid": qid,
        "env_id": q.get("env_id"),
        "category": q.get("category"),
        "eval_type": etype,
        "score": 0.0,
        "flag": None,
        "question_sha256": canonical_hash(q),
        "run_record_sha256": canonical_hash(record) if record is not None else None,
        "scoring_protocol": ev.get("protocol"),
    }

    if record is None:
        out.update({"flag": "missing_record", "note": "no driver record for qid"})
        return out

    out["native_failure"] = native_failure(record)
    answer_text = record.get("answer_text")
    if isinstance(answer_text, str):
        answer_text = answer_text.strip()
    if record.get("error"):
        out.update({"flag": "driver_error", "note": "driver reported an error (details omitted)"})
        return out
    if not answer_text:
        out.update({"flag": "empty_answer", "note": "driver produced no answer text"})
        return out

    # family-level context
    family = q.get("family") or infer_family(q)
    gold = q.get("answer")
    if gold is not None and not isinstance(gold, str):
        gold = str(gold)

    # LongMemEval-V2 protocol: boxed extraction + UNKNOWN => 0 (all eval types)
    parsed_boxed = None
    is_unk = False
    if etype in ("phrase_set", "phrase_set_ordered", "mc_choice", "mc_choice_set",
                 "llm_abstention", "llm_gotchas") or family == "lme_v2":
        parsed_boxed = det.extract_boxed_answer(answer_text)
        is_unk = det.is_unknown(parsed_boxed)

    try:
        validate_question_protocol(q, family)
        if etype in det.DETERMINISTIC_TYPES:
            prediction = parsed_boxed if (family == "lme_v2" and parsed_boxed is not None) else answer_text
            res = det.score_deterministic(etype, params, prediction, gold, question=q)
            out.update({"score": float(res["score"]), "details": res["details"]})
        elif etype == "llm_judge" and family == "longmemeval":
            res = judges.judge_longmemeval(
                q.get("question"), gold, answer_text,
                params.get("prompt_variant", "single-session-user"),
                bool(params.get("abstention", False)),
            )
            out.update({k: v for k, v in res.items() if k != "judge_meta"})
            out["judge_latency_ms"] = res.get("judge_meta", {}).get("latency_ms")
        elif etype == "llm_judge" and family == "locomo_plus":
            res = judges.judge_locomo_plus(
                params.get("evidence_cue", ""), answer_text, question=q.get("question", ""),
                max_tokens=judge_settings["locomo_plus"]["max_tokens"] if judge_settings else None,
            )
            out.update({k: v for k, v in res.items() if k != "judge_meta"})
            out["judge_latency_ms"] = res.get("judge_meta", {}).get("latency_ms")
        elif etype == "llm_abstention":
            final = parsed_boxed if parsed_boxed is not None else answer_text
            res = judges.judge_lme_v2_abstention(
                q.get("question"), gold, answer_text, final
            )
            out.update({k: v for k, v in res.items() if k != "judge_meta"})
            out["judge_latency_ms"] = res.get("judge_meta", {}).get("latency_ms")
        elif etype == "llm_gotchas":
            final = parsed_boxed if parsed_boxed is not None else answer_text
            res = judges.judge_lme_v2_gotchas(
                q.get("question"), gold, answer_text, final
            )
            out.update({k: v for k, v in res.items() if k != "judge_meta"})
            out["judge_latency_ms"] = res.get("judge_meta", {}).get("latency_ms")
        else:
            out.update({"flag": "unknown_eval_type", "note": f"eval type {etype!r} not handled"})
            return out
    except Exception as e:  # judge/parse failures must not abort the run
        out.update({"flag": "eval_error", "note": f"{type(e).__name__}: evaluation failed (details omitted)"})
        return out

    # driver timing context (not a score input)
    if record.get("latency_ms") is not None:
        out["driver_latency_ms"] = record["latency_ms"]

    if family == "lme_v2":
        out["is_unknown"] = is_unk
        out["parsed_boxed"] = parsed_boxed
        if is_unk:
            out["score"] = 0.0
            out["unknown_forced_zero"] = True
    return out


def infer_family(q: dict) -> str:
    """Infer the benchmark family from question shape (eval params carry hints)."""
    ev = q.get("eval") or {}
    p = ev.get("params") or {}
    if p.get("prompt_variant") is not None or q.get("question_date") is not None:
        return "longmemeval"
    if p.get("evidence_cue") is not None or q.get("relation_type") is not None:
        return "locomo_plus"
    if q.get("raw_eval_function") is not None or q.get("domain") is not None:
        return "lme_v2"
    if q.get("presented_question") is not None or ev.get("type") in ("f1", "abstain_f1"):
        return "locomo"
    return "unknown"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(args) -> dict:
    run_path = Path(args.run)
    if not run_path.is_file():
        raise FileNotFoundError(f"run file not found: {run_path}")
    qdir = Path(args.qs_dir) if args.qs_dir else DEFAULT_DATA_COMMON / args.bench
    if args.full and not args.qs_dir:
        qdir = qdir / "full"
    inputs = [run_path, *qdir.glob("qs_*.jsonl")]
    config = getattr(args, "models_json", None) or os.environ.get("MEMORY_BENCH_MODELS_JSON")
    if config:
        inputs.append(Path(config))
    out_path = check_output_path(args.out, inputs)
    all_questions, question_files = read_questions(qdir)
    for question in all_questions:
        validate_question_protocol(question, args.bench)
    questions = all_questions[:args.limit] if args.limit else all_questions
    raw_run = run_path.read_bytes()
    dedup = getattr(args, "dedup", "last")
    records, dupes = load_run(run_path, variant=args.variant, raw_bytes=raw_run, dedup=dedup)
    for question in questions:
        record = records.get(question["qid"])
        validate_driver_question(record, question, args.bench)
    identity = evaluator_identity()

    t0 = time.time()
    results: List[dict] = []

    def work(q):
        return evaluate_question(q, records.get(q.get("qid")), judge_settings=identity["judge_settings"])

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, q) for q in questions]
            for fut in as_completed(futs):
                results.append(fut.result())
        # restore question order
        by_qid = {r["qid"]: r for r in results}
        results = [by_qid[q["qid"]] for q in questions]
    else:
        results = [work(q) for q in questions]

    # aggregates
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_cat[str(r.get("category"))].append(r)

    categories = {
        cat: aggregate(rows)
        for cat, rows in sorted(by_cat.items())
    }

    run_meta = {}
    if records:
        sample = next(iter(records.values()))
        run_meta = {
            "run_id": sample.get("run_id"),
            "variant": sample.get("variant"),
            "benchmark": sample.get("benchmark", args.bench),
        }

    unknown_stats = None
    if any(r.get("is_unknown") is not None for r in results):
        n = len(results)
        n_unk = sum(1 for r in results if r.get("is_unknown"))
        correct = sum(1 for r in results if r.get("score", 0) >= 0.5 and not r.get("is_unknown"))
        unknown_stats = {
            "count": n_unk,
            "pct_unknown": round(n_unk / n, 4) if n else 0.0,
            "pct_correct_excl_unknown": round(correct / n, 4) if n else 0.0,
        }

    client = judges.get_client() if any(r.get("eval_type") in JUDGE_TYPES for r in results) else None
    scores = {
        "schema": "memory-bench-scores/1",
        "bench": args.bench,
        "run_file": str(run_path),
        "run_sha256": hashlib.sha256(raw_run).hexdigest(),
        "run": run_meta,
        "scoring_provenance": make_provenance(all_questions, question_files, questions, records,
                                               args.variant, dedup, identity),
        "overall": aggregate(results),
        "by_category": categories,
        "counts": {
            "questions_total": len(questions),
            "records": len(records),
            "duplicate_qids_in_run": dupes,
            "questions_without_record": sum(1 for q in questions if q.get("qid") not in records),
            "flagged": sum(1 for r in results if r.get("flag")),
        },
        "judge_stats": dict(client.stats) if client else None,
        "lme_v2_unknown": unknown_stats,
        "eval_wall_s": round(time.time() - t0, 1),
        "results": results,
    }
    atomic_write_text(out_path, json.dumps(scores, ensure_ascii=False, indent=1))

    # console summary
    print(f"[{args.bench}] {len(results)} questions scored -> {out_path}")
    ov = scores["overall"]
    print(f"  overall: mean={ov['mean_score']} acc@0.5={ov['acc_ge_0.5']} "
          f"wilson95=[{ov['wilson95_lo']}, {ov['wilson95_hi']}] flagged={ov['n_flagged']}")
    for cat, agg in categories.items():
        print(f"  {cat}: n={agg['n']} mean={agg['mean_score']} acc@0.5={agg['acc_ge_0.5']}")
    if client:
        print(f"  judge stats: {client.stats}")
    return scores


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="driver output run.jsonl")
    ap.add_argument("--bench", required=True, choices=list(FAMILIES))
    ap.add_argument("--out", required=True, help="output scores.json path")
    ap.add_argument("--qs-dir", default=None, help="override questions directory")
    ap.add_argument("--full", action="store_true", help="use <family>/full question sets")
    ap.add_argument("--workers", type=int, default=1, help="parallel judge workers (default 1)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N questions (0=all)")
    ap.add_argument("--models-json", default=None,
                    help="explicit DGX config (or MEMORY_BENCH_MODELS_JSON); no profile fallback")
    ap.add_argument("--judge-cache", default=None,
                    help="cache directory (or MEMORY_BENCH_JUDGE_CACHE); otherwise no disk cache")
    ap.add_argument("--allow-network", action="store_true",
                    help="opt in to authorized DGX GLM-5.3 calls on cache misses")
    ap.add_argument("--variant", default=None, help="only score records with this variant field (base/nomem)")
    ap.add_argument("--dedup", choices=["error", "first", "last"], default="last",
                    help="duplicate run qid policy (default last, recorded in provenance)")
    args = ap.parse_args(argv)
    if args.workers < 1 or args.limit < 0:
        ap.error("--workers must be positive and --limit must be nonnegative")
    try:
        # Config validation is explicit and never implies permission to call a model.
        judges.get_client(models_json=args.models_json, cache_dir=args.judge_cache,
                          allow_network=args.allow_network)
        run(args)
    except (OSError, ValueError) as exc:
        ap.exit(2, f"evaluation failed: {type(exc).__name__} (check explicit input/config paths)\n")


if __name__ == "__main__":
    main()
