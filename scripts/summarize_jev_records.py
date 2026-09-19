#!/usr/bin/env python3
"""Summarize bounded Jev JSONL metadata without printing prompt or result content."""

import argparse
import json
from collections import Counter
from pathlib import Path

SCHEMAS = {"jev.compare/1", "jev.active/1", "jev.compaction/1", "jev.run/1"}
MODES = {"off", "compare", "active", "compare-active"}
MAX_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_RECORDS = 200_000


def number(value):
    return value if type(value) in (int, float) and 0 <= value <= 2**53 else None


def action_number(value):
    if isinstance(value, str) and value.isascii() and value.isdigit() and len(value) <= 16:
        return number(int(value))
    return None


def summarize(records):
    groups = {}
    ignored = 0
    for row in records:
        if (not isinstance(row, dict) or not isinstance(row.get("schema_version"), str)
                or row["schema_version"] not in SCHEMAS):
            ignored += 1
            continue
        mode = row.get("mode")
        mode = "compare-active" if mode == "compare_and_active" else mode
        mode = mode if isinstance(mode, str) and mode in MODES else "unknown"
        enabled = row.get("compaction_enabled")
        compact = "on" if enabled is True else "off" if enabled is False else "unknown"
        key = (mode, compact)
        group = groups.setdefault(key, {
            "counts": Counter(), "requests": {}, "questions": set(), "question_counts": {},
            "comparison": set(), "active": set(), "compaction": {}, "runs": {},
            "retrieval": {}, "confidences": {},
        })
        group["counts"]["records"] += 1
        request_id = row.get("request_id")
        session_id = row.get("session_id")
        if not isinstance(request_id, str) or not isinstance(session_id, str):
            ignored += 1
            continue
        request_key = (session_id, request_id)
        schema = row["schema_version"]
        metrics = row.get("observed_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        attempts = number(row.get("attempt", row.get("attempts")))
        if schema != "jev.run/1" and attempts is not None and attempts > 0:
            request = group["requests"].setdefault(request_key, {
                "attempts": 0, "attempt_count_known": True, "duration_ms": None,
                "jev_input_tokens": None, "jev_output_tokens": None,
            })
            for token_field in ("jev_input_tokens", "jev_output_tokens"):
                tokens = number(metrics.get(token_field))
                if tokens is not None:
                    request[token_field] = max(request[token_field] or 0, tokens)
            request["attempts"] = max(request["attempts"], attempts)
            request["attempt_count_known"] &= row.get("attempt_count_known") is not False
            duration = number(row.get("duration_ms"))
            if duration is not None:
                request["duration_ms"] = max(request["duration_ms"] or 0, duration)
        question_id = row.get("question_id")
        question_key = (*request_key, question_id) if isinstance(question_id, str) else None
        if schema in {"jev.compare/1", "jev.active/1"} and question_key and attempts is not None and attempts > 0:
            group["questions"].add(question_key)
        if schema == "jev.compaction/1" and attempts is not None and attempts > 0:
            count = number(metrics.get("question_count"))
            if count is not None:
                group["question_counts"][request_key] = max(group["question_counts"].get(request_key, 0), count)
        if schema == "jev.compare/1" and question_key and question_key not in group["comparison"]:
            group["comparison"].add(question_key)
            group["counts"]["comparison_questions"] += 1
            agreement = row.get("agreement")
            if isinstance(agreement, str) and agreement in {"agree", "disagree", "noncomparable"}:
                group["counts"][f"comparison_{agreement}"] += 1
        if schema == "jev.active/1" and question_key and question_key not in group["active"]:
            group["active"].add(question_key)
            counts = group["counts"]
            counts["active_questions"] += 1
            counts["recommendations"] += row.get("selected_value") is not None
            counts["accepted"] += row.get("acceptance") in ("accepted", "accepted_no_effect")
            counts["applied"] += row.get("applied") is True
            counts["refused"] += row.get("outcome") == "refused"
            counts["fallbacks"] += row.get("fallback_reason") is not None
        confidence = number(row.get("confidence"))
        if question_key and confidence is not None and confidence <= 1:
            group["confidences"][question_key] = confidence
        category = row.get("category")
        if isinstance(category, str) and category in {"context_relevance", "memory_relevance"}:
            baseline = row.get("baseline_action")
            actual = row.get("actual_action") if schema == "jev.active/1" else baseline
            if isinstance(actual, dict) and "candidate_count" in actual:
                key = (*request_key, category)
                if key not in group["retrieval"] or schema == "jev.active/1":
                    group["retrieval"][key] = actual
        if schema == "jev.compaction/1":
            # Batch-only rows carry transport metadata, not a second history measurement.
            if "chars_before" in metrics or "chars_before" in row:
                group["compaction"][request_key] = {**row, **metrics}
        if schema == "jev.run/1":
            group["runs"][request_key] = {**row, **metrics}

    output = []
    for (mode, compact), group in sorted(groups.items()):
        durations = [r["duration_ms"] for r in group["requests"].values() if r["duration_ms"] is not None]
        item = {
            "mode": mode, "compaction": compact,
            "counts": dict(sorted(group["counts"].items())),
            "jev_requests": len(group["requests"]),
            "jev_transport_attempts": (
                sum(r["attempts"] for r in group["requests"].values())
                if all(r["attempt_count_known"] for r in group["requests"].values()) else None
            ),
            "jev_transport_attempts_lower_bound": sum(r["attempts"] for r in group["requests"].values()),
            "jev_unique_questions": len(group["questions"]) + sum(group["question_counts"].values()),
            "jev_latency_ms": {
                "samples": len(durations),
                "mean": sum(durations) / len(durations) if durations else None,
                "max": max(durations) if durations else None,
            },
            "jev_usage_tokens": {}, "compaction_totals": {}, "run_totals": {}, "retrieval_totals": {},
            "raw_confidence_distribution": {
                "below_0.5": sum(value < 0.5 for value in group["confidences"].values()),
                "0.5_to_below_0.9": sum(0.5 <= value < 0.9 for value in group["confidences"].values()),
                "0.9_to_1": sum(value >= 0.9 for value in group["confidences"].values()),
            },
        }
        for field in ("jev_input_tokens", "jev_output_tokens"):
            values = [request[field] for request in group["requests"].values()]
            item["jev_usage_tokens"][field] = sum(values) if values and all(value is not None for value in values) else None
        for destination, source, fields in [
            ("compaction_totals", "compaction", (
                "messages_before", "messages_after", "chars_before", "chars_after",
                "estimated_tokens_before", "estimated_tokens_after", "calls_evaluated",
                "calls_removed", "results_removed", "results_truncated",
            )),
            ("run_totals", "runs", (
                "agent_turns", "logical_primary_calls", "primary_llm_calls", "model_input_tokens",
                "model_output_tokens", "model_cache_read_tokens", "model_cache_write_tokens", "elapsed_ms",
            )),
        ]:
            rows = list(group[source].values())
            item[destination]["samples"] = len(rows)
            for field in fields:
                values = [number(row.get(field)) for row in rows]
                # Partial or unavailable measurements must not look like complete totals.
                item[destination][field] = sum(values) if values and all(v is not None for v in values) else None
        for category in ("context_relevance", "memory_relevance"):
            rows = [row for key, row in group["retrieval"].items() if key[-1] == category]
            totals = {"samples": len(rows)}
            for field in ("candidate_count", "retained_count", "removed_count",
                          "estimated_candidate_tokens_before", "estimated_candidate_tokens_after"):
                values = [action_number(row.get(field)) for row in rows]
                totals[field] = sum(values) if values and all(value is not None for value in values) else None
            item["retrieval_totals"][category] = totals
        before = item["compaction_totals"].get("chars_before")
        after = item["compaction_totals"].get("chars_after")
        item["compaction_totals"]["reduction_ratio"] = (
            (before - after) / before if before and after is not None else None
        )
        output.append(item)
    return {
        "schema_version": "jev.summary/1", "groups": output, "ignored_records": ignored,
        "note": "Request IDs deduplicate shared Compare+Active calls. Unknown measurements are null. Savings and task quality require matched external evaluations.",
    }


def read_records(path):
    records = []
    malformed = 0
    scanned = 0
    truncated = False
    with Path(path).open("rb") as stream:
        while len(records) + malformed < MAX_RECORDS:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            scanned += len(raw)
            if scanned > MAX_BYTES or len(raw) > MAX_LINE_BYTES:
                truncated = True
                break
            try:
                records.append(json.loads(raw))
            except (ValueError, UnicodeDecodeError):
                malformed += 1
        else:
            truncated = True
    return records, {"malformed_lines": malformed, "input_truncated": truncated, "bytes_scanned": scanned}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="Path to Jev records.jsonl")
    parser.add_argument("--runs", type=Path, help="Optional path to metadata-only Jev runs.jsonl")
    args = parser.parse_args()
    records, input_status = read_records(args.records)
    run_status = None
    if args.runs:
        runs, run_status = read_records(args.runs)
        records.extend(runs)
    report = summarize(records)
    report["input"] = input_status
    if run_status is not None:
        report["run_input"] = run_status
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
