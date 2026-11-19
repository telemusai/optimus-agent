"""Deterministic non-timing R4 report fixtures for rejection controls."""
from __future__ import annotations
import copy

HASH_A = "a" * 64
HASH_B = "b" * 64
SUMMARY = "synthetic mock summary"


def phase(name: str):
    return {
        "name": name,
        "start_mark": name + "_start",
        "end_mark": name + "_end",
        "duration_ms": 1.25,
        "clock": "std::time::Instant",
        "boundary": "synthetic non-timing boundary label",
        "overhead": "documented synthetic overhead",
        "cpu_scope": "local-only; interval contains no mock wait or mock generation interval",
    }


def continuation(status: str = "available"):
    if status == "available":
        return {
            "status": "available",
            "boundary": "real append_compaction followed by build_session_context(None)",
            "message_count": 7,
            "context_sha256": HASH_B,
            "compaction_entries": 1,
        }
    return {
        "status": "unavailable",
        "message_count": None,
        "context_sha256": None,
        "reason": "compaction did not complete; no continuation was committed",
    }


def base(lane: str, outcome: str = "success"):
    wall = lane == "wall"
    return {
        "schema": "compaction-r4-lane-v1",
        "lane": lane,
        "build": {
            "configuration": "wall-uninstrumented" if wall else "instrumented",
            "feature": "compaction-r4-uninstrumented-wall" if wall else None,
            "profile": "release",
            "instrumentation": [] if wall else ["phase-marks", "counting-global-allocator"],
        },
        "run": {"pid": 1234, "exe_sha256": HASH_A, "process_state": "fresh-process"},
        "fixture": {"name": "synthetic", "record_count": 10, "transcript_sha256": HASH_B},
        "provider_network": "none; registered in-process deterministic mock only",
        "synthetic_durations": "non-decision",
        "outcome": {"status": outcome, "error": None if outcome == "success" else "synthetic injected outcome"},
        "continuation": continuation("available" if outcome == "success" else "unavailable"),
    }


def provider(outcome: str = "success"):
    return {
        "schema": "r4-minimal-provider-v1",
        "kind": "deterministic-in-process-mock",
        "wait": {
            "requested_ms": 25,
            "source": "controlled CLI input to tokio::time::sleep",
            "observed_duration_ms": None,
            "unavailable_reason": "controlled wait only; no estimate",
        },
        "generation": {
            "summary_render_iterations": 3,
            "source": "controlled CLI input",
            "output_text": SUMMARY,
            "generated_units_are_not_duration": True,
        },
        "call_count": 0 if outcome == "cancelled" else 1,
        "injection": {"success": "none", "failure": "failure", "cancelled": "cancelled"}[outcome],
    }


def phases(lane: str, outcome: str):
    if outcome == "cancelled":
        return []
    names = ["prepare_compaction", "compact_pre_provider", "compact_post_provider"]
    if lane == "minimal-provider":
        names.extend(["convert_to_llm", "serialize_conversation", "summary_dispatch_preparation", "response_parse"])
    if outcome == "success":
        names.extend(["append_compaction", "build_continuation"])
    return [phase(name) for name in names]


def lane_report(lane: str, outcome: str = "success"):
    report = base(lane, outcome)
    if lane in {"replay-phase", "minimal-provider"}:
        report["local_replay"] = {
            "schema": "r4-local-phase-v1",
            "phases": phases(lane, outcome),
            "provider_time_policy": "no provider wait or generation value is subtracted, estimated or included in local phase arithmetic",
        }
        report["provider"] = provider(outcome)
    elif lane == "allocation":
        report["allocation"] = {
            "status": "available",
            "source": "synthetic source label",
            "scope": "synthetic scope",
            "count": 19,
            "bytes_requested": 4096,
            "note": "synthetic control only",
        }
    return report


def external_wall():
    child = lane_report("wall")
    return {
        "schema": "compaction-r4-external-wall-v1",
        "lane": "wall",
        "observation": {"launch_state": "fresh-process", "page_cache_state": "unobserved", "thermal_state": "unobserved"},
        "external_wall": {
            "duration_ms": 10.5,
            "clock": "time.perf_counter_ns",
            "meaning": "whole fresh child process spawn-to-terminal synthetic control",
        },
        "child": {"exit_code": 0, "pid": 1234, "report_sha256": HASH_B, "report_path": "synthetic.json"},
        "child_report": child,
        "synthetic_durations": "non-decision",
    }


def clone(value):
    return copy.deepcopy(value)
