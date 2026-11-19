#!/usr/bin/env python3
"""Fresh-process external wall driver and strict R4 lane report reader."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

LANE_SCHEMA = "compaction-r4-lane-v1"
WALL_SCHEMA = "compaction-r4-external-wall-v1"
LANES = {"wall", "allocation", "replay-phase", "minimal-provider"}
HEX64 = set("0123456789abcdef")


class Reject(ValueError):
    pass


def reject(stage: str, detail: str) -> None:
    raise Reject(f"[{stage}] {detail}")


def exact_object(value: Any, keys: set[str], stage: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        reject(stage, "object required")
    actual = set(value)
    if actual != keys:
        reject(stage, f"field set differs: missing={sorted(keys-actual)} extra={sorted(actual-keys)}")
    return value


def integer(value: Any, stage: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        reject(stage, "strict integer required; bool/float/string/null rejected")
    if value < minimum:
        reject(stage, f"integer below {minimum}")
    return value


def number(value: Any, stage: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        reject(stage, "finite numeric value required; bool/string/null rejected")
    result = float(value)
    if not math.isfinite(result):
        reject(stage, "NaN and infinity rejected")
    if positive and result <= 0 or not positive and result < 0:
        reject(stage, "numeric value outside admitted range")
    return result


def string(value: Any, stage: str) -> str:
    if not isinstance(value, str) or not value:
        reject(stage, "nonempty string required")
    return value


def digest(value: Any, stage: str) -> str:
    text = string(value, stage)
    if len(text) != 64 or set(text) - HEX64:
        reject(stage, "lowercase SHA-256 required")
    return text


def no_duration_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_ms") or key == "duration_ms":
                reject("wall-lane-exclusivity", f"internal duration {path}.{key} leaked into wall child")
            no_duration_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            no_duration_keys(child, f"{path}[{index}]")


def validate_continuation(value: Any, outcome: str) -> None:
    if not isinstance(value, dict):
        reject("continuation", "object required")
    status = value.get("status")
    if status == "available":
        if outcome != "success":
            reject("continuation-accounting", "committed continuation requires success")
        exact_object(value, {"status", "boundary", "message_count", "context_sha256", "compaction_entries"}, "continuation")
        if value["boundary"] != "real append_compaction followed by build_session_context(None)":
            reject("continuation-boundary", "real continuation boundary label changed")
        integer(value["message_count"], "continuation-message-count", 1)
        digest(value["context_sha256"], "continuation-digest")
        if integer(value["compaction_entries"], "continuation-count") != 1:
            reject("continuation-accounting", "exactly one compaction entry required")
    elif status == "unavailable":
        if outcome == "success":
            reject("continuation-accounting", "successful compaction requires continuation")
        exact_object(value, {"status", "message_count", "context_sha256", "reason"}, "continuation")
        if value["message_count"] is not None or value["context_sha256"] is not None:
            reject("continuation-unavailable", "unavailable continuation values must be null")
        string(value["reason"], "continuation-reason")
    else:
        reject("continuation-status", "available or unavailable required")


def validate_provider(value: Any, outcome: str) -> None:
    exact_object(value, {"schema", "kind", "wait", "generation", "call_count", "injection"}, "provider")
    if value["schema"] != "r4-minimal-provider-v1" or value["kind"] != "deterministic-in-process-mock":
        reject("provider-identity", "wrong mock schema/kind")
    wait = exact_object(value["wait"], {"requested_ms", "source", "observed_duration_ms", "unavailable_reason"}, "provider-wait")
    integer(wait["requested_ms"], "provider-wait-value")
    if wait["source"] != "controlled CLI input to tokio::time::sleep":
        reject("provider-wait-source", "wait must be a controlled input")
    if wait["observed_duration_ms"] is not None:
        reject("provider-wait-estimate", "observed provider wait is unavailable and must be null")
    string(wait["unavailable_reason"], "provider-wait-reason")
    generation = exact_object(value["generation"], {"summary_render_iterations", "source", "output_text", "generated_units_are_not_duration"}, "provider-generation")
    integer(generation["summary_render_iterations"], "provider-generation-units", 1)
    if generation["source"] != "controlled CLI input" or generation["generated_units_are_not_duration"] is not True:
        reject("provider-generation-source", "controlled non-duration generation units required")
    string(generation["output_text"], "provider-generation-output")
    calls = integer(value["call_count"], "provider-call-count")
    if outcome == "success" and calls < 1:
        reject("provider-call-accounting", "success requires a mock call")
    if outcome == "cancelled":
        if calls != 0:
            reject("provider-cancellation", "pre-provider cancellation must have zero calls")
    elif calls != 1:
        reject("provider-call-accounting", "exactly one call required")
    if value["injection"] != {"success": "none", "failure": "failure", "cancelled": "cancelled"}[outcome]:
        reject("provider-injection", "injection and outcome differ")


def validate_phases(value: Any, lane: str, outcome: str) -> None:
    exact_object(value, {"schema", "phases", "provider_time_policy"}, "local-replay")
    if value["schema"] != "r4-local-phase-v1":
        reject("phase-schema", "wrong local phase schema")
    if "no provider wait or generation value is subtracted" not in value["provider_time_policy"]:
        reject("provider-local-separation", "provider exclusion policy missing")
    phases = value["phases"]
    if not isinstance(phases, list):
        reject("phase-list", "array required")
    if outcome == "cancelled":
        if phases != []:
            reject("cancellation-phase-receipt", "pre-provider cancellation has no local compact phases")
        return
    minimum = {"prepare_compaction", "compact_pre_provider", "compact_post_provider", "append_compaction", "build_continuation"}
    if lane == "minimal-provider":
        minimum |= {"convert_to_llm", "serialize_conversation", "summary_dispatch_preparation", "response_parse"}
    if outcome == "failure":
        minimum -= {"append_compaction", "build_continuation"}
    names: list[str] = []
    for phase in phases:
        exact_object(phase, {"name", "start_mark", "end_mark", "duration_ms", "clock", "boundary", "overhead", "cpu_scope"}, "phase-row")
        name = string(phase["name"], "phase-name")
        names.append(name)
        string(phase["start_mark"], "phase-start")
        string(phase["end_mark"], "phase-end")
        number(phase["duration_ms"], "phase-duration")
        if phase["clock"] != "std::time::Instant":
            reject("phase-clock", "monotonic clock identity required")
        string(phase["boundary"], "phase-boundary")
        string(phase["overhead"], "phase-overhead")
        if phase["cpu_scope"] != "local-only; interval contains no mock wait or mock generation interval":
            reject("provider-local-separation", "wrong CPU scope")
    if len(names) != len(set(names)) or set(names) != minimum:
        reject("phase-coverage", f"expected {sorted(minimum)}, got {sorted(names)}")


def validate_allocation(value: Any) -> None:
    status = value.get("status") if isinstance(value, dict) else None
    if status == "available":
        exact_object(value, {"status", "source", "scope", "count", "bytes_requested", "note"}, "allocation")
        string(value["source"], "allocation-source")
        string(value["scope"], "allocation-scope")
        integer(value["count"], "allocation-count", 1)
        integer(value["bytes_requested"], "allocation-bytes", 1)
        string(value["note"], "allocation-note")
    elif status == "unavailable":
        exact_object(value, {"status", "source", "count", "bytes_requested", "reason"}, "allocation")
        if value["source"] is not None or value["count"] is not None or value["bytes_requested"] is not None:
            reject("allocation-unavailable", "unavailable counters must be null, never zero or estimated")
        string(value["reason"], "allocation-reason")
    else:
        reject("allocation-status", "available or unavailable required")


def validate_lane_report(report: Any, expected_lane: str | None = None, expected_build: str | None = None) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema") != LANE_SCHEMA:
        reject("schema", f"{LANE_SCHEMA} object required")
    lane = report.get("lane")
    if lane not in LANES:
        reject("lane", "unknown lane")
    if expected_lane is not None and lane != expected_lane:
        reject("lane-identity", f"expected {expected_lane}, got {lane}")
    if lane == "wall":
        no_duration_keys(report)
    common = {"schema", "lane", "build", "run", "fixture", "provider_network", "synthetic_durations", "outcome", "continuation"}
    if lane in {"replay-phase", "minimal-provider"}:
        common |= {"local_replay", "provider"}
    if lane == "allocation":
        common.add("allocation")
    exact_object(report, common, "report-fields")
    wall = lane == "wall"
    build = exact_object(report["build"], {"configuration", "feature", "profile", "instrumentation"}, "build")
    configuration = "wall-uninstrumented" if wall else "instrumented"
    if build["configuration"] != configuration:
        reject("build-lane-identity", f"lane {lane} requires {configuration}")
    if expected_build is not None and configuration != expected_build:
        reject("build-identity", f"expected {expected_build}, got {configuration}")
    if build["feature"] != ("compaction-r4-uninstrumented-wall" if wall else None):
        reject("build-feature", "wrong feature identity")
    if build["profile"] not in {"release", "debug"}:
        reject("build-profile", "release or debug required")
    expected_instrumentation = [] if wall else ["phase-marks", "counting-global-allocator"]
    if build["instrumentation"] != expected_instrumentation:
        reject("instrumentation-identity", "instrumentation does not match build")
    run = exact_object(report["run"], {"pid", "exe_sha256", "process_state"}, "run")
    integer(run["pid"], "run-pid", 1)
    digest(run["exe_sha256"], "run-exe")
    if run["process_state"] != "fresh-process":
        reject("run-process-state", "fresh-process required")
    fixture = exact_object(report["fixture"], {"name", "record_count", "transcript_sha256"}, "fixture")
    string(fixture["name"], "fixture-name")
    integer(fixture["record_count"], "fixture-records", 1)
    digest(fixture["transcript_sha256"], "fixture-digest")
    if report["provider_network"] != "none; registered in-process deterministic mock only" or report["synthetic_durations"] != "non-decision":
        reject("report-identity", "mock/non-decision labels changed")
    outcome = exact_object(report["outcome"], {"status", "error"}, "outcome")
    status = outcome["status"]
    if status not in {"success", "failure", "cancelled"}:
        reject("outcome", "success/failure/cancelled required")
    if status == "success" and outcome["error"] is not None:
        reject("outcome-error", "success error must be null")
    if status != "success":
        string(outcome["error"], "outcome-error")
    validate_continuation(report["continuation"], status)
    if lane in {"replay-phase", "minimal-provider"}:
        validate_phases(report["local_replay"], lane, status)
        validate_provider(report["provider"], status)
    if lane == "allocation":
        validate_allocation(report["allocation"])
    return report


def validate_external_wall(report: Any) -> dict[str, Any]:
    exact_object(report, {"schema", "lane", "observation", "external_wall", "child", "child_report", "synthetic_durations"}, "external-wall")
    if report["schema"] != WALL_SCHEMA or report["lane"] != "wall" or report["synthetic_durations"] != "non-decision":
        reject("external-wall-identity", "wrong schema/lane/decision label")
    observation = exact_object(report["observation"], {"launch_state", "page_cache_state", "thermal_state"}, "external-observation")
    if observation != {"launch_state": "fresh-process", "page_cache_state": "unobserved", "thermal_state": "unobserved"}:
        reject("external-observation", "only observed launch state and explicit unknown page/thermal state admitted")
    wall = exact_object(report["external_wall"], {"duration_ms", "clock", "meaning"}, "external-wall-duration")
    number(wall["duration_ms"], "external-wall-duration-ms", positive=True)
    if wall["clock"] != "time.perf_counter_ns" or "spawn-to-terminal" not in wall["meaning"]:
        reject("external-wall-clock", "external monotonic spawn-to-terminal identity required")
    child = exact_object(report["child"], {"exit_code", "pid", "report_sha256", "report_path"}, "external-child")
    if integer(child["exit_code"], "child-exit") != 0:
        reject("child-exit", "successful child required")
    child_pid = integer(child["pid"], "child-pid", 1)
    digest(child["report_sha256"], "child-report-digest")
    string(child["report_path"], "child-report-path")
    inner = validate_lane_report(report["child_report"], "wall", "wall-uninstrumented")
    if inner["run"]["pid"] != child_pid:
        reject("child-pid-identity", "Popen and child report PID differ")
    return report


def load_report(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except Exception as error:
        reject("malformed-json", f"{path}: {error}")


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)


def run_wall(args: argparse.Namespace) -> int:
    child_out = args.child_out.resolve()
    out = args.out.resolve()
    command = [str(args.exe.resolve()), "r4-lane", "--lane", "wall", "--fixture", str(args.fixture.resolve()), "--work", str(args.work.resolve()), "--out", str(child_out)]
    start = time.perf_counter_ns()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(timeout=args.timeout_secs)
    duration_ms = (time.perf_counter_ns() - start) / 1_000_000
    if process.returncode != 0:
        reject("wall-child", f"exit {process.returncode}; stdout={stdout[:1000]!r}; stderr={stderr[:4000]!r}")
    child_bytes = child_out.read_bytes()
    child_report = validate_lane_report(load_report(child_out), "wall", "wall-uninstrumented")
    if child_report["run"]["pid"] != process.pid:
        reject("child-pid-identity", "Popen PID differs from reported PID")
    report = {
        "schema": WALL_SCHEMA,
        "lane": "wall",
        "observation": {"launch_state": "fresh-process", "page_cache_state": "unobserved", "thermal_state": "unobserved"},
        "external_wall": {"duration_ms": duration_ms, "clock": "time.perf_counter_ns", "meaning": "whole fresh child process spawn-to-terminal; excludes no startup or teardown and contains no internal phase meaning"},
        "child": {"exit_code": process.returncode, "pid": process.pid, "report_sha256": hashlib.sha256(child_bytes).hexdigest(), "report_path": str(child_out)},
        "child_report": child_report,
        "synthetic_durations": "non-decision",
    }
    validate_external_wall(report)
    write_new(out, report)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    wall = commands.add_parser("wall")
    wall.add_argument("--exe", type=Path, required=True)
    wall.add_argument("--fixture", type=Path, required=True)
    wall.add_argument("--work", type=Path, required=True)
    wall.add_argument("--out", type=Path, required=True)
    wall.add_argument("--child-out", type=Path, required=True)
    wall.add_argument("--timeout-secs", type=int, default=900)
    validate = commands.add_parser("validate")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--lane", choices=sorted(LANES))
    validate.add_argument("--build-configuration", choices=["instrumented", "wall-uninstrumented"])
    validate.add_argument("--external-wall", action="store_true")
    validate.add_argument("--out", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "wall":
        return run_wall(args)
    report = load_report(args.report)
    if args.external_wall:
        validate_external_wall(report)
    else:
        validate_lane_report(report, args.lane, args.build_configuration)
    write_new(args.out, {"schema": "r4-validation-receipt-v1", "status": "accepted", "report": str(args.report.resolve()), "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest()})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Reject as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
