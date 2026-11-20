#!/usr/bin/env python3
"""Fail-closed R5 compaction correctness and timing campaign driver.

Real Gate A and Gate B are separate, explicitly authorized operations.  The
stdlib unit controls inject process, telemetry and introspection adapters and
never invoke a compiled example or a real fixture.  This module does not build
or modify production code and never creates its own Gate B authorization.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from compaction_paired_protocol import query_process_image, set_child_affinity
from compaction_paired_stats import analyze as independent_paired_analyze
from compaction_r4_lanes import (
    Reject as R4Reject,
    load_report as load_r4_report,
    validate_external_wall,
    validate_lane_report,
)

SCHEMA = "compaction-r5-campaign-plan-v1"
GATE_A_SUMMARY_SCHEMA = "compaction-r5-gate-a-summary-v1"
GATE_A_REVIEW_SCHEMA = "compaction-r5-gate-a-independent-review-v1"
GATE_B_AUTH_SCHEMA = "compaction-r5-gate-b-authorization-v1"
R4_BASE = "b3408eadf6d354b70ceea1fafe08b15134c642a3"
R4_COMMIT = "b8ae3191a346f7387fc47389d6cd20ef38cdd58f"
R4_PATCH_SHA256 = "8cf79a8011f55b99d2c06042cc7e577cd0c3b5001181c4a1295bac113315e313"
R4_INSTRUMENTED_EXE_SHA256 = "f84a06cf2e0cf3abf0021163a52187895456ae56462bb81561d6e05c169c2541"
R4_WALL_EXE_SHA256 = "f871b210d1f8c0b013443905c7003eaeafbadb4e60a656cd3e2353b118edbaf6"
R3_REPLAY_EXE_SHA256 = "f4de1f158c741a5b7773ce77042d283a27998ee673a440e897364d4d5152cd0d"
R4_CHANGED_PATHS = (
    ".changes/compaction-replay-r4-lanes.md",
    "crates/pi-coding-agent/Cargo.toml",
    "crates/pi-coding-agent/examples/compaction_replay.rs",
    "crates/pi-coding-agent/examples/compaction_replay/r4_lanes.rs",
    "scripts/compaction-r4-lanes.md",
    "scripts/compaction_r4_lanes.py",
    "scripts/compaction_r4_lanes_mock.py",
    "scripts/test_compaction_r4_lanes.py",
)
R5_SOURCE_PATHS = (
    "scripts/compaction_baseline_campaign.py",
    "scripts/test_compaction_baseline_campaign.py",
    "scripts/compaction-baseline-campaign.md",
    ".changes/compaction-replay-r5-baseline-campaign.md",
)
ARTIFACT_ROOT = Path("C:/Users/openclawuser/Optimus-Assistant/profile/workspace/optimus-compaction-r5-baseline-artifacts")
TARGET_ROOT = Path("C:/Users/openclawuser/Optimus-Assistant/profile/workspace/optimus-compaction-r5-baseline-target")
TARGETS = ("instrumented", "wall", "r3-parity")
MASK = 0x555
EXPECTED_CONTROL_COUNTS = {
    "r1": (34, 3, 31),
    "r2": (24, 3, 21),
    "r3": (71, 4, 67),
    "r4": (72, 10, 62),
}
FULL_PROJECTION_ALGORITHM = "sha256 over compact serde_json encoding of the complete in-memory semantic section before report serialization"
FULL_PROJECTION_NOTE = "Complete values are retained in cases; these are full-content hashes, not length or scalar checksums"
PARITY_SECTIONS = (
    "case", "fixture_manifest", "token_outputs", "cut_output",
    "prepare_compaction", "conversion", "serialization",
    "replicated_chunk_slicing", "session_manager__raw_branch_semantic",
    "session_manager__normalized_restored_messages", "session_manager__context",
    "session_manager__checkpoint_state", "session_manager__protected_custom_state",
    "jev_prepare_context",
)
CASE_SUMMARY_CALLS = {
    "small-ascii": 1, "small-cjk": 1, "small-emoji": 1,
    "small-mixed-crlf": 1, "small-tool-huge": 1, "small-many-small": 1,
    "medium-ascii": 2, "medium-cjk": 1, "medium-emoji": 2,
    "medium-mixed-crlf": 2, "medium-tool-huge": 2, "medium-many-small": 2,
    "large-ascii": 4, "large-cjk": 3, "large-emoji": 3,
    "large-mixed-crlf": 3, "large-tool-huge": 2, "large-many-small": 4,
    "near-threshold-ascii": 2, "xlarge-ascii": 10, "xlarge-tool-huge": 3,
}
CASE_ORDER = (
    "small-ascii", "small-cjk", "small-emoji", "small-mixed-crlf",
    "small-tool-huge", "small-many-small", "medium-ascii", "medium-cjk",
    "medium-emoji", "medium-mixed-crlf", "medium-tool-huge",
    "medium-many-small", "large-ascii", "large-cjk", "large-emoji",
    "large-mixed-crlf", "large-tool-huge", "large-many-small",
    "near-threshold-ascii", "xlarge-ascii", "xlarge-tool-huge",
)
R3_ONE_SUMMARY_CALL_CASES = frozenset(name for name, count in CASE_SUMMARY_CALLS.items() if count == 1)
# Retained name for the R3 replay expectation only.  R4 timing does not infer
# eligibility from the number of successful R3 summary calls.
EXPECTED_ONE_CALL_CASES = R3_ONE_SUMMARY_CALL_CASES
R4_ONE_CALL_TIMING_CASES = (
    "small-ascii", "small-cjk", "small-mixed-crlf", "medium-cjk",
)
R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES = (
    "small-emoji", "small-tool-huge", "small-many-small",
)
R4_NONELIGIBILITY_REASON = "duplicate-summary-runner-instrumentation"
R4_DUPLICATE_SUMMARY_MARKER = "duplicate phase mark summary_runner_entry"
ONE_SHOT_LANES = ("wall", "allocation", "replay-phase", "minimal-provider")
WARM_CASES = (
    "small-ascii", "small-cjk", "small-emoji", "small-mixed-crlf",
    "small-tool-huge", "small-many-small", "medium-tool-huge",
    "large-tool-huge", "xlarge-tool-huge",
)
UNAVAILABLE_WARM_PREFLIGHTS = (
    ("medium-ascii", 880), ("medium-cjk", 728), ("medium-emoji", 896),
    ("medium-mixed-crlf", 534), ("medium-many-small", 1848),
    ("large-ascii", 880), ("large-cjk", 731), ("large-emoji", 938),
    ("large-mixed-crlf", 554), ("large-many-small", 1945),
    ("near-threshold-ascii", 1156), ("xlarge-ascii", 909),
)
JEV_INPUT_LIMIT_REASON = "compaction_input_limit"
JEV_INPUT_LIMIT_STDERR = "error: [jev-prepare] unexpected production Jev prepare_context error: compaction_input_limit"
WARM_KERNELS = (
    "estimate_tokens_all_messages",
    "estimate_context_tokens_full_scan_no_usable_usage",
    "estimate_context_tokens_with_usage_short_circuit",
    "find_cut_point", "prepare_compaction", "convert_to_llm",
    "serialize_conversation", "summary_chunk_slicing_fixture_window",
    "summary_chunk_slicing_small_window", "session_manager_build_session_context",
    "session_manager_get_branch",
)
TELEMETRY_FIELDS = (
    "cpu_total_percent", "processor_frequency", "processor_queue",
    "per_logical_processor", "available_memory_bytes", "active_power_scheme",
)
OBSERVATION_STATES = {
    "launch_state": "fresh-process",
    "page_cache_state": "unobserved",
    "thermal_state": "unobserved",
}
SAFE_INHERITED_ENV = (
    "PATH", "SystemRoot", "ComSpec", "TEMP", "TMP", "USERPROFILE",
    "LOCALAPPDATA", "PROGRAMDATA", "SYSTEMDRIVE", "HOME",
)
SECRET_NAME = re.compile(r"(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|API_?KEY)", re.I)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")


class Reject(ValueError):
    """An intended-stage fail-closed rejection."""

    def __init__(self, stage: str, detail: str):
        self.stage = stage
        super().__init__(f"[{stage}] {detail}")


def require(condition: Any, stage: str, detail: str) -> None:
    if not condition:
        raise Reject(stage, detail)


def exact_object(value: Any, keys: Iterable[str], stage: str) -> dict[str, Any]:
    require(isinstance(value, dict), stage, "object required")
    expected = set(keys)
    actual = set(value)
    require(actual == expected, stage, f"field mismatch missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    return value


def strict_int(value: Any, stage: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, stage, f"integer >= {minimum} required, got {value!r}")
    return value


def finite_positive(value: Any, stage: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), stage, f"number required, got {value!r}")
    result = float(value)
    require(math.isfinite(result) and result > 0.0, stage, f"finite positive number required, got {value!r}")
    return result


def finite_nonnegative(value: Any, stage: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), stage, f"number required, got {value!r}")
    result = float(value)
    require(math.isfinite(result) and result >= 0.0, stage, f"finite nonnegative number required, got {value!r}")
    return result


def nonempty_string(value: Any, stage: str) -> str:
    require(isinstance(value, str) and bool(value), stage, "nonempty string required")
    return value


def sha256_text(value: Any, stage: str) -> str:
    text = nonempty_string(value, stage)
    require(SHA256_RE.fullmatch(text) is not None, stage, "lowercase SHA-256 required")
    return text


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Reject("structured-hash", str(error)) from error


def structured_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise Reject("malformed-json", f"{path}: {error}") from error


def write_new_json(path: Path, value: Any) -> dict[str, Any]:
    body = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    return write_new_bytes(path, body)


def write_new_bytes(path: Path, body: bytes) -> dict[str, Any]:
    require(not path.exists(), "output-collision", f"refuse existing output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(body)
    return file_identity(path)


def file_identity(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        require(resolved.is_file(), "file-identity", f"not a regular file: {resolved}")
        data = resolved.read_bytes()
    except OSError as error:
        raise Reject("file-identity", f"{path}: {error}") from error
    return {"path": str(resolved), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def file_map_identities(root: Path, expected: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in expected.items():
        relative_path = Path(relative)
        require(".." not in relative_path.parts, "unsafe-path", relative)
        identity = file_identity(relative_path if relative_path.is_absolute() else root / relative_path)
        require(identity["sha256"] == expected_hash, "stale-or-frozen-hash", f"{relative} hash changed")
        result[relative] = identity
    return result


def exact_compare(expected: Any, actual: Any, pointer: str = "$") -> None:
    """Exact values, concrete JSON types, array order and finite IEEE754 values."""
    if expected is None or isinstance(expected, (str, bool, int)):
        require(type(actual) is type(expected) and actual == expected, "exact-value-parity", pointer)
        return
    if isinstance(expected, float):
        require(type(actual) is float and math.isfinite(expected) and math.isfinite(actual), "exact-ieee754-parity", pointer)
        require(actual.hex() == expected.hex(), "exact-ieee754-parity", pointer)
        return
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "exact-array-parity", pointer)
        for index, (left, right) in enumerate(zip(expected, actual)):
            exact_compare(left, right, f"{pointer}/{index}")
        return
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and list(actual) == list(expected), "exact-object-parity", pointer)
        for key in expected:
            exact_compare(expected[key], actual[key], f"{pointer}/{key}")
        return
    raise Reject("exact-value-parity", f"unsupported JSON value at {pointer}")


def control_signature(report: Any, family: str, expected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total, accepted, rejected = EXPECTED_CONTROL_COUNTS[family]
    require(isinstance(report, dict) and report.get("status") == "ok", f"{family}-control-status", "status ok required")
    require(type(report.get("check_count")) is int and report["check_count"] == total, f"{family}-control-count", f"exactly {total} checks required")
    checks = report.get("checks")
    require(isinstance(checks, list) and len(checks) == total, f"{family}-control-count", "checks array length differs from check_count")
    signature: list[dict[str, Any]] = []
    for index, row in enumerate(checks):
        require(isinstance(row, dict), f"{family}-control-row", str(index))
        name = nonempty_string(row.get("name"), f"{family}-control-name")
        is_accepted = row.get("accepted") is True
        is_rejected = row.get("rejected") is True
        require(is_accepted != is_rejected, f"{family}-control-outcome", name)
        stage = row.get("intended_stage") if is_rejected else None
        if is_rejected:
            stage = nonempty_string(stage, f"{family}-control-stage")
            error_text = row.get("error")
            if error_text is None and isinstance(row.get("receipt"), dict):
                error_text = row["receipt"].get("stderr")
            require(isinstance(error_text, str) and stage in error_text, f"{family}-control-stage", name)
        signature.append({"name": name, "outcome": "accepted" if is_accepted else "rejected", "intended_stage": stage})
    require(sum(row["outcome"] == "accepted" for row in signature) == accepted, f"{family}-accepted-count", str(accepted))
    require(sum(row["outcome"] == "rejected" for row in signature) == rejected, f"{family}-rejected-count", str(rejected))
    require(signature == list(expected), f"{family}-ordered-signature", "ordered name/outcome/stage signature differs from pre-registration")
    return signature


def signing_control_summary(reports: Mapping[str, Any], expected: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    require(set(reports) == set(EXPECTED_CONTROL_COUNTS) and set(expected) == set(EXPECTED_CONTROL_COUNTS), "control-family-coverage", "exact R1/R2/R3/R4 families required")
    signatures = {family: control_signature(reports[family], family, expected[family]) for family in EXPECTED_CONTROL_COUNTS}
    return {"schema": "compaction-r5-ordered-controls-v1", "status": "ok", "signatures": signatures}


FROZEN_SUCCESSFUL_REPLAY_SHA256 = "16b82c5c6b71c3ffe22bfb683c8fd7d3d5e5f14de6114818d2c2057f24a723ea"

FROZEN_SUCCESSFUL_RESTORED_IDENTITIES = {   'small-ascii': {   'pairing': {   'tool_calls': 0,
                                      'tool_results': 0,
                                      'matched_results': 0,
                                      'unmatched_results': 0,
                                      'calls_without_result': 0},
                       'retained_state_sha256': '43c3e636ec9a06b601923e8903a28d13d6921697fa43f14a67fb195a2a020a7e'},
    'small-cjk': {   'pairing': {   'tool_calls': 0,
                                    'tool_results': 0,
                                    'matched_results': 0,
                                    'unmatched_results': 0,
                                    'calls_without_result': 0},
                     'retained_state_sha256': '239569a099be01417e820e16b7796e2c5f6a611952e8bd123d933e1a4babaa52'},
    'small-emoji': {   'pairing': {   'tool_calls': 0,
                                      'tool_results': 0,
                                      'matched_results': 0,
                                      'unmatched_results': 0,
                                      'calls_without_result': 0},
                       'retained_state_sha256': '43c3e636ec9a06b601923e8903a28d13d6921697fa43f14a67fb195a2a020a7e'},
    'small-mixed-crlf': {   'pairing': {   'tool_calls': 0,
                                           'tool_results': 0,
                                           'matched_results': 0,
                                           'unmatched_results': 0,
                                           'calls_without_result': 0},
                            'retained_state_sha256': '239569a099be01417e820e16b7796e2c5f6a611952e8bd123d933e1a4babaa52'},
    'small-tool-huge': {   'pairing': {   'tool_calls': 0,
                                          'tool_results': 0,
                                          'matched_results': 0,
                                          'unmatched_results': 0,
                                          'calls_without_result': 0},
                           'retained_state_sha256': 'c365b0e1db313a0d8d73fb346d5c7dabd1de557e5274dcdcc7d318dd8651589f'},
    'small-many-small': {   'pairing': {   'tool_calls': 0,
                                           'tool_results': 0,
                                           'matched_results': 0,
                                           'unmatched_results': 0,
                                           'calls_without_result': 0},
                            'retained_state_sha256': '43c3e636ec9a06b601923e8903a28d13d6921697fa43f14a67fb195a2a020a7e'},
    'medium-ascii': {   'pairing': {   'tool_calls': 21,
                                       'tool_results': 21,
                                       'matched_results': 21,
                                       'unmatched_results': 0,
                                       'calls_without_result': 0},
                        'retained_state_sha256': '09f2d71e51deb5a9c716b67711c637d31c0e39b912d2c4041a2a3444a342047c'},
    'medium-cjk': {   'pairing': {   'tool_calls': 23,
                                     'tool_results': 23,
                                     'matched_results': 23,
                                     'unmatched_results': 0,
                                     'calls_without_result': 0},
                      'retained_state_sha256': '7560d6e88b83582638afbe2a425451812499a9e664ea9f8f655cc506fdc4a950'},
    'medium-emoji': {   'pairing': {   'tool_calls': 28,
                                       'tool_results': 28,
                                       'matched_results': 28,
                                       'unmatched_results': 0,
                                       'calls_without_result': 0},
                        'retained_state_sha256': '5a289dc73b9aa1448c81392713230ba02373eca2cf172371eda6ffbbc6005856'},
    'medium-mixed-crlf': {   'pairing': {   'tool_calls': 15,
                                            'tool_results': 15,
                                            'matched_results': 15,
                                            'unmatched_results': 0,
                                            'calls_without_result': 0},
                             'retained_state_sha256': '1b25880e0b2cf807c19d71bbf19d7b8b33f972666d46ea5c141bda74d227336f'},
    'medium-tool-huge': {   'pairing': {   'tool_calls': 2,
                                           'tool_results': 2,
                                           'matched_results': 2,
                                           'unmatched_results': 0,
                                           'calls_without_result': 0},
                            'retained_state_sha256': '2c13f5dc5bedc480dd603c2ff9a9d2678ac02b6d7cf3895127f7147983120ef0'},
    'medium-many-small': {   'pairing': {   'tool_calls': 19,
                                            'tool_results': 19,
                                            'matched_results': 19,
                                            'unmatched_results': 0,
                                            'calls_without_result': 0},
                             'retained_state_sha256': '6aadd053bbb7e8294294c8082ba6cd89686017205e0649bece5a8b9c55a95084'},
    'large-ascii': {   'pairing': {   'tool_calls': 27,
                                      'tool_results': 27,
                                      'matched_results': 27,
                                      'unmatched_results': 0,
                                      'calls_without_result': 0},
                       'retained_state_sha256': '6ee2eaf3ae8a7e8aa2b3a111bf4f79fed9301f590b3183535ddc3b2bb155d77f'},
    'large-cjk': {   'pairing': {   'tool_calls': 20,
                                    'tool_results': 20,
                                    'matched_results': 20,
                                    'unmatched_results': 0,
                                    'calls_without_result': 0},
                     'retained_state_sha256': 'ef18a659bfa9dd8624f75ae1c0c7e902ac83ac8700e604ba78d797c7969b25d9'},
    'large-emoji': {   'pairing': {   'tool_calls': 34,
                                      'tool_results': 34,
                                      'matched_results': 34,
                                      'unmatched_results': 0,
                                      'calls_without_result': 0},
                       'retained_state_sha256': 'ed7dabfe0b1b1913ffa0bb985ece6aee058e572e005134afaeb7ffc6d565e1b7'},
    'large-mixed-crlf': {   'pairing': {   'tool_calls': 19,
                                           'tool_results': 19,
                                           'matched_results': 19,
                                           'unmatched_results': 0,
                                           'calls_without_result': 0},
                            'retained_state_sha256': '3eb5190fbf425c27247a8c7d3ae4658bf78ac43d8a41fe93f1347eae8329d548'},
    'large-tool-huge': {   'pairing': {   'tool_calls': 0,
                                          'tool_results': 0,
                                          'matched_results': 0,
                                          'unmatched_results': 0,
                                          'calls_without_result': 0},
                           'retained_state_sha256': '16426849732630041c51aa651c1a61bb667e48bc0378f61d63eddf1f816d09a9'},
    'large-many-small': {   'pairing': {   'tool_calls': 20,
                                           'tool_results': 20,
                                           'matched_results': 20,
                                           'unmatched_results': 0,
                                           'calls_without_result': 0},
                            'retained_state_sha256': 'f418d1013d83f52039afe1d7f32f4e0f638e424ce2141285ebdf8005a1dcbe7e'},
    'near-threshold-ascii': {   'pairing': {   'tool_calls': 38,
                                               'tool_results': 38,
                                               'matched_results': 38,
                                               'unmatched_results': 0,
                                               'calls_without_result': 0},
                                'retained_state_sha256': '6e7023c2fefc62863b78c8a81ff63a01baae065882a2ac11c370159d72255736'},
    'xlarge-ascii': {   'pairing': {   'tool_calls': 7,
                                       'tool_results': 7,
                                       'matched_results': 7,
                                       'unmatched_results': 0,
                                       'calls_without_result': 0},
                        'retained_state_sha256': '8ea91a0cb17209bd077d7be6c7ae7b5231f8fa3255df20a5e928db4757412363'},
    'xlarge-tool-huge': {   'pairing': {   'tool_calls': 0,
                                           'tool_results': 0,
                                           'matched_results': 0,
                                           'unmatched_results': 0,
                                           'calls_without_result': 0},
                            'retained_state_sha256': 'cb043fff9f37572854ebb3af8b991d44b0a8f931af7200b85c5d6b73196f8d23'}}


def validate_pairing(value: Any, stage: str, require_nonempty: bool = True) -> None:
    keys = ("tool_calls", "tool_results", "matched_results", "unmatched_results", "calls_without_result", "duplicate_tool_call_ids", "duplicate_tool_result_ids")
    exact_object(value, keys, stage)
    for key in keys:
        strict_int(value[key], stage)
    require(value["tool_calls"] == value["tool_results"] == value["matched_results"], stage, "tool call/result mismatch")
    require(all(value[key] == 0 for key in keys[3:]), stage, "unmatched or duplicate tool IDs")
    if require_nonempty:
        require(value["tool_calls"] > 0, stage, "nonempty tool pairing required")


def validate_replay_case(
    report: Any,
    case_name: str,
    summary_calls: int,
    *,
    failure: bool = False,
    context_window: float = 1_000_000.0,
    timeout_secs: int = 900,
    frozen_restored_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """R1 durable writer/fresh-reader anchors, independent of varying PIDs/paths."""
    exact_object(report, ("harness", "subcommand", "schema", "env", "runtime", "config", "case_count", "cases", "failures"), "replay-report")
    require(report["harness"] == "compaction_replay" and report["subcommand"] == "replay" and type(report["schema"]) is int, "replay-report", "wrong replay identity")
    require(report["failures"] == [] and report["case_count"] == 1 and len(report["cases"]) == 1, "replay-case-count", "one successful case required")
    config = report["config"]
    require(config.get("provider_delay_ms") == 0 and config.get("expect_summary_calls") == summary_calls, "replay-summary-N", "provider delay or expected summary N differs")
    require(config.get("expect_outcome") == ("summary-failure" if failure else "successful-compaction"), "replay-outcome-config", "wrong outcome")
    require(config.get("context_window") == context_window and config.get("timeout_secs") == timeout_secs, "replay-anchor-settings", "wrong context or timeout")
    if failure:
        require(config.get("summary_failure") == "R1 intentional summary failure", "replay-failure-message", "exact injected message required")
    else:
        require(config.get("summary_failure") is None, "replay-failure-message", "success must have no injected failure")
    case = report["cases"][0]
    require(case.get("case") == case_name, "replay-case-order", case_name)
    trigger = case.get("trigger", {})
    require(trigger.get("expected_fire") is True and trigger.get("fired") is True, "replay-trigger", "expected fire did not fire")
    counts = case.get("provider_call_counts")
    require(counts == {"summary": summary_calls, "turn": 1}, "replay-provider-counts", f"expected summary={summary_calls}, turn=1")
    expected_admission = case.get("expected_admission")
    require(expected_admission == {"summary_calls": summary_calls, "turn_calls": 1, "injected_summary_failure": config["summary_failure"]}, "replay-summary-N", "expected admission differs")
    outcome = case.get("outcome", {})
    require(outcome.get("compaction_entries") == (0 if failure else 1), "replay-compaction-count", "wrong durable compaction count")
    require(outcome.get("transient_failure_notice_count") == (1 if failure else 0), "replay-transient-notice", "wrong live notice count")
    delta = case.get("durable_turn_delta", {})
    require(delta.get("seed_unchanged") is True and delta.get("added_compaction_entries") == (0 if failure else 1), "replay-durable-graph", "seed changed or wrong checkpoint count")
    require(delta.get("added_message_counts") == {"user": 1, "assistant": 1, "tool_results": 0, "other": 0}, "replay-durable-turns", "unexpected durable turns")
    validate_pairing(case.get("raw_durable_pairing"), "replay-raw-pairing")
    restored_pairing = case.get("restored_pairing")
    if frozen_restored_identity is None:
        validate_pairing(restored_pairing, "replay-restored-pairing")
    else:
        exact_object(frozen_restored_identity, ("pairing", "retained_state_sha256"), "replay-restored-pairing-frozen-baseline")
        expected_pairing = frozen_restored_identity["pairing"]
        exact_object(expected_pairing, ("tool_calls", "tool_results", "matched_results", "unmatched_results", "calls_without_result"), "replay-restored-pairing-frozen-baseline")
        # Empty restored pairing is allowed only through this explicit frozen, per-case binding.
        # Raw durable pairing above retains its independent strict nonempty check.
        validate_pairing(restored_pairing, "replay-restored-pairing", require_nonempty=False)
        actual_pairing = {key: restored_pairing[key] for key in expected_pairing}
        require(actual_pairing == expected_pairing, "replay-restored-pairing-frozen-baseline", case_name)
    reader = case.get("fresh_process_reopen", {}).get("report", {})
    require(reader.get("read_only_reopen") is True and reader.get("fresh_process") is True and reader.get("additional_user_or_provider_turns") == 0, "replay-reader", "fresh read-only reader required")
    require(reader.get("durable_sha256") == reader.get("rehydrated_branch_sha256") and reader.get("durable_sha256") is not None, "replay-reader-graph", "reader graph hashes differ")
    require(reader.get("retained_state_sha256") is not None, "replay-reader-state", "reader retained-state hash missing")
    require(case.get("digests", {}).get("reopen_context_sha256") == reader.get("retained_state_sha256"), "replay-reader-parity", "writer/reader retained digest differs")
    if frozen_restored_identity is not None:
        require(reader["retained_state_sha256"] == frozen_restored_identity["retained_state_sha256"], "replay-restored-retained-state-frozen-baseline", case_name)
    writer_pid = strict_int(report.get("env", {}).get("pid"), "replay-writer-pid", 1)
    reader_pid = strict_int(reader.get("reader", {}).get("pid"), "replay-reader-pid", 1)
    require(writer_pid != reader_pid, "replay-writer-reader-pid", "writer and reader PIDs must differ")
    writer_exe = sha256_text(report["env"].get("exe_sha256"), "replay-exe")
    require(report["env"].get("exe_sha256_source") == "computed-from-running-exe" and reader["reader"].get("exe_sha256") == writer_exe, "replay-exe", "writer/reader running image identity differs")
    require(reader.get("parity") == "full graph + exact raw durable branch + normalized durable build_session_context + protected custom/checkpoint state", "replay-reader-parity", "full parity label changed")
    require(isinstance(case.get("protected_state"), dict) and isinstance(case["protected_state"].get("checkpoints"), list), "replay-protected-state", "protected checkpoints missing")
    require(len(case["protected_state"]["checkpoints"]) == (0 if failure else 1), "replay-checkpoint-links", "wrong protected checkpoint count")
    return {
        "case": case_name, "outcome": "summary-failure" if failure else "successful-compaction",
        "summary_calls": summary_calls, "ordinary_provider_turns": 1,
        "compaction_entries": outcome["compaction_entries"],
        "writer_pid": writer_pid, "reader_pid": reader_pid, "exe_sha256": writer_exe,
        "retained_state_sha256": reader["retained_state_sha256"],
        "structured_sha256": structured_sha256({
            "raw_durable_pairing": case["raw_durable_pairing"], "restored_pairing": case["restored_pairing"],
            "durable_turn_delta": delta, "reader_graph": {key: reader[key] for key in ("durable_sha256", "rehydrated_branch_sha256", "retained_state_sha256", "graph_entry_count", "branch_entry_count")},
            "protected_state_keys": sorted(case["protected_state"]),
        }),
    }


def aggregate_all21(reports: Mapping[str, Any], expected_order: Sequence[str] = CASE_ORDER) -> dict[str, Any]:
    names = list(reports)
    require(tuple(names) == tuple(expected_order) == CASE_ORDER, "all21-case-order", "missing, extra or reordered all21 case")
    anchors = {
        name: validate_replay_case(
            reports[name], name, CASE_SUMMARY_CALLS[name],
            frozen_restored_identity=FROZEN_SUCCESSFUL_RESTORED_IDENTITIES[name],
        ) for name in CASE_ORDER
    }
    return {"schema": "compaction-r5-all21-replay-v1", "status": "ok", "case_count": 21, "case_order": list(CASE_ORDER), "cases": anchors, "failures": []}


def parity_section(case: Mapping[str, Any], section: str) -> Any:
    require(section in PARITY_SECTIONS, "parity-section-coverage", section)
    if section == "case":
        return case.get("case")
    if section.startswith("session_manager__"):
        manager = case.get("session_manager")
        require(isinstance(manager, dict), "parity-section-coverage", "session_manager")
        key = section.removeprefix("session_manager__")
        require(key in manager, "parity-section-coverage", f"session_manager/{key}")
        return manager[key]
    require(section in case, "parity-section-coverage", section)
    return case[section]


def compare_local_parity(baseline: Any, candidate: Any, *, case_names: Sequence[str] = CASE_ORDER, scope: str = "all21-exact-full14") -> dict[str, Any]:
    for label, report in (("R3", baseline), ("R4", candidate)):
        require(isinstance(report, dict) and report.get("status") == "ok" and report.get("non_timed") is True and report.get("performance_claim") is False, "local-parity-status", f"{label} status changed")
        require(report.get("config", {}).get("chunk_context_window") == 32768.0, "local-parity-settings", f"{label} chunk window differs")
    expected_order = tuple(sorted(case_names))
    require(
        (tuple(case_names) == CASE_ORDER and scope == "all21-exact-full14")
        or (tuple(case_names) == ("small-ascii",) and scope == "small-ascii-exact-full14"),
        "local-parity-scope", "all21 or small-ascii exactfull14 scope required",
    )
    baseline_cases = baseline.get("cases")
    candidate_cases = candidate.get("cases")
    require(isinstance(baseline_cases, list) and isinstance(candidate_cases, list) and len(baseline_cases) == len(candidate_cases) == len(expected_order), "all21-case-count", f"exactly {len(expected_order)} local-parity cases required")
    hashes: dict[str, str] = {}
    for index, (left, right) in enumerate(zip(baseline_cases, candidate_cases)):
        expected_name = expected_order[index]
        require(left.get("case") == right.get("case") == expected_name, "all21-case-order", f"case {index} differs")
        left_projection = left.get("full_projection_sha256")
        right_projection = right.get("full_projection_sha256")
        for projection in (left_projection, right_projection):
            exact_object(projection, ("algorithm", "note", "sections"), "parity-projection-schema")
            require(projection["algorithm"] == FULL_PROJECTION_ALGORITHM and projection["note"] == FULL_PROJECTION_NOTE, "parity-projection-schema", "algorithm/note changed")
            require(tuple(projection["sections"]) == PARITY_SECTIONS, "parity-projection-sections", "missing, extra or reordered section")
            for digest in projection["sections"].values():
                sha256_text(digest, "parity-projection-section")
        for section in PARITY_SECTIONS:
            exact_compare(parity_section(left, section), parity_section(right, section), f"/{expected_name}/{section}")
        exact_compare(left_projection, right_projection, f"/{expected_name}/full_projection_sha256")
        # The text must also agree as an exact Unicode string and as UTF-8 bytes.
        left_text = parity_section(left, "serialization").get("text") if isinstance(parity_section(left, "serialization"), dict) else None
        right_text = parity_section(right, "serialization").get("text") if isinstance(parity_section(right, "serialization"), dict) else None
        require(isinstance(left_text, str) and isinstance(right_text, str), "serialization-utf8", expected_name)
        require(left_text.encode("utf-8") == right_text.encode("utf-8"), "serialization-utf8", expected_name)
        hashes[expected_name] = structured_sha256({section: parity_section(left, section) for section in PARITY_SECTIONS})
    require(candidate.get("comparison") == "exact-full-output-against-baseline", "local-parity-comparison", "candidate comparison label changed")
    return {"schema": "compaction-r5-local-parity-v1", "status": "ok", "scope": scope, "case_count": len(expected_order), "sections": list(PARITY_SECTIONS), "structured_sha256": hashes, "comparison": "exact values/types/order/UTF8/finite-IEEE754 plus built-in full projections"}


def validate_telemetry(value: Any) -> dict[str, Any]:
    exact_object(value, ("before", "after", "observation", "required_fields"), "telemetry-shape")
    require(value["required_fields"] == list(TELEMETRY_FIELDS), "telemetry-required-fields", "field list cannot be relaxed")
    require(value["observation"] == OBSERVATION_STATES, "telemetry-state-label", "launch/cache/thermal labels are not observed facts")
    for phase in ("before", "after"):
        block = exact_object(value[phase], TELEMETRY_FIELDS, "telemetry-fields")
        for field in TELEMETRY_FIELDS:
            item = exact_object(block[field], ("value", "unavailable_reason"), "telemetry-value")
            if item["value"] is None:
                nonempty_string(item["unavailable_reason"], "telemetry-unavailable")
            else:
                require(item["unavailable_reason"] is None, "telemetry-unavailable", "available value must have null reason")
                if field == "active_power_scheme":
                    nonempty_string(item["value"], "telemetry-power")
                elif isinstance(item["value"], (int, float)) and not isinstance(item["value"], bool):
                    require(math.isfinite(float(item["value"])), "telemetry-nonfinite", field)
                elif field != "per_logical_processor":
                    raise Reject("telemetry-type", field)
        require(isinstance(block["per_logical_processor"]["value"], (list, dict)) or block["per_logical_processor"]["value"] is None, "telemetry-per-cpu", "array/object/null required")
    return value


def validate_affinity(value: Any, *, nested_wall: bool = False) -> dict[str, Any]:
    exact_object(value, ("requested_mask", "children"), "affinity-shape")
    require(value["requested_mask"] == hex(MASK), "affinity-mask", "only 0x555 is admitted")
    require(isinstance(value["children"], list) and value["children"], "affinity-mask", "real child evidence required")
    pids: set[int] = set()
    for child in value["children"]:
        exact_object(child, ("pid", "role", "observed_mask", "running_image"), "affinity-child")
        pid = strict_int(child["pid"], "affinity-pid", 1)
        require(pid not in pids, "affinity-duplicate-pid", str(pid))
        pids.add(pid)
        require(child["role"] in ({"wrapper", "production"} if nested_wall else {"production"}), "affinity-role", child["role"])
        require(child["observed_mask"] == hex(MASK), "affinity-mask", str(pid))
        nonempty_string(child["running_image"], "running-exe-identity")
    roles = [child["role"] for child in value["children"]]
    require("production" in roles and (not nested_wall or "wrapper" in roles), "affinity-coverage", "production/wrapper observation missing")
    return value


def validate_launch_image(spec: ProcessSpec, running_image: str) -> None:
    """Validate nested wall wrapper separately from its expected production executable."""
    if spec.expected_running_image is None:
        return
    if spec.nested_wall:
        require(os.path.normcase(str(Path(running_image).resolve())) == os.path.normcase(str(Path(sys.executable).resolve())), "wall-wrapper-image", "nested wall launcher must be the registered Python interpreter")
    else:
        require(os.path.normcase(str(Path(running_image).resolve(strict=True))) == os.path.normcase(str(spec.expected_running_image.resolve(strict=True))), "running-exe-identity", running_image)


def correlate_production_pid(receipt: ProcessReceipt, report: Any, lane: str) -> int:
    affinity = validate_affinity(receipt.affinity, nested_wall=(lane == "wall"))
    children = affinity["children"]
    require(affinity["requested_mask"] == hex(MASK) and isinstance(children, list), "production-affinity", "strict child-only 0x555 evidence required")
    by_role: dict[str, list[Mapping[str, Any]]] = {}
    for child in children:
        by_role.setdefault(child.get("role"), []).append(child)
    production = by_role.get("production", [])
    require(len(production) == 1, "production-pid-correlation", "exactly one observed production child required")
    if lane == "bench-local-warm":
        reported = report["env"]["pid"]
    elif lane == "wall":
        require(validate_external_wall(report) is not None, "production-wall-report", "wall report required")
        reported = report["child_report"]["run"]["pid"]
        require(report["child"]["pid"] == reported, "production-pid-correlation", "wall parent/child report PID mismatch")
        wrapper = by_role.get("wrapper", [])
        require(len(wrapper) == 1 and wrapper[0]["pid"] == receipt.pid, "production-wrapper-correlation", "observed wrapper differs from executor direct child")
    else:
        reported = report["run"]["pid"]
    reported = strict_int(reported, "production-pid", 1)
    require(strict_int(production[0].get("pid"), "production-pid", 1) == reported, "production-pid-correlation", "output-reported production PID differs from affinity-observed PID")
    if lane != "wall":
        require(reported == receipt.pid, "production-pid-correlation", "direct production PID must equal the executor child PID")
    expected_image = receipt.spec.expected_running_image
    require(expected_image is not None, "production-image", "expected production image missing")
    for image in (production[0].get("running_image"), receipt.running_image if lane != "wall" else None):
        if image is not None:
            require(Path(image).resolve() == Path(expected_image).resolve(), "production-image-correlation", "production running image differs")
    return reported


def make_run_matrix(eligible_cases: Sequence[str]) -> list[dict[str, Any]]:
    require(tuple(eligible_cases) == R4_ONE_CALL_TIMING_CASES, "r4-eligibility", "exact four supported R4 timing cases in fixture order required; the R3 one-summary-call set is not a timing candidate list")
    matrix: list[dict[str, Any]] = []
    order = 0
    for lane in ONE_SHOT_LANES:
        for case in eligible_cases:
            for observation in range(30):
                matrix.append({"order": order, "lane": lane, "case": case, "observation": observation, "fresh_processes": 1, "warmup": None, "iterations": None, "p50_from_inner_samples": False})
                order += 1
    for case in WARM_CASES:
        for observation in range(30):
            matrix.append({"order": order, "lane": "bench-local-warm", "case": case, "observation": observation, "fresh_processes": 1, "warmup": 10, "iterations": 30, "p50_from_inner_samples": True})
            order += 1
    require(len(matrix) == 750, "run-matrix-count", "expected 480 one-call and 270 warm baseline processes")
    return matrix


def validate_unavailable_warm_preflights(records: Any, fixture_root: Path, artifact_root: Path) -> list[dict[str, Any]]:
    require(isinstance(records, list) and len(records) == len(UNAVAILABLE_WARM_PREFLIGHTS), "warm-preflight-count", "exact twelve non-timed unavailable records required")
    for (case_name, history), raw_row in zip(UNAVAILABLE_WARM_PREFLIGHTS, records):
        stage = f"warm-preflight-record:{case_name}"
        require(isinstance(raw_row, dict) and raw_row.get("case") == case_name, stage, "exact case order and identity required")
        require(type(raw_row.get("history")) is int and raw_row["history"] == history, stage, "history differs from exact R12 binding")
        require(raw_row.get("schema") == "compaction-r5-warm-preflight-unavailable-v1" and raw_row.get("status") == "unavailable", stage, "unavailable record required")
        require(raw_row.get("reason") == JEV_INPUT_LIMIT_REASON and raw_row.get("partial_parity") is False, stage, "input-limit reason required")
        expected_error = f"error: {case_name}: non-timed preflight failed: [jev-prepare] unexpected production Jev prepare_context error: compaction_input_limit"
        require(raw_row.get("error") == expected_error, "warm-preflight-error", "exact normalized native error required")
        require(type(raw_row.get("exit_code")) is int and raw_row["exit_code"] == 1, stage, "native exit 1 required")
        require(raw_row.get("no_output") is True and raw_row.get("stdout_sha256") == hashlib.sha256(b"").hexdigest(), "warm-preflight-output", "empty stdout and no report required")
        require(raw_row.get("timed_observations") == [] and raw_row.get("statistics") is None and raw_row.get("timing_executed") is False and raw_row.get("non_timed") is True, "warm-preflight-no-timing", "no timed observation or statistics may be fabricated")
        require(type(raw_row.get("fresh_processes")) is int and raw_row["fresh_processes"] == 1, "warm-preflight-fresh-count", "one synthetic/native process attempt only")
        require(raw_row.get("config") == {"warmup": 10, "iterations": 30, "chunk_context_window": 32768}, stage, "warmup10/iters30/chunk32768 required")
        require(raw_row.get("stage") == "gate-a/warm-preflight-unavailable" and raw_row.get("run_id") == case_name, stage, "non-timed GateA stage required")
        expected_argv = bench_command(Path(nonempty_string(raw_row.get("exe"), stage)), fixture_root / case_name, artifact_root / "gate-a/warm-preflight-unavailable" / case_name / "out.json")
        require(tuple(raw_row.get("argv", ())) == expected_argv, "warm-preflight-argv", "exact single-case bench-local argv required")
        require(raw_row.get("argv_sha256") == structured_sha256(expected_argv), "warm-preflight-argv-hash", "argv digest differs")
        sha256_text(raw_row.get("exe_sha256"), "warm-preflight-exe-hash")
        require(file_identity(Path(raw_row["exe"]))["sha256"] == raw_row["exe_sha256"], "warm-preflight-exe-hash", "executable identity differs")
        sha256_text(raw_row.get("stderr_sha256"), "warm-preflight-stderr-hash")
        sha256_text(raw_row.get("process_receipt_sha256"), "warm-preflight-process-receipt")
    return list(records)



def validate_r4_noneligible_case(case_name: str, fixture_root: Path, exe: Path, expected_exe_sha256: str, receipt: ProcessReceipt, report: Any) -> dict[str, Any]:
    """Bind one unavailable, noneligible native duplicate-summary observation.

    The panic thread ID is runtime data.  It is neither the native process PID
    nor an allowlisted constant.  Only the stderr grammar and panic source
    location are fixed by this control.
    """
    stage = "r4-noneligible-case"
    require(case_name in R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES, stage, "case must be one of the three exact duplicate-summary instrumentation negatives")
    spec = receipt.spec
    require(spec.stage == "gate-a/r4-eligibility" and spec.run_id == case_name, stage, "native stage/run identity differs")
    expected_argv = r4_lane_command(exe, "replay-phase", fixture_root / case_name, spec.work, spec.out)
    require(tuple(spec.argv) == expected_argv, "r4-noneligible-argv", "argv must match the same case and isolated work/output paths")
    require(dict(spec.allowlisted_env) == {}, "r4-noneligible-env", "native process must use the empty allowlisted environment")
    require(spec.expected_exit == -1 and not spec.pin_affinity and not spec.nested_wall, "r4-noneligible-spec", "one exploratory eligibility process, not a timed or nested child")
    require(spec.profile_root is None and not spec.create_profile and spec.profile_id is None and receipt.profile is None, "r4-noneligible-profile", "ordinary native process must not use an isolated user profile")
    require(type(receipt.pid) is int and receipt.pid > 0, "r4-noneligible-pid", "positive native process PID required")
    require(receipt.exit_code == 101, "r4-noneligible-exit", "native duplicate-summary instrumentation panic must be exit 101")
    stdout_bytes = spec.stdout.read_bytes()
    require(stdout_bytes == b"" and report is None and not spec.out.exists(), "r4-noneligible-output", "empty stdout and no out.json/report must both hold")
    stderr_bytes = spec.stderr.read_bytes()
    require(isinstance(stderr_bytes, bytes), "r4-noneligible-stderr", "native stderr bytes required")
    try:
        stderr_text = stderr_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Reject("r4-noneligible-stderr", "native stderr is not strict UTF-8") from error
    panic_pattern = re.compile(
        r"\A\nthread 'main' \(([0-9]+)\) panicked at .*r4_lanes\.rs:200:9:\n"
        + re.escape(R4_DUPLICATE_SUMMARY_MARKER)
        + r"\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n\Z"
    )
    panic_match = panic_pattern.fullmatch(stderr_text)
    require(panic_match is not None, "r4-native-panic-boundary", "stderr must be the exact summary_runner_entry panic at r4_lanes.rs:200:9, with one dynamic thread ID")
    panic_thread_id = int(panic_match.group(1))
    require(panic_thread_id > 0, "r4-native-thread-id", "positive dynamic panic thread ID required")
    require(panic_thread_id != receipt.pid, "r4-native-thread-id", "panic thread ID must be recorded separately from, and must differ from, the native process PID")
    require(stderr_text.count(R4_DUPLICATE_SUMMARY_MARKER) == 1, "r4-native-panic-boundary", "exactly one duplicate-summary marker required")
    raw_receipt = load_json(spec.work / "receipt.json")
    require(raw_receipt.get("schema") == "compaction-r5-process-receipt-v1", "r4-noneligible-receipt", "native process receipt schema required")
    require(raw_receipt.get("stage") == spec.stage and raw_receipt.get("run_id") == case_name, "r4-noneligible-receipt", "native process receipt identity differs")
    require(tuple(raw_receipt.get("argv", ())) == expected_argv, "r4-noneligible-receipt", "persisted native argv differs")
    require(raw_receipt.get("allowlisted_env") == {} and raw_receipt.get("profile") is None, "r4-noneligible-receipt", "persisted native environment/profile differs")
    require(raw_receipt.get("sequential_adapter_pid") == receipt.pid and raw_receipt.get("exit_code") == 101, "r4-noneligible-receipt", "persisted native PID or exit code differs")
    exe_identity = file_identity(exe)
    require(exe_identity["sha256"] == expected_exe_sha256, "r4-noneligible-exe", "native executable SHA256 differs")
    stdout_identity = file_identity(spec.stdout)
    stderr_identity = file_identity(spec.stderr)
    receipt_identity = file_identity(spec.work / "receipt.json")
    return validate_r4_noneligible_records([{
        "schema": "compaction-r5-r4-noneligible-native-v1",
        "status": "unavailable_noneligible",
        "case": case_name,
        "order": next(index for index, item in enumerate(CASE_ORDER) if item == case_name),
        "lane": "replay-phase",
        "reason": R4_NONELIGIBILITY_REASON,
        "duplicate_phase_mark": "summary_runner_entry",
        "panic_location": "r4_lanes.rs:200:9",
        "native_pid": receipt.pid,
        "panic_thread_id": panic_thread_id,
        "exit_code": receipt.exit_code,
        "argv": list(spec.argv),
        "argv_sha256": structured_sha256(spec.argv),
        "allowlisted_env": dict(spec.allowlisted_env),
        "profile": None,
        "exe": str(exe),
        "exe_sha256": expected_exe_sha256,
        "stdout": stdout_identity,
        "stdout_bytes": 0,
        "stdout_empty": True,
        "stderr": stderr_identity,
        "stderr_text": stderr_text,
        "out_json": None,
        "out_json_absent": True,
        "report_absent": True,
        "process_receipt": receipt_identity,
        "process_receipt_sha256": receipt_identity["sha256"],
        "timing_observation": False,
        "timed_observations": [],
        "statistics": None,
        "jev_skip": False,
        "fresh_processes": 1,
    }], (case_name,))[0]


def validate_r4_noneligible_records(records: Any, expected_cases: Sequence[str] = R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES) -> list[dict[str, Any]]:
    expected = tuple(expected_cases)
    require(expected == R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES or (len(expected) == 1 and expected[0] in R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES), "r4-noneligible-case", "record scope must be the exact three-case set or one named member of that set")
    require(type(records) is list and len(records) == len(expected), "r4-noneligible-count", "exactly the expected native unavailable/noneligible record count required")
    require(tuple(row.get("case") if isinstance(row,dict) else None for row in records) == expected, "r4-noneligible-case", "expected cases must be present exactly once in the required order")
    for row in records:
        stage = "r4-noneligible-record"
        require(isinstance(row,dict), stage, "native record must be an object")
        require(row.get("schema") == "compaction-r5-r4-noneligible-native-v1" and row.get("status") == "unavailable_noneligible", stage, "schema/status must not classify a native panic as success")
        case_name = row["case"]
        require(row.get("order") == CASE_ORDER.index(case_name) and row.get("lane") == "replay-phase", stage, "native case/lane/order differs")
        require(row.get("reason") == R4_NONELIGIBILITY_REASON and row.get("duplicate_phase_mark") == "summary_runner_entry" and row.get("panic_location") == "r4_lanes.rs:200:9", stage, "native instrumentation boundary differs")
        require(type(row.get("native_pid")) is int and row["native_pid"] > 0 and type(row.get("panic_thread_id")) is int and row["panic_thread_id"] > 0, stage, "dynamic native IDs must be positive integers")
        require(row["panic_thread_id"] != row["native_pid"], stage, "panic thread ID must remain distinct from the native PID")
        require(row.get("exit_code") == 101 and type(row.get("fresh_processes")) is int and row.get("fresh_processes") == 1, stage, "exact native exit/process identity required")
        require(row.get("argv_sha256") == structured_sha256(row.get("argv")), stage, "native argv digest differs")
        require(row.get("allowlisted_env") == {} and row.get("profile") is None, stage, "native process must have an empty allowlisted environment and no profile")
        sha256_text(row.get("exe_sha256"), "r4-noneligible-exe")
        require(row.get("stdout_empty") is True and row.get("stdout_bytes") == 0 and row.get("out_json") is None and row.get("out_json_absent") is True and row.get("report_absent") is True, "r4-noneligible-output", "no stdout or report may be invented")
        stdout_identity = row.get("stdout")
        require(isinstance(stdout_identity,dict) and stdout_identity.get("bytes") == 0 and stdout_identity.get("sha256") == hashlib.sha256(b"").hexdigest(), stage, "empty raw stdout identity required")
        stderr_text = row.get("stderr_text")
        require(isinstance(stderr_text,str), stage, "raw native stderr text required")
        stderr_identity = row.get("stderr")
        require(isinstance(stderr_identity,dict) and stderr_identity.get("bytes") == len(stderr_text.encode("utf-8")) and stderr_identity.get("sha256") == hashlib.sha256(stderr_text.encode("utf-8")).hexdigest(), stage, "raw stderr identity does not match retained stderr text")
        require(stderr_text.count(R4_DUPLICATE_SUMMARY_MARKER) == 1 and "r4_lanes.rs:200:9" in stderr_text, stage, "retained native panic text differs")
        require(row.get("process_receipt_sha256") == (row.get("process_receipt") or {}).get("sha256"), stage, "native receipt identity differs")
        require(row.get("timing_observation") is False and row.get("timed_observations") == [] and row.get("statistics") is None and row.get("jev_skip") is False, stage, "no timing value or arbitrary unavailable classification may be fabricated")
    return list(records)


def validate_run_matrix(matrix: Any, eligible_cases: Sequence[str]) -> list[dict[str, Any]]:
    require(matrix == make_run_matrix(eligible_cases), "run-matrix-identity", "matrix differs from exact baseline protocol")
    require(len({row["order"] for row in matrix}) == len(matrix), "run-matrix-order", "duplicate order")
    return matrix


def make_balanced_orders(seed: int) -> list[dict[str, Any]]:
    require(type(seed) is int, "pair-seed", "integer seed required")
    orders = [["baseline", "candidate"], ["candidate", "baseline"]] * 15
    random.Random(seed).shuffle(orders)
    result = [{"pair_id": index, "order": order} for index, order in enumerate(orders)]
    require(sum(item["order"][0] == "baseline" for item in result) == 15 and sum(item["order"][0] == "candidate" for item in result) == 15, "pair-balance", "15 AB and 15 BA required")
    return result


def paired_statistics(runs: Iterable[dict[str, Any]], orders: Any, seed: int) -> dict[str, Any]:
    expected = make_balanced_orders(seed)
    require(orders == expected and len(orders) == 30, "pair-order", "seed/order replay failed or wrong pair count")
    # Independent implementation retained from R3; it never sees inner samples.
    return independent_paired_analyze(runs, orders, seed=seed, bootstrap_samples=10000, confidence=0.95)


def paired_supported_statistics(runs: Iterable[dict[str, Any]], orders: Any, seed: int, optional_metrics: Sequence[str] = ("jev_prepare_context",)) -> dict[str, Any]:
    """Compare universal metrics on 30 balanced pairs; optional metrics only where both arms have numbers."""
    source_runs = [dict(run) for run in runs]
    universal_runs = []
    for run in source_runs:
        medians = dict(run.get("medians", {}))
        for metric in optional_metrics:
            medians.pop(metric, None)
        universal_runs.append({**run, "medians": medians})
    result: dict[str, Any] = {"universal": paired_statistics(universal_runs, orders, seed), "optional": {}}
    expected_orders = make_balanced_orders(seed)
    for metric in optional_metrics:
        matched = [index for index, order in enumerate(expected_orders) if metric in source_runs[2*index].get("medians", {}) and metric in source_runs[2*index+1].get("medians", {})]
        require(all((metric in source_runs[2*index].get("medians", {})) == (metric in source_runs[2*index+1].get("medians", {})) for index in matched), "paired-optional-matching", metric)
        selected_orders = [dict(expected_orders[index], pair_id=new_index) for new_index, index in enumerate(matched)]
        counts = {"AB": sum(order["order"] == ["baseline", "candidate"] for order in selected_orders), "BA": sum(order["order"] == ["candidate", "baseline"] for order in selected_orders)}
        item: dict[str, Any] = {"matched_pairs": len(matched), "orders": selected_orders, "order_counts": counts}
        if not matched:
            item.update({"status": "no-matched-observations", "numeric_comparison_supported": False})
        elif counts["AB"] == counts["BA"]:
            selected_runs = []
            for new_index, old_index in enumerate(matched):
                selected_runs.append({**source_runs[2*old_index], "pair_id": new_index, "role": "baseline", "medians": {metric: source_runs[2*old_index]["medians"][metric]}})
                selected_runs.append({**source_runs[2*old_index+1], "pair_id": new_index, "role": "candidate", "medians": {metric: source_runs[2*old_index+1]["medians"][metric]}})
            item.update({"status": "matched-balanced-pairs", "numeric_comparison_supported": True, "analysis": independent_paired_analyze(selected_runs, selected_orders, seed=seed, bootstrap_samples=10000, confidence=0.95)})
        else:
            item.update({"status": "matched-unbalanced-pairs", "numeric_comparison_supported": False, "reason": "The strict independent paired analyzer requires balanced AB/BA; no CI was fabricated."})
        result["optional"][metric] = item
    return result


def validate_protected_hashes(value: Any, stage: str = "protected-hashes") -> dict[str, dict[str, str]]:
    require(isinstance(value, dict) and set(value) == {"source", "fixture", "toolchain", "script", "exe", "run_matrix"}, stage, "six complete protected classes required")
    result: dict[str, dict[str, str]] = {}
    for kind, entries in value.items():
        require(isinstance(entries, dict) and entries, stage, f"{kind} must enumerate protected files")
        result[kind] = {}
        for path, digest in entries.items():
            nonempty_string(path, stage)
            result[kind][path] = sha256_text(digest, stage)
    return result


def validate_gate_a_review(review: Any, gate_a_summary: Any, gate_a_summary_bytes: bytes) -> dict[str, Any]:
    exact_object(review, ("schema", "reviewer", "verdict", "reviewed_at", "root_message_id", "gate_a_summary_sha256", "source", "protected_hashes"), "gate-a-review-schema")
    require(review["schema"] == GATE_A_REVIEW_SCHEMA and review["verdict"] == "APPROVED_GATE_A_CORRECTNESS_ONLY", "gate-a-review", "independent approval required")
    nonempty_string(review["reviewer"], "gate-a-reviewer")
    nonempty_string(review["reviewed_at"], "gate-a-review-time")
    nonempty_string(review["root_message_id"], "gate-a-review-message")
    require(review["gate_a_summary_sha256"] == hashlib.sha256(gate_a_summary_bytes).hexdigest(), "gate-a-stale-review", "summary hash differs")
    source = exact_object(review["source"], ("base", "commit", "patch_sha256", "run_matrix_sha256"), "gate-a-source")
    require(source["base"] == R4_BASE and source["commit"] == R4_COMMIT and source["patch_sha256"] == R4_PATCH_SHA256, "gate-a-source", "R4 base+patch identity differs")
    sha256_text(source["run_matrix_sha256"], "run-matrix-hash")
    require(gate_a_summary.get("schema") == GATE_A_SUMMARY_SCHEMA and gate_a_summary.get("status") == "ok" and gate_a_summary.get("timing_executed") is False, "gate-a-summary", "green non-timing summary required")
    require(source["run_matrix_sha256"] == gate_a_summary.get("run_matrix_sha256"), "gate-a-run-matrix", "review and summary matrices differ")
    reviewed_protected = validate_protected_hashes(review["protected_hashes"], "gate-a-review-protected-hashes")
    require(reviewed_protected == validate_protected_hashes(gate_a_summary.get("protected_hashes"), "gate-a-summary-protected-hashes"), "gate-a-protected-hashes", "review and Gate A summary protected inputs differ")
    return review


def authorize_gate_b(authorization: Any, review_path: Path, gate_a_summary_path: Path, run_matrix: Any, *, executor_binaries: Mapping[str, Path], protected_hashes: Any) -> dict[str, Any]:
    summary_bytes = gate_a_summary_path.read_bytes()
    summary = load_json(gate_a_summary_path)
    review = validate_gate_a_review(load_json(review_path), summary, summary_bytes)
    require(authorization is not None, "gate-b-authorization", "explicit root authorization artifact required")
    exact_object(authorization, ("schema", "authorized_by", "root_message_id", "issued_at", "gate_a_summary_sha256", "gate_a_review_sha256", "run_matrix_sha256", "fresh_timing_builds", "protected_hashes", "baseline_only", "no_network", "no_comparator", "r4_one_shot_protocol"), "gate-b-authorization-schema")
    require(authorization["schema"] == GATE_B_AUTH_SCHEMA and authorization["authorized_by"] == "root", "gate-b-authorization", "root-issued artifact required")
    nonempty_string(authorization["root_message_id"], "gate-b-message")
    nonempty_string(authorization["issued_at"], "gate-b-time")
    require(authorization["gate_a_summary_sha256"] == hashlib.sha256(summary_bytes).hexdigest(), "gate-b-stale-summary", "authorization summary hash differs")
    require(authorization["gate_a_review_sha256"] == file_identity(review_path)["sha256"], "gate-b-stale-review", "authorization review hash differs")
    matrix_hash = structured_sha256(run_matrix)
    require(authorization["run_matrix_sha256"] == summary["run_matrix_sha256"] == review["source"]["run_matrix_sha256"] == matrix_hash, "gate-b-run-matrix", "authorization matrix differs")
    require(authorization["baseline_only"] is True and authorization["no_network"] is True and authorization["no_comparator"] is True, "gate-b-limits", "baseline/no-network/no-comparator limits are required")
    protocol = exact_object(authorization["r4_one_shot_protocol"], ("fresh_processes_per_case_lane", "invented_warmup_or_inner_samples", "aggregate_single_observations_only"), "gate-b-one-shot")
    require(protocol == {"fresh_processes_per_case_lane": 30, "invented_warmup_or_inner_samples": False, "aggregate_single_observations_only": True}, "gate-b-one-shot", "explicit R4 fresh-process exception differs")
    actual_protected = validate_protected_hashes(protected_hashes, "gate-b-plan-protected-hashes")
    authorized_protected = validate_protected_hashes(authorization["protected_hashes"], "gate-b-authorization-protected-hashes")
    require(authorized_protected == actual_protected, "gate-b-protected-hashes", "authorization must exactly match all actual plan protected inputs")
    reviewed_protected = validate_protected_hashes(summary.get("protected_hashes"), "gate-a-summary-protected-hashes")
    for kind in actual_protected:
        if kind != "exe":
            require(actual_protected[kind] == reviewed_protected[kind], "gate-b-protected-hashes", f"fresh timing plan changed protected {kind} inputs")
    require(set(executor_binaries) == {"instrumented", "wall", "r3-parity"}, "gate-b-executor-coverage", "three registered executor paths required")
    builds = authorization["fresh_timing_builds"]
    require(isinstance(builds, dict) and set(builds) == {"instrumented", "wall"}, "timing-build-coverage", "fresh instrumented and wall builds required")
    for name, build in builds.items():
        exact_object(build, ("exe", "exe_sha256", "build_log", "build_log_sha256", "source_commit", "target_root", "feature", "profile", "fresh_build"), "timing-build-identity")
        exe = Path(build["exe"]).resolve()
        actual_exe = Path(executor_binaries[name]).resolve()
        require(exe == actual_exe, "gate-b-executor-not-authorized", f"actual {name} executor is not the authorized fresh timing executable")
        require(file_identity(exe)["sha256"] == build["exe_sha256"], "timing-build-hash", name)
        require(actual_protected["exe"].get(str(executor_binaries[name])) == build["exe_sha256"], "timing-plan-binary-hash", f"plan {name} binary is not bound in protected hashes")
        require(exe.is_relative_to(TARGET_ROOT / name), "timing-build-target", name)
        require(file_identity(Path(build["build_log"]))["sha256"] == build["build_log_sha256"], "timing-build-log", name)
        require(build["source_commit"] == R4_COMMIT and Path(build["target_root"]) == TARGET_ROOT / name and build["fresh_build"] is True, "timing-build-provenance", name)
        sha256_text(build["build_log_sha256"], "timing-build-log")
        require(build["profile"] == "release" and build["feature"] == ("compaction-r4-uninstrumented-wall" if name == "wall" else None), "timing-build-feature", name)
    r3_path = Path(executor_binaries["r3-parity"])
    require(actual_protected["exe"].get(str(r3_path)) == R3_REPLAY_EXE_SHA256, "timing-r3-protected-identity", "registered R3 executable is not the frozen approved non-timing reproduction")
    return authorization


def validate_warm_output(output: Any, case_name: str) -> dict[str, float]:
    require(isinstance(output, dict) and output.get("subcommand") == "bench-local" and type(output.get("schema")) is int, "warm-output-identity", "bench-local schema 1 required")
    config = exact_object(output.get("config"), ("iters", "warmup", "chunk_context_window"), "warm-output-config")
    require(strict_int(config["iters"], "warm-output-config") == 30 and strict_int(config["warmup"], "warm-output-config") == 10, "warm-output-config", "warmup10/iters30 strict integers required")
    require(type(config["chunk_context_window"]) is float and config["chunk_context_window"] == 32768.0, "warm-output-config", "chunk32768.0 required")
    cases = output.get("cases")
    require(isinstance(cases, list) and len(cases) == 1 and cases[0].get("case") == case_name, "warm-output-case", "exact requested case required")
    kernels = cases[0].get("kernels")
    require(isinstance(kernels, list), "warm-output-kernels", "kernel array required")
    by_name: dict[str, Any] = {}
    for kernel in kernels:
        name = nonempty_string(kernel.get("name"), "warm-kernel-name")
        require(name not in by_name, "warm-duplicate-kernel", name)
        by_name[name] = kernel
    expected = set(WARM_KERNELS) | {"jev_prepare_context"}
    require(set(by_name) == expected, "warm-kernel-coverage", f"missing={sorted(expected-set(by_name))} extra={sorted(set(by_name)-expected)}")
    jev = by_name["jev_prepare_context"]
    if jev.get("outcome") == "expected-skip":
        require(jev.get("stats") is None and jev.get("skip_class") in ("no_compaction_candidates", "insufficient_compaction_reduction"), "warm-jev-skip", "expected-skip requires null stats and one of the two named skip classes")
    else:
        require(jev.get("outcome") == "prepared", "warm-jev-outcome", "prepared or expected-skip required")
    medians: dict[str, float] = {}
    for name in WARM_KERNELS + (("jev_prepare_context",) if jev.get("outcome") == "prepared" else ()):
        kernel = by_name[name]
        require(type(kernel.get("warmup")) is int and kernel["warmup"] == 10, "warm-kernel-warmup", name)
        stats = exact_object(kernel.get("stats"), ("n", "mean_ms", "p50_ms", "p95_ms", "stddev_ms", "min_ms", "max_ms"), "warm-kernel-stats")
        require(type(stats["n"]) is int and stats["n"] == 30, "warm-inner-count", name)
        for stat_name, stat_value in stats.items():
            if stat_name == "n":
                continue
            if stat_name == "stddev_ms":
                finite_nonnegative(stat_value, "warm-stat-nonnegative")
            else:
                finite_positive(stat_value, "warm-stat-nonfinite")
        medians[name] = stats["p50_ms"]
        # Raw reports retain n=30 summary stats. Only p50 becomes an outer value.
    preflight = output.get("non_timed_preflight")
    require(isinstance(preflight, list) and len(preflight) == 1 and preflight[0].get("case") == case_name, "warm-preflight", "exact case preflight required")
    sha256_text(preflight[0].get("full_output_sha256"), "warm-preflight-hash")
    return medians


def validate_lane_report_r5(report: Any, lane: str, build: str) -> dict[str, Any]:
    """Use R4 validation without rejecting valid finite zero allocator counters."""
    if lane != "allocation":
        return validate_lane_report(report, lane, build)
    probe = copy.deepcopy(report)
    allocation = probe.get("allocation") if isinstance(probe, dict) else None
    if isinstance(allocation, dict) and allocation.get("status") == "available":
        if allocation.get("count") == 0:
            allocation["count"] = 1
        if allocation.get("bytes_requested") == 0:
            allocation["bytes_requested"] = 1
    validate_lane_report(probe, lane, build)
    original = report["allocation"]
    if original.get("status") == "unavailable":
        return report
    exact_object(original, ("status", "source", "scope", "count", "bytes_requested", "note"), "allocation")
    strict_int(original["count"], "allocation-count", 0)
    strict_int(original["bytes_requested"], "allocation-bytes", 0)
    return report


def validate_warm_observation(output: Any, case_name: str) -> tuple[dict[str, float], str]:
    values = validate_warm_output(output, case_name)
    kernels = output["cases"][0]["kernels"]
    jev = next(kernel for kernel in kernels if kernel["name"] == "jev_prepare_context")
    state = jev["outcome"]
    require(("jev_prepare_context" in values) == (state == "prepared"), "warm-jev-state", "numeric Jev values require prepared state")
    return values, state


def extract_one_shot(report: Any, lane: str) -> dict[str, Any]:
    try:
        if lane == "wall":
            validate_external_wall(report)
            value = finite_positive(report["external_wall"]["duration_ms"], "timing-numeric")
            return {"process_total_ms": value}
        validate_lane_report_r5(report, lane, "instrumented")
        require(report["outcome"]["status"] == "success" and report["continuation"]["status"] == "available", "r4-lane-outcome", "successful real continuation required")
        if lane == "allocation":
            allocation = report["allocation"]
            if allocation["status"] == "unavailable":
                return {"allocation_count": None, "allocation_bytes_requested": None, "unavailable_reason": allocation["reason"]}
            return {"allocation_count": allocation["count"], "allocation_bytes_requested": allocation["bytes_requested"], "unavailable_reason": None}
        values = {f"phase/{phase['name']}": finite_positive(phase["duration_ms"], "timing-numeric") for phase in report["local_replay"]["phases"]}
        if lane == "minimal-provider":
            require("phase/response_parse" in values, "provider-local-separation", "response_parse phase missing")
        return values
    except R4Reject as error:
        raise Reject("r4-report-validation", str(error)) from error


def summarize_fresh_processes(records: Sequence[Mapping[str, Any]], lane: str, case_name: str) -> dict[str, Any]:
    require(lane in ONE_SHOT_LANES and len(records) == 30, "baseline-observation-count", "30 one-shot fresh processes required")
    pids: set[int] = set()
    identities: set[str] = set()
    metrics: dict[str, list[float]] = {}
    unavailable: set[str] = set()
    for index, record in enumerate(records):
        require(record.get("status") == "admitted" and record.get("lane") == lane and record.get("case") == case_name, "baseline-run-admission", str(index))
        require(record.get("observation") == index and record.get("fresh_processes") == 1, "baseline-run-order", str(index))
        pid = strict_int(record.get("pid"), "baseline-pid", 1)
        require(pid not in pids, "pid-collision", str(pid))
        pids.add(pid)
        identity = sha256_text(record.get("identity_sha256"), "baseline-identity")
        identities.add(identity)
        values = record.get("values")
        require(isinstance(values, dict), "baseline-values", str(index))
        if "unavailable_reason" in record:
            if record["unavailable_reason"] is not None:
                unavailable.add(nonempty_string(record["unavailable_reason"], "allocation-unavailable"))
            values = {key: value for key, value in values.items() if value is not None}
        for name, value in values.items():
            numeric = finite_nonnegative(value, "allocation-numeric") if lane == "allocation" else finite_positive(value, "timing-numeric")
            metrics.setdefault(name, []).append(numeric)
    require(len(identities) == 1, "baseline-identity-drift", "source/fixture/exe/run-matrix identity changed")
    if unavailable:
        require(not metrics and len(unavailable) == 1, "allocation-unavailable-mixed", "available/unavailable data cannot be mixed")
        statistics_output: Any = {"status": "unavailable", "reason": next(iter(unavailable)), "numeric_values": None}
    else:
        require(all(len(values) == 30 for values in metrics.values()), "baseline-metric-count", "30 numeric observations required")
        statistics_output = {}
        for name, values in sorted(metrics.items()):
            ordered = sorted(values)
            statistics_output[name] = {"n": 30, "min": ordered[0], "p50": statistics.median(ordered), "p95": ordered[math.ceil(0.95 * 30) - 1], "max": ordered[-1]}
    return {"schema": "compaction-r5-one-shot-stats-v1", "status": "ok", "lane": lane, "case": case_name, "observation_unit": "one value per independent fresh process; no inner samples or warmups invented", "raw_observations": records, "statistics": statistics_output, "decision_evidence": False}


def validate_stage_inventory(root: Path, expected_relative_files: Iterable[str]) -> dict[str, Any]:
    expected_list = list(expected_relative_files)
    expected = {str(Path(name)) for name in expected_list}
    require(len(expected) == len(expected_list), "stage-file-plan", "duplicate expected file")
    actual_paths = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()} if root.exists() else set()
    require(actual_paths == expected, "stage-file-inventory", f"missing={sorted(expected-actual_paths)} extra={sorted(actual_paths-expected)}")
    identities = {name: file_identity(root / name) for name in sorted(actual_paths)}
    return {"schema": "compaction-r5-stage-inventory-v1", "status": "ok", "root": str(root), "files": identities}


def _live_git(root: Path, arguments: Sequence[str], git_runner: Any = None) -> dict[str, Any]:
    """Read-only Git identity probe with a fixed command catalog; never uses a shell."""
    allowed = {
        "head": ("rev-parse", "HEAD"),
        "parent": ("rev-parse", "HEAD^"),
        "branch": ("branch", "--show-current"),
        "scope": ("status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"),
        "untracked": ("ls-files", "--others", "--exclude-standard"),
        "tracked-diff": ("diff", "--name-only", "HEAD", "--"),
        "r4-paths": ("diff", "--name-only", R4_BASE, R4_COMMIT, "--"),
        "r4-patch": ("diff", "--binary", R4_BASE, R4_COMMIT),
    }
    actual = tuple(arguments)
    require(actual in allowed.values(), "command-catalog", f"unsupported Git argv: {list(arguments)!r}")
    argv = ["git", "-C", str(root), *actual]
    if git_runner is None:
        completed = subprocess.run(argv, cwd=str(root), capture_output=True, text=True, encoding="utf-8", timeout=30, check=False)
        result = {"argv": argv, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    else:
        probe = git_runner(root, list(actual)) if callable(git_runner) else git_runner.run(root, list(actual))
        exact_object(probe, ("argv", "returncode", "stdout", "stderr"), "git-probe-receipt")
        result = dict(probe)
        require(result["argv"] in (argv, list(actual)), "git-probe-identity", "prob command differs from requested catalog argv")
    require(all(type(item) is str for item in result["argv"]) and type(result["returncode"]) is int and type(result["stdout"]) is str and type(result["stderr"]) is str, "git-probe-schema", "invalid Git result types")
    require(result["returncode"] == 0, "git-probe-exit", f"{list(actual)!r}: {result['stderr']}")
    return result


def admit_source_identity(receipt: Any, git_runner: Any = None) -> dict[str, Any]:
    """Live-verify reviewed R4 base+patch and exactly four intentional R5 files."""
    exact_object(receipt, ("worktree", "branch", "head", "base", "parent", "r4_patch_path", "r4_patch_sha256", "r4_changed_paths", "r5_new_paths", "unexpected_paths", "production_diff", "approval_message_id", "source_file_sha256"), "source-identity-schema")
    root = Path(nonempty_string(receipt["worktree"], "source-worktree")).resolve()
    require((root / ".git").exists(), "source-worktree", "worktree git metadata missing")
    require(receipt["branch"] == "fix/compaction-r5-baseline" and receipt["head"] == R4_COMMIT and receipt["base"] == receipt["parent"] == R4_BASE, "source-commit-identity", "branch/head/base/parent differs")
    require(receipt["r4_changed_paths"] == list(R4_CHANGED_PATHS), "r4-patch-scope", "eight approved R4 paths differ")
    require(receipt["r5_new_paths"] == list(R5_SOURCE_PATHS), "r5-file-scope", "four approved R5 paths differ")
    require(receipt["unexpected_paths"] == [] and receipt["production_diff"] == [], "source-unexpected-diff", "unexpected or production source changes")
    nonempty_string(receipt["approval_message_id"], "source-approval")
    require(receipt["r4_patch_sha256"] == R4_PATCH_SHA256, "r4-patch-hash", "approved patch hash differs")
    require(file_identity(Path(receipt["r4_patch_path"]))["sha256"] == R4_PATCH_SHA256, "r4-patch-file", "approved patch bytes differ")
    file_map = receipt["source_file_sha256"]
    require(isinstance(file_map, dict) and set(file_map) == set(R5_SOURCE_PATHS), "r5-source-hashes", "four R5 hashes required")
    for relative, expected in file_map.items():
        sha256_text(expected, "r5-source-hash")
        require(file_identity(root / relative)["sha256"] == expected, "r5-source-hash", relative)
    probes = {
        "head": _live_git(root, ("rev-parse", "HEAD"), git_runner),
        "parent": _live_git(root, ("rev-parse", "HEAD^"), git_runner),
        "branch": _live_git(root, ("branch", "--show-current"), git_runner),
        "scope": _live_git(root, ("status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"), git_runner),
        "untracked": _live_git(root, ("ls-files", "--others", "--exclude-standard"), git_runner),
        "tracked-diff": _live_git(root, ("diff", "--name-only", "HEAD", "--"), git_runner),
        "r4-paths": _live_git(root, ("diff", "--name-only", R4_BASE, R4_COMMIT, "--"), git_runner),
        "r4-patch": _live_git(root, ("diff", "--binary", R4_BASE, R4_COMMIT), git_runner),
    }
    require(probes["head"]["stdout"].strip() == R4_COMMIT and probes["parent"]["stdout"].strip() == R4_BASE and probes["branch"]["stdout"].strip() == receipt["branch"], "git-live-identity", "live HEAD, parent or branch differs")
    expected_status = sorted(f"?? {path}" for path in R5_SOURCE_PATHS)
    expected_untracked = sorted(R5_SOURCE_PATHS)
    live_status = probes["scope"]["stdout"].splitlines()
    live_untracked = [item for item in probes["untracked"]["stdout"].splitlines() if item]
    live_tracked = [item for item in probes["tracked-diff"]["stdout"].splitlines() if item]
    live_r4_paths = [item for item in probes["r4-paths"]["stdout"].splitlines() if item]
    require(live_status == expected_status, "git-live-scope", f"status must be exact four R5 paths, got {live_status!r}")
    require(live_untracked == expected_untracked, "git-live-untracked", f"untracked paths differ: {live_untracked!r}")
    require(live_tracked == [], "git-live-production-diff", f"tracked modifications present: {live_tracked!r}")
    require(live_r4_paths == sorted(R4_CHANGED_PATHS), "git-live-r4-scope", f"R4 commit paths differ: {live_r4_paths!r}")
    require(hashlib.sha256(probes["r4-patch"]["stdout"].encode("utf-8")).hexdigest() == R4_PATCH_SHA256, "git-live-r4-patch", "live R4 commit patch differs from approved patch")
    git_evidence = {name: {**probe, "stdout_sha256": hashlib.sha256(probe["stdout"].encode("utf-8")).hexdigest()} for name, probe in probes.items()}
    return {"schema": "compaction-r5-reviewed-source-v2", "status": "admitted", "base_plus_patch": True, "clean_head_claim": False, "r5_delta": list(R5_SOURCE_PATHS), "r4_commit": R4_COMMIT, "r4_patch_sha256": R4_PATCH_SHA256, "live_git_verified": True, "git_evidence": git_evidence}


class IntegrityGuard:
    """Hash all protected input classes before and after every child process."""

    def __init__(self, source_root: Path, classes: Mapping[str, Mapping[str, str]]):
        require(set(classes) == {"source", "fixture", "toolchain", "script", "exe", "run_matrix"}, "integrity-classes", "exact protected input classes required")
        self.source_root = source_root
        self.classes = {name: dict(paths) for name, paths in classes.items()}
        self.before = self.verify()
        self.runtime_receipts: list[dict[str, Any]] = []

    def register_runtime_protected(self, artifact_root: Path, inputs: Mapping[Path, str]) -> dict[str, Any]:
        entries: dict[str, Any] = {}
        for path, expected in inputs.items():
            resolved = path.resolve()
            expected = sha256_text(expected, "runtime-protected-hash")
            require(resolved.is_relative_to(artifact_root.resolve()) and str(resolved) not in self.classes["fixture"], "runtime-protected-input", "runtime protected input must be a unique artifact-local file")
            require(file_identity(resolved)["sha256"] == expected, "runtime-protected-hash", str(resolved))
            self.classes["fixture"][str(resolved)] = expected
            entries[str(resolved)] = file_identity(resolved)
        self.before = self.verify()
        receipt = {"artifact_root": str(artifact_root.resolve()), "inputs": entries}
        self.runtime_receipts.append(receipt)
        return receipt

    def is_runtime_protected(self, path: Path, expected: str) -> bool:
        resolved = path.resolve()
        return any(entries["inputs"].get(str(resolved), {}).get("sha256") == expected for entries in self.runtime_receipts)

    def verify(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, expected in self.classes.items():
            result[name] = file_map_identities(self.source_root, expected)
        return result

    def after(self, before: Mapping[str, Any]) -> dict[str, Any]:
        after = self.verify()
        require(after == before, "postprocess-hash-drift", "source/fixture/toolchain/script/exe/run-matrix bytes changed")
        return after


@dataclasses.dataclass(frozen=True)
class ProcessSpec:
    stage: str
    run_id: str
    argv: tuple[str, ...]
    work: Path
    out: Path
    stdout: Path
    stderr: Path
    expected_exit: int = 0
    timeout_secs: int = 900
    allowlisted_env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    pin_affinity: bool = False
    nested_wall: bool = False
    expected_running_image: Path | None = None
    profile_root: Path | None = None
    create_profile: bool = False
    profile_id: str | None = None
    existing_output_sha256: str | None = None


@dataclasses.dataclass
class ProcessReceipt:
    spec: ProcessSpec
    pid: int
    exit_code: int
    running_image: str
    before_hashes: dict[str, Any]
    after_hashes: dict[str, Any]
    telemetry: dict[str, Any]
    affinity: dict[str, Any] | None
    started_utc: str
    ended_utc: str
    inventory: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    profile: Mapping[str, Any] | None = None


class TelemetryAdapter(Protocol):
    def collect(self, phase: str, output: Path) -> dict[str, Any]: ...


class DescendantObserver(Protocol):
    def observe(self, process: subprocess.Popen[bytes], expected_image: Path, output: Path) -> dict[str, Any]: ...


class ProcessAdapter(Protocol):
    def run(self, spec: ProcessSpec) -> ProcessReceipt: ...



if os.name == "nt":
    import ctypes


class CommandTelemetry:
    """Read-only host telemetry probes; unavailable counters stay null."""

    LOAD_SCRIPT = (
        "$os=Get-CimInstance Win32_OperatingSystem; "
        "$sys=Get-CimInstance Win32_PerfFormattedData_PerfOS_System; "
        "$cpu=Get-CimInstance Win32_Processor | Select-Object -First 1; "
        "$cores=@(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | "
        "Where-Object {$_.Name -ne '_Total'} | ForEach-Object {[PSCustomObject]@{name=$_.Name;cpu_total_percent=$_.PercentProcessorTime}}); "
        "[PSCustomObject]@{cpu_total_percent=$sys.PercentProcessorTime;"
        "processor_frequency=([double]$cpu.CurrentClockSpeed*1000000);"
        "processor_queue=$sys.ProcessorQueueLength;per_logical_processor=$cores;"
        "available_memory_bytes=([double]$os.FreePhysicalMemory*1024)} | ConvertTo-Json -Depth 4 -Compress"
    )

    @staticmethod
    def _probe(argv: Sequence[str]) -> tuple[Any, str | None, dict[str, Any]]:
        require(tuple(argv) in {("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", CommandTelemetry.LOAD_SCRIPT), ("powercfg.exe", "/getactivescheme")}, "command-catalog", "telemetry probe is outside the supported fixed argv catalog")
        try:
            result = subprocess.run(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
            receipt = {"argv": list(argv), "exit_code": result.returncode, "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(), "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(), "utc": utc_now()}
            if result.returncode != 0:
                return None, f"probe exit {result.returncode}: {result.stderr.decode(errors='replace')[:300]}", receipt
            return result.stdout, None, receipt
        except (OSError, subprocess.TimeoutExpired) as error:
            return None, str(error), {"argv": list(argv), "error": str(error), "utc": utc_now()}

    def collect(self, phase: str, output: Path) -> dict[str, Any]:
        raw, error, load_receipt = self._probe(("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", self.LOAD_SCRIPT))
        load = None
        if raw is not None:
            try:
                load = json.loads(raw.decode("utf-8-sig"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (UnicodeError, json.JSONDecodeError, ValueError) as parse_error:
                error = f"telemetry JSON: {parse_error}"
        # powercfg /getactivescheme is read-only. No power mutation occurs.
        power_raw, power_error, power_receipt = self._probe(("powercfg.exe", "/getactivescheme"))
        block: dict[str, Any] = {}
        for field in TELEMETRY_FIELDS:
            if field == "active_power_scheme":
                if power_raw is not None:
                    value = power_raw.decode("utf-8-sig", errors="strict").strip()
                    block[field] = {"value": value or None, "unavailable_reason": None if value else "empty powercfg output"}
                else:
                    block[field] = {"value": None, "unavailable_reason": power_error or "power probe unavailable"}
                continue
            value = load.get(field) if isinstance(load, dict) else None
            reason = None if value is not None else (error or f"missing telemetry field {field}")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
                value, reason = None, f"nonfinite {field}"
            block[field] = {"value": value if error is None or field == "per_logical_processor" else None, "unavailable_reason": reason if value is not None else reason}
        evidence = {"phase": phase, "block": block, "probe_receipts": {"load": load_receipt, "power": power_receipt}, "provider_network": "none"}
        write_new_json(output, evidence)
        return block


if os.name == "nt":
    class WindowsDescendants:
        """Observe running production wall children without changing their mask."""

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = (
                ("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong),
                ("th32ProcessID", ctypes.c_ulong), ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", ctypes.c_ulong), ("cntThreads", ctypes.c_ulong),
                ("th32ParentProcessID", ctypes.c_ulong), ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.c_ulong), ("szExeFile", ctypes.c_wchar * 260),
            )

        def __init__(self, timeout_secs: float = 30.0):
            self.timeout_secs = timeout_secs

        @staticmethod
        def _mask_and_image(pid: int) -> tuple[str, str]:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                raise Reject("wall-descendant-open", f"PID {pid}: {ctypes.get_last_error()}")
            try:
                process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
                if not kernel32.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
                    raise Reject("affinity-mask", f"PID {pid}: GetProcessAffinityMask failed")
                size = ctypes.c_ulong(32768)
                buffer = ctypes.create_unicode_buffer(size.value)
                if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                    raise Reject("running-exe-identity", f"PID {pid}: image query failed")
                return hex(process_mask.value), buffer.value
            finally:
                kernel32.CloseHandle(handle)

        def _children(self, parent_pid: int) -> list[int]:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
            kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
            kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
            if snapshot == ctypes.c_size_t(-1).value:
                raise Reject("wall-descendant-snapshot", str(ctypes.get_last_error()))
            rows = []
            try:
                entry = self.PROCESSENTRY32W()
                entry.dwSize = ctypes.sizeof(entry)
                if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                    while True:
                        if entry.th32ParentProcessID == parent_pid:
                            rows.append(entry.th32ProcessID)
                        if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                            break
                return rows
            finally:
                kernel32.CloseHandle(snapshot)

        def observe(self, process: subprocess.Popen[bytes], expected_image: Path, output: Path) -> dict[str, Any]:
            expected = os.path.normcase(str(expected_image.resolve(strict=True)))
            seen: dict[int, dict[str, Any]] = {}
            deadline = time.monotonic() + self.timeout_secs
            while process.poll() is None and time.monotonic() < deadline:
                for pid in self._children(process.pid):
                    if pid in seen:
                        continue
                    mask, image = self._mask_and_image(pid)
                    if os.path.normcase(str(Path(image).resolve(strict=True))) == expected:
                        seen[pid] = {"pid": pid, "role": "production", "observed_mask": mask, "running_image": image}
                if seen:
                    break
                time.sleep(0.01)
            evidence = {"children": list(seen.values()), "observation": "CreateToolhelp32Snapshot direct-child scan, OpenProcess image/mask read only", "wrapper_pid": process.pid}
            write_new_json(output, evidence)
            if not seen:
                raise Reject("wall-descendant-identity", "no production wall child observed before timeout")
            return evidence


def validated_allowlisted_env(explicit: Mapping[str, str]) -> dict[str, str]:
    require(isinstance(explicit, Mapping), "environment-allowlist", "mapping required")
    result = {name: os.environ[name] for name in SAFE_INHERITED_ENV if name in os.environ and not SECRET_NAME.search(name)}
    for name, value in explicit.items():
        require(isinstance(name, str) and re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) is not None and not SECRET_NAME.search(name), "environment-allowlist", f"unsafe variable {name!r}")
        require(isinstance(value, str), "environment-allowlist", name)
        result[name] = value
    return result


def profile_inventory(root: Path) -> dict[str, Any]:
    require(root.is_dir(), "profile-isolation", f"missing isolated profile root {root}")
    result: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "profile-isolation", f"symlink or reparse-like alias not admitted: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = file_identity(path)
    return result


class SubprocessAdapter:
    """Real adapter, dormant until a later separately authorized campaign call."""

    def __init__(self, guard: IntegrityGuard, telemetry: TelemetryAdapter, descendants: DescendantObserver):
        self.guard = guard
        self.telemetry = telemetry
        self.descendants = descendants
        self.seen_pids: set[int] = set()
        self.seen_outputs: set[str] = set()

    def _prepare_profile(self, spec: ProcessSpec) -> dict[str, Any] | None:
        if spec.profile_root is None:
            return None
        profile_id = spec.profile_id
        require(profile_id in {"focused99", "charcount15"}, "profile-isolation", "unknown isolated profile identity")
        profile = spec.profile_root.resolve()
        marker = profile / ".r5-profile.json"
        if spec.create_profile:
            require(not profile.exists(), "profile-collision", str(profile))
            profile.mkdir(parents=True, exist_ok=False)
            write_new_json(marker, {"schema": "compaction-r5-isolated-profile-v1", "profile_id": profile_id, "root": str(profile), "provider_network": "none", "mutable_scope": "test-profile-only"})
        require(profile.is_dir() and marker.is_file(), "profile-isolation", "isolated profile must be created by the first focused process")
        metadata = load_json(marker)
        exact_object(metadata, ("schema", "profile_id", "root", "provider_network", "mutable_scope"), "profile-identity")
        require(metadata["schema"] == "compaction-r5-isolated-profile-v1" and metadata["profile_id"] == profile_id and Path(metadata["root"]) == profile and metadata["provider_network"] == "none" and metadata["mutable_scope"] == "test-profile-only", "profile-identity", str(profile))
        marker_hash = file_identity(marker)["sha256"]
        before = profile_inventory(profile)
        require(marker.relative_to(profile).as_posix() in {Path(key).as_posix() for key in before}, "profile-isolation", "protected marker missing from before inventory")
        return {"root": profile, "marker": marker, "marker_sha256": marker_hash, "before": before}

    def _finish_profile(self, context: dict[str, Any] | None) -> dict[str, Any] | None:
        if context is None:
            return None
        after = profile_inventory(context["root"])
        after_hash = file_identity(context["marker"])["sha256"]
        require(after_hash == context["marker_sha256"], "profile-hash-drift", "isolated profile protected marker changed")
        return {"root": str(context["root"]), "protected_marker": file_identity(context["marker"]), "before": context["before"], "after": after, "allowed_mutable_scope": "inside isolated profile root only; glob_both_snapshots_retained"}

    def run(self, spec: ProcessSpec) -> ProcessReceipt:
        run_root = spec.work.parent
        require(all(path.is_relative_to(run_root) for path in (spec.work, spec.out, spec.stdout, spec.stderr)), "process-output-layout", spec.run_id)
        if run_root.exists():
            require(spec.existing_output_sha256 is not None, "output-collision", f"pre-existing run root {run_root}")
            existing_files = [path for path in run_root.rglob("*") if path.is_file()]
            require(existing_files == [spec.out], "output-collision", "only the pre-registered protected collision output may pre-exist")
        for path in (spec.work, spec.out, spec.stdout, spec.stderr):
            key = os.path.normcase(str(path.resolve(strict=False)))
            require(key not in self.seen_outputs, "output-collision", str(path))
            if path.exists():
                allowed_existing = path == spec.out and spec.existing_output_sha256 is not None and file_identity(path)["sha256"] == spec.existing_output_sha256 and self.guard.is_runtime_protected(path, spec.existing_output_sha256)
                require(allowed_existing, "output-collision", str(path))
            self.seen_outputs.add(key)
        before = self.guard.verify()
        require(before == self.guard.before, "preprocess-hash-drift", spec.run_id)
        env = validated_allowlisted_env(spec.allowlisted_env)
        if spec.profile_root is not None:
            require(spec.profile_root.is_relative_to(self.guard.source_root.parent), "profile-isolation", "profile must stay under the campaign workspace")
            require(env.get("PRIME_AGENT_CODING_AGENT_DIR") == str(spec.profile_root.resolve()), "profile-isolation", "required PRIME profile environment missing or mismatched")
        profile_context = self._prepare_profile(spec)
        spec.work.mkdir(parents=True, exist_ok=False)
        telemetry_before = self.telemetry.collect("before", spec.work / "telemetry-before.json")
        stdout_handle = write_new_bytes(spec.stdout, b"")
        stderr_handle = write_new_bytes(spec.stderr, b"")
        started = utc_now()
        process: subprocess.Popen[bytes] | None = None
        affinity: dict[str, Any] | None = None
        try:
            with spec.stdout.open("r+b") as stdout, spec.stderr.open("r+b") as stderr:
                process = subprocess.Popen(list(spec.argv), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env)
                require(process.pid not in self.seen_pids, "pid-collision", str(process.pid))
                self.seen_pids.add(process.pid)
                running_image = query_process_image(process)
                validate_launch_image(spec, running_image)
                if spec.pin_affinity:
                    direct = set_child_affinity(process, MASK)
                    children = [{"pid": process.pid, "role": "wrapper" if spec.nested_wall else "production", "observed_mask": direct["observed_mask"], "running_image": running_image}]
                    affinity = {"requested_mask": hex(MASK), "children": children}
                    if spec.nested_wall:
                        require(spec.expected_running_image is not None, "wall-running-image", "production wall image required")
                        nested = self.descendants.observe(process, spec.expected_running_image, spec.work / "descendants.json")
                        affinity["children"].extend(nested["children"])
                    validate_affinity(affinity, nested_wall=spec.nested_wall)
                try:
                    exit_code = process.wait(timeout=spec.timeout_secs)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise Reject("process-timeout", spec.run_id)
        except Exception:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            raise
        ended = utc_now()
        after = self.guard.after(before)
        telemetry_after = self.telemetry.collect("after", spec.work / "telemetry-after.json")
        profile_evidence = self._finish_profile(profile_context)
        telemetry = validate_telemetry({"before": telemetry_before, "after": telemetry_after, "observation": dict(OBSERVATION_STATES), "required_fields": list(TELEMETRY_FIELDS)})
        inventory = {str(path.relative_to(run_root)): file_identity(path) for path in sorted(run_root.rglob("*")) if path.is_file()}
        if spec.expected_exit >= 0:
            require(exit_code == spec.expected_exit, "process-exit", f"{spec.run_id}: got {exit_code}, expected {spec.expected_exit}")
        return ProcessReceipt(spec, process.pid, exit_code, running_image, before, after, telemetry, affinity, started, ended, inventory, profile_evidence)


def replay_command(exe: Path, fixture: Path, work: Path, out: Path, summary_calls: int, *, failure: bool = False, anchor: bool = False) -> tuple[str, ...]:
    args = [str(exe), "replay", "--fixture", str(fixture), "--context-window", "20000" if anchor else "1000000", "--provider-delay-ms", "0", "--timeout-secs", "90" if anchor else "900", "--work", str(work), "--out", str(out), "--expect-outcome", "summary-failure" if failure else "successful-compaction", "--expect-summary-calls", str(summary_calls)]
    if failure:
        args.extend(["--summary-failure", "R1 intentional summary failure"])
    return tuple(args)


def r4_lane_command(exe: Path, lane: str, fixture: Path, work: Path, out: Path, *, unavailable_allocation: bool = False, inject: str = "none") -> tuple[str, ...]:
    require(lane in {"allocation", "replay-phase", "minimal-provider"}, "lane-cli", lane)
    args = [str(exe), "r4-lane", "--lane", lane, "--fixture", str(fixture), "--work", str(work), "--out", str(out)]
    if lane == "allocation":
        args.extend(["--allocation-source", "unavailable" if unavailable_allocation else "counting-global"])
    else:
        args.extend(["--mock-wait-ms", "25" if lane == "minimal-provider" else "0", "--generation-units", "3" if lane == "minimal-provider" else "1", "--inject", inject])
    return tuple(args)


def wall_command(python: Path, script: Path, wall_exe: Path, fixture: Path, work: Path, child_out: Path, out: Path) -> tuple[str, ...]:
    return (str(python), str(script), "wall", "--exe", str(wall_exe), "--fixture", str(fixture), "--work", str(work), "--child-out", str(child_out), "--out", str(out))


def bench_command(exe: Path, fixture: Path, out: Path) -> tuple[str, ...]:
    return (str(exe), "bench-local", "--fixture", str(fixture), "--warmup", "10", "--iters", "30", "--chunk-context-window", "32768", "--out", str(out))


def no_gate_a_timing_flags(argv: Sequence[str]) -> None:
    forbidden = {"--timing", "--benchmark", "--paired", "--candidate", "--baseline-exe"}
    require(not any(arg in forbidden or arg.startswith("--timing-") for arg in argv), "gate-a-timing-exclusion", "timing flag in Gate A")


def parse_rust_test_log(text: str, expected_names: Sequence[str], expected_suites: int, expected_passed: int, expected_filtered: int | None = None) -> dict[str, Any]:
    names: list[str] = []
    suite_results: list[int] = []
    filtered_total = 0
    pattern = re.compile(r"^test (\S+) \.\.\. ok$", re.M)
    names.extend(pattern.findall(text))
    result_pattern = re.compile(r"^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; \d+ measured; (\d+) filtered out;", re.M)
    for passed, failed, ignored, filtered in result_pattern.findall(text):
        require(int(failed) == 0 and int(ignored) == 0, "focused-test-result", "failed/ignored tests are not admitted")
        suite_results.append(int(passed))
        filtered_total += int(filtered)
    require(len(suite_results) == expected_suites and sum(suite_results) == expected_passed and names == list(expected_names), "focused-test-matrix", "suite count/count/ordered names differ")
    if expected_filtered is not None:
        require(filtered_total == expected_filtered, "focused-test-filtered", str(filtered_total))
    return {"schema": "compaction-r5-focused-tests-v1", "status": "ok", "suites": expected_suites, "passed": expected_passed, "failed": 0, "ignored": 0, "ordered_names": names, "filtered_out": filtered_total}




INVALID_CLI_SPECS = (
    ("iters-zero", "cli-iters", "bench-local", (("--iters", "0"),)),
    ("warmup-zero", "cli-warmup", "bench-local", (("--warmup", "0"),)),
    ("iters-fractional", "cli-iters", "bench-local", (("--iters", "1.5"),)),
    ("chunk-nan", "cli-chunk-context-window", "bench-local", (("--chunk-context-window", "NaN"),)),
    ("replay-context-zero", "cli-context-window", "replay", (("--context-window", "0"),)),
    ("replay-delay-negative", "cli-provider-delay-ms", "replay", (("--provider-delay-ms", "-1"),)),
    ("replay-timeout-fractional", "cli-timeout-secs", "replay", (("--timeout-secs", "1.5"),)),
    ("replay-summary-calls-zero", "cli-expect-summary-calls", "replay", (("--expect-summary-calls", "0"),)),
    ("duplicate-option", "cli-parse", "bench-local", (("--iters", "1"), ("--iters", "2"))),
)


FOCUSED_TEST_SUITES = (
    "compaction_parity_suite", "compaction_kernel_parity", "compaction_recovery",
    "compaction_turn_flow", "compaction_parallel_fallback", "compaction_length_recovery",
    "compaction_chat_length_recovery", "compaction_anthropic_length_recovery",
    "compaction_claude_refusal", "compaction_summary_guard", "compaction_observability",
    "compaction_fidelity", "compaction_summary_path", "compaction_model_threshold",
)


R4_PROVENANCE_LOGS = {
    "focused99": ARTIFACT_ROOT.parent / "optimus-compaction-r4-artifacts/logs/focused-correctness-final.log",
    "charcount15": ARTIFACT_ROOT.parent / "optimus-compaction-r4-artifacts/logs/charcount-final.log",
}
R4_TEST_BINARY_ROOT = ARTIFACT_ROOT.parent / "optimus-compaction-r4-target/instrumented/release/deps"
CHARCOUNT_FILTER = "core::compaction::charcount::tests::"


def invalid_cli_command(exe: Path, fixture: Path, out: Path, name: str) -> tuple[tuple[str, ...], str]:
    matches = [row for row in INVALID_CLI_SPECS if row[0] == name]
    require(len(matches) == 1, "invalid-cli-plan", name)
    _, stage, subcommand, options = matches[0]
    argv = [str(exe), subcommand, "--fixture", str(fixture), "--out", str(out)] + [item for option in options for item in option]
    if subcommand == "replay" and name != "replay-summary-calls-zero":
        argv.extend(["--expect-summary-calls", "2"])
    return tuple(argv), stage


def summarize_warm_processes(records: Sequence[Mapping[str, Any]], case_name: str) -> dict[str, Any]:
    require(len(records) == 30, "baseline-observation-count", "30 warm fresh processes required")
    universal = set(WARM_KERNELS)
    optional = "jev_prepare_context"
    pids: set[int] = set()
    identities: set[str] = set()
    metric_rows: dict[str, list[float]] = {}
    jev_states: dict[str, int] = {"prepared": 0, "expected-skip": 0}
    for index, record in enumerate(records):
        require(record.get("status") == "admitted" and record.get("lane") == "bench-local-warm" and record.get("case") == case_name, "baseline-run-admission", str(index))
        require(record.get("observation") == index and record.get("fresh_processes") == 1, "baseline-run-order", str(index))
        pid = strict_int(record.get("pid"), "baseline-pid", 1)
        require(pid not in pids, "pid-collision", str(pid))
        pids.add(pid)
        identities.add(sha256_text(record.get("identity_sha256"), "baseline-identity"))
        require(record.get("warmup") == 10 and record.get("iterations") == 30, "warm-output-config", str(index))
        state = record.get("jev_state")
        require(state in jev_states, "warm-jev-state", "prepared or expected-skip required")
        jev_states[state] += 1
        values = record.get("values")
        require(isinstance(values, dict), "warm-outer-values", "typed medians object required")
        expected = universal | ({optional} if state == "prepared" else set())
        require(set(values) == expected, "warm-outer-keys", f"prepared state {state} and metric coverage disagree")
        for name, value in values.items():
            metric_rows.setdefault(name, []).append(finite_positive(value, "timing-numeric"))
    require(len(identities) == 1, "baseline-identity", "identity drift")
    for name in universal:
        require(len(metric_rows.get(name, [])) == 30, "baseline-observation-count", f"universal kernel {name} needs exactly 30 outer p50 values")
    require(set(metric_rows) == universal | ({optional} if jev_states["prepared"] else set()), "warm-outer-keys", "unexpected optional medians")
    stats: dict[str, Any] = {}
    for name, values in sorted(metric_rows.items()):
        ordered = sorted(values)
        stats[name] = {"n": len(values), "unit": "stats.p50_ms from each prepared fresh warmup10/iters30 process", "min": ordered[0], "p50": statistics.median(ordered), "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], "max": ordered[-1]}
    if optional not in stats:
        stats[optional] = {"status": "all-expected-skip", "n_prepared": 0, "n_expected_skip": 30, "numeric_comparison_supported": False}
    return {"schema": "compaction-r5-warm-stats-v2", "status": "ok", "case": case_name, "lane": "bench-local-warm", "raw_observations": records, "statistics": stats, "jev_states": jev_states, "decision_evidence": False}


R4_RUST_REPLAY_RELATIVE = "crates/pi-coding-agent/examples/compaction_replay.rs"
R4_RUST_CREATE_NEW_FORMAT = '.map_err(|error| format!("create_new {}: {error}", path.display()))?;'
R4_RUST_MAIN_ERROR_BLOCK = 'Err(error) => {\n            eprintln!("error: {error}");\n            std::process::exit(1);\n        }'


def validate_r4_collision_rejection(*, exit_code: int, stderr: str, protected_output: Path, before: Mapping[str, Any], after: Mapping[str, Any], produced_new_output: Any, dependent_statistics: Any, attempt_count: int, rust_source: str) -> dict[str, Any]:
    """Validate actual R4 no-clobber evidence, not a made-up CLI error literal."""
    require(type(exit_code) is int and exit_code == 1, "r4-collision-exit", "actual compiled R4 main error exit 1 required")
    require(produced_new_output is None, "r4-collision-new-output", "collision must not create a dependent output document")
    require(dependent_statistics is None or dependent_statistics == [] or dependent_statistics == {}, "r4-collision-dependent-statistics", "collision must not produce dependent statistics")
    require(attempt_count == 1, "r4-collision-attempt-count", "one actual no-clobber CLI invocation; retries and replacements forbidden")
    expected_identity_path = str(protected_output.resolve())
    require(before.get("path") == expected_identity_path, "r4-collision-protected-path", "before identity must be the exact protected output")
    require(after == before, "r4-output-unchanged", "protected pre-existing output identity or bytes changed")
    require(isinstance(stderr, str), "r4-collision-stderr", "strictly decoded stderr required")
    require(rust_source.count(R4_RUST_CREATE_NEW_FORMAT) == 1 and rust_source.count(R4_RUST_MAIN_ERROR_BLOCK) == 1, "r4-rust-contract", "frozen R4 writer/main error expressions differ")
    # The terminal newline is produced by eprintln!. No other stderr text is admitted.
    line = stderr.removesuffix("\r\n").removesuffix("\n")
    require("\n" not in line and "\r" not in line, "r4-collision-stderr", "exactly one main error line required")
    expected_prefix = f"error: create_new {protected_output}: "
    require(line.startswith(expected_prefix), "r4-collision-stderr", f"expected main error prefix and create_new exact protected output: {expected_prefix}")
    os_error = line[len(expected_prefix):]
    require(bool(os_error.strip()), "r4-collision-os-error", "nonempty OS error text required after exact protected output")
    return {
        "rejected": True,
        "intended_stage": "create_new",
        "main_error_prefix": "error: ",
        "create_new_prefix": "create_new ",
        "protected_output_path": str(protected_output),
        "os_error_text": os_error,
        "exit_code": exit_code,
        "output_unchanged": True,
        "output_before": dict(before),
        "output_after": dict(after),
        "produced_new_output": None,
        "dependent_statistics": dependent_statistics,
        "attempt_count": attempt_count,
        "rust_source_contract": {
            "path": R4_RUST_REPLAY_RELATIVE,
            "utf8_text_sha256": hashlib.sha256(rust_source.encode("utf-8")).hexdigest(),
            "writer_expression": R4_RUST_CREATE_NEW_FORMAT,
            "main_error_expression": R4_RUST_MAIN_ERROR_BLOCK,
        },
    }


class CampaignExecutor:
    """Real campaign orchestration; constructed only with an explicit real adapter."""

    def __init__(self, adapter: ProcessAdapter, artifact_root: Path, fixture_root: Path, repo_root: Path, instrumented_exe: Path, wall_exe: Path, r3_exe: Path, *, git_runner: Any = None):
        self.adapter = adapter
        self.root = artifact_root
        self.fixture_root = fixture_root
        self.instrumented_exe = instrumented_exe
        self.wall_exe = wall_exe
        self.r3_exe = r3_exe
        self.repo_root = repo_root
        self.git_runner = git_runner
        self.focused_executables: dict[Path, dict[str, Any]] = {}
        self.all_pids: set[int] = set()
        self.commands: list[dict[str, Any]] = []

    def _spec(self, stage: str, run_id: str, argv: Sequence[str], *, expected_exit: int = 0, pin: bool = False, nested_wall: bool = False, running_image: Path | None = None, out_path: Path | None = None, profile_root: Path | None = None, create_profile: bool = False, profile_id: str | None = None, existing_output_sha256: str | None = None) -> ProcessSpec:
        run_root = self.root / stage / run_id
        out = out_path if out_path is not None else run_root / "out.json"
        env = dict(self.focused_executables[Path(argv[0]).resolve()]["env"]) if profile_root is not None and Path(argv[0]).resolve() in self.focused_executables else {}
        if profile_root is not None:
            env["PRIME_AGENT_CODING_AGENT_DIR"] = str(profile_root.resolve())
        if Path(argv[0]).resolve() == Path(sys.executable).resolve():
            env["PYTHONDONTWRITEBYTECODE"] = "1"
        return ProcessSpec(stage, run_id, tuple(str(item) for item in argv), run_root / "work", out, run_root / "stdout.bin", run_root / "stderr.bin", allowlisted_env=env, expected_exit=expected_exit, pin_affinity=pin, nested_wall=nested_wall, expected_running_image=running_image, profile_root=profile_root, create_profile=create_profile, profile_id=profile_id, existing_output_sha256=existing_output_sha256)

    def register_focused_plan(self, focused: Mapping[str, Any]) -> None:
        require(isinstance(focused, dict) and set(focused) == {"focused99", "charcount15"}, "focused-command-catalog", "exact focused and charcount plans required")
        self.focused_executables = {}
        for kind, suite_count, passed, filtered in (("focused99", 14, 99, None), ("charcount15", 1, 15, 4499)):
            spec = focused[kind]
            require(isinstance(spec, dict) and set(spec) == {"expected_names", "provenance_sha256", "binaries"}, "focused-command-catalog", "direct binary identity/profile plan fields required")
            log_path = R4_PROVENANCE_LOGS[kind]
            require(sha256_text(spec["provenance_sha256"], "focused-provenance") == file_identity(log_path)["sha256"], "focused-provenance", kind)
            log_text = log_path.read_text(encoding="utf-8")
            expected_names = [match.group(1) for match in (re.fullmatch(r"test (.+) \.\.\. ok", line) for line in log_text.splitlines()) if match]
            require(len(expected_names) == passed and spec["expected_names"] == expected_names, "focused-provenance", "expected names are not the frozen R4 ordered names")
            parse_rust_test_log(log_text, expected_names, suite_count, passed, filtered)
            launches = []
            for line in log_text.splitlines():
                match = re.search(r"Running (?:unittests )?([^\s]+) \(([^)]+\.exe)\)", line)
                if not match:
                    continue
                source_name, reported_exe = match.group(1), Path(match.group(2))
                if kind == "focused99" and source_name.startswith("tests" + chr(92)):
                    launches.append((Path(source_name).stem, reported_exe))
                elif kind == "charcount15" and source_name.replace("/", chr(92)) == "src" + chr(92) + "lib.rs":
                    launches.append(("pi_coding_agent", reported_exe))
            require(len(launches) == suite_count and isinstance(spec["binaries"], list) and len(spec["binaries"]) == suite_count, "focused-provenance", f"wrong direct test binary count {kind}: {len(launches)} expected {suite_count}; launches={launches!r}")
            require({suite for suite, _ in launches} == (set(FOCUSED_TEST_SUITES) if kind == "focused99" else {"pi_coding_agent"}), "focused-provenance", "suite paths differ from frozen R4 commands")
            profile_root = (self.root / "gate-a/profiles" / kind).resolve()
            for index, (suite, relative_exe) in enumerate(launches):
                exe = (self.root.parent / relative_exe).resolve()
                require(exe.parent == R4_TEST_BINARY_ROOT.resolve() and exe.is_file(), "focused-binary-path", str(exe))
                entry = spec["binaries"][index]
                exact_object(entry, ("exe", "sha256"), "focused-binary-identity")
                actual_hash = file_identity(exe)["sha256"]
                require(Path(entry["exe"]).resolve() == exe and entry["sha256"] == actual_hash, "focused-binary-hash", f"{kind} binary path/hash differs")
                require(re.fullmatch(re.escape(suite) + r"-[0-9a-f]{16}\.exe", exe.name), "focused-binary-path", exe.name)
                argv = (str(exe), "--test-threads=1") if kind == "focused99" else (str(exe), CHARCOUNT_FILTER, "--test-threads=1")
                require(exe not in self.focused_executables, "focused-binary-collision", str(exe))
                self.focused_executables[exe] = {"kind": kind, "index": index, "argv": argv, "sha256": actual_hash, "profile_root": profile_root, "profile_id": kind, "env": {"PRIME_AGENT_CODING_AGENT_DIR": str(profile_root)}, "suite": suite, "provenance_log": file_identity(log_path)}

    def validate_external_argv(self, argv: Sequence[str], *, spec: ProcessSpec | None = None) -> None:
        """Every child argv must match the fixed supported CLI catalog."""
        require(len(argv) >= 2 and all(isinstance(item, str) and item for item in argv), "command-catalog", "nonempty string argv required")
        executable = Path(argv[0]).resolve()
        if executable in {self.instrumented_exe.resolve(), self.wall_exe.resolve(), self.r3_exe.resolve()}:
            subcommands = command_catalog()["replay"]
            subcommand = argv[1]
            require(subcommand in subcommands, "command-catalog", f"unsupported replay subcommand {subcommand!r}")
            options = {
                "self-test-r1": {"--work", "--out"},
                "self-test-r2": {"--fixture", "--work", "--out"},
                "replay": {"--fixture", "--context-window", "--provider-delay-ms", "--timeout-secs", "--work", "--out", "--expect-outcome", "--expect-summary-calls", "--summary-failure"},
                "verify-reopen": {"--fixture", "--out"},
                "verify-local": {"--fixture", "--out", "--expectation"},
                "bench-local": {"--fixture", "--warmup", "--iters", "--chunk-context-window", "--out"},
                "r4-lane": {"--lane", "--fixture", "--work", "--out", "--allocation-source", "--mock-wait-ms", "--generation-units", "--inject"},
            }[subcommand]
            tail = list(argv[2:])
            require(len(tail) % 2 == 0, "command-catalog", "replay options must be explicit key/value pairs")
            require(all(tail[index] in options for index in range(0, len(tail), 2)), "command-catalog", f"unsupported {subcommand} option in {tail!r}")
            path_options = {"--fixture", "--work", "--out", "--expectation", "--child-out"}
            for index in range(0, len(tail), 2):
                key, value = tail[index], tail[index+1]
                require(value is not None, "command-catalog", key)
                if key == "--fixture":
                    fixture = Path(value).resolve()
                    require(fixture == self.fixture_root.resolve() or fixture.is_relative_to(self.fixture_root.resolve()), "command-catalog", "fixture escapes protected root")
                if key in {"--work", "--out", "--child-out"}:
                    target = Path(value).resolve()
                    require(target.is_relative_to(self.root.resolve()), "command-catalog", f"{key} escapes artifact root")
            return
        if executable == Path(sys.executable).resolve():
            script = Path(argv[1]).resolve()
            require(script == (self.repo_root / "scripts" / script.name).resolve() and script.name in command_catalog()["python"], "command-catalog", "Python helper must be an approved in-repo script")
            tail = list(argv[2:])
            if script.name == "compaction_r4_lanes.py":
                require(tail and tail[0] in {"wall", "validate"}, "command-catalog", "unsupported wall/validate subcommand")
                allowed = {"wall": {"--exe", "--fixture", "--work", "--child-out", "--out"}, "validate": {"--report", "--lane", "--out"}}[tail[0]]
                options = tail[1:]
                require(len(options) % 2 == 0 and all(options[index] in allowed for index in range(0, len(options), 2)), "command-catalog", f"unsupported Python lane options {tail!r}")
                values_list = options[0::2]
                require(len(values_list) == len(set(values_list)), "command-catalog", "duplicate Python options are not admitted")
                values = dict(zip(values_list, options[1::2]))
                require(set(values) == allowed, "command-catalog", f"exact {tail[0]} options required")
                if tail[0] == "wall":
                    require(Path(values["--exe"]).resolve() == self.wall_exe.resolve(), "command-catalog", "wall --exe must be the registered wall executable")
                    require(Path(values["--fixture"]).resolve().is_relative_to(self.fixture_root.resolve()), "command-catalog", "wall fixture escapes protected root")
                else:
                    report_path = Path(values["--report"]).resolve()
                    out_path = Path(values["--out"]).resolve()
                    require(report_path.is_relative_to(self.root.resolve()) and report_path.is_file(), "command-catalog", "validate --report must be an existing file inside the artifact root")
                    require(out_path.is_relative_to(self.root.resolve()), "command-catalog", "validate --out must remain inside the artifact root")
                    require(values["--lane"] in {"allocation", "replay-phase", "minimal-provider", "wall"}, "command-catalog", "validate --lane must be a supported public lane")
                return
            if script.name == "test_compaction_paired_protocol.py":
                require(len(tail) == 1, "command-catalog", "R3 controls accept exactly one artifact output")
                require(Path(tail[0]).resolve().is_relative_to(self.root.resolve()), "command-catalog", "control output escapes artifacts")
                return
            if script.name == "test_compaction_r4_lanes.py":
                require(len(tail) == 1 and Path(tail[0]).resolve().is_relative_to(self.root.resolve()), "command-catalog", "R4 controls accept exactly one artifact output")
                return
            raise Reject("command-catalog", "no generic Python argv adapter is supported")
        if executable in self.focused_executables:
            identity = self.focused_executables[executable]
            require(tuple(argv) == identity["argv"], "focused-command-catalog", "direct test argv must match frozen provenance exactly")
            require(file_identity(executable)["sha256"] == identity["sha256"], "focused-binary-hash", str(executable))
            require(spec is not None and spec.profile_id == identity["profile_id"] and spec.profile_root == identity["profile_root"], "profile-isolation", "focused child requires the registered isolated profile")
            require(spec.allowlisted_env.get("PRIME_AGENT_CODING_AGENT_DIR") == str(identity["profile_root"].resolve()), "profile-isolation", "isolated PRIME profile env missing or mismatched")
            return
        raise Reject("command-catalog", f"unregistered executable {argv[0]!r}")

    def _run(self, stage: str, run_id: str, argv: Sequence[str], *, gate_a: bool = False, expected_exit: int = 0, pin: bool = False, nested_wall: bool = False, running_image: Path | None = None, out_path: Path | None = None, profile_root: Path | None = None, create_profile: bool = False, profile_id: str | None = None, existing_output_sha256: str | None = None) -> tuple[ProcessReceipt, Any]:
        spec = self._spec(stage, run_id, argv, expected_exit=expected_exit, pin=pin, nested_wall=nested_wall, running_image=running_image, out_path=out_path, profile_root=profile_root, create_profile=create_profile, profile_id=profile_id, existing_output_sha256=existing_output_sha256)
        if gate_a:
            no_gate_a_timing_flags(spec.argv)
        try:
            self.validate_external_argv(spec.argv, spec=spec)
            receipt = self.adapter.run(spec)
        except Exception as error:
            failure = {"schema": "compaction-r5-process-failure-v1", "status": "failed", "stage": getattr(error, "stage", "process-adapter"), "error": str(error), "run_id": run_id, "argv": tuple(str(item) for item in argv), "allowlisted_env": dict(spec.allowlisted_env), "retained": {name: file_identity(path) for name, path in (("out", spec.out), ("stdout", spec.stdout), ("stderr", spec.stderr)) if path.is_file()}, "dependent_statistics": None}
            self.commands.append(failure)
            write_new_json(spec.work.parent / "failure.json", failure)
            raise
        pids = {receipt.pid}
        if receipt.affinity:
            pids.update(child["pid"] for child in receipt.affinity["children"])
        require(not (pids & self.all_pids), "pid-collision", f"reused PID in {stage}/{run_id}")
        self.all_pids.update(pids)
        if spec.existing_output_sha256 is not None:
            require(file_identity(spec.out)["sha256"] == spec.existing_output_sha256, "protected-output-drift", "pre-registered collision output changed")
            output = None
            raw = None
        else:
            output = load_json(spec.out) if spec.out.exists() else None
            raw = file_identity(spec.out) if output is not None else None
        row = {"stage": stage, "run_id": run_id, "argv": spec.argv, "allowlisted_env": spec.allowlisted_env, "sequential_adapter_pid": receipt.pid, "exit_code": receipt.exit_code, "started_utc": receipt.started_utc, "ended_utc": receipt.ended_utc, "raw_output": raw, "preexisting_output_sha256": spec.existing_output_sha256, "profile": receipt.profile}
        self.commands.append(row)
        write_new_json(spec.work / "receipt.json", {"schema": "compaction-r5-process-receipt-v1", **row, "before_hashes": receipt.before_hashes, "after_hashes": receipt.after_hashes, "telemetry": receipt.telemetry, "affinity": receipt.affinity, "inventory": receipt.inventory})
        return receipt, output

    def _expected_failure(self, stage: str, name: str, argv: Sequence[str], intended_stage: str) -> dict[str, Any]:
        receipt, output = self._run(stage, name, argv, gate_a=True, expected_exit=-1)
        require(receipt.exit_code != 0 and output is None, "invalid-cli-no-output", name)
        text = receipt.spec.stderr.read_text(encoding="utf-8", errors="strict")
        require(intended_stage in text, "invalid-cli-stage", name)
        return {"name": name, "outcome": "rejected", "intended_stage": intended_stage, "exit_code": receipt.exit_code}


    def _run_jev_unavailable(self, stage: str, run_id: str, argv: Sequence[str], *, expected_error: str) -> dict[str, Any]:
        receipt, output = self._run(stage, run_id, argv, gate_a=True, expected_exit=1)
        require(receipt.exit_code == 1, "jev-unavailable-exit", f"{run_id}: native exit 1 required")
        stdout_bytes = receipt.spec.stdout.read_bytes()
        require(output is None and not receipt.spec.out.exists() and stdout_bytes == b"", "jev-unavailable-output", f"{run_id}: empty stdout and no output report required")
        actual_error = receipt.spec.stderr.read_text(encoding="utf-8", errors="strict").rstrip("\r\n")
        require(actual_error == expected_error, "jev-unavailable-error", f"{run_id}: exact normalized Jev input-limit error required")
        return {
            "stage": stage,
            "run_id": run_id,
            "exe": str(argv[0]),
            "exe_sha256": file_identity(Path(argv[0]))["sha256"],
            "argv": tuple(str(item) for item in argv),
            "argv_sha256": structured_sha256(tuple(str(item) for item in argv)),
            "exit_code": receipt.exit_code,
            "error": actual_error,
            "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "stderr_sha256": hashlib.sha256(receipt.spec.stderr.read_bytes()).hexdigest(),
            "no_output": True,
            "process_receipt_sha256": file_identity(receipt.spec.work / "receipt.json")["sha256"],
        }

    def run_local_parity(self) -> tuple[dict[str, Any], dict[str, Any]]:
        parity_root = self.root / "gate-a/local-parity"
        full21_out = parity_root / "r3-baseline/out.json"
        full21_argv = (self.r3_exe, "verify-local", "--fixture", self.fixture_root, "--out", full21_out)
        unavailable = self._run_jev_unavailable("gate-a/local-parity", "r3-baseline", full21_argv, expected_error=JEV_INPUT_LIMIT_STDERR)
        unavailable.update({
            "schema": "compaction-r5-local-parity-unavailable-v1",
            "status": "unavailable",
            "reason": JEV_INPUT_LIMIT_REASON,
            "identity": "large-ascii-first-in-lexical-full21",
            "partial_parity": False,
        })
        baseline_out = parity_root / "r3-baseline-small/out.json"
        candidate_out = parity_root / "r4-candidate-small/out.json"
        baseline = self._run("gate-a/local-parity", "r3-baseline-small", (self.r3_exe, "verify-local", "--fixture", self.fixture_root / "small-ascii", "--out", baseline_out), gate_a=True)[1]
        candidate = self._run("gate-a/local-parity", "r4-candidate-small", (self.instrumented_exe, "verify-local", "--fixture", self.fixture_root / "small-ascii", "--out", candidate_out, "--expectation", baseline_out), gate_a=True)[1]
        parity = compare_local_parity(baseline, candidate, case_names=("small-ascii",), scope="small-ascii-exact-full14")
        return unavailable, parity

    def run_unavailable_warm_preflight(self) -> list[dict[str, Any]]:
        records = []
        for case_name, history in UNAVAILABLE_WARM_PREFLIGHTS:
            out = self.root / "gate-a/warm-preflight-unavailable" / case_name / "out.json"
            argv = bench_command(self.instrumented_exe, self.fixture_root / case_name, out)
            expected_error = f"error: {case_name}: non-timed preflight failed: [jev-prepare] unexpected production Jev prepare_context error: compaction_input_limit"
            row = self._run_jev_unavailable("gate-a/warm-preflight-unavailable", case_name, argv, expected_error=expected_error)
            row.update({
                "schema": "compaction-r5-warm-preflight-unavailable-v1",
                "status": "unavailable",
                "case": case_name,
                "history": history,
                "reason": JEV_INPUT_LIMIT_REASON,
                "partial_parity": False,
                "fresh_processes": 1,
                "config": {"warmup": 10, "iterations": 30, "chunk_context_window": 32768},
                "timed_observations": [],
                "statistics": None,
                "non_timed": True,
                "timing_executed": False,
            })
            records.append(row)
        return validate_unavailable_warm_preflights(records, self.fixture_root, self.root)


    def run_r4_negative_controls(self) -> dict[str, Any]:
        """Fresh executed R4 lanes/readers, all with synthetic non-decision data."""
        stage = "gate-a/r4-negative"
        fixture = self.fixture_root / "small-ascii"
        accepted: dict[str, Any] = {}
        first_allocation_report = None
        for name, unavailable in (("allocation-available", False), ("allocation-unavailable", True)):
            argv = r4_lane_command(self.instrumented_exe, "allocation", fixture, self.root / stage / name / "work", self.root / stage / name / "out.json", unavailable_allocation=unavailable)
            _, report = self._run(stage, name, argv, gate_a=True)
            if name == "allocation-available":
                first_allocation_report = report
            accepted[name] = extract_one_shot(report, "allocation")
        for name, lane in (("replay-phase", "replay-phase"), ("minimal-provider", "minimal-provider")):
            argv = r4_lane_command(self.instrumented_exe, lane, fixture, self.root / stage / name / "work", self.root / stage / name / "out.json")
            _, report = self._run(stage, name, argv, gate_a=True)
            accepted[name] = extract_one_shot(report, lane)
        for name, injection in (("provider-failure", "failure"), ("provider-cancelled", "cancelled")):
            argv = r4_lane_command(self.instrumented_exe, "minimal-provider", fixture, self.root / stage / name / "work", self.root / stage / name / "out.json", inject=injection)
            _, report = self._run(stage, name, argv, gate_a=True)
            validate_lane_report(report, "minimal-provider", "instrumented")
            require(report["outcome"]["status"] == injection and report["continuation"]["status"] == "unavailable", "r4-negative-outcome", name)
            require(report["provider"]["call_count"] == (0 if injection == "cancelled" else 1), "r4-negative-provider-calls", name)
            accepted[name] = {"outcome": injection, "provider_calls": report["provider"]["call_count"], "continuation": report["continuation"]["status"]}
        name = "external-wall"
        out = self.root / stage / name / "out.json"
        argv = wall_command(Path(sys.executable), self.repo_root / "scripts/compaction_r4_lanes.py", self.wall_exe, fixture, out.parent / "work", out.parent / "child.json", out)
        _, report = self._run(stage, name, argv, gate_a=True, nested_wall=True, running_image=self.wall_exe)
        accepted[name] = extract_one_shot(report, "wall")

        negative_specs: list[tuple[str, tuple[str, ...], str, int]] = []
        for name, exe, args, intended, expected_code in (
            ("instrumented-wall-rejected", self.instrumented_exe, ("wall",), "r4-build-lane-identity", 1),
            ("wall-allocation-rejected", self.wall_exe, ("allocation",), "r4-build-lane-identity", 1),
            ("provider-control-in-allocation-rejected", self.instrumented_exe, ("allocation", "--mock-wait-ms", "0"), "r4-lane-exclusivity", 1),
            ("float-generation-rejected", self.instrumented_exe, ("minimal-provider", "--generation-units", "1.5"), "cli-generation-units", 1),
            ("nan-wait-rejected", self.instrumented_exe, ("minimal-provider", "--mock-wait-ms", "NaN"), "cli-mock-wait-ms", 1),
            ("allocation-source-wrong-lane-rejected", self.instrumented_exe, ("minimal-provider", "--allocation-source", "unavailable"), "r4-allocation-source", 1),
        ):
            out = self.root / stage / name / "out.json"
            argv = [str(exe), "r4-lane", "--lane", args[0], "--fixture", str(fixture), "--work", str(out.parent / "work"), "--out", str(out)]
            argv.extend(args[1:])
            negative_specs.append((name, tuple(argv), intended, expected_code))
        negative_specs.append(("duplicate-lane-rejected", (str(self.instrumented_exe), "r4-lane", "--lane", "allocation", "--lane", "allocation"), "cli-parse", 2))
        rejected = [self._expected_failure(stage, name, argv, intended) for name, argv, intended, _ in negative_specs]
        for item, (_, _, _, expected_code) in zip(rejected, negative_specs):
            require(item["exit_code"] == expected_code, "r4-negative-exit", item["name"])

        reader_sources = {
            "reader-truncated": (b'{"schema": ', "malformed-json"),
            "reader-non-array": (b"[]", "[schema] compaction-r4-lane-v1 object required"),
            "reader-nonfinite": (b'{"x": NaN}', "malformed-json"),
        }
        reader_results = []
        validator = self.repo_root / "scripts/compaction_r4_lanes.py"
        for name, (body, intended_stage) in reader_sources.items():
            input_path = self.root / stage / f"{name}-input.json"
            write_new_bytes(input_path, body)
            out = self.root / stage / name / "out.json"
            argv = (sys.executable, validator, "validate", "--report", input_path, "--lane", "wall", "--out", out)
            result = self._expected_failure(stage, name, argv, intended_stage)
            require(result["exit_code"] == 2, "r4-reader-exit", name)
            reader_results.append(result)
        wrong_input = self.root / stage / "reader-wrong-lane-input.json"
        _, provider_report = self._run(stage, "reader-wrong-lane-source", r4_lane_command(self.instrumented_exe, "minimal-provider", fixture, self.root / stage / "reader-wrong-lane-source/work", self.root / stage / "reader-wrong-lane-source/out.json"), gate_a=True)
        validate_lane_report(provider_report, "minimal-provider", "instrumented")
        write_new_json(wrong_input, provider_report)
        name = "reader-wrong-lane"
        out = self.root / stage / name / "out.json"
        result = self._expected_failure(stage, name, (sys.executable, validator, "validate", "--report", wrong_input, "--lane", "wall", "--out", out), "lane-identity")
        require(result["exit_code"] == 2, "r4-reader-exit", name)
        reader_results.append(result)

        name = "create-new-collision-rejected"
        collision_path = self.root / stage / name / "out.json"
        require(first_allocation_report is not None, "r4-output-collision", "accepted allocation report is the collision fixture")
        write_new_json(collision_path, first_allocation_report)
        before = file_identity(collision_path)
        guard = getattr(self.adapter, "guard", None)
        require(isinstance(guard, IntegrityGuard), "protected-collision-input", "stock protected-input guard required")
        registration = guard.register_runtime_protected(self.root, {collision_path: before["sha256"]})
        write_new_json(collision_path.parent.parent / "create-new-collision-protected-input.json", {"schema": "compaction-r5-runtime-collision-input-v1", **registration})
        argv = r4_lane_command(self.instrumented_exe, "allocation", fixture, collision_path.parent / "work", collision_path)
        command_count_before = len(self.commands)
        receipt, output = self._run(stage, name, argv, gate_a=True, expected_exit=-1, existing_output_sha256=before["sha256"])
        stderr = receipt.spec.stderr.read_text(encoding="utf-8", errors="strict")
        collision = validate_r4_collision_rejection(
            exit_code=receipt.exit_code,
            stderr=stderr,
            protected_output=collision_path,
            before=before,
            after=file_identity(collision_path),
            produced_new_output=output,
            dependent_statistics=[],
            attempt_count=len(self.commands) - command_count_before,
            rust_source=(self.repo_root / R4_RUST_REPLAY_RELATIVE).read_text(encoding="utf-8"),
        )
        collision["protected_input"] = registration
        return {"schema": "compaction-r5-r4-negative-v3", "status": "ok", "accepted_count": len(accepted), "process_rejections": rejected, "reader_rejections": reader_results, "collision": collision, "synthetic_durations": "non-decision"}


    def run_all21_replays(self, summary_out: Path | None = None) -> dict[str, Any]:
        """Consume each actual writer.json; publish no aggregate until every case passes."""
        replay_reports: dict[str, Any] = {}
        for case_name in CASE_ORDER:
            work = self.root / "gate-a/replay-all21" / case_name / "work"
            out = self.root / "gate-a/replay-all21" / case_name / "writer.json"
            argv = replay_command(self.instrumented_exe, self.fixture_root / case_name, work, out, CASE_SUMMARY_CALLS[case_name])
            receipt, output = self._run("gate-a/replay-all21", case_name, argv, gate_a=True, out_path=out)
            require(output is not None and out.resolve() == receipt.spec.out.resolve(), "replay-output-path", case_name)
            replay_reports[case_name] = output
        all21 = aggregate_all21(replay_reports)
        if summary_out is not None:
            write_new_json(summary_out, all21)
        return all21

    def run_r4_eligibility(self) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run fresh supported cases and the three native panic negatives.

        This method is used only by an independently authorized Gate A.  Its
        synthetic adapter tests do not constitute Gate A or timing evidence.
        """
        eligible: list[str] = []
        eligibility: dict[str, Any] = {}
        noneligible_records: list[dict[str, Any]] = []
        candidates = tuple(name for name in CASE_ORDER if name in R4_ONE_CALL_TIMING_CASES or name in R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES)
        require(len(candidates) == 7 and len(set(candidates)) == 7 and set(candidates) == set(R4_ONE_CALL_TIMING_CASES + R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES), "r4-eligibility", "R13 preflight must include exactly four supported cases and three named native negatives")
        for case_name in candidates:
            out = self.root / "gate-a/r4-eligibility" / case_name / "out.json"
            argv = r4_lane_command(self.instrumented_exe, "replay-phase", self.fixture_root / case_name, out.parent / "work", out)
            receipt, report = self._run("gate-a/r4-eligibility", case_name, argv, gate_a=True, expected_exit=-1)
            if receipt.exit_code == 0:
                require(case_name in R4_ONE_CALL_TIMING_CASES, "r4-eligibility-supported-case", f"success outside the four supported cases is not admission evidence: {case_name}")
                extract_one_shot(report, "replay-phase")
                eligible.append(case_name)
                eligibility[case_name] = {"eligible": True}
            elif receipt.exit_code == 101:
                raw = validate_r4_noneligible_case(case_name, self.fixture_root, self.instrumented_exe, R4_INSTRUMENTED_EXE_SHA256, receipt, report)
                noneligible_records.append(raw)
                eligibility[case_name] = {"eligible": False, "status": "unavailable_noneligible", "reason": R4_NONELIGIBILITY_REASON, "native_pid": raw["native_pid"], "panic_thread_id": raw["panic_thread_id"]}
            else:
                raise Reject("r4-preflight-exit", f"case {case_name} must be a four-case successful preflight or a three-case native exit-101 negative; got exit {receipt.exit_code}")
        require(tuple(eligible) == R4_ONE_CALL_TIMING_CASES, "r4-eligibility", f"exact four supported cases required; actual eligible cases: {eligible}")
        validate_r4_noneligible_records(noneligible_records)
        continuations: dict[str, Any] = {}
        for case_name in eligible:
            out = self.root / "gate-a/r4-eligibility" / f"{case_name}-instrumented-continuation" / "out.json"
            argv = r4_lane_command(self.instrumented_exe, "replay-phase", self.fixture_root / case_name, out.parent / "work", out)
            _, inst = self._run("gate-a/r4-eligibility", f"{case_name}-instrumented-continuation", argv, gate_a=True)
            out = self.root / "gate-a/r4-eligibility" / f"{case_name}-wall-continuation" / "out.json"
            argv = wall_command(Path(sys.executable), self.repo_root / "scripts/compaction_r4_lanes.py", self.wall_exe, self.fixture_root / case_name, out.parent / "work", out.parent / "child.json", out)
            _, wall = self._run("gate-a/r4-eligibility", f"{case_name}-wall-continuation", argv, gate_a=True, nested_wall=True, running_image=self.wall_exe)
            extract_one_shot(inst, "replay-phase")
            extract_one_shot(wall, "wall")
            left = inst["continuation"]
            right = wall["child_report"]["continuation"]
            require(left == right and left["status"] == "available", "r4-continuation-parity", case_name)
            if case_name == "small-ascii":
                require(left["message_count"] == 18 and left["context_sha256"] == "f7a5df688012e2c31f1e36a98708a2f596a2abe6ce56fd810cabbdb5d38d317a", "r4-small-ascii-anchor", "18/context digest anchor differs")
            continuations[case_name] = left
        return eligible, eligibility, continuations, {row["case"]: row for row in noneligible_records}



    def run_gate_a(self, source_proof: Mapping[str, Any], expected_controls: Mapping[str, Sequence[Mapping[str, Any]]], focused: Mapping[str, Any], protected_hashes: Any) -> dict[str, Any]:
        """Execute only after separate real-Gate-A root authorization."""
        admitted_source = admit_source_identity(source_proof, git_runner=self.git_runner)
        approved_non_timing_exes = ((self.instrumented_exe, R4_INSTRUMENTED_EXE_SHA256, "instrumented"), (self.wall_exe, R4_WALL_EXE_SHA256, "wall"), (self.r3_exe, R3_REPLAY_EXE_SHA256, "r3-parity"))
        for exe, expected, label in approved_non_timing_exes:
            require(file_identity(exe)["sha256"] == expected, "gate-a-binary-identity", f"{label} is not the approved non-timing reproduction executable")
        self.register_focused_plan(focused)
        guard = getattr(self.adapter, "guard", None)
        require(isinstance(guard, IntegrityGuard), "focused-protected-coverage", "real Gate A requires the stock protected-input guard")
        for exe, identity in self.focused_executables.items():
            require(guard.classes["exe"].get(str(exe)) == identity["sha256"], "focused-protected-coverage", f"focused executable missing from protected hash manifest: {exe}")
        controls_root = self.root / "gate-a" / "controls"
        reports: dict[str, Any] = {}
        reports["r1"] = self._run("gate-a/controls", "r1", (self.instrumented_exe, "self-test-r1", "--work", controls_root / "r1/work", "--out", controls_root / "r1/out.json"), gate_a=True)[1]
        reports["r2"] = self._run("gate-a/controls", "r2", (self.instrumented_exe, "self-test-r2", "--fixture", self.fixture_root / "small-ascii", "--work", controls_root / "r2/work", "--out", controls_root / "r2/out.json"), gate_a=True)[1]
        reports["r3"] = self._run("gate-a/controls", "r3", (sys.executable, self.repo_root / "scripts/test_compaction_paired_protocol.py", controls_root / "r3/out.json"), gate_a=True)[1]
        reports["r4"] = self._run("gate-a/controls", "r4", (sys.executable, self.repo_root / "scripts/test_compaction_r4_lanes.py", controls_root / "r4/out.json"), gate_a=True)[1]
        controls = signing_control_summary(reports, expected_controls)
        # The R4 executable writes its nested work/reopen outputs below the
        # isolated process work root; the per-process paths above are reporters.
        invalid = []
        for name, _, _, _ in INVALID_CLI_SPECS:
            argv, stage = invalid_cli_command(self.instrumented_exe, self.fixture_root / "small-ascii", controls_root / name / "forbidden.json", name)
            invalid.append(self._expected_failure("gate-a/controls", f"invalid-{name}", argv, stage))
        all21 = self.run_all21_replays(self.root / "gate-a/replay-all21/summary.json")
        anchors = {}
        for name, failure, calls in (("small-ascii-success", False, 2), ("small-ascii-summary-failure", True, 1)):
            out = self.root / "gate-a/controls" / name / "out.json"
            argv = replay_command(self.instrumented_exe, self.fixture_root / "small-ascii", out.parent / "work", out, calls, failure=failure, anchor=True)
            _, report = self._run("gate-a/controls", name, argv, gate_a=True)
            anchors[name] = validate_replay_case(report, "small-ascii", calls, failure=failure, context_window=20000.0, timeout_secs=90)
        full21_unavailable, parity = self.run_local_parity()
        eligible, eligibility, continuations, noneligible_r4 = self.run_r4_eligibility()
        r4_negative = self.run_r4_negative_controls()
        focused_results = {}
        for kind, suite_count, passed, filtered in (("focused99", 14, 99, None), ("charcount15", 1, 15, 4499)):
            spec = focused[kind]
            logs = []
            commands = sorted((row for row in self.focused_executables.values() if row["kind"] == kind), key=lambda row: row["index"])
            for command in commands:
                receipt, _ = self._run(f"gate-a/{kind}", f"command-{command['index']:02d}", command["argv"], gate_a=True, profile_root=command["profile_root"], create_profile=(command["index"] == 0), profile_id=command["profile_id"])
                require(receipt.profile is not None, "profile-isolation", "focused before/after profile inventories required")
                logs.append(receipt.spec.stdout.read_text(encoding="utf-8", errors="strict") + receipt.spec.stderr.read_text(encoding="utf-8", errors="strict"))
            focused_results[kind] = parse_rust_test_log("\n".join(logs), spec["expected_names"], suite_count, passed, filtered)
        unavailable_warm_preflight = self.run_unavailable_warm_preflight()
        matrix = make_run_matrix(eligible)
        matrix_hash = structured_sha256(matrix)
        admitted_protected = validate_protected_hashes(protected_hashes, "gate-a-protected-hashes")
        summary = {"schema": GATE_A_SUMMARY_SCHEMA, "status": "ok", "timing_executed": False, "source": admitted_source, "protected_hashes": admitted_protected, "non_timing_executable_sha256": {"instrumented": R4_INSTRUMENTED_EXE_SHA256, "wall": R4_WALL_EXE_SHA256, "r3-parity": R3_REPLAY_EXE_SHA256}, "controls": controls, "invalid_cli": invalid, "all21": all21, "anchors": anchors, "local_parity": parity, "local_parity_unavailable": full21_unavailable, "warm_preflight_unavailable": unavailable_warm_preflight, "r4_eligibility": eligibility, "r4_noneligible": noneligible_r4, "r4_continuation": continuations, "focused_tests": focused_results, "r4_negative": r4_negative, "eligible_cases": eligible, "unavailable_noneligible_r4_cases": list(R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES), "run_matrix_sha256": matrix_hash, "run_matrix_count": len(matrix), "unreviewed": True}
        write_new_json(self.root / "gate-a/summary.json", summary)
        return summary

    def run_gate_b(self, eligible_cases: Sequence[str], authorization_path: Path, review_path: Path, gate_a_summary_path: Path, protected_hashes: Any) -> dict[str, Any]:
        matrix = validate_run_matrix(make_run_matrix(eligible_cases), eligible_cases)
        authorization = authorize_gate_b(load_json(authorization_path), review_path, gate_a_summary_path, matrix, executor_binaries={"instrumented": self.instrumented_exe, "wall": self.wall_exe, "r3-parity": self.r3_exe}, protected_hashes=protected_hashes)
        results: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in matrix:
            lane, case_name, index = row["lane"], row["case"], row["observation"]
            run_id = f"{row['order']:04d}-{case_name}-{index:02d}"
            stage = f"timing/{lane}/{case_name}"
            out = self.root / stage / run_id / "out.json"
            argv: tuple[str, ...]
            running_image = self.wall_exe if lane == "wall" else self.instrumented_exe
            if lane == "wall":
                argv = wall_command(Path(sys.executable), self.repo_root / "scripts/compaction_r4_lanes.py", self.wall_exe, self.fixture_root / case_name, out.parent / "work", out.parent / "child.json", out)
            elif lane == "bench-local-warm":
                argv = bench_command(self.instrumented_exe, self.fixture_root / case_name, out)
            else:
                argv = r4_lane_command(self.instrumented_exe, lane, self.fixture_root / case_name, out.parent / "work", out)
            receipt, report = self._run(stage, run_id, argv, pin=True, nested_wall=(lane == "wall"), running_image=running_image)
            require(receipt.affinity is not None, "affinity-coverage", run_id)
            if lane == "bench-local-warm":
                values, jev_state = validate_warm_observation(report, case_name)
            else:
                values = extract_one_shot(report, lane)
                jev_state = None
            production_pid = correlate_production_pid(receipt, report, lane)
            identity_hash = structured_sha256({"static": receipt.before_hashes, "row": row, "authorization": authorization["run_matrix_sha256"]})
            result: dict[str, Any] = {"status": "admitted", "lane": lane, "case": case_name, "observation": index, "fresh_processes": 1, "pid": production_pid, "identity_sha256": identity_hash, "values": values}
            if lane == "allocation":
                result["unavailable_reason"] = values.pop("unavailable_reason")
            if lane == "bench-local-warm":
                result.update({"warmup": 10, "iterations": 30, "jev_state": jev_state})
            results.setdefault(lane, {}).setdefault(case_name, []).append(result)
        summaries = []
        for lane, cases in results.items():
            for case_name, records in cases.items():
                stats = summarize_warm_processes(records, case_name) if lane == "bench-local-warm" else summarize_fresh_processes(records, lane, case_name)
                write_new_json(self.root / "timing" / lane / case_name / "outer-runs.json", records)
                write_new_json(self.root / "timing" / lane / case_name / "stats.json", stats)
                summaries.append({"lane": lane, "case": case_name, "observations": len(records), "stats": stats["statistics"]})
        report = {"schema": "compaction-r5-baseline-report-v1", "status": "ok", "baseline_only": True, "no_comparator": True, "provider_network": "none", "run_count": len(matrix), "results": summaries, "performance_claim": False, "synthetic_provider_durations": "non-decision", "page_cache_state": "unobserved", "thermal_state": "unobserved"}
        write_new_json(self.root / "postrun/baseline-report.json", report)
        return report


def command_catalog() -> dict[str, tuple[str, ...]]:
    """Supported CLI only; no alternate production or network entry points."""
    return {"replay": ("self-test-r1", "self-test-r2", "replay", "verify-reopen", "verify-local", "bench-local", "r4-lane"), "python": ("test_compaction_paired_protocol.py", "test_compaction_r4_lanes.py", "compaction_r4_lanes.py")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="compaction-r5-campaign-driver synthetic-review build; real gates closed")
    parser.add_argument("command", nargs="?", choices=("validate-plan", "aggregate-controls", "aggregate-replay", "compare-parity", "execute-campaign"), help="receipt readers or a separately authorized campaign")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--gate-a-summary", type=Path)
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "execute-campaign":
            require(args.plan is not None, "campaign-plan", "--plan required")
            plan = load_json(args.plan)
            exact_object(plan, ("schema", "gate", "source", "fixture", "toolchain", "scripts", "binaries", "run_matrix", "expected_controls", "expected_summary_calls", "expected_case_order", "authorization", "artifact_root", "fixture_root", "worktree", "protected_hashes", "focused"), "plan-fields")
            require(plan["schema"] == SCHEMA and tuple(plan["expected_case_order"]) == CASE_ORDER and plan["expected_summary_calls"] == CASE_SUMMARY_CALLS, "plan-identity", "case/summary plan differs")
            require(Path(plan["artifact_root"]) == ARTIFACT_ROOT and Path(plan["worktree"]) == Path(plan["source"]["worktree"]) and Path(plan["fixture_root"]) == Path(plan["fixture"]["root"]), "plan-root-identity", "artifact/worktree/fixture roots differ")
            require(set(plan["protected_hashes"]) == {"source", "fixture", "toolchain", "script", "exe", "run_matrix"}, "integrity-classes", "six protected input classes required")
            require(os.name == "nt", "campaign-platform", "real campaign currently requires the reviewed Windows x86-64 adapter")
            guard = IntegrityGuard(Path(plan["worktree"]), plan["protected_hashes"])
            telemetry_adapter = CommandTelemetry()
            descendants = WindowsDescendants()
            adapter = SubprocessAdapter(guard, telemetry_adapter, descendants)
            paths = plan["binaries"]
            require(set(paths) == {"instrumented", "wall", "r3-parity"}, "binary-plan", "three binary identities required")
            executor = CampaignExecutor(adapter, Path(plan["artifact_root"]), Path(plan["fixture_root"]), Path(plan["worktree"]), Path(paths["instrumented"]), Path(paths["wall"]), paths["r3-parity"] and Path(paths["r3-parity"]))
            if plan["gate"] == "gate-a":
                gate_auth = exact_object(plan["authorization"], ("approved", "root_message_id", "scope"), "gate-a-authorization")
                require(gate_auth["approved"] is True and gate_auth["scope"] == "real-gate-a-after-independent-r5-source-control-review", "gate-a-authorization", "distinct real-Gate-A root authorization required")
                nonempty_string(gate_auth["root_message_id"], "gate-a-authorization-message")
                summary = executor.run_gate_a(plan["source"], plan["expected_controls"], plan["focused"], plan["protected_hashes"])
                print(json.dumps({"schema": summary["schema"], "status": summary["status"], "timing_executed": False}))
            else:
                require(plan["gate"] == "gate-b" and args.authorization is not None and args.review is not None and args.gate_a_summary is not None, "gate-b-cli-input", "authorization, review and Gate A summary required")
                summary = load_json(args.gate_a_summary)
                require(summary.get("schema") == GATE_A_SUMMARY_SCHEMA and summary.get("status") == "ok" and summary.get("timing_executed") is False, "gate-a-summary", "eligible cases require green non-timing summary")
                report = executor.run_gate_b(summary["eligible_cases"], args.authorization, args.review, args.gate_a_summary, plan["protected_hashes"])
                print(json.dumps({"schema": report["schema"], "status": report["status"], "run_count": report["run_count"]}))
            return 0
        require(args.input is not None and args.out is not None, "cli-input", "--input and --out required")
        value = load_json(args.input)
        if args.command == "validate-plan":
            require(isinstance(value, dict) and value.get("schema") == SCHEMA, "plan-schema", SCHEMA)
            exact_object(value, ("schema", "gate", "source", "fixture", "toolchain", "scripts", "binaries", "run_matrix", "expected_controls", "expected_summary_calls", "expected_case_order", "authorization", "artifact_root", "fixture_root", "worktree", "protected_hashes", "focused"), "plan-fields")
            require(value["gate"] in {"gate-a", "gate-b"} and value["expected_summary_calls"] == CASE_SUMMARY_CALLS and tuple(value["expected_case_order"]) == CASE_ORDER, "plan-identity", "wrong case or summary plan")
            admit_source_identity(value["source"])
            require(Path(value["artifact_root"]) == ARTIFACT_ROOT and Path(value["worktree"]) == Path(value["source"]["worktree"]) and Path(value["fixture_root"]) == Path(value["fixture"]["root"]), "plan-root-identity", "artifact/worktree/fixture roots differ")
            result = {"schema": "compaction-r5-plan-receipt-v1", "status": "accepted", "gate": value["gate"]}
        elif args.command == "aggregate-controls":
            require(args.expected is not None, "cli-input", "--expected signatures required")
            result = signing_control_summary(value, load_json(args.expected))
        elif args.command == "aggregate-replay":
            result = aggregate_all21(value)
        else:
            require(args.expected is not None, "cli-input", "--expected R3 baseline required")
            result = compare_local_parity(load_json(args.expected), value)
        write_new_json(args.out, result)
        print(json.dumps({"status": result.get("status", "accepted"), "out": str(args.out)}))
        return 0
    except (Reject, R4Reject, OSError, KeyError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
