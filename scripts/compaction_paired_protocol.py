#!/usr/bin/env python3
"""Fresh-process paired compaction protocol with fail-closed provenance.

R3 qualification uses only synthetic/non-decision commands. A production-like
plan must be independently authorized before any Optimus timing invocation.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import uuid
from typing import Any

from compaction_paired_stats import StageError, analyze, write_new_json

ARMS = ("baseline", "candidate")
SCHEMA = "r3-paired-protocol-v1"
REQUIRED_LOAD_FIELDS = (
    "cpu_total_percent", "processor_frequency", "processor_queue",
    "available_memory_bytes", "per_logical_processor",
)
FULL_PROJECTION_ALGORITHM = 'sha256 over compact serde_json encoding of the complete in-memory semantic section before report serialization'
FULL_PROJECTION_NOTE = 'Complete values are retained in cases; these are full-content hashes, not length or scalar checksums'
FULL_PROJECTION_SECTIONS = ('case', 'fixture_manifest', 'token_outputs', 'cut_output', 'prepare_compaction', 'conversion', 'serialization', 'replicated_chunk_slicing', 'session_manager__raw_branch_semantic', 'session_manager__normalized_restored_messages', 'session_manager__context', 'session_manager__checkpoint_state', 'session_manager__protected_custom_state', 'jev_prepare_context')
SAFE_INHERITED_ENV = (
    "PATH", "SystemRoot", "ComSpec", "TEMP", "TMP", "USERPROFILE",
    "LOCALAPPDATA", "PROGRAMDATA", "SYSTEMDRIVE", "HOME",
)
SAFE_EXPLICIT_ENV = re.compile(r"^(?:R3_MOCK_[A-Z0-9_]+|PRIME_AGENT_COMPACTION_[A-Z0-9_]+|PRIME_AGENT_COMPACT_RUST_COUNT|COMPACTION_REPLAY_[A-Z0-9_]+)$")
SECRET_NAME = re.compile(r"(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|API_?KEY)", re.I)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise StageError("file-identity", f"not a regular file: {resolved}")
    data = resolved.read_bytes()
    return {"path": str(resolved), "bytes": len(data), "sha256": sha256_bytes(data)}


def path_identity(path: Path) -> dict[str, Any]:
    try:
        return file_identity(path)
    except OSError as error:
        raise StageError("file-identity", f"cannot read {path}: {error}") from error


def require(condition: Any, stage: str, message: str) -> None:
    if not condition:
        raise StageError(stage, message)


def same_file_path(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a.resolve(strict=False))) == os.path.normcase(str(b.resolve(strict=False)))


def load_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageError("plan-load", str(error)) from error
    require(isinstance(plan, dict) and plan.get("schema") == SCHEMA, "plan-schema", f"schema must be {SCHEMA}")
    for key in ("seed", "outer_pairs", "warmup", "iterations", "output_root", "lane", "fixtures", "expected_fixture_sha256", "arms", "metrics", "telemetry", "toolchain"):
        require(key in plan, "plan-schema", f"missing {key}")
    require(isinstance(plan["seed"], int) and not isinstance(plan["seed"], bool), "plan-seed", "seed must be an integer")
    for key, expected in (("outer_pairs", 30), ("warmup", 10), ("iterations", 30)):
        require(type(plan[key]) is int and plan[key] == expected, "plan-protocol-count", f"{key} must be exactly {expected} with strict int-not-bool type, got {plan[key]!r}")
    require(plan.get("protocol_class") in ("synthetic-non-decision", "authorized-r4-timing"), "plan-class", "unknown protocol_class")
    require(plan["protocol_class"] == "synthetic-non-decision" or plan.get("r4_timing_authorized") is True, "timing-authorization", "R3 driver refuses Optimus timing without explicit R4 authorization")
    require(set(plan["arms"]) == set(ARMS), "plan-arms", "arms must be baseline and candidate")
    require(isinstance(plan["metrics"], list) and plan["metrics"], "plan-metrics", "metrics must be nonempty")
    for metric in plan["metrics"]:
        require(isinstance(metric, dict) and isinstance(metric.get("case"), str) and metric["case"] and isinstance(metric.get("kernel"), str) and metric["kernel"], "plan-metrics", "each metric needs nonempty case and kernel")
    metric_cases = [metric["case"] for metric in plan["metrics"]]
    require(len(set(metric_cases)) == len(metric_cases), "plan-metrics", "each expected parity case must be unique")
    lane = plan["lane"]
    require(lane.get("kind") in ("parallel-production", "single-cpu-microkernel", "process-cold-replay"), "plan-lane", "invalid lane kind")
    require(lane.get("process_state") in ("warm", "process-cold"), "plan-lane", "process_state must be warm or process-cold; page-cache-cold is not claimed")
    cpus = lane.get("cpu_ids")
    require(isinstance(cpus, list) and cpus and all(isinstance(cpu, int) and not isinstance(cpu, bool) and cpu >= 0 for cpu in cpus), "lane-cpu-mask", "cpu_ids must be nonnegative integers")
    require(len(set(cpus)) == len(cpus), "lane-cpu-mask", "duplicate logical CPU")
    mask = sum(1 << cpu for cpu in cpus)
    require(lane.get("mask") == hex(mask), "lane-cpu-mask", f"mask must be {hex(mask)} for {cpus}")
    if lane["kind"] == "parallel-production":
        require(cpus == [0, 2, 4, 6, 8, 10], "lane-cpu-mask", "parallel production lane must prefer 0,2,4,6,8,10")
    if lane["kind"] == "single-cpu-microkernel":
        require(len(cpus) == 1, "lane-cpu-mask", "single-CPU microkernel lane needs exactly one CPU")
    require(isinstance(lane.get("topology"), dict) and lane["topology"], "lane-topology", "topology evidence is required")
    require(isinstance(plan["output_root"], str) and plan["output_root"], "plan-output", "output_root is required")
    return plan


def make_orders(seed: int, pairs: int) -> list[dict[str, Any]]:
    sequence = [["baseline", "candidate"], ["candidate", "baseline"]] * (pairs // 2)
    random.Random(seed).shuffle(sequence)
    return [{"pair_id": index, "order": order} for index, order in enumerate(sequence)]


def validate_orders(plan: dict[str, Any]) -> list[dict[str, Any]]:
    orders = make_orders(plan["seed"], plan["outer_pairs"])
    expected = plan.get("expected_order")
    if expected is not None:
        require(expected == orders, "order-replay", "recorded expected_order does not reproduce from seed")
    counts = (sum(item["order"][0] == "baseline" for item in orders), sum(item["order"][0] == "candidate" for item in orders))
    require(counts == (15, 15), "ab-ba-balance", f"orders are not balanced 15/15: {counts}")
    return orders


def validate_source(arm_name: str, arm: dict[str, Any], protocol_class: str) -> dict[str, Any]:
    root = Path(arm["source_root"]).resolve(strict=True)
    expected_files = arm.get("source_sha256")
    require(isinstance(expected_files, dict) and expected_files, "source-identity", f"{arm_name} source_sha256 is required")
    identities = {}
    for relative, expected_sha in expected_files.items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "source-identity", f"unsafe source path {relative}")
        identity = path_identity(root / relative)
        require(identity["sha256"] == expected_sha, "source-hash", f"{arm_name} {relative} hash mismatch")
        identities[relative] = identity
    if protocol_class == "synthetic-non-decision":
        identity = arm.get("synthetic_source_identity")
        require(isinstance(identity, dict) and identity.get("dirty") is False, "source-dirty", f"{arm_name} synthetic source must be explicitly clean")
        require(re.fullmatch(r"[0-9a-f]{40}", identity.get("head", "")) is not None, "source-identity", "synthetic head must be a full hash-like identity")
        require(re.fullmatch(r"[0-9a-f]{40}", identity.get("base", "")) is not None, "source-identity", "synthetic base must be a full hash-like identity")
        return {"root": str(root), **identity, "dirty_status": [], "files": identities, "identity_method": "computed listed source bytes; synthetic head/base labels are non-decision fixtures"}
    head = run_capture(["git", "-C", str(root), "rev-parse", "HEAD"], "source-head").strip()
    require(head == arm.get("head"), "source-head", f"{arm_name} HEAD {head} differs from plan")
    base = arm.get("base", "")
    require(re.fullmatch(r"[0-9a-f]{40}", base) is not None, "source-base", "full base hash required")
    run_capture(["git", "-C", str(root), "merge-base", "--is-ancestor", base, head], "source-base")
    status = run_capture(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"], "source-dirty").splitlines()
    require(not status, "source-dirty", f"{arm_name} source is dirty: {status}")
    diff = run_capture_bytes(["git", "-C", str(root), "diff", "--binary", head, "--"], "source-diff")
    return {"root": str(root), "head": head, "base": base, "dirty": False, "dirty_status": status, "diff_bytes": len(diff), "diff_sha256": sha256_bytes(diff), "files": identities, "identity_method": "git + computed file bytes"}


def validate_toolchain(plan: dict[str, Any]) -> dict[str, Any]:
    toolchain = plan["toolchain"]
    for key in ("rust_version", "llvm_version", "target", "profile", "locked", "offline", "rustc_path", "cargo_path"):
        require(key in toolchain, "toolchain-identity", f"missing toolchain {key}")
    require(toolchain["profile"] == "release" and toolchain["locked"] is True and toolchain["offline"] is True, "toolchain-config", "toolchain must be release, locked and offline")
    rustc = path_identity(Path(toolchain["rustc_path"]))
    cargo = path_identity(Path(toolchain["cargo_path"]))
    evidence = {"declared": toolchain, "rustc": rustc, "cargo": cargo}
    if plan["protocol_class"] == "authorized-r4-timing":
        verbose = run_capture([rustc["path"], "-Vv"], "toolchain-rustc")
        values = dict(line.split(": ", 1) for line in verbose.splitlines() if ": " in line)
        require(values.get("release") == toolchain["rust_version"], "toolchain-rustc", f"rustc release mismatch: {values.get('release')}")
        require(values.get("LLVM version") == toolchain["llvm_version"], "toolchain-rustc", f"LLVM mismatch: {values.get('LLVM version')}")
        require(values.get("host") == toolchain["target"], "toolchain-target", f"target mismatch: {values.get('host')}")
        cargo_version = run_capture([cargo["path"], "--version"], "toolchain-cargo").strip()
        require(cargo_version == f"cargo {toolchain['rust_version']}" or cargo_version.startswith(f"cargo {toolchain['rust_version']} "), "toolchain-cargo", f"Cargo version mismatch: {cargo_version}")
        for arm_name in ARMS:
            build_argv = plan["arms"][arm_name].get("build_argv")
            require(isinstance(build_argv, list) and build_argv, "toolchain-build-argv", f"{arm_name} build_argv required")
            require(all(flag in build_argv for flag in ("--release", "--locked", "--offline")), "toolchain-build-argv", f"{arm_name} build is not release/locked/offline")
            require("build" in build_argv and "--manifest-path" in build_argv, "toolchain-build-argv", f"{arm_name} build must name the source manifest")
        evidence.update({"rustc_vv": verbose, "cargo_version": cargo_version})
    else:
        evidence["identity_method"] = "computed compiler/Cargo bytes; declared versions are synthetic non-decision labels"
    return evidence


def validate_static(plan: dict[str, Any]) -> dict[str, Any]:
    toolchain = validate_toolchain(plan)
    target_dirs = [Path(plan["arms"][arm]["target_dir"]).resolve(strict=True) for arm in ARMS]
    require(not same_file_path(target_dirs[0], target_dirs[1]), "unique-target", "baseline and candidate target dirs must differ")
    fixture_paths = {name: {} for name in ARMS}
    expected_fixture = plan["expected_fixture_sha256"]
    require(set(expected_fixture) == {"transcript", "manifest"}, "fixture-identity", "expected fixture hashes need transcript and manifest")
    arm_evidence = {}
    for arm_name in ARMS:
        arm = plan["arms"][arm_name]
        source = validate_source(arm_name, arm, plan["protocol_class"])
        binary = path_identity(Path(arm["exe"]))
        require(binary["sha256"] == arm.get("exe_sha256"), "binary-hash", f"{arm_name} executable hash mismatch")
        for kind in ("transcript", "manifest"):
            identity = path_identity(Path(plan["fixtures"][arm_name][kind]))
            require(identity["sha256"] == expected_fixture[kind], "fixture-hash", f"{arm_name} {kind} differs from frozen plan")
            fixture_paths[arm_name][kind] = identity
        require(arm.get("argv") and arm.get("env") is not None and arm.get("expected_runtime"), "plan-arms", f"{arm_name} argv/env/expected_runtime required")
        for name, value in arm["env"].items():
            require(SAFE_EXPLICIT_ENV.fullmatch(name) is not None and not SECRET_NAME.search(name), "environment-allowlist", f"unsafe environment variable {name}")
            require(isinstance(value, str), "environment-allowlist", f"environment {name} must be text")
        arm_evidence[arm_name] = {"source": source, "binary_before_all": binary, "target_dir": str(target_dirs[ARMS.index(arm_name)]), "fixtures_before_all": fixture_paths[arm_name]}
    if plan["protocol_class"] == "authorized-r4-timing":
        require(not same_file_path(Path(plan["arms"]["baseline"]["exe"]), Path(plan["arms"]["candidate"]["exe"])), "unique-binary", "production arms require distinct executable paths")
        for kind in ("transcript", "manifest"):
            require(not same_file_path(Path(plan["fixtures"]["baseline"][kind]), Path(plan["fixtures"]["candidate"][kind])), "unique-fixture-copy", f"arms require separate {kind} copies")
    return {"arms": arm_evidence, "toolchain": toolchain, "compiled_at": utc_now()}


def run_capture(argv: list[str], stage: str) -> str:
    return run_capture_bytes(argv, stage).decode("utf-8", errors="strict")


def run_capture_bytes(argv: list[str], stage: str) -> bytes:
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as error:
        raise StageError(stage, f"cannot execute {argv[0]}: {error}") from error
    if result.returncode != 0:
        raise StageError(stage, f"command exited {result.returncode}: {result.stderr.decode('utf-8', errors='replace')[:500]}")
    return result.stdout


def validate_parity(plan: dict[str, Any]) -> dict[str, Any]:
    parity = plan.get("parity")
    if plan["protocol_class"] == "synthetic-non-decision":
        return {"class": "synthetic preflight fingerprints compared per pair"}
    require(isinstance(parity, dict), "parity-provenance", "authorized timing requires verify-local receipts")
    reports = {}
    for arm_name in ARMS:
        spec = parity[arm_name]
        identity = path_identity(Path(spec["path"]))
        require(identity["sha256"] == spec["sha256"], "parity-provenance", f"{arm_name} parity receipt hash mismatch")
        try:
            report = json.loads(Path(identity["path"]).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise StageError("parity-admission", f"{arm_name} verify-local receipt is malformed JSON") from error
        require(isinstance(report, dict) and report.get("status") == "ok" and report.get("non_timed") is True and report.get("performance_claim") is False, "parity-admission", f"{arm_name} verify-local not admitted")
        reports[arm_name] = {"identity": identity, "report": report}
    require(reports["candidate"]["report"].get("comparison") == "exact-full-output-against-baseline", "parity-admission", "candidate is not an exact full-output comparison")

    def projections(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
        cases = report.get("cases")
        require(isinstance(cases, list) and bool(cases), "parity-admission", "verify-local cases must be a nonempty list")
        result: dict[str, dict[str, Any]] = {}
        for case in cases:
            require(isinstance(case, dict), "parity-admission", "verify-local case is not an object")
            case_name = case.get("case")
            projection = case.get("full_projection_sha256")
            require(isinstance(case_name, str) and bool(case_name), "parity-admission", "verify-local case name missing")
            require(case_name not in result, "parity-admission", f"duplicate verify-local case {case_name}")
            require(isinstance(projection, dict) and set(projection) == {"algorithm", "sections", "note"}, "parity-projection-schema", f"full_projection_sha256 for {case_name} must contain exactly algorithm, sections and note")
            require(projection.get("algorithm") == FULL_PROJECTION_ALGORITHM, "parity-projection-schema", f"unexpected full projection algorithm for {case_name}")
            require(projection.get("note") == FULL_PROJECTION_NOTE, "parity-projection-schema", f"unexpected full projection note for {case_name}")
            sections = projection.get("sections")
            require(isinstance(sections, dict) and set(sections) == set(FULL_PROJECTION_SECTIONS), "parity-projection-sections", f"full projection must have exactly the expected 14 sections for {case_name}")
            for section_name in FULL_PROJECTION_SECTIONS:
                digest = sections[section_name]
                require(isinstance(digest, str) and bool(digest) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "parity-projection-section", f"invalid full projection hash for {case_name}/{section_name}")
            result[case_name] = projection
        return result

    expected_cases = {metric["case"] for metric in plan["metrics"]}
    baseline_projection = projections(reports["baseline"]["report"])
    candidate_projection = projections(reports["candidate"]["report"])
    require(set(baseline_projection) == expected_cases and set(candidate_projection) == expected_cases, "parity-admission", "verify-local case coverage differs from expected metric cases")
    require(baseline_projection == candidate_projection, "parity-admission", "baseline/candidate full projection hashes differ")
    return {"class": "verify-local exact full outputs", "arms": reports}



def expand(value: str, **tokens: Any) -> str:
    return value.format(**tokens)


def child_environment(arm: dict[str, Any]) -> dict[str, str]:
    env = {name: os.environ[name] for name in SAFE_INHERITED_ENV if name in os.environ}
    env.update(arm["env"])
    return env


def query_process_image(process: subprocess.Popen[bytes]) -> str:
    if os.name != "nt":
        try:
            return str(Path(f"/proc/{process.pid}/exe").resolve(strict=True))
        except OSError as error:
            raise StageError("running-exe-identity", str(error)) from error
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_ulong(32768)
    buffer = ctypes.create_unicode_buffer(size.value)
    success = kernel32.QueryFullProcessImageNameW(ctypes.c_size_t(process._handle), 0, buffer, ctypes.byref(size))  # type: ignore[attr-defined]
    if not success:
        raise StageError("running-exe-identity", f"QueryFullProcessImageNameW failed with {ctypes.get_last_error()}")
    return str(Path(buffer.value).resolve(strict=True))


def set_child_affinity(process: subprocess.Popen[bytes], mask: int) -> dict[str, Any]:
    if os.name != "nt":
        try:
            os.sched_setaffinity(process.pid, {index for index in range(mask.bit_length()) if mask & (1 << index)})
            actual = os.sched_getaffinity(process.pid)
        except (AttributeError, OSError) as error:
            raise StageError("child-affinity", str(error)) from error
        return {"requested_mask": hex(mask), "observed_cpu_ids": sorted(actual), "observed_mask": hex(sum(1 << cpu for cpu in actual)), "method": "sched_setaffinity on child PID only"}
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = ctypes.c_size_t(process._handle)  # type: ignore[attr-defined]
    if not kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
        raise StageError("child-affinity", f"SetProcessAffinityMask failed with {ctypes.get_last_error()}")
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise StageError("child-affinity", f"GetProcessAffinityMask failed with {ctypes.get_last_error()}")
    require(process_mask.value == mask, "child-affinity", f"observed mask {process_mask.value:#x} differs from {mask:#x}")
    return {"requested_mask": hex(mask), "observed_mask": hex(process_mask.value), "system_mask": hex(system_mask.value), "method": "Set/GetProcessAffinityMask on child handle only; agent and system affinity unchanged"}


def collect_probe(argv: list[str], output: Path, stage: str, pair_id: int, phase: str) -> dict[str, Any]:
    expanded = [expand(item, pair_id=pair_id, phase=phase) for item in argv]
    started = utc_now()
    result = subprocess.run(expanded, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    identity = {"argv": expanded, "utc": started, "exit_code": result.returncode, "stdout_sha256": sha256_bytes(result.stdout), "stderr_sha256": sha256_bytes(result.stderr)}
    require(result.returncode == 0, stage, f"probe exited {result.returncode}: {result.stderr.decode(errors='replace')[:300]}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise StageError(stage, f"probe output is not JSON: {error}") from error
    require(isinstance(value, dict), stage, "probe output must be an object")
    write_new_json(output, {"identity": identity, "value": value})
    identity["output"] = path_identity(output)
    return {"identity": identity, "value": value}


def collect_block(plan: dict[str, Any], pair_id: int, phase: str, root: Path) -> dict[str, Any]:
    spec = plan["telemetry"]
    required_fields = spec.get("required_load_fields", list(REQUIRED_LOAD_FIELDS))
    require(required_fields == list(REQUIRED_LOAD_FIELDS), "telemetry-load", "required_load_fields is fixed by the documented contract and cannot be relaxed")
    load = collect_probe(spec["load_argv"], root / f"pair-{pair_id:02d}-{phase}-load.json", "telemetry-load", pair_id, phase)
    power = collect_probe(spec["power_argv"], root / f"pair-{pair_id:02d}-{phase}-power.json", "telemetry-power", pair_id, phase)
    for field in REQUIRED_LOAD_FIELDS:
        require(field in load["value"], "telemetry-load", f"missing available load field {field}")
    require("active_power_scheme" in power["value"], "telemetry-power", "missing active power scheme")
    return {"load": load, "power": power, "limitations": plan.get("measurement_limitations", []), "required_load_fields": list(REQUIRED_LOAD_FIELDS)}



def json_pointer(value: Any, pointer: str) -> Any:
    current = value
    if pointer == "":
        return current
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        require(isinstance(current, dict) and part in current, "output-shape", f"missing JSON pointer {pointer}")
        current = current[part]
    return current


def expected_leaves(expected: Any, actual: Any, prefix: str = "runtime") -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), "runtime-mode", f"{prefix} is not an object")
        for key, value in expected.items():
            require(key in actual, "runtime-mode", f"missing {prefix}/{key}")
            expected_leaves(value, actual[key], f"{prefix}/{key}")
    else:
        require(actual == expected and type(actual) is type(expected), "runtime-mode", f"{prefix} expected {expected!r}, got {actual!r}")


def extract_medians(plan: dict[str, Any], output: dict[str, Any], elapsed_ms: float) -> dict[str, float]:
    lane = plan["lane"]
    if lane["process_state"] == "process-cold":
        require(plan["arms"]["baseline"].get("output_kind") == "replay", "output-shape", "process-cold lane requires replay output")
        require(output.get("failures") == [] and type(output.get("case_count")) is int and output["case_count"] > 0, "replay-admission", "replay has failures or no cases")
        require(math.isfinite(elapsed_ms) and elapsed_ms > 0, "numeric-input", "process duration is invalid")
        return {"process_total_ms": elapsed_ms}
    config = output.get("config")
    require(isinstance(config, dict), "output-config", "config must be an object")
    require(type(config.get("warmup")) is int and config["warmup"] == plan["warmup"], "output-config", "warmup must be exact int-not-bool config")
    require(type(config.get("iters")) is int and config["iters"] == plan["iterations"], "output-config", "iterations must be exact int-not-bool config")
    mapping = {(case.get("case"), kernel.get("name")): kernel for case in output.get("cases", []) for kernel in case.get("kernels", [])}
    medians = {}
    for metric in plan["metrics"]:
        key = (metric["case"], metric["kernel"])
        require(key in mapping, "output-shape", f"missing kernel {key}")
        kernel = mapping[key]
        require(type(kernel.get("warmup")) is int and kernel["warmup"] == plan["warmup"], "output-config", f"{key} warmup must be exact int-not-bool")
        stats = kernel.get("stats")
        require(isinstance(stats, dict) and type(stats.get("n")) is int and stats["n"] == plan["iterations"], "output-config", f"{key} inner sample count must be exact int-not-bool")
        value = stats.get("p50_ms")
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value > 0, "numeric-input", f"{key} p50_ms must be finite and positive")
        medians[f"{key[0]}/{key[1]}"] = float(value)
    return medians



def preflight_fingerprints(plan: dict[str, Any], output: dict[str, Any]) -> list[tuple[str, str]]:
    preflight = output.get("non_timed_preflight")
    require(isinstance(preflight, list) and bool(preflight), "parity-admission", "non_timed_preflight must be a nonempty list")
    expected_cases = {metric["case"] for metric in plan["metrics"]}
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in preflight:
        require(isinstance(item, dict), "parity-admission", "non_timed_preflight case is not an object")
        case_name = item.get("case")
        digest = item.get("full_output_sha256")
        require(isinstance(case_name, str) and bool(case_name), "parity-admission", "non_timed_preflight case name missing")
        require(case_name in expected_cases, "parity-admission", f"unexpected non_timed_preflight case {case_name}")
        require(case_name not in seen, "parity-admission", f"duplicate non_timed_preflight case {case_name}")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "parity-admission", f"invalid full_output_sha256 for {case_name}")
        seen.add(case_name)
        result.append((case_name, digest))
    require(seen == expected_cases, "parity-admission", "non_timed_preflight does not cover every expected case")
    return sorted(result)



def execute_run(plan: dict[str, Any], static: dict[str, Any], pair_id: int, arm_name: str, sequence_index: int, root: Path) -> dict[str, Any]:
    arm = plan["arms"][arm_name]
    run_id = f"pair-{pair_id:02d}-{sequence_index}-{arm_name}"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = root / "outputs" / f"{run_id}.json"
    stdout_path = run_dir / "stdout.bin"
    stderr_path = run_dir / "stderr.bin"
    record_path = run_dir / "record.json"
    for path in (output_path, stdout_path, stderr_path, record_path):
        require(not path.exists(), "output-create-new", f"output already exists: {path}")
    tokens = {
        "exe": str(Path(arm["exe"]).resolve()), "out": str(output_path),
        "fixture": str(Path(plan["fixtures"][arm_name]["transcript"]).resolve().parent),
        "transcript": str(Path(plan["fixtures"][arm_name]["transcript"]).resolve()),
        "manifest": str(Path(plan["fixtures"][arm_name]["manifest"]).resolve()),
        "warmup": str(plan["warmup"]), "iterations": str(plan["iterations"]),
        "run_id": run_id, "pair_id": str(pair_id), "arm": arm_name,
    }
    argv = [expand(str(part), **tokens) for part in arm["argv"]]
    require(Path(argv[0]).resolve() == Path(arm["exe"]).resolve(), "argv-executable", "argv[0] must be the declared executable")
    env = child_environment(arm)
    before_fixture = {kind: path_identity(Path(plan["fixtures"][arm_name][kind])) for kind in ("transcript", "manifest")}
    require(all(before_fixture[kind]["sha256"] == plan["expected_fixture_sha256"][kind] for kind in before_fixture), "fixture-before", "fixture changed before run")
    before_binary = path_identity(Path(arm["exe"]))
    started_utc = utc_now()
    started_ns = time.perf_counter_ns()
    process = None
    stdout = stdout_path.open("xb")
    stderr = stderr_path.open("xb")
    partial: dict[str, Any] = {"schema": "r3-outer-run-v1", "run_id": run_id, "pair_id": pair_id, "arm": arm_name, "sequence_index": sequence_index, "status": "failed", "argv": argv, "child_env": env, "started_utc": started_utc, "fixtures_before": before_fixture}
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env)
        partial["pid"] = process.pid
        partial["affinity"] = set_child_affinity(process, int(plan["lane"]["mask"], 16))
        running_path = query_process_image(process)
        require(same_file_path(Path(running_path), Path(arm["exe"])), "running-exe-identity", f"child image {running_path} differs from {arm['exe']}")
        partial["running_executable"] = path_identity(Path(running_path))
        exit_code = process.wait(timeout=plan.get("timeout_sec", 300))
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        partial["exit_code"] = exit_code
        partial["ended_utc"] = utc_now()
        partial["elapsed_ms_process_cold_label_only"] = elapsed_ms
        require(exit_code == 0, "child-exit", f"child exited {exit_code}")
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise StageError("child-timeout", f"child exceeded timeout {error.timeout}") from error
    except StageError:
        partial["ended_utc"] = utc_now()
        if process is not None:
            partial["exit_code"] = process.returncode
        for label, stream_path in (("stdout", stdout_path), ("stderr", stderr_path)):
            if stream_path.exists():
                partial[label] = path_identity(stream_path)
        write_new_json(record_path, partial)
        raise
    finally:
        stdout.close()
        stderr.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    after_fixture = {kind: path_identity(Path(plan["fixtures"][arm_name][kind])) for kind in ("transcript", "manifest")}
    require(after_fixture == before_fixture, "fixture-after", "fixture path/bytes/hash changed during run")
    after_binary = path_identity(Path(arm["exe"]))
    require(after_binary == before_binary, "binary-after", "binary identity changed during run")
    partial.update({"fixtures_after": after_fixture, "binary_before": before_binary, "binary_after": after_binary, "stdout": path_identity(stdout_path), "stderr": path_identity(stderr_path)})
    require(output_path.is_file(), "output-missing", "child did not create declared output")
    raw_output_identity = path_identity(output_path)
    try:
        output = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StageError("output-json", str(error)) from error
    require(isinstance(output, dict), "output-shape", "top-level child output must be a JSON object")
    reported_env = output.get("env")
    require(isinstance(reported_env, dict), "output-shape", "output env missing")
    require(reported_env.get("pid") == process.pid, "output-pid", f"child output PID {reported_env.get('pid')} differs from launched PID {process.pid}")
    require(reported_env.get("argv") == argv, "output-argv", "child output argv differs from launched argv")
    require(reported_env.get("exe_sha256_source") == "computed-from-running-exe", "binary-stamp", "stamp override is not accepted")
    require(reported_env.get("exe_sha256") == before_binary["sha256"], "binary-hash", "child computed exe hash differs from outer computed hash")
    expected_leaves(arm["expected_runtime"], output.get("runtime"))
    medians = extract_medians(plan, output, elapsed_ms)
    partial.update({
        "status": "admitted", "raw_output": raw_output_identity,
        "raw_output_semantics": output if plan["protocol_class"] == "synthetic-non-decision" else "retained at raw_output path",
        "runtime": output.get("runtime"), "medians": medians,
        "preflight_fingerprints": preflight_fingerprints(plan, output) if plan["lane"]["process_state"] == "warm" else None,
        "process_cold_label": "fresh OS process total, including startup and fixture work" if plan["lane"]["process_state"] == "process-cold" else "warm inner-kernel outer median; not page-cache-cold",
        "source": static["arms"][arm_name]["source"], "target_dir": static["arms"][arm_name]["target_dir"],
    })
    write_new_json(record_path, partial)
    partial["record"] = path_identity(record_path)
    return partial


def run_protocol(plan_path: Path) -> dict[str, Any]:
    plan = load_plan(plan_path)
    plan_identity = path_identity(plan_path)
    root = Path(plan["output_root"]).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "outputs").mkdir(exist_ok=True)
    (root / "telemetry").mkdir(exist_ok=True)
    summary_path = root / "summary.json"
    require(not summary_path.exists(), "output-create-new", f"summary exists: {summary_path}")
    orders = validate_orders(plan)
    write_new_json(root / "orders.json", orders)
    static = validate_static(plan)
    parity = validate_parity(plan)
    order_identity = path_identity(root / "orders.json")
    write_new_json(root / "static-provenance.json", static)
    runs: list[dict[str, Any]] = []
    block_evidence = []
    try:
        for pair_id, order_record in enumerate(orders):
            before = collect_block(plan, pair_id, "before", root / "telemetry")
            pair_runs = {}
            for sequence_index, arm_name in enumerate(order_record["order"]):
                try:
                    record = execute_run(plan, static, pair_id, arm_name, sequence_index, root)
                except StageError as error:
                    run_id = f"pair-{pair_id:02d}-{sequence_index}-{arm_name}"
                    run_dir = root / "runs" / run_id
                    failure_path = run_dir / "failure.json"
                    retained = {name: path_identity(run_dir / name) for name in ("record.json", "stdout.bin", "stderr.bin") if (run_dir / name).exists()}
                    raw_path = root / "outputs" / f"{run_id}.json"
                    if raw_path.exists():
                        retained["raw_output.json"] = path_identity(raw_path)
                    failure_record = {"run_id": run_id, "pair_id": pair_id, "arm": arm_name, "sequence_index": sequence_index, "status": "failed", "stage": error.stage, "error": str(error), "retained": retained}
                    write_new_json(failure_path, failure_record)
                    failure_record["record"] = path_identity(failure_path)
                    runs.append(failure_record)
                    raise
                runs.append(record)
                pair_runs[arm_name] = record
            if plan["lane"]["process_state"] == "warm":
                require(pair_runs["baseline"]["preflight_fingerprints"] == pair_runs["candidate"]["preflight_fingerprints"], "parity-admission", f"pair {pair_id} non-timed full-output preflight differs")
            after = collect_block(plan, pair_id, "after", root / "telemetry")
            block_evidence.append({"pair_id": pair_id, "order": order_record["order"], "before": before, "after": after})
    except StageError as error:
        failure = {"schema": "r3-protocol-failure-v1", "stage": error.stage, "error": str(error), "utc": utc_now(), "plan": plan_identity, "raw_run_records": [{"run_id": run["run_id"], "record": run.get("record")} for run in runs]}
        write_new_json(root / "failure.json", failure)
        raise
    serializable_runs = [{key: value for key, value in run.items() if key not in ("raw_output_semantics", "source")} for run in runs]
    write_new_json(root / "outer-runs.json", serializable_runs)
    statistics = analyze(serializable_runs, orders, seed=plan["seed"], bootstrap_samples=plan.get("bootstrap_samples", 10_000))
    summary = {
        "schema": "r3-paired-result-v1", "status": "qualified-synthetic-non-decision" if plan["protocol_class"] == "synthetic-non-decision" else "timing-admission-complete",
        "performance_claim": False, "decision_evidence": plan["protocol_class"] != "synthetic-non-decision",
        "protocol_class": plan["protocol_class"], "plan": plan_identity, "orders": order_identity,
        "lane": plan["lane"], "static_provenance": path_identity(root / "static-provenance.json"),
        "parity": parity, "raw_runs": path_identity(root / "outer-runs.json"),
        "telemetry_blocks": block_evidence, "statistics": statistics,
        "limitations": plan.get("measurement_limitations", []) + ["No page-cache-cold claim is made", "Thermal, host scheduling, SMT/cache contention and unavailable counters remain limitations"],
        "raw_protocol": "All argv, child env allowlist, PID/run/UTC identities, executable/fixture before-after hashes, affinity and raw outputs are retained",
    }
    write_new_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = run_protocol(args.plan)
    except (StageError, OSError, KeyError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "performance_claim": summary["performance_claim"], "pair_count": summary["statistics"]["pair_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
