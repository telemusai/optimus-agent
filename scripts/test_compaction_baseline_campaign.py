#!/usr/bin/env python3
"""R5 synthetic stdlib controls; no real fixtures, compiled examples or provider/network calls."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import compaction_baseline_campaign as campaign
import compaction_r4_lanes_mock as r4mock

A = "a" * 64
B = "b" * 64
C = "c" * 64


def pairing(calls: int = 1) -> dict:
    return {"tool_calls": calls, "tool_results": calls, "matched_results": calls, "unmatched_results": 0, "calls_without_result": 0, "duplicate_tool_call_ids": 0, "duplicate_tool_result_ids": 0}


def replay(case_name: str = "small-ascii", summary_calls: int = 1, failure: bool = False, *, use_frozen_restored_identity: bool = False) -> dict:
    count = 0 if failure else 1
    reader = {
        "schema": 1, "operation": "verify-reopen",
        "reader": {"pid": 2002, "exe_sha256": A},
        "input_bytes_before_after": 10, "input_sha256_before_after": B,
        "durable_sha256": B, "graph_entry_count": 5,
        "rehydrated_branch_sha256": B, "retained_state_sha256": C,
        "branch_entry_count": 5, "restored_message_count": 3,
        "raw_durable_pairing": pairing(), "restored_pairing": pairing(),
        "additional_user_or_provider_turns": 0, "read_only_reopen": True,
        "fresh_process": True,
        "parity": "full graph + exact raw durable branch + normalized durable build_session_context + protected custom/checkpoint state",
    }
    item = {
        "case": case_name,
        "trigger": {"expected_fire": True, "fired": True},
        "expected_admission": {"summary_calls": summary_calls, "turn_calls": 1, "injected_summary_failure": "R1 intentional summary failure" if failure else None},
        "durable_turn_delta": {"seed_unchanged": True, "added_message_counts": {"user": 1, "assistant": 1, "tool_results": 0, "other": 0}, "added_compaction_entries": count, "graph_entry_count": 5, "branch_entry_count": 5, "added_entry_count": 3, "added_custom_types": []},
        "raw_durable_pairing": pairing(), "restored_pairing": pairing(),
        "protected_state": {"checkpoints": ([{"type": "compaction"}] if not failure else []), "custom_messages": [], "session_controls": []},
        "provider_call_counts": {"summary": summary_calls, "turn": 1},
        "outcome": {"compaction_entries": count, "transient_failure_notice_count": 1 if failure else 0},
        "digests": {"retained_state_sha256": C, "reopen_context_sha256": C}, "fresh_process_reopen": {"report": reader},
    }
    if use_frozen_restored_identity:
        expected_identity = campaign.FROZEN_SUCCESSFUL_RESTORED_IDENTITIES[case_name]
        restored_calls = expected_identity["pairing"]["tool_calls"]
        restored_identity_hash = expected_identity["retained_state_sha256"]
        item["restored_pairing"] = pairing(restored_calls)
        reader["restored_pairing"] = pairing(restored_calls)
        reader["retained_state_sha256"] = restored_identity_hash
        item["digests"]["retained_state_sha256"] = restored_identity_hash
        item["digests"]["reopen_context_sha256"] = restored_identity_hash
    return {
        "harness": "compaction_replay", "subcommand": "replay", "schema": 1,
        "env": {"pid": 1001, "exe_sha256": A, "exe_sha256_source": "computed-from-running-exe"},
        "runtime": {}, "config": {
            "provider_delay_ms": 0, "summary_failure": "R1 intentional summary failure" if failure else None,
            "expect_outcome": "summary-failure" if failure else "successful-compaction",
            "expect_summary_calls": summary_calls, "context_window": 1000000.0,
            "keep_recent": None, "timeout_secs": 900,
        },
        "case_count": 1, "cases": [item], "failures": [],
    }


def parity_report() -> dict:
    cases = []
    for name in sorted(campaign.CASE_ORDER):
        row = {"case": name}
        for section in campaign.PARITY_SECTIONS:
            if section == "case":
                continue
            if section == "serialization":
                value = {"text": f"UTF8 {name} \u2603", "float": 1.25, "ordered": [1, 2]}
            else:
                value = {"source_case": name, "section": section, "finite": 1.0}
            if section.startswith("session_manager__"):
                row.setdefault("session_manager", {})[section.removeprefix("session_manager__")] = value
            else:
                row[section] = value
        row["full_projection_sha256"] = {
            "algorithm": campaign.FULL_PROJECTION_ALGORITHM,
            "note": campaign.FULL_PROJECTION_NOTE,
            "sections": {section: A for section in campaign.PARITY_SECTIONS},
        }
        cases.append(row)
    return {"schema": 2, "status": "ok", "comparison": "exact-full-output-against-baseline", "non_timed": True, "performance_claim": False, "config": {"chunk_context_window": 32768.0}, "cases": cases}


def control_report(family: str) -> dict:
    total, accepted, rejected = campaign.EXPECTED_CONTROL_COUNTS[family]
    rows = []
    for index in range(accepted):
        rows.append({"name": f"accepted-{index:02d}", "accepted": True})
    for index in range(rejected):
        rows.append({"name": f"rejected-{index:02d}", "rejected": True, "intended_stage": f"stage-{index:02d}", "error": f"[stage-{index:02d}] synthetic"})
    return {"schema": "synthetic", "status": "ok", "check_count": total, "checks": rows}


def control_expected(family: str) -> list[dict]:
    rows = control_report(family)["checks"]
    return [{"name": row["name"], "outcome": "accepted" if row.get("accepted") else "rejected", "intended_stage": row.get("intended_stage")} for row in rows]


def telemetry() -> dict:
    block = {}
    for field in campaign.TELEMETRY_FIELDS:
        if field == "active_power_scheme":
            block[field] = {"value": "synthetic-balanced-scheme", "unavailable_reason": None}
        elif field == "per_logical_processor":
            block[field] = {"value": [], "unavailable_reason": None}
        else:
            block[field] = {"value": None, "unavailable_reason": "synthetic unavailable"}
    return {"before": copy.deepcopy(block), "after": copy.deepcopy(block), "observation": dict(campaign.OBSERVATION_STATES), "required_fields": list(campaign.TELEMETRY_FIELDS)}


def affinity(pid: int = 101, wall: bool = False) -> dict:
    rows = [{"pid": pid, "role": "production", "observed_mask": "0x555", "running_image": "synthetic-production.exe"}]
    if wall:
        rows.insert(0, {"pid": 99, "role": "wrapper", "observed_mask": "0x555", "running_image": "synthetic-python.exe"})
    return {"requested_mask": "0x555", "children": rows}


def stats() -> dict:
    return {"n": 30, "mean_ms": 1.25, "p50_ms": 1.25, "p95_ms": 1.25, "stddev_ms": 0.1, "min_ms": 1.0, "max_ms": 1.5}


def warm(case_name: str = "small-ascii", jev_outcome: str = "prepared") -> dict:
    kernels = []
    for name in campaign.WARM_KERNELS:
        kernels.append({"name": name, "warmup": 10, "stats": stats(), "allocs_first_iter": {"count": 1, "bytes": 2}, "detail": {}})
    if jev_outcome == "expected-skip":
        kernels.append({"name": "jev_prepare_context", "outcome": "expected-skip", "skip_class": "no_compaction_candidates", "stats": None})
    else:
        kernels.append({"name": "jev_prepare_context", "outcome": "prepared", "warmup": 10, "stats": stats(), "detail": {}})
    return {
        "harness": "compaction_replay", "subcommand": "bench-local", "schema": 1,
        "env": {"pid": 321, "exe_sha256": A}, "runtime": {},
        "config": {"iters": 30, "warmup": 10, "chunk_context_window": 32768.0},
        "non_timed_preflight": [{"case": case_name, "full_output_sha256": B}],
        "cases": [{"case": case_name, "kernels": kernels}],
    }


def fresh_records(lane: str = "wall", count: int = 30) -> list[dict]:
    rows = []
    for index in range(count):
        value = ({"process_total_ms": 1.0 + index / 100.0} if lane == "wall" else {f"phase/{lane}": 2.0 + index / 100.0})
        rows.append({"status": "admitted", "lane": lane, "case": "small-ascii", "observation": index, "fresh_processes": 1, "pid": 5000 + index, "identity_sha256": A, "values": value})
    return rows


def paired_inputs() -> tuple[list[dict], list[dict]]:
    orders = campaign.make_balanced_orders(418393)
    rows = []
    for order in orders:
        for sequence_index, arm in enumerate(order["order"]):
            rows.append({"pair_id": order["pair_id"], "arm": arm, "sequence_index": sequence_index, "status": "admitted", "medians": {"metric": (1.0 if arm == "baseline" else 1.1) + order["pair_id"] / 1000}})
    return rows, orders


def protected_hashes(executables: dict[str, Path] | None = None) -> dict:
    rows = {kind: {f"synthetic-{kind}.input": A} for kind in ("source", "fixture", "toolchain", "script", "run_matrix")}
    rows["exe"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in executables.values()} if executables else {"synthetic-instrumented.exe": A, "synthetic-wall.exe": B, "synthetic-r3.exe": A}
    return rows


def direct_focused_plan(root: Path) -> tuple[dict, dict[str, str]]:
    deps = root / "optimus-compaction-r4-target/instrumented/release/deps"
    deps.mkdir(parents=True)
    logs = {}
    binaries_by_kind = {"focused99": [], "charcount15": []}
    expected_by_kind = {"focused99": [], "charcount15": []}
    sections = {}
    position = 0
    focused_text = []
    for suite_index, suite in enumerate(campaign.FOCUSED_TEST_SUITES):
        body = f"synthetic direct binary for {suite}; never executed".encode()
        digest = hashlib.sha256(body).hexdigest()
        exe = deps / f"{suite}-{digest[:16]}.exe"; exe.write_bytes(body)
        binaries_by_kind["focused99"].append({"exe": str(exe), "sha256": digest})
        relative = exe.relative_to(root).as_posix()
        focused_text.append(f"     Running tests\{suite}.rs ({relative})")
        count = 7 if suite_index < 13 else 8
        focused_text.append(f"running {count} tests")
        section = ["running " + str(count) + " tests"]
        for _ in range(count):
            name = f"{suite}::case_{position:03d}"; position += 1
            expected_by_kind["focused99"].append(name)
            focused_text.append(f"test {name} ... ok")
            section.append(f"test {name} ... ok")
        result_line = f"test result: ok. {count} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"
        focused_text.append(result_line); section.append(result_line)
        sections[str(exe)] = "\n".join(section)
    body = b"synthetic direct charcount binary; never executed"
    digest = hashlib.sha256(body).hexdigest()
    char_exe = deps / f"pi_coding_agent-{digest[:16]}.exe"; char_exe.write_bytes(body)
    binaries_by_kind["charcount15"].append({"exe": str(char_exe), "sha256": digest})
    char_text = [f"    Finished `test` profile [unoptimized] target(s)", f"     Running unittests src/lib.rs ({char_exe.relative_to(root).as_posix()})", "running 15 tests"]
    section = ["running 15 tests"]
    for index in range(15):
        name = f"core::compaction::charcount::tests::case_{index:02d}"
        expected_by_kind["charcount15"].append(name)
        char_text.append(f"test {name} ... ok"); section.append(f"test {name} ... ok")
    char_text.append("test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 4499 filtered out;")
    section.append(char_text[-1]); sections[str(char_exe)] = "\n".join(section)
    for kind, content in (("focused99", "\n".join(focused_text) + "\n"), ("charcount15", "\n".join(char_text) + "\n")):
        log = root / f"{kind}.log"; log.write_text(content)
        logs[kind] = {"path": log, "sha256": hashlib.sha256(log.read_bytes()).hexdigest(), "expected_names": expected_by_kind[kind]}
    plan = {kind: {"expected_names": logs[kind]["expected_names"], "provenance_sha256": logs[kind]["sha256"], "binaries": binaries_by_kind[kind]} for kind in logs}
    return plan, sections


class FakeGit:
    def __init__(self, patch: Path):
        self.calls = []
        self.patch = patch.read_bytes().decode("utf-8")
        self.outputs = {
            ("rev-parse", "HEAD"): campaign.R4_COMMIT + "\n",
            ("rev-parse", "HEAD^"): campaign.R4_BASE + "\n",
            ("branch", "--show-current"): "fix/compaction-r5-baseline\n",
            ("status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"): "\n".join(sorted(f"?? {path}" for path in campaign.R5_SOURCE_PATHS)) + "\n",
            ("ls-files", "--others", "--exclude-standard"): "\n".join(sorted(campaign.R5_SOURCE_PATHS)) + "\n",
            ("diff", "--name-only", "HEAD", "--"): "",
            ("diff", "--name-only", campaign.R4_BASE, campaign.R4_COMMIT, "--"): "\n".join(sorted(campaign.R4_CHANGED_PATHS)) + "\n",
            ("diff", "--binary", campaign.R4_BASE, campaign.R4_COMMIT): self.patch,
        }

    def __call__(self, root: Path, arguments: list[str]) -> dict:
        key = tuple(arguments)
        self.calls.append(key)
        return {"argv": arguments, "returncode": 0, "stdout": self.outputs[key], "stderr": ""}


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


class FakeTelemetry:
    def collect(self, phase: str, output: Path) -> dict:
        return copy.deepcopy(telemetry()[phase])


class RecordingAdapter:
    def __init__(self):
        self.receipts = []
        self.seen = set()
        self.missing_output_run = None
        self.summary_call_delta = 0

    def run(self, spec: campaign.ProcessSpec) -> campaign.ProcessReceipt:
        key = (spec.stage, spec.run_id)
        assert key not in self.seen
        self.seen.add(key)
        spec.work.mkdir(parents=True)
        argv = list(spec.argv)
        if len(argv) > 1 and argv[1] == "replay":
            fixture = Path(argv[argv.index("--fixture") + 1]).name
            summary_calls = int(argv[argv.index("--expect-summary-calls") + 1]) + self.summary_call_delta
            if spec.run_id != self.missing_output_run:
                campaign.write_new_json(spec.out, replay(fixture, summary_calls, use_frozen_restored_identity=True))
        else:
            campaign.write_new_json(spec.out, replay())
        spec.stdout.write_bytes(b"synthetic stdout")
        spec.stderr.write_bytes(b"")
        before = {"synthetic": {"script.py": {"path": "synthetic", "bytes": 1, "sha256": A}}}
        receipt = campaign.ProcessReceipt(spec, 9000 + len(self.receipts), 0, "synthetic-injected-adapter", before, copy.deepcopy(before), telemetry(), affinity(9000 + len(self.receipts)), "synthetic-start", "synthetic-end")
        self.receipts.append(receipt)
        return receipt

def small_ascii_parity_report():
    report = parity_report()
    report["cases"] = [row for row in report["cases"] if row["case"] == "small-ascii"]
    return report


def r12_executor(root, adapter):
    for name in ("inst.exe", "wall.exe", "r3.exe"):
        (root / name).write_bytes(("synthetic executable " + name).encode("ascii"))
    return campaign.CampaignExecutor(adapter, root / "artifacts", root / "fixtures", root / "repo", root / "inst.exe", root / "wall.exe", root / "r3.exe")


class R12JevAdapter(RecordingAdapter):
    """Synthetic R3/R4 reports and expected native failures; no OS child."""

    def __init__(self):
        super().__init__()
        self.error_override = None
        self.negative_exit = 1
        self.negative_stdout = b""
        self.write_negative_output = False

    def _receipt(self, spec, exit_code, stderr, stdout=b"", report=None):
        key = (spec.stage, spec.run_id)
        assert key not in self.seen
        self.seen.add(key)
        spec.work.mkdir(parents=True)
        if report is not None:
            assert not spec.out.exists(), "synthetic R12 reports must be new"
            # Native R4 uses serde_json preserve_order; do not use the R5 sorted-key evidence writer here.
            spec.out.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        elif self.write_negative_output:
            campaign.write_new_json(spec.out, {"synthetic_unexpected_output": True})
        spec.stdout.write_bytes(stdout)
        spec.stderr.write_bytes((stderr + "\n").encode("utf-8"))
        before = {"synthetic": {"script.py": {"path": "synthetic", "bytes": 1, "sha256": A}}}
        receipt = campaign.ProcessReceipt(spec, 9000 + len(self.receipts), exit_code, "synthetic-r12-adapter", before, copy.deepcopy(before), telemetry(), affinity(9000 + len(self.receipts)), "synthetic-start", "synthetic-end")
        self.receipts.append(receipt)
        return receipt

    def run(self, spec):
        argv = list(spec.argv)
        if "verify-local" in argv:
            fixture = Path(argv[argv.index("--fixture") + 1])
            if fixture.name != "small-ascii":
                return self._receipt(spec, self.negative_exit, self.error_override or campaign.JEV_INPUT_LIMIT_STDERR, self.negative_stdout)
            return self._receipt(spec, 0, "", b"synthetic local parity stdout", small_ascii_parity_report())
        if "bench-local" in argv:
            case_name = Path(argv[argv.index("--fixture") + 1]).name
            error = self.error_override or f"error: {case_name}: non-timed preflight failed: [jev-prepare] unexpected production Jev prepare_context error: compaction_input_limit"
            return self._receipt(spec, self.negative_exit, error, self.negative_stdout)
        raise AssertionError(f"unexpected synthetic R12 argv: {argv}")




R13_NATIVE_NEGATIVE_SAMPLE_FACTS = {
    # Raw independently reviewed probe examples, not future thread/PID allowlists.
    "small-emoji": {"native_pid": 29160, "panic_thread_id": 20712, "stderr_sha256": "5e28b74c02892782c01c699c2d6e358e13f11f64abde7529a2e49621667f5537"},
    "small-tool-huge": {"native_pid": 24644, "panic_thread_id": 2756, "stderr_sha256": "9a7d6957918e66bc8966781b830c68496b3b0d4f98738fab7d068d67742a360b"},
    "small-many-small": {"native_pid": 17680, "panic_thread_id": 27000, "stderr_sha256": "7a6f6641301a2f23f4b07c0bc0dbf6012f65661a1a37b1c15385ab849d521793"},
}


def r13_native_panic(case_name, *, thread_id=None, source_line="200", marker=campaign.R4_DUPLICATE_SUMMARY_MARKER):
    sample = R13_NATIVE_NEGATIVE_SAMPLE_FACTS[case_name]
    dynamic_thread = sample["panic_thread_id"] if thread_id is None else thread_id
    return "\n" + (
        f"thread 'main' ({dynamic_thread}) panicked at crates\\pi-coding-agent\\examples\\compaction_replay\\r4_lanes.rs:{source_line}:9:\n"
        f"{marker}\n"
        "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"
    )


class R13EligibilityAdapter(R12JevAdapter):
    """Synthetic process results; never a native, fixture, provider or timing run."""

    def __init__(self):
        super().__init__()
        self.negative_exit_overrides = {}
        self.negative_stdout_overrides = {}
        self.negative_stderr_overrides = {}
        self.negative_report_cases = set()
        self.thread_id_overrides = {}
        self.unsupported_success_cases = set()
        self.additional_negative_cases = set()
        self.transposed_argv_case = None

    def _native_image(self, spec):
        return str(spec.expected_running_image or Path(spec.argv[0]))

    def run(self, spec):
        if spec.run_id.endswith("-wall-continuation"):
            report = r4mock.external_wall()
            case_name = Path(spec.argv[spec.argv.index("--fixture") + 1]).name
        elif spec.run_id.endswith("-instrumented-continuation"):
            case_name = Path(spec.argv[spec.argv.index("--fixture") + 1]).name
            report = r4mock.lane_report("replay-phase")
        elif list(spec.argv)[1] == "r4-lane":
            case_name = Path(spec.argv[spec.argv.index("--fixture") + 1]).name
            if self.transposed_argv_case == case_name:
                raw_argv = list(spec.argv)
                fixture_index = raw_argv.index("--fixture") + 1
                raw_argv[fixture_index] = str(Path(raw_argv[fixture_index]).with_name("small-tool-huge" if case_name == "small-emoji" else "small-emoji"))
                spec = dataclasses.replace(spec, argv=tuple(raw_argv))
            if case_name in campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES and case_name not in self.unsupported_success_cases:
                stderr = self.negative_stderr_overrides.get(case_name, r13_native_panic(case_name, thread_id=self.thread_id_overrides.get(case_name)))
                stdout = self.negative_stdout_overrides.get(case_name, b"")
                native_exit = self.negative_exit_overrides.get(case_name, 101)
                report = r4mock.lane_report("replay-phase") if case_name in self.negative_report_cases else None
                receipt = self._receipt(spec, native_exit, stderr.rstrip("\r\n"), stdout, report)
                native_pid = R13_NATIVE_NEGATIVE_SAMPLE_FACTS[case_name]["native_pid"]
            elif case_name in self.additional_negative_cases:
                receipt = self._receipt(spec, 101, r13_native_panic("small-emoji").rstrip("\r\n"), b"")
                native_pid = receipt.pid
            else:
                supported_report = r4mock.lane_report("replay-phase")
                supported_report["fixture"]["name"] = case_name
                receipt = self._receipt(spec, 0, "", b"", supported_report)
                native_pid = receipt.pid
            receipt.pid = native_pid
            # Synthetic eligibility is not timed-affinity evidence.
            receipt.affinity = None
            receipt.running_image = self._native_image(spec)
            self.receipts[-1] = receipt
            return receipt
        else:
            raise AssertionError(f"unexpected synthetic R13 eligibility argv: {list(spec.argv)}")
        if report.get("lane") == "wall":
            child_report = report["child_report"]
        else:
            child_report = report
        child_report["fixture"]["name"] = case_name
        continuation = {
            "status": "available",
            "boundary": "real append_compaction followed by build_session_context(None)",
            "message_count": 18 if case_name == "small-ascii" else 2,
            "context_sha256": "f7a5df688012e2c31f1e36a98708a2f596a2abe6ce56fd810cabbdb5d38d317a" if case_name == "small-ascii" else B,
            "compaction_entries": 1,
        }
        child_report["continuation"] = continuation
        receipt = self._receipt(spec, 0, "", b"", report)
        receipt.running_image = self._native_image(spec)
        receipt.affinity = None
        return receipt


class R14ReaderAdapter(R12JevAdapter):
    """Case-specific synthetic R4 bytes; no native, parser, provider or timing process."""

    READER_BODIES = {
        "reader-truncated": b'{"schema": ',
        "reader-non-array": b"[]",
        "reader-nonfinite": b'{"x": NaN}',
    }
    READER_STAGES = {
        "reader-truncated": "malformed-json",
        "reader-non-array": "[schema] compaction-r4-lane-v1 object required",
        "reader-nonfinite": "malformed-json",
        "reader-wrong-lane": "lane-identity",
    }
    PROCESS_ERRORS = {
        "instrumented-wall-rejected": (1, b"error: [r4-build-lane-identity] synthetic wall rejection\n"),
        "wall-allocation-rejected": (1, b"error: [r4-build-lane-identity] synthetic allocation rejection\n"),
        "provider-control-in-allocation-rejected": (1, b"error: [r4-lane-exclusivity] synthetic exclusivity rejection\n"),
        "float-generation-rejected": (1, b"error: [cli-generation-units] synthetic integer requirement\n"),
        "nan-wait-rejected": (1, b"error: [cli-mock-wait-ms] synthetic finite integer requirement\n"),
        "allocation-source-wrong-lane-rejected": (1, b"error: [r4-allocation-source] synthetic lane rejection\n"),
        "duplicate-lane-rejected": (2, b"error: [cli-parse] duplicate --lane\n"),
    }

    def __init__(self):
        super().__init__()
        self.guard = None
        self.reader_stderr_overrides = {}
        self.reader_stdout_overrides = {}
        self.reader_exit_overrides = {}
        self.reader_output_cases = set()
        self.received_argv = {}

    def run(self, spec):
        name = spec.run_id
        self.received_argv[name] = list(spec.argv)
        if name in {"allocation-available", "allocation-unavailable", "replay-phase", "minimal-provider", "provider-failure", "provider-cancelled", "external-wall", "reader-wrong-lane-source"}:
            if name == "external-wall":
                report = r4mock.external_wall()
            else:
                lane = "allocation" if name.startswith("allocation-") else ("replay-phase" if name == "replay-phase" else "minimal-provider")
                outcome = "failure" if name == "provider-failure" else ("cancelled" if name == "provider-cancelled" else "success")
                report = r4mock.lane_report(lane, outcome)
                if name == "allocation-unavailable":
                    report["allocation"] = {"status": "unavailable", "source": None, "count": None, "bytes_requested": None, "reason": "synthetic allocation unavailable"}
                if name in {"provider-failure", "provider-cancelled"}:
                    report["outcome"]["status"] = outcome
                    report["continuation"]["status"] = "unavailable"
                    report["provider"]["call_count"] = 0 if outcome == "cancelled" else 1
            return self._receipt(spec, 0, "", b"", report)
        if name in self.PROCESS_ERRORS:
            exit_code, stderr = self.PROCESS_ERRORS[name]
            return self._receipt(spec, exit_code, stderr.decode("ascii").rstrip("\n"), b"")
        if name in self.READER_STAGES:
            if name in self.READER_BODIES:
                report_path = Path(spec.argv[spec.argv.index("--report") + 1])
                assert report_path.read_bytes() == self.READER_BODIES[name], f"raw reader bytes changed for {name}"
            stderr = self.reader_stderr_overrides.get(name)
            if stderr is None:
                stderr = f"error: [{self.READER_STAGES[name]}] synthetic R14 {name}\r\n".encode("ascii")
                if name == "reader-non-array":
                    stderr = b'[schema] compaction-r4-lane-v1 object required\r\n'
            stdout = self.reader_stdout_overrides.get(name, b"")
            exit_code = self.reader_exit_overrides.get(name, 2)
            self.write_negative_output = name in self.reader_output_cases
            report = {"synthetic_unexpected_reader_output": True} if self.write_negative_output else None
            receipt = self._receipt(spec, exit_code, stderr.decode("utf-8").rstrip("\r\n"), stdout, report)
            spec.stderr.write_bytes(stderr)
            return receipt
        if name == "create-new-collision-rejected":
            stderr = f"error: create_new {spec.out}: synthetic file exists\n"
            return self._receipt(spec, 1, stderr.rstrip("\n"), b"")
        raise AssertionError(f"unexpected synthetic R14 argv: {list(spec.argv)}")


def r14_executor(root, adapter):
    for name in ("inst.exe", "wall.exe", "r3.exe"):
        (root / name).write_bytes(("synthetic R14 executable " + name).encode("ascii"))
    (root / "fixtures" / "small-ascii").mkdir(parents=True)
    marker = root / "protected.json"
    marker.write_bytes(b"synthetic protected input\n")
    digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    classes = {kind: {str(marker): digest} for kind in ("source", "fixture", "toolchain", "script", "exe", "run_matrix")}
    adapter.guard = campaign.IntegrityGuard(root, classes)
    return campaign.CampaignExecutor(adapter, root / "artifacts", root / "fixtures", SCRIPT_DIR.parent, root / "inst.exe", root / "wall.exe", root / "r3.exe")


class R5SyntheticControls(unittest.TestCase):
    def test_r14_reader_expectations_and_exact_synthetic_receipt_bytes(self):
        expected = {
            "reader-truncated": (b'{"schema": ', "malformed-json", b"error: [malformed-json] synthetic R14 reader-truncated\r\n"),
            "reader-non-array": (b"[]", "[schema] compaction-r4-lane-v1 object required", b'[schema] compaction-r4-lane-v1 object required\r\n'),
            "reader-nonfinite": (b'{"x": NaN}', "malformed-json", b"error: [malformed-json] synthetic R14 reader-nonfinite\r\n"),
            "reader-wrong-lane": (None, "lane-identity", b"error: [lane-identity] synthetic R14 reader-wrong-lane\r\n"),
        }
        with tempfile.TemporaryDirectory() as temp:
            adapter = R14ReaderAdapter()
            executor = r14_executor(Path(temp), adapter)
            report = executor.run_r4_negative_controls()
            self.assertEqual(report["accepted_count"], 7)
            self.assertEqual(len(report["process_rejections"]), 7)
            self.assertEqual(len(report["reader_rejections"]), 4)
            self.assertEqual(len(adapter.receipts), 20)
            for item in report["reader_rejections"]:
                name = item["name"]
                body, stage, stderr = expected[name]
                self.assertEqual(item, {"name": name, "outcome": "rejected", "intended_stage": stage, "exit_code": 2})
                root = Path(temp) / "artifacts/gate-a/r4-negative"
                self.assertEqual((root / name / "stderr.bin").read_bytes(), stderr)
                self.assertEqual((root / name / "stdout.bin").read_bytes(), b"")
                self.assertFalse((root / name / "out.json").exists())
                if body is not None:
                    self.assertEqual((root / f"{name}-input.json").read_bytes(), body)
                receipt = json.loads((root / name / "work/receipt.json").read_text())
                self.assertEqual(receipt["argv"], adapter.received_argv[name])
                self.assertEqual(receipt["exit_code"], 2)
                self.assertIsNone(receipt["raw_output"])
            self.assertEqual(report["process_rejections"][-1]["intended_stage"], "cli-parse")
            self.assertEqual(report["collision"]["attempt_count"], 1)

    def test_r14_rejects_wrong_reader_labels_protocol_changes_and_bad_argv(self):
        exact_non_array = b'[schema] compaction-r4-lane-v1 object required\r\n'
        moved = b"error: [malformed-json] synthetic old misclassification\r\n"
        cases = (
            ("non-array-old-label", "reader-non-array", "invalid-cli-stage", {"stderr": moved}),
            ("non-array-wrong-schema", "reader-non-array", "invalid-cli-stage", {"stderr": b"error: [schema] unrelated object required\r\n"}),
            ("non-array-unrelated", "reader-non-array", "invalid-cli-stage", {"stderr": b"error: [unrelated] not the reader error\r\n"}),
            ("non-array-missing", "reader-non-array", "invalid-cli-stage", {"stderr": b""}),
            ("non-array-line-extra-without-full-line", "reader-non-array", "invalid-cli-stage", {"stderr": b"error: [schema] object-required by another reader\nextra line\r\n"}),
            ("truncated-label-moved", "reader-truncated", "invalid-cli-stage", {"stderr": exact_non_array}),
            ("nonfinite-label-moved", "reader-nonfinite", "invalid-cli-stage", {"stderr": exact_non_array}),
            ("non-array-zero-exit", "reader-non-array", "invalid-cli-no-output", {"exit": 0}),
            ("non-array-one-exit", "reader-non-array", "r4-reader-exit", {"exit": 1}),
            ("non-array-report-output", "reader-non-array", "invalid-cli-no-output", {"output": True}),
        )
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            for label, case, expected_stage, override in cases:
                with self.subTest(case=label):
                    root = temp_root / label
                    root.mkdir()
                    adapter = R14ReaderAdapter()
                    if "stderr" in override:
                        adapter.reader_stderr_overrides[case] = override["stderr"]
                    if "exit" in override:
                        adapter.reader_exit_overrides[case] = override["exit"]
                    if override.get("output"):
                        adapter.reader_output_cases.add(case)
                    executor = r14_executor(root, adapter)
                    with self.assertRaises(campaign.Reject) as caught:
                        executor.run_r4_negative_controls()
                    self.assertEqual(caught.exception.stage, expected_stage)

            root = temp_root / "bad-argv"
            root.mkdir()
            adapter = R14ReaderAdapter()
            executor = r14_executor(root, adapter)
            report = executor.root / "synthetic/non-array-input.json"
            campaign.write_new_bytes(report, b"[]")
            out = executor.root / "synthetic/reader-non-array/out.json"
            validator = executor.repo_root / "scripts/compaction_r4_lanes.py"
            argv = [sys.executable, validator, "validate", "--report", report, "--lane", "wall", "--out", out]
            for label, mutated in (
                ("unsupported-lane", [*argv[:-3], "network", *argv[-2:]]),
                ("outside-out", [*argv[:-1], root / "outside.json"]),
                ("duplicate-option", [*argv, "--lane", "wall"]),
            ):
                with self.subTest(argv=label):
                    before = len(adapter.receipts)
                    self.assert_stage("command-catalog", executor._expected_failure, "synthetic/reader-non-array", label, mutated, "[schema] compaction-r4-lane-v1 object required")
                    self.assertEqual(len(adapter.receipts), before)

    def test_r13_four_supported_cases_full_reports_and_three_native_negatives(self):
        for override in (False, True):
            with self.subTest(dynamic_thread_override=override), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                adapter = R13EligibilityAdapter()
                if override:
                    adapter.thread_id_overrides = {case: facts["panic_thread_id"] + 100 for case, facts in R13_NATIVE_NEGATIVE_SAMPLE_FACTS.items()}
                executor = r12_executor(root, adapter)
                synthetic_exe_sha = hashlib.sha256((root / "inst.exe").read_bytes()).hexdigest()
                with mock.patch.object(campaign, "R4_INSTRUMENTED_EXE_SHA256", synthetic_exe_sha):
                    eligible, eligibility, continuations, noneligible = executor.run_r4_eligibility()
                self.assertEqual(tuple(eligible), campaign.R4_ONE_CALL_TIMING_CASES)
                self.assertEqual(tuple(continuations), campaign.R4_ONE_CALL_TIMING_CASES)
                self.assertEqual(tuple(noneligible), campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES)
                self.assertEqual(tuple(eligibility), tuple(name for name in campaign.CASE_ORDER if name in campaign.R3_ONE_SUMMARY_CALL_CASES))
                self.assertEqual(len(adapter.receipts), 15)
                self.assertEqual(len(executor.all_pids), 15)
                self.assertEqual({receipt.pid for receipt in adapter.receipts if receipt.spec.run_id in campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES}, {R13_NATIVE_NEGATIVE_SAMPLE_FACTS[case]["native_pid"] for case in campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES})
                self.assertTrue(all(receipt.affinity is None for receipt in adapter.receipts))
                self.assertEqual(sum(receipt.spec.run_id in campaign.R4_ONE_CALL_TIMING_CASES for receipt in adapter.receipts), 4)
                self.assertEqual(sum(receipt.spec.run_id.endswith("-instrumented-continuation") for receipt in adapter.receipts), 4)
                self.assertEqual(sum(receipt.spec.run_id.endswith("-wall-continuation") for receipt in adapter.receipts), 4)
                for case_name in campaign.R4_ONE_CALL_TIMING_CASES:
                    self.assertIs(eligibility[case_name]["eligible"], True)
                    report = campaign.load_json(executor.root / "gate-a/r4-eligibility" / case_name / "out.json")
                    campaign.extract_one_shot(report, "replay-phase")
                    self.assertEqual(report["fixture"]["name"], case_name)
                    self.assertEqual(continuations[case_name]["status"], "available")
                for case_name in campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES:
                    row = noneligible[case_name]
                    sample = R13_NATIVE_NEGATIVE_SAMPLE_FACTS[case_name]
                    expected_thread = sample["panic_thread_id"] + 100 if override else sample["panic_thread_id"]
                    expected_stderr = r13_native_panic(case_name, thread_id=expected_thread)
                    self.assertEqual(row["native_pid"], sample["native_pid"])
                    self.assertEqual(row["panic_thread_id"], expected_thread)
                    self.assertNotEqual(row["panic_thread_id"], row["native_pid"])
                    self.assertEqual(row["stderr_text"], expected_stderr)
                    self.assertEqual(row["stderr"]["sha256"], hashlib.sha256(expected_stderr.encode()).hexdigest())
                    self.assertEqual(row["process_receipt_sha256"], hashlib.sha256((executor.root / "gate-a/r4-eligibility" / case_name / "work/receipt.json").read_bytes()).hexdigest())
                    self.assertEqual(row["exit_code"], 101)
                    self.assertIs(eligibility[case_name]["eligible"], False)
                    self.assertEqual(eligibility[case_name]["status"], "unavailable_noneligible")
                    self.assertIs(row["timing_observation"], False)
                    self.assertIs(row["jev_skip"], False)
                    self.assertEqual(row["timed_observations"], [])
                    self.assertIsNone(row["statistics"])
                    self.assertIsNone(row["out_json"])
                    self.assertEqual(row["stdout"]["sha256"], hashlib.sha256(b"").hexdigest())
                    self.assertFalse((executor.root / "gate-a/r4-eligibility" / case_name / "out.json").exists())
                self.assertEqual(campaign.validate_r4_noneligible_records(list(noneligible.values())), list(noneligible.values()))
                incomplete = list(noneligible.values())[:-1]
                self.assert_stage("r4-noneligible-count", campaign.validate_r4_noneligible_records, incomplete)
                transposed = list(noneligible.values())
                transposed[0], transposed[1] = transposed[1], transposed[0]
                self.assert_stage("r4-noneligible-case", campaign.validate_r4_noneligible_records, transposed)

    def test_r13_native_negative_rejects_wrong_case_marker_source_exit_argv_stdout_and_report(self):
        variants = (
            ("success", "unsupported_success_cases", "r4-eligibility-supported-case"),
            ("unexpected_exit", "negative_exit_overrides", "r4-preflight-exit"),
            ("wrong_marker", "negative_stderr_overrides", "r4-native-panic-boundary"),
            ("wrong_source_line", "negative_stderr_overrides", "r4-native-panic-boundary"),
            ("empty_thread_id", "thread_id_overrides", "r4-native-thread-id"),
            ("thread_equals_native_pid", "thread_id_overrides", "r4-native-thread-id"),
            ("nonempty_stdout", "negative_stdout_overrides", "r4-noneligible-output"),
            ("report_present", "negative_report_cases", "r4-noneligible-output"),
            ("transposed_argv", "transposed_argv_case", "r4-noneligible-argv"),
        )
        for case_name in campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES:
            for label, knob, expected_stage in variants:
                with self.subTest(case=case_name, variant=label), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp); adapter = R13EligibilityAdapter()
                    if label == "success":
                        adapter.unsupported_success_cases.add(case_name)
                    elif label == "unexpected_exit":
                        adapter.negative_exit_overrides[case_name] = 2
                    elif label == "wrong_marker":
                        adapter.negative_stderr_overrides[case_name] = r13_native_panic(case_name, marker="duplicate phase mark unrelated_stage")
                    elif label == "wrong_source_line":
                        adapter.negative_stderr_overrides[case_name] = r13_native_panic(case_name, source_line="201")
                    elif label == "empty_thread_id":
                        adapter.thread_id_overrides[case_name] = 0
                    elif label == "thread_equals_native_pid":
                        adapter.thread_id_overrides[case_name] = R13_NATIVE_NEGATIVE_SAMPLE_FACTS[case_name]["native_pid"]
                    elif label == "nonempty_stdout":
                        adapter.negative_stdout_overrides[case_name] = b"unexpected noneligible stdout"
                    elif label == "report_present":
                        adapter.negative_report_cases.add(case_name)
                    elif label == "transposed_argv":
                        adapter.transposed_argv_case = case_name
                    executor = r12_executor(root, adapter)
                    synthetic_exe_sha = hashlib.sha256((root / "inst.exe").read_bytes()).hexdigest()
                    with mock.patch.object(campaign, "R4_INSTRUMENTED_EXE_SHA256", synthetic_exe_sha):
                        self.assert_stage(expected_stage, executor.run_r4_eligibility)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); adapter = R13EligibilityAdapter(); adapter.additional_negative_cases.add("small-ascii")
            executor = r12_executor(root, adapter)
            synthetic_exe_sha = hashlib.sha256((root / "inst.exe").read_bytes()).hexdigest()
            with mock.patch.object(campaign, "R4_INSTRUMENTED_EXE_SHA256", synthetic_exe_sha):
                self.assert_stage("r4-noneligible-case", executor.run_r4_eligibility)

    def test_r13_matrix_keeps_four_r4_cases_separate_from_r3_and_warm_expectations(self):
        self.assertEqual(campaign.R3_ONE_SUMMARY_CALL_CASES, campaign.EXPECTED_ONE_CALL_CASES)
        self.assertEqual(campaign.R4_ONE_CALL_TIMING_CASES, ("small-ascii", "small-cjk", "small-mixed-crlf", "medium-cjk"))
        self.assertEqual(campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES, ("small-emoji", "small-tool-huge", "small-many-small"))
        self.assertEqual(campaign.R3_ONE_SUMMARY_CALL_CASES - set(campaign.R4_ONE_CALL_TIMING_CASES), set(campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES))
        matrix = campaign.make_run_matrix(campaign.R4_ONE_CALL_TIMING_CASES)
        self.assertEqual(len(matrix), 750)
        self.assertEqual([row["order"] for row in matrix], list(range(750)))
        one_call = [row for row in matrix if row["lane"] != "bench-local-warm"]
        warm = [row for row in matrix if row["lane"] == "bench-local-warm"]
        self.assertEqual(len(one_call), 480)
        self.assertEqual(len(warm), 270)
        self.assertEqual({row["case"] for row in one_call}, set(campaign.R4_ONE_CALL_TIMING_CASES))
        self.assertEqual({row["case"] for row in one_call} & set(campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES), set())
        for lane in campaign.ONE_SHOT_LANES:
            lanes = [row for row in one_call if row["lane"] == lane]
            self.assertEqual(len(lanes), 120)
            self.assertEqual({row["case"] for row in lanes}, set(campaign.R4_ONE_CALL_TIMING_CASES))
            for case_name in campaign.R4_ONE_CALL_TIMING_CASES:
                self.assertEqual(sum(row["case"] == case_name for row in lanes), 30)
        self.assertEqual(len(campaign.WARM_CASES) * 3 * 10, 270)
        self.assertEqual(len(campaign.UNAVAILABLE_WARM_PREFLIGHTS), 12)
        self.assertEqual(set(campaign.CASE_ORDER), set(campaign.CASE_SUMMARY_CALLS))
        self.assertEqual(campaign.CASE_SUMMARY_CALLS["small-ascii"], 1)
        self.assertEqual(campaign.CASE_SUMMARY_CALLS["small-emoji"], 1)
        self.assertEqual(campaign.CASE_SUMMARY_CALLS["small-tool-huge"], 1)
        self.assertEqual(campaign.CASE_SUMMARY_CALLS["small-many-small"], 1)
        self.assertEqual(campaign.CASE_SUMMARY_CALLS["large-ascii"], 4)
        self.assertEqual(campaign.CASE_SUMMARY_CALLS["xlarge-ascii"], 10)
        for names in (campaign.R4_DUPLICATE_SUMMARY_NONELIGIBLE_CASES, ["small-ascii"], []):
            self.assert_stage("r4-eligibility", campaign.make_run_matrix, names)

    def test_r13_retained_sample_identities_and_noneligible_record_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); adapter = R13EligibilityAdapter(); executor = r12_executor(root, adapter)
            with mock.patch.object(campaign, "R4_INSTRUMENTED_EXE_SHA256", hashlib.sha256((root / "inst.exe").read_bytes()).hexdigest()):
                _, _, _, noneligible = executor.run_r4_eligibility()
            for case_name, sample in R13_NATIVE_NEGATIVE_SAMPLE_FACTS.items():
                raw = r13_native_panic(case_name).encode("utf-8")
                self.assertEqual(hashlib.sha256(raw).hexdigest(), sample["stderr_sha256"])
                self.assertNotEqual(sample["native_pid"], sample["panic_thread_id"])
            self.assertEqual(noneligible["small-emoji"]["stderr"]["sha256"], "5e28b74c02892782c01c699c2d6e358e13f11f64abde7529a2e49621667f5537")
            self.assertEqual(noneligible["small-tool-huge"]["stderr"]["sha256"], "9a7d6957918e66bc8966781b830c68496b3b0d4f98738fab7d068d67742a360b")
            self.assertEqual(noneligible["small-many-small"]["stderr"]["sha256"], "7a6f6641301a2f23f4b07c0bc0dbf6012f65661a1a37b1c15385ab849d521793")
            for variant in ("success", "native_pid", "argv", "stderr", "report", "timing", "processes"):
                rows = list(noneligible.values())
                row = copy.deepcopy(rows[0]); other = rows[1:]
                if variant == "success": row["status"] = "success"
                elif variant == "native_pid": row["native_pid"] = row["panic_thread_id"]
                elif variant == "argv": row["argv"][1] = "replay"
                elif variant == "stderr": row["stderr_text"] = row["stderr_text"].replace("200:9", "201:9")
                elif variant == "report": row["out_json"] = {"status": "success"}; row["out_json_absent"] = False
                elif variant == "timing": row["statistics"] = {"duration_ms": 0.0}
                elif variant == "processes": row["fresh_processes"] = 2
                self.assert_stage({"success": "r4-noneligible-record", "native_pid": "r4-noneligible-record", "argv": "r4-noneligible-record", "stderr": "r4-noneligible-record", "report": "r4-noneligible-output", "timing": "r4-noneligible-record", "processes": "r4-noneligible-record"}[variant], campaign.validate_r4_noneligible_records, [row] + other)


    def assert_stage(self, stage: str, function, *args, **kwargs):
        with self.assertRaises(campaign.Reject) as caught:
            function(*args, **kwargs)
        self.assertEqual(stage, caught.exception.stage, str(caught.exception))

    def test_valid_ordered_control_families(self):
        reports = {family: control_report(family) for family in campaign.EXPECTED_CONTROL_COUNTS}
        expected = {family: control_expected(family) for family in campaign.EXPECTED_CONTROL_COUNTS}
        result = campaign.signing_control_summary(reports, expected)
        self.assertEqual([result["signatures"][family][0]["outcome"] for family in result["signatures"]], ["accepted"] * 4)
        for family in campaign.EXPECTED_CONTROL_COUNTS:
            self.assertEqual(len(result["signatures"][family]), campaign.EXPECTED_CONTROL_COUNTS[family][0])

    def test_r3_rejection_receipt_stderr_is_accepted(self):
        report = control_report("r3")
        for row in report["checks"]:
            if row.get("rejected"):
                row["receipt"] = {"exit_code": 2, "stderr": row.pop("error")}
        self.assertEqual(len(campaign.control_signature(report, "r3", control_expected("r3"))), 71)

    def test_control_count_order_stage_and_outcome_rejections(self):
        expected = control_expected("r4")
        wrong_count = control_report("r4"); wrong_count["check_count"] = 71
        self.assert_stage("r4-control-count", campaign.control_signature, wrong_count, "r4", expected)
        wrong_order = control_report("r4"); wrong_order["checks"].reverse()
        self.assert_stage("r4-ordered-signature", campaign.control_signature, wrong_order, "r4", expected)
        wrong_stage = control_report("r4"); wrong_stage["checks"][10]["error"] = "[other] synthetic"
        self.assert_stage("r4-control-stage", campaign.control_signature, wrong_stage, "r4", expected)
        both = control_report("r4"); both["checks"][0]["rejected"] = True
        self.assert_stage("r4-control-outcome", campaign.control_signature, both, "r4", expected)

    def test_replay_success_failure_and_all21(self):
        success = campaign.validate_replay_case(replay(), "small-ascii", 1)
        self.assertEqual((success["writer_pid"], success["reader_pid"]), (1001, 2002))
        failed = campaign.validate_replay_case(replay(failure=True), "small-ascii", 1, failure=True)
        self.assertEqual((failed["compaction_entries"], failed["outcome"]), (0, "summary-failure"))
        reports = {name: replay(name, count, use_frozen_restored_identity=True) for name, count in campaign.CASE_SUMMARY_CALLS.items()}
        self.assertEqual(campaign.aggregate_all21(reports)["case_count"], 21)

    def test_replay_wrong_summary_pid_pairing_and_checkpoint_rejections(self):
        wrong_n = replay(); wrong_n["provider_call_counts"] = {}
        self.assert_stage("replay-report", campaign.validate_replay_case, wrong_n, "small-ascii", 1)
        wrong_n = replay(); wrong_n["cases"][0]["provider_call_counts"]["summary"] = 2
        self.assert_stage("replay-provider-counts", campaign.validate_replay_case, wrong_n, "small-ascii", 1)
        wrong_n = replay()
        self.assert_stage("replay-summary-N", campaign.validate_replay_case, wrong_n, "small-ascii", 2)
        same_pid = replay(); same_pid["cases"][0]["fresh_process_reopen"]["report"]["reader"]["pid"] = 1001
        self.assert_stage("replay-writer-reader-pid", campaign.validate_replay_case, same_pid, "small-ascii", 1)
        bad_pair = replay(); bad_pair["cases"][0]["raw_durable_pairing"]["duplicate_tool_call_ids"] = 1
        self.assert_stage("replay-raw-pairing", campaign.validate_replay_case, bad_pair, "small-ascii", 1)
        extra_checkpoint = replay(); extra_checkpoint["cases"][0]["protected_state"]["checkpoints"].append({})
        self.assert_stage("replay-checkpoint-links", campaign.validate_replay_case, extra_checkpoint, "small-ascii", 1)
        wrong_reader_graph = replay(); wrong_reader_graph["cases"][0]["fresh_process_reopen"]["report"]["rehydrated_branch_sha256"] = A
        self.assert_stage("replay-reader-graph", campaign.validate_replay_case, wrong_reader_graph, "small-ascii", 1)

    def test_all21_missing_extra_and_reordered_cases(self):
        reports = {name: replay(name, count, use_frozen_restored_identity=True) for name, count in campaign.CASE_SUMMARY_CALLS.items()}
        missing = dict(reports); missing.pop("small-cjk")
        self.assert_stage("all21-case-order", campaign.aggregate_all21, missing)
        extra = dict(reports); extra["unexpected"] = replay("unexpected", 1)
        self.assert_stage("all21-case-order", campaign.aggregate_all21, extra)
        reordered = dict(reports)
        first = reordered.pop("small-ascii"); reordered["small-ascii"] = first
        self.assert_stage("all21-case-order", campaign.aggregate_all21, reordered)

    def test_exact_full14_parity(self):
        result = campaign.compare_local_parity(parity_report(), parity_report())
        self.assertEqual((result["case_count"], len(result["sections"])), (21, 14))

    def test_parity_value_type_array_hash_utf8_and_coverage_rejections(self):
        bad = parity_report(); bad["cases"][0]["case"] = "changed"
        self.assert_stage("all21-case-order", campaign.compare_local_parity, parity_report(), bad)
        bad = parity_report(); bad["cases"][0]["serialization"]["ordered"].reverse()
        self.assert_stage("exact-value-parity", campaign.compare_local_parity, parity_report(), bad)
        bad = parity_report(); bad["cases"][0]["full_projection_sha256"]["sections"]["case"] = B
        self.assert_stage("exact-value-parity", campaign.compare_local_parity, parity_report(), bad)
        bad = parity_report(); bad["cases"][0]["serialization"]["text"] += "changed"
        self.assert_stage("exact-value-parity", campaign.compare_local_parity, parity_report(), bad)
        bad = parity_report(); bad["cases"] = bad["cases"][:20]
        self.assert_stage("all21-case-count", campaign.compare_local_parity, parity_report(), bad)
        bad = parity_report(); bad["cases"][0].pop("prepare_compaction")
        self.assert_stage("parity-section-coverage", campaign.compare_local_parity, parity_report(), bad)
        bad = parity_report(); bad["cases"][0]["session_manager"].pop("checkpoint_state")
        self.assert_stage("parity-section-coverage", campaign.compare_local_parity, parity_report(), bad)
        bad = parity_report(); bad["cases"][0]["full_projection_sha256"]["sections"].pop("case")
        self.assert_stage("parity-projection-sections", campaign.compare_local_parity, parity_report(), bad)

    def test_malformed_nonfinite_truncated_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, body in (("truncated.json", b'{"schema":'), ("nonfinite.json", b'{"x":NaN}'), ("array.json", b"[]")):
                path = root / name; path.write_bytes(body)
                if name != "array.json":
                    self.assert_stage("malformed-json", campaign.load_json, path)
                else:
                    self.assertEqual(campaign.load_json(path), [])
            output = root / "new.json"
            campaign.write_new_json(output, {"ok": True})
            self.assert_stage("output-collision", campaign.write_new_json, output, {"ok": False})
            self.assertEqual(json.loads(output.read_text()), {"ok": True})

    def test_file_hash_reuse_stale_and_stage_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "a.txt").write_text("a")
            expected = {"a.txt": hashlib.sha256(b"a").hexdigest()}
            self.assertEqual(campaign.file_map_identities(root, expected)["a.txt"]["bytes"], 1)
            self.assert_stage("stale-or-frozen-hash", campaign.file_map_identities, root, {"a.txt": A})
            self.assert_stage("unsafe-path", campaign.file_map_identities, root, {"../escape": A})
            campaign.validate_stage_inventory(root, ["a.txt"])
            (root / "extra.txt").write_text("extra")
            self.assert_stage("stage-file-inventory", campaign.validate_stage_inventory, root, ["a.txt"])
            (root / "extra.txt").unlink()
            self.assert_stage("stage-file-inventory", campaign.validate_stage_inventory, root, ["a.txt", "missing.txt"])

    def test_exact_run_matrix_and_protocol_counts(self):
        eligible = list(campaign.R4_ONE_CALL_TIMING_CASES)
        matrix = campaign.make_run_matrix(eligible)
        self.assertEqual(len(matrix), 750)
        self.assertEqual(sum(row["fresh_processes"] for row in matrix), 750)
        self.assertEqual(sum(row["lane"] == "bench-local-warm" for row in matrix), 270)
        warm_rows = [row for row in matrix if row["lane"] == "bench-local-warm"]
        self.assertEqual(tuple(row["case"] for row in warm_rows[::30]), campaign.WARM_CASES)
        self.assertFalse({row["case"] for row in warm_rows} & {case for case, _ in campaign.UNAVAILABLE_WARM_PREFLIGHTS})
        self.assertFalse(any(row["lane"] == "bench-local-warm" and row["warmup"] != 10 for row in matrix))
        self.assertTrue(all(row["iterations"] == 30 for row in warm_rows))
        tampered = copy.deepcopy(matrix)
        tampered[-1]["case"] = "medium-ascii"
        self.assert_stage("run-matrix-identity", campaign.validate_run_matrix, tampered, eligible)
        self.assertTrue(all(row["p50_from_inner_samples"] is False for row in matrix if row["lane"] != "bench-local-warm"))
        self.assert_stage("run-matrix-identity", campaign.validate_run_matrix, matrix[:-1], eligible)
        wrong = list(eligible); wrong[0], wrong[1] = wrong[1], wrong[0]
        self.assert_stage("r4-eligibility", campaign.make_run_matrix, wrong)
        orders = campaign.make_balanced_orders(418393)
        self.assertEqual(len(orders), 30)
        self.assertEqual(sum(row["order"][0] == "baseline" for row in orders), 15)
    def test_small_ascii_exact_full14_and_full21_unavailable(self):
        baseline = small_ascii_parity_report()
        candidate = small_ascii_parity_report()
        accepted = campaign.compare_local_parity(baseline, candidate, case_names=("small-ascii",), scope="small-ascii-exact-full14")
        self.assertEqual(accepted["scope"], "small-ascii-exact-full14")
        self.assertEqual((accepted["case_count"], len(accepted["sections"])), (1, 14))
        self.assertEqual(tuple(accepted["structured_sha256"]), ("small-ascii",))
        extra = parity_report()
        self.assert_stage("all21-case-count", campaign.compare_local_parity, extra, small_ascii_parity_report(), case_names=("small-ascii",), scope="small-ascii-exact-full14")
        wrong_name = small_ascii_parity_report(); wrong_name["cases"][0]["case"] = "small-cjk"
        self.assert_stage("all21-case-order", campaign.compare_local_parity, wrong_name, small_ascii_parity_report(), case_names=("small-ascii",), scope="small-ascii-exact-full14")
        wrong_order = small_ascii_parity_report(); wrong_order["cases"][0]["serialization"]["ordered"].reverse()
        self.assert_stage("exact-value-parity", campaign.compare_local_parity, wrong_order, small_ascii_parity_report(), case_names=("small-ascii",), scope="small-ascii-exact-full14")
        wrong_text = small_ascii_parity_report(); wrong_text["cases"][0]["serialization"]["text"] += "changed"
        self.assert_stage("exact-value-parity", campaign.compare_local_parity, wrong_text, small_ascii_parity_report(), case_names=("small-ascii",), scope="small-ascii-exact-full14")
        wrong_projection = small_ascii_parity_report(); wrong_projection["cases"][0]["full_projection_sha256"]["sections"]["case"] = B
        self.assert_stage("exact-value-parity", campaign.compare_local_parity, wrong_projection, small_ascii_parity_report(), case_names=("small-ascii",), scope="small-ascii-exact-full14")
        self.assert_stage("local-parity-scope", campaign.compare_local_parity, baseline, candidate, case_names=("large-ascii",), scope="small-ascii-exact-full14")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); adapter = R12JevAdapter(); executor = r12_executor(root, adapter)
            unavailable, parity = executor.run_local_parity()
            self.assertEqual(len(adapter.receipts), 3)
            full21 = adapter.receipts[0]
            self.assertEqual(full21.exit_code, 1)
            self.assertEqual(full21.spec.stdout.read_bytes(), b"")
            self.assertFalse(full21.spec.out.exists())
            self.assertEqual(full21.spec.argv, (str(root / "r3.exe"), "verify-local", "--fixture", str(root / "fixtures"), "--out", str(root / "artifacts/gate-a/local-parity/r3-baseline/out.json")))
            self.assertEqual(unavailable["status"], "unavailable")
            self.assertEqual(unavailable["reason"], "compaction_input_limit")
            self.assertEqual(unavailable["identity"], "large-ascii-first-in-lexical-full21")
            self.assertIs(unavailable["partial_parity"], False)
            self.assertEqual(unavailable["error"], "error: [jev-prepare] unexpected production Jev prepare_context error: compaction_input_limit")
            for receipt, exe in zip(adapter.receipts[1:], (root / "r3.exe", root / "inst.exe")):
                self.assertEqual(Path(receipt.spec.argv[list(receipt.spec.argv).index("--fixture") + 1]), root / "fixtures/small-ascii")
                self.assertEqual(receipt.spec.argv[0], str(exe))
                serialized_report = json.loads(receipt.spec.out.read_text(encoding="utf-8"))
                self.assertEqual(serialized_report["cases"][0]["case"], "small-ascii")
                self.assertEqual(tuple(serialized_report["cases"][0]["full_projection_sha256"]["sections"]), campaign.PARITY_SECTIONS)
            self.assertEqual(parity["scope"], "small-ascii-exact-full14")
            small_expectation = adapter.receipts[2].spec.argv[list(adapter.receipts[2].spec.argv).index("--expectation") + 1]
            self.assertEqual(small_expectation, str(adapter.receipts[1].spec.out))

        with tempfile.TemporaryDirectory() as temp:
            adapter = R12JevAdapter(); adapter.error_override = "error: [jev-prepare] unexpected wrong error"
            executor = r12_executor(Path(temp), adapter)
            self.assert_stage("jev-unavailable-error", executor.run_local_parity)


    def test_r12_warm_outcomes_prepared_or_exact_expected_skips(self):
        for case_name in campaign.WARM_CASES:
            values = campaign.validate_warm_output(warm(case_name), case_name)
            self.assertIn("jev_prepare_context", values)
            report = warm(case_name)
            jev = next(kernel for kernel in report["cases"][0]["kernels"] if kernel["name"] == "jev_prepare_context")
            self.assertEqual(jev["outcome"], "prepared")
            self.assertEqual(jev["stats"]["n"], 30)
        for skip_class in ("no_compaction_candidates", "insufficient_compaction_reduction"):
            report = warm("small-ascii", "expected-skip")
            report["cases"][0]["kernels"][-1]["skip_class"] = skip_class
            values, state = campaign.validate_warm_observation(report, "small-ascii")
            self.assertEqual(state, "expected-skip")
            self.assertNotIn("jev_prepare_context", values)
            self.assertIsNone(report["cases"][0]["kernels"][-1]["stats"])
        bad = warm("small-ascii", "expected-skip")
        bad["cases"][0]["kernels"][-1]["skip_class"] = "synthetic_other_skip"
        self.assert_stage("warm-jev-skip", campaign.validate_warm_output, bad, "small-ascii")

    def test_twelve_unavailable_warm_preflight_boundaries(self):
        expected = (
            ("medium-ascii", 880), ("medium-cjk", 728), ("medium-emoji", 896),
            ("medium-mixed-crlf", 534), ("medium-many-small", 1848),
            ("large-ascii", 880), ("large-cjk", 731), ("large-emoji", 938),
            ("large-mixed-crlf", 554), ("large-many-small", 1945),
            ("near-threshold-ascii", 1156), ("xlarge-ascii", 909),
        )
        self.assertEqual(campaign.UNAVAILABLE_WARM_PREFLIGHTS, expected)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); adapter = R12JevAdapter(); executor = r12_executor(root, adapter)
            records = executor.run_unavailable_warm_preflight()
            self.assertEqual(tuple((row["case"], row["history"]) for row in records), expected)
            self.assertEqual(len(adapter.receipts), 12)
            for row, receipt in zip(records, adapter.receipts):
                case_name, history = row["case"], row["history"]
                expected_error = f"error: {case_name}: non-timed preflight failed: [jev-prepare] unexpected production Jev prepare_context error: compaction_input_limit"
                self.assertEqual(row["error"], expected_error)
                self.assertEqual(receipt.exit_code, 1)
                self.assertEqual(row["fresh_processes"], 1)
                self.assertIsNone(row["statistics"])
                self.assertEqual(row["timed_observations"], [])
                self.assertIs(row["timing_executed"], False)
                self.assertIs(row["non_timed"], True)
                self.assertEqual(receipt.spec.stdout.read_bytes(), b"")
                self.assertFalse(receipt.spec.out.exists())
                self.assertEqual(receipt.spec.argv, campaign.bench_command(root / "inst.exe", root / "fixtures" / case_name, root / "artifacts/gate-a/warm-preflight-unavailable" / case_name / "out.json"))
                self.assertEqual(row["exe_sha256"], hashlib.sha256((root / "inst.exe").read_bytes()).hexdigest())
                self.assertEqual(row["argv_sha256"], campaign.structured_sha256(receipt.spec.argv))
            self.assertEqual(campaign.validate_unavailable_warm_preflights(records, root / "fixtures", root / "artifacts"), records)
            missing = copy.deepcopy(records); missing.pop()
            self.assert_stage("warm-preflight-count", campaign.validate_unavailable_warm_preflights, missing, root / "fixtures", root / "artifacts")
            wrong_error = copy.deepcopy(records); wrong_error[0]["error"] = "error: changed"
            self.assert_stage("warm-preflight-error", campaign.validate_unavailable_warm_preflights, wrong_error, root / "fixtures", root / "artifacts")
            wrong_output = copy.deepcopy(records); wrong_output[0]["no_output"] = False
            self.assert_stage("warm-preflight-output", campaign.validate_unavailable_warm_preflights, wrong_output, root / "fixtures", root / "artifacts")
            fabricated = copy.deepcopy(records); fabricated[0]["statistics"] = {"mean_ms": 0.0}
            self.assert_stage("warm-preflight-no-timing", campaign.validate_unavailable_warm_preflights, fabricated, root / "fixtures", root / "artifacts")
            duplicated_process = copy.deepcopy(records); duplicated_process[0]["fresh_processes"] = 2
            self.assert_stage("warm-preflight-fresh-count", campaign.validate_unavailable_warm_preflights, duplicated_process, root / "fixtures", root / "artifacts")
            eligible = list(campaign.R4_ONE_CALL_TIMING_CASES)
            warm_cases = {row["case"] for row in campaign.make_run_matrix(eligible) if row["lane"] == "bench-local-warm"}
            self.assertFalse(warm_cases & {case for case, _ in expected})

        for option, stage in (("stdout", "jev-unavailable-output"), ("output", "jev-unavailable-output"), ("error", "jev-unavailable-error"), ("exit", "jev-unavailable-exit")):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as temp:
                adapter = R12JevAdapter()
                if option == "stdout": adapter.negative_stdout = b"unexpected stdout"
                if option == "output": adapter.write_negative_output = True
                if option == "error": adapter.error_override = "error: changed"
                if option == "exit": adapter.negative_exit = 0
                executor = r12_executor(Path(temp), adapter)
                self.assert_stage(stage, executor.run_unavailable_warm_preflight)



    def test_valid_one_shot_and_unavailable_allocation_aggregation(self):
        result = campaign.summarize_fresh_processes(fresh_records(), "wall", "small-ascii")
        self.assertEqual(result["statistics"]["process_total_ms"]["n"], 30)
        unavailable = fresh_records("allocation")
        for row in unavailable:
            row["values"] = {"allocation_count": None, "allocation_bytes_requested": None}
            row["unavailable_reason"] = "synthetic unavailable"
        result = campaign.summarize_fresh_processes(unavailable, "allocation", "small-ascii")
        self.assertEqual(result["statistics"]["status"], "unavailable")

    def test_one_shot_count_pid_identity_and_nonfinite_rejections(self):
        self.assert_stage("baseline-observation-count", campaign.summarize_fresh_processes, fresh_records(count=29), "wall", "small-ascii")
        rows = fresh_records(); rows[1]["pid"] = rows[0]["pid"]
        self.assert_stage("pid-collision", campaign.summarize_fresh_processes, rows, "wall", "small-ascii")
        rows = fresh_records(); rows[1]["identity_sha256"] = B
        self.assert_stage("baseline-identity-drift", campaign.summarize_fresh_processes, rows, "wall", "small-ascii")
        rows = fresh_records(); rows[0]["values"]["process_total_ms"] = float("nan")
        self.assert_stage("timing-numeric", campaign.summarize_fresh_processes, rows, "wall", "small-ascii")
        rows = fresh_records(); rows[0]["observation"] = 1
        self.assert_stage("baseline-run-order", campaign.summarize_fresh_processes, rows, "wall", "small-ascii")

    def test_valid_warm_prepared_and_expected_skip(self):
        prepared = campaign.validate_warm_output(warm(), "small-ascii")
        self.assertEqual(len(prepared), 12)
        skipped = campaign.validate_warm_output(warm(jev_outcome="expected-skip"), "small-ascii")
        self.assertEqual(len(skipped), 11)
        self.assertNotIn("jev_prepare_context", skipped)
        rows = []
        for index in range(30):
            state = "expected-skip" if index % 3 == 0 else "prepared"
            outer = {key: value for key, value in prepared.items() if state == "prepared" or key != "jev_prepare_context"}
            rows.append({"status": "admitted", "lane": "bench-local-warm", "case": "small-ascii", "observation": index, "fresh_processes": 1, "pid": 4000 + index, "identity_sha256": A, "values": outer, "jev_state": state, "warmup": 10, "iterations": 30})
        result = campaign.summarize_warm_processes(rows, "small-ascii")
        self.assertEqual(result["statistics"]["estimate_tokens_all_messages"]["n"], 30)
        self.assertEqual(result["statistics"]["jev_prepare_context"]["n"], 20)
        for name in campaign.WARM_KERNELS:
            self.assertEqual(result["statistics"][name]["n"], 30)

    def test_warm_wrong_config_kernel_stats_and_preflight_rejections(self):
        bad = warm(); bad["config"]["iters"] = 30.0
        self.assert_stage("warm-output-config", campaign.validate_warm_output, bad, "small-ascii")
        bad = warm(); bad["cases"][0]["kernels"] = bad["cases"][0]["kernels"][:-1]
        self.assert_stage("warm-kernel-coverage", campaign.validate_warm_output, bad, "small-ascii")
        bad = warm(); bad["cases"][0]["kernels"][0]["stats"]["n"] = 29
        self.assert_stage("warm-inner-count", campaign.validate_warm_output, bad, "small-ascii")
        bad = warm(); bad["cases"][0]["kernels"][0]["stats"]["p50_ms"] = float("inf")
        self.assert_stage("warm-stat-nonfinite", campaign.validate_warm_output, bad, "small-ascii")
        bad = warm(); bad["non_timed_preflight"][0]["full_output_sha256"] = "bad"
        self.assert_stage("warm-preflight-hash", campaign.validate_warm_output, bad, "small-ascii")
        bad = warm(jev_outcome="expected-skip"); bad["cases"][0]["kernels"][-1]["stats"] = stats()
        self.assert_stage("warm-jev-skip", campaign.validate_warm_output, bad, "small-ascii")

    def test_independent_paired_stats_and_wrong_order(self):
        rows, orders = paired_inputs()
        result = campaign.paired_statistics(rows, orders, 418393)
        self.assertEqual((result["pair_count"], result["ab_count"], result["ba_count"]), (30, 15, 15))
        self.assertEqual(result["metrics"]["metric"]["bootstrap"]["samples"], 10000)
        reversed_orders = list(reversed(orders))
        self.assert_stage("pair-order", campaign.paired_statistics, rows, reversed_orders, 418393)

    def test_r4_one_shot_lane_build_feature_phase_and_allocation_controls(self):
        wall = r4mock.external_wall()
        self.assertIn("process_total_ms", campaign.extract_one_shot(wall, "wall"))
        provider = r4mock.lane_report("minimal-provider")
        values = campaign.extract_one_shot(provider, "minimal-provider")
        self.assertIn("phase/response_parse", values)
        provider["local_replay"]["phases"][0]["cpu_scope"] = "includes mock wait"
        self.assert_stage("r4-report-validation", campaign.extract_one_shot, provider, "minimal-provider")
        allocation = r4mock.lane_report("allocation")
        allocation["build"]["feature"] = "compaction-r4-uninstrumented-wall"
        self.assert_stage("r4-report-validation", campaign.extract_one_shot, allocation, "allocation")
        allocation = r4mock.lane_report("allocation")
        allocation["allocation"] = {"status": "unavailable", "source": None, "count": None, "bytes_requested": None, "reason": "synthetic unavailable"}
        values = campaign.extract_one_shot(allocation, "allocation")
        self.assertIsNone(values["allocation_count"])
        wrong = r4mock.lane_report("allocation")
        self.assert_stage("r4-report-validation", campaign.extract_one_shot, wrong, "minimal-provider")

    def test_telemetry_null_reason_missing_and_state_rejections(self):
        self.assertEqual(campaign.validate_telemetry(telemetry())["required_fields"], list(campaign.TELEMETRY_FIELDS))
        bad = telemetry(); bad["before"]["available_memory_bytes"] = {"value": 0, "unavailable_reason": None}
        self.assertEqual(campaign.validate_telemetry(bad)["before"]["available_memory_bytes"]["value"], 0)
        bad = telemetry(); bad["before"]["available_memory_bytes"] = {"value": None, "unavailable_reason": None}
        self.assert_stage("telemetry-unavailable", campaign.validate_telemetry, bad)
        bad = telemetry(); bad["required_fields"] = bad["required_fields"][:-1]
        self.assert_stage("telemetry-required-fields", campaign.validate_telemetry, bad)
        bad = telemetry(); del bad["before"]["processor_queue"]
        self.assert_stage("telemetry-fields", campaign.validate_telemetry, bad)
        bad = telemetry(); bad["observation"]["page_cache_state"] = "warm"
        self.assert_stage("telemetry-state-label", campaign.validate_telemetry, bad)
        bad = telemetry(); bad["after"]["cpu_total_percent"] = {"value": float("nan"), "unavailable_reason": None}
        self.assert_stage("telemetry-nonfinite", campaign.validate_telemetry, bad)

    def test_affinity_child_only_mask_and_nested_coverage(self):
        self.assertEqual(campaign.validate_affinity(affinity())["children"][0]["observed_mask"], "0x555")
        self.assertEqual(len(campaign.validate_affinity(affinity(wall=True), nested_wall=True)["children"]), 2)
        bad = affinity(); bad["children"][0]["observed_mask"] = "0xaaa"
        self.assert_stage("affinity-mask", campaign.validate_affinity, bad)
        bad = affinity(wall=True); bad["children"] = bad["children"][:1]
        self.assert_stage("affinity-coverage", campaign.validate_affinity, bad, nested_wall=True)
        self.assert_stage("affinity-mask", campaign.validate_affinity, {"requested_mask": "0x555", "children": []})

    def test_authorization_valid_missing_stale_and_wrong_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "target"
            eligible = list(campaign.R4_ONE_CALL_TIMING_CASES)
            matrix = campaign.make_run_matrix(eligible)
            matrix_hash = campaign.structured_sha256(matrix)
            builds = {}
            executables = {}
            for name, feature in (("instrumented", None), ("wall", "compaction-r4-uninstrumented-wall")):
                exe = target / name / "compaction_replay.exe"; exe.parent.mkdir(parents=True); exe.write_text(name)
                log = root / f"{name}.log"; log.write_text("fresh synthetic build")
                executables[name] = exe
                builds[name] = {"exe": str(exe), "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "build_log": str(log), "build_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(), "source_commit": campaign.R4_COMMIT, "target_root": str(target / name), "feature": feature, "profile": "release", "fresh_build": True}
            r3 = root / "r3.exe"; r3.write_text("synthetic-r3")
            executables["r3-parity"] = r3
            protected = protected_hashes(executables)
            protected["exe"][str(r3)] = campaign.R3_REPLAY_EXE_SHA256
            summary = {"schema": campaign.GATE_A_SUMMARY_SCHEMA, "status": "ok", "timing_executed": False, "run_matrix_sha256": matrix_hash, "protected_hashes": protected}
            summary_path = root / "summary.json"; summary_path.write_bytes(encoded(summary))
            review = {"schema": campaign.GATE_A_REVIEW_SCHEMA, "reviewer": "synthetic-independent", "verdict": "APPROVED_GATE_A_CORRECTNESS_ONLY", "reviewed_at": "synthetic", "root_message_id": "synthetic-review", "gate_a_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(), "source": {"base": campaign.R4_BASE, "commit": campaign.R4_COMMIT, "patch_sha256": campaign.R4_PATCH_SHA256, "run_matrix_sha256": matrix_hash}, "protected_hashes": protected}
            review_path = root / "review.json"; review_path.write_bytes(encoded(review))
            auth = {"schema": campaign.GATE_B_AUTH_SCHEMA, "authorized_by": "root", "root_message_id": "synthetic-root", "issued_at": "synthetic", "gate_a_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(), "gate_a_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(), "run_matrix_sha256": matrix_hash, "fresh_timing_builds": builds, "protected_hashes": protected, "baseline_only": True, "no_network": True, "no_comparator": True, "r4_one_shot_protocol": {"fresh_processes_per_case_lane": 30, "invented_warmup_or_inner_samples": False, "aggregate_single_observations_only": True}}
            kwargs = {"executor_binaries": executables, "protected_hashes": protected}
            with mock.patch.object(campaign, "TARGET_ROOT", target):
                self.assertEqual(campaign.authorize_gate_b(auth, review_path, summary_path, matrix, **kwargs)["baseline_only"], True)
                self.assert_stage("gate-b-authorization", campaign.authorize_gate_b, None, review_path, summary_path, matrix, **kwargs)
                stale = copy.deepcopy(auth); stale["gate_a_summary_sha256"] = A
                self.assert_stage("gate-b-stale-summary", campaign.authorize_gate_b, stale, review_path, summary_path, matrix, **kwargs)
                stale = copy.deepcopy(auth); stale["run_matrix_sha256"] = A
                self.assert_stage("gate-b-run-matrix", campaign.authorize_gate_b, stale, review_path, summary_path, matrix, **kwargs)
                stale = copy.deepcopy(auth); stale["baseline_only"] = False
                self.assert_stage("gate-b-limits", campaign.authorize_gate_b, stale, review_path, summary_path, matrix, **kwargs)
                stale = copy.deepcopy(auth); stale["fresh_timing_builds"]["wall"]["fresh_build"] = False
                self.assert_stage("timing-build-provenance", campaign.authorize_gate_b, stale, review_path, summary_path, matrix, **kwargs)
                stale = copy.deepcopy(auth); stale["fresh_timing_builds"]["wall"]["feature"] = None
                self.assert_stage("timing-build-feature", campaign.authorize_gate_b, stale, review_path, summary_path, matrix, **kwargs)
                stale = copy.deepcopy(auth); stale["fresh_timing_builds"]["wall"]["exe_sha256"] = A
                self.assert_stage("timing-build-hash", campaign.authorize_gate_b, stale, review_path, summary_path, matrix, **kwargs)
                stale = copy.deepcopy(auth); stale["r4_one_shot_protocol"]["invented_warmup_or_inner_samples"] = True
                self.assert_stage("gate-b-one-shot", campaign.authorize_gate_b, stale, review_path, summary_path, matrix, **kwargs)
                wrong_executors = dict(executables); wrong_executors["instrumented"] = root / "unauthorized.exe"
                self.assert_stage("gate-b-executor-not-authorized", campaign.authorize_gate_b, auth, review_path, summary_path, matrix, executor_binaries=wrong_executors, protected_hashes=protected)
                changed_inputs = copy.deepcopy(protected); changed_inputs["script"]["synthetic-script.input"] = B
                self.assert_stage("gate-b-protected-hashes", campaign.authorize_gate_b, auth, review_path, summary_path, matrix, executor_binaries=executables, protected_hashes=changed_inputs)
                stale_auth = copy.deepcopy(auth); stale_auth["protected_hashes"]["toolchain"]["synthetic-toolchain.input"] = B
                self.assert_stage("gate-b-protected-hashes", campaign.authorize_gate_b, stale_auth, review_path, summary_path, matrix, **kwargs)

    def test_gate_a_wrong_or_stale_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); summary_path = root / "summary.json"
            matrix = campaign.make_run_matrix(campaign.R4_ONE_CALL_TIMING_CASES)
            protected = protected_hashes()
            summary = {"schema": campaign.GATE_A_SUMMARY_SCHEMA, "status": "ok", "timing_executed": False, "run_matrix_sha256": campaign.structured_sha256(matrix), "protected_hashes": protected}
            summary_path.write_bytes(encoded(summary))
            review = {"schema": campaign.GATE_A_REVIEW_SCHEMA, "reviewer": "synthetic", "verdict": "APPROVED_GATE_A_CORRECTNESS_ONLY", "reviewed_at": "synthetic", "root_message_id": "synthetic", "gate_a_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(), "source": {"base": campaign.R4_BASE, "commit": campaign.R4_COMMIT, "patch_sha256": campaign.R4_PATCH_SHA256, "run_matrix_sha256": campaign.structured_sha256(matrix)}, "protected_hashes": protected}
            self.assertEqual(campaign.validate_gate_a_review(review, summary, summary_path.read_bytes())["verdict"], "APPROVED_GATE_A_CORRECTNESS_ONLY")
            bad = copy.deepcopy(review); bad["verdict"] = "SELF_APPROVED"
            self.assert_stage("gate-a-review", campaign.validate_gate_a_review, bad, summary, summary_path.read_bytes())
            bad = copy.deepcopy(review); bad["gate_a_summary_sha256"] = A
            self.assert_stage("gate-a-stale-review", campaign.validate_gate_a_review, bad, summary, summary_path.read_bytes())
            bad = copy.deepcopy(review); bad["source"]["commit"] = "0" * 40
            self.assert_stage("gate-a-source", campaign.validate_gate_a_review, bad, summary, summary_path.read_bytes())
            bad = copy.deepcopy(review); bad["protected_hashes"]["fixture"]["synthetic-fixture.input"] = B
            self.assert_stage("gate-a-protected-hashes", campaign.validate_gate_a_review, bad, summary, summary_path.read_bytes())


    def test_reviewed_source_base_patch_and_four_file_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".git").write_text("synthetic git metadata")
            for relative in campaign.R5_SOURCE_PATHS:
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(relative)
            patch = root / "approved.patch"; patch.write_bytes(b"synthetic patch\n")
            patch_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
            receipt = {
                "worktree": str(root), "branch": "fix/compaction-r5-baseline",
                "head": campaign.R4_COMMIT, "base": campaign.R4_BASE, "parent": campaign.R4_BASE,
                "r4_patch_path": str(patch), "r4_patch_sha256": patch_hash,
                "r4_changed_paths": list(campaign.R4_CHANGED_PATHS), "r5_new_paths": list(campaign.R5_SOURCE_PATHS),
                "unexpected_paths": [], "production_diff": [], "approval_message_id": "synthetic-approved-r4",
                "source_file_sha256": {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in campaign.R5_SOURCE_PATHS},
            }
            git = FakeGit(patch)
            with mock.patch.object(campaign, "R4_PATCH_SHA256", patch_hash):
                result = campaign.admit_source_identity(receipt, git_runner=git)
                self.assertEqual((result["base_plus_patch"], result["clean_head_claim"], result["live_git_verified"]), (True, False, True))
                self.assertEqual(len(git.calls), 8)
                wrong = copy.deepcopy(receipt); wrong["r4_changed_paths"] = wrong["r4_changed_paths"][:-1]
                self.assert_stage("r4-patch-scope", campaign.admit_source_identity, wrong, git_runner=git)
                wrong = copy.deepcopy(receipt); wrong["unexpected_paths"] = ["unexpected.py"]
                self.assert_stage("source-unexpected-diff", campaign.admit_source_identity, wrong, git_runner=git)
                wrong = copy.deepcopy(receipt); wrong["head"] = "0" * 40
                self.assert_stage("source-commit-identity", campaign.admit_source_identity, wrong, git_runner=git)
                wrong = copy.deepcopy(receipt); wrong["source_file_sha256"][campaign.R5_SOURCE_PATHS[0]] = A
                self.assert_stage("r5-source-hash", campaign.admit_source_identity, wrong, git_runner=git)
                live = copy.deepcopy(git); live.outputs = dict(git.outputs)
                live.outputs[("rev-parse", "HEAD")] = "0" * 40 + "\n"
                self.assert_stage("git-live-identity", campaign.admit_source_identity, receipt, git_runner=live)
                live = copy.deepcopy(git); live.outputs = dict(git.outputs)
                key = ("status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching")
                live.outputs[key] = live.outputs[key] + "!! production.py\n"
                self.assert_stage("git-live-scope", campaign.admit_source_identity, receipt, git_runner=live)
                live = copy.deepcopy(git); live.outputs = dict(git.outputs)
                live.outputs[("diff", "--name-only", "HEAD", "--")] = "pi/extensions/coding-agent/src/core/compaction/mod.rs\n"
                self.assert_stage("git-live-production-diff", campaign.admit_source_identity, receipt, git_runner=live)
                live = copy.deepcopy(git); live.outputs = dict(git.outputs)
                live.outputs[("diff", "--name-only", campaign.R4_BASE, campaign.R4_COMMIT, "--")] = "only-one-path\n"
                self.assert_stage("git-live-r4-scope", campaign.admit_source_identity, receipt, git_runner=live)
                live = copy.deepcopy(git); live.outputs = dict(git.outputs)
                live.outputs[("diff", "--binary", campaign.R4_BASE, campaign.R4_COMMIT)] = "wrong patch\n"
                self.assert_stage("git-live-r4-patch", campaign.admit_source_identity, receipt, git_runner=live)

    def test_end_to_end_all21_writer_admission_and_aggregate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = RecordingAdapter()
            executor = campaign.CampaignExecutor(adapter, root / "artifacts", root / "fixtures", root / "repo", root / "inst.exe", root / "wall.exe", root / "r3.exe")
            aggregate_path = root / "artifacts/gate-a/replay-all21/summary.json"
            result = executor.run_all21_replays(aggregate_path)
            self.assertEqual((result["case_count"], len(adapter.receipts)), (21, 21))
            self.assertEqual(json.loads(aggregate_path.read_text())["case_count"], 21)
            for case_name, receipt in zip(campaign.CASE_ORDER, adapter.receipts):
                self.assertEqual(receipt.spec.out.name, "writer.json")
                self.assertEqual(receipt.spec.out.parent.name, case_name)
                self.assertTrue(receipt.spec.out.is_file())

    def test_all21_writer_failure_has_no_dependent_aggregate(self):
        for mode in ("wrong-N", "missing-output"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                adapter = RecordingAdapter()
                if mode == "wrong-N":
                    adapter.summary_call_delta = 1
                else:
                    adapter.missing_output_run = "small-ascii"
                executor = campaign.CampaignExecutor(adapter, root / "artifacts", root / "fixtures", root / "repo", root / "inst.exe", root / "wall.exe", root / "r3.exe")
                aggregate_path = root / "artifacts/gate-a/replay-all21/summary.json"
                self.assert_stage("replay-summary-N" if mode == "wrong-N" else "replay-output-path", executor.run_all21_replays, aggregate_path)
                self.assertFalse(aggregate_path.exists())
                self.assertEqual(len(adapter.receipts), 21 if mode == "wrong-N" else 1)

    def test_jev_nearest_rank_p95_for_two_ten_and_thirty_prepared(self):
        for prepared_count in (2, 10, 30):
            rows = []
            for index in range(30):
                state = "prepared" if index < prepared_count else "expected-skip"
                values = {name: float(index + 1) for name in campaign.WARM_KERNELS}
                if state == "prepared":
                    values["jev_prepare_context"] = 0.1 * (index + 1)
                rows.append({"status": "admitted", "lane": "bench-local-warm", "case": "small-ascii", "observation": index, "fresh_processes": 1, "pid": 11000 + index, "identity_sha256": A, "values": values, "jev_state": state, "warmup": 10, "iterations": 30})
            stats = campaign.summarize_warm_processes(rows, "small-ascii")["statistics"]
            for name in campaign.WARM_KERNELS:
                self.assertEqual(stats[name]["n"], 30)
                self.assertEqual(stats[name]["p95"], 29.0)
            self.assertEqual(stats["jev_prepare_context"]["n"], prepared_count)
            self.assertEqual(stats["jev_prepare_context"]["p95"], {2: 0.2, 10: 1.0, 30: 0.1 * 29}[prepared_count])
            self.assertEqual(stats["jev_prepare_context"]["p50"], statistics.median([0.1 * (index + 1) for index in range(prepared_count)]))

    def test_wrong_lane_reaches_retained_r4_validator_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = SCRIPT_DIR.parent
            artifacts = root / "artifacts"
            report_path = artifacts / "synthetic/wrong-lane-source.json"
            campaign.write_new_json(report_path, r4mock.lane_report("minimal-provider"))
            matrix_path = root / "run-matrix.json"; matrix_path.write_text("synthetic\n")
            script = repo / "scripts/compaction_r4_lanes.py"
            python_exe = Path(sys.executable).resolve()
            classes = {
                "source": {"scripts/compaction_baseline_campaign.py": hashlib.sha256((repo / "scripts/compaction_baseline_campaign.py").read_bytes()).hexdigest()},
                "fixture": {str(report_path): hashlib.sha256(report_path.read_bytes()).hexdigest()},
                "toolchain": {"scripts/compaction_r4_lanes.py": hashlib.sha256(script.read_bytes()).hexdigest()},
                "script": {"scripts/compaction_r4_lanes_mock.py": hashlib.sha256((repo / "scripts/compaction_r4_lanes_mock.py").read_bytes()).hexdigest()},
                "exe": {str(python_exe): hashlib.sha256(python_exe.read_bytes()).hexdigest()},
                "run_matrix": {str(matrix_path): hashlib.sha256(matrix_path.read_bytes()).hexdigest()},
            }
            adapter = campaign.SubprocessAdapter(campaign.IntegrityGuard(repo, classes), FakeTelemetry(), mock.Mock())
            executor = campaign.CampaignExecutor(adapter, artifacts, root / "fixtures", repo, root / "inst.exe", root / "wall.exe", root / "r3.exe")
            out = artifacts / "synthetic/reader-routing/reader-wrong-lane/out.json"
            argv = (sys.executable, script, "validate", "--report", report_path, "--lane", "wall", "--out", out)
            result = executor._expected_failure("synthetic/reader-routing", "reader-wrong-lane", argv, "lane-identity")
            self.assertEqual((result["exit_code"], result["intended_stage"], result["outcome"]), (2, "lane-identity", "rejected"))
            self.assertFalse(out.exists())
            receipt = json.loads((out.parent / "work/receipt.json").read_text())
            self.assertEqual(receipt["exit_code"], 2)
            self.assertIsNone(receipt["raw_output"])
            self.assertIn("lane-identity", (out.parent / "stderr.bin").read_text())
            self.assert_stage("command-catalog", executor.validate_external_argv, [sys.executable, str(script), "validate", "--report", str(report_path), "--lane", "network", "--out", str(out)])
            self.assert_stage("command-catalog", executor.validate_external_argv, [sys.executable, str(script), "validate", "--report", str(matrix_path), "--lane", "wall", "--out", str(out)])

    def test_protected_collision_reaches_python_r4_reader_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); repo = SCRIPT_DIR.parent; artifacts = root / "artifacts"
            input_path = artifacts / "synthetic/collision-input.json"
            campaign.write_new_json(input_path, r4mock.lane_report("minimal-provider"))
            matrix_path = root / "run-matrix.json"; matrix_path.write_text("synthetic\n")
            python_exe = Path(sys.executable).resolve()
            script = repo / "scripts/compaction_r4_lanes.py"
            classes = {
                "source": {"scripts/compaction_baseline_campaign.py": hashlib.sha256((repo / "scripts/compaction_baseline_campaign.py").read_bytes()).hexdigest()},
                "fixture": {str(input_path): hashlib.sha256(input_path.read_bytes()).hexdigest()},
                "toolchain": {"scripts/compaction_r4_lanes.py": hashlib.sha256(script.read_bytes()).hexdigest()},
                "script": {"scripts/compaction_r4_lanes_mock.py": hashlib.sha256((repo / "scripts/compaction_r4_lanes_mock.py").read_bytes()).hexdigest()},
                "exe": {str(python_exe): hashlib.sha256(python_exe.read_bytes()).hexdigest()},
                "run_matrix": {str(matrix_path): hashlib.sha256(matrix_path.read_bytes()).hexdigest()},
            }
            guard = campaign.IntegrityGuard(repo, classes)
            adapter = campaign.SubprocessAdapter(guard, FakeTelemetry(), mock.Mock())
            executor = campaign.CampaignExecutor(adapter, artifacts, root / "fixtures", repo, root / "inst.exe", root / "wall.exe", root / "r3.exe")
            out = artifacts / "synthetic/collision/collision/out.json"
            campaign.write_new_json(out, r4mock.lane_report("minimal-provider"))
            original = hashlib.sha256(out.read_bytes()).hexdigest()
            registration = guard.register_runtime_protected(artifacts, {out: original})
            argv = [sys.executable, script, "validate", "--report", input_path, "--lane", "minimal-provider", "--out", out]
            receipt, produced = executor._run("synthetic/collision", "collision", argv, expected_exit=-1, existing_output_sha256=original)
            self.assertEqual(receipt.exit_code, 1)
            self.assertIsNone(produced)
            self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(), original)
            self.assertIn("FileExistsError", (out.parent / "stderr.bin").read_text())
            self.assertTrue(guard.is_runtime_protected(out, original))
            self.assertFalse((out.parent / "stats.json").exists())
            self.assertEqual(registration["inputs"][str(out.resolve())]["sha256"], original)
            # An unregistered existing output is still stopped by stock layout before any Python child.
            unregistered = artifacts / "synthetic/layout/collision/out.json"
            campaign.write_new_json(unregistered, r4mock.lane_report("minimal-provider"))
            bad_argv = [sys.executable, script, "validate", "--report", input_path, "--lane", "minimal-provider", "--out", unregistered]
            self.assert_stage("output-collision", executor._run, "synthetic/layout", "collision", bad_argv, expected_exit=-1)
            self.assertTrue((out.parent / "work/receipt.json").is_file())

    def test_direct_binary_identity_environment_and_profile_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, sections = direct_focused_plan(root)
            with mock.patch.object(campaign, "R4_PROVENANCE_LOGS", {kind: root / f"{kind}.log" for kind in plan}), mock.patch.object(campaign, "R4_TEST_BINARY_ROOT", root / "optimus-compaction-r4-target/instrumented/release/deps"):
                adapter = RecordingAdapter()
                executor = campaign.CampaignExecutor(adapter, root / "artifacts", root / "fixtures", root / "repo", root / "inst.exe", root / "wall.exe", root / "r3.exe")
                executor.register_focused_plan(plan)
                self.assertEqual(len(executor.focused_executables), 15)
                protected_exe = {row["argv"][0]: row["sha256"] for row in executor.focused_executables.values()}
                self.assertEqual(len(protected_exe), 15)
                self.assertTrue(all(len(digest) == 64 for digest in protected_exe.values()))
                first = next(row for row in executor.focused_executables.values() if row["kind"] == "focused99" and row["index"] == 0)
                profile = first["profile_root"]
                receipt, _ = executor._run("gate-a/focused99", "command-00", first["argv"], gate_a=True, profile_root=profile, create_profile=True, profile_id="focused99")
                self.assertEqual(receipt.spec.allowlisted_env["PRIME_AGENT_CODING_AGENT_DIR"], str(profile.resolve()))
                char = next(row for row in executor.focused_executables.values() if row["kind"] == "charcount15")
                self.assertEqual(char["argv"][1:2], (campaign.CHARCOUNT_FILTER,))
                adapter.receipts = []
                self.assert_stage("profile-isolation", executor._run, "gate-a/focused99", "missing-profile", first["argv"], gate_a=True)
                self.assertEqual(len(adapter.receipts), 0)
                self.assert_stage("focused-command-catalog", executor._run, "gate-a/focused99", "wrong-arg", first["argv"] + ("--nocapture",), gate_a=True, profile_root=profile, profile_id="focused99")
                self.assertEqual(len(adapter.receipts), 0)
                cargo = [str(root / "cargo.exe"), "test", "--manifest-path", str(root / "repo/Cargo.toml"), "--release"]
                self.assert_stage("command-catalog", executor._run, "gate-a/focused99", "forbidden-cargo", cargo, gate_a=True)
                self.assertEqual(len(adapter.receipts), 0)
                changed = copy.deepcopy(plan); changed["focused99"]["binaries"][0]["sha256"] = A
                self.assert_stage("focused-binary-hash", executor.register_focused_plan, changed)
                substituted = copy.deepcopy(plan); substituted["focused99"]["expected_names"][0] = "manifest-substitution"
                self.assert_stage("focused-provenance", executor.register_focused_plan, substituted)
            marker_file = root / "immutable-input.json"; marker_file.write_text("immutable\n")
            classes = {kind: {str(marker_file): hashlib.sha256(marker_file.read_bytes()).hexdigest()} for kind in ("source", "fixture", "toolchain", "script", "exe", "run_matrix")}
            stock = campaign.SubprocessAdapter(campaign.IntegrityGuard(root, classes), FakeTelemetry(), mock.Mock())
            spec = campaign.ProcessSpec("synthetic", "profile", ("synthetic",), root / "artifacts/gate-a/focused99/work", root / "artifacts/gate-a/focused99/out.json", root / "artifacts/gate-a/focused99/stdout", root / "artifacts/gate-a/focused99/stderr", profile_root=root / "artifacts/gate-a/profiles/focused99", create_profile=True, profile_id="focused99")
            context = stock._prepare_profile(spec)
            (spec.profile_root / "generated.txt").write_text("synthetic profile mutation inside the isolated root\n")
            evidence = stock._finish_profile(context)
            self.assertIn("generated.txt", evidence["after"])
            self.assertNotIn("generated.txt", evidence["before"])
            context["marker"].write_text("tampered protected marker\n")
            self.assert_stage("profile-hash-drift", stock._finish_profile, context)
            self.assertFalse((spec.work.parent / "stats.json").exists())
            self.assert_stage("command-catalog", campaign.CommandTelemetry._probe, ("cmd.exe", "/c", "arbitrary"))

    def test_finite_zero_allocation_and_stddev_semantics(self):
        zero_allocation = r4mock.lane_report("allocation")
        zero_allocation["allocation"]["count"] = 0
        zero_allocation["allocation"]["bytes_requested"] = 0
        values = campaign.extract_one_shot(zero_allocation, "allocation")
        self.assertEqual((values["allocation_count"], values["allocation_bytes_requested"]), (0, 0))
        zero_warm = warm()
        zero_warm["cases"][0]["kernels"][0]["stats"]["stddev_ms"] = 0
        self.assertEqual(campaign.validate_warm_output(zero_warm, "small-ascii")["estimate_tokens_all_messages"], 1.25)
        bad_warm = warm(); bad_warm["cases"][0]["kernels"][0]["stats"]["stddev_ms"] = -0.1
        self.assert_stage("warm-stat-nonnegative", campaign.validate_warm_output, bad_warm, "small-ascii")
        bad_warm = warm(); bad_warm["cases"][0]["kernels"][0]["stats"]["p50_ms"] = 0
        self.assert_stage("warm-stat-nonfinite", campaign.validate_warm_output, bad_warm, "small-ascii")
        rows = fresh_records("allocation")
        for row in rows:
            row["values"] = {"allocation_count": 0, "bytes_requested": 0}
        result = campaign.summarize_fresh_processes(rows, "allocation", "small-ascii")
        self.assertEqual(result["statistics"]["allocation_count"]["max"], 0.0)

    def test_jev_all_skip_and_supported_matched_pairs(self):
        prepared = campaign.validate_warm_output(warm(), "small-ascii")
        rows = []
        for index in range(30):
            rows.append({"status": "admitted", "lane": "bench-local-warm", "case": "small-ascii", "observation": index, "fresh_processes": 1, "pid": 7000 + index, "identity_sha256": A, "values": {name: value for name, value in prepared.items() if name != "jev_prepare_context"}, "jev_state": "expected-skip", "warmup": 10, "iterations": 30})
        result = campaign.summarize_warm_processes(rows, "small-ascii")
        self.assertEqual(result["statistics"]["jev_prepare_context"]["status"], "all-expected-skip")
        bad = copy.deepcopy(rows); bad[0]["values"]["jev_prepare_context"] = 1.0
        self.assert_stage("warm-outer-keys", campaign.summarize_warm_processes, bad, "small-ascii")
        bad = copy.deepcopy(rows); bad[0]["values"].pop("estimate_tokens_all_messages")
        self.assert_stage("warm-outer-keys", campaign.summarize_warm_processes, bad, "small-ascii")
        paired, orders = paired_inputs()
        for index, order in enumerate(orders):
            if index in (0, 15):
                paired[2*index]["medians"]["jev_prepare_context"] = 1.0
                paired[2*index+1]["medians"]["jev_prepare_context"] = 1.1
            elif index == 1:
                paired[2*index]["medians"]["jev_prepare_context"] = 1.0
        with mock.patch.object(campaign, "independent_paired_analyze", wraps=campaign.independent_paired_analyze) as analyze:
            stats = campaign.paired_supported_statistics(paired, orders, 418393)
        optional = stats["optional"]["jev_prepare_context"]
        self.assertEqual((optional["matched_pairs"], optional["order_counts"]), (2, {"AB": 1, "BA": 1}))
        self.assertTrue(optional["numeric_comparison_supported"])
        self.assertEqual(analyze.call_count, 2)
        unbalanced = copy.deepcopy(paired)
        unbalanced[30]["medians"].pop("jev_prepare_context")
        unbalanced[31]["medians"].pop("jev_prepare_context")
        unsupported = campaign.paired_supported_statistics(unbalanced, orders, 418393)["optional"]["jev_prepare_context"]
        self.assertEqual(unsupported["status"], "matched-unbalanced-pairs")
        self.assertFalse(unsupported["numeric_comparison_supported"])
        self.assertNotIn("analysis", unsupported)

    def test_production_pid_receipt_and_wall_grandchild_correlation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = root / "synthetic-production.exe"
            expected.write_bytes(b"synthetic identity only; never executed")
            def make_receipt(pid, observed, lane):
                observed = copy.deepcopy(observed)
                for child in observed["children"]:
                    if child["role"] == "production":
                        child["running_image"] = str(expected)
                spec = campaign.ProcessSpec("timing", "case", (str(expected),), root / "work", root / "out.json", root / "stdout", root / "stderr", pin_affinity=True, nested_wall=(lane == "wall"), expected_running_image=expected)
                return campaign.ProcessReceipt(spec, pid, 0, str(expected), {}, {}, telemetry(), observed, "synthetic", "synthetic")
            direct = r4mock.lane_report("allocation")
            direct["run"]["pid"] = 101
            receipt = make_receipt(101, affinity(101), "allocation")
            self.assertEqual(campaign.correlate_production_pid(receipt, direct, "allocation"), 101)
            wrong = copy.deepcopy(direct); wrong["run"]["pid"] = 102
            self.assert_stage("production-pid-correlation", campaign.correlate_production_pid, receipt, wrong, "allocation")
            wrong_receipt = make_receipt(102, affinity(101), "allocation")
            self.assert_stage("production-pid-correlation", campaign.correlate_production_pid, wrong_receipt, direct, "allocation")
            mismatched_direct = make_receipt(102, affinity(102), "allocation")
            self.assert_stage("production-pid-correlation", campaign.correlate_production_pid, mismatched_direct, direct, "allocation")
            wall = r4mock.external_wall()
            wall["child_report"]["run"]["pid"] = 202
            wall["child"]["pid"] = 202
            nested = affinity(202, wall=True)
            wall_receipt = make_receipt(99, nested, "wall")
            self.assertEqual(campaign.correlate_production_pid(wall_receipt, wall, "wall"), 202)
            campaign.validate_launch_image(wall_receipt.spec, sys.executable)
            self.assert_stage("wall-wrapper-image", campaign.validate_launch_image, wall_receipt.spec, str(expected))
            campaign.validate_launch_image(receipt.spec, str(expected))
            wrong_wall = copy.deepcopy(wall); wrong_wall["child_report"]["run"]["pid"] = 203; wrong_wall["child"]["pid"] = 203
            self.assert_stage("production-pid-correlation", campaign.correlate_production_pid, wall_receipt, wrong_wall, "wall")
            wrong_wrapper = make_receipt(98, nested, "wall")
            self.assert_stage("production-wrapper-correlation", campaign.correlate_production_pid, wrong_wrapper, wall, "wall")

    def test_injected_process_adapter_and_gate_a_timing_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = RecordingAdapter()
            executor = campaign.CampaignExecutor(adapter, root / "artifacts", root / "fixtures", root / "repo", root / "inst.exe", root / "wall.exe", root / "r3.exe")
            argv = campaign.replay_command(root / "inst.exe", root / "fixtures/small-ascii", root / "artifacts/gate-a/synthetic/work", root / "artifacts/gate-a/synthetic/out.json", 1)
            receipt, report = executor._run("gate-a", "synthetic", argv, gate_a=True)
            self.assertEqual(report["case_count"], 1)
            self.assertEqual(len(adapter.receipts), 1)
            self.assertEqual(receipt.telemetry["observation"]["page_cache_state"], "unobserved")
            bad = argv + ("--timing-enabled",)
            self.assert_stage("gate-a-timing-exclusion", executor._run, "gate-a", "forbidden", bad, gate_a=True)


    def test_injected_process_failure_retains_receipt_without_statistics(self):
        class FailingAdapter:
            def run(self, spec):
                spec.work.mkdir(parents=True)
                spec.stdout.write_bytes(b"synthetic failure")
                spec.stderr.write_bytes(b"[synthetic-exit] failed")
                raise campaign.Reject("synthetic-exit", "injected process failure")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executor = campaign.CampaignExecutor(FailingAdapter(), root / "artifacts", root / "fixtures", root / "repo", root / "inst.exe", root / "wall.exe", root / "r3.exe")
            argv = campaign.replay_command(root / "inst.exe", root / "fixtures/small-ascii", root / "artifacts/gate-a/failure/work", root / "artifacts/gate-a/failure/out.json", 1)
            self.assert_stage("synthetic-exit", executor._run, "gate-a", "failure", argv, gate_a=True)
            failure = json.loads((root / "artifacts/gate-a/failure/failure.json").read_text())
            self.assertEqual(failure["dependent_statistics"], None)
            self.assertIn("stderr", failure["retained"])
            self.assertFalse((root / "artifacts/gate-a/failure/work/receipt.json").exists())

    def test_allowlisted_environment_secret_refusal(self):
        env = campaign.validated_allowlisted_env({"R5_SYNTHETIC_MODE": "yes"})
        self.assertEqual(env["R5_SYNTHETIC_MODE"], "yes")
        self.assert_stage("environment-allowlist", campaign.validated_allowlisted_env, {"R5_API_KEY": "forbidden"})
        self.assert_stage("environment-allowlist", campaign.validated_allowlisted_env, {"R5_MODE": 1})

    def test_cli_command_shapes_no_timing_gate_a(self):
        exe = Path("synthetic.exe"); fixture = Path("synthetic-fixture")
        replay_argv = campaign.replay_command(exe, fixture, Path("work"), Path("out"), 2, failure=True, anchor=True)
        self.assertIn("--summary-failure", replay_argv)
        self.assertIn("20000", replay_argv)
        campaign.no_gate_a_timing_flags(replay_argv)
        for args in (replay_argv + ("--timing",), replay_argv + ("--timing-enabled",)):
            self.assert_stage("gate-a-timing-exclusion", campaign.no_gate_a_timing_flags, args)
        for name, expected_stage, _, _ in campaign.INVALID_CLI_SPECS:
            argv, stage = campaign.invalid_cli_command(exe, fixture, Path("new"), name)
            self.assertEqual(stage, expected_stage)
            self.assertNotIn("--timing", argv)
        bench = campaign.bench_command(exe, fixture, Path("out"))
        self.assertEqual(bench[bench.index("--warmup") + 1:bench.index("--warmup") + 2][0], "10")
        self.assertEqual(bench[bench.index("--iters") + 1:bench.index("--iters") + 2][0], "30")

    def test_rust_collision_main_create_new_contract_and_evidence(self):
        # Read the frozen Rust expressions only; this synthetic control never runs Cargo or an exe.
        rust_path = SCRIPT_DIR.parent / campaign.R4_RUST_REPLAY_RELATIVE
        rust_source = rust_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp:
            protected = Path(temp) / "gate-a/r4-negative/create-new-collision-rejected/out.json"
            protected.parent.mkdir(parents=True)
            protected.write_bytes(b"unchanged protected synthetic report\n")
            before = campaign.file_identity(protected)
            good_error = f"error: create_new {protected}: The file exists. (os error 80)\n"
            valid = dict(exit_code=1, stderr=good_error, protected_output=protected, before=before, after=before.copy(), produced_new_output=None, dependent_statistics=[], attempt_count=1, rust_source=rust_source)
            accepted = campaign.validate_r4_collision_rejection(**valid)
            self.assertTrue(accepted["rejected"])
            self.assertEqual(accepted["intended_stage"], "create_new")
            self.assertEqual(accepted["main_error_prefix"], "error: ")
            self.assertEqual(accepted["protected_output_path"], str(protected))
            self.assertEqual(accepted["os_error_text"], "The file exists. (os error 80)")
            self.assertTrue(accepted["output_unchanged"])
            self.assertIsNone(accepted["produced_new_output"])
            self.assertEqual(accepted["dependent_statistics"], [])
            self.assertEqual(accepted["attempt_count"], 1)
            self.assertEqual(accepted["rust_source_contract"]["path"], "crates/pi-coding-agent/examples/compaction_replay.rs")
            # The exact frozen Rust main and writer expressions must remain the basis of admission.
            self.assert_stage("r4-rust-contract", campaign.validate_r4_collision_rejection, **{**valid, "rust_source": rust_source.replace("create_new {}: {error}", "create_other {}: {error}")})
            bad_stderr = [
                ("ERROR: create_new " + str(protected) + ": The file exists. (os error 80)\n", "r4-collision-stderr"),
                ("error: output-collision\n", "r4-collision-stderr"),
                ("error: create_new " + str(protected) + ".different: The file exists. (os error 80)\n", "r4-collision-stderr"),
                ("error: create_new " + str(protected) + ": \r\n", "r4-collision-os-error"),
                (good_error + "error: unrelated second line\n", "r4-collision-stderr"),
            ]
            for stderr, stage in bad_stderr:
                self.assert_stage(stage, campaign.validate_r4_collision_rejection, **{**valid, "stderr": stderr})
            self.assert_stage("r4-collision-exit", campaign.validate_r4_collision_rejection, **{**valid, "exit_code": 0})
            self.assert_stage("r4-output-unchanged", campaign.validate_r4_collision_rejection, **{**valid, "after": {**before, "sha256": "0" * 64}})
            self.assert_stage("r4-collision-new-output", campaign.validate_r4_collision_rejection, **{**valid, "produced_new_output": {"synthetic": "replacement report"}})
            self.assert_stage("r4-collision-dependent-statistics", campaign.validate_r4_collision_rejection, **{**valid, "dependent_statistics": [{"median_ms": 1}]})
            self.assert_stage("r4-collision-attempt-count", campaign.validate_r4_collision_rejection, **{**valid, "attempt_count": 2})
            self.assert_stage("r4-collision-protected-path", campaign.validate_r4_collision_rejection, **{**valid, "before": {**before, "path": str(protected) + ".different"}})

    def test_spec_prevents_python_bytecode_without_touching_native_env(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            instrumented, wall, oracle = root / "instrumented.exe", root / "wall.exe", root / "oracle.exe"
            adapter = mock.Mock()
            guard = object.__new__(campaign.IntegrityGuard)
            guard.classes = {"exe": {}}
            adapter.guard = guard
            executor = campaign.CampaignExecutor(adapter, root / "artifacts", root / "corpus", root / "repo", instrumented, wall, oracle)
            executor.focused_executables = {}
            python_profile = root / "artifacts/profiles/python"
            for run_id, script_name in (("r3", "test_compaction_paired_protocol.py"), ("r4", "test_compaction_r4_lanes.py"), ("wall-wrapper", "compaction_r4_lanes.py")):
                with self.subTest(python_route=run_id):
                    spec = executor._spec("synthetic", run_id, (sys.executable, root / "repo/scripts" / script_name, root / "artifacts/synthetic" / run_id / "out.json"), profile_root=python_profile, create_profile=True, profile_id="synthetic-python")
                    self.assertEqual(spec.allowlisted_env["PYTHONDONTWRITEBYTECODE"], "1")
                    self.assertEqual(spec.allowlisted_env["PRIME_AGENT_CODING_AGENT_DIR"], str(python_profile.resolve()))
                    self.assertEqual(set(spec.allowlisted_env), {"PYTHONDONTWRITEBYTECODE", "PRIME_AGENT_CODING_AGENT_DIR"})
            native_profile = root / "artifacts/profiles/native"
            native_spec = executor._spec("synthetic", "native", (instrumented, "self-test-r1"), profile_root=native_profile, create_profile=True, profile_id="synthetic-native")
            self.assertNotIn("PYTHONDONTWRITEBYTECODE", native_spec.allowlisted_env)
            self.assertEqual(native_spec.allowlisted_env["PRIME_AGENT_CODING_AGENT_DIR"], str(native_profile.resolve()))
            self.assertEqual(set(native_spec.allowlisted_env), {"PRIME_AGENT_CODING_AGENT_DIR"})
            adapter.run.assert_not_called()

    def test_large_tool_huge_count_is_bound_to_frozen_successful_baseline(self):
        """Bind all N values to frozen successful baseline SHA256 16b82c5c6b71c3ffe22bfb683c8fd7d3d5e5f14de6114818d2c2057f24a723ea."""
        frozen_baseline = {'small-ascii': 1,
         'small-cjk': 1,
         'small-emoji': 1,
         'small-mixed-crlf': 1,
         'small-tool-huge': 1,
         'small-many-small': 1,
         'medium-ascii': 2,
         'medium-cjk': 1,
         'medium-emoji': 2,
         'medium-mixed-crlf': 2,
         'medium-tool-huge': 2,
         'medium-many-small': 2,
         'large-ascii': 4,
         'large-cjk': 3,
         'large-emoji': 3,
         'large-mixed-crlf': 3,
         'large-tool-huge': 2,
         'large-many-small': 4,
         'near-threshold-ascii': 2,
         'xlarge-ascii': 10,
         'xlarge-tool-huge': 3}
        self.assertEqual(campaign.CASE_SUMMARY_CALLS, frozen_baseline)
        self.assertEqual(campaign.CASE_ORDER, tuple(frozen_baseline))
        self.assertEqual(campaign.EXPECTED_ONE_CALL_CASES, frozenset(name for name, count in frozen_baseline.items() if count == 1))
        self.assertEqual(campaign.R3_ONE_SUMMARY_CALL_CASES, campaign.EXPECTED_ONE_CALL_CASES)
        argv = campaign.replay_command(Path("instrumented.exe"), Path("fixture/large-tool-huge"), Path("work"), Path("writer.json"), campaign.CASE_SUMMARY_CALLS["large-tool-huge"])
        self.assertEqual(argv[argv.index("--expect-summary-calls") + 1], "2")
        self.assertEqual(argv[argv.index("--context-window") + 1], "1000000")
        self.assertEqual(argv[argv.index("--provider-delay-ms") + 1], "0")
        self.assertEqual(argv[argv.index("--timeout-secs") + 1], "900")
        wrong_r7_counts = dict(frozen_baseline)
        wrong_r7_counts["large-tool-huge"] = 3
        self.assertNotEqual(campaign.CASE_SUMMARY_CALLS, wrong_r7_counts)

    def test_restored_pairing_binds_exact_frozen_identities(self):
        expected =         {   'small-ascii': {   'pairing': {   'tool_calls': 0,
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
        self.assertEqual(campaign.FROZEN_SUCCESSFUL_REPLAY_SHA256, "16b82c5c6b71c3ffe22bfb683c8fd7d3d5e5f14de6114818d2c2057f24a723ea")
        self.assertEqual(campaign.FROZEN_SUCCESSFUL_RESTORED_IDENTITIES, expected)
        self.assertEqual(tuple(expected), campaign.CASE_ORDER)
        zero_cases = [name for name in campaign.CASE_ORDER if expected[name]["pairing"]["tool_calls"] == 0]
        positive_cases = [name for name in campaign.CASE_ORDER if expected[name]["pairing"]["tool_calls"] > 0]
        self.assertEqual(zero_cases, ["small-ascii", "small-cjk", "small-emoji", "small-mixed-crlf", "small-tool-huge", "small-many-small", "large-tool-huge", "xlarge-tool-huge"])
        self.assertEqual(len(positive_cases), 13)
        for case_name, identity in expected.items():
            with self.subTest(case=case_name, outcome="accepted-frozen-identity"):
                report = replay(case_name, campaign.CASE_SUMMARY_CALLS[case_name], use_frozen_restored_identity=True)
                accepted = campaign.validate_replay_case(report, case_name, campaign.CASE_SUMMARY_CALLS[case_name], frozen_restored_identity=identity)
                self.assertEqual(accepted["case"], case_name)
                self.assertEqual(accepted["retained_state_sha256"], identity["retained_state_sha256"])
        reports = {name: replay(name, count, use_frozen_restored_identity=True) for name, count in campaign.CASE_SUMMARY_CALLS.items()}
        aggregate = campaign.aggregate_all21(reports)
        self.assertEqual((aggregate["case_count"], aggregate["failures"]), (21, []))
        # A structurally valid zero is still rejected for a positive baseline case.
        wrong_zero = replay("medium-ascii", campaign.CASE_SUMMARY_CALLS["medium-ascii"], use_frozen_restored_identity=True)
        wrong_zero["cases"][0]["restored_pairing"] = pairing(0)
        with self.assertRaisesRegex(ValueError, r"\[replay-restored-pairing-frozen-baseline\]"):
            campaign.validate_replay_case(wrong_zero, "medium-ascii", campaign.CASE_SUMMARY_CALLS["medium-ascii"], frozen_restored_identity=expected["medium-ascii"])
        # A structurally valid positive is rejected for an approved-zero case and when its positive count differs.
        for case_name, wrong_calls in (("small-ascii", 1), ("large-ascii", expected["large-ascii"]["pairing"]["tool_calls"] - 1)):
            wrong_positive = replay(case_name, campaign.CASE_SUMMARY_CALLS[case_name], use_frozen_restored_identity=True)
            wrong_positive["cases"][0]["restored_pairing"] = pairing(wrong_calls)
            with self.subTest(case=case_name, outcome="wrong-positive-rejected"), self.assertRaisesRegex(ValueError, r"\[replay-restored-pairing-frozen-baseline\]"):
                campaign.validate_replay_case(wrong_positive, case_name, campaign.CASE_SUMMARY_CALLS[case_name], frozen_restored_identity=expected[case_name])
        # Counts alone are insufficient: the live retained-state hash is also case-bound.
        wrong_hash = replay("large-tool-huge", 2, use_frozen_restored_identity=True)
        wrong_hash["cases"][0]["fresh_process_reopen"]["report"]["retained_state_sha256"] = "f" * 64
        wrong_hash["cases"][0]["digests"]["retained_state_sha256"] = "f" * 64
        wrong_hash["cases"][0]["digests"]["reopen_context_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, r"\[replay-restored-retained-state-frozen-baseline\]"):
            campaign.validate_replay_case(wrong_hash, "large-tool-huge", 2, frozen_restored_identity=expected["large-tool-huge"])
        # Raw durable emptiness is never relaxed, including for an approved-zero restored case.
        raw_empty = replay("small-ascii", 1, use_frozen_restored_identity=True)
        raw_empty["cases"][0]["raw_durable_pairing"] = pairing(0)
        self.assert_stage("replay-raw-pairing", campaign.validate_replay_case, raw_empty, "small-ascii", 1)
        # Omitting the exact frozen binding preserves the original nonempty admission; there is no all-zero fallback.
        unpinned_zero = replay("small-ascii", 1, use_frozen_restored_identity=True)
        self.assert_stage("replay-restored-pairing", campaign.validate_replay_case, unpinned_zero, "small-ascii", 1)
        # The separate anchor path keeps its original positive-pairing semantics.
        anchor_fixture = replay("small-ascii", 2)
        anchor_fixture["config"]["context_window"] = 20000.0
        anchor_fixture["config"]["timeout_secs"] = 90
        anchor = campaign.validate_replay_case(anchor_fixture, "small-ascii", 2, context_window=20000.0, timeout_secs=90)
        self.assertEqual(anchor["retained_state_sha256"], C)

    def test_success_replay_uses_reopen_digest_instead_of_legacy_retained_digest(self):
        case = replay("small-ascii", 2)
        case["config"]["context_window"] = 20000.0
        case["config"]["timeout_secs"] = 90
        row = case["cases"][0]
        row["digests"]["retained_state_sha256"] = A
        self.assertEqual(row["digests"]["reopen_context_sha256"], C)
        self.assertEqual(row["fresh_process_reopen"]["report"]["retained_state_sha256"], C)
        self.assertNotEqual(row["digests"]["retained_state_sha256"], C)
        accepted = campaign.validate_replay_case(case, "small-ascii", 2, context_window=20000.0, timeout_secs=90)
        self.assertEqual(accepted["retained_state_sha256"], C)

    def test_summary_failure_uses_reopen_digest_and_rejects_wrong_reader(self):
        case = replay("small-ascii", 1, failure=True)
        case["config"]["context_window"] = 20000.0
        case["config"]["timeout_secs"] = 90
        row = case["cases"][0]
        row["digests"]["retained_state_sha256"] = A
        self.assertEqual(row["digests"]["reopen_context_sha256"], C)
        self.assertEqual(row["fresh_process_reopen"]["report"]["retained_state_sha256"], C)
        self.assertEqual((row["outcome"]["compaction_entries"], row["outcome"]["transient_failure_notice_count"]), (0, 1))
        self.assertEqual(row["protected_state"]["checkpoints"], [])
        self.assertEqual(row["fresh_process_reopen"]["report"]["restored_message_count"], 3)
        accepted = campaign.validate_replay_case(case, "small-ascii", summary_calls=1, failure=True, context_window=20000.0, timeout_secs=90)
        self.assertEqual(accepted["retained_state_sha256"], C)
        wrong_reader = copy.deepcopy(case)
        wrong_reader["cases"][0]["fresh_process_reopen"]["report"]["retained_state_sha256"] = A
        self.assert_stage("replay-reader-parity", campaign.validate_replay_case, wrong_reader, "small-ascii", summary_calls=1, failure=True, context_window=20000.0, timeout_secs=90)

    def test_wrong_reopen_digest_fails_even_when_legacy_retained_matches_reader(self):
        case = replay("small-ascii", 2)
        case["config"]["context_window"] = 20000.0
        case["config"]["timeout_secs"] = 90
        row = case["cases"][0]
        self.assertEqual(row["digests"]["retained_state_sha256"], C)
        self.assertEqual(row["fresh_process_reopen"]["report"]["retained_state_sha256"], C)
        row["digests"]["reopen_context_sha256"] = A
        self.assert_stage("replay-reader-parity", campaign.validate_replay_case, case, "small-ascii", 2, context_window=20000.0, timeout_secs=90)

    def test_local_parity_matches_actual_lexical_case_order(self):
        expected_order = tuple(sorted(campaign.CASE_ORDER))
        self.assertEqual(expected_order[0], "large-ascii")
        baseline = parity_report()
        candidate = parity_report()
        self.assertEqual(tuple(row["case"] for row in baseline["cases"]), expected_order)
        self.assertEqual(tuple(row["case"] for row in candidate["cases"]), expected_order)
        self.assertEqual(baseline["cases"][0]["case"], "large-ascii")
        self.assertEqual(len(set(row["case"] for row in baseline["cases"])), 21)
        accepted = campaign.compare_local_parity(baseline, candidate)
        self.assertEqual(accepted["case_count"], 21)
        self.assertEqual(tuple(accepted["structured_sha256"]), expected_order)
        self.assertEqual(accepted["sections"], list(campaign.PARITY_SECTIONS))

        extra = parity_report()
        extra["cases"].append(copy.deepcopy(extra["cases"][0]))
        self.assert_stage("all21-case-count", campaign.compare_local_parity, extra, parity_report())

        duplicate = parity_report()
        duplicate["cases"][1] = copy.deepcopy(duplicate["cases"][0])
        self.assert_stage("all21-case-order", campaign.compare_local_parity, duplicate, parity_report())

        for label in ("R3", "R4"):
            with self.subTest(argument_swap=label):
                swapped_baseline = parity_report()
                swapped_candidate = parity_report()
                target = swapped_baseline if label == "R3" else swapped_candidate
                target["cases"][0], target["cases"][-1] = target["cases"][-1], target["cases"][0]
                self.assert_stage("all21-case-order", campaign.compare_local_parity, swapped_baseline, swapped_candidate)

        legacy_baseline = parity_report()
        legacy_candidate = parity_report()
        for report in (legacy_baseline, legacy_candidate):
            report["cases"].sort(key=lambda row: campaign.CASE_ORDER.index(row["case"]))
        self.assertEqual(legacy_baseline["cases"][0]["case"], "small-ascii")
        self.assert_stage("all21-case-order", campaign.compare_local_parity, legacy_baseline, legacy_candidate)

        unknown = parity_report()
        unknown["cases"][0]["case"] = "unknown-case"
        self.assert_stage("all21-case-order", campaign.compare_local_parity, unknown, parity_report())

    def test_r3_self_test_uses_positional_report_path(self):
        # The successful R4 control used one positional report path.
        commands_path = SCRIPT_DIR.parent.parent / "optimus-compaction-r4-artifacts/commands-final.json"
        retained = next(row for row in json.loads(commands_path.read_text())["commands"] if row["name"] == "R3-71-and-R4-72-protocol-controls")
        self.assertEqual(retained["exit_code"], 0)
        frozen_r3 = retained["command"].split(" && ", 1)[0].replace(chr(92), "/")
        quoted = frozen_r3.split('"')
        self.assertEqual(Path(quoted[3]).name, "test_compaction_paired_protocol.py")
        self.assertEqual(Path(quoted[5]).name, "r3-protocol-controls-final.json")
        self.assertIn(f'"{quoted[3]}" "{quoted[5]}"', frozen_r3)
        self.assertNotIn("--out", frozen_r3)
        script = (SCRIPT_DIR / "test_compaction_paired_protocol.py").read_text()
        self.assertIn("parser_out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None", script)
        self.assertIn('with parser_out.open("x", encoding="utf-8"', script)
        frozen_report_path = SCRIPT_DIR.parent.parent / "optimus-compaction-r4-artifacts/controls/r3-protocol-controls-final.json"
        frozen_report = json.loads(frozen_report_path.read_text())
        self.assertEqual(frozen_report["schema"], "r3-protocol-controls-v1")
        self.assertEqual(frozen_report["status"], "ok")
        self.assertEqual(frozen_report["check_count"], 71)
        self.assertEqual(len(frozen_report["checks"]), 71)
        self.assertFalse(frozen_report["performance_claim"])
        # Capture the real Gate A argv. No child or real fixture runs.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            instrumented, wall, oracle = root / "instrumented.exe", root / "wall.exe", root / "oracle.exe"
            hashes = {str(instrumented): campaign.R4_INSTRUMENTED_EXE_SHA256, str(wall): campaign.R4_WALL_EXE_SHA256, str(oracle): campaign.R3_REPLAY_EXE_SHA256}
            guard = object.__new__(campaign.IntegrityGuard)
            guard.classes = {"exe": {}}
            adapter = mock.Mock()
            adapter.guard = guard
            executor = campaign.CampaignExecutor(adapter, root / "artifacts", root / "corpus", root / "repo", instrumented, wall, oracle)
            executor.focused_executables = {}
            seen = {}
            def capture(stage, name, argv, **kwargs):
                seen[name] = tuple(str(value) for value in argv)
                if name == "r3":
                    raise RuntimeError("stop-after-r3-argv-capture")
                return None, {}
            def synthetic_identity(path):
                return {"path": str(path), "bytes": 0, "sha256": hashes[str(path)]}
            with mock.patch.object(campaign, "admit_source_identity", return_value={"synthetic": True}), mock.patch.object(campaign, "file_identity", side_effect=synthetic_identity), mock.patch.object(executor, "register_focused_plan"), mock.patch.object(executor, "_run", side_effect=capture):
                with self.assertRaisesRegex(RuntimeError, "stop-after-r3-argv-capture"):
                    executor.run_gate_a({}, {}, {}, {})
            argv = seen["r3"]
            expected_out = root / "artifacts/gate-a/controls/r3/out.json"
            self.assertEqual(len(argv), 3)
            self.assertEqual(argv[0], sys.executable)
            self.assertEqual(argv[1], str(root / "repo/scripts/test_compaction_paired_protocol.py"))
            self.assertEqual(argv[2], str(expected_out))
            self.assertNotIn("--out", argv[2:])
            executor.validate_external_argv(argv)
            rejected = (
                argv[:2] + ("--out", argv[2]),
                argv[:2] + ("--report", argv[2]),
                argv[:2] + (argv[2], "extra"),
                argv[:2] + ("--out",),
                argv[:2] + (str(root / "escaped-r3-report.json"),),
                argv[:2] + (str(root / "artifacts/../escaped-r3-report.json"),),
            )
            for invalid_argv in rejected:
                with self.subTest(invalid_argv=invalid_argv):
                    self.assert_stage("command-catalog", executor.validate_external_argv, invalid_argv)

    def test_r2_self_test_uses_frozen_single_case_fixture(self):
        # The successful retained command passed one case to the R3 helper.
        commands_path = SCRIPT_DIR.parent.parent / "optimus-compaction-r4-artifacts/commands-final.json"
        retained = next(row for row in json.loads(commands_path.read_text())["commands"] if row["name"] == "R1-R2-replay-and-exact-local-parity")
        self.assertEqual(retained["exit_code"], 0)
        normalized_command = retained["command"].replace("\\", "/")
        expected_case = normalized_command.split("/fixtures-unmodified/", 1)[1].split('"', 1)[0]
        self.assertEqual(expected_case, "small-ascii")
        helper_path = SCRIPT_DIR.parent.parent / "optimus-compaction-r3-artifacts/scripts/run-r3-replay-controls.py"
        helper = helper_path.read_text()
        self.assertIn('FIXTURE = pathlib.Path(sys.argv[4]).resolve()', helper)
        self.assertIn('"self-test-r2", ["self-test-r2", "--fixture", str(FIXTURE)', helper)
        rust_source = (SCRIPT_DIR.parent / "crates/pi-coding-agent/examples/compaction_replay/local_parity.rs").read_text()
        self.assertIn('ensure_stage(cases.len() == 1, "selftest-fixture", "R2 self-test requires exactly one fixture case")?', rust_source)
        # Observe the real run_gate_a control argv without executing any child or opening a fixture.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            instrumented, wall, oracle = root / "instrumented.exe", root / "wall.exe", root / "oracle.exe"
            hashes = {str(instrumented): campaign.R4_INSTRUMENTED_EXE_SHA256, str(wall): campaign.R4_WALL_EXE_SHA256, str(oracle): campaign.R3_REPLAY_EXE_SHA256}
            guard = object.__new__(campaign.IntegrityGuard)
            guard.classes = {"exe": {}}
            adapter = mock.Mock(); adapter.guard = guard
            executor = campaign.CampaignExecutor(adapter, root / "artifacts", root / "corpus", root / "repo", instrumented, wall, oracle)
            executor.focused_executables = {}
            seen = {}
            def capture(stage, name, argv, **kwargs):
                seen[name] = tuple(str(value) for value in argv)
                if name == "r3":
                    raise RuntimeError("stop-after-r2-argv-capture")
                return None, {}
            def synthetic_identity(path):
                return {"path": str(path), "bytes": 0, "sha256": hashes[str(path)]}
            with mock.patch.object(campaign, "admit_source_identity", return_value={"synthetic": True}), mock.patch.object(campaign, "file_identity", side_effect=synthetic_identity), mock.patch.object(executor, "register_focused_plan"), mock.patch.object(executor, "_run", side_effect=capture):
                with self.assertRaisesRegex(RuntimeError, "stop-after-r2-argv-capture"):
                    executor.run_gate_a({}, {}, {}, {})
            argv = seen["r2"]
            self.assertEqual(argv[1], "self-test-r2")
            actual_fixture = argv[argv.index("--fixture") + 1]
            self.assertEqual(actual_fixture, str(root / "corpus" / expected_case))
            self.assertNotEqual(actual_fixture, str(root / "corpus"))
            # Synthetic wrong-count failure: mirror the read-only Rust contract.
            # This exercises R5 failure routing without any real fixture or binary.
            class WrongCountAdapter:
                def run(self, spec):
                    spec.work.mkdir(parents=True)
                    spec.stderr.write_bytes(("error: [selftest-fixture] R2 self-test requires exactly one fixture case" + chr(10)).encode())
                    raise campaign.Reject("selftest-fixture", "R2 self-test requires exactly one fixture case")
            wrong_executor = campaign.CampaignExecutor(WrongCountAdapter(), root / "artifacts", root / "corpus", root / "repo", instrumented, wall, oracle)
            wrong_argv = (instrumented, "self-test-r2", "--fixture", root / "corpus", "--work", root / "artifacts/gate-a/controls/r2-wrong-count/work", "--out", root / "artifacts/gate-a/controls/r2-wrong-count/out.json")
            observed_case_counts = {str(root / "corpus"): len(campaign.CASE_ORDER), str(root / "corpus" / expected_case): 1}
            self.assertNotEqual(observed_case_counts[str(root / "corpus")], 1)
            self.assertEqual(observed_case_counts[str(actual_fixture)], 1)
            self.assert_stage("selftest-fixture", wrong_executor._run, "gate-a/controls", "r2-wrong-count", wrong_argv, gate_a=True)
            failure_path = root / "artifacts/gate-a/controls/r2-wrong-count/failure.json"
            failure = json.loads(failure_path.read_text())
            self.assertIsNone(failure["dependent_statistics"])
            self.assertFalse((root / "artifacts/gate-a/controls/r2-wrong-count/out.json").exists())
            self.assertIn("stderr", failure["retained"])

    def test_rust_test_matrix_counts_and_names(self):
        expected = [f"test_case_{index:02d}" for index in range(99)]
        text = []
        position = 0
        for suite in range(14):
            text.append(f"running {7 if suite < 13 else 8} tests")
            for _ in range(7 if suite < 13 else 8):
                text.append(f"test {expected[position]} ... ok"); position += 1
            text.append(f"test result: ok. {7 if suite < 13 else 8} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;")
        result = campaign.parse_rust_test_log("\n".join(text), expected, 14, 99)
        self.assertEqual(result["passed"], 99)
        self.assert_stage("focused-test-matrix", campaign.parse_rust_test_log, "\n".join(text), list(reversed(expected)), 14, 99)
        failed = "\n".join(text).replace("0 failed;", "1 failed;", 1)
        self.assert_stage("focused-test-result", campaign.parse_rust_test_log, failed, expected, 14, 99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
