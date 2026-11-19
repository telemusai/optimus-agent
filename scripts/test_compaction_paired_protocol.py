#!/usr/bin/env python3
"""Process-level deterministic R3 protocol controls; synthetic data only."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
DRIVER = SCRIPT_DIR / "compaction_paired_protocol.py"
STATS = SCRIPT_DIR / "compaction_paired_stats.py"
MOCK = SCRIPT_DIR / "compaction_protocol_mock.py"
BASE_PYTHON = getattr(sys, "_base_executable", sys.executable)
ARMS = ("baseline", "candidate")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_plan(root: Path) -> tuple[Path, dict]:
    plan_dir = root / "plan"
    plan_dir.mkdir(parents=True)
    transcript = b'{"synthetic":true,"records":[]}\n'
    manifest = b'{"schema":1,"name":"synthetic","file":"transcript.json","keep_recent_tokens_hint":275.0}\n'
    fixture_hashes = {"transcript": hashlib.sha256(transcript).hexdigest(), "manifest": hashlib.sha256(manifest).hexdigest()}
    arms = {}
    fixtures = {}
    exe_hash = sha(Path(BASE_PYTHON))
    for arm_index, arm in enumerate(ARMS):
        arm_root = plan_dir / arm
        fixture_dir = arm_root / "fixture"
        fixture_dir.mkdir(parents=True)
        (fixture_dir / "transcript.json").write_bytes(transcript)
        (fixture_dir / "manifest.json").write_bytes(manifest)
        source_dir = arm_root / "source"
        source_dir.mkdir()
        source_copy = source_dir / "mock_source.py"
        source_copy.write_bytes(MOCK.read_bytes())
        target = arm_root / "target"
        target.mkdir()
        fake_head = hashlib.sha256(f"{arm}-head".encode()).hexdigest()[:40]
        fake_base = hashlib.sha256(f"{arm}-base".encode()).hexdigest()[:40]
        arms[arm] = {
            "source_root": str(source_dir),
            "source_sha256": {"mock_source.py": sha(source_copy)},
            "synthetic_source_identity": {"head": fake_head, "base": fake_base, "dirty": False},
            "target_dir": str(target),
            "exe": BASE_PYTHON,
            "exe_sha256": exe_hash,
            "argv": [BASE_PYTHON, str(MOCK), "run", "--out", "{out}"],
            "env": {
                "R3_MOCK_ARM": arm,
                "R3_MOCK_EXE_SHA256": exe_hash,
                "R3_MOCK_FIXTURE_TRANSCRIPT": str(fixture_dir / "transcript.json"),
            },
            "expected_runtime": {
                "charcount": {"requested_mode": "auto", "effective_backend": "avx2", "dispatch_diagnostic": "synthetic-self-check-ok"},
                "parallel": {"requested_mode": "auto", "available_parallelism": 6, "effective_workers": 6, "fallback_count": 0},
            },
            "output_kind": "mock-bench",
        }
        fixtures[arm] = {"transcript": str(fixture_dir / "transcript.json"), "manifest": str(fixture_dir / "manifest.json")}
    plan = {
        "schema": "r3-paired-protocol-v1",
        "protocol_class": "synthetic-non-decision",
        "seed": 418393,
        "outer_pairs": 30,
        "warmup": 10,
        "iterations": 30,
        "bootstrap_samples": 2000,
        "output_root": str(root / "result"),
        "lane": {
            "kind": "parallel-production",
            "process_state": "warm",
            "cpu_ids": [0, 2, 4, 6, 8, 10],
            "mask": "0x555",
            "topology": {"package_count": 1, "physical_cores": 6, "logical_processors": 12, "smt_siblings": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], "source": "synthetic copy of qualified campaign topology"},
        },
        "fixtures": fixtures,
        "expected_fixture_sha256": fixture_hashes,
        "arms": arms,
        "metrics": [{"case": "synthetic", "kernel": "mock_kernel"}],
        "telemetry": {
            "load_argv": [BASE_PYTHON, str(MOCK), "load"],
            "power_argv": [BASE_PYTHON, str(MOCK), "power"],
            "required_load_fields": ["cpu_total_percent", "processor_frequency", "processor_queue", "available_memory_bytes", "per_logical_processor"],
        },
        "measurement_limitations": [
            "All values and counters are deterministic synthetic qualification inputs, not measured Optimus performance or host telemetry.",
            "SMT, cache, thermal and host scheduling controls cannot be established by this synthetic control.",
        ],
        "toolchain": {
            "rust_version": "1.98.0-synthetic-label",
            "llvm_version": "synthetic-no-LLVM-claim",
            "target": "synthetic-x86_64-topology",
            "profile": "release",
            "locked": True,
            "offline": True,
            "rustc_path": BASE_PYTHON,
            "cargo_path": BASE_PYTHON,
        },
        "timeout_sec": 30,
    }
    path = plan_dir / "plan.json"
    write_new(path, plan)
    return path, plan


def invoke(commands: list[str]) -> dict:
    result = subprocess.run(commands, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return {"command": commands, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def control(checks: list[dict], name: str, receipt: dict, expected_stage: str | None = None, accepted: bool = False) -> None:
    if accepted:
        okay = receipt["exit_code"] == 0
        check = {"name": name, "accepted": okay, "receipt": receipt}
    else:
        okay = receipt["exit_code"] != 0 and f"[{expected_stage}]" in receipt["stderr"]
        check = {"name": name, "rejected": okay, "intended_stage": expected_stage, "receipt": receipt}
    checks.append(check)
    if not okay:
        raise AssertionError(f"control failed: {json.dumps(check, indent=2)}")


def synthetic_runs(orders: list[dict]) -> list[dict]:
    runs = []
    for item in orders:
        for index, arm in enumerate(item["order"]):
            runs.append({"pair_id": item["pair_id"], "arm": arm, "sequence_index": index, "status": "admitted", "medians": {"synthetic/mock": 9.0 if arm == "candidate" else 10.0}})
    return runs


def main() -> int:
    parser_out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    with tempfile.TemporaryDirectory(prefix="r3-protocol-controls-") as temp:
        root = Path(temp)
        checks: list[dict] = []
        plan_path, plan = build_plan(root / "valid")
        valid = invoke([sys.executable, str(DRIVER), "--plan", str(plan_path)])
        control(checks, "valid_30_pair_synthetic_protocol_accepted", valid, accepted=True)
        valid_root = Path(plan["output_root"])
        summary = json.loads((valid_root / "summary.json").read_text())
        assert summary["statistics"]["pair_count"] == 30
        assert summary["statistics"]["ab_count"] == summary["statistics"]["ba_count"] == 15
        assert summary["performance_claim"] is False and summary["decision_evidence"] is False
        assert len(summary["telemetry_blocks"]) == 30
        orders = json.loads((valid_root / "orders.json").read_text())
        runs = json.loads((valid_root / "outer-runs.json").read_text())
        assert len(runs) == 60 and len({run["run_id"] for run in runs}) == 60
        assert all(run["pid"] != os.getpid() and run["affinity"]["observed_mask"] == "0x555" for run in runs)
        assert all(run["preflight_fingerprints"] == [["synthetic", "a" * 64]] for run in runs)
        # Independent stats CLI, with no protocol driver import or process logic.
        stats_receipt = invoke([sys.executable, str(STATS), "--runs", str(valid_root / "outer-runs.json"), "--orders", str(valid_root / "orders.json"), "--seed", str(plan["seed"]), "--bootstrap-samples", "2000", "--out", str(root / "independent-stats.json")])
        control(checks, "independent_paired_stats_accepted", stats_receipt, accepted=True)
        independent = json.loads((root / "independent-stats.json").read_text())
        assert independent == summary["statistics"]
        # Exact seed/order replay in a second output root.
        replay_plan = copy.deepcopy(plan)
        replay_plan["output_root"] = str(root / "valid" / "replay-result")
        replay_plan["expected_order"] = orders
        replay_path = root / "valid" / "replay-plan.json"
        write_new(replay_path, replay_plan)
        replay_receipt = invoke([sys.executable, str(DRIVER), "--plan", str(replay_path)])
        control(checks, "seed_order_replay_accepted", replay_receipt, accepted=True)
        assert json.loads((Path(replay_plan["output_root"]) / "orders.json").read_text()) == orders

        def protocol_failure(name: str, stage: str, mutate) -> None:
            case_root = root / name
            path, value = build_plan(case_root)
            mutate(value, case_root)
            value["output_root"] = str(case_root / "result")
            failed_plan = case_root / "mutated-plan.json"
            write_new(failed_plan, value)
            control(checks, name, invoke([sys.executable, str(DRIVER), "--plan", str(failed_plan)]), stage)

        protocol_failure("wrong_fixture_rejected", "fixture-hash", lambda p, r: p["expected_fixture_sha256"].__setitem__("transcript", "0" * 64))
        protocol_failure("wrong_binary_rejected", "binary-hash", lambda p, r: p["arms"]["candidate"].__setitem__("exe_sha256", "0" * 64))
        protocol_failure("wrong_source_rejected", "source-hash", lambda p, r: p["arms"]["baseline"]["source_sha256"].__setitem__("mock_source.py", "0" * 64))
        protocol_failure("dirty_source_rejected", "source-dirty", lambda p, r: p["arms"]["baseline"]["synthetic_source_identity"].__setitem__("dirty", True))
        protocol_failure("shared_target_rejected", "unique-target", lambda p, r: p["arms"]["candidate"].__setitem__("target_dir", p["arms"]["baseline"]["target_dir"]))
        protocol_failure("wrong_mask_rejected", "lane-cpu-mask", lambda p, r: p["lane"].__setitem__("mask", "0x554"))
        def wrong_production_lane(p, r):
            # Mask matches these CPUs, so this must reach the dedicated production CPU-list rejection.
            p["lane"]["cpu_ids"] = [1, 3, 5, 7, 9, 11]
            p["lane"]["mask"] = "0xaaa"
        protocol_failure("wrong_parallel_lane_rejected", "lane-cpu-mask", wrong_production_lane)
        assert "parallel production lane must prefer 0,2,4,6,8,10" in checks[-1]["receipt"]["stderr"]
        for name, value in [
            ("plan_warmup_float_rejected", ("warmup", 10.0)),
            ("plan_warmup_bool_rejected", ("warmup", True)),
            ("plan_iterations_float_rejected", ("iterations", 30.0)),
            ("plan_iterations_bool_rejected", ("iterations", True)),
        ]:
            key, bad_value = value
            protocol_failure(name, "plan-protocol-count", lambda p, r, key=key, bad_value=bad_value: p.__setitem__(key, bad_value))
        protocol_failure("wrong_order_replay_rejected", "order-replay", lambda p, r: p.__setitem__("expected_order", list(reversed(orders))))
        for name, stage, fault in [
            ("failed_child_rejected", "child-exit", "nonzero"),
            ("missing_output_rejected", "output-missing", "missing-output"),
            ("wrong_config_rejected", "output-config", "wrong-warmup"),
            ("wrong_inner_count_rejected", "output-config", "wrong-count"),
            ("wrong_mode_rejected", "runtime-mode", "wrong-mode"),
            ("wrong_pid_rejected", "output-pid", "wrong-pid"),
            ("stamp_only_rejected", "binary-stamp", "stamp-only"),
            ("nan_rejected", "numeric-input", "nan"),
            ("infinity_rejected", "numeric-input", "infinity"),
            ("parity_rejected", "parity-admission", "wrong-parity"),
            ("missing_parity_rejected", "parity-admission", "missing-parity"),
            ("empty_parity_rejected", "parity-admission", "empty-parity"),
            ("malformed_parity_hash_rejected", "parity-admission", "malformed-parity-hash"),
            ("missing_parity_hash_rejected", "parity-admission", "missing-parity-hash"),
            ("missing_parity_case_rejected", "parity-admission", "missing-parity-case"),
            ("wrong_parity_case_rejected", "parity-admission", "wrong-parity-case"),
            ("duplicate_parity_case_rejected", "parity-admission", "duplicate-parity-case"),
            ("malformed_json_rejected", "output-json", "malformed-json"),
            ("non_object_output_rejected", "output-shape", "non-object"),
            ("float_warmup_rejected", "output-config", "float-warmup"),
            ("bool_warmup_rejected", "output-config", "bool-warmup"),
            ("float_iterations_rejected", "output-config", "float-iters"),
            ("bool_iterations_rejected", "output-config", "bool-iters"),
            ("float_inner_count_rejected", "output-config", "float-inner-count"),
            ("bool_inner_count_rejected", "output-config", "bool-inner-count"),
        ]:
            def mutate_fault(p, r, fault=fault):
                p["arms"]["baseline"]["env"]["R3_MOCK_FAULT"] = fault
            protocol_failure(name, stage, mutate_fault)
        protocol_failure("fixture_mutation_rejected", "fixture-after", lambda p, r: p["arms"]["baseline"]["env"].__setitem__("R3_MOCK_FAULT", "fixture-mutate"))
        protocol_failure("missing_telemetry_rejected", "telemetry-load", lambda p, r: p["telemetry"]["load_argv"].extend(["--outcome", "missing-queue"]))
        protocol_failure("telemetry_relaxation_rejected", "telemetry-load", lambda p, r: p["telemetry"]["required_load_fields"].pop())

        def existing_output(p, r):
            result = Path(p["output_root"])
            result.mkdir(parents=True)
            (result / "summary.json").write_text("collision", encoding="utf-8")
        protocol_failure("create_new_collision_rejected", "output-create-new", existing_output)

        # Pure non-timing parity controls use copies of the real preserved structured verify-local reports.
        # They invoke validate_parity directly; no bench, mock child or timing command is launched.
        preserved_artifacts = SCRIPT_DIR.parent.parent / "optimus-compaction-r3-artifacts"
        preserved_baseline = preserved_artifacts / "verify-local-baseline.json"
        preserved_candidate = preserved_artifacts / "verify-local-candidate.json"
        parity_program = (
            "import json,sys;from pathlib import Path;"
            "sys.path.insert(0,sys.argv[1]);"
            "from compaction_paired_protocol import validate_parity;"
            "result=validate_parity(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')));"
            "print(json.dumps({'status':'accepted','class':result['class'],'pure_non_timing_control':True}))"
        )

        def projection_control(name, stage=None, accepted=False, mutate=None):
            case_root = root / name
            unused_plan_path, value = build_plan(case_root)
            baseline = json.loads(preserved_baseline.read_text(encoding="utf-8"))
            candidate = json.loads(preserved_candidate.read_text(encoding="utf-8"))
            assert baseline["cases"][0]["case"] == "small-ascii"
            assert set(baseline["cases"][0]["full_projection_sha256"]) == {"algorithm", "sections", "note"}
            assert len(baseline["cases"][0]["full_projection_sha256"]["sections"]) == 14
            value["protocol_class"] = "authorized-r4-timing"
            value["metrics"] = [{"case": "small-ascii", "kernel": "mock_kernel"}]
            if mutate is not None:
                mutate(value, baseline, candidate)
            baseline_path = case_root / "preserved-baseline-copy.json"
            candidate_path = case_root / "preserved-candidate-copy.json"
            write_new(baseline_path, baseline)
            write_new(candidate_path, candidate)
            value["parity"] = {
                "baseline": {"path": str(baseline_path), "sha256": sha(baseline_path)},
                "candidate": {"path": str(candidate_path), "sha256": sha(candidate_path)},
            }
            plan_path = case_root / "structured-parity-plan.json"
            write_new(plan_path, value)
            receipt = invoke([sys.executable, "-c", parity_program, str(SCRIPT_DIR), str(plan_path)])
            control(checks, name, receipt, stage, accepted=accepted)
            if accepted:
                assert '"pure_non_timing_control": true' in receipt["stdout"]
            return receipt

        projection_control("structured_projection_valid_accepted", accepted=True)

        def projection(baseline_or_candidate):
            return baseline_or_candidate["cases"][0]["full_projection_sha256"]

        structured_failures = [
            ("structured_projection_scalar_rejected", "parity-projection-schema", lambda p,b,c: b["cases"][0].__setitem__("full_projection_sha256", "a" * 64)),
            ("structured_projection_missing_sections_rejected", "parity-projection-schema", lambda p,b,c: projection(b).pop("sections")),
            ("structured_projection_unexpected_content_rejected", "parity-projection-schema", lambda p,b,c: projection(b).__setitem__("unexpected", "content")),
            ("structured_projection_wrong_algorithm_rejected", "parity-projection-schema", lambda p,b,c: projection(b).__setitem__("algorithm", "wrong algorithm")),
            ("structured_projection_wrong_note_rejected", "parity-projection-schema", lambda p,b,c: projection(b).__setitem__("note", "wrong note")),
            ("structured_projection_missing_section_rejected", "parity-projection-sections", lambda p,b,c: projection(b)["sections"].pop("case")),
            ("structured_projection_unexpected_section_rejected", "parity-projection-sections", lambda p,b,c: projection(b)["sections"].__setitem__("unexpected", "a" * 64)),
            ("structured_projection_empty_section_hash_rejected", "parity-projection-section", lambda p,b,c: projection(b)["sections"].__setitem__("case", "")),
            ("structured_projection_malformed_section_hash_rejected", "parity-projection-section", lambda p,b,c: projection(b)["sections"].__setitem__("case", "A" * 64)),
            ("structured_projection_wrong_section_value_type_rejected", "parity-projection-section", lambda p,b,c: projection(b)["sections"].__setitem__("case", [])),
            ("structured_projection_wrong_sections_type_rejected", "parity-projection-sections", lambda p,b,c: projection(b).__setitem__("sections", [])),
            ("structured_projection_section_mismatch_rejected", "parity-admission", lambda p,b,c: projection(c)["sections"].__setitem__("case", "c" * 64)),
            ("structured_projection_wrong_case_coverage_rejected", "parity-admission", lambda p,b,c: b["cases"][0].__setitem__("case", "unexpected-case")),
        ]
        for name, stage, mutate in structured_failures:
            projection_control(name, stage, mutate=mutate)

        # Standalone statistical admission failures. Every case executes the CLI.
        stats_cases = []
        missing = [run for run in runs if not (run["pair_id"] == 0 and run["arm"] == "candidate")]
        stats_cases.append(("missing_run_rejected", "stats-missing-run", missing, orders))
        duplicate = runs + [copy.deepcopy(next(run for run in runs if run["pair_id"] == 0 and run["arm"] == "baseline"))]
        stats_cases.append(("duplicate_run_rejected", "stats-duplicate-run", duplicate, orders))
        failed = copy.deepcopy(runs)
        failed[0]["status"] = "failed"
        stats_cases.append(("failed_run_rejected", "stats-failed-run", failed, orders))
        unbalanced = copy.deepcopy(orders)
        unbalanced[0]["order"] = unbalanced[1]["order"]
        stats_cases.append(("ab_ba_imbalance_rejected", "stats-ab-ba-balance", synthetic_runs(unbalanced), unbalanced))
        duplicate_pair = copy.deepcopy(orders)
        duplicate_pair[1]["pair_id"] = 0
        stats_cases.append(("duplicate_pair_rejected", "stats-duplicate-pair", synthetic_runs(duplicate_pair), duplicate_pair))
        wrong_unit = copy.deepcopy(runs)
        wrong_unit[0]["sequence_index"] = 1 - wrong_unit[0]["sequence_index"]
        stats_cases.append(("paired_unit_order_rejected", "stats-paired-unit", wrong_unit, orders))
        bad_stat = copy.deepcopy(runs)
        bad_stat[0]["medians"] = {key: 0 for key in bad_stat[0]["medians"]}
        stats_cases.append(("nonpositive_stat_rejected", "stats-numeric-input", bad_stat, orders))
        for name, bad_value in [("order_pair_bool_rejected", False), ("order_pair_float_rejected", 0.0)]:
            bad_orders = copy.deepcopy(orders)
            bad_orders[0]["pair_id"] = bad_value
            stats_cases.append((name, "stats-order", synthetic_runs(bad_orders), bad_orders))
        for name, bad_value in [("run_pair_bool_rejected", False), ("run_pair_float_rejected", 0.0), ("run_sequence_bool_rejected", True), ("run_sequence_float_rejected", 0.0)]:
            bad_runs = copy.deepcopy(runs)
            key = "pair_id" if "pair_" in name else "sequence_index"
            bad_runs[0][key] = bad_value
            stats_cases.append((name, "stats-paired-unit", bad_runs, orders))
        for name, stage, case_runs, case_orders in stats_cases:
            case_dir = root / name
            case_dir.mkdir()
            write_new(case_dir / "runs.json", case_runs)
            write_new(case_dir / "orders.json", case_orders)
            receipt = invoke([sys.executable, str(STATS), "--runs", str(case_dir / "runs.json"), "--orders", str(case_dir / "orders.json"), "--seed", "418393", "--bootstrap-samples", "1000", "--out", str(case_dir / "out.json")])
            control(checks, name, receipt, stage)
            assert not (case_dir / "out.json").exists()
        report = {
            "schema": "r3-protocol-controls-v1",
            "status": "ok",
            "check_count": len(checks),
            "accepted": sum(bool(check.get("accepted")) for check in checks),
            "named_rejections": sum(bool(check.get("rejected")) for check in checks),
            "synthetic_non_decision": True,
            "optimus_timing_executed": False,
            "performance_claim": False,
            "checks": checks,
        }
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if parser_out:
            with parser_out.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
        print(json.dumps({"status": "ok", "checks": len(checks), "accepted": report["accepted"], "named_rejections": report["named_rejections"], "performance_claim": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
