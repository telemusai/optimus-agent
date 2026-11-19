#!/usr/bin/env python3
"""Deterministic synthetic child/probe for R3 protocol qualification only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "load", "power"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--outcome")
    args = parser.parse_args()
    if args.action in ("load", "power"):
        if args.action == "load":
            value = {
                "cpu_total_percent": 12.5,
                "processor_frequency": {"reported_nominal_mhz": 3194, "not_inferred_effective_clock": True},
                "processor_queue": 0,
                "available_memory_bytes": 96_000_000_000,
                "per_logical_processor": [{"cpu": cpu, "raw_counter": 10.0 + cpu} for cpu in range(12)],
                "synthetic_non_decision": True,
            }
        else:
            value = {"active_power_scheme": {"name": "Synthetic scheme", "guid": "00000000-0000-0000-0000-000000000000"}, "synthetic_non_decision": True}
        if args.action == "load" and args.outcome == "missing-queue":
            value.pop("processor_queue")
        print(json.dumps(value))
        return 0
    assert args.out is not None
    time.sleep(0.05)  # Allows the parent to inspect affinity and process image.
    fault = os.environ.get("R3_MOCK_FAULT", "")
    if fault == "malformed-json":
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write("{not-valid-json")
        return 0
    if fault == "non-object":
        write_new(args.out, [])
        return 0
    if fault == "fixture-mutate":
        fixture = Path(os.environ["R3_MOCK_FIXTURE_TRANSCRIPT"])
        fixture.write_bytes(fixture.read_bytes() + b" ")
    if fault == "nonzero":
        print("synthetic requested failure", file=sys.stderr)
        return 7
    if fault != "missing-output":
        p50 = 10.0 if os.environ.get("R3_MOCK_ARM") == "baseline" else 9.0
        if fault == "nan":
            p50 = float("nan")
        elif fault == "infinity":
            p50 = float("inf")
        n = 29 if fault == "wrong-count" else 30
        if fault == "float-inner-count":
            n = 30.0
        elif fault == "bool-inner-count":
            n = True
        warmup = 9 if fault == "wrong-warmup" else 10
        if fault == "float-warmup":
            warmup = 10.0
        elif fault == "bool-warmup":
            warmup = True
        iters = 30.0 if fault == "float-iters" else True if fault == "bool-iters" else 30
        runtime = {
            "charcount": {"requested_mode": "auto", "effective_backend": "avx2", "dispatch_diagnostic": "synthetic-self-check-ok"},
            "parallel": {"requested_mode": "auto", "available_parallelism": 6, "effective_workers": 6, "fallback_count": 0},
        }
        if fault == "wrong-mode":
            runtime["charcount"]["effective_backend"] = "reference"
        pid = os.getpid() + 1 if fault == "wrong-pid" else os.getpid()
        source = "--stamp" if fault == "stamp-only" else "computed-from-running-exe"
        checksum_source = "b" * 64 if fault == "wrong-parity" else "a" * 64
        result = {
            "harness": "r3-synthetic-mock",
            "subcommand": "bench-local",
            "schema": 1,
            "env": {
                "exe_sha256": os.environ["R3_MOCK_EXE_SHA256"],
                "exe_sha256_source": source,
                "pid": pid,
                "argv": [sys.executable] + sys.argv,
            },
            "runtime": runtime,
            "config": {"warmup": warmup, "iters": iters},
            "non_timed_preflight": [{"case": "synthetic", "full_output_sha256": checksum_source}],
            "cases": [{
                "case": "synthetic",
                "kernels": [{
                    "name": "mock_kernel",
                    "warmup": warmup,
                    "stats": {"n": n, "p50_ms": p50, "min": p50, "max": p50},
                    "checksum": 123.0,
                }],
            }],
            "performance_claim": False,
            "synthetic_non_decision": True,
        }
        if fault == "missing-parity":
            result.pop("non_timed_preflight")
        elif fault == "empty-parity":
            result["non_timed_preflight"] = []
        elif fault == "missing-parity-hash":
            result["non_timed_preflight"][0].pop("full_output_sha256")
        elif fault == "malformed-parity-hash":
            result["non_timed_preflight"][0]["full_output_sha256"] = "ABC-invalid"
        elif fault == "missing-parity-case":
            result["non_timed_preflight"][0].pop("case")
        elif fault == "wrong-parity-case":
            result["non_timed_preflight"][0]["case"] = "unexpected-case"
        elif fault == "duplicate-parity-case":
            result["non_timed_preflight"].append(dict(result["non_timed_preflight"][0]))
        if fault == "reuse-output":
            args.out.write_text("occupied", encoding="utf-8")
        write_new(args.out, result)
    print("synthetic non-decision run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
