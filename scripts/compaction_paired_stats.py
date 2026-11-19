#!/usr/bin/env python3
"""Paired outer-run statistics for the compaction benchmark protocol.

This module has no process execution or benchmark discovery logic. Inputs are
fresh-process outer medians, never inner-iteration samples.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import median
from typing import Any, Iterable

ARMS = ("baseline", "candidate")


class StageError(ValueError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


def require_finite_positive(value: Any, stage: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageError(stage, f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise StageError(stage, f"{label} must be finite and > 0, got {value!r}")
    return result


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise StageError("stats-bootstrap", "empty bootstrap distribution")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def validate_orders(orders: Any, expected_pairs: int | None = None) -> list[dict[str, Any]]:
    if not isinstance(orders, list) or not orders:
        raise StageError("stats-order", "orders must be a nonempty list")
    pair_ids: set[int] = set()
    counts = {"AB": 0, "BA": 0}
    for item in orders:
        if not isinstance(item, dict) or set(item) != {"pair_id", "order"}:
            raise StageError("stats-order", "each order needs exactly pair_id and order")
        pair_id = item["pair_id"]
        order = item["order"]
        if type(pair_id) is not int or pair_id < 0:
            raise StageError("stats-order", f"invalid pair_id {pair_id!r}")
        if pair_id in pair_ids:
            raise StageError("stats-duplicate-pair", f"duplicate pair_id {pair_id}")
        pair_ids.add(pair_id)
        if order not in (["baseline", "candidate"], ["candidate", "baseline"]):
            raise StageError("stats-order", f"pair {pair_id} is not AB or BA: {order!r}")
        counts["AB" if order[0] == "baseline" else "BA"] += 1
    if pair_ids != set(range(len(orders))):
        raise StageError("stats-missing-pair", "pair IDs must be contiguous from zero")
    if counts["AB"] != counts["BA"]:
        raise StageError("stats-ab-ba-balance", f"unbalanced orders: {counts}")
    if expected_pairs is not None and len(orders) != expected_pairs:
        raise StageError("stats-missing-pair", f"expected {expected_pairs} pairs, got {len(orders)}")
    return orders


def analyze(
    runs: Iterable[dict[str, Any]],
    orders: Any,
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Analyze paired outer medians with a deterministic percentile bootstrap.

    Each run record has pair_id, arm, status="admitted", and medians, a map
    from metric name to that fresh process's positive finite median.
    """
    checked_orders = validate_orders(orders)
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1000:
        raise StageError("stats-bootstrap", "bootstrap_samples must be an integer >= 1000")
    if not 0.0 < confidence < 1.0:
        raise StageError("stats-bootstrap", "confidence must be between zero and one")
    by_pair: dict[int, dict[str, dict[str, Any]]] = {i: {} for i in range(len(checked_orders))}
    for record in runs:
        if not isinstance(record, dict):
            raise StageError("stats-run-record", "run record is not an object")
        pair_id = record.get("pair_id")
        arm = record.get("arm")
        if type(pair_id) is not int:
            raise StageError("stats-paired-unit", f"pair_id must be int-not-bool, got {pair_id!r}")
        if pair_id not in by_pair or arm not in ARMS:
            raise StageError("stats-paired-unit", f"invalid pair/arm: {pair_id!r}/{arm!r}")
        if arm in by_pair[pair_id]:
            raise StageError("stats-duplicate-run", f"duplicate {arm} run for pair {pair_id}")
        if record.get("status") != "admitted":
            raise StageError("stats-failed-run", f"pair {pair_id} {arm} was not admitted")
        expected_index = checked_orders[pair_id]["order"].index(arm)
        if type(record.get("sequence_index")) is not int or record["sequence_index"] != expected_index:
            raise StageError("stats-paired-unit", f"pair {pair_id} {arm} sequence_index does not match recorded order")
        medians = record.get("medians")
        if not isinstance(medians, dict) or not medians:
            raise StageError("stats-numeric-input", f"pair {pair_id} {arm} has no medians")
        by_pair[pair_id][arm] = record
    for pair_id, pair in by_pair.items():
        if set(pair) != set(ARMS):
            missing = sorted(set(ARMS) - set(pair))
            raise StageError("stats-missing-run", f"pair {pair_id} missing {missing}")
    first_metrics = set(by_pair[0]["baseline"]["medians"])
    paired: dict[str, list[dict[str, float]]] = {name: [] for name in sorted(first_metrics)}
    for pair_id in range(len(checked_orders)):
        base_map = by_pair[pair_id]["baseline"]["medians"]
        cand_map = by_pair[pair_id]["candidate"]["medians"]
        if set(base_map) != first_metrics or set(cand_map) != first_metrics:
            raise StageError("stats-paired-unit", f"pair {pair_id} metric sets differ")
        for name in sorted(first_metrics):
            baseline = require_finite_positive(base_map[name], "stats-numeric-input", f"pair {pair_id} {name} baseline")
            candidate = require_finite_positive(cand_map[name], "stats-numeric-input", f"pair {pair_id} {name} candidate")
            ratio = candidate / baseline
            log_ratio = math.log(ratio)
            if not math.isfinite(ratio) or not math.isfinite(log_ratio):
                raise StageError("stats-numeric-input", f"pair {pair_id} {name} ratio is nonfinite")
            paired[name].append({"pair_id": pair_id, "baseline": baseline, "candidate": candidate, "log_ratio": log_ratio})
    rng = random.Random(seed)
    alpha = (1.0 - confidence) / 2.0
    metrics: dict[str, Any] = {}
    for name, rows in paired.items():
        n = len(rows)
        logs = [row["log_ratio"] for row in rows]
        boot: list[float] = []
        for _ in range(bootstrap_samples):
            indexes = [rng.randrange(n) for _ in range(n)]
            value = sum(logs[index] for index in indexes) / n
            if not math.isfinite(value):
                raise StageError("stats-bootstrap", f"nonfinite bootstrap mean for {name}")
            boot.append(value)
        boot.sort()
        low = percentile(boot, alpha)
        high = percentile(boot, 1.0 - alpha)
        baselines = [row["baseline"] for row in rows]
        candidates = [row["candidate"] for row in rows]
        mean_log = sum(logs) / n
        metrics[name] = {
            "outer_pairs": n,
            "decision_unit": "fresh-process outer median; inner samples are not independent observations",
            "baseline_outer_median": median(baselines),
            "candidate_outer_median": median(candidates),
            "baseline_min": min(baselines),
            "baseline_max": max(baselines),
            "candidate_min": min(candidates),
            "candidate_max": max(candidates),
            "mean_paired_log_ratio_candidate_over_baseline": mean_log,
            "geometric_mean_candidate_over_baseline": math.exp(mean_log),
            "paired_log_ratio_ci95": [low, high],
            "paired_ratio_ci95": [math.exp(low), math.exp(high)],
            "bootstrap": {"method": "paired case-resampling percentile CI over outer pairs", "samples": bootstrap_samples, "seed": seed, "confidence": confidence},
            "paired_observations": rows,
        }
    return {
        "schema": "r3-paired-stats-v1",
        "pair_count": len(checked_orders),
        "orders": checked_orders,
        "ab_count": sum(order["order"][0] == "baseline" for order in checked_orders),
        "ba_count": sum(order["order"][0] == "candidate" for order in checked_orders),
        "metrics": metrics,
        "limitation": "Descriptive protocol output only; this computation does not establish a production performance claim.",
    }


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path, help="JSON array of outer run records")
    parser.add_argument("--orders", required=True, type=Path, help="JSON array from the protocol driver")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        runs = json.loads(args.runs.read_text(encoding="utf-8"))
        orders = json.loads(args.orders.read_text(encoding="utf-8"))
        if not isinstance(runs, list):
            raise StageError("stats-run-record", "runs file must contain an array")
        result = analyze(runs, orders, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
        write_new_json(args.out, result)
    except (StageError, OSError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
