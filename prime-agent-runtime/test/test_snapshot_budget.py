from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rlm import repl
from rlm import snapshot as snapshot_store


class SlowValue:
    """A value whose serialization is deliberately slow (budget test helper)."""

    def __init__(self, seconds: float, tag: str) -> None:
        self.seconds = seconds
        self.tag = tag

    def __reduce_ex__(self, protocol: object) -> object:
        time.sleep(self.seconds)
        return SlowValue, (self.seconds, self.tag)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SlowValue) and (self.seconds, self.tag) == (other.seconds, other.tag)


class BudgetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        self.legacy_path = os.path.join(self.directory, "kernel-state.dill")
        self.manifest_path = os.path.join(self.directory, "kernel-state.json")
        self.cas_root = os.path.join(self.directory, "kernel-state.v2")

    def snapshot(self, namespace, **options):
        return repl._snapshot_state(
            namespace,
            self.legacy_path,
            self.manifest_path,
            int(options.pop("max_bytes", 1 << 20)),
            int(options.pop("max_variable_bytes", 1 << 20)),
            bool(options.pop("prune_oversized", False)),
            snapshot_format=str(options.pop("snapshot_format", "auto")),
            cas_root=self.cas_root,
            budget_ms=options.pop("budget_ms", None),
        )

    def restore(self, *, source: str = "auto") -> tuple[dict, dict]:
        target: dict = {}
        result = repl._restore_state(target, self.legacy_path, cas_root=self.cas_root, source=source)
        return target, result

    def generation(self) -> dict:
        with open(os.path.join(self.cas_root, "CURRENT.json"), encoding="utf-8") as handle:
            pointer = json.load(handle)
        path = os.path.join(self.cas_root, "generations", f"{pointer['current']['generation']}.json")
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


class SnapshotFormatDefaultTest(BudgetTestCase):
    def test_auto_starts_fresh_sessions_on_cas_v2(self) -> None:
        result = self.snapshot({"value": 1})
        self.assertNotIn("error", result)
        self.assertEqual(result["format"], "cas-v2")
        self.assertTrue(os.path.isdir(self.cas_root))
        self.assertFalse(os.path.exists(self.legacy_path))
        restored, info = self.restore()
        self.assertEqual(restored, {"value": 1})
        self.assertEqual(info["format"], "cas-v2")

    def test_auto_continues_an_existing_legacy_session(self) -> None:
        first = self.snapshot({"value": 1}, snapshot_format="legacy")
        self.assertEqual(first["format"], "legacy")
        second = self.snapshot({"value": 2})
        self.assertNotIn("error", second)
        self.assertEqual(second["format"], "legacy")
        self.assertTrue(os.path.isfile(self.legacy_path))
        self.assertFalse(os.path.exists(self.cas_root))
        restored, _ = self.restore()
        self.assertEqual(restored, {"value": 2})

    def test_auto_continues_an_existing_cas_session(self) -> None:
        self.snapshot({"value": 1})
        second = self.snapshot({"value": 2})
        self.assertEqual(second["format"], "cas-v2")
        restored, _ = self.restore()
        self.assertEqual(restored, {"value": 2})

    def test_legacy_flag_is_the_rollback_for_fresh_sessions(self) -> None:
        result = self.snapshot({"value": 1}, snapshot_format="legacy")
        self.assertEqual(result["format"], "legacy")
        self.assertTrue(os.path.isfile(self.legacy_path))
        self.assertFalse(os.path.exists(self.cas_root))
        restored, _ = self.restore()
        self.assertEqual(restored, {"value": 1})

    def test_budget_ms_is_validated(self) -> None:
        for bad in (-1, True, "50"):
            result = self.snapshot({"value": 1}, budget_ms=bad)
            self.assertIn("error", result)


class PerVariableChangeDetectionTest(BudgetTestCase):
    def test_unchanged_immutable_value_reuses_its_blob(self) -> None:
        blob = b"payload-bytes" * 1024
        namespace: dict[str, object] = {"kept": blob}
        first = self.snapshot(namespace)
        self.assertNotIn("error", first)
        self.assertEqual(first["metrics"]["serialization_reused_names"], 0)
        second = self.snapshot(namespace)
        self.assertNotIn("error", second)
        self.assertEqual(second["metrics"]["serialization_reused_names"], 1)
        self.assertEqual(second["logical_bytes"], first["logical_bytes"])
        # The blob already exists on disk, so only the generation and pointer
        # metadata are rewritten (write dedupe; written/serialized << 1).
        self.assertLess(second["written_bytes"], second["logical_bytes"])
        restored, _ = self.restore()
        self.assertEqual(restored, {"kept": blob})

    def test_rebound_immutable_value_is_reserialized(self) -> None:
        self.snapshot({"kept": b"first-value" * 512})
        second = self.snapshot({"kept": b"second-value" * 512})
        self.assertNotIn("error", second)
        self.assertEqual(second["metrics"]["serialization_reused_names"], 0)
        restored, _ = self.restore()
        self.assertEqual(restored, {"kept": b"second-value" * 512})

    def test_in_place_mutable_mutation_is_never_served_from_cache(self) -> None:
        mutable = {"items": [1, 2, 3]}
        self.snapshot({"box": mutable})
        mutable["items"].append(4)
        second = self.snapshot({"box": mutable})
        self.assertNotIn("error", second)
        self.assertEqual(second["metrics"]["serialization_reused_names"], 0)
        restored, _ = self.restore()
        self.assertEqual(restored["box"]["items"], [1, 2, 3, 4])

    def test_deleted_name_evicts_its_cached_blob(self) -> None:
        namespace: dict[str, object] = {"gone": b"gone-value" * 512, "kept": 5}
        self.snapshot(namespace)
        del namespace["gone"]
        second = self.snapshot(namespace)
        self.assertNotIn("error", second)
        self.assertEqual(second["saved"], ["kept"])
        restored, _ = self.restore()
        self.assertEqual(restored, {"kept": 5})


class BudgetAwarePartialSnapshotTest(BudgetTestCase):
    def test_cas_partial_snapshot_lands_and_records_dropped_names(self) -> None:
        namespace: dict[str, object] = {
            "fast": "ok",
            "slow": SlowValue(0.3, "slow"),
            "tail_a": 1,
            "tail_b": 2,
        }
        result = self.snapshot(namespace, budget_ms=100)
        self.assertNotIn("error", result)
        self.assertEqual(result["saved"], ["fast", "slow"])
        self.assertEqual(result["dropped"], ["tail_a", "tail_b"])
        self.assertEqual(result["metrics"]["dropped_names_count"], 2)
        generation = self.generation()
        self.assertEqual(generation["dropped"], ["tail_a", "tail_b"])
        self.assertEqual(generation["savedNames"], ["fast", "slow"])
        restored, _ = self.restore()
        self.assertEqual(sorted(restored), ["fast", "slow"])

    def test_legacy_partial_snapshot_lands_and_records_dropped_names(self) -> None:
        namespace: dict[str, object] = {
            "fast": "ok",
            "slow": SlowValue(0.3, "slow"),
            "tail_a": 1,
            "tail_b": 2,
        }
        result = self.snapshot(namespace, snapshot_format="legacy", budget_ms=100)
        self.assertNotIn("error", result)
        self.assertEqual(result["format"], "legacy")
        self.assertEqual(result["saved"], ["fast", "slow"])
        self.assertEqual(result["dropped"], ["tail_a", "tail_b"])
        self.assertEqual(result["metrics"]["dropped_names_count"], 2)
        with open(self.manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["dropped"], ["tail_a", "tail_b"])
        restored, _ = self.restore()
        self.assertEqual(sorted(restored), ["fast", "slow"])

    def test_slow_names_are_ordered_last_so_fast_names_land_first(self) -> None:
        namespace: dict[str, object] = {
            "a_slow": SlowValue(0.3, "slow-a"),
            "z_slow": SlowValue(0.3, "slow-z"),
            "b_fast": 1,
            "c_fast": 2,
        }
        # First save records per-name durations (no budget: nothing is dropped).
        first = self.snapshot(namespace)
        self.assertNotIn("error", first)
        self.assertEqual(first["saved"], ["a_slow", "b_fast", "c_fast", "z_slow"])
        # Second save orders by recorded duration: fast names first. Once the
        # fast names are secured, neither 300 ms name fits the remaining
        # budget, so both are dropped without being attempted.
        second = self.snapshot(namespace, budget_ms=100)
        self.assertNotIn("error", second)
        self.assertEqual(second["saved"], ["b_fast", "c_fast"])
        self.assertEqual(second["dropped"], ["a_slow", "z_slow"])
        restored, _ = self.restore()
        self.assertEqual(sorted(restored), ["b_fast", "c_fast"])

    def test_predicted_overrun_is_skipped_once_one_name_is_secured(self) -> None:
        namespace: dict[str, object] = {
            "a_slow": SlowValue(0.3, "slow"),
            "b_fast": 1,
        }
        first = self.snapshot(namespace)
        self.assertNotIn("error", first)
        # After the first save the slow name's cost is known (300 ms); with a
        # 100 ms budget it cannot fit, so once "b_fast" is secured the slow
        # name is dropped without attempting it.
        second = self.snapshot(namespace, budget_ms=100)
        self.assertNotIn("error", second)
        self.assertEqual(second["saved"], ["b_fast"])
        self.assertEqual(second["dropped"], ["a_slow"])
        # A lone slow name is still attempted: progress beats an empty snapshot.
        lone: dict[str, object] = {"only": SlowValue(0.3, "lone")}
        third = self.snapshot(lone, budget_ms=100)
        self.assertNotIn("error", third)
        self.assertEqual(third["saved"], ["only"])
        self.assertEqual(third["dropped"], [])

    def test_exhausted_budget_still_commits_an_empty_generation(self) -> None:
        result = self.snapshot({"value": 1}, budget_ms=0)
        self.assertNotIn("error", result)
        self.assertEqual(result["saved"], [])
        self.assertEqual(result["dropped"], ["value"])
        self.assertEqual(result["metrics"]["dropped_names_count"], 1)
        restored, _ = self.restore()
        self.assertEqual(restored, {})

    def test_ordering_helper_ranks_unknown_names_at_the_median(self) -> None:
        history = snapshot_store.snapshot_history_for_root(
            os.path.join(self.directory, "unused-root")
        )
        history.durations_ms.update({"slow": 500.0, "quick_a": 1.0, "quick_b": 2.0})
        ordered = snapshot_store.order_names_by_serialization_cost(
            ["slow", "new_name", "quick_b", "quick_a"], history
        )
        # The median of (1.0, 2.0, 500.0) is 2.0, so the unknown name ties with
        # quick_b and the name breaks the tie.
        self.assertEqual(ordered, ["quick_a", "new_name", "quick_b", "slow"])


if __name__ == "__main__":
    unittest.main()
