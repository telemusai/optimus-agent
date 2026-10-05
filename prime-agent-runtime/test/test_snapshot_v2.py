from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import dill

from rlm import repl
from rlm.bash import BashHandle, BashResult
from rlm import snapshot as snapshot_store
from rlm import snapshot_serializer


class MutableBox:
    def __init__(self, value: object) -> None:
        self.value = value


def make_closure(items: list[int]):
    def total() -> int:
        return sum(items)

    return total


class CasSnapshotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        self.legacy_path = os.path.join(self.directory, "kernel-state.dill")
        self.manifest_path = os.path.join(self.directory, "kernel-state.json")
        self.cas_root = os.path.join(self.directory, "kernel-state.v2")

    def legacy_snapshot(self, namespace: dict[str, object], **options: object) -> dict[str, object]:
        return repl._snapshot_state(
            namespace,
            self.legacy_path,
            self.manifest_path,
            int(options.get("max_bytes", 1 << 20)),
            int(options.get("max_variable_bytes", 1 << 20)),
            bool(options.get("prune_oversized", False)),
            # Explicit legacy: `auto` now starts fresh sessions on CAS v2 (A5).
            snapshot_format="legacy",
        )

    def cas_snapshot(self, namespace: dict[str, object], **options: object) -> dict[str, object]:
        return repl._snapshot_state(
            namespace,
            self.legacy_path,
            self.manifest_path,
            int(options.get("max_bytes", 1 << 20)),
            int(options.get("max_variable_bytes", 1 << 20)),
            bool(options.get("prune_oversized", False)),
            snapshot_format=str(options.get("snapshot_format", "cas-v2")),
            cas_root=self.cas_root,
        )

    def restore(self, *, source: str = "auto", namespace: dict[str, object] | None = None) -> tuple[dict[str, object], dict[str, object]]:
        target = {} if namespace is None else namespace
        result = repl._restore_state(
            target,
            self.legacy_path,
            cas_root=self.cas_root,
            source=source,
        )
        return target, result

    def pointer(self) -> dict[str, object]:
        with open(os.path.join(self.cas_root, "CURRENT.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def generation(self, reference: dict[str, object] | None = None) -> dict[str, object]:
        selected = reference or self.pointer()["current"]
        assert isinstance(selected, dict)
        path = os.path.join(self.cas_root, "generations", f"{selected['generation']}.json")
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def blob_path(self, entry: dict[str, object]) -> str:
        return os.path.join(self.cas_root, "blobs", f"{entry['sha256']}.blob")


class SnapshotV2RoundTripTest(CasSnapshotTestCase):
    def test_nested_in_place_same_object_custom_and_primitive_changes_reach_current(self) -> None:
        nested = {"numbers": [1, 2], "inner": {"enabled": False}}
        box = MutableBox(["before"])
        namespace: dict[str, object] = {"count": 1, "nested": nested, "box": box}
        first = self.cas_snapshot(namespace)
        self.assertNotIn("error", first)

        nested["numbers"].append(3)
        nested["inner"]["enabled"] = True
        box.value.append("after")
        namespace["count"] = 2
        second = self.cas_snapshot(namespace, snapshot_format="auto")
        self.assertNotIn("error", second)
        self.assertNotEqual(first["generation"], second["generation"])

        current, restored = self.restore()
        self.assertEqual(restored["format"], "cas-v2")
        self.assertEqual(current["count"], 2)
        self.assertEqual(current["nested"], nested)
        self.assertEqual(current["box"].value, ["before", "after"])

        previous, rollback = self.restore(source="previous")
        self.assertTrue(rollback["rolled_back"])
        self.assertTrue(rollback["unsaved_work_possible"])
        self.assertEqual(previous["count"], 1)
        self.assertEqual(previous["nested"]["numbers"], [1, 2])
        self.assertEqual(previous["box"].value, ["before"])

    def test_cross_name_aliases_remain_independent_while_within_name_aliases_and_cycles_survive(self) -> None:
        shared: list[object] = [1]
        cycle: list[object] = []
        cycle.append(cycle)
        holder = {"left": shared, "right": shared, "cycle": cycle}
        result = self.cas_snapshot({"alias_a": shared, "alias_b": shared, "holder": holder})
        self.assertNotIn("error", result)

        restored, _ = self.restore()
        self.assertIsNot(restored["alias_a"], restored["alias_b"])
        restored_holder = restored["holder"]
        self.assertIs(restored_holder["left"], restored_holder["right"])
        self.assertIs(restored_holder["cycle"], restored_holder["cycle"][0])

        generation = self.generation()
        alias_entries = [entry for entry in generation["entries"] if entry["name"].startswith("alias_")]
        self.assertEqual(alias_entries[0]["sha256"], alias_entries[1]["sha256"])
        self.assertTrue(os.path.isfile(self.blob_path(alias_entries[0])))

    def test_closures_and_modules_are_reserialized_on_every_snapshot(self) -> None:
        captured = [1]
        closure = make_closure(captured)
        namespace = {"closure": closure, "math_module": math}
        seen = {"closure": 0, "module": 0}
        real_dump = snapshot_serializer._dump_with_dill

        def counting_dump(dill_module, value, stream, *args, **kwargs):
            if value is closure:
                seen["closure"] += 1
            if value is math:
                seen["module"] += 1
            return real_dump(dill_module, value, stream, *args, **kwargs)

        with mock.patch.object(snapshot_serializer, "_dump_with_dill", counting_dump):
            first = self.cas_snapshot(namespace)
            captured.append(2)
            second = self.cas_snapshot(namespace)
        self.assertNotIn("error", first)
        self.assertNotIn("error", second)
        self.assertEqual(seen, {"closure": 2, "module": 2})

        current, _ = self.restore()
        self.assertEqual(current["closure"](), 3)
        self.assertEqual(current["math_module"].sqrt(9), 3)

    def test_deletion_and_rebinding_are_generation_local(self) -> None:
        namespace: dict[str, object] = {"gone": "old", "rebound": [1]}
        self.cas_snapshot(namespace)
        del namespace["gone"]
        namespace["rebound"] = {"new": 2}
        self.cas_snapshot(namespace)

        current, _ = self.restore()
        self.assertNotIn("gone", current)
        self.assertEqual(current["rebound"], {"new": 2})
        previous, _ = self.restore(source="previous")
        self.assertEqual(previous, {"gone": "old", "rebound": [1]})

    def test_live_handles_are_skipped_and_completed_results_restore(self) -> None:
        live = object.__new__(BashHandle)
        complete = BashResult(exit_code=0, output="done", duration=0.25)
        result = self.cas_snapshot({"live": live, "nested_live": {"handle": live}, "complete": complete})
        skipped = {item["name"]: item["reason"] for item in result["skipped"]}
        self.assertIn("runtime-owned process handle", skipped["live"])
        self.assertIn("runtime-owned process handle", skipped["nested_live"])
        restored, _ = self.restore()
        self.assertEqual(restored["complete"], complete)
        self.assertNotIn("live", restored)
        self.assertNotIn("nested_live", restored)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed in candidate runtime")
    def test_numpy_array_and_view_in_place_mutations_restore_values_without_invented_cross_name_aliases(self) -> None:
        import numpy as np

        array = np.arange(8)
        view = array[::2]
        self.cas_snapshot({"array": array, "view": view})
        view += 100
        self.cas_snapshot({"array": array, "view": view})
        restored, _ = self.restore()
        np.testing.assert_array_equal(restored["array"], np.array([100, 1, 102, 3, 104, 5, 106, 7]))
        np.testing.assert_array_equal(restored["view"], np.array([100, 102, 104, 106]))
        self.assertFalse(np.shares_memory(restored["array"], restored["view"]))

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas not installed in candidate runtime")
    def test_dataframe_in_place_mutation_restores_latest_generation(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({"value": [1, 2]})
        self.cas_snapshot({"frame": frame})
        frame.loc[1, "value"] = 9
        self.cas_snapshot({"frame": frame})
        restored, _ = self.restore()
        pd.testing.assert_frame_equal(restored["frame"], pd.DataFrame({"value": [1, 9]}))


class SnapshotV2CapAndMetadataTest(CasSnapshotTestCase):
    def test_legacy_equivalent_outer_envelope_cap_retains_the_same_prefix(self) -> None:
        source = {"a": "first", "b": "second"}
        blobs = {name: dill.dumps(value) for name, value in source.items()}
        cap = len(dill.dumps(blobs)) - 1
        legacy = self.legacy_snapshot(source, max_bytes=cap, max_variable_bytes=cap)

        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        v2_path = os.path.join(other.name, "kernel-state.dill")
        v2_manifest = os.path.join(other.name, "kernel-state.json")
        v2_root = os.path.join(other.name, "kernel-state.v2")
        cas = repl._snapshot_state(
            source,
            v2_path,
            v2_manifest,
            cap,
            cap,
            False,
            snapshot_format="cas-v2",
            cas_root=v2_root,
        )
        self.assertEqual(cas["saved"], legacy["saved"])
        self.assertEqual(cas["skipped"], legacy["skipped"])
        self.assertEqual(cas["bytes"], legacy["bytes"])
        self.assertLessEqual(cas["bytes"], cap)

    def test_per_variable_prune_and_aggregate_skip_keep_baseline_policy(self) -> None:
        namespace: dict[str, object] = {"small": b"a" * 32, "big": b"b" * 4096, "tail": b"c" * 1100}
        result = self.cas_snapshot(
            namespace,
            max_bytes=1024,
            max_variable_bytes=2048,
            prune_oversized=True,
        )
        reasons = {item["name"]: item["reason"] for item in result["skipped"]}
        self.assertIn("per-variable", reasons["big"])
        self.assertIn("aggregate", reasons["tail"])
        self.assertEqual(result["pruned"], ["big"])
        self.assertNotIn("big", namespace)
        self.assertIn("tail", namespace)

    def test_zero_cap_fails_before_initializing_v2(self) -> None:
        result = self.cas_snapshot({}, max_bytes=0, max_variable_bytes=0)
        self.assertIn("exceeds aggregate snapshot size cap", result["error"])
        self.assertFalse(os.path.lexists(self.cas_root))

    def test_authoritative_generation_owns_all_result_metadata_and_current_pins_hash_and_size(self) -> None:
        namespace = {"saved": 1, "too_big": b"x" * 4096}
        result = self.cas_snapshot(namespace, max_variable_bytes=512, prune_oversized=True)
        pointer = self.pointer()
        current = pointer["current"]
        generation_path = os.path.join(self.cas_root, "generations", f"{current['generation']}.json")
        generation_bytes = Path(generation_path).read_bytes()
        generation = json.loads(generation_bytes)

        import hashlib

        self.assertEqual(current["sha256"], hashlib.sha256(generation_bytes).hexdigest())
        self.assertEqual(current["size"], len(generation_bytes))
        self.assertEqual(generation["savedNames"], result["saved"])
        self.assertEqual(generation["skipped"], result["skipped"])
        self.assertEqual(generation["pruned"], result["pruned"])
        self.assertEqual(generation["legacyEnvelopeBytes"], result["bytes"])
        self.assertEqual(generation["logicalSerializedBytes"], result["logical_bytes"])
        self.assertEqual(generation["maxBytes"], 1 << 20)
        self.assertEqual(generation["maxVariableBytes"], 512)

    def test_metrics_are_numeric_only_and_separate_logical_from_actual_writes(self) -> None:
        first = self.cas_snapshot({"unchanged": b"x" * 4096})
        second = self.cas_snapshot({"unchanged": b"x" * 4096})
        for result in (first, second):
            metrics = result["metrics"]
            self.assertTrue(all(value is None or isinstance(value, (int, float)) for value in metrics.values()))
            self.assertNotIn("name", metrics)
            self.assertNotIn("value", metrics)
            self.assertEqual(metrics["serialized_bytes"], result["logical_bytes"])
            self.assertEqual(metrics["written_bytes"], result["written_bytes"])
        self.assertEqual(first["logical_bytes"], second["logical_bytes"])
        self.assertLess(second["written_bytes"], first["written_bytes"])

    def test_varied_data_control_writes_changed_blobs_instead_of_implying_universal_dedup(self) -> None:
        rng = random.Random(20260910)
        first_values = {f"v{index}": rng.randbytes(4096) for index in range(16)}
        first = self.cas_snapshot(first_values, max_bytes=4 << 20, max_variable_bytes=1 << 20)
        second_values = {f"v{index}": rng.randbytes(4096) for index in range(16)}
        second = self.cas_snapshot(second_values, max_bytes=4 << 20, max_variable_bytes=1 << 20)
        self.assertNotEqual(first["generation"], second["generation"])
        self.assertGreaterEqual(second["written_bytes"], second["logical_bytes"])
        self.assertEqual(len(self.generation()["entries"]), len(second_values))


class SnapshotV2RecoveryTest(CasSnapshotTestCase):
    def setUp(self) -> None:
        super().setUp()
        legacy = self.legacy_snapshot({"legacy_value": "explicit-only"})
        self.assertNotIn("error", legacy)

    def test_empty_v2_root_is_visible_and_never_falls_back_to_legacy(self) -> None:
        os.mkdir(self.cas_root)
        namespace, result = self.restore()
        self.assertEqual(namespace, {})
        self.assertIn("FORMAT", result["error"])
        recovered, explicit = self.restore(source="legacy")
        self.assertEqual(recovered["legacy_value"], "explicit-only")
        self.assertTrue(explicit["legacy_recovery"])
        self.assertTrue(explicit["unsaved_work_possible"])

    def test_first_migration_interrupted_before_format_commit_requires_explicit_legacy_recovery(self) -> None:
        real_write = snapshot_store._atomic_write_bytes

        def interrupt_format(path, data, written):
            if os.path.basename(path) == "FORMAT":
                raise KeyboardInterrupt
            return real_write(path, data, written)

        with mock.patch.object(snapshot_store, "_atomic_write_bytes", interrupt_format):
            with self.assertRaises(KeyboardInterrupt):
                self.cas_snapshot({"new": 1})
        self.assertTrue(os.path.isdir(self.cas_root))
        _, automatic = self.restore()
        self.assertIn("FORMAT", automatic["error"])
        explicit, _ = self.restore(source="legacy")
        self.assertEqual(explicit["legacy_value"], "explicit-only")

    def test_first_migration_interrupted_after_format_before_current_is_visible(self) -> None:
        real_write = snapshot_store._atomic_write_bytes

        def interrupt_current(path, data, written):
            if os.path.basename(path) == "CURRENT.json":
                raise KeyboardInterrupt
            return real_write(path, data, written)

        with mock.patch.object(snapshot_store, "_atomic_write_bytes", interrupt_current):
            with self.assertRaises(KeyboardInterrupt):
                self.cas_snapshot({"new": 1})
        self.assertTrue(os.path.isfile(os.path.join(self.cas_root, "FORMAT")))
        self.assertFalse(os.path.lexists(os.path.join(self.cas_root, "CURRENT.json")))
        _, result = self.restore()
        self.assertIn("CURRENT.json is missing", result["error"])

    def test_present_current_with_missing_format_is_corrupt_not_legacy(self) -> None:
        self.cas_snapshot({"new": 1})
        os.remove(os.path.join(self.cas_root, "FORMAT"))
        namespace, result = self.restore()
        self.assertEqual(namespace, {})
        self.assertIn("FORMAT", result["error"])

    def test_sigint_at_current_commit_is_parked_prunes_once_and_does_not_bleed(self) -> None:
        real_write = snapshot_store._atomic_write_bytes
        namespace: dict[str, object] = {"small": 1, "big": b"x" * 4096}

        def signal_after_current(path, data, written):
            result = real_write(path, data, written)
            if os.path.basename(path) == "CURRENT.json":
                signal.raise_signal(signal.SIGINT)
            return result

        previous_handler = signal.getsignal(signal.SIGINT)
        with mock.patch.object(snapshot_store, "_atomic_write_bytes", signal_after_current):
            result = self.cas_snapshot(namespace, max_variable_bytes=512, prune_oversized=True)
        self.assertNotIn("error", result)
        self.assertEqual(result["pruned"], ["big"])
        self.assertEqual(namespace, {"small": 1})
        self.assertIs(signal.getsignal(signal.SIGINT), previous_handler)

    def test_disk_full_before_current_leaves_prior_generation_and_pruned_namespace_intact(self) -> None:
        first = self.cas_snapshot({"value": "old"})
        pointer_before = Path(self.cas_root, "CURRENT.json").read_bytes()
        namespace: dict[str, object] = {"value": "new", "big": b"x" * 4096}
        real_write = snapshot_store._atomic_write_bytes

        def fail_generation(path, data, written):
            if os.path.basename(os.path.dirname(path)) == "generations":
                raise OSError("disk full")
            return real_write(path, data, written)

        with mock.patch.object(snapshot_store, "_atomic_write_bytes", fail_generation):
            failed = self.cas_snapshot(namespace, max_variable_bytes=512, prune_oversized=True)
        self.assertIn("disk full", failed["error"])
        self.assertEqual(Path(self.cas_root, "CURRENT.json").read_bytes(), pointer_before)
        self.assertIn("big", namespace)
        restored, _ = self.restore()
        self.assertEqual(restored["value"], "old")
        self.assertEqual(first["generation"], self.pointer()["current"]["generation"])

    def test_corrupt_current_manifest_fails_without_automatic_previous_rollback(self) -> None:
        self.cas_snapshot({"value": "old"})
        self.cas_snapshot({"value": "new"})
        current = self.pointer()["current"]
        path = os.path.join(self.cas_root, "generations", f"{current['generation']}.json")
        with open(path, "ab") as handle:
            handle.write(b"corrupt")
        namespace, result = self.restore()
        self.assertEqual(namespace, {})
        self.assertIn("mismatch", result["error"])
        previous, rollback = self.restore(source="previous")
        self.assertEqual(previous["value"], "old")
        self.assertTrue(rollback["rolled_back"])

    def test_missing_and_corrupt_blobs_are_visible(self) -> None:
        for mode in ("missing", "corrupt"):
            with self.subTest(mode=mode):
                directory = tempfile.TemporaryDirectory()
                self.addCleanup(directory.cleanup)
                legacy_path = os.path.join(directory.name, "kernel-state.dill")
                manifest_path = os.path.join(directory.name, "kernel-state.json")
                cas_root = os.path.join(directory.name, "kernel-state.v2")
                repl._snapshot_state(
                    {"value": mode},
                    legacy_path,
                    manifest_path,
                    1 << 20,
                    1 << 20,
                    False,
                    snapshot_format="cas-v2",
                    cas_root=cas_root,
                )
                with open(os.path.join(cas_root, "CURRENT.json"), encoding="utf-8") as handle:
                    pointer = json.load(handle)
                generation_path = os.path.join(cas_root, "generations", f"{pointer['current']['generation']}.json")
                with open(generation_path, encoding="utf-8") as handle:
                    generation = json.load(handle)
                blob_path = os.path.join(cas_root, "blobs", f"{generation['entries'][0]['sha256']}.blob")
                if mode == "missing":
                    os.remove(blob_path)
                else:
                    Path(blob_path).write_bytes(b"wrong")
                target: dict[str, object] = {}
                result = repl._restore_state(target, legacy_path, cas_root=cas_root)
                self.assertIn("CAS v2 load failed", result["error"])
                self.assertEqual(target, {})

    def test_current_reference_size_hash_and_generation_identity_are_validated(self) -> None:
        self.cas_snapshot({"value": 1})
        current_path = Path(self.cas_root, "CURRENT.json")
        valid_pointer = current_path.read_bytes()
        for field, replacement in (("size", 1), ("sha256", "0" * 64), ("generation", "f" * 32)):
            with self.subTest(field=field):
                pointer = json.loads(valid_pointer)
                pointer["current"][field] = replacement
                current_path.write_text(json.dumps(pointer), encoding="utf-8")
                _, result = self.restore()
                self.assertIn("CAS v2 load failed", result["error"])
                current_path.write_bytes(valid_pointer)

    def test_valid_hash_manifest_over_restore_caps_fails_before_blob_allocation(self) -> None:
        import hashlib

        self.cas_snapshot({"value": "small"}, max_bytes=1024, max_variable_bytes=512)
        pointer_path = Path(self.cas_root, "CURRENT.json")
        pointer = self.pointer()
        current = pointer["current"]
        generation_path = Path(
            self.cas_root,
            "generations",
            f"{current['generation']}.json",
        )
        generation = json.loads(generation_path.read_bytes())
        generation["entries"][0]["size"] = 513
        generation["logicalSerializedBytes"] = 513
        generation["legacyEnvelopeBytes"] = 600
        generation_bytes = snapshot_store._encode_json(generation)
        generation_path.write_bytes(generation_bytes)
        current["size"] = len(generation_bytes)
        current["sha256"] = hashlib.sha256(generation_bytes).hexdigest()
        pointer_path.write_bytes(snapshot_store._encode_json(pointer))

        target: dict[str, object] = {}
        result = repl._restore_state(
            target,
            self.legacy_path,
            cas_root=self.cas_root,
            max_bytes=1024,
            max_variable_bytes=512,
        )
        self.assertIn("per-variable restore cap", result["error"])
        self.assertEqual(target, {})

    def test_valid_hash_manifest_over_aggregate_restore_cap_fails_without_large_fixture(self) -> None:
        import hashlib

        self.cas_snapshot({"value": "small"}, max_bytes=1024, max_variable_bytes=512)
        pointer_path = Path(self.cas_root, "CURRENT.json")
        pointer = self.pointer()
        current = pointer["current"]
        generation_path = Path(self.cas_root, "generations", f"{current['generation']}.json")
        generation = json.loads(generation_path.read_bytes())
        generation["legacyEnvelopeBytes"] = 1025
        generation_bytes = snapshot_store._encode_json(generation)
        generation_path.write_bytes(generation_bytes)
        current["size"] = len(generation_bytes)
        current["sha256"] = hashlib.sha256(generation_bytes).hexdigest()
        pointer_path.write_bytes(snapshot_store._encode_json(pointer))

        target: dict[str, object] = {}
        result = repl._restore_state(
            target,
            self.legacy_path,
            cas_root=self.cas_root,
            max_bytes=1024,
            max_variable_bytes=512,
        )
        self.assertIn("aggregate restore cap", result["error"])
        self.assertEqual(target, {})

    def test_valid_generation_requires_reader_caps_large_enough_for_committed_data(self) -> None:
        saved = self.cas_snapshot({"value": "small"}, max_bytes=1024, max_variable_bytes=512)
        generation = self.generation()
        entry_size = generation["entries"][0]["size"]

        for limits, expected in (
            ({"max_bytes": 1024, "max_variable_bytes": entry_size - 1}, "per-variable restore cap"),
            ({"max_bytes": saved["bytes"] - 1, "max_variable_bytes": 512}, "aggregate restore cap"),
        ):
            with self.subTest(expected=expected):
                target: dict[str, object] = {}
                result = repl._restore_state(
                    target,
                    self.legacy_path,
                    cas_root=self.cas_root,
                    **limits,
                )
                self.assertIn(expected, result["error"])
                self.assertEqual(target, {})

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_generation_directory_reparse_escape_is_rejected(self) -> None:
        self.cas_snapshot({"value": "safe"})
        generations = Path(self.cas_root, "generations")
        outside = Path(self.directory, "outside-generations")
        generations.rename(outside)
        try:
            os.symlink(outside, generations, target_is_directory=True)
        except OSError as error:
            outside.rename(generations)
            self.skipTest(f"directory symlink creation unavailable: {error}")
        target: dict[str, object] = {}
        result = repl._restore_state(target, self.legacy_path, cas_root=self.cas_root)
        self.assertIn("plain directory", result["error"])
        self.assertEqual(target, {})

    def test_explicit_legacy_writer_refuses_existing_v2_and_auto_continues_it(self) -> None:
        first = self.cas_snapshot({"value": 1})
        refused = self.cas_snapshot({"value": 2}, snapshot_format="legacy")
        self.assertIn("refusing", refused["error"])
        continued = self.cas_snapshot({"value": 3}, snapshot_format="auto")
        self.assertNotIn("error", continued)
        self.assertNotEqual(first["generation"], continued["generation"])
        restored, _ = self.restore()
        self.assertEqual(restored["value"], 3)

    def test_older_host_omitting_cas_root_still_detects_and_continues_v2(self) -> None:
        first = self.cas_snapshot({"value": "v2"})
        target: dict[str, object] = {}
        restored = repl._restore_state(target, self.legacy_path)
        self.assertEqual(restored["generation"], first["generation"])
        self.assertEqual(target["value"], "v2")

        continued = repl._snapshot_state(
            {"value": "continued"},
            self.legacy_path,
            self.manifest_path,
            1 << 20,
            1 << 20,
            False,
        )
        self.assertEqual(continued["format"], "cas-v2")
        self.assertNotEqual(continued["generation"], first["generation"])

    def test_explicit_export_makes_current_generation_readable_by_legacy_reader(self) -> None:
        snapshot = self.cas_snapshot({"value": {"nested": [1, 2, 3]}})
        exported = repl._export_legacy_state(
            self.legacy_path,
            self.manifest_path,
            self.cas_root,
            "current",
        )
        self.assertEqual(exported["source_generation"], snapshot["generation"])
        target: dict[str, object] = {}
        restored = repl._restore_state(target, self.legacy_path, source="legacy")
        self.assertEqual(restored["format"], "legacy")
        self.assertTrue(restored["legacy_recovery"])
        self.assertTrue(restored["unsaved_work_possible"])
        self.assertEqual(target["value"], {"nested": [1, 2, 3]})

    def test_gc_retains_current_previous_and_active_restore_blobs(self) -> None:
        self.cas_snapshot({"value": "one"})
        first_generation = self.generation()
        first_blob = self.blob_path(first_generation["entries"][0])
        with snapshot_store.open_cas_generation(self.cas_root, "current"):
            self.cas_snapshot({"value": "two"})
            self.cas_snapshot({"value": "three"})
            self.assertTrue(os.path.isfile(first_blob))
        self.cas_snapshot({"value": "four"})
        self.assertFalse(os.path.lexists(first_blob))
        pointer = self.pointer()
        retained = {pointer["current"]["generation"], pointer["previous"]["generation"]}
        generation_files = {Path(path).stem for path in Path(self.cas_root, "generations").glob("*.json")}
        self.assertEqual(generation_files, retained)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_blob_reparse_escape_is_rejected_without_touching_target(self) -> None:
        self.cas_snapshot({"value": "safe"})
        generation = self.generation()
        blob = self.blob_path(generation["entries"][0])
        outside = Path(self.directory, "outside.bin")
        outside.write_bytes(b"outside")
        os.remove(blob)
        try:
            os.symlink(outside, blob)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        _, result = self.restore()
        self.assertIn("plain file", result["error"])
        self.assertEqual(outside.read_bytes(), b"outside")


if __name__ == "__main__":
    unittest.main()
