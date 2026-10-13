"""Unchanged-outcome checkpoint writes must be skipped, never rewritten.

A busy session schedules a debounced snapshot after every successful cell;
when the namespace did not change, that save re-pickled and rewrote the entire
payload (hundreds of MB on large states). These tests pin the skip contract:
the first save writes, an identical outcome skips with zero disk writes, any
real change writes, and the durable state is never left stale or corrupt.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rlm import repl
from rlm import snapshot as snapshot_store


class SkipUnchangedTestCase(unittest.TestCase):
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
            int(options.get("max_bytes", 64 << 20)),
            int(options.get("max_variable_bytes", 32 << 20)),
            bool(options.get("prune_oversized", False)),
            snapshot_format="legacy",
        )

    def cas_snapshot(self, namespace: dict[str, object], **options: object) -> dict[str, object]:
        return repl._snapshot_state(
            namespace,
            self.legacy_path,
            self.manifest_path,
            int(options.get("max_bytes", 64 << 20)),
            int(options.get("max_variable_bytes", 32 << 20)),
            bool(options.get("prune_oversized", False)),
            snapshot_format="cas-v2",
            cas_root=self.cas_root,
        )

    def restore(self, namespace: dict[str, object] | None = None) -> tuple[dict[str, object], dict[str, object]]:
        target = {} if namespace is None else namespace
        result = repl._restore_state(target, self.legacy_path, cas_root=self.cas_root)
        return target, result


class LegacySkipUnchangedTest(SkipUnchangedTestCase):
    def test_second_save_of_unchanged_outcome_skips_the_disk_write(self) -> None:
        namespace = {"records": {"left": [1, 2], "right": [3, 4]}, "flag": True}
        first = self.legacy_snapshot(namespace)
        self.assertNotIn("error", first)
        self.assertNotIn("skipped_unchanged", first)
        before = Path(self.legacy_path).read_bytes()
        before_stat = os.stat(self.legacy_path)
        before_manifest = Path(self.manifest_path).read_bytes()

        second = self.legacy_snapshot(namespace)
        self.assertNotIn("error", second)
        self.assertTrue(second["skipped_unchanged"])
        self.assertEqual(second["written_bytes"], 0)
        self.assertEqual(second["metrics"]["written_bytes"], 0)
        self.assertEqual(second["saved"], first["saved"])
        self.assertEqual(second["bytes"], first["bytes"])
        self.assertEqual(Path(self.legacy_path).read_bytes(), before)
        self.assertEqual(Path(self.manifest_path).read_bytes(), before_manifest)
        after_stat = os.stat(self.legacy_path)
        self.assertEqual((after_stat.st_size, after_stat.st_mtime_ns), (before_stat.st_size, before_stat.st_mtime_ns))
        self.assertFalse(list(Path(self.directory).glob("*.tmp")))

    def test_in_place_mutation_of_a_nested_list_writes_again(self) -> None:
        namespace = {"records": {"left": [1, 2], "right": [3, 4]}}
        self.legacy_snapshot(namespace)
        namespace["records"]["left"].append(3)  # same objects, mutated in place
        second = self.legacy_snapshot(namespace)
        self.assertNotIn("error", second)
        self.assertNotIn("skipped_unchanged", second)
        self.assertGreater(second["written_bytes"], 0)
        restored, result = self.restore()
        self.assertNotIn("error", result)
        self.assertEqual(restored["records"]["left"], [1, 2, 3])

    def test_reassigned_variable_and_new_variable_write_again(self) -> None:
        namespace = {"value": 1}
        self.legacy_snapshot(namespace)
        namespace["value"] = 2
        second = self.legacy_snapshot(namespace)
        self.assertNotIn("skipped_unchanged", second)
        namespace["value"] = 2
        namespace["extra"] = "new"
        third = self.legacy_snapshot(namespace)
        self.assertNotIn("skipped_unchanged", third)

    def test_restore_then_unchanged_save_skips(self) -> None:
        namespace = {"records": {"left": [1, 2]}, "flag": True}
        self.legacy_snapshot(namespace)
        restored, result = self.restore()
        self.assertNotIn("error", result)
        self.assertEqual(restored, namespace)
        after_restore = self.legacy_snapshot(restored)
        self.assertNotIn("error", after_restore)
        self.assertTrue(after_restore["skipped_unchanged"])
        self.assertEqual(after_restore["written_bytes"], 0)

    def test_restored_namespace_with_new_work_writes(self) -> None:
        namespace = {"records": {"left": [1, 2]}}
        self.legacy_snapshot(namespace)
        restored, _ = self.restore()
        restored["added"] = [9, 9, 9]
        second = self.legacy_snapshot(restored)
        self.assertNotIn("skipped_unchanged", second)
        final, result = self.restore()
        self.assertNotIn("error", result)
        self.assertEqual(final["added"], [9, 9, 9])

    def test_interrupted_save_keeps_last_valid_checkpoint_and_skippable_state(self) -> None:
        namespace = {"value": "committed"}
        self.legacy_snapshot(namespace)
        before = Path(self.legacy_path).read_bytes()
        real_replace = os.replace

        def fail_payload_replace(source, destination):
            if destination == self.legacy_path:
                raise OSError("synthetic interruption")
            return real_replace(source, destination)

        import unittest.mock as mock

        with mock.patch("os.replace", side_effect=fail_payload_replace):
            failed = self.legacy_snapshot({"value": "interrupted"})
        self.assertIn("synthetic interruption", failed["error"])
        self.assertEqual(Path(self.legacy_path).read_bytes(), before)
        self.assertFalse(list(Path(self.directory).glob("*.tmp")))
        # The last valid checkpoint is still skippable and still restorable.
        retried = self.legacy_snapshot({"value": "committed"})
        self.assertTrue(retried["skipped_unchanged"])
        restored, result = self.restore()
        self.assertNotIn("error", result)
        self.assertEqual(restored, {"value": "committed"})

    def test_deleted_or_touched_payload_forces_a_real_write(self) -> None:
        namespace = {"value": "kept"}
        self.legacy_snapshot(namespace)
        os.remove(self.legacy_path)
        second = self.legacy_snapshot(namespace)
        self.assertNotIn("skipped_unchanged", second)
        self.assertTrue(os.path.isfile(self.legacy_path))
        # Same bytes but a moved mtime: the identity check must refuse the skip.
        third = self.legacy_snapshot(namespace)
        self.assertTrue(third["skipped_unchanged"])
        future = time.time() + 60
        os.utime(self.legacy_path, (future, future))
        fourth = self.legacy_snapshot(namespace)
        self.assertNotIn("skipped_unchanged", fourth)

    def test_older_payload_returning_after_a_newer_save_still_writes(self) -> None:
        self.legacy_snapshot({"value": "a"})
        self.legacy_snapshot({"value": "b"})
        third = self.legacy_snapshot({"value": "a"})
        self.assertNotIn("skipped_unchanged", third)
        restored, _ = self.restore()
        self.assertEqual(restored["value"], "a")

    def test_pruning_save_skips_only_when_the_whole_outcome_repeats_and_still_prunes(self) -> None:
        namespace: dict[str, object] = {"small": 1, "big": b"x" * 4096}
        first = self.legacy_snapshot(namespace, max_variable_bytes=512, prune_oversized=True)
        self.assertNotIn("error", first)
        self.assertEqual(first["pruned"], ["big"])
        self.assertEqual(namespace, {"small": 1})
        namespace["big"] = b"x" * 4096  # re-added oversized value, same outcome
        second = self.legacy_snapshot(namespace, max_variable_bytes=512, prune_oversized=True)
        self.assertNotIn("error", second)
        self.assertTrue(second["skipped_unchanged"])
        self.assertEqual(second["pruned"], ["big"])
        self.assertEqual(namespace, {"small": 1})  # the prune side effect still ran

    def test_rapid_alternating_saves_never_corrupt_the_checkpoint(self) -> None:
        namespace: dict[str, object] = {"counter": 0}
        for step in range(6):
            namespace["counter"] = step
            changed = self.legacy_snapshot(namespace)
            self.assertNotIn("skipped_unchanged", changed)
            unchanged = self.legacy_snapshot(namespace)
            self.assertTrue(unchanged["skipped_unchanged"])
        restored, result = self.restore()
        self.assertNotIn("error", result)
        self.assertEqual(restored["counter"], 5)
        self.assertFalse(list(Path(self.directory).rglob("*.tmp")))


class CasSkipUnchangedTest(SkipUnchangedTestCase):
    def test_second_save_of_unchanged_outcome_skips_generation_and_pointer(self) -> None:
        first = self.cas_snapshot({"value": {"nested": [1, 2]}, "flag": True})
        self.assertNotIn("error", first)
        pointer_path = Path(self.cas_root, "CURRENT.json")
        pointer_before = pointer_path.read_bytes()
        pointer_stat = pointer_path.stat()
        generations_before = {path.name for path in Path(self.cas_root, "generations").glob("*.json")}

        second = self.cas_snapshot({"value": {"nested": [1, 2]}, "flag": True})
        self.assertNotIn("error", second)
        self.assertTrue(second["skipped_unchanged"])
        self.assertEqual(second["written_bytes"], 0)
        self.assertEqual(second["generation"], first["generation"])
        self.assertEqual(pointer_path.read_bytes(), pointer_before)
        pointer_after = pointer_path.stat()
        self.assertEqual((pointer_after.st_size, pointer_after.st_mtime_ns), (pointer_stat.st_size, pointer_stat.st_mtime_ns))
        self.assertEqual({path.name for path in Path(self.cas_root, "generations").glob("*.json")}, generations_before)
        self.assertEqual(second["metrics"]["snapshot_skipped_unchanged"], 1)
        self.assertEqual(second["metrics"]["written_bytes"], 0)

    def test_changed_value_writes_a_new_generation(self) -> None:
        first = self.cas_snapshot({"value": 1})
        second = self.cas_snapshot({"value": 2})
        self.assertNotIn("skipped_unchanged", second)
        self.assertNotEqual(second["generation"], first["generation"])
        self.assertGreater(second["written_bytes"], 0)
        restored, _ = self.restore()
        self.assertEqual(restored["value"], 2)

    def test_older_payload_returning_after_a_newer_save_writes_not_skips(self) -> None:
        self.cas_snapshot({"value": "a"})
        self.cas_snapshot({"value": "b"})
        third = self.cas_snapshot({"value": "a"})
        self.assertNotIn("skipped_unchanged", third)
        restored, _ = self.restore()
        self.assertEqual(restored["value"], "a")

    def test_restore_then_unchanged_save_skips(self) -> None:
        self.cas_snapshot({"value": {"nested": [1, 2]}})
        restored, result = self.restore()
        self.assertNotIn("error", result)
        second = self.cas_snapshot(restored)
        self.assertTrue(second["skipped_unchanged"])
        self.assertEqual(second["written_bytes"], 0)

    def test_missing_blob_never_falsely_skips(self) -> None:
        first = self.cas_snapshot({"value": "kept"})
        generation = first["generation"]
        entry = json.loads(Path(self.cas_root, "generations", f"{generation}.json").read_text())["entries"][0]
        blob = Path(self.cas_root, "blobs", f"{entry['sha256']}.blob")
        os.remove(blob)
        pointer_before = Path(self.cas_root, "CURRENT.json").read_bytes()
        second = self.cas_snapshot({"value": "kept"})
        # Damaged durable state must never be reported as a skipped save: the
        # write path's own completeness validation fails visibly instead (the
        # pre-existing behavior for a missing current-generation blob).
        self.assertNotIn("skipped_unchanged", second)
        self.assertIn("error", second)
        self.assertEqual(Path(self.cas_root, "CURRENT.json").read_bytes(), pointer_before)

    def test_prune_side_effect_still_runs_on_a_skipped_save(self) -> None:
        namespace: dict[str, object] = {"small": 1, "big": b"x" * 4096}
        first = self.cas_snapshot(namespace, max_variable_bytes=512, prune_oversized=True)
        self.assertEqual(first["pruned"], ["big"])
        namespace["big"] = b"x" * 4096
        second = self.cas_snapshot(namespace, max_variable_bytes=512, prune_oversized=True)
        self.assertTrue(second["skipped_unchanged"])
        self.assertEqual(second["pruned"], ["big"])
        self.assertEqual(namespace, {"small": 1})


class RepresentativeSizeSkipTest(SkipUnchangedTestCase):
    """A ~64 MiB namespace: the skip must eliminate the unchanged rewrite."""

    @staticmethod
    def representative_namespace() -> dict[str, object]:
        import random

        return {
            f"var_{index}": {"rows": [random.Random(index).randbytes(512 * 1024) for _ in range(8)]}
            for index in range(16)
        }

    def test_representative_legacy_state_skips_unchanged_saves(self) -> None:
        namespace = self.representative_namespace()
        first = self.legacy_snapshot(namespace)
        self.assertNotIn("error", first)
        self.assertGreater(first["bytes"], 60 << 20)
        self.assertGreater(first["written_bytes"], 60 << 20)

        second = self.legacy_snapshot(namespace)
        self.assertTrue(second["skipped_unchanged"])
        self.assertEqual(second["written_bytes"], 0)
        self.assertEqual(second["metrics"]["snapshot_skipped_unchanged"], 1)

        namespace["var_0"]["rows"][0] = b"mutated" * 512 * 1024  # one variable mutated in place
        third = self.legacy_snapshot(namespace)
        self.assertNotIn("skipped_unchanged", third)
        self.assertGreater(third["written_bytes"], 60 << 20)
        restored, result = self.restore()
        self.assertNotIn("error", result)
        self.assertEqual(restored["var_0"]["rows"][0], b"mutated" * 512 * 1024)

    def test_representative_cas_state_skips_unchanged_saves(self) -> None:
        namespace = self.representative_namespace()
        first = self.cas_snapshot(namespace)
        self.assertGreater(first["bytes"], 60 << 20)

        second = self.cas_snapshot(namespace)
        self.assertTrue(second["skipped_unchanged"])
        self.assertEqual(second["written_bytes"], 0)

        namespace["var_3"]["rows"][0] = b"mutated" * 512 * 1024
        third = self.cas_snapshot(namespace)
        self.assertNotIn("skipped_unchanged", third)
        # One changed variable: only that blob plus metadata are written again.
        self.assertLess(third["written_bytes"], 8 << 20)
        restored, result = self.restore()
        self.assertNotIn("error", result)
        self.assertEqual(restored["var_3"]["rows"][0], b"mutated" * 512 * 1024)


if __name__ == "__main__":
    unittest.main()
