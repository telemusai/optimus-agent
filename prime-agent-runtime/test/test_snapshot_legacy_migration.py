"""One-way legacy -> CAS v2 migration of existing kernel sessions.

An existing legacy session kept rewriting its full payload on every
debounced save. The next automatic save now migrates it onto CAS v2 with an
atomic directory flip: the legacy pair is never touched, a crash leaves
either the untouched legacy state or a fully committed v2 root, and older
binaries (or the explicit legacy rollback flag) keep a readable fallback.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

import dill

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rlm import repl
from rlm import snapshot as snapshot_store


class LegacyMigrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        self.legacy_path = os.path.join(self.directory, "kernel-state.dill")
        self.manifest_path = os.path.join(self.directory, "kernel-state.json")
        self.cas_root = os.path.join(self.directory, "kernel-state.v2")

    def snapshot(self, namespace: dict[str, object], **options: object) -> dict[str, object]:
        return repl._snapshot_state(
            namespace,
            self.legacy_path,
            self.manifest_path,
            int(options.get("max_bytes", 1 << 20)),
            int(options.get("max_variable_bytes", 1 << 20)),
            bool(options.get("prune_oversized", False)),
            snapshot_format=str(options.get("snapshot_format", "auto")),
            cas_root=self.cas_root,
            budget_ms=options.get("budget_ms", None) if "budget_ms" in options else None,
        )

    def restore(self, namespace: dict[str, object] | None = None) -> tuple[dict[str, object], dict[str, object]]:
        target = {} if namespace is None else namespace
        result = repl._restore_state(target, self.legacy_path, cas_root=self.cas_root)
        return target, result


class LegacyMigrationTest(LegacyMigrationTestCase):
    def test_next_automatic_save_migrates_and_keeps_the_legacy_pair(self) -> None:
        first = self.snapshot({"value": 1, "nested": {"rows": [1, 2]}}, snapshot_format="legacy")
        self.assertEqual(first["format"], "legacy")
        legacy_before = Path(self.legacy_path).read_bytes()
        manifest_before = Path(self.manifest_path).read_bytes()

        migrated = self.snapshot({"value": 2, "nested": {"rows": [1, 2]}})
        self.assertNotIn("error", migrated)
        self.assertEqual(migrated["format"], "cas-v2")
        self.assertTrue(migrated["migrated_from_legacy"])
        self.assertGreater(migrated["written_bytes"], 0)
        # The legacy pair is preserved byte-for-byte for older readers.
        self.assertEqual(Path(self.legacy_path).read_bytes(), legacy_before)
        self.assertEqual(Path(self.manifest_path).read_bytes(), manifest_before)
        # The flipped root is a complete, valid CAS v2 state.
        self.assertEqual(
            Path(self.cas_root, "FORMAT").read_bytes(),
            snapshot_store.CAS_FORMAT_MARKER,
        )
        pointer = json.loads(Path(self.cas_root, "CURRENT.json").read_text())
        self.assertEqual(pointer["current"]["generation"], migrated["generation"])
        self.assertIsNone(pointer["previous"])
        restored, info = self.restore()
        self.assertEqual(restored, {"value": 2, "nested": {"rows": [1, 2]}})
        self.assertEqual(info["format"], "cas-v2")
        self.assertFalse(list(Path(self.directory).glob("*.migrating-*")))

    def test_unchanged_save_after_migration_skips(self) -> None:
        self.snapshot({"value": 1}, snapshot_format="legacy")
        migrated = self.snapshot({"value": 2})
        self.assertTrue(migrated["migrated_from_legacy"])
        second = self.snapshot({"value": 2})
        self.assertNotIn("migrated_from_legacy", second)
        self.assertTrue(second["skipped_unchanged"])
        self.assertEqual(second["generation"], migrated["generation"])

    def test_changed_save_after_migration_writes_only_changed_blobs(self) -> None:
        self.snapshot({"kept": b"stable" * 512, "moving": b"first" * 512}, snapshot_format="legacy")
        migrated = self.snapshot({"kept": b"stable" * 512, "moving": b"first" * 512})
        self.assertTrue(migrated["migrated_from_legacy"])
        changed = self.snapshot({"kept": b"stable" * 512, "moving": b"second" * 512})
        self.assertNotIn("skipped_unchanged", changed)
        self.assertNotIn("migrated_from_legacy", changed)
        self.assertLess(changed["written_bytes"], changed["logical_bytes"])
        restored, _ = self.restore()
        self.assertEqual(restored["moving"], b"second" * 512)
        self.assertEqual(restored["kept"], b"stable" * 512)

    def test_explicit_legacy_flag_still_writes_legacy_after_legacy_saves(self) -> None:
        first = self.snapshot({"value": 1}, snapshot_format="legacy")
        self.assertEqual(first["format"], "legacy")
        second = self.snapshot({"value": 2}, snapshot_format="legacy")
        self.assertEqual(second["format"], "legacy")
        self.assertFalse(os.path.exists(self.cas_root))
        restored, _ = self.restore()
        self.assertEqual(restored["value"], 2)

    def test_failed_flip_falls_back_to_the_legacy_writer(self) -> None:
        self.snapshot({"value": 1}, snapshot_format="legacy")
        real_rename = os.rename

        def fail_flip(source: str, destination: str) -> None:
            if destination == self.cas_root:
                raise OSError("synthetic flip failure")
            return real_rename(source, destination)

        with mock.patch("os.rename", side_effect=fail_flip):
            fallback = self.snapshot({"value": 2})
        self.assertNotIn("error", fallback)
        self.assertEqual(fallback["format"], "legacy")
        self.assertNotIn("skipped_unchanged", fallback)
        self.assertFalse(os.path.exists(self.cas_root))
        self.assertFalse(list(Path(self.directory).glob("*.migrating-*")))
        restored, _ = self.restore()
        self.assertEqual(restored["value"], 2)
        # The migration retries and succeeds on the next automatic save.
        migrated = self.snapshot({"value": 3})
        self.assertTrue(migrated["migrated_from_legacy"])
        restored, _ = self.restore()
        self.assertEqual(restored["value"], 3)

    def test_interrupted_migration_leaves_legacy_state_intact(self) -> None:
        self.snapshot({"value": "committed"}, snapshot_format="legacy")
        legacy_before = Path(self.legacy_path).read_bytes()
        real_write = snapshot_store._atomic_write_bytes

        def interrupt_current(path: str, data: bytes, written: list[int]) -> None:
            if os.path.basename(path) == "CURRENT.json":
                raise KeyboardInterrupt
            return real_write(path, data, written)

        with mock.patch.object(snapshot_store, "_atomic_write_bytes", interrupt_current):
            with self.assertRaises(KeyboardInterrupt):
                self.snapshot({"value": "interrupted"})
        # Neither the legacy pair nor the live root changed; the temp root was
        # cleaned up, so auto-restore still sees the intact legacy session.
        self.assertEqual(Path(self.legacy_path).read_bytes(), legacy_before)
        self.assertFalse(os.path.exists(self.cas_root))
        self.assertFalse(list(Path(self.directory).glob("*.migrating-*")))
        restored, info = self.restore()
        self.assertEqual(restored["value"], "committed")
        self.assertEqual(info["format"], "legacy")

    def test_stale_abandoned_migration_temp_roots_are_cleaned_up(self) -> None:
        self.snapshot({"value": 1}, snapshot_format="legacy")
        stale = Path(self.directory, "kernel-state.v2.migrating-stale")
        stale.mkdir()
        (stale / "blobs").mkdir()
        fresh = Path(self.directory, "kernel-state.v2.migrating-fresh")
        fresh.mkdir()
        past = 0.0
        os.utime(stale, (past, past))

        migrated = self.snapshot({"value": 2})
        self.assertTrue(migrated["migrated_from_legacy"])
        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())  # recent temp roots are never touched

    def test_fresher_legacy_payload_written_by_an_older_binary_wins_auto_restore(self) -> None:
        self.snapshot({"value": "legacy"}, snapshot_format="legacy")
        self.snapshot({"value": "migrated"})
        self.assertTrue(os.path.isdir(self.cas_root))
        # Simulate an older binary without CAS support writing the legacy pair
        # after the flip (the migration never rewrites it, so do it directly).
        with open(self.legacy_path, "wb") as handle:
            dill.dump({"value": dill.dumps("downgraded-work")}, handle)
        import time

        # The older binary wrote the pair one second after the flip.
        marker = time.time() + 1
        os.utime(self.legacy_path, (marker, marker))

        restored, info = self.restore()
        self.assertEqual(restored, {"value": "downgraded-work"})
        self.assertEqual(info["format"], "legacy")
        self.assertTrue(info["legacy_recovery"])
        self.assertTrue(info["unsaved_work_possible"])
        # The next automatic save re-merges that work into a new generation,
        # which then becomes the newest durable state.
        time.sleep(1.1)
        merged = self.snapshot(restored)
        self.assertEqual(merged["format"], "cas-v2")
        final, final_info = self.restore()
        self.assertEqual(final, {"value": "downgraded-work"})
        self.assertEqual(final_info["format"], "cas-v2")

    def test_pointer_not_newer_than_a_fresh_export_keeps_cas_restore(self) -> None:
        self.snapshot({"value": "v2"})
        self.snapshot({"value": "v2"})  # unchanged: skipped, pointer untouched
        # An explicit export refreshes the legacy pair; it is content-equal to
        # the current generation, and the export is not newer than the pointer.
        exported = repl._export_legacy_state(
            self.legacy_path, self.manifest_path, self.cas_root, "current"
        )
        self.assertNotIn("error", exported)
        os.utime(self.legacy_path, (0.0, 0.0))  # make the export older
        restored, info = self.restore()
        self.assertEqual(restored, {"value": "v2"})
        self.assertEqual(info["format"], "cas-v2")


if __name__ == "__main__":
    unittest.main()
