from __future__ import annotations

import copyreg
import datetime
import decimal
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dill
from rlm import repl
from rlm import snapshot_serializer
from rlm.snapshot_serializer import dump_snapshot_value


class CustomList(list):
    reductions = 0

    def __reduce_ex__(self, protocol):
        type(self).reductions += 1
        return (type(self), (list(self),))


class SnapshotSpeedTests(unittest.TestCase):
    def serialize(self, value, cap=1 << 20):
        stream = io.BytesIO()
        dump_snapshot_value(dill, value, stream, repl._CappedWriter(stream, cap))
        return stream.getvalue()

    def test_builtin_graph_uses_native_serializer_and_keeps_aliases_cycles(self):
        shared = [None, True, 1, 1.5, "text", b"bytes"]
        cycle = []
        cycle.append(cycle)
        value = {"a": shared, "b": shared, "cycle": cycle, "set": {1, 2}, "frozen": frozenset({3}), "tuple": (4, 5)}
        with mock.patch.object(snapshot_serializer, "_dump_with_dill", side_effect=AssertionError("slow serializer used")):
            blob = self.serialize(value)
        restored = dill.loads(blob)
        self.assertIs(restored["a"], restored["b"])
        self.assertIs(restored["cycle"], restored["cycle"][0])
        self.assertEqual(restored["a"], shared)
        shared.append("changed")
        self.assertEqual(dill.loads(self.serialize(value))["a"][-1], "changed")

    def test_custom_reducer_runs_once_and_partial_buffer_is_reset(self):
        CustomList.reductions = 0
        value = ["prefix" * 20000, CustomList([1, 2])]
        restored = dill.loads(self.serialize(value))
        self.assertIsInstance(restored[1], CustomList)
        self.assertEqual(restored, value)
        self.assertEqual(CustomList.reductions, 1)

    def test_dynamic_closure_and_class_keep_dill_semantics(self):
        captured = [4]
        class LocalClass:
            def total(self):
                return sum(captured)
        value = {"object": LocalClass(), "function": lambda: sum(captured)}
        captured.append(5)
        restored = dill.loads(self.serialize(value))
        self.assertEqual(restored["object"].total(), 9)
        self.assertEqual(restored["function"](), 9)

    def test_native_and_fallback_paths_enforce_variable_cap(self):
        for value in (["x" * 10000], [CustomList([1]), "x" * 10000]):
            with self.assertRaises(repl._SnapshotSizeLimitExceeded):
                self.serialize(value, cap=100)

    def test_builtin_reducer_values_fall_back_without_changing_their_types(self):
        for value in (bytearray(b"mutable"), complex(1, 2), range(5)):
            restored = dill.loads(self.serialize(value))
            self.assertIs(type(restored), type(value))
            self.assertEqual(restored, value)

    def test_dates_decimals_and_nested_aliases_use_native_serializer(self):
        offset = datetime.timezone(datetime.timedelta(hours=10), "AEST")
        shared = [datetime.datetime(2026, 9, 17, 12, 30, tzinfo=offset, fold=1),
                  datetime.date(2026, 9, 17), datetime.time(12, 30, tzinfo=offset, fold=1),
                  datetime.timedelta(microseconds=-1), offset,
                  decimal.Decimal("12345678901234567890.000000001")]
        value = {"a": shared, "b": shared, "self": None}
        value["self"] = value
        with mock.patch.object(snapshot_serializer, "_dump_with_dill", side_effect=AssertionError("slow serializer used")):
            restored = dill.loads(self.serialize(value))
        self.assertEqual(restored["a"], shared)
        self.assertIs(restored["a"], restored["b"])
        self.assertIs(restored, restored["self"])
        self.assertEqual(restored["a"][0].fold, 1)
        self.assertEqual(restored["a"][2].fold, 1)
        shared.append(decimal.Decimal("9.5"))
        self.assertEqual(dill.loads(self.serialize(value))["a"][-1], shared[-1])

    def test_custom_date_subclass_and_timezone_keep_dill_semantics(self):
        class CustomDate(datetime.date):
            pass
        class CustomZone(datetime.tzinfo):
            def utcoffset(self, dt):
                return datetime.timedelta(hours=7)
            def dst(self, dt):
                return datetime.timedelta(0)
        for value in [CustomDate(2026, 9, 17), datetime.datetime(2026, 9, 17, tzinfo=CustomZone())]:
            with mock.patch.object(snapshot_serializer, "_dump_with_dill", wraps=snapshot_serializer._dump_with_dill) as fallback:
                restored = dill.loads(self.serialize(value))
            self.assertEqual(restored, value)
            fallback.assert_called_once()

    def test_registered_date_reducers_are_not_bypassed(self):
        def reduction(value):
            return datetime.date, (2000, 1, 1)
        with mock.patch.dict(copyreg.dispatch_table, {datetime.date: reduction}):
            with mock.patch.object(snapshot_serializer, "_dump_with_dill", wraps=snapshot_serializer._dump_with_dill) as fallback:
                restored = dill.loads(self.serialize(datetime.date(2026, 9, 17)))
            self.assertEqual(restored, datetime.date(2000, 1, 1))
            fallback.assert_called_once()
        def dill_reduction(pickler, value):
            pickler.save_reduce(datetime.date, (2001, 1, 1), obj=value)
        with mock.patch.dict(dill.Pickler.dispatch, {datetime.date: dill_reduction}):
            with mock.patch.object(snapshot_serializer, "_dump_with_dill", wraps=snapshot_serializer._dump_with_dill) as fallback:
                restored = dill.loads(self.serialize(datetime.date(2026, 9, 17)))
            self.assertEqual(restored, datetime.date(2001, 1, 1))
            fallback.assert_called_once()

    def test_mixed_date_custom_reducer_runs_only_once(self):
        CustomList.reductions = 0
        value = [datetime.date(2026, 9, 17), CustomList([1]), decimal.Decimal("1.234")]
        restored = dill.loads(self.serialize(value))
        self.assertEqual(restored, value)
        self.assertEqual(CustomList.reductions, 1)

    def test_mixed_values_keep_caps_and_legacy_cas_latest_state(self):
        with self.assertRaises(repl._SnapshotSizeLimitExceeded):
            self.serialize([datetime.date(2026, 9, 17), "x" * 10000], cap=100)
        for snapshot_format in ("legacy", "cas-v2"):
            with tempfile.TemporaryDirectory() as root:
                path, manifest = os.path.join(root, "state.dill"), os.path.join(root, "state.json")
                shared = [datetime.date(2026, 9, 17), decimal.Decimal("1.1")]
                namespace = {"records": {"left": shared, "right": shared}}
                for amount in ("2.2", "3.3"):
                    shared[1] = decimal.Decimal(amount)
                    result = repl._snapshot_state(namespace, path, manifest, 1 << 20, 1 << 20, False,
                                                  snapshot_format=snapshot_format)
                    self.assertNotIn("error", result)
                    restored = {}
                    self.assertNotIn("error", repl._restore_state(restored, path))
                    self.assertEqual(restored["records"]["left"][1], decimal.Decimal(amount))
                    self.assertIs(restored["records"]["left"], restored["records"]["right"])

    def test_legacy_overflow_writes_only_final_payload_and_restores_latest(self):
        namespace = {f"v{i}": bytes([i]) * 4096 for i in range(24)}
        blobs = {name: dill.dumps(value) for name, value in namespace.items()}
        cap = len(dill.dumps(blobs)) - 1
        with tempfile.TemporaryDirectory() as root:
            path, manifest = os.path.join(root, "state.dill"), os.path.join(root, "state.json")
            # Explicit legacy: `auto` now starts fresh sessions on CAS v2 (A5).
            result = repl._snapshot_state(
                namespace, path, manifest, cap, cap, False, snapshot_format="legacy"
            )
            self.assertNotIn("error", result)
            self.assertEqual(result["written_bytes"], os.path.getsize(path) + os.path.getsize(manifest))
            self.assertLessEqual(os.path.getsize(path), cap)
            self.assertTrue(result["skipped"])
            restored = {}
            self.assertNotIn("error", repl._restore_state(restored, path))
            self.assertEqual(restored, {name: namespace[name] for name in result["saved"]})
            self.assertFalse(list(Path(root).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
