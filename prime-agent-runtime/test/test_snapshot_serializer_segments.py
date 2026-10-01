from __future__ import annotations

import io
import math
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dill
from rlm import repl, snapshot, snapshot_serializer as serializer


class Box:
    def __init__(self, values):
        self.values = values


def old_fragment(buffer, writer, protocol):
    """Frozen original byte-copy algorithm, used only as an offline oracle."""
    data = buffer.getvalue()
    if data[:2] != bytes((pickle.PROTO[0], protocol)) or not data.endswith(pickle.STOP):
        return None
    if protocol >= 4:
        start = max(2, writer.last_write_start)
        if data[start:start + 1] == pickle.FRAME:
            size = int.from_bytes(data[start + 1:start + 9], "little")
            if size != len(data) - start - 9 or size < 1:
                return None
            data = data[:start + 1] + (size - 1).to_bytes(8, "little") + data[start + 9:]
        elif len(data) - start >= 4:
            return None
    return (data[2:-1],)


class SnapshotSegmentTests(unittest.TestCase):
    def setUp(self):
        settings = dill.settings.copy()
        self.addCleanup(dill.settings.update, settings)
        dill.settings["recurse"] = True

    def serialize(self, value, cap=4 << 20, metrics=None):
        output = io.BytesIO()
        serializer.dump_snapshot_value(dill, value, output, repl._CappedWriter(output, cap), metrics)
        return output.getvalue()

    def test_final_frame_segments_have_exact_golden_bytes_and_immutable_backing(self):
        for protocol in (3, 4, 5):
            for value in ([], list(range(50000)), [b"\x95.\x80" * 30000] + [0] * 128):
                with self.subTest(protocol=protocol, size=len(value)):
                    buffer = io.BytesIO()
                    writer = serializer._ProbeWriter(buffer, 4 << 20)
                    serializer._SubgraphPickler(writer, dill, protocol=protocol).dump(value)
                    expected = old_fragment(buffer, writer, protocol)
                    parts = serializer._value_fragment(buffer, writer, protocol)
                    self.assertIsNotNone(parts)
                    self.assertEqual(b"".join(parts), b"".join(expected))
                    views = [part for part in parts if isinstance(part, memoryview)]
                    self.assertTrue(views)
                    self.assertTrue(all(view.readonly and type(view.obj) is bytes for view in views))
                    backing = buffer.getvalue()
                    self.assertTrue(all(view.obj is backing for view in views))
                    buffer.seek(0)
                    buffer.write(b"changed")
                    buffer.close()
                    self.assertEqual(b"".join(parts), b"".join(expected), "segments own immutable backing")
        data = b"\x80\x04\x95" + (2).to_bytes(8, "little") + b"N."
        buffer = io.BytesIO(data)
        writer = serializer._ProbeWriter(buffer, 100)
        self.assertEqual(b"".join(serializer._value_fragment(buffer, writer, 4)),
                         b"\x95" + (1).to_bytes(8, "little") + b"N")

    def test_unknown_or_truncated_frame_layout_keeps_fallback(self):
        for data in (b"bad", b"\x80\x04abc.", b"\x80\x04\x95\x01.",
                     b"\x80\x04\x95" + (999).to_bytes(8, "little") + b"N."):
            buffer = io.BytesIO(data)
            writer = serializer._ProbeWriter(buffer, 1000)
            self.assertIsNone(serializer._value_fragment(buffer, writer, 4))

    def test_full_custom_graph_bytes_match_frozen_fragment_path_with_aliases_cycles(self):
        for protocol in (3, 4, 5):
            dill.settings["protocol"] = protocol
            shared = ["alias"]
            first = [shared] * 200
            second = [b"\x95.\x80" * 30000] + list(range(50000)) + [shared]
            cycle = [0] * 128
            cycle.append(cycle)
            value = Box([first, second, tuple(first), Box(shared), cycle])
            with mock.patch.object(serializer, "_value_fragment", old_fragment):
                expected = self.serialize(value)
            metrics = serializer.SnapshotPathMetrics()
            actual = self.serialize(value, metrics=metrics)
            self.assertEqual(actual, expected)
            restored = dill.loads(actual).values
            self.assertIs(restored[0][0], restored[1][-1])
            self.assertIs(restored[0][0], restored[2][0])
            self.assertIs(restored[0][0], restored[3].values)
            self.assertIs(restored[4][-1], restored[4])
            self.assertGreater(metrics.values["serialization_fragment_bytes"], 0)
            self.assertGreater(metrics.values["serialization_fragment_segments"], 0)

    def test_segment_cap_preflight_matches_single_write_without_partial_output(self):
        parts = (memoryview(b"abc"), b"def", memoryview(b"gh"))
        for writer_type, error_type in ((repl._CappedWriter, repl._SnapshotSizeLimitExceeded),
                                       (snapshot.CappedWriter, snapshot.SnapshotSizeLimitExceeded)):
            for remaining in (7, 8, 9):
                output = io.BytesIO()
                writer = writer_type(output, remaining + 2)
                writer.write(b"xy")
                if remaining < 8:
                    with self.assertRaises(error_type):
                        writer.write_segments(parts)
                    self.assertEqual(output.getvalue(), b"xy")
                    self.assertEqual(writer.written, 2)
                else:
                    self.assertEqual(writer.write_segments(parts), 8)
                    self.assertEqual(output.getvalue(), b"xyabcdefgh")
                    self.assertEqual(writer.written, 10)

    def test_unknown_writer_keeps_one_exact_fragment_write(self):
        class Pickler:
            def __init__(self):
                self.writes = []
            def _file_write(self, data):
                self.writes.append(data)
        pickler = Pickler()
        serializer._write_fragment(pickler, object(), (memoryview(b"ab"), b"cd"))
        self.assertEqual(pickler.writes, [b"abcd"])
        writer = repl._CappedWriter(io.BytesIO(), 100)
        with mock.patch.object(writer, "write_segments", side_effect=AssertionError("wrapped write bypassed")):
            serializer._write_fragment(pickler, writer, (memoryview(b"ef"), b"gh"))
        self.assertEqual(pickler.writes, [b"abcd", b"efgh"])

    def test_interrupted_segment_does_not_install_native_memo(self):
        picklers = []
        before = []
        real_init = dill.Pickler.__init__
        def capture_init(pickler, *args, **kwargs):
            real_init(pickler, *args, **kwargs)
            picklers.append(pickler)
        class InterruptWriter(repl._CappedWriter):
            def write_segments(self, parts):
                before.append(picklers[-1].memo.copy())
                self.write(parts[0])
                raise KeyboardInterrupt("synthetic partial fragment")
        buffer = io.BytesIO()
        rows = list(range(50000))
        with mock.patch.object(dill.Pickler, "__init__", capture_init):
            with self.assertRaises(KeyboardInterrupt):
                serializer.dump_snapshot_value(dill, Box(rows), buffer, InterruptWriter(buffer, 4 << 20))
        self.assertEqual(len(before), 1)
        self.assertEqual(picklers[-1].memo, before[0])
        self.assertNotIn(id(rows), picklers[-1].memo)
        self.assertFalse("save" in picklers[-1].__dict__, "wrapper must release captures on failure")

    def test_interrupted_variable_keeps_previous_snapshot_and_unpruned_namespace(self):
        for snapshot_format, writer_type in (("legacy", repl._CappedWriter), ("cas-v2", snapshot.CappedWriter)):
            with self.subTest(format=snapshot_format), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "state.dill")
                manifest = os.path.join(directory, "state.json")
                old = repl._snapshot_state({"value": "old"}, path, manifest, 4 << 20, 4 << 20,
                                           False, snapshot_format=snapshot_format)
                self.assertNotIn("error", old)
                namespace = {"oversized": b"x" * (4 << 20), "value": Box(list(range(50000)))}
                committed = []
                def interrupted(writer, parts):
                    writer.write(parts[0])
                    raise KeyboardInterrupt("synthetic write cancellation")
                with mock.patch.object(writer_type, "write_segments", interrupted):
                    with self.assertRaises(KeyboardInterrupt):
                        repl._snapshot_state(namespace, path, manifest, 4 << 20, 1 << 20,
                                             True, committed, snapshot_format=snapshot_format)
                self.assertEqual(committed, [])
                self.assertIn("oversized", namespace)
                restored = {}
                self.assertNotIn("error", repl._restore_state(restored, path))
                self.assertEqual(restored, {"value": "old"})
                self.assertFalse(list(Path(directory).rglob("*.tmp")))

    def test_envelope_count_matches_exact_dill_bytes_all_protocols_aliases_and_caps(self):
        shared = b"\x95\x80.\x00" * 20000
        for protocol in (0, 1, 2, 3, 4, 5):
            dill.settings["protocol"] = protocol
            for payload in ({}, {"a": b""}, {"a": shared, "b": shared},
                            {f"name{index}": bytes([index % 256]) * 31 for index in range(1001)}):
                expected = dill.dumps(payload)
                for cap in (0, len(expected) - 1, len(expected), len(expected) + 1):
                    with self.subTest(protocol=protocol, names=len(payload), cap=cap):
                        metrics = {"serialization_envelope_count_ms": 0.0,
                                   "serialization_envelope_count_calls": 0}
                        actual = snapshot._count_envelope(dill, payload, cap, metrics)
                        self.assertEqual(actual, len(expected) if cap >= len(expected) else None)
                        self.assertEqual(metrics["serialization_envelope_count_calls"], 1)
                        self.assertGreaterEqual(metrics["serialization_envelope_count_ms"], 0)

    def test_custom_envelope_dispatch_is_still_exact_dill(self):
        def save_bytes(pickler, value):
            pickler.save_reduce(bytearray, (list(value),), obj=value)
        with mock.patch.dict(dill.Pickler.dispatch, {bytes: save_bytes}):
            payload = {"a": b"synthetic"}
            expected = dill.dumps(payload)
            self.assertEqual(snapshot._count_envelope(dill, payload, len(expected)), len(expected))
            self.assertIsNone(snapshot._count_envelope(dill, payload, len(expected) - 1))

    def test_cap_selection_restore_and_legacy_write_bytes_match_original_path(self):
        for protocol in (3, 4, 5):
            dill.settings["protocol"] = protocol
            for snapshot_format in ("legacy", "cas-v2"):
                namespace = {"first": Box(list(range(200))), "tail": b"x" * 4096, "last": [1, 2]}
                for cap in (1000, 4200, 1 << 20):
                    with self.subTest(protocol=protocol, format=snapshot_format, cap=cap):
                        results = []
                        payloads = []
                        for original in (True, False):
                            with tempfile.TemporaryDirectory() as directory:
                                path = os.path.join(directory, "state.dill")
                                manifest = os.path.join(directory, "state.json")
                                with mock.patch.object(serializer, "_value_fragment",
                                                       old_fragment if original else serializer._value_fragment):
                                    result = repl._snapshot_state(dict(namespace), path, manifest, cap, cap,
                                                                  False, snapshot_format=snapshot_format)
                                self.assertNotIn("error", result)
                                restored = {}
                                self.assertNotIn("error", repl._restore_state(restored, path))
                                self.assertEqual(sorted(restored), result["saved"])
                                if "first" in restored:
                                    self.assertEqual(restored["first"].values, namespace["first"].values)
                                results.append({key: result[key] for key in ("saved", "skipped", "pruned", "bytes")})
                                if snapshot_format == "legacy":
                                    blob = Path(path).read_bytes()
                                    payload = dill.loads(blob)
                                    self.assertEqual(blob, dill.dumps(payload))
                                    payloads.append(blob)
                        self.assertEqual(results[0], results[1])
                        if payloads:
                            self.assertEqual(payloads[0], payloads[1])

    def test_detail_metrics_are_content_free_numeric_and_partition_probe_cost(self):
        for snapshot_format in ("legacy", "cas-v2"):
            with tempfile.TemporaryDirectory() as directory:
                namespace = {"private_saved_name": Box(list(range(200))),
                             "private_skipped_name": Box(list(range(10000)))}
                result = repl._snapshot_state(namespace, os.path.join(directory, "state.dill"),
                                              os.path.join(directory, "state.json"), 1 << 20, 2000,
                                              False, snapshot_format=snapshot_format)
                self.assertNotIn("error", result)
                metrics = result["metrics"]
                self.assertTrue(all(value is None or (type(value) in (float, int) and math.isfinite(value)
                                                      and value >= 0) for value in metrics.values()))
                self.assertNotIn("private_saved_name", repr(metrics))
                self.assertNotIn("private_skipped_name", repr(metrics))
                self.assertGreater(metrics["serialization_native_probe_attempts"], 0)
                self.assertGreater(metrics["serialization_fragment_bytes"], 0)
                self.assertGreater(metrics["serialization_envelope_count_calls"], 0)
                self.assertEqual(metrics["snapshot_legacy_captures"], int(snapshot_format == "legacy"))
                self.assertEqual(metrics["snapshot_cas_captures"], int(snapshot_format == "cas-v2"))
                self.assertAlmostEqual(metrics["serialization_native_probe_ms"],
                                       metrics["serialization_native_probe_saved_ms"]
                                       + metrics["serialization_native_probe_skipped_ms"])
                self.assertGreaterEqual(metrics["serialization_native_probe_rejected"], 1)
                self.assertEqual(result["saved"], ["private_saved_name"])


if __name__ == "__main__":
    unittest.main()
