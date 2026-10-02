"""Recovery must never open a path captured by a prior kernel file handle."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import dill

from rlm import repl, snapshot
from rlm.snapshot_safety import SnapshotFileHandleError, load_snapshot_value


class Box:
    def __init__(self, value):
        self.value = value


def unsafe_dump(dill_module, value, buffer, writer, metrics=None):
    dill_module.dump(value, writer)


class SnapshotFileHandleTests(unittest.TestCase):
    def setUp(self):
        previous = dill.settings.copy()
        self.addCleanup(dill.settings.update, previous)
        dill.settings["recurse"] = True

    def capture(self, namespace, root, fmt):
        result = repl._snapshot_state(namespace, str(root / "state.dill"),
                                      str(root / "state.json"), 1 << 20, 1 << 20,
                                      True, snapshot_format=fmt)
        self.assertNotIn("error", result)
        return result

    def test_new_snapshots_skip_open_and_closed_handles_nested_in_graphs(self):
        for fmt in ("legacy", "cas-v2"):
            for mode in ("w", "a", "x", "r", "wb", "ab", "xb", "rb"):
                for closed in (False, True):
                    with self.subTest(format=fmt, mode=mode, closed=closed), tempfile.TemporaryDirectory() as d:
                        root = Path(d)
                        path = root / "synthetic.txt"
                        if mode.startswith("r"):
                            path.write_bytes(b"readable")
                        handle = path.open(mode)
                        try:
                            if closed:
                                handle.close()
                            graph = [0] * 128 + [Box(handle)]
                            graph.append(graph)
                            namespace = {"handle": handle, "nested": graph, "safe": [42]}
                            result = self.capture(namespace, root, fmt)
                            self.assertEqual(result["saved"], ["safe"])
                            self.assertEqual(result["pruned"], [])
                            self.assertEqual({x["name"] for x in result["skipped"]}, {"handle", "nested"})
                            self.assertTrue(all("file/stream handles" in x["reason"] for x in result["skipped"]))
                            self.assertIs(namespace["handle"], handle)
                            self.assertEqual(handle.closed, closed)
                            restored = {}
                            reply = repl._restore_state(restored, str(root / "state.dill"))
                            self.assertNotIn("error", reply)
                            self.assertEqual(restored, {"safe": [42]})
                        finally:
                            handle.close()

    def test_old_snapshots_reject_handles_before_reopening_existing_or_missing_paths(self):
        for fmt in ("legacy", "cas-v2"):
            for mode in ("w", "a", "x", "r", "wb", "ab", "xb", "rb"):
                for missing in (False, True):
                    with self.subTest(format=fmt, mode=mode, missing=missing), tempfile.TemporaryDirectory() as d:
                        root = Path(d)
                        path = root / "synthetic.txt"
                        if mode.startswith("r"):
                            path.write_bytes(b"readable")
                        with path.open(mode) as handle:
                            pass
                        namespace = {"handle": handle, "nested": Box([handle]), "safe": 42}
                        with mock.patch.object(repl, "dump_snapshot_value", unsafe_dump), \
                             mock.patch.object(snapshot, "dump_snapshot_value", unsafe_dump):
                            result = self.capture(namespace, root, fmt)
                        self.assertEqual(result["saved"], ["handle", "nested", "safe"])
                        path.write_bytes(b"SURVIVOR")
                        if missing:
                            path.unlink()
                        with mock.patch.object(dill._dill, "_create_filehandle", side_effect=AssertionError("reopened")) as opener:
                            restored = {}
                            reply = repl._restore_state(restored, str(root / "state.dill"))
                        opener.assert_not_called()
                        self.assertNotIn("error", reply)
                        self.assertEqual(restored, {"safe": 42})
                        self.assertEqual({x["name"] for x in reply["failed"]}, {"handle", "nested"})
                        self.assertTrue(all("refusing to reopen" in x["reason"] for x in reply["failed"]))
                        self.assertEqual(path.exists(), not missing)
                        if not missing:
                            self.assertEqual(path.read_bytes(), b"SURVIVOR")

    def test_legacy_envelope_cannot_reopen_a_file_before_variable_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "synthetic.txt"
            with path.open("w") as handle:
                pass
            (root / "state.dill").write_bytes(dill.dumps({"unexpected": handle}))
            path.write_bytes(b"SURVIVOR")
            restored = {"existing": 7}
            reply = repl._restore_state(restored, str(root / "state.dill"))
            self.assertIn("refusing to reopen", reply["error"])
            self.assertEqual(restored, {"existing": 7})
            self.assertEqual(path.read_bytes(), b"SURVIVOR")

    def test_historical_dill_module_name_is_also_blocked(self):
        # Protocol 0 GLOBAL points to the same constructor's older module name.
        with self.assertRaises(SnapshotFileHandleError):
            load_snapshot_value(dill, b"cdill.dill\n_create_filehandle\n.")

    def test_memory_buffers_still_round_trip_in_both_formats(self):
        for fmt in ("legacy", "cas-v2"):
            with self.subTest(format=fmt), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                binary, text = io.BytesIO(b"data"), io.StringIO("text")
                binary.seek(2)
                text.seek(1)
                namespace = {"binary": binary, "text": text, "safe": Box([1, 2])}
                result = self.capture(namespace, root, fmt)
                self.assertEqual(result["saved"], sorted(namespace))
                restored = {}
                reply = repl._restore_state(restored, str(root / "state.dill"))
                self.assertEqual(reply["failed"], [])
                self.assertEqual(restored["binary"].getvalue(), b"data")
                self.assertEqual(restored["binary"].tell(), 2)
                self.assertEqual(restored["text"].getvalue(), "text")
                self.assertEqual(restored["text"].tell(), 1)
                self.assertEqual(restored["safe"].value, [1, 2])


if __name__ == "__main__":
    unittest.main()
