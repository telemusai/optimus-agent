from __future__ import annotations

import functools
import tempfile
import unittest
from pathlib import Path

from rlm.harness import HarnessState
from test_repl import ReplProcess, one, stream_text


class KernelBoundsTest(unittest.TestCase):
    def setUp(self):
        self.repl = ReplProcess()
        self.addCleanup(self.repl.close)
        self.repl.ready()

    def test_large_unicode_stream_is_chunked_without_loss(self):
        events = self.repl.execute("unicode", "import sys\n_ = sys.stdout.write('😀' * 150000)")
        frames = [event for event in events if event["event"] == "stdout"]
        self.assertGreater(len(frames), 1)
        self.assertTrue(all(len(frame["text"]) <= 65536 for frame in frames))
        self.assertTrue(all(frame["id"] == "unicode" for frame in frames))
        self.assertEqual(stream_text(events, "stdout"), "😀" * 150000)

    def test_large_result_and_exception_are_bounded_and_kernel_survives(self):
        events = self.repl.execute("result", "'😀' * 2000000")
        result = one(events, "result")["text"]
        self.assertLess(len(result), 1048700)
        self.assertIn("text truncated", result)
        events = self.repl.execute("error", "raise ValueError('😀' * 2000000)")
        error = one(events, "error")
        self.assertLess(len(error["evalue"]), 1048700)
        self.assertLess(len("".join(error["traceback"])), 1048700)
        self.assertTrue(all(len(line) < 32 * 1024 * 1024 for line in self.repl.raw_lines))
        self.assertEqual(one(self.repl.execute("next", "6 * 7"), "result")["text"], "42")

    def test_oversized_and_nonfinite_payloads_fail_before_host_dispatch(self):
        for index, code in enumerate([
            "from rlm.repl import emit\nemit({'text/plain': '😀' * 1500000})",
            "from rlm.repl import host_request\nawait host_request({'value': '😀' * 1500000})",
            "from rlm.repl import host_request\nawait host_request({'value': float('nan')})",
        ]):
            with self.subTest(code=code):
                events = self.repl.execute(str(index), code)
                self.assertEqual(one(events, "error")["ename"], "ValueError")
                self.assertIsNone(one(events, "display"))
                self.assertIsNone(one(events, "host_request"))
                self.assertEqual(one(events, "done")["status"], "error")
        events = self.repl.execute("pending", "from rlm.repl import _pending_host\nlen(_pending_host)")
        self.assertEqual(one(events, "result")["text"], "0")


class HarnessValidationTest(unittest.TestCase):
    def test_invalid_writes_preserve_memory_and_disk(self):
        with tempfile.TemporaryDirectory() as root:
            state = HarnessState(Path(root) / "state.json")
            state.create_memory("Keep", "Valid content", id="keep")
            before = state.file_path.read_bytes()
            for fields in [
                {"content": ["bad"]}, {"content": ""}, {"title": None},
                {"id": []}, {"metadata": []}, {"metadata": {"bad": float("nan")}},
                {"arguments": {"bad": object()}}, {"reference": "bad"}, {"path": 3},
                {"source": ""},
            ]:
                for operation in (state.create_memory, functools.partial(state.upsert, "memory")):
                    with self.subTest(fields=fields, operation=str(operation)):
                        args = {"title": "New", "content": "Valid", "id": "new", **fields}
                        with self.assertRaises(ValueError):
                            operation(**args)
                        self.assertEqual(state.file_path.read_bytes(), before)
                if "id" not in fields:
                    with self.subTest(fields=fields, operation="update"):
                        with self.assertRaises(ValueError):
                            state.update_memory(**{"id": "keep", "title": "Keep", "content": "Valid", **fields})
                        self.assertEqual(state.get("memory", "keep").content, "Valid content")
                        self.assertEqual(state.get("memory", "keep").version, 1)
                        self.assertEqual(state.file_path.read_bytes(), before)
            self.assertEqual(HarnessState(state.file_path).get("memory", "keep").content, "Valid content")

    def test_refinement_validation_happens_before_persistence(self):
        with tempfile.TemporaryDirectory() as root:
            state = HarnessState(Path(root) / "state.json")
            state.record_refinement("Observed", ["Fixed"], evidence="test", outcome="pass")
            before = state.file_path.read_bytes()
            for fields in [{"trigger": ""}, {"changes": {}}, {"changes": ["ok", 3]},
                           {"evidence": []}, {"outcome": None}, {"id": ""}]:
                with self.subTest(fields=fields):
                    with self.assertRaises(ValueError):
                        state.record_refinement(**{"trigger": "Valid", "changes": "Fixed", **fields})
                    self.assertEqual(state.file_path.read_bytes(), before)
                    self.assertEqual(len(state.refinements), 1)

    def test_generic_skill_writes_require_python_reference(self):
        with tempfile.TemporaryDirectory() as root:
            state = HarnessState(Path(root) / "state.json")
            with self.assertRaises(ValueError):
                state.create("skill", "Skill", "Body")
            self.assertEqual(state.list(), [])
