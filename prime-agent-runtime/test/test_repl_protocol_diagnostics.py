"""Broken protocol output must reach the host log without recursive capture."""

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from rlm import repl


class ProtocolDiagnosticsTests(unittest.TestCase):
    def test_closed_protocol_logs_to_original_stderr_without_deadlocking(self):
        result = subprocess.run(
            [sys.executable, "-c", """
import os
from rlm import repl
repl._setup_fds()
assert not os.get_inheritable(repl._host_stderr_fd)
os.close(repl._protocol_fd)
repl._send({'event': 'stdout', 'text': 'private-frame-content'})
repl._send({'event': 'done'})
os._exit(0)
"""],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr.count("rlm.repl: dropped protocol frame:"), 2)
        self.assertNotIn("private-frame-content", result.stderr)

    def test_failed_diagnostic_write_is_best_effort(self):
        with mock.patch.object(repl.os, "write", side_effect=OSError("closed")) as write:
            repl._send({"event": "done"})
        self.assertEqual(write.call_count, 2)

    def test_zero_byte_write_cannot_spin_forever(self):
        with mock.patch.object(repl.os, "write", side_effect=[0, 1]) as write:
            repl._send({"event": "done"})
        self.assertEqual(write.call_count, 2)
        self.assertIn(b"no progress", write.call_args.args[1])

    def test_partial_writes_still_deliver_one_complete_frame(self):
        chunks = []
        def write(_fd, data):
            chunks.append(bytes(data[:3]))
            return len(chunks[-1])
        with mock.patch.object(repl.os, "write", side_effect=write):
            repl._send({"event": "done"})
        self.assertEqual(b"".join(chunks), b'{"event":"done"}\n')


if __name__ == "__main__":
    unittest.main()
