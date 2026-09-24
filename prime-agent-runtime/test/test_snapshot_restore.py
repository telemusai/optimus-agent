from __future__ import annotations

import os
import sys
import tempfile
import unittest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rlm import repl
from rlm.snapshot_restore import prepare_restored_values
from test_repl import ReplProcess, one


class SnapshotRestoreFunctionsTest(unittest.TestCase):
    def roundtrip(self, code, probe, expected):
        for snapshot_format in ("legacy", "cas-v2"):
            with self.subTest(format=snapshot_format), tempfile.TemporaryDirectory() as directory:
                kernel = ReplProcess()
                try:
                    kernel.ready()
                    self.assertEqual(one(kernel.execute("setup", code), "done")["status"], "ok")
                    path = os.path.join(directory, "state.dill")
                    kernel.send({
                        "type": "snapshot", "id": "save", "path": path,
                        "manifest_path": os.path.join(directory, "state.json"),
                        "snapshot_format": snapshot_format,
                    })
                    saved = one(kernel.until_done("save"), "done")
                    self.assertEqual(saved["status"], "ok")
                    kernel.send({"type": "restore", "id": "restore", "path": path})
                    restored = one(kernel.until_done("restore"), "done")
                    self.assertEqual(restored["status"], "ok")
                    self.assertEqual(restored["failed"], [])
                    events = kernel.execute("probe", probe)
                    self.assertEqual(one(events, "done")["status"], "ok", events)
                    self.assertEqual(one(events, "result")["text"], expected)
                finally:
                    kernel.close()

    def test_functions_see_reassigned_and_new_globals(self):
        self.roundtrip(
            "G = 1\ndef reader():\n    return G\ndef prober():\n    return late",
            "G = 2\nlate = 'live'\n(reader(), prober())",
            "(2, 'live')",
        )

    def test_defaults_closures_and_attributes_keep_live_functions_and_shared_cells(self):
        self.roundtrip(
            "G = 1\ndef helper():\n    return G\n"
            "run = lambda fn=helper: fn()\nrun2 = lambda *, fn=helper: fn()\n"
            "closed = (lambda fn: lambda: fn())(helper)\nrun.callback = helper\n"
            "def make():\n    n = 0\n    def get():\n        return n\n"
            "    def set(v):\n        nonlocal n\n        n = v\n"
            "    get.set = set\n    return get\ncounter = make()",
            "G = 2\ncounter.set(5)\n(run(), run2(), closed(), counter(), run.callback())",
            "(2, 2, 2, 5, 2)",
        )

    def test_partial_args_keywords_and_attributes(self):
        self.roundtrip(
            "import functools\nG = 1\ndef helper():\n    return G\n"
            "def apply(fn, *, other):\n    return fn(), other()\n"
            "wrapped = functools.partial(apply, helper, other=helper)\n"
            "wrapped.label = 'x'\nwrapped.callback = helper",
            "G = 2\n(wrapped(), wrapped.label, wrapped.callback())",
            "((2, 2), 'x', 2)",
        )

    def test_container_aliases_cycles_and_partial_cycles(self):
        self.roundtrip(
            "import functools\nG = 1\ndef reader():\n    return G\n"
            "bundle = [reader, reader, {'read': reader}, (reader,)]\nbundle.append(bundle)\n"
            "args = []\nwrapped = functools.partial(len, args)\nargs.append(wrapped)\n"
            "wrapped.self = wrapped",
            "G = 2\n(bundle[0](), bundle[2]['read'](), bundle[3][0](), "
            "bundle[0] is bundle[1], bundle[4] is bundle, "
            "wrapped.args[0][0] is wrapped, wrapped.self is wrapped)",
            "(2, 2, 2, True, True, True, True)",
        )

    def test_private_function_globals_backfill_respects_exclusions(self):
        self.roundtrip(
            "exec('PUB = 42\\n_hidden = 2\\nIn = 3\\nOut = 4\\nmcp = 5\\n"
            "def reader():\\n    return PUB', pn:={'__name__': '__main__'})\n"
            "reader = pn['reader']",
            "(reader(), '_hidden' in dir(), 'In' in dir(), 'Out' in dir(), 'mcp' in dir())",
            "(42, False, False, False, False)",
        )

    def test_backfill_partial_cycle(self):
        self.roundtrip(
            "exec('import functools\\nG = 1\\ndef base():\\n    return G, wrapped\\n"
            "wrapped = functools.partial(base)\\ndef entry():\\n    return wrapped()', "
            "pn:={'__name__': '__main__'})\nentry = pn['entry']",
            "G = 2\n(entry()[0], entry()[1] is wrapped)",
            "(2, True)",
        )

    def test_existing_and_restored_values_win_over_backfill(self):
        for snapshot_format in ("legacy", "cas-v2"):
            with self.subTest(format=snapshot_format), tempfile.TemporaryDirectory() as directory:
                source = {"__name__": "__main__"}
                exec("G = 1\nH = 2\ndef reader():\n    return G, H", source)
                path = os.path.join(directory, "state.dill")
                result = repl._snapshot_state(
                    {"reader": source["reader"], "H": 7}, path,
                    os.path.join(directory, "state.json"), 1 << 20, 1 << 20, False,
                    snapshot_format=snapshot_format,
                )
                self.assertNotIn("error", result)
                target = {"G": 9}
                result = repl._restore_state(target, path)
                self.assertEqual(result["failed"], [])
                self.assertEqual(target["reader"](), (9, 7))

    def test_imported_function_and_live_globals_are_not_walked(self):
        import posixpath
        live = {"__name__": "__main__", "imported": posixpath.join}
        values, backfill, failures = prepare_restored_values({"live": live, "fn": posixpath.join}, live, set())
        self.assertIs(values["live"], live)
        self.assertIs(values["fn"], posixpath.join)
        self.assertEqual(backfill, [])
        self.assertEqual(failures, [])

    def test_failed_revival_does_not_publish_its_backfill(self):
        class Broken(dict):
            def items(self):
                raise ValueError("broken restored container")

        source = {"__name__": "__main__", "SECRET": 42}
        exec("def reader():\n    return SECRET", source)
        source["reader"].bad = Broken()
        target = {}
        values, backfill, failures = prepare_restored_values({"reader": source["reader"], "ok": 3}, target, set())
        self.assertEqual(values, {"ok": 3})
        self.assertEqual(backfill, [])
        self.assertEqual(target, {})
        self.assertEqual(failures[0]["name"], "reader")


if __name__ == "__main__":
    unittest.main()
