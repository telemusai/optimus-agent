import ast
import builtins
import contextlib
import copy
import inspect
import io
import json
import types
import unittest

import rlm.code_search as code_search
from rlm.code_search import from_ripgrep, present

class CodeSearchTests(unittest.TestCase):
    def test_ripgrep_roundtrip_keeps_locations_and_original_candidates(self):
        raw = json.dumps({"type":"match","data":{"path":{"text":"auth/session.rs"},"lines":{"text":"expiry = ttl"},"line_number":42}})
        candidates = from_ripgrep(raw)
        before = copy.deepcopy(candidates)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            present(candidates)
        data = json.loads(output.getvalue())
        self.assertEqual(data["schema"], "rlm.code-search/1")
        self.assertEqual(data["candidates"][0]["line"], 42)
        self.assertEqual(candidates, before)

    def test_all_candidate_kinds_and_mandatory_are_preserved(self):
        items = [{"kind":kind,"path":"src/a.rs","snippet":"fn foo()","mandatory":True} for kind in ["file","symbol","grep","reference","test"]]
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            present(items)
        self.assertEqual(json.loads(output.getvalue())["candidates"],items)

    def test_rejects_invalid_or_oversized_envelopes_without_partial_output(self):
        valid={"kind":"file","path":"a.rs"}
        for items in [[valid]*501,[dict(valid,line=0)],[dict(valid,line=True)],[dict(valid,kind="command")],[dict(valid,snippet="x"*4097)],[dict(valid,unknown="data")]]:
            output=io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(ValueError):
                present(items)
            self.assertEqual(output.getvalue(),"")

    def test_dataset_path_matches_parse_and_present_without_file_reads(self):
        # Regression for the LP_JUDGE_MAX_TOKENS incident: a code search may
        # legitimately receive a dataset-path match in its captured rg output;
        # parse+present stay bounded, and neither helper reads any file.
        events = [
            {"type": "begin", "data": {"path": {"text": "crates/agent/src/judge.rs"}}},
            {"type": "match", "data": {"path": {"text": "crates/agent/src/judge.rs"},
                                       "lines": {"text": "const LP_JUDGE_MAX_TOKENS: u32 = 8192;"},
                                       "line_number": 12}},
            {"type": "match", "data": {"path": {"text": "data/common/longmemeval_m.json"},
                                       "lines": {"text": "{\"question_id\": \"q1\"}"},
                                       "line_number": 1}},
            {"type": "end", "data": {"path": {"text": "data/common/longmemeval_m.json"}}},
        ]
        raw = "\n".join(json.dumps(event) for event in events)
        self.assertLess(len(raw.encode("utf-8")), 512 * 1024)

        def blocked_open(*args, **kwargs):
            raise AssertionError("code_search helpers must not read files")

        saved_open = builtins.open
        builtins.open = blocked_open
        try:
            candidates = from_ripgrep(raw)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                present(candidates)
        finally:
            builtins.open = saved_open

        paths = [candidate["path"] for candidate in candidates]
        self.assertEqual(paths, ["crates/agent/src/judge.rs", "data/common/longmemeval_m.json"])
        self.assertEqual(candidates[0]["line"], 12)
        data = json.loads(output.getvalue())
        self.assertEqual(data["schema"], "rlm.code-search/1")
        self.assertEqual([c["path"] for c in data["candidates"]], paths)

    def test_module_has_no_file_io_surface(self):
        # By construction: the helper must only import json and collections.abc,
        # and its source must not open, read, or write any file.
        module_names = {
            name for name, value in vars(code_search).items()
            if isinstance(value, types.ModuleType)
        }
        self.assertTrue(module_names.issubset({"json", "collections"}), module_names)
        self.assertTrue(hasattr(code_search.json, "loads"))
        tree = ast.parse(inspect.getsource(code_search))
        called = {
            getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        for forbidden in ("open", "read", "read_text", "read_bytes", "write", "write_text"):
            self.assertNotIn(forbidden, called)


if __name__ == "__main__":
    unittest.main()
