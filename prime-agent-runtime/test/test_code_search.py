import contextlib
import copy
import io
import json
import unittest
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

if __name__ == "__main__":
    unittest.main()
