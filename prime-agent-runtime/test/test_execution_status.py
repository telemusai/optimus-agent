from __future__ import annotations
import asyncio
import json
import sys
import unittest
from unittest.mock import AsyncMock, patch
from rlm.bash import BashResult
from rlm import repl
from rlm.execution import report_script_result
from test_repl import ReplProcess, one


class ScriptReportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.execution = repl._CellExecution()
        self.execution.owner = asyncio.current_task()
        self.token = repl._current_cell_execution.set(self.execution)

    async def asyncTearDown(self):
        repl._current_cell_execution.reset(self.token)

    async def test_completed_cell_failure_and_expected_nonzero(self):
        self.assertTrue(report_script_result(BashResult(7, "ordinary text", 0.1))["isError"])
        self.assertFalse(report_script_result(BashResult(1, "error Traceback", 0), expected_exit_codes=(0,1))["isError"])
        self.assertFalse(report_script_result(BashResult(0, "PermissionError traceback error", 0))["isError"])

    async def test_fatal_stop_source_guard_and_signal_cannot_be_downgraded(self):
        for receipt in ({"status":"fatal","fatal":True,"errorCode":"SOURCE_CHANGED"}, {"status":"aborted"}, {"isError":True}):
            self.assertTrue(report_script_result(BashResult(0,"",0),receipt=receipt)["isError"])
        self.assertTrue(report_script_result(BashResult(-9,"",0),expected_exit_codes=(-9,))["isError"])

    async def test_metadata_is_a_snapshot_and_output_is_not_inspected(self):
        receipt={"schema":"routeworld.native-script-result.v1","status":"error","nativeOutcome":{"nativeCallCommitted":True}}
        report=report_script_result(BashResult(4,"private output not copied",0),receipt=receipt)
        receipt["nativeOutcome"]["nativeCallCommitted"]=False
        report["receipt"]["status"]="ok"
        stored=self.execution.execution_reports[0]
        self.assertTrue(stored["receipt"]["nativeOutcome"]["nativeCallCommitted"])
        self.assertEqual(stored["receipt"]["status"],"error")
        self.assertNotIn("private output",json.dumps(stored))

    async def test_reject_invalid_result_and_receipt(self):
        for result in (None, object(), "Traceback", BashResult(True,"",0), BashResult(0,"",float("nan"))):
            with self.assertRaises(TypeError): report_script_result(result)
        for receipt in ({"fatal":"false"},{"status":"unknown"},{"x":float("nan")}):
            with self.assertRaises(TypeError): report_script_result(BashResult(0,"",0),receipt=receipt)
        with self.assertRaises(ValueError): report_script_result(BashResult(0,"",0),receipt={"x":"x"*17000})

    async def test_background_late_reports_require_a_new_active_cell(self):
        self.execution.finished.set()
        with self.assertRaises(RuntimeError): report_script_result(BashResult(0,"",0))

    async def test_report_limit_is_fail_closed(self):
        for _ in range(32): report_script_result(BashResult(0,"",0))
        with self.assertRaises(RuntimeError): report_script_result(BashResult(0,"",0))


class ScriptProtocolTests(unittest.TestCase):
    def setUp(self):
        self.runtime=ReplProcess()
        self.addCleanup(self.runtime.close)
        self.runtime.ready()

    def test_real_supervised_process_failure_reaches_done_metadata(self):
        command='"'+sys.executable+'" -c "raise SystemExit(7)"'
        code="from rlm.execution import report_script_result\nfrom rlm import bash\nr=await bash("+repr(command)+")\nreport_script_result(r,script_id='offline-fixture')\nprint('cell completed')"
        events=self.runtime.execute("script",code)
        done=one(events,"done")
        self.assertEqual(done["status"],"ok")
        self.assertEqual(done["executionReports"][0]["exitCode"],7)
        self.assertTrue(done["executionReports"][0]["isError"])
        self.assertIsNone(one(events,"error"))
        next_done=one(self.runtime.execute("next","print('error Traceback')"),"done")
        self.assertEqual(next_done["status"],"ok")
        self.assertNotIn("executionReports",next_done)

    def test_real_expected_nonzero_is_not_a_tool_failure(self):
        command='"'+sys.executable+'" -c "raise SystemExit(1)"'
        code="from rlm.execution import report_script_result\nfrom rlm import bash\nr=await bash("+repr(command)+")\nreport_script_result(r,expected_exit_codes=(0,1))"
        done=one(self.runtime.execute("handled",code),"done")
        self.assertFalse(done["executionReports"][0]["isError"])

    def test_genuine_kernel_error_preserves_status_and_unreported_command_is_legacy(self):
        events=self.runtime.execute("error","raise ValueError('fixture')")
        self.assertEqual(one(events,"done")["status"],"error")
        self.assertEqual(one(events,"error")["ename"],"ValueError")
        events=self.runtime.execute("syntax","if False print('fixture')")
        self.assertEqual(one(events,"done")["status"],"error")
        self.assertEqual(one(events,"error")["ename"],"SyntaxError")


if __name__ == "__main__": unittest.main()
