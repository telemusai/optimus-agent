from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import rlm
from rlm import lifecycle
import agent_message


def capability(**changes):
    report = {
        "schema": "optimus.native-lifecycle.v1", "capability": "rlm.stop-retain.v1", "supported": True,
        "targetProfile": {"model": "fixture/local", "api": "openai-completions", "tools": ["ipython"],
                          "processContainment": "windows-job", "scope": "local-invocation-and-contained-descendants"},
        "provenance": {"hostImplementation": "optimus-rust", "protocolVersion": 7, "schemaRevision": 32,
                       "buildFingerprint": "a" * 64, "runtimeSourceSha256": lifecycle.runtime_source_sha256()},
        "activeOnlyMessages": {"capability": "rlm.active-only-message.v1", "supported": True, "schema": "optimus.active-message.v1"},
        "auditResume": {"capability": "rlm.audit-resume.v1", "supported": True},
    }
    report.update(changes)
    return report


def stop_receipt(settled=False):
    return {"schema": "optimus.stop-retain.v1", "accepted": True, "settled": settled,
            "retained": True, "automatic_continuation_fenced": True, "rlm_child_id": "child", "session_id": "session",
            "stop_generation": "stop-1", "acknowledged": {k: settled for k in
                ("model", "tools", "kernel", "owned_processes", "transcript_flushed")}}


class LifecycleContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_host_refuses_without_delete_or_send_fallback(self):
        with patch.object(lifecycle, "host_request", AsyncMock(side_effect=RuntimeError("unknown operation"))) as request:
            with self.assertRaises(lifecycle.UnsupportedCapability):
                await lifecycle.stop_subagent("child")
            self.assertEqual(request.await_count, 1)
            self.assertEqual(request.call_args.args[0], "rlm.lifecycle_capabilities")

    async def test_pending_is_not_settlement(self):
        with patch.object(lifecycle, "host_request", AsyncMock(side_effect=[capability(), stop_receipt()])) as request:
            receipt = await lifecycle.stop_subagent("child", timeout_ms=0)
            self.assertTrue(receipt["accepted"])
            self.assertFalse(receipt["settled"])
            self.assertEqual(request.call_args.args, ("rlm.stop_subagent", {"target": "child", "timeout_ms": 0}))

    async def test_settled_requires_every_domain(self):
        for domain in ("model", "tools", "kernel", "owned_processes", "transcript_flushed"):
            receipt = stop_receipt(True)
            receipt["acknowledged"][domain] = False
            with patch.object(lifecycle, "host_request", AsyncMock(side_effect=[capability(), receipt])):
                with self.assertRaises(lifecycle.UnsupportedCapability):
                    await lifecycle.stop_subagent("child")

    async def test_unsupported_profile_never_sends_stop(self):
        with patch.object(lifecycle, "host_request", AsyncMock(return_value=capability(supported=False))) as request:
            with self.assertRaises(lifecycle.UnsupportedCapability): await lifecycle.stop_subagent("child")
            self.assertEqual(request.await_count, 1)

    async def test_provenance_and_model_must_match(self):
        report = capability()
        report["provenance"]["runtimeSourceSha256"] = "b" * 64
        with patch.object(lifecycle, "host_request", AsyncMock(return_value=report)):
            with self.assertRaises(lifecycle.UnsupportedCapability): await lifecycle.lifecycle_capabilities(model="fixture/local")
        with patch.object(lifecycle, "host_request", AsyncMock(return_value=capability())):
            with self.assertRaises(lifecycle.UnsupportedCapability): await lifecycle.lifecycle_capabilities(model="wrong/model")

    async def test_active_query_returns_original_token_without_new_run(self):
        state = {"schema": "optimus.active-execution.v1", "active": True, "fenced": False,
                 "session_id": "session", "execution_generation": "original", "rlm_child_id": "child"}
        with patch.object(lifecycle, "host_request", AsyncMock(side_effect=[capability(), state])) as request:
            self.assertEqual(await lifecycle.active_execution("child"), state)
            self.assertEqual(request.call_args.args[0], "rlm.active_execution")

    async def test_active_only_send_decline_has_no_default_delivery(self):
        receipt = {"schema": "optimus.active-message.v1", "accepted": False, "deliveryStatus": "declined_generation",
                   "executionGeneration": "old", "messageId": "one", "wakeIfIdle": False}
        with patch.object(lifecycle, "host_request", AsyncMock(return_value=capability())):
            with patch.object(agent_message, "host_request", AsyncMock(return_value=receipt)) as request:
                result = await agent_message.send("note", receiver_role="child", receiver_name="child",
                                                  wake_if_idle=False, execution_generation="old", message_id="one")
                self.assertFalse(result["accepted"])
                self.assertEqual(request.await_count, 1)
                self.assertEqual(request.call_args.args[0], "rlm.send_active_message")

    async def test_active_only_requires_capability_and_exact_receipt_token(self):
        with patch.object(lifecycle, "host_request", AsyncMock(return_value=capability(activeOnlyMessages={}))):
            with patch.object(agent_message, "host_request", AsyncMock()) as request:
                with self.assertRaises(lifecycle.UnsupportedCapability):
                    await agent_message.send("note", receiver_role="child", receiver_name="child",
                                             wake_if_idle=False, execution_generation="g", message_id="id")
                request.assert_not_called()

    async def test_normal_message_default_is_unchanged(self):
        with patch.object(agent_message, "host_request", AsyncMock(return_value={"deliveryStatus": "queued"})) as request:
            await agent_message.send("note", receiver_role="child", receiver_name="child")
            self.assertEqual(request.call_args.args[0], "agent_message.send")
            self.assertNotIn("execution_generation", request.call_args.args[1])

    async def test_explicit_audit_has_new_prompt_and_exact_stop_token(self):
        receipt = {"schema": "optimus.audit-resume.v1", "stop_generation": "stop-1", "accepted": True, "old_work_replayed": False}
        with patch.object(lifecycle, "host_request", AsyncMock(side_effect=[capability(), receipt])) as request:
            await lifecycle.resume_subagent("child", stop_generation="stop-1", prompt="Review the transcript only")
            self.assertEqual(request.call_args.args[0], "rlm.resume_subagent")
            self.assertEqual(request.call_args.args[1]["stop_generation"], "stop-1")

    async def test_bounds_fail_before_host_calls(self):
        with patch.object(lifecycle, "host_request", AsyncMock()) as request:
            for timeout in (-1, True, 10001):
                with self.assertRaises(TypeError): await lifecycle.stop_subagent("child", timeout_ms=timeout)
            request.assert_not_called()

    def test_actual_callable_origins_and_required_signatures(self):
        self.assertEqual(Path(inspect.getsourcefile(lifecycle.stop_subagent)).resolve(), Path(lifecycle.__file__).resolve())
        self.assertEqual(Path(inspect.getsourcefile(agent_message.send)).resolve(), Path(agent_message.__file__).resolve())
        self.assertIn("timeout_ms", inspect.signature(lifecycle.stop_subagent).parameters)
        self.assertIn("target", inspect.signature(lifecycle.lifecycle_capabilities).parameters)
        self.assertIn("wake_if_idle", inspect.signature(agent_message.send).parameters)
        self.assertEqual(rlm.rlm.execution.report_script_result, rlm.execution.report_script_result)


if __name__ == "__main__": unittest.main()
