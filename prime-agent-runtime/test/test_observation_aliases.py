from __future__ import annotations

import asyncio
import importlib
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


rlm_module = importlib.import_module("rlm")
observe_path = Path(__file__).resolve().parents[2] / "resources/agent/skills/agent-observe/src/agent_observe/__init__.py"
spec = importlib.util.spec_from_file_location("agent_observe_test", observe_path)
assert spec is not None and spec.loader is not None
observe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observe)


class ObservationAliasesTest(unittest.TestCase):
    def test_rlm_alias_preserves_name_and_does_not_invent_model(self) -> None:
        payload = {"rlm_child_id": "child", "session_name": "worker", "session_dir": "/tmp/child", "status": "completed"}
        child = rlm_module._subagent_from_payload(payload)
        self.assertEqual(child.name, child.session_name)
        self.assertIsNone(child.model)
        child = rlm_module._subagent_from_payload({**payload, "model": "faux/model"})
        self.assertEqual(child.model, "faux/model")
        with self.assertRaisesRegex(RuntimeError, "invalid model"):
            rlm_module._subagent_from_payload({**payload, "model": {"id": "unknown shape"}})

    def test_observe_aliases_preserve_canonical_fields_without_mutating_reply(self) -> None:
        summary = {"sessionName": "worker", "status": "user", "messageCount": 3}
        reply = {"agent": summary, "messages": [{"text": "visible result", "index": 1}]}
        host = AsyncMock(return_value=reply)
        with patch.object(observe, "host_request", host):
            result = asyncio.run(observe.recent_messages("worker"))
        self.assertEqual(result["agent"]["name"], "worker")
        self.assertEqual(result["agent"]["sessionName"], "worker")
        self.assertEqual(result["agent"]["status"], "user")
        self.assertEqual(result["agent"]["activityStatus"], "attached_idle")
        self.assertEqual(result["agent"]["messageCount"], 3)
        self.assertEqual(result["messages"][0]["content"], result["messages"][0]["text"])
        self.assertNotIn("name", summary)
        self.assertNotIn("content", reply["messages"][0])

    def test_list_get_and_missing_optional_metadata_degrade_locally(self) -> None:
        host = AsyncMock(return_value={"current": {"sessionName": "root"}, "agents": [{"sessionName": "child"}, {"sessionId": "unknown"}]})
        with patch.object(observe, "host_request", host):
            result = asyncio.run(observe.list_agents())
        self.assertEqual(result["current"]["name"], "root")
        self.assertEqual(result["agents"][0]["name"], "child")
        self.assertNotIn("name", result["agents"][1])
        host = AsyncMock(return_value={"agent": {"sessionId": "unknown"}})
        with patch.object(observe, "host_request", host):
            result = asyncio.run(observe.get_agent("unknown"))
        self.assertEqual(result, {"agent": {"sessionId": "unknown"}})


if __name__ == "__main__":
    unittest.main()
