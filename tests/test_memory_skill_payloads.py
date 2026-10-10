"""Offline payload checks against the repository memory skill, not an installed copy."""

import asyncio
import copy
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / "resources/agent/skills/memory/src/memory/__init__.py"
MODULE_NAME = "_memory_skill_payloads_under_test"


def load_memory_skill(host_request):
    runtime = types.ModuleType("rlm")
    runtime.host_request = host_request
    spec = importlib.util.spec_from_file_location(MODULE_NAME, SKILL_PATH)
    module = importlib.util.module_from_spec(spec)
    # Only the import sees the fake runtime; other tests retain their modules.
    with mock.patch.dict(sys.modules, {"rlm": runtime}):
        spec.loader.exec_module(module)
    return module


def source_reference(name="synthetic-file"):
    return {
        "id": name,
        "origin": "file",
        "sha256": "a" * 64,
        "uri": "fixtures/provenance.txt",
        "revision": "synthetic-revision",
        "projectPath": "synthetic-project",
    }


class MemorySkillLoaderTests(unittest.TestCase):
    def test_loads_repository_module_without_registering_fake_modules(self):
        with mock.patch.dict(sys.modules):
            for name in ("rlm", "memory", MODULE_NAME):
                sys.modules.pop(name, None)
            host_request = mock.AsyncMock()
            memory = load_memory_skill(host_request)
            self.assertEqual(Path(memory.__file__).resolve(), SKILL_PATH.resolve())
            self.assertEqual(memory.apply.__module__, MODULE_NAME)
            self.assertIs(memory.host_request, host_request)
            for name in ("rlm", "memory", MODULE_NAME):
                self.assertNotIn(name, sys.modules)

    def test_existing_runtime_and_memory_modules_are_restored(self):
        originals = {name: types.ModuleType(name) for name in ("rlm", "memory", MODULE_NAME)}
        with mock.patch.dict(sys.modules, originals):
            host_request = mock.AsyncMock()
            memory = load_memory_skill(host_request)
            self.assertIs(memory.host_request, host_request)
            for name, module in originals.items():
                self.assertIs(sys.modules[name], module)

    def test_each_load_keeps_its_own_fake_host(self):
        first_host, second_host = mock.AsyncMock(), mock.AsyncMock()
        first = load_memory_skill(first_host)
        second = load_memory_skill(second_host)
        self.assertIsNot(first, second)
        self.assertIs(first.host_request, first_host)
        self.assertIs(second.host_request, second_host)


class MemoryApplyPayloadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.response = {"origin": "synthetic-memory", "result": {"revision": 18}}
        self.host_request = mock.AsyncMock(return_value=self.response)
        self.memory = load_memory_skill(self.host_request)
        self.proposal = {
            "summary": "Synthetic provenance update",
            "rationale": "Offline payload regression",
            "expectedOutcome": "Keep existing source attribution",
            "edits": [{
                "action": "update",
                "kind": "memory",
                "id": "synthetic-memory",
                "title": "Synthetic title",
                "content": "Synthetic content",
            }],
        }
        self.original_proposal = copy.deepcopy(self.proposal)
        self.event_id = "synthetic-event"
        self.revision = 17

    def assert_apply_payload(self, *, host=False, automatic=False, **source_payload):
        expected = {
            "action": "apply",
            "proposal": self.proposal,
            "eventId": self.event_id,
            "revision": self.revision,
            "host": host,
            "automatic": automatic,
            **source_payload,
        }
        self.host_request.assert_called_once_with("memory.request", expected)
        self.host_request.assert_awaited_once_with("memory.request", expected)
        payload = self.host_request.await_args.args[1]
        self.assertIs(payload["proposal"], self.proposal)
        self.assertIs(payload["host"], host)
        self.assertIs(payload["automatic"], automatic)
        self.assertEqual(self.proposal, self.original_proposal)
        if "sources" in source_payload:
            self.assertIs(payload["sources"], source_payload["sources"])
        else:
            self.assertNotIn("sources", payload)

    async def test_omitted_and_none_sources_omit_key(self):
        for kwargs in ({}, {"sources": None}):
            with self.subTest(kwargs=kwargs):
                self.host_request.reset_mock()
                result = await self.memory.apply(
                    self.proposal, event_id=self.event_id, revision=self.revision, **kwargs
                )
                self.assertIs(result, self.response)
                self.assert_apply_payload()

    async def test_explicit_empty_sources_are_forwarded_for_clear(self):
        sources = []
        result = await self.memory.apply(
            self.proposal, event_id=self.event_id, revision=self.revision, sources=sources
        )
        self.assertIs(result, self.response)
        self.assert_apply_payload(sources=sources)
        self.assertEqual(sources, [])

    async def test_nonempty_sources_are_forwarded_without_copy_or_mutation(self):
        for sources in ([source_reference()], [source_reference("first"), source_reference("second")]):
            with self.subTest(sources=sources):
                self.host_request.reset_mock()
                original = copy.deepcopy(sources)
                result = await self.memory.apply(
                    self.proposal, event_id=self.event_id, revision=self.revision, sources=sources
                )
                self.assertIs(result, self.response)
                self.assert_apply_payload(sources=sources)
                self.assertEqual(sources, original)

    async def test_invalid_sources_reach_host_validation_unchanged(self):
        invalid_values = (
            False, 0, 0.0, "", {}, True, 1, "not-an-array",
            {"sources": []}, [None], [{"id": "missing-hash-and-origin"}],
        )
        for sources in invalid_values:
            with self.subTest(sources=sources, value_type=type(sources).__name__):
                self.host_request.reset_mock()
                error = RuntimeError("Synthetic host source validation failure")
                self.host_request.side_effect = error
                original = copy.deepcopy(sources)
                with self.assertRaises(RuntimeError) as caught:
                    await self.memory.apply(
                        self.proposal, event_id=self.event_id, revision=self.revision, sources=sources
                    )
                self.assertIs(caught.exception, error)
                self.assert_apply_payload(sources=sources)
                self.assertEqual(sources, original)

    async def test_other_fields_are_unchanged_for_all_source_modes(self):
        source_modes = ({}, {"sources": None}, {"sources": []}, {"sources": [source_reference()]})
        for kwargs in source_modes:
            for host in (False, True):
                for automatic in (False, True):
                    with self.subTest(kwargs=kwargs, host=host, automatic=automatic):
                        self.host_request.reset_mock()
                        self.event_id = "synthetic-custom-event"
                        self.revision = 0
                        result = await self.memory.apply(
                            proposal=self.proposal, event_id=self.event_id, revision=self.revision,
                            host=host, automatic=automatic, **kwargs
                        )
                        self.assertIs(result, self.response)
                        expected_sources = kwargs if kwargs.get("sources") is not None else {}
                        self.assert_apply_payload(host=host, automatic=automatic, **expected_sources)

    def test_public_signature_is_unchanged(self):
        parameter = inspect.Parameter
        expected = inspect.Signature([
            parameter("proposal", parameter.POSITIONAL_OR_KEYWORD, annotation=dict),
            parameter("event_id", parameter.KEYWORD_ONLY, annotation=str),
            parameter("revision", parameter.KEYWORD_ONLY, annotation=int),
            parameter("sources", parameter.KEYWORD_ONLY, default=None),
            parameter("host", parameter.KEYWORD_ONLY, default=False),
            parameter("automatic", parameter.KEYWORD_ONLY, default=False),
        ])
        self.assertEqual(inspect.signature(self.memory.apply), expected)
        self.assertTrue(inspect.iscoroutinefunction(self.memory.apply))
        self.host_request.assert_not_called()

    async def test_required_keywords_are_still_required(self):
        for kwargs in ({}, {"event_id": self.event_id}, {"revision": self.revision}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    await self.memory.apply(self.proposal, **kwargs)
                self.host_request.assert_not_called()

    async def test_metadata_arguments_remain_keyword_only(self):
        for extra_args in ((self.event_id,), (self.event_id, self.revision),
                           (self.event_id, self.revision, [], False, False)):
            with self.subTest(extra_args=extra_args):
                with self.assertRaises(TypeError):
                    await self.memory.apply(self.proposal, *extra_args)
                self.host_request.assert_not_called()

    async def test_unknown_keywords_are_rejected_before_host_call(self):
        for kwargs in ({"eventId": "wrong-spelling"}, {"unknown": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    await self.memory.apply(
                        self.proposal, event_id=self.event_id, revision=self.revision, **kwargs
                    )
                self.host_request.assert_not_called()

    async def test_host_errors_and_cancellation_propagate_without_retry(self):
        source_modes = ({}, {"sources": None}, {"sources": []}, {"sources": [source_reference()]})
        for kwargs in source_modes:
            for error in (RuntimeError("Synthetic conflict"), ValueError("Synthetic rejection"),
                          asyncio.CancelledError("Synthetic cancellation")):
                with self.subTest(kwargs=kwargs, error=type(error).__name__):
                    self.host_request.reset_mock()
                    self.host_request.side_effect = error
                    with self.assertRaises(type(error)) as caught:
                        await self.memory.apply(
                            self.proposal, event_id=self.event_id, revision=self.revision, **kwargs
                        )
                    self.assertIs(caught.exception, error)
                    expected_sources = kwargs if kwargs.get("sources") is not None else {}
                    self.assert_apply_payload(**expected_sources)

    async def test_handoff_source_semantics_remain_unchanged(self):
        source_modes = (
            {}, {"sources": None}, {"sources": []}, {"sources": False}, {"sources": 0},
            {"sources": ""}, {"sources": {}}, {"sources": [source_reference()]},
        )
        for kwargs in source_modes:
            for automatic in (False, True):
                with self.subTest(kwargs=kwargs, automatic=automatic):
                    self.host_request.reset_mock()
                    result = await self.memory.handoff(
                        "Synthetic task", "Synthetic state", "Synthetic decisions", "Synthetic unresolved",
                        event_id=self.event_id, revision=self.revision, automatic=automatic, **kwargs
                    )
                    expected = {
                        "action": "handoff",
                        "task": "Synthetic task",
                        "state": "Synthetic state",
                        "decisions": "Synthetic decisions",
                        "unresolved": "Synthetic unresolved",
                        "eventId": self.event_id,
                        "revision": self.revision,
                        "sources": kwargs.get("sources") or [],
                        "automatic": automatic,
                    }
                    self.host_request.assert_called_once_with("memory.request", expected)
                    self.host_request.assert_awaited_once_with("memory.request", expected)
                    self.assertIs(result, self.response)
                    if kwargs.get("sources"):
                        self.assertIs(self.host_request.await_args.args[1]["sources"], kwargs["sources"])


if __name__ == "__main__":
    unittest.main()
