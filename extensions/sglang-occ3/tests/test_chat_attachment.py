"""Offline format-only tests, NOT installed SGLang, CUDA, TP or MTP proof.

Exact hash-pinned native methods are AST-compiled with explicit dependency
stubs. Chat/template/engine/parser dependencies are fixtures; Starlette ASGI
is real. Set OCC3_SGLANG_SOURCE to the pinned source checkout for these tests.
"""
from __future__ import annotations

import ast
import asyncio
import functools
import hashlib
import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from occ3_sglang.attachment_contract import (
    AttachmentError, AuthenticatedContext, BackendReadiness, ConsumeWorkerDescriptor, InferenceAttachment,
    Reservation, RECEIPT_HEADER, canonical,
)
from occ3_sglang.chat_adapter import BufferProfile, ChatAttachmentAdapter
from occ3_sglang.use_receipts import TerminalUseEvidence

SOURCE_HASHES = {
    "entrypoints/openai/serving_base.py": "57feb35bfce5f14349fea2d7b1f6ab4a1d36c0ca7ddabeaa388a7d44cf3f8609",
    "entrypoints/openai/serving_chat.py": "a0886cdf125b9c8c98c4618a4a6afee7020095623b7ee6cd874b67b94f9efb36",
    "plugins/hook_registry.py": "0144d8c3e5ee294322894f88b8ad3bd49edad15c124650e4e834d0366815dc7d",
}


def pinned_symbol(relative, class_name, symbol, namespace):
    root = os.environ.get("OCC3_SGLANG_SOURCE")
    if not root:
        pytest.skip("OCC3_SGLANG_SOURCE unavailable: no installed native-source claim")
    file = Path(root) / "python/sglang/srt" / relative
    text = file.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode()).hexdigest() == SOURCE_HASHES[relative]
    tree = ast.parse(text)
    scope = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == class_name).body if class_name else tree.body
    node = next(node for node in scope if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == symbol)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(file), "exec"), namespace)
    return namespace[symbol]


def attachment(**changes):
    value = dict(protocol_version=1, request_id="source-message-request", message_id="message-1",
                 connector_id="connector-1", receiver_session_id="receiver-1", model_fingerprint="sha256:fixture",
                 runtime_id="runtime-1", runtime_generation=4, state_handle="PRIVATE_HANDLE_1",
                 expires_at_epoch=int(time.time()) + 300, method="statebridge-fixture-v1")
    value.update(changes)
    return InferenceAttachment(**value)


class FixtureBackend:
    """Deterministic dependency stub. Its booleans are NOT real device evidence."""
    def __init__(self):
        self.reserves = []
        self.binds = []
        self.finishes = []
        self.retires = []
        self.required_seconds = .1
        self.max_bytes = 65536
        self.reserve_gate = None
        self.finish_gate = None
        self.mutate_evidence = None
        self.mutate_bind = None
        self.mutate_descriptor = None
        self.secret_failure = None
        self.seen = set()

    def readiness(self):
        return BackendReadiness(True, True, True, True, True, True)

    async def reserve(self, a, identity, attempt_id, native):
        self.reserves.append((a, identity, attempt_id, native))
        if a.state_handle in self.seen:
            raise AttachmentError("replay_forbidden")
        self.seen.add(a.state_handle)
        if self.reserve_gate:
            await self.reserve_gate.wait()
        if self.secret_failure == "reserve":
            raise RuntimeError(a.state_handle)
        return Reservation(attempt_id, a.binding_digest, native.request.rid,
                           identity.principal_digest, self.required_seconds,
                           self.max_bytes, native.max_new_tokens, "fixture-reservation")

    def bind(self, reservation, native):
        self.binds.append(reservation)
        if self.mutate_bind:
            self.mutate_bind(native)
        a, identity, _, native_view = self.reserves[-1]
        native.occ3_worker_descriptor = ConsumeWorkerDescriptor(
            reservation.attempt_id, reservation.reservation_id, reservation.native_rid,
            identity.principal_digest, a.binding_digest, a.receiver_session_id,
            a.message_id, a.request_id, a.state_handle, 1, a.runtime_generation, "fixture-source-commit",
            a.runtime_id, a.model_fingerprint, a.method, "b" * 64, 8, 8, 6144,
            "bfloat16", native_view.deadline_monotonic)
        if self.mutate_descriptor:
            native.occ3_worker_descriptor = self.mutate_descriptor(native.occ3_worker_descriptor)
        # This descriptor is synthetic and never reaches an actual scheduler.

    async def finish(self, reservation, completion):
        self.finishes.append(completion)
        if self.finish_gate:
            await self.finish_gate.wait()
        a = self.reserves[-1][0]
        if self.secret_failure == "finish":
            raise RuntimeError(a.state_handle)
        evidence = TerminalUseEvidence(
            reservation.attempt_id, a.binding_digest, reservation.native_rid,
            reservation.principal_digest, "fixture-source-commit", "fixture-injection-commit",
            a.runtime_id, a.runtime_generation, a.model_fingerprint, a.receiver_session_id,
            completion.output_sha256, 8, 8, 8, True, False, 0, True, True, reservation.reservation_id)
        return self.mutate_evidence(evidence) if self.mutate_evidence else evidence

    async def retire(self, attempt_id, native_rid, dispatched):
        self.retires.append((attempt_id, native_rid, dispatched))


class NativeInput(NS):
    pass


class ChatRequest(NS):
    def to_sampling_params(self, **kwargs):
        self.sampling_arguments = kwargs
        return {"max_new_tokens": self.max_tokens, "temperature": .7, "n": self.n,
                "reasoning_effort": self.reasoning_effort}


def chat_request(**changes):
    data = dict(model="fixture-model", n=1, stream=True, input_ids=None, max_tokens=32768,
                reasoning_effort="max", chat_template_kwargs={"fixture": True},
                return_input_ids_in_sglext=False, return_output_ids_in_sglext=False,
                return_prompt_token_ids=False, return_token_ids=False, return_meta_info=False,
                logprobs=False, top_logprobs=None, return_sampling_mask=False,
                sampling_logprobs_mode=None, lora_path=None, bootstrap_host=None,
                bootstrap_port=None, bootstrap_room=None, routed_dp_rank=None,
                disagg_prefill_dp_rank=None, return_hidden_states=False, return_routed_experts=False,
                routed_experts_start_len=None, rid=None, session_id=None, extra_key=None,
                cache_salt=None, priority=None, custom_logit_processor=None,
                messages=[{"role": "system", "content": "normal receiver instructions"},
                          {"role": "tool", "tool_call_id": "prior-tool", "content": "normal receiver tool result"}],
                tools=[{"type": "function", "function": {"name": "handoff", "parameters": {}}}])
    data.update(changes)
    return ChatRequest(**data)


def raw_request(body):
    queue = asyncio.Queue()
    queue.put_nowait({"type": "http.request", "body": body, "more_body": False})
    scope = {"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
             "query_string": b"", "headers": [(b"content-type", b"application/json")],
             "server": ("127.0.0.1", 80), "client": ("127.0.0.1", 9999)}
    return Request(scope, queue.get), queue


@pytest.fixture
def harness():
    backend = FixtureBackend()
    principal = "a" * 64
    async def authorize(raw):
        return AuthenticatedContext(principal, "receiver-1", "fixture-model", "sha256:fixture",
                                    "runtime-1", 4, time.monotonic() + 60)
    adapter = ChatAttachmentAdapter(backend, authorize=authorize,
                                   profile=BufferProfile(max_buffer_bytes=65536, total_working_bytes=524288,
                                                         max_turn_seconds=60, cleanup_seconds=.1,
                                                         disconnect_interval=.002,
                                                         capture_busy_wait_seconds=0))
    class HookType:
        AROUND, AFTER, BEFORE, REPLACE = "around", "after", "before", "replace"
    ns = dict(functools=functools, HookType=HookType)
    wrap = pinned_symbol("plugins/hook_registry.py", None, "_wrap_fn", ns)
    class DS32EncodingError(Exception): pass
    namespace = dict(
        monotonic_time=time.monotonic, time=time, HTTPException=HTTPException, DS32EncodingError=DS32EncodingError,
        logger=logging.getLogger("occ3.fixture.native"), GenerateReqInput=NativeInput,
        EmbeddingReqInput=type("EmbeddingReqInput", (), {}), StreamingResponse=StreamingResponse,
        uses_hunyuan_reasoning_effort=lambda *args: True,
        set_request_reasoning_end_token_ids=lambda *args: None,
        _extract_max_dynamic_patch=lambda request: (None, None), _build_video_config=lambda request: None,
        envs=NS(SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=NS(get=lambda: False)),
    )
    base_handle = pinned_symbol("entrypoints/openai/serving_base.py", "OpenAIServingBase", "handle_request", namespace)
    convert = pinned_symbol("entrypoints/openai/serving_chat.py", "OpenAIServingChat", "_convert_to_internal_request", namespace)
    stream_handle = pinned_symbol("entrypoints/openai/serving_chat.py", "OpenAIServingChat", "_handle_streaming_request", namespace)
    nonstream_handle = pinned_symbol("entrypoints/openai/serving_chat.py", "OpenAIServingChat", "_handle_non_streaming_request", namespace)

    class TM:
        def __init__(self):
            self.request_logger = NS(log_requests=False)
            self.model_config = NS(is_multimodal=False)
            self.calls, self.aborts, self.closed = [], [], 0
            self.first_gate = None
        async def generate_request(self, obj, request=None):
            self.calls.append(obj)
            try:
                if self.first_gate:
                    await self.first_gate.wait()
                yield {"text": "fixture", "meta_info": {"id": obj.rid, "finish_reason": {"type": "stop"}}}
            finally:
                self.closed += 1
        def abort_request(self, rid): self.aborts.append(rid)
        def create_abort_task(self, obj): return None
    TM.generate_request = wrap(TM.generate_request, adapter.generate_around, HookType.AROUND)
    class Chat:
        def __init__(self):
            self.tokenizer_manager = TM()
            self.reasoning_parser = "fixture-reasoning"
            self.template_manager = NS(reasoning_config="unchanged")
            self.is_gpt_oss = False
            self.default_sampling_params = {"fixture-default": True}
            self.chat_encoding_spec = "fixture"
            self.converted = []
            self.frames = []
            self.buffer_gate = None
            self.stream_tail = None
            self.extra_headers = {}
        def _validate_request(self, request): return None
        def create_error_response(self, message, **kwargs):
            return JSONResponse({"error": {"message": message}}, status_code=kwargs.get("status_code", 400))
        def _process_messages(self, request, multimodal):
            self.converted.append((request.messages, request.tools, request.reasoning_effort))
            return NS(prompt_ids=[100, 200, 300], stop=["STOP"], tool_call_constraint="fixture-tools",
                      reasoning_end_token_ids=[900], image_data=None, video_data=None, audio_data=None,
                      modalities=None, require_reasoning=True)
        def _engine_prompt(self, processed, multimodal): return "input_ids", processed.prompt_ids
        def extract_custom_labels(self, raw): return None
        def extract_routed_dp_rank_from_header(self, raw, body): return body
        def _resolve_lora_path(self, model, lora): return lora
        def extract_routing_key(self, raw): return None
        def _should_return_input_ids(self, request): return False
        async def _generate_chat_stream(self, obj, request, raw):
            async for output in self.tokenizer_manager.generate_request(obj, raw):
                for delta in ({"role": "assistant"}, {"reasoning_content": "fixture reasoning"},
                              {"tool_calls": [{"index": 0, "id": "call-fixture", "type": "function",
                                               "function": {"name": "handoff", "arguments": "{\"x\":1}"}}]}):
                    frame = "data: " + json.dumps({"id": obj.rid, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n"
                    self.frames.append(frame.encode())
                    yield frame
                    if self.buffer_gate:
                        await self.buffer_gate.wait()
                tail = self.stream_tail
                if tail is None:
                    tail = ["data: " + json.dumps({"id": obj.rid, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}) + "\n\n", "data: [DONE]\n\n"]
                for frame in tail:
                    self.frames.append(frame.encode())
                    yield frame
        def _build_chat_response(self, request, ret, created):
            return JSONResponse({"id": ret[0]["meta_info"]["id"], "choices": [{"index": 0,
                "message": {"role": "assistant", "reasoning_content": "fixture reasoning",
                            "tool_calls": [{"id": "call-fixture"}]}, "finish_reason": "tool_calls"}]})
    # Original pinned handle and converter bodies, not rewritten lookalikes.
    Chat._convert_to_internal_request = wrap(convert, adapter.convert_after, HookType.AFTER)
    Chat._handle_streaming_request = stream_handle
    Chat._handle_non_streaming_request = nonstream_handle
    Chat.handle_request = wrap(base_handle, adapter.handle_around, HookType.AROUND)
    chat = Chat()
    return NS(adapter=adapter, backend=backend, chat=chat, tm=chat.tokenizer_manager,
              base_handle=base_handle, convert=convert, wrap=wrap, HookType=HookType)


async def call(harness, a=None, request=None, body=None):
    request = request or chat_request()
    a = a or attachment()
    raw, queue = raw_request(body if body is not None else canonical({"occ3": a.wire()}))
    response = await harness.chat.handle_request(request, raw)
    return response, raw, queue


@pytest.mark.asyncio
async def test_native_converter_and_bytes_preserved(harness):
    request = chat_request()
    response, _, _ = await call(harness, request=request)
    assert response.status_code == 200, response.body
    assert response.body == b"".join(harness.chat.frames)
    assert len(harness.tm.calls) == len(harness.backend.reserves) == len(harness.backend.finishes) == 1
    native = harness.tm.calls[0]
    assert native.input_ids == [100, 200, 300]
    assert native.sampling_params == {"max_new_tokens": 32768, "temperature": .7, "n": 1, "reasoning_effort": "max"}
    assert harness.chat.converted == [(request.messages, request.tools, "max")]
    receipt = json.loads(response.headers[RECEIPT_HEADER])
    assert receipt == {"attachment": harness.backend.reserves[0][0].wire(), "consumed": True,
                       "fallback_used": False, "injected_state": True}
    assert response.headers["cache-control"] == "no-store"
    assert harness.tm.closed == 1 and not harness.tm.aborts
    assert harness.adapter._active == 0


@pytest.mark.asyncio
async def test_nonstream_same_native_response_and_closed_iterator(harness):
    response, _, _ = await call(harness, request=chat_request(stream=False))
    assert response.status_code == 200
    assert json.loads(response.body)["choices"][0]["message"]["tool_calls"] == [{"id": "call-fixture"}]
    assert harness.tm.closed == 1
    assert not harness.backend.finishes[0].streaming


@pytest.mark.asyncio
async def test_absent_attachment_returns_original_object_and_error_unchanged(harness):
    raw, _ = raw_request(b'{"unknown":1,"unknown":2}')
    sentinel = object()
    async def original(*args): return sentinel
    assert await harness.adapter.handle_around(original, harness.chat, chat_request(), raw) is sentinel
    async def failing(*args): raise LookupError("ordinary-native-error")
    with pytest.raises(LookupError, match="ordinary-native-error"):
        await harness.adapter.handle_around(failing, harness.chat, chat_request(), raw)
    assert not harness.backend.reserves
    assert harness.adapter.convert_after(sentinel, harness.chat, chat_request(), raw) is sentinel
    def original_gen(*args): return sentinel
    assert harness.adapter.generate_around(original_gen, harness.tm, object(), raw) is sentinel


@pytest.mark.asyncio
async def test_no_attachment_actual_pinned_handler_bypass(harness):
    response, _, _ = await call(harness, request=chat_request(stream=False), body=b'{}')
    assert response.status_code == 200
    assert RECEIPT_HEADER not in response.headers
    assert len(harness.chat.converted) == len(harness.tm.calls) == 1
    assert not harness.backend.reserves


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("receiver_session_id", "wrong"), ("model_fingerprint", "wrong"),
                                        ("runtime_id", "wrong"), ("runtime_generation", 3)])
async def test_identity_mismatch_before_conversion(harness, field, value):
    response, _, _ = await call(harness, attachment(**{field: value}))
    assert response.status_code == 401
    assert not harness.chat.converted and not harness.tm.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("n", 2), ("input_ids", [1]), ("cache_salt", "caller-salt"),
                                        ("extra_key", "caller-key"), ("session_id", "engine-session"),
                                        ("routed_dp_rank", 0), ("lora_path", "adapter")])
async def test_conflicting_profiles_refused_before_dispatch(harness, field, value):
    response, _, _ = await call(harness, request=chat_request(**{field: value}))
    assert response.status_code != 200 and not harness.tm.calls


@pytest.mark.asyncio
async def test_fail_closed_default_and_missing_hook(harness):
    harness.adapter.backend = __import__("occ3_sglang.chat_adapter", fromlist=["UnsupportedBackend"]).UnsupportedBackend()
    response, _, _ = await call(harness)
    assert response.status_code == 503 and not harness.tm.calls
    harness.adapter.backend = harness.backend
    harness.chat._convert_to_internal_request = lambda *args: None
    response, _, _ = await call(harness)
    assert response.status_code == 503 and not harness.tm.calls


@pytest.mark.asyncio
async def test_replay_never_generates_twice(harness):
    a = attachment()
    first, _, _ = await call(harness, a)
    second, _, _ = await call(harness, a)
    assert first.status_code == 200 and second.status_code != 200
    assert len(harness.tm.calls) == len(harness.backend.reserves) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", [lambda obj: obj.sampling_params.update(max_new_tokens=1),
                                     lambda obj: obj.input_ids.append(99)])
async def test_backend_cannot_reduce_budget_or_replace_prompt(harness, mutation):
    harness.backend.mutate_bind = mutation
    response, _, _ = await call(harness)
    assert response.status_code != 200 and not harness.tm.calls
    assert len(harness.backend.retires) == 1


@pytest.mark.asyncio
async def test_full_budget_deadline_refusal_before_dispatch(harness):
    harness.backend.required_seconds = 100
    response, _, _ = await call(harness)
    assert response.status_code != 200 and not harness.tm.calls
    assert harness.backend.reserves[0][3].max_new_tokens == 32768
    assert len(harness.backend.retires) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["reserve", "finish"])
async def test_backend_exception_never_leaks_handle(harness, phase, caplog):
    harness.backend.secret_failure = phase
    a = attachment()
    response, _, _ = await call(harness, a)
    assert response.status_code != 200
    assert a.state_handle not in response.body.decode() + caplog.text
    assert a.state_handle not in repr(a)


@pytest.mark.asyncio
async def test_logging_profile_refused_without_mutating_logger(harness):
    harness.tm.request_logger.log_requests = True
    response, _, _ = await call(harness)
    assert response.status_code != 200 and not harness.tm.calls
    assert harness.tm.request_logger.log_requests is True


@pytest.mark.asyncio
async def test_buffer_bound_suppresses_all_output(harness):
    harness.backend.max_bytes = 5
    response, _, _ = await call(harness)
    assert response.status_code == 413
    assert RECEIPT_HEADER not in response.headers
    assert len(harness.backend.finishes) == 0
    assert harness.tm.closed == 1 and len(harness.tm.aborts) == 1


@pytest.mark.asyncio
async def test_headers_not_sent_until_terminal_backend_evidence(harness):
    gate = harness.backend.finish_gate = asyncio.Event()
    raw, _ = raw_request(canonical({"occ3": attachment().wire()}))
    sent = []
    async def send(event): sent.append(event)
    async def run():
        response = await harness.chat.handle_request(chat_request(), raw)
        await response(raw.scope, raw.receive, send)
    task = asyncio.create_task(run())
    for _ in range(100):
        if harness.backend.finishes: break
        await asyncio.sleep(0)
    assert harness.backend.finishes and sent == []
    gate.set()
    await task
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b"".join(harness.chat.frames)


def test_hook_registration_is_explicit_only(harness):
    registered = []
    registry = NS(register=lambda *args: registered.append(args))
    assert registered == []
    harness.adapter.register_hooks(registry, harness.HookType)
    assert len(registered) == 3
    assert all("/recv" not in target for target, _, _ in registered)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("attempt_id", "wrong"), ("reservation_id", "wrong"), ("native_rid", "wrong"),
    ("principal_digest", "c" * 64), ("attachment_digest", "d" * 64),
    ("receiver_session_id", "wrong"), ("message_id", "wrong"), ("consume_nonce", "wrong"), ("opaque_handle", "wrong"),
    ("runtime_epoch", "wrong"), ("runtime_generation", 99), ("model_fingerprint", "wrong"),
    ("method_id", "wrong"),
])
async def test_worker_descriptor_binding_checked_before_dispatch(harness, field, value):
    harness.backend.mutate_descriptor = lambda d: replace(d, **{field: value})
    response, _, _ = await call(harness)
    assert response.status_code != 200 and not harness.tm.calls
    assert len(harness.backend.retires) == 1


@pytest.mark.asyncio
async def test_no_descriptor_no_generation(harness):
    harness.backend.mutate_descriptor = lambda _: None
    response, _, _ = await call(harness)
    assert response.status_code != 200 and not harness.tm.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["frontend", "tokenizer_descriptor", "scheduler", "all_workers",
                                  "terminal_use", "authenticated_owner"])
async def test_each_role_must_be_ready(harness, field):
    ready = harness.backend.readiness()
    harness.backend.readiness = lambda: replace(ready, **{field: False})
    response, _, _ = await call(harness)
    assert response.status_code == 503 and not harness.chat.converted


@pytest.mark.asyncio
async def test_oversized_header_refused_before_reservation(harness):
    fields = attachment().wire()
    for key in fields:
        if isinstance(fields[key], str): fields[key] = "x" * 256
    response, _, _ = await call(harness, InferenceAttachment.parse(fields))
    assert response.status_code == 413
    assert not harness.backend.reserves and not harness.tm.calls


@pytest.mark.asyncio
async def test_second_native_generation_is_never_dispatched(harness):
    original = harness.chat._generate_chat_stream
    async def twice(obj, request, raw):
        async for frame in original(obj, request, raw): yield frame
        async for _ in harness.tm.generate_request(obj, raw): pass
    harness.chat._generate_chat_stream = twice
    response, _, _ = await call(harness)
    assert response.status_code != 200 and len(harness.tm.calls) == 1
    assert RECEIPT_HEADER not in response.headers


@pytest.mark.asyncio
async def test_reserve_cannot_modify_context_before_bind(harness):
    reserve = harness.backend.reserve
    async def mutate(*args):
        result = await reserve(*args)
        args[-1].request.input_ids.append(777)
        return result
    harness.backend.reserve = mutate
    response, _, _ = await call(harness)
    assert response.status_code != 200 and not harness.tm.calls


@pytest.mark.asyncio
async def test_bind_cannot_modify_other_native_flags(harness):
    harness.backend.mutate_bind = lambda obj: setattr(obj, "stream", False)
    response, _, _ = await call(harness)
    assert response.status_code != 200 and not harness.tm.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,status", [("client_disconnected", 499), ("deadline_exceeded", 504)])
async def test_cut_turn_names_guard_reason_server_side(harness, reason, status, caplog):
    """Name the two guard outcomes without changing the public error body.
    Existing live evidence did not identify which outcome caused the cut."""
    from test_attachment_cancellation import spin_until  # late: it imports this module
    h = harness
    if status == 504:
        h.adapter.profile = replace(h.adapter.profile, max_turn_seconds=.06, deadline_margin_seconds=.001)
        h.backend.required_seconds = .001
    h.chat.buffer_gate = asyncio.Event()  # hold the buffered body like the live cut
    raw, queue = raw_request(canonical({"occ3": attachment().wire()}))
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.chat_adapter"):
        task = asyncio.create_task(h.chat.handle_request(chat_request(), raw))
        await spin_until(lambda: h.chat.frames)
        if status == 499:
            queue.put_nowait({"type": "http.disconnect"})
        response = await asyncio.wait_for(task, 2)
    assert response.status_code == status
    assert json.loads(response.body) == {"error": {"message": reason,
                                                   "type": "occ3_attachment_error", "code": status}}
    assert RECEIPT_HEADER not in response.headers
    lines = [record.getMessage() for record in caplog.records
             if record.name == "occ3_sglang.chat_adapter" and "native turn ended" in record.getMessage()]
    assert len(lines) == 1, "expected exactly one bounded diagnostic for the cut turn"
    assert f"reason={reason}" in lines[0] and f"status={status}" in lines[0]
    assert "dispatched=True" in lines[0]
    # No prompt text, attachment fields or handles reach the diagnostic.
    assert "PRIVATE_HANDLE_1" not in caplog.text and "source-message-request" not in caplog.text
    assert "normal receiver instructions" not in caplog.text
    assert h.adapter._active == 0 and len(h.backend.retires) == 1


@pytest.mark.asyncio
async def test_plain_refusal_is_not_logged_as_a_cut_turn(harness, caplog):
    """Only the two guard codes are named; an ordinary refusal adds no such line."""
    harness.backend.max_bytes = 5
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.chat_adapter"):
        response, _, _ = await call(harness)
    assert response.status_code == 413
    assert not [record for record in caplog.records if "native turn ended" in record.getMessage()]
