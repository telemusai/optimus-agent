"""Semantic leakage and matched negative-control tests for the OCC-3 receiver payload.

Offline only: no GPU, no network, no SGLang install, no Rust build. Pure Python.

Two assertion kinds live here
-----------------------------
REAL   Runs ``occ3_sglang`` module code: ``attachment_contract``, ``use_receipts``
       and the real ``ChatAttachmentAdapter`` driven through a local hook
       harness. The harness re-uses the shared wire fixture from
       ``test_chat_attachment.py`` and mirrors the hash-pinned
       ``sglang.srt.plugins.hook_registry._wrap_fn`` contract, so these
       assertions hold without ``OCC3_SGLANG_SOURCE`` and never claim an
       installed-SGLang result.
MIRROR The receiver prompt is rendered by Rust (``crates/pi-coding-agent``);
       Python cannot import it. ``delivery_record_text``, ``ordinary_payload``,
       ``prepared_handoff_accepts`` and ``positive_case_passed`` are literal
       mirrors of the cited functions. Every mirror names its source
       ``file:line``. If that source changes, this file must change with it.

Goal of the suite: a "positive" OCC-3 trial must not be able to succeed because
the private fact or the expected answer reached the receiver as ordinary text,
and the no-state / unrelated-state controls must be provably matched.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

from occ3_sglang.attachment_contract import (
    AttachmentError, AuthenticatedContext, BackendReadiness, ConsumeWorkerDescriptor,
    InferenceAttachment, RECEIPT_HEADER, Reservation, canonical, parse_opt_in,
)
from occ3_sglang.chat_adapter import BufferProfile, ChatAttachmentAdapter
from occ3_sglang.use_receipts import TerminalUseEvidence, receipt_bytes, validate_evidence
from test_chat_attachment import attachment as wire_fixture

# ---------------------------------------------------------------------------
# Pinned wire allowlist (REAL): the eleven InferenceAttachment fields.
# attachment_contract.py:81-117 and pi-ai openai_completions.rs:2210 require the
# request field to equal this dict exactly, so the tuple is a fixed allowlist.
# ---------------------------------------------------------------------------
ALLOWED_ATTACHMENT_KEYS = (
    "connector_id", "expires_at_epoch", "message_id", "method", "model_fingerprint",
    "protocol_version", "receiver_session_id", "request_id", "runtime_generation",
    "runtime_id", "state_handle",
)
# receipt_bytes() payload keys (use_receipts.py:54-59), sorted by canonical().
ALLOWED_RECEIPT_KEYS = ("attachment", "consumed", "fallback_used", "injected_state")

# ---------------------------------------------------------------------------
# MIRROR constants. Strings are byte-for-byte copies of the Rust sources.
# ---------------------------------------------------------------------------
# agent_messages.rs:2250 and occ3_connectors/sglang.rs:934: the only accepted
# native control summary. It is what the receiver reads as ordinary text.
CONTROL_SUMMARY = "You are the receiver of this attached native state. It is the sender's context, not your own history. As the receiver, answer your current task using that state."
# examples/occ3_native_phase_suite.rs:1564 (no-state control prompt).
NO_STATE_TASK = "Continue the authorized task. No state attachment is available for this negative control."
# examples/occ3_native_handoff.rs:74 (the matched task text used by the handoff suite).
MATCHED_TASK = "Continue the authorized task using only the available information."
# examples/occ3_native_phase_suite.rs:1440 == :1539, byte-identical receiver system prompt.
RECEIVER_SYSTEM_PROMPT = ("Read the private vault code from the attached native state. "
                          "Reply with that code. If unavailable say UNKNOWN. Do not call tools.")
# examples/occ3_native_handoff.rs:67-70: the semantic (animal-category) fixture.
CATEGORY_FACTS = (("sparrow", "BIRD"), ("otter", "MAMMAL"), ("carp", "FISH"), ("iguana", "REPTILE"))
CATEGORY_QUESTION = ("Identify the biological class of the private animal from the available "
                     "information. Reply with exactly BIRD, MAMMAL, FISH, or REPTILE. If the private "
                     "animal is unavailable reply UNKNOWN. Do not call tools.")
# examples/occ3_native_phase_suite.rs:74 and :81.
UNKNOWN_ANSWER = "UNKNOWN"
DEFAULT_MAX_TOKENS = 131072
# core/occ/occ1.rs:28, core/occ/occ2.rs:66: fallback protocol headers.
OCC1_HEADER = "OCC/1"
OCC2_HEADER = "OCC/2"
AGENT_MESSAGE_SOURCE = "agent_message"  # agent_messages.rs:61
HANDOFF_TOOL = "occ3_handoff"           # core/agent_session/occ3_state.rs:27

# Private fixture values. The nonce is the expected answer of the vault case.
NONCE = "vault-0f4c2b9e-6a11-4d3f-9f7b-2c5a8e1d0b63"
PRIVATE_FACT_VAULT = f"The private vault code is {NONCE}."
STATE_HANDLE = "PRIVATE_HANDLE_1"  # the shared wire fixture's state_handle


# ---------------------------------------------------------------------------
# A. Leak detector. Fails on exact, case-insensitive and normalised matches.
# ---------------------------------------------------------------------------
_INVISIBLE = re.compile(r"[\u00ad\u200b-\u200f\u2060\ufeff]")
_WS = re.compile(r"\s+")


def _forms(text):
    """Every form of one string a leak detector must compare against."""
    bare = _INVISIBLE.sub("", text)
    return {
        "exact": text,
        "casefold": bare.casefold(),
        "nfkc_casefold": unicodedata.normalize("NFKC", bare).casefold(),
        "ws_collapsed": _WS.sub(" ", bare).strip().casefold(),
        "ws_removed": _WS.sub("", bare).casefold(),
    }


def assert_no_private_fact(rendered_prompt, forbidden):
    """Fail if any forbidden string reached ``rendered_prompt`` in any form.

    Checked per secret: exact substring, case-insensitive, Unicode NFKC
    case-fold, whitespace-collapsed and whitespace-removed. Zero-width and
    soft-hyphen characters are stripped first, so a split or padded secret is
    still found.
    """
    assert isinstance(rendered_prompt, str)
    secrets = [forbidden] if isinstance(forbidden, str) else list(forbidden)
    haystack = _forms(rendered_prompt)
    for secret in secrets:
        assert isinstance(secret, str) and secret.strip(), "empty forbidden value"
        if len(secret.strip()) < 4:
            raise AssertionError(f"forbidden value too short to test soundly: {secret!r}")
        needle = _forms(secret)
        for name, value in needle.items():
            if not value:
                continue
            if value in haystack[name]:
                raise AssertionError(
                    f"private value {secret!r} reached the ordinary payload "
                    f"(match form: {name})")


# ---------------------------------------------------------------------------
# MIRROR: receiver-side rendering (Rust), reproduced line by line.
# ---------------------------------------------------------------------------
def delivery_record_text(message, *, message_id="message-1", source_session="source",
                         target_session="receiver", relationship="child"):
    """MIRROR agent_messages.rs:1741-1778 (``create_agent_session_message_prompt_with_body``).

    Line order is the contract. The sender/endpoint descriptors are abbreviated
    to their session names because agent_messages.rs:1865-1920 only formats the
    same fields; no secret is ever part of that descriptor.
    """
    label = (f"[from {relationship}]" if relationship == "parent"
             else f"[from {relationship}:{source_session}]")
    return "\n".join([
        label,
        "Agent-to-agent message received.",
        f"Source: {AGENT_MESSAGE_SOURCE}",
        f"From: {source_session}",
        f"To: {target_session}",
        f"Message id: {message_id}",
        "",
        message,
    ])


def control_payload():
    """MIRROR agent_messages.rs:2076-2085 (``PreparedNativeHandoff::control_payload``).

    The model-visible message is the envelope control summary only; the state
    handle, receipts and source text are never part of it.
    """
    return {"id": "message-1", "source": AGENT_MESSAGE_SOURCE, "message": CONTROL_SUMMARY,
            "from": {"session_id": "source", "session_name": "source"},
            "from_relationship": "child", "target": {"session_id": "receiver",
                                                     "session_name": "receiver"}}


CASES = ("positive", "no_state", "unrelated_state", "shuffled_state",
         "token_reference_only", "random_vectors", "full_text")


def ordinary_payload(case, *, private_fact=PRIVATE_FACT_VAULT, system_prompt=RECEIVER_SYSTEM_PROMPT,
                     tool_result=None, native_record=False, text=None):
    """MIRROR of the receiver request body BEFORE the reserved ``occ3`` field.

    pi-ai inserts ``occ3`` only after ``on_payload`` ran
    (openai_completions.rs:3036-3071), so this dict is exactly what the ordinary
    prompt renderer and the audit hook see. ``native_record`` adds the OCC-3
    record to prove the parity test is sensitive to it.

    Every state-bearing control keeps the positive control's ordinary text
    (``CONTROL_SUMMARY``): the injected state never appears in the ordinary
    payload, so a matched control differs only in the reserved native record.
    """
    assert case in CASES
    body = CONTROL_SUMMARY if text is None else text
    if case == "no_state":
        user_text = NO_STATE_TASK if text is None else text
    elif case == "full_text":
        user_text = f"{body}\n\n{private_fact}"
    else:
        user_text = delivery_record_text(body)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}]
    if tool_result is not None:
        messages.append({"role": "tool", "tool_call_id": "call-1", "content": tool_result})
    payload = {"model": "glm-5.3", "stream": True, "max_tokens": DEFAULT_MAX_TOKENS,
               "messages": messages,
               "tools": [{"type": "function",
                          "function": {"name": HANDOFF_TOOL, "parameters": {"type": "object"}}}]}
    if native_record:
        payload["occ3"] = wire_fixture().wire()
    return payload


def payload_texts(payload):
    """All model-visible text of one payload: every message content, in order."""
    texts = []
    for message in payload.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(block.get("text", "") for block in content
                         if isinstance(block, dict) and isinstance(block.get("text"), str))
    return texts


def _message_body(message):
    """Ordinary task text of one user message, without the delivery-record envelope."""
    text = message["content"]
    if text.startswith("[from "):        # agent_messages.rs:1741-1778 label line
        return text.split("\n\n", 1)[1]
    return text


def ordinary_task_texts(payload):
    return [_message_body(message) for message in ordinary_user_messages(payload)]


def parity_sha256(payload):
    """Ordinary-prompt hash with the OCC-3 record and the record envelope removed."""
    body = dict(payload)
    body.pop("occ3", None)
    body["messages"] = [
        {**message, "content": _message_body(message)} if message.get("role") == "user"
        else message for message in body["messages"]]
    return hashlib.sha256(canonical(body)).hexdigest()


def payload_sha256(payload, *, strip_native=True):
    """MIRROR of the suite's ``body_sha256_before_attachment`` (handoff.rs:457)."""
    body = dict(payload)
    if strip_native:
        body.pop("occ3", None)
    return hashlib.sha256(canonical(body)).hexdigest()


def ordinary_user_messages(payload):
    return [m for m in payload.get("messages", []) if m.get("role") == "user"]


# ---------------------------------------------------------------------------
# E/F. MIRROR of the two OCC-3 acceptance predicates (Rust).
# ---------------------------------------------------------------------------
def prepared_handoff_accepts(control_summary):
    """MIRROR agent_messages.rs:2245-2253 (``PreparedNativeHandoff::new``).

    Only the pinned control summary may travel as the ordinary message body, so
    an OCC-1/OCC-2/TXT body cannot be the control summary of a native handoff.
    """
    return control_summary == CONTROL_SUMMARY


def positive_case_passed(*, action_ok=True, answer=NONCE, nonce=NONCE, pure_text=True,
                         delivered=True, receipt_count=1, lease_state="Consumed",
                         receiver_tool_calls=0, source_tool_calls=1, source_state_ok=True,
                         facts_absent=True, max_tokens=DEFAULT_MAX_TOKENS, request_count=1,
                         evidence_valid=True):
    """MIRROR examples/occ3_native_phase_suite.rs:1497-1508.

    ``lease_state`` mirrors the receiver audit projection
    (occ3/inference.rs:298, ``format!("{:?}", lease.state())``); ``Consumed`` is
    the only accepted state. ``evidence_valid`` mirrors Python
    ``validate_evidence`` (use_receipts.py:62-91): a passing trial needs both.
    """
    return all((
        action_ok,
        answer == nonce,
        pure_text,
        delivered,
        receipt_count == 1,
        lease_state == "Consumed",
        receiver_tool_calls == 0,
        source_tool_calls == 1,
        source_state_ok,
        facts_absent,
        max_tokens == DEFAULT_MAX_TOKENS,
        request_count == 1,
        evidence_valid,
    ))


def no_state_case_passed(*, action_ok=True, answer=UNKNOWN_ANSWER, nonce=NONCE, pure_text=True,
                         receipt_count=0, receiver_records=0, tool_calls=0, facts_absent=True,
                         max_tokens=DEFAULT_MAX_TOKENS, request_count=1):
    """MIRROR examples/occ3_native_phase_suite.rs:1578-1587."""
    return all((
        action_ok,
        answer == UNKNOWN_ANSWER,
        answer != nonce,
        pure_text,
        receipt_count == 0,
        receiver_records == 0,
        tool_calls == 0,
        facts_absent,
        max_tokens == DEFAULT_MAX_TOKENS,
        request_count == 1,
    ))


# ---------------------------------------------------------------------------
# REAL: a local hook harness around the actual ChatAttachmentAdapter.
# ---------------------------------------------------------------------------
def _wrap_fn(original_fn, hook, hook_type):
    """MIRROR of the hash-pinned ``sglang.srt.plugins.hook_registry._wrap_fn``."""
    if hook_type == "after":
        @functools.wraps(original_fn)
        def after_wrapper(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            modified = hook(result, *args, **kwargs)
            return modified if modified is not None else result
        after_wrapper.__wrapped__ = original_fn
        return after_wrapper
    if hook_type == "around":
        @functools.wraps(original_fn)
        def around_wrapper(*args, **kwargs):
            return hook(original_fn, *args, **kwargs)
        around_wrapper.__wrapped__ = original_fn
        return around_wrapper
    raise ValueError("unsupported hook type")


class _HookType:
    AROUND, AFTER = "around", "after"


class _RecordingBackend:
    """Deterministic dependency stub. Its booleans are NOT device evidence."""

    def __init__(self):
        self.reserves, self.binds, self.finishes, self.retires = [], [], [], []
        self.mutate_evidence = None

    def readiness(self):
        return BackendReadiness(True, True, True, True, True, True)

    async def reserve(self, a, identity, attempt_id, native):
        self.reserves.append((a, identity, attempt_id, native))
        return Reservation(attempt_id, a.binding_digest, native.request.rid,
                           identity.principal_digest, .1, 65536, native.max_new_tokens,
                           "fixture-reservation")

    def bind(self, reservation, native):
        self.binds.append(reservation)
        a, identity, _, view = self.reserves[-1]
        native.occ3_worker_descriptor = ConsumeWorkerDescriptor(
            reservation.attempt_id, reservation.reservation_id, reservation.native_rid,
            identity.principal_digest, a.binding_digest, a.receiver_session_id, a.message_id,
            a.request_id, a.state_handle, 1, a.runtime_generation, "fixture-source-commit",
            a.runtime_id, a.model_fingerprint, a.method, "b" * 64, 8, 8, 6144, "bfloat16",
            view.deadline_monotonic)

    async def finish(self, reservation, completion):
        self.finishes.append(completion)
        a = self.reserves[-1][0]
        evidence = TerminalUseEvidence(
            reservation.attempt_id, a.binding_digest, reservation.native_rid,
            reservation.principal_digest, "fixture-source-commit", "fixture-injection-commit",
            a.runtime_id, a.runtime_generation, a.model_fingerprint, a.receiver_session_id,
            completion.output_sha256, 8, 8, 8, True, False, 0, True, True,
            reservation.reservation_id)
        return self.mutate_evidence(evidence) if self.mutate_evidence else evidence

    async def retire(self, attempt_id, native_rid, dispatched):
        self.retires.append((attempt_id, native_rid, dispatched))


class _ChatRequest(NS):
    def to_sampling_params(self, **_kwargs):
        return {"max_new_tokens": self.max_tokens, "temperature": .7, "n": self.n,
                "reasoning_effort": self.reasoning_effort}


def chat_request(**changes):
    data = dict(model="glm-5.3", n=1, stream=True, input_ids=None, max_tokens=32768,
                reasoning_effort="max", chat_template_kwargs={"fixture": True},
                return_meta_info=False, return_hidden_states=False, rid=None,
                messages=[{"role": "system", "content": RECEIVER_SYSTEM_PROMPT},
                          {"role": "user", "content": MATCHED_TASK}],
                tools=[{"type": "function",
                        "function": {"name": HANDOFF_TOOL, "parameters": {}}}])
    data.update(changes)
    return _ChatRequest(**data)


def raw_request(body):
    queue = __import__("asyncio").Queue()
    queue.put_nowait({"type": "http.request", "body": body, "more_body": False})
    scope = {"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
             "query_string": b"", "headers": [(b"content-type", b"application/json")],
             "server": ("127.0.0.1", 80), "client": ("127.0.0.1", 9999)}
    return Request(scope, queue.get)


class _Harness(NS):
    pass


def build_adapter_harness():
    """Real ``ChatAttachmentAdapter`` with local serving/tokenizer stubs."""
    backend = _RecordingBackend()

    async def authorize(_raw):
        fixture = wire_fixture()
        return AuthenticatedContext("a" * 64, fixture.receiver_session_id, "glm-5.3",
                                    fixture.model_fingerprint, fixture.runtime_id,
                                    fixture.runtime_generation, time.monotonic() + 60)

    adapter = ChatAttachmentAdapter(
        backend, authorize=authorize,
        profile=BufferProfile(max_buffer_bytes=65536, total_working_bytes=524288,
                              max_turn_seconds=60, cleanup_seconds=.1,
                              disconnect_interval=.002))

    class TokenizerManager:
        def __init__(self):
            self.request_logger = NS(log_requests=False)
            self.model_config = NS(is_multimodal=False)
            self.calls = []
            self.aborts = []
            self.closed = 0

        async def generate_request(self, obj, request=None):
            self.calls.append(obj)
            try:
                yield {"text": "fixture", "meta_info": {"id": obj.rid,
                                                        "finish_reason": {"type": "stop"}}}
            finally:
                self.closed += 1

        def abort_request(self, rid):
            self.aborts.append(rid)

        def create_abort_task(self, obj):
            return None

    TokenizerManager.generate_request = _wrap_fn(
        TokenizerManager.generate_request, adapter.generate_around, _HookType.AROUND)

    class Chat:
        def __init__(self):
            self.tokenizer_manager = TokenizerManager()
            self.frames = []
            self.template_inputs = []

        def _convert_to_internal_request(self, request, raw_request):
            self.template_inputs.append((request.messages, request.tools))
            native = NS(input_ids=[100, 200, 300],
                        sampling_params=request.to_sampling_params(),
                        stream=True, rid=None, text=None, occ3_worker_descriptor=None)
            return native, request

        async def _stream(self, native, raw_request):
            async for output in self.tokenizer_manager.generate_request(native, raw_request):
                for delta in ({"role": "assistant", "content": "fixture"},
                              {"content": " answer"}):
                    frame = "data: " + json.dumps(
                        {"id": native.rid,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n"
                    self.frames.append(frame.encode())
                    yield frame
                tail = "data: " + json.dumps(
                    {"id": native.rid,
                     "choices": [{"index": 0, "delta": {},
                                  "finish_reason": output["meta_info"]["finish_reason"]["type"]}]}) + "\n\n"
                self.frames.append(tail.encode())
                yield tail
                done = "data: [DONE]\n\n"
                self.frames.append(done.encode())
                yield done

        async def handle_request(self, request, raw_request):
            native, _processed = self._convert_to_internal_request(request, raw_request)
            generator = self._stream(native, raw_request)
            # Pinned SGLang kick-starts the generator before HTTP 200
            # (serving_chat.py:_handle_streaming_request).
            first = await generator.__anext__()

            async def prepend():
                yield first
                async for chunk in generator:
                    yield chunk

            return StreamingResponse(prepend(), media_type="text/event-stream")

    Chat.handle_request = _wrap_fn(Chat.handle_request, adapter.handle_around, _HookType.AROUND)
    Chat._convert_to_internal_request = _wrap_fn(
        Chat._convert_to_internal_request, adapter.convert_after, _HookType.AFTER)
    chat = Chat()
    return _Harness(adapter=adapter, backend=backend, chat=chat,
                    tm=chat.tokenizer_manager)


async def run_attached_turn(attachment=None, request=None, body=None):
    harness = build_adapter_harness()
    attachment = attachment or wire_fixture()
    request = request or chat_request()
    raw = raw_request(body if body is not None else canonical({"occ3": attachment.wire()}))
    response = await harness.chat.handle_request(request, raw)
    return harness, request, response, attachment


# ===========================================================================
# A. Detector self-tests.
# ===========================================================================
@pytest.mark.parametrize("leak", [
    NONCE,
    NONCE.upper(),
    NONCE.title(),
    f"note: {NONCE}",
    NONCE.replace("0f4c", "0f4\nc"),
    NONCE.replace("vault-", "vault\u200b-"),
    f"  {NONCE}  ",
])
def test_detector_rejects_exact_case_and_whitespace_variants(leak):
    with pytest.raises(AssertionError, match="reached the ordinary payload"):
        assert_no_private_fact(leak, [NONCE])


@pytest.mark.parametrize("clean", [
    "", "You are the receiver of this attached native state. It is the sender's context, not your own history. As the receiver, answer your current task using that state.",
    "vault-0f4c2b9e-6a11-4d3f-9f7b-2c5a8e1d0b6",  # truncated by one character
    "UNKNOWN",
])
def test_detector_accepts_clean_text(clean):
    assert_no_private_fact(clean, [NONCE])


def test_detector_refuses_unusably_short_secrets():
    with pytest.raises(AssertionError, match="too short"):
        assert_no_private_fact("abc", ["ab"])


# ===========================================================================
# B. The positive receiver's ordinary payload carries no private value.
# ===========================================================================
@pytest.mark.parametrize("case,fact,secrets", [
    ("positive", PRIVATE_FACT_VAULT, [NONCE, PRIVATE_FACT_VAULT, STATE_HANDLE]),
    ("positive", f"The private animal is a {CATEGORY_FACTS[1][0]}.",
     [CATEGORY_FACTS[1][0], f"The private animal is a {CATEGORY_FACTS[1][0]}.", STATE_HANDLE]),
])
def test_positive_receiver_payload_has_no_private_value(case, fact, secrets):
    payload = ordinary_payload(case, private_fact=fact)
    rendered = "\n".join(payload_texts(payload))
    assert_no_private_fact(rendered, secrets)
    # The forbidden values must not hide inside the serialized payload either.
    assert_no_private_fact(json.dumps(payload, ensure_ascii=False), secrets)


def test_positive_receiver_payload_is_one_ordinary_user_message():
    payload = ordinary_payload("positive")
    assert len(ordinary_user_messages(payload)) == 1
    assert payload["messages"][1]["content"].startswith("[from child:source]")
    assert CONTROL_SUMMARY in payload["messages"][1]["content"]


def test_detector_flags_the_full_text_leak_control():
    """The suite's own ``full-text`` control must trip the detector."""
    payload = ordinary_payload("full_text")
    with pytest.raises(AssertionError, match="reached the ordinary payload"):
        assert_no_private_fact("\n".join(payload_texts(payload)), [NONCE])


def test_detector_flags_a_tool_result_leak():
    """A fact smuggled through a tool result is caught on the same channel."""
    payload = ordinary_payload("positive", tool_result=PRIVATE_FACT_VAULT)
    with pytest.raises(AssertionError, match="reached the ordinary payload"):
        assert_no_private_fact("\n".join(payload_texts(payload)), [NONCE])


def test_expected_answer_class_is_never_bound_to_the_private_fact():
    """The class token appears once, in the fixed enumeration, never with the animal.

    This is the weakest sound check available for the semantic case: the system
    prompt enumerates all four classes (handoff.rs:70), so a class-only leak test
    can never work. Any extra occurrence of the expected class is a leak.
    """
    animal, expected = CATEGORY_FACTS[1]
    payload = ordinary_payload("positive", system_prompt=CATEGORY_QUESTION,
                               private_fact=f"The private animal is a {animal}.")
    texts = payload_texts(payload)
    joined = "\n".join(texts)
    assert animal not in joined
    assert len(re.findall(rf"\b{expected}\b", joined)) == 1, "extra answer-class occurrence"
    for text in texts:
        assert not (re.search(rf"\b{expected}\b", text) and animal in text)


async def test_real_adapter_keeps_the_private_values_out_of_the_model_prompt():
    harness, request, response, wire = await run_attached_turn()
    assert response.status_code == 200
    native = harness.tm.calls[0]
    # The native request carries token ids only: no prompt text at all.
    assert native.input_ids == [100, 200, 300] and native.text is None
    seen = harness.chat.template_inputs[0]
    assert_no_private_fact(json.dumps(seen, ensure_ascii=False),
                           [NONCE, PRIVATE_FACT_VAULT, wire.state_handle])
    assert_no_private_fact(json.dumps(request.messages, ensure_ascii=False),
                           [NONCE, PRIVATE_FACT_VAULT, wire.state_handle])
    # The handle exists in exactly one place: the reserved occ3 field.
    body = json.loads(canonical({"occ3": wire.wire()}))
    assert body["occ3"]["state_handle"] == wire.state_handle


# ===========================================================================
# C. The native delivery record is an exact allowlist, with no plaintext.
# ===========================================================================
def test_attachment_wire_allowlist_is_pinned():
    assert tuple(sorted(InferenceAttachment.__dataclass_fields__)) == ALLOWED_ATTACHMENT_KEYS
    parsed = parse_opt_in(canonical({"occ3": wire_fixture().wire()}))
    assert tuple(sorted(parsed.wire())) == ALLOWED_ATTACHMENT_KEYS
    for extra in ("source_text", "private_fact", "expected_answer", "nonce"):
        with pytest.raises(AttachmentError):
            parse_opt_in(canonical({"occ3": {**wire_fixture().wire(), extra: NONCE}}))


async def test_native_receipt_header_is_an_exact_allowlist_without_plaintext():
    harness, _request, response, wire = await run_attached_turn()
    assert response.status_code == 200, response.body
    raw = dict(response.raw_headers)
    assert RECEIPT_HEADER.encode() in raw
    receipt = json.loads(raw[RECEIPT_HEADER.encode()])
    assert tuple(receipt) == ALLOWED_RECEIPT_KEYS
    assert tuple(receipt["attachment"]) == ALLOWED_ATTACHMENT_KEYS
    assert receipt["attachment"] == wire.wire()
    assert receipt["consumed"] is True
    assert receipt["fallback_used"] is False
    assert receipt["injected_state"] is True
    # The header is exactly receipt_bytes() for this attachment: one producer.
    assert raw[RECEIPT_HEADER.encode()] == receipt_bytes(wire)
    header_text = raw[RECEIPT_HEADER.encode()].decode("ascii")
    assert_no_private_fact(header_text, [NONCE, PRIVATE_FACT_VAULT])
    # The capability handle appears once, as its allowlisted field, and never
    # inside the receipt flags.
    assert header_text.count(wire.state_handle) == 1
    assert wire.state_handle not in json.dumps({k: v for k, v in receipt.items()
                                                if k != "attachment"})


async def test_the_occ3_field_never_reaches_the_model_visible_body():
    """pi-ai refuses a forged reserved field after on_payload (openai_completions.rs:3043)."""
    harness, _request, response, wire = await run_attached_turn()
    assert response.status_code == 200
    model_visible = json.dumps(harness.chat.template_inputs[0], ensure_ascii=False)
    assert "\"occ3\":" not in model_visible and "state_handle" not in model_visible
    assert wire.state_handle not in model_visible


# ===========================================================================
# D. Ordinary-prompt parity: positive vs no-state.
# ===========================================================================
def test_every_control_has_exactly_one_ordinary_user_message():
    for case in CASES:
        payload = ordinary_payload(case)
        assert len(ordinary_user_messages(payload)) == 1, case


def test_state_bearing_controls_are_byte_identical_to_the_positive_payload():
    """Unrelated/shuffled/random/token-reference differ only in the native record."""
    reference = payload_sha256(ordinary_payload("positive"))
    for case in ("unrelated_state", "shuffled_state", "token_reference_only", "random_vectors"):
        assert payload_sha256(ordinary_payload(case)) == reference, case
        recorded = ordinary_payload(case, native_record=True)
        assert payload_sha256(recorded, strip_native=False) != reference, case


def test_parity_ignores_only_the_delivery_record_envelope():
    """The parity key is the ordinary task text: the envelope carries no secret."""
    positive = ordinary_payload("positive")
    matched_no_state = ordinary_payload("no_state", text=CONTROL_SUMMARY)
    assert ordinary_task_texts(positive) == ordinary_task_texts(matched_no_state) == [CONTROL_SUMMARY]
    assert parity_sha256(matched_no_state) == parity_sha256(positive)
    # ... while a plain user turn and the record-framed turn are not byte-equal.
    assert payload_sha256(matched_no_state) != payload_sha256(positive)


@pytest.mark.xfail(strict=True, reason=(
    "unmet control: the positive receiver reads the envelope control summary "
    "while the no-state receiver reads a prompt that announces the missing "
    "state (phase_suite.rs:1564), so the two ordinary prompts are not matched"))
def test_parity_ordinary_payload_is_byte_identical_positive_vs_no_state():
    positive = ordinary_payload("positive")
    no_state = ordinary_payload("no_state")
    assert len(ordinary_user_messages(positive)) == len(ordinary_user_messages(no_state)) == 1
    assert ordinary_task_texts(no_state) == ordinary_task_texts(positive)
    assert parity_sha256(no_state) == parity_sha256(positive)


def test_native_record_is_the_only_difference_between_matched_cases():
    without = ordinary_payload("positive")
    with_record = ordinary_payload("positive", native_record=True)
    assert payload_sha256(with_record, strip_native=False) != payload_sha256(without)
    assert payload_sha256(with_record) == payload_sha256(without)
    stripped = {k: v for k, v in with_record.items() if k != "occ3"}
    assert payload_sha256(stripped) == payload_sha256(without)


@pytest.mark.xfail(strict=True, reason=(
    "unmet control: the semantic case's expected answer is enumerable from the "
    "receiver system prompt (handoff.rs:70 lists BIRD/MAMMAL/FISH/REPTILE), so "
    "a class-only answer is reachable without any state"))
def test_semantic_case_answer_is_not_enumerable_from_the_system_prompt():
    _animal, expected = CATEGORY_FACTS[1]
    assert expected not in CATEGORY_QUESTION


# ===========================================================================
# E. OCC-1 / OCC-2 / TXT fallback content cannot pass a native case.
# ===========================================================================
@pytest.mark.parametrize("body", [
    f"{OCC1_HEADER} from=source to=receiver id=m1\n{PRIVATE_FACT_VAULT}",
    f"{OCC2_HEADER} sym-v1 D=fixture@1\nsymbolic body",
    PRIVATE_FACT_VAULT,
    f"{MATCHED_TASK}\n\n{PRIVATE_FACT_VAULT}",
    "",
])
def test_fallback_bodies_are_not_accepted_as_a_native_control_summary(body):
    assert not prepared_handoff_accepts(body)


def test_only_the_pinned_control_summary_is_accepted():
    assert prepared_handoff_accepts(CONTROL_SUMMARY)
    assert not prepared_handoff_accepts(NO_STATE_TASK)


async def _completed_turn(harness=None):
    """Run one real attached turn and return the objects the backend saw."""
    harness = harness or build_adapter_harness()
    attachment = wire_fixture()
    raw = raw_request(canonical({"occ3": attachment.wire()}))
    response = await harness.chat.handle_request(chat_request(), raw)
    assert response.status_code == 200, response.body
    reservation = harness.backend.binds[-1]
    completion = harness.backend.finishes[-1]
    descriptor = harness.tm.calls[0].occ3_worker_descriptor
    evidence = TerminalUseEvidence(
        reservation.attempt_id, attachment.binding_digest, reservation.native_rid,
        harness.backend.reserves[-1][1].principal_digest, "fixture-source-commit",
        "fixture-injection-commit", attachment.runtime_id, attachment.runtime_generation,
        attachment.model_fingerprint, attachment.receiver_session_id,
        completion.output_sha256, 8, 8, 8, True, False, 0, True, True,
        reservation.reservation_id)
    return harness, attachment, reservation, descriptor, completion, evidence


@pytest.mark.parametrize("change", [
    {"fallback_used": True},           # OCC-1/OCC-2/TXT fallback data
    {"injected_prefix_count": 0},      # no latent prefix was injected
    {"private_cache_namespace": False},
    {"gpu_resident": False},
    {"completed": False},
    {"num_retractions": 1},
])
async def test_fallback_and_text_only_evidence_cannot_pass_the_success_predicate(change):
    harness, attachment, reservation, descriptor, completion, good = await _completed_turn()
    validate_evidence(attachment, reservation, descriptor, completion, good)
    with pytest.raises(AttachmentError, match="invalid_evidence"):
        validate_evidence(attachment, reservation, descriptor, completion,
                          replace(good, **change))


async def test_a_turn_with_fallback_evidence_returns_no_receipt():
    """A case cannot pass on fallback data: the real adapter fails closed."""
    harness = build_adapter_harness()
    harness.backend.mutate_evidence = lambda evidence: replace(evidence, fallback_used=True)
    attachment = wire_fixture()
    raw = raw_request(canonical({"occ3": attachment.wire()}))
    refused = await harness.chat.handle_request(chat_request(), raw)
    assert refused.status_code != 200
    assert RECEIPT_HEADER.encode() not in dict(refused.raw_headers)
    assert len(harness.backend.finishes) == 1 and len(harness.backend.retires) == 1
    assert harness.adapter._active == 0


def test_receipt_shaped_bytes_alone_are_not_a_success_predicate():
    """receipt_bytes() is a size-bound header, not use evidence.

    Its three verdict fields are constants (use_receipts.py:54-59), so the same
    blob is produced for any attachment, used or not. The adapter emits it
    verbatim (chat_adapter.py:481-482, :840), therefore a score that reads only
    the header would pass a text-only delivery: scoring must require
    validate_evidence() plus a Consumed lease.
    """
    wire = wire_fixture()
    placeholder = json.loads(receipt_bytes(wire))
    other = json.loads(receipt_bytes(replace(wire, state_handle="OTHER_HANDLE_2")))
    assert placeholder["consumed"] is True
    assert placeholder["fallback_used"] is False
    assert placeholder["injected_state"] is True
    assert ({k: v for k, v in placeholder.items() if k != "attachment"}
            == {k: v for k, v in other.items() if k != "attachment"})


# ===========================================================================
# F. An answer obtained without a Consumed lease is not a success.
# ===========================================================================
@pytest.mark.parametrize("lease_state", [None, "Prepared", "Submitted", "Uncertain",
                                         "Cancelled"])
def test_answer_without_a_consumed_lease_is_not_a_success(lease_state):
    assert positive_case_passed(answer=NONCE, lease_state=lease_state) is False


def test_answer_with_a_consumed_lease_and_valid_evidence_is_a_success():
    assert positive_case_passed(answer=NONCE, lease_state="Consumed") is True


def test_success_needs_both_the_lease_and_the_evidence_record():
    assert positive_case_passed(lease_state="Consumed", evidence_valid=False) is False
    assert positive_case_passed(lease_state="Submitted", evidence_valid=True) is False


@pytest.mark.parametrize("answer", [UNKNOWN_ANSWER, "", NONCE.upper(), f" {NONCE}"])
def test_answers_other_than_the_exact_nonce_are_not_a_success(answer):
    assert positive_case_passed(answer=answer, lease_state="Consumed") is False


def test_no_state_control_passes_only_with_the_exact_unknown_answer():
    assert no_state_case_passed() is True
    assert no_state_case_passed(answer=NONCE) is False
    assert no_state_case_passed(receipt_count=1) is False
    assert no_state_case_passed(receiver_records=1) is False
