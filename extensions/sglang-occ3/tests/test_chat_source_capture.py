"""Offline source-format/ASGI tests, not real GPU capture or semantic proof.

Pinned actual converter/handler methods use dependency stubs from receiver
fixtures. Starlette Request/Response and cancellation are real. No model load.
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from starlette.responses import JSONResponse

from occ3_sglang.attachment_contract import (
    AttachmentError, AuthenticatedContext, BackendReadiness, CaptureIntent,
    CaptureWorkerDescriptor, SourceCaptureReceipt, SourceEnrollment, Reservation,
    SOURCE_RECEIPT_HEADER, SPAN_POLICIES_V2, canonical, digest, normalize_worker_descriptor,
    parse_opt_in, strict_json,
)
from occ3_sglang.use_receipts import (
    NativeOutputAudit, SourceCaptureCommit, SourceCaptureEvidence,
    source_receipt_upper_bound,
)
from test_chat_attachment import FixtureBackend, attachment, chat_request, harness, raw_request
from test_attachment_cancellation import spin_until
from test_span_markers import FakeTokenizer, NoEncodeTokenizer

FIXTURE = json.loads((Path(__file__).parent / "fixtures/occ3_source_v1.json").read_text())


def intent(**changes):
    value = dict(capture_id="capture-1", source_session_id="source-1", source_request_id="client-attempt-1",
                 turn_nonce="turn-1", model_fingerprint="sha256:fixture", runtime_id="runtime-1",
                 runtime_generation=4, method="statebridge-fixture-v1", retained_count_limit=128,
                 max_capture_bytes=128 * 1024 * 1024, expires_at_epoch=int(time.time()) + 300)
    value.update(changes)
    return CaptureIntent(**value)


class SourceFixtureBackend(FixtureBackend):
    """Synthetic backend journal, never real CUDA capture evidence."""
    def __init__(self):
        super().__init__()
        self.enrollments, self.prepares, self.source_binds, self.source_finishes = [], [], [], []
        self.enroll_gate = self.prepare_gate = self.source_finish_gate = None
        self.mutate_enrollment = self.mutate_commit = self.mutate_source_evidence = None
        self.mutate_source_descriptor = self.mutate_source_bind = None
        self.source_failure = None
        self.phase_check = None

    def readiness(self):
        return BackendReadiness(True, True, True, True, True, True, True)

    async def enroll_source(self, a, identity, attempt_id, native_rid, deadline):
        self.enrollments.append((a, identity, attempt_id, native_rid, deadline))
        if self.phase_check: self.phase_check()
        if self.enroll_gate: await self.enroll_gate.wait()
        if self.source_failure == "enroll": raise RuntimeError("PRIVATE_SOURCE_HANDLE")
        value = SourceEnrollment(attempt_id, a.binding_digest, native_rid, identity.principal_digest,
            "PRIVATE_SOURCE_HANDLE", 17, a.expires_at_epoch, a.source_session_id, a.runtime_id,
            a.runtime_generation, a.model_fingerprint, a.method, 8, "b" * 64, 77, 6144, "bfloat16")
        return self.mutate_enrollment(value) if self.mutate_enrollment else value

    async def prepare_source(self, enrollment, native):
        self.prepares.append((enrollment, native))
        if self.prepare_gate: await self.prepare_gate.wait()
        if self.source_failure == "prepare": raise RuntimeError(enrollment.source_handle)
        return Reservation(enrollment.attempt_id, enrollment.intent_digest, native.request.rid,
            enrollment.principal_digest, self.required_seconds, self.max_bytes,
            native.max_new_tokens, "source-reservation-1")

    def bind_source(self, enrollment, reservation, native):
        self.source_binds.append((enrollment, reservation))
        a = self.enrollments[-1][0]
        if self.mutate_source_bind: self.mutate_source_bind(native)
        descriptor = CaptureWorkerDescriptor(enrollment.attempt_id, reservation.reservation_id,
            enrollment.native_rid, enrollment.principal_digest, a.runtime_id, a.model_fingerprint,
            a.method, self.prepares[-1][1].deadline_monotonic, a.capture_id, a.source_session_id,
            a.turn_nonce, a.runtime_generation, a.span_policy, a.retained_count_limit, a.max_capture_bytes,
            "" if a.span_policy not in SPAN_POLICIES_V2 else "glm45_glm47_xml_kv.v1")
        native.occ3_worker_descriptor = (self.mutate_source_descriptor(descriptor)
                                        if self.mutate_source_descriptor else descriptor)

    async def finish_source(self, enrollment, reservation, completion, tool_provenance):
        self.source_finishes.append((enrollment, reservation, completion, tool_provenance))
        if self.source_finish_gate: await self.source_finish_gate.wait()
        if self.source_failure == "finish": raise RuntimeError(enrollment.source_handle)
        commit = SourceCaptureCommit("capture-commit-1", 10, 18, 8, -1,
            "final_layer_prenorm_predictor", 77, 6144, "bfloat16", "b" * 64, 8, 8,
            "c" * 64, enrollment.expires_at_epoch, 8 * 6144 * 2)
        if self.mutate_commit: commit = self.mutate_commit(commit)
        result = SourceCaptureEvidence(enrollment, reservation.reservation_id, commit,
            completion.output_sha256, completion.native_rid, tool_provenance.binding_digest,
            True, True, 0)
        return self.mutate_source_evidence(result) if self.mutate_source_evidence else result


@pytest.fixture
def source_harness(harness):
    harness.backend = SourceFixtureBackend()
    harness.adapter.backend = harness.backend
    async def authorize(raw):
        return AuthenticatedContext("a" * 64, "source-1", "fixture-model", "sha256:fixture",
                                    "runtime-1", 4, time.monotonic() + 60)
    harness.adapter.authorize = authorize
    # Complete native rendered tool object for the source provenance audit.
    def build(request, ret, created):
        return JSONResponse({"id": ret[0]["meta_info"]["id"], "choices": [{"index": 0,
            "message": {"role": "assistant", "reasoning_content": "fixture reasoning",
                        "tool_calls": [{"id": "call-fixture", "type": "function",
                                        "function": {"name": "handoff", "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}]})
    harness.chat._build_chat_response = build
    return harness


def source_raw(a=None, body=None, encoding="identity"):
    raw, queue = raw_request(body if body is not None else canonical({"occ3_capture": (a or intent()).wire()}))
    if encoding is not None:
        raw.scope["headers"].append((b"accept-encoding", encoding.encode()))
    return raw, queue


async def source_call(h, a=None, request=None, body=None, encoding="identity"):
    raw, queue = source_raw(a, body, encoding)
    response = await h.chat.handle_request(request or chat_request(), raw)
    return response, raw, queue


def test_shared_source_fixture_v2_units():
    v2 = FIXTURE["v2"]
    a = CaptureIntent.parse(v2["intent"])
    assert canonical(a.wire()).decode() == v2["intent_canonical"]
    assert a.binding_digest == v2["intent_sha256"]
    assert a.schema == "occ3.capture.v2" and a.span_policy == "content_tail_v2"
    r = SourceCaptureReceipt.parse(v2["receipt"])
    assert len(r.wire()) == 39 and r.schema == "occ3.source.v2"
    assert canonical(r.wire()).decode() == v2["receipt_canonical"]
    assert digest(r.wire()) == v2["receipt_sha256"]
    assert r.byte_count == r.retained_count * r.hidden_dim * 2
    assert v2["max_valid_receipt_bytes"] <= 2048
    for case in v2["negative_intents"]:
        with pytest.raises(AttachmentError):
            CaptureIntent.parse(case["wire"])
    for case in v2["negative_receipts"]:
        with pytest.raises(AttachmentError):
            SourceCaptureReceipt.parse(case["wire"])


def test_shared_source_fixture_canonical_bytes_and_units():
    a = CaptureIntent.parse(FIXTURE["intent"])
    r = SourceCaptureReceipt.parse(FIXTURE["receipt"])
    assert len(a.wire()) == 13 and len(r.wire()) == 21
    assert canonical(a.wire()).decode() == FIXTURE["intent_canonical"]
    assert a.binding_digest == FIXTURE["intent_sha256"]
    assert canonical(r.wire()).decode() == FIXTURE["receipt_canonical"]
    assert digest(r.wire()) == FIXTURE["receipt_sha256"]
    assert r.output_sha256 == hashlib.sha256(FIXTURE["response_body_utf8"].encode()).hexdigest()
    assert r.byte_count == r.retained_count * r.hidden_dim * 2
    assert canonical(FIXTURE["canonical_probe"]).decode() == FIXTURE["canonical_probe_bytes"]
    assert digest(FIXTURE["canonical_probe"]) == FIXTURE["canonical_probe_sha256"]
    assert json.loads(source_receipt_upper_bound()) == FIXTURE["max_valid_receipt"]
    assert len(source_receipt_upper_bound()) == FIXTURE["max_valid_receipt_bytes"] <= 2048


@pytest.mark.parametrize("case", FIXTURE["negative_intents"], ids=lambda c: c["name"])
def test_invalid_source_intents(case):
    with pytest.raises(AttachmentError): CaptureIntent.parse(case["wire"])


@pytest.mark.parametrize("case", FIXTURE["negative_receipts"], ids=lambda c: c["name"])
def test_invalid_source_receipts(case):
    with pytest.raises(AttachmentError): SourceCaptureReceipt.parse(case["wire"])


@pytest.mark.parametrize("key", ["duplicate_intent_json", "duplicate_receipt_json"])
def test_duplicate_source_fields(key):
    with pytest.raises(AttachmentError): strict_json(FIXTURE[key])


V2_BLOCK = dict(span_schema="occ3.span.v1", span_policy="content_tail_v2",
                span_marker_profile="glm45_glm47_xml_kv.v1",
                span_marker_schema="occ3.span.markers.v1",
                span_reasoning_terminator="thinking_end", span_turn_end_attested=True,
                span_thinking_count=2, span_content_count=6, span_tool_call_count=0,
                span_tool_arguments_count=0, span_control_count=2, span_unknown_count=0,
                span_window_start_position=9, span_window_end_position=19,
                span_contiguous=True, span_contains_final_content=True,
                span_contains_tool_call=False, span_message_truncated=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_v2_source_flow_injects_markers_and_receipts_the_span_block(source_harness, stream):
    h = source_harness
    h.tm.tokenizer = FakeTokenizer()
    h.backend.mutate_commit = lambda commit: replace(commit, **V2_BLOCK)
    h.backend.mutate_source_descriptor = lambda d: replace(d, marker_profile="glm45_glm47_xml_kv.v1")
    a = intent(schema="occ3.capture.v2", span_policy="content_tail_v2")
    response, _, _ = await source_call(h, a, chat_request(stream=stream))
    assert response.status_code == 200, response.body
    native = h.tm.calls[0]
    custom = native.sampling_params["custom_params"]
    assert custom["__optimus_occ3_span_marker_thinking_end"] == [101]
    assert custom["__optimus_occ3_span_marker_tool_call_start"] == [102]
    assert custom["__optimus_occ3_span_marker_turn_end"] == [108]
    assert custom["__optimus_occ3_span_stop_ids"] == [108]
    assert custom["__optimus_occ3_span_marker_profile"] == "glm45_glm47_xml_kv.v1"
    # The ordinary sampling keys are untouched by the injection.
    assert native.sampling_params["max_new_tokens"] == 32768
    assert native.sampling_params["n"] == 1
    receipt = SourceCaptureReceipt.parse(json.loads(response.headers[SOURCE_RECEIPT_HEADER]))
    assert receipt.schema == "occ3.source.v2" and len(receipt.wire()) == 39
    assert receipt.span_policy == "content_tail_v2" and receipt.span_content_count == 6
    assert (receipt.accepted_span_start, receipt.accepted_span_end) == (10, 18)
    assert receipt.retained_count == 8 and receipt.span_window_start_position == 9
    assert receipt.span_marker_profile == "glm45_glm47_xml_kv.v1"
    provenance = h.backend.source_finishes[0][3]
    assert provenance.scope == "content_tail_v2"
    assert h.adapter._active == 0 and not h.backend.retires


@pytest.mark.asyncio
async def test_v2_marker_channel_cannot_be_spoofed_from_the_request(source_harness):
    h = source_harness
    h.tm.tokenizer = FakeTokenizer()
    h.backend.mutate_commit = lambda commit: replace(commit, **V2_BLOCK)
    h.backend.mutate_source_descriptor = lambda d: replace(d, marker_profile="glm45_glm47_xml_kv.v1")
    request = chat_request()
    original = request.to_sampling_params

    def spoofed(**kwargs):
        value = original(**kwargs)
        value["custom_params"] = {"__optimus_occ3_span_marker_thinking_end": [999],
                                  "client_key": 1}
        return value

    request.to_sampling_params = spoofed
    a = intent(schema="occ3.capture.v2", span_policy="content_tail_v2")
    response, _, _ = await source_call(h, a, request)
    assert response.status_code == 200, response.body
    custom = h.tm.calls[0].sampling_params["custom_params"]
    # The attested entries overwrite the spoofed occ3 key; unrelated keys pass.
    assert custom["__optimus_occ3_span_marker_thinking_end"] == [101]
    assert custom["client_key"] == 1


@pytest.mark.asyncio
async def test_v2_commit_without_a_block_is_invalid_evidence(source_harness):
    h = source_harness
    h.tm.tokenizer = FakeTokenizer()
    h.backend.mutate_source_descriptor = lambda d: replace(d, marker_profile="glm45_glm47_xml_kv.v1")
    a = intent(schema="occ3.capture.v2", span_policy="content_tail_v2")
    response, _, _ = await source_call(h, a, chat_request())
    assert response.status_code != 200 and SOURCE_RECEIPT_HEADER not in response.headers
    assert len(h.backend.retires) == 1 and len(h.tm.calls) == 1


@pytest.mark.asyncio
async def test_v2_descriptor_marker_profile_mismatch_is_invalid_evidence(source_harness):
    h = source_harness
    h.tm.tokenizer = FakeTokenizer()
    h.backend.mutate_commit = lambda commit: replace(commit, **V2_BLOCK)
    h.backend.mutate_source_descriptor = lambda d: replace(d, marker_profile="other.profile.v1")
    a = intent(schema="occ3.capture.v2", span_policy="content_tail_v2")
    response, _, _ = await source_call(h, a, chat_request())
    assert response.status_code != 200 and SOURCE_RECEIPT_HEADER not in response.headers
    assert len(h.backend.retires) == 1


@pytest.mark.asyncio
async def test_v2_unattestable_tokenizer_declines_before_generation(source_harness):
    h = source_harness
    h.tm.tokenizer = NoEncodeTokenizer()
    a = intent(schema="occ3.capture.v2", span_policy="content_tail_v2")
    response, _, _ = await source_call(h, a, chat_request())
    assert response.status_code == 409 and SOURCE_RECEIPT_HEADER not in response.headers
    assert not h.tm.calls and len(h.backend.retires) == 1


@pytest.mark.asyncio
async def test_v2_receipt_upper_bound_reserves_the_header_budget(source_harness):
    # 1300 bytes hold the v1 upper bound (1234) but not the v2 one (1863):
    # the same profile serves v1 and refuses v2 before generation.
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, max_receipt_bytes=1300)
    response, _, _ = await source_call(h, intent())
    assert response.status_code == 200
    h.adapter.profile = replace(h.adapter.profile, max_receipt_bytes=1300)
    h.tm.tokenizer = FakeTokenizer()
    a = intent(schema="occ3.capture.v2", span_policy="content_tail_v2")
    response, _, _ = await source_call(h, a, chat_request())
    assert response.status_code != 200 and SOURCE_RECEIPT_HEADER not in response.headers
    # Only the v1 call reached generation; the v2 call added no enrollment.
    assert len(h.tm.calls) == 1 and len(h.backend.enrollments) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_actual_source_native_converter_and_one_generation(source_harness, stream):
    h = source_harness
    h.backend.phase_check = lambda: (h.chat.converted == [] and h.tm.calls == []) or pytest.fail("late enrollment")
    a, request = intent(), chat_request(stream=stream)
    response, _, _ = await source_call(h, a, request)
    assert response.status_code == 200, response.body
    assert len(h.backend.enrollments) == len(h.backend.prepares) == len(h.tm.calls) == len(h.backend.source_finishes) == 1
    assert not h.backend.reserves and not h.backend.finishes
    native = h.tm.calls[0]
    assert native.rid != a.source_request_id
    assert native.rid == h.backend.enrollments[0][3]
    assert native.input_ids == [100, 200, 300]
    assert native.sampling_params == {"max_new_tokens": 32768, "temperature": .7, "n": 1, "reasoning_effort": "max"}
    assert h.chat.converted == [(request.messages, request.tools, request.reasoning_effort)]
    assert type(native.occ3_worker_descriptor) is CaptureWorkerDescriptor
    receipt = SourceCaptureReceipt.parse(json.loads(response.headers[SOURCE_RECEIPT_HEADER]))
    assert receipt.intent_digest == a.binding_digest
    assert receipt.output_sha256 == hashlib.sha256(response.body).hexdigest()
    assert receipt.native_rid == receipt.response_id == native.rid
    assert response.headers["content-encoding"] == "identity"
    assert response.headers["cache-control"] == "no-store"
    provenance = h.backend.source_finishes[0][3]
    assert [(c.call_id, c.name) for c in provenance.tool_calls] == [("call-fixture", "handoff")]
    if stream: assert response.body == b"".join(h.chat.frames)
    assert h.adapter._active == 0 and not h.backend.retires


@pytest.mark.asyncio
async def test_source_native_refusal_logs_inner_error_detail(source_harness, caplog):
    """Live TS11 regression: an inner ValueError surfaces through the ordinary
    path's first-chunk catch as an error response; the native wrapper keeps the
    generic public 409 but must record the refusal seam and the bounded inner
    message server-side so live refusals are diagnosable offline."""
    h = source_harness

    async def failing_stream(obj, request, raw):
        raise ValueError("exceeds the model's maximum context length of 2048 tokens")
        yield  # pragma: no cover

    h.chat._generate_chat_stream = failing_stream
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.chat_adapter"):
        response, _, _ = await source_call(h)
    assert response.status_code == 409
    assert json.loads(response.body) == {"error": {"message": "native_completion_failed",
                                                   "type": "occ3_attachment_error", "code": 409}}
    lines = [r.getMessage() for r in caplog.records
             if "native completion refused" in r.getMessage()]
    assert lines, "expected a server-side diagnostic for the refused native turn"
    line = lines[0]
    # The first-chunk ValueError fires before the generate hook runs, so the
    # reservation is never made: the diagnostic must name that seam AND carry
    # the bounded inner message (this is what TS11's live 409 could not show).
    assert "reservation_missing" in line and "maximum context length" in line
    # Identifiers and handles stay out of the diagnostic.
    assert "capture-1" not in line and "source-1" not in line


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "occ3_unsupported_profile", "occ3_admission_failed", "occ3_full_budget_unavailable",
    "occ3_busy", "occ3_invalid_native_request", "occ3_cancelled",
    "occ3_retraction_unsupported", "occ3_commit_failed",
])
async def test_source_scheduler_abort_folds_to_observed_409_and_names_code(source_harness, caplog, message):
    """Live TS11 regression, red-first: the scheduler refuses armed turns
    before prefill by aborting the request; the ordinary stream then delivers
    exactly one abort output, which folds through the pinned first-chunk
    ValueError catch into the observed generic 409. The refusal code must be
    recorded server-side: it is the only place the abort code survives."""
    h = source_harness

    async def aborted_generate(tm, obj, request=None):
        # Class-level hook original: receives (self, obj, raw) like the real
        # tokenizer manager generator the scheduler abort reaches.
        yield {"text": "", "output_ids": [],
               "meta_info": {"id": obj.rid,
                              "finish_reason": {"type": "abort", "message": message,
                                                 "status_code": 503}}}

    type(h.tm).generate_request = h.wrap(aborted_generate, h.adapter.generate_around, h.HookType.AROUND)
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.chat_adapter"):
        response, _, _ = await source_call(h)
    assert response.status_code == 409
    assert json.loads(response.body) == {"error": {"message": "native_completion_failed",
                                                   "type": "occ3_attachment_error", "code": 409}}
    aborted = [r.getMessage() for r in caplog.records if "native turn aborted" in r.getMessage()]
    assert aborted, "expected the scheduler refusal code in the server-side diagnostic"
    assert f"refusal={message}" in aborted[0] and "status=503" in aborted[0]
    # The public body stays generic; the PR #200 post-check line still names the fold seam.
    assert message.encode() not in response.body
    refused = [r.getMessage() for r in caplog.records if "native completion refused" in r.getMessage()]
    assert refused and "inner_error_response" in refused[0]


@pytest.mark.asyncio
async def test_source_scheduler_abort_payload_never_logged(source_harness, caplog):
    """Non-occ3 abort payloads carry arbitrary text; only the bounded
    refusal shape is ever recorded server-side."""
    h = source_harness

    async def aborted_generate(tm, obj, request=None):
        yield {"text": "",
               "meta_info": {"id": obj.rid,
                              "finish_reason": {"type": "abort",
                                                 "message": "Abort in waiting queue with capture-1 payload"}}}

    type(h.tm).generate_request = h.wrap(aborted_generate, h.adapter.generate_around, h.HookType.AROUND)
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.chat_adapter"):
        response, _, _ = await source_call(h)
    assert response.status_code == 409
    aborted = [r.getMessage() for r in caplog.records if "native turn aborted" in r.getMessage()]
    assert aborted and "refusal=unclassified" in aborted[0]
    assert "capture-1" not in aborted[0] and "payload" not in aborted[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["both", "unknown", "null", "duplicate", "missing_schema", "client_rid", "not_ready", "encoding", "encoding_absent", "header_bound"])
async def test_source_refuses_before_generation(source_harness, change):
    h, a, request = source_harness, intent(), chat_request()
    value = {"occ3_capture": a.wire()}
    body = None
    encoding = "identity"
    if change == "both": value["occ3"] = attachment().wire()
    if change == "unknown": value["occ3_capture"]["source_text"] = "not allowed"
    if change == "null": value["occ3_capture"] = None
    if change == "missing_schema": del value["occ3_capture"]["schema"]
    if change == "duplicate": body = canonical(value).replace(b'"capture_id":', b'"capture_id":"again","capture_id":')
    if change == "client_rid": request.rid = a.source_request_id
    if change == "not_ready": h.backend.readiness = lambda: BackendReadiness(True, True, True, True, True, True, False)
    if change == "encoding": encoding = "gzip"
    if change == "encoding_absent": encoding = None
    if change == "header_bound": h.adapter.profile = replace(h.adapter.profile, max_receipt_bytes=1200)
    response, _, _ = await source_call(h, a, request, body or canonical(value), encoding)
    assert response.status_code != 200 and SOURCE_RECEIPT_HEADER not in response.headers
    assert not h.tm.calls and not h.chat.converted and not h.backend.enrollments


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("source_session_id", "other"), ("model_fingerprint", "wrong"), ("runtime_id", "old"), ("runtime_generation", 2),
])
async def test_source_trusted_identity_binding(source_harness, field, value):
    response, _, _ = await source_call(source_harness, intent(**{field: value}))
    assert response.status_code == 401
    assert not source_harness.backend.enrollments and not source_harness.tm.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("attempt_id", "wrong"), ("intent_digest", "d" * 64), ("native_rid", "other"),
    ("principal_digest", "d" * 64), ("source_session_id", "other"), ("runtime_id", "old"),
    ("runtime_generation", 3), ("model_fingerprint", "other"), ("method", "other"), ("expires_at_epoch", 1),
])
async def test_source_enrollment_binding_before_converter(source_harness, field, value):
    h = source_harness
    h.backend.mutate_enrollment = lambda e: replace(e, **{field: value})
    response, _, _ = await source_call(h)
    assert response.status_code != 200 and not h.tm.calls and not h.chat.converted
    assert len(h.backend.retires) == 1 and h.adapter._active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("capture_commit_id", ""), ("accepted_span_start", 18), ("accepted_span_end", 17),
    ("retained_count", 129), ("predictor_position_offset", 0), ("representation", "mtp"),
    ("layer_index", 1), ("hidden_dim", 512), ("dtype", "float32"), ("layout_digest", "f" * 64),
    ("ready_ranks", 7), ("expected_ranks", 1), ("accepted_sequence_digest", "invalid"),
    ("expires_at_epoch", 1), ("byte_count", 0),
])
async def test_source_worker_commit_must_match_actual_profile(source_harness, field, value):
    h = source_harness
    h.backend.mutate_commit = lambda e: replace(e, **{field: value})
    response, _, _ = await source_call(h)
    assert response.status_code != 200 and SOURCE_RECEIPT_HEADER not in response.headers
    assert len(h.tm.calls) == 1 and len(h.backend.retires) == 1
    assert b"PRIVATE_SOURCE_HANDLE" not in response.body


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("reservation_id", "other"), ("bound_output_sha256", "f" * 64), ("response_id", "client-attempt-1"),
    ("tool_provenance_digest", "f" * 64), ("completed", False), ("gpu_resident", False), ("num_retractions", 1),
])
async def test_source_terminal_evidence_not_advertisement(source_harness, field, value):
    h = source_harness
    h.backend.mutate_source_evidence = lambda e: replace(e, **{field: value})
    response, _, _ = await source_call(h)
    assert response.status_code != 200 and SOURCE_RECEIPT_HEADER not in response.headers
    assert len(h.tm.calls) == 1 and len(h.backend.retires) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("capture_id", "other"), ("source_session_id", "other"),
    ("runtime_generation", 1), ("source_turn_nonce", "other"), ("max_prefix_tokens", 64), ("max_capture_bytes", 1024)])
async def test_source_worker_descriptor_exact_binding(source_harness, field, value):
    h = source_harness
    h.backend.mutate_source_descriptor = lambda d: replace(d, **{field: value})
    response, _, _ = await source_call(h)
    assert response.status_code != 200 and not h.tm.calls and len(h.backend.retires) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["enroll", "prepare", "first_prefetch", "buffer", "evidence"])
async def test_source_disconnect_never_emits_provisional_header(source_harness, phase):
    h, gate = source_harness, asyncio.Event()
    if phase == "enroll": h.backend.enroll_gate = gate
    if phase == "prepare": h.backend.prepare_gate = gate
    if phase == "first_prefetch": h.tm.first_gate = gate
    if phase == "buffer": h.chat.buffer_gate = gate
    if phase == "evidence": h.backend.source_finish_gate = gate
    raw, queue = source_raw()
    sent = []
    async def send(event): sent.append(event)
    async def run():
        response = await h.chat.handle_request(chat_request(), raw)
        await response(raw.scope, raw.receive, send)
        return response
    task = asyncio.create_task(run())
    reached = {"enroll": lambda: h.backend.enrollments, "prepare": lambda: h.backend.prepares,
               "first_prefetch": lambda: h.tm.calls, "buffer": lambda: h.chat.frames,
               "evidence": lambda: h.backend.source_finishes}[phase]
    await spin_until(reached)
    assert sent == []
    queue.put_nowait({"type": "http.disconnect"})
    response = await asyncio.wait_for(task, 2)
    assert response.status_code == 499 and SOURCE_RECEIPT_HEADER not in response.headers
    assert sent[0]["status"] == 499 and len(h.backend.retires) == 1
    if phase in ("enroll", "prepare"): assert not h.tm.calls
    else: assert h.tm.closed == 1 and len(h.tm.aborts) == 1
    assert h.adapter._active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["enroll", "prepare", "first_prefetch", "evidence"])
async def test_source_deadline_and_no_replay(source_harness, phase):
    h, gate, a = source_harness, asyncio.Event(), intent()
    h.backend.required_seconds = .001
    h.adapter.profile = replace(h.adapter.profile, max_turn_seconds=.04, deadline_margin_seconds=.001)
    if phase == "enroll": h.backend.enroll_gate = gate
    if phase == "prepare": h.backend.prepare_gate = gate
    if phase == "first_prefetch": h.tm.first_gate = gate
    if phase == "evidence": h.backend.source_finish_gate = gate
    response, _, _ = await source_call(h, a)
    assert response.status_code == 504 and len(h.backend.retires) == 1
    calls = len(h.tm.calls)
    response, _, _ = await source_call(h, a)
    assert response.status_code == 409 and len(h.tm.calls) == calls


@pytest.mark.asyncio
async def test_source_uncertain_enrollment_holds_quota_until_settled(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, max_active=1, cleanup_seconds=.02)
    started, release = asyncio.Event(), asyncio.Event()
    original = h.backend.enroll_source
    async def stubborn(*args):
        enrolled = await original(*args)
        started.set()
        try: await asyncio.Event().wait()
        except asyncio.CancelledError: await release.wait()
        return enrolled
    h.backend.enroll_source = stubborn
    raw, queue = source_raw()
    task = asyncio.create_task(h.chat.handle_request(chat_request(), raw))
    await started.wait()
    queue.put_nowait({"type": "http.disconnect"})
    response = await asyncio.wait_for(task, 1)
    assert response.status_code == 499 and h.adapter._active == 1 and h.adapter._quarantined
    assert not h.tm.calls and len(h.backend.retires) == 1
    other, _, _ = await source_call(h, intent(capture_id="other"))
    assert other.status_code == 429 and len(h.backend.enrollments) == 1
    release.set()
    await spin_until(lambda: h.adapter._active == 0)
    assert not h.tm.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["enroll", "prepare", "finish"])
async def test_source_private_failure_never_logged_or_echoed(source_harness, phase, caplog):
    h = source_harness
    h.backend.source_failure = phase
    response, _, _ = await source_call(h)
    assert response.status_code != 200
    assert b"PRIVATE_SOURCE_HANDLE" not in response.body
    assert "PRIVATE_SOURCE_HANDLE" not in caplog.text
    assert SOURCE_RECEIPT_HEADER not in response.headers


@pytest.mark.asyncio
async def test_source_full_budget_preflight_and_no_native_mutation(source_harness):
    h = source_harness
    h.backend.required_seconds = 70
    response, _, _ = await source_call(h)
    assert response.status_code == 409 and not h.tm.calls
    assert h.backend.prepares[0][1].max_new_tokens == 32768


@pytest.mark.asyncio
async def test_source_bind_cannot_replace_prompt(source_harness):
    h = source_harness
    h.backend.mutate_source_bind = lambda native: setattr(native, "input_ids", [999])
    response, _, _ = await source_call(h)
    assert response.status_code != 200 and not h.tm.calls


@pytest.mark.parametrize("chunk_size", [1, 3, 97, 4096])
def test_source_hash_includes_exact_utf8_finish_done_and_eof(chunk_size):
    body = FIXTURE["response_body_utf8"].encode() + b" \t\r\n"
    audit = NativeOutputAudit(65536, source=True)
    for start in range(0, len(body), chunk_size): audit.feed(body[start:start + chunk_size])
    completion = audit.finish("server-native-1")
    assert completion.output_sha256 == hashlib.sha256(body).hexdigest()
    assert audit.source_provenance().tool_calls == ()


@pytest.mark.parametrize("tail", [b'data: {"error":{}}\n\n', b'data: [DONE]\n\n', b"unexpected", b'data: {}\n\n'])
def test_source_rejects_protocol_after_done(tail):
    audit = NativeOutputAudit(65536, source=True)
    with pytest.raises(AttachmentError):
        audit.feed(FIXTURE["response_body_utf8"].encode() + tail)
        audit.finish("server-native-1")


@pytest.mark.asyncio
async def test_source_ambiguous_tools_fail_without_guessed_span(source_harness):
    h = source_harness
    def build(request, ret, created):
        return JSONResponse({"id": ret[0]["meta_info"]["id"], "choices": [{"index": 0,
            "message": {"tool_calls": [{"id": "call-1", "function": {"name": "handoff"}},
                                       {"id": "call-2", "function": {"name": "handoff"}}]},
            "finish_reason": "tool_calls"}]})
    h.chat._build_chat_response = build
    response, _, _ = await source_call(h, request=chat_request(stream=False))
    assert response.status_code != 200 and len(h.tm.calls) == 1 and not h.backend.source_finishes


@pytest.mark.asyncio
async def test_source_response_encoding_refused(source_harness):
    h = source_harness
    original = h.chat._build_chat_response
    def build(*args):
        response = original(*args)
        response.headers["content-encoding"] = "gzip"
        return response
    h.chat._build_chat_response = build
    response, _, _ = await source_call(h, request=chat_request(stream=False))
    assert response.status_code != 200 and not h.backend.source_finishes


def sample_descriptor():
    return CaptureWorkerDescriptor("attempt", "reservation", "rid", "a" * 64, "runtime", "model",
        "method", 1000.0, "capture", "source", "turn", 4, "whole_completed_response_tail_v1", 128, 128 * 1024 * 1024)


def test_normalize_worker_descriptor_none_and_exact_instance():
    assert normalize_worker_descriptor(None) is None
    value = sample_descriptor()
    assert normalize_worker_descriptor(value) is value
    assert normalize_worker_descriptor(asdict(value)) == value
    object.__setattr__(value, "runtime_generation", -1)
    with pytest.raises(AttachmentError): normalize_worker_descriptor(value)


@pytest.mark.parametrize("change", [
    {"extra": "text"}, {"operation": "unknown"}, {"schema": "unknown"}, {"runtime_generation": True},
    {"deadline": float("nan")}, {"deadline": float("inf")}, {"deadline": True},
    {"max_capture_bytes": 128 * 1024 * 1024 + 1}, {"max_prefix_tokens": 129},
    {"source_turn_nonce": object()}, {"native_rid": {"tensor": [1]}},
])
def test_normalize_worker_descriptor_rejects_tensor_like_or_unknown(change):
    data = asdict(sample_descriptor()); data.update(change)
    with pytest.raises(AttachmentError): normalize_worker_descriptor(data)


@pytest.mark.parametrize("value", [b"pickle", [], object(), {"schema": "occ3.sglang.worker.v1", "operation": "capture"}])
def test_normalize_worker_descriptor_no_arbitrary_passthrough(value):
    with pytest.raises(AttachmentError): normalize_worker_descriptor(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["enroll", "prepare", "first_prefetch", "buffer", "evidence"])
async def test_source_task_cancellation_has_no_replay(source_harness, phase):
    h, gate, a = source_harness, asyncio.Event(), intent()
    if phase == "enroll": h.backend.enroll_gate = gate
    if phase == "prepare": h.backend.prepare_gate = gate
    if phase == "first_prefetch": h.tm.first_gate = gate
    if phase == "buffer": h.chat.buffer_gate = gate
    if phase == "evidence": h.backend.source_finish_gate = gate
    raw, _ = source_raw(a)
    task = asyncio.create_task(h.chat.handle_request(chat_request(), raw))
    reached = {"enroll": lambda: h.backend.enrollments, "prepare": lambda: h.backend.prepares,
               "first_prefetch": lambda: h.tm.calls, "buffer": lambda: h.chat.frames,
               "evidence": lambda: h.backend.source_finishes}[phase]
    await spin_until(reached)
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert len(h.backend.retires) == 1 and h.adapter._active == 0
    calls = len(h.tm.calls)
    result, _, _ = await source_call(h, a)
    assert result.status_code != 200 and len(h.tm.calls) == calls


@pytest.mark.asyncio
async def test_source_late_commit_after_disconnect_never_becomes_response(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, max_active=1, cleanup_seconds=.02)
    started, release = asyncio.Event(), asyncio.Event()
    original = h.backend.finish_source
    async def stubborn(*args):
        evidence = await original(*args)
        started.set()
        try: await asyncio.Event().wait()
        except asyncio.CancelledError: await release.wait()
        return evidence
    h.backend.finish_source = stubborn
    raw, queue = source_raw()
    task = asyncio.create_task(h.chat.handle_request(chat_request(), raw))
    await started.wait()
    queue.put_nowait({"type": "http.disconnect"})
    response = await asyncio.wait_for(task, 1)
    assert response.status_code == 499 and SOURCE_RECEIPT_HEADER not in response.headers
    assert h.adapter._active == 1 and h.adapter._quarantined and len(h.backend.retires) == 1
    release.set()
    await spin_until(lambda: h.adapter._active == 0)
    assert len(h.tm.calls) == 1 and len(h.backend.source_finishes) == 1
    assert SOURCE_RECEIPT_HEADER not in response.headers


@pytest.mark.asyncio
async def test_source_asgi_headers_wait_for_terminal_evidence(source_harness):
    h = source_harness
    h.backend.source_finish_gate = asyncio.Event()
    raw, _ = source_raw()
    sent = []
    async def send(event): sent.append(event)
    async def run():
        response = await h.chat.handle_request(chat_request(), raw)
        await response(raw.scope, raw.receive, send)
        return response
    task = asyncio.create_task(run())
    await spin_until(lambda: h.backend.source_finishes)
    assert h.tm.closed == 1 and h.chat.frames and sent == []
    h.backend.source_finish_gate.set()
    result = await asyncio.wait_for(task, 1)
    assert sent[0]["status"] == 200
    assert SOURCE_RECEIPT_HEADER.encode() in dict(sent[0]["headers"])
    assert b"".join(message.get("body", b"") for message in sent) == result.body == b"".join(h.chat.frames)


@pytest.mark.asyncio
async def test_source_buffer_bound_suppresses_output_and_receipt(source_harness):
    h = source_harness
    h.backend.max_bytes = 32
    response, _, _ = await source_call(h)
    assert response.status_code == 413 and SOURCE_RECEIPT_HEADER not in response.headers
    assert not h.backend.source_finishes and len(h.backend.retires) == 1 and len(h.tm.calls) == 1


@pytest.mark.asyncio
async def test_source_requires_backend_and_authorizer(source_harness):
    from occ3_sglang.chat_adapter import UnsupportedBackend, unavailable_identity
    h = source_harness
    backend = h.backend
    h.adapter.backend = UnsupportedBackend()
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and not h.tm.calls
    h.adapter.backend = backend
    h.adapter.authorize = unavailable_identity
    response, _, _ = await source_call(h)
    assert response.status_code == 401 and not h.tm.calls and not backend.enrollments


@pytest.mark.asyncio
async def test_source_short_commit_missing_evidence_fails(source_harness):
    h = source_harness
    async def missing(*args): return None
    h.backend.finish_source = missing
    response, _, _ = await source_call(h)
    assert response.status_code != 200 and SOURCE_RECEIPT_HEADER not in response.headers
    assert len(h.tm.calls) == 1 and len(h.backend.retires) == 1


def test_source_tool_fragments_reconstruct_only_observed_id_and_name():
    audit = NativeOutputAudit(65536, source=True)
    def frame(delta, finish=None):
        return b"data: " + canonical({"id": "r", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + b"\n\n"
    body = (frame({"tool_calls": [{"index": 0, "id": "call-", "function": {"name": "hand"}}]})
            + frame({"tool_calls": [{"index": 0, "id": "1", "function": {"name": "off", "arguments": "{}"}}]})
            + frame({}, "tool_calls") + b"data: [DONE]\n\n")
    for byte in body: audit.feed(bytes([byte]))
    audit.finish("r")
    calls = audit.source_provenance().tool_calls
    assert len(calls) == 1 and calls[0].call_id == "call-1" and calls[0].name == "handoff"
    assert "arguments" not in asdict(calls[0])


@pytest.mark.parametrize("change", [{"dtype": {}}, {"hidden_dim": True}, {"byte_count": True},
    {"ready_ranks": True}, {"predictor_position_offset": True}, {"layer_index": 2**32}])
def test_source_receipt_exact_primitive_types(change):
    value = dict(FIXTURE["receipt"]); value.update(change)
    with pytest.raises(AttachmentError): SourceCaptureReceipt.parse(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", [True, None, float("nan"), float("inf")])
async def test_source_missing_or_nonfinite_trusted_deadline_refuses(source_harness, deadline):
    h = source_harness
    async def authorize(raw):
        return AuthenticatedContext("a" * 64, "source-1", "fixture-model", "sha256:fixture", "runtime-1", 4, deadline)
    h.adapter.authorize = authorize
    response, _, _ = await source_call(h)
    assert response.status_code != 200 and not h.backend.enrollments and not h.tm.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [True, None])
async def test_source_invalid_known_time_bound_is_not_capacity_proof(source_harness, seconds):
    h = source_harness
    h.backend.required_seconds = seconds
    response, _, _ = await source_call(h)
    assert response.status_code != 200 and not h.tm.calls
