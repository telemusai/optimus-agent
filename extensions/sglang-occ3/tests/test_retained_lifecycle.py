"""Retained-capture publication mode: content-only capture + orchestrator dispatch.

RED-first suite for the occ3.capture.v3 / occ3.publish.v2 wire pair. The
legacy same-turn tool protocol (occ3.capture.v1/v2 with occ3.publish.v1)
must stay byte-identical: parse/wire/digest round-trips of old bodies are
pinned here alongside the new-mode positives and refusals.
"""
import json
import time
from dataclasses import asdict, replace
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
import httpx

from occ3_sglang.attachment_contract import (AttachmentError, CaptureIntent,
                                             CaptureWorkerDescriptor, SOURCE_RECEIPT_HEADER,
                                             SourceCaptureReceipt, canonical, digest)
from occ3_sglang.latent_registry import LatentRegistry
from occ3_sglang.latent_types import (LatentError, Publication, SessionEnrollment,
                                      decode_descriptor)
from occ3_sglang.latent_control_api import BrokerAuthority, create_router
from test_chat_attachment import harness
from test_chat_source_capture import V2_BLOCK, intent, source_call, source_harness
from test_latent_backend import health
from test_latent_types import capture, profile
from test_span_markers import FakeTokenizer

CAPTURE_MODE = "content_only_orchestrator_v1"
NO_TOOL_POLICY = "content_tail_no_tool_v2"


def intent_body(**changes):
    value = dict(capture_id="capture", source_session_id="source", source_request_id="client-rid",
                 turn_nonce="turn", model_fingerprint="model-hash", runtime_id="epoch",
                 runtime_generation=9, method=profile().method, retained_count_limit=8,
                 max_capture_bytes=128 * 1024 * 1024, expires_at_epoch=1000,
                 schema="occ3.capture.v3", span_policy=NO_TOOL_POLICY, capture_mode=CAPTURE_MODE)
    value.update(changes)
    return value


def legacy_intent_body(**changes):
    value = intent_body(schema="occ3.capture.v1", span_policy="whole_completed_response_tail_v1")
    value.pop("capture_mode")
    value.update(changes)
    return value


# ---------------------------------------------------------------------------
# a) v3 CaptureIntent parsing.
# ---------------------------------------------------------------------------

def test_v3_intent_parse_accepts_content_only_no_tool():
    intent = CaptureIntent.parse(intent_body())
    assert intent.schema == "occ3.capture.v3"
    assert intent.capture_mode == CAPTURE_MODE
    assert intent.span_policy == NO_TOOL_POLICY
    assert CaptureIntent.parse(intent.wire()) == intent
    assert "capture_mode" in intent.wire() and len(intent.wire()) == 14


def test_capture_mode_on_v1_or_v2_body_is_refused():
    with pytest.raises(AttachmentError):
        CaptureIntent.parse(intent_body(schema="occ3.capture.v1",
                                        span_policy="whole_completed_response_tail_v1"))
    with pytest.raises(AttachmentError):
        CaptureIntent.parse(intent_body(schema="occ3.capture.v2"))
    # Even an empty capture_mode may not ride a v1/v2 wire.
    with pytest.raises(AttachmentError):
        CaptureIntent.parse(legacy_intent_body(capture_mode=""))


def test_v3_with_wrong_span_policy_is_refused():
    for policy in ("content_tail_v2", "content_tail_last64_v2", "content_tail_last32_v2",
                   "whole_completed_response_tail_v1"):
        with pytest.raises(AttachmentError):
            CaptureIntent.parse(intent_body(span_policy=policy))


def test_v3_with_wrong_capture_mode_is_refused():
    for mode in ("", "same_turn_tool_v1", "content_only_orchestrator_v2", 5):
        with pytest.raises(AttachmentError):
            CaptureIntent.parse(intent_body(capture_mode=mode))


def test_v3_body_without_capture_mode_is_refused():
    body = intent_body()
    del body["capture_mode"]
    with pytest.raises(AttachmentError):
        CaptureIntent.parse(body)


def test_v1_v2_intents_refuse_a_non_empty_capture_mode():
    with pytest.raises(AttachmentError):
        CaptureIntent(**legacy_intent_body(capture_mode=CAPTURE_MODE))
    with pytest.raises(AttachmentError):
        CaptureIntent(**intent_body(schema="occ3.capture.v2", capture_mode=CAPTURE_MODE))


def test_legacy_v1_v2_wire_and_digest_are_unchanged():
    v1 = CaptureIntent.parse(legacy_intent_body())
    v2 = CaptureIntent.parse(legacy_intent_body(schema="occ3.capture.v2",
                                                 span_policy="content_tail_v2"))
    for intent in (v1, v2):
        wire = intent.wire()
        assert "capture_mode" not in wire and len(wire) == 13
        assert CaptureIntent.parse(wire) == intent
        # The digest is over the legacy field set only: byte-identical wire.
        assert intent.binding_digest == digest(wire)
    assert v1.capture_mode == "" and v2.capture_mode == ""


# ---------------------------------------------------------------------------
# b) Publication modes.
# ---------------------------------------------------------------------------

def publication(**changes):
    v = dict(publication_id="publication", source_handle="handle", principal_digest="a" * 64,
             source_session_id="source", receiver_session_id="receiver", message_id="message",
             source_turn_nonce="turn", tool_call_id="tool", expires_at_epoch=300)
    v.update(changes)
    return Publication(**v)


def test_orchestrator_publication_requires_the_empty_tool_call_id():
    with pytest.raises((LatentError, AttachmentError)):
        publication(dispatch_mode="orchestrator_v1", tool_call_id="tool")
    ok = publication(dispatch_mode="orchestrator_v1", tool_call_id="")
    assert ok.dispatch_mode == "orchestrator_v1" and ok.tool_call_id == ""


def test_model_tool_publication_keeps_the_identifier_requirement():
    assert publication().dispatch_mode == "model_tool_v1"
    for bad in ("", "with space", 5, None):
        with pytest.raises((LatentError, AttachmentError)):
            publication(tool_call_id=bad)
    # The identifier skip applies to tool_call_id only, never to other fields.
    with pytest.raises((LatentError, AttachmentError)):
        publication(dispatch_mode="orchestrator_v1", tool_call_id="", publication_id="with space")


# ---------------------------------------------------------------------------
# c) registry.publish dispatch modes.
# ---------------------------------------------------------------------------

def registry():
    clock = [100.0]
    r = LatentRegistry(profile(), epoch=lambda: clock[0], monotonic=lambda: 100.0)
    for session in ("source", "receiver"):
        r.enroll(SessionEnrollment("a" * 64, session, "model-hash", "epoch", 9, "family", 500))
    return r, clock


def ready_record(r, *, policy=None, capture_mode=None, tool_binding=None):
    changes = {}
    if policy is not None:
        changes.update(span_policy=policy, marker_profile="glm45_glm47_xml_kv.v1")
    if capture_mode is not None:
        changes["capture_mode"] = capture_mode
    descriptor = capture(**changes)
    rec = r.reserve_capture(descriptor, "handle", 123, 400, 128 * 1024 * 1024, 4096)
    rec.state, rec.capture_commit_id = "ready", "commit"
    rec.source_metadata = dict(capture_commit_id="commit", expected_ranks=8, retained_count=8,
        hidden_dim=6144, dtype="bfloat16", layout_digest="b" * 64)
    if tool_binding is None:
        tool_binding = "tool" if policy is None else ""
    r.bind_output("attempt", dict(native_rid="rid", response_id="rid", output_sha256="c" * 64,
        tool_call_id=tool_binding, tool_name="occ3_handoff" if tool_binding else "",
        tool_provenance_digest="d" * 64))
    return rec


def no_tool_record(r, *, policy=NO_TOOL_POLICY, capture_mode=CAPTURE_MODE):
    return ready_record(r, policy=policy, capture_mode=capture_mode, tool_binding="")


def test_orchestrator_publish_on_a_no_tool_record_publishes():
    r, _ = registry(); no_tool_record(r)
    snapshot = r.publish(publication(dispatch_mode="orchestrator_v1", tool_call_id=""))
    assert snapshot["state"] == "published"
    assert r.publications["publication"].dispatch_mode == "orchestrator_v1"


def test_orchestrator_publish_is_one_shot_and_idempotent():
    r, _ = registry(); no_tool_record(r)
    pub = publication(dispatch_mode="orchestrator_v1", tool_call_id="")
    assert r.publish(pub)["state"] == "published"
    assert r.publish(pub)["state"] == "published"
    with pytest.raises(LatentError):
        r.publish(publication(dispatch_mode="orchestrator_v1", tool_call_id="", message_id="different"))


@pytest.mark.parametrize("changes", [dict(source_turn_nonce="wrong"), dict(receiver_session_id="stranger"),
                                     dict(principal_digest="d" * 64), dict(expires_at_epoch=999)])
def test_orchestrator_publish_keeps_identity_and_expiry_checks(changes):
    r, _ = registry(); no_tool_record(r)
    with pytest.raises(LatentError):
        r.publish(publication(dispatch_mode="orchestrator_v1", tool_call_id="", **changes))


def test_orchestrator_publish_refused_on_a_tool_bound_record():
    r, _ = registry(); ready_record(r)
    with pytest.raises(LatentError):
        r.publish(publication(dispatch_mode="orchestrator_v1", tool_call_id=""))


def test_orchestrator_publish_refused_without_the_no_tool_span_policy():
    # The descriptor gate refuses this combination at construction, so the
    # record is hand-mutated to probe the registry's own policy check.
    r, _ = registry(); rec = no_tool_record(r)
    object.__setattr__(rec.descriptor, "span_policy", "content_tail_v2")
    with pytest.raises(LatentError):
        r.publish(publication(dispatch_mode="orchestrator_v1", tool_call_id=""))


def test_orchestrator_publish_refused_without_the_negotiated_capture_mode():
    r, _ = registry()
    ready_record(r, policy=NO_TOOL_POLICY, capture_mode="", tool_binding="")
    with pytest.raises(LatentError):
        r.publish(publication(dispatch_mode="orchestrator_v1", tool_call_id=""))


def test_orchestrator_publish_refused_without_any_output_binding():
    r, _ = registry(); rec = no_tool_record(r)
    rec.output_binding = None
    with pytest.raises(LatentError):
        r.publish(publication(dispatch_mode="orchestrator_v1", tool_call_id=""))


def test_model_tool_publish_refused_on_an_empty_binding():
    r, _ = registry(); no_tool_record(r)
    with pytest.raises(LatentError):
        r.publish(publication())


def test_legacy_model_tool_publish_still_passes():
    r, _ = registry(); ready_record(r)
    assert r.publish(publication())["state"] == "published"
    assert r.publish(publication())["state"] == "published"


def test_unknown_dispatch_mode_is_refused_not_defaulted():
    r, _ = registry(); no_tool_record(r)
    pub = publication(dispatch_mode="orchestrator_v1", tool_call_id="")
    object.__setattr__(pub, "dispatch_mode", "carrier_pigeon_v1")
    with pytest.raises(LatentError):
        r.publish(pub)
    legacy = publication()
    object.__setattr__(legacy, "dispatch_mode", "carrier_pigeon_v1")
    with pytest.raises(LatentError):
        r.publish(legacy)


# ---------------------------------------------------------------------------
# d) + e) publish route parsing and digest consistency.
# ---------------------------------------------------------------------------

class PublishSpyClient:
    def __init__(self, p):
        self.p, self.calls = p, []
    async def call(self, op, payload, command_id=None):
        self.calls.append((op, payload, command_id))
        if op == "health":
            return health(self.p)
        return dict(source_handle=payload.get("source_handle", "handle"), state="published",
                    expires_at_epoch=300)
    async def close(self):
        pass


@pytest.fixture
def route():
    p = profile(); authority = BrokerAuthority(p, principal_id="server-owned")
    client = PublishSpyClient(p)
    async def refresh():
        return health(p)
    backend = NS(client=client, refresh_readiness=refresh, profile=p)
    for session in ("source", "receiver"):
        authority.remember_session(SessionEnrollment(
            authority.principal_digest, session, p.model_fingerprint, p.runtime_epoch, 9,
            "family", int(time.time()) + 100))
    app = FastAPI()
    @app.middleware("http")
    async def auth(request, call_next):
        authority.clear_verified(request)
        authority.mark_verified(request, "server-owned", deadline_monotonic=time.monotonic() + 300)
        return await call_next(request)
    app.include_router(create_router(backend, authority))
    transport = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return transport, client, authority


def publish_v1_body(**changes):
    value = dict(schema="occ3.publish.v1", publication_id="publication", source_handle="handle",
                 source_session_id="source", receiver_session_id="receiver", message_id="message",
                 source_turn_nonce="turn", tool_call_id="tool", expires_at_epoch=300)
    value.update(changes)
    return value


def publish_v2_body(**changes):
    value = dict(schema="occ3.publish.v2", publication_id="publication", source_handle="handle",
                 source_session_id="source", receiver_session_id="receiver", message_id="message",
                 source_turn_nonce="turn", dispatch_mode="orchestrator_v1", expires_at_epoch=300)
    value.update(changes)
    return value


async def test_publish_route_v1_body_stays_model_tool(route):
    client, spy, _ = route
    response = await client.post("/occ3/v1/native/publish", json=publish_v1_body())
    assert response.status_code == 200, response.text
    op, payload, _ = spy.calls[-1]
    assert op == "publish" and payload["dispatch_mode"] == "model_tool_v1"
    assert payload["tool_call_id"] == "tool"
    assert response.json()["schema"] == "occ3.publication.v1"


async def test_publish_route_v2_body_parses_to_orchestrator(route):
    client, spy, _ = route
    response = await client.post("/occ3/v1/native/publish", json=publish_v2_body())
    assert response.status_code == 200, response.text
    op, payload, _ = spy.calls[-1]
    assert op == "publish"
    assert payload["dispatch_mode"] == "orchestrator_v1" and payload["tool_call_id"] == ""
    assert response.json()["state"] == "published"


async def test_publish_route_unknown_schema_is_refused(route):
    client, spy, _ = route
    for schema in ("occ3.publish.v3", "occ3.publish.v0", ""):
        response = await client.post("/occ3/v1/native/publish", json=publish_v2_body(schema=schema))
        assert response.status_code in (400, 409), schema
    assert not spy.calls


async def test_publish_route_v2_with_tool_call_id_is_refused(route):
    client, spy, _ = route
    response = await client.post("/occ3/v1/native/publish",
                                 json=publish_v2_body(tool_call_id="tool"))
    assert response.status_code in (400, 409)
    # A v2 body missing dispatch_mode matches neither field set either.
    body = publish_v2_body(); del body["dispatch_mode"]
    response = await client.post("/occ3/v1/native/publish", json=body)
    assert response.status_code in (400, 409)
    assert not spy.calls


async def test_publish_route_v2_with_unknown_dispatch_mode_is_refused(route):
    client, spy, _ = route
    response = await client.post("/occ3/v1/native/publish",
                                 json=publish_v2_body(dispatch_mode="carrier_pigeon_v1"))
    assert response.status_code in (400, 409)
    assert not spy.calls


async def test_publish_route_v2_digest_over_the_specified_field_set(route):
    client, _, authority = route
    response = await client.post("/occ3/v1/native/publish", json=publish_v2_body())
    assert response.status_code == 200, response.text
    expected = digest(dict(publication_id="publication", source_handle="handle",
        principal_digest=authority.principal_digest, source_session_id="source",
        receiver_session_id="receiver", message_id="message", source_turn_nonce="turn",
        dispatch_mode="orchestrator_v1", expires_at_epoch=300))
    assert response.json()["binding_digest"] == expected
    # The Rust v2 client computes the same digest over its own body minus the
    # schema key plus the principal digest.
    rust_side = digest({k: v for k, v in publish_v2_body().items() if k != "schema"}
                       | {"principal_digest": authority.principal_digest})
    assert rust_side == expected


async def test_publish_route_v1_digest_stays_the_legacy_field_set(route):
    client, _, authority = route
    response = await client.post("/occ3/v1/native/publish", json=publish_v1_body())
    assert response.status_code == 200, response.text
    expected = digest(dict(publication_id="publication", source_handle="handle",
        principal_digest=authority.principal_digest, source_session_id="source",
        receiver_session_id="receiver", message_id="message", source_turn_nonce="turn",
        tool_call_id="tool", expires_at_epoch=300))
    assert response.json()["binding_digest"] == expected
    rust_side = digest({k: v for k, v in publish_v1_body().items() if k != "schema"}
                       | {"principal_digest": authority.principal_digest})
    assert rust_side == expected


# ---------------------------------------------------------------------------
# 5) + 6) descriptor threading through the backend path.
# ---------------------------------------------------------------------------

def test_capture_descriptor_carries_capture_mode_under_the_v3_rules():
    descriptor = capture(span_policy=NO_TOOL_POLICY, marker_profile="glm45_glm47_xml_kv.v1",
                        capture_mode=CAPTURE_MODE)
    assert descriptor.capture_mode == CAPTURE_MODE
    assert decode_descriptor(asdict(descriptor)) == descriptor
    # Empty means the legacy same-turn tool protocol.
    assert capture().capture_mode == ""
    # A non-empty mode must be the v3 value and must agree with the policy.
    with pytest.raises(AttachmentError):
        capture(span_policy=NO_TOOL_POLICY, marker_profile="glm45_glm47_xml_kv.v1",
                capture_mode="same_turn_tool_v1")
    with pytest.raises(AttachmentError):
        capture(span_policy="content_tail_v2", marker_profile="glm45_glm47_xml_kv.v1",
                capture_mode=CAPTURE_MODE)
    with pytest.raises(AttachmentError):
        capture(capture_mode=CAPTURE_MODE)


async def test_enroll_source_threads_capture_mode_into_the_descriptor():
    from test_latent_backend import fixture
    backend, client, identity, expiry = fixture(); p = backend.profile
    client.tm = NS(tokenizer=FakeTokenizer())
    intent = CaptureIntent(capture_id="capture", source_session_id="source",
        source_request_id="client-rid", turn_nonce="turn", model_fingerprint=p.model_fingerprint,
        runtime_id=p.runtime_epoch, runtime_generation=9, method=p.method, retained_count_limit=8,
        max_capture_bytes=128 * 1024 * 1024, expires_at_epoch=expiry, schema="occ3.capture.v3",
        span_policy=NO_TOOL_POLICY, capture_mode=CAPTURE_MODE)
    await backend.enroll_source(intent, identity, "attempt", "native-rid", identity.deadline_monotonic)
    descriptor = backend.attempts["attempt"]["descriptor"]
    assert descriptor.capture_mode == CAPTURE_MODE and descriptor.span_policy == NO_TOOL_POLICY


async def test_enroll_source_keeps_the_legacy_descriptor_empty():
    from test_latent_backend import fixture
    backend, client, identity, expiry = fixture(); p = backend.profile
    intent = CaptureIntent(capture_id="capture", source_session_id="source",
        source_request_id="client-rid", turn_nonce="turn", model_fingerprint=p.model_fingerprint,
        runtime_id=p.runtime_epoch, runtime_generation=9, method=p.method, retained_count_limit=8,
        max_capture_bytes=128 * 1024 * 1024, expires_at_epoch=expiry)
    await backend.enroll_source(intent, identity, "attempt", "native-rid", identity.deadline_monotonic)
    assert backend.attempts["attempt"]["descriptor"].capture_mode == ""


async def test_finish_source_binds_the_empty_tool_identity_for_a_no_tool_turn():
    from occ3_sglang.chat_adapter import NativeRequestView
    from occ3_sglang.use_receipts import NativeCompletion, SourceToolProvenance
    from test_latent_backend import fixture
    backend, client, identity, expiry = fixture(); p = backend.profile
    intent = CaptureIntent(capture_id="capture", source_session_id="source",
        source_request_id="client-rid", turn_nonce="turn", model_fingerprint=p.model_fingerprint,
        runtime_id=p.runtime_epoch, runtime_generation=9, method=p.method, retained_count_limit=8,
        max_capture_bytes=128 * 1024 * 1024, expires_at_epoch=expiry)
    enrollment = await backend.enroll_source(intent, identity, "attempt", "native-rid",
                                             identity.deadline_monotonic)
    native = NS(rid="native-rid", sampling_params={"max_new_tokens": 200}, input_ids=[1, 2], stream=True)
    view = NativeRequestView(native, 200, "s", True, identity.deadline_monotonic)
    reservation = await backend.prepare_source(enrollment, view)
    backend.bind_source(enrollment, reservation, native)
    client.record["source"] = dict(capture_commit_id="commit", accepted_span_start=2,
        accepted_span_end=10, retained_count=8, predictor_position_offset=-1,
        representation="final_layer_prenorm_predictor", layer_index=77, hidden_dim=6144,
        dtype="bfloat16", layout_digest="b" * 64, ready_ranks=8, expected_ranks=8,
        accepted_sequence_digest="c" * 64, expires_at_epoch=expiry, byte_count=8 * 6144 * 2)
    completion = NativeCompletion("native-rid", "stop", "d" * 64, 50, True)
    await backend.finish_source(enrollment, reservation, completion, SourceToolProvenance(tool_calls=()))
    binding = client.calls[-1][1]["binding"]
    assert binding["tool_call_id"] == "" and binding["tool_name"] == ""


# ---------------------------------------------------------------------------
# End to end: a v3 content-only turn through the real chat adapter.
# ---------------------------------------------------------------------------

async def test_v3_content_only_turn_flows_through_the_chat_adapter(source_harness):
    from starlette.responses import JSONResponse

    h = source_harness
    h.tm.tokenizer = FakeTokenizer()
    # A no-tool turn: plain content deltas, finish_reason stop, no tool_calls.
    async def content_stream(obj, request, raw):
        async for output in h.tm.generate_request(obj, raw):
            for delta in ({"role": "assistant"}, {"content": "fixture content answer"}):
                frame = ("data: " + json.dumps({"id": obj.rid, "choices": [
                    {"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n")
                h.chat.frames.append(frame.encode())
                yield frame
        for frame in ("data: " + json.dumps({"id": obj.rid, "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n",
                "data: [DONE]\n\n"):
            h.chat.frames.append(frame.encode())
            yield frame
    h.chat._generate_chat_stream = content_stream

    def build(request, ret, created):
        return JSONResponse({"id": ret[0]["meta_info"]["id"], "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "fixture content answer"},
            "finish_reason": "stop"}]})
    h.chat._build_chat_response = build
    block = dict(V2_BLOCK, span_policy=NO_TOOL_POLICY)
    h.backend.mutate_commit = lambda commit: replace(commit, **block)
    a = intent(schema="occ3.capture.v3", span_policy=NO_TOOL_POLICY, capture_mode=CAPTURE_MODE)
    response, _, _ = await source_call(h, a)
    assert response.status_code == 200, response.body
    receipt = SourceCaptureReceipt.parse(json.loads(response.headers[SOURCE_RECEIPT_HEADER]))
    assert receipt.schema == "occ3.source.v2" and receipt.span_policy == NO_TOOL_POLICY
    assert receipt.span_contains_tool_call is False
    assert receipt.intent_digest == a.binding_digest
    provenance = h.backend.source_finishes[0][3]
    assert provenance.tool_calls == () and provenance.scope == NO_TOOL_POLICY
    descriptor = h.tm.calls[0].occ3_worker_descriptor
    assert descriptor.capture_mode == CAPTURE_MODE
    assert h.adapter._active == 0 and not h.backend.retires


async def test_v3_descriptor_capture_mode_mismatch_is_invalid_evidence(source_harness):
    # The adapter's source-descriptor agreement includes the capture mode:
    # a descriptor that dropped the negotiated mode is invalid evidence.
    h = source_harness
    h.tm.tokenizer = FakeTokenizer()
    h.backend.mutate_commit = lambda commit: replace(commit, **dict(V2_BLOCK, span_policy=NO_TOOL_POLICY))
    h.backend.mutate_source_descriptor = lambda d: replace(d, capture_mode="")
    a = intent(schema="occ3.capture.v3", span_policy=NO_TOOL_POLICY, capture_mode=CAPTURE_MODE)
    response, _, _ = await source_call(h, a)
    assert response.status_code != 200 and SOURCE_RECEIPT_HEADER not in response.headers
    assert len(h.backend.retires) == 1
