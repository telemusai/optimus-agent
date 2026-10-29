"""Wire/proof parsing tests. Synthetic evidence never proves CUDA use."""
import json
from dataclasses import replace

import pytest

from occ3_sglang.attachment_contract import (
    AttachmentError, CaptureIntent, InferenceAttachment, MAX_RECEIPT_BYTES,
    RECEIPT_HEADER, SPAN_POLICIES_V2, SourceCaptureReceipt, canonical, parse_opt_in,
)
from occ3_sglang.use_receipts import (NativeOutputAudit, receipt_bytes,
                                     source_receipt_upper_bound, source_receipt_v2_upper_bound)
from test_chat_attachment import attachment, call, harness

#: The shared v1 wire receipt (21 fields), verbatim from the pinned fixture.
FIXTURE_RECEIPT = {"intent_digest": "be06b7f857cd1a1ae4a803579ee7ba0b1800617ff64dd3a79f4c9b0f8cc1c11e",
                   "source_handle": "PRIVATE_fixture_unpublished", "native_rid": "server-native-1",
                   "response_id": "server-native-1", "capture_commit_id": "capture-commit-1",
                   "accepted_span_start": 18446744073709551487,
                   "accepted_span_end": 18446744073709551615, "retained_count": 128,
                   "predictor_position_offset": -1,
                   "representation": "final_layer_prenorm_predictor", "layer_index": 77,
                   "hidden_dim": 6144, "dtype": "bfloat16", "layout_digest": "b" * 64,
                   "ready_ranks": 8, "expected_ranks": 8,
                   "accepted_sequence_digest": "c" * 64,
                   "output_sha256": "db8e4c17b56165465210eb4653a11cac7e07339686c769a65723405a950f9a43",
                   "expires_at_epoch": 18446744073709551615, "byte_count": 1572864,
                   "schema": "occ3.source.v1"}
FIXTURE_RECEIPT_FIELDS = tuple(FIXTURE_RECEIPT)


@pytest.mark.parametrize("change", [
    {"protocol_version": True}, {"protocol_version": 2}, {"runtime_generation": -1},
    {"runtime_generation": 2**64}, {"expires_at_epoch": 1.5},
    {"state_handle": "bad secret space"}, {"state_handle": "a" * 257},
    {"message_id": "x\r\n"}, {"method": "é"}, {"request_id": ""},
])
def test_typed_wire_rejects_invalid_fields(change):
    with pytest.raises(AttachmentError): attachment(**change)


@pytest.mark.parametrize("kind", ["null", "unknown", "duplicate", "root_duplicate", "missing", "capture"])
def test_strict_opt_in(kind):
    obj = attachment().wire()
    if kind == "unknown": obj["source_text"] = "must not be accepted"
    if kind == "missing": del obj["message_id"]
    body = canonical({"occ3": obj})
    if kind == "null": body = b'{"occ3":null}'
    if kind == "duplicate": body = body.replace(b'"protocol_version":1', b'"protocol_version":1,"protocol_version":1')
    if kind == "root_duplicate": body = b'{"model":"x","model":"y","occ3":' + canonical(obj) + b'}'
    if kind == "capture": body = b'{"occ3_capture":{}}'
    with pytest.raises(AttachmentError): parse_opt_in(body)


def test_escaped_opt_in_key_is_not_silently_ignored():
    body = rb'{"occ\u0033":' + canonical(attachment().wire()) + b'}'
    assert parse_opt_in(body) is not None


def test_shared_receipt_size_limit_before_generation():
    assert MAX_RECEIPT_BYTES == 2048
    fields = attachment().wire()
    for key in fields:
        if isinstance(fields[key], str): fields[key] = "a" * 256
    with pytest.raises(AttachmentError, match="buffer_limit"):
        receipt_bytes(InferenceAttachment.parse(fields))


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("attempt_id", "wrong"), ("attachment_digest", "wrong"), ("native_rid", "wrong"),
    ("principal_digest", "wrong"), ("runtime_id", "wrong"), ("runtime_generation", 0),
    ("model_fingerprint", "wrong"), ("receiver_session_id", "wrong"),
    ("source_capture_commit_id", ""), ("injection_commit_id", ""),
    ("output_sha256", "wrong"), ("injected_prefix_count", 0), ("injected_prefix_count", True),
    ("ready_ranks", 7), ("expected_ranks", 0), ("completed", False),
    ("fallback_used", True), ("num_retractions", 1), ("private_cache_namespace", False),
    ("gpu_resident", False),
])
async def test_exact_terminal_evidence_required(harness, field, value):
    harness.backend.mutate_evidence = lambda e: replace(e, **{field: value})
    response, _, _ = await call(harness)
    assert response.status_code != 200
    assert RECEIPT_HEADER not in response.headers
    assert len(harness.tm.calls) == 1 and len(harness.backend.retires) == 1


def event(rid="r", finish=None, delta=None):
    return b"data: " + canonical({"id": rid, "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}) + b"\n\n"


@pytest.mark.parametrize("body", [
    b"data: [DONE]\n\n", event(finish="stop"),
    event(finish="stop") + b"data: [DONE]garbage\n\n",
    event(finish="stop") + b"data: [DONE]\n\ndata: [DONE]\n\n",
    event(finish="stop") + b'data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n',
    event(finish="error") + b"data: [DONE]\n\n",
    event(finish="stop") + event(rid="other") + b"data: [DONE]\n\n",
    event(finish="stop") + event(delta={"content": "after-finish"}) + b"data: [DONE]\n\n",
])
def test_invalid_sse_cannot_prove_completion(body):
    audit = NativeOutputAudit(65536)
    with pytest.raises(AttachmentError):
        audit.feed(body)
        audit.finish("r")


def test_sse_arbitrary_byte_splits_and_exact_digest():
    import hashlib
    body = event(delta={"reasoning_content": "before tool"}) + event(finish="tool_calls") + b"data: [DONE]\n\n"
    audit = NativeOutputAudit(65536)
    for byte in body: audit.feed(bytes([byte]))
    result = audit.finish("r")
    assert result.output_sha256 == hashlib.sha256(body).hexdigest()
    assert result.output_bytes == len(body)


@pytest.mark.asyncio
async def test_native_error_plus_done_is_not_success(harness):
    harness.chat.stream_tail = ['data: {"error":{"message":"not success"}}\n\n', 'data: [DONE]\n\n']
    response, _, _ = await call(harness)
    assert response.status_code != 200
    assert RECEIPT_HEADER not in response.headers
    assert not harness.backend.finishes


def test_v2_span_policies_are_schema_gated():
    base = ["capture", "source-session", "source-request", "turn", "model", "runtime",
            1, "statebridge", 128, 128 * 1024 * 1024, 1000]
    for policy in SPAN_POLICIES_V2:
        intent = CaptureIntent(*base, schema="occ3.capture.v2", span_policy=policy)
        assert intent.span_policy == policy
        # A v2 name is refused under the v1 schema.
        with pytest.raises(AttachmentError):
            CaptureIntent(*base, span_policy=policy)
    # The v1 default is refused under the v2 schema, and so is an unknown schema.
    with pytest.raises(AttachmentError):
        CaptureIntent(*base, schema="occ3.capture.v2")
    with pytest.raises(AttachmentError):
        CaptureIntent(*base, schema="occ3.capture.v3", span_policy=SPAN_POLICIES_V2[0])


def span_block(**changes):
    values = {"span_schema": "occ3.span.v1", "span_policy": "content_tail_v2",
              "span_marker_profile": "glm45_glm47_xml_kv.v1",
              "span_marker_schema": "occ3.span.markers.v1",
              "span_reasoning_terminator": "thinking_end",
              "span_turn_end_attested": True,
              "span_thinking_count": 2, "span_content_count": 6,
              "span_tool_call_count": 0, "span_tool_arguments_count": 0,
              "span_control_count": 2, "span_unknown_count": 0,
              "span_window_start_position": 9, "span_window_end_position": 19,
              "span_contiguous": True, "span_contains_final_content": True,
              "span_contains_tool_call": False, "span_message_truncated": False}
    values.update(changes)
    return values


def v2_receipt(**changes):
    block = changes.pop("block", {})
    values = {"intent_digest": "f" * 64, "source_handle": "handle", "native_rid": "rid",
              "response_id": "rid", "capture_commit_id": "commit",
              "accepted_span_start": 10, "accepted_span_end": 18, "retained_count": 8,
              "predictor_position_offset": -1, "representation": "final_layer_prenorm_predictor",
              "layer_index": 77, "hidden_dim": 6144, "dtype": "bfloat16",
              "layout_digest": "b" * 64, "ready_ranks": 8, "expected_ranks": 8,
              "accepted_sequence_digest": "c" * 64, "output_sha256": "d" * 64,
              "expires_at_epoch": 1000, "byte_count": 8 * 6144 * 2,
              "schema": "occ3.source.v2"}
    values.update(changes)
    return SourceCaptureReceipt(**values, **span_block(**block))


def test_v2_receipt_wire_and_parse_round_trip():
    receipt = v2_receipt()
    wire = receipt.wire()
    assert len(wire) == 39
    assert set(wire) - set(span_block()) == set(FIXTURE_RECEIPT_FIELDS)
    assert SourceCaptureReceipt.parse(wire) == receipt
    # A v1 receipt keeps the 21-field wire: the fixture canonical bytes pin it.
    v1 = dict(FIXTURE_RECEIPT)
    assert len(v1) == 21 and "span_policy" not in v1
    assert SourceCaptureReceipt(**v1).schema == "occ3.source.v1"
    assert len(SourceCaptureReceipt(**v1).wire()) == 21


def test_v2_receipt_parse_requires_all_or_none_of_the_block():
    wire = v2_receipt().wire()
    missing = dict(wire); del missing["span_content_count"]
    with pytest.raises(AttachmentError):
        SourceCaptureReceipt.parse(missing)
    extra = dict(wire); extra["span_invented"] = 1
    with pytest.raises(AttachmentError):
        SourceCaptureReceipt.parse(extra)
    # A v1-schema receipt refuses any block field, and a v2 schema without one.
    with pytest.raises(AttachmentError):
        SourceCaptureReceipt.parse(dict(v2_receipt().wire(), schema="occ3.source.v1"))
    with pytest.raises(AttachmentError):
        SourceCaptureReceipt.parse({k: v for k, v in v2_receipt().wire().items()
                                    if not k.startswith("span_")})


def test_v2_span_block_cross_field_invariants_fail_closed():
    for changes in ({"span_content_count": 129},
                   {"span_content_count": 5},
                   {"span_thinking_count": 0},
                   {"span_policy": "content_tail_no_tool_v2",
                    "span_contains_tool_call": True},
                   {"span_turn_end_attested": False},
                   {"span_contiguous": False},
                   {"span_schema": "occ3.span.v2"},
                   {"span_policy": "whole_completed_response_tail_v1"},
                   {"span_marker_schema": "occ3.span.markers.v2"},
                   {"span_reasoning_terminator": "guessed"},
                   {"span_contiguous": True, "span_turn_end_attested": True, "span_contains_final_content": 1}):
        with pytest.raises(AttachmentError):
            v2_receipt(block=changes)


def test_v2_receipt_upper_bound_is_measured_and_under_the_cap():
    upper = source_receipt_v2_upper_bound(MAX_RECEIPT_BYTES)
    assert len(upper) <= 2048
    parsed = SourceCaptureReceipt.parse(json.loads(upper.decode()))
    assert len(parsed.wire()) == 39
    assert parsed.span_policy == max(SPAN_POLICIES_V2)
    v1 = source_receipt_upper_bound(MAX_RECEIPT_BYTES)
    assert len(v1) < len(upper) <= MAX_RECEIPT_BYTES


def test_capture_intent_is_owner_only_negotiated_directive():
    intent = CaptureIntent("capture", "source-session", "source-request", "turn", "model", "runtime",
                           1, "statebridge", 128, 128 * 1024 * 1024, 1000)
    assert "source-session" not in repr(intent)
    with pytest.raises(AttachmentError):
        replace(intent, retained_count_limit=129)
    with pytest.raises(AttachmentError):
        replace(intent, span_policy="guessed-tool-substring")
