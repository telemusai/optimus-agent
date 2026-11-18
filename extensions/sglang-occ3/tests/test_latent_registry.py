"""Real stdlib registry tests; fake events model lifetime, not CUDA execution."""
from dataclasses import replace
import pytest

from occ3_sglang.latent_registry import LatentRegistry
from occ3_sglang.latent_types import LatentError, Publication, SessionEnrollment
from occ3_sglang.attachment_contract import ConsumeWorkerDescriptor
from test_latent_types import capture, profile


class Event:
    def __init__(self, done=False):
        self.done = done
    def query(self):
        return self.done


def registry():
    clock = [100.0]
    r = LatentRegistry(profile(), epoch=lambda:clock[0], monotonic=lambda:100.0)
    for session in ("source", "receiver"):
        r.enroll(SessionEnrollment("a"*64, session, "model-hash", "epoch", 9, "family", 500))
    return r, clock


def ready(r):
    rec = r.reserve_capture(capture(), "handle", 123, 400, 128*1024*1024, 4096)
    rec.state, rec.capture_commit_id = "ready", "commit"
    rec.source_metadata = dict(capture_commit_id="commit", expected_ranks=8, retained_count=8,
        hidden_dim=6144, dtype="bfloat16", layout_digest="b"*64)
    r.bind_output("attempt", dict(native_rid="rid", response_id="rid", output_sha256="c"*64,
                                 tool_call_id="tool", tool_name="occ3_handoff", tool_provenance_digest="d"*64))
    return rec


def publication(**changes):
    v = dict(publication_id="publication", source_handle="handle", principal_digest="a"*64,
        source_session_id="source", receiver_session_id="receiver", message_id="message",
        source_turn_nonce="turn", tool_call_id="tool", expires_at_epoch=300)
    v.update(changes)
    return Publication(**v)


def consume(**changes):
    v = dict(attempt_id="consume", reservation_id="cr", native_rid="receiver-rid", principal_digest="a"*64,
        attachment_digest="e"*64, receiver_session_id="receiver", message_id="message", consume_nonce="nonce",
        opaque_handle="handle", state_generation=123, runtime_generation=9, source_capture_commit_id="commit",
        runtime_epoch="epoch", model_fingerprint="model-hash", method_id=profile().method,
        tp_layout_digest="b"*64, expected_rank_count=8, expected_prefix_tokens=8, hidden_dim=6144,
        dtype="bfloat16", deadline=900.0)
    v.update(changes)
    if v.get("prefix_offset") and "prefix_layout" not in changes:
        v["prefix_layout"] = "marker"  # a positive offset names its layout
    return ConsumeWorkerDescriptor(**v)


def test_owner_publish_single_use_and_idempotency():
    r, _ = registry(); rec = ready(r)
    r.publish(publication())
    assert r.publish(publication())["state"] == "published"
    r.reserve_consume(consume())
    assert r.reserve_consume(consume()) is rec
    with pytest.raises(LatentError):
        r.reserve_consume(consume(consume_nonce="other"))
    with pytest.raises(LatentError):
        r.publish(publication(message_id="different"))


@pytest.mark.parametrize("changes", [dict(principal_digest="d"*64), dict(receiver_session_id="stranger"),
    dict(source_turn_nonce="wrong"), dict(tool_call_id="wrong"), dict(expires_at_epoch=999)])
def test_publication_rejects_wrong_identity(changes):
    r, _ = registry(); ready(r)
    with pytest.raises(LatentError):
        r.publish(publication(**changes))


@pytest.mark.parametrize("changes", [dict(runtime_generation=10), dict(state_generation=124),
    dict(expected_prefix_tokens=16), dict(tp_layout_digest="f"*64), dict(source_capture_commit_id="other"),
    dict(receiver_session_id="other"), dict(message_id="other")])
def test_lease_all_bindings(changes):
    r, _ = registry(); ready(r); r.publish(publication())
    with pytest.raises(LatentError):
        r.reserve_consume(consume(**changes))


def test_retire_does_not_free_inflight():
    r, _ = registry(); rec = ready(r)
    event = Event(); rec.events.append(event); rec.prefix = object()
    snap = r.retire(handle="handle")
    assert snap["revoked"] and not snap["gpu_freed"] and r.charged_bytes == 4096
    with pytest.raises(LatentError):
        r.get("handle")
    event.done = True; r.reap()
    assert rec.gpu_freed and rec.prefix is None and r.charged_bytes == 0


def test_expiry_and_quota_remain_charged_until_event_done():
    r, clock = registry(); rec = ready(r)
    event = Event(); rec.events.append(event)
    clock[0] = 401; r.reap()
    assert rec.state == "expired" and r.charged_bytes == 4096
    event.done = True; r.reap()
    assert r.charged_bytes == 0


def test_session_domain_and_output_binding_required():
    r, _ = registry(); rec = ready(r)
    rec.output_binding = None
    with pytest.raises(LatentError):
        r.publish(publication())
    rec.output_binding = dict(tool_call_id="tool",tool_name="occ3_handoff")
    r.sessions[("a"*64,"receiver")] = replace(r.sessions[("a"*64,"receiver")], access_domain="other")
    with pytest.raises(LatentError):
        r.publish(publication())


@pytest.mark.parametrize("tool_name", ["agent_message", "ipython", "functions.agent_message", ""])
def test_only_exact_experimental_handoff_tool_publishes(tool_name):
    r,_=registry(); rec=ready(r); rec.output_binding["tool_name"]=tool_name
    with pytest.raises(LatentError): r.publish(publication())


def test_status_contains_no_payload_objects():
    r,_=registry(); rec=ready(r)
    rec.prefix=object(); rec.retained=[object()]
    snapshot=rec.snapshot()
    assert "prefix" not in snapshot and "retained" not in snapshot and "descriptor" not in snapshot


def test_capture_nonce_cannot_be_reenrolled_with_new_handle():
    r,_=registry(); ready(r)
    with pytest.raises(LatentError):
        r.reserve_capture(capture(),"different",123,400,128*1024*1024,4096)


def test_receiver_attempt_cannot_alias_existing_source_attempt():
    r,_=registry(); ready(r); r.publish(publication())
    with pytest.raises(LatentError): r.reserve_consume(consume(attempt_id="attempt"))
    assert r.attempts["attempt"]=="handle" and r.records["handle"].consumer is None


def test_completed_start_event_is_not_gpu_free_after_partial_submission():
    r,_=registry(); record=ready(r)
    record.prefix=object(); record.inflight=1; record.events.append(Event(done=True))
    assert r.retire(handle="handle")["gpu_freed"] is False
    r.reap()
    assert record.prefix is not None and r.charged_bytes==4096
    # Only worker actual fence or successful end event may clear inflight.
    record.inflight=0; r.reap()
    assert record.gpu_freed and r.charged_bytes==0
