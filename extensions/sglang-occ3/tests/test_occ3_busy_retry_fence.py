"""Retry design (a): a pre-dispatch busy refusal frees its own replay fence.

Offline adapter/registry tests with the pinned SGLang handler and dependency
stubs; not live scheduler evidence.
"""
import asyncio
import json
import time
from dataclasses import replace

import pytest

from occ3_sglang.attachment_contract import AttachmentError, canonical
from occ3_sglang.latent_types import LatentError
from test_chat_attachment import attachment, chat_request, harness, raw_request  # noqa: F401
from test_chat_source_capture import intent, source_call, source_harness  # noqa: F401
from test_latent_registry import consume, publication, ready, registry


def _aborting(h, messages):
    """Native generator yielding one scheduler abort per call (or a normal turn)."""
    original = type(h.tm).generate_request.__wrapped__ if hasattr(
        type(h.tm).generate_request, "__wrapped__") else None
    calls = []

    async def generate(tm, obj, request=None):
        tm.calls.append(obj)
        message = messages[len(calls)] if len(calls) < len(messages) else None
        calls.append(message)
        if message is None:
            yield {"text": "fixture", "meta_info": {"id": obj.rid, "finish_reason": {"type": "stop"}}}
            return
        yield {"text": "", "output_ids": [],
               "meta_info": {"id": obj.rid, "finish_reason": {"type": "abort", "message": message,
                                                               "status_code": 503}}}
    type(h.tm).generate_request = h.wrap(generate, h.adapter.generate_around, h.HookType.AROUND)
    return calls


class _Proof:
    """Worker-side non-use proof stub: records calls, result is configurable."""
    def __init__(self, result="released", error=None):
        self.calls, self.result, self.error = [], result, error

    async def __call__(self, attempt_id, native_rid):
        self.calls.append((attempt_id, native_rid))
        if self.error:
            raise LatentError(self.error)
        return self.result


async def consume_call(h, a=None):
    raw, _ = raw_request(canonical({"occ3": (a or attachment()).wire()}))
    return await h.chat.handle_request(chat_request(), raw)


def _reusable_consume(h):
    # The fixture refuses a second reserve of the same handle; the real worker
    # refuses through the registry, which the release proof resets.
    reserve = h.backend.reserve
    async def again(a, *args):
        h.backend.seen.discard(a.state_handle)
        return await reserve(a, *args)
    h.backend.reserve = again


def assert_consume_replay(h, response, reserves, calls):
    """The consume fence fires inside the native stream, where the pinned
    first-chunk seam flattens replay_forbidden into the pre-existing 409
    native_completion_failed shape; the fence itself is proven by the absence
    of any second reserve or native dispatch."""
    assert response.status_code == 409
    assert len(h.backend.reserves) == reserves and len(h.tm.calls) == calls


# -------------------------------------------------------------- consume path
@pytest.mark.asyncio
@pytest.mark.parametrize("message,public", [("occ3_busy_batch", "scheduler_busy"),
                                            ("occ3_busy", "admission_capacity")])
async def test_consume_busy_refusal_then_identical_resend_is_admitted(harness, message, public):
    h = harness
    _reusable_consume(h)
    proof = h.backend.release_unused_attempt = _Proof()
    _aborting(h, [message])
    a = attachment()
    first = await consume_call(h, a)
    assert first.status_code == 503 and json.loads(first.body)["error"]["message"] == public
    assert first.headers["retry-after"] == "5"
    assert len(proof.calls) == 1 and not h.backend.retires   # lease returned, not revoked
    assert not h.adapter._attempts
    second = await consume_call(h, a)
    assert second.status_code == 200
    assert len(h.backend.reserves) == 2


@pytest.mark.asyncio
async def test_consume_fence_stays_when_the_worker_cannot_prove_non_use(harness):
    h = harness
    _reusable_consume(h)
    h.backend.release_unused_attempt = _Proof(error="state")
    _aborting(h, ["occ3_busy_batch"])
    a = attachment()
    first = await consume_call(h, a)
    assert first.status_code == 503
    assert len(h.backend.retires) == 1      # normal cleanup retired the lease
    assert len(h.adapter._attempts) == 1
    _before = (len(h.backend.reserves), len(h.tm.calls))
    second = await consume_call(h, a)
    assert_consume_replay(h, second, *_before)


@pytest.mark.asyncio
async def test_consume_post_dispatch_admission_timeout_keeps_the_fence(harness):
    """Opus review: admission_timeout can follow dispatch (first-event bound)."""
    h = harness
    _reusable_consume(h)
    proof = h.backend.release_unused_attempt = _Proof()
    _aborting(h, ["occ3_admission_timeout"])
    a = attachment()
    first = await consume_call(h, a)
    assert first.status_code == 503 and b"admission_timeout" in first.body
    assert not proof.calls and len(h.backend.retires) == 1
    _before = (len(h.backend.reserves), len(h.tm.calls))
    second = await consume_call(h, a)
    assert_consume_replay(h, second, *_before)


@pytest.mark.asyncio
async def test_consume_first_event_timeout_after_dispatch_keeps_the_fence(harness):
    h = harness
    _reusable_consume(h)
    proof = h.backend.release_unused_attempt = _Proof()
    h.adapter.profile = replace(h.adapter.profile, admission_first_event_seconds=.02)
    h.tm.first_gate = asyncio.Event()       # never set: no first native event
    a = attachment()
    first = await consume_call(h, a)
    assert first.status_code == 503 and b"admission_timeout" in first.body
    assert len(h.tm.calls) == 1 and not proof.calls
    h.tm.first_gate = None
    _before = (len(h.backend.reserves), len(h.tm.calls))
    second = await consume_call(h, a)
    assert_consume_replay(h, second, *_before)


@pytest.mark.asyncio
async def test_consume_pre_dispatch_admission_timeout_releases_the_fence(harness):
    h = harness
    _reusable_consume(h)
    proof = h.backend.release_unused_attempt = _Proof()
    h.adapter.profile = replace(h.adapter.profile, admission_control_seconds=.02)
    h.backend.reserve_gate = asyncio.Event()  # reserve never answers in time
    a = attachment()
    first = await consume_call(h, a)
    assert first.status_code == 503 and b"admission_timeout" in first.body
    assert not h.tm.calls and len(proof.calls) == 1
    h.backend.reserve_gate = None
    second = await consume_call(h, a)
    assert second.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["occ3_cancelled", "occ3_unsupported_profile", "occ3_commit_failed"])
async def test_consume_other_refusals_keep_the_fence(harness, message):
    h = harness
    _reusable_consume(h)
    proof = h.backend.release_unused_attempt = _Proof()
    _aborting(h, [message])
    a = attachment()
    first = await consume_call(h, a)
    assert first.status_code == 409 and not proof.calls
    _before = (len(h.backend.reserves), len(h.tm.calls))
    second = await consume_call(h, a)
    assert_consume_replay(h, second, *_before)


@pytest.mark.asyncio
async def test_consume_resend_after_a_completed_turn_is_replay(harness):
    h = harness
    _reusable_consume(h)
    h.backend.release_unused_attempt = _Proof()
    a = attachment()
    assert (await consume_call(h, a)).status_code == 200
    _before = (len(h.backend.reserves), len(h.tm.calls))
    second = await consume_call(h, a)
    assert_consume_replay(h, second, *_before)


@pytest.mark.asyncio
async def test_concurrent_duplicate_while_in_flight_is_replay(harness):
    h = harness
    _reusable_consume(h)
    h.backend.release_unused_attempt = _Proof()
    h.tm.first_gate = gate = asyncio.Event()
    a = attachment()
    first = asyncio.create_task(consume_call(h, a))
    while not h.tm.calls:
        await asyncio.sleep(0)
    _before = (len(h.backend.reserves), len(h.tm.calls))
    duplicate = await consume_call(h, a)
    assert_consume_replay(h, duplicate, *_before)
    gate.set()
    assert (await first).status_code == 200


@pytest.mark.asyncio
async def test_a_context_removes_only_its_own_fence(harness):
    """A stale context never deletes a fence that another attempt now owns."""
    h = harness
    from types import SimpleNamespace as NS
    h.adapter._attempts["f"] = (time.time() + 60, "owner")
    h.adapter._unfence(NS(fence="f", attempt_id="stale"))
    assert "f" in h.adapter._attempts
    h.adapter._unfence(NS(fence="f", attempt_id="owner"))
    assert "f" not in h.adapter._attempts


# -------------------------------------------------------------- capture path
@pytest.mark.asyncio
async def test_capture_registry_busy_then_identical_resend_is_admitted(source_harness):
    h = source_harness
    original = h.backend.enroll_source
    state = {"busy": True}
    async def busy(*args):
        if state.pop("busy", False):
            raise AttachmentError("scheduler_busy", 503)
        return await original(*args)
    h.backend.enroll_source = busy
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    first, _, _ = await source_call(h)
    assert first.status_code == 503 and first.headers["retry-after"] == "5"
    assert not proof.calls and not h.tm.calls   # registry decline is the proof
    second, _, _ = await source_call(h)
    assert second.status_code == 200


@pytest.mark.asyncio
async def test_capture_scheduler_busy_batch_then_resend_is_admitted(source_harness):
    h = source_harness
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    _aborting(h, ["occ3_busy_batch"])
    first, _, _ = await source_call(h)
    assert first.status_code == 503 and b"scheduler_busy" in first.body
    assert len(proof.calls) == 1
    assert len(h.backend.retires) == 1          # unadmitted record still retired
    second, _, _ = await source_call(h)
    assert second.status_code == 200


@pytest.mark.asyncio
async def test_capture_post_dispatch_admission_timeout_keeps_the_fence(source_harness):
    h = source_harness
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    _aborting(h, ["occ3_admission_timeout"])
    first, _, _ = await source_call(h)
    assert first.status_code == 503 and not proof.calls
    second, _, _ = await source_call(h)
    assert second.status_code == 409 and b"replay_forbidden" in second.body


@pytest.mark.asyncio
async def test_capture_busy_after_output_keeps_the_fence(source_harness):
    """A busy code after content flowed is a mid-turn abort, never a release."""
    h = source_harness
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    async def generate(tm, obj, request=None):
        tm.calls.append(obj)
        yield {"text": "x", "meta_info": {"id": obj.rid}}
        yield {"text": "", "meta_info": {"id": obj.rid, "finish_reason": {
            "type": "abort", "message": "occ3_busy_batch", "status_code": 503}}}
    type(h.tm).generate_request = h.wrap(generate, h.adapter.generate_around, h.HookType.AROUND)
    first, _, _ = await source_call(h)
    assert first.status_code != 200 and not proof.calls
    second, _, _ = await source_call(h)
    assert second.status_code == 409 and b"replay_forbidden" in second.body


# ------------------------------------------------------------ registry proof
def _leased():
    r, _ = registry()
    ready(r)
    r.publish(publication())
    descriptor = consume()
    record = r.reserve_consume(descriptor)
    return r, record, descriptor


def test_release_unused_lease_restores_publication_and_nonce_once():
    r, record, descriptor = _leased()
    r.release_unused_lease(descriptor.attempt_id, descriptor.principal_digest)
    assert record.state == "published" and record.consumer is None
    assert descriptor.attempt_id in r.tombstone_attempts
    # The same attachment leases again under a fresh attempt.
    again = replace(descriptor, attempt_id="consume-2", reservation_id="res-2")
    assert r.reserve_consume(again).consumer == again
    # The declined attempt id can never be reused.
    with pytest.raises(LatentError, match="replay"):
        r.reserve_consume(replace(descriptor, reservation_id="res-3"))
    # After use, the nonce is single-use again.
    record.state, record.admitted_consume = "terminal", True
    with pytest.raises(LatentError):
        r.release_unused_lease("consume-2", descriptor.principal_digest)
    record.state, record.consumer = "published", None
    with pytest.raises(LatentError, match="replay"):
        r.reserve_consume(replace(again, attempt_id="consume-4", reservation_id="res-4"))


@pytest.mark.parametrize("mutate", [
    lambda rec: setattr(rec, "admitted_consume", True),
    lambda rec: setattr(rec, "state", "injected"),
    lambda rec: rec.injected_positions.add(0),
    lambda rec: setattr(rec, "inflight", 1),
    lambda rec: setattr(rec, "revoked", True),
])
def test_release_unused_lease_refuses_any_sign_of_use(mutate):
    r, record, descriptor = _leased()
    mutate(record)
    nonces = dict(r.used_nonces)
    with pytest.raises(LatentError):
        r.release_unused_lease(descriptor.attempt_id, descriptor.principal_digest)
    assert r.used_nonces == nonces and record.consumer == descriptor


def test_release_unused_lease_checks_the_principal():
    r, record, descriptor = _leased()
    with pytest.raises(LatentError, match="identity"):
        r.release_unused_lease(descriptor.attempt_id, "b" * 64)
    assert record.state == "leased"
