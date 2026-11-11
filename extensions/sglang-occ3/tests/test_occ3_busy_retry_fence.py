"""Retry design (a): a pre-dispatch busy refusal frees its own replay fence.

A source capture additionally swallows a proven pre-admission busy decline and
re-dispatches the same native request inside a bounded server-side busy-wait
window (BufferProfile.capture_busy_wait_seconds; 0 disables it), instead of
surfacing the refusal for the client to resend.

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


# --------------------------- bounded capture busy re-dispatch (server-side)
@pytest.mark.asyncio
async def test_capture_busy_decline_then_admission_after_one_redispatch(source_harness):
    """A first-and-only busy decline of a source capture is swallowed and the
    SAME native request re-dispatched: same rid, one enrollment, one prepare,
    one reservation, one finish - never a re-enroll, retire or second fence."""
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    _aborting(h, ["occ3_busy_batch"])
    from occ3_sglang import metrics
    before = metrics.snapshot()["counters"].get("capture_busy_retry", 0)
    response, _, _ = await source_call(h)
    after = metrics.snapshot()["counters"].get("capture_busy_retry", 0)
    assert response.status_code == 200
    assert len(h.tm.calls) == 2
    assert h.tm.calls[0] is h.tm.calls[1]
    assert h.tm.calls[0].rid == h.backend.enrollments[0][3]
    assert len(h.chat.converted) == len(h.backend.source_binds) == 1
    enrollment, reservation = h.backend.source_binds[0]
    assert h.backend.source_finishes[0][:2] == (enrollment, reservation)
    assert h.tm.calls[0].occ3_worker_descriptor.attempt_id == enrollment.attempt_id
    assert len(h.backend.enrollments) == 1 and len(h.backend.prepares) == 1
    assert len(h.backend.source_finishes) == 1
    assert not h.backend.retires
    assert h.adapter._active == 0
    assert after == before + 1


@pytest.mark.asyncio
async def test_capture_two_busy_declines_then_admission(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    _aborting(h, ["occ3_busy_batch", "occ3_busy_batch"])
    response, _, _ = await source_call(h)
    assert response.status_code == 200
    assert len(h.tm.calls) == 3
    assert h.adapter._active == 0


@pytest.mark.asyncio
async def test_capture_busy_after_output_with_wait_enabled_keeps_the_fence(source_harness):
    """Mid-turn busy (content already flowed) never re-dispatches, even with
    the busy-wait window enabled."""
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    async def generate(tm, obj, request=None):
        tm.calls.append(obj)
        yield {"text": "x", "meta_info": {"id": obj.rid}}
        yield {"text": "", "meta_info": {"id": obj.rid, "finish_reason": {
            "type": "abort", "message": "occ3_busy_batch", "status_code": 503}}}
    type(h.tm).generate_request = h.wrap(generate, h.adapter.generate_around, h.HookType.AROUND)
    first, _, _ = await source_call(h)
    assert first.status_code != 200 and len(h.tm.calls) == 1 and not proof.calls
    second, _, _ = await source_call(h)
    assert second.status_code == 409 and b"replay_forbidden" in second.body


@pytest.mark.asyncio
async def test_capture_busy_retry_stops_when_the_deadline_no_longer_fits(source_harness):
    """The reserved generation must still fit the turn deadline after one
    cadence; otherwise the decline surfaces immediately as before."""
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10,
                                deadline_margin_seconds=59)
    _aborting(h, ["occ3_busy_batch"])
    from occ3_sglang import metrics
    before = metrics.snapshot()["counters"].get("capture_busy_retry", 0)
    first, _, _ = await source_call(h)
    assert first.status_code == 503 and b"scheduler_busy" in first.body
    assert len(h.tm.calls) == 1
    assert metrics.snapshot()["counters"].get("capture_busy_retry", 0) == before


@pytest.mark.asyncio
async def test_capture_admission_timeout_with_wait_enabled_never_retries(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    _aborting(h, ["occ3_admission_timeout"])
    first, _, _ = await source_call(h)
    assert first.status_code == 503 and b"admission_timeout" in first.body
    assert len(h.tm.calls) == 1
    second, _, _ = await source_call(h)
    assert second.status_code == 409 and b"replay_forbidden" in second.body


@pytest.mark.asyncio
async def test_capture_busy_retry_blocked_by_request_snapshot_mutation(source_harness):
    """A dispatch that mutated the request never re-sends it."""
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    async def generate(tm, obj, request=None):
        tm.calls.append(obj)
        obj.sampling_params["max_new_tokens"] = 2
        yield {"text": "", "meta_info": {"id": obj.rid, "finish_reason": {
            "type": "abort", "message": "occ3_busy_batch", "status_code": 503}}}
    type(h.tm).generate_request = h.wrap(generate, h.adapter.generate_around, h.HookType.AROUND)
    first, _, _ = await source_call(h)
    assert first.status_code == 503 and b"scheduler_busy" in first.body
    assert len(h.tm.calls) == 1


@pytest.mark.asyncio
async def test_capture_admission_capacity_decline_also_redispatches(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    _aborting(h, ["occ3_busy"])
    response, _, _ = await source_call(h)
    assert response.status_code == 200
    assert len(h.tm.calls) == 2
    assert h.adapter._active == 0



# ------------------------------------------- bounded wait and race boundaries
@pytest.mark.parametrize("field,value", [
    ("capture_busy_retry_interval", 0), ("capture_busy_retry_interval", -1),
    ("capture_busy_retry_interval", float("inf")), ("capture_busy_retry_interval", float("nan")),
    ("capture_busy_wait_seconds", -1), ("capture_busy_wait_seconds", float("inf")),
    ("capture_busy_wait_seconds", float("nan")),
])
def test_capture_busy_profile_rejects_unbounded_or_zero_cadence(field, value):
    from occ3_sglang.chat_adapter import BufferProfile
    with pytest.raises(ValueError, match="invalid buffer profile"):
        BufferProfile(**{field: value})


def test_capture_busy_profile_zero_wait_disables_and_headers_are_source_only():
    from occ3_sglang.chat_adapter import BufferProfile, ChatAttachmentAdapter
    assert BufferProfile(capture_busy_wait_seconds=0).capture_busy_wait_seconds == 0
    adapter = ChatAttachmentAdapter()
    error = AttachmentError("scheduler_busy", 503)
    assert adapter._error(error, source=True).headers["x-optimus-occ3-admission-window"] == "25.0"
    assert adapter._error(error).headers["x-optimus-occ3-admission-window"] == "15.0"


def _busy_output(obj):
    return {"text": "", "output_ids": [], "meta_info": {"id": obj.rid, "finish_reason": {
        "type": "abort", "message": "occ3_busy_batch", "status_code": 503}}}


def _pause_retry(monkeypatch):
    """Gate only adapter sleeps; guard and cleanup await the real event loop."""
    from types import SimpleNamespace
    from occ3_sglang import chat_adapter
    entered, resume = asyncio.Event(), asyncio.Event()

    async def pause(delay):
        entered.set()
        await resume.wait()

    monkeypatch.setattr(chat_adapter, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": pause}))
    return entered, resume


@pytest.mark.asyncio
async def test_capture_busy_window_exhausts_and_unfences_only_after_retire(source_harness):
    from occ3_sglang import metrics
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=.12,
                                capture_busy_retry_interval=.01)
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    _aborting(h, ["occ3_busy_batch"] * 1000)
    retiring, release = asyncio.Event(), asyncio.Event()
    retire = h.backend.retire

    async def tracked(*args):
        assert h.adapter._attempts
        retiring.set()
        await release.wait()
        return await retire(*args)

    h.backend.retire = tracked
    before = metrics.snapshot()["counters"].get("capture_busy_retry", 0)
    started = time.monotonic()
    task = asyncio.create_task(source_call(h))
    await asyncio.wait_for(retiring.wait(), 2)
    assert h.adapter._attempts and not task.done()
    release.set()
    response, _, _ = await asyncio.wait_for(task, 2)
    assert response.status_code == 503 and b"scheduler_busy" in response.body
    assert time.monotonic() - started >= .08
    assert 3 <= len(h.tm.calls) < 20
    assert metrics.snapshot()["counters"].get("capture_busy_retry", 0) - before == len(h.tm.calls) - 1
    assert len(h.backend.enrollments) == len(h.backend.prepares) == len(proof.calls) == 1
    assert len(h.backend.retires) == 1 and not h.backend.source_finishes
    assert not h.adapter._attempts and h.adapter._active == 0 and not h.chat.frames
    _aborting(h, [])
    assert (await source_call(h))[0].status_code == 200


@pytest.mark.asyncio
async def test_capture_delayed_first_decline_does_not_restart_the_busy_window(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=8,
                                capture_busy_retry_interval=1)
    offset = [0.0]
    h.adapter.monotonic = lambda: time.monotonic() + offset[0]
    bounds = []

    async def generate(tm, obj, request=None):
        tm.calls.append(obj)
        ctx = request.state._optimus_occ3_chat_context
        bounds.append((ctx.capture_busy_deadline, ctx.first_event_deadline))
        offset[0] = 7.5  # Inside the 8 s first-event bound, too late for a 1 s retry.
        yield _busy_output(obj)

    type(h.tm).generate_request = h.wrap(generate, h.adapter.generate_around, h.HookType.AROUND)
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and b"scheduler_busy" in response.body
    assert len(h.tm.calls) == 1
    assert bounds[0][0] == bounds[0][1]  # Both bounds use the INITIAL dispatch.
    assert response.headers["x-optimus-occ3-admission-window"] == "23.0"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["busy_window", "turn_deadline", "descriptor_deadline", "ttl"])
async def test_capture_busy_rechecks_deadlines_after_sleep(source_harness, monkeypatch, boundary):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    if boundary == "descriptor_deadline":
        h.backend.mutate_source_descriptor = lambda d: replace(d, deadline=time.monotonic() + 3)
    _aborting(h, ["occ3_busy_batch"])
    entered, resume = _pause_retry(monkeypatch)
    task = asyncio.create_task(source_call(h))
    await asyncio.wait_for(entered.wait(), 2)
    if boundary == "ttl":
        h.adapter.epoch = lambda: time.time() + 400
    else:
        offset = {"busy_window": 11, "turn_deadline": 59.5, "descriptor_deadline": 3}[boundary]
        h.adapter.monotonic = lambda: time.monotonic() + offset
    resume.set()
    response, _, _ = await asyncio.wait_for(task, 2)
    # A guard can observe expiry first; either outcome must forbid re-dispatch.
    assert response.status_code in (503, 504)
    assert len(h.tm.calls) == 1 and not h.backend.source_finishes
    assert h.adapter._active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["sampling", "prompt", "rid", "descriptor"])
async def test_capture_busy_rechecks_identity_after_sleep(source_harness, monkeypatch, mutation):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    _aborting(h, ["occ3_busy_batch"])
    entered, resume = _pause_retry(monkeypatch)
    task = asyncio.create_task(source_call(h))
    await asyncio.wait_for(entered.wait(), 2)
    obj = h.tm.calls[0]
    if mutation == "sampling":
        obj.sampling_params["max_new_tokens"] = 2
    elif mutation == "prompt":
        obj.input_ids.append(99)
    elif mutation == "rid":
        obj.rid = "other-native-request"
    else:
        obj.occ3_worker_descriptor = replace(obj.occ3_worker_descriptor)
    resume.set()
    response, _, _ = await asyncio.wait_for(task, 2)
    assert response.status_code == 503 and b"scheduler_busy" in response.body
    assert len(h.tm.calls) == 1 and not h.backend.source_finishes
    assert h.adapter._active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_capture_busy_wait_cancellation_or_disconnect_never_redispatches(source_harness, monkeypatch, cancel):
    from test_chat_source_capture import source_raw
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    _aborting(h, ["occ3_busy_batch"])
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    entered, _ = _pause_retry(monkeypatch)
    raw, queue = source_raw()
    task = asyncio.create_task(h.chat.handle_request(chat_request(), raw))
    await asyncio.wait_for(entered.wait(), 2)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    else:
        queue.put_nowait({"type": "http.disconnect"})
        response = await asyncio.wait_for(task, 2)
        assert response.status_code == 499
    assert len(h.tm.calls) == len(h.backend.retires) == 1
    assert not proof.calls and not h.backend.source_finishes
    assert h.adapter._attempts and h.adapter._active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["output", "busy", "exception", "stall"])
async def test_capture_busy_decline_requires_clean_stream_end(source_harness, tail):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10,
                                capture_busy_retry_interval=.01, admission_first_event_seconds=.05)
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")

    async def generate(tm, obj, request=None):
        tm.calls.append(obj)
        yield _busy_output(obj)
        if tail == "output":
            yield {"text": "unexpected", "meta_info": {"id": obj.rid}}
        elif tail == "busy":
            yield _busy_output(obj)
        elif tail == "exception":
            raise RuntimeError("fixture stream failure")
        else:
            await asyncio.Event().wait()

    type(h.tm).generate_request = h.wrap(generate, h.adapter.generate_around, h.HookType.AROUND)
    response, _, _ = await source_call(h)
    assert response.status_code == (503 if tail == "stall" else 409)
    assert len(h.tm.calls) == 1 and not proof.calls and not h.chat.frames
    assert h.adapter._attempts and h.adapter._active == 0
    assert (await source_call(h))[0].status_code == 409


@pytest.mark.asyncio
async def test_capture_admission_timeout_after_busy_retry_keeps_the_fence(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10,
                                capture_busy_retry_interval=.01)
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    _aborting(h, ["occ3_busy_batch", "occ3_admission_timeout"])
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and b"admission_timeout" in response.body
    assert len(h.tm.calls) == 2 and not proof.calls
    assert (await source_call(h))[0].status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("exhaust", [False, True])
async def test_capture_busy_keepalive_stream_success_or_bounded_exhaustion(source_harness, exhaust):
    from occ3_sglang.attachment_contract import KEEPALIVE_REQUEST_HEADER, KEEPALIVE_REQUEST_MODE
    from test_chat_source_capture import source_raw
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=.12,
                                capture_busy_retry_interval=.01, keepalive_seconds=.002)
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    _aborting(h, ["occ3_busy_batch"] * (1000 if exhaust else 1))
    raw, _ = source_raw()
    raw.scope["headers"].append((KEEPALIVE_REQUEST_HEADER.encode(), KEEPALIVE_REQUEST_MODE.encode()))
    response = await h.chat.handle_request(chat_request(), raw)
    events = []

    async def send(event):
        events.append(event)

    await asyncio.wait_for(response(raw.scope, raw.receive, send), 2)
    body = b"".join(e.get("body", b"") for e in events)
    assert response.status_code == 200 and b": occ3-keepalive" in body
    assert len(h.tm.calls) >= 2 and h.adapter._active == 0
    if exhaust:
        frames = [line[6:] for line in body.splitlines() if line.startswith(b"data: ")]
        assert len(frames) == 1 and json.loads(frames[0])["error"]["message"] == "scheduler_busy"
        assert b"[DONE]" not in body and not h.backend.source_finishes
        assert h.adapter._attempts and not proof.calls  # Handed-off streams keep the fence.
    else:
        assert b"[DONE]" in body and b'"error"' not in body
        assert len(h.backend.source_finishes) == 1 and not h.backend.retires


@pytest.mark.asyncio
async def test_receiver_does_not_use_the_capture_busy_window(harness):
    h = harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10,
                                capture_busy_retry_interval=.01)
    _aborting(h, ["occ3_busy_batch"])
    response = await consume_call(h)
    assert response.status_code == 503 and b"scheduler_busy" in response.body
    assert response.headers["x-optimus-occ3-admission-window"] == "15.0"
    assert len(h.tm.calls) == 1


# ------------------------------------------- registry active-slot busy wait
def _slot_busy(h, declines, quota=False):
    """enroll_source declines with the registry's typed active-slot busy (or
    quota) for the first ``declines`` calls, then enrolls normally."""
    from occ3_sglang.attachment_contract import ActiveSlotBusy
    original = h.backend.enroll_source
    at = []

    async def enroll(*args):
        if len(at) < declines:
            h.backend.enrollments.append(args)
            at.append(time.monotonic())
            raise AttachmentError("scheduler_busy", 503) if quota else ActiveSlotBusy()
        return await original(*args)

    h.backend.enroll_source = enroll
    return at


@pytest.mark.asyncio
async def test_capture_registry_active_slot_busy_waits_then_admits(source_harness):
    """Re-enroll inside the window with the same attempt and native rid. The
    waits run past the 7 s-style control bound (scaled down here): the
    control deadline moves by exactly the time spent waiting."""
    from occ3_sglang import metrics
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=1,
                                capture_busy_retry_interval=.03, admission_control_seconds=.05)
    declines = _slot_busy(h, 3)
    before = metrics.snapshot()["counters"].get("capture_registry_busy_retry", 0)
    response, _, _ = await source_call(h)
    assert response.status_code == 200, response.body
    assert len(declines) == 3 and len(h.backend.enrollments) == 4
    assert len({e[2] for e in h.backend.enrollments}) == 1     # one attempt id
    assert len({e[3] for e in h.backend.enrollments}) == 1     # one native rid
    assert len(h.tm.calls) == 1 and len(h.backend.source_finishes) == 1
    assert not h.backend.retires and h.adapter._active == 0
    assert metrics.snapshot()["counters"].get("capture_registry_busy_retry", 0) - before == 3
    # The fence stays for the admitted capture: an identical resend is a replay.
    assert (await source_call(h))[0].status_code == 409


@pytest.mark.asyncio
async def test_capture_registry_quota_never_waits(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    _slot_busy(h, 1, quota=True)
    started = time.monotonic()
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and b"scheduler_busy" in response.body
    assert time.monotonic() - started < .5
    assert len(h.backend.enrollments) == 1 and not h.tm.calls
    # Historical proof path: the registry decline unfences, a resend admits.
    assert (await source_call(h))[0].status_code == 200


@pytest.mark.asyncio
async def test_capture_registry_busy_zero_window_refuses_like_today(source_harness):
    h = source_harness
    _slot_busy(h, 1)
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and response.headers["retry-after"] == "5"
    assert response.headers["x-optimus-occ3-admission-window"] == "15.0"
    assert len(h.backend.enrollments) == 1 and not h.tm.calls
    assert (await source_call(h))[0].status_code == 200


@pytest.mark.asyncio
async def test_capture_registry_busy_window_exhausts_then_refuses_like_today(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=.12,
                                capture_busy_retry_interval=.01)
    declines = _slot_busy(h, 1000)
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and b"scheduler_busy" in response.body
    assert float(response.headers["x-optimus-occ3-admission-window"]) == pytest.approx(15.12)
    assert 3 <= len(declines) < 20
    assert declines[-1] - declines[0] <= .12 + .05
    assert not h.tm.calls and h.adapter._active == 0
    assert not h.adapter._attempts            # registry decline is the proof
    declines.extend([0.0] * 1000)             # the slot is free now
    assert (await source_call(h))[0].status_code == 200


@pytest.mark.asyncio
async def test_capture_registry_and_scheduler_waits_share_one_window(source_harness):
    """A registry wait anchors the window; the scheduler re-dispatch draws on
    the remainder instead of a fresh window from its own dispatch."""
    h = source_harness
    window = .2
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=window,
                                capture_busy_retry_interval=.01)
    declines = _slot_busy(h, 8)
    dispatches = []

    async def generate(tm, obj, request=None):
        tm.calls.append(obj)
        ctx = request.state._optimus_occ3_chat_context
        dispatches.append((time.monotonic(), ctx.capture_busy_deadline))
        yield _busy_output(obj)

    type(h.tm).generate_request = h.wrap(generate, h.adapter.generate_around, h.HookType.AROUND)
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and b"scheduler_busy" in response.body
    assert len(declines) == 8 and len(dispatches) >= 1
    anchor = dispatches[0][1]
    assert all(deadline == anchor for _, deadline in dispatches)
    assert declines[0] + window <= anchor <= declines[0] + window + .01   # anchored at the first decline
    assert all(at <= anchor for at, _ in dispatches)
    assert dispatches[-1][0] - declines[0] <= window + .05


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_capture_registry_busy_wait_cancellation_stops_reenroll(source_harness, monkeypatch, cancel):
    from test_chat_source_capture import source_raw
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    _slot_busy(h, 1)
    entered, _ = _pause_retry(monkeypatch)
    raw, queue = source_raw()
    task = asyncio.create_task(h.chat.handle_request(chat_request(), raw))
    await asyncio.wait_for(entered.wait(), 2)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    else:
        queue.put_nowait({"type": "http.disconnect"})
        response = await asyncio.wait_for(task, 2)
        assert response.status_code == 499
    assert len(h.backend.enrollments) == 1 and not h.tm.calls
    assert h.adapter._active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["turn_deadline", "ttl"])
async def test_capture_registry_busy_rechecks_bounds_after_sleep(source_harness, monkeypatch, boundary):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10)
    _slot_busy(h, 1)
    entered, resume = _pause_retry(monkeypatch)
    task = asyncio.create_task(source_call(h))
    await asyncio.wait_for(entered.wait(), 2)
    if boundary == "ttl":
        h.adapter.epoch = lambda: time.time() + 400
    else:
        h.adapter.monotonic = lambda: time.monotonic() + 59.5
    resume.set()
    response, _, _ = await asyncio.wait_for(task, 2)
    assert response.status_code in (503, 504)
    assert len(h.backend.enrollments) == 1 and not h.tm.calls
    assert h.adapter._active == 0


@pytest.mark.asyncio
async def test_capture_registry_busy_keepalive_wait_never_emits_refusal(source_harness):
    """The registry wait runs before the keepalive hand-off: a successful
    wait streams a normal 200 with no SSE refusal event."""
    from occ3_sglang.attachment_contract import KEEPALIVE_REQUEST_HEADER, KEEPALIVE_REQUEST_MODE
    from test_chat_source_capture import source_raw
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=1,
                                capture_busy_retry_interval=.01, keepalive_seconds=.002)
    _slot_busy(h, 2)
    raw, _ = source_raw()
    raw.scope["headers"].append((KEEPALIVE_REQUEST_HEADER.encode(), KEEPALIVE_REQUEST_MODE.encode()))
    response = await h.chat.handle_request(chat_request(), raw)
    events = []

    async def send(event):
        events.append(event)

    await asyncio.wait_for(response(raw.scope, raw.receive, send), 2)
    body = b"".join(e.get("body", b"") for e in events)
    assert response.status_code == 200
    assert b"[DONE]" in body and b'"error"' not in body
    assert len(h.backend.enrollments) == 3 and len(h.backend.source_finishes) == 1


def _skewed_sleep(h, monkeypatch, overshoot):
    """Adapter clock = real clock + skew; each adapter sleep advances the
    skew by the requested delay plus ``overshoot`` (a late timer wake-up)."""
    from types import SimpleNamespace
    from occ3_sglang import chat_adapter
    skew = [0.0]
    h.adapter.monotonic = lambda: time.monotonic() + skew[0]
    real_sleep = asyncio.sleep

    async def sleep(delay):
        skew[0] += delay + overshoot
        await real_sleep(0)

    monkeypatch.setattr(chat_adapter, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": sleep}))
    return skew


def _record_waits(h):
    """(control deadline before, after, adapter time spent) per registry wait."""
    waits = []
    original = h.adapter._registry_busy_wait

    async def wait(ctx):
        before, started = ctx.admission_control_deadline, h.adapter.monotonic()
        try:
            return await original(ctx)
        finally:
            waits.append((before, ctx.admission_control_deadline, h.adapter.monotonic() - started,
                          ctx.capture_busy_deadline))

    h.adapter._registry_busy_wait = wait
    return waits


@pytest.mark.asyncio
async def test_capture_registry_wait_charges_control_extension_once_within_window(
        source_harness, monkeypatch):
    """Late timer wake-ups (overshoot) are charged as waited time, once per
    wait; the sum never exceeds the window."""
    h = source_harness
    window = 1.0
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=window,
                                capture_busy_retry_interval=.05, admission_control_seconds=.5)
    _skewed_sleep(h, monkeypatch, overshoot=.02)
    declines = _slot_busy(h, 5)
    waits = _record_waits(h)
    response, _, _ = await source_call(h)
    assert response.status_code == 200, response.body
    assert len(declines) == 5 and len(waits) == 5
    for (before, after, spent, _), (next_before, *_rest) in zip(waits, waits[1:]):
        assert after == next_before                      # charged once, no reset
    extensions = [after - before for before, after, _, _ in waits]
    assert all(.07 <= e <= spent for e, (_, _, spent, _) in zip(extensions, waits))
    assert sum(extensions) <= window


@pytest.mark.asyncio
async def test_capture_registry_wait_overshooting_the_window_refuses_without_extension(
        source_harness, monkeypatch):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=.1,
                                capture_busy_retry_interval=.05)
    _skewed_sleep(h, monkeypatch, overshoot=1.0)
    declines = _slot_busy(h, 1)
    waits = _record_waits(h)
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and b"scheduler_busy" in response.body
    assert len(declines) == 1 and len(waits) == 1 and not h.tm.calls
    before, after, _, _ = waits[0]
    assert after == before
    assert (await source_call(h))[0].status_code == 200


@pytest.mark.asyncio
async def test_capture_registry_wait_never_refills_the_control_budget(source_harness):
    """A busy decline that arrives late in the control budget: the wait only
    defers the deadline by the time waited, so the next slow enroll still
    times out on the original control remainder."""
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=1,
                                capture_busy_retry_interval=.01, admission_control_seconds=.2)
    original = h.backend.enroll_source
    from occ3_sglang.attachment_contract import ActiveSlotBusy
    calls = []

    async def enroll(*args):
        calls.append(time.monotonic())
        await asyncio.sleep(.15)
        if len(calls) == 1:
            raise ActiveSlotBusy()
        return await original(*args)

    h.backend.enroll_source = enroll
    started = time.monotonic()
    response, _, _ = await source_call(h)
    elapsed = time.monotonic() - started
    assert response.status_code == 503 and b"admission_timeout" in response.body
    assert len(calls) == 2 and not h.tm.calls
    assert elapsed < .2 + .01 + .15


@pytest.mark.asyncio
async def test_registry_wait_publishes_control_extension_before_sleep_and_fences_duplicates(
        source_harness, monkeypatch):
    from test_chat_source_capture import source_raw
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=1,
                                capture_busy_retry_interval=.08, admission_control_seconds=.03)
    _slot_busy(h, 1)
    waits = _record_waits(h)
    before = []
    original = h.adapter._registry_busy_wait

    async def tracked(ctx):
        before.append(ctx.admission_control_deadline)
        return await original(ctx)

    h.adapter._registry_busy_wait = tracked
    entered, resume = _pause_retry(monkeypatch)
    raw, _ = source_raw()
    task = asyncio.create_task(h.chat.handle_request(chat_request(), raw))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        ctx = raw.state._optimus_occ3_chat_context
        assert ctx.admission_control_deadline == pytest.approx(before[0] + .08, abs=1e-6)
        duplicate, _, _ = await source_call(h)
        assert duplicate.status_code == 409 and b"replay_forbidden" in duplicate.body
        await asyncio.sleep(.05)  # Cross the original control deadline while the guard runs.
        assert not task.done()
        resume.set()
        response = await asyncio.wait_for(task, 1)
        assert response.status_code == 200, response.body
        assert len(h.backend.enrollments) == 2 and len(h.tm.calls) == 1
        assert len(waits) == 1 and waits[0][1] - waits[0][0] >= .05
    finally:
        resume.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_registry_stalled_wait_is_bounded_by_guard(source_harness, monkeypatch):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=.1,
                                capture_busy_retry_interval=.03, admission_control_seconds=.02)
    _slot_busy(h, 1)
    entered, _ = _pause_retry(monkeypatch)
    task = asyncio.create_task(source_call(h))
    await asyncio.wait_for(entered.wait(), 1)
    response, _, _ = await asyncio.wait_for(task, .5)
    assert response.status_code == 503 and b"admission_timeout" in response.body
    assert len(h.backend.enrollments) == 1 and not h.tm.calls
    assert h.adapter._active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_registry_late_rpc_reply_cannot_extend_or_bypass_control_deadline(source_harness, busy):
    from occ3_sglang.attachment_contract import ActiveSlotBusy
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=1,
                                capture_busy_retry_interval=.01, admission_control_seconds=.2)
    skew = [0.0]
    h.adapter.monotonic = lambda: time.monotonic() + skew[0]
    original = h.backend.enroll_source
    calls = []

    async def late(*args):
        calls.append(args)
        enrollment = await original(*args) if not busy else None
        skew[0] = .25
        if busy:
            raise ActiveSlotBusy()
        return enrollment

    h.backend.enroll_source = late
    from occ3_sglang import metrics
    before = metrics.snapshot()["counters"].get("capture_registry_busy_retry", 0)
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and b"admission_timeout" in response.body
    assert len(calls) == 1 and not h.tm.calls and not h.backend.source_finishes
    assert metrics.snapshot()["counters"].get("capture_registry_busy_retry", 0) == before


# ------------------------------- pinned SGLang dispatch preamble (R1 regression)
@pytest.fixture
def pinned_sources():
    """Real patched SGLang sources (pinned e00930c5 + scheduler-cache-ipc
    manifest), via the same applier as test_latent_ipc_patch. Skips without
    the attested snapshot (OCC3_ATTESTED_SRT)."""
    from test_latent_ipc_patch import AppliedSource
    sources = AppliedSource()
    try:
        yield sources
    finally:
        sources.close()


def _pinned_preamble(sources):
    """The pinned generate_request preamble, proven from the source, and the
    real _set_default_priority body."""
    import ast
    node = sources.node("managers/tokenizer_manager.py", "generate_request", "TokenizerManager")
    assert [ast.unparse(s) for s in node.body[:3]] == [
        "self.auto_create_handle_loop()", "obj.normalize_batch_and_arguments()",
        "self._set_default_priority(obj)"]
    return sources.function("managers/tokenizer_manager.py", "_set_default_priority", "TokenizerManager")


def _pinned_native(h, monkeypatch, sources, priority, mutate=None):
    cls = sources.generate_class()
    monkeypatch.setitem(h.convert.__globals__, "GenerateReqInput", cls)
    monkeypatch.setattr(type(h.tm), "_set_default_priority", _pinned_preamble(sources), raising=False)
    h.tm.enable_priority_scheduling = priority
    h.tm.default_priority_value = 3 if priority else None
    states = []

    async def generate(tm, obj, request=None):
        tm.calls.append(obj)
        obj.normalize_batch_and_arguments()   # pinned preamble, same order
        tm._set_default_priority(obj)
        states.append({k: v for k, v in vars(obj).items() if k != "occ3_worker_descriptor"})
        if len(tm.calls) == 1:
            if mutate:
                mutate(obj)
            yield _busy_output(obj)
            return
        yield {"text": "fixture", "meta_info": {"id": obj.rid, "finish_reason": {"type": "stop"}}}

    type(h.tm).generate_request = h.wrap(generate, h.adapter.generate_around, h.HookType.AROUND)
    return cls, states


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [False, True])
async def test_capture_busy_redispatch_survives_pinned_normalization(
        source_harness, monkeypatch, pinned_sources, priority):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10,
                                capture_busy_retry_interval=.01)
    cls, states = _pinned_native(h, monkeypatch, pinned_sources, priority)
    response, _, _ = await source_call(h)
    assert response.status_code == 200, response.body
    assert len(h.tm.calls) == 2 and h.tm.calls[0] is h.tm.calls[1]
    obj = h.tm.calls[0]
    assert type(obj) is cls
    assert (obj.is_single, obj.batch_size, obj.parallel_sample_num) == (True, 1, 1)
    assert obj.priority == (3 if priority else None)
    assert obj.occ3_worker_descriptor.attempt_id == h.backend.enrollments[0][2]
    assert states[0] == states[1]              # the preamble is idempotent
    assert len(h.backend.enrollments) == len(h.backend.source_finishes) == 1
    assert not h.backend.retires and h.adapter._active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["budget", "priority", "extra"])
async def test_capture_busy_redispatch_refuses_mutation_beyond_pinned_normalization(
        source_harness, monkeypatch, pinned_sources, change):
    def mutate(obj):
        if change == "budget":
            obj.sampling_params["max_new_tokens"] = 2
        elif change == "priority":
            obj.priority = 7
        else:
            obj.unexpected_state = True
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, capture_busy_wait_seconds=10,
                                capture_busy_retry_interval=.01)
    _pinned_native(h, monkeypatch, pinned_sources, False, mutate)
    response, _, _ = await source_call(h)
    assert response.status_code == 503 and b"scheduler_busy" in response.body
    assert len(h.tm.calls) == 1 and not h.backend.source_finishes


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


# ------------------------------------------ Fable review: backend and worker
class _StubClient:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    async def call(self, operation, payload, command_id=None):
        self.calls.append((operation, payload, command_id))
        reply = self.replies[operation]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _backend_with(descriptor, replies):
    from occ3_sglang import latent_backend as lb
    backend = object.__new__(lb.LatentBackend)
    backend.client = _StubClient(replies)
    backend.attempts = {descriptor.attempt_id: {"descriptor": descriptor, "principal": "a" * 64}}
    return backend


@pytest.mark.asyncio
async def test_backend_release_unused_attempt_consume_branch():
    descriptor = consume()
    backend = _backend_with(descriptor, {"release_lease": {"state": "published"}})
    assert await backend.release_unused_attempt(descriptor.attempt_id, descriptor.native_rid) == "released"
    assert backend.client.calls == [("release_lease",
                                     {"attempt_id": descriptor.attempt_id, "principal_digest": "a" * 64},
                                     "release-" + descriptor.reservation_id)]
    assert backend.attempts[descriptor.attempt_id]["retired"]
    # Once released, the frontend never releases or proves the attempt again.
    with pytest.raises(LatentError, match="state"):
        await backend.release_unused_attempt(descriptor.attempt_id, descriptor.native_rid)


@pytest.mark.asyncio
async def test_backend_release_unused_attempt_consume_refusal_propagates():
    descriptor = consume()
    backend = _backend_with(descriptor, {"release_lease": LatentError("state")})
    with pytest.raises(LatentError, match="state"):
        await backend.release_unused_attempt(descriptor.attempt_id, descriptor.native_rid)
    assert not backend.attempts[descriptor.attempt_id].get("retired")


@pytest.mark.asyncio
@pytest.mark.parametrize("record,ok", [({"state": "reserved", "revoked": False}, True),
                                       ({"state": "capturing", "revoked": False}, False),
                                       ({"state": "reserved", "revoked": True}, False)])
async def test_backend_release_unused_attempt_capture_branch(record, ok):
    from test_latent_types import capture
    descriptor = capture()
    backend = _backend_with(descriptor, {"status": record})
    if ok:
        assert await backend.release_unused_attempt(descriptor.attempt_id, descriptor.native_rid) == "unadmitted"
    else:
        with pytest.raises(LatentError, match="state"):
            await backend.release_unused_attempt(descriptor.attempt_id, descriptor.native_rid)
    assert [c[0] for c in backend.client.calls] == ["status"]
    assert not backend.attempts[descriptor.attempt_id].get("retired")   # caller retires


@pytest.mark.asyncio
async def test_backend_release_unused_attempt_checks_the_native_rid():
    descriptor = consume()
    backend = _backend_with(descriptor, {"release_lease": {}})
    with pytest.raises(LatentError, match="identity"):
        await backend.release_unused_attempt(descriptor.attempt_id, "other-rid")
    assert not backend.client.calls


def _worker_runtime(r):
    from occ3_sglang import latent_worker as w
    from test_latent_types import profile
    runtime = object.__new__(w.WorkerRuntime)
    runtime.profile = profile(); runtime.registry = r; runtime.commands = {}
    runtime.poisoned = False; runtime.streams = {}; runtime.calibration_capture = None
    runtime._agreed_epoch = 100.0
    runtime.gather = lambda value: [value] * 8
    runtime.consensus = lambda value: value
    runtime.fence = lambda: None
    return runtime


def test_worker_control_release_lease_returns_the_lease_on_every_rank():
    from occ3_sglang.latent_types import control_request
    r, record, descriptor = _leased()
    runtime = _worker_runtime(r)
    reply = runtime.control(control_request("release-1", "release_lease", {
        "attempt_id": descriptor.attempt_id, "principal_digest": descriptor.principal_digest}))
    assert reply["ok"], reply
    assert record.state == "published" and record.consumer is None
    # A replayed command id returns the journaled reply, never a second release.
    again = runtime.control(control_request("release-1", "release_lease", {
        "attempt_id": descriptor.attempt_id, "principal_digest": descriptor.principal_digest}))
    assert again == reply


@pytest.mark.parametrize("payload", [{"attempt_id": "consume"},
                                     {"attempt_id": "consume", "principal_digest": "a" * 64, "x": 1}])
def test_worker_control_release_lease_requires_the_exact_payload(payload):
    from occ3_sglang.latent_types import control_request
    r, record, _ = _leased()
    reply = _worker_runtime(r).control(control_request("release-2", "release_lease", payload))
    assert not reply["ok"] and record.state == "leased"


def test_worker_control_release_lease_refuses_an_admitted_lease():
    from occ3_sglang.latent_types import control_request
    r, record, descriptor = _leased()
    record.admitted_consume = True
    reply = _worker_runtime(r).control(control_request("release-3", "release_lease", {
        "attempt_id": descriptor.attempt_id, "principal_digest": descriptor.principal_digest}))
    assert not reply["ok"] and record.state == "leased" and record.consumer == descriptor


# --------------------------------------- Fable review 4: consume busy decline
def test_consume_behind_an_active_capture_declines_busy_without_mutation():
    r, _ = registry()
    ready(r)
    r.publish(publication())
    from test_latent_types import capture
    r.reserve_capture(capture(attempt_id="other", reservation_id="other-res", capture_id="other-cap"),
                      "other-handle", 1, 400, 128 * 1024 * 1024, 4096)
    before = (dict(r.used_nonces), dict(r.attempts), r.records["handle"].state)
    with pytest.raises(LatentError) as caught:
        r.reserve_consume(consume())
    assert caught.value.code == "busy"
    assert (dict(r.used_nonces), dict(r.attempts), r.records["handle"].state) == before


@pytest.mark.asyncio
async def test_consume_registry_busy_then_identical_resend_is_admitted(harness):
    h = harness
    _reusable_consume(h)
    reserve = h.backend.reserve
    state = {"busy": True}
    async def busy(a, *args):
        if state.pop("busy", False):
            h.backend.reserves.append((a,) + args)
            raise AttachmentError("scheduler_busy", 503)
        return await reserve(a, *args)
    h.backend.reserve = busy
    proof = h.backend.release_unused_attempt = _Proof()
    a = attachment()
    first = await consume_call(h, a)
    assert first.status_code == 503 and first.headers["retry-after"] == "5"
    assert json.loads(first.body)["error"]["message"] == "scheduler_busy"
    assert not proof.calls and not h.tm.calls    # registry decline is the proof
    second = await consume_call(h, a)
    assert second.status_code == 200


# --------------------------------------- Fable review 5: unfence after retire
@pytest.mark.asyncio
async def test_capture_busy_batch_unfences_only_after_the_record_is_retired(source_harness):
    h = source_harness
    proof = h.backend.release_unused_attempt = _Proof("unadmitted")
    order = []
    retire = h.backend.retire
    async def tracked(*args):
        order.append(("retire", bool(h.adapter._attempts)))
        return await retire(*args)
    h.backend.retire = tracked
    _aborting(h, ["occ3_busy_batch"])
    first, _, _ = await source_call(h)
    assert first.status_code == 503 and len(proof.calls) == 1
    assert order == [("retire", True)]          # fenced while retiring, retired once
    assert not h.adapter._attempts


@pytest.mark.asyncio
async def test_capture_busy_batch_keeps_the_fence_when_the_retire_fails(source_harness):
    h = source_harness
    h.backend.release_unused_attempt = _Proof("unadmitted")
    calls = []
    async def failing(*args):
        calls.append(args)
        raise LatentError("state")
    h.backend.retire = failing
    _aborting(h, ["occ3_busy_batch"])
    first, _, _ = await source_call(h)
    assert first.status_code == 503
    assert len(calls) == 2                      # early retire, then the cleanup retry
    second, _, _ = await source_call(h)
    assert second.status_code == 409 and b"replay_forbidden" in second.body
