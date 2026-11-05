"""ASGI cancellation tests with deterministic engine/backend dependencies."""
import asyncio
import time
from dataclasses import replace

import pytest

from occ3_sglang.attachment_contract import RECEIPT_HEADER, canonical
from test_chat_attachment import attachment, chat_request, harness, raw_request


async def spin_until(predicate):
    for _ in range(300):
        if predicate(): return
        await asyncio.sleep(0)
    assert predicate(), "fixture did not reach checkpoint"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["reserve", "first_prefetch", "buffer", "evidence"])
async def test_disconnect_before_any_asgi_output(harness, phase):
    gate = asyncio.Event()
    if phase == "reserve": harness.backend.reserve_gate = gate
    if phase == "first_prefetch": harness.tm.first_gate = gate
    if phase == "buffer": harness.chat.buffer_gate = gate
    if phase == "evidence": harness.backend.finish_gate = gate
    raw, receive_queue = raw_request(canonical({"occ3": attachment().wire()}))
    sent = []
    async def send(event): sent.append(event)
    async def run():
        response = await harness.chat.handle_request(chat_request(), raw)
        # Real ASGI invocation after failed handler, never provisional response start.
        await response(raw.scope, raw.receive, send)
        return response
    task = asyncio.create_task(run())
    reached = {"reserve": lambda: harness.backend.reserves,
               "first_prefetch": lambda: harness.tm.calls,
               "buffer": lambda: harness.chat.frames,
               "evidence": lambda: harness.backend.finishes}[phase]
    await spin_until(reached)
    assert sent == []
    receive_queue.put_nowait({"type": "http.disconnect"})
    response = await asyncio.wait_for(task, 2)
    assert response.status_code == 499
    assert RECEIPT_HEADER not in response.headers
    assert sent[0]["status"] == 499
    assert len(harness.backend.retires) == 1
    if phase == "reserve":
        assert harness.tm.calls == [] and harness.tm.aborts == []
    else:
        assert harness.tm.closed == 1
        assert harness.tm.aborts == [harness.tm.calls[0].rid]
    assert harness.adapter._active == 0


@pytest.mark.asyncio
async def test_task_cancellation_closes_native_and_never_replays(harness):
    harness.tm.first_gate = asyncio.Event()
    a = attachment()
    raw, _ = raw_request(canonical({"occ3": a.wire()}))
    task = asyncio.create_task(harness.chat.handle_request(chat_request(), raw))
    await spin_until(lambda: harness.tm.calls)
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert harness.tm.closed == 1 and len(harness.tm.aborts) == 1
    assert len(harness.backend.retires) == 1 and harness.adapter._active == 0
    raw2, _ = raw_request(canonical({"occ3": a.wire()}))
    result = await harness.chat.handle_request(chat_request(), raw2)
    assert result.status_code != 200 and len(harness.tm.calls) == 1


@pytest.mark.asyncio
async def test_deadline_during_first_chunk_aborts(harness):
    harness.tm.first_gate = asyncio.Event()
    harness.backend.required_seconds = .001
    harness.adapter.profile = replace(harness.adapter.profile, max_turn_seconds=.04,
                                      deadline_margin_seconds=.001)
    raw, _ = raw_request(canonical({"occ3": attachment().wire()}))
    response = await harness.chat.handle_request(chat_request(), raw)
    assert response.status_code == 504
    assert harness.tm.closed == 1 and len(harness.tm.aborts) == 1
    assert len(harness.backend.retires) == 1


@pytest.mark.asyncio
async def test_quota_rejects_without_second_generation(harness):
    harness.adapter.profile = replace(harness.adapter.profile, max_active=1)
    harness.tm.first_gate = asyncio.Event()
    raw, queue = raw_request(canonical({"occ3": attachment().wire()}))
    task = asyncio.create_task(harness.chat.handle_request(chat_request(), raw))
    await spin_until(lambda: harness.tm.calls)
    second, _ = raw_request(canonical({"occ3": attachment(state_handle="other-handle").wire()}))
    response = await harness.chat.handle_request(chat_request(), second)
    # Adapter concurrency refusal is proven non-use: the retryable busy contract.
    assert response.status_code == 503 and response.headers["retry-after"] == "5"
    assert len(harness.tm.calls) == 1
    queue.put_nowait({"type": "http.disconnect"})
    await task
    assert harness.adapter._active == 0


@pytest.mark.asyncio
async def test_uncooperative_reservation_quarantines_quota_and_cannot_dispatch_late(harness):
    harness.adapter.profile = replace(harness.adapter.profile, max_active=1, cleanup_seconds=.02)
    started, release = asyncio.Event(), asyncio.Event()
    reserve = harness.backend.reserve
    async def stubborn(*args):
        reservation = await reserve(*args)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
        return reservation
    harness.backend.reserve = stubborn
    raw, queue = raw_request(canonical({"occ3": attachment().wire()}))
    task = asyncio.create_task(harness.chat.handle_request(chat_request(), raw))
    await asyncio.wait_for(started.wait(), 1)
    queue.put_nowait({"type": "http.disconnect"})
    response = await asyncio.wait_for(task, 1)
    assert response.status_code == 499
    assert harness.adapter._active == 1 and harness.adapter._quarantined
    assert not harness.tm.calls and len(harness.backend.retires) == 1
    other, _ = raw_request(canonical({"occ3": attachment(state_handle="new-handle").wire()}))
    denied = await harness.chat.handle_request(chat_request(), other)
    assert denied.status_code == 503 and b"scheduler_busy" in denied.body
    release.set()
    await spin_until(lambda: harness.adapter._active == 0)
    assert not harness.tm.calls and not harness.adapter._quarantined


@pytest.mark.asyncio
async def test_authorizer_wait_is_bounded_and_cannot_start_inference(harness):
    started = asyncio.Event()
    async def pending(raw):
        started.set()
        await asyncio.Event().wait()
    harness.adapter.authorize = pending
    raw, queue = raw_request(canonical({"occ3": attachment().wire()}))
    task = asyncio.create_task(harness.chat.handle_request(chat_request(), raw))
    await started.wait()
    assert harness.adapter._active == 1 and not harness.chat.converted
    queue.put_nowait({"type": "http.disconnect"})
    response = await asyncio.wait_for(task, 1)
    assert response.status_code == 499
    assert not harness.tm.calls and not harness.backend.reserves
    assert harness.adapter._active == 0


@pytest.mark.asyncio
async def test_deadline_cut_latency_stays_within_bounded_cleanup_grace(harness):
    """A responsive fixture loop bounds cancellation-resistant cleanup.

    This small-scale check does not bound event-loop stalls on a GPU server
    or identify the cause of a live interruption.
    """
    turn, cleanup = .30, .05
    harness.adapter.profile = replace(
        harness.adapter.profile, max_turn_seconds=turn, cleanup_seconds=cleanup,
        disconnect_interval=.005, deadline_margin_seconds=.001)
    harness.backend.required_seconds = .001
    # Live producer: the native turn keeps its stream open past the window.
    harness.chat.buffer_gate = asyncio.Event()
    blocker = asyncio.Event()
    async def stubborn(*_args):
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            await blocker.wait()  # cancellation-resistant, like a wedged engine call
    harness.tm.abort_request = stubborn
    harness.backend.retire = stubborn
    raw, _ = raw_request(canonical({"occ3": attachment().wire()}))
    started = time.monotonic()
    response = await asyncio.wait_for(harness.chat.handle_request(chat_request(), raw), 2)
    elapsed = time.monotonic() - started
    blocker.set()
    await spin_until(lambda: not harness.adapter._quarantined)
    assert response.status_code == 504, (response.status_code, response.body)
    assert elapsed >= turn, (elapsed, turn)
    assert elapsed <= turn + 3 * cleanup, (elapsed, turn, cleanup)
