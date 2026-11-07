"""Keepalive streaming mode tests: offline, format-only, no GPU/SGLang install.

Covers the negotiated x-optimus-occ3-keepalive request opt-in for armed
attached streaming turns: strict admission, SSE comment keepalive cadence
while the inner native stream is stalled, the deferred receipt event before
the unchanged trailing [DONE] frame, byte-exact reconstruction of the inner
body after comment stripping, bounded post-start failure events and the
charge/lease settle on completion, failure and client disconnect. Synthetic
backend fixtures are NOT real device evidence. The ordinary buffered contract
stays regression-tested for requests without the negotiation header; the client
now offers keepalive by default.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import replace

import pytest
from starlette.responses import StreamingResponse

from occ3_sglang.attachment_contract import (
    AttachmentError, canonical, DEFERRED_RECEIPT_HEADER, DEFERRED_RECEIPT_MODE,
    KEEPALIVE_REQUEST_HEADER, KEEPALIVE_REQUEST_MODE, RECEIPT_HEADER,
    SOURCE_RECEIPT_HEADER,
)
from occ3_sglang.use_receipts import parse_terminal_event
from test_chat_attachment import attachment, chat_request, harness, raw_request
from test_chat_source_capture import intent, source_harness  # noqa: F401  (fixture import)

KEEPALIVE_FRAME = b": occ3-keepalive\n\n"
RECEIPT_EVENT_PREFIX = b"event: optimus.occ3.receipt\ndata: "


def keepalive_raw(body, header=KEEPALIVE_REQUEST_MODE, source=False):
    raw, _ = raw_request(body)
    if header is not None:
        raw.scope["headers"].append((KEEPALIVE_REQUEST_HEADER.encode(), header.encode()))
    if source:
        raw.scope["headers"].append((b"accept-encoding", b"identity"))
    return raw


def extract_receipt_event(body):
    start = body.rindex(RECEIPT_EVENT_PREFIX)
    return body[start:body.index(b"\n\n", start) + 2]


def strip_comments_and_receipt(body):
    """Client-side reconstruction: keepalive comment frames and the receipt
    event frame are transport framing, not attested body bytes."""
    lines = body.split(b"\n")
    out = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(b":"):
            # One comment event frame: the comment line plus its terminating
            # blank line, exactly as the server emits it.
            if index + 1 < len(lines) and lines[index + 1] == b"":
                index += 2
            else:
                index += 1
            continue
        if line.startswith(b"event: optimus.occ3.receipt"):
            # The receipt event frame: event line, data line, blank line.
            assert index + 2 < len(lines) and lines[index + 2] == b""
            index += 3
            continue
        out.append(line)
        index += 1
    return b"\n".join(out)


def fast_profile(h, seconds):
    h.adapter.profile = replace(h.adapter.profile, keepalive_seconds=seconds)


async def spin_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    assert predicate(), "fixture did not reach checkpoint"


async def drive(h, raw, request=None):
    """Run handle_request and collect the streamed client bytes with times."""
    response = await h.chat.handle_request(request or chat_request(), raw)
    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert response.headers.get(DEFERRED_RECEIPT_HEADER) == DEFERRED_RECEIPT_MODE
    assert response.headers.get("content-encoding") == "identity"
    assert response.headers.get("cache-control") == "no-store"
    assert RECEIPT_HEADER not in response.headers
    assert SOURCE_RECEIPT_HEADER not in response.headers
    events = []

    async def send(event):
        events.append((time.monotonic(), event))

    await response(raw.scope, raw.receive, send)
    body = b"".join(event.get("body", b"") for _, event in events
                    if event["type"] == "http.response.body")
    return response, events, body


@pytest.mark.asyncio
async def test_opt_in_header_is_strict(harness):
    for bad in ("v2", "", "keepalive-v1 "):
        raw = keepalive_raw(canonical({"occ3": attachment().wire()}), header=bad)
        response = await harness.chat.handle_request(chat_request(), raw)
        assert response.status_code == 409
    duplicate = keepalive_raw(canonical({"occ3": attachment().wire()}))
    duplicate.scope["headers"].append((KEEPALIVE_REQUEST_HEADER.encode(), b"keepalive-v1"))
    response = await harness.chat.handle_request(chat_request(), duplicate)
    assert response.status_code == 409
    assert not harness.tm.calls and not harness.backend.reserves


@pytest.mark.asyncio
async def test_keepalive_nonstream_refuses_without_dispatch(harness):
    raw = keepalive_raw(canonical({"occ3": attachment().wire()}))
    response = await harness.chat.handle_request(chat_request(stream=False), raw)
    assert response.status_code == 409
    assert not harness.tm.calls and not harness.backend.reserves


@pytest.mark.asyncio
async def test_default_stays_buffered_with_receipt_header(harness):
    raw, _ = raw_request(canonical({"occ3": attachment().wire()}))
    response = await harness.chat.handle_request(chat_request(), raw)
    assert not isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert RECEIPT_HEADER in response.headers
    assert DEFERRED_RECEIPT_HEADER not in response.headers


@pytest.mark.asyncio
async def test_keepalive_receiver_flow_cadence_and_completion(harness):
    fast_profile(harness, 0.05)
    # The pinned native handler kick-starts the first rendered frame inside
    # handle_request, so the stall must sit between later frames.
    gate = harness.chat.buffer_gate = asyncio.Event()
    raw = keepalive_raw(canonical({"occ3": attachment().wire()}))
    response = await harness.chat.handle_request(chat_request(), raw)
    events = []

    async def send(event):
        events.append((time.monotonic(), event))

    task = asyncio.create_task(response(raw.scope, raw.receive, send))
    await spin_until(lambda: sum(1 for _, event in events
                                 if event["type"] == "http.response.body"
                                 and event.get("body") == KEEPALIVE_FRAME) >= 3)
    keepalive_times = [t for t, event in events
                       if event["type"] == "http.response.body"
                       and event.get("body") == KEEPALIVE_FRAME]
    for earlier, later in zip(keepalive_times, keepalive_times[1:]):
        assert later - earlier >= 0.045
    gate.set()
    await task
    body = b"".join(event.get("body", b"") for _, event in events
                    if event["type"] == "http.response.body")
    inner = b"".join(harness.chat.frames)
    assert body.count(KEEPALIVE_FRAME) >= 3
    # The receipt event sits between the last chat frame and the exact [DONE].
    envelope = parse_terminal_event(extract_receipt_event(body))
    assert envelope["kind"] == "receiver"
    assert envelope["body_sha256"] == hashlib.sha256(inner).hexdigest()
    assert envelope["native_body_bytes"] == len(inner)
    assert envelope["receipt"]["consumed"] is True
    assert envelope["receipt"]["fallback_used"] is False
    # Stripping transport framing reconstructs the attested inner bytes exactly.
    assert strip_comments_and_receipt(body) == inner
    # Settled like the buffered path: no aborts, no retained charge.
    await spin_until(lambda: harness.adapter._active == 0)
    assert not harness.tm.aborts
    assert harness.tm.closed == 1
    assert len(harness.backend.finishes) == 1


@pytest.mark.asyncio
async def test_keepalive_source_flow_deferred_receipt_event(source_harness):
    fast_profile(source_harness, 0.02)
    gate = source_harness.chat.buffer_gate = asyncio.Event()
    raw = keepalive_raw(canonical({"occ3_capture": intent().wire()}), source=True)
    response = await source_harness.chat.handle_request(chat_request(), raw)
    events = []

    async def send(event):
        events.append(event)

    task = asyncio.create_task(response(raw.scope, raw.receive, send))
    await spin_until(lambda: sum(1 for event in events
                                 if event["type"] == "http.response.body"
                                 and event.get("body") == KEEPALIVE_FRAME) >= 1)
    gate.set()
    await task
    body = b"".join(event.get("body", b"") for event in events
                    if event["type"] == "http.response.body")
    inner = b"".join(source_harness.chat.frames)
    assert body.count(KEEPALIVE_FRAME) >= 1
    envelope = parse_terminal_event(extract_receipt_event(body))
    assert envelope["kind"] == "source"
    assert envelope["body_sha256"] == hashlib.sha256(inner).hexdigest()
    assert envelope["native_body_bytes"] == len(inner)
    assert envelope["receipt"]["output_sha256"] == hashlib.sha256(inner).hexdigest()
    assert strip_comments_and_receipt(body) == inner
    assert len(source_harness.backend.source_finishes) == 1
    await spin_until(lambda: source_harness.adapter._active == 0)
    assert not source_harness.tm.aborts


@pytest.mark.asyncio
async def test_keepalive_failure_emits_bounded_error_event(harness):
    fast_profile(harness, 0.02)
    harness.backend.secret_failure = "finish"
    gate = harness.chat.buffer_gate = asyncio.Event()
    raw = keepalive_raw(canonical({"occ3": attachment().wire()}))
    response = await harness.chat.handle_request(chat_request(), raw)
    events = []

    async def send(event):
        events.append(event)

    task = asyncio.create_task(response(raw.scope, raw.receive, send))
    await spin_until(lambda: sum(1 for event in events
                                 if event["type"] == "http.response.body"
                                 and event.get("body") == KEEPALIVE_FRAME) >= 1)
    gate.set()
    await task
    body = b"".join(event.get("body", b"") for event in events
                    if event["type"] == "http.response.body")
    assert body.count(KEEPALIVE_FRAME) >= 1
    # No native bytes and no receipt event leak on failure.
    assert b"choices" not in body
    assert b"optimus.occ3.receipt" not in body
    error_frames = [line for line in body.split(b"\n") if line.startswith(b"data: ")]
    assert len(error_frames) == 1
    payload = json.loads(error_frames[0][6:])
    assert payload["error"]["type"] == "occ3_attachment_error"
    assert payload["error"]["message"] == "attachment_failed"
    assert payload["error"]["code"] == 409
    # The settle still ran: retire happened, charge released.
    await spin_until(lambda: harness.adapter._active == 0)
    assert harness.backend.retires
    assert harness.tm.closed == 1


@pytest.mark.asyncio
async def test_keepalive_client_disconnect_settles(harness):
    fast_profile(harness, 0.02)
    harness.chat.buffer_gate = asyncio.Event()  # Never released: the drain stalls.
    raw, queue = raw_request(canonical({"occ3": attachment().wire()}))
    raw.scope["headers"].append((KEEPALIVE_REQUEST_HEADER.encode(), KEEPALIVE_REQUEST_MODE.encode()))
    response = await harness.chat.handle_request(chat_request(), raw)
    events = []

    async def send(event):
        events.append(event)

    task = asyncio.create_task(response(raw.scope, raw.receive, send))
    await spin_until(lambda: sum(1 for e in events
                                 if e.get("body") == KEEPALIVE_FRAME) >= 3)
    queue.put_nowait({"type": "http.disconnect"})
    await asyncio.wait_for(task, timeout=5)
    # The response already started, so no error status exists; the settle is
    # the contract: abort the native request, retire the attempt, release.
    await spin_until(lambda: harness.adapter._active == 0)
    assert harness.tm.aborts
    assert harness.backend.retires
    assert harness.tm.closed == 1

@pytest.mark.asyncio
async def test_keepalive_receive_window_frames_flow(harness):
    """The receive window (slow first inner output) is covered by frames.

    TM.first_gate models the receiver's native generation time-to-first-output:
    the reserved-prefix prefill plus the first decode step. The pinned native
    handler awaits that first frame inside handle_request, so before the
    receive-phase extension the response could not even start. With the
    extension the deferred-receipt response starts immediately and the body
    races the native handler task with the keepalive cadence.
    """
    fast_profile(harness, 0.05)
    gate = harness.tm.first_gate = asyncio.Event()
    raw = keepalive_raw(canonical({"occ3": attachment().wire()}))
    response = await harness.chat.handle_request(chat_request(), raw)
    assert isinstance(response, StreamingResponse)
    assert response.headers.get(DEFERRED_RECEIPT_HEADER) == DEFERRED_RECEIPT_MODE
    events = []

    async def send(event):
        events.append((time.monotonic(), event))

    task = asyncio.create_task(response(raw.scope, raw.receive, send))
    await spin_until(lambda: sum(1 for _, event in events
                                  if event["type"] == "http.response.body"
                                  and event.get("body") == KEEPALIVE_FRAME) >= 3)
    keepalive_times = [t for t, event in events
                       if event["type"] == "http.response.body"
                       and event.get("body") == KEEPALIVE_FRAME]
    for earlier, later in zip(keepalive_times, keepalive_times[1:]):
        assert later - earlier >= 0.045
    # No native byte leaked while the first inner output was still pending.
    assert not any(b"choices" in (event.get("body") or b"")
                   for _, event in events if event["type"] == "http.response.body")
    gate.set()
    await task
    body = b"".join(event.get("body", b"") for _, event in events
                   if event["type"] == "http.response.body")
    inner = b"".join(harness.chat.frames)
    assert body.count(KEEPALIVE_FRAME) >= 3
    envelope = parse_terminal_event(extract_receipt_event(body))
    assert envelope["kind"] == "receiver"
    assert envelope["body_sha256"] == hashlib.sha256(inner).hexdigest()
    assert strip_comments_and_receipt(body) == inner
    await spin_until(lambda: harness.adapter._active == 0)
    assert not harness.tm.aborts
    assert harness.tm.closed == 1
    assert len(harness.backend.finishes) == 1


@pytest.mark.asyncio
async def test_keepalive_receive_window_disconnect_settles(harness):
    """A client that vanishes inside the receive window still settles.

    The body never gets past the receive race, so the settle path must cancel
    the native handler task, abort the already-dispatched native request and
    release the charge.
    """
    fast_profile(harness, 0.02)
    harness.tm.first_gate = asyncio.Event()  # Never released.
    raw, queue = raw_request(canonical({"occ3": attachment().wire()}))
    raw.scope["headers"].append((KEEPALIVE_REQUEST_HEADER.encode(), KEEPALIVE_REQUEST_MODE.encode()))
    response = await harness.chat.handle_request(chat_request(), raw)
    events = []

    async def send(event):
        events.append(event)

    task = asyncio.create_task(response(raw.scope, raw.receive, send))
    await spin_until(lambda: sum(1 for e in events
                                 if e.get("body") == KEEPALIVE_FRAME) >= 3)
    queue.put_nowait({"type": "http.disconnect"})
    await asyncio.wait_for(task, timeout=5)
    await spin_until(lambda: harness.adapter._active == 0)
    assert harness.tm.aborts
    assert harness.backend.retires


@pytest.mark.asyncio
async def test_steady_decode_keeps_client_visible_cadence(harness):
    """Fast inner chunks are buffered, but cannot starve transport comments."""
    fast_profile(harness, 0.02)
    harness.adapter.profile = replace(harness.adapter.profile, disconnect_interval=0.05)
    original = harness.chat._generate_chat_stream

    async def steady_stream(obj, request, raw):
        async for frame in original(obj, request, raw):
            yield frame
            if len(harness.chat.frames) == 1:
                for _ in range(120):
                    await asyncio.sleep(0.001)
                    chunk = "data: " + json.dumps({"id": obj.rid, "choices": [
                        {"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}) + "\n\n"
                    harness.chat.frames.append(chunk.encode())
                    yield chunk

    harness.chat._generate_chat_stream = steady_stream
    _, events, body = await drive(
        harness, keepalive_raw(canonical({"occ3": attachment().wire()})))
    times = [stamp for stamp, event in events if event.get("body") == KEEPALIVE_FRAME]
    assert len(times) >= 3
    # Allow scheduler jitter, but not an unbounded client-visible idle gap.
    visible = [stamp for stamp, event in events
               if event["type"] == "http.response.start" or event.get("body")]
    assert all(later - earlier < 0.1 for earlier, later in zip(visible, visible[1:]))
    assert strip_comments_and_receipt(body) == b"".join(harness.chat.frames)
    await spin_until(lambda: harness.adapter._active == 0)
    assert not harness.tm.aborts


@pytest.mark.asyncio
@pytest.mark.parametrize("keepalive", [False, True])
async def test_task_cancellation_has_distinct_summary(harness, caplog, keepalive):
    """Explicit task cancellation is not evidence of client disconnect."""
    import logging
    caplog.set_level(logging.INFO)
    harness.chat.buffer_gate = asyncio.Event()
    fast_profile(harness, 0.01)
    streaming = asyncio.Event()
    raw = keepalive_raw(canonical({"occ3": attachment().wire()}),
                        header=KEEPALIVE_REQUEST_MODE if keepalive else None)

    async def run():
        response = await harness.chat.handle_request(chat_request(), raw)
        if keepalive:
            async for _ in response.body_iterator:
                streaming.set()

    task = asyncio.create_task(run())
    # Ensure the streaming owner is inside its body, not at the hand-off seam.
    await spin_until(lambda: streaming.is_set() if keepalive else bool(harness.chat.frames))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    prefix = "occ3-bridge: native turn summary "
    await spin_until(lambda: harness.adapter._active == 0 and any(
        record.message.startswith(prefix) for record in caplog.records))
    row, = [json.loads(record.message[len(prefix):]) for record in caplog.records
            if record.message.startswith(prefix)]
    assert row["reason"] == "cancelled"
    assert row["status"] == 499
    assert row["keepalive"] is keepalive
    assert harness.tm.aborts
    assert harness.backend.retires
