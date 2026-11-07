"""Output-only telemetry regression coverage (no engine required)."""
import json
import logging

import pytest
from occ3_sglang.turn_diagnostics import TurnDiagnostics
from occ3_sglang.attachment_contract import canonical
from test_chat_attachment import attachment, chat_request, harness, raw_request


def rows(caplog):
    return [json.loads(r.message.split("occ3-bridge: native turn summary ", 1)[1])
            for r in caplog.records if r.message.startswith("occ3-bridge: native turn summary ")]


@pytest.mark.parametrize("setting,expected", [(None, 0), ("bad", 0), ("-2", 0), ("99999", 4096), ("3", 3)])
def test_bounded_output_only(monkeypatch, caplog, setting, expected):
    if setting is None:
        monkeypatch.delenv("OCC3_DIAG_TURN_TEXT", raising=False)
    else:
        monkeypatch.setenv("OCC3_DIAG_TURN_TEXT", setting)
    caplog.set_level(logging.INFO)
    d = TurnDiagnostics(lambda: 10, "attempt", "receiver", True)
    d.inner_event()
    payload = {"choices": [{"delta": {"content": 'a\n"secret', "reasoning_content": "thought"}, "finish_reason": "stop"}],
               "usage": {"completion_tokens": 12, "completion_tokens_details": {"reasoning_tokens": 4}},
               "prompt": "NEVER_LOG", "headers": "NEVER_LOG",
               "authorization": "AUTH_SENTINEL", "receipt": {"token": "RECEIPT_SENTINEL"}}
    frame = b"data: " + json.dumps(payload).encode() + b"\n\n"
    d.feed(frame[:17]); d.feed(frame[17:])
    d.emit(True, 20); d.emit(True, 20)
    row, = rows(caplog)
    assert row["content_chars"] == 9 and row["reasoning_chars"] == 7
    assert row["completion_tokens"] == 12 and row["reasoning_tokens"] == 4
    assert row["event_count"] == 1 and row["first_inner_event_seconds"] == 0
    assert "NEVER_LOG" not in caplog.text
    assert "AUTH_SENTINEL" not in caplog.text and "RECEIPT_SENTINEL" not in caplog.text
    assert d.limit == expected
    if expected:
        assert len(row["content_first"]) <= expected
        assert "\n" not in caplog.records[-1].message
    else:
        assert "content_first" not in row and "secret" not in caplog.text


@pytest.mark.parametrize("complete,failure,now,expected", [
    (True, None, 1, "complete"), (False, None, 1, "client_disconnected"),
    (False, None, 30, "deadline_exceeded"),
    (False, ("native_completion_failed", 409), 1, "native_completion_failed")])
def test_outcome(caplog, complete, failure, now, expected):
    caplog.set_level(logging.INFO)
    d = TurnDiagnostics(lambda: now, "attempt", "source", False)
    if failure:
        d.fail(*failure)
    d.emit(complete, 20)
    row, = rows(caplog)
    assert row["reason"] == expected
    assert row["first_inner_event_seconds"] is None


@pytest.mark.asyncio
async def test_buffered_summary(harness, caplog):
    caplog.set_level(logging.INFO)
    raw, _ = raw_request(canonical({"occ3": attachment().wire()}))
    response = await harness.chat.handle_request(chat_request(), raw)
    assert response.status_code == 200
    row, = rows(caplog)
    assert row["reason"] == "complete"
    assert row["event_count"] > 0
    assert row["keepalive"] is False


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_split_frames(ending):
    d = TurnDiagnostics(lambda: 0, "attempt", "source", False)
    frame = b'data: {"choices":[{"delta":{"reasoning":"hello"}}]}' + ending * 2
    for byte in frame:
        d.feed(bytes([byte]))
    assert d.chars["reasoning"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("keepalive", [False, True])
@pytest.mark.parametrize("disconnect", [False, True])
async def test_cut_summary(harness, caplog, keepalive, disconnect):
    import asyncio
    from dataclasses import replace
    from test_chat_stream import spin_until
    from occ3_sglang.attachment_contract import KEEPALIVE_REQUEST_HEADER, KEEPALIVE_REQUEST_MODE
    caplog.set_level(logging.INFO)
    h = harness
    h.adapter.profile = replace(h.adapter.profile, max_turn_seconds=.12, deadline_margin_seconds=.001,
                                keepalive_seconds=.01)
    h.backend.required_seconds = .001
    gate = asyncio.Event()
    original_stream = h.chat._generate_chat_stream
    async def partial_stream(obj, request, raw):
        async for frame in original_stream(obj, request, raw):
            yield frame
            yield "data: " + json.dumps({"id": obj.rid, "choices": [{"index": 0,
                "delta": {"content": "partial visible", "reasoning_content": "partial reasoning"},
                "finish_reason": None}]}) + "\n\n"
            await gate.wait()
    h.chat._generate_chat_stream = partial_stream
    raw, queue = raw_request(canonical({"occ3": attachment().wire()}))
    if keepalive:
        raw.scope["headers"].append((KEEPALIVE_REQUEST_HEADER.encode(), KEEPALIVE_REQUEST_MODE.encode()))
    async def run():
        response = await h.chat.handle_request(chat_request(), raw)
        if keepalive:
            async for _ in response.body_iterator:
                pass
    task = asyncio.create_task(run())
    await spin_until(lambda: h.chat.frames)
    await asyncio.sleep(.02)  # allow the partial renderer frame to reach the drain
    if disconnect:
        queue.put_nowait({"type": "http.disconnect"})
    await asyncio.wait_for(task, 2)
    row, = rows(caplog)
    assert row["reason"] == ("client_disconnected" if disconnect else "deadline_exceeded")
    assert row["keepalive"] is keepalive
    assert row["event_count"] > 0

    assert row["content_chars"] == len("partial visible")
    assert row["reasoning_chars"] == len("partial reasoning")


@pytest.mark.asyncio
async def test_keepalive_success_summary(harness, caplog):
    from test_chat_stream import drive, keepalive_raw
    caplog.set_level(logging.INFO)
    await drive(harness, keepalive_raw(canonical({"occ3": attachment().wire()})))
    row, = rows(caplog)
    assert row["reason"] == "complete" and row["keepalive"] is True
    assert row["reasoning_chars"] == len("fixture reasoning")
