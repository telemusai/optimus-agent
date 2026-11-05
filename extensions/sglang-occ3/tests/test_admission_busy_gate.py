"""Admission busy-batch gate: a capture admits only against an empty batch.

A capture turn runs as an isolated cohort for its whole decode, so ordinary
inflight work must drain before its prefill. A consume admits while ordinary
rows decode (its prefill is a one-request extend batch, then it decodes in
the shared batch); it declines only behind another eager row. Parking the one-shot native attempt in the
ordinary queue would hold it behind unrelated decode traffic until the
queue bound aborts it, so admission against a busy batch declines fast,
explicitly and rank-uniformly instead: the scheduler wrapper folds the
decline into the bounded AbortReq(503, "occ3_admission_failed") shape the
sender treats as retryable with a fresh attempt. A decline mutates nothing:
the request keeps its ordinary shape and never enters the eager cohort.
"""
from array import array
from types import SimpleNamespace as NS

import pytest

from occ3_sglang import latent_worker as w
from occ3_sglang.latent_types import LatentError
from test_latent_types import capture
from test_latent_registry import consume


def req(descriptor=None):
    return NS(rid="rid", output_ids=array("q"), origin_input_ids=array("q", [1, 2, 3]),
              origin_input_ids_unpadded=array("q", [1, 2, 3]),
              full_untruncated_fill_ids=array("q", [1, 2, 3]),
              extend_range=NS(end=3), session=None, multimodal_inputs=None,
              positional_embed_overrides=None, occ3_worker_descriptor=descriptor,
              retraction_count=0)


def capture_record():
    return NS(state="reserved", admitted_capture=False, state_generation=123)


def consume_record():
    return NS(state="leased", admitted_consume=False, state_generation=123)


def runtime_for(record):
    return NS(poisoned=False, resolve=lambda value, **kw: record,
              preflight=lambda phase, check: check())


def active(rid="ordinary"):
    return NS(rid=rid, finished=lambda: False)


def finished(rid="ordinary"):
    return NS(rid=rid, finished=lambda: True)


def scheduler(runtime, *, inflight=(), chunked=None):
    return NS(occ3_runtime=runtime, collect_inflight_reqs=lambda: list(inflight),
              chunked_req=chunked)


def bind(monkeypatch, runtime):
    monkeypatch.setattr(w, "_RUNTIME", runtime)
    return runtime


def test_empty_batch_admits_immediately(monkeypatch):
    record = capture_record()
    runtime = bind(monkeypatch, runtime_for(record))
    request = req()
    w.admit_request(scheduler(runtime), request, capture())
    assert record.admitted_capture and record.state == "capturing"
    assert request.occ3_worker_descriptor is not None and request.occ3_force_eager


def test_missing_batch_state_admits(monkeypatch):
    # Schedulers without inflight surface (tests, early bootstrap) are not busy.
    record = capture_record()
    runtime = bind(monkeypatch, runtime_for(record))
    w.admit_request(NS(occ3_runtime=runtime), req(), capture())
    assert record.admitted_capture


def test_busy_running_batch_declines_before_any_mutation(monkeypatch):
    record = capture_record()
    runtime = bind(monkeypatch, runtime_for(record))
    request = req()
    with pytest.raises(LatentError) as caught:
        w.admit_request(scheduler(runtime, inflight=[active()]), request, capture())
    assert caught.value.code == "quota"
    # Nothing mutated: no descriptor binding, no eager flag, no capture state.
    assert request.occ3_worker_descriptor is None
    assert getattr(request, "occ3_force_eager", None) is None
    assert record.admitted_capture is False and record.state == "reserved"


def test_consume_admits_while_two_ordinary_requests_decode(monkeypatch):
    # A consume prefill runs as its own one-request extend batch and then
    # decodes in the shared batch, so ordinary decode traffic never blocks it.
    record = consume_record()
    runtime = bind(monkeypatch, runtime_for(record))
    request = req()
    descriptor = consume(native_rid="rid")
    w.admit_request(scheduler(runtime, inflight=[active("a"), active("b")]), request, descriptor)
    assert record.admitted_consume
    assert request.occ3_attached and request.occ3_force_eager and request.occ3_private_prefix
    assert len(request.origin_input_ids) == 3 + descriptor.expected_prefix_tokens


def test_consume_declines_without_punching_prefix_holes_behind_an_eager_row(monkeypatch):
    # An eager row (a capture turn, or another consume's injected prefill)
    # drains new prefills; the consume declines busy, pre-use, unmutated.
    record = consume_record()
    runtime = bind(monkeypatch, runtime_for(record))
    request = req()
    eager = NS(rid="capture", finished=lambda: False, occ3_force_eager=True)
    with pytest.raises(LatentError) as caught:
        w.admit_request(scheduler(runtime, inflight=[active(), eager]), request, consume(native_rid="rid"))
    assert caught.value.code == "quota"
    assert list(request.origin_input_ids) == [1, 2, 3]
    assert request.occ3_worker_descriptor is None and getattr(request, "occ3_attached", None) is None
    assert record.admitted_consume is False and record.state == "leased"


def test_capture_still_declines_while_others_decode(monkeypatch):
    record = capture_record()
    runtime = bind(monkeypatch, runtime_for(record))
    request = req()
    with pytest.raises(LatentError) as caught:
        w.admit_request(scheduler(runtime, inflight=[active("a"), active("b")]), request, capture())
    assert caught.value.code == "quota"
    assert request.occ3_worker_descriptor is None
    assert record.admitted_capture is False and record.state == "reserved"


def test_chunked_prefill_counts_as_busy(monkeypatch):
    runtime = bind(monkeypatch, runtime_for(capture_record()))
    with pytest.raises(LatentError) as caught:
        w.admit_request(scheduler(runtime, chunked=active("chunk")), req(), capture())
    assert caught.value.code == "quota"


def test_finished_inflight_requests_do_not_block(monkeypatch):
    record = capture_record()
    runtime = bind(monkeypatch, runtime_for(record))
    request = req()
    w.admit_request(scheduler(runtime, inflight=[finished(), finished("done-2")]), request, capture())
    assert record.admitted_capture and request.occ3_force_eager


def test_busy_decline_happens_inside_the_admission_preflight(monkeypatch):
    # The gate must decline inside the all-rank preflight so every rank
    # agrees on the same request/error before any typed cancel or mutation.
    record = capture_record()
    seen = {}

    def preflight(phase, check):
        seen["phase"] = phase
        try:
            check()
        except LatentError as error:
            seen["code"] = error.code
            raise
        raise AssertionError("expected the busy decline inside preflight")

    runtime = NS(poisoned=False, resolve=lambda value, **kw: record, preflight=preflight)
    bind(monkeypatch, runtime)
    with pytest.raises(LatentError) as caught:
        w.admit_request(scheduler(runtime, inflight=[active()]), req(), capture())
    assert seen["phase"] == "admit" and seen["code"] == "quota"
    assert caught.value.code == "quota"


def test_busy_decline_never_silently_queues_a_zero_event_turn(monkeypatch):
    # The observable contract of "no silent zero-event stream past the
    # bound": a declined request keeps the fully ordinary shape, so the
    # scheduler wrapper's explicit AbortReq(503, "occ3_admission_failed")
    # is the only outcome (pinned separately in test_latent_scheduler_patch
    # test_incomplete_worker_admission_never_queues_or_infers).
    record = capture_record()
    runtime = bind(monkeypatch, runtime_for(record))
    request = req()
    before = sorted(vars(request))
    with pytest.raises(LatentError):
        w.admit_request(scheduler(runtime, inflight=[active(), active("other")]), request, capture())
    assert sorted(vars(request)) == before
    assert getattr(request, "occ3_force_eager", None) is None


# ---- adapter integration: the scheduler busy refusal through the public wire ----
# These reuse the gated pinned-native harness from test_chat_attachment
# (OCC3_SGLANG_SOURCE); without that claim they skip, exactly like the chat
# suites. They prove the F-correction contract: the busy AbortReq reaches the
# client as an observable, differentiated, retryable 503 before any body byte.

import json as _json
import logging as _logging
from pathlib import Path as _Path

import pytest as _pytest

# Registers the gated pinned-native harness fixture from the chat suite.
pytest_plugins = ["test_chat_attachment"]

from occ3_sglang.attachment_contract import RECEIPT_HEADER as _RECEIPT_HEADER
from test_chat_attachment import attachment as _attachment, call as _call


@_pytest.mark.asyncio
async def test_scheduler_busy_abort_is_public_503_with_retry_after(harness, caplog):
    async def busy_generate(tm, obj, request=None):
        yield {"text": "", "meta_info": {"id": obj.rid,
               "finish_reason": {"type": "abort", "message": "occ3_busy_batch",
                                  "status_code": 503}}}

    type(harness.tm).generate_request = harness.wrap(
        busy_generate, harness.adapter.generate_around, harness.HookType.AROUND)
    with caplog.at_level(_logging.ERROR, logger="occ3_sglang.chat_adapter"):
        response, _, _ = await _call(harness)
    assert response.status_code == 503
    assert _json.loads(response.body) == {"error": {"message": "scheduler_busy",
                                                    "type": "occ3_attachment_error",
                                                    "code": 503}}
    assert response.headers.get("retry-after") == "5"
    aborted = [r.getMessage() for r in caplog.records if "native turn aborted" in r.getMessage()]
    assert aborted and "refusal=occ3_busy_batch" in aborted[0]
    # Only the fixed public code; never the raw refusal payload.
    assert b"occ3_busy_batch" not in response.body


@_pytest.mark.asyncio
async def test_busy_to_empty_race_retry_succeeds(harness):
    """A retry after the busy 503 is a fresh attempt: a new attachment id
    runs the ordinary empty-batch path to a successful completion."""
    busy = {"still": True}

    async def racing_generate(tm, obj, request=None):
        if busy["still"]:
            busy["still"] = False
            yield {"text": "", "meta_info": {"id": obj.rid,
                   "finish_reason": {"type": "abort", "message": "occ3_busy_batch",
                                      "status_code": 503}}}
            return
        yield {"text": "fixture", "meta_info": {"id": obj.rid,
                                                "finish_reason": {"type": "stop"}}}

    type(harness.tm).generate_request = harness.wrap(
        racing_generate, harness.adapter.generate_around, harness.HookType.AROUND)
    first, _, _ = await _call(harness)
    assert first.status_code == 503
    assert _json.loads(first.body)["error"]["message"] == "scheduler_busy"
    second, _, _ = await _call(harness, a=_attachment(state_handle="PRIVATE_HANDLE_2"))
    assert second.status_code == 200, second.body
    assert _RECEIPT_HEADER in second.headers


@_pytest.mark.asyncio
async def test_other_abort_codes_keep_the_generic_completion_fold(harness):
    async def abort_generate(tm, obj, request=None):
        yield {"text": "", "meta_info": {"id": obj.rid,
               "finish_reason": {"type": "abort", "message": "occ3_admission_failed",
                                  "status_code": 503}}}

    type(harness.tm).generate_request = harness.wrap(
        abort_generate, harness.adapter.generate_around, harness.HookType.AROUND)
    response, _, _ = await _call(harness)
    assert response.status_code == 409
    assert _json.loads(response.body)["error"]["message"] == "native_completion_failed"
    assert "retry-after" not in response.headers


def test_refusal_detail_sanitizes_arbitrary_keys_and_nested_values():
    from occ3_sglang.chat_adapter import _refusal_detail
    detail = _refusal_detail({
        "type": "abort",
        "message": "Abort in waiting queue with capture-1 payload",
        "secret-key-with-arbitrary-payload-text": 1,
        "caps KEY": True,
        "status_code": 503,
        "nested": {"x": "y"},
        17: "non-string key",
    })
    assert "capture-1" not in detail and "payload" not in detail
    assert "secret-key-with-arbitrary-payload-text" not in detail
    assert "caps KEY" not in detail and "17" not in detail
    assert "type=<str>" in detail and "status_code=503" in detail
    assert "<key>=1" in detail and "nested=<dict>" in detail and "message=<str>" in detail


def test_published_bounds_match_profile_and_manifest_window():
    """Retry-After stays advisory; the window header states the accurate
    end-to-end pre-first-event maximum (control + first-event), and the
    first-event bound must strictly cover the patched scheduler's armed
    admission window so a healthy scheduler's sweep abort always arrives
    inside the adapter's bound."""
    from occ3_sglang import chat_adapter
    assert chat_adapter.ADMISSION_BUSY_RETRY_AFTER_SECONDS == 5
    manifest = _json.loads((_Path(__file__).resolve().parents[1]
                            / "patches/scheduler-cache-ipc.v1.json").read_text(encoding="utf-8"))
    after = next(r["after"] for f in manifest["files"] if f["path"] == "managers/scheduler.py"
                 for r in f["replacements"] if "occ3_queue_deadline = min(" in r["after"])
    assert ("time.monotonic() + %d.0"
            % chat_adapter.ADMISSION_BUSY_RETRY_AFTER_SECONDS) in after
    # The scheduler window (5 s) is covered by the first-event bound, and the
    # published header is the sum of the two profile bounds.
    profile = harness_profile()
    assert profile.admission_first_event_seconds > 5.0
    assert profile.admission_control_seconds + profile.admission_first_event_seconds == 15.0


def harness_profile():
    from occ3_sglang.chat_adapter import BufferProfile
    return BufferProfile()


# ---- final gate: every client-observable admission case through the public wire ----

import asyncio as _asyncio
import time as _time
from dataclasses import replace as _replace

from occ3_sglang.attachment_contract import (
    KEEPALIVE_REQUEST_HEADER as _KEEPALIVE_HEADER,
    KEEPALIVE_REQUEST_MODE as _KEEPALIVE_MODE,
    canonical as _canonical,
)
from test_attachment_cancellation import spin_until as _spin_until
from test_chat_attachment import chat_request as _chat_request, raw_request as _raw_request


async def _abort_call(harness, message, *, stream=True, outputs=None):
    """Public call whose native stream aborts with the given refusal code.

    outputs overrides the yielded outputs entirely (for the already-admitted
    mid-stream case); by default the abort is the first and only output.
    """
    async def generate(tm, obj, request=None):
        for out in (outputs or [{"text": "", "meta_info": {
            "id": obj.rid, "finish_reason": {"type": "abort", "message": message,
                                              "status_code": 503}}}]):
            if out["meta_info"].get("id") is None:
                out["meta_info"]["id"] = obj.rid
            yield out

    type(harness.tm).generate_request = harness.wrap(
        generate, harness.adapter.generate_around, harness.HookType.AROUND)
    return await _call(harness, request=_chat_request(stream=stream))


@_pytest.mark.asyncio
@_pytest.mark.parametrize("message,public", [
    ("occ3_busy", "admission_capacity"),
    ("occ3_admission_timeout", "admission_timeout"),
])
async def test_admission_family_first_output_is_public_503(harness, message, public):
    response, _, _ = await _abort_call(harness, message)
    assert response.status_code == 503
    assert _json.loads(response.body)["error"]["message"] == public
    assert response.headers.get("retry-after") == "5"
    # Accurate end-to-end admission maximum: control bound (7.0) +
    # first-event bound (8.0); the manifest's 5 s scheduler window sits
    # inside the first-event bound.
    assert response.headers.get("x-optimus-occ3-admission-window") == "15.0"


@_pytest.mark.asyncio
async def test_user_cancel_keeps_409_under_its_own_code(harness):
    response, _, _ = await _abort_call(harness, "occ3_cancelled")
    assert response.status_code == 409
    assert _json.loads(response.body)["error"]["message"] == "client_cancelled"
    assert "retry-after" not in response.headers


@_pytest.mark.asyncio
async def test_midstream_abort_after_content_keeps_generic_fold(harness):
    """Generation already admitted: the same scheduler code after content
    flowed is a completion failure, never a retryable admission shape."""
    response, _, _ = await _abort_call(harness, "occ3_admission_timeout", outputs=[
        {"text": "partial native content", "meta_info": {"finish_reason": None}},
        {"text": "", "meta_info": {"finish_reason": {
            "type": "abort", "message": "occ3_admission_timeout", "status_code": 503}}},
    ])
    assert response.status_code == 409
    assert _json.loads(response.body)["error"]["message"] == "native_completion_failed"
    assert "retry-after" not in response.headers


@_pytest.mark.asyncio
async def test_nonstream_busy_abort_is_public_503(harness):
    response, _, _ = await _abort_call(harness, "occ3_busy", stream=False)
    assert response.status_code == 503
    assert _json.loads(response.body)["error"]["message"] == "admission_capacity"
    assert response.headers.get("retry-after") == "5"


@_pytest.mark.asyncio
async def test_keepalive_busy_abort_emits_structured_terminal_error(harness):
    """Post-start admission unavailability: the committed SSE stream ends
    with one bounded structured error event carrying the retryable code,
    status and retry advice - no HTTP status claim after headers."""
    response, raw, _ = await _keepalive_busy_response(harness)
    events = []

    async def send(event):
        events.append(event)

    task = _asyncio.create_task(response(raw.scope, raw.receive, send))
    await task
    body = b"".join(event.get("body", b"") for event in events
                    if event["type"] == "http.response.body")
    # No native bytes and no receipt on a refused turn.
    assert b"choices" not in body
    error_frames = [line for line in body.split(b"\n") if line.startswith(b"data: ")]
    assert len(error_frames) == 1
    payload = _json.loads(error_frames[0][6:])
    assert payload["error"]["message"] == "scheduler_busy"
    assert payload["error"]["code"] == 503
    assert payload["error"]["retryAfterSeconds"] == 5
    await _spin_until(lambda: harness.adapter._active == 0)


async def _keepalive_busy_response(harness):
    async def busy_generate(tm, obj, request=None):
        yield {"text": "", "meta_info": {"id": obj.rid,
               "finish_reason": {"type": "abort", "message": "occ3_busy_batch",
                                  "status_code": 503}}}

    type(harness.tm).generate_request = harness.wrap(
        busy_generate, harness.adapter.generate_around, harness.HookType.AROUND)
    raw, queue = _raw_request(_canonical({"occ3": _attachment().wire()}))
    raw.scope["headers"].append((_KEEPALIVE_HEADER.encode(), _KEEPALIVE_MODE.encode()))
    response = await harness.chat.handle_request(_chat_request(), raw)
    return response, raw, queue


@_pytest.mark.asyncio
async def test_control_deadline_bounds_pre_admission_waits(harness):
    """A control plane that never answers is admission unavailability within
    the profile bound - not a turn-length wait, and no captured state is
    reused (fresh attempt only)."""
    harness.adapter.profile = _replace(harness.adapter.profile, admission_control_seconds=0.05)
    harness.backend.reserve_gate = _asyncio.Event()  # never set
    response, _, _ = await _call(harness)
    assert response.status_code == 503
    assert _json.loads(response.body)["error"]["message"] == "admission_timeout"
    assert response.headers.get("retry-after") == "5"
    await _spin_until(lambda: harness.adapter._active == 0)


@_pytest.mark.asyncio
async def test_never_yielding_stream_is_bounded_admission_timeout(harness):
    """A scheduler that never produces any event - not even an abort - is
    admission unavailability, cut at the first-event bound with the bounded
    retryable 503, fail-closed, NOT a silent zero-event stream held to the
    turn deadline (the original failure mode)."""
    harness.adapter.profile = _replace(harness.adapter.profile,
                                        admission_first_event_seconds=0.1)

    async def hanging_generate(tm, obj, request=None):
        await _asyncio.Event().wait()
        yield {}

    type(harness.tm).generate_request = harness.wrap(
        hanging_generate, harness.adapter.generate_around, harness.HookType.AROUND)
    started = _time.monotonic()
    response, _, _ = await _call(harness)
    elapsed = _time.monotonic() - started
    assert response.status_code == 503
    assert _json.loads(response.body)["error"]["message"] == "admission_timeout"
    assert response.headers.get("retry-after") == "5"
    # Cut at the 0.1 s admission bound, far below the 60 s turn deadline.
    assert elapsed < 5.0
    await _spin_until(lambda: harness.adapter._active == 0)


@_pytest.mark.asyncio
async def test_stall_after_first_event_uses_turn_deadline(harness):
    """Admission was demonstrated by the first native output; a stall after
    it is a generation-phase timeout (504 deadline_exceeded at the turn
    deadline), never the admission shape."""
    harness.adapter.profile = _replace(harness.adapter.profile, max_turn_seconds=0.4,
                                        deadline_margin_seconds=0.05,
                                        admission_first_event_seconds=0.1)

    async def stalling_generate(tm, obj, request=None):
        yield {"text": "first native event",
               "meta_info": {"id": obj.rid, "finish_reason": None}}
        await _asyncio.Event().wait()
        yield {}

    type(harness.tm).generate_request = harness.wrap(
        stalling_generate, harness.adapter.generate_around, harness.HookType.AROUND)
    response, _, _ = await _call(harness)
    assert response.status_code == 504
    assert _json.loads(response.body)["error"]["message"] == "deadline_exceeded"
    await _spin_until(lambda: harness.adapter._active == 0)


@_pytest.mark.asyncio
async def test_admitted_generation_continues_past_admission_bound(harness):
    """The admission bound never caps admitted generation: once the first
    native event arrives, generation may run past the first-event bound and
    still complete normally with its receipt."""
    harness.adapter.profile = _replace(harness.adapter.profile,
                                        admission_first_event_seconds=0.1)

    async def slow_after_first(tm, obj, request=None):
        yield {"text": "first native event",
               "meta_info": {"id": obj.rid, "finish_reason": None}}
        await _asyncio.sleep(0.3)  # Past the 0.1 s admission bound.
        yield {"text": "done", "meta_info": {"id": obj.rid,
               "finish_reason": {"type": "stop"}}}

    type(harness.tm).generate_request = harness.wrap(
        slow_after_first, harness.adapter.generate_around, harness.HookType.AROUND)

    async def one_frame_per_output(obj, request, raw):
        # The default fake renders a full frame set (with its own [DONE])
        # per generator output; this turn needs ONE set for two outputs.
        chat = harness.chat
        async for output in chat.tokenizer_manager.generate_request(obj, raw):
            reason = output.get("meta_info", {}).get("finish_reason")
            finish = reason.get("type") if isinstance(reason, dict) else None
            frame = "data: " + _json.dumps({"id": obj.rid, "choices": [
                {"index": 0, "delta": {}, "finish_reason": finish}]}) + "\n\n"
            chat.frames.append(frame.encode())
            yield frame
        done = "data: [DONE]\n\n"
        chat.frames.append(done.encode())
        yield done

    harness.chat._generate_chat_stream = one_frame_per_output
    response, _, _ = await _call(harness)
    assert response.status_code == 200
    assert response.headers.get(_RECEIPT_HEADER)
    await _spin_until(lambda: harness.adapter._active == 0)


@_pytest.mark.asyncio
async def test_consecutive_slow_control_calls_share_one_deadline(harness):
    """Enroll and prepare (any sequence of pre-dispatch control calls)
    draw on ONE request-scoped absolute deadline: two calls that would each
    fit a per-call bound are still cut when their TOTAL exceeds it. With the
    old per-call deadline both 0.15 s calls succeed (0.30 s); the shared
    0.20 s deadline cuts the second at ~0.20 s."""
    from occ3_sglang.attachment_contract import AttachmentError

    async def slow_control():
        await _asyncio.sleep(0.15)
        return "ok"

    ctx = NS(cancel_requested=False,
             admission_control_deadline=_time.monotonic() + 0.20)
    started = _time.monotonic()
    first = await harness.adapter._bounded_admission(slow_control(), ctx)
    assert first == "ok"  # The enroll phase fits the shared bound.
    with _pytest.raises(AttachmentError) as caught:
        await harness.adapter._bounded_admission(slow_control(), ctx)  # prepare_source
    elapsed = _time.monotonic() - started
    assert caught.value.code == "admission_timeout"
    # Cut by the shared deadline, not a fresh per-call one.
    assert elapsed < 0.29


@_pytest.mark.asyncio
async def test_hanging_readiness_is_bounded(harness):
    """A readiness refresh that never answers is inside the same
    request-scoped admission control deadline: bounded retryable 503, not
    an unbounded wait."""
    from occ3_sglang.attachment_contract import BackendReadiness

    ready = harness.backend.readiness()
    # scheduler is one of the receiver_ready conjunction fields.
    harness.backend.readiness = lambda: _replace(ready, scheduler=False)

    async def hanging_refresh():
        await _asyncio.Event().wait()

    harness.backend.refresh_readiness = hanging_refresh
    harness.adapter.profile = _replace(harness.adapter.profile,
                                        admission_control_seconds=0.1)
    response, _, _ = await _call(harness)
    assert response.status_code == 503
    assert _json.loads(response.body)["error"]["message"] == "admission_timeout"
    await _spin_until(lambda: harness.adapter._active == 0)


@_pytest.mark.asyncio
async def test_keepalive_no_native_event_emits_bounded_admission_error(harness):
    """Post-start, no native event at all: the committed stream ends with
    one bounded structured admission_timeout error event within the
    first-event bound - well before the turn deadline - instead of keepalive
    frames until the turn deadline."""
    harness.adapter.profile = _replace(harness.adapter.profile,
                                        admission_first_event_seconds=0.1,
                                        keepalive_seconds=0.02)

    async def hanging_generate(tm, obj, request=None):
        await _asyncio.Event().wait()
        yield {}

    type(harness.tm).generate_request = harness.wrap(
        hanging_generate, harness.adapter.generate_around, harness.HookType.AROUND)
    raw, _ = _raw_request(_canonical({"occ3": _attachment().wire()}))
    raw.scope["headers"].append((_KEEPALIVE_HEADER.encode(), _KEEPALIVE_MODE.encode()))
    response = await harness.chat.handle_request(_chat_request(), raw)
    events = []

    async def send(event):
        events.append(event)

    started = _time.monotonic()
    await response(raw.scope, raw.receive, send)
    elapsed = _time.monotonic() - started
    body = b"".join(event.get("body", b"") for event in events
                    if event["type"] == "http.response.body")
    # Keepalive frames flowed while admission was pending...
    assert b": occ3-keepalive" in body
    # ...then exactly one bounded terminal error event, no native bytes.
    error_frames = [line for line in body.split(b"\n") if line.startswith(b"data: ")]
    assert len(error_frames) == 1
    payload = _json.loads(error_frames[0][6:])
    assert payload["error"]["message"] == "admission_timeout"
    assert payload["error"]["code"] == 503
    assert payload["error"]["retryAfterSeconds"] == 5
    # Cut at the admission bound, far below the 60 s turn deadline.
    assert elapsed < 5.0
    await _spin_until(lambda: harness.adapter._active == 0)
