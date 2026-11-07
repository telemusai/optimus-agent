"""Payload-free failure telemetry regressions."""
import logging
from occ3_sglang.latent_types import LatentError
from occ3_sglang import turn_diagnostics as diag


def _deep_failure():
    secret = "private-rid-handle-principal-prompt"
    raise LatentError("not_ready")


def test_failure_origin_is_bounded_and_payload_free(caplog):
    with caplog.at_level(logging.ERROR):
        try:
            _deep_failure()
        except LatentError as error:
            diag.log_failure(error, phase="admission")
    assert '"code":"not_ready"' in caplog.text
    assert '"function":"_deep_failure"' in caplog.text
    assert '"module":"test_parallel_handoff_diagnostics.py"' in caplog.text
    assert "private-rid-handle-principal-prompt" not in caplog.text
    assert len(caplog.text) < 16384


def test_unknown_error_payload_is_not_logged(caplog):
    with caplog.at_level(logging.ERROR):
        diag.log_failure(ValueError("private-rid-handle-principal-prompt"))
    assert '"code":"unknown"' in caplog.text
    assert "private-rid" not in caplog.text


def test_failure_snapshot_saturates(monkeypatch):
    monkeypatch.setitem(diag._FAILURE_STATE, "error_count", 2147483647)
    diag.log_failure(LatentError("busy"))
    result = diag.failure_snapshot()
    assert result["error_count"] == 2147483647
    assert result["last_error_code"] == "busy"
    assert 0 <= result["first_failure_unix_seconds"] <= 9999999999
    assert result["last_error_count"] >= 1
    assert len(result) == 4


def test_logging_failure_does_not_escape(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("logger failed")
    monkeypatch.setattr(diag.LOG, "error", broken)
    diag.log_failure(LatentError("invalid"))


def test_parallel_admission_and_error_release(monkeypatch, caplog):
    """Both threads reach admission; one owns the slot while the other declines."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from dataclasses import replace
    from test_latent_registry import registry
    from test_latent_types import capture
    from occ3_sglang.latent_types import admission_diagnostics_snapshot
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    r, _ = registry()
    barrier = Barrier(2)
    outcomes = []
    def contender(number):
        barrier.wait(timeout=5)
        descriptor = replace(capture(), attempt_id="attempt" + str(number))
        try:
            r.reserve_capture(descriptor, "private-handle" + str(number), 123, 400,
                              128*1024*1024, 4096)
            return descriptor.attempt_id
        except LatentError as error:
            assert error.code == "busy"
            return None
    with caplog.at_level(logging.INFO):
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(contender, range(2)))
        accepted = [value for value in outcomes if value]
        assert len(accepted) == 1
        # Simulate the successful admission followed by a native error cleanup.
        try:
            raise LatentError("invalid")
        except LatentError:
            r.retire(attempt_id=accepted[0])
    rows = [json.loads(record.message.split("admission ", 1)[1])
            for record in caplog.records if "occ3-bridge: admission " in record.message]
    assert any(row.get("code") == "busy" and row["active"] == 1 for row in rows)
    assert any(row["operation"] == "retire" and row["outcome"] == "exit"
               and row["active"] == 0 for row in rows)
    snapshot = admission_diagnostics_snapshot()
    assert snapshot["last"]["retire"]["active"] == 0
    assert "private-handle" not in caplog.text
    assert len(json.dumps(snapshot)) < 16384
    assert all(len(record.message) < 16384 for record in caplog.records)


def test_frontend_charge_and_release(monkeypatch, caplog):
    from types import SimpleNamespace
    from occ3_sglang.chat_adapter import ChatAttachmentAdapter
    from occ3_sglang.attachment_contract import AttachmentError
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    adapter = object.__new__(ChatAttachmentAdapter)
    adapter.profile = SimpleNamespace(max_active=1, max_buffer_bytes=16, total_working_bytes=64)
    adapter._active = adapter._terminal_active = 0
    adapter._terminal_attempts = set()
    first = SimpleNamespace(charged=False)
    second = SimpleNamespace(charged=False)
    with caplog.at_level(logging.INFO):
        adapter._charge(first)
        try:
            adapter._charge(second)
            assert False, "second admission must decline"
        except AttachmentError as error:
            assert error.code == "scheduler_busy"
        adapter._release(second)
        assert adapter._active == 1
        adapter._release(first)
        adapter._release(first)
    assert adapter._active == 0
    assert '"operation":"frontend_release","outcome":"exit","active":0' in caplog.text


def test_lock_trace_fast_path_and_opt_in(monkeypatch, caplog):
    from occ3_sglang import latent_types
    from test_latent_registry import registry
    r, _ = registry()
    monkeypatch.setattr(latent_types, "_ADMISSION_LOCK_TRACE", False)
    with caplog.at_level(logging.DEBUG):
        assert latent_types.admission_lock(r) is r.lock
    monkeypatch.setattr(latent_types, "_ADMISSION_LOCK_TRACE", True)
    with caplog.at_level(logging.INFO):
        assert latent_types.admission_lock(r) is r.lock
    with caplog.at_level(logging.DEBUG):
        with latent_types.admission_lock(r):
            pass
    assert [record.message for record in caplog.records] == [
        "occ3-bridge: admission lock " + event
        for event in ("acquire_wait", "acquired", "releasing", "released")]


def test_real_backend_control_backlog(monkeypatch, caplog):
    import json
    from types import SimpleNamespace
    from occ3_sglang.latent_backend import LatentBackend, ControlClient
    from occ3_sglang.latent_types import admission_event
    from test_latent_types import profile
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    client = ControlClient(SimpleNamespace(), max_pending=3)
    client.pending["private-request"] = object()
    client._uncertain["private-request"] = object()
    client._operations["private-request"] = object()
    backend = LatentBackend(client, profile(), None)
    with caplog.at_level(logging.INFO):
        admission_event("enroll_source", "entry", owner=backend)
    row = json.loads(caplog.records[-1].message.split("admission ")[1])
    assert (row["pending"], row["uncertain"], row["operations"], row["max_pending"]) == (1, 1, 1, 3)
    assert "private-request" not in caplog.text


def _retention_adapter():
    from types import SimpleNamespace
    from occ3_sglang.chat_adapter import ChatAttachmentAdapter
    adapter = object.__new__(ChatAttachmentAdapter)
    adapter.profile = SimpleNamespace(cleanup_seconds=0.001, max_active=1)
    adapter._active = adapter._terminal_active = 1
    adapter._quarantined = set()
    adapter._terminal_tasks = set()
    adapter._terminal_attempts = set()
    return adapter


import pytest


@pytest.mark.asyncio
async def test_cleanup_and_iterator_failures_logged(caplog):
    from types import SimpleNamespace
    adapter = _retention_adapter()
    async def failed():
        raise LatentError("not_ready")
    class Iterator:
        def aclose(self):
            raise LatentError("invalid")
    def abort(_):
        raise LatentError("busy")
    ctx = SimpleNamespace(cleanup_started=False, cancel_requested=False,
        dispatched=True, native_rid="private-rid", tm=SimpleNamespace(abort_request=abort),
        attempted=False, retired=False, response_iterator=Iterator(), outer=None, inner=None)
    with caplog.at_level(logging.ERROR):
        await adapter._bounded_cleanup(failed(), ctx)
        await adapter._cleanup(ctx)
    for code in ("not_ready", "invalid", "busy"):
        assert '"code":"' + code + '"' in caplog.text
    assert '"function":"failed"' in caplog.text
    assert '"function":"aclose"' in caplog.text
    assert "private-rid" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["timeout", "cancel", "disconnect"])
async def test_retained_ledger_and_quarantine_state(cause, monkeypatch, caplog):
    import asyncio
    import json
    from types import SimpleNamespace
    from occ3_sglang.chat_adapter import _TerminalLedger
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    adapter = _retention_adapter()
    ledger = _TerminalLedger()
    abort = ledger.register("abort")
    retire = ledger.register("retire")
    ledger.register("close")
    ledger.register("emission")
    ctx = SimpleNamespace(terminal_ledger=ledger, terminal_released=False,
        terminal_settled=True, completed=False, terminal_failure_latch=True,
        cancel_requested=True, owner_finished=False, dispatched=True)
    finish = asyncio.Event()
    async def resistant():
        try:
            await finish.wait()
        except asyncio.CancelledError:
            await finish.wait()
    task = asyncio.create_task(resistant())
    await asyncio.sleep(0)  # Schedule the controlled task; no wall-clock polling.
    with caplog.at_level(logging.INFO):
        if cause == "timeout":
            await adapter._bounded_cleanup(task_wrapper(task), ctx)
        elif cause == "cancel":
            task.cancel()
            adapter._quarantine(task, ctx)
        else:
            adapter._quarantine(task, ctx)
        ledger.resolve(abort, False)
        ledger.resolve(retire, True)
        adapter._terminal_try_release(ctx)
        assert adapter._active == 1
        finish.set()
        await task
        await asyncio.sleep(0)
    rows = [json.loads(record.message.split("retention ")[1])
            for record in caplog.records if "occ3-bridge: retention " in record.message]
    assert any(row["quarantined"] > 0 for row in rows)
    gate = next(row for row in rows if row["event"] == "terminal_gate")
    assert gate["negative"] == 1 and gate["pending"] == 2
    assert gate["abort_latched"] == 1 and gate["close_pending"] == 1
    assert gate["emission_pending"] == 1 and not gate["released"]
    assert all(len(record.message) < 16384 for record in caplog.records)


async def task_wrapper(task):
    return await task


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["parse", "terminal_admission"])
async def test_early_refusal_is_logged(boundary, monkeypatch, caplog):
    from types import SimpleNamespace
    from occ3_sglang import chat_adapter
    from occ3_sglang.attachment_contract import AttachmentError
    adapter = _retention_adapter()
    async def body():
        return b"private-prompt"
    def refuse(*args):
        raise AttachmentError("unsupported_profile", 409)
    monkeypatch.setattr(chat_adapter, "parse_opt_in", refuse if boundary == "parse" else lambda _: object())
    monkeypatch.setattr(chat_adapter, "parse_terminal_mode", lambda _: object())
    adapter._terminal_admission = refuse
    with caplog.at_level(logging.INFO):
        response = await adapter.handle_around(None, None, object(), SimpleNamespace(body=body))
    assert response.status_code == 409
    assert '"code":"unsupported_profile"' in caplog.text
    assert "private-prompt" not in caplog.text


@pytest.mark.asyncio
async def test_terminal_cleanup_sync_failures_logged(monkeypatch, caplog):
    from types import SimpleNamespace
    from occ3_sglang.chat_adapter import _TerminalLedger
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    adapter = _retention_adapter()
    class Iterator:
        def aclose(self):
            raise LatentError("invalid")
    def abort(_):
        raise LatentError("busy")
    def retire(*args):
        raise LatentError("not_ready")
    adapter.backend = SimpleNamespace(retire=retire)
    ctx = SimpleNamespace(terminal_settled=False, terminal_ledger=_TerminalLedger(),
        terminal_watchdog_task=None, completed=False, terminal_released=False,
        terminal_failure_latch=False, cancel_requested=False, cleanup_started=False,
        dispatched=True, native_rid="private-rid", tm=SimpleNamespace(abort_request=abort),
        attempted=True, attempt_id="private-attempt", retire_confirmed=False, retire_started=False,
        response_iterator=Iterator(), outer=None, inner=None)
    with caplog.at_level(logging.INFO):
        await adapter._terminal_settle_body(ctx)
    for code, function in (("busy", "abort"), ("not_ready", "retire"), ("invalid", "aclose")):
        assert '"code":"' + code + '"' in caplog.text
        assert '"function":"' + function + '"' in caplog.text
    assert all(op[1] == "latched" for op in ctx.terminal_ledger.ops.values())
    assert len(ctx.terminal_ledger.ops) == 3
    assert adapter._active == adapter._terminal_active == 1
    assert "private-rid" not in caplog.text and "private-attempt" not in caplog.text


@pytest.mark.asyncio
async def test_async_probe_one_error_and_cancel_not_counted(monkeypatch, caplog):
    import asyncio
    from occ3_sglang.latent_types import admission_probe
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    monkeypatch.setitem(diag._FAILURE_STATE, "error_count", 0)
    error = LatentError("busy")
    @admission_probe("enroll_source")
    async def failed(owner):
        raise error
    @admission_probe("enroll_source")
    async def cancelled(owner):
        raise asyncio.CancelledError()
    with caplog.at_level(logging.INFO):
        with pytest.raises(LatentError):
            await failed(object())
        # The outer adapter boundary observes exactly the same exception.
        diag.log_failure(error, phase="admission")
        with pytest.raises(asyncio.CancelledError):
            await cancelled(object())
    assert diag.failure_snapshot()["error_count"] == 1
    assert caplog.text.count("occ3-bridge: native failure ") == 1
    assert '"code":"unknown"' not in caplog.text
    assert '"outcome":"exit"' in caplog.text


def test_attachment_refusals_info_and_admission_code(monkeypatch, caplog):
    from occ3_sglang.attachment_contract import AttachmentError
    from occ3_sglang.latent_types import admission_event
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    with caplog.at_level(logging.INFO):
        error = AttachmentError("scheduler_busy", 503)
        admission_event("frontend_charge", "error", error=error)
        diag.log_failure(error)
    assert all(record.levelno == logging.INFO for record in caplog.records)
    assert '"code":"scheduler_busy"' in caplog.records[0].message


def test_worker_poison_and_unavailable_gates(monkeypatch, caplog):
    import os
    import sys
    from types import SimpleNamespace as NS
    from occ3_sglang import latent_worker as worker
    runtime = NS(poisoned=False, members=tuple([os.getpid()] * 8))
    monkeypatch.setattr(worker, "_RUNTIME", runtime)
    scheduler = NS(occ3_runtime=runtime, tp_group=NS(cpu_group=object()))
    def gather(states, value, **kwargs):
        states[:] = [dict(value) for _ in states]
    torch = NS(distributed=NS(get_world_size=lambda _: 8, all_gather_object=gather))
    monkeypatch.setitem(sys.modules, "torch", torch)
    with caplog.at_level(logging.ERROR):
        worker._poison_unless_uniform(runtime, LatentError("ranks"), "abort_fence")
        worker._poison_unless_uniform(runtime, LatentError("ranks"), "abort_fence")
        with pytest.raises(LatentError, match="unavailable"):
            worker.handle_control(scheduler, {})
        with pytest.raises(LatentError, match="unavailable"):
            worker.bootstrap_worker(scheduler)
    assert caplog.text.count("runtime poisoned") == 1
    assert '"code":"unavailable"' in caplog.text
    assert '"function":"_runtime"' in caplog.text
    assert '"function":"_bootstrap_worker"' in caplog.text


def test_worker_probes_skip_optout_and_include_rank(monkeypatch, caplog):
    from types import SimpleNamespace as NS
    from occ3_sglang import latent_worker as worker
    from test_latent_registry import registry
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    reg, _ = registry()
    runtime = NS(rank=3, poisoned=True, commands={}, registry=reg)
    monkeypatch.setattr(worker, "_RUNTIME", runtime)
    scheduler = NS(occ3_runtime=runtime)
    request = NS(occ3_worker_descriptor=None)
    with caplog.at_level(logging.INFO):
        worker.admit_request(scheduler, request, None)
        worker.abort_request(scheduler, request, "unused")
    assert not caplog.records
    with caplog.at_level(logging.INFO):
        with pytest.raises(LatentError):
            worker.admit_request(scheduler, request, object())
    assert '"rank":3' in caplog.text
    assert '"operation":"worker_admit"' in caplog.text


def test_status_block_contains_counts_and_fences(monkeypatch, caplog):
    import json
    from types import SimpleNamespace as NS
    from occ3_sglang import startup_state, latent_types
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    machine = NS(state=NS(value="NATIVE_READY"), code="ready", stage="done", attempt=1,
        experimental_selected=True, preflight_reached=True, backend_constructed=True,
        worker_readiness_reached=True, hooks_prepared=True, route_mount_attempted=True,
        rollback_completed=False)
    monkeypatch.setattr(startup_state, "current", lambda: machine)
    monkeypatch.setattr(startup_state, "_served_fingerprint", lambda: "fixed")
    monkeypatch.setattr(startup_state, "registry", lambda: NS(counts=lambda: {}))
    adapter = _retention_adapter()
    adapter._attempts = {"private-fence": object()}
    latent_types.admission_event("frontend_charge", "entry", owner=adapter)
    block = startup_state.status_payload()["parallel_handoff_diagnostics"]
    assert set(block) == {"last_error_code", "last_error_count", "error_count", "first_failure_unix_seconds", "admission"}
    assert block["admission"]["last"]["frontend_charge"]["fences"] == 1
    assert 0 <= block["error_count"] <= 2147483647
    assert len(json.dumps(block)) < 16384
    assert "private-fence" not in json.dumps(block)
    monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE")
    assert startup_state.status_payload()["parallel_handoff_diagnostics"]["admission"] == {}


@pytest.mark.asyncio
async def test_post_dispatch_probe_preserves_outer_phase(monkeypatch, caplog):
    import json
    from types import SimpleNamespace
    from occ3_sglang.latent_types import admission_probe
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    monkeypatch.setitem(diag._FAILURE_STATE, "error_count", 0)
    @admission_probe("finish_source")
    async def finish_source(owner):
        raise LatentError("not_ready")
    with caplog.at_level(logging.INFO):
        try:
            await finish_source(object())
        except LatentError as error:
            diag.log_failure(error, SimpleNamespace(dispatched=True), "terminal")
            diag.log_failure(error, SimpleNamespace(dispatched=True), "cleanup")
    rows = [json.loads(record.message.split("native failure ")[1])
            for record in caplog.records if "occ3-bridge: native failure " in record.message]
    assert len(rows) == 1
    assert rows[0]["phase"] == "terminal" and rows[0]["dispatched"] is True
    assert rows[0]["code"] == "not_ready"
    assert diag.failure_snapshot()["error_count"] == 1
