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
