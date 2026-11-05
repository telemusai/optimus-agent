"""S1/S2/M2: OCC-3 busy admission and process-lifetime bounds (offline, stdlib)."""
import asyncio
import time
from dataclasses import asdict, replace

import pytest

from occ3_sglang import latent_backend as lb
from occ3_sglang import latent_worker as w
from occ3_sglang.attachment_contract import AttachmentError, CaptureIntent, canonical
from occ3_sglang.latent_registry import LatentRegistry
from occ3_sglang.latent_types import LatentError, SessionEnrollment, control_request
from test_latent_types import capture, profile
from test_latent_registry import publication, consume
from test_chat_attachment import attachment, chat_request, harness, raw_request  # noqa: F401
from test_chat_source_capture import intent, source_call, source_harness  # noqa: F401


def registry(grace=0):
    clock = [100.0]
    r = LatentRegistry(profile(), epoch=lambda: clock[0], monotonic=lambda: 100.0,
                       prune_grace_seconds=grace)
    for session in ("source", "receiver"):
        r.enroll(SessionEnrollment("a" * 64, session, "model-hash", "epoch", 9, "family", 3100))
    return r, clock


def reserve(r, n, expiry=400):
    return r.reserve_capture(capture(attempt_id=f"attempt-{n}", reservation_id=f"res-{n}",
                                     capture_id=f"capture-{n}"),
                             f"handle-{n}", n, expiry, 128 * 1024 * 1024, 4096)


# ---------------------------------------------------------------- S1 registry
def test_active_turn_decline_is_busy_and_mutation_free():
    r, _ = registry()
    reserve(r, 1)
    before = (dict(r.records), dict(r.attempts), r.charged_bytes)
    with pytest.raises(LatentError) as caught:
        reserve(r, 2)
    assert caught.value.code == "busy"
    assert (dict(r.records), dict(r.attempts), r.charged_bytes) == before


@pytest.mark.parametrize("state", ["reserved", "capturing", "leased", "injected"])
def test_every_live_state_blocks_with_busy(state):
    r, _ = registry()
    reserve(r, 1).state = state
    with pytest.raises(LatentError, match="busy"):
        reserve(r, 2)


def test_other_quota_kinds_stay_quota():
    r, _ = registry()
    with pytest.raises(LatentError) as caught:
        r.reserve_capture(capture(), "h", 1, 400, 128 * 1024 * 1024, 2**62)
    assert caught.value.code == "quota"


# ------------------------------------------------------------- S1 backend map
class _Client:
    def __init__(self, error=None):
        self.error, self.calls = error, []
        self.tm = None

    async def call(self, operation, payload, command_id=None):
        self.calls.append(operation)
        if operation == "health":
            return dict(_HEALTH)
        if operation == "reserve_capture" and self.error:
            raise LatentError(self.error)
        return {"state": "retired", "revoked": True, "gpu_freed": True}


class _Authority:
    def check_session(self, principal, session):
        return None


def _backend(error):
    p = profile()
    global _HEALTH
    _HEALTH = dict(runtime_epoch=p.runtime_epoch, runtime_generation=p.runtime_generation,
                   model_fingerprint=p.model_fingerprint, method_id=p.method,
                   hidden_dim=p.hidden_dim, layer_index=p.layer_index,
                   ready_ranks=p.expected_ranks, expected_ranks=p.expected_ranks,
                   profile_digest=p.binding_digest, source_capture=True,
                   continuous_prefix=True, terminal_journal=True,
                   tp_layout_digest="b" * 64, dtype="bfloat16")
    return lb.LatentBackend(_Client(error), p, _Authority())


def _identity(p):
    from occ3_sglang.attachment_contract import AuthenticatedContext
    return AuthenticatedContext("a" * 64, "source-1", "glm", p.model_fingerprint,
                                p.runtime_epoch, p.runtime_generation, time.monotonic() + 60)


def _intent(p, n=1):
    return intent(capture_id=f"capture-{n}", model_fingerprint=p.model_fingerprint,
                  runtime_id=p.runtime_epoch, runtime_generation=p.runtime_generation,
                  method=p.method)


@pytest.mark.asyncio
async def test_backend_maps_registry_busy_to_scheduler_busy_503():
    b = _backend("busy")
    p = b.profile
    with pytest.raises(AttachmentError) as caught:
        await b.enroll_source(_intent(p), _identity(p), "attempt-1", "rid-1", time.monotonic() + 30)
    assert (caught.value.code, caught.value.status) == ("scheduler_busy", 503)
    assert b.attempts["attempt-1"]["retired"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["quota", "replay", "identity", "expired"])
async def test_backend_keeps_other_latent_errors(code):
    b = _backend(code)
    p = b.profile
    with pytest.raises(LatentError) as caught:
        await b.enroll_source(_intent(p), _identity(p), "attempt-1", "rid-1", time.monotonic() + 30)
    assert caught.value.code == code


# ---------------------------------------------------- S1 public HTTP contract
@pytest.mark.asyncio
async def test_active_occ3_turn_refusal_is_public_busy_contract(source_harness):
    h = source_harness
    original = h.backend.enroll_source
    async def busy(*args):
        await original(*args)
        raise AttachmentError("scheduler_busy", 503)
    h.backend.enroll_source = busy
    response, _, _ = await source_call(h)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert 1 <= int(response.headers["retry-after"]) <= 30
    assert b'"message":"scheduler_busy"' in response.body
    assert not h.tm.calls                      # never dispatched
    assert len(h.backend.retires) == 1         # retired before the reply
    assert h.adapter._active == 0 and not h.adapter._quarantined
    assert "x-optimus-occ3-source-receipt" not in {k.lower() for k in response.headers}
    # A fresh arm is not blocked by leftover adapter state.
    h.backend.enroll_source = original
    again, _, _ = await source_call(h, intent(capture_id="fresh"))
    assert again.status_code == 200


# ------------------------------------------------------------------- S2
def test_registry_survives_more_than_max_records_arm_retire_cycles():
    r, clock = registry(grace=0)
    for n in range(3 * r.profile.max_records):
        reserve(r, n, expiry=clock[0] + 300)
        r.retire(attempt_id=f"attempt-{n}")
        clock[0] += 1
    assert len(r.records) <= 2
    # Replay of a pruned attempt or handle stays refused while still valid.
    with pytest.raises(LatentError, match="replay"):
        reserve(r, 3 * r.profile.max_records - 2, expiry=clock[0] + 300)
    snap = r.retire(attempt_id=f"attempt-{3 * r.profile.max_records - 2}")
    assert snap["state"] == "retired" and snap["revoked"] and snap["gpu_freed"]
    # Tombstones are bounded by the records' own expiry.
    clock[0] += 400
    r._prune()
    assert not r.tombstone_attempts and not r.tombstone_handles


def test_prune_never_drops_live_or_unfreed_records():
    class Event:
        def query(self): return False
    r, clock = registry(grace=0)
    live = reserve(r, 1)
    live.state = "ready"
    r._prune()
    assert "handle-1" in r.records
    live.events.append(Event())
    r.retire(handle="handle-1")
    clock[0] += 10
    r._prune()
    assert "handle-1" in r.records and not r.records["handle-1"].gpu_freed


def test_prune_respects_grace():
    r, clock = registry(grace=60)
    reserve(r, 1); r.retire(handle="handle-1")
    clock[0] += 30; r._prune()
    assert "handle-1" in r.records
    clock[0] += 31; r._prune()
    assert "handle-1" not in r.records and "handle-1" in r.tombstone_handles


def test_published_and_consumed_record_keeps_replay_protection_after_prune():
    from test_latent_registry import ready, registry as base_registry
    r, clock = base_registry()
    r.prune_grace_seconds = 0
    ready(r); r.publish(publication()); r.reserve_consume(consume())
    r.retire(handle="handle")
    r._prune()
    assert "handle" not in r.records
    with pytest.raises(LatentError, match="replay"):
        r.publish(publication())
    with pytest.raises(LatentError):
        r.reserve_consume(consume())
    with pytest.raises(LatentError):
        r.reserve_consume(consume(attempt_id="attempt"))
    # The pruned source attempt id can never be re-reserved while valid.
    with pytest.raises(LatentError, match="replay"):
        r.reserve_capture(capture(), "new-handle", 1, 400, 128 * 1024 * 1024, 4096)


def test_worker_control_survives_many_arm_retire_cycles():
    r, clock = registry(grace=0)
    runtime = object.__new__(w.WorkerRuntime)
    runtime.profile = profile(); runtime.registry = r; runtime.commands = {}
    runtime.poisoned = False; runtime.streams = {}; runtime.calibration_capture = None
    runtime._agreed_epoch = clock[0]
    runtime.gather = lambda value: [value] * 8
    runtime.consensus = lambda value: value
    runtime.fence = lambda: None
    runtime.capture_budget = lambda count, policy=None: 4096
    n_cycles = 20 * r.profile.max_records
    for n in range(n_cycles):
        d = capture(attempt_id=f"attempt-{n}", reservation_id=f"res-{n}", capture_id=f"capture-{n}")
        reply = runtime.control(control_request(f"reserve-{n}", "reserve_capture", {
            "descriptor": asdict(d), "source_handle": f"handle-{n}", "state_generation": n,
            "expires_at_epoch": clock[0] + 300}))
        assert reply["ok"], (n, reply["error"])
        reply = runtime.control(control_request(f"retire-{n}", "retire",
                                {"attempt_id": f"attempt-{n}", "principal_digest": "a" * 64}))
        assert reply["ok"] and reply["result"]["revoked"]
        clock[0] += 0.5
        runtime._agreed_epoch = clock[0]
    # 5120 cycles = 10240 commands, far past the max_records * 16 = 4096 cap.
    assert len(runtime.commands) <= 2 * (2 * w.COMMAND_CACHE_SECONDS + 2)


@pytest.mark.asyncio
async def test_backend_attempt_table_survives_more_than_max_records_turns():
    b = _backend(None)
    p = b.profile
    for n in range(3 * p.max_records):
        await b.enroll_source(_intent(p, n), _identity(p), f"attempt-{n}", f"rid-{n}",
                              time.monotonic() + 30)
        await b.retire(f"attempt-{n}", f"rid-{n}", False)
    assert len(b.attempts) <= 2


# ------------------------------------------------------------------- M2
@pytest.mark.asyncio
async def test_adapter_concurrency_refusal_is_busy_contract(source_harness):
    h = source_harness
    h.adapter.profile = replace(h.adapter.profile, max_active=1)
    gate = asyncio.Event()
    h.backend.enroll_gate = gate
    raw, queue = __import__("test_chat_source_capture").source_raw()
    first = asyncio.create_task(h.chat.handle_request(chat_request(), raw))
    while not h.backend.enrollments:
        await asyncio.sleep(0)
    second, _, _ = await source_call(h, intent(capture_id="other"))
    assert second.status_code == 503 and second.headers["retry-after"] == "5"
    assert b'"message":"scheduler_busy"' in second.body
    assert len(h.backend.enrollments) == 1      # second never reached the backend
    gate.set()
    await first
