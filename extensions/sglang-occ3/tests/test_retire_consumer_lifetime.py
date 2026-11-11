"""CPU lifecycle tests: real worker/registry/backend, synthetic CUDA witnesses."""
from types import SimpleNamespace as NS

import pytest

from occ3_sglang import latent_worker as worker
from occ3_sglang.latent_backend import LatentBackend
from occ3_sglang.latent_types import LatentError, control_request
from occ3_sglang.use_receipts import NativeCompletion
from test_latent_registry import consume, publication, ready, registry
from test_latent_types import profile


def leased_runtime(monkeypatch):
    """Use the real journal and control path without allocating CUDA buffers."""
    journal, clock = registry()
    record = ready(journal)
    journal.publish(publication())
    descriptor = consume()
    journal.reserve_consume(descriptor)
    record.admitted_consume = True
    record.state = "injected"
    record.injected_positions = set(range(descriptor.expected_prefix_tokens))
    record.injection_commit_id = "injected"
    record.prefix = NS(is_cuda=True)  # Synthetic witness, not a GPU validation.
    runtime = object.__new__(worker.WorkerRuntime)
    runtime.profile = profile()
    runtime.registry = journal
    runtime.commands = {}
    runtime.poisoned = False
    runtime.streams = {}
    runtime.torch = None
    runtime.rank = 0
    runtime.gather = lambda value: [value] * runtime.profile.expected_ranks
    monkeypatch.setattr(worker, "_RUNTIME", runtime)
    return runtime, record, descriptor, clock


class WorkerClient:
    """Pass backend status calls through the real scheduler control handler."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = 0

    async def call(self, operation, payload, command_id=None):
        self.calls += 1
        reply = self.runtime.control(control_request(
            command_id or f"backend-{self.calls}", operation, payload))
        if not reply["ok"]:
            raise LatentError(reply["error"])
        return reply["result"]


def retire_request(record, descriptor, form="handle", **changes):
    if form == "handle":
        payload = {"source_handle": record.handle, "source_session_id": "source"}
    else:
        payload = {"attempt_id": record.descriptor.attempt_id if form == "source" else descriptor.attempt_id}
    payload["principal_digest"] = descriptor.principal_digest
    payload.update(changes)
    return control_request("early-retire", "retire", payload)


def commit_terminal(runtime, descriptor):
    req = NS(occ3_worker_descriptor=descriptor, rid=descriptor.native_rid,
             output_ids=[7], retraction_count=0, skip_radix_cache_insert=True,
             occ3_private_kv=True, occ3_private_prefix=True)
    worker.commit_result(NS(occ3_runtime=runtime), req, NS(), 0, 0, 1,
                         terminal=True, forward_iter=1)
    return req


def frontend(runtime, descriptor):
    backend = object.__new__(LatentBackend)
    backend.client = WorkerClient(runtime)
    reservation = NS(attempt_id=descriptor.attempt_id,
                     native_rid=descriptor.native_rid,
                     reservation_id=descriptor.reservation_id)
    backend.attempts = {descriptor.attempt_id: {
        "descriptor": descriptor, "reservation": reservation}}
    completion = NativeCompletion(descriptor.native_rid, "stop", "d" * 64, 1, True)
    return backend, reservation, completion


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["handle", "source", "consumer"])
@pytest.mark.parametrize("timing", ["before_commit", "after_commit"])
async def test_early_retire_preserves_terminal_commit_and_backend_receipt(monkeypatch, form, timing):
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    released = []
    record.occ3_content_state = NS(release=lambda: released.append(True))
    request = retire_request(record, descriptor, form)
    if timing == "after_commit":
        commit_terminal(runtime, descriptor)
    for _ in range(2):
        refused = runtime.control(request)
        assert not refused["ok"] and refused["error"] == "busy"
        assert not record.revoked and not runtime.poisoned and not released
    if timing == "before_commit":
        commit_terminal(runtime, descriptor)
    assert record.state == "terminal" and not record.revoked
    # Even a repeated source cleanup between terminal commit and proof read
    # must not destroy the journal needed by the frontend.
    assert runtime.control(request)["error"] == "busy"
    backend, reservation, completion = frontend(runtime, descriptor)
    evidence = await backend.finish(reservation, completion)
    assert evidence.injection_commit_id == "injected"
    assert record.consumer_proof_read and record.revoked and released == [True]
    assert record.gpu_freed and runtime.registry.charged_bytes == 0
    assert await backend.finish(reservation, completion) == evidence
    retired = runtime.control(request)
    assert retired["ok"] and retired["result"]["revoked"]


@pytest.mark.asyncio
async def test_finish_without_pending_retire_keeps_fast_path_and_immutable_reply(monkeypatch):
    runtime, record, descriptor, clock = leased_runtime(monkeypatch)
    commit_terminal(runtime, descriptor)
    backend, reservation, completion = frontend(runtime, descriptor)
    evidence = await backend.finish(reservation, completion)
    assert record.consumer_proof_read and not record.revoked
    assert runtime.control(retire_request(record, descriptor))["ok"]
    clock[0] += 61
    runtime.registry._prune()
    assert record.handle not in runtime.registry.records
    # A retransmitted proof-read command replays its original snapshot, not
    # the later retired record/tombstone. The outer request replay fence remains.
    assert await backend.finish(reservation, completion) == evidence


@pytest.mark.asyncio
async def test_early_finish_can_retry_same_command_only_after_real_terminal(monkeypatch):
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    backend, reservation, completion = frontend(runtime, descriptor)
    with pytest.raises(LatentError, match="not_ready"):
        await backend.finish(reservation, completion)
    assert not record.consumer_proof_read and not record.revoked
    commit_terminal(runtime, descriptor)
    assert (await backend.finish(reservation, completion)).injection_commit_id == "injected"


@pytest.mark.parametrize("payload", [
    {"attempt_id": "attempt", "principal_digest": "a" * 64},
    {"attempt_id": "consume", "principal_digest": "b" * 64},
])
def test_finish_rejects_wrong_attempt_or_principal_without_releasing_hold(monkeypatch, payload):
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    commit_terminal(runtime, descriptor)
    assert runtime.control(retire_request(record, descriptor))["error"] == "busy"
    reply = runtime.control(control_request("wrong-finish", "finish_consume", payload))
    assert reply["error"] == "identity" and not reply["ok"]
    assert not record.consumer_proof_read and not record.revoked and not runtime.poisoned


@pytest.mark.parametrize("changes", [
    {"source_session_id": "other"}, {"principal_digest": "b" * 64},
])
def test_unauthorized_retire_cannot_request_deferred_cleanup(monkeypatch, changes):
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    reply = runtime.control(retire_request(record, descriptor, **changes))
    assert reply["error"] == "identity" and not record.retire_requested
    assert not record.revoked


def test_scheduler_abort_completes_pending_retire_and_keeps_abort_latch(monkeypatch):
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    released = []
    record.occ3_content_state = NS(release=lambda: released.append(True))
    assert runtime.control(retire_request(record, descriptor))["error"] == "busy"
    req = NS(occ3_worker_descriptor=descriptor, rid=descriptor.native_rid)
    worker.abort_request(NS(occ3_runtime=runtime), req, "cancelled")
    assert record.revoked and record.gpu_freed and req.occ3_abort_done
    assert not runtime.poisoned and released == [True]
    worker.abort_request(NS(occ3_runtime=runtime), req, "cancelled")
    assert released == [True]
    assert runtime.control(control_request("after-abort", "finish_consume", {
        "attempt_id": descriptor.attempt_id, "principal_digest": descriptor.principal_digest}))["error"] == "not_ready"


def test_abandoned_consumer_still_expires_without_forging_proof(monkeypatch):
    runtime, record, descriptor, clock = leased_runtime(monkeypatch)
    assert runtime.control(retire_request(record, descriptor))["error"] == "busy"
    clock[0] = record.expires_at_epoch
    runtime.registry.reap()
    assert record.revoked and record.state == "expired" and record.gpu_freed
    assert not record.consumer_proof_read


@pytest.mark.asyncio
async def test_deferred_cleanup_preserves_event_and_inflight_memory_guards(monkeypatch):
    from test_latent_registry import Event
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    assert runtime.control(retire_request(record, descriptor))["error"] == "busy"
    commit_terminal(runtime, descriptor)
    event = Event()
    record.events.append(event)
    record.inflight = 1
    backend, reservation, completion = frontend(runtime, descriptor)
    await backend.finish(reservation, completion)
    assert record.revoked and not record.gpu_freed and record.prefix is not None
    event.done = True
    runtime.registry.reap()
    assert not record.gpu_freed
    record.inflight = 0
    runtime.registry.reap()
    assert record.gpu_freed and runtime.registry.charged_bytes == 0


def test_never_admitted_lease_retire_does_not_wait_for_impossible_proof(monkeypatch):
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    record.admitted_consume = False
    record.state = "leased"
    record.injected_positions.clear()
    record.injection_commit_id = None
    assert runtime.control(retire_request(record, descriptor))["ok"]
    assert record.revoked and not record.retire_requested


def test_rank_disagreement_never_returns_a_successful_finish_receipt(monkeypatch):
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    commit_terminal(runtime, descriptor)
    def disagree(value):
        if isinstance(value, dict) and set(value) == {"error", "result"}:
            return [value, {"error": "not_ready", "result": {}}]
        return [value] * runtime.profile.expected_ranks
    runtime.gather = disagree
    reply = runtime.control(control_request("rank-finish", "finish_consume", {
        "attempt_id": descriptor.attempt_id, "principal_digest": descriptor.principal_digest}))
    assert not reply["ok"] and reply["error"] == "ranks" and runtime.poisoned


@pytest.mark.asyncio
async def test_missing_required_pin_evidence_never_releases_proof_hold(monkeypatch):
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    runtime.dsa_pin_active = True
    runtime.sparse_index_topk = 0
    assert runtime.control(retire_request(record, descriptor))["error"] == "busy"
    with pytest.raises(LatentError, match="numerical"):
        commit_terminal(runtime, descriptor)
    assert record.state == "injected" and not record.consumer_proof_read
    backend, reservation, completion = frontend(runtime, descriptor)
    with pytest.raises(LatentError, match="not_ready"):
        await backend.finish(reservation, completion)
    assert not record.revoked


def test_finish_command_cannot_rebind_its_attempt_after_success(monkeypatch):
    runtime, record, descriptor, _ = leased_runtime(monkeypatch)
    commit_terminal(runtime, descriptor)
    payload = {"attempt_id": descriptor.attempt_id, "principal_digest": descriptor.principal_digest}
    assert runtime.control(control_request("bound-finish", "finish_consume", payload))["ok"]
    with pytest.raises(LatentError, match="replay"):
        runtime.control(control_request("bound-finish", "finish_consume", {
            **payload, "attempt_id": record.descriptor.attempt_id}))
