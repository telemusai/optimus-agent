"""Engine-latch regressions: rank-divergent gather outcomes and their fix.

Reproduces the 2026-10-31 P7-pilot latch class offline (single process, no
GPU): a turn killed at a deadline boundary straddled the expiry gates
rank-to-rank, the gathered outcomes diverged, and WorkerRuntime.poisoned
latched every later native turn until restart.

Fix under test: (F1) rank-agreed clocks refreshed from rank 0 by every
gather, (F2) uniform LatentError declines never poison, (F3) the control
retire path fences before snapshotting gpu_freed.
"""
from types import SimpleNamespace as NS

import pytest

from occ3_sglang import latent_worker as w
from occ3_sglang.latent_types import LatentError


def _runtime(**over):
    base = dict(poisoned=False, _agreed_epoch=2000000000.0, _agreed_monotonic=100.0)
    base.update(over)
    rt = NS(**base)
    return rt


class _Event:
    def __init__(self, done):
        self._done = done

    def query(self):
        return self._done


def test_uniform_decline_does_not_poison_abort_fence(monkeypatch):
    """T2: a LatentError raised identically on every rank must not latch."""
    runtime = _runtime()
    original = w.WorkerRuntime.preflight

    def uniform_decline(self, phase, check, *, descriptor=None):
        raise LatentError("expired")

    runtime.preflight = lambda phase, check, descriptor=None: (_ for _ in ()).throw(LatentError("expired"))
    monkeypatch.setattr(w, "_RUNTIME", runtime)
    scheduler = NS(occ3_runtime=runtime)
    r = NS(rid="rid", occ3_worker_descriptor=NS(attempt_id="a1", native_rid="rid"), kv=NS(holds_kv=False))
    with pytest.raises(LatentError):
        w.abort_request(scheduler, r, "cancelled")
    assert runtime.poisoned is False


def test_real_disagreement_still_poisons():
    """A forced cross-rank divergence (code=ranks) must still latch."""
    runtime = _runtime()
    exc = LatentError("ranks")
    w._poison_unless_uniform(runtime, exc, "test")
    assert runtime.poisoned is True


def test_engine_fault_poisons():
    """A non-LatentError engine fault must still latch."""
    runtime = _runtime()
    w._poison_unless_uniform(runtime, RuntimeError("engine"), "test")
    assert runtime.poisoned is True


def test_gather_agrees_clocks_from_rank_zero(monkeypatch):
    """T1/F1b: rank 0's clock slot becomes the agreed clock on every rank."""
    runtime = _runtime()
    captured = {}

    class Torch:
        class distributed:
            @staticmethod
            def all_gather_object(result, value, group=None):
                captured["rank0"] = value
                # Eight ranks; only rank 0's slot must win.
                for i in range(8):
                    result[i] = {"v": value["v"], "e": 111.0, "m": 222.0}

    class Group:
        cpu_group = object()

    runtime.torch = Torch
    runtime.group = Group()
    runtime.profile = NS(expected_ranks=8)
    out = w.WorkerRuntime.gather(runtime, "payload")
    assert out == ["payload"] * 8
    assert runtime._agreed_epoch == 111.0 and runtime._agreed_monotonic == 222.0


def test_resolve_uses_agreed_clocks(monkeypatch):
    """T1/F1c: the deadline verdict reads the agreed clock, never the wall."""
    runtime = _runtime(_agreed_epoch=2000000000.0, _agreed_monotonic=100.0)
    descriptor = NS(attempt_id="a1", deadline=99.0)  # expired under the agreed clock
    record = NS(revoked=False, expires_at_epoch=2000000100.0,
                descriptor=descriptor, consumer=descriptor)
    runtime.registry = NS(source_for_attempt=lambda aid: record)
    with pytest.raises(LatentError) as err:
        w.WorkerRuntime.resolve(runtime, descriptor, check_deadline=True)
    assert err.value.code == "expired"


def test_control_retire_fences_before_snapshot(monkeypatch):
    """T3/F3: the control retire op fences before it snapshots gpu_freed."""
    calls = []
    runtime = _runtime()

    class Registry:
        def __init__(self):
            self.attempts = {}
            self.records = {"h": 1}

        def get(self, handle, allow_retired=True):
            return NS(descriptor=NS(principal_digest="p", source_session_id="s"))

        def retire(self, handle=None, attempt_id=None):
            calls.append(("retire", handle))
            return {"revoked": True}

        def reap(self):
            calls.append(("reap",))

    runtime.registry = Registry()
    runtime.fence = lambda: calls.append(("fence",))
    runtime.gather = lambda value: [value] * 8
    runtime.consensus = lambda value: None
    runtime.commands = {}
    runtime.profile = NS(max_records=16)
    envelope = {"schema": "occ3.control.v1", "command_id": "cmd-1",
                "operation": "retire",
                "payload": {"source_handle": "h", "source_session_id": "s",
                            "principal_digest": "p"}}
    monkeypatch.setattr(w, "_release_content_state", lambda record: None)
    w.WorkerRuntime.control(runtime, envelope)
    order = [c[0] for c in calls]
    assert "fence" in order and "retire" in order
    assert order.index("fence") < order.index("retire"), "retire must snapshot post-fence"
    assert runtime.poisoned is False
