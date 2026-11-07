"""Tiny CPU rows verify label blending and root-before-broadcast ordering."""
from types import SimpleNamespace as NS

import pytest
from occ3_sglang import latent_worker as w
from occ3_sglang.statebridge import anchor_to_labels
from occ3_sglang.latent_types import LatentError
from test_latent_types import profile, capture


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.7, 1.0])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_exact_blend(alpha, dtype):
    torch = pytest.importorskip("torch")
    prefix = torch.tensor([[1.234, -2.345], [-3.456, 4.567]], dtype=getattr(torch, dtype))
    embedding = torch.tensor([[5., 6.], [7., 8.]], dtype=prefix.dtype)
    before = prefix.clone()
    actual = anchor_to_labels(prefix, embedding, alpha=alpha)
    if alpha == 0:
        assert actual is prefix
    expected = ((1-alpha)*prefix.float()+alpha*embedding.float()).to(prefix.dtype)
    assert torch.equal(actual, expected)
    assert torch.equal(prefix, before)
    if alpha == 1:
        assert torch.equal(actual, embedding)


@pytest.mark.parametrize("alpha", [True, False, "0.5", None, -0.3, 0.2, 1.1, float("nan"), float("inf")])
def test_invalid_blend_alpha(alpha):
    with pytest.raises(LatentError):
        anchor_to_labels(None, None, alpha=alpha)


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.7, 1.0])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_worker_blends_on_root_before_broadcast(monkeypatch, alpha, dtype):
    torch = pytest.importorskip("torch")
    hidden = torch.arange(16, dtype=getattr(torch, dtype)).reshape(2, 8)
    embedded = hidden + 8
    aligned = hidden.float() * .25123
    runtime = object.__new__(w.WorkerRuntime)
    runtime.torch, runtime.device, runtime.dtype = torch, "cpu", dtype
    runtime.profile, runtime.rank = profile(
        label_anchor_alpha_allowed=(0.0, 0.3, 0.5, 0.7, 1.0),
        label_anchor_alpha_default=0.0), 0
    runtime.group = NS(ranks=[0], device_group=None)
    runtime.target_norm = None
    runtime.embedding = lambda tokens: embedded
    runtime.aligner = lambda *args: (aligned, None)
    runtime.calibration_digest, runtime.layout_digest = "", "layout"
    bindings = {}
    def preflight(phase, fn):
        result = fn()
        bindings[phase] = result
        return result
    runtime.preflight = preflight
    runtime.gather = lambda error: [error]
    runtime.consensus = lambda metadata: None
    runtime.event = lambda: NS(synchronize=lambda: None, query=lambda: True)
    record = NS(descriptor=capture(label_anchor_alpha=alpha), accepted_count=2,
                retained=[hidden, torch.tensor([1, 2])], events=[], live_refs=[], expires_at_epoch=123)
    seen = []
    expected = ((1-alpha)*aligned.float()+alpha*embedded.float()).to(hidden.dtype)
    def broadcast(prefix, **kwargs):
        assert torch.equal(prefix, expected)
        seen.append("broadcast")
    def nearest(prefix, *args, **kwargs):
        assert seen == ["broadcast"]
        assert torch.equal(prefix, expected)
        seen.append("nearest")
        return prefix
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(w, "anchor_tiled", nearest)
    monkeypatch.setattr(w, "_probe_alignment", lambda *args, **kwargs: None)
    monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE", raising=False)
    runtime.align(record, NS(output_ids=[1, 2]))
    assert seen == ["broadcast", "nearest"]
    assert torch.equal(record.prefix, expected)
    assert record.source_metadata["label_anchor_alpha"] == alpha
    assert bindings["align_prepare"]["label_anchor_alpha"] == alpha


def test_zero_anchor_workspace_needs_no_vocab_scan():
    from occ3_sglang.statebridge import workspace_bound
    # Label scratch fits inside the solver's existing count x hidden allowance.
    assert workspace_bound(128, 6144, 8, 256, 0) >= 4 * 4 * 128 * 6144


def test_existing_prepare_handshake_rejects_alpha_disagreement():
    runtime = object.__new__(w.WorkerRuntime)
    runtime.poisoned = False
    def disagree(value):
        peer = {**value, "binding": {"label_anchor_alpha": 0.7}}
        return [value, peer]
    runtime.gather = disagree
    with pytest.raises(LatentError) as caught:
        runtime.preflight("align_prepare", lambda: {"label_anchor_alpha": 0.3})
    assert caught.value.code == "ranks" and runtime.poisoned
