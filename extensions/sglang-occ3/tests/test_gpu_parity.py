"""CPU/GPU parity for both frozen calibration methods (spec section 13, item 20).

The whole module skips when CUDA is unavailable; it never fakes a GPU. On a
CUDA host the fits still run on the CPU in FP32 (the real calibration path),
the frozen artifacts move to cuda:0, and apply/serving run on the GPU. GPU
rows are compared against CPU rows with conditioning-aware tolerances: the
FP32 solve bound is eps32 * cond, so the asserted tolerance is
max(1e-4, cond * eps32) with the conditioning estimate computed in FP64 from
the fit data and recorded per fixture. Fixtures stay small (hundreds of
KB) and the allocated GPU memory is asserted under 300 MB.
"""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable on this host; CPU/GPU parity needs a CUDA device")

EPS32 = float(torch.finfo(torch.float32).eps)
WELL_CONDITIONED_REL = 1e-4
GPU_MEMORY_LIMIT_BYTES = 300 * 1024 * 1024


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _orthonormal(rows, columns, generator):
    matrix, _ = torch.linalg.qr(torch.randn(rows, columns, generator=generator))
    return matrix[:, :columns].contiguous()


def _partition(hidden, embeddings, turn_ids, manifest_digest=""):
    row_turn_ids = tuple(turn_ids[index % len(turn_ids)] for index in range(hidden.shape[0]))
    return {"hidden": hidden, "embeddings": embeddings, "row_turn_ids": row_turn_ids,
            "turn_ids": tuple(turn_ids), "manifest_digest": manifest_digest}


def _turn_ids(count=8):
    return tuple(f"turn-{index:03d}" for index in range(count))


def _provenance_a():
    from occ3_sglang.statebridge_calibrated import CalibrationProvenance
    return CalibrationProvenance(
        layer_index=77, model_fingerprint="sha256:" + "a" * 64, checkpoint_revision="rev-1",
        tokenizer_fingerprint="sha256:" + "b" * 64, embedding_id="glm53-tp8-vocab",
        capture_method_id="glm53.final-prenorm.capture.v1")


def _provenance_b():
    from occ3_sglang.statebridge_calibrated_procrustes import ProcrustesProvenance
    return ProcrustesProvenance(
        layer_index=77, model_fingerprint="sha256:" + "a" * 64, checkpoint_revision="rev-1",
        tokenizer_fingerprint="sha256:" + "b" * 64, embedding_id="glm53-tp8-vocab",
        capture_method_id="glm53.final-prenorm.capture.v1",
        capture_span_policy="whole_completed_response_tail_v1")


def _relative(left, right):
    return float((left - right).norm() / right.norm().clamp_min(1e-30))


def _tolerance(conditioning):
    """Conditioning-aware FP32 tolerance: eps32 * cond is the solve bound."""
    return max(WELL_CONDITIONED_REL, conditioning * EPS32)


def _to_device(artifact, device):
    """Move every tensor of a frozen artifact; scalars and binding stay put."""
    changes = {}
    for field in dataclasses.fields(artifact):
        value = getattr(artifact, field.name)
        if torch.is_tensor(value):
            changes[field.name] = value.to(device=device, dtype=torch.float32).contiguous()
    return dataclasses.replace(artifact, **changes)


def _dual_system_conditioning(artifact):
    """cond(G + ridge I) of the frozen dual system, computed in FP64 on CPU."""
    gram = artifact.gram.double()
    system = gram + artifact.ridge * torch.eye(gram.shape[0], dtype=torch.float64)
    eigenvalues = torch.linalg.eigvalsh(system)
    return float(eigenvalues.max() / eigenvalues.min())


def _whitening_conditioning(artifact):
    """max/min whitening scale ratio, the conditioning analog for method B."""
    scales = artifact.scale_hidden.double()
    return float(scales.max() / scales.min())


class _SharedFactor:
    """H and E pass through one shared rank-rank factor, so one exact map exists."""

    def __init__(self, rows=128, dim=64, rank=16, seed=100):
        generator = _gen(seed)
        self.factor = torch.randn(rows, rank, generator=generator)
        self.left = _orthonormal(dim, rank, generator)
        self.right = _orthonormal(dim, rank, generator)
        self.dim, self.rank = dim, rank

    def rows(self, count, seed):
        generator = _gen(seed)
        factor = torch.randn(count, self.rank, generator=generator)
        return factor @ self.left.T, factor @ self.right.T

    def partition(self, seed=101):
        hidden, embeddings = self.rows(self.factor.shape[0], seed)
        return _partition(hidden, embeddings, _turn_ids())


def _conditioned_fixture(seed, smin, rho):
    """A harder-conditioned dual ridge fixture: logspace singular directions."""
    generator = _gen(seed)
    rows, dim = 96, 24
    u = torch.randn(rows, dim, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    scales = torch.logspace(0, smin, dim)
    hidden = (u * scales) @ rotation.T
    embeddings = hidden @ (torch.randn(dim, dim, generator=generator) / dim)
    query = torch.randn(12, dim, generator=generator)
    return _partition(hidden.contiguous(), embeddings.contiguous(), _turn_ids()), rho, query


def test_dual_ridge_gpu_parity_across_conditionings():
    """Method A on cuda:0 matches the CPU artifact row for row.

    The shared-factor fixture stays well conditioned (cond ~6e2, tolerance
    1e-4); the logspace fixture reaches cond ~3.4e4 and the tolerance loosens
    to the eps32 * cond bound with the estimate recorded per fixture.
    """
    from occ3_sglang import statebridge_calibrated as module
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    shared_query, _ = model.rows(32, seed=102)
    fixtures = [(model.partition(seed=101), 0.02, shared_query),
                _conditioned_fixture(seed=55, smin=-5.0, rho=0.002)]
    for partition, rho, query in fixtures:
        artifact = module.fit_map(partition, {"rho": rho})
        conditioning = _dual_system_conditioning(artifact)
        assert conditioning > 1.0
        cpu = module.apply_map(artifact, query)
        gpu_artifact = _to_device(artifact, "cuda:0")
        assert str(gpu_artifact.gram.device) == "cuda:0"
        gpu = module.apply_map(gpu_artifact, query.to("cuda:0"))
        assert gpu.device.type == "cuda"
        tolerance = _tolerance(conditioning)
        assert _relative(gpu.cpu(), cpu) < tolerance
        nu = 2.5
        served_cpu = module.serving_transform({"kind": "n1"}, cpu, nu=nu)
        served_gpu = module.serving_transform({"kind": "n1"}, gpu, nu=nu)
        assert _relative(served_gpu.cpu(), served_cpu) < tolerance
        assert torch.allclose(served_gpu.norm(dim=-1).cpu(),
                              served_cpu.norm(dim=-1), atol=1e-4)
    assert torch.cuda.memory_allocated() < GPU_MEMORY_LIMIT_BYTES


def test_procrustes_gpu_parity_for_variants():
    """Method B on cuda:0 matches the CPU artifact row for row."""
    from occ3_sglang import statebridge_calibrated_procrustes as module
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    partition = model.partition(seed=201)
    query, _ = model.rows(32, seed=202)
    for variant in ("o", "plain"):
        artifact = module.fit_map(partition, {"r": 16, "eta": 0.01, "variant": variant})
        conditioning = _whitening_conditioning(artifact)
        cpu = module.apply_map(artifact, query)
        gpu_artifact = _to_device(artifact, "cuda:0")
        gpu = module.apply_map(gpu_artifact, query.to("cuda:0"))
        assert gpu.device.type == "cuda"
        assert _relative(gpu.cpu(), cpu) < _tolerance(conditioning)
        discard_cpu = module.residual_discard_ratio(artifact, query)
        discard_gpu = module.residual_discard_ratio(gpu_artifact, query.to("cuda:0"))
        assert abs(discard_gpu - discard_cpu) < 1e-5
        nu = 2.5
        served_cpu = module.serving_transform({"kind": "n1"}, cpu, nu=nu)
        served_gpu = module.serving_transform({"kind": "n1"}, gpu, nu=nu)
        assert _relative(served_gpu.cpu(), served_cpu) < _tolerance(conditioning)
    assert torch.cuda.memory_allocated() < GPU_MEMORY_LIMIT_BYTES


def test_gpu_cached_factor_paths_do_not_refactorise(monkeypatch):
    """Second and later applies reuse the device-resident factor, per method."""
    from collections import OrderedDict
    from occ3_sglang import statebridge_calibrated as module_a
    from occ3_sglang import statebridge_calibrated_procrustes as module_b
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    partition = model.partition(seed=301)
    query, _ = model.rows(16, seed=302)
    counts = {"cholesky": 0}
    original = torch.linalg.cholesky

    def counting(system, *args, **kwargs):
        counts["cholesky"] += 1
        return original(system, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counting)
    artifact_a = module_a.fit_map(partition, {"rho": 0.02}, provenance=_provenance_a())
    assert counts["cholesky"] == 1
    gpu_a = _to_device(artifact_a, "cuda:0")
    first_a = module_a.apply_map(gpu_a, query.to("cuda:0"))
    assert counts["cholesky"] == 2
    second_a = module_a.apply_map(gpu_a, query.to("cuda:0"))
    third_a = module_a.apply_map(gpu_a, query[:8].to("cuda:0"))
    assert counts["cholesky"] == 2  # the cuda factorisation is reused
    assert torch.equal(first_a, second_a)
    assert torch.allclose(third_a.cpu(), first_a[:8].cpu(), atol=1e-5)
    monkeypatch.setattr(module_b, "_FACTOR_CACHE", OrderedDict())
    artifact_b = module_b.fit_map(partition, {"r": 16, "eta": 0.01, "variant": "o"},
                                  provenance=_provenance_b())
    gpu_b = _to_device(artifact_b, "cuda:0")
    first_b = module_b.apply_map(gpu_b, query.to("cuda:0"))
    second_b = module_b.apply_map(gpu_b, query.to("cuda:0"))
    assert torch.equal(first_b, second_b)
    assert len(module_b._FACTOR_CACHE) == 1
    key = next(iter(module_b._FACTOR_CACHE))
    assert key == (id(gpu_b), "cuda:0")
    assert str(module_b._FACTOR_CACHE[key][1]["rotation"].device) == "cuda:0"
