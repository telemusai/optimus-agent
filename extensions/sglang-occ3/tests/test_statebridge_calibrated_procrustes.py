"""Numerical, leakage and artifact tests for the frozen low-rank Procrustes map."""
from __future__ import annotations

import dataclasses
import functools
import inspect
import json

import pytest

torch = pytest.importorskip("torch")


def _generator(seed):
    return torch.Generator().manual_seed(seed)


def _orthonormal(rows, columns, gen):
    matrix, _ = torch.linalg.qr(torch.randn(rows, columns, generator=gen))
    return matrix[:, :columns].contiguous()


def _provenance(**overrides):
    from occ3_sglang.statebridge_calibrated_procrustes import ProcrustesProvenance
    values = dict(layer_index=77, model_fingerprint="sha256:" + "a" * 64,
                  checkpoint_revision="rev-1", tokenizer_fingerprint="sha256:" + "b" * 64,
                  embedding_id="glm53-tp8-vocab",
                  capture_method_id="glm53.final-prenorm.capture.v1",
                  capture_span_policy="whole_completed_response_tail_v1")
    values.update(overrides)
    return ProcrustesProvenance(**values)


def _partition(hidden, embeddings, turn_ids, manifest_digest=""):
    row_turn_ids = tuple(turn_ids[index % len(turn_ids)] for index in range(hidden.shape[0]))
    return {"hidden": hidden, "embeddings": embeddings, "row_turn_ids": row_turn_ids,
            "turn_ids": tuple(turn_ids), "manifest_digest": manifest_digest}


def _turns(count=8):
    return tuple(f"turn-{index}" for index in range(count))


class _SharedFactor:
    """H and E pass through one shared rank-k factor, so one exact map exists."""

    def __init__(self, rows=128, dim=64, rank=16, seed=100, noise=0.0):
        gen = _generator(seed)
        self.factor = torch.randn(rows, rank, generator=gen)
        self.left = _orthonormal(dim, rank, gen)
        self.right = _orthonormal(dim, rank, gen)
        self.dim, self.rank, self.noise = dim, rank, noise

    def rows(self, count, seed):
        gen = _generator(seed)
        factor = torch.randn(count, self.rank, generator=gen)
        hidden = factor @ self.left.T
        embeddings = factor @ self.right.T
        if self.noise:
            shake = torch.randn(count, self.dim, generator=gen)
            hidden = hidden + self.noise * (shake - (shake @ self.left) @ self.left.T)
            embeddings = embeddings + self.noise * (shake - (shake @ self.right) @ self.right.T)
        return hidden, embeddings

    def off_rows(self, count, seed, scale=0.5):
        """In-span rows plus energy that lies outside span(P_H) by construction."""
        hidden, embeddings = self.rows(count, seed)
        gen = _generator(seed + 9000)
        noise = torch.randn(count, self.dim, generator=gen)
        noise = noise - (noise @ self.left) @ self.left.T
        return hidden + scale * noise, embeddings

    def partition(self, seed=101):
        hidden, embeddings = self.rows(self.factor.shape[0], seed)
        return _partition(hidden, embeddings, _turns())


def _reflection_partition(rows=128, dim=64, rank=16, seed=200):
    """The E score space is the H score space through a designed reflection.

    A = P S V_A^T and B = (P W) S V_B^T with W = diag(1, ..., 1, -1): the
    relative orientation of the two rank-r score spaces carries a reflection
    with distinct singular values, so the O(r) optimum in the fitted
    coordinates has det < 0 on the pinned torch build and any silent det = +1
    correction is strictly suboptimal and detectable in the objective.
    """
    gen = _generator(seed)
    shared = _orthonormal(rows, rank, gen)
    flip = torch.eye(rank)
    flip[-1, -1] = -1.0
    scales = torch.linspace(3.0, 1.0, rank)
    hidden_left = _orthonormal(dim, rank, gen)
    embedding_left = _orthonormal(dim, rank, gen)
    mean_hidden = torch.randn(1, dim, generator=gen)
    mean_embedding = torch.randn(1, dim, generator=gen)
    hidden = mean_hidden + shared @ torch.diag(scales) @ hidden_left.T
    embedding = mean_embedding + (shared @ flip) @ torch.diag(scales) @ embedding_left.T
    return _partition(hidden.contiguous(), embedding.contiguous(), _turns())


def _reference_fit(hidden, embeddings, rank, eta, variant="o"):
    """Independent float64 reimplementation of the spec 10.1-10.5 construction."""
    h, e = hidden.double(), embeddings.double()
    mu_h, mu_e = h.mean(0, keepdim=True), e.mean(0, keepdim=True)
    a, b = h - mu_h, e - mu_e
    _, s_h, vh_h = torch.linalg.svd(a, full_matrices=False)
    _, s_e, vh_e = torch.linalg.svd(b, full_matrices=False)
    p_h = vh_h[:rank].T.contiguous()
    p_e = vh_e[:rank].T.contiguous()
    g_h = s_h[:rank].pow(2) / h.shape[0]
    g_e = s_e[:rank].pow(2) / h.shape[0]
    d_h = (g_h + eta * g_h.mean()).sqrt()
    d_e = (g_e + eta * g_e.mean()).sqrt()
    if variant == "plain":
        x, y = a @ p_h, b @ p_e
    else:
        x, y = (a @ p_h) / d_h, (b @ p_e) / d_e
    j = x.T @ y
    u, singular, vh = torch.linalg.svd(j)
    if variant == "so":
        flip = torch.ones(rank, dtype=torch.float64)
        flip[-1] = torch.det(u @ vh)
        rotation = u @ torch.diag(flip) @ vh
    else:
        rotation = u @ vh
    a0 = 0.0
    if variant == "plain":
        a0 = float(torch.trace(rotation.T @ j) / x.pow(2).sum())
    return {"mu_h": mu_h, "mu_e": mu_e, "p_h": p_h, "p_e": p_e, "d_h": d_h, "d_e": d_e,
            "x": x, "y": y, "j": j, "rotation": rotation, "a0": a0, "singular": singular}


def _reference_apply(params, hidden, variant="o"):
    centred = hidden.double() - params["mu_h"]
    if variant == "plain":
        return params["mu_e"] + params["a0"] * (centred @ params["p_h"]
                                                @ params["rotation"] @ params["p_e"].T)
    reduced = (centred @ params["p_h"]) / params["d_h"]
    return params["mu_e"] + ((reduced @ params["rotation"]) * params["d_e"]) @ params["p_e"].T


def _scores(artifact, hidden, embeddings):
    """Whitened FIT scores rebuilt from the artifact, in its own conventions."""
    a = hidden.double() - artifact.mean_hidden.double()
    b = embeddings.double() - artifact.mean_embedding.double()
    if artifact.variant == "plain":
        return a @ artifact.basis_hidden.double(), b @ artifact.basis_embedding.double()
    return ((a @ artifact.basis_hidden.double()) / artifact.scale_hidden.double(),
            (b @ artifact.basis_embedding.double()) / artifact.scale_embedding.double())


@functools.lru_cache(maxsize=None)
def _fitted(variant="o", seed=101, rank=16, provenance=True, noise=0.0, rows=128, dim=64):
    from occ3_sglang.statebridge_calibrated_procrustes import fit_map
    model = _SharedFactor(rows=rows, dim=dim, rank=rank, seed=100, noise=noise)
    return fit_map(model.partition(seed=seed), {"r": rank, "eta": 0.01, "variant": variant},
                   provenance=_provenance() if provenance else None)


def _split(raw):
    from occ3_sglang.statebridge_calibrated_procrustes import ARTIFACT_MAGIC
    length = int.from_bytes(raw[len(ARTIFACT_MAGIC):16], "big")
    return json.loads(raw[16:16 + length].decode("ascii")), raw[16 + length:]


def _rebuild(header, payload):
    """Re-encode a header and recompute its artifact digest, as a correct writer would."""
    from occ3_sglang.statebridge_calibrated_procrustes import (ARTIFACT_MAGIC, _artifact_digest,
                                                               _canonical)
    header = dict(header)
    header["artifact_digest"] = _artifact_digest(header)
    raw_header = _canonical(header)
    return ARTIFACT_MAGIC + len(raw_header).to_bytes(8, "big") + raw_header + payload


def _raw(header, payload):
    """Encode a header verbatim, without recomputing any digest."""
    from occ3_sglang.statebridge_calibrated_procrustes import ARTIFACT_MAGIC, _canonical
    raw_header = _canonical(header)
    return ARTIFACT_MAGIC + len(raw_header).to_bytes(8, "big") + raw_header + payload


def _relative(left, right):
    return float((left - right).norm() / right.norm().clamp_min(1e-30))


def test_low_rank_known_rotation_is_recovered_in_its_supported_subspace():
    from occ3_sglang.statebridge_calibrated_procrustes import apply_map
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    artifact = _fitted()
    query, truth = model.rows(48, seed=77)
    predicted = apply_map(artifact, query)
    assert _relative(predicted, truth) < 1e-4
    fit_hidden, fit_embeddings = model.rows(128, seed=101)
    params = _reference_fit(fit_hidden, fit_embeddings, 16, 0.01)
    assert _relative(predicted.double(), _reference_apply(params, query)) < 1e-4
    single = apply_map(artifact, query[:1])
    assert single.shape == (1, 64) and bool(torch.isfinite(single).all())
    assert _relative(single, truth[:1]) < 1e-4
    assert artifact.binding.rank_hidden == 16 and artifact.binding.rank_embedding == 16


def test_reflection_fixture_keeps_the_det_negative_optimum():
    """The O(r) fit must keep a reflection; forcing det = +1 would be suboptimal."""
    from occ3_sglang.statebridge_calibrated_procrustes import fit_map
    partition = _reflection_partition()
    artifact = fit_map(partition, {"r": 16, "eta": 0.01, "variant": "o"},
                       provenance=_provenance())
    x, y = _scores(artifact, partition["hidden"], partition["embeddings"])
    u, singular, vh = torch.linalg.svd(x.T @ y)
    optimum = u @ vh
    # Deterministic on the pinned torch build: this fixture's optimum reflects.
    assert float(torch.det(optimum)) < 0
    assert torch.allclose(artifact.rotation.double(), optimum, atol=1e-4)
    assert float(torch.det(artifact.rotation.double())) < 0
    objective = ((x @ artifact.rotation.double() - y).pow(2).sum())
    lower = x.pow(2).sum() + y.pow(2).sum() - 2 * singular.sum()
    assert float(objective - lower) < 1e-3 * float(lower.clamp_min(1.0))
    flip = torch.ones(16, dtype=torch.float64)
    flip[-1] = float(torch.det(optimum))
    proper = u @ torch.diag(flip) @ vh
    proper_objective = ((x @ proper - y).pow(2).sum())
    assert float(proper_objective - objective) >= 2 * float(singular.min()) * (1 - 1e-3)
    so = fit_map(partition, {"r": 16, "eta": 0.01, "variant": "so"}, provenance=_provenance())
    x_so, y_so = _scores(so, partition["hidden"], partition["embeddings"])
    so_objective = ((x_so @ so.rotation.double() - y_so).pow(2).sum())
    assert so_objective > objective
    assert abs(float(torch.det(so.rotation.double())) - 1.0) < 1e-3


def test_objective_matches_the_fp64_reference_and_scipy():
    from occ3_sglang.statebridge_calibrated_procrustes import fit_map
    partition = _reflection_partition()
    artifact = fit_map(partition, {"r": 16, "eta": 0.01, "variant": "o"})
    params = _reference_fit(partition["hidden"], partition["embeddings"], 16, 0.01)
    trace = float(torch.trace(params["rotation"].T @ params["j"]))
    assert abs(trace - float(params["singular"].sum())) <= 1e-12 * abs(trace)
    scipy = pytest.importorskip("scipy")
    from scipy.linalg import orthogonal_procrustes
    reference_rotation, scale = orthogonal_procrustes(params["x"].numpy(), params["y"].numpy())
    assert abs(scale - trace) <= 1e-12 * abs(scale)
    assert torch.allclose(params["rotation"], torch.from_numpy(reference_rotation), atol=1e-12)
    x, y = _scores(artifact, partition["hidden"], partition["embeddings"])
    module_trace = float(torch.trace(artifact.rotation.double().T @ (x.T @ y)))
    assert abs(module_trace - scale) <= 1e-4 * abs(scale)


def test_rank_deficiency_is_refused_without_null_direction_completion():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated_procrustes import fit_map
    model = _SharedFactor(rows=128, dim=64, rank=8, seed=300)
    partition = model.partition(seed=301)
    with pytest.raises(LatentError) as caught:
        fit_map(partition, {"r": 16, "eta": 0.01, "variant": "o"})
    assert caught.value.code == "numerical"
    with pytest.raises(LatentError) as caught:
        fit_map(partition, {"r": 64, "eta": 0.01, "variant": "o"})
    assert caught.value.code == "numerical"
    small = _SharedFactor(rows=64, dim=64, rank=16, seed=310)
    with pytest.raises(LatentError) as caught:
        fit_map(small.partition(seed=311), {"r": 128, "eta": 0.01, "variant": "o"})
    assert caught.value.code == "profile"
    narrow = _SharedFactor(rows=128, dim=16, rank=16, seed=320)
    with pytest.raises(LatentError) as caught:
        fit_map(narrow.partition(seed=321), {"r": 32, "eta": 0.01, "variant": "o"})
    assert caught.value.code == "profile"
    for hyperparams in ({"r": 17, "eta": 0.01, "variant": "o"},
                        {"r": 16, "eta": 0.02, "variant": "o"},
                        {"r": 16, "eta": 0.01, "variant": "x"}):
        with pytest.raises(LatentError) as caught:
            fit_map(_SharedFactor().partition(), hyperparams)
        assert caught.value.code == "profile"
    supported = fit_map(_SharedFactor().partition(), {"r": 16, "eta": 0.01, "variant": "o"},
                        provenance=_provenance())
    assert supported.binding.rank_hidden == 16 and supported.r == 16


def test_pca_sign_flips_leave_the_prediction_unchanged():
    from occ3_sglang.statebridge_calibrated_procrustes import (artifact_digest, apply_map,
                                                               fit_map)
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100, noise=0.25)
    artifact = fit_map(model.partition(seed=401), {"r": 16, "eta": 0.01, "variant": "o"},
                       provenance=_provenance())
    query, _ = model.rows(24, seed=402)
    predicted = apply_map(artifact, query)
    gen = _generator(403)
    left = torch.diag((torch.randint(0, 2, (16,), generator=gen) * 2 - 1).float())
    right = torch.diag((torch.randint(0, 2, (16,), generator=gen) * 2 - 1).float())
    hidden_flipped = dataclasses.replace(
        artifact, basis_hidden=artifact.basis_hidden @ left, rotation=left @ artifact.rotation)
    embedding_flipped = dataclasses.replace(
        artifact, basis_embedding=artifact.basis_embedding @ right,
        rotation=artifact.rotation @ right)
    both_flipped = dataclasses.replace(
        hidden_flipped, basis_embedding=embedding_flipped.basis_embedding,
        rotation=left @ artifact.rotation @ right)
    for flipped in (hidden_flipped, embedding_flipped, both_flipped):
        moved = apply_map(flipped, query)
        assert torch.allclose(moved, predicted, atol=1e-5, rtol=1e-5)
    assert artifact_digest(hidden_flipped) != artifact_digest(artifact)


def test_residual_energy_outside_the_source_subspace_is_discarded_and_measured():
    from occ3_sglang.statebridge_calibrated_procrustes import apply_map, residual_discard_ratio
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    artifact = _fitted()
    in_span, truth = model.rows(32, seed=501)
    off_span, _ = model.off_rows(32, seed=501)
    assert residual_discard_ratio(artifact, in_span) < 1e-5
    assert residual_discard_ratio(artifact, off_span) > 0.05
    assert torch.allclose(apply_map(artifact, in_span), apply_map(artifact, off_span),
                          atol=1e-5, rtol=1e-5)
    assert _relative(apply_map(artifact, in_span), truth) < 1e-4


def test_apply_map_never_reads_reference_embeddings():
    from occ3_sglang.statebridge_calibrated_procrustes import apply_map
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    partition = model.partition(seed=601)
    artifact = _fitted(seed=601)
    query, _ = model.rows(16, seed=602)
    first = apply_map(artifact, query)
    swapped = dict(partition)
    swapped["embeddings"] = torch.randn_like(partition["embeddings"])
    assert torch.equal(apply_map(artifact, query), first)
    assert "embeddings" not in inspect.signature(apply_map).parameters
    with pytest.raises(TypeError):
        apply_map(artifact, query, embeddings=swapped["embeddings"])


def test_evaluation_rows_cannot_mutate_the_frozen_artifact():
    from occ3_sglang.statebridge_calibrated_procrustes import (TENSOR_NAMES, apply_map,
                                                               artifact_digest,
                                                               serialize_artifact)
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    artifact = _fitted(seed=701)
    before = serialize_artifact(artifact)
    digest = artifact_digest(artifact)
    frozen = {name: getattr(artifact, name).clone() for name in TENSOR_NAMES}
    query, _ = model.rows(16, seed=702)
    apply_map(artifact, query)
    apply_map(artifact, query * 50.0)
    apply_map(artifact, query[:1])
    assert serialize_artifact(artifact) == before
    assert artifact_digest(artifact) == digest
    for name in TENSOR_NAMES:
        assert torch.equal(getattr(artifact, name), frozen[name])
    assert artifact.binding == _fitted(seed=701).binding
    from occ3_sglang import statebridge_calibrated_procrustes as module
    for index in range(10):
        apply_map(_fitted(seed=703 + index), query)
    assert len(module._FACTOR_CACHE) <= module._FACTOR_CACHE_LIMIT


def test_serialization_round_trip_preserves_predictions_and_binding():
    from occ3_sglang.latent_types import METHOD, LatentError
    from occ3_sglang.statebridge_calibrated_procrustes import (METHOD_ID_SO, apply_map,
                                                               artifact_digest, describe,
                                                               deserialize_artifact, fit_map,
                                                               serialize_artifact)
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    partition = _partition(*model.rows(128, seed=801), _turns(), "c" * 64)
    artifact = fit_map(partition, {"r": 16, "eta": 0.01, "variant": "o"},
                       provenance=_provenance())
    raw = serialize_artifact(artifact)
    assert serialize_artifact(artifact) == raw
    back = deserialize_artifact(raw, expected_method_id=artifact.method_id,
                               expected_hidden_dim=artifact.hidden_dim,
                               expected_provenance=_provenance(),
                               expected_artifact_digest=artifact_digest(artifact))
    assert describe(back) == describe(artifact)
    query, _ = model.rows(12, seed=802)
    assert torch.equal(apply_map(back, query), apply_map(artifact, query))
    so = fit_map(partition, {"r": 16, "eta": 0.01, "variant": "so"}, provenance=_provenance())
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(serialize_artifact(so), expected_method_id=artifact.method_id)
    assert caught.value.code == "identity"
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(raw, expected_method_id=METHOD)
    assert caught.value.code == "identity"
    header, payload = _split(raw)
    forged = dict(header)
    forged["method_id"] = METHOD
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(forged, payload))
    assert caught.value.code == "identity"
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(raw, expected_hidden_dim=artifact.hidden_dim + 1)
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(raw, expected_artifact_digest="f" * 64)
    assert caught.value.code == "identity"
    tampered = dict(header, artifact_digest=("0" if header["artifact_digest"][0] != "0"
                                             else "1") + header["artifact_digest"][1:])
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_raw(tampered, payload))
    assert caught.value.code == "identity"
    corrupted = payload[:-1] + bytes([payload[-1] ^ 0xFF])
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(header, corrupted))
    assert caught.value.code == "identity"
    missing = dict(header)
    del missing["variant"]
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(missing, payload))
    assert caught.value.code == "profile"
    wrong_schema = dict(header, schema="optimus.occ3.calibration-artifact.v1")
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(wrong_schema, payload))
    assert caught.value.code == "profile"
    shapes = dict(header["tensor_shapes"], rotation=[17, 17])
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(dict(header, tensor_shapes=shapes), payload))
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(b"NOTPRD1!" + raw[8:])
    assert caught.value.code == "profile"
    from occ3_sglang.statebridge_calibrated_procrustes import ARTIFACT_MAGIC, _canonical
    duplicate = _canonical(header)
    duplicated = duplicate.replace(b'{"', b'{"r":16,"', 1)
    assert duplicated.count(b'"r":') == 2
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(ARTIFACT_MAGIC + len(duplicated).to_bytes(8, "big")
                             + duplicated + payload)
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        serialize_artifact(fit_map(partition, {"r": 16, "eta": 0.01, "variant": "o"}))
    assert caught.value.code == "profile"


def test_payload_is_read_without_pickle(monkeypatch):
    from occ3_sglang.statebridge_calibrated_procrustes import deserialize_artifact
    seen = {}
    original = torch.load

    def recording(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording)
    from occ3_sglang.statebridge_calibrated_procrustes import serialize_artifact
    deserialize_artifact(serialize_artifact(_fitted()))
    assert seen["weights_only"] is True and seen["map_location"] == "cpu"


def test_nonfinite_zero_energy_and_shape_corruption_are_refused():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated_procrustes import apply_map, fit_map
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    hidden, embeddings = model.rows(128, seed=901)
    broken = hidden.clone()
    broken[4, 4] = float("nan")
    with pytest.raises(LatentError) as caught:
        fit_map(_partition(broken, embeddings, _turns()), {"r": 16, "eta": 0.01})
    assert caught.value.code == "numerical"
    broken_e = embeddings.clone()
    broken_e[7, 2] = float("inf")
    with pytest.raises(LatentError) as caught:
        fit_map(_partition(hidden, broken_e, _turns()), {"r": 16, "eta": 0.01})
    assert caught.value.code == "numerical"
    flat = torch.full((128, 64), 0.5)
    with pytest.raises(LatentError) as caught:
        fit_map(_partition(flat, embeddings, _turns()), {"r": 16, "eta": 0.01})
    assert caught.value.code == "numerical"
    artifact = _fitted(seed=901)
    query, _ = model.rows(8, seed=902)
    poisoned = query.clone()
    poisoned[0, 0] = float("nan")
    with pytest.raises(LatentError) as caught:
        apply_map(artifact, poisoned)
    assert caught.value.code == "numerical"
    with pytest.raises(LatentError) as caught:
        apply_map(artifact, torch.randn(8, 32))
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        fit_map(_partition(hidden, embeddings, _turns(), "short"), {"r": 16, "eta": 0.01})
    assert caught.value.code == "profile"


def test_a_no_signal_corpus_fits_and_is_measured_poor_not_refused():
    """fit_map has no quality gate: a no-signal corpus fits and is recorded honestly."""
    from occ3_sglang.statebridge_calibrated_procrustes import apply_map, describe, fit_map
    gen = _generator(1001)
    dim, rows = 64, 128
    left = _orthonormal(dim, 32, gen)
    right = _orthonormal(dim, 32, gen)
    hidden = torch.randn(rows, 32, generator=gen) @ left.T
    embeddings = torch.randn(rows, 32, generator=gen) @ right.T
    partition = _partition(hidden, embeddings, _turns())
    artifact = fit_map(partition, {"r": 16, "eta": 0.01, "variant": "o"},
                       provenance=_provenance())
    summary = describe(artifact)
    assert summary["bound"] == 1
    assert 0.0 <= summary["rotation_defect"] <= 1e-3
    query_hidden = torch.randn(48, 32, generator=gen) @ left.T
    query_embeddings = torch.randn(48, 32, generator=gen) @ right.T
    predicted = apply_map(artifact, query_hidden)
    assert bool(torch.isfinite(predicted).all())
    residual = float((predicted - query_embeddings).pow(2).sum())
    mean_baseline = float((query_embeddings - embeddings.mean(0, keepdim=True)).pow(2).sum())
    r2 = 1.0 - residual / mean_baseline
    assert r2 < 0.10
    good_model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    good = fit_map(good_model.partition(seed=1002), {"r": 16, "eta": 0.01, "variant": "o"},
                   provenance=_provenance())
    good_query, good_truth = good_model.rows(48, seed=1003)
    good_residual = float((apply_map(good, good_query) - good_truth).pow(2).sum())
    good_baseline = float((good_truth - good_model.partition(seed=1002)["embeddings"]
                           .mean(0, keepdim=True)).pow(2).sum())
    assert 1.0 - good_residual / good_baseline > 0.90


def test_serving_transform_variants():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated_procrustes import serving_transform
    gen = _generator(1101)
    z_raw = torch.randn(12, 64, generator=gen)
    assert serving_transform(None, z_raw) is z_raw
    served = serving_transform({"kind": "n1"}, z_raw, nu=2.5)
    assert served.shape == z_raw.shape
    assert torch.allclose(served.norm(dim=-1), torch.full((12,), 2.5), atol=1e-5)
    assert not torch.allclose(served, z_raw)
    zero = z_raw.clone()
    zero[3] = 0.0
    with pytest.raises(LatentError) as caught:
        serving_transform({"kind": "n1"}, zero, nu=2.5)
    assert caught.value.code == "numerical"
    nonfinite = z_raw.clone()
    nonfinite[5, 1] = float("inf")
    with pytest.raises(LatentError) as caught:
        serving_transform({"kind": "n1"}, nonfinite, nu=2.5)
    assert caught.value.code == "numerical"
    for bad_nu in (None, 0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(LatentError) as caught:
            serving_transform({"kind": "n1"}, z_raw, nu=bad_nu)
        assert caught.value.code == "profile"
    for bad_config in ("n1", {}, {"kind": "n2"}, {"kind": "n1", "extra": 1}, 0, []):
        with pytest.raises(LatentError) as caught:
            serving_transform(bad_config, z_raw, nu=2.5)
        assert caught.value.code == "profile"


def test_numerical_rank_is_recorded_separately_from_the_requested_rank():
    from occ3_sglang.statebridge_calibrated_procrustes import describe, fit_map
    exact = fit_map(_SharedFactor(rows=128, dim=64, rank=16, seed=100).partition(),
                    {"r": 16, "eta": 0.01, "variant": "o"}, provenance=_provenance())
    assert exact.r == 16
    assert exact.binding.rank_hidden == 16
    assert exact.binding.rank_embedding == 16
    assert 1.0 <= exact.binding.participation_hidden <= 16.0 + 1e-6
    noisy = fit_map(_SharedFactor(rows=128, dim=64, rank=16, seed=100, noise=0.5)
                    .partition(seed=1201), {"r": 16, "eta": 0.01, "variant": "o"},
                    provenance=_provenance())
    assert noisy.r == 16
    assert noisy.binding.rank_hidden > 16
    assert noisy.binding.rank_embedding > 16
    assert noisy.binding.participation_hidden > 16.0
    summary = describe(noisy)
    assert summary["r"] != summary["rank_hidden"]
    assert summary["rank_hidden"] == noisy.binding.rank_hidden
    big = fit_map(_SharedFactor(rows=256, dim=160, rank=128, seed=1210).partition(seed=1211),
                  {"r": 128, "eta": 0.01, "variant": "o"}, provenance=_provenance())
    assert big.r == 128 and big.binding.rank_hidden == 128
    assert set(describe(big)) >= {"r", "rank_hidden", "rank_embedding",
                                  "participation_hidden", "participation_embedding",
                                  "whitening_gain_hidden", "whitening_gain_embedding"}


def test_each_variant_has_its_own_method_identity():
    from occ3_sglang.statebridge_calibrated_procrustes import (METHOD_ID, METHOD_ID_PLAIN,
                                                               METHOD_ID_SO, VARIANT_METHOD_IDS,
                                                               apply_map, fit_map)
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    partition = model.partition(seed=1301)
    variants = {variant: fit_map(partition, {"r": 16, "eta": 0.01, "variant": variant},
                                 provenance=_provenance()) for variant in ("o", "so", "plain")}
    assert variants["o"].method_id == METHOD_ID
    assert variants["so"].method_id == METHOD_ID_SO
    assert variants["plain"].method_id == METHOD_ID_PLAIN
    assert len({artifact.method_id for artifact in variants.values()}) == 3
    for variant, artifact in variants.items():
        assert artifact.method_id == VARIANT_METHOD_IDS[variant]
        assert artifact.binding.method_id == artifact.method_id
    assert variants["o"].a0 == 0.0 and variants["so"].a0 == 0.0
    assert 0.5 < variants["plain"].a0 < 1.5
    query, truth = model.rows(24, seed=1302)
    for variant in ("o", "plain"):
        assert _relative(apply_map(variants[variant], query), truth) < 1e-4
    assert abs(variants["so"].binding.rotation_determinant - 1.0) < 1e-3
    plain_reference = _reference_fit(partition["hidden"], partition["embeddings"], 16, 0.01,
                                    "plain")
    assert abs(plain_reference["a0"] - variants["plain"].a0) < 1e-4


def test_describe_is_scalar_only_and_never_carries_tensor_content():
    from occ3_sglang.statebridge_calibrated_procrustes import TENSOR_NAMES, describe
    bound = describe(_fitted())
    unbound = describe(_fitted(provenance=False))
    for summary in (bound, unbound):
        assert all(type(value) in (int, float, str) for value in summary.values())
        assert not any(name in summary for name in TENSOR_NAMES)
    assert bound["bound"] == 1 and unbound["bound"] == 0
    assert {"method_id", "variant", "r", "eta", "a0", "rows", "hidden_dim", "bytes",
            "rotation_defect", "rank_hidden"} <= set(bound)
    assert "corpus_digest" in bound and "corpus_digest" not in unbound
    assert len(bound["corpus_digest"]) == 64


def test_variant_artifacts_round_trip_under_their_own_identities():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated_procrustes import (VARIANT_METHOD_IDS,
                                                               artifact_digest, apply_map,
                                                               deserialize_artifact,
                                                               serialize_artifact)
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    partition = model.partition(seed=1601)
    query, truth = model.rows(20, seed=1602)
    for variant in ("o", "so", "plain"):
        artifact = _fitted(variant=variant, seed=1601)
        raw = serialize_artifact(artifact)
        back = deserialize_artifact(raw, expected_method_id=VARIANT_METHOD_IDS[variant],
                                    expected_provenance=_provenance(),
                                    expected_artifact_digest=artifact_digest(artifact))
        assert back.method_id == VARIANT_METHOD_IDS[variant]
        assert back.variant == variant and back.a0 == artifact.a0
        assert torch.equal(apply_map(back, query), apply_map(artifact, query))
        if variant in ("o", "plain"):
            # The prescribed SO construction flips one direction whenever the
            # O(r) optimum reflects, so only O(r) and the comparator carry an
            # exactness claim on this fixture.
            assert _relative(apply_map(artifact, query), truth) < 1e-4
        else:
            assert abs(artifact.binding.rotation_determinant - 1.0) < 1e-3
        others = [value for name, value in VARIANT_METHOD_IDS.items() if name != variant]
        for other in others:
            with pytest.raises(LatentError) as caught:
                deserialize_artifact(raw, expected_method_id=other)
            assert caught.value.code == "identity"


def test_hyperparameters_and_partition_shape_are_validated():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated_procrustes import (DEFAULT_HYPERPARAMETERS,
                                                               HYPERPARAMETER_FIELDS,
                                                               fit_map)
    assert HYPERPARAMETER_FIELDS == ("r", "eta", "variant")
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    partition = model.partition(seed=1401)
    for hyperparams in ({"r": 16, "eta": 0.01, "variant": "o", "extra": 1},
                       {"r": True, "eta": 0.01}, {"r": 16.0, "eta": 0.01}):
        with pytest.raises(LatentError) as caught:
            fit_map(partition, hyperparams)
        assert caught.value.code == "profile"
    broken = dict(partition)
    del broken["manifest_digest"]
    with pytest.raises(LatentError) as caught:
        fit_map(broken, {"r": 16, "eta": 0.01})
    assert caught.value.code == "profile"
    broken = dict(partition, extra=1)
    with pytest.raises(LatentError) as caught:
        fit_map(broken, {"r": 16, "eta": 0.01})
    assert caught.value.code == "profile"
    short_rows = dict(partition, row_turn_ids=partition["row_turn_ids"][:-1])
    with pytest.raises(LatentError) as caught:
        fit_map(short_rows, {"r": 16, "eta": 0.01})
    assert caught.value.code == "profile"
    duplicate = dict(partition, turn_ids=("turn-1", "turn-1"))
    with pytest.raises(LatentError) as caught:
        fit_map(duplicate, {"r": 16, "eta": 0.01})
    assert caught.value.code == "profile"
    mismatch = dict(partition, turn_ids=("turn-1",))
    with pytest.raises(LatentError) as caught:
        fit_map(mismatch, {"r": 16, "eta": 0.01})
    assert caught.value.code == "profile"
    tiny = _SharedFactor(rows=32, dim=64, rank=16, seed=1410)
    with pytest.raises(LatentError) as caught:
        fit_map(tiny.partition(seed=1411), {"r": 16, "eta": 0.01})
    assert caught.value.code == "profile"
    big = _SharedFactor(rows=256, dim=160, rank=128, seed=1420)
    defaulted = fit_map(big.partition(seed=1421), {}, provenance=_provenance())
    assert defaulted.variant == DEFAULT_HYPERPARAMETERS["variant"] == "o"
    assert defaulted.eta == DEFAULT_HYPERPARAMETERS["eta"] == 0.01
    assert defaulted.r == DEFAULT_HYPERPARAMETERS["r"] == 128


def test_required_provenance_fields_and_corpus_digest_binding():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated_procrustes import (ProcrustesArtifact,
                                                               ProcrustesBinding,
                                                               ProcrustesProvenance,
                                                               corpus_digest,
                                                               deserialize_artifact,
                                                               required_provenance_fields,
                                                               serialize_artifact)
    fields = required_provenance_fields(ProcrustesProvenance)
    for artifact_class in (ProcrustesBinding, ProcrustesArtifact):
        assert required_provenance_fields(artifact_class) == fields
    assert "capture_span_policy" in fields and "layer_index" in fields
    with pytest.raises(LatentError):
        required_provenance_fields(dict)
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    hidden, embeddings = model.rows(128, seed=1501)
    base = corpus_digest(hidden, embeddings, turn_ids=("turn-1", "turn-2"))
    assert corpus_digest(hidden.clone(), embeddings.clone(),
                         turn_ids=("turn-1", "turn-2")) == base
    assert corpus_digest(hidden, embeddings, turn_ids=("turn-1",)) != base
    assert corpus_digest(hidden, embeddings) != base
    other, _ = model.rows(128, seed=1502)
    assert corpus_digest(other, embeddings, turn_ids=("turn-1", "turn-2")) != base
    with pytest.raises(LatentError):
        corpus_digest(hidden, embeddings, turn_ids=("turn-1", "turn-1"))
    with pytest.raises(LatentError):
        corpus_digest(hidden, embeddings, turn_ids=("turn 1",))
    artifact = _fitted(seed=1501)
    raw = serialize_artifact(artifact)
    mismatches = {"layer_index": 78, "model_fingerprint": "other-value-1",
                  "checkpoint_revision": "other-value-1",
                  "tokenizer_fingerprint": "other-value-1",
                  "embedding_id": "other-value-1", "capture_method_id": "other-value-1",
                  "capture_span_policy": "other-value-1"}
    for field, value in mismatches.items():
        with pytest.raises(LatentError) as caught:
            deserialize_artifact(raw, expected_provenance=_provenance(**{field: value}))
        assert caught.value.code == "identity"
    assert deserialize_artifact(raw, expected_provenance=_provenance()) is not None


def test_content_digest_pins_promotion_evidence_like_method_a():
    import re as _re

    from occ3_sglang.calibration_corpus import (bind_validation_evidence, evaluate_map,
                                               validate_for_activation)
    from occ3_sglang.statebridge_calibrated_procrustes import (apply_map, deserialize_artifact,
                                                               serialize_artifact)
    artifact = _fitted(seed=1701)
    assert _re.fullmatch(r"[0-9a-f]{64}", artifact.content_digest)
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    hidden, embeddings = model.rows(96, seed=1702)
    validation = _partition(hidden, embeddings, _turns(6))
    record = evaluate_map(apply_map, artifact, validation)
    assert record["rows"] == 96
    assert record["artifact_digest"] == artifact.content_digest
    policy = {"min_r2_standard": 0.1, "min_turns": 1}
    assert validate_for_activation(artifact, record, policy) == ("allow", None)
    pinned = bind_validation_evidence(artifact, record, {"kind": "n1"})
    assert validate_for_activation(pinned, record, policy) == ("allow", None)
    other = _fitted(seed=1703)
    foreign = evaluate_map(apply_map, other, validation)
    assert validate_for_activation(artifact, foreign, policy) == ("refuse", "artifact_digest")
    assert validate_for_activation(pinned, foreign, policy) == ("refuse", "validation_evidence")
    round_trip = deserialize_artifact(serialize_artifact(artifact))
    assert round_trip.content_digest == artifact.content_digest
