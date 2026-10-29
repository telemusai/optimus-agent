"""Spec section 13 numerical and leakage battery across both frozen methods.

Existing suites already cover part of the list (cited per item in the campaign
report); this file closes the remaining gaps for the dual ridge (method A) and
the low-rank Procrustes (method B): harder-conditioned dual/primal agreement,
row-vector orientation, SPD solves, the method-B cached-factor path, method-B
promotion honesty through the shared gate, rank deficiency for the dual ridge,
discarded components outside the supported subspace for the dual ridge,
method-B evaluation and split leakage invariants, cross-method artifact
refusal, and zero-energy handling.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

FP64_DUAL_PRIMAL_REL = 1e-10  # fp64 solves, cond(system) <= 4e4, measured ~2e-12
FP32_MODULE_REL = 1e-4        # module fp32 vs fp64 reference, well conditioned
EPS32 = float(torch.finfo(torch.float32).eps)
ORIENTATION_ATOL = 1e-5        # fp32 matmul blocking, measured <= 3.1e-7
NULL_SPACE_ATOL = 1e-5         # fp32 representation noise, measured <= 4.2e-7


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


def _provenance_b(**overrides):
    from occ3_sglang.statebridge_calibrated_procrustes import ProcrustesProvenance
    values = dict(layer_index=77, model_fingerprint="sha256:" + "a" * 64,
                  checkpoint_revision="rev-1", tokenizer_fingerprint="sha256:" + "b" * 64,
                  embedding_id="glm53-tp8-vocab",
                  capture_method_id="glm53.final-prenorm.capture.v1",
                  capture_span_policy="whole_completed_response_tail_v1")
    values.update(overrides)
    return ProcrustesProvenance(**values)


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

    def partition(self, seed=101, turn_ids=None):
        hidden, embeddings = self.rows(self.factor.shape[0], seed)
        return _partition(hidden, embeddings, turn_ids or _turn_ids())


def _relative(left, right):
    return float((left - right).norm() / right.norm().clamp_min(1e-30))


def _dual_reference(hidden, embeddings, rho, query):
    """Independent fp64 dual solve; returns predictions and the ridge value."""
    h, e = hidden.double(), embeddings.double()
    mu_h, mu_e = h.mean(0, keepdim=True), e.mean(0, keepdim=True)
    a, b = h - mu_h, e - mu_e
    gram = a @ a.mT
    lam = rho * float(gram.diagonal().mean())
    eye = torch.eye(gram.shape[0], dtype=torch.float64)
    coef = torch.linalg.solve(gram + lam * eye, b)
    return mu_e + (query.double() - mu_h) @ a.mT @ coef, lam


def _turn_corpus(count, rows_per_turn, dim, rank, seed, independent):
    from occ3_sglang.calibration_corpus import TurnBlock
    generator = _gen(seed)
    left = _orthonormal(dim, rank, generator)
    right = _orthonormal(dim, rank, generator)
    blocks = []
    for index in range(count):
        if independent:
            hidden = torch.randn(rows_per_turn, dim, generator=generator)
            embeddings = torch.randn(rows_per_turn, dim, generator=generator)
        else:
            factor = torch.randn(rows_per_turn, rank, generator=generator)
            hidden, embeddings = factor @ left.T, factor @ right.T
        blocks.append(TurnBlock(turn_id=f"turn-{index:03d}", hidden=hidden,
                                embeddings=embeddings))
    return blocks


def _split(corpus):
    from occ3_sglang.calibration_corpus import split_manifest
    return split_manifest(corpus, fit_turns=4, validation_turns=4, test_turns=4)


def _fitted_b(corpus, variant="o"):
    from occ3_sglang.statebridge_calibrated_procrustes import fit_map
    manifest = _split(corpus)
    artifact = fit_map(manifest["fit"], {"r": 16, "eta": 0.01, "variant": variant},
                       provenance=_provenance_b())
    return manifest, artifact


def test_dual_and_primal_ridge_agree_on_hard_conditioned_fixtures():
    """Item 1: the identity holds under recorded conditioning, not only on flat data.

    Singular values logspace(1, smin) with smin in {-3, -5} give cond(G) up to
    1e10; the ridge keeps cond(system) <= 3.5e4 (recorded per case).
    """
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    for seed, smin, rho in ((55, -3.0, 0.02), (56, -5.0, 0.002)):
        generator = _gen(seed)
        rows, dim = 96, 24
        u = torch.randn(rows, dim, generator=generator, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator,
                                                 dtype=torch.float64))
        scales = torch.logspace(0, smin, dim, dtype=torch.float64)
        hidden = (u * scales) @ rotation.T
        weights = torch.randn(dim, dim, generator=generator, dtype=torch.float64) / dim
        embeddings = hidden @ weights
        message = torch.randn(12, dim, generator=generator, dtype=torch.float64)
        mu_h, mu_e = hidden.mean(0, keepdim=True), embeddings.mean(0, keepdim=True)
        a, b = hidden - mu_h, embeddings - mu_e
        gram = a @ a.mT
        lam = rho * float(gram.diagonal().mean())
        primal = torch.linalg.solve(a.mT @ a + lam * torch.eye(dim, dtype=torch.float64),
                                    a.mT @ b)
        dual = a.mT @ torch.linalg.solve(
            gram + lam * torch.eye(rows, dtype=torch.float64), b)
        z_primal = mu_e + (message - mu_h) @ primal
        z_dual = mu_e + (message - mu_h) @ dual
        system = gram + lam * torch.eye(rows, dtype=torch.float64)
        eigenvalues = torch.linalg.eigvalsh(system)
        cond = float(eigenvalues.max() / eigenvalues.min())
        assert cond >= 1e3, "fixture is not actually harder conditioned"
        assert _relative(z_primal, z_dual) < FP64_DUAL_PRIMAL_REL
        # Module fp32 against the fp64 dual reference, tolerance conditioning-aware:
        # eps32 * cond is the fp32 solve bound; measured values are ~1e-5.
        artifact = fit_map(_partition(hidden.float(), embeddings.float(), _turn_ids()),
                           {"rho": rho})
        served = apply_map(artifact, message.float()).double()
        assert _relative(served, z_dual) < max(FP32_MODULE_REL, cond * EPS32)
        assert artifact.ridge == pytest.approx(lam, rel=1e-6)


def test_row_vector_orientation_and_dimensions():
    """Item 2: rows are samples; orientation misuse is refused, not reinterpreted."""
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang import statebridge_calibrated as module_a
    from occ3_sglang import statebridge_calibrated_procrustes as module_b
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    hidden, embeddings = model.rows(128, seed=201)
    query, _ = model.rows(20, seed=202)
    artifacts = {
        "a": module_a.fit_map(_partition(hidden, embeddings, _turn_ids()), {"rho": 0.02}),
        "b": module_b.fit_map(_partition(hidden, embeddings, _turn_ids()),
                              {"r": 16, "eta": 0.01, "variant": "o"})}
    apply = {"a": module_a.apply_map, "b": module_b.apply_map}
    for name, artifact in artifacts.items():
        assert artifact.hidden_dim == 64 and artifact.rows == 128
        predicted = apply[name](artifact, query)
        assert predicted.shape == (20, 64)
        # A row's prediction depends only on that row: split batches and
        # permutations must reproduce the stacked result (fp32 blocking tolerance).
        first = apply[name](artifact, query[:12])
        second = apply[name](artifact, query[12:])
        assert torch.allclose(torch.cat([first, second]), predicted, atol=ORIENTATION_ATOL)
        permutation = torch.randperm(20, generator=_gen(203))
        restored = apply[name](artifact, query[permutation])[torch.argsort(permutation)]
        assert torch.allclose(restored, predicted, atol=ORIENTATION_ATOL)
        # A transposed query is a different matrix, not the same data: refused.
        with pytest.raises(LatentError) as caught:
            apply[name](artifact, query.T)
        assert caught.value.code == "profile"
    # Fitting column-oriented data collapses below the row floor: refused.
    small = _SharedFactor(rows=64, dim=32, rank=8, seed=210)
    hidden32, embeddings32 = small.rows(64, seed=211)
    transposed = _partition(hidden32.T, embeddings32.T, _turn_ids())
    with pytest.raises(LatentError) as caught:
        module_a.fit_map(transposed, {})
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        module_b.fit_map(transposed, {"r": 16, "eta": 0.01})
    assert caught.value.code == "profile"


def test_positive_regularisation_yields_a_valid_spd_solve():
    """Item 3: the dual system is SPD by construction; B stays orthonormal."""
    from occ3_sglang import statebridge_calibrated as module_a
    from occ3_sglang import statebridge_calibrated_procrustes as module_b
    generator = _gen(301)
    hidden = torch.randn(128, 32, generator=generator)
    embeddings = hidden @ torch.randn(32, 32, generator=generator) * 0.1
    base = torch.randn(32, 16, generator=generator)
    duplicated = base.repeat(2, 1).contiguous()
    fixtures = {
        "well_conditioned": (hidden, embeddings),
        "rank_deficient": (duplicated, torch.randn(64, 16, generator=generator))}
    for (fit_hidden, fit_embeddings) in fixtures.values():
        artifact = module_a.fit_map(_partition(fit_hidden, fit_embeddings, _turn_ids(8)),
                                    {"rho": 0.02})
        assert artifact.ridge > 0.0
        system = artifact.gram + artifact.ridge * torch.eye(artifact.rows)
        eigenvalues = torch.linalg.eigvalsh(system.double())
        assert float(eigenvalues.min()) > 0.0
        torch.linalg.cholesky(system.double())
        assert 0.0 <= artifact.solve_residual < 1e-4
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    procrustes = module_b.fit_map(model.partition(seed=302),
                                  {"r": 16, "eta": 0.01, "variant": "o"},
                                  provenance=_provenance_b())
    rotation = procrustes.rotation.double()
    eigenvalues = torch.linalg.eigvalsh(rotation.T @ rotation)
    assert float(eigenvalues.min()) > 1.0 - 1e-3 and float(eigenvalues.max()) < 1.0 + 1e-3
    assert bool((procrustes.scale_hidden > 0).all())
    assert bool((procrustes.scale_embedding > 0).all())
    assert procrustes.binding.rotation_defect <= 1e-3


def test_procrustes_cached_factor_path_is_built_once_and_reused(monkeypatch):
    """Item 4 for method B: one bundle per (artifact, device); A is covered already.

    The cached path must agree with a per-call fp64 reference rebuilt from the
    artifact tensors, and a repeated apply must not rebuild the bundle.
    """
    from collections import OrderedDict
    from occ3_sglang import statebridge_calibrated_procrustes as module
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    artifact = module.fit_map(model.partition(seed=401),
                              {"r": 16, "eta": 0.01, "variant": "o"})
    monkeypatch.setattr(module, "_FACTOR_CACHE", OrderedDict())
    query, _ = model.rows(16, seed=402)
    first = module.apply_map(artifact, query)
    second = module.apply_map(artifact, query)
    third = module.apply_map(artifact, query[:8])
    assert torch.equal(first, second)
    assert len(module._FACTOR_CACHE) == 1
    key = next(iter(module._FACTOR_CACHE))
    assert key == (id(artifact), "cpu")
    # Per-call fp64 reference from the artifact tensors in the documented order.
    centred = query.double() - artifact.mean_hidden.double()
    reduced = (centred @ artifact.basis_hidden.double()) / artifact.scale_hidden.double()
    reference = artifact.mean_embedding.double() + (
        (reduced @ artifact.rotation.double()) * artifact.scale_embedding.double()
    ) @ artifact.basis_embedding.double().T
    assert _relative(first.double(), reference) < FP32_MODULE_REL
    assert torch.allclose(third, first[:8], atol=ORIENTATION_ATOL)


def test_method_b_no_signal_corpus_is_recorded_and_refused_honestly():
    """Item 6 for method B: poor quality is measured, then refused by the gate.

    The shared gate binds a method-B record through its artifact content
    digest (same discipline as method A), so poor quality is measured,
    recorded and refused without ever being truncated.
    """
    from occ3_sglang.calibration_corpus import (evaluate_map, validate_for_activation,
                                                validate_metrics_record)
    from occ3_sglang import statebridge_calibrated_procrustes as module
    null_manifest, null_artifact = _fitted_b(
        _turn_corpus(12, 16, 64, 16, seed=9001, independent=True))
    record = evaluate_map(module.apply_map, null_artifact, null_manifest["validation"])
    assert validate_metrics_record(record)
    assert record["artifact_digest"] == null_artifact.content_digest
    assert record["r2_standard"] < 0.0  # negative and recorded, never truncated
    assert validate_for_activation(null_artifact, record,
                                   {"min_r2_standard": 0.1}) == ("refuse", "min_r2_standard")
    good_manifest, good_artifact = _fitted_b(
        _turn_corpus(12, 16, 64, 16, seed=9002, independent=False))
    good_record = evaluate_map(module.apply_map, good_artifact, good_manifest["validation"])
    assert validate_metrics_record(good_record)
    assert good_record["r2_standard"] > 0.9
    assert validate_for_activation(good_artifact, good_record,
                                   {"min_r2_standard": 0.1}) == ("allow", None)
    assert null_manifest["fit"]["hidden"].shape[0] == record["rows"]


def test_method_a_recovers_a_low_rank_map_within_the_ridge_shrinkage_bound():
    """Item 7 for method A: the exact map is recovered up to the ridge shrinkage.

    The dual ridge shrinks each fit direction by lam / (lam + s^2); that bound,
    computed in fp64 from the fit spectrum, must explain the remaining error.
    """
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    hidden, embeddings = model.rows(128, seed=501)
    artifact = fit_map(_partition(hidden, embeddings, _turn_ids()), {"rho": 0.002})
    query, truth = model.rows(32, seed=502)
    predicted = apply_map(artifact, query)
    reference, lam = _dual_reference(hidden, embeddings, 0.002, query)
    assert _relative(predicted.double(), reference) < FP32_MODULE_REL
    centred = hidden.double() - hidden.double().mean(0, keepdim=True)
    spectrum = torch.linalg.eigvalsh(centred.T @ centred)
    nonzero = spectrum[spectrum > spectrum.max() * 1e-8]
    shrinkage = lam / (lam + float(nonzero.min()))
    assert _relative(predicted.double(), truth.double()) < 2.0 * shrinkage


def test_procrustes_objective_matches_scipy_across_small_fixtures():
    """Item 9: torch O(r) construction against scipy on several small fixtures.

    scipy 1.15.3 in the pinned venv. FP64 fixtures match to 1e-12; the module
    fp32 artifact is compared after rebuilding its scores in fp64 (measured
    5.5e-7 relative on rotation, 1e-4 asserted).
    """
    scipy = pytest.importorskip("scipy")
    from scipy.linalg import orthogonal_procrustes
    from occ3_sglang import statebridge_calibrated_procrustes as module
    for seed, rows, columns in ((1, 40, 6), (2, 40, 6), (3, 24, 5), (4, 24, 5)):
        generator = _gen(seed)
        left = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
        right = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
        expected, scale = orthogonal_procrustes(left.numpy(), right.numpy())
        u, singular, vh = torch.linalg.svd(left.T @ right)
        actual = u @ vh
        assert torch.allclose(torch.from_numpy(expected), actual, atol=1e-12)
        assert abs(scale - float(singular.sum())) <= 1e-12 * abs(scale)
        assert abs(abs(float(torch.det(actual))) - 1.0) < 1e-12
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    hidden, embeddings = model.rows(128, seed=601)
    for variant in ("o", "plain"):
        artifact = module.fit_map(_partition(hidden, embeddings, _turn_ids()),
                                  {"r": 16, "eta": 0.01, "variant": variant})
        a = hidden.double() - artifact.mean_hidden.double()
        b = embeddings.double() - artifact.mean_embedding.double()
        scores_a = a @ artifact.basis_hidden.double()
        scores_b = b @ artifact.basis_embedding.double()
        if variant == "o":
            scores_a = scores_a / artifact.scale_hidden.double()
            scores_b = scores_b / artifact.scale_embedding.double()
        expected, scale = orthogonal_procrustes(scores_a.numpy(), scores_b.numpy())
        assert _relative(artifact.rotation.double(),
                         torch.from_numpy(expected)) < FP32_MODULE_REL
        objective = float(torch.trace(artifact.rotation.double().T
                                      @ (scores_a.T @ scores_b)))
        assert abs(objective - scale) <= 1e-3 * abs(scale)
        if variant == "plain":
            denominator = float(scores_a.pow(2).sum())
            assert abs(artifact.a0 - scale / denominator) <= 1e-3 * abs(artifact.a0)


def test_method_a_rank_deficiency_is_solved_uniquely():
    """Item 10 for method A: duplicated rows keep the solve SPD and deterministic.

    The gram is PSD with fp32 noise in its zero directions; the ridge keeps
    the system SPD, so no null direction is completed arbitrarily and two fits
    of the same partition are identical.
    """
    from occ3_sglang import statebridge_calibrated as module
    generator = _gen(701)
    base_hidden = torch.randn(32, 16, generator=generator)
    base_embeddings = torch.randn(32, 16, generator=generator)
    hidden = base_hidden.repeat(2, 1).contiguous()
    embeddings = base_embeddings.repeat(2, 1).contiguous()
    partition = _partition(hidden, embeddings, _turn_ids(8))
    artifact = module.fit_map(partition, {"rho": 0.02})
    system = artifact.gram + artifact.ridge * torch.eye(artifact.rows)
    assert float(torch.linalg.eigvalsh(system.double()).min()) > 0.0
    query = torch.randn(8, 16, generator=generator)
    reference, _ = _dual_reference(hidden, embeddings, 0.02, query)
    assert _relative(module.apply_map(artifact, query).double(),
                     reference) < FP32_MODULE_REL
    repeated = module.fit_map(partition, {"rho": 0.02})
    assert torch.equal(artifact.gram, repeated.gram)
    assert torch.equal(module.apply_map(artifact, query), module.apply_map(repeated, query))


def test_method_a_discards_components_outside_the_fit_row_space():
    """Item 12 for method A: only projections onto the fit row space survive.

    Perturbing query rows along the null space of the fit residuals cannot
    change predictions beyond fp32 representation noise; an in-span change of
    the same size must change them. The null cut uses a 1e-5 relative
    threshold because the fp32 corpus keeps ~1e-7 relative singular noise.
    """
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    hidden, embeddings = model.rows(128, seed=801)
    artifact = fit_map(_partition(hidden, embeddings, _turn_ids()), {"rho": 0.002})
    centred = (hidden - hidden.mean(0, keepdim=True)).double()
    _, singular, vh = torch.linalg.svd(centred, full_matrices=True)
    keep = int((singular > singular[0] * 1e-5).sum())
    null_space = vh[keep:].T.contiguous()
    assert keep == 16 and null_space.shape == (64, 48)
    query, _ = model.rows(24, seed=802)
    predicted = apply_map(artifact, query)
    generator = _gen(803)
    coefficients = torch.randn(24, null_space.shape[1], generator=generator)
    for alpha in (1.0, 8.0):
        perturbed = query + (alpha * (coefficients.double() @ null_space.T)).float()
        assert torch.allclose(apply_map(artifact, perturbed), predicted,
                              atol=NULL_SPACE_ATOL)
    assert _relative(apply_map(artifact, query + 0.5 * query).double(),
                     predicted.double()) > 0.1


def test_method_b_evaluation_cannot_mutate_the_frozen_artifact():
    """Item 13 for method B through the shared evaluation, not only apply."""
    from occ3_sglang.calibration_corpus import evaluate_map
    from occ3_sglang import statebridge_calibrated_procrustes as module
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    artifact = module.fit_map(model.partition(seed=901),
                              {"r": 16, "eta": 0.01, "variant": "o"},
                              provenance=_provenance_b())
    frozen = {name: getattr(artifact, name).clone() for name in module.TENSOR_NAMES}
    digest = module.artifact_digest(artifact)
    binding = artifact.binding
    generator = _gen(902)
    hostile = _partition(torch.randn(96, 64, generator=generator),
                         torch.randn(96, 64, generator=generator), _turn_ids(6))
    record = evaluate_map(module.apply_map, artifact, hostile)
    assert record["rows"] == 96
    for name, value in frozen.items():
        assert torch.equal(value, getattr(artifact, name))
    assert module.artifact_digest(artifact) == digest
    assert artifact.binding == binding


def test_method_b_fit_is_unaffected_by_validation_and_test_content():
    """Item 15 for method B: identical fit turns give identical frozen artifacts."""
    from occ3_sglang import statebridge_calibrated_procrustes as module
    corpus = _turn_corpus(12, 16, 64, 16, seed=1001, independent=False)
    other = _turn_corpus(12, 16, 64, 16, seed=1002, independent=False)
    first = _split(corpus)
    keep = set(first["fit"]["turn_ids"])
    replaced = [block if block.turn_id in keep else other[index]
                for index, block in enumerate(corpus)]
    second = _split(replaced)
    assert first["fit"]["manifest_digest"] == second["fit"]["manifest_digest"]
    assert (first["validation"]["manifest_digest"]
            != second["validation"]["manifest_digest"])
    assert first["test"]["manifest_digest"] != second["test"]["manifest_digest"]
    left = module.fit_map(first["fit"], {"r": 16, "eta": 0.01, "variant": "o"},
                           provenance=_provenance_b())
    right = module.fit_map(second["fit"], {"r": 16, "eta": 0.01, "variant": "o"},
                           provenance=_provenance_b())
    assert left.binding == right.binding
    assert module.artifact_digest(left) == module.artifact_digest(right)
    for name in module.TENSOR_NAMES:
        assert torch.equal(getattr(left, name), getattr(right, name))


def test_method_b_raw_and_served_metrics_stay_distinct():
    """Item 16 for method B: the shared evaluation keeps raw and served apart."""
    from occ3_sglang.calibration_corpus import evaluate_map, validate_metrics_record
    from occ3_sglang import statebridge_calibrated_procrustes as module
    manifest, artifact = _fitted_b(
        _turn_corpus(12, 16, 64, 16, seed=1101, independent=False))
    raw = evaluate_map(module.apply_map, artifact, manifest["validation"])
    assert validate_metrics_record(raw)
    assert raw["served_r2_standard"] == raw["r2_standard"]
    assert raw["served_row_norm_mean"] == raw["raw_row_norm_mean"]
    nu = 3.0 * raw["raw_row_norm_mean"]
    served = evaluate_map(module.apply_map, artifact, manifest["validation"],
                         serving={"kind": "n1"}, nu=nu)
    assert validate_metrics_record(served)
    assert served["nu"] == nu and served["serving"] == {"kind": "n1"}
    assert served["served_row_norm_mean"] == pytest.approx(nu, rel=1e-4)
    assert served["raw_row_norm_mean"] != pytest.approx(nu, rel=1e-4)
    assert served["served_r2_standard"] != served["r2_standard"]
    assert served["served_r2_standard"] < served["r2_standard"]


def test_cross_method_artifacts_are_refused():
    """Item 18: a container of one method is never reinterpreted by the other."""
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang import statebridge_calibrated as module_a
    from occ3_sglang import statebridge_calibrated_procrustes as module_b
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    hidden, embeddings = model.rows(128, seed=1201)
    artifact_a = module_a.fit_map(_partition(hidden, embeddings, _turn_ids()), {"rho": 0.02},
                                  provenance=_provenance_a())
    artifact_b = module_b.fit_map(_partition(hidden, embeddings, _turn_ids()),
                                  {"r": 16, "eta": 0.01, "variant": "o"},
                                  provenance=_provenance_b())
    raw_a = module_a.serialize_artifact(artifact_a)
    raw_b = module_b.serialize_artifact(artifact_b)
    with pytest.raises(LatentError) as caught:
        module_b.deserialize_artifact(raw_a)
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        module_a.deserialize_artifact(raw_b)
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        module_a.deserialize_artifact(raw_a, expected_method_id=module_b.METHOD_ID)
    assert caught.value.code == "identity"
    with pytest.raises(LatentError) as caught:
        module_b.deserialize_artifact(raw_b, expected_method_id=module_a.METHOD_ID)
    assert caught.value.code == "identity"
    with pytest.raises(LatentError) as caught:
        module_b.deserialize_artifact(raw_b, expected_method_id=module_b.METHOD_ID_SO)
    assert caught.value.code == "identity"


def test_zero_energy_rows_are_refused_or_fitted_honestly_per_method():
    """Item 19: zero-energy inputs are refused; degenerate targets fit honestly.

    A (dual ridge) refuses a constant hidden block (the solve is undefined)
    and accepts a constant target (no quality gate), predicting its mean.
    B refuses a constant target as well because whitening needs its energy.
    Nonfinite rows elsewhere (NaN at A fit, Inf at B fit, nonfinite at apply)
    are covered by the existing suites; the Inf-in-embeddings A fit refusal is
    asserted here.
    """
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang import statebridge_calibrated as module_a
    from occ3_sglang import statebridge_calibrated_procrustes as module_b
    model = _SharedFactor(rows=128, dim=64, rank=16, seed=100)
    hidden, embeddings = model.rows(128, seed=1301)
    flat_hidden = torch.full((64, 32), 0.5)
    varying = torch.randn(64, 32, generator=_gen(1302))
    with pytest.raises(LatentError) as caught:
        module_a.fit_map(_partition(flat_hidden, varying, _turn_ids()), {})
    assert caught.value.code == "numerical"
    poisoned = embeddings.clone()
    poisoned[3, 3] = float("inf")
    with pytest.raises(LatentError) as caught:
        module_a.fit_map(_partition(hidden, poisoned, _turn_ids()), {"rho": 0.02})
    assert caught.value.code == "numerical"
    constant_target = torch.full((64, 32), 0.5)
    degenerate = module_a.fit_map(_partition(varying, constant_target, _turn_ids()), {})
    assert torch.equal(degenerate.mean_embedding, constant_target[0:1])
    predicted = module_a.apply_map(degenerate, varying[:8])
    assert torch.equal(predicted, predicted[:1].expand_as(predicted))
    flat_embeddings = torch.full((128, 64), 0.25)
    with pytest.raises(LatentError) as caught:
        module_b.fit_map(_partition(hidden, flat_embeddings, _turn_ids()),
                         {"r": 16, "eta": 0.01, "variant": "o"})
    assert caught.value.code == "numerical"
