"""Fit/apply/evaluate/promotion separation for the calibrated dual ridge.

Spec sections 8, 9, 11, 12, 13: fit_map fits one frozen partition with no
internal split and no quality gate; apply_map returns raw rows through a
cached dual solve; evaluate_map records the complete metric set on an
independent partition without gating; validate_for_activation is the only
promotion gate. The tests here check the separation and leakage properties;
module math and the artifact container have their own files.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _model(rank, dim, seed=1000):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rank, dim, generator=generator) / rank ** 0.5,
            torch.randn(rank, dim, generator=generator) / rank ** 0.5)


def _sample(model, rows, kappa=0.9, seed=1):
    left, right = model
    generator = torch.Generator().manual_seed(seed)
    factor = torch.randn(rows, left.shape[0], generator=generator)
    noise_h = torch.randn(rows, left.shape[1], generator=generator) * (1.0 - kappa) ** 0.5
    noise_e = torch.randn(rows, right.shape[1], generator=generator) * (1.0 - kappa) ** 0.5
    return (kappa ** 0.5 * (factor @ left) + noise_h,
            kappa ** 0.5 * (factor @ right) + noise_e)


def _turns(count, rows=40, dim=32, rank=8, kappa=0.9, seed=0, independent=False,
           family_size=0, start=0):
    from occ3_sglang.calibration_corpus import TurnBlock
    left, right = _model(rank, dim, seed=1000)
    generator = torch.Generator().manual_seed(seed)
    blocks = []
    for index in range(start, start + count):
        factor = torch.randn(rows, rank, generator=generator)
        noise_h = torch.randn(rows, dim, generator=generator)
        noise_e = torch.randn(rows, dim, generator=generator)
        if independent:
            hidden = torch.randn(rows, dim, generator=generator)
            embeddings = torch.randn(rows, dim, generator=generator)
        else:
            hidden = kappa ** 0.5 * (factor @ left) + (1.0 - kappa) ** 0.5 * noise_h
            embeddings = kappa ** 0.5 * (factor @ right) + (1.0 - kappa) ** 0.5 * noise_e
        family = None if family_size == 0 else f"fam-{index // family_size:03d}"
        blocks.append(TurnBlock(turn_id=f"turn-{index:03d}", hidden=hidden,
                                embeddings=embeddings, family=family))
    return blocks


def _manifest(count=16, rows=40, seed=0, **kwargs):
    from occ3_sglang.calibration_corpus import split_manifest
    return split_manifest(_turns(count, rows=rows, seed=seed, **kwargs),
                          fit_turns=8, validation_turns=4, test_turns=4)


def _partition(hidden, embeddings, turn_ids=("turn-000", "turn-001"), digest=""):
    rows = hidden.shape[0]
    per_turn = rows // len(turn_ids)
    assert per_turn * len(turn_ids) == rows
    row_turn_ids = tuple(turn_id for turn_id in turn_ids for _ in range(per_turn))
    return {"hidden": hidden, "embeddings": embeddings, "row_turn_ids": row_turn_ids,
            "turn_ids": tuple(turn_ids), "manifest_digest": digest}


def _provenance():
    from occ3_sglang.statebridge_calibrated import CalibrationProvenance
    return CalibrationProvenance(
        layer_index=77, model_fingerprint="sha256:" + "a" * 64, checkpoint_revision="rev-1",
        tokenizer_fingerprint="sha256:" + "b" * 64, embedding_id="glm53-tp8-vocab",
        capture_method_id="glm53.final-prenorm.capture.v1")


def test_fit_map_has_no_internal_split_and_no_quality_gate():
    """Sixty-four rows fit, and a no-relation corpus fits instead of refusing.

    The removed gate needed a spare held-out block and refused a corpus that
    did not beat the mean predictor; measurement now belongs to evaluate_map.
    """
    from occ3_sglang.statebridge_calibrated import fit_map
    hidden, embeddings = _sample(_model(8, 32), rows=64, kappa=0.9, seed=3)
    artifact = fit_map(_partition(hidden, embeddings), {"rho": 0.02})
    assert artifact.rows == 64 and artifact.ridge_rel == 0.02
    unrelated_h, _ = _sample(_model(8, 32, seed=3001), rows=64, kappa=0.75, seed=31)
    _, unrelated_e = _sample(_model(8, 32, seed=3002), rows=64, kappa=0.75, seed=32)
    noise = fit_map(_partition(unrelated_h, unrelated_e), {"rho": 0.02})
    assert noise.rows == 64


def test_a_no_signal_corpus_is_recorded_and_never_promoted():
    """Poor quality is measured, recorded and then refused by the gate only."""
    from occ3_sglang.calibration_corpus import evaluate_map, validate_for_activation
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    manifest = _manifest(independent=True, seed=11)
    artifact = fit_map(manifest["fit"], {"rho": 0.02}, provenance=_provenance())
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    assert record["r2_standard"] is None or record["r2_standard"] < 0.3
    assert validate_for_activation(artifact, record, {"min_r2_standard": 0.1}) == (
        "refuse", "min_r2_standard")
    strong = _manifest(kappa=1.0, seed=12)
    good = fit_map(strong["fit"], {"rho": 0.002}, provenance=_provenance())
    good_record = evaluate_map(apply_map, good, strong["validation"])
    assert good_record["r2_standard"] > 0.9
    assert validate_for_activation(good, good_record, {"min_r2_standard": 0.1}) == (
        "allow", None)


def test_dual_and_primal_ridge_agree_on_fp64_fixtures():
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    generator = torch.Generator().manual_seed(7)
    rows, dim, held = 64, 16, 12
    hidden = torch.randn(rows, dim, generator=generator, dtype=torch.float64)
    embeddings = hidden @ torch.randn(dim, dim, generator=generator,
                                      dtype=torch.float64) * 0.1
    message = torch.randn(held, dim, generator=generator, dtype=torch.float64)
    rho = 0.02
    mu_h = hidden.mean(0, keepdim=True)
    mu_e = embeddings.mean(0, keepdim=True)
    a, b = hidden - mu_h, embeddings - mu_e
    gram = a @ a.mT
    lam = rho * float(gram.diagonal().mean())
    eye_rows = torch.eye(rows, dtype=torch.float64)
    primal = torch.linalg.solve(a.mT @ a + lam * torch.eye(dim, dtype=torch.float64),
                                a.mT @ b)
    dual = a.mT @ torch.linalg.solve(gram + lam * eye_rows, b)
    z_primal = mu_e + (message - mu_h) @ primal
    z_dual = mu_e + (message - mu_h) @ dual
    assert float((z_primal - z_dual).norm() / z_primal.norm()) < 1e-8
    h32, e32, m32 = hidden.float(), embeddings.float(), message.float()
    mu_h32 = h32.mean(0, keepdim=True)
    mu_e32 = e32.mean(0, keepdim=True)
    a32, b32 = h32 - mu_h32, e32 - mu_e32
    product = a32 @ a32.mT
    gram32 = 0.5 * (product + product.mT)
    lam32 = rho * float(gram32.diagonal().mean())
    coef = torch.linalg.solve(gram32.double() + lam32 * eye_rows, b32.double())
    reference = (mu_e32.double() + (m32.double() - mu_h32.double())
                 @ a32.double().mT @ coef)
    artifact = fit_map(_partition(h32, e32), {"rho": rho})
    served = apply_map(artifact, m32)
    assert float((served.double() - reference).norm() / reference.norm()) < 1e-4


def test_a_known_linear_map_generalises_to_independent_rows():
    from occ3_sglang.calibration_corpus import evaluate_map
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    manifest = _manifest(kappa=1.0, rows=40, seed=17)
    artifact = fit_map(manifest["fit"], {"rho": 0.002})
    record = evaluate_map(apply_map, artifact, manifest["validation"])
    assert record["r2_standard"] > 0.99
    fit_hidden = manifest["fit"]["hidden"]
    fit_embeddings = manifest["fit"]["embeddings"]
    message = manifest["validation"]["hidden"]
    mu_h = fit_hidden.mean(0, keepdim=True)
    mu_e = fit_embeddings.mean(0, keepdim=True)
    a, b = fit_hidden - mu_h, fit_embeddings - mu_e
    product = a @ a.mT
    gram = 0.5 * (product + product.mT)
    lam = 0.002 * float(gram.diagonal().mean())
    coef = torch.linalg.solve(gram.double() + lam * torch.eye(gram.shape[0]),
                              b.double())
    reference = mu_e.double() + (message.double() - mu_h.double()) @ a.double().mT @ coef
    served = apply_map(artifact, message)
    assert float((served.double() - reference).norm() / reference.norm()) < 1e-4


def test_apply_map_caches_the_dual_solve_per_artifact(monkeypatch):
    from occ3_sglang.statebridge_calibrated import (apply_map, deserialize_artifact,
                                                    fit_map, serialize_artifact)
    manifest = _manifest(seed=19)
    counts = {"cholesky": 0}
    original = torch.linalg.cholesky

    def counting(system, *args, **kwargs):
        counts["cholesky"] += 1
        return original(system, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counting)
    artifact = fit_map(manifest["fit"], {"rho": 0.02}, provenance=_provenance())
    assert counts["cholesky"] == 1
    first = apply_map(artifact, manifest["validation"]["hidden"])
    assert counts["cholesky"] == 1
    apply_map(artifact, manifest["validation"]["hidden"][:64])
    assert counts["cholesky"] == 1
    raw = serialize_artifact(artifact)
    back = deserialize_artifact(raw)
    second = apply_map(back, manifest["validation"]["hidden"])
    assert counts["cholesky"] == 2
    apply_map(back, manifest["validation"]["hidden"][:64])
    assert counts["cholesky"] == 2
    assert torch.equal(first, second)


def test_cached_solve_agrees_with_the_per_call_reference_solve():
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    manifest = _manifest(seed=23)
    artifact = fit_map(manifest["fit"], {"rho": 0.02})
    message = manifest["validation"]["hidden"]
    served = apply_map(artifact, message)
    system = artifact.gram + artifact.ridge * torch.eye(artifact.rows)
    weights = torch.cholesky_solve(
        ((message.float() - artifact.mean_hidden) @ artifact.hidden_residual.mT).mT,
        torch.linalg.cholesky(system)).mT
    reference = artifact.mean_embedding + weights @ artifact.embedding_residual
    assert float((served - reference).norm() / reference.norm()) < 1e-4


def test_reference_embeddings_cannot_change_predictions():
    """apply_map never reads a reference embedding; only the hidden rows matter."""
    from occ3_sglang.calibration_corpus import evaluate_map
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    manifest = _manifest(seed=29)
    artifact = fit_map(manifest["fit"], {"rho": 0.02})
    hidden = manifest["validation"]["hidden"]
    first = apply_map(artifact, hidden)
    altered = {**manifest["validation"],
               "embeddings": torch.randn_like(manifest["validation"]["embeddings"])}
    evaluate_map(apply_map, artifact, altered)
    assert torch.equal(first, apply_map(artifact, hidden))


def test_evaluation_cannot_mutate_the_artifact():
    from occ3_sglang.calibration_corpus import evaluate_map
    from occ3_sglang.statebridge_calibrated import (TENSOR_NAMES, apply_map, fit_map,
                                                    serialize_artifact)
    manifest = _manifest(seed=31)
    artifact = fit_map(manifest["fit"], {"rho": 0.02}, provenance=_provenance())
    before = serialize_artifact(artifact)
    tensors = {name: getattr(artifact, name).clone() for name in TENSOR_NAMES}
    hostile = {**manifest["validation"],
               "hidden": torch.randn_like(manifest["validation"]["hidden"]),
               "embeddings": torch.randn_like(manifest["validation"]["embeddings"])}
    evaluate_map(apply_map, artifact, hostile)
    for name, value in tensors.items():
        assert torch.equal(value, getattr(artifact, name))
    assert serialize_artifact(artifact) == before


def test_fit_means_are_unaffected_by_validation_and_test_content():
    """The fit artifact is identical when only validation/test turns change."""
    from occ3_sglang.calibration_corpus import split_manifest
    from occ3_sglang.statebridge_calibrated import (TENSOR_NAMES, artifact_digest,
                                                    fit_map)
    corpus = _turns(16, rows=40, seed=41)
    first = split_manifest(corpus, fit_turns=8, validation_turns=4, test_turns=4)
    keep = set(first["fit"]["turn_ids"])
    other = _turns(16, rows=40, seed=77)
    replaced = [block if block.turn_id in keep else other[index]
                for index, block in enumerate(corpus)]
    second = split_manifest(replaced, fit_turns=8, validation_turns=4, test_turns=4)
    assert first["fit"]["manifest_digest"] == second["fit"]["manifest_digest"]
    assert (first["validation"]["manifest_digest"]
            != second["validation"]["manifest_digest"])
    assert first["test"]["manifest_digest"] != second["test"]["manifest_digest"]
    left = fit_map(first["fit"], {"rho": 0.02}, provenance=_provenance())
    right = fit_map(second["fit"], {"rho": 0.02}, provenance=_provenance())
    assert left.fit_manifest_digest == right.fit_manifest_digest
    assert artifact_digest(left) == artifact_digest(right)
    for name in TENSOR_NAMES:
        assert torch.equal(getattr(left, name), getattr(right, name))
