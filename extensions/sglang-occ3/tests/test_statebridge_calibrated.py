"""Module surface, math and numerical guards for the frozen dual-ridge map."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _model(rank=8, dim=64, seed=1000):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rank, dim, generator=generator) / rank ** 0.5,
            torch.randn(rank, dim, generator=generator) / rank ** 0.5)


def _sample(model, rows=200, kappa=0.9, seed=1):
    left, right = model
    generator = torch.Generator().manual_seed(seed)
    factor = torch.randn(rows, left.shape[0], generator=generator)
    noise_h = torch.randn(rows, left.shape[1], generator=generator) * (1.0 - kappa) ** 0.5
    noise_e = torch.randn(rows, right.shape[1], generator=generator) * (1.0 - kappa) ** 0.5
    return kappa ** 0.5 * (factor @ left) + noise_h, kappa ** 0.5 * (factor @ right) + noise_e


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


def _fitted(rows=200, dim=64, kappa=0.9, seed=1, rho=0.02):
    from occ3_sglang.statebridge_calibrated import fit_map
    hidden, embeddings = _sample(_model(dim=dim), rows=rows, kappa=kappa, seed=seed)
    return fit_map(_partition(hidden, embeddings), {"rho": rho})


def test_method_identity_and_schema_version_are_pinned():
    from occ3_sglang.statebridge_calibrated import (ARTIFACT_SCHEMA,
                                                    DEFAULT_HYPERPARAMETERS,
                                                    HYPERPARAMETER_FIELDS, METHOD_ID)
    assert METHOD_ID == "statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1"
    assert ARTIFACT_SCHEMA == "optimus.occ3.calibration-artifact.v2"
    assert HYPERPARAMETER_FIELDS == ("rho",)
    assert DEFAULT_HYPERPARAMETERS == {"rho": 0.02}


def test_partition_must_carry_exactly_the_contract_fields():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import PARTITION_FIELDS, fit_map
    hidden, embeddings = _sample(_model(dim=32), rows=64, kappa=0.9, seed=5)
    partition = _partition(hidden, embeddings)
    assert set(PARTITION_FIELDS) == {"hidden", "embeddings", "row_turn_ids", "turn_ids",
                                     "manifest_digest"}
    assert fit_map(partition, {}).ridge_rel == 0.02
    with pytest.raises(LatentError) as caught:
        fit_map({**partition, "extra": 1}, {})
    assert caught.value.code == "profile"
    for missing in ("hidden", "embeddings", "row_turn_ids", "turn_ids", "manifest_digest"):
        with pytest.raises(LatentError):
            fit_map({k: v for k, v in partition.items() if k != missing}, {})
    with pytest.raises(LatentError):
        fit_map({**partition, "row_turn_ids": partition["row_turn_ids"][:-1]}, {})
    with pytest.raises(LatentError):
        fit_map({**partition, "turn_ids": ("turn-000", "turn-000")}, {})
    with pytest.raises(LatentError):
        fit_map({**partition, "turn_ids": ("turn-009",)}, {})
    with pytest.raises(LatentError):
        fit_map({**partition, "manifest_digest": "not-a-digest"}, {})


def test_hyperparameters_are_explicit_and_validated():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import fit_map
    hidden, embeddings = _sample(_model(dim=32), rows=64, kappa=0.9, seed=9)
    partition = _partition(hidden, embeddings)
    assert fit_map(partition, {"rho": 0.2}).ridge_rel == 0.2
    for bad in ({"rho": 0.0}, {"rho": 1.0}, {"rho": -0.1}, {"rho": float("nan")},
                {"rank": 8}, {"rho": 0.02, "extra": 1}):
        with pytest.raises(LatentError) as caught:
            fit_map(partition, bad)
        assert caught.value.code == "profile"


def test_fit_refuses_the_row_bounds():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import (CALIBRATION_ROWS_LIMIT,
                                                    MIN_CALIBRATION_ROWS, fit_map)
    assert MIN_CALIBRATION_ROWS == 64 and CALIBRATION_ROWS_LIMIT == 2048
    hidden, embeddings = _sample(_model(dim=32), rows=32, kappa=0.9, seed=11)
    with pytest.raises(LatentError) as caught:
        fit_map(_partition(hidden, embeddings), {})
    assert caught.value.code == "profile"
    tall = torch.randn(CALIBRATION_ROWS_LIMIT + 1, 8,
                      generator=torch.Generator().manual_seed(2))
    with pytest.raises(LatentError) as caught:
        fit_map(_partition(tall, tall.clone(), turn_ids=("turn-a", "turn-b", "turn-c")), {})
    assert caught.value.code == "profile"


def test_fit_refuses_nonfinite_or_mismatched_calibration_rows():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import fit_map
    hidden, embeddings = _sample(_model(dim=32), rows=64, kappa=0.9, seed=13)
    hidden[4, 4] = float("nan")
    with pytest.raises(LatentError) as caught:
        fit_map(_partition(hidden, embeddings), {})
    assert caught.value.code == "numerical"
    hidden, embeddings = _sample(_model(dim=32), rows=64, kappa=0.9, seed=14)
    with pytest.raises(LatentError) as caught:
        fit_map(_partition(hidden, embeddings[:32]), {})
    assert caught.value.code == "profile"


def test_fit_records_the_solve_residual_and_symmetrises_the_gram():
    artifact = _fitted()
    assert type(artifact.solve_residual) is float
    assert 0.0 <= artifact.solve_residual < 1e-2
    assert torch.equal(artifact.gram, artifact.gram.mT)


def test_fit_is_permutation_invariant_over_partition_rows():
    """Re-ordering the calibration rows cannot change the frozen transform."""
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    hidden, embeddings = _sample(_model(dim=64), rows=200, kappa=0.9, seed=21)
    partition = _partition(hidden, embeddings)
    first = fit_map(partition, {"rho": 0.02})
    order = torch.randperm(hidden.shape[0], generator=torch.Generator().manual_seed(6))
    row_ids = tuple(partition["row_turn_ids"][int(index)] for index in order)
    second = fit_map({"hidden": hidden[order], "embeddings": embeddings[order],
                      "row_turn_ids": row_ids,
                      "turn_ids": tuple(sorted(set(row_ids))),
                      "manifest_digest": ""}, {"rho": 0.02})
    inverse = torch.argsort(order)
    restored = second.gram.index_select(0, inverse).index_select(1, inverse)
    assert first.ridge == pytest.approx(second.ridge, rel=1e-5)
    assert torch.allclose(first.gram, restored, atol=1e-4)
    assert torch.allclose(first.mean_hidden, second.mean_hidden, atol=1e-6)
    message = hidden[:16]
    assert torch.allclose(apply_map(first, message), apply_map(second, message),
                          atol=1e-4)


def test_quality_rises_with_the_signal_on_an_independent_partition():
    from occ3_sglang.calibration_corpus import evaluate_map
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    values = []
    for kappa in (0.0, 0.5, 1.0):
        hidden, embeddings = _sample(_model(dim=32), rows=160, kappa=kappa, seed=25)
        held_h, held_e = _sample(_model(dim=32), rows=64, kappa=kappa, seed=26)
        artifact = fit_map(_partition(hidden, embeddings, ("turn-a", "turn-b", "turn-c",
                                                           "turn-d", "turn-e")),
                          {"rho": 0.02})
        held = _partition(held_h, held_e, ("turn-x", "turn-y"))
        record = evaluate_map(apply_map, artifact, held)
        assert record["r2_standard"] is not None
        values.append(record["r2_standard"])
    assert values[0] < values[1] < values[2]
    assert values[2] > 0.9


def test_apply_refuses_bad_shapes_and_rows():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import apply_map
    artifact = _fitted(rows=128, dim=64)
    with pytest.raises(LatentError) as caught:
        apply_map(artifact, torch.randn(32, 32,
                                        generator=torch.Generator().manual_seed(3)))
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        apply_map(artifact, torch.randn(1, 64,
                                        generator=torch.Generator().manual_seed(4)))
    assert caught.value.code == "profile"
    broken = torch.randn(8, 64, generator=torch.Generator().manual_seed(5))
    broken[0, 0] = float("inf")
    with pytest.raises(LatentError) as caught:
        apply_map(artifact, broken)
    assert caught.value.code == "numerical"


def test_serving_transform_accepts_a_zero_dim_tensor_nu():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import serving_transform
    z = torch.ones(4, 3)
    z = z / z.norm(dim=-1, keepdim=True)
    nu = torch.tensor(12.5)
    served = serving_transform({"kind": "n1"}, z, nu=nu)
    assert bool(torch.allclose(served.norm(dim=-1), torch.full((4,), 12.5)))
    with pytest.raises(LatentError) as caught:
        serving_transform({"kind": "n1"}, z, nu=torch.tensor(float("nan")))
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        serving_transform({"kind": "n1"}, z, nu=torch.ones(2))
    assert caught.value.code == "profile"


def test_apply_returns_raw_rows_and_serving_transform_calibrates():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map, serving_transform
    hidden, embeddings = _sample(_model(dim=64), rows=128, kappa=0.9, seed=31)
    artifact = fit_map(_partition(hidden, embeddings), {"rho": 0.02})
    z = apply_map(artifact, hidden[:32])
    norms = z.norm(dim=-1)
    assert torch.isfinite(z).all()
    assert float(norms.std()) > 0.0
    assert torch.equal(serving_transform(None, z), z)
    served = serving_transform({"kind": "n1"}, z, nu=2.5)
    assert torch.allclose(served.norm(dim=-1), torch.full_like(norms, 2.5), atol=1e-5)
    zero = z.clone()
    zero[0] = 0.0
    for config in (None, {"kind": "n1"}):
        with pytest.raises(LatentError) as caught:
            serving_transform(config, zero, nu=2.5)
        assert caught.value.code == "numerical"
    nonfinite = z.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(LatentError) as caught:
        serving_transform(None, nonfinite)
    assert caught.value.code == "numerical"
    with pytest.raises(LatentError) as caught:
        serving_transform({"kind": "n1"}, z)
    assert caught.value.code == "profile"
    with pytest.raises(LatentError) as caught:
        serving_transform({"kind": "n2"}, z, nu=2.5)
    assert caught.value.code == "profile"


def test_cholesky_failure_is_a_named_numerical_failure(monkeypatch):
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import (apply_map, deserialize_artifact,
                                                    fit_map, serialize_artifact)
    hidden, embeddings = _sample(_model(dim=64), rows=128, kappa=0.9, seed=41)
    artifact = fit_map(_partition(hidden, embeddings), {}, provenance=_provenance())
    raw = serialize_artifact(artifact)

    def refusing(system, *args, **kwargs):
        raise RuntimeError("cholesky failed")

    monkeypatch.setattr(torch.linalg, "cholesky", refusing)
    back = deserialize_artifact(raw)
    with pytest.raises(LatentError) as caught:
        apply_map(back, torch.randn(8, 64,
                                     generator=torch.Generator().manual_seed(6)))
    assert caught.value.code == "numerical"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_describe_is_scalar_only_and_receipt_safe():
    from occ3_sglang.statebridge_calibrated import CalibrationProvenance, describe, fit_map
    hidden, embeddings = _sample(_model(dim=64), rows=128, kappa=0.9, seed=51)
    partition = _partition(hidden, embeddings, digest="a" * 64)
    unbound = fit_map(partition, {})
    provenance = CalibrationProvenance(
        layer_index=77, model_fingerprint="sha256:" + "a" * 64, checkpoint_revision="rev-1",
        tokenizer_fingerprint="sha256:" + "b" * 64, embedding_id="glm53-tp8-vocab",
        capture_method_id="glm53.final-prenorm.capture.v1")
    bound = fit_map(partition, {}, provenance=provenance)
    for summary, expect_bound in ((describe(unbound), 0), (describe(bound), 1)):
        assert all(type(value) in (int, float, str) for value in summary.values())
        assert summary["bound"] == expect_bound
    assert describe(bound)["serving_kind"] == "none"
    assert describe(bound)["rho"] == 0.02
    assert describe(bound)["capture_span_policy"] == "whole_completed_response_tail_v1"
    assert describe(bound)["fit_manifest_digest"] == "a" * 64
    assert "heldout_r2" not in describe(bound)
    assert "solve_residual" in describe(bound)


def test_required_provenance_fields_list_the_span_policy():
    from occ3_sglang.statebridge_calibrated import required_provenance_fields
    fields = required_provenance_fields()
    assert set(fields) >= {"layer_index", "model_fingerprint", "checkpoint_revision",
                           "tokenizer_fingerprint", "embedding_id", "capture_method_id",
                           "capture_span_policy", "predictor_position_offset", "precision"}


def test_artifact_is_small_and_describable():
    from occ3_sglang.statebridge_calibrated import CALIBRATION_ROWS_LIMIT, describe
    artifact = _fitted(rows=256, dim=128)
    summary = describe(artifact)
    assert summary["rows"] == 256 and summary["hidden_dim"] == 128
    assert summary["bytes"] < 2 * CALIBRATION_ROWS_LIMIT * 6144 * 4
    assert all(type(v) in (int, float, str) for v in summary.values())


def test_frozen_map_lands_near_the_embedding_manifold_and_keeps_state_dependence():
    """The two properties the shipped per-message fit cannot hold at once."""
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map, serving_transform
    model = _model(dim=256, seed=2000)
    hidden, embeddings = _sample(model, rows=512, kappa=0.75, seed=21)
    artifact = fit_map(_partition(hidden, embeddings), {"rho": 0.02})
    message_h, message_e = _sample(model, rows=64, kappa=0.75, seed=22)
    target = float(message_e.norm(dim=-1).mean())
    z = apply_map(artifact, message_h)
    served = serving_transform({"kind": "n1"}, z, nu=target)
    cosine = float((served * message_e).sum(-1).div(
        served.norm(dim=-1) * message_e.norm(dim=-1)).mean())
    other = serving_transform({"kind": "n1"},
                              apply_map(artifact, _sample(model, rows=64, kappa=0.75,
                                                          seed=23)[0]), nu=target)
    change = float((other - served).norm() / served.norm())
    assert cosine > 0.6, f"on-manifold cosine too low: {cosine}"
    assert change > 0.3, f"state dependence too low: {change}"
