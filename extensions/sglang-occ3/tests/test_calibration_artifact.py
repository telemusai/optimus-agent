"""Artifact integrity, binding and provenance for the calibrated map (schema v2)."""
from __future__ import annotations

import functools
import json

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


def _provenance():
    from occ3_sglang.statebridge_calibrated import CalibrationProvenance
    return CalibrationProvenance(
        layer_index=77, model_fingerprint="sha256:" + "a" * 64, checkpoint_revision="rev-1",
        tokenizer_fingerprint="sha256:" + "b" * 64, embedding_id="glm53-tp8-vocab",
        capture_method_id="glm53.final-prenorm.capture.v1")


def _partition(hidden, embeddings, turn_ids=("turn-1", "turn-2"), digest=""):
    rows = hidden.shape[0]
    per_turn = rows // len(turn_ids)
    assert per_turn * len(turn_ids) == rows
    row_turn_ids = tuple(turn_id for turn_id in turn_ids for _ in range(per_turn))
    return {"hidden": hidden, "embeddings": embeddings, "row_turn_ids": row_turn_ids,
            "turn_ids": tuple(turn_ids), "manifest_digest": digest}


@functools.lru_cache(maxsize=None)
def _fitted(rows=200, dim=64, seed=1, provenance=True, digest=""):
    from occ3_sglang.statebridge_calibrated import fit_map
    hidden, embeddings = _sample(_model(dim=dim), rows=rows, seed=seed)
    return fit_map(_partition(hidden, embeddings, digest=digest), {},
                   provenance=_provenance() if provenance else None)


def _split(raw):
    from occ3_sglang.statebridge_calibrated import ARTIFACT_MAGIC
    length = int.from_bytes(raw[len(ARTIFACT_MAGIC):16], "big")
    return json.loads(raw[16:16 + length].decode("ascii")), raw[16 + length:]


def _rebuild(header, payload):
    from occ3_sglang.statebridge_calibrated import ARTIFACT_MAGIC, _artifact_digest, _canonical
    header = dict(header)
    header["artifact_digest"] = _artifact_digest(header)
    raw_header = _canonical(header)
    return ARTIFACT_MAGIC + len(raw_header).to_bytes(8, "big") + raw_header + payload


def _raw(header, payload):
    from occ3_sglang.statebridge_calibrated import ARTIFACT_MAGIC, _canonical
    raw_header = _canonical(header)
    return ARTIFACT_MAGIC + len(raw_header).to_bytes(8, "big") + raw_header + payload


def test_round_trip_preserves_predictions_and_binding():
    from occ3_sglang.statebridge_calibrated import (artifact_digest, apply_map,
                                                    deserialize_artifact,
                                                    serialize_artifact)
    artifact = _fitted()
    raw = serialize_artifact(artifact)
    assert serialize_artifact(artifact) == raw
    back = deserialize_artifact(raw, expected_method_id=artifact.method_id,
                               expected_hidden_dim=artifact.hidden_dim,
                               expected_provenance=_provenance(),
                               expected_artifact_digest=artifact_digest(artifact))
    assert (back.rows, back.hidden_dim, back.ridge, back.ridge_rel,
            back.solve_residual) == (artifact.rows, artifact.hidden_dim, artifact.ridge,
                                     artifact.ridge_rel, artifact.solve_residual)
    for name in ("mean_hidden", "mean_embedding", "hidden_residual", "embedding_residual",
                 "gram"):
        assert torch.equal(getattr(back, name), getattr(artifact, name))
    message, _ = _sample(_model(dim=64), rows=16, seed=9)
    first = apply_map(artifact, message)
    second = apply_map(back, message)
    assert torch.allclose(first, second, atol=1e-4)
    assert torch.equal(first, second)


def test_header_carries_the_v2_binding_fields():
    from occ3_sglang.statebridge_calibrated import (BINDING_FIELDS, HEADER_FIELDS,
                                                    serialize_artifact)
    header, _ = _split(serialize_artifact(_fitted()))
    assert set(header) == set(HEADER_FIELDS)
    assert {"method_id", "hidden_dim", "layer_index", "rows", "ridge", "ridge_rel",
            "solve_residual", "hyperparameters", "capture_span_policy",
            "model_fingerprint", "checkpoint_revision", "tokenizer_fingerprint",
            "embedding_id", "capture_method_id", "predictor_position_offset",
            "precision", "corpus_digest", "fit_manifest_digest",
            "validation_record_digest", "validation_artifact_digest", "serving"} <= set(BINDING_FIELDS)
    assert header["schema"] == "optimus.occ3.calibration-artifact.v2"
    assert header["hyperparameters"] == {"rho": header["ridge_rel"]}
    assert header["capture_span_policy"] == "whole_completed_response_tail_v1"
    assert header["fit_manifest_digest"] == ""
    assert header["validation_record_digest"] == ""
    assert header["validation_artifact_digest"] == ""
    assert header["serving"] is None


def test_an_artifact_of_the_previous_schema_is_refused():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import deserialize_artifact, serialize_artifact
    header, payload = _split(serialize_artifact(_fitted()))
    forged = dict(header, schema="optimus.occ3.calibration-artifact.v1")
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(forged, payload))
    assert caught.value.code == "profile"


def test_refuses_a_mismatched_method_id():
    from occ3_sglang.latent_types import METHOD, LatentError
    from occ3_sglang.statebridge_calibrated import (CalibrationBinding, deserialize_artifact,
                                                    serialize_artifact)
    artifact = _fitted()
    raw = serialize_artifact(artifact)
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(raw, expected_method_id=METHOD)
    assert caught.value.code == "identity"
    header, payload = _split(raw)
    forged = dict(header, method_id=METHOD)
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(forged, payload))
    assert caught.value.code == "identity"
    scalars = dict(artifact.binding.scalars(), method_id=METHOD)
    with pytest.raises(LatentError) as caught:
        CalibrationBinding(**scalars)
    assert caught.value.code == "identity"


def test_refuses_a_mismatched_hidden_dim():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import deserialize_artifact, serialize_artifact
    artifact = _fitted()
    raw = serialize_artifact(artifact)
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(raw, expected_hidden_dim=artifact.hidden_dim + 1)
    assert caught.value.code == "profile"
    header, payload = _split(raw)
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(
            _rebuild(dict(header, hidden_dim=artifact.hidden_dim + 1), payload))
    assert caught.value.code == "profile"


@pytest.mark.parametrize("field", ["layer_index", "model_fingerprint", "checkpoint_revision",
                                   "tokenizer_fingerprint", "embedding_id", "capture_method_id",
                                   "capture_span_policy"])
def test_refuses_each_mismatched_provenance_field(field):
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import (CalibrationProvenance,
                                                    deserialize_artifact, serialize_artifact)
    raw = serialize_artifact(_fitted())
    values = {"layer_index": 77, "model_fingerprint": "sha256:" + "a" * 64,
              "checkpoint_revision": "rev-1",
              "tokenizer_fingerprint": "sha256:" + "b" * 64,
              "embedding_id": "glm53-tp8-vocab",
              "capture_method_id": "glm53.final-prenorm.capture.v1",
              "capture_span_policy": "whole_completed_response_tail_v1"}
    values[field] = {"layer_index": 78}.get(field, "other-value-1")
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(raw, expected_provenance=CalibrationProvenance(**values))
    assert caught.value.code == "identity"


def test_refuses_a_tampered_or_missing_digest():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import (ARTIFACT_MAGIC, artifact_digest,
                                                    deserialize_artifact,
                                                    serialize_artifact)
    artifact = _fitted()
    raw = serialize_artifact(artifact)
    digest = artifact_digest(artifact)
    header, payload = _split(raw)
    forged = dict(header, artifact_digest=("0" if digest[0] != "0" else "1") + digest[1:])
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_raw(forged, payload))
    assert caught.value.code == "identity"
    missing = dict(header)
    del missing["artifact_digest"]
    raw_header = json.dumps(missing, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(
            ARTIFACT_MAGIC + len(raw_header).to_bytes(8, "big") + raw_header + payload)
    assert caught.value.code == "profile"
    tampered_payload = payload[:-1] + bytes([payload[-1] ^ 0xFF])
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(header, tampered_payload))
    assert caught.value.code == "identity"
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(raw, expected_artifact_digest="f" * 64)
    assert caught.value.code == "identity"


def test_refuses_a_resaved_artifact_under_a_pinned_digest():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import (artifact_digest, deserialize_artifact,
                                                    serialize_artifact)
    original = _fitted(seed=1)
    other = _fitted(seed=2)
    raw_other = serialize_artifact(other)
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(raw_other, expected_artifact_digest=artifact_digest(original))
    assert caught.value.code == "identity"


def test_refuses_a_shape_mismatch_between_gram_and_hidden_dim():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import deserialize_artifact, serialize_artifact
    artifact = _fitted()
    raw = serialize_artifact(artifact)
    header, payload = _split(raw)
    shapes = dict(header["tensor_shapes"], gram=[artifact.rows, artifact.rows + 1])
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(dict(header, tensor_shapes=shapes), payload))
    assert caught.value.code == "profile"
    shapes = dict(header["tensor_shapes"], hidden_residual=[artifact.rows,
                                                            artifact.hidden_dim + 8])
    with pytest.raises(LatentError) as caught:
        deserialize_artifact(_rebuild(dict(header, tensor_shapes=shapes), payload))
    assert caught.value.code == "profile"


def test_payload_is_read_without_pickle(monkeypatch):
    from occ3_sglang.statebridge_calibrated import deserialize_artifact, serialize_artifact
    seen = {}
    original = torch.load

    def recording(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording)
    deserialize_artifact(serialize_artifact(_fitted()))
    assert seen["weights_only"] is True and seen["map_location"] == "cpu"


def test_serialize_refuses_an_unbound_artifact():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import serialize_artifact
    with pytest.raises(LatentError) as caught:
        serialize_artifact(_fitted(provenance=False))
    assert caught.value.code == "profile"


def test_serialize_refuses_a_stale_content_digest():
    """A rebuilt artifact that kept a stale digest cannot re-serialize itself."""
    from dataclasses import replace
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import serialize_artifact
    first = _fitted(seed=1)
    second = _fitted(seed=2)
    stale = replace(second, content_digest=first.content_digest)
    with pytest.raises(LatentError) as caught:
        serialize_artifact(stale)
    assert caught.value.code == "identity"


def test_content_digest_is_set_by_fit_and_by_load():
    from occ3_sglang.statebridge_calibrated import (artifact_digest, deserialize_artifact,
                                                    serialize_artifact)
    artifact = _fitted()
    assert artifact.content_digest == artifact_digest(artifact)
    back = deserialize_artifact(serialize_artifact(artifact))
    assert back.content_digest == artifact_digest(artifact)


def test_describe_is_scalar_only_and_never_carries_tensor_content():
    from occ3_sglang.statebridge_calibrated import describe
    bound = describe(_fitted())
    unbound = describe(_fitted(provenance=False))
    for summary in (bound, unbound):
        assert all(type(value) in (int, float, str) for value in summary.values())
        assert all(name not in ("mean_hidden", "mean_embedding", "hidden_residual",
                                "embedding_residual", "gram") for name in summary)
    assert bound["bound"] == 1 and unbound["bound"] == 0
    assert bound["corpus_digest"] != unbound.get("corpus_digest")
    assert {"method_id", "rows", "hidden_dim", "ridge", "ridge_rel", "solve_residual",
            "rho", "bytes"} <= set(bound)
    assert len(bound["corpus_digest"]) == 64


def test_corpus_digest_binds_the_rows_and_the_turn_ids():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import corpus_digest
    hidden, embeddings = _sample(_model(dim=64), rows=64, seed=71)
    other, _ = _sample(_model(dim=64), rows=64, seed=72)
    base = corpus_digest(hidden, embeddings, turn_ids=["turn-1"])
    assert corpus_digest(hidden.clone(), embeddings.clone(), turn_ids=["turn-1"]) == base
    assert corpus_digest(hidden, embeddings, turn_ids=["turn-1", "turn-2"]) != base
    assert corpus_digest(other, embeddings, turn_ids=["turn-1"]) != base
    assert corpus_digest(hidden, embeddings) != base
    with pytest.raises(LatentError) as caught:
        corpus_digest(hidden, embeddings, turn_ids=["turn-1", "turn-1"])
    assert caught.value.code == "profile"
    with pytest.raises(LatentError):
        corpus_digest(hidden, embeddings, turn_ids=["turn 1"])
    with pytest.raises(LatentError):
        corpus_digest(hidden, embeddings[:-1])


def test_binding_refuses_an_unbound_or_mismatched_scalar():
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import CalibrationBinding
    scalars = dict(_fitted().binding.scalars())
    CalibrationBinding(**scalars)
    for changes in ({"rows": 63}, {"hidden_dim": 2}, {"ridge": 0.0}, {"ridge_rel": 1.0},
                    {"solve_residual": float("nan")}, {"solve_residual": -1.0},
                    {"hyperparameters": {"rho": 0.5}}, {"corpus_digest": "short"},
                    {"fit_manifest_digest": "short"},
                    {"validation_record_digest": "short"},
                    {"validation_artifact_digest": "short"},
                    {"capture_span_policy": "bad policy"},
                    {"serving": {"kind": "n2"}}, {"serving": "n1"},
                    {"precision": "bfloat16"}, {"predictor_position_offset": -2},
                    {"model_fingerprint": ""}):
        with pytest.raises(LatentError):
            CalibrationBinding(**dict(scalars, **changes))


def test_the_binding_accepts_a_pinned_validation_evidence():
    from occ3_sglang.statebridge_calibrated import CalibrationBinding
    scalars = dict(_fitted().binding.scalars())
    pinned = CalibrationBinding(**{**scalars, "validation_record_digest": "c" * 64,
                                    "validation_artifact_digest": "d" * 64,
                                    "serving": {"kind": "n1"}})
    assert pinned.validation_record_digest == "c" * 64
    assert pinned.serving == {"kind": "n1"}
