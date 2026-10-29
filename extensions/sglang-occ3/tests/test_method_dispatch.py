"""Two-method dispatch end to end: registry -> loader -> aligner -> served rows.

Both frozen alignment modules implement the wave-1 interface contract; these
tests fit small synthetic corpora for BOTH methods, serialize the artifacts,
load them through the dispatch with a runtime profile and apply them through
aligner_for, and check the identity refusals the campaign requires.
"""
from __future__ import annotations

from dataclasses import replace
import importlib

import pytest

torch = pytest.importorskip("torch")

from occ3_sglang import alignment_dispatch, latent_types, statebridge, statebridge_calibrated
from occ3_sglang import statebridge_calibrated_procrustes as procrustes
from occ3_sglang.latent_types import LatentError, RuntimeProfile
from occ3_sglang.native_profile import (CALIBRATED_METHOD, PER_MESSAGE_METHOD,
                                        PROCRUSTES_METHOD, PROCRUSTES_METHOD_PLAIN,
                                        PROCRUSTES_METHOD_SO, method_descriptor,
                                        registered_methods)

DIM, ROWS = 6144, 128
FINGERPRINT = "sha256:" + "a" * 64
RANK = 16


def _provenance(**changes):
    values = dict(layer_index=77, model_fingerprint=FINGERPRINT,
                  checkpoint_revision="rev-1", tokenizer_fingerprint="sha256:" + "b" * 64,
                  embedding_id="glm53-tp8-vocab",
                  capture_method_id="glm53.final-prenorm.capture.v1")
    values.update(changes)
    return statebridge_calibrated.CalibrationProvenance(**values)


def _procrustes_provenance(**changes):
    values = dict(layer_index=77, model_fingerprint=FINGERPRINT,
                  checkpoint_revision="rev-1", tokenizer_fingerprint="sha256:" + "b" * 64,
                  embedding_id="glm53-tp8-vocab",
                  capture_method_id="glm53.final-prenorm.capture.v1",
                  capture_span_policy="whole_completed_response_tail_v1")
    values.update(changes)
    return procrustes.ProcrustesProvenance(**values)


def _corpus(rows=ROWS, dim=DIM, rank=RANK, kappa=0.9, seed=0):
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    right = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    factor = torch.randn(rows, rank, generator=generator)
    hidden = kappa ** 0.5 * (factor @ left) + (1 - kappa) ** 0.5 * torch.randn(rows, dim, generator=generator)
    embeddings = kappa ** 0.5 * (factor @ right) + (1 - kappa) ** 0.5 * torch.randn(rows, dim, generator=generator)
    return hidden, embeddings


def _partition(hidden, embeddings, turn_ids):
    return {"hidden": hidden, "embeddings": embeddings,
            "row_turn_ids": tuple(turn_id for turn_id in turn_ids
                                  for _ in range(hidden.shape[0] // len(turn_ids))),
            "turn_ids": tuple(turn_ids), "manifest_digest": ""}


def _dual_ridge_artifact(serving=None):
    hidden, embeddings = _corpus(seed=1)
    artifact = statebridge_calibrated.fit_map(
        _partition(hidden, embeddings, ("turn-0", "turn-1")), {"rho": 0.02},
        provenance=_provenance())
    if serving is not None:
        artifact = replace(artifact,
                           binding=replace(artifact.binding, serving=serving),
                           content_digest=None)
    return artifact


def _procrustes_artifact(hyperparams=None):
    hidden, embeddings = _corpus(seed=2)
    return procrustes.fit_map(_partition(hidden, embeddings, ("turn-0", "turn-1")),
                              hyperparams or {"r": RANK, "eta": 0.01, "variant": "o"},
                              provenance=_procrustes_provenance())


def _profile(method, path="", digest=""):
    return RuntimeProfile("epoch", 9, FINGERPRINT, "glm-5.3", method=method,
                         calibration_path=str(path), calibration_digest=digest)


def _message(rows=8):
    return _corpus(rows=rows, seed=30)[0]


# -- registry -----------------------------------------------------------------

def test_registry_holds_both_methods_with_their_own_modules_and_schemas():
    methods = registered_methods()
    assert methods == (PER_MESSAGE_METHOD, CALIBRATED_METHOD, PROCRUSTES_METHOD,
                       PROCRUSTES_METHOD_SO, PROCRUSTES_METHOD_PLAIN)
    for method in methods:
        descriptor = method_descriptor(method)
        assert descriptor["predictor_position_offset"] == -1
        assert descriptor["precision"] == "float32"
        assert descriptor["solver"].count(".") == 2
        if descriptor["requires_calibration_artifact"]:
            module = importlib.import_module(descriptor["artifact_module"])
            assert module.METHOD_ID in methods or method in (
                PROCRUSTES_METHOD, PROCRUSTES_METHOD_SO, PROCRUSTES_METHOD_PLAIN)
            assert module.ARTIFACT_SCHEMA == descriptor["artifact_schema"]
            assert module.apply_map.__module__ == descriptor["artifact_module"]
        else:
            assert descriptor["artifact_module"] is None
            assert descriptor["artifact_schema"] is None
    dual = method_descriptor(CALIBRATED_METHOD)
    assert dual["artifact_module"] == "occ3_sglang.statebridge_calibrated"
    assert dual["alignment"] == "frozen-dual-ridge.v1"
    assert dual["solver"] == "occ3_sglang.statebridge_calibrated.apply_map"
    challenger = method_descriptor(PROCRUSTES_METHOD)
    assert challenger["artifact_module"] == "occ3_sglang.statebridge_calibrated_procrustes"
    assert challenger["alignment"] == "frozen-lowrank-whitened-procrustes.v1"
    assert challenger["artifact_schema"] == procrustes.ARTIFACT_SCHEMA
    # The packaged surface stays the committed two-method advertisement; the
    # challengers are registered for dispatch but are not yet pinnable there.
    from occ3_sglang.native_profile import advertised_methods
    assert advertised_methods() == (PER_MESSAGE_METHOD, CALIBRATED_METHOD)


def test_a_runtime_profile_may_name_a_registered_challenger():
    assert latent_types.advertised_method(PROCRUSTES_METHOD)
    assert latent_types.advertised_method(PROCRUSTES_METHOD_SO)
    profile = _profile(PROCRUSTES_METHOD, "C:/abs/path.occ3cal", "c" * 64)
    assert profile.method == PROCRUSTES_METHOD
    assert latent_types.RuntimeProfile("epoch", 9, FINGERPRINT, "glm-5.3",
                                       method=procrustes.METHOD_ID_PLAIN).method == (
        procrustes.METHOD_ID_PLAIN)
    with pytest.raises(latent_types.LatentError):
        latent_types.RuntimeProfile("epoch", 9, FINGERPRINT, "glm-5.3",
                                    method="statebridge.glm53.final-prenorm-predictor.unknown.v1")


# -- end to end: fit, serialize, load, apply ----------------------------------

def _write(tmp_path, name, artifact, module):
    path = tmp_path / name
    path.write_bytes(module.serialize_artifact(artifact))
    return path


def test_dual_ridge_artifact_loads_and_applies_through_dispatch(tmp_path):
    artifact = _dual_ridge_artifact()
    path = _write(tmp_path, "dual.occ3cal", artifact, statebridge_calibrated)
    profile = _profile(CALIBRATED_METHOD, path, artifact.content_digest)
    loaded = alignment_dispatch.load_frozen_artifact(profile)
    assert loaded.method_id == CALIBRATED_METHOD
    aligner = alignment_dispatch.aligner_for(profile, loaded)
    hidden, embeddings = _corpus(rows=8, seed=30)
    prefix, info = aligner(hidden, embeddings, 1.0)
    assert isinstance(info, alignment_dispatch.CalibratedAlignmentInfo)
    assert info.method == CALIBRATED_METHOD and info.rows == ROWS and info.hidden_dim == DIM
    assert info.artifact_digest == profile.calibration_digest
    # N0 is the fit default: the served rows ARE the raw apply_map rows.
    assert torch.allclose(prefix, statebridge_calibrated.apply_map(loaded, hidden))
    assert torch.allclose(prefix.norm(dim=-1),
                          statebridge_calibrated.apply_map(loaded, hidden).norm(dim=-1))


def test_procrustes_artifact_loads_and_applies_through_dispatch(tmp_path):
    artifact = _procrustes_artifact()
    path = _write(tmp_path, "prd.occ3cal", artifact, procrustes)
    digest = procrustes.artifact_digest(artifact)
    profile = _profile(PROCRUSTES_METHOD, path, digest)
    loaded = alignment_dispatch.load_frozen_artifact(profile)
    assert loaded.method_id == PROCRUSTES_METHOD
    aligner = alignment_dispatch.aligner_for(profile, loaded)
    hidden, embeddings = _corpus(rows=8, seed=31)
    prefix, info = aligner(hidden, embeddings, 1.0)
    assert info.method == PROCRUSTES_METHOD and info.hidden_dim == DIM
    assert info.artifact_digest == digest
    # The procrustes binding schema records no serving choice: the aligner
    # serves raw apply_map rows and never renormalises them.
    assert aligner.serving is None
    assert torch.allclose(prefix, procrustes.apply_map(loaded, hidden))


def test_the_other_methods_artifact_is_refused_by_identity(tmp_path):
    dual = _dual_ridge_artifact()
    prd = _procrustes_artifact()
    # In memory: a dual-ridge artifact under a procrustes profile.
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(_profile(PROCRUSTES_METHOD), dual)
    assert caught.value.code == "identity"
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(_profile(CALIBRATED_METHOD), prd)
    assert caught.value.code == "identity"
    # On disk: serialized dual-ridge bytes behind a procrustes reference. The
    # procrustes loader fails closed on the foreign container (wrong magic) at
    # load: the method's own deserializer decides what its bytes may be.
    path = _write(tmp_path, "dual.occ3cal", dual, statebridge_calibrated)
    profile = _profile(PROCRUSTES_METHOD, path, dual.content_digest)
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.load_frozen_artifact(profile)
    assert caught.value.code == "profile"
    prd_path = _write(tmp_path, "prd.occ3cal", prd, procrustes)
    dual_profile = _profile(CALIBRATED_METHOD, prd_path, procrustes.artifact_digest(prd))
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.load_frozen_artifact(dual_profile)
    assert caught.value.code == "profile"


def test_a_binding_for_another_model_or_layer_is_refused():
    other_model = _dual_ridge_artifact()
    artifact = replace(other_model,
                       binding=replace(other_model.binding,
                                       model_fingerprint="sha256:" + "f" * 64),
                       content_digest=None)
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(_profile(CALIBRATED_METHOD), artifact)
    assert caught.value.code == "identity"
    other_layer = _dual_ridge_artifact()
    artifact = replace(other_layer,
                       binding=replace(other_layer.binding, layer_index=76),
                       content_digest=None)
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(_profile(CALIBRATED_METHOD), artifact)
    assert caught.value.code == "identity"


def test_provenance_equality_is_refused_at_load(tmp_path):
    artifact = _dual_ridge_artifact()
    path = _write(tmp_path, "dual.occ3cal", artifact, statebridge_calibrated)
    profile = _profile(CALIBRATED_METHOD, path, artifact.content_digest)
    for changes in ({"model_fingerprint": "sha256:" + "f" * 64},
                    {"layer_index": 76},
                    {"tokenizer_fingerprint": "sha256:" + "e" * 64},
                    {"checkpoint_revision": "rev-2"},
                    {"embedding_id": "glm53-tp8-vocab-x"},
                    {"capture_method_id": "glm53.final-prenorm.capture.v2"}):
        with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
            alignment_dispatch.load_frozen_artifact(
                profile, provenance=_provenance(**changes))
        assert caught.value.code == "identity"
    # The matching provenance loads.
    assert alignment_dispatch.load_frozen_artifact(
        profile, provenance=_provenance()).binding.method_id == CALIBRATED_METHOD
    # A provenance of the wrong method class is refused, not reinterpreted.
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.load_frozen_artifact(
            profile, provenance=_procrustes_provenance())
    assert caught.value.code == "profile"
    # And the procrustes loader refuses a dual-ridge provenance the same way.
    prd = _procrustes_artifact()
    prd_path = _write(tmp_path, "prd.occ3cal", prd, procrustes)
    prd_profile = _profile(PROCRUSTES_METHOD, prd_path, procrustes.artifact_digest(prd))
    with pytest.raises(alignment_dispatch.AlignmentRefusal):
        alignment_dispatch.load_frozen_artifact(
            prd_profile, provenance=_provenance())


# -- N0 vs N1 -----------------------------------------------------------------

def test_n0_does_not_renormalise_and_n1_renormalises_to_nu():
    hidden, embeddings = _corpus(rows=8, seed=32)
    nu = 2.5
    n0 = _dual_ridge_artifact()
    aligner = alignment_dispatch.aligner_for(_profile(CALIBRATED_METHOD), n0)
    prefix, _ = aligner(hidden, embeddings, nu)
    raw = statebridge_calibrated.apply_map(n0, hidden)
    assert torch.allclose(prefix, raw)
    n1 = _dual_ridge_artifact(serving={"kind": "n1"})
    aligner = alignment_dispatch.aligner_for(_profile(CALIBRATED_METHOD), n1)
    assert aligner.serving == {"kind": "n1"}
    prefix, _ = aligner(hidden, embeddings, nu)
    assert torch.allclose(prefix.norm(dim=-1), torch.full((hidden.shape[0],), nu), atol=1e-4)
    # N1 without the vocabulary norm statistic refuses rather than guessing.
    with pytest.raises(LatentError) as caught:
        aligner(hidden, embeddings, None)
    assert caught.value.code == "profile"


def test_no_aligner_reads_message_reference_embeddings(monkeypatch):
    """Calibrated methods map hidden rows only; embeddings are never read."""
    hidden, embeddings = _corpus(rows=8, seed=33)
    for method, artifact in ((CALIBRATED_METHOD, _dual_ridge_artifact()),
                             (PROCRUSTES_METHOD, _procrustes_artifact())):
        aligner = alignment_dispatch.aligner_for(_profile(method), artifact)
        wrong = torch.full_like(embeddings, float("nan"))
        prefix, _ = aligner(hidden, wrong, 1.0)
        assert bool(torch.isfinite(prefix).all())


def test_the_verified_digest_is_reused_instead_of_redigesting(monkeypatch):
    artifact = _dual_ridge_artifact()
    calls = []
    real = statebridge_calibrated.artifact_digest
    monkeypatch.setattr(statebridge_calibrated, "artifact_digest",
                        lambda *args, **kwargs: calls.append(args) or real(*args, **kwargs))
    alignment_dispatch.aligner_for(_profile(CALIBRATED_METHOD), artifact)
    assert calls == []  # content digest set by the verified fit is trusted
    unbound_hidden, unbound_embeddings = _corpus(seed=3)
    unverified = statebridge_calibrated.fit_map(
        _partition(unbound_hidden, unbound_embeddings, ("turn-0", "turn-1")), {"rho": 0.02})
    with pytest.raises(alignment_dispatch.AlignmentRefusal):
        alignment_dispatch.aligner_for(_profile(CALIBRATED_METHOD), unverified)
    alignment_dispatch.artifact_resident_bytes(artifact)
    assert calls == []  # resident bytes read tensor sizes, not content


def test_an_unbound_artifact_is_never_served():
    hidden, embeddings = _corpus(seed=4)
    unbound = statebridge_calibrated.fit_map(
        _partition(hidden, embeddings, ("turn-0", "turn-1")), {"rho": 0.02})
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(_profile(CALIBRATED_METHOD), unbound)
    assert caught.value.code == "profile"


def test_artifact_resident_bytes_read_the_methods_own_tensor_set():
    dual = _dual_ridge_artifact()
    prd = _procrustes_artifact()
    assert alignment_dispatch.artifact_resident_bytes(dual) == statebridge_calibrated.artifact_bytes(dual)
    assert alignment_dispatch.artifact_resident_bytes(prd) == procrustes.artifact_bytes(prd)
    assert alignment_dispatch.artifact_resident_bytes(dual) > 0


def test_per_message_method_still_resolves_to_the_shipped_solver(monkeypatch):
    calls = []
    original = statebridge.align_hidden_sequence
    monkeypatch.setattr(statebridge, "align_hidden_sequence",
                        lambda *args, **kwargs: calls.append(args) or original(*args, **kwargs))
    apply_calls = []
    monkeypatch.setattr(statebridge_calibrated, "apply_map",
                        lambda *args, **kwargs: apply_calls.append(args))
    aligner = alignment_dispatch.aligner_for(_profile(PER_MESSAGE_METHOD))
    assert aligner.serving == {"kind": "n1"}
    hidden, embeddings = _corpus(rows=8, seed=34)
    prefix, info = aligner(hidden, embeddings, 1.0)
    assert len(calls) == 1 and apply_calls == []
    assert isinstance(info, statebridge.AlignmentInfo) and info.hidden_dim == DIM
