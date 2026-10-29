"""Native method registry: advertised identity for both bridge methods."""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


def _profile(method):
    return {"model_fingerprint": "sha256:" + "a" * 64, "model_id": "glm-5.3",
            "expected_ranks": 8, "hidden_dim": 6144, "layer_index": 77,
            "max_prefix_tokens": 128, "max_live_bytes": 268435456,
            "workspace_bytes": 134217728, "max_records": 256, "max_active": 1,
            "max_ttl_seconds": 3600, "anchor_alpha": 0.0, "tile_rows": 256,
            "method": method}


def _binding(method, *, hidden_dim=6144, layer_index=77,
             model_fingerprint="sha256:" + "a" * 64):
    from occ3_sglang.statebridge_calibrated import CalibrationBinding
    return CalibrationBinding(
        method_id=method, hidden_dim=hidden_dim, rows=512, layer_index=layer_index,
        ridge=1.5, ridge_rel=0.02, solve_residual=0.01, hyperparameters={"rho": 0.02},
        capture_span_policy="whole_completed_response_tail_v1",
        model_fingerprint=model_fingerprint, checkpoint_revision="rev-1",
        tokenizer_fingerprint="sha256:" + "b" * 64, embedding_id="glm53-tp8-vocab",
        capture_method_id="glm53.final-prenorm.capture.v1",
        predictor_position_offset=-1, precision="float32", corpus_digest="c" * 64,
        fit_manifest_digest="d" * 64, validation_record_digest="",
        validation_artifact_digest="")


def test_calibrated_method_is_advertised_with_its_own_identity():
    from occ3_sglang import native_profile as profile
    from occ3_sglang import statebridge_calibrated
    assert profile.CALIBRATED_METHOD == statebridge_calibrated.METHOD_ID
    assert profile.CALIBRATED_METHOD in profile.advertised_methods()
    descriptor = profile.method_descriptor(profile.CALIBRATED_METHOD)
    assert descriptor["method_id"] == statebridge_calibrated.METHOD_ID
    assert descriptor["requires_calibration_artifact"] is True
    assert descriptor["fit_source"] == "independent-calibration-turns"
    assert descriptor["alignment"] == "frozen-dual-ridge.v1"
    assert descriptor["solver"].endswith("statebridge_calibrated.apply_map")
    assert descriptor["predictor_position_offset"] == -1
    assert descriptor["precision"] == "float32"


def test_message_method_still_resolves_to_the_message_solver():
    import importlib
    from occ3_sglang import latent_types, native_profile as profile, statebridge, statebridge_calibrated
    assert profile.PER_MESSAGE_METHOD == latent_types.METHOD
    descriptor = profile.method_descriptor(profile.PER_MESSAGE_METHOD)
    assert descriptor["fit_source"] == "message-rows"
    assert descriptor["requires_calibration_artifact"] is False
    assert descriptor["alignment"] == latent_types.ALIGNMENT
    module_name, function_name = descriptor["solver"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert getattr(module, function_name) is statebridge.align_hidden_sequence
    calibrated = profile.method_descriptor(profile.CALIBRATED_METHOD)["solver"]
    module_name, function_name = calibrated.rsplit(".", 1)
    # Wave-1 module surface: the registry solver must resolve to the
    # calibrated method's real apply function (fit_map/apply_map contract).
    assert getattr(importlib.import_module(module_name), function_name) is (
        statebridge_calibrated.apply_map)
    assert profile.advertised_methods() == (profile.PER_MESSAGE_METHOD,
                                            profile.CALIBRATED_METHOD)


def test_method_ids_are_distinct_and_only_one_needs_an_artifact():
    from occ3_sglang import native_profile as profile
    first, second = profile.advertised_methods()
    assert first != second
    assert profile.method_descriptor(first)["requires_calibration_artifact"] is False
    assert profile.method_descriptor(second)["requires_calibration_artifact"] is True
    for method in (first, second):
        assert profile.require_method(method) == method
        receipt = profile.receipt_descriptor(method)
        assert set(receipt) == {"method_id", "representation", "alignment",
                                "predictor_position_offset", "precision"}
        assert all(type(value) in (str, int) for value in receipt.values())
        assert receipt["method_id"] == method


def test_receipt_descriptor_agrees_with_the_source_receipt_type():
    from occ3_sglang.attachment_contract import AttachmentError, SourceCaptureReceipt
    from occ3_sglang import native_profile as profile
    receipt = profile.receipt_descriptor(profile.PER_MESSAGE_METHOD)
    value = {"intent_digest": "1" * 64, "source_handle": "handle-1", "native_rid": "rid-1",
             "response_id": "response-1", "capture_commit_id": "commit-1",
             "accepted_span_start": 0, "accepted_span_end": 8, "retained_count": 8,
             "predictor_position_offset": receipt["predictor_position_offset"],
             "representation": receipt["representation"], "layer_index": 77,
             "hidden_dim": 6144, "dtype": "bfloat16", "layout_digest": "2" * 64,
             "ready_ranks": 8, "expected_ranks": 8, "accepted_sequence_digest": "3" * 64,
             "output_sha256": "4" * 64, "expires_at_epoch": 2000000000, "byte_count": 8 * 6144 * 2,
             "schema": "occ3.source.v1"}
    SourceCaptureReceipt.parse(value)
    # The receipt type pins this literal, so identity separation is carried by the
    # negotiated method id, not by a per-method representation string.
    with pytest.raises(AttachmentError):
        SourceCaptureReceipt.parse(dict(value, representation="frozen_dual_ridge"))


def test_profile_validator_accepts_only_advertised_methods():
    from occ3_sglang import native_profile as profile
    from occ3_sglang.native_profile import ProfileError
    for method in profile.advertised_methods():
        assert profile._validate_profile(_profile(method)) == _profile(method)
    for method in ("statebridge.glm53.final-prenorm-predictor.continuous-prefix.v2",
                   "statebridge.unknown.v1"):
        with pytest.raises(ProfileError) as caught:
            profile._validate_profile(_profile(method))
        assert caught.value.code == "profile_invalid"
        with pytest.raises(ProfileError):
            profile.method_descriptor(method)


def test_pinned_profile_may_name_the_calibrated_method(tmp_path):
    from occ3_sglang import native_profile as profile

    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")

    sources = {"schema": "optimus.occ3.native-sources.v1",
               "source_revision": "e00930c5489053f26d86b179cee0d087f846acbb",
               "sglang_version": "0.5.21",
               "manifests": [{"name": "overlay.json", "sha256": "a" * 64}],
               "files": [{"root": "occ3_sglang", "path": "native_profile.py",
                          "sha256": "c" * 64}]}
    baked = types.ModuleType("occ3_sglang._native_profile_digest")
    raw_profile = encode(_profile(profile.CALIBRATED_METHOD))
    raw_sources = encode(sources)
    (tmp_path / "native_profile.json").write_bytes(raw_profile)
    (tmp_path / "native_sources.json").write_bytes(raw_sources)
    baked.PROFILE_SHA256 = hashlib.sha256(raw_profile).hexdigest()
    baked.SOURCE_SHA256 = hashlib.sha256(raw_sources).hexdigest()
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(profile, "__file__",
                                             str(tmp_path / "native_profile.py")))
        stack.enter_context(mock.patch.dict(sys.modules, {baked.__name__: baked}))
        loaded = profile.load_pinned_profile()
    assert loaded["method"] == profile.CALIBRATED_METHOD


def test_artifact_with_the_wrong_method_id_is_rejected():
    from occ3_sglang import native_profile as profile
    from occ3_sglang.native_profile import ProfileError
    calibrated_profile = _profile(profile.CALIBRATED_METHOD)
    message_profile = _profile(profile.PER_MESSAGE_METHOD)
    binding = _binding(profile.CALIBRATED_METHOD)
    assert profile.validate_method_binding(binding, calibrated_profile)["method_id"] == (
        profile.CALIBRATED_METHOD)
    assert profile.validate_method_binding(None, message_profile)["method_id"] == (
        profile.PER_MESSAGE_METHOD)
    with pytest.raises(ProfileError):
        profile.validate_method_binding(binding, message_profile)
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import CalibrationBinding
    # A binding can never be constructed under the message method; only a forged
    # object can claim it, and the profile validator still refuses it.
    with pytest.raises(LatentError) as caught:
        CalibrationBinding(**dict(_binding(profile.CALIBRATED_METHOD).scalars(),
                                  method_id=profile.PER_MESSAGE_METHOD))
    assert caught.value.code == "identity"
    import dataclasses
    forged = dataclasses.replace(binding)
    object.__setattr__(forged, "method_id", profile.PER_MESSAGE_METHOD)
    with pytest.raises(ProfileError):
        profile.validate_method_binding(forged, calibrated_profile)
    with pytest.raises(ProfileError):
        profile.validate_method_binding(None, calibrated_profile)
    with pytest.raises(ProfileError):
        profile.validate_method_binding(binding, calibrated_profile,
                                        negotiated_method_id=profile.PER_MESSAGE_METHOD)


def test_artifact_must_match_the_pinned_profile_identity():
    from occ3_sglang import native_profile as profile
    from occ3_sglang.native_profile import ProfileError
    calibrated_profile = _profile(profile.CALIBRATED_METHOD)
    assert profile.validate_method_binding(_binding(profile.CALIBRATED_METHOD),
                                           calibrated_profile)
    for kwargs in ({"hidden_dim": 512}, {"layer_index": 78},
                   {"model_fingerprint": "sha256:" + "f" * 64}):
        with pytest.raises(ProfileError) as caught:
            profile.validate_method_binding(_binding(profile.CALIBRATED_METHOD, **kwargs),
                                            calibrated_profile)
        assert caught.value.code == "profile_invalid"


def test_calibrated_map_cannot_be_served_behind_the_message_method():
    import dataclasses

    torch = pytest.importorskip("torch")
    from occ3_sglang import native_profile as profile
    from occ3_sglang.latent_types import LatentError
    from occ3_sglang.statebridge_calibrated import apply_map, fit_map
    generator = torch.Generator().manual_seed(0)
    rank, dim, rows = 8, 64, 200
    left = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    right = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    factor = torch.randn(rows, rank, generator=generator)
    hidden = 0.95 ** 0.5 * (factor @ left) + 0.05 ** 0.5 * torch.randn(rows, dim, generator=generator)
    embeddings = 0.95 ** 0.5 * (factor @ right) + 0.05 ** 0.5 * torch.randn(rows, dim, generator=generator)
    partition = {"hidden": hidden, "embeddings": embeddings,
                 "row_turn_ids": ("turn-1",) * rows, "turn_ids": ("turn-1",),
                 "manifest_digest": ""}
    artifact = fit_map(partition, {"rho": 0.02})
    apply_map(artifact, hidden[:8])
    with pytest.raises(LatentError) as caught:
        apply_map(dataclasses.replace(artifact, method_id=profile.PER_MESSAGE_METHOD),
                  hidden[:8])
    assert caught.value.code == "identity"


def test_profile_defaults_to_the_per_message_method_and_accepts_the_calibrated_one():
    from occ3_sglang import latent_types
    from occ3_sglang import native_profile as profile
    fields = dict(runtime_epoch="epoch", runtime_generation=9,
                  model_fingerprint="sha256:" + "a" * 64, model_id="glm-5.3")
    defaulted = latent_types.RuntimeProfile(**fields)
    assert defaulted.method == latent_types.METHOD == profile.PER_MESSAGE_METHOD
    calibrated = latent_types.RuntimeProfile(**fields, method=profile.CALIBRATED_METHOD)
    assert calibrated.method == profile.CALIBRATED_METHOD
    # The method id is part of the profile identity, so the two cannot collide.
    assert calibrated.binding_digest != defaulted.binding_digest
    for refused in ("statebridge.glm53.final-prenorm-predictor.unknown.v1", "", None, 7, True):
        with pytest.raises(latent_types.LatentError) as caught:
            latent_types.RuntimeProfile(**fields, method=refused)
        assert caught.value.code == "profile"


def test_install_attests_and_packaging_packages_the_calibrated_method():
    from deploy.v3 import native_package
    from occ3_sglang import latent_types
    from occ3_sglang import native_install
    from occ3_sglang import native_profile as profile
    from occ3_sglang import statebridge_calibrated
    for module in ("statebridge_calibrated.py", "calibration_corpus.py",
                   "calibration_probe.py", "align_diagnostics.py"):
        assert module in native_install._REQUIRED_PLUGIN
        # Attestation only covers modules the packaged flat tree really holds.
        assert (Path(native_install.__file__).parent / module).is_file()
    assert statebridge_calibrated.METHOD_ID in native_package.shipped_methods()
    assert latent_types.METHOD in native_package.shipped_methods()
    assert native_package.shipped_methods() == profile.advertised_methods()
    # The former local literal now resolves from the registry, so it cannot drift.
    assert native_package.METHOD == profile.PER_MESSAGE_METHOD
    assert native_package.default_method() == profile.PER_MESSAGE_METHOD
    with pytest.raises(AttributeError):
        native_package.undeclared_method_literal


def test_packaged_profile_may_pin_either_advertised_method_and_nothing_else():
    from deploy.v3 import native_package
    from occ3_sglang import native_profile as profile
    profile_value = _profile(profile.PER_MESSAGE_METHOD)
    assert json.loads(native_package._profile(profile_value))["method"] == profile.PER_MESSAGE_METHOD
    calibrated = _profile(profile.CALIBRATED_METHOD)
    assert json.loads(native_package._profile(calibrated))["method"] == profile.CALIBRATED_METHOD
    with pytest.raises(native_package.PackageError) as caught:
        native_package._profile(_profile("statebridge.glm53.final-prenorm-predictor.unknown.v1"))
    assert caught.value.code == "profile_method_not_advertised"
    # A malformed id never reaches the registry check; it is refused as a name.
    with pytest.raises(native_package.PackageError) as caught:
        native_package._profile(_profile(""))
    assert caught.value.code == "invalid_profile_identifier"


def test_profile_falls_back_to_the_per_message_method_without_a_registry(monkeypatch):
    """An embedded host may stub the registry; only the local default stays nameable."""
    import sys
    import types as stub_types
    from occ3_sglang import latent_types
    fields = dict(runtime_epoch="epoch", runtime_generation=9,
                  model_fingerprint="sha256:" + "a" * 64, model_id="glm-5.3")
    stub = stub_types.ModuleType("occ3_sglang.native_profile")
    stub.load_pinned_profile = lambda: {}
    monkeypatch.setitem(sys.modules, "occ3_sglang.native_profile", stub)
    assert latent_types.advertised_method(latent_types.METHOD)
    assert not latent_types.advertised_method(
        "statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1")
    assert latent_types.RuntimeProfile(**fields).method == latent_types.METHOD
    with pytest.raises(latent_types.LatentError):
        latent_types.RuntimeProfile(
            **fields, method="statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1")


def test_static_registry_read_matches_the_imported_registry():
    """The offline build path parses the registry instead of importing it; the
    parsed view must equal the imported one exactly, so packaging can never
    accept a method the installed runtime does not advertise (or refuse one it
    does)."""
    from deploy.v3 import native_package
    from occ3_sglang import native_profile as profile

    parsed = native_package._static_registry()
    imported = {method: dict(profile.method_descriptor(method))
                for method in profile.advertised_methods()}
    assert parsed == imported
    assert set(parsed) == set(native_package.shipped_methods())
    assert len(parsed) >= 2
    assert parsed[profile.PER_MESSAGE_METHOD]["requires_calibration_artifact"] is False
    assert parsed[profile.CALIBRATED_METHOD]["requires_calibration_artifact"] is True
