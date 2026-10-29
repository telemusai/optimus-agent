"""Stdlib contract tests; no installed SGLang, Torch, GPU or model."""
from dataclasses import asdict, replace
import math
import pytest

from occ3_sglang.latent_types import (LatentError, RuntimeProfile, bounded_payload,
    control_request, validate_control_payload, decode_descriptor)
from occ3_sglang.attachment_contract import CaptureWorkerDescriptor


def profile(**changes):
    return RuntimeProfile("epoch", 9, "model-hash", "glm", **changes)


def capture(**changes):
    values = dict(attempt_id="attempt", reservation_id="reserve", native_rid="rid",
        principal_digest="a" * 64, runtime_epoch="epoch", model_fingerprint="model-hash",
        method_id=profile().method, deadline=1000.0, capture_id="capture", source_session_id="source",
        source_turn_nonce="turn", runtime_generation=9, span_policy="whole_completed_response_tail_v1",
        max_prefix_tokens=8, max_capture_bytes=128 * 1024 * 1024)
    values.update(changes)
    return CaptureWorkerDescriptor(**values)


@pytest.mark.parametrize("value", [object(), {"x": object()}, {"x": float("nan")}, {"x": 2**64}, [0]*129])
def test_no_arbitrary_objects_in_control(value):
    with pytest.raises(LatentError):
        bounded_payload(value)


def test_exact_control_envelope():
    req = control_request("id", "health", {})
    assert validate_control_payload(req) == req
    with pytest.raises(LatentError):
        validate_control_payload({**req, "tokens": [1]})
    reply = dict(schema="occ3.control.result.v1", command_id="id", ok=True, result={}, error="")
    assert validate_control_payload(reply, result=True) == reply
    with pytest.raises(LatentError):
        validate_control_payload({**reply, "error": "oops"}, result=True)


def test_descriptor_is_chat_canonical():
    original = capture()
    assert decode_descriptor(asdict(original)) == original
    assert decode_descriptor(None) is None
    assert profile().tile_rows == 256
    with pytest.raises(LatentError):
        profile(expected_ranks=1)
    with pytest.raises(LatentError):
        profile(max_active=2)


@pytest.mark.parametrize("field", ["max_active","max_records","max_live_bytes","workspace_bytes","tile_rows"])
def test_profile_boolean_not_integer(field):
    with pytest.raises((LatentError,ValueError)):
        profile(**{field:True})


# Registry-driven method naming: the per-message default, the frozen dual ridge
# and the registered challengers are all nameable; nothing else is.

def test_registered_methods_may_be_named_and_unknown_ids_are_refused():
    from occ3_sglang import latent_types
    from occ3_sglang.native_profile import (CALIBRATED_METHOD, PER_MESSAGE_METHOD,
                                            PROCRUSTES_METHOD, PROCRUSTES_METHOD_PLAIN,
                                            PROCRUSTES_METHOD_SO)
    for method in (PER_MESSAGE_METHOD, CALIBRATED_METHOD, PROCRUSTES_METHOD,
                   PROCRUSTES_METHOD_SO, PROCRUSTES_METHOD_PLAIN):
        assert latent_types.advertised_method(method)
        assert profile(method=method).method == method
    for refused in ("statebridge.glm53.final-prenorm-predictor.unknown.v1",
                    "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v2", "", None):
        assert not latent_types.advertised_method(refused)
        with pytest.raises(LatentError):
            profile(method=refused)


def test_the_registry_table_drives_the_artifact_reference_check():
    # A calibration reference may only be carried by a method whose registry
    # entry requires a calibration artifact; the per-message method may not.
    from occ3_sglang.native_profile import (CALIBRATED_METHOD, PER_MESSAGE_METHOD,
                                            PROCRUSTES_METHOD)
    for method in (CALIBRATED_METHOD, PROCRUSTES_METHOD):
        values = profile(method=method, calibration_path="C:/abs/frozen.occ3cal",
                         calibration_digest="a" * 64)
        assert values.calibration_path.endswith(".occ3cal")
    for method in (PER_MESSAGE_METHOD,):
        with pytest.raises(LatentError) as caught:
            profile(method=method, calibration_path="C:/abs/frozen.occ3cal",
                    calibration_digest="a" * 64)
        assert caught.value.code == "profile"


def test_a_stubbed_registry_leaves_only_the_local_default_nameable(monkeypatch):
    import sys
    import types as stub_types
    module = stub_types.ModuleType("occ3_sglang.native_profile")
    module.load_pinned_profile = lambda: {}
    monkeypatch.setitem(sys.modules, "occ3_sglang.native_profile", module)
    assert latent_types_module().advertised_method(latent_types_module().METHOD)
    assert not latent_types_module().advertised_method(
        "statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1")
    assert not latent_types_module().advertised_method(
        "statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes.v1")
    fields = dict(runtime_epoch="epoch", runtime_generation=9,
                  model_fingerprint="model-hash", model_id="glm")
    with pytest.raises(LatentError):
        latent_types_module().RuntimeProfile(
            **fields, method="statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1")
    # A stubbed registry also cannot vouch for an artifact reference.
    with pytest.raises(LatentError):
        latent_types_module().RuntimeProfile(
            **fields, method=latent_types_module().METHOD,
            calibration_path="C:/abs/frozen.occ3cal", calibration_digest="a" * 64)


@pytest.fixture(autouse=True)
def _registry_module_identity():
    """Every test leaves the registry module identity untouched.

    Tests here delete or stub the sys.modules entry for
    occ3_sglang.native_profile. A later import of a DIFFERENT module object
    would split ProfileError identity and poison tests that compare classes
    across import statements (for example test_native_method_registry.py).
    The snapshot check catches that before it leaks out of this file.
    """
    import sys
    import occ3_sglang.native_profile as before
    yield
    assert sys.modules.get("occ3_sglang.native_profile") is before
    import occ3_sglang
    assert occ3_sglang.native_profile is before


def test_the_registry_source_is_parsed_without_importing_it(monkeypatch):
    # The importless fallback reads the registry's own constructs only: the
    # parsed table equals the imported one, and a source that is not the
    # registry's shape fails closed to the local default.
    import sys
    real_import = builtins_module().__import__

    def no_native_profile(name, *args, **kwargs):
        # Relative imports reach __import__ with the bare module name and the
        # package context in level, so a name-only check must block them too:
        # a real re-import here would bind a SECOND module object.
        if name == "occ3_sglang.native_profile" or name == "native_profile":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins_module(), "__import__", no_native_profile)
    monkeypatch.delitem(sys.modules, "occ3_sglang.native_profile")
    latent = latent_types_module()
    table = latent._parsed_registry_table()
    assert table is not None and table[latent.METHOD] is False
    assert table["statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1"] is True
    assert table["statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes.v1"] is True
    assert latent.advertised_method(latent.METHOD)
    assert latent.advertised_method(
        "statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1")
    assert not latent.advertised_method("statebridge.glm53.final-prenorm-predictor.unknown.v1")


def latent_types_module():
    from occ3_sglang import latent_types
    return latent_types


def builtins_module():
    import builtins
    return builtins
