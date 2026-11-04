"""Offline tests for the plan-declared calibrated-artifact read-only bind.

The capability is exactly one profile-bound file: the plan pins whole-file
identity (sha256 + size) and the staged profile evidence; the verified
profile's calibration_path is the sole authority for the basename, with the
same basename on the pooled source side and the container destination side.
No arbitrary host path is ever expressible through the plan.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import docker_config as v3_dc
from deploy.v3 import lifecycle as life
from deploy.v3.lifecycle import Failure, validate_plan

from test_controller import V3_CANDIDATE_IMAGE, v3_original


BASENAME = "calibrated-dual-ridge-9a7394dea73cf8d3.bin"
DESTINATION = "/occ3/calibration/" + BASENAME
CALIBRATION_DIGEST = "9a7394dea73cf8d33fd651bf8b43d62d739c94d0940bd6f4ae494fd83ba0be90"


def calibrated_original():
    """Original shaped like the measured DGX neo-glm53 mount declaration."""
    doc = v3_original()
    doc["Config"]["Cmd"] = ["-m", "sglang.launch_server", "--api-key", "synthetic-key",
                            "--model-path", "/data/neo-glm53/model"]
    doc["HostConfig"]["Mounts"] = [{"Type": "bind", "Source": "/data/neo-glm53",
                                    "Target": "/data/neo-glm53"}]
    doc["Mounts"] = [{"Type": "bind", "Source": "/data/neo-glm53",
                      "Destination": "/data/neo-glm53", "Mode": "", "RW": True,
                      "Propagation": "rprivate"}]
    return doc


def binding():
    return {"source": v3_dc.CALIBRATION_SOURCE_ROOT + "/" + BASENAME,
            "destination": DESTINATION, "basename": BASENAME}


def calibrated_create(original, body, name="candidate"):
    """Offline create+inspect model materializing declared mounts (Docker 29.6.2)."""
    candidate = deepcopy(original)
    candidate.update(Id="c" * 64, Name="/" + name, Image=body["Image"])
    candidate["Config"] = {key: deepcopy(value) for key, value in body.items()
                           if key not in ("HostConfig", "NetworkingConfig")}
    host = deepcopy(body["HostConfig"])
    if "OomKillDisable" in host:
        if host["OomKillDisable"] is None:
            host["OomKillDisable"] = False
    else:
        host["OomKillDisable"] = None
    candidate["HostConfig"] = host
    candidate["State"] = {"Status": "created", "Running": False}
    declared = []
    for entry in host.get("Mounts") or []:
        declared.append(entry)
    for bind in host.get("Binds") or []:
        parts = bind.split(":")
        declared.append({"Type": "bind", "Source": parts[0], "Target": parts[1],
                         "ReadOnly": len(parts) > 2 and "ro" in parts[2].split(",")})
    candidate["Mounts"] = [{"Type": "bind", "Source": entry["Source"],
                             "Destination": entry["Target"],
                             "Mode": "ro" if entry.get("ReadOnly") else "",
                             "RW": not entry.get("ReadOnly", False),
                             "Propagation": "rprivate"} for entry in declared]
    return candidate


# Logical plan paths (production DGX namespace shapes); the I/O seam maps them
# to real tmp fixtures on any validation host. Production never sets the
# resolver, so the logical path IS the host path there.
LOGICAL_SOURCE_ROOT = "/pooled-calibration"
LOGICAL_PROFILE_PATH = "/staged/native-profile-input.json"


@pytest.fixture
def calibrated(tmp_path, monkeypatch):
    artifact = tmp_path / BASENAME
    payload = b"synthetic-calibration-payload" * 64
    artifact.write_bytes(payload)
    profile = {"calibration_path": DESTINATION, "calibration_digest": CALIBRATION_DIGEST}
    profile_bytes = json.dumps(profile).encode("utf-8")
    profile_path = tmp_path / "native-profile-input.json"
    profile_path.write_bytes(profile_bytes)
    monkeypatch.setattr(v3_dc, "CALIBRATION_SOURCE_ROOT", LOGICAL_SOURCE_ROOT)
    real = {LOGICAL_PROFILE_PATH: profile_path,
            LOGICAL_SOURCE_ROOT + "/" + BASENAME: artifact}
    monkeypatch.setattr(v3_dc, "_HOST_PATH_RESOLVER", real.__getitem__)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload),
            "profile_path": LOGICAL_PROFILE_PATH,
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "profile_bytes": len(profile_bytes),
            "artifact": artifact, "payload": payload, "profile_path_object": profile_path}


def capability_plan(calibrated, **changes):
    value = {"attempt_id": "occ3-live-20261026t095633z", "original_id": "a" * 64,
             "original_image_id": "sha256:" + "a" * 64,
             "candidate_image_id": "sha256:" + "b" * 64,
             "base_url": "http://127.0.0.1:30001",
             "ingress_base_url": "http://neo-k3-ingress:8080",
             "model": "glm-5.3", "mode": "production",
             "calibrated_artifact": {"sha256": calibrated["sha256"],
                                      "size": calibrated["size"]},
             "calibrated_profile_evidence": {"path": calibrated["profile_path"],
                                             "sha256": calibrated["profile_sha256"],
                                             "bytes": calibrated["profile_bytes"]},
             "static_input": {"sha256": calibrated["profile_sha256"]},
             "native_validator": {"static_input_sha256": calibrated["profile_sha256"]}}
    value.update(changes)
    return value


# ---------------------------------------------------------------- builder


def test_builder_appends_calibrated_readonly_bind_spec():
    original = calibrated_original()
    before = deepcopy(original)
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate",
                                      None, binding())
    assert body["HostConfig"]["Mounts"] == [
        {"Type": "bind", "Source": "/data/neo-glm53", "Target": "/data/neo-glm53"},
        {"Type": "bind", "Source": binding()["source"], "Target": DESTINATION,
         "ReadOnly": True}]
    assert body["HostConfig"]["Binds"] is None
    assert original == before  # the original inspect is never mutated


def test_builder_without_binding_is_unchanged():
    original = calibrated_original()
    plain = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate")
    with_binding = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate", None, None)
    assert plain == with_binding


@pytest.mark.parametrize("bad", [
    {"source": "/etc/calibrated-x.bin", "destination": DESTINATION, "basename": BASENAME},
    {"source": binding()["source"], "destination": "/etc/calibrated-x.bin", "basename": BASENAME},
    {"source": binding()["source"], "destination": DESTINATION, "basename": "other.bin"},
    {"source": binding()["source"], "destination": DESTINATION},
    {"source": binding()["source"], "destination": DESTINATION, "basename": BASENAME, "extra": 1},
    {"source": "/data/neo-glm53/p5-artifact-pooled/../root.bin", "destination": DESTINATION,
     "basename": "root.bin"},
])
def test_builder_refuses_bindings_outside_the_fixed_namespaces(bad):
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.build_create_request(calibrated_original(), V3_CANDIDATE_IMAGE,
                                   "candidate", None, bad)


def test_builder_refuses_destination_already_declared():
    original = calibrated_original()
    original["HostConfig"]["Binds"] = ["/data/neo-glm53:" + DESTINATION + ":ro"]
    original["HostConfig"]["Mounts"] = None
    original["Mounts"] = [{"Type": "bind", "Source": "/data/neo-glm53",
                           "Destination": DESTINATION, "Mode": "ro", "RW": False,
                           "Propagation": "rprivate"}]
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate", None, binding())


# ---------------------------------------------------------------- parity


def test_roundtrip_parity_accepts_the_declared_addition():
    original = calibrated_original()
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate", None, binding())
    candidate = calibrated_create(original, body)
    evidence = v3_dc.assert_candidate_parity(original, candidate, V3_CANDIDATE_IMAGE,
                                             None, binding())
    assert evidence["declared_mount_additions"] == [DESTINATION]
    assert evidence["actual_mount_count"] == 1  # original mounts only
    assert evidence["actual_mounts_exact"] is True


def test_parity_fails_when_the_addition_is_missing_from_the_candidate():
    original = calibrated_original()
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate")
    candidate = calibrated_create(original, body)
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.assert_candidate_parity(original, candidate, V3_CANDIDATE_IMAGE, None, binding())


def test_parity_fails_on_undeclared_extra_mount():
    original = calibrated_original()
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate", None, binding())
    candidate = calibrated_create(original, body)
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.assert_candidate_parity(original, candidate, V3_CANDIDATE_IMAGE)


@pytest.mark.parametrize("mode,propagation", [("", "rprivate"), ("ro", ""), ("ro", "private"),
                                              ("", "private")])
def test_parity_accepts_only_bounded_record_spellings(mode, propagation):
    original = calibrated_original()
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate", None, binding())
    candidate = calibrated_create(original, body)
    candidate["Mounts"][-1].update(Mode=mode, Propagation=propagation)
    v3_dc.assert_candidate_parity(original, candidate, V3_CANDIDATE_IMAGE, None, binding())


@pytest.mark.parametrize("record", [
    {"Mode": "rw"}, {"Propagation": "shared"}, {"RW": True}, {"Source": "/data/other.bin"},
])
def test_parity_refuses_unbounded_record_spellings(record):
    original = calibrated_original()
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate", None, binding())
    candidate = calibrated_create(original, body)
    candidate["Mounts"][-1].update(record)
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.assert_candidate_parity(original, candidate, V3_CANDIDATE_IMAGE, None, binding())


def test_parity_keeps_original_mount_records_exact():
    original = calibrated_original()
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate", None, binding())
    candidate = calibrated_create(original, body)
    candidate["Mounts"][0]["RW"] = False
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.assert_candidate_parity(original, candidate, V3_CANDIDATE_IMAGE, None, binding())


def test_parity_hostconfig_must_carry_the_declared_readonly_spec():
    original = calibrated_original()
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate", None, binding())
    candidate = calibrated_create(original, body)
    candidate["HostConfig"]["Mounts"][-1]["ReadOnly"] = False
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.assert_candidate_parity(original, candidate, V3_CANDIDATE_IMAGE, None, binding())


def test_old_parity_receipt_defaults_to_no_additions():
    original = calibrated_original()
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate")
    candidate = calibrated_create(original, body)
    evidence = v3_dc.assert_candidate_parity(original, candidate, V3_CANDIDATE_IMAGE)
    assert evidence["declared_mount_additions"] == []


# ---------------------------------------------------------------- plan schema


def test_validate_plan_accepts_the_capability(calibrated):
    plan = validate_plan(capability_plan(calibrated))
    assert plan["calibrated_artifact"] == {"sha256": calibrated["sha256"],
                                           "size": calibrated["size"]}


def test_validate_plan_ignores_absent_capability(calibrated):
    plan = {"attempt_id": "occ3-live-20261026t095633z", "original_id": "a" * 64,
            "original_image_id": "sha256:" + "a" * 64,
            "candidate_image_id": "sha256:" + "b" * 64,
            "base_url": "http://127.0.0.1:30001",
            "ingress_base_url": "http://neo-k3-ingress:8080", "model": "glm-5.3"}
    plan = validate_plan(plan)
    assert "calibrated_artifact" not in plan


@pytest.mark.parametrize("change", [
    {"calibrated_artifact": {"sha256": "z" * 64, "size": 10}},
    {"calibrated_artifact": {"sha256": "a" * 63, "size": 10}},
    {"calibrated_artifact": {"sha256": "a" * 64, "size": True}},
    {"calibrated_artifact": {"sha256": "a" * 64, "size": 0}},
    {"calibrated_artifact": {"sha256": "a" * 64, "size": v3_dc.CALIBRATION_ARTIFACT_MAX_BYTES + 1}},
    {"calibrated_artifact": {"sha256": "a" * 64}},
    {"calibrated_profile_evidence": {"path": "relative/path.json", "sha256": "a" * 64, "bytes": 8}},
    {"calibrated_profile_evidence": {"path": "/abs.json", "sha256": "a" * 64, "bytes": 0}},
    {"calibrated_profile_evidence": {"path": "/abs.json", "sha256": "a" * 64}},
    {"calibrated_profile_evidence": {"path": "/abs.json", "sha256": "a" * 64, "bytes": True}},
    {"calibrated_profile_evidence": {"path": "/abs.json", "sha256": "a" * 64,
                                     "bytes": v3_dc._CALIBRATION_PROFILE_MAX_BYTES + 1}},
    {"static_input": {"sha256": "b" * 64}},
    {"native_validator": {"static_input_sha256": "b" * 64}},
])
def test_validate_plan_refuses_malformed_capabilities(calibrated, change):
    plan = capability_plan(calibrated, **change)
    with pytest.raises(Failure, match="invalid calibrated artifact capability"):
        validate_plan(plan)


# ------------------------------------------------------- profile verification


def test_verify_calibrated_profile_returns_the_declared_binding(calibrated):
    spec = v3_dc.calibrated_capability(capability_plan(calibrated))
    verified = v3_dc.verify_calibrated_profile(spec)
    assert verified == {"calibration_path": DESTINATION,
                        "calibration_digest": CALIBRATION_DIGEST}


def test_verify_calibrated_profile_refuses_mutated_bytes(calibrated):
    tampered = b"T" * calibrated["profile_bytes"]
    calibrated["profile_path_object"].write_bytes(tampered)
    spec = v3_dc.calibrated_capability(capability_plan(calibrated))
    with pytest.raises(v3_dc.ConfigError, match="calibrated_profile_digest_mismatch"):
        v3_dc.verify_calibrated_profile(spec)


def test_verify_calibrated_profile_refuses_size_mismatch(calibrated):
    calibrated["profile_bytes"] += 1
    spec = v3_dc.calibrated_capability(capability_plan(calibrated))
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.verify_calibrated_profile(spec)


def test_capability_refuses_oversized_profile(calibrated):
    oversized = json.dumps({"calibration_path": DESTINATION,
                            "calibration_digest": CALIBRATION_DIGEST}).encode() + b" " * v3_dc._CALIBRATION_PROFILE_MAX_BYTES
    calibrated["profile_path_object"].write_bytes(oversized)
    plan = capability_plan(calibrated)
    plan["calibrated_profile_evidence"].update(
        sha256=hashlib.sha256(oversized).hexdigest(), bytes=len(oversized))
    plan["static_input"]["sha256"] = hashlib.sha256(oversized).hexdigest()
    plan["native_validator"]["static_input_sha256"] = hashlib.sha256(oversized).hexdigest()
    with pytest.raises(v3_dc.ConfigError, match="invalid_calibrated_profile_evidence"):
        v3_dc.calibrated_capability(plan)


@pytest.mark.parametrize("profile,code", [
    ({"calibration_path": "/occ3/other/file.bin"}, "calibrated_profile_path_invalid"),
    ({"calibration_path": "/occ3/calibration/" + BASENAME[:-1] + ".txt"}, "calibrated_profile_path_invalid"),
    ({"calibration_path": DESTINATION}, "calibrated_profile_digest_invalid"),
    ({"calibration_path": DESTINATION, "calibration_digest": "z" * 64}, "calibrated_profile_digest_invalid"),
    (["not", "a", "dict"], "calibrated_profile_malformed"),
])
def test_verify_calibrated_profile_refuses_malformed_content(calibrated, profile, code):
    payload = json.dumps(profile).encode("utf-8")
    calibrated["profile_path_object"].write_bytes(payload)
    plan = capability_plan(calibrated)
    plan["calibrated_profile_evidence"].update(
        sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
    plan["static_input"]["sha256"] = hashlib.sha256(payload).hexdigest()
    plan["native_validator"]["static_input_sha256"] = hashlib.sha256(payload).hexdigest()
    spec = v3_dc.calibrated_capability(plan)
    with pytest.raises(v3_dc.ConfigError, match=code):
        v3_dc.verify_calibrated_profile(spec)


def maybe_symlink(path, target):
    try:
        os.symlink(str(target), str(path))
    except OSError:
        pytest.skip("symlink creation unavailable on this host")


def test_verify_calibrated_profile_refuses_symlinked_profile(calibrated, tmp_path, monkeypatch):
    link = tmp_path / "profile-link.json"
    maybe_symlink(link, calibrated["profile_path_object"])
    monkeypatch.setattr(v3_dc, "_HOST_PATH_RESOLVER",
                        lambda path: str(link) if path == LOGICAL_PROFILE_PATH else path)
    spec = v3_dc.calibrated_capability(capability_plan(calibrated))
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.verify_calibrated_profile(spec)


# ------------------------------------------------------ artifact verification


def test_verify_calibrated_artifact_returns_identity(calibrated):
    monkey_root = v3_dc.CALIBRATION_SOURCE_ROOT
    identity = v3_dc.verify_calibrated_artifact(
        monkey_root + "/" + BASENAME, calibrated["sha256"], calibrated["size"])
    assert len(identity) == 4


def test_verify_calibrated_artifact_refuses_digest_mismatch(calibrated):
    calibrated["artifact"].write_bytes(calibrated["payload"][:-1] + b"X")
    with pytest.raises(v3_dc.ConfigError, match="calibrated_artifact_digest_mismatch"):
        v3_dc.verify_calibrated_artifact(v3_dc.CALIBRATION_SOURCE_ROOT + "/" + BASENAME,
                                          calibrated["sha256"], calibrated["size"])


def test_verify_calibrated_artifact_refuses_size_mismatch(calibrated):
    with pytest.raises(v3_dc.ConfigError, match="calibrated_artifact_size_mismatch"):
        v3_dc.verify_calibrated_artifact(v3_dc.CALIBRATION_SOURCE_ROOT + "/" + BASENAME,
                                         calibrated["sha256"], calibrated["size"] - 1)


def test_verify_calibrated_artifact_refuses_symlink_source(calibrated, tmp_path, monkeypatch):
    link = tmp_path / ("link-" + BASENAME)
    maybe_symlink(link, calibrated["artifact"])
    logical = LOGICAL_SOURCE_ROOT + "/" + BASENAME
    monkeypatch.setattr(v3_dc, "_HOST_PATH_RESOLVER",
                        lambda path: str(link) if path == logical else path)
    with pytest.raises(v3_dc.ConfigError):
        v3_dc.verify_calibrated_artifact(logical, calibrated["sha256"], calibrated["size"])


def test_verify_calibrated_artifact_refuses_hardlinked_source(calibrated, tmp_path, monkeypatch):
    extra = tmp_path / ("extra-" + BASENAME)
    try:
        os.link(str(calibrated["artifact"]), str(extra))
    except OSError:
        pytest.skip("hardlink creation unavailable on this host")
    logical = LOGICAL_SOURCE_ROOT + "/" + BASENAME
    monkeypatch.setattr(v3_dc, "_HOST_PATH_RESOLVER",
                        lambda path: str(extra) if path == logical else path)
    with pytest.raises(v3_dc.ConfigError, match="calibrated_artifact_not_single_regular_file"):
        v3_dc.verify_calibrated_artifact(logical, calibrated["sha256"], calibrated["size"])


# ------------------------------------------------------------- derivation


def test_derive_calibrated_binding_uses_the_verified_profile(calibrated):
    verified = v3_dc.verify_calibrated_profile(
        v3_dc.calibrated_capability(capability_plan(calibrated)))
    derived = v3_dc.derive_calibrated_binding(verified, calibrated_original())
    assert derived == binding()


def test_derive_calibrated_binding_refuses_shadowing_original_mounts(calibrated):
    shadowed = {"calibration_path": "/data/neo-glm53/" + BASENAME,
                "calibration_digest": CALIBRATION_DIGEST}
    with pytest.raises(v3_dc.ConfigError, match="calibrated_mount_shadows_original"):
        v3_dc.derive_calibrated_binding(shadowed, calibrated_original())


# --------------------------------------------- pre-mutation recheck + state


def state_with_capability(calibrated, identity):
    return {"plan": capability_plan(calibrated),
            "calibration_source": v3_dc.CALIBRATION_SOURCE_ROOT + "/" + BASENAME,
            "calibration_destination": DESTINATION,
            "calibration_basename": BASENAME,
            "calibration_source_identity": list(identity)}


def test_verify_calibrated_source_passes_unchanged_files(calibrated):
    identity = v3_dc.verify_calibrated_artifact(
        v3_dc.CALIBRATION_SOURCE_ROOT + "/" + BASENAME, calibrated["sha256"], calibrated["size"])
    binding_out = life.verify_calibrated_source(state_with_capability(calibrated, identity))
    assert binding_out["destination"] == DESTINATION


def test_verify_calibrated_source_refuses_mutated_artifact(calibrated):
    identity = v3_dc.verify_calibrated_artifact(
        v3_dc.CALIBRATION_SOURCE_ROOT + "/" + BASENAME, calibrated["sha256"], calibrated["size"])
    calibrated["artifact"].write_bytes(calibrated["payload"][:-1] + b"X")
    with pytest.raises(Failure, match="calibrated_artifact_digest_mismatch"):
        life.verify_calibrated_source(state_with_capability(calibrated, identity))


def test_verify_calibrated_source_refuses_identity_change(calibrated):
    identity = v3_dc.verify_calibrated_artifact(
        v3_dc.CALIBRATION_SOURCE_ROOT + "/" + BASENAME, calibrated["sha256"], calibrated["size"])
    state = state_with_capability(calibrated, identity)
    state["calibration_source_identity"] = list(identity[:2]) + [1, 2]
    with pytest.raises(Failure, match="calibrated source identity changed"):
        life.verify_calibrated_source(state)


def test_binding_from_state_refuses_capability_without_evidence(calibrated):
    state = {"plan": capability_plan(calibrated)}
    with pytest.raises(Failure, match="missing journaled binding"):
        life.calibrated_binding_from_state(state)


def test_binding_from_state_is_none_without_capability():
    assert life.calibrated_binding_from_state({"plan": {}}) is None


# ------------------------------------------------- probe exclusion + rollback


def test_probe_request_never_carries_the_addition():
    original = calibrated_original()
    probe = v3_dc.build_artifact_probe_request(original, V3_CANDIDATE_IMAGE, "probe")
    for entry in probe["HostConfig"].get("Mounts") or []:
        assert entry["Target"] != DESTINATION
    assert DESTINATION not in json.dumps(probe)


def test_rollback_parity_on_original_is_unchanged():
    original = calibrated_original()
    body = v3_dc.build_create_request(original, V3_CANDIDATE_IMAGE, "candidate", None, binding())
    candidate = calibrated_create(original, body)
    # The restored original never carries the candidate-only addition.
    v3_dc.assert_candidate_parity(original, deepcopy(original), original["Image"])


# ------------------------------------------------------- reconcile (crash path)


class ReconcileAPI:
    def __init__(self, docs):
        self.docs = docs

    def inspect(self, identity):
        return deepcopy(self.docs[identity])


def reconcile_fixture(tmp_path, calibrated, with_binding):
    original = calibrated_original()
    plan = capability_plan(calibrated)
    evidence = None
    if with_binding:
        identity = v3_dc.verify_calibrated_artifact(
            v3_dc.CALIBRATION_SOURCE_ROOT + "/" + BASENAME, calibrated["sha256"], calibrated["size"])
        evidence = {"source": v3_dc.CALIBRATION_SOURCE_ROOT + "/" + BASENAME,
                    "destination": DESTINATION, "basename": BASENAME,
                    "identity": list(identity)}
    journal = life.Journal.create(tmp_path / "attempt", plan, original, calibrated=evidence)
    state = journal.read()
    nonce = "ab" * 16
    create_name = life.candidate_create_name(state["candidate_name"], nonce)
    bind = life.calibrated_binding_from_state(state) if with_binding else None
    request = v3_dc.build_create_request(original, state["candidate_image_id"], create_name, None, bind)
    candidate = calibrated_create(original, request, name=create_name)
    candidate["Image"] = state["candidate_image_id"]
    record = {"schema": 1, "attempt_id": state["attempt_id"], "resource_nonce": nonce,
              "name": create_name, "final_name": state["candidate_name"],
              "image_id": state["candidate_image_id"], "owner": None,
              "container_id": candidate["Id"], "phase": "CREATE_INTENT",
              "request_sha256": life._request_digest(request),
              "cleanup_admissible": False, "cleanup_complete": False}
    life.private_json(journal.directory / "candidate-create-intent.json", record)
    api = ReconcileAPI({original["Id"]: original, candidate["Id"]: candidate})
    return api, journal


def test_reconcile_verifies_the_calibrated_candidate(tmp_path, calibrated):
    api, journal = reconcile_fixture(tmp_path, calibrated, with_binding=True)
    record = life.reconcile_candidate_create(api, journal, error=None)
    assert record["reconciliation"] == "verified_cleanup_only"


def test_reconcile_refuses_capability_without_journaled_binding(tmp_path, calibrated):
    api, journal = reconcile_fixture(tmp_path, calibrated, with_binding=False)
    record = life.reconcile_candidate_create(api, journal, error=None)
    assert record["reconciliation"] == "unresolved"
    assert record["cleanup_admissible"] is False


def test_journaled_calibration_fields_are_immutable(tmp_path, calibrated):
    _, journal = reconcile_fixture(tmp_path, calibrated, with_binding=True)
    state = journal.read()
    assert state["calibration_destination"] == DESTINATION
    with pytest.raises(Failure, match="immutable journal field"):
        journal.update(calibration_source="/data/other/source.bin")
    with pytest.raises(Failure, match="immutable journal field"):
        journal.update(calibration_source_identity=[1, 2, 3, 4])


def test_journal_without_capability_has_no_calibration_fields(tmp_path, calibrated):
    original = calibrated_original()
    plan = {"attempt_id": "offline-attempt-0001", "original_id": original["Id"],
            "original_image_id": original["Image"],
            "candidate_image_id": "sha256:" + "b" * 64,
            "base_url": "http://127.0.0.1:18111",
            "ingress_base_url": "http://neo-k3-ingress:8080", "model": "fixture-model",
            "mode": "production"}
    journal = life.Journal.create(tmp_path / "plain-attempt", plan, original)
    state = journal.read()
    assert "calibration_source" not in state
    assert "calibration_destination" not in state
