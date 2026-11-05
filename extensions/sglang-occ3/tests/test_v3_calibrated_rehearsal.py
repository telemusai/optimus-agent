"""Offline tests for the optional calibrated capability rehearsal scenario.

The capability arrives as a bounded exact-field JSON input, is validated
through the committed capability schema, and adds exactly one exit-fixture
scenario. The default ten scenarios stay byte-identical and never touch the
pooled artifact. No Docker daemon, no GPU, no network.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import controller as ctl
from deploy.v3 import docker_config as v3_dc
from deploy.v3 import lifecycle as life
from deploy.v3 import rehearsal as rh
from deploy.v3.lifecycle import Failure, REHEARSAL_LABEL, validate_plan

from test_v3_lifecycle import (CANDIDATE, ORIGINAL, FakeDocker, original_doc,
                               approval, plan as lifecycle_plan, prepared)


BASENAME = "calibrated-dual-ridge-9a7394dea73cf8d3.bin"
DESTINATION = "/occ3/calibration/" + BASENAME
CALIBRATION_DIGEST = "9a7394dea73cf8d33fd651bf8b43d62d739c94d0940bd6f4ae494fd83ba0be90"
LOGICAL_SOURCE_ROOT = "/pooled-calibration"
LOGICAL_PROFILE_PATH = "/staged/native-profile-input.json"
SOURCE = LOGICAL_SOURCE_ROOT + "/" + BASENAME
# The real rehearsal-mode admission receipt (lifecycle.Services, rehearsal
# branch); the guard pins this shape and the drift test keeps it honest.
RECEIPT = {"ok": True, "scope": "disposable_faults_only_not_gate8"}


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
    real = {LOGICAL_PROFILE_PATH: profile_path, SOURCE: artifact}
    monkeypatch.setattr(v3_dc, "_HOST_PATH_RESOLVER", real.__getitem__)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload),
            "profile_path": LOGICAL_PROFILE_PATH,
            "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "profile_bytes": len(profile_bytes),
            "artifact": artifact, "payload": payload}


def input_document(calibrated):
    return {"calibrated_artifact": {"sha256": calibrated["sha256"],
                                     "size": calibrated["size"]},
            "calibrated_profile_evidence": {"path": calibrated["profile_path"],
                                             "sha256": calibrated["profile_sha256"],
                                             "bytes": calibrated["profile_bytes"]}}


# ------------------------------------------------------------ input loading


def test_load_calibrated_input_accepts_the_exact_document(calibrated, tmp_path):
    path = tmp_path / "calibrated-input.json"
    path.write_text(json.dumps(input_document(calibrated)), encoding="utf-8")
    capability = rh.load_calibrated_input(path)
    assert capability == {"sha256": calibrated["sha256"], "size": calibrated["size"],
                          "profile_path": LOGICAL_PROFILE_PATH,
                          "profile_sha256": calibrated["profile_sha256"],
                          "profile_bytes": calibrated["profile_bytes"]}


@pytest.mark.parametrize("document", [
    {"calibrated_artifact": {"sha256": "a" * 64, "size": 10}},
    {"calibrated_artifact": {"sha256": "a" * 64, "size": 10},
     "calibrated_profile_evidence": {"path": "/p.json", "sha256": "b" * 64, "bytes": 8},
     "extra": 1},
    {"calibrated_artifact": {"sha256": "a" * 64, "size": 10},
     "calibrated_profile_evidence": "not-a-dict"},
    ["not", "a", "dict"],
])
def test_load_calibrated_input_refuses_unknown_or_missing_fields(document, tmp_path):
    path = tmp_path / "calibrated-input.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Failure, match="invalid calibrated input document"):
        rh.load_calibrated_input(path)


def test_load_calibrated_input_refuses_missing_file(tmp_path):
    with pytest.raises(Failure, match="calibrated input unreadable"):
        rh.load_calibrated_input(tmp_path / "absent-calibrated-input.json")


def test_load_calibrated_input_refuses_malformed_json(tmp_path):
    path = tmp_path / "calibrated-input.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(Failure, match="invalid calibrated input document"):
        rh.load_calibrated_input(path)


@pytest.mark.parametrize("bad", [
    {"sha256": "a" * 63, "size": 10},
    {"sha256": "a" * 64, "size": 0},
    {"sha256": "a" * 64, "size": v3_dc.CALIBRATION_ARTIFACT_MAX_BYTES + 1},
    {"sha256": "a" * 64, "size": True},
])
def test_load_calibrated_input_refuses_schema_violations(calibrated, tmp_path, bad):
    document = input_document(calibrated)
    document["calibrated_artifact"] = bad
    path = tmp_path / "calibrated-input.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Failure, match="invalid calibrated input capability"):
        rh.load_calibrated_input(path)


# ------------------------------------------------------- scenario assembly


def test_default_scenarios_are_exactly_ten_without_capability():
    scenarios = rh.rehearsal_scenarios(None)
    assert scenarios == list(rh.SCENARIOS)
    assert len(scenarios) == 10
    assert not any(scenario.get("calibrated") for scenario in scenarios)
    names = [scenario["name"] for scenario in scenarios]
    assert len(set(names)) == len(names)


def test_capability_input_appends_the_exit_scenario(calibrated):
    capability = v3_dc.calibrated_capability(
        dict(input_document(calibrated),
             static_input={"sha256": calibrated["profile_sha256"]},
             native_validator={"static_input_sha256": calibrated["profile_sha256"]}))
    scenarios = rh.rehearsal_scenarios(capability)
    assert len(scenarios) == 11
    assert scenarios[-1] == {"name": "calibrated-bind-startup-exit", "image": "exit",
                             "calibrated": True}
    names = [scenario["name"] for scenario in scenarios]
    assert len(set(names)) == len(names)


def test_scenario_assembly_enforces_exit_image_and_uniqueness(calibrated, monkeypatch):
    capability = v3_dc.calibrated_capability(
        dict(input_document(calibrated),
             static_input={"sha256": calibrated["profile_sha256"]},
             native_validator={"static_input_sha256": calibrated["profile_sha256"]}))
    monkeypatch.setattr(rh, "CALIBRATED_SCENARIO",
                        {"name": "calibrated-bind-startup-exit", "image": "healthy",
                         "calibrated": True})
    with pytest.raises(Failure, match="exit fixture"):
        rh.rehearsal_scenarios(capability)
    monkeypatch.setattr(rh, "CALIBRATED_SCENARIO",
                        {"name": "startup-exit", "image": "exit", "calibrated": True})
    with pytest.raises(Failure, match="unique"):
        rh.rehearsal_scenarios(capability)


# ------------------------------------------------------------ plan wiring


def test_calibrated_plan_keys_none_changes_nothing():
    assert rh.calibrated_plan_keys(None) == {}
    plan = {"attempt_id": "x"}
    plan.update(rh.calibrated_plan_keys(None))
    assert plan == {"attempt_id": "x"}


def test_calibrated_plan_keys_maps_the_four_fields(calibrated):
    capability = v3_dc.calibrated_capability(
        dict(input_document(calibrated),
             static_input={"sha256": calibrated["profile_sha256"]},
             native_validator={"static_input_sha256": calibrated["profile_sha256"]}))
    keys = rh.calibrated_plan_keys(capability)
    assert set(keys) == {"calibrated_artifact", "calibrated_profile_evidence",
                         "static_input", "native_validator"}
    assert keys["static_input"] == {"sha256": calibrated["profile_sha256"]}
    assert keys["native_validator"] == {"static_input_sha256": calibrated["profile_sha256"]}
    plan = lifecycle_plan(**keys)
    plan = validate_plan(plan)
    assert v3_dc.calibrated_capability(plan)["sha256"] == calibrated["sha256"]


# --------------------------------------------------- prepare refusals (offline)


class MountDocker(FakeDocker):
    """FakeDocker materializing declared HostConfig.Mounts into top-level records."""
    def create(self, body, name):
        identity = CANDIDATE
        self._before("create", identity)
        document = copy.deepcopy(self.docs[ORIGINAL])
        document.update(Id=identity, Image=body["Image"], Name="/" + name)
        document["Config"] = {k: copy.deepcopy(v) for k, v in body.items()
                              if k not in ("HostConfig", "NetworkingConfig")}
        document["HostConfig"] = copy.deepcopy(body["HostConfig"])
        document["State"] = {"Running": False, "Restarting": False, "ExitCode": 0,
                             "Status": "created", "StartedAt": ""}
        document["Mounts"] = [{"Type": "bind", "Source": entry["Source"],
                               "Destination": entry["Target"],
                               "Mode": "ro" if entry.get("ReadOnly") else "",
                               "RW": not entry.get("ReadOnly", False),
                               "Propagation": "rprivate"}
                              for entry in body["HostConfig"].get("Mounts") or []]
        self.docs[identity] = document
        return identity


def rehearsal_prepared(tmp_path, calibrated, with_capability):
    api = MountDocker()
    owner = "ab12cd34ef567890"
    doc = api.docs[ORIGINAL]
    doc["Name"] = "/occ3-v3-rehearsal-" + owner + "-original"
    doc["Config"]["Labels"] = {REHEARSAL_LABEL: owner}
    changes = {"mode": "rehearsal", "rehearsal_owner": owner}
    if with_capability:
        capability = v3_dc.calibrated_capability(
            dict(input_document(calibrated),
                 static_input={"sha256": calibrated["profile_sha256"]},
                 native_validator={"static_input_sha256": calibrated["profile_sha256"]}))
        changes.update(rh.calibrated_plan_keys(capability))
    api, journal, services = prepared(tmp_path, api=api, **changes)
    life.consume_authorization(journal, approval(journal))
    return api, journal, services


def test_prepare_refuses_absent_profile_before_any_create(tmp_path, calibrated, monkeypatch):
    monkeypatch.setattr(v3_dc, "_HOST_PATH_RESOLVER",
                        lambda path: "/nonexistent/profile.json" if path == LOGICAL_PROFILE_PATH else path)
    with pytest.raises(v3_dc.ConfigError):
        rehearsal_prepared(tmp_path, calibrated, with_capability=True)
    # The refusal happens before any container is created; only the original exists.
    # (prepare raised before Journal.create wrote anything, so no create event exists.)


def test_prepare_refuses_wrong_artifact_digest_before_any_create(tmp_path, calibrated, monkeypatch):
    calibrated["artifact"].write_bytes(calibrated["payload"][:-1] + b"X")
    with pytest.raises(v3_dc.ConfigError, match="calibrated_artifact_digest_mismatch"):
        rehearsal_prepared(tmp_path, calibrated, with_capability=True)


# ------------------------------------------------- controller admission gating


def test_controller_journals_admission_for_plain_rehearsal(tmp_path, calibrated):
    api, journal, services = rehearsal_prepared(tmp_path, calibrated, with_capability=False)
    ctl.Controller(api, journal, services=services, guard=lambda: None).run()
    state = journal.read()
    assert state["validation_admission"] == {"ok": True, "scope": "offline-fixture-contract"}


def test_controller_retains_admission_write_for_calibrated_rehearsal(tmp_path, calibrated):
    api, journal, services = rehearsal_prepared(tmp_path, calibrated, with_capability=True)
    ctl.Controller(api, journal, services=services, guard=lambda: None).run()
    state = journal.read()
    # The pre-mutation admission write is unconditional for every plan; the
    # journaled receipt is exactly what the services layer returned.
    assert state["validation_admission"] == services.pre_mutation_admission(state)
    assert state["calibration_destination"] == DESTINATION
    assert sum(op == "create" for op, _ in api.events) == 1


def test_rehearsal_admission_receipt_pin_matches_real_services():
    # Drift guard: the pinned receipt must equal what the production
    # lifecycle.Services actually returns in rehearsal mode (offline
    # constructible; the rehearsal branch performs no I/O).
    services = life.Services(original_doc(), {"mode": "rehearsal"})
    assert services.pre_mutation_admission({"plan": {"mode": "rehearsal"}}) == rh.CALIBRATED_ADMISSION_RECEIPT
    assert rh.CALIBRATED_ADMISSION_RECEIPT == RECEIPT


# ------------------------------------------------------- sanitized echo


def echo_candidate():
    return {"Id": "c" * 64, "Image": "sha256:" + "b" * 64,
            "Name": "/occ3-v3-rehearsal-ab12cd34ef567890-candidate",
            "Config": {"Cmd": ["--api-key", rh.FIXTURE_KEY], "Env": ["A=1"]},
            "HostConfig": {"Mounts": [{"Type": "bind", "Source": SOURCE,
                                        "Target": DESTINATION, "ReadOnly": True}]},
            "Mounts": [{"Type": "bind", "Source": SOURCE, "Destination": DESTINATION,
                        "Mode": "", "RW": False, "Propagation": "rprivate"}],
            "State": {"Status": "exited", "ExitCode": 7, "Running": False}}


def test_persist_calibrated_echo_is_sanitized(tmp_path):
    path = rh.persist_calibrated_echo(tmp_path, echo_candidate())
    text = path.read_text(encoding="utf-8")
    assert "--api-key" not in text
    assert rh.FIXTURE_KEY not in text
    echo = json.loads(text)
    assert echo["schema"] == "occ3-calibrated-rehearsal-echo/v1"
    assert echo["source_document"] == "candidate-inspect"
    assert echo["declared_hostconfig_mounts"] == [
        {"Type": "bind", "Source": SOURCE, "Target": DESTINATION, "ReadOnly": True}]
    assert echo["resolved_actual_mounts"][0]["Destination"] == DESTINATION
    assert echo["state"] == {"Status": "exited", "ExitCode": 7, "Running": False}
    assert "Config" not in echo and "Env" not in echo


# ------------------------------------------- capability evidence assertions


def capability_state(**changes):
    state = {"calibration_source": SOURCE, "calibration_destination": DESTINATION,
             "calibration_basename": BASENAME, "gates": {}, "restart_count": 0,
             "validation_admission": dict(RECEIPT)}
    state.update(changes)
    return state


def capability_plan(calibrated):
    return dict(input_document(calibrated),
                static_input={"sha256": calibrated["profile_sha256"]},
                native_validator={"static_input_sha256": calibrated["profile_sha256"]})


def assert_evidence(calibrated, state=None, candidate=None, current=None, plan=None,
                    journal_dir=None):
    if journal_dir is None:
        with tempfile.TemporaryDirectory() as directory:
            return assert_evidence(calibrated, state, candidate, current, plan, Path(directory))
    return rh.assert_calibrated_capability_evidence(
        state or capability_state(), {"Mounts": []}, current or {"Mounts": []},
        candidate or echo_candidate(), plan or capability_plan(calibrated), journal_dir)


def test_assert_evidence_accepts_the_exact_shape(calibrated):
    evidence = assert_evidence(calibrated)
    assert evidence["calibration_source"] == SOURCE
    assert evidence["calibration_destination"] == DESTINATION
    assert evidence["declared_exactly_one_readonly_bind"] is True
    assert evidence["pre_mutation_admission_retained"] is True
    assert evidence["post_start_validation_absent"] is True


@pytest.mark.parametrize("admission", [
    None,
    {"ok": False, "scope": "disposable_faults_only_not_gate8"},
    {"ok": True, "scope": "native_latent_validated"},
    {"ok": True},
    "not-a-dict",
])
def test_assert_evidence_requires_the_pre_mutation_admission_receipt(calibrated, admission):
    with pytest.raises(Failure, match="admission receipt"):
        assert_evidence(calibrated, state=capability_state(validation_admission=admission))


@pytest.mark.parametrize("gate", ["service_initial", "service_final", "live_before",
                                  "live_after", "native_initial", "native_final", "restart"])
def test_assert_evidence_refuses_any_journaled_gate(calibrated, gate):
    with pytest.raises(Failure, match="validation gate"):
        assert_evidence(calibrated, state=capability_state(gates={gate: {"ok": True}}))


@pytest.mark.parametrize("phase", ["WAIT_NATIVE_INITIAL", "WAIT_NATIVE_FINAL",
                                   "RESTART_REQUESTED", "POST_RESTART",
                                   "GATES_COMPLETE", "COMPLETE"])
def test_assert_evidence_refuses_forbidden_validation_phases(calibrated, phase):
    with pytest.raises(Failure, match="forbidden validation phase"):
        assert_evidence(calibrated, state=capability_state(phase=phase))


@pytest.mark.parametrize("evidence_file", list(rh.CALIBRATED_FORBIDDEN_FILES))
def test_assert_evidence_refuses_validation_evidence_files(calibrated, tmp_path, evidence_file):
    (tmp_path / evidence_file).write_text("{}", encoding="utf-8")
    with pytest.raises(Failure, match="left validation evidence"):
        assert_evidence(calibrated, journal_dir=tmp_path)


def test_assert_evidence_refuses_restart_request_even_with_zero_count(calibrated):
    with pytest.raises(Failure, match="validation restart"):
        assert_evidence(calibrated, state=capability_state(
            restart_request={"nonce": "n", "candidate_id": "c" * 64, "epoch": "e"}))


def test_assert_evidence_refuses_restart_progression(calibrated):
    with pytest.raises(Failure, match="validation restart"):
        assert_evidence(calibrated, state=capability_state(restart_count=1))


def test_assert_evidence_refuses_persistence_record(calibrated):
    with pytest.raises(Failure, match="persistence record"):
        assert_evidence(calibrated, state=capability_state(
            persistence={"container_id": "a" * 64}))


def test_assert_evidence_refuses_progression_behind_a_recovery_phase(calibrated, tmp_path):
    # A recovery-phase overwrite must not conceal durable progression: the
    # phase is a legitimate RESTORED, but the gate and checkpoint evidence fail.
    (tmp_path / "checkpoint-initial.json").write_text("{}", encoding="utf-8")
    state = capability_state(phase="RESTORED", gates={"live_before": {"ok": True}})
    with pytest.raises(Failure):
        assert_evidence(calibrated, state=state, journal_dir=tmp_path)


def test_assert_evidence_accepts_legitimate_recovery_phases(calibrated):
    for phase in ("RESTORED", "MUTATING", "STARTING", "NOOP"):
        evidence = assert_evidence(calibrated, state=capability_state(phase=phase))
        assert evidence["post_start_validation_absent"] is True


def test_assert_evidence_refuses_wrong_journal_binding(calibrated):
    with pytest.raises(Failure, match="journal binding"):
        assert_evidence(calibrated, state=capability_state(
            calibration_source="/pooled-calibration/other.bin"))


@pytest.mark.parametrize("declared", [
    [], [{"Type": "bind", "Source": SOURCE, "Target": DESTINATION, "ReadOnly": False}],
    [{"Type": "bind", "Source": SOURCE, "Target": DESTINATION, "ReadOnly": True}] * 2,
])
def test_assert_evidence_refuses_wrong_declaration(calibrated, declared):
    candidate = echo_candidate()
    candidate["HostConfig"]["Mounts"] = declared
    with pytest.raises(Failure, match="declaration"):
        assert_evidence(calibrated, candidate=candidate)


@pytest.mark.parametrize("record", [
    {"Mode": "rw"}, {"RW": True}, {"Source": "/pooled-calibration/other.bin"},
    {"Propagation": "shared"},
])
def test_assert_evidence_refuses_unbounded_resolved_records(calibrated, record):
    candidate = echo_candidate()
    candidate["Mounts"][0].update(record)
    with pytest.raises(Failure, match="resolved record|resolved mounts"):
        assert_evidence(calibrated, candidate=candidate)


def test_assert_evidence_refuses_extra_resolved_mounts(calibrated):
    candidate = echo_candidate()
    candidate["Mounts"].append(dict(candidate["Mounts"][0]))
    with pytest.raises(Failure, match="resolved mounts"):
        assert_evidence(calibrated, candidate=candidate)


def test_assert_evidence_refuses_restored_original_with_mounts(calibrated):
    with pytest.raises(Failure, match="restored disposable original"):
        assert_evidence(calibrated, current={"Mounts": [dict(echo_candidate()["Mounts"][0])]})
