"""Wire-parity pins for the validator's per-boot RuntimeProfile derivation.

The live icalib window (candidate window icalib-20261029T032600Z) refused every
honest report with report_profile_derivation_mismatch: the served profile digest
covers the FULL RuntimeProfile fieldset (occ3_sglang.latent_types.RuntimeProfile,
which always carries calibration_path/calibration_digest, empty for the
per-message method) plus the observed runtime_epoch/runtime_generation, while the
validator derived a 14-field per-message subset.

Every expected value here comes from the real wire function
(RuntimeProfile.binding_digest), never a re-implementation, so the pin cannot
silently diverge from the digest the candidate serves again.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

from occ3_sglang.latent_types import RuntimeProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import lifecycle as life
from deploy.v3 import native_validator as nv
from deploy.v3.live_validation import NativeHandoffValidator, VALIDATOR_KEY
from occ3_sglang import native_profile as registry

PER_MESSAGE = registry.PER_MESSAGE_METHOD
CALIBRATED = registry.CALIBRATED_METHOD
CALIBRATION_PATH = "/occ3/calibration/synthetic-frozen-artifact.bin"
CALIBRATION_DIGEST = "5" * 64
ORIGINAL = "1" * 64
CANDIDATE = "c" * 64
IMAGE = "sha256:" + "a" * 64
CANDIDATE_IMAGE = "sha256:" + "b" * 64
PIN_SCHEMA = "occ3.admission.pin.v1"
FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "native_phase_suite_v1.json").read_text())
EXAMPLE = FIXTURE["example"]
PINNED = FIXTURE["admission_pinned_inputs"]
ATTEMPT = EXAMPLE["attempt"]["attempt_id"]
EPOCH_BEFORE = EXAMPLE["attempt"]["candidate_started_at_before"]
CHALLENGE_BEFORE = EXAMPLE["challenge_before"]["challenge"]
PLAN_FIELDS = ("dtype", "layout_strategy_digest")


def original_doc():
    return {"Id": ORIGINAL, "Image": IMAGE, "Name": "/fixture-service",
            "Config": {"Image": IMAGE, "Hostname": "fixture-host",
                       "Entrypoint": ["python3"],
                       "Cmd": ["-m", "sglang.launch_server", "--api-key", "k"],
                       "Env": ["A=1"], "WorkingDir": "/w", "User": "", "Labels": {}},
            "HostConfig": {"NetworkMode": "bridge", "Runtime": "runc", "Mounts": [],
                           "Binds": None, "RestartPolicy": {"Name": "always"},
                           "AutoRemove": False, "PortBindings": {}},
            "Mounts": [], "NetworkSettings": {"Networks": {"bridge": {}}},
            "State": {"Running": True, "Restarting": False, "ExitCode": 0,
                      "Status": "running", "StartedAt": "epoch-original"}}


def plan(**changes):
    value = {"attempt_id": ATTEMPT, "original_id": ORIGINAL, "original_image_id": IMAGE,
             "candidate_image_id": CANDIDATE_IMAGE, "model": "SYNTHETIC-model",
             "base_url": "http://127.0.0.1:18111", "ingress_base_url": "http://127.0.0.1:18112",
             "startup_timeout_s": 5, "deployment_timeout_s": 20, "rollback_timeout_s": 5,
             "proof_timeout_s": 5, "controller_timeout_s": 5, "supervisor_lease_s": 5,
             "poll_interval_s": 0.005, "request_timeout_s": 5, "stop_timeout_s": 1,
             "probe_timeout_s": 1, "validation_timeout_s": 5}
    value.update(changes)
    return life.validate_plan(value)


def native_plan(block=None):
    return plan(validator=VALIDATOR_KEY,
                native_validator=copy.deepcopy(block if block is not None else PINNED))


def calibrated_pinned():
    """Synthetic artifact-requiring method admission (registry-advertised id)."""
    block = copy.deepcopy(PINNED)
    block["method_id"] = CALIBRATED
    block["static_input_sha256"] = "7" * 64
    profile = block["admission_template"]["source_profile"]
    profile["method"] = CALIBRATED
    profile["calibration_path"] = CALIBRATION_PATH
    profile["calibration_digest"] = CALIBRATION_DIGEST
    identity = block["admission_template"]["identity"]
    identity["latent_method"] = CALIBRATED
    block["admission_template"]["fingerprint"] = "sha256:" + nv._canonical_sha256(identity)
    return block


def observed():
    return copy.deepcopy(EXAMPLE["report_before"]["observed"])


def wire_digest(profile, incarnation):
    """The digest the candidate serves: the real RuntimeProfile binding digest."""
    static = {name: value for name, value in profile.items() if name not in PLAN_FIELDS}
    return RuntimeProfile(**static, runtime_epoch=incarnation["runtime_epoch"],
                          runtime_generation=incarnation["runtime_generation"]).binding_digest


def per_message_profile():
    return copy.deepcopy(PINNED["admission_template"]["source_profile"])


def calibrated_profile():
    return copy.deepcopy(calibrated_pinned()["admission_template"]["source_profile"])


def make_journal(tmp_path, plan_value):
    return life.Journal.create(tmp_path / "attempt", plan_value, original_doc())


def start_journal(journal):
    journal.update(candidate_id=CANDIDATE, candidate_started=True, candidate_epoch=EPOCH_BEFORE,
                   gates={"service_initial": {"ok": True}, "native_initial": {"ok": True}})


def publish(journal, report, digest, static_input):
    document = copy.deepcopy(report)
    document["completed_at"] = int(time.time()) + 30
    life.private_json(journal.directory / "native-phase-suite-before_restart.json", document)
    life.private_json(journal.directory / "admission-pin-before_restart.json",
                      {"schema": PIN_SCHEMA, "phase": "before_restart", "attempt_id": ATTEMPT,
                       "static_input_sha256": static_input,
                       "runtime_epoch": document["observed"]["runtime_epoch"],
                       "runtime_generation": document["observed"]["runtime_generation"],
                       "profile_digest": digest, "written_at": int(time.time())},
                      exclusive=True)


def run_before(journal):
    state = journal.read()
    return NativeHandoffValidator(state["plan"], journal, state, timeout=5.0,
                                  challenge_factory=lambda: CHALLENGE_BEFORE).before_restart()


# -- the derivation must be the wire fieldset, for both registered shapes -----
def test_per_message_derivation_equals_the_wire_binding_digest():
    incarnation = observed()
    profile = per_message_profile()
    assert nv._derived_profile_digest(profile, incarnation) == wire_digest(profile, incarnation)


def test_calibrated_derivation_equals_the_wire_binding_digest():
    incarnation = observed()
    profile = calibrated_profile()
    assert nv._derived_profile_digest(profile, incarnation) == wire_digest(profile, incarnation)


def test_derivation_fieldset_is_the_wire_fieldset_for_every_method():
    wire_fields = set(RuntimeProfile.__dataclass_fields__)
    incarnation = observed()
    for profile in (per_message_profile(), calibrated_profile()):
        assert nv._method_descriptor(profile["method"])[2] is not None
        assert set(nv._derivation_value(profile, incarnation)) == wire_fields


def test_per_message_derivation_carries_the_empty_calibration_reference():
    # The wire RuntimeProfile always carries the pair; a per-message profile that
    # names none is the empty pair, never an absent pair.
    value = nv._derivation_value(per_message_profile(), observed())
    assert (value["calibration_path"], value["calibration_digest"]) == ("", "")


def test_profile_digest_is_per_boot_volatile_for_the_wire_fieldset():
    profile = per_message_profile()
    base = observed()
    first = wire_digest(profile, base)
    assert nv._derived_profile_digest(profile, base) == first
    for changed in ({"runtime_epoch": "f" * 32}, {"runtime_generation": 42}):
        other = wire_digest(profile, dict(base, **changed))
        assert other != first
        assert nv._derived_profile_digest(profile, dict(base, **changed)) == other


# -- the honest report must be admitted end to end ---------------------------
def test_before_phase_admits_the_honest_per_message_wire_digest(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    start_journal(journal)
    profile = per_message_profile()
    digest = wire_digest(profile, observed())
    report = copy.deepcopy(EXAMPLE["report_before"])
    report["binding"]["profile_digest"] = digest
    publish(journal, report, digest, nv.PIN_STATIC_INPUT_SHA256)
    result = run_before(journal)
    assert result["ok"] is True, result["code"]


def test_before_phase_admits_the_honest_calibrated_wire_digest(tmp_path):
    block = calibrated_pinned()
    journal = make_journal(tmp_path, native_plan(block))
    start_journal(journal)
    profile = copy.deepcopy(block["admission_template"]["source_profile"])
    digest = wire_digest(profile, observed())
    report = copy.deepcopy(EXAMPLE["report_before"])
    report["binding"]["method_id"] = CALIBRATED
    report["binding"]["identity_fingerprint"] = block["admission_template"]["fingerprint"]
    report["binding"]["profile_digest"] = digest
    publish(journal, report, digest, block["static_input_sha256"])
    result = run_before(journal)
    assert result["ok"] is True, result["code"]


# -- negatives: the uniform fieldset must still bind the artifact reference ---
def test_swapped_calibration_digest_is_not_admitted(tmp_path):
    # A self-consistent fabricated pair (pin and binding carry the swapped
    # artifact's digest) is still refused: the derivation receipt from the
    # admission's own profile is what binds the frozen artifact.
    block = calibrated_pinned()
    journal = make_journal(tmp_path, native_plan(block))
    start_journal(journal)
    profile = copy.deepcopy(block["admission_template"]["source_profile"])
    digest = wire_digest(profile, observed())
    swapped = wire_digest(dict(profile, calibration_digest="6" * 64), observed())
    assert swapped != digest
    report = copy.deepcopy(EXAMPLE["report_before"])
    report["binding"]["method_id"] = CALIBRATED
    report["binding"]["identity_fingerprint"] = block["admission_template"]["fingerprint"]
    report["binding"]["profile_digest"] = swapped
    publish(journal, report, swapped, block["static_input_sha256"])
    result = run_before(journal)
    assert result["ok"] is False and result["code"] == "report_profile_derivation_mismatch"


def test_per_message_digest_binding_a_reference_is_not_admitted(tmp_path):
    # The wire can never serve it: RuntimeProfile refuses a frozen-artifact
    # reference on a method whose registry entry consumes none, so this forged
    # digest (same canonical hashing form) is not this profile's derivation.
    journal = make_journal(tmp_path, native_plan())
    start_journal(journal)
    profile = per_message_profile()
    digest = wire_digest(profile, observed())
    forged = nv._canonical_sha256(dict(nv._derivation_value(profile, observed()),
                                       calibration_path=CALIBRATION_PATH,
                                       calibration_digest=CALIBRATION_DIGEST))
    assert forged != digest
    report = copy.deepcopy(EXAMPLE["report_before"])
    report["binding"]["profile_digest"] = forged
    publish(journal, report, forged, nv.PIN_STATIC_INPUT_SHA256)
    result = run_before(journal)
    assert result["ok"] is False and result["code"] == "report_profile_derivation_mismatch"
