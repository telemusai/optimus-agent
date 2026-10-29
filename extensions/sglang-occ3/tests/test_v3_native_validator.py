"""Offline adapter proofs. No server, GPU, helper resolution or live model.

Every fixture value is synthetic. These tests exercise the real adapter
verification logic, the real Services selection, the real Journal and the real
exit_aware wrapper used by the controller. No native binary is executed: the
phase-suite reports are explicit synthetic observations only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import lifecycle as life
from deploy.v3 import native_validator as nv
from deploy.v3.live_validation import (LiveValidator, NativeHandoffValidator,
                                       SourceKvBaselineValidator, UnselectedValidator,
                                       VALIDATOR_KEY)

ORIGINAL = "1" * 64
CANDIDATE = "c" * 64
IMAGE = "sha256:" + "a" * 64
CANDIDATE_IMAGE = "sha256:" + "b" * 64
PROXY = "9" * 64
PROXY_IMAGE = "sha256:" + "9" * 64
KEY = "synthetic-private-key-not-real"
FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "native_phase_suite_v1.json").read_text())
EXAMPLE = FIXTURE["example"]
PINNED = FIXTURE["admission_pinned_inputs"]
ATTEMPT = EXAMPLE["attempt"]["attempt_id"]
EPOCH_BEFORE = EXAMPLE["attempt"]["candidate_started_at_before"]
EPOCH_AFTER = EXAMPLE["attempt"]["candidate_started_at_after"]
CHALLENGE_BEFORE = EXAMPLE["challenge_before"]["challenge"]
CHALLENGE_AFTER = EXAMPLE["challenge_after"]["challenge"]
PIN_SCHEMA = "occ3.admission.pin.v1"
# Frozen offline fixture pin digests (revision-4 admission pins, kept as the
# per-phase pin-file values the example reports bind and re-derive).
FIXTURE_PIN_BEFORE = "1a4da9e7993afcc0191d02a838047454e32572f29a843befda036944eb77a906"
FIXTURE_PIN_AFTER = "9f4db2e622b62425ba6e763c018f4e1e1241303f922fd6b62944aea1653ea22c"


def original_doc():
    return {"Id": ORIGINAL, "Image": IMAGE, "Name": "/fixture-service",
            "Config": {"Image": IMAGE, "Hostname": "fixture-host",
                       "Entrypoint": ["python3"],
                       "Cmd": ["-m", "sglang.launch_server", "--api-key", KEY],
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


def native_plan(**changes):
    return plan(validator=VALIDATOR_KEY, native_validator=copy.deepcopy(PINNED), **changes)


def make_journal(tmp_path, plan_value):
    return life.Journal.create(tmp_path / "attempt", plan_value, original_doc())


def make_validator(journal, *, timeout=5.0, phase="before_restart"):
    challenge = CHALLENGE_BEFORE if phase == "before_restart" else CHALLENGE_AFTER
    state = journal.read()
    return NativeHandoffValidator(state["plan"], journal, state, timeout=timeout,
                                  challenge_factory=lambda: challenge)


def pin_document(phase, observed, *, digest=None, **overrides):
    """Per-phase admission pin as the executor writes it after step 0.

    The pin binds the attempt, the pinned static input file digest and THIS
    phase's observed incarnation; profile_digest defaults to the derivation
    receipt the identity query already proved (recomputed from the pinned
    static profile fields plus the observed incarnation).
    """
    profile = PINNED["admission_template"]["source_profile"]
    document = {
        "schema": PIN_SCHEMA,
        "phase": phase,
        "attempt_id": ATTEMPT,
        "static_input_sha256": nv.PIN_STATIC_INPUT_SHA256,
        "runtime_epoch": observed["runtime_epoch"],
        "runtime_generation": observed["runtime_generation"],
        "profile_digest": digest if digest is not None
        else nv._derived_profile_digest(profile, observed),
        "written_at": int(time.time()),
    }
    document.update(overrides)
    return document


def write_pin(journal, phase, observed, **overrides):
    """Write-once per-phase pin file into the attempt (journal) directory."""
    document = pin_document(phase, observed, **overrides)
    life.private_json(journal.directory / ("admission-pin-" + phase + ".json"),
                      document, exclusive=True)
    return document


def write_report(journal, phase, report, *, completed_at=None, pin=True):
    """Publish the phase report and, by default, its honest pin file.

    profile_digest is incarnation-bound: tests that change a phase's observed
    epoch/generation get a pin re-derived for that incarnation so the binding
    check and the derivation receipt stay consistent with the mutation under
    test. pin=False leaves the pin to the test (fabrication negatives).
    """
    document = copy.deepcopy(report)
    document["completed_at"] = completed_at if completed_at is not None else int(time.time()) + 30
    life.private_json(journal.directory / ("native-phase-suite-" + phase + ".json"), document)
    if pin:
        write_pin(journal, phase, document["observed"])


def before_state(journal):
    journal.update(candidate_id=CANDIDATE, candidate_started=True, candidate_epoch=EPOCH_BEFORE,
                   gates={"service_initial": {"ok": True}, "native_initial": {"ok": True}})


def after_state(journal):
    journal.update(candidate_epoch=EPOCH_AFTER, restart_done=True, restart_count=1,
                   gates={"service_initial": {"ok": True}, "native_initial": {"ok": True},
                          "live_before": {"ok": True}, "restart": {"ok": True}})


# -- selection ----------------------------------------------------------------
def test_absent_key_keeps_default_blocked_live_validator(tmp_path):
    journal = make_journal(tmp_path, plan())
    services = life.Services(journal.original(), journal.read()["plan"], journal=journal)
    selected = services.validator(journal.read())
    assert isinstance(selected, LiveValidator) and not isinstance(selected, NativeHandoffValidator)
    assert selected.pre_mutation_admission()["ok"] is False


def test_native_key_selects_adapter(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    services = life.Services(journal.original(), journal.read()["plan"], journal=journal)
    selected = services.validator(journal.read())
    assert isinstance(selected, NativeHandoffValidator)


def test_unknown_key_fails_closed_without_fallback(tmp_path):
    journal = make_journal(tmp_path, plan(validator="someone-elses-validator"))
    services = life.Services(journal.original(), journal.read()["plan"], journal=journal)
    selected = services.validator(journal.read())
    assert isinstance(selected, UnselectedValidator)
    for method in ("pre_mutation_admission", "before_restart", "after_restart"):
        result = getattr(selected, method)()
        assert result["ok"] is False and result["code"] == "unknown_validator_selection"


def test_native_key_without_journal_fails_closed(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    services = life.Services(journal.original(), journal.read()["plan"])
    selected = services.validator(journal.read())
    assert isinstance(selected, UnselectedValidator)
    assert selected.pre_mutation_admission()["code"] == "validator_journal_unavailable"


def test_services_admission_order_blocks_without_pinned_inputs(tmp_path):
    journal = make_journal(tmp_path, plan(validator=VALIDATOR_KEY))
    services = life.Services(journal.original(), journal.read()["plan"], journal=journal)
    admission = services.pre_mutation_admission(journal.read())
    assert admission["ok"] is False
    assert admission["code"] == "native_latent_validation_unavailable"


def test_services_admission_passes_with_pinned_inputs(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    services = life.Services(journal.original(), journal.read()["plan"], journal=journal)
    admission = services.pre_mutation_admission(journal.read())
    assert admission["ok"] is True
    assert admission["admission_scope"] == "offline_inputs_only"
    assert admission["runtime_binding"] == "deferred_to_post_start"


def test_baseline_validator_unchanged():
    assert issubclass(SourceKvBaselineValidator, LiveValidator)
    assert SourceKvBaselineValidator("http://fixture", KEY, "m")._result(
        "before_restart", {})["restart_authorized"] is False


# -- admission inputs --------------------------------------------------------
def test_admission_accepts_synthetic_pinned_inputs(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is True
    assert result["gpu_proof"] == "not_asserted_preflight"
    assert result["live_success_claimed"] is False
    assert result["http_calls_performed"] == 0 and result["credentials_resolved"] == 0
    assert result["binary_mirror_verified"] == "absent_pinned_digest_only"
    # Static-only admission: the per-phase digests live in attempt-dir pin
    # files, never in the frozen plan.
    assert result["bindings"]["profile_digest_source"].startswith(
        "per-phase admission pin files")
    assert "profile_digest_before_restart" not in result["bindings"]
    assert "profile_digest_after_restart" not in result["bindings"]


@pytest.mark.parametrize("mutate,code", [
    (lambda p: p.pop("native_validator"), "native_validator_inputs_missing"),
    (lambda p: p.update(native_validator={"method_id": PER_MESSAGE}),
     "admission_fieldset_invalid"),
    (lambda p: p["native_validator"].update(method_id="other-method"), "method_mismatch"),
    (lambda p: p["native_validator"]["harness_artifact"].update(
        binary_sha256="not-a-digest"), "harness_binary_sha256_invalid"),
    (lambda p: p["native_validator"]["harness_artifact"].update(
        platform="linux-x86_64"), "harness_platform_unsupported"),
    (lambda p: p["native_validator"]["source_manifest"]["files"].update(
        {"../escape.rs": "a" * 64}), "source_manifest_path_invalid"),
    (lambda p: p["native_validator"]["interface_versions"].update(
        report="optimus.occ3.native-attempt.v1"), "report_schema_version_unknown"),
    (lambda p: p["native_validator"]["offline_selftest"].update(exit_code=1),
     "offline_selftest_not_clean"),
    (lambda p: p["native_validator"]["offline_selftest"].update(
        binary_sha256="f" * 64), "offline_selftest_binary_mismatch"),
    (lambda p: p["native_validator"]["admission_template"]["identity"].update(
        model_id="tampered"), "identity_fingerprint_mismatch"),
    (lambda p: p["native_validator"]["admission_template"]["source_profile"].update(
        dtype="int8"), "source_dtype_unsupported"),
    (lambda p: p["native_validator"]["admission_template"]["source_profile"].update(
        expected_ranks=4), "expected_ranks_not_tp8"),
    (lambda p: p["native_validator"]["budget_resource_receipt"].pop("root_reviewed"),
     "budget_receipt_invalid"),
    (lambda p: p["native_validator"]["max_budget"].update(default_max_tokens=4096),
     "default_max_tokens_changed"),
    (lambda p: p["native_validator"]["max_budget"].update(context_window=8192),
     "context_window_changed"),
    (lambda p: p["native_validator"]["max_budget"].update(thinking_level="medium"),
     "thinking_level_not_max"),
    (lambda p: p["native_validator"]["layout_strategy"].update(
        bound_at="at_admission"), "layout_bound_not_post_start"),
    (lambda p: p["native_validator"]["endpoint"].update(
        origin="http://synthetic-native.invalid"), "endpoint_origin_not_https"),
    (lambda p: p["native_validator"]["endpoint"].update(control_root="/occ3/v1/"),
     "control_root_invalid"),
    (lambda p: p["native_validator"]["credential_references"].update(
        api_key="SYNTHETIC KEY VALUE"), "api_key_reference_invalid"),
    (lambda p: p["native_validator"]["credential_references"]["headers"].update(
        {"Authorization": "!helper"}), "header_name_not_allowed"),
    (lambda p: p["native_validator"]["control_spec"].update(max_attempts=2),
     "multiple_attempts_allowed"),
    (lambda p: p["native_validator"]["control_spec"].update(no_replay=False),
     "replay_allowed"),
])
def test_admission_refuses_each_invalid_input(tmp_path, mutate, code):
    journal = make_journal(tmp_path, native_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    mutate(plan_value)
    validator = NativeHandoffValidator(plan_value, journal, state)
    result = validator.pre_mutation_admission()
    assert result["ok"] is False and result["code"] == code


def test_admission_never_resolves_or_leaks_references(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    result = make_validator(journal).pre_mutation_admission()
    dumped = json.dumps(result)
    assert "!synthetic-helper-reference" not in dumped
    assert "!synthetic-pomerium-helper" not in dumped


def test_adapter_source_has_no_network_or_env_surface():
    source = (Path(nv.__file__)).read_text()
    for forbidden in ("urlopen", "requests", "os.environ", "subprocess", "socket"):
        assert forbidden not in source


def test_binary_mirror_digest_is_verified_when_present(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    mirror = journal.directory / "harness-binary-mirror"
    mirror.write_bytes(b"synthetic binary bytes")
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "binary_mirror_digest_mismatch"
    import hashlib as _h
    mirror.write_bytes(b"x" * 64)
    state = journal.read()
    pinned = copy.deepcopy(state["plan"]["native_validator"])
    pinned["harness_artifact"]["binary_sha256"] = _h.sha256(b"x" * 64).hexdigest()
    pinned["offline_selftest"]["binary_sha256"] = pinned["harness_artifact"]["binary_sha256"]
    validator = NativeHandoffValidator(dict(state["plan"], native_validator=pinned),
                                        journal, state)
    result = validator.pre_mutation_admission()
    assert result["ok"] is True
    assert result["binary_mirror_verified"] == "verified_mirrored_bytes_no_execution"


# -- before phase ------------------------------------------------------------
def test_before_restart_passes_with_retention_record(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    result = make_validator(journal).before_restart()
    assert result["ok"] is True, result
    assert result["cleanup_scope"] == "positive_and_no_state_only"
    retained = result["negative_control_retained"]
    assert retained["retention_reason"] == "planned_restart"
    assert retained["artifact_sha256"] == EXAMPLE["report_before"]["negative_control_retained"]["artifact_sha256"]
    assert result["gpu_proof"] is False and result["physical_gpu_free_proven"] is False
    assert result["ordinary_serving_nonregression"] == "separate_controller_service_final_gate"
    assert (journal.directory / "validator-challenge-before_restart.json").exists()
    assert json.dumps(result).count(CHALLENGE_BEFORE) == 0


@pytest.mark.parametrize("missing", ["service_initial", "native_initial"])
def test_before_restart_requires_prior_gates(tmp_path, missing):
    journal = make_journal(tmp_path, native_plan())
    gates = {"service_initial": {"ok": True}, "native_initial": {"ok": True}}
    gates.pop(missing)
    journal.update(candidate_id=CANDIDATE, candidate_started=True, candidate_epoch=EPOCH_BEFORE,
                   gates=gates)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    result = make_validator(journal).before_restart()
    assert result["ok"] is False
    assert result["code"] == "phase_gate_missing:" + missing


def test_before_restart_requires_started_candidate(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    journal.update(gates={"service_initial": {"ok": True}, "native_initial": {"ok": True}})
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "candidate_not_started"


def test_before_restart_timeout_is_bounded_and_blocked(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    result = make_validator(journal, timeout=0.05).before_restart()
    assert result["ok"] is False
    assert result["code"] == "native_phase_suite_report_timeout"
    assert (journal.directory / "validator-challenge-before_restart.json").exists()


def test_before_restart_challenge_is_written_once(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    validator = make_validator(journal)
    assert validator.before_restart()["ok"] is True
    again = make_validator(journal).before_restart()
    assert again["ok"] is False
    assert again["code"] == "challenge_already_exists_no_replay"


@pytest.mark.parametrize("mutate,code", [
    (lambda r: r.update(validator_challenge="wrong"), "challenge_binding_mismatch"),
    (lambda r: r.update(attempt_id="other-attempt-0002"), "attempt_binding_mismatch"),
    (lambda r: r.update(candidate_id="d" * 64), "candidate_binding_mismatch"),
    (lambda r: r.update(candidate_started_at="other-epoch"), "started_at_binding_mismatch"),
    (lambda r: r.update(schema="optimus.occ3.native-attempt.v1"), "report_schema_unknown"),
    (lambda r: r.update(phase="after_restart"), "report_phase_mismatch"),
    (lambda r: r.update(extra_field=True), "report_fieldset_invalid"),
    (lambda r: r["harness"].update(binary_sha256="f" * 64),
     "report_binary_binding_mismatch"),
    (lambda r: r["harness"].update(source_manifest_sha256="0" * 64),
     "report_source_manifest_mismatch"),
    (lambda r: r["binding"].update(identity_fingerprint="sha256:" + "0" * 64),
     "report_identity_binding_mismatch"),
    (lambda r: r["binding"].update(endpoint_origin="https://other.invalid"),
     "report_origin_binding_mismatch"),
    (lambda r: r["observed"].update(expected_ranks=4), "observed_ranks_mismatch"),
    (lambda r: r["observed"].update(ready_ranks=7), "all_ranks_not_ready"),
    (lambda r: r["cases"]["positive"].update(answer_sha256="0" * 64),
     "positive_answer_not_nonce_hash"),
    (lambda r: r["cases"]["positive"].update(lease_state="queued"), "lease_not_consumed"),
    (lambda r: r["cases"]["positive"].update(consumed_before_tools=False),
     "consumed_not_before_tools"),
    (lambda r: r["cases"]["positive"]["receiver_request_audit"].update(max_tokens=4096),
     "positive_audit_invalid:budget_not_default_max"),
    (lambda r: r["cases"]["positive"]["receiver_request_audit"].update(request_count=2),
     "positive_audit_invalid:request_count_invalid"),
    (lambda r: r["cases"]["positive"]["receiver_request_audit"].update(
        source_facts_absent=False), "positive_audit_invalid:source_facts_present"),
    (lambda r: r["cases"]["no_state"].update(answer_sha256="0" * 64),
     "no_state_not_exact_unknown"),
    (lambda r: r["cases"]["no_state"].update(messages=1), "no_state_side_effects_present"),
    (lambda r: r["cases"]["no_state"].update(semantic_nonce_match=True),
     "no_state_nonce_leaked"),
    (lambda r: r["negative_control_retained"].update(retention_reason="convenience"),
     "retention_reason_invalid"),
    (lambda r: r["negative_control_retained"].update(
        runtime_epoch="other-epoch"), "retained_epoch_mismatch"),
    (lambda r: r.update(gpu_free_proven=True), "gpu_free_overclaimed"),
    (lambda r: r.update(logical_cleanup_confirmed=False), "logical_cleanup_unconfirmed"),
    (lambda r: r["cases"]["positive"].update(case_passed=False), "positive_case_failed"),
    (lambda r: r["cases"].pop("stale_handle_pre"), "cases_fieldset_invalid"),
])
def test_before_restart_report_negatives(tmp_path, mutate, code):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    report = copy.deepcopy(EXAMPLE["report_before"])
    mutate(report)
    write_report(journal, "before_restart", report)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == code, result


def test_before_restart_rejects_completion_before_challenge(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"], completed_at=1)
    result = make_validator(journal).before_restart()
    assert result["code"] == "completion_outside_window"


def test_before_restart_rejects_duplicate_json_keys(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    raw = json.dumps(EXAMPLE["report_before"])[:-1] + ',"phase":"before_restart"}'
    path = journal.directory / "native-phase-suite-before_restart.json"
    path.write_text(raw)
    result = make_validator(journal).before_restart()
    assert result["code"] == "duplicate_json_key"


def test_before_restart_rejects_non_finite_numbers(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    raw = json.dumps(EXAMPLE["report_before"]).replace("1760000000", "NaN")
    (journal.directory / "native-phase-suite-before_restart.json").write_text(raw)
    result = make_validator(journal).before_restart()
    assert result["code"] == "non_finite_number"


def test_before_restart_rejects_symlinked_report(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    real = journal.directory / "real-report.json"
    real.write_text(json.dumps(EXAMPLE["report_before"]))
    link = journal.directory / "native-phase-suite-before_restart.json"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    result = make_validator(journal).before_restart()
    assert result["code"] == "symlink_refused"


# -- after phase -------------------------------------------------------------
def test_after_restart_passes_with_chained_stale_refusal(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    assert make_validator(journal).before_restart()["ok"] is True
    after_state(journal)
    write_report(journal, "after_restart", EXAMPLE["report_after"])
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is True, result
    refusal = result["stale_handle_refusal"]
    assert refusal["http_status"] == 409 and refusal["error_code"] == "state"
    assert refusal["attempts"] == 1
    assert refusal["artifact_sha256"] == EXAMPLE["report_before"]["negative_control_retained"]["artifact_sha256"]
    assert result["stale_binding"]["before_runtime_epoch"] != result["stale_binding"]["after_runtime_epoch"]


def test_after_restart_requires_before_phase_files(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    after_state(journal)
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is False
    assert result["code"] == "before_phase_binding_unavailable"


def test_after_restart_requires_restart_gates(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    assert make_validator(journal).before_restart()["ok"] is True
    journal.update(candidate_epoch=EPOCH_AFTER, gates={"live_before": {"ok": True}})
    write_report(journal, "after_restart", EXAMPLE["report_after"])
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is False
    assert result["code"] == "phase_gate_missing:restart"


@pytest.mark.parametrize("mutate,code", [
    (lambda r: r["cases"]["stale_handle_post"].update(http_status=401),
     "stale_refusal_not_409"),
    (lambda r: r["cases"]["stale_handle_post"].update(http_status=200),
     "stale_refusal_not_409"),
    (lambda r: r["cases"]["stale_handle_post"].update(error_code="expired"),
     "stale_refusal_code_not_state"),
    (lambda r: r["cases"]["stale_handle_post"].update(error_code="replay"),
     "stale_refusal_code_not_state"),
    (lambda r: r["cases"]["stale_handle_post"].update(attempts=2),
     "stale_probe_attempted_twice"),
    (lambda r: r["cases"]["stale_handle_post"].update(artifact_sha256="0" * 64),
     "stale_artifact_binding_mismatch"),
    (lambda r: r["observed"].update(runtime_epoch="synthetic-runtime-epoch-before"),
     "runtime_epoch_unchanged"),
    (lambda r: r["observed"].update(runtime_generation=41), "runtime_generation_not_advanced"),
    (lambda r: r["cases"]["stale_handle_post"].update(
        fresh_enrollment_runtime_epoch="synthetic-runtime-epoch-before"),
     "fresh_enrollment_epoch_mismatch"),
])
def test_after_restart_negatives(tmp_path, mutate, code):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    assert make_validator(journal).before_restart()["ok"] is True
    after_state(journal)
    report = copy.deepcopy(EXAMPLE["report_after"])
    mutate(report)
    if code == "runtime_epoch_unchanged":
        report["cases"]["stale_handle_post"]["fresh_enrollment_runtime_epoch"] = \
            report["observed"]["runtime_epoch"]
    # The incarnation-bound binding and the after pin file stay
    # derivation-consistent with the mutated observed incarnation so the
    # mutation's OWN check fires.
    profile = PINNED["admission_template"]["source_profile"]
    derived = nv._derived_profile_digest(profile, report["observed"])
    report["binding"]["profile_digest"] = derived
    write_report(journal, "after_restart", report)
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is False and result["code"] == code, result


# -- controller wiring -------------------------------------------------------
class InertAPI:
    def __init__(self):
        self.doc = {"Id": CANDIDATE, "Image": CANDIDATE_IMAGE,
                    "State": {"Running": True, "Restarting": False, "Dead": False}}

    def inspect(self, identity):
        assert identity == self.doc["Id"]
        return self.doc


def test_exit_aware_wrapper_returns_blocked_result_and_gate_fails_closed(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    validator = make_validator(journal, timeout=0.05)
    result = life.exit_aware(InertAPI(), CANDIDATE, CANDIDATE_IMAGE,
                             validator.before_restart, time.monotonic() + 2,
                             guard=lambda: None, interval=0.05)
    assert result["ok"] is False
    with pytest.raises(life.Failure):
        journal.gate("live_before", result)


def test_exit_aware_wrapper_passes_verified_result_to_gate(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    validator = make_validator(journal)
    result = life.exit_aware(InertAPI(), CANDIDATE, CANDIDATE_IMAGE,
                             validator.before_restart, time.monotonic() + 5,
                             guard=lambda: None, interval=0.05)
    assert result["ok"] is True
    journal.gate("live_before", result)
    assert journal.read()["gates"]["live_before"]["cleanup_scope"] == "positive_and_no_state_only"


def test_results_stay_in_private_attempt_scope(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    result = make_validator(journal).before_restart()
    dumped = json.dumps(result)
    assert KEY not in dumped
    assert "!synthetic-helper-reference" not in dumped
    for name in ("validator-challenge-before_restart.json",
                 "admission-pin-before_restart.json",
                 "native-phase-suite-before_restart.json"):
        assert (journal.directory / name).exists()
    assert not (journal.directory / "native-phase-suite-after_restart.json").exists()


# -- V1: report publication atomicity contract (contract_revision 2) ---------
class StagingWriter:
    """Test harness ONLY: simulates the required harness-side publication.

    Stages into a uniquely named private file, then publishes at the final
    name without replacing an existing target. This simulation is not proof of
    real Windows no-replace atomic publication or ACL behavior.
    """

    def __init__(self, journal):
        self.directory = journal.directory

    def publish(self, phase, document, *, complete=True):
        import uuid
        staging = self.directory / ("stage-" + uuid.uuid4().hex + ".json")
        payload = json.dumps(document)
        if not complete:
            payload = payload[: max(1, len(payload) // 2)]
        with staging.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        final = self.directory / ("native-phase-suite-" + phase + ".json")
        if final.exists():
            staging.unlink()
            return False
        os.rename(staging, final)
        return True


def test_incomplete_private_staging_is_invisible_during_wait(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_pin(journal, "before_restart", EXAMPLE["report_before"]["observed"])
    writer = StagingWriter(journal)
    import threading

    def finish():
        time.sleep(0.15)
        writer.publish("before_restart", dict(EXAMPLE["report_before"],
                                               completed_at=int(time.time()) + 30))

    threading.Thread(target=finish).start()
    result = make_validator(journal).before_restart()
    assert result["ok"] is True, result
    assert not any(p.name.startswith("stage-") for p in journal.directory.iterdir())


def test_complete_final_publication_within_deadline_is_accepted(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_pin(journal, "before_restart", EXAMPLE["report_before"]["observed"])
    writer = StagingWriter(journal)
    assert writer.publish("before_restart", dict(EXAMPLE["report_before"],
                                                 completed_at=int(time.time()) + 30)) is True
    result = make_validator(journal).before_restart()
    assert result["ok"] is True, result


def test_malformed_final_is_immediate_refusal_not_retried(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    final = journal.directory / "native-phase-suite-before_restart.json"
    final.write_text('{"schema": "optimus.occ3.native-phase-suite.v1", "phase":')
    started = time.monotonic()
    result = make_validator(journal, timeout=2.0).before_restart()
    elapsed = time.monotonic() - started
    assert result["ok"] is False and result["code"] == "malformed_json"
    assert elapsed < 1.0
    # A late valid publication cannot rescue the burned one-shot challenge.
    final.write_text(json.dumps(dict(EXAMPLE["report_before"],
                                     completed_at=int(time.time()) + 30)))
    again = make_validator(journal).before_restart()
    assert again["code"] == "challenge_already_exists_no_replay"


def test_truncated_mid_write_staging_does_not_burn_challenge(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_pin(journal, "before_restart", EXAMPLE["report_before"]["observed"])
    writer = StagingWriter(journal)
    import threading

    def finish():
        time.sleep(0.15)
        writer.publish("before_restart", dict(EXAMPLE["report_before"],
                                               completed_at=int(time.time()) + 30))

    thread = threading.Thread(target=finish)
    thread.start()
    # Simulate an abandoned partial staging visible during the wait.
    staging = journal.directory / "stage-partial.json"
    staging.write_text(json.dumps(EXAMPLE["report_before"])[:200])
    result = make_validator(journal).before_restart()
    thread.join()
    assert result["ok"] is True, result


def test_existing_final_target_is_never_replaced(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_pin(journal, "before_restart", EXAMPLE["report_before"]["observed"])
    writer = StagingWriter(journal)
    first = dict(EXAMPLE["report_before"], completed_at=int(time.time()) + 30)
    assert writer.publish("before_restart", first) is True
    tampered = dict(first, validator_challenge="different")
    assert writer.publish("before_restart", tampered) is False
    result = make_validator(journal).before_restart()
    assert result["ok"] is True
    on_disk = json.loads((journal.directory / "native-phase-suite-before_restart.json").read_text())
    assert on_disk["validator_challenge"] == first["validator_challenge"]


# -- V2: bounded JSON depth guard -------------------------------------------
def _deep_payload(depth, report):
    base = json.dumps(report)
    i = base.find('"observed"')
    j = base.find('"suite_nonce_sha256"')
    inner = '"x"'
    for _ in range(depth):
        inner = '{"observed":' + inner + "}"
    return base[:i] + '"observed": ' + inner + ", " + base[j:]


def test_overdepth_report_under_2mib_is_refused_not_crash(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    payload = _deep_payload(3000, EXAMPLE["report_before"])
    assert len(payload) <= 2 * 1024 * 1024
    (journal.directory / "native-phase-suite-before_restart.json").write_text(payload)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "malformed_json"


def test_exact_depth_boundary_is_accepted(tmp_path):
    document = json.loads(json.dumps(EXAMPLE["report_before"]))
    assert nv._json_depth_ok(json.dumps(document).encode()) is True


@pytest.mark.parametrize("payload,ok", [
    (b'{"a": {"b": {"c": 1}}}', True),
    (b'{"a": "not } closed", "b": [1, 2]}', True),
    (rb'{"a": "quote \\" and backslash \\\\" and } brace"}', True),
    (b'{"a": "bracket } and brace { inside string"}', True),
    (b'{"a": {"b":', False),
    (b'["a", "b"', False),
    (b'{"a": "unterminated}', False),
])
def test_json_depth_scan_escapes_and_closings(payload, ok):
    assert nv._json_depth_ok(payload) is ok


def test_depth_guard_covers_admission_documents_too(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    deep = {"schema": "x", "n": "x"}
    node = deep
    for _ in range(100):
        node["n"] = {"n": "x"}
        node = node["n"]
    result = NativeHandoffValidator({"validator": VALIDATOR_KEY, "native_validator": deep},
                                    journal, journal.read()).pre_mutation_admission()
    assert result["code"] == "malformed_json"


def test_public_phase_methods_translate_recursion_to_refusal(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    payload = _deep_payload(50000, EXAMPLE["report_before"])
    assert len(payload) <= 2 * 1024 * 1024
    (journal.directory / "native-phase-suite-before_restart.json").write_text(payload)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "malformed_json"


# -- P3: generation must advance under the controlled profile ----------------
def _after_case(journal, *, generation, epoch="synthetic-runtime-epoch-after"):
    """Rewrite the after report's incarnation AND its derivation-consistent pin.

    The after pin file is re-derived for the synthetic incarnation under test
    so the binding and derivation checks pass; the restart-chain invariants
    are then exercised on their own merits.
    """
    write_report(journal, "after_restart", EXAMPLE["report_after"], pin=False)
    report = json.loads((journal.directory / "native-phase-suite-after_restart.json").read_text())
    report["observed"]["runtime_generation"] = generation
    report["observed"]["runtime_epoch"] = epoch
    report["cases"]["stale_handle_post"]["fresh_enrollment_runtime_epoch"] = epoch
    profile = PINNED["admission_template"]["source_profile"]
    derived = nv._derived_profile_digest(profile, report["observed"])
    report["binding"]["profile_digest"] = derived
    life.private_json(journal.directory / "native-phase-suite-after_restart.json", report)
    write_pin(journal, "after_restart", report["observed"])


def test_fresh_epoch_and_advanced_generation_accepted(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    assert make_validator(journal).before_restart()["ok"] is True
    after_state(journal)
    _after_case(journal, generation=42)
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is True, result
    assert result["stale_binding"]["generation_advanced"] is True


@pytest.mark.parametrize("generation", [41, 40])
def test_unchanged_or_decreased_generation_refused_despite_fresh_epoch(tmp_path, generation):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    assert make_validator(journal).before_restart()["ok"] is True
    after_state(journal)
    _after_case(journal, generation=generation)
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is False
    assert result["code"] == "runtime_generation_not_advanced"


def test_unchanged_epoch_refused_even_with_advanced_generation(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    assert make_validator(journal).before_restart()["ok"] is True
    after_state(journal)
    _after_case(journal, generation=42, epoch="synthetic-runtime-epoch-before")
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is False
    assert result["code"] == "runtime_epoch_unchanged"


def test_volatile_reset_runtime_is_not_admitted_no_auto_retry(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    assert make_validator(journal).before_restart()["ok"] is True
    after_state(journal)
    _after_case(journal, generation=1, epoch="synthetic-runtime-epoch-after")
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is False
    assert result["code"] == "runtime_generation_not_advanced"
    # No automatic reset/repair/retry: the failed gate record is not rewritten.
    assert journal.read().get("gates", {}).get("live_after") is None


# -- V2 follow-on: strict UTF-8-only encoding bound ---------------------------
def _nested(depth):
    inner = 0
    for _ in range(depth):
        inner = [inner]
    return inner


@pytest.mark.parametrize("enc", ["utf-16", "utf-32"])
def test_non_utf8_small_payload_refused(tmp_path, enc):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    payload = json.dumps(dict(EXAMPLE["report_before"],
                              completed_at=int(time.time()) + 30)).encode(enc)
    (journal.directory / "native-phase-suite-before_restart.json").write_bytes(payload)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "malformed_json"


@pytest.mark.parametrize("enc", ["utf-16", "utf-32"])
def test_non_utf8_depth41_bypass_refused(tmp_path, enc):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    document = dict(EXAMPLE["report_before"], completed_at=int(time.time()) + 30)
    document["observed"] = {"nested": _nested(41)}
    payload = json.dumps(document, ensure_ascii=False).encode(enc)
    assert len(payload) <= 2 * 1024 * 1024
    (journal.directory / "native-phase-suite-before_restart.json").write_bytes(payload)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "malformed_json"


def test_valid_utf8_unicode_strings_accepted(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    document = dict(EXAMPLE["report_before"], completed_at=int(time.time()) + 30)
    unicode_epoch = "synthetic-runtime-\u00e9poch-\u2222\u0122\U0001F600"
    document["observed"] = dict(document["observed"], runtime_epoch=unicode_epoch)
    document["negative_control_retained"] = dict(document["negative_control_retained"],
                                                 runtime_epoch=unicode_epoch)
    # The per-boot digest re-derives from the unicode epoch (canonical ASCII
    # escaping on both sides), so the phase pin file and binding follow it.
    profile = PINNED["admission_template"]["source_profile"]
    derived = nv._derived_profile_digest(profile, document["observed"])
    document["binding"] = dict(document["binding"], profile_digest=derived)
    write_pin(journal, "before_restart", document["observed"])
    (journal.directory / "native-phase-suite-before_restart.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8")
    result = make_validator(journal).before_restart()
    assert result["ok"] is True, result


def test_invalid_utf8_bytes_refused(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    (journal.directory / "native-phase-suite-before_restart.json").write_bytes(
        b'{"schema": "\xff\xfe invalid"}')
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "malformed_json"


def test_utf8_bom_refused_fail_closed(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    payload = b"\xef\xbb\xbf" + json.dumps(
        dict(EXAMPLE["report_before"], completed_at=int(time.time()) + 30)).encode("utf-8")
    (journal.directory / "native-phase-suite-before_restart.json").write_bytes(payload)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "malformed_json"


def test_truncated_multibyte_utf8_refused(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    payload = b'{"a": "\xe2\x82"}'
    (journal.directory / "native-phase-suite-before_restart.json").write_bytes(payload)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "malformed_json"


def test_depth_guard_bytes_input_requires_strict_utf8():
    assert nv._json_depth_ok(b'{"a": {"b": 1}}') is True
    assert nv._json_depth_ok(json.dumps({"a": 1}).encode("utf-16")) is False
    assert nv._json_depth_ok(json.dumps({"a": 1}).encode("utf-32")) is False
    assert nv._json_depth_ok(b'\xff\xfe') is False
    assert nv._json_depth_ok('{"a": "\u00e9"}') is True


# -- P4: incarnation-bound profile_digest (epoch-aware repair) ----------------
# The REAL pinned static profile (extensions/sglang-occ3/deploy/native-build-inputs/
# native-profile-input.json, file sha256
# 6e676f1c6fc795ba412c4327168b51c8ef6f65056664638c4954eec58e75348a) plus the
# TS8 live window's observed incarnation reproduces the byte-exact wire digest
# the server reported; a different boot must never reproduce it.
TS8_STATIC_PROFILE = {
    "anchor_alpha": 0.0,
    "expected_ranks": 8,
    "hidden_dim": 6144,
    "layer_index": 77,
    "max_active": 1,
    "max_live_bytes": 268435456,
    "max_prefix_tokens": 128,
    "max_records": 256,
    "max_ttl_seconds": 3600,
    "method": "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1",
    "model_fingerprint": "sha256:db2c8830820da8bd3370bbfacf0799e33085ef7b8b1ca27840b88e3e8101b732",
    "model_id": "glm-5.3",
    "tile_rows": 256,
    "workspace_bytes": 134217728,
}
TS8_WIRE_DIGEST = "0c901d0f47335f0e4087fbf960279e1b02d6f4bd5219d63224dd8487f00423d9"
TS8_RUNTIME_EPOCH = "c2847f3de4fc43d1897644eda8dc81a9"
TS8_RUNTIME_GENERATION = 1


def test_ts8_incarnation_reproduces_byte_exact_wire_digest():
    observed = {"runtime_epoch": TS8_RUNTIME_EPOCH,
                "runtime_generation": TS8_RUNTIME_GENERATION}
    assert nv._derived_profile_digest(TS8_STATIC_PROFILE, observed) == TS8_WIRE_DIGEST


def test_profile_digest_is_per_boot_volatile():
    base = {"runtime_epoch": TS8_RUNTIME_EPOCH,
            "runtime_generation": TS8_RUNTIME_GENERATION}
    first = nv._derived_profile_digest(TS8_STATIC_PROFILE, base)
    other_epoch = nv._derived_profile_digest(
        TS8_STATIC_PROFILE, dict(base, runtime_epoch="f" * 32))
    other_generation = nv._derived_profile_digest(
        TS8_STATIC_PROFILE, dict(base, runtime_generation=2))
    assert first != other_epoch and first != other_generation
    assert other_epoch != other_generation


def test_fixture_pins_rederive_from_static_profile():
    profile = PINNED["admission_template"]["source_profile"]
    before = nv._derived_profile_digest(profile, EXAMPLE["report_before"]["observed"])
    after = nv._derived_profile_digest(profile, EXAMPLE["report_after"]["observed"])
    assert before == FIXTURE_PIN_BEFORE == EXAMPLE["report_before"]["binding"]["profile_digest"]
    assert after == FIXTURE_PIN_AFTER == EXAMPLE["report_after"]["binding"]["profile_digest"]
    assert before != after


def test_admission_refuses_single_static_profile_digest_fieldset(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    profile = plan_value["native_validator"]["admission_template"]["source_profile"]
    profile["profile_digest"] = FIXTURE_PIN_BEFORE
    validator = NativeHandoffValidator(plan_value, journal, state)
    result = validator.pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "source_profile_invalid"


def test_admission_refuses_placeholder_per_phase_pin_fields(tmp_path):
    # TS9 defect shape: fill-at-window placeholder pins still carried inside
    # the admission. The repaired admission is static-only; per-phase pins
    # belong in write-once attempt-dir pin files, so this must be refused
    # pre-mutation.
    journal = make_journal(tmp_path, native_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    profile = plan_value["native_validator"]["admission_template"]["source_profile"]
    profile["profile_digest_before_restart"] = "0" * 64
    profile["profile_digest_after_restart"] = "1" * 64
    validator = NativeHandoffValidator(plan_value, journal, state)
    result = validator.pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "source_profile_invalid"


def test_before_report_refuses_binding_not_equal_to_phase_pin(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    report = copy.deepcopy(EXAMPLE["report_before"])
    report["binding"]["profile_digest"] = "e" * 64
    write_report(journal, "before_restart", report)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "report_profile_binding_mismatch"


def test_before_report_refuses_missing_pin_file(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"], pin=False)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "pin_file_missing"


def test_before_report_refuses_wrong_epoch_pin(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_pin(journal, "before_restart", EXAMPLE["report_before"]["observed"],
              runtime_epoch="synthetic-runtime-epoch-after")
    write_report(journal, "before_restart", EXAMPLE["report_before"], pin=False)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "pin_epoch_mismatch"


def test_before_report_refuses_non_deriving_observed_incarnation(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    report = copy.deepcopy(EXAMPLE["report_before"])
    # The pin still carries the report's binding digest, but the observed
    # incarnation no longer re-derives it: the fabricated-epoch anti-receipt.
    report["observed"]["runtime_epoch"] = "synthetic-runtime-epoch-fabricated"
    report["negative_control_retained"]["runtime_epoch"] = report["observed"]["runtime_epoch"]
    write_pin(journal, "before_restart", report["observed"],
              digest=report["binding"]["profile_digest"])
    write_report(journal, "before_restart", report, pin=False)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "report_profile_derivation_mismatch"


def test_after_report_refuses_derivation_mismatch(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    assert make_validator(journal).before_restart()["ok"] is True
    after_state(journal)
    report = copy.deepcopy(EXAMPLE["report_after"])
    report["observed"]["runtime_generation"] = 43
    report["cases"]["stale_handle_post"]["fresh_enrollment_runtime_epoch"] = \
        report["observed"]["runtime_epoch"]
    write_pin(journal, "after_restart", report["observed"],
              digest=report["binding"]["profile_digest"])
    write_report(journal, "after_restart", report, pin=False)
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is False and result["code"] == "report_profile_derivation_mismatch"


def test_after_phase_refuses_equal_before_and_after_pins(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    before_state(journal)
    write_report(journal, "before_restart", EXAMPLE["report_before"])
    assert make_validator(journal).before_restart()["ok"] is True
    after_state(journal)
    report = copy.deepcopy(EXAMPLE["report_after"])
    # Fabricated pre-restart: the after incarnation is rewritten to the
    # before one (the only way an equal pin stays derivation-consistent) and
    # the after pin file is an exact copy of the before pin.
    report["observed"] = copy.deepcopy(EXAMPLE["report_before"]["observed"])
    report["cases"]["stale_handle_post"]["fresh_enrollment_runtime_epoch"] = \
        report["observed"]["runtime_epoch"]
    report["binding"]["profile_digest"] = EXAMPLE["report_before"]["binding"]["profile_digest"]
    before_pin = json.loads(
        (journal.directory / "admission-pin-before_restart.json").read_text())
    life.private_json(journal.directory / "admission-pin-after_restart.json",
                      dict(before_pin, phase="after_restart"), exclusive=True)
    write_report(journal, "after_restart", report, pin=False)
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is False
    assert result["code"] == "profile_digest_pins_not_per_boot"


# -- registered-method admission (registry-driven validator) ------------------
# The single-method rejection: a plan naming the calibrated method used to be
# refused at the hardcoded method constant. The validator now consults the
# registry; every registered method with its declared fieldset validates and
# every unregistered id still fails closed.
from occ3_sglang import native_profile as registry

PER_MESSAGE = registry.PER_MESSAGE_METHOD
CALIBRATED = registry.CALIBRATED_METHOD


def calibrated_pinned():
    """Synthetic calibrated-method pinned inputs (registry-advertised id)."""
    block = copy.deepcopy(PINNED)
    block["method_id"] = CALIBRATED
    block["static_input_sha256"] = "7" * 64
    profile = block["admission_template"]["source_profile"]
    profile["method"] = CALIBRATED
    profile["calibration_path"] = "/occ3/calibration/synthetic-frozen-artifact.bin"
    profile["calibration_digest"] = "5" * 64
    identity = block["admission_template"]["identity"]
    identity["latent_method"] = CALIBRATED
    block["admission_template"]["fingerprint"] = "sha256:" + nv._canonical_sha256(identity)
    return block


def calibrated_plan(**changes):
    return plan(validator=VALIDATOR_KEY, native_validator=calibrated_pinned(), **changes)


def expected_static_input(plan_value):
    """The static derivation input digest this plan's pins must re-pin."""
    block = plan_value["native_validator"]
    if block["method_id"] == PER_MESSAGE:
        return nv.PIN_STATIC_INPUT_SHA256
    return block["static_input_sha256"]


def pin_document_for(plan_value, phase, observed, **overrides):
    document = {
        "schema": PIN_SCHEMA,
        "phase": phase,
        "attempt_id": ATTEMPT,
        "static_input_sha256": expected_static_input(plan_value),
        "runtime_epoch": observed["runtime_epoch"],
        "runtime_generation": observed["runtime_generation"],
        "profile_digest": nv._derived_profile_digest(
            plan_value["native_validator"]["admission_template"]["source_profile"], observed),
        "written_at": int(time.time()),
    }
    document.update(overrides)
    return document


def write_pin_for(plan_value, journal, phase, observed, **overrides):
    life.private_json(journal.directory / ("admission-pin-" + phase + ".json"),
                      pin_document_for(plan_value, phase, observed, **overrides), exclusive=True)


def calibrated_report(plan_value, phase):
    """Example report rewritten onto the calibrated method and its profile."""
    report = copy.deepcopy(EXAMPLE["report_before" if phase == "before_restart"
                                    else "report_after"])
    block = plan_value["native_validator"]
    report["binding"]["method_id"] = block["method_id"]
    report["binding"]["identity_fingerprint"] = block["admission_template"]["fingerprint"]
    report["binding"]["profile_digest"] = nv._derived_profile_digest(
        block["admission_template"]["source_profile"], report["observed"])
    return report


def write_calibrated_report(plan_value, journal, phase, report):
    report = copy.deepcopy(report)
    report["completed_at"] = int(time.time()) + 30
    life.private_json(journal.directory / ("native-phase-suite-" + phase + ".json"), report)
    write_pin_for(plan_value, journal, phase, report["observed"])


def test_calibrated_method_plan_is_admitted_at_preflight(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is True, result
    assert result["method_id"] == CALIBRATED
    assert result["bindings"]["identity_fingerprint"] == \
        calibrated_pinned()["admission_template"]["fingerprint"]


def test_unregistered_method_string_is_still_refused(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    plan_value["native_validator"]["method_id"] = \
        "statebridge.glm53.final-prenorm-predictor.unknown.v1"
    result = NativeHandoffValidator(plan_value, journal, state).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "method_mismatch"


def test_calibrated_method_runs_the_full_phase_flow(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = state["plan"]
    before_state(journal)
    write_calibrated_report(plan_value, journal, "before_restart",
                           calibrated_report(plan_value, "before_restart"))
    result = make_validator(journal).before_restart()
    assert result["ok"] is True, result
    assert result["method_id"] == CALIBRATED
    challenge = json.loads(
        (journal.directory / "validator-challenge-before_restart.json").read_text())
    assert challenge["method_id"] == CALIBRATED
    after_state(journal)
    write_calibrated_report(plan_value, journal, "after_restart",
                           calibrated_report(plan_value, "after_restart"))
    result = make_validator(journal, phase="after_restart").after_restart()
    assert result["ok"] is True, result


def test_per_message_profile_may_not_carry_a_calibration_reference(tmp_path):
    journal = make_journal(tmp_path, native_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    profile = plan_value["native_validator"]["admission_template"]["source_profile"]
    profile["calibration_path"] = "/occ3/calibration/synthetic-frozen-artifact.bin"
    profile["calibration_digest"] = "5" * 64
    result = NativeHandoffValidator(plan_value, journal, state).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "source_profile_invalid"


def test_calibrated_profile_requires_its_calibration_reference(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    profile = plan_value["native_validator"]["admission_template"]["source_profile"]
    profile.pop("calibration_path")
    profile.pop("calibration_digest")
    result = NativeHandoffValidator(plan_value, journal, state).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "source_profile_invalid"


@pytest.mark.parametrize("field,value,code", [
    ("calibration_path", "relative/path.bin", "source_calibration_path_invalid"),
    ("calibration_path", "", "source_calibration_path_invalid"),
    ("calibration_digest", "not-a-digest", "source_calibration_digest_invalid"),
])
def test_calibrated_reference_must_be_absolute_and_digest_pinned(tmp_path, field, value, code):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    profile = plan_value["native_validator"]["admission_template"]["source_profile"]
    profile[field] = value
    result = NativeHandoffValidator(plan_value, journal, state).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == code


def test_calibrated_identity_must_name_the_same_method(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    identity = plan_value["native_validator"]["admission_template"]["identity"]
    identity["latent_method"] = PER_MESSAGE
    plan_value["native_validator"]["admission_template"]["fingerprint"] = \
        "sha256:" + nv._canonical_sha256(identity)
    result = NativeHandoffValidator(plan_value, journal, state).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "identity_method_mismatch"


def test_calibrated_source_profile_must_name_the_same_method(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    profile = plan_value["native_validator"]["admission_template"]["source_profile"]
    profile["method"] = PER_MESSAGE
    result = NativeHandoffValidator(plan_value, journal, state).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "source_method_mismatch"


def test_calibrated_plan_without_declared_static_input_is_refused(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = copy.deepcopy(state["plan"])
    plan_value["native_validator"].pop("static_input_sha256")
    result = NativeHandoffValidator(plan_value, journal, state).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "admission_fieldset_invalid"


def test_calibrated_pin_must_repin_the_declared_static_input(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = state["plan"]
    before_state(journal)
    report = calibrated_report(plan_value, "before_restart")
    report["completed_at"] = int(time.time()) + 30
    life.private_json(journal.directory / "native-phase-suite-before_restart.json", report)
    write_pin_for(plan_value, journal, "before_restart", report["observed"],
                  static_input_sha256=nv.PIN_STATIC_INPUT_SHA256)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "pin_static_input_mismatch"


# -- readiness records cannot survive an incompatible change ------------------
def test_profile_digest_binds_the_calibration_artifact():
    profile = calibrated_pinned()["admission_template"]["source_profile"]
    observed = EXAMPLE["report_before"]["observed"]
    base = nv._derived_profile_digest(profile, observed)
    swapped = nv._derived_profile_digest(dict(profile, calibration_digest="6" * 64), observed)
    assert base != swapped
    # The per-message profile digests stay byte-identical to the frozen wire
    # contract: no calibration fields enter its derivation.
    assert nv._derived_profile_digest(
        PINNED["admission_template"]["source_profile"], observed) == FIXTURE_PIN_BEFORE


def test_readiness_record_cannot_survive_an_artifact_digest_change(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = state["plan"]
    before_state(journal)
    write_calibrated_report(plan_value, journal, "before_restart",
                           calibrated_report(plan_value, "before_restart"))
    assert make_validator(journal).before_restart()["ok"] is True
    # The frozen artifact is swapped under the plan without re-admission: the
    # after-phase binding still names the OLD artifact's profile digest, so the
    # derivation receipt must refuse it by name.
    after_state(journal)
    swapped = copy.deepcopy(plan_value)
    swapped["native_validator"]["admission_template"]["source_profile"]["calibration_digest"] = \
        "6" * 64
    report = calibrated_report(plan_value, "after_restart")
    report["completed_at"] = int(time.time()) + 30
    life.private_json(journal.directory / "native-phase-suite-after_restart.json", report)
    write_pin_for(plan_value, journal, "after_restart", report["observed"])
    validator = NativeHandoffValidator(swapped, journal, journal.read())
    result = validator.after_restart()
    assert result["ok"] is False and result["code"] == "report_profile_derivation_mismatch"


def test_readiness_record_cannot_survive_a_profile_digest_change(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = state["plan"]
    before_state(journal)
    report = calibrated_report(plan_value, "before_restart")
    report["completed_at"] = int(time.time()) + 30
    life.private_json(journal.directory / "native-phase-suite-before_restart.json", report)
    write_pin_for(plan_value, journal, "before_restart", report["observed"],
                  profile_digest="e" * 64)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "report_profile_binding_mismatch"


def test_readiness_record_cannot_survive_a_method_change(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = state["plan"]
    before_state(journal)
    report = calibrated_report(plan_value, "before_restart")
    report["binding"]["method_id"] = PER_MESSAGE
    report["completed_at"] = int(time.time()) + 30
    life.private_json(journal.directory / "native-phase-suite-before_restart.json", report)
    write_pin_for(plan_value, journal, "before_restart", report["observed"])
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "report_method_mismatch"


def test_readiness_record_cannot_survive_a_runtime_generation_change(tmp_path):
    journal = make_journal(tmp_path, calibrated_plan())
    state = journal.read()
    plan_value = state["plan"]
    before_state(journal)
    report = calibrated_report(plan_value, "before_restart")
    report["completed_at"] = int(time.time()) + 30
    life.private_json(journal.directory / "native-phase-suite-before_restart.json", report)
    write_pin_for(plan_value, journal, "before_restart", report["observed"],
                  runtime_generation=report["observed"]["runtime_generation"] + 1)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "pin_generation_mismatch"
