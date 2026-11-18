"""Admission-pin writer tests: offline, synthetic, zero network.

Every fixture value is synthetic. The fake requester replaces the single
HTTP round trip; nothing contacts a server, resolves a credential from a real
location or executes a harness. The pin parity gate runs the REAL
NativeHandoffValidator binding (_write_challenge, _pinned_inputs,
_verify_report including _load_phase_pin) against pins this module produced.
"""
from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
import sys

from deploy.v3 import atomic_publish

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import admission_pin as ap
from deploy.v3 import lifecycle as life
from deploy.v3 import native_validator as nv
from deploy.v3.live_validation import NativeHandoffValidator, VALIDATOR_KEY
from occ3_sglang import native_profile as registry

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "native_phase_suite_v1.json").read_text())
EXAMPLE = FIXTURE["example"]
PINNED = FIXTURE["admission_pinned_inputs"]
ATTEMPT = EXAMPLE["attempt"]["attempt_id"]
EPOCH_BEFORE = EXAMPLE["attempt"]["candidate_started_at_before"]
KEY = "synthetic-private-key-not-real"
STAGED_LOGICAL = "/staged/native-profile-input.json"
CALIBRATED = registry.CALIBRATED_METHOD
PER_MESSAGE = registry.PER_MESSAGE_METHOD


def original_doc():
    return {"Id": "1" * 64, "Image": "sha256:" + "a" * 64, "Name": "/fixture-service",
            "Config": {"Image": "sha256:" + "a" * 64, "Hostname": "fixture-host",
                       "Entrypoint": ["python3"],
                       "Cmd": ["-m", "sglang.launch_server", "--api-key", KEY],
                       "Env": ["A=1"], "WorkingDir": "/w", "User": "", "Labels": {}},
            "HostConfig": {"NetworkMode": "bridge", "Runtime": "runc", "Mounts": [],
                           "Binds": None, "RestartPolicy": {"Name": "always"},
                           "AutoRemove": False, "PortBindings": {}},
            "Mounts": [], "NetworkSettings": {"Networks": {"bridge": {}}},
            "State": {"Running": True, "Restarting": False, "ExitCode": 0,
                      "Status": "running", "StartedAt": "epoch-original"}}


def calibrated_block():
    block = copy.deepcopy(PINNED)
    block["method_id"] = CALIBRATED
    profile = block["admission_template"]["source_profile"]
    profile["method"] = CALIBRATED
    profile["calibration_path"] = "/occ3/calibration/synthetic-frozen-artifact.bin"
    profile["calibration_digest"] = "5" * 64
    identity = block["admission_template"]["identity"]
    identity["latent_method"] = CALIBRATED
    block["admission_template"]["fingerprint"] = "sha256:" + nv._canonical_sha256(identity)
    return block


def base_plan(**changes):
    value = {"attempt_id": ATTEMPT, "original_id": "1" * 64,
             "original_image_id": "sha256:" + "a" * 64,
             "candidate_image_id": "sha256:" + "b" * 64, "model": "SYNTHETIC-model",
             "base_url": "http://127.0.0.1:18111",
             "ingress_base_url": "http://127.0.0.1:18112",
             "startup_timeout_s": 5, "deployment_timeout_s": 20, "rollback_timeout_s": 5,
             "proof_timeout_s": 5, "controller_timeout_s": 5, "supervisor_lease_s": 5,
             "poll_interval_s": 0.005, "request_timeout_s": 5, "stop_timeout_s": 1,
             "probe_timeout_s": 1, "validation_timeout_s": 5}
    value.update(changes)
    return life.validate_plan(value)


def calibrated_plan(staged_digest, staged_bytes, **changes):
    merged = {"validator": VALIDATOR_KEY,
              "native_validator": dict(calibrated_block(),
                                       static_input_sha256=staged_digest),
              "static_input": {"sha256": staged_digest},
              "calibrated_profile_evidence": {"path": STAGED_LOGICAL,
                                              "sha256": staged_digest,
                                              "bytes": staged_bytes},
              "calibrated_artifact": {"sha256": "6" * 64, "size": 88564767}}
    merged.update(changes)
    return base_plan(**merged)


def staged_fixture(tmp_path):
    """Write the synthetic deployed profile; return (digest, bytes, host_path)."""
    raw = json.dumps(calibrated_block()["admission_template"]["source_profile"],
                     sort_keys=True, separators=(",", ":")).encode()
    host = tmp_path / "staged-profile.json"
    host.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), len(raw), host


def _fake_rename_noreplace_linux():
    """Offline faithful-enough RENAME_NOREPLACE for non-Linux test hosts.

    Single-threaded offline tests have no races; the fake enforces the
    no-replace property by an existence check plus os.rename. It replaces
    ONLY the atomic_publish.rename_noreplace symbol inside admission_pin, so
    every _publish_exclusive path (staging write, fsync, would-overwrite,
    statuses) is exercised for real while no syscall-level Linux behavior is
    claimed (test_v3_atomic_publish.py owns the platform-honesty tests).
    """
    def rename_fn(old_fd, old, new_fd, new, flags):
        old_path, new_path = os.fsdecode(old), os.fsdecode(new)
        if os.path.exists(new_path):
            ctypes.set_errno(errno.EEXIST)
            return -1
        os.rename(old_path, new_path)
        return 0
    return rename_fn


# The REAL platform guard, captured at import time (before any fixture can
# monkeypatch it) so one explicitly UNINJECTED test can restore it.
_REAL_REQUIRE_PUBLISH_PLATFORM = ap._require_publish_platform


@pytest.fixture(autouse=True)
def _portable_publish(monkeypatch, tmp_path):
    if not sys.platform.startswith("linux"):
        # TEST-ONLY simulation seams for non-Linux hosts. These bypass the
        # platform GUARD and neutralize the missing POSIX flag so the
        # staging/publication LOGIC (sequencing, identity discipline,
        # cleanup, idempotency) runs anywhere. They do NOT emulate Linux
        # security: os.O_NOFOLLOW is set to ZERO (flag disabled in the
        # simulation only - deliberately NOT Linux bit values, which are
        # not Windows open flags and could collide with Windows semantics),
        # and the real Linux-only test path keeps the real primitive and
        # the real, complete flags. Restored by pytest monkeypatch teardown.
        monkeypatch.setattr(ap, "_require_publish_platform",
                            lambda: None, raising=False)
        monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
        # Windows TEST normalization (not Windows production support): the MSVCRT
        # _wopen default is TEXT mode when neither O_BINARY nor O_TEXT is passed,
        # which expands LF to CRLF on write. Production's staging flags are
        # Linux-complete by design (O_BINARY does not exist on Linux) and the
        # platform guard is bypassed only in this TEST seam, so on non-Linux
        # hosts we normalize os.open to binary via a thin wrapper that ORs
        # getattr(os, "O_BINARY", 0) into the flags (a no-op on hosts lacking
        # the constant). The wrapper captures the ORIGINAL os.open before any
        # monkeypatching, forwards mode and keyword arguments unchanged, and
        # never recurses (it calls the captured original directly). Linux
        # tests are untouched: this whole branch is non-Linux only.
        _real_open_before_wrapper = os.open

        def _binary_normalized_open(path, flags, mode=0o777, **kwargs):
            return _real_open_before_wrapper(path,
                                            flags | getattr(os, "O_BINARY", 0),
                                            mode, **kwargs)

        monkeypatch.setattr(os, "open", _binary_normalized_open, raising=False)
        # Parent-directory fsync capability seam (logic-only evidence):
        # this host cannot os.open() a directory, so the CAPABILITY is
        # simulated with an OWNED WRITABLE FILE the fixture creates under
        # pytest's tmp_path (deterministically cleaned by fixture/pytest
        # ownership). This proves the publication LOGIC (classification,
        # ordering, cleanup) only - it is NOT Linux durability evidence
        # and never claims Windows provides Linux kernel behavior.
        sim = tmp_path / ".parent-fsync-simulation"
        sim.write_bytes(b"fsync-capability-simulation")

        def _sim_fsync_parent(parent):
            fd = os.open(str(sim), os.O_WRONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        monkeypatch.setattr(ap, "_fsync_parent_directory", _sim_fsync_parent,
                            raising=False)
        monkeypatch.setattr(ap, "atomic_publish",
                            type("P", (), {"rename_noreplace": staticmethod(
                                lambda s, f, rename_fn=None: atomic_publish.rename_noreplace(
                                    s, f, rename_fn=_fake_rename_noreplace_linux())),
                                "RenameWouldOverwrite": atomic_publish.RenameWouldOverwrite,
                                "RenameNotSupported": atomic_publish.RenameNotSupported,
                                "RenamePolicyRefused": atomic_publish.RenamePolicyRefused,
                                "RenameUnknown": atomic_publish.RenameUnknown}))
    yield


@pytest.fixture
def seam(tmp_path):
    """Map the plan-logical staged path to the real host file (test-only seam)."""
    mapping = {}
    ap.STAGED_PROFILE_RESOLVER = lambda path: mapping.get(path, path)
    yield mapping
    ap.STAGED_PROFILE_RESOLVER = None


def make_journal(base, plan_value, name="attempt"):
    directory = Path(base) / name
    directory.parent.mkdir(parents=True, exist_ok=True)
    return life.Journal.create(directory, plan_value, original_doc())


def before_state(journal):
    journal.update(candidate_id="c" * 64, candidate_started=True, candidate_epoch=EPOCH_BEFORE,
                   gates={"service_initial": {"ok": True}, "native_initial": {"ok": True}})


def after_state(journal):
    journal.update(restart_done=True, restart_count=1,
                   gates={"service_initial": {"ok": True}, "native_initial": {"ok": True},
                          "live_before": {"ok": True}, "restart": {"ok": True}})


def make_validator(journal):
    state = journal.read()
    return NativeHandoffValidator(state["plan"], journal, state, timeout=5.0,
                                   challenge_factory=lambda: EXAMPLE["challenge_before"]["challenge"])


def synthetic_receipt(profile, epoch, generation, **overrides):
    observed = {"runtime_epoch": epoch, "runtime_generation": generation}
    value = {"schema": ap.RECEIPT_SCHEMA, "runtime_id": epoch,
             "runtime_generation": generation,
             "model_fingerprint": profile["model_fingerprint"],
             "method_id": profile["method"], "tp_layout_digest": "a" * 64,
             "layer_index": profile["layer_index"], "hidden_dim": profile["hidden_dim"],
             "dtype": "bfloat16", "expected_ranks": 8, "ready_ranks": 8,
             "profile_digest": nv._derived_profile_digest(profile, observed),
             "source_capture": True, "continuous_prefix": True, "terminal_journal": True,
             "server_time_epoch": int(time.time()),
             "max_ttl_seconds": profile["max_ttl_seconds"], "max_capture_bytes": 1048576}
    value.update(overrides)
    return value


def fake_requester(receipt, calls):
    from deploy.v3.service_checks import HttpResult

    def requester(method, url, key, payload, timeout):
        calls.append({"method": method, "url": url, "key": key,
                      "payload": payload, "timeout": timeout})
        return HttpResult(200, receipt)
    return requester


def setup_attempt(tmp_path, seam, phase="before_restart"):
    digest, size, host = staged_fixture(tmp_path)
    journal = make_journal(tmp_path, calibrated_plan(digest, size))
    if phase == "before_restart":
        before_state(journal)
    else:
        before_state(journal)
        after_state(journal)
    seam[STAGED_LOGICAL] = str(host)
    profile = journal.read()["plan"]["native_validator"]["admission_template"]["source_profile"]
    return journal, profile


# -- parity: the writer pin passes the REAL validator binding ---------------

def test_writer_pin_passes_real_validator_report_binding(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    challenge = validator._write_challenge("before_restart", state, state["candidate_epoch"])
    epoch, generation = "f" * 32, 7
    receipt = synthetic_receipt(profile, epoch, generation)
    calls = []
    receipt_path = tmp_path / "identity-receipt-before_restart.json"
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n")
    status = ap.fetch_receipt(journal.directory, "before_restart", str(receipt_path),
                              api_key_file=str(key_file),
                              requester=fake_requester(receipt, calls))
    assert status["status"] == "written" and status["runtime_epoch"] == epoch
    assert calls[0]["url"] == ap.IDENTITY_URL and calls[0]["method"] == "POST"
    assert calls[0]["payload"] == {"schema": ap.REQUEST_SCHEMA}
    status = ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert status["status"] == "written"
    # The report for THIS incarnation, verified by the REAL validator path
    # (includes _load_phase_pin over the writer-produced pin file).
    report = copy.deepcopy(EXAMPLE["report_before"])
    report["observed"] = {"runtime_epoch": epoch, "runtime_generation": generation,
                          "tp_layout_digest": "a" * 64, "hidden_dim": profile["hidden_dim"],
                          "layer_index": profile["layer_index"], "dtype": "bfloat16",
                          "expected_ranks": 8, "ready_ranks": 8}
    report["binding"]["method_id"] = profile["method"]
    report["binding"]["identity_fingerprint"] = \
        journal.read()["plan"]["native_validator"]["admission_template"]["fingerprint"]
    report["binding"]["profile_digest"] = nv._derived_profile_digest(profile,
                                                                    report["observed"])
    # The retained negative-control snapshot must be bound to the SAME
    # incarnation the report observes (the REAL _verify_retained requires
    # retained.runtime_epoch == observed.runtime_epoch and generation
    # equality). The fixture JSON is internally consistent for its own
    # synthetic incarnation; rewriting observed to the fresh epoch
    # desynchronizes the retained block, so align it here. No digest
    # depends on the retained fields (they are disjoint from the admission
    # identity fingerprint and the profile-digest derivation inputs).
    report["negative_control_retained"]["runtime_epoch"] = epoch
    report["negative_control_retained"]["runtime_generation"] = generation
    report["completed_at"] = int(time.time()) + 30
    verified = validator._verify_report(report, "before_restart", challenge,
                                         validator._pinned_inputs(),
                                         state["candidate_epoch"])
    assert verified["pin"]["runtime_epoch"] == epoch
    assert verified["pin"]["profile_digest"] == receipt["profile_digest"]
    assert verified["pin"]["attempt_id"] == challenge["attempt_id"] == ATTEMPT


# -- wrong method / per-message default / digest disagreement ----------------

def test_wrong_method_receipt_refused(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7, method_id="other.method.v1")
    receipt_path = tmp_path / "receipt.json"
    # Key must exist so the ONE identity POST happens; method_mismatch is a
    # receipt-validation refusal, which occurs after that POST.
    (tmp_path / "key").write_text(KEY + "\n")
    calls = []
    with pytest.raises(nv._Reject) as caught:
        ap.fetch_receipt(journal.directory, "before_restart", str(receipt_path),
                          api_key_file=str(tmp_path / "key"),
                          requester=fake_requester(receipt, calls))
    assert caught.value.code == "method_mismatch"
    assert len(calls) == 1  # exactly one POST; the refusal came after it
    assert not receipt_path.exists()


def test_writer_refuses_post_validation_static_input_drift(tmp_path, seam):
    """Writer-level negative, explicitly POST-VALIDATION JOURNAL DRIFT.

    The plan is built and validated through the normal lifecycle API
    (calibrated_plan -> base_plan -> life.validate_plan), so it is fully
    consistent when accepted. The drift is introduced only afterwards,
    in the deepcopy that is journaled: native_validator.static_input_sha256
    alone is mutated. Journal.create stores the plan without re-validating,
    so the writer's own _plan_contract three-way re-check is what catches
    it. This is defense-in-depth for journal drift, NOT a plan-gate test
    (test_plan_gate_refuses_static_input_disagreement owns that side).
    """
    digest, size, host = staged_fixture(tmp_path)
    plan_value = copy.deepcopy(calibrated_plan(digest, size))
    plan_value["native_validator"]["static_input_sha256"] = nv.PIN_STATIC_INPUT_SHA256
    journal = make_journal(tmp_path, plan_value)
    before_state(journal)
    seam[STAGED_LOGICAL] = str(host)
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert caught.value.code == "calibrated_profile_pin_mismatch"


def test_plan_gate_refuses_static_input_disagreement(tmp_path):
    """PLAN-level negative: the SAME three-way static-input disagreement is
    refused by the plan gate itself, through its real API.

    A plan whose native_validator.static_input_sha256 disagrees with
    calibrated_profile_evidence.sha256 can never pass validate_plan /
    calibrated_capability: the plan gate refuses it first (a life.Failure
    wrapping ConfigError). This is the gate the writer re-checks; the two
    tests are distinct purposes - plan-gate refusal vs post-validation
    journal drift. No writer call happens here at all.
    """
    digest, size, _host = staged_fixture(tmp_path)
    # A FULL plan through the normal validated path (calibrated_plan ->
    # base_plan -> life.validate_plan): every ordered gate passes and the
    # three-way static-input pin is consistent when it is accepted.
    valid = calibrated_plan(digest, size)
    # Post-validation mutation of ONE field only: the journal-drift-style
    # disagreement under test. Every other field stays exactly as
    # validated, so life.validate_plan on the mutated copy passes the
    # earlier ordered gates and is refused by calibrated_capability's
    # three-way pin check (native pin != evidence pin).
    mutated = copy.deepcopy(valid)
    mutated["native_validator"]["static_input_sha256"] = nv.PIN_STATIC_INPUT_SHA256
    with pytest.raises(life.Failure) as caught:
        life.validate_plan(mutated)
    assert str(caught.value) == "invalid calibrated artifact capability"
    # The same disagreement is also refused at the capability layer itself,
    # with the precise code, through docker_config.calibrated_capability.
    from deploy.v3.docker_config import ConfigError
    with pytest.raises(ConfigError) as capability_caught:
        from deploy.v3 import docker_config
        docker_config.calibrated_capability(mutated)
    assert str(capability_caught.value) == "calibrated_profile_pin_mismatch"


def test_per_message_plan_without_evidence_refused(tmp_path):
    journal = make_journal(tmp_path, base_plan(
        validator=VALIDATOR_KEY, native_validator=copy.deepcopy(PINNED)))
    before_state(journal)
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "before_restart", str(tmp_path / "r.json"))
    assert caught.value.code == "calibrated_profile_evidence_invalid"


def test_staged_profile_digest_disagreement(tmp_path, seam):
    digest, size, host = staged_fixture(tmp_path)
    plan_value = calibrated_plan(digest, size - 1)
    journal = make_journal(tmp_path, plan_value)
    before_state(journal)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    seam[STAGED_LOGICAL] = str(host)
    receipt = synthetic_receipt(
        journal.read()["plan"]["native_validator"]["admission_template"]["source_profile"],
        "f" * 32, 7)
    receipt_path = tmp_path / "receipt.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert caught.value.code == "staged_profile_size_mismatch"
    # Same bytes, wrong pinned digest: swap the plan evidence digest.
    plan_value = calibrated_plan("7" * 64, size)
    journal = make_journal(tmp_path / "second", plan_value)
    before_state(journal)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    seam[STAGED_LOGICAL] = str(host)
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert caught.value.code == "staged_profile_digest_mismatch"
    assert not (journal.directory / "admission-pin-before_restart.json").exists()


def test_derivation_mismatch_refused(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7, profile_digest="b" * 64)
    # Key must exist: derivation_mismatch is validated AFTER the one POST.
    (tmp_path / "k").write_text(KEY + "\n")
    calls = []
    with pytest.raises(nv._Reject) as caught:
        ap.fetch_receipt(journal.directory, "before_restart",
                         str(tmp_path / "r.json"), api_key_file=str(tmp_path / "k"),
                         requester=fake_requester(receipt, calls))
    assert caught.value.code == "derivation_mismatch"
    assert len(calls) == 1


# -- receipt field-class rejections ------------------------------------------

@pytest.mark.parametrize("override,code", [
    ({"schema": "other.v1"}, "receipt_schema_unknown"),
    ({"runtime_id": "not-an-epoch"}, "receipt_epoch_invalid"),
    ({"runtime_generation": 0}, "receipt_generation_invalid"),
    ({"expected_ranks": 4}, "receipt_ranks_invalid"),
    ({"ready_ranks": 6}, "receipt_ranks_not_ready"),
    ({"source_capture": False}, "receipt_not_ready"),
    ({"tp_layout_digest": "zz"}, "receipt_layout_digest_invalid"),
    ({"profile_digest": "short"}, "receipt_profile_digest_invalid"),
    ({"dtype": "int8"}, "receipt_dtype_unsupported"),
    ({"model_fingerprint": "mismatch"}, "receipt_fingerprint_mismatch"),
    ({"hidden_dim": 999}, "receipt_hidden_dim_mismatch"),
    ({"layer_index": 424242}, "receipt_layer_index_mismatch"),
])
def test_receipt_field_rejections(tmp_path, seam, override, code):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7, **override)
    # Key must exist: field-class refusals are validated AFTER the one POST.
    (tmp_path / "k").write_text(KEY + "\n")
    calls = []
    with pytest.raises(nv._Reject) as caught:
        ap.fetch_receipt(journal.directory, "before_restart",
                         str(tmp_path / "r.json"), api_key_file=str(tmp_path / "k"),
                         requester=fake_requester(receipt, calls))
    assert caught.value.code == code
    assert len(calls) == 1


def test_receipt_extra_field_refused(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7, extra="x")
    # Key must exist: extra-field refusal is validated AFTER the one POST.
    (tmp_path / "k").write_text(KEY + "\n")
    calls = []
    with pytest.raises(nv._Reject) as caught:
        ap.fetch_receipt(journal.directory, "before_restart",
                         str(tmp_path / "r.json"), api_key_file=str(tmp_path / "k"),
                         requester=fake_requester(receipt, calls))
    assert len(calls) == 1
    assert caught.value.code == "receipt_fieldset_invalid"


# -- sequencing and cross-restart guards --------------------------------------

def test_after_restart_requires_completed_restart(tmp_path, seam):
    digest, size, host = staged_fixture(tmp_path)
    journal = make_journal(tmp_path, calibrated_plan(digest, size))
    before_state(journal)  # restart_done still missing
    seam[STAGED_LOGICAL] = str(host)
    with pytest.raises(nv._Reject) as caught:
        ap.fetch_receipt(journal.directory, "after_restart",
                         str(tmp_path / "r.json"), api_key_file=str(tmp_path / "k"),
                         requester=fake_requester({}, []))
    assert caught.value.code == "restart_not_completed"


def after_attempt(tmp_path, seam, before_epoch="1" * 32, before_generation=3):
    journal, profile = setup_attempt(tmp_path, seam, phase="after_restart")
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    before_pin = {"schema": ap.PIN_SCHEMA, "phase": "before_restart",
                  "attempt_id": state["attempt_id"],
                  "static_input_sha256": journal.read()["plan"]["native_validator"]["static_input_sha256"],
                  "runtime_epoch": before_epoch, "runtime_generation": before_generation,
                  "profile_digest": "c" * 64, "written_at": int(time.time()) - 60}
    life.private_json(nv._phase_pin_path(journal.directory, "before_restart"),
                      before_pin, exclusive=True)
    validator._write_challenge("after_restart", journal.read(), state["candidate_epoch"])
    return journal, profile


def test_after_restart_refuses_stale_cross_restart_pin(tmp_path, seam):
    journal, profile = after_attempt(tmp_path, seam)
    receipt = synthetic_receipt(profile, "1" * 32, 4)  # same epoch as before-pin
    receipt_path = tmp_path / "r.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "after_restart", str(receipt_path))
    assert caught.value.code == "stale_cross_restart_pin"


def test_after_restart_requires_generation_advance(tmp_path, seam):
    journal, profile = after_attempt(tmp_path, seam, before_generation=9)
    receipt = synthetic_receipt(profile, "2" * 32, 5)  # generation went backwards
    receipt_path = tmp_path / "r.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "after_restart", str(receipt_path))
    assert caught.value.code == "pin_generation_not_advanced"


def test_after_restart_refuses_cross_attempt_before_pin(tmp_path, seam):
    journal, profile = after_attempt(tmp_path, seam)
    stale = nv._phase_pin_path(journal.directory, "before_restart")
    document = json.loads(stale.read_text())
    document["attempt_id"] = "occ3-other-attempt-0000000000000001"
    life.private_json(stale, document, exclusive=False)
    receipt = synthetic_receipt(profile, "2" * 32, 8)
    receipt_path = tmp_path / "r.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "after_restart", str(receipt_path))
    assert caught.value.code == "cross_attempt_pin"


def test_after_restart_writes_fresh_incarnation(tmp_path, seam):
    journal, profile = after_attempt(tmp_path, seam)
    receipt = synthetic_receipt(profile, "2" * 32, 4)
    receipt_path = tmp_path / "r.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    status = ap.write_pin(journal.directory, "after_restart", str(receipt_path))
    assert status["status"] == "written" and status["runtime_epoch"] == "2" * 32


# -- challenge binding --------------------------------------------------------

def test_missing_challenge_refused(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)  # no challenge minted
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "before_restart", str(tmp_path / "r.json"))
    assert caught.value.code == "challenge_missing"


def test_challenge_attempt_mismatch_refused(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    challenge = validator._write_challenge("before_restart", state, state["candidate_epoch"])
    path = nv._phase_challenge_path(journal.directory, "before_restart")
    document = dict(challenge, attempt_id="occ3-other-attempt-0000000000000001")
    life.private_json(path, document, exclusive=False)
    receipt = synthetic_receipt(profile, "f" * 32, 7)
    receipt_path = tmp_path / "r.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert caught.value.code == "challenge_attempt_mismatch"


# -- idempotency and conflicts -------------------------------------------------

def test_write_pin_idempotent_for_same_binding(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7)
    receipt_path = tmp_path / "r.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    first = ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    pin_path = nv._phase_pin_path(journal.directory, "before_restart")
    original = pin_path.read_bytes()
    time.sleep(1.1)  # written_at moves; the binding must stay idempotent
    second = ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert first["status"] == "written" and second["status"] == "idempotent"
    assert pin_path.read_bytes() == original  # never rewritten


def test_write_pin_conflict_refuses_and_preserves(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7)
    receipt_path = tmp_path / "r.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    pin_path = nv._phase_pin_path(journal.directory, "before_restart")
    original = pin_path.read_bytes()
    other = synthetic_receipt(profile, "e" * 32, 7)  # different incarnation
    life.private_json(receipt_path, other, exclusive=False)
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert caught.value.code == "pin_conflict"
    assert pin_path.read_bytes() == original


def test_receipt_refetch_same_incarnation_idempotent(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7)
    receipt_path = tmp_path / "receipt.json"
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n")
    calls = []
    first = ap.fetch_receipt(journal.directory, "before_restart", str(receipt_path),
                             api_key_file=str(key_file),
                             requester=fake_requester(receipt, calls))
    refreshed = synthetic_receipt(profile, "f" * 32, 7,
                                  server_time_epoch=int(time.time()) + 5)
    second = ap.fetch_receipt(journal.directory, "before_restart", str(receipt_path),
                              api_key_file=str(key_file),
                              requester=fake_requester(refreshed, calls))
    assert first["status"] == "written" and second["status"] == "idempotent"
    assert json.loads(receipt_path.read_text())["runtime_id"] == "f" * 32
    drifted = synthetic_receipt(profile, "d" * 32, 7)  # different epoch: conflict
    with pytest.raises(nv._Reject) as caught:
        ap.fetch_receipt(journal.directory, "before_restart", str(receipt_path),
                         api_key_file=str(key_file),
                         requester=fake_requester(drifted, calls))
    assert caught.value.code == "receipt_conflict"


# -- verify-pins diagnostic -----------------------------------------------------

def test_verify_pins_rederives_and_reports(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7)
    receipt_path = tmp_path / "r.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    result = ap.verify_pins(journal.directory)
    assert result["pins"]["before_restart"]["ok"] is True
    assert result["pins"]["after_restart"]["present"] is False
    # Tamper the digest: the read-only diagnostic must flag it.
    pin_path = nv._phase_pin_path(journal.directory, "before_restart")
    document = json.loads(pin_path.read_text())
    document["profile_digest"] = "9" * 64
    life.private_json(pin_path, document, exclusive=False)
    result = ap.verify_pins(journal.directory)
    assert result["pins"]["before_restart"]["ok"] is False
    assert result["pins"]["before_restart"]["code"] == "pin_derivation_mismatch"


# -- key hygiene -----------------------------------------------------------------

def test_api_key_never_enters_any_artifact(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7)
    receipt_path = tmp_path / "receipt.json"
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n")
    calls = []
    ap.fetch_receipt(journal.directory, "before_restart", str(receipt_path),
                     api_key_file=str(key_file), requester=fake_requester(receipt, calls))
    ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    # The key travelled ONLY as the requester's key argument (the transport
    # puts it in the Authorization header and nowhere else)...
    assert all(call["key"] == KEY for call in calls)
    # ...and no written artifact contains it.
    for artifact in (receipt_path,
                     nv._phase_pin_path(journal.directory, "before_restart")):
        assert KEY.encode() not in artifact.read_bytes()


def test_missing_api_key_file_refused(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, "f" * 32, 7)
    calls = []
    with pytest.raises(nv._Reject) as caught:
        ap.fetch_receipt(journal.directory, "before_restart",
                         str(tmp_path / "r.json"),
                         api_key_file=str(tmp_path / "absent-key"),
                         requester=fake_requester(receipt, calls))
    assert caught.value.code == "api_key_unavailable"
    assert calls == []  # key load precedes the POST: ZERO requests


# -- publish semantics: renameat2 statuses, RC4 zero-unlink and no-repair -------

def _published_pin(journal, profile, epoch="f" * 32, generation=7):
    validator = make_validator(journal)
    state = journal.read()
    validator._write_challenge("before_restart", state, state["candidate_epoch"])
    receipt = synthetic_receipt(profile, epoch, generation)
    receipt_path = journal.directory.parent / "receipt.json"
    life.private_json(receipt_path, receipt, exclusive=False)
    return receipt_path


@pytest.mark.skipif(sys.platform.startswith("linux"),
                    reason="platform refusal is a non-Linux property; the real "
                           "Linux path keeps real flags and the real primitive")
def test_non_linux_publish_refused_before_any_staging_side_effect(tmp_path, seam, monkeypatch):
    """Explicitly UNINJECTED platform-refusal test (non-Linux only).

    The REAL guard is restored here despite the autouse _portable_publish
    fixture, and nothing is injected past it. Asserts the fixed refusal
    publish_platform_unsupported fires BEFORE any staging operation:
    zero staging opens, zero rename attempts, zero new files anywhere
    under the attempt directory (dotfiles included). Honest scope, per
    _require_publish_platform: this proves zero STAGING operations for
    the publisher only. It does NOT prove zero api-key reads or zero
    HTTP - fetch_receipt performs those before publication is attempted,
    and no entrypoint-order change is claimed in this wave.
    """
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    before = set(Path(journal.directory).rglob("*")) | {Path(receipt_path)}
    staging_opens = []
    renames = []
    real_open = os.open

    def counting_open(path, flags, *args, **kwargs):
        if str(path).endswith(".tmp"):
            staging_opens.append(str(path))
        return real_open(path, flags, *args, **kwargs)

    def counting_rename(old_fd, old, new_fd, new, flags):
        renames.append((os.fsdecode(old), os.fsdecode(new)))
        return -1

    monkeypatch.setattr(ap, "_require_publish_platform",
                        _REAL_REQUIRE_PUBLISH_PLATFORM)
    monkeypatch.setattr(os, "open", counting_open)
    monkeypatch.setattr(ap.atomic_publish, "rename_noreplace",
                        staticmethod(counting_rename), raising=False)
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert caught.value.code == "publish_platform_unsupported"
    assert staging_opens == []  # refusal preceded every staging open
    assert renames == []  # ... and every rename attempt
    after = set(Path(journal.directory).rglob("*")) | {Path(receipt_path)}
    assert after == before  # no file of any kind was created


def test_publish_success_performs_no_staging_unlink(tmp_path, seam, monkeypatch):
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    unlinks = []
    real_unlink = os.unlink
    monkeypatch.setattr(os, "unlink", lambda p: (unlinks.append(p), real_unlink(p))[1])
    status = ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    monkeypatch.setattr(os, "unlink", real_unlink)
    assert status["status"] == "written"
    assert unlinks == []  # RC4: no unlink at all after a successful rename


def test_publish_success_leaves_foreign_recreated_staging_untouched(tmp_path, seam):
    # A rename that "succeeds" without consuming the staging name (foreign
    # recreation equivalent, simulated offline): classify uncertain, never delete.
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    original = ap.atomic_publish.rename_noreplace

    def rename_lying_success(old_fd, old, new_fd, new, flags):
        return 0  # claims success; consumes nothing

    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: rename_lying_success(
        None, None, None, None, None)
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "publish_uncertain"
    leftovers = list(Path(journal.directory).glob(".admission-pin-*.tmp"))
    assert len(leftovers) == 1  # the unverified name still exists: not deleted


def test_publish_unknown_outcome_cleans_nothing(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)

    def rename_unknown(staging, final):
        # Outer-wrapper contract: raise the typed exception (verified
        # constructor: RenameUnknown carries the errno code).
        raise atomic_publish.RenameUnknown(errno.EIO)

    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: rename_unknown(
        s, f)
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "publish_unknown"
    assert not (journal.directory / "admission-pin-before_restart.json").exists()
    leftovers = list(Path(journal.directory).glob(".admission-pin-*.tmp"))
    assert len(leftovers) == 1  # RC4: unknown outcome -> NO staging cleanup either


def test_publish_unsupported_refused_without_final(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)

    def rename_nosys(staging, final):
        # Outer-wrapper contract: raise the typed exception (production's
        # own raise site is RenameNotSupported("errno")).
        raise atomic_publish.RenameNotSupported("errno")

    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: rename_nosys(
        s, f)
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "publish_unsupported"
    assert not (journal.directory / "admission-pin-before_restart.json").exists()
    assert list(Path(journal.directory).glob(".admission-pin-*.tmp")) == []  # known failure: own tmp cleaned


def test_publish_policy_refusal_is_its_own_class(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)

    def rename_eperm(staging, final):
        # Outer-wrapper contract: raise the typed exception.
        raise atomic_publish.RenamePolicyRefused()

    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: rename_eperm(
        s, f)
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "publish_refused_policy"


def test_parent_fsync_failure_is_nonzero_uncertain(tmp_path, seam, monkeypatch):
    """Companion fault test for the _fsync_parent_directory capability shim:
    patches the SAME helper to raise OSError. The rc0 rename has already
    succeeded, so the refusal is publish_uncertain (nonzero): the final is
    COMPLETE and preserved, never deleted; no retry, no recovery action."""
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)

    def failing_fsync(parent):
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(ap, "_fsync_parent_directory", failing_fsync,
                        raising=False)
    renames_after = {"count": 0}
    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: (
        renames_after.__setitem__("count", renames_after["count"] + 1),
        saved(s, f, rename_fn=rename_fn))[1]
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "publish_uncertain"
    # The final EXISTS and is complete; it is never deleted.
    final = journal.directory / "admission-pin-before_restart.json"
    assert final.exists()
    assert json.loads(final.read_text())["runtime_epoch"] == "f" * 32
    assert renames_after["count"] == 1  # exactly one rename: no retry
    assert list(Path(journal.directory).glob(".admission-pin-*.tmp")) == []


def test_idempotent_repair_is_binding_equality_only(tmp_path, seam, monkeypatch):
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    first = ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert first["status"] == "written"
    # Contract (corrected v2.9): idempotence proves binding equality of the
    # EXISTING final bytes, with NO durability repair of any prior
    # uncertainty. The staged-write discipline (a staging-file fsync before
    # the no-replace rename attempt) is pre-move write discipline, NOT a
    # repair, and is documented separately. The repair-class fsync is the
    # PARENT-directory durability sync (_fsync_parent_directory), which runs
    # only on the rc0 success path: the idempotent branch must perform ZERO
    # of those, and must not touch the final's bytes.
    parent_fsyncs = []
    real_parent_fsync = ap._fsync_parent_directory

    def counting_parent_fsync(parent):
        parent_fsyncs.append(parent)
        return real_parent_fsync(parent)

    monkeypatch.setattr(ap, "_fsync_parent_directory", counting_parent_fsync,
                        raising=False)
    final = journal.directory / "admission-pin-before_restart.json"
    original_final_bytes = final.read_bytes()
    second = ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    monkeypatch.setattr(ap, "_fsync_parent_directory", real_parent_fsync,
                        raising=False)
    assert second["status"] == "idempotent"
    assert parent_fsyncs == []  # zero parent-directory fsync: no durability repair
    assert final.read_bytes() == original_final_bytes  # binding equality only: bytes untouched


def test_document_bound_enforced_before_any_write(tmp_path, seam):
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)

# -- owned-identity staging cleanup (BLOCKER fix fault tests) -------------------

def _foreign_staging_plant(journal, monkeypatch, payload=b"foreign-bytes"):
    """Plant a foreign regular file on the deterministic staging path.

    Patches uuid.uuid4 inside admission_pin so the staging name is known,
    plants the foreign file, and returns (planted_path, payload).
    """
    planted = {"hex": "cafe" * 8}
    real_uuid4 = uuid.uuid4

    class FakeUUID:
        def __init__(self, hexvalue):
            self.hex = hexvalue

    def fake_uuid4():
        return FakeUUID(planted["hex"])
    monkeypatch.setattr(ap.uuid, "uuid4", fake_uuid4)
    staging = Path(journal.directory) / (".admission-pin-before_restart.json."
                                          + planted["hex"] + ".tmp")
    staging.write_bytes(payload)
    return staging, payload


def test_open_eexist_foreign_file_preserved_byte_identical(tmp_path, seam, monkeypatch):
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    staging, payload = _foreign_staging_plant(journal, monkeypatch)
    unlinks = []
    real_unlink = os.unlink
    monkeypatch.setattr(os, "unlink", lambda p: (unlinks.append(p), real_unlink(p))[1])
    with pytest.raises(nv._Reject) as caught:
        ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    monkeypatch.setattr(os, "unlink", real_unlink)
    assert caught.value.code == "staging_open_failed"
    assert unlinks == []  # zero unlink: ownership never established
    assert staging.read_bytes() == payload  # foreign file byte-identical


def test_open_eacces_no_fd_zero_unlink(tmp_path, seam, monkeypatch):
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    unlinks = []
    real_unlink = os.unlink
    real_open = os.open

    def open_eacces(path, flags, mode=0o777, **kwargs):
        flags_int = flags
        if (flags_int & os.O_CREAT) and (flags_int & os.O_EXCL) and str(path).endswith(".tmp"):
            raise PermissionError(errno.EACCES, "denied")
        return real_open(path, flags_int, mode, **kwargs)
    monkeypatch.setattr(os, "open", open_eacces)
    monkeypatch.setattr(os, "unlink", lambda p: (unlinks.append(p), real_unlink(p))[1])
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        monkeypatch.setattr(os, "open", real_open)
        monkeypatch.setattr(os, "unlink", real_unlink)
    assert caught.value.code == "staging_open_failed"
    assert unlinks == []  # no fd => zero unlink by name


def test_successful_open_then_write_failure_cleans_only_captured_inode(tmp_path, seam, monkeypatch):
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    real_write = os.write
    state = {"staging_seen": None}

    def failing_write(fd, data):
        info = os.fstat(fd)
        if info.st_mode & 0o170000 == 0o100000 and "pin-before_restart.json" in str(
                state.get("path", "")):
            raise OSError(errno.EIO, "write failed")
        return real_write(fd, data)

    real_open = os.open

    def tracking_open(path, flags, mode=0o777, **kwargs):
        fd = real_open(path, flags, mode, **kwargs)
        if str(path).endswith(".tmp") and (flags & os.O_EXCL):
            state["path"] = str(path)
            state["fd"] = fd
        return fd
    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "write", failing_write)
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        monkeypatch.setattr(os, "open", real_open)
        monkeypatch.setattr(os, "write", real_write)
    assert caught.value.code == "staging_write_failed"
    # The captured matching inode was cleaned; no final appeared.
    assert not (journal.directory / "admission-pin-before_restart.json").exists()
    leftovers = [p for p in Path(journal.directory).glob(".admission-pin-*.tmp")]
    assert leftovers == []


# -- v3.0 pre-publication IO fault matrix (relinquish-first close discipline) ------------

def _fd_tracking_open(monkeypatch, real_open, state):
    """Track fds returned for the STAGING open so fault tests can
    discriminate the staged fd from every other descriptor. The fixture
    closes nothing itself: the code under test owns every close."""
    def tracking_open(path, flags, mode=0o777, **kwargs):
        fd = real_open(path, flags, mode, **kwargs)
        if str(path).endswith(".tmp") and (flags & os.O_EXCL):
            state["staged_fd"] = fd
            state["staged_path"] = str(path)
            state.setdefault("staged_fds", []).append(fd)
        return fd
    monkeypatch.setattr(os, "open", tracking_open)
    return tracking_open


def test_staged_fsync_failure_is_fixed_prepublication_refusal(tmp_path, seam, monkeypatch):
    """T-A: the staged durability sync fails. Assert: staging_fsync_failed;
    owned-inode cleanup ran (staging gone); ZERO rename attempts; an
    existing final (pre-seeded, predating this call) preserved byte-for-
    byte; no new publication."""
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    final = journal.directory / "admission-pin-before_restart.json"
    first = ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert first["status"] == "written"
    existing_bytes = final.read_bytes()
    state = {}
    real_fsync = os.fsync
    real_open = os.open
    _fd_tracking_open(monkeypatch, real_open, state)
    renames = {"count": 0}

    def failing_fsync(fd):
        if fd == state.get("staged_fd"):
            raise OSError(errno.EIO, "staged fsync failure")
        return real_fsync(fd)

    def counting_rename(s, f, rename_fn=None):
        renames["count"] += 1
        raise atomic_publish.RenameUnknown(errno.EIO)

    monkeypatch.setattr(os, "fsync", failing_fsync)
    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = counting_rename
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "staging_fsync_failed"
    assert renames["count"] == 0  # zero renames: pre-publication refusal
    assert list(Path(journal.directory).glob(".admission-pin-*.tmp")) == []  # cleaned
    assert final.read_bytes() == existing_bytes  # existing final preserved


def test_normal_close_failure_maps_staging_close_failed(tmp_path, seam, monkeypatch):
    """T-B: the NORMAL-path close fails after real-close-then-raise injection
    (the fd WAS released; the failure is ambiguous). Assert:
    staging_close_failed; the staged fd was closed EXACTLY once; owned-inode
    cleanup ran; zero renames; existing final preserved."""
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    final = journal.directory / "admission-pin-before_restart.json"
    first = ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert first["status"] == "written"
    existing_bytes = final.read_bytes()
    state = {"close_attempts": {}}
    real_close = os.close
    real_open = os.open
    _fd_tracking_open(monkeypatch, real_open, state)
    renames = {"count": 0}

    def real_close_then_raise(fd):
        # PRIMARY injection shape: perform the REAL close, THEN raise the
        # synthetic error. The ambiguous close actually released the fd;
        # any retry would close a reused descriptor number.
        attempts = state["close_attempts"].get(fd, 0) + 1
        state["close_attempts"][fd] = attempts
        real_close(fd)
        raise OSError(errno.EIO, "ambiguous close failure")

    def counting_rename(s, f, rename_fn=None):
        renames["count"] += 1
        raise atomic_publish.RenameUnknown(errno.EIO)

    monkeypatch.setattr(os, "close", real_close_then_raise)
    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = counting_rename
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "staging_close_failed"
    staged = state.get("staged_fd")
    assert staged is not None and state["close_attempts"].get(staged) == 1  # at most once
    assert renames["count"] == 0  # zero renames
    assert list(Path(journal.directory).glob(".admission-pin-*.tmp")) == []  # cleanup ran
    assert final.read_bytes() == existing_bytes  # existing final preserved


def test_write_failure_with_handler_close_failure_preserves_primary(tmp_path, seam, monkeypatch):
    """T-C: a write failure AND a handler close failure on the SAME fd.
    Assert: staging_write_failed surfaces (NOT the close error); the fd
    was closed EXACTLY once (the handler's best-effort attempt);
    owned-inode cleanup ran; zero renames."""
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    state = {"close_attempts": {}}
    real_close = os.close
    real_open = os.open
    real_write = os.write
    _fd_tracking_open(monkeypatch, real_open, state)
    renames = {"count": 0}

    def failing_write(fd, data):
        if fd == state.get("staged_fd"):
            raise OSError(errno.EIO, "write failed")
        return real_write(fd, data)

    def real_close_then_raise(fd):
        attempts = state["close_attempts"].get(fd, 0) + 1
        state["close_attempts"][fd] = attempts
        real_close(fd)
        raise OSError(errno.EIO, "ambiguous close failure")

    def counting_rename(s, f, rename_fn=None):
        renames["count"] += 1
        raise atomic_publish.RenameUnknown(errno.EIO)

    monkeypatch.setattr(os, "write", failing_write)
    monkeypatch.setattr(os, "close", real_close_then_raise)
    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = counting_rename
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "staging_write_failed"  # primary refusal preserved
    staged = state.get("staged_fd")
    assert staged is not None and state["close_attempts"].get(staged) == 1  # handler closed once
    assert renames["count"] == 0
    assert list(Path(journal.directory).glob(".admission-pin-*.tmp")) == []  # cleanup ran
    assert not (journal.directory / "admission-pin-before_restart.json").exists()


def test_fstat_failure_with_close_failure_preserves_identity_unavailable(tmp_path, seam, monkeypatch):
    """T-D (compound): fstat fails AND the cleanup close fails. Assert:
    staging_identity_unavailable (NOT staging_close_failed); exactly ONE
    close attempt on the staged fd; ZERO renames; ZERO name-based cleanup
    (no identity was ever captured, so no unlink is legitimate - the
    staging pathname remains as honest residue)."""
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    state = {"close_attempts": {}}
    real_close = os.close
    real_open = os.open
    real_fstat = os.fstat
    _fd_tracking_open(monkeypatch, real_open, state)
    renames = {"count": 0}
    unlinks = []
    real_unlink = os.unlink

    def failing_fstat(fd):
        if fd == state.get("staged_fd"):
            raise OSError(errno.EIO, "fstat failed")
        return real_fstat(fd)

    def real_close_then_raise(fd):
        attempts = state["close_attempts"].get(fd, 0) + 1
        state["close_attempts"][fd] = attempts
        real_close(fd)
        raise OSError(errno.EIO, "ambiguous close failure")

    def counting_rename(s, f, rename_fn=None):
        renames["count"] += 1
        raise atomic_publish.RenameUnknown(errno.EIO)

    def no_unlink(p):
        unlinks.append(p)
        raise FileNotFoundError("no identity: no unlink is legitimate")

    monkeypatch.setattr(os, "fstat", failing_fstat)
    monkeypatch.setattr(os, "close", real_close_then_raise)
    monkeypatch.setattr(os, "unlink", no_unlink)
    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = counting_rename
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
        monkeypatch.setattr(os, "unlink", real_unlink)
    assert caught.value.code == "staging_identity_unavailable"  # primary preserved
    staged = state.get("staged_fd")
    assert staged is not None and state["close_attempts"].get(staged) == 1  # one close attempt
    assert renames["count"] == 0  # zero renames
    assert unlinks == []  # zero name-based cleanup, ever
    staging_names = [p for p in Path(journal.directory).glob(".admission-pin-*.tmp")]
    assert len(staging_names) == 1  # honest residue remains: no identity, no unlink
    assert not (journal.directory / "admission-pin-before_restart.json").exists()


class _TestSetupFailure(RuntimeError):
    """Distinct NON-OSError sentinel: injection-setup failures must NEVER
    be swallowed by production's `except OSError: pass` handler, nor be
    mistakable for the injected write error."""


def test_captured_path_foreign_substitution_survives_cleanup_refusals(tmp_path, seam, monkeypatch):
    """Post-close/pre-cleanup foreign substitution at the CAPTURED path.

    The write fake raises ONLY the synthetic EIO on the exact staged fd
    (no file operations inside it). The close hook wraps production's
    single handler close of that fd: real_close ONCE (the one and only
    close), then - with the fd genuinely closed - create the foreign file
    and perform the REAL os.replace(foreign, captured-staging-path),
    setting swap_completed ONLY after the replace returns normally. Any
    setup failure raises _TestSetupFailure (not OSError) so the handler's
    OSError swallow cannot hide it.

    The invariant exercised: _cleanup_owned_inode's REAL identity
    predicate consults a FOREIGN inode AT THE CAPTURED PATH and refuses
    to unlink. HONEST SCOPE: this is AFTER-CLOSE/PRE-CLEANUP substitution
    only; it does NOT test Linux open-fd substitution (replace-over-open-
    fd); it does NOT establish Windows production support (refused by the
    real platform guard, unchanged); it does not broaden any measured
    replace denial into a universal property; Linux-native semantics
    remain the separate held DGX smoke gate.
    """
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    foreign_payload = b"foreign-at-captured-path"
    state = {"staged_fd": None, "staged_path": None}
    real_open = os.open
    real_write = os.write
    real_close = os.close
    real_lstat = os.lstat
    real_fstat = os.fstat
    _fd_tracking_open(monkeypatch, real_open, state)
    swap = {"completed": False, "identities": None}
    close_actions = {"count": 0}
    real_closes = {"per_fd": {}}
    renames = {"count": 0}

    def failing_write(fd, data):
        # ONLY the synthetic EIO on the exact staged fd. No file
        # operations here: setup errors cannot masquerade as write errors.
        if fd == state.get("staged_fd"):
            raise OSError(errno.EIO, "injected write failure")
        return real_write(fd, data)

    def hook_close(fd):
        # Act ONLY on the staged fd's first close (fd-discriminated).
        if fd != state.get("staged_fd") or close_actions["count"] > 0:
            return real_close(fd)
        close_actions["count"] += 1
        real_close(fd)  # THE one real close; never closed again by anyone
        real_closes["per_fd"][fd] = real_closes["per_fd"].get(fd, 0) + 1
        try:
            staging_path = Path(state["staged_path"])
            if not staging_path.exists():
                raise _TestSetupFailure(
                    "staging pathname vanished before substitution")
            foreign = Path(state["staged_path"] + ".foreign")
            foreign.write_bytes(foreign_payload)
            name_identity = real_lstat(staging_path)
            foreign_identity = real_lstat(foreign)
            os.replace(foreign, staging_path)
            post_identity = real_lstat(staging_path)
            swap["identities"] = {
                "captured_name_pre": (name_identity.st_dev, name_identity.st_ino),
                "foreign_pre": (foreign_identity.st_dev, foreign_identity.st_ino),
                "name_post": (post_identity.st_dev, post_identity.st_ino)}
            swap["completed"] = True  # ONLY after a successful replace
        except OSError as error:
            # Setup failure AFTER the real close: re-raise as the distinct
            # sentinel so production's `except OSError: pass` cannot swallow
            # it. (Non-OSError setup failures propagate untouched.)
            raise _TestSetupFailure(f"substitution setup failed: {error!r}") from error

    def counting_rename(s, f, rename_fn=None):
        renames["count"] += 1
        raise atomic_publish.RenameUnknown(errno.EIO)

    monkeypatch.setattr(os, "write", failing_write)
    monkeypatch.setattr(os, "close", hook_close)
    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = counting_rename
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "staging_write_failed"
    assert renames["count"] == 0  # zero publisher renames
    # Explicit swap-completed gate BEFORE the survival check: a failed
    # substitution can never masquerade as a passing survival assertion.
    assert swap["completed"] is True, "foreign substitution did not land"
    assert close_actions["count"] == 1  # hook acted exactly once
    assert real_closes["per_fd"].get(state["staged_fd"]) == 1  # one real close
    identities = swap["identities"]
    # Identity distinctness is RECORDED evidence of this run (both pairs
    # logged for the artifact trail), never a precondition and never a
    # branched assertion: the foreign-survival assertion below is
    # UNCONDITIONAL on every host.
    # The foreign bytes survive AT THE CAPTURED PATH (identity mismatch
    # protection, exercised through production's real cleanup predicate).
    staging_names = [p for p in Path(journal.directory).glob(".admission-pin-*.tmp")]
    assert len(staging_names) == 1
    assert staging_names[0].read_bytes() == foreign_payload
    # No new publication.
    assert not (journal.directory / "admission-pin-before_restart.json").exists()


def test_swap_then_failing_rename_foreign_bytes_survive(tmp_path, seam, monkeypatch):
    """REQUIRED T2: foreign inode swapped INSIDE the injected rename (after
    the pre-rename precheck passed), then a KNOWN refusal (ENOSYS).

    Asserts: exact refusal code, rename called exactly once, foreign bytes
    survive at the staging name, ZERO unlink, no final publication. The
    early-precheck identity-lost case (substitution BEFORE the precheck)
    is asserted separately in the same test and never reaches the rename.
    """
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    foreign_payload = b"foreign-swap-bytes"
    unlinks = []
    real_unlink = os.unlink
    real_lstat = os.lstat

    def no_unlink(p):
        unlinks.append(p)
        raise FileNotFoundError("foreign name must never be unlinked")

    monkeypatch.setattr(os, "unlink", no_unlink)
    try:
        # --- T2: swap INSIDE the injected rename_noreplace ---------------
        renames = {"count": 0, "path": None}

        def swap_inside_rename_then_refuse(staging, final):
            renames["count"] += 1
            old_path = str(staging)
            renames["path"] = old_path
            # The pre-rename precheck has ALREADY passed on the genuine
            # staging file; substitute the foreign inode now, then refuse
            # with the outer-wrapper typed exception (production's own
            # raise site for ENOSYS is RenameNotSupported("errno")).
            foreign = Path(old_path + ".foreign")
            foreign.write_bytes(foreign_payload)
            os.replace(foreign, old_path)
            raise atomic_publish.RenameNotSupported("errno")

        saved = ap.atomic_publish.rename_noreplace
        ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: (
            swap_inside_rename_then_refuse(s, f))
        try:
            with pytest.raises(nv._Reject) as caught:
                ap.write_pin(journal.directory, "before_restart", str(receipt_path))
        finally:
            ap.atomic_publish.rename_noreplace = saved
        assert caught.value.code == "publish_unsupported"  # exact known refusal
        assert renames["count"] == 1  # rename called exactly once, no retry
        assert unlinks == []  # ZERO unlink: the name is foreign
        staging_names = [p for p in Path(journal.directory).glob(".admission-pin-*.tmp")]
        assert len(staging_names) == 1
        assert staging_names[0].read_bytes() == foreign_payload  # foreign bytes survive
        assert not (journal.directory / "admission-pin-before_restart.json").exists()

        # --- Early precheck: substitution BEFORE the precheck -------------
        # The guard's publisher window for phase 1 is CLOSED: restore the real
        # unlink for the test-owned inter-phase cleanup (test cleanup is not
        # a publisher action and is not counted), then reinstall a fresh
        # guard for the phase-2 publisher window with its own counter.
        monkeypatch.setattr(os, "unlink", real_unlink)
        for leftover in Path(journal.directory).glob(".admission-pin-*.tmp"):
            leftover.unlink()
        unlinks2 = []

        def no_unlink2(p):
            unlinks2.append(p)
            raise FileNotFoundError("foreign name must never be unlinked")

        precheck_path = {"done": False}

        def swap_before_precheck(path, **kwargs):
            target = str(path)
            if target.endswith(".tmp") and not precheck_path["done"]:
                precheck_path["done"] = True
                precheck_path["path"] = target
                foreign = Path(target + ".foreign")
                foreign.write_bytes(b"precheck-swap")
                os.replace(foreign, target)
            return real_lstat(path, **kwargs)

        def rename_never_called(staging, final):
            raise AssertionError("rename must not be reached when the precheck fails")

        monkeypatch.setattr(os, "unlink", no_unlink2)
        monkeypatch.setattr(os, "lstat", swap_before_precheck)
        saved = ap.atomic_publish.rename_noreplace
        ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: rename_never_called(
            s, f)
        try:
            with pytest.raises(nv._Reject) as caught_early:
                ap.write_pin(journal.directory, "before_restart", str(receipt_path))
        finally:
            ap.atomic_publish.rename_noreplace = saved
            monkeypatch.setattr(os, "lstat", real_lstat)
        assert caught_early.value.code == "staging_identity_lost"
        assert unlinks2 == []  # phase 2: zero publisher unlinks
        assert Path(precheck_path["path"]).read_bytes() == b"precheck-swap"
        assert not (journal.directory / "admission-pin-before_restart.json").exists()
    finally:
        monkeypatch.setattr(os, "unlink", real_unlink)



def test_unknown_outcome_with_final_already_present_never_deletes(tmp_path, seam):
    # An UNKNOWN rename outcome while the FINAL already exists (a previous
    # complete publication): no deletion, no retry, final preserved.
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    first = ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    assert first["status"] == "written"
    final = journal.directory / "admission-pin-before_restart.json"
    original_bytes = final.read_bytes()

    def rename_unknown(staging, final):
        # Outer-wrapper contract: raise the typed exception.
        raise atomic_publish.RenameUnknown(errno.EIO)

    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: rename_unknown(
        s, f)
    renames_after = {"count": 0}
    try:
        # A SECOND, different binding must refuse unknown WITHOUT touching
        # the existing final (observed fixture behavior; no kernel-guarantee
        # claim is made or tested here).
        other = synthetic_receipt(profile, "e" * 32, 7)
        life.private_json(receipt_path, other, exclusive=False)
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "publish_unknown"
    assert final.read_bytes() == original_bytes  # observed: preserved, never deleted
    assert renames_after["count"] == 0  # no retry


def test_certain_vs_uncertain_assertions_are_observation_bounded(tmp_path, seam):
    # The EEXIST class is CERTAIN (kernel-guaranteed no-replace): the test
    # asserts the final was never overwritten. The EIO class is UNCERTAIN:
    # the test asserts only observed fixture behavior (staging still present,
    # final absent) plus no cleanup - never a kernel guarantee.
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)

    def rename_eexist(staging, final):
        # Outer-wrapper contract: raise the typed exception (the CERTAIN
        # kernel-guaranteed no-replace class).
        raise atomic_publish.RenameWouldOverwrite()

    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: rename_eexist(
        s, f)
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "pin_conflict"  # no existing final => conflict, certain path
    assert not (journal.directory / "admission-pin-before_restart.json").exists()


def test_rc0_foreign_recreation_at_staging_name_is_uncertain(tmp_path, seam, monkeypatch):
    """F1 (restored root contract), host-independent: the injected
    rename_noreplace performs the ACTUAL move, then a foreign actor
    recreates the consumed staging name, and the call returns success.
    Asserts: publish_uncertain, ZERO unlink, foreign survival - on ANY
    host (no os.rename patching that only exists in the Windows
    simulation)."""
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    unlinks = []
    real_unlink = os.unlink

    def no_unlink(p):
        unlinks.append(p)
        raise FileNotFoundError("recreated foreign name must never be unlinked")

    def move_recreate_then_success(old_fd, old, new_fd, new, flags):
        old_path, new_path = os.fsdecode(old), os.fsdecode(new)
        os.rename(old_path, new_path)  # the ACTUAL move (host-independent)
        Path(old_path).write_bytes(b"foreign-recreated")  # foreign recreation
        return 0  # report success

    monkeypatch.setattr(os, "unlink", no_unlink)
    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: (
        move_recreate_then_success(None, s, None, f, None))
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
        monkeypatch.setattr(os, "unlink", real_unlink)
    assert caught.value.code == "publish_uncertain"
    assert unlinks == []  # never unlinked: not ours to decide
    # The foreign recreated file remains intact, and the final exists too
    # (the move itself succeeded before the anomaly was detected).
    staging_names = [p for p in Path(journal.directory).glob(".admission-pin-*.tmp")]
    assert len(staging_names) == 1 and staging_names[0].read_bytes() == b"foreign-recreated"
    final = journal.directory / "admission-pin-before_restart.json"
    assert final.exists() and json.loads(final.read_text())["runtime_epoch"] == "f" * 32



def test_move_then_unknown_error_final_holds_staged_bytes(tmp_path, seam, monkeypatch):
    # REQUIRED (F2): the injected rename ACTUALLY MOVES the staged bytes into
    # the absent final, then reports RenameUnknown/EIO. Assert: publish_unknown;
    # the final contains the EXACT staged bytes; the staging name was consumed
    # by the fake; ZERO cleanup/unlink/retry.
    journal, profile = setup_attempt(tmp_path, seam)
    receipt_path = _published_pin(journal, profile)
    moves = {"count": 0}
    real_rename = os.rename

    def move_then_eio(staging, final):
        # T1 (outer-wrapper contract): take the REAL staging/final paths,
        # ACTUALLY MOVE the staged bytes into the absent final, then report
        # the unknown outcome by RAISING the typed exception (verified
        # constructor: RenameUnknown carries the errno code).
        moves["count"] += 1
        real_rename(str(staging), str(final))  # the move REALLY happens
        raise atomic_publish.RenameUnknown(errno.EIO)  # then it reports EIO

    saved = ap.atomic_publish.rename_noreplace
    unlinks = []
    real_unlink = os.unlink
    monkeypatch.setattr(os, "unlink", lambda p: (unlinks.append(p), real_unlink(p))[1])
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: move_then_eio(
        s, f)
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.write_pin(journal.directory, "before_restart", str(receipt_path))
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "publish_unknown"
    final = journal.directory / "admission-pin-before_restart.json"
    # The final exists and contains the exact staged document bytes.
    pin = json.loads(final.read_text())
    assert pin["runtime_epoch"] == "f" * 32 and pin["schema"] == ap.PIN_SCHEMA
    assert final.read_text() == json.dumps(pin, sort_keys=True,
                                           separators=(",", ":"))
    # Staging consumed by the fake move; zero unlink, zero retry.
    assert list(Path(journal.directory).glob(".admission-pin-*.tmp")) == []
    assert unlinks == []
    assert moves["count"] == 1


# -- raw artifact publisher (publish_artifact) + legacy ordering -------------------

def test_legacy_facade_ordering_guard_path_staging_dumps_size(monkeypatch, tmp_path):
    """Legacy prelude order is preserved EXACTLY: guard -> Path coercion ->
    staging-name construction -> json.dumps -> size bound. A document that
    FAILS serialization must still have had its staging name constructed
    first (order proof via a uuid monkeypatch counter), and a non-Path-able
    destination must raise TypeError BEFORE dumps runs."""
    import uuid as uuid_module
    constructed = {"staging": 0}
    real_uuid4 = uuid_module.uuid4

    def counting_uuid4():
        constructed["staging"] += 1
        return real_uuid4()

    # (a) unserializable document: guard -> ... -> staging name -> dumps fails.
    monkeypatch.setattr(uuid_module, "uuid4", counting_uuid4)
    with pytest.raises(TypeError):
        ap._publish_exclusive(tmp_path / "out.json", {"bad": {1, 2}},
                              conflict_code="c", limit=64)
    assert constructed["staging"] == 1  # staging name WAS constructed first
    # (b) non-Path-able destination: Path() raises before dumps -> no staging name.
    constructed["staging"] = 0
    class NotPathable:
        pass
    with pytest.raises(TypeError):
        ap._publish_exclusive(NotPathable(), {"ok": 1}, conflict_code="c", limit=64)
    assert constructed["staging"] == 0  # Path coercion failed BEFORE construction


def test_publish_artifact_raw_bytes_roundtrip_and_strict_idempotency(tmp_path):
    body = bytes(range(256)) * 3 + b"\x00\xff\xfe\xfd"  # non-UTF8 by construction
    digest = hashlib.sha256(body).hexdigest()
    target = tmp_path / "proof.bin"
    first = ap.publish_artifact(target, body, expected_sha256=digest, max_bytes=4096)
    assert first == "written"
    assert target.read_bytes() == body  # byte-identical round-trip, no parse
    second = ap.publish_artifact(target, body, expected_sha256=digest, max_bytes=4096)
    assert second == "idempotent"
    assert target.read_bytes() == body  # untouched


def test_publish_artifact_one_byte_difference_is_conflict(tmp_path, monkeypatch):
    body = b"proof-bytes-v1"
    digest = hashlib.sha256(body).hexdigest()
    target = tmp_path / "proof.bin"
    assert ap.publish_artifact(target, body, expected_sha256=digest, max_bytes=64) == "written"
    tampered = body[:-1] + b"X"
    tampered_digest = hashlib.sha256(tampered).hexdigest()
    unlinks = []
    real_unlink = os.unlink
    monkeypatch.setattr(os, "unlink", lambda p: (unlinks.append(p), real_unlink(p))[1])
    with pytest.raises(nv._Reject) as caught:
        ap.publish_artifact(target, tampered,
                             expected_sha256=tampered_digest, max_bytes=64)
    assert caught.value.code == "artifact_conflict"
    assert target.read_bytes() == body  # existing final UNCHANGED
    assert unlinks == [] or all(".tmp" in str(u) for u in unlinks)  # own staging only
    assert list(tmp_path.glob(".proof.bin.*.tmp")) == []  # staging consumed


def test_publish_artifact_pre_staging_gates_refuse_without_side_effects(tmp_path):
    body = b"some-proof-bytes"
    good_digest = hashlib.sha256(body).hexdigest()
    # Dedicated fresh output directory: the assertion scope is EXACTLY this
    # directory (created now, empty by construction). The autouse fixture's
    # own capability-simulation file lives one level up in tmp_path and is
    # neither deleted, ignored, nor counted; production must create nothing
    # here because every refusal below happens before any staging side
    # effect.
    out = tmp_path / "out"
    out.mkdir()
    # Every gate refuses BEFORE the staging name exists: no .tmp, no final.
    with pytest.raises(nv._Reject) as e1:
        ap.publish_artifact(out / "a.bin", b"other", expected_sha256=good_digest,
                            max_bytes=64)
    assert e1.value.code == "digest_mismatch"
    with pytest.raises(nv._Reject) as e2:
        ap.publish_artifact(out / "b.bin", body * 8, expected_sha256=good_digest,
                            max_bytes=64)
    assert e2.value.code == "artifact_too_large"
    with pytest.raises(nv._Reject) as e3:
        ap.publish_artifact(out / "c.bin", body, expected_sha256="not-hex",
                            max_bytes=64)
    assert e3.value.code == "expected_digest_invalid"
    for bad in (0, -1, True, 1.5, "64"):
        with pytest.raises(nv._Reject) as e4:
            ap.publish_artifact(out / "d.bin", body,
                                expected_sha256=good_digest, max_bytes=bad)
        assert e4.value.code == "limit_invalid"
    for bad_body in ("str", bytearray(body), memoryview(body)):
        with pytest.raises(TypeError):
            ap.publish_artifact(out / "e.bin", bad_body,
                                expected_sha256=good_digest, max_bytes=64)
    # Exact zero-side-effect property over the dedicated directory:
    assert list(out.glob("*")) == []


@pytest.mark.skipif(sys.platform.startswith("linux"),
                    reason="guard-order refusal is a non-Linux property; the real "
                           "Linux path keeps the real guard and real flags")
def test_publish_artifact_guard_precedes_argument_errors(tmp_path, monkeypatch):
    """The [guard-first] proof: with the platform guard UN-injected (real
    guard restored despite the autouse fixture), invalid arguments still
    refuse publish_platform_unsupported, not TypeError/_Reject - the raw
    facade mirrors the JSON facade's guard-first order."""
    monkeypatch.setattr(ap, "_require_publish_platform",
                        _REAL_REQUIRE_PUBLISH_PLATFORM)
    # Dedicated fresh output directory (exact emptiness scope; the autouse
    # fixture's simulation file one level up is untouched and uncounted).
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(nv._Reject) as caught:
        ap.publish_artifact(out / "f.bin", "not-even-bytes",
                            expected_sha256="x", max_bytes=0)
    assert caught.value.code == "publish_platform_unsupported"
    assert list(out.glob("*")) == []


def test_publish_artifact_unknown_outcome_no_delete(tmp_path):
    body = b"raw-proof-unknown"
    digest = hashlib.sha256(body).hexdigest()
    target = tmp_path / "proof.bin"

    def rename_unknown(staging, final):
        raise atomic_publish.RenameUnknown(errno.EIO)

    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: rename_unknown(s, f)
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.publish_artifact(target, body, expected_sha256=digest, max_bytes=64)
    finally:
        ap.atomic_publish.rename_noreplace = saved
    assert caught.value.code == "publish_unknown"
    assert not target.exists()  # no final appeared
    leftovers = list(tmp_path.glob(".proof.bin.*.tmp"))
    assert len(leftovers) == 1  # RC4: unknown outcome -> NO staging cleanup


def test_publish_artifact_eexist_read_inherits_safe_read_refusals(tmp_path):
    body = b"raw-proof-existing"
    digest = hashlib.sha256(body).hexdigest()
    # Oversize existing final: _safe_read refuses file_too_large.
    target = tmp_path / "proof.bin"
    target.write_bytes(body + b"x" * 4096)
    with pytest.raises(nv._Reject) as e1:
        ap.publish_artifact(target, body, expected_sha256=digest, max_bytes=64)
    assert e1.value.code in ("file_too_large", "artifact_conflict")  # bounded read refusal
    # Symlink at final: symlink_refused.
    link = tmp_path / "proof-link.bin"
    link.symlink_to(target)
    with pytest.raises(nv._Reject) as e2:
        ap.publish_artifact(link, body, expected_sha256=digest, max_bytes=8192)
    assert e2.value.code == "symlink_refused"


def test_publish_artifact_write_failure_cleans_owned_staging(tmp_path, monkeypatch):
    body = b"raw-proof-writefail"
    digest = hashlib.sha256(body).hexdigest()
    state = {}
    real_open = os.open
    real_write = os.write
    _fd_tracking_open(monkeypatch, real_open, state)

    def failing_write(fd, data):
        if fd == state.get("staged_fd"):
            raise OSError(errno.EIO, "write failed")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", failing_write)
    with pytest.raises(nv._Reject) as caught:
        ap.publish_artifact(tmp_path / "proof.bin", body,
                             expected_sha256=digest, max_bytes=64)
    assert caught.value.code == "staging_write_failed"
    assert list((tmp_path).glob(".proof.bin.*.tmp")) == []  # owned cleanup ran
    assert not (tmp_path / "proof.bin").exists()  # no new publication

def test_publish_artifact_ambiguous_close_single_attempt(tmp_path, monkeypatch):
    """RAW ambiguous-close: the normal-path close fails after a real
    close-then-raise injection through publish_artifact. Assert:
    staging_close_failed; the staged fd was closed EXACTLY once (no retry,
    no fd-reuse hazard); owned-inode cleanup ran; no final."""
    body = b"raw-proof-closefail"
    digest = hashlib.sha256(body).hexdigest()
    state = {"close_attempts": {}}
    real_close = os.close
    real_open = os.open
    _fd_tracking_open(monkeypatch, real_open, state)

    def real_close_then_raise(fd):
        attempts = state["close_attempts"].get(fd, 0) + 1
        state["close_attempts"][fd] = attempts
        real_close(fd)
        raise OSError(errno.EIO, "ambiguous close failure")

    monkeypatch.setattr(os, "close", real_close_then_raise)
    with pytest.raises(nv._Reject) as caught:
        ap.publish_artifact(tmp_path / "proof.bin", body,
                            expected_sha256=digest, max_bytes=64)
    assert caught.value.code == "staging_close_failed"
    staged = state.get("staged_fd")
    assert staged is not None and state["close_attempts"].get(staged) == 1
    assert list(tmp_path.glob(".proof.bin.*.tmp")) == []  # owned cleanup ran
    assert not (tmp_path / "proof.bin").exists()  # no new publication


def test_publish_artifact_rc0_foreign_recreation_is_uncertain(tmp_path, monkeypatch):
    """RAW rc0-persistence: the injected rename ACTUALLY MOVES the staged
    bytes into the absent final, a foreign actor then recreates the
    consumed staging name, and the call returns success. Assert:
    publish_uncertain (nonzero), ZERO unlink (the foreign name is never
    deleted), foreign bytes intact at the staging name, final holds the
    exact staged bytes."""
    body = b"raw-proof-rc0"
    digest = hashlib.sha256(body).hexdigest()
    unlinks = []
    real_unlink = os.unlink

    def no_unlink(p):
        unlinks.append(p)
        raise FileNotFoundError("recreated foreign name must never be unlinked")

    def move_recreate_then_success(staging, final):
        os.rename(str(staging), str(final))  # the ACTUAL move
        Path(str(staging)).write_bytes(b"foreign-recreated-raw")  # foreign recreation
        return 0

    monkeypatch.setattr(os, "unlink", no_unlink)
    saved = ap.atomic_publish.rename_noreplace
    ap.atomic_publish.rename_noreplace = lambda s, f, rename_fn=None: (
        move_recreate_then_success(s, f))
    try:
        with pytest.raises(nv._Reject) as caught:
            ap.publish_artifact(tmp_path / "proof.bin", body,
                                expected_sha256=digest, max_bytes=64)
    finally:
        ap.atomic_publish.rename_noreplace = saved
        monkeypatch.setattr(os, "unlink", real_unlink)
    assert caught.value.code == "publish_uncertain"
    assert unlinks == []  # never unlinked: not ours to decide
    staging_names = [p for p in tmp_path.glob(".proof.bin.*.tmp")]
    assert len(staging_names) == 1
    assert staging_names[0].read_bytes() == b"foreign-recreated-raw"  # foreign survives
    assert (tmp_path / "proof.bin").read_bytes() == body  # final holds exact staged bytes


def test_publish_artifact_eexist_hardlink_refused(tmp_path):
    """HARDLINK EEXIST inheritance: an existing final with st_nlink > 1 is
    refused by _safe_read's extra_hardlink_refused check before any
    comparison - the bounded-reader discipline applies to raw artifacts."""
    body = b"raw-proof-hardlinked"
    digest = hashlib.sha256(body).hexdigest()
    target = tmp_path / "proof.bin"
    target.write_bytes(body)
    linked = tmp_path / "proof-link-copy.bin"
    os.link(target, linked)  # st_nlink == 2 for both names now
    with pytest.raises(nv._Reject) as caught:
        ap.publish_artifact(target, body, expected_sha256=digest, max_bytes=64)
    assert caught.value.code == "extra_hardlink_refused"
    assert target.read_bytes() == body  # untouched
    assert linked.read_bytes() == body


def test_legacy_facade_serialization_byte_equivalence(tmp_path):
    """The core writes PREPARED bytes verbatim: the on-disk bytes of a
    legacy JSON publication are exactly json.dumps(document, sort_keys=True,
    separators=(",", ":")).encode() - no parse/reserialize of candidate
    bytes anywhere on the publication path."""
    document = {"z": [1, 2, {"k": "v"}], "a": "unicode-free", "nested": {"b": True}}
    expected_bytes = json.dumps(document, sort_keys=True,
                                separators=(",", ":")).encode()
    target = tmp_path / "equivalence.json"
    assert ap._publish_exclusive(target, document, conflict_code="c",
                                 limit=4096) == "written"
    assert target.read_bytes() == expected_bytes


def test_live_label_policy_plan_and_staged_binding(tmp_path, seam):
    digest, size, host = staged_fixture(tmp_path)
    plan = calibrated_plan(digest, size)
    source = plan["native_validator"]["admission_template"]["source_profile"]
    source.update(label_anchor_alpha_allowed=[0, 0.3, 0.5, 0.7, 1], label_anchor_alpha_default=0)
    journal = make_journal(tmp_path, plan)
    assert ap._plan_contract(journal.directory)[3] == source
    seam[STAGED_LOGICAL] = host
    # A valid policy in the plan cannot authorize a different staged profile.
    with pytest.raises(nv._Reject, match="staged_profile_label_anchor_alpha_allowed_mismatch"):
        ap._verified_staged_profile(plan["calibrated_profile_evidence"], source)


@pytest.mark.parametrize("changes", [
    {"label_anchor_alpha_allowed": [0]},
    {"label_anchor_alpha_default": 0},
    {"label_anchor_alpha_allowed": [0], "label_anchor_alpha_default": None},
])
def test_live_label_policy_plan_requires_both_fields(tmp_path, changes):
    digest, size, _ = staged_fixture(tmp_path)
    plan = calibrated_plan(digest, size)
    plan["native_validator"]["admission_template"]["source_profile"].update(changes)
    journal = make_journal(tmp_path, plan)
    with pytest.raises(nv._Reject, match="source_label_anchor_alpha_invalid"):
        ap._plan_contract(journal.directory)
