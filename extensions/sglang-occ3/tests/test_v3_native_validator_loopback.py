"""Exact loopback-origin admission for the native phase-suite validator.

The single admitted non-HTTPS endpoint origin is the byte-exact local SSH
tunnel http://127.0.0.1:30101: the tunnel fronts the serving container's own
API-key auth layer, so its pinned credential references keep the MANDATORY
API-key reference and may pin an EMPTY header set. Every other HTTP origin
(near misses by host, port, path, query, userinfo, fragment, slash or case)
stays refused, every HTTPS admission still requires non-empty permitted
headers (including HTTPS on the same loopback host and port), the structural
fieldsets stay exact, and the report origin binding equality stays exact in
both directions. Offline synthetic fixtures only: no server, no GPU, no
helper resolution, no network.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import lifecycle as life
from deploy.v3 import native_validator as nv
from deploy.v3.live_validation import NativeHandoffValidator, VALIDATOR_KEY

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "native_phase_suite_v1.json").read_text())
PINNED = FIXTURE["admission_pinned_inputs"]
EXAMPLE = FIXTURE["example"]
ATTEMPT = EXAMPLE["attempt"]["attempt_id"]
EPOCH_BEFORE = EXAMPLE["attempt"]["candidate_started_at_before"]
CHALLENGE_BEFORE = EXAMPLE["challenge_before"]["challenge"]
HTTPS_ORIGIN = PINNED["endpoint"]["origin"]
KEY = "synthetic-private-key-not-real"
ORIGINAL = "1" * 64
CANDIDATE = "c" * 64
IMAGE = "sha256:" + "a" * 64
CANDIDATE_IMAGE = "sha256:" + "b" * 64
PIN_SCHEMA = "occ3.admission.pin.v1"
LOOPBACK = nv.ADMITTED_LOOPBACK_ORIGIN
# A literal pasted key shape (hex, no "!" prefix, digit-led): the api_key
# entry must stay a REFERENCE, never a resolved or literal value.
PASTED_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


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


def pinned_with(*, origin=LOOPBACK, headers=None, api_key=None):
    """Fixture pinned inputs rewritten onto one chosen endpoint origin.

    Defaults are the exact admitted loopback tunnel: an empty header set and
    the fixture's unchanged (still mandatory) API-key reference.
    """
    block = copy.deepcopy(PINNED)
    block["endpoint"]["origin"] = origin
    if headers is not None:
        block["credential_references"]["headers"] = headers
    if api_key is not None:
        block["credential_references"]["api_key"] = api_key
    return block


def loopback_plan(**changes):
    return plan(validator=VALIDATOR_KEY, native_validator=pinned_with(), **changes)


def make_journal(tmp_path, plan_value):
    return life.Journal.create(tmp_path / "attempt", plan_value, original_doc())


def make_validator(journal, *, timeout=5.0):
    state = journal.read()
    return NativeHandoffValidator(state["plan"], journal, state, timeout=timeout,
                                  challenge_factory=lambda: CHALLENGE_BEFORE)


def before_state(journal):
    journal.update(candidate_id=CANDIDATE, candidate_started=True, candidate_epoch=EPOCH_BEFORE,
                   gates={"service_initial": {"ok": True}, "native_initial": {"ok": True}})


def write_pin(journal, phase, observed):
    document = {
        "schema": PIN_SCHEMA, "phase": phase, "attempt_id": ATTEMPT,
        "static_input_sha256": nv.PIN_STATIC_INPUT_SHA256,
        "runtime_epoch": observed["runtime_epoch"],
        "runtime_generation": observed["runtime_generation"],
        "profile_digest": nv._derived_profile_digest(
            PINNED["admission_template"]["source_profile"], observed),
        "written_at": int(time.time()),
    }
    life.private_json(journal.directory / ("admission-pin-" + phase + ".json"),
                      document, exclusive=True)


def write_report(journal, phase, report, *, pin=True):
    document = copy.deepcopy(report)
    document["completed_at"] = int(time.time()) + 30
    life.private_json(journal.directory / ("native-phase-suite-" + phase + ".json"), document)
    if pin:
        write_pin(journal, phase, document["observed"])


# -- exact admission ---------------------------------------------------------
def test_exact_loopback_origin_admitted_with_empty_headers(tmp_path):
    journal = make_journal(tmp_path, loopback_plan())
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is True, result
    assert result["bindings"]["endpoint_origin"] == LOOPBACK
    assert result["http_calls_performed"] == 0 and result["credentials_resolved"] == 0
    assert result["live_success_claimed"] is False


def test_loopback_admission_keeps_the_exact_admission_fieldsets(tmp_path):
    # The exception is value-level only: structural fieldsets stay exact.
    # Each mutation gets its own fresh journal directory: Journal.create
    # requires an existing parent and a not-yet-existing attempt directory.
    for case, (mutate, code) in enumerate([
        (lambda b: b["endpoint"].update(extra_field=True), "endpoint_invalid"),
        (lambda b: b["credential_references"].update(extra_field=True),
         "credential_references_invalid"),
        (lambda b: b["credential_references"].pop("headers"),
         "credential_references_invalid"),
    ]):
        pinned = pinned_with()
        mutate(pinned)
        parent = tmp_path / ("case-" + str(case))
        parent.mkdir()
        journal = make_journal(parent, plan(validator=VALIDATOR_KEY,
                                            native_validator=pinned))
        result = make_validator(journal).pre_mutation_admission()
        assert result["ok"] is False and result["code"] == code, result


def test_loopback_admission_through_services_selection(tmp_path):
    journal = make_journal(tmp_path, loopback_plan())
    services = life.Services(journal.original(), journal.read()["plan"], journal=journal)
    admission = services.pre_mutation_admission(journal.read())
    assert admission["ok"] is True, admission
    assert admission["bindings"]["endpoint_origin"] == LOOPBACK


def test_loopback_full_before_restart_phase_flow(tmp_path):
    journal = make_journal(tmp_path, loopback_plan())
    before_state(journal)
    report = copy.deepcopy(EXAMPLE["report_before"])
    report["binding"]["endpoint_origin"] = LOOPBACK
    write_report(journal, "before_restart", report)
    result = make_validator(journal).before_restart()
    assert result["ok"] is True, result
    assert (journal.directory / "validator-challenge-before_restart.json").exists()


# -- near misses stay refused ------------------------------------------------
NEAR_MISSES = [
    "http://127.0.0.1:30101/",          # trailing slash
    "http://127.0.0.1:30101//",         # double slash
    "http://127.0.0.1:30102",           # the spare tunnel port
    "http://127.0.0.1:30001",           # the DGX-side target port
    "http://127.0.0.1",                 # no port
    "http://localhost:30101",           # localhost name
    "http://[::1]:30101",               # IPv6 loopback
    "http://127.0.0.1.:30101",          # trailing-dot host
    "http://127.0.0.1:30101/v1",        # path
    "http://127.0.0.1:30101?q=1",       # query
    "http://127.0.0.1:30101#f",         # fragment
    "http://user@127.0.0.1:30101",      # userinfo
    "http://user:pw@127.0.0.1:30101",   # userinfo with password
    "HTTP://127.0.0.1:30101",           # case-variant scheme
    " http://127.0.0.1:30101",         # leading whitespace
    "",                                  # empty string
]


@pytest.mark.parametrize("origin", NEAR_MISSES + [None, 123])
def test_every_loopback_near_miss_is_refused(tmp_path, origin):
    pinned = pinned_with(origin=origin)
    journal = make_journal(tmp_path, plan(validator=VALIDATOR_KEY,
                                          native_validator=pinned))
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is False, origin
    assert result["code"] == "endpoint_origin_not_https"


# -- HTTPS behavior unchanged ------------------------------------------------
def test_https_with_nonempty_permitted_headers_still_admitted(tmp_path):
    journal = make_journal(tmp_path, plan(validator=VALIDATOR_KEY,
                                          native_validator=pinned_with(origin=HTTPS_ORIGIN)))
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is True, result
    assert result["bindings"]["endpoint_origin"] == HTTPS_ORIGIN


def test_https_with_empty_headers_still_refused(tmp_path):
    pinned = pinned_with(origin=HTTPS_ORIGIN, headers={})
    journal = make_journal(tmp_path, plan(validator=VALIDATOR_KEY,
                                          native_validator=pinned))
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "credential_headers_invalid"


def test_https_on_the_same_loopback_hostport_still_requires_headers(tmp_path):
    # The empty-header exception is keyed to the exact admitted origin, not
    # to the host or port: HTTPS on 127.0.0.1:30101 is an ordinary HTTPS
    # admission and still needs non-empty permitted headers.
    pinned = pinned_with(origin="https://127.0.0.1:30101", headers={})
    journal = make_journal(tmp_path, plan(validator=VALIDATOR_KEY,
                                          native_validator=pinned))
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == "credential_headers_invalid"


# -- loopback credential references ------------------------------------------
@pytest.mark.parametrize("mutate,code", [
    (lambda b: b["credential_references"].pop("headers"),
     "credential_references_invalid"),
    (lambda b: b["credential_references"].update(headers=["x"]),
     "credential_headers_invalid"),
    (lambda b: b["credential_references"].update(headers="X-Pomerium-Authorization"),
     "credential_headers_invalid"),
    (lambda b: b["credential_references"].update(headers=None),
     "credential_headers_invalid"),
    (lambda b: b["credential_references"].update(
        headers={"authorization": "!helper"}), "header_name_not_allowed"),
    (lambda b: b["credential_references"].update(
        headers={"x-not-allowed": "!helper"}), "header_name_not_allowed"),
    (lambda b: b["credential_references"].update(
        headers={"X-Pomerium-Authorization": "literal value"}),
     "header_reference_invalid"),
    (lambda b: b["credential_references"].update(api_key=PASTED_KEY),
     "api_key_reference_invalid"),
    (lambda b: b["credential_references"].update(api_key=""),
     "api_key_reference_invalid"),
    (lambda b: b["credential_references"].pop("api_key"),
     "credential_references_invalid"),
])
def test_loopback_credential_negatives_stay_refused(tmp_path, mutate, code):
    pinned = pinned_with()
    mutate(pinned)
    journal = make_journal(tmp_path, plan(validator=VALIDATOR_KEY,
                                          native_validator=pinned))
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is False and result["code"] == code, result


def test_loopback_may_still_pin_nonempty_permitted_headers(tmp_path):
    # The branch relaxes ONLY the emptiness rule: a loopback admission that
    # does pin headers still validates names and references as before.
    pinned = pinned_with(headers={"X-Pomerium-Authorization": "!synthetic-pomerium-helper"})
    journal = make_journal(tmp_path, plan(validator=VALIDATOR_KEY,
                                          native_validator=pinned))
    result = make_validator(journal).pre_mutation_admission()
    assert result["ok"] is True, result
    assert result["bindings"]["endpoint_origin"] == LOOPBACK


# -- report binding equality stays exact -------------------------------------
@pytest.mark.parametrize("wrong_origin", [
    HTTPS_ORIGIN,
    "http://127.0.0.1:30101/",
    "http://127.0.0.1:30102",
])
def test_loopback_pinned_report_must_bind_the_exact_loopback_origin(
        tmp_path, wrong_origin):
    journal = make_journal(tmp_path, loopback_plan())
    before_state(journal)
    report = copy.deepcopy(EXAMPLE["report_before"])
    report["binding"]["endpoint_origin"] = wrong_origin
    write_report(journal, "before_restart", report)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False, wrong_origin
    assert result["code"] == "report_origin_binding_mismatch"


def test_https_pinned_report_must_not_bind_the_loopback_origin(tmp_path):
    journal = make_journal(tmp_path, plan(validator=VALIDATOR_KEY,
                                          native_validator=pinned_with(origin=HTTPS_ORIGIN)))
    before_state(journal)
    report = copy.deepcopy(EXAMPLE["report_before"])
    report["binding"]["endpoint_origin"] = LOOPBACK
    write_report(journal, "before_restart", report)
    result = make_validator(journal).before_restart()
    assert result["ok"] is False and result["code"] == "report_origin_binding_mismatch"
