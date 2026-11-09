"""Offline fault injection. No Docker daemon, provider, GPU or real key is used."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import controller as ctl
from deploy.v3 import docker_config as config
from deploy.v3 import lifecycle as life
from deploy.v3.docker_api import DockerAPI, DockerError, container_id

ORIGINAL = "1" * 64
CANDIDATE = "2" * 64
IMAGE = "sha256:" + "a" * 64
CANDIDATE_IMAGE = "sha256:" + "b" * 64
KEY = "synthetic-private-key-not-real"
PROXY = "9" * 64
PROXY_IMAGE = "sha256:" + "c" * 64
NGINX_CONFIG = b"static offline nginx config"


def original_doc():
    return {
        "Id": ORIGINAL, "Image": IMAGE, "Name": "/fixture-service",
        "Config": {"Image": IMAGE, "Hostname": "fixture-host", "Entrypoint": ["python3"],
                   "Cmd": ["-m", "sglang.launch_server", "--api-key", KEY, "--literal", ""],
                   "Env": ["A=1", "B=two", "A=3"], "WorkingDir": "/w", "User": "",
                   "Labels": {}},
        "HostConfig": {"NetworkMode": "bridge", "Runtime": "runc", "Mounts": [],
                       "Binds": None, "RestartPolicy": {"Name": "always", "MaximumRetryCount": 0},
                       "AutoRemove": False, "PortBindings": {}},
        "Mounts": [], "NetworkSettings": {"Networks": {"bridge": {"NetworkID": "net-1", "Aliases": None}}},
        "State": {"Running": True, "Restarting": False, "ExitCode": 0,
                  "Status": "running", "StartedAt": "epoch-original"},
    }


class FakeDocker:
    def __init__(self):
        self.docs = {ORIGINAL: original_doc()}
        proxy = copy.deepcopy(self.docs[ORIGINAL])
        proxy.update(Id=PROXY, Image=PROXY_IMAGE, Name="/neo-k3-ingress")
        proxy["NetworkSettings"] = {"Networks": {"fixture-ingress": {"NetworkID": "net-9"}}}
        self.docs[PROXY] = proxy
        self.nginx_config = NGINX_CONFIG
        self.nginx_fail_if_candidate = False
        self.events, self.faults = [], {}
        self.behavior = "healthy"
        self.starts = 0
        self.socket_path = "/fake/not-a-docker-socket"
        self.deadline = None

    def fail(self, operation, identity, where="after"):
        self.faults[(operation, identity)] = where

    def _before(self, operation, identity):
        self.events.append((operation, identity))
        if self.faults.get((operation, identity)) == "before":
            del self.faults[(operation, identity)]
            raise OSError(KEY)

    def _after(self, operation, identity):
        if self.faults.get((operation, identity)) == "after":
            del self.faults[(operation, identity)]
            raise OSError(KEY)

    def inspect(self, identity):
        self._before("inspect", identity)
        if identity not in self.docs:
            matches = [doc for doc in self.docs.values() if doc.get("Name") == "/" + identity]
            if len(matches) != 1:
                raise DockerError("inspect", 404)
            return copy.deepcopy(matches[0])
        return copy.deepcopy(self.docs[identity])

    def inspect_image(self, identity):
        return {"Id": identity, "Config": {"Env": [], "Volumes": None, "OnBuild": None, "Labels": None}}

    def create(self, body, name):
        identity = CANDIDATE
        self._before("create", identity)
        document = copy.deepcopy(self.docs[ORIGINAL])
        document.update(Id=identity, Image=body["Image"], Name="/" + name)
        document["Config"] = {k: copy.deepcopy(v) for k, v in body.items() if k not in ("HostConfig", "NetworkingConfig")}
        document["HostConfig"] = copy.deepcopy(body["HostConfig"])
        document["State"] = {"Running": False, "Restarting": False, "ExitCode": 0,
                             "Status": "created", "StartedAt": ""}
        document["Mounts"] = copy.deepcopy(self.docs[ORIGINAL]["Mounts"]) if body["HostConfig"].get("Mounts") else []
        self.docs[identity] = document
        return identity

    def rename(self, identity, name):
        self._before("rename", identity)
        for other_id, other in self.docs.items():
            if other_id != identity and other["Name"] == "/" + name:
                raise DockerError("rename", 409)
        self.docs[identity]["Name"] = "/" + name
        self._after("rename", identity)

    def stop(self, identity, seconds=5):
        self._before("stop", identity)
        self.docs[identity]["State"].update(Running=False, Restarting=False, Status="exited")
        self._after("stop", identity)

    def start(self, identity):
        self._before("start", identity)
        self.starts += 1
        self.docs[identity]["State"].update(Running=True, Status="running", StartedAt="epoch-" + str(self.starts))
        if identity == CANDIDATE and self.behavior == "exit":
            self.docs[identity]["State"].update(Running=False, Status="exited", ExitCode=7)
        self._after("start", identity)

    def nginx(self, identity, operation, timeout=10):
        self._before("nginx-" + operation, identity)
        code = 1 if self.nginx_fail_if_candidate and operation == "reload" and not self.docs[ORIGINAL]["State"]["Running"] else 0
        result = {"exit_code": code, "stdout": self.nginx_config if operation == "config" else b"", "stderr": b""}
        self._after("nginx-" + operation, identity)
        return result

    def remove(self, identity):
        self._before("remove", identity)
        if self.docs[identity]["State"]["Running"]:
            raise DockerError("remove", 409)
        del self.docs[identity]


class FakeServices:
    def __init__(self, api):
        self.api, self.calls = api, []
        self.fail_restore = False
        self.live_ok = False
        self.admission_ok = True
        self.identity_proof = None

    def pre_mutation_admission(self, state):
        return {"ok": self.admission_ok, "scope": "offline-fixture-contract"}

    def health(self):
        self.calls.append("health")
        candidate = self.api.docs.get(CANDIDATE)
        unready = candidate and candidate["State"]["Running"] and self.api.behavior == "unready"
        return {"ok": not unready, "http_status": 503 if unready else 200}

    def occ3_identity(self):
        self.calls.append("occ3_identity")
        if self.identity_proof is not None:
            return self.identity_proof
        return {"ok": True, "check": "occ3_identity", "http_status": 200,
                "error_code": None, "route_present": True}

    def chat(self):
        self.calls.append("chat")
        return {"ok": not self.fail_restore}

    def tools(self):
        self.calls.append("tools")
        return {"ok": True}

    def ingress(self):
        self.calls.append("ingress")
        return {"ok": True, "scope": "offline-fake"}

    def validator(self, state):
        return self

    def before_restart(self):
        self.calls.append("live_before")
        return {"ok": self.live_ok, "gate8_pass": self.live_ok}

    def after_restart(self):
        self.calls.append("live_after")
        return {"ok": self.live_ok, "gate8_pass": self.live_ok}


def plan(**changes):
    value = {"attempt_id": "offline-attempt-0001", "original_id": ORIGINAL,
             "original_image_id": IMAGE, "candidate_image_id": CANDIDATE_IMAGE,
             "model": "fixture-model", "base_url": "http://127.0.0.1:18111",
             "ingress_base_url": "http://neo-k3-ingress:8080",
             "ingress": {"container_id": PROXY, "image_id": PROXY_IMAGE,
                         "container_name": "neo-k3-ingress", "network_name": "fixture-ingress",
                         "network_id": "net-9", "config_sha256": hashlib.sha256(NGINX_CONFIG).hexdigest()},
             "startup_timeout_s": 1.5, "deployment_timeout_s": 20,
             "rollback_timeout_s": 5, "proof_timeout_s": 0.05,
             "controller_timeout_s": 0.15, "supervisor_lease_s": 0.1,
             "poll_interval_s": 0.005, "request_timeout_s": 0.1,
             "stop_timeout_s": 1, "probe_timeout_s": 0.1, "validation_timeout_s": 0.2}
    value.update(changes)
    return value


def prepared(tmp_path, api=None, **changes):
    api = api or FakeDocker()
    reference_path = tmp_path / "independent-inspect.json"
    reference_path.write_text(json.dumps(api.docs[ORIGINAL]), encoding="utf-8")
    changes.setdefault("inspect_schema_evidence", {"path": str(reference_path),
                       "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest()})
    journal = life.prepare(api, plan(**changes), tmp_path / "attempt",
                           probe=lambda *_: {"result": "PASS", "cpu_only": True, "scope": "offline-fake"})
    return api, journal, FakeServices(api)


def approval(journal):
    state = journal.read()
    path = journal.directory / "offline-receipt.txt"
    path.write_text("offline fixture receipt only", encoding="utf-8")
    reference = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"action": "one-controlled-swap", "approved": True,
            **{name: state[name] for name in ("attempt_id", "original_id", "original_image_id", "candidate_id", "candidate_image_id")},
            "expires_at": time.time() + 60, "offline": reference, "rehearsal": reference,
            "source_attestation": reference, "ingress_refresh": "unchanged-config-nginx-graceful-reload"}


def authorized(tmp_path, **changes):
    api, journal, services = prepared(tmp_path, **changes)
    life.consume_authorization(journal, approval(journal))
    return api, journal, services


def swapped(api, journal):
    state = journal.read()
    journal.update(mutation_intent=True, phase="MUTATING")
    api.rename(ORIGINAL, state["preserved_name"])
    api.stop(ORIGINAL)
    api.rename(CANDIDATE, state["original_name"])
    api.start(CANDIDATE)
    journal.update(original_stopped=True, candidate_started=True,
                   candidate_epoch=api.docs[CANDIDATE]["State"]["StartedAt"])


def assert_restored(api, journal):
    state = journal.read()
    assert api.docs[ORIGINAL]["Name"] == "/" + state["original_name"]
    assert api.docs[ORIGINAL]["State"]["Running"]
    assert not api.docs[CANDIDATE]["State"]["Running"]
    assert api.docs[ORIGINAL]["Config"] == journal.original()["Config"]
    assert api.docs[ORIGINAL]["HostConfig"] == journal.original()["HostConfig"]
    assert KEY not in (journal.directory / "journal.json").read_text()


def test_plan_default_startup_timeout_matches_shell_lanes():
    """A plan that omits the deployment-health deadline defaults to the
    same 1500 s wait deploy.sh/rollback.sh/occ3_controller apply (PR #258
    rationale: a 13.5-minute healthy overlay leaves only ~6.5 min margin
    at the old 1200 s default)."""
    value = plan()
    value.pop("startup_timeout_s")
    validated = life.validate_plan(value)
    assert validated["startup_timeout_s"] == 1500


def test_prepare_keeps_exact_unstarted_inspected_candidate(tmp_path):
    api, journal, _ = prepared(tmp_path)
    assert journal.read()["candidate_id"] == CANDIDATE
    assert api.docs[CANDIDATE]["State"]["Status"] == "created"
    assert api.docs[CANDIDATE]["Config"]["Cmd"] == api.docs[ORIGINAL]["Config"]["Cmd"]
    assert api.docs[CANDIDATE]["Config"]["Cmd"][-1] == ""
    assert not any(op in ("start", "stop", "rename", "remove") for op, _ in api.events)
    assert sum(op == "create" for op, _ in api.events) == 1
    with pytest.raises(FileExistsError):
        life.prepare(api, journal.read()["plan"], tmp_path / "attempt", probe=lambda *_: {})


def test_missing_mount_and_false_builder_parity_refused_before_original_mutation(tmp_path, monkeypatch):
    api = FakeDocker()
    api.docs[ORIGINAL]["HostConfig"]["Mounts"] = [{"Type": "bind", "Source": "/fixture", "Target": "/model", "ReadOnly": True}]
    api.docs[ORIGINAL]["Mounts"] = [{"Type": "bind", "Source": "/fixture", "Destination": "/model", "RW": False}]
    real_builder = config.build_create_request
    def broken(*args):
        body = real_builder(*args)
        body["HostConfig"]["Mounts"] = []
        return body
    monkeypatch.setattr(config, "build_create_request", broken)
    with pytest.raises(config.ConfigError, match="actual_mount_destinations_differ"):
        prepared(tmp_path, api)
    assert api.docs[ORIGINAL]["State"]["Running"]
    assert not any(op in ("stop", "rename", "start") for op, _ in api.events)


def test_journal_ids_authorization_and_terminal_state_cannot_be_reset(tmp_path):
    api, journal, _ = authorized(tmp_path)
    for change in ({"original_id": "3" * 64}, {"candidate_id": "3" * 64},
                   {"authorization_consumed": False}):
        with pytest.raises(life.Failure):
            journal.update(**change)
    journal.update(mutation_intent=True)
    with pytest.raises(life.Failure):
        journal.update(mutation_intent=False)
    journal.update(phase="RESTORED")
    with pytest.raises(life.Failure):
        journal.update(phase="PREPARED")
    with pytest.raises(life.Failure):
        life.consume_authorization(journal, approval(journal))


@pytest.mark.parametrize("field,value", [("approved", False), ("attempt_id", "other-attempt"),
    ("candidate_id", "3" * 64), ("original_image_id", CANDIDATE_IMAGE), ("expires_at", 1)])
def test_authorization_bound_to_exact_attempt(tmp_path, field, value):
    api, journal, _ = prepared(tmp_path)
    consent = approval(journal)
    consent[field] = value
    with pytest.raises(life.Failure):
        life.consume_authorization(journal, consent)
    assert not journal.read()["authorization_consumed"]
    assert api.docs[ORIGINAL]["State"]["Running"]


def test_recovery_before_mutation_is_verified_noop_not_rename_or_stop(tmp_path):
    api, journal, service = authorized(tmp_path)
    before = list(api.events)
    result = life.recover(api, journal, services=service)
    assert result["result"] == "NOOP" and result["verified"]
    assert not any(op in ("stop", "rename", "start") for op, _ in api.events[len(before):])
    assert service.calls == ["health", "chat", "ingress"]


@pytest.mark.parametrize("operation,identity", [("rename", ORIGINAL), ("stop", ORIGINAL),
                                                ("rename", CANDIDATE), ("start", CANDIDATE)])
def test_oserror_after_mutation_restores_original_by_id(tmp_path, operation, identity):
    api, journal, service = authorized(tmp_path)
    api.fail(operation, identity)
    runner = ctl.Controller(api, journal, services=service, guard=lambda: None)
    assert runner.run() == 1
    assert_restored(api, journal)
    assert journal.read()["recovery"]["verified"]
    assert sum(op == "create" for op, _ in api.events) == 1


@pytest.mark.parametrize("operation,identity", [("stop", CANDIDATE), ("rename", CANDIDATE),
                                                ("rename", ORIGINAL), ("start", ORIGINAL)])
@pytest.mark.parametrize("where", ["before", "after"])
def test_recovery_reconciles_uncertain_docker_calls(tmp_path, operation, identity, where):
    api, journal, service = authorized(tmp_path)
    swapped(api, journal)
    api.fail(operation, identity, where)
    result = life.recover(api, journal, services=service)
    assert result["result"] == "RESTORED" and result["verified"]
    assert_restored(api, journal)


def test_recovery_has_own_budget_after_expired_deployment(tmp_path):
    api, journal, service = authorized(tmp_path, deployment_timeout_s=0.0001)
    swapped(api, journal)
    assert life.recover(api, journal, services=service)["verified"]
    assert_restored(api, journal)


def test_restoration_requires_real_checks_not_start_alone(tmp_path):
    api, journal, service = authorized(tmp_path, rollback_timeout_s=0.5)
    swapped(api, journal)
    service.fail_restore = True
    result = life.recover(api, journal, services=service)
    assert result["result"] == "RESTORE_FAILED" and result["verified"] is False
    assert api.docs[ORIGINAL]["State"]["Running"]


def test_journal_io_failure_does_not_prevent_id_based_recovery(tmp_path, monkeypatch):
    api, journal, service = authorized(tmp_path)
    snapshot = journal.read()
    swapped(api, journal)
    monkeypatch.setattr(journal, "read", lambda: (_ for _ in ()).throw(OSError(KEY)))
    monkeypatch.setattr(journal, "update", lambda **_: (_ for _ in ()).throw(OSError(KEY)))
    result = life.recover(api, journal, services=service, state=snapshot, original=original_doc())
    assert result["verified"] and result["result"] == "RESTORED"
    assert api.docs[ORIGINAL]["Name"] == "/fixture-service"
    assert api.docs[ORIGINAL]["State"]["Running"]


def test_unknown_original_identity_is_never_touched(tmp_path):
    api, journal, service = authorized(tmp_path)
    api.docs[ORIGINAL]["Id"] = "3" * 64
    result = life.recover(api, journal, services=service)
    assert result["result"] == "RESTORE_FAILED"
    assert not any(op in ("start", "stop", "rename") for op, _ in api.events)


def test_candidate_exit_interrupts_inflight_health_without_waiting_for_http(tmp_path):
    api, journal, _ = authorized(tmp_path)
    swapped(api, journal)
    unblock = threading.Event()
    def blocked_health():
        api.docs[CANDIDATE]["State"].update(Running=False, ExitCode=7, Status="exited")
        unblock.wait(1)
        return {"ok": True}
    begun = time.monotonic()
    try:
        with pytest.raises(life.Failure, match="exited"):
            life.exit_aware(api, CANDIDATE, CANDIDATE_IMAGE, blocked_health,
                            begun + 2, interval=0.005)
        assert time.monotonic() - begun < 0.3
    finally:
        unblock.set()


@pytest.mark.parametrize("behavior", ["exit", "unready"])
def test_startup_exit_and_running_unready_both_restore(tmp_path, behavior):
    api, journal, service = authorized(tmp_path)
    api.behavior = behavior
    runner = ctl.Controller(api, journal, services=service, guard=lambda: None)
    assert runner.run() == 1
    assert_restored(api, journal)
    if behavior == "exit":
        assert service.calls == ["health", "chat", "ingress"]


@pytest.mark.parametrize("after_rename", [False, True])
def test_supervisor_disappearance_aborts_controller_and_recovers(tmp_path, after_rename):
    api, journal, service = authorized(tmp_path)
    def guard():
        if not after_rename or api.docs[ORIGINAL]["Name"] != "/fixture-service":
            raise life.Failure("supervisor unavailable")
    runner = ctl.Controller(api, journal, services=service, guard=guard)
    assert runner.run() == 1
    assert_restored(api, journal)
    if not after_rename:
        assert not any(op in ("rename", "stop", "start") for op, _ in api.events)


def test_native_checkpoint_pending_never_counts_as_success(tmp_path):
    api, journal, service = authorized(tmp_path)
    runner = ctl.Controller(api, journal, services=service, guard=lambda: None)
    assert runner.run() == 1
    assert (journal.directory / "checkpoint-initial.json").exists()
    assert "live_before" not in service.calls
    assert "native_initial" not in journal.read()["gates"]
    assert_restored(api, journal)


def native_receipt(journal, checkpoint):
    path = journal.directory / ("evidence-" + checkpoint["phase"] + ".txt")
    path.write_text("offline external proof fixture", encoding="utf-8")
    return {**{k: checkpoint[k] for k in ("attempt_id", "candidate_id", "process_epoch", "phase", "challenge", "model")},
            "kind": "native-optimus-glm-response", "verified": True, "fallback_used": False,
            "native_session_id": "offline-test-only", "verified_at": time.time(),
            "evidence": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}


def test_full_offline_state_machine_requires_two_distinct_epoch_proofs(tmp_path):
    api, journal, service = authorized(tmp_path, proof_timeout_s=2)
    service.live_ok = True
    def guardian_and_external_operator_fixture():
        state = journal.read()
        request = state.get("restart_request")
        if request and not state.get("restart_count"):
            journal.update(restart_count=1)
            life.private_json(journal.directory / "restart-permit.json", {
                "attempt_id": state["attempt_id"], "candidate_id": CANDIDATE,
                "nonce": request["nonce"], "epoch": request["epoch"],
                "expires_monotonic": time.monotonic() + 1})
        for phase in ("initial", "final"):
            checkpoint = journal.directory / ("checkpoint-" + phase + ".json")
            receipt = journal.directory / ("native-proof-" + phase + ".json")
            if checkpoint.exists() and not receipt.exists():
                life.private_json(receipt, native_receipt(journal, life.read_json(checkpoint)))
    runner = ctl.Controller(api, journal, services=service, guard=guardian_and_external_operator_fixture)
    assert runner.run() == 0
    assert journal.read()["phase"] == "GATES_COMPLETE"  # supervisor alone may commit COMPLETE
    assert journal.read()["restart_count"] == 1
    first = life.read_json(journal.directory / "checkpoint-initial.json")
    last = life.read_json(journal.directory / "checkpoint-final.json")
    assert first["challenge"] != last["challenge"]
    assert first["process_epoch"] != last["process_epoch"]
    wrong = native_receipt(journal, first)
    with pytest.raises(life.Failure):
        ctl.proof_matches(wrong, last)
    assert sum(op == "create" for op, _ in api.events) == 1


def test_service_initial_aborts_on_occ3_routes_absent_before_ingress_and_proof(tmp_path):
    api, journal, service = authorized(tmp_path)
    service.identity_proof = {"ok": False, "check": "occ3_identity", "http_status": 404,
                              "error_code": "occ3_routes_absent", "route_present": False}
    assert ctl.Controller(api, journal, services=service, guard=lambda: None).run() == 1
    state = journal.read()
    # Abort at the first gate: no proof wait, no restart, no journaled service gate.
    assert not (journal.directory / "checkpoint-initial.json").exists()
    assert state["restart_count"] == 0 and "service_initial" not in state["gates"]
    assert not [entry for entry in state["history"] if entry["operation"] == "candidate_ingress_reload"]
    assert state["ingress_refresh"]["operation"] == "recovery_ingress_reload"
    # The only nginx reload is the recovery reload of the restored original.
    reloads = [i for i, event in enumerate(api.events) if event == ("nginx-reload", PROXY)]
    assert len(reloads) == 1
    assert api.events.index(("start", ORIGINAL)) < reloads[0]
    assert service.calls == ["health", "occ3_identity", "health", "chat", "ingress"]
    assert state["recovery"]["verified"] and state["recovery"]["result"] == "RESTORED"
    assert_restored(api, journal)


def test_service_gate_failure_detail_names_error_code_and_http_status(tmp_path):
    api, journal, service = authorized(tmp_path)
    swapped(api, journal)
    service.identity_proof = {"ok": False, "check": "occ3_identity", "http_status": 404,
                              "error_code": "occ3_routes_absent"}
    runner = ctl.Controller(api, journal, services=service, guard=lambda: None)
    with pytest.raises(life.Failure,
                       match=r"service gate failed: occ3_identity \(occ3_routes_absent http=404\)"):
        runner.service_gate("service_initial")
    assert "service_initial" not in journal.read()["gates"]
    assert not any(op == "nginx-reload" for op, _ in api.events)


DECLINED_IDENTITY_PROOF = {"ok": False, "check": "occ3_identity", "http_status": 503,
                           "error_code": "occ3_declined", "declined_code": "unsupported_profile",
                           "stage": "install", "experimental_selected": True, "preflight_reached": True,
                           "backend_constructed": True, "worker_readiness_reached": False,
                           "hooks_prepared": True, "route_mount_attempted": False,
                           "rollback_completed": True}


def test_service_gate_failure_detail_carries_the_declined_code(tmp_path):
    api, journal, service = authorized(tmp_path)
    swapped(api, journal)
    service.identity_proof = dict(DECLINED_IDENTITY_PROOF)
    runner = ctl.Controller(api, journal, services=service, guard=lambda: None)
    with pytest.raises(life.Failure,
                       match=r"service gate failed: occ3_identity \(occ3_declined http=503\) "
                             r"declined=unsupported_profile stage=install"):
        runner.service_gate("service_initial")
    state = journal.read()
    assert "service_initial" not in state["gates"]
    assert state["occ3_decline"] == {"gate": "service_initial", "field": "occ3_identity",
                                     "evidence": DECLINED_IDENTITY_PROOF}
    assert not any(op == "nginx-reload" for op, _ in api.events)


def test_service_initial_aborts_on_occ3_declined_fail_fast_before_ingress_and_proof(tmp_path):
    api, journal, service = authorized(tmp_path)
    service.identity_proof = dict(DECLINED_IDENTITY_PROOF)
    assert ctl.Controller(api, journal, services=service, guard=lambda: None).run() == 1
    state = journal.read()
    # Abort at the first gate: no proof wait, no restart, no journaled service gate.
    assert not (journal.directory / "checkpoint-initial.json").exists()
    assert state["restart_count"] == 0 and "service_initial" not in state["gates"]
    assert state["occ3_decline"]["gate"] == "service_initial"
    assert state["occ3_decline"]["evidence"]["declined_code"] == "unsupported_profile"
    assert state["occ3_decline"]["evidence"]["stage"] == "install"
    assert not [entry for entry in state["history"] if entry["operation"] == "candidate_ingress_reload"]
    assert state["ingress_refresh"]["operation"] == "recovery_ingress_reload"
    # The only nginx reload is the recovery reload of the restored original.
    reloads = [i for i, event in enumerate(api.events) if event == ("nginx-reload", PROXY)]
    assert len(reloads) == 1
    assert api.events.index(("start", ORIGINAL)) < reloads[0]
    # Candidate is stopped and renamed -failed BEFORE the original is renamed back and started.
    candidate_stop = max(i for i, (op, ident) in enumerate(api.events)
                         if op == "stop" and ident == CANDIDATE)
    candidate_failed = max(i for i, (op, ident) in enumerate(api.events)
                           if op == "rename" and ident == CANDIDATE)
    original_back = [i for i, (op, ident) in enumerate(api.events)
                     if op == "rename" and ident == ORIGINAL][-1]
    original_start = max(i for i, (op, ident) in enumerate(api.events)
                        if op == "start" and ident == ORIGINAL)
    assert candidate_stop < original_back and candidate_failed < original_back < original_start
    assert service.calls == ["health", "occ3_identity", "health", "chat", "ingress"]
    assert state["recovery"]["verified"] and state["recovery"]["result"] == "RESTORED"
    assert_restored(api, journal)


def test_service_final_also_requires_occ3_identity(tmp_path):
    api, journal, service = authorized(tmp_path, proof_timeout_s=2)
    service.live_ok = True
    failing = {"ok": False, "check": "occ3_identity", "http_status": 503,
               "error_code": "occ3_identity_unavailable"}
    def guardian_and_external_operator_fixture():
        state = journal.read()
        if state.get("restart_count"):
            service.identity_proof = failing
        request = state.get("restart_request")
        if request and not state.get("restart_count"):
            journal.update(restart_count=1)
            life.private_json(journal.directory / "restart-permit.json", {
                "attempt_id": state["attempt_id"], "candidate_id": CANDIDATE,
                "nonce": request["nonce"], "epoch": request["epoch"],
                "expires_monotonic": time.monotonic() + 1})
        for phase in ("initial", "final"):
            checkpoint = journal.directory / ("checkpoint-" + phase + ".json")
            receipt = journal.directory / ("native-proof-" + phase + ".json")
            if checkpoint.exists() and not receipt.exists():
                life.private_json(receipt, native_receipt(journal, life.read_json(checkpoint)))
    runner = ctl.Controller(api, journal, services=service, guard=guardian_and_external_operator_fixture)
    assert runner.run() == 1
    state = journal.read()
    assert state["restart_count"] == 1 and "service_final" not in state["gates"]
    assert not (journal.directory / "checkpoint-final.json").exists()
    assert service.calls.count("occ3_identity") == 2
    assert state["recovery"]["verified"]
    assert_restored(api, journal)


def test_stale_worker_does_not_recover_or_mutate(tmp_path):
    api, journal, service = prepared(tmp_path)
    runner = ctl.Controller(api, journal, services=service, guard=lambda: None)
    with pytest.raises(life.Failure):
        runner.run()
    assert not any(op in ("start", "stop", "rename") for op, _ in api.events)


def test_cli_defaults_safe_and_has_no_rotation_options():
    assert ctl.main([]) == 2
    source = Path(ctl.__file__).read_text()
    assert "new-key" not in source
    with pytest.raises(ValueError):
        container_id("short-name")


def test_docker_api_uses_private_json_body_not_commandline(monkeypatch):
    seen = {}
    class Response:
        status = 201
        def read(self, limit):
            return json.dumps({"ApiVersion": "1.54"} if seen["path"] == "/version" else {"Id": CANDIDATE}).encode()
    class Connection:
        def __init__(self, path, timeout):
            seen["socket"] = path
        def request(self, method, path, body, headers):
            seen.update(method=method, path=path, body=body, headers=headers)
        def getresponse(self):
            return Response()
        def close(self):
            pass
    from deploy.v3 import docker_api
    monkeypatch.setattr(docker_api, "UnixConnection", Connection)
    assert DockerAPI().create({"Cmd": ["--api-key", KEY]}, "fixture-only") == CANDIDATE
    assert KEY.encode() in seen["body"]
    assert KEY not in seen["path"]
    assert seen["method"] == "POST"


def test_missing_candidate_does_not_prevent_original_restoration(tmp_path):
    api, journal, service = authorized(tmp_path)
    swapped(api, journal)
    del api.docs[CANDIDATE]
    result = life.recover(api, journal, services=service)
    assert result["verified"] and result["candidate_missing"]
    assert api.docs[ORIGINAL]["State"]["Running"]
    assert api.docs[ORIGINAL]["Name"] == "/fixture-service"


def test_missing_candidate_with_unknown_name_occupant_never_stops_unknown(tmp_path):
    api, journal, service = authorized(tmp_path, rollback_timeout_s=0.1)
    swapped(api, journal)
    del api.docs[CANDIDATE]
    other = copy.deepcopy(api.docs[ORIGINAL])
    other.update(Id="3" * 64, Name="/fixture-service")
    other["State"]["Running"] = True
    api.docs["3" * 64] = other
    before = len(api.events)
    result = life.recover(api, journal, services=service)
    assert result["result"] == "RESTORE_FAILED"
    assert not any(identity == "3" * 64 for operation, identity in api.events[before:])
    assert api.docs["3" * 64]["State"]["Running"]


def test_later_independent_recovery_can_record_success_without_reopening_attempt(tmp_path):
    # Real fsync admission records need a realistic offline Windows budget.
    api, journal, service = authorized(tmp_path, rollback_timeout_s=1.5)
    swapped(api, journal)
    service.fail_restore = True
    first = life.recover(api, journal, services=service)
    assert first["result"] == "RESTORE_FAILED"
    service.fail_restore = False
    second = life.recover(api, journal, services=service)
    assert second["verified"]
    assert journal.read()["phase"] == "RESTORED"
    evidence = [life.read_json(p) for p in journal.directory.glob("recovery-evidence-*.json")]
    assert {item["result"] for item in evidence} == {"RESTORE_FAILED", "RESTORED"}
    with pytest.raises(life.Failure):
        journal.update(phase="AUTHORIZED")
    with pytest.raises(life.Failure):
        life.consume_authorization(journal, approval(journal))


def test_guardian_rejects_automatic_restart_even_if_running_again(tmp_path):
    from deploy.v3.supervisor import decide
    api, journal, _ = authorized(tmp_path)
    swapped(api, journal)
    api.docs[CANDIDATE]["RestartCount"] = 1
    now = time.monotonic()
    state = journal.read()
    assert decide(journal, state, child_code=None, now=now, begun=now,
                  heartbeat={"attempt_id": state["attempt_id"], "pid": None,
                             "heartbeat": now, "status": "running"}, api=api) == "candidate_unplanned_restart"


def test_api_negotiates_full_current_version_and_refuses_downlevel_pin(monkeypatch):
    api = DockerAPI()
    calls = []
    def exchange(method, path, **kwargs):
        calls.append(path)
        return json.dumps({"ApiVersion": "1.54", "MinAPIVersion": "1.44"} if path == "/version" else {"Id": ORIGINAL}).encode()
    monkeypatch.setattr(api, "_exchange", exchange)
    assert api.inspect(ORIGINAL)["Id"] == ORIGINAL
    assert calls == ["/version", "/v1.54/containers/" + ORIGINAL + "/json"]
    old = DockerAPI(version="v1.45")
    monkeypatch.setattr(old, "_exchange", exchange)
    with pytest.raises(DockerError, match="negotiation"):
        old.inspect(ORIGINAL)


@pytest.mark.parametrize("location", ["top", "Config", "HostConfig", "endpoint", "mount"])
def test_independent_inspect_schema_rejects_symmetric_api_field_loss(location):
    actual = original_doc()
    reference = copy.deepcopy(actual)
    if location == "top":
        reference["NewEngineField"] = None
    elif location in ("Config", "HostConfig"):
        reference[location]["NewEngineField"] = None
    elif location == "endpoint":
        reference["NetworkSettings"]["Networks"]["bridge"]["GwPriority"] = 10
    else:
        reference["Mounts"] = [{"Type": "bind", "Source": "/x", "Destination": "/x", "RW": True}]
    with pytest.raises(life.Failure):
        life.assert_inspect_schema(actual, reference)


def test_ingress_preflight_never_reloads_and_unauthorized_reload_refused(tmp_path):
    api, journal, _ = prepared(tmp_path)
    assert ("nginx-config", PROXY) in api.events
    assert ("nginx-reload", PROXY) not in api.events
    with pytest.raises(life.Failure, match="consumed"):
        life.refresh_ingress(api, journal, journal.read(), time.monotonic() + 1)
    consent = approval(journal)
    del consent["ingress_refresh"]
    with pytest.raises(life.Failure, match="reload approval"):
        life.consume_authorization(journal, consent)


@pytest.mark.parametrize("tamper", ["image", "name", "network", "configuration"])
def test_nginx_reload_refuses_changed_proxy_identity_or_configuration(tmp_path, tamper):
    api, journal, _ = authorized(tmp_path)
    if tamper == "image":
        api.docs[PROXY]["Image"] = IMAGE
    elif tamper == "name":
        api.docs[PROXY]["Name"] = "/other"
    elif tamper == "network":
        api.docs[PROXY]["NetworkSettings"]["Networks"]["fixture-ingress"]["NetworkID"] = "other"
    else:
        api.nginx_config = b"modified config"
    with pytest.raises(life.Failure):
        life.refresh_ingress(api, journal, journal.read(), time.monotonic() + 1)
    assert ("nginx-reload", PROXY) not in api.events


def test_nginx_reload_is_after_readiness_and_failure_restores_and_refreshes_original(tmp_path):
    api, journal, services = authorized(tmp_path)
    api.nginx_fail_if_candidate = True
    runner = ctl.Controller(api, journal, services=services, guard=lambda: None)
    assert runner.run() == 1
    assert_restored(api, journal)
    reload_indices = [i for i, event in enumerate(api.events) if event == ("nginx-reload", PROXY)]
    assert len(reload_indices) == 2
    first, restored = reload_indices
    candidate_start = api.events.index(("start", CANDIDATE))
    original_start = api.events.index(("start", ORIGINAL))
    assert candidate_start < first < original_start < restored
    for index in reload_indices:
        assert ("nginx-test", PROXY) in api.events[:index]
    assert journal.read()["recovery"]["verified"]


def test_noop_recovery_does_not_reload_ingress(tmp_path):
    api, journal, services = authorized(tmp_path)
    assert life.recover(api, journal, services=services)["result"] == "NOOP"
    assert ("nginx-reload", PROXY) not in api.events


def test_probe_is_independently_inspected_before_start(tmp_path, monkeypatch):
    api = FakeDocker()
    journal = life.Journal.create(tmp_path / "probe-attempt", life.validate_plan(plan()), api.docs[ORIGINAL])
    body = config.build_create_request(api.docs[ORIGINAL], CANDIDATE_IMAGE, "fixture-probe")
    monkeypatch.setattr(config, "build_artifact_probe_request", lambda *_: body)
    inspected = []
    def refuse(original, probe, expected):
        inspected.append(probe["Id"])
        raise config.ConfigError("actual_probe_mounts_missing")
    monkeypatch.setattr(config, "assert_artifact_probe_parity", refuse)
    with pytest.raises(config.ConfigError):
        life.artifact_probe(api, journal)
    assert inspected == [CANDIDATE]
    assert not any(operation == "start" for operation, _ in api.events)


def test_original_must_be_observed_stopped_before_candidate_can_start(tmp_path, monkeypatch):
    api, journal, services = authorized(tmp_path)
    real_stop = api.stop
    def ineffective(identity, seconds=5):
        if identity == ORIGINAL:
            api.events.append(("stop-returned-without-effect", identity))
            return
        return real_stop(identity, seconds)
    monkeypatch.setattr(api, "stop", ineffective)
    assert ctl.Controller(api, journal, services=services, guard=lambda: None).run() == 1
    assert ("start", CANDIDATE) not in api.events
    assert_restored(api, journal)


@pytest.mark.parametrize("where", ["before", "after"])
def test_recovery_nginx_transport_uncertainty_is_reconciled(tmp_path, where):
    api, journal, services = authorized(tmp_path)
    swapped(api, journal)
    api.fail("nginx-reload", PROXY, where)
    result = life.recover(api, journal, services=services)
    assert result["verified"]
    assert_restored(api, journal)


def test_nginx_exec_uses_only_fixed_argv_and_demultiplexes_privately(monkeypatch):
    api = DockerAPI()
    calls = []
    def api_json(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"Id": "8" * 64} if path.endswith("/exec") else {"Running": False, "ExitCode": 0}
    data = b"private config content"
    frame = bytes([1, 0, 0, 0]) + len(data).to_bytes(4, "big") + data
    monkeypatch.setattr(api, "json", api_json)
    monkeypatch.setattr(api, "request", lambda *a, **k: frame)
    proof = api.nginx(PROXY, "config")
    assert proof["stdout"] == data and proof["exit_code"] == 0
    assert calls[0][2]["body"]["Cmd"] == ["nginx", "-T"]
    assert calls[0][2]["body"]["Privileged"] is False
    with pytest.raises(ValueError):
        api.nginx(PROXY, "sh -c arbitrary-command")


def test_complete_is_sealed_but_verified_emergency_restoration_can_supersede_it(tmp_path):
    api, journal, services = authorized(tmp_path)
    swapped(api, journal)
    journal.update(phase="COMPLETE")
    with pytest.raises(life.Failure):
        journal.update(phase="AUTHORIZED")
    assert life.recover(api, journal, services=services)["verified"]
    assert journal.read()["phase"] == "RESTORED"


def test_controller_rechecks_local_validator_admission_before_any_mutation(tmp_path):
    api, journal, services = authorized(tmp_path)
    services.admission_ok = False
    assert ctl.Controller(api, journal, services=services, guard=lambda: None).run() == 1
    assert not any(op in ("start", "stop", "rename") for op, _ in api.events)
    assert journal.read()["phase"] == "NOOP"


class HelperDocker(FakeDocker):
    def __init__(self, fault=None):
        super().__init__()
        self.fault = fault
        self.last_body = None

    def inspect_image(self, identity):
        result = super().inspect_image(identity)
        if self.fault == "image-hook":
            result["Config"]["Env"] = ["LD_PRELOAD=/unsafe.so"]
        if self.fault == "image-volume":
            result["Config"]["Volumes"] = {"/unsafe": {}}
        return result

    def create(self, body, name):
        self.last_body = copy.deepcopy(body)
        identity = super().create(body, name)
        self.docs[identity]["NetworkSettings"] = {"Networks": {"fixture-ingress": {"NetworkID": "net-9"}}}
        if self.fault == "actual-hook":
            self.docs[identity]["Config"]["Env"].append("LD_PRELOAD=/unsafe.so")
        if self.fault == "actual-mount":
            self.docs[identity]["Mounts"] = [{"Type": "bind", "Source": "/unsafe", "Destination": "/unsafe", "RW": True}]
        if self.fault == "unknown-owner":
            self.docs[identity]["Config"]["Labels"]["occ3.v3.helper-nonce"] = "other"
        if self.fault in ("lost-response", "unknown-owner"):
            raise OSError(KEY)
        return identity

    def start(self, identity):
        super().start(identity)
        self.docs[identity]["State"].update(Running=False, Status="exited", ExitCode=0)

    def logs(self, identity, limit=65536):
        return json.dumps({"status": 200, "body": {"scope": "offline-helper-only"}})


def helper_setup(tmp_path, fault=None):
    api = HelperDocker(fault)
    journal = life.Journal.create(tmp_path / "helper-attempt", life.validate_plan(plan(request_timeout_s=2)), api.docs[ORIGINAL])
    requester = life.IngressRequester(api, journal.original(), journal.read()["plan"], journal)
    return api, journal, requester


@pytest.mark.parametrize("fault", ["image-hook", "image-volume"])
def test_ingress_helper_rejects_unsafe_image_before_create(tmp_path, fault):
    api, journal, requester = helper_setup(tmp_path, fault)
    result = requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
    assert result.status is None
    assert not any(operation in ("create", "start") for operation, _ in api.events)


@pytest.mark.parametrize("fault", ["actual-hook", "actual-mount"])
def test_ingress_helper_inspects_actual_unstarted_isolation_before_start(tmp_path, fault):
    api, journal, requester = helper_setup(tmp_path, fault)
    result = requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
    assert result.status is None
    assert ("create", CANDIDATE) in api.events
    assert ("start", CANDIDATE) not in api.events
    assert not api.docs[CANDIDATE]["State"]["Running"]


def test_ingress_helper_lost_create_response_reconciles_exact_nonce_config_without_start(tmp_path):
    api, journal, requester = helper_setup(tmp_path, "lost-response")
    result = requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
    assert result.status is None
    assert CANDIDATE not in api.docs
    assert ("start", CANDIDATE) not in api.events
    intents = list((journal.directory / "helpers").glob("*.json"))
    assert len(intents) == 1
    record = life.read_json(intents[0])
    assert record["state"] == "REMOVED" and record["container_id"] == CANDIDATE
    assert len(record["configuration_sha256"]) == 64
    assert len(record["image_configuration_sha256"]) == 64
    assert KEY not in intents[0].read_text()


def test_ingress_helper_lost_response_does_not_adopt_unknown_named_occupant(tmp_path):
    api, journal, requester = helper_setup(tmp_path, "unknown-owner")
    assert requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1).status is None
    assert CANDIDATE in api.docs
    assert not any(op in ("start", "stop", "remove") for op, identity in api.events if identity == CANDIDATE)
    record = life.read_json(next((journal.directory / "helpers").glob("*.json")))
    assert record["container_id"] is None
    assert record["state"] == "CLEANUP_PENDING"


def test_ingress_helper_safe_path_journals_before_create_and_cleans_exact_id(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    real_create = api.create
    def checked_create(body, name):
        paths = list((journal.directory / "helpers").glob("*.json"))
        assert len(paths) == 1
        record = life.read_json(paths[0])
        assert record["state"] == "CREATE_INTENT" and record["name"] == name
        return real_create(body, name)
    api.create = checked_create
    assert requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1).status == 200
    assert ("start", CANDIDATE) in api.events and ("remove", CANDIDATE) in api.events
    assert CANDIDATE not in api.docs
    assert KEY not in " ".join(api.last_body["Cmd"])
    assert "threading.Timer" in api.last_body["Cmd"][1]
    assert "ProxyHandler({})" in api.last_body["Cmd"][1]
    assert KEY in next(value for value in api.last_body["Env"] if value.startswith("OCC3_PRIVATE_REQUEST="))


def test_recovery_reconciles_only_recorded_running_helper_after_controller_loss(tmp_path, monkeypatch):
    api, journal, requester = helper_setup(tmp_path)
    with monkeypatch.context() as context:
        # Inert crash-window simulation: skip requester's finally, then use the actual recovery reconciler.
        context.setattr(life, "reconcile_helper", lambda *args: {"removed": False})
        response = requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
        assert response.status is None and response.error_code == "ingress_helper_cleanup_pending"
    api.docs[CANDIDATE]["State"].update(Running=True, Status="running")
    unknown = copy.deepcopy(api.docs[CANDIDATE])
    unknown.update(Id="4" * 64, Name="/unrelated-helper")
    unknown["Config"]["Labels"] = {"owner": "unrelated"}
    api.docs["4" * 64] = unknown
    results = life.reconcile_helpers(api, journal)
    assert results[0]["removed"] is True
    assert ("stop", CANDIDATE) in api.events and CANDIDATE not in api.docs
    assert api.docs["4" * 64]["State"]["Running"]
    assert not any(op in ("stop", "remove") and identity == "4" * 64 for op, identity in api.events)


def test_disposable_network_name_lookup_is_exact_readonly_and_returns_private_raw_doc(monkeypatch):
    api = DockerAPI()
    calls = []
    name = "occ3-v3-rehearsal-" + "a" * 16 + "-net"
    document = {"Id": "7" * 64, "Name": name, "Driver": "bridge", "Labels": {"owner": "fixture"}}
    def query(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return document
    monkeypatch.setattr(api, "json", query)
    assert api.inspect_network_name(name) is document
    assert calls == [("GET", "/networks/" + name, {})]


@pytest.mark.parametrize("name", ["bridge", "neo-k3-ingress-net", "a" * 64,
    "occ3-v3-rehearsal-" + "a" * 16 + "-net-extra", "../network",
    "occ3-v3-rehearsal-" + "a" * 16 + "-net?verbose=true", None])
def test_disposable_network_name_lookup_rejects_nonexact_names_before_api(monkeypatch, name):
    api = DockerAPI()
    monkeypatch.setattr(api, "json", lambda *args, **kwargs: pytest.fail("invalid name reached Engine API"))
    with pytest.raises(ValueError):
        api.inspect_network_name(name)


@pytest.mark.parametrize("status", [404, 409])
def test_disposable_network_name_lookup_never_lists_or_falls_back(monkeypatch, status):
    api = DockerAPI()
    calls = []
    def missing(method, path, **kwargs):
        calls.append(path)
        raise DockerError("network inspect", status)
    monkeypatch.setattr(api, "json", missing)
    name = "occ3-v3-rehearsal-" + "b" * 16 + "-net"
    with pytest.raises(DockerError) as caught:
        api.inspect_network_name(name)
    assert caught.value.status == status
    assert calls == ["/networks/" + name]


def test_disposable_network_name_lookup_refuses_mismatched_server_name(monkeypatch):
    api = DockerAPI()
    monkeypatch.setattr(api, "json", lambda *args, **kwargs: {"Id": "7" * 64, "Name": "unknown"})
    with pytest.raises(DockerError):
        api.inspect_network_name("occ3-v3-rehearsal-" + "c" * 16 + "-net")


def test_successful_helper_response_with_remove_failure_is_not_http_success(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    def failed_remove(identity):
        raise OSError(KEY)
    api.remove = failed_remove
    response = requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
    assert response.status is None and response.error_code == "ingress_helper_cleanup_pending"
    path = next((journal.directory / "helpers").glob("*.json"))
    record = life.read_json(path)
    assert record["container_id"] == CANDIDATE and record["state"] == "CLEANUP_PENDING"
    assert record["cleanup_error"] == "OSError"
    assert KEY not in path.read_text()
    assert CANDIDATE in api.docs


def test_ingress_gate_separates_service_semantics_from_failed_helper_cleanup(tmp_path, monkeypatch):
    from deploy.v3 import service_checks
    api, journal, requester = helper_setup(tmp_path)
    def failed_remove(identity):
        raise OSError(KEY)
    api.remove = failed_remove
    def check_private_response(base, key, model, timeout, requester, route):
        response = requester("GET", base + "/v1/models", key, None, timeout)
        return {"ok": response.status == 200, "scope": "offline_semantic_stub_only"}
    monkeypatch.setattr(service_checks, "check_ingress", check_private_response)
    services = life.Services(journal.original(), journal.read()["plan"], api=api, journal=journal)
    proof = services.ingress()
    assert proof["service_ok"] is True
    assert proof["cleanup_complete"] is False and proof["ok"] is False
    assert proof["error_code"] == "ingress_helper_cleanup_pending"
    assert proof["helper_cleanup"] == [{"removed": False, "error": "OSError"}]
    assert KEY not in json.dumps(proof)


def test_recovery_keeps_service_restored_separate_and_later_closes_exact_cleanup(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    real_remove = api.remove
    def failed_remove(identity):
        raise OSError(KEY)
    api.remove = failed_remove
    assert requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1).status is None
    services = FakeServices(api)
    failed = life.recover(api, journal, services=services)
    assert failed["service_restored"] is True and failed["service_result"] == "NOOP"
    assert failed["verified"] is False and failed["result"] == "RESTORE_FAILED"
    assert failed["cleanup_complete"] is False and failed["helpers_settled"] is False
    assert failed["error"] == "helper_cleanup_pending"
    assert "chat" in services.calls and "ingress" in services.calls
    assert api.docs[ORIGINAL]["State"]["Running"]
    assert CANDIDATE in api.docs
    api.remove = real_remove
    api.events.clear()
    settled = life.recover(api, journal, services=services)
    assert settled["result"] == "NOOP" and settled["verified"] is True
    assert settled["service_restored"] and settled["cleanup_complete"] and settled["helpers_settled"]
    assert CANDIDATE not in api.docs
    assert ("remove", CANDIDATE) in api.events
    assert not any(op in ("create", "start", "stop", "rename") for op, _ in api.events)
    assert journal.read()["phase"] == "NOOP"
    evidence = [life.read_json(path) for path in journal.directory.glob("recovery-evidence-*.json")]
    assert any(item["verified"] is False for item in evidence) and any(item["verified"] is True for item in evidence)


def test_recovery_final_cleanup_includes_helpers_created_by_recovery_ingress(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    services = FakeServices(api)
    def failed_remove(identity):
        raise OSError(KEY)
    api.remove = failed_remove
    def ingress():
        response, cleanup = requester.service_response("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
        return {"ok": False, "service_ok": response.status == 200,
                "cleanup_complete": cleanup["removed"], "helper_cleanup": [cleanup]}
    services.ingress = ingress
    result = life.recover(api, journal, services=services)
    assert result["service_restored"] is True and result["verified"] is False
    assert result["cleanup_complete"] is False
    assert result["helper_cleanup"] == [{"removed": False, "error": "OSError"}]
    assert CANDIDATE in api.docs


def test_recovery_does_not_repeatedly_create_helpers_after_unsettled_failed_service_proof(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    services = FakeServices(api)
    def failed_remove(identity):
        raise OSError(KEY)
    api.remove = failed_remove
    calls = []
    def ingress():
        calls.append(1)
        requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
        return {"ok": False, "service_ok": False, "cleanup_complete": False}
    services.ingress = ingress
    result = life.recover(api, journal, services=services)
    assert result["verified"] is False and result["service_restored"] is False
    assert len(calls) == 1
    assert api.docs[ORIGINAL]["State"]["Running"]


def test_helper_known_immutable_id_404_closes_but_unknown_create_name_404_does_not(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    real_remove = api.remove
    def remove_then_lose_response(identity):
        real_remove(identity)
        raise OSError(KEY)
    api.remove = remove_then_lose_response
    assert requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1).status is None
    path = next((journal.directory / "helpers").glob("*.json"))
    settled = life.reconcile_helper(api, path)
    assert settled["removed"] and settled["already_absent"]
    unknown = life.read_json(path)
    unknown.update(container_id=None, state="CREATE_INTENT")
    life.private_json(path, unknown)
    pending = life.reconcile_helper(api, path)
    assert pending == {"removed": False, "unresolved_create": True}
    assert life.read_json(path)["state"] == "CREATE_INTENT"


def test_helper_journal_bound_prevents_any_further_create(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    directory = journal.directory / "helpers"
    directory.mkdir()
    for number in range(128):
        (directory / (str(number) + ".json")).write_text("{}")
    response = requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
    assert response.status is None
    assert not any(operation == "create" for operation, _ in api.events)


def test_abandoned_verification_before_helper_intent_cannot_create_after_verified_recovery(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_image = api.inspect_image
    responses = []
    def blocked_image(identity):
        records = [life.read_json(path) for path in (journal.directory / "helper-operations").glob("*.json")]
        assert len(records) == 1 and records[0]["state"] == "PREFLIGHT"
        assert records[0]["owner"]["pid"] == os.getpid() and records[0]["generation"]
        entered.set()
        assert release.wait(10), "offline barrier release timeout"
        return real_image(identity)
    api.inspect_image = blocked_image
    def operation():
        try:
            responses.append(requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1))
            return {"ok": False}
        finally:
            finished.set()
    def guard():
        if entered.is_set():
            raise life.Failure("synthetic abandoned check")
    try:
        with pytest.raises(life.Failure):
            life.exit_aware(api, ORIGINAL, IMAGE, operation, time.monotonic() + 3, guard=guard, interval=0.005)
        assert entered.is_set() and not finished.is_set()
        restored = life.recover(api, journal, services=FakeServices(api))
        assert restored["verified"] and restored["service_restored"]
        assert restored["helper_operations_settled"] and restored["helper_cleanup"] == []
        assert restored["helper_operations"][0]["revoked"]
        mark = len(api.events)
        release.set()
        assert finished.wait(5)
        assert not any(op in ("create", "start") for op, _ in api.events[mark:])
        assert CANDIDATE not in api.docs
        assert life.read_json(journal.directory / "helper-admission.json")["mode"] == "closed"
    finally:
        release.set()
        assert finished.wait(5)


@pytest.mark.parametrize("lost_response", [False, True])
def test_submitted_create_is_pending_across_recovery_until_exact_effect_settles(tmp_path, lost_response):
    api, journal, requester = helper_setup(tmp_path, "lost-response" if lost_response else None)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_create = api.create
    def delayed_create(body, name):
        op = life.read_json(next((journal.directory / "helper-operations").glob("*.json")))
        assert op["state"] == "CREATE_IN_FLIGHT" and op["mutation"] == "create"
        intent = life.read_json(journal.directory / "helpers" / (op["helper_nonce"] + ".json"))
        assert intent["name"] == name and intent["container_id"] is None
        entered.set()
        assert release.wait(10), "offline barrier release timeout"
        return real_create(body, name)
    api.create = delayed_create
    def operation():
        try:
            requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
            return {"ok": False}
        finally:
            finished.set()
    def guard():
        if entered.is_set():
            raise life.Failure("synthetic abandoned submitted create")
    try:
        with pytest.raises(life.Failure):
            life.exit_aware(api, ORIGINAL, IMAGE, operation, time.monotonic() + 3, guard=guard, interval=0.005)
        unresolved = life.recover(api, journal, services=FakeServices(api))
        assert unresolved["service_restored"] and not unresolved["verified"]
        assert not unresolved["helper_operations_settled"] and not unresolved["cleanup_complete"]
        assert any(item.get("unresolved_create") for item in unresolved["helper_cleanup"])
        assert api.docs[ORIGINAL]["State"]["Running"]
        assert not finished.is_set()
        release.set()
        assert finished.wait(5)
        assert CANDIDATE not in api.docs
        assert not any(op == "start" for op, _ in api.events)
        assert len([op for op, _ in api.events if op == "create"]) == 1
        settled = life.recover(api, journal, services=FakeServices(api))
        assert settled["verified"] and settled["helper_operations_settled"]
        assert settled["result"] == "NOOP"
    finally:
        release.set()
        assert finished.wait(5)


def test_unknown_create_effect_never_settles_from_finished_thread_or_owner_death(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    def unknown_create(body, name):
        raise OSError("synthetic unknown Engine response")
    api.create = unknown_create
    response = requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
    assert response.status is None
    for _ in range(2):
        result = life.recover(api, journal, services=FakeServices(api))
        assert result["service_restored"] and not result["verified"]
        assert not result["helper_operations_settled"]
    record = life.read_json(next((journal.directory / "helper-operations").glob("*.json")))
    assert record["finished"] and record["mutation"] == "create"
    assert not any(op in ("start", "stop", "remove") for op, _ in api.events)


def test_recovery_owns_distinct_helper_admission_and_closes_it_before_receipt(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    assert requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1).status == 200
    deployment_gate = life.read_json(journal.directory / "helper-admission.json")
    services = FakeServices(api)
    def ingress():
        response = requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
        return {"ok": response.status == 200}
    services.ingress = ingress
    result = life.recover(api, journal, services=services)
    assert result["verified"] and result["helper_operations_settled"]
    records = [life.read_json(path) for path in (journal.directory / "helper-operations").glob("*.json")]
    assert {record["mode"] for record in records} == {"deployment", "recovery"}
    assert len({record["generation"] for record in records}) == 2
    final = life.read_json(journal.directory / "helper-admission.json")
    assert final["mode"] == "closed" and final["generation"] != deployment_gate["generation"]
    before = list(api.events)
    assert requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1).status is None
    assert api.events == before


def test_lost_candidate_create_reply_records_cleanup_only_identity_without_retry(tmp_path):
    api = FakeDocker()
    real_create = api.create
    def lost_create(body, name):
        intent_path = tmp_path / "attempt" / "candidate-create-intent.json"
        intent = life.read_json(intent_path)
        assert intent["phase"] == "CREATE_INTENT" and intent["name"] == name
        assert len(intent["resource_nonce"]) == 32 and len(intent["request_sha256"]) == 64
        real_create(body, name)
        raise OSError(KEY)
    api.create = lost_create
    with pytest.raises(OSError):
        prepared(tmp_path, api=api)
    journal = life.Journal(tmp_path / "attempt")
    record = life.read_json(journal.directory / "candidate-create-intent.json")
    assert record["container_id"] == CANDIDATE and record["cleanup_admissible"]
    assert record["reconciliation"] == "verified_cleanup_only"
    assert journal.read()["phase"] == "PREFLIGHT_FAILED" and journal.read()["candidate_id"] is None
    assert len([op for op, _ in api.events if op == "create"]) == 1
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)


def test_revoked_inflight_start_cannot_resurrect_positive_removed_immutable_helper(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_start = api.start
    def delayed_start(identity):
        record = life.read_json(next((journal.directory / "helper-operations").glob("*.json")))
        assert record["state"] == "START_IN_FLIGHT" and record["container_id"] == identity
        entered.set()
        assert release.wait(10)
        return real_start(identity)
    api.start = delayed_start
    def operation():
        try:
            requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1)
            return {"ok": False}
        finally:
            finished.set()
    def guard():
        if entered.is_set():
            raise life.Failure("synthetic abandoned start")
    try:
        with pytest.raises(life.Failure):
            life.exit_aware(api, ORIGINAL, IMAGE, operation, time.monotonic() + 3, guard=guard, interval=0.005)
        result = life.recover(api, journal, services=FakeServices(api))
        # Unlike an unknown create, removed immutable ID is positive settlement:
        # the already-submitted start cannot recreate that ID.
        assert result["verified"] and result["helper_operations_settled"]
        assert CANDIDATE not in api.docs
        assert ("remove", CANDIDATE) in api.events
        release.set()
        assert finished.wait(5)
        assert CANDIDATE not in api.docs and api.docs[ORIGINAL]["State"]["Running"]
        assert len([op for op, _ in api.events if op == "create"]) == 1
    finally:
        release.set()
        assert finished.wait(5)


def test_corrupt_operation_link_cannot_count_as_helper_settlement(tmp_path):
    api, journal, requester = helper_setup(tmp_path)
    assert requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1).status == 200
    path = next((journal.directory / "helper-operations").glob("*.json"))
    record = life.read_json(path)
    record["attempt_id"] = "different-attempt"
    life.private_json(path, record)
    result = life.recover(api, journal, services=FakeServices(api))
    assert result["service_restored"] and not result["verified"]
    assert not result["helper_operations_settled"]


@pytest.mark.parametrize("fault", ["missing", "foreign", "config"])
def test_uncertain_candidate_create_reconciliation_refuses_unknown_or_altered(tmp_path, fault):
    api = FakeDocker()
    real_create = api.create
    def lost_create(body, name):
        real_create(body, name)
        if fault == "missing":
            del api.docs[CANDIDATE]
        elif fault == "foreign":
            api.docs[CANDIDATE]["Name"] = "/unrelated"
        else:
            api.docs[CANDIDATE]["Config"]["Env"].append("LD_PRELOAD=/unsafe")
        raise OSError(KEY)
    api.create = lost_create
    with pytest.raises(OSError):
        prepared(tmp_path, api=api)
    record = life.read_json(tmp_path / "attempt" / "candidate-create-intent.json")
    assert record["reconciliation"] == "unresolved" and not record["cleanup_admissible"]
    assert record["container_id"] is None
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)


def fixed_candidate_name():
    return "occ3-v3-" + plan()["attempt_id"] + "-candidate"


def test_candidate_create_uses_fresh_nonce_bound_temporary_name(tmp_path):
    api, journal, _ = prepared(tmp_path)
    state = journal.read()
    record = life.read_json(journal.directory / "candidate-create-intent.json")
    assert re.fullmatch(r"[a-f0-9]{32}", record["resource_nonce"])
    assert record["final_name"] == state["candidate_name"] == fixed_candidate_name()
    assert record["name"] == state["candidate_name"] + "-n" + record["resource_nonce"]
    assert api.docs[CANDIDATE]["Name"] == "/" + record["name"]
    assert record["phase"] == "VERIFIED_STOPPED_OWN"
    assert record["cleanup_admissible"] is False and record["cleanup_complete"] is False


def test_preexisting_same_name_config_candidate_occupant_409_never_adopts(tmp_path):
    api = FakeDocker()
    name = fixed_candidate_name()
    body = config.build_create_request(api.docs[ORIGINAL], CANDIDATE_IMAGE, name)
    foreign_id = api.create(body, name)
    api.events.clear()
    seen = {}
    def rejected_create(create_body, create_name):
        seen["name"] = create_name
        raise DockerError("create conflict", 409)
    api.create = rejected_create
    with pytest.raises(DockerError):
        prepared(tmp_path, api=api)
    journal = life.Journal(tmp_path / "attempt")
    record = life.read_json(journal.directory / "candidate-create-intent.json")
    assert seen["name"] != name and seen["name"].startswith(name + "-n")
    assert record["container_id"] is None and record["cleanup_admissible"] is False
    assert record["reconciliation"] == "unresolved" and record["create_outcome"] == "rejected"
    assert record["create_error_status"] == 409 and record["owner"] is None
    assert journal.read()["phase"] == "PREFLIGHT_FAILED" and journal.read()["candidate_id"] is None
    assert foreign_id in api.docs and not api.docs[foreign_id]["State"]["Running"]
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)
    assert len([op for op, _ in api.events if op == "create"]) == 0


def test_uncertain_candidate_create_never_adopts_fixed_name_occupant(tmp_path):
    # A preexisting container with the exact fixed candidate name and full
    # configuration exists while this attempt's create never took effect.
    api = FakeDocker()
    name = fixed_candidate_name()
    body = config.build_create_request(api.docs[ORIGINAL], CANDIDATE_IMAGE, name)
    foreign_id = api.create(body, name)
    api.events.clear()
    def lost_create(create_body, create_name):
        assert create_name.startswith(name + "-n")
        raise OSError(KEY)
    api.create = lost_create
    with pytest.raises(OSError):
        prepared(tmp_path, api=api)
    record = life.read_json(tmp_path / "attempt" / "candidate-create-intent.json")
    assert record["container_id"] is None and record["cleanup_admissible"] is False
    assert record["reconciliation"] == "unresolved" and record["create_outcome"] == "uncertain"
    assert record["create_error_status"] is None
    assert foreign_id in api.docs
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)


def test_uncertain_candidate_create_binds_only_nonce_named_own_container(tmp_path):
    api = FakeDocker()
    real_create = api.create
    seen = {}
    def lost_create(body, name):
        seen["name"] = name
        identity = real_create(body, name)
        raise OSError(KEY)
    api.create = lost_create
    with pytest.raises(OSError):
        prepared(tmp_path, api=api)
    record = life.read_json(tmp_path / "attempt" / "candidate-create-intent.json")
    assert record["name"] == seen["name"]
    assert record["name"] == record["final_name"] + "-n" + record["resource_nonce"]
    assert record["container_id"] == CANDIDATE and record["cleanup_admissible"] is True
    assert record["reconciliation"] == "verified_cleanup_only"
    assert len(record["actual_configuration_sha256"]) == 64
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)


def test_recovery_accepts_candidate_at_nonce_create_name_after_prepare_failure(tmp_path):
    api = FakeDocker()
    api.fail("inspect", CANDIDATE, "before")
    with pytest.raises(OSError):
        prepared(tmp_path, api=api)
    journal = life.Journal(tmp_path / "attempt")
    state = journal.read()
    assert state["candidate_id"] == CANDIDATE and state["phase"] == "PREFLIGHT_FAILED"
    record = life.read_json(journal.directory / "candidate-create-intent.json")
    assert api.docs[CANDIDATE]["Name"] == "/" + record["name"]
    assert record["container_id"] == CANDIDATE and record["cleanup_admissible"] is True
    result = life.recover(api, journal, services=FakeServices(api))
    assert result["result"] == "NOOP" and result["verified"]
    assert api.docs[CANDIDATE]["State"]["Status"] == "created"
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)


def probe_original_doc():
    document = original_doc()
    document["Config"]["Cmd"] = ["-m", "sglang.launch_server", "--model-path", "/data/model",
                                 "--api-key", KEY, "--load-format", "safetensors"]
    document["HostConfig"]["Mounts"] = [{"Type": "bind", "Source": "/data", "Target": "/data"}]
    document["Mounts"] = [{"Type": "bind", "Source": "/data", "Destination": "/data", "RW": True}]
    return document


class ArtifactProbeDocker(FakeDocker):
    """Engine-faithful NetworkSettings and one-shot exit for the actual probe path."""

    def __init__(self):
        super().__init__()
        self.docs[ORIGINAL] = probe_original_doc()

    def create(self, body, name):
        identity = super().create(body, name)
        if body.get("NetworkDisabled") is True:
            self.docs[identity]["NetworkSettings"] = {"Networks": {"none": {}}}
        return identity

    def start(self, identity):
        super().start(identity)
        self.docs[identity]["State"].update(Running=False, Status="exited", ExitCode=0)

    def logs(self, identity, limit=65536):
        return json.dumps({"schema": "occ3-artifact-preflight/v1", "result": "PASS", "cpu_only": True,
                           "loader_profile": "sglang-0.5.21-glm-dsa-native-fp8",
                           "model_config_json_valid": True, "tokenizer_config_json_valid": True,
                           "weight_shards": 2, "weight_data_bytes_read": 4096,
                           "weight_header_bytes_read": 64, "tokenizer_artifacts": 3,
                           "total_bytes_read": 8192})


def probe_plan(tmp_path, api):
    reference = tmp_path / "probe-inspect.json"
    reference.write_text(json.dumps(api.docs[ORIGINAL]), encoding="utf-8")
    return plan(inspect_schema_evidence={"path": str(reference),
               "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()})


def probe_prepared(tmp_path, api):
    return life.prepare(api, probe_plan(tmp_path, api), tmp_path / "attempt")


def test_artifact_probe_durable_intent_precedes_create_and_id_precedes_start(tmp_path):
    api = ArtifactProbeDocker()
    journal_dir = tmp_path / "attempt"
    real_create, real_start = api.create, api.start
    def checked_create(body, name):
        if body.get("NetworkDisabled") is True:
            ledger = life.read_json(journal_dir / "artifact-probe-intent.json")
            assert ledger["phase"] == "CREATE_INTENT" and ledger["container_id"] is None
            assert ledger["name"] == name
            assert name == ledger["final_name"] + "-n" + ledger["resource_nonce"]
            assert len(ledger["resource_nonce"]) == 32 and len(ledger["request_sha256"]) == 64
            assert ledger["final_name"] == fixed_candidate_name() + "-artifact"
            assert ledger["cleanup_admissible"] is False
        return real_create(body, name)
    def checked_start(identity):
        ledger = life.read_json(journal_dir / "artifact-probe-intent.json")
        assert ledger["container_id"] == identity and ledger["phase"] == "VERIFIED_STOPPED_OWN"
        assert ledger["cleanup_admissible"] is True
        return real_start(identity)
    api.create, api.start = checked_create, checked_start
    journal = probe_prepared(tmp_path, api)
    ledger = life.read_json(journal_dir / "artifact-probe-intent.json")
    assert ledger["phase"] == "REMOVED" and ledger["cleanup_complete"] is True
    assert ledger["container_id"] == CANDIDATE
    assert (journal.directory / "artifact-proof.json").exists()
    state = journal.read()
    assert state["probe_id"] == CANDIDATE and state["phase"] == "PREPARED"
    assert sum(op == "remove" for op, _ in api.events) == 1
    assert ("start", CANDIDATE) in api.events


def test_artifact_probe_path_accepts_valueless_enable_http2(tmp_path):
    # Production prepare's normal probe path, offline: the original launch
    # carries the transport-only valueless flag end to end.
    api = ArtifactProbeDocker()
    api.docs[ORIGINAL]["Config"]["Cmd"].append("--enable-http2")
    journal = probe_prepared(tmp_path, api)
    state = journal.read()
    assert state["phase"] == "PREPARED" and state["probe_id"] == CANDIDATE
    assert state["probe_parity"]["result"] == "PASS"
    assert state["probe_parity"]["isolated_stdlib_command_exact"] is True
    assert (journal.directory / "artifact-proof.json").exists()
    assert ("start", CANDIDATE) in api.events and ("remove", CANDIDATE) in api.events


def test_lost_artifact_probe_create_reply_records_cleanup_identity_without_removal(tmp_path):
    api = ArtifactProbeDocker()
    real_create = api.create
    created = []
    def lost_create(body, name):
        ledger = life.read_json(tmp_path / "attempt" / "artifact-probe-intent.json")
        assert ledger["phase"] == "CREATE_INTENT" and ledger["container_id"] is None
        identity = real_create(body, name)
        created.append((identity, name))
        raise OSError(KEY)
    api.create = lost_create
    with pytest.raises(OSError):
        probe_prepared(tmp_path, api)
    ledger = life.read_json(tmp_path / "attempt" / "artifact-probe-intent.json")
    assert len(created) == 1 and created[0][1] == ledger["name"]
    assert created[0][1] == ledger["final_name"] + "-n" + ledger["resource_nonce"]
    assert created[0][0] == ledger["container_id"]
    assert ledger["reconciliation"] == "verified_cleanup_only"
    assert ledger["phase"] == "VERIFIED_OWN_RETAINED" and ledger["cleanup_admissible"] is True
    assert ledger["cleanup_complete"] is False and ledger["observed_stopped"] is True
    assert len(ledger["actual_configuration_sha256"]) == 64
    # Zero mutation after the unknown response: the own stopped probe is retained.
    assert created[0][0] in api.docs and not api.docs[created[0][0]]["State"]["Running"]
    state = life.Journal(tmp_path / "attempt").read()
    assert "probe_id" not in state and state["phase"] == "PREFLIGHT_FAILED"
    assert state["operation"] == "create_artifact_probe"
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)


def test_lost_artifact_probe_create_without_effect_stays_unresolved_without_retry(tmp_path):
    api = ArtifactProbeDocker()
    attempts = []
    def lost_create(body, name):
        attempts.append(name)
        raise OSError(KEY)
    api.create = lost_create
    with pytest.raises(OSError):
        probe_prepared(tmp_path, api)
    ledger = life.read_json(tmp_path / "attempt" / "artifact-probe-intent.json")
    assert len(attempts) == 1
    assert ledger["phase"] == "UNRESOLVED_CREATE" and ledger["reconciliation"] == "unresolved"
    assert ledger["container_id"] is None and ledger["cleanup_admissible"] is False
    assert ledger["create_outcome"] == "uncertain" and ledger["create_error_status"] is None
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)


@pytest.mark.parametrize("tamper", ["environment", "name"])
def test_uncertain_artifact_probe_create_refuses_altered_probe(tmp_path, tamper):
    api = ArtifactProbeDocker()
    real_create = api.create
    def lost_create(body, name):
        identity = real_create(body, name)
        if tamper == "environment":
            api.docs[identity]["Config"]["Env"].append("LD_PRELOAD=/unsafe.so")
        else:
            api.docs[identity]["Name"] = "/unrelated"
        raise OSError(KEY)
    api.create = lost_create
    with pytest.raises(OSError):
        probe_prepared(tmp_path, api)
    ledger = life.read_json(tmp_path / "attempt" / "artifact-probe-intent.json")
    assert ledger["phase"] == "UNRESOLVED_CREATE" and ledger["reconciliation"] == "unresolved"
    assert ledger["container_id"] is None and ledger["cleanup_admissible"] is False
    assert CANDIDATE in api.docs
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)


def test_rejected_artifact_probe_create_never_adopts(tmp_path):
    api = ArtifactProbeDocker()
    def rejected_create(body, name):
        raise DockerError("create conflict", 409)
    api.create = rejected_create
    with pytest.raises(DockerError):
        probe_prepared(tmp_path, api)
    ledger = life.read_json(tmp_path / "attempt" / "artifact-probe-intent.json")
    assert ledger["phase"] == "UNRESOLVED_CREATE" and ledger["reconciliation"] == "unresolved"
    assert ledger["container_id"] is None and ledger["cleanup_admissible"] is False
    assert ledger["create_outcome"] == "rejected" and ledger["create_error_status"] == 409
    assert not any(op in ("start", "stop", "remove", "rename") for op, _ in api.events)


def test_lost_artifact_probe_start_reply_records_identity_without_stop_or_remove(tmp_path):
    api = ArtifactProbeDocker()
    def lost_start(identity):
        api.docs[identity]["State"].update(Running=True, Status="running")
        raise OSError(KEY)
    api.start = lost_start
    with pytest.raises(OSError):
        probe_prepared(tmp_path, api)
    ledger = life.read_json(tmp_path / "attempt" / "artifact-probe-intent.json")
    assert ledger["container_id"] == CANDIDATE
    assert ledger["reconciliation"] == "verified_cleanup_only"
    assert ledger["phase"] == "VERIFIED_OWN_RETAINED" and ledger["cleanup_admissible"] is True
    assert ledger["observed_running"] is True and ledger["cleanup_complete"] is False
    assert len(ledger["actual_configuration_sha256"]) == 64
    # Zero mutation after the unknown response: the own running probe is retained.
    assert CANDIDATE in api.docs and api.docs[CANDIDATE]["State"]["Running"]
    assert life.Journal(tmp_path / "attempt").read()["phase"] == "PREFLIGHT_FAILED"
    assert not any(op in ("stop", "remove", "rename") for op, _ in api.events)


def test_explicit_artifact_probe_cleanup_revalidates_then_removes_verified_own_probe(tmp_path):
    api = ArtifactProbeDocker()
    real_create = api.create
    def lost_create(body, name):
        identity = real_create(body, name)
        raise OSError(KEY)
    api.create = lost_create
    with pytest.raises(OSError):
        probe_prepared(tmp_path, api)
    journal = life.Journal(tmp_path / "attempt")
    before = list(api.events)
    outcome = life.explicit_artifact_probe_cleanup(api, journal)
    assert outcome["removed"] is True
    assert CANDIDATE not in api.docs
    assert ("remove", CANDIDATE) in api.events[len(before):]
    assert not any(op in ("start", "stop", "rename", "create") for op, _ in api.events[len(before):])
    ledger = life.read_json(journal.directory / "artifact-probe-intent.json")
    assert ledger["reconciliation"] == "explicit_cleanup_removed" and ledger["cleanup_complete"] is True
    assert ledger["cleanup_admissible"] is False
    # Separately invoked cleanup preserves the original failed attempt.
    assert journal.read()["phase"] == "PREFLIGHT_FAILED"


@pytest.mark.parametrize("tamper", ["foreign", "config", "running", "missing"])
def test_explicit_artifact_probe_cleanup_refuses_unknown_or_altered_without_mutation(tmp_path, tamper):
    api = ArtifactProbeDocker()
    real_create = api.create
    def lost_create(body, name):
        identity = real_create(body, name)
        raise OSError(KEY)
    api.create = lost_create
    with pytest.raises(OSError):
        probe_prepared(tmp_path, api)
    if tamper == "foreign":
        api.docs[CANDIDATE]["Name"] = "/unrelated"
    elif tamper == "config":
        api.docs[CANDIDATE]["Config"]["Env"].append("LD_PRELOAD=/unsafe.so")
    elif tamper == "running":
        api.docs[CANDIDATE]["State"].update(Running=True, Status="running")
    else:
        del api.docs[CANDIDATE]
    journal = life.Journal(tmp_path / "attempt")
    before = list(api.events)
    outcome = life.explicit_artifact_probe_cleanup(api, journal)
    if tamper == "missing":
        assert outcome["removed"] is True and outcome.get("already_absent") is True
    else:
        assert outcome["removed"] is False and outcome.get("refused")
    # Refusal is read-only: no mutation of any kind was issued.
    assert not any(op in ("start", "stop", "remove", "rename", "create") for op, _ in api.events[len(before):])
    assert journal.read()["phase"] == "PREFLIGHT_FAILED"
    ledger = life.read_json(journal.directory / "artifact-probe-intent.json")
    assert ledger.get("cleanup_refused") or ledger.get("observed_absent")


def test_helper_operation_admission_is_bound_to_process_owner_not_only_generation(tmp_path, monkeypatch):
    api, journal, requester = helper_setup(tmp_path)
    assert requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1).status == 200
    gate = life.read_json(journal.directory / "helper-admission.json")
    assert gate["owner"]["pid"] == os.getpid()
    other = {**gate["owner"], "process_nonce": "f" * 32}
    monkeypatch.setattr(life, "helper_owner_identity", lambda: other)
    before = list(api.events)
    assert requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 1).status is None
    assert api.events == before


def test_stalled_old_helper_preflight_never_delays_mutated_original_restoration(tmp_path):
    api, journal, services = authorized(tmp_path)
    swapped(api, journal)
    requester = life.IngressRequester(api, journal.original(), journal.read()["plan"], journal)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original_inspect_image = api.inspect_image
    def delayed_image(identity):
        entered.set()
        assert release.wait(10)
        return original_inspect_image(identity)
    api.inspect_image = delayed_image
    def operation():
        try:
            requester("GET", "http://neo-k3-ingress:8080/v1/models", KEY, None, 2)
            return {"ok": False}
        finally:
            finished.set()
    def guard():
        if entered.is_set():
            raise life.Failure("synthetic abandonment after serving mutation")
    try:
        with pytest.raises(life.Failure):
            life.exit_aware(api, CANDIDATE, CANDIDATE_IMAGE, operation, time.monotonic() + 3,
                           guard=guard, interval=0.005)
        result = life.recover(api, journal, services=services)
        assert result["verified"] and result["result"] == "RESTORED"
        assert not finished.is_set()  # Restoration did not join the blocked old work.
        assert_restored(api, journal)
        before = list(api.events)
        release.set()
        assert finished.wait(5)
        assert not any(op in ("create", "start") for op, _ in api.events[len(before):])
    finally:
        release.set()
        assert finished.wait(5)


def test_operation_fence_io_failure_preserves_service_first_and_withholds_clean_receipt(tmp_path, monkeypatch):
    api, journal, services = authorized(tmp_path)
    swapped(api, journal)
    def unavailable(*args, **kwargs):
        raise OSError(KEY)
    monkeypatch.setattr(life, "rotate_helper_admission", unavailable)
    result = life.recover(api, journal, services=services)
    assert result["service_restored"] and not result["verified"]
    assert result["operation_fence_error"] == "OSError" and not result["cleanup_complete"]
    assert KEY not in json.dumps(result)
    assert_restored(api, journal)
