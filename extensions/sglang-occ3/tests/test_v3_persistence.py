"""Candidate persistence policy: a healthy candidate is not a failed experiment.

The regression this file covers is the deployment behaviour that rolled a
HEALTHY candidate back only because a research proof deadline elapsed. The
plan field is `persistence_policy` with two values: `rollback_on_success`
(default, and the policy when the field is absent) restores the preserved
original at the verified commit point, and `persist_candidate_on_success`
keeps the verified candidate running as the new production service.

Offline only: FakeDocker, fixture service proofs and synthetic credentials.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_v3_lifecycle import (CANDIDATE, CANDIDATE_IMAGE, IMAGE, KEY, ORIGINAL,
                               FakeDocker, approval, assert_restored, authorized,
                               native_receipt, plan, prepared, swapped)
from deploy.v3 import controller as ctl
from deploy.v3 import lifecycle as life
from deploy.v3 import supervisor as sup

PERSIST = life.PERSIST_CANDIDATE_ON_SUCCESS
ROLLBACK = life.ROLLBACK_ON_SUCCESS
PRESERVED = {"container_id": ORIGINAL, "image_id": IMAGE, "name": "fixture-service"}


def persist_plan(**changes):
    changes.setdefault("persistence_policy", PERSIST)
    changes.setdefault("preserved_original", dict(PRESERVED))
    return changes


def persist_authorized(tmp_path, **changes):
    return authorized(tmp_path, **persist_plan(**changes))


class Finished:
    """A supervisor child that has already finished with the given exit code."""
    pid = 999

    def __init__(self, code=0):
        self.code = code

    def poll(self):
        return self.code


def verified_success(api, journal):
    """The exact verified completion state both commit policies start from."""
    swapped(api, journal)
    journal.update(phase="GATES_COMPLETE", restart_count=1,
                   gates={name: {"ok": True} for name in sup.REQUIRED_GATES})


def run_successful_guardian(tmp_path, api, journal, services, monkeypatch):
    holder = {}

    def finish_gates():
        verified_success(api, journal)
        holder["child"] = Finished(0)
        return holder["child"]

    monkeypatch.setattr(sup, "process_identity", lambda pid: {"pid": pid, "start_ticks": "offline"})
    guardian_process = sup.Supervisor(api, journal, services=services, process_factory=finish_gates,
                                      lock_root=tmp_path / "locks", fence_root=tmp_path / "fences")
    return guardian_process.run(approval(journal))


def native_chain_guardian(journal, *, initial=True, final=True, permit=True,
                          close_window_when=None):
    """Supervisor stand-in providing only the selected native proof links.

    Receipts are written when their checkpoint appears, the validation restart
    permit only when selected, and the operator maintenance window closes at a
    chosen journal state, so an incomplete chain stays incomplete.
    """
    closed = []
    def guardian():
        state = journal.read()
        request = state.get("restart_request")
        if request and not state.get("restart_count") and permit:
            journal.update(restart_count=1)
            life.private_json(journal.directory / "restart-permit.json", {
                "attempt_id": state["attempt_id"], "candidate_id": CANDIDATE,
                "nonce": request["nonce"], "epoch": request["epoch"],
                "expires_monotonic": time.monotonic() + 1})
        for phase, provide in (("initial", initial), ("final", final)):
            checkpoint = journal.directory / ("checkpoint-" + phase + ".json")
            receipt = journal.directory / ("native-proof-" + phase + ".json")
            if provide and checkpoint.exists() and not receipt.exists():
                life.private_json(receipt, native_receipt(journal, life.read_json(checkpoint)))
        # One-shot closure only: the real external guard is a liveness check,
        # so after the single raise this seam must stay a no-op for the rest of
        # the run (wait_ready calls the guard during persistence verification
        # and converts any repeated deadline into a readiness failure).
        if close_window_when is not None and not closed and close_window_when(state):
            closed.append(True)
            raise ctl.MaintenanceWindowClosed("operator maintenance window closed")
    return guardian


def guardian_persists_completed_chain(api, journal, services):
    """A controller that finishes the whole chain, then closes the window."""
    def close_after_complete(state):
        return sup.REQUIRED_GATES <= (state.get("gates") or {}).keys()
    return ctl.Controller(api, journal, services=services,
                          guard=native_chain_guardian(journal, close_window_when=close_after_complete))


# --- (a) success + rollback policy still rolls back (regression) ------------


def test_success_with_rollback_policy_restores_production(tmp_path, monkeypatch):
    api, journal, services = prepared(tmp_path)
    assert run_successful_guardian(tmp_path, api, journal, services, monkeypatch) == 0
    state = journal.read()
    assert state["phase"] == "RESTORED"
    assert state["persistence_decision"] == life.ROLLED_BACK_ON_SUCCESS
    assert state["persistence"]["policy"] == ROLLBACK
    assert state["persistence"]["recovery_verified"] is True
    assert state["persistence"]["original"]["container_id"] == ORIGINAL
    assert state["persistence"]["original"]["image_id"] == IMAGE
    command = state["persistence"]["recovery_command"]
    assert "deploy.v3.controller restore --attempt-dir" in command
    assert str(journal.directory.resolve()) in command
    assert_restored(api, journal)


# --- (b) success + persist policy leaves the candidate running ---------------


def test_success_with_persist_policy_keeps_the_candidate_running(tmp_path, monkeypatch):
    api, journal, services = prepared(tmp_path, **persist_plan())
    assert run_successful_guardian(tmp_path, api, journal, services, monkeypatch) == 0
    state = journal.read()
    assert state["phase"] == "COMPLETE"
    assert state["persistence_decision"] == life.PERSISTED_CANDIDATE
    record = state["persistence"]
    assert record["decision"] == life.PERSISTED_CANDIDATE and record["policy"] == PERSIST
    assert record["candidate_id"] == CANDIDATE and record["candidate_image_id"] == CANDIDATE_IMAGE
    assert record["candidate_name"] == state["original_name"]
    # Preserved original container metadata, image and launch configuration.
    assert set(record["original"]) == {"container_id", "image_id", "name",
                                      "configuration_sha256"}
    assert record["original"]["container_id"] == ORIGINAL
    assert record["original"]["image_id"] == IMAGE
    assert record["original"]["name"] == "fixture-service"
    assert len(record["original"]["configuration_sha256"]) == 64
    assert state["preserved_original"] == record["original"]
    assert "deploy.v3.controller restore" in record["recovery_command"]
    assert api.docs[CANDIDATE]["State"]["Running"] is True
    assert api.docs[CANDIDATE]["Name"] == "/" + state["original_name"]
    assert api.docs[ORIGINAL]["State"]["Running"] is False
    assert api.docs[ORIGINAL]["Name"] == "/" + state["preserved_name"]
    result = json.loads((journal.directory / "supervisor-result.json").read_text())
    assert result["result"] == "COMPLETE" and result["persistence_decision"] == life.PERSISTED_CANDIDATE
    assert KEY not in (journal.directory / "journal.json").read_text()


# --- (c) an incomplete native proof chain is a restoration, never retention --


@pytest.mark.parametrize("policy", [ROLLBACK, PERSIST])
def test_missing_initial_native_proof_deadline_restores_under_both_policies(tmp_path, policy):
    changes = persist_plan() if policy == PERSIST else {}
    api, journal, services = authorized(tmp_path, **changes)
    # No external proof receipts are provided, so the experimental proof deadline
    # fires while the initial native proof is still missing.
    assert ctl.Controller(api, journal, services=services, guard=lambda: None).run() == 1
    state = journal.read()
    assert state["plan"]["persistence_policy"] == policy
    assert state["phase"] == "RESTORED"
    assert state["recovery"]["reason"] == "controller_ExperimentDeadline"
    assert life.READINESS_GATE in state["gates"] and "native_initial" not in state["gates"]
    if policy == PERSIST:
        assert "persistence" not in state and life.PERSISTENCE_GATE not in state["gates"]
        assert state.get("persistence_decision") != life.PERSISTED_CANDIDATE
    assert_restored(api, journal)


def test_missing_final_native_proof_restores_under_persist(tmp_path):
    api, journal, services = persist_authorized(tmp_path, proof_timeout_s=2)
    services.live_ok = True
    guardian = native_chain_guardian(journal, initial=True, final=False)
    assert ctl.Controller(api, journal, services=services, guard=guardian).run() == 1
    state = journal.read()
    gates = state["gates"]
    # Six of the seven chain gates passed; only the final proof link is missing.
    for name in ("service_initial", "native_initial", "live_before", "restart", "live_after", "service_final"):
        assert gates[name]["ok"] is True
    assert "native_final" not in gates
    assert state["recovery"]["reason"] == "controller_ExperimentDeadline"
    assert state["phase"] == "RESTORED" and "persistence" not in state
    assert_restored(api, journal)


def test_window_closing_after_live_before_but_before_live_after_restores(tmp_path):
    api, journal, services = persist_authorized(tmp_path, proof_timeout_s=30)
    services.live_ok = True
    # The initial proof is provided and live_before passes, but the restart
    # permit never arrives and the window closes in the permit wait.
    guardian = native_chain_guardian(
        journal, initial=True, final=False, permit=False,
        close_window_when=lambda state: bool(state.get("restart_request")))
    assert ctl.Controller(api, journal, services=services, guard=guardian).run() == 1
    state = journal.read()
    gates = state["gates"]
    assert gates["live_before"]["ok"] is True and "live_after" not in gates
    assert state.get("restart_count") != 1
    assert state["recovery"]["reason"] == "controller_MaintenanceWindowClosed"
    assert state["phase"] == "RESTORED" and "persistence" not in state
    assert_restored(api, journal)


def test_deployment_deadline_never_persists_a_candidate(tmp_path):
    api, journal, services = persist_authorized(tmp_path)
    def guardian():
        if life.READINESS_GATE in (journal.read().get("gates") or {}):
            raise ctl.Deadline("controller deployment deadline")
    assert ctl.Controller(api, journal, services=services, guard=guardian).run() == 1
    state = journal.read()
    assert state["recovery"]["reason"] == "controller_Deadline"
    assert state["phase"] == "RESTORED" and "persistence" not in state
    assert_restored(api, journal)


def test_persist_applies_requires_every_required_gate(tmp_path):
    api, journal, services = persist_authorized(tmp_path)
    runner = ctl.Controller(api, journal, services=services, guard=lambda: None)
    journal.update(candidate_started=True)
    chain = sorted(sup.REQUIRED_GATES)
    for missing in chain:
        partial = [name for name in chain if name != missing]
        journal.update(gates={name: {"ok": True} for name in partial})
        assert runner.persist_applies(ctl.ExperimentDeadline(
            "external native Optimus proof deadline")) is False
    journal.update(gates={name: {"ok": True} for name in chain})
    assert runner.persist_applies(ctl.ExperimentDeadline(
        "external native Optimus proof deadline")) is True
    assert runner.persist_applies(ctl.MaintenanceWindowClosed(
        "operator maintenance window closed")) is True
    # A failed gate in the chain is also an incomplete chain.
    journal.update(gates={**{name: {"ok": True} for name in chain}, "native_final": {"ok": False}})
    assert runner.persist_applies(ctl.ExperimentDeadline(
        "external native Optimus proof deadline")) is False
    # A deadline that is not a research proof or window expiry never applies.
    journal.update(gates={name: {"ok": True} for name in chain})
    assert runner.persist_applies(ctl.Deadline("controller deployment deadline")) is False


def test_persist_candidate_refuses_an_incomplete_chain(tmp_path):
    api, journal, services = persist_authorized(tmp_path)
    runner = ctl.Controller(api, journal, services=services, guard=lambda: None)
    journal.update(candidate_started=True, gates={life.READINESS_GATE: {"ok": True}})
    with pytest.raises(life.Failure, match="complete native proof chain"):
        runner.persist_candidate("external native Optimus proof deadline")
    assert "persistence" not in journal.read()


def test_complete_chain_window_closure_retains_the_candidate(tmp_path):
    api, journal, services = persist_authorized(tmp_path, proof_timeout_s=30)
    services.live_ok = True
    runner = guardian_persists_completed_chain(api, journal, services)
    assert runner.run() == 0
    state = journal.read()
    # Every required gate passed before the window closed, so retention applies.
    assert all(state["gates"][name]["ok"] is True for name in sup.REQUIRED_GATES)
    assert state["phase"] == life.PERSISTED_PHASE
    assert state["persistence_decision"] == life.PERSISTED_CANDIDATE
    assert state["persistence"]["reason"] == "operator maintenance window closed"
    assert state["gates"][life.PERSISTENCE_GATE]["ok"] is True
    assert api.docs[CANDIDATE]["State"]["Running"] is True
    assert api.docs[ORIGINAL]["State"]["Running"] is False


# --- (c2) the supervisor independently rejects a partial persistence -------


def forged_persistence(api, journal, gates):
    """A durable PERSISTED_CANDIDATE record with the given proof-chain gates."""
    if api.docs[CANDIDATE].get("Name") != "/" + journal.read()["original_name"]:
        swapped(api, journal)
    life.record_persistence(journal, journal.read(), api.docs[CANDIDATE], journal.original(),
                            decision=life.PERSISTED_CANDIDATE, reason="forged",
                            gate=life.PERSISTENCE_GATE)
    journal.update(phase=life.PERSISTED_PHASE,
                   gates={name: {"ok": True} for name in gates})


def test_persisted_state_requires_the_complete_native_proof_chain(tmp_path):
    api, journal, services = persist_authorized(tmp_path)
    complete = sorted(sup.REQUIRED_GATES) + [life.PERSISTENCE_GATE]
    for missing in sup.REQUIRED_GATES:
        forged_persistence(api, journal, [name for name in complete if name != missing])
        assert sup.persisted(journal.read()) is False, missing
    forged_persistence(api, journal, complete)
    assert sup.persisted(journal.read()) is True
    # One failed gate inside the chain also breaks the independent acceptance.
    journal.update(gates={**{name: {"ok": True} for name in complete}, "restart": {"ok": False}})
    assert sup.persisted(journal.read()) is False


def test_guardian_recovers_a_forged_partial_persisted_state(tmp_path, monkeypatch):
    api, journal, services = prepared(tmp_path, **persist_plan())
    def forge_partial():
        forged_persistence(api, journal, [life.READINESS_GATE, life.PERSISTENCE_GATE,
                                          "native_initial", "live_before"])
        return Finished(0)
    monkeypatch.setattr(sup, "process_identity", lambda pid: {"pid": pid, "start_ticks": "offline"})
    guardian = sup.Supervisor(api, journal, services=services, process_factory=forge_partial,
                              lock_root=tmp_path / "locks", fence_root=tmp_path / "fences")
    assert guardian.run(approval(journal)) == 1
    state = journal.read()
    assert state["phase"] == "RESTORED" and state["recovery"]["verified"]
    assert_restored(api, journal)


def test_guardian_commits_a_complete_chain_persisted_state(tmp_path, monkeypatch):
    api, journal, services = prepared(tmp_path, **persist_plan())
    def complete_then_persist():
        verified_success(api, journal)
        journal.update(gates={**journal.read()["gates"], life.PERSISTENCE_GATE: {"ok": True}})
        life.record_persistence(journal, journal.read(), api.docs[CANDIDATE], journal.original(),
                                decision=life.PERSISTED_CANDIDATE,
                                reason="external native Optimus proof deadline",
                                gate=life.PERSISTENCE_GATE)
        journal.update(phase=life.PERSISTED_PHASE)
        return Finished(0)
    monkeypatch.setattr(sup, "process_identity", lambda pid: {"pid": pid, "start_ticks": "offline"})
    guardian = sup.Supervisor(api, journal, services=services, process_factory=complete_then_persist,
                              lock_root=tmp_path / "locks", fence_root=tmp_path / "fences")
    assert guardian.run(approval(journal)) == 0
    state = journal.read()
    assert state["phase"] == "COMPLETE"
    assert state["persistence_decision"] == life.PERSISTED_CANDIDATE
    assert state["persistence"]["reason"] == "supervisor_persist"
    assert api.docs[CANDIDATE]["State"]["Running"] is True
    assert api.docs[ORIGINAL]["State"]["Running"] is False


def test_research_stage_deadlines_are_recorded_and_never_fatal(tmp_path):
    api, journal, services = persist_authorized(
        tmp_path, source_capture_timeout_s=0.01, receiver_inference_timeout_s=0.01, proof_timeout_s=0.05)
    # The stage deadlines stay non-fatal records; the restoration comes from the
    # incomplete proof chain at the experimental deadline, never from the stages.
    assert ctl.Controller(api, journal, services=services, guard=lambda: None).run() == 1
    state = journal.read()
    assert state["phase"] == "RESTORED"
    stages = state["stage_deadlines"]
    assert set(stages) == {"initial: source capture deadline", "initial: receiver inference deadline"}
    assert all(entry["fatal"] is False for entry in stages.values())
    assert {entry["field"] for entry in stages.values()} == {"source_capture_timeout_s",
                                                            "receiver_inference_timeout_s"}
    assert "persistence" not in state
    assert_restored(api, journal)


# --- (d) a failed health check still rolls back under the persist policy -----


def test_startup_health_failure_under_persist_still_restores_production(tmp_path):
    api, journal, services = persist_authorized(tmp_path, startup_timeout_s=0.4)
    api.behavior = "unready"
    assert ctl.Controller(api, journal, services=services, guard=lambda: None).run() == 1
    state = journal.read()
    assert state["phase"] == "RESTORED" and state["recovery"]["verified"]
    assert state.get("persistence_decision") != life.PERSISTED_CANDIDATE
    assert "persistence" not in state and life.PERSISTENCE_GATE not in state["gates"]
    assert_restored(api, journal)


def test_persistence_readiness_gate_failure_restores_production(tmp_path):
    """The persistence gate repeats every readiness check, so it cannot persist a bad candidate."""
    api, journal, services = persist_authorized(tmp_path)
    real_identity = services.occ3_identity
    def identity():
        gates = (journal.read().get("gates") or {})
        if life.READINESS_GATE in gates:
            return {"ok": False, "check": "occ3_identity", "http_status": 503,
                    "error_code": "occ3_identity_unavailable"}
        return real_identity()
    services.occ3_identity = identity
    assert ctl.Controller(api, journal, services=services, guard=lambda: None).run() == 1
    state = journal.read()
    assert state["phase"] == "RESTORED" and "persistence" not in state
    assert life.PERSISTENCE_GATE not in state["gates"]
    assert_restored(api, journal)


# --- (e) an ordinary-inference failure is a service health failure ----------


def test_ordinary_inference_failure_under_persist_restores_production(tmp_path):
    api, journal, services = persist_authorized(tmp_path)
    def chat():
        if api.docs[CANDIDATE]["State"]["Running"]:
            return {"ok": False, "check": "chat", "http_status": 200, "error_code": "chat_semantics",
                    "semantic_match": False}
        return {"ok": True, "check": "chat"}
    services.chat = chat
    assert ctl.Controller(api, journal, services=services, guard=lambda: None).run() == 1
    state = journal.read()
    assert state["phase"] == "RESTORED" and state["recovery"]["verified"]
    assert state["recovery"]["reason"] == "controller_Failure"
    assert "service_initial" not in state["gates"] and "persistence" not in state
    assert_restored(api, journal)


def test_service_health_experiment_signal_under_persist_restores_production(tmp_path):
    api, journal, services = persist_authorized(tmp_path)
    def guard():
        signal = journal.directory / "experiment-signal-initial.json"
        if not signal.exists() and life.READINESS_GATE in (journal.read().get("gates") or {}):
            life.private_json(signal, {"schema": life.EXPERIMENT_SIGNAL_SCHEMA, "phase": "initial",
                                       "code": "receiver_transport_error"})
    assert ctl.Controller(api, journal, services=services, guard=guard).run() == 1
    state = journal.read()
    assert state["experiment_signals"]["initial"]["rollback_eligible"] is True
    assert state["phase"] == "RESTORED"
    assert_restored(api, journal)


# --- (f) a semantic experiment failure never triggers a rollback -----------


@pytest.mark.parametrize("policy", [ROLLBACK, PERSIST])
def test_semantic_experiment_failure_never_triggers_a_rollback(tmp_path, policy):
    changes = persist_plan() if policy == PERSIST else {}
    api, journal, services = authorized(tmp_path, proof_timeout_s=2, **changes)
    services.live_ok = True
    def guardian_and_external_executor():
        state = journal.read()
        request = state.get("restart_request")
        if request and not state.get("restart_count"):
            journal.update(restart_count=1)
            life.private_json(journal.directory / "restart-permit.json", {
                "attempt_id": state["attempt_id"], "candidate_id": CANDIDATE,
                "nonce": request["nonce"], "epoch": request["epoch"],
                "expires_monotonic": time.monotonic() + 1})
        for phase in ("initial", "final"):
            signal = journal.directory / ("experiment-signal-" + phase + ".json")
            if not signal.exists():
                life.private_json(signal, {"schema": life.EXPERIMENT_SIGNAL_SCHEMA, "phase": phase,
                                           "code": "receiver_answer_mismatch"})
            checkpoint = journal.directory / ("checkpoint-" + phase + ".json")
            receipt = journal.directory / ("native-proof-" + phase + ".json")
            if checkpoint.exists() and not receipt.exists():
                life.private_json(receipt, native_receipt(journal, life.read_json(checkpoint)))
    runner = ctl.Controller(api, journal, services=services, guard=guardian_and_external_executor)
    assert runner.run() == 0
    state = journal.read()
    assert state["phase"] == "GATES_COMPLETE"
    recorded = state["experiment_signals"]
    assert set(recorded) == {"initial", "final"}
    assert all(entry["classification"] == life.EXPERIMENT_SEMANTIC for entry in recorded.values())
    assert all(entry["rollback_eligible"] is False for entry in recorded.values())
    assert all(entry["code_known"] is True for entry in recorded.values())
    assert state.get("recovery") is None
    assert api.docs[CANDIDATE]["State"]["Running"] is True and api.docs[ORIGINAL]["State"]["Running"] is False


def test_unknown_experiment_code_stays_in_the_non_fatal_class():
    assert life.classify_experiment_signal(
        {"schema": life.EXPERIMENT_SIGNAL_SCHEMA, "code": "something_new"}) == life.EXPERIMENT_SEMANTIC
    assert life.classify_experiment_signal(
        {"schema": life.EXPERIMENT_SIGNAL_SCHEMA, "code": "receiver_timeout"}) == life.EXPERIMENT_SERVICE_HEALTH
    with pytest.raises(life.Failure, match="schema"):
        life.classify_experiment_signal({"code": "receiver_timeout"})


# --- (g) operator-driven manual rollback still works under the persist policy


def test_operator_manual_restore_after_persistence(tmp_path, monkeypatch):
    api, journal, services = prepared(tmp_path, **persist_plan())
    assert run_successful_guardian(tmp_path, api, journal, services, monkeypatch) == 0
    assert journal.read()["phase"] == "COMPLETE"
    assert api.docs[CANDIDATE]["State"]["Running"] is True
    result = ctl.restore(api, journal, services=services)
    assert result["verified"] is True and result["result"] == "RESTORED"
    assert result["reason"] == "operator_manual_restore"
    state = journal.read()
    assert state["phase"] == "RESTORED"
    assert state["persistence"]["manual_restore"] == "RESTORED"
    assert state["persistence"]["manual_restore_verified"] is True
    # The persistence decision itself is history and stays recorded.
    assert state["persistence_decision"] == life.PERSISTED_CANDIDATE
    assert state["persistence"]["original"]["container_id"] == ORIGINAL
    assert_restored(api, journal)
    assert ctl.main(["restore"]) == 2  # the documented CLI command requires --attempt-dir


def test_manual_restore_reports_failure_when_production_cannot_be_reverified(tmp_path, monkeypatch):
    api, journal, services = prepared(tmp_path, rollback_timeout_s=1.5, **persist_plan())
    assert run_successful_guardian(tmp_path, api, journal, services, monkeypatch) == 0
    services.fail_restore = True
    result = ctl.restore(api, journal, services=services)
    assert result["verified"] is False and result["result"] == "RESTORE_FAILED"
    assert api.docs[ORIGINAL]["State"]["Running"] is True
    assert not api.docs[CANDIDATE]["State"]["Running"]
    assert journal.read()["phase"] == "RESTORE_FAILED"


# --- (h) the plan validator refuses a persist policy without the record -----


def test_plan_validator_rejects_persist_without_preserved_original():
    with pytest.raises(life.Failure, match="preserved original container record"):
        life.validate_plan(plan(persistence_policy=PERSIST))
    with pytest.raises(life.Failure, match="differs from the attempt original"):
        life.validate_plan(plan(persistence_policy=PERSIST,
                                preserved_original={"container_id": "3" * 64, "image_id": IMAGE,
                                                    "name": "fixture-service"}))
    with pytest.raises(life.Failure, match="invalid preserved original container record"):
        life.validate_plan(plan(preserved_original={"container_id": ORIGINAL}))
    with pytest.raises(life.Failure, match="invalid preserved original container name"):
        life.validate_plan(plan(persistence_policy=PERSIST,
                                preserved_original={"container_id": ORIGINAL, "image_id": IMAGE,
                                                    "name": "../escape"}))
    with pytest.raises(life.Failure, match="invalid candidate persistence policy"):
        life.validate_plan(plan(persistence_policy="keep_the_candidate"))
    assert life.validate_plan(plan())["persistence_policy"] == ROLLBACK
    assert life.validate_plan(plan(**persist_plan()))["persistence_policy"] == PERSIST


def test_prepare_refuses_a_preserved_record_that_differs_from_the_original(tmp_path):
    api = FakeDocker()
    with pytest.raises(life.Failure, match="differs from the inspected original"):
        prepared(tmp_path, api=api, persistence_policy=PERSIST,
                 preserved_original={"container_id": ORIGINAL, "image_id": IMAGE,
                                     "name": "not-this-service"})
    assert api.docs[ORIGINAL]["State"]["Running"]
    assert not any(op in ("start", "stop", "rename", "remove") for op, _ in api.events)


# --- timeout types are separate fields, never one shared budget -------------


def test_timeout_types_are_distinct_plan_fields():
    values = list(life.TIMEOUT_FIELDS.values())
    assert len(set(values)) == len(values)
    assert life.RESEARCH_STAGE_DEADLINES == {"source capture deadline": "source_capture_timeout_s",
                                             "receiver inference deadline": "receiver_inference_timeout_s"}
    configured = life.validate_plan(plan(request_timeout_s=7, source_capture_timeout_s=8,
                                         receiver_inference_timeout_s=9, proof_timeout_s=10,
                                         startup_timeout_s=11, maintenance_window_s=12))
    policy = life.timeout_policy(configured)
    assert policy["model_request_timeout_s"] == 7
    assert policy["native_source_capture_deadline_s"] == 8
    assert policy["receiver_inference_deadline_s"] == 9
    assert policy["experimental_proof_deadline_s"] == 10
    assert policy["deployment_health_timeout_s"] == 11
    assert policy["operator_maintenance_window_s"] == 12
    assert policy["candidate_persistence_policy"] == ROLLBACK
    for field in ("source_capture_timeout_s", "receiver_inference_timeout_s", "maintenance_window_s"):
        with pytest.raises(life.Failure, match="invalid deadline option"):
            life.validate_plan(plan(**{field: 0}))


def test_prepare_records_the_named_deadlines_in_the_journal(tmp_path):
    api, journal, _ = persist_authorized(tmp_path)
    state = journal.read()
    policy = state["timeout_policy"]
    assert policy["candidate_persistence_policy"] == PERSIST
    for name, field in life.TIMEOUT_FIELDS.items():
        assert policy[name] == state["plan"][field]
    assert state["preserved_original"]["container_id"] == ORIGINAL
    assert state["preserved_original"]["image_id"] == IMAGE
    assert len(state["preserved_original"]["configuration_sha256"]) == 64
    assert KEY not in (journal.directory / "journal.json").read_text()


def test_maintenance_window_closing_mid_proof_restores_under_persist(tmp_path):
    api, journal, services = persist_authorized(tmp_path, maintenance_window_s=1.0, proof_timeout_s=5)
    # The window closes while the initial native proof is still missing, so the
    # persist policy restores instead of retaining an unproven candidate.
    assert ctl.Controller(api, journal, services=services, guard=lambda: None).run() == 1
    state = journal.read()
    assert state["phase"] == "RESTORED"
    assert state["recovery"]["reason"] == "controller_MaintenanceWindowClosed"
    assert "persistence" not in state
    assert_restored(api, journal)


# --- a persisted candidate can be replaced by a later controlled attempt ----


def test_persisted_candidate_can_be_replaced_by_a_later_controlled_attempt(tmp_path, monkeypatch):
    api, journal, services = prepared(tmp_path, **persist_plan())
    assert run_successful_guardian(tmp_path, api, journal, services, monkeypatch) == 0
    assert journal.read()["phase"] == "COMPLETE"
    reference = tmp_path / "later-inspect.json"
    reference.write_text(json.dumps(api.docs[CANDIDATE]), encoding="utf-8")
    later = life.prepare(api, plan(attempt_id="later-upgrade-0001", original_id=CANDIDATE,
                                   original_image_id=CANDIDATE_IMAGE,
                                   candidate_image_id="sha256:" + "d" * 64,
                                   inspect_schema_evidence={
                                       "path": str(reference),
                                       "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}),
                         tmp_path / "later", probe=lambda *_: {"result": "PASS", "cpu_only": True})
    assert later.read()["phase"] == "PREPARED"
    # The new production container is not fenced, so the later attempt is admitted.
    assert not (journal.directory / "fences" / (CANDIDATE + ".json")).exists()
    sup.claim_original_fence(later, journal.directory / "fences", {"pid": 0, "start_ticks": "offline"})
    assert (journal.directory / "fences" / (CANDIDATE + ".json")).exists()
