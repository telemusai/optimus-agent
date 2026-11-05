"""Safe-default V3 CLI. Production mutation requires a prepared, approved attempt.

Run from the extension root: python -m deploy.v3.controller preflight --plan FILE
--attempt-dir NEW_DIRECTORY. swap only launches an independent supervisor; it
never implies readiness. Native Optimus proof must be supplied externally twice.

Candidate persistence is a plan field, `persistence_policy`, with exactly two
values: `rollback_on_success` (default, and the policy when the field is absent)
restores the preserved original once the attempt reaches its verified commit
point, and `persist_candidate_on_success` keeps the verified candidate running
as the new production service. The persist policy requires the plan's
`preserved_original` record and never relaxes a readiness gate: an elapsed
research deadline (experimental proof, maintenance window) is not a service
failure, so a healthy candidate is persisted instead of rolled back, while a
genuine health failure still restores production. The explicit operator
recovery path is `python -m deploy.v3.controller restore --attempt-dir DIR`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import time

from .docker_api import DockerAPI
from .lifecycle import (Deadline, Failure, FileLock, Journal, Services,
                        calibrated_binding_from_state, check_supervisor, checked_inspect,
                        exit_aware, is_running, prepare, private_json, read_json, recover,
                        refresh_ingress, require_running, verify_calibrated_source, verify_receipt,
                        wait_ready, ExperimentDeadline, MaintenanceWindowClosed,
                        PERSIST_CANDIDATE_ON_SUCCESS, PERSISTED_CANDIDATE, PERSISTED_PHASE,
                        PERSISTENCE_GATE, READINESS_GATE, RESEARCH_STAGE_DEADLINES,
                        EXPERIMENT_SERVICE_HEALTH, classify_experiment_signal, record_experiment_signal,
                        record_persistence)


def proof_matches(receipt, checkpoint):
    required = ("attempt_id", "candidate_id", "process_epoch", "phase", "challenge")
    if (not isinstance(receipt, dict) or any(receipt.get(k) != checkpoint[k] for k in required) or
            receipt.get("kind") != "native-optimus-glm-response" or
            receipt.get("verified") is not True or receipt.get("fallback_used") is not False or
            not isinstance(receipt.get("native_session_id"), str) or not receipt["native_session_id"] or
            receipt.get("model") != checkpoint["model"] or
            not checkpoint["created_at"] <= receipt.get("verified_at", 0) <= time.time() or
            time.time() > checkpoint["expires_at"]):
        raise Failure("external native proof is missing, stale or mismatched")
    verify_receipt(receipt.get("evidence"))
    return {"ok": True, "kind": receipt["kind"], "phase": checkpoint["phase"],
            "candidate_id": checkpoint["candidate_id"], "process_epoch": checkpoint["process_epoch"],
            "native_session_id": receipt["native_session_id"], "evidence": receipt["evidence"]}


class Controller:
    def __init__(self, api, journal, *, services=None, guard=None):
        self.api, self.journal = api, journal
        self.state, self.original = journal.read(), journal.original()
        self.plan = self.state["plan"]
        self.services = services or Services(self.original, self.plan, api=api, journal=journal)
        self.end = time.monotonic() + self.plan["deployment_timeout_s"]
        self.window_end = time.monotonic() + self.plan["maintenance_window_s"]
        self.verifying_persistence = False
        self.needs_handshake = guard is None
        self.external_guard = guard or (lambda: check_supervisor(journal))

    def persist_mode(self):
        return self.plan.get("persistence_policy") == PERSIST_CANDIDATE_ON_SUCCESS

    def readiness_verified(self):
        evidence = (self.journal.read().get("gates") or {}).get(READINESS_GATE)
        return isinstance(evidence, dict) and evidence.get("ok") is True

    def full_proof_chain_verified(self):
        """The complete native validation chain, exactly as supervisor.complete() requires.

        Retention may not rest on readiness alone: every required gate of the
        authoritative supervisor contract must be present and ok, so a missing,
        failed or still-pending native proof link is a restoration, never a
        persistence.
        """
        from .supervisor import REQUIRED_GATES
        gates = self.journal.read().get("gates") or {}
        return REQUIRED_GATES <= gates.keys() and all(
            isinstance(gates[name], dict) and gates[name].get("ok") is True for name in REQUIRED_GATES)

    def guard(self):
        self.external_guard()
        now = time.monotonic()
        if not self.verifying_persistence:
            if self.persist_mode() and now >= self.window_end:
                raise MaintenanceWindowClosed("operator maintenance window closed")
            if now >= self.end:
                raise Deadline("controller deployment deadline")
        private_json(self.journal.directory / "controller.json", {
            "attempt_id": self.state["attempt_id"], "pid": os.getpid(),
            "heartbeat": time.monotonic(), "status": "running",
        })

    def persist_applies(self, error):
        """Only a research deadline or a closed window can persist a healthy candidate.

        Fail-safe retention: the deadline or window expiry must arrive after the
        complete native proof chain, not merely after readiness. An incomplete
        chain (a missing, failed or still-pending proof link) is a restoration.
        """
        return (isinstance(error, (ExperimentDeadline, MaintenanceWindowClosed)) and
                self.persist_mode() and self.readiness_verified() and
                self.journal.read().get("candidate_started") is True and
                self.full_proof_chain_verified())

    def persist_candidate(self, reason):
        """Declare persistence only after every readiness gate passes again.

        Health, ordinary inference, authentication, native OCC-3 readiness with
        TP8 ranks/runtime identity, tools and the source/receiver routes all
        repeat here, so an unhealthy candidate is never declared persistent.
        Retention additionally requires the complete native proof chain, so a
        failed declaration falls through to restoration, never to retention.
        """
        state = self.journal.read()
        if state["plan"].get("persistence_policy") != PERSIST_CANDIDATE_ON_SUCCESS:
            raise Failure("candidate persistence requires the persist policy")
        if not self.readiness_verified():
            raise Failure("candidate persistence requires the initial readiness gate")
        if not self.full_proof_chain_verified():
            raise Failure("candidate persistence requires the complete native proof chain")
        previous = (self.end, self.window_end, self.verifying_persistence)
        self.verifying_persistence = True
        self.end = time.monotonic() + (self.plan["startup_timeout_s"] + self.plan["request_timeout_s"] +
                                       self.plan["probe_timeout_s"] + 5)
        self.window_end = self.end
        try:
            self.service_gate(PERSISTENCE_GATE)
        finally:
            self.end, self.window_end, self.verifying_persistence = previous
        candidate = require_running(self.api, state["candidate_id"], state["candidate_image_id"])
        record_persistence(self.journal, state, candidate, self.original,
                           decision=PERSISTED_CANDIDATE, reason=str(reason), gate=PERSISTENCE_GATE)
        self.journal.update(phase=PERSISTED_PHASE)
        return 0

    def observe_experiment_signal(self, phase):
        """Classify one durable experiment signal; only a health code is fatal."""
        path = self.journal.directory / ("experiment-signal-" + phase + ".json")
        if not path.exists():
            return
        signal = read_json(path)
        classification = classify_experiment_signal(signal)
        record_experiment_signal(self.journal, phase, signal, classification)
        if classification == EXPERIMENT_SERVICE_HEALTH:
            raise Failure("experiment reported a service health failure: " + str(signal.get("code")))

    def note_research_deadlines(self, phase, started):
        """Source capture and receiver inference deadlines are recorded, never fatal."""
        now = time.monotonic()
        due = {phase + ": " + name: (name, field) for name, field in RESEARCH_STAGE_DEADLINES.items()
               if now >= started + self.plan[field]}
        if not due:
            return
        recorded = dict(self.journal.read().get("stage_deadlines") or {})
        for key, (name, field) in due.items():
            recorded.setdefault(key, {"phase": phase, "deadline": name, "field": field,
                                      "fatal": False, "recorded_at": time.time()})
        self.journal.update(stage_deadlines=recorded)

    def mutate(self, operation, action, **changes):
        self.guard()
        self.journal.intent(operation, **changes)
        self.guard()
        action()
        self.guard()

    def check(self, operation, timeout=None):
        deadline = min(self.end, time.monotonic() + (timeout or self.plan["request_timeout_s"] + 2))
        return exit_aware(self.api, self.state["candidate_id"], self.state["candidate_image_id"],
                          operation, deadline, guard=self.guard, interval=self.plan["poll_interval_s"])

    def wait_health(self):
        observations = {"checks": 0, "started_monotonic": time.monotonic()}
        def observe(evidence):
            observations.update(checks=observations["checks"] + 1,
                last_http_status=evidence.get("http_status") if isinstance(evidence, dict) else None,
                last_ok=evidence.get("ok") if isinstance(evidence, dict) else False,
                observed_running_monotonic=time.monotonic())
            self.journal.update(readiness_observations=observations)
        return wait_ready(self.api, self.state["candidate_id"], self.state["candidate_image_id"],
                          self.services.health, min(self.end, time.monotonic() + self.plan["startup_timeout_s"]),
                          guard=self.guard, interval=self.plan["poll_interval_s"], observation=observe)

    def service_gate(self, name):
        evidence = {"ok": True}
        evidence["health"] = self.wait_health()
        for field in ("occ3_identity", "chat", "tools", "ingress"):
            if field == "ingress":
                evidence["routing_refresh"] = refresh_ingress(self.api, self.journal, self.journal.read(),
                    min(self.end, time.monotonic() + 45), guard=self.guard)
            proof = self.check(getattr(self.services, field))
            if not isinstance(proof, dict) or proof.get("ok") is not True:
                detail = "" if not isinstance(proof, dict) else " (%s http=%s)" % (
                    proof.get("error_code"), proof.get("http_status"))
                if isinstance(proof, dict) and proof.get("error_code") == "occ3_declined":
                    detail += " declined=%s stage=%s" % (proof.get("declined_code"), proof.get("stage"))
                    # A decline is terminal for the server lifetime: keep the exact
                    # code past process exit for the abort review, best effort only.
                    with contextlib.suppress(Exception):
                        self.journal.update(occ3_decline={"gate": name, "field": field,
                                                          "evidence": proof})
                raise Failure("service gate failed: " + field + detail)
            evidence[field] = proof
        self.journal.gate(name, evidence)

    def checkpoint(self, phase):
        candidate = require_running(self.api, self.state["candidate_id"], self.state["candidate_image_id"])
        checkpoint = {
            "attempt_id": self.state["attempt_id"], "candidate_id": self.state["candidate_id"],
            "process_epoch": candidate["State"]["StartedAt"], "phase": phase,
            "model": self.plan["model"], "kind": "native-optimus-glm-response",
            "challenge": secrets.token_hex(24), "created_at": time.time(),
            "expires_at": time.time() + self.plan["proof_timeout_s"], "status": "awaiting_external_proof",
        }
        private_json(self.journal.directory / ("checkpoint-" + phase + ".json"), checkpoint, exclusive=True)
        self.journal.update(phase="WAIT_NATIVE_" + phase.upper())
        receipt_path = self.journal.directory / ("native-proof-" + phase + ".json")
        started = time.monotonic()
        deadline = min(self.end, started + self.plan["proof_timeout_s"])
        while True:
            self.guard()
            self.observe_experiment_signal(phase)
            self.note_research_deadlines(phase, started)
            document = require_running(self.api, self.state["candidate_id"], self.state["candidate_image_id"])
            if document["State"].get("StartedAt") != checkpoint["process_epoch"]:
                raise Failure("candidate restarted during native proof checkpoint")
            if receipt_path.exists():
                proof = proof_matches(read_json(receipt_path), checkpoint)
                self.journal.gate("native_" + phase, proof)
                return
            now = time.monotonic()
            if now >= deadline:
                raise ExperimentDeadline("external native Optimus proof deadline")
            time.sleep(min(self.plan["poll_interval_s"], max(0, deadline - now)))

    def planned_restart(self):
        state = self.journal.read()
        before = require_running(self.api, state["candidate_id"], state["candidate_image_id"])
        request = {"nonce": secrets.token_hex(16), "candidate_id": state["candidate_id"],
                   "epoch": before["State"]["StartedAt"]}
        if state["restart_count"] != 0:
            raise Failure("single restart already consumed")
        self.journal.update(phase="RESTART_REQUESTED", restart_request=request)
        permit_path = self.journal.directory / "restart-permit.json"
        deadline = min(self.end, time.monotonic() + 10)
        while not permit_path.exists():
            self.guard()
            require_running(self.api, state["candidate_id"], state["candidate_image_id"])
            if time.monotonic() >= deadline:
                raise Deadline("supervisor did not grant restart")
            time.sleep(self.plan["poll_interval_s"])
        permit = read_json(permit_path)
        if (permit.get("nonce") != request["nonce"] or permit.get("candidate_id") != state["candidate_id"] or
                permit.get("attempt_id") != state["attempt_id"] or permit.get("epoch") != request["epoch"] or
                time.monotonic() >= permit.get("expires_monotonic", 0)):
            raise Failure("invalid supervisor restart permit")
        self.mutate("validation_stop_candidate", lambda: self.api.stop(state["candidate_id"], seconds=self.plan["stop_timeout_s"]))
        self.mutate("validation_start_candidate", lambda: self.api.start(state["candidate_id"]))
        after = require_running(self.api, state["candidate_id"], state["candidate_image_id"])
        if after["State"].get("StartedAt") == request["epoch"]:
            raise Failure("validation restart did not change process epoch")
        self.journal.update(candidate_epoch=after["State"]["StartedAt"])
        self.wait_health()
        if time.monotonic() >= permit["expires_monotonic"]:
            raise Deadline("validation restart permit expired")
        self.journal.update(restart_done=True, phase="POST_RESTART")
        self.journal.gate("restart", {"ok": True, "before_epoch": request["epoch"],
                                     "after_epoch": after["State"]["StartedAt"]})

    def rehearsal_pause(self, point):
        """Only owned disposable fixtures can expose a deterministic fault-injection pause."""
        if self.plan["mode"] != "rehearsal" or self.plan.get("fault_pause") != point:
            return
        from .lifecycle import assert_rehearsal_original
        assert_rehearsal_original(self.original, self.plan)
        private_json(self.journal.directory / "fault-checkpoint.json", {
            "point": point, "controller_pid": os.getpid(), "at": time.monotonic(),
            "attempt_id": self.state["attempt_id"],
        }, exclusive=True)
        while True:
            self.guard()
            time.sleep(self.plan["poll_interval_s"])

    def deploy(self):
        from .docker_config import assert_candidate_parity
        if self.needs_handshake:
            handshake_end = time.monotonic() + min(10, self.plan["controller_timeout_s"])
            while True:
                try:
                    self.external_guard()
                    break
                except (Failure, OSError, ValueError):
                    if time.monotonic() >= handshake_end:
                        raise Failure("supervisor handshake did not complete") from None
                    time.sleep(min(0.05, self.plan["poll_interval_s"]))
        self.guard()
        admission = self.services.pre_mutation_admission(self.state)
        if not isinstance(admission, dict) or admission.get("ok") is not True:
            raise Failure("native latent validation unavailable before mutation")
        self.journal.update(validation_admission=admission)
        current = require_running(self.api, self.state["original_id"], self.state["original_image_id"])
        if current["Name"] != "/" + self.state["original_name"]:
            raise Failure("original name changed after preflight")
        assert_candidate_parity(self.original, current, self.state["original_image_id"])
        candidate = checked_inspect(self.api, self.state["candidate_id"], self.state["candidate_image_id"])
        if candidate.get("State", {}).get("Status") != "created" or is_running(candidate):
            raise Failure("exact preflight candidate is no longer unstarted")
        assert_candidate_parity(self.original, candidate, self.state["candidate_image_id"],
                                self.state["plan"].get("environment_additions"),
                                calibrated_binding_from_state(self.state))
        # Full profile-pin and artifact re-verification on the Docker host,
        # before the first production mutation touches the original.
        verify_calibrated_source(self.state)
        self.rehearsal_pause("before_mutation")
        self.mutate("rename_original", lambda: self.api.rename(self.state["original_id"], self.state["preserved_name"]),
                    mutation_intent=True, phase="MUTATING")
        self.rehearsal_pause("after_rename")
        self.mutate("stop_original", lambda: self.api.stop(self.state["original_id"], seconds=self.plan["stop_timeout_s"]))
        stopped = checked_inspect(self.api, self.state["original_id"], self.state["original_image_id"])
        if stopped.get("State", {}).get("Running") or stopped.get("State", {}).get("Restarting"):
            raise Failure("original did not stop; candidate must remain unstarted")
        self.journal.update(original_stopped=True)
        self.mutate("rename_candidate", lambda: self.api.rename(self.state["candidate_id"], self.state["original_name"]))
        self.mutate("start_candidate", lambda: self.api.start(self.state["candidate_id"]))
        candidate = require_running(self.api, self.state["candidate_id"], self.state["candidate_image_id"])
        if candidate.get("RestartCount", 0) != 0:
            raise Failure("candidate restarted during initial startup")
        self.journal.update(candidate_started=True, candidate_epoch=candidate["State"]["StartedAt"], phase="STARTING")
        self.rehearsal_pause("after_start")
        self.service_gate(READINESS_GATE)
        if self.persist_mode():
            # A readiness-verified candidate is no longer bounded by the deployment
            # deadline: from here the operator maintenance window bounds the attempt.
            self.end = max(self.end, self.window_end)
        self.checkpoint("initial")
        validator = self.services.validator(self.state)
        self.journal.gate("live_before", self.check(validator.before_restart, self.plan["validation_timeout_s"]))
        self.planned_restart()
        self.journal.gate("live_after", self.check(validator.after_restart, self.plan["validation_timeout_s"]))
        self.service_gate("service_final")
        self.checkpoint("final")
        self.guard()
        self.journal.update(phase="GATES_COMPLETE")
        return 0

    def run(self):
        with FileLock(self.journal.directory / "controller.lock"):
            if self.state["phase"] != "AUTHORIZED" or not self.state["authorization_consumed"]:
                raise Failure("worker has no consumed unused attempt authorization")
            try:
                return self.deploy()
            except BaseException as error:
                if self.persist_applies(error):
                    try:
                        return self.persist_candidate(error)
                    except BaseException:
                        error = sys.exc_info()[1]
                with contextlib.suppress(Exception):
                    private_json(self.journal.directory / "controller.json", {
                        "attempt_id": self.state["attempt_id"], "pid": os.getpid(),
                        "heartbeat": time.monotonic(), "status": "recovering", "error": type(error).__name__,
                    })
                    self.state = self.journal.read()
                result = recover(self.api, self.journal, services=self.services, state=self.state,
                                 original=self.original, reason="controller_" + type(error).__name__)
                return 1 if result.get("verified") else 2


def restore(api, journal, *, services=None, reason="operator_manual_restore"):
    """Explicit operator rollback of one attempt, including a persisted candidate.

    This is the same verified restoration the automatic paths use; it never
    deletes a container or an image and it never reopens the attempt.
    """
    result = recover(api, journal, services=services, reason=reason)
    with contextlib.suppress(Exception):
        record = journal.read().get("persistence")
        if isinstance(record, dict):
            journal.update(persistence=dict(record, manual_restore=result.get("result"),
                                            manual_restore_verified=bool(result.get("verified"))))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="preflight",
                        choices=("preflight", "swap", "status", "worker", "rehearsal", "restore"))
    parser.add_argument("--plan")
    parser.add_argument("--attempt-dir")
    parser.add_argument("--approval")
    parser.add_argument("--fixture-base-image")
    parser.add_argument("--output-dir")
    parser.add_argument("--calibrated-input")
    args = parser.parse_args(argv)
    def interrupted(*_):
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.command == "rehearsal":
            from .rehearsal import load_calibrated_input, run_rehearsals
            if not args.fixture_base_image or not args.output_dir:
                parser.error("rehearsal requires --fixture-base-image and new --output-dir")
            calibrated = (load_calibrated_input(args.calibrated_input)
                          if args.calibrated_input is not None else None)
            result = run_rehearsals(DockerAPI(), args.fixture_base_image, args.output_dir,
                                    calibrated_input=calibrated)
            print(json.dumps(result))
            return 0 if result.get("result") == "PASS" else 1
        if not args.attempt_dir:
            parser.error("--attempt-dir is required; no deployment action was taken")
        if args.command == "preflight":
            if not args.plan:
                parser.error("preflight requires --plan")
            plan = read_json(args.plan)
            journal = prepare(DockerAPI(plan.get("docker_socket", "/var/run/docker.sock")),
                              plan, args.attempt_dir)
            state = journal.read()
            print(json.dumps({"result": "PREPARED_NOT_DEPLOYED", "attempt_id": state["attempt_id"],
                              "candidate_id": state["candidate_id"]}))
            return 0
        journal = Journal(args.attempt_dir)
        state = journal.read()
        if args.command == "status":
            print(json.dumps({key: state.get(key) for key in ("attempt_id", "phase", "sequence",
                             "original_id", "candidate_id", "operation", "recovery")}))
            return 0
        if args.command == "restore":
            api = DockerAPI(state["plan"].get("docker_socket", "/var/run/docker.sock"))
            result = restore(api, journal)
            print(json.dumps({key: result.get(key) for key in
                              ("result", "reason", "verified", "service_result", "cleanup_complete")}))
            return 0 if result.get("verified") else 2
        if args.command == "swap":
            if not args.approval:
                parser.error("swap requires explicit --approval plan")
            from .supervisor import launch
            print(json.dumps(launch(journal, args.approval)))
            return 0
        api = DockerAPI(state["plan"].get("docker_socket", "/var/run/docker.sock"))
        return Controller(api, journal).run()
    except BaseException as error:
        if isinstance(error, SystemExit):
            return int(error.code or 0)
        print(json.dumps({"result": "FAILED", "error": type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
