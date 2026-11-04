"""Independent Linux controller guardian with its own recovery deadline.

The controller is a separate session/process. This process owns the original-ID
lock and never treats exit code zero alone as deployment success.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import secrets
from pathlib import Path
import signal
import subprocess
import sys
import time

from .docker_api import DockerAPI
from .lifecycle import (Failure, FileLock, Journal, Services, calibrated_binding_from_state,
                        checked_inspect, consume_authorization, is_running, private_json,
                        process_identity, read_json, recover, reconcile_helpers, helpers_settled,
                        rotate_helper_admission, reconcile_helper_operations, helper_operations_settled,
                        PERSIST_CANDIDATE_ON_SUCCESS, PERSISTED_CANDIDATE, PERSISTED_PHASE,
                        PERSISTENCE_GATE, READINESS_GATE, ROLLBACK_ON_SUCCESS, ROLLED_BACK_ON_SUCCESS,
                        preserved_original_record, record_persistence, recovery_command)

REQUIRED_GATES = {"service_initial", "native_initial", "live_before", "restart",
                  "live_after", "service_final", "native_final"}
# Persistence needs the full readiness gate, not the experimental protocol.
PERSISTENCE_GATES = {READINESS_GATE, PERSISTENCE_GATE}


def complete(state):
    gates = state.get("gates", {})
    return (state.get("phase") == "GATES_COMPLETE" and state.get("authorization_consumed") is True and
            state.get("restart_count") == 1 and REQUIRED_GATES <= gates.keys() and
            all(gates[name].get("ok") is True for name in REQUIRED_GATES))


def persisted(state):
    """A fully verified candidate whose persistence decision is already durable.

    Independent acceptance gate: the durable record and the persistence gates
    are necessary but not sufficient. Retention is accepted only when the
    complete REQUIRED_GATES native validation chain is present and ok, so a
    forged or partial PERSISTED_CANDIDATE state with a missing, failed or
    still-pending proof link is rejected and restored instead.
    """
    record = state.get("persistence")
    record = record if isinstance(record, dict) else {}
    original = record.get("original") if isinstance(record.get("original"), dict) else {}
    gates = state.get("gates", {})
    return (state.get("phase") == PERSISTED_PHASE and state.get("authorization_consumed") is True and
            state.get("persistence_decision") == PERSISTED_CANDIDATE and
            record.get("decision") == PERSISTED_CANDIDATE and
            record.get("policy") == PERSIST_CANDIDATE_ON_SUCCESS and
            record.get("candidate_id") == state.get("candidate_id") and
            record.get("candidate_image_id") == state.get("candidate_image_id") and
            original.get("container_id") == state.get("original_id") and
            original.get("image_id") == state.get("original_image_id") and
            isinstance(record.get("recovery_command"), str) and bool(record["recovery_command"]) and
            PERSISTENCE_GATES <= gates.keys() and
            all(gates[name].get("ok") is True for name in PERSISTENCE_GATES) and
            REQUIRED_GATES <= gates.keys() and
            all(isinstance(gates[name], dict) and gates[name].get("ok") is True
                for name in REQUIRED_GATES))


def claim_original_fence(journal, fence_root, identity):
    state = journal.read()
    root = Path(fence_root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink():
        raise Failure("unsafe original fence directory")
    if os.name == "posix":
        info = root.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise Failure("original fence directory must be private and owned")
    path = root / (state["original_id"] + ".json")
    private_json(path, {"schema": 1, "attempt_id": state["attempt_id"],
                       "original_id": state["original_id"], "attempt_dir": str(journal.directory.resolve()),
                       "guardian_identity": identity, "claimed_at": time.time(),
                       "automatic_retirement": False}, exclusive=True)
    return path


def restart_permitted(journal, state, now):
    path = journal.directory / "restart-permit.json"
    if not path.exists() or state.get("restart_done"):
        return False
    permit = read_json(path)
    return (permit.get("attempt_id") == state["attempt_id"] and
            permit.get("candidate_id") == state["candidate_id"] and
            permit.get("nonce") == state.get("restart_request", {}).get("nonce") and
            now < permit.get("expires_monotonic", 0))


def decide(journal, state, *, child_code, now, begun, heartbeat, api):
    """One monitor tick, also used by offline fault-injection tests."""
    if child_code is not None:
        if child_code == 0 and complete(state):
            return "complete"
        if child_code == 0 and persisted(state):
            return "persist"
        return "controller_exit"
    if now - begun >= state["plan"]["deployment_timeout_s"]:
        return "deployment_deadline"
    if heartbeat:
        if (heartbeat.get("attempt_id") != state["attempt_id"] or
                heartbeat.get("pid") != state.get("controller_pid")):
            return "controller_identity"
        if heartbeat.get("status") == "recovering":
            if now - heartbeat.get("heartbeat", 0) >= state["plan"]["rollback_timeout_s"]:
                return "controller_recovery_deadline"
            return None
        if now - heartbeat.get("heartbeat", 0) >= state["plan"]["controller_timeout_s"]:
            return "controller_unresponsive"
    elif now - begun >= min(10, state["plan"]["controller_timeout_s"]):
        return "controller_no_handshake"
    if state.get("candidate_started"):
        candidate = checked_inspect(api, state["candidate_id"], state["candidate_image_id"])
        permitted = restart_permitted(journal, state, now)
        if candidate.get("RestartCount", 0) != 0:
            return "candidate_unplanned_restart"
        if not permitted and not is_running(candidate):
            return "candidate_exit"
        if not permitted and candidate.get("State", {}).get("StartedAt") != state.get("candidate_epoch"):
            return "candidate_unplanned_restart"
    if state.get("original_stopped") and state.get("phase") not in ("RESTORED", "NOOP", "RESTORE_FAILED"):
        original = checked_inspect(api, state["original_id"], state["original_image_id"])
        if is_running(original):
            return "preserved_original_restarted"
    return None


class Supervisor:
    def __init__(self, api, journal, *, services=None, process_factory=None,
                 lock_root="/run/lock/occ3-v3", fence_root=None):
        self.api, self.journal = api, journal
        self.state, self.original = journal.read(), journal.original()
        self.services = services or Services(self.original, self.state["plan"], api=api, journal=journal)
        self.process_factory = process_factory or self._spawn
        self.child = None
        self.lock_root = Path(lock_root)
        # Fences default to the attempt directory so a non-root deployment
        # needs no system path or one-time sudo mkdir/chown (the previous
        # /var/lib/occ3-v3/attempt-fences default did). Cross-attempt
        # serialization stays on the host-wide original-ID lock above;
        # deployments wanting a shared host-wide fence namespace for
        # cross-directory refusals pass fence_root explicitly.
        self.fence_root = (Path(fence_root) if fence_root is not None
                           else journal.directory / "fences")
        self.nonce = secrets.token_hex(16)
        self.identity = process_identity(os.getpid())
        if self.identity is None:
            raise Failure("independent supervision requires Linux process identity")

    def _spawn(self):
        log = open(self.journal.directory / "controller-process.log", "ab", buffering=0)
        try:
            return subprocess.Popen([sys.executable, "-m", "deploy.v3.controller", "worker",
                                     "--attempt-dir", str(self.journal.directory)],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    start_new_session=True, close_fds=True)
        finally:
            log.close()

    def publish(self, status="watching"):
        private_json(self.journal.directory / "supervisor.json", {
            "attempt_id": self.state["attempt_id"], "pid": os.getpid(),
            "identity": self.identity, "heartbeat": time.monotonic(), "status": status,
            "controller_pid": self.child.pid if self.child else None, "nonce": self.nonce,
        })

    def permit_restart(self, state):
        request = state.get("restart_request")
        if not request or state.get("restart_count"):
            return
        if (state.get("phase") != "RESTART_REQUESTED" or
                state.get("gates", {}).get("live_before", {}).get("ok") is not True or
                request.get("candidate_id") != state["candidate_id"] or
                request.get("epoch") != state.get("candidate_epoch")):
            raise Failure("invalid restart request")
        duration = state["plan"]["startup_timeout_s"] + state["plan"]["stop_timeout_s"] + 15
        self.journal.update(restart_count=1)
        private_json(self.journal.directory / "restart-permit.json", {
            "attempt_id": state["attempt_id"], "candidate_id": state["candidate_id"],
            "nonce": request["nonce"], "epoch": request["epoch"],
            "expires_monotonic": time.monotonic() + duration,
        }, exclusive=True)

    def terminate_controller(self):
        if self.child and self.child.poll() is None:
            self.child.kill()
            try:
                self.child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                raise Failure("controller could not be terminated") from None

    def commit(self, reason, begun):
        """Verified commit point: persist the candidate or restore production.

        Both policies first prove the exact completion state (candidate running
        under the original name on the candidate image, original stopped under
        the preserved name, unchanged launch parity) and settle secret helper
        cleanup. The persistence policy only decides what happens next.
        """
        policy = self.state["plan"].get("persistence_policy")
        if reason == "persist" and policy != PERSIST_CANDIDATE_ON_SUCCESS:
            raise Failure("persistence commit without the persist policy")
        candidate = checked_inspect(self.api, self.state["candidate_id"], self.state["candidate_image_id"])
        original = checked_inspect(self.api, self.state["original_id"], self.state["original_image_id"])
        if (not is_running(candidate) or is_running(original) or candidate.get("RestartCount", 0) != 0 or
                candidate["State"].get("StartedAt") != self.state.get("candidate_epoch") or
                candidate.get("Name") != "/" + self.state["original_name"] or
                original.get("Name") != "/" + self.state["preserved_name"]):
            raise Failure("completion state changed")
        from .docker_config import assert_candidate_parity
        assert_candidate_parity(self.original, candidate, self.state["candidate_image_id"],
                                self.state["plan"].get("environment_additions"),
                                calibrated_binding_from_state(self.state))
        assert_candidate_parity(self.original, original, self.state["original_image_id"])
        rotate_helper_admission(self.journal, "closed")
        cleanup = reconcile_helpers(self.api, self.journal)
        operations = reconcile_helper_operations(self.journal)
        if not helpers_settled(cleanup) or not helper_operations_settled(operations):
            raise Failure("secret helper cleanup or operation pending at completion")
        gate = PERSISTENCE_GATE if reason == "persist" else "service_final"
        gates = sorted(PERSISTENCE_GATES if reason == "persist" else REQUIRED_GATES)
        if policy == PERSIST_CANDIDATE_ON_SUCCESS:
            record_persistence(self.journal, self.state, candidate, self.original,
                               decision=PERSISTED_CANDIDATE, reason="supervisor_" + reason, gate=gate)
            self.journal.update(phase="COMPLETE", cleanup_complete=True, helper_cleanup=cleanup,
                                helper_operations=operations, helper_operations_settled=True)
            # The durable commit is authoritative; secondary log failure does not undo it.
            with contextlib.suppress(Exception):
                self.publish("complete")
                private_json(self.journal.directory / "supervisor-result.json", {
                    "result": "COMPLETE", "persistence_decision": PERSISTED_CANDIDATE,
                    "verified_gates": gates,
                    "preserved_original": preserved_original_record(self.state["plan"], self.original),
                    "recovery_command": recovery_command(self.journal.directory),
                    "cleanup_complete": True, "helper_cleanup": cleanup,
                    "helper_operations": operations, "helper_operations_settled": True,
                    "duration_s": time.monotonic() - begun,
                })
            return 0
        # rollback_on_success: the attempt is one time-boxed experiment, so the
        # verified original returns through the same restoration the failure
        # paths use. The decision is durable before any mutation of that path.
        record_persistence(self.journal, self.state, candidate, self.original,
                           decision=ROLLED_BACK_ON_SUCCESS, reason="supervisor_" + reason, gate=gate)
        detected = time.monotonic()
        result = recover(self.api, self.journal, services=self.services, state=self.state,
                         original=self.original, reason="persistence_policy_" + ROLLBACK_ON_SUCCESS)
        result["detection_to_verified_s"] = time.monotonic() - detected
        result["attempt_duration_s"] = time.monotonic() - begun
        with contextlib.suppress(Exception):
            private_json(self.journal.directory / "supervisor-result.json", result)
            self.publish("rolled_back_on_success" if result.get("verified") else "restore_failed")
            record = self.journal.read().get("persistence")
            if isinstance(record, dict):
                self.journal.update(persistence=dict(record, recovery_result=result.get("result"),
                                                     recovery_verified=bool(result.get("verified"))))
        return 0 if result.get("verified") else 2

    def run(self, approval):
        begun, reason = time.monotonic(), "supervisor_failure"
        # A host-wide namespace prevents parallel attempts in different directories.
        lock_root = self.lock_root
        lock_root.mkdir(mode=0o700, exist_ok=True)
        if lock_root.is_symlink():
            raise Failure("unsafe original lock directory")
        if os.name == "posix":
            info = lock_root.stat()
            if info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise Failure("original lock directory must be private and owned")
        with FileLock(lock_root / (self.state["original_id"] + ".lock")), FileLock(self.journal.directory / "supervisor.lock"):
            # Admission failures never enter recovery or touch a stale attempt.
            if (self.fence_root / (self.state["original_id"] + ".json")).exists():
                raise Failure("original is fenced by a prior attempt; operator review required")
            admission = self.services.pre_mutation_admission(self.state)
            if not isinstance(admission, dict) or admission.get("ok") is not True:
                raise Failure("native latent validation unavailable before authorization")
            consume_authorization(self.journal, approval)
            claim_original_fence(self.journal, self.fence_root, self.identity)
            self.journal.update(supervision_nonce=self.nonce, validation_admission=admission)
            self.state = self.journal.read()
            try:
                self.publish()
                self.child = self.process_factory()
                child_identity = process_identity(self.child.pid)
                if self.child.poll() is None and child_identity is None:
                    raise Failure("controller process-start identity unavailable")
                self.journal.update(controller_pid=self.child.pid, controller_identity=child_identity)
                self.publish()
                while True:
                    self.publish()
                    self.state = self.journal.read()
                    self.permit_restart(self.state)
                    heartbeat_path = self.journal.directory / "controller.json"
                    heartbeat = read_json(heartbeat_path) if heartbeat_path.exists() else None
                    reason = decide(self.journal, self.state, child_code=self.child.poll(),
                                    now=time.monotonic(), begun=begun, heartbeat=heartbeat, api=self.api)
                    if reason:
                        break
                    time.sleep(self.state["plan"]["poll_interval_s"])
                if reason in ("complete", "persist"):
                    return self.commit(reason, begun)
            except BaseException as error:
                reason = "supervisor_" + type(error).__name__
            detected = time.monotonic()
            with contextlib.suppress(Exception):
                private_json(self.journal.directory / "recover-request.json", {"reason": reason, "detected_monotonic": detected})
            # Stop the writer before rollback. No name-based kill or process-prefix cleanup.
            try:
                self.terminate_controller()
            except BaseException:
                with contextlib.suppress(Exception):
                    self.publish("controller_termination_failed")
                return 2
            with contextlib.suppress(Exception):
                self.state = self.journal.read()
            with contextlib.suppress(Exception):
                self.publish("recovering")
            result = recover(self.api, self.journal, services=self.services,
                             state=self.state, original=self.original, reason=reason)
            result["detection_to_verified_s"] = time.monotonic() - detected
            result["attempt_duration_s"] = time.monotonic() - begun
            with contextlib.suppress(Exception):
                private_json(self.journal.directory / "supervisor-result.json", result)
                self.publish("recovered" if result.get("verified") else "restore_failed")
            return 1 if result.get("verified") else 2


def launch(journal, approval_path):
    if os.name != "posix":
        raise Failure("Docker supervision requires Linux")
    if journal.read()["phase"] != "PREPARED":
        raise Failure("only a prepared unused attempt can be launched")
    # This reservation is never removed, including if spawn fails.
    private_json(journal.directory / "supervisor-launch.json", {"at": time.time()}, exclusive=True)
    log = open(journal.directory / "supervisor-process.log", "ab", buffering=0)
    try:
        child = subprocess.Popen([sys.executable, "-m", "deploy.v3.supervisor",
                                  "--attempt-dir", str(journal.directory),
                                  "--approval", str(Path(approval_path).resolve())],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 close_fds=True, start_new_session=True)
    finally:
        log.close()
    return {"supervisor_pid": child.pid, "attempt_id": journal.read()["attempt_id"],
            "status": "launched_not_verified"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--approval", required=True)
    args = parser.parse_args(argv)
    def interrupted(*_):
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        journal = Journal(args.attempt_dir)
        state = journal.read()
        api = DockerAPI(state["plan"].get("docker_socket", "/var/run/docker.sock"), timeout=2)
        return Supervisor(api, journal).run(read_json(args.approval))
    except BaseException as error:
        print(json_error(error), file=sys.stderr)
        return 2


def json_error(error):
    import json
    return json.dumps({"result": "FAILED", "error": type(error).__name__})


if __name__ == "__main__":
    raise SystemExit(main())
