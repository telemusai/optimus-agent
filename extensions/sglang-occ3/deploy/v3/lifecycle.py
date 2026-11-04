"""One-attempt state machine and ID-only recovery shared by production/rehearsals.

No launch argument, environment value, credential or HTTP response body is logged.
The original inspect is retained privately; the original container is never updated.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shlex
import stat
import sys
import threading
import time
from urllib.parse import urlsplit

from .docker_api import DockerAPI, DockerError, container_id, image_id


class Failure(RuntimeError):
    pass


class Deadline(Failure):
    pass


class RunningUnready(Deadline):
    pass


class ExperimentDeadline(Deadline):
    """A research deadline elapsed. It is not, by itself, a service-health failure."""


class MaintenanceWindowClosed(Deadline):
    """The operator-scheduled maintenance window closed."""


TERMINAL = {"COMPLETE", "RESTORED", "NOOP", "RESTORE_FAILED", "PREFLIGHT_FAILED"}
REHEARSAL_LABEL = "occ3.v3.rehearsal"
IMMUTABLE = {"attempt_id", "original_id", "original_image_id", "candidate_image_id",
             "original_name", "preserved_name", "candidate_name", "failed_name",
             "plan", "created_at", "schema",
             "calibration_source", "calibration_destination", "calibration_basename",
             "calibration_source_identity"}
DEFAULTS = {"startup_timeout_s": 1200, "deployment_timeout_s": 3600,
            "rollback_timeout_s": 900, "proof_timeout_s": 300,
            "controller_timeout_s": 45, "supervisor_lease_s": 5,
            "poll_interval_s": 0.5, "request_timeout_s": 120,
            "probe_timeout_s": 120, "stop_timeout_s": 30,
            "validation_timeout_s": 900,
            "source_capture_timeout_s": 900, "receiver_inference_timeout_s": 300,
            "maintenance_window_s": 3600}
# One named deadline per plan field, never one shared budget. Each named deadline
# has exactly one documented reaction (see deploy/v3/README.md):
#   model request, deployment health, controller deployment -> service failure (rollback)
#   native source capture, receiver inference                -> recorded research stage
#   experimental proof                                      -> persistence policy decision
#   operator maintenance window                             -> persistence policy decision
TIMEOUT_FIELDS = {"model_request_timeout_s": "request_timeout_s",
                  "deployment_health_timeout_s": "startup_timeout_s",
                  "controller_deployment_deadline_s": "deployment_timeout_s",
                  "native_source_capture_deadline_s": "source_capture_timeout_s",
                  "receiver_inference_deadline_s": "receiver_inference_timeout_s",
                  "experimental_proof_deadline_s": "proof_timeout_s",
                  "operator_maintenance_window_s": "maintenance_window_s",
                  "rollback_deadline_s": "rollback_timeout_s"}
RESEARCH_STAGE_DEADLINES = {"source capture deadline": "source_capture_timeout_s",
                            "receiver inference deadline": "receiver_inference_timeout_s"}
# Candidate persistence policy: what happens to the verified candidate once the
# attempt reaches its verified commit point.
ROLLBACK_ON_SUCCESS = "rollback_on_success"
PERSIST_CANDIDATE_ON_SUCCESS = "persist_candidate_on_success"
PERSISTENCE_POLICIES = (ROLLBACK_ON_SUCCESS, PERSIST_CANDIDATE_ON_SUCCESS)
PERSISTED_CANDIDATE = "persisted_candidate_on_success"
ROLLED_BACK_ON_SUCCESS = "rolled_back_on_success"
PERSISTED_PHASE = "CANDIDATE_PERSISTED"
READINESS_GATE = "service_initial"
PERSISTENCE_GATE = "service_persisted"
# Experiment signals are a fixed vocabulary split in two disjoint classes. Only
# the service-health class may ever cause a restart or a rollback; a semantic
# experiment result is evidence and nothing else.
EXPERIMENT_SIGNAL_SCHEMA = "occ3.experiment.signal.v1"
EXPERIMENT_SEMANTIC = "semantic_experiment"
EXPERIMENT_SERVICE_HEALTH = "service_health"
EXPERIMENT_SEMANTIC_CODES = frozenset({
    "receiver_answer_mismatch", "receiver_partial_answer", "receiver_fallback_used",
    "receiver_fact_mismatch", "source_capture_absent", "proof_not_delivered"})
EXPERIMENT_SERVICE_HEALTH_CODES = frozenset({
    "receiver_transport_error", "receiver_timeout", "receiver_http_failure",
    "source_route_unavailable", "source_capture_transport_error", "runtime_identity_lost"})
_PRESERVED_RECORD_FIELDS = frozenset({"container_id", "image_id", "name"})
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def private_json(path, document, *, exclusive=False):
    path = Path(path)
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    temp = path if exclusive else path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if not exclusive:
            os.replace(temp, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if not exclusive and temp.exists():
            temp.unlink()


def read_json(path, limit=4 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink():
        raise Failure("symlink state file refused")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise Failure("state file too large")
    return json.loads(data)


class FileLock:
    """Advisory lock with a bounded wait; independent handles for each writer."""
    def __init__(self, path, wait=0):
        self.path, self.wait, self.handle = Path(path), wait, None

    def __enter__(self):
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self.handle = os.fdopen(fd, "r+b", buffering=0)
        if os.fstat(fd).st_size == 0:
            self.handle.write(b"0")
        end = time.monotonic() + self.wait
        while True:
            try:
                self.handle.seek(0)
                if os.name == "posix":
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    import msvcrt
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return self
            except OSError:
                if time.monotonic() >= end:
                    self.handle.close()
                    self.handle = None
                    raise Failure("attempt lock busy") from None
                time.sleep(min(0.02, max(0, end - time.monotonic())))

    def __exit__(self, *_):
        if self.handle is not None:
            if os.name == "posix":
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()
            self.handle = None


def validate_plan(plan):
    plan = dict(plan)
    attempt = plan.get("attempt_id", "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,47}", attempt):
        raise Failure("invalid attempt ID")
    container_id(plan.get("original_id"))
    image_id(plan.get("original_image_id"))
    image_id(plan.get("candidate_image_id"))
    if plan["candidate_image_id"] == plan["original_image_id"]:
        raise Failure("candidate must be a distinct pinned image")
    if plan.get("mode", "production") not in ("production", "rehearsal"):
        raise Failure("invalid attempt mode")
    plan.setdefault("mode", "production")
    for field in ("base_url", "ingress_base_url"):
        parsed = urlsplit(plan.get(field, ""))
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise Failure("invalid service URL")
    if not isinstance(plan.get("model"), str) or not plan["model"]:
        raise Failure("explicit served model required")
    for field, default in DEFAULTS.items():
        value = plan.setdefault(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise Failure("invalid deadline option")
    validate_persistence(plan)
    from .docker_config import environment_addition_entries, calibrated_capability
    try:
        environment_addition_entries(plan.setdefault("environment_additions", []))
    except ValueError:
        raise Failure("invalid environment additions") from None
    try:
        calibrated_capability(plan)
    except ValueError:
        raise Failure("invalid calibrated artifact capability") from None
    return plan


def validate_persistence(plan):
    """The persistence policy, and the preserved original it depends on.

    Persisting a candidate is only admissible when the plan names the preserved
    production container explicitly. A rollback-policy plan may omit the record,
    but a malformed or inconsistent record is refused under either policy.
    """
    policy = plan.setdefault("persistence_policy", ROLLBACK_ON_SUCCESS)
    if policy not in PERSISTENCE_POLICIES:
        raise Failure("invalid candidate persistence policy")
    record = plan.get("preserved_original")
    if record is None:
        if policy == PERSIST_CANDIDATE_ON_SUCCESS:
            raise Failure("persist policy requires a preserved original container record")
        return policy
    if not isinstance(record, dict) or set(record) != _PRESERVED_RECORD_FIELDS:
        raise Failure("invalid preserved original container record")
    container_id(record["container_id"])
    image_id(record["image_id"])
    if not isinstance(record["name"], str) or not _CONTAINER_NAME.fullmatch(record["name"]):
        raise Failure("invalid preserved original container name")
    if record["container_id"] != plan["original_id"] or record["image_id"] != plan["original_image_id"]:
        raise Failure("preserved original container record differs from the attempt original")
    return policy


def timeout_policy(plan):
    """Distinct named deadlines as recorded in the journal; one plan field each."""
    record = {name: plan[field] for name, field in TIMEOUT_FIELDS.items()}
    record.update({"candidate_persistence_policy": plan["persistence_policy"],
                   "research_stage_deadlines": dict(RESEARCH_STAGE_DEADLINES)})
    return record


def calibrated_binding_from_state(state):
    """The journaled calibrated binding, or None for plans without the capability.

    A nonempty capability without its set-once journaled derived evidence is an
    inconsistent attempt (crash between journal creation and the verified
    prepare): refuse rather than re-derive from an unverified source.
    """
    if state["plan"].get("calibrated_artifact") is None:
        return None
    fields = ("calibration_source", "calibration_destination", "calibration_basename")
    if any(state.get(name) is None for name in fields):
        raise Failure("calibrated capability missing journaled binding")
    return {"source": state["calibration_source"],
            "destination": state["calibration_destination"],
            "basename": state["calibration_basename"]}


def verify_calibrated_source(state):
    """Full re-verification of profile pin and artifact bytes before mutation.

    Runs on the Docker host immediately before the first production mutation:
    the staged profile is re-read and digest-pinned, the artifact is re-read
    with the full fd-read verification, and the file identity must still match
    the one journaled at prepare. Docker re-resolves the source path at mount
    time, so this narrows (never eliminates) the path race; the in-container
    runtime digest check remains the standing content gate.
    """
    from .docker_config import (calibrated_capability, verify_calibrated_artifact,
                                verify_calibrated_profile)
    capability = calibrated_capability(state["plan"])
    if capability is None:
        return None
    binding = calibrated_binding_from_state(state)
    try:
        verified = verify_calibrated_profile(capability)
        if verified["calibration_path"] != binding["destination"]:
            raise Failure("calibrated profile path changed after prepare")
        identity = verify_calibrated_artifact(binding["source"], capability["sha256"],
                                              capability["size"])
    except ValueError as error:
        raise Failure(str(error)) from None
    if list(identity) != list(state.get("calibration_source_identity") or ()):
        raise Failure("calibrated source identity changed after prepare")
    return binding


def preserved_configuration_digest(original):
    """Digest of the preserved launch configuration with the credential removed."""
    from .docker_config import extract_api_key
    key = extract_api_key(original)
    config = dict(original.get("Config") or {})
    for field in ("Entrypoint", "Cmd", "Env"):
        config[field] = ["<credential>" if value == key else value
                         for value in config.get(field) or []]
    projection = {"config": config, "host": original.get("HostConfig"),
                  "mounts": original.get("Mounts"),
                  "networks": sorted((original.get("NetworkSettings") or {}).get("Networks") or {})}
    return hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preserved_original_record(plan, original):
    """Non-secret identity of the production container kept for recovery."""
    record = {"container_id": container_id(original["Id"]), "image_id": image_id(original["Image"]),
              "name": original["Name"].lstrip("/"),
              "configuration_sha256": preserved_configuration_digest(original)}
    claimed = plan.get("preserved_original")
    if claimed is not None and (claimed["container_id"] != record["container_id"] or
                                claimed["image_id"] != record["image_id"] or
                                claimed["name"] != record["name"]):
        raise Failure("preserved original container record differs from the inspected original")
    return record


def recovery_command(directory):
    """Explicit operator recovery path for one attempt; contains no credential."""
    return (sys.executable + " -m deploy.v3.controller restore --attempt-dir "
            + shlex.quote(str(Path(directory).resolve())))


def record_persistence(journal, state, candidate, original, *, decision, reason, gate=None, extra=None):
    """Durable persistence decision with candidate identity and the preserved original."""
    record = {"schema": 1, "decision": decision, "policy": state["plan"]["persistence_policy"],
              "reason": reason, "attempt_id": state["attempt_id"],
              "candidate_id": state["candidate_id"], "candidate_image_id": state["candidate_image_id"],
              "candidate_name": (candidate.get("Name") or "").lstrip("/"),
              "candidate_epoch": (candidate.get("State") or {}).get("StartedAt"),
              "candidate_restart_count": candidate.get("RestartCount", 0),
              "original": preserved_original_record(state["plan"], original),
              "readiness_gate": gate,
              "recovery_command": recovery_command(journal.directory),
              "recorded_at": time.time()}
    if extra:
        record.update(extra)
    journal.update(persistence=record, persistence_decision=decision)
    return record


def classify_experiment_signal(signal):
    """Split one fixed-vocabulary experiment signal into its only two classes.

    An unrecognized code is kept in the non-fatal class and flagged, so no
    semantic result can be escalated into a restart or a rollback.
    """
    if not isinstance(signal, dict) or signal.get("schema") != EXPERIMENT_SIGNAL_SCHEMA:
        raise Failure("experiment signal schema mismatch")
    if signal.get("code") in EXPERIMENT_SERVICE_HEALTH_CODES:
        return EXPERIMENT_SERVICE_HEALTH
    return EXPERIMENT_SEMANTIC


def record_experiment_signal(journal, phase, signal, classification):
    """Journal one experiment signal. This function never mutates the service."""
    signals = dict(journal.read().get("experiment_signals") or {})
    entry = {"phase": phase, "code": signal.get("code"), "classification": classification,
             "code_known": signal.get("code") in EXPERIMENT_SEMANTIC_CODES | EXPERIMENT_SERVICE_HEALTH_CODES,
             "rollback_eligible": classification == EXPERIMENT_SERVICE_HEALTH,
             "recorded_at": time.time()}
    signals[phase] = entry
    journal.update(experiment_signals=signals)
    return entry


def assert_rehearsal_original(original, plan):
    owner = plan.get("rehearsal_owner", "")
    expected = "occ3-v3-rehearsal-" + owner + "-original"
    host = original.get("HostConfig", {})
    if (not re.fullmatch(r"[a-f0-9]{16}", owner) or original.get("Name") != "/" + expected or
            original.get("Config", {}).get("Labels", {}).get(REHEARSAL_LABEL) != owner or
            original.get("Mounts") or host.get("Binds") or host.get("Mounts") or
            host.get("DeviceRequests") or host.get("Devices") or host.get("Privileged") or
            host.get("Runtime") not in (None, "", "runc")):
        raise Failure("not an owned CPU-only disposable original")


class Journal:
    def __init__(self, directory):
        self.directory = Path(directory)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise Failure("invalid attempt directory")
        if os.name == "posix":
            info = self.directory.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise Failure("attempt directory must be private and owned")

    @classmethod
    def create(cls, directory, plan, original, *, calibrated=None):
        directory = Path(directory)
        # Existing directories, even empty ones, are stale attempts, not resumable deployments.
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        journal = cls(directory)
        name = original["Name"].lstrip("/")
        prefix = ("occ3-v3-rehearsal-" + plan["rehearsal_owner"] if plan["mode"] == "rehearsal"
                  else "occ3-v3-" + plan["attempt_id"])
        private_json(directory / "original.json", original, exclusive=True)
        state = {
            "schema": 1, "attempt_id": plan["attempt_id"], "plan": plan,
            "original_id": container_id(original["Id"]),
            "original_image_id": image_id(original["Image"]),
            "candidate_image_id": plan["candidate_image_id"], "candidate_id": None,
            "original_name": name, "candidate_name": prefix + "-candidate",
            "preserved_name": prefix + "-preserved", "failed_name": prefix + "-failed",
            "created_at": time.time(), "phase": "PREFLIGHT", "sequence": 0,
            "authorization_consumed": False, "mutation_intent": False,
            "restart_count": 0, "gates": {}, "history": [],
        }
        if calibrated is not None:
            # Immutable attempt evidence seeded atomically with the journal:
            # the derived calibrated binding and the verified source identity
            # exist from creation, so no later update can change or omit them.
            state.update(calibration_source=calibrated["source"],
                        calibration_destination=calibrated["destination"],
                        calibration_basename=calibrated["basename"],
                        calibration_source_identity=list(calibrated["identity"]))
        private_json(directory / "journal.json", state, exclusive=True)
        return journal

    def read(self):
        return read_json(self.directory / "journal.json")

    def original(self):
        return read_json(self.directory / "original.json")

    def update(self, **changes):
        with FileLock(self.directory / "journal.lock", wait=3):
            state = self.read()
            for key in IMMUTABLE & changes.keys():
                if changes[key] != state.get(key):
                    raise Failure("immutable journal field")
            if "candidate_id" in changes:
                container_id(changes["candidate_id"])
                if state.get("candidate_id") not in (None, changes["candidate_id"]):
                    raise Failure("candidate ID already bound")
            if state.get("authorization_consumed") and changes.get("authorization_consumed") is False:
                raise Failure("authorization cannot be reset")
            if state.get("mutation_intent") and changes.get("mutation_intent") is False:
                raise Failure("mutation intent cannot be reset")
            if state.get("phase") in TERMINAL and changes.get("phase", state["phase"]) != state["phase"]:
                recovery_phase = changes.get("phase")
                evidence = changes.get("recovery", {})
                recovery_only = (state["phase"] in {"RESTORE_FAILED", "RESTORED", "NOOP", "COMPLETE"} and
                                 recovery_phase in {"RESTORE_FAILED", "RESTORED", "NOOP"} and
                                 evidence.get("result") == recovery_phase and
                                 evidence.get("verified") is (recovery_phase != "RESTORE_FAILED"))
                if not recovery_only:
                    raise Failure("terminal journal cannot be reopened")
            state.update(changes)
            state["sequence"] += 1
            state["updated_monotonic"] = time.monotonic()
            private_json(self.directory / "journal.json", state)
            return state

    def intent(self, operation, **changes):
        state = self.read()
        history = state.get("history", []) + [{"operation": operation, "at": time.monotonic()}]
        if len(history) > 128:
            raise Failure("one-attempt operation limit exceeded")
        return self.update(operation=operation, history=history, **changes)

    def gate(self, name, evidence):
        if not isinstance(evidence, dict) or evidence.get("ok") is not True:
            raise Failure("verification gate failed: " + name)
        gates = dict(self.read().get("gates", {}))
        gates[name] = evidence
        self.update(gates=gates)


def process_identity(pid):
    if os.name != "posix":
        raise Failure("process supervision requires Linux")
    try:
        fields = Path("/proc", str(int(pid)), "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return None
        return {"pid": int(pid), "start_ticks": fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def check_supervisor(journal):
    lease = read_json(journal.directory / "supervisor.json")
    state = journal.read()
    if (lease.get("attempt_id") != state["attempt_id"] or lease.get("status") != "watching" or
            lease.get("controller_pid") != os.getpid() or lease.get("nonce") != state.get("supervision_nonce") or
            process_identity(lease.get("pid")) != lease.get("identity") or
            time.monotonic() - lease.get("heartbeat", 0) > state["plan"]["supervisor_lease_s"]):
        raise Failure("independent supervisor unavailable")
    if (journal.directory / "recover-request.json").exists():
        raise Failure("supervisor requested recovery")


def checked_inspect(api, identity, expected_image):
    result = api.inspect(container_id(identity))
    if result.get("Id") != identity or result.get("Image") != expected_image:
        raise Failure("immutable container/image identity mismatch")
    return result


def is_running(document):
    state = document.get("State", {})
    return state.get("Running") is True and not state.get("Restarting", False) and not state.get("Dead", False)


def require_running(api, identity, expected_image):
    document = checked_inspect(api, identity, expected_image)
    if not is_running(document):
        raise Failure("container exited or is restarting")
    return document


# Documented POST /containers/create rejections that prove the request created
# nothing. Any other outcome (transport loss, 5xx, unknown exception) may still
# have been accepted by the Engine and stays uncertain.
_CREATE_REJECTED_STATUSES = frozenset({400, 404, 409})
# Bounded read-only grace while an uncertain create may still take effect.
_PROBE_RECONCILE_GRACE_S = 5.0


def create_rejection_is_definitive(error):
    """A definite create rejection is never a possibly-successful own create."""
    return isinstance(error, DockerError) and error.status in _CREATE_REJECTED_STATUSES


def _nonce_bound_name(final_name, resource_nonce, marker):
    """Fresh cryptographic nonce-bound temporary create name.

    The nonce is an ownership token for one named create request, never a new
    credential and never part of the serving configuration. Only this attempt,
    which generated the nonce and durably recorded it before create, can be
    named by it; the final serving/staging name stays exactly as before.
    """
    if not isinstance(final_name, str) or not re.fullmatch(r"[a-f0-9]{32}", resource_nonce or ""):
        raise Failure("invalid resource nonce")
    name = final_name + marker + resource_nonce
    if len(name) > 128:
        raise Failure("nonce-bound create name too long")
    return name


def candidate_create_name(candidate_name, resource_nonce):
    return _nonce_bound_name(candidate_name, resource_nonce, "-n")


def artifact_probe_create_name(candidate_name, resource_nonce):
    return _nonce_bound_name(candidate_name, resource_nonce, "-artifact-n")


def _request_digest(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _actual_configuration_digest(document):
    return hashlib.sha256(json.dumps({key: document.get(key) for key in
        ("Config", "HostConfig", "Mounts")}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_HELPER_CONTEXT = threading.local()
_PROCESS_NONCE = secrets.token_hex(16)


def helper_owner_identity():
    identity = process_identity(os.getpid()) if os.name == "posix" else None
    if identity is None and os.name == "posix":
        raise Failure("helper operation owner identity unavailable")
    # The nonce also distinguishes interpreter lifetimes in offline Windows tests.
    return {"process": identity, "pid": os.getpid(), "process_nonce": _PROCESS_NONCE}


def _helper_gate(journal):
    path = journal.directory / "helper-admission.json"
    if path.exists():
        return read_json(path)
    if journal.read().get("phase") in TERMINAL:
        raise Failure("helper admission closed by terminal attempt")
    gate = {"attempt_id": journal.read()["attempt_id"], "generation": secrets.token_hex(16),
            "mode": "deployment", "owner": helper_owner_identity()}
    private_json(path, gate, exclusive=True)
    return gate


def rotate_helper_admission(journal, mode, *, attempt_id=None):
    """A short local transaction, never held over a Docker/network operation."""
    if mode not in ("recovery", "closed"):
        raise Failure("invalid helper admission mode")
    with FileLock(journal.directory / "helper-operations.lock", wait=0.2):
        gate = {"attempt_id": attempt_id or journal.read()["attempt_id"], "generation": secrets.token_hex(16),
                "mode": mode, "owner": helper_owner_identity()}
        private_json(journal.directory / "helper-admission.json", gate)
    return gate


class VerificationWork:
    """Local cancellation plus durable operation revocation for a daemon check."""
    def __init__(self):
        self.cancelled = threading.Event()
        self.parent = getattr(_HELPER_CONTEXT, "work", None)
        self.scope = getattr(_HELPER_CONTEXT, "scope", None)
        self.operations = []
        self.lock = threading.Lock()

    def check(self):
        if self.cancelled.is_set():
            raise Failure("verification operation abandoned")
        if self.parent:
            self.parent.check()

    def register(self, operation):
        with self.lock:
            self.operations.append(operation)

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            operations = list(self.operations)
        for operation in operations:
            with contextlib.suppress(Exception):
                operation.revoke()


class HelperOperation:
    """Durable admission BEFORE blocking preflight, and write-ahead mutation state.

    Revocation does not pretend that an already submitted create cannot complete.
    Such work remains pending until its exact owned helper is positively settled.
    """
    def __init__(self, journal):
        self.journal = journal
        self.work = getattr(_HELPER_CONTEXT, "work", None)
        self.scope = getattr(_HELPER_CONTEXT, "scope", None)
        self.nonce = secrets.token_hex(16)
        self.directory = journal.directory / "helper-operations"
        self.directory.mkdir(mode=0o700, exist_ok=True)
        if self.directory.is_symlink():
            raise Failure("unsafe helper operation directory")
        self.path = self.directory / (self.nonce + ".json")
        with FileLock(journal.directory / "helper-operations.lock", wait=0.2):
            if self.work:
                self.work.check()
            gate = _helper_gate(journal)
            if (gate["mode"] == "closed" or gate.get("owner") != helper_owner_identity() or
                    (self.scope is None and gate["mode"] != "deployment")):
                raise Failure("helper operation admission revoked")
            if self.scope is not None and (self.scope != gate or gate["owner"] != helper_owner_identity()):
                raise Failure("recovery helper operation admission mismatch")
            if len(list(self.directory.glob("*.json"))) >= 128:
                raise Failure("helper operation limit exceeded")
            record = {"schema": 1, "attempt_id": gate["attempt_id"], "nonce": self.nonce,
                      "generation": gate["generation"], "mode": gate["mode"], "owner": helper_owner_identity(),
                      "state": "PREFLIGHT", "revoked": False, "finished": False,
                      "helper_nonce": None, "mutation": None, "created_at": time.monotonic()}
            private_json(self.path, record, exclusive=True)
            if self.work:
                self.work.register(self)

    def _check(self, record):
        if self.work:
            self.work.check()
        gate = read_json(self.journal.directory / "helper-admission.json")
        if (record.get("revoked") or record.get("finished") or record.get("state") == "REVOKED_SETTLED" or
                gate.get("generation") != record.get("generation") or gate.get("mode") != record.get("mode") or
                gate.get("owner") != record.get("owner") or record.get("owner") != helper_owner_identity()):
            raise Failure("helper operation admission revoked")

    def check(self):
        with FileLock(self.journal.directory / "helper-operations.lock", wait=0.2):
            self._check(read_json(self.path))

    def create_intent(self, image, name, nonce, body, image_document):
        with FileLock(self.journal.directory / "helper-operations.lock", wait=0.2):
            record = read_json(self.path)
            self._check(record)
            if record.get("state") != "PREFLIGHT" or record.get("helper_nonce"):
                raise Failure("helper operation create already attempted")
            path = helper_intent(self.journal, image, name, nonce, body, image_document,
                                 operation_nonce=self.nonce, generation=record["generation"])
            record.update(helper_nonce=nonce, state="CREATE_IN_FLIGHT", mutation="create")
            private_json(self.path, record)
            return path

    def acknowledge(self, mutation, identity):
        with FileLock(self.journal.directory / "helper-operations.lock", wait=0.2):
            record = read_json(self.path)
            if record.get("container_id") not in (None, identity):
                raise Failure("helper operation immutable ID changed")
            record.update(container_id=container_id(identity), state=mutation.upper() + "_ACKNOWLEDGED", mutation=None)
            private_json(self.path, record)

    def start_intent(self):
        with FileLock(self.journal.directory / "helper-operations.lock", wait=0.2):
            record = read_json(self.path)
            self._check(record)
            if record.get("state") != "CREATE_ACKNOWLEDGED" or not record.get("container_id"):
                raise Failure("helper operation start is not a single acknowledged create")
            record.update(state="START_IN_FLIGHT", mutation="start")
            private_json(self.path, record)

    def revoke(self):
        with FileLock(self.journal.directory / "helper-operations.lock", wait=0.2):
            record = read_json(self.path)
            record["revoked"] = True
            private_json(self.path, record)

    def finish(self):
        with FileLock(self.journal.directory / "helper-operations.lock", wait=0.2):
            record = read_json(self.path)
            record["finished"] = True
            # Keep an uncertain submitted mutation visible. Completion of a Python
            # call or owner death cannot prove the Engine did not accept its work.
            if not record.get("mutation"):
                record["state"] = "FINISHED"
            private_json(self.path, record)


def reconcile_helper_operations(journal):
    directory = journal.directory / "helper-operations"
    if not directory.exists():
        return []
    if directory.is_symlink():
        raise Failure("unsafe helper operation directory")
    with FileLock(journal.directory / "helper-operations.lock", wait=0.2):
        gate = read_json(journal.directory / "helper-admission.json")
        paths = sorted(directory.glob("*.json"))
        if len(paths) > 128:
            raise Failure("helper operation limit exceeded")
        outcomes = []
        for path in paths:
            record = read_json(path)
            if (record.get("schema") != 1 or record.get("attempt_id") != gate.get("attempt_id") or
                    not re.fullmatch(r"[a-f0-9]{32}", record.get("nonce", "")) or path.stem != record["nonce"] or
                    not re.fullmatch(r"[a-f0-9]{32}", record.get("generation", "")) or
                    record.get("mode") not in ("deployment", "recovery") or
                    not isinstance(record.get("owner"), dict) or type(record.get("finished")) is not bool or
                    type(record.get("revoked")) is not bool):
                raise Failure("invalid helper operation record")
            revoked = record.get("revoked") or record.get("generation") != gate.get("generation")
            helper_nonce = record.get("helper_nonce")
            helper = None
            if helper_nonce:
                if not re.fullmatch(r"[a-f0-9]{32}", helper_nonce):
                    raise Failure("invalid operation helper identity")
                helper_path = journal.directory / "helpers" / (helper_nonce + ".json")
                if helper_path.exists():
                    helper = read_json(helper_path)
                    if (helper.get("nonce") != helper_nonce or helper.get("attempt_id") != record["attempt_id"] or
                            helper.get("operation_nonce") != record["nonce"] or
                            helper.get("operation_generation") != record["generation"] or
                            record.get("container_id") not in (None, helper.get("container_id"))):
                        raise Failure("helper operation ownership link differs")
            positively_removed = bool(helper and helper.get("state") == "REMOVED" and helper.get("container_id"))
            settled = (record.get("state") == "REVOKED_SETTLED" or
                       (record.get("finished") is True and not record.get("mutation")) or
                       (revoked and record.get("state") == "PREFLIGHT" and not helper_nonce) or
                       (revoked and positively_removed))
            if settled and revoked:
                record.update(revoked=True, state="REVOKED_SETTLED")
                private_json(path, record)
            outcomes.append({"operation_nonce": record["nonce"], "settled": bool(settled),
                             "revoked": bool(revoked), "state": record["state"]})
        return outcomes


def helper_operations_settled(outcomes):
    return isinstance(outcomes, list) and all(isinstance(item, dict) and item.get("settled") is True for item in outcomes)


def exit_aware(api, identity, expected_image, operation, deadline, *, guard=lambda: None,
               interval=0.5):
    """Monitor Docker continuously and revoke abandoned daemon work admission."""
    result = queue.Queue(maxsize=1)
    work = VerificationWork()
    def run():
        _HELPER_CONTEXT.work, _HELPER_CONTEXT.scope = work, work.scope
        try:
            work.check()
            result.put((True, operation()))
        except BaseException as error:
            result.put((False, type(error).__name__))
        finally:
            _HELPER_CONTEXT.work = _HELPER_CONTEXT.scope = None
    require_running(api, identity, expected_image)
    guard()
    threading.Thread(target=run, daemon=True).start()
    try:
        while True:
            guard()
            require_running(api, identity, expected_image)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Deadline("verification deadline exceeded")
            try:
                ok, value = result.get(timeout=min(interval, remaining))
            except queue.Empty:
                continue
            guard()
            require_running(api, identity, expected_image)
            if not ok:
                raise Failure("verification raised " + value)
            return value
    finally:
        work.cancel()


def wait_ready(api, identity, expected_image, health, deadline, *, guard=lambda: None,
               interval=0.5, observation=lambda evidence: None):
    while time.monotonic() < deadline:
        try:
            evidence = exit_aware(api, identity, expected_image, health, deadline,
                                  guard=guard, interval=interval)
        except Deadline:
            require_running(api, identity, expected_image)
            raise RunningUnready("running container readiness deadline") from None
        observation(evidence)
        if isinstance(evidence, dict) and evidence.get("ok") is True:
            return evidence
        guard()
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
    require_running(api, identity, expected_image)
    raise RunningUnready("running container did not become ready")


def _probe_intent_identity(ledger, state):
    if (ledger.get("attempt_id") != state["attempt_id"] or
            ledger.get("name") != artifact_probe_create_name(state["candidate_name"],
                                                             ledger.get("resource_nonce")) or
            ledger.get("image_id") != state["candidate_image_id"]):
        raise Failure("artifact probe intent identity differs")


def _reconcile_artifact_probe_create(api, journal, ledger_path, error):
    """Read-only settlement of an uncertain artifact-probe create.

    A definitive Engine rejection proves this attempt created nothing and never
    adopts any occupant. An uncertain reply may record cleanup identity only
    for a stopped container at this attempt's fresh nonce-bound name whose
    image, mounts, environment and isolation still match the durable intent
    exactly. Nothing is stopped, removed, started or retried here; the verified
    own stopped probe is retained for separately invoked explicit cleanup, and
    a missing, foreign or altered resource stays unresolved and untouched.
    """
    from .docker_config import assert_artifact_probe_parity, build_artifact_probe_request
    ledger, state = read_json(ledger_path), journal.read()
    rejected = create_rejection_is_definitive(error)
    ledger.update(create_outcome="rejected" if rejected else "uncertain",
                  create_error_status=getattr(error, "status", None))
    if rejected:
        ledger.update(phase="UNRESOLVED_CREATE", reconciliation="unresolved", cleanup_admissible=False)
        private_json(ledger_path, ledger)
        return
    deadline = time.monotonic() + min(state["plan"]["probe_timeout_s"], _PROBE_RECONCILE_GRACE_S)
    document = None
    while document is None:
        try:
            document = api.inspect(ledger.get("container_id") or ledger["name"])
        except (DockerError, OSError):
            # An unacknowledged create may still take effect; absence is not proof.
            if time.monotonic() >= deadline:
                break
            time.sleep(min(state["plan"]["poll_interval_s"], max(0.0, deadline - time.monotonic())))
    identity = None
    try:
        _probe_intent_identity(ledger, state)
        if document is None:
            raise Failure("artifact probe create effect is missing or still unknown")
        identity = container_id(document.get("Id"))
        if (ledger.get("container_id") not in (None, identity) or
                document.get("Name") != "/" + ledger["name"] or
                document.get("Image") != ledger["image_id"] or
                document.get("State", {}).get("Status") != "created" or
                document.get("State", {}).get("Running")):
            raise Failure("artifact probe reconciliation is not exact and stopped")
        request = build_artifact_probe_request(journal.original(), ledger["image_id"], ledger["name"])
        if _request_digest(request) != ledger.get("request_sha256"):
            raise Failure("artifact probe create intent request differs")
        assert_artifact_probe_parity(journal.original(), document, ledger["image_id"])
        ledger.update(phase="VERIFIED_OWN_RETAINED", container_id=identity,
                      reconciliation="verified_cleanup_only", cleanup_admissible=True,
                      observed_stopped=True,
                      actual_configuration_sha256=_actual_configuration_digest(document))
    except BaseException as reconcile_error:
        ledger.update(phase="UNRESOLVED_CREATE", reconciliation="unresolved", cleanup_admissible=False,
                      reconciliation_error=type(reconcile_error).__name__)
        private_json(ledger_path, ledger)
        return
    # Root correction: an unknown create response records read-only cleanup
    # identity/reconciliation only. Zero automatic stop/remove/start/retry.
    private_json(ledger_path, ledger)


def _reconcile_artifact_probe_execution(api, journal, ledger_path, error):
    """Read-only settlement of an uncertain probe start by its bound immutable ID.

    Root correction: an unknown start response records read-only cleanup
    identity/reconciliation only. Zero stop/remove/start/retry; a verified own
    running or exited probe is retained for separately invoked explicit cleanup.
    """
    ledger, state = read_json(ledger_path), journal.read()
    deadline = time.monotonic() + min(state["plan"]["probe_timeout_s"], _PROBE_RECONCILE_GRACE_S)
    document = None
    while document is None:
        try:
            document = checked_inspect(api, ledger["container_id"], ledger["image_id"])
        except DockerError as inspect_error:
            if inspect_error.status == 404:
                # An acknowledged immutable ID cannot later name another container.
                ledger.update(phase="VERIFIED_ABSENT", reconciliation="verified_absent", observed_absent=True)
                private_json(ledger_path, ledger)
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(min(state["plan"]["poll_interval_s"], max(0.0, deadline - time.monotonic())))
        except OSError:
            if time.monotonic() >= deadline:
                break
            time.sleep(min(state["plan"]["poll_interval_s"], max(0.0, deadline - time.monotonic())))
    if document is None:
        # The probe state stayed unknown; it is retained untouched.
        ledger.update(phase="UNRESOLVED_EXECUTION", reconciliation="unresolved", cleanup_admissible=False)
        private_json(ledger_path, ledger)
        return
    try:
        if document.get("Name") != "/" + ledger["name"]:
            raise Failure("artifact probe identity differs from durable intent")
        ledger.update(phase="VERIFIED_OWN_RETAINED", reconciliation="verified_cleanup_only",
                      cleanup_admissible=True,
                      observed_running=bool(document.get("State", {}).get("Running")),
                      actual_configuration_sha256=_actual_configuration_digest(document))
    except BaseException as reconcile_error:
        ledger.update(phase="UNRESOLVED_EXECUTION", reconciliation="unresolved", cleanup_admissible=False,
                      reconciliation_error=type(reconcile_error).__name__)
    private_json(ledger_path, ledger)


def explicit_artifact_probe_cleanup(api, journal):
    """Separately invoked exact-ID cleanup of this attempt's own retained probe.

    Revalidates the durable intent, the bound immutable ID, the nonce-bound
    name, the image identity and the full actual configuration before any
    effect. Refuses unknown, foreign, altered or running resources without
    mutation, and never touches the failed attempt's journal phase. There is
    no prefix or list cleanup; only this exact immutable ID is removed.
    """
    from .docker_config import assert_artifact_probe_parity, build_artifact_probe_request
    ledger_path = journal.directory / "artifact-probe-intent.json"
    ledger, state = read_json(ledger_path), journal.read()
    result = {"removed": False, "attempt_id": state["attempt_id"]}
    try:
        _probe_intent_identity(ledger, state)
        identity = ledger.get("container_id")
        if not isinstance(identity, str):
            raise Failure("explicit probe cleanup requires a bound immutable ID")
        identity = container_id(identity)
        request = build_artifact_probe_request(journal.original(), ledger["image_id"], ledger["name"])
        if _request_digest(request) != ledger.get("request_sha256"):
            raise Failure("artifact probe intent request differs")
        try:
            document = api.inspect(identity)
        except DockerError as error:
            if error.status != 404:
                raise
            ledger.update(reconciliation="verified_absent", observed_absent=True)
            private_json(ledger_path, ledger)
            return dict(result, removed=True, already_absent=True)
        if (document.get("Id") != identity or document.get("Name") != "/" + ledger["name"] or
                document.get("Image") != ledger["image_id"]):
            raise Failure("explicit probe cleanup identity mismatch")
        expected = ledger.get("actual_configuration_sha256")
        if expected is not None and expected != _actual_configuration_digest(document):
            raise Failure("explicit probe cleanup configuration changed")
        if document.get("State", {}).get("Running"):
            raise Failure("explicit probe cleanup refuses a running probe")
        if document.get("State", {}).get("Status") == "created":
            assert_artifact_probe_parity(journal.original(), document, ledger["image_id"])
        api.remove(identity)
        ledger.update(reconciliation="explicit_cleanup_removed", cleanup_complete=True, cleanup_admissible=False)
        private_json(ledger_path, ledger)
        return dict(result, removed=True)
    except BaseException as error:
        reason = type(error).__name__
        with contextlib.suppress(Exception):
            ledger.update(cleanup_refused=reason)
            private_json(ledger_path, ledger)
        return dict(result, removed=False, refused=reason)


def artifact_probe(api, journal):
    from .docker_config import assert_artifact_probe_parity, assert_probe_image, build_artifact_probe_request
    state, original = journal.read(), journal.original()
    assert_probe_image(api.inspect_image(state["candidate_image_id"]), state["candidate_image_id"], original)
    nonce = secrets.token_hex(16)
    name = artifact_probe_create_name(state["candidate_name"], nonce)
    body = build_artifact_probe_request(original, state["candidate_image_id"], name)
    ledger_path = journal.directory / "artifact-probe-intent.json"
    ledger = {"schema": 1, "attempt_id": state["attempt_id"], "resource_nonce": nonce,
              "name": name, "final_name": state["candidate_name"] + "-artifact",
              "image_id": state["candidate_image_id"], "owner": state["plan"].get("rehearsal_owner"),
              "container_id": None, "phase": "CREATE_INTENT", "request_sha256": _request_digest(body),
              "cleanup_admissible": False, "cleanup_complete": False}
    private_json(ledger_path, ledger, exclusive=True)
    journal.intent("create_artifact_probe")
    try:
        identity = api.create(body, name)
    except BaseException as error:
        _reconcile_artifact_probe_create(api, journal, ledger_path, error)
        raise
    # The immutable ID is durably bound before any execution of the resource.
    ledger.update(container_id=container_id(identity), phase="CREATE_ACKNOWLEDGED")
    private_json(ledger_path, ledger)
    journal.update(probe_id=identity)
    try:
        document = checked_inspect(api, identity, state["candidate_image_id"])
        if document.get("Name") != "/" + name:
            raise Failure("artifact probe create name binding differs")
        probe_parity = assert_artifact_probe_parity(original, document, state["candidate_image_id"])
        journal.update(probe_parity=probe_parity)
        host = document.get("HostConfig", {})
        if host.get("Runtime") != "runc" or host.get("DeviceRequests") or host.get("Devices") or host.get("Privileged"):
            raise Failure("artifact probe exposed devices")
    except BaseException:
        # An actual mismatch stays unresolved and untouched; it is never adopted.
        ledger.update(phase="UNRESOLVED_ACTUAL_MISMATCH", reconciliation="unresolved", cleanup_admissible=False)
        private_json(ledger_path, ledger)
        raise
    ledger.update(phase="VERIFIED_STOPPED_OWN", cleanup_admissible=True,
                  actual_configuration_sha256=_actual_configuration_digest(document))
    private_json(ledger_path, ledger)
    try:
        api.start(identity)
    except BaseException as error:
        _reconcile_artifact_probe_execution(api, journal, ledger_path, error)
        raise
    ledger.update(phase="START_ACKNOWLEDGED")
    private_json(ledger_path, ledger)
    end = time.monotonic() + state["plan"]["probe_timeout_s"]
    while time.monotonic() < end:
        document = checked_inspect(api, identity, state["candidate_image_id"])
        if not document.get("State", {}).get("Running"):
            if document.get("State", {}).get("ExitCode") != 0:
                ledger.update(phase="PROBE_FAILED")
                private_json(ledger_path, ledger)
                raise Failure("artifact probe failed")
            try:
                proof = json.loads(api.logs(identity))
            except (ValueError, UnicodeError):
                ledger.update(phase="PROOF_INVALID")
                private_json(ledger_path, ledger)
                raise Failure("artifact probe proof malformed") from None
            if (proof.get("schema") != "occ3-artifact-preflight/v1" or
                    proof.get("result") != "PASS" or proof.get("cpu_only") is not True or
                    proof.get("loader_profile") != "sglang-0.5.21-glm-dsa-native-fp8" or
                    proof.get("model_config_json_valid") is not True or
                    proof.get("tokenizer_config_json_valid") is not True or
                    any(type(proof.get(k)) is not int or proof[k] <= 0 for k in (
                        "weight_shards", "weight_data_bytes_read", "weight_header_bytes_read",
                        "tokenizer_artifacts", "total_bytes_read"))):
                ledger.update(phase="PROOF_INVALID")
                private_json(ledger_path, ledger)
                raise Failure("artifact probe did not prove CPU-only artifact reads")
            private_json(journal.directory / "artifact-proof.json", proof)
            try:
                api.remove(identity)
            except BaseException as remove_error:
                ledger.update(phase="REMOVE_PENDING", remove_error=type(remove_error).__name__)
                private_json(ledger_path, ledger)
                raise
            ledger.update(phase="REMOVED", cleanup_complete=True, cleanup_admissible=False)
            private_json(ledger_path, ledger)
            return proof
        time.sleep(min(state["plan"]["poll_interval_s"], max(0, end - time.monotonic())))
    api.stop(identity, seconds=1)
    ledger.update(phase="TIMEOUT_STOPPED")
    private_json(ledger_path, ledger)
    raise Deadline("artifact probe timeout")


def inspect_ingress(api, plan):
    route = plan.get("ingress", {})
    identity = container_id(route.get("container_id"))
    expected_image = image_id(route.get("image_id"))
    document = require_running(api, identity, expected_image)
    endpoint = document.get("NetworkSettings", {}).get("Networks", {}).get(route.get("network_name"), {})
    if (document.get("Name") != "/" + route.get("container_name", "") or
            endpoint.get("NetworkID") != route.get("network_id") or not route.get("network_id")):
        raise Failure("pinned ingress route mismatch")
    if plan.get("mode") == "rehearsal" and plan.get("rehearsal_ingress") is True:
        owner = plan.get("rehearsal_owner", "")
        prefix = "occ3-v3-rehearsal-" + owner
        network = api.inspect_network(route["network_id"])
        if (not re.fullmatch(r"[a-f0-9]{16}", owner) or route.get("container_name") != prefix + "-proxy" or
                route.get("network_name") != prefix + "-net" or network.get("Id") != route["network_id"] or
                network.get("Name") != route["network_name"] or
                network.get("Labels", {}).get(REHEARSAL_LABEL) != owner or
                document.get("Config", {}).get("Labels", {}).get(REHEARSAL_LABEL) != owner):
            raise Failure("unowned disposable ingress refused")
    elif (plan.get("mode") != "production" or route.get("container_name") != "neo-k3-ingress" or
          plan["ingress_base_url"].rstrip("/") != "http://neo-k3-ingress:8080"):
        raise Failure("production ingress route mismatch")
    return document


def capture_ingress(api, journal):
    state = journal.read()
    plan = state["plan"]
    document = inspect_ingress(api, plan)
    result = api.nginx(document["Id"], "config", timeout=10)
    if result["exit_code"] != 0 or not result["stdout"]:
        raise Failure("nginx configuration capture failed")
    digest = hashlib.sha256(result["stdout"]).hexdigest()
    if digest != plan["ingress"].get("config_sha256"):
        raise Failure("nginx configuration differs from independent preflight digest")
    snapshot = {"container_id": document["Id"], "image_id": document["Image"],
                "network_id": plan["ingress"]["network_id"], "config_sha256": digest}
    journal.update(ingress_snapshot=snapshot)
    return snapshot


def refresh_ingress(api, journal, state, deadline, *, guard=lambda: None, recovery=False):
    if state["plan"]["mode"] == "rehearsal" and state["plan"].get("rehearsal_ingress") is not True:
        return {"ok": True, "scope": "not_exercised_disposable_has_no_nginx"}
    if not state.get("authorization_consumed"):
        raise Failure("ingress refresh requires consumed authorization")
    snapshot = state.get("ingress_snapshot", {})
    def remaining():
        guard()
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise Deadline("ingress refresh deadline")
        return min(10, budget)
    document = inspect_ingress(api, state["plan"])
    if snapshot.get("container_id") != document["Id"] or snapshot.get("image_id") != document["Image"]:
        raise Failure("ingress snapshot identity mismatch")
    captured = api.nginx(document["Id"], "config", timeout=remaining())
    if (captured["exit_code"] != 0 or not captured["stdout"] or
            hashlib.sha256(captured["stdout"]).hexdigest() != snapshot.get("config_sha256")):
        raise Failure("refuse nginx reload after configuration change")
    tested = api.nginx(document["Id"], "test", timeout=remaining())
    if tested["exit_code"] != 0:
        raise Failure("nginx configuration test failed")
    operation = "recovery_ingress_reload" if recovery else "candidate_ingress_reload"
    if recovery:
        with contextlib.suppress(Exception):
            journal.intent(operation)
    else:
        journal.intent(operation)
    remaining()
    inspect_ingress(api, state["plan"])
    reloaded = api.nginx(document["Id"], "reload", timeout=remaining())
    if reloaded["exit_code"] != 0:
        raise Failure("nginx graceful reload failed")
    proof = {"ok": True, "operation": operation, "container_id": document["Id"],
             "config_sha256": snapshot["config_sha256"], "configuration_unchanged": True,
             "service_verification_still_required": True}
    if recovery:
        with contextlib.suppress(Exception):
            journal.update(ingress_refresh=proof)
    else:
        journal.update(ingress_refresh=proof)
    return proof


def assert_inspect_schema(original, reference):
    """Independent current CLI-inspect schema guards against versioned field loss.

    Reference values may be sanitized. Only ID/image and field presence are
    compared here; exact launch values are checked by the separate parity code.
    """
    if isinstance(reference, list) and len(reference) == 1:
        reference = reference[0]
    if not isinstance(reference, dict) or original.get("Id") != reference.get("Id") or original.get("Image") != reference.get("Image"):
        raise Failure("independent inspect schema identity mismatch")
    for left, right in ((original, reference), (original.get("Config"), reference.get("Config")),
                        (original.get("HostConfig"), reference.get("HostConfig")),
                        (original.get("NetworkSettings"), reference.get("NetworkSettings"))):
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            raise Failure("Docker API omitted or changed independent inspect fields")
    left_networks = original["NetworkSettings"].get("Networks", {})
    right_networks = reference["NetworkSettings"].get("Networks", {})
    if set(left_networks) != set(right_networks):
        raise Failure("independent inspect network schema mismatch")
    for name in left_networks:
        if set(left_networks[name]) != set(right_networks[name]):
            raise Failure("Docker API omitted endpoint configuration fields")
    left_mounts = original.get("Mounts", [])
    right_mounts = reference.get("Mounts", [])
    if (len(left_mounts) != len(right_mounts) or
            sorted(sorted(x) for x in left_mounts) != sorted(sorted(x) for x in right_mounts)):
        raise Failure("Docker API omitted mount fields")
    return {"ok": True, "scope": "independent_cli_inspect_field_presence"}


def reconcile_candidate_create(api, journal, error=None):
    """Read-only identity capture after a create failure. Never reopen deployment.

    A definitive Engine rejection of the create proves this attempt created
    nothing, so the occupant of the requested name is never adopted. After an
    uncertain reply, cleanup may bind only to a stopped container at this
    attempt's fresh nonce-bound create name whose full serving configuration
    still matches the original exactly. Missing, foreign or altered resources
    stay unresolved and untouched; deployment is never retried or reopened.
    """
    from .docker_config import assert_candidate_parity, build_create_request
    state, original = journal.read(), journal.original()
    path = journal.directory / "candidate-create-intent.json"
    record = read_json(path)
    rejected = create_rejection_is_definitive(error)
    record.update(create_outcome="rejected" if rejected else "uncertain",
                  create_error_status=getattr(error, "status", None))
    try:
        if (record.get("attempt_id") != state["attempt_id"] or record.get("final_name") != state["candidate_name"] or
                record.get("name") != candidate_create_name(state["candidate_name"], record.get("resource_nonce")) or
                record.get("image_id") != state["candidate_image_id"]):
            raise Failure("candidate create intent identity differs")
        if rejected:
            raise Failure("rejected create cannot adopt the occupant of the requested name")
        binding = calibrated_binding_from_state(state)
        request = build_create_request(original, state["candidate_image_id"], record["name"],
                                       state["plan"].get("environment_additions"), binding)
        if _request_digest(request) != record["request_sha256"]:
            raise Failure("candidate create intent request differs")
        document = api.inspect(record.get("container_id") or record["name"])
        identity = container_id(document.get("Id"))
        if (record.get("container_id") not in (None, identity) or document.get("Name") != "/" + record["name"] or
                document.get("Image") != record["image_id"] or document.get("State", {}).get("Status") != "created" or
                document.get("State", {}).get("Running")):
            raise Failure("candidate create reconciliation is not exact and stopped")
        assert_candidate_parity(original, document, state["candidate_image_id"],
                                state["plan"].get("environment_additions"), binding)
        record.update(container_id=identity, reconciliation="verified_cleanup_only", cleanup_admissible=True,
                      actual_configuration_sha256=_actual_configuration_digest(document))
    except Exception as reconcile_error:
        record.update(reconciliation="unresolved", cleanup_admissible=False,
                      reconciliation_error=type(reconcile_error).__name__)
    private_json(path, record)
    return record


def prepare(api, plan, directory, *, probe=artifact_probe):
    from .docker_config import (assert_candidate_parity, build_create_request, extract_api_key,
                                calibrated_capability, derive_calibrated_binding,
                                verify_calibrated_artifact, verify_calibrated_profile)
    plan = validate_plan(plan)
    original = checked_inspect(api, plan["original_id"], plan["original_image_id"])
    if not is_running(original):
        raise Failure("original service is not running")
    if plan["mode"] == "rehearsal":
        assert_rehearsal_original(original, plan)
    else:
        reference = plan.get("inspect_schema_evidence")
        verify_receipt(reference)
        assert_inspect_schema(original, read_json(reference["path"]))
    extract_api_key(original)
    capability = calibrated_capability(plan)
    binding = artifact_identity = None
    if capability is not None:
        # Verified staged profile first; its calibration_path is the sole
        # authority for the artifact filename (never a digest). The artifact
        # bytes are verified on this Docker host before anything is created.
        verified = verify_calibrated_profile(capability)
        binding = derive_calibrated_binding(verified, original)
        artifact_identity = verify_calibrated_artifact(binding["source"], capability["sha256"],
                                                       capability["size"])
    preserved = preserved_original_record(plan, original)
    deadlines = timeout_policy(plan)
    if api.inspect_image(plan["candidate_image_id"]).get("Id") != plan["candidate_image_id"]:
        raise Failure("candidate image identity mismatch")
    journal = Journal.create(directory, plan, original,
                              calibrated=(None if binding is None else
                                          {"source": binding["source"],
                                           "destination": binding["destination"],
                                           "basename": binding["basename"],
                                           "identity": artifact_identity}))
    journal.update(docker_api_version=getattr(api, "version", "offline-test-adapter"),
                   preserved_original=preserved, timeout_policy=deadlines)
    try:
        if plan["mode"] == "production":
            capture_ingress(api, journal)
            proof = probe(api, journal)
        else:
            if plan.get("rehearsal_ingress") is True:
                capture_ingress(api, journal)
            proof = {"result": "PASS", "cpu_only": True, "kind": "disposable-fixture-only"}
        journal.update(artifact_proof=proof)
        state = journal.read()
        # The candidate is created under a fresh nonce-bound temporary name.
        # Only this attempt can be named by the nonce, so a container at that
        # exact name with the exact planned configuration demonstrates this
        # attempt's own request identity. The final staging name is unchanged.
        resource_nonce = secrets.token_hex(16)
        create_name = candidate_create_name(state["candidate_name"], resource_nonce)
        body = build_create_request(original, state["candidate_image_id"], create_name,
                                    state["plan"].get("environment_additions"), binding)
        journal.intent("create_candidate")
        create_record = {"schema": 1, "attempt_id": state["attempt_id"], "resource_nonce": resource_nonce,
                         "name": create_name, "final_name": state["candidate_name"],
                         "image_id": state["candidate_image_id"],
                         "owner": plan.get("rehearsal_owner"), "container_id": None, "phase": "CREATE_INTENT",
                         "request_sha256": _request_digest(body),
                         "cleanup_admissible": False, "cleanup_complete": False}
        private_json(journal.directory / "candidate-create-intent.json", create_record, exclusive=True)
        identity = api.create(body, create_name)
        create_record.update(container_id=container_id(identity), phase="CREATE_ACKNOWLEDGED")
        private_json(journal.directory / "candidate-create-intent.json", create_record)
        journal.update(candidate_id=identity)
        candidate = checked_inspect(api, identity, state["candidate_image_id"])
        if candidate.get("Name") != "/" + create_name:
            raise Failure("candidate create name binding differs")
        if is_running(candidate) or candidate.get("State", {}).get("Status") != "created":
            raise Failure("candidate was not created stopped")
        evidence = assert_candidate_parity(original, candidate, state["candidate_image_id"],
                                           state["plan"].get("environment_additions"), binding)
        private_json(journal.directory / "candidate-inspect.json", candidate, exclusive=True)
        create_record.update(phase="VERIFIED_STOPPED_OWN")
        private_json(journal.directory / "candidate-create-intent.json", create_record)
        journal.update(phase="PREPARED", parity=evidence)
        return journal
    except BaseException as error:
        with contextlib.suppress(Exception):
            journal.update(phase="PREFLIGHT_FAILED", failure=type(error).__name__)
        with contextlib.suppress(Exception):
            if (journal.directory / "candidate-create-intent.json").exists():
                reconcile_candidate_create(api, journal, error)
        raise


def verify_receipt(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise Failure("evidence path and digest required")
    path = Path(reference["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise Failure("invalid bounded evidence file")
    if not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]):
        raise Failure("invalid evidence digest")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != reference["sha256"]:
        raise Failure("evidence digest mismatch")
    return digest


def consume_authorization(journal, approval):
    state = journal.read()
    if state["phase"] != "PREPARED" or state["authorization_consumed"]:
        raise Failure("stale or already consumed attempt")
    if (approval.get("action") != "one-controlled-swap" or
            approval.get("approved") is not True or
            approval.get("attempt_id") != state["attempt_id"] or
            approval.get("original_id") != state["original_id"] or
            approval.get("original_image_id") != state["original_image_id"] or
            approval.get("candidate_id") != state["candidate_id"] or
            approval.get("candidate_image_id") != state["candidate_image_id"] or
            not time.time() < approval.get("expires_at", 0) <= time.time() + 3600):
        raise Failure("fresh exact one-attempt approval required")
    if state["plan"]["mode"] == "production":
        if approval.get("ingress_refresh") != "unchanged-config-nginx-graceful-reload":
            raise Failure("explicit unchanged-config ingress reload approval required")
        for gate in ("offline", "rehearsal", "source_attestation"):
            verify_receipt(approval.get(gate))
    private_json(journal.directory / "consumed-authorization.json", approval, exclusive=True)
    journal.update(authorization_consumed=True, authorized_at=time.time(), phase="AUTHORIZED")


class Services:
    def __init__(self, original, plan, *, api=None, journal=None):
        from .docker_config import extract_api_key
        self.key, self.plan = extract_api_key(original), plan
        self.original, self.journal = original, journal
        self.api = api or DockerAPI(plan.get("docker_socket", "/var/run/docker.sock"))

    def pre_mutation_admission(self, state):
        if self.plan["mode"] == "rehearsal":
            return {"ok": True, "scope": "disposable_faults_only_not_gate8"}
        validator = self.validator(state)
        admission = getattr(validator, "pre_mutation_admission", None)
        if not callable(admission):
            return {"ok": False, "code": "validated_native_latent_proof_adapter_not_implemented"}
        result = admission()
        if not isinstance(result, dict) or result.get("ok") is not True:
            return {"ok": False, "code": "native_latent_validation_unavailable"}
        return result

    def health(self):
        from .service_checks import check_health
        return check_health(self.plan["base_url"], self.key, timeout=min(5, self.plan["request_timeout_s"]))

    def occ3_identity(self):
        from .service_checks import check_occ3_identity
        return check_occ3_identity(self.plan["base_url"], self.key,
                                   timeout=min(15, self.plan["request_timeout_s"]))

    def chat(self):
        from .service_checks import check_chat
        return check_chat(self.plan["base_url"], self.key, self.plan["model"], timeout=self.plan["request_timeout_s"])

    def tools(self):
        from .service_checks import check_tools
        return check_tools(self.plan["base_url"], self.key, self.plan["model"], timeout=self.plan["request_timeout_s"])

    def ingress(self):
        from .service_checks import check_chat, check_ingress
        if self.plan["mode"] == "rehearsal":
            proof = check_chat(self.plan["ingress_base_url"], self.key, self.plan["model"],
                               timeout=self.plan["request_timeout_s"])
            return {"ok": proof["ok"], "scope": "disposable_fixture_only", "chat": proof}
        requester = IngressRequester(self.api, self.original, self.plan, self.journal)
        cleanups = []
        def observe_service(*args):
            # Private HTTP evidence is evaluated separately from helper ownership.
            # Only the aggregate below is a service/deployment gate.
            response, cleanup = requester.service_response(*args)
            cleanups.append(cleanup)
            return response
        proof = check_ingress(self.plan["ingress_base_url"], self.key, self.plan["model"],
                              timeout=self.plan["request_timeout_s"], requester=observe_service, route="neo-k3-ingress")
        proof["service_ok"] = proof.get("ok") is True
        proof["cleanup_complete"] = all(item.get("removed") is True for item in cleanups)
        proof["helper_cleanup"] = cleanups
        proof["ok"] = proof["service_ok"] and proof["cleanup_complete"]
        if not proof["cleanup_complete"]:
            proof["error_code"] = "ingress_helper_cleanup_pending"
        proof["route_container_id"] = self.plan.get("ingress", {}).get("container_id")
        proof["route_network_id"] = self.plan.get("ingress", {}).get("network_id")
        return proof

    def validator(self, state):
        """Explicit reviewed plan-key selection; absent key stays blocked."""
        from .live_validation import LiveValidator
        selected = self.plan.get("validator")
        if not selected:
            return LiveValidator(self.plan["base_url"], self.key, self.plan["model"],
                                 timeout=self.plan["request_timeout_s"],
                                 attempt_id=state["attempt_id"], candidate_id=state["candidate_id"])
        from .native_validator import (NativeHandoffValidator, UnselectedValidator,
                                       VALIDATOR_KEY)
        if selected == VALIDATOR_KEY:
            if self.journal is None:
                return UnselectedValidator(selected, code="validator_journal_unavailable")
            return NativeHandoffValidator(self.plan, self.journal, state,
                                          timeout=self.plan["validation_timeout_s"])
        return UnselectedValidator(selected)


def candidate_allowed_names(journal, state):
    """Journal staging names plus this attempt's own nonce-bound create name."""
    names = {state["candidate_name"], state["original_name"], state["failed_name"]}
    with contextlib.suppress(Exception):
        record = read_json(journal.directory / "candidate-create-intent.json")
        if (record.get("final_name") == state["candidate_name"] and
                record.get("name") == candidate_create_name(state["candidate_name"], record.get("resource_nonce"))):
            names.add(record["name"])
    return names


def recover(api, journal, *, services=None, state=None, original=None, reason="failure"):
    """Reconcile uncertain calls by immutable ID under a new, independent budget.

    Only idempotent restoration operations may be retried. Candidate deployment
    is never retried. Transport loss after a successful Docker operation is
    resolved by inspecting its effect, not by assuming that it failed.
    """
    from .docker_config import assert_candidate_parity
    state, original = state or journal.read(), original or journal.original()
    plan = state["plan"]
    services = services or Services(original, plan, api=api, journal=journal)
    started = time.monotonic()
    deadline = started + plan["rollback_timeout_s"]
    old_deadline = getattr(api, "deadline", None)
    api.deadline = deadline
    result = {"result": "RESTORE_FAILED", "reason": reason, "started_monotonic": started,
              "verified": False, "service_restored": False, "cleanup_complete": False, "helpers_settled": False}

    def remaining():
        if time.monotonic() >= deadline:
            raise Deadline("independent recovery deadline exceeded")

    def pause():
        remaining()
        time.sleep(min(plan["poll_interval_s"], max(0, deadline - time.monotonic())))

    def inspect(identity, expected_image, *, missing_ok=False):
        while True:
            remaining()
            try:
                return checked_inspect(api, identity, expected_image)
            except DockerError as error:
                if error.status == 404:
                    if missing_ok:
                        return None
                    raise Failure("immutable recovery container is missing") from None
                pause()
            except OSError:
                pause()

    def evidence_write(**changes):
        # Disk-full/journal errors must not prevent recovery by captured IDs.
        with contextlib.suppress(Exception):
            journal.update(**changes)

    def reconcile(operation, action, identity, expected_image, predicate):
        while True:
            document = inspect(identity, expected_image)
            if predicate(document):
                return document
            remaining()
            evidence_write(operation=operation)
            try:
                action()
            except (DockerError, OSError):
                pass
            document = inspect(identity, expected_image)
            if predicate(document):
                return document
            pause()

    def cleanup_snapshot(*, initial=False):
        prior_deadline = api.deadline
        if initial:
            # Container restoration has priority over helper cleanup.
            api.deadline = min(deadline, time.monotonic() + min(5, plan["rollback_timeout_s"] / 10))
        try:
            outcomes = reconcile_helpers(api, journal)
        except Exception as error:
            outcomes = [{"removed": False, "error": type(error).__name__}]
        finally:
            api.deadline = prior_deadline
        result["helper_cleanup"] = outcomes
        try:
            operations = reconcile_helper_operations(journal)
        except Exception as error:
            operations = [{"settled": False, "error": type(error).__name__}]
        result["helper_operations"] = operations
        result["helper_operations_settled"] = helper_operations_settled(operations)
        result["helpers_settled"] = (helpers_settled(outcomes) and result["helper_operations_settled"] and
                                     "operation_fence_error" not in result)
        result["cleanup_complete"] = result["helpers_settled"]

    try:
        with FileLock(journal.directory / "recovery.lock"):
            previous_scope = getattr(_HELPER_CONTEXT, "scope", None)
            recovery_scope = None
            try:
                try:
                    recovery_scope = rotate_helper_admission(journal, "recovery", attempt_id=state["attempt_id"])
                    _HELPER_CONTEXT.scope = recovery_scope
                except Exception as error:
                    result["operation_fence_error"] = type(error).__name__
                current = inspect(state["original_id"], state["original_image_id"])
                assert_candidate_parity(original, current, state["original_image_id"])
                current_name = current.get("Name", "").lstrip("/")
                if current_name not in (state["original_name"], state["preserved_name"]):
                    raise Failure("original name changed outside attempt")
                untouched = current_name == state["original_name"] and is_running(current)
                candidate = None
                if state.get("candidate_id"):
                    candidate = inspect(state["candidate_id"], state["candidate_image_id"], missing_ok=True)
                    result["candidate_missing"] = candidate is None
                    if candidate and candidate.get("Name", "").lstrip("/") not in candidate_allowed_names(journal, state):
                        raise Failure("candidate name changed outside attempt")
                observed_mutation = (state.get("authorization_consumed") and
                                     (current_name == state["preserved_name"] or
                                      (candidate and candidate.get("Name") == "/" + state["original_name"])))
                if not state.get("mutation_intent") and not observed_mutation:
                    if not untouched or (candidate and is_running(candidate)):
                        raise Failure("pre-mutation state changed outside attempt")
                    result["result"] = "NOOP"
                else:
                    if candidate:
                        candidate = reconcile("recovery_stop_candidate",
                            lambda: api.stop(state["candidate_id"], seconds=min(plan["stop_timeout_s"], max(1, int(deadline - time.monotonic())))),
                            state["candidate_id"], state["candidate_image_id"],
                            lambda doc: not doc.get("State", {}).get("Running") and not doc.get("State", {}).get("Restarting"))
                        if candidate.get("Name") == "/" + state["original_name"]:
                            reconcile("recovery_rename_candidate",
                                lambda: api.rename(state["candidate_id"], state["failed_name"]),
                                state["candidate_id"], state["candidate_image_id"],
                                lambda doc: doc.get("Name") == "/" + state["failed_name"])
                    current = reconcile("recovery_rename_original",
                        lambda: api.rename(state["original_id"], state["original_name"]),
                        state["original_id"], state["original_image_id"],
                        lambda doc: doc.get("Name") == "/" + state["original_name"])
                    reconcile("recovery_start_original", lambda: api.start(state["original_id"]),
                              state["original_id"], state["original_image_id"], is_running)
                    result["result"] = "RESTORED"
                service_result = result["result"]
                cleanup_snapshot(initial=True)
                # A running original is not yet a restored service.
                while True:
                    wait_ready(api, state["original_id"], state["original_image_id"], services.health,
                               deadline, interval=plan["poll_interval_s"])
                    if result["result"] == "RESTORED":
                        while True:
                            try:
                                refresh_ingress(api, journal, state, deadline, recovery=True)
                                break
                            except (DockerError, OSError):
                                pause()
                    verified = True
                    for name in ("chat", "ingress"):
                        proof = exit_aware(api, state["original_id"], state["original_image_id"],
                                           getattr(services, name), deadline, interval=plan["poll_interval_s"])
                        result.setdefault("service_proofs", {})[name] = proof
                        service_ok = (proof.get("service_ok", proof.get("ok")) if name == "ingress" else proof.get("ok")) if isinstance(proof, dict) else False
                        if service_ok is not True:
                            verified = False
                            break
                    if verified:
                        break
                    cleanup_snapshot(initial=True)
                    if not result["helpers_settled"]:
                        # Do not create repeated ingress clients behind unresolved
                        # ownership/cleanup. Leave exact records for later recovery.
                        raise Failure("service proof failed with helper cleanup pending")
                    pause()
                current = inspect(state["original_id"], state["original_image_id"])
                assert_candidate_parity(original, current, state["original_image_id"])
                if not is_running(current) or current.get("Name") != "/" + state["original_name"]:
                    raise Failure("restored service identity/state mismatch")
                final_candidate = (inspect(state["candidate_id"], state["candidate_image_id"], missing_ok=True)
                                   if state.get("candidate_id") else None)
                if final_candidate and final_candidate.get("State", {}).get("Running"):
                    raise Failure("candidate still running after restoration")
                result.update(service_restored=True, service_result=service_result)
            finally:
                try:
                    rotate_helper_admission(journal, "closed", attempt_id=state["attempt_id"])
                except Exception as error:
                    result["operation_fence_error"] = type(error).__name__
                _HELPER_CONTEXT.scope = previous_scope
                cleanup_snapshot()
            result["verified"] = result["service_restored"] and result["helpers_settled"]
            result["result"] = result["service_result"] if result["verified"] else "RESTORE_FAILED"
            if not result["helpers_settled"]:
                result["error"] = "helper_cleanup_pending"
    except BaseException as error:
        result.update(result="RESTORE_FAILED", verified=False, error=type(error).__name__)
    finally:
        api.deadline = old_deadline
    result["duration_s"] = time.monotonic() - started
    evidence_write(phase=result["result"], recovery=result)
    with contextlib.suppress(Exception):
        private_json(journal.directory / ("recovery-evidence-" + str(time.monotonic_ns()) + ".json"), result, exclusive=True)
        private_json(journal.directory / "recovery-result.json", result)
    return result


_INGRESS_SCRIPT = r"""
import json, os, threading, urllib.request, urllib.error
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
spec = json.loads(os.environ["OCC3_PRIVATE_REQUEST"])
watchdog = threading.Timer(float(spec["timeout"]) + 2, lambda: os._exit(124))
watchdog.daemon = True
watchdog.start()
headers = {"Content-Type": "application/json", "Accept": "application/json"}
if spec["key"] is not None:
    headers["Authorization"] = "Bearer " + spec["key"]
body = json.dumps(spec["payload"]).encode() if spec["payload"] is not None else None
try:
    req = urllib.request.Request(spec["url"], data=body, headers=headers, method=spec["method"])
    try:
        response = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(req, timeout=spec["timeout"])
    except urllib.error.HTTPError as error:
        response = error
    with response:
        status = response.code
        raw = response.read(262145)
    if len(raw) > 262144:
        result = {"status": status, "error_code": "response_too_large"}
    else:
        result = {"status": status, "body": json.loads(raw) if raw.strip() else None}
except Exception:
    result = {"status": None, "error_code": "ingress_transport"}
print(json.dumps(result, separators=(",", ":")))
watchdog.cancel()
"""


def _environment_map(values):
    if not isinstance(values, list) or len(values) > 256:
        raise Failure("invalid helper environment")
    result = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value or "\x00" in value:
            raise Failure("invalid helper environment")
        name, content = value.split("=", 1)
        if not name or name in result:
            raise Failure("duplicate helper environment")
        result[name] = content
    return result


def helper_configuration_digest(config, host, mounts, image, name):
    # Exact request-relevant configuration; dynamic Engine IDs/addresses are excluded.
    selected_host = {key: host.get(key) for key in (
        "Runtime", "ReadonlyRootfs", "NetworkMode", "IpcMode", "PidMode")}
    selected_host.update({key: host.get(key) or [] for key in (
        "Devices", "DeviceRequests", "DeviceCgroupRules", "CapAdd", "CapDrop", "Mounts", "Binds",
        "VolumesFrom", "ExtraHosts", "Dns", "DnsOptions", "DnsSearch")})
    selected_host.update({key: bool(host.get(key)) for key in ("Privileged", "AutoRemove", "PublishAllPorts")})
    selected_host["PortBindings"] = host.get("PortBindings") or {}
    selected_host["RestartPolicy"] = host.get("RestartPolicy")
    security = host.get("SecurityOpt") or []
    selected_host["SecurityOpt"] = ["no-new-privileges" if value == "no-new-privileges:true" else value for value in security]
    projection = {"image": image, "name": name, "mounts": mounts, "host": selected_host,
                  "entrypoint": config.get("Entrypoint"), "cmd": config.get("Cmd"),
                  "env": _environment_map(config.get("Env")), "user": config.get("User"),
                  "workdir": config.get("WorkingDir"), "labels": config.get("Labels") or {},
                  "healthcheck": config.get("Healthcheck", {}).get("Test"), "tty": bool(config.get("Tty"))}
    return hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def helper_intent(journal, image, name, nonce, body, image_document, *, operation_nonce, generation):
    directory = journal.directory / "helpers"
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink():
        raise Failure("unsafe helper journal directory")
    if len(list(directory.glob("*.json"))) >= 128:
        raise Failure("helper journal limit exceeded")
    path = directory / (nonce + ".json")
    config = dict(body)
    environment = _environment_map(image_document.get("Config", {}).get("Env") or [])
    environment.update(_environment_map(body["Env"]))
    config["Env"] = [key + "=" + value for key, value in environment.items()]
    labels = dict(image_document.get("Config", {}).get("Labels") or {})
    labels.update(body["Labels"])
    config["Labels"] = labels
    record = {"schema": 1, "attempt_id": journal.read()["attempt_id"], "image_id": image_id(image),
              "name": name, "nonce": nonce, "container_id": None,
              "operation_nonce": operation_nonce, "operation_generation": generation,
              "configuration_sha256": helper_configuration_digest(config, body["HostConfig"], [], image, name),
              "image_configuration_sha256": hashlib.sha256(json.dumps(image_document.get("Config", {}), sort_keys=True).encode()).hexdigest(),
              "state": "CREATE_INTENT", "at": time.monotonic()}
    private_json(path, record, exclusive=True)
    return path


def helper_document(api, record):
    identity = record.get("container_id")
    document = api.inspect(identity or record["name"])
    full_id = container_id(document.get("Id"))
    labels = document.get("Config", {}).get("Labels") or {}
    if ((identity and full_id != identity) or document.get("Image") != record["image_id"] or
            document.get("Name") != "/" + record["name"] or
            labels.get("occ3.v3.probe-attempt") != record["attempt_id"] or
            labels.get("occ3.v3.helper-nonce") != record["nonce"]):
        raise Failure("helper ownership/identity mismatch")
    digest = helper_configuration_digest(document.get("Config", {}), document.get("HostConfig", {}),
                                         document.get("Mounts"), document["Image"], record["name"])
    if digest != record.get("configuration_sha256"):
        raise Failure("helper full inspected configuration differs from create intent")
    return document


def reconcile_helper(api, intent_path):
    record = read_json(intent_path)
    if record.get("state") == "REMOVED":
        return {"removed": True, "container_id": record.get("container_id")}
    try:
        try:
            document = helper_document(api, record)
        except DockerError as error:
            if error.status != 404:
                raise
            if not record.get("container_id"):
                # The create may still complete. A name 404 is NOT settlement.
                return {"removed": False, "unresolved_create": True}
            # An immutable acknowledged ID cannot later refer to another container.
            record["state"] = "REMOVED"
            with contextlib.suppress(OSError):
                private_json(intent_path, record)
            return {"removed": True, "container_id": record["container_id"], "already_absent": True}
        identity = document["Id"]
        record.update(container_id=identity, state="CLEANUP_INTENT")
        with contextlib.suppress(OSError):
            private_json(intent_path, record)
        if document.get("State", {}).get("Running"):
            api.stop(identity, seconds=1)
        api.remove(identity)
        record["state"] = "REMOVED"
        record.pop("cleanup_error", None)
        with contextlib.suppress(OSError):
            private_json(intent_path, record)
        return {"removed": True, "container_id": identity}
    except Exception as error:
        record.update(state="CLEANUP_PENDING", cleanup_error=type(error).__name__)
        with contextlib.suppress(OSError):
            private_json(intent_path, record)
        raise


def reconcile_helpers(api, journal):
    directory = journal.directory / "helpers"
    if not directory.exists():
        return []
    if directory.is_symlink():
        raise Failure("unsafe helper journal directory")
    paths = sorted(directory.glob("*.json"))
    if len(paths) > 128:
        raise Failure("helper journal limit exceeded")
    results = []
    for path in paths:
        if getattr(api, "deadline", None) is not None and time.monotonic() >= api.deadline:
            results.append({"removed": False, "error": "cleanup_deadline"})
            break
        try:
            results.append(reconcile_helper(api, path))
        except Exception as error:
            results.append({"removed": False, "error": type(error).__name__})
    return results


def helpers_settled(results):
    return isinstance(results, list) and all(isinstance(item, dict) and item.get("removed") is True for item in results)


def assert_ingress_probe(document, image, body, record, route):
    host, config = document.get("HostConfig", {}), document.get("Config", {})
    if (document.get("Id") != record.get("container_id") or document.get("Image") != record["image_id"] or
            document.get("Name") != "/" + record["name"] or
            document.get("State", {}).get("Status") != "created" or document.get("State", {}).get("Running") or
            document.get("Mounts") != [] or host.get("Mounts") or host.get("Binds") or host.get("VolumesFrom") or
            host.get("Runtime") != "runc" or host.get("Devices") or host.get("DeviceRequests") or
            host.get("DeviceCgroupRules") or host.get("Privileged") or host.get("CapAdd") or
            host.get("ReadonlyRootfs") is not True or host.get("CapDrop") != ["ALL"] or
            host.get("SecurityOpt") not in (["no-new-privileges"], ["no-new-privileges:true"]) or
            host.get("NetworkMode") != route["network_id"] or host.get("PortBindings") or
            host.get("PidMode") not in (None, "") or host.get("IpcMode") != "private" or
            host.get("RestartPolicy", {}).get("Name") != "no" or host.get("AutoRemove") or
            config.get("Entrypoint") != ["python3", "-I", "-S"] or config.get("Cmd") != body["Cmd"] or
            config.get("User") != body["User"] or config.get("WorkingDir") != body["WorkingDir"] or
            config.get("Healthcheck", {}).get("Test") != ["NONE"] or config.get("Tty")):
        raise Failure("actual unstarted ingress helper isolation differs")
    labels = config.get("Labels") or {}
    if labels.get("occ3.v3.helper-nonce") != record["nonce"] or labels.get("occ3.v3.probe-attempt") != record["attempt_id"]:
        raise Failure("actual ingress helper owner differs")
    expected = _environment_map(image.get("Config", {}).get("Env") or [])
    expected.update(_environment_map(body["Env"]))
    actual = _environment_map(config.get("Env"))
    if actual != expected:
        raise Failure("actual ingress helper environment differs")
    for name, value in actual.items():
        if ((name.startswith(("LD_", "DYLD_")) or name in {
                "GCONV_PATH", "BASH_ENV", "ENV", "ZDOTDIR", "PYTHONPATH", "PYTHONHOME",
                "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONINSPECT"}) and value):
            raise Failure("actual ingress helper has executable environment hook")
    if actual.get("NVIDIA_VISIBLE_DEVICES") != "void" or actual.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise Failure("actual ingress helper GPU visibility differs")
    networks = document.get("NetworkSettings", {}).get("Networks", {})
    if set(networks) != {route["network_name"]}:
        raise Failure("actual ingress helper network membership differs")
    if networks[route["network_name"]].get("NetworkID") not in (None, "", route["network_id"]):
        raise Failure("actual ingress helper network identity differs")
    return {"ok": True, "scope": "unstarted_cpu_ingress_helper"}


class IngressRequester:
    """CPU-only temporary client addressing the actual pinned nginx proxy, not the backend."""
    def __init__(self, api, original, plan, journal=None):
        self.api, self.original, self.plan, self.journal = api, original, plan, journal
        self.route = plan.get("ingress", {})

    def __call__(self, method, url, key, payload, timeout):
        from .service_checks import HttpResult
        response, cleanup = self.service_response(method, url, key, payload, timeout)
        if cleanup.get("removed") is not True:
            return HttpResult(None, error_code="ingress_helper_cleanup_pending")
        return response

    def service_response(self, method, url, key, payload, timeout):
        """Private response plus mandatory separate cleanup result, never a gate."""
        from .docker_config import assert_probe_image, probe_environment
        from .service_checks import HttpResult
        route = self.route
        identity, intent, operation = None, None, None
        response = HttpResult(None, error_code="ingress_route_or_probe_failed")
        cleanup = {"removed": True, "created": False}
        try:
            if self.journal is None:
                raise Failure("ingress helper requires durable ownership journal")
            operation = HelperOperation(self.journal)
            operation.check()
            identity = container_id(route.get("container_id"))
            proxy = inspect_ingress(self.api, self.plan)
            base = self.plan["ingress_base_url"].rstrip("/")
            if not url.startswith(base + "/"):
                raise Failure("ingress request must traverse the pinned proxy")
            image = self.api.inspect_image(self.original["Image"])
            assert_probe_image(image, self.original["Image"], self.original)
            nonce = secrets.token_hex(16)
            name = "occ3-v3-" + self.plan["attempt_id"] + "-ingress-" + nonce
            spec = {"method": method, "url": url, "key": key, "payload": payload,
                    "timeout": max(0.1, min(timeout, self.plan["request_timeout_s"]))}
            body = {"Image": self.original["Image"], "Entrypoint": ["python3", "-I", "-S"],
                    "Cmd": ["-c", _INGRESS_SCRIPT], "Env": probe_environment() + [
                        "OCC3_PRIVATE_REQUEST=" + json.dumps(spec, separators=(",", ":"))],
                    "Labels": {"occ3.v3.probe-attempt": self.plan["attempt_id"], "occ3.v3.helper-nonce": nonce},
                    "User": self.original.get("Config", {}).get("User") or "",
                    "WorkingDir": self.original.get("Config", {}).get("WorkingDir") or "/",
                    "Healthcheck": {"Test": ["NONE"]}, "Tty": False,
                    "HostConfig": {"Runtime": "runc", "DeviceRequests": [], "Devices": [],
                                   "DeviceCgroupRules": [], "Privileged": False, "CapAdd": [],
                                   "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                                   "ReadonlyRootfs": True, "IpcMode": "private", "PidMode": "",
                                   "NetworkMode": route["network_id"], "PortBindings": {},
                                   "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0}, "AutoRemove": False},
                    "NetworkingConfig": {"EndpointsConfig": {route["network_id"]: {}}}}
            end = time.monotonic() + timeout
            intent = operation.create_intent(self.original["Image"], name, nonce, body, image)
            operation.check()
            probe = self.api.create(body, name)
            operation.acknowledge("create", probe)
            record = read_json(intent)
            if record.get("container_id") not in (None, probe):
                raise Failure("helper response identity differs from recovery record")
            record.update(container_id=container_id(probe))
            if record.get("state") != "REMOVED":
                record["state"] = "CREATED"
            private_json(intent, record)
            operation.check()
            document = helper_document(self.api, record)
            assert_ingress_probe(document, image, body, record, route)
            record["state"] = "START_INTENT"
            private_json(intent, record)
            operation.start_intent()
            operation.check()
            self.api.start(probe)
            operation.acknowledge("start", probe)
            while time.monotonic() < end:
                operation.check()
                document = helper_document(self.api, record)
                networks = document.get("NetworkSettings", {}).get("Networks", {})
                if networks.get(route["network_name"], {}).get("NetworkID") != route["network_id"]:
                    raise Failure("started ingress helper network identity differs")
                if not document.get("State", {}).get("Running"):
                    if document.get("State", {}).get("ExitCode") != 0:
                        raise Failure("ingress probe exit")
                    data = json.loads(self.api.logs(probe, limit=524288))
                    inspect_ingress(self.api, self.plan)
                    response = HttpResult(data.get("status"), data.get("body"), data.get("error_code"))
                    break
                time.sleep(min(self.plan["poll_interval_s"], max(0, end - time.monotonic())))
            else:
                response = HttpResult(None, error_code="ingress_deadline")
        except Exception:
            response = HttpResult(None, error_code="ingress_route_or_probe_failed")
        finally:
            if intent:
                try:
                    cleanup = reconcile_helper(self.api, intent)
                except Exception as error:
                    cleanup = {"removed": False, "error": type(error).__name__}
            if operation:
                try:
                    operation.finish()
                    operation_outcomes = reconcile_helper_operations(self.journal)
                    if not helper_operations_settled(operation_outcomes):
                        cleanup = dict(cleanup, removed=False, operation_pending=True)
                except Exception as error:
                    cleanup = dict(cleanup, removed=False, operation_error=type(error).__name__)
        return response, cleanup
