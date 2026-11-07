"""Tensor-free backend records. GPU objects live only in latent_worker/registry."""
from __future__ import annotations

import ast
import math
import inspect
import json
import logging
import threading
from functools import wraps
from contextlib import contextmanager
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any

from .attachment_contract import (CaptureWorkerDescriptor, ConsumeWorkerDescriptor,
                                  canonical, digest, identifier, uint)

METHOD = "statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1"
ALIGNMENT = "union-subspace-procrustes.v1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
_REGISTRY_MODULE_NAME = __package__ + ".native_profile"
_REGISTRY_SOURCE = "native_profile.py"
_REGISTRY_SOURCE_LIMIT = 2 * 1024 * 1024
_METHOD_TABLE = {"method_id", "requires_calibration_artifact"}


# Only source-defined detail tags may leave the worker. No request-derived strings.
_ADMISSION_DETAILS = frozenset((
    'latent_registry.reserve_capture.descriptor_type',
    'latent_registry.reserve_capture.method_deadline',
    'latent_registry.reserve_capture.budget_owner_expiry',
    'latent_registry.reserve_capture.existing_attempt',
    'latent_registry.reserve_capture.tombstone_replay',
    'latent_registry.reserve_capture.record_capacity',
    'latent_registry.reserve_capture.tombstone_capacity',
    'latent_registry.reserve_capture.active_slot',
    'latent_registry.reserve_capture.byte_capacity',
    'latent_worker.admit_request.request_identity',
    'latent_worker.admit_request.capture_reserved',
    'latent_worker.admit_request.consume_leased',
    'latent_worker.admit_request.request_shape',
    'latent_worker.admit_request.embedding_override',
    'latent_worker.admit_request.already_attached',
    'latent_worker.admit_request.capture_inflight',
    'latent_worker.admit_request.consume_eager_inflight',
    'latent_worker.admit_request.marker_attestation',
    'latent_worker.admit_request.prefix_offset',
    'latent_worker.admit_request.consume_capacity',
    'latent_worker.admit_request.sparse_pin',
    'latent_worker.capture_budget.prefix_count',
    'latent_worker.capture_budget.span_policy',
    'latent_worker.capture_budget.workspace_capacity',
    'latent_backend._slot.attempt_slot',
    'latent_backend._reservation.native_identity',
    'latent_backend._reservation.native_bounds',
    'latent_backend.enroll_source.profile_identity',
    'latent_backend.enroll_source.retained_limit',
    'latent_backend.enroll_source.source_request_identity',
    'latent_backend.prepare_source.source_reservation',
))
_ADMISSION_OPERATIONS = frozenset(("reserve_capture", "reserve_consume", "release_lease",
    "retire", "reap", "worker_admit", "enroll_source", "prepare_source",
    "release_unused", "backend_retire", "frontend_charge", "frontend_release",
    "terminal_release", "worker_abort", "worker_control", "finish_source", "quarantine", "terminal_gate", "terminal_resolve"))
_ADMISSION_LOCK = threading.Lock()
_ADMISSION_COUNTS = {}
_ADMISSION_LOGS = 0
_ADMISSION_LAST = {}


def admission_diagnostics_snapshot():
    """Process-local counters only; not a cross-rank status or proof of release."""
    try:
        if os.environ.get("OCC3_DIAG_TOKEN_TRACE") != "1":
            return {}
        with _ADMISSION_LOCK:
            return {"counts": dict(_ADMISSION_COUNTS), "logged": _ADMISSION_LOGS,
                    "last": {k: dict(v) for k, v in _ADMISSION_LAST.items()}}
    except Exception:
        return {}


def admission_event(operation, outcome, *, registry=None, error=None, owner=None):
    """Bounded scalar-only diagnostics; failures never alter admission/cleanup."""
    global _ADMISSION_LOGS
    try:
        if os.environ.get("OCC3_DIAG_TOKEN_TRACE") != "1":
            return
        if operation not in _ADMISSION_OPERATIONS or outcome not in ("entry", "exit", "error"):
            return
        row = {"operation": operation, "outcome": outcome}
        if isinstance(error, LatentError):
            row.update(code=error.code, detail=error.detail)
        if registry is not None:
            with registry.lock:
                records = tuple(registry.records.values())
                row.update(active=sum(not r.revoked and r.state in
                    ("reserved", "capturing", "leased", "injected") for r in records),
                    max_active=registry.profile.max_active, total=len(records),
                    max_records=registry.profile.max_records,
                    inflight=sum(r.inflight for r in records),
                    reservation=sum(not r.revoked and r.state == "reserved" for r in records),
                    charged_bytes=registry.charged_bytes,
                    max_live_bytes=registry.profile.max_live_bytes,
                    revoked=sum(r.revoked for r in records),
                    gpu_freed=sum(r.gpu_freed for r in records),
                    retained_rows=sum(len(r.retained) for r in records),
                    admitted_capture=sum(r.admitted_capture for r in records),
                    admitted_consume=sum(r.admitted_consume for r in records),
                    leased=sum(not r.revoked and r.state == "leased" for r in records),
                    capturing=sum(not r.revoked and r.state == "capturing" for r in records))
        if owner is not None:
            if hasattr(owner, "_active"):
                row.update(active=owner._active, max_active=owner.profile.max_active,
                           terminal_active=owner._terminal_active,
                           terminal_attempts=len(owner._terminal_attempts),
                           quarantined=len(getattr(owner, "_quarantined", ())),
                           terminal_tasks=len(getattr(owner, "_terminal_tasks", ())))
            if hasattr(owner, "attempts"):
                row["attempts"] = len(owner.attempts)
            control = getattr(owner, "client", None)
            if hasattr(control, "pending"):
                row.update(pending=len(control.pending), uncertain=len(control._uncertain),
                           operations=len(control._operations), max_pending=control.max_pending)
            if hasattr(owner, "poisoned"):
                row["poisoned"] = bool(owner.poisoned)
                row["commands"] = len(owner.commands)
        with _ADMISSION_LOCK:
            _ADMISSION_LAST[operation] = dict(row)
            key = operation + "." + outcome
            _ADMISSION_COUNTS[key] = min(2**63 - 1, _ADMISSION_COUNTS.get(key, 0) + 1)
            _ADMISSION_LOGS = min(2**63 - 1, _ADMISSION_LOGS + 1)
        logging.getLogger("occ3_sglang.admission").info(
            "occ3-bridge: admission %s", json.dumps(row, separators=(",", ":")))
    except Exception:
        pass


def admission_probe(operation, *, registry=False):
    """Count entries and every exit, including cancellation, without inspecting args."""
    def decorate(function):
        def event(args, outcome, error=None):
            try:
                owner = args[0]
                runtime = getattr(owner, "occ3_runtime", None)
                owner = runtime if runtime is not None else owner
                reg = owner if registry else getattr(owner, "registry", None)
                admission_event(operation, outcome, registry=reg, error=error, owner=owner)
                if error is not None:
                    from .turn_diagnostics import log_failure
                    log_failure(error, phase="admission")
            except Exception:
                pass
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def asynchronous(*args, **kwargs):
                event(args, "entry")
                try:
                    return await function(*args, **kwargs)
                except BaseException as error:
                    event(args, "error", error)
                    raise
                finally:
                    event(args, "exit")
            return asynchronous
        @wraps(function)
        def synchronous(*args, **kwargs):
            event(args, "entry")
            try:
                return function(*args, **kwargs)
            except BaseException as error:
                event(args, "error", error)
                raise
            finally:
                event(args, "exit")
        return synchronous
    return decorate


_ADMISSION_LOCK_TRACE = os.environ.get("OCC3_DIAG_TOKEN_TRACE") == "1"
_ADMISSION_LOG = logging.getLogger("occ3_sglang.admission")


def admission_lock(registry):
    """Return the original lock unless import-time opt-in and DEBUG are enabled."""
    try:
        if _ADMISSION_LOCK_TRACE and _ADMISSION_LOG.isEnabledFor(logging.DEBUG):
            return _traced_admission_lock(registry)
    except Exception:
        pass
    return registry.lock


def _admission_lock_note(event):
    try:
        _ADMISSION_LOG.debug("occ3-bridge: admission lock %s", event)
    except Exception:
        pass


@contextmanager
def _traced_admission_lock(registry):
    """Observe the existing lock boundary without adding a second lock scope."""
    _admission_lock_note("acquire_wait")
    with registry.lock:
        _admission_lock_note("acquired")
        try:
            yield
        finally:
            _admission_lock_note("releasing")
    _admission_lock_note("released")


class LatentError(RuntimeError):
    def __init__(self, code="latent_failed", members=None, *, detail=None):
        allowed = {"latent_failed", "unavailable", "profile", "identity", "expired",
                   "quota", "replay", "state", "lineage", "ranks", "invalid",
                   "not_ready", "retired", "numerical", "cancelled", "busy"}
        self.code = code if code in allowed else "latent_failed"
        # Bounded sanitised names of the profile members that tripped; a live
        # decline names every failing check in one attempt instead of one.
        if (isinstance(members, (list, tuple)) and members
                and all(isinstance(name, str) and name.isascii()
                        and len(name) <= 32 for name in members)):
            self.members = ",".join(members)[:128]
        else:
            self.members = None
        self.detail = detail if detail in _ADMISSION_DETAILS else None
        super().__init__(self.code)


class UniformRequestCancellation(LatentError):
    """Only emitted after same-request, same-error all-rank preflight."""
    def __init__(self, code, native_rid, attempt_id):
        if code not in ("expired", "retired"):
            raise LatentError("invalid")
        identifier(native_rid)
        identifier(attempt_id)
        self.native_rid, self.attempt_id = native_rid, attempt_id
        super().__init__(code)


def require(condition, code="invalid", *, detail=None):
    if not condition:
        raise LatentError(code, detail=detail)


def require_members(checks, code="profile"):
    """Evaluate every (name, holds) member, then decline naming all failures.

    No short-circuit: a decline carries the bounded member names so one live
    window identifies every tripping profile check at that stage. Member
    conditions are positive, matching require(condition).
    """
    failed = [name for name, holds in checks if not holds]
    if failed:
        raise LatentError(code, members=failed)


def _registry_constants(tree):
    """Module-level constant assignments of one parsed module: name -> value."""
    constants = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (str, int, float, bool))):
            constants[node.targets[0].id] = node.value.value
    return constants


def _registry_value(node, environment):
    """Evaluate one registry expression: a constant, a known name, or a sign."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in environment:
        return environment[node.id]
    if (isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in (int, float)
            and isinstance(node.op, (ast.USub, ast.UAdd))):
        operand = node.operand.value
        return -operand if isinstance(node.op, ast.USub) else operand
    return None


def _parsed_registry_table():
    """The method table parsed from the registry source; no occ3_sglang import.

    The model is the offline static parse the deployment packaging path uses:
    only the registry's own constructs are understood and anything else fails
    closed to None. Used only when the registry module itself cannot be
    imported; a registry that is present but unusable (a stubbed embedded
    host) never reaches this, only the local default stays nameable there.
    """
    try:
        with open(os.path.join(os.path.dirname(__file__), _REGISTRY_SOURCE), "rb") as handle:
            raw = handle.read(_REGISTRY_SOURCE_LIMIT + 1)
    except OSError:
        return None
    if not 0 < len(raw) <= _REGISTRY_SOURCE_LIMIT:
        return None
    try:
        tree = ast.parse(raw.decode("utf-8"), _REGISTRY_SOURCE)
    except (SyntaxError, ValueError, UnicodeError, RecursionError):
        return None
    environment = dict(_registry_constants(tree))
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.module and node.level >= 1
                and re.fullmatch(r"[A-Za-z0-9_.]+", node.module) and "\\" not in node.module
                and "/" not in node.module):
            for alias in node.names:
                if alias.name == "*":
                    return None
                try:
                    sibling = os.path.join(os.path.dirname(__file__),
                                           node.module.split(".")[-1] + ".py")
                    with open(sibling, "rb") as handle:
                        source = handle.read(_REGISTRY_SOURCE_LIMIT + 1)
                    imported = ast.parse(source.decode("utf-8"), node.module)
                except (OSError, SyntaxError, ValueError, UnicodeError, RecursionError):
                    return None
                constants = _registry_constants(imported)
                if alias.name not in constants:
                    return None
                environment[alias.asname or alias.name] = constants[alias.name]
    parameters = entry = None
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_method_entry"):
            arguments = node.args
            if (arguments.posonlyargs or arguments.vararg or arguments.kwonlyargs
                    or arguments.kwarg or arguments.defaults or entry is not None):
                return None
            parameters = [item.arg for item in arguments.args]
            if (len(node.body) != 1 or not isinstance(node.body[0], ast.Return)
                    or not isinstance(node.body[0].value, ast.Dict)):
                return None
            entry = node.body[0].value
    if not parameters or entry is None:
        return None
    table = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.endswith("_METHODS")
                and isinstance(node.value, ast.DictComp) and len(node.value.generators) == 1
                and isinstance(node.value.generators[0].iter, (ast.Tuple, ast.List))):
            for call in node.value.generators[0].iter.elts:
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "_method_entry"):
                    return None
                bound = dict(zip(parameters, call.args))
                for keyword in call.keywords:
                    bound[keyword.arg] = keyword.value
                if len(bound) != len(parameters):
                    return None
                scope = {name: _registry_value(value, environment)
                         for name, value in bound.items()}
                descriptor = {}
                for key, value in zip(entry.keys, entry.values):
                    if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                        return None
                    descriptor[key.value] = _registry_value(value, scope)
                if set(descriptor) < _METHOD_TABLE:
                    return None
                method_id, requires = descriptor["method_id"], descriptor["requires_calibration_artifact"]
                if (type(method_id) is not str or type(requires) is not bool
                        or method_id in table):
                    return None
                table[method_id] = requires
    return table or None


def _registered_table():
    """Method ids the registry holds and whether each requires an artifact.

    native_profile.py owns the one method table this build may name. The
    registry is imported lazily (latent_types must stay importable while
    native_profile imports it). When the registry module is present but
    unusable, or no table can be read at all, only the per-message method
    latent_types itself defines stays nameable, and it requires no artifact.
    """
    try:
        from .native_profile import ProfileError, method_descriptor, registered_methods
    except ImportError:
        if _REGISTRY_MODULE_NAME in sys.modules:
            return {METHOD: False}
        return _parsed_registry_table() or {METHOD: False}
    try:
        return {method_id: method_descriptor(method_id)["requires_calibration_artifact"]
                for method_id in registered_methods()}
    except ProfileError:
        # A registry that cannot describe its own entries is broken, not open.
        return {METHOD: False}


def advertised_method(value):
    """True when value is a method id the native registry registers.

    The registry decides which methods may be negotiated: the per-message
    default below, the frozen dual-ridge method and the registered challengers.
    An embedded host that stubs the registry can name only the local default.
    """
    if type(value) is not str:
        return False
    return value in _registered_table()


def sha(value):
    require(isinstance(value, str) and HEX.fullmatch(value) is not None)
    return value


def bounded_payload(value, *, maximum=16384):
    """Reject arbitrary objects/tensors before any serializer can call enc_hook."""
    def walk(v, depth=0):
        require(depth <= 12)
        if v is None or type(v) in (bool, str, int):
            if type(v) is int:
                require(-(2**63) <= v < 2**64)
            if isinstance(v, str):
                require(len(v) <= 1024)
            return
        if type(v) is float:
            require(math.isfinite(v))
            return
        if type(v) is dict:
            require(len(v) <= 64 and all(type(k) is str for k in v))
            for child in v.values():
                walk(child, depth + 1)
            return
        if type(v) in (list, tuple):
            require(len(v) <= 128)
            for child in v:
                walk(child, depth + 1)
            return
        raise LatentError("invalid")
    walk(value)
    require(len(canonical(value)) <= maximum, "quota")
    return value


def decode_descriptor(value):
    if value is None:
        return None
    from .attachment_contract import normalize_worker_descriptor
    return normalize_worker_descriptor(value)


@dataclass(frozen=True)
class RuntimeProfile:
    runtime_epoch: str
    runtime_generation: int
    model_fingerprint: str
    model_id: str
    expected_ranks: int = 8
    hidden_dim: int = 6144
    layer_index: int = 77
    max_prefix_tokens: int = 128
    max_live_bytes: int = 256 * 1024 * 1024
    workspace_bytes: int = 128 * 1024 * 1024
    max_records: int = 256
    max_active: int = 1
    max_ttl_seconds: int = 3600
    anchor_alpha: float = 0.0
    tile_rows: int = 256
    # A registered native method id. The default is the per-message method, so
    # an unpinned caller is unchanged; the frozen calibration methods are
    # different ids with their own artifacts, never a reinterpretation of this
    # one. The registry decides which ids may be named here.
    method: str = METHOD
    # The frozen calibration artifact this profile is bound to: an absolute path
    # and the artifact_digest that the file at that path must contain. Both are
    # empty for the per-message method, which consumes no artifact. The worker
    # resolves the pair through the method's own deserializer (registry
    # dispatch) before any aligner exists, so a mismatched file is never loaded.
    calibration_path: str = ""
    calibration_digest: str = ""

    def __post_init__(self):
        for key in ("runtime_epoch", "model_fingerprint", "model_id"):
            identifier(getattr(self, key))
        uint(self.runtime_generation)
        for key in ("expected_ranks", "hidden_dim", "layer_index", "max_prefix_tokens",
                    "max_live_bytes", "workspace_bytes", "max_records", "max_active",
                    "max_ttl_seconds", "tile_rows"):
            uint(getattr(self, key))
        require(type(self.anchor_alpha) in (int, float) and math.isfinite(self.anchor_alpha), "profile")
        require(self.expected_ranks == 8 and self.hidden_dim == 6144 and self.layer_index == 77, "profile")
        require(self.max_prefix_tokens in (8, 16, 32, 64, 128)
                and advertised_method(self.method), "profile")
        for name in ("calibration_path", "calibration_digest"):
            require(type(getattr(self, name)) is str, "profile")
        # A reference is all-or-nothing, absolute and digest-pinned; the
        # per-message method consumes no artifact, so it may not carry one.
        require(bool(self.calibration_path) == bool(self.calibration_digest), "profile")
        if self.calibration_path:
            require(0 < len(self.calibration_path) <= 512
                    and "\0" not in self.calibration_path
                    and os.path.isabs(self.calibration_path), "profile")
            require(HEX.fullmatch(self.calibration_digest) is not None, "identity")
            # Registry-driven, method-agnostic: a reference may only be carried
            # by a method whose registry entry requires a calibration artifact.
            require(_registered_table().get(self.method, False), "profile")
        require(self.max_active == 1 and 1 <= self.max_records <= 4096, "profile")
        require(0 < self.max_live_bytes <= 2**30 and 0 < self.workspace_bytes <= 2**30, "profile")
        require(self.anchor_alpha in (0.0, 0.3) and 1 <= self.tile_rows <= 4096, "profile")
        require(0 < self.max_ttl_seconds <= 3600, "profile")

    @property
    def binding_digest(self):
        return digest(asdict(self))


@dataclass(frozen=True, repr=False)
class RequestSlice:
    descriptor: CaptureWorkerDescriptor | ConsumeWorkerDescriptor
    extend_start: int
    extend_len: int
    prefix_len: int
    output_start: int
    final_chunk: bool

    def __post_init__(self):
        for x in (self.extend_start, self.extend_len, self.prefix_len, self.output_start):
            require(type(x) is int and x >= 0)
        require(type(self.final_chunk) is bool)


@dataclass(frozen=True, repr=False)
class SessionEnrollment:
    principal_digest: str
    session_id: str
    model_fingerprint: str
    runtime_epoch: str
    runtime_generation: int
    access_domain: str
    expires_at_epoch: int

    def __post_init__(self):
        sha(self.principal_digest)
        for v in (self.session_id, self.model_fingerprint, self.runtime_epoch, self.access_domain):
            identifier(v)
        uint(self.runtime_generation)
        uint(self.expires_at_epoch)


#: How a publication was dispatched. The default keeps the legacy same-turn
#: model tool protocol; orchestrator_v1 is the retained-capture mode where
#: the orchestrator publishes a no-tool content turn.
DISPATCH_MODEL_TOOL_V1 = "model_tool_v1"
DISPATCH_ORCHESTRATOR_V1 = "orchestrator_v1"


@dataclass(frozen=True, repr=False)
class Publication:
    publication_id: str
    source_handle: str
    principal_digest: str
    source_session_id: str
    receiver_session_id: str
    message_id: str
    source_turn_nonce: str
    tool_call_id: str
    expires_at_epoch: int
    dispatch_mode: str = DISPATCH_MODEL_TOOL_V1

    def __post_init__(self):
        sha(self.principal_digest)
        require(self.dispatch_mode in (DISPATCH_MODEL_TOOL_V1, DISPATCH_ORCHESTRATOR_V1), "invalid")
        for k, v in asdict(self).items():
            if k in ("principal_digest", "expires_at_epoch"):
                continue
            if k == "tool_call_id" and self.dispatch_mode == DISPATCH_ORCHESTRATOR_V1:
                # Orchestrator dispatch names no tool call: the field must be
                # the empty string, and only in this mode.
                require(type(v) is str and v == "", "invalid")
                continue
            identifier(v)
        uint(self.expires_at_epoch)

    @property
    def binding_digest(self):
        # Field-consistent with the schema the client used: the legacy digest
        # is over the nine v1 fields; the v2 digest swaps tool_call_id for
        # dispatch_mode. No mode is ever defaulted onto the other field set.
        if self.dispatch_mode == DISPATCH_ORCHESTRATOR_V1:
            return digest({k: v for k, v in asdict(self).items() if k != "tool_call_id"})
        return digest({k: v for k, v in asdict(self).items() if k != "dispatch_mode"})


def control_request(command_id, operation, payload):
    identifier(command_id)
    require(operation in {"health", "enroll_session", "reserve_capture", "reserve_consume",
                          "publish", "publication_status", "status", "retire", "bind_output",
                          "release_lease"})
    bounded_payload(payload)
    return {"schema": "occ3.control.v1", "command_id": command_id,
            "operation": operation, "payload": payload}


def parse_control(value):
    bounded_payload(value)
    require(type(value) is dict and set(value) == {"schema", "command_id", "operation", "payload"})
    require(value["schema"] == "occ3.control.v1" and type(value["payload"]) is dict)
    return control_request(value["command_id"], value["operation"], value["payload"])


def validate_control_payload(value, *, result=False):
    if not result:
        return parse_control(value)
    bounded_payload(value)
    require(type(value) is dict and set(value) == {"schema", "command_id", "ok", "result", "error"})
    require(value["schema"] == "occ3.control.result.v1" and type(value["ok"]) is bool)
    identifier(value["command_id"])
    require(type(value["result"]) is dict and type(value["error"]) is str)
    require((value["ok"] and not value["error"]) or (not value["ok"] and not value["result"]))
    return value
