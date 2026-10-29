"""Tensor-free backend records. GPU objects live only in latent_worker/registry."""
from __future__ import annotations

import ast
import math
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


class LatentError(RuntimeError):
    def __init__(self, code="latent_failed", members=None):
        allowed = {"latent_failed", "unavailable", "profile", "identity", "expired",
                   "quota", "replay", "state", "lineage", "ranks", "invalid",
                   "not_ready", "retired", "numerical", "cancelled"}
        self.code = code if code in allowed else "latent_failed"
        # Bounded sanitised names of the profile members that tripped; a live
        # decline names every failing check in one attempt instead of one.
        if (isinstance(members, (list, tuple)) and members
                and all(isinstance(name, str) and name.isascii()
                        and len(name) <= 32 for name in members)):
            self.members = ",".join(members)[:128]
        else:
            self.members = None
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


def require(condition, code="invalid"):
    if not condition:
        raise LatentError(code)


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

    def __post_init__(self):
        sha(self.principal_digest)
        for k, v in asdict(self).items():
            if k not in ("principal_digest", "expires_at_epoch"):
                identifier(v)
        uint(self.expires_at_epoch)

    @property
    def binding_digest(self):
        return digest(asdict(self))


def control_request(command_id, operation, payload):
    identifier(command_id)
    require(operation in {"health", "enroll_session", "reserve_capture", "reserve_consume",
                          "publish", "publication_status", "status", "retire", "bind_output"})
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
