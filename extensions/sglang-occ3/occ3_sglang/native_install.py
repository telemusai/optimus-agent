"""Explicit pre-serving native installation. Ordinary plugin loading stays dormant.

No environment/key/model settings are created here. Only the trusted server
bootstrap may call install_native_runtime; HTTP requests cannot install a backend.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field

from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class InstallationError(Exception):
    """Fixed, non-sensitive installation refusal."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


# Per-boot native turn admission window. Armed capture turns buffer their whole
# response server-side, so a thinking-heavy armed turn can legitimately run
# past the 300s default admission window (live TS13 evidence: the armed decode
# died with deadline_exceeded/504 at ~301s mid-generation while the descriptor
# legitimately declared deadline_seconds=540). Trusted deployments serving
# armed turns opt in per-boot with OCC3_MAX_TURN_SECONDS; the default stays
# 300s for every other boot, explicit caller profiles are never overridden,
# and a declared-but-invalid value fails closed with a fixed refusal code.
_MAX_TURN_DEFAULT_SECONDS = 300.0
_MAX_TURN_BOUNDS_SECONDS = (60.0, 1800.0)


def _max_turn_seconds_from_env():
    """Read the per-boot turn admission window. Trusted bootstrap only.

    Absent -> default. Unparseable or outside [60, 1800] -> InstallationError
    so a mistyped declaration can never silently serve the default window
    (or an unbounded one) to armed turns.
    """
    raw = os.environ.get("OCC3_MAX_TURN_SECONDS")
    if raw is None:
        return _MAX_TURN_DEFAULT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        raise InstallationError("max_turn_seconds_invalid") from None
    if not (math.isfinite(value)
            and _MAX_TURN_BOUNDS_SECONDS[0] <= value <= _MAX_TURN_BOUNDS_SECONDS[1]):
        raise InstallationError("max_turn_seconds_invalid") from None
    return value


@dataclass(repr=False)
class _BrokerGate:
    authority: object
    principal_id: str
    api_key: str = field(repr=False)
    decide: object = field(repr=False)
    normal_level: object = field(repr=False)
    admission_seconds: float = 300.0

    def verified(self, scope):
        # NORMAL permits OPTIONS/no-config/health; none may create a broker grant.
        if (scope.get("method") != "POST" or not isinstance(self.api_key, str)
                or not self.api_key):
            return False
        values = [value for name, value in scope.get("headers", ())
                  if name.lower() == b"authorization"]
        if len(values) != 1:
            return False  # Original serving auth remains authoritative for ordinary traffic.
        try:
            authorization = values[0].decode("latin-1")
            parts = authorization.split(" ", 1)
            if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
                return False
            decision = self.decide(method="POST", path=scope["path"],
                authorization_header=authorization, api_key=self.api_key,
                admin_api_key=None, auth_level=self.normal_level)
            return decision.allowed is True
        except Exception:
            return False


class VerifiedBrokerMiddleware:
    """ASGI-only principal marker. Never reads a body or replaces auth middleware."""
    def __init__(self, app, *, gate):
        self.app, self._gate = app, gate

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        scoped = (path == "/v1/chat/completions" or path.startswith("/occ3/v1/native/"))
        if scope.get("type") != "http" or not scoped or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        # A fresh request-local mapping prevents persistent/shared scope state grants.
        scope = dict(scope)
        scope["state"] = dict(scope.get("state") or {})
        request = Request(scope, receive=receive)
        authority = self._gate.authority
        try:
            authority.clear_verified(request)
            if self._gate.verified(scope):
                authority.mark_verified(request, self._gate.principal_id,
                    deadline_monotonic=time.monotonic() + self._gate.admission_seconds)
        except Exception:
            response = JSONResponse({"error": {"code": "principal_unavailable"}},
                                    status_code=503, headers={"cache-control": "no-store"})
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            # ASGI response completion/disconnect, not handle_request return, owns lifetime.
            authority.clear_verified(request)


def _frontend_detail(error):
    """Bounded frontend-member token, or None for every other decline."""
    members = getattr(error, "frontend_members", None)
    if isinstance(members, str) and members and members.isascii():
        return ("frontend_members=" + members)[:128]
    return None


def _decline_detail(error):
    """Bounded sanitised root-cause token for the generic decline paths.

    Never the exception text: only fixed latent codes, a fixed timeout marker,
    or the bare exception class name. Receipt taxonomies stay unchanged; the
    detail exists so a live decline distinguishes a control-path timeout or a
    worker refusal from unrelated local failures.
    """
    from .latent_types import LatentError
    if isinstance(error, LatentError):
        members = getattr(error, "members", None)
        if isinstance(members, str) and members and members.isascii():
            return ("latent_" + error.code + "[" + members + "]")[:128]
        return "latent_" + error.code
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "control_timeout"
    name = type(error).__name__
    return "exception_" + name if name.isidentifier() else "exception_unknown"


def _existing_auth(app, tm):
    args = getattr(tm, "server_args", None)
    key = getattr(args, "api_key", None)
    if type(key) is not str or not key:
        raise InstallationError("original_api_key_required")
    registrations = []
    for middleware in app.user_middleware:
        cls = middleware.cls
        if (getattr(cls, "__module__", None) == "sglang.srt.utils.auth"
                and getattr(cls, "__qualname__", None)
                == "add_api_key_middleware.<locals>._ApiKeyASGIMiddleware"):
            registrations.append(middleware)
    if len(registrations) != 1:
        raise InstallationError("original_auth_registration_required")
    options = getattr(registrations[0], "kwargs", None)
    try:
        identical = (type(options) is dict and options.get("fastapi_app") is app
                     and type(options.get("api_key")) is str
                     and secrets.compare_digest(key, options["api_key"]))
    except Exception:
        identical = False
    if not identical:
        raise InstallationError("original_auth_mismatch")
    return key


def _verify_control_plane_sources():
    """Attested bytes for the control-plane module only; never executes it."""
    import hashlib
    import os
    import re
    import stat
    from pathlib import Path

    from .native_profile import load_source_attestation

    attestation = load_source_attestation()
    if (type(attestation) is not dict
            or attestation.get("schema") != "optimus.occ3.native-sources.v1"
            or attestation.get("source_revision") != _SOURCE_REVISION
            or attestation.get("sglang_version") != "0.5.21"
            or type(attestation.get("files")) is not list):
        raise InstallationError("source_attestation_invalid")
    selected = None
    for item in attestation["files"]:
        if (type(item) is not dict or set(item) != {"root", "path", "sha256"}
                or item["root"] != "occ3_sglang"):
            continue
        if item["path"] == "startup_state.py":
            if selected is not None:
                raise InstallationError("source_attestation_invalid")
            selected = item
    if selected is None:
        raise InstallationError("source_attestation_incomplete")
    expected = selected["sha256"]
    if type(expected) is not str or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise InstallationError("source_attestation_invalid")
    path = Path(__file__).absolute().parent / "startup_state.py"
    if any(entry.is_symlink() for entry in (path, *path.parents)):
        raise InstallationError("source_symlink")
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_BINARY", 0))
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 < before.st_size <= _MAX_SOURCE_FILE):
            raise InstallationError("source_file_invalid")
        with os.fdopen(fd, "rb") as file:
            fd = None
            data = file.read(_MAX_SOURCE_FILE + 1)
            after = os.fstat(file.fileno())
        signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if (signature(before) != signature(after) or len(data) > _MAX_SOURCE_FILE
                or len(data) != before.st_size):
            raise InstallationError("source_changed")
        if hashlib.sha256(data).hexdigest() != expected:
            raise InstallationError("source_digest_mismatch")
        return {str(path.resolve()): data.decode("utf-8")}
    except InstallationError:
        raise
    except Exception:
        raise InstallationError("source_unreadable") from None
    finally:
        if fd is not None:
            os.close(fd)


def register_control_plane(app, tokenizer_manager):
    """Mount the three control routes; never on unverified original auth.

    Idempotent via the control-plane marker with route-list restore. A
    refusal never blocks ordinary serving: the state stays DISABLED with
    the fixed refusal code and one sanitized ERROR line is logged.
    """
    from .startup_state import (CONTROL_PLANE_ATTR, CONTROL_PLANE_ROUTES,
                                record_control_refusal, record_control_registered)

    attempt = uuid.uuid4().hex[:8]

    def refused(code, detail=None):
        record_control_refusal(code, attempt=attempt)
        if detail is None:
            logger.error("occ3-bridge: control plane registration decline "
                         "(attempt=%s stage=prepare code=%s)", attempt, code)
        else:
            logger.error("occ3-bridge: control plane registration decline "
                         "(attempt=%s stage=prepare code=%s detail=%s)",
                         attempt, code, detail)
        return False

    try:
        _existing_auth(app, tokenizer_manager)
    except InstallationError as error:
        return refused(error.code)
    if getattr(app, CONTROL_PLANE_ATTR, False):
        record_control_registered()
        return True
    previous_routes = list(app.router.routes)
    try:
        from .startup_state import create_control_router
        router = create_control_router()
        actual = sorted((getattr(route, "path", None),
                         tuple(sorted(getattr(route, "methods", None) or ())))
                        for route in router.routes)
        if actual != sorted(CONTROL_PLANE_ROUTES):
            raise InstallationError("native_router_unverified")
        documents = _verify_control_plane_sources()
        for route in router.routes:
            _verify_function_provenance(route.endpoint, documents)
        app.include_router(router)
    except InstallationError as error:
        app.router.routes[:] = previous_routes
        return refused(error.code)
    except Exception as error:
        app.router.routes[:] = previous_routes
        return refused("experimental_startup_refused", _decline_detail(error))
    setattr(app, CONTROL_PLANE_ATTR, True)
    record_control_registered()
    return True


@dataclass(frozen=True)
class InstallationReceipt:
    installed: bool
    code: str
    source_ready: bool = False
    receiver_ready: bool = False
    runtime_id: str | None = None
    runtime_generation: int | None = None
    model_fingerprint: str | None = None

    def evidence(self):
        return {"installed": self.installed, "code": self.code,
                "source_ready": self.source_ready, "receiver_ready": self.receiver_ready,
                "runtime_id": self.runtime_id, "runtime_generation": self.runtime_generation,
                "model_fingerprint": self.model_fingerprint, "supported": False,
                "live_proof": False, "gate8_pass": False}


@dataclass(repr=False)
class NativeInstallation:
    receipt: InstallationReceipt
    app: object = field(repr=False)
    tm: object = field(repr=False)
    backend: object = field(repr=False)
    authority: object = field(repr=False)
    adapter: object = field(repr=False)
    principal_id: str = field(repr=False)
    hooks: tuple = field(default=(), repr=False)
    registry: object = field(default=None, repr=False)


_install_lock = threading.Lock()
_INSTALLED_ATTR = "_occ3_native_installation"


_SOURCE_REVISION = "e00930c5489053f26d86b179cee0d087f846acbb"
_MAX_SOURCE_FILE = 2 * 1024 * 1024
_MAX_SOURCE_TOTAL = 32 * 1024 * 1024
_REQUIRED_SGLANG = {
    "utils/auth.py", "plugins/hook_registry.py", "managers/io_struct.py",
    "managers/tokenizer_manager.py", "managers/scheduler.py",
    "managers/schedule_batch.py", "entrypoints/openai/serving_base.py",
    "entrypoints/openai/serving_chat.py", "entrypoints/openai/protocol.py",
    "entrypoints/http_server.py", "models/glm4_moe.py", "models/deepseek_v2.py",
    "models/deepseek_nextn.py", "layers/logits_processor.py",
    "speculative/eagle_worker_common.py", "speculative/eagle_worker_v2.py",
    "model_executor/model_runner.py", "model_executor/forward_batch_info.py",
    "model_executor/runner/decode_cuda_graph_runner.py",
    "model_executor/runner/prefill_cuda_graph_runner.py",
    "speculative/eagle_draft_cuda_graph_runner.py",
    "speculative/eagle_draft_extend_cuda_graph_runner.py",
    "managers/utils.py", "managers/schedule_policy.py",
    "managers/scheduler_components/request_receiver.py",
    "managers/scheduler_components/batch_result_processor.py", "mem_cache/common.py",
}
_REQUIRED_PLUGIN = {
    "native_install.py", "native_profile.py", "plugin.py", "chat_adapter.py",
    "attachment_contract.py", "use_receipts.py", "latent_backend.py",
    "latent_control_api.py", "latent_worker.py", "latent_types.py",
    "latent_registry.py", "statebridge.py", "startup_state.py",
    # The independent-calibration methods and their evaluation surface are part
    # of the installed byte identity: an unattested copy could be swapped under
    # a profile that names a calibrated method id.
    "statebridge_calibrated.py", "statebridge_calibrated_procrustes.py",
    "calibration_corpus.py", "calibration_probe.py", "alignment_dispatch.py",
    "align_diagnostics.py",
}


def _verify_installed_sources(attestation, roots):
    """The packaged digest authenticates expectations; bounded reads check reality."""
    import hashlib
    import os
    import re
    import stat
    from pathlib import Path, PurePosixPath

    if (type(attestation) is not dict
            or set(attestation) != {"schema", "source_revision", "sglang_version", "manifests", "files"}
            or attestation["schema"] != "optimus.occ3.native-sources.v1"
            or attestation["source_revision"] != _SOURCE_REVISION
            or attestation["sglang_version"] != "0.5.21"
            or type(attestation["files"]) is not list
            or not 1 <= len(attestation["files"]) <= 128):
        raise InstallationError("source_attestation_invalid")
    documents, identities, total = {}, set(), 0
    for item in attestation["files"]:
        if type(item) is not dict or set(item) != {"root", "path", "sha256"}:
            raise InstallationError("source_attestation_invalid")
        root_name, relative, expected = item["root"], item["path"], item["sha256"]
        if (root_name not in roots or type(relative) is not str or len(relative) > 180
                or not re.fullmatch(r"[A-Za-z0-9_/-]+[.]py", relative)
                or type(expected) is not str or not re.fullmatch(r"[a-f0-9]{64}", expected)):
            raise InstallationError("source_attestation_invalid")
        parts = PurePosixPath(relative).parts
        if (relative.startswith("/") or any(x in (".", "..") for x in parts)
                or str(PurePosixPath(relative)) != relative or (root_name, relative) in identities):
            raise InstallationError("source_attestation_invalid")
        identities.add((root_name, relative))
        root = Path(os.path.abspath(roots[root_name]))
        target = root.joinpath(*parts)
        if any(p.is_symlink() for p in [target, *target.parents]):
            raise InstallationError("source_symlink")
        if not target.resolve().is_relative_to(root):
            raise InstallationError("source_attestation_invalid")
        fd = None
        try:
            fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or not 0 <= before.st_size <= _MAX_SOURCE_FILE):
                raise InstallationError("source_file_invalid")
            with os.fdopen(fd, "rb") as file:
                fd = None
                data = file.read(_MAX_SOURCE_FILE + 1)
                after = os.fstat(file.fileno())
            signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            if signature(before) != signature(after) or len(data) > _MAX_SOURCE_FILE:
                raise InstallationError("source_changed")
            total += len(data)
            if total > _MAX_SOURCE_TOTAL or hashlib.sha256(data).hexdigest() != expected:
                raise InstallationError("source_digest_mismatch")
            documents[str(target.resolve())] = data.decode("utf-8")
        except InstallationError:
            raise
        except Exception:
            raise InstallationError("source_unreadable") from None
        finally:
            if fd is not None:
                os.close(fd)
    required = {("sglang_srt", path) for path in _REQUIRED_SGLANG}
    required |= {("occ3_sglang", path) for path in _REQUIRED_PLUGIN}
    if not required <= identities:
        raise InstallationError("source_attestation_incomplete")
    return documents


def _code_signature(code):
    import types
    def constant(value):
        if isinstance(value, types.CodeType):
            return ("code", _code_signature(value))
        if type(value) is tuple:
            return ("tuple", tuple(constant(x) for x in value))
        if type(value) is frozenset:
            return ("frozenset", frozenset(constant(x) for x in value))
        return (type(value).__name__, value)
    return (code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_nlocals, code.co_stacksize, code.co_flags, code.co_code,
            tuple(constant(x) for x in code.co_consts), code.co_names,
            code.co_varnames, code.co_freevars, code.co_cellvars, code.co_exceptiontable)


def _verify_function_provenance(function, documents):
    import sys
    import types
    from pathlib import Path
    if type(function) is not types.FunctionType or hasattr(function, "__wrapped__"):
        raise InstallationError("hook_provenance_invalid")
    path = str(Path(function.__code__.co_filename).resolve())
    source = documents.get(path)
    if source is None:
        raise InstallationError("hook_source_unattested")
    module = compile(source, path, "exec", dont_inherit=True, optimize=sys.flags.optimize)
    stack, matches = [module], []
    while stack:
        code = stack.pop()
        if code.co_qualname == function.__qualname__:
            matches.append(code)
        stack.extend(x for x in code.co_consts if isinstance(x, types.CodeType))
    if len(matches) != 1 or _code_signature(matches[0]) != _code_signature(function.__code__):
        raise InstallationError("hook_provenance_invalid")
    # The hooked native methods have only the reviewed None default or none.
    if (function.__defaults__ not in (None, (None,))
            or function.__kwdefaults__ not in (None, {})):
        raise InstallationError("hook_provenance_invalid")


def _resolve_runtime_device():
    """The process's runtime device, or None when it cannot be established.

    sglang 0.5.21 leaves ``server_args.device`` unset (None) when the
    deployment relies on auto-detection; the tokenizer process resolves the
    device through ``sglang.srt.utils.common.get_device`` during argument
    resolution. Auto-detect CUDA launches therefore carry ``device=None`` on
    the argv object, and the frontend profile must confirm the runtime device
    instead of demanding the literal attribute.
    """
    try:
        from sglang.srt.utils.common import get_device
        value = get_device()
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _frontend_members(app, tm):
    """The frontend profile as named pass/fail members (no short-circuit).

    Every member is evaluated so a decline names each tripping check instead
    of revealing one per live window. Values are bounded booleans over fixed
    config members; nothing secret is observed.
    """
    args = getattr(tm, "server_args", None)
    declared = getattr(args, "device", None)
    resolved = declared if declared is not None else _resolve_runtime_device()
    return (
        ("tokenizer_worker_num", getattr(args, "tokenizer_worker_num", None) != 1),
        ("device", resolved != "cuda"),
        ("tp_size", getattr(args, "tp_size", None) != 8),
        ("dp_size", getattr(args, "dp_size", None) != 1),
        ("pp_size", getattr(args, "pp_size", None) != 1),
        ("enable_lora", bool(getattr(args, "enable_lora", False))),
        ("disaggregation_mode",
         getattr(args, "disaggregation_mode", "null") not in (None, "null")),
        ("skip_tokenizer_init", bool(getattr(args, "skip_tokenizer_init", False))),
        ("enable_trace", bool(getattr(args, "enable_trace", False))),
        ("log_requests", bool(getattr(args, "log_requests", False))),
        ("request_logger",
         bool(getattr(getattr(tm, "request_logger", None), "log_requests", True))),
        ("root_path", bool(getattr(app, "root_path", ""))),
    )


def _runtime_preflight(app, tm):
    import sys
    args = getattr(tm, "server_args", None)
    module = sys.modules.get("sglang")
    if getattr(module, "__version__", None) != "0.5.21":
        raise InstallationError("unsupported_runtime")
    failed = [name for name, trips in _frontend_members(app, tm) if trips]
    if failed:
        error = InstallationError("unsupported_frontend_profile")
        error.frontend_members = ",".join(failed)[:128]
        raise error
    if app.middleware_stack is not None:
        raise InstallationError("pre_serving_install_required")
    return _existing_auth(app, tm)


@dataclass(repr=False)
class _Components:
    profile: object
    client: object
    backend: object
    authority: object
    adapter: object
    router: object
    principal_id: str
    gate: _BrokerGate
    registry: object
    hook_type: object
    wrap: object
    owners: dict
    documents: dict


def _sglang_srt_root(package):
    # Pinned sglang.srt is a namespace package (no __init__.py or __file__).
    # Accept one concrete package location only; never search arbitrary roots.
    from pathlib import Path
    try:
        locations = tuple(package.__path__)
        if len(locations) != 1 or type(locations[0]) is not str:
            raise InstallationError("source_root_unavailable")
        root = Path(locations[0]).absolute()
        file = getattr(package, "__file__", None)
        if (not root.is_dir() or any(path.is_symlink() for path in (root, *root.parents))
                or (file is not None and Path(file).absolute().parent != root)):
            raise InstallationError("source_root_unavailable")
        return root
    except InstallationError:
        raise
    except Exception:
        raise InstallationError("source_root_unavailable") from None


def _verify_descriptor_overlay(generate, tokenized, control_req, control_out):
    # GenerateReqInput is a dataclass; the pinned IPC classes are msgspec.Struct,
    # not dataclasses. Require the concrete append-only declared wire fields.
    if ("occ3_worker_descriptor" not in getattr(generate, "__dataclass_fields__", {})
            or "occ3_worker_descriptor" not in getattr(tokenized, "__struct_fields__", ())
            or any(not {"rid", "payload"} <= set(getattr(cls, "__struct_fields__", ()))
                   for cls in (control_req, control_out))):
        raise InstallationError("descriptor_overlay_unavailable")


async def _construct_components(app, tm, key, buffer_profile):
    """Concrete factories only. No user-supplied backend/profile/callback injection."""
    import sys
    from pathlib import Path
    from .native_profile import load_source_attestation
    from .latent_worker import load_runtime_profile
    from .latent_backend import ControlClient, LatentBackend
    from .latent_control_api import BrokerAuthority, create_router
    from .chat_adapter import ChatAttachmentAdapter
    from sglang.srt.utils.auth import AuthLevel, decide_request_auth
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType, _wrap_fn

    documents = _verify_installed_sources(load_source_attestation(), {
        "sglang_srt": _sglang_srt_root(sys.modules.get("sglang.srt")),
        "occ3_sglang": Path(__file__).parent,
    })
    from sglang.srt.managers.io_struct import (
        GenerateReqInput, TokenizedGenerateReqInput, Occ3ControlReq, Occ3ControlOutput,
    )
    _verify_descriptor_overlay(GenerateReqInput, TokenizedGenerateReqInput,
                               Occ3ControlReq, Occ3ControlOutput)
    for function in (decide_request_auth, _wrap_fn):
        _verify_function_provenance(function, documents)
    for middleware in app.user_middleware:
        if getattr(middleware.cls, "__qualname__", "") == "add_api_key_middleware.<locals>._ApiKeyASGIMiddleware":
            _verify_function_provenance(middleware.cls.__call__, documents)
    # The native lifespan normally imports this lazily. Import only after the
    # installed source bytes are attested, before lifespan constructs its handler.
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    tokenizer_module = sys.modules.get("sglang.srt.managers.tokenizer_manager")
    chat_class = OpenAIServingChat
    tokenizer_class = getattr(tokenizer_module, "TokenizerManager", None)
    if (chat_class is None or tokenizer_class is None or type(tm) is not tokenizer_class
            or tm.serving_chat_class is not chat_class):
        raise InstallationError("unsupported_serving_class")
    owners = {"sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat": chat_class,
              "sglang.srt.managers.tokenizer_manager.TokenizerManager": tokenizer_class}
    profile = load_runtime_profile()
    principal_id = "occ3-broker-" + uuid.uuid4().hex
    authority = BrokerAuthority(profile, principal_id=principal_id)
    client = ControlClient(tm)
    try:
        backend = LatentBackend(client, profile, authority,
            max_output_bytes=buffer_profile.max_buffer_bytes)
        adapter = ChatAttachmentAdapter(backend, authorize=authority.authorize,
                                        profile=buffer_profile)
        router = create_router(backend, authority)
        if ({getattr(route, "path", None) for route in router.routes} != set(_CONTROL_PATHS)
                or len(router.routes) != len(_CONTROL_PATHS)
                or any(getattr(route, "methods", None) != {"POST"} for route in router.routes)):
            raise InstallationError("native_router_unverified")
        for route in router.routes:
            _verify_function_provenance(route.endpoint, documents)
        gate = _BrokerGate(authority, principal_id, key, decide_request_auth,
                           AuthLevel.NORMAL, buffer_profile.max_turn_seconds)
        return _Components(profile, client, backend, authority, adapter, router,
                           principal_id, gate, HookRegistry, HookType, _wrap_fn,
                           owners, documents)
    except BaseException:
        try:
            await client.close()
        except BaseException:
            raise InstallationError("factory_cleanup_unconfirmed") from None
        if getattr(tm, "_occ3_control_client", None) is client or not client.closed:
            raise InstallationError("factory_cleanup_unconfirmed")
        raise


_CONTROL_PATHS = (
    "/occ3/v1/native/sessions/enroll", "/occ3/v1/native/publish",
    "/occ3/v1/native/publication/status", "/occ3/v1/native/retire",
)
_EXPECTED_HOOKS = (
    ("sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat.handle_request", "AROUND"),
    ("sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat._convert_to_internal_request", "AFTER"),
    ("sglang.srt.managers.tokenizer_manager.TokenizerManager.generate_request", "AROUND"),
)
_MISSING = object()
_cleanup_tasks = set()


def _check_route_conflicts(app):
    from starlette.routing import Match
    if len(app.router.routes) > 1024:
        raise InstallationError("route_profile_unsupported")
    for path in _CONTROL_PATHS:
        scope = {"type": "http", "method": "POST", "path": path, "root_path": "",
                 "headers": [], "query_string": b""}
        for route in app.router.routes:
            match, _ = route.matches(scope)
            if match == Match.FULL:
                raise InstallationError("native_route_conflict")


def _prepare_hooks(components):
    registry = components.registry
    if type(registry._patched) is not set or not isinstance(registry._hooks, dict):
        raise InstallationError("unsupported_hook_registry")
    specs = components.adapter.hook_specs()
    if tuple((target, kind) for target, _, kind in specs) != _EXPECTED_HOOKS:
        raise InstallationError("unsupported_hook_targets")
    pending = []
    for target, hook, kind in specs:
        if target in registry._patched or registry._hooks.get(target):
            raise InstallationError("hook_target_occupied")
        owner_name, attribute = target.rsplit(".", 1)
        owner = components.owners.get(owner_name)
        if owner is None or getattr(hook, "__self__", None) is not components.adapter:
            raise InstallationError("unsupported_hook_targets")
        original = getattr(owner, attribute)
        expected_module = owner_name.rsplit(".", 1)[0]
        expected_qualname = owner_name.rsplit(".", 1)[1] + "." + attribute
        if attribute == "handle_request":
            expected_module = "sglang.srt.entrypoints.openai.serving_base"
            expected_qualname = "OpenAIServingBase.handle_request"
        if (getattr(original, "__module__", None) != expected_module
                or getattr(original, "__qualname__", None) != expected_qualname):
            raise InstallationError("hook_provenance_invalid")
        _verify_function_provenance(original, components.documents)
        raw = vars(owner).get(attribute, _MISSING)
        kind_value = getattr(components.hook_type, kind)
        wrapped = components.wrap(original, hook, kind_value)
        pending.append((target, owner, attribute, raw, wrapped, hook, kind_value, original))
    return pending


def _owned_routes(app, router, previous):
    endpoints = {getattr(route, "endpoint", None) for route in router.routes}
    return [route for route in app.router.routes if not any(route is item for item in previous)
            and (getattr(route, "original_router", None) is router
                 or getattr(route, "endpoint", _MISSING) in endpoints)]


@dataclass(repr=False)
class _CommitSnapshot:
    router: object
    route_list: object
    routes: tuple
    middleware_list: object
    middleware: tuple
    middleware_calls: tuple
    middleware_codes: tuple
    auth_options: tuple
    server_args: object
    openapi: object
    key: str = field(repr=False)
    registry_hooks: object = None
    registry_patched: object = None
    hook_codes: tuple = ()


def _snapshot_precommit(app, tm, key):
    middleware = tuple(app.user_middleware)
    return _CommitSnapshot(app.router, app.router.routes, tuple(app.router.routes),
        app.user_middleware, middleware,
        tuple(getattr(item.cls, "__call__", None) for item in middleware),
        tuple(getattr(getattr(item.cls, "__call__", None), "__code__", None) for item in middleware),
        tuple(dict(getattr(item, "kwargs", {})) for item in middleware),
        tm.server_args, app.openapi_schema, key)


def _same_sequence(current, original):
    return len(current) == len(original) and all(a is b for a, b in zip(current, original))


def _revalidate_precommit(app, tm, components, hooks, snapshot, *, closed=False):
    """No await: final ownership fence before touching shared pre-serving state."""
    try:
        if (getattr(app, _INSTALLED_ATTR, None) is not None
                or app.middleware_stack is not None or app.router is not snapshot.router
                or app.router.routes is not snapshot.route_list
                or not _same_sequence(app.router.routes, snapshot.routes)
                or app.user_middleware is not snapshot.middleware_list
                or not _same_sequence(app.user_middleware, snapshot.middleware)
                or not _same_sequence(tuple(getattr(item.cls, "__call__", None)
                    for item in app.user_middleware), snapshot.middleware_calls)
                or not _same_sequence(tuple(getattr(getattr(item.cls, "__call__", None), "__code__", None)
                    for item in app.user_middleware), snapshot.middleware_codes)
                or any(getattr(item, "kwargs", {}) != options
                       for item, options in zip(app.user_middleware, snapshot.auth_options))
                or tm.server_args is not snapshot.server_args
                or app.openapi_schema is not snapshot.openapi
                or not secrets.compare_digest(_runtime_preflight(app, tm), snapshot.key)
                or components.client.tm is not tm
                or components.backend.client is not components.client
                or (not closed and (getattr(tm, "_occ3_control_client", None) is not components.client
                    or components.client.closed is not False or components.backend.closed is not False))
                or (closed and (getattr(tm, "_occ3_control_client", None) is not None
                    or components.client.closed is not True or components.backend.closed is not True
                    or getattr(components.client, "has_uncertain_control", True) is not False))
                or components.registry._hooks is not snapshot.registry_hooks
                or components.registry._patched is not snapshot.registry_patched):
            raise InstallationError("installation_ownership_changed")
        _check_route_conflicts(app)  # Also detects in-place route path/method changes.
        if len(hooks) != len(snapshot.hook_codes):
            raise InstallationError("installation_ownership_changed")
        for (target, owner, attribute, raw, wrapped, hook, kind, original), code in zip(hooks, snapshot.hook_codes):
            if (getattr(owner, attribute, _MISSING) is not original
                    or original.__code__ is not code
                    or vars(owner).get(attribute, _MISSING) is not raw
                    or target in components.registry._patched
                    or components.registry._hooks.get(target)):
                raise InstallationError("installation_ownership_changed")
            _verify_function_provenance(original, components.documents)
        for item in app.user_middleware:
            if getattr(item.cls, "__qualname__", "") == "add_api_key_middleware.<locals>._ApiKeyASGIMiddleware":
                _verify_function_provenance(item.cls.__call__, components.documents)
    except Exception:
        # Never propagate mutable app/auth/plugin values or backend details.
        raise InstallationError("installation_ownership_changed") from None


def _rollback(app, components, hooks, route_objects, middleware_objects, openapi_before):
    complete = True
    for target, owner, attribute, raw, wrapped, hook, _, original in reversed(hooks):
        current = getattr(owner, attribute, _MISSING)
        if current is wrapped:
            if raw is _MISSING:
                delattr(owner, attribute)
            else:
                setattr(owner, attribute, raw)
            components.registry._patched.discard(target)
        elif current is not original or (raw is _MISSING and attribute in vars(owner)):
            complete = False  # Never overwrite an unknown concurrent wrapper.
        entries = components.registry._hooks.get(target, [])
        retained = [item for item in entries if item[1] is not hook]
        if retained:
            components.registry._hooks[target] = retained
        else:
            components.registry._hooks.pop(target, None)
    app.router.routes[:] = [route for route in app.router.routes
                           if not any(route is owned for owned in route_objects)]
    app.user_middleware[:] = [item for item in app.user_middleware
                             if not any(item is owned for owned in middleware_objects)]
    app.openapi_schema = openapi_before
    return complete


async def _close_components(components):
    if components is None:
        return True
    task = asyncio.create_task(components.backend.close())
    try:
        done, _ = await asyncio.wait({task}, timeout=1.0)
        if done:
            try:
                task.result()
                return (components.backend.closed is True and components.client.closed is True
                        and getattr(components.client, "has_uncertain_control", True) is False
                        and getattr(components.client.tm, "_occ3_control_client", None) is not components.client)
            except BaseException:
                return False
        task.cancel()
        _cleanup_tasks.add(task)
        def settled(done):
            _cleanup_tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(settled)
        return False
    except BaseException:
        task.cancel()
        _cleanup_tasks.add(task)
        def settled_after_cancel(done):
            _cleanup_tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(settled_after_cancel)
        raise


async def install_native_runtime(app, tokenizer_manager, *, experimental_opt_in=False,
                                 buffer_profile=None):
    """Explicit trusted startup only; returns safe installation, never live proof.

    It mounts actual A routes/backend and the combined native source/receiver
    adapter only after pinned source, original auth and ordered worker readiness.
    No active uninstall exists. A serving adapter must never be discarded.
    """
    from . import startup_state
    if experimental_opt_in is not True:
        startup_state.record_attempt_code("explicit_experimental_opt_in_required")
        return InstallationReceipt(False, "explicit_experimental_opt_in_required")
    if not _install_lock.acquire(blocking=False):
        startup_state.record_attempt_code("installation_busy")
        return InstallationReceipt(False, "installation_busy")
    attempt = uuid.uuid4().hex[:8]
    components = None
    hooks, owned_routes, owned_middleware = [], [], []
    openapi_before = getattr(app, "openapi_schema", None)
    cancelled = False
    committed = False
    mutation_started = False
    # Hoisted fence outcome: nothing pending before the transaction begins.
    rolled_back = True
    closed = True
    code = None
    snapshot = None
    detail = None
    try:
        existing = getattr(app, _INSTALLED_ATTR, None)
        if existing is not None:
            if (type(existing) is NativeInstallation and existing.tm is tokenizer_manager
                    and getattr(tokenizer_manager, "_occ3_control_client", None) is existing.backend.client
                    and all(getattr(owner, attribute, None) is wrapped
                            and target in existing.registry._patched
                            for target, owner, attribute, _, wrapped, _, _, _ in existing.hooks)):
                return existing.receipt
            startup_state.record_attempt_code("installation_owner_mismatch", attempt=attempt)
            return InstallationReceipt(False, "installation_owner_mismatch")
        # Only the lock holder mutates the machine from here to the finalize site.
        startup_state.record_installing(attempt)
        key = _runtime_preflight(app, tokenizer_manager)
        startup_state.mark_preflight_reached()
        if getattr(tokenizer_manager, "_occ3_control_client", None) is not None:
            raise InstallationError("control_client_occupied")
        _check_route_conflicts(app)
        from .chat_adapter import BufferProfile
        # Only the default profile takes the per-boot turn window; a trusted
        # caller's explicit profile passes through untouched.
        profile = (BufferProfile(max_turn_seconds=_max_turn_seconds_from_env())
                   if buffer_profile is None else buffer_profile)
        if type(profile) is not BufferProfile or profile.max_buffer_bytes > 4194304:
            raise InstallationError("unsupported_buffer_profile")
        snapshot = _snapshot_precommit(app, tokenizer_manager, key)
        components = await _construct_components(app, tokenizer_manager, key, profile)
        startup_state.mark_backend_constructed()
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        snapshot.registry_hooks = components.registry._hooks
        snapshot.registry_patched = components.registry._patched
        hooks = _prepare_hooks(components)
        startup_state.mark_hooks_prepared()
        snapshot.hook_codes = tuple(original.__code__ for _, _, _, _, _, _, _, original in hooks)
        await components.backend.refresh_readiness()
        startup_state.mark_worker_readiness_reached()
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        readiness = components.backend.readiness()
        if readiness.receiver_ready is not True or readiness.source_ready is not True:
            raise InstallationError("backend_not_ready")
        _revalidate_precommit(app, tokenizer_manager, components, hooks, snapshot)
        # No awaits from final ownership fence through app/registry mutation.
        mutation_started = True
        startup_state.mark_route_mount_attempted()
        routes_before = list(app.router.routes)
        try:
            app.include_router(components.router)
        finally:
            owned_routes = _owned_routes(app, components.router, routes_before)
        middleware_before = list(app.user_middleware)
        try:
            app.add_middleware(VerifiedBrokerMiddleware, gate=components.gate)
        finally:
            owned_middleware = [m for m in app.user_middleware
                if not any(m is item for item in middleware_before)
                and m.cls is VerifiedBrokerMiddleware
                and getattr(m, "kwargs", {}).get("gate") is components.gate]
        for target, owner, attribute, raw, wrapped, hook, kind, original in hooks:
            components.registry.register(target, hook, kind)
            setattr(owner, attribute, wrapped)
            components.registry._patched.add(target)
        for _, owner, attribute, _, _, hook, _, _ in hooks:
            if not components.adapter._has_hook(getattr(owner, attribute), hook.__name__):
                raise InstallationError("hook_install_unverified")
        app.openapi_schema = None
        p = components.profile
        receipt = InstallationReceipt(True, "installed_experimental_not_live_validated",
            source_ready=True, receiver_ready=True, runtime_id=p.runtime_epoch,
            runtime_generation=p.runtime_generation, model_fingerprint=p.model_fingerprint)
        installation = NativeInstallation(receipt, app, tokenizer_manager, components.backend,
            components.authority, components.adapter, components.principal_id,
            tuple(hooks), components.registry)
        setattr(app, _INSTALLED_ATTR, installation)
        committed = True
        startup_state.record_native_ready(attempt=attempt, installation=installation)
        logger.info("occ3-bridge: native runtime mounted (attempt=%s stage=install code=%s)",
                    attempt, receipt.code)
        return receipt
    except asyncio.CancelledError:
        cancelled = True
        code = "installation_cancelled"
    except InstallationError as error:
        code = error.code
        detail = _frontend_detail(error)
    except Exception as error:
        code = "installation_unavailable"
        detail = _decline_detail(error)
    finally:
        # Installed adapters retain ownership. Failure closes only this transaction's client.
        try:
            if components is not None and not committed:
                try:
                    if mutation_started:
                        rolled_back = _rollback(app, components, hooks, owned_routes, owned_middleware, openapi_before)
                    else:
                        # No app/hook mutation occurred. Preserve foreign changes,
                        # including its marker/OpenAPI, and still close our client.
                        try:
                            _revalidate_precommit(app, tokenizer_manager, components, hooks, snapshot)
                            rolled_back = True
                        except InstallationError:
                            rolled_back = False
                    closed = await _close_components(components)
                    if closed and rolled_back:
                        try:
                            # Cleanup itself awaited: only a fresh no-await fence
                            # can certify safe ordinary startup after refusal.
                            _revalidate_precommit(app, tokenizer_manager, components, hooks, snapshot, closed=True)
                        except InstallationError:
                            rolled_back = False
                    if not rolled_back or not closed:
                        code = "installation_cleanup_unconfirmed"
                except asyncio.CancelledError:
                    cancelled = True
                    code = "installation_cleanup_unconfirmed"
                except Exception:
                    code = "installation_cleanup_unconfirmed"
        finally:
            _install_lock.release()
    if cancelled:
        startup_state.record_declined(code, stage="install", attempt=attempt,
                                       rollback_completed=rolled_back and closed)
        raise asyncio.CancelledError
    if detail:
        logger.error("occ3-bridge: native installation decline (attempt=%s stage=install code=%s detail=%s)",
                     attempt, code, detail)
    else:
        logger.error("occ3-bridge: native installation decline (attempt=%s stage=install code=%s)",
                     attempt, code)
    startup_state.record_declined(code, stage="install", attempt=attempt,
                                   rollback_completed=rolled_back and closed)
    return InstallationReceipt(False, code)


@dataclass(frozen=True)
class StartupReceipt:
    selected: bool
    prepared: bool
    code: str

    def evidence(self):
        return {"selected": self.selected, "prepared": self.prepared, "code": self.code,
                "live_proof": False, "gate8_pass": False, "supported": False}


_STARTUP_ATTR = "_occ3_native_startup_receipt"
_SERVER_ATTR = "_occ3_native_startup_owner"


def _read_startup_literal():
    """Read fixed build-generated literals without executing this module's body."""
    import ast
    import os
    import re
    import stat
    from pathlib import Path

    file = Path(__file__).absolute().parent / "_native_profile_digest.py"
    if any(path.is_symlink() for path in (file, *file.parents)):
        raise InstallationError("startup_selection_unsafe")
    fd = None
    try:
        fd = os.open(file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_mode & 0o022 or not 0 < before.st_size <= 4096):
            raise InstallationError("startup_selection_unsafe")
        with os.fdopen(fd, "rb") as stream:
            fd = None
            raw = stream.read(4097)
            after = os.fstat(stream.fileno())
        signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if signature(before) != signature(after) or len(raw) > 4096:
            raise InstallationError("startup_selection_unsafe")
        values = {}
        for statement in ast.parse(raw.decode("utf-8")).body:
            if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)
                    or statement.targets[0].id not in
                    {"EXPERIMENTAL_STARTUP", "PROFILE_SHA256", "SOURCE_SHA256"}
                    or statement.targets[0].id in values
                    or not isinstance(statement.value, ast.Constant)):
                raise InstallationError("startup_selection_invalid")
            values[statement.targets[0].id] = statement.value.value
        if type(values.get("EXPERIMENTAL_STARTUP", False)) is not bool:
            raise InstallationError("startup_selection_invalid")
        if not values.get("EXPERIMENTAL_STARTUP", False):
            return None
        if (set(values) != {"EXPERIMENTAL_STARTUP", "PROFILE_SHA256", "SOURCE_SHA256"}
                or any(type(values[name]) is not str or not re.fullmatch("[a-f0-9]{64}", values[name])
                       for name in ("PROFILE_SHA256", "SOURCE_SHA256"))):
            raise InstallationError("startup_selection_invalid")
        return values
    except FileNotFoundError:
        return None
    except InstallationError:
        raise
    except Exception:
        raise InstallationError("startup_selection_invalid") from None
    finally:
        if fd is not None:
            os.close(fd)


def _verify_startup_artifacts(selection):
    # The generated module has already been parsed as literals only. Its raw hash
    # belongs to the external packaging receipt, not a self-referential manifest.
    from importlib import import_module
    from .native_profile import load_pinned_profile, load_source_attestation
    module = import_module("occ3_sglang._native_profile_digest")
    if any(getattr(module, name, None) != value for name, value in selection.items()):
        raise InstallationError("startup_selection_changed")
    load_pinned_profile()
    load_source_attestation()


def _verify_native_server(server, app, tm):
    """Check the one native Uvicorn instance; never replace a global class method."""
    import os
    import stat
    import sys
    from pathlib import Path
    import uvicorn
    import uvicorn.server

    server_class = uvicorn.server.Server
    config = getattr(server, "config", None)
    args = getattr(tm, "server_args", None)
    if (uvicorn.Server is not server_class or type(server) is not server_class
            or type(config) is not uvicorn.Config or config.app is not app
            or config.loaded or config.workers != 1 or config.reload
            or config.loop != "uvloop" or config.root_path
            or getattr(args, "tokenizer_worker_num", None) != 1
            or getattr(args, "enable_http2", False) or getattr(args, "enable_ssl_refresh", False)
            or getattr(app, "is_single_tokenizer_mode", False) is not True
            or getattr(app, "server_args", None) is not args
            or app.middleware_stack is not None
            or "serve" in vars(server) or "run" in vars(server)):
        raise InstallationError("unsupported_native_startup")
    module = sys.modules.get("uvicorn.server")
    file = Path(getattr(module, "__file__", "")).absolute()
    if any(path.is_symlink() for path in (file, *file.parents)):
        raise InstallationError("startup_server_provenance")
    fd = None
    try:
        fd = os.open(file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 262144:
            raise InstallationError("startup_server_provenance")
        with os.fdopen(fd, "rb") as stream:
            fd = None
            source = stream.read(262145).decode("utf-8")
            after = os.fstat(stream.fileno())
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise InstallationError("startup_server_provenance")
        for name in ("run", "serve"):
            function = getattr(server_class, name)
            if (function.__module__ != "uvicorn.server" or function.__qualname__ != "Server." + name
                    or getattr(server, name).__self__ is not server
                    or getattr(server, name).__func__ is not function):
                raise InstallationError("startup_server_provenance")
            _verify_function_provenance(function, {str(file.resolve()): source})
        if not inspect.iscoroutinefunction(server_class.serve):
            raise InstallationError("startup_server_provenance")
    except InstallationError:
        raise
    except Exception:
        raise InstallationError("startup_server_provenance") from None
    finally:
        if fd is not None:
            os.close(fd)
    return server.serve


@dataclass(repr=False)
class _StartupOwner:
    app: object
    tm: object
    original_serve: object
    wrapped_serve: object = None
    started: bool = False


def prepare_selected_native_server(server, app, tokenizer_manager):
    """Build-selected, default-Uvicorn-only preparation, before native run().

    The wrapper is awaited by native run() on its original event loop. It does
    not create a loop or change Uvicorn arguments/options/stop hooks. Absent or
    false fixed build selection does not load profiles, backends or controls.
    """
    from . import startup_state
    attempt = uuid.uuid4().hex[:8]
    selection = None
    try:
        selection = _read_startup_literal()
        if selection is None:
            receipt = StartupReceipt(False, False, "experimental_startup_not_selected")
            setattr(app, _STARTUP_ATTR, receipt.evidence())
            startup_state.record_prepare("experimental_startup_not_selected",
                                         selected=False, attempt=attempt)
            logger.info("occ3-bridge: native startup decline (attempt=%s stage=prepare "
                        "code=experimental_startup_not_selected selected=false)", attempt)
            return receipt
        # Phase split: the control plane registers before any owner check or
        # GPU-side validation, so declines stay observable over HTTP.
        register_control_plane(app, tokenizer_manager)
        existing = getattr(server, _SERVER_ATTR, None)
        if existing is not None:
            if (type(existing) is _StartupOwner and existing.app is app
                    and existing.tm is tokenizer_manager
                    and getattr(server, "serve", None) is existing.wrapped_serve):
                return StartupReceipt(True, True, "experimental_startup_prepared")
            raise InstallationError("startup_owner_mismatch")
        _verify_startup_artifacts(selection)
        original = _verify_native_server(server, app, tokenizer_manager)
    except InstallationError as error:
        # Nothing was dispatched or patched in this synchronous preparation.
        receipt = StartupReceipt(selection is not None, False, error.code)
        setattr(app, _STARTUP_ATTR, receipt.evidence())
        startup_state.record_prepare(error.code, selected=selection is not None,
                                     attempt=attempt)
        logger.error("occ3-bridge: native startup decline (attempt=%s stage=prepare code=%s selected=%s)",
                     attempt, error.code, selection is not None)
        return receipt
    except Exception as error:
        receipt = StartupReceipt(selection is not None, False, "experimental_startup_refused")
        setattr(app, _STARTUP_ATTR, receipt.evidence())
        startup_state.record_prepare("experimental_startup_refused",
                                     selected=selection is not None, attempt=attempt)
        logger.error("occ3-bridge: native startup decline (attempt=%s stage=prepare code=experimental_startup_refused "
                     "selected=%s detail=%s)", attempt, selection is not None,
                     _decline_detail(error))
        return receipt

    owner = _StartupOwner(app, tokenizer_manager, original)
    async def selected_serve(*args, **kwargs):
        if (owner.started or getattr(server, _SERVER_ATTR, None) is not owner
                or getattr(server, "serve", None) is not owner.wrapped_serve):
            raise InstallationError("startup_owner_mismatch")
        owner.started = True
        try:
            result = await install_native_runtime(app, tokenizer_manager, experimental_opt_in=True)
        except BaseException as error:
            # Cancellation/unknown exceptions never turn an unverified partial
            # transaction into an ordinary serving candidate.
            setattr(app, _STARTUP_ATTR, {"code": "native_startup_aborted", "supported": False,
                                        "live_proof": False, "gate8_pass": False})
            # A raise-path code always supersedes the recorded decline code.
            startup_state.record_superseded("native_startup_aborted",
                                            attempt=uuid.uuid4().hex[:8])
            if isinstance(error, asyncio.CancelledError):
                raise
            raise InstallationError("native_startup_aborted") from None
        if type(result) is not InstallationReceipt:
            raise InstallationError("native_startup_cleanup_unconfirmed")
        setattr(app, _STARTUP_ATTR, result.evidence())
        if not result.installed:
            logger.error("occ3-bridge: native startup decline (attempt=%s stage=install code=%s)",
                         uuid.uuid4().hex[:8], result.code)
        if (result.code in {"installation_cleanup_unconfirmed", "factory_cleanup_unconfirmed",
                            "installation_busy", "installation_owner_mismatch", "hook_target_occupied",
                            "control_client_occupied", "installation_ownership_changed"}
                or (not result.installed and (getattr(app, _INSTALLED_ATTR, None) is not None
                    or getattr(tokenizer_manager, "_occ3_control_client", None) is not None
                    or _cleanup_tasks))):
            startup_state.record_superseded("native_startup_cleanup_unconfirmed",
                                            attempt=uuid.uuid4().hex[:8])
            raise InstallationError("native_startup_cleanup_unconfirmed")
        # A declined installation reaches here only after verified owned rollback
        # and control settlement. No success/capability claim follows a refusal.
        return await owner.original_serve(*args, **kwargs)

    owner.wrapped_serve = selected_serve
    setattr(server, _SERVER_ATTR, owner)
    try:
        setattr(server, "serve", selected_serve)
    except BaseException:
        if getattr(server, _SERVER_ATTR, None) is owner:
            delattr(server, _SERVER_ATTR)
        raise InstallationError("startup_wrapper_unavailable") from None
    receipt = StartupReceipt(True, True, "experimental_startup_prepared")
    setattr(app, _STARTUP_ATTR, receipt.evidence())
    startup_state.record_prepare("experimental_startup_prepared", selected=True,
                                 attempt=attempt)
    return receipt
