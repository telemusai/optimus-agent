"""Startup phase-split state machine and the minimal pre-install control plane.

The module owns a process-wide singleton that records how far the experimental
native startup got, plus the three control routes that must answer BEFORE the
native installation succeeds or declines. Scheduler processes import this
module through plugin.load(); the import itself never mutates any app and the
routes are only mounted by explicit registration call sites.

Truthfulness rules (normative, see the phase-split design):
- ``supported`` stays false in every state; the server never self-certifies
  live proof. Only the controller validates. ``methods`` lists the method
  ids the installed registry advertises (build capability), which is not a
  readiness or live-proof claim.
- ``code``/``stage``/``attempt`` are fixed-taxonomy values or null; exception
  text, paths, principals and secrets never enter a payload or log line.
- The identity route never fabricates the incarnation tuple: before
  NATIVE_READY it answers 503 ``not_ready``; afterwards it delegates to the
  extracted runtime handler against the installed backend only.
"""

from __future__ import annotations

import dataclasses
import enum
import re
import threading
from dataclasses import dataclass, field

from . import __version__, metrics
from .api import _state
from .identity import compute_identity, fingerprint
from .latent_control_api import control_error_response, runtime_identity
from .latent_types import LatentError
from .registry import registry

STATUS_SCHEMA = "optimus.occ3.status.v1"
CAPABILITY_SCHEMA = "optimus.occ3.capability.v1"
CONTROL_PLANE_ATTR = "_occ3_control_plane_installed"
CONTROL_PLANE_ROUTES = (
    ("/occ3/v1/status", ("GET",)),
    ("/occ3/v1/capabilities", ("GET",)),
    ("/occ3/v1/native/runtime/identity", ("POST",)),
)
CONTROL_PLANE_PATHS = tuple(path for path, _ in CONTROL_PLANE_ROUTES)

_STAGES = frozenset({"prepare", "install"})
_ATTEMPT = re.compile(r"[0-9a-f]{8}\Z")
# Every code the receipts, InstallationError raises and serve-abort paths can
# emit today. Anything else is dropped rather than surfaced (fail-closed).
_FIXED_CODES = frozenset({
    "backend_not_ready", "control_client_occupied", "descriptor_overlay_unavailable",
    "explicit_experimental_opt_in_required", "experimental_startup_not_selected",
    "experimental_startup_prepared", "experimental_startup_refused",
    "factory_cleanup_unconfirmed", "hook_install_unverified",
    "hook_provenance_invalid", "hook_source_unattested", "hook_target_occupied",
    "installation_busy", "installation_cancelled", "installation_cleanup_unconfirmed",
    "installation_owner_mismatch", "installation_ownership_changed",
    "installation_unavailable", "installed_experimental_not_live_validated",
    "native_route_conflict", "native_router_unverified", "native_startup_aborted",
    "native_startup_cleanup_unconfirmed", "original_api_key_required",
    "original_auth_mismatch", "original_auth_registration_required",
    "pre_serving_install_required", "route_profile_unsupported",
    "source_attestation_incomplete", "source_attestation_invalid", "source_changed",
    "source_digest_mismatch", "source_file_invalid", "source_root_unavailable",
    "source_symlink", "source_unreadable", "startup_owner_mismatch",
    "startup_selection_changed", "startup_selection_invalid", "startup_selection_unsafe",
    "startup_server_provenance", "startup_wrapper_unavailable",
    "unsupported_buffer_profile", "unsupported_frontend_profile",
    "unsupported_hook_registry", "unsupported_hook_targets",
    "unsupported_native_startup", "unsupported_runtime", "unsupported_serving_class",
})


class State(enum.Enum):
    """Server-side states only; LIVE_VERIFIED/AVAILABLE are controller-side."""

    DISABLED = "DISABLED"
    CONTROL_REGISTERED = "CONTROL_REGISTERED"
    INSTALLING = "INSTALLING"
    DECLINED = "DECLINED"
    NATIVE_READY = "NATIVE_READY"


@dataclass(repr=False)
class StartupState:
    state: State = State.DISABLED
    code: str | None = None
    stage: str | None = None
    attempt: str | None = None
    experimental_selected: bool = False
    preflight_reached: bool = False
    backend_constructed: bool = False
    worker_readiness_reached: bool = False
    hooks_prepared: bool = False
    route_mount_attempted: bool = False
    rollback_completed: bool = False
    installation: object = field(default=None, repr=False)


_LOCK = threading.Lock()
_SINGLETON = StartupState()


def _fixed_code(code):
    return code if type(code) is str and code in _FIXED_CODES else None


def _fixed_attempt(attempt):
    return attempt if type(attempt) is str and _ATTEMPT.fullmatch(attempt) else None


def current() -> StartupState:
    """Snapshot for payloads/tests; the private installation handle is dropped."""
    with _LOCK:
        return dataclasses.replace(_SINGLETON, installation=None)


def reset() -> None:
    """Test-only: restore the pristine DISABLED machine."""
    global _SINGLETON
    with _LOCK:
        _SINGLETON = StartupState()


def record_control_registered() -> None:
    """CONTROL_REGISTERED once the control routes ride verified original auth."""
    with _LOCK:
        if _SINGLETON.state in (State.DECLINED, State.NATIVE_READY):
            return
        _SINGLETON.state = State.CONTROL_REGISTERED


def record_control_refusal(code, *, attempt) -> None:
    """Registration refusal never transitions; the fixed code stays observable."""
    with _LOCK:
        _SINGLETON.code = _fixed_code(code)
        _SINGLETON.stage = "prepare"
        _SINGLETON.attempt = _fixed_attempt(attempt)


def record_prepare(code, *, selected, attempt) -> None:
    """Prepare-phase receipt: code/stage only; the state never transitions."""
    with _LOCK:
        _SINGLETON.code = _fixed_code(code)
        _SINGLETON.stage = "prepare"
        _SINGLETON.attempt = _fixed_attempt(attempt)
        _SINGLETON.experimental_selected = bool(selected)


def record_attempt_code(code, *, attempt=None) -> None:
    """Non-transitioning install outcome (busy/opt-in/owner mismatch)."""
    with _LOCK:
        _SINGLETON.code = _fixed_code(code)
        _SINGLETON.stage = "install"
        if attempt is not None:
            _SINGLETON.attempt = _fixed_attempt(attempt)


def record_installing(attempt) -> None:
    """INSTALLING is set only by the lock-holding install attempt."""
    with _LOCK:
        _SINGLETON.state = State.INSTALLING
        _SINGLETON.code = None
        _SINGLETON.stage = "install"
        _SINGLETON.attempt = _fixed_attempt(attempt)
        _SINGLETON.experimental_selected = True
        _SINGLETON.preflight_reached = False
        _SINGLETON.backend_constructed = False
        _SINGLETON.worker_readiness_reached = False
        _SINGLETON.hooks_prepared = False
        _SINGLETON.route_mount_attempted = False
        _SINGLETON.rollback_completed = False
        _SINGLETON.installation = None


def _mark_in_flight(name):
    with _LOCK:
        if _SINGLETON.state is State.INSTALLING:
            setattr(_SINGLETON, name, True)


def mark_preflight_reached() -> None:
    _mark_in_flight("preflight_reached")


def mark_backend_constructed() -> None:
    _mark_in_flight("backend_constructed")


def mark_hooks_prepared() -> None:
    _mark_in_flight("hooks_prepared")


def mark_worker_readiness_reached() -> None:
    _mark_in_flight("worker_readiness_reached")


def mark_route_mount_attempted() -> None:
    _mark_in_flight("route_mount_attempted")


def record_declined(code, *, stage, attempt, rollback_completed) -> None:
    """DECLINED at the decline-finalize site; carries the receipt code."""
    with _LOCK:
        _SINGLETON.state = State.DECLINED
        _SINGLETON.code = _fixed_code(code)
        _SINGLETON.stage = stage if stage in _STAGES else None
        _SINGLETON.attempt = _fixed_attempt(attempt)
        _SINGLETON.rollback_completed = bool(rollback_completed)
        _SINGLETON.installation = None


def record_superseded(code, *, stage="install", attempt=None) -> None:
    """Raise-path outcome supersedes any earlier code; booleans are preserved."""
    with _LOCK:
        _SINGLETON.state = State.DECLINED
        _SINGLETON.code = _fixed_code(code)
        _SINGLETON.stage = stage if stage in _STAGES else _SINGLETON.stage
        if attempt is not None:
            _SINGLETON.attempt = _fixed_attempt(attempt)


def record_native_ready(*, attempt, installation) -> None:
    """NATIVE_READY at the mounted site; the only path to identity 200."""
    with _LOCK:
        _SINGLETON.state = State.NATIVE_READY
        _SINGLETON.code = "installed_experimental_not_live_validated"
        _SINGLETON.stage = "install"
        _SINGLETON.attempt = _fixed_attempt(attempt)
        _SINGLETON.experimental_selected = True
        _SINGLETON.preflight_reached = True
        _SINGLETON.backend_constructed = True
        _SINGLETON.worker_readiness_reached = True
        _SINGLETON.hooks_prepared = True
        _SINGLETON.route_mount_attempted = True
        _SINGLETON.rollback_completed = False
        _SINGLETON.installation = installation


def _reason(machine):
    if machine.state is State.DECLINED:
        return "native_install_declined"
    if machine.state is State.NATIVE_READY:
        return "native_receiver_not_live_validated"
    return "native_latent_receiver_not_implemented"


def _native_view():
    """Readiness booleans come only from the installation receipt."""
    with _LOCK:
        machine = _SINGLETON
        if machine.state is not State.NATIVE_READY or machine.installation is None:
            return {"state": machine.state.value, "code": machine.code, "installed": False,
                    "source_ready": False, "receiver_ready": False, "live_validated": False}
        receipt = getattr(machine.installation, "receipt", None)
        return {"state": machine.state.value, "code": machine.code, "installed": True,
                "source_ready": bool(getattr(receipt, "source_ready", False)),
                "receiver_ready": bool(getattr(receipt, "receiver_ready", False)),
                "live_validated": False}


def status_payload():
    """Superset of the old StatusView; old fields verbatim, fixed key set."""
    identity = compute_identity(_state().tokenizer_manager)
    machine = current()
    return {
        "schema": STATUS_SCHEMA,
        "state": machine.state.value,
        "code": machine.code,
        "stage": machine.stage,
        "attempt": machine.attempt,
        "experimental_selected": machine.experimental_selected,
        "preflight_reached": machine.preflight_reached,
        "backend_constructed": machine.backend_constructed,
        "worker_readiness_reached": machine.worker_readiness_reached,
        "hooks_prepared": machine.hooks_prepared,
        "route_mount_attempted": machine.route_mount_attempted,
        "rollback_completed": machine.rollback_completed,
        "plugin": "occ3-bridge",
        "version": __version__,
        "enabled": False,
        "model_fingerprint": fingerprint(identity),
        "channels": registry().counts(),
        "metrics": metrics.snapshot(),
    }


def _advertised_methods():
    """Registry method ids: what this build supports, never what is live.

    Read from the method registry (occ3_sglang.native_profile), which is
    metadata-only: importing it pulls no torch, no sglang and no model state.
    A registry that cannot be read advertises nothing rather than guessing.
    """
    try:
        from .native_profile import advertised_methods
        return list(advertised_methods())
    except Exception:
        return []


def capabilities_payload():
    """Superset of the old advertisement; availability is never advertised."""
    identity = compute_identity(_state().tokenizer_manager)
    machine = current()
    return {
        "schema": CAPABILITY_SCHEMA,
        "supported": False,
        "reason": _reason(machine),
        "experimental_prompt_kv_path": "/occ3/v1/latent/capabilities",
        "model_fingerprint": fingerprint(identity),
        "connector_id": "sglang-occ3",
        "methods": _advertised_methods(),
        "experimental_baselines": ["text-radix", "source-prompt-kv-handle-v1"],
        "protocol": "occ3-v1",
        "api_version": "v1",
        "same_model_only": True,
        "verification": "cached_tokens",
        "cache": {"salt_prefix": "occ3-", "eviction": "lru"},
        "native": _native_view(),
    }


def _ready_installation():
    """Private handle for the delegated identity route; NATIVE_READY only."""
    with _LOCK:
        if _SINGLETON.state is State.NATIVE_READY:
            return _SINGLETON.installation
    return None


def create_control_router():
    """The three control-plane routes; one owner for all three paths."""
    from fastapi import APIRouter, Request

    router = APIRouter()

    async def status() -> dict:
        return status_payload()

    async def capabilities() -> dict:
        return capabilities_payload()

    async def identity(request):
        try:
            installation = _ready_installation()
            if installation is None:
                raise LatentError("not_ready")
            return await runtime_identity(request, installation.backend,
                                          installation.authority)
        except Exception as error:
            return control_error_response(error)

    # Avoid postponed local Request annotation evaluation in FastAPI.
    identity.__annotations__["request"] = Request
    router.add_api_route("/occ3/v1/status", status, methods=["GET"])
    router.add_api_route("/occ3/v1/capabilities", capabilities, methods=["GET"])
    router.add_api_route("/occ3/v1/native/runtime/identity", identity, methods=["POST"])
    return router
