"""HTTP control routes for OCC-3 (mounted under /occ3/v1).

Security: the routes are served by the same FastAPI app as /generate,
so the server-level api-key middleware covers them (paths under
/occ3/ are auth level NORMAL). The plugin deliberately avoids any
/health* path prefix so nothing here bypasses the key.

All routes are control-plane only: no tensors, no GPU state and no
model weights ever appear in requests or responses.

GET /occ3/v1/status and GET /occ3/v1/capabilities moved to
startup_state.create_control_router() (phase split): they must answer
before native installation resolves, so their owner is the startup
control plane, not this dormant bridge router.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .identity import IDENTITY_FIELDS, fingerprint
from .native_profile import DECLARED_IDENTITY, ProfileError, load_pinned_profile
from .security import DEFAULT_TTL_S

router = APIRouter(prefix="/occ3/v1", tags=["occ3"])


def _state():
    """Live server state (module attribute read at request time)."""
    import sglang.srt.entrypoints.http_server as http_server

    state = getattr(http_server, "_global_state", None)
    if state is None or getattr(state, "tokenizer_manager", None) is None:
        raise HTTPException(status_code=503, detail="server not ready")
    return state


def _tm():
    return _state().tokenizer_manager


class SendBody(BaseModel):
    channel: Optional[str] = None
    text: Optional[str] = None
    input_ids: Optional[List[int]] = None
    embeds: Optional[List[List[float]]] = None
    positions: Optional[List[int]] = None
    ttl_s: float = DEFAULT_TTL_S


class RecvBody(BaseModel):
    channel: str
    suffix: Optional[str] = None
    suffix_ids: Optional[List[int]] = None
    max_new_tokens: int = 256
    temperature: float = 0.0


class ReleaseBody(BaseModel):
    channel: str


@router.get("/identity")
async def identity_route() -> dict:
    """The deployment's declared-launch identity, pinned in the profile.

    The fields are the declared values, never runtime-derived: without a
    pinned declared-identity block the route fails closed rather than serving
    a computed fallback. The optimus-agent client replaces ``latent_method``
    with the runtime-attested method id before hashing, so this route serves
    the legacy label for that field.
    """
    try:
        profile = load_pinned_profile()
    except ProfileError:
        raise HTTPException(status_code=503,
                            detail="declared identity not pinned") from None
    declared = profile.get(DECLARED_IDENTITY)
    if not isinstance(declared, dict):
        raise HTTPException(status_code=503, detail="declared identity not pinned")
    identity = {name: declared[name] for name in IDENTITY_FIELDS}
    return {
        "schema": "optimus.occ3.identity.v1",
        "fields": identity,
        "fingerprint": fingerprint(identity),
        "field_order": list(IDENTITY_FIELDS),
    }


@router.get("/channels")
async def channels() -> dict:
    return {"channels": [], "available": False, "reason": "legacy_unbound_api_disabled"}


@router.post("/send")
async def send() -> dict:
    raise HTTPException(410, "legacy_unbound_api_disabled_use_opt_in_handles")


@router.post("/recv")
async def recv() -> dict:
    raise HTTPException(410, "legacy_unbound_api_disabled_use_opt_in_handles")


@router.post("/release")
async def release() -> dict:
    raise HTTPException(410, "legacy_unbound_api_disabled_use_opt_in_handles")
