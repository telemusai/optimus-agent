"""Strict zero-inference control routes and verified administrative broker context."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict

from .attachment_contract import (AttachmentError, AuthenticatedContext, CaptureIntent,
                                  InferenceAttachment, digest, identifier, strict_json)
from starlette.responses import JSONResponse
from .latent_types import (LatentError, Publication, SessionEnrollment, bounded_payload,
                           require, sha)


class BrokerAuthority:
    def __init__(self, profile, *, principal_id, epoch=time.time):
        identifier(principal_id)
        self.profile, self.principal_id = profile, principal_id
        self.principal_digest = hashlib.sha256(principal_id.encode("ascii")).hexdigest()
        self._marker = object()
        # Wall clock for permit expiry; injectable like LatentRegistry's.
        self.epoch = epoch
        self.sessions = {}
        self.operations = {}

    def clear_verified(self, request):
        if hasattr(request.state, "_occ3_verified_broker"):
            delattr(request.state, "_occ3_verified_broker")

    def mark_verified(self, request, principal_id, *, deadline_monotonic=None):
        """ONLY call from existing-key verification middleware after NORMAL grant."""
        require(principal_id == self.principal_id, "identity")
        if deadline_monotonic is not None:
            require(type(deadline_monotonic) in (int, float) and math.isfinite(deadline_monotonic)
                    and deadline_monotonic > time.monotonic(), "expired")
        request.state._occ3_verified_broker = (self._marker, self.principal_digest, deadline_monotonic)

    def verified(self, request):
        value = getattr(request.state, "_occ3_verified_broker", None)
        require(type(value) is tuple and len(value) == 3 and value[0] is self._marker
                and value[1] == self.principal_digest, "identity")
        return value[1], value[2]

    def check_session(self, principal, session):
        value = self.sessions.get((principal, session))
        require(value is not None and value.expires_at_epoch > self.epoch(), "identity")
        return value

    def _prune(self):
        """Drop expired enrollments, as LatentRegistry.enroll does.

        The table lives for the whole process. Without this, expired permits
        kept their quota slots: after profile.max_records enrollments every new
        one was refused with quota (HTTP 429) until a restart, and renewing a
        session whose permit had expired was refused as an identity conflict.
        Either way the client declined activation and installed no
        occ3_handoff tool. Live enrollments are never dropped.
        """
        now = self.epoch()
        for key in [k for k, s in self.sessions.items() if s.expires_at_epoch <= now]:
            del self.sessions[key]

    def remember_session(self, enrollment):
        require(enrollment.principal_digest == self.principal_digest, "identity")
        key = (enrollment.principal_digest, enrollment.session_id)
        self._prune()
        old = self.sessions.get(key)
        require(old is None or old == enrollment, "identity")
        require(old is not None or len(self.sessions) < self.profile.max_records, "quota")
        self.sessions[key] = enrollment

    async def authorize(self, request):
        try:
            principal, deadline = self.verified(request)
            body = strict_json(await request.body())
            require(isinstance(body, dict), "invalid")
            if "occ3_capture" in body and "occ3" not in body:
                intent = CaptureIntent.parse(body["occ3_capture"])
                session = intent.source_session_id
            elif "occ3" in body and "occ3_capture" not in body:
                session = InferenceAttachment.parse(body["occ3"]).receiver_session_id
            else:
                raise LatentError("invalid")
            enrollment = self.check_session(principal, session)
            require(body.get("model") == self.profile.model_id, "identity")
            if deadline is None:
                raise AttachmentError("deadline_unavailable", 409)
            return AuthenticatedContext(principal, session, self.profile.model_id,
                enrollment.model_fingerprint, enrollment.runtime_epoch,
                enrollment.runtime_generation,
                min(deadline, time.monotonic() + enrollment.expires_at_epoch - self.epoch()))
        except LatentError:
            raise AttachmentError("identity_mismatch", 401) from None


async def _body(request):
    data = bytearray()
    async for part in request.stream():
        require(len(data) + len(part) <= 16384, "quota")
        data.extend(part)
    value = strict_json(bytes(data))
    require(isinstance(value, dict))
    value = dict(value)
    bounded_payload(value)
    return value


def control_error_response(error=None):
    """Shared fixed-code control envelope; never any exception text."""
    code = error.code if isinstance(error, (LatentError, AttachmentError)) else "unavailable"
    status = (401 if code in ("identity", "identity_mismatch") else 429 if code == "quota"
              else 400 if code in ("invalid", "invalid_attachment")
              else 503 if code in ("unavailable", "not_ready") else 409)
    return JSONResponse({"error": {"code": code}}, status_code=status,
                        headers={"cache-control": "no-store"})


async def runtime_identity(request, backend, authority):
    """Read-only current-incarnation view for the authorized proof harness.

    Extracted verbatim from the former identity branch of the control router
    (body contract, tuple fields and the fixed error envelope are unchanged);
    the startup control plane delegates here only after NATIVE_READY.
    """
    try:
        principal, _ = authority.verified(request)
        value = await _body(request)
        body = _fields(value, "occ3.runtime.identity.request.v1", ())
        health = await backend.refresh_readiness()
        result = {"schema": "occ3.runtime.identity.v1",
            "runtime_id": health["runtime_epoch"],
            "runtime_generation": health["runtime_generation"],
            "model_fingerprint": health["model_fingerprint"],
            "method_id": health["method_id"],
            "tp_layout_digest": health["tp_layout_digest"],
            "layer_index": health["layer_index"], "hidden_dim": health["hidden_dim"],
            "dtype": health["dtype"], "expected_ranks": health["expected_ranks"],
            "ready_ranks": health["ready_ranks"], "profile_digest": health["profile_digest"],
            "source_capture": health["source_capture"],
            "continuous_prefix": health["continuous_prefix"],
            "terminal_journal": health["terminal_journal"],
            "server_time_epoch": int(time.time()),
            "max_ttl_seconds": backend.profile.max_ttl_seconds,
            "max_capture_bytes": health["max_capture_bytes"]}
        bounded_payload(result)
        return JSONResponse(result, headers={"cache-control": "no-store"})
    except (LatentError, AttachmentError) as error:
        return control_error_response(error)
    except Exception:
        return control_error_response()


def _fields(value, schema, fields):
    require(set(value) == {"schema", *fields} and value["schema"] == schema)
    return {k: v for k, v in value.items() if k != "schema"}


def create_router(backend, authority):
    """Returns dormant router; caller explicitly installs original-key middleware."""
    from fastapi import APIRouter, Request
    router = APIRouter()

    async def execute(request, operation):
        try:
            principal, _ = authority.verified(request)
            value = await _body(request)
            if operation == "enroll":
                body = _fields(value, "occ3.sessions.enroll.v1", ("operation_id", "session_id", "model_fingerprint",
                    "runtime_id", "runtime_generation", "access_domain", "expires_at_epoch"))
                identifier(body["operation_id"])
                enrollment = SessionEnrollment(principal, body["session_id"], body["model_fingerprint"],
                    body["runtime_id"], body["runtime_generation"], body["access_domain"], body["expires_at_epoch"])
                health = await backend.client.call("enroll_session", asdict(enrollment),
                                                   command_id="enroll-" + digest(body["operation_id"]))
                # Real worker profile validation precedes local authority membership.
                await backend.refresh_readiness()
                authority.remember_session(enrollment)
                result = {"schema": "occ3.sessions.enrolled.v1", "operation_id": body["operation_id"],
                    "session_id": body["session_id"], "principal_digest": principal,
                    "runtime_id": enrollment.runtime_epoch, "runtime_generation": enrollment.runtime_generation,
                    "model_fingerprint": enrollment.model_fingerprint, "expires_at_epoch": enrollment.expires_at_epoch,
                    "readiness": health}
            elif operation == "publish":
                if value.get("schema") == "occ3.publish.v2":
                    # Retained capture: no tool call exists to name, so the v2
                    # body carries dispatch_mode instead of tool_call_id and
                    # the publication parses with the empty tool identity.
                    body = _fields(value, "occ3.publish.v2", ("publication_id", "source_handle", "source_session_id",
                        "receiver_session_id", "message_id", "source_turn_nonce", "dispatch_mode", "expires_at_epoch"))
                    publication = Publication(principal_digest=principal, tool_call_id="", **body)
                else:
                    body = _fields(value, "occ3.publish.v1", ("publication_id", "source_handle", "source_session_id",
                        "receiver_session_id", "message_id", "source_turn_nonce", "tool_call_id", "expires_at_epoch"))
                    publication = Publication(principal_digest=principal, **body)
                authority.check_session(principal, publication.source_session_id)
                authority.check_session(principal, publication.receiver_session_id)
                record = await backend.client.call("publish", asdict(publication),
                                                   command_id="publish-" + digest(publication.publication_id))
                result = {"schema": "occ3.publication.v1", "publication_id": publication.publication_id,
                    "binding_digest": publication.binding_digest, "state": record["state"],
                    "source_handle": record["source_handle"], "expires_at_epoch": record["expires_at_epoch"]}
            elif operation == "status":
                body = _fields(value, "occ3.publication.status.v1", ("publication_id",))
                identifier(body["publication_id"])
                record = await backend.client.call("publication_status", {**body, "principal_digest": principal})
                result = {"schema": "occ3.publication.v1", "publication_id": body["publication_id"], "state": record["state"]}
                if record["state"] != "absent":
                    result.update(binding_digest=record["binding_digest"], source_handle=record["source_handle"],
                                  expires_at_epoch=record["expires_at_epoch"])
            else:
                body = _fields(value, "occ3.retire.v1", ("operation_id", "source_handle", "source_session_id"))
                for v in body.values():
                    identifier(v)
                authority.check_session(principal, body["source_session_id"])
                record = await backend.client.call("retire", {"source_handle": body["source_handle"],
                    "source_session_id": body["source_session_id"], "principal_digest": principal},
                    command_id="retire-native-" + digest(body["operation_id"]))
                result = {"schema": "occ3.retired.v1", "operation_id": body["operation_id"],
                    "source_handle": body["source_handle"], "revoked": record["revoked"],
                    "gpu_freed": record["gpu_freed"], "state": record["state"]}
            bounded_payload(result)
            return JSONResponse(result, headers={"cache-control": "no-store"})
        except (LatentError, AttachmentError) as error:
            return control_error_response(error)
        except Exception:
            return control_error_response()

    # Avoid postponed local Request annotation evaluation in FastAPI.
    async def enroll(request):
        return await execute(request, "enroll")
    async def publish(request):
        return await execute(request, "publish")
    async def status(request):
        return await execute(request, "status")
    async def retire(request):
        return await execute(request, "retire")
    for endpoint in (enroll, publish, status, retire):
        endpoint.__annotations__["request"] = Request
    for path, endpoint in (("sessions/enroll", enroll), ("publish", publish),
                           ("publication/status", status), ("retire", retire)):
        router.add_api_route("/occ3/v1/native/" + path, endpoint, methods=["POST"])
    return router
