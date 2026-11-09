"""Frontend control bridge and native chat backend. No model/tensor imports."""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import asdict

from .attachment_contract import (AttachmentError, BackendReadiness, CaptureWorkerDescriptor,
                                  ConsumeWorkerDescriptor, Reservation, SourceEnrollment,
                                  SPAN_POLICIES_V2, digest)
from .span_policy import resolve_runtime_markers
from .latent_types import LatentError, admission_probe, control_request, require, validate_control_payload
from .use_receipts import (SourceCaptureCommit, SourceCaptureEvidence, TerminalUseEvidence,
                           SPAN_BLOCK_FIELDS)

logger = logging.getLogger(__name__)

#: Same switch the worker's _CalibrationCapture.from_env reads. The frontend
#: cannot see the worker's collector object, but both processes share the
#: server environment, so this is the earliest seam that can refuse a v1
#: capture on a collection run with a clear 4xx instead of a silently empty corpus.
CALIBRATION_CAPTURE_ENV = "OCC3_CALIBRATION_CAPTURE_DIR"


def calibration_collection_armed(environ=None):
    """True when the server was started for calibration-corpus collection."""
    environ = os.environ if environ is None else environ
    return bool(environ.get(CALIBRATION_CAPTURE_ENV, ""))


#: The 15 capture-commit fields the worker reports for every source capture.
#: Method identity (method_id, artifact_digest) and the bounded occ3.span block
#: ride the same source dict but are projected separately.
_SOURCE_COMMIT_FIELDS = ("capture_commit_id", "accepted_span_start", "accepted_span_end",
                          "retained_count", "predictor_position_offset", "representation",
                          "layer_index", "hidden_dim", "dtype", "layout_digest", "ready_ranks",
                          "expected_ranks", "accepted_sequence_digest", "expires_at_epoch",
                          "byte_count")


#: Seconds past an attempt's native deadline before its frontend state is
#: pruned. Covers a quarantined retire (control timeout <= 120 s).
ATTEMPT_PRUNE_GRACE_SECONDS = 150

#: How many dispatch timeouts an unanswered control op may stay in the
#: bounded pending table before the entry is released. The worker always
#: replies to Occ3ControlReq, so a reply missing after several dispatch
#: timeouts is a lost IPC reply, not a slow operation.
UNCERTAIN_TTL_MULTIPLE = 4


class ControlClient:
    def __init__(self, tokenizer_manager, *, timeout_seconds=30, max_pending=64):
        require(0 < timeout_seconds <= 120 and 0 < max_pending <= 256)
        require(getattr(tokenizer_manager, "_occ3_control_client", None) is None, "state")
        self.tm, self.timeout_seconds, self.max_pending = tokenizer_manager, timeout_seconds, max_pending
        # command_id -> (envelope digest, dispatch monotonic). The digest
        # identifies the dispatched envelope; the timestamp bounds the entry.
        self.pending = {}
        self._uncertain = {}
        # command_id -> retirement monotonic. An aged-out (swept) command id
        # stays unusable for a further sweep window: the operation may still
        # be running server-side, so a caller reusing the id would let a
        # merely-delayed reply for the old operation settle the new future
        # (deliver matches command_id only, and the result envelope carries
        # no request echo to compare against).
        self._retired = {}
        self._operations = {}
        self.closed = False
        tokenizer_manager._occ3_control_client = self

    def _sweep_uncertain(self):
        """Release pending-table slots for replies that can no longer arrive.

        A lost control reply (operation cancelled or timed out after the
        transport accepted bytes) otherwise leaks its entry forever, and
        max_pending leaked entries refuse every later control op with
        "quota" until process restart. Fail-closed on uncertainty: an
        aged-out entry is released from quota accounting only. It is never
        treated as proof the operation did not run, and a late reply for it
        is ignored exactly like any unknown command id (the caller's future
        is already settled or cancelled by then). The swept id moves to a
        bounded retirement table for one further sweep window: the operation
        may still be live server-side, and a reused id would otherwise let a
        merely-delayed reply for the old operation settle the new future.
        Entries with a live, cancellation-resistant operation task still
        hold their quota slot through self._operations, which this sweep
        never touches.
        """
        now = time.monotonic()
        horizon = self.timeout_seconds * UNCERTAIN_TTL_MULTIPLE
        cutoff = now - horizon
        stale = [command_id for command_id, (_, dispatched_at) in self._uncertain.items()
                 if dispatched_at <= cutoff]
        for command_id in stale:
            self._retired[command_id] = now
            del self._uncertain[command_id]
        expired = [command_id for command_id, retired_at in self._retired.items()
                   if retired_at <= cutoff]
        for command_id in expired:
            del self._retired[command_id]
        if stale or expired:
            self._detach_if_settled()

    async def call(self, operation, payload, command_id=None):
        self._sweep_uncertain()
        require(not self.closed and len(self._uncertain) < self.max_pending
                and len(self.pending) < self.max_pending and len(self._operations) < self.max_pending, "quota")
        from sglang.srt.managers.io_struct import Occ3ControlReq
        command_id = command_id or uuid.uuid4().hex
        envelope = control_request(command_id, operation, payload)
        require(command_id not in self.pending and command_id not in self._uncertain
                and command_id not in self._retired, "replay")
        future = asyncio.get_running_loop().create_future()
        self.pending[command_id] = future
        try:
            self.tm.auto_create_handle_loop()
            request = Occ3ControlReq(rid=command_id, payload=envelope)
            # Dispatch can fail after transport accepted bytes. Keep ownership
            # until a valid same-command terminal reply is actually observed.
            self._uncertain[command_id] = (digest(envelope), time.monotonic())
            async def dispatch_and_wait():
                await self.tm._async_dispatch_to_scheduler(request)
                return await asyncio.shield(future)
            operation = asyncio.create_task(dispatch_and_wait())
            self._operations[command_id] = operation
            operation.add_done_callback(lambda task: self._operation_done(command_id, task))
            done, _ = await asyncio.wait({operation}, timeout=self.timeout_seconds)
            if not done:
                operation.cancel()
                raise asyncio.TimeoutError()
            reply = operation.result()
            validate_control_payload(reply, result=True)
            if not reply["ok"]:
                raise LatentError(reply["error"])
            return reply["result"]
        finally:
            operation = self._operations.get(command_id)
            if operation is not None and not operation.done():
                operation.cancel()  # Never await cancellation-resistant transport.
            self.pending.pop(command_id, None)
            if not future.done():
                future.cancel()

    def _operation_done(self, command_id, task):
        self._operations.pop(command_id, None)
        if not task.cancelled():
            task.exception()  # Consume a late failure without logging payloads.
        self._detach_if_settled()

    def _detach_if_settled(self):
        if self.closed and not self.has_uncertain_control and getattr(self.tm, "_occ3_control_client", None) is self:
            delattr(self.tm, "_occ3_control_client")

    @property
    def has_uncertain_control(self):
        return bool(self._uncertain or any(not task.done() for task in self._operations.values()))

    def deliver(self, envelope):
        value = validate_control_payload(envelope, result=True)
        command_id = value["command_id"]
        if command_id not in self._uncertain:
            return
        self._uncertain.pop(command_id)
        future = self.pending.get(command_id)
        if future is not None and not future.done():
            future.set_result(value)
        self._detach_if_settled()

    async def close(self):
        self.closed = True
        for future in self.pending.values():
            if not future.done():
                future.cancel()
        self.pending.clear()
        for operation in self._operations.values():
            if not operation.done():
                operation.cancel()
        self._detach_if_settled()


def deliver_control(tokenizer_manager, payload):
    client = getattr(tokenizer_manager, "_occ3_control_client", None)
    if client is not None:
        client.deliver(payload)


class LatentBackend:
    def __init__(self, client, profile, authority, *, max_output_bytes=4194304, required_seconds=1.0):
        require(0 < max_output_bytes <= 4194304 and 0 < required_seconds <= 30)
        self.client, self.profile, self.authority = client, profile, authority
        self.max_output_bytes, self.required_seconds = max_output_bytes, required_seconds
        self._health = None
        self._health_at = 0.0
        self.attempts = {}
        self.closed = False

    async def refresh_readiness(self):
        health = await self.client.call("health", {})
        p = self.profile
        require((health["runtime_epoch"], health["runtime_generation"], health["model_fingerprint"],
                 health["method_id"], health["hidden_dim"], health["layer_index"], health["ready_ranks"],
                 health["expected_ranks"], health["profile_digest"]) ==
                (p.runtime_epoch, p.runtime_generation, p.model_fingerprint, p.method, p.hidden_dim,
                 p.layer_index, p.expected_ranks, p.expected_ranks, p.binding_digest), "identity")
        require(all(health[x] is True for x in ("source_capture", "continuous_prefix", "terminal_journal")), "unavailable")
        self._health, self._health_at = health, time.monotonic()
        return dict(health)

    def readiness(self):
        # A successful ordered all-rank control reply is required. Caller refreshes
        # before enrollment/dispatch; stale startup receipts never authorize use.
        ready = not self.closed and self._health is not None and time.monotonic() - self._health_at < 30
        return BackendReadiness(frontend=ready, tokenizer_descriptor=ready, scheduler=ready,
                                all_workers=ready, terminal_use=ready, authenticated_owner=ready,
                                source_capture=ready)

    def _identity(self, identity, session):
        p = self.profile
        require((identity.session_id, identity.model_fingerprint, identity.runtime_id, identity.runtime_generation) ==
                (session, p.model_fingerprint, p.runtime_epoch, p.runtime_generation), "identity")
        self.authority.check_session(identity.principal_digest, session)

    def _slot(self, attempt):
        self._prune()
        require(attempt not in self.attempts and len(self.attempts) < self.profile.max_records, "replay", detail="latent_backend._slot.attempt_slot")

    def _prune(self):
        """Drop per-turn frontend state that no caller can use any more.

        An entry is dropped once its retire was acknowledged, or once its
        native deadline (monotonic) is more than ATTEMPT_PRUNE_GRACE_SECONDS
        in the past: every reserve/bind/finish/retire for an attempt runs
        inside its own turn, and a quarantined retire is bounded by the
        control timeout. Attempt ids are fresh per request, and worker-side
        replay protection lives in the registry, so pruning never reopens a
        replay. Without this, max_records (256) armed turns per process
        exhausted the table and every later turn was refused.
        """
        cutoff = time.monotonic() - ATTEMPT_PRUNE_GRACE_SECONDS
        stale = [key for key, state in self.attempts.items()
                 if state.get("retired") or state["descriptor"].deadline <= cutoff]
        for key in stale:
            del self.attempts[key]

    def _reservation(self, descriptor, binding, native):
        require(native.request.rid == descriptor.native_rid and native.max_new_tokens > 0, "identity", detail="latent_backend._reservation.native_identity")
        require(descriptor.deadline <= native.deadline_monotonic and
                descriptor.deadline > time.monotonic() + self.required_seconds, "expired", detail="latent_backend._reservation.native_bounds")
        # A positive admission floor, NOT a prediction of full inference latency.
        return Reservation(descriptor.attempt_id, binding, descriptor.native_rid,
                           descriptor.principal_digest, self.required_seconds, self.max_output_bytes,
                           native.max_new_tokens, descriptor.reservation_id)

    async def reserve(self, attachment, identity, attempt_id, native):
        self._identity(identity, attachment.receiver_session_id)
        require((attachment.runtime_id, attachment.runtime_generation, attachment.model_fingerprint, attachment.method)
                == (self.profile.runtime_epoch, self.profile.runtime_generation,
                    self.profile.model_fingerprint, self.profile.method), "identity")
        require(attachment.expires_at_epoch > time.time(), "expired")
        offset = getattr(native, "prefix_offset", 0)
        require(type(offset) is int and 0 <= offset <= 2**20, "profile")
        layout = getattr(native, "prefix_layout", "marker" if offset else "prepend")
        require(layout in ("prepend", "marker", "last_user")
                and (layout != "prepend") == (offset > 0), "profile")
        self._slot(attempt_id)
        await self.refresh_readiness()
        record = await self.client.call("status", {"source_handle": attachment.state_handle,
                                                  "principal_digest": identity.principal_digest})
        require(record["state"] == "published" and record["expires_at_epoch"] >= attachment.expires_at_epoch, "state")
        source = record["source"]
        descriptor = ConsumeWorkerDescriptor(
            attempt_id=attempt_id, reservation_id=uuid.uuid4().hex, native_rid=native.request.rid,
            principal_digest=identity.principal_digest, attachment_digest=attachment.binding_digest,
            receiver_session_id=attachment.receiver_session_id, message_id=attachment.message_id,
            consume_nonce=attachment.request_id, opaque_handle=attachment.state_handle,
            state_generation=record["state_generation"], runtime_generation=attachment.runtime_generation,
            source_capture_commit_id=source["capture_commit_id"], runtime_epoch=attachment.runtime_id,
            model_fingerprint=attachment.model_fingerprint, method_id=attachment.method,
            tp_layout_digest=source["layout_digest"], expected_rank_count=source["expected_ranks"],
            expected_prefix_tokens=source["retained_count"], hidden_dim=source["hidden_dim"], dtype=source["dtype"],
            deadline=native.deadline_monotonic, prefix_offset=offset,
            prefix_layout=layout)
        reservation = self._reservation(descriptor, attachment.binding_digest, native)
        self.attempts[attempt_id] = {"descriptor": descriptor, "reservation": reservation,
                                     "principal": identity.principal_digest}
        try:
            await self.client.call("reserve_consume", {"descriptor": asdict(descriptor)},
                                   command_id="reserve-" + descriptor.reservation_id)
        except LatentError as error:
            if error.code != "busy":
                raise
            # Proven non-use, same contract as enroll_source: the registry
            # declines "busy" before taking the nonce, attempt or lease.
            self.attempts[attempt_id]["retired"] = True
            raise AttachmentError("scheduler_busy", 503) from None
        return reservation

    def bind(self, reservation, native):
        state = self.attempts.get(reservation.attempt_id)
        require(state is not None and state.get("reservation") == reservation, "identity")
        require(native.rid == reservation.native_rid and getattr(native, "occ3_worker_descriptor", None) is None, "replay")
        native.occ3_worker_descriptor = state["descriptor"]

    async def finish(self, reservation, completion):
        state = self.attempts.get(reservation.attempt_id)
        require(state is not None and state["reservation"] == reservation and completion.native_rid == reservation.native_rid, "identity")
        d = state["descriptor"]
        record = await self.client.call("status", {"attempt_id": d.attempt_id, "principal_digest": d.principal_digest})
        require(record["state"] == "terminal" and not record["revoked"], "not_ready")
        t = record["terminal"]
        require(t["native_rid"] == d.native_rid and record["source"]["capture_commit_id"] == d.source_capture_commit_id, "identity")
        require(t["completed"] is True and t["gpu_resident"] is True and t["num_retractions"] == 0
                and t["private_cache_namespace"] is True, "state")
        return TerminalUseEvidence(d.attempt_id, d.attachment_digest, d.native_rid, d.principal_digest,
            d.source_capture_commit_id, t["injection_commit_id"], d.runtime_epoch, d.runtime_generation,
            d.model_fingerprint, d.receiver_session_id, completion.output_sha256,
            t["injected_prefix_count"], t["ready_ranks"], t["expected_ranks"], True, False, 0, True, True,
            reservation.reservation_id)

    @admission_probe("enroll_source")
    async def enroll_source(self, intent, identity, attempt_id, native_rid, deadline):
        self._identity(identity, intent.source_session_id)
        require(intent.method == self.profile.method and intent.runtime_id == self.profile.runtime_epoch
                and intent.runtime_generation == self.profile.runtime_generation
                and intent.model_fingerprint == self.profile.model_fingerprint, "identity", detail="latent_backend.enroll_source.profile_identity")
        require(intent.retained_count_limit in (8, 16, 32, 64, 128), "profile", detail="latent_backend.enroll_source.retained_limit")
        require(native_rid != intent.source_request_id, "identity", detail="latent_backend.enroll_source.source_request_identity")
        alpha = self.profile.resolve_label_anchor(intent.label_anchor_alpha)
        self._slot(attempt_id)
        health = await self.refresh_readiness()
        marker_profile = ""
        if intent.span_policy in SPAN_POLICIES_V2:
            # v2 opt-in: the marker profile must be tokenizer-attested in this
            # process before any reservation exists (doc section 6.1). The
            # marker ids themselves still never travel on the descriptor; they
            # use the per-request custom_params channel. The model config
            # supplies the generation-config stop set (GLM role tokens).
            marker_profile = resolve_runtime_markers(
                self.client.tm.tokenizer, getattr(self.client.tm, "model_config", None))[0].profile
        elif calibration_collection_armed():
            # A v1 turn is structurally uncollectable: the calibration handoff
            # needs the v2 span selection. Refuse the enrollment loudly rather
            # than run a collection window that accepts turns and writes no rows.
            logger.error("occ3-bridge: calibration collection is armed (%s) but the source "
                         "intent arms span policy v1 (%s); v1 turns are never collected. "
                         "Re-arm the client with a content_tail_v2 family policy "
                         "(OCC3_SPAN_POLICY) or unset the collection directory.",
                         CALIBRATION_CAPTURE_ENV, intent.span_policy)
            raise LatentError("profile")
        descriptor = CaptureWorkerDescriptor(attempt_id=attempt_id, reservation_id=uuid.uuid4().hex,
            native_rid=native_rid, principal_digest=identity.principal_digest,
            runtime_epoch=intent.runtime_id, model_fingerprint=intent.model_fingerprint,
            method_id=intent.method, deadline=min(deadline, identity.deadline_monotonic),
            capture_id=intent.capture_id, source_session_id=intent.source_session_id,
            source_turn_nonce=intent.turn_nonce, runtime_generation=intent.runtime_generation,
            span_policy=intent.span_policy, max_prefix_tokens=intent.retained_count_limit,
            max_capture_bytes=intent.max_capture_bytes, marker_profile=marker_profile,
            capture_mode=intent.capture_mode, label_anchor_alpha=alpha)
        enrollment = SourceEnrollment(attempt_id, intent.binding_digest, native_rid,
            identity.principal_digest, uuid.uuid4().hex, int(uuid.uuid4().hex[:16], 16),
            intent.expires_at_epoch, intent.source_session_id, intent.runtime_id, intent.runtime_generation,
            intent.model_fingerprint, intent.method, health["expected_ranks"], health["tp_layout_digest"],
            health["layer_index"], health["hidden_dim"], health["dtype"])
        # Store before await so cancellation cleanup can find uncertain enrollment.
        self.attempts[attempt_id] = {"descriptor": descriptor, "enrollment": enrollment,
                                    "principal": identity.principal_digest}
        try:
            await self.client.call("reserve_capture", {"descriptor": asdict(descriptor),
                "source_handle": enrollment.source_handle, "state_generation": enrollment.state_generation,
                "expires_at_epoch": enrollment.expires_at_epoch}, command_id="reserve-" + descriptor.reservation_id)
        except LatentError as error:
            if error.code not in ("busy", "quota"):
                raise
            # Proven non-use: the registry declines "busy" and every
            # reserve-capture quota require (byte, record, tombstone or
            # budget-owner capacity) BEFORE creating any record, attempt or
            # byte charge, on every rank (a rank mismatch would have come
            # back as "ranks"). Nothing was dispatched and no receipt can
            # exist, so this is the same retryable contract as the
            # scheduler busy-batch refusal: a distinct 503 with Retry-After
            # instead of a terminal 409 that strands the sender. The local
            # entry is marked retired so it is pruned.
            self.attempts[attempt_id]["retired"] = True
            raise AttachmentError("scheduler_busy", 503) from None
        return enrollment

    @admission_probe("prepare_source")
    async def prepare_source(self, enrollment, native):
        state = self.attempts.get(enrollment.attempt_id)
        require(state is not None and state.get("enrollment") == enrollment and "reservation" not in state, "replay", detail="latent_backend.prepare_source.source_reservation")
        reservation = self._reservation(state["descriptor"], enrollment.intent_digest, native)
        state["reservation"] = reservation
        return reservation

    def bind_source(self, enrollment, reservation, native):
        state = self.attempts.get(enrollment.attempt_id)
        require(state is not None and state.get("enrollment") == enrollment, "identity")
        self.bind(reservation, native)

    @admission_probe("finish_source")
    async def finish_source(self, enrollment, reservation, completion, provenance):
        state = self.attempts.get(enrollment.attempt_id)
        require(state is not None and state.get("enrollment") == enrollment and state.get("reservation") == reservation, "identity")
        require(completion.native_rid == enrollment.native_rid and len(provenance.tool_calls) <= 1, "identity")
        binding = {"native_rid": completion.native_rid, "response_id": completion.native_rid,
                   "output_sha256": completion.output_sha256,
                   "tool_call_id": provenance.tool_calls[0].call_id if provenance.tool_calls else "",
                   "tool_name": provenance.tool_calls[0].name if provenance.tool_calls else "",
                   "tool_provenance_digest": provenance.binding_digest}
        record = await self.client.call("bind_output", {"attempt_id": enrollment.attempt_id,
            "principal_digest": enrollment.principal_digest, "binding": binding},
            command_id="output-" + reservation.reservation_id)
        require(record["state"] == "ready" and record["source_handle"] == enrollment.source_handle
                and record["state_generation"] == enrollment.state_generation, "identity")
        source = record["source"]
        require(isinstance(source, dict), "identity")
        # The worker's method_metadata() rides the same source dict (method_id
        # and, for a calibrated method, artifact_digest) since wave 445d1cdd4.
        # Those are method identity, not capture-commit fields: project the
        # commit, cross-check the method id, never splat the worker metadata.
        method_id = source.get("method_id")
        require(method_id is None or method_id == self.profile.method, "identity")
        artifact_digest = source.get("artifact_digest")
        require(artifact_digest is None
                or (type(artifact_digest) is str and 1 <= len(artifact_digest) <= 64), "identity")
        require(source.get("label_anchor_alpha", 0.0) == state["descriptor"].label_anchor_alpha, "identity")
        require(all(name in source for name in _SOURCE_COMMIT_FIELDS), "identity")
        block = {name: source.get(name) for name in SPAN_BLOCK_FIELDS}
        commit = SourceCaptureCommit(**{name: source[name] for name in _SOURCE_COMMIT_FIELDS},
                                     label_anchor_alpha=source.get("label_anchor_alpha", 0.0), **block)
        require(commit.ready_ranks == commit.expected_ranks == enrollment.expected_ranks
                and commit.layout_digest == enrollment.layout_digest, "ranks")
        return SourceCaptureEvidence(enrollment, reservation.reservation_id, commit,
            completion.output_sha256, completion.native_rid, provenance.binding_digest, True, True, 0)

    @admission_probe("release_unused")
    async def release_unused_attempt(self, attempt_id, native_rid):
        """Worker-side proof that a busy-declined attempt never ran.

        Consume: the worker returns the lease and its nonce to the published
        state (registry.release_unused_lease re-proves non-admission first).
        Capture: the record must still be "reserved" (admission flips it to
        "capturing"); the caller's normal retire then revokes it. Any other
        state raises, and the caller keeps its replay fence.
        """
        state = self.attempts.get(attempt_id)
        require(state is not None and not state.get("retired"), "state")
        descriptor = state["descriptor"]
        require(descriptor.native_rid == native_rid, "identity")
        if type(descriptor) is ConsumeWorkerDescriptor:
            await self.client.call("release_lease", {"attempt_id": attempt_id, "principal_digest": state["principal"]},
                                   command_id="release-" + descriptor.reservation_id)
            state["retired"] = True
            return "released"
        record = await self.client.call("status", {"attempt_id": attempt_id, "principal_digest": state["principal"]})
        require(record["state"] == "reserved" and not record["revoked"], "state")
        return "unadmitted"

    @admission_probe("backend_retire")
    async def retire(self, attempt_id, native_rid, dispatched):
        state = self.attempts.get(attempt_id)
        if state is None:
            return
        if native_rid is not None:
            require(state["descriptor"].native_rid == native_rid, "identity")
        await self.client.call("retire", {"attempt_id": attempt_id, "principal_digest": state["principal"]},
                               command_id="retire-" + state["descriptor"].reservation_id)
        state["retired"] = True

    async def close(self):
        self.closed = True
        self._health = None
        # Do not fabricate physical-free acknowledgments during installer rollback.
        await self.client.close()
