"""Rank-local ownership/CAS registry; never exports GPU objects in metadata."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .attachment_contract import (CAPTURE_MODE_CONTENT_ONLY, CaptureWorkerDescriptor,
                                  ConsumeWorkerDescriptor, digest)
from .latent_types import (DISPATCH_MODEL_TOOL_V1, DISPATCH_ORCHESTRATOR_V1, LatentError,
                           Publication, RuntimeProfile, SessionEnrollment, require, sha)
from .span_policy import POLICY_NO_TOOL


@dataclass(repr=False)
class StateRecord:
    handle: str
    state_generation: int
    descriptor: CaptureWorkerDescriptor
    expires_at_epoch: int
    capture_budget: int
    reserved_bytes: int
    state: str = "reserved"
    prefix: Any = None
    retained: list = field(default_factory=list)
    events: list = field(default_factory=list)
    live_refs: list = field(default_factory=list)
    accepted_count: int = 0
    retained_tokens: list = field(default_factory=list)
    capture_commit_id: str | None = None
    publication: Publication | None = None
    consumer: ConsumeWorkerDescriptor | None = None
    injected_positions: set = field(default_factory=set)
    injection_commit_id: str | None = None
    source_metadata: dict = field(default_factory=dict)
    terminal_metadata: dict = field(default_factory=dict)
    output_binding: dict | None = None
    revoked: bool = False
    gpu_freed: bool = False
    admitted_capture: bool = False
    admitted_consume: bool = False
    inflight: int = 0

    def snapshot(self):
        return {"source_handle": self.handle, "state_generation": self.state_generation,
                "state": self.state, "expires_at_epoch": self.expires_at_epoch,
                "revoked": self.revoked, "gpu_freed": self.gpu_freed,
                "source": dict(self.source_metadata), "terminal": dict(self.terminal_metadata)}


class LatentRegistry:
    def __init__(self, profile: RuntimeProfile, *, epoch=time.time, monotonic=time.monotonic):
        self.profile, self.epoch, self.monotonic = profile, epoch, monotonic
        self.records: dict[str, StateRecord] = {}
        self.attempts: dict[str, str] = {}
        self.sessions: dict[tuple[str, str], SessionEnrollment] = {}
        self.publications: dict[str, Publication] = {}
        self.used_nonces: dict[str, str] = {}
        self.charged_bytes = 0
        self.lock = threading.RLock()

    def enroll(self, enrollment: SessionEnrollment):
        with self.lock:
            self._profile(enrollment.runtime_epoch, enrollment.runtime_generation,
                          enrollment.model_fingerprint)
            self._expiry(enrollment.expires_at_epoch)
            key = (enrollment.principal_digest, enrollment.session_id)
            old = self.sessions.get(key)
            require(old is None or old == enrollment, "identity")
            require(old is not None or len(self.sessions) < self.profile.max_records, "quota")
            self.sessions[key] = enrollment

    def _profile(self, epoch, generation, fingerprint):
        p = self.profile
        require((epoch, generation, fingerprint) ==
                (p.runtime_epoch, p.runtime_generation, p.model_fingerprint), "identity")

    def _expiry(self, expiry):
        require(self.epoch() < expiry <= self.epoch() + self.profile.max_ttl_seconds + 1, "expired")

    def session(self, principal, session):
        value = self.sessions.get((principal, session))
        require(value is not None and value.expires_at_epoch > self.epoch(), "identity")
        return value

    def reserve_capture(self, descriptor, handle, generation, expiry, budget, reserved_bytes):
        with self.lock:
            require(type(descriptor) is CaptureWorkerDescriptor)
            self._profile(descriptor.runtime_epoch, descriptor.runtime_generation, descriptor.model_fingerprint)
            require(descriptor.method_id == self.profile.method and descriptor.deadline > self.monotonic(), "expired")
            owner = self.session(descriptor.principal_digest, descriptor.source_session_id)
            self._expiry(expiry)
            require(expiry <= owner.expires_at_epoch and 0 < reserved_bytes <= budget == descriptor.max_capture_bytes, "quota")
            existing = self.attempts.get(descriptor.attempt_id)
            if existing:
                record = self.records[existing]
                require(record.handle == handle and record.descriptor == descriptor and not record.revoked, "replay")
                return record
            require(handle not in self.records and len(self.records) < self.profile.max_records, "quota")
            require(self.charged_bytes + reserved_bytes <= self.profile.max_live_bytes, "quota")
            require(sum(not r.revoked and r.state in ("reserved", "capturing", "leased", "injected")
                        for r in self.records.values()) < self.profile.max_active, "quota")
            record = StateRecord(handle, generation, descriptor, expiry, budget, reserved_bytes)
            self.records[handle] = record
            self.attempts[descriptor.attempt_id] = handle
            self.charged_bytes += reserved_bytes
            return record

    def source_for_attempt(self, attempt_id):
        handle = self.attempts.get(attempt_id)
        require(handle in self.records, "state")
        return self.records[handle]

    def get(self, handle, *, allow_retired=False):
        record = self.records.get(handle)
        require(record is not None, "state")
        if not allow_retired:
            require(not record.revoked and record.expires_at_epoch > self.epoch(), "expired")
        return record

    def publish(self, publication: Publication):
        with self.lock:
            old = self.publications.get(publication.publication_id)
            if old is not None:
                require(old == publication, "replay")
                return self.get(old.source_handle, allow_retired=True).snapshot()
            record = self.get(publication.source_handle)
            d = record.descriptor
            require(record.state == "ready" and record.output_binding is not None, "not_ready")
            require((publication.principal_digest, publication.source_session_id, publication.source_turn_nonce) ==
                    (d.principal_digest, d.source_session_id, d.source_turn_nonce), "identity")
            source = self.session(publication.principal_digest, publication.source_session_id)
            target = self.session(publication.principal_digest, publication.receiver_session_id)
            require(source.access_domain == target.access_domain, "identity")
            if publication.dispatch_mode == DISPATCH_ORCHESTRATOR_V1:
                # Retained capture: the turn ended with no tool call, so the
                # bound output identity is the empty tool pair, and the record
                # must come from a v3 content-only capture under the no-tool
                # span policy (the descriptor mirrors the negotiated intent).
                require(record.output_binding["tool_call_id"] == ""
                        and record.output_binding["tool_name"] == ""
                        and d.span_policy == POLICY_NO_TOOL
                        and d.capture_mode == CAPTURE_MODE_CONTENT_ONLY, "identity")
            elif publication.dispatch_mode == DISPATCH_MODEL_TOOL_V1:
                require(publication.tool_call_id == record.output_binding["tool_call_id"]
                        and record.output_binding["tool_name"] == "occ3_handoff", "identity")
            else:
                # Unknown dispatch modes are refused, never defaulted.
                require(False, "invalid")
            require(self.epoch() < publication.expires_at_epoch <= min(record.expires_at_epoch, target.expires_at_epoch), "expired")
            record.publication, record.state = publication, "published"
            self.publications[publication.publication_id] = publication
            return record.snapshot()

    def reserve_consume(self, descriptor):
        with self.lock:
            require(type(descriptor) is ConsumeWorkerDescriptor)
            self._profile(descriptor.runtime_epoch, descriptor.runtime_generation, descriptor.model_fingerprint)
            record = self.get(descriptor.opaque_handle)
            self.session(descriptor.principal_digest, descriptor.receiver_session_id)
            pub = record.publication
            require(pub is not None and pub.expires_at_epoch > self.epoch(), "identity")
            require((descriptor.principal_digest, descriptor.receiver_session_id, descriptor.message_id) ==
                    (pub.principal_digest, pub.receiver_session_id, pub.message_id), "identity")
            require(descriptor.state_generation == record.state_generation and
                    descriptor.source_capture_commit_id == record.capture_commit_id and
                    descriptor.deadline > self.monotonic(), "identity")
            source = record.source_metadata
            require((descriptor.expected_rank_count, descriptor.expected_prefix_tokens, descriptor.hidden_dim,
                     descriptor.dtype, descriptor.tp_layout_digest, descriptor.method_id) ==
                    (source["expected_ranks"], source["retained_count"], source["hidden_dim"],
                     source["dtype"], source["layout_digest"], self.profile.method), "identity")
            if record.consumer is not None:
                require(record.consumer == descriptor, "replay")
                return record
            require(record.state == "published" and descriptor.attempt_id not in self.attempts, "replay")
            nonce_key = digest([descriptor.principal_digest, descriptor.receiver_session_id, descriptor.consume_nonce])
            require(nonce_key not in self.used_nonces, "replay")
            require(sum(r.state in ("reserved", "capturing", "leased", "injected") and not r.revoked
                        for r in self.records.values()) < self.profile.max_active, "quota")
            self.used_nonces[nonce_key] = descriptor.reservation_id
            self.attempts[descriptor.attempt_id] = record.handle
            record.consumer, record.state = descriptor, "leased"
            return record

    def bind_output(self, attempt_id, binding):
        with self.lock:
            record = self.source_for_attempt(attempt_id)
            require(record.state == "ready" and not record.revoked, "state")
            require(set(binding) == {"native_rid", "response_id", "output_sha256", "tool_call_id",
                                     "tool_name", "tool_provenance_digest"}, "invalid")
            require(binding["native_rid"] == record.descriptor.native_rid
                    and binding["response_id"] == binding["native_rid"], "identity")
            sha(binding["output_sha256"])
            sha(binding["tool_provenance_digest"])
            if record.output_binding is not None:
                require(record.output_binding == binding, "replay")
            record.output_binding = dict(binding)
            return record.snapshot()

    def retire(self, attempt_id=None, handle=None):
        with self.lock:
            if handle is None:
                handle = self.attempts.get(attempt_id)
            if handle not in self.records:
                return {"state": "absent", "revoked": True, "gpu_freed": True}
            record = self.records[handle]
            record.revoked, record.state = True, "retired"
            self.reap()
            return record.snapshot()

    def reap(self):
        """No synchronize/free while any producer/consumer event is incomplete."""
        with self.lock:
            for record in self.records.values():
                if record.expires_at_epoch <= self.epoch():
                    record.revoked, record.state = True, "expired"
                if (record.revoked and not record.gpu_freed and record.inflight == 0
                        and all(e.query() for e in record.events)):
                    record.prefix = None
                    record.retained.clear()
                    record.live_refs.clear()
                    record.events.clear()
                    record.retained_tokens.clear()
                    record.gpu_freed = True
                    self.charged_bytes -= record.reserved_bytes
