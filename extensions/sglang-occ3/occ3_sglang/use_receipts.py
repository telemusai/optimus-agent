"""Private completed-use evidence and byte-preserving native output validation.

These records are not GPU attestation by themselves. Only the injected trusted
backend may supply evidence from its actual worker completion/commit journal.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .attachment_contract import (
    AttachmentError, InferenceAttachment, MAX_RECEIPT_BYTES, Reservation,
    SourceCaptureReceipt, SourceEnrollment, bounded_identifier, canonical, digest,
    identifier, sha256_hex, strict_json, uint, MARKER_SCHEMA_V1, SOURCE_SCHEMA_V1,
    SOURCE_SCHEMA_V2, SPAN_POLICY_V1, SPAN_POLICIES_V2, SPAN_SCHEMA_V1, SPAN_TERMINATORS,
)

VALID_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "function_call"})


@dataclass(frozen=True, repr=False)
class NativeCompletion:
    native_rid: str
    finish_reason: str
    output_sha256: str
    output_bytes: int
    streaming: bool


@dataclass(frozen=True, repr=False)
class TerminalUseEvidence:
    attempt_id: str
    attachment_digest: str
    native_rid: str
    principal_digest: str
    source_capture_commit_id: str
    injection_commit_id: str
    runtime_id: str
    runtime_generation: int
    model_fingerprint: str
    receiver_session_id: str
    output_sha256: str
    injected_prefix_count: int
    ready_ranks: int
    expected_ranks: int
    completed: bool
    fallback_used: bool
    num_retractions: int
    private_cache_namespace: bool
    gpu_resident: bool
    reservation_id: str


def receipt_bytes(attachment: InferenceAttachment, limit: int = MAX_RECEIPT_BYTES) -> bytes:
    value = canonical({"attachment": attachment.wire(), "consumed": True,
                       "fallback_used": False, "injected_state": True})
    if len(value) > min(limit, MAX_RECEIPT_BYTES):
        raise AttachmentError("buffer_limit", 413)
    return value


def validate_evidence(attachment, reservation, descriptor, completion, evidence):
    if type(evidence) is not TerminalUseEvidence:
        raise AttachmentError("invalid_evidence")
    expected = (
        evidence.attempt_id == reservation.attempt_id,
        evidence.reservation_id == reservation.reservation_id == descriptor.reservation_id,
        evidence.source_capture_commit_id == descriptor.source_capture_commit_id,
        evidence.expected_ranks == descriptor.expected_rank_count,
        evidence.injected_prefix_count == descriptor.expected_prefix_tokens,
        evidence.attachment_digest == attachment.binding_digest == reservation.attachment_digest,
        evidence.native_rid == completion.native_rid == reservation.native_rid,
        evidence.principal_digest == reservation.principal_digest,
        evidence.runtime_id == attachment.runtime_id,
        evidence.runtime_generation == attachment.runtime_generation,
        evidence.model_fingerprint == attachment.model_fingerprint,
        evidence.receiver_session_id == attachment.receiver_session_id,
        evidence.output_sha256 == completion.output_sha256,
        evidence.completed is True, evidence.fallback_used is False,
        type(evidence.num_retractions) is int and evidence.num_retractions == 0,
        evidence.private_cache_namespace is True, evidence.gpu_resident is True,
        type(evidence.injected_prefix_count) is int and 0 < evidence.injected_prefix_count <= 128,
        type(evidence.ready_ranks) is int and type(evidence.expected_ranks) is int
        and evidence.ready_ranks == evidence.expected_ranks > 0,
        isinstance(evidence.source_capture_commit_id, str)
        and bool(re.fullmatch(r"[A-Za-z0-9._:/-]{1,256}", evidence.source_capture_commit_id)),
        isinstance(evidence.injection_commit_id, str)
        and bool(re.fullmatch(r"[A-Za-z0-9._:/-]{1,256}", evidence.injection_commit_id)),
    )
    if not all(expected):
        raise AttachmentError("invalid_evidence")


class NativeOutputAudit:
    """Inspect native SSE without changing any of its bytes or parser/tool deltas."""

    def __init__(self, max_bytes: int, *, source: bool = False):
        self.source = source
        self.tool_calls = {}
        self.max_bytes = max_bytes
        self.size = 0
        self.sha = hashlib.sha256()
        self.pending = b""
        self.rid = None
        self.finish_reason = None
        self.done = False

    def _object(self, obj):
        if not isinstance(obj, dict) or "error" in obj or obj.get("object") == "error":
            raise AttachmentError("native_completion_failed")
        rid = obj.get("id")
        if not isinstance(rid, str) or (self.rid is not None and rid != self.rid):
            raise AttachmentError("native_completion_failed")
        self.rid = rid
        choices = obj.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise AttachmentError("native_completion_failed")
        if not choices and self.finish_reason is None:
            # Usage-only frames are terminal ornaments: they may appear only
            # after the single successful finish, never before it.
            raise AttachmentError("native_completion_failed")
        for choice in choices:
            if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0:
                raise AttachmentError("native_completion_failed")
            if self.source:
                self._source_tools(choice)
            finish = choice.get("finish_reason")
            if finish is not None:
                if finish not in VALID_FINISH_REASONS or self.finish_reason is not None:
                    raise AttachmentError("native_completion_failed")
                self.finish_reason = finish
            elif self.finish_reason is not None and choice.get("delta"):
                raise AttachmentError("native_completion_failed")

    def _source_tools(self, choice):
        value = choice.get("delta", choice.get("message", {}))
        if not isinstance(value, dict) or value.get("function_call") is not None:
            raise AttachmentError("unsupported_profile")
        calls = value.get("tool_calls")
        if calls is None:
            return
        if not isinstance(calls, list):
            raise AttachmentError("native_completion_failed")
        streaming = "delta" in choice
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                raise AttachmentError("native_completion_failed")
            slot = call.get("index") if streaming else index
            if type(slot) is not int or slot != 0:
                # Conservative whole-response v1: never guess among multiple calls.
                raise AttachmentError("unsupported_profile")
            target = self.tool_calls.setdefault(slot, {"id": "", "name": ""})
            if call.get("type", "function") != "function":
                raise AttachmentError("unsupported_profile")
            function = call.get("function", {})
            if not isinstance(function, dict):
                raise AttachmentError("native_completion_failed")
            for key, fragment in (("id", call.get("id")), ("name", function.get("name"))):
                if fragment is not None:
                    if type(fragment) is not str or len(target[key]) + len(fragment) > 256:
                        raise AttachmentError("native_completion_failed")
                    target[key] += fragment

    def source_provenance(self, scope: str = SPAN_POLICY_V1):
        if not self.source:
            raise AttachmentError("unsupported_profile")
        calls = tuple(SourceToolCall(value["id"], value["name"]) for value in self.tool_calls.values())
        if self.finish_reason in ("tool_calls", "function_call") and not calls:
            raise AttachmentError("native_completion_failed")
        return SourceToolProvenance(calls, scope)

    def _account(self, chunk: bytes):
        self.size += len(chunk)
        if self.size > self.max_bytes:
            raise AttachmentError("buffer_limit", 413)
        self.sha.update(chunk)

    def feed(self, chunk: bytes):
        self._account(chunk)
        self.pending += chunk
        while True:
            match = re.search(rb"\r?\n\r?\n", self.pending)
            if match is None:
                break
            frame, self.pending = self.pending[:match.start()], self.pending[match.end():]
            if not frame:
                continue
            if self.done:
                raise AttachmentError("native_completion_failed")
            lines = frame.replace(b"\r\n", b"\n").split(b"\n")
            if any(not line.startswith(b"data: ") for line in lines):
                raise AttachmentError("native_completion_failed")
            data = b"\n".join(line[6:] for line in lines)
            if data == b"[DONE]":
                self.done = True
            else:
                try:
                    self._object(strict_json(data))
                except AttachmentError:
                    raise AttachmentError("native_completion_failed") from None

    def finish(self, rid: str) -> NativeCompletion:
        pending = self.pending.strip(b" \t\r\n") if self.source and self.done else self.pending
        if pending or not self.done or not self.finish_reason or self.rid != rid:
            raise AttachmentError("native_completion_failed")
        return NativeCompletion(rid, self.finish_reason, self.sha.hexdigest(), self.size, True)

    def nonstream(self, body: bytes, rid: str) -> NativeCompletion:
        self._account(body)
        try:
            self._object(strict_json(body))
        except AttachmentError:
            raise AttachmentError("native_completion_failed") from None
        if self.rid != rid or not self.finish_reason:
            raise AttachmentError("native_completion_failed")
        return NativeCompletion(rid, self.finish_reason, self.sha.hexdigest(), self.size, False)


TERMINAL_DONE_FRAME = b"data: [DONE]\n\n"  # exactly 14 bytes
TERMINAL_DONE_DATA = b"data: [DONE]"
TERMINAL_EVENT_NAME = "optimus.occ3.receipt"
TERMINAL_EVENT_PREFIX = b"event: optimus.occ3.receipt\ndata: "
TERMINAL_SCHEMA = "occ3.terminal.v1"
TERMINAL_EVENT_LIMIT = 4096
TERMINAL_RECEIPT_LIMIT = 2048
TERMINAL_ID_LIMIT = 96


class TerminalOutputAudit(NativeOutputAudit):
    """Incremental terminal-SSE audit: hash before parse, never a whole-body buffer.

    Enforces the exact pinned grammar: original frames in original order, one
    successful finish for choice 0, usage-only frames allowed after finish,
    exactly one 14-byte ``data: [DONE]\n\n`` frame, zero bytes after it. CRLF
    framing, named/error frames, duplicate DONE and trailing bytes are
    refusals. The digest covers every original byte including DONE; only the
    server-emitted private receipt event is outside this hash domain.
    """

    def __init__(self, *, max_frame_bytes: int, max_native_bytes: int,
                 max_json_depth: int, max_json_nodes: int, source: bool = False):
        super().__init__(max_native_bytes, source=source)
        self.max_frame_bytes = max_frame_bytes
        self.max_json_depth = max_json_depth
        self.max_json_nodes = max_json_nodes
        self.done_frame = None

    def _account(self, chunk: bytes):
        self.size += len(chunk)
        if self.size > self.max_bytes:
            raise AttachmentError("buffer_limit", 413)
        self.sha.update(chunk)

    def _bounded_json(self, value):
        nodes = 0
        stack = [(value, 1)]
        while stack:
            node, level = stack.pop()
            nodes += 1
            if nodes > self.max_json_nodes or level > self.max_json_depth:
                raise AttachmentError("buffer_limit", 413)
            if isinstance(node, dict):
                stack.extend((item, level + 1) for item in node.values())
            elif isinstance(node, list):
                stack.extend((item, level + 1) for item in node)

    def feed(self, chunk: bytes) -> list:
        """Hash, validate and return the original frames now safe to forward.

        Each returned element is one validated complete original frame
        including its original ``\n\n`` delimiter. The DONE frame is validated
        and held, never returned; the caller emits it only after terminal
        backend proof.
        """
        self._account(chunk)
        self.pending += chunk
        forward = []
        while True:
            if b"\r\n\r\n" in self.pending:
                # CRLF framing is a refusal, not a tolerated variant.
                raise AttachmentError("native_completion_failed")
            match = re.search(rb"\n\n", self.pending)
            if match is None:
                break
            frame, self.pending = (self.pending[:match.start()],
                                   self.pending[match.end():])
            if len(self.pending) > self.max_frame_bytes:
                raise AttachmentError("buffer_limit", 413)
            if self.done:
                raise AttachmentError("native_completion_failed")
            if not frame:
                continue
            if frame == TERMINAL_DONE_DATA:
                if self.finish_reason is None:
                    raise AttachmentError("native_completion_failed")
                self.done = True
                self.done_frame = TERMINAL_DONE_FRAME
                continue
            lines = frame.split(b"\n")
            if any(not line.startswith(b"data: ") for line in lines):
                raise AttachmentError("native_completion_failed")
            data = b"\n".join(line[6:] for line in lines)
            try:
                value = strict_json(data)
                self._bounded_json(value)
                self._object(value)
            except AttachmentError:
                raise AttachmentError("native_completion_failed") from None
            forward.append(frame + b"\n\n")
        if len(self.pending) > self.max_frame_bytes:
            raise AttachmentError("buffer_limit", 413)
        return forward

    def finish(self, rid: str) -> NativeCompletion:
        if self.pending or self.done_frame is None or not self.finish_reason or self.rid != rid:
            raise AttachmentError("native_completion_failed")
        return NativeCompletion(rid, self.finish_reason, self.sha.hexdigest(), self.size, True)


def terminal_envelope(kind: str, native_rid: str, response_id: str, body_sha256: str,
                      native_body_bytes: int, receipt) -> dict:
    """Exact seven-field terminal envelope; inner receipt contracts stay unchanged."""
    if kind not in ("source", "receiver"):
        raise AttachmentError("invalid_evidence")
    bounded_identifier(native_rid, TERMINAL_ID_LIMIT)
    bounded_identifier(response_id, TERMINAL_ID_LIMIT)
    sha256_hex(body_sha256)
    if type(native_body_bytes) is not int or not 0 < uint(native_body_bytes):
        raise AttachmentError("invalid_evidence")
    if not isinstance(receipt, dict) or not receipt:
        raise AttachmentError("invalid_evidence")
    return {"schema": TERMINAL_SCHEMA, "kind": kind, "native_rid": native_rid,
            "response_id": response_id, "body_sha256": body_sha256,
            "native_body_bytes": native_body_bytes, "receipt": receipt}


def terminal_event_bytes(envelope: dict) -> bytes:
    """Encode the single private receipt event; whole frame <=4096, receipt <=2048."""
    if len(canonical(envelope["receipt"])) > TERMINAL_RECEIPT_LIMIT:
        raise AttachmentError("buffer_limit", 413)
    frame = TERMINAL_EVENT_PREFIX + canonical(envelope) + b"\n\n"
    if len(frame) > TERMINAL_EVENT_LIMIT:
        raise AttachmentError("buffer_limit", 413)
    return frame


def parse_terminal_event(frame: bytes) -> dict:
    """Strict decoder for tests and the future native consumer."""
    if not frame.startswith(TERMINAL_EVENT_PREFIX) or not frame.endswith(b"\n\n"):
        raise AttachmentError("invalid_evidence")
    envelope = strict_json(frame[len(TERMINAL_EVENT_PREFIX):-2])
    if (not isinstance(envelope, dict)
            or set(envelope) != {"schema", "kind", "native_rid", "response_id",
                                 "body_sha256", "native_body_bytes", "receipt"}
            or envelope["schema"] != TERMINAL_SCHEMA
            or envelope["kind"] not in ("source", "receiver")
            or not isinstance(envelope["receipt"], dict)):
        raise AttachmentError("invalid_evidence")
    terminal_envelope(envelope["kind"], envelope["native_rid"], envelope["response_id"],
                      envelope["body_sha256"], envelope["native_body_bytes"], envelope["receipt"])
    return envelope


@dataclass(frozen=True, repr=False)
class SourceToolCall:
    call_id: str
    name: str

    def __post_init__(self):
        identifier(self.call_id)
        identifier(self.name)


#: The v2 receipt span-block field names, in fixed order.
SPAN_BLOCK_FIELDS = ("span_schema", "span_policy", "span_marker_profile",
                     "span_marker_schema", "span_reasoning_terminator",
                     "span_turn_end_attested", "span_thinking_count",
                     "span_content_count", "span_tool_call_count",
                     "span_tool_arguments_count", "span_control_count",
                     "span_unknown_count", "span_window_start_position",
                     "span_window_end_position", "span_contiguous",
                     "span_contains_final_content", "span_contains_tool_call",
                     "span_message_truncated")


@dataclass(frozen=True, repr=False)
class SourceToolProvenance:
    tool_calls: tuple[SourceToolCall, ...]
    scope: str = SPAN_POLICY_V1

    def __post_init__(self):
        if (self.scope not in (SPAN_POLICY_V1,) + SPAN_POLICIES_V2
                or type(self.tool_calls) is not tuple or len(self.tool_calls) > 1
                or any(type(call) is not SourceToolCall for call in self.tool_calls)):
            raise AttachmentError("unsupported_profile")
        for call in self.tool_calls:
            SourceToolCall(call.call_id, call.name)

    @property
    def binding_digest(self):
        return digest(asdict(self))


@dataclass(frozen=True, repr=False)
class SourceCaptureCommit:
    """Actual worker commit metadata. No frontend byte digest is invented here.

    The 15 v1 fields are exactly the v1 receipt's non-identity fields.  The
    optional occ3.span.v1 block is all-present for a content_tail_v2 capture and
    all-None for a v1 capture; the deep cross-field invariants are enforced
    where the block becomes public, in SourceCaptureReceipt.
    """
    capture_commit_id: str
    accepted_span_start: int
    accepted_span_end: int
    retained_count: int
    predictor_position_offset: int
    representation: str
    layer_index: int
    hidden_dim: int
    dtype: str
    layout_digest: str
    ready_ranks: int
    expected_ranks: int
    accepted_sequence_digest: str
    expires_at_epoch: int
    byte_count: int
    span_schema: str | None = None
    span_policy: str | None = None
    span_marker_profile: str | None = None
    span_marker_schema: str | None = None
    span_reasoning_terminator: str | None = None
    span_turn_end_attested: bool | None = None
    span_thinking_count: int | None = None
    span_content_count: int | None = None
    span_tool_call_count: int | None = None
    span_tool_arguments_count: int | None = None
    span_control_count: int | None = None
    span_unknown_count: int | None = None
    span_window_start_position: int | None = None
    span_window_end_position: int | None = None
    span_contiguous: bool | None = None
    span_contains_final_content: bool | None = None
    span_contains_tool_call: bool | None = None
    span_message_truncated: bool | None = None

    def __post_init__(self):
        block = [getattr(self, name) for name in SPAN_BLOCK_FIELDS]
        present = [value is not None for value in block]
        if all(present):
            for name in ("span_schema", "span_policy", "span_marker_profile",
                         "span_marker_schema", "span_reasoning_terminator"):
                if type(getattr(self, name)) is not str:
                    raise AttachmentError("invalid_evidence")
            for name in ("span_turn_end_attested", "span_contiguous",
                         "span_contains_final_content", "span_contains_tool_call",
                         "span_message_truncated"):
                if type(getattr(self, name)) is not bool:
                    raise AttachmentError("invalid_evidence")
            for name in ("span_thinking_count", "span_content_count", "span_tool_call_count",
                         "span_tool_arguments_count", "span_control_count",
                         "span_unknown_count", "span_window_start_position",
                         "span_window_end_position"):
                if type(getattr(self, name)) is not int:
                    raise AttachmentError("invalid_evidence")
        elif any(present):
            raise AttachmentError("invalid_evidence")

    @property
    def is_v2(self) -> bool:
        return self.span_schema is not None

    def span_block(self) -> dict:
        """The bounded span block, or None for a v1 capture."""
        if not self.is_v2:
            return None
        return {name: getattr(self, name) for name in SPAN_BLOCK_FIELDS}


@dataclass(frozen=True, repr=False)
class SourceCaptureEvidence:
    """finish_source binds frontend output/provenance to genuine worker commit."""
    enrollment: SourceEnrollment
    reservation_id: str
    commit: SourceCaptureCommit
    bound_output_sha256: str
    response_id: str
    tool_provenance_digest: str
    completed: bool
    gpu_resident: bool
    num_retractions: int


def source_receipt_bytes(receipt: SourceCaptureReceipt, limit: int = MAX_RECEIPT_BYTES) -> bytes:
    if type(receipt) is not SourceCaptureReceipt:
        raise AttachmentError("invalid_evidence")
    validated = SourceCaptureReceipt.parse(receipt.wire())
    value = canonical(validated.wire())
    if len(value) > min(limit, MAX_RECEIPT_BYTES):
        raise AttachmentError("buffer_limit", 413)
    return value


def source_receipt_upper_bound(limit: int = MAX_RECEIPT_BYTES) -> bytes:
    """Valid maximum-width wire: reserve BEFORE source enrollment/generation."""
    receipt = SourceCaptureReceipt(
        intent_digest="f" * 64, source_handle="x" * 128, native_rid="x" * 96,
        response_id="x" * 96, capture_commit_id="x" * 96,
        accepted_span_start=2**64 - 1 - 128, accepted_span_end=2**64 - 1,
        retained_count=128, predictor_position_offset=-1,
        representation="final_layer_prenorm_predictor", layer_index=2**32 - 1,
        hidden_dim=2**32 - 1, dtype="bfloat16", layout_digest="f" * 64,
        ready_ranks=2**32 - 1, expected_ranks=2**32 - 1,
        accepted_sequence_digest="f" * 64, output_sha256="f" * 64,
        expires_at_epoch=2**64 - 1, byte_count=128 * (2**32 - 1) * 2)
    return source_receipt_bytes(receipt, limit)


def source_receipt_v2_upper_bound(limit: int = MAX_RECEIPT_BYTES) -> bytes:
    """Valid maximum-width occ3.source.v2 wire: the doc's unmeasured case.

    Same worst-case v1 fields as :func:`source_receipt_upper_bound`, plus the
    bounded span block at its widest admissible values.  The adapter's header
    admission uses this for a v2 capture so the pre-dispatch budget reserves
    what the receipt can actually cost.
    """
    receipt = SourceCaptureReceipt(
        intent_digest="f" * 64, source_handle="x" * 128, native_rid="x" * 96,
        response_id="x" * 96, capture_commit_id="x" * 96,
        accepted_span_start=2**64 - 1 - 128, accepted_span_end=2**64 - 1,
        retained_count=128, predictor_position_offset=-1,
        representation="final_layer_prenorm_predictor", layer_index=2**32 - 1,
        hidden_dim=2**32 - 1, dtype="bfloat16", layout_digest="f" * 64,
        ready_ranks=2**32 - 1, expected_ranks=2**32 - 1,
        accepted_sequence_digest="f" * 64, output_sha256="f" * 64,
        expires_at_epoch=2**64 - 1, byte_count=128 * (2**32 - 1) * 2,
        schema=SOURCE_SCHEMA_V2,
        span_schema=SPAN_SCHEMA_V1, span_policy=max(SPAN_POLICIES_V2),
        span_marker_profile="x" * 64, span_marker_schema=MARKER_SCHEMA_V1,
        span_reasoning_terminator="tool_call_start", span_turn_end_attested=True,
        span_thinking_count=128, span_content_count=128, span_tool_call_count=128,
        span_tool_arguments_count=128, span_control_count=128, span_unknown_count=128,
        span_window_start_position=0, span_window_end_position=2**64 - 1,
        span_contiguous=False, span_contains_final_content=True,
        span_contains_tool_call=True, span_message_truncated=True)
    return source_receipt_bytes(receipt, limit)


def validate_source_enrollment(intent, identity, attempt_id, native_rid, enrollment):
    if type(enrollment) is not SourceEnrollment:
        raise AttachmentError("invalid_evidence")
    SourceEnrollment(**asdict(enrollment))
    if (enrollment.attempt_id != attempt_id or enrollment.intent_digest != intent.binding_digest
            or enrollment.native_rid != native_rid or native_rid == intent.source_request_id
            or enrollment.principal_digest != identity.principal_digest
            or enrollment.source_session_id != intent.source_session_id
            or enrollment.runtime_id != intent.runtime_id
            or enrollment.runtime_generation != intent.runtime_generation
            or enrollment.model_fingerprint != intent.model_fingerprint
            or enrollment.method != intent.method
            or enrollment.expires_at_epoch > intent.expires_at_epoch):
        raise AttachmentError("invalid_evidence")


def validate_source_evidence(intent, enrollment, reservation, descriptor, completion, provenance, evidence):
    if (type(evidence) is not SourceCaptureEvidence
            or type(evidence.enrollment) is not SourceEnrollment
            or type(evidence.commit) is not SourceCaptureCommit
            or evidence.enrollment != enrollment
            or evidence.reservation_id != reservation.reservation_id
            or evidence.bound_output_sha256 != completion.output_sha256
            or evidence.response_id != completion.native_rid
            or evidence.tool_provenance_digest != provenance.binding_digest
            or evidence.completed is not True or evidence.gpu_resident is not True
            or type(evidence.num_retractions) is not int or evidence.num_retractions != 0):
        raise AttachmentError("invalid_evidence")
    SourceEnrollment(**asdict(evidence.enrollment))
    commit = evidence.commit
    if (reservation.attachment_digest != intent.binding_digest
            or reservation.native_rid != completion.native_rid or completion.native_rid != enrollment.native_rid
            or descriptor.native_rid != completion.native_rid
            or commit.retained_count > intent.retained_count_limit
            or commit.expected_ranks != enrollment.expected_ranks
            or commit.layer_index != enrollment.layer_index or commit.hidden_dim != enrollment.hidden_dim
            or commit.dtype != enrollment.dtype or commit.layout_digest != enrollment.layout_digest
            or commit.expires_at_epoch != enrollment.expires_at_epoch
            or commit.byte_count > intent.max_capture_bytes):
        raise AttachmentError("invalid_evidence")
    # Schema-gated receipt: a v2 intent requires the bounded span block and the
    # v2 source schema; a v1 intent refuses any block (doc section 6.2).
    block = {name: getattr(commit, name) for name in SPAN_BLOCK_FIELDS}
    if intent.span_policy in SPAN_POLICIES_V2:
        if (commit.span_schema is None or commit.span_policy != intent.span_policy
                or commit.span_marker_profile != descriptor.marker_profile):
            raise AttachmentError("invalid_evidence")
        schema = SOURCE_SCHEMA_V2
    else:
        if commit.span_schema is not None:
            raise AttachmentError("invalid_evidence")
        schema = SOURCE_SCHEMA_V1
    return SourceCaptureReceipt(
        intent_digest=intent.binding_digest, source_handle=enrollment.source_handle,
        native_rid=completion.native_rid, response_id=evidence.response_id,
        output_sha256=completion.output_sha256, schema=schema,
        **{name: value for name, value in asdict(commit).items() if name not in SPAN_BLOCK_FIELDS},
        **block)
