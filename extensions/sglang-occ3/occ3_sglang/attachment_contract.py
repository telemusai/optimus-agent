"""Strict, tensor-free OCC3 chat attachment contracts. No hooks run on import."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

RECEIPT_HEADER = "x-optimus-occ3-receipt"
MAX_RECEIPT_BYTES = 2048
#: Capture-intent schemas. v2 exists only to admit the content_tail_v2 span
#: policy family; every v1 wire keeps its exact meaning.
CAPTURE_SCHEMA_V1 = "occ3.capture.v1"
CAPTURE_SCHEMA_V2 = "occ3.capture.v2"
#: Source receipt schemas. v2 keeps the 21 v1 fields and adds the bounded
#: occ3.span.v1 block of SOURCE-SPAN-SELECTION.md section 6.2.
SOURCE_SCHEMA_V1 = "occ3.source.v1"
SOURCE_SCHEMA_V2 = "occ3.source.v2"
#: The only v1 span policy. It stays the default everywhere and is admitted
#: only under CAPTURE_SCHEMA_V1.
SPAN_POLICY_V1 = "whole_completed_response_tail_v1"
#: The v2 policy family (SOURCE-SPAN-SELECTION.md section 4). Admitted only
#: under CAPTURE_SCHEMA_V2; no last_accepted member exists.
SPAN_POLICIES_V2 = ("content_tail_no_tool_v2", "content_tail_v2",
                    "content_tail_last64_v2", "content_tail_last32_v2")
#: Version of the receipt span block and of the marker table shape.
SPAN_SCHEMA_V1 = "occ3.span.v1"
MARKER_SCHEMA_V1 = "occ3.span.markers.v1"
SPAN_TERMINATORS = ("thinking_end", "tool_call_start", "none")
_IDENTIFIER = re.compile(r"[A-Za-z0-9._:/-]{1,256}\Z")


class AttachmentError(ValueError):
    """Only fixed public error codes, never backend exceptions or identifiers."""

    def __init__(self, code: str = "attachment_failed", status: int = 409):
        allowed = {
            "invalid_attachment", "unsupported_attachment", "attachment_expired",
            "identity_mismatch", "unsupported_profile", "deadline_unavailable",
            "deadline_exceeded", "buffer_limit", "quota_exceeded", "replay_forbidden",
            "invalid_evidence", "native_completion_failed", "client_disconnected",
            "attachment_failed", "capture_transport_unnegotiated", "cleanup_pending",
        }
        self.code = code if code in allowed else "attachment_failed"
        self.status = status if status in (400, 401, 409, 413, 429, 499, 503, 504) else 409
        super().__init__(self.code)


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise AttachmentError("invalid_attachment", 400)
    return value


def uint(value: Any, maximum: int = 2**64 - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise AttachmentError("invalid_attachment", 400)
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True,
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class _Object(dict):
    def __init__(self, pairs):
        super().__init__(pairs)
        self.duplicate = len(pairs) != len(self)


def _bad_constant(_):
    raise AttachmentError("invalid_attachment", 400)


def strict_json(data: bytes | str) -> Any:
    try:
        obj = json.loads(data, object_pairs_hook=_Object, parse_constant=_bad_constant)
        def check(node):
            if isinstance(node, _Object):
                if node.duplicate:
                    raise AttachmentError("invalid_attachment", 400)
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)
        check(obj)
        return obj
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise AttachmentError("invalid_attachment", 400) from None


@dataclass(frozen=True, repr=False)
class InferenceAttachment:
    protocol_version: int
    request_id: str
    message_id: str
    connector_id: str
    receiver_session_id: str
    model_fingerprint: str
    runtime_id: str
    runtime_generation: int
    state_handle: str
    expires_at_epoch: int
    method: str

    def __post_init__(self):
        for name in ("request_id", "message_id", "connector_id", "receiver_session_id",
                     "model_fingerprint", "runtime_id", "state_handle", "method"):
            identifier(getattr(self, name))
        if uint(self.protocol_version, 2**32 - 1) != 1:
            raise AttachmentError("unsupported_attachment", 400)
        uint(self.runtime_generation)
        uint(self.expires_at_epoch)

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise AttachmentError("invalid_attachment", 400)
        return cls(**value)

    def wire(self) -> dict:
        return asdict(self)

    @property
    def binding_digest(self) -> str:
        return digest(self.wire())

    def __repr__(self):
        return "InferenceAttachment(<private>)"


def parse_opt_in(body: bytes) -> InferenceAttachment | CaptureIntent | None:
    """Ordinary JSON is not made stricter; strict parsing applies only to opt-in."""
    try:
        parsed = json.loads(body, object_pairs_hook=_Object)
    except (ValueError, TypeError, RecursionError, UnicodeError):
        return None  # The native ingress owns ordinary invalid JSON.
    if not isinstance(parsed, dict) or not ({"occ3", "occ3_capture"} & parsed.keys()):
        return None
    parsed = strict_json(body)
    if "occ3" in parsed and "occ3_capture" in parsed:
        raise AttachmentError("unsupported_profile", 409)
    if "occ3_capture" in parsed:
        return CaptureIntent.parse(parsed["occ3_capture"])
    return InferenceAttachment.parse(parsed["occ3"])


@dataclass(frozen=True, repr=False)
class CaptureIntent:
    """Negotiated owner-only capture on the actual source generating turn."""
    capture_id: str
    source_session_id: str
    source_request_id: str
    turn_nonce: str
    model_fingerprint: str
    runtime_id: str
    runtime_generation: int
    method: str
    retained_count_limit: int
    max_capture_bytes: int
    expires_at_epoch: int
    schema: str = "occ3.capture.v1"
    span_policy: str = "whole_completed_response_tail_v1"

    def __post_init__(self):
        for name in ("capture_id", "source_session_id", "source_request_id", "turn_nonce",
                     "model_fingerprint", "runtime_id", "method"):
            identifier(getattr(self, name))
        uint(self.runtime_generation)
        uint(self.expires_at_epoch)
        if (not 1 <= uint(self.retained_count_limit, 128) <= 128
                or uint(self.max_capture_bytes) == 0):
            raise AttachmentError("invalid_attachment", 400)
        # Schema-gated span policy: the v1 default is unchanged, and the v2
        # family is admitted only under the v2 schema (doc section 6.1).
        if self.schema == CAPTURE_SCHEMA_V1:
            admitted = self.span_policy == SPAN_POLICY_V1
        elif self.schema == CAPTURE_SCHEMA_V2:
            admitted = self.span_policy in SPAN_POLICIES_V2
        else:
            admitted = False
        if not admitted:
            raise AttachmentError("unsupported_profile", 409)

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise AttachmentError("invalid_attachment", 400)
        return cls(**value)

    def wire(self) -> dict:
        return asdict(self)

    @property
    def binding_digest(self) -> str:
        return digest(self.wire())


@dataclass(frozen=True, repr=False)
class AuthenticatedContext:
    """Issued by trusted routing/auth middleware, never populated from occ3 alone."""
    principal_digest: str
    session_id: str
    model_id: str
    model_fingerprint: str
    runtime_id: str
    runtime_generation: int
    deadline_monotonic: float

    def check(self, attachment: InferenceAttachment | CaptureIntent, requested_model: str):
        session_id = (attachment.source_session_id if type(attachment) is CaptureIntent
                      else attachment.receiver_session_id)
        if (self.session_id != session_id
                or self.model_id != requested_model
                or self.model_fingerprint != attachment.model_fingerprint
                or self.runtime_id != attachment.runtime_id
                or self.runtime_generation != attachment.runtime_generation
                or not re.fullmatch(r"[a-f0-9]{64}", self.principal_digest)):
            raise AttachmentError("identity_mismatch", 401)


@dataclass(frozen=True)
class BackendReadiness:
    frontend: bool = False
    tokenizer_descriptor: bool = False
    scheduler: bool = False
    all_workers: bool = False
    terminal_use: bool = False
    authenticated_owner: bool = False
    source_capture: bool = False

    @property
    def receiver_ready(self) -> bool:
        return all(value is True for value in (
            self.frontend, self.tokenizer_descriptor, self.scheduler,
            self.all_workers, self.terminal_use, self.authenticated_owner))


    @property
    def source_ready(self) -> bool:
        return self.receiver_ready and self.source_capture is True


@dataclass(frozen=True, repr=False)
class Reservation:
    attempt_id: str
    attachment_digest: str
    native_rid: str
    principal_digest: str
    required_seconds: float
    max_output_bytes: int
    max_new_tokens: int
    reservation_id: str


def bounded_identifier(value, maximum):
    identifier(value)
    if len(value) > maximum:
        raise AttachmentError("invalid_evidence")
    return value


def sha256_hex(value):
    if type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise AttachmentError("invalid_evidence")
    return value


DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
MAX_CAPTURE_BYTES = 128 * 1024 * 1024
SOURCE_RECEIPT_HEADER = "x-optimus-occ3-source"


@dataclass(frozen=True, repr=False)
class SourceEnrollment:
    """Backend owner-only reservation. No usable source lease before terminal proof."""
    attempt_id: str
    intent_digest: str
    native_rid: str
    principal_digest: str
    source_handle: str
    state_generation: int
    expires_at_epoch: int
    source_session_id: str
    runtime_id: str
    runtime_generation: int
    model_fingerprint: str
    method: str
    expected_ranks: int
    layout_digest: str
    layer_index: int
    hidden_dim: int
    dtype: str

    def __post_init__(self):
        for name in ("attempt_id", "source_session_id", "runtime_id", "model_fingerprint", "method"):
            identifier(getattr(self, name))
        bounded_identifier(self.native_rid, 96)
        bounded_identifier(self.source_handle, 128)
        for name in ("intent_digest", "principal_digest", "layout_digest"):
            sha256_hex(getattr(self, name))
        uint(self.state_generation)
        uint(self.runtime_generation)
        uint(self.expires_at_epoch)
        uint(self.layer_index, 2**32 - 1)
        if (uint(self.expected_ranks, 2**32 - 1) == 0
                or uint(self.hidden_dim, 2**32 - 1) == 0
                or type(self.dtype) is not str or self.dtype not in DTYPE_BYTES):
            raise AttachmentError("invalid_evidence")


#: The 21 v1 fields; a v1 wire carries exactly these and nothing else.
_SOURCE_RECEIPT_V1_FIELDS = (
    "intent_digest", "source_handle", "native_rid", "response_id", "capture_commit_id",
    "accepted_span_start", "accepted_span_end", "retained_count", "predictor_position_offset",
    "representation", "layer_index", "hidden_dim", "dtype", "layout_digest", "ready_ranks",
    "expected_ranks", "accepted_sequence_digest", "output_sha256", "expires_at_epoch",
    "byte_count", "schema",
)
#: The bounded occ3.span.v1 block of the v2 receipt (doc section 6.2). The
#: class counts cover the retained window; the aligned span stays
#: accepted_span_start/_end and retained_count.
_SOURCE_RECEIPT_SPAN_FIELDS = (
    "span_schema", "span_policy", "span_marker_profile", "span_marker_schema",
    "span_reasoning_terminator", "span_turn_end_attested",
    "span_thinking_count", "span_content_count", "span_tool_call_count",
    "span_tool_arguments_count", "span_control_count", "span_unknown_count",
    "span_window_start_position", "span_window_end_position", "span_contiguous",
    "span_contains_final_content", "span_contains_tool_call", "span_message_truncated",
)


@dataclass(frozen=True, repr=False)
class SourceCaptureReceipt:
    """Strict 21-field source wire plus the schema-gated v2 span block."""
    intent_digest: str
    source_handle: str
    native_rid: str
    response_id: str
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
    output_sha256: str
    expires_at_epoch: int
    byte_count: int
    schema: str = SOURCE_SCHEMA_V1
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
        for name in ("intent_digest", "layout_digest", "accepted_sequence_digest", "output_sha256"):
            sha256_hex(getattr(self, name))
        bounded_identifier(self.source_handle, 128)
        for name in ("native_rid", "response_id", "capture_commit_id"):
            bounded_identifier(getattr(self, name), 96)
        for name in ("accepted_span_start", "accepted_span_end", "expires_at_epoch", "byte_count"):
            uint(getattr(self, name))
        uint(self.layer_index, 2**32 - 1)
        if (type(self.predictor_position_offset) is not int or self.predictor_position_offset != -1
                or self.representation != "final_layer_prenorm_predictor"
                or not 1 <= uint(self.retained_count, 128) <= 128
                or self.accepted_span_end <= self.accepted_span_start
                or self.retained_count > self.accepted_span_end - self.accepted_span_start
                or uint(self.hidden_dim, 2**32 - 1) == 0
                or uint(self.ready_ranks, 2**32 - 1) == 0
                or uint(self.expected_ranks, 2**32 - 1) != self.ready_ranks
                or type(self.dtype) is not str or self.dtype not in DTYPE_BYTES
                or self.byte_count != self.retained_count * self.hidden_dim * DTYPE_BYTES[self.dtype]):
            raise AttachmentError("invalid_evidence")
        block = [getattr(self, name) for name in _SOURCE_RECEIPT_SPAN_FIELDS]
        if self.schema == SOURCE_SCHEMA_V1:
            # A v1 wire carries exactly the 21 fields; any span block is refused.
            if any(value is not None for value in block):
                raise AttachmentError("invalid_evidence")
        elif self.schema == SOURCE_SCHEMA_V2:
            if any(value is None for value in block):
                raise AttachmentError("invalid_evidence")
            self._check_span_block()
        else:
            raise AttachmentError("invalid_evidence")

    def _check_span_block(self):
        """Doc section 6.2 cross-field invariants, both ends identical."""
        counts = [getattr(self, name) for name in
                  ("span_thinking_count", "span_content_count", "span_tool_call_count",
                   "span_tool_arguments_count", "span_control_count", "span_unknown_count")]
        if (self.span_schema != SPAN_SCHEMA_V1
                or self.span_policy not in SPAN_POLICIES_V2
                or self.span_marker_schema != MARKER_SCHEMA_V1
                or self.span_reasoning_terminator not in SPAN_TERMINATORS):
            raise AttachmentError("invalid_evidence")
        bounded_identifier(self.span_marker_profile, 64)
        for name in ("span_turn_end_attested", "span_contiguous", "span_contains_final_content",
                     "span_contains_tool_call", "span_message_truncated"):
            if type(getattr(self, name)) is not bool:
                raise AttachmentError("invalid_evidence")
        for value in counts:
            if type(value) is not int or not 0 <= uint(value, 128) <= 128:
                raise AttachmentError("invalid_evidence")
        width = uint(self.span_window_end_position) - uint(self.span_window_start_position)
        # A contiguous window's rows exactly fill its width; a non-contiguous
        # one has at least one gap, so its rows are strictly fewer. Either way
        # the counts must be honest about which one they are.
        counts_fit = sum(counts) == width if self.span_contiguous else sum(counts) < width
        if (width <= 0 or not counts_fit or self.retained_count > sum(counts)):
            raise AttachmentError("invalid_evidence")
        if self.span_policy == "content_tail_no_tool_v2" and self.span_contains_tool_call:
            raise AttachmentError("invalid_evidence")
        if self.span_turn_end_attested is False and not self.span_message_truncated:
            raise AttachmentError("invalid_evidence")

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise AttachmentError("invalid_evidence")
        v1 = set(_SOURCE_RECEIPT_V1_FIELDS)
        if set(value) not in (v1, v1 | set(_SOURCE_RECEIPT_SPAN_FIELDS)):
            raise AttachmentError("invalid_evidence")
        return cls(**value)

    def wire(self):
        value = asdict(self)
        if self.schema == SOURCE_SCHEMA_V1:
            return {name: value[name] for name in _SOURCE_RECEIPT_V1_FIELDS}
        return value


@dataclass(frozen=True, repr=False)
class ConsumeWorkerDescriptor:
    """Internal worker IPC control only. Tensor resolution happens in each worker."""
    attempt_id: str
    reservation_id: str
    native_rid: str
    principal_digest: str
    attachment_digest: str
    receiver_session_id: str
    message_id: str
    consume_nonce: str
    opaque_handle: str
    state_generation: int
    runtime_generation: int
    source_capture_commit_id: str
    runtime_epoch: str
    model_fingerprint: str
    method_id: str
    tp_layout_digest: str
    expected_rank_count: int
    expected_prefix_tokens: int
    hidden_dim: int
    dtype: str
    deadline: float
    schema: str = "occ3.sglang.worker.v1"
    operation: str = "consume"
    #: Absolute row where the continuous prefix starts in the receiver
    #: sequence. 0 is the legacy prepend layout; a positive value comes from an
    #: attested prompt marker and must stay inside the ordinary prompt.
    prefix_offset: int = 0

    def __post_init__(self):
        import math
        for name in ("attempt_id", "reservation_id", "native_rid", "receiver_session_id",
                     "message_id", "consume_nonce", "opaque_handle", "source_capture_commit_id",
                     "runtime_epoch", "model_fingerprint", "method_id", "dtype"):
            identifier(getattr(self, name))
        for name in ("principal_digest", "attachment_digest", "tp_layout_digest"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[a-f0-9]{64}", getattr(self, name)):
                raise AttachmentError("invalid_evidence")
        uint(self.state_generation)
        uint(self.runtime_generation)
        uint(self.prefix_offset, 2**20)
        if (not 1 <= uint(self.expected_rank_count, 1024) <= 1024
                or not 1 <= uint(self.expected_prefix_tokens, 128) <= 128
                or not 1 <= uint(self.hidden_dim, 2**20) <= 2**20
                or type(self.deadline) not in (int, float) or not math.isfinite(self.deadline)
                or self.deadline <= 0 or self.schema != "occ3.sglang.worker.v1"
                or self.operation != "consume"):
            raise AttachmentError("invalid_evidence")


@dataclass(frozen=True, repr=False)
class CaptureWorkerDescriptor:
    """Future source enrollment control; it cannot supply source text or IDs."""
    attempt_id: str
    reservation_id: str
    native_rid: str
    principal_digest: str
    runtime_epoch: str
    model_fingerprint: str
    method_id: str
    deadline: float
    capture_id: str
    source_session_id: str
    source_turn_nonce: str
    runtime_generation: int
    span_policy: str
    max_prefix_tokens: int
    max_capture_bytes: int
    #: Resolved marker-profile identifier (doc section 6.1). Empty for the v1
    #: span policy; a bounded identifier for the v2 family. The marker ids
    #: themselves never travel here: they use the private custom_params channel.
    marker_profile: str = ""
    schema: str = "occ3.sglang.worker.v1"
    operation: str = "capture"

    def __post_init__(self):
        import math
        for name in ("attempt_id", "reservation_id", "native_rid", "runtime_epoch",
                     "model_fingerprint", "method_id", "capture_id", "source_session_id",
                     "source_turn_nonce"):
            identifier(getattr(self, name))
        uint(self.runtime_generation)
        if (not isinstance(self.principal_digest, str)
                or not re.fullmatch(r"[a-f0-9]{64}", self.principal_digest)
                or not 1 <= uint(self.max_prefix_tokens, 128) <= 128
                or not 1 <= uint(self.max_capture_bytes, MAX_CAPTURE_BYTES) <= MAX_CAPTURE_BYTES
                or type(self.deadline) not in (int, float) or not math.isfinite(self.deadline)
                or self.deadline <= 0 or self.schema != "occ3.sglang.worker.v1"
                or self.operation != "capture"
                or self.span_policy not in (SPAN_POLICY_V1,) + SPAN_POLICIES_V2):
            raise AttachmentError("invalid_evidence")
        if self.span_policy == SPAN_POLICY_V1:
            if self.marker_profile != "":
                raise AttachmentError("invalid_evidence")
        elif not 1 <= len(self.marker_profile) <= 64 or identifier(self.marker_profile) is None:
            raise AttachmentError("invalid_evidence")


TERMINAL_MODE = "terminal-sse-v1"
TERMINAL_MODE_FIELD = "occ3_receipt_mode"
TERMINAL_ACK_HEADER = "x-optimus-occ3-mode"


def parse_terminal_mode(body: bytes) -> str | None:
    """Strict sibling-field negotiation for the terminal receipt mode.

    Absent field: ordinary legacy behavior, returns None. A present field must
    be exactly the negotiated mode string; duplicate keys, wrong types and
    unknown values are refusals, never silent downgrades. Ordinary invalid
    JSON stays the native ingress owner's business.
    """
    try:
        parsed = json.loads(body, object_pairs_hook=_Object)
    except (ValueError, TypeError, RecursionError, UnicodeError):
        return None
    if not isinstance(parsed, dict) or TERMINAL_MODE_FIELD not in parsed:
        return None
    if parsed.duplicate:
        raise AttachmentError("invalid_attachment", 400)
    mode = parsed[TERMINAL_MODE_FIELD]
    if type(mode) is not str or mode != TERMINAL_MODE:
        raise AttachmentError("unsupported_profile", 409)
    return mode


def normalize_worker_descriptor(value):
    """Strict tensor-free IPC normalization, NOT authorization or handle resolution."""
    if value is None:
        return None
    types = (CaptureWorkerDescriptor, ConsumeWorkerDescriptor)
    if type(value) in types:
        data = {name: getattr(value, name) for name in value.__dataclass_fields__}
    elif type(value) is dict:
        data = value
    else:
        raise AttachmentError("invalid_evidence")
    if any(type(key) is not str or type(item) not in (str, int, float)
           for key, item in data.items()):
        raise AttachmentError("invalid_evidence")
    if data.get("schema") != "occ3.sglang.worker.v1":
        raise AttachmentError("invalid_evidence")
    kind = {"capture": CaptureWorkerDescriptor, "consume": ConsumeWorkerDescriptor}.get(data.get("operation"))
    if kind is None or set(data) != set(kind.__dataclass_fields__):
        raise AttachmentError("invalid_evidence")
    try:
        validated = kind(**data)
    except (TypeError, ValueError, OverflowError):
        raise AttachmentError("invalid_evidence") from None
    return value if type(value) is kind else validated
