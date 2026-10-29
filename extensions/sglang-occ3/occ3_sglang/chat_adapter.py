"""Opt-in native chat format hooks. No plugin registration or GPU backend here.

The original converter, native generator and chat renderer each run once.
Only a trusted backend can bind worker state or return completed-use evidence.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from starlette.responses import JSONResponse, Response, StreamingResponse

from .attachment_contract import (
    AttachmentError, AuthenticatedContext, BackendReadiness, CaptureIntent,
    CaptureWorkerDescriptor, ConsumeWorkerDescriptor, InferenceAttachment, MAX_RECEIPT_BYTES, RECEIPT_HEADER,
    SOURCE_RECEIPT_HEADER, SourceEnrollment, Reservation, SPAN_POLICIES_V2, TERMINAL_ACK_HEADER, TERMINAL_MODE,
    canonical, digest, identifier, normalize_worker_descriptor, parse_opt_in, parse_terminal_mode,
)
from .span_policy import resolve_runtime_markers, wire_marker_entries
from .prefix_layout import LAYOUT_PREPEND, PREFIX_MARKER, PrefixLayout, resolve_prefix_layout
from .use_receipts import (
    NativeCompletion, NativeOutputAudit, TerminalOutputAudit, TerminalUseEvidence,
    receipt_bytes, terminal_envelope, terminal_event_bytes,
    validate_evidence, SourceCaptureEvidence, SourceToolProvenance, source_receipt_bytes,
    source_receipt_upper_bound, source_receipt_v2_upper_bound, validate_source_enrollment,
    validate_source_evidence,
)

CHAT_TARGET = "sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat"
TOKENIZER_TARGET = "sglang.srt.managers.tokenizer_manager.TokenizerManager"
_CONTEXT = "_optimus_occ3_chat_context"
logger = logging.getLogger(__name__)

# Redaction set for adapter diagnostic lines. Inner failures are logged with
# these detail codes only; request bytes, identifiers and handles stay out.
_DIAGNOSTIC_CAUSE = "cause"


def _inner_error_detail(response, limit):
    """Bounded diagnostic from an inner error response, or "".

    The ordinary serving path already returns create_error_response(str(e))
    bodies to clients, so this message class is public-by-design; the native
    path refuses to forward it and records only this bounded server-side line.
    """
    try:
        body = getattr(response, "body", None)
        if not isinstance(body, (bytes, bytearray)) or not body:
            return ""
        text = body[:4096].decode("utf-8", "replace")
        try:
            value = json.loads(text)
            # Both documented ordinary-path bodies: the OpenAI-style
            # {"error":{"message":...}} wrapper and the pinned sglang
            # ErrorResponse flat {"message":...} model dump.
            message = value["error"]["message"] if isinstance(value.get("error"), dict) else value["message"]
        except Exception:
            message = text
        return message[:limit] if isinstance(message, str) else ""
    except Exception:
        return ""


# Fixed-shape refusal codes the scheduler-side occ3 abort paths emit. Only
# this bounded shape is ever logged; any other abort payload stays unlogged.
_REFUSAL_CODE = re.compile(r"occ3_[a-z][a-z_]{0,62}\Z")


def _refusal_code(reason) -> str:
    if not isinstance(reason, dict):
        return "unclassified"
    message = reason.get("message")
    if isinstance(message, str) and _REFUSAL_CODE.fullmatch(message):
        return message
    return "unclassified"


def _refusal_status(reason) -> str:
    if not isinstance(reason, dict):
        return ""
    status = reason.get("status_code")
    return str(status) if type(status) is int and 400 <= status <= 599 else ""


# These guard outcomes occur before the buffered inference response is sent.
# A disconnected client may never receive the JSON error; retain a bounded log.
# Other AttachmentError outcomes stay unlogged here.
_CUT_TURN = frozenset({("client_disconnected", 499), ("deadline_exceeded", 504)})


class NativeGenerateRequest(Protocol):
    rid: str | None
    input_ids: list[int] | None
    sampling_params: dict
    stream: bool


@dataclass(frozen=True, repr=False)
class NativeRequestView:
    request: NativeGenerateRequest
    max_new_tokens: int
    sampling_digest: str
    streaming: bool
    deadline_monotonic: float
    #: Absolute row of the reserved continuous prefix in the receiver sequence.
    prefix_offset: int = 0


class ChatStateBackend(Protocol):
    """Worker descriptor transport and CAS/journal are backend responsibilities.

    reserve must atomically fence the handle across all frontend processes.
    Cancellation may occur before reserve returns: retire must find attempt_id.
    bind installs only the typed descriptor, never replaces prompt or sampling.
    finish must await same-rid all-rank injection and terminal worker commit.
    """
    def readiness(self) -> BackendReadiness: ...
    async def reserve(self, attachment: InferenceAttachment, identity: AuthenticatedContext,
                      attempt_id: str, native: NativeRequestView) -> Reservation: ...
    def bind(self, reservation: Reservation, native: NativeGenerateRequest) -> None: ...
    async def finish(self, reservation: Reservation,
                     completion: NativeCompletion) -> TerminalUseEvidence: ...
    async def retire(self, attempt_id: str, native_rid: str | None,
                     dispatched: bool) -> None: ...


class SourceCaptureBackend(ChatStateBackend, Protocol):
    """Owner enrollment before conversion; full constraint checks after conversion.

    required_seconds only rejects a known deadline conflict. It is not a proof
    that a MAX turn will finish. Backend admission must separately check actual
    context, output-byte and aggregate memory limits without changing sampling.
    """
    async def enroll_source(self, intent: CaptureIntent, identity: AuthenticatedContext,
                            attempt_id: str, native_rid: str, deadline: float) -> SourceEnrollment: ...
    async def prepare_source(self, enrollment: SourceEnrollment,
                             native: NativeRequestView) -> Reservation: ...
    def bind_source(self, enrollment: SourceEnrollment, reservation: Reservation,
                    native: NativeGenerateRequest) -> None: ...
    async def finish_source(self, enrollment: SourceEnrollment, reservation: Reservation,
                            completion: NativeCompletion,
                            tool_provenance: SourceToolProvenance) -> SourceCaptureEvidence: ...


class UnsupportedBackend:
    def readiness(self):
        return BackendReadiness()

    async def reserve(self, attachment, identity, attempt_id, native):
        raise AttachmentError("unsupported_attachment", 503)

    def bind(self, reservation, native):
        raise AttachmentError("unsupported_attachment", 503)

    async def finish(self, reservation, completion):
        raise AttachmentError("unsupported_attachment", 503)

    async def enroll_source(self, intent, identity, attempt_id, native_rid, deadline):
        raise AttachmentError("unsupported_attachment", 503)

    async def prepare_source(self, enrollment, native):
        raise AttachmentError("unsupported_attachment", 503)

    def bind_source(self, enrollment, reservation, native):
        raise AttachmentError("unsupported_attachment", 503)

    async def finish_source(self, enrollment, reservation, completion, tool_provenance):
        raise AttachmentError("unsupported_attachment", 503)

    async def retire(self, attempt_id, native_rid, dispatched):
        return None


async def unavailable_identity(raw_request):
    raise AttachmentError("identity_mismatch", 401)


@dataclass(frozen=True)
class TerminalProfile:
    """Explicit trusted process-local limits for the terminal receipt mode.

    Default is None: no runtime/Env/HTTP flag can enable the mode. A profile
    object is injected only by the trusted installer/harness constructor. Its
    numbers are admission bounds for finite synthetic validation, never a
    full-MAX capacity proof, and never a lowering of the native request budget.
    """

    max_frame_bytes: int
    max_native_bytes: int
    max_json_depth: int
    max_json_nodes: int
    max_terminal_active: int
    max_attempts: int = 256

    def __post_init__(self):
        integer_fields = (self.max_frame_bytes, self.max_native_bytes,
                          self.max_json_depth, self.max_json_nodes,
                          self.max_terminal_active, self.max_attempts)
        if any(type(x) is not int or x <= 0 for x in integer_fields):
            raise ValueError("invalid terminal profile")
        if self.max_frame_bytes > self.max_native_bytes:
            raise ValueError("invalid terminal profile")
        # Coherent admission accounting: every concurrently active terminal
        # request holds a distinct attempt registration, so the attempts cap
        # must dominate the active cap. An incoherent profile fails closed.
        if self.max_attempts < self.max_terminal_active:
            raise ValueError("invalid terminal profile")


@dataclass(frozen=True)
class BufferProfile:
    max_buffer_bytes: int = 4 * 1024 * 1024
    max_active: int = 2
    total_working_bytes: int = 32 * 1024 * 1024
    max_attempts: int = 1024
    max_ttl_seconds: float = 3600
    max_turn_seconds: float = 300
    cleanup_seconds: float = 1
    disconnect_interval: float = 0.05
    deadline_margin_seconds: float = 1
    max_receipt_bytes: int = MAX_RECEIPT_BYTES
    aggregate_header_bytes: int = 3072
    ordinary_header_reserve: int = 768

    def __post_init__(self):
        integer_fields = (self.max_buffer_bytes, self.max_active, self.total_working_bytes,
                          self.max_attempts, self.max_receipt_bytes,
                          self.aggregate_header_bytes, self.ordinary_header_reserve)
        if any(type(x) is not int or x <= 0 for x in integer_fields):
            raise ValueError("invalid buffer profile")
        if (self.max_receipt_bytes > MAX_RECEIPT_BYTES
                or self.aggregate_header_bytes >= 4096
                or self.max_receipt_bytes + self.ordinary_header_reserve + 64
                > self.aggregate_header_bytes
                or self.total_working_bytes < 4 * self.max_buffer_bytes):
            raise ValueError("invalid buffer profile")
        if any(not math.isfinite(x) or x <= 0 for x in (
            self.max_ttl_seconds, self.max_turn_seconds, self.cleanup_seconds,
            self.disconnect_interval, self.deadline_margin_seconds,
        )):
            raise ValueError("invalid buffer profile")


class _TerminalLedger:
    """Per-context registry of required terminal operations (ALL-work barrier).

    Every owned operation of one terminal context — native next, backend
    proof, abort, retire, iterator close, ASGI emission — registers a token
    BEFORE it starts and resolves exactly once. Charges may be released only
    when EVERY registered operation is positive: a pending, cancelled, failed
    or timed-out operation latches fail-closed retention until it actually
    completes positively (a late completion resolves exactly once); a
    cancelled operation stays unknown forever. Retire-positive is ONE
    required acknowledgment, never permission to release while any other
    required operation is pending or unknown.
    """

    __slots__ = ("ops", "_next")

    def __init__(self):
        self.ops = {}   # token -> [kind, state, task-ref]
        self._next = 0

    def register(self, kind):
        self._next += 1
        self.ops[self._next] = [kind, "pending", None]
        return self._next

    def attach(self, token, task):
        entry = self.ops.get(token)
        if entry is not None:
            entry[2] = task

    def kind(self, token):
        entry = self.ops.get(token)
        return entry[0] if entry is not None else None

    def state(self, token):
        entry = self.ops.get(token)
        return entry[1] if entry is not None else "latched"

    def resolve(self, token, positive):
        entry = self.ops.get(token)
        if entry is None or entry[1] != "pending":
            return  # exactly-once outcome state
        entry[1] = "positive" if positive else "latched"
        entry[2] = None  # completed tasks may drop the object; the state stays

    def resolve_task(self, token, task):
        if task.cancelled():
            self.resolve(token, False)
            return
        try:
            exception = task.exception()
        except BaseException:
            self.resolve(token, False)
            return
        # A known normal StopAsyncIteration is an explicit positive native
        # EOF outcome, not a failure (root repair F2).
        if isinstance(exception, StopAsyncIteration):
            self.resolve(token, True)
            return
        self.resolve(token, exception is None)

    def all_positive(self):
        return all(entry[1] == "positive" for entry in self.ops.values())


@dataclass(repr=False)
class _Context:
    attachment: InferenceAttachment | CaptureIntent
    identity: AuthenticatedContext | None
    raw_request: Any
    attempt_id: str
    deadline: float
    receipt: bytes
    native: Any = None
    tm: Any = None
    reservation: Reservation | None = None
    descriptor: ConsumeWorkerDescriptor | CaptureWorkerDescriptor | None = None
    enrollment: SourceEnrollment | None = None
    inner: Any = None
    outer: Any = None
    response_iterator: Any = None
    generation_calls: int = 0
    dispatched: bool = False
    attempted: bool = False
    exhausted: bool = False
    native_terminal: bool = False
    completed: bool = False
    cancel_requested: bool = False
    owner_finished: bool = False
    cleanup_started: bool = False
    retired: bool = False
    charged: bool = False
    native_rid: str | None = None
    terminal_mode: str | None = None
    terminal_handed_off: bool = False
    # Terminal lifetime bookkeeping: explicit per-context acquisition flags
    # (exactly-once release; another context's counts can never be touched)
    # and tracked-settle state. retire_started/retire_confirmed are separate:
    # an unknown cleanup is NOT completion.
    terminal_active_charged: bool = False
    terminal_attempt_charged: bool = False
    terminal_released: bool = False
    terminal_settled: bool = False
    retire_started: bool = False
    retire_confirmed: bool = False
    terminal_settle_task: Any = None
    terminal_retire_task: Any = None
    terminal_watchdog_task: Any = None
    terminal_failure_latch: bool = False
    terminal_response_task: Any = None
    terminal_ledger: Any = None
    # Receiver continuous-prefix placement resolved from the rendered prompt.
    prefix_offset: int = 0
    prefix_layout: str = "prepend"


class ChatAttachmentAdapter:
    def __init__(self, backend: ChatStateBackend | None = None, *,
                 authorize: Callable[[Any], Awaitable[AuthenticatedContext]] = unavailable_identity,
                 profile: BufferProfile | None = None,
                 terminal_profile: TerminalProfile | None = None,
                 monotonic: Callable[[], float] = time.monotonic,
                 epoch: Callable[[], float] = time.time):
        self.terminal_profile = terminal_profile  # None: the mode is refused before dispatch.
        self._terminal_active = 0
        self._terminal_attempts: set[str] = set()
        self.backend = backend if backend is not None else UnsupportedBackend()
        self.authorize = authorize
        self.profile = profile or BufferProfile()
        self.monotonic, self.epoch = monotonic, epoch
        self._active = 0
        self._attempts: dict[str, float] = {}
        self._quarantined: set[asyncio.Task] = set()
        # Terminal-only tracked tasks: unresolved abort/close operations and
        # quarantined retires that retain charged ownership until they settle.
        self._terminal_tasks: set[asyncio.Task] = set()
        self._terminal_quarantined: set[asyncio.Task] = set()

    def hook_specs(self):
        """Return exact pinned-source targets; caller must register/apply explicitly."""
        return (
            (CHAT_TARGET + ".handle_request", self.handle_around, "AROUND"),
            (CHAT_TARGET + "._convert_to_internal_request", self.convert_after, "AFTER"),
            (TOKENIZER_TARGET + ".generate_request", self.generate_around, "AROUND"),
        )

    def register_hooks(self, registry, hook_type):
        """Registration only. Does not apply hooks, load a model or advertise readiness."""
        for target, hook, kind in self.hook_specs():
            registry.register(target, hook, getattr(hook_type, kind))

    def _has_hook(self, method, name):
        function = getattr(method, "__func__", method)
        seen = set()
        while callable(function) and id(function) not in seen:
            seen.add(id(function))
            try:
                hook = inspect.getclosurevars(function).nonlocals.get("hook")
            except (TypeError, ValueError):
                hook = None
            if (getattr(hook, "__self__", None) is self
                    and getattr(hook, "__name__", None) == name):
                return True
            function = getattr(function, "__wrapped__", None)
        return False

    def _charge(self, ctx):
        profile = self.profile
        if (self._active >= profile.max_active
                or (self._active + 1) * profile.max_buffer_bytes * 4 > profile.total_working_bytes):
            raise AttachmentError("quota_exceeded", 429)
        self._active += 1
        ctx.charged = True

    def _release(self, ctx):
        if ctx.charged:
            self._active -= 1
            ctx.charged = False

    @staticmethod
    def _error(error):
        return JSONResponse({"error": {"message": error.code,
                            "type": "occ3_attachment_error", "code": error.status}},
                            status_code=error.status, headers={"cache-control": "no-store"})

    async def handle_around(self, original, serving, request, raw_request):
        try:
            body = await raw_request.body()
            attachment = parse_opt_in(body)
            terminal_mode = parse_terminal_mode(body)
        except AttachmentError as error:
            return self._error(error)
        if attachment is None:
            if terminal_mode is not None:
                # Mode without an attachment/capture is a refusal, never a silent downgrade.
                return self._error(AttachmentError("unsupported_profile", 409))
            return await original(serving, request, raw_request)
        if terminal_mode is not None:
            try:
                admitted = self._terminal_admission(request, raw_request)
            except AttachmentError as error:
                return self._error(error)
            if not admitted:
                return self._error(AttachmentError("unsupported_profile", 409))
        ctx = None
        try:
            source = type(attachment) is CaptureIntent
            readiness = self.backend.readiness()
            if (type(readiness) is not BackendReadiness or not (
                    readiness.source_ready if source else readiness.receiver_ready)) and not callable(
                        getattr(self.backend, "refresh_readiness", None)):
                raise AttachmentError("unsupported_attachment", 503)
            if source:
                encodings = raw_request.headers.getlist("accept-encoding")
                if len(encodings) != 1 or encodings[0].strip().lower() != "identity":
                    raise AttachmentError("unsupported_profile")
                if getattr(request, "rid", None) is not None:
                    raise AttachmentError("unsupported_profile")
            if (not self._has_hook(serving._convert_to_internal_request, "convert_after")
                    or not self._has_hook(serving.tokenizer_manager.generate_request, "generate_around")):
                raise AttachmentError("unsupported_attachment", 503)
            if getattr(serving.tokenizer_manager.request_logger, "log_requests", False):
                raise AttachmentError("unsupported_profile", 409)
            if type(getattr(request, "n", None)) is not int or request.n != 1:
                raise AttachmentError("unsupported_profile", 409)
            if (getattr(request, "input_ids", None) is not None
                    or getattr(request, "return_meta_info", False)
                    or getattr(request, "return_hidden_states", False)):
                raise AttachmentError("unsupported_profile", 409)
            if getattr(raw_request.state, _CONTEXT, None) is not None:
                raise AttachmentError("replay_forbidden")
            now = self.epoch()
            if attachment.expires_at_epoch <= now:
                raise AttachmentError("attachment_expired")
            if attachment.expires_at_epoch - now > self.profile.max_ttl_seconds:
                raise AttachmentError("unsupported_profile")
            if source and attachment.span_policy in SPAN_POLICIES_V2:
                # The v2 source receipt carries the bounded span block, so the
                # pre-dispatch header budget reserves its measured upper bound.
                receipt = source_receipt_v2_upper_bound(self.profile.max_receipt_bytes)
            elif source:
                receipt = source_receipt_upper_bound(self.profile.max_receipt_bytes)
            else:
                receipt = receipt_bytes(attachment, self.profile.max_receipt_bytes)
            header = SOURCE_RECEIPT_HEADER if source else RECEIPT_HEADER
            if (len(receipt) + len(header) + 4 + self.profile.ordinary_header_reserve
                    > self.profile.aggregate_header_bytes):
                raise AttachmentError("buffer_limit", 413)
            clock = self.monotonic()
            deadline = min(clock + self.profile.max_turn_seconds,
                           clock + attachment.expires_at_epoch - self.epoch())
            ctx = _Context(attachment, None, raw_request, uuid.uuid4().hex, deadline, receipt)
            ctx.tm = serving.tokenizer_manager
            ctx.terminal_mode = terminal_mode
            if terminal_mode is not None:
                profile = self.terminal_profile
                if len(self._terminal_attempts) >= profile.max_attempts:
                    raise AttachmentError("quota_exceeded", 429)
                self._terminal_attempts.add(ctx.attempt_id)
                ctx.terminal_attempt_charged = True
                self._terminal_active += 1
                ctx.terminal_active_charged = True
            self._charge(ctx)
            setattr(raw_request.state, _CONTEXT, ctx)
            return await self._guard(
                self._authorize_and_run(original, serving, request, raw_request, ctx), ctx)
        except AttachmentError as error:
            if ctx is not None and (error.code, error.status) in _CUT_TURN:
                # Bounded cut-turn record: fixed code, status and whether native
                # work was already dispatched. No handle, body or prompt text.
                logger.error("occ3-bridge: native turn ended before any response "
                             "(reason=%s status=%s dispatched=%s)",
                             error.code, error.status, ctx.dispatched)
            return self._error(error)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Same bounded diagnostic contract as the native_completion_failed
            # seam: exception class name only, never payloads, identifiers or
            # handles. The public response stays a fixed generic code.
            logger.error("occ3-bridge: native turn failed (inner=%s)",
                         type(error).__name__)
            return self._error(AttachmentError())
        finally:
            if ctx is not None:
                try:
                    # After terminal hand-off the owner (streaming iterator,
                    # ASGI response lifecycle, watchdog) owns cleanup; running
                    # it here would abort a live stream that is still
                    # forwarding original bytes. Before hand-off, a terminal
                    # context settles through the same tracked owner path so
                    # charges are retained until the backend retire is
                    # positively confirmed; the legacy buffered path is
                    # unchanged.
                    if not ctx.completed and not ctx.terminal_handed_off:
                        if ctx.terminal_mode is None:
                            await self._bounded_cleanup(self._cleanup(ctx), ctx)
                        else:
                            await self._terminal_settle(ctx)
                finally:
                    ctx.owner_finished = True
                    if ctx.terminal_mode is None:
                        if not ctx.terminal_handed_off and not any(
                                getattr(task, "_occ3_context", None) is ctx for task in self._quarantined):
                            self._release(ctx)
                    # Terminal charges are released ONLY by the tracked settle:
                    # known success, or a positively confirmed retire (immediate
                    # or quarantined-late). Unknown cleanup retains both charge
                    # domains and must refuse new work at the limits.
                    if getattr(raw_request.state, _CONTEXT, None) is ctx and not ctx.terminal_handed_off:
                        delattr(raw_request.state, _CONTEXT)

    async def _authorize_and_run(self, original, serving, request, raw_request, ctx):
        identity = await self.authorize(raw_request)
        if ctx.cancel_requested:
            raise AttachmentError("client_disconnected", 499)
        if type(identity) is not AuthenticatedContext:
            raise AttachmentError("identity_mismatch", 401)
        identity.check(ctx.attachment, request.model)
        clock = self.monotonic()
        if (type(identity.deadline_monotonic) not in (int, float)
                or not math.isfinite(identity.deadline_monotonic) or identity.deadline_monotonic <= clock):
            raise AttachmentError("deadline_unavailable", 409)
        ctx.identity = identity
        ctx.deadline = min(ctx.deadline, identity.deadline_monotonic)
        source = type(ctx.attachment) is CaptureIntent
        readiness = self.backend.readiness()
        if type(readiness) is not BackendReadiness or not (
                readiness.source_ready if source else readiness.receiver_ready):
            # Refresh expired health only after authenticating this request.
            await self.backend.refresh_readiness()
            if ctx.cancel_requested:
                raise AttachmentError("client_disconnected", 499)
            readiness = self.backend.readiness()
            if type(readiness) is not BackendReadiness or not (
                    readiness.source_ready if source else readiness.receiver_ready):
                raise AttachmentError("unsupported_attachment", 503)
        if source:
            ctx.native_rid = uuid.uuid4().hex
            self._fence(ctx)
            enrollment = await self.backend.enroll_source(
                ctx.attachment, identity, ctx.attempt_id, ctx.native_rid, ctx.deadline)
            if ctx.cancel_requested:
                raise AttachmentError("client_disconnected", 499)
            validate_source_enrollment(ctx.attachment, identity, ctx.attempt_id, ctx.native_rid, enrollment)
            ctx.enrollment = enrollment
            remaining = enrollment.expires_at_epoch - self.epoch()
            if remaining <= 0:
                raise AttachmentError("attachment_expired")
            ctx.deadline = min(ctx.deadline, self.monotonic() + remaining)
        return await self._run(original, serving, request, raw_request, ctx)

    def convert_after(self, result, serving, request, raw_request=None):
        ctx = getattr(getattr(raw_request, "state", None), _CONTEXT, None)
        if ctx is None:
            return result
        if ctx.native is not None or not isinstance(result, tuple) or len(result) != 2:
            raise AttachmentError("unsupported_profile")
        native, processed = result
        if processed is not request:
            raise AttachmentError("unsupported_profile")
        conflicts = ("cache_salt", "extra_key", "session_id", "lora_path", "routed_dp_rank",
                     "disagg_prefill_dp_rank", "bootstrap_host", "input_embeds",
                     "positional_embed_overrides", "image_data", "video_data", "audio_data")
        if any(getattr(native, key, None) is not None for key in conflicts):
            raise AttachmentError("unsupported_profile")
        if type(ctx.attachment) is CaptureIntent:
            if ctx.enrollment is None or native.rid is not None:
                raise AttachmentError("unsupported_profile")
            native.rid = ctx.native_rid
        ctx.native = native
        return result

    def generate_around(self, original, tm, obj, request=None):
        ctx = getattr(getattr(request, "state", None), _CONTEXT, None)
        if ctx is None:
            return original(tm, obj, request)
        ctx.outer = self._generate(original, tm, obj, request, ctx)
        return ctx.outer

    def _prefix_layout(self, obj, tm):
        """Resolve the continuous-prefix row for this receiver request.

        A marker in the rendered prompt selects a mid-sequence layout and the
        prompt is pinned to the exact token ids used to compute the offset.
        Without a marker the legacy prepend layout is kept unchanged.
        """
        text = getattr(obj, "text", None)
        if not isinstance(text, str) or PREFIX_MARKER not in text:
            # No marker: the legacy prepend layout is kept byte-identical and
            # no tokenizer call is made on the ordinary path.
            return PrefixLayout(LAYOUT_PREPEND, 0), None
        tokenizer = getattr(tm, "tokenizer", None)
        if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
            raise AttachmentError("unsupported_profile")
        try:
            layout = resolve_prefix_layout(text, tokenizer)
            pinned = list(tokenizer.encode(text, add_special_tokens=False))
        except AttachmentError:
            raise
        except Exception:
            raise AttachmentError("unsupported_profile") from None
        if not 0 < layout.offset < len(pinned):
            raise AttachmentError("unsupported_profile")
        return layout, pinned

    def _fence(self, ctx):
        now = self.epoch()
        self._attempts = {key: expiry for key, expiry in self._attempts.items() if expiry > now}
        a = ctx.attachment
        if type(a) is CaptureIntent:
            identity = ["capture", ctx.identity.principal_digest, a.source_session_id, a.capture_id]
        else:
            identity = ["consume", a.state_handle]
        fence = digest([a.runtime_id, a.runtime_generation, *identity])
        if fence in self._attempts:
            raise AttachmentError("replay_forbidden")
        if len(self._attempts) >= self.profile.max_attempts:
            raise AttachmentError("quota_exceeded", 429)
        self._attempts[fence] = a.expires_at_epoch
        ctx.attempted = True  # Before any backend await, including uncertain admission.

    async def _generate(self, original, tm, obj, request, ctx):
        try:
            if ctx.native is not obj or ctx.tm is not tm or ctx.generation_calls:
                raise AttachmentError("replay_forbidden")
            ctx.generation_calls += 1
            sampling = copy.deepcopy(obj.sampling_params)
            if not isinstance(sampling, dict):
                raise AttachmentError("unsupported_profile")
            budget = sampling.get("max_new_tokens")
            if type(budget) is not int or budget <= 0 or sampling.get("n", 1) != 1:
                raise AttachmentError("unsupported_profile")
            if obj.rid is not None and (not isinstance(obj.rid, str) or not obj.rid):
                raise AttachmentError("unsupported_profile")
            if obj.rid is None:
                obj.rid = uuid.uuid4().hex
            identifier(obj.rid)
            ctx.native_rid = obj.rid
            source = type(ctx.attachment) is CaptureIntent
            if not source:
                self._fence(ctx)
            elif ctx.enrollment is None or ctx.native_rid != ctx.enrollment.native_rid:
                raise AttachmentError("invalid_evidence")
            if getattr(obj, "occ3_worker_descriptor", None) is not None:
                raise AttachmentError("unsupported_profile")
            if not source:
                # Reserve the receiver's continuous-prefix rows at the attested
                # prompt marker. The ordinary prompt is never rewritten: the
                # token ids are pinned here so the scheduler-side reservation
                # and this offset cannot diverge.
                layout, pinned = self._prefix_layout(obj, tm)
                if pinned is not None:
                    obj.text, obj.input_ids = None, pinned
                object.__setattr__(ctx, "prefix_offset", layout.offset)
                object.__setattr__(ctx, "prefix_layout", layout.layout)
            elif ctx.attachment.span_policy in SPAN_POLICIES_V2:
                # v2 opt-in: the attested marker ids travel on the request's
                # private custom_params channel, resolved once per tokenizer.
                # The adapter's entries overwrite any client-supplied occ3 key,
                # so the channel cannot be spoofed from the request body.
                custom = obj.sampling_params.get("custom_params")
                if custom is not None and not isinstance(custom, dict):
                    raise AttachmentError("unsupported_profile")
                try:
                    entries = wire_marker_entries(*resolve_runtime_markers(tm.tokenizer))
                except Exception:
                    raise AttachmentError("unsupported_profile", 409) from None
                obj.sampling_params["custom_params"] = {**(custom or {}), **entries}
            protected = copy.deepcopy({k: v for k, v in vars(obj).items()
                                       if k != "occ3_worker_descriptor"})
            view = NativeRequestView(obj, budget, digest(sampling), bool(obj.stream),
                                     ctx.deadline, prefix_offset=ctx.prefix_offset)
            reservation = (await self.backend.prepare_source(ctx.enrollment, view) if source else
                           await self.backend.reserve(ctx.attachment, ctx.identity, ctx.attempt_id, view))
            if ctx.cancel_requested:
                raise AttachmentError("client_disconnected", 499)
            if (type(reservation) is not Reservation
                    or reservation.attempt_id != ctx.attempt_id
                    or reservation.attachment_digest != ctx.attachment.binding_digest
                    or reservation.principal_digest != ctx.identity.principal_digest
                    or reservation.native_rid != ctx.native_rid
                    or reservation.max_new_tokens != budget
                    or type(reservation.required_seconds) not in (int, float)
                    or not math.isfinite(reservation.required_seconds)
                    or reservation.required_seconds <= 0
                    or type(reservation.max_output_bytes) is not int
                    or not 0 < reservation.max_output_bytes <= self.profile.max_buffer_bytes):
                raise AttachmentError("invalid_evidence")
            ctx.reservation = reservation
            if (reservation.required_seconds + self.profile.deadline_margin_seconds
                    > ctx.deadline - self.monotonic()):
                raise AttachmentError("deadline_unavailable")
            if {k: v for k, v in vars(obj).items() if k != "occ3_worker_descriptor"} != protected:
                raise AttachmentError("unsupported_profile")
            result = (self.backend.bind_source(ctx.enrollment, reservation, obj) if source else
                      self.backend.bind(reservation, obj))
            if result is not None:
                if inspect.iscoroutine(result):
                    result.close()
                raise AttachmentError("unsupported_profile")
            descriptor = normalize_worker_descriptor(getattr(obj, "occ3_worker_descriptor", None))
            if (descriptor is None
                    or descriptor.attempt_id != ctx.attempt_id
                    or descriptor.reservation_id != reservation.reservation_id
                    or descriptor.native_rid != ctx.native_rid
                    or descriptor.principal_digest != ctx.identity.principal_digest
                    or descriptor.runtime_epoch != ctx.attachment.runtime_id
                    or descriptor.runtime_generation != ctx.attachment.runtime_generation
                    or descriptor.model_fingerprint != ctx.attachment.model_fingerprint
                    or descriptor.method_id != ctx.attachment.method
                    or descriptor.deadline > ctx.deadline
                    or descriptor.deadline < self.monotonic() + reservation.required_seconds):
                raise AttachmentError("invalid_evidence")
            if source:
                if (type(descriptor) is not CaptureWorkerDescriptor
                        or descriptor.capture_id != ctx.attachment.capture_id
                        or descriptor.source_session_id != ctx.attachment.source_session_id
                        or descriptor.source_turn_nonce != ctx.attachment.turn_nonce
                        or descriptor.span_policy != ctx.attachment.span_policy
                        or descriptor.max_prefix_tokens != ctx.attachment.retained_count_limit
                        or descriptor.max_capture_bytes != ctx.attachment.max_capture_bytes):
                    raise AttachmentError("invalid_evidence")
            elif (type(descriptor) is not ConsumeWorkerDescriptor
                    or descriptor.attachment_digest != ctx.attachment.binding_digest
                    or descriptor.receiver_session_id != ctx.attachment.receiver_session_id
                    or descriptor.message_id != ctx.attachment.message_id
                    or descriptor.consume_nonce != ctx.attachment.request_id
                    or descriptor.opaque_handle != ctx.attachment.state_handle
                    or descriptor.prefix_offset != ctx.prefix_offset):
                raise AttachmentError("invalid_evidence")
            ctx.descriptor = descriptor
            if {k: v for k, v in vars(obj).items() if k != "occ3_worker_descriptor"} != protected:
                raise AttachmentError("unsupported_profile")
            if self.epoch() >= ctx.attachment.expires_at_epoch:
                raise AttachmentError("attachment_expired")
            ctx.dispatched = True  # Conservative: original may send before first yield.
            ctx.inner = original(tm, obj, request)
            async for output in ctx.inner:
                if obj.rid != ctx.native_rid or not isinstance(output, dict):
                    raise AttachmentError("native_completion_failed")
                meta = output.get("meta_info", {})
                if not isinstance(meta, dict) or meta.get("id") != ctx.native_rid:
                    raise AttachmentError("native_completion_failed")
                if ctx.native_terminal:
                    raise AttachmentError("native_completion_failed")
                reason = meta.get("finish_reason")
                if reason is not None:
                    if not isinstance(reason, dict) or reason.get("type") not in ("stop", "length"):
                        # Scheduler-side refusals abort the request before prefill; the
                        # ordinary stream delivers that as one abort output whose message
                        # is a fixed occ3 refusal code. The public code stays generic;
                        # the bounded code is recorded server-side so live refusals are
                        # diagnosable offline. Other abort payloads are never logged.
                        logger.error("occ3-bridge: native turn aborted (refusal=%s status=%s)",
                                     _refusal_code(reason), _refusal_status(reason))
                        raise AttachmentError("native_completion_failed")
                    ctx.native_terminal = True
                yield output
            ctx.exhausted = True
        except asyncio.CancelledError:
            raise
        except AttachmentError:
            raise
        except Exception as exc:
            # Unexpected non-abort failures also fold into the generic 409; record
            # the bounded exception class server-side so live refusals caused by
            # adapter/engine defects are diagnosable offline (no payload logged).
            logger.error("occ3-bridge: native turn failed (class=%s)", type(exc).__name__)
            raise AttachmentError() from None

    async def _run(self, original, serving, request, raw_request, ctx):
        response = await original(serving, request, raw_request)
        if ctx.cancel_requested:
            raise AttachmentError("client_disconnected", 499)
        if ctx.reservation is None or ctx.generation_calls != 1 or getattr(response, "status_code", 200) != 200:
            # The public code stays generic; the container log records which
            # condition fired so live refusals are diagnosable offline. No
            # request bytes, identifiers or backend handles are logged.
            logger.error(
                "occ3-bridge: native completion refused (attempt=%s cause=%s generation_calls=%s "
                "inner_status=%s inner_message=%r stream=%s)",
                ctx.attempt_id and "bound" or "none",
                "reservation_missing" if ctx.reservation is None else
                ("generation_calls" if ctx.generation_calls != 1 else "inner_error_response"),
                ctx.generation_calls, getattr(response, "status_code", 200),
                _inner_error_detail(response, 200),
                bool(getattr(request, "stream", False)),
            )
            raise AttachmentError("native_completion_failed")
        source = type(ctx.attachment) is CaptureIntent
        headers = list(getattr(response, "raw_headers", [(b"content-type", b"application/json")]))
        if source:
            encodings = [value.strip().lower() for key, value in headers if key.lower() == b"content-encoding"]
            if encodings not in ([], [b"identity"]):
                raise AttachmentError("unsupported_profile")
        if ctx.terminal_mode is not None:
            return await self._run_terminal_stream(response, headers, ctx)
        audit = NativeOutputAudit(ctx.reservation.max_output_bytes, source=source)
        if getattr(request, "stream", False):
            iterator = getattr(response, "body_iterator", None)
            if iterator is None:
                raise AttachmentError("native_completion_failed")
            ctx.response_iterator = iterator
            chunks = []
            async for chunk in iterator:
                if isinstance(chunk, str):
                    chunk = chunk.encode(getattr(response, "charset", "utf-8"))
                if not isinstance(chunk, bytes):
                    raise AttachmentError("native_completion_failed")
                audit.feed(chunk)
                chunks.append(chunk)
            body = b"".join(chunks)
            completion = audit.finish(ctx.native_rid)
        else:
            if isinstance(response, Response):
                body = response.body
            elif callable(getattr(response, "model_dump_json", None)):
                body = response.model_dump_json().encode("utf-8")
            else:
                raise AttachmentError("native_completion_failed")
            completion = audit.nonstream(body, ctx.native_rid)
        if not ctx.exhausted:
            try:
                await ctx.outer.__anext__()
            except StopAsyncIteration:
                pass
            else:
                raise AttachmentError("native_completion_failed")
        if not ctx.native_terminal or not ctx.exhausted:
            raise AttachmentError("native_completion_failed")
        if ctx.cancel_requested:
            raise AttachmentError("client_disconnected", 499)
        if source:
            provenance = audit.source_provenance(ctx.attachment.span_policy)
            evidence = await self.backend.finish_source(ctx.enrollment, ctx.reservation, completion, provenance)
            if ctx.cancel_requested:
                raise AttachmentError("client_disconnected", 499)
            receipt = validate_source_evidence(ctx.attachment, ctx.enrollment, ctx.reservation,
                                              ctx.descriptor, completion, provenance, evidence)
            if self.epoch() >= receipt.expires_at_epoch:
                raise AttachmentError("attachment_expired")
            ctx.receipt = source_receipt_bytes(receipt, self.profile.max_receipt_bytes)
        else:
            evidence = await self.backend.finish(ctx.reservation, completion)
            if ctx.cancel_requested:
                raise AttachmentError("client_disconnected", 499)
            validate_evidence(ctx.attachment, ctx.reservation, ctx.descriptor, completion, evidence)
        if any(key.lower() in (RECEIPT_HEADER.encode(), SOURCE_RECEIPT_HEADER.encode()) for key, _ in headers):
            raise AttachmentError("invalid_evidence")
        headers = [(k, v) for k, v in headers if k.lower() != b"cache-control"]
        header = SOURCE_RECEIPT_HEADER if source else RECEIPT_HEADER
        headers.extend([(header.encode(), ctx.receipt), (b"cache-control", b"no-store")])
        if source and not any(key.lower() == b"content-encoding" for key, _ in headers):
            headers.append((b"content-encoding", b"identity"))
        # Reserve space for ASGI/proxy ordinary headers beyond the native response.
        native_header_bytes = sum(len(k) + len(v) + 4 for k, v in headers)
        if native_header_bytes + self.profile.ordinary_header_reserve > self.profile.aggregate_header_bytes:
            raise AttachmentError("buffer_limit", 413)
        if self.epoch() >= ctx.attachment.expires_at_epoch:
            raise AttachmentError("attachment_expired")
        result = Response(body, status_code=200)
        result.raw_headers = headers
        # The original delayed abort background is replaced by eager tracked cleanup.
        ctx.completed = True
        return result

    def _terminal_admission(self, request, raw_request) -> bool:
        """Terminal-only admission checks. False means refusal before dispatch.

        With terminal_profile None the mode is refused and no capability is
        advertised; no Env/runtime/HTTP boolean can enable it.
        """
        if self.terminal_profile is None:
            raise AttachmentError("unsupported_attachment", 503)
        # Resolve the strict-overlay capability BEFORE any authorization,
        # enrollment, reservation, converter or native dispatch side effect:
        # a baseline (unpatched) runtime must refuse with zero backend calls.
        if self._terminal_strict_flag() is None:
            raise AttachmentError("unsupported_attachment", 503)
        if not getattr(request, "stream", False):
            return False
        encodings = raw_request.headers.getlist("accept-encoding")
        if len(encodings) != 1 or encodings[0].strip().lower() != "identity":
            return False
        excluded = ("logprobs", "top_logprobs", "return_hidden_states", "return_routed_experts",
                    "return_prompt_token_ids", "return_token_ids", "return_meta_info",
                    "return_input_ids_in_sglext", "return_output_ids_in_sglext",
                    "return_sampling_mask", "sampling_logprobs_mode", "return_cached_tokens_details",
                    "return_spec_tokens_details", "routed_experts_start_len")
        if any(getattr(request, field, None) not in (None, False, 0) for field in excluded):
            return False
        stream_options = getattr(request, "stream_options", None)
        if stream_options is not None and (getattr(stream_options, "include_usage", False)
                                            or getattr(stream_options, "continuous_usage_stats", False)):
            return False
        if not self._terminal_tools_admissible(getattr(request, "tools", None) or []):
            return False
        if self._terminal_active >= self.terminal_profile.max_terminal_active:
            raise AttachmentError("quota_exceeded", 429)
        return True

    def _terminal_tools_admissible(self, tools) -> bool:
        """Classify request tool schemas with the parser's own predicate.

        A tool whose parameters need buffered-argument interpretation is only
        admissible under the strict-stream overlay (the patched detector raises
        instead of re-emitting). Oversized or unclassifiable schemas refuse
        pre-dispatch. No schema or tool definition is rewritten.
        """
        try:
            from sglang.srt.function_call.glm47_moe_detector import _needs_argument_context
        except Exception:
            return False
        for tool in tools:
            function = getattr(tool, "function", tool)
            parameters = getattr(function, "parameters", None)
            if parameters is None:
                parameters = {}
            if not isinstance(parameters, dict):
                return False
            if len(canonical(parameters)) > self.terminal_profile.max_frame_bytes:
                return False
            try:
                if _needs_argument_context(parameters):
                    continue  # Admitted under the strict-stream overlay only.
            except Exception:
                return False
        return True

    def _terminal_strict_flag(self):
        """The patched detector's ContextVar, or None when the overlay is absent."""
        try:
            from sglang.srt.function_call.glm47_moe_detector import _OCC3_STRICT_STREAM
            return _OCC3_STRICT_STREAM
        except Exception:
            return None

    def _release_terminal(self, ctx):
        """Exactly-once terminal charge release, guarded per context.

        A context that never acquired a charge (quota or admission refusal
        before the increment) cannot decrement anything: another live
        request's counts are never touched.
        """
        if ctx.terminal_attempt_charged:
            ctx.terminal_attempt_charged = False
            self._terminal_attempts.discard(ctx.attempt_id)
        if ctx.terminal_active_charged:
            ctx.terminal_active_charged = False
            if self._terminal_active > 0:
                self._terminal_active -= 1

    async def _run_terminal_stream(self, response, headers, ctx):
        """Terminal passthrough: forward P early, hold D, prove, then event+D+EOF.

        Runs only after the full admission chain (identity, reservation,
        bind, descriptor validation) has already succeeded exactly as in the
        buffered mode; the same backend finish/finish_source proof runs at
        generator EOF before any private event is emitted. A single
        _TerminalOwner owns the whole post-handoff lifetime: the passthrough
        generator's finally, the ASGI response lifecycle and a bounded
        watchdog all converge on one tracked settle. Nothing relies on
        async-generator GC finalization, __del__ or the caller closing
        anything.
        """
        iterator = getattr(response, "body_iterator", None)
        if iterator is None:
            raise AttachmentError("native_completion_failed")
        strict = self._terminal_strict_flag()
        if strict is None:
            # Belt for direct calls; admission already refuses pre-dispatch.
            raise AttachmentError("unsupported_attachment", 503)
        source = type(ctx.attachment) is CaptureIntent
        profile = self.terminal_profile
        audit = TerminalOutputAudit(max_frame_bytes=profile.max_frame_bytes,
                                    max_native_bytes=profile.max_native_bytes,
                                    max_json_depth=profile.max_json_depth,
                                    max_json_nodes=profile.max_json_nodes,
                                    source=source)
        out_headers = [(k, v) for k, v in headers
                       if k.lower() not in (b"content-encoding",)]
        if any(key.lower() in (RECEIPT_HEADER.encode(), SOURCE_RECEIPT_HEADER.encode(),
                               TERMINAL_ACK_HEADER.encode()) for key, _ in out_headers):
            raise AttachmentError("invalid_evidence")
        out_headers.extend([(b"content-encoding", b"identity"),
                            (TERMINAL_ACK_HEADER.encode(), ctx.terminal_mode.encode("ascii")),
                            (b"cache-control", b"no-store")])
        ctx.response_iterator = iterator
        ctx.terminal_ledger = _TerminalLedger()
        owner = self._TerminalOwner(self, ctx, strict)
        gen = self._terminal_passthrough(iterator, audit, strict, ctx, owner)
        ctx.terminal_handed_off = True  # The owner now owns charges and cleanup.
        # The original delayed-abort background is not carried on the opted
        # path: the owner settles abort/retire deterministically instead.
        # The bounded watchdog covers a response that is never invoked at all.
        owner.start_watchdog()
        return self._TerminalStreamingResponse(
            self._TerminalStream(gen, owner),
            status_code=200,
            headers={k.decode("latin-1"): v.decode("latin-1") for k, v in out_headers},
            media_type="text/event-stream", owner=owner)

    class _TerminalOwner:
        """Single coherent lifetime owner for one terminal passthrough context.

        All settlement entry points (passthrough finally, ASGI response
        lifecycle, watchdog, explicit aclose, pre-handoff refusal) converge
        on the adapter's one tracked settle task. Charges are retained until
        known success or a positively confirmed backend retire.
        """

        def __init__(self, adapter, ctx, strict):
            self.adapter = adapter
            self.ctx = ctx
            self.strict = strict
            self.emission_token = None

        def open_emission(self):
            # Called when the ASGI response lifecycle starts: the emission
            # is a required operation, resolved only when that lifecycle has
            # ended (no further fenced send can be issued for the response).
            if self.emission_token is None:
                self.emission_token = self.ctx.terminal_ledger.register("emission")
            return self.emission_token

        async def ensure_settled(self):
            await self.adapter._terminal_settle(self.ctx)

        async def asgi_settle(self):
            # Emission quiescence: super().__call__ has returned or raised,
            # so no further fenced send can be issued for this response.
            # Resolve the emission op and re-evaluate the ALL-work release
            # predicate. Never waits on the settle task unbounded and never
            # self-awaits: a cancellation during the bounded wait still
            # resolves the emission below (the settle task continues on its
            # own; it never awaits the response task).
            try:
                await self.adapter._terminal_settle(self.ctx)
            except asyncio.CancelledError:
                pass
            token, self.emission_token = self.emission_token, None
            if token is not None:
                self.ctx.terminal_ledger.resolve(token, True)
                self.adapter._terminal_try_release(self.ctx)

        def start_watchdog(self):
            try:
                self.ctx.terminal_watchdog_task = asyncio.create_task(self._watchdog())
            except RuntimeError:
                self.ctx.terminal_watchdog_task = None  # no running loop: nothing to watch

        async def _watchdog(self):
            ctx, adapter = self.ctx, self.adapter
            try:
                while True:
                    if ctx.completed or ctx.terminal_settled:
                        return
                    if adapter.monotonic() >= ctx.deadline:
                        await adapter._terminal_settle(ctx)
                        return
                    try:
                        if await ctx.raw_request.is_disconnected():
                            await adapter._terminal_settle(ctx)
                            return
                    except Exception:
                        pass
                    await asyncio.sleep(adapter.profile.disconnect_interval)
            except asyncio.CancelledError:
                raise

        async def watch(self, awaitable, kind):
            """Bounded await for a post-handoff native/proof operation.

            Registers the operation in the context ledger BEFORE it starts,
            races it against the absolute effective deadline and client
            disconnect, and rechecks the deadline AFTER the operation
            completes and BEFORE its result is forwarded: a result arriving
            after the deadline resolves its own op but is never forwarded as
            stream success. A stalled operation is cancelled with bounded
            grace and, if it resists, stays context-owned and strongly
            tracked with charged retention. A known normal StopAsyncIteration
            is an explicit positive native EOF outcome, not a failure.
            """
            ctx, adapter = self.ctx, self.adapter
            ledger = ctx.terminal_ledger
            token = ledger.register(kind)
            task = asyncio.ensure_future(awaitable)
            try:
                while True:
                    if asyncio.current_task().cancelling():
                        raise asyncio.CancelledError
                    remaining = ctx.deadline - adapter.monotonic()
                    if remaining <= 0:
                        raise AttachmentError("deadline_exceeded", 504)
                    if await ctx.raw_request.is_disconnected():
                        raise AttachmentError("client_disconnected", 499)
                    if asyncio.current_task().cancelling():
                        raise asyncio.CancelledError
                    done, _ = await asyncio.wait(
                        {task}, timeout=min(remaining, adapter.profile.disconnect_interval))
                    if done:
                        if await ctx.raw_request.is_disconnected():
                            raise AttachmentError("client_disconnected", 499)
                        if asyncio.current_task().cancelling():
                            raise asyncio.CancelledError
                        # Post-await deadline recheck (before forwarding):
                        # late completion is not stream success.
                        if adapter.monotonic() >= ctx.deadline:
                            ledger.resolve_task(token, task)
                            adapter._terminal_try_release(ctx)
                            raise AttachmentError("deadline_exceeded", 504)
                        try:
                            result = task.result()
                        except StopAsyncIteration:
                            # Normal native EOF: explicit positive outcome.
                            ledger.resolve(token, True)
                            adapter._terminal_try_release(ctx)
                            raise
                        ledger.resolve(token, True)
                        adapter._terminal_try_release(ctx)
                        return result
            except StopAsyncIteration:
                raise
            except BaseException:
                ctx.cancel_requested = True
                raise
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait({task}, timeout=adapter.profile.cleanup_seconds)
                    except asyncio.CancelledError:
                        pass
                    finally:
                        if not task.done():
                            # Resistant survivor after the bounded cancel
                            # grace: context-owned, strongly tracked; a late
                            # completion resolves the op (positive only if
                            # it finishes normally). Never silently dropped.
                            adapter._terminal_track(task, ctx, token)
                else:
                    ledger.resolve_task(token, task)
                    adapter._terminal_try_release(ctx)

    class _TerminalStream:
        """Async-iterator handle for the terminal passthrough.

        Explicit aclose is honored deterministically for both a suspended
        and a never-started passthrough: the tracked settle runs either way,
        so an abandoned response never leaks quota or backend state.
        """

        def __init__(self, gen, owner):
            self._gen = gen
            self._owner = owner

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self._gen.__anext__()

        async def aclose(self):
            try:
                await asyncio.wait_for(
                    self._gen.aclose(), timeout=self._owner.adapter.profile.cleanup_seconds)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            finally:
                await self._owner.ensure_settled()

        def __getattr__(self, name):
            return getattr(self._gen, name)

    class _TerminalStreamingResponse(StreamingResponse):
        """StreamingResponse with a fenced ASGI emission and lifecycle settle.

        The emission itself — response start, every body frame, the private
        event, D and the final clean EOF — is fenced: once the irreversible
        failure latch is set, the absolute deadline has expired or the client
        is gone, no NEW send may be initiated. A send that already entered
        the underlying transport before the failure has an uncertain wire
        outcome that cannot be retracted; no further success bytes, clean
        EOF or release_success may be produced afterwards. The response
        task is captured so the tracked settle can fence (cancel) a stalled
        emission; quiescence is established here in the finally (after
        super().__call__ has returned or raised, no further fenced send can
        be issued), never by GC or a destructor.
        """

        def __init__(self, content, *args, owner=None, **kwargs):
            super().__init__(content, *args, **kwargs)
            self._occ3_owner = owner

        async def __call__(self, scope, receive, send):
            owner = self._occ3_owner
            if owner is None:
                await super().__call__(scope, receive, send)
                return
            adapter, ctx = owner.adapter, owner.ctx
            ctx.terminal_response_task = asyncio.current_task()
            owner.open_emission()

            async def fenced_send(message):
                # Fence EVERY emission event before it starts.
                if ctx.terminal_failure_latch or ctx.cancel_requested:
                    raise AttachmentError("client_disconnected", 499)
                if adapter.monotonic() >= ctx.deadline:
                    raise AttachmentError("deadline_exceeded", 504)
                if await ctx.raw_request.is_disconnected():
                    raise AttachmentError("client_disconnected", 499)
                return await send(message)

            try:
                await super().__call__(scope, receive, fenced_send)
            finally:
                owner, self._occ3_owner = self._occ3_owner, None
                if owner is not None:
                    await owner.asgi_settle()

    async def _terminal_passthrough(self, iterator, audit, strict, ctx, owner):
        """Stream-lifetime body: bytes out, proof at EOF, settle in finally.

        Success is only marked after the post-D recheck; charges are then
        released by the ALL-work predicate (the ASGI emission must also have
        settled — the producer yielding D is never sufficient). Once the
        failure latch is set no further success bytes or clean EOF may be
        newly initiated, even when the consumer later unblocks.
        """
        token = None
        try:
            while True:
                if ctx.terminal_failure_latch or ctx.terminal_settled:
                    raise AttachmentError("native_completion_failed")
                if await ctx.raw_request.is_disconnected():
                    raise AttachmentError("client_disconnected", 499)
                if self.monotonic() >= ctx.deadline:
                    raise AttachmentError("deadline_exceeded", 504)
                try:
                    token = strict.set(True)
                    try:
                        chunk = await owner.watch(iterator.__anext__(), "native")
                    finally:
                        strict.reset(token)
                        token = None
                except StopAsyncIteration:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                if not isinstance(chunk, bytes):
                    raise AttachmentError("native_completion_failed")
                for frame in audit.feed(chunk):
                    yield frame
            completion = audit.finish(audit.rid)
            if audit.rid != ctx.native_rid:
                raise AttachmentError("native_completion_failed")
            if not ctx.native_terminal or not ctx.exhausted:
                raise AttachmentError("native_completion_failed")
            await owner.watch(self._terminal_backend_proof(ctx, completion, audit), "proof")
            # Recheck AFTER the awaited proof and BEFORE emitting success: a
            # late proof must not produce success bytes.
            if ctx.cancel_requested or ctx.terminal_failure_latch:
                raise AttachmentError("client_disconnected", 499)
            if self.monotonic() >= ctx.deadline:
                raise AttachmentError("deadline_exceeded", 504)
            source = type(ctx.attachment) is CaptureIntent
            inner = json.loads(ctx.receipt.decode("ascii"))
            envelope = terminal_envelope("source" if source else "receiver", ctx.native_rid,
                                         completion.native_rid, completion.output_sha256,
                                         completion.output_bytes, inner)
            yield terminal_event_bytes(envelope)
            # Emitted bytes cannot be retracted; still, a settle racing the
            # DONE frame must not complete an uncertain delivery as success.
            if ctx.cancel_requested or ctx.terminal_failure_latch \
                    or self.monotonic() >= ctx.deadline:
                raise AttachmentError("deadline_exceeded", 504)
            yield audit.done_frame
            # Post-D recheck (after the yield resumes, before completing):
            # an expired, cancelled or latched stream must not finish as
            # success and must not lead to a clean EOF, even if the
            # consumer unblocks later.
            if (ctx.terminal_failure_latch or ctx.cancel_requested
                    or self.monotonic() >= ctx.deadline
                    or await ctx.raw_request.is_disconnected()):
                raise AttachmentError("deadline_exceeded", 504)
            ctx.completed = True
            # No charge release here: the ALL-work predicate (including the
            # still-open ASGI emission) decides, in the settle/finally path.
            return
        except GeneratorExit:
            # The client abandoned the stream mid-flight. Delivery state is
            # uncertain: no further bytes are emitted and the backend commit,
            # if any, does not prove delivery. The tracked settle in finally
            # aborts/retires with fail-closed charge retention.
            raise
        finally:
            if token is not None:
                strict.reset(token)
            await owner.ensure_settled()

    def _terminal_release_charges(self, ctx):
        """Release both owned frontend charge domains exactly once."""
        if ctx.terminal_released:
            return
        ctx.terminal_released = True
        self._release(ctx)
        self._release_terminal(ctx)

    def _terminal_try_release(self, ctx):
        """ALL-work barrier: release only when EVERY required op is positive.

        State predicate (all must hold, checked in this order):
          1. not ctx.terminal_released (exactly-once);
          2. ctx.terminal_settled: the single settle body has entered and
             registered the context's FULL obligation set (abort/retire/
             close registration happens before any await inside it), or the
             success settle ran — a mid-stream positive resolution inside
             watch() is a PARTIAL op set and must never release;
          3. ctx.completed or ctx.terminal_failure_latch: the stream ended
             as known success (proof+D+clean post-D recheck) or the
             irreversible failure latch is set — a still-live stream holds
             its charges even when every op observed so far is positive;
          4. every registered op (native/proof/abort/retire/close/emission)
             is positive; one pending/cancelled/failed/unknown operation
             retains BOTH charge domains and the attempt registration, and
             new work is then refused at the limits.
        """
        if ctx.terminal_released:
            return
        if not ctx.terminal_settled:
            return  # obligations not registered yet: never a partial-set release
        if not (ctx.completed or ctx.terminal_failure_latch):
            return  # stream still live: success unproven, failure not latched
        ledger = ctx.terminal_ledger
        if ledger is None or not ledger.all_positive():
            return
        self._terminal_release_charges(ctx)

    def _terminal_track(self, task, ctx, token):
        """Strongly track an unresolved context-owned task (charged retention).

        The task and its ledger token stay context-owned; a late completion
        resolves the op exactly once (positive only if it finishes normally)
        and re-evaluates the release predicate. Never a release by itself.
        """
        self._terminal_tasks.add(task)
        ctx.terminal_ledger.attach(token, task)

        def settled(future):
            self._terminal_tasks.discard(future)
            self._terminal_op_done(ctx, token, future)

        task.add_done_callback(settled)

    def _terminal_op_done(self, ctx, token, task):
        """Resolve one required op by the task outcome; re-check release."""
        ledger = ctx.terminal_ledger
        ledger.resolve_task(token, task)
        if ledger.kind(token) == "retire" and ledger.state(token) == "positive":
            ctx.retire_confirmed = True
        self._terminal_try_release(ctx)

    async def _terminal_bounded_op(self, awaitable, ctx, token):
        task = asyncio.ensure_future(awaitable)
        await self._terminal_bounded_op_task(task, ctx, token)

    async def _terminal_bounded_op_task(self, task, ctx, token):
        """Bounded wait for one required cleanup operation (abort/retire/close).

        Cancellation, timeout or an uncooperative task is NOT completion:
        after the bounded cancel grace an unresolved task stays quarantined
        with charged ownership until it settles on its own.
        """
        ledger = ctx.terminal_ledger
        ledger.attach(token, task)
        try:
            await asyncio.wait({task}, timeout=self.profile.cleanup_seconds)
        except asyncio.CancelledError:
            pass
        if task.done():
            self._terminal_op_done(ctx, token, task)
            return
        task.cancel()
        try:
            await asyncio.wait({task}, timeout=self.profile.cleanup_seconds)
        except asyncio.CancelledError:
            pass
        if task.done():
            self._terminal_op_done(ctx, token, task)
            return
        # Resistant after the bounded cancel grace: stays pending with
        # charged ownership; a late positive completion resolves once.
        self._terminal_track(task, ctx, token)

    async def _terminal_settle(self, ctx):
        """Create-or-await the single tracked settle task; never blocks unbounded."""
        task = ctx.terminal_settle_task
        if task is None:
            try:
                ctx.terminal_settle_task = task = asyncio.create_task(
                    self._terminal_settle_body(ctx))
            except RuntimeError:
                return  # no running loop: nothing further can be settled here
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.profile.cleanup_seconds * 8 + self.profile.disconnect_interval)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            # The settle task continues on its own; charges settle per its
            # outcome (retained unless positively confirmed or success).
            pass

    async def _terminal_settle_body(self, ctx):
        """One tracked settle for a terminal context (failure or abandonment).

        ALL-work barrier: every required operation of the context (native
        next, proof, abort, retire, iterator closes, ASGI emission) must be
        positive before the charge-release predicate can fire. Registration
        of the settle obligations happens BEFORE any await (and before the
        emission fence), so the predicate never observes a partial set. A
        cancelled, failed or still-unknown operation is NOT completion:
        charges stay held (fail-closed) and new work is refused at the
        limits. No automatic retry of uncertain backend operations.
        """
        if ctx.terminal_settled:
            return
        ctx.terminal_settled = True
        if ctx.terminal_ledger is None:
            ctx.terminal_ledger = _TerminalLedger()
        ledger = ctx.terminal_ledger
        # Stop the watchdog (orchestration task, superseded by this settle).
        # Cancel only — never await it from here: the watchdog itself awaits
        # this settle task, so awaiting it back would be a self-wait cycle.
        watchdog = ctx.terminal_watchdog_task
        if watchdog is not None and not watchdog.done() \
                and asyncio.current_task() is not watchdog:
            watchdog.cancel()
        if ctx.completed:
            # Known success: no abort/retire/close obligations exist (the
            # native iterator already reached its positive EOF, the proof is
            # positive, and a valid positive capture keeps its intended
            # backend lease). Release only when every registered required
            # operation — including the still-open ASGI emission — is
            # positive.
            self._terminal_try_release(ctx)
            return
        # Irreversible failure/abandonment latch: from this point no success
        # bytes, no clean EOF and no release_success may be newly produced,
        # even when the consumer later unblocks.
        ctx.terminal_failure_latch = True
        ctx.cancel_requested = True
        ctx.cleanup_started = True
        # Register EVERY settle obligation up front, before running or
        # awaiting any of them and before fencing the emission.
        abort_token = retire_token = None
        close_tokens = []
        if ctx.dispatched and ctx.native_rid is not None:
            abort_token = ledger.register("abort")
        if ctx.attempted and not ctx.retire_confirmed and not ctx.retire_started:
            retire_token = ledger.register("retire")
        # A still-pending native operation is driving the whole iterator
        # chain (response_iterator -> outer -> inner): their close obligation
        # is subsumed by that operation's own settlement, so no close is
        # registered (and none attempted) while it is in flight. A positive
        # native completion IS the iterators' settlement; a latched one keeps
        # the fail-closed retention regardless.
        native_in_flight = any(e[0] == "native" and e[1] == "pending"
                               for e in ledger.ops.values())
        if not native_in_flight:
            for iterator in (ctx.response_iterator, ctx.outer, ctx.inner):
                if iterator is not None and callable(getattr(iterator, "aclose", None)):
                    close_tokens.append((ledger.register("close"), iterator))
        # Active ASGI emissions are FENCED, not cancelled: the fenced send
        # refuses every NEW frame/EOF once this latch is set, the passthrough
        # loop-top refuses new work, and the emission op keeps the charges
        # quarantined until the lifecycle truly ends (its own finally resolves
        # the op). A stalled consumer send therefore cannot authorize any
        # future frame, clean EOF or new request while charged. This body
        # never cancels or awaits the response task, so no self-await cycle
        # and no caller-task cancellation pollution can occur.
        # Abort the native request (bounded, best-effort, tracked).
        if abort_token is not None:
            try:
                value = ctx.tm.abort_request(ctx.native_rid)
                if inspect.isawaitable(value):
                    await self._terminal_bounded_op(value, ctx, abort_token)
                else:
                    ledger.resolve(abort_token, True)
            except Exception:
                ledger.resolve(abort_token, False)
        # Tracked retire: a positive return is ONE required ack, never
        # permission to release while any other required op pends.
        if retire_token is not None:
            ctx.retire_started = True
            try:
                task = asyncio.ensure_future(
                    self.backend.retire(ctx.attempt_id, ctx.native_rid, ctx.dispatched))
            except Exception:
                task = None
            if task is None:
                ledger.resolve(retire_token, False)
            else:
                ctx.terminal_retire_task = task
                await self._terminal_bounded_op_task(task, ctx, retire_token)
        # Close owned iterators (bounded, best-effort, tracked).
        for token, iterator in close_tokens:
            try:
                await self._terminal_bounded_op(iterator.aclose(), ctx, token)
            except Exception:
                ledger.resolve(token, False)
        self._terminal_try_release(ctx)

    async def _terminal_backend_proof(self, ctx, completion, audit):
        """Run the same genuine backend proof as the buffered path, then bind it."""
        source = type(ctx.attachment) is CaptureIntent
        if source:
            provenance = audit.source_provenance(ctx.attachment.span_policy)
            evidence = await self.backend.finish_source(ctx.enrollment, ctx.reservation, completion, provenance)
            if ctx.cancel_requested:
                raise AttachmentError("client_disconnected", 499)
            receipt = validate_source_evidence(ctx.attachment, ctx.enrollment, ctx.reservation,
                                               ctx.descriptor, completion, provenance, evidence)
            if self.epoch() >= receipt.expires_at_epoch:
                raise AttachmentError("attachment_expired")
            ctx.receipt = source_receipt_bytes(receipt, self.profile.max_receipt_bytes)
        else:
            evidence = await self.backend.finish(ctx.reservation, completion)
            if ctx.cancel_requested:
                raise AttachmentError("client_disconnected", 499)
            validate_evidence(ctx.attachment, ctx.reservation, ctx.descriptor, completion, evidence)

    async def _guard(self, operation, ctx):
        task = asyncio.create_task(operation)
        try:
            while True:
                # Starlette's nonblocking disconnect probe uses an AnyIO cancel
                # scope. Recheck task cancellation if that scope swallowed it.
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
                remaining = ctx.deadline - self.monotonic()
                if remaining <= 0:
                    raise AttachmentError("deadline_exceeded", 504)
                if await ctx.raw_request.is_disconnected():
                    raise AttachmentError("client_disconnected", 499)
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
                done, _ = await asyncio.wait({task}, timeout=min(remaining, self.profile.disconnect_interval))
                if done:
                    if await ctx.raw_request.is_disconnected():
                        raise AttachmentError("client_disconnected", 499)
                    if asyncio.current_task().cancelling():
                        raise asyncio.CancelledError
                    return task.result()
        except BaseException:
            ctx.cancel_requested = True
            ctx.completed = False
            raise
        finally:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait({task}, timeout=self.profile.cleanup_seconds)
                finally:
                    if not task.done():
                        self._quarantine(task, ctx)
            if task.done() and not task.cancelled():
                task.exception()  # Retrieve failure without exposing its message.

    def _quarantine(self, task, ctx):
        task._occ3_context = ctx
        self._quarantined.add(task)
        def settled(future):
            self._quarantined.discard(future)
            if not future.cancelled():
                future.exception()
            if ctx.owner_finished and not any(getattr(t, "_occ3_context", None) is ctx for t in self._quarantined):
                self._release(ctx)
        task.add_done_callback(settled)

    async def _bounded_cleanup(self, awaitable, ctx):
        task = asyncio.create_task(awaitable)
        try:
            await asyncio.wait({task}, timeout=self.profile.cleanup_seconds)
        finally:
            if not task.done():
                task.cancel()
                self._quarantine(task, ctx)
            elif not task.cancelled():
                task.exception()

    async def _cleanup(self, ctx):
        if ctx.cleanup_started:
            return
        ctx.cleanup_started = True
        ctx.cancel_requested = True
        if ctx.dispatched and ctx.native_rid is not None:
            try:
                value = ctx.tm.abort_request(ctx.native_rid)
                if inspect.isawaitable(value):
                    await self._bounded_cleanup(value, ctx)
            except Exception:
                pass
        if ctx.attempted and not ctx.retired:
            ctx.retired = True
            try:
                await self._bounded_cleanup(
                    self.backend.retire(ctx.attempt_id, ctx.native_rid, ctx.dispatched), ctx)
            except Exception:
                pass
        for iterator in (ctx.response_iterator, ctx.outer, ctx.inner):
            if iterator is not None and callable(getattr(iterator, "aclose", None)):
                try:
                    await self._bounded_cleanup(iterator.aclose(), ctx)
                except Exception:
                    pass
