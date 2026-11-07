"""Bounded, output-only native turn telemetry; never inspect request payloads."""
import json
from functools import wraps
import logging
import os
import re

LOG = logging.getLogger("occ3_sglang.turn_diagnostics")

MAX_LOG_CHARS = 16 * 1024
LOG_PREFIX = "occ3-bridge: native turn summary "


def _isolated(method):
    """Telemetry failures must never alter native completion or cleanup."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if self.disabled and method.__name__ not in ("emit", "inner_event"):
            return
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self.disabled = True
            self.pending = b""
            # Fixed operation codes only: exception messages may contain output.
            if self.diag_error is None:
                self.diag_error = method.__name__ + "_failed"
    return guarded


class TurnDiagnostics:
    """Own one summary and bounded output excerpts across response handoffs."""

    def __init__(self, clock, attempt_id, channel, keepalive):
        self.clock = clock
        self.started = clock()
        self.attempt_id = attempt_id
        self.channel = channel
        self.keepalive = keepalive
        try:
            self.limit = max(0, min(4096, int(os.environ.get("OCC3_DIAG_TURN_TEXT", "0"))))
        except ValueError:
            self.limit = 0
        self.events = 0
        self.first = None
        self.chars = {"content": 0, "reasoning": 0}
        self.head = dict.fromkeys(self.chars, "")
        self.tail = dict.fromkeys(self.chars, "")
        self.completion_tokens = None
        self.reasoning_tokens = None
        self.finish_reason = None
        self.pending = b""
        self.disabled = False
        self.diag_error = None
        self.failure = None
        self.logged = False

    @_isolated
    def inner_event(self):
        if self.first is None:
            self.first = self.clock() - self.started
        self.events += 1

    def fail(self, reason, status):
        if self.failure is None:
            self.failure = (reason, status)

    @_isolated
    def payload(self, value):
        """Read only renderer output fields, never arbitrary metadata."""
        if not isinstance(value, dict):
            return
        usage = value.get("usage")
        if isinstance(usage, dict):
            for field in ("completion_tokens", "reasoning_tokens"):
                number = usage.get(field)
                if type(number) is int and number >= 0:
                    setattr(self, field, number)
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int:
                self.reasoning_tokens = max(0, details["reasoning_tokens"])
        choices = value.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            reason = choice.get("finish_reason")
            if reason is not None:
                self.finish_reason = reason if reason in ("stop", "length", "tool_calls", "function_call", "content_filter") else "other"
            delta = choice.get("delta", choice.get("message", {}))
            if not isinstance(delta, dict):
                continue
            for name, field in (("content", "content"), ("reasoning", "reasoning_content")):
                text = delta.get(field, delta.get("reasoning", delta.get("reasoning_text")) if name == "reasoning" else None)
                if not isinstance(text, str):
                    continue
                self.chars[name] += len(text)
                if self.limit:
                    self.head[name] = (self.head[name] + text[:self.limit])[:self.limit]
                    self.tail[name] = (self.tail[name] + text[-self.limit:])[-self.limit:]

    @_isolated
    def nonstream(self, body):
        self.payload(json.loads(body))

    @_isolated
    def feed(self, chunk):
        """Parse inner SSE only, bounded independently of the protocol audit."""
        if self.disabled:
            return
        self.pending += chunk
        if len(self.pending) > 1024 * 1024:
            self.pending = b""
            self.disabled = True
            return
        while True:
            boundary = re.search(rb"\r?\n\r?\n", self.pending)
            if boundary is None:
                break
            frame, self.pending = self.pending[:boundary.start()], self.pending[boundary.end():]
            if any(line.startswith(b"event:") for line in frame.splitlines()):
                continue
            data = b"\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:"))
            if data and data != b"[DONE]":
                self.nonstream(data)

    @_isolated
    def emit(self, completed, deadline):
        """Called at the owning lifecycle exit, including abandoned responses."""
        if self.logged:
            return
        self.logged = True
        if self.failure:
            reason, status = self.failure
        elif completed:
            reason, status = "complete", 200
        elif self.clock() >= deadline:
            reason, status = "deadline_exceeded", 504
        else:
            reason, status = "client_disconnected", 499
        row = dict(outcome="complete" if status == 200 else "error", reason=reason,
                   status=status, keepalive=self.keepalive, channel=self.channel,
                   attempt_id=self.attempt_id, elapsed_seconds=self.clock() - self.started,
                   first_inner_event_seconds=self.first, event_count=self.events,
                   content_chars=self.chars["content"], reasoning_chars=self.chars["reasoning"],
                   completion_tokens=self.completion_tokens, reasoning_tokens=self.reasoning_tokens,
                   finish_reason=self.finish_reason, output_parse_limited=self.disabled, diag_error=self.diag_error)
        if self.limit:
            row["text_limit"] = self.limit
            for name in self.chars:
                row[name + "_first"] = self.head[name]
                row[name + "_last"] = self.tail[name]
        encoded = json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        effective_limit = self.limit
        while len(LOG_PREFIX) + len(encoded) > MAX_LOG_CHARS and effective_limit > 0:
            effective_limit //= 2
            row["text_truncated_log_cap"] = effective_limit
            for name in self.chars:
                row[name + "_first"] = self.head[name][:effective_limit]
                row[name + "_last"] = self.tail[name][-effective_limit:] if effective_limit else ""
            encoded = json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        if len(LOG_PREFIX) + len(encoded) > MAX_LOG_CHARS:
            encoded = '{"diag_error":"summary_size_limit"}'
        LOG.info("%s%s", LOG_PREFIX, encoded)
