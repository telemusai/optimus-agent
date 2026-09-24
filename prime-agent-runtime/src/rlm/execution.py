"""Explicit, cell-scoped process outcomes. Never infer failure from output text."""
from __future__ import annotations

import json
import math
from typing import Any

from .bash import BashResult
from . import repl

SCHEMA = "optimus.script-result.v1"
MAX_REPORTS = 32
MAX_RECEIPT_BYTES = 16 * 1024
VERIFICATION_DESIGNATION_KEY = "verification"
VERIFICATION_KIND_TASK_CHECK = "task_check"
MAX_VERIFICATION_LABEL_CHARS = 128


def _check_verification_designation(receipt: dict[str, Any]) -> None:
    """Strict opt-in task-verification designation (mirror of the Rust parser).

    Only ``{"kind": "task_check", "label": str}`` is accepted, exactly. A bare
    exit code or printed text is never verification evidence.
    """
    if VERIFICATION_DESIGNATION_KEY not in receipt:
        return
    designation = receipt[VERIFICATION_DESIGNATION_KEY]
    if not isinstance(designation, dict) or set(designation) != {"kind", "label"}:
        raise TypeError("receipt.verification must be exactly {kind, label}")
    if designation["kind"] != VERIFICATION_KIND_TASK_CHECK:
        raise TypeError("receipt.verification.kind must be task_check")
    label = designation["label"]
    if not isinstance(label, str) or not label.strip() or len(label) > MAX_VERIFICATION_LABEL_CHARS:
        raise TypeError("receipt.verification.label must be a non-empty string of at most 128 characters")


def _text(value: Any, name: str, maximum: int, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise TypeError(f"{name} must be a non-empty string of at most {maximum} characters")
    return value


def _exit_code(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not -(2**31) <= value < 2**32:
        raise TypeError("exit code must be a signed process return code or Windows DWORD")
    return value


def report_script_result(
    result: BashResult,
    *,
    stage: str = "process",
    script_id: str | None = None,
    receipt: dict[str, Any] | None = None,
    expected_exit_codes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    """Report a completed supervised process in the current ipython tool result.

    Ordinary bash results remain caller-owned and unchanged. This opt-in call
    marks unexpected exits or explicit receipt failures as tool errors, even if
    the outer cell succeeds. It does not raise for a reported process failure,
    retry anything, execute a command, or inspect ``result.output``. An expected
    nonzero code must be explicit. Signals and fatal/aborted receipts cannot be
    made successful by an expected-code override. Receipt data must be public.

    Task verification designation (CTRL-001): to mark a report as the task's
    verification, include a strictly-shaped designation in the receipt. The
    host records the real verification evidence only from such designated
    reports; a plain exit 0 is never verification, and the outcome stays the
    host-measured process result (never printed text)::

        from rlm.execution import report_script_result
        report_script_result(
            result,
            receipt={"verification": {"kind": "task_check",
                                      "label": "focused scope: status script"}},
        )

    The label is descriptive public metadata. Designation does not make the
    task complete or verified by itself; the bounded evaluator still judges
    relevance and sufficiency separately.
    """
    if not isinstance(result, BashResult):
        raise TypeError("result must be a completed BashResult, not a handle or printed output")
    code = _exit_code(result.exit_code)
    if (isinstance(result.duration, bool) or not isinstance(result.duration, (int, float))
            or not math.isfinite(result.duration) or result.duration < 0):
        raise TypeError("result.duration must be finite and non-negative")
    stage = _text(stage, "stage", 128)
    script_id = _text(script_id, "script_id", 256, optional=True)
    if not isinstance(expected_exit_codes, tuple) or not 1 <= len(expected_exit_codes) <= 32:
        raise TypeError("expected_exit_codes must be a tuple of 1 to 32 process return codes")
    expected = [_exit_code(item) for item in expected_exit_codes]
    metadata = None
    if receipt is not None:
        if not isinstance(receipt, dict):
            raise TypeError("receipt must be a public JSON object or None")
        for flag in ("isError", "fatal"):
            if flag in receipt and not isinstance(receipt[flag], bool):
                raise TypeError(f"receipt.{flag} must be bool")
        if "status" in receipt and receipt["status"] not in ("ok", "error", "fatal", "aborted"):
            raise TypeError("receipt.status must be ok, error, fatal, or aborted")
        _check_verification_designation(receipt)
        try:
            encoded = json.dumps(receipt, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("receipt must contain finite JSON values") from exc
        if len(encoded.encode("utf-8")) > MAX_RECEIPT_BYTES:
            raise ValueError("receipt exceeds 16 KiB; keep only public status metadata")
        metadata = json.loads(encoded)
    failed = (code < 0 or code not in expected or bool(metadata and (
        metadata.get("isError") is True or metadata.get("fatal") is True
        or metadata.get("status") in ("error", "fatal", "aborted"))))
    report = {"schema": SCHEMA, "stage": stage, "scriptId": script_id,
              "exitCode": code, "durationSeconds": float(result.duration),
              "expectedExitCodes": expected, "isError": failed, "receipt": metadata}
    execution = repl._current_cell_execution.get()
    if execution is None or execution.finished.is_set() or execution.owner is None:
        raise RuntimeError("report_script_result requires an active cell; report completed background work in a new cell")
    if len(execution.execution_reports) >= MAX_REPORTS:
        raise RuntimeError("too many script result reports in one cell")
    execution.execution_reports.append(json.loads(json.dumps(report)))
    return report
