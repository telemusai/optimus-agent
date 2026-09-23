"""Capability-gated native lifecycle interface. No deletion fallback."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import host_request, _collect_target_selector


class UnsupportedCapability(RuntimeError):
    pass


def runtime_source_sha256() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py"), key=lambda path: path.name):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


async def lifecycle_capabilities(*, model: str | None = None, target: Any = None) -> dict[str, Any]:
    """Get a native host report, bound to this runtime's raw-byte source manifest.

    Build fingerprint must also match a separately approved deployment receipt.
    This is not a signature or a trust boundary against a hostile kernel.
    """
    if model is not None and target is not None:
        raise TypeError("choose model or target, not both")
    payload = {"model": model} if model is not None else ({"target": _collect_target_selector(target)} if target is not None else {})
    try:
        report = await host_request("rlm.lifecycle_capabilities", payload)
    except RuntimeError as exc:
        raise UnsupportedCapability("Host lacks the native lifecycle capability report") from exc
    if (not isinstance(report, dict) or report.get("schema") != "optimus.native-lifecycle.v1"
            or report.get("capability") != "rlm.stop-retain.v1"
            or not isinstance(report.get("supported"), bool)):
        raise UnsupportedCapability("Invalid native lifecycle capability report")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("hostImplementation") != "optimus-rust":
        raise UnsupportedCapability("Native host provenance is missing")
    for field in ("protocolVersion", "schemaRevision"):
        value = provenance.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise UnsupportedCapability("Invalid host protocol provenance")
    if provenance.get("runtimeSourceSha256") != runtime_source_sha256():
        raise UnsupportedCapability("Runtime source does not match the native build receipt")
    fingerprint = provenance.get("buildFingerprint")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise UnsupportedCapability("Native build is not provenance-pinned")
    if model is not None and (not isinstance(report.get("targetProfile"), dict)
            or report["targetProfile"].get("model") != model):
        raise UnsupportedCapability("Native report does not bind the requested model")
    return report


async def stop_subagent(target: Any, *, timeout_ms: int = 0) -> dict[str, Any]:
    """Cancel a direct child's execution without deleting its conversation.

    timeout_ms=0 returns admission immediately. Only settled=true with every
    acknowledgement true proves local execution settlement. No deletion fallback.
    """
    selector = _collect_target_selector(target)
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or not 0 <= timeout_ms <= 10000:
        raise TypeError("timeout_ms must be an integer from 0 to 10000")
    report = await lifecycle_capabilities(target=target)
    if not report["supported"]:
        raise UnsupportedCapability("Native target ownership profile is unsupported")
    receipt = await host_request("rlm.stop_subagent", {"target": selector, "timeout_ms": timeout_ms})
    if (not isinstance(receipt, dict) or receipt.get("schema") != "optimus.stop-retain.v1"
            or any(type(receipt.get(key)) is not bool for key in
                   ("accepted", "settled", "retained", "automatic_continuation_fenced"))
            or not isinstance(receipt.get("rlm_child_id"), str)
            or not isinstance(receipt.get("stop_generation"), str)):
        raise UnsupportedCapability("Invalid native retained-stop receipt")
    if receipt.get("settled") is True:
        acknowledged = receipt.get("acknowledged", {})
        if (receipt.get("retained") is not True or receipt.get("automatic_continuation_fenced") is not True
                or any(acknowledged.get(key) is not True for key in ("model", "tools", "kernel", "owned_processes", "transcript_flushed"))):
            raise UnsupportedCapability("Contradictory native settlement receipt")
    return receipt


async def active_execution(target: Any) -> dict[str, Any]:
    """Read the native current-run token. It is never a new-run authorization."""
    report = await lifecycle_capabilities(target=target)
    feature = report.get("activeOnlyMessages", {})
    if feature.get("capability") != "rlm.active-only-message.v1" or feature.get("supported") is not True:
        raise UnsupportedCapability("Native active-only messaging is unsupported")
    receipt = await host_request("rlm.active_execution", {"target": _collect_target_selector(target)})
    if not isinstance(receipt, dict) or receipt.get("schema") != "optimus.active-execution.v1":
        raise UnsupportedCapability("Invalid active-execution receipt")
    return receipt


async def resume_subagent(target: Any, *, stop_generation: str, prompt: str) -> dict[str, Any]:
    """Admit a fresh explicit audit task; never replay the old queue or Python heap."""
    if not isinstance(stop_generation, str) or not stop_generation or len(stop_generation) > 128:
        raise TypeError("stop_generation must be the exact retained-stop token")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > 32000:
        raise TypeError("prompt must be a nonempty bounded audit instruction")
    report = await lifecycle_capabilities(target=target)
    feature = report.get("auditResume", {})
    if feature.get("capability") != "rlm.audit-resume.v1" or feature.get("supported") is not True:
        raise UnsupportedCapability("Native retained audit resume is unsupported")
    receipt = await host_request("rlm.resume_subagent", {
        "target": _collect_target_selector(target), "stop_generation": stop_generation, "prompt": prompt,
    })
    if (not isinstance(receipt, dict) or receipt.get("schema") != "optimus.audit-resume.v1"
            or receipt.get("stop_generation") != stop_generation
            or receipt.get("old_work_replayed") is not False):
        raise UnsupportedCapability("Invalid retained audit receipt")
    return receipt
