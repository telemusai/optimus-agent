"""Tiny rlm-compatible kernel shim for Prime Agent."""

from __future__ import annotations

import sys
import types
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bash import BashHandle, BashResult, bash
from .harness import HarnessEntry, HarnessScope, HarnessState, RefinementEvent, get_harness_state

@dataclass(frozen=True)
class RLMSpawnHandle:
    rlm_child_id: str
    name: str
    session_dir: Path
    model: str


@dataclass(frozen=True)
class RLMCreateSessionHandle:
    active_session_id: str
    session_id: str
    name: str
    session_file: Path
    model: str


@dataclass(frozen=True)
class RLMModel:
    provider: str
    id: str
    name: str
    selector: str


@dataclass(frozen=True)
class RLMSubagent:
    rlm_child_id: str
    active_session_id: str | None
    session_id: str | None
    session_name: str
    session_dir: Path
    status: str
    model: str | None = None

    @property
    def name(self) -> str:
        """Compatibility alias for the canonical session_name field."""
        return self.session_name


@dataclass(frozen=True)
class RLMChildResult:
    rlm_child_id: str
    session_name: str | None
    session_dir: Path | None
    status: str
    settled: bool
    answer_preview: str | None
    error: str | None
    duration_ms: float | None
    tool_use_count: float | None
    replied_since_task: bool | None


def _spawn_handle_from_payload(payload: Any) -> RLMSpawnHandle:
    if not isinstance(payload, dict):
        raise RuntimeError("rlm.run returned an invalid spawn handle")
    child_id = payload.get("rlm_child_id")
    name = payload.get("name")
    session_dir = payload.get("session_dir")
    model = payload.get("model")
    if not all(isinstance(value, str) and value for value in (child_id, name, session_dir, model)):
        raise RuntimeError("rlm.run returned an invalid spawn handle")
    return RLMSpawnHandle(
        rlm_child_id=child_id,
        name=name,
        session_dir=Path(session_dir),
        model=model,
    )


def _create_session_handle_from_payload(payload: Any) -> RLMCreateSessionHandle:
    if not isinstance(payload, dict):
        raise RuntimeError("rlm.create_session returned an invalid payload")
    active_session_id = payload.get("active_session_id")
    session_id = payload.get("session_id")
    name = payload.get("name")
    session_file = payload.get("session_file")
    model = payload.get("model")
    if not all(isinstance(value, str) and value for value in (active_session_id, session_id, name, session_file, model)):
        raise RuntimeError("rlm.create_session returned an invalid payload structure")
    return RLMCreateSessionHandle(
        active_session_id=active_session_id,
        session_id=session_id,
        name=name,
        session_file=Path(session_file),
        model=model,
    )


def _parse_host_reply(request_type: str, reply: dict[str, Any]) -> dict[str, Any]:
    status = reply.get("status")
    if status == "ok":
        return reply["result"]
    if status == "error":
        raise RuntimeError(str(reply.get("error") or f"host request {request_type} failed"))
    raise RuntimeError(f"host request {request_type} returned unexpected status: {status!r}")


async def host_request(request_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Send a typed request to the Prime Agent host and await its reply.

    This is the kernel side of the generic host bridge: Python skills call
    ``await host_request("<type>", {...})`` and the TypeScript host dispatches
    on the type. Raises RuntimeError when the host reports an error or when no
    handler for the type is registered in this session.
    """
    if not isinstance(request_type, str) or not request_type:
        raise TypeError("request_type must be a non-empty str")
    if payload is not None and not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict or None, got {type(payload).__name__}")
    from . import repl

    # request_type goes last so a payload "type" key cannot reroute the request.
    reply = await repl.host_request({**(payload or {}), "type": request_type})
    return _parse_host_reply(request_type, reply)


def emit(data: dict[str, Any]) -> None:
    """Ship one display event (dict of MIME type -> JSON payload) to the host."""
    from . import repl

    repl.emit(data)


async def run(prompt: str, **kwargs: Any) -> RLMSpawnHandle:
    """Spawn a recursive Prime Agent child and return once its task is admitted.

    ``model`` selects a child with an exact ``provider/model`` selector.
    ``thinking`` sets the child reasoning level (e.g. 'off', 'low', 'medium', 'high');
    defaults to the parent level; levels invalid for the resolved model fail the spawn.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    payload = await host_request("rlm.run", {"prompt": prompt, "kwargs": kwargs})
    return _spawn_handle_from_payload(payload)


def _model_from_payload(payload: Any) -> RLMModel:
    if not isinstance(payload, dict):
        raise RuntimeError("rlm.find_models returned an invalid model entry")
    provider = payload.get("provider")
    model_id = payload.get("id")
    name = payload.get("name")
    selector = payload.get("selector")
    if not all(isinstance(value, str) and value for value in (provider, model_id, name, selector)):
        raise RuntimeError("rlm.find_models returned an invalid model entry")
    return RLMModel(provider=provider, id=model_id, name=name, selector=selector)


async def create_session(
    prompt: str,
    name: str | None = None,
    model: str | None = None,
    thinking: str | None = None,
    cwd: str | None = None,
) -> RLMCreateSessionHandle:
    """Create and prompt a resident depth-0 daemon session.

    Only daemon-backed depth-0 sessions support this operation. The optional
    arguments set the session name, model, thinking level, and working directory.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if model is not None:
        kwargs["model"] = model
    if thinking is not None:
        kwargs["thinking"] = thinking
    if cwd is not None:
        kwargs["cwd"] = cwd
    payload = await host_request("rlm.create_session", {"prompt": prompt, "kwargs": kwargs})
    return _create_session_handle_from_payload(payload)


async def find_models(query: str = "", limit: int = 8) -> list[RLMModel]:
    """Search a bounded list of models backed by active user credentials."""
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")
    if not isinstance(limit, int):
        raise TypeError(f"limit must be int, got {type(limit).__name__}")
    payload = await host_request("rlm.find_models", {"query": query, "limit": limit})
    models = payload.get("models")
    if not isinstance(models, list):
        raise RuntimeError("rlm.find_models returned an invalid models list")
    return [_model_from_payload(model) for model in models]


def _subagent_from_payload(payload: Any, operation: str = "rlm.list_subagents") -> RLMSubagent:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{operation} returned an invalid subagent entry")
    child_id = payload.get("rlm_child_id")
    active_session_id = payload.get("active_session_id")
    session_id = payload.get("session_id")
    session_name = payload.get("session_name")
    session_dir = payload.get("session_dir")
    status = payload.get("status")
    model = payload.get("model")
    if not isinstance(child_id, str) or not child_id:
        raise RuntimeError(f"{operation} entry is missing rlm_child_id")
    if active_session_id is not None and not isinstance(active_session_id, str):
        raise RuntimeError(f"{operation} entry has invalid active_session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise RuntimeError(f"{operation} entry has invalid session_id")
    if not isinstance(session_name, str) or not session_name:
        raise RuntimeError(f"{operation} entry is missing session_name")
    if not isinstance(session_dir, str) or not session_dir:
        raise RuntimeError(f"{operation} entry is missing session_dir")
    if status not in {"running", "completed", "error"}:
        raise RuntimeError(f"{operation} entry has invalid status")
    if model is not None and not isinstance(model, str):
        raise RuntimeError(f"{operation} entry has invalid model")
    return RLMSubagent(
        rlm_child_id=child_id,
        active_session_id=active_session_id,
        session_id=session_id,
        session_name=session_name,
        session_dir=Path(session_dir),
        status=status,
        model=model,
    )


async def list_subagents() -> list[RLMSubagent]:
    """List direct RLM children retained by the current parent session."""
    payload = await host_request("rlm.list_subagents")
    entries = payload.get("subagents")
    if not isinstance(entries, list):
        raise RuntimeError("rlm.list_subagents returned an invalid subagents registry")
    return [_subagent_from_payload(entry) for entry in entries]


def _collect_target_selector(target: Any) -> str:
    if isinstance(target, (RLMSpawnHandle, RLMSubagent)):
        return target.rlm_child_id
    if isinstance(target, str) and target.strip():
        return target.strip()
    raise TypeError("collect target must be a spawn handle, subagent row, or non-empty name/id")


def _child_result_from_payload(payload: Any) -> RLMChildResult:
    if not isinstance(payload, dict):
        raise RuntimeError("rlm.collect returned an invalid result entry")
    child_id = payload.get("rlm_child_id")
    status = payload.get("status")
    settled = payload.get("settled")
    if not isinstance(child_id, str) or not child_id:
        raise RuntimeError("rlm.collect entry is missing rlm_child_id")
    if not isinstance(status, str) or status not in {"queued", "running", "done", "error", "cancelled"}:
        raise RuntimeError("rlm.collect entry has invalid status")
    if not isinstance(settled, bool):
        raise RuntimeError("rlm.collect entry has invalid settled flag")

    def optional_string(field: str) -> str | None:
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            raise RuntimeError(f"rlm.collect entry has invalid {field}")
        return value

    def optional_number(field: str) -> float | None:
        value = payload.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise RuntimeError(f"rlm.collect entry has invalid {field}")
        return value

    replied = payload.get("replied_since_task")
    if replied is not None and not isinstance(replied, bool):
        raise RuntimeError("rlm.collect entry has invalid replied_since_task")
    directory = optional_string("session_dir")
    return RLMChildResult(child_id, optional_string("session_name"), Path(directory) if directory else None,
                          status, settled, optional_string("answer_preview"), optional_string("error"),
                          optional_number("duration_ms"), optional_number("tool_use_count"), replied)


async def collect(targets: Any = None, *, timeout_ms: int = 0) -> list[RLMChildResult]:
    """Read bounded result previews for direct children without steering the parent.

    Omitted targets select all direct children. A positive timeout waits for
    settlement up to its deadline; pending children are returned on timeout.
    Existing automatic parent-result delivery remains unchanged.
    """
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or not 0 <= timeout_ms <= 2_147_483_647:
        raise TypeError("timeout_ms must be a non-negative int up to 2147483647")
    if targets is None:
        selectors = []
    elif isinstance(targets, (RLMSpawnHandle, RLMSubagent, str)):
        selectors = [_collect_target_selector(targets)]
    elif isinstance(targets, (list, tuple)):
        selectors = [_collect_target_selector(target) for target in targets]
    else:
        raise TypeError("targets must be a target or list of targets")
    payload = await host_request("rlm.collect", {"targets": selectors, "timeout_ms": timeout_ms})
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("rlm.collect returned an invalid results list")
    return [_child_result_from_payload(result) for result in results]


async def delete_subagent(target: str | RLMSubagent) -> RLMSubagent:
    """Delete one running or retained direct child from the current parent session."""
    if isinstance(target, RLMSubagent):
        selector = target.rlm_child_id
    elif isinstance(target, str):
        selector = target.strip()
        if not selector:
            raise ValueError("target must not be empty")
    else:
        raise TypeError(f"target must be str or RLMSubagent, got {type(target).__name__}")
    payload = await host_request("rlm.delete_subagent", {"target": selector})
    return _subagent_from_payload(payload.get("subagent"), "rlm.delete_subagent")


from .lifecycle import UnsupportedCapability, lifecycle_capabilities, stop_subagent, active_execution, resume_subagent


class _HarnessProxy:
    """Resolve the harness state against the current environment on every access.

    Session env vars may be applied after import, so a state bound at import
    time could freeze an env-less resolution. Resolution must never raise (a
    failure inside the kernel namespace would take down the kernel). When the
    local store is genuinely unconfigured (no session env, e.g. --no-session)
    reads see an empty view but local writes raise instructively instead of
    vanishing on kernel exit; any other resolution failure degrades to a shared
    in-memory store until local resolution starts succeeding.
    """

    _fallback: HarnessState | None = None
    _unpersisted: HarnessState | None = None

    def _resolve(self) -> HarnessState:
        try:
            return get_harness_state()
        except RuntimeError as exc:
            if "Local harness state requires" in str(exc):
                if _HarnessProxy._unpersisted is None:
                    _HarnessProxy._unpersisted = HarnessState(
                        in_memory=True,
                        local_write_error=(
                            f"{exc} This session has no persistent local harness store; "
                            "pass global_=True to persist across sessions."
                        ),
                    )
                return _HarnessProxy._unpersisted
            return self._degraded()
        except Exception:  # pragma: no cover - harness access must never raise
            return self._degraded()

    @staticmethod
    def _degraded() -> HarnessState:
        if _HarnessProxy._fallback is None:
            _HarnessProxy._fallback = HarnessState(in_memory=True)
        return _HarnessProxy._fallback

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __repr__(self) -> str:
        return repr(self._resolve())


_harness_state = _HarnessProxy()


class _RLMCallable:
    harness = _harness_state
    get_harness_state = staticmethod(get_harness_state)

    async def run(self, prompt: str, **kwargs: Any) -> RLMSpawnHandle:
        return await run(prompt, **kwargs)

    async def create_session(
        self,
        prompt: str,
        name: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
        cwd: str | None = None,
    ) -> RLMCreateSessionHandle:
        return await create_session(prompt, name=name, model=model, thinking=thinking, cwd=cwd)

    async def find_models(self, query: str = "", limit: int = 8) -> list[RLMModel]:
        return await find_models(query, limit)

    async def list_subagents(self) -> list[RLMSubagent]:
        return await list_subagents()

    async def collect(self, targets: Any = None, *, timeout_ms: int = 0) -> list[RLMChildResult]:
        return await collect(targets, timeout_ms=timeout_ms)

    async def lifecycle_capabilities(self, *, model: str | None = None, target: Any = None) -> dict[str, Any]:
        return await lifecycle_capabilities(model=model, target=target)

    async def active_execution(self, target: Any) -> dict[str, Any]:
        return await active_execution(target)

    async def stop_subagent(self, target: Any, *, timeout_ms: int = 0) -> dict[str, Any]:
        return await stop_subagent(target, timeout_ms=timeout_ms)

    async def resume_subagent(self, target: Any, *, stop_generation: str, prompt: str) -> dict[str, Any]:
        return await resume_subagent(target, stop_generation=stop_generation, prompt=prompt)

    @property
    def execution(self) -> Any:
        from . import execution
        return execution

    async def delete_subagent(self, target: str | RLMSubagent) -> RLMSubagent:
        return await delete_subagent(target)

    async def __call__(self, prompt: str, **kwargs: Any) -> RLMSpawnHandle:
        return await run(prompt, **kwargs)


rlm = _RLMCallable()
harness = _harness_state


class _CallableModule(types.ModuleType):
    async def __call__(self, prompt: str, **kwargs: Any) -> RLMSpawnHandle:
        return await run(prompt, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = [
    "BashHandle",
    "BashResult",
    "HarnessEntry",
    "HarnessScope",
    "HarnessState",
    "McpIntegration",
    "McpToolError",
    "NotEnabled",
    "RLMCreateSessionHandle",
    "RLMModel",
    "RLMSpawnHandle",
    "RLMSubagent",
    "RLMChildResult",
    "collect",
    "create_session",
    "RefinementEvent",
    "bash",
    "delete_subagent",
    "lifecycle_capabilities",
    "stop_subagent",
    "active_execution",
    "resume_subagent",
    "execution",
    "UnsupportedCapability",
    "emit",
    "find_models",
    "get_harness_state",
    "harness",
    "host_request",
    "list_subagents",
    "rlm",
    "run",
]

# Lazily re-export the MCP base class. Kept lazy so `import rlm` never requires
# the optional `mcp` SDK — only integration packages that subclass it do.
_LAZY_MCP = {"McpIntegration", "McpToolError", "NotEnabled"}


def __getattr__(name: str) -> Any:  # noqa: D401 - module-level lazy attr hook
    if name == "execution":
        from importlib import import_module
        return import_module(".execution", __name__)
    if name in _LAZY_MCP:
        from . import mcp_base

        return getattr(mcp_base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
