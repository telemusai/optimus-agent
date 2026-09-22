"""Read-only Prime Agent session observation skill.

All session lookup and data access live in the TypeScript daemon. These
functions only call the host bridge exposed inside the Prime Agent Python
kernel.
"""

from __future__ import annotations

from typing import Any

from rlm import host_request


def _aliases(result: dict[str, Any]) -> dict[str, Any]:
    """Add local compatibility fields without changing the daemon schema."""
    result = dict(result)
    for key in ("current", "agent"):
        if isinstance(result.get(key), dict):
            result[key] = _summary_aliases(result[key])
    if isinstance(result.get("agents"), list):
        result["agents"] = [_summary_aliases(row) for row in result["agents"]]
    if isinstance(result.get("messages"), list):
        result["messages"] = [
            {**row, "content": row["text"]} if isinstance(row, dict) and "text" in row else row
            for row in result["messages"]
        ]
    return result


def _summary_aliases(summary: Any) -> Any:
    if not isinstance(summary, dict):
        return summary
    return {
        **summary,
        **({"name": summary["sessionName"]} if "sessionName" in summary else {}),
        **({"activityStatus": "attached_idle"} if summary.get("status") == "user" else {}),
    }


async def list_agents() -> dict[str, Any]:
    """List active daemon sessions visible to this agent."""
    return _aliases(await host_request("agent_observe.list"))


async def get_agent(target: str) -> dict[str, Any]:
    """Read one active session summary by active id, session id/name, or suffix."""
    if not isinstance(target, str):
        raise TypeError(f"target must be str, got {type(target).__name__}")
    return _aliases(await host_request("agent_observe.get", {"target": target}))


async def recent_messages(
    target: str,
    limit: int = 8,
    max_chars: int = 800,
) -> dict[str, Any]:
    """Read bounded recent message previews from an active session.

    Args:
        target: Active session id, session id/name, or unambiguous suffix.
        limit: Number of recent messages to return. Host validates 1-50.
        max_chars: Per-message preview size. Host validates 80-2000.
    """
    if not isinstance(target, str):
        raise TypeError(f"target must be str, got {type(target).__name__}")
    if not isinstance(limit, int):
        raise TypeError(f"limit must be int, got {type(limit).__name__}")
    if not isinstance(max_chars, int):
        raise TypeError(f"max_chars must be int, got {type(max_chars).__name__}")
    return _aliases(await host_request(
        "agent_observe.recent",
        {
            "target": target,
            "limit": limit,
            "max_chars": max_chars,
        },
    ))
