"""Prime Agent session-to-session messaging skill.

All routing and sender identity live in the TypeScript daemon. These functions
only call the host bridge exposed inside the Prime Agent kernel.
"""

from __future__ import annotations

from typing import Any, Literal

from rlm import host_request

ReceiverRole = Literal["parent", "sibling", "child"]
_MESSAGE_DISPLAY_MIME = "application/vnd.prime-agent.agent-message+json"
ACTIVE_ONLY_CONTRACT = "rlm.active-only-message.v1"


async def list_agents() -> dict[str, Any]:
    """List this agent's parent, siblings, and children, including inactive family."""
    return await host_request("agent_message.list_agents")


async def send(
    message: str,
    broadcast_message: str | None = None,
    *,
    receiver_role: ReceiverRole | str | None = None,
    receiver_name: str | None = None,
    wake_if_idle: bool = True,
    execution_generation: str | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Send one direct role-addressed message or broadcast to ``"all"``."""
    if type(wake_if_idle) is not bool:
        raise TypeError("wake_if_idle must be bool")
    if not wake_if_idle:
        if broadcast_message is not None or receiver_role != "child" or not receiver_name:
            raise ValueError("active-only delivery requires one owned direct child")
        if not isinstance(message, str) or not message or len(message.encode("utf-8")) > 32000:
            raise ValueError("active-only message must be a nonempty bounded string")
        if not isinstance(execution_generation, str) or not execution_generation or len(execution_generation) > 128:
            raise ValueError("execution_generation is required")
        if not isinstance(message_id, str) or not message_id or len(message_id) > 256:
            raise ValueError("message_id is required")
        from rlm.lifecycle import lifecycle_capabilities, UnsupportedCapability
        capability = await lifecycle_capabilities(target=receiver_name)
        feature = capability.get("activeOnlyMessages", {})
        if (feature.get("capability") != ACTIVE_ONLY_CONTRACT
                or feature.get("supported") is not True):
            raise UnsupportedCapability("Host does not support active-only messages")
        receipt = await host_request("rlm.send_active_message", {
            "target": receiver_name, "message": message,
            "execution_generation": execution_generation, "message_id": message_id,
        })
        if (not isinstance(receipt, dict) or receipt.get("schema") != "optimus.active-message.v1"
                or receipt.get("wakeIfIdle") is not False
                or receipt.get("executionGeneration") != execution_generation
                or receipt.get("messageId") != message_id
                or receipt.get("accepted") is not (receipt.get("deliveryStatus") in ("accepted", "duplicate"))):
            raise UnsupportedCapability("Invalid active-only message receipt")
        return receipt
    if execution_generation is not None or message_id is not None:
        raise ValueError("execution_generation/message_id require wake_if_idle=False")
    roles = ("parent", "sibling", "child")
    if broadcast_message is not None:
        if message != "all":
            raise TypeError(
                "positional agent_message.send targets are not supported; "
                "use receiver_role and receiver_name"
            )
        if receiver_role is not None or receiver_name is not None:
            raise TypeError("broadcast cannot be combined with receiver_role/receiver_name")
        payload: dict[str, Any] = {
            "target": "all",
            "message": broadcast_message,
        }
    else:
        if receiver_role not in roles:
            raise ValueError('receiver_role must be "parent", "sibling", or "child"')
        if not isinstance(message, str):
            raise TypeError(f"message must be str, got {type(message).__name__}")
        if receiver_role == "parent":
            if receiver_name is not None:
                raise ValueError("receiver_name must be omitted for parent messages")
        elif not isinstance(receiver_name, str) or not receiver_name.strip():
            raise ValueError("receiver_name is required for sibling and child messages")
        payload = {
            "message": message,
            "receiver_role": receiver_role,
            "receiver_name": receiver_name,
        }
    receipt = await host_request("agent_message.send", payload)
    receipts = receipt.get("receipts") if isinstance(receipt, dict) else None
    if isinstance(receipts, list):
        for item in receipts:
            if isinstance(item, dict) and "deliveryStatus" in item:
                _emit_sent_message(item)
    else:
        _emit_sent_message(receipt, receiver_role)
    return receipt


def _emit_sent_message(receipt: dict[str, Any], receiver_role: str | None = None) -> None:
    try:
        from rlm import emit

        label = (
            "Agent message queued"
            if receipt.get("deliveryStatus") == "queued"
            else "Agent message sent"
        )
        display_receipt = dict(receipt)
        if receiver_role in ("parent", "sibling", "child"):
            display_receipt["receiverRole"] = receiver_role
        emit(
            {
                _MESSAGE_DISPLAY_MIME: display_receipt,
                "text/plain": label,
            }
        )
    except Exception:
        pass
