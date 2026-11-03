---
name: agent-message
description: Message an agent's parent, siblings, or direct children through the daemon. Use the family roster to discover reachable agents and send direct text without spoofing sender identity.
---

# Agent Message

Send direct messages within the current agent's nuclear family through the
local daemon: parent, siblings, and direct children only. Roots are siblings.
The daemon derives your sender identity from the current session; do not try
to include a `from` field.

Call directly from the kernel:

```python
children = await rlm.list_subagents()
child = next((item for item in children if item.active_session_id), None)
if child is not None:
    receipt = await agent_message.send(
        "Please inspect the latest result.",
        receiver_role="child",
        receiver_name=child.session_name,
    )
    # Keep the child until this follow-up finishes so its result remains observable.
```

## API

- `await agent_message.list_agents()` — returns `current` (`name`, `id`, `depth`)
  and family-scoped `entries` (`relationship`, `name`, `id`, `depth`, `status`)
  for the current agent's parent, siblings, and children. It includes inactive
  family members and sorts parent, siblings by name, then children by name; it
  does not expose a global daemon session list.
- `await agent_message.send(message, receiver_role="parent" | "sibling" | "child", receiver_name=None)` — sends one direct
  text message to an active session. Sending to an idle completed subagent
  starts an ordinary follow-up turn in that same child session and context.
  The child remains available only until its parent session closes. The daemon
  resolves `receiver_role` within the current agent family; `receiver_name` is
  required for siblings and children and omitted for the unique parent.
  `send("all", message)` broadcasts only to the family roster and returns
  `{receipts: [...]}` in roster order; successful entries are ordinary receipts
  and failed entries contain the target id and a short `error`. One failed delivery
  does not reject successful deliveries. Messages always use steering delivery so
  a busy target sees them during its active run. Returns a receipt with a
  `deliveryStatus` field: `"delivered"` means the message reached an idle target's
  context; `"queued"` means a steering message was accepted and will deliver when
  the target's current work allows (`send` does not block waiting for that).
  Delivered receipts carry `deliveredAt`, queued receipts carry `queuedAt`.

## OCC-1 structured messages (default for task and result communication)

For structured communication - task assignments, result reports,
evidence, blocked/cancelled notices - encode the message body as OCC-1
compact structured English. Plain English stays fine for conversational
replies. OCC-1 is a deterministic `KEY=VALUE` line grammar that both sides
can parse losslessly; the kernel-side codec is preinstalled as
`rlm.occ1` (pure stdlib).

```
OCC/1
TASK=t42
KIND=RESULT
STATUS=done
F1=race window on session lock
E1=src:src/session.rs
EL1=218-235
VERIFY=session_race:fail
NEXT=inspect_lock
```

Rules:
- Line 1 is exactly `OCC/1`; every following line is `KEY=VALUE` with
  `KEY = [A-Z][A-Z0-9_]*`; repeated keys (e.g. `NEXT`) keep order.
- Required fields: `TASK`, `KIND` (TASK/PROGRESS/RESULT/EVIDENCE/
  REQUEST_DETAIL/ACK/CORRECTION/BLOCKED/CANCELLED), `STATUS` (pending/
  in_progress/done/partial/blocked/failed/cancelled); `EVIDENCE` also
  needs `E1`.
- Indexed families (`F<n>`, `E<n>`, `ART<n>`, `DEP<n>`) must be
  contiguous from 1; gaps are corruption and must be rejected, never
  partially decoded.
- Reference evidence (`E<n>=<kind>:<locator>` with optional `EL<n>` line
  range, `ED<n>` digest, `ER<n>` revision) instead of pasting file
  contents. `<kind>` must be one of the fixed evidence kinds:
  `src` (repo-relative source path), `commit` (SHA), `log` (test log),
  `build` (build result), `tool` (tool receipt), `artifact` (digest),
  `lines` (source line range), `finding` (prior finding id),
  `file` (any other file path, absolute or workspace-relative),
  `transcript` (session transcript path or id). Any other kind fails
  contract validation and the message is sent as plain OCC-1 without
  the OCC-2 transport wrap.
- Sender identity stays daemon-derived; never put a `SENDER` field in
  outgoing text.

Kernel helpers: `from rlm.occ1 import Occ1Message` -
`Occ1Message.decode(text)` (raises on any malformed input),
`message.encode()`, `message.get/get_all/has`, and
`validate_index_gaps(message)` for the contiguity rule. Malformed OCC-1
must be rejected, never guessed.

## Adaptive OCC-2 transport

OCC-1 remains the authoring format and fallback. OCC-2 is enabled by default
for chats without a saved preference. The user can open `/occ`, inspect
`/occ status`, or use `/occ 2 on` and `/occ 2 off`. Existing explicit off
preferences stay off. The toggle is saved with
that chat and restored on resume. Subagents dynamically inherit their nearest
parent preference unless they have an explicit override. Other chats and
independent forks keep their own settings; the toggle never edits global or
project settings.
Continue authoring valid OCC-1 messages: when enabled, the native transport
negotiates a shared dialect and selects OCC-2 for eligible complete contract
messages using size, economics, codebook availability, and decode-failure guards.
Plain English, OCC-1 deltas, and messages without a compatible peer stay unchanged.
The receiving model gets validated symbolic text with a reusable codebook; invalid
symbolic text uses the validated OCC-1 fallback when available. The durable
record retains both encodings. This mode does not promise lower total token use.

## Experimental OCC-3 native handoffs

Use OCC-3 where the runtime makes it available, only between compatible
OCC-3-supported subagents. Both models must have an enabled `occ3` block in
`models.json` and pass the live handshake with matching exact model/runtime
identity. A configured **DGX MTP (OCC-3)** entry is distinct from an ordinary
DGX MTP entry: a shared display name, provider URL, or model id is not proof of
support. Keep both subagents on the same configured OCC-3 model for these
handoffs; use ordinary messaging for unsupported or mixed pairs.

Discover the configured entry with `await rlm.find_models("DGX MTP (OCC-3)")`
and pass the returned exact `selector` as `model=` when creating both subagents
with `rlm(...)`. For this configured provider the selector may be
`dgx-glm53-occ3/glm-5.3`; use it only if listed. A bare `glm-5.3` can select the
ordinary provider and lose OCC-3 support. Both agents must still enroll and
pass the native identity checks.

Use the selected model's configured inference and OCC-3 endpoints and existing
provider credentials. Do not replace them with remembered SSH tunnels,
known-hosts entries, or gateway routes. An unavailable configured endpoint is
a connectivity failure to report, not permission to invent an alternate route.

The user can select the OCC-3 preference in `/occ`, inspect `/occ status`, or
use `/occ 3 on` and `/occ 3 off`. The explicit chat preference gates native actions dynamically, including
inherited subagent actions. Without an override, an installed connector follows
the global opt-in. A chat preference does not install a missing connector. Native activation separately requires the existing
`agentCommunication.occ3Enabled` and `occ3Experimental` settings, compatible
model configuration, and a successful live connector handshake. Do not change
global settings or invent configuration to make a model eligible.

When the runtime exposes **`occ3_handoff`**, use that native tool for an eligible
handoff of captured source state to an authorized parent, sibling, or child.
The first call on an unarmed session returns `capture_armed`: no message has
been sent. In the next response, emit the concise source facts as assistant
content and call `occ3_handoff` once for that same receiver; include no other
tool calls in that response. This arms bounded source capture before publication.
Follow its schema: supply only `receiver_role` and `receiver_name`. Do not put
source facts in those arguments, fabricate handles, write `OCC/3` envelopes,
or invoke the native tool as a Python `agent_message` function.
`agent_message.send` remains the ordinary text/OCC-1/OCC-2 path; it does not
automatically become a native OCC-3 transfer. If native handoff is unavailable,
continue through that ordinary path. If a handoff has already been admitted or
its result is uncertain, inspect the outcome before retrying or sending a
duplicate text handoff.

The subagent bar flashes for delivered TXT (plain text), OCC1 (structured
English), or OCC2 (validated symbolic encoding). OCC3 requires a completed
native-use receipt. A configured preference, exposed tool, successful
handshake, or queued/admitted handoff does not prove completed use or semantic
accuracy. OCC-3 remains experimental.

## Safety

- Do not delete a child immediately after `send`: delivered follow-ups may still
  be running and queued receipts have not run yet. Wait until observation shows
  the child is idle and its context is no longer needed before calling
  `await rlm.delete_subagent(child)`.
- Reach is limited to parent, siblings, and direct children; relay through an
  intermediate child instead of messaging grandchildren or cousins directly.
- Sender identity is daemon-derived and cannot be spoofed from Python.
- The daemon enforces message size, rate, and pending-queue limits before
  accepting delivery.

## Optional active-only direct-child diagnostics

The default `send` behavior above is unchanged. To prevent an idle or stopped
child from waking, query `await rlm.active_execution(child)` first. Use its
original `execution_generation` with:

```python
await agent_message.send(
    "bounded diagnostic", receiver_role="child", receiver_name=child.rlm_child_id,
    wake_if_idle=False, execution_generation=active["execution_generation"],
    message_id="unique-id",
)
```

The client requires `rlm.active-only-message.v1` and matching native/runtime
provenance. No ordinary-send fallback is allowed. The `optimus.active-message.v1`
receipt is admission only; `accepted`/`duplicate` do not prove model processing.
`declined_idle`, `declined_stopped`, `declined_generation`, and `declined_capacity`
never start or retarget work. Do not replace an expired token with a new one.
Only direct children are allowed. Broadcasts remain ordinary default sends.
See `docs/native-lifecycle.md` for retained stop and explicit audit resume.
