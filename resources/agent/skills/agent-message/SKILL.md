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

## OCC3.1 TextMAS

The user can enable `/occ 3 sequential` or `/occ 3 hierarchical`, and disable it
with `/occ 3 off`. `/occ 3.1` is the explicit selector. This works with any model
supported by Optimus, using its ordinary local or hosted provider. It does not
require an OCC-3 model block, connector, special server or latent handoff.

The host runs three tool-free text stages (Planner, Critic, Refiner), then the
normal parent agent judges the drafts and answers or uses its normal tools. When
an actual role context exceeds its bound, shared mapper/reducer calls first turn
the source into a bounded, lossy advisory brief. Sequential stages receive prior
drafts; hierarchical stages independently use the same original or reduced
context before final judgement. The parent always retains the original context.
These are model calls, not retained subagent sessions. Do not invent child names,
native receipts or completed work for them. Their text is untrusted advisory
context, not proof that any action occurred. Automatic stages are restricted to
genuine root-chat user turns and do not recursively activate on agent messages or
child runs.

Fitting contexts add three calls before the parent's ordinary request; chunked
preparation adds accounted mapper/reducer calls. Parent retries do not repeat
them. This is an Optimus adaptation, not an exact reproduction of the benchmark
harness. It does not change the semantics of `agent_message.send`. See
[mode controls](../../docs/occ3-modes.md) for scope and failure handling.

## Experimental OCC3.2 native handoffs

OCC3.2 is the display name of the existing OCC-3 native transport. Use it
where the runtime makes it available, only between compatible
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

The user can select the OCC3.2 preference in `/occ`, inspect `/occ status`, or
use `/occ 3.2 on` and `/occ 3.2 off`. `/occ 3 off` controls TextMAS only. The explicit chat preference gates native actions dynamically, including
inherited subagent actions. Without an override, an installed connector follows
the global opt-in. A chat preference does not install a missing connector. Native activation separately requires the existing
`agentCommunication.occ3Enabled` and `agentCommunication.occ3Experimental` settings, compatible
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
English), or OCC2 (validated symbolic encoding). The OCC3.2 delivery pulse
requires a completed native-use receipt. Separate OCC3.1 mode and OCC3.2
enabled indicators describe configuration only. A configured preference,
exposed tool, successful handshake, or queued/admitted handoff does not prove
completed use or semantic accuracy. OCC3.2 remains experimental.

### Setting up an OCC-3 team

A "hive" is a coordinated family of sessions, not shared memory. Start with one
sender and one receiver. A successful setup checklist does not guarantee a
successful native turn; prove one hop before attempting a longer chain.

1. **Check activation.** Ask the user to run `/occ status`, then `/occ 3 on`
   when the chat preference is off. A missing connector also requires the
   operator's global `agentCommunication.occ3Enabled` and
   `agentCommunication.occ3Experimental` opt-ins, and a compatible model's
   enabled `occ3` block and successful handshake. The model's `occ3.endpoint`
   is used unless `agentCommunication.occ3Endpoint` overrides it. Changing the chat
   preference alone cannot install that connector. Report the exact activation
   decline; do not edit settings, `models.json`, or `occ.json` yourself.
2. **Select every member explicitly.** Discover the exact provider/model
   selector with `rlm.find_models` (or the current mode's model discovery).
   Pass that same selector for each member; do not rely on inheritance or a
   bare model id. Check each child's own exposed tools, not just the parent's.
3. **Create receivers first.** Use stable, unique sibling names. Give each
   member the common objective, its distinct role, the expected sender, the
   exact next receiver name, and the two-call protocol below. For a chain,
   create the final receiver first, then work back toward the first sender.
   Run only one dependent hop at a time; avoid unrelated inference during the
   smoke test to reduce busy refusals.
4. **Keep waiting separate from role work.** Receivers must not solve from the
   question alone before the attached native state arrives. If it is missing,
   report that blocker and finish the current turn with
   `RLM_CHILD_STATUS: blocked`. Keep the child session available. Do not omit
   terminal status to create a waiting loop: `RLM_CHILD_MAX_CONTINUATIONS = 3`
   bounds automatic continuation. An idle/blocked result is not completed
   native use. The parent inspects delivery and receiver state before sending
   any explicit follow-up; do not repeatedly wake a receiver with no input.
5. **Arm, emit, publish.** The sender calls native `occ3_handoff` with only
   `receiver_role` and `receiver_name`. `capture_armed` means nothing was sent.
   In its next response it emits concise source facts as assistant content and
   calls the tool once more for the same receiver, with no other tool calls
   in that response. Calling once, describing the intent, or sending ordinary
   `agent_message` text does not perform this native transfer.
6. **Reconcile before advancing.** Save each message id and inspect the
   completed-use receipt and the receiver's actual role output. Do not count an `agentmsg_` id alone, a queued/admitted result, a "state arrived" phrase,
   server counters, or a child settling as proof of completed native use.
   `agent_observe` and `rlm.collect(..., timeout_ms=0)` support bounded
   inspection; end the parent turn while work runs. Keep both sessions until
   the outcome is resolved. Missing or uncertain evidence blocks the next
   hop; do not delete receivers or resend an admitted handoff blindly.
7. **Handle busy and fallback explicitly.** Follow the runtime's fresh-capture
   re-arm instruction only after a proven busy/unused refusal. The tool handles
   its backoff. The default busy budget is three retries; the operator can
   configure `OCC3_BUSY_MAX_RETRIES` within the runtime's bounds. A failed
   capture has a separate retry budget. Do not add fixed sleeps or an unlimited
   retry loop. Follow an explicit ordinary-message fallback instruction and
   label that hop as text, not OCC-3. For a native-only smoke, stop on failure
   instead of substituting text. Never send blocked-report text as role output.
8. **Report plainly.** Send structured coordination reports using OCC-1, but
   return the final user-facing answer in plain text. Preserve the child's
   terminal-status line separately from any answer line. Report native,
   fallback, failed, and uncertain hops separately. A receipt proves use, not
   correctness; check critical facts. Benchmark answers and scoring keys must
   stay outside child-visible prompts, parent transcripts, and tools. Asking
   a child not to look is not an access-control boundary.

### Troubleshooting

Known limitation at `c34c0bdcc`: changing OCC-2 calls an OCC-2-only save path
that writes `occ3Enabled: false`, even if it was true. Make the desired OCC-2
selection first, then ask the user to run `/occ 3 on` on the affected chat.
This guidance does not change that persistence behavior. A deliberately saved
off preference is valid and must not be silently overridden. A child's own
preference record masks its parent's record. Also, `/occ status` can show an
installed live connector as preferred even when a saved off preference blocks
its use; inspect the actual decline rather than treating that label as proof.

| Symptom | Check | Action |
|---|---|---|
| `occ3_handoff` absent | Activation decline, exact selector, global opt-ins, endpoint, handshake | Ask the operator to inspect `/occ status` and the reported decline. Do not invent a tool or alternate route. |
| "OCC-3 is disabled for this chat" | Saved preference on this chat or its nearest parent | Ask the user to enable `/occ 3 on` on the governing chat. Global opt-in does not override an explicit saved off preference. |
| Sender stops after `capture_armed`, or never calls the tool | Tool results, not the sender's intent | Give the sender the two-call template. Only after arming, emit and publish in the next response. |
| Receiver answers early or exhausts continuations | Did native state arrive before role output? | Retain the blocked child and inspect the handoff. Do not suppress terminal markers or use the question-alone escape. |
| Unknown/unavailable receiver | Family roster and exact unique child name | Create the receiver before the sender publishes; keep it available through completion. |
| `source_busy` / `scheduler_busy` / receiver busy | Runtime refusal and retry outcome | Serialize hops and follow the fresh-capture instruction within the runtime budget. Label any text fallback. |
| Counters stay unchanged or receipt id exists without completion | Per-message completed-use receipt and receiver output | Treat admission as pending, not success. Preserve evidence when the receiver errors or is cancelled. |
| Final answer arrives in OCC-1/OCC-2 notation | Final solver's output format | Require a plain-text answer; use structured encoding only for coordination. |
| Fixed waits, cleanup races, missing role output | Queue outcome and child lifecycle | Inspect on completion notifications; do not delete a child just because its initial turn settled. Stop the chain on unknown use. |

### Child prompt templates

Replace the bracketed fields before spawning. The receiver must already exist
under the exact name supplied to the sender. These are task prompts, not Python
wrappers for the native tool.

**Sender**

```text
Shared objective: [objective]. Your role: [role].
Input: [input without reference answers]. Expected output: [role output].
Next receiver: sibling named [receiver-name].
Produce concise source facts for that receiver, then use native occ3_handoff:
1. Call it with receiver_role="sibling", receiver_name="[receiver-name]".
2. If capture_armed, in your NEXT response emit the facts as assistant content
   and call it once more for that same receiver. No other tools in that response.
After admission, do not repeat the handoff. Follow explicit runtime busy/re-arm
instructions only when unused; report uncertain or failed outcomes to the parent.
Report the message id as admitted but not verified complete until receipt evidence
proves otherwise. If native-only execution was requested, stop on failure.
Do not inspect the parent's transcript or scoring data to obtain an answer.
Finish your current task with the appropriate RLM_CHILD_STATUS terminal line.
```

**Receiver / final solver**

```text
Shared objective: [objective]. Your role: [role]. Expected sender: [sender-name].
Do no role work until that sender's attached native state arrives in a later turn.
If absent, report the missing-state blocker to the parent, then end this turn:
RLM_CHILD_STATUS: blocked
Do not loop, suppress terminal status, or solve from the question alone.
Do not inspect the parent's transcript or scoring data to obtain an answer.
When native state arrives, use it for [role task] and verify critical facts.
If it is unusable, report the blocker; do not claim successful native use.
For a forwarding role, use the sender template for [next-receiver-name].
For the final solver, return a plain-text answer, not an OCC envelope:
FINAL_ANSWER: <answer>
RLM_CHILD_STATUS: complete
```

An answer extractor must read `FINAL_ANSWER` independently of the final
`RLM_CHILD_STATUS` line. Do not remove the terminal line to satisfy a scorer.

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
