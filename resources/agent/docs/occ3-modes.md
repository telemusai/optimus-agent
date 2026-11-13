# OCC3.1 TextMAS and OCC3.2 latent communication

These are separate features. They can be enabled independently. Neither requires
changing OCC-1 or OCC-2 message encoding.

## Commands

```text
/occ
/occ status
/occ 3 off
/occ 3 sequential
/occ 3 hierarchical
/occ 3.1 sequential
/occ 3.2 on
/occ 3.2 off
```

`3` and `3.1` select TextMAS. `3.2` selects latent communication. Bare selectors
show controls or status without changing a setting. `hierarchal` and `hiearical`
are accepted spelling aliases; stored modes and status always say `hierarchical`.

**Migration from older versions:** `/occ 3 off` now disables TextMAS only.
To disable latent communication, use `/occ 3.2 off`. `/occ 3 on` shows migration
help, not an implicit mode choice. The existing `/occ occ3 on|off` argument alias
continues to mean latent communication. `/occ3` is not a slash command.

## OCC3.1: a text workflow on the selected model

TextMAS is off by default. When enabled, a root chat's genuine user turn starts
three tool-free model calls before the normal parent agent answers or acts:

1. **Planner** proposes an approach.
2. **Critic** examines risks, assumptions and correctness.
3. **Refiner** improves the approach.
4. The normal parent agent acts as **Judger/executor**, using the advisory drafts
   and the original request. Its ordinary tools and approval rules remain intact.

In **sequential** mode, each preparation stage receives the earlier drafts.
In **hierarchical** mode, the three preparation stages independently receive the
same original context, without each other's drafts. The final parent receives
all three. The independent stages currently run serially; hierarchical describes
the flow of information, not a promise of parallel execution.

The preparation stages are native-host model calls, not retained subagent
sessions. They cannot invoke tools, edit files or start more agents. Their text
is untrusted advisory context, not system authority, verified evidence or proof
that an action happened. Hidden reasoning is not passed between stages.
The generated system prompt explains this separation and tells the parent not
to recreate the preparation stages by spawning extra agents. This guidance also
applies with a custom system prompt; it does not claim either mode is enabled.

The workflow uses the selected model's ordinary provider interface. It does not
need an OCC3.2 connector, a special latent server or a particular model. It still
needs a working model provider and its normal authentication. It does not create
an alternative endpoint, change model configuration or silently switch models.

### Scope, cost and failure handling

- Contexts that fit use three extra preparation calls before the normal parent
  request. An actual oversized or model-incompatible role request activates shared
  context-map and reduction calls instead. These auxiliary calls run once per
  preparation run, are reused by all three roles, and are not repeated by normal
  parent retries. Normal tool continuations can add further calls. This is not a
  free or token-saving mode.
- The saved mode is per chat. Resume restores it. Preference lookup can report
  inherited values, but automatic preparation is restricted to genuine root-chat
  user turns. Child sessions and internal, heartbeat or agent-message turns do
  not recursively start more TextMAS stages.
- Input persistence precedes preparation. Normal cancellation also cancels
  preparation. A failed or invalid preparation result stops the turn instead of
  silently falling back to an ordinary execution.
- Each actual mapper, reducer, Planner, Critic and Refiner request uses at most
  256 KiB of serialized context after preparation, including images and any
  earlier role drafts in sequential mode. Every request must also fit the selected
  model's estimated input/output limits. Hidden Thinking is stripped before these
  checks. The raw original transcript and normal parent/Judger request are not
  subject to this preparation cap.
- Oversized source content is partitioned at UTF-8 boundaries with stable source
  labels. Tool results, tool calls and images are included in the source
  projection. Up to 64 mapper calls run with bounded concurrency; ordered reducer
  levels must make progress until one brief remains. Source splitting is lossless,
  but model-generated summaries are necessarily lossy advisory context. A brief
  that cannot be reduced within the bounds fails visibly instead of being tail
  truncated. Complete parent history remains authoritative. OCC3.1 does not
  enlarge the model's ordinary parent context window or bypass its provider
  limits; existing parent compaction and failure handling still apply.
- When the ordinary system prompt leaves insufficient room for worst-case role
  drafts and response capacity, its complete UTF-8 text is included in the bounded
  source map instead of being copied unchanged into every auxiliary request. The
  tool-free role contract remains in force, and its summaries are lossy advisory
  data. This is not system-prompt truncation: the normal parent retains the complete
  original system prompt and history. If even the generic role contract cannot fit,
  preparation still fails visibly before auxiliary provider spending.
- Images that cannot fit a mapper request may be decoded and resized only in the
  preparation projection. An undecodable oversized image fails visibly. Original
  parent images are never resized, replaced or removed by OCC3.1. The exact latest
  user message is also supplied directly to reduced roles when it fits; otherwise
  it is represented in the brief and the parent still receives the exact message.
- Each mapper, reducer or role stage has a 120-second limit, a 128 KiB response
  limit and a 16 KiB public-draft limit. Its total output budget is at most 8192
  tokens, including private reasoning and public draft output, and is reduced to
  the selected model's output and estimated remaining-context limits. This reserve
  does not enlarge the ordinary parent's context or increase accepted draft size.
  Usage has a stable call ordinal for each mapper/reducer call. Exhausted reduction
  caps, irreducible content, invalid images and cancelled calls are not silently
  discarded or treated as free.
- Tool-call output, refusal, missing usable text and incomplete output are not
  valid advisory drafts. A normalized stop reason is included in the visible
  failure without echoing partial drafts or untrusted provider error details. No
  automatic role retries or model fallback are added.
- `/stats` shows a separate **OCC3.1 preparation overhead** category. Its known
  token and estimated-cost subtotals do not include missing reports. Unknown
  usage and cost call counts remain explicit, including interrupted calls.
  These calls are not invented ordinary replies or extra provider identities.
  The ordinary model charts exclude this separately listed overhead.
- The workflow is an Optimus product adaptation. It is not the frozen TextMAS
  benchmark harness. Historical benchmark scores, prompts and four-tool-free-call
  comparisons do not establish the accuracy or cost of this tool-capable workflow.

## OCC3.2: the existing native latent path

OCC3.2 is the display name for the existing experimental OCC-3 native transport.
The APIs and configuration keys keep their existing names: `occ3_handoff`, the
model's `occ3` block, `agentCommunication.occ3Enabled` and `occ3Experimental`.
Enabling TextMAS does not enable or configure this transport.

Both agents still need explicitly supported models, matching verified
model/runtime identities and a successful connector handshake. The first native
handoff can return `capture_armed` without sending anything. Admission is not
completed use. Preserve the existing capture, consent, queue and receipt rules
in the [agent-message guide](../skills/agent-message/SKILL.md) and
[sender queue guide](occ3-queue.md).

## Reading the lights

The bottom status row shows delivery lights only: `TXT`, `OCC1`, `OCC2`,
`OCC3.1` and `OCC3.2`. Each fills and brightens while its channel is in use.
The **OCC3.1** light pulses only while a TextMAS run is actively executing on
the selected model; configuring a mode alone never lights it. The **OCC3.2**
light blinks only while native latent transport is actively delivering, and it
requires a validated completed-use receipt. A model's text claiming a handoff,
a selected mode, a visible tool or a queued request cannot produce that proof.
Mode configuration (OCC3.1 off/sequential/hierarchical, OCC3.2 on/off) is not
shown in the status row; use `/occ` or `/occ 3.2` to inspect it.

Missing metadata on an older host means unavailable, not enabled.

## Host compatibility

TextMAS controls use protocol 7, schema 47 and the negotiated `occ31_textmas`
capability. The host owns `occ31_get_preferences`, `occ31_set_mode` and the
independent latent setter `occ32_set_enabled`. TextMAS has no local-write fallback
when a remote host lacks support. Legacy latent controls can retain their old
local-write path only with an explicit unverified warning; that is not proof of
remote host state. The optional `occ3Modes` state describes configuration only.
Existing latent wire identities and completed-use validation are unchanged.

An older daemon can still attach and run ordinary chats. Update the host before
using TextMAS controls. These controls do not modify global `settings.json`,
`models.json`, provider credentials or a different chat's preferences.
