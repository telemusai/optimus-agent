# OCC-3 Retained-Capture Lifecycle — Design (PR #215)

## Why the old same-turn requirement existed

The legacy source-completion contract required the model to produce, in ONE turn, a
content run AND exactly one `occ3_handoff` tool call:

- client completed hook: `bindings.len() == 1 && bindings[0].tool_name == "occ3_handoff"`
- sglang publish gate: any other tool name or binding count is refused
- server gate: `publication.tool_call_id == output_binding["tool_call_id"]` and the
  bound tool name must be `occ3_handoff`

The same-turn tool call was the publication authority: the server admitted a
publication only against the turn's own tool binding, which is what bound the captured
content to an authenticated dispatch.

Measured failure chain (p8i-p8k): GLM 5.3 reasons before emitting content; small
budgets end with `finish_reason=length` and empty content. The r13 two-turn workaround
(unarmed content turn, armed repeat+tool turn) still demanded the same-turn
content+tool shape in its armed turn, so the failure mode survived.

## The repair: single-turn retained content-only mode

Arm BEFORE the content request, do not install the handoff tool for that turn, and
publish deterministically afterwards through the authenticated family controller.
No model cooperation is required at publication time.

## State machine

Happy path:

    Idle -> CaptureArmed -> Capturing -> CapturedRetained -> PublicationAuthorised
         -> Published -> ReceiverAdmitted -> Consumed -> Retired

- Idle -> CaptureArmed: `arm_source_retained` before the content request; the intent
  carries `occ3.capture.v3`, `capture_mode=content_only_orchestrator_v1`, span policy
  `content_tail_no_tool_v2`.
- CaptureArmed -> Capturing: the single content request runs; the handoff tool is not
  installed on that turn.
- Capturing -> CapturedRetained: the completed hook accepts ZERO tool bindings; the
  retained lease stays bound to its original capture-turn identity (intent digest,
  turn nonce, output sha).
- CapturedRetained -> PublicationAuthorised: the orchestrator authorizes through the
  authenticated family controller (authorize_native_handoff).
- PublicationAuthorised -> Published: `publish_retained` sends `occ3.publish.v2` with
  `dispatch_mode=orchestrator_v1`; one-shot claim (`publication_claimed`).
- Published -> ReceiverAdmitted: receiver-side admission gate.
- ReceiverAdmitted -> Consumed: the receiver model consumes the retained state;
  completed-use receipt.
- Consumed -> Retired: state retirement and bounded resource cleanup.

Failure outcomes (each backed by a named test in `occ3_retained_lifecycle.rs`):

| Failure outcome | Trigger | Backing test |
|---|---|---|
| tool forbidden, tombstone | any tool call in the captured turn | `retained_capture_rejects_unexpected_tool_output` |
| unarmed-then-armed claim refused | an armed later turn cannot claim earlier unarmed content | `unarmed_content_then_armed_tool_only_cannot_claim_earlier_content` |
| mode crossing refused | retained capture cannot publish via legacy model-tool dispatch, and vice versa | `mode_crossing_refused` |
| republish refused | one-shot claim; an earlier capture cannot be republished | `earlier_capture_cannot_be_republished` |
| wrong recipient refused | publication to the wrong recipient refused BEFORE publish | `publish_retained_wrong_recipient_refused` |
| legacy binding count enforced | legacy mode still requires exactly one binding (zero-tool refused there) | `legacy_mode_still_requires_exactly_one_binding` |

## Mode-scoped security contract

- Retained mode: zero tool bindings in the captured turn; ANY tool call fails the mode.
  Span policy must be `content_tail_no_tool_v2`; server rules refuse tool syntax in the
  window and require a content run.
- Publication authority is unchanged: only the authenticated family controller
  (authorize + send_prepared), now with orchestrator dispatch. The server validates the
  v2 publication against the EMPTY output binding plus the negotiated capture_mode
  and span policy.
- One-shot: after `publication_claimed`, republish is refused.
- Recipient: wrong recipient is refused before the publish step.
- Lineage: the retained lease keeps its original capture-turn identity; the receipt
  (`occ3.source.v2`) names that lineage.
- Modes cannot cross in either direction; no field set is defaulted onto the other
  mode (digests cover the mode-specific field sets separately).

## Wire schemas

Capture arming (pi-ai `SourceSpanArming` / intent):

| Version | Schema | capture_mode | Span policy |
|---|---|---|---|
| v1 (legacy) | `occ3.capture.v1` | absent | `whole_completed_response_tail_v1` |
| v2 (legacy) | `occ3.capture.v2` | empty (implicit `same_turn_tool_v1`) | v2 family, default `content_tail_v2` |
| v3 (new) | `occ3.capture.v3` | `content_only_orchestrator_v1` | `content_tail_no_tool_v2` |

v1/v2 wire bytes are unchanged: an empty `capture_mode` is never serialized.

Publication (server control API, exact field sets):

- `occ3.publish.v1` (legacy, unchanged): publication_id, source_handle,
  source_session_id, receiver_session_id, message_id, source_turn_nonce, tool_call_id,
  expires_at_epoch.
- `occ3.publish.v2` (new): the same field set with `tool_call_id` REPLACED by
  `dispatch_mode=orchestrator_v1`; the publication parses with the empty tool identity.

Receipts: a v3 capture produces an `occ3.source.v2` receipt (same fields as v2); the
mode lineage is carried by the capture intent, not by a new receipt schema.

Validator admission (`deploy/v3/native_validator.py`): the control interface must be
the 5-tuple (occ3.sessions.enroll.v1, occ3.publish.v1, occ3.publish.v2,
occ3.publication.status.v1, occ3.retire.v1); a 4-tuple is refused with
`control_schema_version_unknown`.

## Coexistence without weakening the legacy mode

- Legacy intents keep the implicit `same_turn_tool_v1` mode and the
  exactly-one-binding rule; zero-tool turns are still refused there.
- Legacy wire bytes are identical before and after this change.
- A retained capture cannot be published through the model-tool dispatch, and a legacy
  capture cannot be published through the orchestrator dispatch: mode crossing is
  refused in both directions.
- The legacy mode's security proof (tool-bound admission) is untouched; the new mode
  substitutes controller authentication plus the empty-binding check for the tool
  binding, and adds the no-tool span policy as a second, independent guard.

## Live validation state (measured, 2026-11-01 evening; digests in LIVE-RESULTS.md)

The retained lifecycle ran live in three controlled-swap windows (canary-1 1330Z,
canary-2 1615Z, canary-3 1745Z). Measured per arm-A evidence:

- The structural chain held end-to-end every time: arm BEFORE the content request,
  zero tool bindings in the captured turn (`source_tool_calls=0`,
  `span_tool_call_count=0`), one-shot claim burned pre-effect, single v2 publication
  (`POST /occ3/v1/native/publish 200`), receiver admission, real receiver request
  (`request_count=1`, `receiver_tool_calls=0`, `source_facts_absent=true`,
  `occ3_keys_absent=true`), `message_status=delivered`, `consumed=true`,
  `lease_state=Consumed`, then retire (`POST /occ3/v1/native/retire 200`).
- The failure never occurred in the structural chain; it sat at invariant G
  (exact-value recovery) in every window, and at invariant C (coverage) until the two
  red-first fixes (77d57c64c span/turn-end classification, 109ab15db tail-window
  coverage criterion) landed. After 109ab15db, canary-3 measured `coverage_complete
  = true`, `value_fully_in_span=true`, `retained_count == span_accepted_width` (35/35).
- The TOKEN_FULL exact-embedding arm (canary-3 window, control token `texact`,
  readback-confirmed) recovered a fresh private value exactly in 84.9 s: the
  publish -> receiver-admission -> consumption path delivers exact information when
  the calibrated map is bypassed. The map, not the lifecycle, is the remaining defect.
- Receiver answers, when wrong, were mutated strings (39/41 and 40/41 chars) with
  mid-string edits inside retained rows — not missing spans — consistent with a
  distribution-shifted calibrated map rather than a capture or publication failure.
- Canary-4 (pooled refit + re-run) is PENDING; the design above is unchanged by the
  measured results.
