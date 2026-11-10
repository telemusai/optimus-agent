# OCC-3 activation observations and source-verified setup

These notes came from the benchmark driver. Setup details below were corrected
against main `c34c0bdcc`; historical reports of success are not a substitute for
published completed-use receipts. Agents must not edit live configuration to
force eligibility. Ask the operator to apply any needed setting change.

## Symptom
`rlm` subagents spawned on `dgx-glm53-occ3/glm-5.3` reported `occ3_handoff ABSENT`
from their tool list. Calls to the tool (when prompted) returned:
`OCC-3 is disabled for this chat; use /occ 3 on with a supported model`.
Daemon log showed `occ3 activation declined reason=experimental_ack_required`.

## Driver-reported activation failure (2026-11-10)
The driver reported missing opt-in settings and absent native tools. The current
source requires both `agentCommunication.occ3Enabled` and
`agentCommunication.occ3Experimental`. A top-level `occ3Experimental` is not the
setting read by `SettingsManager`. `agentCommunication.occ3Endpoint` is optional
when the model already has `occ3.endpoint`. Earlier notes conflated these keys.
The driver reported activation failures even though:
- models.json `dgx-glm53-occ3` provider block was already `enabled: true`,
  `validation: live-verified`, endpoint `http://127.0.0.1:30001`;
- the sglang server's occ3-bridge plugin reported `NATIVE_READY`
  (`installed_experimental_not_live_validated`) at `/occ3/v1/status`.

## Current setup contract (operator-owned)
The two opt-ins in the effective `settings.json` are:

```json
{
  "agentCommunication": {
    "occ3Enabled": true,
    "occ3Experimental": true
  }
}
```

New session workers read these settings at start. Existing/root sessions keep
their activation state until an activation path runs again; same-model
reselection is not a guaranteed retry. The per-chat preference also gates
native actions dynamically. Ask the user for `/occ status` and `/occ 3 on` on
the governing chat. The current `occ_set_preferences` RPC accepts only the
`occ2Enabled` input; its save path also resets stored `occ3Enabled` to false.
An extra `occ3Enabled` request field is ignored, not equivalent to `/occ 3 on`.
The model must independently pass the live handshake and identity checks;
these settings do not make an incompatible model eligible.

## Verification procedure
1. Spawn a probe pair on the OCC3 model:
   `await rlm(prompt, name=..., model="dgx-glm53-occ3/glm-5.3")`
2. Ask each child to list its tools via agent_message to parent.
3. Activation check only: `occ3_handoff` appears in the tool list, and the daemon log
   (`~/.config/optimus-rust/logs/daemon.sock.<id>.log`) shows
   `occ3 experimental activation enabled session=<child id>`.
4. Full handoff protocol (per SKILL guidance): first call arms capture
   (`capture_armed`, nothing sent); then the sender emits concise source facts
   as assistant content and calls `occ3_handoff` once more for the same
   receiver, with no other tool calls in that response. A completed native-use
   receipt proves delivery; queued/admitted alone does not.

## Driver-reported tool exposure (2026-11-10 ~07:53 UTC)
- occ3-verify-a / occ3-verify-b: both report `occ3_handoff` in tool list.
- Earlier failures (probe-a/b, occprobe-a/b, latent-sender/receiver) were all
  pre-fix; their handoff attempts returned `OCC-3 is disabled for this chat`.

## Constraints (operator directives)
- Model traffic stays on loopback `127.0.0.1:30001` only.
- Never restart/reconfigure/release the GLM-5.3 sglang server.
- Do not send raw requests to the daemon socket; do not restart the daemon.
- Count native use only with a matched completed-use receipt, not an admission
  id alone; log missing evidence and items where the tool was not called.

## SECOND ROOT CAUSE (2026-11-10 ~07:58) — per-chat preference file
Even with global settings fixed, handoff CALLS still returned
`OCC-3 is disabled for this chat`. Cause: the root session's saved chat
preference file
`~/.config/optimus-rust/session-artifacts/<root-session>/occ.json`
contained `{"occ2Enabled": true, "occ3Enabled": false}` (created by an earlier
OCC-2-only `occ_set_preferences` RPC request, whose save path also wrote
`occ3Enabled: false`). Subagents inherit
the nearest parent's saved preference, and an explicit false overrides the
global opt-in. Operator remedy: `/occ 3 on` on the affected chat, not a manual
agent edit of `occ.json`. Inheritance is dynamic and nearest-record-wins;
a child's own record masks the parent. Source review confirms that the OCC-2
save path writes false, but a saved false alone does not establish user intent.
Changing this behavior, including the distinction between no override and
explicit off, needs a separate scoped persistence fix.

## Driver-reported canary (2026-11-10 08:00-08:08)
Chain: clean-sender -> clean-receiver, scheduler otherwise idle.
- Sender: occ3_handoff call 1 -> `capture_armed`; then emitted facts as assistant
  content + call 2. Three `source_busy` retries, then **ADMITTED**, receipt
  `agentmsg_5ab8cfcc-5f41-4c00-ab6e-80fb4a17be4d`, queue id
  `occ3q_83cddcab-...`.
- Receiver: native wrapper turn auto-injected ("You are the receiver of this
  attached native state..."); attached state contained the facts verbatim
  (synthetic canary alpha=7, beta=ORCHID, checksum=55-12-90). The driver reported
  character-for-character reception and a sender queue "completed handoff"
  notice. The full transcript and completed-use receipt are not published here;
  this historical narrative is not independent completion proof.
- Note: `/occ3/v1/status` counters (send_ok/capture_ok) did NOT increment
  despite the driver's reported delivery; do not use them as the native-use
  signal. Correlate each admission id with a completed-use receipt instead.
- Under concurrent benchmark load, handoffs hit `scheduler_busy` retries and
  may fall back to ordinary messaging after the retry budget; the tool's
  busy-retry budget is ~3. Record fallbacks per item.

## Benchmark condition 'occ3_sequential'
- 4-agent chain (Planner -> Critic -> Refiner -> Judger), all children on
  dgx-glm53-occ3/glm-5.3, receivers spawned before senders.
- Each sender was instructed to use the two-call protocol (arm -> emit ->
  publish). Judgers were instructed to emit FINAL_ANSWER: <ans>.
- Historical scoring includes five answer recoveries; see README.md. Rows
  record reported hop status, some admission ids, native_hops/total_hops, and
  wall time, but not a complete set of completed-use receipts.
- See smoke-validation.md for the failed post-update smoke and current limits.

Source anchors: `core/settings_manager.rs` (opt-ins and optional endpoint),
`core/occ/session_settings.rs` (save and inheritance), and
`modes/interactive/occ_control.rs` (`/occ 3 on`), under
`crates/pi-coding-agent/src/`.
