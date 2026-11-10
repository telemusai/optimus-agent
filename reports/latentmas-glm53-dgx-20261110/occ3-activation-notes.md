# OCC3 Activation Fix — Root Cause and Procedure

## Symptom
`rlm` subagents spawned on `dgx-glm53-occ3/glm-5.3` reported `occ3_handoff ABSENT`
from their tool list. Calls to the tool (when prompted) returned:
`OCC-3 is disabled for this chat; use /occ 3 on with a supported model`.
Daemon log showed `occ3 activation declined reason=experimental_ack_required`.

## Root Cause (confirmed 2026-11-10)
`~/.config/optimus-rust/settings.json` had NO `agentCommunication.occ3Enabled`,
NO `agentCommunication.occ3Endpoint`, and NO top-level `occ3Experimental` key.
Every session therefore declined OCC3 activation silently (DisabledByPolicy —
no log line), so the `occ3_handoff` tool was never installed into subagent
sessions, even though:
- models.json `dgx-glm53-occ3` provider block was already `enabled: true`,
  `validation: live-verified`, endpoint `http://127.0.0.1:30001`;
- the sglang server's occ3-bridge plugin reported `NATIVE_READY`
  (`installed_experimental_not_live_validated`) at `/occ3/v1/status`.

## Fix (operator-applied; keep as-is)
`settings.json` must contain:

```json
{
  "agentCommunication": {
    "occ3Enabled": true,
    "occ3Endpoint": "http://127.0.0.1:30001"
  },
  "occ3Experimental": true
}
```

New session workers read these settings at start. Existing/root sessions keep
their old settings until a model change — for a running chat, the per-chat
preference also gates native actions (daemon RPC `occ_set_preferences` with
`occ2Enabled: true, occ3Enabled: true`, equivalent to `/occ 3 on`).

## Verification procedure
1. Spawn a probe pair on the OCC3 model:
   `await rlm(prompt, name=..., model="dgx-glm53-occ3/glm-5.3")`
2. Ask each child to list its tools via agent_message to parent.
3. PASS: `occ3_handoff` appears in the tool list, and the daemon log
   (`~/.config/optimus-rust/logs/daemon.sock.<id>.log`) shows
   `occ3 experimental activation enabled session=<child id>`.
4. Full handoff protocol (per SKILL guidance): first call arms capture
   (`capture_armed`, nothing sent); then the sender emits concise source facts
   as assistant content and calls `occ3_handoff` once more for the same
   receiver, with no other tool calls in that response. A completed native-use
   receipt proves delivery; queued/admitted alone does not.

## Confirmed working (2026-11-10 ~07:53 UTC)
- occ3-verify-a / occ3-verify-b: both report `occ3_handoff` in tool list.
- Earlier failures (probe-a/b, occprobe-a/b, latent-sender/receiver) were all
  pre-fix; their handoff attempts returned `OCC-3 is disabled for this chat`.

## Constraints (operator directives)
- Model traffic stays on loopback `127.0.0.1:30001` only.
- Never restart/reconfigure/release the GLM-5.3 sglang server.
- Do not send raw requests to the daemon socket; do not restart the daemon.
- An item counts as OCC3 only with a recorded handoff receipt; log items
  where the tool was not called.

## SECOND ROOT CAUSE (2026-11-10 ~07:58) — per-chat preference file
Even with global settings fixed, handoff CALLS still returned
`OCC-3 is disabled for this chat`. Cause: the root session's saved chat
preference file
`~/.config/optimus-rust/session-artifacts/<root-session>/occ.json`
contained `{"occ2Enabled": true, "occ3Enabled": false}` (created by an earlier
occ_set_preferences daemon RPC that only persisted occ2). Subagents inherit
the nearest parent's saved preference, and an explicit false overrides the
global opt-in. Fix: set `occ3Enabled: true` in that file (equivalent to
`/occ 3 on` persisted state). Children then inherit OCC-3 on at spawn time.

## END-TO-END VERIFICATION (2026-11-10 08:00-08:08)
Chain: clean-sender -> clean-receiver, scheduler otherwise idle.
- Sender: occ3_handoff call 1 -> `capture_armed`; then emitted facts as assistant
  content + call 2. Three `source_busy` retries, then **ADMITTED**, receipt
  `agentmsg_5ab8cfcc-5f41-4c00-ab6e-80fb4a17be4d`, queue id
  `occ3q_83cddcab-...`.
- Receiver: native wrapper turn auto-injected ("You are the receiver of this
  attached native state..."); attached state contained the facts verbatim
  (SECRET BUNDLE alpha=7, beta=ORCHID, checksum=55-12-90) — values appear in NO
  ordinary transcript entry, proving the latent attachment path. Receiver
  confirmed character-for-character reception; sender queue logged
  "completed handoff".
- Note: `/occ3/v1/status` counters (send_ok/capture_ok) did NOT increment
  despite the verified native delivery; do not use them as the native-use
  signal. Use per-hop receipts (agentmsg ids) and queue notices instead.
- Under concurrent benchmark load, handoffs hit `scheduler_busy` retries and
  may fall back to ordinary messaging after the retry budget; the tool's
  busy-retry budget is ~3. Record fallbacks per item.

## Benchmark condition 'occ3_sequential'
- 4-agent chain (Planner -> Critic -> Refiner -> Judger), all children on
  dgx-glm53-occ3/glm-5.3, receivers spawned before senders.
- Each sender forced to call occ3_handoff exactly once per protocol
  (arm -> emit -> publish). Judger must end with FINAL_ANSWER: <ans>.
- Scoring: same repo extract/normalize. Per item: handoff status per hop,
  receipt ids, native_hops/total_hops, wall time.
