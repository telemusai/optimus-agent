# Post-update smoke validation — 2026-11-10

**FAIL. Phase 2 and the main runner remain paused.** The pass rule required
three native hops with completed-handoff receipts and a parseable final answer.
The correct answer alone did not satisfy it. This diagnostic does not update the
historical benchmark denominators or demonstrate a native-state accuracy benefit.

## Stack and saved evidence

- Server: `neo-glm53`, overlay image `1b7d9ea2ab2d`, based on the historical
  image with `latent_backend.py`, `latent_registry.py`, and
  `latent_control_api.py` replaced from main `c34c0bdcc`.
- Client: `c34c0bdcc`. Source child prompts differ from the historical v3 prompts.
- The original `results/glm53_gsm8k_occ3_v6.jsonl` row is retained unchanged.
  Its `native_hops: 1` is the driver's admission-based label, not verified use.
- `server-smoke-excerpt.txt` contains bounded, normalized server diagnostics
  read by the parent after the run. It is not the full server log.
- The DGX driver's `results/PHASE1_SMOKE_RESULT.md` reports continuation
  exhaustion and parent-transcript access. Its assertion that there was no
  server failure is contradicted by the server evidence below; do not adopt
  that root-cause conclusion. Full child transcripts are not included here.

## Observations

| Field | Recorded value |
|---|---|
| Item | GSM8K test item 0 |
| Answer / strict answer | `18` / `18`, both scored correct |
| Driver-labelled native hops | 1/3, based on admission |
| Wall time | 520.4 s |
| Token usage | Absent from the row |
| First-role-output timestamps | Null for all hops |

- Planner -> Critic: admitted message
  `agentmsg_21eaaf0e-4b39-4989-83e9-421a7e20589e`. The row records arrival at
  211.4 s. At 10:08:54Z the server reports `capture commit failed` with
  `UniformRequestCancellation code=retired`, then `occ3_commit_failed` (503)
  and receiver `native_completion_failed` (409). A receiver turn did run;
  the claim that the state simply never woke the Critic is not sufficient.
- Critic -> Refiner: no receipt; the row reports `unknown` with
  `fallback_used: true`. The driver report says it forwarded the Critic's
  blocked-status text despite no role output. That is not a valid role handoff.
- Refiner -> Judger: the driver report records admitted message
  `agentmsg_90e2da56-9034-459a-9d8c-e262ba3f3cb8`, while the row omits the id
  and reports `unknown`, with arrival at 518.3 s. The server's receiver turn
  ends `cancelled` (499) at 10:13:59Z. The cause of cancellation is not
  established by this excerpt.

The driver report says receivers exhausted their three automatic continuations
before input arrived, and that the Judger answered before the final state
arrived. It also reports the Critic reading the parent's transcript through
`agent_observe`, where scoring context was accessible. These driver-reported
findings are not independently established by the published per-item row.
They are enough to reject a claim of isolated benchmark accuracy.

## Limits and next validation

The excerpt establishes server-side failure as well as driver problems; it does
not establish the ultimate cause or that PR #279 introduced the failure.
The overlay also carries earlier backend-control changes. Do not infer
causality from the before/after deployment alone.

Keep the historical data unchanged. Before a new scored run, isolate gold
answers from child-visible transcripts/files/tools, require actual role output,
correlate admission ids with completed-use receipts, record usage and ordering,
and retain receivers until delivery is resolved. Run a one-hop smoke before
attempting the four-role chain. The guidance changes are not a claim that the
runtime failure has been fixed. No server restart or further benchmark run was
performed while preparing this note.
