---
kind: fix
---

# occ3: consume admits beside ordinary decode and decodes in the shared batch

## Summary

A consume (receiver) turn no longer needs an empty batch. Its prefill runs as
a one-request eager extend batch while ordinary rows keep decoding; after the
final injected prefill chunk it leaves the eager cohort, merges into the
ordinary running batch and decodes with CUDA graphs and normal batching.
Capture turns are unchanged: exclusive cohort and the immediate pre-use 503
`occ3_busy_batch` when other work is inflight.

## Semantics

- `Req.occ3_attached` (new, set at admission) marks a worker-owned row for its
  whole life; abort matching, the deadline sweep, grammar abort split,
  retraction/`release_req`/`reset_for_retract`, the running-token
  reservation and result settle key on it. `occ3_force_eager` now means only
  "isolated eager cohort".
- `release_consume_cohort` clears the row flag in result settle on the
  scheduler thread, only for a non-finished consume whose injection is
  complete, on the final chunk (not `chunked_req`) of an extend batch.
  `filter_batch` / `merge_batch` recompute the stale batch flag before
  `get_next_batch_to_run` merges `last_batch` (invariant, tested).
- Consume settle is terminal-only: no per-decode-step `commit_lineage`
  gather in a shared batch. The terminal receipt reads only integers, tokens
  and injection metadata, so co-batched graph decode numerics cannot change
  it.
- `prepare_batch` keys on eager rows and requires a one-row batch; a shared
  batch never carries `occ3_context`, so no OCC-3 hook runs inside a forward
  shared with ordinary requests.
- An attached row is never a preemption victim and is retracted last (its
  retraction aborts it, pre-existing contract); ordinary rows beside it stay
  preemptible and retractable. The batch-wide preemption guard is removed.
- `_occ3_prefill_selection` drains only for an eager row; a decoding
  post-injection consume never blocks other prefills.
- Admission: capture still declines (`quota` → 503 `scheduler_busy`) with any
  active row; consume declines only behind another eager row.
- `abort_request` is latched after retire so an overlap overshoot tick does
  not gather again.

## Deploy

Changes `patches/scheduler-cache-ipc.v1.json` (schedule_batch, schedule_policy,
scheduler, batch_result_processor, forward_batch_info): native image rebuild
(MODE=full). No `RuntimeProfile` change.
