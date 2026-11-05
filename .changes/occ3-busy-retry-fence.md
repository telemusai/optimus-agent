---
kind: fix
---

# occ3: a pre-dispatch busy refusal frees its own replay fence

## Summary

Resending the same OCC-3 request after a retryable busy 503 returned 409
`replay_forbidden`: the adapter fence (`_fence`) was never released. It is
now released only when the attempt provably never ran.

## Semantics

- Release cases (503 only, no native output observed, buffered path):
  the registry `busy` decline at capture enroll; the scheduler
  pre-admission declines `occ3_busy_batch` / `occ3_busy`; an
  `admission_timeout` raised before dispatch (control plane did not answer).
- Never released: `admission_timeout` after dispatch (first-event bound),
  any busy code after content flowed, keepalive streams, 409/429, and every
  other error or uncertain outcome.
- Except for the registry enroll decline, the worker re-proves non-use first
  (`release_unused_attempt`). A consume runs the new all-rank control
  operation `release_lease`: the registry checks the lease is still
  `leased`, never admitted, no injected rows, no inflight work and not
  revoked, then returns the nonce and lease to `published` and tombstones the
  declined attempt id. A capture checks its record is still `reserved`
  (never admitted) and is retired as before. If the proof is not positive the
  fence stays and cleanup retires as before.
- Fence entries carry their owner attempt id; a context removes only its own
  entry, on the same event loop as `_fence`.

## Deploy

Adapter, backend, registry, worker and control vocabulary change: native
image rebuild (MODE=full). No `RuntimeProfile` change.
