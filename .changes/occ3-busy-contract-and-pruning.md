---
kind: fix
---

# occ3: busy contract for active-turn and adapter refusals; prune terminal state

## Summary

- An active-OCC-3-turn capture-reservation refusal is now HTTP 503
  `scheduler_busy` with `Retry-After` (proven non-use), not a generic failure.
- The `_charge` adapter concurrency refusal is the same 503 `scheduler_busy`
  contract instead of 429; the other `quota_exceeded` sites are unchanged.
- Terminal registry records, backend attempt slots and worker control-reply
  cache entries are pruned after bounded grace windows, so a long-lived
  server keeps admitting after many arm/retire cycles.
- `capture_prenorm` / `stage_verify` drop a redundant `clone()` after
  `index_select` (already a fresh tensor).
