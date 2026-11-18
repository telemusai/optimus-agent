---
kind: fix
---

# occ3: resend the same source capture after a pre-admission busy refusal

## Summary

When the server refuses an armed source capture as busy before admission
(HTTP 503 `scheduler_busy`, `admission_capacity`, `admission_timeout` or
`occ3_busy_batch`, before any stream event), the client now waits for the
published `Retry-After` (5 s if absent, at most 30 s) and resends the
identical request up to 2 times inside the same lease. Before, every busy
refusal closed the lease and the agent needed a whole model turn to re-arm a
fresh capture. A sender that meets a briefly busy server is now admitted in
seconds instead of losing about 10 s per re-arm. The two publish enrolls also
run in parallel.

## Semantics

- The lease stays `Submitted` across resends and is submitted once. The
  resend is byte-identical (same `capture_id` and `turn_nonce`).
- A resend is attempted only if the delay plus a 15 s admission headroom
  (the server's published pre-first-event bound) fits both the lease
  deadline and the intent expiry, so a resend never expires in flight.
- After the final busy refusal, the lease closes as proven non-use exactly
  as before (`Cancelled` with a busy record), and the sender's own busy
  budget and backoff still apply.
- A `409 replay_forbidden` on a resend means the server has not released its
  replay fence (servers before the matching fence-release change). The first
  refusal was proven pre-use, so this closes as that busy refusal, not as
  uncertain.
- A cancel signal or the lease deadline during the `Retry-After` sleep also
  closes as the busy refusal, because nothing is in flight. An explicit lease
  cancel (`lease.cancel()`) keeps its conservative `Uncertain`.
- No retry on `429`, on any other `409` or `503`, on a first-attempt `409`,
  or after any stream byte. The receiver (consume) path is unchanged.
- Known gap: with the keepalive stream opt-in, a busy refusal arrives as an
  in-body SSE error and is not resent; keepalive is off by default.
