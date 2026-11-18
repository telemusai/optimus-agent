---
kind: fix
---

# occ3: reject armed admission against a busy batch; never log abort payload text

## Summary

Armed (`occ3_force_eager`) requests admitted while ordinary decode traffic was
running were parked in the waiting queue behind it, holding the one-shot native
attempt until the 30 s queue bound aborted it. Admission now declines up front
against a busy batch, inside the all-rank admission preflight; the scheduler
wrapper sends the decline as its own bounded code (`occ3_busy_batch`) and the
adapter restores an observable HTTP 503 `scheduler_busy` with a published
`Retry-After` (5 s, mirroring the patched armed admission window) — fast,
explicit and retryable with a fresh attempt, with no silent zero-event stream
and no replay. The patched armed admission window shrinks from 30 s to 5 s:
admission itself declines immediately against a busy batch, so the window now
only bounds the post-admission cohort race, enforced fail-closed by the
existing queue-bound sweep.

Also fixes a pre-existing diagnostic leak: `_refusal_detail` in
`occ3_sglang/chat_adapter.py` rendered the first 64 characters of string
values, and arbitrary key names, from unclassified abort payloads into the
container log. String values now reduce to `<str>` and only bounded
structural keys survive; non-string scalars are unchanged.

## Semantics

- Busy gate lives in `admit_request`'s `validate()`, after the identity/replay
  guards: `scheduler.collect_inflight_reqs()` plus `chunked_req`, counting only
  unfinished rows. Empty batch admits immediately.
- The decline is a `LatentError("quota")` raised inside
  `runtime.preflight("admit", ...)`, so every rank agrees on the same
  request/error before any mutation; a rank disagreement still poisons the
  runtime (fail-closed, unchanged).
- The patched scheduler wrapper maps that decline to the distinct
  `occ3_busy_batch` code (`AbortReq` 503); every other admission failure keeps
  `occ3_admission_failed`.
- The pinned first-chunk ValueError seam in serving_chat flattens the raised
  `AttachmentError` into an inner 400 whose message is exactly the fixed
  public code; `_run` restores `AttachmentError("scheduler_busy", 503)`
  from that bounded message before the generic completion fold. The default
  buffered contract therefore delivers the 503 with `Retry-After` before any
  body byte; the keepalive opt-in carries the same fixed code in its bounded
  SSE error event.
- Client-side chain is unchanged and one-shot preserving: the 503 folds to
  `occ3_inference_uncertain` / `request_interrupted` (permanent, no
  auto-replay); the PR235 resolve-hook tombstone counts the failure and
  resets the slot so the next turn re-arms with a fresh capture (budget 3).
  A receiver-side retry is a fresh envelope; claimed-but-not-consumed
  deliveries still refuse silent re-claim.

- Final gate (F1): the queue-bound sweep abort now carries its own bounded
  reason (`occ3_admission_timeout`), and `_occ3_abort_attached` threads an
  `occ3_`-prefixed bounded abort message through as the refusal reason while
  any other abort — user or client cancellation, or a malformed message —
  keeps `cancelled`. Armed-cap saturation maps to `occ3_busy` with its own
  public `admission_capacity` 503; a genuine user cancellation keeps the
  historical 409 under its own bounded code (`client_cancelled`). These
  admission codes only take the retryable shape when they are the first and
  only stream output (nothing was generated); the same scheduler code after
  content has flowed stays a generic `native_completion_failed` completion
  failure.
- Final gate (F2, revised after root review): ONE request-scoped absolute
  admission control deadline (`ctx.admission_control_deadline`, set once
  when the attempt is charged) is shared by EVERY pre-dispatch control
  await — the authorize-time readiness refresh, `enroll_source`, `reserve`
  and `prepare_source` — so consecutive slow control calls compound no
  further than that single bound (`BufferProfile.admission_control_seconds`,
  7 s default). At dispatch the adapter arms a first-event deadline
  (`BufferProfile.admission_first_event_seconds`, 8 s default,
  dispatch-relative, covering the patched scheduler's 5 s armed window plus
  sweep/tokenizer/stream margin): until the FIRST native event arrives,
  `_guard` and the keepalive race cap their waits by that deadline and fail
  closed with the bounded retryable `admission_timeout` 503. Admission is
  never inferred from reserve/prepare success, and a scheduler producing no
  event at all is no longer held silently to the turn deadline. Once the
  first native output has flowed (`ctx.native_started`), the admission
  bounds no longer apply: admitted generation keeps its own turn deadline
  (`_guard` 504 semantics unchanged). Worst-case pre-first-event wall
  clock: 7 s control + 8 s first-event = 15 s, plus at most
  `cleanup_seconds` (1 s default) of fail-closed settle before the client
  sees the 503. The published `x-optimus-occ3-admission-window` header now
  states that accurate end-to-end maximum (15), deliberately distinct from
  the advisory `Retry-After` (5).
- Final gate (F3): the keepalive SSE path restores the same bounded public
  shapes from the flattened inner 400 (it previously folded them to the
  generic completion failure), so a post-start admission refusal ends the
  committed stream with exactly one structured error event carrying the code,
  status and `retryAfterSeconds`; no HTTP status is claimed after headers.

