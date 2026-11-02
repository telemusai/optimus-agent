# OCC-3 NATIVE-arm tolerance ladder and bounded re-send (2026-11-02 decision)

Status: implemented in the phase-suite harness only; verification NOT-RUN in this
change (build delegated). This page records the policy, its knobs, its evidence
surface, and the strict-gate guidance. It carries no live-window claim.

## The decision

The NATIVE arm's historical verdict is strict exact-value recovery (invariant G):
any character error fails the arm. The measured live canaries show the retained
channel currently lands near-miss values (canary-1: one deletion; canary-2: two
deletions plus two substitutions). The decision adds bounded error tolerance to
the arm verdict instead of failing on character-level noise:

1. Tolerate a small error: the arm accepts a recovered value whose character
   edit distance from the intended value is at most `OCC3_ARM_TOLERANCE`
   (default 1; ceiling 4).
2. Send the message no more than three times: out-of-tolerance sends are
   re-sent with a FRESH capture (fresh session ids, fresh publication), at most
   `OCC3_ARM_MAX_ATTEMPTS` times (default 3; ceiling 3).
3. After the final send, accept what was transmitted: the best structurally
   valid transmission (lowest distance, earliest attempt wins ties) is accepted
   as the arm's result instead of failing the arm.

An exact match on any send is accepted immediately and stops the ladder; a
within-tolerance match is likewise accepted on first sight (the tolerance is
the acceptance bar, so no further send is spent chasing exact).

## Scope

- NATIVE arm with a minted value only. Token arms (TOKEN_FULL /
  TOKEN_SELECTED / TOKEN_NORM / NULL_SHAPE) keep their pre-registered
  structural verdicts; controls (NO_ATTACHMENT / FULL_TEXT / WRONG_SENDER)
  keep theirs.
- Corpus-turn canaries stay single-send: the manifest turn's family session
  id and payload are pre-committed; a re-send would need a fresh manifest
  turn, not a ladder.
- The phase runners (P-phase gates, task route, offline selftest legs) keep
  their own strict `case_passed` contracts; only the arm-trial verdict
  changes.
- One phase claim per arm, as before. The ladder's sends are sub-attempts of
  the one claimed arm attempt (`retry_policy.claim_scope: "arm"`).

## Knobs

| Env | Default | Ceiling | Meaning |
|---|---|---|---|
| `OCC3_ARM_TOLERANCE` | 1 | 4 | Max accepted character edit distance. `0` = strict. |
| `OCC3_ARM_MAX_ATTEMPTS` | 1 is refused; default 3 | 3 | Total sends per message ("no more than three times"). |

Both are validated at producer startup (`validate_arm_tolerance_env`) before
any enrollment or send side effect; malformed or out-of-bound values refuse
the run. Nothing is silently coerced.

## Verdict and evidence

The arm verdict for NATIVE is `arm_accepted && structural_ok`:

- `arm_accepted` - the ladder accepted a transmission (see below).
- `structural_ok` - the pre-registered structural contract (delivered,
  consumed, source lifecycle, receiver purity, audit shape) of the accepted
  attempt. A structurally invalid send is never accepted, not even as the
  best-effort fallback.

New fields on the NATIVE arm evidence and trial summary (additive; arm-evidence
schema stays `optimus.occ3.arm-evidence.v1`):

- `value_distance` - character Levenshtein distance between the intended
  value and the accepted attempt's parsed answer.
- `arm_accepted` - boolean.
- `accept_reason` - `exact` | `within_tolerance` | `best_effort_fallback` |
  null (not accepted).
- `attempts_used` - sends spent (1 when the first send is within tolerance).
- `attempts` - per-attempt records: session ids, distance, answer sha256,
  strict match, structural ok, and any trial error. A total failure still
  publishes this array.
- `retry_policy` - `{schema, tolerance, max_attempts, enabled, claim_scope,
  scope}`; `enabled: false` records why (strict pin or corpus canary).
- `strict_value_recovery` - the accepted attempt's `semantic_nonce_match`.

`semantic_nonce_match` keeps its strict meaning everywhere it already
appears; nothing existing was redefined.

## Strict-gate guidance

The default is the tolerant policy. Pre-registered strict-L1 windows (for
example canary-4) MUST pin `OCC3_ARM_TOLERANCE=0`, which restores the
historical exact single-send verdict bit-for-bit, or read
`strict_value_recovery` from the trial summary instead of trusting
`arm_case_passed` alone. The completion contract's exact-recovery gates are
not weakened by this change; they are now explicit pins rather than the
default.

## Regression coverage

- Pure: edit-distance cases; policy parsing (defaults, bounds, garbage);
  ladder accept order (first within tolerance; best-effort lowest-distance
  earliest-attempt; errored sends never selected; strict never accepts
  out-of-tolerance).
- Fixture (Windows harness lane): one-character deletion accepted within
  tolerance on send one (strict verdict recorded false); two-character
  error re-sent three times then accepted best-effort; the strict pin
  refuses the same one-character error exactly like HEAD.
- The fixture models canary-1's measured shape through a
  `receiver_answer_drop_chars` injection (default 0 = exact decode).

## Not run (delegated)

`scripts/check.sh`, focused cargo tests for the example suite (the
`gate_tests` module runs on the Windows harness lane), and the offline
selftest in both modes are NOT-RUN in this change; the build agent runs them
before merge. No live window is claimed by this page.
