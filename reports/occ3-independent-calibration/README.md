# OCC-3 independent calibration campaign

Protocol version 2 - pre-registered before final-test data collection.

This directory holds the versioned evaluation protocol, the exact equations and the
semantic-experiment protocol for the independent-calibration campaign
(branch `fix/occ3-independent-calibration-semantic-validation`, baseline = merged main
`5e2bb750730a2f2d6d9a0c86e555c2ca495a7ffa`). The campaign brief lives outside the repo in
`C:/Users/openclawuser/occ3-icalib-briefs/CAMPAIGN-SPEC.md`; this directory is the repo-side
statement of the same protocol. Where the two disagree, the repo-side files are the versioned
protocol and the brief is the source spec.

## Index

| File | Content |
|---|---|
| `README.md` | This index, the architecture under test, the methods, the status ladder, historical corrections. |
| `EVALUATION-PROTOCOL.md` | Protocol version 2. Replaces the pre-registered acceptance rules of the PR #204/#205 windows; corrects spec section 3 items A-G; defines the new acceptance rules and the dataset plan. |
| `EQUATIONS.md` | Exact equations for both methods, serving normalisation, anchoring and every evaluation metric, with the implementing module citations. |
| `SEMANTIC-PROTOCOL.md` | Pre-registered causal-semantic experiment: first live control, arms, task families, scoring, pilot and final test, promotion gate, statistics and reporting rules. |

## Architecture under test

The architecture is unchanged from the merged state and is not being redesigned:

```
accepted sender hidden states
  -> frozen mathematical alignment
  -> continuous input-embedding prefix
  -> authenticated GPU-resident OCC-3 handoff
  -> receiver inference
```

Non-negotiables carried over from the campaign spec:

- No KV-cache communication. No model-weight changes.
- No gradient training and no learned neural adapter.
- No replacement of the existing OCC-3 bridge; the transport, native sessions, receipts,
  capability negotiation, communication indicator and deployment controller are reused.
- The prefix is `frozen_map(sender_hidden_states)` at inference time; it is never fitted
  per message.
- Closed-form calibration does fit a mathematical mapping. It is described accurately as
  frozen-model, closed-form calibration; it is not claimed that no fitting takes place.
- No automatic merge. Negative results are reported as results.

## Methods under test

| Role | Identity | Fitted on | Solver |
|---|---|---|---|
| Historical baseline (not A or B) | `statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1` | the transmitted rows themselves (in-sample; proven near-copier) | per-message whitened Procrustes (`occ3_sglang/statebridge.py:32`) |
| Method A | `statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1` | FIT partition of independent turns only | frozen dual ridge, M x M Cholesky (`occ3_sglang/statebridge_calibrated.py:67`, `fit_map` :479, `apply_map` :552) |
| Method B, variant `o` | `statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes.v1` | FIT partition of independent turns only | frozen low-rank whitened Procrustes over O(r) (`occ3_sglang/statebridge_calibrated_procrustes.py:89`, `fit_map` :497, `apply_map` :633) |
| Method B, variant `so` | `statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes-so.v1` | same | proper-rotation variant; separate identity and artifact (`statebridge_calibrated_procrustes.py:90`) |
| Method B, variant `plain` | `statebridge.glm53.final-prenorm-predictor.calibrated-lowrank-procrustes-plain.v1` | same | unwhitened comparator with fitted scale a0 (`statebridge_calibrated_procrustes.py:91`) |

Method A keeps its existing identity and artifact schema (`optimus.occ3.calibration-artifact.v2`).
Method B has its own artifact schema (`optimus.occ3.calibration-artifact.lowrank-procrustes.v1`,
`statebridge_calibrated_procrustes.py:99`). Both methods expose the same module surface
(`fit_map` / `apply_map` / `serving_transform` / serialize / deserialize / `describe`) so the
shared registry, dispatch, corpus evaluation, promotion gate and receipts stay method-agnostic.
The historical per-message method remains a baseline for comparison only; it must not silently
become the implementation of A or B.

## Status ladder

| State | Meaning |
|---|---|
| IMPLEMENTED | The code exists on this branch and passes its own guards; no test claim is made. |
| OFFLINE TESTED | Focused tests pass offline in the campaign venv (CPU fixtures; no DGX, no GPU, no network). |
| SOURCE-ATTESTED TESTED | Tests that attest pinned or attested source pass against the named source root. |
| LIVE GPU TESTED | Exercised on the real GLM-5.3 FP8 / SGLang 0.5.21 / TP8 DGX B300 runtime with execution receipts. |
| SEMANTICALLY VERIFIED | Passed the pre-registered semantic promotion gate of protocol version 2 (`SEMANTIC-PROTOCOL.md`). |
| NOT RUN OR BLOCKED | Not executed in this campaign, or blocked by a named blocker. |

Current honest state as of 2026-10-28 (wave-1 offline repairs were still in flight when this
index was written; rows for in-flight work are marked):

| Item | State | Evidence |
|---|---|---|
| Pinned-source attestation repair | SOURCE-ATTESTED TESTED | pristine checkout `e00930c5` (v0.5.21) hash-verifies 11/11; predictor suite 26 passed, chat suites 200 passed (campaign briefs `pinned-source-report.md` sections 1, 5) |
| Method B module and tests | IMPLEMENTED + OFFLINE TESTED | `tests/test_statebridge_calibrated_procrustes.py`, 19 passed (briefs `procrustes-impl-report.md` section 4) |
| Method A fit/apply/evaluate separation | IMPLEMENTED (wave-1, in flight) + OFFLINE TESTED at best | `occ3_sglang/statebridge_calibrated.py` fit_map/apply_map/serving_transform; suite in flight at time of writing |
| Three-partition corpus and evaluation record | IMPLEMENTED (wave-1, in flight) + OFFLINE TESTED at best | `occ3_sglang/calibration_corpus.py` `split_manifest` :196, `evaluate_map` :492, `validate_for_activation` :634 |
| Registry/dispatch integration for both methods | IN FLIGHT (wave-1) | `occ3_sglang/native_profile.py`, `alignment_dispatch.py`; see campaign briefs `python-audit-report.md` section 3 |
| First live control (T-exact / T-norm / N-shape / N-none / F-text) | NOT RUN | `SEMANTIC-PROTOCOL.md` section 3 |
| Calibration corpus collection on GLM-5.3 | NOT RUN | `EVALUATION-PROTOCOL.md` section 5 |
| Live fit / validation / test of either method | NOT RUN | nothing LIVE GPU TESTED exists for A or B |
| Pilot 16 and final 64 semantic tasks | NOT RUN | `SEMANTIC-PROTOCOL.md` sections 8-9 |
| Benchmarks (TXT / OCC-1 / OCC-2 / methods) | NOT RUN | campaign spec section 19 |
| Semantic verification | NOT RUN; nothing is SEMANTICALLY VERIFIED | `SEMANTIC-PROTOCOL.md` section 10 |

The native transport itself (capture, admission, one-shot delivery, receipts, TP8) carries
live evidence from the PR #205 window, but no component of this campaign has run on the GPU.
All campaign infrastructure is OFFLINE TESTED at best; nothing is LIVE GPU TESTED yet.

## Historical corrections (records stand; narrative corrected here)

1. `reports/occ3-alignment-repair/LIVE-WINDOW-1.md` section 6.1 states the host "no longer
   has" a pinned SGLang checkout. That claim is stale. The pristine checkout was re-verified
   on 2026-10-28 at `C:/Users/openclawuser/occ3-latent-validation/references/sglang`
   (`e00930c5489053f26d86b179cee0d087f846acbb`, v0.5.21): all 11 capture-manifest files
   hash-equal the manifest, no content drift, and the in-repo source-attested path is restored
   (campaign briefs `pinned-source-report.md` section 1). The deployed-image attestation
   recorded there (11/11 `after_sha256`) remains valid as an independent cross-check. The old
   report is history and is not edited; this note is the correction of record.
2. The PR #205 window is 2 completed consumptions with semantic misses plus 1 uncertain
   attempt whose handoff was never admitted and which produced no answer text
   (`live-window-1/receipts/cases-case-00/01/02.json`). The prose that the transport was intact in
   all three trials (capture succeeded, handoff admitted, receiver answered) is an
   over-claim relative to the receipts and is not repeated here. The 0/3 semantic result
   and the INCONCLUSIVE verdict stand unchanged; no old trial is reclassified.
3. The shipped per-message alignment is a proven in-sample copier
   (`reports/occ3-alignment-repair/PRODUCTION-SCALE-DEGENERACY.md`); its 0.931 live cosine and
   0.101 within-turn R2 describe that method only, and neither number is evidence about
   Methods A or B.

## What this campaign must not repeat

- A green indicator for a completed-use receipt is not semantic proof; activity state and
  semantic validation status stay separate.
- An Uncertain attempt must never be resent as ordinary text; single-use state leases are
  preserved.
- Metric names must be exact: `r2_standard` and `r2_vs_fit_mean` are different metrics and
  both are reported (`EQUATIONS.md` section 7).
- Negative and undefined metric values are recorded, never suppressed
  (`EVALUATION-PROTOCOL.md` section 3, R4).
