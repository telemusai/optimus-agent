# OCC-3 Phase 3 — Benchmark Iterations (running ledger)

Status 2026-11-02. This page is the Phase 3 benchmark's iteration ledger: one row per
calibrated-map iteration with its measured numbers and the gate for the next step. It runs
beside LIVE-RESULTS.md (window evidence) and TOLERANCE-RETRY.md (ladder policy). Upstream
PRs #221 (tolerance verify fix), #222 (calibration_pool method-union), #223 (AGENTS.md PR
docs) are open; the maintainer merges. Every number below is measured and carries its
source or digest; PENDING items carry the gate that fills them. No projections.

## 1. Iteration ledger

| Iter | Date | Map (content digest / sha256 / bytes) | Measured score | Live strict result | Status |
|---|---|---|---|---|---|
| 0 baseline | through 2026-11-01 | `3e76d627` / `59d60f83` / 72799976 B | own-corpus val r2 0.203625, test 0.282283 (8t/159r, 15t/979r) | canaries 0/3 exact (d=1, d=4, no-value); TOKEN_FULL exact 84.9 s | superseded by iter 1 |
| 1 s9 pooled refit | 2026-11-02 | `9a7394de` / `f6b101e5` / 88564767 B | same-split val r2 0.295305, test 0.307213 (13t/476r, 20t/1341r) | not measured (canary-4 PENDING) | at canary-4 gate |

### Iteration 0 — baseline map (measured; LIVE-RESULTS.md)

Artifact `calibrated-dual-ridge-3e76d627a1f81f91.bin` sha256
`59d60f83587bd8283110dfe57cd0f9b63bc95968dc2eda01251252ef67b945ea` (72799976 B), budget
2048, activation allow. Own-corpus heldout: validation 8t/159r r2 0.203625, test 15t/979r
r2 0.282283. Live: three controlled-swap canaries, all arm-A FAIL at exact recovery —
canary-1 d=1 (one deletion), canary-2 d=4 (two deletions + two substitutions), canary-3
no-value (parsed answer UNKNOWN, coverage proven); the TOKEN_FULL exact-embedding
diagnostic recovered a fresh value exactly in 84.9 s (84993 ms), isolating the defect to
the calibrated map. All NATIVE-style live turns to date: exact 0/8 (near-miss 2/8,
no-value/wrong-value 6/8).

### Iteration 1 — s9 pooled refit (measured 2026-11-02, in-container, DGX evidence)

- Pooled corpus `/data/neo-glm53/occ3-calibration-capture/icalib-pooled-20261102T1115Z`
  (pooled 2026-11-02 11:14Z from legacy-p4 frozen `027b041f…` + supplemental-derived
  `5acf45e5…`): fit 39t/1594r, validation 13t/476r, test 20t/1341r; corpus manifest
  `5d927c3197aea0c1c8c51e06ff20a209081db17aea324e52eaf55fe44af27bcf`, split
  `4faf6076c294637610e40772c6643613088ec0a2dc2b601b17b098528afd82b2`. Method ids
  `continuous-prefix.v1` + `calibrated-dual-ridge.v1` (union; the in-container pool tool
  staged for the refit, sha256 `04055f2e…`, is the reviewed PR #222 fix).
- Refit: 12-run grid (a / b / b-so / b-plain × M512/1024/2048; threads 1, nu
  0.6987816772616573), selection rule max validation r2, no tie: **selected a@M2048**
  (validation r2 0.295305; runner-up a@M1024 0.268187). Fit jsons `p5pooled-*` and
  `p5-artifact-pooled/` on `/data/neo-glm53`.
- Selected artifact `/data/neo-glm53/p5-artifact-pooled/calibrated-dual-ridge-9a7394dea73cf8d3.bin`:
  content digest `9a7394dea73cf8d33fd651bf8b43d62d739c94d0940bd6f4ae494fd83ba0be90`, sha256
  `f6b101e54e6702dfd177ef596d89abba16d740e67207633ce54fd2376d220a29`, 88564767 B; freeze
  refit digest equals selection digest (determinism). Own freeze metrics: fit r2 0.875553 /
  cos 0.965059 / relF 0.334729; validation r2 0.295305 / cos 0.656488 / relF 0.799015 /
  id_recovery 0.277311; test r2 0.307213 / cos 0.667869 / relF 0.787204 / id_recovery
  0.159582.
- **Same-split cross-score 3x2** (all three maps on the identical pooled validation/test
  rows, serving n0, threads 1; `p5-crossscore-pooled-20261102/pooled-split-crossscore.json`
  sha256 `6e8ee780…`, script sha256 `542ff0c2…`; cell = r2 / cos / relF / id_recovery):

| Artifact | validation (13t/476r) | test (20t/1341r) |
|---|---|---|
| baseline `3e76d627` | 0.220867 / 0.6057 / 0.8402 / 0.2626 | 0.244601 / 0.6303 / 0.8220 / 0.1439 |
| supplemental-only `ce2b0a2d` | 0.201935 / 0.5774 / 0.8503 / 0.0525 | 0.190480 / 0.5734 / 0.8509 / 0.0365 |
| **pooled `9a7394de`** | **0.295305 / 0.6565 / 0.7990 / 0.2773** | **0.307213 / 0.6679 / 0.7872 / 0.1596** |

- Deltas pooled − baseline: validation r2 **+0.074438**, cos +0.0508, relF −0.0411,
  id +0.0147; test r2 **+0.062612**, cos +0.0376, relF −0.0348, id +0.0157. Pooled wins
  every metric on both partitions. Ordering: **pooled > baseline > supplemental-only** on
  validation and test; baseline beats supplemental-only on this split (the reverse of
  their own-corpus numbers), settling the earlier cross-corpus ambiguity on a common basis.
- Activation: **ALLOW** (no refusal reason; policy min_turns 8 ≤ validation 13 turns).
- Caveat (recorded with the run): training-set size is not controlled (fit rows 1335 /
  259 / 1594); pooled trains on the union, so part of its edge is more data, not only
  better fitting. Ranking is on identical evaluation rows.
- Reproducibility: a first cross-score run printed identical r2 values and failed at the
  summary write (`KeyError partitions['turns']`); one-line script fix, re-run identical.

### Stop rule

Bench stop rule (pre-registered): M_i = strict exact-recovery rate over the fixed N=64 set;
stop after two consecutive relative improvements r_i < 3%. Status: **iteration 1 is far
above the 3% bar** — the measured same-split gain, val r2 +0.074438 over baseline 0.220867
(computed +33.7% relative) and test +0.062612 over 0.244601 (computed +25.6%), so the map
keeps iterating. The live M_i inputs are PENDING (no bench window has run; M_0 anchor:
exact 0/3 canaries, 0/8 all NATIVE-style turns). Next iterations and their gates:

| Iter | Content | Gate |
|---|---|---|
| 2 | canary-4 strict-L1 on the pooled map | quiet GPU window (refit window measured GPUs 82-100% busy, all 8) + root decision; strict pin `OCC3_ARM_TOLERANCE=0`, harness binary pin `73d79a21…` (no rebuild) |
| 3 | first bench campaign (fixed N=64 manifests) fills M_1 | canary-4 arm-A strict pass; manifest inventory minted |

## 2. Deploy state (measured)

- The pooled artifact exists on disk only. `apply_map` NOT run; no deploy, no swap, no
  restart, no canary turn, no swap-check during any s9/pooled work.
- Serving untouched proof: container `cfcb2f12c7db`, image `579210da8c74`, PID1 lstart
  Sun Nov 1 23:56:04 2026 unchanged across the refit window (etimes 41767 → 42175 s,
  +408 s wall; same container id, same image id, same PID1 start second). Serving still
  runs the baseline artifact `59d60f83` (72799976 B, Oct 31 mtimes).
- Canary-4 PENDING: needs a quiet GPU window (load line recorded during the refit: GPU
  util 82-100% on all 8 GPUs, decode throughput 482-546 tok/s) and the root decision.

## 3. Tolerance-ladder state (policy in TOLERANCE-RETRY.md, PR #221)

- Defaults: `OCC3_ARM_TOLERANCE=1` (ceiling 4), `OCC3_ARM_MAX_ATTEMPTS=3`; strict windows
  pin `=0`. Corpus-turn canaries stay single-send.
- Measured implication of tol=1 on the three recorded arm-A FAILs: canary-1 (d=1) converts
  `within_tolerance` on send one; canary-2 (d=4) converts via `best_effort_fallback` after
  the three-send ladder; canary-3 (no-value) remains failed. Exact recovery stays 0/3
  under any tolerance (no measured d=0 exists). Tabulator cross-check (4 windows / 34
  turns, measured): tolerance 1 → canary-1 window 100% tolerant; tolerance 4 → canary-2
  window also 100%; supplemental window 0% (30 no-value). Single-send d ≤ t is a lower
  bound on ladder acceptance.
- Retune decision: **NOT justified at n=3** (recorded single-send distances: one d=1, one
  d=4, one no-value). The missing measurement is live `attempts[].distance` from actual
  ladder re-sends — none exists yet; canary-4 (strict) plus the ladder-enabled bench
  subset over the fixed N=64 set fill it. Until then the default stays tol=1 / max=3.

## 4. Bench harness state (offline, measured 2026-11-02)

- Portability suite: 124 tests (50 `test_memory_bench_portability` + 74
  `bench.tests.test_local_scoring`), 17.517 s, exit 0; stdlib-only, offline. Run at the
  POOL worktree HEAD `32a7a9269`; the `bench/` tree and the portability test are
  bit-identical (md5) across the TOL / POOL / ICALIB worktrees, so the results transfer.
- Converters + evaluate verified on synthetic fixtures only (43 files; LoCoMo sample run
  5 questions, mean 1.0, run sha256 `5ecb8737…`, scoring provenance sha256 `3767165a…`) —
  synthetic wiring checks, not evidence of model quality.
- Real inputs inventoried: `longmemeval_m.json` 2,737,100,077 B local (converted
  500-question full set + dev5 present); a real run needs a live DGX window (driver not
  in `bench/`, judge needs `--allow-network` + keys; the old 779-entry judge cache is
  version-incompatible). No LME-M full-run shards exist anywhere.

## 5. Not run (with reasons)

1. Canary-4 strict-L1 re-run — PENDING: quiet GPU window + root decision (§2).
2. Serving apply of the pooled map — NOT-RUN by design: root decision; artifact stays on
   disk (§2).
3. First live bench campaign (fixed N=64) — NOT-RUN: prerequisite canary-4 strict pass;
   the fixed manifest inventory is not yet minted.
4. Live ladder re-send dynamics (`attempts[].distance`) — NOT-RUN: no ladder-enabled live
   turn exists yet.
5. Real LME-M driver + judge runs (any family) — NOT-RUN: DGX-gated live work.
6. PR #221 build gates in the recording session — NOT-RUN: offline session, cargo is
   root-managed; the ladder binary exists (sha256 `73d79a21…`) with both-mode offline
   selftests green (retained 1.42 s, legacy 1.51 s, exit 0 both).
7. PR #222 merge — open upstream (maintainer merges); the staged in-container pool tool
   (sha256 `04055f2e…`) already equals the reviewed fix and produced the pooled corpus.
8. Controls B-E, P8 proof chain, pilot re-runs, 64-task battery, S17 activation, S18 —
   NOT-RUN: arm-A stop rule (controls) and offline session (the rest).

Known-issue notes (measured): the DGX binding returned rc 255 twice at ~11:41Z and
self-recovered at ~11:44Z during the pooled window; no user sign-in was involved. The
PR #213 unpinned `receipt_invalid` failures and the API-key rotation follow-up carry
forward unchanged (LIVE-RESULTS.md known issues).
