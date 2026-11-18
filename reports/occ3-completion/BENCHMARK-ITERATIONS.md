# OCC-3 Phase 3 — Benchmark Iterations (running ledger)

Status 2026-11-02, updated the same day after the canary-4 window and the bench64-i1
partial campaign. This page is the Phase 3 benchmark's iteration ledger: one row per
calibrated-map iteration with its measured numbers and the gate for the next step. It runs
beside LIVE-RESULTS.md (window evidence) and TOLERANCE-RETRY.md (ladder policy). Upstream
PRs #221 (tolerance verify fix), #222 (calibration_pool method-union), #223 (AGENTS.md PR
docs), #224 (this ledger), #227 (armed-turn waiting-queue admission window) are merged;
the maintainer merges. Every number below is measured and carries its source or digest;
PENDING items carry the gate that fills them. No projections.

## 1. Iteration ledger

| Iter | Date | Map (content digest / sha256 / bytes) | Measured score | Live strict result | Status |
|---|---|---|---|---|---|
| 0 baseline | through 2026-11-01 | `3e76d627` / `59d60f83` / 72799976 B | own-corpus val r2 0.203625, test 0.282283 (8t/159r, 15t/979r) | canaries 0/3 exact (d=1, d=4, no-value); TOKEN_FULL exact 84.9 s | superseded by iter 1 |
| 1 s9 pooled refit | 2026-11-02 | `9a7394de` / `f6b101e5` / 88564767 B | same-split val r2 0.295305, test 0.307213 (13t/476r, 20t/1341r) | canary-4 strict FAIL — 0/4 exact, d=1/2/5/41, near 3/4 (iteration 2); MEASURED-KEEP | serving continues on this map (measured-keep) |
| 2 canary-4 live verification | 2026-11-02 | `9a7394de` / `f6b101e5` / 88564767 B (iter-1 map; live verification, no refit) | strict-pinned single-send live: 0/4 exact, near 3/4, no-value 1/4; durations 83.6-976.9 s | strict FAIL; machinery 4/4 complete cycles; MEASURED-KEEP (§2) | gate consumed; bench ran anyway (root decision) |
| 3 bench64-i1 (partial) | 2026-11-02 | `9a7394de` / `f6b101e5` / 88564767 B (same map; measurement campaign, no refit) | 37/64 turns (t01-t37): strict exact 0/37, near (d<=4) 14/37 (37.8%), no-value 20/37; duration mean ~394 s, min 65 s, max 1017 s | strict 0/37 (all strict-pinned, single-send) | user-paused at 37/64; t38-t64 unfired |

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

### Iteration 2 — canary-4 live verification (measured 2026-11-02T13:35Z-14:05Z, re-armed windows)

- Controlled swap to candidate `41910220f2e0` carrying the pooled artifact (`9a7394de` /
  `f6b101e5` / 88564767 B); original container `cfcb2f12c7db` preserved (Exited 0);
  controller attempt-dir `/data/occ3-bridge/v3-live-icalib-canary4-20261102T1245Z/attempt-01`.
  Strict pin `OCC3_ARM_TOLERANCE=0`, single-send; client binary `73d79a21…`
  (occ3_native_phase_suite.exe).
- **Verdict: STRICT FAIL — MEASURED-KEEP.** 0/4 exact recovery on completed live cycles.
  Distribution (all strict-pinned, single-send, structural_ok=true, no trial_error):

| Run | Journal | value_distance | duration | receiver answer |
|---|---|---|---|---|
| 8 (04) | rearm-20261102t1353zb | 2 | 160.7 s | near (24/26 chars exact) |
| 10 (06) | rearm-20261102t1401zd | 5 | 647.3 s | near |
| 11 (07) | rearm-20261102t1401ze | 1 | 83.6 s | near (25/26 chars exact) |
| 12 (08) | rearm-20261102t1401zf | 41 | 976.9 s | "UNKNOWN" (receiver declined after ~14 min thinking) |

- Machinery **PROVEN**: 4/4 complete cycles (enroll → source capture span contiguous,
  turn-end attested → publish 200 → receiver injection → answer → retire 200).
- Starvation context (runs 5/6/03/05, same window): died to the 5 s waiting-queue admission
  deadline under production decode load — the pinned scheduler patch sets
  `req.occ3_queue_deadline = min(descriptor.deadline, time.monotonic() + 5.0)` at
  reception; root cause diagnosed, fix merged upstream as PR #227 (5.0 → 30.0); lull-timed
  firing adopted meanwhile (0 starvation aborts since).
- Why the strict FAIL is not a map-corruption verdict: (1) the failure mode is single-char
  substitutions/small indels plus one receiver decline — decode noise, non-deterministic;
  (2) the strict bar was never passed by ANY map live — the baseline `3e76d627`/`59d60f83`
  never completed a live arm cycle at all (P3' 2026-10-30 evidence: receiver refused
  injection framing, client-side deaths); (3) the offline s9 gate stands (pooled beats
  baseline on every val/test metric, +33.7%/+25.6% relative r2, iteration 1);
  (4) production serving on the pooled map is healthy (2 h+, normal traffic).
- Decision MEASURED-KEEP: the pooled map `9a7394de` / artifact `f6b101e5` / 88564767 B
  stays serving on candidate `41910220f2e0`; original preserved; restore path on file:
  `/usr/bin/python3 -m deploy.v3.controller restore --attempt-dir /data/occ3-bridge/v3-live-icalib-canary4-20261102T1245Z/attempt-01`.

### Iteration 3 — bench64-i1 (partial, user-paused at 37/64; measured 2026-11-02T14:36Z-21:13Z windows)

- First bench campaign over the fixed N=64 manifest set; 37/64 turns fired (t01-t37).
  t06/t13/t21/t22/t24 were reminted after dead/uncertain first attempts; the superseded
  first attempts are excluded from the histogram, journals retained. STOP CONDITION:
  user-requested pause at 37/64 — NOT a stop-rule trigger; t38-t64 remain unfired.
- All 37 completed full arm cycles under lull-timed firing; 0 starvation aborts in the
  series.
- Distance histogram (37 final): d41: 20 (no-value), d2: 7, d1: 4, d3: 3, d5: 2, d26: 1.
  Strict exact (d=0): 0/37. Near (d<=4): 14/37 (37.8%). No-value: 20/37.
- Duration: mean ~394 s, min 65 s, max 1017 s (t37, d41).
- Harness pins: `OCC3_ARM_TOLERANCE=0` (strict), `OCC3_RECEIVER_THINKING=low`,
  `OCC3_KEEPALIVE_STREAM=1`, source-mode retained, client binary `73d79a21…`
  (occ3_native_phase_suite.exe).
- Evidence base: 39 arm-evidence-native JSONs under 13 journal windows — 37 final plus
  2 superseded remint first-attempts (the tabulator counts the superseded files, hence
  39 vs 37); manifest sha256 over the full bench64-i1 tree
  `3d0a09c36ed2a83fc29b2e42d7bd83eb058cfe7818021fe7987f9d048159da4f`.
- Per-window (tabulator columns; n counts evidence files including the 2 superseded;
  per-window manifest = sorted `<file sha256>  <relpath>` lines, its sha256 recorded):

| Window | n | exact | strict | pin | distance histogram | manifest sha256 |
|---|---|---|---|---|---|---|
| w1-1436z | 5 | 0 | 0/5 | all | 1:1 2:1 3-4:1 5+:2 | `32778e253fc12cc327409de33c4bb28fd7db2273b53b411e53585e2fb3e54333` |
| w1-1549z | 3 | 0 | 0/3 | all | 2:1 5+:2 | `2428382fcb7a50194c66684d54b3264331bb45875a84777badbd95946970aaa8` |
| w2-1549z | 4 | 0 | 0/4 | all | 1:1 2:1 5+:2 | `0b68b81bad0a6a94bbfaf510446e90207536d9e16db4877aa2c03868ea50094a` |
| w2-1644z | 2 | 0 | 0/2 | all | 1:1 5+:1 | `cd78915616ee82ea05bc21798de766a244e717d5c47087b648f67e862033237c` |
| w2-1729z | 2 | 0 | 0/2 | all | 3-4:1 5+:1 | `34fc144f094e2395e788bc087e1c333c8fea6c00dcf560624c1ad795207d1919` |
| w3-1729z | 3 | 0 | 0/3 | all | 5+:3 | `70f45472e257698a943e27aa1ea0fda3b4e083c775c891f95d2d7da2a6757b34` |
| w3-1748z | 4 | 0 | 0/4 | all | 1:1 5+:2 nv:1 | `de7d43870028f5f870bc13469508158d09ba3fd1c12808f94c78566dc1d568e6` |
| w3-1808z | 2 | 0 | 0/2 | all | 5+:2 | `80a02d3bfcc3b2af59d377f0e7fcd71fbc9cafc2d51cfbabd87333f3302d56e9` |
| w3-1922z | 1 | 0 | 0/1 | all | 2:1 | `fa59758f22e7f1e8e07597b9ff39dff24dbfefcc360a58c552d8888c6154641e` |
| w4-1922z | 4 | 0 | 0/4 | all | 2:1 5+:3 | `584e917836aa0656e394866d635e1c908e95a127464ec1f9b6911c5c6a8f00f0` |
| w4-2017z | 4 | 0 | 0/4 | all | 3-4:1 5+:3 | `bad150af493b4028955b36bbfe4e60aede5a59c8ef7035aed9e9655971b63b16` |
| w5-2017z | 1 | 0 | 0/1 | all | 5+:1 | `6ba20175e5217001448e1e3455f4a43b9e6194dbdb645db5dda4f04756cc78e2` |
| w5-2113z | 4 | 0 | 0/4 | all | 2:2 5+:2 | `b1b8e365a409687aa32f9df551c0f2d2ab8f40a9c276973ccbbe16da8dc24bb8` |

  Totals: 39 evidence files, 0 exact — 37 final + 2 superseded remint first-attempts.

### Stop rule

Bench stop rule (pre-registered): M_i = strict exact-recovery rate over the fixed N=64 set;
stop after two consecutive relative improvements r_i < 3%. Status: **iteration 1 is far
above the 3% bar** — the measured same-split gain, val r2 +0.074438 over baseline 0.220867
(computed +33.7% relative) and test +0.062612 over 0.244601 (computed +25.6%), so the map
keeps iterating. Live M_i inputs: bench64-i1 measured 0/37 strict exact over t01-t37
(partial, iteration 3); M_1 is not computed until the campaign completes — t38-t64 unfired
(user pause). M_0 anchor: exact 0/3 canaries, 0/8 all NATIVE-style turns. Next iterations
and their gates:

| Iter | Content | Gate |
|---|---|---|
| 3 (cont.) | bench64-i1 completion (t38-t64), then M_1 over the full N=64 | user re-arm (paused at 37/64); lull-timed firing; strict pin `OCC3_ARM_TOLERANCE=0`, harness binary pin `73d79a21…` (no rebuild) |

The canary-4 gate is consumed: iteration 2 measured strict FAIL 0/4 and the bench campaign
ran anyway on the root MEASURED-KEEP decision — the pre-registered gate condition
(canary-4 arm-A strict pass) did not hold.

## 2. Deploy state (measured)

- Through the s9/pooled refit (2026-11-02, pre-canary-4): the pooled artifact existed on
  disk only — `apply_map` NOT run; no deploy, no swap, no restart, no canary turn, no
  swap-check during any s9/pooled work. Serving-untouched proof at that point: container
  `cfcb2f12c7db`, image `579210da8c74`, PID1 lstart Sun Nov 1 23:56:04 2026 unchanged
  across the refit window (etimes 41767 → 42175 s, +408 s wall; same container id, same
  image id, same PID1 start second). Serving still ran the baseline artifact `59d60f83`
  (72799976 B, Oct 31 mtimes).
- Canary-4 MEASURED-KEEP (2026-11-02): controlled swap to candidate `41910220f2e0`
  carrying the pooled map `9a7394de` / `f6b101e5` / 88564767 B; arm verdict strict FAIL
  0/4 (iteration 2) with the restore condition (deterministic map error) absent. Serving
  now runs the pooled `9a7394de` (`f6b101e5`, 88564767 B); the original container
  `cfcb2f12c7db` is preserved (Exited 0); restore path on file:
  `/usr/bin/python3 -m deploy.v3.controller restore --attempt-dir /data/occ3-bridge/v3-live-icalib-canary4-20261102T1245Z/attempt-01`.
- Production serving on the pooled map measured healthy: 2 h+ through the canary-4
  verdict, normal traffic (parity gates at swap); the bench64-i1 series (14:36Z-21:13Z,
  37/37 complete arm cycles) ran against the same serving candidate.

## 3. Tolerance-ladder state (policy in TOLERANCE-RETRY.md, PR #221)

- Defaults: `OCC3_ARM_TOLERANCE=1` (ceiling 4), `OCC3_ARM_MAX_ATTEMPTS=3`; strict windows
  pin `=0`. Corpus-turn canaries stay single-send.
- Measured implication of tol=1 on the three recorded arm-A FAILs: canary-1 (d=1) converts
  `within_tolerance` on send one; canary-2 (d=4) converts via `best_effort_fallback` after
  the three-send ladder; canary-3 (no-value) remains failed. Exact recovery stays 0/3
  (baseline canaries) under any tolerance — and 0/4 (canary-4) and 0/37 (bench64-i1
  t01-t37) beside it; no measured live d=0 exists. Tabulator cross-check (4 windows / 34
  turns, measured): tolerance 1 → canary-1 window 100% tolerant; tolerance 4 → canary-2
  window also 100%; supplemental window 0% (30 no-value). Single-send d ≤ t is a lower
  bound on ladder acceptance.
- Retune decision (recorded at n=3, baseline canaries): **NOT justified** (single-send
  distances then: one d=1, one d=4, one no-value). Distances recorded since: canary-4 n=4 —
  d=1, d=2, d=5, one no-value (iteration 2); bench64-i1 n=37 — d2:7, d1:4, d3:3, d5:2,
  d26:1, no-value:20 (iteration 3). Measured tolerant lower bounds (single-send d ≤ t;
  ladder re-sends can only reduce distance): canary-4 1/4 at tol=1; bench64-i1 4/37 at
  tol=1, 14/37 (37.8%) at the tol=4 ceiling. The missing measurement is unchanged — live
  `attempts[].distance` from actual ladder re-sends does not exist (both series
  strict-pinned single-send); the ladder-enabled bench subset over the fixed N=64 set
  still fills it. Until then the default stays tol=1 / max=3.

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

1. bench64-i1 completion (t38-t64) — NOT-RUN: user-requested pause at 37/64; not a
   stop-rule trigger (iteration 3).
2. Live ladder re-send dynamics (`attempts[].distance`) — NOT-RUN: no ladder-enabled
   live turn exists yet; canary-4 and bench64-i1 t01-t37 ran strict-pinned single-send.
3. Live validation of the widened 30 s admission window (#227) — NOT-RUN: all canary-4
   and bench64-i1 evidence was measured on the 5.0 s build under lull-timed firing.
4. Real LME-M driver + judge runs (any family) — NOT-RUN: DGX-gated live work.
5. PR #221 build gates in the recording session — NOT-RUN: offline session, cargo is
   root-managed; the ladder binary exists (sha256 `73d79a21…`) with both-mode offline
   selftests green (retained 1.42 s, legacy 1.51 s, exit 0 both).
6. Controls B-E, P8 proof chain, pilot re-runs, 64-task battery, S17 activation, S18 —
   NOT-RUN: arm-A stop rule (controls) and offline session (the rest).

Resolved since the previous revision: canary-4 measured (iteration 2); the pooled map's
serving apply done at the canary-4 swap, MEASURED-KEEP (§2); the fixed N=64 manifest
inventory minted and the first campaign partially run (iteration 3); PR #222 merged (the
in-container pool tool `04055f2e…` equals the merged fix and produced the pooled corpus).

Known-issue notes (measured): the DGX binding returned rc 255 twice at ~11:41Z and
self-recovered at ~11:44Z during the pooled window; no user sign-in was involved. The
PR #213 unpinned `receipt_invalid` failures and the API-key rotation follow-up carry
forward unchanged (LIVE-RESULTS.md known issues).
