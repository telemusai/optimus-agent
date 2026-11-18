# OCC-3 Benchmark & Optimisation Campaign — occ3-bench-opt-20261109

Repository reports directory for the campaign. Worktree:
`c:/users/openclawuser/optimus-occ3-bench-opt-20261109`, branch
`perf/occ3-live-benchmark-optimisation` (HEAD `c1790003f`). Campaign evidence workspace:
`occ3-bench-opt-20261109/` next to this repo's profile workspace (ledger, probes, worker
briefs).

## What this campaign measures

End-to-end task accuracy, protocol fidelity, reliability and wire cost of the four
production communication channels — **TXT / OCC-1 / OCC-2 / OCC-3-native** — between two
real daemon sessions on the SAME model (`dgx-glm53-occ3/glm-5.3`), on the 50-task dev
matrix (first 10 per family A-E, seed 42), through production seams only (no synthetic
backends, no prompt-only stand-ins). The adapter orchestrates real sessions and reads
durable delivery records; every measured byte travels a production path.

## Document index

| File | Contents |
|---|---|
| `BASELINE.md` | Baseline manifest: client/server/model/alignment identities, alpha and capacity policy (from BASELINE-MANIFEST.json) |
| `PROTOCOL.md` | The three evaluation tracks, the four arms, fairness and isolation rules, verdict taxonomy, fail-the-run guards |
| `ITERATIONS.md` | Append-only iteration log: commits, smoke runs 1-5, root-caused defects and fix status, NOT-RUN list |

## Current status (2026-11-10, honest)

- **Baseline: COMPLETE** — stage-1 live probe 10/10 receipts (2026-11-09T13:36Z); client
  ee2b9e56 (0.1.44), server image 9728d909, runtime 79019420 gen 1 TP8 8/8, calibration
  9a7394de, ordinary completion 133 ms. See BASELINE.md.
- **Harness bring-up: IN PROGRESS** — four-way adapter implemented and offline-validated
  (selftest 36/36); five live txt-arm smoke runs so far, each exposing and then fixing a
  harness defect (daemon-socket wiring `d77689a7a`, model-spec split `da82b6ccf`, launcher
  env, held-parent redesign `c1790003f`). Smoke run 5 still failed 5/5 tasks — all 10/10
  attempts at the parent-spawn DONE-detection stage; two defects root-caused (brittle DONE
  detection; CLI send reaching only roster-active sessions), fixwave1 in flight. See
  ITERATIONS.md defect register.
- **Comparative numbers: NONE YET — NOT-RUN.** No arm has completed a live trial; no
  accuracy, native-success or wire-cost numbers exist for any arm. Historical scores from
  earlier campaigns are reference-only (different gateway/config).
- Twin-guard fix `bc4a6375b` is committed on the branch but the live client build
  (0.1.44) does not carry it yet.

## Next steps (dependency chain, from the ledger)

fixwave1 commit (DONE-detection + child keepalive) -> root verify -> ds-executor re-smoke
(WMI, one at a time) -> pooled four-way matrix (Tracks A/B/C per PROTOCOL.md).
