# OCC-3 PR #215 — Live Results

Status 2026-11-01. This file records the live (DGX) evidence state for PR #215. All
offline gates are green (see REGRESSION-RESULTS.md). Every live item below is NOT-RUN;
no live number exists yet. Nothing in this file is projected.

## Offline gate summary (this session, all green)

| Gate | Result |
|------|--------|
| `git diff --check` | clean |
| `scripts/check.sh` | pass (295 python tests, 12 skipped) |
| `cargo check --locked --workspace --all-targets` | pass |
| pi-ai `source_capture` / `inference_attachment` | 10/10, 12/12 |
| 6 occ3 test binaries (62 tests) | pinned 62/62; unpinned 48/62 (14 failures, all `receipt_invalid`, pre-existing PR #213 fixture issue) |
| `occ3_source_handoff` | pinned 11/11; unpinned 4/11 (same known issue) |
| `--lib occ3` | 86 passed, 2 ignored |
| extension pytest | 1986 passed, 449 skipped, 2 xfailed, 0 failed |
| `test_retained_lifecycle.py` + `test_v3_native_validator.py` | 179 passed, 2 skipped |
| phase-suite selftest | retained exit 0; legacy exit 0 (byte-compat) |

## Live items

| Item | Status |
|------|--------|
| Live DGX canary (two independent sessions, fresh private value, full-span capture through `publish_retained` to receiver Consumed receipt) | NOT-RUN |
| P8 proof chain in retained mode | NOT-RUN |
| Pilot cases f2-03, f3-01, f3-02, f4-01, f4-02 (re-run) | NOT RE-RUN |
| 64-task battery | NOT-RUN |
| Normal-runtime activation (user models.json occ3 block + occ3Enabled/occ3Experimental) | NOT configured |
| Benchmarks | NOT measured |
| OCC3 light (live, receipt-backed) | NOT-RUN (offline fixtures only) |
| DGX environment | untouched; healthy calibrated candidate: neo-glm53, image d983f301, identity efef608a, 8/8 ranks |

## Canary protocol that WILL be run (campaign spec section 12, summarized)

1. Use two independent enrolled source/receiver sessions with fresh one-shot handles; never replay a consumed handle between arms.
2. The private value must be fresh, short, and independent of task ID, request ID, filename, handle, artifact name and routing metadata.
3. Arm semantic-content capture BEFORE the source request.
4. Generate the content WITHOUT the handoff tool.
5. Verify accepted content and full value coverage; verify CapturedRetained and its original receipt.
6. Publish that exact retained state through the authorised route (`publish_retained`).
7. Verify receiver admission, real model consumption, and the completed-use (Consumed) receipt.
8. Read the receiver's fresh final answer and require exact recovery of the private value.
9. Verify state retirement and bounded resource cleanup.
10. On failure, identify the earliest failed invariant (do not jump to model calibration); then run the four controls: different private value, wrong-sender state, matched-shape null state, no-attachment baseline.

Execution constraints recorded by root: the canary fires only after all GLM workers
are released, through a single executor, never in parallel with other live work.

## Canary results

Not started. Rows will be appended here only after the run, with measured values or
NOT-RUN; a successful content-only capture is not counted as a successful handoff
until publication and consumption complete.
