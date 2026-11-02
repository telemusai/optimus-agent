# OCC-3 PR #215 — Current State

Milestone update, 2026-11-01 (evening). Branch `fix/occ3-glm53-full-message-recovery`,
worktree `C:/Users/openclawuser/optimus-occ3-icalib-20261028`. All offline milestones
are implemented and gated green; the live campaign has now run three controlled-swap
canary windows plus one exact-embedding diagnostic (measured values in
LIVE-RESULTS.md). Exact-value recovery is NOT yet achieved, so the campaign remains
open. This is a milestone update, not the final completion report.

## Implemented

- `occ3.capture.v3` retained content-only capture: `capture_mode=content_only_orchestrator_v1`,
  span policy `content_tail_no_tool_v2`.
- Publication through the authenticated family controller via `occ3.publish.v2`,
  `dispatch_mode=orchestrator_v1`. Zero tool bindings required in the captured turn.
- Single-turn retained workflow (replaces the old r13 two-turn unarmed/armed workflow):
  arm BEFORE the content request, handoff tool not installed for that turn,
  deterministic orchestrator publication after.
- Legacy same-turn tool protocol unchanged and byte-identical on the wire.
- Mode isolation: retained and legacy modes cannot cross in either direction.
- Live fixes from the campaign (each verified red-first): 77d57c64c span/turn-end
  classification so the stop row stays a content row (canary-1 invariant C);
  109ab15db coverage criterion for tail windows so a complete content run inside a
  128-row retention window passes (canary-2 invariant C).
- Supplemental-calibration tooling 0f7be46d7: host-backed Option A capture dir
  wiring, one-family-per-turn corpus design, offline pool tool
  (`deploy/v3/calibration_pool.py`) with 16 new tests (see REGRESSION-RESULTS.md).

## Commit chain (root-recorded; branch range 62fa6f414..0f7be46d7, 11 commits)

| Commit | Subject |
|--------|---------|
| `77d57c64c` | span fix: turn-end stop row classified as content; fail-closed retained==width rule (canary-1 C-invariant, red-first) |
| `109ab15db` | coverage criterion fix for tail windows (canary-2 C-invariant, red-first) |
| `0f7be46d7` | supplemental-calibration tooling (Option A capture dir, one-family-per-turn corpus, offline pool tool + tests) |
| (range total) | 11 commits also covering the retained lifecycle contract and the retained task-route; r17 client sha256 1525df15… |

The earlier documented chain 7cccaae4d..c4d310929 (13:00Z milestone) sits on the same
branch; the range above is the current campaign record.

## Gates (gate-runner, 13:00Z session, all green)

| Gate | Result |
|------|--------|
| `git diff --check` | clean |
| `scripts/check.sh` | pass (295 python tests, 12 skipped) |
| `cargo check --locked --workspace --all-targets` | pass |
| pi-ai `source_capture` / `inference_attachment` | 10/10, 12/12 |
| 6 occ3 test binaries (62 tests) | pinned 62/62 GREEN; unpinned 48/62, 14 failures all `occ3Cause=receipt_invalid` (pre-existing PR #213 fixture issue) |
| `occ3_source_handoff` (11 tests) | pinned 11/11; unpinned 4/11 (same known issue) |
| `cargo test -p pi-coding-agent --lib occ3` | 86 passed, 2 ignored |
| extension pytest `tests/` | 1986 passed, 449 skipped, 2 xfailed, 0 failed |
| `test_retained_lifecycle.py` + `test_v3_native_validator.py` | 179 passed, 2 skipped |
| phase-suite selftest | retained mode exit 0; legacy mode exit 0 (byte-compat) |

Evening additions (offline, measured): calibration-pool red-first evidence and the
full-suite runs are recorded in REGRESSION-RESULTS.md.

## Live campaign state (measured; details and digests in LIVE-RESULTS.md)

- Three controlled swaps, all with identity efef608a / artifact 59d60f83 (72799976 B)
  / TP8 unchanged: canary-1 (1330Z, candidate 60fc24ad), canary-2 (1615Z, 33572fd0),
  canary-3 (1745Z, 9110a771, image cff9b321…).
- Canary-1: arm A FAIL at C — span truncation (33/34 content rows; parsed answer
  40/41 chars); fixed red-first in 77d57c64c.
- Canary-2: arm A FAIL at C — coverage criterion wrong for tail windows (31/31
  retained but gated by span_message_truncated; receiver answer 39/41 chars);
  fixed red-first in 109ab15db. Controls B-E not fired (stop rule).
- Canary-3: coverage PROVEN (value_fully_in_span, retained 35 == width 35); arm A
  zero recovery; TOKEN_FULL exact-embedding diagnostic PASSED EXACT (fresh value,
  84.9 s) — injection chain sound, calibrated map distribution-shifted.
- s9 supplemental calibration STARTED (window icalib-sup-20261101T2003Z):
  one-family-per-turn correction, host-backed Option A corpus, paced 420 s caps
  under GPU contention, Pomerium expiry worked around via the documented loopback
  tunnel. All four written reopen conditions met.

## Remaining (NOT-RUN / PENDING)

- Canary-4: pooled refit of the calibrated map, then re-run. PENDING.
- Canary controls B-E (different value, wrong-sender, matched-shape null,
  no-attachment). NOT fired.
- P8 proof chain in retained mode. NOT-RUN.
- Five pending pilot cases (f2-03, f3-01, f3-02, f4-01, f4-02) + one coherent
  16-case set. NOT RE-RUN.
- 64-task battery. NOT-RUN.
- Normal-runtime activation (S17: user models.json occ3 block +
  occ3Enabled/occ3Experimental). NOT configured.
- S18. NOT-RUN.
- OCC3 light: verified only via offline fixtures; live verification NOT-RUN.
- Benchmarks. NOT measured.
- Known-issue follow-ups: serving API key rotation (exposed once in a docker-logs
  dump; post-campaign), stale validate_descriptors.py (METHOD_ID constant + v1
  manifest shape), PR #213 receipt fixtures (21 unpinned failures).
