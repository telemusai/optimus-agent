# OCC-3 PR #215 — Final Report (milestone update, evening)

Date 2026-11-01 (evening update). Branch `fix/occ3-glm53-full-message-recovery`,
worktree `C:/Users/openclawuser/optimus-occ3-icalib-20261028`, campaign commit range
62fa6f414..0f7be46d7 (11 commits). This is an honest milestone update to PR #215,
NOT the final completion report. The repair and its offline verification are done; the
live campaign has run three canary windows and one exact-embedding diagnostic with
measured results, but exact-value recovery is not yet achieved, so the campaign's live
acceptance criteria remain unmet. No number below is projected.

## Status vs the completion contract

| Contract area | Status |
|---------------|--------|
| Lifecycle repair (retained content-only capture + orchestrator publication) | DONE (commits through c4d310929, offline gates green) |
| Harness repairs (7.1 suite error propagation, 7.2 answer-position snapshot, 7.3 source wire effort) | DONE (in `b731a0736`, phase-suite selftests green both modes) |
| Offline gates (10/10) | DONE (see REGRESSION-RESULTS.md; evening red-first additions recorded there) |
| Live DGX canary (section 12) | MEASURED, NOT PASSED — canary-1 FAIL at C (span truncation, fixed 77d57c64c); canary-2 FAIL at C (coverage criterion, fixed 109ab15db, receiver 39/41 chars); canary-3 coverage PROVEN, arm A zero recovery, TOKEN_FULL diagnostic PASS EXACT (84.9 s) |
| Canary controls B-E | NOT fired (stop rule after canary-2 arm A failed) |
| s9 supplemental calibration | STARTED (window icalib-sup-20261101T2003Z); refit PENDING |
| Canary-4 (refit map + re-run) | PENDING — NOT-RUN |
| P8 proof chain in retained mode (section 13) | PENDING — NOT-RUN |
| Five pending pilot cases (f2-03, f3-01, f3-02, f4-01, f4-02) re-run + coherent 16-case set | PENDING — NOT RE-RUN |
| 64-task battery (section 14) | PENDING — NOT-RUN |
| Normal-runtime activation (section 17) | PENDING — NOT configured |
| S18 | PENDING — NOT-RUN |
| Benchmarks (section 19) | PENDING — NOT measured |
| OCC3 light (live, receipt-backed) | PENDING — offline fixtures only |

## Measured evidence (offline, 13:00Z session)

- `git diff --check` clean; `scripts/check.sh` pass (295 python tests, 12 skipped);
  `cargo check --locked --workspace --all-targets` pass.
- pi-ai `source_capture` 10/10, `inference_attachment` 12/12.
- Six occ3 binaries: pinned 62/62 GREEN; unpinned 48/62 with 14 failures, all
  `occ3Cause=receipt_invalid` (pre-existing PR #213 v1-receipt fixture issue).
- `occ3_source_handoff`: pinned 11/11, unpinned 4/11 (same known issue).
- Lib `occ3`: 86 passed, 2 ignored. Extension pytest: 1986 passed, 449 skipped,
  2 xfailed, 0 failed. Retained lifecycle + v3 validator: 179 passed, 2 skipped.
- Phase-suite selftest: retained mode exit 0; legacy mode exit 0 (byte-compat).
- New tests: Rust `occ3_retained_lifecycle` (7), Python `test_retained_lifecycle.py`
  (35), validator 5-tuple admission test; evening: `test_calibration_pool.py` (16).

## Measured evidence (live, evening; digests in LIVE-RESULTS.md)

- Three controlled swaps, identity/artifact/TP unchanged across all: canary-1
  1330Z (candidate 60fc24ad), canary-2 1615Z (33572fd0, image 3731a00c…, tree
  77d57c64c, r16 client 7cd02b9c…), canary-3 1745Z (9110a771, image cff9b321…,
  tree 109ab15db, r17 client 1525df15…); identity efef608a…, artifact
  59d60f83… (72799976 B), TP8 8/8 ranks.
- Canary-1 arm A: FAIL at C — span truncation, 33/34 content rows, parsed answer
  40/41 chars (one deletion at value char 10; the earlier "final char" reading was
  an inference). Fixed red-first in 77d57c64c.
- Canary-2 arm A (summary sha256 d19ca027…): FAIL at C — retained 31 == width 31
  (77d57c64c rule passes) but coverage gated by span_message_truncated (accepted
  902 rows > 128-row window); receiver answer 39/41 chars with two mid-string
  deletions + two substitutions inside retained rows. Fixed red-first in 109ab15db.
- Canary-3 arm A (evidence sha256 f63aedb2…): coverage PROVEN (value_fully_in_span,
  retained 35 == width 35, coverage_complete=true), delivered/consumed/Consumed, but
  zero recovery (parsed_answer UNKNOWN). TOKEN_FULL diagnostic (evidence sha256
  a68cb112…): PASS EXACT — fresh value, 84.9 s, semantic_nonce_match=true. Injection
  chain sound; calibrated map distribution-shifted.
- s9 supplemental collection (window icalib-sup-20261101T2003Z): one-family-per-turn
  correction (server enrollment one-shot per principal+family), host-backed Option A
  corpus, paced 420 s caps under GPU contention, Pomerium expiry worked around via the
  documented loopback tunnel; all four written reopen conditions met.

## What remains, in order

1. Canary-4: pool the legacy + supplemental corpora, full P5 refit of the map,
   redeploy, re-run arm A to exact recovery.
2. Fire the four canary controls (different value, wrong-sender, matched-shape null,
   no-attachment baseline).
3. P8 proof chain in retained mode (section 13).
4. Re-run the five pending pilot cases plus affected earlier cases; one coherent
   16-case result set; keep old attempts separate.
5. 64-task battery (section 14); S18.
6. Normal-runtime activation (section 17) with receipt-backed OCC3 light.
7. Benchmarks (section 19), one-time calibration cost separated from per-message cost.

## Known issues carried forward

- Pre-existing `receipt_invalid` unpinned failures (PR #213 v1-receipt fixtures):
  4 binaries / 21 tests, green under the v1 pin (REGRESSION-RESULTS.md).
- Serving API key exposed once in a docker-logs dump; rotation pending post-campaign.
- validate_descriptors.py stale for supplemental descriptors (METHOD_ID constant +
  v1 manifest shape); flagged, not yet fixed.
- Harness pins: stage plans still carry the r13 pins (160613f7…) while r16/r17
  binaries ran descriptor-self-bound (pin-discrepancy.txt); a future window can mint
  harness-pins-r17.json if plan-level parity is needed.
- Scratch-venv statebridge reflection fixture failure (torch-build-dependent) fails
  identically pre/post the calibration tooling change; not part of this PR.

## Sign-off criteria (from the campaign's final acceptance checklist)

The LIVE, PRODUCT and SUBMISSION blocks of the acceptance checklist are unchecked
until the items above are measured. PR #215 stays open with measured offline numbers
plus the measured live canary/diagnostic numbers; no automatic merge.
