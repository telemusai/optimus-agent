# OCC-3 PR #215 — Final Report (milestone update)

Date 2026-11-01. Branch `fix/occ3-glm53-full-message-recovery`, worktree
`C:/Users/openclawuser/optimus-occ3-icalib-20261028`, base `91bfde04f`, tip `c4d310929`.

This is an honest milestone update to PR #215, NOT the final completion report. The
repair and its offline verification are done; the live chain has not been run, so the
campaign's live acceptance criteria are all unmeasured. No number below is projected.

## Status vs the completion contract

| Contract area | Status |
|---------------|--------|
| Lifecycle repair (retained content-only capture + orchestrator publication) | DONE (commits `7cccaae4d..c4d310929`, offline gates green) |
| Harness repairs (7.1 suite error propagation, 7.2 answer-position snapshot, 7.3 source wire effort) | DONE (in `b731a0736`, phase-suite selftests green both modes) |
| Offline gates (10/10) | DONE (see REGRESSION-RESULTS.md) |
| Live DGX canary (section 12) | PENDING — NOT-RUN |
| P8 proof chain in retained mode (section 13) | PENDING — NOT-RUN |
| Five pending pilot cases (f2-03, f3-01, f3-02, f4-01, f4-02) re-run + coherent 16-case set | PENDING — NOT RE-RUN |
| 64-task battery (section 14) | PENDING — NOT-RUN |
| Normal-runtime activation (section 17) | PENDING — NOT configured |
| Benchmarks (section 19) | PENDING — NOT measured |
| OCC3 light (live, receipt-backed) | PENDING — offline fixtures only |

## Measured evidence (offline, this session)

- `git diff --check` clean; `scripts/check.sh` pass (295 python tests, 12 skipped);
  `cargo check --locked --workspace --all-targets` pass.
- pi-ai `source_capture` 10/10, `inference_attachment` 12/12.
- Six occ3 binaries: pinned 62/62 GREEN; unpinned 48/62 with 14 failures, all
  `occ3Cause=receipt_invalid` (pre-existing PR #213 v1-receipt fixture issue; every
  affected test green under `OCC3_SPAN_POLICY=whole_completed_response_tail_v1`).
- `occ3_source_handoff`: pinned 11/11, unpinned 4/11 (same known issue).
- Lib `occ3`: 86 passed, 2 ignored. Extension pytest: 1986 passed, 449 skipped,
  2 xfailed, 0 failed. Retained lifecycle + v3 validator: 179 passed, 2 skipped.
- Phase-suite selftest: retained mode exit 0 (`retained_source_leg` `case_passed=true`,
  capture v3 + mode + policy + publish.v2 + `tool_call_id_absent=true`); legacy mode
  exit 0 (byte-compat).
- New tests: Rust `occ3_retained_lifecycle` (7), Python `test_retained_lifecycle.py`
  (35), validator 5-tuple admission test.
- DGX untouched and healthy: calibrated candidate neo-glm53, image d983f301,
  identity efef608a, 8/8 ranks.

## What remains, in order

1. Live DGX canary (section 12): two independent enrolled sessions, fresh private
   value independent of all metadata, full-span capture through `publish_retained` to
   a receiver Consumed receipt, exact recovery required, then the four controls
   (different value, wrong-sender, matched-shape null, no-attachment).
2. P8 proof chain in retained mode (section 13): current client binary, current
   candidate image, correct method/artifact pins, genuine runtime observations,
   correct receipts, atomic evidence publication, restart/stale-handle controls.
3. Re-run the five pending pilot cases (f2-03, f3-01, f3-02, f4-01, f4-02) plus
   affected earlier cases; produce one coherent 16-case result set; keep old attempts
   separate (no merged artificial perfect run).
4. 64-task battery (section 14).
5. Normal-runtime activation (section 17): user models.json occ3 block +
   occ3Enabled/occ3Experimental for the compatible pair, real parent/subagent family,
   receipt-backed OCC3 light, no uncertain native-to-text replay.
6. Benchmarks (section 19) with the listed latency/size/failure metrics, one-time
   calibration cost separated from per-message cost.
7. OCC3 light verification live (receipt-backed indicator behavior: no pulse for
   pending/failed/uncertain use, no duplicate pulse after reconnect, no
   unrelated-family pulse, restart invalidates stale state).

## Known issues carried forward

- Pre-existing `receipt_invalid` unpinned failures (PR #213 v1-receipt fixtures vs v2
  default arming): 4 binaries / 21 tests, all green under the v1 pin, documented in
  REGRESSION-RESULTS.md. Fixture fix belongs to a PR #213 follow-up.

## Sign-off criteria (from the campaign's final acceptance checklist)

The LIVE, PRODUCT and SUBMISSION blocks of the acceptance checklist are unchecked
until items 1-7 above are measured. PR #215 stays open with measured offline numbers
only; no automatic merge.
