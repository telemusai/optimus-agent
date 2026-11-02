# OCC-3 PR #215 — Live Results

Status 2026-11-01, evening update. The live DGX campaign RAN: three controlled swaps
(canary-1/2/3), the arm-A canary in each window, one exact-embedding diagnostic arm
(TOKEN_FULL), and the s9 supplemental-calibration collection window. Everything below
is measured from the window evidence; every unfinished item keeps its PENDING / NOT-RUN
marker. Nothing here is projected.

## Offline gate summary (13:00Z gate run, all green; see REGRESSION-RESULTS.md)

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

Tonight's live-fix commits (77d57c64c, 109ab15db) and the supplemental-calibration
tooling (0f7be46d7) carry their own red-first evidence; see REGRESSION-RESULTS.md.

## Live items

| Item | Status |
|------|--------|
| Live DGX canary arm A (main, retained) | MEASURED, 3 windows, all FAIL at exact-value recovery (details below) |
| Canary controls B-E (different value, wrong-sender, matched-shape null, no-attachment) | NOT fired — canary-2 stop rule after arm A failed; task/descriptor substitutions minted, private values recorded sha-only |
| TOKEN_FULL exact-embedding diagnostic (15D) | MEASURED — PASS, exact recovery of a fresh value in 84.9 s |
| s9 supplemental calibration collection | STARTED — window icalib-sup-20261101T2003Z (details below) |
| Canary-4 (pooled refit of the map, re-run) | PENDING |
| P8 proof chain in retained mode | NOT-RUN |
| Pilot cases f2-03, f3-01, f3-02, f4-01, f4-02 (re-run) + 16-case set | NOT RE-RUN |
| 64-task battery | NOT-RUN |
| Normal-runtime activation (S17: user models.json occ3 block + occ3Enabled/occ3Experimental) | NOT configured |
| S18 | NOT-RUN |
| Benchmarks | NOT measured |
| OCC3 light (live, receipt-backed) | NOT-RUN (offline fixtures only) |
| DGX environment | MEASURED each window (table below) |

## The three controlled swaps (window + candidate + image; identity/artifact/TP unchanged)

| Window | Candidate | Image (sha256:) | Server tree | Client binary |
|---|---|---|---|---|
| icalib-canary-20261101T1330Z | 60fc24ad | 4e95ab9d… (running since 14:08:56Z) | pre-fix | not recorded in this evidence set |
| icalib-canary2-20261101T1615Z | 33572fd0 | 3731a00c… | 77d57c64c | r16 sha256 7cd02b9c… (18909696 B) |
| icalib-canary3-20261101T1745Z | 9110a771 | cff9b321… | 109ab15db | r17 sha256 1525df15… |

Unchanged across all three windows (measured identity receipts): model fingerprint
sha256:efef608a…, method `statebridge.glm53.final-prenorm-predictor.calibrated-dual-ridge.v1`,
artifact `calibrated-dual-ridge-3e76d627a1f81f91.bin` sha256 59d60f83… (72799976 B),
TP8 (expected/ready ranks 8/8, tp_layout_digest a55bf3e0…), layer 77, hidden 6144, bfloat16.

## Canary results (measured; window + digest cited per row)

| Arm | Window / evidence | Verdict | Earliest failed invariant | Measured detail |
|---|---|---|---|---|
| A main, canary-1 | icalib-canary-20261101T1330Z, candidate 60fc24ad (numbers from the canary-2 span-selection diagnostic) | FAIL | C — span truncation at the turn-end boundary | 33 of 34 accepted content rows retained (the stop row fell out of the content run); parsed answer 40 of 41 chars — one deletion measured at value char 10 (the earlier "final char" reading was an inference). Fixed red-first in 77d57c64c. |
| A main, canary-2 | icalib-canary2-20261101T1615Z, candidate 33572fd0, tree 77d57c64c, r16 7cd02b9c…, arm-summary sha256 d19ca027… | FAIL | C — coverage criterion wrong for tail windows | retained 31 == accepted width 31 (the 77d57c64c fail-closed rule PASSES); coverage_complete=false purely from span_message_truncated (accepted_total 902 > the 128-row retention window, so any longer turn could never pass); receiver answer 39/41 chars — two mid-string deletions + two substitutions, all inside retained rows. Fixed red-first in 109ab15db. Controls B-E not fired (stop rule). Arm 729 s. |
| A main, canary-3 | icalib-canary3-20261101T1745Z, candidate 9110a771, image cff9b321…, tree 109ab15db, r17 1525df15…, evidence sha256 f63aedb2… | FAIL (zero recovery) | G — exact-value recovery | Coverage PROVEN: value_fully_in_span=true, retained 35 == width 35, coverage_complete=true, value_in_sender_content=true; delivered + consumed + lease Consumed, receiver request_count 1, receiver_tool_calls 0; parsed_answer UNKNOWN, semantic_nonce_match=false. Arm 657 s. |
| TOKEN_FULL diagnostic | same window, evidence sha256 a68cb112…, trial token-full-22241d64… | PASS | — | Exact-embedding arm (control token `texact`, hook-selected, readback-confirmed) recovered a FRESH private value EXACTLY: exact_match=true, semantic_nonce_match=true, coverage gate green, arm 84993 ms (~84.9 s), fresh value minted for this arm. |

Interpretation (measured chain): with coverage proven and the exact-embedding arm
passing, the injection chain is sound end-to-end; the arm-A zero recovery isolates the
defect to the calibrated map (distribution-shifted), which is what opened the s9
supplemental calibration. A content-only capture is still not counted as a successful
handoff; no arm-A window has yet produced exact recovery.

Concurrent-load lines (measured, read-only docker-logs sampling): canary-2's arm shared
the GPU with one external 118,912-cached-token turn at 16:42:28 plus its resident KV
(the 9-16 req/min burst was 16:56-17:02, AFTER the arm); canary-3 fired at 18:15:10Z
after seven silent minutes (17:36-17:42 at 0 chat/min; trailing 5 min: 10 chat requests,
9 x 200 + the driver's own 409 probe).

## s9 supplemental calibration (started; window icalib-sup-20261101T2003Z)

- Trigger: canary-3 isolated the failure to the calibrated map (TOKEN_FULL exact pass,
  NATIVE zero recovery with coverage proven); all four written reopen conditions met
  (root's campaign record).
- Design correction (measured): server enrollment is one-shot per (principal, family)
  with exact payload equality, so the 6-families-x-4 design was corrected to
  one-family-per-turn (manifest-map-delta.json: 20 new single-family turns u05..u24
  beyond the original t01..t04).
- Host-backed corpus (Option A): capture dir under /data/neo-glm53 (the container's
  only RW bind); corpus files landed (13 recorded in the runner log).
- Pacing: turns gated on health/GPU; caps measured at 240 s and 420 s under GPU
  contention; aborted-at-cap turns and minted spares are recorded in turns.jsonl.
- Pomerium expiry worked around via the documented loopback tunnel (turn endpoint
  http://127.0.0.1:30101/v1).
- Pooled refit + re-validation (canary-4): PENDING.

## Known issues (live campaign)

- The serving API key was exposed once in a docker-logs dump during the campaign;
  rotation is planned post-campaign (root's record; the value is not reproduced here).
- validate_descriptors.py is stale for the supplemental descriptors (hardcoded
  METHOD_ID constant + v1 manifest shape); flagged, NOT yet fixed.
- Harness pins: the stage plan still carries the r13 harness binary pins (160613f7…)
  while the r17 binary actually run (1525df15…) is self-bound by every descriptor and
  re-verified at runtime (pin-discrepancy.txt; re-staging deferred by root's decision).
- Unpinned `receipt_invalid` failures (21 tests) remain a PR #213 fixture follow-up
  (see REGRESSION-RESULTS.md).
