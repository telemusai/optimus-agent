# OCC-3 PR #215 — Regression Results (offline)

Gate run of 2026-11-01 (gate-runner, serial execution, single warm target dir
`target-occ3compl`). Branch `fix/occ3-glm53-full-message-recovery` at `c4d310929`
(commits `7cccaae4d..c4d310929` after base `91bfde04f`). "Pinned" means
`OCC3_SPAN_POLICY=whole_completed_response_tail_v1`; "unpinned" means the env unset.

## Gate table

| # | Gate | Result |
|---|------|--------|
| 1 | `git diff --check` | PASS (clean) |
| 2 | `bash scripts/check.sh` | PASS (295 python tests, 12 skipped) |
| 3 | `cargo check --locked --workspace --all-targets` | PASS |
| 4 | pi-ai `source_capture` / `inference_attachment` | PASS (10/10, 12/12) |
| 5 | six occ3 test binaries (62 tests) | unpinned 48/62 (14 fail, all `receipt_invalid`); pinned 62/62 GREEN |
| 6 | `occ3_source_handoff` (11 tests) | unpinned 4/11 (7 fail, all `receipt_invalid`); pinned 11/11 |
| 7 | `cargo test -p pi-coding-agent --lib occ3` | PASS (86 passed, 2 ignored, 0 failed) |
| 8 | `extensions/sglang-occ3` pytest `tests/ -q` | PASS (1986 passed, 449 skipped, 2 xfailed, 0 failed) |
| 9 | `test_retained_lifecycle.py` + `test_v3_native_validator.py` | PASS (179 passed, 2 skipped) |
| 10 | phase-suite selftest | retained PASS + legacy PASS (both exit 0; retained leg `case_passed=true`) |

## Gate 5 per-binary breakdown (six binaries, 62 tests)

Command: `cargo test --locked -p pi-coding-agent --test occ3_retained_lifecycle
--test occ3_native_inference --test occ3_bridge_offline --test occ3_latent_activity
--test occ3_method_identity --test occ3_retained_state_proof`

| Binary | Tests | Unpinned | Unpinned failures | Pinned |
|--------|-------|----------|-------------------|--------|
| occ3_retained_lifecycle | 7 | 7/7 | 0 | 7/7 |
| occ3_native_inference + occ3_bridge_offline | 34 | 34/34 | 0 | 34/34 |
| occ3_latent_activity | 7 | 3/7 | 4 | 7/7 |
| occ3_method_identity | 4 | 2/4 | 2 | 4/4 |
| occ3_retained_state_proof | 10 | 2/10 | 8 | 10/10 |
| Total | 62 | 48/62 | 14 | 62/62 |

## Gate 6: occ3_source_handoff (11 tests, run separately)

Unpinned 4/11 (7 failures, all `occ3Cause=receipt_invalid`); pinned 11/11.

## New-test inventory (this PR)

- Rust `crates/pi-coding-agent/tests/occ3_retained_lifecycle.rs` — 7 tests:
  content-only accept; tool output refused (tombstone); unarmed-then-armed claim
  refused; legacy exactly-one-binding incl zero-tool refusal; one-shot no-republish;
  wrong-recipient refused before publish; mode crossing refused.
- Python `extensions/sglang-occ3/tests/test_retained_lifecycle.py` — 35 tests:
  intent admission, publication modes, registry gates, route v1/v2 parsing, digest
  consistency (both schemas), e2e content-only v3 turn through the real chat adapter.
- Python `test_v3_native_validator.py::test_admission_requires_the_publish_v2_control_interface`
  — 5-tuple admitted, 4-tuple refused `control_schema_version_unknown`.

## Known issue: unpinned receipt_invalid failures (pre-existing, NOT from PR #215)

- Cause: PR #213 did not update the v1-receipt fixtures when v2 arming became the
  default; unpinned runs arm v2 and the legacy fixtures fail receipt validation.
- Blast radius: 4 binaries / 21 tests — occ3_source_handoff 7, occ3_latent_activity 4,
  occ3_method_identity 2, occ3_retained_state_proof 8. Every unpinned failure in the
  gate run carries `occ3Cause=receipt_invalid`; no other failure signature appears.
- Workaround: set `OCC3_SPAN_POLICY=whole_completed_response_tail_v1` (v1 pin). Every
  affected test is green under the pin (62/62 in gate 5, 11/11 in gate 6).
- Pre-existence: confirmed on the current tree and on the baseline (fails identically
  with the PR #215 changes reverted).
- Why not fixed in this PR: the fixture update belongs to a PR #213 follow-up. Fixing
  it here would mix fixture policy into the lifecycle change; the issue is documented
  and flagged instead.

## Environment notes (harness-only, no product code touched)

- One warm target dir (`target-occ3compl`); all cargo runs serial.
- Corrupt registry-crate artifacts left by earlier killed parallel builds were purged
  and rebuilt cleanly before the gate run.
