# OCC-3 PR #215 — Current State

Milestone update, 2026-11-01. Branch `fix/occ3-glm53-full-message-recovery`, worktree
`C:/Users/openclawuser/optimus-occ3-icalib-20261028`, base main-merge `91bfde04f`.
All offline milestones are implemented and gated green. The live DGX chain is NOT-RUN.
This is a milestone update, not the final completion report.

## Implemented

- `occ3.capture.v3` retained content-only capture: `capture_mode=content_only_orchestrator_v1`,
  span policy `content_tail_no_tool_v2`.
- Publication through the authenticated family controller via `occ3.publish.v2`,
  `dispatch_mode=orchestrator_v1`. Zero tool bindings required in the captured turn;
  any tool call in that turn fails the mode.
- Single-turn retained workflow (replaces the old r13 two-turn unarmed/armed workflow):
  arm BEFORE the content request, handoff tool not installed for that turn,
  deterministic orchestrator publication after.
- Legacy same-turn tool protocol unchanged and byte-identical on the wire.
- Mode isolation: retained and legacy modes cannot cross in either direction.

## Commit chain (verified read-only, `git log --oneline -5`)

| Commit | Subject | Size |
|--------|---------|------|
| `7cccaae4d` | retained content-only capture mode with orchestrator publication (lifecycle contract) | 16 files, +466/-64 |
| `40a79907f` | red-first retained-lifecycle regression suites | 5 files, +1519/-3 |
| `b731a0736` | phase-suite retained source leg and harness repairs (7.1/7.2/7.3, validator 5-tuple, changelog) | 6 files, +823/-142 |
| `c4d310929` | stage-3 native profile input for the calibrated candidate | 1 file, +1 |

## Gates (gate-runner, this session, all green)

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

## New tests

- Rust `occ3_retained_lifecycle`: 7 tests.
- Python `test_retained_lifecycle.py`: 35 tests.
- Validator: `test_admission_requires_the_publish_v2_control_interface` (5-tuple
  admitted, 4-tuple refused `control_schema_version_unknown`).

## Remaining (NOT-RUN)

- Live DGX canary: two independent sessions, fresh private value, full-span capture
  through `publish_retained` to a receiver Consumed receipt. NOT-RUN.
- P8 proof chain in retained mode. NOT-RUN.
- Five pending pilot cases (f2-03, f3-01, f3-02, f4-01, f4-02). NOT RE-RUN.
- 64-task battery. NOT-RUN.
- Normal-runtime activation (user models.json occ3 block + occ3Enabled/occ3Experimental).
  NOT configured.
- OCC3 light: verified only via offline fixtures; live verification NOT-RUN.
- Benchmarks. NOT measured.
- DGX: healthy on the calibrated candidate (neo-glm53, image d983f301, identity
  efef608a, 8/8 ranks), untouched this session.
