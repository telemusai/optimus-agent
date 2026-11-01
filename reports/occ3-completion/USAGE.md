# OCC-3 PR #215 — Usage (offline verification)

Worktree `C:/Users/openclawuser/optimus-occ3-icalib-20261028`, branch
`fix/occ3-glm53-full-message-recovery`. All commands below are offline.

Build discipline: one warm target dir (`CARGO_TARGET_DIR=target-occ3compl`), cargo runs
serial only. Do not create new target dirs.

## Phase-suite selftest (retained mode)

1. Build: `cargo build -p pi-coding-agent --example occ3_native_phase_suite`
2. Run retained mode (exit 0 expected):
   `target-occ3compl/debug/examples/occ3_native_phase_suite.exe --source-mode retained --offline <fresh_dir1>`
   - Expected evidence: `retained_source_leg` `case_passed=true`; capture `occ3.capture.v3`
     + `capture_mode` + span policy + `occ3.publish.v2`; `tool_call_id_absent=true`.
3. Run legacy mode for byte-compat (exit 0 expected):
   `target-occ3compl/debug/examples/occ3_native_phase_suite.exe --offline <fresh_dir2>`

General form:
`occ3_native_phase_suite [--arm ARM] [--source-mode legacy|retained] [--corpus-turn TURN_MANIFEST_JSON] [--task TASK_JSON] [--offline [DIR] | --live PRIVATE_DESCRIPTOR_PATH]`

Use a fresh empty directory for every `--offline` run.

## Lifecycle tests

- Rust (7 tests): `cargo test --locked -p pi-coding-agent --test occ3_retained_lifecycle`
- Six-binary gate (62 tests):
  `cargo test --locked -p pi-coding-agent --test occ3_retained_lifecycle --test occ3_native_inference --test occ3_bridge_offline --test occ3_latent_activity --test occ3_method_identity --test occ3_retained_state_proof`
- `occ3_source_handoff` (11 tests; unpinned run shows the known receipt_invalid
  failures, pinned run is 11/11):
  `cargo test --locked -p pi-coding-agent --test occ3_source_handoff`
- Lib filter (86 passed, 2 ignored): `cargo test --locked -p pi-coding-agent --lib occ3`

## Python tests

From `extensions/sglang-occ3`:

- Full suite: `python -m pytest tests/ -q` (1986 passed, 449 skipped, 2 xfailed)
- Focused (179 passed, 2 skipped):
  `python -m pytest tests/test_retained_lifecycle.py tests/test_v3_native_validator.py -q`

## Workspace checks

- `git diff --check`
- `bash scripts/check.sh` (295 python tests, 12 skipped)
- `cargo check --locked --workspace --all-targets`

## Environment variables

- `OCC3_SOURCE_THINKING=low|high|max` — rewrites `reasoning_effort` on the source
  wire; validated at startup like `OCC3_RECEIVER_THINKING` (the 7.3 repair). Unset
  keeps source requests byte-identical.
- `OCC3_DUMP_SOURCE_PAYLOAD=<path>` — writes the serialized source request body AFTER
  any mutation; the audit records whether the body carried the applied effort. With
  both this and `OCC3_SOURCE_THINKING` unset, no hook is installed.
- `OCC3_SPAN_POLICY` — operator override for the armed span policy; unset arms the v2
  default (`content_tail_v2`). Admitted names: `whole_completed_response_tail_v1` (v1
  pin) and the v2 family `content_tail_no_tool_v2`, `content_tail_v2`,
  `content_tail_last64_v2`, `content_tail_last32_v2`. Unknown names are refused, not
  defaulted. The v1 pin is the documented workaround for the pre-existing
  `receipt_invalid` fixture issue (see REGRESSION-RESULTS.md).

## User configuration — NOT yet live (pending live validation)

The following is planned but NOT configured and NOT validated live. Do not enable in a
real user setup before the live canary, P8 chain and pilot re-runs pass:

- User `models.json` occ3 block plus the existing experimental OCC-3 settings
  (`occ3Enabled` / `occ3Experimental`) for the specific compatible GLM-5.3 endpoint.
- Campaign spec section 17 requires for activation: back up changed config files only;
  enable only the explicitly configured compatible pair (no global advertisement);
  bind the selected method and full artifact digest; perform the live
  readiness/identity handshake; create a real parent/subagent family; run a normal
  permitted handoff through the repaired lifecycle; verify receiver use and answer;
  verify the receipt-backed OCC3 light; keep the states configured-enabled,
  live-ready, native-delivery-completed and semantically-qualified distinct.
- No hardcoded DGX endpoints or credentials in defaults; secrets stay in the existing
  configuration/secret mechanism.

Status: NOT-RUN. This section becomes instructions only after live validation.
