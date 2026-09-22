# Rust-only repository validation

The TypeScript application and root npm workspace are removed. Rust still needs the Python runtime, bundled Python skills, themes, documentation, and browser HTML export assets. These are packaged with the native executable; `resources/agent/package.json` is identity metadata only.

## Migration checks (Linux, Rust 1.95.0)

- `bash scripts/check.sh`: Cargo workspace/all-target checking, 13 packaging/launcher tests, resource inventory, and whitespace checks passed. Existing compiler warnings remain.
- Shared crate unit tests passed: `pi-ai` 725, `pi-agent-core` 50, `pi-tui` 251, and `pi-jev` 61. Native configuration tests passed (31).
- Native integration tests passed: `rust_only_resources` (2), `slash_command_matrix` (2), `native_cli` (1), and `transcript_t06` (4). The resource test relocates the executable, gives its bundle a distinct HTML template, and exports a session without a TypeScript tree or launcher environment overrides.
- The relocated Python skill tests passed (2 agent-message and 3 observation tests). Both relocated performance-report tests passed.
- A real archive was staged, installed into a private temporary prefix, and launched successfully. The source launcher was also exercised. Installer tests cover failed version checks and restoration of the previous launcher after activation failure.
- Linux was validated locally. Windows/macOS are covered by the workflow's native compile matrix; those platforms were not executed locally.

CI uses these focused native migration checks in place of the removed npm pipeline. The command matrix now checks native registry/alias coverage and dispatch ownership, including grouped match arms; it no longer reads deleted TypeScript source or claims that dispatch presence proves behavior parity.

## Wider-suite limitations

The complete existing suites are not green. `./test.sh` remains available for broader diagnosis; focused CI does not claim to replace that coverage.

The coding-agent library run completed with 2,743 passed, 10 failed, and 2 ignored. Failures were in:

- `enabled_skill_is_importable` (requires an explicitly prepared ambient kernel venv).
- `t11_harness_paths_agree_uses_the_artifact_dir` (memory lock acquisition).
- `t11_lifecycle_hooks_without_handlers_are_inert` and `session_shutdown_helper_reports_whether_handlers_existed` (shutdown handler expectations).
- `t11_memory_extension_veto_is_reachable` (refinement veto behavior).
- `real_session_prompts_the_faux_provider_and_delivers_events` (missing scratch directory).
- `t09_timed_out_stop_finishes_cleanup` (missing test process executable).
- `messaging_safety_supervisor_records_durable_delivery_outcomes` (Unix socket path length).
- `t14_a13_new_with_prompt_prompts_verbatim` (new-session prompt event).
- `reclaims_a_legacy_directory_lock_without_an_owner` (lock reclamation).

The Python runtime suite ran 395 tests with one error and eight skips. `test_save_new_file_keeps_restrictive_umask` creates a mode-000 file, then the unchanged harness implementation attempts to read it to calculate its generation. Both that implementation and test are unchanged by this cleanup.

The optional `scripts/test-native-subagents.mjs` probe successfully exercised Python tool admission and two child agents, then failed on resume with “Cannot admit a session action while queued session input is suspended.” The same failure was reproduced with the previously installed `3dba6a7c7` binary, using a private profile and the same synthetic provider. Its Rust/runtime sources match the cleanup's base commit `4b09f7283`; the installed application and user state were not modified.

These limitations remain separate from removal of the reference tree. This change does not claim to repair them or establish complete feature parity.
