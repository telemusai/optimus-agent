# Persistence, lifecycle, and test-isolation review

Reviewer: Codex persistence reviewer. Date: 2026-09-19 UTC.

## Scope and boundary

Reviewed frozen candidate `C:/Users/openclawuser/optimus-flash-repairs-20260919/source`, whose repair baseline is `cad234a17589f5838407421fb435bd74d5ceda76`. Read repository AGENTS, REPAIR_BRIEF, CODEX_HANDOFF, lifecycle/memory dispositions and both reviewer defect reports. Fully read Python harness, Rust refinement, session lease, atomic writer, supervisor maintenance and maintenance tests. Agent-session integration was traced at its refinement apply/save section, not audited as a whole; the core reviewer, who has read that file, owns its integration edit.

After the independent review identified blocking safety gaps, the parent authorized focused fixes in **combined source only**: `C:/Users/openclawuser/optimus-flash-repairs-20260919/codex-wrapup/source`. Frozen originals, production runtime, session data, gateway, and Git state were not modified. No build, application test, popup reproduction, live daemon call, or process termination was performed by this reviewer. Test results below are **not_run_by_this_reviewer**, not claimed passes; root owns execution and final readiness.

## Confirmed blockers in the frozen candidate and repairs

### PERSIST-01 / high: deferred orphan evidence could be destroyed on failed persistence

`crates/pi-coding-agent/src/modes/daemon/supervisor_maintenance.rs`, `record_deferred_orphans` and `recover_uncertain_worker_operations`: the side record used an ordinary overwrite, converted read failure to no prior records, discarded older records above 64, and ignored the persistence result before clearing the authoritative orphan journal. Logging PIDs to stderr did not satisfy durable evidence retention. A sharing/permission/disk failure could remove the only structured evidence for unresolved process ownership.

Combined repair: bounded validated side-record read, atomic replacement with file/directory fsync, exact owner validation, identity-tuple deduplication for crash replay, capacity as a refusal boundary rather than truncation. Failure retains the authoritative journal and returns a stable permanent cleanup error, allowing existing park logic to retain descriptor/journals without an endless retry loop. No bare-PID kill authority was added. The side record explicitly describes unknown ownership, not successful cleanup.

Focused tests in `supervisor_maintenance_tests.rs` (Windows; use only synthetic dead worker descriptors and the test process's non-killable PID-only record):

- `deferred_orphan_write_failure_retains_source_and_parks_cleanup`
- `deferred_orphan_capacity_and_malformed_evidence_never_discard_records`
- `deferred_orphan_retry_is_idempotent_at_capacity`

Root filter: `deferred_orphan_`. Cases assert original bytes preserved, parked descriptor retained, all older side records retained, bounded/idempotent replay, current test process alive, and no temporary-file leak.

### PERSIST-02 / high: a recovered read failure or concurrent valid save could erase healthy memory

Frozen Python `prime-agent-runtime/src/rlm/harness.py:247`, `:292`, `:380`, `:422`: load failure yields an empty view but records the current mtime. When read access recovers without an mtime change, `_sync_from_disk` does not reload, and save-time classification sees healthy JSON then overwrites it with the empty-derived cached state. Reclassification protects unreadable/corrupt bytes only; it does not prove the proposed state was based on the current healthy bytes. Two healthy writers can also interleave between sync and atomic replacement; atomic replacement prevents torn files, not lost updates.

Frozen Rust `core/refinement/refinement.rs`, `load_harness_state`, `load_harness_state_details`, `save_harness_state`, plus `agent_session.rs` refinement apply/save: the plain loader drops failure status and later save accepts healthy current JSON. This permits the equivalent empty-fallback overwrite after a transient read failure. Proposal baseline checking does not cover another process saving between apply's state load and its final save.

Combined repair:

- Load details retain a SHA-256 generation of the exact bytes read, separate from missing and unreadable status.
- Python reloads after failed reads/saves and on a byte-generation change, including changes that preserve mtime. A failed save invalidates its mutated cache so unsaved changes are not later presented as durable.
- Python/Rust writers coordinate on the same persistent `harness_state.json.lock` using OS file locks. Python uses `msvcrt.locking` on Windows / `flock` on Unix; Rust uses `File::try_lock`. One-second bounded acquisition; process death releases the lock. The stable lock file is deliberately not unlinked, avoiding split-inode locks.
- Under the lock, final save requires exactly the loaded generation and refuses any unreadable load baseline even if access has since recovered. A changed generation is reported without automatic replay of the stale mutation.
- Corrupt bytes can only be quarantined/replaced against the generation actually loaded; unreadable bytes remain untouched.
- Rust exports `save_harness_state_checked(dir, state, baseline)`. The core reviewer owns applying it to `AgentSession` with `load_harness_state_details`; the unguarded-construction wrapper remains for existing fixture construction, not production read/modify/write use.

Five Python tests (`test_harness.py`, filter `persistence_`) cover: transient read denial then recovery, same-mtime valid external write, writer interleaving after sync with exact disk-byte retention and cache reload, missing-to-created race, lock contention and recovery without orphan temps.

Five Rust tests (`refinement.rs`, filter `harness_checked_save_`) cover: unreadable baseline after recovered access, valid generation conflict, missing-to-created race, corrupt generation conflict/matching quarantine, shared lock bounded contention/release. Existing atomic-save directory assertions now allow the intentional stable `.lock` file and continue to reject temporary files.

Additional verification requested from root: synthetic cross-runtime Windows lock interoperability in both directions, and eventual successful save after holder exit. Same-runtime unit tests are not a substitute for that integration proof. No claim is made that an arbitrary noncooperating external editor participates in the lock protocol.

### PERSIST-03 / medium: dead-owner startup sweep was not bounded on its failure paths

Frozen `core/session_lease.rs:787`: the loop checked `result.scanned >= 256`, but unreadable owner files and guard acquisition failures never incremented `scanned`. An all-corrupt/all-contended tree could be fully scanned; each contested guard could also sleep through its one-second acquisition loop. This is a concrete new startup-lag route, not an aesthetic counter issue.

Combined repair: at most 256 directory-entry attempts regardless of outcome, a 250ms elapsed budget checked between filesystem operations, and nonblocking opportunistic guard acquisition that never removes another holder's guard. Normal explicit lease acquisition keeps its existing coordination behavior. Existing result fields keep their semantics; failed/unknown owners remain separately counted and retained. Individual filesystem operations are still OS calls and are not claimed to have a hard real-time deadline.

Tests: `sweep_bounds_unreadable_entries_instead_of_only_successful_scans`, `sweep_never_waits_on_or_reclaims_a_contended_guard`, plus existing `sweep_reclaims_only_dead_owner_leases`. Filter `sweep_`.

### PERSIST-04 / medium: failed Windows liveness probes could authorize reclaiming an unverified live lease

Frozen `core/session_lease.rs:423`: `OpenProcess` or `GetExitCodeProcess` failure returned false, which `is_lease_owner_alive` interprets as dead. In particular, access denied is not proof that a process no longer exists. The new global sweep expanded the impact of that preexisting helper assumption.

Combined repair: only the nonexistent-PID open result (`ERROR_INVALID_PARAMETER`) or a successfully observed exited process counts as dead. Access denied/unknown probe errors remain conservatively alive; unknown start identity remains conservatively alive as before. Oversized i64 PID values are rejected instead of wrapping to another u32 process ID. No process kill behavior was introduced.

Test: `windows_lease_cleanup_requires_proof_of_death_not_a_failed_probe`; checks denied/unknown/active/exited/nonexistent outcomes and current test process liveness.

## Popup test isolation diagnosis

### Browser popup: source-confirmed test side effect

Frozen `modes/interactive/components/login_dialog.rs:1042`, `show_auth_renders_the_link_and_instructions`, creates the real dialog and calls `show_auth("https://example.test/auth", ...)`. Production `show_auth` at line 451 invokes the platform browser opener. Hiding the `rundll32` launcher console does not stop it from opening a visible browser. This exact fixture URL matches the screenshot.

Smallest safe fix: inject a no-op/recording URL opener into the UI test fixture (or a test-only side-effect gate), assert that the opener is requested once with the expected URL, and keep production launch behavior explicit. Do not replace the URL with a working real site or rerun the unisolated test. Root owns this repair and validation.

### Ping terminal: exact fixture found; visible-launch attribution remains unproven

Frozen `core/package_manager.rs:4160`, `run_command_capture_times_out_and_kills_the_child`, uses `cmd /c "ping -n 12 127.0.0.1 > NUL"`. However, `run_command_capture` at line 3027 invokes `spawn_hidden` at line 3046 and captures stdout/stderr; Windows helpers already set `CREATE_NO_WINDOW` (`utils/child_process.rs:296`). Thus the source proves the exact command is a test fixture, but not that this already-hidden capture path produced the visible unredirected terminal in the screenshot. A different/manual launch route remains possible. Do not overstate this attribution.

Smallest safe verification: audit the actual invoking test/command and central spawn flags without reopening it. Prefer an inert test-owned sleeper/helper with piped output and explicit hidden flags over a terminal-visible command. The timeout test must still exercise cancellation and child reaping; merely skipping it or suppressing its output is not adequate. Root owns this work.

Later read-only trace also found `native_supervisor.rs`, `worker_stderr_is_observable_and_worker_survives_launcher_close`: it launches `cmd /c "echo PARITY-SENTINEL-STDERR 1>&2 & ping -n 12 127.0.0.1 > nul"` through the detached-worker helper. That helper combines detached/new-process-group and hidden flags. This is another exact fixture candidate, not proof of the screenshot's launcher. It was not rerun or changed by this reviewer.

## Jev optional read-only status adapter follow-up

Parent authorized this bounded follow-up after the persistence report. Fully read `types.rs`, `daemon_agent_connection.rs`, and `in_process_agent_connection.rs` before editing. The public status accessor and UI consumer are owned by the Jev reviewer; daemon handler/durable session identity integration is owned by root.

Implemented in combined source:

- Optional `AgentConnection::get_jev_status` returns `Ok(None)` by default, so unsupported adapters remain locally unsupported without breaking startup.
- Daemon adapter checks `jev_control`, requests `jev_get_status` for the captured `activeSessionId` with a 2-second timeout and reconnect replay disabled, validates `pipeline` as object or null, rejects a reply if the active session changed, and preserves transcript-cache freshness.
- In-process adapter reads the canonical session header ID and `core::jev_bridge::session_status_snapshot`, without cloning the transcript/provider snapshot. Both supported adapters return `Some({"pipeline": object-or-null})`; missing observations remain null, never fabricated zero metrics.
- No model/subagent control, credential reads/writes, or Jev/provider network calls were added by these adapters.

Five isolated fake-transport regressions were written in `modes/agent_connection/jev_status_tests.rs`; **none executed by this reviewer**. Filter `jev_status_`: unsupported means no RPC; object/null preserved with exact request timeout/no-replay and cache stability; malformed envelopes fail; worker error propagates without retry; reply after session switch is rejected. Existing `invalidates_cached_snapshot_matches_the_switch` now includes `jev_get_status`.

Read-only routing verification: `daemon_protocol::daemon_command_plane` marks all three optional Jev commands as session-plane, and the supervisor public allowlist contains them. Compatible direct routing selects the worker. In `native_supervisor.rs` (fully read), there is no supervisor-local Jev response arm: generic dispatch resolves the caller-visible active session to its worker, rewrites the worker active ID, and forwards the command to that worker. The supervisor does not answer from its own empty Jev registry. Root still needs the worker handler to resolve the active ID into the durable UUID used by observations/settings.

**Open concern, not repaired:** generic native-supervisor forwarding uses the 24-hour `REQUEST_TIMEOUT`, independent of the caller's 2-second status timeout. A hung Jev status worker request can therefore outlive the UI caller and retain the supervisor command/eviction read fence; repeated polls could accumulate work. Smallest bounded repair would apply a short outer deadline to read-only Jev forwarding, including resolution/connection where appropriate, without reconnect replay or background retries. Root was notified; no native-supervisor edits were made by this reviewer.

## Candidate work retained / limitations

The flash repairs did provide useful source improvements: invalid UTF-8 classified as readable corruption; atomic recovery copies rather than pinning torn backups; permission denial separated from missing state; readable corruption preserved before replacement; observed history-append failure no longer silently reported as fully durable; native Windows process identity avoids the PowerShell lookup hot path. Those are meaningful, but their original focused tests did not exercise the lost-update and evidence-write-failure windows above.

No whole-repository correctness claim is made. In particular, malformed individual entries inside an otherwise valid JSON object still follow the existing tolerant parsing contract; this review did not redesign schema recovery. Cross-process lock interoperability and root-run regression results are readiness gates, not assumed. No production cutover has occurred as part of this review.

## Changed files owned by this reviewer

- `crates/pi-coding-agent/src/modes/daemon/supervisor_maintenance.rs`
- `crates/pi-coding-agent/src/modes/daemon/supervisor_maintenance_tests.rs`
- `crates/pi-coding-agent/src/core/refinement/refinement.rs`
- `crates/pi-coding-agent/src/core/session_lease.rs`
- `prime-agent-runtime/src/rlm/harness.py`
- `prime-agent-runtime/test/test_harness.py`
- `crates/pi-coding-agent/src/modes/agent_connection/types.rs`
- `crates/pi-coding-agent/src/modes/agent_connection/daemon_agent_connection.rs`
- `crates/pi-coding-agent/src/modes/agent_connection/in_process_agent_connection.rs`
- `crates/pi-coding-agent/src/modes/agent_connection/jev_status_tests.rs` (new)

Core reviewer owns the nonoverlapping `agent_session.rs` checked-save integration. Root owns compilation, all test execution, final report aggregation and launch/cutover authority.

## Frozen handoff at user stop

Changes frozen immediately on root's user-cost/stop instruction. No further source edits, audits, or test execution authorized for this reviewer. The ten owned source/test files above and this report are the exact handoff. All 21 newly written tests (16 persistence/lifecycle, 5 Jev adapter) are unrun by this reviewer; no root results are assumed here. Required open gates: compilation, focused regressions, Python/Rust Windows lock interoperability, checked-save production integration verification, real durable-ID Jev handler integration, and the supervisor timeout concern above. This is **unfinished draft-PR handoff**, not cutover-ready or production-verified. No cutover performed.
