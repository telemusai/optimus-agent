# Windows repairs + Jev comparison: unfinished implementation handoff

## Release verdict — DO NOT MERGE OR CUT OVER YET

The user requested that the current work be frozen and published to save further Codex token cost, then explicitly confirmed overriding the repository's normal passing-checks-before-commit requirement for this unfinished draft. This is a transfer of unfinished work, NOT a verified release. Production Optimus was not cut over, restarted, reconfigured, or replaced. Do not infer acceptance from the Flash teams finishing or from the existence of this PR.

Baseline: `cad234a17589f5838407421fb435bd74d5ceda76` (upstream main, merge of PR51). Both Flash candidates used this baseline. Codex imported their manifest-verified changes into a separate checkout, reviewed them with three reviewers, and made additional corrections. The frozen source directories and original attribution were retained.

The only overlapping imported file was `crates/pi-coding-agent/src/modes/daemon/daemon_mode.rs`. The published Jev diff was malformed; its exact per-file Git diff was regenerated and applied over the repair version. This file needs particular integration attention. Cargo.lock only adds the local `pi-jev` package and its dependency edges; the inspected delta changes no existing external dependency version.

## Reading order and authority

1. This handoff is the current aggregate status and release boundary.
2. `core-review.md`, `jev-review.md`, `persistence-review.md` describe Codex corrections and unresolved defects, with test names and code pointers.
3. `flash-repair-dispositions.json`, `flash-memory-dispositions.md`, and the two `flash-*-handoff.md` files preserve the teams' original claims and attribution. Their word "fixed" means a team disposition, NOT independent combined-tree acceptance. Later Codex findings override those claims.
4. `COMPACTION_REQUIREMENT.md` and `GATEWAY_CREDENTIAL_REQUIREMENT.md` preserve the unresolved incident evidence and proof requirements.

No raw user conversations, credentials, live session copies, build trees, virtual environments, or large monitoring archives are intentionally included in this PR. Local evidence paths below are provenance, not prerequisites for understanding the next steps.

## Implemented work retained in this branch

### Windows/core repairs from the Flash teams

- Threshold compaction failure cooldown (5 seconds, bounded exponential backoff through 120 seconds), clearer terminal diagnostics, and start/outcome monitoring parity. Manual compaction and overflow recovery remain distinct.
- Responses/Codex transport diagnostics: empty billed completions, distinct send acknowledgement versus actual response headers, interruption categories, logical retry-group accounting. These are not proof of faster provider reasoning, and no new live Azure/GitHub WebSocket performance result is claimed.
- Content-free agent-message delivery journal, lifecycle/recovery-journal retirement changes, orphan identity handling, bounded stop settlement, absolute Windows taskkill resolution, startup dead-owner lease sweep, and native process-identity lookup.
- Harness corruption/unreadable-state distinction, recovery-copy preservation, durable writes, Windows rename retry, refinement-history failure reporting, canonical refinement IDs, and kernel-restore diagnostics. Python narrative refinement events are not fabricated as rollback-capable Rust refinement results.
- Monitoring schema/documentation/summarizer updates, including TypeScript parity changes. See team dispositions for each audit ID and explicit non-defects/deferrals.

### Jev feature from the Flash team

- New `crates/pi-jev` client, typed Noul/Choice/Score validation, eleven shadow evaluator categories, bounded scheduler/correlation records, mock transport, and tests.
- `/jev` menu, per-chat Off/Compare preference, masked first-use key input using Windows current-user DPAPI, footer/status components, observer bridge, and optional capability-gated daemon commands.
- Off is default. `active` is reserved and disabled. Compare records `applied=false`, zero actually avoided main-model calls, and hypothetical outcomes only. No primary provider/model/effort selection or subagent control authority is granted in any mode.
- IMPORTANT: evaluator existence is not operational coverage. Category 6 has no approved model allowlist, category 8 no memory input, and category 7 no real context evidence. Explicit skips are honest; do not advertise all eleven as effective optimizations. Comparison still makes separate Jev API calls and can disclose data: see the security blocker below.

### Additional Codex corrections

- Retained original orphan evidence when sidecar persistence fails; reject unreadable sidecars and overflow instead of dropping unresolved ownership records.
- Added checked harness-load/save generations and bounded cooperating Rust/Python file locks, including refusal to overwrite an unreadable or stale baseline. Integrated checked save into production refinement apply.
- Bounded lease sweep attempts even for unreadable/contended entries; access-denied Windows process probes no longer count as proof of death.
- Corrected premature retry-group settlement and stale start-time inheritance; strengthened exact grouping and cancellation regressions.
- Added failed-summary/history-preservation/queued-human-input regression; it is not yet run on the combined tree.
- Jev dormant registration now permits first-use Compare; shared persistent cancellation, dispatch/completion mode and credential gates, scheduler shutdown/drop, correct credential filename/generation, same-key rewrite invalidation, and cheap optional transcript entry count.
- Added worker-local telemetry, optional read-only status getter with client-side 2-second deadline, truthful unknown values, semantic-event/footer updates, and bounded/cancellable credential preparation with an atomic commit gate.
- Corrected daemon active-selector versus durable session UUID usage for Jev settings/status and child preference inheritance; initial attached-chat footer is sent through the existing extension status event. These newest integration paths remain untested.
- Corrected several frozen test compile/fixture assertions without removing the no-model/no-subagent authority checks.

See reviewer reports for exact authored changes and any item deliberately left unimplemented at freeze. No new broad repair work should be inferred from proposals in those reports.

## Actual validation at freeze

| Gate | Observed result | Scope limitation |
| --- | --- | --- |
| Combined `pi-jev` unit tests | 9/9 pass | Includes DPAPI cancellation/commit and scheduler lifetime/cancellation tests; later bridge/UI edits are outside this crate |
| Combined `pi-jev` client integration tests | 74/74 pass | Mock/synthetic only; no real API acceptance |
| Combined `pi-jev` comparison tests | 26/28 pass | Two outdated expectations corrected afterward, NOT rerun: cancellation terminal records and redaction marker spelling |
| Combined coding-agent library check | FAILED: two E0599 errors in initial attach-footer helper | Both were caused by `&self` where existing `next_id`/`write` require `self: &Arc<Self>`; receiver corrected afterward, NOT rechecked. Last check emitted 173 warnings; final source is not certified to compile |
| New coding-agent UI, bridge, status, persistence, retry and compaction tests | Authored; NOT run on final combined source | Test compile/runtime defects may remain |
| TypeScript report test and `npm run check` | NOT run | Required acceptance work remains |
| Final binary, packaged Python runtime, Windows standalone launch, real-provider checks | NOT performed | No cutover-ready artifact exists |

Earlier Flash-only results are retained in their handoffs but MUST NOT be relabelled as combined-tree passes. In particular, the repair team's Python Windows failure set was reported base-identical and its full coding-agent suite had environment/race failures. Reproduce baseline versus candidate in separate build targets before changing unrelated code. Some earlier Jev tests ran on an earlier source revision, not its frozen handoff.

## Priority continuation list — concrete fixes and proofs

### P0: Jev data minimization before any real sensitive Compare session

`crates/pi-coding-agent/src/core/jev_bridge.rs`: ToolCall serializes full raw arguments before truncation; AgentEnd clones/joins text before truncation. Raw arguments can contain `api_key`, password, authorization headers or credentials for unrelated systems. This was identified at freeze; the proposed fix was NOT implemented.

Smallest repair: omit raw tool arguments entirely from shadow observations; current tool-choice evaluators can use tool identity. Build bounded task/result excerpts incrementally, with deliberate secret redaction before payload construction. Document residual sensitive-content disclosure and require explicit Compare consent. Add synthetic secret-bearing ToolCall/AgentEnd fixtures and assert secrets are absent from the actual mock transport request, not merely from logs. Test multi-megabyte text without allocating its complete serialized copy. Do not claim generic redaction makes arbitrary private task text safe.

### P1: compile and execute the integrated feature paths

Re-run coding-agent library check after the receiver fix. Compile all relevant test targets before interpreting authored regression counts. Run `jev_ui_tests`, `jev_compare_tests`, and `jev_status_` tests. The new same-key-envelope regression leaves a synthetic file until TempDir cleanup: reconcile its lifecycle with the existing final file-allowlist assertion. Verify attach/snapshot ordering, current-session status ownership, reconnect with a different active selector, parent/child durable-ID preference inheritance, old client/new daemon and new client/old daemon behavior. No Jev command may become an unconditional startup dependency.

### P1: Jev configuration/lifetime residuals

- `pi-jev/src/config.rs`: settings use a fixed temp filename/read-modify-write without a cross-process transaction. Use a unique atomic temp plus cooperating lock/generation check; prove two workers changing different sessions do not lose updates.
- `pi-jev/src/client.rs`: inspect cancellation-safe in-flight accounting (RAII guard) and Retry-After handling (do not retry earlier than the server's requested delay; if outside the remaining budget, return a bounded skip).
- Native `/jev off` is a shared settings write, not immediate worker cancellation RPC. Dispatch/completion gates reject stale accepted results, but an already dispatched HTTP request can drain to its bounded deadline. Keep status truthful or add a capability-gated explicit mode-change notification, not a mutating status getter.
- `modes/daemon/native_supervisor.rs` generic fallback forwards optional Jev reads with the existing 24-hour worker request timeout. Client getter times out in 2 seconds but that does not cancel supervisor-side work. Add a bounded read-only forward deadline and prove no leaked forwarding task/eviction-fence hold. Do not replay mutations.

### P1: durable child-result delivery remains open

`core/agent_session.rs::deliver_pending_rlm_results_once` ignores send errors and can mark a task replied/handled despite no confirmed delivery. The reviewer found this at the final boundary. This repair is completely UNIMPLEMENTED; no uncertain-delivery marker/schema changes were started. Do not assume it is fixed from unrelated delivery telemetry.

Simply retrying every error risks duplicates after an ambiguous remote acknowledgement. Preserve result evidence; distinguish known pre-admission rejection from uncertain delivery; persist an uncertainty marker or use an end-to-end idempotent receipt contract before automatic replay. Re-read live task state after awaiting an explicit parent reply to avoid stale-snapshot duplicate sends. Required regressions: valid queued/delivered receipt, paused/rate-limit rejection then retry, unknown error and reload, concurrent explicit reply, and exactly-once/no silent-loss behavior. No live agent messages needed.

### P1: compaction incident is OPEN, not solved by cooldown

See COMPACTION_REQUIREMENT.md. The exact provider reason for repeated incomplete handoffs was not proved. Identical 13,107-token responses were only a possible output-limit clue. Keep the existing transcript/checkpoint and pending inputs intact. Test explicit max-output termination separately from unknown/incomplete/refusal/cancel/transport failure; only justified bounded retry is allowed. Run the queued-human regression, manual/overflow recovery, later successful automatic compaction and split-summary sibling cancellation. Never accept an unusable partial summary merely to hide the error. Preserve the custom native compaction/checkpoints, 250K trigger and full provider context window.

### P1: credential-helper warning is OPEN

See GATEWAY_CREDENTIAL_REQUIREMENT.md. `core/resolve_config_value.rs` is unchanged in this branch. It collapses timeout/nonzero/missing executable/empty stdout into one warning and can reveal raw configured command text. The installed helper's cold-start health checks can exceed the caller's 10-second budget, but that is not proved to be the screenshot's cause. Success-path pipe-reader joins can also outlive the deadline if descendants retain pipes.

Use fake helper/health/task abstractions and synthetic credentials. Preserve structured redacted reason/exit/elapsed data, align bounded deadlines and output draining, do not log stdout/secret arguments or run the production helper. Prove failure then success, wrong-build refusal, persistent failure, cancellation and pending-input preservation. Do not change Ollama routing, stored keys or live gateway settings to make a test pass.

### P1: persistence and lifecycle acceptance

Run reviewer tests: `deferred_orphan_`, `harness_checked_save_`, `sweep_`, `windows_lease_cleanup_requires_proof_of_death_not_a_failed_probe`, Python `test_persistence_`. Prove Windows Rust/Python lock interoperability in BOTH directions with synthetic files; same-runtime tests are insufficient. Confirm lost-update refusal reloads cleanly and does not retry a stale mutation. Verify unreadable sidecars/original journal retention and no truncation of unresolved ownership evidence. No noncooperating external-editor safety is promised by a cooperating lock.

### P1: isolate tests that launch user-visible applications

Confirmed browser source: `modes/interactive/components/login_dialog.rs::show_auth_renders_the_link_and_instructions` calls the real OS opener for `https://example.test/auth`. Inject a recording/no-op opener in tests, preserving production behavior and asserting the expected launch request. This remains UNFIXED; do not rerun the full battery first.

Exact ping fixtures exist in `core/package_manager.rs::run_command_capture_times_out_and_kills_the_child` and `modes/daemon/native_supervisor.rs::worker_stderr_is_observable_and_worker_survives_launcher_close`. The specific visible-window launcher remains unproven. The former already uses hidden captured spawn; the latter has detached-process flags. Audit flags and use a test-owned hidden sleeper that still proves timeout/reaping/launcher independence. Never kill generic terminal/conhost or production processes.

### P2: remaining audit/monitoring gaps and comparison

- BUSY-FLAG-01 age-based injection suppression remains unimplemented.
- A8: 35 of 38 old supervisor diagnostic sites remain untimestamped.
- D-04 delivery-journal duplicate-suppression regression missing.
- B6 long live-sample validation and actual Azure/GitHub gateway/WebSocket end-to-end acceptance unperformed. Unknown decomposition fields must stay null, not invented zeros.
- Historical/unsupported hypotheses are recorded in team dispositions; do not turn every audit suspicion into a patch.
- GLM-versus-DeepSeek assessment: GLM appeared faster to finish in these runs, but no controlled code-quality comparison was completed. Both outputs needed integration/Codex corrections. Attribution: repair compaction+lifecycle=GLM, memory+streaming=DeepSeek; Jev client+UI=DeepSeek, comparison pipeline=GLM, coordinator=GLM, followed by mixed review fixes. Task complexity, build-gate waiting and later model handover confound timings. If needed, review frozen lane patches/tests against equal criteria; never attribute combined corrected output to one model. This comparison is explicitly deferred to avoid further token cost.

## Suggested isolated verification commands

Use a fresh private HOME/USERPROFILE/agent/session/artifact/harness/memory/registry directory, unique pipe, fake provider and NO live keys. Serialize heavy builds, use separate candidate/baseline CARGO_TARGET_DIR, and bound command duration. Windows source Python must precede the stale installed rlm package on PYTHONPATH.

```text
cargo check --locked --offline -p pi-coding-agent --lib
cargo test --locked --offline -p pi-jev -- --test-threads=1
cargo test --locked --offline -p pi-agent-core --lib -- --test-threads=1
cargo test --locked --offline -p pi-coding-agent --lib transport_retry_tests -- --test-threads=1
cargo test --locked --offline -p pi-coding-agent --lib compaction_retry_backoff_tests -- --test-threads=1
cargo test --locked --offline -p pi-coding-agent --lib jev_status_ -- --test-threads=1
cargo test --locked --offline -p pi-coding-agent --test jev_ui_tests --test jev_compare_tests -- --test-threads=1
cargo test --locked --offline -p pi-ai --test codex_native_websocket -- --test-threads=1
```

Then named persistence/summary/observability suites from the reviewer reports. Run `node --test test/performance-metrics-compaction-report.test.mjs` from `packages/coding-agent` (node:test, NOT Vitest). Run repository `npm run check` with full output. The user requested the unfinished draft instead of completing these gates; their omission is not a pass. Do not use `npm test`, `npm run build`, or `npm run dev`.

## Packaging and cutover — separate future authorization

No packaged binary or verified release manifest has been produced from this combined source. After fixes/tests, freeze source hashes, build an isolated artifact and a fresh candidate Python runtime, verify imported sessions/checkpoints/config compatibility with synthetic data, and prove independent Windows launch through the validated scheduled-task path (no Codex process ancestor). Consult the local Prime/Optimus compatibility record before any live change. Preserve production provider settings, native compaction and subagent API compatibility; do not wholesale-copy upstream installer logic.

Get explicit user authorization for cutover/restart. Keep current Optimus running until then. This PR authorizes neither production replacement nor automatic merge.

## Local provenance (not committed bulk data)

- Combined source: `C:/Users/openclawuser/optimus-flash-repairs-20260919/codex-wrapup/source`
- Review reports and isolated logs: sibling `codex-wrapup/reviews` and `codex-wrapup/logs`
- Frozen repairs: `C:/Users/openclawuser/optimus-flash-repairs-20260919/source`
- Frozen Jev: `C:/Users/openclawuser/optimus-jev-compare-20260919/source`
- Original audit: `C:/Users/openclawuser/optimus-windows-core-review-20260919-022913`
- Original lane attribution: each run's manifests, reports and comparison-evidence directories; no large historical session snapshots were copied during this wrap-up.

All Optimus implementation teams were safely handed off and observed idle; their chats/work were retained. Three Codex reviewers were told to freeze at the user's final cost-saving instruction. Remaining local uncommitted `scripts/rust-env.cmd` line-ending noise is deliberately excluded from publication.
