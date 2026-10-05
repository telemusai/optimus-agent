# Rebase Report — monitor-review perf & reliability changeset onto upstream/main

- **Source commit:** `910c22ce8` on `fix/monitor-review-perf-reliability-20261005-034630` (base `7702f441b`, v0.1.20)
- **Target base:** `upstream/main` = `32fc5a1dc` (v0.1.24; includes #111 snapshot/UI/Jev local-work reduction, #115 compaction local acceleration, #116 provider HTTP pooling + single-pass SSE, v0.1.21–v0.1.23 reliability patches)
- **New branch:** `fix/monitor-review-perf-reliability-rebased` (2 commits on top of `32fc5a1dc`)
- **Method:** semantic port. The port started from `upstream/main` content per file and re-applied our changes; a cherry-pick was used only to stage the clean (non-conflicting) files, and every one of the 18 conflicting files was resolved by hand against upstream's structure.

## Per-area outcome

### A1 — SSE inter-chunk idle deadline (ported, adapted to upstream's new SSE readers)
- `sse_frames.rs`: idle-timeout resolver (`PRIME_AGENT_SSE_IDLE_TIMEOUT_MS`, default 120_000 ms, clamp 5_000..600_000, unparseable → default, test-only override atomic) ported unchanged; upstream's `SseFrames` byte framing untouched.
- `openai_completions.rs`: **ported onto upstream's single-pass `SseLineReader`** (from #116). `read_sse_data` gained the `idle_timeout` parameter and a `tokio::select!` sleep arm that emits `StreamError::sse_idle_timeout` and drops the response body. `run_stream_body` resolves the deadline once before spawning the reader. `StreamError::sse_idle_timeout` constructor (code `sse_idle_timeout`, retryable classification) ported.
- `anthropic.rs`: `SseMessageReader` gained the same `idle_timeout` + `tokio::select!` around `chunks.next()` (upstream's #116 line-scan reader preserved).
- **Dropped as upstream-duplicate:** none for A1 (upstream has no SSE idle deadline).
- Tests ported: `sse_idle_timeout_error_classifies_as_retryable_not_permanent`, `read_sse_data_aborts_a_silent_body_and_drops_the_connection`, sse_frames resolver test. Upstream's `SsePayload` now carries `{payload, parsed}`; the stall test was adapted to that shape.

### A8 — dgx session-affinity headers (ported clean)
- `detect_compat`: `is_dgx` (`provider == "dgx" || starts_with("dgx-")`) → `send_session_affinity_headers: is_dgx`; per-model `compatCompletions.sendSessionAffinityHeaders` override still wins via `get_compat`. Test `detects_dgx_session_affinity_and_keeps_the_models_json_opt_out` ported; upstream's own affinity header tests retained.

### A9 — credential/command resolution cache (ported clean; ours-only files)
- `resolve_config_value.rs` process-global cache with TTL (`PRIME_AGENT_CREDENTIAL_CACHE_TTL_MS`, default 30 s, 0 = off) + generation-based invalidation `invalidate_resolved_command_values()`; invalidation wired into `auth_storage.rs` and `model_registry.rs` stale hooks. Integration test `tests/credential_refresh_window.rs` ported.

### A10 — endpoint health demotion (ported clean, minus one local variant)
- `model_registry.rs`: EWMA stats, `record_endpoint_observation`, reorder-only `get_available` partition, `PRIME_AGENT_ENDPOINT_HEALTH_DEMOTION` (default on), probe/staleness recovery + tests ported.
- **Dropped as local-variant-only (not upstream-duplicate, but unsupported by upstream pi-ai):** the `defaultMaxTokens`/`sglangTokenBudget` model-definition plumbing that shipped inside our `model_registry.rs` hunk. It requires `pi-ai` `Model.default_max_tokens` and `OpenAICompletionsCompat.sglang_token_budget`, which exist only in the local pi-ai variant (never merged upstream; upstream #116 replaced the SGLang preflight with transport pooling). Keeping it would not compile against upstream `pi-ai`. Registered in the `.changes` fragment as out of scope for this port.

### A2/A7 + B1–B4 — compaction (ported; upstream #115 acceleration preserved)
- Retry ladder, cooldowns, 5-minute summary deadline with tail-truncation fallback, input-admission release (`compaction_input_admission_released`, `compaction_blocks_session_input`), `CompactionSettings` fields `prompt`/`summary_budget_mode`/`trigger_threshold`/`deadline_ms` (B1–B4, A7) all ported. Flag defaults unchanged (legacy/off; deadline is a bound, not a model-output change).
- Upstream #115's dispatched SIMD char-count kernel, borrowed-state and single-copy serialization path are untouched.
- **Adapted:** upstream test files `tests/compaction_kernel_parity.rs` and `examples/compaction_replay.rs` call `build_summarization_prompt`/`CompactionSettings` with the pre-port signatures; they were updated to pass legacy defaults (`COMPACTION_PROMPT_LEGACY`, `SUMMARY_BUDGET_MODE_LEGACY`, `trigger_threshold: None`, `deadline_ms: 0.0`). This is the standard porting cost of a signature change, not a behavior change.
- `scripts/validate_compaction_quality.py` ported unchanged (formulas identical to the ported ladder).

### A3/A4 — failure-reason telemetry (ported; upstream allowlists not duplicated)
- `pi-agent-core/performance_metrics.rs`: `PerformanceMetricErrorClass`, `PerformanceMetricFailure` (+ `from_stream_failure_info`/`from_anyhow_error`/`from_assistant_message`/`classify_message`), bounded `error_message`/`http_status`/`stage` on events & records, sanitizers, `Timeout` outcome, `DaemonLifecycle` operation, `tool` identity field, `UiTickFallbackSkipped`/`UiAckDeadlineMs`/`RenderMs`/`DiffMs` measurements. Upstream's #111 measurements (Serialization*, SnapshotCas/LegacyCaptures, Ui* submit timings, UiEventApply/UiTick ops) kept — only missing variants added.
- `pi-coding-agent/core/performance_metrics.rs`: recorder opt-in error-text gate (`PRIME_AGENT_PERFORMANCE_METRICS_ERROR_TEXT`, default off), sanitize/allowlist for the new fields, `MemoryFileIo::pair` test seam, tests ported.
- **Adaptation (documented):** the recorder's `OUTCOMES` allowlist in our original commit omitted the new `Timeout` variant, so ack-deadline timeouts would never persist. The port adds `Timeout` to the allowlist (6 entries) so A13's classification is actually recorded end-to-end.
- `agent_loop.rs` failure-detail threading ported clean.

### A15 — daemon lifecycle metrics + reconnect backoff (ported clean)
- New `daemon_lifecycle_metrics.rs` + wiring in `daemon_client.rs` (backoff ladder 1s/2s/5s/10s cap), `native_supervisor.rs`, `supervisor_maintenance.rs`, `daemon_session_summarizer.rs`, `daemon_supervisor.rs`, parity tests. Upstream's v0.1.23 daemon fixes (dead-client resume, detached daemons) preserved.

### A5/A6/B5 — snapshots (ported; upstream #111 preserved)
- Python: `snapshot.py` CAS-v2 default for fresh sessions, per-variable change detection (`_forget_missing_names`, `_record_duration`), budget-aware partial snapshots (`snapshot_budget_ms`, adaptive 60–180 s timeout), `DroppedNamesCount`/`serialization_reused_names` metrics, adaptive debounce flag (default legacy). Upstream #111's serializer segments/file-handle work (`snapshot_serializer.py`, `snapshot_safety.py`) preserved; the two new metric fields were added to the envelope.
- Rust: `repl_manager.rs` (adaptive timeout state, budget fields, `KernelSnapshotDebounceMode`), `state_snapshot.rs`, `kernel/shared.rs`, `kernel/performance_metrics.rs` ported; upstream's #111 snapshot-local-work reduction kept. The snapshot metadata test's measurement count was updated 33 → 35 for the two added fields.
- Tests: `test_snapshot_budget.py` (new) and `test_snapshot_v2.py`/`test_snapshot_speed.py` updates ported.

### A11/A12/A13 — UI render memoization, idle tick gate, ack deadline (ported; upstream #111 receipts preserved)
- `pi-tui`: `RenderPhaseTimings`, `run_pending_render` returning phases, `render_cache.rs` (StyledRenderCache), keybindings epoch ported (ours-only files).
- `native_host.rs`: `RowRenderCache`/`render_row_cached`/`render_style_revision` row memoization (A11), gated idle fallback tick (A12, `OPTIMUS_UI_IDLE_TICK_PAINT`, default on = old behavior), `expire_ack_deadlines()` sweep (A13), phase-timed `ui_metrics.rendered(elapsed, &phases)` (A15).
- `native_host_metrics.rs`: `fallback_ticks_skipped`, per-phase sums, `settle_timeout` with deadline-capped duration, `UiMetrics::with_recorder_and_ack_deadline`, `OPTIMUS_UI_ACK_DEADLINE_MS` (default 10 s; `0` disables) ported. Upstream's `SubmissionTicket::id()` was **kept** (upstream's receipts/stash-restore from v0.1.21–24 need it; our local variant had removed it).
- **Dropped as upstream-duplicate:** `HostEvent::TimedConnection`, `SubmissionReply`, `submission_is_prompt`, `submission_attachment_invalidated`, `apply_submission_reply`, `submit_from_editor`, `submit_with_metrics` ticket flow, `message_projection_is_identity`, and `apply_event_legacy_for_speed` (all already in upstream via #111, including the receipts integration our local variant did not have).
- `interactive_mode.rs`: `idle_fallback_paint_enabled()` ported (auto-merged).
- Component render revisions (`agent_message.rs`, `assistant_message.rs`, `user_message.rs`, `theme.rs` style epoch) ported.

### Other files
- `.changes/monitor-review-perf-reliability.md` ported; one bullet adjusted to note the `defaultMaxTokens`/`sglangTokenBudget` plumbing is out of scope against upstream `pi-ai` (see A10 above).
- `docs/performance-metrics.md`, `prime-agent-runtime/src/rlm/repl.md`, `test_repl.py` updates ported.
- `scripts/rust-env.cmd`: the branch's first commit normalizes its line endings to CRLF to match `.gitattributes` (the local checkout had LF; content unchanged).

## Conflicts and resolutions (18 files)
| File | Resolution |
|---|---|
| `pi-agent-core/performance_metrics.rs` | union merge: upstream #111 measurement/operation additions + our A3/A4/A15 additions; `Timeout` added to recorder allowlist (see A3/A4) |
| `pi-ai/providers/openai_completions.rs` | **re-ported by hand onto upstream's single-pass reader**: dgx affinity, `sse_idle_timeout` ctor, `read_sse_data` idle deadline, run_stream_body wiring + 3 tests. Local `sglang_budget`/`http_client_pool` modules dropped in favor of upstream `shared_http.rs` pooling |
| `pi-coding-agent/core/kernel/repl_manager.rs` | union merge of upstream #111 kernel changes + our A5/A6 snapshot state/metrics; measurement count test updated 33→35 |
| `pi-coding-agent/core/performance_metrics.rs` | hand-ported onto upstream (git treats it as binary due to a literal NUL in an upstream comment): all A3 recorder gate/sanitizer/test additions re-applied on upstream's bytes |
| `pi-coding-agent/modes/interactive/native_host.rs` | **re-ported by hand onto upstream**: A11 row cache, A12 idle tick gate, A13 expiry sweep, A15 phase timings. Upstream receipts/stash/errors/#111 test helpers kept |
| `pi-coding-agent/modes/interactive/native_host_metrics.rs` | upstream file + our A12/A13/A15 additions via patch; `SubmissionTicket::id()` kept for upstream receipts |
| `prime-agent-runtime/src/rlm/repl.py` | union merge: our `_record_duration` + `dropped_names_count` metric fields onto upstream #111 serializer path |
| `prime-agent-runtime/src/rlm/snapshot.py` | union merge: `_record_duration` history hook + two envelope metric fields onto upstream #111 snapshot work |
| 10 remaining files | auto-merged cleanly by cherry-pick, verified by compile + tests |

## Validation (publication tree, this branch)
| Check | Result |
|---|---|
| `cmd.exe /c cargo.cmd check --locked --workspace --all-targets` | **exit 0** |
| `python scripts/check-rust-layout.py` | **exit 0** |
| `python -m unittest discover -s tests -p 'test_*.py'` | **OK — 33 tests (10 skipped)** |
| `python -m compileall` on all touched `.py` | **OK** |
| dill python: `test_snapshot_budget.py` | **OK — 15 tests** |
| dill python: `test_snapshot_v2.py` | **OK — 31 tests** |
| dill python: `test_snapshot_speed.py` | 11 tests, **1 failure — pre-existing on pristine upstream/main** (verified in a clean worktree: `test_custom_reducer_runs_once_and_partial_buffer_is_reset`, dill cannot resolve `__main__.CustomList`) |
| dill python: `test_snapshot_custom_speed.py` | 19 tests, **1 failure — pre-existing on pristine upstream/main** (same dill `__main__` class-resolution issue with `CountedList`) |
| dill python: `test_snapshot_serializer_segments.py` | **OK — 11 tests** |
| `cargo test -p pi-coding-agent --test` fidelity / summary_guard / observability / length_recovery / credential_refresh_window | **all pass** (7+7+2+1+1) |
| Smoke (beyond the fast path): `pi-ai` sse_idle + read_sse_data_aborts + detects_dgx_session; `pi-coding-agent` endpoint_health (6), provider_retry (19), resolve_config_value single-threaded (14), daemon_lifecycle_metrics (4), compaction lib (51), kernel_settlement with `KERNEL_CONTAINMENT_TEST_PYTHON` set (15) | **all pass** |
| `kernel_settlement_*` without `KERNEL_CONTAINMENT_TEST_PYTHON` | panics on the missing env var — by design (env-gated), not a regression; passes when the var is set |

## Known limitations / follow-ups
1. `test_snapshot_speed.py` / `test_snapshot_custom_speed.py` each fail 1 test, **identically on pristine upstream/main** (dill `__main__` class resolution) — documented, not fixed (not ours to fix).
2. The `defaultMaxTokens` + `sglangTokenBudget` registry plumbing from the local pi-ai variant is not portable without the matching `pi-ai` types and is left out (see A10). Everything else from the 50-file changeset is present.
3. The full targeted cargo battery (pi-tui full, agent_message/assistant_message/ipython_cell/ui_tests/keybindings, a7_backoff, b_fix_flag, a2_input_release, core::performance_metrics) was skipped per the fast-path instruction; GitHub CI will run it. The smoke subset above (the riskiest hand-ported areas) is green.
