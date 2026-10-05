# Stream stall watchdog + truncated-response surfacing

Two focused reliability fixes for the agent loop's provider stream pump, both
offline (no provider calls), on top of upstream `32fc5a1dc` (v0.1.24).

## Motivation (incident telemetry)

A supervisor session sent a routine progress-check prompt on a 223,489-token
context. The model (`glm-5.3`) generated exactly 131,072 output tokens (the hard
output cap) over ~5.8 minutes, then the local pipeline stalled for 43.4 minutes
(`local_drain_ms = 2,604,141` - the only >60s drain in the session's 112 metric
files) with no timeout, no error, and no recovery until the user killed it:

- `dispatch_to_response_headers_ms = 5,465`
- `dispatch_to_first_event_ms = 5,465`
- `dispatch_to_network_terminal_ms = 348,596`
- `total_ms = 2,952,737` (`outcome = cancelled` only because the user stopped it)

Root-cause chain: (1) no watchdog on stream progress / local drain,
(2) runaway generation accepted silently, (3) the giant response stalled local
finalization.

## Fix 1 - stream stall watchdog (default ON)

`crates/pi-agent-core/src/stream_watchdog.rs` (new module) plus wiring in
`stream_assistant_response_inner` (`crates/pi-agent-core/src/agent_loop.rs`).

- **Event-gap timeout** (`PRIME_AGENT_STREAM_EVENT_GAP_MS`, default `120000`,
  clamp `5000..600000`, `0` disables, unparseable falls back to the default):
  the `response.next()` await is wrapped in `tokio::time::timeout` inside both
  existing `select!`s. The timer resets on every event. On expiry the attempt
  aborts.
- **Overall stream deadline** (`PRIME_AGENT_STREAM_DEADLINE_MS`, default
  `900000`, clamp `60000..7200000`, `0` disables, unparseable falls back to the
  default): wall-clock cap for the whole stream phase of one attempt, measured
  from pump-loop start. Enforced both by a `sleep_until` arm in the `select!`s
  (covers a silent stream when the gap is disabled or larger) and by an
  up-front elapsed check each iteration (covers a continuous event trickle
  that could otherwise starve the timer arm).
- **Abort path**: `response.request_cancel()` (aborts spawned producer tasks
  and ends the stream - the `closeIterator()` intent), then `close_stream(...)`
  (the loop's existing abort-token mirror, exactly like the cancellation arm),
  then the attempt is finished with a terminal assistant message:
  `stop_reason = "error"`, a user-facing `error_message`, empty content, and a
  `provider_stream_failure` diagnostic with `kind = "stream_stall"` plus
  `limit` (`event_gap` | `stream_deadline`), `elapsedMs`, `configuredMs`,
  `attempt`, `partialContentBlocks`. The turn then ends through the existing
  `STOP_REASON_ERROR` branch of `run_loop` (metrics settled as `Failure`,
  partial replaced, `MessageStart`/`MessageEnd` emitted), so the session's
  auto-retry takes over. This bounds the 43-minute hang even if events keep
  trickling.
- **Structured warning** (one line, `agent-core.stream-watchdog` component,
  via `pi_ai::log::get_logger`, matching repo conventions): elapsed, which
  limit, configured value, attempt ordinal, provider/model, partial block
  count.
- **Error classification**: typed, explicit. `pi_ai::utils::stream_failure`
  gains `KIND_STREAM_STALL` / `KIND_TRUNCATED_RESPONSE` constants;
  `pi-coding-agent`'s `provider_retry.rs` gains `is_stream_watchdog_failure()`
  and exempts watchdog failures from `cannot_replay_provider_failure()`
  (a stalled/truncated attempt never executed its partial content, so a retry
  cannot replay work). `stream_stall` is not on the permanent-kind list, so
  both `complete_with_provider_retry` and the session's `is_retryable_error`
  retry or fail over instead of treating the stall as terminal.

### Why the abort returns a provider-failure message, not `Err`

`stream_assistant_response_inner` errors propagate out of the agent loop and
are converted by `Agent::handle_run_failure` into an `agent_lifecycle_failure`
diagnostic, which `is_agent_lifecycle_failure` permanently refuses to retry.
Returning the stall as a terminal error assistant message (the same shape every
provider already uses for mid-stream failures) keeps the retry path intact.
This is the "explicit typed classification" path from the task brief: the
classification is the `provider_stream_failure` diagnostic kind, not
Display-string heuristics.

## Fix 2 - truncated-response surfacing + opt-in retry (default OFF)

- **Telemetry** (always on): when a completed assistant message ends with
  `stop_reason == "length"` (`pi_ai::types::STOP_REASON_LENGTH`), one
  structured warning is logged (`agent-core.stream-watchdog`), including
  `outputTokens` / `inputTokens` / `totalTokens` when the provider reported
  usage, plus provider/model and attempt ordinal. No behavior change.
- **Opt-in retry** (`PRIME_AGENT_RETRY_TRUNCATED_RESPONSE`, default off;
  `1/true/yes/on`, case-insensitive, enables; anything else keeps it off):
  when enabled and the turn's message has `stop_reason == "length"`, the
  message is converted into the same retryable provider-failure shape as Fix 1
  (`kind = "truncated_response"`, usage preserved, content dropped so the
  replay guard stays open) and the turn ends through the same retry path as
  the stall error. When disabled, the truncated message is accepted exactly as
  before.
- Default-off is deliberate: legitimate long generations (large code writes)
  can hit length caps, and blind retry changes semantics. The watchdog (Fix 1)
  is the default-on protection; this flag is for operators who want
  truncation to fail over.
- Surfacing happens in both completion paths of the pump loop (the
  `Done`/`Error` terminal-event branch and the post-loop `result()` branch),
  before `finish_request_metrics`, so an opted-in conversion settles the
  attempt as `Failure` rather than `Success`.

## Environment variables

| Variable | Default | Clamp | Disable | Unparseable |
| --- | --- | --- | --- | --- |
| `PRIME_AGENT_STREAM_EVENT_GAP_MS` | `120000` | `5000..600000` | `0` | default |
| `PRIME_AGENT_STREAM_DEADLINE_MS` | `900000` | `60000..7200000` | `0` | default |
| `PRIME_AGENT_RETRY_TRUNCATED_RESPONSE` | off | - | any value except `1/true/yes/on` | off (default) |

## Files changed

- `crates/pi-ai/src/utils/stream_failure.rs` - `KIND_STREAM_STALL`,
  `KIND_TRUNCATED_RESPONSE` constants.
- `crates/pi-agent-core/src/stream_watchdog.rs` (new) - config parsing, timers,
  stall/truncation message builders, logging, unit tests, shared env-guard
  test fixtures.
- `crates/pi-agent-core/src/lib.rs` - register the module.
- `crates/pi-agent-core/src/agent_loop.rs` - watchdog wiring in the pump loop
  (`WatchdogNext` select arms + deadline head-check), `abort_stalled_stream`
  tail, truncated-response surfacing in both completion branches,
  `stream_watchdog_loop_tests`.
- `crates/pi-agent-core/Cargo.toml` - dev-dependency `tokio` `test-util`
  feature for `start_paused` clock tests (no runtime dependency change).
- `crates/pi-coding-agent/src/core/provider_retry.rs` -
  `is_stream_watchdog_failure`, replay-guard exemption, tests.

No performance-metric measurements or schema were added or changed (the PR
must stay mergeable against open #117, which also touches `agent_loop.rs` and
`provider_retry.rs`; hunks here are small and localized).

## Tests

`pi-agent-core` (`cargo test -p pi-agent-core --lib stream_watchdog`):

- `watchdog_defaults_apply_when_the_environment_is_unset`
- `watchdog_values_clamp_into_their_bounds`
- `watchdog_zero_disables_and_unparseable_values_fall_back_to_defaults`
- `watchdog_environment_round_trip`
- `truncated_response_flag_matches_only_enabled_values`
- `truncated_response_flag_defaults_off_from_the_environment`
- `stalled_stream_aborts_within_the_event_gap_with_a_retryable_error`
  (never-yielding StreamFn, paused clock)
- `stream_deadline_bounds_a_trickling_stream` (events every 20s, deadline 60s)
- `paced_events_within_the_gap_do_not_trip_the_watchdog` (events every 4s over
  12s against a 5s gap: proves the timer resets on every event)
- `length_capped_response_is_accepted_and_logged_by_default` (asserts the
  accepted message is unchanged and exactly one structured warning with
  `outputTokens` is emitted)
- `length_capped_response_becomes_a_retryable_failure_when_opted_in`

`pi-coding-agent` (`cargo test -p pi-coding-agent --lib provider_retry`):

- `stream_watchdog_failures_are_retryable_even_with_partial_content`
- `complete_with_provider_retry_retries_a_stream_stall_failure`
- all 16 pre-existing `provider_retry` tests still pass.

## Validation

Run from the tree root via the `cargo.cmd` wrapper (MSVC toolchain):

- `cargo.cmd check --locked --workspace --all-targets` - pass
  (pre-existing warnings in `pi-ai` and `pi-agent-core/tests/rlm_t10.rs`
  remain; none introduced by this change; see "Pre-existing issues" below).
- `cargo.cmd test --locked -p pi-agent-core --lib stream_watchdog` - 11/11
  pass.
- `cargo.cmd test --locked -p pi-coding-agent --lib provider_retry` - 17 pass,
  1 unrelated failure (`dispatch_delivery_tests::provider_retry_reuses_the_same_durable_input_without_duplicate_transcript_entries`,
  a name collision with the filter); it passes in the full `--lib` run, see
  "Pre-existing issues" below.
- `cargo.cmd test --locked -p pi-agent-core --lib -- --test-threads=1` (the CI
  invocation for the shared crates) - 73/73 pass, including the 11 watchdog
  tests.
- `python scripts/check-rust-layout.py` - pass.
- Full `cargo.cmd test --locked -p pi-coding-agent --lib` (extra diligence,
  not a required gate): 3076 passed, 29 failed, 6 ignored. All 29 failures are
  machine-environment or parallel-run issues unrelated to this change (details
  under "Pre-existing issues"); the mission-relevant
  `provider_retry_reuses_the_same_durable_input_without_duplicate_transcript_entries`
  PASSED in this full run.

### Pre-existing issues (verified, not touched)

- `agent_loop.rs` `mod tests` already contained an unused
  `CaptureMetrics` struct on the base commit; the resulting dead-code warning
  is pre-existing.
- `provider_retry_reuses_the_same_durable_input_without_duplicate_transcript_entries`
  (an agent-session dispatch test whose *name* matches the `provider_retry`
  filter) panics with `Theme not initialized. Call initTheme() first.` when
  run in a filtered subset where no earlier test initialized the process-wide
  theme, then times out at its own 10s bounded dispatch. A captured
  `RUST_BACKTRACE=1` run shows the panic path is
  `jev_bridge::footer_status_text` -> `theme::theme()` (theme.rs:1184) inside
  the session's extension runner - no frame touches the stream pump, the
  watchdog, or the retry classification changed here (the fixture stream
  produces a `rate_limit` failure immediately; no stall or length path is
  exercised). The test passes in a full `--lib` run, where earlier tests
  initialize the theme; CI never runs this test (its job runs only filtered
  subsets).
- The 29 full-`--lib` failures are pre-existing on this machine/upstream, not
  caused by this change:
  - `native_lifecycle_metadata_is_optional_and_never_grants_unknown_ownership`
    asserts `schemaRevision == 32` while the pristine tree's
    `DAEMON_SCHEMA_REVISION` is 37 (verified: no diff in `daemon_protocol.rs`;
    the test is out of sync with the constant on upstream `32fc5a1dc`).
  - `real_session_prompts_the_faux_provider_and_delivers_events` needs the
    untracked `.port-env/tmp` fixture directory (`PathError ... NotFound`).
  - `spawn_context_env_uses_the_shell_env_owner` fails a PATH assertion on
    this machine (`path_entry.starts_with(&bin_dir)`).
  - 10 Windows job-object containment tests and 5 kernel/repl settlement
    tests spawn real OS processes/Python kernels; the kernel settlement tests
    panic on a missing environment variable (`Err(NotPresent)` at
    `repl_manager.rs:4876`).
  - `enabled_skill_is_importable` panics `program not found` for `uv pip
    install` (uv not on this machine's PATH).
  - `admission_waiters_are_visible_to_the_daemon_snapshot` and
    `session_shutdown_helper_reports_whether_handlers_existed` fail
    machine-specific daemon/extension-runner state assertions.
  - Three failures (`cancel_clears_scheduled_messages_and_releases_waiters`,
    `catch_up_prunes_missing_sessions_and_counts_ledgers`,
    `offline_mode_skips_the_download`) pass when re-run in isolation -
    parallel-run interference; the repo's CI runs this crate's tests only in
    filtered subsets.
  - The remaining failures are UI/theme-cache (neon repaint, frozen JSON
    adapter) tests with machine-environment dependencies.
  None of the failing tests touch the stream pump, the watchdog, the retry
  classification, or length surfacing changed here, and no failure mode has a
  causal path from this diff (the pump restructure is behavior-preserving for
  streams that complete: 73/73 `pi-agent-core` lib tests and 3076
  `pi-coding-agent` tests pass, including every loop-driven faux-provider
  session test).

## Deviations from the task brief

- The stall abort does not return an `Err` from
  `stream_assistant_response_inner`; it returns the retryable error assistant
  message described above (see "Why the abort returns a provider-failure
  message"). `response.request_cancel()` and `close_stream(...)` are called
  exactly as prescribed.
- The watchdog code lives in a new `stream_watchdog.rs` module rather than
  inline in `agent_loop.rs`, keeping the `agent_loop.rs` hunks minimal for the
  eventual merge with #117. The wiring, aborts, and tests required by the
  brief are all in `agent_loop.rs` / the new module.
- `tokio` gained a `test-util` dev-dependency in `pi-agent-core` for
  `start_paused` clock control in the timing tests.
