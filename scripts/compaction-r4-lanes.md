# Compaction R4 timing-lane separation

R4 adds benchmark/example instrumentation only. It does not change production compaction, provider, persistence, cancellation, or recovery code. The default `compaction_replay` build retains the R1/R2/R3 schemas and allocator behavior.

## Build identities

Use separate targets and fresh processes. Do not compare the two configurations or calculate an optimization ratio across them.

```text
# R1-R3 behavior plus instrumented R4 lanes
cargo build --locked --offline --release -p pi-coding-agent --example compaction_replay
# Only the R4 wall lane; System allocator and no R4 phase/allocator reads
cargo build --locked --offline --release -p pi-coding-agent \
  --features compaction-r4-uninstrumented-wall --example compaction_replay
```

The feature is declared by `pi-coding-agent` but is read only by the replay example. Any future differential pair must use the same feature set, target, profile, fixtures, and lane meaning.

## One selected lane per process

```text
compaction_replay r4-lane --lane wall --fixture CASE --work WORK --out REPORT
compaction_replay r4-lane --lane allocation --fixture CASE --work WORK --out REPORT
compaction_replay r4-lane --lane replay-phase --fixture CASE --work WORK --out REPORT \
  --mock-wait-ms 25 --generation-units 3 --inject none
compaction_replay r4-lane --lane minimal-provider --fixture CASE --work WORK --out REPORT \
  --mock-wait-ms 25 --generation-units 3 --inject none
```

The instrumented executable rejects `wall`. The wall executable rejects every other lane. Provider controls belong only to `replay-phase` and `minimal-provider`. `allocation-source` belongs only to `allocation`. Bad numbers, unknown options, wrong build identity, or mixed lane options fail before output creation.

Every report has `schema: compaction-r4-lane-v1`, exact lane/build/feature/profile identity, fresh child PID and executable SHA-256, fixture identity, a no-network label, and `synthetic_durations: non-decision`. Python `scripts/compaction_r4_lanes.py validate` enforces exact field sets and rejects booleans, strings, non-finite numbers, malformed JSON, wrong lane identity, and moved duration fields.

### 1. Uninstrumented operator wall

The feature-selected wall binary uses Rust's ordinary `System` global allocator. Its selected path creates no `PhaseMarks`, reads no allocation counters, and emits no internal duration. Run it through the external driver:

```text
python scripts/compaction_r4_lanes.py wall --exe WALL_EXE --fixture CASE \
  --work WORK --child-out CHILD.json --out WALL.json
```

The Python parent measures `time.perf_counter_ns` around one fresh child from spawn to terminal. `compaction-r4-external-wall-v1` contains that duration only in `external_wall.duration_ms`. The child and parent PIDs are matched. No internal duration may appear in the wall child.

`launch_state` is the directly observed `fresh-process`. `page_cache_state` and `thermal_state` are `unobserved`; a new PID, elapsed time, or earlier run never justifies a cold/warm page-cache label.

### 2. Allocation

A separate instrumented process reads the existing benchmark `CountingAllocator` counters through Rust `GlobalAlloc`. It reports actual allocation event count and requested bytes for the documented post-preparation/compact scope. These are not retained bytes or an estimate. An unavailable run emits `status: unavailable`, null source/counters, and a reason. Zero is never an unavailable sentinel.

The allocation report has no phase or external wall duration. Use `--allocation-source unavailable` only for the explicit unavailable qualification path.

### 3. Replay phase

`r4-local-phase-v1` rows record actual `std::time::Instant` boundaries. Every row names its start/end marks, exact boundary, overhead label, and `local-only` CPU scope. The ordinary successful phase set is:

- `prepare_compaction`: around real `prepare_compaction`.
- `compact_pre_provider`: real `compact` entry to summary runner entry.
- `compact_post_provider`: summary runner return to real `compact` exit.
- `append_compaction`: real durable `append_compaction` plus flush.
- `build_continuation`: real `build_session_context(None)` after append.

No local value is calculated by subtracting a provider value. No provider wait or generation interval lies inside these boundaries. The mock call envelope is intentionally not relabeled as local CPU.

### 4. Minimal provider

This lane adds `convert_to_llm`, conversation serialization, summary dispatch preparation, and `response_parse` boundaries. `response_parse` starts after terminal mock event emission and ends at summary runner return, so stream consumption/parsing excludes the controlled wait and generation. The provider is a registered in-process deterministic fixture. Its base URL is non-routable and no provider/network API is used.

`provider.wait.requested_ms` is the controlled CLI input supplied to `tokio::time::sleep`. Its observed duration is null with an unavailable reason. `generation.summary_render_iterations` is a controlled work repetition count, not a duration or throughput estimate. Provider controls and quantities stay under `provider`; local duration rows stay under `local_replay`. The mock call passes `ProviderRetryPolicy { enabled: false, max_retries: 0, .. }`, so synthetic failure accounting contains exactly one attempt and no retry wait.

## Continuation, failure, and cancellation

The useful-continuation boundary is real `prepare_compaction` -> `compact` -> `SessionManager::append_compaction` -> `build_session_context(None)`. A successful receipt contains the rebuilt message count, exact normalized context SHA-256, and exactly one appended compaction entry. An injected provider failure or pre-provider cancellation commits no continuation; its count/digest are null with a reason. Cancellation before provider entry has no fabricated local compact phases and zero provider calls.

Run `--inject failure` or `--inject cancelled` only in the two provider-reporting lanes. Expected injected outcomes produce typed receipts. Unexpected compaction failure is a process error. The qualification fixture uses a 1,000-token keep-recent setting and deliberately stays on the non-splitting, single-summary-call boundary. A split turn or chunked/multiple summary call is rejected explicitly rather than being mislabeled as the minimal provider boundary.

## Qualification and non-goals

`python scripts/test_compaction_r4_lanes.py OUT.json` runs deterministic synthetic positive/negative controls for lane exclusivity, labels, strict numeric and schema rejection, unavailable allocation counters, provider/local separation, continuation accounting, malformed JSON, cancellation, and failure receipts. Synthetic values are non-decision checks, not measured Optimus performance.

Full workspace tests, real all-fixture campaigns, real providers, Optimus timing distributions, performance admission, and production optimization are outside R4 qualification. This tooling measures the existing compaction path; it does not assert that every production provider has the same internal boundaries.
