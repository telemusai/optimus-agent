# Local performance metrics

Prime's performance metrics are a disposable, process-local JSONL sidecar. They are off by default. They do not change session JSONL, provider checkpoints, prompts, tool output, provider request bodies, or recovery journals.

## Enablement and files

The coding-agent boundary creates a recorder only when `PRIME_AGENT_PERFORMANCE_METRICS` is exactly one of `1`, `true`, `yes`, or `on` (case-insensitive). `PRIME_AGENT_PERFORMANCE_METRICS_DIR` can select an absolute sidecar root. The default is `<agentDir>/performance-metrics`.

Each recorder uses a unique file named `performance-v1-<session>-<instance>.jsonl`. Rotation creates `.1`, `.2`, and `.3` by default. The file-count and byte limits apply to one recorder instance and its own prefix only. They do not currently impose a directory-wide bound across repeated session reopen or unrelated recorder instances. Directory-wide retention remains a live gate because deleting files from active or unrelated sessions is unsafe without an ownership protocol. Sidecars can be deleted without session rollback. Disabling the opt-in is the complete behavior rollback.

Defaults and hard clamps:

| Limit | Default | Accepted range |
|---|---:|---:|
| Buffered records | 512 | 1–4096 |
| Buffered UTF-8 bytes | 256 KiB | 1 KiB–4 MiB and no more than one file minus one record |
| One encoded record | 8 KiB | 512 B–16 KiB and no more than half a file |
| One file | 4 MiB | 4 KiB–64 MiB |
| Files per recorder | 4 | 1–16 |
| Flush interval | 1000 ms | 100–60000 ms |
| Best-effort close wait | 1000 ms | 1–10000 ms |

`record()` only validates, encodes, and appends to the bounded memory buffer. File work uses asynchronous APIs on a timer, explicit `flush()`, or `close()`. There is no synchronous write per streamed token. Flush is single-flight: calls made while file I/O is pending share the same promise and coalesce into at most one boolean-requested follow-up drain, rather than building a promise queue. `close()` is best effort and resolves after its configured bound even if file I/O never settles. Full buffers, oversized records, encoding errors, rotation failures, and write failures drop disposable metrics rather than block an agent. The next successful flush writes a `recorder` record with `dropped_count`.

## Common API

`@earendil-works/pi-agent-core` exports these exact symbols from `packages/agent/src/performance-metrics.ts`:

- `PERFORMANCE_METRICS_SCHEMA_VERSION`
- `PerformanceMetricOperation`
- `PerformanceMetricOutcome`
- `PerformanceMetricMeasurement`
- `PerformanceMetricComponent`
- `PerformanceMetricCorrelation`
- `PerformanceMetricIdentity`
- `PerformanceMetricUsageV1`
- `PerformanceMetricEvent`
- `PerformanceMetricRecordV1`
- `PerformanceMetricRecorder`
- `AgentLoopLogicalRequestSettlement`
- `AgentLoopPerformanceMetrics`
- `elapsedMetricMs(start, end)`
- `safeRecordPerformanceMetric(recorder, event)`
- `performanceMetricUsageFromAssistant(message)`

`@earendil-works/pi-coding-agent` exports these exact symbols from `packages/coding-agent/src/core/performance-metrics.ts`:

- `PerformanceMetricFileIO`
- `LocalPerformanceMetricRecorderOptions`
- `EnvironmentPerformanceMetricRecorderOptions`
- `LocalPerformanceMetricRecorder`
- `createLocalPerformanceMetricRecorder(options)`
- `createLocalPerformanceMetricRecorderFromEnvironment(options)`

The two factory functions return `undefined` if local metrics are disabled or recorder construction is unsafe. Callers must preserve that as the normal no-metrics path.

### Operation and measurement allowlists

Operations are `logical_request`, `provider_attempt`, `tool`, `snapshot`, `compaction`, `file_retry`, `session_reopen`, `session_input`, and `recorder`. The Rust host also records `compaction_prepare`, `compaction_history`, `compaction_prefix`, `compaction_native`, `compaction_persist`, and `compaction_restore`.

Measurements are `total_ms`, `wait_ms`, `dispatch_to_response_headers_ms`, `transport_open_ack_ms`, `dispatch_to_first_event_ms`, `dispatch_to_first_visible_ms`, `local_gateway_wait_ms`, `upstream_wait_ms`, `serialization_ms`, `serialization_cpu_ms`, `write_ms`, `queue_ms`, `next_cell_delay_ms`, `reopen_ms`, `serialized_bytes`, `written_bytes`, `read_bytes`, `retry_count`, `attempt_count`, `attempt_ordinal`, and `dropped_count`.

`transport_open_ack_ms` is the pre-network `onPayload` edge to a transport-level send acknowledgement. A WebSocket transport has no HTTP response yet when it reports that its socket accepted the bytes, so it calls `onResponse` with the headers `x-optimus-transport: websocket` and `x-optimus-response-edge: transport_send_ack`. The host records that instant here and leaves `dispatch_to_response_headers_ms` `null`, because no HTTP header edge was observed. A response without the ack marker keeps the normal header-edge meaning, so the two stages are never mixed. The Azure WebSocket `transport_open_ack_ms` distribution and the SSE `dispatch_to_response_headers_ms` distribution measure different events and must not be compared as one population.

A measurement is a finite nonnegative number or `null`. `null` means unavailable. It is not zero. Unknown keys and arbitrary runtime fields are removed. Correlation and provider/model/API strings are control-character sanitized and length bounded. The schema has no prompt, content, arguments, error text, path, header, credential, reasoning text, or checkpoint field.

Every persisted record adds:

```ts
{
  schemaVersion: 1,
  sequence: number,
  recordedAt: string,
  correlation: {
    sessionId: string,
    logicalRequestId?: string,
    providerAttemptId?: string,
    toolCallId?: string
  }
}
```

`recordedAt` is only for log ordering. All elapsed values come from `PerformanceMetricRecorder.monotonicNow()`. A caller measures one operation as follows:

```ts
const startedAt = recorder?.monotonicNow();
try {
  await operation();
  safeRecordPerformanceMetric(recorder, {
    operation: "snapshot",
    identity: { component: "snapshot" },
    outcome: "success",
    measurements: {
      total_ms: elapsedMetricMs(startedAt, recorder?.monotonicNow()),
      serialization_ms: measuredSerializationWallMs,
      serialization_cpu_ms: measuredSerializationCpuMs,
      write_ms: measuredWriteMs,
      queue_ms: measuredQueueMs,
      serialized_bytes: measuredSerializedBytes,
      written_bytes: measuredWrittenBytes,
      next_cell_delay_ms: measuredNextCellDelayMs,
    },
  });
} catch (error) {
  safeRecordPerformanceMetric(recorder, {
    operation: "snapshot",
    identity: { component: "snapshot" },
    outcome: "failure",
    measurements: { total_ms: elapsedMetricMs(startedAt, recorder?.monotonicNow()) },
  });
  throw error;
}
```

Never pass `error`, tool names, file paths, source values, or serialized data to the event. Telemetry must be recorded after authoritative work, and failure to record must be ignored.

## Agent request semantics

`AgentOptions.performanceMetrics`, public `Agent.performanceMetrics`, and `AgentLoopConfig.performanceMetrics` accept:

```ts
{
  recorder: PerformanceMetricRecorder;
  logicalRequestId?: string;
  logicalRequestStartedAt?: number;
  providerAttemptNumber?: number;
  hostOwnsLogicalRequestTerminal?: boolean;
  logicalRequestSettlement?: AgentLoopLogicalRequestSettlement;
}
```

The host can reuse `logicalRequestId`, preserve `logicalRequestStartedAt`, and increment `providerAttemptNumber` around a retry. `providerAttemptNumber` becomes `attempt_ordinal`; it is only a host-observed stream invocation ordinal. It is not proof of an SDK-internal HTTP attempt. `attempt_count` is the count of host-observed attempts in the group and is reported once the group's own settlement knows it. Unknown upstream/transport-internal retries remain unavailable and are never inferred.

`hostOwnsLogicalRequestTerminal` and its process-local `logicalRequestSettlement` handle are internal and never enter provider options. The handle is shared only by attempts in one host retry group and makes settlement idempotent across cancellation races. When host ownership is true, the agent loop still emits each `provider_attempt` but leaves the single outer `logical_request` terminal to the host. `AgentSession` uses that boundary for its local retry group, so total logical elapsed time includes the locally observed retry wait and settles once on success, exhaustion, failure, or cancellation. A standalone `Agent` leaves the flag absent and emits its one-attempt logical terminal itself.

The agent loop removes `performanceMetrics` before calling the provider. With metrics enabled it wraps the existing callbacks only:

- `wait_ms`: for a one-attempt logical request, start to successful completion of the provider's pre-network `onPayload` callback. It is `null` for a multi-attempt retry group because start-to-final-dispatch includes prior provider execution and is not a measured wait.
- `dispatch_to_response_headers_ms`: that pre-network callback edge to the provider's `onResponse` callback.
- `dispatch_to_first_event_ms`: the same edge to the first assistant stream event.
- `dispatch_to_first_visible_ms`: the same edge to the first non-empty text delta. A reasoning-only stream has `null` visible latency.
- `total_ms`: start to terminal response, error, or cancellation. For an outer retry-group logical record this is the whole group; its dispatch split measurements describe only the final locally observed attempt.

The `onPayload` edge is not a socket-write timestamp. It can include provider SDK work before actual I/O. Client-observed response latency can include the local gateway, network, and upstream work. Therefore `local_gateway_wait_ms` and `upstream_wait_ms` stay `null` unless a separately instrumented component establishes them. They are structurally unavailable to the client, not zero and not unmeasured-but-imminent. Client TTFT is never called server queue time.

`transport_websocket` is `1` for a WebSocket attempt and `0` for an SSE attempt. Both the Azure/`github-copilot` WebSocket transport and the `openai-codex` native WebSocket path set it: the Codex SSE path labels its real HTTP response, and a Codex WebSocket attempt labels itself through the content-free `transport_ws` observation stage after the payload is sent. A provider that reached the wire therefore always carries a transport label, and a missing label means the attempt never reached the wire.

`attempt_count` is the number of provider attempts the logical request's own settlement observed, so it is reported for a completed group. A `provider_attempt` record keeps `attempt_count` `null`: one record is one attempt, and its `attempt_ordinal` already says which one it is, so a per-attempt count of `1` would add no information and could be misread as the group count. `provider_attempt.wait_ms` is reported only for ordinal 1: for that attempt it is the same request-start to `onPayload` window as `logical_request.wait_ms`. A retried attempt has no single honest wait and stays `null`.

Each local stream invocation emits one terminal `provider_attempt`. A standalone one-attempt Agent also emits one `logical_request`. A retry-owning host emits exactly one outer logical terminal for the complete retry group. Authoritative raw usage belongs only to `provider_attempt`; the outer logical terminal omits usage. Reporting must not sum nested operation durations. Tool execution emits one terminal `tool` record. Stream deltas do not write metric records.

## Usage and overlap

### Rust compaction phases

Compaction uses the session recorder when metrics are enabled. A start event has outcome `started` (legacy files may omit the outcome); its terminal event reuses the same correlation IDs with `success`, `failure`, `cancelled`, or `unavailable`, and only terminal rows count as attempts. Group related phases by `actionId`. Preparation includes authentication, history selection, and extension preparation. Persistence measures the durable compaction append. Restoration measures live-context replacement, extension notification, and kernel/provider restoration. The outer `compaction` duration includes the complete shared operation; phase durations overlap and must not be summed.

History and split-turn prefix summaries have separate logical request IDs. Each actual summary completion invocation, including a local retry, records one terminal `provider_attempt` with `identity.component: "compaction"`; these attempts must be separated from ordinary agent requests. The existing content-free observer records response headers, first raw/thinking/tool/text event, stream terminal, local drain, and provider usage when available. Its stages are `raw_event`, `thinking`, `tool`, `text`, `terminal`, and `transport_ws`, and its contract is append-only: a new stage is additive, and an unknown stage must be ignored rather than treated as an error. `transport_ws` records only "the payload was sent over a WebSocket" for a provider whose WebSocket transport never reports a response header edge; it carries no content or timing. A fast HTTP 200/header event does not measure completion of the response body. Unknown timings remain null. Error and cancellation outcomes contain no provider error text, prompts, summaries, reasoning text, request headers, credentials, or checkpoints.

Native checkpoint requests retain their own `compaction_native` phase. Native providers without stream observations have only their measured phase duration. An explicit unsupported native result is `unavailable`; malformed checkpoints are failures. A cancelled or dropped split-summary operation cancels its child requests without cancelling the parent session token.

Rust queue records also include `input_agent_message`: 1 identifies a structured agent message; 0 identifies other input and is not, by itself, proof of human origin. This field contains no message text.

### Agent-message delivery telemetry (D-04)

Cross-worker agent messaging is at-most-once by design: once a payload may have been accepted by the target worker, it is never replayed. The supervisor therefore writes a bounded, content-free delivery journal at `<descriptor dir>/agent-message-delivery-journal.jsonl`, one JSON line per attempted forward.

```ts
{
  version: 1;
  sourceActiveSessionId?: string;
  targetActiveSessionId: string;
  messageId?: string;          // the target worker's own `agentmsg_<uuid>` receipt id
  outcome: "delivered" | "queued" | "rejected" | "uncertain";
  reasonCode?: string;         // fixed code, never raw error text
  recordedAt: string;
}
```

Read it as follows:

- One record per attempted forward, written only by the supervisor hop that forwards the message.
- `uncertain` means the forward was attempted but the result is unknown, for example a lost response or an invalid receipt. The send is not retried, so `uncertain` is a terminal outcome for that send.
- `rejected` means the message provably never left the supervisor, for example a self-send, a denied reach, a missing source, or a target worker that is no longer running.
- The journal is telemetry only. It never changes a delivery result, never retries, and never fails a send. Records are bounded (`512` retained) and the file self-compacts.
- Records carry no message text, session names, sender names, or raw error strings.

The report accepts the Rust stream measurements (`dispatch_to_first_raw_ms`, `dispatch_to_first_thinking_ms`, `dispatch_to_first_tool_ms`, `dispatch_to_first_text_ms`, `dispatch_to_network_terminal_ms`, `local_drain_ms`, and `transport_websocket`) without treating unknown values as zero.

`PerformanceMetricUsageV1` keeps provider observations and local estimates separate:

```ts
{
  source: "provider" | "local_estimate";
  inputTokens: number | null;
  cachedInputTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  totalTokens: number | null;
  cachedInputIncludedInInput: boolean | null;
  reasoningIncludedInOutput: boolean | null;
  estimator?: string;
}
```

`queue_ms` for `session_input` means acceptance to primary delivery of that input, and that window includes any in-flight turn the input waited behind. A long `queue_ms` is therefore not by itself evidence of a slow queue implementation; it is the elapsed wait for the session to become free. The record carries no currently-running action id, so a long wait cannot yet be attributed to one specific turn.

`inputTokens` and `outputTokens` retain the concrete provider observation's own inclusive or exclusive meaning. When an overlap flag is `true`, the nested category must not be added again. When it is `false`, the categories are disjoint, but the provider `totalTokens` still remains authoritative as its own field. When it is null, no arithmetic is justified. A local estimate must use `source: "local_estimate"` and name its estimator. Neither form is a bill or cost.

The existing normalized `AssistantMessage.usage` initializes every field to zero, including when a provider supplied no usage. The fallback therefore treats each zero field as unavailable, even when a different normalized field is positive, and records only positive field values. Exact zero is authoritative only through a concrete provider's raw observation callback. Overlap and reasoning details remain null unless that concrete provider established them. This avoids manufacturing authoritative zeros.

Responses-family raw usage can carry `output_tokens_details.reasoning_tokens`. Its optional, local-only observation callback preserves raw inclusive `input_tokens` and `output_tokens`, exact zero values, cached and reasoning subcounts, and overlap flags. Callback errors are contained and the callback is never serialized into a provider request. Providers without an authoritative raw observation keep each unavailable field null.

## Hooks for later builders

All hooks use the same recorder instance for a session.

- IO: record one `file_retry` after the retry operation completes, with `retry_count` and monotonic `total_ms`. The SDK records one `session_reopen` only when `SessionManager` exposes bytes from the primary transcript read. That narrow event sets actual `read_bytes` and leaves `reopen_ms` and `total_ms` null because no shared outer load clock exists. Header and repair probes are excluded. Do not put paths or session text in a record.
- Context: pass retry correlation into `Agent.performanceMetrics`; record one `compaction` with `total_ms` and authoritative usage if available. Text and native compaction remain distinct through `identity.component` plus the existing provider/API identity; do not add a free-form route label.
- Kernel: record one `snapshot` after each completed/cancelled/failed snapshot with separately measured `queue_ms`, wall-clock `serialization_ms`, `serialization_cpu_ms`, `write_ms`, `serialized_bytes`, `written_bytes`, and `next_cell_delay_ms`. CPU time is the serializer thread CPU clock where the Python runtime provides it; an explicitly named process CPU fallback or null must be used otherwise. Values unavailable across the Python/TypeScript boundary stay null.

The recorder must not become a session/recovery dependency. Flush on normal isolated disposal, but never delay or fail authoritative shutdown solely to save telemetry.

## Local reporting and benchmark

Run the summary with the installation's Node binary:

```text
node packages/coding-agent/scripts/summarize-performance-metrics.mjs --dir <metrics-root> --json
```

The summary keeps provider and estimated usage separate and never double-adds overlap categories.

The bounded synthetic overhead benchmark is:

```text
npm exec -- tsx packages/coding-agent/scripts/benchmark-performance-metrics.ts --output <evidence-json>
```

It uses three synthetic sessions, at least 200 logical requests per session, warm-up, and repeated measured trials. It makes no provider calls.
