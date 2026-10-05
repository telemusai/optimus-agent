//! Port of packages/agent/src/performance-metrics.ts
//!
//! `PerformanceMetricRecorder` is a TypeScript interface implemented by the
//! coding-agent. In Rust it is a trait object; every call site keeps the
//! TypeScript defensive containment so a third-party recorder failure can never
//! change agent behaviour.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use indexmap::IndexMap;
use pi_ai::types::AssistantMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PERFORMANCE_METRICS_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceMetricOperation {
    LogicalRequest,
    ProviderAttempt,
    Tool,
    Snapshot,
    Compaction,
    CompactionPrepare,
    CompactionHistory,
    CompactionPrefix,
    CompactionNative,
    CompactionPersist,
    CompactionRestore,
    FileRetry,
    SessionReopen,
    SessionInput,
    UiInput,
    UiInputAck,
    UiEventApply,
    UiTick,
    UiRender,
    UiMenuOpen,
    UiSessionOpen,
    Recorder,
    DaemonLifecycle,
}

impl PerformanceMetricOperation {
    pub fn as_str(self) -> &'static str {
        match self {
            PerformanceMetricOperation::LogicalRequest => "logical_request",
            PerformanceMetricOperation::ProviderAttempt => "provider_attempt",
            PerformanceMetricOperation::Tool => "tool",
            PerformanceMetricOperation::Snapshot => "snapshot",
            PerformanceMetricOperation::Compaction => "compaction",
            PerformanceMetricOperation::CompactionPrepare => "compaction_prepare",
            PerformanceMetricOperation::CompactionHistory => "compaction_history",
            PerformanceMetricOperation::CompactionPrefix => "compaction_prefix",
            PerformanceMetricOperation::CompactionNative => "compaction_native",
            PerformanceMetricOperation::CompactionPersist => "compaction_persist",
            PerformanceMetricOperation::CompactionRestore => "compaction_restore",
            PerformanceMetricOperation::FileRetry => "file_retry",
            PerformanceMetricOperation::SessionReopen => "session_reopen",
            PerformanceMetricOperation::SessionInput => "session_input",
            PerformanceMetricOperation::UiInput => "ui_input",
            PerformanceMetricOperation::UiInputAck => "ui_input_ack",
            PerformanceMetricOperation::UiEventApply => "ui_event_apply",
            PerformanceMetricOperation::UiTick => "ui_tick",
            PerformanceMetricOperation::UiRender => "ui_render",
            PerformanceMetricOperation::UiMenuOpen => "ui_menu_open",
            PerformanceMetricOperation::UiSessionOpen => "ui_session_open",
            PerformanceMetricOperation::Recorder => "recorder",
            PerformanceMetricOperation::DaemonLifecycle => "daemon_lifecycle",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceMetricOutcome {
    /// An in-flight start row. A paired terminal row with the same correlation
    /// IDs follows; only terminal outcomes (success/failure/cancelled/
    /// unavailable/timeout) count as completed attempts.
    Started,
    Success,
    Failure,
    Cancelled,
    Unavailable,
    /// The operation exceeded an explicit deadline. Terminal, like the other
    /// non-started outcomes. Used by the UI ack deadline so a prompt that never
    /// receives its acknowledgement is classified as a timeout rather than
    /// reported as an inflated failure duration.
    Timeout,
}

impl PerformanceMetricOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            PerformanceMetricOutcome::Started => "started",
            PerformanceMetricOutcome::Success => "success",
            PerformanceMetricOutcome::Failure => "failure",
            PerformanceMetricOutcome::Cancelled => "cancelled",
            PerformanceMetricOutcome::Unavailable => "unavailable",
            PerformanceMetricOutcome::Timeout => "timeout",
        }
    }
}

/// A3: bounded failure-classification token for failure/cancelled events.
///
/// Monitored failure events carried no error class, status, or message (0 of
/// 4,466 in a 15-day window), so even a 141-attempt auth storm stayed
/// undiagnosable. The class is a fixed vocabulary derived from structured
/// provider failures, transport errors, or bounded message heuristics; it
/// never carries raw provider text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceMetricErrorClass {
    Auth,
    RateLimit,
    Network,
    Server,
    Client,
    Timeout,
    Cancelled,
    Unknown,
}

impl PerformanceMetricErrorClass {
    pub fn as_str(self) -> &'static str {
        match self {
            PerformanceMetricErrorClass::Auth => "auth",
            PerformanceMetricErrorClass::RateLimit => "rate_limit",
            PerformanceMetricErrorClass::Network => "network",
            PerformanceMetricErrorClass::Server => "server",
            PerformanceMetricErrorClass::Client => "client",
            PerformanceMetricErrorClass::Timeout => "timeout",
            PerformanceMetricErrorClass::Cancelled => "cancelled",
            PerformanceMetricErrorClass::Unknown => "unknown",
        }
    }

    /// Inverse of [`as_str`] for recorder-side validation of persisted tokens.
    pub fn from_token(token: &str) -> Option<Self> {
        Some(match token {
            "auth" => PerformanceMetricErrorClass::Auth,
            "rate_limit" => PerformanceMetricErrorClass::RateLimit,
            "network" => PerformanceMetricErrorClass::Network,
            "server" => PerformanceMetricErrorClass::Server,
            "client" => PerformanceMetricErrorClass::Client,
            "timeout" => PerformanceMetricErrorClass::Timeout,
            "cancelled" => PerformanceMetricErrorClass::Cancelled,
            "unknown" => PerformanceMetricErrorClass::Unknown,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceMetricMeasurement {
    TotalMs,
    WaitMs,
    DispatchToResponseHeadersMs,
    TransportOpenAckMs,
    DispatchToFirstEventMs,
    DispatchToFirstVisibleMs,
    DispatchToFirstRawMs,
    DispatchToFirstThinkingMs,
    DispatchToFirstToolMs,
    DispatchToFirstTextMs,
    DispatchToNetworkTerminalMs,
    LocalDrainMs,
    TransportWebsocket,
    LocalGatewayWaitMs,
    UpstreamWaitMs,
    SerializationMs,
    SerializationCpuMs,
    SerializationMaxVariableMs,
    SerializationSlowVariables,
    SerializationSavedMs,
    SerializationSkippedMs,
    SerializationNativeValues,
    SerializationDillValues,
    SerializationNativeMs,
    SerializationDillMs,
    SerializationNativeProbeMs,
    SerializationNativeProbeBytes,
    SerializationNativeProbeAttempts,
    SerializationNativeProbeRejected,
    SerializationFragmentPrepareMs,
    SerializationFragmentWriteMs,
    SerializationFragmentBytes,
    SerializationFragmentSegments,
    SerializationBufferResetMs,
    SerializationBlobExtractMs,
    SerializationNativeProbeSavedMs,
    SerializationNativeProbeSkippedMs,
    SerializationEnvelopeCountMs,
    SerializationEnvelopeCountCalls,
    SerializationEnvelopeWriteMs,
    SnapshotCasCaptures,
    SnapshotLegacyCaptures,
    WriteMs,
    QueueMs,
    InputAgentMessage,
    NextCellDelayMs,
    ReopenMs,
    SerializedBytes,
    WrittenBytes,
    ReadBytes,
    RetryCount,
    AttemptCount,
    AttemptOrdinal,
    DroppedCount,
    FrameCount,
    UiSubmitToTaskMs,
    UiSubmitToAwaitMs,
    UiSubmitToReplyMs,
    UiSubmitToReceiptRenderMs,
    UiAttachmentGeneration,
    UiPendingCount,
    UiEventCount,
    UiTickFallbackCount,
    UiTickFallbackSkipped,
    UiAckDeadlineMs,
    RenderMs,
    DiffMs,    MaxMs,
}

impl PerformanceMetricMeasurement {
    pub fn as_str(self) -> &'static str {
        match self {
            PerformanceMetricMeasurement::TotalMs => "total_ms",
            PerformanceMetricMeasurement::WaitMs => "wait_ms",
            PerformanceMetricMeasurement::DispatchToResponseHeadersMs => "dispatch_to_response_headers_ms",
            PerformanceMetricMeasurement::TransportOpenAckMs => "transport_open_ack_ms",
            PerformanceMetricMeasurement::DispatchToFirstEventMs => "dispatch_to_first_event_ms",
            PerformanceMetricMeasurement::DispatchToFirstVisibleMs => "dispatch_to_first_visible_ms",
            PerformanceMetricMeasurement::DispatchToFirstRawMs => "dispatch_to_first_raw_ms",
            PerformanceMetricMeasurement::DispatchToFirstThinkingMs => "dispatch_to_first_thinking_ms",
            PerformanceMetricMeasurement::DispatchToFirstToolMs => "dispatch_to_first_tool_ms",
            PerformanceMetricMeasurement::DispatchToFirstTextMs => "dispatch_to_first_text_ms",
            PerformanceMetricMeasurement::DispatchToNetworkTerminalMs => "dispatch_to_network_terminal_ms",
            PerformanceMetricMeasurement::LocalDrainMs => "local_drain_ms",
            PerformanceMetricMeasurement::TransportWebsocket => "transport_websocket",
            PerformanceMetricMeasurement::LocalGatewayWaitMs => "local_gateway_wait_ms",
            PerformanceMetricMeasurement::UpstreamWaitMs => "upstream_wait_ms",
            PerformanceMetricMeasurement::SerializationMs => "serialization_ms",
            PerformanceMetricMeasurement::SerializationCpuMs => "serialization_cpu_ms",
            PerformanceMetricMeasurement::SerializationMaxVariableMs => "serialization_max_variable_ms",
            PerformanceMetricMeasurement::SerializationSlowVariables => "serialization_slow_variables",
            PerformanceMetricMeasurement::SerializationSavedMs => "serialization_saved_ms",
            PerformanceMetricMeasurement::SerializationSkippedMs => "serialization_skipped_ms",
            PerformanceMetricMeasurement::SerializationNativeValues => "serialization_native_values",
            PerformanceMetricMeasurement::SerializationDillValues => "serialization_dill_values",
            PerformanceMetricMeasurement::SerializationNativeMs => "serialization_native_ms",
            PerformanceMetricMeasurement::SerializationDillMs => "serialization_dill_ms",
            PerformanceMetricMeasurement::SerializationNativeProbeMs => "serialization_native_probe_ms",
            PerformanceMetricMeasurement::SerializationNativeProbeBytes => "serialization_native_probe_bytes",
            PerformanceMetricMeasurement::SerializationNativeProbeAttempts => "serialization_native_probe_attempts",
            PerformanceMetricMeasurement::SerializationNativeProbeRejected => "serialization_native_probe_rejected",
            PerformanceMetricMeasurement::SerializationFragmentPrepareMs => "serialization_fragment_prepare_ms",
            PerformanceMetricMeasurement::SerializationFragmentWriteMs => "serialization_fragment_write_ms",
            PerformanceMetricMeasurement::SerializationFragmentBytes => "serialization_fragment_bytes",
            PerformanceMetricMeasurement::SerializationFragmentSegments => "serialization_fragment_segments",
            PerformanceMetricMeasurement::SerializationBufferResetMs => "serialization_buffer_reset_ms",
            PerformanceMetricMeasurement::SerializationBlobExtractMs => "serialization_blob_extract_ms",
            PerformanceMetricMeasurement::SerializationNativeProbeSavedMs => "serialization_native_probe_saved_ms",
            PerformanceMetricMeasurement::SerializationNativeProbeSkippedMs => "serialization_native_probe_skipped_ms",
            PerformanceMetricMeasurement::SerializationEnvelopeCountMs => "serialization_envelope_count_ms",
            PerformanceMetricMeasurement::SerializationEnvelopeCountCalls => "serialization_envelope_count_calls",
            PerformanceMetricMeasurement::SerializationEnvelopeWriteMs => "serialization_envelope_write_ms",
            PerformanceMetricMeasurement::SnapshotCasCaptures => "snapshot_cas_captures",
            PerformanceMetricMeasurement::SnapshotLegacyCaptures => "snapshot_legacy_captures",
            PerformanceMetricMeasurement::WriteMs => "write_ms",
            PerformanceMetricMeasurement::QueueMs => "queue_ms",
            PerformanceMetricMeasurement::InputAgentMessage => "input_agent_message",
            PerformanceMetricMeasurement::NextCellDelayMs => "next_cell_delay_ms",
            PerformanceMetricMeasurement::ReopenMs => "reopen_ms",
            PerformanceMetricMeasurement::SerializedBytes => "serialized_bytes",
            PerformanceMetricMeasurement::WrittenBytes => "written_bytes",
            PerformanceMetricMeasurement::ReadBytes => "read_bytes",
            PerformanceMetricMeasurement::RetryCount => "retry_count",
            PerformanceMetricMeasurement::AttemptCount => "attempt_count",
            PerformanceMetricMeasurement::AttemptOrdinal => "attempt_ordinal",
            PerformanceMetricMeasurement::DroppedCount => "dropped_count",
            PerformanceMetricMeasurement::FrameCount => "frame_count",
            PerformanceMetricMeasurement::UiSubmitToTaskMs => "ui_submit_to_task_ms",
            PerformanceMetricMeasurement::UiSubmitToAwaitMs => "ui_submit_to_await_ms",
            PerformanceMetricMeasurement::UiSubmitToReplyMs => "ui_submit_to_reply_ms",
            PerformanceMetricMeasurement::UiSubmitToReceiptRenderMs => "ui_submit_to_receipt_render_ms",
            PerformanceMetricMeasurement::UiAttachmentGeneration => "ui_attachment_generation",
            PerformanceMetricMeasurement::UiPendingCount => "ui_pending_count",
            PerformanceMetricMeasurement::UiEventCount => "ui_event_count",
            PerformanceMetricMeasurement::UiTickFallbackCount => "ui_tick_fallback_count",
            PerformanceMetricMeasurement::UiTickFallbackSkipped => "ui_tick_fallback_skipped",
            PerformanceMetricMeasurement::UiAckDeadlineMs => "ui_ack_deadline_ms",
            PerformanceMetricMeasurement::RenderMs => "render_ms",
            PerformanceMetricMeasurement::DiffMs => "diff_ms",            PerformanceMetricMeasurement::MaxMs => "max_ms",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceMetricComponent {
    Agent,
    Provider,
    Tool,
    Snapshot,
    Compaction,
    Persistence,
    Session,
    Recorder,
}

impl PerformanceMetricComponent {
    pub fn as_str(self) -> &'static str {
        match self {
            PerformanceMetricComponent::Agent => "agent",
            PerformanceMetricComponent::Provider => "provider",
            PerformanceMetricComponent::Tool => "tool",
            PerformanceMetricComponent::Snapshot => "snapshot",
            PerformanceMetricComponent::Compaction => "compaction",
            PerformanceMetricComponent::Persistence => "persistence",
            PerformanceMetricComponent::Session => "session",
            PerformanceMetricComponent::Recorder => "recorder",
        }
    }
}

/// `Partial<Record<PerformanceMetricMeasurement, number | null>>`.
///
/// The TypeScript value type is `number | null`, so an explicit JSON null is
/// observable and must stay distinguishable from an absent key. `IndexMap`
/// keeps the insertion order an object literal has in TypeScript.
pub type PerformanceMetricMeasurements = IndexMap<PerformanceMetricMeasurement, Option<f64>>;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetricCorrelation {
    #[serde(rename = "actionId", default, skip_serializing_if = "Option::is_none")]
    pub action_id: Option<String>,
    #[serde(rename = "logicalRequestId", default, skip_serializing_if = "Option::is_none")]
    pub logical_request_id: Option<String>,
    #[serde(rename = "providerAttemptId", default, skip_serializing_if = "Option::is_none")]
    pub provider_attempt_id: Option<String>,
    #[serde(rename = "toolCallId", default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetricIdentity {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<PerformanceMetricComponent>,
    /// A4: the tool metric's tool name. 62% of monitored tool failures were
    /// unattributable because only the opaque tool-call id was recorded. Like
    /// the provider/model/API strings this is a short control-character
    /// sanitized, length-bounded identifier, never tool arguments or output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
}

/// Provider totals and categories must not be summed again. Overlap fields stay
/// null unless the emitting provider establishes their exact semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetricUsageV1 {
    pub source: String,
    #[serde(rename = "inputTokens")]
    pub input_tokens: Option<f64>,
    #[serde(rename = "cachedInputTokens")]
    pub cached_input_tokens: Option<f64>,
    #[serde(rename = "outputTokens")]
    pub output_tokens: Option<f64>,
    #[serde(rename = "reasoningTokens")]
    pub reasoning_tokens: Option<f64>,
    #[serde(rename = "totalTokens")]
    pub total_tokens: Option<f64>,
    #[serde(rename = "cachedInputIncludedInInput")]
    pub cached_input_included_in_input: Option<bool>,
    #[serde(rename = "reasoningIncludedInOutput")]
    pub reasoning_included_in_output: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator: Option<String>,
}

impl PerformanceMetricUsageV1 {
    /// `source: "provider"` with every availability field null.
    pub fn provider_unavailable() -> Self {
        Self {
            source: "provider".to_string(),
            input_tokens: None,
            cached_input_tokens: None,
            output_tokens: None,
            reasoning_tokens: None,
            total_tokens: None,
            cached_input_included_in_input: None,
            reasoning_included_in_output: None,
            estimator: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetricEvent {
    pub operation: PerformanceMetricOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation: Option<PerformanceMetricCorrelation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<PerformanceMetricIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<PerformanceMetricOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurements: Option<PerformanceMetricMeasurements>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<PerformanceMetricUsageV1>,
    /// A15: fixed lowercase stage token for staged operations (for example
    /// `daemon_lifecycle` start/stop/crash/stale_detected/relaunch). A bounded
    /// `[a-z0-9_]` token of at most 32 characters, never free-form text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    /// A3: bounded failure classification for failure/cancelled outcomes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_class: Option<PerformanceMetricErrorClass>,
    /// A3: provider HTTP status observed for the failed attempt, when any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    /// A3: bounded error text, opt-in at the recorder through
    /// `PRIME_AGENT_PERFORMANCE_METRICS_ERROR_TEXT=1` (default off). Emitters
    /// must pass [`sanitize_performance_metric_error_message`] output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

impl PerformanceMetricEvent {
    pub fn new(operation: PerformanceMetricOperation) -> Self {
        Self {
            operation,
            correlation: None,
            identity: None,
            outcome: None,
            measurements: None,
            usage: None,
            stage: None,
            error_class: None,
            http_status: None,
            error_message: None,
        }
    }
}

/// `PerformanceMetricRecordV1 extends Omit<PerformanceMetricEvent, "correlation">`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetricRecordV1 {
    pub operation: PerformanceMetricOperation,
    #[serde(rename = "schemaVersion")]
    pub schema_version: i64,
    pub sequence: i64,
    #[serde(rename = "recordedAt")]
    pub recorded_at: String,
    pub correlation: PerformanceMetricRecordCorrelation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<PerformanceMetricIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<PerformanceMetricOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurements: Option<PerformanceMetricMeasurements>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<PerformanceMetricUsageV1>,
    /// A15: fixed lowercase stage token (see [`PerformanceMetricEvent::stage`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    /// A3: bounded failure classification (see
    /// [`PerformanceMetricEvent::error_class`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_class: Option<PerformanceMetricErrorClass>,
    /// A3: provider HTTP status observed for the failed attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    /// A3: bounded opt-in error text (recorder-gated; default dropped).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// `PerformanceMetricCorrelation & { sessionId: string }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetricRecordCorrelation {
    #[serde(rename = "actionId", default, skip_serializing_if = "Option::is_none")]
    pub action_id: Option<String>,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    #[serde(rename = "logicalRequestId", default, skip_serializing_if = "Option::is_none")]
    pub logical_request_id: Option<String>,
    #[serde(rename = "providerAttemptId", default, skip_serializing_if = "Option::is_none")]
    pub provider_attempt_id: Option<String>,
    #[serde(rename = "toolCallId", default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Scope accepted by `PerformanceMetricRecorder.nextId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceMetricIdScope {
    LogicalRequest,
    ProviderAttempt,
}

/// A recorder must never make agent work wait for telemetry persistence.
///
/// TypeScript `flush()`/`close()` return promises; the Rust trait keeps them
/// synchronous because no call site in this crate awaits them.
pub trait PerformanceMetricRecorder: Send + Sync {
    fn session_id(&self) -> &str;
    fn monotonic_now(&self) -> f64;
    fn next_id(&self, scope: PerformanceMetricIdScope) -> String;
    fn record(&self, event: PerformanceMetricEvent);
    fn flush(&self);
    fn close(&self);
}

/// `@internal` Process-local exactly-once state shared by attempts in one host retry group.
#[derive(Debug, Default)]
pub struct AgentLoopLogicalRequestSettlement {
    pub settled: AtomicBool,
    pub max_provider_attempt_number: AtomicU64,
}

impl AgentLoopLogicalRequestSettlement {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_settled(&self) -> bool {
        self.settled.load(Ordering::SeqCst)
    }

    /// Returns true when this call performed the first settlement.
    pub fn settle_once(&self) -> bool {
        !self.settled.swap(true, Ordering::SeqCst)
    }

    pub fn max_provider_attempt_number(&self) -> u64 {
        self.max_provider_attempt_number.load(Ordering::SeqCst)
    }

    pub fn observe_provider_attempt_number(&self, value: u64) -> u64 {
        let mut current = self.max_provider_attempt_number.load(Ordering::SeqCst);
        while value > current {
            match self.max_provider_attempt_number.compare_exchange(
                current,
                value,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return value,
                Err(observed) => current = observed,
            }
        }
        current
    }
}

/// Optional correlation supplied by a host around one low-level agent run.
#[derive(Clone)]
pub struct AgentLoopPerformanceMetrics {
    pub recorder: Arc<dyn PerformanceMetricRecorder>,
    pub logical_request_id: Option<String>,
    pub logical_request_started_at: Option<f64>,
    /// Host-observed stream invocation ordinal, not an SDK-internal retry count.
    pub provider_attempt_number: Option<u64>,
    /// Defers the outer logical-request terminal to a host that groups local retries.
    /// The Agent loop still records each locally observed provider attempt.
    pub host_owns_logical_request_terminal: bool,
    /// `@internal` Shared only across the host-observed attempts of this logical request.
    pub logical_request_settlement: Option<Arc<AgentLoopLogicalRequestSettlement>>,
}

impl AgentLoopPerformanceMetrics {
    pub fn new(recorder: Arc<dyn PerformanceMetricRecorder>) -> Self {
        Self {
            recorder,
            logical_request_id: None,
            logical_request_started_at: None,
            provider_attempt_number: None,
            host_owns_logical_request_terminal: false,
            logical_request_settlement: None,
        }
    }
}

impl std::fmt::Debug for AgentLoopPerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentLoopPerformanceMetrics")
            .field("logical_request_id", &self.logical_request_id)
            .field("provider_attempt_number", &self.provider_attempt_number)
            .field("host_owns_logical_request_terminal", &self.host_owns_logical_request_terminal)
            .finish_non_exhaustive()
    }
}

/// `elapsedMetricMs` - monotonic deltas only; unavailable or backwards clocks
/// return `None`.
pub fn elapsed_metric_ms(start: Option<f64>, end: Option<f64>) -> Option<f64> {
    let (start, end) = (start?, end?);
    if !start.is_finite() || !end.is_finite() || end < start {
        return None;
    }
    Some(end - start)
}

/// Defensively contains third-party recorder failures at every call site.
pub fn safe_record_performance_metric(
    recorder: Option<&Arc<dyn PerformanceMetricRecorder>>,
    event: PerformanceMetricEvent,
) {
    let Some(recorder) = recorder else { return };
    // Performance telemetry is disposable and must not change agent behavior.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recorder.record(event)));
}

/// Maximum length of an opt-in `error_message` value, in characters.
pub const PERFORMANCE_METRICS_ERROR_MESSAGE_MAX_CHARS: usize = 256;

/// A3: bounds and sanitizes an error message before it can reach an event.
///
/// Control characters become `?` (matching the correlation/identity
/// sanitization contract) and the text is truncated to
/// [`PERFORMANCE_METRICS_ERROR_MESSAGE_MAX_CHARS`] characters. Recorders must
/// sanitize again before persisting: emitters cannot be trusted to pre-bound
/// third-party error text.
pub fn sanitize_performance_metric_error_message(message: Option<&str>) -> Option<String> {
    let message = message?.trim();
    if message.is_empty() {
        return None;
    }
    let sanitized: String = message
        .chars()
        .map(|character| {
            let code = character as u32;
            if code <= 0x1f || code == 0x7f {
                '?'
            } else {
                character
            }
        })
        .take(PERFORMANCE_METRICS_ERROR_MESSAGE_MAX_CHARS)
        .collect();
    if sanitized.is_empty() {
        None
    } else {
        Some(sanitized)
    }
}

/// A3: A15 stage tokens are fixed lowercase snake-case words. This bound keeps
/// the stage a closed vocabulary even if a future emitter adds one.
pub const PERFORMANCE_METRICS_STAGE_MAX_CHARS: usize = 32;

/// A3/A15: validates a stage token (`[a-z0-9_]{1,32}`), returning it unchanged
/// or `None` so no free-form text can enter the stage field.
pub fn sanitize_performance_metric_stage(stage: Option<&str>) -> Option<String> {
    let stage = stage?.trim();
    if stage.is_empty() {
        return None;
    }
    let valid = !stage.is_empty()
        && stage.len() <= PERFORMANCE_METRICS_STAGE_MAX_CHARS
        && stage
            .chars()
            .all(|character| character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_');
    if valid {
        Some(stage.to_string())
    } else {
        None
    }
}

/// A3: the failure detail one terminal event should carry. Only the bounded
/// class, status, and opt-in message; no structured provider payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceMetricFailure {
    pub class: PerformanceMetricErrorClass,
    pub http_status: Option<u16>,
    /// Already passed through [`sanitize_performance_metric_error_message`].
    pub message: Option<String>,
}

impl PerformanceMetricFailure {
    /// A user-initiated cancellation carries no provider status or text.
    pub fn cancelled() -> Self {
        Self {
            class: PerformanceMetricErrorClass::Cancelled,
            http_status: None,
            message: None,
        }
    }

    /// An unclassified failure with an optional observed status.
    pub fn unknown_with_status(http_status: Option<u16>) -> Self {
        Self {
            class: PerformanceMetricErrorClass::Unknown,
            http_status,
            message: None,
        }
    }

    /// Maps the shared provider failure classification plus its HTTP status
    /// onto the bounded metric vocabulary.
    pub fn from_stream_failure_info(info: &pi_ai::utils::stream_failure::StreamFailureInfo) -> Self {
        Self {
            class: error_class_from_stream_failure_kind(&info.kind, info.status),
            http_status: bounded_http_status(info.status),
            // `raw` is a provider payload and must never enter telemetry; the
            // short provider error type (e.g. "overloaded_error") is the only
            // bounded text this path contributes.
            message: sanitize_performance_metric_error_message(info.provider_error_type.as_deref()),
        }
    }

    /// Best-effort classification of an `anyhow` error from the agent loop:
    /// a structured [`pi_ai::utils::stream_failure::StreamFailureError`] in the
    /// chain wins, then transport errors, then bounded message heuristics.
    pub fn from_anyhow_error(error: &anyhow::Error) -> Self {
        for cause in error.chain() {
            if let Some(failure) = cause.downcast_ref::<pi_ai::utils::stream_failure::StreamFailureError>() {
                return Self::from_stream_failure_info(&failure.info);
            }
            if let Some(transport) = cause.downcast_ref::<reqwest::Error>() {
                return Self {
                    class: if transport.is_timeout() {
                        PerformanceMetricErrorClass::Timeout
                    } else {
                        PerformanceMetricErrorClass::Network
                    },
                    http_status: transport
                        .status()
                        .map(|status| status.as_u16())
                        .and_then(|status| (100..=599).contains(&status).then_some(status)),
                    message: sanitize_performance_metric_error_message(Some(&error.to_string())),
                };
            }
        }
        Self::classify_message(&error.to_string())
    }

    /// Classifies a terminal provider message: the persisted
    /// `provider_stream_failure` diagnostic is authoritative, then the raw stop
    /// reason, then the bounded error-message heuristics.
    pub fn from_assistant_message(message: &AssistantMessage) -> Self {
        for diagnostic in message.diagnostics.iter().flatten() {
            if diagnostic.type_ != "provider_stream_failure" {
                continue;
            }
            let details = diagnostic.details.as_ref();
            let kind = details
                .and_then(|details| details.get("kind"))
                .and_then(Value::as_str);
            let status = details
                .and_then(|details| details.get("status"))
                .and_then(|value| value.as_i64());
            let text = diagnostic
                .error
                .as_ref()
                .map(|error| error.message.as_str())
                .unwrap_or_default();
            return Self {
                class: error_class_from_stream_failure_kind(kind.unwrap_or("unknown"), status),
                http_status: bounded_http_status(status),
                message: sanitize_performance_metric_error_message(Some(text)),
            };
        }
        if let Some(raw) = message.stop_reason_raw.as_deref() {
            let kind = pi_ai::utils::stream_failure::classify_stream_failure(Some(raw), None);
            return Self {
                class: error_class_from_stream_failure_kind(kind, None),
                http_status: None,
                message: sanitize_performance_metric_error_message(message.error_message.as_deref()),
            };
        }
        match message.error_message.as_deref() {
            Some(text) => Self::classify_message(text),
            None => Self::unknown_with_status(None),
        }
    }

    /// Bounded message heuristics for errors that carry no structured class.
    pub fn classify_message(text: &str) -> Self {
        Self {
            class: classify_error_message_text(text),
            http_status: None,
            message: sanitize_performance_metric_error_message(Some(text)),
        }
    }
}

/// Keeps only real HTTP statuses; `0`, negatives, and redirects are dropped.
fn bounded_http_status(status: Option<i64>) -> Option<u16> {
    let status = status?;
    (100..=599).contains(&status).then_some(status as u16)
}

/// Maps the provider failure kind vocabulary (plus an optional status) onto
/// the metric error classes.
fn error_class_from_stream_failure_kind(kind: &str, status: Option<i64>) -> PerformanceMetricErrorClass {
    match kind {
        "auth" => PerformanceMetricErrorClass::Auth,
        "rate_limit" => PerformanceMetricErrorClass::RateLimit,
        "overloaded" | "server_error" => PerformanceMetricErrorClass::Server,
        "request_interrupted" => PerformanceMetricErrorClass::Cancelled,
        "refusal" | "safety" | "permission" | "invalid_request" | "malformed_response" => {
            PerformanceMetricErrorClass::Client
        }
        _ => match status {
            Some(status) if (100..=599).contains(&status) => match status {
                408 => PerformanceMetricErrorClass::Timeout,
                429 => PerformanceMetricErrorClass::RateLimit,
                401 => PerformanceMetricErrorClass::Auth,
                status if status >= 500 => PerformanceMetricErrorClass::Server,
                status if status >= 400 => PerformanceMetricErrorClass::Client,
                _ => PerformanceMetricErrorClass::Unknown,
            },
            _ => PerformanceMetricErrorClass::Unknown,
        },
    }
}

/// Heuristic classifier for unstructured error text. Ordered so that the most
/// specific transport verdicts win; every pattern is lowercase-ASCII only.
fn classify_error_message_text(text: &str) -> PerformanceMetricErrorClass {
    static PATTERNS: once_cell::sync::Lazy<Vec<(&'static str, PerformanceMetricErrorClass)>> =
        once_cell::sync::Lazy::new(|| {
            vec![
                ("timed out", PerformanceMetricErrorClass::Timeout),
                ("timeout", PerformanceMetricErrorClass::Timeout),
                ("deadline", PerformanceMetricErrorClass::Timeout),
                ("unauthorized", PerformanceMetricErrorClass::Auth),
                ("invalid api key", PerformanceMetricErrorClass::Auth),
                ("authentication", PerformanceMetricErrorClass::Auth),
                ("api key", PerformanceMetricErrorClass::Auth),
                ("rate limit", PerformanceMetricErrorClass::RateLimit),
                ("too many requests", PerformanceMetricErrorClass::RateLimit),
                ("quota", PerformanceMetricErrorClass::RateLimit),
                ("connection", PerformanceMetricErrorClass::Network),
                ("connect", PerformanceMetricErrorClass::Network),
                ("dns", PerformanceMetricErrorClass::Network),
                ("network", PerformanceMetricErrorClass::Network),
                ("refused", PerformanceMetricErrorClass::Network),
                ("unreachable", PerformanceMetricErrorClass::Network),
                ("reset by peer", PerformanceMetricErrorClass::Network),
                ("broken pipe", PerformanceMetricErrorClass::Network),
                ("overloaded", PerformanceMetricErrorClass::Server),
                ("internal server", PerformanceMetricErrorClass::Server),
                ("server error", PerformanceMetricErrorClass::Server),
                ("bad gateway", PerformanceMetricErrorClass::Server),
                ("service unavailable", PerformanceMetricErrorClass::Server),
                ("invalid request", PerformanceMetricErrorClass::Client),
                ("not found", PerformanceMetricErrorClass::Client),
                ("bad request", PerformanceMetricErrorClass::Client),
                ("invalid", PerformanceMetricErrorClass::Client),
            ]
        });
    let lowered = text.to_lowercase();
    for (pattern, class) in PATTERNS.iter() {
        if lowered.contains(pattern) {
            return *class;
        }
    }
    PerformanceMetricErrorClass::Unknown
}

fn normalized_positive_token_count(value: f64) -> Option<f64> {
    // Normalized Usage uses zero both for an explicit zero and for a missing raw
    // field. Without the provider observation hook, only positive values prove
    // field-level availability.
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

/// Converts existing normalized usage without treating its all-zero placeholder
/// as proof that a provider reported usage.
pub fn performance_metric_usage_from_assistant(message: &AssistantMessage) -> PerformanceMetricUsageV1 {
    let usage = &message.usage;
    let has_authoritative_usage = [usage.input, usage.cache_read, usage.output, usage.total_tokens]
        .into_iter()
        .any(|value| value.is_finite() && value > 0.0);
    if !has_authoritative_usage {
        return PerformanceMetricUsageV1::provider_unavailable();
    }

    PerformanceMetricUsageV1 {
        source: "provider".to_string(),
        input_tokens: normalized_positive_token_count(usage.input),
        cached_input_tokens: normalized_positive_token_count(usage.cache_read),
        output_tokens: normalized_positive_token_count(usage.output),
        reasoning_tokens: None,
        total_tokens: normalized_positive_token_count(usage.total_tokens),
        // Future/custom provider normalizers may use a different overlap contract.
        cached_input_included_in_input: None,
        // The normalized Usage type does not currently retain this provider detail.
        reasoning_included_in_output: None,
        estimator: None,
    }
}

/// Converts a raw provider usage observation. Kept here (rather than in
/// agent-loop.ts) as the shared `PerformanceMetricUsageV1` constructor.
pub fn provider_metric_usage(observation: &pi_ai::types::ProviderUsageObservation) -> PerformanceMetricUsageV1 {
    /// `typeof value === "number" && Number.isFinite(value) && value >= 0`.
    /// `None` and `Some(None)` both mean the raw field was not a number.
    fn token(value: Option<Option<f64>>) -> Option<f64> {
        match value {
            Some(Some(value)) if value.is_finite() && value >= 0.0 => Some(value),
            _ => None,
        }
    }
    /// `typeof value === "boolean"`.
    fn flag(value: Option<Option<bool>>) -> Option<bool> {
        match value {
            Some(Some(value)) => Some(value),
            _ => None,
        }
    }
    PerformanceMetricUsageV1 {
        source: "provider".to_string(),
        input_tokens: token(observation.input_tokens),
        cached_input_tokens: token(observation.cached_input_tokens),
        output_tokens: token(observation.output_tokens),
        reasoning_tokens: token(observation.reasoning_tokens),
        total_tokens: token(observation.total_tokens),
        cached_input_included_in_input: flag(observation.cached_input_included_in_input),
        reasoning_included_in_output: flag(observation.reasoning_included_in_output),
        estimator: None,
    }
}

/// JSON projection of a `PerformanceMetricEvent` used by tests and by recorders
/// that append to JSON-lines files.
pub fn performance_metric_event_to_json(event: &PerformanceMetricEvent) -> Value {
    serde_json::to_value(event).unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::{AssistantMessage, Usage};

    fn assistant_usage(input: f64, cache_read: f64, output: f64) -> AssistantMessage {
        let mut message = AssistantMessage::new("openai-responses", "openai", "gpt-test", 1);
        message.usage = Usage {
            input,
            output,
            cache_read,
            cache_write: 0.0,
            total_tokens: input + cache_read + output,
            cost: pi_ai::types::UsageCost::default(),
        };
        message
    }

    #[test]
    fn uses_monotonic_deltas_and_rejects_unavailable_or_backwards_clocks() {
        assert_eq!(elapsed_metric_ms(Some(10.0), Some(12.5)), Some(2.5));
        assert_eq!(elapsed_metric_ms(None, Some(12.0)), None);
        assert_eq!(elapsed_metric_ms(Some(12.0), Some(10.0)), None);
        assert_eq!(elapsed_metric_ms(Some(f64::NAN), Some(10.0)), None);
    }

    #[test]
    fn does_not_manufacture_provider_usage_from_all_zero_placeholders() {
        let usage = performance_metric_usage_from_assistant(&assistant_usage(0.0, 0.0, 0.0));
        assert_eq!(usage.source, "provider");
        assert_eq!(usage.input_tokens, None);
        assert_eq!(usage.cached_input_tokens, None);
        assert_eq!(usage.output_tokens, None);
        assert_eq!(usage.reasoning_tokens, None);
        assert_eq!(usage.total_tokens, None);
        assert_eq!(usage.cached_input_included_in_input, None);
    }

    #[test]
    fn does_not_promote_per_field_normalized_zero_placeholders() {
        let usage = performance_metric_usage_from_assistant(&assistant_usage(100.0, 0.0, 0.0));
        assert_eq!(usage.input_tokens, Some(100.0));
        assert_eq!(usage.cached_input_tokens, None);
        assert_eq!(usage.output_tokens, None);
        assert_eq!(usage.reasoning_tokens, None);
        assert_eq!(usage.total_tokens, Some(100.0));
        assert_eq!(usage.cached_input_included_in_input, None);
        assert_eq!(usage.reasoning_included_in_output, None);
    }

    #[test]
    fn settlement_is_exactly_once() {
        let settlement = AgentLoopLogicalRequestSettlement::new();
        assert!(settlement.settle_once());
        assert!(!settlement.settle_once());
        assert!(settlement.is_settled());
        settlement.observe_provider_attempt_number(2);
        settlement.observe_provider_attempt_number(1);
        assert_eq!(settlement.max_provider_attempt_number(), 2);
    }

    #[test]
    fn error_message_sanitizer_bounds_and_strips_control_characters() {
        // Trailing whitespace is trimmed before sanitization, so only the
        // interior control character is replaced.
        assert_eq!(
            sanitize_performance_metric_error_message(Some("a\u{0}b\n")),
            Some("a?b".to_string())
        );
        assert_eq!(
            sanitize_performance_metric_error_message(Some("a\u{0}b\u{1}c")),
            Some("a?b?c".to_string())
        );
        let long = "x".repeat(600);
        let sanitized = sanitize_performance_metric_error_message(Some(&long)).expect("bounded");
        assert_eq!(sanitized.chars().count(), PERFORMANCE_METRICS_ERROR_MESSAGE_MAX_CHARS);
        assert_eq!(sanitize_performance_metric_error_message(Some("   ")), None);
        assert_eq!(sanitize_performance_metric_error_message(None), None);
    }

    #[test]
    fn stage_tokens_must_be_bounded_lowercase_words() {
        assert_eq!(sanitize_performance_metric_stage(Some("stale_detected")).as_deref(), Some("stale_detected"));
        assert_eq!(sanitize_performance_metric_stage(Some("start")).as_deref(), Some("start"));
        assert_eq!(sanitize_performance_metric_stage(Some("Stale Detected")), None);
        assert_eq!(sanitize_performance_metric_stage(Some("has space")), None);
        assert_eq!(sanitize_performance_metric_stage(Some(&"x".repeat(33))), None);
        assert_eq!(sanitize_performance_metric_stage(Some("")), None);
        assert_eq!(sanitize_performance_metric_stage(None), None);
    }

    #[test]
    fn stream_failure_kinds_map_to_bounded_error_classes() {
        let failure = |kind: &str, status: Option<i64>| {
            PerformanceMetricFailure::from_stream_failure_info(&pi_ai::utils::stream_failure::StreamFailureInfo {
                kind: kind.to_string(),
                provider_error_type: Some("overloaded_error".to_string()),
                status,
                request_id: None,
                retry_after_ms: None,
                raw: None,
            })
        };
        assert_eq!(failure("auth", Some(401)).class, PerformanceMetricErrorClass::Auth);
        assert_eq!(failure("auth", Some(401)).http_status, Some(401));
        assert_eq!(failure("rate_limit", Some(429)).class, PerformanceMetricErrorClass::RateLimit);
        assert_eq!(failure("overloaded", Some(529)).class, PerformanceMetricErrorClass::Server);
        assert_eq!(failure("server_error", Some(500)).class, PerformanceMetricErrorClass::Server);
        assert_eq!(failure("request_interrupted", None).class, PerformanceMetricErrorClass::Cancelled);
        assert_eq!(failure("invalid_request", Some(400)).class, PerformanceMetricErrorClass::Client);
        assert_eq!(failure("unknown", None).class, PerformanceMetricErrorClass::Unknown);
        // A status alone can classify an otherwise unknown kind.
        assert_eq!(failure("weird", Some(408)).class, PerformanceMetricErrorClass::Timeout);
        assert_eq!(failure("weird", Some(503)).class, PerformanceMetricErrorClass::Server);
        // The bounded message keeps the short provider error type, never the raw payload.
        assert_eq!(failure("auth", None).message.as_deref(), Some("overloaded_error"));
        assert_eq!(failure("auth", None).class, PerformanceMetricErrorClass::Auth);
    }

    #[test]
    fn anyhow_errors_classify_through_the_chain_then_heuristics() {
        let stream_failure = pi_ai::utils::stream_failure::StreamFailureError::new(
            "Provider rate limit exceeded",
            pi_ai::utils::stream_failure::StreamFailureInfo {
                kind: "rate_limit".to_string(),
                provider_error_type: None,
                status: Some(429),
                request_id: None,
                retry_after_ms: None,
                raw: None,
            },
        );
        let error = anyhow::anyhow!(stream_failure);
        let classified = PerformanceMetricFailure::from_anyhow_error(&error);
        assert_eq!(classified.class, PerformanceMetricErrorClass::RateLimit);
        assert_eq!(classified.http_status, Some(429));

        let network = PerformanceMetricFailure::classify_message("connection refused by peer");
        assert_eq!(network.class, PerformanceMetricErrorClass::Network);
        let timeout = PerformanceMetricFailure::classify_message("request timed out after 30s");
        assert_eq!(timeout.class, PerformanceMetricErrorClass::Timeout);
        let auth = PerformanceMetricFailure::classify_message("invalid api key provided");
        assert_eq!(auth.class, PerformanceMetricErrorClass::Auth);
        let unknown = PerformanceMetricFailure::classify_message("something odd happened");
        assert_eq!(unknown.class, PerformanceMetricErrorClass::Unknown);
    }

    #[test]
    fn assistant_message_diagnostics_classify_the_terminal_failure() {
        let mut message = AssistantMessage::new("openai-responses", "openai", "m", 1);
        message.stop_reason = pi_ai::types::STOP_REASON_ERROR.to_string();
        message.diagnostics = Some(vec![pi_ai::utils::diagnostics::AssistantMessageDiagnostic {
            type_: "provider_stream_failure".to_string(),
            timestamp: 0,
            error: Some(pi_ai::utils::diagnostics::DiagnosticErrorInfo {
                name: Some("StreamFailureError".to_string()),
                message: "Provider rate limit exceeded (429)".to_string(),
                stack: None,
                code: None,
            }),
            details: Some(serde_json::Map::from_iter([
                ("kind".to_string(), serde_json::json!("rate_limit")),
                ("status".to_string(), serde_json::json!(429)),
            ])),
        }]);
        let classified = PerformanceMetricFailure::from_assistant_message(&message);
        assert_eq!(classified.class, PerformanceMetricErrorClass::RateLimit);
        assert_eq!(classified.http_status, Some(429));
        assert!(classified.message.as_deref().is_some_and(|text| text.contains("rate limit")));

        let mut raw_only = AssistantMessage::new("openai-responses", "openai", "m", 1);
        raw_only.stop_reason = pi_ai::types::STOP_REASON_ERROR.to_string();
        raw_only.stop_reason_raw = Some("SAFETY".to_string());
        let classified = PerformanceMetricFailure::from_assistant_message(&raw_only);
        assert_eq!(classified.class, PerformanceMetricErrorClass::Client);

        let mut text_only = AssistantMessage::new("openai-responses", "openai", "m", 1);
        text_only.stop_reason = pi_ai::types::STOP_REASON_ERROR.to_string();
        text_only.error_message = Some("connection error while streaming".to_string());
        let classified = PerformanceMetricFailure::from_assistant_message(&text_only);
        assert_eq!(classified.class, PerformanceMetricErrorClass::Network);
    }

    #[test]
    fn cancelled_failures_carry_no_status_or_text() {
        let cancelled = PerformanceMetricFailure::cancelled();
        assert_eq!(cancelled.class, PerformanceMetricErrorClass::Cancelled);
        assert_eq!(cancelled.http_status, None);
        assert_eq!(cancelled.message, None);
    }

    #[test]
    fn measurements_keep_explicit_null_distinct_from_absent() {
        let mut measurements = PerformanceMetricMeasurements::new();
        measurements.insert(PerformanceMetricMeasurement::TotalMs, Some(5.0));
        measurements.insert(PerformanceMetricMeasurement::AttemptCount, None);
        let json = serde_json::to_string(&measurements).unwrap();
        // TypeScript builds the object literal in this insertion order.
        assert_eq!(json, r#"{"total_ms":5.0,"attempt_count":null}"#);
    }
}
