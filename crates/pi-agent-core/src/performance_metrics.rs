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
    UiRender,
    UiMenuOpen,
    UiSessionOpen,
    Recorder,
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
            PerformanceMetricOperation::UiRender => "ui_render",
            PerformanceMetricOperation::UiMenuOpen => "ui_menu_open",
            PerformanceMetricOperation::UiSessionOpen => "ui_session_open",
            PerformanceMetricOperation::Recorder => "recorder",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceMetricOutcome {
    /// An in-flight start row. A paired terminal row with the same correlation
    /// IDs follows; only terminal outcomes (success/failure/cancelled/
    /// unavailable) count as completed attempts.
    Started,
    Success,
    Failure,
    Cancelled,
    Unavailable,
}

impl PerformanceMetricOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            PerformanceMetricOutcome::Started => "started",
            PerformanceMetricOutcome::Success => "success",
            PerformanceMetricOutcome::Failure => "failure",
            PerformanceMetricOutcome::Cancelled => "cancelled",
            PerformanceMetricOutcome::Unavailable => "unavailable",
        }
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
    MaxMs,
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
            PerformanceMetricMeasurement::MaxMs => "max_ms",
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
    fn measurements_keep_explicit_null_distinct_from_absent() {
        let mut measurements = PerformanceMetricMeasurements::new();
        measurements.insert(PerformanceMetricMeasurement::TotalMs, Some(5.0));
        measurements.insert(PerformanceMetricMeasurement::AttemptCount, None);
        let json = serde_json::to_string(&measurements).unwrap();
        // TypeScript builds the object literal in this insertion order.
        assert_eq!(json, r#"{"total_ms":5.0,"attempt_count":null}"#);
    }
}
