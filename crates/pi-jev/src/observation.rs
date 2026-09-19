//! Bounded metadata-only observations. These values cannot control the agent.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

pub const MAX_TRACE_EVENTS: usize = 32;
pub const MAX_ROUTING_CANDIDATES: usize = 8;
pub const MAX_MODEL_ID_CHARS: usize = 120;

/// Classification, not permission to retry. The host keeps all retry authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryFailureKind {
    Transient,
    BadArguments,
    Permission,
    RateLimited,
    ProviderFailure,
    ToolFailure,
    Fatal,
    Unknown,
}

impl RetryFailureKind {
    pub const ALL: [Self; 8] = [
        Self::Transient,
        Self::BadArguments,
        Self::Permission,
        Self::RateLimited,
        Self::ProviderFailure,
        Self::ToolFailure,
        Self::Fatal,
        Self::Unknown,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transient => "transient",
            Self::BadArguments => "bad_arguments",
            Self::Permission => "permission",
            Self::RateLimited => "rate_limited",
            Self::ProviderFailure => "provider_failure",
            Self::ToolFailure => "tool_failure",
            Self::Fatal => "fatal",
            Self::Unknown => "unknown",
        }
    }

    /// Only known structured diagnostic kinds are normalized. Never inspect error text.
    pub fn from_provider_kind(kind: &str) -> Self {
        match kind {
            "overloaded" | "network_error" | "timeout" => Self::Transient,
            "invalid_request" => Self::BadArguments,
            "auth" | "permission" => Self::Permission,
            "rate_limit" => Self::RateLimited,
            "server_error" | "malformed_response" | "request_interrupted" => Self::ProviderFailure,
            "refusal" | "safety" | "agent_lifecycle_failure" => Self::Fatal,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultAssessment {
    Complete,
    Partial,
    Failed,
    Uncertain,
}

impl ResultAssessment {
    pub const ALL: [Self; 4] = [Self::Complete, Self::Partial, Self::Failed, Self::Uncertain];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial => "partial",
            Self::Failed => "failed",
            Self::Uncertain => "uncertain",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceAssessment {
    Good,
    Review,
    RetryRecommended,
    Escalate,
    Suspicious,
}

impl TraceAssessment {
    pub const ALL: [Self; 5] = [
        Self::Good,
        Self::Review,
        Self::RetryRecommended,
        Self::Escalate,
        Self::Suspicious,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Good => "good",
            Self::Review => "review",
            Self::RetryRecommended => "retry_recommended",
            Self::Escalate => "escalate",
            Self::Suspicious => "suspicious",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservedStopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Unknown,
}

impl ObservedStopReason {
    pub fn from_stop_reason(reason: &str) -> Self {
        match reason {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "toolUse" => Self::ToolUse,
            "error" => Self::Error,
            "aborted" => Self::Aborted,
            _ => Self::Unknown,
        }
    }
}

/// Set only by an explicit verification source, never by successful tool execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationEvidence {
    Passed,
    Failed,
    NotRun,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum TraceEvent {
    TurnStarted,
    AssistantEnded {
        stop_reason: ObservedStopReason,
        failure_kind: Option<RetryFailureKind>,
    },
    ToolEnded {
        is_error: bool,
    },
    RetryObserved {
        attempt: u32,
    },
    VerificationObserved {
        outcome: VerificationEvidence,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TraceSummary {
    pub turns: u64,
    pub assistant_messages: u64,
    pub tool_results: u64,
    pub tool_errors: u64,
    pub assistant_errors: u64,
    pub retries_observed: u64,
    pub last_retry_attempt: Option<u32>,
    pub last_stop_reason: Option<ObservedStopReason>,
    pub failure_kind: Option<RetryFailureKind>,
    pub verification: VerificationEvidence,
    pub events_seen: u64,
    pub events_dropped: u64,
    pub recent_events: Vec<TraceEvent>,
}

impl TraceSummary {
    pub fn has_evidence(&self) -> bool {
        self.events_seen > 0
    }

    /// Missing checks are unknown, not passed. Counts describe observations, not quality.
    pub fn evidence_description(&self) -> String {
        format!(
            "turns={}, tool_results={}, tool_errors={}, assistant_errors={}, retries_observed={}, stop={:?}, failure={:?}, verification={:?}; missing evidence is unknown",
            self.turns, self.tool_results, self.tool_errors, self.assistant_errors,
            self.retries_observed, self.last_stop_reason, self.failure_kind, self.verification,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct TraceObserver {
    summary: TraceSummary,
    recent: VecDeque<TraceEvent>,
}

impl TraceObserver {
    pub fn record(&mut self, event: TraceEvent) {
        self.summary.events_seen = self.summary.events_seen.saturating_add(1);
        match event {
            TraceEvent::TurnStarted => {
                self.summary.turns = self.summary.turns.saturating_add(1);
                self.summary.failure_kind = None;
                self.summary.last_stop_reason = None;
                self.summary.verification = VerificationEvidence::Unknown;
            }
            TraceEvent::AssistantEnded {
                stop_reason,
                failure_kind,
            } => {
                self.summary.assistant_messages = self.summary.assistant_messages.saturating_add(1);
                self.summary.last_stop_reason = Some(stop_reason);
                if stop_reason == ObservedStopReason::Error {
                    self.summary.assistant_errors =
                        self.summary.assistant_errors.saturating_add(1);
                    self.summary.failure_kind =
                        Some(failure_kind.unwrap_or(RetryFailureKind::Unknown));
                }
            }
            TraceEvent::ToolEnded { is_error } => {
                self.summary.tool_results = self.summary.tool_results.saturating_add(1);
                if is_error {
                    self.summary.tool_errors = self.summary.tool_errors.saturating_add(1);
                    self.summary.failure_kind = Some(RetryFailureKind::ToolFailure);
                }
            }
            TraceEvent::RetryObserved { attempt } => {
                self.summary.retries_observed = self.summary.retries_observed.saturating_add(1);
                self.summary.last_retry_attempt = Some(attempt);
            }
            TraceEvent::VerificationObserved { outcome } => self.summary.verification = outcome,
        }
        if self.recent.len() == MAX_TRACE_EVENTS {
            self.recent.pop_front();
            self.summary.events_dropped = self.summary.events_dropped.saturating_add(1);
        }
        self.recent.push_back(event);
    }

    pub fn summary(&self) -> TraceSummary {
        let mut summary = self.summary.clone();
        summary.recent_events = self.recent.iter().copied().collect();
        summary
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Actual local measurements for one allowlisted model, never estimates or recommendations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingMetrics {
    pub model: String,
    pub attempts: u64,
    pub failures: u64,
    pub latency_ms: Option<f64>,
    pub cost: Option<f64>,
}

impl RoutingMetrics {
    pub fn is_valid(&self) -> bool {
        valid_model_id(&self.model)
            && self.attempts > 0
            && self.failures <= self.attempts
            && self
                .latency_ms
                .is_none_or(|value| value.is_finite() && value >= 0.0)
            && self
                .cost
                .is_none_or(|value| value.is_finite() && value >= 0.0)
    }

    pub fn observed_success_rate(&self) -> Option<f64> {
        self.is_valid()
            .then(|| (self.attempts - self.failures) as f64 / self.attempts as f64)
    }
}

pub fn valid_model_id(model: &str) -> bool {
    !model.is_empty()
        && model.chars().count() <= MAX_MODEL_ID_CHARS
        && model
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || "-_.:/@".contains(ch))
}
