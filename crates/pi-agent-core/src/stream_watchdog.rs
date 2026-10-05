//! Stream stall watchdog for the agent loop's provider pump.
//!
//! Production incident this guards against: a supervisor session on a
//! 223,489-token context let the model emit the full 131,072-token output cap
//! and then the local pipeline stalled for 43.4 minutes
//! (`local_drain_ms = 2,604,141`) with no timeout, no error, and no recovery
//! until the user killed the session. Two bounds close that gap:
//!
//! - an event-gap timeout (`PRIME_AGENT_STREAM_EVENT_GAP_MS`): abort when no
//!   provider event arrives within the window, resetting on every event;
//! - an overall stream deadline (`PRIME_AGENT_STREAM_DEADLINE_MS`): abort the
//!   whole stream phase of one attempt after a wall-clock cap, even when events
//!   keep trickling.
//!
//! Both aborts surface as a retryable `provider_stream_failure` message
//! (kind `stream_stall`) so the existing provider-retry policy re-issues the
//! request instead of hanging or ending the turn as terminal.

use std::future::Future;
use std::time::Duration;

use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, Model, Usage, STOP_REASON_ERROR,
    STOP_REASON_LENGTH,
};
use pi_ai::utils::diagnostics::{
    append_assistant_message_diagnostic, now_millis, AssistantMessageDiagnostic,
    DiagnosticErrorInfo,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::stream_failure::{KIND_STREAM_STALL, KIND_TRUNCATED_RESPONSE};
use serde_json::{Map, Value};

use crate::types::AgentLoopConfig;

/// `PRIME_AGENT_STREAM_EVENT_GAP_MS`: abort after this long without an event.
pub const STREAM_EVENT_GAP_MS_ENV: &str = "PRIME_AGENT_STREAM_EVENT_GAP_MS";
/// `PRIME_AGENT_STREAM_DEADLINE_MS`: abort the stream phase after this long.
pub const STREAM_DEADLINE_MS_ENV: &str = "PRIME_AGENT_STREAM_DEADLINE_MS";
/// `PRIME_AGENT_RETRY_TRUNCATED_RESPONSE`: retry length-capped completions.
pub const RETRY_TRUNCATED_RESPONSE_ENV: &str = "PRIME_AGENT_RETRY_TRUNCATED_RESPONSE";

pub const DEFAULT_STREAM_EVENT_GAP_MS: u64 = 120_000;
pub const MIN_STREAM_EVENT_GAP_MS: u64 = 5_000;
pub const MAX_STREAM_EVENT_GAP_MS: u64 = 600_000;
pub const DEFAULT_STREAM_DEADLINE_MS: u64 = 900_000;
pub const MIN_STREAM_DEADLINE_MS: u64 = 60_000;
pub const MAX_STREAM_DEADLINE_MS: u64 = 7_200_000;

const LOG_COMPONENT: &str = "agent-core.stream-watchdog";

/// Which watchdog limit expired for one attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamStallLimit {
    /// No provider event arrived within `PRIME_AGENT_STREAM_EVENT_GAP_MS`.
    EventGap,
    /// The whole stream phase exceeded `PRIME_AGENT_STREAM_DEADLINE_MS`.
    StreamDeadline,
}

impl StreamStallLimit {
    fn label(self) -> &'static str {
        match self {
            StreamStallLimit::EventGap => "event_gap",
            StreamStallLimit::StreamDeadline => "stream_deadline",
        }
    }

    pub(crate) fn configured(self, watchdog: &StreamWatchdogConfig) -> Duration {
        match self {
            StreamStallLimit::EventGap => watchdog
                .event_gap
                .expect("the event gap is enabled when the gap limit fires"),
            StreamStallLimit::StreamDeadline => watchdog
                .stream_deadline
                .expect("the stream deadline is enabled when the deadline limit fires"),
        }
    }

    /// Elapsed time that the expired limit actually measured.
    pub(crate) fn elapsed_since(
        self,
        attempt_started: tokio::time::Instant,
        last_event_at: tokio::time::Instant,
    ) -> Duration {
        match self {
            StreamStallLimit::EventGap => last_event_at.elapsed(),
            StreamStallLimit::StreamDeadline => attempt_started.elapsed(),
        }
    }

    fn error_message(self, elapsed: Duration, configured: Duration) -> String {
        match self {
            StreamStallLimit::EventGap => format!(
                "Provider stream stalled: no events for {}ms (event gap limit {}ms)",
                elapsed.as_millis(),
                configured.as_millis()
            ),
            StreamStallLimit::StreamDeadline => format!(
                "Provider stream exceeded the {}ms overall deadline after {}ms",
                configured.as_millis(),
                elapsed.as_millis()
            ),
        }
    }
}

/// The pump loop's next-event outcome under the watchdog.
#[derive(Debug)]
pub(crate) enum WatchdogNext {
    /// An event arrived; `None` means the stream ended.
    Event(Option<AssistantMessageEvent>),
    /// The cancellation signal fired before any event.
    Cancelled,
    /// A watchdog limit expired.
    Stall(StreamStallLimit),
}

/// Await the next provider event, bounded by the event-gap timeout.
///
/// `EventStream::next` is cancel-safe (events stay queued when the future is
/// dropped), so the timeout can wrap it without losing events.
pub(crate) async fn next_with_event_gap(
    response: &AssistantMessageEventStream,
    event_gap: Option<Duration>,
) -> WatchdogNext {
    match event_gap {
        Some(event_gap) => match tokio::time::timeout(event_gap, response.next()).await {
            Ok(next) => WatchdogNext::Event(next),
            Err(_elapsed) => WatchdogNext::Stall(StreamStallLimit::EventGap),
        },
        None => WatchdogNext::Event(response.next().await),
    }
}

/// A future that resolves when the overall stream deadline elapses, and never
/// resolves when the deadline is disabled.
pub(crate) fn deadline_timer(deadline_at: Option<tokio::time::Instant>) -> impl Future<Output = ()> {
    async move {
        match deadline_at {
            Some(deadline_at) => tokio::time::sleep_until(deadline_at).await,
            None => std::future::pending::<()>().await,
        }
    }
}

/// Watchdog limits for one provider attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamWatchdogConfig {
    /// Abort when no event arrives within this window; `None` disables.
    pub event_gap: Option<Duration>,
    /// Abort the whole stream phase after this long; `None` disables.
    pub stream_deadline: Option<Duration>,
}

impl StreamWatchdogConfig {
    pub fn from_env() -> Self {
        Self::from_env_values(
            std::env::var(STREAM_EVENT_GAP_MS_ENV).ok().as_deref(),
            std::env::var(STREAM_DEADLINE_MS_ENV).ok().as_deref(),
        )
    }

    /// Parsing rules shared by both knobs: missing or unparseable values fall
    /// back to the default (never to disabled), `0` disables, and any other
    /// value is clamped into `[min_ms, max_ms]`.
    pub fn from_env_values(event_gap: Option<&str>, stream_deadline: Option<&str>) -> Self {
        Self {
            event_gap: parse_watchdog_ms(
                event_gap,
                DEFAULT_STREAM_EVENT_GAP_MS,
                MIN_STREAM_EVENT_GAP_MS,
                MAX_STREAM_EVENT_GAP_MS,
            ),
            stream_deadline: parse_watchdog_ms(
                stream_deadline,
                DEFAULT_STREAM_DEADLINE_MS,
                MIN_STREAM_DEADLINE_MS,
                MAX_STREAM_DEADLINE_MS,
            ),
        }
    }
}

fn parse_watchdog_ms(
    raw: Option<&str>,
    default_ms: u64,
    min_ms: u64,
    max_ms: u64,
) -> Option<Duration> {
    let Some(raw) = raw else {
        return Some(Duration::from_millis(default_ms));
    };
    let Ok(parsed) = raw.trim().parse::<u64>() else {
        return Some(Duration::from_millis(default_ms));
    };
    if parsed == 0 {
        return None;
    }
    Some(Duration::from_millis(parsed.clamp(min_ms, max_ms)))
}

/// One structured warning line for a watchdog abort.
pub(crate) fn log_stream_stall(
    model: &Model,
    limit: StreamStallLimit,
    elapsed: Duration,
    configured: Duration,
    attempt: u64,
    partial_message: Option<&AssistantMessage>,
) {
    let mut fields = Map::new();
    fields.insert("provider".to_string(), Value::String(model.provider.clone()));
    fields.insert("model".to_string(), Value::String(model.id.clone()));
    fields.insert("api".to_string(), Value::String(model.api.clone()));
    fields.insert("kind".to_string(), Value::String(KIND_STREAM_STALL.to_string()));
    fields.insert("limit".to_string(), Value::String(limit.label().to_string()));
    fields.insert(
        "elapsedMs".to_string(),
        Value::Number((elapsed.as_millis() as i64).into()),
    );
    fields.insert(
        "configuredMs".to_string(),
        Value::Number((configured.as_millis() as i64).into()),
    );
    fields.insert("attempt".to_string(), Value::Number((attempt as i64).into()));
    if let Some(partial) = partial_message {
        fields.insert(
            "partialContentBlocks".to_string(),
            Value::Number((partial.content.len() as i64).into()),
        );
    }
    pi_ai::log::get_logger(LOG_COMPONENT)
        .warn("provider stream stalled; aborting the attempt", Some(fields));
}

/// Build the retryable stall failure that ends the attempt.
///
/// The message deliberately carries no content: no tool call from a stalled
/// partial was ever executed, so re-issuing the request cannot replay work,
/// and an empty content vector keeps the provider-retry replay guard open.
pub(crate) fn create_stream_stall_message(
    config: &AgentLoopConfig,
    partial_message: Option<&AssistantMessage>,
    limit: StreamStallLimit,
    elapsed: Duration,
    configured: Duration,
    attempt: u64,
) -> AssistantMessage {
    let mut message = AssistantMessage::new(
        partial_message
            .map(|partial| partial.api.clone())
            .unwrap_or_else(|| config.model.api.clone()),
        partial_message
            .map(|partial| partial.provider.clone())
            .unwrap_or_else(|| config.model.provider.clone()),
        partial_message
            .map(|partial| partial.model.clone())
            .unwrap_or_else(|| config.model.id.clone()),
        now_millis(),
    );
    message.usage = partial_message
        .map(|partial| partial.usage.clone())
        .unwrap_or_else(Usage::zero);
    message.stop_reason = STOP_REASON_ERROR.to_string();
    message.response_id = partial_message.and_then(|partial| partial.response_id.clone());
    let error_message = limit.error_message(elapsed, configured);
    message.error_message = Some(error_message.clone());

    let mut details = Map::new();
    details.insert("kind".to_string(), Value::String(KIND_STREAM_STALL.to_string()));
    details.insert("limit".to_string(), Value::String(limit.label().to_string()));
    details.insert(
        "elapsedMs".to_string(),
        Value::Number((elapsed.as_millis() as i64).into()),
    );
    details.insert(
        "configuredMs".to_string(),
        Value::Number((configured.as_millis() as i64).into()),
    );
    details.insert("attempt".to_string(), Value::Number((attempt as i64).into()));
    if let Some(partial) = partial_message {
        details.insert(
            "partialContentBlocks".to_string(),
            Value::Number((partial.content.len() as i64).into()),
        );
    }
    append_assistant_message_diagnostic(
        &mut message,
        AssistantMessageDiagnostic {
            type_: "provider_stream_failure".to_string(),
            timestamp: now_millis(),
            error: Some(DiagnosticErrorInfo {
                name: Some("StreamStallError".to_string()),
                message: error_message,
                stack: None,
                code: None,
            }),
            details: Some(details),
        },
    );
    message
}

/// `PRIME_AGENT_RETRY_TRUNCATED_RESPONSE` is enabled by `1/true/yes/on`.
pub fn retry_truncated_responses_enabled() -> bool {
    retry_truncated_responses_from_env_value(
        std::env::var(RETRY_TRUNCATED_RESPONSE_ENV).ok().as_deref(),
    )
}

/// The value-parsing form of [`retry_truncated_responses_enabled`], separated
/// so tests can exercise parsing without mutating process-global state.
pub fn retry_truncated_responses_from_env_value(raw: Option<&str>) -> bool {
    match raw {
        Some(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

/// Surface a length-capped completion: log one structured warning, and when
/// `PRIME_AGENT_RETRY_TRUNCATED_RESPONSE` is enabled, convert the truncated
/// message into a retryable failure instead of accepting it.
pub(crate) fn surface_truncated_response(
    config: &AgentLoopConfig,
    message: AssistantMessage,
    attempt: u64,
) -> AssistantMessage {
    if message.stop_reason != STOP_REASON_LENGTH {
        return message;
    }
    log_truncated_response(&config.model, &message, attempt);
    if !retry_truncated_responses_enabled() {
        return message;
    }
    create_truncated_response_failure(config, &message, attempt)
}

fn usage_fields(usage: &Usage) -> Vec<(&'static str, f64)> {
    [
        ("inputTokens", usage.input),
        ("outputTokens", usage.output),
        ("totalTokens", usage.total_tokens),
    ]
    .into_iter()
    .filter(|(_, value)| value.is_finite() && *value > 0.0)
    .collect()
}

fn log_truncated_response(model: &Model, message: &AssistantMessage, attempt: u64) {
    let mut fields = Map::new();
    fields.insert("provider".to_string(), Value::String(model.provider.clone()));
    fields.insert("model".to_string(), Value::String(model.id.clone()));
    fields.insert("api".to_string(), Value::String(model.api.clone()));
    fields.insert(
        "stopReason".to_string(),
        Value::String(STOP_REASON_LENGTH.to_string()),
    );
    for (name, value) in usage_fields(&message.usage) {
        if let Some(number) = serde_json::Number::from_f64(value) {
            fields.insert(name.to_string(), Value::Number(number));
        }
    }
    fields.insert("attempt".to_string(), Value::Number((attempt as i64).into()));
    fields.insert(
        "retryEnabled".to_string(),
        Value::Bool(retry_truncated_responses_enabled()),
    );
    pi_ai::log::get_logger(LOG_COMPONENT).warn(
        "assistant response ended at the provider output cap",
        Some(fields),
    );
}

/// The opt-in retryable failure for a length-capped completion.
///
/// Like the stall failure, the content is dropped: the truncated text was
/// never executed, and an empty content vector keeps the replay guard open so
/// the host retry path can re-issue the request.
fn create_truncated_response_failure(
    config: &AgentLoopConfig,
    truncated: &AssistantMessage,
    attempt: u64,
) -> AssistantMessage {
    let mut message = AssistantMessage::new(
        if truncated.api.is_empty() {
            config.model.api.clone()
        } else {
            truncated.api.clone()
        },
        if truncated.provider.is_empty() {
            config.model.provider.clone()
        } else {
            truncated.provider.clone()
        },
        if truncated.model.is_empty() {
            config.model.id.clone()
        } else {
            truncated.model.clone()
        },
        now_millis(),
    );
    message.usage = truncated.usage.clone();
    message.response_model = truncated.response_model.clone();
    message.response_id = truncated.response_id.clone();
    message.stop_reason = STOP_REASON_ERROR.to_string();
    let error_message = format!(
        "Provider response ended at the output length cap (stop_reason \"length\"); retrying because {} is enabled",
        RETRY_TRUNCATED_RESPONSE_ENV
    );
    message.error_message = Some(error_message.clone());

    let mut details = Map::new();
    details.insert(
        "kind".to_string(),
        Value::String(KIND_TRUNCATED_RESPONSE.to_string()),
    );
    for (name, value) in usage_fields(&truncated.usage) {
        if let Some(number) = serde_json::Number::from_f64(value) {
            details.insert(name.to_string(), Value::Number(number));
        }
    }
    details.insert("attempt".to_string(), Value::Number((attempt as i64).into()));
    append_assistant_message_diagnostic(
        &mut message,
        AssistantMessageDiagnostic {
            type_: "provider_stream_failure".to_string(),
            timestamp: now_millis(),
            error: Some(DiagnosticErrorInfo {
                name: Some("TruncatedResponseError".to_string()),
                message: error_message,
                stack: None,
                code: None,
            }),
            details: Some(details),
        },
    );
    message
}

/// Shared fixtures for tests that drive the watchdog through process-global
/// environment variables.
#[cfg(test)]
pub(crate) mod testing {
    /// Environment mutation is process-global; every test that mutates the
    /// watchdog variables or runs the agent loop against them takes this lock.
    pub static ENV_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Sets `name` for the guard's lifetime, restoring the previous value on drop.
    pub struct EnvVarGuard {
        name: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        pub fn set(name: &'static str, value: String) -> Self {
            let previous = std::env::var_os(name);
            std::env::set_var(name, value);
            EnvVarGuard { name, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => std::env::set_var(self.name, previous),
                None => std::env::remove_var(self.name),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{EnvVarGuard, ENV_TESTS};
    use super::*;

    #[test]
    fn watchdog_defaults_apply_when_the_environment_is_unset() {
        let watchdog = StreamWatchdogConfig::from_env_values(None, None);
        assert_eq!(watchdog.event_gap, Some(Duration::from_millis(120_000)));
        assert_eq!(watchdog.stream_deadline, Some(Duration::from_millis(900_000)));
    }

    #[test]
    fn watchdog_values_clamp_into_their_bounds() {
        let watchdog = StreamWatchdogConfig::from_env_values(Some("6000"), Some("60001"));
        assert_eq!(watchdog.event_gap, Some(Duration::from_millis(6_000)));
        assert_eq!(watchdog.stream_deadline, Some(Duration::from_millis(60_001)));

        let low = StreamWatchdogConfig::from_env_values(Some("1"), Some("1"));
        assert_eq!(low.event_gap, Some(Duration::from_millis(5_000)));
        assert_eq!(low.stream_deadline, Some(Duration::from_millis(60_000)));

        let high = StreamWatchdogConfig::from_env_values(Some("999999999"), Some("999999999"));
        assert_eq!(high.event_gap, Some(Duration::from_millis(600_000)));
        assert_eq!(high.stream_deadline, Some(Duration::from_millis(7_200_000)));
    }

    #[test]
    fn watchdog_zero_disables_and_unparseable_values_fall_back_to_defaults() {
        let disabled = StreamWatchdogConfig::from_env_values(Some("0"), Some("0"));
        assert_eq!(disabled.event_gap, None);
        assert_eq!(disabled.stream_deadline, None);

        for raw in ["abc", "-1", "1.5", "", "  "] {
            let fallback = StreamWatchdogConfig::from_env_values(Some(raw), Some(raw));
            assert_eq!(fallback.event_gap, Some(Duration::from_millis(120_000)), "{raw:?}");
            assert_eq!(
                fallback.stream_deadline,
                Some(Duration::from_millis(900_000)),
                "{raw:?}"
            );
        }
    }

    #[test]
    fn watchdog_environment_round_trip() {
        let _guard = ENV_TESTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let _gap = EnvVarGuard::set(STREAM_EVENT_GAP_MS_ENV, "7000".to_string());
        let _deadline = EnvVarGuard::set(STREAM_DEADLINE_MS_ENV, "0".to_string());
        let watchdog = StreamWatchdogConfig::from_env();
        assert_eq!(watchdog.event_gap, Some(Duration::from_millis(7_000)));
        assert_eq!(watchdog.stream_deadline, None);
    }

    #[test]
    fn truncated_response_flag_matches_only_enabled_values() {
        for raw in ["1", "true", "TRUE", "yes", "Yes", "on", " ON "] {
            assert!(retry_truncated_responses_from_env_value(Some(raw)), "{raw:?}");
        }
        for raw in ["0", "false", "no", "off", "maybe", "", "  "] {
            assert!(!retry_truncated_responses_from_env_value(Some(raw)), "{raw:?}");
        }
        assert!(!retry_truncated_responses_from_env_value(None));
    }

    #[test]
    fn truncated_response_flag_defaults_off_from_the_environment() {
        let _guard = ENV_TESTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let _flag = EnvVarGuard::set(RETRY_TRUNCATED_RESPONSE_ENV, "garbage".to_string());
        assert!(!retry_truncated_responses_enabled());
    }
}
