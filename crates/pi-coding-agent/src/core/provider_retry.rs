//! Port of packages/coding-agent/src/core/provider-retry.ts

use std::sync::Arc;

use pi_ai::types::AssistantMessage;
use tokio_util::sync::CancellationToken;

use crate::core::settings_manager::SettingsManager;
use crate::utils::sleep::sleep;

/// `interface ProviderRetryPolicy`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderRetryPolicy {
    pub enabled: bool,
    pub max_retries: f64,
    pub base_delay_ms: f64,
    /// Max server-requested retry delay before giving up; 0 disables the cap.
    pub max_retry_delay_ms: f64,
}

/// `providerRetryPolicy(settingsManager)`.
pub fn provider_retry_policy(settings_manager: &SettingsManager) -> ProviderRetryPolicy {
    let retry = settings_manager.get_retry_settings();
    ProviderRetryPolicy {
        enabled: retry.enabled,
        max_retries: retry.max_retries,
        base_delay_ms: retry.base_delay_ms,
        max_retry_delay_ms: settings_manager.get_provider_retry_settings().max_retry_delay_ms,
    }
}

/// `isAgentLifecycleFailure(message)`.
///
/// Local listener/lifecycle crashes are not provider failures; never retry them.
pub fn is_agent_lifecycle_failure(message: &AssistantMessage) -> bool {
    message
        .diagnostics
        .as_ref()
        .map(|diagnostics| {
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.type_ == "agent_lifecycle_failure")
        })
        .unwrap_or(false)
}

/// `isFauxProviderQueueExhausted(message)`.
///
/// The faux test provider's queue running dry is deterministic; retrying it only
/// stalls tests.
pub fn is_faux_provider_queue_exhausted(message: &AssistantMessage) -> bool {
    message.provider == "faux" && message.error_message.as_deref() == Some("No more faux responses queued")
}

/// `providerStreamFailureDetails(message)`.
pub fn provider_stream_failure_details(message: &AssistantMessage) -> Option<serde_json::Map<String, serde_json::Value>> {
    let failure = message
        .diagnostics
        .as_ref()?
        .iter()
        .find(|diagnostic| diagnostic.type_ == "provider_stream_failure")?;
    let details = failure.details.as_ref()?;
    Some(details.clone())
}

/// `providerStreamFailureKind(message)`.
pub fn provider_stream_failure_kind(message: &AssistantMessage) -> Option<String> {
    let kind = provider_stream_failure_details(message)?.get("kind").cloned()?;
    kind.as_str().map(|kind| kind.to_string())
}

/// Reissuing an uncertain request or already-started response could replay work.
/// Empty block placeholders alone are not evidence that output was produced.
pub fn cannot_replay_provider_failure(message: &AssistantMessage) -> bool {
    message.stop_reason == pi_ai::types::STOP_REASON_ERROR
        && (provider_stream_failure_kind(message).as_deref() == Some("request_interrupted")
            || message.content.iter().any(|block| match block {
                pi_ai::types::ContentBlock::Text(text) => !text.text.is_empty(),
                pi_ai::types::ContentBlock::Thinking(thinking) => !thinking.thinking.is_empty(),
                pi_ai::types::ContentBlock::ToolCall(_) => true,
            }))
}

/// `providerStreamFailureRetryAfterMs(message)`.
pub fn provider_stream_failure_retry_after_ms(message: &AssistantMessage) -> Option<f64> {
    let value = provider_stream_failure_details(message)?.get("retryAfterMs").cloned()?;
    let value = value.as_f64()?;
    if value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

/// `isPermanentProviderFailureKind(kind, retriesPerformed)`.
///
/// Deterministic rejections never retry; auth gets one retry before it can be
/// marked stale.
pub fn is_permanent_provider_failure_kind(kind: Option<&str>, retries_performed: f64) -> bool {
    if matches!(kind, Some("invalid_request") | Some("refusal") | Some("safety") | Some("permission") | Some("request_interrupted")) {
        return true;
    }
    retries_performed > 0.0 && kind == Some("auth")
}

/// `type ProviderRetryDelay`.
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderRetryDelay {
    Wait { delay_ms: f64 },
    ExceedsCap { retry_after_ms: f64 },
}

/// `interface ProviderRetryDelayOptions`.
#[derive(Clone)]
pub struct ProviderRetryDelayOptions {
    /// Uniform source in [0, 1]. Injected by deterministic tests.
    pub random: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
    /// Fractional client-backoff jitter.
    pub jitter_ratio: Option<f64>,
}

impl Default for ProviderRetryDelayOptions {
    fn default() -> Self {
        Self {
            random: None,
            jitter_ratio: None,
        }
    }
}

/// `interface ProviderRetryExecutionOptions`.
#[derive(Clone, Default)]
pub struct ProviderRetryExecutionOptions {
    pub random: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
    pub jitter_ratio: Option<f64>,
    pub policy: Option<ProviderRetryPolicy>,
    /// `signal?: AbortSignal`.
    pub signal: Option<CancellationToken>,
    /// Absolute deadline in the same millisecond timebase returned by `now()`.
    pub deadline_at_ms: Option<f64>,
    pub now: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
    pub sleep: Option<Arc<dyn Fn(f64, Option<CancellationToken>) -> pi_ai::types::BoxFuture<()> + Send + Sync>>,
}

/// Node caps timers at 2^31-1 ms; longer delays overflow setTimeout and fire
/// after ~1ms.
const MAX_TIMER_DELAY_MS: f64 = 2_147_483_647.0;
const DEFAULT_PROVIDER_RETRY_BASE_DELAY_MS: f64 = 2_000.0;
pub const PROVIDER_RETRY_JITTER_RATIO: f64 = 0.2;

/// `boundedRandom(source)`.
fn bounded_random(source: &dyn Fn() -> f64) -> f64 {
    let value = source();
    if !value.is_finite() {
        return 0.5;
    }
    value.clamp(0.0, 1.0)
}

/// `providerRetryDelay(attempt, retryAfterMs, policy, options)`.
///
/// Delay before retry `attempt` (1-based), honoring a server-requested
/// not-before time.
pub fn provider_retry_delay(
    attempt: f64,
    retry_after_ms: Option<f64>,
    policy: &ProviderRetryPolicy,
    options: &ProviderRetryDelayOptions,
) -> ProviderRetryDelay {
    if let Some(retry_after_ms) = retry_after_ms {
        if policy.max_retry_delay_ms > 0.0 && retry_after_ms > policy.max_retry_delay_ms {
            return ProviderRetryDelay::ExceedsCap { retry_after_ms };
        }
    }
    // A timer longer than Node's supported range would fire immediately. Failing
    // preserves Retry-After's not-before contract instead of replaying early.
    if let Some(retry_after_ms) = retry_after_ms {
        if retry_after_ms > MAX_TIMER_DELAY_MS {
            return ProviderRetryDelay::ExceedsCap { retry_after_ms };
        }
    }
    let random_source = options
        .random
        .clone()
        .unwrap_or_else(|| Arc::new(|| 0.5));
    let random = bounded_random(random_source.as_ref());
    let requested_jitter_ratio = options.jitter_ratio.unwrap_or(PROVIDER_RETRY_JITTER_RATIO);
    let jitter_ratio = if requested_jitter_ratio.is_finite() {
        requested_jitter_ratio.clamp(0.0, 1.0)
    } else {
        PROVIDER_RETRY_JITTER_RATIO
    };
    let requested_base_delay_ms = policy.base_delay_ms;
    let base_delay_ms = if requested_base_delay_ms.is_finite() {
        requested_base_delay_ms.max(0.0)
    } else {
        DEFAULT_PROVIDER_RETRY_BASE_DELAY_MS
    };
    let exponential = (base_delay_ms * 2f64.powf((attempt - 1.0).max(0.0))).min(MAX_TIMER_DELAY_MS);
    let delay_ms = match retry_after_ms {
        Some(retry_after_ms) if retry_after_ms >= exponential => {
            // Retry-After is a floor, not a client delay to scale down. Add only a
            // bounded positive client offset so peers given the same floor still spread.
            retry_after_ms + (exponential * jitter_ratio * random).round()
        }
        Some(retry_after_ms) => {
            let factor = 1.0 - jitter_ratio + 2.0 * jitter_ratio * random;
            (exponential * factor).round().max(retry_after_ms)
        }
        None => {
            let factor = 1.0 - jitter_ratio + 2.0 * jitter_ratio * random;
            (exponential * factor).round()
        }
    };
    ProviderRetryDelay::Wait {
        delay_ms: delay_ms.min(MAX_TIMER_DELAY_MS),
    }
}

/// `retryWouldExceedDeadline(delayMs, options)`.
fn retry_would_exceed_deadline(delay_ms: f64, options: &ProviderRetryExecutionOptions) -> bool {
    match options.deadline_at_ms {
        Some(deadline_at_ms) => now_of(options) + delay_ms > deadline_at_ms,
        None => false,
    }
}

/// `retryDeadlineReached(options)`.
fn retry_deadline_reached(options: &ProviderRetryExecutionOptions) -> bool {
    match options.deadline_at_ms {
        Some(deadline_at_ms) => now_of(options) >= deadline_at_ms,
        None => false,
    }
}

fn now_of(options: &ProviderRetryExecutionOptions) -> f64 {
    match &options.now {
        Some(now) => now(),
        None => now_millis(),
    }
}

/// Returns `Err(())` when the wait was aborted, matching the TypeScript `sleep`
/// rejection that `completeWithProviderRetry` turns into an abort.
async fn sleep_with_options(delay_ms: f64, options: &ProviderRetryExecutionOptions) -> Result<(), ()> {
    match &options.sleep {
        Some(sleep_fn) => {
            sleep_fn(delay_ms, options.signal.clone()).await;
            Ok(())
        }
        None => sleep(delay_ms.max(0.0) as u64, options.signal.as_ref())
            .await
            .map_err(|_| ()),
    }
}

/// `Date.now()`.
fn now_millis() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

/// `completeWithProviderRetry(attemptCompletion, options)`.
///
/// One-shot completion with the shared retry policy, for consumers outside the
/// AgentSession auto-retry loop (provider SDKs never retry internally).
pub async fn complete_with_provider_retry<F, Fut>(
    attempt_completion: F,
    options: ProviderRetryExecutionOptions,
) -> AssistantMessage
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = AssistantMessage>,
{
    let policy = options
        .policy
        .clone()
        .unwrap_or_else(|| DEFAULT_PROVIDER_RETRY_POLICY.clone());
    let max_retries = if policy.enabled { policy.max_retries } else { 0.0 };
    let mut retries_performed = 0.0;
    loop {
        let message = attempt_completion().await;
        if message.stop_reason != pi_ai::types::STOP_REASON_ERROR {
            return message;
        }
        if options
            .signal
            .as_ref()
            .map(|signal| signal.is_cancelled())
            .unwrap_or(false)
        {
            // A cancel that raced the failure is an abort, not a provider failure.
            return AssistantMessage {
                stop_reason: pi_ai::types::STOP_REASON_ABORTED.to_string(),
                ..message
            };
        }
        if retries_performed >= max_retries
            || is_agent_lifecycle_failure(&message)
            || is_faux_provider_queue_exhausted(&message)
            || cannot_replay_provider_failure(&message)
        {
            return message;
        }
        let kind = provider_stream_failure_kind(&message);
        if is_permanent_provider_failure_kind(kind.as_deref(), retries_performed) {
            return message;
        }
        let delay = provider_retry_delay(
            retries_performed + 1.0,
            provider_stream_failure_retry_after_ms(&message),
            &policy,
            &ProviderRetryDelayOptions {
                random: options.random.clone(),
                jitter_ratio: options.jitter_ratio,
            },
        );
        let delay_ms = match delay {
            ProviderRetryDelay::ExceedsCap { .. } => return message,
            ProviderRetryDelay::Wait { delay_ms } => delay_ms,
        };
        if retry_would_exceed_deadline(delay_ms, &options) {
            return message;
        }
        if sleep_with_options(delay_ms, &options).await.is_err() {
            return AssistantMessage {
                stop_reason: pi_ai::types::STOP_REASON_ABORTED.to_string(),
                ..message
            };
        }
        if options
            .signal
            .as_ref()
            .map(|signal| signal.is_cancelled())
            .unwrap_or(false)
        {
            return AssistantMessage {
                stop_reason: pi_ai::types::STOP_REASON_ABORTED.to_string(),
                ..message
            };
        }
        // Injected and real timers may resume late. Do not dispatch a provider call
        // after the request's owner deadline; preserve the original provider error.
        if retry_deadline_reached(&options) {
            return message;
        }
        retries_performed += 1.0;
    }
}

/// `DEFAULT_PROVIDER_RETRY_POLICY`.
pub fn default_provider_retry_policy() -> ProviderRetryPolicy {
    ProviderRetryPolicy {
        enabled: true,
        max_retries: 3.0,
        base_delay_ms: DEFAULT_PROVIDER_RETRY_BASE_DELAY_MS,
        max_retry_delay_ms: 60_000.0,
    }
}

/// `const DEFAULT_PROVIDER_RETRY_POLICY`.
pub static DEFAULT_PROVIDER_RETRY_POLICY: std::sync::LazyLock<ProviderRetryPolicy> =
    std::sync::LazyLock::new(default_provider_retry_policy);

/// `class ProviderRetryRequestError`-compatible throw for `requestWithProviderRetry`.
///
/// Unary provider requests throw instead of returning an assistant error message.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderRequestError {
    pub message: String,
    pub status: Option<f64>,
    pub retry_after_ms: Option<f64>,
    /// True when the thrown value was a network `TypeError`.
    pub is_type_error: bool,
}

impl ProviderRequestError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
            retry_after_ms: None,
            is_type_error: false,
        }
    }

    pub fn with_status(mut self, status: f64) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_retry_after_ms(mut self, retry_after_ms: f64) -> Self {
        self.retry_after_ms = Some(retry_after_ms);
        self
    }

    pub fn as_type_error(mut self) -> Self {
        self.is_type_error = true;
        self
    }
}

impl std::fmt::Display for ProviderRequestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ProviderRequestError {}

/// `requestWithProviderRetry(attemptRequest, options)`.
pub async fn request_with_provider_retry<T, F, Fut>(
    attempt_request: F,
    options: ProviderRetryExecutionOptions,
) -> Result<T, ProviderRequestError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, ProviderRequestError>>,
{
    let policy = options
        .policy
        .clone()
        .unwrap_or_else(|| DEFAULT_PROVIDER_RETRY_POLICY.clone());
    let max_retries = if policy.enabled { policy.max_retries } else { 0.0 };
    let mut attempt = 0.0;
    loop {
        throw_if_aborted(&options)?;
        match attempt_request().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                throw_if_aborted(&options)?;
                let status = error.status;
                let retry_after = error.retry_after_ms;
                let transient = status
                    .map(|status| status == 408.0 || status == 429.0 || status >= 500.0)
                    .unwrap_or(false)
                    || (error.is_type_error
                        && {
                            let message = error.message.to_lowercase();
                            message.contains("fetch") || message.contains("network") || message.contains("socket")
                        });
                if !transient || attempt >= max_retries {
                    return Err(error);
                }
                let delay = provider_retry_delay(
                    attempt + 1.0,
                    retry_after,
                    &policy,
                    &ProviderRetryDelayOptions {
                        random: options.random.clone(),
                        jitter_ratio: options.jitter_ratio,
                    },
                );
                let delay_ms = match delay {
                    ProviderRetryDelay::ExceedsCap { .. } => return Err(error),
                    ProviderRetryDelay::Wait { delay_ms } => delay_ms,
                };
                if retry_would_exceed_deadline(delay_ms, &options) {
                    return Err(error);
                }
                sleep_with_options(delay_ms, &options).await.map_err(|_| ProviderRequestError::new("Aborted"))?;
                throw_if_aborted(&options)?;
                // Keep the original request failure if the backoff overshot its owner deadline.
                if retry_deadline_reached(&options) {
                    return Err(error);
                }
                attempt += 1.0;
            }
        }
    }
}

/// `signal?.throwIfAborted()`.
fn throw_if_aborted(options: &ProviderRetryExecutionOptions) -> Result<(), ProviderRequestError> {
    if options
        .signal
        .as_ref()
        .map(|signal| signal.is_cancelled())
        .unwrap_or(false)
    {
        return Err(ProviderRequestError::new("The operation was aborted"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::{Usage, STOP_REASON_ERROR, STOP_REASON_STOP};
    use pi_ai::utils::diagnostics::AssistantMessageDiagnostic;
    use serde_json::json;

    fn options(random: f64) -> ProviderRetryDelayOptions {
        ProviderRetryDelayOptions {
            random: Some(Arc::new(move || random)),
            jitter_ratio: None,
        }
    }

    fn error_message(kind: Option<&str>) -> AssistantMessage {
        AssistantMessage {
            stop_reason: STOP_REASON_ERROR.to_string(),
            provider: "anthropic".to_string(),
            diagnostics: Some(vec![AssistantMessageDiagnostic {
                type_: "provider_stream_failure".to_string(),
                timestamp: 0,
                error: None,
                details: kind.map(|kind| json!({ "kind": kind }).as_object().cloned().unwrap()),
            }]),
            usage: Usage::zero(),
            ..Default::default()
        }
    }

    #[test]
    fn policy_defaults_match_the_typescript_constant() {
        let policy = default_provider_retry_policy();
        assert!(policy.enabled);
        assert_eq!(policy.max_retries, 3.0);
        assert_eq!(policy.base_delay_ms, 2000.0);
        assert_eq!(policy.max_retry_delay_ms, 60000.0);
        assert_eq!(*DEFAULT_PROVIDER_RETRY_POLICY, policy);
    }

    #[test]
    fn delay_grows_exponentially_and_respects_the_jitter_bounds() {
        let policy = default_provider_retry_policy();
        let low = provider_retry_delay(1.0, None, &policy, &options(0.0));
        assert_eq!(low, ProviderRetryDelay::Wait { delay_ms: 1600.0 });
        let high = provider_retry_delay(1.0, None, &policy, &options(1.0));
        assert_eq!(high, ProviderRetryDelay::Wait { delay_ms: 2400.0 });
        let third = provider_retry_delay(3.0, None, &policy, &options(0.5));
        assert_eq!(third, ProviderRetryDelay::Wait { delay_ms: 8000.0 });
    }

    #[test]
    fn retry_after_acts_as_a_floor_and_can_exceed_the_cap() {
        let policy = default_provider_retry_policy();
        let delay = provider_retry_delay(1.0, Some(30_000.0), &policy, &options(0.5));
        assert_eq!(delay, ProviderRetryDelay::Wait { delay_ms: 30_200.0 });

        let delay = provider_retry_delay(1.0, Some(120_000.0), &policy, &options(0.5));
        assert_eq!(delay, ProviderRetryDelay::ExceedsCap { retry_after_ms: 120_000.0 });

        let mut no_cap = policy.clone();
        no_cap.max_retry_delay_ms = 0.0;
        let delay = provider_retry_delay(1.0, Some(120_000.0), &no_cap, &options(0.5));
        assert_eq!(delay, ProviderRetryDelay::Wait { delay_ms: 120_200.0 });

        let delay = provider_retry_delay(1.0, Some(MAX_TIMER_DELAY_MS + 1.0), &no_cap, &options(0.5));
        assert_eq!(
            delay,
            ProviderRetryDelay::ExceedsCap {
                retry_after_ms: MAX_TIMER_DELAY_MS + 1.0
            }
        );
    }

    #[test]
    fn retry_after_below_the_exponential_is_clamped_up() {
        let policy = default_provider_retry_policy();
        let delay = provider_retry_delay(3.0, Some(100.0), &policy, &options(0.0));
        // exponential 8000 * 0.8 = 6400, floored up to retryAfter.
        assert_eq!(delay, ProviderRetryDelay::Wait { delay_ms: 6400.0 });
    }

    #[test]
    fn permanent_kinds_never_retry_and_auth_gets_one_retry() {
        for kind in ["invalid_request", "refusal", "safety", "permission"] {
            assert!(is_permanent_provider_failure_kind(Some(kind), 0.0));
        }
        assert!(!is_permanent_provider_failure_kind(Some("auth"), 0.0));
        assert!(is_permanent_provider_failure_kind(Some("auth"), 1.0));
        assert!(!is_permanent_provider_failure_kind(Some("server_error"), 5.0));
        assert!(!is_permanent_provider_failure_kind(None, 5.0));
    }

    #[test]
    fn stream_failure_details_are_read_from_provider_diagnostics() {
        let message = error_message(Some("rate_limit"));
        assert_eq!(provider_stream_failure_kind(&message).as_deref(), Some("rate_limit"));
        assert_eq!(provider_stream_failure_retry_after_ms(&message), None);
        assert!(provider_stream_failure_details(&message).is_some());
        assert_eq!(provider_stream_failure_kind(&AssistantMessage::default()), None);
    }

    #[test]
    fn lifecycle_and_faux_failures_are_identified() {
        let mut message = error_message(None);
        message.diagnostics = Some(vec![AssistantMessageDiagnostic {
            type_: "agent_lifecycle_failure".to_string(),
            timestamp: 0,
            error: None,
            details: None,
        }]);
        assert!(is_agent_lifecycle_failure(&message));

        let mut faux = AssistantMessage {
            provider: "faux".to_string(),
            error_message: Some("No more faux responses queued".to_string()),
            ..Default::default()
        };
        assert!(is_faux_provider_queue_exhausted(&faux));
        faux.error_message = Some("other".to_string());
        assert!(!is_faux_provider_queue_exhausted(&faux));
    }

    #[tokio::test]
    async fn complete_with_provider_retry_returns_success_without_sleeping() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let message = complete_with_provider_retry(
            move || {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    AssistantMessage {
                        stop_reason: STOP_REASON_STOP.to_string(),
                        ..Default::default()
                    }
                }
            },
            ProviderRetryExecutionOptions::default(),
        )
        .await;
        assert_eq!(message.stop_reason, STOP_REASON_STOP);
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn complete_with_provider_retry_stops_after_max_retries() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let mut options = ProviderRetryExecutionOptions::default();
        options.sleep = Some(Arc::new(|_delay, _signal| Box::pin(async {})));
        let message = complete_with_provider_retry(
            move || {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    error_message(Some("server_error"))
                }
            },
            options,
        )
        .await;
        assert_eq!(message.stop_reason, STOP_REASON_ERROR);
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn completion_retry_preserves_partial_or_uncertain_response_without_replay() {
        for (kind, content) in [
            ("request_interrupted", vec![]),
            ("safety", vec![]),
            ("server_error", vec![pi_ai::types::ContentBlock::Text(pi_ai::types::TextContent::new("partial"))]),
            ("server_error", vec![pi_ai::types::ContentBlock::Thinking(pi_ai::types::ThinkingContent::new("partial reasoning"))]),
            ("server_error", vec![pi_ai::types::ContentBlock::ToolCall(pi_ai::types::ToolCall::new("id", "tool", Default::default()))]),
        ] {
            let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counter = attempts.clone();
            let mut options = ProviderRetryExecutionOptions::default();
            options.sleep = Some(Arc::new(|_, _| Box::pin(async {})));
            let response = error_message(Some(kind));
            let response = AssistantMessage { content, ..response };
            let expected = response.clone();
            let actual = complete_with_provider_retry(move || {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let response = response.clone();
                async move { response }
            }, options).await;
            assert_eq!(actual, expected);
            assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1, "{kind}");
        }
    }

    #[tokio::test]
    async fn a_cancelled_signal_turns_a_failure_into_an_abort() {
        let signal = CancellationToken::new();
        signal.cancel();
        let message = complete_with_provider_retry(
            || async { error_message(Some("server_error")) },
            ProviderRetryExecutionOptions {
                signal: Some(signal),
                ..Default::default()
            },
        )
        .await;
        assert_eq!(message.stop_reason, pi_ai::types::STOP_REASON_ABORTED);
    }

    #[tokio::test]
    async fn request_with_provider_retry_rethrows_permanent_failures() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let mut options = ProviderRetryExecutionOptions::default();
        options.sleep = Some(Arc::new(|_delay, _signal| Box::pin(async {})));
        let result: Result<(), ProviderRequestError> = request_with_provider_retry(
            move || {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Err(ProviderRequestError::new("boom").with_status(400.0))
                }
            },
            options,
        )
        .await;
        assert!(result.is_err());
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn request_with_provider_retry_retries_transient_statuses() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let mut options = ProviderRetryExecutionOptions::default();
        options.sleep = Some(Arc::new(|_delay, _signal| Box::pin(async {})));
        let result: Result<u8, ProviderRequestError> = request_with_provider_retry(
            move || {
                let counter = counter.clone();
                async move {
                    let attempt = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if attempt < 2 {
                        Err(ProviderRequestError::new("busy").with_status(429.0))
                    } else {
                        Ok(7)
                    }
                }
            },
            options,
        )
        .await;
        assert_eq!(result.unwrap(), 7);
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 3);
    }
}
