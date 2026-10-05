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
    /// A14: provider id for the unary retry path, where the success value
    /// carries no provider. `None` (the default) keeps this call site out of
    /// per-provider resilience tracking entirely.
    pub provider: Option<String>,
    /// A14 resilience overrides for tests and programmatic callers; `None`
    /// reads the `PRIME_AGENT_*` environment configuration.
    pub resilience: Option<ProviderResiliencePolicy>,
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
    let resilience = options
        .resilience
        .clone()
        .unwrap_or_else(ProviderResiliencePolicy::from_env);
    let max_retries = if policy.enabled { effective_max_retries(&policy, &resilience) } else { 0.0 };
    let mut retries_performed = 0.0;
    loop {
        let message = attempt_completion().await;
        if message.stop_reason != pi_ai::types::STOP_REASON_ERROR {
            record_provider_attempt(&message.provider, false, &resilience, now_of(&options));
            return message;
        }
        record_provider_attempt(&message.provider, true, &resilience, now_of(&options));
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
        // A14: a 401/403-class failure gets one credential refresh before its
        // single retry, so an externally rotated token is picked up instead of
        // replaying the rejected one (dgx-k3 token expiry previously caused a
        // 46-minute outage with zero in-window recoveries). Disabled by default.
        if resilience.auth_refresh_retry
            && retries_performed == 0.0
            && matches!(kind.as_deref(), Some("auth") | Some("permission"))
        {
            refresh_cached_credentials();
        }
        if is_permanent_provider_failure_kind_with_resilience(kind.as_deref(), retries_performed, &resilience) {
            return message;
        }
        // A14 failover preference: once `failover_after_attempts` same-provider
        // attempts failed (this failure included), stop retrying here so the
        // host can switch providers (multi-attempt LRs measured 3-7x cost).
        // Disabled by default.
        if resilience
            .failover_after_attempts
            .is_some_and(|after| retries_performed + 1.0 >= after)
        {
            return message;
        }
        // A14 circuit breaker: while the provider is degraded, do not burn the
        // retry budget on it; return so the host fails over for the cooldown.
        if provider_degraded_for_ms(&message.provider, &resilience, now_of(&options)).is_some() {
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

// ---------------------------------------------------------------------------
// A14: auth-aware retry, per-provider circuit breaker, ordinal cap.
//
// Every knob here is env-configured and DEFAULTS TO THE PRE-A14 BEHAVIOR, per
// the quality contract for behavior-changing fixes. Evidence from the 15-day
// /monitor review (1.77M events): only 47% of requests hitting an attempt
// failure recovered; dgx-k3 token expiry caused a 46-minute outage with zero
// in-window recoveries (401/403 fast-fail terminally); multi-attempt logical
// requests cost 3-7x a single attempt.
// ---------------------------------------------------------------------------

/// Parse a boolean env flag: "1", "true", "yes", "on" (case-insensitive).
fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn env_number(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

/// Env-driven resilience additions to the retry policy. Defaults preserve the
/// exact pre-A14 behavior (everything disabled).
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderResiliencePolicy {
    /// On a 401/403-class failure (kind "auth" or "permission"), refresh the
    /// command-backed credential cache once and retry once before giving up.
    /// Default: false (auth keeps its existing single retry without refresh,
    /// permission stays terminal).
    pub auth_refresh_retry: bool,
    /// Enable the per-provider circuit breaker. Default: false.
    pub circuit_breaker_enabled: bool,
    /// Failure window in milliseconds. Default: 300_000 (5 minutes).
    pub circuit_breaker_window_ms: f64,
    /// Minimum attempts inside the window before the breaker may trip.
    /// Default: 20.
    pub circuit_breaker_min_attempts: u32,
    /// Failure ratio above which the breaker trips. Default: 0.5 (>50%).
    pub circuit_breaker_failure_ratio: f64,
    /// Cooldown in milliseconds during which a tripped provider is skipped
    /// so callers fail over. Default: 120_000 (2 minutes).
    pub circuit_breaker_cooldown_ms: f64,
    /// Hard cap on retry ordinals (attempts per logical request), overriding a
    /// larger configured `max_retries`. `None` keeps `max_retries` as-is.
    pub max_retry_ordinal: Option<f64>,
    /// Stop same-provider retries after this many attempts, so the host fails
    /// over to another provider instead (multi-attempt LRs cost 3-7x).
    /// `None` keeps the current retry count. Default: None.
    pub failover_after_attempts: Option<f64>,
}

impl Default for ProviderResiliencePolicy {
    fn default() -> Self {
        Self {
            auth_refresh_retry: false,
            circuit_breaker_enabled: false,
            circuit_breaker_window_ms: 300_000.0,
            circuit_breaker_min_attempts: 20,
            circuit_breaker_failure_ratio: 0.5,
            circuit_breaker_cooldown_ms: 120_000.0,
            max_retry_ordinal: None,
            failover_after_attempts: None,
        }
    }
}

impl ProviderResiliencePolicy {
    /// Read the env overrides. Unknown or invalid values fall back to defaults
    /// (never enabling a behavior from garbage input).
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            auth_refresh_retry: env_flag("PRIME_AGENT_AUTH_REFRESH_RETRY"),
            circuit_breaker_enabled: env_flag("PRIME_AGENT_PROVIDER_CIRCUIT_BREAKER"),
            circuit_breaker_window_ms: env_number("PRIME_AGENT_CIRCUIT_BREAKER_WINDOW_MS")
                .unwrap_or(defaults.circuit_breaker_window_ms),
            circuit_breaker_min_attempts: env_number("PRIME_AGENT_CIRCUIT_BREAKER_MIN_ATTEMPTS")
                .map(|value| value as u32)
                .unwrap_or(defaults.circuit_breaker_min_attempts),
            circuit_breaker_failure_ratio: env_number("PRIME_AGENT_CIRCUIT_BREAKER_FAILURE_RATIO")
                .map(|value| value.clamp(0.0, 1.0))
                .unwrap_or(defaults.circuit_breaker_failure_ratio),
            circuit_breaker_cooldown_ms: env_number("PRIME_AGENT_CIRCUIT_BREAKER_COOLDOWN_MS")
                .unwrap_or(defaults.circuit_breaker_cooldown_ms),
            max_retry_ordinal: env_number("PRIME_AGENT_MAX_RETRY_ORDINAL"),
            failover_after_attempts: env_number("PRIME_AGENT_FAILOVER_AFTER_ATTEMPTS"),
        }
    }
}

/// Bounded per-provider attempt history for the in-session circuit breaker.
#[derive(Debug, Default)]
struct ProviderAttemptHistory {
    /// (recorded_at_ms, failed) pairs, oldest first.
    attempts: std::collections::VecDeque<(f64, bool)>,
    /// While `now < degraded_until_ms`, the provider is skipped so callers fail over.
    degraded_until_ms: Option<f64>,
}

/// Process-global breaker state, keyed by provider id ("in-session": one
/// registry per agent process, never persisted).
static PROVIDER_ATTEMPT_HISTORIES: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, ProviderAttemptHistory>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Hard cap on retained attempt records per provider, so a pathological
/// provider cannot grow the registry unboundedly between window evictions.
const PROVIDER_ATTEMPT_HISTORY_CAP: usize = 1_024;

/// Record one attempt outcome for `provider` under `policy` at `now_ms`.
///
/// No-op unless the breaker is enabled: with the default policy this function
/// keeps zero state, exactly like pre-A14.
pub fn record_provider_attempt(provider: &str, failed: bool, policy: &ProviderResiliencePolicy, now_ms: f64) {
    if !policy.circuit_breaker_enabled || provider.is_empty() {
        return;
    }
    let mut histories = PROVIDER_ATTEMPT_HISTORIES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let history = histories.entry(provider.to_string()).or_default();
    let window_start = now_ms - policy.circuit_breaker_window_ms;
    while history
        .attempts
        .front()
        .is_some_and(|(recorded_at, _)| *recorded_at < window_start)
    {
        history.attempts.pop_front();
    }
    if history.attempts.len() >= PROVIDER_ATTEMPT_HISTORY_CAP {
        history.attempts.pop_front();
    }
    history.attempts.push_back((now_ms, failed));
    // A success after a trip does not immediately restore the provider: the
    // cooldown keeps failover preferred until it expires (half-open probing).
    if history.degraded_until_ms.is_some_and(|until| until <= now_ms) {
        history.degraded_until_ms = None;
    }
    let recorded = history.attempts.len();
    if (recorded as u32) < policy.circuit_breaker_min_attempts.max(1) {
        return;
    }
    let failures = history.attempts.iter().filter(|(_, failed)| *failed).count();
    let failure_ratio = failures as f64 / recorded as f64;
    if failure_ratio > policy.circuit_breaker_failure_ratio {
        history.degraded_until_ms = Some(now_ms + policy.circuit_breaker_cooldown_ms);
    }
}

/// While the breaker holds `provider` degraded, retries should skip it so the
/// host can fail over. Returns the remaining cooldown in milliseconds.
pub fn provider_degraded_for_ms(provider: &str, policy: &ProviderResiliencePolicy, now_ms: f64) -> Option<f64> {
    if !policy.circuit_breaker_enabled || provider.is_empty() {
        return None;
    }
    let histories = PROVIDER_ATTEMPT_HISTORIES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let history = histories.get(provider)?;
    let remaining = history.degraded_until_ms? - now_ms;
    (remaining > 0.0).then_some(remaining)
}

/// Forget all breaker state (tests and explicit resets).
pub fn reset_provider_attempt_history() {
    PROVIDER_ATTEMPT_HISTORIES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

/// Refresh the command-backed credential cache so the next attempt re-resolves
/// credentials (A9 cache invalidation; for OAuth providers the caller's own
/// stale-marking flow remains the refresh path).
fn refresh_cached_credentials() {
    crate::core::resolve_config_value::invalidate_resolved_command_values();
}

/// The auth-aware permanent check. Pre-A14 semantics are preserved when the
/// resilience policy is default: `auth` keeps exactly one retry and
/// `permission` stays terminal. With `auth_refresh_retry` enabled, both kinds
/// get one credential-refreshed retry.
fn is_permanent_provider_failure_kind_with_resilience(
    kind: Option<&str>,
    retries_performed: f64,
    resilience: &ProviderResiliencePolicy,
) -> bool {
    if matches!(kind, Some("auth") | Some("permission")) && resilience.auth_refresh_retry {
        // One refresh + one retry: the second consecutive auth-class failure is terminal.
        return retries_performed > 0.0;
    }
    is_permanent_provider_failure_kind(kind, retries_performed)
}

/// Effective retry budget: `max_retries`, capped by the optional ordinal cap.
fn effective_max_retries(policy: &ProviderRetryPolicy, resilience: &ProviderResiliencePolicy) -> f64 {
    match resilience.max_retry_ordinal {
        Some(cap) => policy.max_retries.min(cap),
        None => policy.max_retries,
    }
}

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
    /// Provider id for per-provider resilience (A14 circuit breaker,
    /// auth-refresh retry). `None` keeps the pre-A14 behavior of not
    /// participating in per-provider tracking.
    pub provider: Option<String>,
}

impl ProviderRequestError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
            retry_after_ms: None,
            is_type_error: false,
            provider: None,
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

    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
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
    let resilience = options
        .resilience
        .clone()
        .unwrap_or_else(ProviderResiliencePolicy::from_env);
    let max_retries = if policy.enabled { effective_max_retries(&policy, &resilience) } else { 0.0 };
    let mut attempt = 0.0;
    loop {
        throw_if_aborted(&options)?;
        match attempt_request().await {
            Ok(value) => {
                if let Some(provider) = options.provider.as_deref() {
                    record_provider_attempt(provider, false, &resilience, now_of(&options));
                }
                return Ok(value);
            }
            Err(error) => {
                throw_if_aborted(&options)?;
                if let Some(provider) = error.provider.as_deref().or(options.provider.as_deref()) {
                    record_provider_attempt(provider, true, &resilience, now_of(&options));
                }
                let status = error.status;
                let retry_after = error.retry_after_ms;
                // A14: one credential refresh + one retry for 401/403 (disabled by
                // default; pre-A14 these stay terminal for the unary path).
                let auth_refresh_retry = resilience.auth_refresh_retry
                    && attempt == 0.0
                    && status.map(|status| status == 401.0 || status == 403.0).unwrap_or(false);
                let transient = status
                    .map(|status| status == 408.0 || status == 429.0 || status >= 500.0)
                    .unwrap_or(false)
                    || auth_refresh_retry
                    || (error.is_type_error
                        && {
                            let message = error.message.to_lowercase();
                            message.contains("fetch") || message.contains("network") || message.contains("socket")
                        });
                if auth_refresh_retry {
                    refresh_cached_credentials();
                }
                // A14 failover preference and circuit breaker (both disabled by
                // default): stop same-provider retries so the host fails over.
                if resilience
                    .failover_after_attempts
                    .is_some_and(|after| attempt + 1.0 >= after)
                {
                    return Err(error);
                }
                if let Some(provider) = error.provider.as_deref().or(options.provider.as_deref()) {
                    if provider_degraded_for_ms(provider, &resilience, now_of(&options)).is_some() {
                        return Err(error);
                    }
                }
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
    async fn auth_refresh_retry_disabled_keeps_permission_terminal_and_auth_single_retry() {
        // Pre-A14 behavior, pinned: permission is terminal, auth retries exactly
        // once without any credential refresh.
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for (kind, expected) in [("permission", 1), ("auth", 2)] {
            let counter = attempts.clone();
            let mut options = ProviderRetryExecutionOptions::default();
            options.sleep = Some(Arc::new(|_, _| Box::pin(async {})));
            options.resilience = Some(ProviderResiliencePolicy::default());
            let response = error_message(Some(kind));
            let message = complete_with_provider_retry(
                move || {
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let response = response.clone();
                    async move { response }
                },
                options,
            )
            .await;
            assert_eq!(message.stop_reason, STOP_REASON_ERROR);
            assert_eq!(
                attempts.load(std::sync::atomic::Ordering::SeqCst),
                expected,
                "{kind}"
            );
            attempts.store(0, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn auth_refresh_retry_retries_auth_and_permission_once() {
        // With the flag on, a 401/403-class failure refreshes cached
        // credentials and retries once; the second consecutive failure is
        // terminal (no multi-attempt amplification).
        for kind in ["auth", "permission"] {
            let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counter = attempts.clone();
            let mut options = ProviderRetryExecutionOptions::default();
            options.sleep = Some(Arc::new(|_, _| Box::pin(async {})));
            options.resilience = Some(ProviderResiliencePolicy {
                auth_refresh_retry: true,
                ..Default::default()
            });
            let response = error_message(Some(kind));
            let message = complete_with_provider_retry(
                move || {
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let response = response.clone();
                    async move { response }
                },
                options,
            )
            .await;
            assert_eq!(message.stop_reason, STOP_REASON_ERROR);
            assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2, "{kind}");
        }
    }

    #[tokio::test]
    async fn circuit_breaker_trips_above_ratio_within_the_window() {
        reset_provider_attempt_history();
        let policy = ProviderResiliencePolicy {
            circuit_breaker_enabled: true,
            circuit_breaker_window_ms: 300_000.0,
            circuit_breaker_min_attempts: 4,
            circuit_breaker_failure_ratio: 0.5,
            circuit_breaker_cooldown_ms: 120_000.0,
            ..Default::default()
        };
        // 2 of 4 failed = exactly 50%: the breaker requires strictly more.
        for failed in [true, true, false, false] {
            record_provider_attempt("breaker-window", failed, &policy, 1_000.0);
        }
        assert_eq!(provider_degraded_for_ms("breaker-window", &policy, 1_000.0), None);

        // 3 of 5 failed = 60%: trips, and the cooldown remains visible.
        record_provider_attempt("breaker-window", true, &policy, 2_000.0);
        assert_eq!(
            provider_degraded_for_ms("breaker-window", &policy, 2_000.0),
            Some(120_000.0)
        );
        // Still degraded inside the cooldown...
        assert!(provider_degraded_for_ms("breaker-window", &policy, 121_999.0).is_some());
        // ...and released the moment it expires (half-open: attempts resume recording).
        assert_eq!(provider_degraded_for_ms("breaker-window", &policy, 122_000.0), None);

        // After the cooldown the provider is released, but mid-window
        // dilution re-trips while failures dominate (4 of 5 > 50%).
        for failed in [true, true, true, true] {
            record_provider_attempt("breaker-stale", failed, &policy, 0.0);
        }
        assert!(provider_degraded_for_ms("breaker-stale", &policy, 0.0).is_some());
        assert_eq!(provider_degraded_for_ms("breaker-stale", &policy, 120_000.0), None);
        record_provider_attempt("breaker-stale", false, &policy, 130_000.0);
        assert!(provider_degraded_for_ms("breaker-stale", &policy, 130_000.0).is_some());
        // Once the failures age out of the window entirely, a fresh success
        // leaves the provider healthy.
        record_provider_attempt("breaker-stale", false, &policy, 500_000.0);
        assert_eq!(provider_degraded_for_ms("breaker-stale", &policy, 500_000.0), None);

        // Below the minimum attempt count nothing trips, even at 100% failures.
        for _ in 0..3 {
            record_provider_attempt("breaker-min", true, &policy, 5_000.0);
        }
        assert_eq!(provider_degraded_for_ms("breaker-min", &policy, 5_000.0), None);

        // Disabled policy keeps zero state.
        let disabled = ProviderResiliencePolicy::default();
        for _ in 0..50 {
            record_provider_attempt("breaker-off", true, &disabled, 9_000.0);
        }
        assert_eq!(provider_degraded_for_ms("breaker-off", &disabled, 9_000.0), None);
        reset_provider_attempt_history();
    }

    #[tokio::test]
    async fn circuit_breaker_skips_retries_while_degraded() {
        // A dedicated provider id keeps this test independent of the other
        // breaker tests that reset the shared history.
        let failure = || {
            let mut message = error_message(Some("server_error"));
            message.provider = "breaker-skip".to_string();
            message
        };
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let mut options = ProviderRetryExecutionOptions::default();
        options.sleep = Some(Arc::new(|_, _| Box::pin(async {})));
        options.now = Some(Arc::new(|| 10_000.0));
        options.resilience = Some(ProviderResiliencePolicy {
            circuit_breaker_enabled: true,
            circuit_breaker_min_attempts: 2,
            circuit_breaker_failure_ratio: 0.5,
            circuit_breaker_cooldown_ms: 120_000.0,
            ..Default::default()
        });
        let message = complete_with_provider_retry(
            move || {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move { failure() }
            },
            options,
        )
        .await;
        assert_eq!(message.stop_reason, STOP_REASON_ERROR);
        // Two failures trip the breaker (2 of 2 > 50%), so the second failure
        // is returned without a third same-provider attempt.
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert!(provider_degraded_for_ms("breaker-skip", &ProviderResiliencePolicy {
            circuit_breaker_enabled: true,
            circuit_breaker_min_attempts: 2,
            circuit_breaker_failure_ratio: 0.5,
            circuit_breaker_cooldown_ms: 120_000.0,
            ..Default::default()
        }, 10_000.0).is_some());
    }

    #[tokio::test]
    async fn failover_after_attempts_stops_same_provider_retries() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let mut options = ProviderRetryExecutionOptions::default();
        options.sleep = Some(Arc::new(|_, _| Box::pin(async {})));
        options.resilience = Some(ProviderResiliencePolicy {
            failover_after_attempts: Some(2.0),
            ..Default::default()
        });
        let response = error_message(Some("server_error"));
        let message = complete_with_provider_retry(
            move || {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let response = response.clone();
                async move { response }
            },
            options,
        )
        .await;
        assert_eq!(message.stop_reason, STOP_REASON_ERROR);
        // Attempt 1 and 2 run; after the second failure the host is told to
        // fail over instead of attempting 3 and 4 (multi-attempt LRs cost 3-7x).
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn max_retry_ordinal_caps_a_larger_configured_budget() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let mut options = ProviderRetryExecutionOptions::default();
        options.sleep = Some(Arc::new(|_, _| Box::pin(async {})));
        options.policy = Some(ProviderRetryPolicy {
            max_retries: 5.0,
            ..default_provider_retry_policy()
        });
        options.resilience = Some(ProviderResiliencePolicy {
            max_retry_ordinal: Some(3.0),
            ..Default::default()
        });
        let response = error_message(Some("server_error"));
        let message = complete_with_provider_retry(
            move || {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let response = response.clone();
                async move { response }
            },
            options,
        )
        .await;
        assert_eq!(message.stop_reason, STOP_REASON_ERROR);
        // 1 attempt + at most 3 retries: the ordinal cap wins over maxRetries=5.
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 4);
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
