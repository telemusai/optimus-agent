//! Port of packages/coding-agent/src/modes/daemon/daemon-session-summarizer.ts
//!
//! blocked_on: needs core::provider_retry::{ProviderRetryPolicy,
//! providerRetryPolicy, completeWithProviderRetry}; the minimal retry loop and
//! policy resolution are inlined here until that module lands.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use futures::future::BoxFuture;
use pi_agent_core::types::{AgentMessage, CustomMessageContent};
use pi_ai::types::{
    AssistantMessage, Context, ImageOrTextContent, Message, Model, SimpleStreamOptions, StreamOptions,
    TextContent, UserContent, UserMessage, STOP_REASON_ERROR,
};
use regex::Regex;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::active_session_state::{
    ActiveSessionState, ActiveSessionRuntimeSession, AgentStatus, AgentSessionRuntime,
};
use crate::core::model_registry::{ModelRegistry, ResolvedRequestAuth};
use crate::core::settings_manager::SettingsManager;

const SWEEP_INTERVAL_MS: u64 = 25_000;
// Collapse a tool-use loop's rapid turn_end bursts into one summarization.
const SETTLE_DEBOUNCE_MS: u64 = 2_000;
// Idle generations stop retrying (and paying) on unchanged content until the
// backoff elapses, so transient outages and late credentials still recover.
const IDLE_GENERATION_ATTEMPT_LIMIT: u32 = 3;
const IDLE_GENERATION_RETRY_BACKOFF_MS: u64 = 30 * 60_000;

const SUMMARY_MODEL_PROVIDER: &str = "prime-inference";
const SUMMARY_MODEL_ID: &str = "qwen/qwen3-30b-a3b-instruct-2507";

const SUMMARY_CONTEXT_MESSAGES: usize = 8;
const SUMMARY_MAX_CHARS_PER_MESSAGE: usize = 600;
// Generous so a chatty model still closes the tags before truncation.
const SUMMARY_MAX_TOKENS: f64 = 400.0;

pub const AGENT_STATUS_SYSTEM_PROMPT: &str = r#"You generate a status line for an AI coding agent dashboard. You are given the recent conversation between a user and the agent, plus whether the agent is currently working or idle.

Output ONLY these two tags, nothing before, between, or after. Do not think out loud, explain, or count words.
<recap>a present-tense clause, at most 12 words, saying what the agent is doing or just did, no trailing period</recap>
<status>one of NEEDS_INPUT, COMPLETED</status>

STATUS meaning:
- COMPLETED: the agent finished its turn AND the user's request is fully done with nothing left.
- NEEDS_INPUT: the agent finished its turn but the task is not fully done — it asked a question, hit a blocker, or needs more prompting.
When you are unsure between COMPLETED and NEEDS_INPUT, choose NEEDS_INPUT.

Example:
<recap>Refactoring the auth middleware and updating its tests</recap>
<status>NEEDS_INPUT</status>"#;

/// `AgentStatusResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentStatusResult {
    pub summary: String,
    pub task_state: Option<String>,
}

/// Minimal `ProviderRetryPolicy` (core/provider-retry.ts).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProviderRetryPolicy {
    pub enabled: bool,
    pub max_retries: u32,
    pub base_delay_ms: f64,
    pub max_retry_delay_ms: f64,
}

pub const DEFAULT_PROVIDER_RETRY_POLICY: ProviderRetryPolicy = ProviderRetryPolicy {
    enabled: true,
    max_retries: 3,
    base_delay_ms: 2_000.0,
    max_retry_delay_ms: 60_000.0,
};

const PROVIDER_RETRY_JITTER_RATIO: f64 = 0.2;
const MAX_TIMER_DELAY_MS: f64 = 2_147_483_647.0;
const DEFAULT_PROVIDER_RETRY_BASE_DELAY_MS: f64 = 2_000.0;

pub fn provider_retry_policy(settings_manager: &SettingsManager) -> ProviderRetryPolicy {
    let retry = settings_manager.get_retry_settings();
    ProviderRetryPolicy {
        enabled: retry.enabled,
        max_retries: retry.max_retries.max(0.0) as u32,
        base_delay_ms: retry.base_delay_ms,
        max_retry_delay_ms: settings_manager.get_provider_retry_settings().max_retry_delay_ms,
    }
}

/// Resolve the cheap summary model, or None when it has no configured auth.
pub fn resolve_summary_model(registry: &ModelRegistry) -> Option<Model> {
    let model = registry.find(SUMMARY_MODEL_PROVIDER, SUMMARY_MODEL_ID)?;
    if registry.has_configured_auth(&model) {
        Some(model)
    } else {
        None
    }
}

/// `messageText(message.content)` - duck-typed on the serialized content.
pub fn message_text(content: &Value) -> (String, Vec<String>) {
    if let Some(text) = content.as_str() {
        return (text.to_string(), Vec::new());
    }
    let Some(blocks) = content.as_array() else {
        return (String::new(), Vec::new());
    };
    let mut parts: Vec<String> = Vec::new();
    let mut tools: Vec<String> = Vec::new();
    for block in blocks {
        let Some(block) = block.as_object() else {
            continue;
        };
        let type_ = block.get("type").and_then(Value::as_str);
        if type_ == Some("text") {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                parts.push(text.to_string());
            }
        } else if type_ == Some("tool_use") || type_ == Some("toolUse") {
            if let Some(name) = block.get("name").and_then(Value::as_str) {
                tools.push(name.to_string());
            }
        }
    }
    (parts.join("\n"), tools)
}

fn clamp(text: &str, max: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() > max {
        let truncated: String = normalized.chars().take(max).collect();
        format!("{truncated}…")
    } else {
        normalized
    }
}

/// Serialize the trailing messages into a compact prompt body (tool calls by name only).
pub fn build_status_context(messages: &[AgentMessage], is_working: bool) -> String {
    let start = messages.len().saturating_sub(SUMMARY_CONTEXT_MESSAGES);
    let recent = &messages[start..];
    let mut lines: Vec<String> = Vec::new();
    for message in recent {
        let role = message.role();
        if role != "user" && role != "assistant" && role != "toolResult" && role != "custom" {
            continue;
        }
        let content = serde_json::to_value(message)
            .ok()
            .and_then(|value| value.get("content").cloned())
            .unwrap_or(Value::Null);
        let (text, tools) = message_text(&content);
        let body = clamp(&text, SUMMARY_MAX_CHARS_PER_MESSAGE);
        let tool_note = if tools.is_empty() {
            String::new()
        } else {
            let mut unique: Vec<String> = Vec::new();
            for tool in &tools {
                if !unique.contains(tool) {
                    unique.push(tool.clone());
                }
            }
            format!("[tools: {}]", unique.join(", "))
        };
        let rendered = [body, tool_note]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<String>>()
            .join(" ");
        if !rendered.is_empty() {
            lines.push(format!("{role}: {rendered}"));
        }
    }
    let state = if is_working {
        "working"
    } else {
        "idle (finished its turn)"
    };
    format!(
        "<agent-state>{state}</agent-state>\n<conversation>\n{}\n</conversation>",
        lines.join("\n")
    )
}

/// Cuts a word-counting trailer the model sometimes appends.
fn reasoning_trailer() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?i)\s*(?:["”]\s*)?(?:\bthat['’]?s\s+\d+\s*words?\b|\bcount\s*:|\(\d+\)|=\s*\d+\s*words?\b).*"#)
            .expect("valid reasoning trailer regex")
    })
}

fn counting_artifact() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\(\d+\)|=\s*\d+\s*words?\b").expect("valid counting artifact regex"))
}

fn trailing_markers() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[.\s]+$").expect("valid trailing marker regex"))
}

fn leading_quotes() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"^["“']+"#).expect("valid leading quote regex"))
}

fn trailing_quotes() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"["”']+$"#).expect("valid trailing quote regex"))
}

fn recap_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?is)<recap>(.*?)</recap>").expect("valid recap regex"))
}

fn status_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)<status>\s*([a-z_]+)\s*</status>").expect("valid status regex"))
}

fn model_marker_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)present-tense|12 words").expect("valid marker regex"))
}

const MAX_RECAP_WORDS: usize = 16;

fn clean_recap(raw: &str) -> Option<String> {
    let value = reasoning_trailer().replace(raw.trim(), "").to_string();
    let value = leading_quotes().replace(&value, "").to_string();
    let value = trailing_quotes().replace(&value, "").to_string();
    let value = trailing_markers().replace(&value, "").trim().to_string();
    if value.is_empty() || value.starts_with('<') || model_marker_re().is_match(&value) {
        return None;
    }
    if counting_artifact().is_match(&value) || value.split_whitespace().count() > MAX_RECAP_WORDS {
        return None;
    }
    Some(value)
}

/// Take the content of the last `<recap>` and `<status>` tags; idle verdicts default to needs_input.
pub fn parse_agent_status_response(text: &str, is_working: bool) -> Option<AgentStatusResult> {
    // Normalize unicode angle-bracket lookalikes so a tag written with them still parses.
    let cleaned = text.replace(['‹', '＜'], "<").replace(['›', '＞'], ">");

    let summary = recap_re()
        .captures_iter(&cleaned)
        .last()
        .and_then(|captures| captures.get(1).map(|capture| capture.as_str().to_string()))
        .and_then(|raw| clean_recap(&raw));
    let summary = summary?;
    if is_working {
        return Some(AgentStatusResult {
            summary,
            task_state: None,
        });
    }
    let status = status_re()
        .captures_iter(&cleaned)
        .last()
        .and_then(|captures| captures.get(1).map(|capture| capture.as_str().to_uppercase()));
    let task_state = if status.as_deref() == Some("COMPLETED") {
        "completed"
    } else {
        "needs_input"
    };
    Some(AgentStatusResult {
        summary,
        task_state: Some(task_state.to_string()),
    })
}

#[derive(Clone)]
pub struct GenerateAgentStatusParams {
    pub registry: Arc<tokio::sync::Mutex<ModelRegistry>>,
    pub messages: Vec<AgentMessage>,
    pub is_working: bool,
    pub retry_policy: Option<ProviderRetryPolicy>,
    pub signal: Option<CancellationToken>,
}

/// One cheap model call for a fresh status, or None if unavailable/empty/failed.
pub async fn generate_agent_status(params: GenerateAgentStatusParams) -> Option<AgentStatusResult> {
    let GenerateAgentStatusParams {
        registry,
        messages,
        is_working,
        retry_policy,
        signal,
    } = params;
    if messages.is_empty() {
        return None;
    }
    let (model, auth) = {
        let mut registry = registry.lock().await;
        let model = resolve_summary_model(&registry)?;
        let auth: ResolvedRequestAuth = registry.get_api_key_and_headers(&model).await;
        (model, auth)
    };
    if !auth.ok || auth.api_key.is_none() {
        return None;
    }
    let api_key = auth.api_key.clone().unwrap_or_default();
    let headers = auth.headers.clone();
    let context = Context::new(
        Some(AGENT_STATUS_SYSTEM_PROMPT.to_string()),
        vec![Message::User(UserMessage {
            role: "user".to_string(),
            content: UserContent::Blocks(vec![ImageOrTextContent::Text(TextContent::new(build_status_context(
                &messages, is_working,
            )))]),
            provider_context: None,
            timestamp: now_millis(),
        })],
        None,
    );
    let options = SimpleStreamOptions {
        stream: StreamOptions {
            max_tokens: Some(SUMMARY_MAX_TOKENS),
            api_key: Some(api_key),
            headers,
            signal: signal.clone(),
            ..Default::default()
        },
        reasoning: None,
        thinking_budgets: None,
    };

    // One failed attempt would settle an idle session to a stale needs_input verdict.
    let response: AssistantMessage = complete_with_provider_retry(
        &|| {
            let model = model.clone();
            let context = context.clone();
            let options = options.clone();
            Box::pin(async move { pi_ai::stream::complete_simple(&model, &context, Some(&options)).await })
        },
        retry_policy.unwrap_or(DEFAULT_PROVIDER_RETRY_POLICY),
        signal.clone(),
    )
    .await;
    if response.stop_reason == STOP_REASON_ERROR {
        return None;
    }
    let text_content = response
        .content
        .iter()
        .filter_map(|block| block.as_text().map(|text| text.text.clone()))
        .collect::<Vec<String>>()
        .join("\n");
    parse_agent_status_response(&text_content, is_working)
}

type CompleteFuture = BoxFuture<'static, AssistantMessage>;

async fn complete_with_provider_retry(
    attempt_completion: &(dyn Fn() -> CompleteFuture + Send + Sync),
    policy: ProviderRetryPolicy,
    signal: Option<CancellationToken>,
) -> AssistantMessage {
    let max_retries = if policy.enabled { policy.max_retries } else { 0 };
    let mut retries_performed: u32 = 0;
    loop {
        let message = attempt_completion().await;
        if message.stop_reason != STOP_REASON_ERROR {
            return message;
        }
        if signal.as_ref().is_some_and(CancellationToken::is_cancelled) {
            // A cancel that raced the failure is an abort, not a provider failure.
            let mut aborted = message;
            aborted.stop_reason = pi_ai::types::STOP_REASON_ABORTED.to_string();
            return aborted;
        }
        if retries_performed >= max_retries
            || is_agent_lifecycle_failure(&message)
            || is_faux_provider_queue_exhausted(&message)
        {
            return message;
        }
        let kind = provider_stream_failure_kind(&message);
        if is_permanent_provider_failure_kind(kind.as_deref(), retries_performed) {
            return message;
        }
        let Some(delay_ms) = provider_retry_delay(
            retries_performed + 1,
            provider_stream_failure_retry_after_ms(&message),
            &policy,
        ) else {
            return message;
        };
        if crate::utils::sleep::sleep(delay_ms, signal.as_ref()).await.is_err() {
            let mut aborted = message;
            aborted.stop_reason = pi_ai::types::STOP_REASON_ABORTED.to_string();
            return aborted;
        }
        if signal.as_ref().is_some_and(CancellationToken::is_cancelled) {
            let mut aborted = message;
            aborted.stop_reason = pi_ai::types::STOP_REASON_ABORTED.to_string();
            return aborted;
        }
        retries_performed += 1;
    }
}

fn is_agent_lifecycle_failure(message: &AssistantMessage) -> bool {
    message
        .diagnostics
        .as_ref()
        .is_some_and(|diagnostics| diagnostics.iter().any(|diagnostic| diagnostic.type_ == "agent_lifecycle_failure"))
}

fn is_faux_provider_queue_exhausted(message: &AssistantMessage) -> bool {
    message.provider == "faux" && message.error_message.as_deref() == Some("No more faux responses queued")
}

fn provider_stream_failure_details(message: &AssistantMessage) -> Option<serde_json::Map<String, Value>> {
    message
        .diagnostics
        .as_ref()?
        .iter()
        .find(|diagnostic| diagnostic.type_ == "provider_stream_failure")
        .and_then(|diagnostic| diagnostic.details.clone())
}

fn provider_stream_failure_kind(message: &AssistantMessage) -> Option<String> {
    provider_stream_failure_details(message)?
        .get("kind")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn provider_stream_failure_retry_after_ms(message: &AssistantMessage) -> Option<f64> {
    let value = provider_stream_failure_details(message)?.get("retryAfterMs")?.as_f64()?;
    if value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

/// Deterministic rejections never retry; auth gets one retry before it can be marked stale.
fn is_permanent_provider_failure_kind(kind: Option<&str>, retries_performed: u32) -> bool {
    crate::core::provider_retry::is_permanent_provider_failure_kind(kind, f64::from(retries_performed))
}

/// Delay before retry `attempt` (1-based), honoring a server-requested not-before time.
fn provider_retry_delay(attempt: u32, retry_after_ms: Option<f64>, policy: &ProviderRetryPolicy) -> Option<u64> {
    if let Some(retry_after_ms) = retry_after_ms {
        if policy.max_retry_delay_ms > 0.0 && retry_after_ms > policy.max_retry_delay_ms {
            return None;
        }
        if retry_after_ms > MAX_TIMER_DELAY_MS {
            return None;
        }
    }
    let random = bounded_random();
    let base_delay_ms = if policy.base_delay_ms.is_finite() {
        policy.base_delay_ms.max(0.0)
    } else {
        DEFAULT_PROVIDER_RETRY_BASE_DELAY_MS
    };
    let exponential = (base_delay_ms * 2f64.powi(attempt.saturating_sub(1).max(0) as i32)).min(MAX_TIMER_DELAY_MS);
    let delay_ms = match retry_after_ms {
        Some(retry_after_ms) if retry_after_ms >= exponential => {
            retry_after_ms + (exponential * PROVIDER_RETRY_JITTER_RATIO * random).round()
        }
        Some(retry_after_ms) => {
            let factor = 1.0 - PROVIDER_RETRY_JITTER_RATIO + 2.0 * PROVIDER_RETRY_JITTER_RATIO * random;
            (exponential * factor).round().max(retry_after_ms)
        }
        None => {
            let factor = 1.0 - PROVIDER_RETRY_JITTER_RATIO + 2.0 * PROVIDER_RETRY_JITTER_RATIO * random;
            (exponential * factor).round()
        }
    };
    Some(delay_ms.min(MAX_TIMER_DELAY_MS) as u64)
}

fn bounded_random() -> f64 {
    let mut bytes = [0u8; 8];
    let value = match getrandom_f64(&mut bytes) {
        Some(value) => value,
        None => 0.5,
    };
    value.clamp(0.0, 1.0)
}

fn getrandom_f64(bytes: &mut [u8; 8]) -> Option<f64> {
    use rand::RngCore;
    rand::thread_rng().try_fill_bytes(bytes).ok()?;
    let integer = u64::from_le_bytes(*bytes);
    Some((integer >> 11) as f64 / (1u64 << 53) as f64)
}

fn is_session_working(state: &ActiveSessionState) -> bool {
    state.runtime.session.is_session_active
}

/// `state.runtime.session !== session` for cloned snapshots.
fn session_identity(session: &ActiveSessionRuntimeSession) -> (String, Option<String>) {
    (session.session_id.clone(), session.session_file.clone())
}

fn now_millis() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

#[derive(Clone)]
struct FailedIdleGeneration {
    content_key: String,
    attempts: u32,
    last_failure_at: f64,
}

pub type ListSessions = Arc<dyn Fn() -> Vec<Arc<StdMutex<ActiveSessionState>>> + Send + Sync>;
pub type OnStatusChanged = Arc<dyn Fn(&ActiveSessionState) + Send + Sync>;
pub type GenerateAgentStatusFn =
    Arc<dyn Fn(GenerateAgentStatusParams) -> BoxFuture<'static, Option<AgentStatusResult>> + Send + Sync>;

/// Background status summarization for daemon-hosted sessions.
pub struct DaemonSessionSummarizer {
    list_sessions: ListSessions,
    on_status_changed: Option<OnStatusChanged>,
    generate: GenerateAgentStatusFn,
    interval: StdMutex<Option<tokio::task::JoinHandle<()>>>,
    interval_running: Arc<AtomicBool>,
    debounce_timers: StdMutex<HashMap<String, tokio::task::JoinHandle<()>>>,
    in_flight: StdMutex<HashMap<String, CancellationToken>>,
    rerun_requested: StdMutex<HashSet<String>>,
    failed_idle_generations: StdMutex<HashMap<String, FailedIdleGeneration>>,
}

impl DaemonSessionSummarizer {
    pub fn new(list_sessions: ListSessions, on_status_changed: Option<OnStatusChanged>) -> Self {
        Self {
            list_sessions,
            on_status_changed,
            generate: Arc::new(|params| Box::pin(generate_agent_status(params))),
            interval: StdMutex::new(None),
            interval_running: Arc::new(AtomicBool::new(false)),
            debounce_timers: StdMutex::new(HashMap::new()),
            in_flight: StdMutex::new(HashMap::new()),
            rerun_requested: StdMutex::new(HashSet::new()),
            failed_idle_generations: StdMutex::new(HashMap::new()),
        }
    }

    /// Injectable for tests.
    pub fn with_generate(mut self, generate: GenerateAgentStatusFn) -> Self {
        self.generate = generate;
        self
    }

    pub fn start(self: &Arc<Self>) {
        if self.interval_running.swap(true, Ordering::SeqCst) {
            return;
        }
        let weak = Arc::downgrade(self);
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(SWEEP_INTERVAL_MS));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                let Some(summarizer) = weak.upgrade() else {
                    return;
                };
                if !summarizer.interval_running.load(Ordering::SeqCst) {
                    return;
                }
                for state in (summarizer.list_sessions)() {
                    let summarizer = Arc::clone(&summarizer);
                    tokio::spawn(async move {
                        summarizer.summarize(state).await;
                    });
                }
            }
        });
        *self.interval.lock().expect("interval poisoned") = Some(handle);
    }

    pub fn stop(self: &Arc<Self>) {
        self.interval_running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.interval.lock().expect("interval poisoned").take() {
            handle.abort();
        }
        let timers: Vec<tokio::task::JoinHandle<()>> = self
            .debounce_timers
            .lock()
            .expect("timers poisoned")
            .drain()
            .map(|(_, handle)| handle)
            .collect();
        for timer in timers {
            timer.abort();
        }
        let controllers: Vec<CancellationToken> = self
            .in_flight
            .lock()
            .expect("in flight poisoned")
            .drain()
            .map(|(_, controller)| controller)
            .collect();
        for controller in controllers {
            controller.cancel();
        }
        self.rerun_requested.lock().expect("rerun poisoned").clear();
        self.failed_idle_generations
            .lock()
            .expect("failed generations poisoned")
            .clear();
    }

    /// Drop any pending work for a session that is closing.
    pub fn forget(&self, active_session_id: &str) {
        if let Some(timer) = self
            .debounce_timers
            .lock()
            .expect("timers poisoned")
            .remove(active_session_id)
        {
            timer.abort();
        }
        if let Some(controller) = self
            .in_flight
            .lock()
            .expect("in flight poisoned")
            .get(active_session_id)
        {
            controller.cancel();
        }
        self.rerun_requested
            .lock()
            .expect("rerun poisoned")
            .remove(active_session_id);
        self.failed_idle_generations
            .lock()
            .expect("failed generations poisoned")
            .remove(active_session_id);
    }

    /// Seed in-memory status from the persisted entry when a session is added.
    pub fn seed(&self, state: &Arc<StdMutex<ActiveSessionState>>) {
        let mut state = state.lock().expect("state poisoned");
        if state.summary_state.is_some() {
            return;
        }
        if let Some(persisted) = state.runtime.session.latest_agent_status.clone() {
            state.summary_state = Some(persisted);
        }
    }

    /// Called when a session finishes a turn; debounce until the agent settles.
    pub fn notify_activity(self: &Arc<Self>, state: Arc<StdMutex<ActiveSessionState>>) {
        let id = state.lock().expect("state poisoned").active_session_id.clone();
        if let Some(existing) = self
            .debounce_timers
            .lock()
            .expect("timers poisoned")
            .remove(&id)
        {
            existing.abort();
        }
        let weak = Arc::downgrade(self);
        let timer_id = id.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(SETTLE_DEBOUNCE_MS)).await;
            let Some(summarizer) = weak.upgrade() else {
                return;
            };
            summarizer
                .debounce_timers
                .lock()
                .expect("timers poisoned")
                .remove(&timer_id);
            summarizer.summarize(state).await;
        });
        self.debounce_timers
            .lock()
            .expect("timers poisoned")
            .insert(id, handle);
    }

    async fn summarize(self: &Arc<Self>, state: Arc<StdMutex<ActiveSessionState>>) {
        let (id, session, messages, is_working, previous, message_count, streaming) = {
            let state = state.lock().expect("state poisoned");
            let id = state.active_session_id.clone();
            let session: ActiveSessionRuntimeSession = state.runtime.session.clone();
            let messages = session.messages.clone();
            let is_working = is_session_working(&state);
            let previous = state.summary_state.clone();
            let message_count = messages.len();
            let streaming = if is_working {
                session.streaming_message.clone()
            } else {
                None
            };
            (id, session, messages, is_working, previous, message_count, streaming)
        };
        if self.in_flight.lock().expect("in flight poisoned").contains_key(&id) {
            // Run once more after the current pass.
            self.rerun_requested.lock().expect("rerun poisoned").insert(id);
            return;
        }
        if messages.is_empty() {
            return;
        }
        // Idle sessions with a current verdict need no refresh; working sessions
        // always refresh so the recap keeps up with the in-progress turn.
        let content_unchanged = previous
            .as_ref()
            .is_some_and(|previous| previous.based_on_message_count == message_count);
        let owes_idle_verdict = !is_working
            && previous
                .as_ref()
                .map(|previous| previous.task_state.is_none())
                .unwrap_or(true);
        // A blank recap means the model call hasn't succeeded yet; keep retrying.
        let owes_summary = !is_working
            && previous
                .as_ref()
                .map(|previous| previous.summary.is_empty())
                .unwrap_or(true);
        if content_unchanged && !is_working && !owes_idle_verdict && !owes_summary {
            return;
        }
        // The leaf entry id is the branch-tip identity.
        let content_key = format!(
            "{}:{}",
            session.leaf_id.clone().unwrap_or_else(|| "root".to_string()),
            message_count
        );
        let failed = self
            .failed_idle_generations
            .lock()
            .expect("failed generations poisoned")
            .get(&id)
            .cloned();
        if !is_working {
            if let Some(failed) = &failed {
                if failed.content_key == content_key
                    && failed.attempts >= IDLE_GENERATION_ATTEMPT_LIMIT
                    && now_millis() as f64 - failed.last_failure_at < IDLE_GENERATION_RETRY_BACKOFF_MS as f64
                {
                    return;
                }
            }
        }
        // Include the in-progress message so a long streaming turn gets a live recap.
        let mut context_messages = messages.clone();
        if let Some(streaming) = streaming.clone() {
            context_messages.push(streaming);
        }

        let controller = CancellationToken::new();
        self.in_flight
            .lock()
            .expect("in flight poisoned")
            .insert(id.clone(), controller.clone());
        let generated = {
            let (registry, retry_policy) = {
                let state = state.lock().expect("state poisoned");
                (
                    state.runtime.session.model_registry.clone(),
                    state
                        .runtime
                        .session
                        .settings_manager
                        .as_ref()
                        .map(|manager| provider_retry_policy(manager)),
                )
            };
            match registry {
                Some(registry) => {
                    (self.generate)(GenerateAgentStatusParams {
                        registry,
                        messages: context_messages,
                        is_working,
                        retry_policy,
                        signal: Some(controller.clone()),
                    })
                    .await
                }
                None => None,
            }
        };
        if generated.is_some() {
            self.failed_idle_generations
                .lock()
                .expect("failed generations poisoned")
                .remove(&id);
        } else if !is_working && !controller.is_cancelled() {
            // The aborted check keeps a racing forget() from repopulating the map.
            let attempts = failed
                .as_ref()
                .filter(|failed| failed.content_key == content_key)
                .map(|failed| failed.attempts + 1)
                .unwrap_or(1);
            self.failed_idle_generations
                .lock()
                .expect("failed generations poisoned")
                .insert(
                    id.clone(),
                    FailedIdleGeneration {
                        content_key: content_key.clone(),
                        attempts,
                        last_failure_at: now_millis() as f64,
                    },
                );
        }
        // A failed classification on an idle session would spin at "working"
        // forever, so settle it to needs_input.
        let result = generated.clone().or_else(|| {
            if !is_working && (owes_idle_verdict || owes_summary) {
                Some(AgentStatusResult {
                    summary: previous
                        .as_ref()
                        .map(|previous| previous.summary.clone())
                        .unwrap_or_default(),
                    task_state: Some("needs_input".to_string()),
                })
            } else {
                None
            }
        });
        let Some(result) = result else {
            self.finish_pass(&id, state);
            return;
        };
        // Discard if the session closed, was swapped, or moved to a new turn.
        let discard = {
            let state_guard = state.lock().expect("state poisoned");
            let current_session = state_guard.runtime.session.clone();
            controller.is_cancelled()
                || session_identity(&current_session) != session_identity(&session)
                || is_session_working(&state_guard) != is_working
                || current_session.messages.len() != message_count
        };
        if discard {
            self.finish_pass(&id, state);
            return;
        }
        {
            let mut state = state.lock().expect("state poisoned");
            let current_session = state.runtime.session.clone();
            // A working refresh carries no verdict; keep the prior one at the same
            // message count so a still-valid needs_input isn't dropped.
            let task_state = result.task_state.clone().or_else(|| {
                if previous
                    .as_ref()
                    .is_some_and(|previous| previous.based_on_message_count == message_count)
                {
                    previous.as_ref().and_then(|previous| previous.task_state.clone())
                } else {
                    None
                }
            });
            let status = AgentStatus {
                summary: result.summary.clone(),
                task_state,
                based_on_message_count: message_count,
            };
            // An idle settle refreshes the verdict's currency, which drives the
            // roster's activity axis: it must publish even when the text is unchanged.
            let changed = previous.as_ref().map(|previous| previous.summary.clone()) != Some(status.summary.clone())
                || previous.as_ref().and_then(|previous| previous.task_state.clone()) != status.task_state
                || (!is_working
                    && previous
                        .as_ref()
                        .map(|previous| previous.based_on_message_count)
                        != Some(status.based_on_message_count));
            state.summary_state = Some(status.clone());
            // Persist only settled idle verdicts from real classifications that
            // differ from the latest persisted entry.
            if !is_working && generated.is_some() {
                let persisted = current_session.latest_agent_status.clone();
                if persisted.as_ref() != Some(&status) {
                    if let Some(writer) = current_session.agent_status_writer.clone() {
                        let _ = writer.append_agent_status(&status);
                    }
                }
            }
            if changed {
                if let Some(on_status_changed) = &self.on_status_changed {
                    on_status_changed(&state);
                }
            }
        }
        self.finish_pass(&id, state);
    }

    fn finish_pass(self: &Arc<Self>, id: &str, state: Arc<StdMutex<ActiveSessionState>>) {
        self.in_flight.lock().expect("in flight poisoned").remove(id);
        // Re-debounce a request that arrived mid-pass instead of dropping it.
        if self.rerun_requested.lock().expect("rerun poisoned").remove(id) {
            self.notify_activity(state);
        }
    }
}

#[allow(dead_code)]
fn unused_runtime_marker(_runtime: AgentSessionRuntime) {}

#[allow(dead_code)]
fn unused_custom_content_marker(_content: CustomMessageContent) {}

#[cfg(test)]
mod retry_safety_tests {
    use super::*;
    #[tokio::test]
    async fn safety_filtered_status_is_not_retried() {
        let attempts = std::sync::atomic::AtomicUsize::new(0);
        let call = || -> CompleteFuture {
            attempts.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { AssistantMessage {
                stop_reason: "error".into(), error_message: Some("Provider safety filter".into()),
                diagnostics: Some(vec![pi_ai::utils::diagnostics::AssistantMessageDiagnostic {
                    type_: "provider_stream_failure".into(), timestamp: 0, error: None,
                    details: Some(serde_json::json!({"kind":"safety"}).as_object().unwrap().clone()),
                }]), ..Default::default()
            } })
        };
        let result = complete_with_provider_retry(&call,
            ProviderRetryPolicy { base_delay_ms: 0.0, ..DEFAULT_PROVIDER_RETRY_POLICY }, None).await;
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(result.error_message.as_deref(), Some("Provider safety filter"));
    }
}
