//! Port of packages/coding-agent/src/core/compaction/compaction.ts
//!
//! Context compaction for long sessions.
//!
//! Pure functions for compaction logic. The session manager handles I/O,
//! and after compaction the session is reloaded.
//!
//! Type mapping notes:
//! - `Model<Api>` -> `pi_ai::types::Model`.
//! - `AbortSignal` -> `tokio_util::sync::CancellationToken`.
//! - `SummaryCallRunner` is a generic higher-order function in TypeScript; Rust
//!   keeps the same shape with a `&dyn Fn` wrapper that receives a per-call header
//!   map and returns a boxed future.
//! - `SessionEntry` lives in `core/session-manager.ts` (another slice), so the
//!   entry shapes this module reads are declared here as `CompactionSessionEntry`
//!   and recorded in blocked_on.

use std::collections::BTreeSet;
use std::sync::Arc;

use pi_agent_core::types::{AgentMessage, CustomAgentMessage, CustomMessageContent, ThinkingLevel};
use pi_agent_core::performance_metrics::{PerformanceMetricOperation, PerformanceMetricOutcome, PerformanceMetricMeasurement};
use pi_ai::compaction::{
    compaction_matches_model, is_compaction_checkpoint, CompactionOptions,
    ProviderCompactionCheckpoint,
};
use pi_ai::models::get_model_input_limit;
use pi_ai::stream::{compact_simple, complete_simple, supports_compaction};
use pi_ai::types::{
    Api, AssistantMessage, Context, Message, Model, StopReason, Usage,
    STOP_REASON_ABORTED, STOP_REASON_ERROR,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::Value;

use crate::core::compaction::checkpoint::has_provider_checkpoint;
use crate::core::compaction::metrics::CompactionMetrics;
use crate::core::compaction::utils::{
    compute_file_lists, create_file_ops, extract_file_ops_from_message, format_file_operations,
    serialize_conversation, FileOperations, SUMMARIZATION_SYSTEM_PROMPT,
};
use crate::core::messages::{
    branch_summary_to_agent_message, compaction_summary_to_agent_message, convert_to_llm,
    create_branch_summary_message, create_compaction_summary_message, create_custom_message,
    custom_message_to_agent_message, HARNESS_DIGEST_CUSTOM_TYPE,
};
use crate::core::usage::{add_assistant_usage, empty_usage};

/// Details stored in CompactionEntry.details for file tracking
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionDetails {
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_checkpoint: Option<ProviderCompactionCheckpoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SummarySlice {
    pub summary: String,
    pub usage: Option<Usage>,
}

/// Entry shapes read by this module.
///
/// blocked_on: needs core::session_manager::{SessionEntry, CompactionEntry,
/// BranchSummaryEntry, CustomMessageEntry}.
#[derive(Debug, Clone, PartialEq)]
pub enum CompactionSessionEntry {
    Message {
        id: String,
        parent_id: Option<String>,
        message: AgentMessage,
    },
    CustomMessage {
        id: String,
        parent_id: Option<String>,
        custom_type: String,
        content: CustomMessageContent,
        details: Option<Value>,
        display: bool,
        timestamp: String,
    },
    BranchSummary {
        id: String,
        parent_id: Option<String>,
        from_id: String,
        summary: String,
        details: Option<Value>,
        from_hook: Option<bool>,
        timestamp: String,
    },
    Compaction {
        id: String,
        parent_id: Option<String>,
        summary: String,
        first_kept_entry_id: String,
        tokens_before: f64,
        details: Option<Value>,
        from_hook: Option<bool>,
        custom_instructions: Option<String>,
        harness_digest: Option<String>,
        timestamp: String,
    },
    Other {
        id: String,
        parent_id: Option<String>,
        entry_type: String,
    },
}

impl CompactionSessionEntry {
    pub fn id(&self) -> &str {
        match self {
            CompactionSessionEntry::Message { id, .. }
            | CompactionSessionEntry::CustomMessage { id, .. }
            | CompactionSessionEntry::BranchSummary { id, .. }
            | CompactionSessionEntry::Compaction { id, .. }
            | CompactionSessionEntry::Other { id, .. } => id,
        }
    }

    pub fn parent_id(&self) -> Option<&str> {
        match self {
            CompactionSessionEntry::Message { parent_id, .. }
            | CompactionSessionEntry::CustomMessage { parent_id, .. }
            | CompactionSessionEntry::BranchSummary { parent_id, .. }
            | CompactionSessionEntry::Compaction { parent_id, .. }
            | CompactionSessionEntry::Other { parent_id, .. } => parent_id.as_deref(),
        }
    }

    pub fn entry_type(&self) -> &str {
        match self {
            CompactionSessionEntry::Message { .. } => "message",
            CompactionSessionEntry::CustomMessage { .. } => "custom_message",
            CompactionSessionEntry::BranchSummary { .. } => "branch_summary",
            CompactionSessionEntry::Compaction { .. } => "compaction",
            CompactionSessionEntry::Other { entry_type, .. } => entry_type,
        }
    }
}

/// `buildSessionContext(entries).messages`.
pub type SessionContextBuilder<'a> = &'a dyn Fn(&[CompactionSessionEntry]) -> Vec<AgentMessage>;

// ---------------------------------------------------------------------------
// Retry policy surface used by the summary call sites
// ---------------------------------------------------------------------------

/// Mirrors `ProviderRetryPolicy` from core/provider-retry.ts.
///
/// blocked_on: needs core::provider_retry::{ProviderRetryPolicy,
/// complete_with_provider_retry, request_with_provider_retry}.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderRetryPolicy {
    pub enabled: bool,
    pub max_retries: u32,
    pub base_delay_ms: f64,
    /// Max server-requested retry delay before giving up; 0 disables the cap.
    pub max_retry_delay_ms: f64,
}

pub const DEFAULT_PROVIDER_RETRY_POLICY: ProviderRetryPolicy = ProviderRetryPolicy {
    enabled: true,
    max_retries: 3,
    base_delay_ms: DEFAULT_PROVIDER_RETRY_BASE_DELAY_MS,
    max_retry_delay_ms: 60000.0,
};

/// Node caps timers at 2^31-1 ms; longer delays overflow setTimeout and fire after ~1ms.
const MAX_TIMER_DELAY_MS: f64 = 2_147_483_647.0;
const DEFAULT_PROVIDER_RETRY_BASE_DELAY_MS: f64 = 2_000.0;
pub const PROVIDER_RETRY_JITTER_RATIO: f64 = 0.2;

/// Unary provider request failure (`Error & { status?, retryAfterMs? }`).
#[derive(Debug, Clone, Default, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct ProviderRequestError {
    pub message: String,
    pub status: Option<i64>,
    pub retry_after_ms: Option<f64>,
    /// `error instanceof TypeError`.
    pub is_type_error: bool,
}

/// `{kind: "wait"} | {kind: "exceeds-cap"}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProviderRetryDelay {
    Wait(f64),
    ExceedsCap(f64),
}

fn bounded_random() -> f64 {
    let value: f64 = rand::random();
    if !value.is_finite() {
        return 0.5;
    }
    value.clamp(0.0, 1.0)
}

/// Delay before retry `attempt` (1-based), honoring a server-requested not-before time.
fn provider_retry_delay(
    attempt: u32,
    retry_after_ms: Option<f64>,
    policy: &ProviderRetryPolicy,
) -> ProviderRetryDelay {
    if let Some(retry_after_ms) = retry_after_ms {
        if policy.max_retry_delay_ms > 0.0 && retry_after_ms > policy.max_retry_delay_ms {
            return ProviderRetryDelay::ExceedsCap(retry_after_ms);
        }
        if retry_after_ms > MAX_TIMER_DELAY_MS {
            return ProviderRetryDelay::ExceedsCap(retry_after_ms);
        }
    }
    let random = bounded_random();
    let jitter_ratio = PROVIDER_RETRY_JITTER_RATIO;
    let base_delay_ms = if policy.base_delay_ms.is_finite() {
        policy.base_delay_ms.max(0.0)
    } else {
        DEFAULT_PROVIDER_RETRY_BASE_DELAY_MS
    };
    let exponent = attempt.saturating_sub(1) as i32;
    let exponential = (base_delay_ms * 2f64.powi(exponent)).min(MAX_TIMER_DELAY_MS);
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
    ProviderRetryDelay::Wait(delay_ms.min(MAX_TIMER_DELAY_MS))
}

fn is_agent_lifecycle_failure(message: &AssistantMessage) -> bool {
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

fn is_faux_provider_queue_exhausted(message: &AssistantMessage) -> bool {
    message.provider == "faux"
        && message.error_message.as_deref() == Some("No more faux responses queued")
}

fn provider_stream_failure_details(
    message: &AssistantMessage,
) -> Option<serde_json::Map<String, Value>> {
    let failure = message
        .diagnostics
        .as_ref()?
        .iter()
        .find(|diagnostic| diagnostic.type_ == "provider_stream_failure")?;
    failure.details.clone()
}

fn provider_stream_failure_kind(message: &AssistantMessage) -> Option<String> {
    provider_stream_failure_details(message)?
        .get("kind")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn provider_stream_failure_retry_after_ms(message: &AssistantMessage) -> Option<f64> {
    let value = provider_stream_failure_details(message)?
        .get("retryAfterMs")
        .and_then(Value::as_f64)?;
    if value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

/// Deterministic rejections never retry; auth gets one retry before it can be marked stale.
fn is_permanent_provider_failure_kind(kind: Option<&str>, retries_performed: u32) -> bool {
    match kind {
        Some("invalid_request") | Some("refusal") | Some("permission") => true,
        Some("auth") => retries_performed > 0,
        _ => false,
    }
}

/// `completeWithProviderRetry` for the summary call sites.
async fn complete_with_provider_retry(
    // `Send + Sync` so awaiting the attempt inside the caller's `Send` future keeps
    // that future `Send`.
    attempt_completion: &(dyn Fn() -> pi_ai::types::BoxFuture<Result<AssistantMessage, String>>
              + Send
              + Sync),
    policy: Option<&ProviderRetryPolicy>,
    signal: Option<&tokio_util::sync::CancellationToken>,
) -> Result<AssistantMessage, String> {
    let default_policy = DEFAULT_PROVIDER_RETRY_POLICY;
    let policy = policy.unwrap_or(&default_policy);
    let max_retries = if policy.enabled {
        policy.max_retries
    } else {
        0
    };
    let mut retries_performed = 0u32;
    loop {
        let message = attempt_completion().await?;
        if message.stop_reason != STOP_REASON_ERROR {
            return Ok(message);
        }
        if signal.map(|signal| signal.is_cancelled()).unwrap_or(false) {
            // A cancel that raced the failure is an abort, not a provider failure.
            let mut aborted = message;
            aborted.stop_reason = STOP_REASON_ABORTED.to_string();
            return Ok(aborted);
        }
        if retries_performed >= max_retries
            || is_agent_lifecycle_failure(&message)
            || is_faux_provider_queue_exhausted(&message)
            || crate::core::provider_retry::cannot_replay_provider_failure(&message)
        {
            return Ok(message);
        }
        let kind = provider_stream_failure_kind(&message);
        if is_permanent_provider_failure_kind(kind.as_deref(), retries_performed) {
            return Ok(message);
        }
        let delay = provider_retry_delay(
            retries_performed + 1,
            provider_stream_failure_retry_after_ms(&message),
            policy,
        );
        let delay_ms = match delay {
            ProviderRetryDelay::ExceedsCap(_) => return Ok(message),
            ProviderRetryDelay::Wait(delay_ms) => delay_ms,
        };
        if crate::utils::sleep::sleep(delay_ms.max(0.0) as u64, signal)
            .await
            .is_err()
        {
            let mut aborted = message;
            aborted.stop_reason = STOP_REASON_ABORTED.to_string();
            return Ok(aborted);
        }
        if signal.map(|signal| signal.is_cancelled()).unwrap_or(false) {
            let mut aborted = message;
            aborted.stop_reason = STOP_REASON_ABORTED.to_string();
            return Ok(aborted);
        }
        retries_performed += 1;
    }
}

fn error_is_transient(error: &ProviderRequestError) -> bool {
    let status_transient = error
        .status
        .map(|status| status == 408 || status == 429 || status >= 500)
        .unwrap_or(false);
    let network_like = error.is_type_error && {
        let message = error.message.to_lowercase();
        message.contains("fetch") || message.contains("network") || message.contains("socket")
    };
    status_transient || network_like
}

/// `requestWithProviderRetry` for the unary provider compaction request.
async fn request_with_provider_retry(
    // `Send + Sync` so awaiting the attempt inside the caller's `Send` future keeps
    // that future `Send` (same reasoning as `complete_with_provider_retry` above).
    attempt_request: &(dyn Fn() -> pi_ai::types::BoxFuture<
        Result<Option<pi_ai::compaction::ProviderCompactionResult>, ProviderRequestError>,
    > + Send
              + Sync),
    policy: Option<&ProviderRetryPolicy>,
    signal: Option<&tokio_util::sync::CancellationToken>,
) -> Result<Option<pi_ai::compaction::ProviderCompactionResult>, ProviderRequestError> {
    let default_policy = DEFAULT_PROVIDER_RETRY_POLICY;
    let policy = policy.unwrap_or(&default_policy);
    let max_retries = if policy.enabled {
        policy.max_retries
    } else {
        0
    };
    let mut attempt = 0u32;
    loop {
        if signal.map(|signal| signal.is_cancelled()).unwrap_or(false) {
            return Err(provider_request_error("Aborted".to_string()));
        }
        match attempt_request().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if signal.map(|signal| signal.is_cancelled()).unwrap_or(false) {
                    return Err(error);
                }
                if !error_is_transient(&error) || attempt >= max_retries {
                    return Err(error);
                }
                let delay = provider_retry_delay(attempt + 1, error.retry_after_ms, policy);
                let delay_ms = match delay {
                    ProviderRetryDelay::ExceedsCap(_) => return Err(error),
                    ProviderRetryDelay::Wait(delay_ms) => delay_ms,
                };
                if crate::utils::sleep::sleep(delay_ms.max(0.0) as u64, signal)
                    .await
                    .is_err()
                {
                    return Err(error);
                }
                attempt += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prompts and settings
// ---------------------------------------------------------------------------

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.\n\nUse this EXACT format:\n\n## Goal\n[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned by user]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered list of what should happen next]\n\n## Critical Context\n- [Any data, examples, or references needed to continue]\n- [Or \"(none)\" if not applicable]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.\n\nUpdate the existing structured summary with new information. RULES:\n- PRESERVE all existing information from the previous summary\n- ADD new progress, decisions, and context from the new messages\n- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed\n- UPDATE \"Next Steps\" based on what was accomplished\n- PRESERVE exact file paths, function names, and error messages\n- If something is no longer relevant, you may remove it\n\nUse this EXACT format:\n\n## Goal\n[Preserve existing goals, add new ones if the task expanded]\n\n## Constraints & Preferences\n- [Preserve existing, add new ones discovered]\n\n## Progress\n### Done\n- [x] [Include previously done items AND newly completed items]\n\n### In Progress\n- [ ] [Current work - update based on progress]\n\n### Blocked\n- [Current blockers - remove if resolved]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n## Next Steps\n1. [Update based on current state]\n\n## Critical Context\n- [Preserve important context, add new if needed]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const CONSOLIDATING_UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are ONLY the NEW conversation material that moved out of the retained tail. Incorporate it into the existing summary in <previous-summary> tags in this same response. Do not make another merge pass.\n\nUpdate the structured summary using these rules:\n- Preserve every enduring user constraint, preference, correction, refinement, safety rule, and requested acceptance gate. Obsolete progress is not evidence that an enduring constraint expired.\n- Preserve decisions together with their reasons and provenance. Do not merge distinct decisions merely because their wording is similar.\n- Preserve unresolved work and blockers. Mark a blocker resolved or replace an old status only when the new messages demonstrate the superseding fact; record the resolution or replacement so its history remains understandable.\n- Preserve exact file and artifact anchors, including paths, IDs, hashes, sizes, commands, function names, critical errors, and evidence locations needed to inspect durable source material.\n- Preserve kernel-state and continual-harness facts, including useful Python names and persistence/reload warnings.\n- Preserve tool-call/result relationships needed to understand actions. Never treat an opaque provider checkpoint as ordinary text-summary material.\n- Consolidate demonstrably repeated statements into one complete statement. Do not accumulate another bullet for the same unchanged fact. When uncertain whether facts are duplicates or superseded, retain both and state the uncertainty.\n- Add every genuinely new fact. The summary may grow when new information exists. Do not apply a character/token cap, tail truncation, or arbitrary deletion to make it short.\n- Keep exact user wording when the prior summary or new messages identify it as verbatim or critical.\n\nUse this EXACT format:\n\n## Goal\n[Preserve existing goals and add genuinely new goals]\n\n## Constraints & Preferences\n- [Consolidated enduring constraints and preferences]\n- [Or \"(none)\" only if neither source contains any]\n\n## Progress\n### Done\n- [x] [Previously and newly completed work, without duplicate bullets]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Current unresolved blockers, plus resolution provenance for removed blockers when important]\n\n## Key Decisions\n- **[Decision]**: [Reason and provenance]\n\n## Next Steps\n1. [Current ordered steps]\n\n## Critical Context\n- [Exact anchors, errors, kernel/harness facts, and other facts needed to continue]\n- [Or \"(none)\" only if neither source contains any]\n\nBe concise only by removing demonstrable repetition and replacing demonstrably superseded status. Never omit a unique required fact.";

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.\n\nSummarize the prefix to provide context for the retained suffix:\n\n## Original Request\n[What did the user ask for in this turn?]\n\n## Early Progress\n- [Key decisions and work done in the prefix]\n\n## Context for Suffix\n- [Information needed to understand the retained recent work]\n\nBe concise. Focus on what's needed to understand the kept suffix.";

const KERNEL_PERSIST_SUMMARY_NOTE: &str = "Note: the Python kernel keeps running after this summary — every Python variable, import, and helper you defined stays available. The cells that defined them won't appear above, so record in the summary any names worth remembering so you reuse them instead of redefining them.";

/** Details stored in CompactionEntry.details for file tracking */
pub fn build_summarization_prompt(
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    summary_update_policy: &SummaryUpdatePolicy,
) -> String {
    let mut base_prompt = match previous_summary {
        Some(_) => {
            if summary_update_policy == CONSOLIDATE_REPEATED_SUMMARY_POLICY {
                CONSOLIDATING_UPDATE_SUMMARIZATION_PROMPT
            } else {
                UPDATE_SUMMARIZATION_PROMPT
            }
        }
        None => SUMMARIZATION_PROMPT,
    }
    .to_string();
    if let Some(custom_instructions) = custom_instructions {
        base_prompt += &format!("\n\n<user-instructions>\nThe user provided these instructions for this summary. Follow them with high priority while keeping the section format above: emphasize what they ask to focus on, and preserve verbatim anything they ask to remember.\n{custom_instructions}\n</user-instructions>");
    }
    format!("{base_prompt}\n\n{KERNEL_PERSIST_SUMMARY_NOTE}")
}

/// Extract file operations from messages and previous compaction entries.
/// Preserve file operations recorded by prior compactions and current tool calls.
fn extract_file_operations(
    messages: &[AgentMessage],
    entries: &[CompactionSessionEntry],
    prev_compaction_index: i64,
) -> FileOperations {
    let mut file_ops = create_file_ops();
    if prev_compaction_index >= 0 {
        if let Some(CompactionSessionEntry::Compaction {
            details, from_hook, ..
        }) = entries.get(prev_compaction_index as usize)
        {
            if from_hook != &Some(true) {
                if let Some(details) = details {
                    // fromHook field kept for session file compatibility
                    if let Some(read_files) = details.get("readFiles").and_then(Value::as_array) {
                        for entry in read_files {
                            if let Some(path) = entry.as_str() {
                                file_ops.read.insert(path.to_string());
                            }
                        }
                    }
                    if let Some(modified_files) =
                        details.get("modifiedFiles").and_then(Value::as_array)
                    {
                        for entry in modified_files {
                            if let Some(path) = entry.as_str() {
                                file_ops.edited.insert(path.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    for message in messages {
        extract_file_ops_from_message(message, &mut file_ops);
    }
    file_ops
}

/// Extract AgentMessage from an entry if it produces one.
/// Returns None for entries that don't contribute to LLM context.
fn get_message_from_entry(entry: &CompactionSessionEntry) -> Option<AgentMessage> {
    match entry {
        CompactionSessionEntry::Message { message, .. } => Some(message.clone()),
        CompactionSessionEntry::CustomMessage {
            custom_type,
            content,
            display,
            details,
            timestamp,
            ..
        } => Some(custom_message_to_agent_message(create_custom_message(
            custom_type.clone(),
            content.clone(),
            *display,
            details.clone(),
            timestamp,
        ))),
        CompactionSessionEntry::BranchSummary {
            summary,
            from_id,
            timestamp,
            ..
        } => Some(branch_summary_to_agent_message(
            create_branch_summary_message(summary.clone(), from_id.clone(), timestamp),
        )),
        CompactionSessionEntry::Compaction {
            summary,
            tokens_before,
            custom_instructions,
            timestamp,
            ..
        } => Some(compaction_summary_to_agent_message(
            create_compaction_summary_message(
                summary.clone(),
                *tokens_before,
                timestamp,
                custom_instructions.clone(),
                None,
                None,
                None,
            ),
        )),
        CompactionSessionEntry::Other { .. } => None,
    }
}

fn get_message_from_entry_for_compaction(entry: &CompactionSessionEntry) -> Option<AgentMessage> {
    if matches!(entry, CompactionSessionEntry::Compaction { .. }) {
        return None;
    }
    // Harness digests are regenerated on the new compaction head; never summarizer input.
    if let CompactionSessionEntry::CustomMessage { custom_type, .. } = entry {
        if custom_type == HARNESS_DIGEST_CUSTOM_TYPE {
            return None;
        }
    }
    get_message_from_entry(entry)
}

/// Result from compact() - SessionManager adds uuid/parentUuid when saving
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionResult {
    pub summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: f64,
    /// Extension-specific data (e.g., ArtifactIndex, version markers for structured compaction)
    pub details: Option<CompactionDetails>,
    /// What the summarization call(s) billed; persisted on the compaction entry.
    pub usage: Option<Usage>,
}

pub const COMPACT_SKILL_NAME: &str = "compact";

/// Automatic compaction policy; this does not change the model's advertised window.
pub const MAX_COMPACTION_CONTEXT_TOKENS: f64 = 250_000.0;

pub const SUMMARY_UPDATE_POLICY_ENV: &str = "PRIME_AGENT_SUMMARY_UPDATE_POLICY";
pub const CONSOLIDATE_REPEATED_SUMMARY_POLICY: &str = "consolidate-repeated-v1";
pub const SUMMARY_UPDATE_POLICY_OFF: &str = "off";

pub type SummaryUpdatePolicy = String;

pub fn resolve_summary_update_policy(
    configured: Option<&str>,
    environment: Option<&str>,
) -> SummaryUpdatePolicy {
    if configured == Some(CONSOLIDATE_REPEATED_SUMMARY_POLICY)
        || configured == Some(SUMMARY_UPDATE_POLICY_OFF)
    {
        return configured.unwrap_or(SUMMARY_UPDATE_POLICY_OFF).to_string();
    }
    let environment = environment
        .map(str::to_string)
        .or_else(|| std::env::var(SUMMARY_UPDATE_POLICY_ENV).ok());
    if environment.as_deref() == Some(CONSOLIDATE_REPEATED_SUMMARY_POLICY) {
        CONSOLIDATE_REPEATED_SUMMARY_POLICY.to_string()
    } else {
        SUMMARY_UPDATE_POLICY_OFF.to_string()
    }
}

fn trim_trailing_slashes(value: &str) -> &str {
    value.trim_end_matches('/')
}

fn is_staged_azure_native_compaction_model(model: &Model) -> bool {
    let Some(capability) = model.native_compaction.as_ref() else {
        return false;
    };
    model.provider == "azure-openai-managed"
        && model.id == "gpt-6-astra"
        && model.api == "openai-responses"
        && capability.provider == model.provider
        && capability.model == model.id
        && capability.protocol == "openai-responses-compact-v1"
        && capability.api_version == "v1"
        && capability.enabled
        && capability.validation == pi_ai::types::NativeCompactionValidation::LiveVerified
        && trim_trailing_slashes(&capability.endpoint)
            == format!(
                "{}/responses/compact",
                trim_trailing_slashes(&model.base_url)
            )
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompactionSettings {
    pub enabled: bool,
    pub reserve_tokens: f64,
    pub keep_recent_tokens: f64,
    /// Opt-in iterative-summary consolidation. Default off until semantic quality is proven.
    pub summary_update_policy: Option<SummaryUpdatePolicy>,
}

pub fn default_compaction_settings() -> CompactionSettings {
    CompactionSettings {
        enabled: true,
        reserve_tokens: 16384.0,
        keep_recent_tokens: 20000.0,
        summary_update_policy: Some(SUMMARY_UPDATE_POLICY_OFF.to_string()),
    }
}

/// Calculate total context tokens from usage.
/// Uses the native totalTokens field when available, falls back to computing from components.
///
/// Includes output: the assistant's response becomes part of the prompt on the next
/// request, so it counts toward the context the next turn will send.
pub fn calculate_context_tokens(usage: &Usage) -> f64 {
    if usage.total_tokens != 0.0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

/// Get usage from an assistant message if available.
/// Skips aborted and error messages as they don't have valid usage data.
fn get_assistant_usage(msg: &AgentMessage) -> Option<Usage> {
    let AgentMessage::Message(Message::Assistant(assistant)) = msg else {
        return None;
    };
    if assistant.stop_reason != STOP_REASON_ABORTED && assistant.stop_reason != STOP_REASON_ERROR {
        return Some(assistant.usage.clone());
    }
    None
}

/// Find the last non-aborted assistant message usage from session entries.
pub fn get_last_assistant_usage(entries: &[CompactionSessionEntry]) -> Option<Usage> {
    for entry in entries.iter().rev() {
        if let CompactionSessionEntry::Message { message, .. } = entry {
            if let Some(usage) = get_assistant_usage(message) {
                return Some(usage);
            }
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextUsageEstimate {
    pub tokens: f64,
    pub usage_tokens: f64,
    pub trailing_tokens: f64,
    /// `null` when no assistant usage was found.
    pub last_usage_index: Option<usize>,
}

fn get_last_assistant_usage_info(messages: &[AgentMessage]) -> Option<(Usage, usize)> {
    for (index, message) in messages.iter().enumerate().rev() {
        if let Some(usage) = get_assistant_usage(message) {
            return Some((usage, index));
        }
    }
    None
}

/// Estimate context tokens from messages, using the last assistant usage when available.
/// If there are messages after the last usage, estimate their tokens with estimateTokens.
pub fn estimate_context_tokens(messages: &[AgentMessage]) -> ContextUsageEstimate {
    let Some((usage, index)) = get_last_assistant_usage_info(messages) else {
        let mut estimated = 0.0;
        for message in messages {
            estimated += estimate_tokens(message);
        }
        return ContextUsageEstimate {
            tokens: estimated,
            usage_tokens: 0.0,
            trailing_tokens: estimated,
            last_usage_index: None,
        };
    };

    let usage_tokens = calculate_context_tokens(&usage);
    let mut trailing_tokens = 0.0;
    for message in messages.iter().skip(index + 1) {
        trailing_tokens += estimate_tokens(message);
    }

    ContextUsageEstimate {
        tokens: usage_tokens + trailing_tokens,
        usage_tokens,
        trailing_tokens,
        last_usage_index: Some(index),
    }
}

/// Check if compaction should trigger based on context usage.
pub fn should_compact(
    context_tokens: f64,
    context_window: f64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled {
        return false;
    }
    if context_window <= 0.0 {
        return false;
    }
    context_tokens
        >= f64::min(
            MAX_COMPACTION_CONTEXT_TOKENS,
            context_window - settings.reserve_tokens,
        )
}

pub fn should_compact_for_model(
    context_tokens: f64,
    model: &Model,
    settings: &CompactionSettings,
) -> bool {
    let input_limit = get_model_input_limit(model);
    should_compact(context_tokens, input_limit, settings)
}

/// `Math.ceil(chars / 4)` with JavaScript's number semantics.
fn ceil_div4(chars: usize) -> f64 {
    (chars as f64 / 4.0).ceil()
}

fn content_chars(content: &CustomMessageContent) -> usize {
    match content {
        CustomMessageContent::Text(text) => text.chars().count(),
        CustomMessageContent::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                pi_agent_core::types::ContentBlock::Text(text) => text.text.chars().count(),
                pi_agent_core::types::ContentBlock::Image(_) => 4800,
            })
            .sum(),
    }
}

/// Estimate token count for a message using chars/4 heuristic.
/// This is conservative (overestimates tokens).
pub fn estimate_tokens(message: &AgentMessage) -> f64 {
    match message {
        AgentMessage::Message(Message::User(user)) => {
            if let Some(provider_context) = &user.provider_context {
                return provider_context.estimated_tokens;
            }
            let chars = match &user.content {
                pi_ai::types::UserContent::Text(text) => text.chars().count(),
                pi_ai::types::UserContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|block| match block {
                        pi_ai::types::ImageOrTextContent::Text(text) => {
                            Some(text.text.chars().count())
                        }
                        pi_ai::types::ImageOrTextContent::Image(_) => None,
                    })
                    .sum(),
            };
            ceil_div4(chars)
        }
        AgentMessage::Message(Message::Assistant(assistant)) => {
            let mut chars = 0usize;
            for block in &assistant.content {
                match block {
                    pi_ai::types::ContentBlock::Text(text) => chars += text.text.chars().count(),
                    pi_ai::types::ContentBlock::Thinking(thinking) => {
                        chars += thinking.thinking.chars().count()
                    }
                    pi_ai::types::ContentBlock::ToolCall(tool_call) => {
                        chars += tool_call.name.chars().count();
                        chars += serde_json::to_string(&tool_call.arguments)
                            .map(|json| json.chars().count())
                            .unwrap_or(0);
                    }
                }
            }
            ceil_div4(chars)
        }
        AgentMessage::Message(Message::ToolResult(tool_result)) => {
            let mut chars = 0usize;
            for block in &tool_result.content {
                match block {
                    pi_ai::types::ImageOrTextContent::Text(text) => {
                        chars += text.text.chars().count()
                    }
                    pi_ai::types::ImageOrTextContent::Image(_) => chars += 4800,
                }
            }
            ceil_div4(chars)
        }
        AgentMessage::Custom(CustomAgentMessage::BashExecution {
            command, output, ..
        }) => ceil_div4(command.chars().count() + output.chars().count()),
        AgentMessage::Custom(CustomAgentMessage::Custom { content, .. }) => {
            ceil_div4(content_chars(content))
        }
        AgentMessage::Custom(CustomAgentMessage::BranchSummary { summary, .. }) => {
            ceil_div4(summary.chars().count())
        }
        AgentMessage::Custom(CustomAgentMessage::CompactionSummary {
            summary,
            provider_context,
            harness_digest,
            ..
        }) => {
            if let Some(provider_context) = provider_context {
                let digest_tokens = harness_digest
                    .as_ref()
                    .map(|digest| ceil_div4(digest.chars().count()))
                    .unwrap_or(0.0);
                return provider_context.estimated_tokens + digest_tokens;
            }
            let mut chars = summary.chars().count();
            if let Some(digest) = harness_digest {
                chars += digest.chars().count();
            }
            ceil_div4(chars)
        }
    }
}

/// Find valid cut points: indices of user, assistant, custom, or bashExecution messages.
/// Never cut at tool results (they must follow their tool call).
/// When we cut at an assistant message with tool calls, its tool results follow it
/// and will be kept.
/// BashExecutionMessage is treated like a user message (user-initiated context).
fn find_valid_cut_points(
    entries: &[CompactionSessionEntry],
    start_index: usize,
    end_index: usize,
) -> Vec<usize> {
    let mut cut_points: Vec<usize> = Vec::new();
    for index in start_index..end_index {
        let entry = &entries[index];
        match entry {
            CompactionSessionEntry::Message { message, .. } => match message.role() {
                "bashExecution" | "custom" | "branchSummary" | "compactionSummary" | "user"
                | "assistant" => cut_points.push(index),
                "toolResult" => {}
                _ => {}
            },
            _ => {}
        }
        // Branch summaries and custom messages are user-role turn boundaries.
        if matches!(
            entry,
            CompactionSessionEntry::BranchSummary { .. }
                | CompactionSessionEntry::CustomMessage { .. }
        ) {
            cut_points.push(index);
        }
    }
    cut_points
}

/// Find the user message (or bashExecution) that starts the turn containing the given entry index.
/// Returns None if no turn start found before the index.
/// BashExecutionMessage is treated like a user message for turn boundaries.
pub fn find_turn_start_index(
    entries: &[CompactionSessionEntry],
    entry_index: usize,
    start_index: usize,
) -> Option<usize> {
    if entry_index < start_index {
        return None;
    }
    for index in (start_index..=entry_index).rev() {
        let entry = &entries[index];
        if matches!(
            entry,
            CompactionSessionEntry::BranchSummary { .. }
                | CompactionSessionEntry::CustomMessage { .. }
        ) {
            return Some(index);
        }
        if let CompactionSessionEntry::Message { message, .. } = entry {
            if message.role() == "user" || message.role() == "bashExecution" {
                return Some(index);
            }
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutPointResult {
    /// Index of first entry to keep
    pub first_kept_entry_index: usize,
    /// Index of user message that starts the turn being split, or None if not splitting
    pub turn_start_index: Option<usize>,
    /// Whether this cut splits a turn (cut point is not a user message)
    pub is_split_turn: bool,
}

/// Find the cut point in session entries that keeps approximately `keepRecentTokens`.
///
/// Algorithm: Walk backwards from newest, accumulating estimated message sizes.
/// Stop when we've accumulated >= keepRecentTokens. Cut at that point.
///
/// Can cut at user OR assistant messages (never tool results). When cutting at an
/// assistant message with tool calls, its tool results come after and will be kept.
///
/// Only considers entries between `startIndex` and `endIndex` (exclusive).
pub fn find_cut_point(
    entries: &[CompactionSessionEntry],
    start_index: usize,
    end_index: usize,
    keep_recent_tokens: f64,
) -> CutPointResult {
    let cut_points = find_valid_cut_points(entries, start_index, end_index);

    if cut_points.is_empty() {
        return CutPointResult {
            first_kept_entry_index: start_index,
            turn_start_index: None,
            is_split_turn: false,
        };
    }
    let mut accumulated_tokens = 0.0;
    let mut cut_index = cut_points[0]; // Default: keep from first message (not header)

    if end_index > start_index {
        for index in (start_index..end_index).rev() {
            let entry = &entries[index];
            let CompactionSessionEntry::Message { message, .. } = entry else {
                continue;
            };
            accumulated_tokens += estimate_tokens(message);
            if accumulated_tokens >= keep_recent_tokens {
                // No cut point at/after i (trailing tool results): keep only the final turn, not everything.
                cut_index = *cut_points.last().unwrap();
                for candidate in &cut_points {
                    if *candidate >= index {
                        cut_index = *candidate;
                        break;
                    }
                }
                break;
            }
        }
    }
    while cut_index > start_index {
        let prev_entry = &entries[cut_index - 1];
        if matches!(prev_entry, CompactionSessionEntry::Compaction { .. }) {
            break;
        }
        if matches!(prev_entry, CompactionSessionEntry::Message { .. }) {
            break;
        }
        cut_index -= 1;
    }
    let cut_entry = &entries[cut_index];
    let is_user_message = matches!(
        cut_entry,
        CompactionSessionEntry::Message { message, .. } if message.role() == "user"
    );
    // A cut in a non-user turn requires a prefix summary.
    let turn_start_index = if is_user_message {
        None
    } else {
        find_turn_start_index(entries, cut_index, start_index)
    };

    CutPointResult {
        first_kept_entry_index: cut_index,
        turn_start_index,
        is_split_turn: !is_user_message && turn_start_index.is_some(),
    }
}

#[derive(Debug, Clone)]
pub struct CompactionPreparation {
    /// UUID of first entry to keep
    pub first_kept_entry_id: String,
    /// Messages that will be summarized and discarded
    pub messages_to_summarize: Vec<AgentMessage>,
    /// Messages that will be turned into turn prefix summary (if splitting)
    pub turn_prefix_messages: Vec<AgentMessage>,
    /// Whether this is a split turn (cut point in middle of turn)
    pub is_split_turn: bool,
    pub tokens_before: f64,
    /// Summary from previous compaction, for iterative update
    pub previous_summary: Option<String>,
    /// File operations extracted from messagesToSummarize
    pub file_ops: FileOperations,
    /// Compaction settions from settings.jsonl
    pub settings: CompactionSettings,
}

pub fn prepare_compaction(
    path_entries: &[CompactionSessionEntry],
    settings: &CompactionSettings,
    build_session_context: SessionContextBuilder<'_>,
) -> Option<CompactionPreparation> {
    if let Some(last) = path_entries.last() {
        if matches!(last, CompactionSessionEntry::Compaction { .. }) {
            return None;
        }
    }

    let mut prev_compaction_index: i64 = -1;
    for (index, entry) in path_entries.iter().enumerate().rev() {
        if let CompactionSessionEntry::Compaction { details, .. } = entry {
            let details_value = details.clone().unwrap_or(Value::Null);
            if !has_provider_checkpoint(&details_value) {
                prev_compaction_index = index as i64;
                break;
            }
        }
    }

    let mut previous_summary: Option<String> = None;
    let mut boundary_start = 0usize;
    if prev_compaction_index >= 0 {
        if let CompactionSessionEntry::Compaction {
            summary,
            first_kept_entry_id,
            ..
        } = &path_entries[prev_compaction_index as usize]
        {
            previous_summary = Some(summary.clone());
            let first_kept_entry_index = path_entries
                .iter()
                .position(|entry| entry.id() == first_kept_entry_id);
            boundary_start = match first_kept_entry_index {
                Some(index) => index,
                None => prev_compaction_index as usize + 1,
            };
        }
    }
    let boundary_end = path_entries.len();

    let tokens_before = estimate_context_tokens(&build_session_context(path_entries)).tokens;

    let cut_point = find_cut_point(
        path_entries,
        boundary_start,
        boundary_end,
        settings.keep_recent_tokens,
    );
    let first_kept_entry = path_entries.get(cut_point.first_kept_entry_index)?;
    if first_kept_entry.id().is_empty() {
        return None; // Session needs migration
    }
    let first_kept_entry_id = first_kept_entry.id().to_string();

    let history_end = if cut_point.is_split_turn {
        cut_point
            .turn_start_index
            .unwrap_or(cut_point.first_kept_entry_index)
    } else {
        cut_point.first_kept_entry_index
    };
    let mut messages_to_summarize: Vec<AgentMessage> = Vec::new();
    for index in boundary_start..history_end {
        if let Some(message) = get_message_from_entry_for_compaction(&path_entries[index]) {
            messages_to_summarize.push(message);
        }
    }
    let mut turn_prefix_messages: Vec<AgentMessage> = Vec::new();
    if cut_point.is_split_turn {
        let turn_start = cut_point.turn_start_index.unwrap_or(0);
        for index in turn_start..cut_point.first_kept_entry_index {
            if let Some(message) = get_message_from_entry_for_compaction(&path_entries[index]) {
                turn_prefix_messages.push(message);
            }
        }
    }

    // Avoid a compaction that would summarize no history.
    if messages_to_summarize.is_empty()
        && turn_prefix_messages.is_empty()
        && previous_summary.is_none()
    {
        return None;
    }
    let mut file_ops =
        extract_file_operations(&messages_to_summarize, path_entries, prev_compaction_index);
    // Split turns retain their suffix, but their prefix file operations still belong in the summary.
    if cut_point.is_split_turn {
        for message in &turn_prefix_messages {
            extract_file_ops_from_message(message, &mut file_ops);
        }
    }

    Some(CompactionPreparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn: cut_point.is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings: settings.clone(),
    })
}

// ---------------------------------------------------------------------------
// Summary calls
// ---------------------------------------------------------------------------

/// One wire summary call. The argument is the per-call header map that
/// `SummaryCallRunner` supplies.
pub type SummaryCallFn = Arc<
    dyn Fn(
            Option<serde_json::Map<String, Value>>,
        ) -> pi_ai::types::BoxFuture<Result<AssistantMessage, String>>
        + Send
        + Sync,
>;

/// Runs one summary wire call; hosts decorate each call with its own request identity.
pub type SummaryCallRunner = Arc<
    dyn Fn(SummaryCallFn) -> pi_ai::types::BoxFuture<Result<AssistantMessage, String>>
        + Send
        + Sync,
>;

/// `(call) => call(headers)`: the default runner forwards the caller's headers.
pub fn default_summary_call_runner(
    headers: Option<serde_json::Map<String, Value>>,
) -> SummaryCallRunner {
    Arc::new(move |call: SummaryCallFn| call(headers.clone()))
}

fn abort_error() -> String {
    "Aborted".to_string()
}

/// Generate a summary of the conversation using the LLM.
/// If previousSummary is provided, uses the update prompt to merge.
#[allow(clippy::too_many_arguments)]
pub async fn generate_summary(
    current_messages: &[AgentMessage],
    model: &Model,
    reserve_tokens: f64,
    api_key: &str,
    signal: Option<&tokio_util::sync::CancellationToken>,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    thinking_level: Option<&ThinkingLevel>,
    retry: Option<&ProviderRetryPolicy>,
    summary_call: SummaryCallRunner,
    summary_update_policy: &SummaryUpdatePolicy,
) -> Result<SummarySlice, String> {
    generate_summary_with_options(current_messages, model, reserve_tokens, api_key, signal,
        custom_instructions, previous_summary, thinking_level, retry, summary_call,
        summary_update_policy, None, &CompactionMetrics::new(None, model)).await
}

#[allow(clippy::too_many_arguments)]
async fn generate_summary_with_options(
    current_messages: &[AgentMessage], model: &Model, reserve_tokens: f64, api_key: &str,
    signal: Option<&tokio_util::sync::CancellationToken>, custom_instructions: Option<&str>,
    previous_summary: Option<&str>, thinking_level: Option<&ThinkingLevel>,
    retry: Option<&ProviderRetryPolicy>, summary_call: SummaryCallRunner,
    summary_update_policy: &SummaryUpdatePolicy, request_options: Option<&CompactionOptions>,
    metrics: &CompactionMetrics,
) -> Result<SummarySlice, String> {
    let instructions = {
        let custom_instructions = custom_instructions.map(str::to_string);
        let summary_update_policy = summary_update_policy.clone();
        move |previous: Option<&str>| {
            build_summarization_prompt(
                custom_instructions.as_deref(),
                previous,
                &summary_update_policy,
            )
        }
    };
    generate_bounded_summary(
        current_messages,
        model,
        (0.8 * reserve_tokens).floor(),
        api_key,
        signal,
        thinking_level,
        retry,
        summary_call,
        &instructions,
        previous_summary,
        SummaryFormat::Conversation,
        request_options,
        metrics,
    )
    .await
}

/// Reconstructing history after a model switch can exceed the destination's input limit.
fn summary_output_budgets(
    model: &Model,
    requested: f64,
    thinking_level: Option<&ThinkingLevel>,
    request_options: Option<&CompactionOptions>,
) -> Result<(f64, f64), String> {
    let mut ceiling = model.max_tokens.min((get_model_input_limit(model) / 4.0).floor());
    if !ceiling.is_finite() || ceiling < 1.0 {
        return Err("Compaction model has no usable output budget".to_string());
    }
    // Responses counts hidden reasoning inside max_output_tokens. Other APIs
    // keep their adapter-owned reasoning budget behavior (no double addition).
    let reasoning_budgeted = model.reasoning
        && matches!(model.api.as_str(), "openai-responses" | "azure-openai-responses")
        && thinking_level != Some(&ThinkingLevel::Off);
    if reasoning_budgeted {
        // Local ceiling for the added reasoning headroom. Preserve other APIs'
        // preexisting initial output allocation, including Ollama.
        ceiling = ceiling.min(65_536.0);
    }
    let requested = if reasoning_budgeted {
        pi_ai::providers::simple_options::adjust_max_tokens_for_thinking(
            requested,
            ceiling,
            &thinking_level.map(|level| level.as_str()).unwrap_or("medium").to_string(),
            request_options.and_then(|options| options.simple.thinking_budgets.as_ref()),
        ).max_tokens
    } else {
        requested
    };
    let initial = requested.min(ceiling).floor().max(1.0);
    // The subscription Codex serializer does not send max_output_tokens. A
    // larger local option would replay the same wire budget, not recover it.
    if model.api == "openai-codex-responses" {
        return Ok((initial, initial));
    }
    Ok((initial, (initial * 2.0).min(ceiling).min(65_536.0).floor().max(initial)))
}

#[allow(clippy::too_many_arguments)]
async fn generate_bounded_summary(
    messages: &[AgentMessage],
    model: &Model,
    requested_max_tokens: f64,
    api_key: &str,
    signal: Option<&tokio_util::sync::CancellationToken>,
    thinking_level: Option<&ThinkingLevel>,
    retry: Option<&ProviderRetryPolicy>,
    summary_call: SummaryCallRunner,
    // `Send + Sync` for the same reason: the caller awaits inside a `Send` future.
    instructions: &(dyn Fn(Option<&str>) -> String + Send + Sync),
    previous_summary: Option<&str>,
    format: SummaryFormat,
    request_options: Option<&CompactionOptions>,
    metrics: &CompactionMetrics,
) -> Result<SummarySlice, String> {
    let mut phase = metrics.phase(match format {
        SummaryFormat::Conversation => PerformanceMetricOperation::CompactionHistory,
        SummaryFormat::TurnPrefix => PerformanceMetricOperation::CompactionPrefix,
    });
    let requests = phase.requests();
    let result = async {
    let input_limit = get_model_input_limit(model);
    let (max_tokens, retry_max_tokens) = summary_output_budgets(model, requested_max_tokens, thinking_level, request_options)?;
    let mut length_retry_used = false;
    let conversation = serialize_conversation(&convert_to_llm(messages, &Default::default()));
    phase.measurement(PerformanceMetricMeasurement::SerializedBytes, Some(conversation.len() as f64));
    let conversation_chars = conversation.chars().count();
    let mut offset = 0usize;
    let mut summary: Option<String> = previous_summary.map(str::to_string);
    let mut usage = empty_usage();
    loop {
        if signal.map(|signal| signal.is_cancelled()).unwrap_or(false) {
            return Err(abort_error());
        }
        let suffix = format!(
            "{}{}",
            match &summary {
                Some(summary) => format!("<previous-summary>\n{summary}\n</previous-summary>\n\n"),
                None => String::new(),
            },
            instructions(summary.as_deref())
        );
        // Leave output headroom and use a conservative chars/3 estimate for fallback calls.
        // Reserve the possible retry's output headroom before selecting a chunk,
        // so retrying never drops or changes the transcript being summarized.
        let budget = (((f64::min(input_limit, model.context_window - retry_max_tokens) - 1024.0) * 3.0)
            .floor())
            - suffix.chars().count() as f64
            - SUMMARIZATION_SYSTEM_PROMPT.chars().count() as f64
            - 64.0;
        if budget <= 0.0 {
            return Err(
                "Compaction instructions and previous summary exceed the model input budget"
                    .to_string(),
            );
        }
        let chunk: String = slice_chars(&conversation, offset, offset + budget as usize);
        offset += chunk.chars().count();
        let mut attempt_max_tokens = max_tokens;
        let response = loop {
        let max_tokens = attempt_max_tokens;
        let chunk = chunk.clone();
        let model_for_call = model.clone();
        let api_key = api_key.to_string();
        let suffix_for_call = suffix.clone();
        let thinking_level = thinking_level.cloned();
        let signal_for_call = signal.cloned();
        // `SummaryCallFn` is `'static`, so the borrowed retry policy and the
        // borrowed signal are cloned into owned values before the closure.
        let retry_for_call = retry.cloned();
        let base_options = request_options.map(|options| options.simple.clone()).unwrap_or_default();
        let requests_for_call = requests.clone();
        let attempt: SummaryCallFn = Arc::new(
            move |call_headers: Option<serde_json::Map<String, Value>>| {
                let model = model_for_call.clone();
                let api_key = api_key.clone();
                let suffix = suffix_for_call.clone();
                let thinking_level = thinking_level.clone();
                let signal = signal_for_call.clone();
                // The wire call owns its own copy: the retry layer below still
                // needs `signal` to decide whether a failure was a cancel.
                let signal_for_complete = signal.clone();
                let retry_for_call = retry_for_call.clone();
                let chunk = chunk.clone();
                let headers = call_headers.clone();
                let base_options = base_options.clone();
                let requests = requests_for_call.clone();
                Box::pin(async move {
                    let complete = move || {
                        let model = model.clone();
                        let api_key = api_key.clone();
                        let suffix = suffix.clone();
                        let thinking_level = thinking_level.clone();
                        let signal = signal_for_complete.clone();
                        let headers = headers.clone();
                        let chunk = chunk.clone();
                        let base_options = base_options.clone();
                        let requests = requests.clone();
                        Box::pin(async move {
                            let mut options = base_options;
                            options.stream.max_tokens = Some(max_tokens);
                            options.stream.api_key = Some(api_key);
                            options.stream.signal = signal.clone();
                            options.stream.headers = headers.as_ref().and_then(|headers| {
                                let mut map = indexmap::IndexMap::new();
                                for (key, value) in headers {
                                    if let Some(value) = value.as_str() {
                                        map.insert(key.clone(), value.to_string());
                                    }
                                }
                                Some(map)
                            });
                            options.reasoning = None;
                            if model.reasoning {
                                if let Some(level) = thinking_level {
                                    if level != pi_agent_core::types::ThinkingLevel::Off {
                                        options.reasoning = Some(level.as_str().to_string());
                                    }
                                }
                            }
                            let context = Context {
                            system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
                            messages: vec![Message::User(pi_ai::types::UserMessage::new(
                                pi_ai::types::UserContent::Blocks(vec![
                                    pi_ai::types::ImageOrTextContent::Text(pi_ai::types::TextContent::new(
                                        format!("<conversation>\n{chunk}\n</conversation>\n\n{suffix}"),
                                    )),
                                ]),
                                now_millis(),
                            ))],
                            tools: None,
                        };
                            let request_metrics = requests.next();
                            request_metrics.observe(&mut options);
                            let mut message = complete_simple(&model, &context, Some(&options)).await;
                            // Keep one retry owner. An empty successful response is not a usable
                            // checkpoint, so let the existing bounded provider policy retry it.
                            if message.stop_reason == pi_ai::types::STOP_REASON_STOP
                            && !message
                                .content
                                .iter()
                                .any(|part| matches!(part, pi_ai::types::ContentBlock::Text(text) if !text.text.trim().is_empty()))
                        {
                            message.stop_reason = STOP_REASON_ERROR.to_string();
                            message.error_message = Some("Summarization returned an empty summary".to_string());
                        }
                            request_metrics.finish(&message, signal.as_ref().is_some_and(|signal| signal.is_cancelled()));
                            Ok(message)
                        })
                            as pi_ai::types::BoxFuture<Result<AssistantMessage, String>>
                    };
                    complete_with_provider_retry(&complete, retry_for_call.as_ref(), signal.as_ref())
                        .await
                })
            },
        );
        let response = match signal {
            Some(signal) => tokio::select! {
                biased;
                _ = signal.cancelled() => return Err(abort_error()),
                response = (summary_call)(attempt) => response?,
            },
            None => (summary_call)(attempt).await?,
        };
        if signal.map(|signal| signal.is_cancelled()).unwrap_or(false) {
            return Err(abort_error());
        }
        add_assistant_usage(&mut usage, &response.usage);
        // Chat Completions maps only finish_reason=length to this terminal and
        // does not retain the raw reason. Responses also maps generic incomplete
        // to length, so it still requires explicit max_output_tokens evidence.
        let exhausted_output = response.stop_reason_raw.as_deref() == Some("max_output_tokens")
            || (model.api == "openai-completions"
                && matches!(response.stop_reason_raw.as_deref(), None | Some("length")));
        if !length_retry_used && retry_max_tokens > max_tokens
            && response.stop_reason == pi_ai::types::STOP_REASON_LENGTH
            && exhausted_output
            && response.error_message.as_deref().is_none_or(str::is_empty)
            && provider_stream_failure_kind(&response).is_none()
            && !is_agent_lifecycle_failure(&response)
            && !response.content.iter().any(|part| matches!(part, pi_ai::types::ContentBlock::ToolCall(_)))
        {
            length_retry_used = true;
            attempt_max_tokens = retry_max_tokens;
            continue;
        }
        break response;
        };
        if response.stop_reason == STOP_REASON_ERROR || response.stop_reason == STOP_REASON_ABORTED
        {
            let reason = response
                .error_message
                .clone()
                .unwrap_or_else(|| response.stop_reason.clone());
            return Err(format!("Summarization failed: {reason}"));
        }
        let text = response
            .content
            .iter()
            .filter_map(|part| match part {
                pi_ai::types::ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        validate_summary(&response, &text, format)?;
        summary = Some(text);
        if offset >= conversation_chars {
            break;
        }
    }
    Ok(SummarySlice {
        summary: summary.unwrap_or_default(),
        usage: Some(usage),
    })
    }.await;
    phase.finish_result(&result, signal.is_some_and(|signal| signal.is_cancelled()));
    result
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Clone, Copy)]
enum SummaryFormat {
    Conversation,
    TurnPrefix,
}

// Enforce the handoff structure already requested by the summarization prompts.
// There is deliberately no minimum length: "none" is a legitimate section body.
fn validate_summary(response: &AssistantMessage, text: &str, format: SummaryFormat) -> Result<(), String> {
    let fail = |reason: &str| format!("Summarization returned an unusable handoff ({reason}); existing conversation preserved. Retry compaction or select another summarization model.");
    if response.stop_reason != pi_ai::types::STOP_REASON_STOP {
        // A short incomplete response is not proof of a transport failure or an
        // exhausted output budget, so report the actual terminal verbatim (the
        // stop_reason enum and the provider's raw finish signal, if any). A
        // terminal with no usable enum value (for example a stream that closed
        // mid-generation without a finish signal) reports as `unknown`, never
        // as a blank.
        let stop_label = match response.stop_reason.trim() {
            "" => "unknown",
            other => other,
        };
        let raw = response
            .stop_reason_raw
            .as_deref()
            .map(str::trim)
            .filter(|raw| !raw.is_empty());
        let detail = match raw {
            Some(raw) => format!(
                "response did not complete (stop_reason={stop_label}, provider_status={raw})"
            ),
            None => format!("response did not complete (stop_reason={stop_label})"),
        };
        return Err(fail(&detail));
    }
    if response.error_message.as_deref().is_some_and(|message| !message.trim().is_empty()) {
        return Err(fail("provider reported an error"));
    }
    if response.stop_reason_raw.as_deref().is_some_and(|reason| {
        let reason = reason.to_ascii_lowercase();
        ["refusal", "content_filter", "safety", "blocked"].iter().any(|marker| reason.contains(marker))
    }) {
        return Err(fail("provider refused or filtered the summary"));
    }
    if text.trim().is_empty() {
        return Err("Summarization returned an empty summary".to_string());
    }
    let text = text.trim();
    let text = text.split_once('\n').and_then(|(opening, body)| {
        if matches!(opening.trim(), "```" | "```md" | "```markdown") {
            body.strip_suffix("```").map(str::trim)
        } else {
            None
        }
    }).unwrap_or(text);
    let required: &[&str] = match format {
        SummaryFormat::Conversation => &["Goal", "Constraints & Preferences", "Progress", "Key Decisions", "Next Steps", "Critical Context"],
        SummaryFormat::TurnPrefix => &["Original Request", "Early Progress", "Context for Suffix"],
    };
    let mut seen = vec![false; required.len()];
    let mut bodies = vec![false; required.len()];
    let mut current = None;
    let mut fence: Option<&str> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("```") || line.starts_with("~~~") {
            let marker = &line[..3];
            if fence == Some(marker) { fence = None; }
            else if fence.is_none() { fence = Some(marker); }
            continue;
        }
        if fence.is_none() {
            let heading = line.trim_start_matches('#');
            if heading.len() != line.len() && heading.starts_with(char::is_whitespace) {
                let heading = heading.trim().trim_end_matches('#').trim().trim_matches('*').trim().trim_end_matches(':');
                if let Some(index) = required.iter().position(|required| heading.eq_ignore_ascii_case(required)) {
                    // A repeated heading is a formatting slip, not an unusable handoff:
                    // the section content is still present, so keep counting its body.
                    seen[index] = true;
                    current = Some(index);
                }
                continue;
            }
        }
        if let Some(index) = current {
            if line.chars().any(char::is_alphanumeric) {
                bodies[index] = true;
            }
        }
    }
    if seen.iter().any(|seen| !seen) || bodies.iter().any(|body| !body) || fence.is_some() {
        return Err(fail("missing, empty or incomplete handoff sections"));
    }
    Ok(())
}

/// JavaScript `String.prototype.slice(start, end)` by UTF-16 code units; the
/// port uses character offsets, which match for the text this function slices.
fn slice_chars(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

// ---------------------------------------------------------------------------
// compact()
// ---------------------------------------------------------------------------

/// Generate summaries for compaction using prepared data.
/// Returns CompactionResult - SessionManager adds uuid/parentUuid when saving.
#[allow(clippy::too_many_arguments)]
pub async fn compact(
    preparation: &CompactionPreparation,
    model: &Model,
    api_key: &str,
    custom_instructions: Option<&str>,
    signal: Option<&tokio_util::sync::CancellationToken>,
    thinking_level: Option<&ThinkingLevel>,
    summary_call: SummaryCallRunner,
    retry: Option<&ProviderRetryPolicy>,
    provider_context: Option<(&Context, Option<&CompactionOptions>)>,
) -> Result<CompactionResult, String> {
    compact_with_metrics(preparation, model, api_key, custom_instructions, signal, thinking_level,
        summary_call, retry, provider_context, &CompactionMetrics::new(None, model)).await
}

#[allow(clippy::too_many_arguments)]
pub async fn compact_with_metrics(
    preparation: &CompactionPreparation, model: &Model, api_key: &str,
    custom_instructions: Option<&str>, signal: Option<&tokio_util::sync::CancellationToken>,
    thinking_level: Option<&ThinkingLevel>, summary_call: SummaryCallRunner,
    retry: Option<&ProviderRetryPolicy>, provider_context: Option<(&Context, Option<&CompactionOptions>)>,
    metrics: &CompactionMetrics,
) -> Result<CompactionResult, String> {
    // Cancel detached provider workers when the compaction future is dropped or a sibling fails.
    // A child token never cancels the parent session's signal.
    let local_signal = signal.map(|signal| signal.child_token()).unwrap_or_default();
    let _cancel_on_drop = local_signal.clone().drop_guard();
    let signal = Some(&local_signal);
    let request_options = provider_context.and_then(|(_, options)| options);
    let CompactionPreparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings,
    } = preparation;
    let mut native_compaction_unsupported = false;
    if let Some((context, options)) = provider_context {
        if supports_compaction(model) {
            let native_phase = metrics.phase(PerformanceMetricOperation::CompactionNative);
            let model_for_call = model.clone();
            let context = context.clone();
            let options = options.cloned();
            let api_key = api_key.to_string();
            let custom_instructions = custom_instructions.map(str::to_string);
            let signal_for_call = signal.cloned();
            let attempt = Arc::new(move || {
                let model = model_for_call.clone();
                let context = context.clone();
                let options = options.clone();
                let api_key = api_key.clone();
                let custom_instructions = custom_instructions.clone();
                let signal = signal_for_call.clone();
                Box::pin(async move {
                    let mut merged = options.unwrap_or_default();
                    merged.simple.stream.api_key = Some(api_key);
                    merged.simple.stream.signal = signal;
                    merged.custom_instructions = custom_instructions;
                    let model_for_request = model.clone();
                    let context_for_request = context.clone();
                    let merged_for_request = merged.clone();
                    let compact_call = move || {
                        let model = model_for_request.clone();
                        let context = context_for_request.clone();
                        let options = merged_for_request.clone();
                        Box::pin(
                            async move { compact_simple(&model, &context, Some(&options)).await },
                        )
                    };
                    // `compactSimple` resolves to the value (or undefined); the
                    // retry layer owns the error channel, so a provider failure is
                    // mapped to `ProviderRequestError` here.
                    // TS `compactSimple` resolving to `undefined` is a NORMAL
                    // outcome: the caller falls through to the text summarizer
                    // (compaction.ts:895-925). Only a request failure is an error.
                    Ok(compact_call().await)
                })
                    as pi_ai::types::BoxFuture<
                        Result<
                            Option<pi_ai::compaction::ProviderCompactionResult>,
                            ProviderRequestError,
                        >,
                    >
            });
            // `attempt()` already resolves to `ProviderRequestError`, so the
            // closure only has to pin the boxed future at that type.
            let request = || -> pi_ai::types::BoxFuture<
                Result<
                    Option<pi_ai::compaction::ProviderCompactionResult>,
                    ProviderRequestError,
                >,
            > {
                let attempt = attempt.clone();
                Box::pin(async move { attempt().await })
            };
            let remote_result = request_with_provider_retry(&request, retry, signal).await;
            native_phase.finish(match &remote_result {
                _ if local_signal.is_cancelled() => PerformanceMetricOutcome::Cancelled,
                Ok(Some(remote)) if !is_compaction_checkpoint(&serde_json::to_value(&remote.checkpoint).unwrap_or(Value::Null))
                    || !compaction_matches_model(&remote.checkpoint, model) => PerformanceMetricOutcome::Failure,
                Ok(Some(_)) => PerformanceMetricOutcome::Success,
                Ok(None) => PerformanceMetricOutcome::Unavailable,
                Err(_) => PerformanceMetricOutcome::Failure,
            });
            let remote = remote_result.map_err(|error| error.message)?;
            if let Some(remote) = remote {
                if signal.map(|signal| signal.is_cancelled()).unwrap_or(false) {
                    return Err(abort_error());
                }
                if !is_compaction_checkpoint(
                    &serde_json::to_value(&remote.checkpoint).unwrap_or(Value::Null),
                ) || !compaction_matches_model(&remote.checkpoint, model)
                {
                    return Err(
                        "Provider compaction returned an incompatible checkpoint".to_string()
                    );
                }
                let (read_files, modified_files) = compute_file_lists(file_ops);
                return Ok(CompactionResult {
                    summary: "Conversation compacted on the server. The original transcript remains available in session history.".to_string(),
                    first_kept_entry_id: first_kept_entry_id.clone(),
                    tokens_before: *tokens_before,
                    usage: remote.usage,
                    details: Some(CompactionDetails {
                        read_files,
                        modified_files,
                        provider_checkpoint: Some(remote.checkpoint),
                    }),
                });
            }
            native_compaction_unsupported = is_staged_azure_native_compaction_model(model);
        }
    }
    let mut slices: Vec<SummarySlice> = Vec::new();
    let summary: String;
    // `settings.summaryUpdatePolicy ?? SUMMARY_UPDATE_POLICY_OFF` is read once per
    // summarisation call; `generateSummary` takes it by reference.
    let summary_update_policy = settings
        .summary_update_policy
        .clone()
        .unwrap_or_else(|| SUMMARY_UPDATE_POLICY_OFF.to_string());

    if *is_split_turn && !turn_prefix_messages.is_empty() {
        // Split turns make two wire calls with different bodies; each needs its own identity.
        let history_future = async {
            if !messages_to_summarize.is_empty() {
                generate_summary_with_options(
                    messages_to_summarize,
                    model,
                    settings.reserve_tokens,
                    api_key,
                    signal,
                    custom_instructions,
                    previous_summary.as_deref(),
                    thinking_level,
                    retry,
                    summary_call.clone(),
                    &summary_update_policy,
                    request_options,
                    metrics,
                )
                .await
            } else {
                Ok(SummarySlice {
                    summary: "No prior history.".to_string(),
                    usage: None,
                })
            }
        };
        let prefix_future = generate_turn_prefix_summary(
            turn_prefix_messages,
            model,
            settings.reserve_tokens,
            api_key,
            signal,
            thinking_level,
            retry,
            summary_call.clone(),
            request_options,
            metrics,
        );
        let (history_result, turn_prefix_result) = tokio::try_join!(history_future, prefix_future)?;
        slices.push(history_result.clone());
        slices.push(turn_prefix_result.clone());
        summary = format!(
            "{}\n\n---\n\n**Turn Context (split turn):**\n\n{}",
            history_result.summary, turn_prefix_result.summary
        );
    } else {
        let result = generate_summary_with_options(
            messages_to_summarize,
            model,
            settings.reserve_tokens,
            api_key,
            signal,
            custom_instructions,
            previous_summary.as_deref(),
            thinking_level,
            retry,
            summary_call,
            &summary_update_policy,
            request_options,
            metrics,
        )
        .await?;
        summary = result.summary.clone();
        slices.push(result);
    }
    let (read_files, modified_files) = compute_file_lists(file_ops);
    let mut summary = summary + &format_file_operations(&read_files, &modified_files);
    if native_compaction_unsupported {
        summary += "\n\n**Compaction fallback:** The staged Azure Astra native compaction endpoint returned an explicit unsupported response, so this checkpoint was created with the selected model's ordinary text summarizer. Authentication, rate-limit, timeout, cancellation, request-size, and malformed-checkpoint failures do not use this fallback. No provider checkpoint was committed.";
    }

    if first_kept_entry_id.is_empty() {
        return Err("First kept entry has no UUID - session may need migration".to_string());
    }

    let mut usage: Option<Usage> = None;
    for slice in &slices {
        let Some(slice_usage) = &slice.usage else {
            continue;
        };
        let total = usage.get_or_insert_with(empty_usage);
        add_assistant_usage(total, slice_usage);
    }
    Ok(CompactionResult {
        summary,
        first_kept_entry_id: first_kept_entry_id.clone(),
        tokens_before: *tokens_before,
        details: Some(CompactionDetails {
            read_files,
            modified_files,
            provider_checkpoint: None,
        }),
        usage,
    })
}

fn provider_request_error(error: String) -> ProviderRequestError {
    ProviderRequestError {
        message: error,
        ..Default::default()
    }
}

/// Generate a summary for a turn prefix (when splitting a turn).
#[allow(clippy::too_many_arguments)]
async fn generate_turn_prefix_summary(
    messages: &[AgentMessage],
    model: &Model,
    reserve_tokens: f64,
    api_key: &str,
    signal: Option<&tokio_util::sync::CancellationToken>,
    thinking_level: Option<&ThinkingLevel>,
    retry: Option<&ProviderRetryPolicy>,
    summary_call: SummaryCallRunner,
    request_options: Option<&CompactionOptions>,
    metrics: &CompactionMetrics,
) -> Result<SummarySlice, String> {
    let instructions = |_: Option<&str>| TURN_PREFIX_SUMMARIZATION_PROMPT.to_string();
    generate_bounded_summary(
        messages,
        model,
        (0.5 * reserve_tokens).floor(),
        api_key,
        signal,
        thinking_level,
        retry,
        summary_call,
        &instructions,
        None,
        SummaryFormat::TurnPrefix,
        request_options,
        metrics,
    )
    .await
}

#[cfg(test)]
mod summary_retry_safety_tests {
    use super::*;
    use pi_ai::types::ContentBlock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn reasoning_summary_budget_preserves_effort_and_all_caps() {
        let mut model = Model::new("fixture", "fixture", "openai-responses", "fixture", "https://fixture.invalid");
        model.reasoning = true;
        model.context_window = 1_000_000.0;
        model.max_tokens = 131_072.0;
        for api in ["openai-responses", "azure-openai-responses"] {
            model.api = api.to_string();
            assert_eq!(summary_output_budgets(&model, 13_107.0, Some(&ThinkingLevel::Xhigh), None).unwrap(), (29_491.0, 58_982.0));
            assert_eq!(summary_output_budgets(&model, 13_107.0, Some(&ThinkingLevel::Off), None).unwrap(), (13_107.0, 26_214.0));
        }
        model.api = "openai-codex-responses".into();
        assert_eq!(summary_output_budgets(&model, 13_107.0, Some(&ThinkingLevel::Xhigh), None).unwrap(), (13_107.0, 13_107.0));
        model.api = "anthropic-messages".into();
        assert_eq!(summary_output_budgets(&model, 13_107.0, Some(&ThinkingLevel::Xhigh), None).unwrap().0, 13_107.0);
        model.api = "openai-responses".into();
        model.max_tokens = 16_000.0;
        assert_eq!(summary_output_budgets(&model, 13_107.0, Some(&ThinkingLevel::High), None).unwrap(), (16_000.0, 16_000.0));
        model.max_tokens = 131_072.0;
        model.max_input_tokens = Some(40_000.0);
        assert_eq!(summary_output_budgets(&model, 13_107.0, Some(&ThinkingLevel::High), None).unwrap(), (10_000.0, 10_000.0));
        model.max_input_tokens = None;
        let mut options = CompactionOptions::default();
        options.simple.thinking_budgets = Some(pi_ai::types::ThinkingBudgets { high: Some(100_000.0), ..Default::default() });
        assert_eq!(summary_output_budgets(&model, 13_107.0, Some(&ThinkingLevel::High), Some(&options)).unwrap(), (65_536.0, 65_536.0));
    }

    #[test]
    fn incomplete_summary_terminal_reports_the_actual_stop_reason_and_provider_status() {
        let mut incomplete = AssistantMessage::default();
        incomplete.stop_reason = "length".into();
        incomplete.stop_reason_raw = Some("other".into());
        let error = validate_summary(&incomplete, "text", SummaryFormat::Conversation).unwrap_err();
        assert!(error.contains("unusable handoff"), "{error}");
        assert!(
            error.contains(
                "response did not complete (stop_reason=length, provider_status=other)"
            ),
            "{error}"
        );
        assert!(error.contains("existing conversation preserved"), "{error}");

        // A terminal without a provider status still names its stop reason.
        incomplete.stop_reason_raw = None;
        let error = validate_summary(&incomplete, "text", SummaryFormat::Conversation).unwrap_err();
        assert!(
            error.contains("response did not complete (stop_reason=length)"),
            "{error}"
        );

        // A tool-call terminal is reported as such, not as a transport failure.
        let mut tool_call = AssistantMessage::default();
        tool_call.stop_reason = "toolUse".into();
        let error = validate_summary(&tool_call, "text", SummaryFormat::Conversation).unwrap_err();
        assert!(
            error.contains("response did not complete (stop_reason=toolUse)"),
            "{error}"
        );

        // A confirmed output-token exhaustion keeps the dedicated retry above;
        // validate only reports the terminal here.
        incomplete.stop_reason_raw = Some("max_output_tokens".into());
        let error = validate_summary(&incomplete, "text", SummaryFormat::Conversation).unwrap_err();
        assert!(
            error.contains(
                "response did not complete (stop_reason=length, provider_status=max_output_tokens)"
            ),
            "{error}"
        );
    }

    #[test]
    fn an_unknown_terminal_is_reported_as_unknown_not_as_a_blank_enum() {
        // Incident 2026-09-19 shape: a summary call truncated at exactly the
        // computed request budget whose terminal carried no usable enum or raw
        // provider value. The diagnostic must name the terminal as unknown
        // instead of rendering a blank stop_reason.
        let mut unknown = AssistantMessage::default();
        unknown.stop_reason = "  ".into();
        let error = validate_summary(&unknown, "text", SummaryFormat::Conversation).unwrap_err();
        assert!(
            error.contains("response did not complete (stop_reason=unknown)"),
            "{error}"
        );
        assert!(error.contains("unusable handoff"), "{error}");
        assert!(error.contains("existing conversation preserved"), "{error}");
    }

    #[tokio::test]
    async fn an_unknown_terminal_is_not_replayed_by_the_provider_retry() {
        // No proven exhaustion signal means the one-shot larger-output retry
        // must not fire: exactly one provider call, then a truthful rejection.
        let attempts = Arc::new(AtomicUsize::new(0));
        let count = attempts.clone();
        let call = move || -> pi_ai::types::BoxFuture<Result<AssistantMessage, String>> {
            count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(AssistantMessage {
                content: vec![ContentBlock::Text(pi_ai::types::TextContent::new(
                    "summary cut off at the request budget",
                ))],
                stop_reason: String::new(),
                ..Default::default()
            }) })
        };
        let policy = ProviderRetryPolicy { base_delay_ms: 0.0, ..DEFAULT_PROVIDER_RETRY_POLICY };
        let result = complete_with_provider_retry(&call, Some(&policy), None).await.unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 1, "an unknown terminal is not a replayable failure");
        let error = validate_summary(&result, "summary cut off at the request budget", SummaryFormat::Conversation).unwrap_err();
        assert!(error.contains("stop_reason=unknown"), "{error}");
    }

    #[tokio::test]
    async fn summary_partial_failure_is_not_replayed_by_the_local_retry_owner() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let count = attempts.clone();
        let call = move || -> pi_ai::types::BoxFuture<Result<AssistantMessage, String>> {
            count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(AssistantMessage {
                content: vec![ContentBlock::Text(pi_ai::types::TextContent::new("already streamed"))],
                stop_reason: "error".into(), error_message: Some("stream broke".into()),
                ..Default::default()
            }) })
        };
        let policy = ProviderRetryPolicy { base_delay_ms: 0.0, ..DEFAULT_PROVIDER_RETRY_POLICY };
        let result = complete_with_provider_retry(&call, Some(&policy), None).await.unwrap();
        assert_eq!(result.stop_reason, "error");
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}
