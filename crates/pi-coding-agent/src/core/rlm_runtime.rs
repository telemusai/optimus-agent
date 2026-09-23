//! Port of packages/coding-agent/src/core/rlm-runtime.ts
#[path = "rlm_host_capabilities.rs"]
mod host_capabilities;
pub use host_capabilities::native_lifecycle_capabilities;

use std::sync::Arc;

use pi_ai::types::{Model, ServiceTier};
use pi_agent_core::types::ThinkingLevel;
use serde_json::{Map, Value};

use crate::core::agent_session::AgentSession;
use crate::core::extensions::types::ToolDefinition as ExtensionToolDefinition;
use crate::core::kernel::shared::{HostRequestHandler, KernelError};
use crate::core::thinking_levels::THINKING_LEVELS;

pub type BoxFuture<T> = pi_ai::types::BoxFuture<T>;

/// `interface RlmRunRequest`.
///
/// Request emitted by `rlm.run`; cellSourceCode preserves the spawning cell for
/// display.
#[derive(Debug, Clone, Default)]
pub struct RlmRunRequest {
    pub prompt: String,
    pub kwargs: Value,
    pub cell_source_code: Option<String>,
}

/// `interface RlmCreateSessionRequest`.
#[derive(Debug, Clone, Default)]
pub struct RlmCreateSessionRequest {
    pub prompt: String,
    pub kwargs: Value,
}

/// `interface RlmCreateSessionResult`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RlmCreateSessionResult {
    pub active_session_id: String,
    pub session_id: String,
    pub name: String,
    pub session_file: String,
    pub model: String,
}

/// `interface RlmSpawnHandle`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RlmSpawnHandle {
    pub rlm_child_id: String,
    pub name: String,
    pub session_dir: String,
    pub model: String,
}

pub type RlmSubagentRegistryStatus = String;
pub const RLM_SUBAGENT_STATUS_RUNNING: &str = "running";
pub const RLM_SUBAGENT_STATUS_COMPLETED: &str = "completed";
pub const RLM_SUBAGENT_STATUS_ERROR: &str = "error";

/// `interface RlmSubagentRegistryEntry`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RlmSubagentRegistryEntry {
    pub rlm_child_id: String,
    pub active_session_id: Option<String>,
    pub session_id: Option<String>,
    pub session_name: String,
    pub session_dir: String,
    pub status: RlmSubagentRegistryStatus,
}

/// `interface RlmListSubagentsResult`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RlmListSubagentsResult {
    pub subagents: Vec<RlmSubagentRegistryEntry>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RlmCollectResultEntry {
    pub rlm_child_id: String,
    pub session_name: Option<String>,
    pub session_dir: String,
    pub status: String,
    pub settled: bool,
    pub answer_preview: Option<String>,
    pub error: Option<String>,
    pub duration_ms: Option<f64>,
    pub tool_use_count: Option<f64>,
    pub replied_since_task: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RlmCollectResult { pub results: Vec<RlmCollectResultEntry> }

pub type RlmCollectHandler = Arc<dyn Fn(Vec<String>, u64) -> BoxFuture<Result<RlmCollectResult, String>> + Send + Sync>;

pub fn create_rlm_collect_host_handler(handler: RlmCollectHandler) -> HostRequestHandler {
    Arc::new(move |payload| {
        let handler = handler.clone();
        Box::pin(async move {
            let targets = match payload.get("targets") {
                None | Some(Value::Null) => Vec::new(),
                Some(Value::Array(values)) => values.iter().map(|value| {
                    value.as_str().map(str::trim).filter(|value| !value.is_empty()).map(str::to_string)
                        .ok_or_else(|| KernelError::new("rlm.collect targets must be non-empty strings"))
                }).collect::<Result<Vec<_>, _>>()?,
                _ => return Err(KernelError::new("rlm.collect targets must be an array of child ids or names")),
            };
            let timeout_ms = match payload.get("timeout_ms") {
                None | Some(Value::Null) => 0,
                Some(value) => value.as_u64().filter(|value| *value <= 2_147_483_647)
                    .ok_or_else(|| KernelError::new("rlm.collect timeout_ms must be a non-negative integer up to 2147483647"))?,
            };
            let result = handler(targets, timeout_ms).await.map_err(KernelError::new)?;
            serde_json::to_value(result).map_err(|error| KernelError::new(error.to_string()))
        })
    })
}

/// `interface RlmDeleteSubagentResult`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RlmDeleteSubagentResult {
    pub subagent: RlmSubagentRegistryEntry,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
}

pub const DELETE_OUTCOME_DELETED: &str = "deleted";
pub const DELETE_OUTCOME_SKIPPED_RUNNING: &str = "skipped_running";

/// `interface RlmModelMatch`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RlmModelMatch {
    pub provider: String,
    pub id: String,
    pub name: String,
    pub selector: String,
}

/// `interface RlmFindModelsResult`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RlmFindModelsResult {
    pub models: Vec<RlmModelMatch>,
}

/// `type RlmRunHandler`.
pub type RlmRunHandler = Arc<dyn Fn(RlmRunRequest) -> BoxFuture<Result<Value, String>> + Send + Sync>;
/// `type RlmCreateSessionHandler`.
pub type RlmCreateSessionHandler =
    Arc<dyn Fn(RlmCreateSessionRequest) -> BoxFuture<Result<RlmCreateSessionResult, String>> + Send + Sync>;

/// `interface AsyncBashCompletionRequest`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AsyncBashCompletionRequest {
    pub pid: f64,
    pub command: String,
    pub exit_code: f64,
}

/// `type AsyncBashCompletionHandler`.
pub type AsyncBashCompletionHandler = Arc<dyn Fn(AsyncBashCompletionRequest) -> BoxFuture<()> + Send + Sync>;
/// `type RlmListSubagentsHandler`.
pub type RlmListSubagentsHandler =
    Arc<dyn Fn() -> BoxFuture<Result<RlmListSubagentsResult, String>> + Send + Sync>;
/// `type RlmDeleteSubagentHandler`.
pub type RlmDeleteSubagentHandler =
    Arc<dyn Fn(String) -> BoxFuture<Result<RlmDeleteSubagentResult, String>> + Send + Sync>;
/// `type RlmFindModelsHandler`.
pub type RlmFindModelsHandler =
    Arc<dyn Fn(String, f64) -> BoxFuture<Result<RlmFindModelsResult, String>> + Send + Sync>;

const RLM_SUBAGENT_SESSION_NAME_MAX_LENGTH: usize = 64;
pub const DEFAULT_RLM_MODEL_SEARCH_LIMIT: f64 = 8.0;
pub const MAX_RLM_MODEL_SEARCH_LIMIT: f64 = 20.0;

/// `normalizeRequestedRlmSubagentSessionName(value, operation = "rlm.run")`.
pub fn normalize_requested_rlm_subagent_session_name(
    value: Option<&Value>,
    operation: Option<&str>,
) -> Result<Option<String>, String> {
    let operation = operation.unwrap_or("rlm.run");
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        // `undefined` is the only absent marker in the TypeScript; an explicit
        // JSON null is not a string and must fail the same way.
        return Err(format!("{operation} name must be a string"));
    }
    let Some(name) = value.as_str() else {
        return Err(format!("{operation} name must be a string"));
    };
    let name = name.trim();
    if name.is_empty() {
        return Err(format!("{operation} name must not be empty"));
    }
    if name.chars().count() > RLM_SUBAGENT_SESSION_NAME_MAX_LENGTH {
        return Err(format!(
            "{operation} name must be at most {RLM_SUBAGENT_SESSION_NAME_MAX_LENGTH} characters"
        ));
    }
    Ok(Some(name.to_string()))
}

/// `normalizeRequestedRlmSubagentThinkingLevel(value, operation = "rlm.run")`.
pub fn normalize_requested_rlm_subagent_thinking_level(
    value: Option<&Value>,
    operation: Option<&str>,
) -> Result<Option<ThinkingLevel>, String> {
    let operation = operation.unwrap_or("rlm.run");
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Err(format!("{operation} thinking must be a string"));
    }
    let Some(level) = value.as_str() else {
        return Err(format!("{operation} thinking must be a string"));
    };
    let level = level.trim().to_lowercase();
    let Some(known) = THINKING_LEVELS.iter().find(|known| **known == level) else {
        return Err(format!(
            "{operation} thinking must be one of: {}",
            THINKING_LEVELS.join(", ")
        ));
    };
    Ok(Some(match *known {
        "off" => ThinkingLevel::Off,
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::Xhigh,
        _ => ThinkingLevel::Max,
    }))
}

/// `normalizeRequestedRlmSubagentModel(value, operation = "rlm.run")`.
pub fn normalize_requested_rlm_subagent_model(
    value: Option<&Value>,
    operation: Option<&str>,
) -> Result<Option<String>, String> {
    let operation = operation.unwrap_or("rlm.run");
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Err(format!("{operation} model must be a string"));
    }
    let Some(model) = value.as_str() else {
        return Err(format!("{operation} model must be a string"));
    };
    let model = model.trim();
    if model.is_empty() {
        return Err(format!("{operation} model must not be empty"));
    }
    Ok(Some(model.to_string()))
}

/// `createDefaultRlmSubagentSessionName(prompt, childId)`.
///
/// Create a readable, collision-resistant default name usable as an
/// agent-message selector.
pub fn create_default_rlm_subagent_session_name(prompt: &str, child_id: &str) -> String {
    let prompt_slug = normalize_prompt_slug(prompt);
    let id_suffix = {
        let without_prefix = child_id.strip_prefix("sub-").unwrap_or(child_id);
        let cleaned: String = without_prefix
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .collect();
        let tail: String = cleaned
            .chars()
            .rev()
            .take(8)
            .collect::<Vec<char>>()
            .into_iter()
            .rev()
            .collect();
        if tail.is_empty() {
            "child".to_string()
        } else {
            tail
        }
    };
    let fixed_length = "subagent--".chars().count() + id_suffix.chars().count();
    let prompt_slug = if prompt_slug.is_empty() {
        "worker".to_string()
    } else {
        prompt_slug
    };
    let keep = prompt_slug
        .chars()
        .count()
        .min(RLM_SUBAGENT_SESSION_NAME_MAX_LENGTH.saturating_sub(fixed_length).max(1));
    let prompt_part: String = prompt_slug.chars().take(keep).collect();
    let prompt_part = prompt_part.trim_end_matches('-');
    format!(
        "subagent-{}-{}",
        if prompt_part.is_empty() { "worker" } else { prompt_part },
        id_suffix
    )
}

/// `.normalize("NFKD")`, strip combining marks, lowercase, collapse to dashes.
fn normalize_prompt_slug(prompt: &str) -> String {
    let decomposed: String = prompt
        .nfkd()
        .chars()
        .filter(|ch| !(0x0300..=0x036f).contains(&(*ch as u32)))
        .collect();
    let lowered = decomposed.to_lowercase();
    let collapsed: String = lowered
        .chars()
        .map(|ch| if ch.is_ascii_lowercase() || ch.is_ascii_digit() { ch } else { '-' })
        .collect();
    collapsed.trim_matches('-').to_string()
}

/// A tiny NFKD subset: the port only needs Latin combining-mark removal, which
/// is what the slug regex cares about.
trait Nfkd {
    fn nfkd(&self) -> String;
}

impl Nfkd for str {
    fn nfkd(&self) -> String {
        let mut out = String::with_capacity(self.len());
        for ch in self.chars() {
            match ch {
                // Common precomposed Latin letters with diacritics.
                'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => {
                    out.push('a');
                    out.push('\u{0301}');
                }
                'é' | 'è' | 'ê' | 'ë' => {
                    out.push('e');
                    out.push('\u{0301}');
                }
                'í' | 'ì' | 'î' | 'ï' => {
                    out.push('i');
                    out.push('\u{0301}');
                }
                'ó' | 'ò' | 'ô' | 'ö' | 'õ' => {
                    out.push('o');
                    out.push('\u{0301}');
                }
                'ú' | 'ù' | 'û' | 'ü' => {
                    out.push('u');
                    out.push('\u{0301}');
                }
                'ñ' => {
                    out.push('n');
                    out.push('\u{0303}');
                }
                'ç' => {
                    out.push('c');
                    out.push('\u{0327}');
                }
                other => out.push(other),
            }
        }
        out
    }
}

/// `isRecord(value)`.
fn is_record(value: &Value) -> bool {
    value.is_object()
}

/// `normalizeModelSearchText(value)`.
fn normalize_model_search_text(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .filter(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
        .collect()
}

/// `findRlmModelMatches(query, models, limit)`.
pub fn find_rlm_model_matches(query: &str, models: &[Model], limit: f64) -> Vec<RlmModelMatch> {
    let normalized_query = normalize_model_search_text(query.trim());
    let mut candidates: Vec<(f64, String, Model)> = models
        .iter()
        .filter_map(|model| {
            let selector = format!("{}/{}", model.provider, model.id);
            let name = if model.name.is_empty() {
                model.id.clone()
            } else {
                model.name.clone()
            };
            let fields = [selector.clone(), model.id.clone(), name];
            let normalized_fields: Vec<String> = fields
                .iter()
                .map(|field| normalize_model_search_text(field))
                .collect();
            let mut score = if normalized_query.is_empty() {
                0.0
            } else {
                f64::INFINITY
            };
            if !normalized_query.is_empty() {
                let exact_index = normalized_fields
                    .iter()
                    .position(|field| *field == normalized_query);
                let prefix_index = normalized_fields
                    .iter()
                    .position(|field| field.starts_with(&normalized_query));
                let partial_index = normalized_fields
                    .iter()
                    .position(|field| field.contains(&normalized_query));
                if let Some(index) = exact_index {
                    score = index as f64;
                } else if let Some(index) = prefix_index {
                    score = 3.0 + index as f64;
                } else if let Some(index) = partial_index {
                    score = 6.0 + index as f64;
                }
            }
            if !score.is_finite() {
                return None;
            }
            Some((score, selector, model.clone()))
        })
        .collect();
    candidates.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    candidates
        .into_iter()
        .take(limit.max(0.0) as usize)
        .map(|(_, selector, model)| RlmModelMatch {
            provider: model.provider.clone(),
            id: model.id.clone(),
            name: if model.name.is_empty() {
                model.id.clone()
            } else {
                model.name.clone()
            },
            selector,
        })
        .collect()
}

/// `createRlmCreateSessionHostHandler(handler)`.
pub fn create_rlm_create_session_host_handler(handler: RlmCreateSessionHandler) -> HostRequestHandler {
    Arc::new(move |payload: Value| {
        let handler = handler.clone();
        Box::pin(async move {
            let prompt = payload
                .get("prompt")
                .and_then(|prompt| prompt.as_str())
                .ok_or_else(|| KernelError::new("rlm.create_session prompt must be a string"))?
                .to_string();
            let kwargs = payload
                .get("kwargs")
                .filter(|kwargs| is_record(kwargs))
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));
            let result = handler(RlmCreateSessionRequest { prompt, kwargs })
                .await
                .map_err(KernelError::new)?;
            serde_json::to_value(result).map_err(|error| KernelError::new(error.to_string()))
        })
    })
}

/// `createRlmRunHostHandler(handler)`.
///
/// Adapt an RlmRunHandler into the typed `rlm.run` kernel host handler.
pub fn create_rlm_run_host_handler(handler: RlmRunHandler) -> HostRequestHandler {
    Arc::new(move |payload: Value| {
        let handler = handler.clone();
        Box::pin(async move {
            let prompt = payload
                .get("prompt")
                .and_then(|prompt| prompt.as_str())
                .ok_or_else(|| KernelError::new("rlm.run prompt must be a string"))?
                .to_string();
            let kwargs = payload
                .get("kwargs")
                .filter(|kwargs| is_record(kwargs))
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));
            let cell_source_code = payload
                .get("cellSourceCode")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            handler(RlmRunRequest {
                prompt,
                kwargs,
                cell_source_code,
            })
            .await
            .map_err(KernelError::new)
        })
    })
}

/// `createAsyncBashCompletionHostHandler(handler)`.
pub fn create_async_bash_completion_host_handler(handler: AsyncBashCompletionHandler) -> HostRequestHandler {
    Arc::new(move |payload: Value| {
        let handler = handler.clone();
        Box::pin(async move {
            let pid = payload.get("pid").cloned().unwrap_or(Value::Null);
            let command = payload.get("command").cloned().unwrap_or(Value::Null);
            let exit_code = payload.get("exitCode").cloned().unwrap_or(Value::Null);
            let pid_value = pid
                .as_f64()
                .filter(|value| value.fract() == 0.0)
                .ok_or_else(|| KernelError::new("bash.completed pid must be a positive integer"))?;
            if pid_value <= 0.0 {
                return Err(KernelError::new("bash.completed pid must be a positive integer"));
            }
            let command_value = command
                .as_str()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| KernelError::new("bash.completed command must be a non-empty string"))?
                .to_string();
            let exit_code_value = exit_code
                .as_f64()
                .filter(|value| value.fract() == 0.0)
                .ok_or_else(|| KernelError::new("bash.completed exitCode must be an integer"))?;
            handler(AsyncBashCompletionRequest {
                pid: pid_value,
                command: command_value,
                exit_code: exit_code_value,
            })
            .await;
            Ok(Value::Object(Map::new()))
        })
    })
}

/// `createRlmFindModelsHostHandler(handler)`.
///
/// Search a bounded authenticated model catalog without adding it to the system
/// prompt.
pub fn create_rlm_find_models_host_handler(handler: RlmFindModelsHandler) -> HostRequestHandler {
    Arc::new(move |payload: Value| {
        let handler = handler.clone();
        Box::pin(async move {
            let query = payload
                .get("query")
                .and_then(|query| query.as_str())
                .ok_or_else(|| KernelError::new("rlm.find_models query must be a string"))?
                .to_string();
            let limit = match payload.get("limit") {
                None => DEFAULT_RLM_MODEL_SEARCH_LIMIT,
                Some(value) => value
                    .as_f64()
                    .filter(|value| value.fract() == 0.0)
                    .ok_or_else(|| {
                        KernelError::new(format!(
                            "rlm.find_models limit must be an integer from 1 to {MAX_RLM_MODEL_SEARCH_LIMIT}"
                        ))
                    })?,
            };
            if !(1.0..=MAX_RLM_MODEL_SEARCH_LIMIT).contains(&limit) {
                return Err(KernelError::new(format!(
                    "rlm.find_models limit must be an integer from 1 to {MAX_RLM_MODEL_SEARCH_LIMIT}"
                )));
            }
            let result = handler(query, limit).await.map_err(KernelError::new)?;
            serde_json::to_value(RlmFindModelsResult {
                models: result.models,
            })
            .map_err(|error| KernelError::new(error.to_string()))
        })
    })
}

/// `createRlmListSubagentsHostHandler(handler)`.
pub fn create_rlm_list_subagents_host_handler(handler: RlmListSubagentsHandler) -> HostRequestHandler {
    Arc::new(move |_payload: Value| {
        let handler = handler.clone();
        Box::pin(async move {
            let result = handler().await.map_err(KernelError::new)?;
            serde_json::to_value(RlmListSubagentsResult {
                subagents: result.subagents,
            })
            .map_err(|error| KernelError::new(error.to_string()))
        })
    })
}

/// `createRlmDeleteSubagentHostHandler(handler)`.
pub fn create_rlm_delete_subagent_host_handler(handler: RlmDeleteSubagentHandler) -> HostRequestHandler {
    Arc::new(move |payload: Value| {
        let handler = handler.clone();
        Box::pin(async move {
            let target = payload
                .get("target")
                .and_then(|target| target.as_str())
                .map(str::trim)
                .filter(|target| !target.is_empty())
                .ok_or_else(|| KernelError::new("rlm.delete_subagent target must be a non-empty string"))?
                .to_string();
            let result = handler(target).await.map_err(KernelError::new)?;
            let mut object = Map::new();
            object.insert(
                "subagent".to_string(),
                serde_json::to_value(result.subagent)
                    .map_err(|error| KernelError::new(error.to_string()))?,
            );
            if let Some(outcome) = result.outcome {
                object.insert("outcome".to_string(), Value::String(outcome));
            }
            Ok(Value::Object(object))
        })
    })
}

/// `interface RlmSubagentRuntime`.
pub struct RlmSubagentRuntime {
    pub session: Arc<AgentSession>,
}

/// `interface CreateRlmSubagentRuntimeOptions`.
#[derive(Clone)]
pub struct CreateRlmSubagentRuntimeOptions {
    pub parent_session: Arc<AgentSession>,
    pub id: String,
    pub prompt: String,
    pub session_name: String,
    pub session_dir: String,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
    pub service_tier: ServiceTier,
    pub scoped_models: Vec<ScopedModelEntry>,
    pub active_tool_names: Vec<String>,
    pub allowed_tool_names: Option<Vec<String>>,
    pub custom_tools: Vec<ExtensionToolDefinition>,
    pub include_goals: bool,
    pub include_compact_skill: bool,
    pub rlm_depth: f64,
    pub rlm_max_depth: f64,
    pub rlm_parent_node_id: String,
    /// Request ID of the parent model call whose tool call caused this spawn.
    pub spawned_by_request_id: Option<String>,
    /// Source of the Python cell that spawned this subagent, for display.
    pub spawn_code: Option<String>,
    /// Publish the session to the parent before a host makes the runtime
    /// addressable.
    pub on_session_published: Option<Arc<dyn Fn(&Arc<AgentSession>) + Send + Sync>>,
}

/// `Array<{ model: Model<any>; thinkingLevel?: ThinkingLevel }>` entry.
#[derive(Clone)]
pub struct ScopedModelEntry {
    pub model: Model,
    pub thinking_level: Option<ThinkingLevel>,
}

/// `interface CreateRlmRootSessionOptions`.
pub struct CreateRlmRootSessionOptions {
    pub prompt: String,
    pub session_name: Option<String>,
    pub cwd: String,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
}

/// `interface SubagentRuntimeHost`.
pub trait SubagentRuntimeHost: Send + Sync {
    fn supports_retained_stop(&self) -> bool { false }
    fn create_rlm_subagent_runtime(
        &self,
        options: CreateRlmSubagentRuntimeOptions,
    ) -> BoxFuture<Result<RlmSubagentRuntime, String>>;
    /// `createRlmRootSession?(options)`.
    fn create_rlm_root_session(
        &self,
        options: CreateRlmRootSessionOptions,
    ) -> BoxFuture<Result<RlmCreateSessionResult, String>> {
        let _ = options;
        Box::pin(async { Err("createRlmRootSession is not implemented by this host".to_string()) })
    }
    /// Persist host-owned completion before the child becomes
    /// passivation-eligible.
    fn complete_rlm_subagent_runtime(&self, child_id: &str, session: &Arc<AgentSession>) -> bool {
        let _ = (child_id, session);
        true
    }
    /// Release a host-owned child after its detached initial task settles.
    fn release_rlm_subagent_runtime(
        &self,
        runtime: RlmSubagentRuntime,
        options: CreateRlmSubagentRuntimeOptions,
        status: &str,
    ) -> BoxFuture<Result<(), String>> {
        let _ = (options, status);
        Box::pin(async move { runtime.session.dispose_async(None).await; Ok(()) })
    }
    /// Close or remove the host-owned child; session is absent when a persisted
    /// child is still passive.
    fn delete_rlm_subagent_runtime(
        &self,
        child_id: &str,
        session: Option<&Arc<AgentSession>>,
    ) -> BoxFuture<Result<(), String>>;
    fn dispose_rlm_subagent_runtimes(&self) -> BoxFuture<Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model(provider: &str, id: &str, name: &str) -> Model {
        Model {
            provider: provider.to_string(),
            id: id.to_string(),
            name: name.to_string(),
            api: "openai-responses".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn session_name_normalization_rejects_bad_values() {
        assert_eq!(normalize_requested_rlm_subagent_session_name(None, None).unwrap(), None);
        assert_eq!(
            normalize_requested_rlm_subagent_session_name(Some(&json!("  name  ")), None)
                .unwrap()
                .as_deref(),
            Some("name")
        );
        assert_eq!(
            normalize_requested_rlm_subagent_session_name(Some(&json!(1)), None).unwrap_err(),
            "rlm.run name must be a string"
        );
        assert_eq!(
            normalize_requested_rlm_subagent_session_name(Some(&json!("   ")), None).unwrap_err(),
            "rlm.run name must not be empty"
        );
        let too_long = "a".repeat(65);
        assert_eq!(
            normalize_requested_rlm_subagent_session_name(Some(&json!(too_long)), None).unwrap_err(),
            "rlm.run name must be at most 64 characters"
        );
    }

    #[test]
    fn thinking_level_normalization_matches_the_level_list() {
        assert_eq!(
            normalize_requested_rlm_subagent_thinking_level(Some(&json!(" HIGH ")), None)
                .unwrap()
                .map(|level| level.as_str()),
            Some("high")
        );
        assert_eq!(
            normalize_requested_rlm_subagent_thinking_level(Some(&json!("bogus")), None).unwrap_err(),
            format!("rlm.run thinking must be one of: {}", THINKING_LEVELS.join(", "))
        );
        assert_eq!(
            normalize_requested_rlm_subagent_thinking_level(Some(&json!(3)), None).unwrap_err(),
            "rlm.run thinking must be a string"
        );
    }

    #[test]
    fn model_normalization_trims_and_rejects_empty() {
        assert_eq!(
            normalize_requested_rlm_subagent_model(Some(&json!(" openai/gpt ")), None)
                .unwrap()
                .as_deref(),
            Some("openai/gpt")
        );
        assert_eq!(
            normalize_requested_rlm_subagent_model(Some(&json!("")), None).unwrap_err(),
            "rlm.run model must not be empty"
        );
        assert_eq!(
            normalize_requested_rlm_subagent_model(Some(&json!([])), None).unwrap_err(),
            "rlm.run model must be a string"
        );
    }

    #[test]
    fn default_session_names_slug_the_prompt_and_suffix_the_child_id() {
        assert_eq!(
            create_default_rlm_subagent_session_name("Fix the Login Bug!", "sub-abcdef12345678"),
            "subagent-fix-the-login-bug-12345678"
        );
        assert_eq!(
            create_default_rlm_subagent_session_name("   ", "child"),
            "subagent-worker-child"
        );
    }

    #[test]
    fn default_session_names_stay_within_the_length_cap() {
        let long_prompt = "word ".repeat(200);
        let name = create_default_rlm_subagent_session_name(&long_prompt, "sub-12345678");
        assert!(name.chars().count() <= RLM_SUBAGENT_SESSION_NAME_MAX_LENGTH);
    }

    #[test]
    fn model_search_scores_exact_then_prefix_then_partial() {
        let models = vec![
            model("anthropic", "claude-opus", "Claude Opus"),
            model("openai", "gpt-5", "GPT 5"),
            model("openai", "gpt-5-mini", "GPT 5 mini"),
        ];
        let matches = find_rlm_model_matches("openai/gpt-5", &models, 8.0);
        assert_eq!(matches[0].selector, "openai/gpt-5");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[1].selector, "openai/gpt-5-mini");

        let matches = find_rlm_model_matches("claude", &models, 8.0);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "Claude Opus");

        let matches = find_rlm_model_matches("", &models, 2.0);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].selector, "anthropic/claude-opus");
    }

    #[tokio::test]
    async fn run_host_handler_validates_and_forwards_the_payload() {
        let captured = Arc::new(std::sync::Mutex::new(None));
        let sink = captured.clone();
        let handler = create_rlm_run_host_handler(Arc::new(move |request| {
            let sink = sink.clone();
            Box::pin(async move {
                *sink.lock().unwrap() = Some(request.prompt.clone());
                Ok(json!({ "ok": true }))
            })
        }));
        let result = handler(json!({ "prompt": "go", "kwargs": { "a": 1 }, "cellSourceCode": "cell" }))
            .await
            .unwrap();
        assert_eq!(result, json!({ "ok": true }));
        assert_eq!(captured.lock().unwrap().as_deref(), Some("go"));
        assert_eq!(
            handler(json!({ "prompt": 1 })).await.unwrap_err().to_string(),
            "rlm.run prompt must be a string"
        );
    }

    #[tokio::test]
    async fn find_models_host_handler_bounds_the_limit() {
        let handler = create_rlm_find_models_host_handler(Arc::new(|_query, _limit| {
            Box::pin(async { Ok(RlmFindModelsResult { models: Vec::new() }) })
        }));
        assert!(handler(json!({ "query": "x" })).await.is_ok());
        assert!(handler(json!({ "query": "x", "limit": 20 })).await.is_ok());
        assert_eq!(
            handler(json!({ "query": "x", "limit": 21 })).await.unwrap_err().to_string(),
            "rlm.find_models limit must be an integer from 1 to 20"
        );
        assert_eq!(
            handler(json!({ "query": "x", "limit": 2.5 })).await.unwrap_err().to_string(),
            "rlm.find_models limit must be an integer from 1 to 20"
        );
        assert_eq!(
            handler(json!({ "limit": 1 })).await.unwrap_err().to_string(),
            "rlm.find_models query must be a string"
        );
    }

    #[tokio::test]
    async fn backlog_collect_validates_targets_and_deadline_without_side_effects() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handler = create_rlm_collect_host_handler(Arc::new({
            let calls = calls.clone();
            move |targets, timeout| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert_eq!(targets, vec!["child"]);
                assert_eq!(timeout, 0);
                Box::pin(async { Ok(RlmCollectResult::default()) })
            }
        }));
        assert_eq!(handler(json!({"targets": [" child "]})).await.unwrap(), json!({"results": []}));
        for payload in [json!({"targets": true}), json!({"targets": [""]}), json!({"timeout_ms": -1}), json!({"timeout_ms": 0.5}), json!({"timeout_ms": true}), json!({"timeout_ms": 2147483648_u64})] {
            assert!(handler(payload).await.is_err());
        }
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn bash_completion_handler_validates_each_field() {
        let handler = create_async_bash_completion_host_handler(Arc::new(|_request| Box::pin(async {})));
        assert!(handler(json!({ "pid": 1, "command": "ls", "exitCode": 0 })).await.is_ok());
        assert_eq!(
            handler(json!({ "pid": 0, "command": "ls", "exitCode": 0 })).await.unwrap_err().to_string(),
            "bash.completed pid must be a positive integer"
        );
        assert_eq!(
            handler(json!({ "pid": 1, "command": "", "exitCode": 0 })).await.unwrap_err().to_string(),
            "bash.completed command must be a non-empty string"
        );
        assert_eq!(
            handler(json!({ "pid": 1, "command": "ls", "exitCode": "x" })).await.unwrap_err().to_string(),
            "bash.completed exitCode must be an integer"
        );
    }

    #[tokio::test]
    async fn list_and_delete_handlers_project_the_result_shapes() {
        let entry = RlmSubagentRegistryEntry {
            rlm_child_id: "child-1".to_string(),
            active_session_id: None,
            session_id: None,
            session_name: "worker".to_string(),
            session_dir: "/tmp".to_string(),
            status: RLM_SUBAGENT_STATUS_RUNNING.to_string(),
        };
        let list_handler = create_rlm_list_subagents_host_handler(Arc::new({
            let entry = entry.clone();
            move || {
                let entry = entry.clone();
                Box::pin(async move { Ok(RlmListSubagentsResult { subagents: vec![entry] }) })
            }
        }));
        let listed = list_handler(json!({})).await.unwrap();
        assert_eq!(listed["subagents"][0]["rlm_child_id"], json!("child-1"));

        let delete_handler = create_rlm_delete_subagent_host_handler(Arc::new({
            let entry = entry.clone();
            move |target: String| {
                let entry = entry.clone();
                Box::pin(async move {
                    Ok(RlmDeleteSubagentResult {
                        subagent: entry,
                        outcome: Some(format!("deleted:{target}")),
                    })
                })
            }
        }));
        let deleted = delete_handler(json!({ "target": "child-1" })).await.unwrap();
        assert_eq!(deleted["outcome"], json!("deleted:child-1"));
        assert_eq!(
            delete_handler(json!({ "target": "  " })).await.unwrap_err().to_string(),
            "rlm.delete_subagent target must be a non-empty string"
        );
    }
}
