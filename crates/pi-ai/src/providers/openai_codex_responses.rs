//! Port of packages/ai/src/providers/openai-codex-responses.ts
//!
//! NOTE: the TypeScript reads `node:os` through a lazy dynamic import; Rust links it at
//! build time, so `_os` is always available here.
//!
//! Native WebSocket IO feeds the existing connection cache, continuation and SSE fallback
//! state machines. The constructor remains injectable for isolated transport tests.

mod native_web_socket;

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::compaction::{CompactionOptions, ProviderCompactionResult};
use crate::env_api_keys::get_env_api_key;
use crate::models::clamp_thinking_level;
use crate::providers::openai_compaction::{
    build_codex_compacted_window, request_openai_compaction, supports_openai_compaction, CompactionRequestError,
};
use crate::providers::openai_responses_shared::{
    convert_responses_messages, convert_responses_tools, process_responses_stream, ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions, OpenAIResponsesStreamOptions, ResponsesEventStream,
};
use crate::providers::responses_transport::observe_event;
use crate::providers::simple_options::build_base_options;
use crate::session_resources::register_session_resource_cleanup;
use crate::types::{
    AssistantMessage, AssistantMessageEvent, BoxFuture, Context, Model, ProviderResponse, ServiceTier,
    SimpleStreamOptions, StreamOptions, Usage,
};
use crate::utils::diagnostics::{
    format_thrown_value, now_millis, AssistantMessageDiagnostic, DiagnosticErrorInfo, ThrownValue,
};
use crate::utils::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use crate::utils::headers::header_map_to_record;
use crate::utils::now_ms;
use crate::utils::sse_frames::SseFrames;
use crate::utils::stream_failure::{parse_retry_after_ms, record_stream_failure, ThrownStreamError};

/// `const DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"`.
pub const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
/// `const JWT_CLAIM_PATH = "https://api.openai.com/auth"`.
pub const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
/// `const CODEX_TOOL_CALL_PROVIDERS = new Set(["openai", "openai-codex", "opencode"])`.
pub const CODEX_TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];
/// `const WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE = 1009`.
pub const WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE: i32 = 1009;
/// `const CODEX_RESPONSE_STATUSES = new Set([...])`.
pub const CODEX_RESPONSE_STATUSES: [&str; 6] =
    ["completed", "incomplete", "failed", "cancelled", "queued", "in_progress"];
/// `const OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"`.
pub const OPENAI_BETA_RESPONSES_WEBSOCKETS: &str = "responses_websockets=2026-02-06";
/// `const SESSION_WEBSOCKET_CACHE_TTL_MS = 5 * 60 * 1000`.
pub const SESSION_WEBSOCKET_CACHE_TTL_MS: u64 = 5 * 60 * 1000;

/// `export interface OpenAICodexResponsesOptions extends StreamOptions`.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct OpenAICodexResponsesOptions {
    #[serde(flatten)]
    pub stream: StreamOptions,
    /// `reasoningEffort?: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// `reasoningSummary?: "auto" | "concise" | "detailed" | "off" | "on" | null`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_summary: Option<Option<String>>,
    /// `serviceTier?: ResponseCreateParamsStreaming["service_tier"]`
    ///
    /// openai-codex-responses.ts:149 re-declares `serviceTier?: ... | null` on
    /// `OpenAICodexResponsesOptions extends StreamOptions`; it is ONE nullable property
    /// (types.ts:73 `ServiceTier = ... | null`, types.ts:103), carried through `{...base}` in
    /// `streamSimpleOpenAICodexResponses` (openai-codex-responses.ts:354-361) from
    /// `buildBaseOptions` (simple-options.ts:10). `None` = key absent, `Some(None)` = explicit
    /// JSON `null` (types.ts:195-197).
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "deserialize_service_tier")]
    pub service_tier: ServiceTier,
    /// `textVerbosity?: "low" | "medium" | "high"`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_verbosity: Option<String>,
    /// `onOutputItemDone?: OpenAIResponsesStreamOptions["onOutputItemDone"]`
    #[serde(skip)]
    pub on_output_item_done: Option<crate::providers::openai_responses_shared::OnOutputItemDone>,
}

impl OpenAICodexResponsesOptions {
    /// TS: the caller passes `StreamOptions & Record<string, unknown>`; this keeps the
    /// non-serializable fields (signal, on_payload, on_response, on_usage_observation).
    pub fn from_base(base: &StreamOptions) -> Self {
        Self {
            stream: base.clone(),
            reasoning_effort: None,
            reasoning_summary: None,
            // openai-codex-responses.ts:358-361 spreads the base options into the provider
            // options, so the inherited `serviceTier` (simple-options.ts:10) must survive:
            // `buildRequestBody` reads `options?.serviceTier` off those same options
            // (openai-codex-responses.ts:390-392). Hardcoding `None` dropped the tier on every
            // request, so "flex"/"priority" (and the explicit-"default" reset) never reached the
            // server and the response-tier pricing multiplier was lost.
            service_tier: base.service_tier.clone(),
            text_verbosity: None,
            on_output_item_done: None,
        }
    }
}

/// `null` must stay an explicit `null`, not collapse into "absent"
/// (`deserialize_optional_nullable`, types.rs:199-205).
fn deserialize_service_tier<'de, D>(deserializer: D) -> Result<ServiceTier, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Some(Option::<String>::deserialize(deserializer)?))
}

/// `type CodexResponseStatus`.
pub type CodexResponseStatus = String;

/// `interface RequestBody` - a plain JSON object in the TypeScript.
pub type RequestBody = Map<String, Value>;

/// `class CodexApiError extends Error` / `class CodexProtocolError extends Error` /
/// `class WebSocketCloseError extends Error` collapsed into one thrown-value type.
#[derive(Debug, Clone, PartialEq)]
pub struct CodexThrown {
    /// `error.name`
    pub name: &'static str,
    /// `error.message`
    pub message: String,
    pub code: Option<String>,
    pub status: Option<i64>,
    pub retry_after_ms: Option<f64>,
    pub payload: Option<Value>,
    pub close_code: Option<i32>,
    pub close_reason: Option<String>,
    pub was_clean: Option<bool>,
    /// The error as a JSON object, for `extractStreamFailureInfo` / diagnostics.
    pub value: Value,
}

impl CodexThrown {
    fn build(name: &'static str, message: String) -> Self {
        Self {
            name,
            message,
            code: None,
            status: None,
            retry_after_ms: None,
            payload: None,
            close_code: None,
            close_reason: None,
            was_clean: None,
            value: Value::Null,
        }
    }

    fn refresh_value(&mut self) {
        let mut object = Map::new();
        object.insert("name".to_string(), Value::String(self.name.to_string()));
        object.insert("message".to_string(), Value::String(self.message.clone()));
        if let Some(code) = self.code.clone() {
            object.insert("code".to_string(), Value::String(code));
        }
        if let Some(status) = self.status {
            object.insert("status".to_string(), Value::Number(status.into()));
        }
        if let Some(retry_after_ms) = self.retry_after_ms {
            if let Some(number) = serde_json::Number::from_f64(retry_after_ms) {
                object.insert("retryAfterMs".to_string(), Value::Number(number));
            }
        }
        self.value = Value::Object(object);
    }

    /// `new CodexApiError(message, { code, status, retryAfterMs, payload, cause })`.
    pub fn api_error(
        message: impl Into<String>,
        code: Option<String>,
        status: Option<i64>,
        retry_after_ms: Option<f64>,
        payload: Option<Value>,
    ) -> Self {
        let mut error = Self::build("CodexApiError", message.into());
        error.code = code;
        error.status = status;
        error.retry_after_ms = retry_after_ms;
        error.payload = payload;
        error.refresh_value();
        error
    }

    /// `new CodexProtocolError(message, { payload, cause })`.
    pub fn protocol_error(message: impl Into<String>, payload: Option<Value>) -> Self {
        let mut error = Self::build("CodexProtocolError", message.into());
        error.payload = payload;
        error.refresh_value();
        error
    }

    /// `new WebSocketCloseError(message, { code, reason, wasClean })`.
    pub fn web_socket_close(
        message: impl Into<String>,
        code: Option<i32>,
        reason: Option<String>,
        was_clean: Option<bool>,
    ) -> Self {
        let mut error = Self::build("WebSocketCloseError", message.into());
        error.close_code = code;
        error.close_reason = reason;
        error.was_clean = was_clean;
        error.refresh_value();
        error
    }

    /// `new Error(message)`.
    pub fn error(message: impl Into<String>) -> Self {
        let mut error = Self::build("Error", message.into());
        error.refresh_value();
        error
    }

    /// The thrown value as the shared stream-failure helper sees it.
    pub fn to_thrown(&self) -> ThrownStreamError<'_> {
        ThrownStreamError::Value(&self.value)
    }

    /// `formatThrownValue(error)`.
    pub fn format(&self) -> String {
        format_thrown_value(&ThrownValue::Error(self))
    }
}

impl std::fmt::Display for CodexThrown {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for CodexThrown {}

/// `isCodexNonTransportError(error)`.
pub fn is_codex_non_transport_error(error: &CodexThrown) -> bool {
    error.name == "CodexApiError" || error.name == "CodexProtocolError"
}

fn is_stale_codex_continuation_error(error: &CodexThrown) -> bool {
    error.name == "CodexApiError" && error.code.as_deref().is_some_and(|code| code.eq_ignore_ascii_case("previous_response_not_found"))
}

/// `appendAssistantMessageDiagnostic(output, createAssistantMessageDiagnostic(...))`.
fn append_diagnostic(output: &mut AssistantMessage, type_: &str, error: &CodexThrown, details: Map<String, Value>) {
    let diagnostic = AssistantMessageDiagnostic {
        type_: type_.to_string(),
        timestamp: now_millis(),
        error: Some(DiagnosticErrorInfo {
            name: Some(error.name.to_string()),
            message: error.message.clone(),
            stack: None,
            code: error.code.clone().map(Value::String),
        }),
        details: Some(details),
    };
    let mut diagnostics = output.diagnostics.clone().unwrap_or_default();
    diagnostics.push(diagnostic);
    output.diagnostics = Some(diagnostics);
}

/// `new TextEncoder().encode(bodyJson).byteLength`.
fn utf8_byte_length(text: &str) -> usize {
    text.len()
}

/// `Math.max(0, err.resets_at * 1000 - Date.now())`.
fn resets_at_to_retry_after_ms(resets_at: f64) -> f64 {
    (resets_at * 1000.0 - now_ms() as f64).max(0.0)
}

/// `codexUsageLimitMessage(err, status)` result.
pub struct CodexUsageLimitMessage {
    pub friendly_message: String,
    pub retry_after_ms: Option<f64>,
}

/// `codexUsageLimitMessage(err, status)`.
pub fn codex_usage_limit_message(err: &Value, status: Option<i64>) -> Option<CodexUsageLimitMessage> {
    let code = err
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| err.get("type").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string();
    let matches_limit = code.to_lowercase().contains("usage_limit_reached")
        || code.to_lowercase().contains("usage_not_included")
        || code.to_lowercase().contains("rate_limit_exceeded");
    if !matches_limit && status != Some(429) {
        return None;
    }
    let plan = match err.get("plan_type").and_then(Value::as_str) {
        Some(plan_type) => format!(" ({} plan)", plan_type.to_lowercase()),
        None => String::new(),
    };
    let retry_after_ms = err.get("resets_at").and_then(Value::as_f64).map(resets_at_to_retry_after_ms);
    let when = match retry_after_ms {
        Some(retry_after_ms) => format!(" Try again in ~{} min.", (retry_after_ms / 60000.0).round()),
        None => String::new(),
    };
    Some(CodexUsageLimitMessage {
        friendly_message: format!("You have hit your ChatGPT usage limit{}.{}", plan, when)
            .trim()
            .to_string(),
        retry_after_ms,
    })
}

/// `normalizeCodexStatus(status)`.
pub fn normalize_codex_status(status: Option<&Value>) -> Option<CodexResponseStatus> {
    let status = status?.as_str()?;
    if CODEX_RESPONSE_STATUSES.contains(&status) {
        Some(status.to_string())
    } else {
        None
    }
}

/// `getServiceTierCostMultiplier(model, serviceTier)`.
pub fn get_service_tier_cost_multiplier(model: &Model, service_tier: Option<&str>) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority") => {
            if model.id.starts_with("gpt-5.5") {
                2.5
            } else {
                2.0
            }
        }
        _ => 1.0,
    }
}

/// `applyServiceTierPricing(usage, serviceTier, model)`.
pub fn apply_service_tier_pricing(usage: &mut Usage, service_tier: Option<&str>, model: &Model) {
    let multiplier = get_service_tier_cost_multiplier(model, service_tier);
    if multiplier == 1.0 {
        return;
    }

    usage.cost.input *= multiplier;
    usage.cost.output *= multiplier;
    usage.cost.cache_read *= multiplier;
    usage.cost.cache_write *= multiplier;
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

/// `resolveCodexServiceTier(responseServiceTier, requestServiceTier)`.
pub fn resolve_codex_service_tier(response_service_tier: Option<&str>, request_service_tier: Option<&str>) -> Option<String> {
    if response_service_tier == Some("default")
        && (request_service_tier == Some("flex") || request_service_tier == Some("priority"))
    {
        return request_service_tier.map(str::to_string);
    }
    response_service_tier
        .or(request_service_tier)
        .map(str::to_string)
}

/// `resolveCodexUrl(baseUrl?)`.
pub fn resolve_codex_url(base_url: Option<&str>) -> String {
    let raw = match base_url {
        Some(base_url) if !base_url.trim().is_empty() => base_url,
        _ => DEFAULT_CODEX_BASE_URL,
    };
    let normalized = raw.trim_end_matches('/');
    if normalized.ends_with("/codex/responses") {
        return normalized.to_string();
    }
    if normalized.ends_with("/codex") {
        return format!("{}/responses", normalized);
    }
    format!("{}/codex/responses", normalized)
}

/// `resolveCodexWebSocketUrl(baseUrl?)`.
pub fn resolve_codex_web_socket_url(base_url: Option<&str>) -> String {
    let url = resolve_codex_url(base_url);
    let mut parsed = url::Url::parse(&url).unwrap_or_else(|_| url::Url::parse(DEFAULT_CODEX_BASE_URL).expect("valid url"));
    let scheme = match parsed.scheme() {
        "https" => Some("wss"),
        "http" => Some("ws"),
        _ => None,
    };
    if let Some(scheme) = scheme {
        let _ = parsed.set_scheme(scheme);
    }
    parsed.to_string()
}

/// `buildRequestBody(model, context, options?)`.
///
/// openai-codex-responses.ts:369-371 calls `convertResponsesMessages(model, context, ...)` inside
/// the stream IIFE, so a thrown conversion error (the mismatched compaction checkpoint of
/// transform-messages.ts:77) reaches the catch at openai-codex-responses.ts:328-337: the turn ends
/// with `stopReason` "error" and that exact message, and NO request is sent. The port returns the
/// same failure as `Err`, like `build_params` in amazon_bedrock_responses.rs:236-292, instead of
/// defaulting to an empty `input`, which silently dropped the whole conversation context.
pub fn build_request_body(
    model: &Model,
    context: &Context,
    options: Option<&OpenAICodexResponsesOptions>,
) -> Result<RequestBody, String> {
    let messages = convert_responses_messages(
        model,
        context,
        &|provider: &str| CODEX_TOOL_CALL_PROVIDERS.contains(&provider),
        Some(&ConvertResponsesMessagesOptions {
            include_system_prompt: Some(false),
        }),
    )?;

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.id.clone()));
    body.insert("store".to_string(), Value::Bool(false));
    body.insert("stream".to_string(), Value::Bool(true));
    body.insert(
        "instructions".to_string(),
        Value::String(
            context
                .system_prompt
                .clone()
                // TS: `context.systemPrompt || "You are a helpful assistant."` - an
                // empty string is falsy in JS, so it falls back to the default.
                .filter(|prompt| !prompt.is_empty())
                .unwrap_or_else(|| "You are a helpful assistant.".to_string()),
        ),
    );
    body.insert("input".to_string(), Value::Array(messages));
    let mut text = Map::new();
    text.insert(
        "verbosity".to_string(),
        Value::String(
            options
                .and_then(|options| options.text_verbosity.clone())
                .unwrap_or_else(|| "low".to_string()),
        ),
    );
    body.insert("text".to_string(), Value::Object(text));
    body.insert(
        "include".to_string(),
        Value::Array(vec![Value::String("reasoning.encrypted_content".to_string())]),
    );
    if let Some(session_id) = options.and_then(|options| options.stream.session_id.clone()) {
        body.insert("prompt_cache_key".to_string(), Value::String(session_id));
    }
    body.insert("tool_choice".to_string(), Value::String("auto".to_string()));
    body.insert("parallel_tool_calls".to_string(), Value::Bool(true));

    if let Some(temperature) = options.and_then(|options| options.stream.temperature) {
        if let Some(number) = serde_json::Number::from_f64(temperature) {
            body.insert("temperature".to_string(), Value::Number(number));
        }
    }

    // openai-codex-responses.ts:390-392:
    // `if (options?.serviceTier !== undefined) { body.service_tier = options.serviceTier; }`
    // `!== undefined` is true for an explicit `null`, and `JSON.stringify` keeps that key, so
    // `Some(None)` must be written as JSON `null` rather than dropped.
    if let Some(service_tier) = options.and_then(|options| options.service_tier.clone()) {
        body.insert(
            "service_tier".to_string(),
            service_tier.map(Value::String).unwrap_or(Value::Null),
        );
    }

    if let Some(tools) = context.tools.as_ref() {
        if !tools.is_empty() {
            body.insert(
                "tools".to_string(),
                Value::Array(convert_responses_tools(
                    tools,
                    Some(&ConvertResponsesToolsOptions { strict: Some(None) }),
                )),
            );
        }
    }

    if let Some(reasoning_effort) = options.and_then(|options| options.reasoning_effort.clone()) {
        let mapped = if reasoning_effort == "none" {
            // openai-codex-responses.ts:400-402: `model.thinkingLevelMap?.off ?? "none"`.
            // `??` falls through on both null and undefined, so the result is ALWAYS the string
            // "none" at minimum and `if (effort !== null)` (codex-responses.ts:403) is always
            // true here - `body.reasoning` must never be dropped.
            Some(
                model
                    .thinking_level_map_get("off")
                    .flatten()
                    .unwrap_or_else(|| "none".to_string()),
            )
        } else {
            model
                .thinking_level_map_get(&reasoning_effort)
                .flatten()
                .or(Some(reasoning_effort.clone()))
        };
        if let Some(effort) = mapped {
            let mut reasoning = Map::new();
            reasoning.insert("effort".to_string(), Value::String(effort));
            reasoning.insert(
                "summary".to_string(),
                Value::String(
                    options
                        .and_then(|options| options.reasoning_summary.clone())
                        .flatten()
                        .unwrap_or_else(|| "auto".to_string()),
                ),
            );
            body.insert("reasoning".to_string(), Value::Object(reasoning));
        }
    }

    Ok(body)
}

/// `compactOpenAICodexResponses: CompactFunction<"openai-codex-responses">`.
pub async fn compact_openai_codex_responses(
    model: &Model,
    context: &Context,
    options: Option<&CompactionOptions>,
) -> Option<ProviderCompactionResult> {
    try_compact_openai_codex_responses(model, context, options)
        .await
        .unwrap_or(None)
}

/// [`compact_openai_codex_responses`] with the TypeScript `throw` turned into `Err`.
pub async fn try_compact_openai_codex_responses(
    model: &Model,
    context: &Context,
    options: Option<&CompactionOptions>,
) -> Result<Option<ProviderCompactionResult>, String> {
    if !supports_openai_compaction(model) {
        return Ok(None);
    }
    let api_key = options
        .and_then(|options| options.simple.stream.api_key.clone())
        .or_else(|| get_env_api_key(&model.provider));
    let Some(api_key) = api_key else {
        return Err(format!("No API key for provider: {}", model.provider));
    };
    let headers = build_sse_headers(
        model.headers.as_ref(),
        options.and_then(|options| options.simple.stream.headers.as_ref()),
        &extract_account_id(&api_key).map_err(|error| error.message)?,
        &api_key,
        options.and_then(|options| options.simple.stream.session_id.as_deref()),
    );
    let instructions = [
        context.system_prompt.clone(),
        options.and_then(|options| options.custom_instructions.clone()),
    ]
    .into_iter()
    .flatten()
    .filter(|instruction| !instruction.is_empty())
    .collect::<Vec<_>>()
    .join("\n\n");
    let reasoning_effort = options
        .and_then(|options| options.simple.reasoning.clone())
        .filter(|reasoning| reasoning != "off")
        .map(|reasoning| clamp_thinking_level(model, &reasoning));
    let mut compaction_context = context.clone();
    compaction_context.system_prompt = Some(instructions);
    let typed = OpenAICodexResponsesOptions {
        stream: options.map(|options| options.simple.stream.clone()).unwrap_or_default(),
        reasoning_effort,
        reasoning_summary: None,
        // openai-codex-responses.ts:75-85 spreads `{...options}` into buildRequestBody,
        // and buildRequestBody (openai-codex-responses.ts:390-392) emits
        // `if (options?.serviceTier !== undefined) body.service_tier = options.serviceTier;`.
        // CompactionOptions extends SimpleStreamOptions, so the caller's tier must survive here;
        // the pricing below (openai-codex-responses.ts) already prices that tier.
        service_tier: options
            .and_then(|options| options.simple.stream.service_tier.clone()),
        text_verbosity: None,
        on_output_item_done: None,
    };
    let body = build_request_body(model, &compaction_context, Some(&typed))?;
    let input: Vec<Value> = body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    // This is a request control, never part of the durable checkpoint window.
    let mut body = body.clone();
    let mut with_trigger = input.clone();
    let mut trigger = Map::new();
    trigger.insert("type".to_string(), Value::String("compaction_trigger".to_string()));
    with_trigger.push(Value::Object(trigger));
    body.insert("input".to_string(), Value::Array(with_trigger));

    // `decode(response)`: the Codex compaction SSE decoder.
    let observation_options = options.map(|options| options.simple.stream.clone()).unwrap_or_default();
    let decode = Arc::new(
        move |response: reqwest::Response| -> BoxFuture<Result<Value, CompactionRequestError>> {
            let input = input.clone();
            let observation_options = observation_options.clone();
            Box::pin(async move {
                let mut checkpoints: Vec<Value> = Vec::new();
                let mut completed: Option<Value> = None;
                let mut events = Box::pin(parse_sse_response(response));
                while let Some(event) = events.next().await {
                    let event = match event {
                        Ok(event) => event,
                        Err(error) => {
                            return Err(CompactionRequestError::new(error.message, 0, None));
                        }
                    };
                    let event_type = event.get("type").and_then(Value::as_str).unwrap_or_default().to_string();
                    observe_event(&observation_options, &event);
                    if event_type == "response.output_item.done" {
                        if let Some(item) = event.get("item") {
                            if item.is_object() {
                                let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
                                if item_type == "compaction" || item_type == "compaction_summary" {
                                    let mut checkpoint = item.clone();
                                    if let Some(object) = checkpoint.as_object_mut() {
                                        object.insert("type".to_string(), Value::String("compaction".to_string()));
                                    }
                                    checkpoints.push(checkpoint);
                                }
                            }
                        }
                    } else if event_type == "response.completed" || event_type == "response.done" {
                        completed = event.get("response").cloned();
                        break;
                    } else if ["error", "response.failed", "response.incomplete"].contains(&event_type.as_str()) {
                        let failed_response_code = event
                            .get("response")
                            .and_then(|response| response.get("error"))
                            .and_then(|error| error.get("code"))
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        let error_code = event
                            .get("error")
                            .and_then(|error| error.get("code"))
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        let code = error_code
                            .or(failed_response_code)
                            .or_else(|| event.get("code").and_then(Value::as_str).map(str::to_string));
                        match code.as_deref() {
                            // `return undefined` - the caller keeps its history.
                            Some("context_length_exceeded") => return Ok(Value::Null),
                            Some("rate_limit_exceeded") => {
                                return Err(CompactionRequestError::new(
                                    "Server compaction stream failed",
                                    429,
                                    None,
                                ))
                            }
                            Some("server_error") => {
                                return Err(CompactionRequestError::new(
                                    "Server compaction stream failed",
                                    503,
                                    None,
                                ))
                            }
                            _ => {
                                return Err(CompactionRequestError::new(
                                    "Server compaction stream failed before completion",
                                    0,
                                    None,
                                ))
                            }
                        }
                    }
                }
                let Some(completed) = completed else {
                    return Err(CompactionRequestError::new(
                        "Server compaction stream closed before completion",
                        502,
                        None,
                    ));
                };
                let status = completed.get("status");
                let status_is_incomplete = status
                    .map(|status| status.is_null() || status.as_str() != Some("completed"))
                    .unwrap_or(false);
                if status_is_incomplete || checkpoints.len() != 1 {
                    return Err(CompactionRequestError::new(
                        "Server compaction requires a completed response with exactly one encrypted checkpoint",
                        0,
                        None,
                    ));
                }
                let mut payload = Map::new();
                payload.insert(
                    "output".to_string(),
                    Value::Array(build_codex_compacted_window(&input, &checkpoints[0])),
                );
                payload.insert(
                    "usage".to_string(),
                    completed.get("usage").cloned().unwrap_or(Value::Null),
                );
                Ok(Value::Object(payload))
            })
        },
    );

    let result = request_openai_compaction(
        model,
        &resolve_codex_url(Some(&model.base_url)),
        &headers,
        body,
        options,
        Some(decode),
    )
    .await
    .map_err(|error| error.message)?;
    let mut result = result;
    if let Some(result) = result.as_mut() {
        if let Some(usage) = result.usage.as_mut() {
            let service_tier = options
                .and_then(|options| options.simple.stream.service_tier.clone())
                .flatten();
            apply_service_tier_pricing(usage, service_tier.as_deref(), model);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// streamOpenAICodexResponses
// ---------------------------------------------------------------------------

/// `streamOpenAICodexResponses: StreamFunction<"openai-codex-responses", OpenAICodexResponsesOptions>`.
pub fn stream_openai_codex_responses(
    model: &Model,
    context: &Context,
    options: Option<OpenAICodexResponsesOptions>,
) -> AssistantMessageEventStream {
    // TS: `registerSessionResourceCleanup(closeOpenAICodexWebSocketSessions)` runs at
    // module load; Rust registers it on first use.
    ensure_web_socket_session_cleanup_registered();
    let stream = create_assistant_message_event_stream();
    let out = stream.clone();
    let model = model.clone();
    let context = context.clone();
    let options = options.unwrap_or_default();
    tokio::spawn(async move {
        let mut output = AssistantMessage::new(
            "openai-codex-responses".to_string(),
            model.provider.clone(),
            model.id.clone(),
            now_ms(),
        );
        output.usage = Usage::zero();

        match run_openai_codex_responses(&model, &context, &options, &mut output, &out).await {
            Ok(()) => {}
            Err(error) => {
                // partialJson is only a streaming scratch buffer; never persist it.
                // (The port keeps it outside the block, so there is nothing to delete.)
                let aborted = options
                    .stream
                    .signal
                    .as_ref()
                    .map(|signal| signal.is_cancelled())
                    .unwrap_or(false);
                output.stop_reason = if aborted { "aborted".to_string() } else { "error".to_string() };
                output.error_message = Some(error.message.clone());
                record_stream_failure(&model, &mut output, &error.to_thrown());
                out.push(AssistantMessageEvent::Error {
                    reason: output.stop_reason.clone(),
                    error: output.clone(),
                });
                out.end(None);
            }
        }
    });
    stream
}

/// The TypeScript async IIFE body of `streamOpenAICodexResponses`.
async fn run_openai_codex_responses(
    model: &Model,
    context: &Context,
    options: &OpenAICodexResponsesOptions,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> Result<(), CodexThrown> {
    let api_key = options
        .stream
        .api_key
        .clone()
        .or_else(|| get_env_api_key(&model.provider))
        .unwrap_or_default();
    if api_key.is_empty() {
        return Err(CodexThrown::error(format!(
            "No API key for provider: {}",
            model.provider
        )));
    }

    let account_id = extract_account_id(&api_key)?;
    let mut body = build_request_body(model, context, Some(options)).map_err(CodexThrown::error)?;
    if let Some(on_payload) = options.stream.on_payload.clone() {
        let next_body = on_payload(Value::Object(body.clone()), model).await;
        if let Some(next_body) = next_body {
            body = match next_body {
                Value::Object(map) => map,
                _ => Map::new(),
            };
        }
    }
    let websocket_request_id = options
        .stream
        .session_id
        .clone()
        .unwrap_or_else(create_codex_request_id);
    let sse_headers = build_sse_headers(
        model.headers.as_ref(),
        options.stream.headers.as_ref(),
        &account_id,
        &api_key,
        options.stream.session_id.as_deref(),
    );
    let websocket_headers = build_web_socket_headers(
        model.headers.as_ref(),
        options.stream.headers.as_ref(),
        &account_id,
        &api_key,
        &websocket_request_id,
    );
    let body_json = serde_json::to_string(&Value::Object(body.clone())).unwrap_or_default();
    let transport = options.stream.transport.clone().unwrap_or_else(|| "auto".to_string());
    let websocket_disabled_for_session =
        transport != "sse" && is_web_socket_sse_fallback_active(options.stream.session_id.as_deref());
    if websocket_disabled_for_session {
        record_web_socket_sse_fallback(options.stream.session_id.as_deref());
    }

    if transport != "sse" && !websocket_disabled_for_session {
        let mut websocket_error: Option<CodexThrown> = None;
        // A dead cached connection gets one fresh attempt, but a partial turn
        // must never be replayed over either transport.
        for websocket_attempt in 0..2 {
            let websocket_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let websocket_url = resolve_codex_web_socket_url(Some(&model.base_url));
            let request = process_web_socket_stream(
                &websocket_url,
                &body,
                &websocket_headers,
                output,
                stream,
                model,
                {
                    let websocket_started = websocket_started.clone();
                    Arc::new(move || {
                        websocket_started.store(true, std::sync::atomic::Ordering::SeqCst);
                    })
                },
                options,
            );
            let result = match options.stream.timeout_ms {
                Some(timeout) => tokio::time::timeout(
                    std::time::Duration::from_millis(timeout.max(0.0) as u64),
                    request,
                )
                .await
                .unwrap_or_else(|_| Err(CodexThrown::error("Request timed out"))),
                None => request.await,
            };
            match result {
                Ok(()) => {
                    if options
                        .stream
                        .signal
                        .as_ref()
                        .map(|signal| signal.is_cancelled())
                        .unwrap_or(false)
                    {
                        return Err(CodexThrown::error("Request was aborted"));
                    }
                    stream.push(AssistantMessageEvent::Done {
                        reason: output.stop_reason.clone(),
                        message: output.clone(),
                    });
                    stream.end(None);
                    return Ok(());
                }
                Err(error) => {
                    let aborted = options.stream.signal.as_ref().is_some_and(|signal| signal.is_cancelled());
                    if !aborted && websocket_attempt == 0
                        && !websocket_started.load(std::sync::atomic::Ordering::SeqCst)
                        && is_stale_codex_continuation_error(&error)
                    {
                        // process_web_socket_stream discarded the failed chain
                        // and connection. Retry the original full body once.
                        output.response_id = None;
                        continue;
                    }
                    if options
                        .stream
                        .signal
                        .as_ref()
                        .map(|signal| signal.is_cancelled())
                        .unwrap_or(false)
                        || is_codex_non_transport_error(&error)
                        || error.message == "Request timed out"
                    {
                        return Err(error);
                    }
                    if websocket_started.load(std::sync::atomic::Ordering::SeqCst) {
                        let mut details = Map::new();
                        details.insert("configuredTransport".to_string(), Value::String(transport.clone()));
                        details.insert("eventsEmitted".to_string(), Value::Bool(true));
                        details.insert(
                            "phase".to_string(),
                            Value::String("after_message_stream_start".to_string()),
                        );
                        details.insert(
                            "requestBytes".to_string(),
                            Value::Number((utf8_byte_length(&body_json) as i64).into()),
                        );
                        append_diagnostic(output, "provider_transport_failure", &error, details);
                        record_web_socket_failure(options.stream.session_id.as_deref(), &error);
                        return Err(error);
                    }
                    websocket_error = Some(error);
                }
            }
        }
        let websocket_error = websocket_error.unwrap_or_else(|| CodexThrown::error("WebSocket error"));
        let mut details = Map::new();
        details.insert("configuredTransport".to_string(), Value::String(transport.clone()));
        details.insert("fallbackTransport".to_string(), Value::String("sse".to_string()));
        details.insert("eventsEmitted".to_string(), Value::Bool(false));
        details.insert(
            "phase".to_string(),
            Value::String("before_message_stream_start".to_string()),
        );
        details.insert(
            "requestBytes".to_string(),
            Value::Number((utf8_byte_length(&body_json) as i64).into()),
        );
        details.insert("websocketReconnectAttempts".to_string(), Value::Number(1.into()));
        append_diagnostic(output, "provider_transport_failure", &websocket_error, details);
        record_web_socket_failure(options.stream.session_id.as_deref(), &websocket_error);
        record_web_socket_sse_fallback(options.stream.session_id.as_deref());
    }

    if options
        .stream
        .signal
        .as_ref()
        .map(|signal| signal.is_cancelled())
        .unwrap_or(false)
    {
        return Err(CodexThrown::error("Request was aborted"));
    }

    let response = match fetch_codex_response(&resolve_codex_url(Some(&model.base_url)), &sse_headers, &body_json, options).await {
        Ok(response) => response,
        Err(error) => {
            let aborted = options
                .stream
                .signal
                .as_ref()
                .map(|signal| signal.is_cancelled())
                .unwrap_or(false);
            if aborted || error.message == "Request was aborted" {
                return Err(CodexThrown::error("Request was aborted"));
            }
            return Err(error);
        }
    };
    if let Some(on_response) = options.stream.on_response.clone() {
        let mut headers = header_map_to_record(response.headers());
        // Codex has a native WebSocket path and an SSE fallback. The SSE response is a
        // real HTTP header edge, so label it: without a label Codex attempts recorded
        // `transport_websocket = null` in every case and a WS-vs-SSE comparison was
        // impossible (B5).
        headers.insert("x-optimus-transport".into(), "sse".into());
        on_response(
            ProviderResponse {
                status: response.status().as_u16() as i64,
                headers,
            },
            model,
        )
        .await;
    }

    if !response.status().is_success() {
        let parse = parse_error_response(response);
        let error = match options.stream.signal.as_ref() {
            Some(signal) => tokio::select! {
                biased;
                _ = signal.cancelled() => return Err(CodexThrown::error("Request was aborted")),
                result = parse => result,
            },
            None => parse.await,
        };
        return Err(error);
    }

    stream.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });
    process_stream(response, output, stream, model, options).await?;

    if options
        .stream
        .signal
        .as_ref()
        .map(|signal| signal.is_cancelled())
        .unwrap_or(false)
    {
        return Err(CodexThrown::error("Request was aborted"));
    }

    stream.push(AssistantMessageEvent::Done {
        reason: output.stop_reason.clone(),
        message: output.clone(),
    });
    stream.end(None);
    Ok(())
}

/// `streamSimpleOpenAICodexResponses: StreamFunction<"openai-codex-responses", SimpleStreamOptions>`.
pub fn stream_simple_openai_codex_responses(
    model: &Model,
    context: &Context,
    options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let api_key = options
        .as_ref()
        .and_then(|options| options.stream.api_key.clone())
        .or_else(|| get_env_api_key(&model.provider));
    let Some(api_key) = api_key else {
        // openai-codex-responses.ts:350-352 `if (!apiKey) { throw new Error(...) }` - the caller
        // gets a catchable error with this text, not a process abort. A Rust `StreamFunction`
        // returns a stream, so the failure is delivered as the terminal `error` event instead.
        return api_key_error_stream(model, &format!("No API key for provider: {}", model.provider));
    };

    let base = build_base_options(model, options.as_ref(), Some(&api_key));
    let clamped_reasoning = options
        .as_ref()
        .and_then(|options| options.reasoning.clone())
        .map(|reasoning| clamp_thinking_level(model, &reasoning));
    let reasoning_effort = clamped_reasoning.filter(|reasoning| reasoning != "off");

    let mut typed = OpenAICodexResponsesOptions::from_base(&base);
    typed.reasoning_effort = reasoning_effort;
    stream_openai_codex_responses(model, context, Some(typed))
}

/// The `error` event openai-codex-responses.ts:351 gets from throwing
/// "No API key for provider: ..." out of `streamSimpleOpenAICodexResponses`.
fn api_key_error_stream(model: &Model, message: &str) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let mut output = AssistantMessage::new(
        "openai-codex-responses".to_string(),
        model.provider.clone(),
        model.id.clone(),
        now_ms(),
    );
    output.usage = Usage::zero();
    output.stop_reason = "error".to_string();
    output.error_message = Some(message.to_string());
    record_stream_failure(model, &mut output, &ThrownStreamError::Message(message));
    stream.push(AssistantMessageEvent::Error {
        reason: output.stop_reason.clone(),
        error: output,
    });
    stream.end(None);
    stream
}

/// `processStream(response, output, stream, model, options?)`.
async fn process_stream(
    response: reqwest::Response,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    model: &Model,
    options: &OpenAICodexResponsesOptions,
) -> Result<(), CodexThrown> {
    let codex_error: Arc<Mutex<Option<CodexThrown>>> = Arc::new(Mutex::new(None));
    let events: ResponsesEventStream = Box::pin(map_codex_events(
        parse_sse_response(response).inspect({
            let observation_options = options.stream.clone();
            move |event| {
                if let Ok(event) = event { observe_event(&observation_options, event); }
            }
        }),
        codex_error.clone(),
    ));
    let stream_options = OpenAIResponsesStreamOptions {
        on_output_item_done: options.on_output_item_done.clone(),
        // openai-codex-responses.ts:478 `serviceTier: options?.serviceTier`: the request-side tier
        // the process loop falls back to (`response?.service_tier ?? options.serviceTier`,
        // openai-responses-shared.ts:526) before `resolveServiceTier` (codex line 479-480) maps a
        // "default" response back onto a requested "flex"/"priority".
        service_tier: options.service_tier.clone().flatten(),
        resolve_service_tier: Some(Arc::new(|response_tier, request_tier| {
            resolve_codex_service_tier(response_tier, request_tier)
        })),
        apply_service_tier_pricing: Some(Arc::new({
            let model = model.clone();
            move |usage: &mut Usage, service_tier: Option<&str>| {
                apply_service_tier_pricing(usage, service_tier, &model)
            }
        })),
        on_usage_observation: options.stream.on_usage_observation.clone(),
    };
    // The response headers may arrive long before another body chunk. Dropping
    // this future on cancellation releases the SSE body and retains partial output.
    let parse = process_responses_stream(events, output, stream, model, Some(&stream_options));
    let result = match options.stream.signal.as_ref() {
        Some(signal) => tokio::select! {
            biased;
            _ = signal.cancelled() => return Err(CodexThrown::error("Request was aborted")),
            result = parse => result,
        },
        None => parse.await,
    };
    if let Some(error) = codex_error.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take() {
        return Err(error);
    }
    result.map_err(|error| match error {
        crate::providers::openai_responses_shared::ResponsesStreamError::StreamFailure(failure) => {
            CodexThrown::api_error(
                failure.message,
                failure.info.provider_error_type.clone(),
                failure.info.status,
                failure.info.retry_after_ms,
                None,
            )
        }
        crate::providers::openai_responses_shared::ResponsesStreamError::Message(message) => {
            CodexThrown::error(message)
        }
    })
}

// ---------------------------------------------------------------------------
// Event mapping and SSE parsing
// ---------------------------------------------------------------------------

/// `mapCodexEvents(events)`.
///
/// The TypeScript generator `throw`s; a Rust stream cannot, so the thrown value is stored
/// in `error_slot` and the stream ends. The caller re-raises it after the shared
/// `processResponsesStream` loop returns (same catch boundary, same message).
pub fn map_codex_events<S>(
    events: S,
    error_slot: Arc<Mutex<Option<CodexThrown>>>,
) -> Pin<Box<dyn futures::Stream<Item = Value> + Send>>
where
    S: futures::Stream<Item = Result<Value, CodexThrown>> + Send + 'static,
{
    Box::pin(futures::stream::unfold(
        (
            Box::pin(events) as Pin<Box<dyn futures::Stream<Item = Result<Value, CodexThrown>> + Send>>,
            error_slot,
        ),
        |(mut events, error_slot)| async move {
            loop {
                let event = match events.next().await {
                    None => return None,
                    Some(Err(error)) => {
                        *error_slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error);
                        return None;
                    }
                    Some(Ok(event)) => event,
                };
                let type_ = event.get("type").and_then(Value::as_str).map(str::to_string);
                let Some(type_) = type_ else {
                    continue;
                };

                if type_ == "error" {
                    // Errors arrive flat ({ code, message }) or nested under event.error.
                    let flat_code = event
                        .get("code")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let flat_message = event
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let nested = event.get("error").filter(|error| error.is_object());
                    let status = event.get("status_code").and_then(Value::as_i64);
                    let code = if !flat_code.is_empty() {
                        Some(flat_code.clone())
                    } else {
                        nested
                            .and_then(|nested| {
                                nested.get("code").or_else(|| nested.get("type")).and_then(Value::as_str)
                            })
                            .map(str::to_string)
                    };
                    let usage_limit = nested.and_then(|nested| codex_usage_limit_message(nested, status));
                    let message = if !flat_message.is_empty() {
                        flat_message
                    } else {
                        nested
                            .and_then(|nested| nested.get("message"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string()
                    };
                    let fallback = format!(
                        "Codex error: {}",
                        if !message.is_empty() {
                            message
                        } else if let Some(code) = code.clone() {
                            code
                        } else {
                            serde_json::to_string(&event).unwrap_or_default()
                        }
                    );
                    let thrown = CodexThrown::api_error(
                        usage_limit
                            .as_ref()
                            .map(|limit| limit.friendly_message.clone())
                            .unwrap_or(fallback),
                        code,
                        status,
                        usage_limit.and_then(|limit| limit.retry_after_ms),
                        Some(event),
                    );
                    *error_slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(thrown);
                    return None;
                }

                if type_ == "response.failed" {
                    let response = event.get("response");
                    let code = response
                        .and_then(|response| response.get("error"))
                        .and_then(|error| error.get("code"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    let message = response
                        .and_then(|response| response.get("error"))
                        .and_then(|error| error.get("message"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    let thrown = CodexThrown::api_error(
                        message.unwrap_or_else(|| "Codex response failed".to_string()),
                        code,
                        None,
                        None,
                        Some(event),
                    );
                    *error_slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(thrown);
                    return None;
                }

                if type_ == "response.done" || type_ == "response.completed" || type_ == "response.incomplete" {
                    let response = event.get("response").cloned();
                    let normalized_response = response.map(|response| {
                        let status = if type_ == "response.incomplete" {
                            Some("incomplete".to_string())
                        } else {
                            normalize_codex_status(response.get("status"))
                        };
                        let mut normalized = response;
                        if let Some(object) = normalized.as_object_mut() {
                            object.insert(
                                "status".to_string(),
                                match status {
                                    Some(status) => Value::String(status),
                                    None => Value::Null,
                                },
                            );
                        }
                        normalized
                    });
                    let mut output = event;
                    if let Some(object) = output.as_object_mut() {
                        object.insert("type".to_string(), Value::String("response.completed".to_string()));
                        object.insert("response".to_string(), normalized_response.unwrap_or(Value::Null));
                    }
                    return Some((output, (events, error_slot)));
                }

                return Some((event, (events, error_slot)));
            }
        },
    ))
}

/// `parseSSE(response)`: the Codex SSE frame reader.
pub fn parse_sse_response(
    response: reqwest::Response,
) -> Pin<Box<dyn futures::Stream<Item = Result<Value, CodexThrown>> + Send>> {
    parse_sse_chunks(Box::pin(response.bytes_stream()))
}

fn parse_sse_chunks(
    byte_stream: Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
) -> Pin<Box<dyn futures::Stream<Item = Result<Value, CodexThrown>> + Send>> {
    Box::pin(futures::stream::unfold(
        (
            byte_stream,
            SseFrames::default(),
            VecDeque::new(),
            false,
        ),
        |(mut bytes, mut buffer, mut queue, mut finished)| async move {
            loop {
                if let Some(event) = queue.pop_front() {
                    return Some((event, (bytes, buffer, queue, finished)));
                }
                if finished {
                    return None;
                }
                match bytes.next().await {
                    None => {
                        finished = true;
                        continue;
                    }
                    Some(Err(error)) => {
                        finished = true;
                        queue.push_back(Err(CodexThrown::error(error.to_string())));
                        continue;
                    }
                    Some(Ok(chunk)) => {
                        for data in buffer.push(&chunk) {
                            match serde_json::from_str::<Value>(&data) {
                                Ok(parsed) => queue.push_back(Ok(parsed)),
                                Err(cause) => {
                                    let thrown = CodexThrown::protocol_error(
                                        format!(
                                            "Invalid Codex SSE JSON: {}",
                                            format_thrown_value(&ThrownValue::Text(&cause.to_string()))
                                        ),
                                        Some(Value::String(data)),
                                    );
                                    finished = true;
                                    queue.push_back(Err(thrown));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        },
    ))
}

/// `fetch(resolveCodexUrl(model.baseUrl), { method: "POST", headers, body, signal })`.
async fn fetch_codex_response(
    url: &str,
    headers: &IndexMap<String, String>,
    body_json: &str,
    options: &OpenAICodexResponsesOptions,
) -> Result<reqwest::Response, CodexThrown> {
    let mut header_map = reqwest::header::HeaderMap::new();
    for (key, value) in headers {
        if let (Ok(name), Ok(header_value)) = (
            reqwest::header::HeaderName::from_bytes(key.as_bytes()),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            header_map.insert(name, header_value);
        }
    }
    let mut request = reqwest::Client::new()
        .post(url)
        .headers(header_map)
        .body(body_json.to_string());
    if let Some(timeout_ms) = options.stream.timeout_ms {
        request = request.timeout(std::time::Duration::from_millis(timeout_ms.max(0.0) as u64));
    }

    let send = request.send();
    let response = match options.stream.signal.as_ref() {
        Some(signal) => tokio::select! {
            _ = signal.cancelled() => return Err(CodexThrown::error("Request was aborted")),
            result = send => result,
        },
        None => send.await,
    };
    response.map_err(|error| CodexThrown::error(error.to_string()))
}

/// `parseErrorResponse(response)`.
pub async fn parse_error_response(response: reqwest::Response) -> CodexThrown {
    let status = response.status().as_u16() as i64;
    let status_text = response.status().canonical_reason().unwrap_or_default().to_string();
    let headers = headers_to_value(response.headers());
    let raw = response.text().await.unwrap_or_default();
    parse_error_body(status, &status_text, &headers, &raw)
}

/// [`parse_error_response`] without the HTTP response, so the mapping is testable.
pub fn parse_error_body(status: i64, status_text: &str, headers: &Value, raw: &str) -> CodexThrown {
    let retry_after_header = parse_retry_after_ms(Some(headers));
    let mut message = if raw.is_empty() { status_text.to_string() } else { raw.to_string() };
    if message.is_empty() {
        message = "Request failed".to_string();
    }
    let mut code: Option<String> = None;
    let mut retry_after_ms = retry_after_header;

    // `try { ... } catch { /* Unparseable error body: fall back to the raw message. */ }`
    if let Ok(parsed) = serde_json::from_str::<Value>(&raw) {
        let err = parsed.get("error");
        if let Some(err) = err.filter(|error| !error.is_null()) {
            code = err
                .get("code")
                .and_then(Value::as_str)
                .or_else(|| err.get("type").and_then(Value::as_str))
                .map(str::to_string);
            match codex_usage_limit_message(err, Some(status)) {
                Some(usage_limit) => {
                    message = usage_limit.friendly_message;
                    // Neither server delay (Retry-After header, resets_at body) may undercut the other.
                    if let Some(usage_retry_after_ms) = usage_limit.retry_after_ms {
                        retry_after_ms = Some(retry_after_ms.unwrap_or(0.0).max(usage_retry_after_ms));
                    }
                }
                None => {
                    if let Some(err_message) = err.get("message").and_then(Value::as_str) {
                        if !err_message.is_empty() {
                            message = err_message.to_string();
                        }
                    }
                }
            }
        }
    }

    CodexThrown::api_error(message, code, Some(status), retry_after_ms, None)
}

/// The `Headers` object as the shared `parseRetryAfterMs` expects it.
fn headers_to_value(headers: &reqwest::header::HeaderMap) -> Value {
    let mut object = Map::new();
    for (key, value) in headers.iter() {
        object.insert(
            key.as_str().to_string(),
            Value::String(value.to_str().unwrap_or_default().to_string()),
        );
    }
    Value::Object(object)
}

/// `extractAccountId(token)`.
pub fn extract_account_id(token: &str) -> Result<String, CodexThrown> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(CodexThrown::error("Failed to extract accountId from token"));
    }
    let decoded = match base64_decode_jwt_part(parts[1]) {
        Some(decoded) => decoded,
        None => return Err(CodexThrown::error("Failed to extract accountId from token")),
    };
    let payload: Value = match serde_json::from_slice(&decoded) {
        Ok(payload) => payload,
        Err(_) => return Err(CodexThrown::error("Failed to extract accountId from token")),
    };
    match payload
        .get(JWT_CLAIM_PATH)
        .and_then(|claims| claims.get("chatgpt_account_id"))
        .and_then(Value::as_str)
    {
        Some(account_id) if !account_id.is_empty() => Ok(account_id.to_string()),
        _ => Err(CodexThrown::error("Failed to extract accountId from token")),
    }
}

/// `atob(parts[1])` for the JWT payload (standard and URL-safe alphabets).
fn base64_decode_jwt_part(part: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    let normalized: String = part.chars().map(|ch| if ch == '-' { '+' } else if ch == '_' { '/' } else { ch }).collect();
    let padded = match normalized.len() % 4 {
        2 => format!("{}==", normalized),
        3 => format!("{}=", normalized),
        _ => normalized,
    };
    base64::engine::general_purpose::STANDARD.decode(padded).ok()
}

/// `createCodexRequestId()`.
pub fn create_codex_request_id() -> String {
    if std::env::var("CODEX_REQUEST_ID_UUID_V4").is_ok() {
        return uuid::Uuid::new_v4().to_string();
    }
    let millis = now_ms();
    let random = uuid::Uuid::new_v4().simple().to_string();
    format!("codex_{}_{}", millis, &random[..8])
}

/// `buildBaseCodexHeaders(initHeaders, additionalHeaders, accountId, token)`.
pub fn build_base_codex_headers(
    init_headers: Option<&IndexMap<String, String>>,
    additional_headers: Option<&IndexMap<String, String>>,
    account_id: &str,
    token: &str,
) -> IndexMap<String, String> {
    let mut headers: IndexMap<String, String> = init_headers.cloned().unwrap_or_default();
    if let Some(additional_headers) = additional_headers {
        for (key, value) in additional_headers {
            headers.insert(key.clone(), value.clone());
        }
    }
    headers.insert("Authorization".to_string(), format!("Bearer {}", token));
    headers.insert("chatgpt-account-id".to_string(), account_id.to_string());
    headers.insert("originator".to_string(), "pi".to_string());
    // openai-codex-responses.ts:1347 `pi (${_os.platform()} ${_os.release()}; ${_os.arch()})`.
    // `_os` is loaded lazily in the TypeScript; the Rust port resolves the same values from
    // `std::env::consts` and `sysinfo` instead.
    let user_agent = format!("pi ({} {}; {})", os_platform(), os_release(), std::env::consts::ARCH);
    headers.insert("User-Agent".to_string(), user_agent);
    headers
}

/// `_os.platform()`.
///
/// The Node values are the `process.platform` names: "darwin", "win32", "linux", ... while
/// `std::env::consts::OS` yields "macos" and "windows" for the first two.
fn os_platform() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    }
}

/// `_os.release()` - the `uname.release` string on POSIX (sysinfo reads the same kernel
/// fields), falling back to the OS version and finally "unknown" when the platform exposes
/// neither.
fn os_release() -> String {
    sysinfo::System::kernel_version()
        .or_else(sysinfo::System::os_version)
        .filter(|release| !release.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// `buildSSEHeaders(initHeaders, additionalHeaders, accountId, token, sessionId?)`.
pub fn build_sse_headers(
    init_headers: Option<&IndexMap<String, String>>,
    additional_headers: Option<&IndexMap<String, String>>,
    account_id: &str,
    token: &str,
    session_id: Option<&str>,
) -> IndexMap<String, String> {
    let mut headers = build_base_codex_headers(init_headers, additional_headers, account_id, token);
    headers.insert("OpenAI-Beta".to_string(), "responses=experimental".to_string());
    headers.insert("accept".to_string(), "text/event-stream".to_string());
    headers.insert("content-type".to_string(), "application/json".to_string());

    if let Some(session_id) = session_id {
        headers.insert("session_id".to_string(), session_id.to_string());
        headers.insert("x-client-request-id".to_string(), session_id.to_string());
    }

    headers
}

/// `buildWebSocketHeaders(initHeaders, additionalHeaders, accountId, token, requestId)`.
pub fn build_web_socket_headers(
    init_headers: Option<&IndexMap<String, String>>,
    additional_headers: Option<&IndexMap<String, String>>,
    account_id: &str,
    token: &str,
    request_id: &str,
) -> IndexMap<String, String> {
    let mut headers = build_base_codex_headers(init_headers, additional_headers, account_id, token);
    remove_header(&mut headers, "accept");
    remove_header(&mut headers, "content-type");
    remove_header(&mut headers, "OpenAI-Beta");
    remove_header(&mut headers, "openai-beta");
    headers.insert(
        "OpenAI-Beta".to_string(),
        OPENAI_BETA_RESPONSES_WEBSOCKETS.to_string(),
    );
    headers.insert("x-client-request-id".to_string(), request_id.to_string());
    headers.insert("session_id".to_string(), request_id.to_string());
    headers
}

/// `headers.delete(name)` - `Headers` is case-insensitive.
fn remove_header(headers: &mut IndexMap<String, String>, name: &str) {
    let key = headers
        .keys()
        .find(|key| key.eq_ignore_ascii_case(name))
        .cloned();
    if let Some(key) = key {
        headers.shift_remove(&key);
    }
}

// ---------------------------------------------------------------------------
// WebSocket transport
// ---------------------------------------------------------------------------

/// `interface WebSocketLike`.
pub trait WebSocketLike: Send + Sync {
    fn close(&self, code: Option<i32>, reason: Option<&str>);
    fn send(&self, data: &str);
    /// `readyState` - `None` when the runtime does not expose it.
    fn ready_state(&self) -> Option<i32>;
    /// `socket.addEventListener(type, listener)`.
    fn add_event_listener(&self, type_: WebSocketEventType, listener: WebSocketListener);
    /// `socket.removeEventListener(type, listener)`.
    fn remove_event_listener(&self, type_: WebSocketEventType, listener: &WebSocketListener);
}

/// `type WebSocketEventType = "open" | "message" | "error" | "close"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WebSocketEventType {
    Open,
    Message,
    Error,
    Close,
}

/// `type WebSocketListener = (event: unknown) => void`.
pub type WebSocketListener = Arc<dyn Fn(Value) + Send + Sync>;

/// `interface WebSocketLike` constructor: `new WebSocketCtor(url, { headers })`.
pub type WebSocketConstructor = Arc<dyn Fn(&str, IndexMap<String, String>) -> Arc<dyn WebSocketLike> + Send + Sync>;

fn web_socket_constructor_slot() -> &'static Mutex<Option<WebSocketConstructor>> {
    static SLOT: OnceLock<Mutex<Option<WebSocketConstructor>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(Some(native_web_socket::constructor())))
}

/// Override the native factory. `None` explicitly disables WebSocket IO (for embedders/tests).
pub fn set_web_socket_constructor(constructor: Option<WebSocketConstructor>) {
    if let Ok(mut slot) = web_socket_constructor_slot().lock() {
        *slot = constructor;
    }
}

/// `getWebSocketConstructor()`.
pub fn get_web_socket_constructor() -> Option<WebSocketConstructor> {
    web_socket_constructor_slot()
        .lock()
        .ok()
        .and_then(|slot| slot.clone())
}

/// `interface CachedWebSocketContinuationState`.
#[derive(Debug, Clone, Default)]
pub struct CachedWebSocketContinuationState {
    pub last_request_body: RequestBody,
    pub last_response_id: String,
    pub last_response_items: Vec<Value>,
    pub socket_identity: usize,
}

/// `interface CachedWebSocketConnection`.
pub struct CachedWebSocketConnection {
    pub socket: Arc<dyn WebSocketLike>,
    pub busy: bool,
    /// `idleTimer?: ReturnType<typeof setTimeout>`
    pub idle_timer: Option<tokio::task::JoinHandle<()>>,
    pub continuation: Option<CachedWebSocketContinuationState>,
    /// Reusing a session ID must not reuse another endpoint or account's connection.
    connection_identity: String,
}

/// `export interface OpenAICodexWebSocketDebugStats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAICodexWebSocketDebugStats {
    pub requests: i64,
    pub connections_created: i64,
    pub connections_reused: i64,
    pub cached_context_requests: i64,
    pub store_true_requests: i64,
    pub full_context_requests: i64,
    pub delta_requests: i64,
    pub last_input_items: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_delta_input_items: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_previous_response_id: Option<String>,
    pub websocket_failures: i64,
    pub sse_fallbacks: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub websocket_fallback_active: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_web_socket_error: Option<String>,
}

fn web_socket_session_cache() -> &'static Mutex<HashMap<String, Arc<Mutex<CachedWebSocketConnection>>>> {
    static SLOT: OnceLock<Mutex<HashMap<String, Arc<Mutex<CachedWebSocketConnection>>>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(HashMap::new()))
}

fn web_socket_debug_stats() -> &'static Mutex<HashMap<String, OpenAICodexWebSocketDebugStats>> {
    static SLOT: OnceLock<Mutex<HashMap<String, OpenAICodexWebSocketDebugStats>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(HashMap::new()))
}

fn web_socket_sse_fallback_sessions() -> &'static Mutex<std::collections::HashSet<String>> {
    static SLOT: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

/// `getOrCreateWebSocketDebugStats(sessionId)`.
fn get_or_create_web_socket_debug_stats(session_id: &str) -> OpenAICodexWebSocketDebugStats {
    let mut stats = web_socket_debug_stats().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    stats.entry(session_id.to_string()).or_default().clone()
}

/// Writes the stats back after an in-place mutation, mirroring the TS shared object.
fn update_web_socket_debug_stats<F: FnOnce(&mut OpenAICodexWebSocketDebugStats)>(
    session_id: &str,
    update: F,
) {
    let mut stats = web_socket_debug_stats().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = stats.entry(session_id.to_string()).or_default();
    update(entry);
}

/// `export function getOpenAICodexWebSocketDebugStats(sessionId)`.
pub fn get_openai_codex_web_socket_debug_stats(session_id: &str) -> Option<OpenAICodexWebSocketDebugStats> {
    web_socket_debug_stats()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(session_id)
        .cloned()
}

/// `export function resetOpenAICodexWebSocketDebugStats(sessionId?)`.
pub fn reset_openai_codex_web_socket_debug_stats(session_id: Option<&str>) {
    match session_id {
        Some(session_id) => {
            web_socket_debug_stats()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(session_id);
            web_socket_sse_fallback_sessions()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(session_id);
        }
        None => {
            web_socket_debug_stats()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
            web_socket_sse_fallback_sessions()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
        }
    }
}

/// `closeWebSocketSilently(socket, code = 1000, reason = "done")`.
fn close_web_socket_silently(socket: &Arc<dyn WebSocketLike>, code: i32, reason: &str) {
    socket.close(Some(code), Some(reason));
}

/// `closeOpenAICodexWebSocketSessions(sessionId?)`.
pub fn close_openai_codex_web_socket_sessions(session_id: Option<&str>) {
    let close_entry = |entry: &Arc<Mutex<CachedWebSocketConnection>>| {
        let mut entry = entry.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(idle_timer) = entry.idle_timer.take() {
            idle_timer.abort();
        }
        close_web_socket_silently(&entry.socket, 1000, "debug_close");
    };
    let mut cache = web_socket_session_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match session_id {
        Some(session_id) => {
            if let Some(entry) = cache.get(session_id) {
                close_entry(entry);
            }
            cache.remove(session_id);
        }
        None => {
            for entry in cache.values() {
                close_entry(entry);
            }
            cache.clear();
        }
    }
}

/// `registerSessionResourceCleanup(closeOpenAICodexWebSocketSessions)`.
fn register_web_socket_session_cleanup() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _cleanup = register_session_resource_cleanup(Arc::new(|session_id: Option<&str>| {
            close_openai_codex_web_socket_sessions(session_id);
        }));
    });
}

/// The TypeScript runs `registerSessionResourceCleanup(closeOpenAICodexWebSocketSessions)`
/// at module load; Rust has no module initializer, so the stream entry points call this.
pub fn ensure_web_socket_session_cleanup_registered() {
    register_web_socket_session_cleanup();
}

/// `isWebSocketSseFallbackActive(sessionId)`.
fn is_web_socket_sse_fallback_active(session_id: Option<&str>) -> bool {
    match session_id {
        Some(session_id) => web_socket_sse_fallback_sessions()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(session_id),
        None => false,
    }
}

/// `recordWebSocketSseFallback(sessionId)`.
fn record_web_socket_sse_fallback(session_id: Option<&str>) {
    let Some(session_id) = session_id else {
        return;
    };
    let active = is_web_socket_sse_fallback_active(Some(session_id));
    update_web_socket_debug_stats(session_id, |stats| {
        stats.sse_fallbacks += 1;
        stats.websocket_fallback_active = Some(active);
    });
}

/// `recordWebSocketFailure(sessionId, error)`.
fn record_web_socket_failure(session_id: Option<&str>, error: &CodexThrown) {
    let Some(session_id) = session_id else {
        return;
    };
    web_socket_sse_fallback_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(session_id.to_string());

    let message = error.format();
    update_web_socket_debug_stats(session_id, |stats| {
        stats.websocket_failures += 1;
        stats.last_web_socket_error = Some(message.clone());
        stats.websocket_fallback_active = Some(true);
    });
}

/// `isWebSocketReusable(socket)`.
fn is_web_socket_reusable(socket: &Arc<dyn WebSocketLike>) -> bool {
    match socket.ready_state() {
        // If readyState is unavailable, assume the runtime keeps it open/reusable.
        None => true,
        Some(ready_state) => ready_state == 1,
    }
}

/// `scheduleSessionWebSocketExpiry(sessionId, entry)`.
fn schedule_session_web_socket_expiry(session_id: &str, entry: &Arc<Mutex<CachedWebSocketConnection>>) {
    let entry_for_timer = entry.clone();
    let session_id = session_id.to_string();
    let timer = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(SESSION_WEBSOCKET_CACHE_TTL_MS)).await;
        let busy = entry_for_timer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .busy;
        if busy {
            return;
        }
        {
            let entry = entry_for_timer.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            close_web_socket_silently(&entry.socket, 1000, "idle_timeout");
        }
        let mut cache = web_socket_session_cache().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.get(&session_id).map(|entry| Arc::ptr_eq(entry, &entry_for_timer)).unwrap_or(false) {
            cache.remove(&session_id);
        }
    });
    let mut entry = entry.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(previous) = entry.idle_timer.take() {
        previous.abort();
    }
    entry.idle_timer = Some(timer);
}

/// `connectWebSocket(url, headers, signal?)`.
async fn connect_web_socket(
    url: &str,
    headers: &IndexMap<String, String>,
    signal: Option<&tokio_util::sync::CancellationToken>,
) -> Result<Arc<dyn WebSocketLike>, CodexThrown> {
    let Some(web_socket_constructor) = get_web_socket_constructor() else {
        return Err(CodexThrown::error(
            "WebSocket transport is not available in this runtime",
        ));
    };

    // openai-codex-responses.ts:812-813 does `const wsHeaders = headersToRecord(headers);
    // delete wsHeaders["OpenAI-Beta"];`. `headersToRecord` (utils/headers.ts:1-6) copies
    // `Headers.entries()`, whose names the Fetch spec lowercases, and JS object `delete` is
    // case-sensitive - so the mixed-case key never matches the stored `openai-beta` entry and
    // the `OPENAI_BETA_RESPONSES_WEBSOCKETS` header set by `buildWebSocketHeaders`
    // (openai-codex-responses.ts:1384) DOES reach the WebSocket handshake. The port must keep
    // the same header, not strip it.
    let ws_headers = headers.clone();

    // The constructor throws synchronously when the runtime rejects the request.
    let socket = web_socket_constructor(url, ws_headers);

    let (sender, mut receiver) = tokio::sync::mpsc::channel::<Result<Arc<dyn WebSocketLike>, CodexThrown>>(1);
    let sender = Arc::new(Mutex::new(Some(sender)));

    let settle = {
        let sender = sender.clone();
        Arc::new(move |result: Result<Arc<dyn WebSocketLike>, CodexThrown>| {
            let mut sender = sender.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(sender) = sender.take() {
                let _ = sender.try_send(result);
            }
        })
    };

    let on_open: WebSocketListener = {
        let settle = settle.clone();
        let socket = socket.clone();
        Arc::new(move |_event: Value| {
            settle(Ok(socket.clone()));
        })
    };
    let on_error: WebSocketListener = {
        let settle = settle.clone();
        Arc::new(move |event: Value| {
            settle(Err(extract_web_socket_error(&event)));
        })
    };
    let on_close: WebSocketListener = {
        let settle = settle.clone();
        Arc::new(move |event: Value| {
            settle(Err(extract_web_socket_close_error(&event)));
        })
    };

    socket.add_event_listener(WebSocketEventType::Open, on_open.clone());
    socket.add_event_listener(WebSocketEventType::Error, on_error.clone());
    socket.add_event_listener(WebSocketEventType::Close, on_close.clone());

    let listeners = SocketListenerGuard {
        socket: socket.clone(),
        listeners: vec![
            (WebSocketEventType::Open, on_open),
            (WebSocketEventType::Error, on_error),
            (WebSocketEventType::Close, on_close),
        ],
        close_on_drop: true,
    };

    let result = match signal {
        Some(signal) => tokio::select! {
            biased;
            _ = signal.cancelled() => Err(CodexThrown::error("Request was aborted")),
            result = receiver.recv() => result.unwrap_or_else(|| Err(CodexThrown::error("WebSocket error"))),
        },
        None => receiver.recv().await.unwrap_or_else(|| Err(CodexThrown::error("WebSocket error"))),
    };
    let mut listeners = listeners;
    listeners.close_on_drop = result.is_err();

    result
}

struct SocketListenerGuard {
    socket: Arc<dyn WebSocketLike>,
    listeners: Vec<(WebSocketEventType, WebSocketListener)>,
    close_on_drop: bool,
}

impl Drop for SocketListenerGuard {
    fn drop(&mut self) {
        for (kind, listener) in &self.listeners {
            self.socket.remove_event_listener(*kind, listener);
        }
        if self.close_on_drop {
            close_web_socket_silently(&self.socket, 1000, "aborted");
        }
    }
}

/// `extractWebSocketError(event)`.
pub fn extract_web_socket_error(event: &Value) -> CodexThrown {
    if event.is_object() {
        if let Some(message) = event.get("message").and_then(Value::as_str) {
            if !message.is_empty() {
                return CodexThrown::error(message);
            }
        }

        if let Some(nested_error) = event.get("error") {
            if let Some(message) = nested_error.get("message").and_then(Value::as_str) {
                if !message.is_empty() {
                    return CodexThrown::error(message);
                }
            }
        }
    }
    CodexThrown::error("WebSocket error")
}

/// `extractWebSocketCloseError(event)`.
pub fn extract_web_socket_close_error(event: &Value) -> CodexThrown {
    if event.is_object() {
        let code = event.get("code").and_then(Value::as_i64).map(|code| code as i32);
        let reason = event.get("reason").and_then(Value::as_str).map(str::to_string);
        let was_clean = event.get("wasClean").and_then(Value::as_bool);
        let code_text = match code {
            Some(code) => format!(" {}", code),
            None => String::new(),
        };
        let mut reason_text = match reason.as_deref() {
            Some(reason) if !reason.is_empty() => format!(" {}", reason),
            _ => String::new(),
        };
        if reason_text.is_empty() && code == Some(WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE) {
            reason_text = " message too big".to_string();
        }
        return CodexThrown::web_socket_close(
            format!("WebSocket closed{}{}", code_text, reason_text).trim().to_string(),
            code,
            reason.filter(|reason| !reason.is_empty()),
            was_clean,
        );
    }
    CodexThrown::error("WebSocket closed")
}

/// `acquireWebSocket(url, headers, sessionId, signal?)` result.
struct AcquiredWebSocket {
    socket: Arc<dyn WebSocketLike>,
    entry: Option<Arc<Mutex<CachedWebSocketConnection>>>,
    reused: bool,
    keep: bool,
    session_id: Option<String>,
    released: bool,
}

impl Drop for AcquiredWebSocket {
    fn drop(&mut self) {
        if !self.released {
            self.keep = false;
            let session_id = self.session_id.clone();
            self.finish(session_id.as_deref());
        }
    }
}

impl AcquiredWebSocket {
    /// `release({ keep })`.
    fn release(&mut self, session_id: Option<&str>, keep: bool) {
        self.keep = keep;
        self.finish(session_id);
        self.released = true;
    }

    fn finish(&mut self, session_id: Option<&str>) {
        let keep = self.keep;
        let Some(session_id) = session_id else {
            close_web_socket_silently(&self.socket, 1000, "done");
            return;
        };
        let Some(entry) = self.entry.clone() else {
            close_web_socket_silently(&self.socket, 1000, "done");
            return;
        };
        let reusable = {
            let entry = entry.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            is_web_socket_reusable(&entry.socket)
        };
        if !keep || !reusable {
            let socket = entry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .socket
                .clone();
            close_web_socket_silently(&socket, 1000, "done");
            let idle_timer = entry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .idle_timer
                .take();
            if let Some(idle_timer) = idle_timer {
                idle_timer.abort();
            }
            let mut cache = web_socket_session_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let is_same_entry = cache
                .get(session_id)
                .map(|cached| Arc::ptr_eq(cached, &entry))
                .unwrap_or(false);
            if is_same_entry {
                cache.remove(session_id);
            }
            return;
        }
        entry.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).busy = false;
        schedule_session_web_socket_expiry(session_id, &entry);
    }
}

/// `acquireWebSocket(url, headers, sessionId, signal?)`.
async fn acquire_web_socket(
    url: &str,
    headers: &IndexMap<String, String>,
    session_id: Option<&str>,
    signal: Option<&tokio_util::sync::CancellationToken>,
) -> Result<AcquiredWebSocket, CodexThrown> {
    let Some(session_id) = session_id else {
        let socket = connect_web_socket(url, headers, signal).await?;
        return Ok(AcquiredWebSocket {
            socket,
            entry: None,
            reused: false,
            keep: true,
            session_id: None,
            released: false,
        });
    };

    let identity = format!("{:x}", Sha256::digest(serde_json::to_vec(&(url, headers)).unwrap_or_default()));
    let cached = web_socket_session_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(session_id)
        .cloned();
    if let Some(cached) = cached {
        {
            let mut entry = cached.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(idle_timer) = entry.idle_timer.take() {
                idle_timer.abort();
            }
        }
        let (busy, reusable, socket) = {
            let mut entry = cached.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let busy = entry.busy;
            let reusable = is_web_socket_reusable(&entry.socket) && entry.connection_identity == identity;
            if !busy && reusable {
                entry.busy = true;
            }
            (busy, reusable, entry.socket.clone())
        };
        if !busy && reusable {
            return Ok(AcquiredWebSocket {
                socket,
                entry: Some(cached),
                reused: true,
                keep: true,
                session_id: Some(session_id.to_string()),
                released: false,
            });
        }
        if busy {
            let socket = connect_web_socket(url, headers, signal).await?;
            return Ok(AcquiredWebSocket {
                socket,
                entry: None,
                reused: false,
                keep: true,
                session_id: Some(session_id.to_string()),
                released: false,
            });
        }
        if !reusable {
            close_web_socket_silently(&socket, 1000, "done");
            web_socket_session_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(session_id);
        }
    }

    let socket = connect_web_socket(url, headers, signal).await?;
    let entry = Arc::new(Mutex::new(CachedWebSocketConnection {
        socket: socket.clone(),
        busy: true,
        idle_timer: None,
        continuation: None,
        connection_identity: identity,
    }));
    web_socket_session_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(session_id.to_string(), entry.clone());
    Ok(AcquiredWebSocket {
        socket,
        entry: Some(entry),
        reused: false,
        keep: true,
        session_id: Some(session_id.to_string()),
        released: false,
    })
}

/// `decodeWebSocketData(data)`.
pub async fn decode_web_socket_data(data: &Value) -> Option<String> {
    decode_web_socket_data_sync(data)
}

fn decode_web_socket_data_sync(data: &Value) -> Option<String> {
    match data {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let bytes: Vec<u8> = items.iter().filter_map(Value::as_u64).map(|byte| byte as u8).collect();
            Some(String::from_utf8_lossy(&bytes).to_string())
        }
        Value::Object(object) => object.get("text").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

#[derive(Default)]
struct WebSocketParseShared {
    queue: VecDeque<Result<Value, CodexThrown>>,
    done: bool,
    failed: Option<CodexThrown>,
    saw_completion: bool,
}

/// `parseWebSocket(socket, signal?)`.
pub fn parse_web_socket(
    socket: Arc<dyn WebSocketLike>,
    signal: Option<tokio_util::sync::CancellationToken>,
) -> Pin<Box<dyn futures::Stream<Item = Result<Value, CodexThrown>> + Send>> {
    let mut state = WebSocketParseState {
        socket,
        signal,
        queue: VecDeque::new(),
        done: false,
        failed: None,
        saw_completion: false,
        wake: Arc::new(tokio::sync::Notify::new()),
        state: Arc::new(Mutex::new(WebSocketParseShared::default())),
        listeners: None,
        finished: false,
    };
    // Native IO can reply on another thread before the first stream poll.
    state.ensure_listeners();
    Box::pin(futures::stream::unfold(
        state,
        |mut state| async move {
            if state.finished {
                return None;
            }
            state.ensure_listeners();
            loop {
                state.sync_from_shared();
                if state
                    .signal
                    .as_ref()
                    .map(|signal| signal.is_cancelled())
                    .unwrap_or(false)
                {
                    state.finished = true;
                    state.cleanup();
                    return Some((Err(CodexThrown::error("Request was aborted")), state));
                }
                if let Some(event) = state.queue.pop_front() {
                    return Some((event, state));
                }
                if state.done {
                    break;
                }
                let notified = state.wake.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                // Re-check after registering interest so a message cannot be missed.
                // The guard is scoped so it is never held across the await.
                let (has_event, done) = {
                    let shared = state.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    (!shared.queue.is_empty(), shared.done)
                };
                if !has_event && !done {
                    match &state.signal {
                        Some(signal) => tokio::select! {
                            _ = notified => {}
                            _ = signal.cancelled() => {}
                        },
                        None => notified.await,
                    }
                }
            }

            state.sync_from_shared();
            // A TypeScript generator throws once, then is exhausted. `done`
            // tracks the socket; `finished` tracks this generator's final yield.
            state.finished = true;
            state.cleanup();
            if let Some(failed) = state.failed.clone() {
                return Some((Err(failed), state));
            }
            if !state.saw_completion {
                return Some((
                    Err(CodexThrown::error(
                        "WebSocket stream closed before response.completed",
                    )),
                    state,
                ));
            }
            None
        },
    ))
}

struct WebSocketParseState {
    socket: Arc<dyn WebSocketLike>,
    signal: Option<tokio_util::sync::CancellationToken>,
    queue: VecDeque<Result<Value, CodexThrown>>,
    done: bool,
    failed: Option<CodexThrown>,
    saw_completion: bool,
    wake: Arc<tokio::sync::Notify>,
    state: Arc<Mutex<WebSocketParseShared>>,
    listeners: Option<Vec<(WebSocketEventType, WebSocketListener)>>,
    finished: bool,
}

impl Drop for WebSocketParseState {
    fn drop(&mut self) {
        // Async generator finally also runs when its consumer stops early.
        self.cleanup();
    }
}

impl WebSocketParseState {
    /// `socket.addEventListener(...)` for message/error/close.
    fn ensure_listeners(&mut self) {
        if self.listeners.is_some() {
            return;
        }
        let state = self.state.clone();
        let wake = self.wake.clone();

        let on_message: WebSocketListener = Arc::new(move |event: Value| {
            let data = event.get("data").cloned().unwrap_or(Value::Null);
            let Some(text) = decode_web_socket_data_sync(&data) else {
                return;
            };
            if text.is_empty() {
                return;
            }
            match serde_json::from_str::<Value>(&text) {
                Ok(parsed) => {
                    let type_ = parsed.get("type").and_then(Value::as_str).unwrap_or_default().to_string();
                    let mut shared = state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    if type_ == "response.completed"
                        || type_ == "response.done"
                        || type_ == "response.incomplete"
                    {
                        shared.saw_completion = true;
                        shared.done = true;
                    }
                    shared.queue.push_back(Ok(parsed));
                    drop(shared);
                    wake.notify_waiters();
                }
                Err(cause) => {
                    let thrown = CodexThrown::protocol_error(
                        format!("Invalid Codex WebSocket JSON: {}", cause),
                        Some(Value::String(text)),
                    );
                    let mut shared = state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    shared.failed = Some(thrown);
                    shared.done = true;
                    drop(shared);
                    wake.notify_waiters();
                }
            }
        });

        let state_for_error = self.state.clone();
        let wake_for_error = self.wake.clone();
        let on_error: WebSocketListener = Arc::new(move |event: Value| {
            let thrown = extract_web_socket_error(&event);
            let mut shared = state_for_error.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if !shared.saw_completion && shared.failed.is_none() {
                shared.failed = Some(thrown);
            }
            shared.done = true;
            drop(shared);
            wake_for_error.notify_waiters();
        });

        let state_for_close = self.state.clone();
        let wake_for_close = self.wake.clone();
        let on_close: WebSocketListener = Arc::new(move |event: Value| {
            let mut shared = state_for_close.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if shared.saw_completion {
                shared.done = true;
                drop(shared);
                wake_for_close.notify_waiters();
                return;
            }
            if shared.failed.is_none() {
                shared.failed = Some(extract_web_socket_close_error(&event));
            }
            shared.done = true;
            drop(shared);
            wake_for_close.notify_waiters();
        });

        self.socket.add_event_listener(WebSocketEventType::Message, on_message.clone());
        self.socket.add_event_listener(WebSocketEventType::Error, on_error.clone());
        self.socket.add_event_listener(WebSocketEventType::Close, on_close.clone());
        self.listeners = Some(vec![
            (WebSocketEventType::Message, on_message),
            (WebSocketEventType::Error, on_error),
            (WebSocketEventType::Close, on_close),
        ]);
    }

    /// `socket.removeEventListener(...)` in the generator's `finally`.
    fn cleanup(&mut self) {
        let Some(listeners) = self.listeners.take() else {
            return;
        };
        for (type_, listener) in listeners {
            self.socket.remove_event_listener(type_, &listener);
        }
    }

    /// Moves the shared queue/failure state into this generator step.
    fn sync_from_shared(&mut self) {
        let mut shared = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while let Some(event) = shared.queue.pop_front() {
            self.queue.push_back(event);
        }
        self.done = shared.done;
        if self.failed.is_none() {
            self.failed = shared.failed.clone();
        }
        self.saw_completion = shared.saw_completion;
    }
}

/// `requestBodyWithoutInput(body)`.
pub fn request_body_without_input(body: &RequestBody) -> RequestBody {
    let mut rest = body.clone();
    rest.shift_remove("input");
    rest.shift_remove("previous_response_id");
    rest
}

/// `responseInputsEqual(a, b)`.
pub fn response_inputs_equal(a: Option<&Vec<Value>>, b: Option<&Vec<Value>>) -> bool {
    let a = serde_json::to_string(a.unwrap_or(&Vec::new())).unwrap_or_default();
    let b = serde_json::to_string(b.unwrap_or(&Vec::new())).unwrap_or_default();
    a == b
}

/// `requestBodiesMatchExceptInput(a, b)`.
pub fn request_bodies_match_except_input(a: &RequestBody, b: &RequestBody) -> bool {
    let a = serde_json::to_string(&Value::Object(request_body_without_input(a))).unwrap_or_default();
    let b = serde_json::to_string(&Value::Object(request_body_without_input(b))).unwrap_or_default();
    a == b
}

/// `getCachedWebSocketInputDelta(body, continuation)`.
pub fn get_cached_web_socket_input_delta(
    body: &RequestBody,
    continuation: &CachedWebSocketContinuationState,
) -> Option<Vec<Value>> {
    if !request_bodies_match_except_input(body, &continuation.last_request_body) {
        return None;
    }

    let current_input = body.get("input").and_then(Value::as_array).cloned().unwrap_or_default();
    let mut baseline = continuation
        .last_request_body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    baseline.extend(continuation.last_response_items.iter().cloned());
    if current_input.len() < baseline.len() {
        return None;
    }

    let prefix = current_input[..baseline.len()].to_vec();
    if !response_inputs_equal(Some(&prefix), Some(&baseline)) {
        return None;
    }

    Some(current_input[baseline.len()..].to_vec())
}

/// `buildCachedWebSocketRequestBody(entry, body)`.
pub fn build_cached_web_socket_request_body(
    entry: &mut CachedWebSocketConnection,
    body: &RequestBody,
) -> RequestBody {
    let Some(continuation) = entry.continuation.clone() else {
        return body.clone();
    };

    if continuation.socket_identity != Arc::as_ptr(&entry.socket) as *const () as usize {
        entry.continuation = None;
        return body.clone();
    }

    let delta = get_cached_web_socket_input_delta(body, &continuation);
    let Some(delta) = delta else {
        entry.continuation = None;
        return body.clone();
    };
    if continuation.last_response_id.is_empty() {
        entry.continuation = None;
        return body.clone();
    }

    let mut request = body.clone();
    request.insert(
        "previous_response_id".to_string(),
        Value::String(continuation.last_response_id.clone()),
    );
    request.insert("input".to_string(), Value::Array(delta));
    request
}

fn is_output_producing_event(event: &Value) -> bool {
    let kind = event.get("type").and_then(Value::as_str).unwrap_or_default();
    matches!(kind, "response.completed" | "response.incomplete" | "response.done")
        || ["response.output_", "response.reasoning_", "response.content_", "response.refusal.", "response.function_"]
            .iter().any(|prefix| kind.starts_with(prefix))
}

/// Emit Start only once the provider produces output, not handshake metadata.
pub fn start_web_socket_output_on_first_event<S>(
    events: S,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    on_start: Arc<dyn Fn() + Send + Sync>,
    error_slot: Arc<Mutex<Option<CodexThrown>>>,
) -> Pin<Box<dyn futures::Stream<Item = Value> + Send>>
where
    S: futures::Stream<Item = Value> + Send + 'static,
{
    Box::pin(futures::stream::unfold(
        (
            Box::pin(events) as Pin<Box<dyn futures::Stream<Item = Value> + Send>>,
            false,
            output,
            stream,
            on_start,
            error_slot,
        ),
        |(mut events, started, output, stream, on_start, error_slot)| async move {
            let _ = &error_slot;
            let event = events.next().await?;
            let producing_output = is_output_producing_event(&event);
            if !started && producing_output {
                on_start();
                stream.push(AssistantMessageEvent::Start {
                    partial: output.clone(),
                });
            }
            Some((event, (events, started || producing_output, output, stream, on_start, error_slot)))
        },
    ))
}

/// `processWebSocketStream(url, body, headers, output, stream, model, onStart, options?)`.
async fn process_web_socket_stream(
    url: &str,
    body: &RequestBody,
    headers: &IndexMap<String, String>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    model: &Model,
    on_start: Arc<dyn Fn() + Send + Sync>,
    options: &OpenAICodexResponsesOptions,
) -> Result<(), CodexThrown> {
    ensure_web_socket_session_cleanup_registered();
    let mut acquired = acquire_web_socket(
        url,
        headers,
        options.stream.session_id.as_deref(),
        options.stream.signal.as_ref(),
    )
    .await?;
    let socket = acquired.socket.clone();
    let entry = acquired.entry.clone();
    let reused = acquired.reused;
    let mut keep_connection = true;
    let use_cached_context = matches!(options.stream.transport.as_deref(), None | Some("auto" | "websocket-cached"));
    // ChatGPT Codex Responses rejects `store: true` ("Store must be set to false").
    // WebSocket continuation still works via connection-scoped previous_response_id state.
    let full_body = body.clone();
    let request_body = match (use_cached_context, entry.as_ref()) {
        (true, Some(entry)) => {
            let mut entry_guard = entry.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            build_cached_web_socket_request_body(&mut entry_guard, &full_body)
        }
        _ => full_body.clone(),
    };
    if let Some(session_id) = options.stream.session_id.as_deref() {
        get_or_create_web_socket_debug_stats(session_id);
        let input_len = request_body
            .get("input")
            .and_then(Value::as_array)
            .map(|input| input.len() as i64)
            .unwrap_or(0);
        let previous_response_id = request_body
            .get("previous_response_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let store_true = request_body.get("store").and_then(Value::as_bool) == Some(true);
        update_web_socket_debug_stats(session_id, |stats| {
            stats.requests += 1;
            if reused {
                stats.connections_reused += 1;
            } else {
                stats.connections_created += 1;
            }
            if use_cached_context {
                stats.cached_context_requests += 1;
            }
            if store_true {
                stats.store_true_requests += 1;
            }
            stats.last_input_items = input_len;
            match previous_response_id {
                Some(previous_response_id) => {
                    stats.delta_requests += 1;
                    stats.last_delta_input_items = Some(input_len);
                    stats.last_previous_response_id = Some(previous_response_id);
                }
                None => {
                    stats.full_context_requests += 1;
                    stats.last_delta_input_items = None;
                    stats.last_previous_response_id = None;
                }
            }
        });
    }

    let mut request = Map::new();
    request.insert("type".to_string(), Value::String("response.create".to_string()));
    for (key, value) in request_body.iter() {
        request.insert(key.clone(), value.clone());
    }
    let codex_error: Arc<Mutex<Option<CodexThrown>>> = Arc::new(Mutex::new(None));
    let raw_events = parse_web_socket(socket.clone(), options.stream.signal.clone()).inspect({
        let on_start = on_start.clone();
        let observation_options = options.stream.clone();
        move |event| {
            if let Ok(event) = event { observe_event(&observation_options, event); }
            // Output may have side effects even if the shared parser ignores its subtype.
            if event.as_ref().is_ok_and(is_output_producing_event) {
                on_start();
            }
        }
    });
    socket.send(&serde_json::to_string(&Value::Object(request)).unwrap_or_default());
    // B5: this provider's native WebSocket path never reports a response header edge, so
    // every Codex attempt used to record `transport_websocket = null` and a WS-vs-SSE
    // comparison was impossible. The stage is a content-free observation, emitted only to
    // the metrics observer, and it means "the payload was sent over a WebSocket".
    if let Some(observer) = options.stream.on_stream_observation.clone() {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| observer("transport_ws")));
    }
    let events: Pin<Box<dyn futures::Stream<Item = Value> + Send>> = map_codex_events(
        raw_events,
        codex_error.clone(),
    );
    let started_events: Pin<Box<dyn futures::Stream<Item = Value> + Send>> =
        start_web_socket_output_on_first_event(
            events,
            output.clone(),
            stream.clone(),
            on_start,
            codex_error.clone(),
        );

    let stream_options = OpenAIResponsesStreamOptions {
        on_output_item_done: options.on_output_item_done.clone(),
        // openai-codex-responses.ts:1231 `serviceTier: options?.serviceTier`.
        service_tier: options.service_tier.clone().flatten(),
        resolve_service_tier: Some(Arc::new(|response_tier, request_tier| {
            resolve_codex_service_tier(response_tier, request_tier)
        })),
        apply_service_tier_pricing: Some(Arc::new({
            let model = model.clone();
            move |usage: &mut Usage, service_tier: Option<&str>| {
                apply_service_tier_pricing(usage, service_tier, &model)
            }
        })),
        on_usage_observation: options.stream.on_usage_observation.clone(),
    };

    let result = process_responses_stream(
        Box::pin(started_events),
        output,
        stream,
        model,
        Some(&stream_options),
    )
    .await;
    // The Codex generator's own `throw`s surface after the shared loop (see map_codex_events).
    let codex_failure: Option<CodexThrown> = codex_error
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();

    match (result, codex_failure) {
        (Ok(()), None) => {
            if options
                .stream
                .signal
                .as_ref()
                .map(|signal| signal.is_cancelled())
                .unwrap_or(false)
            {
                keep_connection = false;
            } else if use_cached_context {
                if let (Some(entry), Some(response_id)) = (entry.as_ref(), output.response_id.clone()) {
                    let one_message_context =
                        Context::new(None, vec![crate::types::Message::assistant(output.clone())], None);
                    let response_items = convert_responses_messages(
                        model,
                        &one_message_context,
                        &|provider: &str| CODEX_TOOL_CALL_PROVIDERS.contains(&provider),
                        Some(&ConvertResponsesMessagesOptions {
                            include_system_prompt: Some(false),
                        }),
                    )
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|item| item.get("type").and_then(Value::as_str) != Some("function_call_output"))
                    .collect::<Vec<_>>();
                    let mut entry_guard = entry.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    entry_guard.continuation = Some(CachedWebSocketContinuationState {
                        last_request_body: full_body.clone(),
                        last_response_id: response_id,
                        last_response_items: response_items,
                        socket_identity: Arc::as_ptr(&socket) as *const () as usize,
                    });
                }
            }
        }
        (result, codex_failure) => {
            if let Some(entry) = entry.as_ref() {
                entry.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).continuation = None;
            }
            keep_connection = false;
            acquired.release(options.stream.session_id.as_deref(), keep_connection);
            if let Some(error) = codex_failure {
                return Err(error);
            }
            let error = match result {
                Err(error) => error,
                Ok(()) => {
                    return Err(CodexThrown::error("WebSocket stream ended without a result"));
                }
            };
            return Err(match error {
                crate::providers::openai_responses_shared::ResponsesStreamError::StreamFailure(failure) => {
                    CodexThrown::api_error(
                        failure.message,
                        failure.info.provider_error_type.clone(),
                        failure.info.status,
                        failure.info.retry_after_ms,
                        None,
                    )
                }
                crate::providers::openai_responses_shared::ResponsesStreamError::Message(message) => {
                    CodexThrown::error(message)
                }
            });
        }
    }

    acquired.release(options.stream.session_id.as_deref(), keep_connection);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn codex_sse_frames_arrive_before_close_for_every_unicode_split() {
        for delimiter in ["\n", "\r\n", "\r"] {
            let frame = format!("data: {{\"type\":\"response.output_text.delta\",\"delta\":\"Ready ✓ 日本\"}}{delimiter}{delimiter}");
            for split in 0..=frame.len() {
                let chunks = vec![
                    Ok(bytes::Bytes::copy_from_slice(&frame.as_bytes()[..split])),
                    Ok(bytes::Bytes::copy_from_slice(&frame.as_bytes()[split..])),
                ];
                let pending = futures::stream::iter(chunks).chain(futures::stream::pending());
                let mut events = parse_sse_chunks(Box::pin(pending));
                let event = tokio::time::timeout(std::time::Duration::from_millis(100), events.next())
                    .await.expect("complete frame must not wait for body close")
                    .expect("one frame").expect("valid UTF-8 JSON");
                assert_eq!(event["delta"], "Ready ✓ 日本", "delimiter {delimiter:?}, split {split}");
            }
        }
    }

    #[tokio::test]
    async fn codex_sse_keeps_multiline_comments_done_and_protocol_errors() {
        let wire = b": heartbeat\r\nevent: ignored\rdata: {\ndata: \"type\":\"response.created\"}\r\rdata: [DONE]\n\ndata: not-json\r\rdata: {\"type\":\"must-not-deliver\"}\r\r";
        let chunks = futures::stream::iter(vec![Ok(bytes::Bytes::from_static(wire))]);
        let mut events = parse_sse_chunks(Box::pin(chunks));
        assert_eq!(events.next().await.unwrap().unwrap()["type"], "response.created");
        let error = events.next().await.unwrap().unwrap_err();
        assert_eq!(error.name, "CodexProtocolError");
        assert!(error.message.starts_with("Invalid Codex SSE JSON:"));
        assert!(events.next().await.is_none(), "do not deliver any event after the first protocol error");
    }

    #[tokio::test]
    async fn codex_sse_http_body_failure_is_not_silent_success() {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            socket.read(&mut request).await.unwrap();
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1000\r\nConnection: close\r\n\r\ndata: {\"type\":\"response.created\"}\r\n\r\n").await.unwrap();
        });
        let response = reqwest::Client::new().get(format!("http://{address}/fixture"))
            .send().await.unwrap();
        let mut events = parse_sse_response(response);
        assert_eq!(events.next().await.unwrap().unwrap()["type"], "response.created");
        assert!(events.next().await.unwrap().is_err());
        assert!(events.next().await.is_none());
        server.await.unwrap();
    }

    static WEB_SOCKET_TEST_LOCK: Mutex<()> = Mutex::new(());
    use crate::types::{ContentBlock, Message, TextContent, UserContent, UserMessage};
    use serde_json::json;

    fn token(account_id: &str) -> String {
        use base64::Engine;
        let payload = format!("{{\"{}\":{{\"chatgpt_account_id\":\"{}\"}}}}", JWT_CLAIM_PATH, account_id);
        let encoded = base64::engine::general_purpose::STANDARD.encode(payload.as_bytes());
        format!("aaa.{}.bbb", encoded)
    }

    fn model() -> Model {
        let mut model = Model::new(
            "gpt-5.1-codex",
            "GPT-5.1 Codex",
            "openai-codex-responses",
            "openai-codex",
            "https://chatgpt.com/backend-api",
        );
        model.reasoning = true;
        model
    }

    fn context() -> Context {
        Context::new(
            Some("You are a helpful assistant.".to_string()),
            vec![Message::user(UserMessage::new(UserContent::Text("Say hello".to_string()), 1))],
            None,
        )
    }

    #[test]
    fn resolve_codex_url_appends_the_codex_responses_path() {
        assert_eq!(
            resolve_codex_url(Some("https://chatgpt.com/backend-api")),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("https://chatgpt.com/backend-api/")),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("https://chatgpt.com/backend-api/codex")),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("https://x/codex/responses")),
            "https://x/codex/responses"
        );
        assert_eq!(resolve_codex_url(None), "https://chatgpt.com/backend-api/codex/responses");
        assert_eq!(resolve_codex_url(Some("   ")), "https://chatgpt.com/backend-api/codex/responses");
    }

    #[test]
    fn resolve_codex_web_socket_url_switches_the_scheme() {
        assert_eq!(
            resolve_codex_web_socket_url(Some("https://chatgpt.com/backend-api")),
            "wss://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_web_socket_url(Some("http://localhost:8080")),
            "ws://localhost:8080/codex/responses"
        );
    }

    #[test]
    fn build_request_body_matches_typescript_defaults() {
        let model = model();
        let body = build_request_body(&model, &context(), None).unwrap();
        assert_eq!(body["model"], json!("gpt-5.1-codex"));
        assert_eq!(body["store"], json!(false));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["instructions"], json!("You are a helpful assistant."));
        assert_eq!(body["text"], json!({ "verbosity": "low" }));
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["tool_choice"], json!("auto"));
        assert_eq!(body["parallel_tool_calls"], json!(true));
        assert!(!body.contains_key("prompt_cache_key"));
        assert!(!body.contains_key("reasoning"));
        assert!(!body.contains_key("tools"));
        // `includeSystemPrompt: false` keeps the system prompt out of `input`.
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn build_request_body_falls_back_to_the_default_instructions() {
        let model = model();
        let context = Context::new(
            None,
            vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 1))],
            None,
        );
        let body = build_request_body(&model, &context, None).unwrap();
        assert_eq!(body["instructions"], json!("You are a helpful assistant."));
    }

    #[test]
    fn build_request_body_maps_reasoning_none_to_the_off_level() {
        let mut model = model();
        model.thinking_level_map = Some([("off".to_string(), Some("none".to_string()))].into_iter().collect());
        let options = OpenAICodexResponsesOptions {
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        };
        let body = build_request_body(&model, &context(), Some(&options)).unwrap();
        assert_eq!(body["reasoning"], json!({ "effort": "none", "summary": "auto" }));

        let options = OpenAICodexResponsesOptions {
            reasoning_effort: Some("high".to_string()),
            reasoning_summary: Some(Some("concise".to_string())),
            ..Default::default()
        };
        let body = build_request_body(&model, &context(), Some(&options)).unwrap();
        assert_eq!(body["reasoning"], json!({ "effort": "high", "summary": "concise" }));
    }

    #[test]
    fn build_request_body_always_emits_reasoning_for_the_none_effort() {
        // openai-codex-responses.ts:400-403: `model.thinkingLevelMap?.off ?? "none"` can never be
        // null when the effort is "none" (`??` falls through on null AND undefined), so
        // `if (effort !== null)` is always true and `body.reasoning` is always emitted - even
        // when the model has no thinkingLevelMap or maps "off" to null.
        let mut model = model();
        model.thinking_level_map = Some([("off".to_string(), None)].into_iter().collect());
        let options = OpenAICodexResponsesOptions {
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        };
        let body = build_request_body(&model, &context(), Some(&options)).unwrap();
        assert_eq!(body["reasoning"], json!({ "effort": "none", "summary": "auto" }));

        // No thinkingLevelMap at all: `?.off` is undefined and `?? "none"` still yields "none".
        let options = OpenAICodexResponsesOptions {
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        };
        // NOTE: `model` is already shadowed by the local binding above, so the helper must be
        // reached through its path-qualified name here.
        let mut plain = self::model();
        plain.thinking_level_map = None;
        let body = build_request_body(&plain, &context(), Some(&options)).unwrap();
        assert_eq!(body["reasoning"], json!({ "effort": "none", "summary": "auto" }));
    }

    #[test]
    fn build_request_body_uses_strict_null_tools_and_service_tier() {
        let model = model();
        let context = Context::new(
            None,
            Vec::new(),
            Some(vec![crate::types::Tool {
                name: "bash".to_string(),
                description: "Run a command".to_string(),
                parameters: json!({ "type": "object" }),
            }]),
        );
        let options = OpenAICodexResponsesOptions {
            service_tier: Some(Some("flex".to_string())),
            text_verbosity: Some("high".to_string()),
            ..Default::default()
        };
        let body = build_request_body(&model, &context, Some(&options)).unwrap();
        assert_eq!(body["tools"][0]["strict"], json!(null));
        assert_eq!(body["service_tier"], json!("flex"));
        assert_eq!(body["text"]["verbosity"], json!("high"));
    }

    /// openai-codex-responses.ts:390-392 + simple-options.ts:10: the tier must survive `from_base`
    /// (the Codex response-tier pricing multiplier depends on it: an absent field makes the service
    /// pick "auto"/project tier while TS sends the literal "default"), an explicit `null` stays on
    /// the wire, and an absent tier leaves the key off.
    #[test]
    fn from_base_forwards_service_tier_onto_the_codex_request() {
        let model = model();
        let context = context();

        let base = StreamOptions {
            // The agent sets this every turn (pi-agent-core/src/agent.rs:922).
            service_tier: Some(Some("default".to_string())),
            ..Default::default()
        };
        let options = OpenAICodexResponsesOptions::from_base(&base);
        assert_eq!(options.service_tier, Some(Some("default".to_string())));
        assert_eq!(
            build_request_body(&model, &context, Some(&options)).unwrap()["service_tier"],
            json!("default")
        );

        let priority = OpenAICodexResponsesOptions::from_base(&StreamOptions {
            service_tier: Some(Some("priority".to_string())),
            ..Default::default()
        });
        assert_eq!(
            build_request_body(&model, &context, Some(&priority)).unwrap()["service_tier"],
            json!("priority")
        );

        // `serviceTier: null` is the explicit reset TS serializes as `"service_tier": null`.
        let reset = OpenAICodexResponsesOptions::from_base(&StreamOptions {
            service_tier: Some(None),
            ..Default::default()
        });
        let body = build_request_body(&model, &context, Some(&reset)).unwrap();
        assert!(body.contains_key("service_tier"));
        assert_eq!(body["service_tier"], json!(null));

        let absent = OpenAICodexResponsesOptions::from_base(&StreamOptions::default());
        assert!(!build_request_body(&model, &context, Some(&absent))
            .unwrap()
            .contains_key("service_tier"));
    }

    /// The wire shape of `serviceTier` on the provider options: an explicit `null` deserializes to
    /// `Some(None)` (never "absent"), an absent key stays `None`, and only `None` is skipped when
    /// re-serialized (types.rs:195-197 / openai-codex-responses.ts:149).
    #[test]
    fn service_tier_round_trips_through_the_provider_options() {
        let explicit_null: OpenAICodexResponsesOptions =
            serde_json::from_value(json!({ "serviceTier": null })).unwrap();
        assert_eq!(explicit_null.service_tier, Some(None));
        assert_eq!(
            serde_json::to_value(&explicit_null).unwrap().get("serviceTier"),
            Some(&Value::Null)
        );

        let absent: OpenAICodexResponsesOptions = serde_json::from_value(json!({})).unwrap();
        assert_eq!(absent.service_tier, None);
        assert!(serde_json::to_value(&absent).unwrap().get("serviceTier").is_none());

        let tiered: OpenAICodexResponsesOptions =
            serde_json::from_value(json!({ "serviceTier": "flex" })).unwrap();
        assert_eq!(tiered.service_tier, Some(Some("flex".to_string())));
    }

    /// openai-codex-responses.ts:369-371/328-337: a `throw` from `convertResponsesMessages`
    /// (transform-messages.ts:77 mismatched compaction checkpoint) ends the turn with that exact
    /// message; Rust must return `Err` rather than defaulting to an empty `input`, which silently
    /// dropped the whole conversation context.
    #[test]
    fn build_request_body_propagates_a_foreign_compaction_checkpoint() {
        use crate::compaction::ProviderCompactionCheckpoint;
        let model = model();
        let checkpoint = ProviderCompactionCheckpoint {
            version: 1,
            provider: "other-provider".to_string(),
            api: "other-api".to_string(),
            model: "other-model".to_string(),
            base_url: "https://example.invalid".to_string(),
            endpoint: None,
            items: vec![Map::new()],
            estimated_tokens: 1.0,
        };
        let context = Context::new(
            None,
            vec![crate::types::Message::user(crate::types::UserMessage {
                role: crate::types::ROLE_USER.to_string(),
                content: crate::types::UserContent::Text("hi".to_string()),
                provider_context: Some(checkpoint),
                timestamp: 0,
            })],
            None,
        );
        let error = match build_request_body(&model, &context, None) {
            Ok(body) => panic!(
                "build_request_body swallowed the foreign compaction checkpoint error and returned Ok; \
                 the request would be sent with input = {}",
                body.get("input").map(Value::to_string).unwrap_or_default()
            ),
            Err(error) => error,
        };
        assert_eq!(
            error,
            "Compaction checkpoint belongs to another model or provider; rebuild context from the session transcript"
        );
    }

    #[test]
    fn service_tier_multipliers_match_typescript() {
        let model = model();
        assert_eq!(get_service_tier_cost_multiplier(&model, Some("flex")), 0.5);
        assert_eq!(get_service_tier_cost_multiplier(&model, Some("priority")), 2.0);
        assert_eq!(get_service_tier_cost_multiplier(&model, Some("default")), 1.0);

        let gpt_5_5 = Model::new(
            "gpt-5.5-codex",
            "GPT-5.5 Codex",
            "openai-codex-responses",
            "openai-codex",
            "https://chatgpt.com/backend-api",
        );
        assert_eq!(get_service_tier_cost_multiplier(&gpt_5_5, Some("priority")), 2.5);
    }

    #[test]
    fn resolve_codex_service_tier_prefers_the_requested_tier() {
        assert_eq!(
            resolve_codex_service_tier(Some("default"), Some("flex")),
            Some("flex".to_string())
        );
        assert_eq!(
            resolve_codex_service_tier(Some("default"), Some("priority")),
            Some("priority".to_string())
        );
        assert_eq!(resolve_codex_service_tier(Some("flex"), Some("priority")), Some("flex".to_string()));
        assert_eq!(resolve_codex_service_tier(None, Some("flex")), Some("flex".to_string()));
        assert_eq!(resolve_codex_service_tier(Some("default"), None), Some("default".to_string()));
        assert_eq!(resolve_codex_service_tier(None, None), None);
    }

    #[test]
    fn normalize_codex_status_only_accepts_known_statuses() {
        assert_eq!(normalize_codex_status(Some(&json!("completed"))), Some("completed".to_string()));
        assert_eq!(normalize_codex_status(Some(&json!("in_progress"))), Some("in_progress".to_string()));
        assert_eq!(normalize_codex_status(Some(&json!("unknown"))), None);
        assert_eq!(normalize_codex_status(Some(&json!(1))), None);
        assert_eq!(normalize_codex_status(None), None);
    }

    #[test]
    fn extract_account_id_reads_the_jwt_claim() {
        assert_eq!(extract_account_id(&token("acc_test")).unwrap(), "acc_test");
        assert_eq!(
            extract_account_id("aaa.bbb").unwrap_err().message,
            "Failed to extract accountId from token"
        );
        assert_eq!(
            extract_account_id("aaa.notbase64.bbb").unwrap_err().message,
            "Failed to extract accountId from token"
        );
        assert_eq!(
            extract_account_id("aaa.eyJ4IjoxfQ.bbb").unwrap_err().message,
            "Failed to extract accountId from token"
        );
    }

    #[test]
    fn extract_account_id_accepts_url_safe_base64() {
        use base64::Engine;
        let payload = format!("{{\"{}\":{{\"chatgpt_account_id\":\"acc/safe\"}}}}", JWT_CLAIM_PATH);
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload.as_bytes());
        let token = format!("aaa.{}.bbb", encoded);
        assert_eq!(extract_account_id(&token).unwrap(), "acc/safe");
    }

    #[test]
    fn base_headers_match_typescript() {
        let headers = build_base_codex_headers(None, None, "acc_1", "token");
        assert_eq!(headers.get("Authorization").map(String::as_str), Some("Bearer token"));
        assert_eq!(headers.get("chatgpt-account-id").map(String::as_str), Some("acc_1"));
        assert_eq!(headers.get("originator").map(String::as_str), Some("pi"));
        // openai-codex-responses.ts:1347 shape: `pi (<platform> <release>; <arch>)`.
        assert!(headers.get("User-Agent").map(|agent| agent.starts_with("pi (")).unwrap_or(false));
        let user_agent = headers.get("User-Agent").expect("User-Agent is always set");
        assert_eq!(
            user_agent,
            &format!("pi ({} {}; {})", os_platform(), os_release(), std::env::consts::ARCH)
        );
        // `os.release()` is the kernel release on POSIX; it must not degrade to "unknown"
        // on a supported host (the previous port always sent OS_VERSION, which nothing sets).
        assert_ne!(os_release(), "unknown");

        let mut additional = IndexMap::new();
        additional.insert("X-Extra".to_string(), "1".to_string());
        let headers = build_base_codex_headers(None, Some(&additional), "acc_1", "token");
        assert_eq!(headers.get("X-Extra").map(String::as_str), Some("1"));
    }

    #[test]
    fn sse_headers_add_the_experimental_beta_and_session_headers() {
        let headers = build_sse_headers(None, None, "acc_1", "token", None);
        assert_eq!(headers.get("OpenAI-Beta").map(String::as_str), Some("responses=experimental"));
        assert_eq!(headers.get("accept").map(String::as_str), Some("text/event-stream"));
        assert_eq!(headers.get("content-type").map(String::as_str), Some("application/json"));
        assert!(!headers.contains_key("session_id"));

        let headers = build_sse_headers(None, None, "acc_1", "token", Some("session-1"));
        assert_eq!(headers.get("session_id").map(String::as_str), Some("session-1"));
        assert_eq!(headers.get("x-client-request-id").map(String::as_str), Some("session-1"));
    }

    #[test]
    fn web_socket_headers_replace_the_sse_specific_headers() {
        let mut init: IndexMap<String, String> = IndexMap::new();
        init.insert("accept".to_string(), "text/event-stream".to_string());
        init.insert("content-type".to_string(), "application/json".to_string());
        init.insert("openai-beta".to_string(), "old".to_string());
        let headers = build_web_socket_headers(Some(&init), None, "acc_1", "token", "request-1");
        assert!(!headers.contains_key("accept"));
        assert!(!headers.contains_key("content-type"));
        assert_eq!(
            headers.get("OpenAI-Beta").map(String::as_str),
            Some(OPENAI_BETA_RESPONSES_WEBSOCKETS)
        );
        assert_eq!(headers.get("x-client-request-id").map(String::as_str), Some("request-1"));
        assert_eq!(headers.get("session_id").map(String::as_str), Some("request-1"));
    }

    #[test]
    fn codex_usage_limit_message_builds_the_friendly_message() {
        let err = json!({ "code": "usage_limit_reached", "plan_type": "PLUS", "resets_at": (now_ms() as f64 / 1000.0) + 120.0 });
        let message = codex_usage_limit_message(&err, None).unwrap();
        assert!(message.friendly_message.starts_with("You have hit your ChatGPT usage limit (plus plan)."));
        assert!(message.friendly_message.contains("Try again in ~2 min."));
        assert!(message.retry_after_ms.unwrap() > 0.0);

        let err = json!({ "code": "usage_not_included" });
        assert!(codex_usage_limit_message(&err, None).is_some());

        // Status 429 triggers the limit message even without a matching code.
        let err = json!({ "message": "slow down" });
        assert!(codex_usage_limit_message(&err, Some(429)).is_some());

        let err = json!({ "code": "invalid_request" });
        assert!(codex_usage_limit_message(&err, Some(400)).is_none());
    }

    #[test]
    fn cached_websocket_request_body_uses_the_delta() {
        let mut body = Map::new();
        body.insert("model".to_string(), json!("gpt-5.1-codex"));
        body.insert("input".to_string(), json!([{ "type": "message", "role": "user", "content": "a" }]));

        let mut entry = CachedWebSocketConnection {
            socket: Arc::new(FakeSocket::default()),
            busy: false,
            idle_timer: None,
            continuation: None,
            connection_identity: String::new(),
        };
        // No continuation: the full body is sent.
        assert_eq!(build_cached_web_socket_request_body(&mut entry, &body), body);

        entry.continuation = Some(CachedWebSocketContinuationState {
            last_request_body: body.clone(),
            last_response_id: "resp_1".to_string(),
            last_response_items: vec![json!({ "type": "message", "role": "assistant", "content": "hi" })],
            socket_identity: Arc::as_ptr(&entry.socket) as *const () as usize,
        });
        let mut followup = body.clone();
        followup.insert(
            "input".to_string(),
            json!([
                { "type": "message", "role": "user", "content": "a" },
                { "type": "message", "role": "assistant", "content": "hi" },
                { "type": "message", "role": "user", "content": "b" }
            ]),
        );
        let request = build_cached_web_socket_request_body(&mut entry, &followup);
        assert_eq!(request["previous_response_id"], json!("resp_1"));
        assert_eq!(request["input"], json!([{ "type": "message", "role": "user", "content": "b" }]));

        // A body that does not match clears the continuation and sends everything.
        let mut other = followup.clone();
        other.insert("model".to_string(), json!("other-model"));
        let request = build_cached_web_socket_request_body(&mut entry, &other);
        assert_eq!(request["input"].as_array().unwrap().len(), 3);
        assert!(entry.continuation.is_none());
    }

    #[test]
    fn cached_websocket_request_body_rejects_a_non_prefix_delta() {
        let mut body = Map::new();
        body.insert("model".to_string(), json!("gpt-5.1-codex"));
        body.insert("input".to_string(), json!([{ "type": "message", "role": "user", "content": "a" }]));
        let mut entry = CachedWebSocketConnection {
            socket: Arc::new(FakeSocket::default()),
            busy: false,
            idle_timer: None,
            continuation: Some(CachedWebSocketContinuationState {
                last_request_body: body.clone(),
                last_response_id: "resp_1".to_string(),
                last_response_items: vec![json!({ "type": "message", "role": "assistant", "content": "hi" })],
                socket_identity: 0,
            }),
            connection_identity: String::new(),
        };
        let mut rewritten = body.clone();
        rewritten.insert(
            "input".to_string(),
            json!([{ "type": "message", "role": "user", "content": "CHANGED" }]),
        );
        let request = build_cached_web_socket_request_body(&mut entry, &rewritten);
        assert!(!request.contains_key("previous_response_id"));
        assert!(entry.continuation.is_none());
    }

    #[test]
    fn web_socket_close_errors_carry_the_message_too_big_hint() {
        let error = extract_web_socket_close_error(&json!({ "code": 1009, "wasClean": false }));
        assert_eq!(error.message, "WebSocket closed 1009 message too big");
        assert_eq!(error.close_code, Some(1009));
        assert_eq!(error.was_clean, Some(false));

        let error = extract_web_socket_close_error(&json!({ "code": 1006, "reason": "gone" }));
        assert_eq!(error.message, "WebSocket closed 1006 gone");
        assert_eq!(error.close_reason.as_deref(), Some("gone"));

        let error = extract_web_socket_close_error(&Value::Null);
        assert_eq!(error.message, "WebSocket closed");
    }

    #[test]
    fn web_socket_errors_read_message_then_nested_error() {
        assert_eq!(extract_web_socket_error(&json!({ "message": "boom" })).message, "boom");
        assert_eq!(
            extract_web_socket_error(&json!({ "error": { "message": "nested" } })).message,
            "nested"
        );
        assert_eq!(extract_web_socket_error(&json!({})).message, "WebSocket error");
    }

    #[test]
    fn is_codex_non_transport_error_matches_the_two_error_classes() {
        assert!(is_codex_non_transport_error(&CodexThrown::api_error("m", None, None, None, None)));
        assert!(is_codex_non_transport_error(&CodexThrown::protocol_error("m", None)));
        assert!(!is_codex_non_transport_error(&CodexThrown::error("m")));
        assert!(!is_codex_non_transport_error(&CodexThrown::web_socket_close("m", None, None, None)));
    }

    #[test]
    fn reset_and_close_web_socket_sessions_clear_state() {
        let _guard = WEB_SOCKET_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        set_web_socket_constructor(None);
        get_or_create_web_socket_debug_stats("session-a");
        update_web_socket_debug_stats("session-a", |stats| stats.requests = 3);
        assert_eq!(
            get_openai_codex_web_socket_debug_stats("session-a").map(|stats| stats.requests),
            Some(3)
        );
        reset_openai_codex_web_socket_debug_stats(Some("session-a"));
        assert!(get_openai_codex_web_socket_debug_stats("session-a").is_none());

        close_openai_codex_web_socket_sessions(None);
    }

    #[test]
    fn web_socket_transport_is_unavailable_without_a_constructor() {
        let _guard = WEB_SOCKET_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        set_web_socket_constructor(None);
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(connect_web_socket("wss://example.test", &IndexMap::new(), None));
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("expected connect_web_socket to fail without a constructor"),
        };
        assert_eq!(error.message, "WebSocket transport is not available in this runtime");
    }

    /// Minimal fake socket: records sends and can emit events to its listeners.
    #[derive(Default)]
    struct FakeSocket {
        ready_state: std::sync::atomic::AtomicI32,
        sent: Mutex<Vec<String>>,
        listeners: Mutex<HashMap<WebSocketEventType, Vec<WebSocketListener>>>,
        closed: Mutex<Vec<(Option<i32>, Option<String>)>>,
    }

    impl FakeSocket {
        fn emit(&self, type_: WebSocketEventType, event: Value) {
            let listeners = self
                .listeners
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&type_)
                .cloned()
                .unwrap_or_default();
            for listener in listeners {
                listener(event.clone());
            }
        }
    }

    impl WebSocketLike for FakeSocket {
        fn close(&self, code: Option<i32>, reason: Option<&str>) {
            self.closed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push((code, reason.map(str::to_string)));
        }

        fn send(&self, data: &str) {
            self.sent
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(data.to_string());
        }

        fn ready_state(&self) -> Option<i32> {
            Some(self.ready_state.load(std::sync::atomic::Ordering::SeqCst))
        }

        fn add_event_listener(&self, type_: WebSocketEventType, listener: WebSocketListener) {
            self.listeners
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry(type_)
                .or_default()
                .push(listener);
        }

        fn remove_event_listener(&self, type_: WebSocketEventType, listener: &WebSocketListener) {
            let mut listeners = self.listeners.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(entries) = listeners.get_mut(&type_) {
                entries.retain(|entry| !Arc::ptr_eq(entry, listener));
            }
        }
    }

    fn fake_socket() -> Arc<FakeSocket> {
        Arc::new(FakeSocket::default())
    }

    struct ScriptedBacklogSocket {
        inner: FakeSocket,
        scripts: Arc<Mutex<VecDeque<Vec<Value>>>>,
        requests: Arc<Mutex<Vec<Value>>>,
    }

    impl WebSocketLike for ScriptedBacklogSocket {
        fn close(&self, code: Option<i32>, reason: Option<&str>) { self.inner.close(code, reason); }
        fn ready_state(&self) -> Option<i32> { Some(1) }
        fn add_event_listener(&self, kind: WebSocketEventType, listener: WebSocketListener) {
            self.inner.add_event_listener(kind, listener.clone());
            // Resolve the synthetic handshake after the connector installs its
            // listener; ready_state alone does not emit the browser open event.
            if kind == WebSocketEventType::Open { listener(json!({})); }
        }
        fn remove_event_listener(&self, kind: WebSocketEventType, listener: &WebSocketListener) { self.inner.remove_event_listener(kind, listener); }
        fn send(&self, body: &str) {
            self.requests.lock().unwrap().push(serde_json::from_str(body).unwrap());
            let events = self.scripts.lock().unwrap().pop_front().expect("unexpected retry");
            for event in events { self.inner.emit(WebSocketEventType::Message, json!({"data": event.to_string()})); }
        }
    }

    fn backlog_response(id: &str) -> Vec<Value> {
        vec![
            json!({"type": "response.created", "response": {"id": id}}),
            json!({"type": "response.completed", "response": {"id": id, "status": "completed", "output": [], "usage": {"input_tokens":1,"output_tokens":0,"total_tokens":1}}}),
        ]
    }


    fn tool001_context() -> Context {
        let mut context = context();
        context.tools = Some(vec![crate::types::Tool {
            name: "ipython".to_string(),
            description: "Execute local code".to_string(),
            parameters: json!({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}),
        }]);
        context
    }

    #[test]
    fn tool001_interrupted_history_keeps_definitions_and_truthful_missing_result() {
        let model = model();
        for stop_reason in ["aborted", "error", "toolUse"] {
            let mut context = tool001_context();
            context.messages.push(Message::assistant(AssistantMessage {
                api: model.api.clone(), provider: model.provider.clone(), model: model.id.clone(),
                stop_reason: stop_reason.to_string(),
                content: vec![ContentBlock::ToolCall(crate::types::ToolCall::new("call_unfinished", "ipython", Map::new()))],
                ..Default::default()
            }));
            context.messages.push(Message::user(UserMessage::new(UserContent::Text("Continue the requested work".to_string()), 2)));
            let body = build_request_body(&model, &context, None).unwrap();
            assert_eq!(body["tools"].as_array().unwrap().len(), 1);
            assert_eq!(body["tools"][0]["name"], "ipython");
            assert_eq!(body["tool_choice"], "auto");
            let input = body["input"].as_array().unwrap();
            let outputs: Vec<_> = input.iter().filter(|item| item["type"] == "function_call_output").collect();
            if stop_reason == "toolUse" {
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0]["output"], "No result provided");
            } else {
                assert!(outputs.is_empty(), "aborted/error calls are not replayed as successful work");
                assert!(!input.iter().any(|item| item["type"] == "function_call"));
            }
        }
    }

    #[test]
    fn tool001_payload_hook_and_wire_retain_tools_after_interruption_and_cached_followup() {
        let _guard = WEB_SOCKET_TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let previous_constructor = get_web_socket_constructor();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let scripts = Arc::new(Mutex::new(VecDeque::from(vec![
                vec![json!({"type":"response.function_call_arguments.delta","delta":"{}"}),
                     json!({"type":"error","code":"invalid_request","message":"fixture interruption"})],
                backlog_response("tool001-resumed"),
                backlog_response("tool001-cached"),
            ])));
            let requests = Arc::new(Mutex::new(Vec::new()));
            set_web_socket_constructor(Some(Arc::new({
                let scripts = scripts.clone(); let requests = requests.clone();
                move |_, _| Arc::new(ScriptedBacklogSocket { inner: FakeSocket::default(), scripts: scripts.clone(), requests: requests.clone() })
            })));
            let hooked = Arc::new(Mutex::new(Vec::new()));
            let options = OpenAICodexResponsesOptions { stream: StreamOptions {
                api_key: Some(token("synthetic-tool001")),
                session_id: Some("tool001-offline-interruption".to_string()),
                transport: Some("websocket-cached".to_string()),
                timeout_ms: Some(1000.0),
                on_payload: Some(Arc::new({ let hooked = hooked.clone(); move |body, _| {
                    hooked.lock().unwrap().push(body.clone());
                    Box::pin(async move { Some(body) })
                }})),
                ..Default::default()
            }, ..Default::default() };
            let mut context = tool001_context();
            let mut fixture_model = model();
            fixture_model.id = "gpt-6-astra".to_string();
            let interrupted = stream_openai_codex_responses(&fixture_model, &context, Some(options.clone())).result().await;
            assert_eq!(interrupted.stop_reason, "error");
            assert_eq!(requests.lock().unwrap().len(), 1, "observed work must not be replayed");
            context.messages.push(Message::assistant(interrupted));
            context.messages.push(Message::user(UserMessage::new(UserContent::Text("Resume explicitly".to_string()), 2)));
            let resumed = stream_openai_codex_responses(&fixture_model, &context, Some(options.clone())).result().await;
            assert_ne!(resumed.stop_reason, "error", "{:?}", resumed.error_message);
            context.messages.push(Message::user(UserMessage::new(UserContent::Text("Status only".to_string()), 3)));
            let final_result = stream_openai_codex_responses(&fixture_model, &context, Some(options)).result().await;
            assert_ne!(final_result.stop_reason, "error", "{:?}", final_result.error_message);
            let sent = requests.lock().unwrap().clone();
            assert_eq!(sent.len(), 3);
            assert_eq!(hooked.lock().unwrap().len(), 3);
            for (wire, hook) in sent.iter().zip(hooked.lock().unwrap().iter()) {
                assert_eq!(wire["type"], "response.create");
                assert_eq!(wire["tools"], hook["tools"]);
                assert_eq!(wire["tools"][0]["name"], "ipython");
                assert_eq!(wire["tool_choice"], "auto");
            }
            assert!(sent[1].get("previous_response_id").is_none());
            assert_eq!(sent[2]["previous_response_id"], "tool001-resumed");
            close_openai_codex_web_socket_sessions(Some("tool001-offline-interruption"));
        });
        set_web_socket_constructor(previous_constructor);
    }

    #[test]
    fn backlog_codex_chain_reset_is_bounded_and_never_replays_observed_work() {
        let _guard = WEB_SOCKET_TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let previous_constructor = get_web_socket_constructor();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let stale = json!({"type":"error","code":"previous_response_not_found","message":"gone"});
            let scripts = Arc::new(Mutex::new(VecDeque::from(vec![backlog_response("r1"), vec![json!({"type":"response.created","response":{"id":"discard-me"}}), json!({"type":"response.in_progress"}), stale.clone()], backlog_response("r2"), backlog_response("r3")])));
            let requests = Arc::new(Mutex::new(Vec::new()));
            set_web_socket_constructor(Some(Arc::new({ let scripts=scripts.clone(); let requests=requests.clone(); move |_, _| {
                Arc::new(ScriptedBacklogSocket { inner: FakeSocket::default(), scripts:scripts.clone(), requests:requests.clone() })
            }})));
            let mut options = OpenAICodexResponsesOptions { stream: StreamOptions {
                api_key: Some(token("acc_test")), session_id: Some("backlog-chain-reset".into()), transport: Some("websocket-cached".into()), timeout_ms: Some(1000.0), ..Default::default()
            }, ..Default::default() };
            let mut conversation = context();
            for index in 0..3 {
                if index > 0 { conversation.messages.push(Message::user(UserMessage::new(UserContent::Text(format!("turn {index}")), index))); }
                let output = stream_openai_codex_responses(&model(), &conversation, Some(options.clone())).result().await;
                assert_ne!(output.stop_reason, "error", "{:?}", output.error_message);
            }
            let sent = requests.lock().unwrap().clone();
            assert_eq!(sent.len(), 4);
            assert_eq!(sent[1]["previous_response_id"], "r1");
            assert!(sent[2].get("previous_response_id").is_none());
            assert_eq!(sent[2]["input"].as_array().unwrap().len(), 2);
            assert_eq!(sent[3]["previous_response_id"], "r2");

            for (name, events, expected_requests) in [
                ("metadata", vec![json!({"type":"response.created","response":{"id":"inflight"}}), stale.clone()], 2),
                ("extension", vec![json!({"type":"unrecognized.extension"}), stale.clone()], 2),
                ("text", vec![json!({"type":"response.output_text.delta","delta":"visible"}), stale.clone()], 1),
                ("reasoning", vec![json!({"type":"response.reasoning_summary_text.delta","delta":"visible"}), stale.clone()], 1),
                ("tool", vec![json!({"type":"response.function_call_arguments.delta","delta":"{}"}), stale.clone()], 1),
                ("other", vec![json!({"type":"error","code":"invalid_request","message":"bad"})], 1),
                ("repeat", vec![stale.clone()], 2),
            ] {
                *scripts.lock().unwrap() = VecDeque::from(vec![events.clone(), events]);
                requests.lock().unwrap().clear();
                options.stream.session_id = Some(format!("backlog-chain-{name}"));
                let output = stream_openai_codex_responses(&model(), &context(), Some(options.clone())).result().await;
                assert_eq!(output.stop_reason, "error");
                assert_eq!(requests.lock().unwrap().len(), expected_requests, "{name}");
            }
        });
        set_web_socket_constructor(previous_constructor);
    }

    async fn next_socket_event(
        stream: &mut Pin<Box<dyn futures::Stream<Item = Result<Value, CodexThrown>> + Send>>,
    ) -> Option<Result<Value, CodexThrown>> {
        tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
            .await.expect("socket event must settle")
    }

    fn assert_no_socket_listeners(socket: &FakeSocket) {
        assert!(socket.listeners.lock().unwrap().values().all(Vec::is_empty));
    }

    #[tokio::test]
    async fn parse_web_socket_yields_messages_until_completion() {
        let socket = fake_socket();
        socket.ready_state.store(1, std::sync::atomic::Ordering::SeqCst);
        let events: Vec<Value> = vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({ "type": "response.completed", "response": { "id": "resp_1", "status": "completed" } }),
        ];
        let socket_for_emitter = socket.clone();
        tokio::spawn(async move {
            for event in events {
                socket_for_emitter.emit(WebSocketEventType::Message, json!({ "data": event.to_string() }));
                tokio::task::yield_now().await;
            }
        });

        let stream = parse_web_socket(socket, None);
        let collected: Vec<Value> = stream.map(|event| event.unwrap()).collect().await;
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[1]["type"], json!("response.completed"));
    }

    #[tokio::test]
    async fn parse_web_socket_requires_completion() {
        let socket = fake_socket();
        socket.ready_state.store(1, std::sync::atomic::Ordering::SeqCst);
        let socket_for_emitter = socket.clone();
        tokio::spawn(async move {
            socket_for_emitter.emit(
                WebSocketEventType::Message,
                json!({ "data": json!({ "type": "response.created" }).to_string() }),
            );
            tokio::task::yield_now().await;
            socket_for_emitter.emit(WebSocketEventType::Close, json!({ "code": 1006, "wasClean": false }));
        });

        let mut stream = parse_web_socket(socket.clone(), None);
        let created = next_socket_event(&mut stream).await.unwrap().unwrap();
        assert_eq!(created["type"], json!("response.created"));
        let error = next_socket_event(&mut stream).await.unwrap().unwrap_err();
        assert_eq!(error.message, "WebSocket closed 1006");
        assert!(next_socket_event(&mut stream).await.is_none());
        assert_no_socket_listeners(&socket);
    }

    #[tokio::test]
    async fn parse_web_socket_reports_invalid_json() {
        let socket = fake_socket();
        socket.ready_state.store(1, std::sync::atomic::Ordering::SeqCst);
        let socket_for_emitter = socket.clone();
        tokio::spawn(async move {
            socket_for_emitter.emit(WebSocketEventType::Message, json!({ "data": "not json" }));
        });

        let mut stream = parse_web_socket(socket.clone(), None);
        let error = next_socket_event(&mut stream).await.unwrap().unwrap_err();
        assert!(error.message.starts_with("Invalid Codex WebSocket JSON:"));
        assert_eq!(error.name, "CodexProtocolError");
        assert!(next_socket_event(&mut stream).await.is_none());
        assert_no_socket_listeners(&socket);
    }

    #[tokio::test]
    async fn parse_web_socket_abort_wakes_an_idle_stream_and_exhausts_it() {
        let socket = fake_socket();
        let signal = tokio_util::sync::CancellationToken::new();
        let mut stream = parse_web_socket(socket.clone(), Some(signal.clone()));
        let mut pending = Box::pin(stream.next());
        assert!(futures::poll!(&mut pending).is_pending());
        drop(pending);
        signal.cancel();
        let error = next_socket_event(&mut stream).await.unwrap().unwrap_err();
        assert_eq!(error.message, "Request was aborted");
        assert!(next_socket_event(&mut stream).await.is_none());
        assert_no_socket_listeners(&socket);
    }

    #[tokio::test]
    async fn dropping_a_pending_web_socket_stream_removes_its_listeners() {
        let socket = fake_socket();
        let mut stream = parse_web_socket(socket.clone(), None);
        let mut pending = Box::pin(stream.next());
        assert!(futures::poll!(&mut pending).is_pending());
        assert_eq!(socket.listeners.lock().unwrap().values().map(Vec::len).sum::<usize>(), 3);
        drop(pending);
        drop(stream);
        assert_no_socket_listeners(&socket);
    }

    fn map_events(events: Vec<Result<Value, CodexThrown>>) -> (Vec<Value>, Option<CodexThrown>) {
        let slot: Arc<Mutex<Option<CodexThrown>>> = Arc::new(Mutex::new(None));
        let mapped: Vec<Value> = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(map_codex_events(futures::stream::iter(events), slot.clone()).collect());
        let error = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        (mapped, error)
    }

    #[test]
    fn map_codex_events_normalizes_terminal_events() {
        let (mapped, error) = map_events(vec![
            Ok(json!({ "type": "response.created" })),
            Ok(json!({ "type": "response.done", "response": { "status": "completed", "id": "resp_1" } })),
        ]);
        assert!(error.is_none());
        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0]["type"], json!("response.created"));
        assert_eq!(mapped[1]["type"], json!("response.completed"));
        assert_eq!(mapped[1]["response"]["status"], json!("completed"));
    }

    #[test]
    fn map_codex_incomplete_events_preserve_terminal_kind_even_with_unknown_status() {
        let (mapped, error) = map_events(vec![Ok(json!({
            "type": "response.incomplete",
            "response": { "status": "weird" }
        }))]);
        assert!(error.is_none());
        assert_eq!(mapped[0]["response"]["status"], json!("incomplete"));
    }

    #[test]
    fn map_codex_incomplete_events_keep_explicit_reason() {
        let (mapped, error) = map_events(vec![Ok(json!({
            "type": "response.incomplete",
            "response": { "incomplete_details": { "reason": "max_output_tokens" } }
        }))]);
        assert!(error.is_none());
        assert_eq!(mapped[0]["response"]["status"], "incomplete");
        assert_eq!(mapped[0]["response"]["incomplete_details"]["reason"], "max_output_tokens");
    }

    #[test]
    fn map_codex_events_surfaces_flat_error_events() {
        let (mapped, error) = map_events(vec![Ok(json!({
            "type": "error",
            "code": "invalid_request",
            "message": "Rejected"
        }))]);
        assert!(mapped.is_empty());
        let error = error.unwrap();
        assert_eq!(error.name, "CodexApiError");
        assert_eq!(error.message, "Codex error: Rejected");
        assert_eq!(error.code.as_deref(), Some("invalid_request"));
    }

    #[test]
    fn map_codex_events_surfaces_response_failed_events() {
        let (mapped, error) = map_events(vec![Ok(json!({
            "type": "response.failed",
            "response": { "error": { "code": "server_error", "message": "boom" } }
        }))]);
        assert!(mapped.is_empty());
        let error = error.unwrap();
        assert_eq!(error.message, "boom");
        assert_eq!(error.code.as_deref(), Some("server_error"));
    }

    #[test]
    fn map_codex_events_reports_nested_usage_limit_messages() {
        let (mapped, error) = map_events(vec![Ok(json!({
            "type": "error",
            "status_code": 429,
            "error": { "code": "usage_limit_reached", "message": "raw", "plan_type": "plus" }
        }))]);
        assert!(mapped.is_empty());
        let error = error.unwrap();
        assert_eq!(error.code.as_deref(), Some("usage_limit_reached"));
        assert_eq!(error.status, Some(429));
        assert!(error
            .message
            .starts_with("You have hit your ChatGPT usage limit (plus plan)."));
    }

    #[test]
    fn map_codex_events_falls_back_to_the_serialized_event() {
        let event = json!({ "type": "error" });
        let (_mapped, error) = map_events(vec![Ok(event.clone())]);
        let error = error.unwrap();
        assert_eq!(error.message, format!("Codex error: {}", event));
    }

    #[test]
    fn map_codex_events_forwards_upstream_failures() {
        let (_mapped, error) = map_events(vec![Err(CodexThrown::protocol_error("bad json", None))]);
        assert_eq!(error.unwrap().message, "bad json");
    }

    #[test]
    fn create_codex_request_id_has_the_fallback_shape() {
        let mut env = crate::test_env::ScopedEnv::new();
        env.remove("CODEX_REQUEST_ID_UUID_V4");
        let id = create_codex_request_id();
        assert!(id.starts_with("codex_"));
        assert_eq!(id.split('_').count(), 3);
        assert_eq!(id.rsplit('_').next().unwrap().len(), 8);
    }

    #[test]
    fn stream_requires_an_api_key_and_reports_it_in_the_stream() {
        let mut env = crate::test_env::ScopedEnv::new();
        env.remove("OPENAI_API_KEY");
        let model = model();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(async {
            let stream = stream_openai_codex_responses(
                &model,
                &context(),
                Some(OpenAICodexResponsesOptions::from_base(&StreamOptions::default())),
            );
            stream.result().await
        });
        assert_eq!(result.stop_reason, "error");
        assert_eq!(
            result.error_message.as_deref(),
            Some("No API key for provider: openai-codex")
        );
    }

    #[test]
    fn transport_failure_diagnostics_and_web_socket_state_machine() {
        let _guard = WEB_SOCKET_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        // The unreachable-host case of the WebSocket path is exercised through the
        // injectable constructor: one failure before any event, then an SSE fallback.
        set_web_socket_constructor(None);
        let model = model();
        let options = OpenAICodexResponsesOptions {
            stream: StreamOptions {
                api_key: Some(token("acc_test")),
                session_id: Some("fallback-unit".to_string()),
                transport: Some("auto".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        reset_openai_codex_web_socket_debug_stats(Some("fallback-unit"));
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let error = runtime
            .block_on(process_web_socket_stream(
                "wss://unused.invalid/codex/responses",
                &Map::new(),
                &IndexMap::new(),
                &mut AssistantMessage::default(),
                &create_assistant_message_event_stream(),
                &model,
                Arc::new(|| {}),
                &options,
            ))
            .unwrap_err();
        assert_eq!(error.message, "WebSocket transport is not available in this runtime");
        record_web_socket_failure(options.stream.session_id.as_deref(), &error);
        record_web_socket_sse_fallback(options.stream.session_id.as_deref());
        let stats = get_openai_codex_web_socket_debug_stats("fallback-unit").unwrap();
        assert_eq!(stats.websocket_failures, 1);
        assert_eq!(stats.sse_fallbacks, 1);
        assert_eq!(stats.websocket_fallback_active, Some(true));
        assert!(is_web_socket_sse_fallback_active(Some("fallback-unit")));
        assert!(!is_web_socket_sse_fallback_active(None));
        reset_openai_codex_web_socket_debug_stats(Some("fallback-unit"));
        assert!(!is_web_socket_sse_fallback_active(Some("fallback-unit")));
    }

    #[test]
    fn web_socket_handshake_keeps_the_responses_websockets_beta_header() {
        // openai-codex-responses.ts:812-813 deletes the mixed-case "OpenAI-Beta" key from the
        // lowercase record produced by `headersToRecord`, which is a no-op, so the handshake
        // carries the `OPENAI_BETA_RESPONSES_WEBSOCKETS` value set at
        // openai-codex-responses.ts:1384.
        let _guard = WEB_SOCKET_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let captured: Arc<Mutex<Vec<IndexMap<String, String>>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = captured.clone();
        let constructor: WebSocketConstructor = Arc::new(move |_url: &str, headers: IndexMap<String, String>| {
            recorded
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(headers);
            let socket = fake_socket();
            socket.ready_state.store(1, std::sync::atomic::Ordering::SeqCst);
            let opener = socket.clone();
            tokio::spawn(async move {
                tokio::task::yield_now().await;
                opener.emit(WebSocketEventType::Open, json!({}));
            });
            let socket_like: Arc<dyn WebSocketLike> = socket;
            socket_like
        });
        set_web_socket_constructor(Some(constructor));

        let headers = build_web_socket_headers(None, None, "acc_1", "token", "handshake-unit");
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let socket = runtime
            .block_on(connect_web_socket("wss://example.test", &headers, None))
            .unwrap();
        socket.close(Some(1000), Some("done"));
        set_web_socket_constructor(None);

        let seen = captured.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].get("OpenAI-Beta").map(String::as_str),
            Some(OPENAI_BETA_RESPONSES_WEBSOCKETS)
        );
    }

    #[test]
    fn acquire_web_socket_reuses_and_releases_cached_connections() {
        let _guard = WEB_SOCKET_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let sockets: Arc<Mutex<Vec<Arc<FakeSocket>>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = sockets.clone();
        let constructor: WebSocketConstructor = Arc::new(move |_url: &str, _headers: IndexMap<String, String>| {
            let socket = fake_socket();
            socket.ready_state.store(1, std::sync::atomic::Ordering::SeqCst);
            recorded
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(socket.clone());
            // The real runtime fires "open" asynchronously, after the listeners exist.
            let opener = socket.clone();
            tokio::spawn(async move {
                tokio::task::yield_now().await;
                opener.emit(WebSocketEventType::Open, json!({}));
            });
            let socket_like: Arc<dyn WebSocketLike> = socket;
            socket_like
        });
        set_web_socket_constructor(Some(constructor));

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            reset_openai_codex_web_socket_debug_stats(Some("reuse-unit"));
            close_openai_codex_web_socket_sessions(Some("reuse-unit"));

            let first = acquire_web_socket("wss://example.test", &IndexMap::new(), Some("reuse-unit"), None)
                .await
                .unwrap();
            assert!(!first.reused);
            let mut first = first;
            first.release(Some("reuse-unit"), true);

            let second = acquire_web_socket("wss://example.test", &IndexMap::new(), Some("reuse-unit"), None)
                .await
                .unwrap();
            assert!(second.reused);
            let mut second = second;
            second.release(Some("reuse-unit"), true);
            assert_eq!(sockets.lock().unwrap().len(), 1);

            // `keep: false` closes the socket and drops the cache entry.
            let third = acquire_web_socket("wss://example.test", &IndexMap::new(), Some("reuse-unit"), None)
                .await
                .unwrap();
            let mut third = third;
            third.release(Some("reuse-unit"), false);
            assert!(!web_socket_session_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key("reuse-unit"));

            close_openai_codex_web_socket_sessions(Some("reuse-unit"));
            reset_openai_codex_web_socket_debug_stats(Some("reuse-unit"));
        });
        set_web_socket_constructor(None);
    }

    #[test]
    fn acquire_web_socket_without_a_session_closes_immediately() {
        let _guard = WEB_SOCKET_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        set_web_socket_constructor(None);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(acquire_web_socket("wss://example.test", &IndexMap::new(), None, None));
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("expected acquire_web_socket to fail without a constructor"),
        };
        assert_eq!(error.message, "WebSocket transport is not available in this runtime");
    }

    #[test]
    fn stream_simple_reports_a_missing_api_key_in_the_stream() {
        let mut env = crate::test_env::ScopedEnv::new();
        env.remove("OPENAI_API_KEY");
        let model = model();
        // openai-codex-responses.ts:350-352 throws "No API key for provider: ..." before the
        // stream starts; the port delivers that message through the stream's terminal error
        // instead of aborting the process.
        let stream = stream_simple_openai_codex_responses(&model, &context(), None);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(stream.result());
        assert_eq!(result.stop_reason, "error");
        assert_eq!(
            result.error_message.as_deref(),
            Some("No API key for provider: openai-codex")
        );
    }

    #[test]
    fn response_items_are_recorded_for_cached_context() {
        let model = model();
        let mut output = AssistantMessage::new(
            "openai-codex-responses".to_string(),
            "openai-codex".to_string(),
            "gpt-5.1-codex".to_string(),
            1,
        );
        output.content = vec![ContentBlock::Text(TextContent::new("Hello"))];
        output.response_id = Some("resp_1".to_string());
        let one_message_context = Context::new(None, vec![Message::assistant(output)], None);
        let items = convert_responses_messages(
            &model,
            &one_message_context,
            &|provider: &str| CODEX_TOOL_CALL_PROVIDERS.contains(&provider),
            Some(&ConvertResponsesMessagesOptions {
                include_system_prompt: Some(false),
            }),
        )
        .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], json!("message"));
    }

    #[test]
    fn parse_error_body_maps_status_and_code() {
        let error = parse_error_body(
            400,
            "Bad Request",
            &json!({}),
            "{\"error\":{\"code\":\"invalid_request\",\"message\":\"bad\"}}",
        );
        assert_eq!(error.name, "CodexApiError");
        assert_eq!(error.message, "bad");
        assert_eq!(error.code.as_deref(), Some("invalid_request"));
        assert_eq!(error.status, Some(400));
    }

    #[test]
    fn parse_error_body_uses_the_status_text_when_the_body_is_empty() {
        let error = parse_error_body(500, "Internal Server Error", &json!({}), "");
        assert_eq!(error.message, "Internal Server Error");
        assert_eq!(error.status, Some(500));
    }

    #[test]
    fn parse_error_body_keeps_the_raw_body_when_it_is_not_json() {
        let error = parse_error_body(500, "Internal Server Error", &json!({}), "<html>oops</html>");
        assert_eq!(error.message, "<html>oops</html>");
    }

    #[test]
    fn parse_error_body_takes_the_larger_retry_after() {
        let error = parse_error_body(
            429,
            "Too Many Requests",
            &json!({ "retry-after-ms": "100" }),
            &json!({
                "error": { "code": "usage_limit_reached", "resets_at": (now_ms() as f64 / 1000.0) + 600.0 }
            })
            .to_string(),
        );
        assert_eq!(error.code.as_deref(), Some("usage_limit_reached"));
        assert!(error.retry_after_ms.unwrap() > 100.0);
        assert!(error.message.starts_with("You have hit your ChatGPT usage limit."));
    }

    #[test]
    fn parse_error_body_reads_the_retry_after_header() {
        let error = parse_error_body(503, "Service Unavailable", &json!({ "retry-after": "2" }), "");
        assert_eq!(error.retry_after_ms, Some(2000.0));
    }

    #[test]
    fn build_params_and_stream_helpers_use_the_options_defaults() {
        let base = StreamOptions {
            api_key: Some(token("acc_test")),
            transport: Some("sse".to_string()),
            ..Default::default()
        };
        let typed = OpenAICodexResponsesOptions::from_base(&base);
        assert_eq!(typed.stream.transport.as_deref(), Some("sse"));
        let value = serde_json::to_value(&typed).unwrap();
        assert_eq!(value["transport"], json!("sse"));
    }
}


#[cfg(test)]
mod t15_controls_tests {
	//! T15 owner 'controls': E-04 codex `instructions` default must follow the TS
	//! `context.systemPrompt || "You are a helpful assistant."` contract, which
	//! drops ONLY the empty string (whitespace is truthy in JS and is sent verbatim).

	use super::*;
	use crate::types::{Message, UserContent, UserMessage};
	use serde_json::json;

	fn t15_model() -> Model {
		let mut model = Model::new(
			"gpt-5.1-codex",
			"GPT-5.1 Codex",
			"openai-codex-responses",
			"openai-codex",
			"https://chatgpt.com/backend-api",
		);
		model.reasoning = true;
		model
	}

	fn t15_context(system_prompt: Option<&str>) -> Context {
		Context::new(
			system_prompt.map(|prompt| prompt.to_string()),
			vec![Message::user(UserMessage::new(UserContent::Text("Say hello".to_string()), 1))],
			None,
		)
	}

	/// Baseline expectation: `Some("")` reaches the wire as `""` (unwrap_or_else
	/// only covers None), so this test FAILS on the unmodified tree.
	#[test]
	fn t15_empty_optional_values_match_contract_instructions() {
		let model = t15_model();
		let body = build_request_body(&model, &t15_context(Some("")), None).unwrap();
		assert_eq!(
			body["instructions"],
			json!("You are a helpful assistant."),
			"TS `context.systemPrompt || default` falls back on the empty string"
		);
	}

	/// Guard: the JS `||` contract keeps truthy whitespace verbatim (no overreach).
	#[test]
	fn t15_whitespace_system_prompt_is_sent_verbatim() {
		let model = t15_model();
		let body = build_request_body(&model, &t15_context(Some(" ")), None).unwrap();
		assert_eq!(body["instructions"], json!(" "), "JS `||` only drops the empty string");
		let word = build_request_body(&model, &t15_context(Some("real")), None).unwrap();
		assert_eq!(word["instructions"], json!("real"));
	}
}
