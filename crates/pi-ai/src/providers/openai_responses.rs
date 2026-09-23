//! Port of packages/ai/src/providers/openai-responses.ts

use std::sync::Arc;

use futures::StreamExt;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::compaction::{CompactionOptions, ProviderCompactionResult};
use crate::env_api_keys::get_env_api_key;
use crate::models::clamp_thinking_level;
use crate::providers::cloudflare::{is_cloudflare_provider, resolve_cloudflare_base_url};
use crate::providers::github_copilot_headers::{
    build_copilot_dynamic_headers, has_copilot_vision_input, CopilotDynamicHeaderParams,
};
use crate::providers::openai_compaction::{
    request_openai_compaction, supports_openai_compaction, validated_native_compaction_endpoint,
};
use crate::providers::openai_responses_shared::{
    convert_responses_messages, convert_responses_tools, process_responses_stream, ConvertResponsesMessagesOptions,
    OpenAIResponsesStreamOptions, ResponsesEventStream, ResponsesStreamError,
};
use crate::providers::opencode_headers::with_opencode_headers;
use crate::providers::simple_options::build_base_options;
use crate::types::{
    AssistantMessage, AssistantMessageEvent, CacheRetention, Context, Model, ProviderResponse, ServiceTier,
    SimpleStreamOptions, StreamOptions, Usage,
};
use crate::utils::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStream,
};
use crate::utils::headers::header_map_to_record;
use crate::utils::now_ms;
use crate::utils::stream_failure::{
    format_stream_failure_message, record_stream_failure, stream_failure_from_stop_reason, StreamFailureError,
    ThrownStreamError,
};

/// `export interface OpenAIResponsesOptions extends StreamOptions`.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct OpenAIResponsesOptions {
    #[serde(flatten)]
    pub stream: StreamOptions,
    /// `reasoningEffort?: "minimal" | "low" | "medium" | "high" | "xhigh" | "max"`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// `reasoningSummary?: "auto" | "detailed" | "concise" | null`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_summary: Option<Option<String>>,
    /// `serviceTier?: ResponseCreateParamsStreaming["service_tier"]`
    ///
    /// openai-responses.ts:97 re-declares `serviceTier?: ... | null` on
    /// `OpenAIResponsesOptions extends StreamOptions`; it is ONE property (types.ts:73
    /// `ServiceTier = ... | null`, types.ts:103), which `streamSimpleOpenAIResponses`
    /// (openai-responses.ts:188-195) carries through `{...base}` from
    /// `buildBaseOptions` (simple-options.ts:10 `serviceTier: options?.serviceTier`).
    /// `service_tier` is the same nullable shape: `None` = key absent, `Some(None)` = explicit
    /// JSON `null` (types.ts:195-197).
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "deserialize_service_tier")]
    pub service_tier: ServiceTier,
}

/// `null` must stay an explicit `null`, not collapse into "absent"
/// (`deserialize_optional_nullable`, types.rs:199-205).
fn deserialize_service_tier<'de, D>(deserializer: D) -> Result<ServiceTier, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Some(Option::<String>::deserialize(deserializer)?))
}

impl OpenAIResponsesOptions {
    /// TS: the caller passes `StreamOptions & Record<string, unknown>`; this keeps the
    /// non-serializable fields (signal, on_payload, on_response, on_usage_observation).
    ///
    /// openai-responses.ts:192-195 spreads the base options into the provider options, so the
    /// inherited `serviceTier` (simple-options.ts:10) must survive here: `buildParams` reads
    /// `options?.serviceTier` off those same options (openai-responses.ts:279-281). Hardcoding
    /// `None` dropped the tier on every request (`build_params` then omitted `service_tier`,
    /// which the API reads as "auto"), including the explicit `"default"` the agent always sets
    /// (agent.ts:78).
    pub fn from_base(base: &StreamOptions) -> Self {
        Self {
            stream: base.clone(),
            reasoning_effort: None,
            reasoning_summary: None,
            service_tier: base.service_tier.clone(),
        }
    }
}

/// `const OPENAI_TOOL_CALL_PROVIDERS = new Set(["openai", "openai-codex", "opencode"])`.
pub const OPENAI_TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];
/// `const AZURE_MANAGED_COMPACTION_TOOL_CALL_PROVIDERS = new Set([...OPENAI_TOOL_CALL_PROVIDERS, "azure-openai-managed"])`.
pub const AZURE_MANAGED_COMPACTION_TOOL_CALL_PROVIDERS: [&str; 4] =
    ["openai", "openai-codex", "opencode", "azure-openai-managed"];

/// Owns the SDK-style error JSON so [`RunError::Value`] can borrow it like
/// `ThrownStreamError::Value(&Value)` does for the other providers.
struct ThrownValue(Value);

impl ThrownValue {
    fn value(&self) -> &Value {
        &self.0
    }
}

/// The Rust counterpart of a `throw` inside the TypeScript stream body.
enum RunError {
    Failure(StreamFailureError),
    Message(String),
    /// An SDK-style error object (`APIError`) whose fields `extractStreamFailureParts`
    /// reads (`utils/stream-failure.ts:130-167`).
    Value(ThrownValue),
}

impl RunError {
    fn thrown(&self) -> ThrownStreamError<'_> {
        match self {
            RunError::Failure(failure) => ThrownStreamError::Failure(failure),
            RunError::Message(message) => ThrownStreamError::Message(message),
            RunError::Value(value) => ThrownStreamError::Value(value.value()),
        }
    }
}

impl From<String> for RunError {
    fn from(message: String) -> Self {
        RunError::Message(message)
    }
}

/// `compactOpenAIResponses: CompactFunction<"openai-responses">`.
pub async fn compact_openai_responses(
    model: &Model,
    context: &Context,
    options: Option<&CompactionOptions>,
) -> Option<ProviderCompactionResult> {
    try_compact_openai_responses(model, context, options)
        .await
        .unwrap_or(None)
}

/// [`compact_openai_responses`] with the TypeScript `throw` turned into `Err`.
pub async fn try_compact_openai_responses(
    model: &Model,
    context: &Context,
    options: Option<&CompactionOptions>,
) -> Result<Option<ProviderCompactionResult>, String> {
    if !supports_openai_compaction(model) {
        return Ok(None);
    }
    let native_endpoint = validated_native_compaction_endpoint(model);
    let tool_call_providers: &[&str] = if native_endpoint.is_some() {
        &AZURE_MANAGED_COMPACTION_TOOL_CALL_PROVIDERS
    } else {
        &OPENAI_TOOL_CALL_PROVIDERS
    };
    let api_key = options
        .and_then(|options| options.simple.stream.api_key.clone())
        .or_else(|| get_env_api_key(&model.provider));
    let Some(api_key) = api_key else {
        return Err(format!("No API key for provider: {}", model.provider));
    };
    let mut headers: IndexMap<String, String> = model.headers.clone().unwrap_or_default();
    if let Some(options_headers) = options.and_then(|options| options.simple.stream.headers.clone()) {
        for (key, value) in options_headers {
            headers.insert(key, value);
        }
    }
    headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    let instructions = [
        context.system_prompt.clone(),
        options.and_then(|options| options.custom_instructions.clone()),
    ]
    .into_iter()
    .flatten()
    .filter(|instruction| !instruction.is_empty())
    .collect::<Vec<_>>()
    .join("\n\n");
    let url = native_endpoint
        .clone()
        .unwrap_or_else(|| format!("{}/responses/compact", model.base_url.trim_end_matches('/')));

    // `JSON.stringify` drops `undefined` keys, so absent options are omitted while an
    // explicit `null` service tier is kept.
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.id.clone()));
    body.insert(
        "input".to_string(),
        Value::Array(convert_responses_messages(
            model,
            context,
            &|provider: &str| tool_call_providers.contains(&provider),
            Some(&ConvertResponsesMessagesOptions {
                include_system_prompt: Some(false),
            }),
        )?),
    );
    body.insert("instructions".to_string(), Value::String(instructions));
    if let Some(session_id) = options.and_then(|options| options.simple.stream.session_id.clone()) {
        body.insert("prompt_cache_key".to_string(), Value::String(session_id));
    }
    match options.and_then(|options| options.simple.stream.service_tier.clone()) {
        Some(Some(service_tier)) => {
            body.insert("service_tier".to_string(), Value::String(service_tier));
        }
        Some(None) => {
            body.insert("service_tier".to_string(), Value::Null);
        }
        None => {}
    }

    let mut result = request_openai_compaction(model, &url, &headers, body, options, None)
        .await
        .map_err(|error| error.to_string())?;
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

/// `resolveCacheRetention(cacheRetention?)`.
pub fn resolve_cache_retention(cache_retention: Option<&CacheRetention>) -> CacheRetention {
    if let Some(cache_retention) = cache_retention {
        return cache_retention.clone();
    }
    if std::env::var("PI_CACHE_RETENTION").ok().as_deref() == Some("long") {
        return "long".to_string();
    }
    "short".to_string()
}

/// `getCompat(model): Required<OpenAIResponsesCompat>` - returns
/// `(sendSessionIdHeader, supportsLongCacheRetention)`.
pub fn get_compat(model: &Model) -> (bool, bool) {
    let compat = model.compat_responses();
    (
        compat.and_then(|compat| compat.send_session_id_header).unwrap_or(true),
        compat
            .and_then(|compat| compat.supports_long_cache_retention)
            .unwrap_or(true),
    )
}

/// `getPromptCacheRetention(compat, cacheRetention)`.
pub fn get_prompt_cache_retention(
    supports_long_cache_retention: bool,
    cache_retention: &CacheRetention,
) -> Option<String> {
    if cache_retention == "long" && supports_long_cache_retention {
        Some("24h".to_string())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// streamOpenAIResponses
// ---------------------------------------------------------------------------

/// `streamOpenAIResponses: StreamFunction<"openai-responses", OpenAIResponsesOptions>`.
pub fn stream_openai_responses(
    model: &Model,
    context: &Context,
    options: Option<OpenAIResponsesOptions>,
) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let out = stream.clone();
    let model = model.clone();
    let context = context.clone();
    let options = options.unwrap_or_default();
    tokio::spawn(async move {
        let mut output = AssistantMessage::new(model.api.clone(), model.provider.clone(), model.id.clone(), now_ms());
        output.usage = Usage::zero();

        match run_openai_responses(&model, &context, &options, &mut output, &out).await {
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
                output.error_message = Some(format_stream_failure_message(&error.thrown()));
                record_stream_failure(&model, &mut output, &error.thrown());
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

/// The TypeScript async IIFE body of `streamOpenAIResponses`.
async fn run_openai_responses(
    model: &Model,
    context: &Context,
    options: &OpenAIResponsesOptions,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> Result<(), RunError> {
    let api_key = options
        .stream
        .api_key
        .clone()
        .or_else(|| get_env_api_key(&model.provider))
        .unwrap_or_default();
    let cache_retention = resolve_cache_retention(options.stream.cache_retention.as_ref());
    let cache_session_id = if cache_retention == "none" {
        None
    } else {
        options.stream.session_id.clone()
    };
    let client = create_client(
        model,
        context,
        Some(&api_key),
        options.stream.headers.as_ref(),
        cache_session_id.as_deref(),
        options.stream.session_id.as_deref(),
    )?;

    let mut params = build_params(model, context, Some(options)).map_err(RunError::Message)?;
    if let Some(on_payload) = options.stream.on_payload.clone() {
        let next_params = on_payload(Value::Object(params.clone()), model).await;
        if let Some(next_params) = next_params {
            params = match next_params {
                Value::Object(map) => map,
                _ => Map::new(),
            };
        }
    }

    let (events, response_metadata) = match super::responses_transport::try_websocket(model, &client, &params, options)
        .await.map_err(|error| if error.sent {
            RunError::Value(ThrownValue(serde_json::json!({"message":error.message,
                "error":{"code":"responses_request_interrupted","message":error.message}})))
        } else { RunError::Message(error.message) })? {
        Some(result) => result,
        None => {
            let response = send_request(&client, &params, options, &model.provider).await?;
            let mut headers = header_map_to_record(response.headers());
            headers.insert("x-optimus-transport".into(), "sse".into());
            let metadata = ProviderResponse { status: response.status().as_u16() as i64, headers };
            let events: ResponsesEventStream = Box::pin(response.bytes_stream()
                .scan(SseBuffer::default(), |buffer, chunk| {
                    let ready = match chunk {
                        Ok(bytes) => buffer.push(&bytes),
                        Err(error) => {
                            buffer.error = Some(error.to_string());
                            vec![serde_json::json!({"type":"error","code":"responses_request_interrupted","message":"Responses body read failed"})]
                        }
                    };
                    futures::future::ready(Some(ready))
                }).flat_map(futures::stream::iter));
            (events, metadata)
        }
    };
    let request_id = response_metadata.headers.get("x-request-id").cloned();
    if let Some(on_response) = options.stream.on_response.clone() {
        on_response(response_metadata, model).await;
    }
    stream.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let stream_options = OpenAIResponsesStreamOptions {
        on_output_item_done: None,
        // openai-responses.ts:147 `serviceTier: options?.serviceTier`: the request-side tier the
        // process loop falls back to (`response?.service_tier ?? options.serviceTier`,
        // openai-responses-shared.ts:526). A `null` tier prices at 1x either way.
        service_tier: options.service_tier.clone().flatten(),
        resolve_service_tier: None,
        apply_service_tier_pricing: Some(Arc::new({
            let model = model.clone();
            move |usage: &mut Usage, service_tier: Option<&str>| {
                apply_service_tier_pricing(usage, service_tier, &model)
            }
        })),
        on_usage_observation: options.stream.on_usage_observation.clone(),
    };

    let observation_options = options.stream.clone();
    let events = Box::pin(events.inspect(move |event| super::responses_transport::observe_event(&observation_options, event)));
    let parse = process_responses_stream(events, output, stream, model, Some(&stream_options));
    // The producer is a separate task: ending the UI queue alone does not drop
    // a pending SSE body. Cancelling here releases that request as well.
    let parsed = match options.stream.signal.as_ref() {
        Some(signal) => tokio::select! {
            biased;
            _ = signal.cancelled() => return Err(RunError::Message("Request was aborted".to_string())),
            result = parse => result,
        },
        None => parse.await,
    };
    parsed.map_err(|error| match error {
            ResponsesStreamError::StreamFailure(failure) => RunError::Failure(failure),
            ResponsesStreamError::Message(message) => RunError::Message(message),
        })?;

    if options
        .stream
        .signal
        .as_ref()
        .map(|signal| signal.is_cancelled())
        .unwrap_or(false)
    {
        return Err(RunError::Message("Request was aborted".to_string()));
    }

    if output.stop_reason == "aborted" || output.stop_reason == "error" {
        return Err(RunError::Failure(stream_failure_from_stop_reason(
            output.stop_reason_raw.as_deref(),
            request_id.as_deref(),
        )));
    }

    stream.push(AssistantMessageEvent::Done {
        reason: output.stop_reason.clone(),
        message: output.clone(),
    });
    stream.end(None);
    Ok(())
}

/// Terminal `error` stream for a synchronous-configuration failure.
///
/// The TypeScript throws out of `streamSimple` here; a Rust `StreamFunction` returns a stream,
/// so the caller-visible contract of an `AssistantMessageEventStream` (openai-completions.ts:504
/// and azure-openai-responses.ts:140 do the same) is an `error` event carrying the thrown
/// message plus `recordStreamFailure`, never a panic that aborts the whole process.
fn api_key_error_stream(model: &Model, message: &str) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let mut output = AssistantMessage::new(model.api.clone(), model.provider.clone(), model.id.clone(), now_ms());
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

/// `streamSimpleOpenAIResponses: StreamFunction<"openai-responses", SimpleStreamOptions>`.
pub fn stream_simple_openai_responses(
    model: &Model,
    context: &Context,
    options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let api_key = options
        .as_ref()
        .and_then(|options| options.stream.api_key.clone())
        .or_else(|| get_env_api_key(&model.provider));
    let Some(api_key) = api_key else {
        // openai-responses.ts:184-186 `if (!apiKey) { throw new Error(...) }` - a catchable
        // error carrying this exact text, never a process abort. The port cannot throw out of a
        // `StreamFunction`, so it terminates the stream with the same message, like
        // `streamSimpleOpenAICompletions` (openai-completions.ts:504) and azure-openai-responses.ts:140.
        return api_key_error_stream(model, &format!("No API key for provider: {}", model.provider));
    };

    let base = build_base_options(model, options.as_ref(), Some(&api_key));
    let reasoning_effort = resolve_simple_reasoning_effort(model,
        options.as_ref().and_then(|options| options.reasoning.as_deref()));

    let mut typed = OpenAIResponsesOptions::from_base(&base);
    typed.reasoning_effort = reasoning_effort;
    stream_openai_responses(model, context, Some(typed))
}

fn resolve_simple_reasoning_effort(model: &Model, reasoning: Option<&str>) -> Option<String> {
    let level = clamp_thinking_level(model, reasoning?);
    if level != "off" {
        return Some(level);
    }
    // Copilot defaults to server reasoning when omitted. Honor an explicitly
    // advertised off mapping; absent/null mappings keep the existing behavior.
    if model.provider == "github-copilot" {
        model.thinking_level_map_get("off").flatten()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Client / params
// ---------------------------------------------------------------------------

/// `createClient(model, context, apiKey?, optionsHeaders?, cacheSessionId?, conversationId?)`
/// returns the SDK client; the port returns the request description it builds.
pub struct ResponsesClient {
    pub api_key: String,
    pub base_url: String,
    pub default_headers: IndexMap<String, Option<String>>,
}

/// `createClient(...)`.
pub fn create_client(
    model: &Model,
    context: &Context,
    api_key: Option<&str>,
    options_headers: Option<&IndexMap<String, String>>,
    cache_session_id: Option<&str>,
    conversation_id: Option<&str>,
) -> Result<ResponsesClient, String> {
    let mut api_key = api_key.map(str::to_string);
    if api_key.as_deref().map(str::is_empty).unwrap_or(true) {
        match std::env::var("OPENAI_API_KEY") {
            Ok(value) if !value.is_empty() => api_key = Some(value),
            _ => {
                return Err(
                    "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument."
                        .to_string(),
                )
            }
        }
    }
    let api_key = api_key.unwrap_or_default();

    let (send_session_id_header, _supports_long_cache_retention) = get_compat(model);
    let mut headers: IndexMap<String, Option<String>> = IndexMap::new();
    if let Some(model_headers) = model.headers.as_ref() {
        for (key, value) in model_headers {
            headers.insert(key.clone(), Some(value.clone()));
        }
    }
    if model.provider == "github-copilot" {
        let has_images = has_copilot_vision_input(&context.messages);
        let copilot_headers = build_copilot_dynamic_headers(CopilotDynamicHeaderParams {
            messages: &context.messages,
            has_images,
        });
        for (key, value) in copilot_headers {
            headers.insert(key, Some(value));
        }
    }

    if let Some(cache_session_id) = cache_session_id {
        if send_session_id_header {
            headers.insert("session_id".to_string(), Some(cache_session_id.to_string()));
        }
        headers.insert("x-client-request-id".to_string(), Some(cache_session_id.to_string()));
    }

    if let Some(options_headers) = options_headers {
        for (key, value) in options_headers {
            headers.insert(key.clone(), Some(value.clone()));
        }
    }

    let default_headers = if model.provider == "cloudflare-ai-gateway" {
        let mut headers = headers.clone();
        if !headers.contains_key("Authorization") {
            headers.insert("Authorization".to_string(), None);
        }
        headers.insert(
            "cf-aig-authorization".to_string(),
            Some(format!("Bearer {}", api_key)),
        );
        headers
    } else {
        headers
    };

    let base_url = if is_cloudflare_provider(&model.provider) {
        resolve_cloudflare_base_url(model)?
    } else {
        model.base_url.clone()
    };

    Ok(ResponsesClient {
        api_key,
        base_url,
        default_headers: with_opencode_headers(&model.provider, conversation_id, &default_headers),
    })
}

/// `buildParams(model, context, options?)`.
///
/// openai-responses.ts:256 calls `convertResponsesMessages(model, context, ...)` inside the
/// stream IIFE, so a thrown conversion error (the mismatched compaction checkpoint of
/// transform-messages.ts:77) propagates into the catch at openai-responses.ts:161-172: the turn
/// fails with that exact text as `errorMessage` and NO request is sent. The port surfaces the
/// same failure as `Err` (like `build_params` in amazon_bedrock_responses.rs:236-292) instead of
/// defaulting to an empty `input`, which silently dropped the whole conversation context.
pub fn build_params(
    model: &Model,
    context: &Context,
    options: Option<&OpenAIResponsesOptions>,
) -> Result<Map<String, Value>, String> {
    let messages = convert_responses_messages(
        model,
        context,
        &|provider: &str| OPENAI_TOOL_CALL_PROVIDERS.contains(&provider),
        None,
    )?;

    let cache_retention = resolve_cache_retention(options.and_then(|options| options.stream.cache_retention.as_ref()));
    let (_send_session_id_header, supports_long_cache_retention) = get_compat(model);
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.id.clone()));
    params.insert("input".to_string(), Value::Array(messages));
    params.insert("stream".to_string(), Value::Bool(true));
    if cache_retention != "none" {
        if let Some(session_id) = options.and_then(|options| options.stream.session_id.clone()) {
            params.insert("prompt_cache_key".to_string(), Value::String(session_id));
        }
    }
    // openai-responses.ts:265 `prompt_cache_retention: getPromptCacheRetention(compat, cacheRetention)`
    // where getPromptCacheRetention (openai-responses.ts:87-92) returns `"24h" | undefined`.
    // JSON.stringify drops undefined keys, so when the retention is not long-lived TS omits the
    // field entirely; sending an explicit `null` is a wire shape TS never produces.
    if let Some(retention) = get_prompt_cache_retention(supports_long_cache_retention, &cache_retention) {
        params.insert("prompt_cache_retention".to_string(), Value::String(retention));
    }
    params.insert("store".to_string(), Value::Bool(false));

    if let Some(max_tokens) = options.and_then(|options| options.stream.max_tokens) {
        if max_tokens != 0.0 {
            if !max_tokens.is_finite() || max_tokens < 0.0 || max_tokens.fract() != 0.0 || max_tokens >= u64::MAX as f64 {
                return Err("max_output_tokens must be a positive integer".to_string());
            }
            // Rust preserves f64's `.0`; strict Responses endpoints require an integer.
            params.insert(
                "max_output_tokens".to_string(),
                Value::Number((max_tokens as u64).into()),
            );
        }
    }

    if let Some(temperature) = options.and_then(|options| options.stream.temperature) {
        params.insert(
            "temperature".to_string(),
            serde_json::Number::from_f64(temperature)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
    }

    // openai-responses.ts:277-281:
    // `if (options?.serviceTier !== undefined && model.provider !== "github-copilot") {
    //    params.service_tier = options.serviceTier; }`
    // GitHub Copilot rejects the service_tier FIELD itself (400) for every value.
    // Elsewhere it is always sent: absence means "auto" (project tier), not "default".
    // `!== undefined` is true for an explicit `null`, and `JSON.stringify` keeps that key, so
    // `Some(None)` must be written as JSON `null` rather than dropped.
    if model.provider != "github-copilot" {
        if let Some(service_tier) = options.and_then(|options| options.service_tier.clone()) {
            params.insert(
                "service_tier".to_string(),
                service_tier.map(Value::String).unwrap_or(Value::Null),
            );
        }
    }

    if let Some(tools) = context.tools.as_ref() {
        if !tools.is_empty() {
            params.insert("tools".to_string(), Value::Array(convert_responses_tools(tools, None)));
        }
    }

    if model.reasoning {
        let reasoning_effort = options.and_then(|options| options.reasoning_effort.clone());
        let reasoning_summary = options.and_then(|options| options.reasoning_summary.clone()).flatten();
        if reasoning_effort.is_some() || reasoning_summary.is_some() {
            let effort = match reasoning_effort {
                Some(effort) => model
                    .thinking_level_map_get(&effort)
                    .flatten()
                    .unwrap_or(effort),
                None => "medium".to_string(),
            };
            let mut reasoning = Map::new();
            reasoning.insert("effort".to_string(), Value::String(effort));
            reasoning.insert(
                "summary".to_string(),
                Value::String(reasoning_summary.unwrap_or_else(|| "auto".to_string())),
            );
            params.insert("reasoning".to_string(), Value::Object(reasoning));
            params.insert(
                "include".to_string(),
                Value::Array(vec![Value::String("reasoning.encrypted_content".to_string())]),
            );
        } else if model.provider != "github-copilot"
            && !matches!(model.thinking_level_map_get("off"), Some(None))
        {
            let effort = model
                .thinking_level_map_get("off")
                .flatten()
                .unwrap_or_else(|| "none".to_string());
            let mut reasoning = Map::new();
            reasoning.insert("effort".to_string(), Value::String(effort));
            params.insert("reasoning".to_string(), Value::Object(reasoning));
        }
    }

    Ok(params)
}

/// `getServiceTierCostMultiplier(model, serviceTier)`.
pub fn get_service_tier_cost_multiplier(model: &Model, service_tier: Option<&str>) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority") => {
            if model.id == "gpt-5.5" {
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
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

/// `client.responses.create(params, requestOptions).withResponse()`.
async fn send_request(
    client: &ResponsesClient,
    params: &Map<String, Value>,
    options: &OpenAIResponsesOptions,
    provider: &str,
) -> Result<reqwest::Response, RunError> {
    let url = format!("{}/responses", client.base_url.trim_end_matches('/'));
    let mut headers = reqwest::header::HeaderMap::new();
    for (key, value) in client.default_headers.iter() {
        let Some(value) = value else {
            continue;
        };
        if let (Ok(name), Ok(header_value)) = (
            reqwest::header::HeaderName::from_bytes(key.as_bytes()),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            headers.insert(name, header_value);
        }
    }
    if !headers.contains_key(reqwest::header::AUTHORIZATION) {
        if let Ok(value) = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", client.api_key)) {
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
    }

    let mut request = super::responses_transport::http_client(provider)
        .post(&url)
        .headers(headers)
        .json(&Value::Object(params.clone()));
    if let Some(timeout_ms) = options.stream.timeout_ms {
        request = request.timeout(std::time::Duration::from_millis(timeout_ms.max(0.0) as u64));
    }

    let send = request.send();
    let response = match options.stream.signal.as_ref() {
        Some(signal) => tokio::select! {
            _ = signal.cancelled() => return Err(RunError::Message("Request was aborted".to_string())),
            result = send => result,
        },
        None => send.await,
    };
    let response = response.map_err(|error| RunError::Message(error.to_string()))?;
    if !response.status().is_success() {
        return Err(RunError::Value(api_error_from_response(response).await));
    }
    Ok(response)
}

/// The OpenAI SDK `APIError.generate(status, error, message, headers)` failure value.
///
/// TS: `client.responses.create(...)` rejects with the SDK `APIError`
/// (`openai-responses.ts:140`, thrown at `openai-responses.ts:168`), whose `status`, `headers`,
/// `error` body and `message` are what `extractStreamFailureParts` reads
/// (`utils/stream-failure.ts:142-167`).
async fn api_error_from_response(response: reqwest::Response) -> ThrownValue {
    let status = response.status().as_u16() as i64;
    let headers = header_map_to_record(response.headers());
    let text = response.text().await.unwrap_or_default();
    let body: Option<Value> = serde_json::from_str(&text).ok();
    // `APIError.makeMessage(status, error, message)`: the parsed body's `error.message`
    // when present, otherwise the raw text, otherwise "<status> status code (no body)".
    let error_message = body
        .as_ref()
        .and_then(|body| body.get("error"))
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let message = match error_message {
        Some(message) if !message.is_empty() => format!("{status} {message}"),
        _ if !text.is_empty() => format!("{status} {text}"),
        _ => format!("{status} status code (no body)"),
    };
    let mut object = Map::new();
    object.insert("name".to_string(), Value::String("APIError".to_string()));
    object.insert("message".to_string(), Value::String(message));
    object.insert("status".to_string(), Value::Number(status.into()));
    object.insert(
        "headers".to_string(),
        Value::Object(
            headers
                .iter()
                .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                .collect(),
        ),
    );
    // The SDK sets `error` to the body's `error` object when it is one, and otherwise to the
    // whole response body (`errorFromResponse(errorResponse) ?? errorResponse`), which is what
    // `extractStreamFailureParts` then reads for the provider type and message.
    let body_error = body.and_then(|body| match body.get("error") {
        Some(Value::Object(_)) => body.get("error").cloned(),
        _ => Some(body.clone()),
    });
    if let Some(error) = body_error {
        object.insert("error".to_string(), error);
    }
    ThrownValue(Value::Object(object))
}

/// Local SSE buffer for the OpenAI Responses transport (`data: ...` frames).
#[derive(Default)]
pub struct SseBuffer {
    line: Vec<u8>,
    data: Vec<String>,
    skip_lf: bool,
    pub error: Option<String>,
}

impl SseBuffer {
    /// Appends a chunk and returns the complete `data:` payloads it completed.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Value> {
        let mut events: Vec<Value> = Vec::new();
        for &byte in bytes {
            if self.skip_lf {
                self.skip_lf = false;
                if byte == b'\n' { continue; }
            }
            if byte != b'\r' && byte != b'\n' {
                self.line.push(byte);
                continue;
            }
            self.skip_lf = byte == b'\r';
            // Decode only complete lines: a UTF-8 scalar can span body chunks.
            if self.line.is_empty() {
                let data = self.data.join("\n");
                self.data.clear();
                if !data.is_empty() && data.trim() != "[DONE]" {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&data) {
                        events.push(parsed);
                    }
                }
            } else {
                let line = String::from_utf8_lossy(&self.line);
                if let Some(data) = line.strip_prefix("data:") {
                    self.data.push(data.strip_prefix(' ').unwrap_or(data).to_string());
                }
                self.line.clear();
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{InputModality, ModelCost, Tool};
    use indexmap::IndexMap;
    use serde_json::json;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::Notify;
    use tokio_util::sync::CancellationToken;

    fn model() -> Model {
        let mut model = Model::new("gpt-5.4", "GPT-5.4", "openai-responses", "openai", "https://api.openai.com/v1");
        model.reasoning = true;
        model.input = vec![InputModality::Text];
        model.cost = ModelCost::default();
        model
    }

    #[tokio::test]
    async fn sse_cancellation_disconnects_pending_body_without_replay() {
        for provider in ["azure-openai-managed", "github-copilot"] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 4096];
                // Drain the POST body before using the read half as a disconnect oracle.
                loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    assert!(read > 0);
                    request.extend_from_slice(&buffer[..read]);
                    assert!(request.len() < 64 * 1024);
                    if let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..header_end]);
                        assert!(headers.starts_with("POST /responses "));
                        let length = headers.lines().find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length").then(|| value.trim().parse::<usize>().unwrap())
                        }).unwrap();
                        if request.len() >= header_end + 4 + length { break; }
                    }
                }
                let frame = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"pending\"}}\n\n";
                socket.write_all(format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n",
                    frame.len(), frame,
                ).as_bytes()).await.unwrap();
                // No terminal event or end-of-body is sent. Only client cancellation
                // can close this connection; finishing the UI stream is insufficient.
                let closed = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buffer))
                    .await.expect("cancelled provider left the SSE socket open");
                match closed {
                    Ok(0) => {}
                    Err(error) if matches!(error.kind(), std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::ConnectionAborted | std::io::ErrorKind::BrokenPipe) => {}
                    other => panic!("expected client disconnect, got {other:?}"),
                }
                assert!(tokio::time::timeout(Duration::from_millis(100), listener.accept()).await.is_err(),
                    "cancelled request was replayed");
            });
            let mut model = model();
            model.provider = provider.into();
            model.base_url = format!("http://{address}");
            let cancel = CancellationToken::new();
            let raw_event = Arc::new(Notify::new());
            let observed = raw_event.clone();
            let options = OpenAIResponsesOptions { stream: StreamOptions {
                api_key: Some("fake-test-key".into()), signal: Some(cancel.clone()),
                transport: Some("sse".into()), session_id: Some("pending-body-test".into()),
                on_stream_observation: Some(Arc::new(move |phase| {
                    if phase == "raw_event" { observed.notify_one(); }
                })),
                ..Default::default()
            }, ..Default::default() };
            let events = stream_openai_responses(&model, &Context::default(), Some(options));
            tokio::time::timeout(Duration::from_secs(5), raw_event.notified()).await.unwrap();
            assert!(matches!(events.next().await, Some(AssistantMessageEvent::Start { .. })));
            cancel.cancel();
            let end = tokio::time::timeout(Duration::from_secs(5), events.next()).await.unwrap();
            assert!(matches!(end, Some(AssistantMessageEvent::Error { reason, .. }) if reason == "aborted"));
            assert!(events.next().await.is_none());
            server.await.unwrap();
        }
    }

    #[test]
    fn subscription_models_copilot_off_mapping_survives_simple_options() {
        for id in ["gpt-6-sol", "gpt-6-luna"] {
            let model = crate::models::get_model("github-copilot", id).unwrap();
            for (level, expected) in [("off", "none"), ("low", "low"), ("medium", "medium"),
                ("high", "high"), ("xhigh", "xhigh"), ("max", "max")] {
                let options = OpenAIResponsesOptions {
                    reasoning_effort: resolve_simple_reasoning_effort(model, Some(level)),
                    ..Default::default()
                };
                let body = build_params(model, &Context::default(), Some(&options)).unwrap();
                assert_eq!(body["model"], id);
                assert_eq!(body["reasoning"]["effort"], expected);
            }
            assert_eq!(resolve_simple_reasoning_effort(model, None), None);
            // No broad provider default change: an absent mapping still omits off.
            let mut legacy = model.clone();
            legacy.thinking_level_map = None;
            assert_eq!(resolve_simple_reasoning_effort(&legacy, Some("off")), None);
            legacy.provider = "openai".to_string();
            legacy.thinking_level_map = model.thinking_level_map.clone();
            assert_eq!(resolve_simple_reasoning_effort(&legacy, Some("off")), None);
        }
    }

    #[test]
    fn resolve_cache_retention_defaults_to_short() {
        let mut env = crate::test_env::ScopedEnv::new();
        env.remove("PI_CACHE_RETENTION");
        assert_eq!(resolve_cache_retention(None), "short");
        assert_eq!(resolve_cache_retention(Some(&"long".to_string())), "long");
        env.set("PI_CACHE_RETENTION", "long");
        assert_eq!(resolve_cache_retention(None), "long");
        env.remove("PI_CACHE_RETENTION");
    }

    #[test]
    fn compat_defaults_match_typescript() {
        let model = model();
        assert_eq!(get_compat(&model), (true, true));
    }

    #[test]
    fn prompt_cache_retention_only_for_long_retention() {
        assert_eq!(get_prompt_cache_retention(true, &"long".to_string()), Some("24h".to_string()));
        assert_eq!(get_prompt_cache_retention(false, &"long".to_string()), None);
        assert_eq!(get_prompt_cache_retention(true, &"short".to_string()), None);
        assert_eq!(get_prompt_cache_retention(true, &"none".to_string()), None);
    }

    #[test]
    fn service_tier_multipliers_match_typescript() {
        let model = model();
        assert_eq!(get_service_tier_cost_multiplier(&model, Some("flex")), 0.5);
        assert_eq!(get_service_tier_cost_multiplier(&model, Some("priority")), 2.0);
        assert_eq!(get_service_tier_cost_multiplier(&model, Some("default")), 1.0);
        assert_eq!(get_service_tier_cost_multiplier(&model, None), 1.0);

        let gpt_5_5 = Model::new("gpt-5.5", "GPT-5.5", "openai-responses", "openai", "https://api.openai.com/v1");
        assert_eq!(get_service_tier_cost_multiplier(&gpt_5_5, Some("priority")), 2.5);

        let mut usage = Usage {
            input: 1_000_000.0,
            output: 1_000_000.0,
            cache_read: 1_000_000.0,
            cache_write: 1_000_000.0,
            total_tokens: 4_000_000.0,
            cost: crate::types::UsageCost {
                input: 1.0,
                output: 2.0,
                cache_read: 3.0,
                cache_write: 4.0,
                total: 10.0,
            },
        };
        apply_service_tier_pricing(&mut usage, Some("flex"), &model);
        assert_eq!(usage.cost.input, 0.5);
        assert_eq!(usage.cost.output, 1.0);
        assert_eq!(usage.cost.cache_read, 1.5);
        assert_eq!(usage.cost.cache_write, 2.0);
        assert_eq!(usage.cost.total, 5.0);
    }

    #[test]
    fn build_params_sets_defaults_and_omits_absent_optionals() {
        let model = model();
        let context = Context::new(
            Some("be terse".to_string()),
            vec![crate::types::Message::user(crate::types::UserMessage::new(
                crate::types::UserContent::Text("hi".to_string()),
                1,
            ))],
            None,
        );
        let params = build_params(&model, &context, None).unwrap();
        assert_eq!(params["model"], json!("gpt-5.4"));
        assert_eq!(params["stream"], json!(true));
        assert_eq!(params["store"], json!(false));
        // openai-responses.ts:87-92 `getPromptCacheRetention` returns `"24h" | undefined`, and
        // JSON.stringify (openai-responses.ts:265) drops an undefined value, so a non-long-lived
        // retention OMITS the key entirely - TS never sends an explicit null (measured under Node
        // on this host: JSON.stringify({prompt_cache_retention: undefined}) === {"model":"m","store":false}).
        assert!(!params.contains_key("prompt_cache_retention"));
        assert!(!params.contains_key("prompt_cache_key"));
        assert!(!params.contains_key("max_output_tokens"));
        assert!(!params.contains_key("temperature"));
        assert!(!params.contains_key("tools"));
        // reasoning defaults to the "off" mapping when no effort is requested
        assert_eq!(params["reasoning"], json!({ "effort": "none" }));
        assert_eq!(params["input"][0]["role"], json!("developer"));
    }

    #[test]
    fn build_params_sends_service_tier_except_for_copilot() {
        let mut model = model();
        let context = Context::default();
        let options = OpenAIResponsesOptions {
            service_tier: Some(Some("flex".to_string())),
            ..Default::default()
        };
        let params = build_params(&model, &context, Some(&options)).unwrap();
        assert_eq!(params["service_tier"], json!("flex"));

        model.provider = "github-copilot".to_string();
        let params = build_params(&model, &context, Some(&options)).unwrap();
        assert!(!params.contains_key("service_tier"));
    }

    /// openai-responses.ts:277-281 + simple-options.ts:10: the tier the agent always sets
    /// (agent.ts:78 default `"default"`) must survive `from_base` and reach the wire, and an
    /// explicit `null` must stay on the wire as `null` (`!== undefined` is true for null), while
    /// an absent tier leaves the key off entirely.
    #[test]
    fn from_base_forwards_service_tier_onto_the_main_request() {
        let model = model();
        let context = Context::default();

        let base = StreamOptions {
            // The agent sets this every turn (pi-agent-core/src/agent.rs:922).
            service_tier: Some(Some("default".to_string())),
            ..Default::default()
        };
        let options = OpenAIResponsesOptions::from_base(&base);
        assert_eq!(options.service_tier, Some(Some("default".to_string())));
        let params = build_params(&model, &context, Some(&options)).unwrap();
        // Absence would mean "auto" (project tier), so an explicit "default" must stay on the wire.
        assert_eq!(params["service_tier"], json!("default"));

        let flex = OpenAIResponsesOptions::from_base(&StreamOptions {
            service_tier: Some(Some("flex".to_string())),
            ..Default::default()
        });
        assert_eq!(
            build_params(&model, &context, Some(&flex)).unwrap()["service_tier"],
            json!("flex")
        );

        // `serviceTier: null` is the explicit reset TS serializes as `"service_tier": null`.
        let reset = OpenAIResponsesOptions::from_base(&StreamOptions {
            service_tier: Some(None),
            ..Default::default()
        });
        let params = build_params(&model, &context, Some(&reset)).unwrap();
        assert!(params.contains_key("service_tier"));
        assert_eq!(params["service_tier"], json!(null));

        // Absent stays absent.
        let absent = OpenAIResponsesOptions::from_base(&StreamOptions::default());
        assert!(!build_params(&model, &context, Some(&absent)).unwrap().contains_key("service_tier"));

        // ...but github-copilot rejects the FIELD for every value, including an explicit null.
        let mut copilot = model;
        copilot.provider = "github-copilot".to_string();
        assert!(!build_params(&copilot, &context, Some(&reset)).unwrap().contains_key("service_tier"));
    }

    /// The wire shape of `serviceTier` on the provider options: an explicit `null` deserializes to
    /// `Some(None)` (never "absent"), an absent key stays `None`, and only `None` is skipped when
    /// re-serialized (types.rs:195-197 / openai-responses.ts:97).
    #[test]
    fn service_tier_round_trips_through_the_provider_options() {
        let explicit_null: OpenAIResponsesOptions = serde_json::from_value(json!({ "serviceTier": null })).unwrap();
        assert_eq!(explicit_null.service_tier, Some(None));
        assert_eq!(
            serde_json::to_value(&explicit_null).unwrap().get("serviceTier"),
            Some(&Value::Null)
        );

        let absent: OpenAIResponsesOptions = serde_json::from_value(json!({})).unwrap();
        assert_eq!(absent.service_tier, None);
        assert!(serde_json::to_value(&absent).unwrap().get("serviceTier").is_none());

        let tiered: OpenAIResponsesOptions = serde_json::from_value(json!({ "serviceTier": "priority" })).unwrap();
        assert_eq!(tiered.service_tier, Some(Some("priority".to_string())));
    }

    /// openai-responses.ts:256/161-172: a `throw` from `convertResponsesMessages`
    /// (transform-messages.ts:77 mismatched compaction checkpoint) fails the turn with that exact
    /// message; Rust must return `Err` rather than defaulting to an empty `input`, which silently
    /// dropped the whole conversation context and queried the model with an empty prompt.
    #[test]
    fn build_params_propagates_a_foreign_compaction_checkpoint() {
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
        let error = match build_params(&model, &context, None) {
            Ok(params) => panic!(
                "build_params swallowed the foreign compaction checkpoint error and returned Ok; the \
                 request would be sent with input = {}",
                params.get("input").map(Value::to_string).unwrap_or_default()
            ),
            Err(error) => error,
        };
        assert_eq!(
            error,
            "Compaction checkpoint belongs to another model or provider; rebuild context from the session transcript"
        );
    }

    #[test]
    fn build_params_maps_reasoning_effort_through_the_level_map() {
        let mut model = model();
        model.thinking_level_map = Some(
            [
                ("high".to_string(), Some("high-value".to_string())),
                ("off".to_string(), None),
            ]
            .into_iter()
            .collect(),
        );
        let context = Context::default();
        let options = OpenAIResponsesOptions {
            reasoning_effort: Some("high".to_string()),
            ..Default::default()
        };
        let params = build_params(&model, &context, Some(&options)).unwrap();
        assert_eq!(params["reasoning"], json!({ "effort": "high-value", "summary": "auto" }));
        assert_eq!(params["include"], json!(["reasoning.encrypted_content"]));

        // reasoningSummary alone uses the "medium" default effort.
        let options = OpenAIResponsesOptions {
            reasoning_summary: Some(Some("concise".to_string())),
            ..Default::default()
        };
        let params = build_params(&model, &context, Some(&options)).unwrap();
        assert_eq!(params["reasoning"], json!({ "effort": "medium", "summary": "concise" }));
    }

    #[test]
    fn build_params_omits_reasoning_when_off_is_null() {
        let mut model = model();
        model.thinking_level_map = Some([("off".to_string(), None)].into_iter().collect());
        let params = build_params(&model, &Context::default(), None).unwrap();
        assert!(!params.contains_key("reasoning"));
    }

    #[test]
    fn build_params_includes_tools_and_limits() {
        let model = model();
        let context = Context::new(
            None,
            Vec::new(),
            Some(vec![Tool {
                name: "bash".to_string(),
                description: "Run a command".to_string(),
                parameters: json!({ "type": "object" }),
            }]),
        );
        let options = OpenAIResponsesOptions {
            stream: StreamOptions {
                max_tokens: Some(4096.0),
                temperature: Some(0.5),
                session_id: Some("session-1".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let params = build_params(&model, &context, Some(&options)).unwrap();
        assert_eq!(params["max_output_tokens"], json!(4096));
        assert_eq!(params["temperature"], json!(0.5));
        assert_eq!(params["prompt_cache_key"], json!("session-1"));
        assert_eq!(params["tools"][0]["strict"], json!(false));
    }

    #[test]
    fn create_client_sets_session_and_copilot_headers() {
        let model = model();
        let context = Context::default();
        let client = create_client(&model, &context, Some("key"), None, Some("cache-1"), Some("conv-1")).unwrap();
        assert_eq!(
            client.default_headers.get("session_id"),
            Some(&Some("cache-1".to_string()))
        );
        assert_eq!(
            client.default_headers.get("x-client-request-id"),
            Some(&Some("cache-1".to_string()))
        );
        assert_eq!(client.base_url, "https://api.openai.com/v1");

        let mut copilot = model.clone();
        copilot.provider = "github-copilot".to_string();
        let client = create_client(&copilot, &context, Some("key"), None, None, None).unwrap();
        assert_eq!(
            client.default_headers.get("X-Initiator"),
            Some(&Some("user".to_string()))
        );
        assert_eq!(
            client.default_headers.get("Openai-Intent"),
            Some(&Some("conversation-edits".to_string()))
        );
    }

    #[test]
    fn create_client_requires_an_api_key() {
        // Held for the whole body: `create_client` reads the process-global
        // OPENAI_API_KEY, so a parallel test must not set it in between.
        let mut env = crate::test_env::ScopedEnv::new();
        let model = model();
        env.remove("OPENAI_API_KEY");
        let error = match create_client(&model, &Context::default(), Some(""), None, None, None) {
            Err(error) => error,
            Ok(_) => panic!("expected create_client to fail without an API key"),
        };
        assert_eq!(
            error,
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument."
        );
    }

    #[test]
    fn cloudflare_gateway_sets_the_gateway_authorization_header() {
        let mut model = model();
        model.provider = "cloudflare-ai-gateway".to_string();
        model.base_url = "https://gateway.ai.cloudflare.com/v1/acct/gw/openai".to_string();
        let client = create_client(&model, &Context::default(), Some("key"), None, None, None).unwrap();
        assert_eq!(
            client.default_headers.get("cf-aig-authorization"),
            Some(&Some("Bearer key".to_string()))
        );
        // `Authorization: headers.Authorization ?? null` keeps the key with a null value.
        assert_eq!(client.default_headers.get("Authorization"), Some(&None));
    }

    #[test]
    fn opencode_headers_are_applied_last() {
        let mut model = model();
        model.provider = "opencode".to_string();
        let mut headers: IndexMap<String, String> = IndexMap::new();
        headers.insert("X-Test".to_string(), "1".to_string());
        let client = create_client(&model, &Context::default(), Some("key"), Some(&headers), None, Some("session-1"))
            .unwrap();
        assert_eq!(
            client.default_headers.get("User-Agent"),
            Some(&Some("prime-agent".to_string()))
        );
        assert_eq!(
            client.default_headers.get("x-opencode-session"),
            Some(&Some("session-1".to_string()))
        );
        assert_eq!(client.default_headers.get("x-test"), Some(&Some("1".to_string())));
    }

    #[test]
    fn sse_buffer_accepts_crlf_and_preserves_split_unicode_immediately() {
        let frame = "event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Ready ✓ 日本\"}\r\n\r\n";
        let mut buffer = SseBuffer::default();
        let events: Vec<_> = frame.as_bytes().iter().flat_map(|byte| buffer.push(&[*byte])).collect();
        assert_eq!(events.len(), 1, "a complete frame must not wait for stream close");
        assert_eq!(events[0]["delta"], json!("Ready ✓ 日本"));
    }

    #[test]
    fn sse_buffer_handles_mixed_newlines_multiline_and_comments() {
        let mut buffer = SseBuffer::default();
        let events = buffer.push(b": keepalive\r\rdata: {\"type\":\rdata: \"response.created\"}\r\rdata: [DONE]\n\n");
        assert_eq!(events, vec![json!({"type":"response.created"})]);
    }

    #[test]
    fn sse_buffer_splits_frames_and_skips_done() {
        let mut buffer = SseBuffer::default();
        let first = buffer.push(b"data: {\"type\":\"response.created\"}\n\ndata: {\"type\":");
        assert_eq!(first.len(), 1);
        assert_eq!(first[0]["type"], json!("response.created"));
        let second = buffer.push(b"\"response.completed\"}\n\ndata: [DONE]\n\n");
        assert_eq!(second.len(), 1);
        assert_eq!(second[0]["type"], json!("response.completed"));
    }

    #[test]
    fn options_round_trip_from_base_options_keeps_callbacks() {
        let base = StreamOptions {
            temperature: Some(0.2),
            session_id: Some("session".to_string()),
            ..Default::default()
        };
        let typed = OpenAIResponsesOptions::from_base(&base);
        assert_eq!(typed.stream.temperature, Some(0.2));
        assert_eq!(typed.reasoning_effort, None);
        // serde round-trip through register_builtins keeps the serializable fields.
        let value = serde_json::to_value(&typed).unwrap();
        assert_eq!(value["temperature"], json!(0.2));
        assert_eq!(value["sessionId"], json!("session"));
    }

    #[test]
    fn try_compact_returns_none_for_unsupported_models() {
        let model = Model::new("gpt-4o", "GPT-4o", "openai-responses", "openai", "https://api.openai.com/v1");
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(try_compact_openai_responses(&model, &Context::default(), None))
            .unwrap();
        assert!(result.is_none());
    }
}
