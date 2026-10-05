//! Port of packages/ai/src/providers/mistral.ts
//!
//! The TypeScript uses `@mistralai/mistralai` 2.2.1 (`mistral.chat.stream`). This port
//! builds the same `POST {serverURL}/v1/chat/completions` request itself with `reqwest`
//! and parses the SSE stream locally, keeping the SDK's outbound JSON (`snake_case`
//! remaps), header merge order and error text.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::env_api_keys::get_env_api_key;
use crate::models::{calculate_cost, clamp_thinking_level};
use crate::providers::simple_options::build_base_options;
use crate::providers::transform_messages::try_transform_messages;
use crate::types::{
	AssistantMessage, AssistantMessageEvent, ContentBlock, Context, Message, Model, SimpleStreamOptions, StopReason,
	StreamOptions, TextContent, ThinkingContent, ToolCall, Usage,
};
use crate::utils::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use crate::utils::hash::short_hash;
use crate::utils::headers::header_map_to_record;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::now_ms;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::stream_failure::{
	record_stream_failure, stream_failure_from_stop_reason, StreamFailureError, ThrownStreamError,
};

const MISTRAL_TOOL_CALL_ID_LENGTH: usize = 9;
const MAX_MISTRAL_ERROR_BODY_CHARS: usize = 4000;

/// `SDK_METADATA.userAgent` of the pinned `@mistralai/mistralai` 2.2.1 client.
const MISTRAL_USER_AGENT: &str = "speakeasy-sdk/typescript 2.2.1 2.881.4 1.0.0 @mistralai/mistralai";
/// `ServerList[ServerEu]` - the SDK default when `serverURL` is not set.
const MISTRAL_DEFAULT_SERVER_URL: &str = "https://api.mistral.ai";
/// `timeoutMs: options?.timeoutMs || client._options.timeoutMs || 30000`.
const MISTRAL_DEFAULT_TIMEOUT_MS: u64 = 30000;

/// Mistral reasoning-effort values.
pub type MistralReasoningEffort = String;

/// TS: `interface MistralOptions extends StreamOptions`.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MistralOptions {
	#[serde(flatten)]
	pub stream: StreamOptions,
	/// `"auto" | "none" | "any" | "required" | { type: "function"; function: { name: string } }`
	pub tool_choice: Option<Value>,
	pub prompt_mode: Option<String>,
	pub reasoning_effort: Option<MistralReasoningEffort>,
}

impl MistralOptions {
	/// TS: the caller passes `StreamOptions & Record<string, unknown>`; this keeps the
	/// non-serializable fields (signal, on_payload, on_response, on_usage_observation).
	pub fn from_base(base: &StreamOptions) -> Self {
		Self {
			stream: base.clone(),
			tool_choice: None,
			prompt_mode: None,
			reasoning_effort: None,
		}
	}
}

impl std::fmt::Debug for MistralOptions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("MistralOptions")
			.field("stream", &self.stream)
			.field("tool_choice", &self.tool_choice)
			.field("prompt_mode", &self.prompt_mode)
			.field("reasoning_effort", &self.reasoning_effort)
			.finish()
	}
}

/// The TypeScript throws either a plain `Error`, a `StreamFailureError` or a Mistral
/// SDK error (`MistralError`/`SDKError` with `statusCode` and `body`).
#[derive(Debug, Clone, PartialEq)]
pub enum MistralStreamError {
	Failure(StreamFailureError),
	Message(String),
	/// TS: a Mistral SDK error instance. `message` is the SDK-built message,
	/// `statusCode`/`body` feed `formatMistralError`.
	Api {
		message: String,
		status_code: Option<i64>,
		body: Option<String>,
		value: Value,
	},
}

impl MistralStreamError {
	fn as_thrown(&self) -> ThrownStreamError<'_> {
		match self {
			MistralStreamError::Failure(failure) => ThrownStreamError::Failure(failure),
			MistralStreamError::Message(message) => ThrownStreamError::Message(message),
			MistralStreamError::Api { value, .. } => ThrownStreamError::Value(value),
		}
	}

	/// `error instanceof Error ? error.message : ...`
	fn error_message(&self) -> String {
		match self {
			MistralStreamError::Failure(failure) => failure.message.clone(),
			MistralStreamError::Message(message) => message.clone(),
			MistralStreamError::Api { message, .. } => message.clone(),
		}
	}

	/// `error as Error & { statusCode?, body? }` for `formatMistralError`.
	fn status_code(&self) -> Option<i64> {
		match self {
			MistralStreamError::Api { status_code, .. } => *status_code,
			_ => None,
		}
	}

	fn body(&self) -> Option<String> {
		match self {
			MistralStreamError::Api { body, .. } => body.clone(),
			_ => None,
		}
	}
}

impl std::fmt::Display for MistralStreamError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(formatter, "{}", self.error_message())
	}
}

impl std::error::Error for MistralStreamError {}

/// Streams Mistral chat completions through `chat.stream`.
///
/// TS: `streamMistral: StreamFunction<"mistral-conversations", MistralOptions>`.
pub fn stream_mistral(
	model: &Model,
	context: &Context,
	options: Option<MistralOptions>,
) -> AssistantMessageEventStream {
	let stream = create_assistant_message_event_stream();
	let out = stream.clone();
	let model = model.clone();
	let context = context.clone();
	let options = options.unwrap_or_default();
	tokio::spawn(async move {
		let mut output = create_output(&model);

		match run_stream_mistral(&model, &context, &options, &mut output, &out).await {
			Ok(()) => {}
			Err(error) => {
				// partialArgs is only a streaming scratch buffer; never persist it.
				// (The Rust port keeps it outside the block, so there is nothing to delete.)
				let aborted = options
					.stream
					.signal
					.as_ref()
					.map(|signal| signal.is_cancelled())
					.unwrap_or(false);
				output.stop_reason = if aborted { "aborted".to_string() } else { "error".to_string() };
				output.error_message = Some(format_mistral_error(&error));
				record_stream_failure(&model, &mut output, &error.as_thrown());
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

/// The TypeScript async IIFE body of `streamMistral`.
async fn run_stream_mistral(
	model: &Model,
	context: &Context,
	options: &MistralOptions,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
) -> Result<(), MistralStreamError> {
	let api_key = options
		.stream
		.api_key
		.clone()
		.or_else(|| get_env_api_key(&model.provider));
	let Some(api_key) = api_key else {
		return Err(MistralStreamError::Message(format!(
			"No API key for provider: {}",
			model.provider
		)));
	};

	// Intentionally per-request: avoids shared SDK mutable state across concurrent consumers.
	let server_url = if model.base_url.is_empty() {
		MISTRAL_DEFAULT_SERVER_URL.to_string()
	} else {
		model.base_url.clone()
	};

	let normalizer = MistralToolCallIdNormalizer::new();
	let normalize = |id: &str, _model: &Model, _source: &AssistantMessage| normalizer.normalize(id);
	let transformed_messages = try_transform_messages(context.messages.clone(), model, Some(&normalize))
		.map_err(MistralStreamError::Message)?;

	let mut payload = build_chat_payload(model, context, &transformed_messages, Some(options));
	if let Some(on_payload) = options.stream.on_payload.clone() {
		let next_payload = on_payload(Value::Object(payload.clone()), model).await;
		if let Some(next_payload) = next_payload {
			payload = match next_payload {
				Value::Object(map) => map,
				_ => Map::new(),
			};
		}
	}
	// TS: the SDK remaps the hook's result inside `chat.stream`
	// (chatcompletionstreamrequest.js:104-118), so hook-added keys are remapped too.
	wire_chat_payload(&mut payload);
	let response = send_request(&server_url, &api_key, &payload, model, options).await?;
	if let Some(on_response) = options.stream.on_response.clone() {
		on_response(
			crate::types::ProviderResponse {
				status: response.status().as_u16() as i64,
				headers: header_map_to_record(response.headers()),
			},
			model,
		)
		.await;
	}

	stream.push(AssistantMessageEvent::Start {
		partial: output.clone(),
	});
	let mut chunks = MistralChunkStream::new(response, options.stream.signal.clone());
	consume_chat_stream(model, output, stream, &mut chunks).await?;

	if options
		.stream
		.signal
		.as_ref()
		.map(|signal| signal.is_cancelled())
		.unwrap_or(false)
	{
		return Err(MistralStreamError::Message("Request was aborted".to_string()));
	}

	if output.stop_reason == "aborted" || output.stop_reason == "error" {
		return Err(MistralStreamError::Failure(stream_failure_from_stop_reason(
			output.stop_reason_raw.as_deref(),
			None,
		)));
	}

	stream.push(AssistantMessageEvent::Done {
		reason: output.stop_reason.clone(),
		message: output.clone(),
	});
	stream.end(None);
	Ok(())
}

/// Maps provider-agnostic `SimpleStreamOptions` to Mistral request options.
///
/// TS: `streamSimpleMistral: StreamFunction<"mistral-conversations", SimpleStreamOptions>`.
pub fn stream_simple_mistral(
	model: &Model,
	context: &Context,
	options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
	let api_key = options
		.as_ref()
		.and_then(|options| options.stream.api_key.clone())
		.or_else(|| get_env_api_key(&model.provider));
	let Some(api_key) = api_key else {
		// mistral.ts:113-116 `const apiKey = options?.apiKey || getEnvApiKey(model.provider);
		// if (!apiKey) { throw new Error(`No API key for provider: ${model.provider}`); }` - a
		// catchable error, never a process abort. A Rust `StreamFunction` returns a stream, so
		// this terminates the stream with the same message, exactly like
		// `streamSimpleGoogle` (google.ts:291-293) and the other ports.
		return api_key_error_stream(model, &format!("No API key for provider: {}", model.provider));
	};

	let base = build_base_options(model, options.as_ref(), Some(&api_key));
	let clamped_reasoning = options
		.as_ref()
		.and_then(|options| options.reasoning.clone())
		.map(|reasoning| clamp_thinking_level(model, &reasoning));
	let reasoning = clamped_reasoning.filter(|reasoning| reasoning != "off");
	let should_use_reasoning = model.reasoning && reasoning.is_some();

	let mut typed = MistralOptions::from_base(&base);
	typed.prompt_mode = if should_use_reasoning && uses_prompt_mode_reasoning(model) {
		Some("reasoning".to_string())
	} else {
		None
	};
	typed.reasoning_effort = if should_use_reasoning && uses_reasoning_effort(model) {
		Some(map_reasoning_effort(model, reasoning.as_deref().unwrap_or_default()))
	} else {
		None
	};
	stream_mistral(model, context, Some(typed))
}

/// TS: `createOutput(model)`.
fn create_output(model: &Model) -> AssistantMessage {
	AssistantMessage {
		content: Vec::new(),
		api: model.api.clone(),
		provider: model.provider.clone(),
		model: model.id.clone(),
		usage: Usage::zero(),
		stop_reason: "stop".to_string(),
		timestamp: now_ms(),
		..Default::default()
	}
}

/// Terminal `error` stream for a synchronous-configuration failure.
///
/// mistral.ts:113-116 throws out of `streamSimpleMistral`; a Rust `StreamFunction`
/// returns a stream, so the caller-visible contract is an `error` event carrying the
/// thrown message plus `recordStreamFailure` (the same body as `streamMistral`'s own
/// catch block, mistral.ts:91-101), never a panic that would abort the process.
fn api_key_error_stream(model: &Model, message: &str) -> AssistantMessageEventStream {
	let stream = create_assistant_message_event_stream();
	let mut output = create_output(model);
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

/// TS: `createMistralToolCallIdNormalizer()` - the two `Map`s live inside the closure.
///
/// The returned closure is `Fn` (like the TypeScript arrow function), so the maps
/// use interior mutability; it is shared with the streaming tool-call id derivation.
#[derive(Clone, Default)]
struct MistralToolCallIdNormalizer {
	state: Arc<std::sync::Mutex<ToolCallIdState>>,
}

#[derive(Default)]
struct ToolCallIdState {
	id_map: HashMap<String, String>,
	reverse_map: HashMap<String, String>,
}

impl MistralToolCallIdNormalizer {
	fn new() -> Self {
		Self::default()
	}

	fn normalize(&self, id: &str) -> String {
		let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		if let Some(existing) = state.id_map.get(id) {
			return existing.clone();
		}

		let mut attempt = 0usize;
		loop {
			let candidate = derive_mistral_tool_call_id(id, attempt);
			match state.reverse_map.get(&candidate) {
				Some(owner) if owner != id => {
					attempt += 1;
				}
				_ => {
					state.id_map.insert(id.to_string(), candidate.clone());
					state.reverse_map.insert(candidate.clone(), id.to_string());
					return candidate;
				}
			}
		}
	}
}

/// TS: `deriveMistralToolCallId(id, attempt)`.
fn derive_mistral_tool_call_id(id: &str, attempt: usize) -> String {
	let normalized: String = id.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
	if attempt == 0 && normalized.len() == MISTRAL_TOOL_CALL_ID_LENGTH {
		return normalized;
	}
	let seed_base = if normalized.is_empty() { id } else { &normalized };
	let seed = if attempt == 0 {
		seed_base.to_string()
	} else {
		format!("{}:{}", seed_base, attempt)
	};
	short_hash(&seed)
		.chars()
		.filter(|c| c.is_ascii_alphanumeric())
		.take(MISTRAL_TOOL_CALL_ID_LENGTH)
		.collect()
}

/// TS: `formatMistralError(error)`.
fn format_mistral_error(error: &MistralStreamError) -> String {
	match error {
		MistralStreamError::Api { message, .. } => {
			let status_code = error.status_code();
			let body_text = error
				.body()
				.map(|body| body.trim().to_string())
				.filter(|body| !body.is_empty());
			if let (Some(status_code), Some(body_text)) = (status_code, body_text) {
				return format!(
					"Mistral API error ({}): {}",
					status_code,
					truncate_error_text(&body_text, MAX_MISTRAL_ERROR_BODY_CHARS)
				);
			}
			if let Some(status_code) = status_code {
				return format!("Mistral API error ({}): {}", status_code, message);
			}
			message.clone()
		}
		other => other.error_message(),
	}
}

/// TS: `truncateErrorText(text, maxChars)`.
fn truncate_error_text(text: &str, max_chars: usize) -> String {
	if text.len() <= max_chars {
		return text.to_string();
	}
	format!(
		"{}... [truncated {} chars]",
		&text[..max_chars],
		text.len() - max_chars
	)
}

/// TS: `safeJsonStringify(value)`.
fn safe_json_stringify(value: &Value) -> String {
	serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// TS: `buildRequestOptions(model, options?)`.
#[derive(Debug, Clone, Default)]
pub struct MistralRequestOptions {
	pub retries_strategy_none: bool,
	pub headers: IndexMap<String, String>,
}

fn build_request_options(model: &Model, options: &MistralOptions) -> MistralRequestOptions {
	let mut headers: IndexMap<String, String> = IndexMap::new();
	if let Some(model_headers) = &model.headers {
		for (key, value) in model_headers {
			headers.insert(key.clone(), value.clone());
		}
	}
	if let Some(options_headers) = &options.stream.headers {
		for (key, value) in options_headers {
			headers.insert(key.clone(), value.clone());
		}
	}

	// Mistral infrastructure uses `x-affinity` for KV-cache reuse (prefix caching).
	// Respect explicit caller-provided header values.
	if let Some(session_id) = &options.stream.session_id {
		if !headers.keys().any(|key| key.eq_ignore_ascii_case("x-affinity")) {
			headers.insert("x-affinity".to_string(), session_id.clone());
		}
	}

	MistralRequestOptions {
		retries_strategy_none: true,
		headers,
	}
}

/// `mistral.chat.stream(payload, requestOptions)`.
async fn send_request(
	server_url: &str,
	api_key: &str,
	payload: &Map<String, Value>,
	model: &Model,
	options: &MistralOptions,
) -> Result<reqwest::Response, MistralStreamError> {
	let base = server_url.strip_suffix('/').unwrap_or(server_url);
	let url = format!("{}/v1/chat/completions", base);
	let request_options = build_request_options(model, options);

	let mut headers = reqwest::header::HeaderMap::new();
	headers.insert(
		reqwest::header::CONTENT_TYPE,
		reqwest::header::HeaderValue::from_static("application/json"),
	);
	headers.insert(
		reqwest::header::ACCEPT,
		reqwest::header::HeaderValue::from_static("text/event-stream"),
	);
	headers.insert(
		reqwest::header::AUTHORIZATION,
		reqwest::header::HeaderValue::from_str(&format!("Bearer {}", api_key))
			.map_err(|error| MistralStreamError::Message(error.to_string()))?,
	);
	headers.insert(
		reqwest::header::USER_AGENT,
		reqwest::header::HeaderValue::from_static(MISTRAL_USER_AGENT),
	);
	for (key, value) in request_options.headers {
		if let (Ok(name), Ok(header_value)) = (
			reqwest::header::HeaderName::from_bytes(key.as_bytes()),
			reqwest::header::HeaderValue::from_str(&value),
		) {
			headers.insert(name, header_value);
		}
	}

	let mut request = crate::providers::shared_http::shared_client(crate::providers::shared_http::ClientPolicy::Default)
		.post(&url)
		.headers(headers)
		.json(&Value::Object(payload.clone()));
	let timeout_ms = options.stream.timeout_ms.map(|value| value.max(0.0) as u64);
	// TS SDK (`@mistralai/mistralai` sdks.ts `_createRequest`) arms its default
	// request timeout only when the caller supplies no signal:
	// `if (!fetchOptions?.signal && conf.timeoutMs > 0)`. A live caller signal
	// therefore suppresses the 30 s default; an explicit timeout_ms still applies.
	let effective_timeout = match timeout_ms {
		Some(explicit) => Some(explicit),
		None => {
			if options.stream.signal.is_some() {
				None
			} else {
				Some(MISTRAL_DEFAULT_TIMEOUT_MS)
			}
		}
	};
	if let Some(effective_timeout) = effective_timeout {
		request = request.timeout(std::time::Duration::from_millis(effective_timeout));
	}

	let send = request.send();
	let response = match options.stream.signal.as_ref() {
		Some(signal) => tokio::select! {
			_ = signal.cancelled() => return Err(MistralStreamError::Message("Request was aborted".to_string())),
			result = send => result,
		},
		None => send.await,
	};
	let response = response.map_err(|error| MistralStreamError::Message(error.to_string()))?;
	check_response(response).await
}

/// TS: `matchStatusCode({ status }, ["4XX", "5XX"])` -> the SDK raises
/// `HTTPClientError`/`SDKError` with `statusCode` and `body` set.
async fn check_response(response: reqwest::Response) -> Result<reqwest::Response, MistralStreamError> {
	if response.status().is_success() {
		return Ok(response);
	}
	let status_code = response.status().as_u16() as i64;
	let status_text = response.status().canonical_reason().unwrap_or("").to_string();
	let body = response.text().await.unwrap_or_default();
	let message = format!("Status {}: Body: {}", status_code, body);
	let mut value = Map::new();
	value.insert("name".to_string(), Value::String("SDKError".to_string()));
	value.insert("message".to_string(), Value::String(message.clone()));
	value.insert("statusCode".to_string(), Value::Number(status_code.into()));
	value.insert("status".to_string(), Value::Number(status_code.into()));
	value.insert("body".to_string(), Value::String(body.clone()));
	value.insert("statusText".to_string(), Value::String(status_text));
	Err(MistralStreamError::Api {
		message,
		status_code: Some(status_code),
		body: Some(body),
		value: Value::Object(value),
	})
}

/// TS: `buildChatPayload(model, context, messages, options?)`.
pub fn build_chat_payload(
	model: &Model,
	context: &Context,
	messages: &[Message],
	options: Option<&MistralOptions>,
) -> Map<String, Value> {
	let supports_images = model.input.iter().any(|m| matches!(m, crate::types::InputModality::Image));
	let mut payload = Map::new();
	payload.insert("model".to_string(), Value::String(model.id.clone()));
	payload.insert("stream".to_string(), Value::Bool(true));
	payload.insert(
		"messages".to_string(),
		Value::Array(to_chat_messages(messages, supports_images)),
	);

	let options = options.cloned().unwrap_or_default();
	if let Some(tools) = context.tools.as_ref().filter(|tools| !tools.is_empty()) {
		payload.insert("tools".to_string(), Value::Array(to_function_tools(tools)));
	}
	if let Some(temperature) = options.stream.temperature {
		payload.insert(
			"temperature".to_string(),
			serde_json::Number::from_f64(temperature)
				.map(Value::Number)
				.unwrap_or(Value::Null),
		);
	}
	if let Some(max_tokens) = options.stream.max_tokens {
		payload.insert(
			"maxTokens".to_string(),
			serde_json::Number::from_f64(max_tokens)
				.map(Value::Number)
				.unwrap_or(Value::Null),
		);
	}
	if let Some(tool_choice) = options.tool_choice.clone() {
		if let Some(mapped) = map_tool_choice(Some(&tool_choice)) {
			payload.insert("toolChoice".to_string(), mapped);
		}
	}
	if let Some(prompt_mode) = options.prompt_mode.clone() {
		payload.insert("promptMode".to_string(), Value::String(prompt_mode));
	}
	if let Some(reasoning_effort) = options.reasoning_effort.clone() {
		payload.insert("reasoningEffort".to_string(), Value::String(reasoning_effort));
	}

	if let Some(system_prompt) = &context.system_prompt {
		let mut system_message = Map::new();
		system_message.insert("role".to_string(), Value::String("system".to_string()));
		system_message.insert(
			"content".to_string(),
			Value::String(sanitize_surrogates(system_prompt)),
		);
		if let Some(messages) = payload.get_mut("messages").and_then(Value::as_array_mut) {
			messages.insert(0, Value::Object(system_message));
		}
	}

	// TS: `buildChatPayload` returns the camelCase request object; the SDK's outbound
	// `remap$` runs later, inside `mistral.chat.stream`, so the `onPayload` hook sees the
	// camelCase shape (mistral.ts:72-77) and its edits are remapped too. See
	// `wire_chat_payload`.
	payload
}

/// TS: the SDK's outbound `remap$` - camelCase in code, snake_case on the wire.
/// `ChatCompletionStreamRequest$outboundSchema` (chatcompletionstreamrequest.js:104-118)
/// maps these ten keys; `funcs/chatStream.js:30-34` applies it to the payload it is given,
/// i.e. after `onPayload` returned.
fn wire_chat_payload(payload: &mut Map<String, Value>) {
	const REMAP: [(&str, &str); 10] = [
		("topP", "top_p"),
		("maxTokens", "max_tokens"),
		("randomSeed", "random_seed"),
		("responseFormat", "response_format"),
		("toolChoice", "tool_choice"),
		("presencePenalty", "presence_penalty"),
		("frequencyPenalty", "frequency_penalty"),
		("parallelToolCalls", "parallel_tool_calls"),
		("reasoningEffort", "reasoning_effort"),
		("promptMode", "prompt_mode"),
	];
	let mut renamed = Map::new();
	for (key, value) in payload.iter() {
		let target = REMAP
			.iter()
			.find(|(from, _)| *from == key)
			.map(|(_, to)| (*to).to_string())
			.unwrap_or_else(|| key.clone());
		renamed.insert(target, value.clone());
	}
	*payload = renamed;
}

/// TS: `toFunctionTools(tools)`.
fn to_function_tools(tools: &[crate::types::Tool]) -> Vec<Value> {
	tools
		.iter()
		.map(|tool| {
			let mut function = Map::new();
			function.insert("name".to_string(), Value::String(tool.name.clone()));
			function.insert("description".to_string(), Value::String(tool.description.clone()));
			function.insert("parameters".to_string(), strip_symbol_keys(&tool.parameters));
			function.insert("strict".to_string(), Value::Bool(false));
			let mut entry = Map::new();
			entry.insert("type".to_string(), Value::String("function".to_string()));
			entry.insert("function".to_string(), Value::Object(function));
			Value::Object(entry)
		})
		.collect()
}

/// TS: `stripSymbolKeys(value)` - a deep copy that drops symbol keys. JSON values
/// cannot carry symbol keys, so the port rebuilds the structure in the same order.
fn strip_symbol_keys(value: &Value) -> Value {
	match value {
		Value::Array(items) => Value::Array(items.iter().map(strip_symbol_keys).collect()),
		Value::Object(object) => {
			let mut result = Map::new();
			for (key, entry) in object {
				result.insert(key.clone(), strip_symbol_keys(entry));
			}
			Value::Object(result)
		}
		other => other.clone(),
	}
}

/// TS: `{ type: "image_url", imageUrl }` - the SDK's `ImageURLChunk` outbound remap
/// (`imageUrl` -> `image_url`) is applied here so the wire JSON matches.
fn image_url_chunk(url: &str) -> Value {
	let mut chunk = Map::new();
	chunk.insert("type".to_string(), Value::String("image_url".to_string()));
	chunk.insert("image_url".to_string(), Value::String(url.to_string()));
	Value::Object(chunk)
}

/// TS: `toChatMessages(messages, supportsImages)`.
pub fn to_chat_messages(messages: &[Message], supports_images: bool) -> Vec<Value> {
	let mut result: Vec<Value> = Vec::new();

	for msg in messages {
		match msg {
			Message::User(user) => match &user.content {
				crate::types::UserContent::Text(text) => {
					let mut message = Map::new();
					message.insert("role".to_string(), Value::String("user".to_string()));
					message.insert("content".to_string(), Value::String(sanitize_surrogates(text)));
					result.push(Value::Object(message));
					continue;
				}
				crate::types::UserContent::Blocks(blocks) => {
					let had_images = blocks
						.iter()
						.any(|item| matches!(item, crate::types::ImageOrTextContent::Image(_)));
					let content: Vec<Value> = blocks
						.iter()
						.filter(|item| {
							matches!(item, crate::types::ImageOrTextContent::Text(_)) || supports_images
						})
						.map(|item| match item {
							crate::types::ImageOrTextContent::Text(text) => {
								let mut chunk = Map::new();
								chunk.insert("type".to_string(), Value::String("text".to_string()));
								chunk.insert(
									"text".to_string(),
									Value::String(sanitize_surrogates(&text.text)),
								);
								Value::Object(chunk)
							}
							crate::types::ImageOrTextContent::Image(image) => {
								image_url_chunk(&format!("data:{};base64,{}", image.mime_type, image.data))
							}
						})
						.collect();
					if !content.is_empty() {
						let mut message = Map::new();
						message.insert("role".to_string(), Value::String("user".to_string()));
						message.insert("content".to_string(), Value::Array(content));
						result.push(Value::Object(message));
						continue;
					}
					if had_images && !supports_images {
						let mut message = Map::new();
						message.insert("role".to_string(), Value::String("user".to_string()));
						message.insert(
							"content".to_string(),
							Value::String("(image omitted: model does not support images)".to_string()),
						);
						result.push(Value::Object(message));
					}
					continue;
				}
			},
			Message::Assistant(assistant) => {
				let mut content_parts: Vec<Value> = Vec::new();
				let mut tool_calls: Vec<Value> = Vec::new();

				for block in assistant.content.iter() {
					match block {
						ContentBlock::Text(text) => {
							if !text.text.trim().is_empty() {
								let mut chunk = Map::new();
								chunk.insert("type".to_string(), Value::String("text".to_string()));
								chunk.insert(
									"text".to_string(),
									Value::String(sanitize_surrogates(&text.text)),
								);
								content_parts.push(Value::Object(chunk));
							}
							continue;
						}
						ContentBlock::Thinking(thinking) => {
							if !thinking.thinking.trim().is_empty() {
								let mut inner = Map::new();
								inner.insert("type".to_string(), Value::String("text".to_string()));
								inner.insert(
									"text".to_string(),
									Value::String(sanitize_surrogates(&thinking.thinking)),
								);
								let mut chunk = Map::new();
								chunk.insert("type".to_string(), Value::String("thinking".to_string()));
								chunk.insert("thinking".to_string(), Value::Array(vec![Value::Object(inner)]));
								content_parts.push(Value::Object(chunk));
							}
							continue;
						}
						ContentBlock::ToolCall(tool_call) => {
							let mut function = Map::new();
							function.insert("name".to_string(), Value::String(tool_call.name.clone()));
							function.insert(
								"arguments".to_string(),
								Value::String(
									serde_json::to_string(&Value::Object(tool_call.arguments.clone()))
										.unwrap_or_else(|_| "{}".to_string()),
								),
							);
							let mut entry = Map::new();
							entry.insert("id".to_string(), Value::String(tool_call.id.clone()));
							entry.insert("type".to_string(), Value::String("function".to_string()));
							entry.insert("function".to_string(), Value::Object(function));
							tool_calls.push(Value::Object(entry));
						}
					}
				}

				let mut message = Map::new();
				message.insert("role".to_string(), Value::String("assistant".to_string()));
				if !content_parts.is_empty() {
					message.insert("content".to_string(), Value::Array(content_parts.clone()));
				}
				if !tool_calls.is_empty() {
					message.insert("toolCalls".to_string(), Value::Array(tool_calls.clone()));
				}
				if !content_parts.is_empty() || !tool_calls.is_empty() {
					result.push(Value::Object(rename_assistant_message(message)));
				}
				continue;
			}
			Message::ToolResult(tool_result) => {
				let mut tool_content: Vec<Value> = Vec::new();
				let text_result = tool_result
					.content
					.iter()
					.filter_map(|part| match part {
						crate::types::ImageOrTextContent::Text(text) => Some(sanitize_surrogates(&text.text)),
						crate::types::ImageOrTextContent::Image(_) => None,
					})
					.collect::<Vec<_>>()
					.join("\n");
				let has_images = tool_result
					.content
					.iter()
					.any(|part| matches!(part, crate::types::ImageOrTextContent::Image(_)));
				let tool_text = build_tool_result_text(&text_result, has_images, supports_images, tool_result.is_error);
				let mut chunk = Map::new();
				chunk.insert("type".to_string(), Value::String("text".to_string()));
				chunk.insert("text".to_string(), Value::String(tool_text));
				tool_content.push(Value::Object(chunk));
				for part in tool_result.content.iter() {
					if !supports_images {
						continue;
					}
					let crate::types::ImageOrTextContent::Image(image) = part else {
						continue;
					};
					tool_content.push(image_url_chunk(&format!("data:{};base64,{}", image.mime_type, image.data)));
				}
				let mut message = Map::new();
				message.insert("role".to_string(), Value::String("tool".to_string()));
				message.insert("toolCallId".to_string(), Value::String(tool_result.tool_call_id.clone()));
				message.insert("name".to_string(), Value::String(tool_result.tool_name.clone()));
				message.insert("content".to_string(), Value::Array(tool_content));
				result.push(Value::Object(rename_tool_message(message)));
			}
		}
	}

	result
}

/// TS: the `AssistantMessage$outboundSchema` remap (`toolCalls` -> `tool_calls`).
fn rename_assistant_message(message: Map<String, Value>) -> Map<String, Value> {
	let mut renamed = Map::new();
	for (key, value) in message {
		let target = if key == "toolCalls" { "tool_calls".to_string() } else { key };
		renamed.insert(target, value);
	}
	renamed
}

/// TS: the `ToolMessage$outboundSchema` remap (`toolCallId` -> `tool_call_id`).
fn rename_tool_message(message: Map<String, Value>) -> Map<String, Value> {
	let mut renamed = Map::new();
	for (key, value) in message {
		let target = if key == "toolCallId" {
			"tool_call_id".to_string()
		} else {
			key
		};
		renamed.insert(target, value);
	}
	renamed
}

/// TS: `buildToolResultText(text, hasImages, supportsImages, isError)`.
fn build_tool_result_text(text: &str, has_images: bool, supports_images: bool, is_error: bool) -> String {
	let trimmed = text.trim();
	let error_prefix = if is_error { "[tool error] " } else { "" };

	if !trimmed.is_empty() {
		let image_suffix = if has_images && !supports_images {
			"\n[tool image omitted: model does not support images]"
		} else {
			""
		};
		return format!("{}{}{}", error_prefix, trimmed, image_suffix);
	}

	if has_images {
		if supports_images {
			return if is_error {
				"[tool error] (see attached image)".to_string()
			} else {
				"(see attached image)".to_string()
			};
		}
		return if is_error {
			"[tool error] (image omitted: model does not support images)".to_string()
		} else {
			"(image omitted: model does not support images)".to_string()
		};
	}

	if is_error {
		"[tool error] (no tool output)".to_string()
	} else {
		"(no tool output)".to_string()
	}
}

/// TS: `usesReasoningEffort(model)`.
fn uses_reasoning_effort(model: &Model) -> bool {
	model.id == "mistral-small-2603" || model.id == "mistral-small-latest" || model.id == "mistral-medium-3.5"
}

/// TS: `usesPromptModeReasoning(model)`.
fn uses_prompt_mode_reasoning(model: &Model) -> bool {
	model.reasoning && !uses_reasoning_effort(model)
}

/// TS: `mapReasoningEffort(model, level)`.
fn map_reasoning_effort(model: &Model, level: &str) -> MistralReasoningEffort {
	model
		.thinking_level_map_get(level)
		.flatten()
		.unwrap_or_else(|| "high".to_string())
}

/// TS: `mapToolChoice(choice)`.
fn map_tool_choice(choice: Option<&Value>) -> Option<Value> {
	let choice = choice?;
	match choice {
		Value::String(choice) if choice == "auto" || choice == "none" || choice == "any" || choice == "required" => {
			Some(Value::String(choice.clone()))
		}
		Value::Object(object) => {
			let name = object
				.get("function")
				.and_then(|function| function.get("name"))
				.and_then(Value::as_str)
				.unwrap_or_default();
			let mut function = Map::new();
			function.insert("name".to_string(), Value::String(name.to_string()));
			let mut mapped = Map::new();
			mapped.insert("type".to_string(), Value::String("function".to_string()));
			mapped.insert("function".to_string(), Value::Object(function));
			Some(Value::Object(mapped))
		}
		_ => None,
	}
}

/// TS: `mapChatStopReason(reason)`.
fn map_chat_stop_reason(reason: Option<&str>) -> StopReason {
	let Some(reason) = reason else {
		return "stop".to_string();
	};
	match reason {
		"stop" => "stop".to_string(),
		"length" | "model_length" => "length".to_string(),
		"tool_calls" => "toolUse".to_string(),
		"error" => "error".to_string(),
		_ => "stop".to_string(),
	}
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// The SDK's inbound `remap$` (`lib/primitives.js:23-37`): every entry keeps its value
/// and takes the mapped key, or its own key when no mapping applies.
fn remap_object_keys(object: &mut Map<String, Value>, mappings: &[(&str, &str)]) {
	let entries: Vec<(String, Value)> = std::mem::take(object).into_iter().collect();
	for (key, value) in entries {
		let target = mappings
			.iter()
			.find(|(from, _)| *from == key.as_str())
			.map(|(_, to)| (*to).to_string())
			.unwrap_or(key);
		object.insert(target, value);
	}
}

/// TS: the SDK's inbound schemas. `CompletionEvent$inboundSchema` JSON-parses each SSE
/// `data:` payload and pipes it through `CompletionChunk$inboundSchema`, whose nested
/// schemas remap the snake_case wire fields to the camelCase names
/// `consumeChatStream` reads (`mistral.ts:306-310` usage, `mistral.ts:317-321`
/// `finishReason`, `mistral.ts:388-401` `delta.toolCalls`):
///
/// * `usageinfo.js:11-23` - `prompt_tokens`/`completion_tokens`/`total_tokens`;
/// * `completionresponsestreamchoice.js:22-35` - `finish_reason`;
/// * `deltamessage.js:24-40` - `tool_calls`, `tool_call_id`.
fn remap_completion_chunk(chunk: Value) -> Value {
	let Value::Object(mut chunk) = chunk else {
		return chunk;
	};

	if let Some(Value::Object(usage)) = chunk.get_mut("usage") {
		remap_object_keys(
			usage,
			&[
				("prompt_tokens", "promptTokens"),
				("completion_tokens", "completionTokens"),
				("total_tokens", "totalTokens"),
			],
		);
	}

	if let Some(Value::Array(choices)) = chunk.get_mut("choices") {
		for choice in choices.iter_mut() {
			let Value::Object(choice) = choice else {
				continue;
			};
			remap_object_keys(choice, &[("finish_reason", "finishReason")]);
			if let Some(Value::Object(delta)) = choice.get_mut("delta") {
				remap_object_keys(
					delta,
					&[("tool_calls", "toolCalls"), ("tool_call_id", "toolCallId")],
				);
			}
		}
	}

	Value::Object(chunk)
}

/// The Mistral SSE transport (`EventStream` in the SDK).
struct MistralChunkStream {
	chunks: std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
	buffer: String,
	pending: Vec<Value>,
	/// Bytes that form an incomplete UTF-8 sequence at the end of a network chunk.
	/// `parseMessage` decodes each message with `new TextDecoder("utf-8")`, which is
	/// stateful across the chunks that make up one message, so these bytes must wait
	/// for the next chunk instead of becoming U+FFFD.
	pending_bytes: Vec<u8>,
	done: bool,
	finished: bool,
	signal: Option<tokio_util::sync::CancellationToken>,
}

impl MistralChunkStream {
	fn new(response: reqwest::Response, signal: Option<tokio_util::sync::CancellationToken>) -> Self {
		Self {
			chunks: Box::pin(response.bytes_stream()),
			buffer: String::new(),
			pending: Vec::new(),
			pending_bytes: Vec::new(),
			done: false,
			finished: false,
			signal,
		}
	}

	/// TS: `for await (const event of mistralStream)` - each item is `event.data`.
	async fn next(&mut self) -> Result<Option<Value>, MistralStreamError> {
		loop {
			if !self.pending.is_empty() {
				return Ok(Some(self.pending.remove(0)));
			}
			if self.done {
				return Ok(None);
			}
			if self.finished {
				return Ok(None);
			}
			if let Some(signal) = &self.signal {
				if signal.is_cancelled() {
					return Err(MistralStreamError::Message("Request was aborted".to_string()));
				}
			}
			match self.chunks.next().await {
				None => {
					self.finished = true;
					// `findBoundary` failed and `upstream.read()` reported `done`, so the SDK calls
					// `downstream.close()`: a trailing partial message (and the decoder's pending
					// partial bytes) is dropped, never parsed.
					self.pending_bytes.clear();
				}
				Some(Err(error)) => return Err(MistralStreamError::Message(error.to_string())),
				Some(Ok(bytes)) => {
					// The SDK accumulates raw bytes (`concatBuffer`) and decodes each message with
					// `parseMessage`'s `new TextDecoder().decode(chunk)`, so a multi-byte character
					// split across two network chunks stays intact instead of becoming U+FFFD
					// (`String::from_utf8_lossy` per chunk would corrupt every straddling character).
					self.pending_bytes.extend_from_slice(&bytes);
					let chunk_string = self.decode_pending_bytes();
					self.buffer.push_str(&chunk_string);
					self.drain_events()?;
				}
			}
		}
	}

	/// `new TextDecoder().decode(chunk)` - a stateful streaming UTF-8 decode.
	///
	/// Bytes that form an incomplete multi-byte sequence stay in `pending_bytes` until the
	/// next chunk arrives; a genuinely invalid sequence becomes U+FFFD like the WHATWG
	/// decoder. Same shape as `google.rs::decode_pending_bytes`.
	fn decode_pending_bytes(&mut self) -> String {
		let mut text = String::new();
		loop {
			match std::str::from_utf8(&self.pending_bytes) {
				Ok(valid) => {
					text.push_str(valid);
					self.pending_bytes.clear();
					return text;
				}
				Err(error) => {
					let valid_up_to = error.valid_up_to();
					text.push_str(&String::from_utf8_lossy(&self.pending_bytes[..valid_up_to]));
					self.pending_bytes.drain(..valid_up_to);
					match error.error_len() {
						// Incomplete trailing sequence: wait for more bytes.
						None => return text,
						Some(error_length) => {
							self.pending_bytes.drain(..error_length);
							text.push('\u{FFFD}');
						}
					}
				}
			}
		}
	}

	/// TS: `findBoundary` + `parseMessage` + the `[DONE]` short circuit.
	fn drain_events(&mut self) -> Result<(), MistralStreamError> {
		// The earliest of the SDK's eight boundaries, each found from where the
		// previous one ended; the consumed prefix is dropped once per call.
		let mut cursor = 0usize;
		while let Some((index, length)) = next_boundary(&self.buffer, cursor) {
			let message = &self.buffer[cursor..index];
			cursor = index + length;
			let Some(data) = parse_sse_message(message) else {
				continue;
			};
			if data == "[DONE]" {
				self.done = true;
				return Ok(());
			}
			match serde_json::from_str::<Value>(&data) {
				// `CompletionEvent$inboundSchema`: `JSON.parse(data)` then the chunk schema's
				// inbound remap.
				Ok(value) => self.pending.push(remap_completion_chunk(value)),
				Err(error) => {
					return Err(MistralStreamError::Message(format!(
						"malformed json: {}",
						error
					)))
				}
			}
		}
		if cursor > 0 {
			self.buffer.drain(..cursor);
		}
		Ok(())
	}
}

/// The earliest of `["\r\n\r\n", "\r\n\r", "\r\n\n", "\r\r\n", "\n\r\n",
/// "\r\r", "\n\r", "\n\n"]` at or after `from`, as (index, length). At a
/// given index the longest candidate wins, matching the array order that the
/// per-boundary min-index scan uses for its only possible ties
/// (`\r\n\r\n`/`\r\n\r`, `\r\r\n`/`\r\r`, `\n\r\n`/`\n\r`).
fn next_boundary(buffer: &str, from: usize) -> Option<(usize, usize)> {
	let bytes = buffer.as_bytes();
	let mut index = from;
	while index + 1 < bytes.len() {
		let length = match (bytes[index], bytes[index + 1]) {
			(b'\r', b'\n') => {
				if index + 3 < bytes.len() && bytes[index + 2] == b'\r' && bytes[index + 3] == b'\n' {
					4
				} else if index + 2 < bytes.len() && bytes[index + 2] == b'\r' {
					3
				} else if index + 2 < bytes.len() && bytes[index + 2] == b'\n' {
					3
				} else {
					0
				}
			}
			(b'\r', b'\r') => {
				if index + 2 < bytes.len() && bytes[index + 2] == b'\n' {
					3
				} else {
					2
				}
			}
			(b'\n', b'\r') => {
				if index + 2 < bytes.len() && bytes[index + 2] == b'\n' {
					3
				} else {
					2
				}
			}
			(b'\n', b'\n') => 2,
			_ => 0,
		};
		if length > 0 {
			return Some((index, length));
		}
		index += 1;
	}
	None
}

/// TS: `parseMessage(chunk, parse, state, dataRequired)` - returns the joined
/// `data` field, or `None` for a message with no data lines.
fn parse_sse_message(message: &str) -> Option<String> {
	let mut data_lines: Vec<String> = Vec::new();
	let mut ignore = true;
	for line in message.split(|c| c == '\r' || c == '\n') {
		if line.is_empty() || line.starts_with(':') {
			continue;
		}
		ignore = false;
		let index = line.find(':');
		let (field, value) = match index {
			Some(index) if index > 0 => {
				let value = &line[index + 1..];
				let value = value.strip_prefix(' ').unwrap_or(value);
				(&line[..index], value)
			}
			_ => (line, ""),
		};
		if field == "data" {
			data_lines.push(value.to_string());
		}
	}
	if ignore {
		return None;
	}
	if data_lines.is_empty() {
		return None;
	}
	Some(data_lines.join("\n"))
}

/// TS: `consumeChatStream(model, output, stream, mistralStream)`.
async fn consume_chat_stream(
	model: &Model,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
	mistral_stream: &mut MistralChunkStream,
) -> Result<(), MistralStreamError> {
	let mut current_block: Option<CurrentBlock> = None;
	let mut tool_blocks_by_key: IndexMap<String, usize> = IndexMap::new();
	// `partialArgs` is a streaming scratch buffer the TypeScript stores on the block and
	// deletes before the block is persisted; the Rust `ToolCall` has no such field, so the
	// buffer lives beside it, keyed by the block index.
	let mut partial_args_by_index: HashMap<usize, String> = HashMap::new();

	while let Some(chunk) = mistral_stream.next().await? {
		// Mistral's streamed CompletionChunk carries an id field. Keep the first non-empty one,
		// mirroring how OpenAI-style streaming exposes a stable response identifier per stream.
		if output.response_id.is_none() {
			if let Some(id) = chunk.get("id").and_then(Value::as_str) {
				if !id.is_empty() {
					output.response_id = Some(id.to_string());
				}
			}
		}

		if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
			output.usage.input = number_field(usage, "promptTokens");
			output.usage.output = number_field(usage, "completionTokens");
			output.usage.cache_read = 0.0;
			output.usage.cache_write = 0.0;
			let total_tokens = number_field(usage, "totalTokens");
			output.usage.total_tokens = if total_tokens != 0.0 {
				total_tokens
			} else {
				output.usage.input + output.usage.output
			};
			calculate_cost(model, &mut output.usage, None);
		}

		let choice = chunk
			.get("choices")
			.and_then(Value::as_array)
			.and_then(|choices| choices.first());
		let Some(choice) = choice else {
			continue;
		};

		if let Some(finish_reason) = choice.get("finishReason").and_then(Value::as_str) {
			output.stop_reason = map_chat_stop_reason(Some(finish_reason));
			if output.stop_reason == "error" {
				output.stop_reason_raw = Some(finish_reason.to_string());
			}
		}

		let delta = choice.get("delta").cloned().unwrap_or(Value::Object(Map::new()));
		if let Some(content) = delta.get("content").filter(|content| !content.is_null()) {
			let content_items: Vec<Value> = match content {
				Value::String(text) => vec![Value::String(text.clone())],
				Value::Array(items) => items.clone(),
				_ => Vec::new(),
			};
			for item in content_items {
				if let Value::String(text) = &item {
					let text_delta = sanitize_surrogates(text);
					if !matches!(current_block, Some(CurrentBlock::Text(_))) {
						if let Some(block) = current_block.take() {
							finish_current_block(&block, output, stream);
						}
						output.content.push(ContentBlock::Text(TextContent::new(String::new())));
						stream.push(AssistantMessageEvent::TextStart {
							content_index: output.content.len() - 1,
							partial: output.clone(),
						});
						current_block = Some(CurrentBlock::Text(TextContent::new(String::new())));
					}
					if let Some(CurrentBlock::Text(_)) = current_block.as_ref() {
						if let Some(ContentBlock::Text(target)) = output.content.last_mut() {
							target.text.push_str(&text_delta);
						}
					}
					stream.push(AssistantMessageEvent::TextDelta {
						content_index: output.content.len() - 1,
						delta: text_delta,
						partial: output.clone(),
					});
					continue;
				}

				let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
				if item_type == "thinking" {
					let delta_text = item
						.get("thinking")
						.and_then(Value::as_array)
						.map(|parts| {
							parts
								.iter()
								.map(|part| part.get("text").and_then(Value::as_str).unwrap_or_default())
								.collect::<Vec<_>>()
								.join("")
						})
						.unwrap_or_default();
					let thinking_delta = sanitize_surrogates(&delta_text);
					if thinking_delta.is_empty() {
						continue;
					}
					if !matches!(current_block, Some(CurrentBlock::Thinking(_))) {
						if let Some(block) = current_block.take() {
							finish_current_block(&block, output, stream);
						}
						output
							.content
							.push(ContentBlock::Thinking(ThinkingContent::new(String::new())));
						stream.push(AssistantMessageEvent::ThinkingStart {
							content_index: output.content.len() - 1,
							partial: output.clone(),
						});
						current_block = Some(CurrentBlock::Thinking(ThinkingContent::new(String::new())));
					}
					if let Some(CurrentBlock::Thinking(_)) = current_block.as_ref() {
						if let Some(ContentBlock::Thinking(target)) = output.content.last_mut() {
							target.thinking.push_str(&thinking_delta);
						}
					}
					stream.push(AssistantMessageEvent::ThinkingDelta {
						content_index: output.content.len() - 1,
						delta: thinking_delta,
						partial: output.clone(),
					});
					continue;
				}

				if item_type == "text" {
					let text_delta = sanitize_surrogates(item.get("text").and_then(Value::as_str).unwrap_or_default());
					if !matches!(current_block, Some(CurrentBlock::Text(_))) {
						if let Some(block) = current_block.take() {
							finish_current_block(&block, output, stream);
						}
						output.content.push(ContentBlock::Text(TextContent::new(String::new())));
						stream.push(AssistantMessageEvent::TextStart {
							content_index: output.content.len() - 1,
							partial: output.clone(),
						});
						current_block = Some(CurrentBlock::Text(TextContent::new(String::new())));
					}
					if let Some(CurrentBlock::Text(_)) = current_block.as_ref() {
						if let Some(ContentBlock::Text(target)) = output.content.last_mut() {
							target.text.push_str(&text_delta);
						}
					}
					stream.push(AssistantMessageEvent::TextDelta {
						content_index: output.content.len() - 1,
						delta: text_delta,
						partial: output.clone(),
					});
				}
			}
		}

		let tool_calls = delta
			.get("toolCalls")
			.and_then(Value::as_array)
			.cloned()
			.unwrap_or_default();
		for tool_call in tool_calls {
			if let Some(block) = current_block.take() {
				finish_current_block(&block, output, stream);
			}
			let raw_id = tool_call.get("id").and_then(Value::as_str).unwrap_or_default();
			let call_id = if !raw_id.is_empty() && raw_id != "null" {
				raw_id.to_string()
			} else {
				let index = tool_call.get("index").and_then(Value::as_i64).unwrap_or(0);
				derive_mistral_tool_call_id(&format!("toolcall:{}", index), 0)
			};
			let index = tool_call.get("index").and_then(Value::as_i64).unwrap_or(0);
			let key = format!("{}:{}", call_id, index);
			let mut block_index = tool_blocks_by_key.get(&key).copied();

			if let Some(existing_index) = block_index {
				if !matches!(
					output.content.get(existing_index),
					Some(ContentBlock::ToolCall(_))
				) {
					block_index = None;
				}
			}

			if block_index.is_none() {
				let name = tool_call
					.get("function")
					.and_then(|function| function.get("name"))
					.and_then(Value::as_str)
					.unwrap_or_default()
					.to_string();
				let tool_block = ToolCall::new(call_id.clone(), name, Map::new());
				output.content.push(ContentBlock::ToolCall(tool_block));
				tool_blocks_by_key.insert(key.clone(), output.content.len() - 1);
				partial_args_by_index.insert(output.content.len() - 1, String::new());
				stream.push(AssistantMessageEvent::ToolCallStart {
					content_index: output.content.len() - 1,
					partial: output.clone(),
				});
			}

			let function = tool_call.get("function").cloned().unwrap_or(Value::Object(Map::new()));
			let args_delta = match function.get("arguments") {
				Some(Value::String(arguments)) => arguments.clone(),
				Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
				None => serde_json::to_string(&Value::Object(Map::new())).unwrap_or_else(|_| "{}".to_string()),
			};
			let index = tool_blocks_by_key.get(&key).copied().unwrap_or(0);
			let partial_args = format!(
				"{}{}",
				partial_args_by_index.get(&index).cloned().unwrap_or_default(),
				args_delta
			);
			partial_args_by_index.insert(index, partial_args.clone());
			if let Some(ContentBlock::ToolCall(block)) = output.content.get_mut(index) {
				block.arguments = match parse_streaming_json(Some(&partial_args)) {
					Value::Object(map) => map,
					_ => Map::new(),
				};
			}
			stream.push(AssistantMessageEvent::ToolCallDelta {
				content_index: index,
				delta: args_delta,
				partial: output.clone(),
			});
		}
	}

	if let Some(block) = current_block.take() {
		finish_current_block(&block, output, stream);
	}
	for index in tool_blocks_by_key.values().copied().collect::<Vec<usize>>() {
		let Some(ContentBlock::ToolCall(tool_block)) = output.content.get_mut(index) else {
			continue;
		};
		let partial_args = partial_args_by_index.get(&index).cloned().unwrap_or_default();
		tool_block.arguments = match parse_streaming_json(Some(&partial_args)) {
			Value::Object(map) => map,
			_ => Map::new(),
		};
		// Finalize in-place; the scratch buffer is dropped so replay only
		// carries parsed arguments.
		let tool_call = tool_block.clone();
		stream.push(AssistantMessageEvent::ToolCallEnd {
			content_index: index,
			tool_call,
			partial: output.clone(),
		});
	}

	Ok(())
}

/// The block the streaming loop is currently filling.
#[derive(Debug, Clone, PartialEq)]
enum CurrentBlock {
	Text(TextContent),
	Thinking(ThinkingContent),
}

/// TS: `finishCurrentBlock(block?)`.
fn finish_current_block(block: &CurrentBlock, output: &AssistantMessage, stream: &AssistantMessageEventStream) {
	// The accumulated text lives in `output.content` (deltas append in place);
	// `block` only carries the kind.
	match (block, output.content.last()) {
		(CurrentBlock::Text(_), Some(ContentBlock::Text(text))) => {
			stream.push(AssistantMessageEvent::TextEnd {
					content_index: output.content.len() - 1,
					content: text.text.clone(),
					partial: output.clone(),
				});
		}
		(CurrentBlock::Thinking(_), Some(ContentBlock::Thinking(thinking))) => {
			stream.push(AssistantMessageEvent::ThinkingEnd {
					content_index: output.content.len() - 1,
					content: thinking.thinking.clone(),
						partial: output.clone(),
				});
		}
		_ => {}
	}
}
/// TS: `chunk.usage?.promptTokens || 0`.
fn number_field(value: &Value, key: &str) -> f64 {
	value.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

/// Keeps `safeJsonStringify` referenced exactly like the TypeScript module.
pub fn _safe_json_stringify(value: &Value) -> String {
	safe_json_stringify(value)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{
		ImageContent, ImageOrTextContent, InputModality, ModelCost, Tool, ToolResultMessage, UserContent, UserMessage,
	};
	use serde_json::json;

	fn model(id: &str) -> Model {
		let mut model = Model::new(id, id, "mistral-conversations", "mistral", "https://api.mistral.ai");
		model.input = vec![InputModality::Text];
		model.cost = ModelCost::zero();
		model
	}

	fn base_options() -> StreamOptions {
		StreamOptions::default()
	}

	#[test]
	fn options_round_trip_through_serde() {
		let mut options = MistralOptions::from_base(&base_options());
		options.prompt_mode = Some("reasoning".to_string());
		options.reasoning_effort = Some("high".to_string());
		options.tool_choice = Some(json!("required"));
		let value = serde_json::to_value(&options).unwrap();
		assert_eq!(value["promptMode"], json!("reasoning"));
		assert_eq!(value["reasoningEffort"], json!("high"));
		assert_eq!(value["toolChoice"], json!("required"));
		let back: MistralOptions = serde_json::from_value(value).unwrap();
		assert_eq!(back.prompt_mode.as_deref(), Some("reasoning"));
	}

	#[test]
	fn tool_call_id_normalizer_keeps_nine_char_alphanumeric_ids() {
		let normalizer = MistralToolCallIdNormalizer::new();
		assert_eq!(normalizer.normalize("abc123XYZ"), "abc123XYZ");
		// Too short / too long ids are hashed.
		let hashed = normalizer.normalize("short");
		assert_eq!(hashed.len(), MISTRAL_TOOL_CALL_ID_LENGTH);
		assert!(hashed.chars().all(|c| c.is_ascii_alphanumeric()));
		// Repeated calls return the memoised value.
		assert_eq!(normalizer.normalize("short"), hashed);
	}

	#[test]
	fn tool_call_id_normalizer_avoids_collisions_with_an_attempt_suffix() {
		// Both ids normalise to the same 9-char seed; the second must differ.
		let normalizer = MistralToolCallIdNormalizer::new();
		let first = normalizer.normalize("call-1");
		let second = normalizer.normalize("call1");
		assert_ne!(first, second);
		assert_eq!(first.len(), MISTRAL_TOOL_CALL_ID_LENGTH);
		assert_eq!(second.len(), MISTRAL_TOOL_CALL_ID_LENGTH);
	}

	#[test]
	fn derive_tool_call_id_strips_non_alphanumerics() {
		let derived = derive_mistral_tool_call_id("call:1|2", 0);
		assert_eq!(derived.len(), MISTRAL_TOOL_CALL_ID_LENGTH);
		assert!(derived.chars().all(|c| c.is_ascii_alphanumeric()));
		assert_eq!(derive_mistral_tool_call_id("abc123XYZ", 0), "abc123XYZ");
		assert_ne!(derive_mistral_tool_call_id("abcdefghij", 0), derive_mistral_tool_call_id("abcdefghij", 1));
	}

	#[test]
	fn format_mistral_error_matches_typescript_branches() {
		let with_body = MistralStreamError::Api {
			message: "Status 429: Body: slow down".to_string(),
			status_code: Some(429),
			body: Some("  slow down  ".to_string()),
			value: Value::Null,
		};
		assert_eq!(format_mistral_error(&with_body), "Mistral API error (429): slow down");

		let without_body = MistralStreamError::Api {
			message: "Status 500: Body: ".to_string(),
			status_code: Some(500),
			body: Some("   ".to_string()),
			value: Value::Null,
		};
		assert_eq!(
			format_mistral_error(&without_body),
			"Mistral API error (500): Status 500: Body: "
		);

		let plain = MistralStreamError::Message("boom".to_string());
		assert_eq!(format_mistral_error(&plain), "boom");
	}

	#[test]
	fn truncate_error_text_appends_the_truncation_note() {
		assert_eq!(truncate_error_text("abc", 10), "abc");
		assert_eq!(truncate_error_text("abcdef", 3), "abc... [truncated 3 chars]");
	}

	#[test]
	fn map_chat_stop_reason_follows_the_typescript_switch() {
		assert_eq!(map_chat_stop_reason(None), "stop");
		assert_eq!(map_chat_stop_reason(Some("stop")), "stop");
		assert_eq!(map_chat_stop_reason(Some("length")), "length");
		assert_eq!(map_chat_stop_reason(Some("model_length")), "length");
		assert_eq!(map_chat_stop_reason(Some("tool_calls")), "toolUse");
		assert_eq!(map_chat_stop_reason(Some("error")), "error");
		assert_eq!(map_chat_stop_reason(Some("wat")), "stop");
	}

	#[test]
	fn reasoning_routing_helpers_match_typescript() {
		assert!(uses_reasoning_effort(&model("mistral-small-2603")));
		assert!(uses_reasoning_effort(&model("mistral-small-latest")));
		assert!(uses_reasoning_effort(&model("mistral-medium-3.5")));
		assert!(!uses_reasoning_effort(&model("mistral-large-latest")));

		let mut reasoning_model = model("mistral-large-latest");
		reasoning_model.reasoning = true;
		assert!(uses_prompt_mode_reasoning(&reasoning_model));
		let mut effort_model = model("mistral-small-latest");
		effort_model.reasoning = true;
		assert!(!uses_prompt_mode_reasoning(&effort_model));
	}

	#[test]
	fn map_reasoning_effort_prefers_the_thinking_level_map() {
		let mut model = model("mistral-small-latest");
		model.thinking_level_map = Some(
			[("high".to_string(), Some("none".to_string()))]
				.into_iter()
				.collect(),
		);
		assert_eq!(map_reasoning_effort(&model, "high"), "none");
		assert_eq!(map_reasoning_effort(&model, "low"), "high");
	}

	#[test]
	fn map_tool_choice_passes_strings_and_normalises_objects() {
		assert_eq!(map_tool_choice(None), None);
		assert_eq!(map_tool_choice(Some(&json!("auto"))), Some(json!("auto")));
		assert_eq!(map_tool_choice(Some(&json!("required"))), Some(json!("required")));
		assert_eq!(map_tool_choice(Some(&json!("bogus"))), None);
		assert_eq!(
			map_tool_choice(Some(&json!({"type": "function", "function": {"name": "read"}}))),
			Some(json!({"type": "function", "function": {"name": "read"}}))
		);
	}

	#[test]
	fn build_chat_payload_uses_snake_case_wire_keys() {
		let model = model("mistral-large-latest");
		let context = Context::new(
			Some("be nice".to_string()),
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			Some(vec![Tool {
				name: "read".to_string(),
				description: "Read".to_string(),
				parameters: json!({"type": "object"}),
			}]),
		);
		let mut options = MistralOptions::from_base(&base_options());
		options.stream.temperature = Some(0.3);
		options.stream.max_tokens = Some(128.0);
		options.tool_choice = Some(json!("any"));
		options.prompt_mode = Some("reasoning".to_string());
		options.reasoning_effort = Some("high".to_string());
		let mut payload = build_chat_payload(&model, &context, &context.messages, Some(&options));
		wire_chat_payload(&mut payload);
		assert_eq!(
			Value::Object(payload),
			json!({
				"model": "mistral-large-latest",
				"stream": true,
				"messages": [
					{"role": "system", "content": "be nice"},
					{"role": "user", "content": "hi"}
				],
				"tools": [{
					"type": "function",
					"function": {
						"name": "read",
						"description": "Read",
						"parameters": {"type": "object"},
						"strict": false
					}
				}],
				"temperature": 0.3,
				"max_tokens": 128.0,
				"tool_choice": "any",
				"prompt_mode": "reasoning",
				"reasoning_effort": "high"
			})
		);
	}

	/// A one-request HTTP fixture server: it returns the parsed JSON request body it
	/// received and answers with `sse_body` as a `text/event-stream`.
	async fn mistral_fixture_server(
		sse_body: &str,
	) -> (String, std::sync::Arc<std::sync::Mutex<Value>>, tokio::task::JoinHandle<()>) {
		use std::sync::{Arc, Mutex};
		use tokio::io::{AsyncReadExt, AsyncWriteExt};

		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
			.await
			.unwrap();
		let address = listener.local_addr().unwrap();
		let received: Arc<Mutex<Value>> = Arc::new(Mutex::new(Value::Null));
		let server_received = received.clone();
		let sse_body = sse_body.to_string();
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut request = Vec::new();
			let length = loop {
				let mut buffer = [0u8; 4096];
				let count = socket.read(&mut buffer).await.unwrap();
				if count == 0 {
					break 0;
				}
				request.extend_from_slice(&buffer[..count]);
				if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
					let headers = std::str::from_utf8(&request[..end]).unwrap();
					let length = headers
						.lines()
						.filter_map(|line| line.split_once(':'))
						.find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
						.map(|(_, value)| value.trim().parse::<usize>().unwrap())
						.unwrap();
					if request.len() >= end + 4 + length {
						break length;
					}
				}
			};
			let body_start = request
				.windows(4)
				.position(|window| window == b"\r\n\r\n")
				.unwrap()
				+ 4;
			*server_received.lock().unwrap() =
				serde_json::from_slice(&request[body_start..body_start + length]).unwrap();
			let response = format!(
				"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse_body}",
				sse_body.len()
			);
			socket.write_all(response.as_bytes()).await.unwrap();
		});
		(format!("http://{address}"), received, server)
	}

	/// GM-01 end to end (the worst finding: it corrupts cost on EVERY Mistral call):
	/// a real snake_case wire frame must set `usage` + `cost`, the `finish_reason:
	/// "tool_calls"` must become `stopReason: "toolUse"`, and a streamed `tool_calls`
	/// entry must become a `toolCall` block - all of which the dropped inbound remap
	/// silently lost. Breaks if the reader stops calling `remap_completion_chunk`.
	#[tokio::test]
	async fn stream_mistral_applies_wire_usage_cost_stop_reason_and_tool_calls() {
		let frames = concat!(
			"data: {\"id\":\"cmpl-1\",\"model\":\"mistral-large-latest\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
			"{\"id\":\"call-1\",\"index\":0,\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]}}]}\n\n",
			"data: {\"id\":\"cmpl-1\",\"model\":\"mistral-large-latest\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
			"{\"id\":\"call-1\",\"index\":0,\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"a\\\"}\"}}]}}]}\n\n",
			"data: {\"id\":\"cmpl-1\",\"model\":\"mistral-large-latest\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],",
			"\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":500,\"total_tokens\":1500}}\n\n",
			"data: [DONE]\n\n"
		);
		let (base_url, _received, server) = mistral_fixture_server(frames).await;

		let mut model = model("mistral-large-latest");
		model.base_url = base_url;
		// A non-zero price makes a missing `calculate_cost` call visible as cost.total == 0.
		model.cost.input = 3.0;
		model.cost.output = 15.0;
		let context = Context::new(
			None,
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			None,
		);
		let mut options = MistralOptions::from_base(&base_options());
		options.stream.api_key = Some("test-key".to_string());
		options.stream.timeout_ms = Some(5000.0);
		let stream = stream_mistral(&model, &context, Some(options));
		let message = tokio::time::timeout(std::time::Duration::from_secs(5), stream.result())
			.await
			.expect("the fixture stream must complete");
		server.await.unwrap();

		// mistral.ts:305-312.
		assert_eq!(message.usage.input, 1000.0);
		assert_eq!(message.usage.output, 500.0);
		assert_eq!(message.usage.total_tokens, 1500.0);
		assert_eq!(message.response_id.as_deref(), Some("cmpl-1"));
		assert_eq!(message.usage.cost.total, 0.0105);
		// mistral.ts:317-321.
		assert_eq!(message.stop_reason, "toolUse");
		// mistral.ts:388-434.
		let block = message
			.content
			.iter()
			.find_map(|block| match block {
				ContentBlock::ToolCall(tool_call) => Some(tool_call.clone()),
				_ => None,
			})
			.expect("the streamed tool call must produce a toolCall block");
		assert_eq!(block.id, "call-1");
		assert_eq!(block.name, "read");
		assert_eq!(block.arguments["path"], json!("a"));

	}

	/// GM-07 end to end: the hook's payload is remapped by the SDK step
	/// (`funcs/chatStream.js:30-34`), so the bytes on the wire are snake_case even though
	/// the hook (and `buildChatPayload`) use camelCase. Breaks if `wire_chat_payload` is
	/// dropped from `run_stream_mistral`.
	#[tokio::test]
	async fn stream_mistral_sends_the_hook_payload_after_the_wire_remap() {
		use std::sync::Arc;

		let (base_url, received, server) = mistral_fixture_server("data: [DONE]\n\n").await;

		let mut model = model("mistral-large-latest");
		model.base_url = base_url;
		let context = Context::new(
			None,
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			None,
		);
		let mut options = MistralOptions::from_base(&base_options());
		options.stream.api_key = Some("test-key".to_string());
		options.stream.timeout_ms = Some(5000.0);
		// The hook edits the camelCase payload exactly like a TypeScript hook.
		options.stream.on_payload = Some(Arc::new(|payload: Value, _model: &Model| {
			Box::pin(async move {
				let mut payload = payload;
				payload["promptMode"] = json!("reasoning");
				payload["presencePenalty"] = json!(0.25);
				Some(payload)
			})
		}));
		let stream = stream_mistral(&model, &context, Some(options));
		tokio::time::timeout(std::time::Duration::from_secs(5), stream.result())
			.await
			.expect("the fixture stream must complete");
		server.await.unwrap();

		let body = received.lock().unwrap().clone();
		assert_eq!(body["prompt_mode"], json!("reasoning"));
		assert_eq!(body["presence_penalty"], json!(0.25));
		assert!(body.get("promptMode").is_none());
		assert!(body.get("presencePenalty").is_none());
	}

	/// GM-07: mistral.ts:72-77 hands the hook the camelCase `ChatCompletionStreamRequest`
	/// (`buildChatPayload` has no remap), and the SDK remaps the hook's result inside
	/// `chat.stream` (chatcompletionstreamrequest.js:104-118), so hook-added keys must be
	/// remapped too.
	#[test]
	fn wire_chat_payload_remaps_the_hook_result_after_build_chat_payload() {
		let model = model("mistral-large-latest");
		let context = Context::new(
			None,
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			None,
		);
		let mut options = MistralOptions::from_base(&base_options());
		options.stream.max_tokens = Some(128.0);
		options.prompt_mode = Some("reasoning".to_string());

		// The payload handed to `onPayload` is still camelCase.
		let mut payload = build_chat_payload(&model, &context, &context.messages, Some(&options));
		assert_eq!(payload["maxTokens"], json!(128.0));
		assert_eq!(payload["promptMode"], json!("reasoning"));
		assert!(payload.get("max_tokens").is_none());

		// The hook edits/adds camelCase keys, exactly like a TypeScript hook.
		payload.insert("presencePenalty".to_string(), json!(0.5));
		payload.insert("toolChoice".to_string(), json!("auto"));
		wire_chat_payload(&mut payload);
		assert_eq!(payload["max_tokens"], json!(128.0));
		assert_eq!(payload["prompt_mode"], json!("reasoning"));
		assert_eq!(payload["presence_penalty"], json!(0.5));
		assert_eq!(payload["tool_choice"], json!("auto"));
		// Untouched keys and unknown hook keys keep their own name (`remap$` default).
		assert_eq!(payload["model"], json!("mistral-large-latest"));
		assert!(payload.get("presencePenalty").is_none());
	}

	#[test]
	fn build_chat_payload_omits_optional_fields() {
		let model = model("mistral-large-latest");
		let context = Context::new(
			None,
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			None,
		);
		let payload = build_chat_payload(&model, &context, &context.messages, None);
		assert_eq!(
			Value::Object(payload),
			json!({
				"model": "mistral-large-latest",
				"stream": true,
				"messages": [{"role": "user", "content": "hi"}]
			})
		);
	}

	#[test]
	fn to_chat_messages_omits_images_for_text_only_models() {
		let messages = vec![Message::user(UserMessage::new(
			UserContent::Blocks(vec![ImageOrTextContent::Image(ImageContent::new("AAAA", "image/png"))]),
			0,
		))];
		assert_eq!(
			to_chat_messages(&messages, false),
			vec![json!({"role": "user", "content": "(image omitted: model does not support images)"})]
		);

		let messages = vec![Message::user(UserMessage::new(
			UserContent::Blocks(vec![
				ImageOrTextContent::Text(TextContent::new("look")),
				ImageOrTextContent::Image(ImageContent::new("AAAA", "image/png")),
			]),
			0,
		))];
		assert_eq!(
			to_chat_messages(&messages, true),
			vec![json!({
				"role": "user",
				"content": [
					{"type": "text", "text": "look"},
					{"type": "image_url", "image_url": "data:image/png;base64,AAAA"}
				]
			})]
		);
	}

	#[test]
	fn to_chat_messages_maps_assistant_blocks_and_tool_results() {
		let mut assistant = AssistantMessage::default();
		assistant.content = vec![
			ContentBlock::Text(TextContent::new("answer")),
			ContentBlock::Thinking(ThinkingContent::new("hmm")),
			ContentBlock::ToolCall(ToolCall::new("call-1", "read", json!({"path": "a"}).as_object().unwrap().clone())),
		];
		let messages = vec![
			Message::assistant(assistant),
			Message::tool_result(ToolResultMessage::new(
				"call-1",
				"read",
				vec![ImageOrTextContent::Text(TextContent::new("ok"))],
				false,
				0,
			)),
			Message::tool_result(ToolResultMessage::new(
				"call-2",
				"read",
				vec![ImageOrTextContent::Text(TextContent::new("bad"))],
				true,
				0,
			)),
		];
		assert_eq!(
			to_chat_messages(&messages, false),
			vec![
				json!({
					"role": "assistant",
					"content": [
						{"type": "text", "text": "answer"},
						{"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]}
					],
					"tool_calls": [{
						"id": "call-1",
						"type": "function",
						"function": {"name": "read", "arguments": "{\"path\":\"a\"}"}
					}]
				}),
				json!({"role": "tool", "tool_call_id": "call-1", "name": "read", "content": [{"type": "text", "text": "ok"}]}),
				json!({"role": "tool", "tool_call_id": "call-2", "name": "read", "content": [{"type": "text", "text": "[tool error] bad"}]})
			]
		);
	}

	#[test]
	fn build_tool_result_text_matches_typescript_branches() {
		assert_eq!(build_tool_result_text("out", false, true, false), "out");
		assert_eq!(build_tool_result_text("  out  ", false, true, true), "[tool error] out");
		assert_eq!(
			build_tool_result_text("out", true, false, false),
			"out\n[tool image omitted: model does not support images]"
		);
		assert_eq!(build_tool_result_text("", true, true, false), "(see attached image)");
		assert_eq!(build_tool_result_text("", true, true, true), "[tool error] (see attached image)");
		assert_eq!(
			build_tool_result_text("", true, false, false),
			"(image omitted: model does not support images)"
		);
		assert_eq!(
			build_tool_result_text("", true, false, true),
			"[tool error] (image omitted: model does not support images)"
		);
		assert_eq!(build_tool_result_text("", false, true, false), "(no tool output)");
		assert_eq!(build_tool_result_text("", false, true, true), "[tool error] (no tool output)");
	}

	#[test]
	fn request_options_merge_headers_and_add_affinity() {
		let mut model = model("mistral-large-latest");
		let mut model_headers = IndexMap::new();
		model_headers.insert("x-model".to_string(), "1".to_string());
		model.headers = Some(model_headers);
		let mut options = MistralOptions::from_base(&base_options());
		let mut options_headers = IndexMap::new();
		options_headers.insert("x-option".to_string(), "2".to_string());
		options.stream.headers = Some(options_headers);
		options.stream.session_id = Some("session-1".to_string());
		let request_options = build_request_options(&model, &options);
		assert!(request_options.retries_strategy_none);
		assert_eq!(request_options.headers.get("x-model").map(String::as_str), Some("1"));
		assert_eq!(request_options.headers.get("x-option").map(String::as_str), Some("2"));
		assert_eq!(request_options.headers.get("x-affinity").map(String::as_str), Some("session-1"));

		let mut explicit = MistralOptions::from_base(&base_options());
		let mut explicit_headers = IndexMap::new();
		explicit_headers.insert("x-affinity".to_string(), "caller".to_string());
		explicit.stream.headers = Some(explicit_headers);
		explicit.stream.session_id = Some("session-1".to_string());
		let request_options = build_request_options(&model, &explicit);
		assert_eq!(request_options.headers.get("x-affinity").map(String::as_str), Some("caller"));
	}

	#[test]
	fn sse_parser_joins_data_lines_and_stops_at_done() {
		assert_eq!(parse_sse_message("data: a\ndata: b"), Some("a\nb".to_string()));
		assert_eq!(parse_sse_message(":comment\n\ndata:x"), Some("x".to_string()));
		assert_eq!(parse_sse_message(":only-comment"), None);
		assert_eq!(parse_sse_message("event: ping"), None);
	}

	#[test]
	fn chunk_stream_parses_events_and_honours_done() {
		let mut stream = MistralChunkStream {
			chunks: Box::pin(futures::stream::empty()),
			buffer: "data: {\"id\": \"a\"}\n\ndata: [DONE]\n\ndata: {\"id\": \"b\"}\n\n".to_string(),
			pending: Vec::new(),
			pending_bytes: Vec::new(),
			done: false,
			finished: false,
			signal: None,
		};
		stream.drain_events().unwrap();
		assert_eq!(stream.pending, vec![json!({"id": "a"})]);
		assert!(stream.done);
	}

	#[test]
	fn chunk_stream_reports_malformed_json() {
		let mut stream = MistralChunkStream {
			chunks: Box::pin(futures::stream::empty()),
			buffer: "data: {oops}\n\n".to_string(),
			pending: Vec::new(),
			pending_bytes: Vec::new(),
			done: false,
			finished: false,
			signal: None,
		};
		let error = stream.drain_events().unwrap_err();
		assert!(error.error_message().starts_with("malformed json: "));
	}


	/// The offset scan must agree with the per-boundary min-index scan on the
	/// eight boundary forms, including the 3-byte ones and a `\r\n\r\n` that
	/// straddles two pushes (one boundary, not two events).
	#[test]
	fn chunk_stream_splits_mixed_boundaries() {
		let mut stream = MistralChunkStream {
			chunks: Box::pin(futures::stream::empty()),
			buffer: "data: {\"a\":1}\r\n\rdata: {\"b\":2}\n\rdata: {\"c\":3}\r\rdata: [DONE]\r\n\r\n".to_string(),
			pending: Vec::new(),
			pending_bytes: Vec::new(),
			done: false,
			finished: false,
			signal: None,
		};
		stream.drain_events().unwrap();
		assert_eq!(stream.pending, vec![json!({"a": 1}), json!({"b": 2}), json!({"c": 3})]);
		assert!(stream.done);

		let mut split = MistralChunkStream {
			chunks: Box::pin(futures::stream::empty()),
			buffer: "data: {\"a\":1}\r\n".to_string(),
			pending: Vec::new(),
			pending_bytes: Vec::new(),
			done: false,
			finished: false,
			signal: None,
		};
		split.drain_events().unwrap();
		assert!(split.pending.is_empty());
		split.buffer.push_str("\r\ndata: {\"b\":2}\r\n\n");
		split.drain_events().unwrap();
		assert_eq!(split.pending, vec![json!({"a": 1}), json!({"b": 2})]);
	}
	fn byte_stream(chunks: Vec<&[u8]>) -> MistralChunkStream {
		let items: Vec<reqwest::Result<bytes::Bytes>> = chunks
			.into_iter()
			.map(|chunk| Ok(bytes::Bytes::copy_from_slice(chunk)))
			.collect();
		MistralChunkStream {
			chunks: Box::pin(futures::stream::iter(items)),
			buffer: String::new(),
			pending: Vec::new(),
			pending_bytes: Vec::new(),
			done: false,
			finished: false,
			signal: None,
		}
	}

	/// GM-01: `UsageInfo$inboundSchema` (usageinfo.js:11-23) remaps the snake_case wire
	/// fields `prompt_tokens`/`completion_tokens`/`total_tokens` to the camelCase names
	/// `mistral.ts:306-310` reads; without the remap a real Mistral chunk reports 0 tokens
	/// and 0 cost.
	#[test]
	fn completion_chunk_remaps_usage_and_finish_reason_from_the_wire() {
		let chunk = remap_completion_chunk(json!({
			"id": "abc",
			"model": "mistral-large-latest",
			"usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
			"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
		}));
		assert_eq!(chunk["usage"]["promptTokens"], json!(100));
		assert_eq!(chunk["usage"]["completionTokens"], json!(40));
		assert_eq!(chunk["usage"]["totalTokens"], json!(140));
		assert_eq!(chunk["choices"][0]["finishReason"], json!("length"));
		assert_eq!(
			map_chat_stop_reason(chunk["choices"][0]["finishReason"].as_str()),
			"length"
		);

		// `remap$` keeps unmapped keys and drops the snake_case source key.
		assert_eq!(chunk["usage"].get("prompt_tokens"), None);
		assert_eq!(chunk["id"], json!("abc"));
	}

	/// GM-01: `DeltaMessage$inboundSchema` (deltamessage.js:24-40) remaps
	/// `delta.tool_calls` to the `toolCalls` key `mistral.ts:388-401` reads; without it
	/// streamed tool calls are silently dropped.
	#[test]
	fn completion_chunk_remaps_streamed_tool_calls_from_the_wire() {
		let chunk = remap_completion_chunk(json!({
			"id": "abc",
			"model": "mistral-large-latest",
			"choices": [{
				"index": 0,
				"finish_reason": "tool_calls",
				"delta": {
					"tool_calls": [{
						"id": "call-1",
						"index": 0,
						"function": {"name": "read", "arguments": "{\"path\":\"a\"}"}
					}],
					"tool_call_id": "call-1"
				}
			}]
		}));
		let delta = &chunk["choices"][0]["delta"];
		assert_eq!(delta["toolCalls"][0]["id"], json!("call-1"));
		assert_eq!(delta["toolCalls"][0]["function"]["name"], json!("read"));
		assert_eq!(delta["toolCallId"], json!("call-1"));
		assert_eq!(delta.get("tool_calls"), None);
		assert_eq!(
			map_chat_stop_reason(chunk["choices"][0]["finishReason"].as_str()),
			"toolUse"
		);
	}

	/// GM-01: the remap runs on the events the SSE reader yields, so a real wire frame
	/// reaches `consume_chat_stream` with the camelCase keys the TypeScript reads.
	#[tokio::test]
	async fn chunk_stream_remaps_the_wire_frame_before_yielding_it() {
		let frame = concat!(
			"data: {\"id\":\"a\",\"model\":\"m\",",
			"\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10},",
			"\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call-1\",\"index\":0,",
			"\"function\":{\"name\":\"read\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n"
		);
		let mut stream = byte_stream(vec![frame.as_bytes()]);
		let chunk = stream.next().await.unwrap().unwrap();
		assert_eq!(chunk["usage"]["promptTokens"], json!(7));
		assert_eq!(chunk["usage"]["totalTokens"], json!(10));
		assert_eq!(chunk["choices"][0]["finishReason"], json!("tool_calls"));
		assert_eq!(chunk["choices"][0]["delta"]["toolCalls"][0]["id"], json!("call-1"));
	}

	/// GM-04: `parseMessage` decodes each message with the same stateful
	/// `new TextDecoder()` across the chunks that make up that message, so a multi-byte
	/// character split at a network chunk boundary survives instead of becoming U+FFFD.
	#[tokio::test]
	async fn chunk_stream_keeps_multibyte_characters_split_across_chunks() {
		// 'é' is C3 A9 and '🎈' is F0 9F 8E 88; one split lands inside each character.
		let event = "data: {\"id\":\"a\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"caf\u{e9} \u{1F388}\"}}]}\n\n";
		let bytes = event.as_bytes();
		let first_split = event.find('\u{e9}').unwrap() + 1;
		let second_split = event.find('\u{1F388}').unwrap() + 2;
		let mut stream = byte_stream(vec![
			&bytes[..first_split],
			&bytes[first_split..second_split],
			&bytes[second_split..],
		]);
		let chunk = stream.next().await.unwrap().unwrap();
		assert_eq!(
			chunk["choices"][0]["delta"]["content"],
			json!("caf\u{e9} \u{1F388}")
		);
	}

	/// GM-05: mistral.ts:113-116 throws "No API key for provider: ..." out of
	/// `streamSimpleMistral`; the port reports the same message as a terminal `error`
	/// event instead of panicking (a panic would abort the whole process).
	#[tokio::test]
	async fn stream_simple_mistral_reports_a_missing_api_key_through_the_stream() {
		let mut env = crate::test_env::ScopedEnv::new();
		env.remove("MISTRAL_API_KEY");
		let model = model("mistral-large-latest");
		let context = Context::new(None, vec![], None);
		let stream = stream_simple_mistral(&model, &context, None);
		let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
			.await
			.expect("the api-key error must be reported without blocking")
			.expect("the stream must carry the error event");
		let AssistantMessageEvent::Error { reason, error } = event else {
			panic!("expected provider error event")
		};
		assert_eq!(reason, "error");
		assert_eq!(error.stop_reason, "error");
		assert_eq!(
			error.error_message.as_deref(),
			Some("No API key for provider: mistral")
		);
		assert!(stream.is_done());
		assert!(stream.next().await.is_none());
	}

	#[test]
	fn number_field_defaults_to_zero() {
		assert_eq!(number_field(&json!({"promptTokens": 5}), "promptTokens"), 5.0);
		assert_eq!(number_field(&json!({}), "promptTokens"), 0.0);
	}
}


#[cfg(test)]
mod t15_controls_tests {
	//! T15 owner 'controls': E-02 - the Mistral stream must not cap at the SDK's
	//! 30 s default when a caller signal is present (TS SDK
	//! `node_modules/@mistralai/mistralai/src/lib/sdks.ts` arms the default timeout
	//! only `if (!fetchOptions?.signal && conf.timeoutMs > 0)`).
	//!
	//! Isolation: ephemeral loopback ports, synthetic key, no production pipe/port.

	use super::*;
	use crate::types::{ContentBlock, Context, InputModality, Message, Model, ModelCost, StreamOptions, UserContent, UserMessage};
	use tokio::io::AsyncReadExt;
	use tokio_util::sync::CancellationToken;

	fn controls_model(id: &str) -> Model {
		let mut model = Model::new(id, id, "mistral-conversations", "mistral", "https://api.mistral.ai");
		model.input = vec![InputModality::Text];
		model.cost = ModelCost::zero();
		model
	}

	fn controls_context() -> Context {
		Context::new(
			None,
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			None,
		)
	}

	fn controls_options(api_key: &str, timeout_ms: Option<f64>, signal: Option<CancellationToken>) -> MistralOptions {
		let mut options = MistralOptions::from_base(&StreamOptions::default());
		options.stream.api_key = Some(api_key.to_string());
		options.stream.timeout_ms = timeout_ms;
		options.stream.signal = signal;
		options
	}

	/// Sends `frames` Mistral content frames, one every `interval_ms`, then a final
	/// usage/finish frame and `[DONE]`, close-delimited (no Content-Length).
	async fn slow_sse_server(frames: usize, interval_ms: u64) -> String {
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
			.await
			.unwrap();
		let address = listener.local_addr().unwrap();
		tokio::spawn(async move {
			let Ok((mut socket, _)) = listener.accept().await else {
				return;
			};
			let mut request = Vec::new();
			loop {
				let mut buffer = [0u8; 4096];
				match socket.read(&mut buffer).await {
					Ok(0) | Err(_) => return,
					Ok(count) => {
						request.extend_from_slice(&buffer[..count]);
						if request.windows(4).any(|window| window == b"\r\n\r\n") {
							break;
						}
					}
				}
			}
			let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
			if socket.write_all(header.as_bytes()).await.is_err() {
				return;
			}
			for index in 0..frames {
				tokio::time::sleep(std::time::Duration::from_millis(interval_ms)).await;
				let frame = format!(
					"data: {{\"id\":\"cmpl-1\",\"model\":\"mistral-large-latest\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"d{index}\"}}}}]}}\n\n"
				);
				if socket.write_all(frame.as_bytes()).await.is_err() {
					return;
				}
			}
			let final_frame = format!(
				"data: {{\"id\":\"cmpl-1\",\"model\":\"mistral-large-latest\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":1000,\"completion_tokens\":{frames},\"total_tokens\":{}}}}}\n\n",
				1000 + frames
			);
			let _ = socket.write_all(final_frame.as_bytes()).await;
			let _ = socket.write_all(b"data: [DONE]\n\n").await;
			let _ = socket.shutdown().await;
		});
		format!("http://{address}")
	}

	/// Accepts one request and never answers (for explicit-timeout controls).
	async fn stalling_server() -> String {
		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
			.await
			.unwrap();
		let address = listener.local_addr().unwrap();
		tokio::spawn(async move {
			if let Ok((mut socket, _)) = listener.accept().await {
				let mut buffer = [0u8; 4096];
				let _ = socket.read(&mut buffer).await;
				// Never answer within the test window; the socket is dropped with the task.
				tokio::time::sleep(std::time::Duration::from_secs(60)).await;
			}
		});
		format!("http://{address}")
	}

	fn text_of(message: &AssistantMessage) -> String {
		message
			.content
			.iter()
			.filter_map(|block| match block {
				ContentBlock::Text(text) => Some(text.text.clone()),
				_ => None,
			})
			.collect::<Vec<_>>()
			.join("")
	}

	/// E-02 reproduction: a 22-frame stream at 1.6 s/frame (35 s total) with a LIVE
	/// caller signal and no explicit timeout must complete in Optimus exactly like
	/// the TypeScript SDK (which arms no timeout when a signal is present).
	/// Baseline expectation: the unconditional 30 s reqwest timeout kills the stream
	/// with a transport error at ~30 s, so this test FAILS on the unmodified tree.
	#[tokio::test]
	async fn t15_mistral_stream_timeout_respects_caller_signal() {
		let base_url = slow_sse_server(22, 1_600).await;
		let mut model = controls_model("mistral-large-latest");
		model.base_url = base_url;
		let signal = CancellationToken::new();
		let options = controls_options("test-key", None, Some(signal));
		let started = std::time::Instant::now();
		let stream = stream_mistral(&model, &controls_context(), Some(options));
		let message = tokio::time::timeout(std::time::Duration::from_secs(90), stream.result())
			.await
			.expect("the stream must finish well inside the 90 s test window");
		let elapsed = started.elapsed();
		assert_eq!(
			message.stop_reason, "stop",
			"the stream must complete like the TS contract; got stop_reason {:?} after {elapsed:?} (error: {:?})",
			message.stop_reason, message.error_message
		);
		assert!(elapsed >= std::time::Duration::from_millis(33_000), "the fixture must cross the disputed 30 s boundary; elapsed {elapsed:?}");
		let text = text_of(&message);
		assert!(text.contains("d21"), "the final frame's delta must arrive before DONE; got {text:?}");
		assert_eq!(message.usage.input, 1000.0);
		assert_eq!(message.usage.output, 22.0);
	}

	/// Negative control (must stay green in baseline AND candidate): caller
	/// cancellation ends the stream promptly even without any timeout.
	#[tokio::test]
	async fn t15_mistral_stream_caller_cancellation_ends_stream() {
		let base_url = slow_sse_server(100, 1_000).await;
		let mut model = controls_model("mistral-large-latest");
		model.base_url = base_url;
		let signal = CancellationToken::new();
		let options = controls_options("test-key", None, Some(signal.clone()));
		let stream = stream_mistral(&model, &controls_context(), Some(options));
		tokio::time::sleep(std::time::Duration::from_millis(2_000)).await;
		signal.cancel();
		let started = std::time::Instant::now();
		let message = tokio::time::timeout(std::time::Duration::from_secs(15), stream.result())
			.await
			.expect("cancellation must end the stream promptly");
		assert!(started.elapsed() < std::time::Duration::from_secs(5), "cancellation must not wait out any timer; elapsed {:?}", started.elapsed());
		assert_eq!(message.stop_reason, "aborted");
	}

	/// Guard control (must stay green in baseline AND candidate): an explicit
	/// `timeout_ms` still applies even when a caller signal is present, so the
	/// E-02 repair cannot overreach into disabling explicit timeouts.
	#[tokio::test]
	async fn t15_mistral_stream_explicit_timeout_ms_still_applies() {
		let base_url = stalling_server().await;
		let mut model = controls_model("mistral-large-latest");
		model.base_url = base_url;
		let signal = CancellationToken::new();
		let options = controls_options("test-key", Some(2_000.0), Some(signal));
		let started = std::time::Instant::now();
		let stream = stream_mistral(&model, &controls_context(), Some(options));
		let message = tokio::time::timeout(std::time::Duration::from_secs(15), stream.result())
			.await
			.expect("the explicit timeout must end the stream");
		assert!(started.elapsed() < std::time::Duration::from_secs(8), "the explicit 2 s timeout must fire at ~2 s; elapsed {:?}", started.elapsed());
		assert_eq!(message.stop_reason, "error");
		assert!(message.error_message.is_some(), "the timeout must surface as a stream error");
	}
}
