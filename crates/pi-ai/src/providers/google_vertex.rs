//! Port of packages/ai/src/providers/google-vertex.ts
//!
//! The TypeScript uses `@google/genai` in Vertex mode (`vertexai: true`). This port
//! builds the same `POST {baseUrl}/{apiVersion}/[projects/{p}/locations/{l}/]publishers/google/models/{model}:streamGenerateContent?alt=sse`
//! request itself with `reqwest` and parses the SSE stream locally, keeping the SDK's
//! Vertex URL derivation, the API-key vs OAuth-token header rules and the error text.

use std::sync::atomic::{AtomicI64, Ordering};

use futures::StreamExt;
use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::models::{calculate_cost, clamp_thinking_level};
use crate::providers::google_shared::{
	convert_messages, convert_tools, get_google_thinking_budget, is_thinking_part, map_stop_reason, map_tool_choice,
	retain_thought_signature, GoogleThinkingLevel,
};
use crate::providers::simple_options::build_base_options;
use crate::types::{
	AssistantMessage, AssistantMessageEvent, ContentBlock, Context, Model, SimpleStreamOptions, StopReason,
	StreamOptions, TextContent, ThinkingContent, ToolCall, Usage,
};
use crate::utils::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use crate::utils::headers::header_map_to_record;
use crate::utils::now_ms;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::stream_failure::{
	format_stream_failure_message, record_stream_failure, stream_failure_from_stop_reason, StreamFailureError,
	ThrownStreamError,
};

const API_VERSION: &str = "v1";
const GCP_VERTEX_CREDENTIALS_MARKER: &str = "gcp-vertex-credentials";

/// The pinned `@google/genai` version, used for the SDK default headers.
const GOOGLE_GENAI_SDK_VERSION: &str = "1.52.0";
const LIBRARY_LABEL: &str = "google-genai-sdk/1.52.0";
const USER_AGENT_HEADER: &str = "User-Agent";
const GOOGLE_API_CLIENT_HEADER: &str = "x-goog-api-client";
const CONTENT_TYPE_HEADER: &str = "Content-Type";
const GOOGLE_API_KEY_HEADER: &str = "x-goog-api-key";
/// `MULTI_REGIONAL_LOCATIONS = new Set(['us', 'eu'])`.
const MULTI_REGIONAL_LOCATIONS: [&str; 2] = ["us", "eu"];

/// TS: `THINKING_LEVEL_MAP`.
pub fn thinking_level_map(level: &GoogleThinkingLevel) -> &'static str {
	match level.as_str() {
		"THINKING_LEVEL_UNSPECIFIED" => "THINKING_LEVEL_UNSPECIFIED",
		"MINIMAL" => "MINIMAL",
		"LOW" => "LOW",
		"MEDIUM" => "MEDIUM",
		"HIGH" => "HIGH",
		// `Record<GoogleThinkingLevel, ThinkingLevel>` is total, so the lookup always
		// returns one of the five enum values.
		_ => "THINKING_LEVEL_UNSPECIFIED",
	}
}

/// TS: `interface GoogleVertexOptions extends StreamOptions`.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GoogleVertexOptions {
	#[serde(flatten)]
	pub stream: StreamOptions,
	/// `"auto" | "none" | "any"`
	pub tool_choice: Option<String>,
	pub thinking: Option<GoogleVertexThinkingOptions>,
	pub project: Option<String>,
	pub location: Option<String>,
}

/// TS: `thinking?: { enabled: boolean; budgetTokens?: number; level?: GoogleThinkingLevel }`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GoogleVertexThinkingOptions {
	pub enabled: bool,
	/// -1 for dynamic, 0 to disable
	pub budget_tokens: Option<f64>,
	pub level: Option<GoogleThinkingLevel>,
}

impl GoogleVertexOptions {
	/// TS: the caller passes `StreamOptions & Record<string, unknown>`; this keeps the
	/// non-serializable fields (signal, on_payload, on_response, on_usage_observation).
	pub fn from_base(base: &StreamOptions) -> Self {
		Self {
			stream: base.clone(),
			tool_choice: None,
			thinking: None,
			project: None,
			location: None,
		}
	}
}

impl std::fmt::Debug for GoogleVertexOptions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("GoogleVertexOptions")
			.field("stream", &self.stream)
			.field("tool_choice", &self.tool_choice)
			.field("thinking", &self.thinking)
			.field("project", &self.project)
			.field("location", &self.location)
			.finish()
	}
}

/// TS: `let toolCallCounter = 0` - module-level counter shared by all streams.
static TOOL_CALL_COUNTER: AtomicI64 = AtomicI64::new(0);

/// `++toolCallCounter`.
fn next_tool_call_counter() -> i64 {
	TOOL_CALL_COUNTER.fetch_add(1, Ordering::SeqCst) + 1
}

/// The TypeScript throws either a plain `Error`, a `StreamFailureError` or an SDK
/// `ApiError`; the port keeps all three shapes.
#[derive(Debug, Clone, PartialEq)]
pub enum GoogleVertexStreamError {
	Failure(StreamFailureError),
	Message(String),
	/// TS: `throw new ApiError({ message, status })`.
	Api { message: String, status: i64, value: Value },
}

impl GoogleVertexStreamError {
	fn as_thrown(&self) -> ThrownStreamError<'_> {
		match self {
			GoogleVertexStreamError::Failure(failure) => ThrownStreamError::Failure(failure),
			GoogleVertexStreamError::Message(message) => ThrownStreamError::Message(message),
			GoogleVertexStreamError::Api { value, .. } => ThrownStreamError::Value(value),
		}
	}

	/// `error.message`.
	fn message(&self) -> String {
		match self {
			GoogleVertexStreamError::Failure(failure) => failure.message.clone(),
			GoogleVertexStreamError::Message(message) => message.clone(),
			GoogleVertexStreamError::Api { message, .. } => message.clone(),
		}
	}
}

impl std::fmt::Display for GoogleVertexStreamError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(formatter, "{}", self.message())
	}
}

impl std::error::Error for GoogleVertexStreamError {}

/// TS: `streamGoogleVertex: StreamFunction<"google-vertex", GoogleVertexOptions>`.
pub fn stream_google_vertex(
	model: &Model,
	context: &Context,
	options: Option<GoogleVertexOptions>,
) -> AssistantMessageEventStream {
	let stream = create_assistant_message_event_stream();
	let out = stream.clone();
	let model = model.clone();
	let context = context.clone();
	let options = options.unwrap_or_default();
	tokio::spawn(async move {
		let mut output = AssistantMessage {
			content: Vec::new(),
			api: "google-vertex".to_string(),
			provider: model.provider.clone(),
			model: model.id.clone(),
			usage: Usage::zero(),
			stop_reason: "stop".to_string(),
			timestamp: now_ms(),
			..Default::default()
		};

		match run_stream_google_vertex(&model, &context, &options, &mut output, &out).await {
			Ok(()) => {}
			Err(error) => {
				// Remove internal index property used during streaming
				// (the Rust port never adds an `index` field to a block).
				let aborted = options
					.stream
					.signal
					.as_ref()
					.map(|signal| signal.is_cancelled())
					.unwrap_or(false);
				output.stop_reason = if aborted { "aborted".to_string() } else { "error".to_string() };
				output.error_message = Some(format_stream_failure_message(&error.as_thrown()));
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

/// The TypeScript async IIFE body of `streamGoogleVertex`.
async fn run_stream_google_vertex(
	model: &Model,
	context: &Context,
	options: &GoogleVertexOptions,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
) -> Result<(), GoogleVertexStreamError> {
	let api_key = resolve_api_key(Some(options));
	let client = match &api_key {
		Some(api_key) => create_client_with_api_key(model, api_key, options.stream.headers.as_ref())?,
		None => create_client(
			model,
			&resolve_project(Some(options))?,
			&resolve_location(Some(options))?,
			options.stream.headers.as_ref(),
		)?,
	};
	let mut params = build_params(model, context, Some(options))?;
	if let Some(on_payload) = options.stream.on_payload.clone() {
		let next_params = on_payload(params.clone(), model).await;
		if let Some(next_params) = next_params {
			params = next_params;
		}
	}
	let response = send_request(&client, &params, model, options).await?;
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
	let mut current_block: Option<CurrentBlock> = None;
	let mut chunks = SseChunkStream::new(response, options.stream.signal.clone());

	while let Some(chunk) = chunks.next().await? {
		// Vertex uses the same @google/genai GenerateContentResponse type as Gemini.
		// responseId is documented there as an output-only identifier for each response.
		if output.response_id.is_none() {
			if let Some(response_id) = chunk.get("responseId").and_then(Value::as_str) {
				if !response_id.is_empty() {
					output.response_id = Some(response_id.to_string());
				}
			}
		}
		let candidate = chunk.get("candidates").and_then(Value::as_array).and_then(|list| list.first());
		if let Some(parts) = candidate
			.and_then(|candidate| candidate.get("content"))
			.and_then(|content| content.get("parts"))
			.and_then(Value::as_array)
		{
			for part in parts {
				process_part(part, output, stream, &mut current_block)?;
			}
		}

		if let Some(finish_reason) = candidate
			.and_then(|candidate| candidate.get("finishReason"))
			.and_then(Value::as_str)
		{
			output.stop_reason = map_stop_reason(finish_reason).map_err(GoogleVertexStreamError::Message)?;
			if output
				.content
				.iter()
				.any(|block| matches!(block, ContentBlock::ToolCall(_)))
			{
				output.stop_reason = "toolUse".to_string();
			}
			if output.stop_reason == "error" {
				output.stop_reason_raw = Some(finish_reason.to_string());
			}
		}

		if let Some(usage_metadata) = chunk.get("usageMetadata") {
			output.usage = super::google_usage::normalize(usage_metadata);
			calculate_cost(model, &mut output.usage, None);
		}
	}

	if let Some(block) = current_block {
		finish_block(&block, output, stream);
	}

	if options
		.stream
		.signal
		.as_ref()
		.map(|signal| signal.is_cancelled())
		.unwrap_or(false)
	{
		return Err(GoogleVertexStreamError::Message("Request was aborted".to_string()));
	}

	if output.stop_reason == "aborted" || output.stop_reason == "error" {
		return Err(GoogleVertexStreamError::Failure(stream_failure_from_stop_reason(
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

/// The block the streaming loop is currently filling.
#[derive(Debug, Clone, PartialEq)]
enum CurrentBlock {
	Text(TextContent),
	Thinking(ThinkingContent),
}

/// TS: the `if (currentBlock) { ...text_end/thinking_end... }` flush.
fn finish_block(block: &CurrentBlock, output: &AssistantMessage, stream: &AssistantMessageEventStream) {
	match block {
		CurrentBlock::Text(text) => {
			stream.push(AssistantMessageEvent::TextEnd {
				content_index: output.content.len() - 1,
				content: text.text.clone(),
				partial: output.clone(),
			});
		}
		CurrentBlock::Thinking(thinking) => {
			stream.push(AssistantMessageEvent::ThinkingEnd {
				content_index: output.content.len() - 1,
				content: thinking.thinking.clone(),
				partial: output.clone(),
			});
		}
	}
}

/// The inner `for (const part of candidate.content.parts)` body.
fn process_part(
	part: &Value,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
	current_block: &mut Option<CurrentBlock>,
) -> Result<(), GoogleVertexStreamError> {
	if let Some(text) = part.get("text").and_then(Value::as_str) {
		let is_thinking = is_thinking_part(part);
		let needs_new_block = match current_block {
			None => true,
			Some(CurrentBlock::Thinking(_)) => !is_thinking,
			Some(CurrentBlock::Text(_)) => is_thinking,
		};
		if needs_new_block {
			if let Some(block) = current_block.take() {
				finish_block(&block, output, stream);
			}
			if is_thinking {
				output
					.content
					.push(ContentBlock::Thinking(ThinkingContent::new(String::new())));
				stream.push(AssistantMessageEvent::ThinkingStart {
					content_index: output.content.len() - 1,
					partial: output.clone(),
				});
				*current_block = Some(CurrentBlock::Thinking(ThinkingContent::new(String::new())));
			} else {
				output.content.push(ContentBlock::Text(TextContent::new(String::new())));
				stream.push(AssistantMessageEvent::TextStart {
					content_index: output.content.len() - 1,
					partial: output.clone(),
				});
				*current_block = Some(CurrentBlock::Text(TextContent::new(String::new())));
			}
		}

		let thought_signature = part.get("thoughtSignature").and_then(Value::as_str);
		match current_block {
			Some(CurrentBlock::Thinking(thinking)) => {
				thinking.thinking.push_str(text);
				thinking.thinking_signature =
					retain_thought_signature(thinking.thinking_signature.as_deref(), thought_signature);
				if let Some(ContentBlock::Thinking(block)) = output.content.last_mut() {
					block.thinking = thinking.thinking.clone();
					block.thinking_signature = thinking.thinking_signature.clone();
				}
				stream.push(AssistantMessageEvent::ThinkingDelta {
					content_index: output.content.len() - 1,
					delta: text.to_string(),
					partial: output.clone(),
				});
			}
			Some(CurrentBlock::Text(current)) => {
				current.text.push_str(text);
				current.text_signature = retain_thought_signature(current.text_signature.as_deref(), thought_signature);
				if let Some(ContentBlock::Text(block)) = output.content.last_mut() {
					block.text = current.text.clone();
					block.text_signature = current.text_signature.clone();
				}
				stream.push(AssistantMessageEvent::TextDelta {
					content_index: output.content.len() - 1,
					delta: text.to_string(),
					partial: output.clone(),
				});
			}
			None => {}
		}
	}

	if let Some(function_call) = part.get("functionCall") {
		if let Some(block) = current_block.take() {
			finish_block(&block, output, stream);
		}

		let provided_id = function_call.get("id").and_then(Value::as_str);
		let name = function_call.get("name").and_then(Value::as_str).unwrap_or("");
		// TS: `!providedId || output.content.some(...)` - an empty string id is
		// falsy in JS, so it is regenerated.
		let needs_new_id = match provided_id {
			None | Some("") => true,
			Some(provided_id) => output.content.iter().any(|block| match block {
				ContentBlock::ToolCall(tool_call) => tool_call.id == provided_id,
				_ => false,
			}),
		};
		let tool_call_id = if needs_new_id {
			format!("{}_{}_{}", name, now_ms(), next_tool_call_counter())
		} else {
			provided_id.unwrap_or_default().to_string()
		};

		let arguments = match function_call.get("args") {
			Some(Value::Object(map)) => map.clone(),
			_ => Map::new(),
		};
		let tool_call = ToolCall {
			id: tool_call_id,
			name: name.to_string(),
			arguments,
			thought_signature: part
				.get("thoughtSignature")
				.and_then(Value::as_str)
				.map(str::to_string),
			..Default::default()
		};

		output.content.push(ContentBlock::ToolCall(tool_call.clone()));
		stream.push(AssistantMessageEvent::ToolCallStart {
			content_index: output.content.len() - 1,
			partial: output.clone(),
		});
		stream.push(AssistantMessageEvent::ToolCallDelta {
			content_index: output.content.len() - 1,
			delta: serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string()),
			partial: output.clone(),
		});
		stream.push(AssistantMessageEvent::ToolCallEnd {
			content_index: output.content.len() - 1,
			tool_call,
			partial: output.clone(),
		});
	}

	Ok(())
}

// ---------------------------------------------------------------------------
// Client / request
// ---------------------------------------------------------------------------

/// TS: `HttpOptions`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VertexHttpOptions {
	pub base_url: Option<String>,
	/// `baseUrlResourceScope: ResourceScope.COLLECTION`
	pub base_url_resource_scope_collection: bool,
	pub api_version: Option<String>,
	pub headers: Option<IndexMap<String, Option<String>>>,
}

/// The fields of a constructed Vertex `GoogleGenAI` client that this provider reads.
#[derive(Debug, Clone, PartialEq)]
pub struct GoogleVertexClient {
	pub api_key: Option<String>,
	pub project: Option<String>,
	pub location: Option<String>,
	pub api_version: String,
	pub base_url: String,
	/// `baseUrlResourceScope === ResourceScope.COLLECTION`
	pub custom_base_url_collection: bool,
	pub headers: IndexMap<String, Option<String>>,
}

/// TS: `createClient(model, project, location, optionsHeaders?)`.
fn create_client(
	model: &Model,
	project: &str,
	location: &str,
	options_headers: Option<&IndexMap<String, String>>,
) -> Result<GoogleVertexClient, GoogleVertexStreamError> {
	build_vertex_client(model, None, Some(project), Some(location), options_headers)
}

/// TS: `createClientWithApiKey(model, apiKey, optionsHeaders?)`.
fn create_client_with_api_key(
	model: &Model,
	api_key: &str,
	options_headers: Option<&IndexMap<String, String>>,
) -> Result<GoogleVertexClient, GoogleVertexStreamError> {
	build_vertex_client(model, Some(api_key), None, None, options_headers)
}

/// The `new GoogleGenAI({ vertexai: true, ... })` constructor path: `ApiClient`
/// normalises project/location/apiKey, derives the base URL and applies
/// `patchHttpOptions(initHttpOptions, opts.httpOptions)`.
fn build_vertex_client(
	model: &Model,
	api_key: Option<&str>,
	project: Option<&str>,
	location: Option<&str>,
	options_headers: Option<&IndexMap<String, String>>,
) -> Result<GoogleVertexClient, GoogleVertexStreamError> {
	let mut project = project.map(str::to_string);
	let mut location = location.map(str::to_string);
	let mut api_key = api_key.map(str::to_string);
	if project.is_some() && location.is_some() {
		api_key = None;
	} else if api_key.is_some() {
		project = None;
		location = None;
	}

	let http_options = build_http_options(model, options_headers);
	let custom_base_url = http_options.base_url.clone();

	if location.is_none() && api_key.is_none() && custom_base_url.is_none() {
		location = Some("global".to_string());
	}
	let has_sufficient_auth = (project.is_some() && location.is_some()) || api_key.is_some();
	if !has_sufficient_auth && custom_base_url.is_none() {
		return Err(GoogleVertexStreamError::Message(
			"Authentication is not set up. Please provide either a project and location, or an API key, or a custom base URL."
				.to_string(),
		));
	}
	let has_constructor_auth = (project.is_some() && location.is_some()) || api_key.is_some();
	let mut init_base_url: Option<String> = None;
	if custom_base_url.is_some() && !has_constructor_auth {
		init_base_url = custom_base_url.clone();
		project = None;
		location = None;
	} else if api_key.is_some() || location.as_deref() == Some("global") {
		// Vertex Express or global endpoint case.
		init_base_url = Some("https://aiplatform.googleapis.com/".to_string());
	} else if let (Some(project_id), Some(location_id)) = (project.as_deref(), location.as_deref()) {
		let _ = project_id;
		if MULTI_REGIONAL_LOCATIONS.contains(&location_id) {
			init_base_url = Some(format!("https://aiplatform.{}.rep.googleapis.com/", location_id));
		} else {
			init_base_url = Some(format!("https://{}-aiplatform.googleapis.com/", location_id));
		}
	}

	// `patchHttpOptions(initHttpOptions, opts.httpOptions)`.
	let mut base_url = init_base_url.unwrap_or_default();
	let mut api_version = API_VERSION.to_string();
	if let Some(custom) = &http_options.base_url {
		base_url = custom.clone();
	}
	if let Some(custom_api_version) = &http_options.api_version {
		api_version = custom_api_version.clone();
	}

	let mut headers: IndexMap<String, Option<String>> = IndexMap::new();
	if let Some(model_headers) = &model.headers {
		for (key, value) in model_headers {
			headers.insert(key.clone(), Some(value.clone()));
		}
	}
	if let Some(options_headers) = options_headers {
		for (key, value) in options_headers {
			headers.insert(key.clone(), Some(value.clone()));
		}
	}

	Ok(GoogleVertexClient {
		api_key,
		project,
		location,
		api_version,
		base_url,
		custom_base_url_collection: http_options.base_url_resource_scope_collection,
		headers,
	})
}

/// TS: `buildHttpOptions(model, optionsHeaders?)`.
fn build_http_options(
	model: &Model,
	options_headers: Option<&IndexMap<String, String>>,
) -> VertexHttpOptions {
	let mut http_options = VertexHttpOptions::default();
	if let Some(base_url) = resolve_custom_base_url(&model.base_url) {
		http_options.base_url = Some(base_url.clone());
		http_options.base_url_resource_scope_collection = true;
		if base_url_includes_api_version(&base_url) {
			http_options.api_version = Some(String::new());
		}
	}

	if model.headers.is_some() || options_headers.is_some() {
		let mut headers: IndexMap<String, Option<String>> = IndexMap::new();
		if let Some(model_headers) = &model.headers {
			for (key, value) in model_headers {
				headers.insert(key.clone(), Some(value.clone()));
			}
		}
		if let Some(options_headers) = options_headers {
			for (key, value) in options_headers {
				headers.insert(key.clone(), Some(value.clone()));
			}
		}
		http_options.headers = Some(headers);
	}

	http_options
}

/// TS: `resolveCustomBaseUrl(baseUrl)`.
fn resolve_custom_base_url(base_url: &str) -> Option<String> {
	let trimmed = base_url.trim();
	if trimmed.is_empty() || trimmed.contains("{location}") {
		return None;
	}
	Some(trimmed.to_string())
}

/// TS: `baseUrlIncludesApiVersion(baseUrl)`.
fn base_url_includes_api_version(base_url: &str) -> bool {
	match url::Url::parse(base_url) {
		Ok(url) => url.path().split('/').any(is_api_version_segment),
		Err(_) => base_url.split('/').any(is_api_version_segment),
	}
}

/// `^v\d+(?:beta\d*)?$`
fn is_api_version_segment(part: &str) -> bool {
	let Some(rest) = part.strip_prefix('v') else {
		return false;
	};
	let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
	if digits.is_empty() {
		return false;
	}
	let tail = &rest[digits.len()..];
	if tail.is_empty() {
		return true;
	}
	let Some(beta_digits) = tail.strip_prefix("beta") else {
		return false;
	};
	beta_digits.chars().all(|c| c.is_ascii_digit())
}

/// TS: `resolveApiKey(options?)`.
pub fn resolve_api_key(options: Option<&GoogleVertexOptions>) -> Option<String> {
	let api_key = options
		.and_then(|options| options.stream.api_key.as_ref())
		.map(|key| key.trim().to_string())
		.filter(|key| !key.is_empty())
		.or_else(|| {
			std::env::var("GOOGLE_CLOUD_API_KEY")
				.ok()
				.map(|value| value.trim().to_string())
				.filter(|value| !value.is_empty())
		});
	let api_key = api_key?;
	if api_key == GCP_VERTEX_CREDENTIALS_MARKER || is_placeholder_api_key(&api_key) {
		return None;
	}
	Some(api_key)
}

/// TS: `isPlaceholderApiKey(apiKey)` - `/^<[^>]+>$/`.
fn is_placeholder_api_key(api_key: &str) -> bool {
	let mut chars = api_key.chars();
	if chars.next() != Some('<') {
		return false;
	}
	let rest: String = chars.collect();
	let Some(inner) = rest.strip_suffix('>') else {
		return false;
	};
	!inner.is_empty() && !inner.contains('>')
}

/// TS: `resolveProject(options?)`.
pub fn resolve_project(options: Option<&GoogleVertexOptions>) -> Result<String, GoogleVertexStreamError> {
	let project = options
		.and_then(|options| options.project.clone())
		.or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
		.or_else(|| std::env::var("GCLOUD_PROJECT").ok());
	let Some(project) = project else {
		return Err(GoogleVertexStreamError::Message(
			"Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
				.to_string(),
		));
	};
	Ok(project)
}

/// TS: `resolveLocation(options?)`.
pub fn resolve_location(options: Option<&GoogleVertexOptions>) -> Result<String, GoogleVertexStreamError> {
	let location = options
		.and_then(|options| options.location.clone())
		.or_else(|| std::env::var("GOOGLE_CLOUD_LOCATION").ok());
	let Some(location) = location else {
		return Err(GoogleVertexStreamError::Message(
			"Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.".to_string(),
		));
	};
	Ok(location)
}

/// TS: `getRequestUrlInternal(httpOptions)`.
fn request_url(client: &GoogleVertexClient) -> String {
	let base_url = client.base_url.strip_suffix('/').unwrap_or(&client.base_url);
	let mut elements = vec![base_url.to_string()];
	if !client.api_version.is_empty() {
		elements.push(client.api_version.clone());
	}
	elements.join("/")
}

/// TS: `getBaseResourcePath()`.
fn base_resource_path(client: &GoogleVertexClient) -> String {
	format!(
		"projects/{}/locations/{}",
		client.project.clone().unwrap_or_default(),
		client.location.clone().unwrap_or_default()
	)
}

/// TS: `tModel(apiClient, model)` for a Vertex client.
fn t_model(model_id: &str) -> Result<String, String> {
	if model_id.is_empty() {
		return Err("model is required and must be a string".to_string());
	}
	if model_id.contains("..") || model_id.contains('?') || model_id.contains('&') {
		return Err("invalid model parameter".to_string());
	}
	if model_id.starts_with("publishers/") || model_id.starts_with("projects/") || model_id.starts_with("models/") {
		return Ok(model_id.to_string());
	}
	if let Some(index) = model_id.find('/') {
		let publisher = &model_id[..index];
		let model = &model_id[index + 1..];
		return Ok(format!("publishers/{}/models/{}", publisher, model));
	}
	Ok(format!("publishers/google/models/{}", model_id))
}

/// TS: `shouldPrependVertexProjectPath(request, httpOptions)`.
fn should_prepend_project_location(client: &GoogleVertexClient) -> bool {
	if client.custom_base_url_collection {
		return false;
	}
	if client.api_key.is_some() {
		return false;
	}
	// `clientOptions.vertexai` is always true for this provider.
	true
}

/// TS: `getHeadersInternal(httpOptions, url)` - SDK defaults patched with
/// `httpOptions.headers`, then `auth.addAuthHeaders(headers, url)`.
///
/// The Node auth object adds `x-goog-api-key` when an API key is configured,
/// otherwise it signs the request with Google Application Default Credentials
/// (an `Authorization: Bearer <token>` header, supplied here by the caller
/// because Rust has no `google-auth-library`).
pub fn build_request_headers(client: &GoogleVertexClient) -> IndexMap<String, String> {
	let mut headers: IndexMap<String, String> = IndexMap::new();
	headers.insert(USER_AGENT_HEADER.to_string(), LIBRARY_LABEL.to_string());
	headers.insert(GOOGLE_API_CLIENT_HEADER.to_string(), LIBRARY_LABEL.to_string());
	headers.insert(CONTENT_TYPE_HEADER.to_string(), "application/json".to_string());
	for (key, value) in &client.headers {
		if let Some(value) = value {
			headers.insert(key.clone(), value.clone());
		}
	}
	if let Some(api_key) = &client.api_key {
		if !headers.keys().any(|key| key.eq_ignore_ascii_case(GOOGLE_API_KEY_HEADER)) {
			headers.insert(GOOGLE_API_KEY_HEADER.to_string(), api_key.clone());
		}
	} else {
		// NOTE: the TypeScript delegates to `google-auth-library`, which resolves
		// Application Default Credentials and sets `Authorization: Bearer <token>`.
		// Rust has no ADC implementation in this slice, so an explicit
		// `Authorization` header from model/options headers is used as-is and
		// `GOOGLE_VERTEX_ACCESS_TOKEN` is accepted as the ADC token source.
		if !headers.keys().any(|key| key.eq_ignore_ascii_case("authorization")) {
			if let Ok(token) = std::env::var("GOOGLE_VERTEX_ACCESS_TOKEN") {
				if !token.is_empty() {
					headers.insert("Authorization".to_string(), format!("Bearer {}", token));
				}
			}
		}
	}
	headers
}

/// `client.models.generateContentStream(params)`.
async fn send_request(
	client: &GoogleVertexClient,
	params: &Value,
	model: &Model,
	options: &GoogleVertexOptions,
) -> Result<reqwest::Response, GoogleVertexStreamError> {
	let path = format!(
		"{}:streamGenerateContent?alt=sse",
		t_model(&model.id).map_err(GoogleVertexStreamError::Message)?
	);
	let url = if should_prepend_project_location(client) {
		format!("{}/{}/{}", request_url(client), base_resource_path(client), path)
	} else {
		format!("{}/{}", request_url(client), path)
	};
	let body = generate_content_parameters_to_vertex(params);

	let mut headers = reqwest::header::HeaderMap::new();
	for (key, value) in build_request_headers(client) {
		if let (Ok(name), Ok(header_value)) = (
			reqwest::header::HeaderName::from_bytes(key.as_bytes()),
			reqwest::header::HeaderValue::from_str(&value),
		) {
			headers.insert(name, header_value);
		}
	}

	let mut request = reqwest::Client::new().post(&url).headers(headers).json(&body);
	if let Some(timeout_ms) = options.stream.timeout_ms {
		request = request.timeout(std::time::Duration::from_millis(timeout_ms.max(0.0) as u64));
	}

	let send = request.send();
	let response = match options.stream.signal.as_ref() {
		Some(signal) => tokio::select! {
			_ = signal.cancelled() => return Err(GoogleVertexStreamError::Message("Request was aborted".to_string())),
			result = send => result,
		},
		None => send.await,
	};
	let response = response.map_err(|error| GoogleVertexStreamError::Message(error.to_string()))?;
	throw_error_if_not_ok(response).await
}

/// TS: `throwErrorIfNotOK(response)`.
async fn throw_error_if_not_ok(response: reqwest::Response) -> Result<reqwest::Response, GoogleVertexStreamError> {
	if response.status().is_success() {
		return Ok(response);
	}
	let status = response.status().as_u16() as i64;
	let status_text = response.status().canonical_reason().unwrap_or("").to_string();
	let is_json = response
		.headers()
		.get(reqwest::header::CONTENT_TYPE)
		.and_then(|value| value.to_str().ok())
		.map(|value| value.contains("application/json"))
		.unwrap_or(false);
	let body_text = response.text().await.unwrap_or_default();
	let error_body = if is_json {
		serde_json::from_str::<Value>(&body_text).unwrap_or(Value::Null)
	} else {
		let mut error = Map::new();
		error.insert("message".to_string(), Value::String(body_text));
		error.insert("code".to_string(), Value::Number(status.into()));
		error.insert("status".to_string(), Value::String(status_text));
		let mut root = Map::new();
		root.insert("error".to_string(), Value::Object(error));
		Value::Object(root)
	};
	let error_message = serde_json::to_string(&error_body).unwrap_or_else(|_| "null".to_string());
	if (400..600).contains(&status) {
		let mut value = Map::new();
		value.insert("name".to_string(), Value::String("ApiError".to_string()));
		value.insert("message".to_string(), Value::String(error_message.clone()));
		value.insert("status".to_string(), Value::Number(status.into()));
		return Err(GoogleVertexStreamError::Api {
			message: error_message,
			status,
			value: Value::Object(value),
		});
	}
	Err(GoogleVertexStreamError::Message(error_message))
}

/// TS: `generateContentParametersToVertex(apiClient, params)`.
fn generate_content_parameters_to_vertex(params: &Value) -> Value {
	let mut body = Map::new();
	if let Some(contents) = params.get("contents") {
		body.insert("contents".to_string(), contents.clone());
	}
	let config = params.get("config").cloned().unwrap_or(Value::Object(Map::new()));
	generate_content_config_to_vertex(&config, &mut body);
	Value::Object(body)
}

/// TS: `generateContentConfigToVertex(apiClient, config, parentObject)`.
fn generate_content_config_to_vertex(config: &Value, body: &mut Map<String, Value>) {
	let mut generation_config = Map::new();
	if let Some(system_instruction) = config.get("systemInstruction").and_then(Value::as_str) {
		body.insert(
			"systemInstruction".to_string(),
			content_to_vertex(&system_instruction_value(system_instruction)),
		);
	}
	if let Some(temperature) = config.get("temperature") {
		if !temperature.is_null() {
			generation_config.insert("temperature".to_string(), temperature.clone());
		}
	}
	if let Some(max_output_tokens) = config.get("maxOutputTokens") {
		if !max_output_tokens.is_null() {
			generation_config.insert("maxOutputTokens".to_string(), max_output_tokens.clone());
		}
	}
	if let Some(tools) = config.get("tools") {
		if !tools.is_null() {
			body.insert("tools".to_string(), tools.clone());
		}
	}
	if let Some(tool_config) = config.get("toolConfig") {
		if !tool_config.is_null() {
			// `toolConfigToVertex` copies `functionCallingConfig` unchanged.
			body.insert("toolConfig".to_string(), tool_config.clone());
		}
	}
	if let Some(thinking_config) = config.get("thinkingConfig") {
		if !thinking_config.is_null() {
			generation_config.insert("thinkingConfig".to_string(), thinking_config.clone());
		}
	}
	body.insert("generationConfig".to_string(), Value::Object(generation_config));
}

/// TS: `contentToVertex({ parts: [{ text }] })`.
fn system_instruction_value(text: &str) -> Value {
	let mut part = Map::new();
	part.insert("text".to_string(), Value::String(text.to_string()));
	let mut content = Map::new();
	content.insert("parts".to_string(), Value::Array(vec![Value::Object(part)]));
	Value::Object(content)
}

/// TS: `contentToVertex(content)`.
fn content_to_vertex(content: &Value) -> Value {
	let mut result = Map::new();
	if let Some(parts) = content.get("parts").and_then(Value::as_array) {
		result.insert(
			"parts".to_string(),
			Value::Array(parts.iter().map(part_to_vertex).collect()),
		);
	}
	if let Some(role) = content.get("role") {
		if !role.is_null() {
			result.insert("role".to_string(), role.clone());
		}
	}
	Value::Object(result)
}

/// TS: `partToVertex(part)`.
fn part_to_vertex(part: &Value) -> Value {
	let mut result = Map::new();
	if let Some(file_data) = part.get("fileData") {
		if !file_data.is_null() {
			result.insert("fileData".to_string(), file_data.clone());
		}
	}
	if let Some(function_call) = part.get("functionCall") {
		if !function_call.is_null() {
			result.insert("functionCall".to_string(), function_call_to_vertex(function_call));
		}
	}
	if let Some(function_response) = part.get("functionResponse") {
		if !function_response.is_null() {
			result.insert("functionResponse".to_string(), function_response.clone());
		}
	}
	if let Some(inline_data) = part.get("inlineData") {
		if !inline_data.is_null() {
			result.insert("inlineData".to_string(), blob_to_vertex(inline_data));
		}
	}
	if let Some(text) = part.get("text") {
		if !text.is_null() {
			result.insert("text".to_string(), text.clone());
		}
	}
	if let Some(thought) = part.get("thought") {
		if !thought.is_null() {
			result.insert("thought".to_string(), thought.clone());
		}
	}
	if let Some(thought_signature) = part.get("thoughtSignature") {
		if !thought_signature.is_null() {
			result.insert("thoughtSignature".to_string(), thought_signature.clone());
		}
	}
	Value::Object(result)
}

/// TS: `functionCallToVertex({ id, args, name })`.
fn function_call_to_vertex(function_call: &Value) -> Value {
	let mut result = Map::new();
	if let Some(id) = function_call.get("id") {
		if !id.is_null() {
			result.insert("id".to_string(), id.clone());
		}
	}
	if let Some(args) = function_call.get("args") {
		if !args.is_null() {
			result.insert("args".to_string(), args.clone());
		}
	}
	if let Some(name) = function_call.get("name") {
		if !name.is_null() {
			result.insert("name".to_string(), name.clone());
		}
	}
	Value::Object(result)
}

/// TS: `blobToVertex({ data, mimeType })`.
fn blob_to_vertex(blob: &Value) -> Value {
	let mut result = Map::new();
	if let Some(data) = blob.get("data") {
		if !data.is_null() {
			result.insert("data".to_string(), data.clone());
		}
	}
	if let Some(mime_type) = blob.get("mimeType") {
		if !mime_type.is_null() {
			result.insert("mimeType".to_string(), mime_type.clone());
		}
	}
	Value::Object(result)
}

// ---------------------------------------------------------------------------
// SSE transport (`ApiClient.processStreamResponse`)
// ---------------------------------------------------------------------------

/// `@google/genai` dist index.mjs:13233 - the SDK throws this exact message when
/// the response body ends with an unterminated event.
const INCOMPLETE_JSON_SEGMENT_MESSAGE: &str = "Incomplete JSON segment at the end";

/// TS: the SDK `processStreamResponse` inline error probe (dist index.mjs:13238-13260).
///
/// Vertex shares `ApiClient.processStreamResponse` with the Gemini path, so each raw
/// (not yet split) *chunk* string is parsed as JSON; when the parsed object has an
/// `error` member whose `code` is 4xx/5xx the SDK throws an `ApiError` with message
/// ``got status: ${status}. ${JSON.stringify(chunkJson)}`` and `status = error.code`.
/// google-vertex.ts:103 awaits `client.models.generateContentStream`, so the throw
/// reaches the provider's catch instead of completing an empty/partial success.
fn throw_if_inline_error_chunk(chunk_string: &str) -> Result<(), GoogleVertexStreamError> {
	let Ok(chunk_json) = serde_json::from_str::<Value>(chunk_string) else {
		// The SDK only rethrows `ApiError`; every other parse failure is ignored.
		return Ok(());
	};
	let Some(error) = chunk_json.get("error") else {
		return Ok(());
	};
	// `JSON.parse(JSON.stringify(chunkJson['error']))` - a non-object error member
	// has no `code`/`status`, so the SDK's `if (code >= 400 && code < 600)` is false.
	let Some(error_object) = error.as_object() else {
		return Ok(());
	};
	let status = error_object.get("status");
	let code_value = error_object.get("code");
	if let Some(code) = javascript_number(code_value) {
		if (400.0..600.0).contains(&code) {
			let status_text = match status {
				Some(Value::String(text)) => text.clone(),
				Some(other) => other.to_string(),
				None => "undefined".to_string(),
			};
			let message = format!("got status: {}. {}", status_text, chunk_json);
			// `new ApiError({ message, status: code })` - `status` keeps the raw `error.code`
			// value, like the SDK's `this.status = options.status`.
			let mut value = Map::new();
			value.insert("name".to_string(), Value::String("ApiError".to_string()));
			value.insert("message".to_string(), Value::String(message.clone()));
			value.insert(
				"status".to_string(),
				code_value.cloned().unwrap_or(Value::Number((code as i64).into())),
			);
			return Err(GoogleVertexStreamError::Api {
				message,
				status: code as i64,
				value: Value::Object(value),
			});
		}
	}
	Ok(())
}

/// TS: `code >= 400 && code < 600` on the parsed `error.code`.
///
/// JavaScript coerces operands, so `"400"` triggers the same throw as `400` (the probed
/// SDK keeps the raw value on the `ApiError` too: `status="400"`). `undefined`, `null`,
/// an object or a non-numeric string compares false; only the number and numeric-string
/// shapes the Google APIs emit are modelled here.
fn javascript_number(value: Option<&Value>) -> Option<f64> {
	match value {
		Some(Value::Number(number)) => number.as_f64(),
		Some(Value::String(text)) => text.trim().parse::<f64>().ok(),
		_ => None,
	}
}

/// Local SSE reader for `alt=sse` streaming.
struct SseChunkStream {
	chunks: std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
	buffer: String,
	pending: Vec<Value>,
	byte_pending: Vec<u8>,
	finished: bool,
	signal: Option<tokio_util::sync::CancellationToken>,
}

impl SseChunkStream {
	fn new(response: reqwest::Response, signal: Option<tokio_util::sync::CancellationToken>) -> Self {
		Self {
			chunks: Box::pin(response.bytes_stream()),
			buffer: String::new(),
			pending: Vec::new(),
			byte_pending: Vec::new(),
			finished: false,
			signal,
		}
	}

	/// TS: the `for await (const chunk of googleStream)` loop body.
	async fn next(&mut self) -> Result<Option<Value>, GoogleVertexStreamError> {
		loop {
			if !self.pending.is_empty() {
				return Ok(Some(self.pending.remove(0)));
			}
			if self.finished {
				// `@google/genai` dist index.mjs:13231-13235 - when the reader reports `done` a
				// non-empty (trimmed) buffer is a truncated final event and the SDK throws
				// "Incomplete JSON segment at the end"; the decoder is never flushed (there is no
				// `decoder.decode()` call before `reader.releaseLock()`), so its pending partial
				// bytes are dropped instead of being appended to the buffer.
				self.byte_pending.clear();
				if self.buffer.trim().is_empty() {
					return Ok(None);
				}
				return Err(GoogleVertexStreamError::Message(
					INCOMPLETE_JSON_SEGMENT_MESSAGE.to_string(),
				));
			}
			if let Some(signal) = &self.signal {
				if signal.is_cancelled() {
					return Err(GoogleVertexStreamError::Message("Request was aborted".to_string()));
				}
			}
			match self.chunks.next().await {
				None => {
					self.finished = true;
				}
				Some(Err(error)) => return Err(GoogleVertexStreamError::Message(error.to_string())),
				Some(Ok(bytes)) => {
					self.byte_pending.extend_from_slice(&bytes);
					// TS: `const chunkString = decoder.decode(value, { stream: true })` - the
					// decode is stateful, so a multi-byte character split across two network
					// chunks survives instead of turning into U+FFFD.
					let chunk_string = self.decode_pending_bytes();
					throw_if_inline_error_chunk(&chunk_string)?;
					self.buffer.push_str(&chunk_string);
					self.drain_events()?;
				}
			}
		}
	}

	/// TS: `decoder.decode(value, { stream: true })` - a stateful streaming UTF-8 decode.
	///
	/// Bytes that form an incomplete multi-byte sequence stay in `byte_pending` until
	/// the next chunk arrives; a genuinely invalid sequence becomes U+FFFD like the
	/// WHATWG decoder (`String::from_utf8_lossy` per chunk would instead corrupt every
	/// character that straddles a chunk boundary). Same shape as
	/// `anthropic.rs::decode_utf8_stream`.
	fn decode_pending_bytes(&mut self) -> String {
		let mut text = String::new();
		loop {
			match std::str::from_utf8(&self.byte_pending) {
				Ok(valid) => {
					text.push_str(valid);
					self.byte_pending.clear();
					return text;
				}
				Err(error) => {
					let valid_up_to = error.valid_up_to();
					text.push_str(&String::from_utf8_lossy(&self.byte_pending[..valid_up_to]));
					self.byte_pending.drain(..valid_up_to);
					match error.error_len() {
						// Incomplete trailing sequence: wait for more bytes.
						None => return text,
						Some(error_length) => {
							self.byte_pending.drain(..error_length);
							text.push('\u{FFFD}');
						}
					}
				}
			}
		}
	}

	/// TS: the delimiter scan inside `processStreamResponse`.
	fn drain_events(&mut self) -> Result<(), GoogleVertexStreamError> {
		const DELIMITERS: [&str; 3] = ["\n\n", "\r\r", "\r\n\r\n"];
		loop {
			let mut delimiter_index: Option<usize> = None;
			let mut delimiter_length = 0usize;
			for delimiter in DELIMITERS {
				if let Some(index) = self.buffer.find(delimiter) {
					if delimiter_index.map_or(true, |current| index < current) {
						delimiter_index = Some(index);
						delimiter_length = delimiter.len();
					}
				}
			}
			let Some(delimiter_index) = delimiter_index else {
				return Ok(());
			};
			let event_string = self.buffer[..delimiter_index].to_string();
			self.buffer = self.buffer[delimiter_index + delimiter_length..].to_string();
			let trimmed_event = event_string.trim();
			if trimmed_event.starts_with("data:") {
				let processed = trimmed_event["data:".len()..].trim();
				match serde_json::from_str::<Value>(processed) {
					Ok(value) => self.pending.push(value),
					Err(error) => {
						return Err(GoogleVertexStreamError::Message(format!(
							"exception parsing stream chunk {}. {}",
							processed, error
						)))
					}
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

/// TS: `buildParams(model, context, options = {})`.
pub fn build_params(
	model: &Model,
	context: &Context,
	options: Option<&GoogleVertexOptions>,
) -> Result<Value, GoogleVertexStreamError> {
	let contents = convert_messages(model, context).map_err(GoogleVertexStreamError::Message)?;
	let options = options.cloned().unwrap_or_default();

	let mut generation_config = Map::new();
	if let Some(temperature) = options.stream.temperature {
		generation_config.insert(
			"temperature".to_string(),
			serde_json::Number::from_f64(temperature)
				.map(Value::Number)
				.unwrap_or(Value::Null),
		);
	}
	if let Some(max_tokens) = options.stream.max_tokens {
		generation_config.insert(
			"maxOutputTokens".to_string(),
			serde_json::Number::from_f64(max_tokens)
				.map(Value::Number)
				.unwrap_or(Value::Null),
		);
	}

	let mut config = Map::new();
	if !generation_config.is_empty() {
		for (key, value) in &generation_config {
			config.insert(key.clone(), value.clone());
		}
	}
	// TS: `...(context.systemPrompt && { systemInstruction })` - an empty string
	// is falsy in JS, so it must be omitted from the request.
	if let Some(system_prompt) = context.system_prompt.as_ref().filter(|prompt| !prompt.is_empty()) {
		config.insert(
			"systemInstruction".to_string(),
			Value::String(sanitize_surrogates(system_prompt)),
		);
	}
	let tools = context.tools.as_ref().filter(|tools| !tools.is_empty());
	if let Some(tools) = tools {
		if let Some(converted) = convert_tools(tools, false) {
			config.insert("tools".to_string(), Value::Array(converted));
		}
	}

	if tools.is_some() && options.tool_choice.is_some() {
		let mut function_calling_config = Map::new();
		function_calling_config.insert(
			"mode".to_string(),
			Value::String(map_tool_choice(options.tool_choice.as_deref().unwrap_or_default())),
		);
		let mut tool_config = Map::new();
		tool_config.insert(
			"functionCallingConfig".to_string(),
			Value::Object(function_calling_config),
		);
		config.insert("toolConfig".to_string(), Value::Object(tool_config));
	} else {
		config.insert("toolConfig".to_string(), Value::Null);
	}

	let thinking_enabled = options.thinking.as_ref().map(|thinking| thinking.enabled).unwrap_or(false);
	if thinking_enabled && model.reasoning {
		let thinking = options.thinking.clone().unwrap_or_default();
		let mut thinking_config = Map::new();
		thinking_config.insert("includeThoughts".to_string(), Value::Bool(true));
		if let Some(level) = thinking.level {
			thinking_config.insert(
				"thinkingLevel".to_string(),
				Value::String(thinking_level_map(&level).to_string()),
			);
		} else if let Some(budget_tokens) = thinking.budget_tokens {
			thinking_config.insert(
				"thinkingBudget".to_string(),
				serde_json::Number::from_f64(budget_tokens)
					.map(Value::Number)
					.unwrap_or(Value::Null),
			);
		}
		config.insert("thinkingConfig".to_string(), Value::Object(thinking_config));
	} else if model.reasoning && options.thinking.is_some() && !thinking_enabled {
		config.insert("thinkingConfig".to_string(), get_disabled_thinking_config(model));
	}

	if let Some(signal) = &options.stream.signal {
		if signal.is_cancelled() {
			return Err(GoogleVertexStreamError::Message("Request aborted".to_string()));
		}
		// `config.abortSignal` is not serialized; the port keeps the token on the request.
	}

	let mut params = Map::new();
	params.insert("model".to_string(), Value::String(model.id.clone()));
	params.insert("contents".to_string(), Value::Array(contents));
	params.insert("config".to_string(), Value::Object(config));
	Ok(Value::Object(params))
}

/// TS: `type ClampedThinkingLevel = Exclude<PiThinkingLevel, "xhigh" | "max">`.
pub type ClampedThinkingLevel = String;

fn is_gemma4_model(model: &Model) -> bool {
	let id = model.id.to_lowercase();
	id.contains("gemma-4") || id.contains("gemma4")
}

/// TS: `isGemini3ProModel(model)` - `/gemini-3(?:\.\d+)?-pro/`.
fn is_gemini3_pro_model(model: &Model) -> bool {
	matches_gemini3(&model.id.to_lowercase(), "pro")
}

/// TS: `isGemini3FlashModel(model)` - `/gemini-3(?:\.\d+)?-flash/`.
fn is_gemini3_flash_model(model: &Model) -> bool {
	matches_gemini3(&model.id.to_lowercase(), "flash")
}

fn matches_gemini3(id: &str, suffix: &str) -> bool {
	let Some(index) = id.find("gemini-3") else {
		return false;
	};
	let rest = &id[index + "gemini-3".len()..];
	let rest = match rest.strip_prefix('.') {
		Some(rest) => {
			let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
			if digits.is_empty() {
				return false;
			}
			&rest[digits.len()..]
		}
		None => rest,
	};
	rest.starts_with(&format!("-{}", suffix))
}

/// TS: `getDisabledThinkingConfig(model)`.
fn get_disabled_thinking_config(model: &Model) -> Value {
	// Google docs: Gemini 3.1 Pro cannot disable thinking, and Gemini 3 Flash / Flash-Lite
	// do not support full thinking-off either. For Gemini 3 models, use the lowest supported
	// thinkingLevel without includeThoughts so hidden thinking remains invisible to pi.
	let mut config = Map::new();
	if is_gemini3_pro_model(model) {
		config.insert("thinkingLevel".to_string(), Value::String("LOW".to_string()));
		return Value::Object(config);
	}
	if is_gemini3_flash_model(model) || is_gemma4_model(model) {
		config.insert("thinkingLevel".to_string(), Value::String("MINIMAL".to_string()));
		return Value::Object(config);
	}

	// Gemini 2.x supports disabling via thinkingBudget = 0.
	config.insert("thinkingBudget".to_string(), Value::Number(0.into()));
	Value::Object(config)
}

/// Provider thinking levels for Gemini 3 and Gemma 4.
fn get_thinking_level(effort: &ClampedThinkingLevel, model: &Model) -> GoogleThinkingLevel {
	if is_gemini3_pro_model(model) {
		match effort.as_str() {
			"minimal" | "low" => return "LOW".to_string(),
			"medium" | "high" => return "HIGH".to_string(),
			_ => {}
		}
	}
	if is_gemma4_model(model) {
		return match effort.as_str() {
			"minimal" | "low" => "MINIMAL".to_string(),
			"medium" | "high" => "HIGH".to_string(),
			_ => "THINKING_LEVEL_UNSPECIFIED".to_string(),
		};
	}
	match effort.as_str() {
		"minimal" => "MINIMAL".to_string(),
		"low" => "LOW".to_string(),
		"medium" => "MEDIUM".to_string(),
		"high" => "HIGH".to_string(),
		_ => "THINKING_LEVEL_UNSPECIFIED".to_string(),
	}
}

/// TS: `streamSimpleGoogleVertex: StreamFunction<"google-vertex", SimpleStreamOptions>`.
pub fn stream_simple_google_vertex(
	model: &Model,
	context: &Context,
	options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
	let base = build_base_options(model, options.as_ref(), None);
	let reasoning = options.as_ref().and_then(|options| options.reasoning.clone());
	if reasoning.is_none() || reasoning.as_deref() == Some("off") {
		let mut typed = GoogleVertexOptions::from_base(&base);
		typed.thinking = Some(GoogleVertexThinkingOptions {
			enabled: false,
			..Default::default()
		});
		return stream_google_vertex(model, context, Some(typed));
	}

	let clamped_reasoning = clamp_thinking_level(model, reasoning.as_deref().unwrap_or_default());
	let effort: ClampedThinkingLevel = if clamped_reasoning == "off" {
		"high".to_string()
	} else {
		clamped_reasoning
	};

	if is_gemini3_pro_model(model) || is_gemini3_flash_model(model) || is_gemma4_model(model) {
		let mut typed = GoogleVertexOptions::from_base(&base);
		typed.thinking = Some(GoogleVertexThinkingOptions {
			enabled: true,
			budget_tokens: None,
			level: Some(get_thinking_level(&effort, model)),
		});
		return stream_google_vertex(model, context, Some(typed));
	}

	let mut typed = GoogleVertexOptions::from_base(&base);
	typed.thinking = Some(GoogleVertexThinkingOptions {
		enabled: true,
		budget_tokens: Some(get_google_thinking_budget(
			&model.id,
			&effort,
			options.as_ref().and_then(|options| options.thinking_budgets.as_ref()),
		)),
		level: None,
	});
	stream_google_vertex(model, context, Some(typed))
}

/// Keeps `StopReason`, `API_VERSION` and the SDK version referenced like the TS module.
pub fn _markers(reason: &StopReason) -> &str {
	let _ = GOOGLE_GENAI_SDK_VERSION;
	let _ = API_VERSION;
	reason
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{InputModality, Message, ModelCost, Tool, UserContent, UserMessage};
	use serde_json::json;

	fn model(id: &str) -> Model {
		let mut model = Model::new(id, id, "google-vertex", "google-vertex", "");
		model.input = vec![InputModality::Text, InputModality::Image];
		model.reasoning = true;
		model.cost = ModelCost::zero();
		model
	}

	fn base_options() -> StreamOptions {
		StreamOptions::default()
	}

	#[test]
	fn options_round_trip_through_serde() {
		let mut options = GoogleVertexOptions::from_base(&base_options());
		options.tool_choice = Some("none".to_string());
		options.project = Some("proj".to_string());
		options.location = Some("us-central1".to_string());
		options.thinking = Some(GoogleVertexThinkingOptions {
			enabled: true,
			budget_tokens: Some(1024.0),
			level: None,
		});
		let value = serde_json::to_value(&options).unwrap();
		assert_eq!(value["toolChoice"], json!("none"));
		assert_eq!(value["project"], json!("proj"));
		assert_eq!(value["location"], json!("us-central1"));
		let back: GoogleVertexOptions = serde_json::from_value(value).unwrap();
		assert_eq!(back.project.as_deref(), Some("proj"));
		assert_eq!(back.thinking.unwrap().budget_tokens, Some(1024.0));
	}

	#[test]
	fn client_derives_regional_base_url() {
		let model = model("gemini-3-pro");
		let client = create_client(&model, "proj", "europe-west4", None).unwrap();
		assert_eq!(client.base_url, "https://europe-west4-aiplatform.googleapis.com/");
		assert_eq!(client.api_version, "v1");
		assert_eq!(request_url(&client), "https://europe-west4-aiplatform.googleapis.com/v1");
		assert_eq!(base_resource_path(&client), "projects/proj/locations/europe-west4");
	}

	#[test]
	fn client_uses_multi_regional_and_global_endpoints() {
		let model = model("gemini-3-pro");
		let client = create_client(&model, "proj", "us", None).unwrap();
		assert_eq!(client.base_url, "https://aiplatform.us.rep.googleapis.com/");
		let client = create_client(&model, "proj", "global", None).unwrap();
		assert_eq!(client.base_url, "https://aiplatform.googleapis.com/");
	}

	#[test]
	fn client_with_api_key_uses_aiplatform_and_drops_project_and_location() {
		let model = model("gemini-3-pro");
		let client = create_client_with_api_key(&model, "key", None).unwrap();
		assert_eq!(client.base_url, "https://aiplatform.googleapis.com/");
		assert_eq!(client.project, None);
		assert_eq!(client.location, None);
		assert_eq!(client.api_key.as_deref(), Some("key"));
		assert_eq!(client.api_version, "v1");
		assert!(!should_prepend_project_location(&client));
	}

	#[test]
	fn custom_base_url_sets_collection_scope_and_skips_api_version() {
		let mut model = model("gemini-3-pro");
		model.base_url = "https://proxy.example.com/v1".to_string();
		let client = create_client(&model, "proj", "us-central1", None).unwrap();
		assert_eq!(client.base_url, "https://proxy.example.com/v1");
		assert!(client.custom_base_url_collection);
		assert_eq!(client.api_version, "");
		assert!(!should_prepend_project_location(&client));
	}

	#[test]
	fn custom_base_url_without_version_keeps_default_api_version() {
		let mut model = model("gemini-3-pro");
		model.base_url = "https://proxy.example.com".to_string();
		let client = create_client(&model, "proj", "us-central1", None).unwrap();
		assert_eq!(client.api_version, "v1");
		assert_eq!(request_url(&client), "https://proxy.example.com/v1");
	}

	#[test]
	fn location_placeholder_base_url_is_ignored() {
		let mut model = model("gemini-3-pro");
		model.base_url = "https://{location}-aiplatform.googleapis.com".to_string();
		let client = create_client(&model, "proj", "us-central1", None).unwrap();
		assert_eq!(client.base_url, "https://us-central1-aiplatform.googleapis.com/");
		assert!(!client.custom_base_url_collection);
	}

	#[test]
	fn resolve_api_key_rejects_placeholders_and_markers() {
		let mut env = crate::test_env::ScopedEnv::new();
		let mut options = GoogleVertexOptions::from_base(&base_options());
		options.stream.api_key = Some("  real-key  ".to_string());
		assert_eq!(resolve_api_key(Some(&options)).as_deref(), Some("real-key"));

		options.stream.api_key = Some(GCP_VERTEX_CREDENTIALS_MARKER.to_string());
		assert_eq!(resolve_api_key(Some(&options)), None);

		options.stream.api_key = Some("<authenticated>".to_string());
		assert_eq!(resolve_api_key(Some(&options)), None);

		options.stream.api_key = Some("   ".to_string());
		env.remove("GOOGLE_CLOUD_API_KEY");
		assert_eq!(resolve_api_key(Some(&options)), None);
	}

	#[test]
	fn resolve_project_and_location_error_messages_match() {
		let mut env = crate::test_env::ScopedEnv::new();
		env.remove("GOOGLE_CLOUD_PROJECT");
		env.remove("GCLOUD_PROJECT");
		env.remove("GOOGLE_CLOUD_LOCATION");
		let error = resolve_project(None).unwrap_err();
		assert_eq!(
			error.message(),
			"Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
		);
		let error = resolve_location(None).unwrap_err();
		assert_eq!(
			error.message(),
			"Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options."
		);

		env.set("GCLOUD_PROJECT", "from-env");
		assert_eq!(resolve_project(None).unwrap(), "from-env");
		env.remove("GCLOUD_PROJECT");
		env.set("GOOGLE_CLOUD_LOCATION", "us-central1");
		assert_eq!(resolve_location(None).unwrap(), "us-central1");
		env.remove("GOOGLE_CLOUD_LOCATION");
	}

	#[test]
	fn t_model_prefixes_vertex_publishers() {
		assert_eq!(t_model("gemini-3-pro").unwrap(), "publishers/google/models/gemini-3-pro");
		assert_eq!(t_model("meta/llama-3").unwrap(), "publishers/meta/models/llama-3");
		assert_eq!(t_model("publishers/google/models/x").unwrap(), "publishers/google/models/x");
		assert_eq!(t_model("projects/p/locations/l/models/m").unwrap(), "projects/p/locations/l/models/m");
		assert_eq!(t_model("models/m").unwrap(), "models/m");
	}

	#[test]
	fn api_version_segment_matching_follows_the_typescript_regex() {
		assert!(is_api_version_segment("v1"));
		assert!(is_api_version_segment("v1beta"));
		assert!(is_api_version_segment("v1beta2"));
		assert!(!is_api_version_segment("v"));
		assert!(!is_api_version_segment("v1betaX"));
		assert!(base_url_includes_api_version("https://x.example.com/v1"));
		assert!(base_url_includes_api_version("https://x.example.com/v1beta1/"));
		assert!(!base_url_includes_api_version("https://x.example.com/foo"));
	}

	#[test]
	fn request_headers_use_api_key_or_bearer_token() {
		let mut env = crate::test_env::ScopedEnv::new();
		let model = model("gemini-3-pro");
		let client = create_client_with_api_key(&model, "secret", None).unwrap();
		let headers = build_request_headers(&client);
		assert_eq!(headers.get("User-Agent").map(String::as_str), Some("google-genai-sdk/1.52.0"));
		assert_eq!(headers.get("x-goog-api-client").map(String::as_str), Some("google-genai-sdk/1.52.0"));
		assert_eq!(headers.get("Content-Type").map(String::as_str), Some("application/json"));
		assert_eq!(headers.get("x-goog-api-key").map(String::as_str), Some("secret"));
		assert!(headers.get("Authorization").is_none());

		env.set("GOOGLE_VERTEX_ACCESS_TOKEN", "adc-token");
		let client = create_client(&model, "proj", "us-central1", None).unwrap();
		let headers = build_request_headers(&client);
		assert_eq!(headers.get("Authorization").map(String::as_str), Some("Bearer adc-token"));
		assert!(headers.get("x-goog-api-key").is_none());
		env.remove("GOOGLE_VERTEX_ACCESS_TOKEN");
	}

	#[test]
	fn build_params_uses_the_thinking_level_map() {
		let model = model("gemini-3-pro");
		let context = Context::new(None, vec![], None);
		let mut options = GoogleVertexOptions::from_base(&base_options());
		options.thinking = Some(GoogleVertexThinkingOptions {
			enabled: true,
			budget_tokens: None,
			level: Some("MEDIUM".to_string()),
		});
		let params = build_params(&model, &context, Some(&options)).unwrap();
		assert_eq!(
			params["config"]["thinkingConfig"],
			json!({"includeThoughts": true, "thinkingLevel": "MEDIUM"})
		);
	}

	#[test]
	fn build_params_matches_vertex_wire_shape() {
		let model = model("gemini-3-pro");
		let context = Context::new(
			Some("be nice".to_string()),
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			Some(vec![Tool {
				name: "read".to_string(),
				description: "Read".to_string(),
				parameters: json!({"type": "object"}),
			}]),
		);
		let mut options = GoogleVertexOptions::from_base(&base_options());
		options.stream.temperature = Some(0.25);
		options.stream.max_tokens = Some(64.0);
		options.tool_choice = Some("any".to_string());
		options.thinking = Some(GoogleVertexThinkingOptions {
			enabled: true,
			budget_tokens: Some(-1.0),
			level: None,
		});
		let params = build_params(&model, &context, Some(&options)).unwrap();
		assert_eq!(
			params,
			json!({
				"model": "gemini-3-pro",
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
				"config": {
					"temperature": 0.25,
					"maxOutputTokens": 64.0,
					"systemInstruction": "be nice",
					"tools": [{"functionDeclarations": [{"name": "read", "description": "Read", "parametersJsonSchema": {"type": "object"}}]}],
					"toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
					"thinkingConfig": {"includeThoughts": true, "thinkingBudget": -1.0}
				}
			})
		);
	}

	#[test]
	fn build_params_clears_tool_config_and_uses_disabled_thinking() {
		let model = model("gemini-2.5-flash");
		let context = Context::new(None, vec![], None);
		let mut options = GoogleVertexOptions::from_base(&base_options());
		options.thinking = Some(GoogleVertexThinkingOptions {
			enabled: false,
			..Default::default()
		});
		let params = build_params(&model, &context, Some(&options)).unwrap();
		assert_eq!(params["config"]["toolConfig"], json!(null));
		assert_eq!(params["config"]["thinkingConfig"], json!({"thinkingBudget": 0}));

		let gemini_three = self::model("gemini-3-flash");
		let params = build_params(&gemini_three, &context, Some(&options)).unwrap();
		assert_eq!(params["config"]["thinkingConfig"], json!({"thinkingLevel": "MINIMAL"}));

		let gemini_pro = self::model("gemini-3.1-pro");
		let params = build_params(&gemini_pro, &context, Some(&options)).unwrap();
		assert_eq!(params["config"]["thinkingConfig"], json!({"thinkingLevel": "LOW"}));
	}

	#[test]
	fn build_params_errors_when_signal_already_aborted() {
		let model = model("gemini-3-pro");
		let context = Context::new(None, vec![], None);
		let mut options = GoogleVertexOptions::from_base(&base_options());
		let token = tokio_util::sync::CancellationToken::new();
		token.cancel();
		options.stream.signal = Some(token);
		let error = build_params(&model, &context, Some(&options)).unwrap_err();
		assert_eq!(error.message(), "Request aborted");
	}

	#[test]
	fn thinking_level_map_covers_all_enum_values() {
		assert_eq!(thinking_level_map(&"THINKING_LEVEL_UNSPECIFIED".to_string()), "THINKING_LEVEL_UNSPECIFIED");
		assert_eq!(thinking_level_map(&"MINIMAL".to_string()), "MINIMAL");
		assert_eq!(thinking_level_map(&"LOW".to_string()), "LOW");
		assert_eq!(thinking_level_map(&"MEDIUM".to_string()), "MEDIUM");
		assert_eq!(thinking_level_map(&"HIGH".to_string()), "HIGH");
	}

	#[test]
	fn gemini3_thinking_levels_follow_the_typescript_switch() {
		assert_eq!(get_thinking_level(&"minimal".to_string(), &model("gemini-3-pro")), "LOW");
		assert_eq!(get_thinking_level(&"low".to_string(), &model("gemini-3.2-pro")), "LOW");
		assert_eq!(get_thinking_level(&"medium".to_string(), &model("gemini-3-pro")), "HIGH");
		assert_eq!(get_thinking_level(&"high".to_string(), &model("gemini-3-pro")), "HIGH");
		assert_eq!(get_thinking_level(&"minimal".to_string(), &model("gemini-3-flash")), "MINIMAL");
		assert_eq!(get_thinking_level(&"medium".to_string(), &model("gemini-3-flash")), "MEDIUM");
		assert_eq!(get_thinking_level(&"high".to_string(), &model("gemini-3-flash")), "HIGH");
	}

	#[test]
	fn generate_content_parameters_to_vertex_keeps_sdk_field_order() {
		let params = json!({
			"model": "gemini-3-pro",
			"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
			"config": {
				"systemInstruction": "sys",
				"temperature": 0.1,
				"tools": [{"functionDeclarations": []}],
				"toolConfig": {"functionCallingConfig": {"mode": "AUTO"}}
			}
		});
		let body = generate_content_parameters_to_vertex(&params);
		assert_eq!(
			body,
			json!({
				"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
				"systemInstruction": {"parts": [{"text": "sys"}]},
				"tools": [{"functionDeclarations": []}],
				"toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
				"generationConfig": {"temperature": 0.1}
			})
		);
		let keys: Vec<&String> = body.as_object().unwrap().keys().collect();
		assert_eq!(keys, vec!["contents", "systemInstruction", "tools", "toolConfig", "generationConfig"]);
	}

	#[test]
	fn sse_chunk_stream_splits_and_reports_malformed_events() {
		let mut stream = SseChunkStream {
			chunks: Box::pin(futures::stream::empty()),
			buffer: "data: {\"a\": 1}\n\ndata: {\"b\": 2}\r\r".to_string(),
			pending: Vec::new(),
			byte_pending: Vec::new(),
			finished: true,
			signal: None,
		};
		stream.drain_events().unwrap();
		assert_eq!(stream.pending, vec![json!({"a": 1}), json!({"b": 2})]);

		let mut broken = SseChunkStream {
			chunks: Box::pin(futures::stream::empty()),
			buffer: "data: nope\n\n".to_string(),
			pending: Vec::new(),
			byte_pending: Vec::new(),
			finished: true,
			signal: None,
		};
		assert!(broken.drain_events().unwrap_err().message().starts_with("exception parsing stream chunk nope."));
	}

	fn byte_stream(chunks: Vec<&[u8]>) -> SseChunkStream {
		let items: Vec<reqwest::Result<bytes::Bytes>> = chunks
			.into_iter()
			.map(|chunk| Ok(bytes::Bytes::copy_from_slice(chunk)))
			.collect();
		SseChunkStream {
			chunks: Box::pin(futures::stream::iter(items)),
			buffer: String::new(),
			pending: Vec::new(),
			byte_pending: Vec::new(),
			finished: false,
			signal: None,
		}
	}

	/// GM-03: `@google/genai` dist index.mjs:13231-13235 throws
	/// "Incomplete JSON segment at the end" for an unterminated trailing event.
	#[tokio::test]
	async fn sse_chunk_stream_reports_an_unterminated_trailing_event() {
		let mut stream = byte_stream(vec![b"data: {\"candidates\":[]}"]);
		let error = stream.next().await.unwrap_err();
		assert_eq!(error.message(), "Incomplete JSON segment at the end");
		assert_eq!(
			format_stream_failure_message(&error.as_thrown()),
			"Incomplete JSON segment at the end"
		);

		// A stream that ends exactly on a delimiter is still a clean end.
		let mut clean = byte_stream(vec![b"data: {\"candidates\":[]}\n\n"]);
		assert!(clean.next().await.unwrap().is_some());
		assert!(clean.next().await.unwrap().is_none());
	}

	/// GM-02: the SDK's inline error probe (dist index.mjs:13238-13260) is shared with
	/// the Gemini path; a raw JSON error chunk throws an `ApiError`.
	#[tokio::test]
	async fn sse_chunk_stream_surfaces_inline_json_error_chunks() {
		let json = br#"{"error":{"code":403,"message":"denied","status":"PERMISSION_DENIED"}}"#;
		let mut stream = byte_stream(vec![json]);
		let error = stream.next().await.unwrap_err();
		let GoogleVertexStreamError::Api { message, status, value } = &error else {
			panic!("expected ApiError shape, got {error:?}")
		};
		assert_eq!(*status, 403);
		assert_eq!(
			*message,
			"got status: PERMISSION_DENIED. {\"error\":{\"code\":403,\"message\":\"denied\",\"status\":\"PERMISSION_DENIED\"}}"
		);
		assert_eq!(value["name"], json!("ApiError"));
		assert_eq!(value["status"], json!(403));

		// An error member without a 4xx/5xx `code` is not thrown by the SDK probe: it stays
		// in the buffer as an unterminated event (the probed SDK threw
		// "Incomplete JSON segment at the end" for this exact body).
		let no_code = br#"{"error":{"message":"no code"}}"#;
		let mut stream = byte_stream(vec![no_code]);
		assert_eq!(
			stream.next().await.unwrap_err().message(),
			"Incomplete JSON segment at the end"
		);
		assert!(throw_if_inline_error_chunk(r#"{"error":{"message":"no code"}}"#).is_ok());
		assert!(throw_if_inline_error_chunk(r#"{"candidates":[]}"#).is_ok());
		assert!(throw_if_inline_error_chunk("data: {").is_ok());
	}

	/// GM-05 (Vertex): the TS throws from `streamGoogleVertex` for missing credentials,
	/// so the port must report through the stream. `stream_simple_google_vertex` has no
	/// panic on this path; a panic here would hang this test until its timeout.
	#[tokio::test]
	async fn stream_simple_google_vertex_reports_missing_configuration_through_the_stream() {
		let mut env = crate::test_env::ScopedEnv::new();
		if std::env::var("GOOGLE_CLOUD_API_KEY").is_ok() {
			env.remove("GOOGLE_CLOUD_API_KEY");
		}
		env.remove("GOOGLE_CLOUD_PROJECT");
		env.remove("GCLOUD_PROJECT");
		env.remove("GOOGLE_CLOUD_LOCATION");
		let model = model("gemini-3-pro");
		let context = Context::new(None, vec![], None);
		let stream = stream_simple_google_vertex(&model, &context, None);
		let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
			.await
			.unwrap()
			.unwrap();
		let AssistantMessageEvent::Error { reason, error } = event else {
			panic!("expected provider error event")
		};
		assert_eq!(reason, "error");
		assert_eq!(
			error.error_message.as_deref(),
			Some("Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options.")
		);
		assert!(stream.is_done());
		assert!(stream.next().await.is_none());
	}

	/// GM-04: `decoder.decode(value, { stream: true })` keeps a multi-byte character
	/// that straddles a network chunk boundary intact.
	#[tokio::test]
	async fn sse_chunk_stream_keeps_multibyte_characters_split_across_chunks() {
		// 'é' is C3 A9 and '🎈' is F0 9F 8E 88; one split lands inside each character.
		let event = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"caf\u{e9} \u{1F388}\"}]}}]}\n\n";
		let bytes = event.as_bytes();
		let first_split = event.find('\u{e9}').unwrap() + 1;
		let second_split = event.find('\u{1F388}').unwrap() + 2;
		let mut stream = byte_stream(vec![
			&bytes[..first_split],
			&bytes[first_split..second_split],
			&bytes[second_split..],
		]);
		let chunk = stream.next().await.unwrap().unwrap();
		assert_eq!(chunk["candidates"][0]["content"]["parts"][0]["text"], json!("caf\u{e9} \u{1F388}"));
		assert!(stream.next().await.unwrap().is_none());
	}


}


#[cfg(test)]
mod t15_controls_tests {
	//! T15 owner 'controls': E-04 empty-string falsiness contract for google-vertex
	//! (TS `google.ts` guards `...(context.systemPrompt && ...)` and
	//! `!providedId || ...` treat `""` as absent).

	use super::*;
	use crate::types::{Message, ModelCost, UserContent, UserMessage};
	use serde_json::json;

	fn t15_model() -> Model {
		let mut model = Model::new("gemini-2.5-pro", "gemini-2.5-pro", "google-vertex", "google", "");
		model.reasoning = true;
		model.cost = ModelCost::zero();
		model
	}

	fn t15_context(system_prompt: Option<&str>) -> Context {
		Context::new(
			system_prompt.map(|prompt| prompt.to_string()),
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			None,
		)
	}

	/// (a) `Some("")` must omit `systemInstruction` exactly like the TS spread
	/// guard; `Some("rules")` must still be sent; `None` must stay absent.
	#[test]
	fn t15_empty_optional_values_match_contract_system_prompt() {
		let model = t15_model();
		let empty = build_params(&model, &t15_context(Some("")), None).unwrap();
		assert_eq!(
			empty["config"].get("systemInstruction"),
			None,
			"TS spread guard omits empty prompts (context.systemPrompt && systemInstruction)"
		);
		let set = build_params(&model, &t15_context(Some("rules")), None).unwrap();
		assert_eq!(set["config"]["systemInstruction"], json!("rules"));
		let none = build_params(&model, &t15_context(None), None).unwrap();
		assert_eq!(none["config"].get("systemInstruction"), None);
	}

	/// (c) A provided toolcall id of `Some("")` must be regenerated like the TS
	/// `!providedId` guard; a truthy id must still be reused.
	#[test]
	fn t15_empty_optional_values_match_contract_tool_call_id() {
		let model = t15_model();
		let stream = create_assistant_message_event_stream();

		let mut output = AssistantMessage::new(model.api.clone(), model.provider.clone(), model.id.clone(), 0);
		let mut current_block: Option<CurrentBlock> = None;
		let empty_id = json!({"functionCall": {"name": "read", "id": "", "args": {}}});
		process_part(&empty_id, &mut output, &stream, &mut current_block).unwrap();
		let tool_call = output
			.content
			.iter()
			.find_map(|block| match block {
				ContentBlock::ToolCall(tool_call) => Some(tool_call.clone()),
				_ => None,
			})
			.expect("a toolcall block must be produced");
		assert!(!tool_call.id.is_empty(), "TS `!providedId` regenerates ids for empty strings");
		assert!(tool_call.id.starts_with("read_"), "id follows the TS {{name}}_{{ms}}_{{counter}} shape; got {:?}", tool_call.id);

		let mut output2 = AssistantMessage::new(model.api.clone(), model.provider.clone(), model.id.clone(), 0);
		let mut current_block2: Option<CurrentBlock> = None;
		let provided = json!({"functionCall": {"name": "read", "id": "call-1", "args": {}}});
		process_part(&provided, &mut output2, &stream, &mut current_block2).unwrap();
		let reused = output2
			.content
			.iter()
			.find_map(|block| match block {
				ContentBlock::ToolCall(tool_call) => Some(tool_call.clone()),
				_ => None,
			})
			.expect("a toolcall block must be produced");
		assert_eq!(reused.id, "call-1", "TS reuses a truthy provided id");
	}
}
