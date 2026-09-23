//! Port of packages/ai/src/providers/openai-completions.ts
//!
//! NOTE (SDK gap): the TypeScript uses the `openai` SDK client. The Rust port sends the
//! same JSON body with `reqwest` and parses the SSE stream locally. The SDK's own request
//! headers (X-Stainless-*, OpenAI-Organization, ...) are not part of this port's contract;
//! every header the TypeScript sets explicitly (model.headers, copilot, prime team id,
//! session affinity, options.headers, Authorization, cf-aig-authorization) is preserved.

use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::cache_pricing::{get_anthropic_cache_write_cost, has_standard_anthropic_cache_pricing};
use crate::env_api_keys::{get_env_api_key, get_prime_team_id};
use crate::models::{calculate_cost, clamp_thinking_level, CostOverrides};
use crate::providers::cloudflare::{is_cloudflare_provider, resolve_cloudflare_base_url};
use crate::providers::github_copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input, CopilotDynamicHeaderParams};
use crate::providers::opencode_headers::with_opencode_headers;
use crate::providers::simple_options::build_base_options;
use crate::providers::transform_messages::try_transform_messages;
use crate::types::{
	AssistantMessage, AssistantMessageEvent, CacheRetention, ContentBlock, Context, ImageOrTextContent, InputModality,
	Message, Model, SimpleStreamOptions, StreamOptions, TextContent, ThinkingContent, Tool, ToolCall, UserContent,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::stream_failure::{record_stream_failure, ThrownStreamError};

/// JavaScript truthiness for the `if (x)` guards the TypeScript uses on raw JSON.
fn js_truthy(value: &Value) -> bool {
	match value {
		Value::Null => false,
		Value::Bool(value) => *value,
		Value::Number(number) => number.as_f64().map(|value| value != 0.0).unwrap_or(true),
		Value::String(value) => !value.is_empty(),
		Value::Array(_) | Value::Object(_) => true,
	}
}

/// TS: `hasToolHistory(messages)`
fn has_tool_history(messages: &[Message]) -> bool {
	for msg in messages {
		if let Message::ToolResult(_) = msg {
			return true;
		}
		if let Message::Assistant(assistant) = msg {
			if assistant.content.iter().any(|block| matches!(block, ContentBlock::ToolCall(_))) {
				return true;
			}
		}
	}
	false
}

const REASONING_DETAILS_SIGNATURE_TYPE: &str = "openai-completions.reasoning_details.v1";

/// TS: `encodeReasoningDetails(details)`
fn encode_reasoning_details(details: &[Map<String, Value>]) -> String {
	let mut object = Map::new();
	object.insert(
		"type".to_string(),
		Value::String(REASONING_DETAILS_SIGNATURE_TYPE.to_string()),
	);
	object.insert(
		"details".to_string(),
		Value::Array(details.iter().cloned().map(Value::Object).collect()),
	);
	Value::Object(object).to_string()
}

/// TS: `decodeReasoningDetails(signature)`
fn decode_reasoning_details(signature: Option<&str>) -> Option<Vec<Map<String, Value>>> {
	let signature = signature?;
	if !signature.starts_with('{') {
		return None;
	}
	let parsed: Value = serde_json::from_str(signature).ok()?;
	let object = parsed.as_object()?;
	if object.get("type").and_then(Value::as_str) != Some(REASONING_DETAILS_SIGNATURE_TYPE) {
		return None;
	}
	let details = object.get("details")?.as_array()?;
	let mut result: Vec<Map<String, Value>> = Vec::new();
	for detail in details {
		let detail = detail.as_object()?;
		result.push(detail.clone());
	}
	Some(result)
}

/// TS: `interface OpenAICompletionsOptions extends StreamOptions`
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct OpenAICompletionsOptions {
	#[serde(flatten)]
	pub stream: StreamOptions,
	/// `"auto" | "none" | "required" | { type: "function"; function: { name } }`
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_choice: Option<Value>,
	/// `"minimal" | "low" | "medium" | "high" | "xhigh" | "max"`
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning_effort: Option<String>,
	/// Explicit reasoning toggle. None preserves the provider/model default.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning_enabled: Option<bool>,
}

impl OpenAICompletionsOptions {
	/// TS: the caller passes `StreamOptions & Record<string, unknown>`; this keeps the
	/// non-serializable fields (signal, on_payload, on_response, on_usage_observation).
	pub fn from_base(base: &StreamOptions) -> Self {
		Self {
			stream: base.clone(),
			tool_choice: None,
			reasoning_effort: None,
			reasoning_enabled: None,
		}
	}
}

/// TS: `interface OpenAICompatCacheControl`
#[derive(Debug, Clone, PartialEq)]
struct OpenAICompatCacheControl {
	ttl: Option<String>,
}

impl OpenAICompatCacheControl {
	/// `{ type: "ephemeral", ...(ttl ? { ttl } : {}) }`
	fn to_value(&self) -> Value {
		let mut object = Map::new();
		object.insert("type".to_string(), Value::String("ephemeral".to_string()));
		if let Some(ttl) = &self.ttl {
			object.insert("ttl".to_string(), Value::String(ttl.clone()));
		}
		Value::Object(object)
	}
}

/// TS: `ResolvedOpenAICompletionsCompat`
///
/// `open_router_routing` is part of the resolved object in the TypeScript; the
/// payload builder reads `model.compat.openRouterRouting` directly, exactly like
/// the TypeScript, so the resolved copy is only carried for parity.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct ResolvedCompat {
	supports_store: bool,
	supports_developer_role: bool,
	supports_reasoning_effort: bool,
	supports_usage_in_streaming: bool,
	max_tokens_field: String,
	requires_tool_result_name: bool,
	requires_assistant_after_tool_result: bool,
	requires_thinking_as_text: bool,
	requires_reasoning_content_on_assistant_messages: bool,
	thinking_format: String,
	open_router_routing: Value,
	vercel_gateway_routing: Value,
	zai_tool_stream: bool,
	supports_strict_mode: bool,
	cache_control_format: Option<String>,
	send_session_affinity_headers: bool,
	supports_long_cache_retention: bool,
}

/// TS: `resolveCacheRetention(cacheRetention?)`
fn resolve_cache_retention(cache_retention: Option<&CacheRetention>) -> CacheRetention {
	if let Some(cache_retention) = cache_retention {
		if !cache_retention.is_empty() {
			return cache_retention.clone();
		}
	}
	if std::env::var("PI_CACHE_RETENTION").map(|value| value == "long").unwrap_or(false) {
		return "long".to_string();
	}
	"short".to_string()
}

/// TS: `getCompatCacheControl(compat, cacheRetention)`
fn get_compat_cache_control(compat: &ResolvedCompat, cache_retention: &CacheRetention) -> Option<OpenAICompatCacheControl> {
	if compat.cache_control_format.as_deref() != Some("anthropic") || cache_retention == "none" {
		return None;
	}

	let ttl = if cache_retention == "long" && compat.supports_long_cache_retention {
		Some("1h".to_string())
	} else {
		None
	};
	Some(OpenAICompatCacheControl { ttl })
}

// ---------------------------------------------------------------------------
// detectCompat / getCompat
// ---------------------------------------------------------------------------

/// TS: `detectCompat(model)`
fn detect_compat(model: &Model) -> ResolvedCompat {
	let provider = model.provider.as_str();
	let base_url = model.base_url.as_str();

	let is_zai = provider == "zai" || base_url.contains("api.z.ai");
	let is_moonshot = provider == "moonshotai" || provider == "moonshotai-cn" || base_url.contains("api.moonshot.");
	let is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" || base_url.contains("api.cloudflare.com");
	let is_cloudflare_ai_gateway =
		provider == "cloudflare-ai-gateway" || base_url.contains("gateway.ai.cloudflare.com");
	let is_prime_inference = provider == "prime-inference" || base_url.contains("api.pinference.ai");

	let is_non_standard = provider == "cerebras"
		|| base_url.contains("cerebras.ai")
		|| provider == "xai"
		|| base_url.contains("api.x.ai")
		|| base_url.contains("chutes.ai")
		|| base_url.contains("deepseek.com")
		|| is_zai
		|| is_moonshot
		|| provider == "opencode"
		|| base_url.contains("opencode.ai")
		|| is_cloudflare_workers_ai
		|| is_cloudflare_ai_gateway
		|| is_prime_inference;

	let use_max_tokens =
		base_url.contains("chutes.ai") || is_moonshot || is_cloudflare_ai_gateway || is_prime_inference;

	let is_grok = provider == "xai" || base_url.contains("api.x.ai");
	let is_deep_seek = provider == "deepseek" || base_url.contains("deepseek.com");
	let is_anthropic_model = model.id.starts_with("anthropic/");
	let cache_control_format = if is_anthropic_model && (provider == "openrouter" || is_prime_inference) {
		Some("anthropic".to_string())
	} else {
		None
	};

	ResolvedCompat {
		supports_store: !is_non_standard,
		supports_developer_role: !is_non_standard,
		supports_reasoning_effort: !is_grok && !is_zai && !is_moonshot && !is_cloudflare_ai_gateway,
		supports_usage_in_streaming: true,
		max_tokens_field: if use_max_tokens { "max_tokens" } else { "max_completion_tokens" }.to_string(),
		requires_tool_result_name: false,
		requires_assistant_after_tool_result: false,
		requires_thinking_as_text: false,
		requires_reasoning_content_on_assistant_messages: is_deep_seek,
		thinking_format: if is_deep_seek {
			"deepseek"
		} else if is_zai {
			"zai"
		} else if provider == "openrouter" || base_url.contains("openrouter.ai") {
			"openrouter"
		} else {
			"openai"
		}
		.to_string(),
		open_router_routing: Value::Object(Map::new()),
		vercel_gateway_routing: Value::Object(Map::new()),
		zai_tool_stream: false,
		supports_strict_mode: !is_moonshot && !is_cloudflare_ai_gateway && !is_prime_inference,
		cache_control_format,
		send_session_affinity_headers: false,
		supports_long_cache_retention: !(is_cloudflare_workers_ai || is_cloudflare_ai_gateway),
	}
}

/// TS: `getCompat(model)`
fn get_compat(model: &Model) -> ResolvedCompat {
	let detected = detect_compat(model);
	let Some(compat) = model.compat_completions() else {
		return detected;
	};

	ResolvedCompat {
		supports_store: compat.supports_store.unwrap_or(detected.supports_store),
		supports_developer_role: compat.supports_developer_role.unwrap_or(detected.supports_developer_role),
		supports_reasoning_effort: compat.supports_reasoning_effort.unwrap_or(detected.supports_reasoning_effort),
		supports_usage_in_streaming: compat
			.supports_usage_in_streaming
			.unwrap_or(detected.supports_usage_in_streaming),
		max_tokens_field: compat.max_tokens_field.clone().unwrap_or(detected.max_tokens_field),
		requires_tool_result_name: compat.requires_tool_result_name.unwrap_or(detected.requires_tool_result_name),
		requires_assistant_after_tool_result: compat
			.requires_assistant_after_tool_result
			.unwrap_or(detected.requires_assistant_after_tool_result),
		requires_thinking_as_text: compat.requires_thinking_as_text.unwrap_or(detected.requires_thinking_as_text),
		requires_reasoning_content_on_assistant_messages: compat
			.requires_reasoning_content_on_assistant_messages
			.unwrap_or(detected.requires_reasoning_content_on_assistant_messages),
		thinking_format: compat.thinking_format.clone().unwrap_or(detected.thinking_format),
		open_router_routing: Value::Object(Map::new()),
		vercel_gateway_routing: compat
			.vercel_gateway_routing
			.as_ref()
			.map(|routing| serde_json::to_value(routing).unwrap_or(Value::Object(Map::new())))
			.unwrap_or(detected.vercel_gateway_routing),
		zai_tool_stream: compat.zai_tool_stream.unwrap_or(detected.zai_tool_stream),
		supports_strict_mode: compat.supports_strict_mode.unwrap_or(detected.supports_strict_mode),
		cache_control_format: compat
			.cache_control_format
			.clone()
			.or(detected.cache_control_format),
		send_session_affinity_headers: compat
			.send_session_affinity_headers
			.unwrap_or(detected.send_session_affinity_headers),
		supports_long_cache_retention: compat
			.supports_long_cache_retention
			.unwrap_or(detected.supports_long_cache_retention),
	}
}

/// `model.thinkingLevelMap?.[level]` with `null` preserved: None = key absent.
fn thinking_level_mapped(model: &Model, level: &str) -> Option<Option<String>> {
	model.thinking_level_map.as_ref().and_then(|map| map.get(level).cloned())
}

// ---------------------------------------------------------------------------
// createClient
// ---------------------------------------------------------------------------

/// TS: `createClient(...)` result - the pieces the request needs.
#[derive(Debug)]
pub(crate) struct OpenAIClient {
	pub api_key: String,
	pub base_url: String,
	/// `withOpenCodeHeaders(provider, conversationId, defaultHeaders)` - `null` values
	/// stay distinguishable from absent ones (Cloudflare clears Authorization).
	pub default_headers: IndexMap<String, Option<String>>,
}

/// TS: `createClient(model, context, apiKey?, optionsHeaders?, cacheSessionId?, compat?, conversationId?)`
fn create_client(
	model: &Model,
	context: &Context,
	api_key: Option<&str>,
	options_headers: Option<&IndexMap<String, String>>,
	cache_session_id: Option<&str>,
	compat: &ResolvedCompat,
	conversation_id: Option<&str>,
) -> Result<OpenAIClient, String> {
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

	let mut headers: IndexMap<String, Option<String>> = IndexMap::new();
	if let Some(model_headers) = &model.headers {
		for (name, value) in model_headers {
			headers.insert(name.clone(), Some(value.clone()));
		}
	}
	if model.provider == "github-copilot" {
		let has_images = has_copilot_vision_input(&context.messages);
		let copilot_headers = build_copilot_dynamic_headers(CopilotDynamicHeaderParams {
			messages: &context.messages,
			has_images,
		});
		for (name, value) in copilot_headers {
			headers.insert(name, Some(value));
		}
	}

	if model.provider == "prime-inference" {
		if let Some(team_id) = get_prime_team_id() {
			headers.insert("X-Prime-Team-ID".to_string(), Some(team_id));
		}
	}

	if let Some(cache_session_id) = cache_session_id {
		if compat.send_session_affinity_headers {
			headers.insert("session_id".to_string(), Some(cache_session_id.to_string()));
			headers.insert("x-client-request-id".to_string(), Some(cache_session_id.to_string()));
			headers.insert("x-session-affinity".to_string(), Some(cache_session_id.to_string()));
		}
	}

	if let Some(options_headers) = options_headers {
		for (name, value) in options_headers {
			headers.insert(name.clone(), Some(value.clone()));
		}
	}

	let default_headers = if model.provider == "cloudflare-ai-gateway" {
		let mut merged = headers.clone();
		let authorization = headers.get("Authorization").cloned().unwrap_or(None);
		merged.insert("Authorization".to_string(), authorization);
		merged.insert(
			"cf-aig-authorization".to_string(),
			Some(format!("Bearer {api_key}")),
		);
		merged
	} else {
		headers
	};

	let base_url = if is_cloudflare_provider(&model.provider) {
		resolve_cloudflare_base_url(model)?
	} else {
		model.base_url.clone()
	};

	Ok(OpenAIClient {
		api_key,
		base_url,
		default_headers: with_opencode_headers(&model.provider, conversation_id, &default_headers),
	})
}

// ---------------------------------------------------------------------------
// Anthropic-style cache control markers
// ---------------------------------------------------------------------------

/// TS: `applyAnthropicCacheControl(messages, tools, cacheControl)`
fn apply_anthropic_cache_control(
	messages: &mut [Value],
	tools: Option<&mut Vec<Value>>,
	cache_control: &OpenAICompatCacheControl,
) {
	add_cache_control_to_system_prompt(messages, cache_control);
	if let Some(tools) = tools {
		add_cache_control_to_last_tool(tools, cache_control);
	}
	add_cache_control_to_last_conversation_message(messages, cache_control);
}

/// TS: `addCacheControlToSystemPrompt(messages, cacheControl)`
fn add_cache_control_to_system_prompt(messages: &mut [Value], cache_control: &OpenAICompatCacheControl) {
	for message in messages.iter_mut() {
		let role = message.get("role").and_then(Value::as_str).unwrap_or_default().to_string();
		if role == "system" || role == "developer" {
			add_cache_control_to_instruction_message(message, cache_control);
			return;
		}
	}
}

/// TS: `addCacheControlToLastConversationMessage(messages, cacheControl)`
fn add_cache_control_to_last_conversation_message(messages: &mut [Value], cache_control: &OpenAICompatCacheControl) {
	for message in messages.iter_mut().rev() {
		let role = message.get("role").and_then(Value::as_str).unwrap_or_default().to_string();
		if role == "user" || role == "assistant" || role == "tool" {
			if add_cache_control_to_message(message, cache_control) {
				return;
			}
		}
	}
}

/// TS: `addCacheControlToLastTool(tools, cacheControl)`
fn add_cache_control_to_last_tool(tools: &mut [Value], cache_control: &OpenAICompatCacheControl) {
	if tools.is_empty() {
		return;
	}

	let last_tool = tools.last_mut().expect("non-empty tools");
	if let Some(object) = last_tool.as_object_mut() {
		object.insert("cache_control".to_string(), cache_control.to_value());
	}
}

/// TS: `addCacheControlToInstructionMessage(message, cacheControl)`
fn add_cache_control_to_instruction_message(
	message: &mut Value,
	cache_control: &OpenAICompatCacheControl,
) -> bool {
	add_cache_control_to_text_content(message, cache_control)
}

/// TS: `addCacheControlToMessage(message, cacheControl)`
fn add_cache_control_to_message(message: &mut Value, cache_control: &OpenAICompatCacheControl) -> bool {
	let role = message.get("role").and_then(Value::as_str).unwrap_or_default();
	if role == "user" || role == "assistant" || role == "tool" {
		return add_cache_control_to_text_content(message, cache_control);
	}
	false
}

/// TS: `addCacheControlToTextContent(message, cacheControl)`
fn add_cache_control_to_text_content(message: &mut Value, cache_control: &OpenAICompatCacheControl) -> bool {
	let Some(object) = message.as_object_mut() else {
		return false;
	};
	let content = object.get("content").cloned();
	match content {
		Some(Value::String(text)) => {
			if text.is_empty() {
				return false;
			}
			let mut part = Map::new();
			part.insert("type".to_string(), Value::String("text".to_string()));
			part.insert("text".to_string(), Value::String(text));
			part.insert("cache_control".to_string(), cache_control.to_value());
			object.insert("content".to_string(), Value::Array(vec![Value::Object(part)]));
			true
		}
		Some(Value::Array(mut parts)) => {
			for part in parts.iter_mut().rev() {
				let is_text = part.get("type").and_then(Value::as_str) == Some("text");
				if is_text {
					if let Some(part) = part.as_object_mut() {
						part.insert("cache_control".to_string(), cache_control.to_value());
					}
					object.insert("content".to_string(), Value::Array(parts));
					return true;
				}
			}
			false
		}
		_ => false,
	}
}

// ---------------------------------------------------------------------------
// convertMessages
// ---------------------------------------------------------------------------

/// TS: `convertMessages(model, context, compat)`
pub(crate) fn convert_messages(model: &Model, context: &Context, compat: &ResolvedCompat) -> Result<Vec<Value>, String> {
	let mut params: Vec<Value> = Vec::new();

	let normalize_tool_call_id = |id: &str, model: &Model, _source: &AssistantMessage| -> String {
		// Handle pipe-separated IDs from OpenAI Responses API
		// Format: {call_id}|{id} where {id} can be 400+ chars with special chars (+, /, =)
		// These come from providers like github-copilot, openai-codex, opencode
		// Extract just the call_id part and normalize it
		if id.contains('|') {
			let call_id = id.split('|').next().unwrap_or_default();
			// Sanitize to allowed chars and truncate to 40 chars (OpenAI limit)
			let sanitized: String = call_id
				.chars()
				.map(|ch| {
					if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
						ch
					} else {
						'_'
					}
				})
				.collect();
			return sanitized.chars().take(40).collect();
		}

		if model.provider == "openai" {
			return if id.chars().count() > 40 {
				id.chars().take(40).collect()
			} else {
				id.to_string()
			};
		}
		id.to_string()
	};

	let transformed_messages = try_transform_messages(
		context.messages.clone(),
		model,
		Some(&normalize_tool_call_id),
	)?;

	if let Some(system_prompt) = &context.system_prompt {
		if !system_prompt.is_empty() {
			let use_developer_role = model.reasoning && compat.supports_developer_role;
			let role = if use_developer_role { "developer" } else { "system" };
			let mut message = Map::new();
			message.insert("role".to_string(), Value::String(role.to_string()));
			message.insert(
				"content".to_string(),
				Value::String(sanitize_surrogates(system_prompt)),
			);
			params.push(Value::Object(message));
		}
	}

	let mut last_role: Option<String> = None;

	let mut i = 0usize;
	while i < transformed_messages.len() {
		let msg = &transformed_messages[i];
		// Some providers don't allow user messages directly after tool results
		// Insert a synthetic assistant message to bridge the gap
		if compat.requires_assistant_after_tool_result && last_role.as_deref() == Some("toolResult") && msg.role() == "user" {
			let mut bridge = Map::new();
			bridge.insert("role".to_string(), Value::String("assistant".to_string()));
			bridge.insert(
				"content".to_string(),
				Value::String("I have processed the tool results.".to_string()),
			);
			params.push(Value::Object(bridge));
		}

		match msg {
			Message::User(user) => {
				match &user.content {
					UserContent::Text(text) => {
						let mut message = Map::new();
						message.insert("role".to_string(), Value::String("user".to_string()));
						message.insert("content".to_string(), Value::String(sanitize_surrogates(text)));
						params.push(Value::Object(message));
					}
					UserContent::Blocks(blocks) => {
						let mut content: Vec<Value> = Vec::new();
						for item in blocks {
							match item {
								ImageOrTextContent::Text(text) => {
									let mut part = Map::new();
									part.insert("type".to_string(), Value::String("text".to_string()));
									part.insert("text".to_string(), Value::String(sanitize_surrogates(&text.text)));
									content.push(Value::Object(part));
								}
								ImageOrTextContent::Image(image) => {
									let mut url = Map::new();
									url.insert(
										"url".to_string(),
										Value::String(format!("data:{};base64,{}", image.mime_type, image.data)),
									);
									let mut part = Map::new();
									part.insert("type".to_string(), Value::String("image_url".to_string()));
									part.insert("image_url".to_string(), Value::Object(url));
									content.push(Value::Object(part));
								}
							}
						}
						if content.is_empty() {
							// The TS `for` loop's `i++` still runs on `continue`.
							i += 1;
							continue;
						}
						let mut message = Map::new();
						message.insert("role".to_string(), Value::String("user".to_string()));
						message.insert("content".to_string(), Value::Array(content));
						params.push(Value::Object(message));
					}
				}
				last_role = Some(msg.role().to_string());
			}
			Message::Assistant(assistant) => {
				// Some providers don't accept null content, use empty string instead
				let mut assistant_msg = Map::new();
				assistant_msg.insert("role".to_string(), Value::String("assistant".to_string()));
				assistant_msg.insert(
					"content".to_string(),
					if compat.requires_assistant_after_tool_result {
						Value::String(String::new())
					} else {
						Value::Null
					},
				);

				let assistant_text_parts: Vec<Value> = assistant
					.content
					.iter()
					.filter_map(|block| block.as_text())
					.filter(|block| !block.text.trim().is_empty())
					.map(|block| {
						let mut part = Map::new();
						part.insert("type".to_string(), Value::String("text".to_string()));
						part.insert("text".to_string(), Value::String(sanitize_surrogates(&block.text)));
						Value::Object(part)
					})
					.collect();
				let assistant_text: String = assistant_text_parts
					.iter()
					.filter_map(|part| part.get("text").and_then(Value::as_str))
					.collect::<Vec<_>>()
					.join("");

				let replay_reasoning_details: Vec<Map<String, Value>> = assistant
					.content
					.iter()
					.filter_map(|block| block.as_thinking())
					.flat_map(|block| decode_reasoning_details(block.thinking_signature.as_deref()).unwrap_or_default())
					.collect();
				if !replay_reasoning_details.is_empty() {
					assistant_msg.insert(
						"reasoning_details".to_string(),
						Value::Array(replay_reasoning_details.iter().cloned().map(Value::Object).collect()),
					);
				}

				let non_empty_thinking_blocks: Vec<&ThinkingContent> = assistant
					.content
					.iter()
					.filter_map(|block| block.as_thinking())
					.filter(|block| decode_reasoning_details(block.thinking_signature.as_deref()).is_none())
					.filter(|block| !block.thinking.trim().is_empty())
					.collect();
				if !non_empty_thinking_blocks.is_empty() {
					if compat.requires_thinking_as_text {
						// Convert thinking blocks to plain text (no tags to avoid model mimicking them)
						let thinking_text = non_empty_thinking_blocks
							.iter()
							.map(|block| sanitize_surrogates(&block.thinking))
							.collect::<Vec<_>>()
							.join("\n\n");
						let mut parts: Vec<Value> = Vec::new();
						let mut part = Map::new();
						part.insert("type".to_string(), Value::String("text".to_string()));
						part.insert("text".to_string(), Value::String(thinking_text));
						parts.push(Value::Object(part));
						parts.extend(assistant_text_parts.iter().cloned());
						assistant_msg.insert("content".to_string(), Value::Array(parts));
					} else {
						// Always send assistant content as a plain string (OpenAI Chat Completions
						// API standard format). Sending as an array of {type:"text", text:"..."}
						// objects is non-standard and causes some models (e.g. DeepSeek V3.2 via
						// NVIDIA NIM) to mirror the content-block structure literally in their
						// output, producing recursive nesting like [{'type':'text','text':'[{...}]'}].
						if !assistant_text.is_empty() {
							assistant_msg.insert("content".to_string(), Value::String(assistant_text.clone()));
						}

						// thinkingSignature holds the field the provider streamed reasoning in
						// (reasoning_content / reasoning / reasoning_text), not a crypto signature.
						// Prefer reasoning_content when the provider requires it (otherwise the
						// reasoning_content="" default below would clobber the trace); else round-trip
						// into the recorded field; with neither, keep the trace as text rather than
						// inventing an unsupported field.
						let reasoning_text = non_empty_thinking_blocks
							.iter()
							.map(|block| sanitize_surrogates(&block.thinking))
							.collect::<Vec<_>>()
							.join("\n");
						let reasoning_field = if compat.requires_reasoning_content_on_assistant_messages {
							Some("reasoning_content".to_string())
						} else {
							non_empty_thinking_blocks[0]
								.thinking_signature
								.clone()
								.filter(|signature| !signature.is_empty())
						};
						match reasoning_field {
							Some(reasoning_field) => {
								assistant_msg.insert(reasoning_field, Value::String(reasoning_text));
							}
							None => {
								assistant_msg.insert(
									"content".to_string(),
									Value::String(if !assistant_text.is_empty() {
										format!("{reasoning_text}\n\n{assistant_text}")
									} else {
										reasoning_text
									}),
								);
							}
						}
					}
				} else if !assistant_text.is_empty() {
					// Always send assistant content as a plain string (OpenAI Chat Completions
					// API standard format). Sending as an array of {type:"text", text:"..."}
					// objects is non-standard and causes some models (e.g. DeepSeek V3.2 via
					// NVIDIA NIM) to mirror the content-block structure literally in their
					// output, producing recursive nesting like [{'type':'text','text':'[{...}]'}].
					assistant_msg.insert("content".to_string(), Value::String(assistant_text.clone()));
				}

				let tool_calls: Vec<&ToolCall> = assistant
					.content
					.iter()
					.filter_map(|block| block.as_tool_call())
					.collect();
				if !tool_calls.is_empty() {
					let serialized_tool_calls: Vec<Value> = tool_calls
						.iter()
						.map(|tc| {
							let mut function = Map::new();
							function.insert("name".to_string(), Value::String(tc.name.clone()));
							function.insert(
								"arguments".to_string(),
								Value::String(Value::Object(tc.arguments.clone()).to_string()),
							);
							let mut call = Map::new();
							call.insert("id".to_string(), Value::String(tc.id.clone()));
							call.insert("type".to_string(), Value::String("function".to_string()));
							call.insert("function".to_string(), Value::Object(function));
							Value::Object(call)
						})
						.collect();
					assistant_msg.insert("tool_calls".to_string(), Value::Array(serialized_tool_calls));

					let reasoning_details: Vec<Value> = tool_calls
						.iter()
						.filter_map(|tc| tc.thought_signature.as_ref())
						.filter_map(|signature| serde_json::from_str::<Value>(signature).ok())
						.collect();
					if !reasoning_details.is_empty() && replay_reasoning_details.is_empty() {
						assistant_msg.insert("reasoning_details".to_string(), Value::Array(reasoning_details));
					}
				}
				if compat.requires_reasoning_content_on_assistant_messages
					&& model.reasoning
					&& assistant_msg.get("reasoning_content").is_none()
				{
					assistant_msg.insert("reasoning_content".to_string(), Value::String(String::new()));
				}
				if !replay_reasoning_details.is_empty()
					&& assistant_msg.get("content").map(Value::is_null).unwrap_or(false)
					&& assistant_msg.get("tool_calls").is_none()
				{
					assistant_msg.insert("content".to_string(), Value::String(String::new()));
				}
				// Skip assistant messages that have no content and no tool calls.
				// Some providers require "either content or tool_calls, but not none".
				// Other providers also don't accept empty assistant messages.
				// This handles aborted assistant responses that got no content.
				let content = assistant_msg.get("content");
				let has_content = match content {
					Some(Value::Null) | None => false,
					Some(Value::String(text)) => !text.is_empty(),
					Some(Value::Array(items)) => !items.is_empty(),
					Some(_) => true,
				};
				if !has_content
					&& assistant_msg.get("tool_calls").is_none()
					&& replay_reasoning_details.is_empty()
				{
					// The TS `for` loop's `i++` still runs on `continue`.
					i += 1;
					continue;
				}
				params.push(Value::Object(assistant_msg));
				last_role = Some(msg.role().to_string());
			}
			Message::ToolResult(_) => {
				let mut image_blocks: Vec<Value> = Vec::new();
				let mut j = i;

				while j < transformed_messages.len() {
					let Some(tool_msg) = transformed_messages[j].as_tool_result() else {
						break;
					};

					let text_result: String = tool_msg
						.content
						.iter()
						.filter_map(|block| match block {
							ImageOrTextContent::Text(text) => Some(text.text.clone()),
							ImageOrTextContent::Image(_) => None,
						})
						.collect::<Vec<_>>()
						.join("\n");
					let has_images = tool_msg
						.content
						.iter()
						.any(|c| matches!(c, ImageOrTextContent::Image(_)));

					// Always send tool result with text (or placeholder if only images)
					let has_text = !text_result.is_empty();
					let mut tool_result_msg = Map::new();
					tool_result_msg.insert("role".to_string(), Value::String("tool".to_string()));
					tool_result_msg.insert(
						"content".to_string(),
						Value::String(sanitize_surrogates(if has_text {
							text_result.as_str()
						} else if has_images {
							"(see attached image)"
						} else {
							""
						})),
					);
					tool_result_msg.insert(
						"tool_call_id".to_string(),
						Value::String(tool_msg.tool_call_id.clone()),
					);
					if compat.requires_tool_result_name && !tool_msg.tool_name.is_empty() {
						tool_result_msg.insert("name".to_string(), Value::String(tool_msg.tool_name.clone()));
					}
					params.push(Value::Object(tool_result_msg));

					if has_images && model.input.iter().any(|m| matches!(m, InputModality::Image)) {
						for block in &tool_msg.content {
							if let ImageOrTextContent::Image(image) = block {
								let mut url = Map::new();
								url.insert(
									"url".to_string(),
									Value::String(format!("data:{};base64,{}", image.mime_type, image.data)),
								);
								let mut part = Map::new();
								part.insert("type".to_string(), Value::String("image_url".to_string()));
								part.insert("image_url".to_string(), Value::Object(url));
								image_blocks.push(Value::Object(part));
							}
						}
					}

					j += 1;
				}

				i = j.saturating_sub(1);

				if !image_blocks.is_empty() {
					if compat.requires_assistant_after_tool_result {
						let mut bridge = Map::new();
						bridge.insert("role".to_string(), Value::String("assistant".to_string()));
						bridge.insert(
							"content".to_string(),
							Value::String("I have processed the tool results.".to_string()),
						);
						params.push(Value::Object(bridge));
					}

					let mut content: Vec<Value> = Vec::new();
					let mut label = Map::new();
					label.insert("type".to_string(), Value::String("text".to_string()));
					label.insert(
						"text".to_string(),
						Value::String("Attached image(s) from tool result:".to_string()),
					);
					content.push(Value::Object(label));
					content.extend(image_blocks);
					let mut message = Map::new();
					message.insert("role".to_string(), Value::String("user".to_string()));
					message.insert("content".to_string(), Value::Array(content));
					params.push(Value::Object(message));
					last_role = Some("user".to_string());
				} else {
					last_role = Some("toolResult".to_string());
				}
				i += 1;
				continue;
			}
		}

		i += 1;
	}

	Ok(params)
}

/// TS: `convertTools(tools, compat)`
fn convert_tools(tools: &[Tool], compat: &ResolvedCompat) -> Vec<Value> {
	tools
		.iter()
		.map(|tool| {
			let mut function = Map::new();
			function.insert("name".to_string(), Value::String(tool.name.clone()));
			function.insert("description".to_string(), Value::String(tool.description.clone()));
			function.insert("parameters".to_string(), tool.parameters.clone());
			// Only include strict if provider supports it. Some reject unknown fields.
			if compat.supports_strict_mode {
				function.insert("strict".to_string(), Value::Bool(false));
			}
			let mut value = Map::new();
			value.insert("type".to_string(), Value::String("function".to_string()));
			value.insert("function".to_string(), Value::Object(function));
			Value::Object(value)
		})
		.collect()
}

// ---------------------------------------------------------------------------
// buildParams
// ---------------------------------------------------------------------------

/// TS: `buildParams(model, context, options?, compat?, cacheRetention?, cacheControl?)`
fn build_params(
	model: &Model,
	context: &Context,
	options: Option<&OpenAICompletionsOptions>,
	compat: &ResolvedCompat,
	cache_retention: &CacheRetention,
	cache_control: Option<&OpenAICompatCacheControl>,
) -> Result<Value, String> {
	let messages = convert_messages(model, context, compat)?;

	let mut params = Map::new();
	params.insert("model".to_string(), Value::String(model.id.clone()));
	params.insert("messages".to_string(), Value::Array(messages.clone()));
	params.insert("stream".to_string(), Value::Bool(true));
	let session_id = options.and_then(|options| options.stream.session_id.clone());
	let prompt_cache_key = if (model.base_url.contains("api.openai.com") && cache_retention != "none")
		|| (cache_retention == "long" && compat.supports_long_cache_retention)
	{
		session_id.clone()
	} else {
		None
	};
	// `JSON.stringify` drops `undefined` values, so absent keys stay absent.
	if let Some(prompt_cache_key) = prompt_cache_key {
		params.insert("prompt_cache_key".to_string(), Value::String(prompt_cache_key));
	}
	if cache_retention == "long" && compat.supports_long_cache_retention {
		params.insert(
			"prompt_cache_retention".to_string(),
			Value::String("24h".to_string()),
		);
	}

	// TS: `if (compat.supportsUsageInStreaming !== false)` - an explicit `true`
	// is the same as the default, so the resolved value is enough.
	if compat.supports_usage_in_streaming {
		let mut stream_options = Map::new();
		stream_options.insert("include_usage".to_string(), Value::Bool(true));
		params.insert("stream_options".to_string(), Value::Object(stream_options));
	}

	if compat.supports_store {
		params.insert("store".to_string(), Value::Bool(false));
	}

	if let Some(max_tokens) = options.and_then(|options| options.stream.max_tokens).filter(|value| *value != 0.0) {
		if compat.max_tokens_field == "max_tokens" {
			params.insert("max_tokens".to_string(), number(max_tokens));
		} else {
			params.insert("max_completion_tokens".to_string(), number(max_tokens));
		}
	}

	if let Some(temperature) = options.and_then(|options| options.stream.temperature) {
		params.insert("temperature".to_string(), number(temperature));
	}

	let has_tools = context.tools.as_ref().map(|tools| !tools.is_empty()).unwrap_or(false);
	if has_tools {
		let tools = convert_tools(context.tools.as_deref().unwrap_or_default(), compat);
		params.insert("tools".to_string(), Value::Array(tools));
		if compat.zai_tool_stream {
			params.insert("tool_stream".to_string(), Value::Bool(true));
		}
	} else if has_tool_history(&context.messages) {
		// Anthropic (via LiteLLM/proxy) needs tools param when conversation has tool_calls/tool_results
		params.insert("tools".to_string(), Value::Array(Vec::new()));
	}

	if let Some(cache_control) = cache_control {
		let mut messages_value = params.get("messages").cloned().unwrap_or(Value::Array(Vec::new()));
		let mut tools_vec = match params.get("tools").cloned() {
			Some(Value::Array(tools)) => Some(tools),
			_ => None,
		};
		if let Some(messages_array) = messages_value.as_array_mut() {
			apply_anthropic_cache_control(messages_array.as_mut_slice(), tools_vec.as_mut(), cache_control);
		}
		params.insert("messages".to_string(), messages_value);
		if let Some(tools_vec) = tools_vec {
			params.insert("tools".to_string(), Value::Array(tools_vec));
		}
	}

	if let Some(tool_choice) = options.and_then(|options| options.tool_choice.clone()) {
		params.insert("tool_choice".to_string(), tool_choice);
	}

	let reasoning_effort = options.and_then(|options| options.reasoning_effort.clone());
	let reasoning_enabled = options.and_then(|options| options.reasoning_enabled);
	if compat.thinking_format == "zai" && model.reasoning {
		params.insert(
			"enable_thinking".to_string(),
			Value::Bool(reasoning_effort.is_some()),
		);
	} else if compat.thinking_format == "qwen" && model.reasoning {
		params.insert(
			"enable_thinking".to_string(),
			Value::Bool(reasoning_effort.is_some()),
		);
	} else if compat.thinking_format == "qwen-chat-template" && model.reasoning {
		let mut kwargs = Map::new();
		kwargs.insert("enable_thinking".to_string(), Value::Bool(reasoning_effort.is_some()));
		kwargs.insert("preserve_thinking".to_string(), Value::Bool(true));
		params.insert("chat_template_kwargs".to_string(), Value::Object(kwargs));
	} else if compat.thinking_format == "deepseek" && model.reasoning {
		let mut thinking = Map::new();
		thinking.insert(
			"type".to_string(),
			Value::String(if reasoning_effort.is_some() { "enabled" } else { "disabled" }.to_string()),
		);
		params.insert("thinking".to_string(), Value::Object(thinking));
		if let Some(reasoning_effort) = &reasoning_effort {
			params.insert(
				"reasoning_effort".to_string(),
				Value::String(
					thinking_level_mapped(model, reasoning_effort)
						.flatten()
						.unwrap_or_else(|| reasoning_effort.clone()),
				),
			);
		}
	} else if compat.thinking_format == "openrouter" && model.reasoning {
		// OpenRouter distinguishes an omitted reasoning preference (use the model
		// default), an explicit toggle, and an explicit effort selection.
		if reasoning_effort.is_some() && compat.supports_reasoning_effort {
			let reasoning_effort = reasoning_effort.clone().unwrap_or_default();
			let mut reasoning = Map::new();
			reasoning.insert(
				"effort".to_string(),
				Value::String(
					thinking_level_mapped(model, &reasoning_effort)
						.flatten()
						.unwrap_or(reasoning_effort),
				),
			);
			params.insert("reasoning".to_string(), Value::Object(reasoning));
		} else if reasoning_enabled == Some(true) {
			params.insert("reasoning".to_string(), json!({ "enabled": true }));
		} else if reasoning_enabled == Some(false) && thinking_level_mapped(model, "off") != Some(None) {
			params.insert(
				"reasoning".to_string(),
				if compat.supports_reasoning_effort {
					let off = thinking_level_mapped(model, "off").flatten().unwrap_or_else(|| "none".to_string());
					json!({ "effort": off })
				} else {
					json!({ "enabled": false })
				},
			);
		}
	} else if let Some(reasoning_effort) = &reasoning_effort {
		if model.reasoning && compat.supports_reasoning_effort {
			params.insert(
				"reasoning_effort".to_string(),
				Value::String(
					thinking_level_mapped(model, reasoning_effort)
						.flatten()
						.unwrap_or_else(|| reasoning_effort.clone()),
				),
			);
		}
	} else if reasoning_enabled == Some(false) && model.reasoning && compat.supports_reasoning_effort {
		let off_value = thinking_level_mapped(model, "off");
		if off_value != Some(None) {
			params.insert(
				"reasoning_effort".to_string(),
				Value::String(off_value.flatten().unwrap_or_else(|| "none".to_string())),
			);
		}
	}

	if model.base_url.contains("openrouter.ai") {
		if let Some(routing) = model.compat_completions().and_then(|compat| compat.open_router_routing.as_ref()) {
			if let Ok(value) = serde_json::to_value(routing) {
				params.insert("provider".to_string(), value);
			}
		}
	}

	if model.base_url.contains("ai-gateway.vercel.sh") {
		if let Some(routing) = model
			.compat_completions()
			.and_then(|compat| compat.vercel_gateway_routing.as_ref())
		{
			if routing.only.is_some() || routing.order.is_some() {
				let mut gateway_options = Map::new();
				if let Some(only) = &routing.only {
					gateway_options.insert("only".to_string(), json!(only));
				}
				if let Some(order) = &routing.order {
					gateway_options.insert("order".to_string(), json!(order));
				}
				params.insert("providerOptions".to_string(), json!({ "gateway": gateway_options }));
			}
		}
	}

	Ok(Value::Object(params))
}

/// `JSON.stringify` of a JS number: integral values stay integers.
fn number(value: f64) -> Value {
	if value.fract() == 0.0 && value.abs() < 9.007_199_254_740_992e15 {
		Value::Number(serde_json::Number::from(value as i64))
	} else {
		Value::Number(serde_json::Number::from_f64(value).unwrap_or_else(|| serde_json::Number::from(0)))
	}
}

// ---------------------------------------------------------------------------
// Usage / stop reason mapping
// ---------------------------------------------------------------------------

/// TS: `parseChunkUsage(rawUsage, model, cacheWriteCost?)`
pub(crate) fn parse_chunk_usage(raw_usage: &Value, model: &Model, cache_write_cost: Option<f64>) -> crate::types::Usage {
	let prompt_tokens = raw_usage
		.get("prompt_tokens")
		.and_then(Value::as_f64)
		.filter(|value| *value != 0.0)
		.unwrap_or(0.0);
	let prompt_tokens_details = raw_usage.get("prompt_tokens_details");
	let reported_cached_tokens = prompt_tokens_details
		.and_then(|details| details.get("cached_tokens"))
		.and_then(Value::as_f64)
		.or_else(|| raw_usage.get("prompt_cache_hit_tokens").and_then(Value::as_f64))
		.unwrap_or(0.0);
	let cache_write_tokens = prompt_tokens_details
		.and_then(|details| details.get("cache_write_tokens"))
		.and_then(Value::as_f64)
		.filter(|value| *value != 0.0)
		.unwrap_or(0.0);

	// Normalize to pi-ai semantics:
	// - cacheRead: hits from cache created by previous requests only
	// - cacheWrite: tokens written to cache in this request
	// Some OpenAI-compatible providers (observed on OpenRouter) report cached_tokens
	// as (previous hits + current writes). In that case, remove cacheWrite from cacheRead.
	let cache_read_tokens = if cache_write_tokens > 0.0 {
		(reported_cached_tokens - cache_write_tokens).max(0.0)
	} else {
		reported_cached_tokens
	};

	let input = (prompt_tokens - cache_read_tokens - cache_write_tokens).max(0.0);
	// OpenAI completion_tokens already includes reasoning_tokens.
	let output_tokens = raw_usage
		.get("completion_tokens")
		.and_then(Value::as_f64)
		.filter(|value| *value != 0.0)
		.unwrap_or(0.0);
	let mut usage = crate::types::Usage {
		input,
		output: output_tokens,
		cache_read: cache_read_tokens,
		cache_write: cache_write_tokens,
		total_tokens: input + output_tokens + cache_read_tokens + cache_write_tokens,
		cost: crate::types::UsageCost::zero(),
	};
	let overrides = cache_write_cost.map(|cache_write| CostOverrides {
		cache_write: Some(cache_write),
	});
	calculate_cost(model, &mut usage, overrides.as_ref());
	usage
}

/// TS: `mapStopReason(reason)` result.
struct StopReasonResult {
	stop_reason: String,
	error_message: Option<String>,
}

/// TS: `mapStopReason(reason)`
fn map_stop_reason(reason: Option<&Value>) -> StopReasonResult {
	let Some(reason) = reason else {
		return StopReasonResult {
			stop_reason: "stop".to_string(),
			error_message: None,
		};
	};
	if reason.is_null() {
		return StopReasonResult {
			stop_reason: "stop".to_string(),
			error_message: None,
		};
	}
	let reason = match reason {
		Value::String(reason) => reason.clone(),
		other => other.to_string(),
	};
	match reason.as_str() {
		"stop" | "end" => StopReasonResult {
			stop_reason: "stop".to_string(),
			error_message: None,
		},
		"length" => StopReasonResult {
			stop_reason: "length".to_string(),
			error_message: None,
		},
		"function_call" | "tool_calls" => StopReasonResult {
			stop_reason: "toolUse".to_string(),
			error_message: None,
		},
		"content_filter" => StopReasonResult {
			stop_reason: "error".to_string(),
			error_message: Some("Provider finish_reason: content_filter".to_string()),
		},
		"network_error" => StopReasonResult {
			stop_reason: "error".to_string(),
			error_message: Some("Provider finish_reason: network_error".to_string()),
		},
		_ => StopReasonResult {
			stop_reason: "error".to_string(),
			error_message: Some(format!("Provider finish_reason: {reason}")),
		},
	}
}

// ---------------------------------------------------------------------------
// Streaming block bookkeeping
// ---------------------------------------------------------------------------

/// TS: `interface StreamingToolCallBlock extends ToolCall { partialArgs?; streamIndex? }`
#[derive(Debug, Clone, Default)]
struct StreamingToolCallBlock {
	tool_call: ToolCall,
	partial_args: Option<String>,
	stream_index: Option<i64>,
}

/// TS: `type StreamingBlock = TextContent | ThinkingContent | StreamingToolCallBlock`
#[derive(Debug, Clone)]
enum StreamingBlock {
	Text(TextContent),
	Thinking(ThinkingContent),
	ToolCall(StreamingToolCallBlock),
}

/// The streaming accumulator for one response: it mirrors the TypeScript closures
/// (`finishBlock`, `ensureTextBlock`, `ensureThinkingBlock`, `ensureToolCallBlock`)
/// and the `blocks` array that `output.content` aliases.
struct StreamState {
	blocks: Vec<StreamingBlock>,
	text_block: Option<usize>,
	thinking_block: Option<usize>,
	tool_call_blocks_by_index: HashMap<i64, usize>,
	tool_call_blocks_by_id: HashMap<String, usize>,
	reasoning_details_by_index: HashMap<i64, Map<String, Value>>,
	next_reasoning_details_index: i64,
	reasoning_details_block: Option<usize>,
}

impl StreamState {
	fn new() -> Self {
		Self {
			blocks: Vec::new(),
			text_block: None,
			thinking_block: None,
			tool_call_blocks_by_index: HashMap::new(),
			tool_call_blocks_by_id: HashMap::new(),
			reasoning_details_by_index: HashMap::new(),
			next_reasoning_details_index: 0,
			reasoning_details_block: None,
		}
	}

	/// `output.content` - the assistant message content the stream events carry.
	fn content(&self) -> Vec<ContentBlock> {
		self.blocks
			.iter()
			.map(|block| match block {
				StreamingBlock::Text(text) => ContentBlock::Text(text.clone()),
				StreamingBlock::Thinking(thinking) => ContentBlock::Thinking(thinking.clone()),
				StreamingBlock::ToolCall(tool_call) => ContentBlock::ToolCall(tool_call.tool_call.clone()),
			})
			.collect()
	}
}

// ---------------------------------------------------------------------------
// SSE transport (reqwest + a local SSE parser mirroring the SDK)
// ---------------------------------------------------------------------------

/// A thrown provider/SDK error: `message` is what the TypeScript catch turns into
/// `output.errorMessage`, `value` is the structured shape `recordStreamFailure`
/// inspects.
#[derive(Debug, Clone)]
struct StreamError {
	message: String,
	value: Value,
}

/// `openai@6.47.0` `OpenAI.DEFAULT_TIMEOUT = 600000; // 10 minutes`, applied when the
/// caller passes no `options.timeoutMs` (TS: `options?.timeoutMs !== undefined`).
const DEFAULT_TIMEOUT_MS: f64 = 600_000.0;

/// The deadline the SDK waits for RESPONSE HEADERS:
/// `options.timeout = options.timeout ?? this.timeout` (`client.js` buildRequest) fed to
/// `fetchWithTimeout`, whose `finally { clearTimeout(timeout) }` runs as soon as `fetch`
/// resolves. The body that follows has no deadline in either implementation.
fn resolve_header_timeout(timeout_ms: Option<f64>) -> std::time::Duration {
	std::time::Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS).max(0.0) as u64)
}

/// Lets `?` convert the provider-helper `Result<_, String>` errors (e.g. the
/// `No API key for provider` / compaction-checkpoint errors) into a thrown error.
impl From<String> for StreamError {
	fn from(message: String) -> Self {
		StreamError::new(message)
	}
}

impl StreamError {
	fn new(message: impl Into<String>) -> Self {
		let message = message.into();
		let value = json!({ "name": "Error", "message": message });
		Self { message, value }
	}

	/// TS SDK `APIError.makeMessage(status, error, message)` text.
	fn api_error_message(status: Option<u16>, error: Option<&Value>, message: Option<&str>) -> String {
		// `const msg = error?.message ? typeof error.message === 'string' ? error.message :
		// JSON.stringify(error.message) : error ? JSON.stringify(error) : message;`
		let msg = error
			.and_then(|error| error.get("message"))
			.filter(|value| js_truthy(value))
			.map(|value| match value {
				Value::String(text) => text.clone(),
				other => other.to_string(),
			})
			.or_else(|| error.filter(|error| js_truthy(error)).map(|error| error.to_string()))
			.or_else(|| message.map(str::to_string));
		match (msg, status) {
			(Some(msg), Some(status)) if !msg.is_empty() => format!("{status} {msg}"),
			(_, Some(status)) => format!("{status} status code (no body)"),
			(Some(msg), None) => msg,
			(None, None) => "(no status code or body)".to_string(),
		}
	}

	/// The `headers` property of a thrown SDK error (`Headers` -> record).
	fn error_headers_value(headers: &IndexMap<String, String>) -> Value {
		let mut header_map = Map::new();
		for (name, header_value) in headers {
			header_map.insert(name.clone(), Value::String(header_value.clone()));
		}
		Value::Object(header_map)
	}

	/// The OpenAI SDK `APIError.generate(status, errorResponse, message, headers)` throw.
	///
	/// `generate` replaces the whole parsed body with the nested error first
	/// (`const error = errorResponse?.['error'];`) and `APIError` stores that same value as
	/// `this.error`, so `error.error.metadata.raw` (the OpenRouter detail the TS catch appends)
	/// stays reachable while `errorMessage` is the nested `error.message`.
	fn api_error(status: u16, error_response: Option<&Value>, message: Option<&str>, headers: &IndexMap<String, String>) -> Self {
		let payload = error_response.and_then(|body| body.get("error"));
		let text = Self::api_error_message(Some(status), payload, message);
		let mut value = json!({
			"name": "Error",
			"message": text.clone(),
			"status": status,
		});
		if let Some(payload) = payload {
			value["error"] = payload.clone();
		}
		if let Some(object) = value.as_object_mut() {
			object.insert("headers".to_string(), Self::error_headers_value(headers));
		}
		Self { message: text, value }
	}

	/// The SDK's in-stream error throw (`Stream.fromSSEResponse`):
	/// `if (data && data.error) throw new APIError(undefined, data.error, undefined, response.headers);`
	fn in_stream_api_error(error: &Value, headers: &IndexMap<String, String>) -> Self {
		let text = Self::api_error_message(None, Some(error), None);
		let mut value = json!({
			"name": "Error",
			"message": text.clone(),
			"error": error.clone(),
		});
		if let Some(object) = value.as_object_mut() {
			object.insert("headers".to_string(), Self::error_headers_value(headers));
		}
		Self { message: text, value }
	}
}

/// `AbortSignal`-equivalent throw: the SDK raises `APIUserAbortError`.
fn abort_error() -> StreamError {
	StreamError {
		message: "Request was aborted.".to_string(),
		value: json!({ "name": "AbortError", "message": "Request was aborted." }),
	}
}

fn is_cancelled(signal: Option<&tokio_util::sync::CancellationToken>) -> bool {
	signal.map(|signal| signal.is_cancelled()).unwrap_or(false)
}

/// The SSE `data` payloads of one response body, in order.
///
/// Mirrors `_iterSSEMessages` + `SSEDecoder` + `LineDecoder`: a double newline
/// delimits a chunk, `\r` is stripped, `:` comments are ignored, and `[DONE]`
/// terminates the stream.
struct SseDecoder {
	event: Option<String>,
	data: Vec<String>,
}

impl SseDecoder {
	fn new() -> Self {
		Self {
			event: None,
			data: Vec::new(),
		}
	}

	fn decode(&mut self, line: &str) -> Option<(Option<String>, String)> {
		let mut line = line.to_string();
		if line.ends_with('\r') {
			line.pop();
		}
		if line.is_empty() {
			// empty line and we didn't previously encounter any messages
			if self.event.is_none() && self.data.is_empty() {
				return None;
			}
			let event = self.event.take();
			let data = self.data.join("\n");
			self.data.clear();
			return Some((event, data));
		}
		if line.starts_with(':') {
			return None;
		}
		let (fieldname, mut value) = match line.find(':') {
			Some(index) => (line[..index].to_string(), line[index + 1..].to_string()),
			None => (line.clone(), String::new()),
		};
		if value.starts_with(' ') {
			value.remove(0);
		}
		if fieldname == "event" {
			self.event = Some(value);
		} else if fieldname == "data" {
			self.data.push(value);
		}
		None
	}
}

/// `findDoubleNewlineIndex` from the SDK line decoder.
fn find_double_newline_index(data: &[u8]) -> Option<usize> {
	let mut i = 0usize;
	while i + 1 < data.len() {
		if data[i] == b'\r' && data[i + 1] == b'\n' {
			if i + 3 < data.len() && data[i + 2] == b'\r' && data[i + 3] == b'\n' {
				return Some(i + 4);
			}
			if i + 2 < data.len() && data[i + 2] == b'\n' {
				return Some(i + 3);
			}
		}
		if data[i] == b'\n' && data[i + 1] == b'\n' {
			return Some(i + 2);
		}
		i += 1;
	}
	None
}

/// Reads the response body and forwards each `data:` payload as it arrives, so
/// the caller emits events incrementally like the SDK's async iterator.
fn observe_sse_payload(observer: Option<&crate::types::OnStreamObservation>, payload: &str) {
	let Some(observer) = observer else { return; };
	// Observe at the network reader, before parser/UI queues. Never include content,
	// and do no extra JSON parsing when local monitoring is disabled.
	let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		observer("raw_event");
		if payload.starts_with("[DONE]") {
			observer("terminal");
			return;
		}
		let Ok(chunk) = serde_json::from_str::<Value>(payload) else { return; };
		if chunk.get("error").map(js_truthy).unwrap_or(false) {
			observer("terminal");
		}
		let delta = &chunk["choices"][0]["delta"];
		if delta["content"].as_str().map(|text| !text.is_empty()).unwrap_or(false) {
			observer("text");
		}
		if ["reasoning_content", "reasoning", "reasoning_text"].iter()
			.any(|key| delta[*key].as_str().map(|text| !text.is_empty()).unwrap_or(false))
			|| delta["reasoning_details"].as_array().map(|details| !details.is_empty()).unwrap_or(false)
		{
			observer("thinking");
		}
		if delta["tool_calls"].as_array().map(|calls| !calls.is_empty()).unwrap_or(false) {
			observer("tool");
		}
	}));
}

fn observe_chunk_usage(raw: &Value, model: &Model, options: Option<&OpenAICompletionsOptions>, stream: &AssistantMessageEventStream) {
	let Some(observer) = options.and_then(|options| options.stream.on_usage_observation.as_ref()) else { return; };
	let finite = |value: Option<&Value>| value.and_then(Value::as_f64).filter(|value| value.is_finite() && *value >= 0.0);
	let observation = crate::types::ProviderUsageObservation {
		input_tokens: Some(finite(raw.get("prompt_tokens"))),
		cached_input_tokens: Some(finite(raw.get("prompt_tokens_details").and_then(|v| v.get("cached_tokens")))
			.or_else(|| finite(raw.get("prompt_cache_hit_tokens")))),
		output_tokens: Some(finite(raw.get("completion_tokens"))),
		reasoning_tokens: Some(finite(raw.get("completion_tokens_details").and_then(|v| v.get("reasoning_tokens")))),
		total_tokens: Some(finite(raw.get("total_tokens"))),
		// Raw OpenAI-compatible wire totals, not pi-ai's normalized uncached input.
		cached_input_included_in_input: Some(Some(true)),
		reasoning_included_in_output: Some(Some(true)),
	};
	if let Ok(future) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| observer(observation, model))) {
		// Monitoring stays off the output path, but its completion is owned.
		stream.spawn(future);
	}
}

async fn read_sse_data(
	response: reqwest::Response,
	signal: Option<tokio_util::sync::CancellationToken>,
	sender: tokio::sync::mpsc::UnboundedSender<Result<String, StreamError>>,
	observer: Option<crate::types::OnStreamObservation>,
) {
	let mut response = response;
	let mut decoder = SseDecoder::new();
	let mut line_buffer: Vec<u8> = Vec::new();
	let mut done = false;

	loop {
		if is_cancelled(signal.as_ref()) {
			let _ = sender.send(Err(abort_error()));
			return;
		}
		// The SDK's AbortSignal also cancels a pending body read. Checking only
		// between chunks leaves a stalled stream holding its connection open.
		let next_chunk = tokio::select! {
			chunk = response.chunk() => chunk,
			_ = async {
				match signal.as_ref() {
					Some(signal) => signal.cancelled().await,
					None => std::future::pending::<()>().await,
				}
			} => {
				let _ = sender.send(Err(abort_error()));
				return;
			}
			_ = sender.closed() => return,
		};
		let chunk = match next_chunk {
			Ok(Some(chunk)) => chunk,
			Ok(None) => break,
			Err(error) => {
				if is_cancelled(signal.as_ref()) {
					let _ = sender.send(Err(abort_error()));
				} else {
					let _ = sender.send(Err(StreamError::new(error.to_string())));
				}
				return;
			}
		};
		let mut buffer = line_buffer.clone();
		buffer.extend_from_slice(&chunk);
		while let Some(index) = find_double_newline_index(&buffer) {
			let block: Vec<u8> = buffer.drain(..index).collect();
			let text = String::from_utf8_lossy(&block).to_string();
			for line in text.split('\n') {
				if done {
					continue;
				}
				if let Some((_, payload)) = decoder.decode(line) {
					observe_sse_payload(observer.as_ref(), &payload);
					if payload.starts_with("[DONE]") {
						done = true;
						continue;
					}
					if sender.send(Ok(payload)).is_err() {
						return;
					}
				}
			}
		}
		line_buffer = buffer;
	}

	let tail = String::from_utf8_lossy(&line_buffer).to_string();
	for line in tail.split('\n') {
		if done {
			continue;
		}
		if let Some((_, payload)) = decoder.decode(line) {
			observe_sse_payload(observer.as_ref(), &payload);
			if payload.starts_with("[DONE]") {
				done = true;
				continue;
			}
			if sender.send(Ok(payload)).is_err() {
				return;
			}
		}
	}
}

/// TS: the `openai` client request. Sends the same JSON body and headers.
async fn post_chat_completions(
	client: &OpenAIClient,
	body: &Value,
	signal: Option<&tokio_util::sync::CancellationToken>,
	timeout_ms: Option<f64>,
) -> Result<reqwest::Response, StreamError> {
	let url = format!("{}/chat/completions", client.base_url.trim_end_matches('/'));
	let mut headers: IndexMap<String, Option<String>> = IndexMap::new();
	headers.insert(
		"Authorization".to_string(),
		Some(format!("Bearer {}", client.api_key)),
	);
	headers.insert("Content-Type".to_string(), Some("application/json".to_string()));
	headers.insert("Accept".to_string(), Some("application/json".to_string()));
	for (name, value) in &client.default_headers {
		headers.insert(name.clone(), value.clone());
	}

	let mut builder = reqwest::Client::new().post(&url);
	for (name, value) in &headers {
		match value {
			Some(value) => builder = builder.header(name.as_str(), value.as_str()),
			None => {}
		}
	}
	builder = builder.json(body);

	// TS: `...(options?.timeoutMs !== undefined ? { timeout: options.timeoutMs } : {})`
	// -> the openai@6.47.0 SDK falls back to `OpenAI.DEFAULT_TIMEOUT = 600000` (10 minutes),
	// and `fetchWithTimeout` clears that timer in its `finally` once `fetch` resolves, so
	// the deadline covers CONNECT + RESPONSE HEADERS only. It must never be a reqwest
	// `RequestBuilder::timeout` (a TOTAL deadline) because that aborts a long-lived stream.
	let header_timeout = resolve_header_timeout(timeout_ms);

	let request = builder.send();
	let response = match signal {
		Some(signal) => {
			tokio::select! {
				result = tokio::time::timeout(header_timeout, request) => result,
				_ = signal.cancelled() => return Err(abort_error()),
			}
		}
		None => tokio::time::timeout(header_timeout, request).await,
	};
	let response = match response {
		Ok(Ok(response)) => response,
		Ok(Err(error)) => {
			if is_cancelled(signal) {
				return Err(abort_error());
			}
			if error.is_timeout() {
				return Err(StreamError::new("Request timed out."));
			}
			return Err(StreamError::new(format!("Connection error: {error}")));
		}
		// `APIConnectionTimeoutError`: `Request timed out.`
		Err(_elapsed) => {
			if is_cancelled(signal) {
				return Err(abort_error());
			}
			return Err(StreamError::new("Request timed out."));
		}
	};

	let status = response.status().as_u16();
	if status >= 400 {
		let headers_record = crate::utils::headers::header_map_to_record(response.headers());
		let text = response.text().await.unwrap_or_default();
		// SDK: `const errJSON = safeJSON(errText); const errMessage = errJSON ? undefined : errText;`
		// (`safeJSON` returns `undefined` for unparseable text), then
		// `makeStatusError(status, errJSON, errMessage, response.headers)`.
		let parsed: Option<Value> = serde_json::from_str(&text).ok();
		let error_response = parsed.as_ref().filter(|value| js_truthy(value));
		let message = if error_response.is_some() { None } else { Some(text.as_str()) };
		return Err(StreamError::api_error(
			status,
			error_response,
			message,
			&headers_record,
		));
	}

	Ok(response)
}

// ---------------------------------------------------------------------------
// Stream body
// ---------------------------------------------------------------------------

/// The TypeScript async IIFE body, minus the try/catch (the caller adds it).
///
/// `output` is the accumulator the CALLER owns, exactly like the TS `output` const the
/// catch reuses (`stream.push({ type: "error", reason: output.stopReason, error: output })`),
/// so an `Err` leaves the partial content/usage/responseId the stream already produced.
async fn run_stream(
	model: &Model,
	context: &Context,
	options: Option<OpenAICompletionsOptions>,
	output: &mut AssistantMessage,
	stream: &AssistantMessageEventStream,
) -> Result<(), StreamError> {
	let mut state = StreamState::new();
	let result = run_stream_body(model, context, options, output, &mut state, stream).await;
	if result.is_err() {
		// TS: the catch iterates `output.content`, which aliases the live `blocks`
		// array, and strips the streaming scratch buffers (which never reach
		// `StreamState::content`).
		output.content = state.content();
	}
	result
}

/// TS: the body of the `async () => { try { ... } }` IIFE.
async fn run_stream_body(
	model: &Model,
	context: &Context,
	options: Option<OpenAICompletionsOptions>,
	output: &mut AssistantMessage,
	state: &mut StreamState,
	stream: &AssistantMessageEventStream,
) -> Result<(), StreamError> {
	let options_ref = options.as_ref();
	let api_key = options_ref
		.and_then(|options| options.stream.api_key.clone())
		.or_else(|| get_env_api_key(&model.provider))
		.unwrap_or_default();
	let compat = get_compat(model);
	let cache_retention = resolve_cache_retention(options_ref.and_then(|options| options.stream.cache_retention.as_ref()));
	let cache_control = get_compat_cache_control(&compat, &cache_retention);
	let cache_write_cost = match (&cache_control, has_standard_anthropic_cache_pricing(model)) {
		(Some(cache_control), true) => Some(get_anthropic_cache_write_cost(
			model.cost.input,
			if cache_control.ttl.as_deref() == Some("1h") { "1h" } else { "5m" },
			None,
		)),
		_ => None,
	};
	let session_id = options_ref.and_then(|options| options.stream.session_id.clone());
	let cache_session_id = if cache_retention == "none" { None } else { session_id.clone() };
	let client = create_client(
		model,
		context,
		Some(api_key.as_str()),
		options_ref.and_then(|options| options.stream.headers.as_ref()),
		cache_session_id.as_deref(),
		&compat,
		session_id.as_deref(),
	)?;

	let mut params = build_params(model, context, options_ref, &compat, &cache_retention, cache_control.as_ref())?;
	if let Some(on_payload) = options_ref.and_then(|options| options.stream.on_payload.clone()) {
		if let Some(next_params) = on_payload(params.clone(), model).await {
			params = next_params;
		}
	}

	let signal = options_ref.and_then(|options| options.stream.signal.clone());
	let timeout_ms = options_ref.and_then(|options| options.stream.timeout_ms);

	let response = post_chat_completions(&client, &params, signal.as_ref(), timeout_ms).await?;
	let status = response.status().as_u16();
	if let Some(on_response) = options_ref.and_then(|options| options.stream.on_response.clone()) {
		let mut response_record = crate::types::ProviderResponse {
			status: status as i64,
			headers: crate::utils::headers::header_map_to_record(response.headers()),
		};
		response_record.headers.insert("x-optimus-transport".into(), "sse".into());
		on_response(response_record, model).await;
	}
	stream.push(AssistantMessageEvent::Start {
		partial: output.clone(),
	});

	// The SDK throws the in-stream error with `response.headers`
	// (`throw new APIError(undefined, data.error, undefined, response.headers)`), so the
	// headers must be captured while the response is still owned here.
	let response_headers = crate::utils::headers::header_map_to_record(response.headers());

	let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<Result<String, StreamError>>();
	let observer = options_ref.and_then(|options| options.stream.on_stream_observation.clone());
	stream.spawn(read_sse_data(response, signal.clone(), sender, observer));
	while let Some(payload) = receiver.recv().await {
		let payload = payload?;
		let Ok(chunk) = serde_json::from_str::<Value>(&payload) else {
			continue;
		};
		if !chunk.is_object() {
			continue;
		}

		// SDK `Stream.fromSSEResponse`: `if (data && data.error) throw new APIError(...)`,
		// so an error event delivered inside the SSE body fails the stream instead of
		// being silently skipped and reported as a successful run.
		if let Some(error) = chunk.get("error").filter(|error| js_truthy(error)) {
			return Err(StreamError::in_stream_api_error(error, &response_headers));
		}

		// OpenAI documents ChatCompletionChunk.id as the unique chat completion identifier,
		// and each chunk in a streamed completion carries the same id.
		if output.response_id.as_deref().map(str::is_empty).unwrap_or(true) {
			if let Some(id) = chunk.get("id").and_then(Value::as_str) {
				output.response_id = Some(id.to_string());
			}
		}
		if let Some(chunk_model) = chunk.get("model").and_then(Value::as_str) {
			if !chunk_model.is_empty() && chunk_model != model.id {
				if output.response_model.as_deref().map(str::is_empty).unwrap_or(true) {
					output.response_model = Some(chunk_model.to_string());
				}
			}
		}
		let chunk_usage = chunk.get("usage").filter(|value| js_truthy(value));
		if let Some(chunk_usage) = chunk_usage {
			output.usage = parse_chunk_usage(chunk_usage, model, cache_write_cost);
			observe_chunk_usage(chunk_usage, model, options_ref, stream);
		}

		let choice = chunk
			.get("choices")
			.and_then(Value::as_array)
			.and_then(|choices| choices.first())
			.cloned();
		let Some(choice) = choice else {
			continue;
		};

		// Fallback: some providers (e.g., Moonshot) return usage
		// in choice.usage instead of the standard chunk.usage
		if chunk_usage.is_none() {
			if let Some(choice_usage) = choice.get("usage").filter(|value| js_truthy(value)) {
				output.usage = parse_chunk_usage(choice_usage, model, cache_write_cost);
				observe_chunk_usage(choice_usage, model, options_ref, stream);
			}
		}

		if let Some(finish_reason) = choice.get("finish_reason").filter(|value| js_truthy(value)) {
			let finish_reason_result = map_stop_reason(Some(finish_reason));
			output.stop_reason = finish_reason_result.stop_reason;
			if let Some(error_message) = finish_reason_result.error_message {
				output.error_message = Some(error_message);
			}
		}

		if let Some(delta) = choice.get("delta").filter(|delta| !delta.is_null()) {
			let content = delta.get("content").and_then(Value::as_str);
			if let Some(content) = content {
				if !content.is_empty() {
					let index = ensure_text_block(state, output, stream);
					if let StreamingBlock::Text(block) = &mut state.blocks[index] {
						block.text.push_str(content);
					}
					let partial = partial_message(output, state);
					stream.push(AssistantMessageEvent::TextDelta {
						content_index: index,
						delta: content.to_string(),
						partial,
					});
				}
			}

			// Some endpoints return reasoning in reasoning_content (llama.cpp),
			// or reasoning (other openai compatible endpoints)
			// Use the first non-empty reasoning field to avoid duplication
			// (e.g., chutes.ai returns both reasoning_content and reasoning with same content)
			let reasoning_fields = ["reasoning_content", "reasoning", "reasoning_text"];
			let mut found_reasoning_field: Option<&str> = None;
			for field in reasoning_fields {
				if let Some(value) = delta.get(field).and_then(Value::as_str) {
					if !value.is_empty() {
						found_reasoning_field = Some(field);
						break;
					}
				}
			}

			if let Some(found_reasoning_field) = found_reasoning_field {
				if let Some(delta_text) = delta.get(found_reasoning_field).and_then(Value::as_str) {
					if !delta_text.is_empty() {
						let index = ensure_thinking_block(state, output, stream, found_reasoning_field);
						if let StreamingBlock::Thinking(block) = &mut state.blocks[index] {
							block.thinking.push_str(delta_text);
						}
						let partial = partial_message(output, state);
						stream.push(AssistantMessageEvent::ThinkingDelta {
							content_index: index,
							delta: delta_text.to_string(),
							partial,
						});
					}
				}
			}

			if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
				for tool_call in tool_calls {
					let index = ensure_tool_call_block(state, output, stream, tool_call);
					let tool_call_id = tool_call.get("id").and_then(Value::as_str);
					let tool_call_name = tool_call
						.get("function")
						.and_then(|function| function.get("name"))
						.and_then(Value::as_str);
					let mut register_id: Option<String> = None;
					if let StreamingBlock::ToolCall(block) = &mut state.blocks[index] {
						if block.tool_call.id.is_empty() {
							if let Some(tool_call_id) = tool_call_id {
								block.tool_call.id = tool_call_id.to_string();
								register_id = Some(tool_call_id.to_string());
							}
						}
						if block.tool_call.name.is_empty() {
							if let Some(tool_call_name) = tool_call_name {
								block.tool_call.name = tool_call_name.to_string();
							}
						}
					}
					if let Some(register_id) = register_id {
						state.tool_call_blocks_by_id.insert(register_id, index);
					}

					let mut delta_text = String::new();
					if let Some(arguments) = tool_call
						.get("function")
						.and_then(|function| function.get("arguments"))
						.and_then(Value::as_str)
					{
						delta_text = arguments.to_string();
						if let StreamingBlock::ToolCall(block) = &mut state.blocks[index] {
							let partial_args = block.partial_args.get_or_insert_with(String::new);
							partial_args.push_str(arguments);
							let parsed = parse_streaming_json(Some(partial_args.as_str()));
							block.tool_call.arguments = parsed.as_object().cloned().unwrap_or_default();
						}
					}
					let partial = partial_message(output, state);
					stream.push(AssistantMessageEvent::ToolCallDelta {
						content_index: index,
						delta: delta_text,
						partial,
					});
				}
			}

			// `Array.isArray` accepts an empty array, and `reasoningDetailsByIndex`
			// stays empty, so an empty list only skips the block creation below.
			if let Some(reasoning_details) = delta.get("reasoning_details").and_then(Value::as_array) {
				for detail in reasoning_details {
					let Some(detail_record) = detail.as_object() else {
						continue;
					};
					let explicit_index = detail_record.get("index").and_then(Value::as_i64);
					let index = explicit_index.unwrap_or(state.next_reasoning_details_index);
					state.next_reasoning_details_index = state.next_reasoning_details_index.max(index + 1);
					let previous_detail = state.reasoning_details_by_index.get(&index).cloned();
					let mut merged_detail = previous_detail.clone().unwrap_or_default();
					for (key, value) in detail_record {
						merged_detail.insert(key.clone(), value.clone());
					}
					for field in ["text", "summary"] {
						let previous_fragment = previous_detail
							.as_ref()
							.and_then(|detail| detail.get(field))
							.and_then(Value::as_str);
						let fragment = detail_record.get(field).and_then(Value::as_str);
						if let (Some(previous_fragment), Some(fragment)) = (previous_fragment, fragment) {
							merged_detail.insert(
								field.to_string(),
								Value::String(format!("{previous_fragment}{fragment}")),
							);
						}
					}
					state.reasoning_details_by_index.insert(index, merged_detail);
					if detail_record.get("type").and_then(Value::as_str) == Some("reasoning.encrypted")
						&& detail_record.get("id").and_then(Value::as_str).is_some()
						&& detail_record.get("data").map(|data| !data.is_null()).unwrap_or(false)
					{
						let detail_id = detail_record.get("id").and_then(Value::as_str).unwrap_or_default();
						let matching = state.blocks.iter_mut().find(|block| match block {
							StreamingBlock::ToolCall(tool_call) => tool_call.tool_call.id == detail_id,
							_ => false,
						});
						if let Some(StreamingBlock::ToolCall(tool_call)) = matching {
							tool_call.tool_call.thought_signature = Some(Value::Object(detail_record.clone()).to_string());
						}
					}
				}
				if !state.reasoning_details_by_index.is_empty() {
					let index = match state.reasoning_details_block {
						Some(index) => index,
						None => {
							let mut block = ThinkingContent::new("");
							block.redacted = Some(true);
							state.blocks.push(StreamingBlock::Thinking(block));
							let index = state.blocks.len() - 1;
							state.reasoning_details_block = Some(index);
							let partial = partial_message(output, state);
							stream.push(AssistantMessageEvent::ThinkingStart {
								content_index: index,
								partial,
							});
							index
						}
					};
					let mut indexes: Vec<i64> = state.reasoning_details_by_index.keys().copied().collect();
					indexes.sort_unstable();
					let details: Vec<Map<String, Value>> = indexes
						.iter()
						.filter_map(|index| state.reasoning_details_by_index.get(index).cloned())
						.collect();
					if let StreamingBlock::Thinking(block) = &mut state.blocks[index] {
						block.thinking_signature = Some(encode_reasoning_details(&details));
					}
				}
			}
		}
	}

	for index in 0..state.blocks.len() {
		finish_block(state, index, output, stream);
	}
	if is_cancelled(signal.as_ref()) {
		return Err(StreamError::new("Request was aborted"));
	}

	if output.stop_reason == "aborted" {
		return Err(StreamError::new("Request was aborted"));
	}
	if output.stop_reason == "error" {
		return Err(StreamError::new(
			output
				.error_message
				.clone()
				.unwrap_or_else(|| "Provider returned an error stop reason".to_string()),
		));
	}

	output.content = state.content();
	stream.push(AssistantMessageEvent::Done {
		reason: output.stop_reason.clone(),
		message: output.clone(),
	});
	stream.end(None);
	Ok(())
}

// ---------------------------------------------------------------------------
// Block helpers (the TypeScript closures)
// ---------------------------------------------------------------------------

/// The `partial: output` payload: `output.content` aliases the live `blocks` array.
fn partial_message(output: &AssistantMessage, state: &StreamState) -> AssistantMessage {
	let mut partial = output.clone();
	partial.content = state.content();
	partial
}

/// TS: `getContentIndex(block)`
fn finish_block(
	state: &mut StreamState,
	content_index: usize,
	output: &AssistantMessage,
	stream: &AssistantMessageEventStream,
) {
	let partial = partial_message(output, state);
	match &mut state.blocks[content_index] {
		StreamingBlock::Text(block) => {
			let content = block.text.clone();
			stream.push(AssistantMessageEvent::TextEnd {
				content_index,
				content,
				partial,
			});
		}
		StreamingBlock::Thinking(block) => {
			let content = block.thinking.clone();
			stream.push(AssistantMessageEvent::ThinkingEnd {
				content_index,
				content,
				partial,
			});
		}
		StreamingBlock::ToolCall(block) => {
			block.tool_call.arguments = parse_streaming_json(block.partial_args.as_deref())
				.as_object()
				.cloned()
				.unwrap_or_default();
			// Finalize in-place and strip the scratch buffers so replay only
			// carries parsed arguments.
			block.partial_args = None;
			block.stream_index = None;
			let tool_call = block.tool_call.clone();
			stream.push(AssistantMessageEvent::ToolCallEnd {
				content_index,
				tool_call,
				partial,
			});
		}
	}
}

/// TS: `ensureTextBlock()`
fn ensure_text_block(
	state: &mut StreamState,
	output: &AssistantMessage,
	stream: &AssistantMessageEventStream,
) -> usize {
	if state.text_block.is_none() {
		state.blocks.push(StreamingBlock::Text(TextContent::new("")));
		let index = state.blocks.len() - 1;
		state.text_block = Some(index);
		let partial = partial_message(output, state);
		stream.push(AssistantMessageEvent::TextStart {
			content_index: index,
			partial,
		});
	}
	state.text_block.expect("text block")
}

/// TS: `ensureThinkingBlock(thinkingSignature)`
fn ensure_thinking_block(
	state: &mut StreamState,
	output: &AssistantMessage,
	stream: &AssistantMessageEventStream,
	thinking_signature: &str,
) -> usize {
	if state.thinking_block.is_none() {
		let mut block = ThinkingContent::new("");
		block.thinking_signature = Some(thinking_signature.to_string());
		state.blocks.push(StreamingBlock::Thinking(block));
		let index = state.blocks.len() - 1;
		state.thinking_block = Some(index);
		let partial = partial_message(output, state);
		stream.push(AssistantMessageEvent::ThinkingStart {
			content_index: index,
			partial,
		});
	}
	state.thinking_block.expect("thinking block")
}

/// TS: `ensureToolCallBlock(toolCall)`
fn ensure_tool_call_block(
	state: &mut StreamState,
	output: &AssistantMessage,
	stream: &AssistantMessageEventStream,
	tool_call: &Value,
) -> usize {
	let stream_index = tool_call.get("index").and_then(Value::as_i64);
	let tool_call_id = tool_call.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
	let tool_call_name = tool_call
		.get("function")
		.and_then(|function| function.get("name"))
		.and_then(Value::as_str)
		.unwrap_or_default()
		.to_string();

	let mut block_index = stream_index.and_then(|index| state.tool_call_blocks_by_index.get(&index).copied());
	if block_index.is_none() && !tool_call_id.is_empty() {
		block_index = state.tool_call_blocks_by_id.get(&tool_call_id).copied();
	}
	let index = match block_index {
		Some(index) => index,
		None => {
			let block = StreamingToolCallBlock {
				tool_call: ToolCall::new(tool_call_id.clone(), tool_call_name, Map::new()),
				partial_args: Some(String::new()),
				stream_index,
			};
			state.blocks.push(StreamingBlock::ToolCall(block));
			let index = state.blocks.len() - 1;
			if let Some(stream_index) = stream_index {
				state.tool_call_blocks_by_index.insert(stream_index, index);
			}
			if !tool_call_id.is_empty() {
				state.tool_call_blocks_by_id.insert(tool_call_id.clone(), index);
			}
			let partial = partial_message(output, state);
			stream.push(AssistantMessageEvent::ToolCallStart {
				content_index: index,
				partial,
			});
			index
		}
	};

	let mut register_index: Option<i64> = None;
	if let StreamingBlock::ToolCall(block) = &mut state.blocks[index] {
		if let Some(stream_index) = stream_index {
			if block.stream_index.is_none() {
				block.stream_index = Some(stream_index);
				register_index = Some(stream_index);
			}
		}
	}
	if let Some(stream_index) = register_index {
		state.tool_call_blocks_by_index.insert(stream_index, index);
	}
	if !tool_call_id.is_empty() {
		state.tool_call_blocks_by_id.insert(tool_call_id, index);
	}
	index
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// TS: `export const streamOpenAICompletions`
pub fn stream_openai_completions(
	model: &Model,
	context: &Context,
	options: Option<OpenAICompletionsOptions>,
) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new_owned();
	let out = stream.producer_handle();
	let model = model.clone();
	let context = context.clone();
	stream.spawn(async move {
		let signal = options.as_ref().and_then(|options| options.stream.signal.clone());

		// TS: `const output: AssistantMessage = { ... }` lives OUTSIDE the try, so the catch
		// keeps the partial content and the usage parsed before the failure.
		let mut output = AssistantMessage::new(model.api.clone(), model.provider.clone(), model.id.clone(), crate::utils::now_ms());
		output.usage = crate::types::Usage::zero();
		if let Err(error) = run_stream(&model, &context, options.clone(), &mut output, &out).await {
			output.stop_reason = if is_cancelled(signal.as_ref()) {
				"aborted".to_string()
			} else {
				"error".to_string()
			};
			output.error_message = Some(error.message.clone());
			// Some providers via OpenRouter give additional information in this field.
			let raw_metadata = error
				.value
				.get("error")
				.and_then(|error| error.get("metadata"))
				.and_then(|metadata| metadata.get("raw"))
				.and_then(Value::as_str);
			if let Some(raw_metadata) = raw_metadata {
				let current = output.error_message.clone().unwrap_or_default();
				output.error_message = Some(format!("{current}\n{raw_metadata}"));
			}
			let thrown = ThrownStreamError::Value(&error.value);
			record_stream_failure(&model, &mut output, &thrown);
			out.push(AssistantMessageEvent::Error {
				reason: output.stop_reason.clone(),
				error: output,
			});
			out.end(None);
		}
	});
	stream
}

/// TS: `export const streamSimpleOpenAICompletions`
pub fn stream_simple_openai_completions(
	model: &Model,
	context: &Context,
	options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
	let stream = AssistantMessageEventStream::new_owned();
	let out = stream.producer_handle();
	let model = model.clone();
	let context = context.clone();
	stream.spawn(async move {
		let api_key = options
			.as_ref()
			.and_then(|options| options.stream.api_key.clone())
			.filter(|key| !key.is_empty())
			.or_else(|| get_env_api_key(&model.provider));
		let Some(api_key) = api_key else {
			let mut output = AssistantMessage::new(model.api.clone(), model.provider.clone(), model.id.clone(), crate::utils::now_ms());
			output.usage = crate::types::Usage::zero();
			output.stop_reason = "error".to_string();
			output.error_message = Some(format!("No API key for provider: {}", model.provider));
			out.push(AssistantMessageEvent::Error {
				reason: output.stop_reason.clone(),
				error: output,
			});
			out.end(None);
			return;
		};

		let base = build_base_options(&model, options.as_ref(), Some(api_key.as_str()));
		let requested_reasoning = options.as_ref().and_then(|options| options.reasoning.clone());
		let reasoning_specified = requested_reasoning.is_some();
		let clamped_reasoning = requested_reasoning
			.as_deref()
			.map(|reasoning| clamp_thinking_level(&model, reasoning));
		let reasoning_effort = clamped_reasoning
			.clone()
			.filter(|reasoning| reasoning != "off");
		let tool_choice = options
			.as_ref()
			.and_then(|options| options.stream.extra.get("toolChoice").cloned());

		let typed = OpenAICompletionsOptions {
			stream: base,
			tool_choice,
			reasoning_effort,
			reasoning_enabled: if reasoning_specified {
				Some(clamped_reasoning.as_deref() != Some("off"))
			} else {
				None
			},
		};

		// TS: `const output` is created before the try, so the catch keeps the partial
		// content and the usage parsed before the failure.
		let mut output = AssistantMessage::new(model.api.clone(), model.provider.clone(), model.id.clone(), crate::utils::now_ms());
		output.usage = crate::types::Usage::zero();
		if let Err(error) = run_stream(&model, &context, Some(typed), &mut output, &out).await {
			let signal = options.as_ref().and_then(|options| options.stream.signal.clone());
			output.stop_reason = if is_cancelled(signal.as_ref()) {
				"aborted".to_string()
			} else {
				"error".to_string()
			};
			output.error_message = Some(error.message.clone());
			let thrown = ThrownStreamError::Value(&error.value);
			record_stream_failure(&model, &mut output, &thrown);
			out.push(AssistantMessageEvent::Error {
				reason: output.stop_reason.clone(),
				error: output,
			});
			out.end(None);
		}
	});
	stream
}


#[cfg(test)]
mod provider_settlement_tests {
    use super::*;
    use super::tests_support::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::sync::oneshot;

    fn keyed_options() -> OpenAICompletionsOptions {
        OpenAICompletionsOptions {
            stream: StreamOptions { api_key: Some("local-fixture-key".into()), ..Default::default() },
            ..Default::default()
        }
    }

    async fn local_sse(body: &'static str, stalled: bool) -> (Model, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            loop {
                let mut bytes = [0; 4096];
                let count = socket.read(&mut bytes).await.unwrap();
                assert_ne!(count, 0);
                request.extend_from_slice(&bytes[..count]);
                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = std::str::from_utf8(&request[..end]).unwrap();
                    let length: usize = headers.lines().filter_map(|line| line.split_once(':'))
                        .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                        .unwrap().1.trim().parse().unwrap();
                    if request.len() >= end + 4 + length { break; }
                }
            }
            let response = if stalled {
                format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{body}\r\n", body.len())
            } else {
                format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
            };
            socket.write_all(response.as_bytes()).await.unwrap();
            if stalled {
                let mut bytes = [0; 1];
                assert_eq!(socket.read(&mut bytes).await.unwrap(), 0, "reader still alive after stop");
            }
        });
        let mut model = base_model();
        model.base_url = format!("http://{address}");
        (model, server)
    }

    async fn join_server(server: tokio::task::JoinHandle<()>) {
        tokio::time::timeout(Duration::from_secs(5), server).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn owned_normal_and_simple_stop_join_producer_and_stalled_sse() {
        for simple in [false, true] {
            let (model, server) = local_sse("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n", true).await;
            let ctx = context(vec![user_text("fixture")]);
            let stream = if simple {
                stream_simple_openai_completions(&model, &ctx, Some(SimpleStreamOptions {
                    stream: keyed_options().stream, ..Default::default()
                }))
            } else {
                stream_openai_completions(&model, &ctx, Some(keyed_options()))
            };
            loop {
                let event = tokio::time::timeout(Duration::from_secs(5), stream.next()).await.unwrap().unwrap();
                if matches!(event, AssistantMessageEvent::TextDelta { .. }) { break; }
            }
            let receipt = stream.task_receipt();
            assert!(receipt.status().supported);
            assert_eq!(receipt.status().pending_tasks, 2);
            stream.end(None);
            assert_eq!(receipt.status().pending_tasks, 2, "channel close is not task acknowledgement");
            stream.request_cancel();
            let joined = receipt.settle(Duration::from_secs(5)).await;
            assert!(joined.settled, "{joined:?}");
            assert_eq!(joined.completed_tasks + joined.cancelled_tasks, 2);
            assert_eq!(receipt.request_cancel(), joined);
            join_server(server).await;
        }
    }

    #[tokio::test]
    async fn pending_usage_observer_is_owned_after_terminal_result() {
        let (model, server) = local_sse("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n", false).await;
        let mut options = keyed_options();
        let (started, ready) = oneshot::channel();
        let started = Arc::new(Mutex::new(Some(started)));
        options.stream.on_usage_observation = Some(Arc::new(move |_, _| {
            let started = started.lock().unwrap().take().unwrap();
            Box::pin(async move {
                started.send(()).unwrap();
                std::future::pending::<()>().await;
            })
        }));
        let stream = stream_openai_completions(&model, &context(vec![]), Some(options));
        ready.await.unwrap();
        let output = tokio::time::timeout(Duration::from_secs(5), stream.result()).await.unwrap();
        assert_eq!(output.usage.input, 10.0);
        assert_eq!(output.usage.output, 2.0);
        let receipt = stream.task_receipt();
        assert!(receipt.settle(Duration::ZERO).await.pending_tasks >= 1);
        receipt.request_cancel();
        let joined = receipt.settle(Duration::from_secs(5)).await;
        assert!(joined.settled, "{joined:?}");
        assert_eq!(joined.completed_tasks + joined.cancelled_tasks, 3);
        join_server(server).await;
    }

    #[tokio::test]
    async fn stop_during_payload_callback_drops_request_before_any_transport() {
        let mut options = keyed_options();
        let (started, ready) = oneshot::channel();
        let started = Arc::new(Mutex::new(Some(started)));
        options.stream.on_payload = Some(Arc::new(move |_, _| {
            let started = started.lock().unwrap().take().unwrap();
            Box::pin(async move {
                started.send(()).unwrap();
                std::future::pending::<Option<Value>>().await
            })
        }));
        let stream = stream_openai_completions(&base_model(), &context(vec![]), Some(options));
        ready.await.unwrap();
        let receipt = stream.task_receipt();
        receipt.request_cancel();
        let joined = receipt.settle(Duration::from_secs(1)).await;
        assert!(joined.settled, "{joined:?}");
        assert_eq!(joined.cancelled_tasks, 1);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn in_stream_error_does_not_leave_sse_waiting_for_more_body() {
        let (model, server) = local_sse("data: {\"error\":{\"message\":\"fixture-error\"}}\n\n", true).await;
        let stream = stream_openai_completions(&model, &context(vec![]), Some(keyed_options()));
        let output = tokio::time::timeout(Duration::from_secs(5), stream.result()).await.unwrap();
        assert_eq!(output.stop_reason, "error");
        let receipt = stream.task_receipt();
        let joined = receipt.settle(Duration::from_secs(5)).await;
        assert_eq!(joined.pending_tasks, 0);
        assert_eq!(joined.completed_tasks, 2);
        assert!(!joined.cancel_requested);
        join_server(server).await;
    }
}

/// Shared fixtures for the unit tests below.
#[cfg(test)]
mod tests_support {
	use super::*;
	use crate::types::{ImageContent, ModelCost, Usage};

	pub fn base_model() -> Model {
		Model {
			id: "repro-model".to_string(),
			name: "Repro Model".to_string(),
			api: "openai-completions".to_string(),
			provider: "repro-provider".to_string(),
			base_url: "http://127.0.0.1:1".to_string(),
			reasoning: true,
			input: vec![InputModality::Text],
			cost: ModelCost::zero(),
			context_window: 128_000.0,
			max_tokens: 4096.0,
			..Default::default()
		}
	}

	pub fn compat_model(compat: crate::types::OpenAICompletionsCompat) -> Model {
		Model {
			compat: Some(crate::types::Compat::Completions(compat)),
			..base_model()
		}
	}

	pub fn user_text(text: &str) -> Message {
		Message::user(crate::types::UserMessage::new(UserContent::Text(text.to_string()), 1))
	}

	pub fn assistant_message(content: Vec<ContentBlock>, stop_reason: &str) -> Message {
		Message::assistant(AssistantMessage {
			content,
			api: "openai-completions".to_string(),
			provider: "repro-provider".to_string(),
			model: "repro-model".to_string(),
			usage: Usage::zero(),
			stop_reason: stop_reason.to_string(),
			timestamp: 2,
			..Default::default()
		})
	}

	pub fn context(messages: Vec<Message>) -> Context {
		Context::new(None, messages, None)
	}

	pub fn tool_result_with_image(tool_call_id: &str) -> Message {
		Message::tool_result(crate::types::ToolResultMessage::new(
			tool_call_id,
			"read",
			vec![
				ImageOrTextContent::Text(TextContent::new("Read image file [image/png]")),
				ImageOrTextContent::Image(ImageContent::new("ZmFrZQ==", "image/png")),
			],
			false,
			3,
		))
	}
}

#[cfg(test)]
mod tests {
	use super::tests_support::*;
	use super::*;
	use crate::types::OpenAICompletionsCompat;

	// ------------------------------------------------------------------
	// A4-06: SDK APIError text/shape, and the in-stream `data.error` throw
	// ------------------------------------------------------------------

	#[test]
	fn api_error_unwraps_the_nested_error_like_api_error_generate() {
		// SDK `APIError.generate`: `const error = errorResponse?.['error'];`
		// -> message is the NESTED `error.message`, and `this.error` is that same payload.
		let headers = IndexMap::new();
		let body = json!({ "error": { "message": "Incorrect API key provided" } });
		let error = StreamError::api_error(401, Some(&body), None, &headers);
		assert_eq!(error.message, "401 Incorrect API key provided");
		assert_eq!(error.value["error"], json!({ "message": "Incorrect API key provided" }));

		// The OpenRouter raw-metadata append (`error.error.metadata.raw`) stays reachable.
		let body = json!({
			"error": { "message": "Provider error", "metadata": { "raw": "RAW DETAIL" } }
		});
		let error = StreamError::api_error(500, Some(&body), None, &headers);
		assert_eq!(error.message, "500 Provider error");
		assert_eq!(error.value["error"]["metadata"]["raw"], json!("RAW DETAIL"));

		// No nested `error`: `error` itself is `undefined`, so `makeMessage` falls through
		// to the raw text (`errJSON ? undefined : errText`) and then to `(no body)`.
		let body = json!({ "message": "plain message" });
		assert_eq!(StreamError::api_error(401, Some(&body), None, &headers).message, "401 status code (no body)");
		assert_eq!(StreamError::api_error(401, None, None, &headers).message, "401 status code (no body)");
		assert_eq!(StreamError::api_error(401, None, Some("not json"), &headers).message, "401 not json");
		// `typeof error.message === 'string'` is false for a number -> JSON.stringify path.
		let body = json!({ "error": "oops" });
		assert_eq!(StreamError::api_error(500, Some(&body), None, &headers).message, "500 \"oops\"");
	}

	#[test]
	fn in_stream_error_event_matches_the_sdk_throw() {
		// SDK: `throw new APIError(undefined, data.error, undefined, response.headers)`
		// with `status === undefined`, so `makeMessage` has no status prefix.
		let mut headers = IndexMap::new();
		headers.insert("x-request-id".to_string(), "req_1".to_string());
		let payload = json!({ "message": "boom" });
		let error = StreamError::in_stream_api_error(&payload, &headers);
		assert_eq!(error.message, "boom");
		assert_eq!(error.value["error"], payload);
		assert!(error.value.get("status").is_none());
		assert_eq!(error.value["headers"]["x-request-id"], json!("req_1"));

		let payload = json!({ "metadata": { "raw": "RAW" } });
		let error = StreamError::in_stream_api_error(&payload, &headers);
		assert_eq!(error.message, "{\"metadata\":{\"raw\":\"RAW\"}}");
		let payload = json!(null);
		let error = StreamError::in_stream_api_error(&payload, &headers);
		assert_eq!(error.message, "(no status code or body)");
	}

	/// An explicit key so `create_client` never reads the ambient environment.
	fn keyed_options() -> OpenAICompletionsOptions {
		OpenAICompletionsOptions {
			stream: StreamOptions {
				api_key: Some("fixture-key".to_string()),
				..Default::default()
			},
			..Default::default()
		}
	}

	/// A minimal one-response fixture server: reads the request, writes one HTTP
	/// response, then closes.
	async fn serve_http(status: &'static str, body: &'static str) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
		let address = listener.local_addr().unwrap();
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut request = Vec::new();
			loop {
				let mut buffer = [0u8; 4096];
				let count = socket.read(&mut buffer).await.unwrap();
				if count == 0 {
					return;
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
						break;
					}
				}
			}
			let response = format!(
				"HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
				body.len()
			);
			socket.write_all(response.as_bytes()).await.unwrap();
			socket.flush().await.unwrap();
		});
		(address, server)
	}

	#[tokio::test]
	async fn completions_monitoring_records_raw_phases_and_inclusive_usage_without_changing_output() {
		let body = concat!(
			"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"private reasoning\"}}]}\r\n\r\n",
			"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\r\n\r\n",
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"prompt_tokens_details\":{\"cached_tokens\":80},\"completion_tokens\":12,\"completion_tokens_details\":{\"reasoning_tokens\":9},\"total_tokens\":112}}\r\n\r\n",
			"data: [DONE]\r\n\r\n"
		);
		let mut outputs = Vec::new();
		for enabled in [false, true] {
			let (address, server) = serve_http("200 OK", body).await;
			let mut model = base_model();
			model.base_url = format!("http://{address}");
			let mut options = keyed_options();
			let phases = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
			let usages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
			if enabled {
				let capture = phases.clone();
				options.stream.on_stream_observation = Some(std::sync::Arc::new(move |phase| {
					capture.lock().unwrap().push(phase.into());
				}));
				let capture = usages.clone();
				options.stream.on_usage_observation = Some(std::sync::Arc::new(move |usage, _| {
					capture.lock().unwrap().push(usage);
					Box::pin(async {})
				}));
				options.stream.on_response = Some(std::sync::Arc::new(|response, _| {
					assert_eq!(response.headers["x-optimus-transport"], "sse");
					Box::pin(async {})
				}));
			}
			let stream = stream_openai_completions(&model, &context(vec![user_text("hi")]), Some(options));
			let mut done = None;
			while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next()).await.unwrap() {
				match event {
					AssistantMessageEvent::Done { message, .. } => done = Some(message),
					AssistantMessageEvent::Error { error, .. } => panic!("unexpected stream error: {:?}", error.error_message),
					_ => {}
				}
			}
			server.await.unwrap();
			let done = done.unwrap();
			outputs.push((done.content, done.usage));
			if enabled {
				assert_eq!(*phases.lock().unwrap(), vec!["raw_event", "thinking", "raw_event", "text", "raw_event", "raw_event", "terminal"]);
				let usage = usages.lock().unwrap();
				assert_eq!(usage.len(), 1);
				assert_eq!(usage[0].input_tokens, Some(Some(100.0)));
				assert_eq!(usage[0].cached_input_tokens, Some(Some(80.0)));
				assert_eq!(usage[0].reasoning_tokens, Some(Some(9.0)));
				assert_eq!(usage[0].cached_input_included_in_input, Some(Some(true)));
				assert_eq!(usage[0].reasoning_included_in_output, Some(Some(true)));
			}
		}
		assert_eq!(outputs[0], outputs[1]);
		assert_eq!(outputs[0].1.input, 20.0);
	}

	#[tokio::test]
	async fn completions_observers_preserve_unknown_and_zero_counts_and_contain_panics() {
		let stream = AssistantMessageEventStream::new_owned();
		let usages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
		let capture = usages.clone();
		let mut options = keyed_options();
		options.stream.on_usage_observation = Some(std::sync::Arc::new(move |usage, _| {
			capture.lock().unwrap().push(usage);
			Box::pin(async {})
		}));
		observe_chunk_usage(&json!({"prompt_tokens": 0, "prompt_cache_hit_tokens": 0, "completion_tokens": -1}), &base_model(), Some(&options), &stream);
		let observations = usages.lock().unwrap();
		assert_eq!(observations[0].input_tokens, Some(Some(0.0)));
		assert_eq!(observations[0].cached_input_tokens, Some(Some(0.0)));
		assert_eq!(observations[0].output_tokens, Some(None));
		assert_eq!(observations[0].reasoning_tokens, Some(None));
		drop(observations);
		options.stream.on_usage_observation = Some(std::sync::Arc::new(|_, _| panic!("disposable observer")));
		observe_chunk_usage(&json!({}), &base_model(), Some(&options), &stream);
		let observer: crate::types::OnStreamObservation = std::sync::Arc::new(|_| panic!("disposable observer"));
		observe_sse_payload(Some(&observer), "[DONE]");
	}

	#[test]
	fn completions_monitoring_is_content_free_and_not_part_of_the_request_payload() {
		let phases = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
		let capture = phases.clone();
		let observer: crate::types::OnStreamObservation = std::sync::Arc::new(move |phase| capture.lock().unwrap().push(phase.into()));
		observe_sse_payload(Some(&observer), r#"{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"PRIVATE"}}]}}]}"#);
		observe_sse_payload(Some(&observer), r#"{"error":{"message":"PRIVATE"}}"#);
		assert_eq!(*phases.lock().unwrap(), vec!["raw_event", "tool", "raw_event", "terminal"]);
		let model = base_model();
		let ctx = context(vec![user_text("hi")]);
		let compat = get_compat(&model);
		let mut options = keyed_options();
		let before = build_params(&model, &ctx, Some(&options), &compat, &"short".into(), None).unwrap();
		options.stream.on_stream_observation = Some(observer);
		options.stream.on_usage_observation = Some(std::sync::Arc::new(|_, _| Box::pin(async {})));
		assert_eq!(build_params(&model, &ctx, Some(&options), &compat, &"short".into(), None).unwrap(), before);
	}

	#[tokio::test]
	async fn cancellation_closes_a_stalled_stream_and_preserves_partial_output() {
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
		let address = listener.local_addr().unwrap();
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut request = Vec::new();
			loop {
				let mut buffer = [0u8; 4096];
				let count = socket.read(&mut buffer).await.unwrap();
				assert_ne!(count, 0, "client closed before sending the request");
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
						break;
					}
				}
			}
			let body = "data: {\"choices\":[{\"delta\":{\"content\":\"partial text\"}}]}\n\n";
			let response = format!(
				"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{body}\r\n",
				body.len()
			);
			socket.write_all(response.as_bytes()).await.unwrap();
			socket.flush().await.unwrap();
			// Keep the body open without another chunk or [DONE]. Cancellation must
			// release the connection without requiring further provider activity.
			let mut buffer = [0u8; 1];
			assert_eq!(socket.read(&mut buffer).await.unwrap(), 0);
		});
		let mut model = base_model();
		model.base_url = format!("http://{address}");
		let signal = tokio_util::sync::CancellationToken::new();
		let mut options = keyed_options();
		options.stream.signal = Some(signal.clone());
		let stream = stream_openai_completions(&model, &context(vec![user_text("hi")]), Some(options));
		let mut aborted = false;
		while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
			.await
			.expect("stalled stream did not respond to cancellation")
		{
			match event {
				AssistantMessageEvent::TextDelta { .. } => signal.cancel(),
				AssistantMessageEvent::Error { reason, error } => {
					assert_eq!(reason, "aborted");
					assert_eq!(error.stop_reason, "aborted");
					assert_eq!(error.content, vec![ContentBlock::Text(TextContent::new("partial text"))]);
					aborted = true;
				}
				AssistantMessageEvent::Done { .. } => panic!("cancelled stream reported success"),
				_ => {}
			}
		}
		assert!(aborted);
		tokio::time::timeout(std::time::Duration::from_secs(5), server)
			.await
			.expect("cancelled stream kept the HTTP connection open")
			.unwrap();
	}

	#[tokio::test]
	async fn in_stream_error_event_fails_the_stream_and_keeps_partial_output() {
		// SDK `Stream.fromSSEResponse` throws on `data.error`, so a mid-stream error event
		// must produce an `error` event, not a silent `done`.
		let body = concat!(
			"data: {\"id\":\"chatcmpl-1\",\"model\":\"repro-model\",\"choices\":[{\"delta\":{\"content\":\"partial text\"}}]}\n\n",
			"data: {\"error\":{\"message\":\"boom\",\"metadata\":{\"raw\":\"RAW DETAIL\"}}}\n\n",
			"data: [DONE]\n\n"
		);
		let (address, server) = serve_http("200 OK", body).await;
		let mut model = base_model();
		model.base_url = format!("http://{address}");
		let options = keyed_options();
		let stream = stream_openai_completions(&model, &context(vec![user_text("hi")]), Some(options));
		let mut events = Vec::new();
		while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
			.await
			.expect("stream event timeout")
		{
			events.push(event);
		}
		let error = events
			.iter()
			.find_map(|event| match event {
				AssistantMessageEvent::Error { reason, error } => Some((reason.clone(), error.clone())),
				_ => None,
			})
			.expect("error event");
		assert_eq!(error.0, "error");
		assert_eq!(error.1.stop_reason, "error");
		// A4-10: the partial text streamed before the failure is preserved, and the
		// OpenRouter raw metadata is appended (`error.error.metadata.raw`).
		assert_eq!(error.1.error_message.as_deref(), Some("boom\nRAW DETAIL"));
		assert_eq!(
			error.1.content,
			vec![ContentBlock::Text(TextContent::new("partial text"))]
		);
		assert!(events.iter().all(|event| !matches!(event, AssistantMessageEvent::Done { .. })));
		tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
	}

	#[tokio::test]
	async fn error_status_reports_the_nested_sdk_message() {
		// `APIError.generate` unwraps `body.error`, so the message is not the whole JSON body.
		let body = "{\"error\":{\"message\":\"Incorrect API key provided\"}}";
		let (address, server) = serve_http("400 Bad Request", body).await;
		let mut model = base_model();
		model.base_url = format!("http://{address}");
		let options = keyed_options();
		let stream = stream_openai_completions(&model, &context(vec![user_text("hi")]), Some(options));
		let mut events = Vec::new();
		while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
			.await
			.expect("stream event timeout")
		{
			events.push(event);
		}
		let error = events
			.iter()
			.find_map(|event| match event {
				AssistantMessageEvent::Error { error, .. } => Some(error.clone()),
				_ => None,
			})
			.expect("error event");
		assert_eq!(error.error_message.as_deref(), Some("400 Incorrect API key provided"));
		assert!(error.content.is_empty());
		tokio::time::timeout(std::time::Duration::from_secs(5), server).await.unwrap().unwrap();
	}

	#[test]
	fn header_timeout_defaults_to_the_sdk_ten_minutes() {
		assert_eq!(resolve_header_timeout(None), std::time::Duration::from_millis(600_000));
		assert_eq!(resolve_header_timeout(Some(1_500.0)), std::time::Duration::from_millis(1_500));
		assert_eq!(resolve_header_timeout(Some(0.0)), std::time::Duration::from_millis(0));
	}

	// ------------------------------------------------------------------
	// detectCompat / getCompat
	// ------------------------------------------------------------------

	#[test]
	fn detects_non_standard_providers_by_url() {
		let mut model = base_model();
		model.provider = "zai".to_string();
		model.base_url = "https://api.z.ai/v1".to_string();
		let compat = get_compat(&model);
		assert!(!compat.supports_store);
		assert!(!compat.supports_developer_role);
		assert!(!compat.supports_reasoning_effort);
		assert_eq!(compat.thinking_format, "zai");
		assert_eq!(compat.max_tokens_field, "max_completion_tokens");
		assert!(compat.supports_strict_mode);
	}

	#[test]
	fn detects_deepseek_thinking_format_and_reasoning_content() {
		let mut model = base_model();
		model.provider = "deepseek".to_string();
		model.base_url = "https://api.deepseek.com/v1".to_string();
		let compat = get_compat(&model);
		assert_eq!(compat.thinking_format, "deepseek");
		assert!(compat.requires_reasoning_content_on_assistant_messages);
		assert!(!compat.supports_store);
	}

	#[test]
	fn detects_openrouter_and_anthropic_cache_control_format() {
		let mut model = base_model();
		model.provider = "openrouter".to_string();
		model.base_url = "https://openrouter.ai/api/v1".to_string();
		model.id = "anthropic/claude-sonnet-4".to_string();
		let compat = get_compat(&model);
		assert_eq!(compat.thinking_format, "openrouter");
		assert_eq!(compat.cache_control_format.as_deref(), Some("anthropic"));
		assert!(compat.supports_strict_mode);
		assert!(compat.supports_long_cache_retention);
	}

	#[test]
	fn detects_moonshot_max_tokens_and_no_strict_mode() {
		let mut model = base_model();
		model.provider = "moonshotai".to_string();
		model.base_url = "https://api.moonshot.ai/v1".to_string();
		let compat = get_compat(&model);
		assert_eq!(compat.max_tokens_field, "max_tokens");
		assert!(!compat.supports_strict_mode);
		assert!(!compat.supports_reasoning_effort);
		assert_eq!(compat.thinking_format, "openai");
	}

	#[test]
	fn detects_cloudflare_gateway_conservative_fields() {
		let mut model = base_model();
		model.provider = "cloudflare-ai-gateway".to_string();
		model.base_url = "https://gateway.ai.cloudflare.com/v1/a/g/compat".to_string();
		let compat = get_compat(&model);
		assert_eq!(compat.max_tokens_field, "max_tokens");
		assert!(!compat.supports_store);
		assert!(!compat.supports_reasoning_effort);
		assert!(!compat.supports_strict_mode);
		assert!(!compat.supports_long_cache_retention);
	}

	#[test]
	fn explicit_compat_overrides_detected_values() {
		let model = compat_model(OpenAICompletionsCompat {
			supports_store: Some(true),
			supports_developer_role: Some(false),
			max_tokens_field: Some("max_tokens".to_string()),
			thinking_format: Some("qwen".to_string()),
			zai_tool_stream: Some(true),
			send_session_affinity_headers: Some(true),
			..Default::default()
		});
		let compat = get_compat(&model);
		assert!(compat.supports_store);
		assert!(!compat.supports_developer_role);
		assert_eq!(compat.max_tokens_field, "max_tokens");
		assert_eq!(compat.thinking_format, "qwen");
		assert!(compat.zai_tool_stream);
		assert!(compat.send_session_affinity_headers);
	}

	#[test]
	fn openrouter_routing_is_never_inherited_from_compat() {
		let model = compat_model(OpenAICompletionsCompat {
			open_router_routing: Some(crate::types::OpenRouterRouting {
				only: Some(vec!["anthropic".to_string()]),
				..Default::default()
			}),
			..Default::default()
		});
		let compat = get_compat(&model);
		assert_eq!(compat.open_router_routing, json!({}));
	}

	// ------------------------------------------------------------------
	// buildParams
	// ------------------------------------------------------------------

	fn build(model: &Model, context: &Context, options: Option<&OpenAICompletionsOptions>) -> Value {
		let compat = get_compat(model);
		let cache_retention = resolve_cache_retention(
			options.and_then(|options| options.stream.cache_retention.as_ref()),
		);
		let cache_control = get_compat_cache_control(&compat, &cache_retention);
		build_params(model, context, options, &compat, &cache_retention, cache_control.as_ref()).expect("params")
	}

	#[test]
	fn params_include_usage_in_streaming_and_store_defaults() {
		let mut model = base_model();
		model.provider = "openai".to_string();
		model.base_url = "https://api.openai.com/v1".to_string();
		let params = build(&model, &context(vec![user_text("hi")]), None);
		assert_eq!(params["model"], json!("repro-model"));
		assert_eq!(params["stream"], json!(true));
		assert_eq!(params["stream_options"]["include_usage"], json!(true));
		assert_eq!(params["store"], json!(false));
		assert!(params.get("prompt_cache_key").is_none());
		assert!(params.get("prompt_cache_retention").is_none());
	}

	#[test]
	fn params_use_max_completion_tokens_by_default() {
		let mut model = base_model();
		model.provider = "openai".to_string();
		model.base_url = "https://api.openai.com/v1".to_string();
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.stream.max_tokens = Some(1234.0);
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["max_completion_tokens"], json!(1234));
		assert!(params.get("max_tokens").is_none());
	}

	#[test]
	fn params_use_max_tokens_when_compat_requires_it() {
		let model = compat_model(OpenAICompletionsCompat {
			max_tokens_field: Some("max_tokens".to_string()),
			..Default::default()
		});
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.stream.max_tokens = Some(12.0);
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["max_tokens"], json!(12));
		assert!(params.get("max_completion_tokens").is_none());
	}

	#[test]
	fn params_omit_tools_for_empty_tools_array_without_history() {
		let mut model = base_model();
		model.provider = "openai".to_string();
		model.base_url = "https://api.openai.com/v1".to_string();
		let mut ctx = context(vec![user_text("hi")]);
		ctx.tools = Some(Vec::new());
		let params = build(&model, &ctx, None);
		assert!(params.get("tools").is_none());
	}

	#[test]
	fn params_emit_empty_tools_array_when_history_has_tool_calls() {
		let mut model = base_model();
		model.provider = "openai".to_string();
		model.base_url = "https://api.openai.com/v1".to_string();
		let tool_call = ContentBlock::ToolCall(ToolCall::new("t1", "noop", Map::new()));
		let messages = vec![
			user_text("use the tool"),
			assistant_message(vec![tool_call], "toolUse"),
			Message::tool_result(crate::types::ToolResultMessage::new(
				"t1",
				"noop",
				vec![ImageOrTextContent::Text(TextContent::new("done"))],
				false,
				3,
			)),
		];
		let params = build(&model, &context(messages), None);
		assert_eq!(params["tools"], json!([]));
	}

	#[test]
	fn params_add_zai_tool_stream_for_configured_provider() {
		let model = compat_model(OpenAICompletionsCompat {
			zai_tool_stream: Some(true),
			..Default::default()
		});
		let mut ctx = context(vec![user_text("hi")]);
		ctx.tools = Some(vec![Tool {
			name: "read".to_string(),
			description: "Read a file".to_string(),
			parameters: json!({ "type": "object" }),
		}]);
		let params = build(&model, &ctx, None);
		assert_eq!(params["tool_stream"], json!(true));
		assert_eq!(params["tools"][0]["type"], json!("function"));
		assert_eq!(params["tools"][0]["function"]["strict"], json!(false));
	}

	#[test]
	fn params_omit_strict_when_provider_rejects_it() {
		let model = compat_model(OpenAICompletionsCompat {
			supports_strict_mode: Some(false),
			..Default::default()
		});
		let mut ctx = context(vec![user_text("hi")]);
		ctx.tools = Some(vec![Tool {
			name: "read".to_string(),
			description: "Read a file".to_string(),
			parameters: json!({ "type": "object" }),
		}]);
		let params = build(&model, &ctx, None);
		assert!(params["tools"][0]["function"].get("strict").is_none());
	}

	#[test]
	fn params_prompt_cache_key_requires_openai_url_and_retention() {
		let mut model = base_model();
		model.provider = "openai".to_string();
		model.base_url = "https://api.openai.com/v1".to_string();
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.stream.session_id = Some("session-123".to_string());
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["prompt_cache_key"], json!("session-123"));
		assert!(params.get("prompt_cache_retention").is_none());

		options.stream.cache_retention = Some("long".to_string());
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["prompt_cache_key"], json!("session-123"));
		assert_eq!(params["prompt_cache_retention"], json!("24h"));

		options.stream.cache_retention = Some("none".to_string());
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert!(params.get("prompt_cache_key").is_none());
		assert!(params.get("prompt_cache_retention").is_none());
	}

	#[test]
	fn params_omit_prompt_cache_key_for_other_urls_without_long_retention() {
		let model = compat_model(OpenAICompletionsCompat {
			supports_long_cache_retention: Some(false),
			..Default::default()
		});
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.stream.session_id = Some("session-proxy".to_string());
		options.stream.cache_retention = Some("long".to_string());
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert!(params.get("prompt_cache_key").is_none());
		assert!(params.get("prompt_cache_retention").is_none());
	}

	#[test]
	fn params_map_thinking_formats() {
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.reasoning_effort = Some("medium".to_string());

		let zai = compat_model(OpenAICompletionsCompat {
			thinking_format: Some("zai".to_string()),
			..Default::default()
		});
		let params = build(&zai, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["enable_thinking"], json!(true));
		assert!(params.get("reasoning_effort").is_none());

		let qwen = compat_model(OpenAICompletionsCompat {
			thinking_format: Some("qwen".to_string()),
			..Default::default()
		});
		let params = build(&qwen, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["enable_thinking"], json!(true));

		let qwen_template = compat_model(OpenAICompletionsCompat {
			thinking_format: Some("qwen-chat-template".to_string()),
			..Default::default()
		});
		let params = build(&qwen_template, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["chat_template_kwargs"]["enable_thinking"], json!(true));
		assert_eq!(params["chat_template_kwargs"]["preserve_thinking"], json!(true));

		let deepseek = compat_model(OpenAICompletionsCompat {
			thinking_format: Some("deepseek".to_string()),
			..Default::default()
		});
		let params = build(&deepseek, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["thinking"]["type"], json!("enabled"));
		assert_eq!(params["reasoning_effort"], json!("medium"));

		let mut no_effort = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		no_effort.reasoning_enabled = Some(false);
		let params = build(&deepseek, &context(vec![user_text("hi")]), Some(&no_effort));
		assert_eq!(params["thinking"]["type"], json!("disabled"));
		assert!(params.get("reasoning_effort").is_none());
	}

	#[test]
	fn params_use_thinking_level_map_for_deepseek_effort() {
		let mut model = compat_model(OpenAICompletionsCompat {
			thinking_format: Some("deepseek".to_string()),
			..Default::default()
		});
		model.thinking_level_map = Some(
			[("medium".to_string(), Some("deepseek-medium".to_string()))]
				.into_iter()
				.collect(),
		);
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.reasoning_effort = Some("medium".to_string());
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["reasoning_effort"], json!("deepseek-medium"));
	}

	#[test]
	fn params_openrouter_reasoning_variants() {
		let mut model = base_model();
		model.provider = "openrouter".to_string();
		model.base_url = "https://openrouter.ai/api/v1".to_string();

		let mut effort = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		effort.reasoning_effort = Some("high".to_string());
		let params = build(&model, &context(vec![user_text("hi")]), Some(&effort));
		assert_eq!(params["reasoning"]["effort"], json!("high"));

		let mut enabled = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		enabled.reasoning_enabled = Some(true);
		let params = build(&model, &context(vec![user_text("hi")]), Some(&enabled));
		assert_eq!(params["reasoning"]["enabled"], json!(true));

		let mut disabled = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		disabled.reasoning_enabled = Some(false);
		let params = build(&model, &context(vec![user_text("hi")]), Some(&disabled));
		assert_eq!(params["reasoning"]["effort"], json!("none"));
	}

	#[test]
	fn params_openrouter_disabled_reasoning_is_omitted_when_off_is_null() {
		let mut model = base_model();
		model.provider = "openrouter".to_string();
		model.base_url = "https://openrouter.ai/api/v1".to_string();
		model.thinking_level_map = Some([("off".to_string(), None)].into_iter().collect());
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.reasoning_enabled = Some(false);
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert!(params.get("reasoning").is_none());
	}

	#[test]
	fn params_reasoning_effort_for_plain_openai_compat() {
		let mut model = base_model();
		model.provider = "openai".to_string();
		model.base_url = "https://api.openai.com/v1".to_string();
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.reasoning_effort = Some("low".to_string());
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["reasoning_effort"], json!("low"));
	}

	#[test]
	fn params_off_level_uses_none_when_not_mapped() {
		let mut model = base_model();
		model.provider = "openai".to_string();
		model.base_url = "https://api.openai.com/v1".to_string();
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.reasoning_enabled = Some(false);
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["reasoning_effort"], json!("none"));

		model.thinking_level_map = Some([("off".to_string(), None)].into_iter().collect());
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert!(params.get("reasoning_effort").is_none());
	}

	#[test]
	fn params_include_openrouter_routing_from_model_compat() {
		let mut model = base_model();
		model.provider = "openrouter".to_string();
		model.base_url = "https://openrouter.ai/api/v1".to_string();
		model.compat = Some(crate::types::Compat::Completions(OpenAICompletionsCompat {
			open_router_routing: Some(crate::types::OpenRouterRouting {
				only: Some(vec!["anthropic".to_string()]),
				..Default::default()
			}),
			..Default::default()
		}));
		let params = build(&model, &context(vec![user_text("hi")]), None);
		assert_eq!(params["provider"]["only"], json!(["anthropic"]));
	}

	#[test]
	fn params_include_vercel_gateway_routing_only_when_present() {
		let mut model = base_model();
		model.base_url = "https://ai-gateway.vercel.sh/v1".to_string();
		model.compat = Some(crate::types::Compat::Completions(OpenAICompletionsCompat {
			vercel_gateway_routing: Some(crate::types::VercelGatewayRouting {
				order: Some(vec!["anthropic".to_string()]),
				..Default::default()
			}),
			..Default::default()
		}));
		let params = build(&model, &context(vec![user_text("hi")]), None);
		assert_eq!(params["providerOptions"]["gateway"]["order"], json!(["anthropic"]));
		assert!(params["providerOptions"]["gateway"].get("only").is_none());
	}

	#[test]
	fn params_tool_choice_is_forwarded_verbatim() {
		let model = base_model();
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.tool_choice = Some(json!("required"));
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["tool_choice"], json!("required"));
	}

	#[test]
	fn params_temperature_is_forwarded() {
		let model = base_model();
		let mut options = OpenAICompletionsOptions::from_base(&StreamOptions::default());
		options.stream.temperature = Some(0.25);
		let params = build(&model, &context(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["temperature"], json!(0.25));
	}
}

#[cfg(test)]
mod message_tests {
	use super::tests_support::*;
	use super::*;
	use crate::types::{ImageContent, ModelCost, OpenAICompletionsCompat};

	fn full_compat() -> ResolvedCompat {
		ResolvedCompat {
			supports_store: true,
			supports_developer_role: true,
			supports_reasoning_effort: true,
			supports_usage_in_streaming: true,
			max_tokens_field: "max_completion_tokens".to_string(),
			requires_tool_result_name: false,
			requires_assistant_after_tool_result: false,
			requires_thinking_as_text: false,
			requires_reasoning_content_on_assistant_messages: false,
			thinking_format: "openai".to_string(),
			open_router_routing: json!({}),
			vercel_gateway_routing: json!({}),
			zai_tool_stream: false,
			supports_strict_mode: true,
			cache_control_format: None,
			send_session_affinity_headers: false,
			supports_long_cache_retention: true,
		}
	}

	#[test]
	fn replays_thinking_into_the_recorded_field() {
		let mut thinking = ThinkingContent::new("step by step");
		thinking.thinking_signature = Some("reasoning".to_string());
		let messages = vec![
			user_text("hello"),
			assistant_message(
				vec![ContentBlock::Thinking(thinking), ContentBlock::Text(TextContent::new("answer"))],
				"stop",
			),
		];
		let params = convert_messages(&base_model(), &context(messages), &full_compat()).unwrap();
		assert_eq!(params[1]["content"], json!("answer"));
		assert_eq!(params[1]["reasoning"], json!("step by step"));
	}

	#[test]
	fn keeps_unsigned_thinking_as_text() {
		let messages = vec![assistant_message(
			vec![
				ContentBlock::Thinking(ThinkingContent::new("unsigned reasoning")),
				ContentBlock::Text(TextContent::new("answer")),
			],
			"stop",
		)];
		let params = convert_messages(&base_model(), &context(messages), &full_compat()).unwrap();
		assert!(params[0].get("reasoning_content").is_none());
		assert_eq!(params[0]["content"], json!("unsigned reasoning\n\nanswer"));
	}

	#[test]
	fn uses_reasoning_content_when_provider_requires_it() {
		let mut compat = full_compat();
		compat.requires_reasoning_content_on_assistant_messages = true;
		let messages = vec![assistant_message(
			vec![
				ContentBlock::Thinking(ThinkingContent::new("unsigned reasoning")),
				ContentBlock::Text(TextContent::new("answer")),
			],
			"stop",
		)];
		let params = convert_messages(&base_model(), &context(messages), &compat).unwrap();
		assert_eq!(params[0]["reasoning_content"], json!("unsigned reasoning"));
		assert_eq!(params[0]["content"], json!("answer"));
	}

	#[test]
	fn replays_signed_thinking_alongside_a_tool_call() {
		let mut thinking = ThinkingContent::new("deciding to call a tool");
		thinking.thinking_signature = Some("reasoning".to_string());
		let messages = vec![assistant_message(
			vec![
				ContentBlock::Thinking(thinking),
				ContentBlock::ToolCall(ToolCall::new("call-1", "search", Map::new())),
			],
			"toolUse",
		)];
		let params = convert_messages(&base_model(), &context(messages), &full_compat()).unwrap();
		assert_eq!(params[0]["reasoning"], json!("deciding to call a tool"));
		assert_eq!(params[0]["tool_calls"][0]["id"], json!("call-1"));
		assert_eq!(params[0]["tool_calls"][0]["type"], json!("function"));
		assert_eq!(params[0]["tool_calls"][0]["function"]["arguments"], json!("{}"));
	}

	#[test]
	fn thinking_as_text_replay_uses_text_parts() {
		let mut compat = full_compat();
		compat.requires_thinking_as_text = true;
		let messages = vec![assistant_message(
			vec![
				ContentBlock::Thinking(ThinkingContent::new("internal reasoning")),
				ContentBlock::Text(TextContent::new("visible answer")),
			],
			"stop",
		)];
		let params = convert_messages(&base_model(), &context(messages), &compat).unwrap();
		assert_eq!(
			params[0]["content"],
			json!([
				{ "type": "text", "text": "internal reasoning" },
				{ "type": "text", "text": "visible answer" },
			])
		);
	}

	#[test]
	fn thinking_as_text_replay_with_thinking_only() {
		let mut compat = full_compat();
		compat.requires_thinking_as_text = true;
		let messages = vec![assistant_message(
			vec![ContentBlock::Thinking(ThinkingContent::new("internal reasoning"))],
			"stop",
		)];
		let params = convert_messages(&base_model(), &context(messages), &compat).unwrap();
		assert_eq!(params[0]["content"], json!([{ "type": "text", "text": "internal reasoning" }]));
	}

	#[test]
	fn system_prompt_uses_system_role_without_developer_support() {
		let mut compat = full_compat();
		compat.supports_developer_role = false;
		let mut ctx = context(vec![user_text("hi")]);
		ctx.system_prompt = Some("System prompt".to_string());
		let params = convert_messages(&base_model(), &ctx, &compat).unwrap();
		assert_eq!(params[0]["role"], json!("system"));
		assert_eq!(params[0]["content"], json!("System prompt"));

		let params = convert_messages(&base_model(), &ctx, &full_compat()).unwrap();
		assert_eq!(params[0]["role"], json!("developer"));
	}

	#[test]
	fn inserts_assistant_bridge_before_user_after_tool_result() {
		let mut compat = full_compat();
		compat.requires_assistant_after_tool_result = true;
		let messages = vec![
			assistant_message(
				vec![ContentBlock::ToolCall(ToolCall::new("t1", "noop", Map::new()))],
				"toolUse",
			),
			Message::tool_result(crate::types::ToolResultMessage::new(
				"t1",
				"noop",
				vec![ImageOrTextContent::Text(TextContent::new("done"))],
				false,
				3,
			)),
			user_text("continue"),
		];
		let params = convert_messages(&base_model(), &context(messages), &compat).unwrap();
		let roles: Vec<&str> = params
			.iter()
			.map(|message| message["role"].as_str().unwrap_or_default())
			.collect();
		assert_eq!(roles, vec!["assistant", "tool", "assistant", "user"]);
		assert_eq!(params[2]["content"], json!("I have processed the tool results."));
	}

	#[test]
	fn assistant_content_is_null_without_requires_assistant_after_tool_result() {
		let messages = vec![assistant_message(
			vec![ContentBlock::ToolCall(ToolCall::new("t1", "noop", Map::new()))],
			"toolUse",
		)];
		let params = convert_messages(&base_model(), &context(messages), &full_compat()).unwrap();
		assert_eq!(params[0]["content"], Value::Null);
	}

	#[test]
	fn skips_assistant_messages_without_content_or_tool_calls() {
		let messages = vec![user_text("hi"), assistant_message(Vec::new(), "stop"), user_text("again")];
		let params = convert_messages(&base_model(), &context(messages), &full_compat()).unwrap();
		assert_eq!(params.len(), 2);
		assert_eq!(params[0]["role"], json!("user"));
		assert_eq!(params[1]["content"], json!("again"));
	}

	#[test]
	fn tool_result_name_only_when_required() {
		let messages = vec![
			assistant_message(
				vec![ContentBlock::ToolCall(ToolCall::new("t1", "read", Map::new()))],
				"toolUse",
			),
			Message::tool_result(crate::types::ToolResultMessage::new(
				"t1",
				"read",
				vec![ImageOrTextContent::Text(TextContent::new("file contents"))],
				false,
				3,
			)),
		];
		let params = convert_messages(&base_model(), &context(messages.clone()), &full_compat()).unwrap();
		assert!(params[1].get("name").is_none());
		assert_eq!(params[1]["role"], json!("tool"));
		assert_eq!(params[1]["content"], json!("file contents"));
		assert_eq!(params[1]["tool_call_id"], json!("t1"));

		let mut compat = full_compat();
		compat.requires_tool_result_name = true;
		let params = convert_messages(&base_model(), &context(messages), &compat).unwrap();
		assert_eq!(params[1]["name"], json!("read"));
	}

	#[test]
	fn tool_result_with_empty_text_keeps_empty_string() {
		let messages = vec![
			assistant_message(
				vec![ContentBlock::ToolCall(ToolCall::new("tool-1", "bash", Map::new()))],
				"toolUse",
			),
			Message::tool_result(crate::types::ToolResultMessage::new(
				"tool-1",
				"bash",
				vec![ImageOrTextContent::Text(TextContent::new(""))],
				false,
				3,
			)),
		];
		let params = convert_messages(&base_model(), &context(messages), &full_compat()).unwrap();
		let tool_message = params
			.iter()
			.find(|message| message["role"] == json!("tool"))
			.expect("tool message");
		assert_eq!(tool_message["content"], json!(""));
	}

	#[test]
	fn batches_tool_result_images_after_consecutive_tool_results() {
		let mut model = base_model();
		model.provider = "openai".to_string();
		model.input = vec![InputModality::Text, InputModality::Image];

		let messages = vec![
			user_text("Read the images"),
			assistant_message(
				vec![
					ContentBlock::ToolCall(ToolCall::new("tool-1", "read", Map::new())),
					ContentBlock::ToolCall(ToolCall::new("tool-2", "read", Map::new())),
				],
				"toolUse",
			),
			tool_result_with_image("tool-1"),
			tool_result_with_image("tool-2"),
		];
		let params = convert_messages(&model, &context(messages), &full_compat()).unwrap();
		let roles: Vec<&str> = params
			.iter()
			.map(|message| message["role"].as_str().unwrap_or_default())
			.collect();
		assert_eq!(roles, vec!["user", "assistant", "tool", "tool", "user"]);

		let image_message = params.last().unwrap();
		let image_parts: Vec<&Value> = image_message["content"]
			.as_array()
			.unwrap()
			.iter()
			.filter(|part| part["type"] == json!("image_url"))
			.collect();
		assert_eq!(image_parts.len(), 2);
		assert_eq!(image_message["content"][0]["text"], json!("Attached image(s) from tool result:"));
	}

	#[test]
	fn tool_result_images_are_skipped_for_text_only_models() {
		let messages = vec![
			assistant_message(
				vec![ContentBlock::ToolCall(ToolCall::new("tool-1", "read", Map::new()))],
				"toolUse",
			),
			tool_result_with_image("tool-1"),
		];
		let params = convert_messages(&base_model(), &context(messages), &full_compat()).unwrap();
		assert_eq!(params.len(), 2);
		assert_eq!(params[1]["role"], json!("tool"));
	}

	#[test]
	fn user_message_with_only_images_keeps_image_parts() {
		let mut model = base_model();
		model.input = vec![InputModality::Text, InputModality::Image];
		let user = Message::user(crate::types::UserMessage::new(
			UserContent::Blocks(vec![
				ImageOrTextContent::Image(ImageContent::new("ZmFrZQ==", "image/png")),
			]),
			1,
		));
		let params = convert_messages(&model, &context(vec![user]), &full_compat()).unwrap();
		assert_eq!(params.len(), 1);
		assert_eq!(params[0]["role"], json!("user"));
		assert_eq!(
			params[0]["content"],
			json!([{ "type": "image_url", "image_url": { "url": "data:image/png;base64,ZmFrZQ==" } }])
		);
	}

	#[test]
	fn normalize_tool_call_id_truncates_pipe_separated_ids() {
		let mut model = base_model();
		// Cross-provider replay invokes ID normalization; same-source IDs are kept.
		model.provider = "openai".to_string();
		let long_id = format!("call+/{}|tail", "a".repeat(50));
		let expected_id = format!("call__{}", "a".repeat(34));
		let messages = vec![
			assistant_message(
				vec![ContentBlock::ToolCall(ToolCall::new(&long_id, "echo", Map::new()))],
				"toolUse",
			),
			Message::tool_result(crate::types::ToolResultMessage::new(
				long_id.clone(),
				"echo",
				vec![ImageOrTextContent::Text(TextContent::new("hi"))],
				false,
				3,
			)),
		];
		let params = convert_messages(&model, &context(messages), &full_compat()).unwrap();
		assert_eq!(params[0]["tool_calls"][0]["id"], json!(expected_id));
		assert_eq!(params[1]["tool_call_id"], json!(expected_id));
	}

	#[test]
	fn normalize_tool_call_id_truncates_openai_ids_without_pipes() {
		let mut model = base_model();
		model.provider = "openai".to_string();
		let long_id = "b".repeat(50);
		let messages = vec![
			assistant_message(
				vec![ContentBlock::ToolCall(ToolCall::new(&long_id, "echo", Map::new()))],
				"toolUse",
			),
			Message::tool_result(crate::types::ToolResultMessage::new(
				long_id.clone(),
				"echo",
				vec![ImageOrTextContent::Text(TextContent::new("hi"))],
				false,
				3,
			)),
		];
		let params = convert_messages(&model, &context(messages), &full_compat()).unwrap();
		assert_eq!(params[0]["tool_calls"][0]["id"], json!("b".repeat(40)));
		assert_eq!(params[1]["tool_call_id"], json!("b".repeat(40)));
	}

	#[test]
	fn other_providers_keep_long_ids() {
		let mut model = base_model();
		model.provider = "openrouter".to_string();
		let long_id = "c".repeat(50);
		let messages = vec![assistant_message(
			vec![ContentBlock::ToolCall(ToolCall::new(&long_id, "echo", Map::new()))],
			"toolUse",
		)];
		let params = convert_messages(&model, &context(messages), &full_compat()).unwrap();
		assert_eq!(params[0]["tool_calls"][0]["id"], json!("c".repeat(50)));
	}

	#[test]
	fn convert_messages_errors_for_foreign_compaction_checkpoint() {
		let mut user = crate::types::UserMessage::new(UserContent::Text("marker".to_string()), 2);
		user.provider_context = Some(crate::compaction::ProviderCompactionCheckpoint {
			version: 1,
			provider: "openai".to_string(),
			api: "openai-responses".to_string(),
			model: "other-model".to_string(),
			base_url: "https://api.openai.com/v1".to_string(),
			endpoint: None,
			items: vec![Map::new()],
			estimated_tokens: 1.0,
		});
		let error = convert_messages(&base_model(), &context(vec![Message::User(user)]), &full_compat()).unwrap_err();
		assert!(error.contains("rebuild context from the session transcript"));
	}

	// ------------------------------------------------------------------
	// cache control markers
	// ------------------------------------------------------------------

	#[test]
	fn anthropic_cache_control_marks_system_tool_and_last_message() {
		let mut compat = full_compat();
		compat.cache_control_format = Some("anthropic".to_string());
		let mut model = base_model();
		model.provider = "openrouter".to_string();
		model.base_url = "https://example.com/v1".to_string();
		model.id = "anthropic/claude-sonnet-4".to_string();

		let cache_retention = resolve_cache_retention(None);
		let cache_control = get_compat_cache_control(&compat, &cache_retention).expect("cache control");
		let mut ctx = context(vec![user_text("Hello")]);
		ctx.system_prompt = Some("System prompt".to_string());
		ctx.tools = Some(vec![Tool {
			name: "read".to_string(),
			description: "Read a file".to_string(),
			parameters: json!({ "type": "object" }),
		}]);
		let params = build_params(&model, &ctx, None, &compat, &cache_retention, Some(&cache_control)).unwrap();

		assert_eq!(params["messages"][0]["content"][0]["cache_control"], json!({ "type": "ephemeral" }));
		assert_eq!(params["tools"][0]["cache_control"], json!({ "type": "ephemeral" }));
		let last_message = params["messages"].as_array().unwrap().last().unwrap();
		assert_eq!(last_message["role"], json!("user"));
		assert_eq!(last_message["content"][0]["cache_control"], json!({ "type": "ephemeral" }));
	}

	#[test]
	fn anthropic_cache_control_advances_to_tool_result() {
		let mut compat = full_compat();
		compat.cache_control_format = Some("anthropic".to_string());
		let mut model = base_model();
		model.provider = "prime-inference".to_string();
		model.base_url = "https://api.pinference.ai/api/v1".to_string();
		model.id = "anthropic/claude-haiku-4.5".to_string();

		let cache_retention = resolve_cache_retention(None);
		let cache_control = get_compat_cache_control(&compat, &cache_retention).expect("cache control");
		let messages = vec![
			user_text("Read the file"),
			assistant_message(
				vec![ContentBlock::ToolCall(ToolCall::new("tool-1", "read", Map::new()))],
				"toolUse",
			),
			Message::tool_result(crate::types::ToolResultMessage::new(
				"tool-1",
				"read",
				vec![ImageOrTextContent::Text(TextContent::new("file contents"))],
				false,
				3,
			)),
		];
		let params = build_params(
			&model,
			&context(messages),
			None,
			&compat,
			&cache_retention,
			Some(&cache_control),
		)
		.unwrap();
		let last_message = params["messages"].as_array().unwrap().last().unwrap();
		assert_eq!(last_message["role"], json!("tool"));
		assert_eq!(
			last_message["content"][0]["cache_control"],
			json!({ "type": "ephemeral" })
		);
	}

	#[test]
	fn long_retention_cache_control_sets_one_hour_ttl() {
		let mut compat = full_compat();
		compat.cache_control_format = Some("anthropic".to_string());
		let cache_retention = resolve_cache_retention(Some(&"long".to_string()));
		let cache_control = get_compat_cache_control(&compat, &cache_retention).expect("cache control");
		assert_eq!(cache_control.ttl.as_deref(), Some("1h"));
		assert_eq!(cache_control.to_value(), json!({ "type": "ephemeral", "ttl": "1h" }));

		compat.supports_long_cache_retention = false;
		let cache_control = get_compat_cache_control(&compat, &cache_retention).expect("cache control");
		assert!(cache_control.ttl.is_none());
		assert_eq!(cache_control.to_value(), json!({ "type": "ephemeral" }));
	}

	#[test]
	fn cache_control_is_absent_when_retention_is_none() {
		let mut compat = full_compat();
		compat.cache_control_format = Some("anthropic".to_string());
		assert!(get_compat_cache_control(&compat, &"none".to_string()).is_none());
		assert!(get_compat_cache_control(&full_compat(), &"short".to_string()).is_none());
	}

	// ------------------------------------------------------------------
	// usage / stop reason mapping
	// ------------------------------------------------------------------

	#[test]
	fn chunk_usage_normalizes_cache_write_out_of_cache_read() {
		let model = base_model();
		let raw = json!({
			"prompt_tokens": 100,
			"completion_tokens": 10,
			"prompt_tokens_details": { "cached_tokens": 80, "cache_write_tokens": 80 },
			"completion_tokens_details": { "reasoning_tokens": 0 },
		});
		let usage = parse_chunk_usage(&raw, &model, None);
		assert_eq!(usage.input, 20.0);
		assert_eq!(usage.output, 10.0);
		assert_eq!(usage.cache_read, 0.0);
		assert_eq!(usage.cache_write, 80.0);
		assert_eq!(usage.total_tokens, 110.0);
	}

	#[test]
	fn chunk_usage_falls_back_to_prompt_cache_hit_tokens() {
		let model = base_model();
		let raw = json!({
			"prompt_tokens": 50,
			"completion_tokens": 5,
			"prompt_cache_hit_tokens": 20,
		});
		let usage = parse_chunk_usage(&raw, &model, None);
		assert_eq!(usage.input, 30.0);
		assert_eq!(usage.cache_read, 20.0);
		assert_eq!(usage.cache_write, 0.0);
		assert_eq!(usage.total_tokens, 55.0);
	}

	#[test]
	fn chunk_usage_applies_cache_write_cost_override() {
		let mut model = base_model();
		model.cost = ModelCost {
			input: 4.0,
			output: 12.0,
			cache_read: 0.4,
			cache_write: 9.0,
		};
		let raw = json!({
			"prompt_tokens": 100,
			"completion_tokens": 0,
			"prompt_tokens_details": { "cached_tokens": 80, "cache_write_tokens": 80 },
		});
		let usage = parse_chunk_usage(&raw, &model, Some(model.cost.input * 1.25));
		assert!((usage.cost.cache_write - (80.0 * 5.0) / 1_000_000.0).abs() < 1e-12);
	}

	#[test]
	fn stop_reason_mapping_matches_the_typescript() {
		assert_eq!(map_stop_reason(None).stop_reason, "stop");
		assert_eq!(map_stop_reason(Some(&Value::Null)).stop_reason, "stop");
		assert_eq!(map_stop_reason(Some(&json!("stop"))).stop_reason, "stop");
		assert_eq!(map_stop_reason(Some(&json!("end"))).stop_reason, "stop");
		assert_eq!(map_stop_reason(Some(&json!("length"))).stop_reason, "length");
		assert_eq!(map_stop_reason(Some(&json!("function_call"))).stop_reason, "toolUse");
		assert_eq!(map_stop_reason(Some(&json!("tool_calls"))).stop_reason, "toolUse");

		let content_filter = map_stop_reason(Some(&json!("content_filter")));
		assert_eq!(content_filter.stop_reason, "error");
		assert_eq!(
			content_filter.error_message.as_deref(),
			Some("Provider finish_reason: content_filter")
		);

		let network_error = map_stop_reason(Some(&json!("network_error")));
		assert_eq!(network_error.stop_reason, "error");
		assert_eq!(
			network_error.error_message.as_deref(),
			Some("Provider finish_reason: network_error")
		);

		let other = map_stop_reason(Some(&json!("something_new")));
		assert_eq!(other.stop_reason, "error");
		assert_eq!(
			other.error_message.as_deref(),
			Some("Provider finish_reason: something_new")
		);
	}

	// ------------------------------------------------------------------
	// reasoning details encoding
	// ------------------------------------------------------------------

	#[test]
	fn reasoning_details_round_trip() {
		let mut detail = Map::new();
		detail.insert("type".to_string(), json!("reasoning.encrypted"));
		detail.insert("id".to_string(), json!("call-1"));
		detail.insert("data".to_string(), json!("opaque"));
		let encoded = encode_reasoning_details(&[detail.clone()]);
		assert!(encoded.starts_with('{'));
		let decoded = decode_reasoning_details(Some(&encoded)).expect("decoded");
		assert_eq!(decoded, vec![detail]);
	}

	#[test]
	fn reasoning_details_decoding_rejects_foreign_signatures() {
		assert!(decode_reasoning_details(None).is_none());
		assert!(decode_reasoning_details(Some("not json")).is_none());
		assert!(decode_reasoning_details(Some(r#"{"type":"other","details":[]}"#)).is_none());
		assert!(decode_reasoning_details(Some(r#"{"type":"openai-completions.reasoning_details.v1","details":[1]}"#)).is_none());
		assert!(decode_reasoning_details(Some(r#"{"type":"openai-completions.reasoning_details.v1","details":[]}"#)).is_some());
	}

	// ------------------------------------------------------------------
	// client / headers
	// ------------------------------------------------------------------

	#[test]
	fn client_sets_session_affinity_headers_only_when_enabled() {
		let model = compat_model(OpenAICompletionsCompat {
			send_session_affinity_headers: Some(true),
			..Default::default()
		});
		let mut compat = get_compat(&model);
		compat.send_session_affinity_headers = true;
		let client = create_client(&model, &context(vec![user_text("hi")]), Some("key"), None, Some("session-1"), &compat, None)
			.expect("client");
		assert_eq!(client.default_headers.get("session_id"), Some(&Some("session-1".to_string())));
		assert_eq!(
			client.default_headers.get("x-client-request-id"),
			Some(&Some("session-1".to_string()))
		);
		assert_eq!(
			client.default_headers.get("x-session-affinity"),
			Some(&Some("session-1".to_string()))
		);
	}

	#[test]
	fn client_lets_explicit_headers_override_session_affinity() {
		let model = compat_model(OpenAICompletionsCompat {
			send_session_affinity_headers: Some(true),
			..Default::default()
		});
		let compat = get_compat(&model);
		let mut headers = IndexMap::new();
		headers.insert("session_id".to_string(), "override-session".to_string());
		let client = create_client(
			&model,
			&context(vec![user_text("hi")]),
			Some("key"),
			Some(&headers),
			Some("session-1"),
			&compat,
			None,
		)
		.expect("client");
		assert_eq!(
			client.default_headers.get("session_id"),
			Some(&Some("override-session".to_string()))
		);
	}

	#[test]
	fn client_requires_an_api_key() {
		let mut env = crate::test_env::ScopedEnv::new();
		let model = base_model();
		env.remove("OPENAI_API_KEY");
		let compat = get_compat(&model);
		let error = create_client(&model, &context(vec![]), Some(""), None, None, &compat, None).unwrap_err();
		assert_eq!(
			error,
			"OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument."
		);
	}

	#[test]
	fn client_adds_cloudflare_gateway_authorization() {
		let mut model = base_model();
		model.provider = "cloudflare-ai-gateway".to_string();
		model.base_url = "https://gateway.ai.cloudflare.com/v1/a/g/compat".to_string();
		let compat = get_compat(&model);
		let client = create_client(&model, &context(vec![]), Some("cf-token"), None, None, &compat, None)
			.expect("client");
		assert_eq!(
			client.default_headers.get("cf-aig-authorization"),
			Some(&Some("Bearer cf-token".to_string()))
		);
		assert_eq!(client.default_headers.get("Authorization"), Some(&None));
	}

	#[test]
	fn client_keeps_inline_authorization_for_gateway_byok() {
		let mut model = base_model();
		model.provider = "cloudflare-ai-gateway".to_string();
		model.base_url = "https://gateway.ai.cloudflare.com/v1/a/g/compat".to_string();
		let compat = get_compat(&model);
		let mut headers = IndexMap::new();
		headers.insert("Authorization".to_string(), "Bearer upstream-token".to_string());
		let client = create_client(&model, &context(vec![]), Some("cf-token"), Some(&headers), None, &compat, None)
			.expect("client");
		assert_eq!(
			client.default_headers.get("Authorization"),
			Some(&Some("Bearer upstream-token".to_string()))
		);
		assert_eq!(
			client.default_headers.get("cf-aig-authorization"),
			Some(&Some("Bearer cf-token".to_string()))
		);
	}

	#[test]
	fn client_adds_prime_team_header() {
		let mut env = crate::test_env::ScopedEnv::new();
		let mut model = base_model();
		model.provider = "prime-inference".to_string();
		env.set("PRIME_TEAM_ID", "team-1");
		let compat = get_compat(&model);
		let client = create_client(&model, &context(vec![]), Some("key"), None, None, &compat, None).expect("client");
		assert_eq!(
			client.default_headers.get("X-Prime-Team-ID"),
			Some(&Some("team-1".to_string()))
		);
		env.remove("PRIME_TEAM_ID");
	}

	#[test]
	fn client_adds_copilot_dynamic_headers() {
		let mut model = base_model();
		model.provider = "github-copilot".to_string();
		let compat = get_compat(&model);
		let client = create_client(&model, &context(vec![user_text("hi")]), Some("key"), None, None, &compat, None)
			.expect("client");
		assert_eq!(client.default_headers.get("X-Initiator"), Some(&Some("user".to_string())));
		assert_eq!(
			client.default_headers.get("Openai-Intent"),
			Some(&Some("conversation-edits".to_string()))
		);
	}

	#[test]
	fn resolve_cache_retention_prefers_the_explicit_value() {
		let mut env = crate::test_env::ScopedEnv::new();
		env.remove("PI_CACHE_RETENTION");
		assert_eq!(resolve_cache_retention(None), "short");
		assert_eq!(resolve_cache_retention(Some(&"none".to_string())), "none");
		env.set("PI_CACHE_RETENTION", "long");
		assert_eq!(resolve_cache_retention(None), "long");
		env.remove("PI_CACHE_RETENTION");
	}

	#[test]
	fn sse_decoder_joins_multiline_data() {
		let mut decoder = SseDecoder::new();
		assert!(decoder.decode("data: {\"a\":").is_none());
		assert!(decoder.decode("data: 1}").is_none());
		let (_, payload) = decoder.decode("").expect("payload");
		assert_eq!(payload, "{\"a\":\n1}");
	}

	#[test]
	fn sse_decoder_ignores_comments_and_carriage_returns() {
		let mut decoder = SseDecoder::new();
		assert!(decoder.decode(": keep-alive").is_none());
		assert!(decoder.decode("data: x\r").is_none());
		let (_, payload) = decoder.decode("").expect("payload");
		assert_eq!(payload, "x");
		assert!(decoder.decode("").is_none());
	}

	#[test]
	fn find_double_newline_handles_lf_and_crlf() {
		assert_eq!(find_double_newline_index(b"a\n\nb"), Some(3));
		assert_eq!(find_double_newline_index(b"a\r\n\r\nb"), Some(5));
		assert_eq!(find_double_newline_index(b"a\r\n\nb"), Some(4));
		assert_eq!(find_double_newline_index(b"abc"), None);
	}

	#[test]
	fn options_serde_round_trip_keeps_extra_keys() {
		let base = StreamOptions {
			max_tokens: Some(10.0),
			..Default::default()
		};
		let typed = OpenAICompletionsOptions::from_base(&base);
		let encoded = serde_json::to_string(&typed).unwrap();
		let decoded: OpenAICompletionsOptions = serde_json::from_str(&encoded).unwrap();
		assert_eq!(decoded.stream.max_tokens, Some(10.0));
		assert!(decoded.reasoning_effort.is_none());
		assert!(decoded.reasoning_enabled.is_none());
	}
}


#[cfg(test)]
mod t15_controls_tests {
	//! T15 owner 'controls': configured gateway-route payload contracts. The
	//! fixtures mirror a configured gateway model profile with the credential
	//! command redacted to `test-key`; base URLs are inert fixture strings and
	//! nothing connects to the gateway ports.

	use super::*;
	use crate::types::{Compat, Message, ModelCost, OpenAICompletionsCompat, UserContent, UserMessage};
	use indexmap::IndexMap;
	use serde_json::json;

	fn route_compat() -> OpenAICompletionsCompat {
		OpenAICompletionsCompat {
			supports_store: Some(false),
			supports_developer_role: Some(false),
			supports_reasoning_effort: Some(true),
			supports_usage_in_streaming: Some(true),
			max_tokens_field: Some("max_tokens".to_string()),
			supports_strict_mode: Some(false),
			..Default::default()
		}
	}

	fn route_model(id: &str, max_tokens: f64, map: &[(&str, Option<&str>)]) -> Model {
		let mut level_map: IndexMap<String, Option<String>> = IndexMap::new();
		for (level, mapped) in map {
			level_map.insert(level.to_string(), mapped.map(|value| value.to_string()));
		}
		Model {
			id: id.to_string(),
			name: id.to_string(),
			api: "openai-completions".to_string(),
			provider: "gateway-route".to_string(),
			base_url: "http://127.0.0.1:43119/gateway/v1".to_string(),
			reasoning: true,
			input: vec![InputModality::Text, InputModality::Image],
			cost: ModelCost::zero(),
			context_window: 1_000_000.0,
			max_tokens,
			compat: Some(Compat::Completions(route_compat())),
			thinking_level_map: Some(level_map),
			..Default::default()
		}
	}

	fn user_text(text: &str) -> Message {
		Message::user(UserMessage::new(UserContent::Text(text.to_string()), 1))
	}

	fn context_with(messages: Vec<Message>) -> Context {
		Context::new(None, messages, None)
	}

	fn keyed_options() -> OpenAICompletionsOptions {
		OpenAICompletionsOptions {
			stream: StreamOptions {
				api_key: Some("test-key".to_string()),
				..Default::default()
			},
			..Default::default()
		}
	}

	fn effort_options(effort: Option<&str>) -> OpenAICompletionsOptions {
		let mut options = keyed_options();
		options.reasoning_effort = effort.map(|value| value.to_string());
		options
	}

	/// Same pipeline as the sibling tests module's `build` helper (get_compat +
	/// cache retention + build_params); helpers here are file-scope functions.
	fn build(model: &Model, context: &Context, options: Option<&OpenAICompletionsOptions>) -> Value {
		let compat = get_compat(model);
		let cache_retention =
			resolve_cache_retention(options.and_then(|options| options.stream.cache_retention.as_ref()));
		let cache_control = get_compat_cache_control(&compat, &cache_retention);
		build_params(model, context, options, &compat, &cache_retention, cache_control.as_ref())
			.expect("params")
	}

	/// ollama-cloud/deepseek-v4-flash (profile: maxTokens 384000, map max->max,
	/// off->null, provider compat max_tokens + no store + no developer role).
	#[test]
	fn t15_route_fixture_ollama_cloud_deepseek_v4_flash_payload_contract() {
		let model = route_model(
			"deepseek-v4-flash",
			384_000.0,
			&[("off", None), ("max", Some("max"))],
		);
		// The runtime forwards `model.maxTokens` into the stream options before the
		// provider sees them; mirror that here.
		let mut options = effort_options(Some("max"));
		options.stream.max_tokens = Some(384_000.0);
		let params = build(&model, &context_with(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["model"], json!("deepseek-v4-flash"));
		assert_eq!(params["max_tokens"], json!(384000), "route compat keeps the max_tokens field");
		assert!(params.get("max_completion_tokens").is_none());
		assert!(params.get("store").is_none(), "supportsStore false must omit store");
		assert_eq!(params["stream_options"]["include_usage"], json!(true));
		assert_eq!(params["reasoning_effort"], json!("max"), "mapped effort passes through");
		// A null mapping falls back to the requested level (TS `map[level] ?? effort`).
		let low = build(&model, &context_with(vec![user_text("hi")]), Some(&effort_options(Some("low"))));
		assert_eq!(low["reasoning_effort"], json!("low"));
		// `off` mapped to null must omit reasoning_effort entirely (TS `offValue !== null`).
		let mut off = keyed_options();
		off.reasoning_enabled = Some(false);
		let off_params = build(&model, &context_with(vec![user_text("hi")]), Some(&off));
		assert!(off_params.get("reasoning_effort").is_none(), "map.off == null must omit the off effort");
	}

	/// azure-foundry-managed/FW-Kimi-K3 (profile: map low/high/max pass through,
	/// off null; maxTokens 131072).
	#[test]
	fn t15_route_fixture_azure_foundry_fw_kimi_k3_payload_contract() {
		let model = route_model(
			"FW-Kimi-K3",
			131_072.0,
			&[("off", None), ("low", Some("low")), ("high", Some("high")), ("max", Some("max"))],
		);
		let mut options = effort_options(Some("max"));
		options.stream.max_tokens = Some(131_072.0);
		let params = build(&model, &context_with(vec![user_text("hi")]), Some(&options));
		assert_eq!(params["max_tokens"], json!(131072));
		assert_eq!(params["reasoning_effort"], json!("max"));
		let low = build(&model, &context_with(vec![user_text("hi")]), Some(&effort_options(Some("low"))));
		assert_eq!(low["reasoning_effort"], json!("low"));
		// Tool definitions must omit `strict` (supportsStrictMode false).
		let mut ctx = context_with(vec![user_text("hi")]);
		ctx.tools = Some(vec![Tool {
			name: "read".to_string(),
			description: "read".to_string(),
			parameters: serde_json::json!({"type": "object", "properties": {}}),
			..Default::default()
		}]);
		let with_tools = build(&model, &ctx, None);
		let tool = &with_tools["tools"][0];
		assert_eq!(tool["function"]["name"], json!("read"));
		assert!(tool.get("strict").is_none(), "supportsStrictMode false must omit strict");
	}

	/// E-02/E-04 integration: one SSE body exercising text, reasoning, tool-call
	/// assembly and wire usage; abort and in-stream-error classes are covered by
	/// the existing dedicated tests next to this module.
	#[tokio::test]
	async fn t15_stream_events_cover_text_reasoning_tool_usage() {
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let body = concat!(
			"data: {\"id\":\"chatcmpl-1\",\"model\":\"repro-model\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking hard\"}}]}\n\n",
			"data: {\"id\":\"chatcmpl-1\",\"model\":\"repro-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer text\"}}]}\n\n",
			"data: {\"id\":\"chatcmpl-1\",\"model\":\"repro-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call-1\",\"index\":0,\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]}}]}\n\n",
			"data: {\"id\":\"chatcmpl-1\",\"model\":\"repro-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"a\\\"}\"}}]}}]}\n\n",
			"data: {\"id\":\"chatcmpl-1\",\"model\":\"repro-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
			"data: [DONE]\n\n"
		);
		let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await.unwrap();
		let address = listener.local_addr().unwrap();
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut request = Vec::new();
			loop {
				let mut buffer = [0u8; 4096];
				let count = socket.read(&mut buffer).await.unwrap();
				assert_ne!(count, 0, "client closed before sending the request");
				request.extend_from_slice(&buffer[..count]);
				if request.windows(4).any(|window| window == b"\r\n\r\n") {
					break;
				}
			}
			let response = format!(
				"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
				body.len(),
				body
			);
			socket.write_all(response.as_bytes()).await.unwrap();
			socket.shutdown().await.unwrap();
		});
		let mut model = Model {
			id: "repro-model".to_string(),
			name: "Repro Model".to_string(),
			api: "openai-completions".to_string(),
			provider: "gateway-route".to_string(),
			base_url: format!("http://{address}"),
			reasoning: true,
			input: vec![InputModality::Text],
			cost: ModelCost::zero(),
			..Default::default()
		};
		model.compat = Some(Compat::Completions(route_compat()));
		let stream = stream_openai_completions(&model, &context_with(vec![user_text("hi")]), Some(keyed_options()));
		let mut kinds: Vec<&'static str> = Vec::new();
		let mut done: Option<AssistantMessage> = None;
		while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
			.await
			.expect("the fixture stream must finish")
		{
			match event {
				AssistantMessageEvent::ThinkingDelta { delta, .. } => {
					kinds.push("thinking_delta");
					assert_eq!(delta, "thinking hard");
				}
				AssistantMessageEvent::TextDelta { delta, .. } => {
					kinds.push("text_delta");
					assert_eq!(delta, "answer text");
				}
				AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
					kinds.push("toolcall_end");
					assert_eq!(tool_call.name, "read");
					assert_eq!(tool_call.arguments.get("path"), Some(&json!("a")));
				}
				AssistantMessageEvent::Done { reason, message } => {
					kinds.push("done");
					assert_eq!(reason, "toolUse");
					done = Some(message);
				}
				AssistantMessageEvent::Error { error, .. } => panic!("unexpected error event: {:?}", error.error_message),
				_ => {}
			}
		}
		tokio::time::timeout(std::time::Duration::from_secs(5), server).await.expect("server").unwrap();
		assert!(kinds.contains(&"thinking_delta"), "reasoning_content frames must stream as thinking: {kinds:?}");
		assert!(kinds.contains(&"text_delta"), "{kinds:?}");
		assert!(kinds.contains(&"toolcall_end"), "{kinds:?}");
		assert!(kinds.contains(&"done"), "{kinds:?}");
		let message = done.expect("done message");
		assert_eq!(message.stop_reason, "toolUse");
		assert_eq!(message.usage.input, 10.0);
		assert_eq!(message.usage.output, 5.0);
	}
}
