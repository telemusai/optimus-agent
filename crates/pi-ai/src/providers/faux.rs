//! Port of packages/ai/src/providers/faux.ts
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::api_registry::{register_api_provider, unregister_api_providers, ApiProvider};
use crate::types::{
	AssistantMessage, AssistantMessageEvent, ContentBlock, Context, ImageOrTextContent, Message, Model, ModelCost,
	SimpleStreamOptions, StreamOptions, TextContent, ThinkingContent, ToolCall, ToolResultMessage, Usage, UsageCost,
	UserContent,
};
use crate::utils::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use futures::future::BoxFuture;
use futures::FutureExt;
use indexmap::IndexMap;

const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";
const DEFAULT_MODEL_NAME: &str = "Faux Model";
const DEFAULT_BASE_URL: &str = "http://localhost:0";
const DEFAULT_MIN_TOKEN_SIZE: f64 = 3.0;
const DEFAULT_MAX_TOKEN_SIZE: f64 = 5.0;

/// TS: `Date.now()`
fn now_ms() -> i64 {
	crate::utils::now_ms()
}

/// TS: `DEFAULT_USAGE`
fn default_usage() -> Usage {
	Usage::zero()
}

/// TS: `Math.random()`
fn random_f64() -> f64 {
	rand::random::<f64>()
}

/// TS: `Math.random().toString(36).slice(2)`
fn random_base36() -> String {
	let value = (random_f64() * (1u64 << 53) as f64) as u64;
	let mut digits = String::new();
	let mut v = value;
	if v == 0 {
		digits.push('0');
	}
	while v > 0 {
		let d = (v % 36) as u32;
		digits.push(std::char::from_digit(d, 36).unwrap_or('0'));
		v /= 36;
	}
	digits.chars().rev().collect()
}

fn random_id(prefix: &str) -> String {
	format!("{}:{}:{}", prefix, now_ms(), random_base36())
}

/// TS: `estimateTokens(text)` - `Math.ceil(text.length / 4)` where `length` is UTF-16 code units.
fn estimate_tokens(text: &str) -> f64 {
	(text.encode_utf16().count() as f64 / 4.0).ceil()
}

/// TS: `FauxModelDefinition`
#[derive(Debug, Clone, Default)]
pub struct FauxModelDefinition {
	pub id: String,
	pub name: Option<String>,
	pub reasoning: Option<bool>,
	pub input: Option<Vec<String>>,
	pub cost: Option<ModelCost>,
	pub context_window: Option<f64>,
	pub max_tokens: Option<f64>,
}

/// TS: `fauxText(text)`
pub fn faux_text(text: &str) -> TextContent {
	TextContent::new(text)
}

/// TS: `fauxThinking(thinking)`
pub fn faux_thinking(thinking: &str) -> ThinkingContent {
	ThinkingContent::new(thinking)
}

/// TS: `fauxToolCall(name, arguments_, options = {})`
#[derive(Debug, Clone, Default)]
pub struct FauxToolCallOptions {
	pub id: Option<String>,
}

pub fn faux_tool_call(
	name: &str,
	arguments: serde_json::Map<String, serde_json::Value>,
	options: Option<FauxToolCallOptions>,
) -> ToolCall {
	ToolCall::new(
		options.and_then(|o| o.id).unwrap_or_else(|| random_id("tool")),
		name,
		arguments,
	)
}

/// TS: `string | FauxContentBlock | FauxContentBlock[]`
#[derive(Debug, Clone)]
pub enum FauxAssistantContent {
	Text(String),
	Block(ContentBlock),
	Blocks(Vec<ContentBlock>),
}

impl From<&str> for FauxAssistantContent {
	fn from(value: &str) -> Self {
		FauxAssistantContent::Text(value.to_string())
	}
}

impl From<String> for FauxAssistantContent {
	fn from(value: String) -> Self {
		FauxAssistantContent::Text(value)
	}
}

impl From<ContentBlock> for FauxAssistantContent {
	fn from(value: ContentBlock) -> Self {
		FauxAssistantContent::Block(value)
	}
}

impl From<Vec<ContentBlock>> for FauxAssistantContent {
	fn from(value: Vec<ContentBlock>) -> Self {
		FauxAssistantContent::Blocks(value)
	}
}

fn normalize_faux_assistant_content(content: FauxAssistantContent) -> Vec<ContentBlock> {
	match content {
		FauxAssistantContent::Text(text) => vec![ContentBlock::Text(faux_text(&text))],
		FauxAssistantContent::Block(block) => vec![block],
		FauxAssistantContent::Blocks(blocks) => blocks,
	}
}

/// TS: `fauxAssistantMessage(content, options = {})`
#[derive(Debug, Clone, Default)]
pub struct FauxAssistantMessageOptions {
	pub stop_reason: Option<String>,
	pub error_message: Option<String>,
	pub response_id: Option<String>,
	pub timestamp: Option<i64>,
}

pub fn faux_assistant_message(
	content: FauxAssistantContent,
	options: Option<FauxAssistantMessageOptions>,
) -> AssistantMessage {
	let options = options.unwrap_or_default();
	AssistantMessage {
		content: normalize_faux_assistant_content(content),
		api: DEFAULT_API.to_string(),
		provider: DEFAULT_PROVIDER.to_string(),
		model: DEFAULT_MODEL_ID.to_string(),
		usage: default_usage(),
		stop_reason: options.stop_reason.unwrap_or_else(|| "stop".to_string()),
		error_message: options.error_message,
		response_id: options.response_id,
		timestamp: options.timestamp.unwrap_or_else(now_ms),
		..Default::default()
	}
}

/// TS: `{ callCount: number }`
#[derive(Debug, Clone, Default)]
pub struct FauxState {
	pub call_count: Arc<AtomicU64>,
}

impl FauxState {
	pub fn call_count(&self) -> u64 {
		self.call_count.load(Ordering::SeqCst)
	}
}

/// TS: `FauxResponseFactory`
pub type FauxResponseFactory = Arc<
	dyn for<'a> Fn(
			&'a Context,
			Option<&'a StreamOptions>,
			&'a FauxState,
			&'a Model,
		) -> BoxFuture<'a, AssistantMessage>
		+ Send
		+ Sync,
>;

/// TS: `FauxResponseStep`
#[derive(Clone)]
pub enum FauxResponseStep {
	Message(AssistantMessage),
	Factory(FauxResponseFactory),
}

/// TS: `RegisterFauxProviderOptions`
#[derive(Debug, Clone, Default)]
pub struct RegisterFauxProviderOptions {
	pub api: Option<String>,
	pub provider: Option<String>,
	pub models: Option<Vec<FauxModelDefinition>>,
	pub tokens_per_second: Option<f64>,
	pub token_size: Option<FauxTokenSize>,
}

#[derive(Debug, Clone, Default)]
pub struct FauxTokenSize {
	pub min: Option<f64>,
	pub max: Option<f64>,
}

/// TS: `FauxProviderRegistration`
#[derive(Clone)]
pub struct FauxProviderRegistration {
	pub api: String,
	pub models: Vec<Model>,
	state: FauxState,
	inner: Arc<FauxProviderInner>,
}

struct FauxProviderInner {
	source_id: String,
	pending_responses: Mutex<Vec<FauxResponseStep>>,
}

impl FauxProviderRegistration {
	/// TS: `getModel()` - the first registered model.
	pub fn get_model(&self) -> Model {
		self.models
			.first()
			.cloned()
			.expect("faux provider always registers at least one model")
	}

	/// TS: `getModel(modelId)` - `undefined` when the id is unknown.
	pub fn get_model_by_id(&self, model_id: &str) -> Option<Model> {
		self.models.iter().find(|candidate| candidate.id == model_id).cloned()
	}

	/// TS: `setResponses(responses)`
	pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
		*self.inner.pending_responses.lock().unwrap() = responses;
	}

	/// TS: `appendResponses(responses)`
	pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
		self.inner.pending_responses.lock().unwrap().extend(responses);
	}

	/// TS: `getPendingResponseCount()`
	pub fn get_pending_response_count(&self) -> usize {
		self.inner.pending_responses.lock().unwrap().len()
	}

	/// TS: `unregister()`
	pub fn unregister(&self) {
		unregister_api_providers(&self.inner.source_id);
	}

	pub fn call_count(&self) -> u64 {
		self.state.call_count()
	}
}

fn content_to_text(content: &UserContent) -> String {
	match content {
		UserContent::Text(text) => text.clone(),
		UserContent::Blocks(blocks) => blocks
			.iter()
			.map(|block| match block {
				ImageOrTextContent::Text(text) => text.text.clone(),
				ImageOrTextContent::Image(image) => {
					format!("[image:{}:{}]", image.mime_type, image.data.encode_utf16().count())
				}
			})
			.collect::<Vec<_>>()
			.join("\n"),
	}
}

fn image_or_text_blocks_to_text(content: &[ImageOrTextContent]) -> String {
	content
		.iter()
		.map(|block| match block {
			ImageOrTextContent::Text(text) => text.text.clone(),
			ImageOrTextContent::Image(image) => {
				format!("[image:{}:{}]", image.mime_type, image.data.encode_utf16().count())
			}
		})
		.collect::<Vec<_>>()
		.join("\n")
}

fn assistant_content_to_text(content: &[ContentBlock]) -> String {
	content
		.iter()
		.map(|block| match block {
			ContentBlock::Text(text) => text.text.clone(),
			ContentBlock::Thinking(thinking) => thinking.thinking.clone(),
			ContentBlock::ToolCall(tool_call) => format!(
				"{}:{}",
				tool_call.name,
				serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string())
			),
		})
		.collect::<Vec<_>>()
		.join("\n")
}

fn tool_result_to_text(message: &ToolResultMessage) -> String {
	let mut parts = vec![message.tool_name.clone()];
	parts.push(image_or_text_blocks_to_text(&message.content));
	parts.join("\n")
}

/// TS: `message.role`
fn message_role(message: &Message) -> &'static str {
	match message {
		Message::User(_) => "user",
		Message::Assistant(_) => "assistant",
		Message::ToolResult(_) => "toolResult",
	}
}

fn message_to_text(message: &Message) -> String {
	match message {
		Message::User(user) => content_to_text(&user.content),
		Message::Assistant(assistant) => assistant_content_to_text(&assistant.content),
		Message::ToolResult(result) => tool_result_to_text(result),
	}
}

fn serialize_context(context: &Context) -> String {
	let mut parts: Vec<String> = Vec::new();
	if let Some(system_prompt) = &context.system_prompt {
		if !system_prompt.is_empty() {
			parts.push(format!("system:{}", system_prompt));
		}
	}
	for message in &context.messages {
		parts.push(format!("{}:{}", message_role(message), message_to_text(message)));
	}
	if let Some(tools) = &context.tools {
		if !tools.is_empty() {
			parts.push(format!(
				"tools:{}",
				serde_json::to_string(tools).unwrap_or_else(|_| "[]".to_string())
			));
		}
	}
	parts.join("\n\n")
}

fn common_prefix_length(a: &str, b: &str) -> usize {
	let a_units: Vec<u16> = a.encode_utf16().collect();
	let b_units: Vec<u16> = b.encode_utf16().collect();
	let length = a_units.len().min(b_units.len());
	let mut index = 0;
	while index < length && a_units[index] == b_units[index] {
		index += 1;
	}
	index
}

fn with_usage_estimate(
	message: AssistantMessage,
	context: &Context,
	options: Option<&StreamOptions>,
	prompt_cache: &Mutex<HashMap<String, String>>,
) -> AssistantMessage {
	let prompt_text = serialize_context(context);
	let prompt_tokens = estimate_tokens(&prompt_text);
	let output_tokens = estimate_tokens(&assistant_content_to_text(&message.content));
	let mut input = prompt_tokens;
	let mut cache_read = 0.0;
	let mut cache_write = 0.0;
	let session_id = options.and_then(|o| o.session_id.clone());

	if let Some(session_id) = session_id {
		if options.and_then(|o| o.cache_retention.as_deref()) != Some("none") {
			let previous_prompt = prompt_cache.lock().unwrap().get(&session_id).cloned();
			if let Some(previous_prompt) = previous_prompt {
				let cached_chars = common_prefix_length(&previous_prompt, &prompt_text);
				let units: Vec<u16> = previous_prompt.encode_utf16().collect();
				let prefix: String = String::from_utf16_lossy(&units[..cached_chars.min(units.len())]);
				cache_read = estimate_tokens(&prefix);
				let units: Vec<u16> = prompt_text.encode_utf16().collect();
				let rest: String = String::from_utf16_lossy(&units[cached_chars.min(units.len())..]);
				cache_write = estimate_tokens(&rest);
				input = (prompt_tokens - cache_read).max(0.0);
			} else {
				cache_write = prompt_tokens;
			}
			prompt_cache.lock().unwrap().insert(session_id, prompt_text);
		}
	}

	AssistantMessage {
		usage: Usage {
			input,
			output: output_tokens,
			cache_read,
			cache_write,
			total_tokens: input + output_tokens + cache_read + cache_write,
			cost: UsageCost::zero(),
		},
		..message
	}
}

/// TS: `splitStringByTokenSize(text, minTokenSize, maxTokenSize)`
///
/// JavaScript slices by UTF-16 code units; the Rust port slices by `char` while counting UTF-16
/// units, so a surrogate pair is never split (that matches `String.prototype.slice` behaviour on
/// well-formed text and avoids invalid UTF-8).
fn split_string_by_token_size(text: &str, min_token_size: f64, max_token_size: f64) -> Vec<String> {
	let mut chunks: Vec<String> = Vec::new();
	let chars: Vec<char> = text.chars().collect();
	let mut index = 0usize;
	while index < chars.len() {
		let token_size = min_token_size + (random_f64() * (max_token_size - min_token_size + 1.0)).floor();
		let char_size = ((token_size * 4.0) as usize).max(1);
		let mut taken = 0usize;
		let mut units = 0usize;
		while index + taken < chars.len() && units < char_size {
			units += chars[index + taken].len_utf16();
			taken += 1;
		}
		let taken = taken.max(1);
		chunks.push(chars[index..index + taken].iter().collect());
		index += taken;
	}
	if chunks.is_empty() {
		chunks.push(String::new());
	}
	chunks
}

/// TS: `cloneMessage(message, api, provider, modelId)` (uses `structuredClone`)
fn clone_message(message: &AssistantMessage, api: &str, provider: &str, model_id: &str) -> AssistantMessage {
	let cloned = message.clone();
	AssistantMessage {
		api: api.to_string(),
		provider: provider.to_string(),
		model: model_id.to_string(),
		timestamp: cloned.timestamp,
		usage: cloned.usage.clone(),
		..cloned
	}
}

fn create_error_message(error: &str, api: &str, provider: &str, model_id: &str) -> AssistantMessage {
	AssistantMessage {
		content: Vec::new(),
		api: api.to_string(),
		provider: provider.to_string(),
		model: model_id.to_string(),
		usage: default_usage(),
		stop_reason: "error".to_string(),
		error_message: Some(error.to_string()),
		timestamp: now_ms(),
		..Default::default()
	}
}

fn create_aborted_message(partial: &AssistantMessage) -> AssistantMessage {
	AssistantMessage {
		stop_reason: "aborted".to_string(),
		error_message: Some("Request was aborted".to_string()),
		timestamp: now_ms(),
		..partial.clone()
	}
}

async fn schedule_chunk(chunk: &str, tokens_per_second: Option<f64>) {
	match tokens_per_second {
		Some(rate) if rate > 0.0 => {
			let delay_ms = (estimate_tokens(chunk) / rate) * 1000.0;
			tokio::time::sleep(std::time::Duration::from_micros((delay_ms * 1000.0) as u64)).await;
		}
		_ => {
			tokio::task::yield_now().await;
		}
	}
}

fn is_aborted(signal: Option<&tokio_util::sync::CancellationToken>) -> bool {
	signal.map(|s| s.is_cancelled()).unwrap_or(false)
}

async fn stream_with_deltas(
	stream: &AssistantMessageEventStream,
	message: AssistantMessage,
	min_token_size: f64,
	max_token_size: f64,
	tokens_per_second: Option<f64>,
	signal: Option<tokio_util::sync::CancellationToken>,
) {
	let mut partial = AssistantMessage {
		content: Vec::new(),
		..message.clone()
	};
	if is_aborted(signal.as_ref()) {
		let aborted = create_aborted_message(&partial);
		stream.push(AssistantMessageEvent::Error {
			reason: "aborted".to_string(),
			error: aborted.clone(),
		});
		stream.end(Some(aborted));
		return;
	}

	stream.push(AssistantMessageEvent::Start {
		partial: partial.clone(),
	});

	for index in 0..message.content.len() {
		if is_aborted(signal.as_ref()) {
			let aborted = create_aborted_message(&partial);
			stream.push(AssistantMessageEvent::Error {
				reason: "aborted".to_string(),
				error: aborted.clone(),
			});
			stream.end(Some(aborted));
			return;
		}

		let block = message.content[index].clone();

		match block {
			ContentBlock::Thinking(thinking) => {
				partial.content.push(ContentBlock::Thinking(ThinkingContent::new("")));
				stream.push(AssistantMessageEvent::ThinkingStart {
					content_index: index,
					partial: partial.clone(),
				});
				for chunk in split_string_by_token_size(&thinking.thinking, min_token_size, max_token_size) {
					schedule_chunk(&chunk, tokens_per_second).await;
					if is_aborted(signal.as_ref()) {
						let aborted = create_aborted_message(&partial);
						stream.push(AssistantMessageEvent::Error {
							reason: "aborted".to_string(),
							error: aborted.clone(),
						});
						stream.end(Some(aborted));
						return;
					}
					if let ContentBlock::Thinking(current) = &mut partial.content[index] {
						current.thinking.push_str(&chunk);
					}
					stream.push(AssistantMessageEvent::ThinkingDelta {
						content_index: index,
						delta: chunk,
						partial: partial.clone(),
					});
				}
				stream.push(AssistantMessageEvent::ThinkingEnd {
					content_index: index,
					content: thinking.thinking.clone(),
					partial: partial.clone(),
				});
			}
			ContentBlock::Text(text) => {
				partial.content.push(ContentBlock::Text(TextContent::new("")));
				stream.push(AssistantMessageEvent::TextStart {
					content_index: index,
					partial: partial.clone(),
				});
				for chunk in split_string_by_token_size(&text.text, min_token_size, max_token_size) {
					schedule_chunk(&chunk, tokens_per_second).await;
					if is_aborted(signal.as_ref()) {
						let aborted = create_aborted_message(&partial);
						stream.push(AssistantMessageEvent::Error {
							reason: "aborted".to_string(),
							error: aborted.clone(),
						});
						stream.end(Some(aborted));
						return;
					}
					if let ContentBlock::Text(current) = &mut partial.content[index] {
						current.text.push_str(&chunk);
					}
					stream.push(AssistantMessageEvent::TextDelta {
						content_index: index,
						delta: chunk,
						partial: partial.clone(),
					});
				}
				stream.push(AssistantMessageEvent::TextEnd {
					content_index: index,
					content: text.text.clone(),
					partial: partial.clone(),
				});
			}
			ContentBlock::ToolCall(tool_call) => {
				partial.content.push(ContentBlock::ToolCall(ToolCall::new(
					tool_call.id.clone(),
					tool_call.name.clone(),
					serde_json::Map::new(),
				)));
				stream.push(AssistantMessageEvent::ToolCallStart {
					content_index: index,
					partial: partial.clone(),
				});
				let serialized = serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string());
				for chunk in split_string_by_token_size(&serialized, min_token_size, max_token_size) {
					schedule_chunk(&chunk, tokens_per_second).await;
					if is_aborted(signal.as_ref()) {
						let aborted = create_aborted_message(&partial);
						stream.push(AssistantMessageEvent::Error {
							reason: "aborted".to_string(),
							error: aborted.clone(),
						});
						stream.end(Some(aborted));
						return;
					}
					stream.push(AssistantMessageEvent::ToolCallDelta {
						content_index: index,
						delta: chunk,
						partial: partial.clone(),
					});
				}
				if let ContentBlock::ToolCall(current) = &mut partial.content[index] {
					current.arguments = tool_call.arguments.clone();
				}
				stream.push(AssistantMessageEvent::ToolCallEnd {
					content_index: index,
					tool_call: tool_call.clone(),
					partial: partial.clone(),
				});
			}
		}
	}

	if message.stop_reason == "error" || message.stop_reason == "aborted" {
		stream.push(AssistantMessageEvent::Error {
			reason: message.stop_reason.clone(),
			error: message.clone(),
		});
		stream.end(Some(message));
		return;
	}

	stream.push(AssistantMessageEvent::Done {
		reason: message.stop_reason.clone(),
		message: message.clone(),
	});
	stream.end(Some(message));
}

/// TS: `registerFauxProvider(options = {})`
pub fn register_faux_provider(options: Option<RegisterFauxProviderOptions>) -> FauxProviderRegistration {
	let options = options.unwrap_or_default();
	let api = options.api.clone().unwrap_or_else(|| random_id(DEFAULT_API));
	let provider = options.provider.clone().unwrap_or_else(|| DEFAULT_PROVIDER.to_string());
	let source_id = random_id("faux-provider");
	let min_token_size = 1.0f64.max(
		options
			.token_size
			.as_ref()
			.and_then(|t| t.min)
			.unwrap_or(DEFAULT_MIN_TOKEN_SIZE)
			.min(
				options
					.token_size
					.as_ref()
					.and_then(|t| t.max)
					.unwrap_or(DEFAULT_MAX_TOKEN_SIZE),
			),
	);
	let max_token_size = min_token_size.max(
		options
			.token_size
			.as_ref()
			.and_then(|t| t.max)
			.unwrap_or(DEFAULT_MAX_TOKEN_SIZE),
	);
	let tokens_per_second = options.tokens_per_second;
	let state = FauxState::default();
	let prompt_cache: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));

	let model_definitions = match options.models.clone() {
		Some(models) if !models.is_empty() => models,
		_ => vec![FauxModelDefinition {
			id: DEFAULT_MODEL_ID.to_string(),
			name: Some(DEFAULT_MODEL_NAME.to_string()),
			reasoning: Some(false),
			input: Some(vec!["text".to_string(), "image".to_string()]),
			cost: Some(ModelCost::zero()),
			context_window: Some(128000.0),
			max_tokens: Some(16384.0),
		}],
	};
	let models: Vec<Model> = model_definitions
		.into_iter()
		.map(|definition| Model {
			name: definition.name.clone().unwrap_or_else(|| definition.id.clone()),
			id: definition.id.clone(),
			api: api.clone(),
			provider: provider.clone(),
			base_url: DEFAULT_BASE_URL.to_string(),
			reasoning: definition.reasoning.unwrap_or(false),
			input: definition
				.input
				.clone()
				.unwrap_or_else(|| vec!["text".to_string(), "image".to_string()])
				.into_iter()
				.map(|value| match value.as_str() {
					"image" => crate::types::InputModality::Image,
					_ => crate::types::InputModality::Text,
				})
				.collect(),
			cost: definition.cost.clone().unwrap_or_else(ModelCost::zero),
			context_window: definition.context_window.unwrap_or(128000.0),
			max_tokens: definition.max_tokens.unwrap_or(16384.0),
			..Default::default()
		})
		.collect();

	let inner = Arc::new(FauxProviderInner {
		source_id: source_id.clone(),
		pending_responses: Mutex::new(Vec::new()),
	});

	let stream_inner = inner.clone();
	let stream_state = state.clone();
	let stream_prompt_cache = prompt_cache.clone();
	let stream_api = api.clone();
	let stream_provider = provider.clone();
	let stream: crate::types::StreamFunction = Arc::new(move |request_model: &Model, context: &Context, stream_options: Option<&StreamOptions>| {
		let outer = create_assistant_message_event_stream();
		let step = {
			let mut pending = stream_inner.pending_responses.lock().unwrap();
			if pending.is_empty() {
				None
			} else {
				Some(pending.remove(0))
			}
		};
		stream_state.call_count.fetch_add(1, Ordering::SeqCst);

		let task_stream = outer.clone();
		let api = stream_api.clone();
		let provider = stream_provider.clone();
		let state = stream_state.clone();
		let prompt_cache = stream_prompt_cache.clone();
		let request_model = request_model.clone();
		let context = context.clone();
		let stream_options = stream_options.cloned();
		let model_id = request_model.id.clone();

		tokio::spawn(async move {
			// TS faux.ts:436-462 wraps the whole task body in `try { ... } catch (error) { ... }`,
			// so an unexpected failure ENDS the stream with an `error` event instead of leaving
			// `next()`/`result()` callers waiting forever. Rust has no `throw`: the equivalent
			// unexpected failure is a panic inside the task, and an uncaught panic would kill
			// the task without ending the stream.
			let body = async {
				// TS parity: the request payload edge must be observable even for the
				// scripted provider, so extension handlers (e.g. pi-jev
				// `before_provider_request`) see the boundary. The faux provider has no
				// wire request to replace: the returned payload is discarded and the
				// scripted responses are used as-is.
				if let Some(on_payload) = stream_options.as_ref().and_then(|o| o.on_payload.clone()) {
					let payload = serde_json::to_value(&context).unwrap_or_else(|_| serde_json::json!({}));
					let _ = on_payload(payload, &request_model).await;
				}

				if let Some(on_response) = stream_options.as_ref().and_then(|o| o.on_response.clone()) {
					on_response(
						crate::types::ProviderResponse {
							status: 200,
							headers: IndexMap::new(),
						},
						&request_model,
					)
					.await;
				}

				let step = match step {
					Some(step) => step,
					None => {
						let mut message = create_error_message("No more faux responses queued", &api, &provider, &model_id);
						message = with_usage_estimate(
							message,
							&context,
							stream_options.as_ref(),
							&prompt_cache,
						);
						task_stream.push(AssistantMessageEvent::Error {
							reason: "error".to_string(),
							error: message.clone(),
						});
						task_stream.end(Some(message));
						return;
					}
				};

				let resolved = match step {
					FauxResponseStep::Message(message) => message,
					FauxResponseStep::Factory(factory) => {
						factory(&context, stream_options.as_ref(), &state, &request_model).await
					}
				};
				let mut message = clone_message(&resolved, &api, &provider, &model_id);
				message = with_usage_estimate(message, &context, stream_options.as_ref(), &prompt_cache);
				let signal = stream_options.as_ref().and_then(|o| o.signal.clone());
				stream_with_deltas(
					&task_stream,
					message,
					min_token_size,
					max_token_size,
					tokens_per_second,
					signal,
				)
				.await;
			};
			if let Err(panic) = std::panic::AssertUnwindSafe(body).catch_unwind().await {
				// faux.ts:457-460: the catch-all builds `createErrorMessage(error, api, provider, modelId)`,
				// pushes `{ type: "error", reason: "error", error: message }` and calls `outer.end(message)`.
				// faux.ts:274 maps the unknown error with `error instanceof Error ? error.message : String(error)`;
				// the panic payload carries that same message text.
				let panic_text = panic
					.downcast_ref::<&str>()
					.map(|message| (*message).to_string())
					.or_else(|| panic.downcast_ref::<String>().cloned())
					.unwrap_or_else(|| "unknown panic".to_string());
				let message = create_error_message(&panic_text, &api, &provider, &model_id);
				task_stream.push(AssistantMessageEvent::Error {
					reason: "error".to_string(),
					error: message.clone(),
				});
				task_stream.end(Some(message));
			}
		});

		outer
	});

	let simple_stream: crate::types::StreamFunction = {
		let stream = stream.clone();
		Arc::new(move |model: &Model, context: &Context, options: Option<&StreamOptions>| {
			stream(model, context, options)
		})
	};

	register_api_provider(
		ApiProvider {
			api: api.clone(),
			stream,
			stream_simple: simple_stream,
			compact: None,
			supports_compaction: None,
		},
		Some(source_id.clone()),
	);

	FauxProviderRegistration {
		api,
		models,
		state,
		inner,
	}
}

/// Convenience alias mirroring the TS `SimpleStreamOptions` overload of the faux provider.
pub type FauxSimpleStreamOptions = SimpleStreamOptions;

#[cfg(test)]
mod tests {
	use super::*;
	use crate::api_registry::API_REGISTRY_TEST_LOCK;
	use crate::types::{Tool, UserMessage};

	#[test]
	fn estimate_tokens_counts_utf16_units() {
		assert_eq!(estimate_tokens(""), 0.0);
		assert_eq!(estimate_tokens("abcd"), 1.0);
		assert_eq!(estimate_tokens("abcde"), 2.0);
		// Astral chars are 2 UTF-16 units, like JS `text.length`.
		assert_eq!(estimate_tokens("🙈🙈"), 1.0);
	}

	#[test]
	fn faux_text_and_thinking_build_expected_blocks() {
		let text = faux_text("hello");
		assert_eq!(text.text, "hello");
		let thinking = faux_thinking("hmm");
		assert_eq!(thinking.thinking, "hmm");
	}

	#[test]
	fn faux_tool_call_uses_provided_id_or_generates_one() {
		let mut args = serde_json::Map::new();
		args.insert("path".to_string(), serde_json::Value::String("/tmp".to_string()));
		let tool_call = faux_tool_call(
			"read",
			args.clone(),
			Some(FauxToolCallOptions {
				id: Some("call-1".to_string()),
			}),
		);
		assert_eq!(tool_call.id, "call-1");
		assert_eq!(tool_call.name, "read");
		assert_eq!(tool_call.arguments, args);

		let generated = faux_tool_call("read", args, None);
		assert!(generated.id.starts_with("tool:"));
	}

	#[test]
	fn faux_assistant_message_defaults_match_typescript() {
		let message = faux_assistant_message(FauxAssistantContent::Text("hi".to_string()), None);
		assert_eq!(message.api, "faux");
		assert_eq!(message.provider, "faux");
		assert_eq!(message.model, "faux-1");
		assert_eq!(message.stop_reason, "stop");
		assert_eq!(message.usage.total_tokens, 0.0);
		assert_eq!(message.content.len(), 1);
	}

	#[test]
	fn faux_assistant_message_accepts_single_block_and_options() {
		let message = faux_assistant_message(
			FauxAssistantContent::Block(ContentBlock::Thinking(faux_thinking("why"))),
			Some(FauxAssistantMessageOptions {
				stop_reason: Some("error".to_string()),
				error_message: Some("boom".to_string()),
				response_id: Some("resp-1".to_string()),
				timestamp: Some(42),
			}),
		);
		assert_eq!(message.stop_reason, "error");
		assert_eq!(message.error_message.as_deref(), Some("boom"));
		assert_eq!(message.response_id.as_deref(), Some("resp-1"));
		assert_eq!(message.timestamp, 42);
	}

	#[test]
	fn serialize_context_matches_typescript_layout() {
		let mut context = Context::default();
		context.system_prompt = Some("be nice".to_string());
		context.messages = vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))];
		context.tools = Some(vec![Tool {
			name: "read".to_string(),
			description: "read a file".to_string(),
			parameters: serde_json::json!({"type": "object"}),
		}]);
		let serialized = serialize_context(&context);
		assert!(serialized.starts_with("system:be nice\n\nuser:hi\n\ntools:"));
		assert!(serialized.contains("\"name\":\"read\""));
	}

	#[test]
	fn serialize_context_skips_empty_system_prompt_and_tools() {
		let context = Context {
			system_prompt: Some(String::new()),
			messages: vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 0))],
			tools: Some(Vec::new()),
		};
		assert_eq!(serialize_context(&context), "user:hi");
	}

	#[test]
	fn content_to_text_formats_images_like_typescript() {
		let content = UserContent::Blocks(vec![
			ImageOrTextContent::Text(TextContent::new("look")),
			ImageOrTextContent::Image(crate::types::ImageContent::new("AAAA", "image/png")),
		]);
		assert_eq!(content_to_text(&content), "look\n[image:image/png:4]");
	}

	#[test]
	fn assistant_content_to_text_serializes_tool_arguments() {
		let mut args = serde_json::Map::new();
		args.insert("a".to_string(), serde_json::Value::from(1));
		let content = vec![
			ContentBlock::Text(TextContent::new("t")),
			ContentBlock::Thinking(ThinkingContent::new("th")),
			ContentBlock::ToolCall(ToolCall::new("id", "name", args)),
		];
		assert_eq!(assistant_content_to_text(&content), "t\nth\nname:{\"a\":1}");
	}

	#[test]
	fn common_prefix_length_uses_utf16_units() {
		assert_eq!(common_prefix_length("abc", "abd"), 2);
		assert_eq!(common_prefix_length("", "abc"), 0);
		assert_eq!(common_prefix_length("abc", "abc"), 3);
	}

	#[test]
	fn faux_provider_options_keep_provider_defaults() {
		let options = RegisterFauxProviderOptions::default();
		assert!(options.api.is_none());
		assert!(options.provider.is_none());
		assert!(options.models.is_none());
		assert!(options.tokens_per_second.is_none());
		assert!(options.token_size.is_none());
	}

	#[test]
	fn with_usage_estimate_without_session_uses_full_prompt() {
		let context = Context {
			system_prompt: Some("x".repeat(40)),
			messages: Vec::new(),
			tools: None,
		};
		let cache = Mutex::new(HashMap::new());
		let message = with_usage_estimate(faux_assistant_message(FauxAssistantContent::Text("abcd".to_string()), None), &context, None, &cache);
		assert_eq!(message.usage.input, 12.0); // ceil((7 + 40) / 4)
		assert_eq!(message.usage.cache_read, 0.0);
		assert_eq!(message.usage.cache_write, 0.0);
		assert_eq!(message.usage.output, 1.0);
		assert_eq!(message.usage.total_tokens, 13.0);
	}

	#[test]
	fn with_usage_estimate_tracks_prompt_cache_per_session() {
		let context = Context {
			system_prompt: Some("x".repeat(40)),
			messages: Vec::new(),
			tools: None,
		};
		let cache = Mutex::new(HashMap::new());
		let mut options = StreamOptions::default();
		options.session_id = Some("session-1".to_string());

		let first = with_usage_estimate(
			faux_assistant_message(FauxAssistantContent::Text("abcd".to_string()), None),
			&context,
			Some(&options),
			&cache,
		);
		assert_eq!(first.usage.cache_write, 12.0);
		assert_eq!(first.usage.cache_read, 0.0);
		assert_eq!(first.usage.input, 12.0);
		assert_eq!(first.usage.total_tokens, 25.0);

		let second = with_usage_estimate(
			faux_assistant_message(FauxAssistantContent::Text("abcd".to_string()), None),
			&context,
			Some(&options),
			&cache,
		);
		assert_eq!(second.usage.cache_read, 12.0);
		assert_eq!(second.usage.cache_write, 0.0);
		assert_eq!(second.usage.input, 0.0);
		assert_eq!(second.usage.total_tokens, 13.0);
	}

	#[test]
	fn with_usage_estimate_ignores_cache_when_retention_is_none() {
		let context = Context {
			system_prompt: Some("x".repeat(40)),
			messages: Vec::new(),
			tools: None,
		};
		let cache = Mutex::new(HashMap::new());
		let mut options = StreamOptions::default();
		options.session_id = Some("session-1".to_string());
		options.cache_retention = Some("none".to_string());
		let message = with_usage_estimate(
			faux_assistant_message(FauxAssistantContent::Text("abcd".to_string()), None),
			&context,
			Some(&options),
			&cache,
		);
		assert_eq!(message.usage.cache_write, 0.0);
		assert_eq!(message.usage.input, 12.0);
		assert!(cache.lock().unwrap().is_empty());
	}

	#[test]
	fn split_string_by_token_size_never_returns_empty_vec() {
		let chunks = split_string_by_token_size("", 3.0, 5.0);
		assert_eq!(chunks, vec![String::new()]);
		let chunks = split_string_by_token_size("abcdefghijklmnopqrstuvwxyz", 3.0, 5.0);
		assert!(chunks.len() >= 2);
		assert_eq!(chunks.concat(), "abcdefghijklmnopqrstuvwxyz");
	}

	#[test]
	fn clone_message_rewrites_api_provider_and_model() {
		let mut message = faux_assistant_message(FauxAssistantContent::Text("hi".to_string()), None);
		message.timestamp = 7;
		let cloned = clone_message(&message, "other-api", "other-provider", "other-model");
		assert_eq!(cloned.api, "other-api");
		assert_eq!(cloned.provider, "other-provider");
		assert_eq!(cloned.model, "other-model");
		assert_eq!(cloned.timestamp, 7);
	}

	#[test]
	fn create_error_message_uses_stop_reason_error() {
		let message = create_error_message("boom", "api", "provider", "model");
		assert_eq!(message.stop_reason, "error");
		assert_eq!(message.error_message.as_deref(), Some("boom"));
		assert!(message.content.is_empty());
	}

	#[test]
	fn create_aborted_message_keeps_content() {
		let partial = faux_assistant_message(FauxAssistantContent::Text("hi".to_string()), None);
		let aborted = create_aborted_message(&partial);
		assert_eq!(aborted.stop_reason, "aborted");
		assert_eq!(aborted.error_message.as_deref(), Some("Request was aborted"));
		assert_eq!(aborted.content.len(), 1);
	}

	#[tokio::test]
	async fn register_faux_provider_defaults_and_responses() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		let registration = register_faux_provider(None);
		assert!(registration.api.starts_with("faux:"));
		assert_eq!(registration.models.len(), 1);
		let model = registration.get_model();
		assert_eq!(model.id, "faux-1");
		assert_eq!(model.name, "Faux Model");
		assert_eq!(model.base_url, "http://localhost:0");
		assert_eq!(model.context_window, 128000.0);
		assert_eq!(model.max_tokens, 16384.0);
		assert_eq!(model.provider, "faux");
		assert!(registration.get_model_by_id("nope").is_none());
		assert!(registration.get_model_by_id("faux-1").is_some());

		registration.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
			FauxAssistantContent::Text("hello".to_string()),
			None,
		))]);
		assert_eq!(registration.get_pending_response_count(), 1);
		registration.append_responses(vec![FauxResponseStep::Message(faux_assistant_message(
			FauxAssistantContent::Text("again".to_string()),
			None,
		))]);
		assert_eq!(registration.get_pending_response_count(), 2);

		let stream = crate::api_registry::get_api_provider(&registration.api).expect("registered");
		let context = Context::default();
		let out = (stream.stream)(&model, &context, None);
		let mut events = Vec::new();
		while let Some(event) = out.next().await {
			events.push(event);
		}
		assert!(matches!(events.first(), Some(AssistantMessageEvent::Start { .. })));
		assert!(matches!(events.last(), Some(AssistantMessageEvent::Done { .. })));
		let final_message = out.result().await;
		assert_eq!(final_message.stop_reason, "stop");
		assert_eq!(registration.call_count(), 1);
		assert_eq!(registration.get_pending_response_count(), 1);

		registration.unregister();
		assert!(crate::api_registry::get_api_provider(&registration.api).is_none());
	}

	#[tokio::test]
	async fn register_faux_provider_uses_explicit_models_and_api() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
			api: Some("faux-api".to_string()),
			provider: Some("faux-provider".to_string()),
			models: Some(vec![
				FauxModelDefinition {
					id: "small".to_string(),
					name: None,
					reasoning: None,
					input: Some(vec!["text".to_string()]),
					cost: None,
					context_window: None,
					max_tokens: None,
				},
				FauxModelDefinition {
					id: "big".to_string(),
					name: Some("Big".to_string()),
					reasoning: Some(true),
					input: None,
					cost: None,
					context_window: Some(200000.0),
					max_tokens: Some(4096.0),
				},
			]),
			..Default::default()
		}));
		assert_eq!(registration.api, "faux-api");
		assert_eq!(registration.models.len(), 2);
		let first = registration.get_model();
		assert_eq!(first.id, "small");
		// `name ?? id`
		assert_eq!(first.name, "small");
		assert_eq!(first.context_window, 128000.0);
		assert_eq!(first.max_tokens, 16384.0);
		let second = registration.get_model_by_id("big").expect("big model");
		assert_eq!(second.name, "Big");
		assert!(second.reasoning);
		assert_eq!(second.context_window, 200000.0);
		assert_eq!(second.max_tokens, 4096.0);
		assert!(crate::api_registry::get_api_provider("faux-api").is_some());
		registration.unregister();
		assert!(crate::api_registry::get_api_provider("faux-api").is_none());
	}

	#[tokio::test]
	async fn faux_stream_reports_error_when_no_responses_queued() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
			provider: Some("faux-test-empty".to_string()),
			..Default::default()
		}));
		let model = registration.get_model();
		let stream = crate::api_registry::get_api_provider(&registration.api).expect("registered");
		let out = (stream.stream)(&model, &Context::default(), None);
		let mut events = Vec::new();
		while let Some(event) = out.next().await {
			events.push(event);
		}
		match events.last() {
			Some(AssistantMessageEvent::Error { reason, error }) => {
				assert_eq!(reason, "error");
				assert_eq!(error.error_message.as_deref(), Some("No more faux responses queued"));
			}
			other => panic!("expected error event, got {other:?}"),
		}
		registration.unregister();
	}

	#[tokio::test]
	async fn faux_stream_emits_text_deltas_in_order() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
			provider: Some("faux-test-deltas".to_string()),
			token_size: Some(FauxTokenSize {
				min: Some(1.0),
				max: Some(1.0),
			}),
			..Default::default()
		}));
		let model = registration.get_model();
		registration.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
			FauxAssistantContent::Text("abcdefgh".to_string()),
			None,
		))]);
		let stream = crate::api_registry::get_api_provider(&registration.api).expect("registered");
		let out = (stream.stream)(&model, &Context::default(), None);
		let mut kinds: Vec<&'static str> = Vec::new();
		let mut delta_text = String::new();
		while let Some(event) = out.next().await {
			match &event {
				AssistantMessageEvent::Start { .. } => kinds.push("start"),
				AssistantMessageEvent::TextStart { .. } => kinds.push("text_start"),
				AssistantMessageEvent::TextDelta { delta, .. } => {
					kinds.push("text_delta");
					delta_text.push_str(delta);
				}
				AssistantMessageEvent::TextEnd { content, .. } => {
					kinds.push("text_end");
					assert_eq!(content, "abcdefgh");
				}
				AssistantMessageEvent::Done { .. } => kinds.push("done"),
				other => panic!("unexpected event {other:?}"),
			}
		}
		assert_eq!(kinds.first(), Some(&"start"));
		assert_eq!(kinds.last(), Some(&"done"));
		assert_eq!(delta_text, "abcdefgh");
		registration.unregister();
	}

	/// TS faux.ts:436-462 wraps the whole stream task body in `try { ... } catch (error) { ... }`.
	/// When the body fails, the catch builds `createErrorMessage(...)` (faux.ts:457-458), pushes
	/// `{ type: "error", reason: "error", error: message }` and calls `outer.end(message)`
	/// (faux.ts:459-460), so a caller awaiting `next()`/`result()` always terminates.
	///
	/// The Rust equivalent of an unexpected failure is a panic inside the task: an uncaught panic
	/// kills the task, so the stream is never ended and the caller hangs forever.
	///
	/// Every wait below is bounded, so a hang FAILS the test instead of blocking the harness.
	#[tokio::test]
	async fn faux_stream_ends_with_error_when_the_response_factory_panics() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
			provider: Some("faux-test-panic-factory".to_string()),
			..Default::default()
		}));
		let model = registration.get_model();
		registration.set_responses(vec![FauxResponseStep::Factory(Arc::new(|_, _, _, _| {
			Box::pin(async move { panic!("faux factory exploded") })
		}))]);
		let stream = crate::api_registry::get_api_provider(&registration.api).expect("registered");
		let out = (stream.stream)(&model, &Context::default(), None);

		let mut events = Vec::new();
		loop {
			match tokio::time::timeout(std::time::Duration::from_secs(10), out.next()).await {
				Ok(Some(event)) => events.push(event),
				Ok(None) => break,
				Err(_) => panic!("faux stream never ended after the task body panicked (caller hangs)"),
			}
		}

		match events.last() {
			Some(AssistantMessageEvent::Error { reason, error }) => {
				assert_eq!(reason, "error");
				assert_eq!(error.stop_reason, "error");
				// faux.ts:274: `error instanceof Error ? error.message : String(error)`.
				assert_eq!(error.error_message.as_deref(), Some("faux factory exploded"));
			}
			other => panic!("expected the stream to end with an error event, got {other:?}"),
		}

		// `result()` must resolve too: `outer.end(message)` in the TS catch-all (faux.ts:460).
		let final_message = tokio::time::timeout(std::time::Duration::from_secs(10), out.result())
			.await
			.expect("faux stream result() hung after the task body panicked");
		assert_eq!(final_message.stop_reason, "error");
		assert_eq!(final_message.error_message.as_deref(), Some("faux factory exploded"));
		registration.unregister();
	}

	/// The TS `try` block also covers `await streamOptions?.onResponse?.(...)` (faux.ts:437-438),
	/// so a throwing hook ends the stream with an error event instead of hanging it.
	#[tokio::test]
	async fn faux_stream_ends_with_error_when_the_on_response_hook_panics() {
		let _guard = API_REGISTRY_TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		let registration = register_faux_provider(Some(RegisterFauxProviderOptions {
			provider: Some("faux-test-panic-hook".to_string()),
			..Default::default()
		}));
		let model = registration.get_model();
		registration.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
			FauxAssistantContent::Text("never streamed".to_string()),
			None,
		))]);
		let options = StreamOptions {
			on_response: Some(Arc::new(|_, _| Box::pin(async { panic!("faux onResponse exploded") }))),
			..Default::default()
		};
		let stream = crate::api_registry::get_api_provider(&registration.api).expect("registered");
		let out = (stream.stream)(&model, &Context::default(), Some(&options));

		let first = tokio::time::timeout(std::time::Duration::from_secs(10), out.next())
			.await
			.expect("faux stream never ended after the onResponse hook panicked (caller hangs)");
		match first {
			Some(AssistantMessageEvent::Error { reason, error }) => {
				assert_eq!(reason, "error");
				assert_eq!(error.error_message.as_deref(), Some("faux onResponse exploded"));
			}
			other => panic!("expected an error event first, got {other:?}"),
		}

		let after = tokio::time::timeout(std::time::Duration::from_secs(10), out.next())
			.await
			.expect("faux stream did not end after the onResponse hook panicked");
		assert!(after.is_none(), "expected the stream to be finished, got {after:?}");
		registration.unregister();
	}
}
