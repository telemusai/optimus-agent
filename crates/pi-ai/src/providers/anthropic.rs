//! Port of packages/ai/src/providers/anthropic.ts
//!
//! The TypeScript uses `@anthropic-ai/sdk` (0.91.1). This port builds the same
//! `POST /v1/messages` request itself with `reqwest` and parses the SSE stream
//! locally, keeping the SDK's wire shape, header merge order and error text.

use std::collections::HashSet;

use futures::StreamExt;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::cache_pricing::{
	get_anthropic_cache_write_cost, has_standard_anthropic_cache_pricing, AnthropicCacheCreationUsage,
};
use crate::env_api_keys::get_env_api_key;
use crate::models::{calculate_cost, clamp_thinking_level, CostOverrides};
use crate::types::{
	AssistantMessage, AssistantMessageEvent, CacheRetention, ContentBlock, Context, ImageOrTextContent,
	Message, Model, SimpleStreamOptions, StopReason, StreamOptions, TextContent, ThinkingContent, Tool, ToolCall,
	ToolResultMessage, UserContent,
};
use crate::utils::event_stream::{create_assistant_message_event_stream, AssistantMessageEventStream};
use crate::utils::headers::header_map_to_record;
use crate::utils::json_parse::{parse_json_with_repair, parse_streaming_json};
use crate::utils::now_ms;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::stream_failure::{
	classify_stream_failure, format_stream_failure_message, record_stream_failure, stream_failure_from_stop_reason,
	stream_failure_message, truncate_raw_payload, StreamFailureError, StreamFailureInfo, ThrownStreamError,
};

use crate::providers::cloudflare::resolve_cloudflare_base_url;
use crate::providers::github_copilot_headers::{
	build_copilot_dynamic_headers, has_copilot_vision_input, CopilotDynamicHeaderParams,
};
use crate::providers::opencode_headers::with_opencode_headers;
use crate::providers::simple_options::{adjust_max_tokens_for_thinking, build_base_options};
use crate::providers::transform_messages::try_transform_messages;

/// The pinned `@anthropic-ai/sdk` version, used for the `User-Agent` header
/// (`${this.constructor.name}/JS ${VERSION}` -> `Anthropic/JS 0.91.1`).
const ANTHROPIC_SDK_VERSION: &str = "0.91.1";
const ANTHROPIC_USER_AGENT: &str = "Anthropic/JS 0.91.1";

/// Resolve cache retention preference.
/// Defaults to "short" and uses PI_CACHE_RETENTION for backward compatibility.
fn resolve_cache_retention(cache_retention: Option<&CacheRetention>) -> CacheRetention {
	if let Some(retention) = cache_retention {
		if !retention.is_empty() {
			return retention.clone();
		}
	}
	if std::env::var("PI_CACHE_RETENTION").ok().as_deref() == Some("long") {
		return "long".to_string();
	}
	"short".to_string()
}

/// TS: `CacheControlEphemeral` (`{ type: "ephemeral", ttl?: "1h" }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheControlEphemeral {
	#[serde(rename = "type")]
	pub type_: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub ttl: Option<String>,
}

/// TS: `getCacheControl(model, cacheRetention)`.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheControlResult {
	pub retention: CacheRetention,
	pub cache_control: Option<CacheControlEphemeral>,
}

fn get_cache_control(model: &Model, cache_retention: Option<&CacheRetention>) -> CacheControlResult {
	let retention = resolve_cache_retention(cache_retention);
	if retention == "none" {
		return CacheControlResult {
			retention,
			cache_control: None,
		};
	}
	let ttl = if retention == "long" && get_anthropic_compat(model).supports_long_cache_retention {
		Some("1h".to_string())
	} else {
		None
	};
	CacheControlResult {
		retention,
		cache_control: Some(CacheControlEphemeral {
			type_: "ephemeral".to_string(),
			ttl,
		}),
	}
}

// Stealth mode: Mimic Claude Code's tool naming exactly
const CLAUDE_CODE_VERSION: &str = "2.1.281";

// Claude Code 2.x tool names (canonical casing)
// Source: https://cchistory.mariozechner.at/data/prompts-2.1.11.md
// To update: https://github.com/badlogic/cchistory
const CLAUDE_CODE_TOOLS: [&str; 17] = [
	"Read",
	"Write",
	"Edit",
	"Bash",
	"Grep",
	"Glob",
	"AskUserQuestion",
	"EnterPlanMode",
	"ExitPlanMode",
	"KillShell",
	"NotebookEdit",
	"Skill",
	"Task",
	"TaskOutput",
	"TodoWrite",
	"WebFetch",
	"WebSearch",
];

/// TS: `ccToolLookup = new Map(claudeCodeTools.map((t) => [t.toLowerCase(), t]))`
fn cc_tool_lookup(name: &str) -> Option<&'static str> {
	let lower = name.to_lowercase();
	CLAUDE_CODE_TOOLS
		.iter()
		.find(|tool| tool.to_lowercase() == lower)
		.copied()
}

/// TS: `toClaudeCodeName(name)`
fn to_claude_code_name(name: &str) -> String {
	match cc_tool_lookup(name) {
		Some(canonical) => canonical.to_string(),
		None => name.to_string(),
	}
}

/// TS: `fromClaudeCodeName(name, tools?)`
fn from_claude_code_name(name: &str, tools: Option<&Vec<Tool>>) -> String {
	if let Some(tools) = tools {
		if !tools.is_empty() {
			let lower_name = name.to_lowercase();
			if let Some(matched_tool) = tools.iter().find(|tool| tool.name.to_lowercase() == lower_name) {
				return matched_tool.name.clone();
			}
		}
	}
	name.to_string()
}

/// Convert content blocks to Anthropic API format.
///
/// TS: `convertContentBlocks(content)` - returns a bare string when there are no
/// images, otherwise an array of `text` / `image` blocks.
fn convert_content_blocks(content: &[ImageOrTextContent]) -> Value {
	let has_images = content.iter().any(|c| matches!(c, ImageOrTextContent::Image(_)));
	if !has_images {
		let joined = content
			.iter()
			.map(|c| match c {
				ImageOrTextContent::Text(text) => text.text.clone(),
				ImageOrTextContent::Image(_) => String::new(),
			})
			.collect::<Vec<_>>()
			.join("\n");
		return Value::String(sanitize_surrogates(&joined));
	}

	let mut blocks: Vec<Value> = content
		.iter()
		.map(|block| match block {
			ImageOrTextContent::Text(text) => {
				let mut object = Map::new();
				object.insert("type".to_string(), Value::String("text".to_string()));
				object.insert("text".to_string(), Value::String(sanitize_surrogates(&text.text)));
				Value::Object(object)
			}
			ImageOrTextContent::Image(image) => {
				let mut source = Map::new();
				source.insert("type".to_string(), Value::String("base64".to_string()));
				source.insert("media_type".to_string(), Value::String(image.mime_type.clone()));
				source.insert("data".to_string(), Value::String(image.data.clone()));
				let mut object = Map::new();
				object.insert("type".to_string(), Value::String("image".to_string()));
				object.insert("source".to_string(), Value::Object(source));
				Value::Object(object)
			}
		})
		.collect();

	let has_text = blocks.iter().any(|b| b.get("type").and_then(Value::as_str) == Some("text"));
	if !has_text {
		let mut object = Map::new();
		object.insert("type".to_string(), Value::String("text".to_string()));
		object.insert("text".to_string(), Value::String("(see attached image)".to_string()));
		blocks.insert(0, Value::Object(object));
	}

	Value::Array(blocks)
}

pub type AnthropicEffort = String;

pub type AnthropicThinkingDisplay = String;

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// TS: `getAnthropicCompat(model): Required<AnthropicMessagesCompat>`
#[derive(Debug, Clone, PartialEq)]
pub struct RequiredAnthropicMessagesCompat {
	pub supports_eager_tool_input_streaming: bool,
	pub supports_long_cache_retention: bool,
}

fn get_anthropic_compat(model: &Model) -> RequiredAnthropicMessagesCompat {
	match model.compat_anthropic() {
		Some(compat) => RequiredAnthropicMessagesCompat {
			supports_eager_tool_input_streaming: compat.supports_eager_tool_input_streaming.unwrap_or(true),
			supports_long_cache_retention: compat.supports_long_cache_retention.unwrap_or(true),
		},
		None => RequiredAnthropicMessagesCompat {
			supports_eager_tool_input_streaming: true,
			supports_long_cache_retention: true,
		},
	}
}

/// TS: `interface AnthropicOptions extends StreamOptions`.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AnthropicOptions {
	#[serde(flatten)]
	pub stream: StreamOptions,
	/// Enable extended thinking. For Opus 4.6 and Sonnet 4.6: uses adaptive thinking
	/// (model decides when/how much to think). For older models: uses budget-based
	/// thinking with thinkingBudgetTokens.
	pub thinking_enabled: Option<bool>,
	/// Token budget for extended thinking (older models only).
	pub thinking_budget_tokens: Option<f64>,
	/// Effort level for adaptive thinking (Opus 4.6+, Sonnet 4.6, Fable/Mythos).
	pub effort: Option<AnthropicEffort>,
	/// Controls how thinking content is returned in API responses.
	pub thinking_display: Option<AnthropicThinkingDisplay>,
	pub interleaved_thinking: Option<bool>,
	/// `"auto" | "any" | "none" | { type: "tool"; name: string }`
	pub tool_choice: Option<Value>,
	/// TS: `client?: Anthropic` - a pre-built client instance. Rust has no SDK
	/// client object; the port exposes the same branch as an explicit flag so
	/// callers can skip internal client construction (used by AnthropicVertex-style
	/// clients) and supply the transport details themselves.
	pub client: Option<AnthropicClientOverride>,
}

/// TS: `client?: Anthropic` - the fields of an injected SDK client that the
/// provider actually reads (base URL and auth headers).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AnthropicClientOverride {
	pub base_url: Option<String>,
	pub api_key: Option<String>,
	pub auth_token: Option<String>,
	pub headers: Option<IndexMap<String, String>>,
}

impl AnthropicOptions {
	/// TS: the caller passes `StreamOptions & Record<string, unknown>`; this keeps the
	/// non-serializable fields (signal, on_payload, on_response, on_usage_observation).
	pub fn from_base(base: &StreamOptions) -> Self {
		Self {
			stream: base.clone(),
			thinking_enabled: None,
			thinking_budget_tokens: None,
			effort: None,
			thinking_display: None,
			interleaved_thinking: None,
			tool_choice: None,
			client: None,
		}
	}
}

impl std::fmt::Debug for AnthropicOptions {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("AnthropicOptions")
			.field("stream", &self.stream)
			.field("thinking_enabled", &self.thinking_enabled)
			.field("thinking_budget_tokens", &self.thinking_budget_tokens)
			.field("effort", &self.effort)
			.field("thinking_display", &self.thinking_display)
			.field("interleaved_thinking", &self.interleaved_thinking)
			.field("tool_choice", &self.tool_choice)
			.field("client", &self.client)
			.finish()
	}
}

/// TS: `mergeHeaders(...headerSources)`.
///
/// `Record<string, string | null>`: `None` clears a previously merged header and
/// keeps the key (JS object assignment semantics).
fn merge_headers(sources: Vec<Option<IndexMap<String, Option<String>>>>) -> IndexMap<String, Option<String>> {
	let mut merged: IndexMap<String, Option<String>> = IndexMap::new();
	for headers in sources.into_iter().flatten() {
		for (name, value) in headers {
			merged.insert(name, value);
		}
	}
	merged
}

/// TS: `Record<string, string>` -> the `string | null` merge input.
fn record_to_nullable(headers: Option<&IndexMap<String, String>>) -> Option<IndexMap<String, Option<String>>> {
	headers.map(|headers| {
		headers
			.iter()
			.map(|(name, value)| (name.clone(), Some(value.clone())))
			.collect()
	})
}

/// TS: `ServerSentEvent`.
#[derive(Debug, Clone, PartialEq)]
struct ServerSentEvent {
	event: Option<String>,
	data: String,
	raw: Vec<String>,
}

/// TS: `SseDecoderState`.
#[derive(Debug, Clone, Default)]
struct SseDecoderState {
	event: Option<String>,
	data: Vec<String>,
	raw: Vec<String>,
}

const ANTHROPIC_MESSAGE_EVENTS: [&str; 6] = [
	"message_start",
	"message_delta",
	"message_stop",
	"content_block_start",
	"content_block_delta",
	"content_block_stop",
];

/// TS: `flushSseEvent(state)`.
fn flush_sse_event(state: &mut SseDecoderState) -> Option<ServerSentEvent> {
	if state.event.is_none() && state.data.is_empty() {
		return None;
	}

	let event = ServerSentEvent {
		event: state.event.clone(),
		data: state.data.join("\n"),
		raw: state.raw.clone(),
	};
	state.event = None;
	state.data = Vec::new();
	state.raw = Vec::new();
	Some(event)
}

/// TS: `decodeSseLine(line, state)`.
fn decode_sse_line(line: &str, state: &mut SseDecoderState) -> Option<ServerSentEvent> {
	if line.is_empty() {
		return flush_sse_event(state);
	}

	state.raw.push(line.to_string());
	if line.starts_with(':') {
		return None;
	}

	let delimiter_index = line.find(':');
	let field_name = match delimiter_index {
		Some(index) => &line[..index],
		None => line,
	};
	let mut value = match delimiter_index {
		Some(index) => &line[index + 1..],
		None => "",
	};
	if let Some(stripped) = value.strip_prefix(' ') {
		value = stripped;
	}

	if field_name == "event" {
		state.event = Some(value.to_string());
	} else if field_name == "data" {
		state.data.push(value.to_string());
	}

	None
}

/// TS: `nextLineBreakIndex(text)` - `-1` (not found) becomes `None`.
fn next_line_break_index(text: &str) -> Option<usize> {
	let carriage_return_index = text.find('\r');
	let newline_index = text.find('\n');
	match (carriage_return_index, newline_index) {
		(None, newline) => newline,
		(carriage_return, None) => carriage_return,
		(Some(carriage_return), Some(newline)) => Some(carriage_return.min(newline)),
	}
}

/// TS: `consumeLine(text)`.
fn consume_line(text: &str) -> Option<(String, String)> {
	let line_break_index = next_line_break_index(text)?;

	let mut next_index = line_break_index + 1;
	if text.as_bytes()[line_break_index] == b'\r'
		&& text.as_bytes().get(next_index).copied() == Some(b'\n')
	{
		next_index += 1;
	}

	Some((text[..line_break_index].to_string(), text[next_index..].to_string()))
}

/// Streaming UTF-8 decode, mirroring `TextDecoder.decode(value, { stream: true })`.
///
/// Bytes that form an incomplete multi-byte sequence stay in `pending` until the
/// next chunk; invalid sequences become U+FFFD like the WHATWG decoder.
fn decode_utf8_stream(pending: &mut Vec<u8>) -> String {
	let mut out = String::new();
	loop {
		match std::str::from_utf8(pending) {
			Ok(valid) => {
				out.push_str(valid);
				pending.clear();
				return out;
			}
			Err(error) => {
				let valid_up_to = error.valid_up_to();
				out.push_str(&String::from_utf8_lossy(&pending[..valid_up_to]));
				pending.drain(..valid_up_to);
				match error.error_len() {
					// Incomplete trailing sequence: wait for more bytes.
					None => return out,
					Some(error_len) => {
						pending.drain(..error_len);
						out.push('\u{FFFD}');
					}
				}
			}
		}
	}
}

/// TS: `decoder.decode()` at the end of the body.
fn decode_utf8_final(pending: &mut Vec<u8>) -> String {
	let out = decode_utf8_stream(pending);
	if pending.is_empty() {
		return out;
	}
	pending.clear();
	format!("{}\u{FFFD}", out)
}

/// Error shape for the streaming body: the TypeScript throws either a plain
/// `Error` or a `StreamFailureError`.
#[derive(Debug, Clone, PartialEq)]
pub enum AnthropicStreamError {
	Failure(StreamFailureError),
	Message(String),
}

impl AnthropicStreamError {
	fn as_thrown(&self) -> ThrownStreamError<'_> {
		match self {
			AnthropicStreamError::Failure(failure) => ThrownStreamError::Failure(failure),
			AnthropicStreamError::Message(message) => ThrownStreamError::Message(message),
		}
	}

	fn message(&self) -> String {
		match self {
			AnthropicStreamError::Failure(failure) => failure.message.clone(),
			AnthropicStreamError::Message(message) => message.clone(),
		}
	}
}

type ByteStream = std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

/// TS: `iterateSseMessages(body, signal)`.
struct SseMessageReader {
	chunks: ByteStream,
	byte_pending: Vec<u8>,
	buffer: String,
	state: SseDecoderState,
	finished: bool,
	final_decoded: bool,
	trailing_flushed: bool,
	signal: Option<tokio_util::sync::CancellationToken>,
}

impl SseMessageReader {
	fn new(chunks: ByteStream, signal: Option<tokio_util::sync::CancellationToken>) -> Self {
		Self {
			chunks,
			byte_pending: Vec::new(),
			buffer: String::new(),
			state: SseDecoderState::default(),
			finished: false,
			final_decoded: false,
			trailing_flushed: false,
			signal,
		}
	}

	async fn next(&mut self) -> Result<Option<ServerSentEvent>, AnthropicStreamError> {
		loop {
			if let Some((line, rest)) = consume_line(&self.buffer) {
				self.buffer = rest;
				if let Some(event) = decode_sse_line(&line, &mut self.state) {
					return Ok(Some(event));
				}
				continue;
			}

			if self.finished {
				if !self.final_decoded {
					let tail = decode_utf8_final(&mut self.byte_pending);
					self.buffer.push_str(&tail);
					self.final_decoded = true;
					continue;
				}
				if !self.buffer.is_empty() {
					let line = std::mem::take(&mut self.buffer);
					if let Some(event) = decode_sse_line(&line, &mut self.state) {
						return Ok(Some(event));
					}
					continue;
				}
				if !self.trailing_flushed {
					self.trailing_flushed = true;
					if let Some(event) = flush_sse_event(&mut self.state) {
						return Ok(Some(event));
					}
				}
				return Ok(None);
			}

			if let Some(signal) = &self.signal {
				if signal.is_cancelled() {
					return Err(AnthropicStreamError::Message("Request was aborted".to_string()));
				}
			}

			match self.chunks.next().await {
				None => {
					self.finished = true;
				}
				Some(Err(error)) => {
					return Err(AnthropicStreamError::Message(error.to_string()));
				}
				Some(Ok(bytes)) => {
					self.byte_pending.extend_from_slice(&bytes);
					let decoded = decode_utf8_stream(&mut self.byte_pending);
					self.buffer.push_str(&decoded);
				}
			}
		}
	}
}

/// Turn an in-stream `error` SSE event (how Anthropic delivers overloads etc.) into
/// a classified failure.
fn anthropic_sse_error(data: &str, request_id: Option<&str>) -> StreamFailureError {
	let mut error_type: Option<String> = None;
	// Both arms below assign `detail`, like the TS `let detail` + try/catch.
	let detail: Option<String>;
	let mut request_id = request_id.map(str::to_string);
	match parse_json_with_repair(data) {
		Ok(parsed) => {
			error_type = parsed
				.pointer("/error/type")
				.and_then(Value::as_str)
				.map(str::to_string);
			detail = parsed
				.pointer("/error/message")
				.and_then(Value::as_str)
				.map(str::to_string);
			// Proxies may strip the request-id header; the error body carries it too.
			if request_id.is_none() {
				if let Some(body_request_id) = parsed.get("request_id").and_then(Value::as_str) {
					request_id = Some(body_request_id.to_string());
				}
			}
		}
		Err(_) => {
			detail = Some(data.to_string());
		}
	}
	let info = StreamFailureInfo {
		kind: classify_stream_failure(error_type.as_deref(), None).to_string(),
		provider_error_type: error_type,
		status: None,
		request_id,
		retry_after_ms: None,
		raw: Some(truncate_raw_payload(data)),
	};
	StreamFailureError::new(stream_failure_message(&info, detail.as_deref()), info)
}

/// TS: `iterateAnthropicEvents(response, signal, requestId)`.
struct AnthropicEventIterator {
	reader: SseMessageReader,
	saw_message_start: bool,
	saw_message_end: bool,
	request_id: Option<String>,
	finished: bool,
}

impl AnthropicEventIterator {
	fn new(reader: SseMessageReader, request_id: Option<String>) -> Self {
		Self {
			reader,
			saw_message_start: false,
			saw_message_end: false,
			request_id,
			finished: false,
		}
	}

	/// Yields the parsed `RawMessageStreamEvent` JSON objects.
	async fn next(&mut self) -> Result<Option<Value>, AnthropicStreamError> {
		if self.finished {
			return Ok(None);
		}

		loop {
			let sse = match self.reader.next().await? {
				Some(sse) => sse,
				None => {
					self.finished = true;
					if self.saw_message_start && !self.saw_message_end {
						return Err(AnthropicStreamError::Failure(StreamFailureError::new(
							"Anthropic stream ended before message_stop",
							StreamFailureInfo {
								kind: "malformed_response".to_string(),
								request_id: self.request_id.clone(),
								..Default::default()
							},
						)));
					}
					return Ok(None);
				}
			};

			if sse.event.as_deref() == Some("error") {
				return Err(AnthropicStreamError::Failure(anthropic_sse_error(
					&sse.data,
					self.request_id.as_deref(),
				)));
			}

			if !ANTHROPIC_MESSAGE_EVENTS.contains(&sse.event.as_deref().unwrap_or("")) {
				continue;
			}

			match parse_json_with_repair(&sse.data) {
				Ok(event) => {
					match event.get("type").and_then(Value::as_str) {
						Some("message_start") => self.saw_message_start = true,
						Some("message_stop") => self.saw_message_end = true,
						_ => {}
					}
					return Ok(Some(event));
				}
				Err(message) => {
					return Err(AnthropicStreamError::Failure(StreamFailureError::new(
						format!(
							"Could not parse Anthropic SSE event {}: {}; data={}; raw={}",
							sse.event.as_deref().unwrap_or(""),
							message,
							sse.data,
							sse.raw.join("\\n")
						),
						StreamFailureInfo {
							kind: "malformed_response".to_string(),
							request_id: self.request_id.clone(),
							raw: Some(truncate_raw_payload(&sse.data)),
							..Default::default()
						},
					)));
				}
			}
		}
	}
}

/// TS: `interface Block` - `(ThinkingContent | TextContent | (ToolCall & { partialJson: string })) & { index: number }`.
///
/// The `index` field is a streaming-only scratch property: it is deleted at
/// `content_block_stop` and in the catch block, so it is stored beside the
/// content block instead of inside it.
#[derive(Debug, Clone, PartialEq)]
enum AnthropicBlock {
	Text { text: String, index: i64 },
	Thinking {
		thinking: String,
		thinking_signature: String,
		redacted: Option<bool>,
		index: i64,
	},
	ToolCall {
		tool_call: ToolCall,
		partial_json: String,
		index: i64,
	},
}

impl AnthropicBlock {
	fn index(&self) -> i64 {
		match self {
			AnthropicBlock::Text { index, .. } => *index,
			AnthropicBlock::Thinking { index, .. } => *index,
			AnthropicBlock::ToolCall { index, .. } => *index,
		}
	}

	fn to_content_block(&self) -> ContentBlock {
		match self {
			AnthropicBlock::Text { text, .. } => ContentBlock::Text(TextContent::new(text.clone())),
			AnthropicBlock::Thinking {
				thinking,
				thinking_signature,
				redacted,
				..
			} => ContentBlock::Thinking(ThinkingContent {
				type_: crate::types::THINKING_CONTENT_TYPE.to_string(),
				thinking: thinking.clone(),
				thinking_signature: Some(thinking_signature.clone()),
				redacted: *redacted,
			}),
			AnthropicBlock::ToolCall { tool_call, .. } => ContentBlock::ToolCall(tool_call.clone()),
		}
	}
}

/// `blocks.findIndex((b) => b.index === event.index)` - `None` when no block has
/// that index (the TypeScript then reads `undefined` and skips the branch).
fn find_block_index(blocks: &[AnthropicBlock], index: i64) -> Option<usize> {
	blocks.iter().position(|block| block.index() == index)
}

/// Sync the streaming scratch blocks back into `output.content`.
///
/// The TypeScript keeps one array where the `index` field is deleted at
/// `content_block_stop` and in the catch block. The port keeps `output.content`
/// index-clean at all times and projects the scratch state onto it whenever the
/// partial message is pushed.
fn sync_output_content(output: &mut AssistantMessage, blocks: &[AnthropicBlock]) {
	output.content = blocks.iter().map(AnthropicBlock::to_content_block).collect();
}

/// TS: `streamAnthropic`.
pub fn stream_anthropic(
	model: &Model,
	context: &Context,
	options: Option<AnthropicOptions>,
) -> AssistantMessageEventStream {
	let stream = create_assistant_message_event_stream();
	let out = stream.clone();
	let model = model.clone();
	let context = context.clone();
	let options = options.unwrap_or_default();

	tokio::spawn(async move {
		let mut output = AssistantMessage {
			role: "assistant".to_string(),
			content: Vec::new(),
			api: model.api.clone(),
			provider: model.provider.clone(),
			model: model.id.clone(),
			usage: crate::types::Usage {
				input: 0.0,
				output: 0.0,
				cache_read: 0.0,
				cache_write: 0.0,
				total_tokens: 0.0,
				cost: crate::types::UsageCost {
					input: 0.0,
					output: 0.0,
					cache_read: 0.0,
					cache_write: 0.0,
					total: 0.0,
				},
			},
			stop_reason: "stop".to_string(),
			timestamp: now_ms(),
			..Default::default()
		};

		let result = run_stream_anthropic(&out, &mut output, &model, &context, &options).await;

		match result {
			Ok(()) => {}
			Err(error) => {
				output.stop_reason = if options
					.stream
					.signal
					.as_ref()
					.map(|signal| signal.is_cancelled())
					.unwrap_or(false)
				{
					"aborted".to_string()
				} else {
					"error".to_string()
				};
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

/// The TS `(async () => { try { ... } catch (error) { ... } })()` body.
async fn run_stream_anthropic(
	out: &AssistantMessageEventStream,
	output: &mut AssistantMessage,
	model: &Model,
	context: &Context,
	options: &AnthropicOptions,
) -> Result<(), AnthropicStreamError> {
	let is_oauth;
	let client: AnthropicClientOverride;

	if let Some(client_override) = options.client.clone() {
		client = client_override;
		is_oauth = false;
	} else {
		let api_key = options
			.stream
			.api_key
			.clone()
			.or_else(|| get_env_api_key(&model.provider))
			.unwrap_or_default();

		let mut copilot_dynamic_headers: Option<IndexMap<String, String>> = None;
		if model.provider == "github-copilot" {
			let has_images = has_copilot_vision_input(&context.messages);
			copilot_dynamic_headers = Some(build_copilot_dynamic_headers(CopilotDynamicHeaderParams {
				messages: &context.messages,
				has_images,
			}));
		}

		let created = create_client(
			model,
			&api_key,
			options.interleaved_thinking.unwrap_or(true),
			should_use_fine_grained_tool_streaming_beta(model, context),
			options.stream.headers.as_ref(),
			copilot_dynamic_headers.as_ref(),
			options.stream.session_id.as_deref(),
		)?;
		client = created.client;
		is_oauth = created.is_oauth_token;
	}

	let cache_control_result = get_cache_control(model, options.stream.cache_retention.as_ref());
	let cache_control = cache_control_result.cache_control.clone();
	let uses_anthropic_cache_pricing = has_standard_anthropic_cache_pricing(model);
	let mut cache_write_cost = if cache_control.is_some() && uses_anthropic_cache_pricing {
		Some(get_anthropic_cache_write_cost(
			model.cost.input,
			if cache_control.as_ref().and_then(|c| c.ttl.as_deref()) == Some("1h") {
				"1h"
			} else {
				"5m"
			},
			None,
		))
	} else {
		None
	};

	let mut params = build_params(model, context, is_oauth, options, cache_control.as_ref())?;
	if let Some(on_payload) = options.stream.on_payload.clone() {
		let next_params = on_payload(params.clone(), model).await;
		if let Some(next_params) = next_params {
			params = next_params;
		}
	}

	let response = send_messages_request(&client, &params, model, options).await?;
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
	let request_id = response
		.headers()
		.get("request-id")
		.and_then(|value| value.to_str().ok())
		.map(str::to_string);
	out.push(AssistantMessageEvent::Start {
		partial: output.clone(),
	});

	let mut blocks: Vec<AnthropicBlock> = Vec::new();

	let chunks: ByteStream = Box::pin(response.bytes_stream());
	let reader = SseMessageReader::new(chunks, options.stream.signal.clone());
	let mut events = AnthropicEventIterator::new(reader, request_id.clone());

	while let Some(event) = events.next().await? {
		let event_type = event.get("type").and_then(Value::as_str).unwrap_or_default().to_string();
		if event_type == "message_start" {
			output.response_id = event.pointer("/message/id").and_then(Value::as_str).map(str::to_string);
			// Capture initial token usage from message_start event.
			// This ensures we have input token counts even if the stream is aborted early.
			output.usage.input = number_or_zero(event.pointer("/message/usage/input_tokens"));
			output.usage.output = number_or_zero(event.pointer("/message/usage/output_tokens"));
			output.usage.cache_read = number_or_zero(event.pointer("/message/usage/cache_read_input_tokens"));
			output.usage.cache_write = number_or_zero(event.pointer("/message/usage/cache_creation_input_tokens"));
			output.usage.total_tokens =
				output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write;
			if cache_control.is_some() && uses_anthropic_cache_pricing {
				let cache_creation = parse_cache_creation(event.pointer("/message/usage/cache_creation"));
				cache_write_cost = Some(get_anthropic_cache_write_cost(
					model.cost.input,
					if cache_control.as_ref().and_then(|c| c.ttl.as_deref()) == Some("1h") {
						"1h"
					} else {
						"5m"
					},
					cache_creation.as_ref(),
				));
			}
			let overrides = cache_write_cost.map(|cache_write| CostOverrides {
				cache_write: Some(cache_write),
			});
			calculate_cost(model, &mut output.usage, overrides.as_ref());
		} else if event_type == "content_block_start" {
			let index = number_or_zero(event.get("index"));
			match event.pointer("/content_block/type").and_then(Value::as_str) {
				Some("text") => {
					blocks.push(AnthropicBlock::Text {
						text: String::new(),
						index: index as i64,
					});
					sync_output_content(output, &blocks);
					out.push(AssistantMessageEvent::TextStart {
						content_index: output.content.len() - 1,
						partial: output.clone(),
					});
				}
				Some("thinking") => {
					blocks.push(AnthropicBlock::Thinking {
						thinking: String::new(),
						thinking_signature: String::new(),
						redacted: None,
						index: index as i64,
					});
					sync_output_content(output, &blocks);
					out.push(AssistantMessageEvent::ThinkingStart {
						content_index: output.content.len() - 1,
						partial: output.clone(),
					});
				}
				Some("redacted_thinking") => {
					blocks.push(AnthropicBlock::Thinking {
						thinking: "[Reasoning redacted]".to_string(),
						thinking_signature: event
							.pointer("/content_block/data")
							.and_then(Value::as_str)
							.unwrap_or_default()
							.to_string(),
						redacted: Some(true),
						index: index as i64,
					});
					sync_output_content(output, &blocks);
					out.push(AssistantMessageEvent::ThinkingStart {
						content_index: output.content.len() - 1,
						partial: output.clone(),
					});
				}
				Some("tool_use") => {
					let name = event
						.pointer("/content_block/name")
						.and_then(Value::as_str)
						.unwrap_or_default()
						.to_string();
					let name = if is_oauth {
						from_claude_code_name(&name, context.tools.as_ref())
					} else {
						name
					};
					let arguments = match event.pointer("/content_block/input") {
						Some(Value::Object(object)) => object.clone(),
						_ => Map::new(),
					};
					blocks.push(AnthropicBlock::ToolCall {
						tool_call: ToolCall::new(
							event
								.pointer("/content_block/id")
								.and_then(Value::as_str)
								.unwrap_or_default(),
							name,
							arguments,
						),
						partial_json: String::new(),
						index: index as i64,
					});
					sync_output_content(output, &blocks);
					out.push(AssistantMessageEvent::ToolCallStart {
						content_index: output.content.len() - 1,
						partial: output.clone(),
					});
				}
				_ => {}
			}
		} else if event_type == "content_block_delta" {
			let delta_index = number_or_zero(event.get("index")) as i64;
			match event.pointer("/delta/type").and_then(Value::as_str) {
				Some("text_delta") => {
					if let Some(index) = find_block_index(&blocks, delta_index) {
						let delta = event
							.pointer("/delta/text")
							.and_then(Value::as_str)
							.unwrap_or_default()
							.to_string();
						let applied = match &mut blocks[index] {
							AnthropicBlock::Text { text, .. } => {
								text.push_str(&delta);
								true
							}
							_ => false,
						};
						if applied {
							sync_output_content(output, &blocks);
							out.push(AssistantMessageEvent::TextDelta {
								content_index: index,
								delta,
								partial: output.clone(),
							});
						}
					}
				}
				Some("thinking_delta") => {
					if let Some(index) = find_block_index(&blocks, delta_index) {
						let delta = event
							.pointer("/delta/thinking")
							.and_then(Value::as_str)
							.unwrap_or_default()
							.to_string();
						let applied = match &mut blocks[index] {
							AnthropicBlock::Thinking { thinking, .. } => {
								thinking.push_str(&delta);
								true
							}
							_ => false,
						};
						if applied {
							sync_output_content(output, &blocks);
							out.push(AssistantMessageEvent::ThinkingDelta {
								content_index: index,
								delta,
								partial: output.clone(),
							});
						}
					}
				}
				Some("input_json_delta") => {
					if let Some(index) = find_block_index(&blocks, delta_index) {
						let delta = event
							.pointer("/delta/partial_json")
							.and_then(Value::as_str)
							.unwrap_or_default()
							.to_string();
						let applied = match &mut blocks[index] {
							AnthropicBlock::ToolCall {
								tool_call,
								partial_json,
								..
							} => {
								partial_json.push_str(&delta);
								tool_call.arguments = match parse_streaming_json(Some(partial_json.as_str())) {
									Value::Object(object) => object,
									_ => Map::new(),
								};
								true
							}
							_ => false,
						};
						if applied {
							sync_output_content(output, &blocks);
							out.push(AssistantMessageEvent::ToolCallDelta {
								content_index: index,
								delta,
								partial: output.clone(),
							});
						}
					}
				}
				Some("signature_delta") => {
					if let Some(index) = find_block_index(&blocks, delta_index) {
						let delta = event
							.pointer("/delta/signature")
							.and_then(Value::as_str)
							.unwrap_or_default()
							.to_string();
						let applied = match &mut blocks[index] {
							AnthropicBlock::Thinking {
								thinking_signature, ..
							} => {
								thinking_signature.push_str(&delta);
								true
							}
							_ => false,
						};
						if applied {
							sync_output_content(output, &blocks);
						}
					}
				}
				_ => {}
			}
		} else if event_type == "content_block_stop" {
			let stop_index = number_or_zero(event.get("index")) as i64;
			let found = find_block_index(&blocks, stop_index);
			if let Some(index) = found {
				// The TypeScript deletes the scratch `index` here; the port simply
				// stops using it for the block.
				match &blocks[index] {
					AnthropicBlock::Text { text, .. } => {
						out.push(AssistantMessageEvent::TextEnd {
							content_index: index,
							content: text.clone(),
							partial: output.clone(),
						});
					}
					AnthropicBlock::Thinking { thinking, .. } => {
						out.push(AssistantMessageEvent::ThinkingEnd {
							content_index: index,
							content: thinking.clone(),
							partial: output.clone(),
						});
					}
					AnthropicBlock::ToolCall { partial_json, .. } => {
						let mut tool_call = match &blocks[index] {
							AnthropicBlock::ToolCall { tool_call, .. } => tool_call.clone(),
							_ => unreachable!(),
						};
						tool_call.arguments = match parse_streaming_json(Some(partial_json.as_str())) {
							Value::Object(object) => object,
							_ => Map::new(),
						};
						// Finalize in-place and strip the scratch buffer so replay only
						// carries parsed arguments.
						if let AnthropicBlock::ToolCall {
							tool_call: stored,
							partial_json: scratch,
							..
						} = &mut blocks[index]
						{
							*stored = tool_call.clone();
							scratch.clear();
						}
						sync_output_content(output, &blocks);
						out.push(AssistantMessageEvent::ToolCallEnd {
							content_index: index,
							tool_call,
							partial: output.clone(),
						});
					}
				}
			}
		} else if event_type == "message_delta" {
			if let Some(stop_reason) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
				output.stop_reason = map_stop_reason(stop_reason)?;
				if output.stop_reason == "error" {
					output.stop_reason_raw = Some(stop_reason.to_string());
				}
			}
			// Only update usage fields if present (not null).
			// Preserves input_tokens from message_start when proxies omit it in message_delta.
			if let Some(input_tokens) = event.pointer("/usage/input_tokens").filter(|value| !value.is_null()) {
				output.usage.input = number_or_zero(Some(input_tokens));
			}
			if let Some(output_tokens) = event.pointer("/usage/output_tokens").filter(|value| !value.is_null()) {
				output.usage.output = number_or_zero(Some(output_tokens));
			}
			if let Some(cache_read) = event
				.pointer("/usage/cache_read_input_tokens")
				.filter(|value| !value.is_null())
			{
				output.usage.cache_read = number_or_zero(Some(cache_read));
			}
			if let Some(cache_creation) = event
				.pointer("/usage/cache_creation_input_tokens")
				.filter(|value| !value.is_null())
			{
				output.usage.cache_write = number_or_zero(Some(cache_creation));
			}
			// The SDK's MessageDeltaUsage type omits cache_creation, but the wire carries it.
			let delta_cache_creation = parse_cache_creation(event.pointer("/usage/cache_creation"));
			if cache_control.is_some() && uses_anthropic_cache_pricing && delta_cache_creation.is_some() {
				cache_write_cost = Some(get_anthropic_cache_write_cost(
					model.cost.input,
					if cache_control.as_ref().and_then(|c| c.ttl.as_deref()) == Some("1h") {
						"1h"
					} else {
						"5m"
					},
					delta_cache_creation.as_ref(),
				));
			}
			output.usage.total_tokens =
				output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write;
			let overrides = cache_write_cost.map(|cache_write| CostOverrides {
				cache_write: Some(cache_write),
			});
			calculate_cost(model, &mut output.usage, overrides.as_ref());
		}
	}

	if options
		.stream
		.signal
		.as_ref()
		.map(|signal| signal.is_cancelled())
		.unwrap_or(false)
	{
		return Err(AnthropicStreamError::Message("Request was aborted".to_string()));
	}

	if output.stop_reason == "aborted" || output.stop_reason == "error" {
		return Err(AnthropicStreamError::Failure(stream_failure_from_stop_reason(
			output.stop_reason_raw.as_deref(),
			request_id.as_deref(),
		)));
	}

	out.push(AssistantMessageEvent::Done {
		reason: output.stop_reason.clone(),
		message: output.clone(),
	});
	out.end(None);
	Ok(())
}

/// TS: `Number(x) || 0` for the numeric usage/index fields.
fn number_or_zero(value: Option<&Value>) -> f64 {
	match value {
		Some(Value::Number(number)) => number.as_f64().unwrap_or(0.0),
		Some(Value::String(text)) => text.parse::<f64>().unwrap_or(0.0),
		_ => 0.0,
	}
}

/// TS: `usage.cache_creation` (`AnthropicCacheCreationUsage | null`).
fn parse_cache_creation(value: Option<&Value>) -> Option<AnthropicCacheCreationUsage> {
	let value = value?;
	if value.is_null() {
		return None;
	}
	Some(AnthropicCacheCreationUsage {
		ephemeral_5m_input_tokens: number_or_zero(value.get("ephemeral_5m_input_tokens")),
		ephemeral_1h_input_tokens: number_or_zero(value.get("ephemeral_1h_input_tokens")),
	})
}

/// TS: `isAlwaysOnAdaptiveThinkingModel(modelId)`.
///
/// Opus 5.5 and Fable/Mythos models think every turn and reject an explicit
/// `thinking: {type: "disabled"}` (and any sampling params) with a 400.
fn is_always_on_adaptive_thinking_model(model_id: &str) -> bool {
	model_id.contains("opus-5-5") || model_id.contains("opus-5.5")
		|| model_id.contains("fable-5") || model_id.contains("mythos-5") || model_id.contains("mythos-preview")
}

/// Check if a model supports adaptive thinking (Opus 4.6+, Sonnet 4.6).
fn supports_adaptive_thinking(model_id: &str) -> bool {
	// Adaptive-thinking model IDs (with or without date suffix).
	model_id.contains("opus-4-6")
		|| model_id.contains("opus-4.6")
		|| model_id.contains("opus-5")
		|| model_id.contains("opus-4-7")
		|| model_id.contains("opus-4.7")
		|| model_id.contains("opus-4-8")
		|| model_id.contains("opus-4.8")
		|| model_id.contains("sonnet-4-6")
		|| model_id.contains("sonnet-4.6")
		|| model_id.contains("sonnet-5")
		|| model_id.contains("fable-5")
		|| model_id.contains("mythos-5")
		|| model_id.contains("mythos-preview")
}

/// Map ThinkingLevel to Anthropic effort levels for adaptive thinking. The effort is
/// driven by each model's `thinkingLevelMap` (see generate-models.ts); the switch is a
/// fallback for levels without an explicit mapping.
fn map_thinking_level_to_effort(model: &Model, level: Option<&String>) -> AnthropicEffort {
	// Clamp to what the model actually supports so callers that bypass
	// clampThinkingLevel (e.g. passing reasoning: "xhigh" directly) can't send an
	// effort the model lacks - xhigh on a max-only model resolves to max, not xhigh.
	let effective = level.map(|level| clamp_thinking_level(model, level));
	if let Some(effective) = effective.as_ref() {
		if let Some(Some(mapped)) = model.thinking_level_map_get(effective) {
			return mapped;
		}
	}

	match effective.as_deref() {
		Some("minimal") | Some("low") => "low".to_string(),
		Some("medium") => "medium".to_string(),
		Some("high") => "high".to_string(),
		Some("xhigh") => "xhigh".to_string(),
		Some("max") => "max".to_string(),
		_ => "high".to_string(),
	}
}

/// Terminal `error` stream carrying a synchronous-configuration failure's thrown message.
///
/// anthropic.ts:818-821 throws out of `streamSimpleAnthropic`; a Rust `StreamFunction` returns a
/// stream, so the failure is reported the same way the other ports report it
/// (`streamSimpleOpenAICompletions`, openai-completions.ts:504) instead of aborting the process.
fn api_key_error_stream(model: &Model, message: &str) -> AssistantMessageEventStream {
	let stream = create_assistant_message_event_stream();
	let mut output = AssistantMessage {
		role: "assistant".to_string(),
		content: Vec::new(),
		api: model.api.clone(),
		provider: model.provider.clone(),
		model: model.id.clone(),
		usage: crate::types::Usage::zero(),
		stop_reason: "error".to_string(),
		timestamp: now_ms(),
		..Default::default()
	};
	output.error_message = Some(message.to_string());
	record_stream_failure(model, &mut output, &ThrownStreamError::Message(message));
	stream.push(AssistantMessageEvent::Error {
		reason: output.stop_reason.clone(),
		error: output,
	});
	stream.end(None);
	stream
}

/// TS: `streamSimpleAnthropic`.
pub fn stream_simple_anthropic(
	model: &Model,
	context: &Context,
	options: Option<SimpleStreamOptions>,
) -> AssistantMessageEventStream {
	let options = options.unwrap_or_default();
	let api_key = options
		.stream
		.api_key
		.clone()
		.or_else(|| get_env_api_key(&model.provider));
	let Some(api_key) = api_key else {
		// anthropic.ts:818-821 `if (!apiKey) { throw new Error(...) }`.
		return api_key_error_stream(model, &format!("No API key for provider: {}", model.provider));
	};

	let base = build_base_options(model, Some(&options), Some(&api_key));
	let reasoning = options.reasoning.clone();
	if reasoning.is_none() || reasoning.as_deref() == Some("off") {
		return stream_anthropic(
			model,
			context,
			Some(AnthropicOptions {
				thinking_enabled: Some(false),
				..AnthropicOptions::from_base(&base)
			}),
		);
	}

	// For Opus 4.6 and Sonnet 4.6: use adaptive thinking with effort level
	// For older models: use budget-based thinking
	if supports_adaptive_thinking(&model.id) {
		let effort = map_thinking_level_to_effort(model, reasoning.as_ref());
		return stream_anthropic(
			model,
			context,
			Some(AnthropicOptions {
				thinking_enabled: Some(true),
				effort: Some(effort),
				..AnthropicOptions::from_base(&base)
			}),
		);
	}

	let adjusted = adjust_max_tokens_for_thinking(
		base.max_tokens.unwrap_or(0.0),
		model.max_tokens,
		reasoning.as_ref().expect("reasoning present"),
		options.thinking_budgets.as_ref(),
	);

	let mut options = AnthropicOptions {
		thinking_enabled: Some(true),
		thinking_budget_tokens: Some(adjusted.thinking_budget),
		..AnthropicOptions::from_base(&base)
	};
	options.stream.max_tokens = Some(adjusted.max_tokens);
	stream_anthropic(model, context, Some(options))
}

/// TS: `isOAuthToken(apiKey)`.
fn is_oauth_token(api_key: &str) -> bool {
	api_key.contains("sk-ant-oat")
}

/// TS: `createClient(...)` result - the fields of the constructed SDK client that
/// the request path reads.
#[derive(Debug, Clone)]
struct CreatedClient {
	client: AnthropicClientOverride,
	is_oauth_token: bool,
}

/// TS: `createClient(model, apiKey, interleavedThinking, useFineGrainedToolStreamingBeta, optionsHeaders?, dynamicHeaders?, sessionId?)`.
fn create_client(
	model: &Model,
	api_key: &str,
	interleaved_thinking: bool,
	use_fine_grained_tool_streaming_beta: bool,
	options_headers: Option<&IndexMap<String, String>>,
	dynamic_headers: Option<&IndexMap<String, String>>,
	session_id: Option<&str>,
) -> Result<CreatedClient, AnthropicStreamError> {
	// Adaptive thinking models (Opus 4.6, Sonnet 4.6) have interleaved thinking built-in.
	// The beta header is deprecated on Opus 4.6 and redundant on Sonnet 4.6, so skip it.
	let needs_interleaved_beta = interleaved_thinking && !supports_adaptive_thinking(&model.id);
	let mut beta_features: Vec<String> = Vec::new();
	if use_fine_grained_tool_streaming_beta {
		beta_features.push(FINE_GRAINED_TOOL_STREAMING_BETA.to_string());
	}
	if needs_interleaved_beta {
		beta_features.push(INTERLEAVED_THINKING_BETA.to_string());
	}

	if model.provider == "cloudflare-ai-gateway" {
		let base_url = resolve_cloudflare_base_url(model).map_err(AnthropicStreamError::Message)?;
		let mut base: IndexMap<String, Option<String>> = IndexMap::new();
		base.insert("accept".to_string(), Some("application/json".to_string()));
		base.insert(
			"anthropic-dangerous-direct-browser-access".to_string(),
			Some("true".to_string()),
		);
		base.insert(
			"cf-aig-authorization".to_string(),
			Some(format!("Bearer {}", api_key)),
		);
		base.insert("x-api-key".to_string(), None);
		base.insert("Authorization".to_string(), None);
		if !beta_features.is_empty() {
			base.insert("anthropic-beta".to_string(), Some(beta_features.join(",")));
		}

		let headers = merge_headers(vec![
			Some(base),
			record_to_nullable(model.headers.as_ref()),
			record_to_nullable(options_headers),
		]);

		return Ok(CreatedClient {
			client: AnthropicClientOverride {
				base_url: Some(base_url),
				api_key: None,
				auth_token: None,
				headers: Some(nullable_to_record(&headers)),
			},
			is_oauth_token: false,
		});
	}

	if model.provider == "github-copilot" {
		let mut base: IndexMap<String, Option<String>> = IndexMap::new();
		base.insert("accept".to_string(), Some("application/json".to_string()));
		base.insert(
			"anthropic-dangerous-direct-browser-access".to_string(),
			Some("true".to_string()),
		);
		if !beta_features.is_empty() {
			base.insert("anthropic-beta".to_string(), Some(beta_features.join(",")));
		}

		let headers = merge_headers(vec![
			Some(base),
			record_to_nullable(model.headers.as_ref()),
			record_to_nullable(dynamic_headers),
			record_to_nullable(options_headers),
		]);

		return Ok(CreatedClient {
			client: AnthropicClientOverride {
				base_url: Some(model.base_url.clone()),
				api_key: None,
				auth_token: Some(api_key.to_string()),
				headers: Some(nullable_to_record(&headers)),
			},
			is_oauth_token: false,
		});
	}

	if is_oauth_token(api_key) {
		let mut base: IndexMap<String, Option<String>> = IndexMap::new();
		base.insert("accept".to_string(), Some("application/json".to_string()));
		base.insert(
			"anthropic-dangerous-direct-browser-access".to_string(),
			Some("true".to_string()),
		);
		let mut oauth_beta = vec![
			"claude-code-20250219".to_string(),
			"oauth-2025-04-20".to_string(),
		];
		oauth_beta.extend(beta_features.iter().cloned());
		base.insert("anthropic-beta".to_string(), Some(oauth_beta.join(",")));
		base.insert(
			"user-agent".to_string(),
			Some(format!("claude-cli/{}", CLAUDE_CODE_VERSION)),
		);
		base.insert("x-app".to_string(), Some("cli".to_string()));

		let headers = merge_headers(vec![
			Some(base),
			record_to_nullable(model.headers.as_ref()),
			record_to_nullable(options_headers),
		]);

		return Ok(CreatedClient {
			client: AnthropicClientOverride {
				base_url: Some(model.base_url.clone()),
				api_key: None,
				auth_token: Some(api_key.to_string()),
				headers: Some(nullable_to_record(&headers)),
			},
			is_oauth_token: true,
		});
	}

	let mut base: IndexMap<String, Option<String>> = IndexMap::new();
	base.insert("accept".to_string(), Some("application/json".to_string()));
	base.insert(
		"anthropic-dangerous-direct-browser-access".to_string(),
		Some("true".to_string()),
	);
	if !beta_features.is_empty() {
		base.insert("anthropic-beta".to_string(), Some(beta_features.join(",")));
	}

	let merged = merge_headers(vec![
		Some(base),
		record_to_nullable(model.headers.as_ref()),
		record_to_nullable(options_headers),
	]);
	let headers = with_opencode_headers(&model.provider, session_id, &merged);

	Ok(CreatedClient {
		client: AnthropicClientOverride {
			base_url: Some(model.base_url.clone()),
			api_key: Some(api_key.to_string()),
			auth_token: None,
			headers: Some(nullable_to_record(&headers)),
		},
		is_oauth_token: false,
	})
}

/// The TypeScript `defaultHeaders` keeps `null` values so the SDK can delete the
/// matching default header. The request builder consumes them here, so the stored
/// record only carries the values that are actually sent.
fn nullable_to_record(headers: &IndexMap<String, Option<String>>) -> IndexMap<String, String> {
	headers
		.iter()
		.filter_map(|(name, value)| value.as_ref().map(|value| (name.clone(), value.clone())))
		.collect()
}

/// TS: `buildParams(model, context, isOAuthToken, options?, cacheControl?)`.
fn build_params(
	model: &Model,
	context: &Context,
	is_oauth_token: bool,
	options: &AnthropicOptions,
	cache_control: Option<&CacheControlEphemeral>,
) -> Result<Value, AnthropicStreamError> {
	let mut params: Map<String, Value> = Map::new();
	params.insert("model".to_string(), Value::String(model.id.clone()));
	params.insert(
		"messages".to_string(),
		Value::Array(convert_messages(&context.messages, model, is_oauth_token, cache_control)?),
	);
	params.insert(
		"max_tokens".to_string(),
		serde_json::Number::from_f64(
			options
				.stream
				.max_tokens
				// JS `options?.maxTokens || ...`: 0 and NaN are falsy.
				.filter(|max_tokens| *max_tokens != 0.0 && !max_tokens.is_nan())
				.unwrap_or_else(|| {
					// JS `(model.maxTokens / 3) | 0` wraps only the fallback.
					let fallback = (model.max_tokens / 3.0).trunc();
					if fallback.is_finite() {
						(fallback.rem_euclid(4_294_967_296.0) as u32 as i32) as f64
					} else {
						0.0
					}
				}),
		)
		.map(Value::Number)
		.unwrap_or(Value::Null),
	);
	params.insert("stream".to_string(), Value::Bool(true));

	// For OAuth tokens, we MUST include Claude Code identity
	if is_oauth_token {
		let mut system: Vec<Value> = Vec::new();
		let mut identity: Map<String, Value> = Map::new();
		identity.insert("type".to_string(), Value::String("text".to_string()));
		identity.insert(
			"text".to_string(),
			Value::String("You are Claude Code, Anthropic's official CLI for Claude.".to_string()),
		);
		if let Some(cache_control) = cache_control {
			identity.insert("cache_control".to_string(), serde_json::to_value(cache_control).unwrap());
		}
		system.push(Value::Object(identity));
		if let Some(system_prompt) = &context.system_prompt {
			let mut block: Map<String, Value> = Map::new();
			block.insert("type".to_string(), Value::String("text".to_string()));
			block.insert(
				"text".to_string(),
				Value::String(sanitize_surrogates(system_prompt)),
			);
			if let Some(cache_control) = cache_control {
				block.insert("cache_control".to_string(), serde_json::to_value(cache_control).unwrap());
			}
			system.push(Value::Object(block));
		}
		params.insert("system".to_string(), Value::Array(system));
	} else if let Some(system_prompt) = &context.system_prompt {
		// Add cache control to system prompt for non-OAuth tokens
		let mut block: Map<String, Value> = Map::new();
		block.insert("type".to_string(), Value::String("text".to_string()));
		block.insert(
			"text".to_string(),
			Value::String(sanitize_surrogates(system_prompt)),
		);
		if let Some(cache_control) = cache_control {
			block.insert("cache_control".to_string(), serde_json::to_value(cache_control).unwrap());
		}
		params.insert("system".to_string(), Value::Array(vec![Value::Object(block)]));
	}

	// Temperature is incompatible with extended thinking (adaptive or budget-based),
	// and always-on models reject sampling params outright.
	if let Some(temperature) = options.stream.temperature {
		if options.thinking_enabled != Some(true) && !is_always_on_adaptive_thinking_model(&model.id) {
			params.insert(
				"temperature".to_string(),
				serde_json::Number::from_f64(temperature)
					.map(Value::Number)
					.unwrap_or(Value::Null),
			);
		}
	}

	if let Some(tools) = &context.tools {
		if !tools.is_empty() {
			params.insert(
				"tools".to_string(),
				Value::Array(convert_tools(
					tools,
					is_oauth_token,
					get_anthropic_compat(model).supports_eager_tool_input_streaming,
					cache_control,
				)),
			);
		}
	}

	// Configure thinking mode: adaptive (Opus 4.6+ and Sonnet 4.6),
	// budget-based (older models), or explicitly disabled.
	if model.reasoning {
		if options.thinking_enabled == Some(true) {
			// Default to "summarized" so Opus 4.7 and Mythos Preview behave like
			// older Claude 4 models (whose API default is also "summarized").
			let display: AnthropicThinkingDisplay = options
				.thinking_display
				.clone()
				.unwrap_or_else(|| "summarized".to_string());
			if supports_adaptive_thinking(&model.id) {
				// Adaptive thinking: Claude decides when and how much to think.
				let mut thinking: Map<String, Value> = Map::new();
				thinking.insert("type".to_string(), Value::String("adaptive".to_string()));
				thinking.insert("display".to_string(), Value::String(display));
				params.insert("thinking".to_string(), Value::Object(thinking));
				if let Some(effort) = options.effort.clone() {
					// The Anthropic SDK types can lag newly supported effort values such as "xhigh"/"max".
					let mut output_config: Map<String, Value> = Map::new();
					output_config.insert("effort".to_string(), Value::String(effort));
					params.insert("output_config".to_string(), Value::Object(output_config));
				}
			} else {
				let mut thinking: Map<String, Value> = Map::new();
				thinking.insert("type".to_string(), Value::String("enabled".to_string()));
				thinking.insert(
					"budget_tokens".to_string(),
					Value::Number(
						serde_json::Number::from_f64(
							options
								.thinking_budget_tokens
								// JS `options.thinkingBudgetTokens || 1024`.
								.filter(|budget| *budget != 0.0 && !budget.is_nan())
								.unwrap_or(1024.0),
						)
						.unwrap_or_else(|| serde_json::Number::from(0)),
					),
				);
				thinking.insert("display".to_string(), Value::String(display));
				params.insert("thinking".to_string(), Value::Object(thinking));
			}
		} else if options.thinking_enabled == Some(false) && !is_always_on_adaptive_thinking_model(&model.id) {
			let mut thinking: Map<String, Value> = Map::new();
			thinking.insert("type".to_string(), Value::String("disabled".to_string()));
			params.insert("thinking".to_string(), Value::Object(thinking));
		}
	}

	if let Some(metadata) = &options.stream.metadata {
		if let Some(user_id) = metadata.get("user_id") {
			if let Some(user_id) = user_id.as_str() {
				let mut metadata_object: Map<String, Value> = Map::new();
				metadata_object.insert("user_id".to_string(), Value::String(user_id.to_string()));
				params.insert("metadata".to_string(), Value::Object(metadata_object));
			}
		}
	}

	if let Some(tool_choice) = &options.tool_choice {
		if let Some(name) = tool_choice.as_str() {
			let mut choice: Map<String, Value> = Map::new();
			choice.insert("type".to_string(), Value::String(name.to_string()));
			params.insert("tool_choice".to_string(), Value::Object(choice));
		} else {
			params.insert("tool_choice".to_string(), tool_choice.clone());
		}
	}

	Ok(Value::Object(params))
}

/// Normalize tool call IDs to match Anthropic's required pattern and length.
///
/// TS: `id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64)`
fn normalize_tool_call_id(id: &str) -> String {
	let replaced: String = id
		.chars()
		.map(|ch| {
			if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
				ch
			} else {
				'_'
			}
		})
		.collect();
	replaced.chars().take(64).collect()
}

/// TS: `convertMessages(messages, model, isOAuthToken, cacheControl?)`.
fn convert_messages(
	messages: &[Message],
	model: &Model,
	is_oauth_token: bool,
	cache_control: Option<&CacheControlEphemeral>,
) -> Result<Vec<Value>, AnthropicStreamError> {
	let mut params: Vec<Value> = Vec::new();

	let normalize = |id: &str, _model: &Model, _assistant: &AssistantMessage| normalize_tool_call_id(id);
	let transformed_messages = try_transform_messages(
		messages.to_vec(),
		model,
		Some(&normalize as &dyn Fn(&str, &Model, &AssistantMessage) -> String),
	)
	.map_err(AnthropicStreamError::Message)?;

	let mut i = 0usize;
	while i < transformed_messages.len() {
		let msg = &transformed_messages[i];

		match msg {
			Message::User(user) => match &user.content {
				UserContent::Text(text) => {
					if !text.trim().is_empty() {
						let mut param: Map<String, Value> = Map::new();
						param.insert("role".to_string(), Value::String("user".to_string()));
						param.insert("content".to_string(), Value::String(sanitize_surrogates(text)));
						params.push(Value::Object(param));
					}
				}
				UserContent::Blocks(content) => {
					let blocks: Vec<Value> = content
						.iter()
						.map(|item| match item {
							ImageOrTextContent::Text(text) => {
								let mut block: Map<String, Value> = Map::new();
								block.insert("type".to_string(), Value::String("text".to_string()));
								block.insert("text".to_string(), Value::String(sanitize_surrogates(&text.text)));
								Value::Object(block)
							}
							ImageOrTextContent::Image(image) => {
								let mut source: Map<String, Value> = Map::new();
								source.insert("type".to_string(), Value::String("base64".to_string()));
								source.insert("media_type".to_string(), Value::String(image.mime_type.clone()));
								source.insert("data".to_string(), Value::String(image.data.clone()));
								let mut block: Map<String, Value> = Map::new();
								block.insert("type".to_string(), Value::String("image".to_string()));
								block.insert("source".to_string(), Value::Object(source));
								Value::Object(block)
							}
						})
						.collect();
					let filtered_blocks: Vec<Value> = blocks
						.into_iter()
						.filter(|block| {
							if block.get("type").and_then(Value::as_str) == Some("text") {
								return block
									.get("text")
									.and_then(Value::as_str)
									.map(|text| !text.trim().is_empty())
									.unwrap_or(false);
							}
							true
						})
						.collect();
					if filtered_blocks.is_empty() {
						i += 1;
						continue;
					}
					let mut param: Map<String, Value> = Map::new();
					param.insert("role".to_string(), Value::String("user".to_string()));
					param.insert("content".to_string(), Value::Array(filtered_blocks));
					params.push(Value::Object(param));
				}
			},
			Message::Assistant(assistant) => {
				let mut blocks: Vec<Value> = Vec::new();

				for block in &assistant.content {
					match block {
						ContentBlock::Text(text) => {
							if text.text.trim().is_empty() {
								continue;
							}
							let mut entry: Map<String, Value> = Map::new();
							entry.insert("type".to_string(), Value::String("text".to_string()));
							entry.insert("text".to_string(), Value::String(sanitize_surrogates(&text.text)));
							blocks.push(Value::Object(entry));
						}
						ContentBlock::Thinking(thinking) => {
							// Redacted thinking: pass the opaque payload back as redacted_thinking
							if thinking.redacted == Some(true) {
								let mut entry: Map<String, Value> = Map::new();
								entry.insert("type".to_string(), Value::String("redacted_thinking".to_string()));
								entry.insert(
									"data".to_string(),
									Value::String(thinking.thinking_signature.clone().unwrap_or_default()),
								);
								blocks.push(Value::Object(entry));
								continue;
							}
							if thinking.thinking.trim().is_empty() {
								continue;
							}
							// If thinking signature is missing/empty (e.g., from aborted stream),
							// convert to plain text block without <thinking> tags to avoid API rejection
							// and prevent Claude from mimicking the tags in responses
							let signature = thinking.thinking_signature.clone().unwrap_or_default();
							if signature.trim().is_empty() {
								let mut entry: Map<String, Value> = Map::new();
								entry.insert("type".to_string(), Value::String("text".to_string()));
								entry.insert(
									"text".to_string(),
									Value::String(sanitize_surrogates(&thinking.thinking)),
								);
								blocks.push(Value::Object(entry));
							} else {
								let mut entry: Map<String, Value> = Map::new();
								entry.insert("type".to_string(), Value::String("thinking".to_string()));
								entry.insert(
									"thinking".to_string(),
									Value::String(sanitize_surrogates(&thinking.thinking)),
								);
								entry.insert("signature".to_string(), Value::String(signature));
								blocks.push(Value::Object(entry));
							}
						}
						ContentBlock::ToolCall(tool_call) => {
							let mut entry: Map<String, Value> = Map::new();
							entry.insert("type".to_string(), Value::String("tool_use".to_string()));
							entry.insert("id".to_string(), Value::String(tool_call.id.clone()));
							entry.insert(
								"name".to_string(),
								Value::String(if is_oauth_token {
									to_claude_code_name(&tool_call.name)
								} else {
									tool_call.name.clone()
								}),
							);
							entry.insert("input".to_string(), Value::Object(tool_call.arguments.clone()));
							blocks.push(Value::Object(entry));
						}
					}
				}
				if blocks.is_empty() {
					i += 1;
					continue;
				}
				let mut param: Map<String, Value> = Map::new();
				param.insert("role".to_string(), Value::String("assistant".to_string()));
				param.insert("content".to_string(), Value::Array(blocks));
				params.push(Value::Object(param));
			}
			Message::ToolResult(result) => {
				// Collect all consecutive toolResult messages, needed for z.ai Anthropic endpoint
				let mut tool_results: Vec<Value> = Vec::new();

				tool_results.push(tool_result_param(result));

				let mut j = i + 1;
				while j < transformed_messages.len() {
					match &transformed_messages[j] {
						Message::ToolResult(next_msg) => {
							tool_results.push(tool_result_param(next_msg));
							j += 1;
						}
						_ => break,
					}
				}

				i = j - 1;

				let mut param: Map<String, Value> = Map::new();
				param.insert("role".to_string(), Value::String("user".to_string()));
				param.insert("content".to_string(), Value::Array(tool_results));
				params.push(Value::Object(param));
			}
		}

		i += 1;
	}

	// Add cache_control to the last user message to cache conversation history
	if let Some(cache_control) = cache_control {
		if !params.is_empty() {
			let last_index = params.len() - 1;
			let last_message = &mut params[last_index];
			if last_message.get("role").and_then(Value::as_str) == Some("user") {
				let content = last_message.get_mut("content").expect("user content");
				if let Value::Array(blocks) = content {
					if let Some(last_block) = blocks.last_mut() {
						let block_type = last_block.get("type").and_then(Value::as_str);
						if matches!(block_type, Some("text") | Some("image") | Some("tool_result")) {
							if let Value::Object(object) = last_block {
								object.insert(
									"cache_control".to_string(),
									serde_json::to_value(cache_control).unwrap(),
								);
							}
						}
					}
				} else if let Value::String(text) = content {
					let mut block: Map<String, Value> = Map::new();
					block.insert("type".to_string(), Value::String("text".to_string()));
					block.insert("text".to_string(), Value::String(text.clone()));
					block.insert("cache_control".to_string(), serde_json::to_value(cache_control).unwrap());
					*content = Value::Array(vec![Value::Object(block)]);
				}
			}
		}
	}

	Ok(params)
}

/// TS: the `tool_result` content block built for one tool result message.
///
/// `cache_control` is only ever applied to the last user message afterwards, so
/// this helper does not take it.
fn tool_result_param(result: &ToolResultMessage) -> Value {
	let mut entry: Map<String, Value> = Map::new();
	entry.insert("type".to_string(), Value::String("tool_result".to_string()));
	entry.insert("tool_use_id".to_string(), Value::String(result.tool_call_id.clone()));
	entry.insert("content".to_string(), convert_content_blocks(&result.content));
	entry.insert("is_error".to_string(), Value::Bool(result.is_error));
	Value::Object(entry)
}

/// TS: `shouldUseFineGrainedToolStreamingBeta(model, context)`.
fn should_use_fine_grained_tool_streaming_beta(model: &Model, context: &Context) -> bool {
	context.tools.as_ref().map(|tools| !tools.is_empty()).unwrap_or(false)
		&& !get_anthropic_compat(model).supports_eager_tool_input_streaming
}

/// TS: `convertTools(tools, isOAuthToken, supportsEagerToolInputStreaming, cacheControl?)`.
fn convert_tools(
	tools: &[Tool],
	is_oauth_token: bool,
	supports_eager_tool_input_streaming: bool,
	cache_control: Option<&CacheControlEphemeral>,
) -> Vec<Value> {
	let mut converted: Vec<Value> = Vec::with_capacity(tools.len());
	for (index, tool) in tools.iter().enumerate() {
		let schema = &tool.parameters;

		let mut entry: Map<String, Value> = Map::new();
		entry.insert(
			"name".to_string(),
			Value::String(if is_oauth_token {
				to_claude_code_name(&tool.name)
			} else {
				tool.name.clone()
			}),
		);
		entry.insert("description".to_string(), Value::String(tool.description.clone()));
		if supports_eager_tool_input_streaming {
			entry.insert("eager_input_streaming".to_string(), Value::Bool(true));
		}
		let mut input_schema: Map<String, Value> = Map::new();
		input_schema.insert("type".to_string(), Value::String("object".to_string()));
		input_schema.insert(
			"properties".to_string(),
			schema
				.get("properties")
				.cloned()
				.unwrap_or_else(|| Value::Object(Map::new())),
		);
		input_schema.insert(
			"required".to_string(),
			schema
				.get("required")
				.cloned()
				.unwrap_or_else(|| Value::Array(Vec::new())),
		);
		entry.insert("input_schema".to_string(), Value::Object(input_schema));
		if cache_control.is_some() && index == tools.len() - 1 {
			entry.insert(
				"cache_control".to_string(),
				serde_json::to_value(cache_control).unwrap(),
			);
		}
		converted.push(Value::Object(entry));
	}
	converted
}

/// TS: `mapStopReason(reason)`.
fn map_stop_reason(reason: &str) -> Result<StopReason, AnthropicStreamError> {
	match reason {
		"end_turn" => Ok("stop".to_string()),
		"max_tokens" => Ok("length".to_string()),
		"tool_use" => Ok("toolUse".to_string()),
		"refusal" => Ok("error".to_string()),
		// Stop is good enough -> resubmit
		"pause_turn" => Ok("stop".to_string()),
		// We don't supply stop sequences, so this should never happen
		"stop_sequence" => Ok("stop".to_string()),
		// Content flagged by safety filters (not yet in SDK types)
		"sensitive" => Ok("error".to_string()),
		// Handle unknown stop reasons gracefully (API may add new values)
		other => Err(AnthropicStreamError::Message(format!(
			"Unhandled stop reason: {}",
			other
		))),
	}
}

/// TS: `X-Stainless-OS` (`normalizePlatform(process.platform)`).
fn stainless_os() -> String {
	match std::env::consts::OS {
		"windows" => "Windows".to_string(),
		"macos" => "MacOS".to_string(),
		"linux" => "Linux".to_string(),
		"freebsd" => "FreeBSD".to_string(),
		"openbsd" => "OpenBSD".to_string(),
		"android" => "Android".to_string(),
		other => format!("Other:{}", other),
	}
}

/// TS: `X-Stainless-Arch` (`normalizeArch(process.arch)`).
fn stainless_arch() -> String {
	match std::env::consts::ARCH {
		"x86" => "x32".to_string(),
		"x86_64" => "x64".to_string(),
		"arm" => "arm".to_string(),
		"aarch64" => "arm64".to_string(),
		other => format!("other:{}", other),
	}
}

/// Ordered, case-insensitive header bag mirroring the WHATWG `Headers` object the
/// Stainless client builds.
#[derive(Debug, Clone, Default)]
struct HeaderBag {
	entries: Vec<(String, String)>,
}

impl HeaderBag {
	fn set(&mut self, name: &str, value: &str) {
		if let Some(entry) = self
			.entries
			.iter_mut()
			.find(|(existing, _)| existing.eq_ignore_ascii_case(name))
		{
			entry.1 = value.to_string();
			return;
		}
		self.entries.push((name.to_string(), value.to_string()));
	}

	fn get(&self, name: &str) -> Option<&str> {
		self.entries
			.iter()
			.find(|(existing, _)| existing.eq_ignore_ascii_case(name))
			.map(|(_, value)| value.as_str())
	}

	fn remove(&mut self, name: &str) {
		self.entries
			.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
	}

	/// One `buildHeaders` source: object keys always overwrite older headers and
	/// never append; `None` clears the header.
	fn merge_source(&mut self, source: &IndexMap<String, Option<String>>) {
		let mut seen: HashSet<String> = HashSet::new();
		for (name, value) in source {
			let lower = name.to_lowercase();
			if !seen.contains(&lower) {
				seen.insert(lower.clone());
				self.remove(name);
			}
			match value {
				Some(value) => self.set(name, value),
				None => self.remove(name),
			}
		}
	}

	fn apply(&self, mut request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
		for (name, value) in &self.entries {
			request = request.header(name.as_str(), value.as_str());
		}
		request
	}
}

/// TS: `BaseAnthropic.buildHeaders(...)` order for `POST /v1/messages`.
fn build_request_headers(
	client: &AnthropicClientOverride,
	timeout_ms: f64,
	has_body: bool,
) -> HeaderBag {
	let mut headers = HeaderBag::default();

	// idempotencyHeaders: the Anthropic client does not configure an idempotency header.
	// Stainless defaults.
	headers.set("Accept", "application/json");
	headers.set("User-Agent", ANTHROPIC_USER_AGENT);
	headers.set("X-Stainless-Retry-Count", "0");
	headers.set(
		"X-Stainless-Timeout",
		&format!("{}", (timeout_ms / 1000.0).trunc() as i64),
	);
	headers.set("X-Stainless-Lang", "js");
	headers.set("X-Stainless-Package-Version", ANTHROPIC_SDK_VERSION);
	headers.set("X-Stainless-OS", &stainless_os());
	headers.set("X-Stainless-Arch", &stainless_arch());
	headers.set("X-Stainless-Runtime", "node");
	// NOTE: the TypeScript sends `process.version` here; the port has no Node runtime.
	headers.set("X-Stainless-Runtime-Version", "unknown");
	// `dangerouslyAllowBrowser: true` in every createClient() branch.
	headers.set("anthropic-dangerous-direct-browser-access", "true");
	headers.set("anthropic-version", "2023-06-01");

	// `await this.authHeaders(options)` -> apiKeyAuth then bearerAuth.
	if let Some(api_key) = &client.api_key {
		headers.set("x-api-key", api_key);
	}
	if let Some(auth_token) = &client.auth_token {
		headers.set("Authorization", &format!("Bearer {}", auth_token));
	}

	// `this._options.defaultHeaders` (values only; `null` entries cleared earlier).
	if let Some(default_headers) = &client.headers {
		let mut source: IndexMap<String, Option<String>> = IndexMap::new();
		for (name, value) in default_headers {
			source.insert(name.clone(), Some(value.clone()));
		}
		headers.merge_source(&source);
	}

	if has_body {
		headers.set("content-type", "application/json");
	}

	headers
}

/// TS: `BaseAnthropic.buildURL('/v1/messages')`.
fn build_messages_url(base_url: &str) -> String {
	let path = "/v1/messages";
	if base_url.ends_with('/') && path.starts_with('/') {
		format!("{}{}", base_url, &path[1..])
	} else {
		format!("{}{}", base_url, path)
	}
}

/// TS: `Number.isFinite(n) && n >= 0`.
fn finite_non_negative(value: f64) -> Option<f64> {
	if value.is_finite() && value >= 0.0 {
		Some(value)
	} else {
		None
	}
}

/// TS: `parseRetryAfterMs(headers)` (stream-failure.ts:202-211).
fn parse_retry_after_ms(headers: &reqwest::header::HeaderMap) -> Option<f64> {
	let header_value = |name: &str| -> Option<String> {
		headers
			.get(name)
			.and_then(|value| value.to_str().ok())
			.map(str::to_string)
	};
	if let Some(raw) = header_value("retry-after-ms") {
		if let Ok(ms) = raw.trim().parse::<f64>() {
			if let Some(ms) = finite_non_negative(ms) {
				return Some(ms);
			}
		}
	}
	let raw = header_value("retry-after")?;
	if let Ok(seconds) = raw.trim().parse::<f64>() {
		if let Some(seconds) = finite_non_negative(seconds) {
			return Some(seconds * 1000.0);
		}
	}
	// stream-failure.ts:209-210 `const date = Date.parse(raw); return
	// Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());`
	let date = crate::utils::stream_failure::parse_http_date_ms(&raw)?;
	Some((date - now_ms() as f64).max(0.0))
}

/// TS: `APIError.makeMessage(status, error, message)`.
///
/// Shared with the other OpenAI-shaped ports (openai-completions.ts uses the same SDK
/// `APIError.makeMessage`, so openai_completions.rs reuses this helper).
pub(crate) fn api_error_message(status: u16, body: Option<&Value>, raw_text: &str) -> String {
	let message = match body {
		Some(Value::Object(object)) => match object.get("message") {
			Some(Value::String(text)) => text.clone(),
			Some(other) => other.to_string(),
			None => body.map(|body| body.to_string()).unwrap_or_default(),
		},
		Some(other) => other.to_string(),
		None => raw_text.to_string(),
	};
	let message = if message.is_empty() && body.is_none() {
		raw_text.to_string()
	} else {
		message
	};
	if status != 0 && !message.is_empty() {
		return format!("{} {}", status, message);
	}
	if status != 0 {
		return format!("{} status code (no body)", status);
	}
	if !message.is_empty() {
		return message;
	}
	"(no status code or body)".to_string()
}

/// TS: the SDK error class name chosen by `APIError.generate(status, ...)`.
fn api_error_name(status: u16) -> String {
	match status {
		400 => "BadRequestError".to_string(),
		401 => "AuthenticationError".to_string(),
		403 => "PermissionDeniedError".to_string(),
		404 => "NotFoundError".to_string(),
		409 => "ConflictError".to_string(),
		422 => "UnprocessableEntityError".to_string(),
		429 => "RateLimitError".to_string(),
		status if status >= 500 => "InternalServerError".to_string(),
		_ => "APIError".to_string(),
	}
}

/// TS: `extractStreamFailureParts(error)` for an Anthropic SDK `APIError`.
///
/// The TypeScript reads `err.status`, `err.error`, `err.name`, `err.requestID`,
/// `err.headers`; the Rust port rebuilds the same parts from the HTTP response.
fn anthropic_api_error(
	status: u16,
	body: Option<&Value>,
	raw_text: &str,
	headers: &reqwest::header::HeaderMap,
) -> AnthropicStreamError {
	let message_text = api_error_message(status, body, raw_text);

	let mut body_object = body.and_then(Value::as_object);
	if let Some(inner) = body_object.and_then(|object| object.get("error")).and_then(Value::as_object) {
		body_object = Some(inner);
	}
	let body_type = body_object
		.and_then(|object| object.get("type").or_else(|| object.get("code")))
		.and_then(Value::as_str);
	let body_message = body_object.and_then(|object| object.get("message")).and_then(Value::as_str);
	let provider_error_type = body_type.map(str::to_string).unwrap_or_else(|| api_error_name(status));

	let request_id = headers
		.get("request-id")
		.and_then(|value| value.to_str().ok())
		.map(str::to_string);
	let retry_after_ms = parse_retry_after_ms(headers);

	let status_i64 = status as i64;
	let mut kind = classify_stream_failure(Some(&provider_error_type), Some(status_i64)).to_string();
	// Message text is too weak for these verdicts: without a structured type, only
	// the status decides.
	if (kind == "auth" || kind == "permission") && body_type.is_none() {
		kind = classify_stream_failure(None, Some(status_i64)).to_string();
	}

	let info = StreamFailureInfo {
		kind,
		provider_error_type: Some(provider_error_type),
		status: Some(status_i64),
		request_id,
		retry_after_ms,
		raw: None,
	};

	if info.kind == "unknown" {
		// `formatStreamFailureMessage` passes unrecognized errors through verbatim.
		return AnthropicStreamError::Message(message_text);
	}

	AnthropicStreamError::Failure(StreamFailureError::new(
		stream_failure_message(&info, body_message),
		info,
	))
}

/// TS: the transport failure path (`APIConnectionError` / `APIConnectionTimeoutError`).
fn anthropic_connection_error(name: &str, message: &str) -> AnthropicStreamError {
	AnthropicStreamError::Failure(StreamFailureError::new(
		message,
		StreamFailureInfo {
			kind: "unknown".to_string(),
			provider_error_type: Some(name.to_string()),
			..Default::default()
		},
	))
}

/// TS: `client.messages.create({ ...params, stream: true }, requestOptions).asResponse()`.
async fn send_messages_request(
	client: &AnthropicClientOverride,
	params: &Value,
	model: &Model,
	options: &AnthropicOptions,
) -> Result<reqwest::Response, AnthropicStreamError> {
	if options
		.stream
		.signal
		.as_ref()
		.map(|signal| signal.is_cancelled())
		.unwrap_or(false)
	{
		return Err(anthropic_connection_error("APIUserAbortError", "Request was aborted."));
	}

	let timeout_ms = options.stream.timeout_ms.unwrap_or(600_000.0);
	let base_url = client.base_url.clone().unwrap_or_else(|| model.base_url.clone());
	let url = build_messages_url(&base_url);
	let headers = build_request_headers(client, timeout_ms, true);
	let body = serde_json::to_string(params).map_err(|error| AnthropicStreamError::Message(error.to_string()))?;

	let request = reqwest::Client::new()
		.post(&url)
		.body(body)
		.timeout(std::time::Duration::from_millis(timeout_ms as u64));
	let request = headers.apply(request);

	let response = match request.send().await {
		Ok(response) => response,
		Err(error) => {
			if options
				.stream
				.signal
				.as_ref()
				.map(|signal| signal.is_cancelled())
				.unwrap_or(false)
			{
				return Err(anthropic_connection_error("APIUserAbortError", "Request was aborted."));
			}
			return Err(if error.is_timeout() {
				anthropic_connection_error("APIConnectionTimeoutError", "Request timed out.")
			} else {
				anthropic_connection_error("APIConnectionError", "Connection error.")
			});
		}
	};

	if !response.status().is_success() {
		let status = response.status().as_u16();
		let headers = response.headers().clone();
		let raw_text = response.text().await.unwrap_or_default();
		let parsed = parse_json_with_repair(&raw_text).ok();
		return Err(anthropic_api_error(status, parsed.as_ref(), &raw_text, &headers));
	}

	Ok(response)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::{ImageContent, InputModality, Message, ModelCost, ThinkingLevelMap, Tool, UserMessage};
	use bytes::Bytes;

	fn test_model(provider: &str, id: &str) -> Model {
		let mut model = Model::default();
		model.id = id.to_string();
		model.name = id.to_string();
		model.api = "anthropic-messages".to_string();
		model.provider = provider.to_string();
		model.base_url = "https://api.anthropic.com".to_string();
		model.reasoning = true;
		model.max_tokens = 8192.0;
		model.cost = ModelCost {
			input: 3.0,
			output: 15.0,
			cache_read: 0.3,
			cache_write: 3.75,
		};
		model.input = vec![InputModality::Text, InputModality::Image];
		model
	}

	fn context_with_user(text: &str) -> Context {
		Context::new(
			None,
			vec![Message::user(UserMessage::new(
				UserContent::Text(text.to_string()),
				1,
			))],
			None,
		)
	}

	fn sse_stream(body: &str) -> ByteStream {
		Box::pin(futures::stream::iter(vec![Ok(Bytes::from(body.to_string()))]))
	}

	fn block_on<F: std::future::Future>(future: F) -> F::Output {
		tokio::runtime::Builder::new_current_thread()
			.enable_all()
			.build()
			.expect("runtime")
			.block_on(future)
	}

	fn collect_events(body: &str) -> Vec<Value> {
		block_on(async {
			let reader = SseMessageReader::new(sse_stream(body), None);
			let mut iterator = AnthropicEventIterator::new(reader, None);
			let mut events = Vec::new();
			loop {
				match iterator.next().await {
					Ok(Some(event)) => events.push(event),
					Ok(None) => break,
					Err(error) => panic!("unexpected stream error: {}", error.message()),
				}
			}
			events
		})
	}

	#[test]
	fn cache_retention_defaults_to_short_and_reads_the_env_override() {
		let mut env = crate::test_env::ScopedEnv::new();
		assert_eq!(resolve_cache_retention(None), "short");
		assert_eq!(resolve_cache_retention(Some(&"long".to_string())), "long");

		env.set("PI_CACHE_RETENTION", "long");
		assert_eq!(resolve_cache_retention(None), "long");
		env.remove("PI_CACHE_RETENTION");
		assert_eq!(resolve_cache_retention(None), "short");
	}

	#[test]
	fn cache_control_follows_retention_and_long_cache_compat() {
		let model = test_model("anthropic", "claude-sonnet-4-5");

		let none = get_cache_control(&model, Some(&"none".to_string()));
		assert_eq!(none.retention, "none");
		assert!(none.cache_control.is_none());

		let short = get_cache_control(&model, Some(&"short".to_string()));
		let control = short.cache_control.expect("cache control");
		assert_eq!(control.type_, "ephemeral");
		assert_eq!(control.ttl, None);

		let long = get_cache_control(&model, Some(&"long".to_string()));
		assert_eq!(long.cache_control.unwrap().ttl, Some("1h".to_string()));

		let mut disabled = model.clone();
		disabled.compat = Some(crate::types::Compat::Anthropic(crate::types::AnthropicMessagesCompat {
			supports_long_cache_retention: Some(false),
			..Default::default()
		}));
		let long = get_cache_control(&disabled, Some(&"long".to_string()));
		assert_eq!(long.cache_control.unwrap().ttl, None);
	}

	#[test]
	fn anthropic_compat_defaults_to_true() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let compat = get_anthropic_compat(&model);
		assert!(compat.supports_eager_tool_input_streaming);
		assert!(compat.supports_long_cache_retention);

		let mut configured = model.clone();
		configured.compat = Some(crate::types::Compat::Anthropic(crate::types::AnthropicMessagesCompat {
			supports_eager_tool_input_streaming: Some(false),
			supports_long_cache_retention: Some(false),
		}));
		let compat = get_anthropic_compat(&configured);
		assert!(!compat.supports_eager_tool_input_streaming);
		assert!(!compat.supports_long_cache_retention);
	}

	#[test]
	fn claude_code_tool_names_round_trip() {
		assert_eq!(to_claude_code_name("read"), "Read");
		assert_eq!(to_claude_code_name("WEBSEARCH"), "WebSearch");
		assert_eq!(to_claude_code_name("custom"), "custom");

		let tools = vec![Tool {
			name: "read_file".to_string(),
			description: "Read".to_string(),
			parameters: Value::Object(Map::new()),
		}];
		assert_eq!(from_claude_code_name("Read_File", Some(&tools)), "read_file");
		assert_eq!(from_claude_code_name("Read", Some(&tools)), "Read");
		assert_eq!(from_claude_code_name("Read", None), "Read");
	}

	#[test]
	fn convert_content_blocks_joins_text_without_images() {
		let blocks = vec![
			ImageOrTextContent::Text(TextContent::new("a")),
			ImageOrTextContent::Text(TextContent::new("b")),
		];
		assert_eq!(convert_content_blocks(&blocks), Value::String("a\nb".to_string()));
	}

	#[test]
	fn convert_content_blocks_keeps_images_and_prepends_placeholder() {
		let blocks = vec![ImageOrTextContent::Image(ImageContent::new("data", "image/png"))];
		let converted = convert_content_blocks(&blocks);
		let array = converted.as_array().expect("array");
		assert_eq!(array.len(), 2);
		assert_eq!(array[0]["type"], Value::String("text".to_string()));
		assert_eq!(array[0]["text"], Value::String("(see attached image)".to_string()));
		assert_eq!(array[1]["type"], Value::String("image".to_string()));
		assert_eq!(array[1]["source"]["type"], Value::String("base64".to_string()));
		assert_eq!(array[1]["source"]["media_type"], Value::String("image/png".to_string()));
	}

	#[test]
	fn sse_lines_decode_event_data_and_comments() {
		let mut state = SseDecoderState::default();
		assert!(decode_sse_line("event: message_start", &mut state).is_none());
		assert!(decode_sse_line("data: {\"a\":1}", &mut state).is_none());
		assert!(decode_sse_line(": keep-alive", &mut state).is_none());
		let event = decode_sse_line("", &mut state).expect("flushed event");
		assert_eq!(event.event, Some("message_start".to_string()));
		assert_eq!(event.data, "{\"a\":1}");
		assert_eq!(
			event.raw,
			vec![
				"event: message_start".to_string(),
				"data: {\"a\":1}".to_string(),
				": keep-alive".to_string()
			]
		);
	}

	#[test]
	fn sse_data_lines_join_with_newlines_and_blank_state_is_ignored() {
		let mut state = SseDecoderState::default();
		assert!(flush_sse_event(&mut state).is_none());
		decode_sse_line("data: one", &mut state);
		decode_sse_line("data: two", &mut state);
		let event = flush_sse_event(&mut state).expect("event");
		assert_eq!(event.data, "one\ntwo");
		assert_eq!(event.event, None);
	}

	#[test]
	fn consume_line_handles_lf_crlf_and_cr() {
		assert_eq!(consume_line("a\nb"), Some(("a".to_string(), "b".to_string())));
		assert_eq!(consume_line("a\r\nb"), Some(("a".to_string(), "b".to_string())));
		assert_eq!(consume_line("a\rb"), Some(("a".to_string(), "b".to_string())));
		assert_eq!(consume_line("no break"), None);
	}

	#[test]
	fn utf8_stream_decoding_keeps_split_sequences() {
		let bytes = "héllo".as_bytes().to_vec();
		let mut pending = bytes[..2].to_vec();
		assert_eq!(decode_utf8_stream(&mut pending), "h");
		assert_eq!(pending, vec![0xc3]);
		pending.extend_from_slice(&bytes[2..]);
		assert_eq!(decode_utf8_stream(&mut pending), "éllo");
		assert!(pending.is_empty());

		let mut invalid = vec![0xff];
		let decoded = decode_utf8_stream(&mut invalid);
		assert_eq!(decoded, "\u{FFFD}");
		assert!(invalid.is_empty());
	}

	#[test]
	fn iterator_parses_message_events_and_skips_unknown_ones() {
		let body = concat!(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
			"event: ping\ndata: {\"type\":\"ping\"}\n\n",
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
		);
		let events = collect_events(body);
		assert_eq!(events.len(), 3);
		assert_eq!(events[0]["type"], Value::String("message_start".to_string()));
		assert_eq!(events[1]["type"], Value::String("content_block_start".to_string()));
		assert_eq!(events[2]["type"], Value::String("message_stop".to_string()));
	}

	#[test]
	fn iterator_reports_a_stream_that_ends_before_message_stop() {
		let body = "event: message_start\ndata: {\"type\":\"message_start\"}\n\n";
		let error = block_on(async {
			let reader = SseMessageReader::new(sse_stream(body), None);
			let mut iterator = AnthropicEventIterator::new(reader, None);
			assert!(iterator.next().await.unwrap().is_some());
			iterator.next().await.unwrap_err()
		});
		match error {
			AnthropicStreamError::Failure(failure) => {
				assert_eq!(failure.message, "Anthropic stream ended before message_stop");
				assert_eq!(failure.info.kind, "malformed_response");
			}
			other => panic!("unexpected error {:?}", other),
		}
	}

	#[test]
	fn iterator_reports_malformed_sse_json() {
		let body = "event: message_start\ndata: {not json\n\n";
		let error = block_on(async {
			let reader = SseMessageReader::new(sse_stream(body), None);
			let mut iterator = AnthropicEventIterator::new(reader, None);
			iterator.next().await.unwrap_err()
		});
		match error {
			AnthropicStreamError::Failure(failure) => {
				assert!(failure.message.starts_with(
					"Could not parse Anthropic SSE event message_start: "
				));
				assert!(failure.message.contains("; data={not json; raw=event: message_start\\ndata: {not json"));
				assert_eq!(failure.info.kind, "malformed_response");
			}
			other => panic!("unexpected error {:?}", other),
		}
	}

	#[test]
	fn in_stream_error_events_are_classified() {
		let failure = anthropic_sse_error(
			"{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}",
			Some("req_abc"),
		);
		assert_eq!(failure.message, "Provider overloaded (overloaded_error): Overloaded [request_id: req_abc]");
		assert_eq!(failure.info.kind, "overloaded");
		assert_eq!(failure.info.request_id, Some("req_abc".to_string()));
		assert_eq!(failure.info.raw.as_deref(), Some("{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}"));

		let failure = anthropic_sse_error("not json", None);
		assert_eq!(failure.message, "Provider stream failed: not json");
		assert_eq!(failure.info.kind, "unknown");

		let failure = anthropic_sse_error("{\"request_id\":\"req_body\"}", None);
		assert_eq!(failure.info.request_id, Some("req_body".to_string()));
	}

	#[test]
	fn stop_reasons_map_like_typescript() {
		assert_eq!(map_stop_reason("end_turn").unwrap(), "stop");
		assert_eq!(map_stop_reason("max_tokens").unwrap(), "length");
		assert_eq!(map_stop_reason("tool_use").unwrap(), "toolUse");
		assert_eq!(map_stop_reason("refusal").unwrap(), "error");
		assert_eq!(map_stop_reason("pause_turn").unwrap(), "stop");
		assert_eq!(map_stop_reason("stop_sequence").unwrap(), "stop");
		assert_eq!(map_stop_reason("sensitive").unwrap(), "error");
		assert_eq!(
			map_stop_reason("weird").unwrap_err().message(),
			"Unhandled stop reason: weird"
		);
	}

	#[test]
	fn normalize_tool_call_id_replaces_disallowed_characters_and_truncates() {
		assert_eq!(normalize_tool_call_id("call|with/slash"), "call_with_slash");
		let long = "a".repeat(80);
		assert_eq!(normalize_tool_call_id(&long).len(), 64);
	}

	#[test]
	fn adaptive_thinking_detection_matches_model_ids() {
		assert!(supports_adaptive_thinking("claude-opus-4-6-20260101"));
		assert!(supports_adaptive_thinking("claude-opus-4.7"));
		assert!(supports_adaptive_thinking("claude-sonnet-4-6"));
		assert!(supports_adaptive_thinking("claude-fable-5"));
		assert!(supports_adaptive_thinking("claude-mythos-preview"));
		assert!(!supports_adaptive_thinking("claude-sonnet-4-5"));

		assert!(is_always_on_adaptive_thinking_model("claude-fable-5"));
		assert!(is_always_on_adaptive_thinking_model("claude-mythos-5"));
		assert!(is_always_on_adaptive_thinking_model("claude-mythos-preview"));
		assert!(!is_always_on_adaptive_thinking_model("claude-opus-4-6"));
	}

	#[test]
	fn thinking_level_effort_uses_the_map_then_the_fallback_switch() {
		let mut mapped = test_model("anthropic", "claude-opus-4-6");
		let mut map = ThinkingLevelMap::new();
		map.insert("high".to_string(), Some("max".to_string()));
		map.insert("max".to_string(), None);
		mapped.thinking_level_map = Some(map);
		assert_eq!(map_thinking_level_to_effort(&mapped, Some(&"high".to_string())), "max");
		// `max` is unsupported -> clampThinkingLevel walks down to `high`, whose map
		// entry is "max".
		assert_eq!(map_thinking_level_to_effort(&mapped, Some(&"max".to_string())), "max");

		let plain = test_model("anthropic", "claude-sonnet-4-5");
		assert_eq!(map_thinking_level_to_effort(&plain, Some(&"minimal".to_string())), "low");
		assert_eq!(map_thinking_level_to_effort(&plain, Some(&"medium".to_string())), "medium");
		assert_eq!(map_thinking_level_to_effort(&plain, Some(&"xhigh".to_string())), "high");
		assert_eq!(map_thinking_level_to_effort(&plain, None), "high");
	}

	#[test]
	fn build_params_sets_defaults_and_stream_flag() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let context = context_with_user("Say hello.");
		let options = AnthropicOptions::default();
		let params = build_params(&model, &context, false, &options, None).unwrap();

		assert_eq!(params["model"], Value::String("claude-sonnet-4-5".to_string()));
		assert_eq!(params["stream"], Value::Bool(true));
		assert_eq!(params["max_tokens"].as_f64(), Some(2730.0));
		assert!(params.get("system").is_none());
		assert!(params.get("temperature").is_none());
		assert_eq!(params["messages"][0]["role"], Value::String("user".to_string()));
		assert_eq!(params["messages"][0]["content"], Value::String("Say hello.".to_string()));
	}

	#[test]
	fn build_params_preserves_explicit_limits_and_wraps_the_default_like_javascript() {
		let context = context_with_user("hi");
		for (limit, model_limit, expected) in [
			(Some(3.9), 8192.0, Some(3.9)),
			(None, 8.7, Some(2.0)),
			(None, 6_442_450_950.0, Some(-2_147_483_646.0)),
			(None, -6_442_450_950.0, Some(2_147_483_646.0)),
			(Some(0.0), 6_442_450_950.0, Some(-2_147_483_646.0)),
			(Some(f64::NAN), 8.7, Some(2.0)),
			(None, f64::INFINITY, Some(0.0)),
			(None, f64::NAN, Some(0.0)),
			(Some(f64::INFINITY), 8192.0, None),
		] {
			let mut model = test_model("anthropic", "claude-sonnet-4-5");
			model.max_tokens = model_limit;
			let mut options = AnthropicOptions::default();
			options.stream.max_tokens = limit;
			let params = build_params(&model, &context, false, &options, None).unwrap();
			assert_eq!(params["max_tokens"].as_f64(), expected, "limit={limit:?}, model_limit={model_limit}");
			if expected.is_none() {
				assert!(params["max_tokens"].is_null());
			}
		}
	}

	#[test]
	fn build_params_adds_cache_control_to_system_and_last_tool() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let context = Context::new(
			Some("System prompt".to_string()),
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 1))],
			Some(vec![
				Tool {
					name: "read".to_string(),
					description: "Read a file".to_string(),
					parameters: serde_json::json!({"properties": {"path": {"type": "string"}}, "required": ["path"]}),
				},
				Tool {
					name: "write".to_string(),
					description: "Write a file".to_string(),
					parameters: serde_json::json!({}),
				},
			]),
		);
		let cache_control = CacheControlEphemeral {
			type_: "ephemeral".to_string(),
			ttl: Some("1h".to_string()),
		};
		let options = AnthropicOptions::default();
		let params = build_params(&model, &context, false, &options, Some(&cache_control)).unwrap();

		assert_eq!(params["system"][0]["type"], Value::String("text".to_string()));
		assert_eq!(params["system"][0]["text"], Value::String("System prompt".to_string()));
		assert_eq!(params["system"][0]["cache_control"]["ttl"], Value::String("1h".to_string()));
		assert_eq!(params["tools"][0]["name"], Value::String("read".to_string()));
		assert_eq!(params["tools"][0]["eager_input_streaming"], Value::Bool(true));
		assert_eq!(params["tools"][0]["input_schema"]["type"], Value::String("object".to_string()));
		assert_eq!(params["tools"][0]["input_schema"]["required"], serde_json::json!(["path"]));
		assert!(params["tools"][0].get("cache_control").is_none());
		assert_eq!(params["tools"][1]["cache_control"]["type"], Value::String("ephemeral".to_string()));
		assert_eq!(params["tools"][1]["input_schema"]["properties"], serde_json::json!({}));
		assert_eq!(params["tools"][1]["input_schema"]["required"], serde_json::json!([]));
		// cache_control also lands on the last user message.
		assert_eq!(
			params["messages"][0]["content"][0]["cache_control"]["type"],
			Value::String("ephemeral".to_string())
		);
	}

	#[test]
	fn build_params_omits_eager_input_streaming_when_unsupported() {
		let mut model = test_model("anthropic", "claude-sonnet-4-5");
		model.compat = Some(crate::types::Compat::Anthropic(crate::types::AnthropicMessagesCompat {
			supports_eager_tool_input_streaming: Some(false),
			supports_long_cache_retention: None,
		}));
		let context = Context::new(
			None,
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 1))],
			Some(vec![Tool {
				name: "read".to_string(),
				description: "Read".to_string(),
				parameters: serde_json::json!({}),
			}]),
		);
		let options = AnthropicOptions::default();
		let params = build_params(&model, &context, false, &options, None).unwrap();
		assert!(params["tools"][0].get("eager_input_streaming").is_none());
		assert!(should_use_fine_grained_tool_streaming_beta(&model, &context));

		let no_tools = context_with_user("hi");
		assert!(!should_use_fine_grained_tool_streaming_beta(&model, &no_tools));
	}

	#[test]
	fn build_params_uses_claude_code_identity_for_oauth() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let context = Context::new(
			Some("Extra system".to_string()),
			vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 1))],
			None,
		);
		let options = AnthropicOptions::default();
		let params = build_params(&model, &context, true, &options, None).unwrap();
		assert_eq!(params["system"][0]["text"], Value::String("You are Claude Code, Anthropic's official CLI for Claude.".to_string()));
		assert_eq!(params["system"][1]["text"], Value::String("Extra system".to_string()));
	}

	#[test]
	fn build_params_gates_temperature_on_thinking_and_always_on_models() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let context = context_with_user("hi");

		let mut options = AnthropicOptions::default();
		options.stream.temperature = Some(0.5);
		let params = build_params(&model, &context, false, &options, None).unwrap();
		assert_eq!(params["temperature"], serde_json::json!(0.5));

		let mut thinking = options.clone();
		thinking.thinking_enabled = Some(true);
		let params = build_params(&model, &context, false, &thinking, None).unwrap();
		assert!(params.get("temperature").is_none());

		let always_on = test_model("anthropic", "claude-fable-5");
		let params = build_params(&always_on, &context, false, &options, None).unwrap();
		assert!(params.get("temperature").is_none());
	}

	#[test]
	fn opus_55_payload_uses_always_on_adaptive_thinking() {
		for id in ["claude-opus-5-5", "claude-opus-5.5", "claude-opus-5-5-20260922"] {
			let model = test_model("anthropic", id);
			let context = context_with_user("hi");
			let mut options = AnthropicOptions::default();
			options.stream.temperature = Some(0.5);
			options.thinking_enabled = Some(false);
			let params = build_params(&model, &context, false, &options, None).unwrap();
			assert!(params.get("temperature").is_none(), "{id}");
			assert!(params.get("thinking").is_none(), "{id}");
			options.thinking_enabled = Some(true);
			let params = build_params(&model, &context, false, &options, None).unwrap();
			assert_eq!(params["thinking"]["type"], "adaptive", "{id}");
		}
	}

	#[test]
	fn build_params_configures_thinking_modes() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let context = context_with_user("hi");

		let mut budget = AnthropicOptions::default();
		budget.thinking_enabled = Some(true);
		budget.thinking_budget_tokens = Some(4096.0);
		let params = build_params(&model, &context, false, &budget, None).unwrap();
		assert_eq!(params["thinking"]["type"], Value::String("enabled".to_string()));
		assert_eq!(params["thinking"]["budget_tokens"].as_f64(), Some(4096.0));
		assert_eq!(params["thinking"]["display"], Value::String("summarized".to_string()));

		let mut adaptive = AnthropicOptions::default();
		adaptive.thinking_enabled = Some(true);
		adaptive.effort = Some("xhigh".to_string());
		adaptive.thinking_display = Some("omitted".to_string());
		let adaptive_model = test_model("anthropic", "claude-opus-4-7");
		let params = build_params(&adaptive_model, &context, false, &adaptive, None).unwrap();
		assert_eq!(params["thinking"]["type"], Value::String("adaptive".to_string()));
		assert_eq!(params["thinking"]["display"], Value::String("omitted".to_string()));
		assert_eq!(params["output_config"]["effort"], Value::String("xhigh".to_string()));

		let mut disabled = AnthropicOptions::default();
		disabled.thinking_enabled = Some(false);
		let params = build_params(&model, &context, false, &disabled, None).unwrap();
		assert_eq!(params["thinking"]["type"], Value::String("disabled".to_string()));

		let always_on = test_model("anthropic", "claude-mythos-5");
		let params = build_params(&always_on, &context, false, &disabled, None).unwrap();
		assert!(params.get("thinking").is_none());

		let mut non_reasoning = test_model("anthropic", "claude-3-haiku");
		non_reasoning.reasoning = false;
		let params = build_params(&non_reasoning, &context, false, &budget, None).unwrap();
		assert!(params.get("thinking").is_none());
	}

	#[test]
	fn build_params_maps_metadata_and_tool_choice() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let context = context_with_user("hi");

		let mut options = AnthropicOptions::default();
		let mut metadata = Map::new();
		metadata.insert("user_id".to_string(), Value::String("user-1".to_string()));
		metadata.insert("other".to_string(), Value::String("ignored".to_string()));
		options.stream.metadata = Some(metadata);
		options.tool_choice = Some(Value::String("any".to_string()));
		let params = build_params(&model, &context, false, &options, None).unwrap();
		assert_eq!(params["metadata"], serde_json::json!({"user_id": "user-1"}));
		assert_eq!(params["tool_choice"], serde_json::json!({"type": "any"}));

		let mut options = AnthropicOptions::default();
		options.tool_choice = Some(serde_json::json!({"type": "tool", "name": "read"}));
		let params = build_params(&model, &context, false, &options, None).unwrap();
		assert_eq!(params["tool_choice"], serde_json::json!({"type": "tool", "name": "read"}));
	}

	#[test]
	fn convert_messages_skips_empty_content_and_merges_tool_results() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let mut empty_assistant = AssistantMessage::default();
		empty_assistant.content = vec![ContentBlock::Text(TextContent::new("   "))];
		let mut calling_assistant = AssistantMessage::default();
		calling_assistant.content = vec![
			ContentBlock::ToolCall(ToolCall::new("call_1", "read", Map::new())),
			ContentBlock::ToolCall(ToolCall::new("call_2", "read", Map::new())),
		];
		let messages = vec![
			Message::user(UserMessage::new(UserContent::Text("   ".to_string()), 1)),
			Message::user(UserMessage::new(UserContent::Text("real".to_string()), 2)),
			Message::assistant(empty_assistant),
			Message::assistant(calling_assistant),
			Message::tool_result(ToolResultMessage::new(
				"call_1",
				"read",
				vec![ImageOrTextContent::Text(TextContent::new("first"))],
				false,
				3,
			)),
			Message::tool_result(ToolResultMessage::new(
				"call_2",
				"read",
				vec![ImageOrTextContent::Text(TextContent::new("second"))],
				true,
				4,
			)),
		];
		let params = convert_messages(&messages, &model, false, None).unwrap();
		assert_eq!(params.len(), 3);
		assert_eq!(params[1]["content"][0]["type"], Value::String("tool_use".to_string()));
		assert_eq!(params[0]["content"], Value::String("real".to_string()));
		assert_eq!(params[1]["role"], Value::String("assistant".to_string()));
		assert_eq!(params[2]["role"], Value::String("user".to_string()));
		assert_eq!(params[2]["content"].as_array().unwrap().len(), 2);
		assert_eq!(params[2]["content"][0]["type"], Value::String("tool_result".to_string()));
		assert_eq!(params[2]["content"][0]["tool_use_id"], Value::String("call_1".to_string()));
		assert_eq!(params[2]["content"][0]["content"], Value::String("first".to_string()));
		assert_eq!(params[2]["content"][0]["is_error"], Value::Bool(false));
		assert_eq!(params[2]["content"][1]["is_error"], Value::Bool(true));
	}

	#[test]
	fn convert_messages_rewrites_redacted_and_unsigned_thinking() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let mut assistant = AssistantMessage::default();
		assistant.provider = model.provider.clone();
		assistant.api = model.api.clone();
		assistant.model = model.id.clone();
		assistant.content = vec![
			ContentBlock::Thinking(ThinkingContent {
				type_: crate::types::THINKING_CONTENT_TYPE.to_string(),
				thinking: "[Reasoning redacted]".to_string(),
				thinking_signature: Some("opaque".to_string()),
				redacted: Some(true),
			}),
			ContentBlock::Thinking(ThinkingContent::new("no signature")),
			ContentBlock::Thinking(ThinkingContent {
				type_: crate::types::THINKING_CONTENT_TYPE.to_string(),
				thinking: "signed".to_string(),
				thinking_signature: Some("sig".to_string()),
				redacted: None,
			}),
			ContentBlock::Text(TextContent::new("answer")),
			ContentBlock::ToolCall(ToolCall::new("toolu_1", "read", Map::new())),
		];
		let messages = vec![
			Message::assistant(assistant),
			Message::tool_result(ToolResultMessage::new(
				"toolu_1",
				"read",
				vec![ImageOrTextContent::Text(TextContent::new("ok"))],
				false,
				1,
			)),
		];
		let params = convert_messages(&messages, &model, false, None).unwrap();
		let blocks = params[0]["content"].as_array().unwrap();
		assert_eq!(blocks.len(), 5);
		assert_eq!(blocks[0]["type"], Value::String("redacted_thinking".to_string()));
		assert_eq!(blocks[0]["data"], Value::String("opaque".to_string()));
		assert_eq!(blocks[1]["type"], Value::String("text".to_string()));
		assert_eq!(blocks[1]["text"], Value::String("no signature".to_string()));
		assert_eq!(blocks[2]["type"], Value::String("thinking".to_string()));
		assert_eq!(blocks[2]["signature"], Value::String("sig".to_string()));
		assert_eq!(blocks[4]["type"], Value::String("tool_use".to_string()));
		assert_eq!(blocks[4]["name"], Value::String("read".to_string()));
	}

	#[test]
	fn convert_messages_renames_tools_and_normalizes_ids_for_oauth() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let mut assistant = AssistantMessage::default();
		assistant.provider = "other".to_string();
		assistant.api = "other-api".to_string();
		assistant.model = "other-model".to_string();
		assistant.content = vec![ContentBlock::ToolCall(ToolCall::new("call|1", "read", Map::new()))];
		let messages = vec![
			Message::assistant(assistant),
			Message::tool_result(ToolResultMessage::new(
				"call|1",
				"read",
				vec![ImageOrTextContent::Text(TextContent::new("ok"))],
				false,
				1,
			)),
		];
		let params = convert_messages(&messages, &model, true, None).unwrap();
		assert_eq!(params[0]["content"][0]["id"], Value::String("call_1".to_string()));
		assert_eq!(params[0]["content"][0]["name"], Value::String("Read".to_string()));
		assert_eq!(params[1]["content"][0]["tool_use_id"], Value::String("call_1".to_string()));
	}

	#[test]
	fn convert_messages_appends_cache_control_to_the_last_user_message() {
		let model = test_model("anthropic", "claude-sonnet-4-5");
		let cache_control = CacheControlEphemeral {
			type_: "ephemeral".to_string(),
			ttl: None,
		};
		let messages = vec![Message::user(UserMessage::new(
			UserContent::Text("hello".to_string()),
			1,
		))];
		let params = convert_messages(&messages, &model, false, Some(&cache_control)).unwrap();
		assert_eq!(params[0]["content"][0]["type"], Value::String("text".to_string()));
		assert_eq!(params[0]["content"][0]["text"], Value::String("hello".to_string()));
		assert_eq!(params[0]["content"][0]["cache_control"]["type"], Value::String("ephemeral".to_string()));

		let assistant_last = vec![Message::user(UserMessage::new(
			UserContent::Text("hello".to_string()),
			1,
		))];
		let params = convert_messages(&assistant_last, &model, false, Some(&cache_control)).unwrap();
		assert_eq!(params.len(), 1);
	}

	#[test]
	fn request_headers_follow_the_stainless_merge_order() {
		let client = AnthropicClientOverride {
			base_url: Some("https://api.anthropic.com".to_string()),
			api_key: Some("sk-ant-api".to_string()),
			auth_token: None,
			headers: Some(
				[
					("accept".to_string(), "application/json".to_string()),
					("anthropic-dangerous-direct-browser-access".to_string(), "true".to_string()),
					("x-api-key".to_string(), "sk-ant-api".to_string()),
					("anthropic-beta".to_string(), "fine-grained-tool-streaming-2025-05-14".to_string()),
				]
				.into_iter()
				.collect(),
			),
		};
		let headers = build_request_headers(&client, 600_000.0, true);
		let names: Vec<String> = headers.entries.iter().map(|(name, _)| name.clone()).collect();
		assert_eq!(
			names,
			vec![
				"User-Agent",
				"X-Stainless-Retry-Count",
				"X-Stainless-Timeout",
				"X-Stainless-Lang",
				"X-Stainless-Package-Version",
				"X-Stainless-OS",
				"X-Stainless-Arch",
				"X-Stainless-Runtime",
				"X-Stainless-Runtime-Version",
				"anthropic-version",
				"accept",
				"anthropic-dangerous-direct-browser-access",
				"x-api-key",
				"anthropic-beta",
				"content-type"
			]
		);
		assert_eq!(headers.get("X-Stainless-Timeout"), Some("600"));
		assert_eq!(headers.get("anthropic-version"), Some("2023-06-01"));
		assert_eq!(headers.get("anthropic-beta"), Some("fine-grained-tool-streaming-2025-05-14"));
		assert_eq!(headers.get("content-type"), Some("application/json"));
	}

	#[test]
	fn messages_url_joins_base_url_like_the_sdk() {
		assert_eq!(
			build_messages_url("https://api.anthropic.com"),
			"https://api.anthropic.com/v1/messages"
		);
		assert_eq!(
			build_messages_url("https://example.test/proxy/"),
			"https://example.test/proxy/v1/messages"
		);
	}

	#[test]
	fn create_client_branches_match_the_typescript_headers() {
		let mut env = crate::test_env::ScopedEnv::new();
		let mut copilot = test_model("github-copilot", "claude-sonnet-4-5");
		copilot.base_url = "https://api.githubcopilot.com".to_string();
		let created = create_client(
			&copilot,
			"copilot-token",
			true,
			true,
			None,
			Some(&build_copilot_dynamic_headers(CopilotDynamicHeaderParams {
				messages: &[],
				has_images: false,
			})),
			None,
		)
		.unwrap();
		assert!(!created.is_oauth_token);
		let headers = created.client.headers.unwrap();
		assert_eq!(created.client.auth_token.as_deref(), Some("copilot-token"));
		assert_eq!(created.client.api_key, None);
		assert_eq!(headers.get("X-Initiator").map(String::as_str), Some("user"));
		assert_eq!(headers.get("Openai-Intent").map(String::as_str), Some("conversation-edits"));
		assert_eq!(
			headers.get("anthropic-beta").map(String::as_str),
			Some("fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14")
		);

		let oauth = create_client(&test_model("anthropic", "claude-sonnet-4-5"), "sk-ant-oat01-abc", true, false, None, None, None)
			.unwrap();
		assert!(oauth.is_oauth_token);
		assert_eq!(oauth.client.auth_token.as_deref(), Some("sk-ant-oat01-abc"));
		let headers = oauth.client.headers.unwrap();
		assert_eq!(
			headers.get("anthropic-beta").map(String::as_str),
			Some("claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14")
		);
		assert_eq!(headers.get("user-agent").map(String::as_str), Some("claude-cli/2.1.281"));
		assert_eq!(headers.get("x-app").map(String::as_str), Some("cli"));

		let mut opencode = test_model("opencode", "claude-sonnet-4-5");
		opencode.base_url = "https://opencode.test".to_string();
		let created = create_client(&opencode, "key", true, false, None, None, Some("session-1")).unwrap();
		let headers = created.client.headers.unwrap();
		assert_eq!(headers.get("User-Agent").map(String::as_str), Some("prime-agent"));
		assert_eq!(headers.get("x-opencode-session").map(String::as_str), Some("session-1"));

		let mut adaptive = test_model("anthropic", "claude-opus-4-6");
		adaptive.base_url = "https://api.anthropic.com".to_string();
		let created = create_client(&adaptive, "key", true, false, None, None, None).unwrap();
		assert!(created.client.headers.unwrap().get("anthropic-beta").is_none());

		let mut cloudflare = test_model("cloudflare-ai-gateway", "claude-sonnet-4-5");
		cloudflare.base_url = "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/gw/anthropic".to_string();
		env.set("CLOUDFLARE_ACCOUNT_ID", "acct");
		let created = create_client(&cloudflare, "cf-key", true, false, None, None, None).unwrap();
		env.remove("CLOUDFLARE_ACCOUNT_ID");
		assert!(!created.is_oauth_token);
		let headers = created.client.headers.unwrap();
		assert_eq!(created.client.base_url.as_deref(), Some("https://gateway.ai.cloudflare.com/v1/acct/gw/anthropic"));
		assert_eq!(headers.get("cf-aig-authorization").map(String::as_str), Some("Bearer cf-key"));
		assert_eq!(headers.get("anthropic-dangerous-direct-browser-access").map(String::as_str), Some("true"));
		assert!(!headers.contains_key("x-api-key"));
		assert!(!headers.contains_key("Authorization"));
	}

	#[test]
	fn nullable_records_drop_cleared_headers() {
		let mut headers: IndexMap<String, Option<String>> = IndexMap::new();
		headers.insert("Authorization".to_string(), None);
		headers.insert("accept".to_string(), Some("application/json".to_string()));
		let record = nullable_to_record(&headers);
		assert_eq!(record.len(), 1);
		assert_eq!(record.get("accept").map(String::as_str), Some("application/json"));
	}

	#[test]
	fn api_error_messages_and_classification_follow_the_sdk() {
		let body = serde_json::json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}});
		let headers = reqwest::header::HeaderMap::new();
		let error = anthropic_api_error(529, Some(&body), "", &headers);
		match error {
			AnthropicStreamError::Failure(failure) => {
				assert_eq!(failure.message, "Provider overloaded (overloaded_error, 529): Overloaded");
				assert_eq!(failure.info.kind, "overloaded");
				assert_eq!(failure.info.status, Some(529));
			}
			other => panic!("unexpected error {:?}", other),
		}

		let body = serde_json::json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}});
		match anthropic_api_error(429, Some(&body), "", &headers) {
			AnthropicStreamError::Failure(failure) => {
				assert_eq!(failure.info.kind, "rate_limit");
				assert_eq!(failure.message, "Provider rate limit exceeded (rate_limit_error, 429): slow down");
			}
			other => panic!("unexpected error {:?}", other),
		}

		// 403 without a structured type stays permission (never auth).
		match anthropic_api_error(403, None, "Forbidden", &headers) {
			AnthropicStreamError::Failure(failure) => {
				assert_eq!(failure.info.kind, "permission");
				assert_eq!(failure.message, "Provider denied access to the requested resource (PermissionDeniedError, 403)");
			}
			other => panic!("unexpected error {:?}", other),
		}

		// Unrecognized errors pass through verbatim.
		match anthropic_api_error(418, None, "teapot", &headers) {
			AnthropicStreamError::Message(message) => assert_eq!(message, "418 teapot"),
			other => panic!("unexpected error {:?}", other),
		}

		assert_eq!(api_error_message(400, None, ""), "400 status code (no body)");
		assert_eq!(api_error_name(500), "InternalServerError");
		assert_eq!(api_error_name(409), "ConflictError");
		assert_eq!(api_error_name(422), "UnprocessableEntityError");
		assert_eq!(api_error_name(404), "NotFoundError");
		assert_eq!(api_error_name(401), "AuthenticationError");
		assert_eq!(api_error_name(400), "BadRequestError");
		assert_eq!(api_error_name(200), "APIError");
	}

	#[test]
	fn retry_after_headers_are_parsed() {
		let mut headers = reqwest::header::HeaderMap::new();
		assert_eq!(parse_retry_after_ms(&headers), None);
		headers.insert("retry-after-ms", "1500".parse().unwrap());
		assert_eq!(parse_retry_after_ms(&headers), Some(1500.0));
		headers.remove("retry-after-ms");
		headers.insert("retry-after", "2".parse().unwrap());
		assert_eq!(parse_retry_after_ms(&headers), Some(2000.0));
		headers.insert("retry-after", "-1".parse().unwrap());
		assert_eq!(parse_retry_after_ms(&headers), None);
		// stream-failure.ts:209-210: an HTTP-date Retry-After yields the wait until
		// that date (never negative); a non-date falls back to undefined.
		headers.insert(
			"retry-after",
			"Wed, 21 Oct 2099 07:28:00 GMT".parse().unwrap(),
		);
		let date_ms = parse_retry_after_ms(&headers).expect("HTTP-date Retry-After must parse");
		assert!(date_ms > 0.0);
		headers.insert("retry-after", "Wed, 21 Oct".parse().unwrap());
		assert_eq!(parse_retry_after_ms(&headers), None);
	}

	#[test]
	fn cache_creation_usage_is_read_from_the_wire() {
		assert!(parse_cache_creation(None).is_none());
		assert!(parse_cache_creation(Some(&Value::Null)).is_none());
		let usage = parse_cache_creation(Some(&serde_json::json!({
			"ephemeral_5m_input_tokens": 250,
			"ephemeral_1h_input_tokens": 750
		})))
		.expect("usage");
		assert_eq!(usage.ephemeral_5m_input_tokens, 250.0);
		assert_eq!(usage.ephemeral_1h_input_tokens, 750.0);
	}

	#[test]
	fn anthropic_options_round_trip_through_serde() {
		let base = StreamOptions {
			temperature: Some(0.2),
			max_tokens: Some(1024.0),
			api_key: Some("key".to_string()),
			..Default::default()
		};
		let options = AnthropicOptions::from_base(&base);
		let value = serde_json::to_value(&options).unwrap();
		assert_eq!(value["temperature"], serde_json::json!(0.2));
		assert_eq!(value["maxTokens"].as_f64(), Some(1024.0));
		let back: AnthropicOptions = serde_json::from_value(value).unwrap();
		assert_eq!(back.stream.max_tokens, Some(1024.0));
		assert_eq!(back.thinking_enabled, None);

		let with_extras: AnthropicOptions = serde_json::from_value(serde_json::json!({
			"temperature": 0.5,
			"thinkingEnabled": true,
			"thinkingBudgetTokens": 2048,
			"effort": "high",
			"thinkingDisplay": "omitted",
			"interleavedThinking": false,
			"toolChoice": "auto",
			"unknownKey": "kept"
		}))
		.unwrap();
		assert_eq!(with_extras.thinking_enabled, Some(true));
		assert_eq!(with_extras.thinking_budget_tokens, Some(2048.0));
		assert_eq!(with_extras.effort.as_deref(), Some("high"));
		assert_eq!(with_extras.thinking_display.as_deref(), Some("omitted"));
		assert_eq!(with_extras.interleaved_thinking, Some(false));
		assert_eq!(with_extras.tool_choice, Some(Value::String("auto".to_string())));
		assert_eq!(with_extras.stream.extra.get("unknownKey"), Some(&Value::String("kept".to_string())));
	}

	#[test]
	fn anthropic_options_keep_unsupported_cache_retention_values() {
		// `cacheRetention` is an open string union; the provider only branches on
		// "none" / "long" and treats everything else as "short".
		let options: AnthropicOptions = serde_json::from_value(serde_json::json!({
			"cacheRetention": "custom",
			"serviceTier": null
		}))
		.unwrap();
		assert_eq!(options.stream.cache_retention.as_deref(), Some("custom"));
		assert_eq!(resolve_cache_retention(options.stream.cache_retention.as_ref()), "custom");
	}
}
