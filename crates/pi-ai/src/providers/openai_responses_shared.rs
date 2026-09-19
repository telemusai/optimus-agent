//! Port of packages/ai/src/providers/openai-responses-shared.ts
//!
//! The TypeScript talks to the OpenAI SDK's typed `ResponseStreamEvent` union; the Rust
//! port receives the already-parsed SSE payloads as `serde_json::Value` and reads the same
//! fields with the same defaults. Every helper lives in this file (no shared module).

use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use serde_json::{Map, Value};

use crate::models::calculate_cost;
use crate::providers::transform_messages::try_transform_messages;
use crate::types::{
    AssistantMessage, AssistantMessageEvent, Context, ContentBlock, ImageOrTextContent, InputModality, Message, Model,
    ProviderUsageObservation, TextContent, ThinkingContent, Tool, ToolCall, ToolResultMessage, Usage, UserContent,
    TextSignatureV1,
};
use crate::utils::diagnostics::{
    append_assistant_message_diagnostic, now_millis, AssistantMessageDiagnostic,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::hash::short_hash;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::sanitize_unicode::sanitize_surrogates;
use crate::utils::stream_failure::{
    classify_stream_failure, StreamFailureError, StreamFailureInfo, ThrownStreamError,
};

/// `encodeTextSignatureV1(id, phase?)`.
pub fn encode_text_signature_v1(id: &str, phase: Option<&str>) -> String {
    let payload = TextSignatureV1 {
        v: 1,
        id: id.to_string(),
        phase: phase.map(str::to_string),
    };
    serde_json::to_string(&payload).expect("TextSignatureV1 serializes")
}

/// `parseTextSignature(signature)` result.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedTextSignature {
    pub id: String,
    pub phase: Option<String>,
}

/// `parseTextSignature(signature)`.
pub fn parse_text_signature(signature: Option<&str>) -> Option<ParsedTextSignature> {
    let signature = signature?;
    if signature.is_empty() {
        return None;
    }
    if signature.starts_with('{') {
        // `try { ... } catch { /* Fall through to legacy plain-string handling. */ }`
        if let Ok(parsed) = serde_json::from_str::<Value>(signature) {
            if parsed.get("v").and_then(Value::as_i64) == Some(1) {
                if let Some(id) = parsed.get("id").and_then(Value::as_str) {
                    let phase = parsed.get("phase").and_then(Value::as_str);
                    if phase == Some("commentary") || phase == Some("final_answer") {
                        return Some(ParsedTextSignature {
                            id: id.to_string(),
                            phase: phase.map(str::to_string),
                        });
                    }
                    return Some(ParsedTextSignature {
                        id: id.to_string(),
                        phase: None,
                    });
                }
            }
        }
    }
    Some(ParsedTextSignature {
        id: signature.to_string(),
        phase: None,
    })
}

/// TS: `onOutputItemDone?: (item) => void | Promise<void>`
pub type OnOutputItemDone = Arc<dyn Fn(Value) -> crate::types::BoxFuture<()> + Send + Sync>;
/// TS: `resolveServiceTier?: (responseServiceTier, requestServiceTier) => service_tier | undefined`
pub type ResolveServiceTier = Arc<dyn Fn(Option<&str>, Option<&str>) -> Option<String> + Send + Sync>;
/// TS: `applyServiceTierPricing?: (usage, serviceTier) => void`
pub type ApplyServiceTierPricing = Arc<dyn Fn(&mut Usage, Option<&str>) + Send + Sync>;

/// `interface OpenAIResponsesStreamOptions`.
#[derive(Clone, Default)]
pub struct OpenAIResponsesStreamOptions {
    pub on_output_item_done: Option<OnOutputItemDone>,
    /// `serviceTier?: ResponseCreateParamsStreaming["service_tier"]`
    pub service_tier: Option<String>,
    pub resolve_service_tier: Option<ResolveServiceTier>,
    pub apply_service_tier_pricing: Option<ApplyServiceTierPricing>,
    pub on_usage_observation: Option<crate::types::OnUsageObservation>,
}

/// The Rust counterpart of a `throw` inside the TypeScript stream loop.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponsesStreamError {
    /// `throw new StreamFailureError(...)`
    StreamFailure(StreamFailureError),
    /// `throw new Error(...)`
    Message(String),
}

impl ResponsesStreamError {
    /// `error.message`
    pub fn message(&self) -> String {
        match self {
            ResponsesStreamError::StreamFailure(failure) => failure.message.clone(),
            ResponsesStreamError::Message(message) => message.clone(),
        }
    }

    /// The thrown value as seen by the provider's catch block.
    pub fn to_thrown(&self) -> ThrownStreamError<'_> {
        match self {
            ResponsesStreamError::StreamFailure(failure) => ThrownStreamError::Failure(failure),
            ResponsesStreamError::Message(message) => ThrownStreamError::Message(message),
        }
    }
}

impl std::fmt::Display for ResponsesStreamError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message())
    }
}

impl std::error::Error for ResponsesStreamError {}

fn stream_failure(message: impl Into<String>, kind: &str, provider_error_type: Option<&str>) -> ResponsesStreamError {
    ResponsesStreamError::StreamFailure(StreamFailureError::new(
        message,
        StreamFailureInfo {
            kind: kind.to_string(),
            provider_error_type: provider_error_type.map(str::to_string),
            ..Default::default()
        },
    ))
}

/// `finiteToken(value)`: a finite number >= 0, otherwise `null`.
fn finite_token(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(number)) => number.as_f64().filter(|value| value.is_finite() && *value >= 0.0),
        _ => None,
    }
}

/// `interface ConvertResponsesMessagesOptions`.
#[derive(Debug, Clone, Default)]
pub struct ConvertResponsesMessagesOptions {
    pub include_system_prompt: Option<bool>,
}

/// `interface ConvertResponsesToolsOptions`.
#[derive(Debug, Clone, Default)]
pub struct ConvertResponsesToolsOptions {
    /// `strict?: boolean | null`
    pub strict: Option<Option<bool>>,
}

/// `convertResponsesMessages(model, context, allowedToolCallProviders, options?)`.
///
/// Returns `ResponseInput` as JSON items; the TypeScript `throw` inside
/// `transformMessages` becomes `Err`.
pub fn convert_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &dyn Fn(&str) -> bool,
    options: Option<&ConvertResponsesMessagesOptions>,
) -> Result<Vec<Value>, String> {
    let mut messages: Vec<Value> = Vec::new();

    let normalize_id_part = |part: &str| -> String {
        let sanitized: String = part
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .collect();
        let normalized = if sanitized.chars().count() > 64 {
            sanitized.chars().take(64).collect::<String>()
        } else {
            sanitized
        };
        normalized.trim_end_matches('_').to_string()
    };

    let build_foreign_responses_item_id = |item_id: &str| -> String {
        let normalized = format!("fc_{}", short_hash(item_id));
        if normalized.chars().count() > 64 {
            normalized.chars().take(64).collect()
        } else {
            normalized
        }
    };

    let normalize_tool_call_id =
        |id: &str, _target_model: &Model, source: &AssistantMessage| -> String {
            if !allowed_tool_call_providers(&model.provider) {
                return normalize_id_part(id);
            }
            if !id.contains('|') {
                return normalize_id_part(id);
            }
            let mut parts = id.split('|');
            let call_id = parts.next().unwrap_or_default();
            let item_id = parts.next().unwrap_or_default();
            let normalized_call_id = normalize_id_part(call_id);
            let is_foreign_tool_call = source.provider != model.provider || source.api != model.api;
            let mut normalized_item_id = if is_foreign_tool_call {
                build_foreign_responses_item_id(item_id)
            } else {
                normalize_id_part(item_id)
            };
            // OpenAI Responses API requires item id to start with "fc"
            if !normalized_item_id.starts_with("fc_") {
                normalized_item_id = normalize_id_part(&format!("fc_{}", normalized_item_id));
            }
            format!("{}|{}", normalized_call_id, normalized_item_id)
        };

    let transformed_messages =
        try_transform_messages(context.messages.clone(), model, Some(&normalize_tool_call_id))?;

    let include_system_prompt = options.and_then(|options| options.include_system_prompt).unwrap_or(true);
    if include_system_prompt {
        if let Some(system_prompt) = context.system_prompt.as_ref() {
            let role = if model.reasoning { "developer" } else { "system" };
            let mut item = Map::new();
            item.insert("role".to_string(), Value::String(role.to_string()));
            item.insert("content".to_string(), Value::String(sanitize_surrogates(system_prompt)));
            messages.push(Value::Object(item));
        }
    }

    let mut msg_index = 0usize;
    for msg in transformed_messages.iter() {
        match msg {
            Message::User(user) => {
                if let Some(provider_context) = user.provider_context.as_ref() {
                    // The server owns this opaque window; SDK unions can lag new response item types.
                    messages.extend(provider_context.items.iter().cloned().map(Value::Object));
                    // openai-responses-shared.ts:142 `continue` skips the only `msgIndex++`
                    // (openai-responses-shared.ts:265), so this message does NOT advance the
                    // index used by the generated `msg_N` ids at openai-responses-shared.ts:189.
                    continue;
                }
                match &user.content {
                    UserContent::Text(text) => {
                        let mut item = Map::new();
                        item.insert("role".to_string(), Value::String("user".to_string()));
                        item.insert(
                            "content".to_string(),
                            Value::Array(vec![input_text_item(&sanitize_surrogates(text))]),
                        );
                        messages.push(Value::Object(item));
                    }
                    UserContent::Blocks(blocks) => {
                        let content: Vec<Value> = blocks
                            .iter()
                            .map(|item| match item {
                                ImageOrTextContent::Text(text) => input_text_item(&sanitize_surrogates(&text.text)),
                                ImageOrTextContent::Image(image) => input_image_item(&format!(
                                    "data:{};base64,{}",
                                    image.mime_type, image.data
                                )),
                            })
                            .collect();
                        if content.is_empty() {
                            // openai-responses-shared.ts:163 `continue` skips `msgIndex++`
                            // (openai-responses-shared.ts:265).
                            continue;
                        }
                        let mut item = Map::new();
                        item.insert("role".to_string(), Value::String("user".to_string()));
                        item.insert("content".to_string(), Value::Array(content));
                        messages.push(Value::Object(item));
                    }
                }
            }
            Message::Assistant(assistant_msg) => {
                let mut output: Vec<Value> = Vec::new();
                let is_different_model = assistant_msg.model != model.id
                    && assistant_msg.provider == model.provider
                    && assistant_msg.api == model.api;

                for block in assistant_msg.content.iter() {
                    match block {
                        ContentBlock::Thinking(thinking) => {
                            if let Some(signature) = thinking.thinking_signature.as_ref() {
                                if let Ok(reasoning_item) = serde_json::from_str::<Value>(signature) {
                                    output.push(reasoning_item);
                                }
                            }
                        }
                        ContentBlock::Text(text_block) => {
                            let parsed_signature = parse_text_signature(text_block.text_signature.as_deref());
                            // OpenAI requires id to be max 64 characters
                            let mut msg_id = parsed_signature.as_ref().map(|parsed| parsed.id.clone());
                            match msg_id.clone() {
                                None => msg_id = Some(format!("msg_{}", msg_index)),
                                Some(id) if id.chars().count() > 64 => {
                                    msg_id = Some(format!("msg_{}", short_hash(&id)));
                                }
                                Some(_) => {}
                            }
                            let mut item = Map::new();
                            item.insert("type".to_string(), Value::String("message".to_string()));
                            item.insert("role".to_string(), Value::String("assistant".to_string()));
                            item.insert(
                                "content".to_string(),
                                Value::Array(vec![output_text_item(&sanitize_surrogates(&text_block.text))]),
                            );
                            item.insert("status".to_string(), Value::String("completed".to_string()));
                            item.insert("id".to_string(), Value::String(msg_id.unwrap_or_default()));
                            item.insert(
                                "phase".to_string(),
                                match parsed_signature.as_ref().and_then(|parsed| parsed.phase.clone()) {
                                    Some(phase) => Value::String(phase),
                                    None => Value::Null,
                                },
                            );
                            output.push(Value::Object(item));
                        }
                        ContentBlock::ToolCall(tool_call) => {
                            let mut id_parts = tool_call.id.split('|');
                            let call_id = id_parts.next().unwrap_or_default();
                            let item_id_raw = id_parts.next();
                            let mut item_id: Option<String> = item_id_raw.map(str::to_string);

                            // For different-model messages, set id to undefined to avoid pairing
                            // validation. OpenAI tracks which fc_xxx IDs were paired with rs_xxx
                            // reasoning items. By omitting the id, we avoid triggering that
                            // validation (like cross-provider does).
                            if is_different_model {
                                if item_id.as_deref().map(|id| id.starts_with("fc_")).unwrap_or(false) {
                                    item_id = None;
                                }
                            }

                            let mut item = Map::new();
                            item.insert("type".to_string(), Value::String("function_call".to_string()));
                            item.insert(
                                "id".to_string(),
                                match item_id {
                                    Some(item_id) => Value::String(item_id),
                                    None => Value::Null,
                                },
                            );
                            item.insert("call_id".to_string(), Value::String(call_id.to_string()));
                            item.insert("name".to_string(), Value::String(tool_call.name.clone()));
                            item.insert(
                                "arguments".to_string(),
                                Value::String(
                                    serde_json::to_string(&Value::Object(tool_call.arguments.clone()))
                                        .expect("arguments serialize"),
                                ),
                            );
                            output.push(Value::Object(item));
                        }
                    }
                }
                if output.is_empty() {
                    // openai-responses-shared.ts:222 `continue` skips `msgIndex++`
                    // (openai-responses-shared.ts:265).
                    continue;
                }
                messages.extend(output);
            }
            Message::ToolResult(result) => {
                let text_result = tool_result_text(result);
                let has_images = result
                    .content
                    .iter()
                    .any(|content| matches!(content, ImageOrTextContent::Image(_)));
                let has_text = !text_result.is_empty();
                let call_id = result.tool_call_id.split('|').next().unwrap_or_default().to_string();

                let output: Value = if has_images
                    && model.input.iter().any(|modality| matches!(modality, InputModality::Image))
                {
                    let mut content_parts: Vec<Value> = Vec::new();

                    if has_text {
                        content_parts.push(input_text_item(&sanitize_surrogates(&text_result)));
                    }

                    for block in result.content.iter() {
                        if let ImageOrTextContent::Image(image) = block {
                            content_parts.push(input_image_item(&format!(
                                "data:{};base64,{}",
                                image.mime_type, image.data
                            )));
                        }
                    }

                    Value::Array(content_parts)
                } else {
                    let text = if has_text {
                        text_result.clone()
                    } else if has_images {
                        "(see attached image)".to_string()
                    } else {
                        String::new()
                    };
                    Value::String(sanitize_surrogates(&text))
                };

                let mut item = Map::new();
                item.insert("type".to_string(), Value::String("function_call_output".to_string()));
                item.insert("call_id".to_string(), Value::String(call_id));
                item.insert("output".to_string(), output);
                messages.push(Value::Object(item));
            }
        }
        // The only increment, mirroring openai-responses-shared.ts:265: every `continue`
        // above skips it.
        msg_index += 1;
    }

    Ok(messages)
}

/// `{ type: "input_text", text }`.
fn input_text_item(text: &str) -> Value {
    let mut item = Map::new();
    item.insert("type".to_string(), Value::String("input_text".to_string()));
    item.insert("text".to_string(), Value::String(text.to_string()));
    Value::Object(item)
}

/// `{ type: "input_image", detail: "auto", image_url }`.
fn input_image_item(image_url: &str) -> Value {
    let mut item = Map::new();
    item.insert("type".to_string(), Value::String("input_image".to_string()));
    item.insert("detail".to_string(), Value::String("auto".to_string()));
    item.insert("image_url".to_string(), Value::String(image_url.to_string()));
    Value::Object(item)
}

/// `{ type: "output_text", text, annotations: [] }`.
fn output_text_item(text: &str) -> Value {
    let mut item = Map::new();
    item.insert("type".to_string(), Value::String("output_text".to_string()));
    item.insert("text".to_string(), Value::String(text.to_string()));
    item.insert("annotations".to_string(), Value::Array(Vec::new()));
    Value::Object(item)
}

/// `msg.content.filter(c => c.type === "text").map(c => c.text).join("\n")`.
fn tool_result_text(result: &ToolResultMessage) -> String {
    result
        .content
        .iter()
        .filter_map(|content| match content {
            ImageOrTextContent::Text(text) => Some(text.text.clone()),
            ImageOrTextContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `convertResponsesTools(tools, options?)`.
pub fn convert_responses_tools(tools: &[Tool], options: Option<&ConvertResponsesToolsOptions>) -> Vec<Value> {
    let strict = match options {
        Some(options) => match &options.strict {
            Some(strict) => *strict,
            None => Some(false),
        },
        None => Some(false),
    };
    tools
        .iter()
        .map(|tool| {
            let mut item = Map::new();
            item.insert("type".to_string(), Value::String("function".to_string()));
            item.insert("name".to_string(), Value::String(tool.name.clone()));
            item.insert("description".to_string(), Value::String(tool.description.clone()));
            // TypeBox already generates JSON Schema
            item.insert("parameters".to_string(), tool.parameters.clone());
            item.insert(
                "strict".to_string(),
                match strict {
                    Some(strict) => Value::Bool(strict),
                    None => Value::Null,
                },
            );
            Value::Object(item)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// processResponsesStream
// ---------------------------------------------------------------------------

/// The OpenAI SDK stream is an `AsyncIterable<ResponseStreamEvent>`; the port consumes
/// already-parsed JSON event payloads.
pub type ResponsesEventStream = Pin<Box<dyn futures::Stream<Item = Value> + Send>>;

/// `blockIndex()` - the index of the block currently being streamed.
fn block_index(output: &AssistantMessage) -> usize {
    output.content.len().saturating_sub(1)
}

/// The streaming scratch state of the current block. The TypeScript keeps the same object
/// in `output.content` and in `currentBlock`; the port keeps the index plus the scratch
/// buffer (`partialJson` is never persisted into `output.content`).
enum CurrentBlock {
    Thinking { index: usize },
    Text { index: usize },
    ToolCall { index: usize, partial_json: String },
}

impl CurrentBlock {
    fn index(&self) -> usize {
        match self {
            CurrentBlock::Thinking { index } | CurrentBlock::Text { index } => *index,
            CurrentBlock::ToolCall { index, .. } => *index,
        }
    }
}

/// `item.type` for an output item.
fn item_type(value: &Value) -> Option<&str> {
    value.get("type").and_then(Value::as_str)
}

/// `value?.key` for an object.
fn get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key)
}

/// `value?.key` as a string.
fn get_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

/// `JSON.stringify(value)`.
fn stringify(value: &Value) -> String {
    serde_json::to_string(value).expect("value serializes")
}

/// `toolCall.arguments` - the TypeScript keeps whatever `parseStreamingJson` returned;
/// the Rust `ToolCall` field is a JSON object, so a non-object parse result maps to `{}`.
fn as_arguments(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

/// `${value}` for a possibly missing JSON value (JS prints "undefined"/"null").
fn js_display(value: Option<&Value>) -> String {
    match value {
        None => "undefined".to_string(),
        Some(Value::Null) => "null".to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}

/// `value || ""` for strings.
fn string_or_empty(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or_default().to_string()
}

/// `value || 0` for numbers.
fn number_or_zero(value: Option<&Value>) -> f64 {
    value.and_then(Value::as_f64).unwrap_or(0.0)
}

/// `item.summary = item.summary || []` then `push(part)`.
fn push_summary_part(item: &mut Value, part: Value) {
    let Some(object) = item.as_object_mut() else {
        return;
    };
    if !object.get("summary").map(Value::is_array).unwrap_or(false) {
        object.insert("summary".to_string(), Value::Array(Vec::new()));
    }
    if let Some(Value::Array(summary)) = object.get_mut("summary") {
        summary.push(part);
    }
}

/// `item.content = item.content || []` then `push(part)`.
fn push_content_part(item: &mut Value, part: Value) {
    let Some(object) = item.as_object_mut() else {
        return;
    };
    if !object.get("content").map(Value::is_array).unwrap_or(false) {
        object.insert("content".to_string(), Value::Array(Vec::new()));
    }
    if let Some(Value::Array(content)) = object.get_mut("content") {
        content.push(part);
    }
}

/// `lastPart.text += delta` (a missing text behaves like "").
fn append_part_text(part: &mut Value, key: &str, delta: &str) {
    let Some(object) = part.as_object_mut() else {
        return;
    };
    let current = object.get(key).and_then(Value::as_str).unwrap_or_default().to_string();
    object.insert(key.to_string(), Value::String(format!("{}{}", current, delta)));
}

/// The last element of an array field.
fn last_array_item<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key).and_then(Value::as_array).and_then(|items| items.last())
}

/// The last element of an array field, mutable.
fn last_array_item_mut<'a>(value: &'a mut Value, key: &str) -> Option<&'a mut Value> {
    value
        .get_mut(key)
        .and_then(Value::as_array_mut)
        .and_then(|items| items.last_mut())
}

/// `item.summary?.map(s => s.text).join("\n\n") || ""`.
fn joined_part_text(item: &Value, key: &str) -> String {
    match item.get(key).and_then(Value::as_array) {
        Some(parts) => parts
            .iter()
            .map(|part| part.get("text").and_then(Value::as_str).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n\n"),
        None => String::new(),
    }
}

/// `mapStopReason(status)`.
fn map_stop_reason(status: Option<&str>) -> Result<String, ResponsesStreamError> {
    let Some(status) = status else {
        return Ok("stop".to_string());
    };
    match status {
        "completed" => Ok("stop".to_string()),
        "incomplete" => Ok("length".to_string()),
        "failed" | "cancelled" => Ok("error".to_string()),
        "in_progress" | "queued" => Ok("stop".to_string()),
        other => Err(ResponsesStreamError::Message(format!(
            "Unhandled stop reason: {}",
            other
        ))),
    }
}

/// True when the message carries no visible text, no reasoning text and no tool call.
///
/// An empty text or thinking block counts as no content: the provider opened the
/// block but never filled it. A completed turn may legitimately end this way after
/// streaming tokens into reasoning the client cannot render, so this is only used
/// to label the terminal, never to fail or retry it.
pub fn has_no_deliverable_content(message: &AssistantMessage) -> bool {
    !message.content.iter().any(|block| match block {
        ContentBlock::Text(text) => !text.text.trim().is_empty(),
        ContentBlock::Thinking(thinking) => !thinking.thinking.trim().is_empty(),
        ContentBlock::ToolCall(_) => true,
    })
}

/// Record a content-free `empty_completion` diagnostic on a terminal that consumed
/// output tokens but delivered nothing visible.
///
/// Additive only: `stop_reason`, `error_message`, usage and every retry decision stay
/// exactly as the provider reported them. The diagnostic carries counts and the raw
/// stop signal, never content, so it cannot leak text into the transcript.
fn note_empty_completion(output: &mut AssistantMessage, reason: &str) {
    if output.stop_reason != "stop" {
        return;
    }
    if !has_no_deliverable_content(output) {
        return;
    }
    let output_tokens = if output.usage.output.is_finite() {
        output.usage.output
    } else {
        0.0
    };
    if output_tokens <= 0.0 {
        // No billed output means the provider sent nothing at all; that is a
        // different (transport) concern and stays out of this diagnostic.
        return;
    }
    let mut details = Map::new();
    details.insert("reason".to_string(), Value::String(reason.to_string()));
    details.insert("outputTokens".to_string(), serde_json::json!(output_tokens));
    details.insert(
        "totalTokens".to_string(),
        serde_json::json!(if output.usage.total_tokens.is_finite() {
            output.usage.total_tokens
        } else {
            0.0
        }),
    );
    details.insert(
        "contentBlocks".to_string(),
        serde_json::json!(output.content.len() as f64),
    );
    let diagnostic = AssistantMessageDiagnostic {
        type_: "empty_completion".to_string(),
        timestamp: now_millis(),
        error: None,
        details: Some(details),
    };
    append_assistant_message_diagnostic(output, diagnostic);
}

/// The TypeScript `for await (const event of openaiStream) { ... }` body.
pub async fn process_responses_stream(
    mut openai_stream: ResponsesEventStream,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    model: &Model,
    options: Option<&OpenAIResponsesStreamOptions>,
) -> Result<(), ResponsesStreamError> {
    let mut current_item: Option<Value> = None;
    let mut current_block: Option<CurrentBlock> = None;

    while let Some(event) = openai_stream.next().await {
        let event_type = get_str(&event, "type").unwrap_or_default().to_string();

        if event_type == "response.created" {
            if let Some(id) = get(&event, "response").and_then(|response| get_str(response, "id")) {
                output.response_id = Some(id.to_string());
            }
        } else if event_type == "response.output_item.added" {
            let item = get(&event, "item").cloned().unwrap_or(Value::Null);
            match item_type(&item) {
                Some("reasoning") => {
                    current_item = Some(item);
                    let index = output.content.len();
                    output.content.push(ContentBlock::Thinking(ThinkingContent::new("")));
                    current_block = Some(CurrentBlock::Thinking { index });
                    stream.push(AssistantMessageEvent::ThinkingStart {
                        content_index: block_index(output),
                        partial: output.clone(),
                    });
                }
                Some("message") => {
                    current_item = Some(item);
                    let index = output.content.len();
                    output.content.push(ContentBlock::Text(TextContent::new("")));
                    current_block = Some(CurrentBlock::Text { index });
                    stream.push(AssistantMessageEvent::TextStart {
                        content_index: block_index(output),
                        partial: output.clone(),
                    });
                }
                Some("function_call") => {
                    let call_id = get_str(&item, "call_id").unwrap_or_default();
                    let id = get_str(&item, "id").unwrap_or_default();
                    let name = get_str(&item, "name").unwrap_or_default();
                    let partial_json = string_or_empty(get(&item, "arguments"));
                    let tool_call = ToolCall::new(format!("{}|{}", call_id, id), name, Map::new());
                    let index = output.content.len();
                    output.content.push(ContentBlock::ToolCall(tool_call));
                    current_block = Some(CurrentBlock::ToolCall { index, partial_json });
                    current_item = Some(item);
                    stream.push(AssistantMessageEvent::ToolCallStart {
                        content_index: block_index(output),
                        partial: output.clone(),
                    });
                }
                _ => {}
            }
        } else if event_type == "response.reasoning_summary_part.added" {
            if let Some(item) = current_item.as_mut() {
                if item_type(item) == Some("reasoning") {
                    push_summary_part(item, get(&event, "part").cloned().unwrap_or(Value::Null));
                }
            }
        } else if event_type == "response.reasoning_summary_text.delta" {
            let is_reasoning = current_item
                .as_ref()
                .map(|item| item_type(item) == Some("reasoning"))
                .unwrap_or(false);
            if is_reasoning && matches!(current_block, Some(CurrentBlock::Thinking { .. })) {
                let delta = string_or_empty(get(&event, "delta"));
                if let Some(item) = current_item.as_mut() {
                    if !item.get("summary").map(Value::is_array).unwrap_or(false) {
                        if let Some(object) = item.as_object_mut() {
                            object.insert("summary".to_string(), Value::Array(Vec::new()));
                        }
                    }
                    let has_last_part = last_array_item(item, "summary").is_some();
                    if has_last_part {
                        let index = current_block.as_ref().expect("checked").index();
                        if let ContentBlock::Thinking(thinking) = &mut output.content[index] {
                            thinking.thinking.push_str(&delta);
                        }
                        if let Some(last_part) = last_array_item_mut(item, "summary") {
                            append_part_text(last_part, "text", &delta);
                        }
                        stream.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: block_index(output),
                            delta,
                            partial: output.clone(),
                        });
                    }
                }
            }
        } else if event_type == "response.reasoning_summary_part.done" {
            let is_reasoning = current_item
                .as_ref()
                .map(|item| item_type(item) == Some("reasoning"))
                .unwrap_or(false);
            if is_reasoning && matches!(current_block, Some(CurrentBlock::Thinking { .. })) {
                if let Some(item) = current_item.as_mut() {
                    if !item.get("summary").map(Value::is_array).unwrap_or(false) {
                        if let Some(object) = item.as_object_mut() {
                            object.insert("summary".to_string(), Value::Array(Vec::new()));
                        }
                    }
                    let has_last_part = last_array_item(item, "summary").is_some();
                    if has_last_part {
                        let index = current_block.as_ref().expect("checked").index();
                        if let ContentBlock::Thinking(thinking) = &mut output.content[index] {
                            thinking.thinking.push_str("\n\n");
                        }
                        if let Some(last_part) = last_array_item_mut(item, "summary") {
                            append_part_text(last_part, "text", "\n\n");
                        }
                        stream.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: block_index(output),
                            delta: "\n\n".to_string(),
                            partial: output.clone(),
                        });
                    }
                }
            }
        } else if event_type == "response.reasoning_text.delta" {
            let is_reasoning = current_item
                .as_ref()
                .map(|item| item_type(item) == Some("reasoning"))
                .unwrap_or(false);
            if is_reasoning && matches!(current_block, Some(CurrentBlock::Thinking { .. })) {
                let delta = string_or_empty(get(&event, "delta"));
                let index = current_block.as_ref().expect("checked").index();
                if let ContentBlock::Thinking(thinking) = &mut output.content[index] {
                    thinking.thinking.push_str(&delta);
                }
                stream.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: block_index(output),
                    delta,
                    partial: output.clone(),
                });
            }
        } else if event_type == "response.content_part.added" {
            let is_message = current_item
                .as_ref()
                .map(|item| item_type(item) == Some("message"))
                .unwrap_or(false);
            if is_message {
                let part = get(&event, "part").cloned().unwrap_or(Value::Null);
                let part_type = item_type(&part);
                if part_type == Some("refusal") {
                    output.stop_reason_raw = Some("refusal".to_string());
                }
                if part_type == Some("output_text") || part_type == Some("refusal") {
                    if let Some(item) = current_item.as_mut() {
                        push_content_part(item, part);
                    }
                }
            }
        } else if event_type == "response.output_text.delta" {
            let is_message = current_item
                .as_ref()
                .map(|item| item_type(item) == Some("message"))
                .unwrap_or(false);
            if is_message && matches!(current_block, Some(CurrentBlock::Text { .. })) {
                let last_part_type = current_item
                    .as_ref()
                    .and_then(|item| last_array_item(item, "content"))
                    .and_then(item_type)
                    .map(str::to_string);
                if last_part_type.is_none() {
                    continue;
                }
                if last_part_type.as_deref() == Some("output_text") {
                    let delta = string_or_empty(get(&event, "delta"));
                    let index = current_block.as_ref().expect("checked").index();
                    if let ContentBlock::Text(text) = &mut output.content[index] {
                        text.text.push_str(&delta);
                    }
                    if let Some(item) = current_item.as_mut() {
                        if let Some(last_part) = last_array_item_mut(item, "content") {
                            append_part_text(last_part, "text", &delta);
                        }
                    }
                    stream.push(AssistantMessageEvent::TextDelta {
                        content_index: block_index(output),
                        delta,
                        partial: output.clone(),
                    });
                }
            }
        } else if event_type == "response.refusal.delta" {
            output.stop_reason_raw = Some("refusal".to_string());
            let is_message = current_item
                .as_ref()
                .map(|item| item_type(item) == Some("message"))
                .unwrap_or(false);
            if is_message && matches!(current_block, Some(CurrentBlock::Text { .. })) {
                let last_part_type = current_item
                    .as_ref()
                    .and_then(|item| last_array_item(item, "content"))
                    .and_then(item_type)
                    .map(str::to_string);
                if last_part_type.is_none() {
                    continue;
                }
                if last_part_type.as_deref() == Some("refusal") {
                    let delta = string_or_empty(get(&event, "delta"));
                    let index = current_block.as_ref().expect("checked").index();
                    if let ContentBlock::Text(text) = &mut output.content[index] {
                        text.text.push_str(&delta);
                    }
                    if let Some(item) = current_item.as_mut() {
                        if let Some(last_part) = last_array_item_mut(item, "content") {
                            append_part_text(last_part, "refusal", &delta);
                        }
                    }
                    stream.push(AssistantMessageEvent::TextDelta {
                        content_index: block_index(output),
                        delta,
                        partial: output.clone(),
                    });
                }
            }
        } else if event_type == "response.function_call_arguments.delta" {
            let is_function_call = current_item
                .as_ref()
                .map(|item| item_type(item) == Some("function_call"))
                .unwrap_or(false);
            if is_function_call && matches!(current_block, Some(CurrentBlock::ToolCall { .. })) {
                let delta = string_or_empty(get(&event, "delta"));
                let index = current_block.as_ref().expect("checked").index();
                let partial_json = match current_block.as_mut().expect("checked") {
                    CurrentBlock::ToolCall { partial_json, .. } => {
                        partial_json.push_str(&delta);
                        partial_json.clone()
                    }
                    _ => String::new(),
                };
                if let ContentBlock::ToolCall(tool_call) = &mut output.content[index] {
                    tool_call.arguments = as_arguments(parse_streaming_json(Some(&partial_json)));
                }
                stream.push(AssistantMessageEvent::ToolCallDelta {
                    content_index: block_index(output),
                    delta,
                    partial: output.clone(),
                });
            }
        } else if event_type == "response.function_call_arguments.done" {
            let is_function_call = current_item
                .as_ref()
                .map(|item| item_type(item) == Some("function_call"))
                .unwrap_or(false);
            if is_function_call && matches!(current_block, Some(CurrentBlock::ToolCall { .. })) {
                let arguments = string_or_empty(get(&event, "arguments"));
                let previous_partial_json = match current_block.as_mut().expect("checked") {
                    CurrentBlock::ToolCall { partial_json, .. } => {
                        let previous = partial_json.clone();
                        *partial_json = arguments.clone();
                        previous
                    }
                    _ => String::new(),
                };
                let index = current_block.as_ref().expect("checked").index();
                if let ContentBlock::ToolCall(tool_call) = &mut output.content[index] {
                    tool_call.arguments = as_arguments(parse_streaming_json(Some(&arguments)));
                }

                if arguments.starts_with(&previous_partial_json) {
                    let delta = arguments[previous_partial_json.len()..].to_string();
                    if !delta.is_empty() {
                        stream.push(AssistantMessageEvent::ToolCallDelta {
                            content_index: block_index(output),
                            delta,
                            partial: output.clone(),
                        });
                    }
                }
            }
        } else if event_type == "response.output_item.done" {
            if let Some(callback) = options.and_then(|options| options.on_output_item_done.clone()) {
                let item = get(&event, "item").cloned().unwrap_or(Value::Null);
                callback(item).await;
            }
            let item = get(&event, "item").cloned().unwrap_or(Value::Null);

            if item_type(&item) == Some("message") && item.get("content").and_then(Value::as_array)
                .map(|parts| parts.iter().any(|part| item_type(part) == Some("refusal"))).unwrap_or(false)
            {
                output.stop_reason_raw = Some("refusal".to_string());
            }

            if item_type(&item) == Some("reasoning") && matches!(current_block, Some(CurrentBlock::Thinking { .. })) {
                let summary_text = joined_part_text(&item, "summary");
                let content_text = joined_part_text(&item, "content");
                let index = current_block.as_ref().expect("checked").index();
                if let ContentBlock::Thinking(thinking) = &mut output.content[index] {
                    let fallback = thinking.thinking.clone();
                    thinking.thinking = if !summary_text.is_empty() {
                        summary_text
                    } else if !content_text.is_empty() {
                        content_text
                    } else {
                        fallback
                    };
                    thinking.thinking_signature = Some(stringify(&item));
                }
                let content = match &output.content[index] {
                    ContentBlock::Thinking(thinking) => thinking.thinking.clone(),
                    _ => String::new(),
                };
                stream.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: block_index(output),
                    content,
                    partial: output.clone(),
                });
                current_block = None;
            } else if item_type(&item) == Some("message") && matches!(current_block, Some(CurrentBlock::Text { .. })) {
                let text = item
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|content| {
                        content
                            .iter()
                            .map(|part| {
                                if item_type(part) == Some("output_text") {
                                    string_or_empty(get(part, "text"))
                                } else {
                                    string_or_empty(get(part, "refusal"))
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("")
                    })
                    .unwrap_or_default();
                let item_id = get_str(&item, "id").unwrap_or_default();
                let phase = get(&item, "phase").and_then(Value::as_str);
                let index = current_block.as_ref().expect("checked").index();
                if let ContentBlock::Text(text_block) = &mut output.content[index] {
                    text_block.text = text;
                    text_block.text_signature = Some(encode_text_signature_v1(item_id, phase));
                }
                let content = match &output.content[index] {
                    ContentBlock::Text(text_block) => text_block.text.clone(),
                    _ => String::new(),
                };
                stream.push(AssistantMessageEvent::TextEnd {
                    content_index: block_index(output),
                    content,
                    partial: output.clone(),
                });
                current_block = None;
            } else if item_type(&item) == Some("function_call") {
                let args = match current_block.as_ref() {
                    Some(CurrentBlock::ToolCall { partial_json, .. }) if !partial_json.is_empty() => {
                        parse_streaming_json(Some(partial_json))
                    }
                    _ => {
                        let raw = string_or_empty(get(&item, "arguments"));
                        if raw.is_empty() {
                            parse_streaming_json(Some("{}"))
                        } else {
                            parse_streaming_json(Some(&raw))
                        }
                    }
                };
                let arguments = as_arguments(args);

                let tool_call: ToolCall = match current_block.take() {
                    Some(CurrentBlock::ToolCall { index, .. }) => {
                        // Finalize in-place; `partialJson` is only a streaming scratch buffer
                        // and is never persisted into the block.
                        if let ContentBlock::ToolCall(tool_call) = &mut output.content[index] {
                            tool_call.arguments = arguments.clone();
                            tool_call.clone()
                        } else {
                            ToolCall::new(
                                format!(
                                    "{}|{}",
                                    get_str(&item, "call_id").unwrap_or_default(),
                                    get_str(&item, "id").unwrap_or_default()
                                ),
                                get_str(&item, "name").unwrap_or_default(),
                                arguments.clone(),
                            )
                        }
                    }
                    _ => ToolCall::new(
                        format!(
                            "{}|{}",
                            get_str(&item, "call_id").unwrap_or_default(),
                            get_str(&item, "id").unwrap_or_default()
                        ),
                        get_str(&item, "name").unwrap_or_default(),
                        arguments,
                    ),
                };

                current_block = None;
                stream.push(AssistantMessageEvent::ToolCallEnd {
                    content_index: block_index(output),
                    tool_call,
                    partial: output.clone(),
                });
            }
        } else if event_type == "response.completed" || event_type == "response.incomplete" {
            let response = get(&event, "response").cloned().unwrap_or(Value::Null);
            if let Some(id) = get_str(&response, "id") {
                output.response_id = Some(id.to_string());
            }
            if let Some(usage) = get(&response, "usage") {
                if !usage.is_null() {
                    let observation = ProviderUsageObservation {
                        input_tokens: Some(finite_token(get(usage, "input_tokens"))),
                        cached_input_tokens: Some(finite_token(
                            get(usage, "input_tokens_details").and_then(|details| get(details, "cached_tokens")),
                        )),
                        output_tokens: Some(finite_token(get(usage, "output_tokens"))),
                        reasoning_tokens: Some(finite_token(
                            get(usage, "output_tokens_details").and_then(|details| get(details, "reasoning_tokens")),
                        )),
                        total_tokens: Some(finite_token(get(usage, "total_tokens"))),
                        cached_input_included_in_input: Some(Some(true)),
                        reasoning_included_in_output: Some(Some(true)),
                    };
                    if let Some(observer) = options.and_then(|options| options.on_usage_observation.clone()) {
                        // A disposable local observer cannot change provider behavior.
                        let model_for_observer = model.clone();
                        let observed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            observer(observation, &model_for_observer)
                        }));
                        if let Ok(future) = observed {
                            tokio::spawn(async move {
                                let _ = future.await;
                            });
                        }
                    }

                    let cached_tokens =
                        number_or_zero(get(usage, "input_tokens_details").and_then(|details| get(details, "cached_tokens")));
                    output.usage = Usage {
                        // OpenAI includes cached tokens in input_tokens, so subtract to get
                        // non-cached input
                        input: number_or_zero(get(usage, "input_tokens")) - cached_tokens,
                        output: number_or_zero(get(usage, "output_tokens")),
                        cache_read: cached_tokens,
                        cache_write: 0.0,
                        total_tokens: number_or_zero(get(usage, "total_tokens")),
                        cost: Default::default(),
                    };
                }
            }
            calculate_cost(model, &mut output.usage, None);
            if let Some(apply) = options.and_then(|options| options.apply_service_tier_pricing.clone()) {
                let response_service_tier = get_str(&response, "service_tier");
                let request_service_tier = options.and_then(|options| options.service_tier.as_deref());
                let service_tier = match options.and_then(|options| options.resolve_service_tier.clone()) {
                    Some(resolve) => resolve(response_service_tier, request_service_tier),
                    None => response_service_tier.or(request_service_tier).map(str::to_string),
                };
                apply(&mut output.usage, service_tier.as_deref());
            }
            let status = if event_type == "response.incomplete" {
                Some("incomplete")
            } else {
                get_str(&response, "status")
            };
            output.stop_reason = map_stop_reason(status)?;
            if status == Some("incomplete") && output.stop_reason_raw.is_none() {
                // Keep the provider's explicit reason: generic incomplete is
                // not proof of an output-token limit (it may be filtering).
                output.stop_reason_raw = response.get("incomplete_details")
                    .and_then(|details| get_str(details, "reason")).map(str::to_string);
            }
            if response.get("error").is_some_and(|error| !error.is_null()) {
                output.stop_reason = "error".to_string();
                if output.stop_reason_raw.as_deref() != Some("refusal") {
                    output.stop_reason_raw = Some(response.get("error").and_then(|error| get_str(error, "code"))
                        .unwrap_or("failed").to_string());
                }
            }
            if output
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_)))
                && output.stop_reason == "stop"
            {
                output.stop_reason = "toolUse".to_string();
            }
            if output.stop_reason == "error" && output.stop_reason_raw.is_none() {
                if let Some(status) = status {
                    output.stop_reason_raw = Some(status.to_string());
                }
            }
            // A genuinely completed turn that billed output tokens but delivered no
            // text, reasoning or tool call is otherwise indistinguishable from a normal
            // answer. Label it for the operator and telemetry; the terminal status and
            // every retry decision stay untouched. A refusal is its own signal and an
            // incomplete/errored terminal never reaches here.
            if status == Some("completed")
                && output.stop_reason_raw.as_deref() != Some("refusal")
            {
                note_empty_completion(output, "completed_without_deliverable_content");
            }
        } else if event_type == "error" {
            let code = get(&event, "code");
            let message = format!(
                "Error Code {}: {}",
                js_display(code),
                js_display(get(&event, "message"))
            );
            let provider_error_type = code.filter(|value| !value.is_null()).map(|value| js_display(Some(value)));
            let kind = classify_stream_failure(provider_error_type.as_deref(), None);
            return Err(stream_failure(message, kind, provider_error_type.as_deref()));
        } else if event_type == "response.failed" {
            let response = get(&event, "response").cloned().unwrap_or(Value::Null);
            let error = get(&response, "error");
            let details = get(&response, "incomplete_details");
            let provider_error_type = error
                .and_then(|error| get(error, "code"))
                .filter(|value| !value.is_null())
                .map(|value| js_display(Some(value)))
                .or_else(|| {
                    details
                        .and_then(|details| get(details, "reason"))
                        .filter(|value| !value.is_null())
                        .map(|value| js_display(Some(value)))
                });
            let message = if let Some(error) = error.filter(|error| !error.is_null()) {
                format!(
                    "{}: {}",
                    {
                        let code = get(error, "code").map(|value| js_display(Some(value))).unwrap_or_default();
                        if code.is_empty() { "unknown".to_string() } else { code }
                    },
                    {
                        let text = get(error, "message").map(|value| js_display(Some(value))).unwrap_or_default();
                        if text.is_empty() { "no message".to_string() } else { text }
                    }
                )
            } else if let Some(reason) = details
                .and_then(|details| get(details, "reason"))
                .filter(|value| !value.is_null())
            {
                format!("incomplete: {}", js_display(Some(reason)))
            } else {
                "Unknown error (no error details in response)".to_string()
            };
            let kind = classify_stream_failure(provider_error_type.as_deref(), None);
            return Err(stream_failure(message, kind, provider_error_type.as_deref()));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ImageContent, ModelCost, ToolResultMessage, Usage, UserMessage};
    use futures::stream;
    use serde_json::json;

    fn text_model() -> Model {
        let mut model = Model::new("gpt-4o-mini", "GPT-4o mini", "openai-responses", "openai", "https://api.openai.com/v1");
        model.input = vec![InputModality::Text];
        model.cost = ModelCost::default();
        model
    }

    fn vision_model() -> Model {
        let mut model = text_model();
        model.input = vec![InputModality::Text, InputModality::Image];
        model
    }

    fn tool_result(call_id: &str, content: Vec<ImageOrTextContent>) -> ToolResultMessage {
        ToolResultMessage::new(call_id, "bash", content, false, 1)
    }

    fn assistant_tool_call(id: &str) -> AssistantMessage {
        let mut message = AssistantMessage::new("openai-responses", "openai", "gpt-4o-mini", 1);
        message.content = vec![ContentBlock::ToolCall(ToolCall::new(
            id,
            "bash",
            [("cmd".to_string(), json!("true"))].into_iter().collect(),
        ))];
        message.usage = Usage::zero();
        message
    }

    fn allowed(providers: &'static [&'static str]) -> impl Fn(&str) -> bool + 'static {
        move |provider: &str| providers.contains(&provider)
    }

    #[test]
    fn encode_and_parse_text_signature_round_trip() {
        assert_eq!(
            encode_text_signature_v1("msg_1", None),
            "{\"v\":1,\"id\":\"msg_1\"}"
        );
        assert_eq!(
            encode_text_signature_v1("msg_1", Some("final_answer")),
            "{\"v\":1,\"id\":\"msg_1\",\"phase\":\"final_answer\"}"
        );
        assert_eq!(
            parse_text_signature(Some("{\"v\":1,\"id\":\"msg_1\",\"phase\":\"commentary\"}")),
            Some(ParsedTextSignature {
                id: "msg_1".to_string(),
                phase: Some("commentary".to_string())
            })
        );
        // Unknown phase values are dropped.
        assert_eq!(
            parse_text_signature(Some("{\"v\":1,\"id\":\"msg_1\",\"phase\":\"other\"}")),
            Some(ParsedTextSignature {
                id: "msg_1".to_string(),
                phase: None
            })
        );
        // Legacy plain-string handling, including malformed JSON.
        assert_eq!(
            parse_text_signature(Some("legacy-id")),
            Some(ParsedTextSignature {
                id: "legacy-id".to_string(),
                phase: None
            })
        );
        assert_eq!(
            parse_text_signature(Some("{not json")),
            Some(ParsedTextSignature {
                id: "{not json".to_string(),
                phase: None
            })
        );
        assert_eq!(parse_text_signature(None), None);
        assert_eq!(parse_text_signature(Some("")), None);
    }

    #[test]
    fn system_prompt_uses_developer_role_for_reasoning_models() {
        let mut model = text_model();
        let context = Context::new(
            Some("be terse".to_string()),
            vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 1))],
            None,
        );
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        assert_eq!(messages[0]["role"], json!("system"));
        assert_eq!(messages[0]["content"], json!("be terse"));

        model.reasoning = true;
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        assert_eq!(messages[0]["role"], json!("developer"));
    }

    #[test]
    fn include_system_prompt_false_omits_the_system_message() {
        let model = text_model();
        let context = Context::new(
            Some("be terse".to_string()),
            vec![Message::user(UserMessage::new(UserContent::Text("hi".to_string()), 1))],
            None,
        );
        let options = ConvertResponsesMessagesOptions {
            include_system_prompt: Some(false),
        };
        let messages =
            convert_responses_messages(&model, &context, &allowed(&["openai"]), Some(&options)).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"], json!([{ "type": "input_text", "text": "hi" }]));
    }

    #[test]
    fn user_blocks_map_to_input_text_and_input_image() {
        let model = vision_model();
        let context = Context::new(
            None,
            vec![Message::user(UserMessage::new(
                UserContent::Blocks(vec![
                    ImageOrTextContent::Text(TextContent::new("look")),
                    ImageOrTextContent::Image(ImageContent::new("QUJD", "image/png")),
                ]),
                1,
            ))],
            None,
        );
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        assert_eq!(
            messages[0]["content"],
            json!([
                { "type": "input_text", "text": "look" },
                { "type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,QUJD" }
            ])
        );
    }

    #[test]
    fn empty_user_block_list_is_skipped() {
        let model = text_model();
        let context = Context::new(
            None,
            vec![Message::user(UserMessage::new(UserContent::Blocks(vec![]), 1))],
            None,
        );
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn empty_tool_result_text_does_not_emit_the_image_placeholder() {
        let model = vision_model();
        let context = Context::new(
            None,
            vec![
                Message::user(UserMessage::new(UserContent::Text("Run it".to_string()), 1)),
                Message::assistant(assistant_tool_call("tool-1")),
                Message::tool_result(tool_result("tool-1", vec![ImageOrTextContent::Text(TextContent::new(""))])),
            ],
            None,
        );
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        let output = messages
            .iter()
            .find(|item| item["type"] == json!("function_call_output"))
            .unwrap();
        assert_eq!(output["output"], json!(""));
    }

    #[test]
    fn tool_result_images_stay_in_function_call_output() {
        let model = vision_model();
        let context = Context::new(
            None,
            vec![
                Message::user(UserMessage::new(UserContent::Text("Run it".to_string()), 1)),
                Message::assistant(assistant_tool_call("tool-1")),
                Message::tool_result(tool_result(
                    "tool-1",
                    vec![ImageOrTextContent::Image(ImageContent::new("ZmFrZQ==", "image/png"))],
                )),
            ],
            None,
        );
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        let output = messages
            .iter()
            .find(|item| item["type"] == json!("function_call_output"))
            .unwrap();
        assert_eq!(
            output["output"],
            json!([{ "type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,ZmFrZQ==" }])
        );
    }

    #[test]
    fn image_placeholder_used_when_images_are_dropped_by_the_model() {
        let model = text_model();
        let context = Context::new(
            None,
            vec![
                Message::user(UserMessage::new(UserContent::Text("Run it".to_string()), 1)),
                Message::assistant(assistant_tool_call("tool-1")),
                Message::tool_result(tool_result(
                    "tool-1",
                    vec![ImageOrTextContent::Image(ImageContent::new("ZmFrZQ==", "image/png"))],
                )),
            ],
            None,
        );
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        let output = messages
            .iter()
            .find(|item| item["type"] == json!("function_call_output"))
            .unwrap();
        // transformMessages already replaced the image with the placeholder text.
        assert_eq!(output["output"], json!("(tool image omitted: model does not support images)"));
    }

    #[test]
    fn foreign_copilot_tool_call_ids_become_fc_hash() {
        let model = Model::new(
            "gpt-5.3-codex",
            "GPT-5.3 Codex",
            "openai-codex-responses",
            "openai-codex",
            "https://chatgpt.com/backend-api",
        );
        let raw_id = "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA==";

        let mut assistant = AssistantMessage::new("openai-responses", "github-copilot", "gpt-5.3-codex", 1);
        assistant.content = vec![ContentBlock::ToolCall(ToolCall::new(
            raw_id,
            "edit",
            [("path".to_string(), json!("src/styles/app.css"))].into_iter().collect(),
        ))];
        assistant.usage = Usage::zero();
        let context = Context::new(
            Some("You are concise.".to_string()),
            vec![
                Message::user(UserMessage::new(UserContent::Text("Use the tool.".to_string()), 1)),
                Message::assistant(assistant),
                Message::tool_result(tool_result(
                    raw_id,
                    vec![ImageOrTextContent::Text(TextContent::new("ok"))],
                )),
            ],
            None,
        );

        let messages =
            convert_responses_messages(&model, &context, &allowed(&["openai", "openai-codex", "opencode"]), None)
                .unwrap();
        let function_call = messages
            .iter()
            .find(|item| item["type"] == json!("function_call"))
            .unwrap();
        let expected = format!("fc_{}", short_hash(raw_id.split('|').nth(1).unwrap()));
        assert_eq!(function_call["id"], json!(expected));
        assert!(function_call["id"].as_str().unwrap().len() <= 64);
        assert_eq!(function_call["call_id"], json!("call_4VnzVawQXPB9MgYib7CiQFEY"));
        assert_eq!(function_call["arguments"], json!("{\"path\":\"src/styles/app.css\"}"));
    }

    #[test]
    fn disallowed_provider_flattens_the_pipe_in_the_tool_call_id() {
        let model = Model::new("claude", "Claude", "anthropic-messages", "anthropic", "https://api.anthropic.com");
        let mut assistant = AssistantMessage::new("openai-responses", "openai", "gpt-4o", 1);
        assistant.content = vec![ContentBlock::ToolCall(ToolCall::new(
            "call_1|fc_1",
            "bash",
            Map::new(),
        ))];
        assistant.usage = Usage::zero();
        let context = Context::new(
            None,
            vec![
                Message::assistant(assistant),
                Message::tool_result(tool_result("call_1|fc_1", vec![ImageOrTextContent::Text(TextContent::new("ok"))])),
            ],
            None,
        );
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        let function_call = messages
            .iter()
            .find(|item| item["type"] == json!("function_call"))
            .unwrap();
        // A disallowed target provider normalizes the entire identifier before
        // splitting it, so there is no separate Responses item identifier.
        assert_eq!(function_call["id"], Value::Null);
        assert_eq!(function_call["call_id"], json!("call_1_fc_1"));
        let tool_result = messages
            .iter()
            .find(|item| item["type"] == json!("function_call_output"))
            .unwrap();
        assert_eq!(tool_result["call_id"], json!("call_1_fc_1"));
    }

    #[test]
    fn different_model_messages_drop_fc_item_ids() {
        let model = Model::new(
            "gpt-5.3-codex",
            "GPT-5.3 Codex",
            "openai-responses",
            "openai",
            "https://api.openai.com/v1",
        );
        let mut assistant = AssistantMessage::new("openai-responses", "openai", "gpt-4o", 1);
        assistant.content = vec![ContentBlock::ToolCall(ToolCall::new(
            "call_1|fc_1",
            "bash",
            Map::new(),
        ))];
        assistant.usage = Usage::zero();
        let context = Context::new(None, vec![Message::assistant(assistant)], None);
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        let function_call = messages
            .iter()
            .find(|item| item["type"] == json!("function_call"))
            .unwrap();
        assert_eq!(function_call["id"], json!(null));
    }

    #[test]
    fn assistant_text_gets_a_generated_message_id() {
        let model = text_model();
        let mut assistant = AssistantMessage::new("openai-responses", "openai", "gpt-4o-mini", 1);
        assistant.content = vec![ContentBlock::Text(TextContent::new("hello"))];
        assistant.usage = Usage::zero();
        let context = Context::new(None, vec![Message::assistant(assistant)], None);
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        assert_eq!(messages[0]["id"], json!("msg_0"));
        assert_eq!(messages[0]["status"], json!("completed"));
        assert_eq!(messages[0]["phase"], json!(null));
        assert_eq!(
            messages[0]["content"],
            json!([{ "type": "output_text", "text": "hello", "annotations": [] }])
        );
    }

    #[test]
    fn generated_message_ids_skip_the_messages_the_typescript_skips() {
        // openai-responses-shared.ts:189 uses `msgIndex`, and the ONLY increment is
        // openai-responses-shared.ts:265 at the end of the loop body. Every `continue`
        // earlier in the body (provider-context user at :142, empty user content at :163,
        // empty assistant output at :222) therefore leaves the index untouched.
        let model = text_model();

        let checkpoint = crate::compaction::ProviderCompactionCheckpoint {
            version: 1,
            provider: "openai".to_string(),
            api: "openai-responses".to_string(),
            model: "gpt-4o-mini".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            endpoint: None,
            items: vec![Map::new()],
            estimated_tokens: 1.0,
        };

        let mut context_user = UserMessage::new(UserContent::Text("ignored".to_string()), 1);
        context_user.provider_context = Some(checkpoint);
        let mut empty_user = UserMessage::new(UserContent::Blocks(Vec::new()), 2);
        empty_user.provider_context = None;
        let mut empty_assistant = AssistantMessage::new("openai-responses", "openai", "gpt-4o-mini", 3);
        empty_assistant.usage = Usage::zero();
        let mut text_assistant = AssistantMessage::new("openai-responses", "openai", "gpt-4o-mini", 4);
        text_assistant.content = vec![ContentBlock::Text(TextContent::new("hello"))];
        text_assistant.usage = Usage::zero();

        let context = Context::new(
            None,
            vec![
                Message::user(context_user),
                Message::user(empty_user),
                Message::assistant(empty_assistant),
                Message::assistant(text_assistant),
            ],
            None,
        );
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        let message_item = messages
            .iter()
            .find(|item| item["type"] == json!("message"))
            .expect("assistant message item");
        assert_eq!(
            message_item["id"], json!("msg_0"),
            "skipped messages must not advance msgIndex (openai-responses-shared.ts:265)"
        );
    }

    #[test]
    fn long_assistant_message_ids_are_hashed() {
        let model = text_model();
        let mut assistant = AssistantMessage::new("openai-responses", "openai", "gpt-4o-mini", 1);
        let long_id = "m".repeat(70);
        let mut text = TextContent::new("hello");
        text.text_signature = Some(encode_text_signature_v1(&long_id, None));
        assistant.content = vec![ContentBlock::Text(text)];
        assistant.usage = Usage::zero();
        let context = Context::new(None, vec![Message::assistant(assistant)], None);
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        assert_eq!(messages[0]["id"], json!(format!("msg_{}", short_hash(&long_id))));
    }

    #[test]
    fn provider_context_items_are_pushed_verbatim() {
        let model = text_model();
        let checkpoint = crate::compaction::ProviderCompactionCheckpoint {
            version: 1,
            provider: "openai".to_string(),
            api: "openai-responses".to_string(),
            model: "gpt-4o-mini".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            endpoint: None,
            items: vec![[("type".to_string(), json!("compaction"))].into_iter().collect()],
            estimated_tokens: 1.0,
        };
        let mut user = UserMessage::new(UserContent::Text("ignored".to_string()), 1);
        user.provider_context = Some(checkpoint);
        let context = Context::new(None, vec![Message::user(user)], None);
        let messages = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap();
        assert_eq!(messages, vec![json!({ "type": "compaction" })]);
    }

    #[test]
    fn mismatched_compaction_checkpoint_is_an_error() {
        let model = text_model();
        let checkpoint = crate::compaction::ProviderCompactionCheckpoint {
            version: 1,
            provider: "anthropic".to_string(),
            api: "anthropic-messages".to_string(),
            model: "claude".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            endpoint: None,
            items: vec![Map::new()],
            estimated_tokens: 1.0,
        };
        let mut user = UserMessage::new(UserContent::Text("ignored".to_string()), 1);
        user.provider_context = Some(checkpoint);
        let context = Context::new(None, vec![Message::user(user)], None);
        let error = convert_responses_messages(&model, &context, &allowed(&["openai"]), None).unwrap_err();
        assert_eq!(
            error,
            "Compaction checkpoint belongs to another model or provider; rebuild context from the session transcript"
        );
    }

    #[test]
    fn convert_responses_tools_defaults_strict_to_false() {
        let tools = vec![Tool {
            name: "bash".to_string(),
            description: "Run a command".to_string(),
            parameters: json!({ "type": "object" }),
        }];
        let converted = convert_responses_tools(&tools, None);
        assert_eq!(
            converted[0],
            json!({
                "type": "function",
                "name": "bash",
                "description": "Run a command",
                "parameters": { "type": "object" },
                "strict": false
            })
        );
        let converted = convert_responses_tools(
            &tools,
            Some(&ConvertResponsesToolsOptions {
                strict: Some(Some(true)),
            }),
        );
        assert_eq!(converted[0]["strict"], json!(true));
        let converted = convert_responses_tools(
            &tools,
            Some(&ConvertResponsesToolsOptions { strict: Some(None) }),
        );
        assert_eq!(converted[0]["strict"], json!(null));
    }

    fn event_stream(events: Vec<Value>) -> ResponsesEventStream {
        Box::pin(stream::iter(events))
    }

    fn empty_output(model: &Model) -> AssistantMessage {
        let mut output = AssistantMessage::new(model.api.clone(), model.provider.clone(), model.id.clone(), 1);
        output.usage = Usage::zero();
        output
    }

    #[tokio::test]
    async fn text_stream_emits_start_delta_end_and_maps_usage() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let events = vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({ "type": "response.output_item.added", "item": { "type": "message", "id": "msg_1", "content": [] } }),
            json!({ "type": "response.content_part.added", "part": { "type": "output_text", "text": "" } }),
            json!({ "type": "response.output_text.delta", "delta": "Hel" }),
            json!({ "type": "response.output_text.delta", "delta": "lo" }),
            json!({
                "type": "response.output_item.done",
                "item": { "type": "message", "id": "msg_1", "phase": "final_answer",
                          "content": [{ "type": "output_text", "text": "Hello" }] }
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 5,
                        "input_tokens_details": { "cached_tokens": 2 },
                        "output_tokens": 3,
                        "total_tokens": 8
                    }
                }
            }),
        ];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap();

        // The provider caller owns the output stream's terminal event/end.
        assert!(!stream.is_done());
        stream.end(Some(output.clone()));
        let emitted: Vec<String> = drain_events(&stream).await;
        assert_eq!(
            emitted,
            vec!["text_start", "text_delta", "text_delta", "text_end"]
        );
        assert_eq!(output.response_id.as_deref(), Some("resp_1"));
        assert_eq!(output.stop_reason, "stop");
        assert_eq!(output.usage.input, 3.0);
        assert_eq!(output.usage.cache_read, 2.0);
        assert_eq!(output.usage.output, 3.0);
        assert_eq!(output.usage.total_tokens, 8.0);
        match &output.content[0] {
            ContentBlock::Text(text) => {
                assert_eq!(text.text, "Hello");
                assert_eq!(
                    text.text_signature.as_deref(),
                    Some("{\"v\":1,\"id\":\"msg_1\",\"phase\":\"final_answer\"}")
                );
            }
            other => panic!("unexpected block {}", other.content_type()),
        }
    }

    #[tokio::test]
    async fn refusal_marker_survives_deltas_or_done_items_and_successful_completion() {
        for streamed_parts in [true, false] {
            let model = text_model();
            let mut output = empty_output(&model);
            let stream = AssistantMessageEventStream::new();
            let mut events = vec![
                json!({"type":"response.output_item.added","item":{"type":"message","id":"refusal","content":[]}}),
            ];
            if streamed_parts {
                events.push(json!({"type":"response.content_part.added","part":{"type":"refusal","refusal":""}}));
                events.push(json!({"type":"response.refusal.delta","delta":"Cannot summarize."}));
            }
            events.push(json!({"type":"response.output_item.done","item":{"type":"message","id":"refusal","content":[{"type":"refusal","refusal":"Cannot summarize."}]}}));
            events.push(json!({"type":"response.completed","response":{"status":"completed"}}));
            process_responses_stream(event_stream(events), &mut output, &stream, &model, None).await.unwrap();
            assert_eq!(output.stop_reason, "stop", "ordinary visible refusal behavior stays unchanged");
            assert_eq!(output.stop_reason_raw.as_deref(), Some("refusal"));
            assert_eq!(output.content[0].as_text().unwrap().text, "Cannot summarize.");
        }
    }

    /// B1: a completed turn that billed output tokens but delivered nothing visible is
    /// labelled for the operator without changing its terminal status or retry behavior.
    #[tokio::test]
    async fn completed_empty_output_with_billed_tokens_is_labelled_and_stays_a_stop() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let events = vec![
            json!({ "type": "response.created", "response": { "id": "resp_empty" } }),
            json!({ "type": "response.output_item.added", "item": { "type": "reasoning", "id": "rs_empty", "summary": [] } }),
            json!({ "type": "response.output_item.done", "item": { "type": "reasoning", "id": "rs_empty", "summary": [], "content": [] } }),
            json!({ "type": "response.output_item.added", "item": { "type": "message", "id": "msg_empty", "content": [] } }),
            json!({ "type": "response.output_item.done", "item": { "type": "message", "id": "msg_empty", "content": [] } }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_empty",
                    "status": "completed",
                    "usage": { "input_tokens": 10, "output_tokens": 71, "total_tokens": 81 }
                }
            }),
        ];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap();

        // The terminal status and error surface are untouched, so no consumer that
        // treats "stop" as success changes behavior.
        assert_eq!(output.stop_reason, "stop");
        assert!(output.error_message.is_none());
        // The billed output remains visible instead of being hidden by the label.
        assert_eq!(output.usage.output, 71.0);
        assert!(has_no_deliverable_content(&output));

        let diagnostics = output.diagnostics.as_ref().expect("labelled terminal");
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].type_, "empty_completion");
        assert!(diagnostics[0].error.is_none(), "the label carries no error text");
        let details = diagnostics[0].details.as_ref().expect("counts");
        assert_eq!(details["reason"], json!("completed_without_deliverable_content"));
        assert_eq!(details["outputTokens"], json!(71.0));
        assert_eq!(details["totalTokens"], json!(81.0));
        // Nothing in the label can trigger the provider/lifecycle retry classifiers.
        assert!(!diagnostics.iter().any(|diagnostic| matches!(
            diagnostic.type_.as_str(),
            "provider_stream_failure" | "agent_lifecycle_failure"
        )));
    }

    /// B1: the label is additive telemetry. The host retry classifiers read only the
    /// `provider_stream_failure` and `agent_lifecycle_failure` diagnostics, so an
    /// `empty_completion` label cannot make a completed turn retryable or permanent.
    #[tokio::test]
    async fn empty_completion_label_cannot_change_retry_classification() {
        use crate::utils::stream_failure::classify_stream_failure;
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let events = vec![
            json!({ "type": "response.created", "response": { "id": "resp_inert" } }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_inert",
                    "status": "completed",
                    "usage": { "input_tokens": 4, "output_tokens": 12, "total_tokens": 16 }
                }
            }),
        ];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap();
        let diagnostics = output.diagnostics.as_ref().expect("labelled terminal");
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].type_, "empty_completion");
        // Neither classifier the retry owner reads can see this label.
        for name in ["provider_stream_failure", "agent_lifecycle_failure"] {
            assert!(
                !diagnostics.iter().any(|diagnostic| diagnostic.type_ == name),
                "an empty completion must not look like {name}"
            );
        }
        // The turn keeps the same terminal status and carries no provider error type, so
        // the same failure classification as an unlabelled completed turn is unchanged.
        assert_eq!(output.stop_reason, "stop");
        assert_eq!(
            classify_stream_failure(output.stop_reason_raw.as_deref(), None),
            classify_stream_failure(None, None),
        );
        assert_eq!(output.stop_reason_raw, None);
        // The transcript keeps the same shape a downstream consumer expects from a
        // completed turn: content and usage present, no error message.
        assert!(output.error_message.is_none());
        assert_eq!(output.usage.output, 12.0);
    }

    /// B1: legitimate terminals keep their existing shape and carry no label.
    #[tokio::test]
    async fn deliverable_refusal_incomplete_and_unbilled_terminals_stay_unlabelled() {
        struct Case {
            name: &'static str,
            events: Vec<Value>,
            expected_stop_reason: &'static str,
        }
        let cases = vec![
            Case {
                name: "visible text",
                events: vec![
                    json!({ "type": "response.output_item.added", "item": { "type": "message", "id": "m1", "content": [] } }),
                    json!({ "type": "response.content_part.added", "part": { "type": "output_text", "text": "" } }),
                    json!({ "type": "response.output_text.delta", "delta": "answer" }),
                    json!({ "type": "response.completed", "response": { "status": "completed", "usage": { "output_tokens": 7 } } }),
                ],
                expected_stop_reason: "stop",
            },
            Case {
                name: "visible reasoning only",
                events: vec![
                    json!({ "type": "response.output_item.added", "item": { "type": "reasoning", "id": "r1", "summary": [] } }),
                    json!({ "type": "response.reasoning_summary_part.added", "part": { "type": "summary_text", "text": "" } }),
                    json!({ "type": "response.reasoning_summary_text.delta", "delta": "hidden chain" }),
                    json!({ "type": "response.completed", "response": { "status": "completed", "usage": { "output_tokens": 5 } } }),
                ],
                expected_stop_reason: "stop",
            },
            Case {
                name: "tool-only",
                events: vec![
                    json!({ "type": "response.output_item.added", "item": { "type": "function_call", "id": "fc1", "call_id": "c1", "name": "bash", "arguments": "{}" } }),
                    json!({ "type": "response.output_item.done", "item": { "type": "function_call", "id": "fc1", "call_id": "c1", "name": "bash", "arguments": "{}" } }),
                    json!({ "type": "response.completed", "response": { "status": "completed", "usage": { "output_tokens": 9 } } }),
                ],
                expected_stop_reason: "toolUse",
            },
            Case {
                name: "refusal",
                events: vec![
                    json!({ "type": "response.output_item.added", "item": { "type": "message", "id": "m2", "content": [] } }),
                    json!({ "type": "response.content_part.added", "part": { "type": "refusal", "refusal": "" } }),
                    json!({ "type": "response.refusal.delta", "delta": "Cannot help." }),
                    json!({ "type": "response.completed", "response": { "status": "completed", "usage": { "output_tokens": 4 } } }),
                ],
                expected_stop_reason: "stop",
            },
            Case {
                name: "empty text block with billed tokens but delivered refusal text",
                events: vec![
                    json!({ "type": "response.output_item.added", "item": { "type": "message", "id": "m4", "content": [] } }),
                    json!({ "type": "response.output_item.done", "item": { "type": "message", "id": "m4", "content": [{ "type": "refusal", "refusal": "Nope." }] } }),
                    json!({ "type": "response.completed", "response": { "status": "completed", "usage": { "output_tokens": 6 } } }),
                ],
                expected_stop_reason: "stop",
            },
            Case {
                name: "unbilled empty completion",
                events: vec![
                    json!({ "type": "response.output_item.added", "item": { "type": "message", "id": "m3", "content": [] } }),
                    json!({ "type": "response.completed", "response": { "status": "completed" } }),
                ],
                expected_stop_reason: "stop",
            },
        ];
        for case in cases {
            let model = text_model();
            let mut output = empty_output(&model);
            let stream = AssistantMessageEventStream::new();
            process_responses_stream(event_stream(case.events), &mut output, &stream, &model, None)
                .await
                .unwrap();
            assert_eq!(output.stop_reason, case.expected_stop_reason, "case {}", case.name);
            assert!(
                output.diagnostics.is_none(),
                "case {} must not be labelled as an empty completion",
                case.name
            );
        }
    }

    /// B1: an incomplete or errored terminal is never relabelled, because its own
    /// status already explains why nothing was delivered.
    #[tokio::test]
    async fn incomplete_and_errored_terminals_are_never_labelled_empty() {
        for response in [
            json!({ "status": "incomplete", "incomplete_details": { "reason": "max_output_tokens" }, "usage": { "output_tokens": 12 } }),
            json!({ "status": "failed", "error": { "code": "server_error", "message": "boom" }, "usage": { "output_tokens": 3 } }),
        ]
        .into_iter()
        .enumerate()
        {
            let (index, response) = response;
            let model = text_model();
            let mut output = empty_output(&model);
            let stream = AssistantMessageEventStream::new();
            let events = if index == 0 {
                vec![
                    json!({ "type": "response.completed", "response": response }),
                ]
            } else {
                vec![
                    json!({ "type": "response.failed", "response": response }),
                ]
            };
            let result = process_responses_stream(event_stream(events), &mut output, &stream, &model, None).await;
            if index == 0 {
                result.unwrap();
                assert_eq!(output.stop_reason, "length");
                assert!(output.diagnostics.is_none(), "an incomplete terminal keeps its own status");
            } else {
                assert!(result.is_err(), "a failed terminal stays a failure");
            }
        }
    }

    async fn drain_events(stream: &AssistantMessageEventStream) -> Vec<String> {
        let mut types = Vec::new();
        while let Some(event) = stream.next().await {
            types.push(event.event_type().to_string());
        }
        types
    }

    #[tokio::test]
    async fn tool_call_blocks_persist_parsed_arguments_without_partial_json() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let arguments = "{\"path\":\"README.md\",\"content\":\"updated\"}";
        let events = vec![
            json!({
                "type": "response.output_item.added",
                "item": { "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "edit", "arguments": "" }
            }),
            json!({ "type": "response.function_call_arguments.delta", "delta": "{\"path\":\"README.md\"" }),
            json!({ "type": "response.function_call_arguments.delta", "delta": ",\"content\":\"updated\"}" }),
            json!({ "type": "response.function_call_arguments.done", "arguments": arguments }),
            json!({
                "type": "response.output_item.done",
                "item": { "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "edit", "arguments": arguments }
            }),
            json!({ "type": "response.completed", "response": { "status": "completed" } }),
        ];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap();

        assert_eq!(output.content.len(), 1);
        match &output.content[0] {
            ContentBlock::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "call_test|fc_test");
                assert_eq!(tool_call.arguments["path"], json!("README.md"));
                assert_eq!(tool_call.arguments["content"], json!("updated"));
            }
            other => panic!("unexpected block {}", other.content_type()),
        }
        // A tool call with stopReason "stop" becomes "toolUse".
        assert_eq!(output.stop_reason, "toolUse");
    }

    #[tokio::test]
    async fn reasoning_items_keep_summary_text_and_signature() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let item = json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{ "type": "summary_text", "text": "thought\n\nmore" }],
            "encrypted_content": "abc"
        });
        let events = vec![
            json!({ "type": "response.output_item.added", "item": { "type": "reasoning", "id": "rs_1", "summary": [] } }),
            json!({ "type": "response.reasoning_summary_part.added", "part": { "type": "summary_text", "text": "" } }),
            json!({ "type": "response.reasoning_summary_text.delta", "delta": "thought" }),
            json!({ "type": "response.reasoning_summary_part.done" }),
            json!({ "type": "response.reasoning_summary_text.delta", "delta": "more" }),
            json!({ "type": "response.output_item.done", "item": item }),
        ];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap();

        match &output.content[0] {
            ContentBlock::Thinking(thinking) => {
                assert_eq!(thinking.thinking, "thought\n\nmore");
                assert_eq!(thinking.thinking_signature.as_deref(), Some(&stringify(&item)[..]));
            }
            other => panic!("unexpected block {}", other.content_type()),
        }
        // The shared processor leaves the caller's output stream open.
        assert!(!stream.is_done());
        stream.end(Some(output.clone()));
        assert_eq!(
            drain_events(&stream).await,
            vec![
                "thinking_start",
                "thinking_delta",
                "thinking_delta",
                "thinking_delta",
                "thinking_end"
            ]
        );
    }

    #[tokio::test]
    async fn usage_observation_preserves_raw_zeros_and_absent_fields() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let seen: Arc<std::sync::Mutex<Vec<ProviderUsageObservation>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = seen.clone();
        let options = OpenAIResponsesStreamOptions {
            on_usage_observation: Some(Arc::new(move |observation, _model| {
                recorder.lock().unwrap().push(observation);
                Box::pin(async {})
            })),
            ..Default::default()
        };
        let events = vec![json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 0,
                    "input_tokens_details": { "cached_tokens": 0 },
                    "output_tokens": 0,
                    "output_tokens_details": { "reasoning_tokens": 0 },
                    "total_tokens": 0
                }
            }
        })];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, Some(&options))
            .await
            .unwrap();
        // The observer runs in a detached task (the TS does not await it).
        tokio::task::yield_now().await;
        let observations = seen.lock().unwrap().clone();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].input_tokens, Some(Some(0.0)));
        assert_eq!(observations[0].cached_input_tokens, Some(Some(0.0)));
        assert_eq!(observations[0].reasoning_tokens, Some(Some(0.0)));
        assert_eq!(observations[0].cached_input_included_in_input, Some(Some(true)));
        assert_eq!(observations[0].reasoning_included_in_output, Some(Some(true)));

        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let seen: Arc<std::sync::Mutex<Vec<ProviderUsageObservation>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = seen.clone();
        let options = OpenAIResponsesStreamOptions {
            on_usage_observation: Some(Arc::new(move |observation, _model| {
                recorder.lock().unwrap().push(observation);
                Box::pin(async {})
            })),
            ..Default::default()
        };
        let events = vec![json!({
            "type": "response.completed",
            "response": { "status": "completed", "usage": { "input_tokens": 9, "output_tokens": 4 } }
        })];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, Some(&options))
            .await
            .unwrap();
        tokio::task::yield_now().await;
        let observations = seen.lock().unwrap().clone();
        assert_eq!(observations[0].input_tokens, Some(Some(9.0)));
        assert_eq!(observations[0].cached_input_tokens, Some(None));
        assert_eq!(observations[0].reasoning_tokens, Some(None));
        assert_eq!(observations[0].total_tokens, Some(None));
    }

    #[tokio::test]
    async fn map_stop_reason_matches_typescript() {
        let model = text_model();
        for (status, expected) in [
            (Some("completed"), "stop"),
            (Some("incomplete"), "length"),
            (Some("failed"), "error"),
            (Some("cancelled"), "error"),
            (Some("in_progress"), "stop"),
            (Some("queued"), "stop"),
            (None, "stop"),
        ] {
            let mut output = empty_output(&model);
            let stream = AssistantMessageEventStream::new();
            let events = vec![json!({
                "type": "response.completed",
                "response": match status {
                    Some(status) => json!({ "status": status }),
                    None => json!({}),
                }
            })];
            process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
                .await
                .unwrap();
            assert_eq!(output.stop_reason, expected, "status {:?}", status);
        }

        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let events = vec![json!({ "type": "response.completed", "response": { "status": "unknown-status" } })];
        let error = process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap_err();
        assert_eq!(error.message(), "Unhandled stop reason: unknown-status");
    }

    #[tokio::test]
    async fn incomplete_terminals_preserve_exact_reason_and_usage() {
        let model = text_model();
        for kind in ["response.completed", "response.incomplete"] {
            for reason in [Some("max_output_tokens"), Some("content_filter"), Some("unknown"), None] {
                let mut output = empty_output(&model);
                let stream = AssistantMessageEventStream::new();
                let events = vec![json!({
                    "type":kind, "response": { "status":"incomplete", "id":"bounded-summary",
                        "incomplete_details": { "reason":reason },
                        "usage": { "input_tokens":10, "output_tokens":13107, "total_tokens":13117,
                            "output_tokens_details": { "reasoning_tokens":4834 } } }
                })];
                process_responses_stream(event_stream(events), &mut output, &stream, &model, None).await.unwrap();
                assert_eq!(output.stop_reason, "length");
                assert_eq!(output.stop_reason_raw.as_deref(), reason);
                assert_eq!(output.usage.output, 13107.0);
                assert_eq!(output.response_id.as_deref(), Some("bounded-summary"));
            }
        }
    }

    #[tokio::test]
    async fn incomplete_terminal_without_status_is_not_success_and_cannot_erase_refusal() {
        let model = text_model();
        for raw in [None, Some("refusal")] {
            let mut output = empty_output(&model);
            output.stop_reason_raw = raw.map(str::to_string);
            let stream = AssistantMessageEventStream::new();
            let events = vec![json!({ "type":"response.incomplete",
                "response": { "incomplete_details": { "reason":"max_output_tokens" } } })];
            process_responses_stream(event_stream(events), &mut output, &stream, &model, None).await.unwrap();
            assert_eq!(output.stop_reason, "length");
            assert_eq!(output.stop_reason_raw.as_deref(), Some(raw.unwrap_or("max_output_tokens")));
        }
    }

    #[tokio::test]
    async fn incomplete_terminal_with_provider_error_cannot_be_retried_as_length() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let events = vec![json!({ "type":"response.incomplete", "response": {
            "incomplete_details": { "reason":"max_output_tokens" },
            "error": { "code":"content_filter", "message":"filtered" }
        } })];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, None).await.unwrap();
        assert_eq!(output.stop_reason, "error");
        assert_ne!(output.stop_reason_raw.as_deref(), Some("max_output_tokens"));
    }

    #[tokio::test]
    async fn error_events_become_stream_failures() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let events = vec![json!({ "type": "error", "code": "server_error", "message": "boom" })];
        let error = process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap_err();
        assert_eq!(error.message(), "Error Code server_error: boom");
        match error {
            ResponsesStreamError::StreamFailure(failure) => {
                assert_eq!(failure.info.kind, "server_error");
                assert_eq!(failure.info.provider_error_type.as_deref(), Some("server_error"));
            }
            other => panic!("unexpected error {}", other.message()),
        }
    }

    #[tokio::test]
    async fn failed_responses_map_error_details() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let events = vec![json!({
            "type": "response.failed",
            "response": { "error": { "code": "rate_limit_exceeded", "message": "slow down" } }
        })];
        let error = process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap_err();
        assert_eq!(error.message(), "rate_limit_exceeded: slow down");
        match error {
            ResponsesStreamError::StreamFailure(failure) => assert_eq!(failure.info.kind, "rate_limit"),
            other => panic!("unexpected error {}", other.message()),
        }

        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let events = vec![json!({
            "type": "response.failed",
            "response": { "incomplete_details": { "reason": "max_output_tokens" } }
        })];
        let error = process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap_err();
        assert_eq!(error.message(), "incomplete: max_output_tokens");

        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let events = vec![json!({ "type": "response.failed", "response": {} })];
        let error = process_responses_stream(event_stream(events), &mut output, &stream, &model, None)
            .await
            .unwrap_err();
        assert_eq!(error.message(), "Unknown error (no error details in response)");
    }

    #[tokio::test]
    async fn on_output_item_done_runs_before_finalizing_the_block() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let seen: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = seen.clone();
        let options = OpenAIResponsesStreamOptions {
            on_output_item_done: Some(Arc::new(move |item| {
                recorder
                    .lock()
                    .unwrap()
                    .push(item.get("id").and_then(Value::as_str).unwrap_or_default().to_string());
                Box::pin(async {})
            })),
            ..Default::default()
        };
        let events = vec![
            json!({ "type": "response.output_item.added", "item": { "type": "message", "id": "msg_1", "content": [] } }),
            json!({ "type": "response.content_part.added", "part": { "type": "output_text", "text": "" } }),
            json!({
                "type": "response.output_item.done",
                "item": { "type": "message", "id": "msg_1", "content": [{ "type": "output_text", "text": "Hello" }] }
            }),
        ];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, Some(&options))
            .await
            .unwrap();
        assert_eq!(seen.lock().unwrap().clone(), vec!["msg_1".to_string()]);
    }

    #[tokio::test]
    async fn service_tier_pricing_uses_the_resolver() {
        let model = text_model();
        let mut output = empty_output(&model);
        let stream = AssistantMessageEventStream::new();
        let applied: Arc<std::sync::Mutex<Vec<Option<String>>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = applied.clone();
        let options = OpenAIResponsesStreamOptions {
            service_tier: Some("flex".to_string()),
            resolve_service_tier: Some(Arc::new(|response_tier, request_tier| {
                response_tier.or(request_tier).map(str::to_string)
            })),
            apply_service_tier_pricing: Some(Arc::new(move |_usage, service_tier| {
                recorder.lock().unwrap().push(service_tier.map(str::to_string));
            })),
            ..Default::default()
        };
        let events = vec![json!({
            "type": "response.completed",
            "response": { "status": "completed", "service_tier": "default" }
        })];
        process_responses_stream(event_stream(events), &mut output, &stream, &model, Some(&options))
            .await
            .unwrap();
        assert_eq!(applied.lock().unwrap().clone(), vec![Some("default".to_string())]);
    }
}
