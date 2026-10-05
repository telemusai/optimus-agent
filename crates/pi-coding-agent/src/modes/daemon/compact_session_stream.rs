//! Port of packages/coding-agent/src/modes/daemon/compact-session-stream.ts

use std::collections::HashMap;

use pi_ai::types::{AssistantMessage, ContentBlock, ToolCall};
use pi_ai::utils::json_parse::parse_streaming_json;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::daemon_protocol::DaemonEventMeta;

/// The streaming event with its redundant partial message omitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CompactAssistantMessageEvent {
    #[serde(rename = "start")]
    Start {},
    #[serde(rename = "text_start")]
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "text_end")]
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        content: String,
    },
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        content: String,
    },
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "toolCall")]
        tool_call: ToolCall,
    },
    #[serde(rename = "done")]
    Done {
        /// `Extract<StopReason, "stop" | "length" | "toolUse">`
        reason: String,
        message: AssistantMessage,
    },
    #[serde(rename = "error")]
    Error {
        /// `Extract<StopReason, "aborted" | "error">`
        reason: String,
        error: AssistantMessage,
    },
}

/// A compact assistant delta: `assistant_stream_delta` on the worker wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactAssistantDelta {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(rename = "activeSessionId")]
    pub active_session_id: String,
    #[serde(rename = "assistantMessageEvent")]
    pub assistant_message_event: CompactAssistantMessageEvent,
    #[serde(rename = "contentStart", skip_serializing_if = "Option::is_none", default)]
    pub content_start: Option<ContentBlock>,
    #[serde(rename = "toolCallArguments", skip_serializing_if = "Option::is_none", default)]
    pub tool_call_arguments: Option<serde_json::Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub meta: Option<DaemonEventMeta>,
}

/// `createCompactAssistantDelta`: drop the nested partial and forward the
/// message's content slice the client needs to start a block.
pub fn create_compact_assistant_delta(
    outbound: &Value,
) -> Result<Option<CompactAssistantDelta>, String> {
    let Some(candidate) = outbound.as_object() else {
        return Ok(None);
    };
    if candidate.get("type").and_then(Value::as_str) != Some("session_event") {
        return Ok(None);
    }
    let Some(event) = candidate.get("event").and_then(Value::as_object) else {
        return Ok(None);
    };
    if event.get("type").and_then(Value::as_str) != Some("message_update") {
        return Ok(None);
    }
    let Some(message) = event.get("message") else {
        return Ok(None);
    };
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return Ok(None);
    }
    let assistant_message: AssistantMessage =
        AssistantMessage::deserialize(message).map_err(|error| error.to_string())?;
    let event_value = event
        .get("assistantMessageEvent")
        .ok_or_else(|| "message_update without assistantMessageEvent".to_string())?;
    // `CompactAssistantMessageEvent` has no `partial` field and no
    // `deny_unknown_fields`, so the nested partial is ignored without
    // rewriting the object first.
    let compact_event: CompactAssistantMessageEvent =
        CompactAssistantMessageEvent::deserialize(event_value).map_err(|error| error.to_string())?;

    let content_start = compact_content_start(&assistant_message, &compact_event);
    let tool_call_arguments = compact_tool_call_arguments(&assistant_message, &compact_event);
    Ok(Some(CompactAssistantDelta {
        type_: "assistant_stream_delta".to_string(),
        active_session_id: candidate
            .get("activeSessionId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        assistant_message_event: compact_event,
        content_start,
        tool_call_arguments,
        meta: candidate
            .get("meta")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok()),
    }))
}

fn compact_tool_call_arguments(
    message: &AssistantMessage,
    event: &CompactAssistantMessageEvent,
) -> Option<serde_json::Map<String, Value>> {
    let CompactAssistantMessageEvent::ToolCallDelta { content_index, .. } = event else {
        return None;
    };
    match message.content.get(*content_index) {
        Some(ContentBlock::ToolCall(tool_call)) => Some(tool_call.arguments.clone()),
        _ => None,
    }
}

fn compact_content_start(message: &AssistantMessage, event: &CompactAssistantMessageEvent) -> Option<ContentBlock> {
    let content_index = match event {
        CompactAssistantMessageEvent::TextStart { content_index, .. }
        | CompactAssistantMessageEvent::ThinkingStart { content_index, .. }
        | CompactAssistantMessageEvent::ToolCallStart { content_index, .. } => *content_index,
        _ => return None,
    };
    let content = message.content.get(content_index)?;
    match (event, content) {
        (CompactAssistantMessageEvent::TextStart { .. }, ContentBlock::Text(text)) => {
            let mut cleared = text.clone();
            cleared.text = String::new();
            Some(ContentBlock::Text(cleared))
        }
        (CompactAssistantMessageEvent::ThinkingStart { .. }, ContentBlock::Thinking(thinking)) => {
            let mut cleared = thinking.clone();
            cleared.thinking = String::new();
            Some(ContentBlock::Thinking(cleared))
        }
        (CompactAssistantMessageEvent::ToolCallStart { .. }, ContentBlock::ToolCall(tool_call)) => {
            let mut cleared = tool_call.clone();
            cleared.arguments = serde_json::Map::new();
            Some(ContentBlock::ToolCall(cleared))
        }
        _ => None,
    }
}

/// Rebuilds the full `message_update` events a compact stream omits.
pub struct CompactAssistantStreamReconstructor {
    partial_messages: HashMap<String, AssistantMessage>,
    tool_call_json: HashMap<String, String>,
}

impl Default for CompactAssistantStreamReconstructor {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactAssistantStreamReconstructor {
    pub fn new() -> Self {
        Self {
            partial_messages: HashMap::new(),
            tool_call_json: HashMap::new(),
        }
    }

    pub fn seed(&mut self, active_session_id: &str, message: AssistantMessage) {
        self.partial_messages
            .insert(active_session_id.to_string(), message);
    }

    pub fn observe(&mut self, outbound: &Value) {
        let Some(candidate) = outbound.as_object() else {
            return;
        };
        let message_type = candidate.get("type").and_then(Value::as_str).unwrap_or_default();
        if message_type != "session_event" {
            if matches!(
                message_type,
                "session_replaced" | "session_resynced" | "session_closed"
            ) {
                let active_session_id = candidate
                    .get("activeSessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.clear(&active_session_id);
            }
            return;
        }
        let Some(event) = candidate.get("event").and_then(Value::as_object) else {
            return;
        };
        let active_session_id = candidate
            .get("activeSessionId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                let Some(message) = event.get("message") else {
                    return;
                };
                if message.get("role").and_then(Value::as_str) == Some("assistant") {
                    if let Ok(message) = serde_json::from_value::<AssistantMessage>(message.clone()) {
                        self.partial_messages.insert(active_session_id, message);
                    }
                }
            }
            Some("message_end") => self.clear(&active_session_id),
            _ => {}
        }
    }

    pub fn reconstruct(&mut self, delta: &CompactAssistantDelta) -> Option<Value> {
        let active_session_id = delta.active_session_id.clone();
        let partial = self.partial_messages.get_mut(&active_session_id)?;
        let event = &delta.assistant_message_event;
        match event {
            CompactAssistantMessageEvent::TextStart { content_index, .. } => {
                let block = delta
                    .content_start
                    .clone()
                    .unwrap_or_else(|| ContentBlock::Text(default_text_content()));
                set_content(partial, *content_index, block);
            }
            CompactAssistantMessageEvent::TextDelta { content_index, delta: text_delta, .. } => {
                match partial.content.get_mut(*content_index) {
                    Some(ContentBlock::Text(content)) => content.text.push_str(text_delta),
                    _ => return None,
                }
            }
            CompactAssistantMessageEvent::TextEnd { content_index, content, .. } => {
                match partial.content.get_mut(*content_index) {
                    Some(ContentBlock::Text(block)) => block.text = content.clone(),
                    _ => return None,
                }
            }
            CompactAssistantMessageEvent::ThinkingStart { content_index, .. } => {
                let block = delta
                    .content_start
                    .clone()
                    .unwrap_or_else(|| ContentBlock::Thinking(default_thinking_content()));
                set_content(partial, *content_index, block);
            }
            CompactAssistantMessageEvent::ThinkingDelta { content_index, delta: text_delta, .. } => {
                match partial.content.get_mut(*content_index) {
                    Some(ContentBlock::Thinking(content)) => content.thinking.push_str(text_delta),
                    _ => return None,
                }
            }
            CompactAssistantMessageEvent::ThinkingEnd { content_index, content, .. } => {
                match partial.content.get_mut(*content_index) {
                    Some(ContentBlock::Thinking(block)) => block.thinking = content.clone(),
                    _ => return None,
                }
            }
            CompactAssistantMessageEvent::ToolCallStart { content_index, .. } => {
                let Some(ContentBlock::ToolCall(tool_call)) = delta.content_start.clone() else {
                    return None;
                };
                set_content(partial, *content_index, ContentBlock::ToolCall(tool_call));
                self.tool_call_json
                    .insert(Self::tool_call_key(&active_session_id, *content_index), String::new());
            }
            CompactAssistantMessageEvent::ToolCallDelta { content_index, delta: text_delta, .. } => {
                let arguments = match delta.tool_call_arguments.clone() {
                    Some(arguments) => arguments,
                    None => {
                        let key = Self::tool_call_key(&active_session_id, *content_index);
                        let partial_json = format!(
                            "{}{}",
                            self.tool_call_json.get(&key).cloned().unwrap_or_default(),
                            text_delta
                        );
                        self.tool_call_json.insert(key, partial_json.clone());
                        match parse_streaming_json(Some(&partial_json)) {
                            Value::Object(map) => map,
                            _ => serde_json::Map::new(),
                        }
                    }
                };
                match partial.content.get_mut(*content_index) {
                    Some(ContentBlock::ToolCall(content)) => content.arguments = arguments,
                    _ => return None,
                }
            }
            CompactAssistantMessageEvent::ToolCallEnd { content_index, tool_call, .. } => {
                set_content(partial, *content_index, ContentBlock::ToolCall(tool_call.clone()));
                self.tool_call_json
                    .remove(&Self::tool_call_key(&active_session_id, *content_index));
            }
            CompactAssistantMessageEvent::Start { .. }
            | CompactAssistantMessageEvent::Done { .. }
            | CompactAssistantMessageEvent::Error { .. } => return None,
        }
        let message = partial.clone();
        let event_value = serde_json::to_value(event).ok()?;
        let mut event_object = event_value.as_object().cloned().unwrap_or_default();
        let mut outbound = serde_json::json!({
            "type": "session_event",
            "activeSessionId": delta.active_session_id,
            "event": {
                "type": "message_update",
                "message": serde_json::to_value(&message).ok()?,
                "assistantMessageEvent": Value::Object(event_object),
            },
        });
        if let Some(meta) = &delta.meta {
            outbound["meta"] = serde_json::to_value(meta).ok()?;
        }
        Some(outbound)
    }

    pub fn clear(&mut self, active_session_id: &str) {
        self.partial_messages.remove(active_session_id);
        let prefix = format!("{active_session_id}:");
        self.tool_call_json.retain(|key, _| !key.starts_with(&prefix));
    }

    fn tool_call_key(active_session_id: &str, content_index: usize) -> String {
        format!("{active_session_id}:{content_index}")
    }
}

fn set_content(message: &mut AssistantMessage, index: usize, block: ContentBlock) {
    while message.content.len() <= index {
        message.content.push(ContentBlock::Text(default_text_content()));
    }
    message.content[index] = block;
}

fn default_text_content() -> pi_ai::types::TextContent {
    pi_ai::types::TextContent {
        type_: pi_ai::types::TEXT_CONTENT_TYPE.to_string(),
        text: String::new(),
        text_signature: None,
    }
}

fn default_thinking_content() -> pi_ai::types::ThinkingContent {
    pi_ai::types::ThinkingContent {
        type_: pi_ai::types::THINKING_CONTENT_TYPE.to_string(),
        thinking: String::new(),
        thinking_signature: None,
        redacted: None,
    }
}

pub fn is_compact_assistant_delta(value: &Value) -> bool {
    let Some(candidate) = value.as_object() else {
        return false;
    };
    candidate.get("type").and_then(Value::as_str) == Some("assistant_stream_delta")
        && candidate.get("activeSessionId").and_then(Value::as_str).is_some()
        && candidate
            .get("assistantMessageEvent")
            .is_some_and(Value::is_object)
}

/// A `toolCall` content block helper for callers that build deltas directly.
pub fn tool_call_content(tool_call: ToolCall) -> ContentBlock {
    ContentBlock::ToolCall(tool_call)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_ai::types::{Usage, UsageCost};

    fn assistant_message(text: &str) -> Value {
        serde_json::json!({
            "role": "assistant",
            "content": [{ "type": "text", "text": text }],
            "api": "openai-completions",
            "provider": "openai",
            "model": "gpt",
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 }
            },
            "stopReason": "stop",
            "timestamp": 0
        })
    }

    fn usage() -> Usage {
        Usage {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total_tokens: 0.0,
            cost: UsageCost::default(),
        }
    }

    fn partial_with_text(text: &str) -> AssistantMessage {
        AssistantMessage {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text(pi_ai::types::TextContent {
                type_: pi_ai::types::TEXT_CONTENT_TYPE.to_string(),
                text: text.to_string(),
                text_signature: None,
            })],
            api: "openai-completions".to_string(),
            provider: "openai".to_string(),
            model: "gpt".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: usage(),
            stop_reason: "stop".to_string(),
            stop_reason_raw: None,
            error_message: None,
            timestamp: 0,
        }
    }

    #[test]
    fn creates_a_compact_delta_from_a_message_update() {
        let outbound = serde_json::json!({
            "type": "session_event",
            "activeSessionId": "a",
            "event": {
                "type": "message_update",
                "message": assistant_message("hi"),
                "assistantMessageEvent": {
                    "type": "text_delta",
                    "contentIndex": 0,
                    "delta": "hi",
                    "partial": assistant_message("hi")
                }
            }
        });
        let delta = create_compact_assistant_delta(&outbound)
            .expect("parses")
            .expect("delta");
        assert_eq!(delta.type_, "assistant_stream_delta");
        assert_eq!(delta.active_session_id, "a");
        assert!(matches!(
            delta.assistant_message_event,
            CompactAssistantMessageEvent::TextDelta { .. }
        ));
        let wire = serde_json::to_value(&delta).expect("serializes");
        assert!(is_compact_assistant_delta(&wire));
        assert!(wire["assistantMessageEvent"].get("partial").is_none());
        let decoded: CompactAssistantDelta = serde_json::from_value(wire).expect("compact wire round trip");
        assert_eq!(decoded, delta);
    }

    #[test]
    fn content_start_carries_an_empty_block() {
        let outbound = serde_json::json!({
            "type": "session_event",
            "activeSessionId": "a",
            "event": {
                "type": "message_update",
                "message": assistant_message("hi"),
                "assistantMessageEvent": { "type": "text_start", "contentIndex": 0 }
            }
        });
        let delta = create_compact_assistant_delta(&outbound)
            .expect("parses")
            .expect("delta");
        match delta.content_start {
            Some(ContentBlock::Text(text)) => assert!(text.text.is_empty()),
            other => panic!("unexpected content start: {other:?}"),
        }
    }

    #[test]
    fn non_message_update_events_are_ignored() {
        assert!(create_compact_assistant_delta(&serde_json::json!({
            "type": "session_event",
            "event": { "type": "message_start" }
        }))
        .expect("parses")
        .is_none());
        assert!(create_compact_assistant_delta(&serde_json::json!({ "type": "session_replaced" }))
            .expect("parses")
            .is_none());
    }

    #[test]
    fn reconstruction_rebuilds_text_updates() {
        let mut reconstructor = CompactAssistantStreamReconstructor::new();
        let message = partial_with_text("");
        reconstructor.seed("a", message);
        let delta = CompactAssistantDelta {
            type_: "assistant_stream_delta".to_string(),
            active_session_id: "a".to_string(),
            assistant_message_event: CompactAssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "hello".to_string(),
            },
            content_start: None,
            tool_call_arguments: None,
            meta: None,
        };
        let rebuilt = reconstructor.reconstruct(&delta).expect("rebuilt");
        assert_eq!(rebuilt["event"]["type"], "message_update");
        assert_eq!(rebuilt["event"]["message"]["content"][0]["text"], "hello");
    }

    #[test]
    fn reconstruction_returns_none_without_a_partial() {
        let mut reconstructor = CompactAssistantStreamReconstructor::new();
        let delta = CompactAssistantDelta {
            type_: "assistant_stream_delta".to_string(),
            active_session_id: "missing".to_string(),
            assistant_message_event: CompactAssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "x".to_string(),
            },
            content_start: None,
            tool_call_arguments: None,
            meta: None,
        };
        assert!(reconstructor.reconstruct(&delta).is_none());
    }

    #[test]
    fn observe_clears_on_message_end_and_session_replacement() {
        let mut reconstructor = CompactAssistantStreamReconstructor::new();
        reconstructor.seed("a", partial_with_text("x"));
        reconstructor.observe(&serde_json::json!({
            "type": "session_event",
            "activeSessionId": "a",
            "event": { "type": "message_end", "message": assistant_message("x") }
        }));
        let delta = CompactAssistantDelta {
            type_: "assistant_stream_delta".to_string(),
            active_session_id: "a".to_string(),
            assistant_message_event: CompactAssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "x".to_string(),
            },
            content_start: None,
            tool_call_arguments: None,
            meta: None,
        };
        assert!(reconstructor.reconstruct(&delta).is_none());

        reconstructor.seed("a", partial_with_text("x"));
        reconstructor.observe(&serde_json::json!({
            "type": "session_replaced",
            "activeSessionId": "a"
        }));
        assert!(reconstructor.reconstruct(&delta).is_none());
    }

    #[test]
    fn tool_call_deltas_accumulate_partial_json() {
        let mut reconstructor = CompactAssistantStreamReconstructor::new();
        let mut message = partial_with_text("");
        message.content = vec![ContentBlock::ToolCall(ToolCall {
            type_: pi_ai::types::TOOL_CALL_TYPE.to_string(),
            id: "t1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::Map::new(),
            thought_signature: None,
        })];
        reconstructor.seed("a", message);
        let delta = CompactAssistantDelta {
            type_: "assistant_stream_delta".to_string(),
            active_session_id: "a".to_string(),
            assistant_message_event: CompactAssistantMessageEvent::ToolCallDelta {
                content_index: 0,
                delta: "{\"cmd\":\"ls\"}".to_string(),
            },
            content_start: None,
            tool_call_arguments: None,
            meta: None,
        };
        let rebuilt = reconstructor.reconstruct(&delta).expect("rebuilt");
        assert_eq!(rebuilt["event"]["message"]["content"][0]["arguments"]["cmd"], "ls");
    }
}
