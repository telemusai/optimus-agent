//! Port of packages/agent/src/proxy.ts
//!
//! Requests go through a proxy server instead of calling LLM providers directly.
//! `fetch` becomes `reqwest`; `ReadableStreamDefaultReader` becomes the reqwest
//! byte stream; `AbortSignal` becomes a `CancellationToken`.

use futures::StreamExt;
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ContentBlock, Context, Model, SimpleStreamOptions,
    TextContent, ThinkingContent, ToolCall, Usage,
};
use pi_ai::utils::event_stream::EventStream;
use pi_ai::utils::json_parse::parse_streaming_json;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

/// `class ProxyMessageEventStream extends EventStream<AssistantMessageEvent, AssistantMessage>`.
#[derive(Clone)]
pub struct ProxyMessageEventStream(EventStream<AssistantMessageEvent, AssistantMessage>);

impl Default for ProxyMessageEventStream {
    fn default() -> Self {
        Self::new()
    }
}

impl ProxyMessageEventStream {
    pub fn new() -> Self {
        Self(EventStream::new(
            Box::new(|event: &AssistantMessageEvent| {
                matches!(
                    event,
                    AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
                )
            }),
            Box::new(|event: &AssistantMessageEvent| match event {
                AssistantMessageEvent::Done { message, .. } => message.clone(),
                AssistantMessageEvent::Error { error, .. } => error.clone(),
                _ => panic!("Unexpected event type"),
            }),
        ))
    }

    pub fn push(&self, event: AssistantMessageEvent) {
        self.0.push(event);
    }

    pub fn end(&self, result: Option<AssistantMessage>) {
        self.0.end(result);
    }

    pub async fn next(&self) -> Option<AssistantMessageEvent> {
        self.0.next().await
    }

    pub async fn result(&self) -> AssistantMessage {
        self.0.result().await
    }

    pub fn is_done(&self) -> bool {
        self.0.is_done()
    }

    pub fn into_stream(self) -> impl futures::Stream<Item = AssistantMessageEvent> {
        self.0.into_stream()
    }
}

/// `type ProxyAssistantMessageEvent` - the event protocol the proxy server sends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProxyAssistantMessageEvent {
    #[serde(rename = "start")]
    Start,
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
        #[serde(rename = "contentSignature", default, skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
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
        #[serde(rename = "contentSignature", default, skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
    },
    #[serde(rename = "toolcall_start")]
    ToolcallStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
    },
    #[serde(rename = "toolcall_delta")]
    ToolcallDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "toolcall_end")]
    ToolcallEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    #[serde(rename = "done")]
    Done {
        /// `Extract<StopReason, "stop" | "length" | "toolUse">`
        reason: String,
        usage: Usage,
    },
    #[serde(rename = "error")]
    Error {
        /// `Extract<StopReason, "aborted" | "error">`
        reason: String,
        #[serde(rename = "errorMessage", default, skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        usage: Usage,
    },
}

/// `type ProxySerializableStreamOptions`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProxySerializableStreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(rename = "maxTokens", default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(rename = "cacheRetention", default, skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<String>,
    #[serde(rename = "sessionId", default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<serde_json::Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<String>,
    #[serde(rename = "thinkingBudgets", default, skip_serializing_if = "Option::is_none")]
    pub thinking_budgets: Option<pi_ai::types::ThinkingBudgets>,
    #[serde(rename = "serviceTier", default, skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_service_tier")]
    pub service_tier: pi_ai::types::ServiceTier,
}

fn deserialize_service_tier<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<pi_ai::types::ServiceTier, D::Error> {
    Option::<String>::deserialize(deserializer).map(Some)
}

/// `interface ProxyStreamOptions extends ProxySerializableStreamOptions`.
#[derive(Debug, Clone, Default)]
pub struct ProxyStreamOptions {
    pub serializable: ProxySerializableStreamOptions,
    /// `signal?: AbortSignal`
    pub signal: Option<CancellationToken>,
    pub auth_token: String,
    pub proxy_url: String,
}

/// `buildProxyRequestOptions(options)`.
fn build_proxy_request_options(options: &ProxyStreamOptions) -> ProxySerializableStreamOptions {
    ProxySerializableStreamOptions {
        temperature: options.serializable.temperature,
        max_tokens: options.serializable.max_tokens,
        reasoning: options.serializable.reasoning.clone(),
        cache_retention: options.serializable.cache_retention.clone(),
        session_id: options.serializable.session_id.clone(),
        headers: options.serializable.headers.clone(),
        metadata: options.serializable.metadata.clone(),
        transport: options.serializable.transport.clone(),
        thinking_budgets: options.serializable.thinking_budgets.clone(),
        service_tier: options.serializable.service_tier.clone(),
    }
}

/// `ProxyStreamOptions` built from the `SimpleStreamOptions` an `Agent` passes in.
impl ProxyStreamOptions {
    pub fn from_simple(
        options: Option<SimpleStreamOptions>,
        auth_token: impl Into<String>,
        proxy_url: impl Into<String>,
    ) -> Self {
        let options = options.unwrap_or_default();
        Self {
            serializable: ProxySerializableStreamOptions {
                temperature: options.stream.temperature,
                max_tokens: options.stream.max_tokens,
                reasoning: options.reasoning.clone(),
                cache_retention: options.stream.cache_retention.clone(),
                session_id: options.stream.session_id.clone(),
                headers: options
                    .stream
                    .headers
                    .as_ref()
                    .map(|headers| headers.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect()),
                metadata: options.stream.metadata.clone(),
                transport: options.stream.transport.clone(),
                thinking_budgets: options.thinking_budgets.clone(),
                service_tier: options.stream.service_tier.clone(),
            },
            signal: options.stream.signal.clone(),
            auth_token: auth_token.into(),
            proxy_url: proxy_url.into(),
        }
    }
}

/// Partial `AssistantMessage` the proxy events accumulate into. The TypeScript
/// keeps a hidden `partialJson` on tool-call blocks while streaming; Rust keeps
/// the same string next to the partial message so `toolcall_end` can drop it.
struct ProxyPartialState {
    partial: AssistantMessage,
    partial_json: Map<String, Value>,
}

impl ProxyPartialState {
    fn new(model: &Model) -> Self {
        Self {
            partial: AssistantMessage {
                role: pi_ai::types::ROLE_ASSISTANT.to_string(),
                content: Vec::new(),
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                diagnostics: None,
                usage: Usage::zero(),
                stop_reason: pi_ai::types::STOP_REASON_STOP.to_string(),
                stop_reason_raw: None,
                error_message: None,
                timestamp: pi_ai::utils::now_ms(),
            },
            partial_json: Map::new(),
        }
    }

    /// JavaScript array assignment: growing beyond the current length leaves holes.
    fn set_block(&mut self, index: usize, block: ContentBlock) {
        while self.partial.content.len() <= index {
            self.partial
                .content
                .push(ContentBlock::Text(TextContent::new(String::new())));
        }
        self.partial.content[index] = block;
    }

    fn block_mut(&mut self, index: usize) -> Option<&mut ContentBlock> {
        self.partial.content.get_mut(index)
    }
}

/// `processProxyEvent(proxyEvent, partial)`.
fn process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    state: &mut ProxyPartialState,
) -> Result<Option<AssistantMessageEvent>, anyhow::Error> {
    match proxy_event {
        ProxyAssistantMessageEvent::Start => Ok(Some(AssistantMessageEvent::Start {
            partial: state.partial.clone(),
        })),

        ProxyAssistantMessageEvent::TextStart { content_index } => {
            state.set_block(content_index, ContentBlock::Text(TextContent::new(String::new())));
            Ok(Some(AssistantMessageEvent::TextStart {
                content_index,
                partial: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::TextDelta {
            content_index,
            delta,
        } => {
            let is_text = matches!(state.block_mut(content_index), Some(ContentBlock::Text(_)));
            if !is_text {
                return Err(anyhow::anyhow!("Received text_delta for non-text content"));
            }
            if let Some(ContentBlock::Text(content)) = state.block_mut(content_index) {
                content.text.push_str(&delta);
            }
            Ok(Some(AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                partial: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::TextEnd {
            content_index,
            content_signature,
        } => {
            let is_text = matches!(state.block_mut(content_index), Some(ContentBlock::Text(_)));
            if !is_text {
                return Err(anyhow::anyhow!("Received text_end for non-text content"));
            }
            let content = match state.block_mut(content_index) {
                Some(ContentBlock::Text(content)) => {
                    content.text_signature = content_signature;
                    content.text.clone()
                }
                _ => String::new(),
            };
            Ok(Some(AssistantMessageEvent::TextEnd {
                content_index,
                content,
                partial: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::ThinkingStart { content_index } => {
            state.set_block(content_index, ContentBlock::Thinking(ThinkingContent::new(String::new())));
            Ok(Some(AssistantMessageEvent::ThinkingStart {
                content_index,
                partial: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::ThinkingDelta {
            content_index,
            delta,
        } => {
            let is_thinking = matches!(state.block_mut(content_index), Some(ContentBlock::Thinking(_)));
            if !is_thinking {
                return Err(anyhow::anyhow!("Received thinking_delta for non-thinking content"));
            }
            if let Some(ContentBlock::Thinking(content)) = state.block_mut(content_index) {
                content.thinking.push_str(&delta);
            }
            Ok(Some(AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                partial: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::ThinkingEnd {
            content_index,
            content_signature,
        } => {
            let is_thinking = matches!(state.block_mut(content_index), Some(ContentBlock::Thinking(_)));
            if !is_thinking {
                return Err(anyhow::anyhow!("Received thinking_end for non-thinking content"));
            }
            let content = match state.block_mut(content_index) {
                Some(ContentBlock::Thinking(content)) => {
                    content.thinking_signature = content_signature;
                    content.thinking.clone()
                }
                _ => String::new(),
            };
            Ok(Some(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
                partial: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::ToolcallStart {
            content_index,
            id,
            tool_name,
        } => {
            state.set_block(
                content_index,
                ContentBlock::ToolCall(ToolCall::new(id, tool_name, Map::new())),
            );
            state
                .partial_json
                .insert(content_index.to_string(), Value::String(String::new()));
            Ok(Some(AssistantMessageEvent::ToolCallStart {
                content_index,
                partial: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::ToolcallDelta {
            content_index,
            delta,
        } => {
            let is_tool_call = matches!(state.block_mut(content_index), Some(ContentBlock::ToolCall(_)));
            if !is_tool_call {
                return Err(anyhow::anyhow!("Received toolcall_delta for non-toolCall content"));
            }
            let mut partial_json = match state.partial_json.get(&content_index.to_string()) {
                Some(Value::String(text)) => text.clone(),
                _ => String::new(),
            };
            partial_json.push_str(&delta);
            state
                .partial_json
                .insert(content_index.to_string(), Value::String(partial_json.clone()));
            let arguments = match parse_streaming_json(Some(&partial_json)) {
                Value::Object(map) => map,
                _ => Map::new(),
            };
            if let Some(ContentBlock::ToolCall(content)) = state.block_mut(content_index) {
                content.arguments = arguments;
            }
            Ok(Some(AssistantMessageEvent::ToolCallDelta {
                content_index,
                delta,
                partial: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::ToolcallEnd { content_index } => {
            let tool_call = match state.block_mut(content_index) {
                Some(ContentBlock::ToolCall(content)) => content.clone(),
                _ => return Ok(None),
            };
            // `delete (content as any).partialJson`
            state.partial_json.remove(&content_index.to_string());
            Ok(Some(AssistantMessageEvent::ToolCallEnd {
                content_index,
                tool_call,
                partial: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::Done { reason, usage } => {
            state.partial.stop_reason = reason.clone();
            state.partial.usage = usage;
            Ok(Some(AssistantMessageEvent::Done {
                reason,
                message: state.partial.clone(),
            }))
        }

        ProxyAssistantMessageEvent::Error {
            reason,
            error_message,
            usage,
        } => {
            state.partial.stop_reason = reason.clone();
            state.partial.error_message = error_message;
            state.partial.usage = usage;
            Ok(Some(AssistantMessageEvent::Error {
                reason,
                error: state.partial.clone(),
            }))
        }
    }
}

/// The body the proxy server receives: `{ model, context, options }`.
#[derive(Debug, Clone, Serialize)]
struct ProxyRequestBody {
    model: Model,
    context: Context,
    options: ProxySerializableStreamOptions,
}

/// Stream function that proxies through a server instead of calling LLM providers
/// directly. The server strips the partial field from delta events to reduce
/// bandwidth, so the partial message is reconstructed client-side.
///
/// Use this as the `streamFn` option when creating an Agent that needs to go
/// through a proxy.
pub fn stream_proxy(model: Model, context: Context, options: ProxyStreamOptions) -> ProxyMessageEventStream {
    let stream = ProxyMessageEventStream::new();
    let stream_for_task = stream.clone();

    tokio::spawn(async move {
        let mut state = ProxyPartialState::new(&model);
        let signal = options.signal.clone();

        let body = ProxyRequestBody {
            model: model.clone(),
            context,
            options: build_proxy_request_options(&options),
        };
        let url = format!("{}/api/stream", options.proxy_url);

        let client = reqwest::Client::new();
        let request = client
            .post(url)
            .header("Authorization", format!("Bearer {}", options.auth_token))
            .header("Content-Type", "application/json")
            .json(&body);

        let response = match request
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                let error_message = error.to_string();
                let reason = if signal.as_ref().map(|signal| signal.is_cancelled()).unwrap_or(false) {
                    pi_ai::types::STOP_REASON_ABORTED
                } else {
                    pi_ai::types::STOP_REASON_ERROR
                };
                state.partial.stop_reason = reason.to_string();
                state.partial.error_message = Some(error_message);
                stream_for_task.push(AssistantMessageEvent::Error {
                    reason: reason.to_string(),
                    error: state.partial.clone(),
                });
                stream_for_task.end(None);
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let status_text = status.canonical_reason().unwrap_or_default().to_string();
            let mut error_message = format!("Proxy error: {} {}", status.as_u16(), status_text);
            if let Ok(error_data) = response.json::<Value>().await {
                if let Some(error) = error_data.get("error").and_then(Value::as_str) {
                    error_message = format!("Proxy error: {error}");
                }
            }
            let reason = if signal.as_ref().map(|signal| signal.is_cancelled()).unwrap_or(false) {
                pi_ai::types::STOP_REASON_ABORTED
            } else {
                pi_ai::types::STOP_REASON_ERROR
            };
            state.partial.stop_reason = reason.to_string();
            state.partial.error_message = Some(error_message);
            stream_for_task.push(AssistantMessageEvent::Error {
                reason: reason.to_string(),
                error: state.partial.clone(),
            });
            stream_for_task.end(None);
            return;
        }

        // `let mut reader = response.body!.getReader();`
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();

        'outer: loop {
            let chunk = match signal {
                Some(ref signal) => {
                    let signal = signal.clone();
                    tokio::select! {
                        biased;
                        _ = signal.cancelled() => {
                            let error_message = "Request aborted by user".to_string();
                            state.partial.stop_reason = pi_ai::types::STOP_REASON_ABORTED.to_string();
                            state.partial.error_message = Some(error_message);
                            stream_for_task.push(AssistantMessageEvent::Error {
                                reason: pi_ai::types::STOP_REASON_ABORTED.to_string(),
                                error: state.partial.clone(),
                            });
                            stream_for_task.end(None);
                            return;
                        }
                        next = byte_stream.next() => next,
                    }
                }
                None => byte_stream.next().await,
            };

            let Some(chunk) = chunk else {
                break;
            };
            let value = match chunk {
                Ok(value) => value,
                Err(error) => {
                    let error_message = error.to_string();
                    let reason = if signal.as_ref().map(|signal| signal.is_cancelled()).unwrap_or(false) {
                        pi_ai::types::STOP_REASON_ABORTED
                    } else {
                        pi_ai::types::STOP_REASON_ERROR
                    };
                    state.partial.stop_reason = reason.to_string();
                    state.partial.error_message = Some(error_message);
                    stream_for_task.push(AssistantMessageEvent::Error {
                        reason: reason.to_string(),
                        error: state.partial.clone(),
                    });
                    stream_for_task.end(None);
                    return;
                }
            };

            if signal.as_ref().map(|signal| signal.is_cancelled()).unwrap_or(false) {
                let error_message = "Request aborted by user".to_string();
                state.partial.stop_reason = pi_ai::types::STOP_REASON_ABORTED.to_string();
                state.partial.error_message = Some(error_message);
                stream_for_task.push(AssistantMessageEvent::Error {
                    reason: pi_ai::types::STOP_REASON_ABORTED.to_string(),
                    error: state.partial.clone(),
                });
                stream_for_task.end(None);
                return;
            }

            // `decoder.decode(value, { stream: true })` - UTF-8 is lossy-decoded the
            // same way a streaming TextDecoder handles partial sequences.
            buffer.push_str(&String::from_utf8_lossy(&value));
            let mut lines: Vec<String> = buffer.split('\n').map(|line| line.to_string()).collect();
            buffer = lines.pop().unwrap_or_default();

            for line in lines {
                if let Some(data) = line.strip_prefix("data: ") {
                    let data = data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    let proxy_event: ProxyAssistantMessageEvent = match serde_json::from_str(data) {
                        Ok(event) => event,
                        Err(error) => {
                            let error_message = error.to_string();
                            state.partial.stop_reason = pi_ai::types::STOP_REASON_ERROR.to_string();
                            state.partial.error_message = Some(error_message);
                            stream_for_task.push(AssistantMessageEvent::Error {
                                reason: pi_ai::types::STOP_REASON_ERROR.to_string(),
                                error: state.partial.clone(),
                            });
                            stream_for_task.end(None);
                            break 'outer;
                        }
                    };
                    match process_proxy_event(proxy_event, &mut state) {
                        Ok(Some(event)) => {
                            stream_for_task.push(event);
                            if stream_for_task.is_done() {
                                return;
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            let error_message = error.to_string();
                            state.partial.stop_reason = pi_ai::types::STOP_REASON_ERROR.to_string();
                            state.partial.error_message = Some(error_message);
                            stream_for_task.push(AssistantMessageEvent::Error {
                                reason: pi_ai::types::STOP_REASON_ERROR.to_string(),
                                error: state.partial.clone(),
                            });
                            stream_for_task.end(None);
                            break 'outer;
                        }
                    }
                }
            }
        }

        if signal.as_ref().map(|signal| signal.is_cancelled()).unwrap_or(false) {
            let error_message = "Request aborted by user".to_string();
            state.partial.stop_reason = pi_ai::types::STOP_REASON_ABORTED.to_string();
            state.partial.error_message = Some(error_message);
            stream_for_task.push(AssistantMessageEvent::Error {
                reason: pi_ai::types::STOP_REASON_ABORTED.to_string(),
                error: state.partial.clone(),
            });
            stream_for_task.end(None);
            return;
        }

        if !stream_for_task.is_done() {
            state.partial.stop_reason = pi_ai::types::STOP_REASON_ERROR.to_string();
            state.partial.error_message = Some("Proxy stream truncated before completion".to_string());
            stream_for_task.push(AssistantMessageEvent::Error {
                reason: pi_ai::types::STOP_REASON_ERROR.to_string(),
                error: state.partial.clone(),
            });
        }
        stream_for_task.end(None);
    });

    stream
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn fixture(body: String, hold_open: bool) -> (String, tokio::sync::oneshot::Receiver<Value>, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let request = loop {
                let mut chunk = [0u8; 4096];
                let count = socket.read(&mut chunk).await.unwrap();
                assert_ne!(count, 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..end]);
                    let length: usize = headers.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length").then(|| value.trim().parse().unwrap())
                    }).unwrap();
                    if bytes.len() >= end + 4 + length {
                        break serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
                    }
                }
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len() + if hold_open { 1000 } else { 0 },
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            let _ = sender.send(request);
            if hold_open {
                std::future::pending::<()>().await;
            }
        });
        (format!("http://{address}"), receiver, task)
    }

    fn start(url: String, options: Option<SimpleStreamOptions>) -> ProxyMessageEventStream {
        stream_proxy(Model::default(), Context::default(), ProxyStreamOptions::from_simple(options, "synthetic-proxy-token", url))
    }

    async fn finish(stream: &ProxyMessageEventStream) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
        tokio::time::timeout(Duration::from_secs(3), async {
            let mut events = Vec::new();
            while let Some(event) = stream.next().await { events.push(event); }
            (events, stream.result().await)
        }).await.expect("proxy final result must settle")
    }

    #[tokio::test]
    async fn truncated_stream_returns_error_and_preserves_partial_text() {
        for body in ["", "data: {\"type\":\"text_start\",\"contentIndex\":0}\n\ndata: {\"type\":\"text_delta\",\"contentIndex\":0,\"delta\":\"partial\"}\n\n"] {
            let (url, _, task) = fixture(body.to_string(), false).await;
            let (events, result) = finish(&start(url, None)).await;
            assert_eq!(result.stop_reason, "error");
            assert_eq!(result.error_message.as_deref(), Some("Proxy stream truncated before completion"));
            assert_eq!(events.iter().filter(|event| matches!(event, AssistantMessageEvent::Error { .. })).count(), 1);
            if !body.is_empty() {
                assert!(matches!(&result.content[0], ContentBlock::Text(text) if text.text == "partial"));
            }
            task.await.unwrap();
        }
    }

    #[tokio::test]
    async fn terminal_events_finish_without_waiting_for_server_eof() {
        for event in [
            serde_json::json!({"type":"done", "reason":"stop", "usage":Usage::zero()}),
            serde_json::json!({"type":"error", "reason":"error", "errorMessage":"provider failure", "usage":Usage::zero()}),
        ] {
            let (url, _, task) = fixture(format!("data: {event}\n\n"), true).await;
            let (events, result) = finish(&start(url, None)).await;
            task.abort();
            assert_eq!(events.len(), 1);
            assert_eq!(result.stop_reason, event["reason"].as_str().unwrap());
            assert_eq!(result.error_message.as_deref(), event["errorMessage"].as_str());
        }
    }

    #[tokio::test]
    async fn cancellation_stays_aborted_and_service_tier_reaches_proxy() {
        let (url, request, task) = fixture("data: {\"type\":\"start\"}\n\n".to_string(), true).await;
        let signal = CancellationToken::new();
        let mut options = SimpleStreamOptions::default();
        options.stream.signal = Some(signal.clone());
        options.stream.service_tier = Some(Some("priority".to_string()));
        options.stream.api_key = Some("client-local-key".to_string());
        let stream = start(url, Some(options));
        let request = tokio::time::timeout(Duration::from_secs(3), request).await.unwrap().unwrap();
        assert_eq!(request["options"]["serviceTier"], "priority");
        assert!(request["options"].get("apiKey").is_none());
        assert!(request["options"].get("signal").is_none());
        signal.cancel();
        let (_, result) = finish(&stream).await;
        task.abort();
        assert_eq!(result.stop_reason, "aborted");
    }

    #[test]
    fn unspecified_service_tier_is_omitted() {
        let options = build_proxy_request_options(&ProxyStreamOptions::default());
        assert!(serde_json::to_value(options).unwrap().get("serviceTier").is_none());
        let explicit_null: ProxySerializableStreamOptions = serde_json::from_value(serde_json::json!({"serviceTier":null})).unwrap();
        assert_eq!(explicit_null.service_tier, Some(None));
        assert_eq!(serde_json::to_value(explicit_null).unwrap()["serviceTier"], Value::Null);
    }
}
