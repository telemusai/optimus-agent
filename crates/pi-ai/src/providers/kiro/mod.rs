//! Native Kiro transport. Optimus retains the tool loop; Kiro CLI is used only for login ownership.
//! Protocol reference: hongyilyu/pi-kiro and satiyap/pi-kiro-api (MIT; see NOTICE).
pub mod auth;
pub mod catalog;
mod event;
mod request;
#[cfg(test)]
mod tests;

use crate::types::*;
use crate::utils::event_stream::{
    create_assistant_message_event_stream, AssistantMessageEventStream,
};
use auth::{Credential, ENVELOPE};
use futures_util::StreamExt;
use serde_json::Value;
use std::time::Duration;

pub const DEFAULT_ENDPOINT: &str = "https://runtime.us-east-1.kiro.dev/";

/// Explicit regions use a URL distinct from the bundled default sentinel.
pub fn endpoint_for_region(region: &str) -> Result<String, String> {
    if !auth::valid_region(region) {
        return Err("Invalid Kiro region (expected an AWS region such as us-east-1)".into());
    }
    Ok(format!("https://runtime.{region}.kiro.dev"))
}

fn runtime_region(url: &url::Url) -> Option<&str> {
    url.host_str()?
        .strip_prefix("runtime.")?
        .strip_suffix(".kiro.dev")
        .filter(|region| auth::valid_region(region))
}

pub(crate) struct Access {
    credential: Credential,
    api_key: bool,
}
impl Access {
    pub(crate) fn parse(key: &str) -> Result<Self, String> {
        if let Some(encoded) = key.strip_prefix(ENVELOPE) {
            let credential: Credential =
                serde_json::from_str(encoded).map_err(|_| "Invalid Kiro OAuth credential")?;
            if credential.access.is_empty() || !auth::valid_region(&credential.region) {
                return Err("Invalid Kiro OAuth credential".into());
            }
            return Ok(Self {
                credential,
                api_key: false,
            });
        }
        if !key.starts_with("ksk_") {
            return Err(
                "Configure KIRO_API_KEY or use /login kiro with an existing Kiro CLI login".into(),
            );
        }
        let region = std::env::var("KIRO_API_REGION").unwrap_or_else(|_| "us-east-1".into());
        if !auth::valid_region(&region) {
            return Err("Invalid KIRO_API_REGION".into());
        }
        Ok(Self {
            credential: Credential {
                access: key.into(),
                region,
                profile_arn: None,
            },
            api_key: true,
        })
    }
    pub(crate) fn root(&self, base: &str) -> Result<url::Url, String> {
        let default = if self.api_key {
            format!("https://q.{}.amazonaws.com/", self.credential.region)
        } else {
            DEFAULT_ENDPOINT.to_string()
        };
        let base = if base.is_empty() || base == DEFAULT_ENDPOINT {
            &default
        } else {
            base
        };
        let mut url = url::Url::parse(base).map_err(|_| "Invalid Kiro baseUrl")?;
        if url.scheme() != "https"
            && !(url.scheme() == "http"
                && matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]")))
        {
            return Err("Kiro baseUrl must use HTTPS (HTTP is allowed for local fixtures)".into());
        }
        if !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(
                "Kiro baseUrl must not contain credentials, query parameters or a fragment".into(),
            );
        }
        let loopback = matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"));
        if !self.api_key && runtime_region(&url).is_none() && !(cfg!(test) && loopback) {
            return Err(
                "Kiro CLI OAuth credentials are restricted to regional Kiro endpoints".into(),
            );
        }
        if self.api_key {
            if let Some(region) = runtime_region(&url).map(str::to_owned) {
                url.set_host(Some(&format!("q.{region}.amazonaws.com")))
                    .map_err(|_| "Invalid Kiro API-key endpoint")?;
            }
        }
        let path = url
            .path()
            .trim_end_matches('/')
            .trim_end_matches("/generateAssistantResponse")
            .to_string();
        url.set_path(&format!("{path}/"));
        Ok(url)
    }
    pub(crate) fn management_root(&self, base: &str) -> Result<url::Url, String> {
        let mut root = self.root(base)?;
        if !self.api_key {
            if let Some(region) = runtime_region(&root).map(str::to_owned) {
                root.set_host(Some(&format!("management.{region}.kiro.dev")))
                    .map_err(|_| "Invalid Kiro management endpoint")?;
            }
        }
        Ok(root)
    }
    pub(crate) fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut request = request.bearer_auth(&self.credential.access);
        if self.api_key {
            request = request.header("tokentype", "API_KEY").header(
                "User-Agent",
                concat!("Optimus-Agent/", env!("CARGO_PKG_VERSION")),
            );
        } else {
            // Kiro routes CLI OAuth subscriptions using this legacy application marker.
            // Retain Optimus's identity; no borrowed SDK version or device ID is needed.
            let user_agent = concat!(
                "Optimus-Agent/",
                env!("CARGO_PKG_VERSION"),
                " app/AmazonQ-For-CLI"
            );
            request = request
                .header("User-Agent", user_agent)
                .header("x-amz-user-agent", user_agent);
        }
        request
    }
    pub(crate) fn origin(&self) -> &'static str {
        if self.api_key {
            "AI_EDITOR"
        } else {
            "KIRO_CLI"
        }
    }
    pub(crate) fn request(
        &self,
        client: &reqwest::Client,
        url: url::Url,
        target: &str,
    ) -> reqwest::RequestBuilder {
        let mut request = self
            .authorize(client.post(url))
            .header("x-amzn-codewhisperer-optout", "true")
            .header("amz-sdk-invocation-id", uuid::Uuid::new_v4().to_string());
        if self.api_key {
            request = request
                .header("Content-Type", "application/x-amz-json-1.0")
                .header("X-Amz-Target", target);
        } else {
            request = request.header("Content-Type", "application/json");
        }
        request
    }
    pub(crate) fn error(&self, message: &str) -> String {
        // Provider bodies can echo request values. Never allow the bearer token into diagnostics.
        message
            .replace(&self.credential.access, "[redacted]")
            .chars()
            .take(1500)
            .collect()
    }
}

pub(crate) fn client() -> Result<reqwest::Client, String> {
    crate::providers::shared_http::try_shared_client(crate::providers::shared_http::ClientPolicy::Kiro)
        .map_err(|_| "Cannot initialize Kiro HTTP client".into())
}

pub(crate) async fn response_error(response: reqwest::Response, access: &Access) -> String {
    let status = response.status();
    let mut bytes = vec![];
    let mut chunks = response.bytes_stream();
    while let Some(Ok(chunk)) = chunks.next().await {
        bytes.extend_from_slice(&chunk[..chunk.len().min(8192 - bytes.len())]);
        if bytes.len() >= 8192 {
            break;
        }
    }
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let message = value
        .get("message")
        .or_else(|| value.get("Message"))
        .and_then(Value::as_str)
        .unwrap_or("Request rejected by Kiro");
    let detail = access.error(message);
    if status.as_u16() == 403 {
        format!("Kiro access denied (HTTP 403): {detail}")
    } else if status.as_u16() == 413 || message.contains("CONTENT_LENGTH_EXCEEDS_THRESHOLD") {
        "Kiro context_length_exceeded; conversation preserved".into()
    } else {
        format!("Kiro HTTP {}: {detail}", status.as_u16())
    }
}

pub fn stream_kiro(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> AssistantMessageEventStream {
    stream_simple_kiro(
        model,
        context,
        Some(&SimpleStreamOptions {
            stream: options.cloned().unwrap_or_default(),
            ..Default::default()
        }),
    )
}

pub fn stream_simple_kiro(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let stream = create_assistant_message_event_stream();
    let sink = stream.clone();
    let model = model.clone();
    let context = context.clone();
    let options = options.cloned().unwrap_or_default();
    stream.spawn(async move {
        let mut output = AssistantMessage::new(
            &model.api,
            &model.provider,
            &model.id,
            crate::utils::now_ms(),
        );
        sink.push(AssistantMessageEvent::Start {
            partial: output.clone(),
        });
        let signal = options.stream.signal.clone().unwrap_or_default();
        let result = tokio::select! {
            biased;
            _ = signal.cancelled() => Err("Kiro request cancelled".to_string()),
            result = run(&model, &context, &options, &sink, &mut output) => result,
        };
        match result {
            Ok(()) => {
                sink.push(AssistantMessageEvent::Done {
                    reason: output.stop_reason.clone(),
                    message: output.clone(),
                });
            }
            Err(error) => {
                output.stop_reason = if signal.is_cancelled() {
                    "aborted"
                } else {
                    "error"
                }
                .into();
                output.error_message = Some(error);
                sink.push(AssistantMessageEvent::Error {
                    reason: output.stop_reason.clone(),
                    error: output.clone(),
                });
            }
        }
        sink.end(Some(output));
    });
    stream
}

async fn run(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    stream: &AssistantMessageEventStream,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    let key = options
        .stream
        .api_key
        .clone()
        .or_else(|| std::env::var("KIRO_API_KEY").ok())
        .ok_or("No Kiro credential; run /login kiro or configure KIRO_API_KEY")?;
    let access = Access::parse(&key)?;
    let timeout = Duration::from_millis(
        options
            .stream
            .timeout_ms
            .filter(|n| n.is_finite() && *n > 0.0)
            .unwrap_or(180_000.0)
            .min(900_000.0) as u64,
    );
    let client = client()?;
    let mut url = access.root(&model.base_url)?;
    if !access.api_key {
        url = url
            .join("generateAssistantResponse")
            .map_err(|_| "Invalid Kiro endpoint")?;
    }
    let mut body = request::build(
        model,
        context,
        options,
        access.origin(),
        access.credential.profile_arn.as_deref(),
    )?;
    if let Some(callback) = &options.stream.on_payload {
        if let Some(next) = callback(body.clone(), model).await {
            body = next;
        }
    }
    let mut request = access
        .request(
            &client,
            url,
            "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
        )
        .header("Accept", "application/vnd.amazon.eventstream")
        .header("x-amzn-kiro-agent-mode", "vibe");
    for headers in [model.headers.as_ref(), options.stream.headers.as_ref()]
        .into_iter()
        .flatten()
    {
        for (name, value) in headers {
            if !matches!(
                name.to_ascii_lowercase().as_str(),
                "authorization" | "tokentype" | "x-amz-target"
            ) {
                request = request.header(name, value);
            }
        }
    }
    let response = tokio::time::timeout(timeout, request.json(&body).send())
        .await
        .map_err(|_| "Kiro response timed out")?
        .map_err(|_| "Kiro network request failed")?;
    if let Some(callback) = &options.stream.on_response {
        callback(
            ProviderResponse {
                status: i64::from(response.status().as_u16()),
                headers: response
                    .headers()
                    .iter()
                    .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.to_string(), v.to_string())))
                    .collect(),
            },
            model,
        )
        .await;
    }
    if !response.status().is_success() {
        return Err(tokio::time::timeout(
            Duration::from_secs(10),
            response_error(response, &access),
        )
        .await
        .unwrap_or_else(|_| "Kiro rejected the request; error body timed out".into()));
    }
    let mut chunks = response.bytes_stream();
    let mut buffer = vec![];
    let mut state = State::default();
    while let Some(chunk) = tokio::time::timeout(timeout, chunks.next())
        .await
        .map_err(|_| "Kiro stream idle timeout")?
    {
        let chunk = chunk.map_err(|_| "Kiro stream interrupted")?;
        buffer.extend_from_slice(&chunk);
        for frame in event::decode(&mut buffer)? {
            if matches!(
                frame.headers.get(":message-type").map(String::as_str),
                Some("exception" | "error")
            ) {
                return Err(format!(
                    "Kiro stream error: {}",
                    access.error(
                        frame.payload["message"]
                            .as_str()
                            .unwrap_or("provider exception")
                    )
                ));
            }
            state.process(&frame.payload, output, stream)?;
        }
    }
    if !buffer.is_empty() {
        return Err("Kiro stream ended inside an event frame".into());
    }
    if state.tool.is_some() {
        return Err("Kiro stream ended before a tool call was complete".into());
    }
    state.flush_text(output, stream, true);
    state.close_text(output, stream);
    let has_tool = output
        .content
        .iter()
        .any(|c| matches!(c, ContentBlock::ToolCall(_)));
    let has_text = output
        .content
        .iter()
        .any(|c| c.as_text().is_some_and(|t| !t.text.is_empty()));
    if !has_tool && !has_text {
        return Err("Kiro returned no usable response".into());
    }
    output.stop_reason = if has_tool {
        "toolUse"
    } else if state.complete && !state.thinking {
        "stop"
    } else {
        "length"
    }
    .into();
    if !state.exact_usage {
        output.usage.input = state
            .context_percentage
            .map(|pct| (pct / 100.0 * model.context_window).round())
            .unwrap_or_else(|| (body.to_string().chars().count() as f64 / 4.0).ceil());
        output.usage.output = (output
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text(b) => b.text.chars().count(),
                ContentBlock::Thinking(b) => b.thinking.chars().count(),
                ContentBlock::ToolCall(b) => serde_json::to_string(&b.arguments)
                    .unwrap_or_default()
                    .chars()
                    .count(),
            })
            .sum::<usize>() as f64
            / 4.0)
            .ceil();
    }
    output.usage.total_tokens = output.usage.input
        + output.usage.output
        + output.usage.cache_read
        + output.usage.cache_write;
    crate::models::calculate_cost(model, &mut output.usage, None);
    if state.exact_usage {
        if let Some(callback) = &options.stream.on_usage_observation {
            callback(
                ProviderUsageObservation {
                    input_tokens: Some(Some(output.usage.input)),
                    output_tokens: Some(Some(output.usage.output)),
                    total_tokens: Some(Some(output.usage.total_tokens)),
                    ..Default::default()
                },
                model,
            )
            .await;
        }
    }
    Ok(())
}

#[derive(Default)]
struct State {
    text_index: Option<usize>,
    text: String,
    thinking: bool,
    reasoning_done: bool,
    close_tag: String,
    tool: Option<(usize, String)>,
    complete: bool,
    context_percentage: Option<f64>,
    exact_usage: bool,
}

impl State {
    fn close_text(&mut self, output: &AssistantMessage, stream: &AssistantMessageEventStream) {
        if let Some(index) = self.text_index.take() {
            let event = match &output.content[index] {
                ContentBlock::Text(b) => AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content: b.text.clone(),
                    partial: output.clone(),
                },
                ContentBlock::Thinking(b) => AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content: b.thinking.clone(),
                    partial: output.clone(),
                },
                _ => return,
            };
            stream.push(event);
        }
    }
    fn emit(
        &mut self,
        text: String,
        output: &mut AssistantMessage,
        stream: &AssistantMessageEventStream,
    ) {
        if text.is_empty() {
            return;
        }
        let index = *self.text_index.get_or_insert_with(|| {
            let index = output.content.len();
            output.content.push(if self.thinking {
                ContentBlock::Thinking(ThinkingContent::default())
            } else {
                ContentBlock::Text(TextContent::default())
            });
            stream.push(if self.thinking {
                AssistantMessageEvent::ThinkingStart {
                    content_index: index,
                    partial: output.clone(),
                }
            } else {
                AssistantMessageEvent::TextStart {
                    content_index: index,
                    partial: output.clone(),
                }
            });
            index
        });
        match &mut output.content[index] {
            ContentBlock::Text(block) => block.text.push_str(&text),
            ContentBlock::Thinking(block) => block.thinking.push_str(&text),
            _ => {}
        }
        stream.push(if self.thinking {
            AssistantMessageEvent::ThinkingDelta {
                content_index: index,
                delta: text,
                partial: output.clone(),
            }
        } else {
            AssistantMessageEvent::TextDelta {
                content_index: index,
                delta: text,
                partial: output.clone(),
            }
        });
    }
    fn flush_text(
        &mut self,
        output: &mut AssistantMessage,
        stream: &AssistantMessageEventStream,
        finish: bool,
    ) {
        const TAGS: [(&str, &str); 4] = [
            ("<thinking>", "</thinking>"),
            ("<think>", "</think>"),
            ("<reasoning>", "</reasoning>"),
            ("<thought>", "</thought>"),
        ];
        if !self.thinking && !self.reasoning_done {
            let trimmed = self.text.trim_start();
            if let Some((open, close)) = TAGS.iter().find(|(tag, _)| trimmed.starts_with(tag)) {
                self.text = trimmed[open.len()..].into();
                self.thinking = true;
                self.close_tag = close.to_string();
            } else if !finish
                && (trimmed.is_empty() || TAGS.iter().any(|(tag, _)| tag.starts_with(trimmed)))
            {
                return;
            } else {
                self.reasoning_done = true;
            }
        }
        if self.thinking {
            if let Some(end) = self.text.find(&self.close_tag) {
                let text = self.text[..end].to_owned();
                self.text = self.text[end + self.close_tag.len()..].to_owned();
                self.emit(text, output, stream);
                self.close_text(output, stream);
                self.thinking = false;
                self.reasoning_done = true;
            } else if !finish {
                let hold = (1..self.close_tag.len())
                    .rev()
                    .find(|len| self.text.ends_with(&self.close_tag[..*len]))
                    .unwrap_or(0);
                let safe = self.text.len() - hold;
                let text = self.text[..safe].to_owned();
                self.text = self.text[safe..].to_owned();
                self.emit(text, output, stream);
                return;
            }
        }
        let text = std::mem::take(&mut self.text);
        self.emit(text, output, stream);
    }
    fn process(
        &mut self,
        value: &Value,
        output: &mut AssistantMessage,
        stream: &AssistantMessageEventStream,
    ) -> Result<(), String> {
        if value.get("error").is_some() || value.get("Error").is_some() {
            return Err("Kiro reported an error in the response stream".into());
        }
        if let Some(content) = value["content"].as_str() {
            self.text.push_str(content);
            self.flush_text(output, stream, false);
        }
        if let Some(pct) = value["contextUsagePercentage"]
            .as_f64()
            .filter(|pct| pct.is_finite() && (0.0..=100.0).contains(pct))
        {
            self.context_percentage = Some(pct);
            self.complete = true;
        }
        let usage = value.get("usage").unwrap_or(value);
        if let (Some(input), Some(out)) = (
            usage["inputTokens"].as_u64(),
            usage["outputTokens"].as_u64(),
        ) {
            output.usage.input = input as f64;
            output.usage.output = out as f64;
            self.exact_usage = true;
        }
        if let Some(id) = value["toolUseId"].as_str() {
            let same = self.tool.as_ref().is_some_and(|(index, _)| {
                output.content[*index]
                    .as_tool_call()
                    .is_some_and(|c| c.id == id)
            });
            if !same {
                if self.tool.is_some() {
                    return Err(
                        "Kiro started a tool call before completing the preceding call".into(),
                    );
                }
                let name = value["name"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or("Kiro tool call has no name")?;
                self.flush_text(output, stream, true);
                self.close_text(output, stream);
                let index = output.content.len();
                output.content.push(ContentBlock::ToolCall(ToolCall::new(
                    id,
                    name,
                    Default::default(),
                )));
                self.tool = Some((index, String::new()));
                stream.push(AssistantMessageEvent::ToolCallStart {
                    content_index: index,
                    partial: output.clone(),
                });
            }
        }
        if let Some(input) = value.get("input") {
            let (index, arguments) = self
                .tool
                .as_mut()
                .ok_or("Kiro returned tool input without a tool call")?;
            let delta = if let Some(text) = input.as_str() {
                text.to_string()
            } else if input.is_object() {
                input.to_string()
            } else {
                return Err("Invalid Kiro tool input".into());
            };
            arguments.push_str(&delta);
            if arguments.len() > 16 * 1024 * 1024 {
                return Err("Kiro tool input is too large".into());
            }
            stream.push(AssistantMessageEvent::ToolCallDelta {
                content_index: *index,
                delta,
                partial: output.clone(),
            });
        }
        if value["stop"].as_bool() == Some(true) {
            if let Some((index, arguments)) = self.tool.take() {
                let parsed = serde_json::from_str::<serde_json::Map<String, Value>>(
                    if arguments.trim().is_empty() {
                        "{}"
                    } else {
                        &arguments
                    },
                )
                .map_err(|_| "Kiro returned invalid tool arguments; tool was not executed")?;
                let ContentBlock::ToolCall(call) = &mut output.content[index] else {
                    unreachable!()
                };
                call.arguments = parsed;
                let call = call.clone();
                stream.push(AssistantMessageEvent::ToolCallEnd {
                    content_index: index,
                    tool_call: call,
                    partial: output.clone(),
                });
            }
        }
        Ok(())
    }
}
