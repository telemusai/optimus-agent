//! Port of packages/ai/src/types.ts
//!
//! Type mapping notes (docs/PORT-RULES.md):
//! - TypeScript union-of-literals types such as `Api = KnownApi | (string & {})`
//!   carry no runtime distinction, so they stay `String` with the known values
//!   listed in `KNOWN_APIS` / `KNOWN_PROVIDERS` and the `API_*` / `PROVIDER_*`
//!   constants.
//! - `Model<TApi>` is not generic in Rust. `TApi` only selects the `compat`
//!   shape, which is modelled with the `Compat` enum.
//! - `AbortSignal` -> `tokio_util::sync::CancellationToken`.
//! - `Map<K,V>` -> `indexmap::IndexMap` where iteration order is observable
//!   (model headers), otherwise `serde_json::Map` (which preserves insertion
//!   order through serde_json's `preserve_order` feature).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::compaction::{
    CompactionOptions, ProviderCompactionCheckpoint, ProviderCompactionResult,
};
use crate::utils::diagnostics::AssistantMessageDiagnostic;
use crate::utils::event_stream::AssistantMessageEventStream;

pub use crate::utils::event_stream::AssistantMessageEventStream as AssistantMessageEventStreamReExport;

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

// ---------------------------------------------------------------------------
// Api / Provider
// ---------------------------------------------------------------------------

pub type KnownApi = String;
pub type Api = String;

pub const API_OPENAI_COMPLETIONS: &str = "openai-completions";
pub const API_MISTRAL_CONVERSATIONS: &str = "mistral-conversations";
pub const API_OPENAI_RESPONSES: &str = "openai-responses";
pub const API_AZURE_OPENAI_RESPONSES: &str = "azure-openai-responses";
pub const API_OPENAI_CODEX_RESPONSES: &str = "openai-codex-responses";
pub const API_ANTHROPIC_MESSAGES: &str = "anthropic-messages";
pub const API_BEDROCK_CONVERSE_STREAM: &str = "bedrock-converse-stream";
pub const API_BEDROCK_RESPONSES: &str = "bedrock-responses";
pub const API_GOOGLE_GENERATIVE_AI: &str = "google-generative-ai";
pub const API_KIRO: &str = "kiro-api";
pub const API_GOOGLE_VERTEX: &str = "google-vertex";

/// The `KnownApi` union in declaration order.
pub const KNOWN_APIS: [&str; 11] = [
    API_OPENAI_COMPLETIONS,
    API_MISTRAL_CONVERSATIONS,
    API_OPENAI_RESPONSES,
    API_AZURE_OPENAI_RESPONSES,
    API_OPENAI_CODEX_RESPONSES,
    API_ANTHROPIC_MESSAGES,
    API_BEDROCK_CONVERSE_STREAM,
    API_BEDROCK_RESPONSES,
    API_GOOGLE_GENERATIVE_AI,
    API_GOOGLE_VERTEX,
    API_KIRO,
];

pub type KnownProvider = String;
pub type Provider = String;

pub const PROVIDER_AMAZON_BEDROCK: &str = "amazon-bedrock";
pub const PROVIDER_ANTHROPIC: &str = "anthropic";
pub const PROVIDER_KIRO: &str = "kiro";
pub const PROVIDER_GOOGLE: &str = "google";
pub const PROVIDER_GOOGLE_VERTEX: &str = "google-vertex";
pub const PROVIDER_OPENAI: &str = "openai";
pub const PROVIDER_AZURE_OPENAI_RESPONSES: &str = "azure-openai-responses";
pub const PROVIDER_OPENAI_CODEX: &str = "openai-codex";
pub const PROVIDER_PRIME_INFERENCE: &str = "prime-inference";
pub const PROVIDER_DEEPSEEK: &str = "deepseek";
pub const PROVIDER_GITHUB_COPILOT: &str = "github-copilot";
pub const PROVIDER_XAI: &str = "xai";
pub const PROVIDER_GROQ: &str = "groq";
pub const PROVIDER_CEREBRAS: &str = "cerebras";
pub const PROVIDER_OPENROUTER: &str = "openrouter";
pub const PROVIDER_VERCEL_AI_GATEWAY: &str = "vercel-ai-gateway";
pub const PROVIDER_ZAI: &str = "zai";
pub const PROVIDER_MISTRAL: &str = "mistral";
pub const PROVIDER_MINIMAX: &str = "minimax";
pub const PROVIDER_MINIMAX_CN: &str = "minimax-cn";
pub const PROVIDER_MOONSHOTAI: &str = "moonshotai";
pub const PROVIDER_MOONSHOTAI_CN: &str = "moonshotai-cn";
pub const PROVIDER_HUGGINGFACE: &str = "huggingface";
pub const PROVIDER_FIREWORKS: &str = "fireworks";
pub const PROVIDER_OPENCODE: &str = "opencode";
pub const PROVIDER_OPENCODE_GO: &str = "opencode-go";
pub const PROVIDER_KIMI_CODING: &str = "kimi-coding";
pub const PROVIDER_CLOUDFLARE_WORKERS_AI: &str = "cloudflare-workers-ai";
pub const PROVIDER_CLOUDFLARE_AI_GATEWAY: &str = "cloudflare-ai-gateway";
pub const PROVIDER_XIAOMI: &str = "xiaomi";
pub const PROVIDER_XIAOMI_TOKEN_PLAN_CN: &str = "xiaomi-token-plan-cn";
pub const PROVIDER_XIAOMI_TOKEN_PLAN_AMS: &str = "xiaomi-token-plan-ams";
pub const PROVIDER_XIAOMI_TOKEN_PLAN_SGP: &str = "xiaomi-token-plan-sgp";

/// The `KnownProvider` union in declaration order.
pub const KNOWN_PROVIDERS: [&str; 33] = [
    PROVIDER_AMAZON_BEDROCK,
    PROVIDER_ANTHROPIC,
    PROVIDER_KIRO,
    PROVIDER_GOOGLE,
    PROVIDER_GOOGLE_VERTEX,
    PROVIDER_OPENAI,
    PROVIDER_AZURE_OPENAI_RESPONSES,
    PROVIDER_OPENAI_CODEX,
    PROVIDER_PRIME_INFERENCE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GITHUB_COPILOT,
    PROVIDER_XAI,
    PROVIDER_GROQ,
    PROVIDER_CEREBRAS,
    PROVIDER_OPENROUTER,
    PROVIDER_VERCEL_AI_GATEWAY,
    PROVIDER_ZAI,
    PROVIDER_MISTRAL,
    PROVIDER_MINIMAX,
    PROVIDER_MINIMAX_CN,
    PROVIDER_MOONSHOTAI,
    PROVIDER_MOONSHOTAI_CN,
    PROVIDER_HUGGINGFACE,
    PROVIDER_FIREWORKS,
    PROVIDER_OPENCODE,
    PROVIDER_OPENCODE_GO,
    PROVIDER_KIMI_CODING,
    PROVIDER_CLOUDFLARE_WORKERS_AI,
    PROVIDER_CLOUDFLARE_AI_GATEWAY,
    PROVIDER_XIAOMI,
    PROVIDER_XIAOMI_TOKEN_PLAN_CN,
    PROVIDER_XIAOMI_TOKEN_PLAN_AMS,
    PROVIDER_XIAOMI_TOKEN_PLAN_SGP,
];

// ---------------------------------------------------------------------------
// Thinking levels
// ---------------------------------------------------------------------------

pub type ThinkingLevel = String;
pub type ModelThinkingLevel = String;

pub const THINKING_LEVEL_MINIMAL: &str = "minimal";
pub const THINKING_LEVEL_LOW: &str = "low";
pub const THINKING_LEVEL_MEDIUM: &str = "medium";
pub const THINKING_LEVEL_HIGH: &str = "high";
pub const THINKING_LEVEL_XHIGH: &str = "xhigh";
pub const THINKING_LEVEL_MAX: &str = "max";
pub const MODEL_THINKING_LEVEL_OFF: &str = "off";

pub const THINKING_LEVELS: [&str; 6] = [
    THINKING_LEVEL_MINIMAL,
    THINKING_LEVEL_LOW,
    THINKING_LEVEL_MEDIUM,
    THINKING_LEVEL_HIGH,
    THINKING_LEVEL_XHIGH,
    THINKING_LEVEL_MAX,
];

/// `Partial<Record<ModelThinkingLevel, string | null>>`.
/// `None` value = the level is absent from the map, `Some(None)` = explicit `null`
/// (level unsupported), `Some(Some(v))` = mapped provider value.
pub type ThinkingLevelMap = IndexMap<String, Option<String>>;

/// Token budgets for each thinking level (token-based providers only).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ThinkingBudgets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimal: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<f64>,
}

// ---------------------------------------------------------------------------
// Transport / cache / tier
// ---------------------------------------------------------------------------

pub type CacheRetention = String;
pub const CACHE_RETENTION_NONE: &str = "none";
pub const CACHE_RETENTION_SHORT: &str = "short";
pub const CACHE_RETENTION_LONG: &str = "long";

pub type Transport = String;
pub const TRANSPORT_SSE: &str = "sse";
pub const TRANSPORT_WEBSOCKET: &str = "websocket";
pub const TRANSPORT_WEBSOCKET_CACHED: &str = "websocket-cached";
pub const TRANSPORT_AUTO: &str = "auto";

/// `ServiceTier = "auto" | "default" | "flex" | "scale" | "priority" | null`.
/// `None` means the key is absent; `Some(None)` is an explicit JSON `null`.
pub type ServiceTier = Option<Option<String>>;

fn deserialize_optional_nullable<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

// ---------------------------------------------------------------------------
// Stream options
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProviderResponse {
    pub status: i64,
    pub headers: IndexMap<String, String>,
}

/// Content-free raw token observation for process-local performance metrics.
///
/// The TypeScript fields are `number | null` and optional. `None` = absent,
/// `Some(None)` = explicit JSON `null`, `Some(Some(n))` = number.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderUsageObservation {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_optional_nullable")]
    pub input_tokens: Option<Option<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_optional_nullable")]
    pub cached_input_tokens: Option<Option<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_optional_nullable")]
    pub output_tokens: Option<Option<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_optional_nullable")]
    pub reasoning_tokens: Option<Option<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_optional_nullable")]
    pub total_tokens: Option<Option<f64>>,
    /// True when raw inputTokens already includes cachedInputTokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_optional_nullable")]
    pub cached_input_included_in_input: Option<Option<bool>>,
    /// True when raw outputTokens already includes reasoningTokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_optional_nullable")]
    pub reasoning_included_in_output: Option<Option<bool>>,
}

/// `onPayload?: (payload: unknown, model: Model<Api>) => unknown | undefined | Promise<...>`
pub type OnPayload = Arc<dyn Fn(Value, &Model) -> BoxFuture<Option<Value>> + Send + Sync>;
/// `onResponse?: (response: ProviderResponse, model: Model<Api>) => void | Promise<void>`
pub type OnResponse = Arc<dyn Fn(ProviderResponse, &Model) -> BoxFuture<()> + Send + Sync>;
/// `onUsageObservation?: (observation, model) => void | Promise<void>`
pub type OnUsageObservation =
    Arc<dyn Fn(ProviderUsageObservation, &Model) -> BoxFuture<()> + Send + Sync>;
/// Content-free local transport phases; never sent in provider requests.
pub type OnStreamObservation = Arc<dyn Fn(&str) + Send + Sync>;

#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<f64>,
    /// `signal?: AbortSignal`
    #[serde(skip)]
    pub signal: Option<CancellationToken>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Preferred transport for providers that support multiple transports.
    /// Providers that do not support this option ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_optional_nullable")]
    pub service_tier: ServiceTier,
    /// Prompt cache retention preference. Providers map this to their supported
    /// values. Default: "short".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
    /// Optional session identifier for providers that support session-based caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional callback for inspecting or replacing provider payloads before sending.
    /// Return None to keep the payload unchanged.
    #[serde(skip)]
    pub on_payload: Option<OnPayload>,
    /// Optional callback invoked after an HTTP response is received and before
    /// its body stream is consumed.
    #[serde(skip)]
    pub on_response: Option<OnResponse>,
    /// Local observation only. Providers must never serialize this callback.
    #[serde(skip)]
    pub on_usage_observation: Option<OnUsageObservation>,
    #[serde(skip)]
    pub on_stream_observation: Option<OnStreamObservation>,
    /// Optional custom HTTP headers to include in API requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    /// HTTP request timeout in milliseconds for providers/SDKs that support it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<f64>,
    /// Optional metadata to include in API requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
    /// `ProviderStreamOptions = StreamOptions & Record<string, unknown>`: extra
    /// keys that no first-party option claims.
    #[serde(flatten, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

impl std::fmt::Debug for StreamOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamOptions")
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("signal", &self.signal)
            .field("api_key", &self.api_key)
            .field("transport", &self.transport)
            .field("service_tier", &self.service_tier)
            .field("cache_retention", &self.cache_retention)
            .field("session_id", &self.session_id)
            .field(
                "on_payload",
                &self.on_payload.as_ref().map(|_| "<callback>"),
            )
            .field(
                "on_response",
                &self.on_response.as_ref().map(|_| "<callback>"),
            )
            .field(
                "on_usage_observation",
                &self.on_usage_observation.as_ref().map(|_| "<callback>"),
            )
            .field("headers", &self.headers)
            .field("timeout_ms", &self.timeout_ms)
            .field("metadata", &self.metadata)
            .field("extra", &self.extra)
            .finish()
    }
}

impl PartialEq for StreamOptions {
    fn eq(&self, other: &Self) -> bool {
        self.temperature == other.temperature
            && self.max_tokens == other.max_tokens
            && self.api_key == other.api_key
            && self.transport == other.transport
            && self.service_tier == other.service_tier
            && self.cache_retention == other.cache_retention
            && self.session_id == other.session_id
            && self.headers == other.headers
            && self.timeout_ms == other.timeout_ms
            && self.metadata == other.metadata
            && self.extra == other.extra
    }
}

/// `ProviderStreamOptions = StreamOptions & Record<string, unknown>`.
pub type ProviderStreamOptions = StreamOptions;

/// Unified options with reasoning passed to streamSimple() and completeSimple().
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimpleStreamOptions {
    #[serde(flatten)]
    pub stream: StreamOptions,
    /// Explicit model reasoning selection. Omit to preserve the provider default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ModelThinkingLevel>,
    /// Custom token budgets for thinking levels (token-based providers only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budgets: Option<ThinkingBudgets>,
}

/// `StreamFunction<TApi, TOptions>`.
///
/// Contract:
/// - Must return an AssistantMessageEventStream.
/// - Once invoked, request/model/runtime failures should be encoded in the
///   returned stream, not thrown.
/// - Error termination must produce an AssistantMessage with stopReason
///   "error" or "aborted" and errorMessage, emitted via the stream protocol.
pub type StreamFunction = Arc<
    dyn Fn(&Model, &Context, Option<&StreamOptions>) -> AssistantMessageEventStream + Send + Sync,
>;

/// `CompactFunction<TApi>`: undefined means unsupported; failures must leave the
/// caller's history intact.
pub type CompactFunction = Arc<
    dyn Fn(
            &Model,
            &Context,
            Option<&CompactionOptions>,
        ) -> BoxFuture<Option<ProviderCompactionResult>>
        + Send
        + Sync,
>;

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextSignatureV1 {
    pub v: i64,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

pub const TEXT_CONTENT_TYPE: &str = "text";
pub const THINKING_CONTENT_TYPE: &str = "thinking";
pub const IMAGE_CONTENT_TYPE: &str = "image";
pub const TOOL_CALL_TYPE: &str = "toolCall";

fn role_user() -> String {
    ROLE_USER.to_string()
}

fn role_assistant() -> String {
    ROLE_ASSISTANT.to_string()
}

fn role_tool_result() -> String {
    ROLE_TOOL_RESULT.to_string()
}

fn text_content_type() -> String {
    TEXT_CONTENT_TYPE.to_string()
}

fn thinking_content_type() -> String {
    THINKING_CONTENT_TYPE.to_string()
}

fn image_content_type() -> String {
    IMAGE_CONTENT_TYPE.to_string()
}

fn tool_call_type() -> String {
    TOOL_CALL_TYPE.to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    #[serde(rename = "type", default = "text_content_type")]
    pub type_: String,
    pub text: String,
    /// e.g., for OpenAI responses, message metadata (legacy id string or
    /// TextSignatureV1 JSON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

impl TextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            type_: TEXT_CONTENT_TYPE.to_string(),
            text: text.into(),
            text_signature: None,
        }
    }
}

impl Default for TextContent {
    fn default() -> Self {
        Self::new("")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    #[serde(rename = "type", default = "thinking_content_type")]
    pub type_: String,
    pub thinking: String,
    /// e.g., for OpenAI responses, the reasoning item ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// When true, the thinking content was redacted by safety filters. The opaque
    /// encrypted payload is stored in `thinking_signature` so it can be passed back
    /// to the API for multi-turn continuity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}

impl ThinkingContent {
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            type_: THINKING_CONTENT_TYPE.to_string(),
            thinking: thinking.into(),
            thinking_signature: None,
            redacted: None,
        }
    }
}

impl Default for ThinkingContent {
    fn default() -> Self {
        Self::new("")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageContent {
    #[serde(rename = "type", default = "image_content_type")]
    pub type_: String,
    /// base64 encoded image data
    pub data: String,
    /// e.g., "image/jpeg", "image/png"
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

impl ImageContent {
    pub fn new(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            type_: IMAGE_CONTENT_TYPE.to_string(),
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    #[serde(rename = "type", default = "tool_call_type")]
    pub type_: String,
    pub id: String,
    pub name: String,
    pub arguments: Map<String, Value>,
    /// Google-specific: opaque signature for reusing thought context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: Map<String, Value>,
    ) -> Self {
        Self {
            type_: TOOL_CALL_TYPE.to_string(),
            id: id.into(),
            name: name.into(),
            arguments,
            thought_signature: None,
        }
    }
}

impl Default for ToolCall {
    fn default() -> Self {
        Self::new("", "", Map::new())
    }
}

/// `(TextContent | ThinkingContent | ToolCall)` for assistant messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text(TextContent),
    #[serde(rename = "thinking")]
    Thinking(ThinkingContent),
    #[serde(rename = "toolCall")]
    ToolCall(ToolCall),
}

impl ContentBlock {
    pub fn as_text(&self) -> Option<&TextContent> {
        match self {
            ContentBlock::Text(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_thinking(&self) -> Option<&ThinkingContent> {
        match self {
            ContentBlock::Thinking(thinking) => Some(thinking),
            _ => None,
        }
    }

    pub fn as_tool_call(&self) -> Option<&ToolCall> {
        match self {
            ContentBlock::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        }
    }

    pub fn content_type(&self) -> &'static str {
        match self {
            ContentBlock::Text(_) => TEXT_CONTENT_TYPE,
            ContentBlock::Thinking(_) => THINKING_CONTENT_TYPE,
            ContentBlock::ToolCall(_) => TOOL_CALL_TYPE,
        }
    }
}

/// `(TextContent | ImageContent)` for tool results and user message blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ImageOrTextContent {
    #[serde(rename = "text")]
    Text(TextContent),
    #[serde(rename = "image")]
    Image(ImageContent),
}

impl From<TextContent> for ImageOrTextContent {
    fn from(value: TextContent) -> Self {
        ImageOrTextContent::Text(value)
    }
}

impl From<ImageContent> for ImageOrTextContent {
    fn from(value: ImageContent) -> Self {
        ImageOrTextContent::Image(value)
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
    pub total: f64,
}

impl UsageCost {
    pub fn zero() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input: f64,
    pub output: f64,
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
    #[serde(rename = "totalTokens")]
    pub total_tokens: f64,
    pub cost: UsageCost,
}

impl Usage {
    /// `DEFAULT_USAGE` shape: all counters zero.
    pub fn zero() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Stop reasons
// ---------------------------------------------------------------------------

pub type StopReason = String;

pub const STOP_REASON_STOP: &str = "stop";
pub const STOP_REASON_LENGTH: &str = "length";
pub const STOP_REASON_TOOL_USE: &str = "toolUse";
pub const STOP_REASON_ERROR: &str = "error";
pub const STOP_REASON_ABORTED: &str = "aborted";

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

pub const ROLE_USER: &str = "user";
pub const ROLE_ASSISTANT: &str = "assistant";
pub const ROLE_TOOL_RESULT: &str = "toolResult";

/// `content: string | (TextContent | ImageContent)[]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<ImageOrTextContent>),
}

impl Default for UserContent {
    fn default() -> Self {
        UserContent::Text(String::new())
    }
}

impl UserContent {
    pub fn text(&self) -> String {
        match self {
            UserContent::Text(text) => text.clone(),
            UserContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    ImageOrTextContent::Text(text) => Some(text.text.clone()),
                    ImageOrTextContent::Image(_) => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    #[serde(default = "role_user")]
    pub role: String,
    pub content: UserContent,
    /// Provider-owned replacement for this message, supplied by a durable
    /// compaction checkpoint.
    #[serde(rename = "providerContext", skip_serializing_if = "Option::is_none")]
    pub provider_context: Option<ProviderCompactionCheckpoint>,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
}

impl UserMessage {
    pub fn new(content: UserContent, timestamp: i64) -> Self {
        Self {
            role: ROLE_USER.to_string(),
            content,
            provider_context: None,
            timestamp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    #[serde(default = "role_assistant")]
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub api: Api,
    pub provider: Provider,
    pub model: String,
    /// Concrete `chunk.model` when different from the requested `model`
    /// (e.g. OpenRouter `auto` -> `anthropic/...`).
    #[serde(rename = "responseModel", skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// Provider-specific response/message identifier when the upstream API exposes one.
    #[serde(rename = "responseId", skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Redacted provider/runtime diagnostics for failures and recoveries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    pub usage: Usage,
    #[serde(rename = "stopReason")]
    pub stop_reason: StopReason,
    /// Provider's raw finish/refusal signal when available (e.g. "refusal", "SAFETY").
    #[serde(rename = "stopReasonRaw", skip_serializing_if = "Option::is_none")]
    pub stop_reason_raw: Option<String>,
    #[serde(rename = "errorMessage", skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
}

impl Default for AssistantMessage {
    fn default() -> Self {
        Self {
            role: ROLE_ASSISTANT.to_string(),
            content: Vec::new(),
            api: String::new(),
            provider: String::new(),
            model: String::new(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: STOP_REASON_STOP.to_string(),
            stop_reason_raw: None,
            error_message: None,
            timestamp: 0,
        }
    }
}

impl AssistantMessage {
    pub fn new(
        api: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
        timestamp: i64,
    ) -> Self {
        Self {
            api: api.into(),
            provider: provider.into(),
            model: model.into(),
            timestamp,
            ..Default::default()
        }
    }
}

impl crate::utils::diagnostics::HasDiagnostics for AssistantMessage {
    fn diagnostics_field(&mut self) -> &mut Option<Vec<AssistantMessageDiagnostic>> {
        &mut self.diagnostics
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultMessage {
    #[serde(default = "role_tool_result")]
    pub role: String,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    /// Supports text and images
    pub content: Vec<ImageOrTextContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(rename = "isError")]
    pub is_error: bool,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
}

impl ToolResultMessage {
    pub fn new(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: Vec<ImageOrTextContent>,
        is_error: bool,
        timestamp: i64,
    ) -> Self {
        Self {
            role: ROLE_TOOL_RESULT.to_string(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content,
            details: None,
            is_error,
            timestamp,
        }
    }
}

impl Default for ToolResultMessage {
    fn default() -> Self {
        Self::new("", "", Vec::new(), false, 0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "user")]
    User(UserMessage),
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    #[serde(rename = "toolResult")]
    ToolResult(ToolResultMessage),
}

impl Message {
    pub fn user(message: UserMessage) -> Self {
        Message::User(message)
    }

    pub fn assistant(message: AssistantMessage) -> Self {
        Message::Assistant(message)
    }

    pub fn tool_result(message: ToolResultMessage) -> Self {
        Message::ToolResult(message)
    }

    pub fn role(&self) -> &'static str {
        match self {
            Message::User(_) => ROLE_USER,
            Message::Assistant(_) => ROLE_ASSISTANT,
            Message::ToolResult(_) => ROLE_TOOL_RESULT,
        }
    }

    pub fn as_user(&self) -> Option<&UserMessage> {
        match self {
            Message::User(message) => Some(message),
            _ => None,
        }
    }

    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            Message::Assistant(message) => Some(message),
            _ => None,
        }
    }

    pub fn as_assistant_mut(&mut self) -> Option<&mut AssistantMessage> {
        match self {
            Message::Assistant(message) => Some(message),
            _ => None,
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResultMessage> {
        match self {
            Message::ToolResult(message) => Some(message),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tools and context
// ---------------------------------------------------------------------------

/// `Tool<TParameters extends TSchema>`; `TSchema` becomes `serde_json::Value`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Context {
    #[serde(rename = "systemPrompt", skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl Context {
    pub fn new(
        system_prompt: Option<String>,
        messages: Vec<Message>,
        tools: Option<Vec<Tool>>,
    ) -> Self {
        Self {
            system_prompt,
            messages,
            tools,
        }
    }
}

// ---------------------------------------------------------------------------
// Stream event protocol
// ---------------------------------------------------------------------------

/// Event protocol for AssistantMessageEventStream.
///
/// Streams should emit `start` before partial updates, then terminate with either:
/// - `done` carrying the final successful AssistantMessage, or
/// - `error` carrying the final AssistantMessage with stopReason "error" or "aborted"
///   and errorMessage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantMessageEvent {
    #[serde(rename = "start")]
    Start { partial: AssistantMessage },
    #[serde(rename = "text_start")]
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "text_end")]
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        partial: AssistantMessage,
    },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "toolCall")]
        tool_call: ToolCall,
        partial: AssistantMessage,
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

impl AssistantMessageEvent {
    pub fn event_type(&self) -> &'static str {
        match self {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::ToolCallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolCallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolCallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        }
    }

    /// The `partial` message carried by every non-terminal event.
    pub fn partial(&self) -> Option<&AssistantMessage> {
        match self {
            AssistantMessageEvent::Start { partial }
            | AssistantMessageEvent::TextStart { partial, .. }
            | AssistantMessageEvent::TextDelta { partial, .. }
            | AssistantMessageEvent::TextEnd { partial, .. }
            | AssistantMessageEvent::ThinkingStart { partial, .. }
            | AssistantMessageEvent::ThinkingDelta { partial, .. }
            | AssistantMessageEvent::ThinkingEnd { partial, .. }
            | AssistantMessageEvent::ToolCallStart { partial, .. }
            | AssistantMessageEvent::ToolCallDelta { partial, .. }
            | AssistantMessageEvent::ToolCallEnd { partial, .. } => Some(partial),
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Compatibility settings
// ---------------------------------------------------------------------------

pub const THINKING_FORMAT_OPENAI: &str = "openai";
pub const THINKING_FORMAT_OPENROUTER: &str = "openrouter";
pub const THINKING_FORMAT_DEEPSEEK: &str = "deepseek";
pub const THINKING_FORMAT_ZAI: &str = "zai";
pub const THINKING_FORMAT_QWEN: &str = "qwen";
pub const THINKING_FORMAT_QWEN_CHAT_TEMPLATE: &str = "qwen-chat-template";

pub const MAX_TOKENS_FIELD_MAX_COMPLETION_TOKENS: &str = "max_completion_tokens";
pub const MAX_TOKENS_FIELD_MAX_TOKENS: &str = "max_tokens";

/// Compatibility settings for OpenAI-compatible completions APIs.
/// Use this to override URL-based auto-detection for custom providers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAICompletionsCompat {
    /// Whether the provider supports the `store` field. Default: auto-detected from URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_store: Option<bool>,
    /// Whether the provider supports the `developer` role (vs `system`). Default: auto-detected from URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    /// Whether the provider supports `reasoning_effort`. Default: auto-detected from URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    /// Whether the provider supports `stream_options: { include_usage: true }` for token
    /// usage in streaming responses. Default: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    /// Which field to use for max tokens. Default: auto-detected from URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<String>,
    /// Whether tool results require the `name` field. Default: auto-detected from URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_tool_result_name: Option<bool>,
    /// Whether a user message after tool results requires an assistant message in between.
    /// Default: auto-detected from URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_assistant_after_tool_result: Option<bool>,
    /// Whether thinking blocks must be converted to text blocks with <thinking> delimiters.
    /// Default: auto-detected from URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_thinking_as_text: Option<bool>,
    /// Whether all replayed assistant messages must include an empty reasoning_content field
    /// when reasoning is enabled. Default: auto-detected from URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    /// Format for reasoning/thinking parameter. Default: "openai".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_format: Option<String>,
    /// OpenRouter-specific routing preferences. Only used when baseUrl points to OpenRouter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_router_routing: Option<OpenRouterRouting>,
    /// Vercel AI Gateway routing preferences. Only used when baseUrl points to Vercel AI Gateway.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vercel_gateway_routing: Option<VercelGatewayRouting>,
    /// Whether z.ai supports top-level `tool_stream: true` for streaming tool call deltas.
    /// Default: false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zai_tool_stream: Option<bool>,
    /// Whether the provider supports the `strict` field in tool definitions. Default: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    /// Cache control convention for prompt caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control_format: Option<String>,
    /// Whether to send known session-affinity headers from `options.sessionId` when
    /// caching is enabled. Default: false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    /// Whether the provider supports long prompt cache retention. Default: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
}

/// Validation state for an exact native compaction route. Only `live-verified` may execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeCompactionValidation {
    Unverified,
    DocumentationVerified,
    LiveVerified,
}

pub const NATIVE_COMPACTION_PROTOCOL_OPENAI_RESPONSES_COMPACT_V1: &str =
    "openai-responses-compact-v1";
pub const NATIVE_COMPACTION_API_VERSION_V1: &str = "v1";

/// Explicit capability for an OpenAI Responses v1 compact endpoint.
///
/// This is model scoped on purpose. A shared serializer, display name, or
/// provider-wide boolean is not evidence that another route implements this protocol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeCompactionCapability {
    /// `"openai-responses-compact-v1"`
    pub protocol: String,
    pub provider: String,
    pub model: String,
    pub endpoint: String,
    /// `"v1"`
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub enabled: bool,
    pub validation: NativeCompactionValidation,
}

/// Compatibility settings for OpenAI Responses APIs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAIResponsesCompat {
    /// Whether to send the OpenAI `session_id` cache-affinity header from
    /// `options.sessionId` when caching is enabled. Default: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_id_header: Option<bool>,
    /// Whether the provider supports `prompt_cache_retention: "24h"`. Default: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
}

/// Compatibility settings for Anthropic Messages-compatible APIs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicMessagesCompat {
    /// Whether the provider accepts per-tool `eager_input_streaming`.
    /// When false, the Anthropic provider omits `tools[].eager_input_streaming`
    /// and sends the legacy `fine-grained-tool-streaming-2025-05-14` beta header
    /// for tool-enabled requests. Default: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    /// Whether the provider supports Anthropic long cache retention. Default: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
}

/// `TApi extends "openai-completions" ? OpenAICompletionsCompat : ...`
///
/// The TypeScript picks the compat shape from the model's `api` value, so the
/// Rust port sniffs the incoming keys instead of relying on `untagged`, which
/// would always match `OpenAICompletionsCompat` first (every field is optional).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Compat {
    Completions(OpenAICompletionsCompat),
    Responses(OpenAIResponsesCompat),
    Anthropic(AnthropicMessagesCompat),
}

impl<'de> Deserialize<'de> for Compat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Some(object) = value.as_object() else {
            return Err(serde::de::Error::custom("expected a compat object"));
        };
        if object.contains_key("supportsEagerToolInputStreaming") {
            return serde_json::from_value(value)
                .map(Compat::Anthropic)
                .map_err(serde::de::Error::custom);
        }
        if object.contains_key("sendSessionIdHeader") {
            return serde_json::from_value(value)
                .map(Compat::Responses)
                .map_err(serde::de::Error::custom);
        }
        serde_json::from_value(value)
            .map(Compat::Completions)
            .map_err(serde::de::Error::custom)
    }
}

impl Compat {
    pub fn as_completions(&self) -> Option<&OpenAICompletionsCompat> {
        match self {
            Compat::Completions(compat) => Some(compat),
            _ => None,
        }
    }

    pub fn as_responses(&self) -> Option<&OpenAIResponsesCompat> {
        match self {
            Compat::Responses(compat) => Some(compat),
            _ => None,
        }
    }

    pub fn as_anthropic(&self) -> Option<&AnthropicMessagesCompat> {
        match self {
            Compat::Anthropic(compat) => Some(compat),
            _ => None,
        }
    }
}

/// Parse a `compat` value for a model whose `api` is `api`.
///
/// `Model<TApi>` (packages/ai/src/types.ts:494-500) picks the compat interface
/// from `api`:
///   - `"openai-completions"`                              -> `OpenAICompletionsCompat`
///   - `"openai-responses"` / `"openai-codex-responses"`   -> `OpenAIResponsesCompat`
///   - `"anthropic-messages"`                              -> `AnthropicMessagesCompat`
///   - any other `api`                                     -> `never`
///
/// TypeScript gets this for free from the type parameter, so a `models.json`
/// entry may omit the discriminator keys and still receive the right shape. The
/// Rust port has no type parameter, so it must derive the shape from `api`. The
/// `None` arm reproduces TS `never`: a compat object is not a valid value for any
/// other `api`, so it is rejected instead of being silently coerced into
/// `OpenAICompletionsCompat`.
pub fn parse_compat_for_api(api: &str, value: Value) -> Option<Compat> {
    let expected = match api {
        "openai-completions" => Compat::Completions(OpenAICompletionsCompat::default()),
        "openai-responses" | "openai-codex-responses" => {
            Compat::Responses(OpenAIResponsesCompat::default())
        }
        "anthropic-messages" => Compat::Anthropic(AnthropicMessagesCompat::default()),
        _ => return None,
    };
    coerce_compat(expected, value)
}

/// Re-shape `value` into the same compat variant as `target`.
fn coerce_compat(target: Compat, value: Value) -> Option<Compat> {
    match target {
        Compat::Completions(_) => serde_json::from_value(value).ok().map(Compat::Completions),
        Compat::Responses(_) => serde_json::from_value(value).ok().map(Compat::Responses),
        Compat::Anthropic(_) => serde_json::from_value(value).ok().map(Compat::Anthropic),
    }
}

/// OpenRouter provider routing preferences.
/// Controls which upstream providers OpenRouter routes requests to.
/// Sent as the `provider` field in the OpenRouter API request body.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterRouting {
    /// Whether to allow backup providers to serve requests. Default: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Whether to filter providers to only those that support all parameters in the
    /// request. Default: false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// Data collection setting. "allow" (default): allow providers that may
    /// store/train on data. "deny": only use providers that don't collect user data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<String>,
    /// Whether to restrict routing to only ZDR (Zero Data Retention) endpoints.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    /// Whether to restrict routing to only models that allow text distillation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enforce_distillable_text: Option<bool>,
    /// An ordered list of provider names/slugs to try in sequence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    /// List of provider names/slugs to exclusively allow for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// List of provider names/slugs to skip for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    /// A list of quantization levels to filter providers by.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<String>>,
    /// Sorting strategy. Can be a string or an object with `by` and `partition`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<OpenRouterSort>,
    /// Maximum price per million tokens (USD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_price: Option<OpenRouterMaxPrice>,
    /// Preferred minimum throughput (tokens/second).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<OpenRouterPercentiles>,
    /// Preferred maximum latency (seconds).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<OpenRouterPercentiles>,
}

/// `string | { by?: string; partition?: string | null }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenRouterSort {
    Name(String),
    Object {
        #[serde(skip_serializing_if = "Option::is_none")]
        by: Option<String>,
        /// `partition?: string | null`
        #[serde(skip_serializing_if = "Option::is_none")]
        partition: Option<Option<String>>,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterMaxPrice {
    /// `number | string`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Value>,
}

/// `number | { p50?, p75?, p90?, p99? }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenRouterPercentiles {
    Single(f64),
    Percentiles {
        #[serde(skip_serializing_if = "Option::is_none")]
        p50: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        p75: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        p90: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        p99: Option<f64>,
    },
}

/// Vercel AI Gateway routing preferences.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VercelGatewayRouting {
    /// List of provider slugs to exclusively use for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// List of provider slugs to try in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputModality {
    Text,
    Image,
}

impl InputModality {
    pub fn as_str(&self) -> &'static str {
        match self {
            InputModality::Text => "text",
            InputModality::Image => "image",
        }
    }
}

/// `cost: { input, output, cacheRead, cacheWrite }` in $/million tokens.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
}

impl ModelCost {
    pub fn zero() -> Self {
        Self::default()
    }
}

/// Model interface for the unified model system.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: Provider,
    pub base_url: String,
    pub reasoning: bool,
    /// Maps pi thinking levels to provider/model-specific values.
    /// Missing keys use provider defaults. null marks a level as unsupported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    pub input: Vec<InputModality>,
    pub cost: ModelCost,
    pub context_window: f64,
    /// Separate input ceiling when the provider reserves part of the context for output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<f64>,
    pub max_tokens: f64,
    /// Flagship model surfaced above non-featured models of the same provider in pickers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub featured: Option<bool>,
    /// Exact, explicitly validated native compaction route. Absent means unsupported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_compaction: Option<NativeCompactionCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    /// Compatibility overrides for OpenAI-compatible APIs. If not set,
    /// auto-detected from baseUrl.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<Compat>,
}

impl Model {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        api: impl Into<String>,
        provider: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            api: api.into(),
            provider: provider.into(),
            base_url: base_url.into(),
            ..Default::default()
        }
    }

    /// `model.compat` narrowed to `OpenAICompletionsCompat`.
    pub fn compat_completions(&self) -> Option<&OpenAICompletionsCompat> {
        self.compat.as_ref().and_then(Compat::as_completions)
    }

    /// `model.compat` narrowed to `OpenAIResponsesCompat`.
    pub fn compat_responses(&self) -> Option<&OpenAIResponsesCompat> {
        self.compat.as_ref().and_then(Compat::as_responses)
    }

    /// `model.compat` narrowed to `AnthropicMessagesCompat`.
    pub fn compat_anthropic(&self) -> Option<&AnthropicMessagesCompat> {
        self.compat.as_ref().and_then(Compat::as_anthropic)
    }

    /// `model.thinkingLevelMap?.[level]`: None = key absent, Some(None) = null.
    pub fn thinking_level_map_get(&self, level: &str) -> Option<Option<String>> {
        self.thinking_level_map
            .as_ref()
            .and_then(|map| map.get(level).cloned())
    }
}

/// `model.thinkingLevelMap?.[level]` as a free function.
pub fn thinking_level_map_get(map: &ThinkingLevelMap, level: &str) -> Option<Option<String>> {
    map.get(level).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stream_options_round_trip_the_typescript_wire_keys_and_nullable_tier() {
        let wire = json!({"maxTokens":128.0,"apiKey":"synthetic","sessionId":"session", "cacheRetention":"custom", "timeoutMs":500.0, "serviceTier":null, "reasoning":"high", "thinkingBudgets":{"high":64.0}, "providerSpecific":true});
        let options: SimpleStreamOptions = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(options.stream.max_tokens, Some(128.0));
        assert_eq!(options.stream.cache_retention.as_deref(), Some("custom"));
        assert_eq!(options.stream.service_tier, Some(None));
        assert_eq!(options.thinking_budgets.as_ref().unwrap().high, Some(64.0));
        assert_eq!(serde_json::to_value(&options).unwrap(), wire);
        let absent: StreamOptions = serde_json::from_value(json!({})).unwrap();
        assert_eq!(absent.service_tier, None);
        assert!(serde_json::to_value(absent)
            .unwrap()
            .get("serviceTier")
            .is_none());
    }

    #[test]
    fn content_signatures_keep_the_typescript_field_names() {
        let wire = json!({"role":"assistant", "api":"api", "provider":"provider", "model":"model", "content":[
            {"type":"text","text":"text","textSignature":"text-sig"},
            {"type":"thinking","thinking":"thought","thinkingSignature":"thinking-sig"},
            {"type":"toolCall","id":"call","name":"tool","arguments":{},"thoughtSignature":"tool-sig"}
        ],"usage":serde_json::to_value(Usage::default()).unwrap(),"stopReason":"stop","timestamp":1});
        let message: AssistantMessage = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(
            message.content[0]
                .as_text()
                .unwrap()
                .text_signature
                .as_deref(),
            Some("text-sig")
        );
        assert_eq!(
            message.content[1]
                .as_thinking()
                .unwrap()
                .thinking_signature
                .as_deref(),
            Some("thinking-sig")
        );
        assert_eq!(
            message.content[2]
                .as_tool_call()
                .unwrap()
                .thought_signature
                .as_deref(),
            Some("tool-sig")
        );
        assert_eq!(serde_json::to_value(message).unwrap(), wire);
    }

    #[test]
    fn assistant_message_serialises_camel_case_fields() {
        let message =
            AssistantMessage::new("anthropic-messages", "anthropic", "claude-sonnet-4", 1234);
        let value = serde_json::to_value(&message).unwrap();
        assert_eq!(value["role"], json!("assistant"));
        assert_eq!(value["api"], json!("anthropic-messages"));
        assert_eq!(value["stopReason"], json!("stop"));
        assert_eq!(value["timestamp"], json!(1234));
        assert_eq!(value["usage"]["cacheRead"], json!(0.0));
        assert_eq!(value["usage"]["totalTokens"], json!(0.0));
        assert!(value.get("responseModel").is_none());
        assert!(value.get("errorMessage").is_none());
    }

    #[test]
    fn user_message_content_union_round_trips() {
        let text = UserMessage::new(UserContent::Text("hi".to_string()), 1);
        assert_eq!(
            serde_json::to_value(&text).unwrap(),
            json!({"role": "user", "content": "hi", "timestamp": 1})
        );

        let blocks = UserMessage::new(
            UserContent::Blocks(vec![ImageOrTextContent::Text(TextContent::new("a"))]),
            2,
        );
        assert_eq!(
            serde_json::to_value(&blocks).unwrap(),
            json!({"role": "user", "content": [{"type": "text", "text": "a"}], "timestamp": 2})
        );
    }

    #[test]
    fn message_enum_uses_role_tag() {
        let message = Message::ToolResult(ToolResultMessage::new(
            "call-1",
            "bash",
            Vec::new(),
            true,
            5,
        ));
        let value = serde_json::to_value(&message).unwrap();
        assert_eq!(value["role"], json!("toolResult"));
        assert_eq!(value["toolCallId"], json!("call-1"));
        assert_eq!(value["isError"], json!(true));
    }

    #[test]
    fn content_block_tag_round_trip() {
        let block = ContentBlock::ToolCall(ToolCall::new("id", "bash", Map::new()));
        let value = serde_json::to_value(&block).unwrap();
        assert_eq!(value["type"], json!("toolCall"));
        let back: ContentBlock = serde_json::from_value(value).unwrap();
        assert_eq!(block, back);
    }

    #[test]
    fn assistant_message_event_tags_match_typescript() {
        let partial = AssistantMessage::default();
        let event = AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "x".to_string(),
            partial: partial.clone(),
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["type"], json!("text_delta"));
        assert_eq!(value["contentIndex"], json!(0));
        assert_eq!(value["delta"], json!("x"));

        let done = AssistantMessageEvent::Done {
            reason: STOP_REASON_STOP.to_string(),
            message: partial.clone(),
        };
        assert_eq!(serde_json::to_value(&done).unwrap()["type"], json!("done"));

        let tool_end = AssistantMessageEvent::ToolCallEnd {
            content_index: 2,
            tool_call: ToolCall::default(),
            partial,
        };
        let value = serde_json::to_value(&tool_end).unwrap();
        assert_eq!(value["type"], json!("toolcall_end"));
        assert_eq!(value["toolCall"]["type"], json!("toolCall"));
    }

    #[test]
    fn model_serialises_expected_field_names() {
        let mut model = Model::new(
            "gpt-5",
            "GPT-5",
            API_OPENAI_RESPONSES,
            PROVIDER_OPENAI,
            "https://api.openai.com/v1",
        );
        model.reasoning = true;
        model.input = vec![InputModality::Text, InputModality::Image];
        model.context_window = 400000.0;
        model.max_tokens = 128000.0;
        let value = serde_json::to_value(&model).unwrap();
        assert_eq!(value["baseUrl"], json!("https://api.openai.com/v1"));
        assert_eq!(value["contextWindow"], json!(400000.0));
        assert_eq!(value["maxTokens"], json!(128000.0));
        assert_eq!(value["input"], json!(["text", "image"]));
        assert!(value.get("thinkingLevelMap").is_none());
        assert!(value.get("nativeCompaction").is_none());
    }

    #[test]
    fn compat_untagged_round_trip() {
        let completions: Compat = serde_json::from_value(json!({
            "supportsDeveloperRole": false,
            "thinkingFormat": "zai",
            "zaiToolStream": true
        }))
        .unwrap();
        assert_eq!(
            completions
                .as_completions()
                .unwrap()
                .thinking_format
                .as_deref(),
            Some("zai")
        );

        let responses: Compat =
            serde_json::from_value(json!({"sendSessionIdHeader": false})).unwrap();
        assert_eq!(
            responses.as_responses().unwrap().send_session_id_header,
            Some(false)
        );
    }

    #[test]
    fn native_compaction_validation_kebab_case() {
        assert_eq!(
            serde_json::to_value(NativeCompactionValidation::LiveVerified).unwrap(),
            json!("live-verified")
        );
        assert_eq!(
            serde_json::to_value(NativeCompactionValidation::DocumentationVerified).unwrap(),
            json!("documentation-verified")
        );
    }

    #[test]
    fn thinking_level_map_distinguishes_absent_and_null() {
        let mut map = ThinkingLevelMap::new();
        map.insert(THINKING_LEVEL_HIGH.to_string(), None);
        map.insert(THINKING_LEVEL_LOW.to_string(), Some("low".to_string()));
        let mut model = Model::default();
        model.thinking_level_map = Some(map);
        assert_eq!(
            model.thinking_level_map_get(THINKING_LEVEL_HIGH),
            Some(None)
        );
        assert_eq!(
            model.thinking_level_map_get(THINKING_LEVEL_LOW),
            Some(Some("low".to_string()))
        );
        assert_eq!(model.thinking_level_map_get(THINKING_LEVEL_MAX), None);
    }

    #[test]
    fn provider_usage_observation_null_vs_absent() {
        let observation = ProviderUsageObservation {
            input_tokens: Some(None),
            output_tokens: Some(Some(12.0)),
            ..Default::default()
        };
        let value = serde_json::to_value(&observation).unwrap();
        assert_eq!(value, json!({"inputTokens": null, "outputTokens": 12.0}));
    }
}
