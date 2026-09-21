//! Port of packages/agent/src/types.ts
//!
//! The TypeScript `CustomAgentMessages` interface is augmented by
//! packages/coding-agent/src/core/messages.ts via declaration merging, so
//! `AgentMessage = Message | CustomAgentMessages[keyof CustomAgentMessages]`.
//! In Rust there is no declaration merging, so [`AgentMessage`] is an enum with
//! the same member set: the three pi-ai message shapes plus the four custom
//! message kinds that the coding agent registers (`bashExecution`, `custom`,
//! `branchSummary`, `compactionSummary`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use indexmap::IndexMap;
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, ImageContent, Message, Model, ServiceTier,
    SimpleStreamOptions, TextContent, ToolResultMessage,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::performance_metrics::AgentLoopPerformanceMetrics;

/// Type of the streaming entry point used by the agent loop (TypeScript `StreamFn`).
///
/// Contract:
/// - Must not throw or return a rejected promise for request/model/runtime failures.
/// - Must return an `AssistantMessageEventStream`.
/// - Failures must be encoded in the returned stream via protocol events and a
///   final `AssistantMessage` with `stopReason` "error" or "aborted" and `errorMessage`.
pub type StreamFn = Arc<
    dyn Fn(
            Model,
            pi_ai::types::Context,
            SimpleStreamOptions,
        ) -> futures::future::BoxFuture<'static, pi_ai::utils::event_stream::AssistantMessageEventStream>
        + Send
        + Sync,
>;

/// Configuration for how tool calls from a single assistant message are executed.
///
/// - `Sequential`: each tool call is prepared, executed, and finalized before the next one starts.
/// - `Parallel`: tool calls are prepared sequentially, then allowed tools execute concurrently.
///   `tool_execution_end` is emitted in tool completion order after each tool is finalized,
///   while tool-result message artifacts are emitted later in assistant source order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolExecutionMode {
    Sequential,
    Parallel,
}

/// A tool-call content block emitted by an assistant message.
pub type AgentToolCall = pi_ai::types::ToolCall;

/// Result returned from `beforeToolCall`.
///
/// Returning `{ block: true }` prevents the tool from executing. The loop emits an
/// error tool result instead. `reason` becomes the text shown in that error result.
/// If omitted, a default blocked message is used.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BeforeToolCallResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Partial override returned from `afterToolCall`.
///
/// Merge semantics are field-by-field:
/// - `content`: if provided, replaces the tool result content array in full
/// - `details`: if provided, replaces the tool result details value in full
/// - `isError`: if provided, replaces the tool result error flag
/// - `terminate`: if provided, replaces the early-termination hint
///
/// Omitted fields keep the original executed tool result values.
/// There is no deep merge for `content` or `details`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AfterToolCallResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentBlock>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// Hint that the agent should stop after the current tool batch.
    /// Early termination only happens when every finalized tool result in the batch sets this to true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminate: Option<bool>,
}

/// Context passed to `beforeToolCall` after arguments are validated.
#[derive(Clone)]
pub struct BeforeToolCallContext {
    /// Assistant message that requested the call.
    pub assistant_message: AssistantMessage,
    /// Raw tool-call block from `assistantMessage.content`.
    pub tool_call: AgentToolCall,
    /// Validated arguments for the target tool schema.
    pub args: Value,
    /// Agent context when this call is prepared.
    pub context: AgentContext,
}

/// Context passed to `afterToolCall`.
#[derive(Clone)]
pub struct AfterToolCallContext {
    /// Assistant message that requested the call.
    pub assistant_message: AssistantMessage,
    /// Raw tool-call block from `assistantMessage.content`.
    pub tool_call: AgentToolCall,
    /// Validated arguments for the target tool schema.
    pub args: Value,
    /// Executed result before any `afterToolCall` overrides.
    pub result: AgentToolResult,
    /// Whether the executed result is currently treated as an error.
    pub is_error: bool,
    /// Agent context when this call is finalized.
    pub context: AgentContext,
}

/// Context passed to `shouldStopAfterTurn` and `getContinuationMessages`.
#[derive(Clone)]
pub struct ShouldStopAfterTurnContext {
    /// Assistant message that completed the turn.
    pub message: AssistantMessage,
    /// Tool-result messages included in the preceding `turn_end` event.
    pub tool_results: Vec<ToolResultMessage>,
    /// Context after appending the turn's assistant message and tool results.
    pub context: AgentContext,
    /// Messages returned by this invocation; prompts include initial prompts,
    /// continuations exclude prior context.
    pub new_messages: Vec<AgentMessage>,
}

pub type GetContinuationMessagesContext = ShouldStopAfterTurnContext;

/// Thinking/reasoning level for models that support it.
///
/// Note: "xhigh" and "max" are only supported by selected model families. Use the
/// model thinking-level metadata from pi-ai to detect support for a concrete model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl Default for ThinkingLevel {
    fn default() -> Self {
        ThinkingLevel::Off
    }
}

impl ThinkingLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingLevel::Off => "off",
            ThinkingLevel::Minimal => "minimal",
            ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
            ThinkingLevel::Xhigh => "xhigh",
            ThinkingLevel::Max => "max",
        }
    }
}

/// Final or partial result produced by a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentToolResult {
    /// Text or image content returned to the model.
    pub content: Vec<ContentBlock>,
    /// Structured details for logs or UI rendering.
    pub details: Value,
    /// Hint that the agent should stop after the current tool batch.
    /// Early termination only happens when every finalized tool result in the batch sets this to true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminate: Option<bool>,
}

impl AgentToolResult {
    pub fn new(content: Vec<ContentBlock>, details: Value) -> Self {
        Self {
            content,
            details,
            terminate: None,
        }
    }
}

/// Text or image content, the only two block kinds a tool result can carry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentBlock {
    Text(TextContent),
    Image(ImageContent),
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text(TextContent::new(text))
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            ContentBlock::Image(_) => None,
        }
    }
}

/// Callback used by tools to publish partial execution updates.
pub type AgentToolUpdateCallback = Arc<dyn Fn(AgentToolResult) + Send + Sync>;

/// Tool definition used by the agent runtime.
#[derive(Clone)]
pub struct AgentTool {
    pub name: String,
    pub description: String,
    /// TypeBox/JSON schema for the tool parameters.
    pub parameters: Value,
    /// Human-readable label for UI display.
    pub label: String,
    /// Optional compatibility shim for raw tool-call arguments before schema validation.
    /// Must return an object that matches `TParameters`.
    pub prepare_arguments: Option<Arc<dyn Fn(Value) -> Value + Send + Sync>>,
    /// Execute the tool call. Return an error instead of encoding errors in `content`.
    pub execute: Arc<
        dyn Fn(
                String,
                Value,
                Option<tokio_util::sync::CancellationToken>,
                Option<AgentToolUpdateCallback>,
            ) -> futures::future::BoxFuture<'static, Result<AgentToolResult, anyhow::Error>>
            + Send
            + Sync,
    >,
    /// Per-tool execution mode override.
    pub execution_mode: Option<ToolExecutionMode>,
}

impl std::fmt::Debug for AgentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("label", &self.label)
            .field("execution_mode", &self.execution_mode)
            .finish_non_exhaustive()
    }
}

/// Context snapshot passed to the low-level agent loop and tool hooks.
#[derive(Debug, Clone, Default)]
pub struct AgentContext {
    /// System prompt included with the request.
    pub system_prompt: String,
    /// Transcript visible to the model.
    pub messages: Vec<AgentMessage>,
    /// Tools available for this run.
    pub tools: Option<Vec<AgentTool>>,
}

/// Extensible custom message kinds registered by the coding agent
/// (TypeScript declaration merging of `CustomAgentMessages`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum CustomAgentMessage {
    /// Message type for bash executions via the ! command.
    #[serde(rename = "bashExecution")]
    BashExecution {
        command: String,
        output: String,
        #[serde(rename = "exitCode")]
        exit_code: Option<i64>,
        cancelled: bool,
        truncated: bool,
        #[serde(rename = "fullOutputPath", default, skip_serializing_if = "Option::is_none")]
        full_output_path: Option<String>,
        timestamp: i64,
        /// If true, this message is excluded from LLM context (!! prefix)
        #[serde(rename = "excludeFromContext", default, skip_serializing_if = "Option::is_none")]
        exclude_from_context: Option<bool>,
    },
    /// Message type for extension-injected messages via sendMessage().
    #[serde(rename = "custom")]
    Custom {
        #[serde(rename = "customType")]
        custom_type: String,
        content: CustomMessageContent,
        display: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        timestamp: i64,
    },
    #[serde(rename = "branchSummary")]
    BranchSummary {
        summary: String,
        #[serde(rename = "fromId")]
        from_id: String,
        timestamp: i64,
    },
    #[serde(rename = "compactionSummary")]
    CompactionSummary {
        summary: String,
        /// Complete opaque provider window, used instead of the display summary.
        #[serde(rename = "providerContext", default, skip_serializing_if = "Option::is_none")]
        provider_context: Option<pi_ai::compaction::ProviderCompactionCheckpoint>,
        #[serde(rename = "tokensBefore")]
        tokens_before: f64,
        /// Number of retained messages that precede this summary in transcript presentation.
        #[serde(rename = "retainedMessageCount", default, skip_serializing_if = "Option::is_none")]
        retained_message_count: Option<f64>,
        /// User instructions that guided the summary (from `/compact <instructions>`)
        #[serde(rename = "customInstructions", default, skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
        /// Harness digest snapshot rendered before the summary in LLM context.
        #[serde(rename = "harnessDigest", default, skip_serializing_if = "Option::is_none")]
        harness_digest: Option<String>,
        timestamp: i64,
    },
}

impl CustomAgentMessage {
    pub fn role(&self) -> &'static str {
        match self {
            CustomAgentMessage::BashExecution { .. } => "bashExecution",
            CustomAgentMessage::Custom { .. } => "custom",
            CustomAgentMessage::BranchSummary { .. } => "branchSummary",
            CustomAgentMessage::CompactionSummary { .. } => "compactionSummary",
        }
    }
}

/// `content: string | (TextContent | ImageContent)[]` for custom messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CustomMessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Union of the LLM-visible pi-ai messages and the coding agent custom messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentMessage {
    Message(Message),
    Custom(CustomAgentMessage),
}

impl AgentMessage {
    /// Mirrors `message.role` in TypeScript. `ToolResult` reads as "toolResult"
    /// and `Custom` reads as "custom", matching the declared discriminant.
    pub fn role(&self) -> &str {
        match self {
            AgentMessage::Message(message) => message.role(),
            AgentMessage::Custom(custom) => custom.role(),
        }
    }

    pub fn as_message(&self) -> Option<&Message> {
        match self {
            AgentMessage::Message(message) => Some(message),
            AgentMessage::Custom(_) => None,
        }
    }
}

impl From<Message> for AgentMessage {
    fn from(value: Message) -> Self {
        AgentMessage::Message(value)
    }
}

impl From<AssistantMessage> for AgentMessage {
    fn from(value: AssistantMessage) -> Self {
        AgentMessage::Message(Message::Assistant(value))
    }
}

impl From<pi_ai::types::UserMessage> for AgentMessage {
    fn from(value: pi_ai::types::UserMessage) -> Self {
        AgentMessage::Message(Message::User(value))
    }
}

impl From<ToolResultMessage> for AgentMessage {
    fn from(value: ToolResultMessage) -> Self {
        AgentMessage::Message(Message::ToolResult(value))
    }
}

/// Public agent state.
///
/// The TypeScript `AgentState` declares `tools` and `messages` as accessor
/// properties so implementations can copy assigned arrays before storing them.
/// Rust has no accessors, so the fields are plain and `Agent` copies on
/// assignment (see `crate::agent::Agent::set_state`).
#[derive(Clone)]
pub struct AgentState {
    /// System prompt sent with each model request.
    pub system_prompt: String,
    /// Model used for future turns.
    pub model: Model,
    /// Requested reasoning level for future turns.
    pub thinking_level: ThinkingLevel,
    /// Requested provider service tier for future turns.
    pub service_tier: ServiceTier,
    /// Available tools. Assigning a new list copies its top-level array.
    pub tools: Option<Vec<AgentTool>>,
    /// Conversation transcript. Assigning a new list copies its top-level array.
    pub messages: Vec<AgentMessage>,
    /// True while processing a prompt or continuation, including awaited `agent_end` listeners.
    pub is_streaming: bool,
    /// Partial assistant message for the active streamed response, if any.
    pub streaming_message: Option<AgentMessage>,
    /// Tool-call IDs currently executing.
    pub pending_tool_calls: BTreeSet<String>,
    /// Error from the most recent failed or aborted assistant turn, if any.
    pub error_message: Option<String>,
}

impl Default for AgentState {
    /// `createMutableAgentState()` defaults: empty prompt, `DEFAULT_MODEL`, "off",
    /// `"default"` service tier, empty tools/messages, idle runtime fields.
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            model: pi_ai::types::Model::new("unknown", "unknown", "unknown", "unknown", ""),
            thinking_level: ThinkingLevel::Off,
            service_tier: Some(Some("default".to_string())),
            tools: None,
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: BTreeSet::new(),
            error_message: None,
        }
    }
}

/// Events emitted by the Agent for UI updates.
///
/// `agent_end` is the last event emitted for a run, but awaited `Agent.subscribe()`
/// listeners for that event are still part of run settlement. The agent becomes
/// idle only after those listeners finish.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum AgentEvent {
    /// Starts and ends one agent run; `agent_end` carries all messages produced by that run.
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    /// One assistant response and its resulting tool calls.
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    /// Lifecycle events for user, assistant, and tool-result messages.
    MessageStart {
        message: AgentMessage,
    },
    /// Only emitted for assistant messages during streaming.
    MessageUpdate {
        message: AgentMessage,
        assistant_message_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: AgentMessage,
    },
    /// Tool execution events; parallel calls may end in completion rather than source order.
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
    },
}

impl AgentEvent {
    /// `event.type` in the TypeScript union.
    pub fn type_name(&self) -> &'static str {
        match self {
            AgentEvent::AgentStart => "agent_start",
            AgentEvent::AgentEnd { .. } => "agent_end",
            AgentEvent::TurnStart => "turn_start",
            AgentEvent::TurnEnd { .. } => "turn_end",
            AgentEvent::MessageStart { .. } => "message_start",
            AgentEvent::MessageUpdate { .. } => "message_update",
            AgentEvent::MessageEnd { .. } => "message_end",
            AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
            AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
            AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        }
    }

    /// JSON projection that uses the TypeScript event names for `type` and the
    /// original camelCase field names. `assistant_message_event` and the tool
    /// result payloads are carried through as their own serialized shapes.
    pub fn to_json(&self) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("type".to_string(), Value::String(self.type_name().to_string()));
        match self {
            AgentEvent::AgentStart | AgentEvent::TurnStart => {}
            AgentEvent::AgentEnd { messages } => {
                object.insert("messages".to_string(), serde_json::to_value(messages).unwrap_or(Value::Null));
            }
            AgentEvent::TurnEnd { message, tool_results } => {
                object.insert("message".to_string(), serde_json::to_value(message).unwrap_or(Value::Null));
                object.insert(
                    "toolResults".to_string(),
                    serde_json::to_value(tool_results).unwrap_or(Value::Null),
                );
            }
            AgentEvent::MessageStart { message } | AgentEvent::MessageEnd { message } => {
                object.insert("message".to_string(), serde_json::to_value(message).unwrap_or(Value::Null));
            }
            AgentEvent::MessageUpdate {
                message,
                assistant_message_event,
            } => {
                object.insert("message".to_string(), serde_json::to_value(message).unwrap_or(Value::Null));
                object.insert(
                    "assistantMessageEvent".to_string(),
                    serde_json::to_value(assistant_message_event).unwrap_or(Value::Null),
                );
            }
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => {
                object.insert("toolCallId".to_string(), Value::String(tool_call_id.clone()));
                object.insert("toolName".to_string(), Value::String(tool_name.clone()));
                object.insert("args".to_string(), args.clone());
            }
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                args,
                partial_result,
            } => {
                object.insert("toolCallId".to_string(), Value::String(tool_call_id.clone()));
                object.insert("toolName".to_string(), Value::String(tool_name.clone()));
                object.insert("args".to_string(), args.clone());
                object.insert(
                    "partialResult".to_string(),
                    serde_json::to_value(partial_result).unwrap_or(Value::Null),
                );
            }
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => {
                object.insert("toolCallId".to_string(), Value::String(tool_call_id.clone()));
                object.insert("toolName".to_string(), Value::String(tool_name.clone()));
                object.insert("result".to_string(), serde_json::to_value(result).unwrap_or(Value::Null));
                object.insert("isError".to_string(), Value::Bool(*is_error));
            }
        }
        Value::Object(object)
    }
}

/// Hook callbacks used by the low-level agent loop. All callbacks are
/// `Send + Sync` because the loop is driven from async tasks.
#[derive(Clone, Default)]
pub struct AgentLoopConfig {
    /// Base simple stream options (temperature, maxTokens, headers, metadata, ...).
    pub stream_options: SimpleStreamOptions,
    pub model: Model,
    /// Optional disposable local measurements. `None` preserves the zero-overhead default.
    pub performance_metrics: Option<AgentLoopPerformanceMetrics>,
    /// Converts `AgentMessage[]` to LLM-compatible `Message[]` before each LLM call.
    ///
    /// Contract: must not throw or reject. Return a safe fallback value instead.
    pub convert_to_llm: Option<Arc<dyn Fn(Vec<AgentMessage>) -> futures::future::BoxFuture<'static, Vec<Message>> + Send + Sync>>,
    /// Optional transform applied to the context before `convertToLlm`.
    pub transform_context: Option<
        Arc<
            dyn Fn(Vec<AgentMessage>, Option<tokio_util::sync::CancellationToken>) -> futures::future::BoxFuture<'static, Vec<AgentMessage>>
                + Send
                + Sync,
        >,
    >,
    /// Resolves the system prompt immediately before each LLM call.
    pub get_system_prompt: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    /// AWAITED once per provider request, immediately before the request
    /// context is built (after `turn_start` emission, before the LLM call).
    /// Arguments: the zero-based request index within this run (matches the
    /// session turn counter: reset at AgentStart, incremented at TurnEnd) and
    /// the loop's cancellation signal. `None` (default) preserves the
    /// zero-overhead default path byte-for-byte.
    pub before_request: Option<
        Arc<
            dyn Fn(
                    u64,
                    Option<tokio_util::sync::CancellationToken>,
                ) -> futures::future::BoxFuture<'static, anyhow::Result<()>>
                + Send
                + Sync,
        >,
    >,
    /// Resolves an API key dynamically for each LLM call.
    ///
    /// Contract: must not throw or reject. Return `None` when no key is available.
    pub get_api_key: Option<Arc<dyn Fn(String) -> futures::future::BoxFuture<'static, Option<String>> + Send + Sync>>,
    /// Called after each turn fully completes and `turn_end` has been emitted.
    pub should_stop_after_turn: Option<
        Arc<
            dyn Fn(ShouldStopAfterTurnContext) -> futures::future::BoxFuture<'static, bool>
                + Send
                + Sync,
        >,
    >,
    /// Called synchronously after a completed turn and before polling work for another turn.
    pub should_stop_before_turn: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    /// Returns steering messages to inject into the conversation mid-run.
    pub get_steering_messages: Option<Arc<dyn Fn() -> futures::future::BoxFuture<'static, Vec<AgentMessage>> + Send + Sync>>,
    /// Returns follow-up messages to process after the agent would otherwise stop.
    pub get_follow_up_messages: Option<Arc<dyn Fn() -> futures::future::BoxFuture<'static, Vec<AgentMessage>> + Send + Sync>>,
    /// Returns continuation messages when the agent would otherwise stop.
    pub get_continuation_messages: Option<
        Arc<
            dyn Fn(
                    GetContinuationMessagesContext,
                    Option<tokio_util::sync::CancellationToken>,
                ) -> futures::future::BoxFuture<'static, Vec<AgentMessage>>
                + Send
                + Sync,
        >,
    >,
    /// Tool execution mode. Defaults to `Parallel`.
    pub tool_execution: Option<ToolExecutionMode>,
    /// Called before a tool is executed, after arguments have been validated.
    pub before_tool_call: Option<
        Arc<
            dyn Fn(
                    BeforeToolCallContext,
                    Option<tokio_util::sync::CancellationToken>,
                ) -> futures::future::BoxFuture<'static, anyhow::Result<Option<BeforeToolCallResult>>>
                + Send
                + Sync,
        >,
    >,
    /// Called after a tool finishes executing, before `tool_execution_end` and
    /// tool-result message events are emitted.
    pub after_tool_call: Option<
        Arc<
            dyn Fn(
                    AfterToolCallContext,
                    Option<tokio_util::sync::CancellationToken>,
                ) -> futures::future::BoxFuture<'static, anyhow::Result<Option<AfterToolCallResult>>>
                + Send
                + Sync,
        >,
    >,
}

impl std::fmt::Debug for AgentLoopConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentLoopConfig")
            .field("model", &self.model.id)
            .field("tool_execution", &self.tool_execution)
            .finish_non_exhaustive()
    }
}

impl AgentLoopConfig {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Provider options handed to the stream function: the base simple stream
    /// options plus `apiKey`, `signal`, and the observed callbacks.
    pub fn provider_options(&self) -> SimpleStreamOptions {
        self.stream_options.clone()
    }

    pub fn resolved_tool_execution(&self) -> ToolExecutionMode {
        self.tool_execution.unwrap_or(ToolExecutionMode::Parallel)
    }
}

/// `Record<string, string>` helper used by provider options.
pub type StringRecord = BTreeMap<String, String>;

/// `Record<string, unknown>` helper used by provider options.
pub type ValueRecord = IndexMap<String, Value>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_message_role_matches_typescript_discriminants() {
        let user = AgentMessage::from(pi_ai::types::UserMessage {
            role: "user".to_string(),
            content: pi_ai::types::UserContent::Text("hi".to_string()),
            provider_context: None,
            timestamp: 1,
        });
        assert_eq!(user.role(), "user");

        let tool_result = AgentMessage::from(ToolResultMessage {
            role: "toolResult".to_string(),
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            content: Vec::new(),
            details: None,
            is_error: false,
            timestamp: 2,
        });
        assert_eq!(tool_result.role(), "toolResult");

        let custom = AgentMessage::Custom(CustomAgentMessage::Custom {
            custom_type: "notification".to_string(),
            content: CustomMessageContent::Text("body".to_string()),
            display: true,
            details: None,
            timestamp: 3,
        });
        assert_eq!(custom.role(), "custom");
    }

    #[test]
    fn thinking_level_serializes_lowercase() {
        assert_eq!(serde_json::to_value(ThinkingLevel::Xhigh).unwrap(), Value::String("xhigh".into()));
        assert_eq!(serde_json::from_value::<ThinkingLevel>(Value::String("off".into())).unwrap(), ThinkingLevel::Off);
    }

    #[test]
    fn tool_execution_mode_defaults_to_parallel() {
        let config = AgentLoopConfig::new(Model::new("unknown", "unknown", "unknown", "unknown", ""));
        assert_eq!(config.resolved_tool_execution(), ToolExecutionMode::Parallel);
        assert_eq!(serde_json::to_value(ToolExecutionMode::Sequential).unwrap(), Value::String("sequential".into()));
    }

    #[test]
    fn agent_event_type_names_match_typescript() {
        assert_eq!(AgentEvent::AgentStart.type_name(), "agent_start");
        assert_eq!(AgentEvent::TurnStart.type_name(), "turn_start");
        assert_eq!(
            AgentEvent::ToolExecutionEnd {
                tool_call_id: "c".into(),
                tool_name: "t".into(),
                result: AgentToolResult::new(vec![], Value::Object(Default::default())),
                is_error: true,
            }
            .type_name(),
            "tool_execution_end"
        );
    }

    #[test]
    fn agent_events_round_trip_the_flat_typescript_wire_shape() {
        let message = AssistantMessage::new("faux", "faux", "faux-1", 1);
        let agent_message = AgentMessage::from(message.clone());
        let result = AgentToolResult::new(vec![ContentBlock::text("done")], serde_json::json!({}));
        let events = vec![
            AgentEvent::AgentStart,
            AgentEvent::AgentEnd { messages: vec![agent_message.clone()] },
            AgentEvent::TurnStart,
            AgentEvent::TurnEnd { message: agent_message.clone(), tool_results: Vec::new() },
            AgentEvent::MessageStart { message: agent_message.clone() },
            AgentEvent::MessageUpdate {
                message: agent_message.clone(),
                assistant_message_event: AssistantMessageEvent::Start { partial: message },
            },
            AgentEvent::MessageEnd { message: agent_message },
            AgentEvent::ToolExecutionStart {
                tool_call_id: "call-1".into(), tool_name: "echo".into(), args: serde_json::json!({}),
            },
            AgentEvent::ToolExecutionUpdate {
                tool_call_id: "call-1".into(), tool_name: "echo".into(), args: serde_json::json!({}),
                partial_result: result.clone(),
            },
            AgentEvent::ToolExecutionEnd {
                tool_call_id: "call-1".into(), tool_name: "echo".into(), result, is_error: false,
            },
        ];
        for event in events {
            let value = serde_json::to_value(&event).unwrap();
            assert_eq!(value, event.to_json());
            assert_eq!(serde_json::from_value::<AgentEvent>(value).unwrap(), event);
        }
        let result: AfterToolCallResult = serde_json::from_value(serde_json::json!({"isError": true})).unwrap();
        assert_eq!(result.is_error, Some(true));
    }
}
