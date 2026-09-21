//! Port of packages/agent/src/agent.ts
//!
//! `Agent` owns its state behind a `Mutex` because the TypeScript object is
//! shared by reference while it also drives an async run. `AbortController`
//! becomes `CancellationToken`, and listeners are awaited in subscription order
//! exactly like the TypeScript `for (const listener of this.listeners)`.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock};

use futures::future::BoxFuture;
use pi_ai::types::{
    AssistantMessage, ImageContent, ImageOrTextContent, Message, Model, TextContent,
    ThinkingBudgets, Usage, UserContent, UserMessage,
};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{run_agent_loop, run_agent_loop_continue};
use crate::performance_metrics::AgentLoopPerformanceMetrics;
use crate::types::{
    AfterToolCallContext, AfterToolCallResult, AgentContext, AgentEvent, AgentLoopConfig,
    AgentMessage, AgentState, BeforeToolCallContext, BeforeToolCallResult,
    GetContinuationMessagesContext, ShouldStopAfterTurnContext, StreamFn, ToolExecutionMode,
};

/// Why [`Agent::continue_`] refused to start a continuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentContinueErrorCode {
    Busy,
    NothingToContinue,
}

/// Typed precondition failure from [`Agent::continue_`], so callers classify by
/// code instead of message text.
#[derive(Debug, Clone)]
pub struct AgentContinueError {
    pub code: AgentContinueErrorCode,
    pub message: String,
}

impl std::fmt::Display for AgentContinueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AgentContinueError {}

/// `type QueueMode = "all" | "one-at-a-time"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMode {
    All,
    OneAtATime,
}

impl Default for QueueMode {
    /// `options.steeringMode ?? "one-at-a-time"`.
    fn default() -> Self {
        QueueMode::OneAtATime
    }
}

impl QueueMode {
    pub fn as_str(self) -> &'static str {
        match self {
            QueueMode::All => "all",
            QueueMode::OneAtATime => "one-at-a-time",
        }
    }
}

/// `class PendingMessageQueue`.
#[derive(Debug, Default)]
pub struct PendingMessageQueue {
    pub mode: QueueMode,
    batches: Vec<Vec<AgentMessage>>,
}

impl PendingMessageQueue {
    pub fn new(mode: QueueMode) -> Self {
        Self {
            mode,
            batches: Vec::new(),
        }
    }

    /// `enqueue(message)`.
    pub fn enqueue(&mut self, batch: Vec<AgentMessage>) {
        if !batch.is_empty() {
            self.batches.push(batch);
        }
    }

    /// `enqueue(message: AgentMessage)` - a single message becomes one batch.
    pub fn enqueue_one(&mut self, message: AgentMessage) {
        self.enqueue(vec![message]);
    }

    pub fn has_items(&self) -> bool {
        !self.batches.is_empty()
    }

    /// `drain()`.
    pub fn drain(&mut self) -> Vec<AgentMessage> {
        if self.mode == QueueMode::All {
            let drained = self.batches.iter().flatten().cloned().collect();
            self.batches = Vec::new();
            return drained;
        }

        match self.batches.first() {
            Some(first) => {
                let first = first.clone();
                self.batches = self.batches.iter().skip(1).cloned().collect();
                first
            }
            None => Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.batches = Vec::new();
    }

    /// `removeWhere(predicate)`.
    pub fn remove_where(
        &mut self,
        predicate: &(dyn Fn(&AgentMessage) -> bool + Send + Sync),
    ) -> Vec<AgentMessage> {
        let mut removed: Vec<AgentMessage> = Vec::new();
        let mut retained: Vec<Vec<AgentMessage>> = Vec::new();
        for batch in self.batches.drain(..) {
            if batch.iter().any(|message| predicate(message)) {
                removed.extend(batch);
            } else {
                retained.push(batch);
            }
        }
        self.batches = retained;
        removed
    }
}

/// `interface AgentOptions`.
#[derive(Clone)]
pub struct AgentOptions {
    pub initial_state: Option<AgentState>,
    pub convert_to_llm:
        Option<Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<'static, Vec<Message>> + Send + Sync>>,
    pub transform_context: Option<
        Arc<
            dyn Fn(
                    Vec<AgentMessage>,
                    Option<CancellationToken>,
                ) -> BoxFuture<'static, Vec<AgentMessage>>
                + Send
                + Sync,
        >,
    >,
    pub stream_fn: Option<StreamFn>,
    pub get_api_key:
        Option<Arc<dyn Fn(String) -> BoxFuture<'static, Option<String>> + Send + Sync>>,
    pub on_payload: Option<pi_ai::types::OnPayload>,
    pub on_response: Option<pi_ai::types::OnResponse>,
    pub before_tool_call: Option<
        Arc<
            dyn Fn(
                    BeforeToolCallContext,
                    Option<CancellationToken>,
                )
                    -> BoxFuture<'static, anyhow::Result<Option<BeforeToolCallResult>>>
                + Send
                + Sync,
        >,
    >,
    pub after_tool_call: Option<
        Arc<
            dyn Fn(
                    AfterToolCallContext,
                    Option<CancellationToken>,
                )
                    -> BoxFuture<'static, anyhow::Result<Option<AfterToolCallResult>>>
                + Send
                + Sync,
        >,
    >,
    pub should_stop_after_turn:
        Option<Arc<dyn Fn(ShouldStopAfterTurnContext) -> BoxFuture<'static, bool> + Send + Sync>>,
    pub should_stop_before_turn: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    pub get_continuation_messages: Option<
        Arc<
            dyn Fn(
                    GetContinuationMessagesContext,
                    Option<CancellationToken>,
                ) -> BoxFuture<'static, Vec<AgentMessage>>
                + Send
                + Sync,
        >,
    >,
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
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Option<String>,
    pub tool_execution: Option<ToolExecutionMode>,
    pub performance_metrics: Option<AgentLoopPerformanceMetrics>,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            initial_state: None,
            convert_to_llm: None,
            transform_context: None,
            stream_fn: None,
            get_api_key: None,
            on_payload: None,
            on_response: None,
            before_tool_call: None,
            after_tool_call: None,
            should_stop_after_turn: None,
            should_stop_before_turn: None,
            get_continuation_messages: None,
            before_request: None,
            steering_mode: None,
            follow_up_mode: None,
            session_id: None,
            thinking_budgets: None,
            transport: None,
            tool_execution: None,
            performance_metrics: None,
        }
    }
}

impl std::fmt::Debug for AgentOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentOptions")
            .field("session_id", &self.session_id)
            .field("tool_execution", &self.tool_execution)
            .finish_non_exhaustive()
    }
}

/// `function defaultConvertToLlm(messages)`.
pub fn default_convert_to_llm(messages: Vec<AgentMessage>) -> Vec<Message> {
    messages
        .into_iter()
        .filter_map(|message| match message {
            AgentMessage::Message(message) => Some(message),
            AgentMessage::Custom(_) => None,
        })
        .collect()
}

/// `const DEFAULT_MODEL` in agent.ts.
pub fn default_model() -> Model {
    Model::new("unknown", "unknown", "unknown", "unknown", "")
}

/// `ActiveRun`.
struct ActiveRun {
    aborted: CancellationToken,
    idle: Arc<tokio::sync::Notify>,
    settled: Arc<AtomicBool>,
}

/// `class Agent`.
pub struct Agent {
    state: Arc<Mutex<AgentState>>,
    listeners: Mutex<Vec<ListenerEntry>>,
    steering_queue: Arc<Mutex<PendingMessageQueue>>,
    follow_up_queue: Arc<Mutex<PendingMessageQueue>>,
    pub convert_to_llm:
        Mutex<Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<'static, Vec<Message>> + Send + Sync>>,
    pub transform_context: Mutex<
        Option<
            Arc<
                dyn Fn(
                        Vec<AgentMessage>,
                        Option<CancellationToken>,
                    ) -> BoxFuture<'static, Vec<AgentMessage>>
                    + Send
                    + Sync,
            >,
        >,
    >,
    pub stream_fn: Mutex<StreamFn>,
    pub get_api_key:
        Mutex<Option<Arc<dyn Fn(String) -> BoxFuture<'static, Option<String>> + Send + Sync>>>,
    pub on_payload: Mutex<Option<pi_ai::types::OnPayload>>,
    pub on_response: Mutex<Option<pi_ai::types::OnResponse>>,
    pub before_tool_call: Mutex<
        Option<
            Arc<
                dyn Fn(
                        BeforeToolCallContext,
                        Option<CancellationToken>,
                    )
                        -> BoxFuture<'static, anyhow::Result<Option<BeforeToolCallResult>>>
                    + Send
                    + Sync,
            >,
        >,
    >,
    pub after_tool_call: Mutex<
        Option<
            Arc<
                dyn Fn(
                        AfterToolCallContext,
                        Option<CancellationToken>,
                    )
                        -> BoxFuture<'static, anyhow::Result<Option<AfterToolCallResult>>>
                    + Send
                    + Sync,
            >,
        >,
    >,
    pub should_stop_after_turn: Mutex<
        Option<Arc<dyn Fn(ShouldStopAfterTurnContext) -> BoxFuture<'static, bool> + Send + Sync>>,
    >,
    pub should_stop_before_turn: Mutex<Option<Arc<dyn Fn() -> bool + Send + Sync>>>,
    pub get_continuation_messages: Mutex<
        Option<
            Arc<
                dyn Fn(
                        GetContinuationMessagesContext,
                        Option<CancellationToken>,
                    ) -> BoxFuture<'static, Vec<AgentMessage>>
                    + Send
                    + Sync,
            >,
        >,
    >,
    pub before_request: Mutex<
        Option<
            Arc<
                dyn Fn(
                        u64,
                        Option<tokio_util::sync::CancellationToken>,
                    )
                        -> futures::future::BoxFuture<'static, anyhow::Result<()>>
                    + Send
                    + Sync,
            >,
        >,
    >,
    active_run: Mutex<Option<Arc<ActiveRun>>>,
    pub session_id: Mutex<Option<String>>,
    pub thinking_budgets: Mutex<Option<ThinkingBudgets>>,
    pub transport: Mutex<String>,
    pub tool_execution: Mutex<ToolExecutionMode>,
    pub performance_metrics: Mutex<Option<AgentLoopPerformanceMetrics>>,
}

/// `listeners: Set<(event, signal) => Promise<void> | void>` - insertion order is
/// observable because listeners are awaited in subscription order.
struct ListenerEntry {
    id: u64,
    listener:
        Arc<dyn Fn(AgentEvent, Option<CancellationToken>) -> BoxFuture<'static, ()> + Send + Sync>,
}

fn next_listener_id() -> u64 {
    static NEXT: OnceLock<Mutex<u64>> = OnceLock::new();
    let counter = NEXT.get_or_init(|| Mutex::new(0));
    let mut guard = counter
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard += 1;
    *guard
}

impl Agent {
    /// `constructor(options: AgentOptions = {})`.
    pub fn new(options: AgentOptions) -> Arc<Self> {
        let mut initial_state = options.initial_state.clone().unwrap_or_default();
        // `initialState?.tools?.slice() ?? []` / `messages?.slice() ?? []`
        initial_state.tools = initial_state.tools.clone();
        initial_state.messages = initial_state.messages.clone();

        let steering_mode = options.steering_mode.unwrap_or(QueueMode::OneAtATime);
        let follow_up_mode = options.follow_up_mode.unwrap_or(QueueMode::OneAtATime);

        let agent = Arc::new(Self {
            state: Arc::new(Mutex::new(initial_state)),
            listeners: Mutex::new(Vec::new()),
            steering_queue: Arc::new(Mutex::new(PendingMessageQueue::new(steering_mode))),
            follow_up_queue: Arc::new(Mutex::new(PendingMessageQueue::new(follow_up_mode))),
            convert_to_llm: Mutex::new(options.convert_to_llm.unwrap_or_else(|| {
                Arc::new(|messages| Box::pin(async move { default_convert_to_llm(messages) }))
            })),
            transform_context: Mutex::new(options.transform_context),
            // `options.streamFn ?? streamSimple` - the loop resolves the default.
            stream_fn: Mutex::new(options.stream_fn.unwrap_or_else(default_stream_fn)),
            get_api_key: Mutex::new(options.get_api_key),
            on_payload: Mutex::new(options.on_payload),
            on_response: Mutex::new(options.on_response),
            before_tool_call: Mutex::new(options.before_tool_call),
            after_tool_call: Mutex::new(options.after_tool_call),
            should_stop_after_turn: Mutex::new(options.should_stop_after_turn),
            should_stop_before_turn: Mutex::new(options.should_stop_before_turn),
            get_continuation_messages: Mutex::new(options.get_continuation_messages),
            before_request: Mutex::new(options.before_request),
            active_run: Mutex::new(None),
            session_id: Mutex::new(options.session_id),
            thinking_budgets: Mutex::new(options.thinking_budgets),
            transport: Mutex::new(options.transport.unwrap_or_else(|| "auto".to_string())),
            tool_execution: Mutex::new(
                options
                    .tool_execution
                    .unwrap_or(ToolExecutionMode::Parallel),
            ),
            performance_metrics: Mutex::new(options.performance_metrics),
        });
        agent
    }

    /// `subscribe(listener)` - returns the unsubscribe function.
    pub fn subscribe(
        self: &Arc<Self>,
        listener: Arc<
            dyn Fn(AgentEvent, Option<CancellationToken>) -> BoxFuture<'static, ()> + Send + Sync,
        >,
    ) -> UnsubscribeHandle {
        let id = next_listener_id();
        self.listeners
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(ListenerEntry { id, listener });
        UnsubscribeHandle {
            agent: self.clone(),
            id,
        }
    }

    fn remove_listener(&self, id: u64) {
        let mut listeners = self
            .listeners
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        listeners.retain(|entry| entry.id != id);
    }

    /// `get state(): AgentState` - a copy, because Rust callers assign fields and
    /// store the result back with [`Agent::set_state`].
    pub fn state(&self) -> AgentState {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Read a small state projection without cloning the conversation. The
    /// callback must not call back into the agent or hold this lock across IO.
    pub fn read_state<T>(&self, read: impl FnOnce(&AgentState) -> T) -> T {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        read(&state)
    }

    /// `set state` for the accessor fields: assigning `tools` or `messages`
    /// copies the provided top-level array.
    pub fn set_state(&self, mut state: AgentState) {
        state.tools = state.tools.clone();
        state.messages = state.messages.clone();
        *self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = state;
    }

    /// Mutate the state in place - the Rust equivalent of field assignment on the
    /// live object (`agent.state.messages.push(...)`).
    pub fn update_state<F: FnOnce(&mut AgentState)>(&self, update: F) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        update(&mut state);
    }

    /// `set steeringMode(mode)`.
    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.steering_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .mode = mode;
    }

    /// `get steeringMode(): QueueMode`.
    pub fn steering_mode(&self) -> QueueMode {
        self.steering_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .mode
    }

    /// `set followUpMode(mode)`.
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .mode = mode;
    }

    /// `get followUpMode(): QueueMode`.
    pub fn follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .mode
    }

    /// Queue a message batch to be injected after the current assistant turn finishes.
    pub fn steer(&self, message: Vec<AgentMessage>) {
        self.steering_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .enqueue(message);
    }

    /// Queue a message batch to run only after the agent would otherwise stop.
    pub fn follow_up(&self, message: Vec<AgentMessage>) {
        self.follow_up_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .enqueue(message);
    }

    pub fn clear_steering_queue(&self) {
        self.steering_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// `removeQueuedMessages(predicate)`.
    pub fn remove_queued_messages(
        &self,
        predicate: &(dyn Fn(&AgentMessage) -> bool + Send + Sync),
    ) -> Vec<AgentMessage> {
        let mut removed = self
            .steering_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove_where(predicate);
        removed.extend(
            self.follow_up_queue
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove_where(predicate),
        );
        removed
    }

    pub fn has_queued_messages(&self) -> bool {
        self.steering_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .has_items()
            || self
                .follow_up_queue
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .has_items()
    }

    /// `get signal(): AbortSignal | undefined`.
    pub fn signal(&self) -> Option<CancellationToken> {
        self.active_run
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|run| run.aborted.clone())
    }

    /// `abort()`.
    pub fn abort(&self) {
        if let Some(run) = self
            .active_run
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            run.aborted.cancel();
        }
    }

    /// Resolve when the current run and all awaited event listeners have finished.
    pub async fn wait_for_idle(&self) {
        let run = self
            .active_run
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let Some(run) = run else {
            return;
        };
        // Register interest before awaiting: `Notify::notify_waiters` stores no permit, so a
        // waiter that is not registered yet would miss `finish_run` and park forever.
        let notified = run.idle.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        #[cfg(test)]
        idle_wait_probe::after_settled_check();
        if run.settled.load(AtomicOrdering::SeqCst) {
            return;
        }
        notified.await;
    }

    /// `reset()`.
    pub fn reset(&self) {
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.messages = Vec::new();
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls = BTreeSet::new();
            state.error_message = None;
        }
        self.clear_follow_up_queue();
        self.clear_steering_queue();
    }

    /// `async prompt(...)`.
    pub async fn prompt(self: &Arc<Self>, input: PromptInput) -> anyhow::Result<()> {
        if self.is_running() {
            return Err(anyhow::anyhow!(
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
            ));
        }
        let messages = self.normalize_prompt_input(input);
        self.run_prompt_messages(messages, false).await
    }

    /// `async continue()`.
    pub async fn continue_(self: &Arc<Self>) -> Result<(), AgentContinueError> {
        if self.is_running() {
            return Err(AgentContinueError {
                code: AgentContinueErrorCode::Busy,
                message: "Agent is already processing. Wait for completion before continuing."
                    .to_string(),
            });
        }

        let last_message = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.messages.last().cloned()
        };
        let Some(last_message) = last_message else {
            if self.run_queued_messages().await? {
                return Ok(());
            }
            return Err(AgentContinueError {
                code: AgentContinueErrorCode::NothingToContinue,
                message: "No messages to continue from".to_string(),
            });
        };

        if last_message.role() == "assistant" {
            if self.run_queued_messages().await? {
                return Ok(());
            }
            return Err(AgentContinueError {
                code: AgentContinueErrorCode::NothingToContinue,
                message: "Cannot continue from message role: assistant".to_string(),
            });
        }

        if last_message.role() == "custom" && self.run_queued_messages().await? {
            return Ok(());
        }

        self.run_continuation()
            .await
            .map_err(|error| AgentContinueError {
                code: AgentContinueErrorCode::Busy,
                message: error.to_string(),
            })
    }
}

/// The unsubscribe function returned by [`Agent::subscribe`].
pub struct UnsubscribeHandle {
    agent: Arc<Agent>,
    id: u64,
}

impl UnsubscribeHandle {
    pub fn unsubscribe(&self) {
        self.agent.remove_listener(self.id);
    }
}

/// `prompt(message | message[])` and `prompt(input, images?)`.
#[derive(Debug, Clone)]
pub enum PromptInput {
    Messages(Vec<AgentMessage>),
    Message(Box<AgentMessage>),
    Text {
        input: String,
        images: Vec<ImageContent>,
    },
}

impl Agent {
    /// `runQueuedMessages()` - drains steering first, then follow-ups.
    async fn run_queued_messages(self: &Arc<Self>) -> Result<bool, AgentContinueError> {
        let queued_steering = self
            .steering_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain();
        if !queued_steering.is_empty() {
            self.run_prompt_messages(queued_steering, true)
                .await
                .map_err(|error| AgentContinueError {
                    code: AgentContinueErrorCode::Busy,
                    message: error.to_string(),
                })?;
            return Ok(true);
        }

        let queued_follow_ups = self
            .follow_up_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain();
        if !queued_follow_ups.is_empty() {
            self.run_prompt_messages(queued_follow_ups, false)
                .await
                .map_err(|error| AgentContinueError {
                    code: AgentContinueErrorCode::Busy,
                    message: error.to_string(),
                })?;
            return Ok(true);
        }

        Ok(false)
    }

    fn is_running(&self) -> bool {
        self.active_run
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }

    /// `normalizePromptInput(input, images)`.
    fn normalize_prompt_input(&self, input: PromptInput) -> Vec<AgentMessage> {
        match input {
            PromptInput::Messages(messages) => messages,
            PromptInput::Message(message) => vec![*message],
            PromptInput::Text { input, images } => {
                let mut content: Vec<ImageOrTextContent> =
                    vec![ImageOrTextContent::Text(TextContent::new(input))];
                // `content.push(...images)`
                content.extend(images.into_iter().map(ImageOrTextContent::Image));
                vec![AgentMessage::from(UserMessage {
                    role: pi_ai::types::ROLE_USER.to_string(),
                    content: UserContent::Blocks(content),
                    provider_context: None,
                    timestamp: pi_ai::utils::now_ms(),
                })]
            }
        }
    }

    /// `runWithLifecycle(executor)`.
    async fn run_with_lifecycle<F>(self: &Arc<Self>, executor: F) -> anyhow::Result<()>
    where
        F: FnOnce(Option<CancellationToken>) -> BoxFuture<'static, anyhow::Result<()>>,
    {
        if self.is_running() {
            return Err(anyhow::anyhow!("Agent is already processing."));
        }

        let aborted = CancellationToken::new();
        let run = Arc::new(ActiveRun {
            aborted: aborted.clone(),
            idle: Arc::new(tokio::sync::Notify::new()),
            settled: Arc::new(AtomicBool::new(false)),
        });
        *self
            .active_run
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(run.clone());

        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.is_streaming = true;
            state.streaming_message = None;
            state.error_message = None;
        }

        let outcome = executor(Some(aborted.clone())).await;
        match outcome {
            Ok(()) => {}
            Err(error) => {
                let aborted_flag = aborted.is_cancelled();
                self.handle_run_failure(error, aborted_flag).await;
            }
        }
        self.finish_run();
        Ok(())
    }

    /// `handleRunFailure(error, aborted)`.
    async fn handle_run_failure(self: &Arc<Self>, error: anyhow::Error, aborted: bool) {
        let model = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.model.clone()
        };
        let error_message = error.to_string();
        let failure_message = AgentMessage::from(AssistantMessage {
            role: pi_ai::types::ROLE_ASSISTANT.to_string(),
            content: vec![pi_ai::types::ContentBlock::Text(TextContent::new(""))],
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: if aborted {
                None
            } else {
                Some(vec![
                    pi_ai::utils::diagnostics::create_assistant_message_diagnostic(
                        "agent_lifecycle_failure",
                        &pi_ai::utils::diagnostics::ThrownValue::Text(&error_message),
                        Some({
                            let mut details = serde_json::Map::new();
                            details.insert(
                                "source".to_string(),
                                Value::String("run_with_lifecycle".to_string()),
                            );
                            details
                        }),
                    ),
                ])
            },
            usage: Usage::zero(),
            stop_reason: if aborted {
                pi_ai::types::STOP_REASON_ABORTED.to_string()
            } else {
                pi_ai::types::STOP_REASON_ERROR.to_string()
            },
            stop_reason_raw: None,
            error_message: Some(error_message.clone()),
            timestamp: pi_ai::utils::now_ms(),
        });

        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.error_message = Some(error_message);
        }
        // `await this.processEvents(...).catch(() => undefined)` - `process_events` is
        // `async` and infallible here, so no catch is needed.
        self.process_events(AgentEvent::MessageStart {
            message: failure_message.clone(),
        })
        .await;
        self.process_events(AgentEvent::MessageEnd {
            message: failure_message.clone(),
        })
        .await;
        self.process_events(AgentEvent::AgentEnd {
            messages: vec![failure_message],
        })
        .await;
    }

    /// `finishRun()`.
    fn finish_run(&self) {
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls = BTreeSet::new();
        }
        let run = self
            .active_run
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(run) = run {
            run.settled.store(true, AtomicOrdering::SeqCst);
            run.idle.notify_waiters();
        }
    }

    /// `processEvents(event)` - reduces internal state, then awaits listeners.
    pub async fn process_events(self: &Arc<Self>, event: AgentEvent) {
        match &event {
            AgentEvent::MessageStart { message } => {
                self.update_state(|state| state.streaming_message = Some(message.clone()));
            }
            AgentEvent::MessageUpdate { message, .. } => {
                self.update_state(|state| state.streaming_message = Some(message.clone()));
            }
            AgentEvent::MessageEnd { message } => {
                self.update_state(|state| {
                    state.streaming_message = None;
                    state.messages.push(message.clone());
                });
            }
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                self.update_state(|state| {
                    let mut pending = state.pending_tool_calls.clone();
                    pending.insert(tool_call_id.clone());
                    state.pending_tool_calls = pending;
                });
            }
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                self.update_state(|state| {
                    let mut pending = state.pending_tool_calls.clone();
                    pending.remove(tool_call_id);
                    state.pending_tool_calls = pending;
                });
            }
            AgentEvent::TurnEnd { message, .. } => {
                let error_message = match message {
                    AgentMessage::Message(Message::Assistant(assistant)) => {
                        assistant.error_message.clone()
                    }
                    _ => None,
                };
                if let Some(error_message) = error_message {
                    self.update_state(|state| state.error_message = Some(error_message));
                }
            }
            AgentEvent::AgentEnd { .. } => {
                self.update_state(|state| state.streaming_message = None);
            }
            // `tool_execution_update` does not change agent state.
            AgentEvent::ToolExecutionUpdate { .. }
            | AgentEvent::AgentStart
            | AgentEvent::TurnStart => {}
        }

        let signal = self.signal();
        let Some(signal) = signal else {
            // `throw new Error("Agent listener invoked outside active run")`
            panic!("Agent listener invoked outside active run");
        };
        let listeners: Vec<_> = self
            .listeners
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|entry| entry.listener.clone())
            .collect();
        for listener in listeners {
            listener(event.clone(), Some(signal.clone())).await;
        }
    }
}

/// `streamSimple` - the `options.streamFn ?? streamSimple` default of `Agent`.
pub fn default_stream_fn() -> StreamFn {
    Arc::new(|model, context, options| {
        let stream = pi_ai::stream::stream_simple(&model, &context, Some(&options));
        Box::pin(async move { stream })
    })
}

impl Agent {
    /// `runPromptMessages(messages, options)`.
    async fn run_prompt_messages(
        self: &Arc<Self>,
        messages: Vec<AgentMessage>,
        skip_initial_steering_poll: bool,
    ) -> anyhow::Result<()> {
        let agent = self.clone();
        self.run_with_lifecycle(move |signal| {
            let agent = agent.clone();
            Box::pin(async move {
                let context = agent.create_context_snapshot();
                let config = agent.create_loop_config(Some(skip_initial_steering_poll));
                let stream_fn = agent.stream_fn_handle();
                let emit = agent.process_events_sink();
                run_agent_loop(messages, context, config, emit, signal, Some(stream_fn)).await?;
                Ok(())
            })
        })
        .await
    }

    /// `runContinuation()`.
    async fn run_continuation(self: &Arc<Self>) -> anyhow::Result<()> {
        let agent = self.clone();
        self.run_with_lifecycle(move |signal| {
            let agent = agent.clone();
            Box::pin(async move {
                let context = agent.create_context_snapshot();
                let config = agent.create_loop_config(None);
                let stream_fn = agent.stream_fn_handle();
                let emit = agent.process_events_sink();
                run_agent_loop_continue(context, config, emit, signal, Some(stream_fn)).await?;
                Ok(())
            })
        })
        .await
    }

    fn stream_fn_handle(&self) -> StreamFn {
        self.stream_fn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn process_events_sink(self: &Arc<Self>) -> crate::agent_loop::AgentEventSink {
        let agent = self.clone();
        Arc::new(move |event| {
            let agent = agent.clone();
            Box::pin(async move {
                agent.process_events(event).await;
                Ok(())
            })
        })
    }

    /// `createContextSnapshot()`.
    fn create_context_snapshot(&self) -> AgentContext {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        AgentContext {
            system_prompt: state.system_prompt.clone(),
            messages: state.messages.clone(),
            tools: state.tools.clone(),
        }
    }

    /// `createLoopConfig(options)`.
    fn create_loop_config(&self, options: Option<bool>) -> AgentLoopConfig {
        let skip_initial_steering_poll = Arc::new(AtomicBool::new(options.unwrap_or(false)));
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let service_tier = state.service_tier.clone();
        let model = state.model.clone();
        let thinking_level = state.thinking_level;
        drop(state);

        let mut config = AgentLoopConfig::new(model);
        config.stream_options.reasoning = Some(thinking_level.as_str().to_string());
        config.stream_options.stream.service_tier = service_tier;
        config.stream_options.stream.session_id = self
            .session_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        config.stream_options.stream.transport = Some(
            self.transport
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
        );
        config.stream_options.thinking_budgets = self
            .thinking_budgets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        // The `Agent.onPayload` / `Agent.onResponse` hooks are stored as stream options,
        // exactly where `createLoopConfig` merges them into the loop config.
        config.stream_options.stream.on_payload = self
            .on_payload
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        config.stream_options.stream.on_response = self
            .on_response
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        config.performance_metrics = self
            .performance_metrics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        config.before_tool_call = self
            .before_tool_call
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        config.after_tool_call = self
            .after_tool_call
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        config.tool_execution = Some(
            *self
                .tool_execution
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        config.convert_to_llm = Some(
            self.convert_to_llm
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
        );
        config.transform_context = self
            .transform_context
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        config.get_api_key = self
            .get_api_key
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        {
            let hook = self
                .should_stop_after_turn
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            // `async (context) => this.shouldStopAfterTurn?.(context) ?? false`
            config.should_stop_after_turn = Some(Arc::new(
                move |context: ShouldStopAfterTurnContext| match hook.as_ref() {
                    Some(hook) => hook(context),
                    None => Box::pin(async { false }),
                },
            ));
        }
        {
            let hook = self
                .should_stop_before_turn
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            config.should_stop_before_turn = Some(Arc::new(move || {
                hook.as_ref().map(|hook| hook()).unwrap_or(false)
            }));
        }
        {
            // `getSystemPrompt: () => this._state.systemPrompt` - the loop already
            // falls back to `context.systemPrompt`, which is the state snapshot, so
            // the hook is only installed when the state carries one.
            // ROOT-CONTRACT v7/v9 request-local derivation: resolve the state's
            // system prompt LIVE at every provider request (including inner
            // tool/continuation turns), so a host that refreshes
            // `state.system_prompt` after an assessment (e.g. a skill hint
            // store) is consumed by the NEXT request, while the canonical
            // base content itself is owned by the host session.
            let state = self.state.clone();
            config.get_system_prompt = Some(Arc::new(move || {
                state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .system_prompt
                    .clone()
            }));
        }
        {
            let hook = self
                .before_request
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            config.before_request = hook;
        }
        {
            let steering_queue = self.steering_queue.clone();
            let skip = skip_initial_steering_poll.clone();
            config.get_steering_messages = Some(Arc::new(move || {
                let steering_queue = steering_queue.clone();
                let skip = skip.clone();
                Box::pin(async move {
                    if skip.swap(false, AtomicOrdering::SeqCst) {
                        return Vec::new();
                    }
                    steering_queue
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .drain()
                })
            }));
        }
        {
            let follow_up_queue = self.follow_up_queue.clone();
            config.get_follow_up_messages = Some(Arc::new(move || {
                let follow_up_queue = follow_up_queue.clone();
                Box::pin(async move {
                    follow_up_queue
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .drain()
                })
            }));
        }
        {
            let hook = self
                .get_continuation_messages
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            config.get_continuation_messages = Some(Arc::new(
                move |context: GetContinuationMessagesContext,
                      signal: Option<CancellationToken>| {
                    match hook.as_ref() {
                        Some(hook) => hook(context, signal),
                        None => Box::pin(async { Vec::new() }),
                    }
                },
            ));
        }

        config
    }
}

/// T10 seam (owner rlm-agentcore, D-03): the lost-wakeup window in
/// `Agent::wait_for_idle` is only a couple of instructions wide, so a test cannot
/// reach it by scheduling alone. This probe runs *inside* that window and lets the
/// test release a finishing run at exactly that point. It is compiled out of
/// non-test builds and is not exported outside this crate.
#[cfg(test)]
pub mod idle_wait_probe {
    use std::sync::{Mutex, OnceLock};

    struct Probe {
        reached: std::sync::Arc<tokio::sync::Semaphore>,
        resume: std::sync::Arc<tokio::sync::Semaphore>,
    }

    fn slot() -> &'static Mutex<Option<Probe>> {
        static SLOT: OnceLock<Mutex<Option<Probe>>> = OnceLock::new();
        SLOT.get_or_init(|| Mutex::new(None))
    }

    /// Arm the probe. The next `wait_for_idle` call that observes `settled == false`
    /// signals `reached` and then blocks until `resume` has a permit.
    pub fn arm() -> (
        std::sync::Arc<tokio::sync::Semaphore>,
        std::sync::Arc<tokio::sync::Semaphore>,
    ) {
        let reached = std::sync::Arc::new(tokio::sync::Semaphore::new(0));
        let resume = std::sync::Arc::new(tokio::sync::Semaphore::new(0));
        *slot()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Probe {
            reached: reached.clone(),
            resume: resume.clone(),
        });
        (reached, resume)
    }

    pub(super) fn after_settled_check() {
        let probe = {
            let mut slot = slot()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            slot.take()
        };
        if let Some(probe) = probe {
            probe.reached.add_permits(1);
            futures::executor::block_on(async {
                probe.resume.acquire().await.unwrap().forget();
            });
        }
    }
}

/// T10 lane (owner rlm-agentcore): D-03 `Agent::wait_for_idle` lost-wakeup race.
///
/// In-crate `#[cfg(test)]` module: the disputed window is only reachable through the
/// crate-private seam `idle_wait_probe`. Production entry points under test: `Agent::prompt`
/// and `Agent::wait_for_idle`.
#[cfg(test)]
mod rlm_t10_tests {
    use super::*;
    use pi_ai::providers::faux::{
        faux_assistant_message, register_faux_provider, FauxAssistantMessageOptions,
        FauxResponseStep,
    };
    use std::time::Duration;
    use tokio::sync::Semaphore;
    use tokio::time::timeout;

    /// The awaited-value budget: far above any scheduling delay, far below any watchdog.
    const WAIT_BUDGET: Duration = Duration::from_millis(250);
    /// Bounded safety net so a stuck barrier can never hang the test binary.
    const SEAM_BUDGET: Duration = Duration::from_secs(5);
    /// One deterministic reproduction plus about ten bounded repetitions of the schedule.
    const REPETITIONS: usize = 10;

    fn faux_agent() -> (pi_ai::providers::faux::FauxProviderRegistration, Arc<Agent>) {
        let provider = register_faux_provider(None);
        provider.set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
            "done".into(),
            Some(FauxAssistantMessageOptions {
                timestamp: Some(7),
                ..Default::default()
            }),
        ))]);
        let agent = Agent::new(AgentOptions {
            initial_state: Some(AgentState {
                model: provider.get_model(),
                ..Default::default()
            }),
            ..Default::default()
        });
        (provider, agent)
    }

    /// A waiter that observes `settled == false` must still be woken by the run that finishes
    /// afterwards. `finish_run` calls `Notified::notify_waiters`, which stores no permit, so a
    /// waiter that has not registered yet loses the wakeup and parks forever. The `agent_end`
    /// listener holds the run open so completion is placed *inside* the disputed window.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn idle_wait_has_no_lost_wakeup() {
        let mut parked = Vec::new();
        for repetition in 0..REPETITIONS {
            let (provider, agent) = faux_agent();
            let (reached, resume) = idle_wait_probe::arm();
            let entered = Arc::new(Semaphore::new(0));
            let release = Arc::new(Semaphore::new(0));
            let _subscription = agent.subscribe(Arc::new({
                let entered = entered.clone();
                let release = release.clone();
                move |event, _| {
                    let entered = entered.clone();
                    let release = release.clone();
                    Box::pin(async move {
                        if !matches!(event, AgentEvent::AgentEnd { .. }) {
                            return;
                        }
                        entered.add_permits(1);
                        release.acquire().await.unwrap().forget();
                    })
                }
            }));
            let run = tokio::spawn({
                let agent = agent.clone();
                async move {
                    agent
                        .prompt(PromptInput::Text {
                            input: "hi".to_string(),
                            images: Vec::new(),
                        })
                        .await
                }
            });
            // The run is inside its own `agent_end` listener, so it has not settled yet.
            timeout(SEAM_BUDGET, entered.acquire())
                .await
                .expect("the run never reached agent_end")
                .unwrap()
                .forget();
            assert!(agent.signal().is_some(), "the run must still be active");
            let waiter = tokio::spawn({
                let agent = agent.clone();
                async move { agent.wait_for_idle().await }
            });
            // The waiter parks inside the window: after the settled check, before registration.
            timeout(SEAM_BUDGET, reached.acquire())
                .await
                .expect("wait_for_idle never reached the awaited-value region")
                .unwrap()
                .forget();
            // Complete the run: `finish_run` notifies while the waiter may be unregistered.
            release.add_permits(1);
            timeout(SEAM_BUDGET, run)
                .await
                .expect("the run never finished")
                .unwrap()
                .unwrap();
            assert!(agent.signal().is_none(), "the run must have settled");
            resume.add_permits(1);
            match timeout(WAIT_BUDGET, waiter).await {
                Ok(joined) => joined.expect("the waiter task panicked"),
                Err(_) => parked.push(repetition),
            }
            provider.unregister();
        }
        assert!(
            parked.is_empty(),
            "wait_for_idle parked after the run finished (lost wakeup) in repetitions {parked:?} of {REPETITIONS}"
        );
    }

    /// Negative control: an idle agent and an already-settled run resolve every waiter at once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn idle_wait_resolves_for_idle_and_settled_runs() {
        let idle = Agent::new(AgentOptions::default());
        timeout(WAIT_BUDGET, idle.wait_for_idle())
            .await
            .expect("an idle agent must resolve immediately");

        let (provider, agent) = faux_agent();
        agent
            .prompt(PromptInput::Text {
                input: "hi".to_string(),
                images: Vec::new(),
            })
            .await
            .unwrap();
        let mut waiters = Vec::new();
        for _ in 0..8 {
            let agent = agent.clone();
            waiters.push(tokio::spawn(async move { agent.wait_for_idle().await }));
        }
        for waiter in waiters {
            timeout(WAIT_BUDGET, waiter)
                .await
                .expect("a settled run must resolve every waiter")
                .unwrap();
        }
        provider.unregister();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentToolResult, ThinkingLevel};
    use pi_ai::providers::faux::{
        faux_assistant_message, register_faux_provider, FauxAssistantMessageOptions,
        FauxResponseStep,
    };
    use serde_json::json;
    use std::time::Duration;
    use tokio::sync::Semaphore;
    use tokio::time::timeout;

    fn test_agent() -> Arc<Agent> {
        Agent::new(AgentOptions::default())
    }

    #[test]
    fn default_state_matches_typescript() {
        let agent = test_agent();
        let state = agent.state();
        assert_eq!(state.system_prompt, "");
        assert_eq!(state.model.id, "unknown");
        assert_eq!(state.thinking_level, ThinkingLevel::Off);
        assert_eq!(state.service_tier, Some(Some("default".to_string())));
        assert!(state.tools.is_none());
        assert!(state.messages.is_empty());
        assert!(!state.is_streaming);
        assert!(state.streaming_message.is_none());
        assert!(state.pending_tool_calls.is_empty());
        assert!(state.error_message.is_none());
        assert_eq!(agent.steering_mode(), QueueMode::OneAtATime);
        assert_eq!(agent.follow_up_mode(), QueueMode::OneAtATime);
        assert_eq!(*agent.transport.lock().unwrap(), "auto");
        assert_eq!(
            *agent.tool_execution.lock().unwrap(),
            ToolExecutionMode::Parallel
        );
    }

    #[test]
    fn assigning_tools_and_messages_copies_the_array() {
        let agent = test_agent();
        let mut state = agent.state();
        let mut tools = Vec::new();
        tools.push(crate::types::AgentTool {
            name: "bash".to_string(),
            description: "run".to_string(),
            parameters: json!({}),
            label: "Bash".to_string(),
            prepare_arguments: None,
            execute: Arc::new(|_, _, _, _| {
                Box::pin(async { Ok(AgentToolResult::new(Vec::new(), json!({}))) })
            }),
            execution_mode: None,
        });
        state.tools = Some(tools);
        agent.set_state(state);
        let first = agent.state().tools.clone().unwrap();
        let mut second = first.clone();
        second.push(first[0].clone());
        assert_eq!(
            agent.state().tools.as_ref().unwrap().len(),
            1,
            "the stored array is copied"
        );
    }

    #[test]
    fn pending_message_queue_drains_by_mode() {
        let mut queue = PendingMessageQueue::new(QueueMode::All);
        queue.enqueue_one(AgentMessage::from(assistant("a")));
        queue.enqueue(vec![
            AgentMessage::from(assistant("b")),
            AgentMessage::from(assistant("c")),
        ]);
        assert!(queue.has_items());
        assert_eq!(queue.drain().len(), 3);
        assert!(!queue.has_items());

        let mut one_at_a_time = PendingMessageQueue::new(QueueMode::OneAtATime);
        one_at_a_time.enqueue_one(AgentMessage::from(assistant("a")));
        one_at_a_time.enqueue_one(AgentMessage::from(assistant("b")));
        assert_eq!(one_at_a_time.drain().len(), 1);
        assert_eq!(one_at_a_time.drain().len(), 1);
        assert!(one_at_a_time.drain().is_empty());
    }

    #[test]
    fn remove_queued_messages_drops_whole_batches() {
        let agent = test_agent();
        agent.steer(vec![
            AgentMessage::from(assistant("keep")),
            AgentMessage::from(assistant("drop")),
        ]);
        agent.follow_up(vec![AgentMessage::from(assistant("drop"))]);
        let removed = agent.remove_queued_messages(&|message: &AgentMessage| match message {
            AgentMessage::Message(Message::Assistant(assistant)) => assistant.content.iter().any(|block| {
                matches!(block, pi_ai::types::ContentBlock::Text(text) if text.text == "drop")
            }),
            _ => false,
        });
        // The steering batch contains both messages and is dropped whole; the
        // follow-up batch is dropped too.
        assert_eq!(removed.len(), 3);
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn clear_all_queues_empties_both_queues() {
        let agent = test_agent();
        agent.steer(vec![AgentMessage::from(assistant("s"))]);
        agent.follow_up(vec![AgentMessage::from(assistant("f"))]);
        agent.clear_all_queues();
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn reset_clears_transcript_queues_and_runtime_state() {
        let agent = test_agent();
        agent.update_state(|state| {
            state.messages.push(AgentMessage::from(assistant("a")));
            state.is_streaming = true;
            state.error_message = Some("boom".to_string());
            state.pending_tool_calls.insert("call-1".to_string());
        });
        agent.steer(vec![AgentMessage::from(assistant("s"))]);
        agent.follow_up(vec![AgentMessage::from(assistant("f"))]);
        agent.reset();
        let state = agent.state();
        assert!(state.messages.is_empty());
        assert!(!state.is_streaming);
        assert!(state.error_message.is_none());
        assert!(state.pending_tool_calls.is_empty());
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn default_convert_to_llm_filters_custom_messages() {
        let converted = default_convert_to_llm(vec![
            AgentMessage::from(assistant("a")),
            AgentMessage::Custom(crate::types::CustomAgentMessage::BranchSummary {
                summary: "s".to_string(),
                from_id: "id".to_string(),
                timestamp: 1,
            }),
        ]);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role(), "assistant");
    }

    #[test]
    fn normalize_prompt_input_matches_typescript_shapes() {
        let agent = test_agent();
        let messages = agent.normalize_prompt_input(PromptInput::Text {
            input: "hello".to_string(),
            images: vec![ImageContent::new("data", "image/png")],
        });
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            AgentMessage::Message(Message::User(user)) => {
                assert_eq!(user.role, "user");
                match &user.content {
                    UserContent::Blocks(blocks) => {
                        assert_eq!(blocks.len(), 2);
                        assert!(matches!(blocks[0], ImageOrTextContent::Text(_)));
                        assert!(matches!(blocks[1], ImageOrTextContent::Image(_)));
                    }
                    UserContent::Text(_) => panic!("expected content blocks"),
                }
            }
            other => panic!("expected a user message, got {other:?}"),
        }
    }

    #[test]
    fn prompt_rejects_a_second_run_while_one_is_active() {
        let agent = test_agent();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            // `activeRun` is the guard: a run in flight rejects a second prompt.
            let aborted = CancellationToken::new();
            *agent.active_run.lock().unwrap() = Some(Arc::new(ActiveRun {
                aborted,
                idle: Arc::new(tokio::sync::Notify::new()),
                settled: Arc::new(AtomicBool::new(false)),
            }));
            let error = agent
                .prompt(PromptInput::Text {
                    input: "hello".to_string(),
                    images: Vec::new(),
                })
                .await
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
            );
            *agent.active_run.lock().unwrap() = None;
        });
    }

    #[test]
    fn subscribe_returns_a_working_unsubscribe_handle() {
        let agent = test_agent();
        let handle = agent.subscribe(Arc::new(|_, _| Box::pin(async {})));
        assert_eq!(agent.listeners.lock().unwrap().len(), 1);
        handle.unsubscribe();
        assert_eq!(agent.listeners.lock().unwrap().len(), 0);
    }

    #[test]
    fn agent_continue_reports_typed_precondition_codes() {
        let agent = test_agent();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let error = runtime.block_on(agent.continue_()).unwrap_err();
        assert_eq!(error.code, AgentContinueErrorCode::NothingToContinue);
        assert_eq!(error.message, "No messages to continue from");

        agent.update_state(|state| state.messages.push(AgentMessage::from(assistant("a"))));
        let error = runtime.block_on(agent.continue_()).unwrap_err();
        assert_eq!(
            error.message,
            "Cannot continue from message role: assistant"
        );
    }

    async fn assert_async_stop_hook(first_stop: Option<bool>) {
        let provider = register_faux_provider(None);
        provider.set_responses(vec![
            FauxResponseStep::Message(faux_assistant_message(
                pi_ai::types::ContentBlock::ToolCall(pi_ai::types::ToolCall::new(
                    "echo-1",
                    "echo",
                    serde_json::Map::new(),
                ))
                .into(),
                Some(FauxAssistantMessageOptions {
                    stop_reason: Some("toolUse".to_string()),
                    ..Default::default()
                }),
            )),
            FauxResponseStep::Message(faux_assistant_message("complete".into(), None)),
        ]);
        let started = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));
        let contexts = Arc::new(Mutex::new(Vec::<ShouldStopAfterTurnContext>::new()));
        let agent = Agent::new(AgentOptions {
            initial_state: Some(AgentState {
                model: provider.get_model(),
                tools: Some(vec![crate::types::AgentTool {
                    name: "echo".to_string(),
                    description: "Echo tool".to_string(),
                    parameters: json!({"type": "object", "properties": {}}),
                    label: "Echo".to_string(),
                    prepare_arguments: None,
                    execute: Arc::new(|_, _, _, _| {
                        Box::pin(async {
                            Ok(AgentToolResult::new(
                                vec![crate::types::ContentBlock::text("echoed")],
                                json!({}),
                            ))
                        })
                    }),
                    execution_mode: None,
                }]),
                ..Default::default()
            }),
            should_stop_after_turn: Some(Arc::new({
                let contexts = contexts.clone();
                let started = started.clone();
                let release = release.clone();
                move |context| {
                    let contexts = contexts.clone();
                    let started = started.clone();
                    let release = release.clone();
                    Box::pin(async move {
                        let first_turn = {
                            let mut contexts = contexts.lock().unwrap();
                            contexts.push(context);
                            contexts.len() == 1
                        };
                        if !first_turn {
                            return true;
                        }
                        started.add_permits(1);
                        release.acquire().await.unwrap().forget();
                        first_stop.unwrap_or(false)
                    })
                }
            })),
            ..Default::default()
        });
        agent.follow_up(vec![AgentMessage::from(UserMessage {
            role: "user".to_string(),
            content: UserContent::Text("follow up should stay queued".to_string()),
            provider_context: None,
            timestamp: 2,
        })]);
        let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
        let _subscription = agent.subscribe(Arc::new({
            let events = events.clone();
            move |event, _| {
                let events = events.clone();
                Box::pin(async move {
                    events.lock().unwrap().push(event);
                })
            }
        }));
        let run = tokio::spawn({
            let agent = agent.clone();
            async move {
                agent
                    .prompt(PromptInput::Text {
                        input: "echo something".to_string(),
                        images: Vec::new(),
                    })
                    .await
            }
        });
        timeout(Duration::from_secs(5), started.acquire())
            .await
            .expect("async stop hook did not start")
            .unwrap()
            .forget();

        assert!(!run.is_finished(), "the run must await the pending hook");
        assert!(agent.state().is_streaming);
        assert_eq!(provider.call_count(), 1);
        assert!(matches!(
            events.lock().unwrap().last(),
            Some(AgentEvent::TurnEnd { .. })
        ));
        {
            let contexts = contexts.lock().unwrap();
            let context = &contexts[0];
            assert_eq!(context.message.stop_reason, "toolUse");
            assert_eq!(context.tool_results.len(), 1);
            assert_eq!(context.tool_results[0].tool_call_id, "echo-1");
            assert!(!context.tool_results[0].is_error);
            assert_eq!(
                context
                    .context
                    .messages
                    .iter()
                    .map(AgentMessage::role)
                    .collect::<Vec<_>>(),
                ["user", "assistant", "toolResult"],
            );
            assert_eq!(context.new_messages, context.context.messages);
            assert_eq!(context.new_messages, agent.state().messages);
        }

        if first_stop.is_none() {
            agent.abort();
        } else {
            release.add_permits(1);
        }
        timeout(Duration::from_secs(5), run)
            .await
            .expect("pending hook blocked run settlement")
            .unwrap()
            .unwrap();
        let expected_turns = if first_stop == Some(false) { 2 } else { 1 };
        assert_eq!(provider.call_count(), expected_turns);
        assert_eq!(contexts.lock().unwrap().len(), expected_turns as usize);
        let state = agent.state();
        assert!(!state.is_streaming);
        assert!(state.error_message.is_none());
        assert!(state.pending_tool_calls.is_empty());
        assert_eq!(state.messages.len(), expected_turns as usize + 2);
        assert!(
            agent.has_queued_messages(),
            "stopping must retain queued follow-ups"
        );
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::TurnEnd { .. }))
                .count(),
            expected_turns as usize
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::AgentEnd { .. }))
                .count(),
            1
        );
        match events.last().unwrap() {
            AgentEvent::AgentEnd { messages } => assert_eq!(messages, &state.messages),
            event => panic!("expected agent_end, got {event:?}"),
        }
        assert!(
            !state.messages.iter().any(|message| matches!(
                message,
                AgentMessage::Message(Message::Assistant(message))
                    if message.stop_reason == "aborted" || message.stop_reason == "error"
            )),
            "post-turn cancellation must not append an aborted assistant"
        );
        provider.unregister();
    }

    #[tokio::test]
    async fn async_stop_after_turn_false_waits_then_continues_tool_cycle() {
        assert_async_stop_hook(Some(false)).await;
    }

    #[tokio::test]
    async fn async_stop_after_turn_true_waits_then_preserves_queued_work() {
        assert_async_stop_hook(Some(true)).await;
    }

    #[tokio::test]
    async fn async_stop_after_turn_cancellation_settles_completed_turn() {
        assert_async_stop_hook(None).await;
    }

    #[tokio::test]
    async fn async_context_auth_tool_and_continuation_hooks_keep_turn_order() {
        let provider = register_faux_provider(None);
        provider.set_responses(vec![
            FauxResponseStep::Message(faux_assistant_message(
                pi_ai::types::ContentBlock::ToolCall(pi_ai::types::ToolCall::new(
                    "echo-1",
                    "echo",
                    serde_json::Map::new(),
                ))
                .into(),
                Some(FauxAssistantMessageOptions {
                    stop_reason: Some("toolUse".to_string()),
                    ..Default::default()
                }),
            )),
            FauxResponseStep::Message(faux_assistant_message("complete".into(), None)),
        ]);
        let trace = Arc::new(Mutex::new(Vec::new()));
        let agent = Agent::new(AgentOptions {
            initial_state: Some(AgentState {
                model: provider.get_model(),
                tools: Some(vec![crate::types::AgentTool {
                    name: "echo".to_string(),
                    description: "Echo tool".to_string(),
                    label: "Echo".to_string(),
                    parameters: json!({"type": "object", "properties": {}}),
                    prepare_arguments: None,
                    execute: Arc::new(|_, _, _, _| {
                        Box::pin(async {
                            Ok(AgentToolResult::new(
                                vec![crate::types::ContentBlock::text("original")],
                                json!({}),
                            ))
                        })
                    }),
                    execution_mode: None,
                }]),
                ..Default::default()
            }),
            before_request: Some(Arc::new({
                let trace = trace.clone();
                move |request_index, signal| {
                    let trace = trace.clone();
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        assert!(signal.is_some());
                        trace.lock().unwrap().push(if request_index == 0 {
                            "request-0"
                        } else {
                            "request-1"
                        });
                        Ok(())
                    })
                }
            })),
            transform_context: Some(Arc::new({
                let trace = trace.clone();
                move |messages, signal| {
                    let trace = trace.clone();
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        assert!(signal.is_some());
                        trace.lock().unwrap().push("transform");
                        messages
                    })
                }
            })),
            convert_to_llm: Some(Arc::new({
                let trace = trace.clone();
                move |messages| {
                    let trace = trace.clone();
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        trace.lock().unwrap().push("convert");
                        default_convert_to_llm(messages)
                    })
                }
            })),
            get_api_key: Some(Arc::new({
                let trace = trace.clone();
                move |provider| {
                    let trace = trace.clone();
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        assert_eq!(provider, "faux");
                        trace.lock().unwrap().push("auth");
                        None
                    })
                }
            })),
            before_tool_call: Some(Arc::new({
                let trace = trace.clone();
                move |context, signal| {
                    let trace = trace.clone();
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        assert!(signal.is_some());
                        assert_eq!(context.tool_call.id, "echo-1");
                        assert_eq!(context.assistant_message.stop_reason, "toolUse");
                        trace.lock().unwrap().push("before");
                        Ok(None)
                    })
                }
            })),
            after_tool_call: Some(Arc::new({
                let trace = trace.clone();
                move |context, signal| {
                    let trace = trace.clone();
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        assert!(signal.is_some());
                        assert_eq!(context.result.content[0].as_text(), Some("original"));
                        trace.lock().unwrap().push("after");
                        Ok(Some(AfterToolCallResult {
                            content: Some(vec![crate::types::ContentBlock::text("overridden")]),
                            details: Some(json!({"changed": true})),
                            is_error: Some(true),
                            terminate: Some(true),
                        }))
                    })
                }
            })),
            get_continuation_messages: Some(Arc::new({
                let trace = trace.clone();
                move |context, signal| {
                    let trace = trace.clone();
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        assert!(signal.is_some());
                        trace.lock().unwrap().push("continuation");
                        if context.tool_results.is_empty() {
                            return Vec::new();
                        }
                        assert_eq!(context.new_messages.len(), 3);
                        assert!(context.tool_results[0].is_error);
                        assert_eq!(
                            context.tool_results[0].details,
                            Some(json!({"changed": true}))
                        );
                        assert!(
                            matches!(&context.tool_results[0].content[0], ImageOrTextContent::Text(text) if text.text == "overridden")
                        );
                        vec![AgentMessage::from(UserMessage {
                            role: "user".to_string(),
                            content: UserContent::Text("continue".to_string()),
                            provider_context: None,
                            timestamp: 2,
                        })]
                    })
                }
            })),
            ..Default::default()
        });
        timeout(
            Duration::from_secs(5),
            agent.prompt(PromptInput::Text {
                input: "start".to_string(),
                images: Vec::new(),
            }),
        )
        .await
        .expect("async hook pipeline did not settle")
        .unwrap();
        assert_eq!(provider.call_count(), 2);
        assert!(agent.state().error_message.is_none());
        assert_eq!(
            *trace.lock().unwrap(),
            [
                "request-0",
                "transform",
                "convert",
                "auth",
                "before",
                "after",
                "continuation",
                "request-1",
                "transform",
                "convert",
                "auth",
                "continuation",
            ]
        );
        assert_eq!(
            agent
                .state()
                .messages
                .iter()
                .map(AgentMessage::role)
                .collect::<Vec<_>>(),
            ["user", "assistant", "toolResult", "user", "assistant",]
        );
        provider.unregister();
    }

    fn assistant(text: &str) -> AssistantMessage {
        let mut message = AssistantMessage::new("openai-responses", "openai", "mock", 1);
        message
            .content
            .push(pi_ai::types::ContentBlock::Text(TextContent::new(text)));
        message
    }
}
