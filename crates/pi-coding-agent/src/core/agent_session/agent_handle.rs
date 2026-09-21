//! Private bridge from the session's Agent surface to the real foundation Agent.

#[path = "runtime_bridge.rs"]
mod runtime_bridge;

use super::{
    AfterToolCallHook, AgentHandle, BeforeRequestHook, BeforeToolCallHook, BoxFuture,
    GetContinuationMessagesHook,
};
use pi_agent_core::agent::{Agent, AgentContinueError, PromptInput, QueueMode};
use pi_agent_core::performance_metrics::AgentLoopPerformanceMetrics;
use pi_agent_core::types::{
    AgentEvent, AgentMessage, AgentState, ShouldStopAfterTurnContext, StreamFn, ToolExecutionMode,
};
use pi_ai::types::{Message, OnPayload, OnResponse};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

impl AgentHandle for Arc<Agent> {
    fn side_question_options(&self) -> Option<pi_agent_core::agent::AgentOptions> {
        Some(pi_agent_core::agent::AgentOptions {
            initial_state: Some(self.state()),
            convert_to_llm: Some(
                self.convert_to_llm
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
            ),
            transform_context: self
                .transform_context
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            stream_fn: Some(crate::core::semantic_edges::unwrap_semantic_edge_stream_fn(
                &self.stream_fn(),
            )),
            get_api_key: self
                .get_api_key
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            on_payload: self
                .on_payload
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            on_response: self
                .on_response
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            session_id: self
                .session_id
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            thinking_budgets: self
                .thinking_budgets
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            tool_execution: Some(
                *self
                    .tool_execution
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
            ),
            ..Default::default()
        })
    }
    fn state(&self) -> AgentState {
        Agent::state(self)
    }

    fn model(&self) -> pi_ai::types::Model {
        self.read_state(|state| state.model.clone())
    }
    fn thinking_level(&self) -> pi_agent_core::types::ThinkingLevel {
        self.read_state(|state| state.thinking_level)
    }
    fn service_tier(&self) -> pi_ai::types::ServiceTier {
        self.read_state(|state| state.service_tier.clone())
    }
    fn system_prompt(&self) -> String {
        self.read_state(|state| state.system_prompt.clone())
    }
    fn message_count(&self) -> usize {
        self.read_state(|state| state.messages.len())
    }
    fn messages(&self) -> Vec<AgentMessage> {
        self.read_state(|state| state.messages.clone())
    }
    fn streaming_message(&self) -> Option<AgentMessage> {
        self.read_state(|state| state.streaming_message.clone())
    }
    fn active_tool_names(&self) -> Vec<String> {
        self.read_state(|state| {
            state
                .tools
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(|tool| tool.name.clone())
                .collect()
        })
    }

    fn set_state(&self, state: AgentState) {
        Agent::set_state(self, state);
    }

    fn update_state(&self, update: Box<dyn FnOnce(&mut AgentState) + Send>) {
        Agent::update_state(self, update);
    }

    fn subscribe(
        &self,
        listener: Arc<dyn Fn(AgentEvent, Option<CancellationToken>) -> BoxFuture<()> + Send + Sync>,
    ) -> Box<dyn Fn() + Send + Sync> {
        let subscription = Agent::subscribe(self, listener);
        Box::new(move || subscription.unsubscribe())
    }

    fn set_before_tool_call(&self, hook: BeforeToolCallHook) {
        *self
            .before_tool_call
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(Arc::new(move |context, signal| {
            let result = hook(context, signal);
            Box::pin(async move { result.await.map_err(anyhow::Error::msg) })
        }));
    }

    fn set_after_tool_call(&self, hook: AfterToolCallHook) {
        *self
            .after_tool_call
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(Arc::new(move |context, signal| {
            let result = hook(context, signal);
            Box::pin(async move { result.await.map_err(anyhow::Error::msg) })
        }));
    }

    fn set_get_continuation_messages(&self, hook: GetContinuationMessagesHook) {
        *self
            .get_continuation_messages
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(hook);
    }

    fn set_should_stop_before_turn(&self, hook: Arc<dyn Fn() -> bool + Send + Sync>) {
        *self
            .should_stop_before_turn
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(hook);
    }

    fn set_before_request(&self, hook: BeforeRequestHook) {
        *self
            .before_request
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(hook);
    }

    fn set_should_stop_after_turn(
        &self,
        hook: Arc<dyn Fn(ShouldStopAfterTurnContext) -> BoxFuture<bool> + Send + Sync>,
    ) {
        *self
            .should_stop_after_turn
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(hook);
    }

    fn set_stream_fn(&self, stream_fn: StreamFn) {
        *self
            .stream_fn
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = stream_fn;
    }

    fn stream_fn(&self) -> StreamFn {
        self.stream_fn
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    fn abort(&self) {
        Agent::abort(self);
    }

    fn wait_for_idle(&self) -> BoxFuture<()> {
        let agent = self.clone();
        Box::pin(async move { Agent::wait_for_idle(&agent).await })
    }

    fn prompt(&self, messages: Vec<AgentMessage>) -> BoxFuture<Result<(), String>> {
        let agent = self.clone();
        Box::pin(async move {
            claim_error_metric_settlement(&agent);
            Agent::prompt(&agent, PromptInput::Messages(messages))
                .await
                .map_err(|error| error.to_string())
        })
    }

    fn continue_(&self) -> BoxFuture<Result<(), AgentContinueError>> {
        let agent = self.clone();
        Box::pin(async move {
            claim_error_metric_settlement(&agent);
            Agent::continue_(&agent).await
        })
    }

    fn is_streaming(&self) -> bool {
        self.read_state(|state| state.is_streaming)
    }

    fn has_queued_messages(&self) -> bool {
        Agent::has_queued_messages(self)
    }

    fn clear_all_queues(&self) {
        Agent::clear_all_queues(self);
    }

    fn remove_queued_messages(
        &self,
        predicate: Arc<dyn Fn(&AgentMessage) -> bool + Send + Sync>,
    ) -> Vec<AgentMessage> {
        Agent::remove_queued_messages(self, predicate.as_ref())
    }

    fn follow_up(&self, message: AgentMessage) {
        Agent::follow_up(self, vec![message]);
    }

    fn set_follow_up_mode(&self, mode: String) {
        Agent::set_follow_up_mode(self, queue_mode(&mode));
    }

    fn set_steering_mode(&self, mode: String) {
        Agent::set_steering_mode(self, queue_mode(&mode));
    }

    fn set_convert_to_llm(
        &self,
        convert: Arc<dyn Fn(Vec<AgentMessage>) -> BoxFuture<Vec<Message>> + Send + Sync>,
    ) {
        *self
            .convert_to_llm
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = convert;
    }

    fn set_transform_context(
        &self,
        transform: Arc<
            dyn Fn(Vec<AgentMessage>, Option<CancellationToken>) -> BoxFuture<Vec<AgentMessage>>
                + Send
                + Sync,
        >,
    ) {
        *self
            .transform_context
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(transform);
    }

    fn set_get_api_key(
        &self,
        get_api_key: Arc<dyn Fn(String) -> BoxFuture<Option<String>> + Send + Sync>,
    ) {
        *self
            .get_api_key
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(get_api_key);
    }

    fn set_on_payload(&self, hook: OnPayload) {
        *self
            .on_payload
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(hook);
    }

    fn set_on_response(&self, hook: OnResponse) {
        *self
            .on_response
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(hook);
    }

    fn set_tool_execution(&self, mode: String) {
        *self
            .tool_execution
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = match mode.as_str() {
            "sequential" => ToolExecutionMode::Sequential,
            _ => ToolExecutionMode::Parallel,
        };
    }

    fn performance_metrics(&self) -> Option<AgentLoopPerformanceMetrics> {
        self.performance_metrics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    fn set_performance_metrics(&self, metrics: Option<AgentLoopPerformanceMetrics>) {
        *self
            .performance_metrics
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = metrics;
    }

    fn signal(&self) -> Option<CancellationToken> {
        Agent::signal(self)
    }
}

fn claim_error_metric_settlement(agent: &Arc<Agent>) {
    if let Some(metrics) = agent
        .performance_metrics
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .as_mut()
    {
        // The session decides whether an error is retried after message_end.
        // Claim that decision before the first attempt, not after it failed.
        metrics.host_owns_logical_request_terminal = true;
    }
}

fn queue_mode(mode: &str) -> QueueMode {
    match mode {
        "all" => QueueMode::All,
        _ => QueueMode::OneAtATime,
    }
}
