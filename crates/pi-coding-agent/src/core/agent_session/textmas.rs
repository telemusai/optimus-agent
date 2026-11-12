//! One-shot, session-owned TextMAS preparation. Advisory calls have no Agent/tool loop.
use super::*;
use crate::core::occ::textmas::{self as contract, TextMasCallStatus, TextMasMode, TextMasRole, TextMasUsageRecord};
use pi_ai::types::{AssistantMessageEvent, Context, SimpleStreamOptions};
use pi_ai::utils::event_stream::StreamTaskReceipt;

type ContextTransform = Arc<dyn Fn(Vec<AgentMessage>, Option<CancellationToken>) -> BoxFuture<Vec<AgentMessage>> + Send + Sync>;

#[derive(Default)]
pub(super) struct TextMasState {
    context_hook_installed: bool,
    original_transform: Option<ContextTransform>,
    run: Option<TextMasRun>,
}

#[derive(Clone)]
struct TextMasRun {
    action_id: String,
    input_ids: Vec<String>,
    last_user: AgentMessage,
    mode: TextMasMode,
    model: Model,
    signal: CancellationToken,
    phase: Phase,
}

#[derive(Clone)]
enum Phase {
    Armed,
    Preparing,
    Ready { advisory: UserMessage, snapshot: Option<PreparedSnapshot> },
    Failed(String),
}

#[derive(Clone)]
struct PreparedSnapshot {
    source: Vec<AgentMessage>,
    transformed: Vec<AgentMessage>,
}

/// Covers the complete logical turn, including ordinary final-request retries.
/// Dropping the owner clears drafts; no later internal continuation can acquire them.
pub(super) struct TurnGuard(Weak<AgentSession>);
impl TurnGuard {
    pub(super) fn new(session: &Arc<AgentSession>) -> Self { Self(Arc::downgrade(session)) }
}
impl Drop for TurnGuard {
    fn drop(&mut self) {
        if let Some(session) = self.0.upgrade() {
            if let Some(run) = session.textmas.lock().unwrap().run.take() { run.signal.cancel(); }
        }
    }
}

/// The outer Agent may drop before_request on abort. Never leave a consumed arm reusable.
struct PreparationGuard { session: Weak<AgentSession>, action_id: String, completed: bool }
impl Drop for PreparationGuard {
    fn drop(&mut self) {
        if !self.completed {
            if let Some(session) = self.session.upgrade() {
                session.fail_textmas(&self.action_id, "OCC3.1 preparation was cancelled or interrupted; no automatic replay".into());
            }
        }
    }
}

/// Started usage is durable before invocation. Cancellation records unknown rather than free.
/// Provider task receipts share the parent execution scope and are cancelled on every exit.
struct CallGuard {
    session: Weak<AgentSession>,
    action_id: String,
    role: TextMasRole,
    signal: CancellationToken,
    receipt: StreamTaskReceipt,
    usage: Option<Usage>,
    settled: bool,
}
impl Drop for CallGuard {
    fn drop(&mut self) {
        self.signal.cancel();
        self.receipt.request_cancel();
        if !self.settled {
            if let Some(session) = self.session.upgrade() {
                let record = TextMasUsageRecord::new(&self.action_id, self.role, TextMasCallStatus::Cancelled, self.usage.clone());
                if session.persist_textmas_usage(&record).is_err() {
                    session.fail_textmas(&self.action_id, "OCC3.1 usage could not be saved; no automatic replay".into());
                }
            }
        }
    }
}

impl AgentSession {
    /// Genuine root user delivery is the only activation authority, not request_index or preference alone.
    pub(super) fn arm_textmas(self: &Arc<Self>, turns: &[QueuedSessionAction]) -> Result<(), String> {
        if self.rlm_depth != 0 || self.semantic_parent_session_id.is_some() { return Ok(()); }
        use crate::core::session_action_store::ActionSource;
        use crate::core::extensions::types::InputSource;
        let Some(action) = turns.iter().find(|action| {
            matches!(action.source, ActionSource::Input(InputSource::Interactive | InputSource::Rpc))
                && primary_delivery_record(action).ok().is_some_and(|record| matches!(record.message, DeliveryMessage::User(_)))
        }) else { return Ok(()); };
        let Some(file) = self.session_file() else { return Ok(()); };
        let mode = crate::core::occ::textmas_settings::resolve(&file)
            .map_err(|_| "OCC3.1 session preference could not be loaded; no model request was started".to_string())?.mode;
        if mode == TextMasMode::Off { return Ok(()); }
        if self.agent.side_question_options().is_none() {
            return Err("OCC3.1 requires the native provider invocation interface".into());
        }
        // Persisted reservations make restart/retry fail closed rather than repeat spending.
        if self.session_manager.lock().unwrap().get_entries().iter()
            .filter_map(contract::usage_record).any(|record| record.action_id == action.id) {
            return Err("OCC3.1 preparation for this action was already attempted; automatic replay is disabled".into());
        }
        let last_user = turns.iter().filter_map(|action| primary_delivery_record(action).ok())
            .map(|record| agent_message_from_delivery(&record.message))
            .filter(|message| matches!(message, AgentMessage::Message(Message::User(_))))
            .last().ok_or("OCC3.1 delivery contains no user request")?;
        self.install_textmas_context_hook()?;
        let mut state = self.textmas.lock().unwrap();
        if state.run.is_some() { return Err("OCC3.1 already owns an active turn".into()); }
        state.run = Some(TextMasRun { action_id: action.id.clone(), input_ids: turns.iter().map(|action| action.id.clone()).collect(), last_user, mode, model: self.agent.model(), signal: CancellationToken::new(), phase: Phase::Armed });
        Ok(())
    }

    /// Compose, never replace, the existing extension/Jev context transform. Only run-owned USER data is added.
    fn install_textmas_context_hook(self: &Arc<Self>) -> Result<(), String> {
        let mut state = self.textmas.lock().unwrap();
        if state.context_hook_installed { return Ok(()); }
        let original = self.agent.side_question_options().ok_or("OCC3.1 provider context is unavailable")?.transform_context;
        state.original_transform = original.clone();
        let weak = Arc::downgrade(self);
        self.agent.set_transform_context(Arc::new(move |messages, signal| {
            let original = original.clone();
            let weak = weak.clone();
            Box::pin(async move {
                // Reuse exactly the transformed initial snapshot: extension hooks run once, as before.
                let cached = weak.upgrade().and_then(|session| {
                    let mut state = session.textmas.lock().unwrap();
                    let run = state.run.as_mut()?;
                    if run.signal.is_cancelled() { return None; }
                    match &mut run.phase {
                        Phase::Ready { snapshot, .. } if snapshot.as_ref().is_some_and(|snapshot| snapshot.source == messages) =>
                            snapshot.take().map(|snapshot| snapshot.transformed),
                        _ => None,
                    }
                });
                let latest_user = messages.iter().rev().find(|message| matches!(message, AgentMessage::Message(Message::User(_)))).cloned();
                let mut messages = match cached {
                    Some(messages) => messages,
                    None => match original { Some(transform) => transform(messages, signal.clone()).await, None => messages },
                };
                if signal.as_ref().is_some_and(CancellationToken::is_cancelled) { return messages; }
                if let Some(session) = weak.upgrade() {
                    let state = session.textmas.lock().unwrap();
                    if let Some(run) = &state.run {
                        // Bind the full delivered batch, not only its first queued action.
                        // Auxiliary questions and later steering carry a different latest user input.
                        if !run.signal.is_cancelled() && run.model == session.agent.model() && latest_user.as_ref() == Some(&run.last_user) {
                            if let Phase::Ready { advisory, .. } = &run.phase {
                                messages.push(AgentMessage::Message(Message::User(advisory.clone())));
                            }
                        }
                    }
                }
                messages
            })
        }));
        state.context_hook_installed = true;
        Ok(())
    }

    /// True while this session owns an active TextMAS run (armed, preparing,
    /// or ready for the parent turn). Display signal only.
    pub fn is_textmas_active(&self) -> bool {
        self.textmas.lock().unwrap().run.as_ref()
            .is_some_and(|run| !matches!(run.phase, Phase::Failed(_)))
    }

    pub(super) fn textmas_failure(&self) -> Option<String> {
        self.textmas.lock().unwrap().run.as_ref().and_then(|run| match &run.phase {
            Phase::Failed(error) => Some(error.clone()), _ => None,
        })
    }

    /// Explicit abort changes the pump epoch, unlike a model switch. Its deferred-error
    /// branch cannot finish an already durable Running input. Once Agent/events settle,
    /// cancel only this failed preparation's inputs so their completion tickets settle.
    pub(super) fn settle_aborted_textmas_inputs(&self, epoch: u64, error: &str) -> Result<(), String> {
        if self.session_input_pump_epoch.load(Ordering::SeqCst) == epoch { return Ok(()); }
        let input_ids = {
            let state = self.textmas.lock().unwrap();
            let Some(run) = state.run.as_ref().filter(|run| matches!(run.phase, Phase::Failed(_))) else { return Ok(()); };
            run.input_ids.clone()
        };
        // Do not hold the TextMAS mutex while the action store or cancellation path runs.
        let (candidates, tickets) = {
            let store = self.action_store.lock().unwrap();
            let candidates: Vec<QueuedSessionAction> = store.owned_actions().into_iter()
                .filter(|action| input_ids.contains(&action.id)
                    && action.lifecycle.state() == ActionLifecycleState::Running
                    && primary_delivery_record(action).is_ok_and(|record| record.durable))
                .collect();
            let tickets = candidates.iter().map(|action| store.ticket_for(action)).collect::<Result<Vec<_>, _>>()?;
            (candidates, tickets)
        };
        if candidates.is_empty() { return Ok(()); }
        let expected = candidates.len();
        let cancelled = self.cancel_session_actions(&|_| true, error, Some(candidates));
        if cancelled.len() != expected {
            let failure = format!("{error}; OCC3.1 cancelled input settlement could not be fully recorded");
            // A cleanup failure must be visible to the waiting callers, never a success.
            for ticket in tickets { ticket.settle_completed(Some(failure.clone())); }
            return Err(failure);
        }
        Ok(())
    }

    fn fail_textmas(&self, action_id: &str, error: String) {
        if let Some(run) = self.textmas.lock().unwrap().run.as_mut().filter(|run| run.action_id == action_id) {
            run.signal.cancel();
            run.phase = Phase::Failed(error);
        }
    }

    /// A model switch cannot reinterpret drafts or leave their original provider running.
    pub(super) fn cancel_textmas_for_model_change(&self, model: &Model) {
        let changed = self.textmas.lock().unwrap().run.as_ref().filter(|run| &run.model != model).map(|run| run.action_id.clone());
        if let Some(action) = changed {
            self.fail_textmas(&action, "OCC3.1 model changed during the turn; preparation was cancelled".into());
            self.agent.abort();
        }
    }

    /// Called after durable input MessageEnd and within the parent's cancellation lifecycle.
    pub(super) async fn prepare_textmas(self: &Arc<Self>, signal: Option<CancellationToken>) -> Result<(), String> {
        let run = {
            let mut state = self.textmas.lock().unwrap();
            let Some(run) = state.run.as_mut() else { return Ok(()); };
            match &run.phase {
                Phase::Ready { .. } => return Ok(()),
                Phase::Failed(error) => return Err(error.clone()),
                Phase::Preparing => return Err("OCC3.1 preparation is already in flight".into()),
                Phase::Armed => {},
            }
            run.phase = Phase::Preparing;
            run.clone()
        };
        let mut guard = PreparationGuard { session: Arc::downgrade(self), action_id: run.action_id.clone(), completed: false };
        let result = self.run_textmas_preparation(&run, signal).await;
        match result {
            Ok((advisory, snapshot)) => {
                let mut state = self.textmas.lock().unwrap();
                let current = state.run.as_mut().filter(|current| current.action_id == run.action_id)
                    .ok_or("OCC3.1 turn ended during preparation")?;
                if current.signal.is_cancelled() || current.model != self.agent.model() {
                    return Err("OCC3.1 preparation was cancelled".into());
                }
                current.phase = Phase::Ready { advisory, snapshot: Some(snapshot) };
                guard.completed = true;
                Ok(())
            }
            Err(error) => {
                self.fail_textmas(&run.action_id, error.clone());
                guard.completed = true;
                Err(error)
            }
        }
    }

    async fn run_textmas_preparation(self: &Arc<Self>, run: &TextMasRun, parent_signal: Option<CancellationToken>) -> Result<(UserMessage, PreparedSnapshot), String> {
        let durable = run.input_ids.iter().all(|id| self.action_by_id(id).as_ref()
            .and_then(|action| primary_delivery_record(action).ok()).is_some_and(|record| record.durable));
        if !durable || self.has_failed_dispatch_persistence() {
            return Err("OCC3.1 requires a durably saved user request before preparation".into());
        }
        let options = self.agent.side_question_options().ok_or("OCC3.1 provider interface is unavailable")?;
        let convert = options.convert_to_llm.ok_or("OCC3.1 provider context conversion is unavailable")?;
        let source = self.agent.messages();
        let transform = self.textmas.lock().unwrap().original_transform.clone();
        // Bound the effective, extension-transformed and policy-converted snapshot, not stale full history.
        let (transformed, messages) = tokio::select! {
            biased;
            _ = run.signal.cancelled() => return Err("OCC3.1 preparation was cancelled".into()),
            _ = cancelled(parent_signal.as_ref()) => return Err("OCC3.1 preparation was cancelled".into()),
            result = tokio::time::timeout(std::time::Duration::from_secs(contract::STAGE_TIMEOUT_SECONDS), async {
                let transformed = match transform { Some(transform) => transform(source.clone(), parent_signal.clone()).await, None => source.clone() };
                let messages = convert(transformed.clone()).await;
                (transformed, messages)
            }) => result.map_err(|_| "OCC3.1 context preparation timed out")?,
        };
        let original = Context::new(Some(self.agent.system_prompt()), messages, None);
        contract::check_context_bound(&original)?;
        let mut drafts = Vec::new();
        for role in TextMasRole::ALL {
            if run.signal.is_cancelled() || parent_signal.as_ref().is_some_and(CancellationToken::is_cancelled) || self.agent.model() != run.model {
                return Err("OCC3.1 preparation was cancelled or its model changed".into());
            }
            let context = contract::role_context(&original, run.mode, role, &drafts)?;
            let draft = self.run_textmas_role(run, role, context, parent_signal.as_ref()).await?;
            drafts.push((role, draft));
        }
        let advisory = contract::advisory_message(&drafts);
        let mut final_context = original;
        final_context.messages.push(Message::User(advisory.clone()));
        contract::check_context_bound(&final_context)?;
        Ok((advisory, PreparedSnapshot { source, transformed }))
    }

    async fn run_textmas_role(self: &Arc<Self>, run: &TextMasRun, role: TextMasRole, context: Context, parent_signal: Option<&CancellationToken>) -> Result<String, String> {
        let output_budget = contract::output_budget(&run.model, &context)?;
        let started = TextMasUsageRecord::new(&run.action_id, role, TextMasCallStatus::Started, None);
        self.persist_textmas_usage(&started)?;
        let receipt = StreamTaskReceipt::new_unsupported();
        if let Some(scope) = self.agent.execution_scope() {
            scope.register_stream(receipt.clone()).map_err(|_| "OCC3.1 execution scope was stopped")?;
        }
        let mut guard = CallGuard { session: Arc::downgrade(self), action_id: run.action_id.clone(), role, signal: run.signal.child_token(), receipt: receipt.clone(), usage: None, settled: false };
        let options = self.agent.side_question_options().ok_or("OCC3.1 provider interface is unavailable")?;
        let mut request = SimpleStreamOptions::default();
        request.stream.signal = Some(guard.signal.clone());
        request.stream.timeout_ms = Some(contract::STAGE_TIMEOUT_SECONDS as f64 * 1000.0);
        request.stream.max_tokens = Some(output_budget);
        request.stream.service_tier = self.agent.service_tier();
        request.stream.transport = Some(self.settings_manager.lock().unwrap().get_transport());
        request.stream.session_id = Some(format!("{}:occ31:{}:{}", self.session_id(), run.action_id, role.as_str()));
        request.reasoning = Some(self.agent.thinking_level().as_str().into());
        request.thinking_budgets = options.thinking_budgets;
        // Use the real selected provider closure: registry auth, headers, and custom APIs all remain intact.
        // No on_payload extension, latent lease, source capture, or tools are installed on advisory requests.
        let stream_fn = crate::core::semantic_edges::unwrap_semantic_edge_stream_fn(&self.agent.stream_fn());
        let result = tokio::select! {
            biased;
            _ = run.signal.cancelled() => Err((TextMasCallStatus::Cancelled, "preparation was cancelled".to_string())),
            _ = cancelled(parent_signal) => Err((TextMasCallStatus::Cancelled, "preparation was cancelled".to_string())),
            result = tokio::time::timeout(std::time::Duration::from_secs(contract::STAGE_TIMEOUT_SECONDS), async {
                if let Some(get_key) = options.get_api_key { request.stream.api_key = get_key(run.model.provider.clone()).await; }
                let stream = receipt.scope(async { stream_fn(run.model.clone(), context, request).await }).await;
                while let Some(event) = stream.next().await {
                    if let Some(partial) = event.partial() {
                        guard.usage = Some(partial.usage.clone());
                        if serde_json::to_vec(partial).map_or(true, |bytes| bytes.len() > contract::MAX_STAGE_RESPONSE_BYTES) {
                            return Err((TextMasCallStatus::Failed, "preparation response exceeded its bound".into()));
                        }
                        if partial.content.iter().any(|block| matches!(block, pi_ai::types::ContentBlock::ToolCall(_))) {
                            return Err((TextMasCallStatus::Failed, "preparation returned a tool call; no tool was executed".into()));
                        }
                    }
                    let terminal = match event { AssistantMessageEvent::Done { message, .. } => Some(message), AssistantMessageEvent::Error { error, .. } => Some(error), _ => None };
                    if let Some(message) = terminal {
                        guard.usage = Some(message.usage.clone());
                        return contract::public_draft(&message).map_err(|error| (TextMasCallStatus::Failed, error));
                    }
                }
                // Some injected/native adapters end directly with a terminal result.
                if let Some(message) = stream.result_if_ready() {
                    guard.usage = Some(message.usage.clone());
                    contract::public_draft(&message).map_err(|error| (TextMasCallStatus::Failed, error))
                } else { Err((TextMasCallStatus::Failed, "provider ended without a result".into())) }
            }) => result.unwrap_or_else(|_| Err((TextMasCallStatus::TimedOut, "preparation timed out".into()))),
        };
        let status = match &result { Ok(_) => TextMasCallStatus::Completed, Err((status, _)) => *status };
        let terminal = TextMasUsageRecord::new(&run.action_id, role, status, guard.usage.clone());
        // No second write on a failed terminal persist: the reservation remains unknown and replay stays blocked.
        guard.settled = true;
        self.persist_textmas_usage(&terminal)?;
        result.map_err(|(_, error)| format!("OCC3.1 {}: {error}. No final execution or automatic preparation retry.", role.as_str()))
    }

    fn persist_textmas_usage(&self, record: &TextMasUsageRecord) -> Result<(), String> {
        let mut manager = self.session_manager.lock().unwrap();
        manager.append_custom_entry_with_rollback(contract::USAGE_CUSTOM_TYPE, Some(serde_json::to_value(record).map_err(|_| "OCC3.1 usage encoding failed")?))
            .and_then(|_| manager.flush_now())
            .map_err(|_| "OCC3.1 usage could not be saved; no final execution or automatic replay".to_string())?;
        *self.own_usage_memo.lock().unwrap() = None;
        Ok(())
    }
}

async fn cancelled(signal: Option<&CancellationToken>) {
    match signal { Some(signal) => signal.cancelled().await, None => std::future::pending::<()>().await }
}
