//! One-shot, session-owned TextMAS preparation. Advisory calls have no Agent/tool loop.
use super::*;
use crate::core::occ::textmas::{self as contract, TextMasCallStatus, TextMasMode, TextMasRole, TextMasUsageRecord};
use crate::core::occ::textmas_context as context_contract;
use futures::{stream, StreamExt, TryStreamExt};
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
    call_index: Option<u32>,
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
                let record = TextMasUsageRecord::new_indexed(&self.action_id, self.role, self.call_index, TextMasCallStatus::Cancelled, self.usage.clone());
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
                            snapshot.as_ref().map(|snapshot| snapshot.transformed.clone()),
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

    async fn run_textmas_preparation(
        self: &Arc<Self>,
        run: &TextMasRun,
        parent_signal: Option<CancellationToken>,
    ) -> Result<(UserMessage, PreparedSnapshot), String> {
        let durable = run.input_ids.iter().all(|id| {
            self.action_by_id(id)
                .as_ref()
                .and_then(|action| primary_delivery_record(action).ok())
                .is_some_and(|record| record.durable)
        });
        if !durable || self.has_failed_dispatch_persistence() {
            return Err("OCC3.1 requires a durably saved user request before preparation".into());
        }
        let options = self
            .agent
            .side_question_options()
            .ok_or("OCC3.1 provider interface is unavailable")?;
        let convert = options
            .convert_to_llm
            .ok_or("OCC3.1 provider context conversion is unavailable")?;
        let source = self.agent.messages();
        let transform = self.textmas.lock().unwrap().original_transform.clone();
        // Prepare the transformed snapshot used by tool-free roles and reused by the normal parent.
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
        // Only the tool-free role requests are subject to the preparation context bound.
        // The normal parent keeps this full history and appends advice in its own context hook.
        let mut active_original = original.clone();
        let mut preparation_note = None;
        if run.mode == TextMasMode::Hierarchical {
            let mut needs_reduction = false;
            for role in TextMasRole::ALL {
                match role_request(&run.model, &original, run.mode, role, &[]) {
                    Ok(_) => {}
                    Err(error) if error.reducible => needs_reduction = true,
                    Err(error) => return Err(error.message),
                }
            }
            if needs_reduction {
                let reduction = self
                    .reduce_textmas_context(run, &original, parent_signal.as_ref())
                    .await?;
                active_original = reduction.original;
                preparation_note = Some(reduction.note);
            }
        }
        let mut drafts = Vec::new();
        for role in TextMasRole::ALL {
            ensure_textmas_current(self, run, parent_signal.as_ref())?;
            let context = match role_request(&run.model, &active_original, run.mode, role, &drafts)
            {
                Ok(context) => context,
                Err(error) if error.reducible && preparation_note.is_none() => {
                    let reduction = self
                        .reduce_textmas_context(run, &original, parent_signal.as_ref())
                        .await?;
                    active_original = reduction.original;
                    preparation_note = Some(reduction.note);
                    role_request(&run.model, &active_original, run.mode, role, &drafts)
                        .map_err(|error| error.message)?
                }
                Err(error) => return Err(error.message),
            };
            let draft = self
                .run_textmas_role(run, role, None, context, parent_signal.as_ref())
                .await?;
            drafts.push((role, draft));
        }
        let advisory = contract::advisory_message_with_note(&drafts, preparation_note.as_deref());
        Ok((
            advisory,
            PreparedSnapshot {
                source,
                transformed,
            },
        ))
    }

    async fn reduce_textmas_context(
        self: &Arc<Self>,
        run: &TextMasRun,
        original: &Context,
        parent_signal: Option<&CancellationToken>,
    ) -> Result<TextMasReduction, String> {
        ensure_textmas_current(self, run, parent_signal)?;
        let (prepared, resized_images) = resize_textmas_images(
            self, original, run, parent_signal,
        ).await?;
        ensure_textmas_current(self, run, parent_signal)?;
        // Planning validates every static envelope and final role capacity before auxiliary spend.
        let plan = context_contract::TextMasPreparationPlan::new_with_control(
            &prepared,
            run.mode,
            &run.model,
            || ensure_textmas_current(self, run, parent_signal),
        )?;
        let map_count = plan.map_contexts.len();
        let map_calls = stream::iter(plan.map_contexts.iter().cloned().enumerate().map(
            |(index, context)| async move {
                let call_index = u32::try_from(index)
                    .map_err(|_| "OCC3.1 mapper call index overflow".to_string())?;
                self.run_textmas_role(
                    run,
                    TextMasRole::Mapper,
                    Some(call_index),
                    context,
                    parent_signal,
                )
                .await
            },
        ))
        .buffered(3)
        .try_collect::<Vec<_>>()
        .await?;
        let mut briefs = map_calls;
        let mut reduce_count = 0usize;
        while briefs.len() > 1 {
            ensure_textmas_current(self, run, parent_signal)?;
            let contexts = plan.reduction_contexts_with_control(
                &run.model,
                &briefs,
                || ensure_textmas_current(self, run, parent_signal),
            )?;
            let first_index = reduce_count;
            let context_count = contexts.len();
            briefs = stream::iter(contexts.into_iter().enumerate().map(
                |(offset, context)| async move {
                    let call_index = u32::try_from(first_index + offset)
                        .map_err(|_| "OCC3.1 reducer call index overflow".to_string())?;
                    self.run_textmas_role(
                        run,
                        TextMasRole::Reducer,
                        Some(call_index),
                        context,
                        parent_signal,
                    )
                    .await
                },
            ))
            .buffered(3)
            .try_collect::<Vec<_>>()
            .await?;
            reduce_count = first_index + context_count;
        }
        let brief = briefs
            .pop()
            .ok_or_else(|| "OCC3.1 reduction returned no brief".to_string())?;
        let original = plan.reduced_original(&brief);
        let latest = if plan.include_latest_user() {
            "included exactly"
        } else {
            "represented in the brief"
        };
        let system_source = if plan.system_instructions_in_source() {
            "complete system instructions were mapped into the lossy auxiliary brief; the parent retains them verbatim; "
        } else {
            ""
        };
        let note = format!(
            "Used {map_count} shared context-map and {reduce_count} context-reduction calls because an actual role request exceeded its bound; {system_source}resized {resized_images} preparation-only image(s); latest user message {latest}. Summaries are lossy advisory context, and the complete parent history remains authoritative."
        );
        Ok(TextMasReduction { original, note })
    }

    async fn run_textmas_role(
        self: &Arc<Self>,
        run: &TextMasRun,
        role: TextMasRole,
        call_index: Option<u32>,
        context: Context,
        parent_signal: Option<&CancellationToken>,
    ) -> Result<String, String> {
        self.run_textmas_role_with_timeout(
            run,
            role,
            call_index,
            context,
            parent_signal,
            std::time::Duration::from_secs(contract::STAGE_TIMEOUT_SECONDS),
        )
        .await
    }

    async fn run_textmas_role_with_timeout(
        self: &Arc<Self>,
        run: &TextMasRun,
        role: TextMasRole,
        call_index: Option<u32>,
        context: Context,
        parent_signal: Option<&CancellationToken>,
        stage_timeout: std::time::Duration,
    ) -> Result<String, String> {
        ensure_textmas_current(self, run, parent_signal)?;
        let output_budget = contract::output_budget(&run.model, &context)?;
        let started = TextMasUsageRecord::new_indexed(
            &run.action_id,
            role,
            call_index,
            TextMasCallStatus::Started,
            None,
        );
        self.persist_textmas_usage(&started)?;
        let receipt = StreamTaskReceipt::new_unsupported();
        if let Some(scope) = self.agent.execution_scope() {
            scope
                .register_stream(receipt.clone())
                .map_err(|_| "OCC3.1 execution scope was stopped")?;
        }
        let mut guard = CallGuard {
            session: Arc::downgrade(self),
            action_id: run.action_id.clone(),
            role,
            call_index,
            signal: run.signal.child_token(),
            receipt: receipt.clone(),
            usage: None,
            settled: false,
        };
        let options = self
            .agent
            .side_question_options()
            .ok_or("OCC3.1 provider interface is unavailable")?;
        let mut request = SimpleStreamOptions::default();
        request.stream.signal = Some(guard.signal.clone());
        request.stream.timeout_ms = Some(contract::STAGE_TIMEOUT_SECONDS as f64 * 1000.0);
        request.stream.max_tokens = Some(output_budget);
        request.stream.service_tier = self.agent.service_tier();
        request.stream.transport = Some(self.settings_manager.lock().unwrap().get_transport());
        let call_suffix = call_index
            .map(|index| format!(":{index}"))
            .unwrap_or_default();
        request.stream.session_id = Some(format!(
            "{}:occ31:{}:{}{call_suffix}",
            self.session_id(),
            run.action_id,
            role.as_str()
        ));
        request.reasoning = Some(self.agent.thinking_level().as_str().into());
        request.thinking_budgets = options.thinking_budgets;
        // Use the real selected provider closure: registry auth, headers, and custom APIs all remain intact.
        // No on_payload extension, latent lease, source capture, or tools are installed on advisory requests.
        let stream_fn =
            crate::core::semantic_edges::unwrap_semantic_edge_stream_fn(&self.agent.stream_fn());
        let request_receipt = receipt.clone();
        let model = run.model.clone();
        let result = run_textmas_provider_request(
            &run.signal,
            parent_signal,
            stage_timeout,
            &mut guard.usage,
            async move {
                if let Some(get_key) = options.get_api_key { request.stream.api_key = get_key(model.provider.clone()).await; }
                let stream = request_receipt.scope(async { stream_fn(model, context, request).await }).await;
                let result_stream = stream.clone();
                (Box::pin(stream.into_stream()), move || result_stream.result_if_ready())
            },
        )
        .await;
        let status = match &result {
            Ok(_) => TextMasCallStatus::Completed,
            Err((status, _)) => *status,
        };
        let terminal = TextMasUsageRecord::new_indexed(
            &run.action_id,
            role,
            call_index,
            status,
            guard.usage.clone(),
        );
        // No second write on a failed terminal persist: the reservation remains unknown and replay stays blocked.
        guard.settled = true;
        self.persist_textmas_usage(&terminal)?;
        result.map_err(|(_, error)| {
            format!(
                "OCC3.1 {}: {error}. No final execution or automatic preparation retry.",
                role.as_str()
            )
        })
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

type TextMasProviderResult = Result<String, (TextMasCallStatus, String)>;

async fn run_textmas_provider_request<S, R, F>(
    run_signal: &CancellationToken,
    parent_signal: Option<&CancellationToken>,
    stage_timeout: std::time::Duration,
    usage: &mut Option<Usage>,
    open_stream: F,
) -> TextMasProviderResult
where
    S: futures::Stream<Item = AssistantMessageEvent> + Unpin,
    R: Fn() -> Option<AssistantMessage>,
    F: std::future::Future<Output = (S, R)>,
{
    let deadline = tokio::time::Instant::now() + stage_timeout;
    if tokio::time::Instant::now() >= deadline {
        return Err((TextMasCallStatus::TimedOut, "preparation timed out".into()));
    }
    let (mut stream, result_if_ready) = tokio::select! {
        biased;
        _ = run_signal.cancelled() => return Err((TextMasCallStatus::Cancelled, "preparation was cancelled".to_string())),
        _ = cancelled(parent_signal) => return Err((TextMasCallStatus::Cancelled, "preparation was cancelled".to_string())),
        result = tokio::time::timeout_at(deadline, open_stream) => {
            result.map_err(|_| (TextMasCallStatus::TimedOut, "preparation timed out".to_string()))?
        },
    };
    loop {
        // Check before every select: a continuously ready stream must not starve either signal.
        if run_signal.is_cancelled() || parent_signal.is_some_and(CancellationToken::is_cancelled) {
            return Err((TextMasCallStatus::Cancelled, "preparation was cancelled".to_string()));
        }
        if tokio::time::Instant::now() >= deadline {
            return Err((TextMasCallStatus::TimedOut, "preparation timed out".into()));
        }
        let next = tokio::select! {
            biased;
            _ = run_signal.cancelled() => return Err((TextMasCallStatus::Cancelled, "preparation was cancelled".to_string())),
            _ = cancelled(parent_signal) => return Err((TextMasCallStatus::Cancelled, "preparation was cancelled".to_string())),
            result = tokio::time::timeout_at(deadline, stream.next()) => match result {
                Ok(next) => next,
                Err(_) => return Err((TextMasCallStatus::TimedOut, "preparation timed out".into())),
            },
        };
        let Some(event) = next else { break };
        if let Some(partial) = event.partial() {
            *usage = Some(partial.usage.clone());
            if serde_json::to_vec(partial).map_or(true, |bytes| bytes.len() > contract::MAX_STAGE_RESPONSE_BYTES) {
                return Err((TextMasCallStatus::Failed, "preparation response exceeded its bound".into()));
            }
            if partial.content.iter().any(|block| matches!(block, pi_ai::types::ContentBlock::ToolCall(_))) {
                return Err((TextMasCallStatus::Failed, "preparation returned a tool call; no tool was executed".into()));
            }
        }
        let terminal = match event { AssistantMessageEvent::Done { message, .. } => Some(message), AssistantMessageEvent::Error { error, .. } => Some(error), _ => None };
        if let Some(message) = terminal {
            *usage = Some(message.usage.clone());
            return contract::public_draft(&message).map_err(|error| (TextMasCallStatus::Failed, error));
        }
    }
    // Some injected/native adapters end directly with a terminal result.
    if let Some(message) = result_if_ready() {
        *usage = Some(message.usage.clone());
        contract::public_draft(&message).map_err(|error| (TextMasCallStatus::Failed, error))
    } else {
        Err((TextMasCallStatus::Failed, "provider ended without a result".into()))
    }
}

struct RoleRequestError {
    message: String,
    reducible: bool,
}
struct TextMasReduction {
    original: Context,
    note: String,
}

fn role_request(
    model: &Model,
    original: &Context,
    mode: TextMasMode,
    role: TextMasRole,
    drafts: &[(TextMasRole, String)],
) -> Result<Context, RoleRequestError> {
    let context = match contract::try_role_context(original, mode, role, drafts) {
        Ok(context) => context,
        Err(contract::TextMasRoleContextError::TooLarge) => {
            return Err(RoleRequestError {
                message: "OCC3.1 role context exceeds its serialized bound".into(),
                reducible: true,
            });
        }
        Err(error) => {
            return Err(RoleRequestError {
                message: error.to_string(),
                reducible: false,
            });
        }
    };
    if let Err(message) = contract::output_budget(model, &context) {
        return Err(RoleRequestError {
            message,
            reducible: true,
        });
    }
    Ok(context)
}

fn ensure_textmas_current(
    session: &Arc<AgentSession>,
    run: &TextMasRun,
    parent_signal: Option<&CancellationToken>,
) -> Result<(), String> {
    if run.signal.is_cancelled()
        || parent_signal.is_some_and(CancellationToken::is_cancelled)
        || session.agent.model() != run.model
    {
        Err("OCC3.1 preparation was cancelled or its model changed".into())
    } else {
        Ok(())
    }
}

async fn resize_textmas_images(
    session: &Arc<AgentSession>,
    original: &Context,
    run: &TextMasRun,
    parent_signal: Option<&CancellationToken>,
) -> Result<(Context, usize), String> {
    use crate::utils::image_resize::{
        ImageContent as ResizeImage, ImageResizeOptions, resize_image,
    };

    let mut sources = Vec::new();
    for (message_index, message) in original.messages.iter().enumerate() {
        ensure_textmas_current(session, run, parent_signal)?;
        match message {
            Message::User(user) => {
                if let UserContent::Blocks(blocks) = &user.content {
                    for (block_index, block) in blocks.iter().enumerate() {
                        ensure_textmas_current(session, run, parent_signal)?;
                        if let pi_ai::types::ImageOrTextContent::Image(image) = block {
                            sources.push((
                                message_index,
                                block_index,
                                format!(
                                    "message {} user image {}",
                                    message_index + 1,
                                    block_index + 1
                                ),
                                image.clone(),
                            ));
                        }
                    }
                }
            }
            Message::ToolResult(result) => {
                for (block_index, block) in result.content.iter().enumerate() {
                    ensure_textmas_current(session, run, parent_signal)?;
                    if let pi_ai::types::ImageOrTextContent::Image(image) = block {
                        sources.push((
                            message_index,
                            block_index,
                            format!(
                                "message {} tool result {} image {}",
                                message_index + 1,
                                result.tool_name,
                                block_index + 1
                            ),
                            image.clone(),
                        ));
                    }
                }
            }
            Message::Assistant(_) => {}
        }
    }
    let mut prepared = original.clone();
    let mut resized_count = 0usize;
    for (message_index, block_index, label, image) in sources {
        ensure_textmas_current(session, run, parent_signal)?;
        let limit = context_contract::preparation_image_limit_with_mode(
            original,
            &run.model,
            run.mode,
            &label,
            &image,
        )?;
        if image.data.len() <= limit {
            continue;
        }
        if limit < 2 {
            return Err(format!("OCC3.1 has no image byte budget for {label}"));
        }
        let source = ResizeImage {
            kind: image.type_.clone(),
            data: image.data.clone(),
            mime_type: image.mime_type.clone(),
        };
        let resized = resize_image(
            &source,
            Some(ImageResizeOptions {
                max_bytes: Some(limit as f64),
                ..Default::default()
            }),
        )
        .await
        .ok_or_else(|| format!("OCC3.1 could not decode or resize {label} for preparation"))?;
        ensure_textmas_current(session, run, parent_signal)?;
        if resized.data.len() > limit {
            return Err(format!(
                "OCC3.1 resized {label} still exceeds its preparation budget"
            ));
        }
        let replacement = pi_ai::types::ImageContent::new(resized.data, resized.mime_type);
        match &mut prepared.messages[message_index] {
            Message::User(user) => {
                let UserContent::Blocks(blocks) = &mut user.content else {
                    return Err("OCC3.1 image projection changed unexpectedly".into());
                };
                blocks[block_index] = pi_ai::types::ImageOrTextContent::Image(replacement);
            }
            Message::ToolResult(result) => {
                result.content[block_index] = pi_ai::types::ImageOrTextContent::Image(replacement);
            }
            Message::Assistant(_) => {
                return Err("OCC3.1 image projection changed unexpectedly".into());
            }
        }
        resized_count += 1;
    }
    Ok((prepared, resized_count))
}

async fn cancelled(signal: Option<&CancellationToken>) {
    match signal { Some(signal) => signal.cancelled().await, None => std::future::pending::<()>().await }
}

#[cfg(test)]
#[path = "textmas_deadline_tests.rs"]
mod deadline_tests;
