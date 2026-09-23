//! Detached RLM runs and the session's agent-message host wrapper.
//! Reference: agent-session.ts `_startRlmChildRun` and child lifecycle methods.

use super::*;
use crate::core::agent_messages::{
    AgentFamilyRosterResult, AgentSessionMessageReceipt, AgentSessionMessageSendInput,
};
use crate::core::messages::RlmChildFailureDetails;

pub(super) struct SessionMessageController(pub std::sync::Weak<AgentSession>);

impl AgentSessionMessageController for SessionMessageController {
    fn roster(&self) -> BoxFuture<Result<AgentFamilyRosterResult, String>> {
        let weak = self.0.clone();
        Box::pin(async move {
            let session = weak.upgrade().ok_or("Session disposed")?;
            let result = session
                .handle_agent_message_host_request("agent_message.list_agents", None)?
                .await?;
            serde_json::from_value(result).map_err(|error| error.to_string())
        })
    }

    fn await_pending_child_publication(
        &self,
        selector: String,
    ) -> BoxFuture<Result<Option<String>, String>> {
        let weak = self.0.clone();
        Box::pin(async move {
            weak.upgrade()
                .ok_or("Session disposed")?
                .await_pending_rlm_child_publication(&selector)
                .await
        })
    }

    fn send_agent_message(
        &self,
        input: AgentSessionMessageSendInput,
    ) -> BoxFuture<Result<AgentSessionMessageReceipt, String>> {
        let weak = self.0.clone();
        Box::pin(async move {
            weak.upgrade()
                .ok_or("Session disposed")?
                .send_agent_message(input.target, input.message, input.receiver_role)
                .await
        })
    }
}

struct NameReservation {
    parent: std::sync::Weak<AgentSession>,
    name: String,
}

impl NameReservation {
    fn new(parent: &Arc<AgentSession>, name: &str) -> Result<Self, String> {
        if !parent
            .pending_rlm_subagent_session_names
            .lock()
            .unwrap()
            .insert(name.into())
        {
            return Err(format_agent_session_name_unavailable(
                name,
                (parent.rlm_depth + 1) as f64,
            ));
        }
        Ok(Self {
            parent: Arc::downgrade(parent),
            name: name.into(),
        })
    }
}

impl Drop for NameReservation {
    fn drop(&mut self) {
        if let Some(parent) = self.parent.upgrade() {
            parent
                .pending_rlm_subagent_session_names
                .lock()
                .unwrap()
                .remove(&self.name);
        }
    }
}

fn custom_notice(message: CustomAgentMessage) -> CustomMessage {
    match message {
        CustomAgentMessage::Custom {
            custom_type,
            content,
            display,
            details,
            timestamp,
        } => CustomMessage {
            role: "custom".into(),
            custom_type,
            content,
            display,
            details,
            timestamp,
        },
        _ => unreachable!("RLM notice constructors return custom messages"),
    }
}

impl AgentSession {
    /// `_startRlmChildRun`: admit immediately; an owned task completes startup,
    /// execution, durable parent notification, retention and cleanup.
    pub(super) async fn start_rlm_child_run(
        self: &Arc<Self>,
        prompt: &str,
        kwargs: &Map<String, Value>,
        spawn_code: Option<String>,
    ) -> Result<RlmSpawnHandle, String> {
        let spawned_by_request_id = self
            .is_streaming()
            .then(|| self.semantic_edges.lock().unwrap().last_turn_request_id())
            .flatten();
        let mut unsupported: Vec<_> = kwargs
            .keys()
            .filter(|key| !matches!(key.as_str(), "name" | "model" | "thinking"))
            .cloned()
            .collect();
        if !unsupported.is_empty() {
            unsupported.sort();
            return Err(format!(
                "Unsupported rlm.run kwargs: {}",
                unsupported.join(", ")
            ));
        }
        let requested_name =
            normalize_requested_rlm_subagent_session_name(kwargs.get("name"), None)?;
        let requested_model = normalize_requested_rlm_subagent_model(kwargs.get("model"), None)?;
        let requested_thinking =
            normalize_requested_rlm_subagent_thinking_level(kwargs.get("thinking"), None)?;
        if let Some(name) = &requested_name {
            assert_direct_agent_message_target(name)?;
        }
        if self.rlm_depth >= self.rlm_max_depth() {
            return Err(format!(
                "RLM recursion depth limit reached (RLM_DEPTH={}, RLM_MAX_DEPTH={})",
                self.rlm_depth,
                self.rlm_max_depth()
            ));
        }
        let mut reservation = requested_name
            .as_deref()
            .map(|name| NameReservation::new(self, name))
            .transpose()?;
        let selection = async {
            if let Some(name) = &requested_name {
                self.assert_rlm_subagent_session_name_available(name, true)
                    .await?;
            }
            self.resolve_rlm_subagent_model(requested_model.as_deref(), "subagent")
                .await
        }
        .await;
        let model = selection?.model;
        if let Some(thinking) = requested_thinking {
            let supported = get_supported_thinking_levels(&model);
            if !supported
                .iter()
                .any(|level| level == &thinking_level_name(&thinking))
            {
                return Err(format!("Requested thinking level \"{}\" is not supported by model \"{}/{}\"; supported levels: {}", thinking_level_name(&thinking), model.provider, model.id, supported.join(", ")));
            }
        }
        if self.disposed.load(Ordering::SeqCst) || self.disposing.load(Ordering::SeqCst) {
            return Err("Cannot spawn a subagent after its parent was disposed".into());
        }
        let session_dir = self.create_child_rlm_session_dir()?;
        let id = Path::new(&session_dir)
            .file_name()
            .ok_or("Invalid child session directory")?
            .to_string_lossy()
            .into_owned();
        let name =
            requested_name.unwrap_or_else(|| create_default_rlm_subagent_session_name(prompt, &id));
        if reservation.is_none() {
            reservation = Some(NameReservation::new(self, &name)?);
        }
        self.assert_rlm_subagent_session_name_available(&name, true)
            .await?;
        // The parent's effective mode is snapshotted at child creation, so a
        // later global change cannot silently alter an existing child chat.
        // Best-effort: failing to record inheritance must not fail the spawn,
        // and a skipped record resolves to the same global default (built-in
        // Off) - never a silent Compare.
        if let Some(agent_dir) = self.agent_dir.as_deref() {
            let bridge = crate::modes::interactive::native_host::JevModeBridge::new(
                std::path::Path::new(agent_dir),
            );
            let _ = bridge.inherit_into_child(&id, &self.session_id(), None);
        }
        let handle = RlmSpawnHandle {
            rlm_child_id: id.clone(),
            name: name.clone(),
            session_dir: session_dir.clone(),
            model: format!("{}/{}", model.provider, model.id),
        };
        let mut options =
            self.create_rlm_subagent_runtime_options(RlmSubagentRuntimeOptionsInput {
                id: id.clone(),
                prompt: prompt.into(),
                session_name: name.clone(),
                session_dir: session_dir.clone(),
                thinking_level: Some(requested_thinking.unwrap_or_else(|| {
                    clamp_thinking_level_for_model(&model, self.thinking_level())
                })),
                model: model.clone(),
                spawn_code,
                spawned_by_request_id,
            })?;
        let run = Arc::new(Mutex::new(RlmChildRun {
            id: id.clone(),
            prompt: prompt.into(),
            session_name: name,
            session_dir,
            model: Some(model),
            status: "queued".into(),
            duration_ms: None,
            answer_preview: None,
            tool_use_count: 0.0,
            activity: None,
            error: None,
            abort: Arc::new(noop_rlm_child_abort),
            publication: create_agent_message_deferred(),
            settlement: create_agent_message_deferred(),
            session: None,
            settled: false,
            suppress_terminal_notice: None,
            abandoned_for_quiescence: None,
            detached_deletion: None,
            deletion_cleanup: None,
            deletion_cleanup_observer: None,
            deletion_reservation: create_agent_message_deferred(),
            deletion_cleanup_failed: None,
            deletion_run_finished: None,
            deletion_notice: None,
            deletion_failure_notice: None,
            deletion_needs_completion_notice: None,
            complete_deletion: None,
            report_deletion_cleanup_failure: None,
            emit_update: None,
            last_emitted_update: None,
            unsubscribe: None,
        }));
        let weak_parent = Arc::downgrade(self);
        let weak_run = Arc::downgrade(&run);
        run.lock().unwrap().complete_deletion = Some(Arc::new(move || {
            let parent = weak_parent.upgrade();
            let run = weak_run.upgrade();
            Box::pin(async move {
                if let (Some(parent), Some(run)) = (parent, run) {
                    parent.publish_deletion_notice(&run, None).await;
                }
            })
        }));
        let weak_parent = Arc::downgrade(self);
        let weak_run = Arc::downgrade(&run);
        run.lock().unwrap().report_deletion_cleanup_failure = Some(Arc::new(move |error| {
            let parent = weak_parent.upgrade();
            let run = weak_run.upgrade();
            Box::pin(async move {
                if let (Some(parent), Some(run)) = (parent, run) {
                    parent.publish_deletion_notice(&run, Some(error)).await;
                }
            })
        }));
        let weak_parent = Arc::downgrade(self);
        let weak_run = Arc::downgrade(&run);
        run.lock().unwrap().emit_update = Some(Arc::new(move || {
            if let (Some(parent), Some(run)) = (weak_parent.upgrade(), weak_run.upgrade()) {
                parent.emit_child_run_update(&run);
            }
        }));
        let weak_run = Arc::downgrade(&run);
        let weak_parent = Arc::downgrade(self);
        options.on_session_published = Some(Arc::new(move |child| {
            if let Some(run) = weak_run.upgrade() {
                let weak_child = Arc::downgrade(child);
                let abort: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                    if let Some(child) = weak_child.upgrade() {
                        tokio::spawn(Box::pin(async move {
                            let _ = child.abort().await;
                        }));
                    }
                });
                let cancelled = {
                    let mut run = run.lock().unwrap();
                    run.session = Some(child.clone());
                    run.abort = abort.clone();
                    run.publication.resolve();
                    run.status == "cancelled"
                };
                let retained = weak_parent.upgrade().and_then(|parent| {
                    parent.retained_stop_ids.lock().unwrap().get(&run.lock().unwrap().id).cloned()
                });
                if let Some(generation) = retained {
                    let id = run.lock().unwrap().id.clone();
                    child.request_retained_stop(&id, &generation);
                } else if cancelled {
                    abort();
                }
            }
        }));
        self.active_rlm_child_runs
            .lock()
            .unwrap()
            .insert(id, run.clone());
        drop(reservation);
        self.unsettled_rlm_child_runs
            .lock()
            .unwrap()
            .push(run.clone());
        self.emit_child_run_update(&run);
        let parent = self.clone();
        let parent_assistant = self.find_last_assistant_message();
        tokio::spawn(Box::pin(async move {
            let started = std::time::Instant::now();
            let outcome = std::panic::AssertUnwindSafe(parent.execute_child_run(
                run.clone(),
                options.clone(),
                parent_assistant,
            ))
            .catch_unwind()
            .await;
            let outcome = outcome.unwrap_or_else(|_| Err("Subagent task panicked".into()));
            parent
                .settle_child_run(
                    run,
                    options,
                    outcome,
                    started.elapsed().as_secs_f64() * 1000.0,
                )
                .await;
        }));
        Ok(handle)
    }

    async fn publish_deletion_notice(
        self: &Arc<Self>,
        run: &Arc<Mutex<RlmChildRun>>,
        error: Option<String>,
    ) {
        if self.disposed.load(Ordering::SeqCst) || self.disposing.load(Ordering::SeqCst) {
            return;
        }
        let (notice, pending, existing) = {
            let mut run = run.lock().unwrap();
            if run.suppress_terminal_notice == Some(true)
                || (error.is_none() && run.deletion_needs_completion_notice != Some(true))
            {
                return;
            }
            let notice = match error.as_ref() {
                Some(error) => create_rlm_child_failure_message(RlmChildFailureDetails {
                    child_id: run.id.clone(), session_name: run.session_name.clone(),
                    error: format!("Deletion cleanup failed; retry rlm.delete_subagent(\"{}\") before completion: {error}", run.id),
                }, now_ms_i64()),
                None => create_rlm_child_terminal_notice_message(RlmChildTerminalNoticeDetails {
                    kind: "cancelled".into(), child_id: run.id.clone(), session_name: run.session_name.clone(),
                    reason: run.error.clone().or_else(|| Some("Deleted by parent orchestrator".into())), last_assistant_text_preview: None,
                }, now_ms_i64()),
            };
            let slot = if error.is_some() {
                &mut run.deletion_failure_notice
            } else {
                &mut run.deletion_notice
            };
            let existing = slot.is_some();
            let pending = slot
                .get_or_insert_with(|| Arc::new(create_agent_message_deferred()))
                .clone();
            (notice, pending, existing)
        };
        if existing {
            let _ = pending.wait().await;
            return;
        }
        match self.defer_rlm_terminal_notice(custom_notice(notice)).await {
            Ok(()) => pending.resolve(),
            Err(error) => {
                eprintln!("Could not publish subagent deletion: {error}");
                pending.reject(error);
            }
        }
    }

    fn emit_child_run_update(&self, run: &Arc<Mutex<RlmChildRun>>) {
        let current = run.lock().unwrap().clone();
        let child = self.rlm_child_snapshot_for_run(&current, None);
        let serialized = serde_json::to_string(&child).ok();
        {
            let mut run = run.lock().unwrap();
            if run.last_emitted_update == serialized {
                return;
            }
            run.last_emitted_update = serialized;
        }
        self.emit(AgentSessionEvent::RlmChildUpdate { child });
    }

    async fn execute_child_run(
        self: &Arc<Self>,
        run: Arc<Mutex<RlmChildRun>>,
        options: CreateRlmSubagentRuntimeOptions,
        parent_assistant: Option<AssistantMessage>,
    ) -> Result<(), String> {
        let publish = options.on_session_published.clone();
        let runtime = self.create_rlm_subagent_runtime(options).await?;
        let child = runtime.session;
        if let Some(publish) = publish {
            publish(&child);
        }
        {
            let mut run = run.lock().unwrap();
            if run.status == "cancelled" {
                return Err(run
                    .error
                    .clone()
                    .unwrap_or_else(|| "RLM child cancelled".into()));
            }
            run.status = "running".into();
        }
        self.emit_child_run_update(&run);
        let weak_parent = Arc::downgrade(self);
        let weak_run = Arc::downgrade(&run);
        let running_tools = Arc::new(AtomicU64::new(0));
        let weak_child = Arc::downgrade(&child);
        let unsubscribe = child.subscribe(Arc::new(move |event| {
            let (Some(parent), Some(run)) = (weak_parent.upgrade(), weak_run.upgrade()) else { return; };
            if matches!(event, AgentSessionEvent::RlmChildUpdate { .. }) { parent.emit(event); return; }
            let value = serde_json::to_value(&event).unwrap_or(Value::Null);
            let kind = event.type_name();
            let assistant = value.get("message").filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
                .and_then(|message| serde_json::from_value::<AssistantMessage>(message.clone()).ok());
            if kind == "message_end" {
                if let (Some(target), Some(assistant)) = (&parent_assistant, &assistant) {
                    if !matches!(assistant.stop_reason.as_str(), "error" | "aborted") {
                        let messages = weak_child.upgrade().map(|child| child.messages()).unwrap_or_default();
                        let index = messages.iter().position(|message| matches!(message, AgentMessage::Message(Message::Assistant(candidate)) if assistant_message_key(candidate) == assistant_message_key(assistant))).unwrap_or(messages.len());
                        parent.attribute_child_completion(target, assistant, rlm_child_usage_origin(&messages, index), false);
                    }
                }
            }
            {
                let mut run = run.lock().unwrap();
                match kind {
                    "agent_start" => run.activity = Some(RlmChildAgentActivity { kind: "waiting".into(), tool_name: None }),
                    "agent_end" => run.activity = None,
                    "message_start" | "message_update" | "message_end" if assistant.is_some() => {
                        let text = compact_rlm_text(&read_assistant_text(assistant.as_ref().unwrap()), 160);
                        if !text.is_empty() { run.answer_preview = Some(text); }
                        if kind != "message_end" { run.activity = Some(RlmChildAgentActivity { kind: "writing".into(), tool_name: None }); }
                    }
                    "tool_execution_start" => {
                        run.tool_use_count += 1.0;
                        running_tools.fetch_add(1, Ordering::SeqCst);
                        run.activity = Some(RlmChildAgentActivity { kind: "executing".into(), tool_name: value.get("toolName").and_then(Value::as_str).map(str::to_string) });
                    }
                    "tool_execution_end" => {
                        let previous = running_tools.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |count| Some(count.saturating_sub(1))).unwrap_or(0);
                        if previous <= 1 { run.activity = Some(RlmChildAgentActivity { kind: "waiting".into(), tool_name: None }); }
                    }
                    _ => {}
                }
            }
            parent.emit_child_run_update(&run);
        }));
        run.lock().unwrap().unsubscribe = Some(unsubscribe);
        let current = run.lock().unwrap().clone();
        let content = format!("[task from parent]\n\n{}", current.prompt);
        let before_reply = child.parent_reply_count.load(Ordering::SeqCst);
        child.prompt_and_wait(&content, Some(PromptOptions {
            expand_prompt_templates: Some(false), source: Some(crate::core::session_action_store::InputSource::Extension),
            custom_message: Some(CustomMessage {
                role: "custom".into(), custom_type: AGENT_MESSAGE_CUSTOM_TYPE.into(), content: CustomMessageContent::Text(content.clone()), display: true,
                details: Some(serde_json::json!({ "id": format!("spawn:{}", current.id), "message": current.prompt, "from": { "sessionId": self.session_id(), "sessionName": self.session_name(), "activeSessionId": self.current_active_session_id().await }, "fromRelationship": "parent" })),
                timestamp: now_ms_i64(),
            }), ..Default::default()
        })).await?;
        child.wait_for_rlm_quiescence(None).await?;
        {
            let mut run = run.lock().unwrap();
            if let Some(error) = &run.error {
                return Err(error.clone());
            }
            let continuation = child.rlm_continuation.lock().unwrap();
            if continuation.terminal_status.as_deref() == Some("failed") {
                run.status = "error".into();
                run.error = Some(
                    continuation
                        .pending_result
                        .as_ref()
                        .and_then(|result| result.reason.clone())
                        .unwrap_or_else(|| "Child reported failure".into()),
                );
            } else {
                run.status = "done".into();
            }
            // An explicit reply already delivered the result through agent_message.
            if child.parent_reply_count.load(Ordering::SeqCst) != before_reply {
                run.suppress_terminal_notice = Some(true);
            }
        }
        Ok(())
    }

    fn attribute_child_completion(
        self: &Arc<Self>,
        target: &AssistantMessage,
        child: &AssistantMessage,
        origin: &'static str,
        after_drain: bool,
    ) {
        // Usage changes after each child completion; identify the parent message
        // without its mutable usage so later completions still find the same row.
        let entry = self
            .session_manager
            .lock()
            .unwrap()
            .get_entries()
            .into_iter()
            .find(|entry| {
                entry.get("type").and_then(Value::as_str) == Some("message")
                    && entry
                        .get("message")
                        .and_then(|message| {
                            serde_json::from_value::<AssistantMessage>(message.clone()).ok()
                        })
                        .is_some_and(|mut message| {
                            message.usage = target.usage.clone();
                            assistant_message_key(&message) == assistant_message_key(target)
                        })
            });
        let Some(entry) = entry else {
            if !after_drain {
                let parent = self.clone();
                let target = target.clone();
                let child = child.clone();
                tokio::spawn(async move {
                    parent.await_agent_event_queue().await;
                    parent.attribute_child_completion(&target, &child, origin, true);
                });
            }
            return;
        };
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            return;
        };
        let mut manager = self.session_manager.lock().unwrap();
        let current = manager
            .get_entry(id)
            .and_then(|entry| entry.get("message").cloned())
            .and_then(|message| serde_json::from_value::<AssistantMessage>(message).ok());
        let mut usage = current
            .map(|message| message.usage)
            .unwrap_or_else(|| target.usage.clone());
        attribute_child_usage(&mut usage, &child.usage);
        if let Err(error) =
            manager.append_child_usage_attribution(id, &child.usage, &usage, Some(origin))
        {
            eprintln!("Could not persist subagent usage: {error}");
        }
        drop(manager);
        *self.own_usage_memo.lock().unwrap() = None;
    }

    async fn settle_child_run(
        self: &Arc<Self>,
        run: Arc<Mutex<RlmChildRun>>,
        options: CreateRlmSubagentRuntimeOptions,
        outcome: Result<(), String>,
        duration_ms: f64,
    ) {
        let completed = outcome.is_ok();
        {
            let mut run = run.lock().unwrap();
            if let Err(error) = outcome {
                run.publication.reject(error.clone());
                if run.status != "cancelled" {
                    run.status = "error".into();
                    run.error = Some(error);
                }
            }
            run.duration_ms = Some(duration_ms);
            run.activity = None;
        }
        let current = run.lock().unwrap().clone();
        if let Some(child) = &current.session {
            if current.status != "cancelled" {
                let request = child
                    .semantic_edges
                    .lock()
                    .unwrap()
                    .last_committed_request_id();
                if let Some(request) = request {
                    self.semantic_edges
                        .lock()
                        .unwrap()
                        .record_child_returned(&child.session_id(), Some(&request));
                }
            }
        }
        if current.session.is_none() {
            let mut child = self.rlm_child_snapshot_for_run(&current, None);
            child.status = "cancelled".into();
            self.emit(AgentSessionEvent::RlmChildUpdate { child });
        } else {
            self.emit_child_run_update(&run);
        }
        if current.detached_deletion.is_none() && current.suppress_terminal_notice != Some(true) {
            let notice = if current.status == "error" {
                create_rlm_child_failure_message(
                    RlmChildFailureDetails {
                        child_id: current.id.clone(),
                        session_name: current.session_name.clone(),
                        error: current
                            .error
                            .clone()
                            .unwrap_or_else(|| "unknown error".into()),
                    },
                    now_ms_i64(),
                )
            } else {
                let preview = current
                    .session
                    .as_ref()
                    .and_then(|child| child.find_last_assistant_message())
                    .map(|message| read_assistant_text(&message));
                create_rlm_child_terminal_notice_message(
                    RlmChildTerminalNoticeDetails {
                        kind: if current.status == "cancelled" {
                            "cancelled"
                        } else {
                            "completed_without_reply"
                        }
                        .into(),
                        child_id: current.id.clone(),
                        session_name: current.session_name.clone(),
                        reason: current.error.clone(),
                        last_assistant_text_preview: preview
                            .map(|text| text.chars().take(8000).collect()),
                    },
                    now_ms_i64(),
                )
            };
            if let Err(error) = self.defer_rlm_terminal_notice(custom_notice(notice)).await {
                eprintln!("Could not publish subagent completion: {error}");
            }
        }
        if current.detached_deletion.is_some() {
            run.lock().unwrap().deletion_run_finished = Some(true);
            if let Some(child) = current.session.clone() {
                let cleanup = self.ensure_rlm_run_deletion_cleanup(&current, &child).await;
                let snapshot = run.lock().unwrap().clone();
                if self
                    .observe_rlm_run_deletion_cleanup(
                        snapshot,
                        current.detached_deletion.clone().unwrap(),
                        child,
                        cleanup,
                    )
                    .await
                {
                    let snapshot = run.lock().unwrap().clone();
                    self.finish_rlm_run_deletion(&snapshot).await;
                }
            } else {
                self.finish_rlm_run_deletion(&current).await;
            }
            return;
        }
        if self.finish_retained_rlm_run(&run, &current) { return; }
        let retained = completed
            && current
                .session
                .as_ref()
                .is_some_and(|child| self.register_rlm_child_session(&current.id, child.clone()));
        if retained {
            self.active_rlm_child_runs
                .lock()
                .unwrap()
                .remove(&current.id);
            if let Some(unsubscribe) = current.unsubscribe.clone() {
                self.rlm_child_unsubscribes
                    .lock()
                    .unwrap()
                    .insert(current.id.clone(), unsubscribe);
            }
        } else {
            // This is the last synchronous boundary before any release await.
            // Stop and cleanup compete under one parent admission gate.
            if let Some(child) = &current.session {
                if !self.try_claim_rlm_runtime_release(&current.id, child) {
                    if self.finish_retained_rlm_run(&run, &current) { return; }
                    // Explicit deletion may have superseded retention under the same gate.
                    if !self.try_claim_rlm_runtime_release(&current.id, child) { return; }
                }
            }
            if let Some(unsubscribe) = &current.unsubscribe {
                unsubscribe();
            }
            if let Some(child) = &current.session {
                let host = self.subagent_runtime_host.lock().unwrap().clone();
                if let Some(host) = host {
                    if let Err(error) = host
                        .release_rlm_subagent_runtime(
                            RlmSubagentRuntime {
                                session: child.clone(),
                            },
                            options,
                            &current.status,
                        )
                        .await
                    {
                        eprintln!("Could not release subagent: {error}");
                    }
                } else {
                    child.dispose_async(None).await;
                }
            }
            if current.status != "error" {
                self.active_rlm_child_runs
                    .lock()
                    .unwrap()
                    .remove(&current.id);
            }
        }
        {
            let mut run = run.lock().unwrap();
            run.unsubscribe = None;
            run.abort = Arc::new(noop_rlm_child_abort);
            run.settled = true;
            run.settlement.resolve();
        }
        self.unsettled_rlm_child_runs
            .lock()
            .unwrap()
            .retain(|other| !Arc::ptr_eq(other, &run));
        self.maybe_resume_goal_continuation_after_rlm_work();
    }

    /// Retain the addressable child when stop wins before release admission.
    /// No await or destructive action is allowed between that decision and registration.
    pub(in crate::core::agent_session) fn finish_retained_rlm_run(self: &Arc<Self>, run: &Arc<Mutex<RlmChildRun>>, current: &RlmChildRun) -> bool {
        let _admission = self.rlm_child_lifecycle_admission.lock().unwrap();
        if self.rlm_child_release_claims.lock().unwrap().contains(&current.id) { return false; }
        let Some(generation) = self.retained_stop_ids.lock().unwrap().get(&current.id).cloned() else { return false; };
        if let Some(child) = &current.session {
            child.request_retained_stop(&current.id, &generation);
            if !self.register_rlm_child_session(&current.id, child.clone()) {
                // A failed host row update must never turn retain into destructive release.
                self.rlm_child_sessions.lock().unwrap().insert(current.id.clone(), RetainedRlmChild {
                    session: child.clone(), run: Some(run.clone()),
                });
            }
        }
        if current.session.is_some() {
            self.active_rlm_child_runs.lock().unwrap().remove(&current.id);
        }
        if let Some(unsubscribe) = current.unsubscribe.clone() {
            self.rlm_child_unsubscribes.lock().unwrap().insert(current.id.clone(), unsubscribe);
        }
        let mut current_run = run.lock().unwrap();
        current_run.settled = true; // Initial task only; retained stop has separate acknowledgements.
        current_run.settlement.resolve();
        drop(current_run);
        self.unsettled_rlm_child_runs.lock().unwrap().retain(|other| !Arc::ptr_eq(other, run));
        true
    }

    pub fn register_rlm_child_session(
        self: &Arc<Self>,
        child_id: &str,
        child: Arc<AgentSession>,
    ) -> bool {
        if self.disposed.load(Ordering::SeqCst)
            || self.disposing.load(Ordering::SeqCst)
            || self
                .deleting_rlm_children
                .lock()
                .unwrap()
                .contains_key(child_id)
            || self
                .deleted_rlm_child_ids
                .lock()
                .unwrap()
                .contains(child_id)
        {
            return false;
        }
        let host = self.subagent_runtime_host.lock().unwrap().clone();
        if host.is_some_and(|host| !host.complete_rlm_subagent_runtime(child_id, &child)) {
            return false;
        }
        let run = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(child_id)
            .cloned();
        self.rlm_child_sessions.lock().unwrap().insert(
            child_id.into(),
            RetainedRlmChild {
                session: child,
                run,
            },
        );
        true
    }

    pub fn get_rlm_child_snapshots(self: &Arc<Self>) -> Vec<RlmChildAgentSnapshot> {
        let mut snapshots = Vec::new();
        for owner in self.rlm_subtree_sessions() {
            let runs: Vec<_> = owner
                .active_rlm_child_runs
                .lock()
                .unwrap()
                .values()
                .map(|run| run.lock().unwrap().clone())
                .collect();
            let retained = owner.rlm_child_sessions.lock().unwrap().clone();
            let mut recorded = HashSet::new();
            for run in runs {
                if run.detached_deletion.is_some()
                    || owner
                        .deleting_rlm_children
                        .lock()
                        .unwrap()
                        .contains_key(&run.id)
                    || owner
                        .deleted_rlm_child_ids
                        .lock()
                        .unwrap()
                        .contains(&run.id)
                    || owner.is_unbound_terminal_rlm_child_run(&run)
                {
                    continue;
                }
                recorded.insert(run.id.clone());
                snapshots.push(owner.rlm_child_snapshot_for_run(&run, None));
            }
            for (id, retained) in retained {
                if recorded.contains(&id)
                    || owner
                        .deleting_rlm_children
                        .lock()
                        .unwrap()
                        .contains_key(&id)
                    || owner.deleted_rlm_child_ids.lock().unwrap().contains(&id)
                {
                    continue;
                }
                let child = retained.session;
                let mut snapshot = match retained.run {
                    Some(run) => {
                        owner.rlm_child_snapshot_for_run(&run.lock().unwrap().clone(), Some(child))
                    }
                    None => owner.rlm_child_snapshot_for_session(&id, &child),
                };
                if owner
                    .rlm_child_cleanup_failures
                    .lock()
                    .unwrap()
                    .contains_key(&id)
                {
                    snapshot.status = "cancelled".into();
                }
                snapshots.push(snapshot);
            }
        }
        snapshots
    }

    pub fn get_rlm_child_session(self: &Arc<Self>, child_id: &str) -> Option<Arc<Self>> {
        for owner in self.rlm_subtree_sessions() {
            let active = owner
                .active_rlm_child_runs
                .lock()
                .unwrap()
                .get(child_id)
                .and_then(|run| run.lock().unwrap().session.clone());
            if active.is_some() {
                return active;
            }
            if let Some(child) = owner.rlm_child_sessions.lock().unwrap().get(child_id) {
                return Some(child.session.clone());
            }
        }
        None
    }

    pub fn cancel_rlm_child_run_by_id(self: &Arc<Self>, child_id: &str, reason: &str) -> bool {
        for owner in self.rlm_subtree_sessions() {
            let run = owner
                .active_rlm_child_runs
                .lock()
                .unwrap()
                .get(child_id)
                .cloned();
            if let Some(run) = run {
                let current = run.lock().unwrap().clone();
                if !matches!(current.status.as_str(), "running" | "queued") && !current.settled {
                    run.lock().unwrap().suppress_terminal_notice = Some(true);
                    return true;
                }
                if owner.cancel_rlm_child_run(&current, reason) {
                    owner.emit_child_run_update(&run);
                    return true;
                }
            }
            if let Some(child) = owner.get_rlm_child_session(child_id) {
                if child.is_session_active() {
                    tokio::spawn(Box::pin(async move {
                        let _ = child.abort().await;
                    }));
                    return true;
                }
            }
        }
        false
    }
}
