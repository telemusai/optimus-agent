use super::*;
use crate::core::execution_mode::ExecutionMode;

const ENTRY: &str = "execution_mode";

impl AgentSession {
    /// F6 yields after the current tool batch instead of waiting for the model's
    /// entire investigation to end. Keep a continuation in the ordinary queue,
    /// so stop/cancel, persistence and newer user input retain their ownership.
    pub(super) fn preserve_work_after_mode_switch(self: &Arc<Self>, context: &ShouldStopAfterTurnContext) {
        if context.tool_results.is_empty() || self.explicitly_stopped() { return; }
        let queued = self.action_store.lock().unwrap().queued_actions(None);
        let switching = queued.iter().any(|action| matches!(&action.payload,
            QueuedActionPayload::SessionCommand(input) if input.base.command.name == "mode" && !input.base.command.args.is_empty()));
        if !switching || queued.iter().any(|action| matches!(action.payload, QueuedActionPayload::Turn(_))) { return; }
        let text = "Continue the user's unfinished task after the queued execution mode change. Use the current system prompt and available tools; the previous tool batch has completed.";
        let message = CustomMessage {
            role: "custom".into(), timestamp: now_ms_i64(),
            custom_type: "executionModeContinuation".into(),
            content: CustomMessageContent::Text(text.into()),
            display: false,
            details: None,
        };
        let action = self.create_prepared_turn_action("followUp", text, None, Some(PreparedTurnActionOptions {
            custom_message: Some(message), source: Some("internal".into()), queue_visible: Some(false),
            queue_key: Some("execution-mode-continuation".into()), ..Default::default()
        }));
        let _ = self.admit_session_input(action, false);
    }

    /// A stopped chat can change settings without admitting or waking queued work.
    pub(super) async fn try_stopped_execution_mode(
        self: &Arc<Self>,
        text: &str,
        options: &PromptOptions,
    ) -> Result<bool, String> {
        if !self.explicitly_stopped()
            || options.internal_prompt == Some(true)
            || options.custom_message.is_some()
            || !matches!(options.source, None
                | Some(crate::core::session_action_store::InputSource::Interactive)
                | Some(crate::core::session_action_store::InputSource::Rpc))
        {
            return Ok(false);
        }
        let Some(command) = parse_session_slash_command(text).filter(|c| c.name == "mode") else {
            return Ok(false);
        };
        let _commit = self.acquire_session_action_commit_fence().await?;
        let result = {
            // Serialize with stop/resume too; never clear the stop or suspend flags.
            let _stop = self.explicit_stop_admission.lock().unwrap();
            if !self.explicitly_stopped() {
                return Ok(false);
            }
            if self.disposed.load(Ordering::SeqCst) || self.disposing.load(Ordering::SeqCst) {
                return Err("Cannot change execution mode while the session is closing.".into());
            }
            if !self.session_input_admission_pauses.lock().unwrap().is_empty() {
                return Err("Cannot change execution mode while session input admission is paused.".into());
            }
            if options.signal.as_ref().is_some_and(|signal| signal.is_cancelled()) {
                return Err("Mode change was cancelled.".into());
            }
            let activity = self.runtime_activity();
            if activity.lower_agent_run || activity.compaction || activity.retry || activity.bash
                || activity.refinement_apply || activity.branch_mutation
            {
                return Err("The session is still stopping. Try changing modes once it is idle.".into());
            }
            self.change_execution_mode(&command.args)?
        };
        self.append_durable_session_command_message(
            &result, &command, true, false, true,
        );
        if let Some(committed) = &options.admission_committed {
            committed();
        }
        once_preflight(options.preflight_result.clone())(true, false);
        self.settle_agent_message(options.agent_message_id.as_deref(), "delivery", None);
        self.settle_agent_message(options.agent_message_id.as_deref(), "completion", None);
        Ok(true)
    }

    pub(super) fn saved_execution_mode(&self) -> Option<ExecutionMode> {
        self.session_manager
            .lock()
            .unwrap()
            .get_branch(None)
            .iter()
            .rev()
            .find(|entry| {
                entry.get("type").and_then(Value::as_str) == Some("custom")
                    && entry.get("customType").and_then(Value::as_str) == Some(ENTRY)
            })
            .and_then(|entry| {
                entry
                    .get("data")?
                    .get("mode")?
                    .as_str()
                    .and_then(|mode| ExecutionMode::parse(mode).ok())
            })
    }

    /// Runs only inside the session commit fence, between agent runs.
    pub(super) fn change_execution_mode(&self, args: &str) -> Result<String, String> {
        let current = ExecutionMode::from_tools(&self.get_active_tool_names());
        let mode = match args.trim() {
            "" => {
                return Ok(format!(
                    "Execution mode: {}",
                    current.map(ExecutionMode::label).unwrap_or("Custom tools")
                ))
            }
            "toggle" => current
                .ok_or("Cannot toggle a custom tool set; use /mode ipython, /mode node or /mode direct")?
                .toggled(),
            value => ExecutionMode::parse(value)?,
        };
        if self.base_tools_override.is_some() {
            return Err("Execution mode switching is unavailable with an SDK tool override".into());
        }
        let names = mode.tools(&self.get_active_tool_names());
        let tools: Vec<_> = {
            let registry = self.tool_registry.lock().unwrap();
            names
                .iter()
                .map(|name| {
                    registry.get(name).cloned().ok_or_else(|| {
                        format!(
                            "Cannot enable {}: tool {name} is unavailable or restricted",
                            mode.label()
                        )
                    })
                })
                .collect::<Result<_, _>>()?
        };
        self.session_manager
            .lock()
            .unwrap()
            .append_custom_entry_with_rollback(
                ENTRY,
                Some(serde_json::json!({"mode": mode.as_str()})),
            )?;
        let prompt = self.rebuild_system_prompt(&names);
        *self.base_system_prompt.lock().unwrap() = prompt.clone();
        *self.owned_skill_hint_projection.lock().unwrap() = None;
        self.agent.update_state(Box::new(move |state| {
            state.system_prompt = prompt;
            state.tools = Some(tools);
        }));
        Ok(format!("Execution mode: {}", mode.label()))
    }
}
