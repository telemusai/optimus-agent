//! Explicit cancellation is a persistent admission boundary, not a model prompt.
use super::*;

impl AgentSession {
    pub(super) fn explicitly_stopped(&self) -> bool {
        let state = self.explicit_stop.lock().unwrap();
        state.generation.is_some() || state.retain_generation.is_some()
    }

    pub(super) fn restore_explicit_stop(&self) {
        let mut state = ExplicitStopState::default();
        let mut retained_receipt = None;
        let mut retained_child = None;
        let mut audit_epoch = None;
        self.session_manager
            .lock()
            .unwrap()
            .visit_branch(None, |entry| {
                let data = entry.get("data").unwrap_or(&Value::Null);
                match entry.get("customType").and_then(Value::as_str) {
                    Some(EXPLICIT_STOP_ENTRY) => {
                        state.generation = data
                            .get("generation")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        if state.generation.is_some() {
                            state.last_generation = state.generation.clone();
                            state.stopped_at = data
                                .get("stoppedAt")
                                .and_then(Value::as_f64)
                                .unwrap_or(f64::INFINITY);
                            state.report_cutoff_at = state.stopped_at;
                        }
                        if state.generation.is_none() {
                            state.report_cutoff_at = data
                                .get("reportCutoffAt")
                                .and_then(Value::as_f64)
                                .unwrap_or(state.report_cutoff_at);
                            state.unannounced_reports = 0;
                        }
                    }
                    Some("prime-agent.retained-stop.v1") => {
                        state.retain_generation = data.get("generation").and_then(Value::as_str).map(str::to_string);
                        retained_receipt = data.get("receipt").cloned();
                        retained_child = data.get("childId").and_then(Value::as_str).map(str::to_string);
                        if let Some(epoch) = data.get("auditEpoch").and_then(Value::as_str) {
                            audit_epoch = Some(epoch.to_string());
                            state.generation = None;
                        }
                    }
                    Some(DEFERRED_REPORT_ENTRY) => {
                        if let Some(id) = data.get("messageId").and_then(Value::as_str) {
                            if state.report_ids.insert(id.to_string()) {
                                state.unannounced_reports += 1;
                            }
                        }
                    }
                    _ => {}
                }
            });
        if state.generation.is_some() || state.retain_generation.is_some() {
            self.session_input_pump_suspended
                .store(true, Ordering::SeqCst);
        }
        if let Some(generation) = state.retain_generation.as_deref() {
            if let Some(scope) = self.agent.execution_scope() { scope.request_cancel(); }
            let jev = crate::core::jev_bridge::request_session_retain_stop(&self.session_id());
            let mut receipt = retained_receipt.unwrap_or_else(|| serde_json::json!({
                "schema":"optimus.stop-retain.v1", "accepted":true, "settled":false,
                "retained":true, "automatic_continuation_fenced":true,
                "status":"failed_settlement", "error_code":"restored_without_settlement_proof",
                "acknowledged":{"model":false,"tools":false,"kernel":false,
                    "owned_processes":false,"transcript_flushed":false},
            }));
            receipt["rlm_child_id"] = serde_json::json!(retained_child);
            receipt["session_id"] = serde_json::json!(self.session_id());
            receipt["stop_generation"] = serde_json::json!(generation);
            receipt["history"] = serde_json::json!({"session_file":self.session_file()});
            receipt["jev_generation"] = serde_json::json!(jev.generation);
            if !jev.settled {
                receipt["settled"] = Value::Bool(false);
                receipt["status"] = Value::String("failed_settlement".into());
                receipt["error_code"] = Value::String("restored_jev_work_unsettled".into());
                receipt["acknowledged"]["model"] = Value::Bool(false);
                receipt["acknowledged"]["tools"] = Value::Bool(false);
            }
            *self.retained_stop.lock().unwrap() = Some(receipt);
        }
        *self.retained_kernel_epoch.lock().unwrap() = audit_epoch;
        *self.explicit_stop.lock().unwrap() = state;
    }

    pub(super) fn begin_explicit_stop(&self) {
        let mut state = self.explicit_stop.lock().unwrap();
        if state.generation.is_some() {
            return;
        }
        let generation = uuid::Uuid::new_v4().to_string();
        // Fail closed in memory even when persistence fails; abort() surfaces that error.
        state.generation = Some(generation.clone());
        state.last_generation = Some(generation.clone());
        state.stopped_at = now_ms();
        state.report_cutoff_at = state.stopped_at;
        self.session_input_pump_suspended
            .store(true, Ordering::SeqCst);
        self.session_input_pump_epoch.fetch_add(1, Ordering::SeqCst);
        let mut manager = self.session_manager.lock().unwrap();
        let result = manager.append_custom_entry(EXPLICIT_STOP_ENTRY,
            Some(serde_json::json!({"version": 1, "generation": generation, "stoppedAt": state.stopped_at})))
            .and_then(|_| manager.flush_now());
        state.persistence_error = result.err();
    }

    pub(super) fn stopped_report(action: &QueuedSessionAction) -> Option<CustomMessage> {
        let record = primary_delivery_record(action).ok()?;
        let DeliveryMessage::Custom(message) = record.message else {
            return None;
        };
        let canonical = agent_message_from_delivery(&DeliveryMessage::Custom(message.clone()));
        (is_agent_session_message(&canonical)
            || message.custom_type == RLM_CHILD_TERMINAL_NOTICE_CUSTOM_TYPE
            || message.custom_type == RLM_CHILD_FAILURE_CUSTOM_TYPE)
            .then_some(message)
    }

    pub(super) fn defer_stopped_report(
        &self,
        message: &CustomMessage,
        message_id: Option<&str>,
    ) -> Result<bool, String> {
        let mut state = self.explicit_stop.lock().unwrap();
        // A fresh parent command is explicit authorization for a stopped child
        // to take new work. Admission checks this again after resume; do not
        // misclassify the command itself as pause-period child-report backlog.
        if state.generation.is_none()
            && self.rlm_depth > 0
            && message
                .details
                .as_ref()
                .and_then(|details| details.get("fromRelationship"))
                .and_then(Value::as_str)
                == Some("parent")
            && (message.timestamp as f64) > state.stopped_at
        {
            return Ok(false);
        }
        // An explicit new task may resume the parent before an old transport
        // backlog arrives. Original report timestamps keep those stale messages
        // behind the previous stop boundary instead of waking the new task.
        let generation = state.generation.clone().or_else(|| {
            ((message.timestamp as f64) <= state.report_cutoff_at)
                .then(|| state.last_generation.clone())
                .flatten()
        });
        let Some(generation) = generation else {
            return Ok(false);
        };
        let id = message_id
            .map(str::to_string)
            .or_else(|| {
                message
                    .details
                    .as_ref()?
                    .get("id")?
                    .as_str()
                    .map(str::to_string)
            })
            .unwrap_or_else(|| {
                agent_message_key_of(&agent_message_from_delivery(&DeliveryMessage::Custom(
                    message.clone(),
                )))
            });
        if state.report_ids.contains(&id) {
            return Ok(true);
        }
        let canonical = agent_message_from_delivery(&DeliveryMessage::Custom(message.clone()));
        let mut manager = self.session_manager.lock().unwrap();
        manager.append_custom_entry(
            DEFERRED_REPORT_ENTRY,
            Some(serde_json::json!({
                "version": 1, "stopGeneration": generation, "messageId": id,
                "receivedAt": now_ms(), "message": canonical,
            })),
        )?;
        manager.flush_now()?;
        state.report_ids.insert(id);
        state.unannounced_reports += 1;
        Ok(true)
    }

    pub(super) fn defer_queued_reports_for_stop(&self) {
        let actions = self.action_store.lock().unwrap().clearable_actions(None);
        let mut saved = HashSet::new();
        for action in actions {
            let Some(report) = Self::stopped_report(&action) else {
                continue;
            };
            match self.defer_stopped_report(&report, action.agent_message_id.as_deref()) {
                Ok(true) => {
                    saved.insert(action.id);
                }
                Ok(false) => {}
                Err(error) => {
                    self.explicit_stop.lock().unwrap().persistence_error = Some(error);
                }
            }
        }
        self.cancel_session_actions(&|action| saved.contains(&action.id),
            "Session stopped; child report preserved in the deferred-agent-report journal, not delivered to the model.", None);
        // Automatic terminal notices must not turn into new prompts after resume.
        let pending = self.pending_next_turn_messages.lock().unwrap().clone();
        for message in pending {
            if !self.is_rlm_terminal_notice(&message) {
                continue;
            }
            if self.defer_stopped_report(&message, None).unwrap_or(false) {
                self.pending_next_turn_messages
                    .lock()
                    .unwrap()
                    .retain(|candidate| candidate != &message);
            }
        }
    }

    pub(super) fn resume_explicit_stop(&self) -> Result<(), String> {
        let _admission = self.explicit_stop_admission.lock().unwrap();
        let mut state = self.explicit_stop.lock().unwrap();
        if state.retain_generation.is_some() {
            return Err("Retained stop requires explicit parent audit authorization".into());
        }
        if state.generation.is_none() {
            return Ok(());
        }
        let count = state.unannounced_reports;
        let report_cutoff_at = now_ms().max(state.stopped_at);
        let mut manager = self.session_manager.lock().unwrap();
        manager.append_custom_entry(
            EXPLICIT_STOP_ENTRY,
            Some(serde_json::json!({"version": 1, "generation": null, "reportCutoffAt": report_cutoff_at})),
        )?;
        manager.flush_now()?;
        state.generation = None;
        state.report_cutoff_at = report_cutoff_at;
        state.persistence_error = None;
        state.unannounced_reports = 0;
        if count > 0 {
            self.pending_next_turn_messages.lock().unwrap().push(CustomMessage {
                role: "custom".into(),
                custom_type: "deferred_agent_reports_notice".into(),
                content: CustomMessageContent::Text(format!(
                    "{count} child reports were saved while this session was explicitly stopped. They did not restart work. Full reports, original timestamps and message IDs remain in transcript custom entries of type {DEFERRED_REPORT_ENTRY}. These belong to the stopped task; do not resume old instructions. Follow the new user request and retrieve those entries only if needed."
                )),
                display: true, details: None, timestamp: now_ms() as i64,
            });
        }
        Ok(())
    }

    pub(super) fn resume_for_new_parent_task(
        &self,
        message: &CustomMessage,
    ) -> Result<bool, String> {
        if self.explicit_stop.lock().unwrap().retain_generation.is_some() { return Ok(false); }
        let is_new_parent_task = self.rlm_depth > 0
            && message
                .details
                .as_ref()
                .and_then(|details| details.get("fromRelationship"))
                .and_then(Value::as_str)
                == Some("parent")
            && (message.timestamp as f64) > self.explicit_stop.lock().unwrap().stopped_at;
        if is_new_parent_task {
            self.resume_explicit_stop()?;
            self.resume_session_input_admission();
        }
        Ok(is_new_parent_task)
    }

    /// The existing user-facing resume_queue command, not automatic checkpoint resumption.
    pub fn resume_stopped_queue(&self) -> bool {
        let was_stopped = self.explicitly_stopped();
        if self.resume_explicit_stop().is_err() {
            return false;
        }
        self.resume_queued_work() || was_stopped
    }
}
