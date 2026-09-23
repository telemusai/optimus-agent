//! Direct-child retained stop: cancellation, execution settlement and history are distinct.
use super::*;
const RETAIN_STOP_ENTRY: &str = "prime-agent.retained-stop.v1";
const CLEANUP_MS: u64 = 10_000; // Existing explicit-Stop negative-settlement bound.

fn pending_receipt(id: &str, generation: &str, child: Option<&AgentSession>) -> Value {
    serde_json::json!({
        "schema":"optimus.stop-retain.v1", "rlm_child_id":id,
        "session_id":child.map(AgentSession::session_id), "stop_generation":generation,
        "accepted":true,"settled":false,"retained":false,
        "automatic_continuation_fenced":false,"status":"pending",
        "acknowledged":{"model":false,"tools":false,"kernel":false,
            "owned_processes":false,"transcript_flushed":false},
        "error_code":null,"history":{"session_file":child.and_then(AgentSession::session_file)}
    })
}
impl AgentSession {
    /// Claim normal cleanup before its first await. Once claimed, stop cannot be accepted.
    /// An existing claim is idempotent for the core -> daemon handoff.
    pub(crate) fn try_claim_rlm_runtime_release(&self, id: &str, child: &Arc<AgentSession>) -> bool {
        let _admission = self.rlm_child_lifecycle_admission.lock().unwrap();
        if self.rlm_child_release_claims.lock().unwrap().contains(id) { return true; }
        if self.retained_stop_ids.lock().unwrap().contains_key(id) { return false; }
        if let Some(generation) = child.retained_stop_generation() {
            self.retained_stop_ids.lock().unwrap().insert(id.to_string(), generation);
            return false;
        }
        self.rlm_child_release_claims.lock().unwrap().insert(id.to_string());
        true
    }

    fn child_cleanup_admitted(&self, id: &str) -> bool {
        self.rlm_child_release_claims.lock().unwrap().contains(id)
            || self.deleting_rlm_children.lock().unwrap().contains_key(id)
            || self.deleted_rlm_child_ids.lock().unwrap().contains(id)
    }

    pub(crate) fn retained_child_stop_generation(&self, id: &str) -> Option<String> {
        self.retained_stop_ids.lock().unwrap().get(id).cloned()
    }

    pub(crate) fn retained_stop_generation(&self) -> Option<String> {
        self.explicit_stop.lock().unwrap().retain_generation.clone()
    }

    // Never use get_rlm_child_session: it traverses descendants and is not authority.
    fn direct_retention_target(&self, selector: &str)
        -> Result<(String, Option<Arc<Mutex<RlmChildRun>>>, Option<Arc<AgentSession>>), String> {
        let runs: Vec<_> = self.active_rlm_child_runs.lock().unwrap().values().cloned().collect();
        for run in runs {
            let current = run.lock().unwrap().clone();
            if current.id == selector || current.session_name == selector {
                if current.detached_deletion.is_some() || self.child_cleanup_admitted(&current.id) {
                    return Err("Child cleanup already admitted".into());
                }
                return Ok((current.id, Some(run), current.session));
            }
        }
        for (id, retained) in self.rlm_child_sessions.lock().unwrap().iter() {
            let name_matches = retained.run.as_ref().is_some_and(|run| run.lock().unwrap().session_name == selector);
            if id == selector || name_matches {
                if self.child_cleanup_admitted(id) { return Err("Child cleanup already admitted".into()); }
                return Ok((id.clone(), retained.run.clone(), Some(retained.session.clone())));
            }
        }
        Err("Target is not an owned direct child".into())
    }

    pub(super) fn active_child_execution(&self, selector: &str) -> Result<Value, String> {
        let (id, _, child) = self.direct_retention_target(selector)?;
        let fenced = child.as_ref().is_some_and(|child| child.explicitly_stopped());
        let generation = child.as_ref().and_then(|child| child.agent.active_execution_generation());
        Ok(serde_json::json!({"schema":"optimus.active-execution.v1", "rlm_child_id":id,
            "session_id":child.as_ref().map(|child| child.session_id()),
            "active":generation.is_some() && !fenced,"fenced":fenced,
            "execution_generation":if fenced { None } else { generation }}))
    }

    pub(super) fn send_active_child_message(&self, payload: Value) -> Result<Value, String> {
        let selector = payload.get("target").and_then(Value::as_str).ok_or("target required")?;
        let generation = payload.get("execution_generation").and_then(Value::as_str).filter(|value| !value.is_empty() && value.len() <= 128).ok_or("execution_generation required")?;
        let message_id = payload.get("message_id").and_then(Value::as_str).filter(|value| !value.is_empty() && value.len() <= 256).ok_or("message_id required")?;
        let message = payload.get("message").and_then(Value::as_str).filter(|value| !value.is_empty() && value.len() <= 32000).ok_or("bounded message required")?;
        let (id, _, child) = self.direct_retention_target(selector)?;
        let delivery = if let Some(child) = &child {
            let _admission = child.explicit_stop_admission.lock().unwrap();
            if child.explicitly_stopped() { "declined_stopped" } else {
                let diagnostic = AgentMessage::Custom(CustomAgentMessage::Custom {
                    custom_type: "active_parent_diagnostic".into(),
                    content: CustomMessageContent::Text(message.to_string()), display: true,
                    details: Some(serde_json::json!({"fromSessionId":self.session_id(),
                        "fromRelationship":"parent","messageId":message_id,
                        "executionGeneration":generation,"wakeIfIdle":false})),
                    timestamp: now_ms_i64(),
                });
                child.agent.steer_active(generation, message_id, diagnostic)
            }
        } else { "declined_idle" };
        Ok(serde_json::json!({"schema":"optimus.active-message.v1", "rlm_child_id":id,
            "session_id":child.as_ref().map(|child| child.session_id()), "wakeIfIdle":false,
            "accepted":matches!(delivery,"accepted"|"duplicate"),"deliveryStatus":delivery,
            "executionGeneration":generation,"messageId":message_id}))
    }

    pub(super) fn retention_profile(&self) -> bool {
        self.agent.execution_scope().is_some()
            && self.model().is_some_and(|model| model.api == "openai-completions")
            && self.get_active_tool_names() == ["ipython"]
            && self.base_tools_override.is_none()
            && self.resource_loader.get_extensions().extensions.is_empty()
    }

    pub(super) async fn lifecycle_capabilities(self: &Arc<Self>, payload: Value) -> Result<Value, String> {
        let (model, supported) = if let Some(selector) = payload.get("target").and_then(Value::as_str) {
            let (_, run, child) = self.direct_retention_target(selector)?;
            let model = child.as_ref().and_then(|child| child.model())
                .or_else(|| run.as_ref().and_then(|run| run.lock().unwrap().model.clone()));
            let supported = child.as_ref().map(|child| child.retention_profile())
                .unwrap_or_else(|| model.as_ref().is_some_and(|model| model.api == "openai-completions")
                    && self.resource_loader.get_extensions().extensions.is_empty()
                    && self.subagent_runtime_host.lock().unwrap().as_ref()
                        .is_some_and(|host| host.supports_retained_stop()));
            (model, supported)
        } else if let Some(selector) = payload.get("model").and_then(Value::as_str) {
            // Metadata only. Do not refresh auth or probe a provider for capability discovery.
            let model = self.model_registry.lock().unwrap().get_all().into_iter()
                .find(|model| format!("{}/{}", model.provider, model.id) == selector);
            let supported = model.as_ref().is_some_and(|model| model.api == "openai-completions")
                && self.resource_loader.get_extensions().extensions.is_empty()
                && self.subagent_runtime_host.lock().unwrap().as_ref()
                    .is_some_and(|host| host.supports_retained_stop());
            (model, supported)
        } else { (None, false) };
        let profile = model.map(|model| serde_json::json!({
            "model":format!("{}/{}", model.provider, model.id),"api":model.api,
            "tools":["ipython"],"processContainment":"windows-job",
            "scope":"local-invocation-and-contained-descendants"
        }));
        Ok(crate::core::rlm_runtime::native_lifecycle_capabilities(profile, supported))
    }

    pub(super) async fn stop_retained_child(self: &Arc<Self>, selector: &str, timeout_ms: u64) -> Result<Value, String> {
        if timeout_ms > CLEANUP_MS { return Err("timeout_ms exceeds 10000".into()); }
        let admission = self.rlm_child_lifecycle_admission.lock().unwrap();
        let (id, run, child) = self.direct_retention_target(selector)?;
        // Record intent BEFORE abort; the ordinary cancelled-release path deletes artifacts.
        let generation = self.retained_stop_ids.lock().unwrap().entry(id.clone())
            .or_insert_with(|| child.as_ref().and_then(|child| child.retained_stop_generation())
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())).clone();
        if let Some(run) = &run {
            let mut run = run.lock().unwrap();
            run.suppress_terminal_notice = Some(true);
            run.status = "cancelled".into();
            run.error = Some("Retained stop requested".into());
        }
        let Some(child) = child else {
            // No published identity/history yet. Publication callback installs the same fence.
            return Ok(pending_receipt(&id, &generation, None));
        };
        child.request_retained_stop(&id, &generation);
        drop(admission);
        if timeout_ms > 0 {
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
            loop {
                let wake = child.retained_stop_notify.notified();
                tokio::pin!(wake); wake.as_mut().enable();
                if child.retained_stop.lock().unwrap().as_ref().is_some_and(|r| r["status"] != "pending") { break; }
                if tokio::time::timeout_at(deadline, wake).await.is_err() { break; }
            }
        }
        let receipt = child.retained_stop.lock().unwrap().clone().unwrap();
        Ok(receipt)
    }

    pub(super) fn request_retained_stop(self: &Arc<Self>, child_id: &str, generation: &str) {
        {
            let _admission = self.explicit_stop_admission.lock().unwrap();
            let mut receipt = self.retained_stop.lock().unwrap();
            if receipt.is_some() { return; }
            // Cancellation precedes filesystem work. Admission remains fenced below.
            self.agent.abort();
            if let Some(scope) = self.agent.execution_scope() { scope.request_cancel(); }
            let jev = crate::core::jev_bridge::request_session_retain_stop(&self.session_id());
            self.begin_explicit_stop();
            self.explicit_stop.lock().unwrap().retain_generation = Some(generation.to_string());
            let mut next = pending_receipt(child_id, generation, Some(self));
            next["jev_generation"] = serde_json::json!(jev.generation);
            let durable = {
                let mut manager = self.session_manager.lock().unwrap();
                manager.append_custom_entry(RETAIN_STOP_ENTRY, Some(serde_json::json!({"generation":generation,"childId":child_id,"jevGeneration":jev.generation})))
                    .and_then(|_| manager.flush_now())
            };
            next["automatic_continuation_fenced"] = Value::Bool(durable.is_ok());
            next["retained"] = Value::Bool(durable.is_ok() && self.session_file().is_some());
            next["acknowledged"]["transcript_flushed"] = Value::Bool(durable.is_ok());
            if durable.is_err() { next["error_code"] = Value::String("stop_checkpoint_failed".into()); }
            *receipt = Some(next);
        }
        self.defer_queued_reports_for_stop();
        // No implicit deletion of descendants. Unsupported non-leaf work fails settlement.
        if let Some(scope) = self.agent.execution_scope() { scope.request_cancel(); }
        self.request_abort_inner(false);
        self.agent.clear_all_queues();
        self.cancel_session_actions(&|_| true, "Retained stop cancelled queued work", None);
        let child = self.clone();
        tokio::spawn(async move { child.settle_retained_execution().await; });
    }

    async fn settle_retained_execution(self: &Arc<Self>) {
        let scope = self.agent.execution_scope();
        let provisioner = self.ipython_kernel_provisioner.lock().unwrap().clone();
        let unsupported_background = !self.retention_profile()
            || !self.active_rlm_child_runs.lock().unwrap().is_empty()
            || !self.rlm_child_sessions.lock().unwrap().is_empty()
            || self.compaction_operation.lock().unwrap().is_some()
            || self.branch_summary_operation.lock().unwrap().is_some()
            || self.refine_in_flight.lock().unwrap().is_some()
            || self.refine_plan_in_flight.lock().unwrap().is_some()
            || self.serialized_plan_in_flight.lock().unwrap().is_some();
        let cleanup = async {
            let model_tools = async {
                match &scope {
                    Some(scope) => Some(scope.settle(std::time::Duration::from_millis(CLEANUP_MS)).await),
                    None => None,
                }
            };
            let kernel = async {
                match provisioner {
                    Some(provisioner) => provisioner.shutdown_and_settle(&self.session_id(), CLEANUP_MS).await.ok(),
                    None => None,
                }
            };
            let session_id = self.session_id();
            let jev = crate::core::jev_bridge::settle_session_retain_stop(
                &session_id, std::time::Duration::from_millis(CLEANUP_MS));
            let (work, kernel, jev) = tokio::join!(model_tools, kernel, jev);
            self.agent.wait_for_idle().await;
            self.await_agent_event_queue().await;
            loop {
                let wake = self.session_action_activity_notify.notified();
                tokio::pin!(wake); wake.as_mut().enable();
                if self.pending_session_action_fence_waiters.load(Ordering::SeqCst) == 0
                    && self.session_action_commit_owner.lock().unwrap().is_none() { break; }
                wake.await;
            }
            (work, kernel, jev)
        };
        let result = tokio::time::timeout(std::time::Duration::from_millis(CLEANUP_MS), cleanup).await;
        let mut receipt = self.retained_stop.lock().unwrap();
        let receipt = receipt.as_mut().expect("retained stop admitted");
        if let Ok((work, kernel, jev)) = result {
            receipt["jev_settled"] = Value::Bool(jev.settled);
            if let Some(work) = work {
                receipt["acknowledged"]["model"] = Value::Bool(work.supported && work.model_settled && !work.failed && jev.settled && !unsupported_background);
                receipt["acknowledged"]["tools"] = Value::Bool(work.supported && work.tools_settled && !work.failed && jev.settled && !unsupported_background);
            }
            if let Some(kernel) = kernel {
                receipt["acknowledged"]["kernel"] = Value::Bool(kernel.supported && kernel.kernel_exited && kernel.settled);
                receipt["acknowledged"]["owned_processes"] = Value::Bool(kernel.supported && kernel.descendants_exited && kernel.settled);
            }
        }
        let flushed = self.session_manager.lock().unwrap().flush_now().is_ok();
        receipt["acknowledged"]["transcript_flushed"] = Value::Bool(flushed);
        let settled = receipt["automatic_continuation_fenced"] == true && receipt["retained"] == true
            && receipt["acknowledged"].as_object().unwrap().values().all(|value| value == &Value::Bool(true));
        receipt["settled"] = Value::Bool(settled);
        receipt["status"] = Value::String(if settled { "settled" } else { "failed_settlement" }.into());
        if !settled && receipt["error_code"].is_null() {
            receipt["error_code"] = Value::String("execution_ownership_or_settlement_unproved".into());
        }
        let persisted = {
            let mut manager = self.session_manager.lock().unwrap();
            manager.append_custom_entry(RETAIN_STOP_ENTRY, Some(serde_json::json!({
                "generation":receipt["stop_generation"], "childId":receipt["rlm_child_id"],
                "receipt":receipt.clone(),
            }))).and_then(|_| manager.flush_now())
        };
        if persisted.is_err() {
            receipt["settled"] = Value::Bool(false);
            receipt["acknowledged"]["transcript_flushed"] = Value::Bool(false);
            receipt["status"] = Value::String("failed_settlement".into());
            receipt["error_code"] = Value::String("stop_checkpoint_failed".into());
        }
        self.retained_stop_notify.notify_waiters();
    }

    pub(super) fn resume_retained_audit(
        self: &Arc<Self>, selector: &str, generation: &str, prompt: &str,
    ) -> Result<Value, String> {
        if prompt.trim().is_empty() || prompt.len() > 32_000 {
            return Err("A bounded explicit audit prompt is required".into());
        }
        let _lifecycle_admission = self.rlm_child_lifecycle_admission.lock().unwrap();
        let (id, _, child) = self.direct_retention_target(selector)?;
        let child = child.ok_or("Child not published")?;
        let admission = child.explicit_stop_admission.lock().unwrap();
        let receipt = child.retained_stop.lock().unwrap().clone().ok_or("No retained stop")?;
        if receipt["stop_generation"] != generation || receipt["settled"] != true {
            return Err("Audit resume requires exact settled stop generation".into());
        }
        if child.disposed.load(Ordering::SeqCst) || child.disposing.load(Ordering::SeqCst) {
            return Err("Child session is disposed".into());
        }
        if !child.session_input_admission_pauses.lock().unwrap().is_empty() {
            return Err("Child session admission is paused".into());
        }
        let jev_generation = receipt["jev_generation"].as_u64().ok_or("Missing Jev stop generation")?;
        if !crate::core::jev_bridge::session_retain_status(&child.session_id()).settled {
            return Err("Jev session work is not settled".into());
        }
        let scope = child.agent.execution_scope().ok_or("Unowned agent")?;
        scope.reset_after_settlement().map_err(str::to_string)?;
        if !crate::core::jev_bridge::begin_session_retain_generation(&child.session_id(), jev_generation) {
            scope.request_cancel();
            return Err("Jev stop generation changed".into());
        }
        let audit_epoch = uuid::Uuid::new_v4().to_string();
        // A new snapshot path prevents the stopped Python heap or clocks from returning.
        *child.retained_kernel_epoch.lock().unwrap() = Some(audit_epoch.clone());
        child.ipython_kernel_provisioner.lock().unwrap().take();
        child.build_runtime(Some(vec!["ipython".into()]), false);
        child.agent.clear_all_queues();
        child.pending_next_turn_messages.lock().unwrap().clear();
        child.cancel_session_actions(&|_| true, "Old retained task cannot resume", None);
        let action = child.create_prepared_turn_action(
            SESSION_INPUT_SCHEDULE_FOLLOW_UP,
            &format!("[Explicit retained-session audit. This is a new task; do not resume old queued work or clocks.]\n{prompt}"),
            None,
            Some(PreparedTurnActionOptions {
                suppress_autonomous_continuation: Some(true),
                resume_if_idle: Some(false), source: Some("internal".into()),
                queue_visible: Some(true), ..Default::default()
            }),
        );
        let action_id = action.id.clone();
        let stop_epoch = child.session_input_pump_epoch.load(Ordering::SeqCst);
        // Restore-style admission cannot wake the pump and keeps the retained fence.
        let admitted = child.admit_session_input_under_stop_fence(
            action, false, true, false, false, admission);
        if let Err(error) = admitted {
            scope.request_cancel();
            crate::core::jev_bridge::request_session_retain_stop(&child.session_id());
            return Err(error);
        }
        let admission = child.explicit_stop_admission.lock().unwrap();
        if child.session_input_pump_epoch.load(Ordering::SeqCst) != stop_epoch {
            child.cancel_session_actions(&|action| action.id == action_id, "Audit admission cancelled", None);
            scope.request_cancel();
            crate::core::jev_bridge::request_session_retain_stop(&child.session_id());
            return Err("Stop changed during audit admission".into());
        }
        let persisted = {
            let mut manager = child.session_manager.lock().unwrap();
            manager.append_custom_entry(RETAIN_STOP_ENTRY, Some(serde_json::json!({
                "generation":null,"auditOf":generation,"auditEpoch":audit_epoch,
            }))).and_then(|_| manager.flush_now())
        };
        if let Err(error) = persisted {
            child.cancel_session_actions(&|action| action.id == action_id, "Audit checkpoint failed", None);
            scope.request_cancel();
            crate::core::jev_bridge::request_session_retain_stop(&child.session_id());
            // Append a compensating fence; never convert a failed save into permission.
            let mut manager = child.session_manager.lock().unwrap();
            let _ = manager.append_custom_entry(RETAIN_STOP_ENTRY, Some(serde_json::json!({
                "generation":generation,"childId":id,"receipt":receipt,
            }))).and_then(|_| manager.flush_now());
            return Err(error);
        }
        {
            let mut state = child.explicit_stop.lock().unwrap();
            state.retain_generation = None;
            state.generation = None;
        }
        *child.retained_stop.lock().unwrap() = None;
        self.retained_stop_ids.lock().unwrap().remove(&id);
        drop(admission);
        child.resume_session_input_admission();
        child.schedule_session_input_pump();
        Ok(serde_json::json!({"schema":"optimus.audit-resume.v1","rlm_child_id":id,
            "session_id":child.session_id(),"accepted":true,"stop_generation":generation,
            "audit_generation":audit_epoch,"action_id":action_id,"old_work_replayed":false}))
    }

}
