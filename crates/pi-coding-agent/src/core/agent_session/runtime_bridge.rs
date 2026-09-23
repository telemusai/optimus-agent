//! Session operations used by the native runtime adapters.

use super::super::*;
use serde_json::json;

impl AgentSession {
    /// Register the notification before checking liveness to cover completion races.
    pub(in crate::core::agent_session) async fn wait_for_background_bash_settlement(&self, arrival_epoch: Option<u64>) -> bool {
        loop {
            let notified = self.session_action_activity_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.explicitly_stopped() || self.disposed.load(Ordering::SeqCst)
                || self.disposing.load(Ordering::SeqCst)
                || self.agent.signal().is_some_and(|signal| signal.is_cancelled())
                || arrival_epoch.is_some_and(|epoch| self.session_input_arrival_epoch.load(Ordering::SeqCst) != epoch)
            {
                return false;
            }
            if !self.has_live_background_bash_handles() { return true; }
            tokio::select! {
                _ = notified => {},
                _ = self.session_action_commit_dispose_abort.cancelled() => return false,
                // Covers an abort without activity and custom kernel implementations.
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {},
            }
        }
    }

    /// `waitForHeadlessIdle()`.
    pub async fn wait_for_headless_idle(self: &Arc<Self>) -> Result<(), String> {
        loop {
            self.wait_for_idle().await?;
            let had_background_work = self.has_live_background_bash_handles();
            if !self.wait_for_background_bash_settlement(None).await {
                return if had_background_work { Err("Background work wait cancelled".into()) } else { Ok(()) };
            }
            if had_background_work { continue; }
            let settlement = self
                .post_compaction_continuation_settlement
                .lock()
                .unwrap()
                .clone();
            let Some(settlement) = settlement else {
                return Ok(());
            };
            let deferred = settlement.lock().unwrap().deferred.clone();
            deferred.wait().await?;
        }
    }

    fn quiescence_children(&self) -> Vec<Arc<Self>> {
        let abandoned = self
            .abandoned_rlm_quiescence_child_ids
            .lock()
            .unwrap()
            .clone();
        let mut children: Vec<_> = self
            .rlm_child_sessions
            .lock()
            .unwrap()
            .iter()
            .filter(|(id, _)| !abandoned.contains(*id))
            .map(|(_, child)| child.session.clone())
            .collect();
        for run in self.active_rlm_child_runs.lock().unwrap().values() {
            let run = run.lock().unwrap();
            if run.abandoned_for_quiescence == Some(true) {
                continue;
            }
            if let Some(child) = &run.session {
                if !children.iter().any(|existing| Arc::ptr_eq(existing, child)) {
                    children.push(child.clone());
                }
            }
        }
        children
    }

    fn runtime_quiescence_work(&self) -> bool {
        self.has_deferred_rlm_terminal_notices()
            || self
                .unsettled_rlm_child_runs
                .lock()
                .unwrap()
                .iter()
                .any(|run| {
                    let run = run.lock().unwrap();
                    !run.settled && run.abandoned_for_quiescence != Some(true)
                })
            || self
                .quiescence_children()
                .iter()
                .any(|child| child.is_session_active() || child.runtime_quiescence_work())
    }

    /// `waitForRlmQuiescence(externalSignal?)`.
    pub fn wait_for_rlm_quiescence(
        self: &Arc<Self>,
        external_signal: Option<CancellationToken>,
    ) -> BoxFuture<Result<(), String>> {
        let session = self.clone();
        Box::pin(async move {
            let cancellation = external_signal
                .map(|signal| signal.child_token())
                .unwrap_or_default();
            session
                .rlm_quiescence_wait_aborts
                .lock()
                .unwrap()
                .push(cancellation.clone());
            struct Cleanup {
                session: Arc<AgentSession>,
                cancellation: CancellationToken,
            }
            impl Drop for Cleanup {
                fn drop(&mut self) {
                    self.cancellation.cancel();
                    self.session
                        .rlm_quiescence_wait_aborts
                        .lock()
                        .unwrap()
                        .retain(|token| !token.is_cancelled());
                }
            }
            let _cleanup = Cleanup {
                session: session.clone(),
                cancellation: cancellation.clone(),
            };
            let operation = async {
                loop {
                    session.wait_for_headless_idle().await?;
                    if session.is_session_active() || session.has_deferred_rlm_terminal_notices() {
                        session
                            .wait_for_session_activity_change(Some(&cancellation))
                            .await?;
                        continue;
                    }
                    let unsettled: Vec<_> = session
                        .unsettled_rlm_child_runs
                        .lock()
                        .unwrap()
                        .iter()
                        .filter_map(|run| {
                            let run = run.lock().unwrap();
                            (!run.settled && run.abandoned_for_quiescence != Some(true))
                                .then(|| run.settlement.clone())
                        })
                        .collect();
                    let children = session.quiescence_children();
                    let child_work = children
                        .iter()
                        .any(|child| child.is_session_active() || child.runtime_quiescence_work());
                    if unsettled.is_empty() && !child_work {
                        return Ok(());
                    }
                    let mut waits: Vec<BoxFuture<Result<(), String>>> = unsettled
                        .into_iter()
                        .map(|deferred| {
                            Box::pin(async move { deferred.wait().await })
                                as BoxFuture<Result<(), String>>
                        })
                        .collect();
                    waits.extend(
                        children
                            .into_iter()
                            .map(|child| child.wait_for_rlm_quiescence(Some(cancellation.clone()))),
                    );
                    futures::future::try_join_all(waits).await?;
                }
            };
            tokio::select! {
                result = operation => result,
                _ = cancellation.cancelled() => Err("RLM quiescence wait cancelled".to_string()),
            }
        })
    }
}

impl Serialize for AgentSessionEvent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use AgentSessionEvent::*;
        let mut value = match self {
            Agent(event) => return event.serialize(serializer),
            IpythonSentAgentMessage {
                tool_call_id,
                message,
            } => json!({"toolCallId":tool_call_id,"message":message}),
            SessionActionUpdate { actions } => json!({"actions":actions}),
            CompactionStart {
                reason,
                custom_instructions,
            } => json!({"reason":reason,"customInstructions":custom_instructions}),
            SessionInfoChanged { name } => json!({"name":name}),
            MessageStart { message } | MessageEnd { message } => json!({"message":message}),
            ModelSelect {
                model,
                previous_model,
                reason,
            } => json!({"model":model,"previousModel":previous_model,"reason":reason}),
            ThinkingLevelChange { level } | ThinkingLevelChanged { level } => {
                json!({"level":level})
            }
            ServiceTierChange { service_tier } | ServiceTierChanged { service_tier } => {
                json!({"serviceTier":service_tier})
            }
            CompactionUpdate { active, reason } | RefinementUpdate { active, reason } => {
                json!({"active":active,"reason":reason})
            }
            RetryUpdate {
                active,
                attempt,
                max_attempts,
                message,
            } => {
                json!({"active":active,"attempt":attempt,"maxAttempts":max_attempts,"message":message})
            }
            TreeNavigated { target_id } => json!({"targetId":target_id}),
            RlmSubagentRemoved {
                child_id,
                session_name,
            } => json!({"childId":child_id,"sessionName":session_name}),
            CompactionEnd {
                reason,
                result,
                aborted,
                will_retry,
                error_message,
                error_severity,
                custom_instructions,
            } => {
                json!({"reason":reason,"result":result.as_ref().map(|result| json!({"summary":result.summary,"firstKeptEntryId":result.first_kept_entry_id,"tokensBefore":result.tokens_before,"details":result.details,"usage":result.usage})),"aborted":aborted,"willRetry":will_retry,"errorMessage":error_message,"errorSeverity":error_severity,"customInstructions":custom_instructions})
            }
            AutoRetryStart {
                attempt,
                max_attempts,
                delay_ms,
                error_message,
            } => {
                json!({"attempt":attempt,"maxAttempts":max_attempts,"delayMs":delay_ms,"errorMessage":error_message})
            }
            AutoRetryEnd {
                success,
                attempt,
                final_error,
            } => json!({"success":success,"attempt":attempt,"finalError":final_error}),
            AuthStale {
                provider,
                source_tokens,
            } => json!({"provider":provider,"sourceTokens":source_tokens}),
            RlmChildUpdate { child } => json!({"child":child}),
            RecapUpdate { recap } => json!({"recap":recap}),
            GoalUpdate { goal } => json!({"goal":goal}),
            BashStart {
                command,
                exclude_from_context,
                transient,
                run_id,
            } => {
                json!({"command":command,"excludeFromContext":exclude_from_context,"transient":transient,"runId":run_id})
            }
            BashOutput { chunk } => json!({"chunk":chunk}),
            BashEnd {
                exit_code,
                cancelled,
                truncated,
                full_output_path,
                error_message,
                transient,
                run_id,
            } => {
                json!({"exitCode":exit_code,"cancelled":cancelled,"truncated":truncated,"fullOutputPath":full_output_path,"errorMessage":error_message,"transient":transient,"runId":run_id})
            }
            RefineComplete { result } => json!({"result":result}),
            RefineFailed { error } => json!({"error":error}),
        };
        value.as_object_mut().expect("event object").insert(
            "type".to_string(),
            Value::String(self.type_name().to_string()),
        );
        value.serialize(serializer)
    }
}
