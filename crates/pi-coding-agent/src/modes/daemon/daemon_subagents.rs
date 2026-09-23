//! Daemon-owned RLM runtimes (daemon-mode.ts `createSubagentRuntimeHost`).

use super::*;
use crate::core::agent_session::AgentSession;
use crate::core::rlm_runtime::{
    CreateRlmSubagentRuntimeOptions, RlmSubagentRuntime, SubagentRuntimeHost,
};

pub(super) struct DaemonSubagentHost {
    daemon: std::sync::Weak<AgentDaemon>,
    parent: std::sync::Weak<StdMutex<ActiveSessionState>>,
}

impl DaemonSubagentHost {
    pub(super) fn new(
        daemon: &Arc<AgentDaemon>,
        parent: &Arc<StdMutex<ActiveSessionState>>,
    ) -> Self {
        Self {
            daemon: Arc::downgrade(daemon),
            parent: Arc::downgrade(parent),
        }
    }

    fn owners(&self) -> Result<(Arc<AgentDaemon>, Arc<StdMutex<ActiveSessionState>>), String> {
        Ok((
            self.daemon.upgrade().ok_or("Daemon disposed")?,
            self.parent.upgrade().ok_or("Parent session disposed")?,
        ))
    }
}

impl SubagentRuntimeHost for DaemonSubagentHost {
    fn supports_retained_stop(&self) -> bool { cfg!(windows) }
    fn create_rlm_subagent_runtime(
        &self,
        options: CreateRlmSubagentRuntimeOptions,
    ) -> pi_ai::types::BoxFuture<Result<RlmSubagentRuntime, String>> {
        let owners = self.owners();
        Box::pin(async move {
            let (daemon, parent) = owners?;
            let options = Arc::new(options);
            let parent_id = parent.lock().unwrap().active_session_id.clone();
            let client_env = parent.lock().unwrap().client_env.clone();
            let mut manager = SessionManager::create(
                &options
                    .parent_session
                    .session_manager
                    .lock()
                    .unwrap()
                    .get_cwd(),
                Some(&options.session_dir),
            )?;
            manager.new_session(Some(&crate::core::session_manager::NewSessionOptions {
                id: None,
                parent_session: options.parent_session.session_file(),
                rlm_depth: Some(options.rlm_depth as i64),
            }))?;
            let state_ref = Arc::new(StdMutex::new(None));
            let get_state: Arc<
                dyn Fn() -> Option<Arc<StdMutex<ActiveSessionState>>> + Send + Sync,
            > = {
                let state_ref = state_ref.clone();
                Arc::new(move || state_ref.lock().unwrap().clone())
            };
            let metadata = serde_json::json!({
                "kind": "subagent", "createdAt": now_millis(), "parentActiveSessionId": parent_id,
                "parentSessionId": options.parent_session.session_id(), "parentSessionFile": options.parent_session.session_file(),
                "rlmChildId": options.id, "rlmParentNodeId": options.rlm_parent_node_id,
                "prompt": options.prompt, "spawnCode": options.spawn_code, "sessionDir": options.session_dir,
            });
            let runtime =
                super::super::daemon_client_env::with_client_env(client_env.as_ref(), || {
                    (daemon.options.create_runtime)(CreateAgentSessionRuntimeInput {
                        factory: Value::Null,
                        cwd: manager.get_cwd(),
                        agent_dir: daemon.options.default_session_config.agent_dir.clone(),
                        session_manager: Arc::new(StdMutex::new(manager)),
                        session_config: None,
                        runtime_metadata: Some(metadata),
                        session_options: SessionRuntimeOptions {
                            subagent_options: Some(options.clone()),
                            model: Some(
                                serde_json::to_value(&options.model).expect("model serializes"),
                            ),
                            agent_message_controller: Some(
                                daemon.create_agent_message_controller(get_state.clone()),
                            ),
                            agent_observe_controller: Some(
                                daemon.create_agent_observe_controller(get_state.clone()),
                            ),
                            rlm_heartbeat_controller: Some(
                                daemon.create_rlm_heartbeat_controller(get_state),
                            ),
                        },
                    })
                })
                .await?;
            let child = runtime
                .session
                .agent_session()
                .ok_or("Subagent runtime has no native session")?;
            if let Err(error) = child.set_session_name(&options.session_name) {
                runtime.session.dispose().await;
                return Err(error);
            }
            if options
                .parent_session
                .get_rlm_child_run_status(&options.id)
                .as_deref()
                == Some("cancelled")
                && options.parent_session.retained_child_stop_generation(&options.id).is_none()
            {
                runtime.session.dispose().await;
                return Err("RLM subagent startup was cancelled".into());
            }
            let created_at = now_millis();
            let state = daemon
                .add_runtime(
                    runtime.clone(),
                    None,
                    Some(Arc::new(move |state| {
                        *state_ref.lock().unwrap() = Some(state.clone());
                        state.lock().unwrap().client_env = client_env.clone();
                    })),
                    None,
                )
                .await?;
            let admission = async {
                let file = child
                    .session_file()
                    .ok_or("Subagent has no durable session file")?;
                let written = daemon.record_rlm_subagent_state(
                    &parent,
                    RlmSubagentStateInput {
                        child_id: options.id.clone(),
                        session_name: options.session_name.clone(),
                        session_dir: options.session_dir.clone(),
                        session_file: file,
                        rlm_depth: options.rlm_depth as i64,
                        rlm_max_depth: options.rlm_max_depth as i64,
                        rlm_parent_node_id: Some(options.rlm_parent_node_id.clone()),
                        prompt: (options.prompt.len() <= 4096).then(|| options.prompt.clone()),
                        spawn_code: options.spawn_code.clone(),
                        model: Some(RlmSubagentModel {
                            provider: options.model.provider.clone(),
                            model_id: options.model.id.clone(),
                        }),
                        status: "running".into(),
                        created_at: Some(created_at),
                    },
                );
                let append = daemon
                    .pending_rlm_spawn_appends
                    .lock()
                    .unwrap()
                    .remove(&format!("{parent_id}#{}", options.id));
                if let Some(append) = append {
                    append.await.map_err(|error| error.to_string())??;
                }
                if !written {
                    return Err("Could not persist subagent display state".to_string());
                }
                if options
                    .parent_session
                    .get_rlm_child_run_status(&options.id)
                    .as_deref()
                    == Some("cancelled")
                    && options.parent_session.retained_child_stop_generation(&options.id).is_none()
                {
                    return Err("RLM subagent startup was cancelled".into());
                }
                Ok(())
            }
            .await;
            if let Err(error) = admission {
                let cleanup = daemon
                    .record_rlm_subagent_deletion(
                        &parent,
                        &options.id,
                        RlmLedgerDeleteReason::Revoked,
                    )
                    .await;
                let _ = daemon
                    .close_session(state, "killed", false, true, None, Some(false))
                    .await;
                if let Err(cleanup) = cleanup {
                    return Err(format!("{error}; spawn cleanup failed: {cleanup}"));
                }
                return Err(format!(
                    "Failed to record RLM subagent spawn for {}: {error}",
                    options.id
                ));
            }
            if let Some(publish) = &options.on_session_published {
                publish(&child);
            }
            daemon.schedule_roster_flush();
            Ok(RlmSubagentRuntime { session: child })
        })
    }

    fn complete_rlm_subagent_runtime(&self, child_id: &str, child: &Arc<AgentSession>) -> bool {
        let Ok((daemon, parent)) = self.owners() else {
            return false;
        };
        let Some(entry) = daemon.child_runtime_state(&parent, child_id) else {
            return false;
        };
        if !entry
            .session
            .agent_session()
            .is_some_and(|session| Arc::ptr_eq(&session, child))
        {
            return false;
        }
        let Some(file) = child.session_file() else {
            return false;
        };
        let metadata = &entry.runtime_metadata;
        let model = child.model();
        let session_dir = child.session_manager.lock().unwrap().get_session_dir();
        let created_at = std::fs::read_to_string(rlm_subagent_display_path(&session_dir))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|entry| entry.get("createdAt").and_then(Value::as_f64));
        let written = daemon.record_rlm_subagent_state(
            &parent,
            RlmSubagentStateInput {
                child_id: child_id.into(),
                session_name: child.session_name().unwrap_or_else(|| child_id.into()),
                session_dir,
                session_file: file,
                rlm_depth: child.rlm_depth(),
                rlm_max_depth: child.rlm_max_depth(),
                rlm_parent_node_id: metadata.rlm_parent_node_id.clone(),
                prompt: metadata
                    .prompt
                    .clone()
                    .filter(|prompt| prompt.len() <= 4096),
                spawn_code: metadata.spawn_code.clone(),
                model: model.map(|model| RlmSubagentModel {
                    provider: model.provider,
                    model_id: model.id,
                }),
                status: "completed".into(),
                created_at,
            },
        );
        daemon.schedule_roster_flush();
        written
    }

    fn release_rlm_subagent_runtime(
        &self,
        runtime: RlmSubagentRuntime,
        options: CreateRlmSubagentRuntimeOptions,
        status: &str,
    ) -> pi_ai::types::BoxFuture<Result<(), String>> {
        let owners = self.owners();
        let cancelled = status == "cancelled";
        Box::pin(async move {
            let (daemon, parent) = owners?;
            if !options.parent_session.try_claim_rlm_runtime_release(&options.id, &runtime.session) {
                // Retention won before cleanup admission; no release await or deletion may start.
                return Ok(());
            }
            let deletion = if cancelled {
                daemon
                    .record_rlm_subagent_deletion(
                        &parent,
                        &options.id,
                        RlmLedgerDeleteReason::Revoked,
                    )
                    .await
            } else {
                Ok(())
            };
            let state = daemon
                .child_runtime_state(&parent, &options.id)
                .filter(|entry| {
                    entry
                        .session
                        .agent_session()
                        .is_some_and(|child| Arc::ptr_eq(&child, &runtime.session))
                });
            let closed = if let Some(state) = state {
                daemon
                    .close_session(
                        state.state.clone(),
                        if cancelled { "killed" } else { "completed" },
                        true,
                        true,
                        None,
                        if cancelled { Some(false) } else { None },
                    )
                    .await
            } else {
                runtime.session.dispose_async(None).await;
                Ok(())
            };
            if cancelled && deletion.is_ok() {
                if let Some(file) = runtime.session.session_file() {
                    daemon
                        .delete_rlm_subagent_artifacts(&options.id, &file)
                        .await;
                }
            }
            deletion.and(closed)
        })
    }

    fn delete_rlm_subagent_runtime(
        &self,
        child_id: &str,
        session: Option<&Arc<AgentSession>>,
    ) -> pi_ai::types::BoxFuture<Result<(), String>> {
        let owners = self.owners();
        let child_id = child_id.to_string();
        let session = session.cloned();
        Box::pin(async move {
            let (daemon, parent) = owners?;
            let state = daemon.child_runtime_state(&parent, &child_id);
            let mut file = state
                .as_ref()
                .and_then(|entry| entry.session.session_file())
                .or_else(|| session.as_ref().and_then(|session| session.session_file()));
            if file.is_none() {
                if let Some(parent_file) = daemon.session_of(&parent).session_file() {
                    if let Some(edge) = daemon
                        .rlm_spawn_ledger()
                        .edges(true)
                        .await
                        .into_iter()
                        .find(|edge| {
                            edge.child_id == child_id
                                && canonical_session_path(&edge.parent)
                                    == canonical_session_path(&parent_file)
                        })
                    {
                        let entry = daemon
                            .passive_rlm_subagent_entry_for_edge(
                                &edge,
                                &daemon.session_of(&parent).session_id(),
                                &parent_file,
                                &mut HashMap::new(),
                                Arc::new(|_| {}),
                            )
                            .await;
                        file = Some(entry.session_file);
                    }
                }
            }
            daemon
                .record_rlm_subagent_deletion(&parent, &child_id, RlmLedgerDeleteReason::User)
                .await?;
            let stale = session
                .as_ref()
                .filter(|session| {
                    state.as_ref().is_some_and(|state| {
                        state
                            .session
                            .agent_session()
                            .is_some_and(|current| !Arc::ptr_eq(&current, session))
                    })
                })
                .cloned();
            let result = if let Some(state) = state {
                daemon
                    .close_session(
                        state.state.clone(),
                        "killed",
                        false,
                        true,
                        None,
                        Some(false),
                    )
                    .await
            } else {
                if let Some(session) = session {
                    session.dispose_async(None).await;
                }
                Ok(())
            };
            if let Some(stale) = stale {
                stale.dispose_async(None).await;
            }
            if let Some(file) = file {
                daemon.cancel_scheduled_jobs_for_session_file(&file);
                daemon.delete_rlm_subagent_artifacts(&child_id, &file).await;
            }
            result
        })
    }

    fn dispose_rlm_subagent_runtimes(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
        let owners = self.owners();
        Box::pin(async move {
            let (daemon, parent) = owners?;
            daemon
                .close_child_sessions(
                    parent,
                    "replaced",
                    true,
                    Arc::new(StdMutex::new(Vec::new())),
                    None,
                )
                .await
        })
    }
}

impl AgentDaemon {
    fn child_runtime_state(
        &self,
        parent: &Arc<StdMutex<ActiveSessionState>>,
        child_id: &str,
    ) -> Option<Arc<DaemonSessionState>> {
        let parent_id = parent.lock().unwrap().active_session_id.clone();
        self.session_states().into_iter().find(|entry| {
            entry.runtime_metadata.kind.as_deref() == Some("subagent")
                && entry.runtime_metadata.parent_active_session_id.as_deref() == Some(&parent_id)
                && entry.runtime_metadata.rlm_child_id.as_deref() == Some(child_id)
        })
    }
}
