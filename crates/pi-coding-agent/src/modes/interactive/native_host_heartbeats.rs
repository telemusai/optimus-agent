//! Heartbeat catalog and native manager callbacks.
use super::*;
use crate::modes::interactive::components::heartbeat_manager::{
    HeartbeatManagerComponent, HeartbeatManagerOptions,
};

pub(super) fn scoped(
    mode: &InteractiveMode,
    catalog: &[wire::AgentConnectionHeartbeat],
) -> Vec<wire::AgentConnectionHeartbeat> {
    let ids: std::collections::HashSet<String> = mode
        .get_scoped_heartbeats()
        .into_iter()
        .map(|h| h.job.id)
        .collect();
    catalog
        .iter()
        .filter(|h| {
            h.job
                .get("id")
                .and_then(|id| id.as_str())
                .is_some_and(|id| ids.contains(id))
        })
        .cloned()
        .collect()
}

pub(super) fn apply_catalog(
    mode: &mut InteractiveMode,
    catalog: &[wire::AgentConnectionHeartbeat],
) {
    mode.heartbeat_catalog = catalog
        .iter()
        .filter_map(|h| {
            Some(local::AgentConnectionHeartbeat {
                job: project_heartbeat(h.job.clone())?,
                session_name: h.session_name.clone(),
                first_message: h.first_message.clone(),
            })
        })
        .collect();
}

pub(super) fn create(
    mode: Rc<RefCell<InteractiveMode>>,
    catalog: Rc<RefCell<Vec<wire::AgentConnectionHeartbeat>>>,
    rows: Rc<Cell<f64>>,
    send: HostEventSender,
    connection: Arc<dyn wire::AgentConnection>,
) -> HeartbeatManagerComponent {
    let close = send.clone();
    let render = send.clone();
    HeartbeatManagerComponent::new(HeartbeatManagerOptions {
        get_heartbeats: Box::new(move || scoped(&mode.borrow(), &catalog.borrow())),
        get_rows: Rc::new(move || rows.get()),
        on_close: Box::new(move || {
            let _ = close.send(HostEvent::CloseHeartbeats);
        }),
        request_render: Box::new(move || {
            let _ = render.send(HostEvent::Render);
        }),
        on_action: Box::new(move |heartbeat, action| {
            let connection = connection.clone();
            let send = send.clone();
            Box::pin(async move {
                let job: crate::core::cron_jobs::AgentCronJob =
                    serde_json::from_value(heartbeat.job.clone()).map_err(|e| e.to_string())?;
                let updated = connection
                    .manage_heartbeat(
                        &job.active_session_id,
                        &job.id,
                        serde_json::Value::String(action),
                    )
                    .await?;
                let _ = send.send(HostEvent::HeartbeatUpdated(heartbeat, updated));
                if let Ok(catalog) = connection.list_heartbeats().await {
                    let _ = send.send(HostEvent::Heartbeats(catalog, false));
                }
                Ok(())
            })
        }),
    })
}

pub(super) fn refresh_delay(catalog: &[wire::AgentConnectionHeartbeat]) -> Option<Duration> {
    let next = catalog
        .iter()
        .filter(|h| h.job["status"] == "active")
        .filter_map(|h| {
            // TypeScript filters with `Number.isFinite(Date.parse(...))`, so a
            // value Node rejects must arm no timer rather than being coerced.
            crate::modes::interactive::components::heartbeat_manager::parse_iso_millis(
                h.job["nextRunAt"].as_str()?,
            )
        })
        .min()?;
    let until = next.saturating_sub(chrono::Utc::now().timestamp_millis());
    Some(Duration::from_millis(if until > 0 {
        until.saturating_add(250).min(60_000) as u64
    } else {
        5_000
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::interactive::interactive_mode::native_host::HostEvent;
    use crate::modes::interactive::interactive_mode::InteractiveMode;
    use crate::modes::interactive::interactive_mode_services::{
        AgentConnectionState, InteractiveModeUiServices, ModelRegistry, SettingsManager,
    };
    use crate::modes::interactive::theme::theme::init_theme;
    use pi_ai::types::{ServiceTier, Transport};
    use serde_json::{json, Value};
    use std::cell::Cell;
    use std::sync::{Arc, Mutex};

    fn heartbeat(
        id: &str,
        status: &str,
        active_session_id: &str,
        session_id: &str,
    ) -> wire::AgentConnectionHeartbeat {
        wire::AgentConnectionHeartbeat {
            job: json!({
                "id": id,
                "status": status,
                "source": "heartbeat",
                "activeSessionId": active_session_id,
                "sessionId": session_id,
                "sessionFile": format!("/tmp/{id}.jsonl"),
                "cwd": "/tmp",
                "prompt": "check the session",
                "schedule": {"kind": "interval", "expression": "every 5m", "intervalMs": 300000},
                "createdAt": "2026-01-01T00:00:00.000Z",
                "updatedAt": "2026-01-01T00:00:00.000Z",
                "runCount": 0.0
            }),
            session_name: None,
            first_message: None,
        }
    }

    fn mode(session_id: &str, active_session_id: Option<&str>) -> InteractiveMode {
        init_theme(Some("prime"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        let services = InteractiveModeUiServices {
            settings_manager: Arc::new(Mutex::new(SettingsManager::in_memory(
                serde_json::Map::new(),
            ))),
            model_registry: Arc::new(Mutex::new(ModelRegistry::in_memory())),
            get_initial_cwd: Box::new(|| "/initial".to_string()),
            get_initial_session_name: Box::new(|| Some("initial".to_string())),
            get_themes: Box::new(Vec::new),
            refresh_mcp_providers: None,
        };
        let mut mode = InteractiveMode::new(InteractiveModeOptions {
            migrated_providers: None,
            model_fallback_message: None,
            startup_notice: None,
            initial_message: None,
            initial_images: None,
            initial_messages: None,
            initial_prompts: None,
            verbose: false,
            agent_connection: Arc::new(()),
            daemon_socket_path: None,
            local_session_host: None,
            bind_local_session_extensions: false,
            ui_services: Some(services),
            on_shutdown: None,
            return_to_agents_view: false,
            force_fullscreen: false,
            agents_view_owns_startup_notices: false,
            session_depth: None,
            session_has_children: false,
            prompt_stash_store: None,
            prompt_stash_session_id: None,
        })
        .expect("mode");
        mode.apply_connection_state_snapshot(AgentConnectionState {
            session_id: session_id.to_string(),
            active_session_id: active_session_id.map(str::to_string),
            ..Default::default()
        });
        mode
    }

    /// `apply_catalog` keeps the wire jobs in `heartbeat_catalog`, and `scoped`
    /// filters them by the current session. `interactive-mode.ts:2706-2715`.
    #[test]
    fn the_catalog_round_trips_and_scopes_to_the_session() {
        let mut mode = mode("s1", Some("active-s1"));
        let catalog = vec![
            heartbeat("own", "active", "active-s1", "s1"),
            heartbeat("child", "paused", "active-child", "s-child"),
            heartbeat("foreign", "active", "active-other", "s-other"),
        ];
        apply_catalog(&mut mode, &catalog);
        assert_eq!(mode.heartbeat_catalog.len(), 3);

        let scoped_ids: Vec<String> = scoped(&mode, &catalog)
            .iter()
            .map(|entry| entry.job["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(scoped_ids, vec!["own".to_string()]);

        // A child's active session joins the scope
        // (`heartbeat-scope.ts:21-24`).
        mode.subagent_snapshots.insert(
            "child".to_string(),
            crate::modes::interactive::interactive_mode_services::AgentConnectionRlmChildAgentSnapshot {
                active_session_id: Some("active-child".into()),
                ..Default::default()
            },
        );
        let scoped_ids: Vec<String> = scoped(&mode, &catalog)
            .iter()
            .map(|entry| entry.job["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(scoped_ids, vec!["own".to_string(), "child".to_string()]);
    }

    /// A malformed job must not be silently presented as a valid heartbeat.
    #[test]
    fn a_malformed_job_is_dropped_from_the_catalog() {
        let mut mode = mode("s1", Some("active-s1"));
        let catalog = vec![
            heartbeat("ok", "active", "active-s1", "s1"),
            wire::AgentConnectionHeartbeat {
                job: json!("not a job"),
                ..Default::default()
            },
        ];
        apply_catalog(&mut mode, &catalog);
        assert_eq!(mode.heartbeat_catalog.len(), 1);
    }

    /// `refreshDelay` (`interactive-mode.ts:9909-9947` and
    /// `native_host_heartbeats.rs:85-101`): the delay is capped at 60 s and an
    /// already-overdue run falls back to 5 s.
    #[test]
    fn the_refresh_delay_is_capped_and_has_an_overdue_fallback() {
        let now = chrono::Utc::now().timestamp_millis();
        let with_next_run = |id: &str, status: &str, offset_ms: i64| {
            let mut entry = heartbeat(id, status, "active-s1", "s1");
            entry.job["nextRunAt"] =
                json!(chrono::DateTime::from_timestamp_millis(now + offset_ms)
                    .unwrap()
                    .to_rfc3339());
            entry
        };

        assert_eq!(
            refresh_delay(&[with_next_run("h", "active", -60_000)]),
            Some(Duration::from_millis(5_000)),
            "an overdue run must use the 5 s fallback"
        );
        assert_eq!(
            refresh_delay(&[with_next_run("h", "active", 3_600_000)]),
            Some(Duration::from_millis(60_000)),
            "a distant run must be capped at 60 s"
        );
        let near = refresh_delay(&[with_next_run("h", "active", 1_000)]).unwrap();
        assert!(
            near >= Duration::from_millis(1_000) && near <= Duration::from_millis(1_250),
            "a near run adds the 250 ms settle margin, got {near:?}"
        );
        assert_eq!(
            refresh_delay(&[with_next_run("h", "paused", 1_000)]),
            None,
            "a paused heartbeat arms no timer"
        );
        assert_eq!(refresh_delay(&[]), None);
    }

    /// Malformed and overflow dates must neither arm a timer nor panic
    /// (`interactive-mode.ts:9916-9917` filters non-finite `Date.parse`).
    #[test]
    fn malformed_dates_arm_no_timer_and_do_not_overflow() {
        for next_run_at in [
            "9223372036854775807-01-01T00:00:00Z",
            "999999999999-01-01T00:00:00Z",
            "2024-13-45T00:00:00Z",
            "2024-01-01T25:00:00Z",
            "2024-01-01T00:00:60Z",
            "not a date",
            "",
            "   ",
        ] {
            let mut entry = heartbeat("h", "active", "active-s1", "s1");
            entry.job["nextRunAt"] = json!(next_run_at);
            assert_eq!(
                refresh_delay(&[entry]),
                None,
                "{next_run_at:?} must not arm a timer"
            );
        }
    }

    /// The manager's action must run through the real callback: the daemon call
    /// is made with the job's own session and id, and the refreshed catalog
    /// reaches the host (`interactive-mode.ts:9957-9976`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_manager_action_reaches_the_daemon_and_republishes_the_catalog() {
        let connection = Arc::new(FakeHeartbeatConnection::default());
        connection.set_catalog(vec![heartbeat("h1", "active", "active-s1", "s1")]);
        let mut mode = mode("s1", Some("active-s1"));
        apply_catalog(&mut mode, &connection.catalog());
        let mode = Rc::new(RefCell::new(mode));
        let catalog = Rc::new(RefCell::new(connection.catalog()));
        let (send, receive) = host_event_channel(64 * 1024);
        let mut manager = create(
            mode.clone(),
            catalog.clone(),
            Rc::new(Cell::new(24.0)),
            send,
            connection.clone() as Arc<dyn wire::AgentConnection>,
        );

        // Open the manager and pick the first action from the list.
        manager.render(80.0);
        manager.handle_input(
            "
",
        );
        manager.handle_input(
            "
",
        );
        for _ in 0..2_000 {
            manager.poll_action();
            if !connection.actions().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        assert_eq!(
            connection.actions(),
            vec![("active-s1".to_string(), "h1".to_string(), json!("pause"))],
            "the action must carry the job's own session and id"
        );
        // The refreshed catalog is republished to the host, and the manager's
        // own view still lists the paused heartbeat.
        let mut published = false;
        while let Ok(event) = receive.try_recv() {
            if let HostEvent::Heartbeats(catalog, false) = event {
                published = true;
                assert_eq!(catalog.len(), 1);
            }
        }
        assert!(published, "the refreshed catalog must reach the host");
        assert!(
            manager
                .render(80.0)
                .iter()
                .any(|line| line.contains("check the session")),
            "the manager keeps listing the heartbeat it just acted on, got: {:?}",
            manager.render(80.0)
        );
    }

    /// A failing action must surface as an error and leave the manager usable
    /// (`heartbeat-manager.ts:228-241`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_manager_action_reports_an_error_and_stays_usable() {
        let connection = Arc::new(FakeHeartbeatConnection::default());
        connection.set_catalog(vec![heartbeat("h1", "active", "active-s1", "s1")]);
        connection.set_action_error(Some("provider unavailable".into()));
        let mut mode = mode("s1", Some("active-s1"));
        apply_catalog(&mut mode, &connection.catalog());
        let mode = Rc::new(RefCell::new(mode));
        let catalog = Rc::new(RefCell::new(connection.catalog()));
        let (send, _receive) = host_event_channel(64 * 1024);
        let mut manager = create(
            mode,
            catalog,
            Rc::new(Cell::new(24.0)),
            send,
            connection.clone() as Arc<dyn wire::AgentConnection>,
        );

        manager.render(80.0);
        manager.handle_input(
            "
",
        );
        manager.handle_input(
            "
",
        );
        for _ in 0..2_000 {
            manager.poll_action();
            if manager
                .render(80.0)
                .join("\n")
                .contains("provider unavailable")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let rendered = manager.render(80.0).join("\n");
        assert!(
            rendered.contains("provider unavailable"),
            "the failure must be shown, got: {rendered}"
        );

        // A retry is still possible after the failure.
        connection.set_action_error(None);
        manager.handle_input(
            "
",
        );
        for _ in 0..2_000 {
            manager.poll_action();
            if connection.actions().len() >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(
            connection.actions().len(),
            2,
            "a retry must reach the daemon"
        );
    }

    /// Test-only `AgentConnection` for the heartbeat paths.
    #[derive(Default)]
    struct FakeHeartbeatConnection {
        catalog: Mutex<Vec<wire::AgentConnectionHeartbeat>>,
        actions: Mutex<Vec<(String, String, serde_json::Value)>>,
        action_error: Mutex<Option<String>>,
    }

    impl FakeHeartbeatConnection {
        fn set_catalog(&self, catalog: Vec<wire::AgentConnectionHeartbeat>) {
            *self.catalog.lock().unwrap() = catalog;
        }

        fn catalog(&self) -> Vec<wire::AgentConnectionHeartbeat> {
            self.catalog.lock().unwrap().clone()
        }

        fn set_action_error(&self, error: Option<String>) {
            *self.action_error.lock().unwrap() = error;
        }

        fn actions(&self) -> Vec<(String, String, serde_json::Value)> {
            self.actions.lock().unwrap().clone()
        }
    }

    fn unsupported<T: Send + 'static>(method: &str) -> pi_ai::types::BoxFuture<Result<T, String>> {
        let method = method.to_string();
        Box::pin(async move {
            Err(format!(
                "FakeHeartbeatConnection does not implement {method}"
            ))
        })
    }

    impl wire::AgentConnection for FakeHeartbeatConnection {
        fn subscribe(
            &self,
            _listener: wire::AgentConnectionEventListener,
        ) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }

        fn on_before_session_invalidate(
            &self,
            _listener: wire::AgentConnectionBeforeSessionInvalidateListener,
        ) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }

        fn get_state(&self) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionState, String>> {
            Box::pin(async { Ok(wire::AgentConnectionState::default()) })
        }

        fn get_initial_snapshot(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionSnapshot, String>> {
            Box::pin(async {
                Ok(wire::AgentConnectionSnapshot {
                    state: wire::AgentConnectionState::default(),
                    ..Default::default()
                })
            })
        }

        fn get_queue(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionQueueState, String>> {
            Box::pin(async { Ok(wire::AgentConnectionQueueState::default()) })
        }

        fn mutate_queued_message(
            &self,
            _lane: &str,
            _index: i64,
            _expected_text: &str,
            _mutation: Value,
        ) -> pi_ai::types::BoxFuture<Result<String, String>> {
            Box::pin(async { Ok("unsupported".to_string()) })
        }

        fn list_heartbeats(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionHeartbeat>, String>> {
            let catalog = self.catalog.lock().unwrap().clone();
            Box::pin(async move { Ok(catalog) })
        }

        fn manage_heartbeat(
            &self,
            active_session_id: &str,
            job_id: &str,
            action: serde_json::Value,
        ) -> pi_ai::types::BoxFuture<Result<serde_json::Value, String>> {
            self.actions.lock().unwrap().push((
                active_session_id.to_string(),
                job_id.to_string(),
                action.clone(),
            ));
            let error = self.action_error.lock().unwrap().clone();
            Box::pin(async move { error.map_or(Ok(serde_json::Value::Null), Err) })
        }

        fn dispose(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            Box::pin(async { Ok(()) })
        }

        fn get_rlm_child_snapshots(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionRlmChildAgentSnapshot>, String>>
        {
            unsupported("get_rlm_child_snapshots")
        }
        fn get_messages(&self) -> pi_ai::types::BoxFuture<Result<Vec<AgentMessage>, String>> {
            unsupported("get_messages")
        }
        fn get_session_header(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<wire::AgentConnectionSessionHeader>, String>>
        {
            unsupported("get_session_header")
        }
        fn get_commands(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionSlashCommand>, String>>
        {
            unsupported("get_commands")
        }
        fn get_resource_snapshot(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionResourceSnapshot, String>>
        {
            unsupported("get_resource_snapshot")
        }
        fn get_model_catalog(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionModelCatalog, String>> {
            unsupported("get_model_catalog")
        }
        fn get_available_models(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionModel>, String>> {
            unsupported("get_available_models")
        }
        fn get_session_stats(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("get_session_stats")
        }
        fn get_context_tree(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("get_context_tree")
        }
        fn get_session_context(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionSessionContext, String>> {
            unsupported("get_session_context")
        }
        fn get_session_tree(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionWatchSessionTree, String>>
        {
            unsupported("get_session_tree")
        }
        fn list_saved_sessions(
            &self,
            _scope: &str,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionSavedSessionInfo>, String>>
        {
            unsupported("list_saved_sessions")
        }
        fn clear_queue(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionQueueState, String>> {
            unsupported("clear_queue")
        }
        fn abort_and_clear_queue(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionQueueState, String>> {
            unsupported("abort_and_clear_queue")
        }
        fn acquire_session_input_pause(
            &self,
            _lease_key: &str,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionSessionInputPause, String>>
        {
            unsupported("acquire_session_input_pause")
        }
        fn list_cron_jobs(
            &self,
            _include_inactive: bool,
        ) -> pi_ai::types::BoxFuture<Result<Vec<Value>, String>> {
            unsupported("list_cron_jobs")
        }
        fn add_cron_job(
            &self,
            _schedule: &str,
            _prompt: &str,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("add_cron_job")
        }
        fn cancel_cron_job(&self, _job_id: &str) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("cancel_cron_job")
        }
        fn get_heartbeat(&self) -> pi_ai::types::BoxFuture<Result<Option<Value>, String>> {
            unsupported("get_heartbeat")
        }
        fn set_heartbeat(
            &self,
            _schedule: &str,
            _instruction: &str,
            _delivery_mode: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("set_heartbeat")
        }
        fn update_heartbeat(
            &self,
            _action: Value,
        ) -> pi_ai::types::BoxFuture<Result<Option<Value>, String>> {
            unsupported("update_heartbeat")
        }
        fn send_agent_message(
            &self,
            _target_active_session_id: &str,
            _message: &str,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("send_agent_message")
        }
        fn get_agent_message_status(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("get_agent_message_status")
        }
        fn pause_agent_messages(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("pause_agent_messages")
        }
        fn resume_agent_messages(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("resume_agent_messages")
        }
        fn clear_agent_messages(&self) -> pi_ai::types::BoxFuture<Result<f64, String>> {
            unsupported("clear_agent_messages")
        }
        fn get_user_messages_for_forking(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Vec<wire::AgentConnectionUserMessage>, String>>
        {
            unsupported("get_user_messages_for_forking")
        }
        fn get_last_assistant_text(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<String>, String>> {
            unsupported("get_last_assistant_text")
        }
        fn get_system_prompt(&self) -> pi_ai::types::BoxFuture<Result<String, String>> {
            unsupported("get_system_prompt")
        }
        fn get_tool_definition(
            &self,
            _name: &str,
        ) -> pi_ai::types::BoxFuture<Result<Option<wire::AgentConnectionToolDefinition>, String>>
        {
            unsupported("get_tool_definition")
        }
        fn set_session_entry_label(
            &self,
            _entry_id: &str,
            _label: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_session_entry_label")
        }
        fn respond_to_extension_ui_request(
            &self,
            _request_id: &str,
            _response: wire::AgentConnectionExtensionUiResponse,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("respond_to_extension_ui_request")
        }
        fn prompt(
            &self,
            _message: &str,
            _options: Option<wire::AgentConnectionPromptOptions>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("prompt")
        }
        fn prompt_and_wait(
            &self,
            _message: &str,
            _options: Option<wire::AgentConnectionPromptOptions>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("prompt_and_wait")
        }
        fn start_side_question(
            &self,
            _id: &str,
            _question: &str,
            _previous_turns: Option<Vec<wire::AgentConnectionSideQuestionTurn>>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("start_side_question")
        }
        fn abort_side_question(&self, _id: &str) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("abort_side_question")
        }
        fn steer(
            &self,
            _message: &str,
            _images: Option<Vec<ImageContent>>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("steer")
        }
        fn follow_up(
            &self,
            _message: &str,
            _images: Option<Vec<ImageContent>>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("follow_up")
        }
        fn abort(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort")
        }
        fn cancel_rlm_child(
            &self,
            _child_id: &str,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("cancel_rlm_child")
        }
        fn wait_for_idle(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("wait_for_idle")
        }
        fn wait_for_headless_completion(
            &self,
            _options: Option<wire::AgentConnectionHeadlessCompletionOptions>,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentAutonomousStatus, String>> {
            unsupported("wait_for_headless_completion")
        }
        fn execute_bash(
            &self,
            _command: &str,
            _options: Option<wire::AgentConnectionExecuteBashOptions>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("execute_bash")
        }
        fn execute_bash_and_wait(
            &self,
            _command: &str,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("execute_bash_and_wait")
        }
        fn abort_bash(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort_bash")
        }
        fn set_model(
            &self,
            _provider: &str,
            _model_id: &str,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionModel, String>> {
            unsupported("set_model")
        }
        fn cycle_model(
            &self,
            _direction: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<Option<wire::AgentConnectionModelCycleResult>, String>>
        {
            unsupported("cycle_model")
        }
        fn set_scoped_models(
            &self,
            _scoped_models: Vec<wire::AgentConnectionScopedModel>,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_scoped_models")
        }
        fn set_thinking_level(
            &self,
            _level: ThinkingLevel,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_thinking_level")
        }
        fn set_service_tier(
            &self,
            _service_tier: ServiceTier,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_service_tier")
        }
        fn cycle_thinking_level(
            &self,
        ) -> pi_ai::types::BoxFuture<Result<Option<ThinkingLevel>, String>> {
            unsupported("cycle_thinking_level")
        }
        fn set_transport(
            &self,
            _transport: Transport,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_transport")
        }
        fn set_steering_mode(&self, _mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_steering_mode")
        }
        fn set_follow_up_mode(&self, _mode: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_follow_up_mode")
        }
        fn set_auto_compaction_enabled(
            &self,
            _enabled: bool,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_auto_compaction_enabled")
        }
        fn set_auto_retry_enabled(
            &self,
            _enabled: bool,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_auto_retry_enabled")
        }
        fn compact(
            &self,
            _custom_instructions: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("compact")
        }
        fn refine(&self, _options: Value) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("refine")
        }
        fn abort_compaction(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort_compaction")
        }
        fn abort_branch_summary(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort_branch_summary")
        }
        fn abort_retry(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("abort_retry")
        }
        fn reload(&self) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("reload")
        }
        fn new_session(
            &self,
            _options: Option<wire::AgentConnectionNewSessionOptions>,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("new_session")
        }
        fn switch_session(
            &self,
            _session_path: &str,
            _options: Option<wire::AgentConnectionSwitchSessionOptions>,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("switch_session")
        }
        fn fork(
            &self,
            _entry_id: &str,
            _options: Option<wire::AgentConnectionForkOptions>,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("fork")
        }
        fn navigate_tree(
            &self,
            _target_id: &str,
            _options: Option<wire::AgentConnectionNavigateTreeOptions>,
        ) -> pi_ai::types::BoxFuture<Result<wire::AgentConnectionNavigateTreeResult, String>>
        {
            unsupported("navigate_tree")
        }
        fn import_from_jsonl(
            &self,
            _input_path: &str,
            _cwd_override: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<bool, String>> {
            unsupported("import_from_jsonl")
        }
        fn export_to_html(
            &self,
            _output_path: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<String, String>> {
            unsupported("export_to_html")
        }
        fn export_to_jsonl(
            &self,
            _output_path: Option<&str>,
        ) -> pi_ai::types::BoxFuture<Result<String, String>> {
            unsupported("export_to_jsonl")
        }
        fn set_session_name(&self, _name: &str) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("set_session_name")
        }
        fn get_rlm_max_depth_status(&self) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("get_rlm_max_depth_status")
        }
        fn set_rlm_max_depth(
            &self,
            _max_depth: f64,
            _options: Option<Value>,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("set_rlm_max_depth")
        }
        fn rename_saved_session(
            &self,
            _session_path: &str,
            _name: &str,
        ) -> pi_ai::types::BoxFuture<Result<(), String>> {
            unsupported("rename_saved_session")
        }
        fn delete_saved_session(
            &self,
            _session_path: &str,
        ) -> pi_ai::types::BoxFuture<Result<Value, String>> {
            unsupported("delete_saved_session")
        }
        fn watch_session(
            &self,
            _active_session_id: &str,
        ) -> pi_ai::types::BoxFuture<
            Result<Option<Box<dyn wire::AgentConnectionSessionWatcher>>, String>,
        > {
            unsupported("watch_session")
        }
    }

    /// Finding D-08: the heartbeat scope predicate must keep the two id spaces
    /// apart, exactly like the TypeScript (`heartbeat-scope.ts:29`).
    mod heartbeat_id_space_tests {
        use super::*;
        // Explicit imports: the outer test module reaches these through its own glob, and
        // a glob import is not re-exported, so name them here instead of relying on it.
        use crate::modes::agent_connection::types as wire;
        use crate::modes::interactive::interactive_mode::InteractiveMode;

    /// The Rust port compares the job's active-session id against the session's
    /// DURABLE id (`heartbeat_scope.rs:35-40`), which mixes the two id spaces, and the
    /// local `AgentCronJob` stand-in has no durable `session_id` field, so
    /// `project_heartbeat` (`native_host.rs:3683-3700`) drops the wire `sessionId`
    /// entirely, while the TypeScript compares `heartbeat.job.sessionId` to the
    /// session's durable id (`heartbeat-scope.ts:29`).
    fn scoped_heartbeat_ids(
        mode: &InteractiveMode,
        catalog: &[wire::AgentConnectionHeartbeat],
    ) -> Vec<String> {
        scoped(mode, catalog)
            .iter()
            .map(|entry| entry.job["id"].as_str().unwrap().to_string())
            .collect()
    }

    /// A job whose DURABLE session id is the current session's must stay in scope even
    /// when its worker id belongs to a superseded worker (a rebound/resumed session).
    /// The port drops it.
    #[test]
    fn a_job_with_the_current_durable_session_id_is_kept() {
        let mut mode = mode("durable-1", Some("worker-1"));
        let catalog = vec![
            heartbeat("rebound", "active", "worker-old", "durable-1"),
            heartbeat("foreign", "active", "worker-other", "durable-other"),
        ];
        apply_catalog(&mut mode, &catalog);
        assert_eq!(mode.heartbeat_catalog.len(), 2, "both jobs project");
        assert_eq!(
            scoped_heartbeat_ids(&mode, &catalog),
            vec!["rebound".to_string()],
            "DEFECT D-08: a heartbeat whose durable sessionId is the current session's must stay in scope"
        );
        let scoped_ids: Vec<String> = mode
            .get_scoped_heartbeats()
            .iter()
            .map(|entry| entry.job.id.clone())
            .collect();
        assert_eq!(
            scoped_ids,
            vec!["rebound".to_string()],
            "the production scope path (`getScopedHeartbeats`) must keep the rebound heartbeat"
        );
    }

    /// The other id space: a job whose WORKER id is in the active set stays in scope
    /// even when its durable session id is foreign (that is the TS second disjunct).
    /// This is the negative control: it must pass before and after the fix.
    #[test]
    fn a_job_in_the_active_session_set_is_kept() {
        let mut mode = mode("durable-1", Some("worker-1"));
        let catalog = vec![heartbeat("own", "active", "worker-1", "durable-1")];
        apply_catalog(&mut mode, &catalog);
        assert_eq!(scoped_heartbeat_ids(&mode, &catalog), vec!["own".to_string()]);

        // A child's worker joins the active set (heartbeat-scope.ts:21-24).
        mode.subagent_snapshots.insert(
            "child".to_string(),
            crate::modes::interactive::interactive_mode_services::AgentConnectionRlmChildAgentSnapshot {
                active_session_id: Some("worker-child".into()),
                ..Default::default()
            },
        );
        let catalog = vec![heartbeat("child-job", "active", "worker-child", "durable-child")];
        apply_catalog(&mut mode, &catalog);
        assert_eq!(
            scoped_heartbeat_ids(&mode, &catalog),
            vec!["child-job".to_string()],
            "a child's active worker id stays in scope"
        );
    }

    /// The mirrored confusion: a job whose WORKER id happens to equal the current
    /// DURABLE session id matches neither TS disjunct (the active set holds worker
    /// ids), so it must be dropped. The port keeps it.
    #[test]
    fn a_job_matching_neither_id_space_is_dropped() {
        let mut mode = mode("durable-1", Some("worker-1"));
        let catalog = vec![heartbeat("collision", "active", "durable-1", "durable-other")];
        apply_catalog(&mut mode, &catalog);
        assert!(
            scoped_heartbeat_ids(&mode, &catalog).is_empty(),
            "DEFECT D-08: a job matching neither id space must be dropped, got {:?}",
            scoped_heartbeat_ids(&mode, &catalog)
        );
        assert!(
            mode.get_scoped_heartbeats().is_empty(),
            "the production scope path must drop it too"
        );
    }

    /// A job whose durable session id is the current session's stays in scope even when
    /// the active-session id is absent from the snapshot: the first TS disjunct does not
    /// depend on the active set.
    #[test]
    fn the_durable_disjunct_does_not_depend_on_the_active_set() {
        let mut mode = mode("durable-1", None);
        let catalog = vec![
            heartbeat("durable-match", "active", "worker-old", "durable-1"),
            heartbeat("worker-only", "active", "worker-old", "durable-other"),
        ];
        apply_catalog(&mut mode, &catalog);
        assert_eq!(
            scoped_heartbeat_ids(&mode, &catalog),
            vec!["durable-match".to_string()],
            "with no active session id only the durable-session disjunct can match"
        );
    }
    }
}
