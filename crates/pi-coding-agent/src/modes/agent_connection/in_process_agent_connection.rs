//! Port of packages/coding-agent/src/modes/agent-connection/in-process-agent-connection.ts
//!
//! In-process adapter over `AgentSessionRuntime`. The runtime host slice has not
//! landed, so the port drives an explicit `InProcessRuntimeHost` seam that
//! mirrors the exact methods this file calls on the TypeScript host.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{BoxFuture, ImageContent, Model, ServiceTier, Transport};
use serde_json::Value;

use crate::core::agent_session::ExtensionBindings;
use crate::core::extensions::runner::{ExtensionErrorListener, ShutdownHandler};
use crate::core::extensions::types::{
    CancelledResult, ExtensionCommandContextActions, ExtensionError, ExtensionUiContext, ForkOptions,
    NavigateTreeOptions, NewSessionOptions, SwitchSessionOptions,
};
use crate::modes::agent_connection::snapshot::{
    create_agent_connection_snapshot, create_agent_connection_state, AgentSessionRuntimeSnapshotSource,
};
use crate::modes::agent_connection::tool_definition::create_agent_connection_tool_definition;
use crate::modes::agent_connection::types::*;

/// The `AgentSessionRuntime` surface this adapter uses.
///
/// Every method name mirrors the TypeScript member it replaces.
pub trait InProcessRuntimeHost: Send + Sync {
    fn session_start_side_question(&self, _id: String, _question: String,
        _previous: Option<Vec<crate::core::side_question::SideQuestionTurn>>,
        _on_event: Arc<dyn Fn(crate::core::side_question::SideQuestionEvent) -> BoxFuture<()> + Send + Sync>,
    ) -> Result<crate::core::side_question::SideQuestionRun, String> {
        Err("This runtime does not provide standalone side-question agents".into())
    }
    fn snapshot_source(&self) -> AgentSessionRuntimeSnapshotSource;
    fn session_header(&self) -> Option<AgentConnectionSessionHeader>;
    fn session_subscribe(&self, listener: AgentConnectionEventListener) -> Box<dyn Fn() + Send + Sync>;
    fn session_wait_for_headless_completion(
        &self,
        options: Option<AgentConnectionHeadlessCompletionOptions>,
    ) -> BoxFuture<Result<AgentAutonomousStatus, String>>;
    fn session_messages(&self) -> Vec<AgentMessage>;
    fn session_state_messages(&self) -> Vec<AgentMessage>;
    fn session_commands(&self) -> Vec<AgentConnectionSlashCommand>;
    fn session_resource_snapshot(&self) -> AgentConnectionResourceSnapshot;
    fn session_model_catalog(&self) -> AgentConnectionModelCatalog;
    fn session_available_models(&self) -> Vec<AgentConnectionModel>;
    fn session_stats(&self) -> Value;
    fn session_context_tree(&self) -> Value;
    fn session_context(&self) -> AgentConnectionSessionContext;
    fn session_tree(&self) -> AgentConnectionWatchSessionTree;
    fn session_rlm_children(&self) -> Vec<AgentConnectionRlmChildAgentSnapshot>;
    fn session_queue(&self) -> AgentConnectionQueueState;
    fn session_mutate_queued_message(&self, lane: &str, index: i64, expected_text: &str, mutation: Value) -> String;
    fn session_clear_queue(&self) -> AgentConnectionQueueState;
    fn session_request_abort(&self);
    fn session_acquire_input_pause(&self) -> AgentConnectionSessionInputPause;
    fn session_get_user_messages_for_forking(&self) -> Vec<AgentConnectionUserMessage>;
    fn session_last_assistant_text(&self) -> Option<String>;
    fn session_system_prompt(&self) -> String;
    fn session_tool_definition(&self, name: &str) -> Option<crate::modes::agent_connection::tool_definition::ToolDefinition>;
    fn session_append_label_change(&self, entry_id: &str, label: Option<&str>);
    fn session_prompt(&self, message: &str, options: Value) -> BoxFuture<Result<(), String>>;
    fn session_prompt_and_wait(&self, message: &str, options: Value) -> BoxFuture<Result<(), String>>;
    fn session_steer(&self, message: &str, images: Option<Vec<ImageContent>>) -> BoxFuture<Result<(), String>>;
    fn session_follow_up(&self, message: &str, images: Option<Vec<ImageContent>>) -> BoxFuture<Result<(), String>>;
    fn session_cancel_rlm_child(&self, child_id: &str) -> bool;
    fn session_wait_for_idle(&self) -> BoxFuture<()>;
    fn session_run_user_bash(&self, command: &str, options: Option<AgentConnectionExecuteBashOptions>) -> BoxFuture<Result<(), String>>;
    fn session_execute_bash(&self, command: &str) -> BoxFuture<Result<Value, String>>;
    fn session_abort_bash(&self);
    fn session_set_model(&self, model: Model) -> BoxFuture<Result<(), String>>;
    fn session_model_registry_available_models(&self) -> BoxFuture<Vec<AgentConnectionModel>>;
    fn session_model_registry_provider_auth_source(&self, provider: &str) -> String;
    fn session_model_registry_find(&self, provider: &str, model_id: &str) -> Option<Model>;
    fn session_cycle_model(&self, direction: &str) -> BoxFuture<Result<Option<AgentConnectionModelCycleResult>, String>>;
    fn session_set_scoped_models(&self, scoped_models: Vec<AgentConnectionScopedModel>);
    fn session_set_thinking_level(&self, level: ThinkingLevel);
    fn session_set_service_tier(&self, service_tier: ServiceTier);
    fn session_cycle_thinking_level(&self) -> Option<ThinkingLevel>;
    fn session_set_transport(&self, transport: Transport);
    fn session_set_steering_mode(&self, mode: &str);
    fn session_set_follow_up_mode(&self, mode: &str);
    fn session_set_auto_compaction_enabled(&self, enabled: bool);
    fn session_set_auto_retry_enabled(&self, enabled: bool);
    fn session_compact(&self, custom_instructions: Option<&str>) -> BoxFuture<Result<Value, String>>;
    fn session_refine(&self, options: Value) -> BoxFuture<Result<Value, String>>;
    fn session_abort_compaction(&self);
    fn session_abort_branch_summary(&self);
    fn session_abort_retry(&self);
    fn session_reload(&self) -> BoxFuture<Result<(), String>>;
    fn session_navigate_tree(
        &self,
        target_id: &str,
        options: Option<AgentConnectionNavigateTreeOptions>,
    ) -> BoxFuture<Result<AgentConnectionNavigateTreeResult, String>>;
    fn session_export_to_html(&self, output_path: Option<&str>) -> BoxFuture<Result<String, String>>;
    fn session_export_to_jsonl(&self, output_path: Option<&str>) -> BoxFuture<Result<String, String>>;
    fn session_set_session_name(&self, name: &str);
    fn session_get_rlm_max_depth_status(&self) -> Value;
    fn session_set_rlm_max_depth(&self, max_depth: f64, options: Option<Value>) -> BoxFuture<Result<Value, String>>;
    /// `session.bindExtensions({ uiContext, commandContextActions, shutdownHandler, onError })`.
    ///
    /// The TypeScript passes live extension objects (in-process-agent-connection.ts:657-682),
    /// so the seam carries the canonical Rust owners, never a JSON value.
    fn session_bind_extensions(
        &self,
        bindings: ExtensionBindings,
    ) -> BoxFuture<Result<(), String>>;
    fn session_watch_child(&self, child_id: &str) -> Option<Box<dyn AgentConnectionSessionWatcher>>;
    fn runtime_new_session(&self, options: Option<AgentConnectionNewSessionOptions>) -> BoxFuture<Result<bool, String>>;
    fn runtime_switch_session(
        &self,
        session_path: &str,
        options: Option<AgentConnectionSwitchSessionOptions>,
    ) -> BoxFuture<Result<bool, String>>;
    fn runtime_fork(&self, entry_id: &str, options: Option<AgentConnectionForkOptions>) -> BoxFuture<Result<Value, String>>;
    fn runtime_import_from_jsonl(
        &self,
        input_path: &str,
        cwd_override: Option<&str>,
    ) -> BoxFuture<Result<bool, String>>;
    fn runtime_set_before_session_invalidate(&self, listener: Option<Arc<dyn Fn() + Send + Sync>>);
    fn runtime_set_rebind_session(&self, listener: Option<Arc<dyn Fn() -> BoxFuture<()> + Send + Sync>>);
    fn runtime_dispose(&self) -> BoxFuture<()>;
}

/// `InProcessHeadlessExtensionOptions` (in-process-agent-connection.ts:74-77).
#[derive(Default, Clone)]
pub struct InProcessHeadlessExtensionOptions {
    pub ui_context: Option<Arc<dyn ExtensionUiContext>>,
    pub shutdown_handler: Option<ShutdownHandler>,
}

/// `SideQuestionRun` surface used by `startSideQuestion`.
pub type SideQuestionRun = Arc<dyn Fn() + Send + Sync>;

pub struct InProcessAgentConnection {
    runtime_host: Arc<dyn InProcessRuntimeHost>,
    listeners: Arc<Mutex<Vec<AgentConnectionEventListener>>>,
    before_session_invalidate_listeners: Arc<Mutex<Vec<AgentConnectionBeforeSessionInvalidateListener>>>,
    side_question_runs: Arc<Mutex<HashMap<String, SideQuestionRun>>>,
    session_input_pauses: Mutex<HashMap<String, AgentConnectionSessionInputPause>>,
    headless_extension_options: Mutex<Option<InProcessHeadlessExtensionOptions>>,
    unsubscribe_session_events: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

impl InProcessAgentConnection {
    pub fn new(runtime_host: Arc<dyn InProcessRuntimeHost>) -> Self {
        let connection = Self {
            runtime_host,
            listeners: Arc::new(Mutex::new(Vec::new())),
            before_session_invalidate_listeners: Arc::new(Mutex::new(Vec::new())),
            side_question_runs: Arc::new(Mutex::new(HashMap::new())),
            session_input_pauses: Mutex::new(HashMap::new()),
            headless_extension_options: Mutex::new(None),
            unsubscribe_session_events: Mutex::new(None),
        };
        connection.bind_current_session_events();
        connection
    }

    pub fn runtime_host(&self) -> &Arc<dyn InProcessRuntimeHost> {
        &self.runtime_host
    }

    pub fn bind_headless_extensions(&self, options: InProcessHeadlessExtensionOptions) -> BoxFuture<Result<(), String>> {
        *self.headless_extension_options.lock().unwrap() = Some(options);
        self.bind_current_session_extensions()
    }

    fn bind_current_session_events(&self) {
        if let Some(previous) = self.unsubscribe_session_events.lock().unwrap().take() {
            previous();
        }
        let listeners = self.listeners.clone();
        let handle = self.runtime_host.session_subscribe(Arc::new(move |event| {
            let listeners = listeners.lock().unwrap().clone();
            Box::pin(async move {
                for listener in listeners {
                    listener(event.clone()).await;
                }
            })
        }));
        *self.unsubscribe_session_events.lock().unwrap() = Some(handle);
    }

    /// `bindCurrentSessionExtensions()` (in-process-agent-connection.ts:655-683).
    ///
    /// The TypeScript builds one typed object: `uiContext` and `shutdownHandler`
    /// come from the headless options, `commandContextActions` is built inline,
    /// and `onError` re-emits the extension error on this connection. Every action
    /// forwards to the same target the TypeScript forwards to.
    fn bind_current_session_extensions(&self) -> BoxFuture<Result<(), String>> {
        let options = self.headless_extension_options.lock().unwrap().clone();
        let host = self.runtime_host.clone();
        let listeners = self.listeners.clone();

        let wait_for_idle_host = host.clone();
        let wait_for_idle: Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync> =
            Arc::new(move || wait_for_idle_host.session_wait_for_idle());

        let new_session_host = host.clone();
        let new_session: Arc<
            dyn Fn(Option<NewSessionOptions>) -> Pin<Box<dyn Future<Output = CancelledResult> + Send>>
                + Send
                + Sync,
        > = Arc::new(move |options: Option<NewSessionOptions>| {
            let host = new_session_host.clone();
            let parent_session = options.and_then(|options| options.parent_session);
            Box::pin(async move {
                match host
                    .runtime_new_session(Some(AgentConnectionNewSessionOptions { parent_session }))
                    .await
                {
                    Ok(cancelled) => CancelledResult { cancelled },
                    Err(error) => {
                        eprintln!("Warning: Could not start a new session from an extension command: {error}");
                        CancelledResult { cancelled: false }
                    }
                }
            })
        });

        let fork_host = host.clone();
        let fork: Arc<
            dyn Fn(String, Option<ForkOptions>) -> Pin<Box<dyn Future<Output = CancelledResult> + Send>>
                + Send
                + Sync,
        > = Arc::new(move |entry_id: String, options: Option<ForkOptions>| {
            let host = fork_host.clone();
            let position = options.and_then(|options| options.position);
            Box::pin(async move {
                match host
                    .runtime_fork(&entry_id, Some(AgentConnectionForkOptions { position }))
                    .await
                {
                    Ok(result) => CancelledResult {
                        cancelled: result.get("cancelled").and_then(Value::as_bool).unwrap_or(false),
                    },
                    Err(error) => {
                        eprintln!("Warning: Could not fork from an extension command: {error}");
                        CancelledResult { cancelled: false }
                    }
                }
            })
        });

        let navigate_tree_host = host.clone();
        let navigate_tree: Arc<
            dyn Fn(String, Option<NavigateTreeOptions>) -> Pin<Box<dyn Future<Output = CancelledResult> + Send>>
                + Send
                + Sync,
        > = Arc::new(move |target_id: String, options: Option<NavigateTreeOptions>| {
            let host = navigate_tree_host.clone();
            let options = options.map(|options| AgentConnectionNavigateTreeOptions {
                summarize: options.summarize,
                custom_instructions: options.custom_instructions,
                replace_instructions: options.replace_instructions,
                label: options.label,
            });
            Box::pin(async move {
                match host.session_navigate_tree(&target_id, options).await {
                    Ok(result) => CancelledResult { cancelled: result.cancelled },
                    Err(error) => {
                        eprintln!("Warning: Could not navigate the session tree from an extension command: {error}");
                        CancelledResult { cancelled: false }
                    }
                }
            })
        });

        let switch_session_host = host.clone();
        let switch_session: Arc<
            dyn Fn(String, Option<SwitchSessionOptions>) -> Pin<Box<dyn Future<Output = CancelledResult> + Send>>
                + Send
                + Sync,
        > = Arc::new(move |session_path: String, _options: Option<SwitchSessionOptions>| {
            let host = switch_session_host.clone();
            Box::pin(async move {
                // The connection-layer options carry `cwdOverride`; the extension
                // options carry `withSession`, so nothing maps onto it here.
                match host.runtime_switch_session(&session_path, None).await {
                    Ok(cancelled) => CancelledResult { cancelled },
                    Err(error) => {
                        eprintln!("Warning: Could not switch sessions from an extension command: {error}");
                        CancelledResult { cancelled: false }
                    }
                }
            })
        });

        let reload_host = host.clone();
        let reload: Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync> =
            Arc::new(move || {
                let host = reload_host.clone();
                Box::pin(async move {
                    if let Err(error) = host.session_reload().await {
                        eprintln!("Warning: Could not reload the session from an extension command: {error}");
                    }
                })
            });

        let bindings = ExtensionBindings {
            ui_context: options.as_ref().and_then(|options| options.ui_context.clone()),
            command_context_actions: Some(ExtensionCommandContextActions {
                wait_for_idle,
                new_session,
                fork,
                navigate_tree,
                switch_session,
                reload,
            }),
            shutdown_handler: options.as_ref().and_then(|options| options.shutdown_handler.clone()),
            on_error: Some(Arc::new(move |error: ExtensionError| {
                let delivery = emit_to_listeners(
                    &listeners,
                    AgentConnectionEvent::ExtensionError {
                        extension_path: error.extension_path.clone(),
                        event: error.event.clone(),
                        error: error.error.clone(),
                    },
                );
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    handle.spawn(delivery);
                }
            })),
        };
        host.session_bind_extensions(bindings)
    }

    fn abort_all_side_questions(&self) {
        let mut runs = self.side_question_runs.lock().unwrap();
        for run in runs.values() {
            run();
        }
        runs.clear();
    }

    fn emit(&self, event: AgentConnectionEvent) -> BoxFuture<()> {
        emit_to_listeners(&self.listeners, event)
    }
}

impl AgentConnection for InProcessAgentConnection {
    fn subscribe(&self, listener: AgentConnectionEventListener) -> Box<dyn Fn() + Send + Sync> {
        self.listeners.lock().unwrap().push(listener.clone());
        let listeners = self.listeners.clone();
        let key_for_removal = listener;
        Box::new(move || {
            let mut guard = listeners.lock().unwrap();
            if let Some(index) = guard.iter().position(|entry| Arc::ptr_eq(entry, &key_for_removal)) {
                guard.remove(index);
            }
        })
    }

    fn on_before_session_invalidate(
        &self,
        listener: AgentConnectionBeforeSessionInvalidateListener,
    ) -> Box<dyn Fn() + Send + Sync> {
        self.before_session_invalidate_listeners.lock().unwrap().push(listener.clone());
        let listeners = self.before_session_invalidate_listeners.clone();
        Box::new(move || {
            let mut guard = listeners.lock().unwrap();
            if let Some(index) = guard.iter().position(|entry| Arc::ptr_eq(entry, &listener)) {
                guard.remove(index);
            }
        })
    }

    fn get_state(&self) -> BoxFuture<Result<AgentConnectionState, String>> {
        let state = create_agent_connection_state(&self.runtime_host.snapshot_source(), None);
        Box::pin(async move { Ok(state) })
    }

    fn supports_jev_dynamic(&self) -> bool { true }
    fn supports_execution_mode(&self) -> bool { true }
    fn supports_node_execution_mode(&self) -> bool { true }

    fn supports_jev_features(&self) -> bool {
        true
    }

    fn get_jev_status(&self) -> BoxFuture<Result<Option<Value>, String>> {
        // The header supplies canonical identity without cloning transcript or
        // provider state. A missing observation remains unknown, never zero.
        let pipeline = self.runtime_host.session_header().and_then(|header| {
            crate::core::jev_bridge::session_status_snapshot(&header.id)
        });
        Box::pin(async move { Ok(Some(serde_json::json!({ "pipeline": pipeline }))) })
    }

    fn get_initial_snapshot(&self) -> BoxFuture<Result<AgentConnectionSnapshot, String>> {
        let snapshot = create_agent_connection_snapshot(&self.runtime_host.snapshot_source(), None);
        Box::pin(async move { Ok(snapshot) })
    }

    fn get_rlm_child_snapshots(&self) -> BoxFuture<Result<Vec<AgentConnectionRlmChildAgentSnapshot>, String>> {
        let children = self.runtime_host.session_rlm_children();
        Box::pin(async move { Ok(children) })
    }

    fn get_messages(&self) -> BoxFuture<Result<Vec<AgentMessage>, String>> {
        let messages = self.runtime_host.session_state_messages();
        Box::pin(async move { Ok(messages) })
    }

    fn get_session_header(&self) -> BoxFuture<Result<Option<AgentConnectionSessionHeader>, String>> {
        let header = self.runtime_host.session_header();
        Box::pin(async move { Ok(header) })
    }

    fn get_commands(&self) -> BoxFuture<Result<Vec<AgentConnectionSlashCommand>, String>> {
        let commands = self.runtime_host.session_commands();
        Box::pin(async move { Ok(commands) })
    }

    fn get_resource_snapshot(&self) -> BoxFuture<Result<AgentConnectionResourceSnapshot, String>> {
        let snapshot = self.runtime_host.session_resource_snapshot();
        Box::pin(async move { Ok(snapshot) })
    }

    fn get_model_catalog(&self) -> BoxFuture<Result<AgentConnectionModelCatalog, String>> {
        let catalog = self.runtime_host.session_model_catalog();
        Box::pin(async move { Ok(catalog) })
    }

    fn get_available_models(&self) -> BoxFuture<Result<Vec<AgentConnectionModel>, String>> {
        let models = self.runtime_host.session_model_registry_available_models();
        Box::pin(async move { Ok(models.await) })
    }

    fn get_session_stats(&self) -> BoxFuture<Result<Value, String>> {
        let stats = self.runtime_host.session_stats();
        Box::pin(async move { Ok(stats) })
    }

    fn get_context_tree(&self) -> BoxFuture<Result<Value, String>> {
        let tree = self.runtime_host.session_context_tree();
        Box::pin(async move { Ok(tree) })
    }

    fn get_session_context(&self) -> BoxFuture<Result<AgentConnectionSessionContext, String>> {
        let context = self.runtime_host.session_context();
        Box::pin(async move { Ok(context) })
    }

    fn get_session_tree(&self) -> BoxFuture<Result<AgentConnectionWatchSessionTree, String>> {
        let tree = self.runtime_host.session_tree();
        Box::pin(async move { Ok(tree) })
    }

    fn list_saved_sessions(
        &self,
        scope: &str,
    ) -> BoxFuture<Result<Vec<AgentConnectionSavedSessionInfo>, String>> {
        let source = self.runtime_host.snapshot_source().session;
        let current = scope == "current";
        Box::pin(async move {
            let session_dir = source.session_dir.as_deref().filter(|directory| !directory.is_empty());
            let sessions = if current {
                crate::core::session_manager::SessionManager::list(&source.cwd, session_dir, None).await
            } else {
                crate::core::session_manager::SessionManager::list_all(None, session_dir).await
            };
            sessions.into_iter().map(saved_session_info).collect()
        })
    }

    fn get_queue(&self) -> BoxFuture<Result<AgentConnectionQueueState, String>> {
        let queue = self.runtime_host.session_queue();
        Box::pin(async move { Ok(queue) })
    }

    fn mutate_queued_message(
        &self,
        lane: &str,
        index: i64,
        expected_text: &str,
        mutation: Value,
    ) -> BoxFuture<Result<String, String>> {
        let status = self
            .runtime_host
            .session_mutate_queued_message(lane, index, expected_text, mutation);
        Box::pin(async move { Ok(status) })
    }

    fn clear_queue(&self) -> BoxFuture<Result<AgentConnectionQueueState, String>> {
        let queue = self.runtime_host.session_clear_queue();
        Box::pin(async move { Ok(queue) })
    }

    fn abort_and_clear_queue(&self) -> BoxFuture<Result<AgentConnectionQueueState, String>> {
        let queue = self.runtime_host.session_clear_queue();
        self.runtime_host.session_request_abort();
        Box::pin(async move { Ok(queue) })
    }

    fn acquire_session_input_pause(
        &self,
        lease_key: &str,
    ) -> BoxFuture<Result<AgentConnectionSessionInputPause, String>> {
        let existing = self.session_input_pauses.lock().unwrap().get(lease_key).cloned();
        if let Some(existing) = existing {
            return Box::pin(async move { Ok(existing) });
        }
        let pause = self.runtime_host.session_acquire_input_pause();
        self.session_input_pauses
            .lock()
            .unwrap()
            .insert(lease_key.to_string(), pause.clone());
        Box::pin(async move { Ok(pause) })
    }

    fn list_cron_jobs(&self, _include_inactive: bool) -> BoxFuture<Result<Vec<Value>, String>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn heartbeat_catalog_supported(&self) -> Option<bool> { Some(false) }

    fn list_heartbeats(&self) -> BoxFuture<Result<Vec<AgentConnectionHeartbeat>, String>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn manage_heartbeat(
        &self,
        _active_session_id: &str,
        _job_id: &str,
        _action: Value,
    ) -> BoxFuture<Result<Value, String>> {
        Box::pin(async { Err("Heartbeats require daemon mode".to_string()) })
    }

    fn add_cron_job(&self, _schedule: &str, _prompt: &str) -> BoxFuture<Result<Value, String>> {
        Box::pin(async { Err("Cron jobs require daemon mode".to_string()) })
    }

    fn cancel_cron_job(&self, _job_id: &str) -> BoxFuture<Result<Value, String>> {
        Box::pin(async { Err("Cron jobs require daemon mode".to_string()) })
    }

    fn get_heartbeat(&self) -> BoxFuture<Result<Option<Value>, String>> {
        Box::pin(async { Ok(None) })
    }

    fn set_heartbeat(
        &self,
        _schedule: &str,
        _instruction: &str,
        _delivery_mode: Option<&str>,
    ) -> BoxFuture<Result<Value, String>> {
        Box::pin(async { Err("Heartbeats require daemon mode".to_string()) })
    }

    fn update_heartbeat(&self, _action: Value) -> BoxFuture<Result<Option<Value>, String>> {
        Box::pin(async { Err("Heartbeats require daemon mode".to_string()) })
    }

    fn send_agent_message(&self, _target_active_session_id: &str, _message: &str) -> BoxFuture<Result<Value, String>> {
        Box::pin(async { Err("Agent messaging requires daemon mode".to_string()) })
    }

    fn get_agent_message_status(&self) -> BoxFuture<Result<Value, String>> {
        Box::pin(async { Err("Agent messaging requires daemon mode".to_string()) })
    }

    fn pause_agent_messages(&self) -> BoxFuture<Result<Value, String>> {
        Box::pin(async { Err("Agent messaging requires daemon mode".to_string()) })
    }

    fn resume_agent_messages(&self) -> BoxFuture<Result<Value, String>> {
        Box::pin(async { Err("Agent messaging requires daemon mode".to_string()) })
    }

    fn clear_agent_messages(&self) -> BoxFuture<Result<f64, String>> {
        Box::pin(async { Err("Agent messaging requires daemon mode".to_string()) })
    }

    fn get_user_messages_for_forking(&self) -> BoxFuture<Result<Vec<AgentConnectionUserMessage>, String>> {
        let messages = self.runtime_host.session_get_user_messages_for_forking();
        Box::pin(async move { Ok(messages) })
    }

    fn get_last_assistant_text(&self) -> BoxFuture<Result<Option<String>, String>> {
        let text = self.runtime_host.session_last_assistant_text();
        Box::pin(async move { Ok(text) })
    }

    fn get_system_prompt(&self) -> BoxFuture<Result<String, String>> {
        let prompt = self.runtime_host.session_system_prompt();
        Box::pin(async move { Ok(prompt) })
    }

    fn get_tool_definition(&self, name: &str) -> BoxFuture<Result<Option<AgentConnectionToolDefinition>, String>> {
        let definition = create_agent_connection_tool_definition(self.runtime_host.session_tool_definition(name).as_ref());
        Box::pin(async move { Ok(definition) })
    }

    fn set_session_entry_label(&self, entry_id: &str, label: Option<&str>) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_append_label_change(entry_id, label);
        Box::pin(async { Ok(()) })
    }

    fn respond_to_extension_ui_request(
        &self,
        _request_id: &str,
        _response: AgentConnectionExtensionUiResponse,
    ) -> BoxFuture<Result<(), String>> {
        // In-process extension UI requests are handled directly by InteractiveMode.
        Box::pin(async { Ok(()) })
    }

    fn prompt(&self, message: &str, options: Option<AgentConnectionPromptOptions>) -> BoxFuture<Result<(), String>> {
        let mut payload = serde_json::Map::new();
        if let Some(options) = &options {
            if let Some(images) = &options.images {
                payload.insert("images".to_string(), serde_json::to_value(images).unwrap_or(Value::Null));
            }
            if let Some(streaming_behavior) = &options.streaming_behavior {
                payload.insert("streamingBehavior".to_string(), Value::String(streaming_behavior.clone()));
                payload.insert("resumeIfIdle".to_string(), Value::Bool(true));
            }
            if let Some(queue_if_busy) = options.queue_if_busy {
                payload.insert("queueIfBusy".to_string(), Value::Bool(queue_if_busy));
            }
            if let Some(source) = &options.source {
                payload.insert("source".to_string(), Value::String(source.clone()));
            }
        }
        self.runtime_host.session_prompt(message, Value::Object(payload))
    }

    fn prompt_and_wait(
        &self,
        message: &str,
        options: Option<AgentConnectionPromptOptions>,
    ) -> BoxFuture<Result<(), String>> {
        let mut payload = serde_json::Map::new();
        if let Some(options) = &options {
            if let Some(images) = &options.images {
                payload.insert("images".to_string(), serde_json::to_value(images).unwrap_or(Value::Null));
            }
            if let Some(streaming_behavior) = &options.streaming_behavior {
                payload.insert("streamingBehavior".to_string(), Value::String(streaming_behavior.clone()));
                payload.insert("resumeIfIdle".to_string(), Value::Bool(true));
            }
            if let Some(queue_if_busy) = options.queue_if_busy {
                payload.insert("queueIfBusy".to_string(), Value::Bool(queue_if_busy));
            }
            if let Some(source) = &options.source {
                payload.insert("source".to_string(), Value::String(source.clone()));
            }
        }
        self.runtime_host.session_prompt_and_wait(message, Value::Object(payload))
    }

    fn start_side_question(
        &self,
        id: &str,
        question: &str,
        previous_turns: Option<Vec<AgentConnectionSideQuestionTurn>>,
    ) -> BoxFuture<Result<(), String>> {
        let mut runs = self.side_question_runs.lock().unwrap();
        if runs.contains_key(id) {
            let message = format!("Side question already exists: {id}");
            return Box::pin(async move { Err(message) });
        }
        let listeners = self.listeners.clone();
        let run = self.runtime_host.session_start_side_question(id.into(), question.into(),
            previous_turns.map(|turns| turns.into_iter().map(|t| crate::core::side_question::SideQuestionTurn { question: t.question, answer: t.answer }).collect()),
            Arc::new(move |event| emit_to_listeners(&listeners, AgentConnectionEvent::SideQuestionEvent { event: AgentConnectionSideQuestionEvent {
                id: event.id, question: event.question, answer: event.answer, status: event.status, error_message: event.error_message,
            } })),
        );
        let run = match run { Ok(run) => run, Err(error) => return Box::pin(async move { Err(error) }) };
        runs.insert(id.to_string(), run.abort);
        drop(runs);
        let runs = self.side_question_runs.clone(); let id = id.to_string();
        tokio::spawn(async move { run.done.await; runs.lock().unwrap().remove(&id); });
        Box::pin(async { Ok(()) })
    }

    fn abort_side_question(&self, id: &str) -> BoxFuture<Result<bool, String>> {
        let removed = self.side_question_runs.lock().unwrap().remove(id);
        match removed {
            Some(run) => {
                run();
                Box::pin(async { Ok(true) })
            }
            None => Box::pin(async { Ok(false) }),
        }
    }

    fn steer(&self, message: &str, images: Option<Vec<ImageContent>>) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_steer(message, images)
    }

    fn follow_up(&self, message: &str, images: Option<Vec<ImageContent>>) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_follow_up(message, images)
    }

    fn abort(&self) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_request_abort();
        Box::pin(async { Ok(()) })
    }

    fn cancel_rlm_child(&self, child_id: &str) -> BoxFuture<Result<bool, String>> {
        let cancelled = self.runtime_host.session_cancel_rlm_child(child_id);
        Box::pin(async move { Ok(cancelled) })
    }

    fn wait_for_idle(&self) -> BoxFuture<Result<(), String>> {
        let idle = self.runtime_host.session_wait_for_idle();
        Box::pin(async move {
            idle.await;
            Ok(())
        })
    }

    fn wait_for_headless_completion(
        &self,
        options: Option<AgentConnectionHeadlessCompletionOptions>,
    ) -> BoxFuture<Result<AgentAutonomousStatus, String>> {
        self.runtime_host.session_wait_for_headless_completion(options)
    }

    fn execute_bash(
        &self,
        command: &str,
        options: Option<AgentConnectionExecuteBashOptions>,
    ) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_run_user_bash(command, options)
    }

    fn execute_bash_and_wait(&self, command: &str) -> BoxFuture<Result<Value, String>> {
        self.runtime_host.session_execute_bash(command)
    }

    fn abort_bash(&self) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_abort_bash();
        Box::pin(async { Ok(()) })
    }

    fn set_model(&self, provider: &str, model_id: &str) -> BoxFuture<Result<AgentConnectionModel, String>> {
        let registry_models = self.runtime_host.session_model_registry_available_models();
        let auth_source = self.runtime_host.session_model_registry_provider_auth_source(provider);
        let stale_model = if auth_source == "stale" {
            self.runtime_host.session_model_registry_find(provider, model_id)
        } else {
            None
        };
        let host = self.runtime_host.clone();
        let provider = provider.to_string();
        let model_id = model_id.to_string();
        Box::pin(async move {
            let available_models = registry_models.await;
            let model = available_models
                .into_iter()
                .find(|candidate| candidate.provider == provider && candidate.id == model_id)
                // Stale-auth providers are excluded from the available list; the lookup
                // never mutates stale state (session.setModel owns the clear).
                .or(stale_model);
            let model = match model {
                Some(model) => model,
                None => return Err(format!("Model not found: {provider}/{model_id}")),
            };
            host.session_set_model(model.clone()).await?;
            Ok(model)
        })
    }

    fn cycle_model(
        &self,
        direction: Option<&str>,
    ) -> BoxFuture<Result<Option<AgentConnectionModelCycleResult>, String>> {
        self.runtime_host.session_cycle_model(direction.unwrap_or("forward"))
    }

    fn set_scoped_models(&self, scoped_models: Vec<AgentConnectionScopedModel>) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_set_scoped_models(scoped_models);
        Box::pin(async { Ok(()) })
    }

    fn set_thinking_level(&self, level: ThinkingLevel) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_set_thinking_level(level);
        Box::pin(async { Ok(()) })
    }

    fn set_service_tier(&self, service_tier: ServiceTier) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_set_service_tier(service_tier);
        Box::pin(async { Ok(()) })
    }

    fn cycle_thinking_level(&self) -> BoxFuture<Result<Option<ThinkingLevel>, String>> {
        let level = self.runtime_host.session_cycle_thinking_level();
        Box::pin(async move { Ok(level) })
    }

    fn set_transport(&self, transport: Transport) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_set_transport(transport);
        Box::pin(async { Ok(()) })
    }

    fn set_steering_mode(&self, mode: &str) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_set_steering_mode(mode);
        Box::pin(async { Ok(()) })
    }

    fn set_follow_up_mode(&self, mode: &str) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_set_follow_up_mode(mode);
        Box::pin(async { Ok(()) })
    }

    fn set_auto_compaction_enabled(&self, enabled: bool) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_set_auto_compaction_enabled(enabled);
        Box::pin(async { Ok(()) })
    }

    fn set_auto_retry_enabled(&self, enabled: bool) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_set_auto_retry_enabled(enabled);
        Box::pin(async { Ok(()) })
    }

    fn compact(&self, custom_instructions: Option<&str>) -> BoxFuture<Result<Value, String>> {
        self.runtime_host.session_compact(custom_instructions)
    }

    fn refine(&self, options: Value) -> BoxFuture<Result<Value, String>> {
        self.runtime_host.session_refine(options)
    }

    fn abort_compaction(&self) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_abort_compaction();
        Box::pin(async { Ok(()) })
    }

    fn abort_branch_summary(&self) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_abort_branch_summary();
        Box::pin(async { Ok(()) })
    }

    fn abort_retry(&self) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_abort_retry();
        Box::pin(async { Ok(()) })
    }

    fn reload(&self) -> BoxFuture<Result<(), String>> {
        self.runtime_host.session_reload()
    }

    fn new_session(&self, options: Option<AgentConnectionNewSessionOptions>) -> BoxFuture<Result<bool, String>> {
        self.runtime_host.runtime_new_session(options)
    }

    fn switch_session(
        &self,
        session_path: &str,
        options: Option<AgentConnectionSwitchSessionOptions>,
    ) -> BoxFuture<Result<bool, String>> {
        self.runtime_host.runtime_switch_session(session_path, options)
    }

    fn fork(
        &self,
        entry_id: &str,
        options: Option<AgentConnectionForkOptions>,
    ) -> BoxFuture<Result<Value, String>> {
        self.runtime_host.runtime_fork(entry_id, options)
    }

    fn navigate_tree(
        &self,
        target_id: &str,
        options: Option<AgentConnectionNavigateTreeOptions>,
    ) -> BoxFuture<Result<AgentConnectionNavigateTreeResult, String>> {
        self.runtime_host.session_navigate_tree(target_id, options)
    }

    fn import_from_jsonl(&self, input_path: &str, cwd_override: Option<&str>) -> BoxFuture<Result<bool, String>> {
        self.runtime_host.runtime_import_from_jsonl(input_path, cwd_override)
    }

    fn export_to_html(&self, output_path: Option<&str>) -> BoxFuture<Result<String, String>> {
        self.runtime_host.session_export_to_html(output_path)
    }

    fn export_to_jsonl(&self, output_path: Option<&str>) -> BoxFuture<Result<String, String>> {
        self.runtime_host.session_export_to_jsonl(output_path)
    }

    fn set_session_name(&self, name: &str) -> BoxFuture<Result<(), String>> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Box::pin(async { Err("Session name cannot be empty".to_string()) });
        }
        self.runtime_host.session_set_session_name(trimmed);
        Box::pin(async { Ok(()) })
    }

    fn get_rlm_max_depth_status(&self) -> BoxFuture<Result<Value, String>> {
        let status = self.runtime_host.session_get_rlm_max_depth_status();
        Box::pin(async move { Ok(status) })
    }

    fn set_rlm_max_depth(&self, max_depth: f64, options: Option<Value>) -> BoxFuture<Result<Value, String>> {
        self.runtime_host.session_set_rlm_max_depth(max_depth, options)
    }

    fn rename_saved_session(&self, session_path: &str, name: &str) -> BoxFuture<Result<(), String>> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Box::pin(async { Err("Session name cannot be empty".to_string()) });
        }
        let current_session_file = self.runtime_host.snapshot_source().session.session_file;
        if let Some(current) = current_session_file {
            if same_path(&current, session_path) {
                self.runtime_host.session_set_session_name(trimmed);
                return Box::pin(async { Ok(()) });
            }
        }
        let session_path = session_path.to_string();
        let name = trimmed.to_string();
        Box::pin(async move {
            crate::core::session_manager::SessionManager::open(&session_path, None, None)?
                .append_session_info(&name)?;
            Ok(())
        })
    }

    fn delete_saved_session(&self, session_path: &str) -> BoxFuture<Result<Value, String>> {
        let session_path = session_path.to_string();
        Box::pin(async move {
            use crate::core::session_file_actions::{
                delete_session_file, DeleteSessionFileMethod, DeleteSessionFileOptions, DeleteSessionFileResult,
            };
            let result = delete_session_file(&session_path, &mut DeleteSessionFileOptions::default());
            Ok(match result {
                DeleteSessionFileResult::Ok { method } => serde_json::json!({
                    "ok": true,
                    "method": match method { DeleteSessionFileMethod::Trash => "trash", DeleteSessionFileMethod::Unlink => "unlink" },
                }),
                DeleteSessionFileResult::Error { error } => serde_json::json!({ "ok": false, "error": error }),
            })
        })
    }

    fn watch_session(
        &self,
        child_id: &str,
    ) -> BoxFuture<Result<Option<Box<dyn AgentConnectionSessionWatcher>>, String>> {
        let watcher = self.runtime_host.session_watch_child(child_id);
        Box::pin(async move { Ok(watcher) })
    }

    fn dispose(&self) -> BoxFuture<Result<(), String>> {
        self.abort_all_side_questions();
        let pauses: Vec<AgentConnectionSessionInputPause> = self
            .session_input_pauses
            .lock()
            .unwrap()
            .drain()
            .map(|(_, pause)| pause)
            .collect();
        let unsubscribe = self.unsubscribe_session_events.lock().unwrap().take();
        self.runtime_host.runtime_set_before_session_invalidate(None);
        self.runtime_host.runtime_set_rebind_session(None);
        let dispose = self.runtime_host.runtime_dispose();
        Box::pin(async move {
            for pause in pauses {
                let _ = pause.release().await;
            }
            if let Some(unsubscribe) = unsubscribe {
                unsubscribe();
            }
            dispose.await;
            Ok(())
        })
    }
}

/// Deliver one event to the current listener roster.
///
/// `emit(event)` (in-process-agent-connection.ts:651) is `void`-returning in the
/// TypeScript, so the extension-error listener hands the future to the runtime.
fn emit_to_listeners(
    listeners: &Arc<Mutex<Vec<AgentConnectionEventListener>>>,
    event: AgentConnectionEvent,
) -> BoxFuture<()> {
    let listeners: Vec<AgentConnectionEventListener> = listeners.lock().unwrap().clone();
    Box::pin(async move {
        let mut deliveries = Vec::with_capacity(listeners.len());
        for listener in listeners {
            deliveries.push(listener(event.clone()));
        }
        for delivery in deliveries {
            let _ = delivery.await;
        }
    })
}

fn saved_session_info(session: crate::core::session_manager::SessionInfo) -> Result<AgentConnectionSavedSessionInfo, String> {
    fn project<T: serde::Serialize, U: serde::de::DeserializeOwned>(value: T) -> Result<U, String> {
        serde_json::to_value(value)
            .and_then(serde_json::from_value)
            .map_err(|error| error.to_string())
    }
    Ok(AgentConnectionSavedSessionInfo {
        path: session.path,
        id: session.id,
        cwd: session.cwd,
        name: session.name,
        state: session.state.map(project).transpose()?,
        parent_session_path: session.parent_session_path,
        rlm_depth: Some(session.rlm_depth as f64),
        created: session.created,
        modified: session.modified,
        message_count: session.message_count as f64,
        first_message: session.first_message,
        all_messages_text: session.all_messages_text,
        agent_status: session.agent_status.map(project).transpose()?,
        usage: session.usage.map(serde_json::to_value).transpose().map_err(|error| error.to_string())?,
    })
}

fn same_path(left: &str, right: &str) -> bool {
    let left_path = std::fs::canonicalize(left).unwrap_or_else(|_| std::path::PathBuf::from(left));
    let right_path = std::fs::canonicalize(right).unwrap_or_else(|_| std::path::PathBuf::from(right));
    left_path == right_path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_path_compares_canonical_paths() {
        let dir = std::env::temp_dir();
        assert!(same_path(&dir.to_string_lossy(), &dir.to_string_lossy()));
    }
}
