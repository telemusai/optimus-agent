//! Live daemon session and runtime adapters.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

use futures::future::BoxFuture;
use std::sync::Mutex;
use serde_json::{json, Map, Value};

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::ServiceTier;

use crate::core::agent_session::{
    AgentSession, ModelSelectOptions, PromptOptions, RlmChildAgentStatus, RlmMaxDepthStatus,
};
use crate::core::cron_jobs::AgentCronJob;
use crate::core::session_manager::SessionManager;
use crate::core::settings_manager::SettingsManager;
use crate::core::slash_commands::parse_slash_command;
use crate::modes::agent_connection::in_process_agent_connection::InProcessRuntimeHost;
use crate::modes::agent_connection::snapshot::{self, AgentsFileEntry, RegisteredCommandEntry, PromptTemplateEntry, SkillEntry, ResourceSkillEntry, ResourcePromptEntry, ResourceExtensionEntry, ResourceThemeEntry, ExtensionLoadError};
use crate::modes::daemon::daemon_extension_binding::ExtensionBindingInput;
use crate::modes::daemon::daemon_mode::{
    DaemonConnectionView, DaemonRuntimeApi, DaemonSession, ForkOptions, HeadlessCompletionOptions,
    ModelIdentity, NavigateTreeOptions, NewSessionRuntimeOptions, PromptInvocation, RefineOptions,
    RunUserBashOptions, SessionInputPause as DaemonSessionInputPause, SessionPathOptions,
    SideQuestionOptions,
};

/// `DaemonSession` over a live `AgentSession`.
///
/// The daemon keeps one of these per resident session, so the seam's members
/// forward to the `AgentSession` methods the TypeScript reads off
/// `state.runtime.session`.
pub struct AgentSessionDaemonAdapter {
    runtime: Arc<super::AgentSessionRuntime>,
    side_questions: Arc<Mutex<std::collections::HashMap<String, Arc<dyn Fn() + Send + Sync>>>>,
}

impl AgentSessionDaemonAdapter {
    pub fn new(runtime: Arc<super::AgentSessionRuntime>) -> Self {
        Self { runtime, side_questions: Arc::new(Mutex::new(std::collections::HashMap::new())) }
    }

    /// The wrapped session, for callers that already hold the adapter.
    pub fn session(&self) -> Arc<AgentSession> {
        self.runtime.session()
    }
}

/// `PromptInvocation` -> the session's `PromptOptions`.
///
/// The daemon seam carries the invocation as a struct of JSON-ish fields
/// (`daemon_mode.rs:1932`); the session owner takes `PromptOptions`. The signal,
/// admission callbacks, content and custom message are mapped too, so the forward is not lossy.
fn prompt_options_from_invocation(invocation: PromptInvocation) -> PromptOptions {
    let mut options = PromptOptions {
        expand_prompt_templates: invocation.expand_prompt_templates,
        streaming_behavior: invocation.streaming_behavior,
        queue_if_busy: invocation.queue_if_busy,
        resume_if_idle: invocation.resume_if_idle,
        skip_input_handlers: invocation.skip_input_handlers,
        agent_message_id: invocation.agent_message_id,
        signal: invocation.signal,
        admission_committed: invocation.admission_committed,
        preflight_result: invocation.preflight_result,
        // ^ C-02: the session reports `(succeeded, queued)`; forwarding only the first
        // flag made the daemon's receipt mapping fall back to "delivered" for a queued
        // admission. `PromptOptions::preflight_result` already has the two-flag shape
        // (`agent_session.rs:770`), so no mapping is needed.
        ..Default::default()
    };
    if let Some(images) = &invocation.images {
        options.images = serde_json::from_value(images.clone()).ok();
    }
    if let Some(content) = &invocation.content {
        options.content = serde_json::from_value(content.clone()).ok();
    }
    if let Some(custom_message) = &invocation.custom_message {
        options.custom_message = serde_json::from_value(custom_message.clone()).ok();
    }
    if let Some(source) = invocation.source.as_deref() {
        options.source = match source {
            "interactive" => Some(crate::core::session_action_store::InputSource::Interactive),
            "rpc" => Some(crate::core::session_action_store::InputSource::Rpc),
            "extension" => Some(crate::core::session_action_store::InputSource::Extension),
            _ => None,
        };
    }
    options
}

/// `session.thinkingLevel` as the daemon wire carries it.
fn thinking_level_value(level: ThinkingLevel) -> String {
    level.as_str().to_string()
}

/// `session.serviceTier` as the daemon wire carries it.
fn service_tier_value(service_tier: &ServiceTier) -> Option<String> {
    service_tier.clone().flatten()
}


/// `BashResult` as the daemon wire carries it (`BashResult` is not serde).
fn bash_result_value(result: &crate::core::bash_executor::BashResult) -> Value {
    let mut object = Map::new();
    object.insert("output".to_string(), Value::String(result.output.clone()));
    object.insert(
        "exitCode".to_string(),
        match result.exit_code {
            Some(code) => Value::from(code),
            None => Value::Null,
        },
    );
    object.insert("cancelled".to_string(), Value::Bool(result.cancelled));
    object.insert("truncated".to_string(), Value::Bool(result.truncated));
    if let Some(path) = &result.full_output_path {
        object.insert("fullOutputPath".to_string(), Value::String(path.clone()));
    }
    Value::Object(object)
}

/// `Model<Api>` as the daemon seam names it. `ModelIdentity` is a re-export of the canonical
/// `pi_ai::types::Model` (daemon_mode.rs:850), not a narrower wire view, so this is the identity:
/// the daemon reads `model.provider` / `model.id` off the same object the registry holds.
fn model_identity_of(model: pi_ai::types::Model) -> ModelIdentity {
    model
}

/// `buildSessionContext()` as the daemon's `get_session_context` wire expects
/// (`session-manager.ts` `SessionContext`).
fn session_context_value(context: &crate::core::session_manager::SessionContext) -> Value {
    let mut object = Map::new();
    object.insert(
        "messages".to_string(),
        serde_json::to_value(&context.messages).unwrap_or(Value::Null),
    );
    object.insert(
        "thinkingLevel".to_string(),
        Value::String(context.thinking_level.clone()),
    );
    object.insert(
        "serviceTier".to_string(),
        serde_json::to_value(&context.service_tier).unwrap_or(Value::Null),
    );
    object.insert(
        "model".to_string(),
        match &context.model {
            Some(model) => json!({ "provider": model.provider, "modelId": model.model_id }),
            None => Value::Null,
        },
    );
    Value::Object(object)
}

/// `thinkingLevel` as `ThinkingLevel`; an unknown string falls back to `Off`,
/// which is the same member the daemon wire emits for "off".
fn parse_thinking_level(level: &str) -> ThinkingLevel {
    match level {
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::Xhigh,
        "max" => ThinkingLevel::Max,
        _ => ThinkingLevel::Off,
    }
}

/// `ServiceTier` is `string | null` on the wire.
fn parse_service_tier(service_tier: &str) -> ServiceTier {
    Some(Some(service_tier.to_string()))
}

/// `getToolDefinition(name)` as the daemon serializes it (the same seven-field
/// projection `create_agent_connection_tool_definition` uses).
fn tool_definition_value(definition: &crate::core::extensions::types::ToolDefinition) -> Value {
    json!({
        "name": definition.name,
        "label": definition.label,
        "description": definition.description,
        "promptSnippet": definition.prompt_snippet,
        "promptGuidelines": definition.prompt_guidelines,
        "parameters": definition.parameters,
        "renderShell": definition.render_shell,
        "replayBuiltInToolName": definition.replay_built_in_tool_name,
    })
}

impl DaemonSession for AgentSessionDaemonAdapter {
    fn agent_session(&self) -> Option<Arc<AgentSession>> { Some(self.session()) }
    /// `session.model` as the daemon's `ModelIdentity` carries it.
    fn model_identity(&self) -> Option<ModelIdentity> {
        self.runtime.session().model().map(model_identity_of)
    }

    /// `session.modelRegistry.find(provider, modelId)`.
    fn find_model(&self, provider: &str, model_id: &str) -> Option<ModelIdentity> {
        let registry = self.runtime.session().model_registry();
        let registry = registry.lock().unwrap();
        registry.find(provider, model_id).map(model_identity_of)
    }

    /// `session.modelRegistry.getProviderAuthStatus(provider).source`.
    fn get_provider_auth_status_source(&self, provider: &str) -> Option<String> {
        let registry = self.runtime.session().model_registry();
        let registry = registry.lock().unwrap();
        registry.get_provider_auth_status(provider).source
    }

    fn session_manager(&self) -> Arc<StdMutex<SessionManager>> {
        Arc::clone(&self.runtime.session().session_manager)
    }

    fn session_id(&self) -> String {
        self.runtime.session().session_id()
    }

    fn session_name(&self) -> Option<String> {
        self.runtime.session().session_name()
    }

    fn session_file(&self) -> Option<String> {
        self.runtime.session().session_file()
    }

    fn is_streaming(&self) -> bool {
        self.runtime.session().is_streaming()
    }

    fn is_compacting(&self) -> bool {
        self.runtime.session().is_compacting()
    }

    fn is_bash_running(&self) -> bool {
        self.runtime.session().is_bash_running()
    }

    fn is_retrying(&self) -> bool {
        self.runtime.session().is_retrying()
    }

    fn is_session_active(&self) -> bool {
        self.runtime.session().is_session_active()
    }

    fn is_foreground_active(&self) -> bool {
        self.runtime.session().is_foreground_active()
    }

    fn has_running_rlm_children(&self) -> bool {
        self.runtime.session().has_running_rlm_children()
    }

    /// `session.unfinishedActionCount` (`_queuedActionCount() + in-flight`).
    fn unfinished_action_count(&self) -> f64 {
        self.runtime.session().unfinished_action_count() as f64
    }

    fn messages(&self) -> Vec<AgentMessage> {
        self.runtime.session().messages()
    }

    fn message_count(&self) -> usize {
        self.runtime.session().message_count()
    }

    /// `session.rlmDepth ?? null` - `None` is `null` on the wire.
    fn rlm_depth(&self) -> Option<i64> {
        Some(self.runtime.session().rlm_depth())
    }

    fn thinking_level(&self) -> Option<String> {
        Some(thinking_level_value(self.runtime.session().thinking_level()))
    }

    fn service_tier(&self) -> Option<String> {
        service_tier_value(&self.runtime.session().service_tier())
    }

    fn system_prompt(&self) -> Option<String> {
        Some(self.runtime.session().system_prompt())
    }

    fn set_current_recap(&self, recap: Option<&str>) {
        self.runtime.session()
            .set_current_recap(recap.map(|recap| recap.to_string()));
    }

    fn set_session_name(&self, name: &str) {
        // The daemon seam returns unit; `setSessionName` reports a failure, so the
        // error is logged rather than dropped silently.
        if let Err(error) = self.runtime.session().set_session_name(name) {
            eprintln!("Warning: Could not set session name: {error}");
        }
    }

    /// `session.getRlmChildRunStatus(childId)` - `RlmChildAgentStatus` is already
    /// the wire string the daemon uses.
    fn get_rlm_child_run_status(&self, child_id: &str) -> Option<String> {
        let status: RlmChildAgentStatus = self.runtime.session().get_rlm_child_run_status(child_id)?;
        Some(status)
    }

    fn remove_queued_follow_up(&self, key: &str) {
        self.runtime.session().remove_queued_follow_up(key);
    }

    /// `session.runUserBash(command, options)`.
    fn run_user_bash(
        &self,
        command: &str,
        options: RunUserBashOptions,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let session = Arc::clone(&self.runtime.session());
        let command = command.to_string();
        // `transient`/`runId` are daemon-side bash-event identity fields; the
        // session's `runUserBash(command, excludeFromContext)` does not accept them.
        Box::pin(async move {
            let result = session
                .run_user_bash(&command, options.exclude_from_context)
                .await?;
            Ok(bash_result_value(&result))
        })
    }

    /// `session.executeBash(command)`.
    fn execute_bash(&self, command: &str) -> BoxFuture<'static, Result<Value, String>> {
        let session = Arc::clone(&self.runtime.session());
        let command = command.to_string();
        Box::pin(async move {
            let result = session.execute_bash(&command, None, None).await?;
            Ok(bash_result_value(&result))
        })
    }

    fn request_abort(&self) {
        self.runtime.session().request_abort();
    }

    fn abort_bash(&self) {
        self.runtime.session().abort_bash();
    }

    /// The daemon seam carries the pause as a bare release closure; the session
    /// returns the `SessionInputPause` struct, so its `release` is wrapped.
    fn acquire_session_input_pause(&self) -> DaemonSessionInputPause {
        let pause = self.runtime.session().acquire_session_input_pause();
        Arc::new(move || pause.release())
    }

    fn set_steering_mode(&self, mode: &str) {
        self.runtime.session().set_steering_mode(mode);
    }

    fn set_follow_up_mode(&self, mode: &str) {
        self.runtime.session().set_follow_up_mode(mode);
    }

    fn set_auto_compaction_enabled(&self, enabled: bool) {
        self.runtime.session().set_auto_compaction_enabled(enabled);
    }

    fn set_auto_retry_enabled(&self, enabled: bool) {
        self.runtime.session().set_auto_retry_enabled(enabled);
    }

    fn abort_compaction(&self) {
        self.runtime.session().abort_compaction();
    }

    fn abort_branch_summary(&self) {
        self.runtime.session().abort_branch_summary();
    }

    fn abort_retry(&self) {
        self.runtime.session().abort_retry();
    }

    fn replied_to_parent_since_task(&self) -> Option<bool> {
        self.runtime.session().replied_to_parent_since_task()
    }

    fn cycle_thinking_level(&self) -> Option<String> {
        self.runtime.session().cycle_thinking_level().map(thinking_level_value)
    }

    fn get_rlm_max_depth_status(&self) -> Value {
        let status: RlmMaxDepthStatus = self.runtime.session().get_rlm_max_depth_status();
        serde_json::to_value(status).unwrap_or(Value::Null)
    }

    fn build_session_context(&self) -> Value {
        session_context_value(&self.runtime.session().build_session_context())
    }

    fn get_session_stats(&self) -> Value {
        serde_json::to_value(self.runtime.session().get_session_stats()).unwrap_or(Value::Null)
    }

    fn get_user_messages_for_forking(&self) -> Vec<Value> {
        self.runtime.session()
            .get_user_messages_for_forking()
            .into_iter()
            .map(|entry| {
                let mut object = Map::new();
                object.insert("entryId".to_string(), Value::String(entry.entry_id));
                object.insert("text".to_string(), Value::String(entry.text));
                Value::Object(object)
            })
            .collect()
    }

    fn get_tool_definition(&self, name: &str) -> Option<Value> {
        self.runtime.session()
            .get_tool_definition(name)
            .map(|definition| tool_definition_value(&definition))
    }

    fn wait_for_idle(&self) -> BoxFuture<'static, ()> {
        let session = Arc::clone(&self.runtime.session());
        Box::pin(async move {
            // The daemon seam reports nothing; `waitForIdle` rejects on a session
            // failure, so the error is logged.
            if let Err(error) = session.wait_for_idle().await {
                eprintln!("Warning: Could not wait for session idle: {error}");
            }
        })
    }

    fn release_acp_mcp_servers(
        &self,
        owner_id: &str,
        server_names: &[String],
    ) -> BoxFuture<'static, Result<(), String>> {
        let session = Arc::clone(&self.runtime.session());
        let owner_id = owner_id.to_string();
        let server_names = server_names.to_vec();
        Box::pin(async move { session.release_acp_mcp_servers(&owner_id, &server_names).await })
    }

    fn replace_acp_mcp_servers(
        &self,
        servers: &[Value],
        owner_id: &str,
    ) -> BoxFuture<'static, Result<(), String>> {
        let session = Arc::clone(&self.runtime.session());
        let owner_id = owner_id.to_string();
        let servers = servers
            .iter()
            .cloned()
            .map(serde_json::from_value)
            .collect::<Result<Vec<_>, _>>();
        Box::pin(async move {
            let servers = servers.map_err(|error| error.to_string())?;
            session.replace_acp_mcp_servers(&servers, &owner_id)
        })
    }

    fn set_thinking_level(&self, level: &str) {
        self.runtime.session().set_thinking_level(parse_thinking_level(level));
    }

    fn set_service_tier(&self, service_tier: &str) {
        self.runtime.session()
            .set_service_tier(parse_service_tier(service_tier));
    }


    // ---------------------------------------------------------------------
    // 34 of the 104 members below forward to a real owner. The rest either
    // forward partially (a real call plus a `blocked_on:` note for the part the
    // port cannot produce) or return an explicit failure / empty value with a
    // `blocked_on:` note naming the owner to add. None are omitted: the trait
    // has no default bodies, so an omitted member is an E0046 that stops the
    // whole crate from compiling and keeps every test from running.
    // ---------------------------------------------------------------------

    fn runtime(&self) -> Arc<dyn DaemonRuntimeApi> { Arc::new(DaemonRuntimeAdapter(self.runtime.clone())) }

    fn settings_manager(&self) -> Option<Arc<StdMutex<SettingsManager>>> { Some(self.runtime.services().settings_manager.clone()) }

    fn session_dir(&self) -> Option<String> { Some(self.session().session_manager.lock().unwrap().get_session_dir()) }

    /// `session.setExecEnvProvider(() => execEnvForSession(state.clientEnv))`.
    ///
    /// `daemon-extension-binding.ts:55` passes the provider built from the
    /// session's client env; `agent-session.ts:9739-9743` assigns it and then
    /// sets `extensions.runtime.getExecEnv = provider`. The loader reads that
    /// slot at exec time (`extensions/loader.rs:316`), which is why the provider
    /// is installed on the loaded runtime's state here.
    fn set_exec_env_provider(&self, client_env: Option<HashMap<String, String>>) {
        let provider: Arc<dyn Fn() -> Option<Map<String, Value>> + Send + Sync> = Arc::new(move || {
            // `execEnvForSession` returns `undefined` for a key the client did not
            // send; the loader maps `Value::Null` back to "unset in the child"
            // (`extensions/loader.rs:431-434`), so `None` becomes `Null` here.
            Some(
                crate::modes::daemon::daemon_client_env::exec_env_for_session(client_env.as_ref())
                    .into_iter()
                    .map(|(key, value)| (key, value.map(Value::String).unwrap_or(Value::Null)))
                    .collect::<Map<String, Value>>(),
            )
        });
        let extensions = self.session().resource_loader().get_extensions();
        extensions
            .runtime
            .state
            .lock()
            .unwrap()
            .get_exec_env = Some(provider);
    }

    /// `state.runtime.setRuntimeEnvScope((fn) => withClientEnv(state.clientEnv, fn))`.
    ///
    /// `daemon-extension-binding.ts:58` wraps every runtime rebuild so extension
    /// load-time captures see the client env; `agent-session-runtime.ts:141`
    /// stores the scope and `:145` `scopedBuild` applies it.
    fn set_runtime_env_scope(&self, client_env: Option<HashMap<String, String>>) {
        let scope: super::RuntimeEnvScope = Arc::new(
            move |build: BoxFuture<Result<super::CreateAgentSessionRuntimeResult, String>>| {
                let client_env = client_env.clone();
                Box::pin(async move {
                    let mut build = Some(build);
                    crate::modes::daemon::daemon_client_env::with_client_env(
                        client_env.as_ref(),
                        move || build.take().expect("the env scope wraps one build"),
                    )
                    .await
                })
            },
        );
        self.runtime.set_runtime_env_scope(Some(scope));
    }

    /// `state.runtime.setSubagentRuntimeHost(callbacks.subagentRuntimeHost)`.
    ///
    /// `daemon-extension-binding.ts:61` always assigns, including `undefined`:
    /// `agent-session-runtime.ts:149-152` then re-binds, and `:548-552` falls
    /// back to the runtime hosting its own subagents. Skipping the `None` case
    /// would leave the previous host installed.
    fn set_subagent_runtime_host(&self, host: Option<Arc<dyn crate::core::rlm_runtime::SubagentRuntimeHost>>) {
        self.runtime.set_subagent_runtime_host(host);
    }

    fn set_rebind_session(&self, rebind: Arc<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>) {
        self.runtime.set_rebind_session(Some(Arc::new(move |_| rebind())));
    }

    fn bind_extensions(&self, binding: ExtensionBindingInput) -> BoxFuture<'static, Result<(), String>> {
        let session = self.session();
        let bindings = crate::core::agent_session::ExtensionBindings {
            ui_context: Some(binding.ui_context),
            command_context_actions: None,
            shutdown_handler: Some(binding.shutdown_handler),
            on_error: Some(Arc::new(move |error| (binding.on_error)(&crate::modes::daemon::daemon_extension_binding::ExtensionBindingError {
                extension_path: Some(error.extension_path), event: Some(error.event), error: error.error,
            }))),
        };
        Box::pin(async move { session.bind_extensions(&bindings).await })
    }

    fn abort_for_update_restart(&self) {
        self.runtime.session().abort_for_update_restart();
    }

    fn connection_view(&self) -> DaemonConnectionView {
        let host = super::InProcessRuntimeHostAdapter::new(self.runtime.clone());
        let mut view = DaemonConnectionView {
            session_id: self.session().session_id(), cwd: self.runtime.cwd(), ..Default::default()
        };
        for command in host.session_commands() {
            match command.source.as_str() {
                "extension" => view.registered_commands.push(RegisteredCommandEntry {
                    invocation_name: command.name.clone(), name: command.registered_name.unwrap_or(command.name),
                    description: command.description, source_info: command.source_info,
                }),
                "prompt" => view.prompt_templates.push(PromptTemplateEntry {
                    name: command.name, description: command.description, argument_hint: command.argument_hint,
                    source_info: command.source_info,
                }),
                "skill" => view.skills.push(SkillEntry {
                    name: command.name.strip_prefix("skill:").unwrap_or(&command.name).to_string(),
                    description: command.description, source_info: command.source_info,
                }),
                _ => unreachable!("the canonical command projection has a closed source set"),
            }
        }
        let resources = host.session_resource_snapshot();
        view.agents_files = resources.context_files.into_iter().map(|entry| AgentsFileEntry { path: entry.path }).collect();
        view.resource_skills = resources.skills.into_iter().map(|entry| ResourceSkillEntry { name: entry.name, description: entry.description, file_path: entry.file_path, source_info: entry.source_info }).collect();
        view.resource_prompts = resources.prompts.into_iter().map(|entry| ResourcePromptEntry { name: entry.name, description: entry.description, argument_hint: entry.argument_hint, file_path: entry.file_path, source_info: entry.source_info }).collect();
        view.resource_extensions = resources.extensions.into_iter().map(|entry| ResourceExtensionEntry { path: entry.path, source_info: entry.source_info }).collect();
        view.resource_themes = resources.themes.into_iter().map(|entry| ResourceThemeEntry { name: entry.name, source_path: entry.source_path, source_info: entry.source_info }).collect();
        view.skill_diagnostics = resources.diagnostics.skills;
        view.prompt_diagnostics = resources.diagnostics.prompts;
        view.theme_diagnostics = resources.diagnostics.themes;
        view.extension_load_errors = resources.diagnostics.extensions.into_iter().map(|entry| ExtensionLoadError { error: entry.message, path: entry.path }).collect();
        view
    }

    fn connection_state(&self, active_session_id: Option<String>) -> Value {
        let host = super::InProcessRuntimeHostAdapter::new(self.runtime.clone());
        serde_json::to_value(snapshot::create_agent_connection_state(&host.snapshot_source(), active_session_id))
            .expect("connection state serializes")
    }

    fn register_rlm_child_session(&self, child_id: &str, session: Arc<dyn DaemonSession>) -> bool {
        session.agent_session().is_some_and(|child| self.session().register_rlm_child_session(child_id, child))
    }

    fn subscribe(&self, listener: Arc<dyn Fn(&Value) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        let unsubscribe = self.session().subscribe(Arc::new(move |event| {
            match serde_json::to_value(event) {
                Ok(event) => listener(&event),
                Err(error) => eprintln!("Could not serialize session event: {error}"),
            }
        }));
        Box::new(move || unsubscribe())
    }

    fn prompt_until_accepted(
        &self,
        message: &str,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let session = Arc::clone(&self.runtime.session());
        let message = message.to_string();
        let options = prompt_options_from_invocation(options);
        Box::pin(async move {
            // Extension commands return lazy Rust futures. Keep polling the command
            // after admission so its dialog can run; the preflight callback above
            // acknowledges the public request before the user answers the dialog.
            let extension_command = options.expand_prompt_templates != Some(false)
                && parse_slash_command(&message).is_some_and(|command| {
                    session.extension_runner().is_some_and(|runner| runner.get_command(&command.name).is_some())
                });
            if extension_command {
                session.prompt(&message, Some(options)).await
            } else {
                session.prompt_until_accepted(&message, Some(options)).await
            }
        })
    }

    fn prompt_and_wait(
        &self,
        message: &str,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let session = Arc::clone(&self.runtime.session());
        let message = message.to_string();
        let options = prompt_options_from_invocation(options);
        Box::pin(async move { session.prompt_and_wait(&message, Some(options)).await })
    }

    fn prompt_heartbeat(
        &self,
        job: &AgentCronJob,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let session = Arc::clone(&self.runtime.session());
        let job = job.clone();
        let options = prompt_options_from_invocation(options);
        Box::pin(async move { session.prompt_heartbeat(&job, Some(options)).await })
    }

    fn accept_agent_message_prompt(
        &self,
        message: &str,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let session = Arc::clone(&self.runtime.session());
        let message = message.to_string();
        let options = prompt_options_from_invocation(options);
        Box::pin(async move {
            session
                .accept_agent_message_prompt(&message, Some(options))
                .await
        })
    }

    fn steer(
        &self,
        message: &str,
        images: Option<Value>,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<(), String>> {
        let session = Arc::clone(&self.runtime.session());
        let message = message.to_string();
        let images = images.and_then(|images| serde_json::from_value(images).ok());
        let options = prompt_options_from_invocation(options);
        Box::pin(async move {
            session
                .steer(
                    &message,
                    images,
                    options.follow_up_queue_key,
                    options.agent_message_id,
                    options.resume_if_idle,
                )
                .await
        })
    }

    fn follow_up(
        &self,
        message: &str,
        images: Option<Value>,
        options: PromptInvocation,
    ) -> BoxFuture<'static, Result<bool, String>> {
        let session = Arc::clone(&self.runtime.session());
        let message = message.to_string();
        let images = images.and_then(|images| serde_json::from_value(images).ok());
        let options = prompt_options_from_invocation(options);
        Box::pin(async move {
            session
                .follow_up(
                    &message,
                    images,
                    options.follow_up_queue_key,
                    options.agent_message_id,
                    options.resume_if_idle,
                )
                .await
        })
    }

    fn restore_steering_message(&self, message: &str, images: Option<Value>, options: PromptInvocation) -> BoxFuture<'static, Result<(), String>> {
        let session = self.session();
        let snapshot = restored_prompt(message, images, options);
        Box::pin(async move { session.restore_steering_message(&snapshot?).await.map(|_| ()) })
    }

    fn restore_follow_up_message(&self, message: &str, images: Option<Value>, options: PromptInvocation) -> BoxFuture<'static, Result<bool, String>> {
        let session = self.session();
        let snapshot = restored_prompt(message, images, options);
        Box::pin(async move { session.restore_follow_up_message(&snapshot?).await })
    }

    fn restore_pending_next_turn_messages(&self, messages: &Value) {
        match serde_json::from_value::<Vec<crate::core::messages::CustomMessage>>(messages.clone()) {
            Ok(messages) => self.session().restore_pending_next_turn_messages(&messages),
            Err(error) => eprintln!("Could not restore pending messages: {error}"),
        }
    }

    fn restore_session_actions(&self, snapshot: &Value) -> BoxFuture<'static, Result<f64, String>> {
        let session = self.session();
        let snapshot = serde_json::from_value(snapshot.clone()).map_err(|error| error.to_string());
        Box::pin(async move { session.restore_session_actions(&snapshot?).await.map(|count| count as f64) })
    }

    fn send_custom_message(&self, message: &Value) -> BoxFuture<'static, Result<(), String>> {
        let session = self.session();
        let message = serde_json::from_value(message.clone()).map_err(|error| error.to_string());
        Box::pin(async move { session.send_custom_message(message?, None, None).await })
    }

    fn resume_queued_work(&self) -> bool {
        // `AgentSession::resumeQueuedWork()` returns `_hasSelectableSessionInput()`
        // AFTER resuming (packages/coding-agent/src/core/agent-session.ts:7516-7521),
        // and `daemon-mode.ts:4633` turns a false result into "No queued work to
        // resume". Reading the pre-resume value here answered with the stale state.
        self.session().resume_stopped_queue()
    }

    fn clear_queued_agent_messages(&self) -> Value {
        let cleared = self.session().clear_queued_agent_messages();
        json!({"steering":cleared.steering,"followUp":cleared.follow_up})
    }

    fn clear_queue(&self) -> Value {
        let cleared = self.session().clear_queue();
        json!({"steering":cleared.steering,"followUp":cleared.follow_up})
    }

    fn mutate_queued_message(&self, lane: &str, index: f64, expected_text: &str, mutation: &Value) -> Value {
        let lane = serde_json::from_value(Value::String(lane.to_string()));
        let mutation = serde_json::from_value(mutation.clone());
        if !index.is_finite() || index.fract() != 0.0 || index < 0.0 || index > i64::MAX as f64 { return json!("invalid"); }
        match (lane, mutation) {
            (Ok(lane), Ok(mutation)) => json!(self.session().mutate_queued_message(lane, index as i64, expected_text, &mutation).as_str()),
            _ => json!("invalid"),
        }
    }

    fn get_steering_message_previews(&self) -> Vec<Value> {
        self.runtime.session()
            .get_steering_message_previews()
            .into_iter()
            .map(Value::String)
            .collect()
    }

    fn get_follow_up_message_previews(&self) -> Vec<Value> {
        self.runtime.session()
            .get_follow_up_message_previews()
            .into_iter()
            .map(Value::String)
            .collect()
    }

    fn cancel_rlm_child_run(&self, child_id: &str) -> bool {
        self.session().cancel_rlm_child_run_by_id(child_id, "Cancelled by user")
    }

    fn delete_inactive_rlm_subagent(
        &self,
        child_id: &str,
        is_resident_child_running: Arc<dyn Fn() -> bool + Send + Sync>,
    ) -> BoxFuture<'static, Result<String, String>> {
        let session = Arc::clone(&self.runtime.session());
        let child_id = child_id.to_string();
        Box::pin(async move {
            session
                .delete_inactive_rlm_subagent(&child_id, is_resident_child_running)
                .await
        })
    }

    fn wait_for_headless_completion(&self, options: HeadlessCompletionOptions) -> BoxFuture<'static, Result<Value, String>> {
        let session = self.session();
        Box::pin(async move {
            let status = crate::modes::headless_completion::wait_for_headless_completion(Arc::new(session),
                crate::modes::headless_completion::HeadlessCompletionOptions { wait_for_rlm_quiescence: options.wait_for_rlm_quiescence }).await?;
            serde_json::to_value(status).map_err(|error| error.to_string())
        })
    }

    fn refresh_available_models(&self) -> BoxFuture<'static, Result<Vec<pi_ai::types::Model>, String>> {
        let registry = self.session().model_registry();
        Box::pin(async move { crate::core::sdk::with_model_registry(registry, |registry| Box::pin(registry.refresh_available_models())).await })
    }

    fn refresh_model_catalog(&self) -> BoxFuture<'static, Result<Value, String>> {
        let registry = self.session().model_registry();
        Box::pin(async move {
            let catalog = crate::core::sdk::with_model_registry(registry, |registry| Box::pin(registry.refresh_model_catalog())).await?;
            Ok(json!({"models":catalog.models,"configuredProviders":catalog.configured_providers}))
        })
    }

    fn set_model(
        &self,
        model: &pi_ai::types::Model,
        wait_for_extensions: bool,
    ) -> BoxFuture<'static, Result<(), String>> {
        let session = Arc::clone(&self.runtime.session());
        let model = model.clone();
        Box::pin(async move {
            session
                .set_model(
                    model,
                    ModelSelectOptions {
                        wait_for_extensions: Some(wait_for_extensions),
                        ..Default::default()
                    },
                )
                .await
        })
    }

    fn cycle_model(&self, direction: &str, wait_for_extensions: bool) -> BoxFuture<'static, Result<Option<pi_ai::types::Model>, String>> {
        let session = self.session();
        let direction = if direction == "backward" { -1 } else { 1 };
        Box::pin(async move { session.cycle_model(Some(direction), ModelSelectOptions { wait_for_extensions: Some(wait_for_extensions) }).await.map(|result| Some(result.model)) })
    }

    /// `session.setScopedModels(command.scopedModels)` (`daemon-mode.ts:5157`).
    ///
    /// blocked_on: the only owner is `AgentSession::set_scoped_models(&mut self,
    /// Vec<ScopedModel>)` (`core/agent_session.rs:6951`), and the session field is
    /// a plain `Vec` held behind `Arc<AgentSession>`, so no `&self` caller can
    /// write it. Owner to add: `core/agent_session.rs`, a `&self` setter over
    /// interior mutability (`Mutex<Vec<ScopedModel>>`), or a `set_scoped_models`
    /// on `AgentSessionRuntime` that rebuilds through `scoped_build`. Needs
    /// `core/agent_session.rs` edited.
    fn set_scoped_models(&self, scoped_models: &Value) {
        let _ = scoped_models;
    }

    /// `state.runtime.session.settingsManager.setTransport(...)` and
    /// `state.runtime.session.agent.transport = ...` (`daemon-mode.ts:5181-5182`).
    ///
    /// The settings half is reachable: this adapter exposes `settings_manager()`.
    /// The live-agent half is not: `AgentSession.agent` is `Arc<dyn AgentHandle>`
    /// (`core/agent_session.rs:2134`) and the `AgentHandle` trait
    /// (`core/agent_session.rs:246-274`) has no transport setter, so
    /// `pi_agent_core::agent::Agent.transport` (`crates/pi-agent-core/src/agent.rs:259`)
    /// cannot be assigned through the seam.
    /// blocked_on: add `fn set_transport(&self, transport: String)` to
    /// `core/agent_session.rs:246` `AgentHandle` (implemented at
    /// `core/agent_session/agent_handle.rs:20` for `Arc<Agent>`, matching
    /// `agent.ts`'s `transport` field). Needs `core/agent_session.rs` and
    /// `agent_handle.rs` edited.
    fn set_transport(&self, transport: &str) {
        if let Some(settings) = self.settings_manager() {
            settings
                .lock()
                .expect("settings manager poisoned")
                .set_transport(transport.to_string());
        }
    }

    /// `await session.compact(command.customInstructions)` uses the same full
    /// manual lifecycle as queued and extension compaction. The internal result
    /// carries `CompactionResult` for extension callbacks; this existing RPC
    /// response remains the unit/null schema.
    fn compact(&self, custom_instructions: Option<&str>) -> BoxFuture<'static, Result<Value, String>> {
        let session = Arc::clone(&self.runtime.session());
        let custom_instructions = custom_instructions.map(|value| value.to_string());
        Box::pin(async move {
            session
                .compact_with_options(custom_instructions.as_deref(), false)
                .await?;
            Ok(Value::Null)
        })
    }

    /// `await session.refine({ instructions, rollbackId, global })`
    /// (`daemon-mode.ts:5218-5222`).
    ///
    /// blocked_on: the only owner is the private
    /// `AgentSession::refine_with_options` (`core/agent_session.rs:12141`), whose
    /// `skip_abort`/`source` internals the TypeScript `refine(options, internal)`
    /// keeps private too; the public path in the port is the `/refine` slash
    /// command (:9571) and `handle_refine_host_request` (:4393), which only
    /// schedules. Owner to add: `core/agent_session.rs`, a
    /// `pub async fn refine(&self, options: RefineOptions) -> Result<RefinementResult, String>`
    /// matching `agent-session.ts:8952`. `RefinementResult`
    /// (`core/refinement/refinement.rs:280`) already derives serde, so the seam
    /// only needs the public path. Needs `core/agent_session.rs` edited.
    fn refine(&self, options: RefineOptions) -> BoxFuture<'static, Result<Value, String>> {
        let _ = options;
        Box::pin(async {
            Err("blocked_on: refine_with_options is private; no public refine path".to_string())
        })
    }

    fn reload(&self) -> BoxFuture<'static, Result<(), String>> {
        let session = self.session();
        Box::pin(async move { session.reload_with_options(None).await })
    }

    fn set_rlm_max_depth(
        &self,
        max_depth: Value,
        global: bool,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let session = Arc::clone(&self.runtime.session());
        let Some(max_depth) = max_depth.as_i64() else {
            // The daemon validates the wire value before it reaches this seam.
            return Box::pin(async { Err("rlmMaxDepth must be a non-negative integer".to_string()) });
        };
        Box::pin(async move {
            session
                .set_rlm_max_depth(max_depth, global)
                .await
                .map(|result| json!(result))
        })
    }

    /// `session.getContextTree()` (`daemon-mode.ts:4909`, `agent-session.ts:13028`).
    ///
    /// blocked_on: every builder the TypeScript calls is private in the port -
    /// `rlm_session_dir_for_reading` (`core/agent_session/runtime_members.rs:3115`),
    /// `context_window_resolver` (:3122) and `subtract_unindexed_child_usage` (:3134)
    /// are `pub(super)`, and `local_harness_state_dir` (`core/agent_session.rs:11168`)
    /// is private. `load_context_tree_child_from_disk` /
    /// `load_context_tree_children_from_disk` (`core/context_tree.rs:385/464`) and
    /// `compute_own_and_total_usage` (:175) are already public, so the assembly is
    /// all that is missing. Owner to add: `core/agent_session/runtime_members.rs`,
    /// a `pub fn get_context_tree(&self) -> ContextTreeNode` matching
    /// `agent-session.ts:13028-13064`. Needs `runtime_members.rs` edited.
    fn get_context_tree(&self) -> Value {
        Value::Null
    }

    fn get_rlm_child_snapshots(&self) -> Vec<Value> {
        self.session().get_rlm_child_snapshots().into_iter().map(|child| serde_json::to_value(child).expect("child snapshot serializes")).collect()
    }

    fn export_to_html(&self, output_path: Option<&str>) -> BoxFuture<'static, Result<String, String>> {
        let session_file = self.runtime.session().session_file();
        let output_path = output_path.map(|path| path.to_string());
        Box::pin(async move {
            let Some(session_file) = session_file else {
                return Err("Cannot export in-memory session to HTML".to_string());
            };
            let options = crate::core::export_html::ExportOptions {
                output_path,
                ..Default::default()
            };
            crate::core::export_html::export_from_file(&session_file, Some(options))
        })
    }

    /// `state.runtime.session.exportToJsonl(command.outputPath)`
    /// (`daemon-mode.ts:5303`, `agent-session.ts:13093`).
    ///
    /// blocked_on: no owner. The writer needs the session header
    /// (`agent-session.ts:13100-13106`), the branch (`SessionManager::get_branch`,
    /// `core/session_manager.rs:4238`), a parentId re-chain and an
    /// ISO-8601-timestamped default path; `SessionManager::iso_now`
    /// (`core/session_manager.rs:892`) is private and no public function writes a
    /// branch as JSONL. Owner to add: `core/agent_session_runtime.rs` (or
    /// `SessionManager`), a `pub fn export_to_jsonl(&self, output_path: Option<&str>) -> Result<String, String>`
    /// matching `agent-session.ts:13093-13121`.
    fn export_to_jsonl(&self, output_path: Option<&str>) -> Result<String, String> {
        let _ = output_path;
        Err("blocked_on: no JSONL export owner".to_string())
    }

    fn get_last_assistant_text(&self) -> String {
        self.runtime.session()
            .messages()
            .iter()
            .rev()
            .find_map(|message| match message {
                // Canonical owner: `read_assistant_text` (side_question.rs:66) returns "" for every
                // message that is not an assistant message, so the last non-empty read wins.
                AgentMessage::Message(pi_ai::types::Message::Assistant(_)) => {
                    Some(crate::core::side_question::read_assistant_text(message))
                }
                _ => None,
            })
            .unwrap_or_default()
    }

    fn navigate_tree(
        &self,
        target_id: &str,
        options: NavigateTreeOptions,
    ) -> BoxFuture<'static, Result<Value, String>> {
        let session = Arc::clone(&self.runtime.session());
        let target_id = target_id.to_string();
        Box::pin(async move {
            session
                .navigate_tree(
                    &target_id,
                    options.summarize,
                    options.custom_instructions.as_deref(),
                )
                .await?;
            // `navigateTree` returns `{ editorText, cancelled, aborted, summaryEntry }`
            // (`agent-session.ts:12621-12625`); the port's owner returns
            // `Result<(), String>` (runtime_members.rs:2774) and the builder that
            // holds the result, `navigate_tree_inner` (:2785), discards the
            // editorText and the extension `cancel` branch.
            // blocked_on: `navigate_tree_inner` is `pub(super)` and must return a
            // result struct instead of `()`; owner to add in
            // `core/agent_session.rs` (the `NavigateTreeResult` type) plus
            // `runtime_members.rs:2785-2789` (the signature). Needs
            // `core/agent_session.rs` and `runtime_members.rs` edited.
            Ok(json!({ "cancelled": false }))
        })
    }

    /// Standalone side agents retain the parent's async auth/stream callbacks,
    /// but never append to the parent transcript or execute its tools.
    fn start_side_question(&self, question: &str, options: SideQuestionOptions) -> BoxFuture<'static, Result<(), String>> {
        let id = options.id;
        let previous = match options.previous_turns.map(serde_json::from_value).transpose() {
            Ok(previous) => previous, Err(error) => return Box::pin(async move { Err(error.to_string()) }),
        };
        let listener = options.on_event;
        let run = crate::core::side_question::native::start(self.session().agent.clone(), id.clone(), question.into(),
            Arc::new(move |event| {
                if let Some(listener) = &listener {
                    if let Ok(event) = serde_json::to_value(event) { listener(&event); }
                }
                Box::pin(async {})
            }), previous, None);
        let run = match run { Ok(run) => run, Err(error) => return Box::pin(async move { Err(error) }) };
        self.side_questions.lock().unwrap().insert(id.clone(), run.abort);
        let runs = self.side_questions.clone();
        tokio::spawn(async move { run.done.await; runs.lock().unwrap().remove(&id); });
        Box::pin(async { Ok(()) })
    }

    fn abort_side_question(&self, id: &str) {
        let abort = self.side_questions.lock().unwrap().remove(id);
        if let Some(abort) = abort { abort(); }
    }

    fn new_session(&self, options: Option<NewSessionRuntimeOptions>) -> BoxFuture<'static, Result<Value, String>> { self.runtime().new_session(options) }

    fn release_rlm_child_session(&self, child_id: &str, session: Arc<dyn DaemonSession>) -> Option<Box<dyn FnOnce() + Send>> {
        let child = session.agent_session()?;
        let release = self.session().release_rlm_child_session(child_id, &child)?;
        Some(Box::new(move || release()))
    }

    fn switch_session(&self, path: &str, options: SessionPathOptions) -> BoxFuture<'static, Result<Value, String>> { self.runtime().switch_session(path, options) }

    fn fork(&self, entry: &str, options: ForkOptions) -> BoxFuture<'static, Result<Value, String>> { self.runtime().fork(entry, options) }

    fn import_from_jsonl(&self, path: &str, cwd: Option<&str>) -> BoxFuture<'static, Result<Value, String>> { self.runtime().import_from_jsonl(path, cwd) }

    fn dispose(&self) -> BoxFuture<'static, ()> {
        let runtime = self.runtime.clone();
        Box::pin(async move { if let Err(error) = runtime.dispose(None).await { eprintln!("Could not dispose runtime: {error}"); } })
    }
}


fn restored_prompt(message: &str, images: Option<Value>, options: PromptInvocation) -> Result<crate::core::agent_session::RestoredPromptInput, String> {
    Ok(crate::core::agent_session::RestoredPromptInput {
        text: message.to_string(),
        content: options.content.map(serde_json::from_value).transpose().map_err(|error| error.to_string())?,
        images: images.or(options.images).map(serde_json::from_value).transpose().map_err(|error| error.to_string())?,
        queue_key: options.queue_key,
        agent_message_id: options.agent_message_id,
        custom_message: options.custom_message.map(serde_json::from_value).transpose().map_err(|error| error.to_string())?,
        prefix_messages: options.prefix_messages.map(serde_json::from_value).transpose().map_err(|error| error.to_string())?,
    })
}

struct DaemonRuntimeAdapter(Arc<super::AgentSessionRuntime>);

#[cfg(test)]
mod telegram_prompt_tests {
    use super::*;

    #[test]
    fn daemon_prompt_preserves_admission_acknowledgements() {
        let results = Arc::new(Mutex::new(Vec::new()));
        let captured = results.clone();
        let options = prompt_options_from_invocation(PromptInvocation {
            preflight_result: Some(Arc::new(move |accepted, queued| {
                captured.lock().unwrap().push((accepted, queued))
            })),
            ..Default::default()
        });
        let report = options.preflight_result.expect("daemon response callback must survive adaptation");
        report(true, false);
        report(false, false);
        assert_eq!(*results.lock().unwrap(), vec![(true, false), (false, false)]);
    }
}

impl DaemonRuntimeApi for DaemonRuntimeAdapter {
    fn new_session(&self, options: Option<NewSessionRuntimeOptions>) -> BoxFuture<'static, Result<Value, String>> {
        let runtime = self.0.clone();
        Box::pin(async move {
            let parent_session = options.and_then(|options| options.parent_session)
                .map(|value| value.as_str().map(str::to_string).ok_or_else(|| "parentSession must be a string".to_string())).transpose()?;
            let result = runtime.new_session(Some(super::NewSessionOptionsInput { parent_session, ..Default::default() })).await?;
            serde_json::to_value(result).map_err(|error| error.to_string())
        })
    }
    fn switch_session(&self, path: &str, options: SessionPathOptions) -> BoxFuture<'static, Result<Value, String>> {
        let runtime = self.0.clone();
        let path = path.to_string();
        Box::pin(async move {
            let result = runtime.switch_session(&path, Some(super::SwitchSessionOptions { cwd_override: options.cwd_override, with_session: None })).await?;
            serde_json::to_value(result).map_err(|error| error.to_string())
        })
    }
    fn fork(&self, entry: &str, options: ForkOptions) -> BoxFuture<'static, Result<Value, String>> {
        let runtime = self.0.clone();
        let entry = entry.to_string();
        Box::pin(async move {
            let result = runtime.fork(&entry, Some(super::ForkOptionsInput { position: options.position, with_session: None })).await?;
            let mut result_value = json!({"cancelled":result.cancelled});
            if let Some(text) = result.selected_text { result_value["selectedText"] = Value::String(text); }
            Ok(result_value)
        })
    }
    fn import_from_jsonl(&self, path: &str, cwd: Option<&str>) -> BoxFuture<'static, Result<Value, String>> {
        let runtime = self.0.clone();
        let path = path.to_string();
        let cwd = cwd.map(str::to_string);
        Box::pin(async move {
            let result = runtime.import_from_jsonl(&path, cwd.as_deref()).await?;
            serde_json::to_value(result).map_err(|error| error.to_string())
        })
    }
}

pub(crate) async fn run_native_daemon_mode(options: crate::main_entry::DaemonModeSeamOptions) -> Result<(), String> {
    use crate::modes::daemon::daemon_mode as daemon;
    let default_config = options.default_session_config.clone();
    let daemon_config = serde_json::from_value(serde_json::to_value(&default_config).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    let factory = options.create_runtime;
    let create_runtime: daemon::CreateAgentSessionRuntimeFactory = Arc::new(move |input| {
        let factory = factory.clone();
        let default_config = default_config.clone();
        Box::pin(async move {
            let metadata = input.runtime_metadata.map(serde_json::from_value).transpose().map_err(|error| error.to_string())?;
            let model = input.session_options.model.map(serde_json::from_value).transpose().map_err(|error| error.to_string())?;
            let mut creation = crate::core::agent_session_services::AgentSessionCreationOptions::default();
            if let Some(options) = &input.session_options.subagent_options {
                creation.thinking_level = Some(options.thinking_level);
                creation.service_tier = Some(options.service_tier.clone());
                creation.scoped_models = Some(options.scoped_models.iter().map(|item| crate::core::agent_session::ScopedModel { model: item.model.clone(), thinking_level: item.thinking_level }).collect());
                creation.initial_active_tool_names = Some(options.active_tool_names.clone());
                creation.allowed_tool_names = options.allowed_tool_names.clone();
                creation.custom_tools = Some(options.custom_tools.clone());
                creation.include_goals = Some(options.include_goals);
                creation.include_compact_skill = Some(options.include_compact_skill);
                creation.rlm_depth = Some(options.rlm_depth as i64);
                creation.rlm_max_depth = Some(options.rlm_max_depth as i64);
                creation.rlm_session_dir = Some(options.session_dir.clone());
                creation.rlm_parent_node_id = Some(options.rlm_parent_node_id.clone());
                creation.rlm_parent_agent = Some(options.parent_session.session_name().unwrap_or_else(|| options.parent_session.session_id()));
                creation.semantic_parent_session_id = Some(options.parent_session.session_id());
                creation.semantic_spawned_by_request_id = options.spawned_by_request_id.clone();
            }
            let runtime = super::create_agent_session_runtime(factory, super::CreateAgentSessionRuntimeInput {
                cwd: input.cwd,
                agent_dir: input.agent_dir.or(default_config.agent_dir.clone()).unwrap_or_else(crate::config::get_agent_dir),
                session_manager: input.session_manager,
                session_start_event: Some(json!({"type":"session_start","reason":"startup"})),
                session_config: input.session_config.or(Some(default_config)),
                session_options: Some(crate::core::agent_session_services::AgentSessionCreationOptions {
                    model,
                    agent_message_controller: input.session_options.agent_message_controller,
                    agent_observe_controller: input.session_options.agent_observe_controller,
                    rlm_heartbeat_controller: input.session_options.rlm_heartbeat_controller,
                    ..creation
                }),
                runtime_metadata: metadata,
                session_lease: None,
            }).await?;
            let metadata = serde_json::from_value(serde_json::to_value(runtime.metadata()).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
            let adapter = Arc::new(AgentSessionDaemonAdapter::new(runtime.clone()));
            Ok(daemon::AgentSessionRuntimeHandle {
                session: adapter.clone(), metadata,
                model_fallback_message: runtime.model_fallback_message(),
                new_session: Some({ let adapter = adapter.clone(); Arc::new(move |options| adapter.new_session(options)) }),
                switch_session: Some({ let adapter = adapter.clone(); Arc::new(move |path, options| adapter.switch_session(&path, options)) }),
                fork: Some({ let adapter = adapter.clone(); Arc::new(move |entry, options| adapter.fork(&entry, options)) }),
                import_from_jsonl: Some({ let adapter = adapter.clone(); Arc::new(move |path, cwd| adapter.import_from_jsonl(&path, cwd.as_deref())) }),
            })
        })
    });
    daemon::run_daemon_mode(daemon::DaemonModeOptions {
        socket_path: options.socket_path, default_session_config: daemon_config, create_runtime,
        worker: options.worker.map(|worker| daemon::DaemonWorkerOptions {
            authentication_token: worker.authentication_token,
            worker_instance_id: worker.worker_instance_id,
            restore_active_session_id: worker.restore_active_session_id,
        }),
    }).await
}
