//! Runtime members of the agent-session.ts port.

use super::*;

#[path = "subagent_runs.rs"]
mod subagent_runs;

use crate::core::agent_messages::{AgentSessionMessageAgentSummary, RUNTIME_KIND_SUBAGENT};
use crate::core::extensions::types::{
    AppendEntryHandler, CompactOptions, CustomMessagePayload, ExtensionActions,
    ExtensionContextActions, GetActiveToolsHandler, GetAllToolsHandler, GetCommandsHandler,
    GetSessionNameHandler, GetThinkingLevelHandler, ProviderActions, RefreshToolsHandler,
    SendMessageHandler, SendMessageOptions, SendUserMessageHandler, SendUserMessageOptions,
    SetActiveToolsHandler, SetLabelHandler, SetModelHandler, SetSessionNameHandler,
    SetThinkingLevelHandler,
};
use crate::core::extensions::types::{
    ProviderConfig as ExtensionProviderConfig, SessionEntry as ExtensionSessionEntry,
};
use crate::core::provider_retry::{
    is_faux_provider_queue_exhausted, is_permanent_provider_failure_kind, provider_retry_delay,
    provider_retry_policy, provider_stream_failure_kind, provider_stream_failure_retry_after_ms,
    ProviderRetryDelay, ProviderRetryDelayOptions,
};
use crate::core::rlm_runtime::{
    create_default_rlm_subagent_session_name, normalize_requested_rlm_subagent_model,
    normalize_requested_rlm_subagent_session_name, normalize_requested_rlm_subagent_thinking_level,
    CreateRlmRootSessionOptions, DELETE_OUTCOME_DELETED, DELETE_OUTCOME_SKIPPED_RUNNING,
    RLM_SUBAGENT_STATUS_COMPLETED, RLM_SUBAGENT_STATUS_ERROR, RLM_SUBAGENT_STATUS_RUNNING,
};
use crate::core::session_stats::SessionStatsTokens;
use crate::core::skills::Skill;
use crate::core::slash_commands::SlashCommandInfo;
use crate::core::source_info::{create_synthetic_source_info, SyntheticSourceInfoOptions};

/// `deleteInactiveRlmSubagent` outcome literals.
///
/// REPAIR CURSOR: `core/rlm_runtime.rs` declares only `DELETE_OUTCOME_DELETED`
/// and `DELETE_OUTCOME_SKIPPED_RUNNING`; the `"not_found"`/`"running"` members
/// of the TypeScript return union have no constant there, while
/// `modes/daemon/daemon_mode.rs:5098` already compares against the `"deleted"` /
/// `"running"` literals. Add the two missing constants to their canonical owner
/// (`core/rlm_runtime.rs`) and import them here.
const DELETE_OUTCOME_NOT_FOUND_LITERAL: &str = "not_found";
const DELETE_OUTCOME_RUNNING_LITERAL: &str = "running";

/// `new ExtensionRunner(..., sessionManager, modelRegistry)`.
///
/// The TypeScript hands the runner the live `sessionManager` and
/// `modelRegistry` objects. Rust `extensions::types` declares the two traits
/// (`ReadonlySessionManager`/`SessionManager`, `ModelRegistry`) but nothing in
/// the crate implements them for the real owners, so the construction call site
/// forwards to them here. Nothing is dropped: every method reads or writes the
/// canonical object.
struct RuntimeSessionManager {
    manager: Arc<Mutex<SessionManager>>,
}

impl crate::core::extensions::types::ReadonlySessionManager for RuntimeSessionManager {
    fn get_session_id(&self) -> String {
        self.manager.lock().unwrap().get_session_id()
    }

    fn get_session_file(&self) -> Option<String> {
        self.manager.lock().unwrap().get_session_file()
    }

    fn get_session_dir(&self) -> String {
        self.manager.lock().unwrap().get_session_dir()
    }

    fn get_entry_count(&self) -> Option<usize> {
        Some(self.manager.lock().unwrap().get_entry_count())
    }

    /// `sessionManager.getEntries()` as the runner's `SessionEntry` shape
    /// (`{ type, id, ...rest }`); `types::SessionEntry` flattens the rest.
    fn get_branch(&self) -> Vec<ExtensionSessionEntry> {
        self.manager
            .lock()
            .unwrap()
            .get_entries()
            .into_iter()
            .map(|mut entry| {
                let id = entry
                    .remove("id")
                    .and_then(|value| value.as_str().map(str::to_string))
                    .unwrap_or_default();
                let entry_type = entry
                    .remove("type")
                    .and_then(|value| value.as_str().map(str::to_string))
                    .unwrap_or_default();
                ExtensionSessionEntry {
                    entry_type,
                    id,
                    extra: entry,
                }
            })
            .collect()
    }
}

impl crate::core::extensions::types::SessionManager for RuntimeSessionManager {}

/// The `modelRegistry` half of the same seam.
struct RuntimeModelRegistry {
    registry: Arc<Mutex<crate::core::model_registry::ModelRegistry>>,
}

impl crate::core::extensions::types::ModelRegistry for RuntimeModelRegistry {
    fn register_provider(&self, name: &str, config: &ExtensionProviderConfig) {
        let input = crate::core::agent_session_services::provider_config_input(config);
        let _ = self.registry.lock().unwrap().register_provider(name, input);
    }

    fn unregister_provider(&self, name: &str) {
        self.registry.lock().unwrap().unregister_provider(name);
    }

    /// `modelRegistry.getApiKeyAndHeaders(model)` - the TypeScript result object
    /// is `{ ok, apiKey, headers, error }`.
    fn get_api_key_and_headers(
        &self,
        model: &Model,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, String>> + Send>> {
        let model = model.clone();
        let registry = self.registry.clone();
        Box::pin(async move {
            // The registry lock is synchronous, so the lookup runs on a blocking
            // worker (`with_model_registry`) instead of holding the guard across the
            // await; same shape as `get_required_request_auth`.
            let resolved = crate::core::sdk::with_model_registry(registry, move |registry| {
                Box::pin(async move { registry.get_api_key_and_headers(&model).await })
            })
            .await
            .unwrap_or_else(|error| {
                crate::core::model_registry::ResolvedRequestAuth {
                    ok: false,
                    api_key: None,
                    headers: None,
                    error: Some(error),
                }
            });
            Ok(serde_json::json!({
                "ok": resolved.ok,
                "apiKey": resolved.api_key,
                "headers": resolved.headers,
                "error": resolved.error,
            }))
        })
    }
}

/// `createSyntheticSourceInfo` for the two synthetic tool sources the runtime
/// registers: `"<builtin:name>"` and `"<sdk:name>"` (agent-session.ts:9962-9986).
fn runtime_source_info(name: &str, source: &str) -> SourceInfo {
    let path = if name.starts_with('<') {
        name.to_string()
    } else {
        format!("<{source}:{name}>")
    };
    create_synthetic_source_info(
        &path,
        &SyntheticSourceInfoOptions {
            source: source.to_string(),
            ..Default::default()
        },
    )
}

/// A session resolved at use time; a callback on a dropped session is a no-op.
fn with_session<R>(
    weak: &Weak<AgentSession>,
    action: impl FnOnce(&Arc<AgentSession>) -> R,
) -> Option<R> {
    weak.upgrade().map(|session| action(&session))
}

/// `runner.getRegisteredCommands()` + prompt templates + skills as
/// `SlashCommandInfo[]` (agent-session.ts:9847-9870).
fn runtime_extension_commands(
    session: &Arc<AgentSession>,
    runner: &ExtensionRunner,
) -> Vec<SlashCommandInfo> {
    let mut commands: Vec<SlashCommandInfo> = runner
        .get_registered_commands()
        .into_iter()
        .map(|command| SlashCommandInfo {
            name: command.invocation_name,
            description: command.command.description.clone(),
            source: crate::core::slash_commands::SLASH_COMMAND_SOURCE_EXTENSION.to_string(),
            source_info: command.command.source_info.clone(),
        })
        .collect();
    commands.extend(
        session
            .prompt_templates()
            .into_iter()
            .map(|template| SlashCommandInfo {
                name: template.name,
                description: Some(template.description),
                source: crate::core::slash_commands::SLASH_COMMAND_SOURCE_PROMPT.to_string(),
                source_info: template.source_info,
            }),
    );
    commands.extend(
        session
            .resource_loader
            .get_skills()
            .skills
            .into_iter()
            .map(|skill| {
                let base = match &skill {
                    Skill::Markdown(skill) => &skill.base,
                    Skill::Python(skill) => &skill.base,
                };
                SlashCommandInfo {
                    name: format!("skill:{}", base.name),
                    description: Some(base.description.clone()),
                    source: crate::core::slash_commands::SLASH_COMMAND_SOURCE_SKILL.to_string(),
                    source_info: base.source_info.clone(),
                }
            }),
    );
    commands
}

/// `_bindExtensionCore` action half (agent-session.ts:9872-9920).
fn build_extension_actions(weak: Weak<AgentSession>, runner: ExtensionRunner) -> ExtensionActions {
    let send_runner = runner.clone();
    let send_weak = weak.clone();
    let send_message: crate::core::extensions::types::SendMessageHandler = Arc::new(
        move |message: CustomMessagePayload, options: Option<SendMessageOptions>| {
            let runner = send_runner.clone();
            let session = send_weak.upgrade();
            let Some(session) = session else { return };
            let custom = CustomMessage {
                role: "custom".to_string(),
                custom_type: message.custom_type.clone(),
                content: custom_message_content_from_value(&message.content),
                display: message.display,
                details: message.details.clone(),
                timestamp: now_ms_i64(),
            };
            let (trigger_turn, deliver_as) = match options {
                Some(options) => (options.trigger_turn, options.deliver_as),
                None => (None, None),
            };
            tokio::task::spawn(async move {
                if let Err(error) = session
                    .send_custom_message(custom, trigger_turn, deliver_as)
                    .await
                {
                    runner.emit_error(ExtensionError {
                        extension_path: "<runtime>".to_string(),
                        event: "send_message".to_string(),
                        error,
                        stack: None,
                    });
                }
            });
        },
    );
    let send_user_runner = runner.clone();
    let send_user_weak = weak.clone();
    let send_user_message: crate::core::extensions::types::SendUserMessageHandler = Arc::new(
        move |content: Value, options: Option<SendUserMessageOptions>| {
            let runner = send_user_runner.clone();
            let Some(session) = send_user_weak.upgrade() else {
                return;
            };
            let parts: Vec<pi_ai::types::ImageOrTextContent> = match &content {
                Value::String(text) => vec![pi_ai::types::ImageOrTextContent::Text(
                    TextContent::new(text),
                )],
                Value::Array(_) => serde_json::from_value(content.clone()).unwrap_or_default(),
                _ => Vec::new(),
            };
            let deliver_as = options.and_then(|options| options.deliver_as);
            let text = normalize_message_content(&parts).0;
            let content = if parts.len() == 1 {
                parts[0].clone()
            } else {
                pi_ai::types::ImageOrTextContent::Text(TextContent::new(&text))
            };
            tokio::task::spawn(async move {
                if let Err(error) = session.send_user_message(&content, deliver_as).await {
                    runner.emit_error(ExtensionError {
                        extension_path: "<runtime>".to_string(),
                        event: "send_user_message".to_string(),
                        error,
                        stack: None,
                    });
                }
            });
        },
    );
    let append_weak = weak.clone();
    let append_entry: crate::core::extensions::types::AppendEntryHandler =
        Arc::new(move |custom_type: String, data: Option<Value>| {
            let _ = with_session(&append_weak, |session| {
                let _ = session
                    .session_manager
                    .lock()
                    .unwrap()
                    .append_custom_entry(&custom_type, data.clone());
            });
        });
    let name_weak = weak.clone();
    let set_session_name: crate::core::extensions::types::SetSessionNameHandler =
        Arc::new(move |name: String| {
            let session = name_weak.upgrade();
            Box::pin(async move {
                let Some(session) = session else { return };
                let _ = session.set_session_name(&name);
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        });
    let get_name_weak = weak.clone();
    let get_session_name: crate::core::extensions::types::GetSessionNameHandler =
        Arc::new(move || with_session(&get_name_weak, |session| session.session_name()).flatten());
    let label_weak = weak.clone();
    let set_label: crate::core::extensions::types::SetLabelHandler =
        Arc::new(move |entry_id: String, label: Option<String>| {
            let _ = with_session(&label_weak, |session| {
                let _ = session
                    .session_manager
                    .lock()
                    .unwrap()
                    .append_label_change(&entry_id, label.as_deref());
            });
        });
    let active_weak = weak.clone();
    let get_active_tools: crate::core::extensions::types::GetActiveToolsHandler =
        Arc::new(move || {
            with_session(&active_weak, |session| session.get_active_tool_names())
                .unwrap_or_default()
        });
    let all_weak = weak.clone();
    let get_all_tools: crate::core::extensions::types::GetAllToolsHandler = Arc::new(move || {
        with_session(&all_weak, |session| {
            session
                .get_all_tools()
                .into_iter()
                .filter_map(|tool| {
                    serde_json::from_value::<crate::core::extensions::types::ToolInfo>(tool).ok()
                })
                .collect()
        })
        .unwrap_or_default()
    });
    let set_active_weak = weak.clone();
    let set_active_tools: crate::core::extensions::types::SetActiveToolsHandler =
        Arc::new(move |tool_names: Vec<String>| {
            let _ = with_session(&set_active_weak, |session| {
                session.set_active_tools_by_name(&tool_names)
            });
        });
    let refresh_weak = weak.clone();
    let refresh_tools: crate::core::extensions::types::RefreshToolsHandler = Arc::new(move || {
        let _ = with_session(&refresh_weak, |session| {
            session.refresh_tool_registry(false, None)
        });
    });
    let commands_weak = weak.clone();
    let commands_runner = runner.clone();
    let get_commands: crate::core::extensions::types::GetCommandsHandler = Arc::new(move || {
        with_session(&commands_weak, |session| {
            runtime_extension_commands(session, &commands_runner)
        })
        .unwrap_or_default()
    });
    let set_model_weak = weak.clone();
    let set_model: crate::core::extensions::types::SetModelHandler =
        Arc::new(move |model: Model| {
            let session = set_model_weak.upgrade();
            Box::pin(async move {
                let Some(session) = session else { return false };
                if !session
                    .model_registry
                    .lock()
                    .unwrap()
                    .has_configured_auth(&model)
                {
                    return false;
                }
                session
                    .set_model(model, ModelSelectOptions::default())
                    .await
                    .is_ok()
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>>
        });
    let get_thinking_weak = weak.clone();
    let get_thinking_level: crate::core::extensions::types::GetThinkingLevelHandler =
        Arc::new(move || {
            with_session(&get_thinking_weak, |session| session.thinking_level())
                .unwrap_or(ThinkingLevel::Off)
        });
    let set_thinking_weak = weak.clone();
    let set_thinking_level: crate::core::extensions::types::SetThinkingLevelHandler =
        Arc::new(move |level: ThinkingLevel| {
            let _ = with_session(&set_thinking_weak, |session| {
                let session = session.clone();
                session.set_thinking_level(level);
            });
        });
    ExtensionActions {
        send_message,
        send_user_message,
        append_entry,
        set_session_name,
        get_session_name,
        set_label,
        get_active_tools,
        get_all_tools,
        set_active_tools,
        refresh_tools,
        get_commands,
        set_model,
        get_thinking_level,
        set_thinking_level,
    }
}

/// `_bindExtensionCore` context-action half (agent-session.ts:9921-9943).
fn build_extension_context_actions(weak: Weak<AgentSession>) -> ExtensionContextActions {
    let model_weak = weak.clone();
    let get_model =
        Arc::new(move || with_session(&model_weak, |session| session.model()).flatten());
    let idle_weak = weak.clone();
    let is_idle = Arc::new(move || {
        with_session(&idle_weak, |session| !session.is_streaming()).unwrap_or(true)
    });
    let signal_weak = weak.clone();
    let get_signal =
        Arc::new(move || with_session(&signal_weak, |session| session.agent.signal()).flatten());
    let abort_weak = weak.clone();
    let abort = Arc::new(move || {
        let Some(session) = abort_weak.upgrade() else {
            return;
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = session.abort().await;
            });
        }
    });
    let pending_weak = weak.clone();
    let has_pending_messages = Arc::new(move || {
        with_session(&pending_weak, |session| session.queued_action_count() > 0).unwrap_or(false)
    });
    // REPAIR CURSOR: the TypeScript calls `this._extensionShutdownHandler?.()`
    // (agent-session.ts:9928). The Rust owner is the private field
    // `agent_session.rs:2235 extension_shutdown_handler`, which has no accessor and
    // no setter in `bind_extensions`, so this port cannot reach it from the child
    // module. Add an accessor/`set_extension_shutdown_handler` to `agent_session.rs`
    // (lead-owned seam) and call it here.
    let shutdown: crate::core::extensions::runner::ShutdownHandler = Arc::new(|| {});
    let usage_weak = weak.clone();
    let get_context_usage = Arc::new(move || {
        with_session(&usage_weak, |session| session.get_context_usage())
            .flatten()
            .map(|usage| crate::core::extensions::types::ContextUsage {
                tokens: usage.tokens,
                context_window: usage.context_window,
                percent: usage.percent,
            })
    });
    let compact_weak = weak.clone();
    let compact = Arc::new(move |options: Option<CompactOptions>| {
        let Some(session) = compact_weak.upgrade() else {
            return;
        };
        let (custom_instructions, on_complete, on_error) = match options {
            Some(options) => (
                options.custom_instructions,
                options.on_complete,
                options.on_error,
            ),
            None => (None, None, None),
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                // REPAIR CURSOR: the TypeScript hands the `CompactionResult` to
                // `options.onComplete` (agent-session.ts:9934). The public Rust
                // owner `compact_with_options` returns `Result<(), String>`; the
                // result object is only built by the private
                // `perform_compaction_unmeasured_full`. Expose that result on the
                // public path and forward it here.
                match session
                    .compact_with_options(custom_instructions.as_deref(), false)
                    .await
                {
                    Ok(()) => {}
                    Err(error) => {
                        if let Some(on_error) = on_error {
                            on_error(error);
                        }
                    }
                }
                let _ = on_complete;
            });
        }
    });
    let prompt_weak = weak.clone();
    let get_system_prompt = Arc::new(move || {
        with_session(&prompt_weak, |session| session.system_prompt()).unwrap_or_default()
    });
    ExtensionContextActions {
        get_model,
        is_idle,
        get_signal,
        abort,
        has_pending_messages,
        shutdown,
        get_context_usage,
        compact,
        get_system_prompt,
    }
}

/// `_bindExtensionCore` provider-action half (agent-session.ts:9944-9953).
fn build_provider_actions(weak: Weak<AgentSession>) -> ProviderActions {
    let register_weak = weak.clone();
    let register_provider = Arc::new(move |name: String, config: ExtensionProviderConfig| {
        let Some(session) = register_weak.upgrade() else {
            return;
        };
        let input = crate::core::agent_session_services::provider_config_input(&config);
        let _ = session
            .model_registry
            .lock()
            .unwrap()
            .register_provider(&name, input);
        session.refresh_current_model_from_registry();
    });
    let unregister_weak = weak;
    let unregister_provider = Arc::new(move |name: String| {
        let Some(session) = unregister_weak.upgrade() else {
            return;
        };
        session
            .model_registry
            .lock()
            .unwrap()
            .unregister_provider(&name);
        session.refresh_current_model_from_registry();
    });
    ProviderActions {
        register_provider: Some(register_provider),
        unregister_provider: Some(unregister_provider),
    }
}

/// Bridges an extension-supplied `user_bash` `operations` value (the extension-facing
/// `extensions::types::BashOperations`, `UserBashEventResult.operations` at
/// `core/extensions/types.ts`) onto the executor's `core::tools::bash::BashOperations`.
///
/// In the TypeScript both are the *same* interface, imported from `../tools/bash.js`
/// (`import type { BashOperations } from "../tools/bash.js";` in
/// `core/extensions/types.ts`), so `executeBash(..., { operations: eventResult?.operations })`
/// (agent-session.ts:12462-12466) passes one object straight through. The Rust port declares
/// two structurally identical traits in two slices; this adapter is the pass-through at the
/// single call site that TypeScript resolves for free. It drops nothing: every argument is
/// forwarded and the exit code is converted exactly once.
struct ExtensionBashOperations {
    operations: Arc<dyn crate::core::extensions::types::BashOperations>,
}

impl crate::core::tools::BashOperations for ExtensionBashOperations {
    fn exec(
        &self,
        command: &str,
        cwd: &str,
        options: crate::core::tools::bash::BashExecOptions,
    ) -> futures::future::BoxFuture<'static, Result<crate::core::tools::bash::BashExecResult, String>>
    {
        let operations = self.operations.clone();
        let command = command.to_string();
        let cwd = cwd.to_string();
        let on_data = options.on_data.clone();
        let signal = options.signal.clone();
        let timeout = options.timeout;
        let env = options.env.map(|env| {
            env.into_iter()
                .map(|(key, value)| (key, Value::String(value)))
                .collect::<Map<String, Value>>()
        });
        Box::pin(async move {
            // The extension trait streams `Vec<u8>` chunks; the executor hands out borrowed
            // slices, so each chunk is copied once before forwarding.
            let forward: Arc<dyn Fn(Vec<u8>) + Send + Sync> = Arc::new(move |data: Vec<u8>| {
                on_data(&data);
            });
            let exit_code = operations
                .exec(command, cwd, forward, signal, timeout, env)
                .await?;
            // The extension trait reports the exit code as a JS number; the executor's
            // `BashExecResult` is `Option<i32>`. An out-of-range code is reported instead of
            // silently truncated, so a bad extension value cannot look like a clean exit.
            let exit_code = match exit_code {
                Some(code) => Some(i32::try_from(code).map_err(|_| {
                    format!("user_bash operations returned an out-of-range exit code: {code}")
                })?),
                None => None,
            };
            Ok(crate::core::tools::bash::BashExecResult { exit_code })
        })
    }
}

impl AgentSession {
    pub async fn bind_extensions(
        self: &Arc<Self>,
        bindings: &ExtensionBindings,
    ) -> Result<(), String> {
        let Some(runner) = self.extension_runner() else {
            return Ok(());
        };
        if let Some(ui) = &bindings.ui_context {
            runner.set_ui_context(Some(ui.clone()));
        }
        if let Some(actions) = &bindings.command_context_actions {
            runner.bind_command_context(Some(actions.clone()));
        }
        if let Some(listener) = &bindings.on_error {
            runner.on_error(listener.clone());
        }
        let event = serde_json::from_value(self.session_start_event.clone())
            .map_err(|error| error.to_string())?;
        runner.emit(event).await;
        self.extend_resources_from_extensions("startup").await
    }

    pub(super) async fn extend_resources_from_extensions(
        self: &Arc<Self>,
        reason: &str,
    ) -> Result<(), String> {
        let Some(runner) = self.extension_runner() else {
            return Ok(());
        };
        if !runner.has_handlers("resources_discover") {
            return Ok(());
        }
        let paths = runner
            .emit_resources_discover(self.cwd.clone(), reason.to_string())
            .await;
        self.resource_loader.extend_resources(
            crate::core::resource_loader::ResourceExtensionPaths {
                skill_paths: Some(self.build_extension_resource_paths(&paths.skill_paths)),
                prompt_paths: Some(self.build_extension_resource_paths(&paths.prompt_paths)),
                theme_paths: Some(self.build_extension_resource_paths(&paths.theme_paths)),
            },
        );
        let prompt = self.rebuild_system_prompt(&self.get_active_tool_names());
        *self.base_system_prompt.lock().unwrap() = prompt.clone();
        let mut state = self.agent.state();
        state.system_prompt = prompt;
        self.agent.set_state(state);
        Ok(())
    }

    pub(super) fn build_extension_resource_paths(
        &self,
        entries: &[crate::core::extensions::runner::ResourcePathEntry],
    ) -> Vec<crate::core::resource_loader::ResourcePathEntry> {
        entries
            .iter()
            .map(|entry| crate::core::resource_loader::ResourcePathEntry {
                path: entry.path.clone(),
                metadata: crate::core::package_manager::PathMetadata {
                    source: self.get_extension_source_label(&entry.extension_path),
                    scope: "temporary".to_string(),
                    origin: "top-level".to_string(),
                    base_dir: if entry.extension_path.starts_with('<') {
                        None
                    } else {
                        Path::new(&entry.extension_path)
                            .parent()
                            .map(|path| path.to_string_lossy().into_owned())
                    },
                },
            })
            .collect()
    }

    pub(super) fn get_extension_source_label(&self, path: &str) -> String {
        if path.starts_with('<') {
            return format!("extension:{}", path.replace(['<', '>'], ""));
        }
        let name = Path::new(path)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        let name = name
            .strip_suffix(".ts")
            .or_else(|| name.strip_suffix(".js"))
            .unwrap_or(&name);
        format!("extension:{name}")
    }

    pub(super) fn apply_extension_bindings(&self, runner: &ExtensionRunner) {
        runner.bind_command_context(self.extension_command_context_actions.clone());
        if let Some(listener) = &self.extension_error_listener {
            runner.on_error(listener.clone());
        }
    }

    pub(super) fn refresh_current_model_from_registry(self: &Arc<Self>) {
        if let Some(current) = self.model() {
            if let Some(model) = self
                .model_registry
                .lock()
                .unwrap()
                .find(&current.provider, &current.id)
            {
                let mut state = self.agent.state();
                state.model = model;
                self.agent.set_state(state);
            }
        }
    }

    pub(super) fn bind_extension_core(self: &Arc<Self>, runner: &ExtensionRunner) {
        // Each callback resolves the session at use time to avoid retaining its owner.
        let weak = Arc::downgrade(self);
        runner.bind_core(
            build_extension_actions(weak.clone(), runner.clone()),
            build_extension_context_actions(weak.clone()),
            Some(build_provider_actions(weak)),
        );
    }

    pub(super) fn refresh_tool_registry(
        self: &Arc<Self>,
        include_all: bool,
        active_tool_names: Option<Vec<String>>,
    ) {
        use crate::core::extensions::types::RegisteredTool;
        use crate::core::extensions::wrapper::{wrap_registered_tools, RunnerSource};
        let Some(runner) = self.extension_runner() else {
            return;
        };
        let previous: HashSet<String> =
            self.tool_registry.lock().unwrap().keys().cloned().collect();
        let mut active = active_tool_names
            .clone()
            .unwrap_or_else(|| self.get_active_tool_names());
        let allowed = self.allowed_tool_names.lock().unwrap().clone();
        let permitted = |name: &str| allowed.as_ref().map_or(true, |names| names.contains(name));
        let mut entries: Vec<RegisteredTool> = self
            .base_tool_definitions
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| permitted(name))
            .map(|(name, definition)| RegisteredTool {
                definition: definition.clone(),
                source_info: runtime_source_info(name, "builtin"),
            })
            .collect();
        let mut custom = runner.get_all_registered_tools();
        custom.extend(
            self.custom_tools
                .iter()
                .chain(self.acp_mcp_tools.lock().unwrap().iter())
                .map(|definition| RegisteredTool {
                    definition: definition.clone(),
                    source_info: runtime_source_info(&definition.name, "sdk"),
                }),
        );
        custom.retain(|entry| permitted(&entry.definition.name));
        if include_all {
            active.extend(custom.iter().map(|entry| entry.definition.name.clone()));
        }
        entries.extend(custom);
        *self.tool_definitions.lock().unwrap() = entries
            .iter()
            .map(|entry| {
                (
                    entry.definition.name.clone(),
                    ToolDefinitionEntry {
                        definition: entry.definition.clone(),
                        source_info: entry.source_info.clone(),
                    },
                )
            })
            .collect();
        let tools = wrap_registered_tools(&entries, RunnerSource::Runner(runner));
        if allowed.is_some() || active_tool_names.is_none() {
            active.extend(
                tools
                    .iter()
                    .filter(|tool| allowed.is_some() || !previous.contains(&tool.name))
                    .map(|tool| tool.name.clone()),
            );
        }
        *self.tool_registry.lock().unwrap() = tools
            .into_iter()
            .map(|tool| (tool.name.clone(), tool))
            .collect();
        active.retain(|name| permitted(name));
        let mut seen = HashSet::new();
        active.retain(|name| seen.insert(name.clone()));
        self.set_active_tools_by_name(&active);
    }

    pub fn build_runtime(
        self: &Arc<Self>,
        active_tool_names: Option<Vec<String>>,
        include_all: bool,
    ) {
        let definitions: BTreeMap<String, crate::core::extensions::types::ToolDefinition> = match &self.base_tools_override {
            Some(tools) => tools.iter().map(|(name, tool)| (name.clone(),
                crate::core::tools::tool_definition_wrapper::create_tool_definition_from_agent_tool(tool).into())).collect(),
            None => {
                let (command_prefix, shell_path) = {
                    let settings = self.settings_manager.lock().unwrap();
                    (settings.get_shell_command_prefix(), settings.get_shell_path())
                };
                // `onLateSentAgentMessage: (toolCallId, message) => this._recordLateIpythonSentAgentMessage(toolCallId, message)`
                // (agent-session.ts:10099-10100; handler at agent-session.ts:1747): an agent message the
                // kernel emits *after* the ipython call already returned must still reach the session so
                // `_rememberLateIpythonSentAgentMessage` can attach it to the tool result and persist it.
                let late_session = Arc::downgrade(self);
                let on_late_sent_agent_message:
                    Option<Arc<dyn Fn(String, crate::core::kernel::shared::KernelSentAgentMessage) + Send + Sync>> =
                    Some(Arc::new(
                        move |tool_call_id: String,
                              message: crate::core::kernel::shared::KernelSentAgentMessage| {
                            let Some(session) = late_session.upgrade() else {
                                return;
                            };
                            // The kernel hands the callback the parsed message; the session records
                            // the `{ toolCallId, message }` entry the TypeScript persists verbatim.
                            if let Ok(message) = serde_json::to_value(message) {
                                session.record_late_ipython_sent_agent_message(&tool_call_id, message);
                            }
                        },
                    ));
                // TS agent-session.ts:10061-10102: the session OWNS the kernel provisioner
                // and hands the same Arc to the ipython tool, so the session's dispose and
                // reload address the exact kernel the tool drives.
                let python_skills: Vec<crate::core::kernel::shared::KernelPythonSkill> =
                    crate::core::skills::get_python_skill_runtime_info(&self.model_visible_skills())
                        .into_iter()
                        .map(|info| crate::core::kernel::shared::KernelPythonSkill {
                            import_name: info.python.import_name,
                            package_path: info.python.package_path,
                            pyproject_path: info.python.pyproject_path,
                            name: info.name,
                        })
                        .collect();
                // Rebuilding (e.g. /reload) replaces the provisioner; the previous kernel's
                // dispose gates the new kernel's startup (agent-session.ts:10071-10091).
                // Begin disposal now, even if no later tool call starts the replacement.
                let previous_provisioner = self.ipython_kernel_provisioner.lock().unwrap().clone();
                let ready_gate = previous_provisioner
                    .map(|previous| previous.replacement_ready_gate());
                // Only the first build (a genuine resume) surfaces the restore notice
                // (agent-session.ts:10077-10080); a later rebuild restores silently.
                let notify_restore = !self.ipython_runtime_built.load(Ordering::SeqCst);
                let snapshot_dir = self.session_manager.lock().unwrap().get_session_artifact_dir()
                    .map(|dir| match self.retained_kernel_epoch.lock().unwrap().as_ref() {
                        Some(epoch) => Path::new(&dir).join(format!("audit-{epoch}"))
                            .to_string_lossy().into_owned(),
                        None => dir,
                    });
                let provisioner = crate::core::tools::ipython::IpythonKernelProvisioner::new(
                    &self.cwd,
                    Some(crate::core::tools::IpythonToolOptions {
                        env: Some(self.rlm_kernel_env().into_iter().collect()),
                        command_prefix: command_prefix.clone(),
                        shell_path: shell_path.clone(),
                        session_id: Some(self.session_id()),
                        host_handlers: Some(self.create_kernel_host_handlers()),
                        python_skills: Some(python_skills),
                        snapshot_dir: snapshot_dir.clone(),
                        performance_metrics: self.agent.performance_metrics().map(|metrics| {
                            Arc::new(crate::core::kernel::performance_metrics::KernelPerformanceMetricAdapter::new(metrics.recorder))
                                as Arc<dyn crate::core::kernel::shared::PerformanceMetricRecorder>
                        }),
                        model_tool_output_policy: Some(crate::core::model_tool_output_policy::resolve_model_tool_output_policy(
                            Some(&self.settings_manager.lock().unwrap().get_model_tool_output_policy()),
                        )),
                        ready_gate,
                        on_background_work_settled: {
                            let weak = Arc::downgrade(self);
                            Some(Arc::new(move || {
                                if let Some(session) = weak.upgrade() {
                                    session.maybe_resume_goal_continuation_after_rlm_work();
                                    session.session_action_activity_notify.notify_waiters();
                                }
                            }))
                        },
                        on_restore: if notify_restore {
                            let weak = Arc::downgrade(self);
                            Some(Arc::new(move |result: crate::core::kernel::state_snapshot::RestoreResult| {
                                if let Some(session) = weak.upgrade() {
                                    session.on_ipython_state_restored(result);
                                }
                            }) as Arc<dyn Fn(crate::core::kernel::state_snapshot::RestoreResult) + Send + Sync>)
                        } else {
                            None
                        },
                        on_late_sent_agent_message: on_late_sent_agent_message.clone(),
                        ..Default::default()
                    }),
                    crate::core::tools::ipython::default_kernel_client_factory(),
                );
                *self.ipython_kernel_provisioner.lock().unwrap() = Some(provisioner.clone());
                let options = crate::core::tools::ToolsOptions { ipython: Some(crate::core::tools::IpythonToolOptions {
                    env: Some(self.rlm_kernel_env().into_iter().collect()),
                    host_handlers: Some(self.create_kernel_host_handlers()), session_id: Some(self.session_id()),
                    command_prefix,
                    shell_path,
                    snapshot_dir,
                    on_late_sent_agent_message,
                    provisioner: Some(provisioner),
                    ..Default::default()
                }) };
                crate::core::tools::create_all_tool_definitions(&self.cwd, Some(&options))
                    .into_iter().map(|(name, definition)| (name.to_string(), definition.into())).collect()
            }
        };
        *self.base_tool_definitions.lock().unwrap() = definitions;
        let loaded = self.resource_loader.get_extensions();
        // The runner reaches both owners through the `extensions::types` traits;
        // the two adapters declared above forward every call to the canonical owners.
        let runner = crate::core::extensions::runner::create_extension_runner(
            loaded.extensions,
            loaded.runtime,
            self.cwd.clone(),
            Arc::new(RuntimeSessionManager {
                manager: self.session_manager.clone(),
            }),
            Arc::new(RuntimeModelRegistry {
                registry: self.model_registry.clone(),
            }),
        );
        self.extension_runner_ref.set(Some(runner.clone()));
        self.bind_extension_core(&runner);
        self.apply_extension_bindings(&runner);
        // TS agent-session.ts:10135-10146: on every (re)build, ACP MCP tools are
        // (re)created against the SAME session-owned kernel provisioner, so MCP
        // requests ride the session's kernel instead of a second one.
        let previous_acp_mcp_tool_names: Vec<String> = self
            .acp_mcp_tools
            .lock()
            .unwrap()
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        let acp_servers = self
            .mcp_manager
            .as_ref()
            .map(|manager| manager.lock().unwrap().get_acp_servers())
            .unwrap_or_default();
        if !acp_servers.is_empty() && self.ipython_kernel_provisioner.lock().unwrap().is_none() {
            panic!("ACP MCP servers require the built-in cpython tool");
        }
        let acp_mcp_tool_definitions: Vec<crate::core::extensions::types::ToolDefinition> =
            if self.ipython_kernel_provisioner.lock().unwrap().is_some() {
                let acp_provisioner = self
                    .ipython_kernel_provisioner
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap();
                crate::core::tools::acp_mcp::create_acp_mcp_tool_definitions(
                    &acp_mcp_tool_configs(&acp_servers),
                    acp_provisioner,
                )
                .expect("ACP MCP tool definitions")
                .into_iter()
                .map(crate::core::extensions::types::ToolDefinition::from)
                .collect()
            } else {
                Vec::new()
            };
        let acp_mcp_tool_names: Vec<String> = acp_mcp_tool_definitions
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        self.assert_acp_mcp_tool_names_available(&acp_mcp_tool_names)
            .expect("ACP MCP tool name conflict");
        for name in previous_acp_mcp_tool_names {
            if let Some(allowed) = self.allowed_tool_names.lock().unwrap().as_mut() {
                allowed.remove(&name);
            }
        }
        for name in &acp_mcp_tool_names {
            if let Some(allowed) = self.allowed_tool_names.lock().unwrap().as_mut() {
                allowed.insert(name.clone());
            }
        }
        *self.acp_mcp_tools.lock().unwrap() = acp_mcp_tool_definitions;
        let active = active_tool_names.unwrap_or_else(|| {
            self.base_tools_override
                .as_ref()
                .map(|tools| tools.iter().map(|(name, _)| name.clone()).collect())
                .unwrap_or_else(|| vec!["ipython".to_string()])
        });
        self.refresh_tool_registry(include_all, Some(active.clone()));
        // TS agent-session.ts:10159-10167: prewarm when configured, or whenever
        // we're resuming a session that already has a kernel snapshot - so its
        // state is revived and the model is told what came back before the first
        // turn, rather than a turn later on first use.
        let prewarm_ipython_kernel = self.prewarm_ipython_kernel;
        let has_snapshot = self
            .session_manager
            .lock()
            .unwrap()
            .get_session_artifact_dir()
            .map(|artifact_dir| {
                std::path::Path::new(&crate::core::kernel::state_snapshot::snapshot_path_in(
                    &artifact_dir,
                ))
                .exists()
            })
            .unwrap_or(false);
        if (prewarm_ipython_kernel || has_snapshot) && active.iter().any(|name| name == "ipython") {
            if let Some(provisioner) = self.ipython_kernel_provisioner.lock().unwrap().clone() {
                provisioner.prewarm();
            }
        }
        // Subsequent builds are in-process rebuilds (/reload), not a fresh resume.
        self.ipython_runtime_built.store(true, Ordering::SeqCst);
    }

    pub(super) fn create_kernel_host_handlers(self: &Arc<Self>) -> HostRequestHandlers {
        use crate::core::kernel::shared::KernelError;
        use crate::core::rlm_runtime::*;
        let mut handlers = HostRequestHandlers::new();
        let weak = Arc::downgrade(self);
        handlers.insert("model.info".to_string(), Arc::new(move |_payload: Value| {
            let weak = weak.clone();
            Box::pin(async move {
                let session = weak.upgrade().ok_or_else(|| KernelError::new("Session disposed"))?;
                let model = session.model();
                Ok(serde_json::json!({
                    "id": model.as_ref().map(|model| &model.id),
                    "provider": model.as_ref().map(|model| &model.provider),
                    "input": model.as_ref().map(|model| model.input.clone()).unwrap_or_default(),
                }))
            })
        }));
        let weak = Arc::downgrade(self);
        handlers.insert(
            "rlm.run".to_string(),
            create_rlm_run_host_handler(Arc::new(move |request| {
                let weak = weak.clone();
                Box::pin(async move {
                    let session = weak.upgrade().ok_or("Parent session disposed")?;
                    let kwargs = request.kwargs.as_object().cloned().unwrap_or_default();
                    serde_json::to_value(
                        session
                            .start_rlm_child_run(&request.prompt, &kwargs, request.cell_source_code)
                            .await?,
                    )
                    .map_err(|error| error.to_string())
                })
            })),
        );
        // `bash.completed` (agent-session.ts:10208-10232): a finished background shell
        // injects one canonical completion message onto the steering lane and returns
        // after acceptance. An admission-pause rejection is retried until admission is
        // no longer paused, matching TS 10225-10230.
        let weak = Arc::downgrade(self);
        handlers.insert("bash.completed".to_string(), create_async_bash_completion_host_handler(Arc::new(move |details| {
            let weak = weak.clone();
            Box::pin(async move {
                let Some(session) = weak.upgrade() else { return; };
                let timestamp = now_ms_i64();
                let message = CustomMessage {
                    role: "custom".to_string(),
                    custom_type: ASYNC_BASH_COMPLETION_CUSTOM_TYPE.to_string(),
                    content: CustomMessageContent::Text(format!(
                        "{ASYNC_BASH_COMPLETION_PREVIEW_LABEL}.\nSource: bash\nCommand completed (pid {}, exit code {}).\nCommand: {}\n\nInspect the saved BashHandle with .poll(), .output(), or .tail(), then continue the task.",
                        details.pid as i64,
                        details.exit_code as i64,
                        serde_json::to_string(&details.command).unwrap_or_default()
                    )),
                    display: true,
                    details: serde_json::to_value(AsyncBashCompletionDetails {
                        pid: details.pid as i64,
                        command: details.command.clone(),
                        exit_code: details.exit_code as i64,
                    })
                    .ok(),
                    timestamp,
                };
                let text = match &message.content {
                    CustomMessageContent::Text(text) => text.clone(),
                    CustomMessageContent::Blocks(_) => String::new(),
                };
                let dispose_abort = session.session_action_commit_dispose_abort.clone();
                loop {
                    let committed = Arc::new(AtomicBool::new(false));
                    let committed_flag = committed.clone();
                    let result = session
                        .prompt_injected_message(
                            &text,
                            clone_custom_message(&message),
                            Some(InternalPromptOptions {
                                base: PromptOptions {
                                    streaming_behavior: Some("steer".to_string()),
                                    queue_if_busy: Some(true),
                                    resume_if_idle: Some(true),
                                    suppress_autonomous_continuation: Some(true),
                                    admission_committed: Some(Arc::new(move || {
                                        committed_flag.store(true, Ordering::SeqCst);
                                    })),
                                    ..Default::default()
                                },
                                return_after_accepted: Some(true),
                                ..Default::default()
                            }),
                            None,
                        )
                        .await;
                    match result {
                        Ok(()) => return,
                        Err(error) => {
                            // TS 10226: only an uncommitted admission pause is retried.
                            if committed.load(Ordering::SeqCst)
                                || !error.contains("session input admission is paused")
                            {
                                return;
                            }
                            while !session.session_input_admission_pauses.lock().unwrap().is_empty()
                                && !dispose_abort.is_cancelled()
                            {
                                // TS 10227-10229: wait for the pause to be released.
                                let signal = dispose_abort.clone();
                                let _ = session
                                    .wait_for_session_activity_change(Some(&signal))
                                    .await;
                            }
                            if dispose_abort.is_cancelled() {
                                return;
                            }
                        }
                    }
                }
            })
        })));
        let weak = Arc::downgrade(self);
        handlers.insert(
            "rlm.create_session".to_string(),
            create_rlm_create_session_host_handler(Arc::new(move |request| {
                let weak = weak.clone();
                Box::pin(async move {
                    weak.upgrade()
                        .ok_or("Parent session disposed")?
                        .create_rlm_session(
                            &request.prompt,
                            &request.kwargs.as_object().cloned().unwrap_or_default(),
                        )
                        .await
                })
            })),
        );
        let weak = Arc::downgrade(self);
        handlers.insert(
            "rlm.find_models".into(),
            create_rlm_find_models_host_handler(Arc::new(move |query, limit| {
                let weak = weak.clone();
                Box::pin(async move {
                    weak.upgrade()
                        .ok_or("Parent session disposed")?
                        .find_rlm_models(&query, limit as i64)
                        .await
                })
            })),
        );
        let weak = Arc::downgrade(self);
        handlers.insert(
            "rlm.list_subagents".into(),
            create_rlm_list_subagents_host_handler(Arc::new(move || {
                let weak = weak.clone();
                Box::pin(async move {
                    weak.upgrade()
                        .ok_or("Parent session disposed")?
                        .list_rlm_subagents()
                        .await
                })
            })),
        );
        let weak = Arc::downgrade(self);
        handlers.insert(
            "rlm.delete_subagent".into(),
            create_rlm_delete_subagent_host_handler(Arc::new(move |target| {
                let weak = weak.clone();
                Box::pin(async move {
                    weak.upgrade()
                        .ok_or("Parent session disposed")?
                        .delete_rlm_subagent(&target)
                        .await
                })
            })),
        );
        let weak = Arc::downgrade(self);
        handlers.insert("rlm.lifecycle_capabilities".into(), Arc::new(move |payload| {
            let weak = weak.clone();
            Box::pin(async move { weak.upgrade().ok_or_else(|| KernelError::new("Parent disposed"))?
                .lifecycle_capabilities(payload).await.map_err(KernelError::new) })
        }));
        let weak = Arc::downgrade(self);
        handlers.insert("rlm.active_execution".into(), Arc::new(move |payload| {
            let weak = weak.clone();
            Box::pin(async move {
                let parent = weak.upgrade().ok_or_else(|| KernelError::new("Parent disposed"))?;
                let target = payload.get("target").and_then(Value::as_str).ok_or_else(|| KernelError::new("target required"))?;
                parent.active_child_execution(target).map_err(KernelError::new)
            })
        }));
        let weak = Arc::downgrade(self);
        handlers.insert("rlm.send_active_message".into(), Arc::new(move |payload| {
            let weak = weak.clone();
            Box::pin(async move { weak.upgrade().ok_or_else(|| KernelError::new("Parent disposed"))?
                .send_active_child_message(payload).map_err(KernelError::new) })
        }));
        let weak = Arc::downgrade(self);
        handlers.insert("rlm.stop_subagent".into(), Arc::new(move |payload| {
            let weak = weak.clone();
            Box::pin(async move {
                let parent = weak.upgrade().ok_or_else(|| KernelError::new("Parent disposed"))?;
                let selector = payload.get("target").and_then(Value::as_str).ok_or_else(|| KernelError::new("target required"))?;
                let timeout = payload.get("timeout_ms").and_then(Value::as_u64).ok_or_else(|| KernelError::new("timeout_ms required"))?;
                parent.stop_retained_child(selector, timeout).await.map_err(KernelError::new)
            })
        }));
        let weak = Arc::downgrade(self);
        handlers.insert("rlm.resume_subagent".into(), Arc::new(move |payload| {
            let weak = weak.clone();
            Box::pin(async move {
                let parent = weak.upgrade().ok_or_else(|| KernelError::new("Parent disposed"))?;
                let field = |name| payload.get(name).and_then(Value::as_str)
                    .ok_or_else(|| KernelError::new(format!("{name} required")));
                parent.resume_retained_audit(field("target")?, field("stop_generation")?, field("prompt")?)
                    .map_err(KernelError::new)
            })
        }));
        let weak = Arc::downgrade(self);
        handlers.insert(
            "rlm.collect".into(),
            crate::core::rlm_runtime::create_rlm_collect_host_handler(Arc::new(
                move |targets, timeout_ms| {
                    let weak = weak.clone();
                    Box::pin(async move {
                        weak.upgrade()
                            .ok_or("Parent session disposed")?
                            .collect_rlm_children(&targets, timeout_ms)
                            .await
                    })
                },
            )),
        );
        // TS agent-session.ts:10267-10272: the agent_message handlers install only
        // when the controller exists AND the agent-message skill is visible to the
        // model (disableModelInvocation skills are not kernel-reachable).
        let visible_kernel_skill_names: std::collections::HashSet<String> = self
            .model_visible_skills()
            .iter()
            .filter(|skill| match skill {
                crate::core::skills::Skill::Markdown(value) => !value.base.disable_model_invocation,
                crate::core::skills::Skill::Python(value) => !value.base.disable_model_invocation,
            })
            .map(|skill| skill.name().to_string())
            .collect();
        if self.agent_message_controller.is_some()
            && visible_kernel_skill_names
                .contains(crate::core::agent_messages::AGENT_MESSAGE_SKILL_NAME)
        {
            handlers.extend(create_agent_message_host_handlers(Arc::new(
                subagent_runs::SessionMessageController(Arc::downgrade(self)),
            )));
        }
        if let Some(controller) = &self.agent_observe_controller {
            handlers.extend(create_agent_observe_host_handlers(controller.clone()));
        }
        for operation in [
            "goal.get",
            "goal.create",
            "goal.complete",
            "compact.run",
            "compact.status",
            "refine.run",
            "refine.status",
            "rlm_heartbeat.list",
            "rlm_heartbeat.create",
            "rlm_heartbeat.update",
            "rlm_heartbeat.delete",
        ] {
            if operation.starts_with("goal.") && !self.include_goals {
                continue;
            }
            if operation.starts_with("compact.") && !self.include_compact_skill {
                continue;
            }
            // TS agent-session.ts:10252-10255: refine handlers are gated on
            // `_autoRefineAllowedForSession()`.
            if operation.starts_with("refine.") && !self.auto_refine_allowed_for_session() {
                continue;
            }
            if operation.starts_with("rlm_heartbeat.")
                && self.rlm_heartbeat_controller.lock().unwrap().is_none()
            {
                continue;
            }
            let weak = Arc::downgrade(self);
            handlers.insert(
                operation.to_string(),
                Arc::new(move |payload: Value| {
                    let weak = weak.clone();
                    Box::pin(async move {
                        let session = weak
                            .upgrade()
                            .ok_or_else(|| KernelError::new("Session disposed"))?;
                        let result = if operation.starts_with("goal.") {
                            session
                                .handle_goal_host_request(operation, Some(&payload))
                                .and_then(|response| {
                                    serde_json::to_value(response)
                                        .map_err(|error| error.to_string())
                                })
                        } else if operation.starts_with("compact.") {
                            session.handle_compact_host_request(operation, Some(&payload))
                        } else if operation.starts_with("refine.") {
                            session.handle_refine_host_request(operation, Some(&payload))
                        } else {
                            session.handle_rlm_heartbeat_host_request(operation, Some(&payload))
                        };
                        result.map_err(KernelError::new)
                    })
                }),
            );
        }
        // TS 10337-10339: `if (this._mcpManager) Object.assign(handlers, this._mcpManager.hostHandlers())`.
        // The MCP manager owns mcp.refresh/mcp.config/mcp.begin_login; a manager-less
        // session exposes none of them.
        if let Some(manager) = &self.mcp_manager {
            let merged = manager.lock().unwrap().host_handlers();
            handlers.extend(merged);
        }
        handlers
    }

    pub async fn reload_with_options(
        self: &Arc<Self>,
        rebind: Option<ExtensionBindings>,
    ) -> Result<(), String> {
        // `await emitSessionShutdownEvent(this._extensionRunner, { type:
        // "session_shutdown", reason: "reload" })` (agent-session.ts:10344-10348).
        // The helper emits only when handlers exist and returns false otherwise.
        if let Some(runner) = self.extension_runner() {
            let _ = crate::core::extensions::runner::emit_session_shutdown_event(
                &runner,
                crate::core::extensions::types::ExtensionEvent::SessionShutdown(
                    crate::core::extensions::types::SessionShutdownPayload {
                        reason: "reload".to_string(),
                        target_session_file: None,
                    },
                ),
            )
            .await;
        }
        self.resource_loader.reload().await;
        self.build_runtime(Some(self.get_active_tool_names()), true);
        if let Some(bindings) = rebind {
            self.bind_extensions(&bindings).await?;
        }
        self.extend_resources_from_extensions("reload").await
    }

    /// `_rlmKernelEnv()`.
    pub(super) fn rlm_kernel_env(&self) -> HashMap<String, String> {
        let mut env = HashMap::from([
            ("RLM_DEPTH".to_string(), self.rlm_depth.to_string()),
            (
                "RLM_MAX_DEPTH".to_string(),
                self.rlm_max_depth().to_string(),
            ),
            // TS agent-session.ts:10382: getGlobalHarnessStateDir() is called with no
            // argument, so it resolves to the process-level getAgentDir() (absolute),
            // never the scoped session agent dir.
            (
                "RLM_GLOBAL_HARNESS_STATE_DIR".to_string(),
                get_global_harness_state_dir(&crate::config::get_agent_dir()),
            ),
        ]);
        if let Some(dir) = self.rlm_session_dir_for_reading() {
            env.insert("RLM_SESSION_DIR".to_string(), dir.clone());
            // TS agent-session.ts:10390: `this._localHarnessStateDir() ??
            // getLocalHarnessStateDir(rlmSessionDir)!` - the session artifact dir
            // wins; ephemeral sessions fall back to the RLM session dir.
            if let Some(local) = self
                .local_harness_state_dir()
                .or_else(|| get_local_harness_state_dir(Some(&dir)))
            {
                env.insert("RLM_HARNESS_STATE_DIR".to_string(), local);
            }
        }
        self.add_websearch_key_env(&mut env);
        env
    }

    /// `_addWebsearchKeyEnv(env)` (agent-session.ts:10396-10417).
    pub(super) fn add_websearch_key_env(&self, env: &mut HashMap<String, String>) {
        if let Some(agent_dir) = &self.agent_dir {
            env.insert(
                "PRIME_AGENT_CODING_AGENT_DIR".to_string(),
                agent_dir.clone(),
            );
        }
        if std::env::var(SERPER_ENV_VAR)
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
        {
            return;
        }
        // Inject only when a websearch skill (bundled or custom) is actually loaded,
        // so the key isn't exposed to kernels that can't use it.
        let websearch_loaded = self
            .resource_loader
            .get_skills()
            .skills
            .iter()
            .any(|skill| skill.name() == WEBSEARCH_SKILL_NAME);
        if !websearch_loaded {
            return;
        }
        let credential = self
            .model_registry
            .lock()
            .unwrap()
            .auth_storage()
            .get(SERPER_CREDENTIAL_ID);
        let Some(crate::core::auth_storage::AuthCredential::ApiKey { key, .. }) = credential else {
            return;
        };
        let resolved = crate::core::resolve_config_value::resolve_config_value(&key)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        if let Some(resolved) = resolved {
            env.insert(SERPER_ENV_VAR.to_string(), resolved);
        }
    }

    /// `_createChildRlmSessionDir()`.
    pub(super) fn create_child_rlm_session_dir(&self) -> Result<String, String> {
        let parent = self
            .rlm_session_dir_for_reading()
            .map(Ok)
            .unwrap_or_else(|| self.create_ephemeral_rlm_session_dir())?;
        std::fs::create_dir_all(&parent).map_err(|error| error.to_string())?;
        for _ in 0..100 {
            let id = uuid::Uuid::new_v4().to_string();
            let dir = PathBuf::from(&parent).join(format!("sub-{}", &id[..8]));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(dir.to_string_lossy().into_owned()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.to_string()),
            }
        }
        Err("Unable to create unique RLM child session directory".to_string())
    }

    /// `_createEphemeralRlmSessionDir()`.
    pub(super) fn create_ephemeral_rlm_session_dir(&self) -> Result<String, String> {
        let dir = PathBuf::from(std::env::temp_dir())
            .join(format!("prime-agent-rlm-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        Ok(dir.to_string_lossy().to_string())
    }

    /// `_contextTokensForCurrentMessages()`.
    pub fn context_tokens_for_current_messages(&self) -> Option<f64> {
        self.find_last_assistant_message()
            .map(|message| calculate_context_tokens(&message.usage))
    }

    /// `setCurrentRecap(recap)`.
    pub fn set_current_recap(&self, recap: Option<String>) {
        let changed = {
            let mut current = self.recap.lock().unwrap();
            if *current == recap {
                false
            } else {
                *current = recap.clone();
                true
            }
        };
        if changed {
            self.emit(AgentSessionEvent::RecapUpdate { recap });
        }
    }

    /// `get repliedToParentSinceTask()`.
    pub fn replied_to_parent_since_task(&self) -> Option<bool> {
        *self.replied_to_parent_since_task.lock().unwrap()
    }

    /// `getCurrentRecap()`.
    pub fn get_current_recap(&self) -> Option<String> {
        self.recap.lock().unwrap().clone()
    }

    /// `_findAssistantEntryForMessage(message)`.
    pub(super) fn find_assistant_entry_for_message(
        &self,
        message: &AssistantMessage,
    ) -> Option<SessionEntry> {
        // The TypeScript compares `entry.message === message` (object identity);
        // Rust values have no identity, so the same message compares through
        // `assistant_message_key`, exactly like the other identity-keyed maps.
        let key = assistant_message_key(message);
        self.session_manager
            .lock()
            .unwrap()
            .get_entries()
            .into_iter()
            .find(|entry| {
                if entry.get("type").and_then(Value::as_str) != Some("message") {
                    return false;
                }
                let Some(value) = entry.get("message") else {
                    return false;
                };
                match agent_message_from_value(value) {
                    AgentMessage::Message(Message::Assistant(assistant)) => {
                        assistant_message_key(&assistant) == key
                    }
                    _ => false,
                }
            })
    }

    /// `_createRlmSubagentRuntimeOptions(options)`.
    pub(super) fn create_rlm_subagent_runtime_options(
        self: &Arc<Self>,
        options: RlmSubagentRuntimeOptionsInput,
    ) -> Result<CreateRlmSubagentRuntimeOptions, String> {
        let thinking_level = options
            .thinking_level
            .unwrap_or_else(|| self.thinking_level());
        Ok(CreateRlmSubagentRuntimeOptions {
            parent_session: self.clone(),
            id: options.id.clone(),
            prompt: options.prompt,
            session_name: options.session_name,
            session_dir: options.session_dir,
            model: options.model,
            thinking_level,
            service_tier: self.service_tier(),
            scoped_models: self
                .scoped_models
                .iter()
                .map(|entry| crate::core::rlm_runtime::ScopedModelEntry {
                    model: entry.model.clone(),
                    thinking_level: entry.thinking_level.clone(),
                })
                .collect(),
            active_tool_names: self.get_active_tool_names(),
            allowed_tool_names: self
                .allowed_tool_names
                .lock()
                .unwrap()
                .clone()
                .map(|names| names.into_iter().collect()),
            custom_tools: self.custom_tools.clone(),
            include_goals: self.include_goals,
            include_compact_skill: self.include_compact_skill,
            rlm_depth: (self.rlm_depth + 1) as f64,
            rlm_max_depth: self.rlm_max_depth() as f64,
            rlm_parent_node_id: options.id,
            spawned_by_request_id: options.spawned_by_request_id,
            spawn_code: options.spawn_code,
            on_session_published: None,
        })
    }

    /// `_createRlmSubagentRuntime(options)`.
    pub(super) async fn create_rlm_subagent_runtime(
        self: &Arc<Self>,
        options: CreateRlmSubagentRuntimeOptions,
    ) -> Result<RlmSubagentRuntime, String> {
        let host = self.subagent_runtime_host.lock().unwrap().clone();
        match host {
            Some(host) => host.create_rlm_subagent_runtime(options).await,
            None => self.create_inline_rlm_subagent_runtime(options),
        }
    }

    /// `_createInlineRlmSubagentRuntime(options)`.
    pub(super) fn create_inline_rlm_subagent_runtime(
        self: &Arc<Self>,
        options: CreateRlmSubagentRuntimeOptions,
    ) -> Result<RlmSubagentRuntime, String> {
        let mut manager = SessionManager::create(&self.cwd, Some(&options.session_dir))?;
        manager.append_model_change(&options.model.provider, &options.model.id)?;
        manager.append_thinking_level_change(&thinking_level_name(&options.thinking_level))?;
        manager.append_service_tier_change(&options.service_tier)?;
        let mut state = self.agent.state();
        state.messages.clear();
        state.tools = Some(Vec::new());
        state.system_prompt.clear();
        state.model = options.model;
        state.thinking_level = options.thinking_level;
        state.service_tier = options.service_tier;
        let agent = pi_agent_core::agent::Agent::new(pi_agent_core::agent::AgentOptions {
            initial_state: Some(state),
            stream_fn: Some(self.agent.stream_fn()),
            session_id: Some(manager.get_session_id()),
            ..Default::default()
        });
        let child = AgentSession::new(AgentSessionConfig {
            agent: Arc::new(agent),
            session_manager: Arc::new(Mutex::new(manager)),
            settings_manager: self.settings_manager.clone(),
            service_tier_preference: None,
            cwd: self.cwd.clone(),
            agent_dir: self.agent_dir.clone(),
            scoped_models: Some(
                options
                    .scoped_models
                    .into_iter()
                    .map(|entry| ScopedModel {
                        model: entry.model,
                        thinking_level: entry.thinking_level,
                    })
                    .collect(),
            ),
            resource_loader: self.resource_loader.clone(),
            custom_tools: Some(options.custom_tools),
            model_registry: self.model_registry.clone(),
            initial_active_tool_names: Some(options.active_tool_names),
            allowed_tool_names: options.allowed_tool_names,
            include_goals: Some(options.include_goals),
            include_compact_skill: Some(options.include_compact_skill),
            agent_message_controller: None,
            agent_observe_controller: None,
            rlm_heartbeat_controller: None,
            mcp_manager: None,
            base_tools_override: self.base_tools_override.clone(),
            extension_runner_ref: None,
            session_start_event: Some(
                serde_json::json!({"type":"session_start", "reason":"startup"}),
            ),
            rlm_depth: Some(options.rlm_depth as i64),
            rlm_max_depth: Some(options.rlm_max_depth as i64),
            rlm_session_dir: Some(options.session_dir),
            rlm_parent_node_id: Some(options.rlm_parent_node_id),
            rlm_parent_agent: Some(self.session_name().unwrap_or_else(|| self.session_id())),
            semantic_parent_session_id: Some(self.session_id()),
            semantic_spawned_by_request_id: options.spawned_by_request_id,
            subagent_runtime_host: None,
            autonomous: None,
            prewarm_ipython_kernel: None,
            auto_refine_reviewer: None,
            serialized_refine: None,
            initial_goal: None,
        })?;
        child.set_session_name(&options.session_name)?;
        if let Some(published) = options.on_session_published {
            published(&child);
        }
        Ok(RlmSubagentRuntime { session: child })
    }

    /// `_abandonRlmRunForQuiescence(run)`.
    pub(super) fn abandon_rlm_run_for_quiescence(&self, run: &RlmChildRun) {
        self.abandoned_rlm_quiescence_child_ids
            .lock()
            .unwrap()
            .insert(run.id.clone());
    }

    /// `_cancelActiveRlmChildRuns(reason)`.
    pub(super) fn cancel_active_rlm_child_runs(&self, reason: &str) {
        let runs: Vec<Arc<Mutex<RlmChildRun>>> = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect();
        for run in runs {
            let run_snapshot = run.lock().unwrap().clone();
            let _ = self.cancel_rlm_child_run(&run_snapshot, reason);
        }
    }

    /// `_cancelRlmChildRun(run, reason)`.
    pub(super) fn cancel_rlm_child_run(&self, run: &RlmChildRun, reason: &str) -> bool {
        let current = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(&run.id)
            .cloned();
        let Some(current) = current else {
            return false;
        };
        let abort = {
            let mut current = current.lock().unwrap();
            if matches!(current.status.as_str(), "done" | "error" | "cancelled") {
                return false;
            }
            current.status = RLM_CHILD_AGENT_STATUS_CANCELLED.to_string();
            current.error = Some(reason.to_string());
            current.abort.clone()
        };
        abort();
        true
    }

    /// `getRlmChildRunStatus(childId)`.
    pub fn get_rlm_child_run_status(&self, child_id: &str) -> Option<RlmChildAgentStatus> {
        self.active_rlm_child_runs
            .lock()
            .unwrap()
            .get(child_id)
            .map(|run| run.lock().unwrap().status.clone())
    }

    /// `_currentActiveSessionId()`.
    pub(super) async fn current_active_session_id(&self) -> Option<String> {
        match &self.agent_message_controller {
            Some(controller) => controller
                .list_agents()
                .await
                .ok()
                .flatten()
                .and_then(|listed| listed.current.map(|current| current.active_session_id)),
            None => None,
        }
    }

    /// `_awaitPendingRlmChildPublication(selector)`.
    pub(super) async fn await_pending_rlm_child_publication(
        &self,
        selector: &str,
    ) -> Result<Option<String>, String> {
        let run = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .values()
            .find(|run| {
                let run = run.lock().unwrap();
                run.id == selector || run.session_name == selector
            })
            .cloned();
        let Some(run) = run else {
            return Ok(None);
        };
        let publication = run.lock().unwrap().publication.clone();
        publication.wait().await?;
        let child = run.lock().unwrap().session.clone();
        Ok(child.map(|child| child.session_id()))
    }

    /// `listRlmSubagents()` includes daemon-owned passive children.
    pub async fn list_rlm_subagents(self: &Arc<Self>) -> Result<RlmListSubagentsResult, String> {
        let listed = match &self.agent_message_controller {
            Some(controller) => controller.list_agents().await?,
            None => None,
        };
        Ok(self.build_rlm_subagent_list(listed))
    }

    /// `_buildRlmSubagentList(listedAgents?)`.
    pub(super) fn build_rlm_subagent_list(
        &self,
        listed_agents: Option<AgentSessionMessageListResult>,
    ) -> RlmListSubagentsResult {
        let mut daemon_children: HashMap<String, AgentSessionMessageAgentSummary> = HashMap::new();
        let parent_active_session_id = listed_agents
            .as_ref()
            .and_then(|listed| listed.current.as_ref())
            .map(|current| current.active_session_id.clone());
        if let Some(parent_active_session_id) = parent_active_session_id {
            for agent in listed_agents
                .as_ref()
                .map(|listed| listed.agents.iter())
                .into_iter()
                .flatten()
            {
                if agent.runtime_kind.as_deref() != Some(RUNTIME_KIND_SUBAGENT) {
                    continue;
                }
                if agent.parent_active_session_id.as_deref()
                    != Some(parent_active_session_id.as_str())
                {
                    continue;
                }
                if let Some(child_id) = agent.rlm_child_id.clone() {
                    daemon_children.insert(child_id, agent.clone());
                }
            }
        }

        let mut subagents: Vec<RlmSubagentRegistryEntry> = Vec::new();
        let mut recorded: HashSet<String> = HashSet::new();
        let runs: Vec<RlmChildRun> = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .values()
            .map(|run| run.lock().unwrap().clone())
            .collect();
        for run in runs.iter() {
            if self
                .deleting_rlm_children
                .lock()
                .unwrap()
                .contains_key(&run.id)
                || run.detached_deletion.is_some()
                || run.status == RLM_CHILD_AGENT_STATUS_CANCELLED
            {
                continue;
            }
            let daemon_child = daemon_children.get(&run.id);
            subagents.push(RlmSubagentRegistryEntry {
                rlm_child_id: run.id.clone(),
                active_session_id: daemon_child.map(|child| child.active_session_id.clone()),
                session_id: daemon_child
                    .map(|child| child.session_id.clone())
                    .or_else(|| run.session.as_ref().map(|session| session.session_id())),
                session_name: daemon_child
                    .and_then(|child| child.session_name.clone())
                    .or_else(|| {
                        run.session
                            .as_ref()
                            .and_then(|session| session.session_name())
                    })
                    .unwrap_or_else(|| run.session_name.clone()),
                session_dir: run.session_dir.clone(),
                status: match run.status.as_str() {
                    "done" => RLM_SUBAGENT_STATUS_COMPLETED.to_string(),
                    "error" => RLM_SUBAGENT_STATUS_ERROR.to_string(),
                    _ => RLM_SUBAGENT_STATUS_RUNNING.to_string(),
                },
            });
            recorded.insert(run.id.clone());
        }
        for (child_id, retained) in self.rlm_child_sessions.lock().unwrap().iter() {
            if self
                .deleting_rlm_children
                .lock()
                .unwrap()
                .contains_key(child_id)
                || recorded.contains(child_id)
                || self
                    .rlm_child_cleanup_failures
                    .lock()
                    .unwrap()
                    .contains_key(child_id)
            {
                continue;
            }
            let daemon_child = daemon_children.get(child_id);
            let session_dir = match retained.session.rlm_session_dir.clone() {
                Some(session_dir) => session_dir,
                None => continue,
            };
            subagents.push(RlmSubagentRegistryEntry {
                rlm_child_id: child_id.clone(),
                active_session_id: daemon_child.map(|child| child.active_session_id.clone()),
                session_id: daemon_child
                    .map(|child| child.session_id.clone())
                    .or_else(|| Some(retained.session.session_id())),
                session_name: daemon_child
                    .and_then(|child| child.session_name.clone())
                    .or_else(|| retained.session.session_name())
                    .unwrap_or_else(|| create_default_rlm_subagent_session_name("", child_id)),
                session_dir,
                status: RLM_SUBAGENT_STATUS_COMPLETED.to_string(),
            });
            recorded.insert(child_id.clone());
        }
        for (child_id, daemon_child) in daemon_children.iter() {
            if recorded.contains(child_id)
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
                || self
                    .rlm_child_cleanup_failures
                    .lock()
                    .unwrap()
                    .contains_key(child_id)
            {
                continue;
            }
            let Some(session_dir) = daemon_child.session_dir.clone() else {
                continue;
            };
            subagents.push(RlmSubagentRegistryEntry {
                rlm_child_id: child_id.clone(),
                active_session_id: Some(daemon_child.active_session_id.clone()),
                session_id: Some(daemon_child.session_id.clone()),
                session_name: daemon_child
                    .session_name
                    .clone()
                    .unwrap_or_else(|| create_default_rlm_subagent_session_name("", child_id)),
                session_dir,
                status: if daemon_child.rlm_child_registry_status.as_deref()
                    == Some(RLM_SUBAGENT_STATUS_COMPLETED)
                {
                    RLM_SUBAGENT_STATUS_COMPLETED.to_string()
                } else {
                    RLM_SUBAGENT_STATUS_ERROR.to_string()
                },
            });
        }
        RlmListSubagentsResult { subagents }
    }

    /// `_rlmSubagentMatchesTarget(entry, target)`.
    pub(super) fn rlm_subagent_matches_target(
        &self,
        entry: &RlmSubagentRegistryEntry,
        target: &str,
    ) -> bool {
        entry.rlm_child_id == target
            || entry.active_session_id.as_deref() == Some(target)
            || entry.session_id.as_deref() == Some(target)
            || entry.session_name == target
    }

    /// `_rlmSubtreeSessions()` - this session plus every live direct or nested
    /// child, de-duplicated by identity so a child in both maps is visited once.
    pub(super) fn rlm_subtree_sessions(self: &Arc<Self>) -> Vec<Arc<AgentSession>> {
        let mut visited: Vec<Arc<AgentSession>> = vec![self.clone()];
        let mut stack: Vec<Arc<AgentSession>> = vec![self.clone()];
        let mut sessions: Vec<Arc<AgentSession>> = Vec::new();
        while let Some(session) = stack.pop() {
            sessions.push(session.clone());
            let children: Vec<Arc<AgentSession>> = session
                .active_rlm_child_runs
                .lock()
                .unwrap()
                .values()
                .filter_map(|run| run.lock().unwrap().session.clone())
                .chain(
                    session
                        .rlm_child_sessions
                        .lock()
                        .unwrap()
                        .values()
                        .map(|retained| retained.session.clone()),
                )
                .collect();
            for child in children {
                if visited
                    .iter()
                    .any(|candidate| Arc::ptr_eq(candidate, &child))
                {
                    continue;
                }
                visited.push(child.clone());
                stack.push(child);
            }
        }
        sessions
    }

    /// `_resolveDirectRlmSubagent(target)`.
    pub(super) async fn resolve_direct_rlm_subagent(
        self: &Arc<Self>,
        target: &str,
    ) -> Result<RlmSubagentRegistryEntry, String> {
        let mut candidates = self.list_rlm_subagents().await?.subagents;
        candidates.extend(
            self.rlm_child_cleanup_failures
                .lock()
                .unwrap()
                .values()
                .cloned(),
        );
        let matches: Vec<RlmSubagentRegistryEntry> = candidates
            .into_iter()
            .filter(|entry| self.rlm_subagent_matches_target(entry, target))
            .collect();
        if matches.is_empty() {
            return Err(format!(
                "No direct RLM subagent matches \"{target}\" in the current parent session"
            ));
        }
        if matches.len() > 1 {
            return Err(format!(
                "RLM subagent selector \"{target}\" is ambiguous in the current parent session"
            ));
        }
        Ok(matches.into_iter().next().expect("checked length"))
    }

    /// `deleteInactiveRlmSubagent(childId, isExternallyRunning)`.
    pub async fn delete_inactive_rlm_subagent(
        self: &Arc<Self>,
        child_id: &str,
        is_externally_running: Arc<dyn Fn() -> bool + Send + Sync>,
    ) -> Result<String, String> {
        for owner in self.rlm_subtree_sessions() {
            let is_running = {
                let owner = owner.clone();
                let child_id = child_id.to_string();
                let is_externally_running = is_externally_running.clone();
                Arc::new(move || {
                    let status = owner
                        .active_rlm_child_runs
                        .lock()
                        .unwrap()
                        .get(&child_id)
                        .map(|run| run.lock().unwrap().status.clone());
                    matches!(status.as_deref(), Some("queued") | Some("running"))
                        || is_externally_running()
                }) as Arc<dyn Fn() -> bool + Send + Sync>
            };
            if is_running() {
                return Ok(DELETE_OUTCOME_RUNNING_LITERAL.to_string());
            }
            let mut candidates = owner.list_rlm_subagents().await?.subagents;
            candidates.extend(
                owner
                    .rlm_child_cleanup_failures
                    .lock()
                    .unwrap()
                    .values()
                    .cloned(),
            );
            let subagent = candidates
                .into_iter()
                .find(|entry| entry.rlm_child_id == child_id);
            let Some(subagent) = subagent else { continue };
            if is_running() {
                return Ok(DELETE_OUTCOME_RUNNING_LITERAL.to_string());
            }
            let is_running_for_deletion = is_running.clone();
            let subagent_for_deletion = subagent.clone();
            let result = owner
                .track_rlm_subagent_deletion(
                    &subagent,
                    Box::new(move |owner| {
                        let is_running = is_running_for_deletion.clone();
                        let subagent = subagent_for_deletion.clone();
                        Box::pin(async move {
                            if is_running() {
                                return Ok(RlmDeleteSubagentResult {
                                    subagent,
                                    outcome: Some(DELETE_OUTCOME_SKIPPED_RUNNING.to_string()),
                                });
                            }
                            owner.delete_resolved_rlm_subagent(&subagent).await
                        })
                    }),
                )
                .await?;
            return Ok(
                if result.outcome.as_deref() == Some(DELETE_OUTCOME_SKIPPED_RUNNING) {
                    DELETE_OUTCOME_RUNNING_LITERAL.to_string()
                } else {
                    DELETE_OUTCOME_DELETED.to_string()
                },
            );
        }
        Ok(DELETE_OUTCOME_NOT_FOUND_LITERAL.to_string())
    }

    /// `deleteRlmSubagent(target)`.
    pub async fn delete_rlm_subagent(
        self: &Arc<Self>,
        target: &str,
    ) -> Result<RlmDeleteSubagentResult, String> {
        // Running and retained children can be reserved synchronously. This keeps
        // them hidden immediately while the async daemon listing checks for a
        // conflicting passive selector.
        // REPAIR CURSOR: `this._deletingRlmChildren` in the TypeScript maps
        // selector -> `{ subagent, promise }`, so the `inFlight` ambiguity guard
        // below cannot be evaluated here: `AgentSession::deleting_rlm_children`
        // (core/agent_session.rs:2120) stores `Arc<AgentMessageDeferred>` only.
        // Store `{ subagent, promise }` there (or add a sibling map) and filter it
        // by `rlm_subagent_matches_target` before `local_matches`.
        let mut local_matches = self.build_rlm_subagent_list(None).subagents;
        local_matches.extend(
            self.rlm_child_cleanup_failures
                .lock()
                .unwrap()
                .values()
                .cloned(),
        );
        let local_matches: Vec<RlmSubagentRegistryEntry> = local_matches
            .into_iter()
            .filter(|entry| self.rlm_subagent_matches_target(entry, target))
            .collect();
        let matching_child_ids: HashSet<String> = local_matches
            .iter()
            .map(|subagent| subagent.rlm_child_id.clone())
            .collect();
        if matching_child_ids.len() > 1 || local_matches.len() > 1 {
            return Err(format!(
                "RLM subagent selector \"{target}\" is ambiguous in the current parent session"
            ));
        }
        if let Some(subagent) = local_matches.into_iter().next() {
            let target = target.to_string();
            let subagent_for_deletion = subagent.clone();
            return self
                .track_rlm_subagent_deletion(
                    &subagent,
                    Box::new(move |owner| {
                        let subagent = subagent_for_deletion.clone();
                        let target = target.clone();
                        Box::pin(async move {
                            let listed_subagents =
                                owner.build_rlm_subagent_list(None).subagents;
                            let passive_matches: Vec<RlmSubagentRegistryEntry> = listed_subagents
                                .into_iter()
                                .filter(|entry| {
                                    entry.rlm_child_id != subagent.rlm_child_id
                                        && owner.rlm_subagent_matches_target(entry, &target)
                                })
                                .collect();
                            if !passive_matches.is_empty() {
                                return Err(format!(
                                    "RLM subagent selector \"{target}\" is ambiguous in the current parent session"
                                ));
                            }
                            owner.delete_resolved_rlm_subagent(&subagent).await
                        })
                    }),
                )
                .await;
        }

        let mut direct_matches = self.list_rlm_subagents().await?.subagents;
        direct_matches.extend(
            self.rlm_child_cleanup_failures
                .lock()
                .unwrap()
                .values()
                .cloned(),
        );
        let direct_matches: Vec<RlmSubagentRegistryEntry> = direct_matches
            .into_iter()
            .filter(|entry| self.rlm_subagent_matches_target(entry, target))
            .collect();
        let direct_child_ids: HashSet<String> = direct_matches
            .iter()
            .map(|subagent| subagent.rlm_child_id.clone())
            .collect();
        if direct_child_ids.len() > 1 {
            return Err(format!(
                "RLM subagent selector \"{target}\" is ambiguous in the current parent session"
            ));
        }
        let subagent = match direct_matches.into_iter().next() {
            Some(subagent) => subagent,
            None => self.resolve_direct_rlm_subagent(target).await?,
        };
        let subagent_for_deletion = subagent.clone();
        self.track_rlm_subagent_deletion(
            &subagent,
            Box::new(move |owner| {
                let subagent = subagent_for_deletion.clone();
                Box::pin(async move { owner.delete_resolved_rlm_subagent(&subagent).await })
            }),
        )
        .await
    }

    /// `_trackRlmSubagentDeletion(subagent, startDeletion)`.
    ///
    /// Repeated calls share admission; detached cleanup keeps the selector reserved
    /// until the run releases its separate deletion reservation.
    pub(super) async fn track_rlm_subagent_deletion(
        self: &Arc<Self>,
        subagent: &RlmSubagentRegistryEntry,
        start_deletion: Box<
            dyn FnOnce(Arc<Self>) -> BoxFuture<Result<RlmDeleteSubagentResult, String>> + Send,
        >,
    ) -> Result<RlmDeleteSubagentResult, String> {
        let child_id = subagent.rlm_child_id.clone();
        let (deletion, existing) = {
            let _admission = self.rlm_child_lifecycle_admission.lock().unwrap();
            let mut pending = self.deleting_rlm_children.lock().unwrap();
            match pending.get(&child_id) {
                Some(deletion) => (deletion.clone(), true),
                None => {
                    let deletion = Arc::new(create_agent_message_deferred());
                    pending.insert(child_id.clone(), deletion.clone());
                    (deletion, false)
                }
            }
        };
        if existing {
            deletion.wait().await?;
            return Ok(RlmDeleteSubagentResult {
                subagent: subagent.clone(),
                outcome: None,
            });
        }
        let result = start_deletion(self.clone()).await;
        match &result {
            Ok(_) => deletion.resolve(),
            Err(error) => deletion.reject(error.clone()),
        }
        // `finally`: release the reservation unless the detached run still owns
        // the selector until its deletion reservation settles.
        let detached = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(&child_id)
            .map(|run| run.lock().unwrap().detached_deletion.is_some())
            .unwrap_or(false);
        if detached {
            let reservation = self
                .active_rlm_child_runs
                .lock()
                .unwrap()
                .get(&child_id)
                .map(|run| run.lock().unwrap().deletion_reservation.clone());
            if let Some(reservation) = reservation {
                let session = self.clone();
                let deletion = deletion.clone();
                tokio::spawn(async move {
                    let _ = reservation.wait().await;
                    let accepted = session
                        .deleting_rlm_children
                        .lock()
                        .unwrap()
                        .get(&child_id)
                        .cloned();
                    if accepted
                        .map(|accepted| Arc::ptr_eq(&accepted, &deletion))
                        .unwrap_or(false)
                    {
                        session
                            .deleting_rlm_children
                            .lock()
                            .unwrap()
                            .remove(&child_id);
                    }
                });
            }
        } else {
            let accepted = self
                .deleting_rlm_children
                .lock()
                .unwrap()
                .get(&child_id)
                .cloned();
            if accepted
                .map(|accepted| Arc::ptr_eq(&accepted, &deletion))
                .unwrap_or(false)
            {
                self.deleting_rlm_children.lock().unwrap().remove(&child_id);
            }
        }
        result
    }

    /// `_deleteRlmSubagentSession(childId, session?)`.
    pub(super) async fn delete_rlm_subagent_session(
        &self,
        child_id: &str,
        session: Option<&Arc<AgentSession>>,
    ) -> Result<(), String> {
        let host = self.subagent_runtime_host.lock().unwrap().clone();
        if let Some(host) = host {
            return host.delete_rlm_subagent_runtime(child_id, session).await;
        }
        if let Some(session) = session {
            session.dispose_async(None).await;
        }
        Ok(())
    }

    /// `_ensureRlmRunDeletionCleanup(run, session)`.
    pub(super) async fn ensure_rlm_run_deletion_cleanup(
        self: &Arc<Self>,
        run: &RlmChildRun,
        session: &Arc<AgentSession>,
    ) -> Arc<AgentMessageDeferred> {
        let current = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(&run.id)
            .cloned();
        let cleanup = if let Some(current) = current {
            let mut current = current.lock().unwrap();
            if let Some(cleanup) = &current.deletion_cleanup {
                return cleanup.clone();
            }
            let cleanup = Arc::new(create_agent_message_deferred());
            current.deletion_cleanup = Some(cleanup.clone());
            cleanup
        } else {
            if let Some(cleanup) = &run.deletion_cleanup {
                return cleanup.clone();
            }
            Arc::new(create_agent_message_deferred())
        };
        let session_for_cleanup = session.clone();
        let owner = self.clone();
        let child_id = run.id.clone();
        let cleanup_for_task = cleanup.clone();
        tokio::spawn(async move {
            let result = owner
                .delete_rlm_subagent_session(&child_id, Some(&session_for_cleanup))
                .await;
            match result {
                Ok(()) => cleanup_for_task.resolve(),
                Err(error) => cleanup_for_task.reject(error),
            }
        });
        cleanup
    }

    /// `_recordRlmRunDeletionCleanupFailure(run, subagent, session, error)`.
    pub(super) async fn record_rlm_run_deletion_cleanup_failure(
        self: &Arc<Self>,
        run: &RlmChildRun,
        subagent: &RlmSubagentRegistryEntry,
        session: &Arc<AgentSession>,
        error: &str,
    ) {
        if self.disposed.load(Ordering::SeqCst) || self.disposing.load(Ordering::SeqCst) {
            if let Some(entry) = self.active_rlm_child_runs.lock().unwrap().get(&run.id) {
                entry.lock().unwrap().suppress_terminal_notice = Some(true);
            }
            session.dispose_async(None).await;
            let settled = self
                .active_rlm_child_runs
                .lock()
                .unwrap()
                .get(&run.id)
                .map(|entry| entry.lock().unwrap().settled)
                .unwrap_or(true);
            if !settled {
                self.finish_rlm_run_deletion(run).await;
            }
            return;
        }
        if let Some(entry) = self.active_rlm_child_runs.lock().unwrap().get(&run.id) {
            let mut entry = entry.lock().unwrap();
            entry.deletion_cleanup = None;
            entry.deletion_cleanup_observer = None;
            entry.deletion_cleanup_failed = Some(true);
            entry.session = Some(session.clone());
            entry.deletion_reservation.resolve();
        }
        self.rlm_child_cleanup_failures
            .lock()
            .unwrap()
            .insert(run.id.clone(), subagent.clone());
        // Make retry admission available before waking the parent model with the
        // retry-required notice.
        let report = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(&run.id)
            .and_then(|entry| {
                entry
                    .lock()
                    .unwrap()
                    .report_deletion_cleanup_failure
                    .clone()
            });
        if let Some(report) = report {
            report(error.to_string()).await;
        }
    }

    /// `_finishRlmRunDeletion(run)`.
    pub(super) async fn finish_rlm_run_deletion(self: &Arc<Self>, run: &RlmChildRun) {
        let complete_deletion = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(&run.id)
            .and_then(|entry| entry.lock().unwrap().complete_deletion.clone());
        if let Some(complete_deletion) = complete_deletion {
            complete_deletion().await;
        }
        let current = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(&run.id)
            .cloned();
        if let Some(current) = current {
            let snapshot = {
                let mut current = current.lock().unwrap();
                current.settled = true;
                current.settlement.resolve();
                current.deletion_reservation.resolve();
                current.clone()
            };
            self.remove_rlm_subagent_tracking(&run.id, Some(&snapshot));
        } else {
            run.settlement.resolve();
            run.deletion_reservation.resolve();
        }
        self.unsettled_rlm_child_runs
            .lock()
            .unwrap()
            .retain(|candidate| candidate.lock().unwrap().id != run.id);
        self.maybe_resume_goal_continuation_after_rlm_work();
    }

    /// `_observeRlmRunDeletionCleanup(run, subagent, session, cleanup)`.
    pub(super) fn observe_rlm_run_deletion_cleanup(
        self: &Arc<Self>,
        run: RlmChildRun,
        subagent: RlmSubagentRegistryEntry,
        session: Arc<AgentSession>,
        cleanup: Arc<AgentMessageDeferred>,
    ) -> BoxFuture<bool> {
        let existing = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(&run.id)
            .and_then(|entry| entry.lock().unwrap().deletion_cleanup_observer.clone());
        if let Some(existing) = existing {
            return Box::pin(async move { existing.wait().await.is_ok() });
        }
        let observer = Arc::new(create_agent_message_deferred());
        if let Some(entry) = self.active_rlm_child_runs.lock().unwrap().get(&run.id) {
            entry.lock().unwrap().deletion_cleanup_observer = Some(observer.clone());
        }
        let owner = self.clone();
        let observer_for_task = observer.clone();
        tokio::spawn(async move {
            match cleanup.wait().await {
                Ok(()) => observer_for_task.resolve(),
                Err(error) => {
                    owner
                        .record_rlm_run_deletion_cleanup_failure(&run, &subagent, &session, &error)
                        .await;
                    observer_for_task.reject(error);
                }
            }
        });
        Box::pin(async move { observer.wait().await.is_ok() })
    }

    /// `_continueFinishedRlmRunDeletion(run, subagent, session)`.
    pub(super) fn continue_finished_rlm_run_deletion(
        self: &Arc<Self>,
        run: RlmChildRun,
        subagent: RlmSubagentRegistryEntry,
        session: Arc<AgentSession>,
    ) {
        let owner = self.clone();
        tokio::spawn(async move {
            let cleanup = owner.ensure_rlm_run_deletion_cleanup(&run, &session).await;
            let observer =
                owner.observe_rlm_run_deletion_cleanup(run.clone(), subagent, session, cleanup);
            let run_finished = owner
                .active_rlm_child_runs
                .lock()
                .unwrap()
                .get(&run.id)
                .map(|entry| entry.lock().unwrap().deletion_run_finished == Some(true))
                .unwrap_or(false);
            if !run_finished {
                return;
            }
            if observer.await {
                owner.finish_rlm_run_deletion(&run).await;
            }
        });
    }

    /// `_removeRlmSubagentTracking(childId, run?)`.
    pub(super) fn remove_rlm_subagent_tracking(
        self: &Arc<Self>,
        child_id: &str,
        run: Option<&RlmChildRun>,
    ) {
        if let Some(run) = run {
            if let Some(unsubscribe) = run.unsubscribe.clone() {
                unsubscribe();
            }
        }
        if let Some(unsubscribe) = self.rlm_child_unsubscribes.lock().unwrap().remove(child_id) {
            unsubscribe();
        }
        self.rlm_child_sessions.lock().unwrap().remove(child_id);
        self.rlm_child_cleanup_failures
            .lock()
            .unwrap()
            .remove(child_id);
        self.abandoned_rlm_quiescence_child_ids
            .lock()
            .unwrap()
            .remove(child_id);
        let owns_entry = match run {
            None => true,
            Some(run) => self
                .active_rlm_child_runs
                .lock()
                .unwrap()
                .get(child_id)
                .map(|entry| entry.lock().unwrap().id == run.id)
                .unwrap_or(false),
        };
        if owns_entry {
            let removed = self.active_rlm_child_runs.lock().unwrap().remove(child_id);
            if let Some(entry) = removed {
                let mut entry = entry.lock().unwrap();
                entry.abort = Arc::new(noop_rlm_child_abort);
                entry.unsubscribe = None;
                entry.session = None;
            }
        }
    }

    /// `_emitRlmSubagentRemoval(subagent)`.
    pub(super) fn emit_rlm_subagent_removal(&self, subagent: &RlmSubagentRegistryEntry) {
        self.emit(AgentSessionEvent::RlmChildUpdate {
            child: RlmChildAgentSnapshot {
                id: subagent.rlm_child_id.clone(),
                parent_id: self.rlm_parent_node_id.clone(),
                active_session_id: subagent.active_session_id.clone(),
                session_name: Some(subagent.session_name.clone()),
                model: None,
                label: subagent.session_name.clone(),
                status: RLM_CHILD_AGENT_STATUS_CANCELLED.to_string(),
                duration_ms: None,
                answer_preview: None,
                tool_use_count: None,
                token_count: None,
                recap: None,
                session_dir: subagent.session_dir.clone(),
                activity: None,
                replied_since_task: None,
                error: Some("Deleted by parent orchestrator".to_string()),
            },
        });
    }

    /// `_deleteResolvedRlmSubagent(subagent)`.
    pub(super) async fn delete_resolved_rlm_subagent(
        self: &Arc<Self>,
        subagent: &RlmSubagentRegistryEntry,
    ) -> Result<RlmDeleteSubagentResult, String> {
        let child_id = subagent.rlm_child_id.clone();
        {
            // Explicit deletion is a separate authorized action, even after a retained stop.
            // Later stop admission must fail before this cleanup reaches its first await.
            let _admission = self.rlm_child_lifecycle_admission.lock().unwrap();
            self.rlm_child_release_claims.lock().unwrap().insert(child_id.clone());
        }
        let run = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(&child_id)
            .cloned();
        if let Some(run) = run {
            {
                let mut entry = run.lock().unwrap();
                if entry.deletion_cleanup_failed == Some(true) {
                    // Reset retry coordination only after selector preflight reaches the
                    // resolved child.
                    entry.deletion_cleanup_failed = Some(false);
                    entry.deletion_failure_notice = None;
                    entry.deletion_reservation = create_agent_message_deferred();
                }
                // The detached task remains the sole lifecycle owner. Mark deletion
                // before cancellation so its catch/finally path cannot race a normal
                // release or terminal notice against the physical delete.
                entry.detached_deletion = Some(subagent.clone());
            }
            let snapshot = run.lock().unwrap().clone();
            if self.cancel_rlm_child_run(&snapshot, "Deleted by parent orchestrator") {
                run.lock().unwrap().deletion_needs_completion_notice = Some(true);
            } else {
                self.emit_rlm_subagent_removal(subagent);
            }
            let live_session = run.lock().unwrap().session.clone();
            let (status, settled) = {
                let entry = run.lock().unwrap();
                (entry.status.clone(), entry.settled)
            };
            if status == RLM_CHILD_AGENT_STATUS_ERROR && live_session.is_none() && settled {
                self.deleted_rlm_child_ids
                    .lock()
                    .unwrap()
                    .insert(child_id.clone());
                let snapshot = run.lock().unwrap().clone();
                self.remove_rlm_subagent_tracking(&child_id, Some(&snapshot));
                return Ok(RlmDeleteSubagentResult {
                    subagent: subagent.clone(),
                    outcome: None,
                });
            }
            if live_session.is_some() && settled {
                {
                    let mut entry = run.lock().unwrap();
                    entry.deletion_run_finished = Some(true);
                    entry.settlement = create_agent_message_deferred();
                    entry.settled = false;
                }
                self.unsettled_rlm_child_runs
                    .lock()
                    .unwrap()
                    .push(run.clone());
            }
            if let Some(live_session) = live_session {
                self.continue_finished_rlm_run_deletion(
                    run.lock().unwrap().clone(),
                    subagent.clone(),
                    live_session,
                );
            }

            // Return once deletion is accepted. The run stays hidden but unsettled
            // until abort-insensitive model/tool work unwinds and cleanup finishes.
            self.deleted_rlm_child_ids
                .lock()
                .unwrap()
                .insert(child_id.clone());
            return Ok(RlmDeleteSubagentResult {
                subagent: subagent.clone(),
                outcome: None,
            });
        }

        self.emit_rlm_subagent_removal(subagent);
        let retained = self
            .rlm_child_sessions
            .lock()
            .unwrap()
            .get(&child_id)
            .map(|child| child.session.clone());
        if let Err(error) = self
            .delete_rlm_subagent_session(&child_id, retained.as_ref())
            .await
        {
            if self.disposed.load(Ordering::SeqCst) || self.disposing.load(Ordering::SeqCst) {
                self.remove_rlm_subagent_tracking(&child_id, None);
                if let Some(retained) = &retained {
                    retained.dispose_async(None).await;
                }
            } else {
                self.rlm_child_cleanup_failures
                    .lock()
                    .unwrap()
                    .insert(child_id.clone(), subagent.clone());
            }
            return Err(error);
        }
        self.deleted_rlm_child_ids
            .lock()
            .unwrap()
            .insert(child_id.clone());
        self.remove_rlm_subagent_tracking(&child_id, None);
        Ok(RlmDeleteSubagentResult {
            subagent: subagent.clone(),
            outcome: None,
        })
    }

    /// `releaseRlmChildSession(childId, session)`.
    pub fn release_rlm_child_session(
        self: &Arc<Self>,
        child_id: &str,
        session: &Arc<AgentSession>,
    ) -> Option<Box<dyn Fn() + Send + Sync>> {
        let run = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .get(child_id)
            .cloned();
        if let Some(run) = run {
            let matches = {
                let current = run.lock().unwrap();
                current.status == "done"
                    && current
                        .session
                        .as_ref()
                        .is_some_and(|child| Arc::ptr_eq(child, session))
            };
            if matches {
                let weak = Arc::downgrade(self);
                let child_id = child_id.to_string();
                return Some(Box::new(move || {
                    let unsubscribe = run.lock().unwrap().unsubscribe.take();
                    if let Some(parent) = weak.upgrade() {
                        let mut runs = parent.active_rlm_child_runs.lock().unwrap();
                        if runs
                            .get(&child_id)
                            .is_some_and(|current| Arc::ptr_eq(current, &run))
                        {
                            runs.remove(&child_id);
                        }
                    }
                    if let Some(unsubscribe) = unsubscribe {
                        unsubscribe();
                    }
                }));
            }
        }
        if !self
            .rlm_child_sessions
            .lock()
            .unwrap()
            .get(child_id)
            .is_some_and(|child| Arc::ptr_eq(&child.session, session))
        {
            return None;
        }
        let weak = Arc::downgrade(self);
        let child_id = child_id.to_string();
        let session = session.clone();
        Some(Box::new(move || {
            if let Some(parent) = weak.upgrade() {
                let removed = {
                    let mut children = parent.rlm_child_sessions.lock().unwrap();
                    if children
                        .get(&child_id)
                        .is_some_and(|child| Arc::ptr_eq(&child.session, &session))
                    {
                        children.remove(&child_id).is_some()
                    } else {
                        false
                    }
                };
                if removed {
                    let unsubscribe = parent
                        .rlm_child_unsubscribes
                        .lock()
                        .unwrap()
                        .remove(&child_id);
                    if let Some(unsubscribe) = unsubscribe {
                        unsubscribe();
                    }
                }
            }
        }))
    }

    /// A read-only, bounded fan-in. Collection never delivers another parent message.
    pub async fn collect_rlm_children(
        &self,
        targets: &[String],
        timeout_ms: u64,
    ) -> Result<crate::core::rlm_runtime::RlmCollectResult, String> {
        use crate::core::rlm_runtime::{RlmCollectResult, RlmCollectResultEntry};
        let mut candidates: std::collections::BTreeMap<_, _> = self
            .active_rlm_child_runs
            .lock()
            .unwrap()
            .iter()
            .map(|(id, run)| (id.clone(), run.clone()))
            .collect();
        let retained = self.rlm_child_sessions.lock().unwrap().clone();
        for (id, child) in &retained {
            if let Some(run) = &child.run {
                candidates.entry(id.clone()).or_insert_with(|| run.clone());
            }
        }
        let deleting: HashSet<_> = self
            .deleting_rlm_children
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        candidates.retain(|id, run| {
            !deleting.contains(id) && run.lock().unwrap().detached_deletion.is_none()
        });
        let mut runs = std::collections::BTreeMap::new();
        if targets.is_empty() {
            runs = candidates;
        } else {
            for target in targets {
                let mut matches = Vec::new();
                for (id, run) in &candidates {
                    let snapshot = run.lock().unwrap().clone();
                    let child = snapshot
                        .session
                        .or_else(|| retained.get(id).map(|entry| entry.session.clone()));
                    if id == target
                        || snapshot.session_name == *target
                        || child.as_ref().is_some_and(|child| {
                            child.session_id() == *target
                                || child.session_name().as_deref() == Some(target)
                        })
                    {
                        matches.push((id.clone(), run.clone()));
                    }
                }
                if matches.len() != 1 {
                    return Err(format!(
                        "RLM child selector {target:?} {} in the current parent session",
                        if matches.is_empty() {
                            "matches no direct child"
                        } else {
                            "is ambiguous"
                        }
                    ));
                }
                let (id, run) = matches.pop().unwrap();
                runs.insert(id, run);
            }
        }
        if timeout_ms > 0 {
            let settlements: Vec<_> = runs
                .values()
                .filter_map(|run| {
                    let run = run.lock().unwrap();
                    (!run.settled).then(|| run.settlement.clone())
                })
                .collect();
            let wait =
                futures::future::join_all(settlements.iter().map(AgentMessageDeferred::wait));
            tokio::select! {
                _ = tokio::time::timeout(std::time::Duration::from_millis(timeout_ms.min(2_147_483_647)), wait) => {}
                _ = self.session_action_commit_dispose_abort.cancelled() => {}
            }
        }
        let results = runs
            .into_values()
            .map(|run| {
                let run = run.lock().unwrap().clone();
                let child = run
                    .session
                    .as_ref()
                    .or_else(|| retained.get(&run.id).map(|entry| &entry.session));
                RlmCollectResultEntry {
                    rlm_child_id: run.id,
                    session_name: child
                        .and_then(|child| child.session_name())
                        .or(Some(run.session_name)),
                    session_dir: run.session_dir,
                    status: run.status,
                    settled: run.settled,
                    answer_preview: run.answer_preview.map(|text| compact_rlm_text(&text, 160)),
                    error: run.error.map(|text| compact_rlm_text(&text, 2000)),
                    duration_ms: run.duration_ms,
                    tool_use_count: Some(run.tool_use_count),
                    replied_since_task: child
                        .and_then(|child| child.replied_to_parent_since_task()),
                }
            })
            .collect();
        Ok(RlmCollectResult { results })
    }

    /// `_rlmChildSnapshotForRun(run, child = run.session ?? retained session)`.
    pub(super) fn rlm_child_snapshot_for_run(
        &self,
        run: &RlmChildRun,
        child: Option<Arc<AgentSession>>,
    ) -> RlmChildAgentSnapshot {
        let child = child.or_else(|| run.session.clone()).or_else(|| {
            self.rlm_child_sessions
                .lock()
                .unwrap()
                .get(&run.id)
                .map(|retained| retained.session.clone())
        });
        let model = child
            .as_ref()
            .and_then(|child| child.model())
            .or_else(|| run.model.clone());
        RlmChildAgentSnapshot {
            id: run.id.clone(),
            parent_id: self.rlm_parent_node_id.clone(),
            active_session_id: None,
            session_name: child
                .as_ref()
                .and_then(|child| child.session_name())
                .or_else(|| Some(run.session_name.clone())),
            model: model.map(|model| format!("{}/{}", model.provider, model.id)),
            label: rlm_child_label(&run.prompt),
            status: run.status.clone(),
            duration_ms: run.duration_ms,
            answer_preview: run.answer_preview.clone(),
            tool_use_count: if run.tool_use_count > 0.0 {
                Some(run.tool_use_count)
            } else {
                None
            },
            token_count: child
                .as_ref()
                .and_then(|child| child.context_tokens_for_current_messages()),
            recap: child.as_ref().and_then(|child| child.get_current_recap()),
            session_dir: run.session_dir.clone(),
            activity: run.activity.clone(),
            replied_since_task: child
                .as_ref()
                .and_then(|child| child.replied_to_parent_since_task()),
            error: run.error.clone(),
        }
    }

    /// `_rlmChildSnapshotForSession(childId, child)`.
    pub(super) fn rlm_child_snapshot_for_session(
        &self,
        child_id: &str,
        child: &Arc<AgentSession>,
    ) -> RlmChildAgentSnapshot {
        let mut answer_preview: Option<String> = None;
        let mut tool_use_count = 0.0;
        let mut messages = child.messages();
        if let Some(streaming) = child.agent.state().streaming_message {
            if streaming.role() == pi_ai::types::ROLE_ASSISTANT {
                messages.push(streaming);
            }
        }
        for message in messages.iter() {
            let AgentMessage::Message(pi_ai::types::Message::Assistant(message)) = message else {
                continue;
            };
            let text = compact_rlm_text(&read_assistant_text(message), 160);
            if !text.is_empty() {
                answer_preview = Some(text);
            }
            tool_use_count += message
                .content
                .iter()
                .filter(|block| matches!(block, pi_ai::types::ContentBlock::ToolCall(_)))
                .count() as f64;
        }
        RlmChildAgentSnapshot {
            id: child_id.to_string(),
            parent_id: self.rlm_parent_node_id.clone(),
            active_session_id: None,
            session_name: child.session_name(),
            model: child
                .model()
                .map(|model| format!("{}/{}", model.provider, model.id)),
            label: child
                .session_name()
                .unwrap_or_else(|| "child agent".to_string()),
            status: RLM_CHILD_AGENT_STATUS_DONE.to_string(),
            duration_ms: None,
            answer_preview,
            tool_use_count: if tool_use_count > 0.0 {
                Some(tool_use_count)
            } else {
                None
            },
            token_count: child.context_tokens_for_current_messages(),
            recap: child.get_current_recap(),
            session_dir: child
                .rlm_session_dir
                .clone()
                .unwrap_or_else(|| child.session_manager.lock().unwrap().get_session_dir()),
            // No run exists (e.g. a child rehydrated after daemon recovery), so live
            // session state is the only source for in-flight follow-up work.
            activity: if child.is_session_active() {
                Some(RlmChildAgentActivity {
                    kind: if child.is_streaming() {
                        "writing".to_string()
                    } else {
                        "waiting".to_string()
                    },
                    tool_name: None,
                })
            } else {
                None
            },
            replied_since_task: child.replied_to_parent_since_task(),
            error: None,
        }
    }

    /// `_isUnboundTerminalRlmChildRun(run)`.
    pub(super) fn is_unbound_terminal_rlm_child_run(&self, run: &RlmChildRun) -> bool {
        matches!(run.status.as_str(), "done" | "error" | "cancelled")
            && run.session.is_none()
            && !self
                .rlm_child_sessions
                .lock()
                .unwrap()
                .contains_key(&run.id)
    }

    /// `hasRunningRlmChildren()`.
    pub fn has_running_rlm_children(&self) -> bool {
        self.active_rlm_child_runs
            .lock()
            .unwrap()
            .values()
            .any(|run| matches!(run.lock().unwrap().status.as_str(), "queued" | "running"))
    }

    /// `_rlmChildSessionSnapshot()`.
    pub(super) fn rlm_child_session_snapshot(&self) -> Vec<Arc<AgentSession>> {
        self.rlm_child_sessions
            .lock()
            .unwrap()
            .values()
            .map(|child| child.session.clone())
            .collect()
    }

    /// `_hasUnsettledRlmQuiescenceWork()`.
    pub(super) fn has_unsettled_rlm_quiescence_work(&self) -> bool {
        !self.unsettled_rlm_child_runs.lock().unwrap().is_empty()
            || !self
                .abandoned_rlm_quiescence_child_ids
                .lock()
                .unwrap()
                .is_empty()
    }

    /// `_assertRlmSubagentSessionNameAvailable(name, ignorePendingName?)`.
    pub(super) async fn assert_rlm_subagent_session_name_available(
        self: &Arc<Self>,
        name: &str,
        ignore_pending_name: bool,
    ) -> Result<(), String> {
        let pending = if ignore_pending_name {
            false
        } else {
            self.pending_rlm_subagent_session_names
                .lock()
                .unwrap()
                .contains(name)
        };
        if pending
            || self
                .list_rlm_subagents()
                .await?
                .subagents
                .iter()
                .any(|child| child.session_name == name)
        {
            return Err(format_agent_session_name_unavailable(
                name,
                (self.rlm_depth + 1) as f64,
            ));
        }
        Ok(())
    }

    /// `_authenticatedRlmModels()`.
    pub(super) async fn authenticated_rlm_models(&self) -> Vec<Model> {
        self.model_registry.lock().unwrap().get_available()
    }

    /// `findRlmModels(query, limit)`.
    pub async fn find_rlm_models(
        self: &Arc<Self>,
        query: &str,
        limit: i64,
    ) -> Result<RlmFindModelsResult, String> {
        Ok(RlmFindModelsResult {
            models: crate::core::rlm_runtime::find_rlm_model_matches(
                query,
                &self.authenticated_rlm_models().await,
                limit as f64,
            ),
        })
    }

    /// `_resolveRlmSubagentModel(reference, target)` (agent-session.ts:11418-11446).
    pub(super) async fn resolve_rlm_subagent_model(
        self: &Arc<Self>,
        reference: Option<&str>,
        target: &str,
    ) -> Result<RlmSubagentModelSelection, String> {
        let parent_model = self.model().ok_or_else(format_no_model_selected_message)?;
        let Some(reference) = reference else {
            return Ok(RlmSubagentModelSelection {
                model: parent_model,
            });
        };
        let normalized_reference = reference.to_lowercase();
        let parent_selector =
            format!("{}/{}", parent_model.provider, parent_model.id).to_lowercase();
        if parent_selector == normalized_reference {
            return Ok(RlmSubagentModelSelection {
                model: parent_model,
            });
        }
        let model = self
            .authenticated_rlm_models()
            .await
            .into_iter()
            .find(|candidate| {
                format!("{}/{}", candidate.provider, candidate.id).to_lowercase()
                    == normalized_reference
            });
        let Some(model) = model else {
            return Err(format!(
                "Requested {target} model \"{reference}\" is unavailable, unauthenticated, or expired"
            ));
        };
        let request_model = model.clone();
        let registry = self.model_registry.clone();
        let auth = crate::core::sdk::with_model_registry(registry, move |registry| {
            Box::pin(async move { registry.get_api_key_and_headers(&request_model).await })
        })
        .await
        .unwrap_or_else(|error| crate::core::model_registry::ResolvedRequestAuth {
            ok: false,
            api_key: None,
            headers: None,
            error: Some(error),
        });
        if !auth.ok {
            return Err(format!(
                "Requested {target} model \"{reference}\" failed authentication preflight"
            ));
        }
        Ok(RlmSubagentModelSelection { model })
    }

    /// `createRlmSession(prompt, kwargs)` (agent-session.ts:11934-11990).
    pub async fn create_rlm_session(
        self: &Arc<Self>,
        prompt: &str,
        kwargs: &Map<String, Value>,
    ) -> Result<RlmCreateSessionResult, String> {
        let operation = "rlm.create_session";
        let mut unsupported: Vec<&str> = kwargs
            .keys()
            .map(String::as_str)
            .filter(|key| !matches!(*key, "name" | "model" | "thinking" | "cwd"))
            .collect();
        if !unsupported.is_empty() {
            unsupported.sort_unstable();
            return Err(format!(
                "Unsupported rlm.create_session kwargs: {}",
                unsupported.join(", ")
            ));
        }
        if prompt.trim().is_empty() {
            return Err("rlm.create_session prompt must not be empty".to_string());
        }
        if self.rlm_depth != 0 {
            return Err("rlm.create_session is available only from a depth-0 session".to_string());
        }
        if self.disposed.load(Ordering::SeqCst) || self.disposing.load(Ordering::SeqCst) {
            return Err(
                "Cannot create a top-level session after the current session was disposed"
                    .to_string(),
            );
        }
        let host = self.subagent_runtime_host.lock().unwrap().clone();
        let Some(host) = host else {
            return Err("rlm.create_session requires a daemon-backed depth-0 session".to_string());
        };

        let session_name =
            normalize_requested_rlm_subagent_session_name(kwargs.get("name"), Some(operation))?;
        let requested_model =
            normalize_requested_rlm_subagent_model(kwargs.get("model"), Some(operation))?;
        let requested_thinking_level = normalize_requested_rlm_subagent_thinking_level(
            kwargs.get("thinking"),
            Some(operation),
        )?;
        if let Some(session_name) = session_name.as_deref() {
            assert_direct_agent_message_target(session_name)?;
        }
        if let Some(raw_cwd) = kwargs.get("cwd") {
            if raw_cwd
                .as_str()
                .map(|value| value.trim().is_empty())
                .unwrap_or(true)
            {
                return Err("rlm.create_session cwd must be a non-empty string".to_string());
            }
        }
        let cwd = match kwargs.get("cwd").and_then(Value::as_str) {
            Some(raw_cwd) => Path::new(&self.cwd)
                .join(raw_cwd.trim())
                .to_string_lossy()
                .into_owned(),
            None => self.cwd.clone(),
        };
        let model_selection = self
            .resolve_rlm_subagent_model(requested_model.as_deref(), "top-level session")
            .await?;
        if let Some(requested_thinking_level) = requested_thinking_level {
            let supported = get_supported_thinking_levels(&model_selection.model);
            let requested = thinking_level_name(&requested_thinking_level);
            if !supported.iter().any(|level| level == &requested) {
                return Err(format!(
                    "Requested thinking level \"{requested}\" is not supported by model \"{}/{}\"; supported levels: {}",
                    model_selection.model.provider,
                    model_selection.model.id,
                    supported.join(", ")
                ));
            }
        }
        let thinking_level = requested_thinking_level.unwrap_or_else(|| {
            clamp_thinking_level_for_model(&model_selection.model, self.thinking_level())
        });
        if self.disposed.load(Ordering::SeqCst) || self.disposing.load(Ordering::SeqCst) {
            return Err(
                "Cannot create a top-level session after the current session was disposed"
                    .to_string(),
            );
        }
        // REPAIR CURSOR: the TypeScript also asks the agent-message controller to
        // reserve the name (`controller.assertSessionNameAvailable`, agent-session.ts:11961).
        // The canonical `AgentSessionMessageController` trait
        // (`core/agent_messages.rs:947`) exposes only roster/awaitPendingChildPublication/
        // sendAgentMessage, so the reservation cannot be requested here. Add
        // `assert_session_name_available` to that trait and call it with
        // `{ name, depth: 0 }` before `create_rlm_root_session`.
        host.create_rlm_root_session(CreateRlmRootSessionOptions {
            prompt: prompt.to_string(),
            session_name,
            cwd,
            model: model_selection.model,
            thinking_level,
        })
        .await
    }

    /// `runRlmChild(prompt, kwargs, spawnCode?)`.
    pub async fn run_rlm_child(
        self: &Arc<Self>,
        prompt: &str,
        kwargs: &Map<String, Value>,
    ) -> Result<RlmSpawnHandle, String> {
        self.start_rlm_child_run(prompt, kwargs, None).await
    }

    /// `_isRetryableError(message)` (agent-session.ts:12000-12019).
    pub(super) fn is_retryable_error(&self, message: &AssistantMessage) -> bool {
        if message.stop_reason != STOP_REASON_ERROR || message.error_message.is_none() {
            return false;
        }
        let context_window = self
            .model()
            .map(|model| model.context_window)
            .unwrap_or(0.0);
        if pi_ai::utils::overflow::is_context_overflow(message, Some(context_window)) {
            return false;
        }
        if self.is_faux_provider_queue_exhausted(message) {
            return false;
        }
        if self.is_agent_lifecycle_failure(message) {
            return false;
        }
        if self.is_structured_permanent_provider_retry_exhausted(message) {
            return false;
        }
        true
    }

    /// `_isFauxProviderQueueExhausted(message)`.
    pub(super) fn is_faux_provider_queue_exhausted(&self, message: &AssistantMessage) -> bool {
        crate::core::provider_retry::is_faux_provider_queue_exhausted(message)
    }

    /// `_isAgentLifecycleFailure(message)`.
    pub(super) fn is_agent_lifecycle_failure(&self, message: &AssistantMessage) -> bool {
        crate::core::provider_retry::is_agent_lifecycle_failure(message)
    }

    /// `_getProviderStreamFailureKind(message)`.
    pub(super) fn get_provider_stream_failure_kind(
        &self,
        message: &AssistantMessage,
    ) -> Option<String> {
        provider_stream_failure_kind(message)
    }

    /// `_isStructuredPermanentProviderRetryExhausted(message)`.
    pub(super) fn is_structured_permanent_provider_retry_exhausted(
        &self,
        message: &AssistantMessage,
    ) -> bool {
        crate::core::provider_retry::cannot_replay_provider_failure(message)
            || is_permanent_provider_failure_kind(
                self.get_provider_stream_failure_kind(message).as_deref(),
                self.retry_attempt.load(Ordering::SeqCst) as f64,
            )
    }

    /// `_isConcreteProviderAuthFailure(message)`.
    pub(super) fn is_concrete_provider_auth_failure(&self, message: &AssistantMessage) -> bool {
        if message.stop_reason != STOP_REASON_ERROR || message.error_message.is_none() {
            return false;
        }
        // Only the provider's structured classification counts as an auth failure.
        self.get_provider_stream_failure_kind(message).as_deref() == Some("auth")
    }

    /// `_captureRetryAuthFailureSource(message)` (agent-session.ts:12043-12060).
    pub(super) fn capture_retry_auth_failure_source(
        &self,
        message: &AssistantMessage,
    ) -> Option<AuthSourceToken> {
        let token = self
            .model_registry
            .lock()
            .unwrap()
            .get_current_provider_auth_source_token(&message.provider)?;
        let mut sources = self.retry_auth_failure_sources.lock().unwrap();
        let known = sources.iter().any(|existing| {
            existing.provider == token.provider
                && existing.source == token.source
                && existing.identity_fingerprint == token.identity_fingerprint
                && existing.value_fingerprint == token.value_fingerprint
        });
        if !known {
            sources.push(token.clone());
        }
        Some(token)
    }

    /// `_markProviderAuthStale(message, authSourceTokens?)` (agent-session.ts:12062-12082).
    pub(super) fn mark_provider_auth_stale(
        &self,
        message: &AssistantMessage,
        auth_source_tokens: Option<&[AuthSourceToken]>,
    ) -> bool {
        let mut marked = false;
        match auth_source_tokens {
            Some(tokens) if !tokens.is_empty() => {
                let mut registry = self.model_registry.lock().unwrap();
                for token in tokens {
                    marked = registry.mark_provider_auth_source_stale(token) || marked;
                }
                drop(registry);
                if marked {
                    self.emit(AgentSessionEvent::AuthStale {
                        provider: message.provider.clone(),
                        source_tokens: Some(tokens.to_vec()),
                    });
                }
            }
            _ => {
                let marked_by_provider = self
                    .model_registry
                    .lock()
                    .unwrap()
                    .mark_provider_auth_stale(&message.provider);
                marked = marked_by_provider;
                if marked {
                    self.emit(AgentSessionEvent::AuthStale {
                        provider: message.provider.clone(),
                        source_tokens: None,
                    });
                }
            }
        }
        marked
    }

    /// `_markProviderAuthStaleForRetryFailure(message, options?)` (agent-session.ts:12084-12101).
    pub(super) fn mark_provider_auth_stale_for_retry_failure(
        &self,
        message: &AssistantMessage,
    ) -> bool {
        let tokens = self.retry_auth_failure_sources.lock().unwrap().clone();
        if tokens.is_empty() {
            return false;
        }
        let marked = self.mark_provider_auth_stale(message, Some(&tokens));
        let _ = marked;
        marked
    }

    /// `_finishActiveRetryWithFailure(message)` (agent-session.ts:12103-12116).
    pub(super) fn finish_active_retry_with_failure(&self, message: &AssistantMessage) {
        let attempt = self.retry_attempt.load(Ordering::SeqCst);
        if attempt == 0 {
            return;
        }
        self.mark_provider_auth_stale_for_retry_failure(message);
        self.emit(AgentSessionEvent::AutoRetryEnd {
            success: false,
            attempt: attempt as i64,
            final_error: message.error_message.clone(),
        });
        self.retry_attempt.store(0, Ordering::SeqCst);
        self.retry_auth_failure_sources.lock().unwrap().clear();
        // The group is exhausted: settle its outer terminal and hand the next turn a
        // fresh correlation, so no later attempt is attributed to this finished group.
        self.close_retry_metric_group(Some(
            pi_agent_core::performance_metrics::PerformanceMetricOutcome::Failure,
        ));
    }

    /// Settle the host-owned logical-request group of the active retry and drop the
    /// group correlation from the agent.
    ///
    /// The host owns the outer `logical_request` terminal while a retry group is open
    /// (`host_owns_logical_request_terminal`). Without this, the group's id, settlement
    /// and attempt ordinal outlive the group: later turns reuse a settled group id, no
    /// new outer terminal is emitted, and their attempts are attributed to a logical
    /// request that finished earlier (B6). Correlation is settled at most once, and
    /// clearing the host-owned metrics restores the ordinary per-turn baseline.
    pub(super) fn close_retry_metric_group(
        &self,
        outcome: Option<pi_agent_core::performance_metrics::PerformanceMetricOutcome>,
    ) {
        if let Some(metrics) = self.agent.performance_metrics() {
            if !metrics.host_owns_logical_request_terminal {
                return;
            }
        } else {
            return;
        }
        if let Some(message) = self.retry_metric_message.lock().unwrap().clone() {
            pi_agent_core::agent_loop::finalize_performance_metric_logical_request(
                &message, outcome,
            );
        }
        if let Some(metrics) = self.agent.performance_metrics() {
            self.agent.set_performance_metrics(Some(
                pi_agent_core::performance_metrics::AgentLoopPerformanceMetrics::new(
                    metrics.recorder.clone(),
                ),
            ));
        }
    }

    /// `_handleRetryableError(message, options?)` (agent-session.ts:12118-12253).
    pub(super) async fn handle_retryable_error(
        self: &Arc<Self>,
        message: &AssistantMessage,
    ) -> bool {
        // Every path that ends the retry instead of starting another attempt must close
        // the host-owned group. Otherwise the group's id, settlement and attempt ordinal
        // stay live after the retry is over and later turns reuse them (B6).
        if self.explicitly_stopped() {
            self.close_retry_metric_group(Some(
                pi_agent_core::performance_metrics::PerformanceMetricOutcome::Cancelled,
            ));
            self.resolve_retry();
            return false;
        }
        let policy = crate::core::provider_retry::provider_retry_policy(
            &self.settings_manager.lock().unwrap(),
        );
        if !policy.enabled {
            self.mark_provider_auth_stale_for_retry_failure(message);
            self.retry_auth_failure_sources.lock().unwrap().clear();
            self.close_retry_metric_group(Some(
                pi_agent_core::performance_metrics::PerformanceMetricOutcome::Failure,
            ));
            self.resolve_retry();
            return false;
        }
        // The TypeScript creates the promise here; the Rust owner already creates
        // it at agent_end (`create_retry_promise_for_agent_end`), so only the
        // resolve half is owned by this member.
        let attempt = self.retry_attempt.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt as f64 > policy.max_retries {
            self.mark_provider_auth_stale_for_retry_failure(message);
            self.emit(AgentSessionEvent::AutoRetryEnd {
                success: false,
                attempt: (attempt - 1) as i64,
                final_error: message.error_message.clone(),
            });
            self.retry_attempt.store(0, Ordering::SeqCst);
            self.retry_auth_failure_sources.lock().unwrap().clear();
            // The retry budget is exhausted, so no further attempt will settle the group.
            self.close_retry_metric_group(Some(
                pi_agent_core::performance_metrics::PerformanceMetricOutcome::Failure,
            ));
            self.resolve_retry();
            return false;
        }
        let delay = provider_retry_delay(
            attempt as f64,
            provider_stream_failure_retry_after_ms(message),
            &policy,
            &ProviderRetryDelayOptions::default(),
        );
        let delay_ms = match delay {
            ProviderRetryDelay::ExceedsCap { retry_after_ms } => {
                self.mark_provider_auth_stale_for_retry_failure(message);
                self.emit(AgentSessionEvent::AutoRetryEnd {
                    success: false,
                    attempt: (attempt - 1) as i64,
                    final_error: Some(format!(
                        "Provider requested a {}s wait before retrying (above retry.provider.maxRetryDelayMs={}ms): {}",
                        (retry_after_ms / 1000.0).ceil(),
                        policy.max_retry_delay_ms,
                        message.error_message.clone().unwrap_or_else(|| "unknown error".to_string()),
                    )),
                });
                self.retry_attempt.store(0, Ordering::SeqCst);
                self.retry_auth_failure_sources.lock().unwrap().clear();
                // No attempt will follow the refused wait, so the group ends here.
                self.close_retry_metric_group(Some(
                    pi_agent_core::performance_metrics::PerformanceMetricOutcome::Failure,
                ));
                self.resolve_retry();
                return false;
            }
            ProviderRetryDelay::Wait { delay_ms } => delay_ms,
        };
        if !self.has_extension_handlers("before_provider_request") {
            self.semantic_edges.lock().unwrap().prepare_turn_retry();
        }
        self.emit(AgentSessionEvent::AutoRetryStart {
            attempt: attempt as i64,
            max_attempts: policy.max_retries as i64,
            delay_ms,
            error_message: message
                .error_message
                .clone()
                .unwrap_or_else(|| "Unknown error".to_string()),
        });
        // The failed assistant message is dropped again so the retry re-issues it.
        self.remove_failed_assistant_from_state(message);
        *self.retry_metric_message.lock().unwrap() = Some(message.clone());
        let controller = CancellationToken::new();
        {
            let _admission = self.explicit_stop_admission.lock().unwrap();
            if self.explicitly_stopped() {
                controller.cancel();
            } else {
                *self.retry_abort_controller.lock().unwrap() = Some(controller.clone());
            }
        }
        let slept = crate::utils::sleep::sleep(delay_ms.max(0.0) as u64, Some(&controller)).await;
        if slept.is_err() {
            let attempt = self.retry_attempt.load(Ordering::SeqCst);
            self.mark_provider_auth_stale_for_retry_failure(message);
            self.retry_attempt.store(0, Ordering::SeqCst);
            *self.retry_abort_controller.lock().unwrap() = None;
            self.emit(AgentSessionEvent::AutoRetryEnd {
                success: false,
                attempt: attempt as i64,
                final_error: Some("Retry cancelled".to_string()),
            });
            self.close_retry_metric_group(Some(
                pi_agent_core::performance_metrics::PerformanceMetricOutcome::Cancelled,
            ));
            self.resolve_retry();
            self.retry_auth_failure_sources.lock().unwrap().clear();
            return false;
        }
        *self.retry_abort_controller.lock().unwrap() = None;
        let retry_generation = self.retry_generation.load(Ordering::SeqCst);
        let correlation =
            pi_agent_core::agent_loop::get_performance_metric_request_correlation(message);
        let session = self.clone();
        let message = message.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Some(metrics) = session.agent.performance_metrics() {
                    let next_attempt = correlation
                        .as_ref()
                        .map(|correlation| correlation.provider_attempt_number)
                        .or(metrics.provider_attempt_number)
                        .unwrap_or(1)
                        + 1;
                    session.agent.set_performance_metrics(Some(
                        pi_agent_core::performance_metrics::AgentLoopPerformanceMetrics {
                            recorder: metrics.recorder.clone(),
                            logical_request_id: correlation
                                .as_ref()
                                .and_then(|correlation| correlation.logical_request_id.clone())
                                .or(metrics.logical_request_id.clone()),
                            logical_request_started_at: correlation
                                .as_ref()
                                .and_then(|correlation| correlation.logical_request_started_at)
                                .or(metrics.logical_request_started_at),
                            provider_attempt_number: Some(next_attempt),
                            host_owns_logical_request_terminal: true,
                            logical_request_settlement: correlation
                                .as_ref()
                                .map(|correlation| correlation.logical_request_settlement.clone())
                                .or(metrics.logical_request_settlement.clone()),
                        },
                    ));
                }
                if let Err(error) = session.agent.continue_().await {
                    // A continue that never starts must still resolve the retry (else
                    // isRetrying sticks forever) unless a newer retry owns the state.
                    if session.retry_generation.load(Ordering::SeqCst) != retry_generation
                        || !session.is_retrying()
                    {
                        return;
                    }
                    session.mark_provider_auth_stale_for_retry_failure(&message);
                    let attempt = session.retry_attempt.load(Ordering::SeqCst);
                    session.retry_attempt.store(0, Ordering::SeqCst);
                    session.retry_auth_failure_sources.lock().unwrap().clear();
                    session.emit(AgentSessionEvent::AutoRetryEnd {
                        success: false,
                        attempt: attempt as i64,
                        final_error: Some(error.to_string()),
                    });
                    // The retry never reached the provider, so nothing will settle the
                    // group for it: close it here as a failure, exactly once.
                    session.close_retry_metric_group(Some(
                        pi_agent_core::performance_metrics::PerformanceMetricOutcome::Failure,
                    ));
                    session.resolve_retry();
                }
            });
        }
        true
    }

    /// `abortRetry()` (agent-session.ts:12255-12276).
    pub fn abort_retry(&self) {
        if let Some(controller) = self.retry_abort_controller.lock().unwrap().take() {
            controller.cancel();
            return;
        }
        let attempt = self.retry_attempt.load(Ordering::SeqCst);
        if attempt > 0 {
            if let Some(controller) = self.auto_compaction_abort_controller.lock().unwrap().take() {
                controller.cancel();
            }
            self.cancel_post_compaction_continue();
            self.emit(AgentSessionEvent::AutoRetryEnd {
                success: false,
                attempt: attempt as i64,
                final_error: Some("Retry cancelled".to_string()),
            });
            self.retry_attempt.store(0, Ordering::SeqCst);
            // The cancelled retry still owns a live group; settle it once and restore the
            // per-turn baseline so the next turn cannot reuse the finished group (B6).
            self.close_retry_metric_group(Some(
                pi_agent_core::performance_metrics::PerformanceMetricOutcome::Cancelled,
            ));
        }
        self.retry_auth_failure_sources.lock().unwrap().clear();
        self.resolve_retry();
    }

    /// `waitForRetry()` (agent-session.ts:12278-12285).
    ///
    /// All observers await the same deferred; only the retry owner clears its slot.
    /// A waiter never consumes pending state or resurrects a settled generation.
    pub(super) async fn wait_for_retry(&self) {
        let promise = { self.retry_promise.lock().unwrap().clone() };
        let Some(promise) = promise else {
            return;
        };
        let _ = promise.wait().await;
        self.agent.wait_for_idle().await;
    }

    /// `get isRetrying()`.
    pub fn is_retrying(&self) -> bool {
        self.retry_promise.lock().unwrap().is_some()
    }

    /// `get hasAcceptedPromptInFlight()` (agent-session.ts:12291-12300).
    pub fn has_accepted_prompt_in_flight(&self) -> bool {
        self.action_store
            .lock()
            .unwrap()
            .unfinished_actions(None)
            .iter()
            .any(|action| {
                let QueuedActionPayload::Turn(turn) = &action.payload else {
                    return false;
                };
                !turn.queue_visible && turn.accepted_before_completion
            })
    }

    /// `get autoRetryEnabled()`.
    pub fn auto_retry_enabled(&self) -> bool {
        self.settings_manager.lock().unwrap().get_retry_enabled()
    }

    /// `setAutoRetryEnabled(enabled)` (agent-session.ts:12306-12308).
    pub fn set_auto_retry_enabled(&self, enabled: bool) {
        self.settings_manager
            .lock()
            .unwrap()
            .set_retry_enabled(enabled);
    }

    pub async fn execute_bash(
        self: &Arc<Self>,
        command: &str,
        on_chunk: Option<Arc<dyn Fn(&str) + Send + Sync>>,
        exclude_from_context: Option<bool>,
    ) -> Result<BashResult, String> {
        self.execute_bash_with_operations(command, on_chunk, exclude_from_context, None)
            .await
    }

    /// `executeBash(command, onChunk, { excludeFromContext, operations, transient })`
    /// (agent-session.ts:12318-12344).
    ///
    /// `operations` is the extension-supplied `user_bash` override; when absent the
    /// built-in local shell runs, exactly like
    /// `options?.operations ?? createLocalBashOperations({ shellPath })` at
    /// agent-session.ts:12339.
    async fn execute_bash_with_operations(
        self: &Arc<Self>,
        command: &str,
        on_chunk: Option<Arc<dyn Fn(&str) + Send + Sync>>,
        exclude_from_context: Option<bool>,
        operations: Option<Arc<dyn crate::core::tools::BashOperations>>,
    ) -> Result<BashResult, String> {
        let controller = CancellationToken::new();
        self.bash_abort_controllers
            .lock()
            .unwrap()
            .push(controller.clone());
        let (prefix, shell_path) = {
            let settings = self.settings_manager.lock().unwrap();
            (
                settings.get_shell_command_prefix(),
                settings.get_shell_path(),
            )
        };
        let resolved = prefix
            .filter(|prefix| !prefix.is_empty())
            .map(|prefix| format!("{prefix}\n{command}"))
            .unwrap_or_else(|| command.to_string());
        // `options?.operations ?? createLocalBashOperations({ shellPath })` (agent-session.ts:12339).
        let operations = operations.unwrap_or_else(|| {
            crate::core::tools::create_local_bash_operations(Some(
                crate::core::tools::LocalBashOperationsOptions { shell_path },
            ))
        });
        let result = crate::core::bash_executor::execute_bash_with_operations(
            &resolved,
            &self.cwd,
            operations,
            Some(crate::core::bash_executor::BashExecutorOptions {
                on_chunk,
                signal: Some(controller.clone()),
            }),
        )
        .await;
        controller.cancel();
        self.bash_abort_controllers
            .lock()
            .unwrap()
            .retain(|token| !token.is_cancelled());
        self.notify_session_input_checkpoint_change();
        let result = result?;
        self.record_bash_result(command, &result, exclude_from_context);
        Ok(result)
    }

    pub async fn run_user_bash(
        self: &Arc<Self>,
        command: &str,
        exclude_from_context: Option<bool>,
    ) -> Result<BashResult, String> {
        if self.user_bash_running.swap(true, Ordering::SeqCst) {
            return Err("A bash command is already running".to_string());
        }
        self.user_bash_abort_requested
            .store(false, Ordering::SeqCst);
        let result = self
            .run_user_bash_locked(command, exclude_from_context, CancellationToken::new())
            .await;
        self.user_bash_running.store(false, Ordering::SeqCst);
        self.notify_session_input_checkpoint_change();
        let result = result?;
        self.emit(AgentSessionEvent::BashEnd {
            exit_code: result.exit_code,
            cancelled: result.cancelled,
            truncated: result.truncated,
            full_output_path: result.full_output_path.clone(),
            error_message: None,
            transient: None,
            run_id: None,
        });
        let session = self.clone();
        tokio::spawn(async move {
            session.drain_queued_messages_after_bash().await;
        });
        Ok(result)
    }

    pub(super) async fn drain_queued_messages_after_bash(self: &Arc<Self>) {
        self.agent.wait_for_idle().await;
        self.schedule_session_input_pump();
    }

    pub(super) async fn run_user_bash_locked(
        self: &Arc<Self>,
        command: &str,
        exclude_from_context: Option<bool>,
        _controller: CancellationToken,
    ) -> Result<BashResult, String> {
        // `const eventResult = await this._extensionRunner.emitUserBash({...})` (agent-session.ts:12415-12420).
        // Extensions may replace the result outright (`result`) or supply the operations
        // `executeBash` should run instead of the built-in local shell (`operations`, types.ts
        // `UserBashEventResult`). Without this dispatch both are unreachable.
        let event_result = match self.extension_runner() {
            Some(runner) => {
                runner
                    .emit_user_bash(serde_json::json!({
                        "type": "user_bash",
                        "command": command,
                        "excludeFromContext": exclude_from_context.unwrap_or(false),
                        "cwd": self.session_manager.lock().unwrap().get_cwd(),
                    }))
                    .await
            }
            None => None,
        };
        // NOTE: `this._extensionRunner.emitUserBash(...)` is unguarded in the TypeScript;
        // this port reaches the runner through the optional `extensionRunnerRef`, so a
        // session without a runner behaves like a session whose extensions all returned void.

        // `this._emit({ type: "bash_start", ... })` happens AFTER the extension dispatch
        // (agent-session.ts:12428-12433), so a handler that runs for the whole dispatch window
        // is not reported as an already-started bash.
        self.emit(AgentSessionEvent::BashStart {
            command: command.to_string(),
            exclude_from_context: exclude_from_context.unwrap_or(false),
            transient: None,
            run_id: None,
        });
        // `if (eventResult?.result) { ... }` (agent-session.ts:12436-12448).
        if let Some(result) = event_result
            .as_ref()
            .and_then(|result| result.result.as_ref())
        {
            let result = BashResult {
                output: result.output.clone(),
                exit_code: result.exit_code,
                cancelled: result.cancelled,
                truncated: result.truncated,
                full_output_path: result.full_output_path.clone(),
            };
            if !result.output.is_empty() {
                self.emit(AgentSessionEvent::BashOutput {
                    chunk: result.output.clone(),
                });
            }
            self.record_bash_result(command, &result, exclude_from_context);
            return Ok(result);
        }
        // `if (this._userBashAbortRequested)` (agent-session.ts:12452-12460): an abort that
        // arrived during the extension dispatch has no abort controller to act on yet.
        if self.user_bash_abort_requested.load(Ordering::SeqCst) {
            let result = BashResult {
                output: String::new(),
                exit_code: None,
                cancelled: true,
                truncated: false,
                full_output_path: None,
            };
            self.record_bash_result(command, &result, exclude_from_context);
            return Ok(result);
        }
        // `operations: eventResult?.operations` (agent-session.ts:12464).
        let operations = event_result
            .and_then(|result| result.operations)
            .map(|operations| {
                Arc::new(ExtensionBashOperations { operations })
                    as Arc<dyn crate::core::tools::BashOperations>
            });
        let weak = Arc::downgrade(self);
        match self
            .execute_bash_with_operations(
                command,
                Some(Arc::new(move |chunk| {
                    if let Some(session) = weak.upgrade() {
                        session.emit(AgentSessionEvent::BashOutput {
                            chunk: chunk.to_string(),
                        });
                    }
                })),
                exclude_from_context,
                operations,
            )
            .await
        {
            Ok(result) => Ok(result),
            Err(error) => {
                let result = BashResult {
                    output: format!("bash failed: {error}"),
                    exit_code: None,
                    cancelled: false,
                    truncated: false,
                    full_output_path: None,
                };
                self.record_bash_result(command, &result, exclude_from_context);
                Ok(result)
            }
        }
    }

    pub fn record_bash_result(
        &self,
        command: &str,
        result: &BashResult,
        exclude_from_context: Option<bool>,
    ) {
        let message = BashExecutionMessage {
            role: "bashExecution".to_string(),
            command: command.to_string(),
            output: result.output.clone(),
            exit_code: result.exit_code,
            cancelled: result.cancelled,
            truncated: result.truncated,
            full_output_path: result.full_output_path.clone(),
            timestamp: now_ms_i64(),
            exclude_from_context,
        };
        if self.is_streaming() {
            self.pending_bash_messages.lock().unwrap().push(message);
        } else {
            let message =
                agent_message_from_value(&serde_json::to_value(message).expect("bash message"));
            let mut state = self.agent.state();
            state.messages.push(message.clone());
            self.agent.set_state(state);
            let _ = self.session_manager.lock().unwrap().append_message(message);
        }
    }

    pub(super) fn flush_pending_bash_messages(&self) {
        let pending = std::mem::take(&mut *self.pending_bash_messages.lock().unwrap());
        for bash in pending {
            let message =
                agent_message_from_value(&serde_json::to_value(bash).expect("bash message"));
            let mut state = self.agent.state();
            state.messages.push(message.clone());
            self.agent.set_state(state);
            let _ = self.session_manager.lock().unwrap().append_message(message);
        }
    }

    pub fn abort_bash(&self) {
        if self.user_bash_running.load(Ordering::SeqCst) {
            self.user_bash_abort_requested.store(true, Ordering::SeqCst);
        }
        for controller in self.bash_abort_controllers.lock().unwrap().iter() {
            controller.cancel();
        }
    }

    pub fn is_bash_running(&self) -> bool {
        self.user_bash_running.load(Ordering::SeqCst)
            || !self.bash_abort_controllers.lock().unwrap().is_empty()
    }

    pub fn has_pending_bash_messages(&self) -> bool {
        !self.pending_bash_messages.lock().unwrap().is_empty()
    }

    /// `getRlmMaxDepthStatus()`.
    pub fn get_rlm_max_depth_status(&self) -> RlmMaxDepthStatus {
        RlmMaxDepthStatus {
            max_depth: self.rlm_max_depth() as f64,
            source: self.rlm_max_depth_source.lock().unwrap().clone(),
        }
    }

    /// `setRlmMaxDepth(maxDepth, options)`.
    pub async fn set_rlm_max_depth(
        self: &Arc<Self>,
        max_depth: i64,
        global: bool,
    ) -> Result<SetRlmMaxDepthResult, String> {
        if !is_non_negative_integer(max_depth as f64) {
            return Err("rlmMaxDepth must be a non-negative integer".to_string());
        }
        *self.rlm_max_depth.lock().unwrap() = max_depth;
        *self.rlm_max_depth_source.lock().unwrap() = if global {
            RLM_MAX_DEPTH_SOURCE_GLOBAL.to_string()
        } else {
            RLM_MAX_DEPTH_SOURCE_CHAT.to_string()
        };
        let _ = self
            .session_manager
            .lock()
            .unwrap()
            .append_custom_entry_with_rollback(
                RLM_MAX_DEPTH_STATE_CUSTOM_TYPE,
                Some(serde_json::json!({ "maxDepth": max_depth as f64 })),
            );
        *self.rlm_max_depth.lock().unwrap() = max_depth;
        *self.rlm_max_depth_source.lock().unwrap() = RLM_MAX_DEPTH_SOURCE_CHAT.to_string();
        // Rebuild the base prompt and re-apply it to the live extension prompt.
        let old_base = self.base_system_prompt.lock().unwrap().clone();
        let rebuilt = self.rebuild_system_prompt(&self.get_active_tool_names());
        *self.base_system_prompt.lock().unwrap() = rebuilt;
        {
            let mut state = self.agent.state();
            state.system_prompt =
                self.refresh_extension_system_prompt(&state.system_prompt, &old_base);
            self.agent.set_state(state);
        }

        let mut global_error: Option<String> = None;
        if global {
            // The settings lock is synchronous, so the write queue is flushed in
            // place instead of holding the guard across an await.
            let (stale, errors) = {
                let mut manager = self.settings_manager.lock().unwrap();
                manager.flush_sync();
                let stale = manager.drain_errors(Some("global"));
                manager.set_rlm_max_depth(max_depth as f64);
                manager.flush_sync();
                (stale, manager.drain_errors(Some("global")))
            };
            for error in stale {
                let _ = error;
            }
            let joined = errors
                .iter()
                .map(|error| error.error.message.clone())
                .collect::<Vec<_>>()
                .join("; ");
            if !joined.is_empty() {
                global_error = Some(joined);
            }
        }

        let source = self.rlm_max_depth_source.lock().unwrap().clone();
        Ok(SetRlmMaxDepthResult {
            max_depth: max_depth as f64,
            source,
            global_saved: global && global_error.is_none(),
            global_error,
        })
    }

    /// `setSessionName(name)`.
    pub fn set_session_name(&self, name: &str) -> Result<(), String> {
        self.session_manager
            .lock()
            .unwrap()
            .append_session_info(name)?;
        // Synchronous subscribers read the persisted name through the same mutex.
        self.emit(AgentSessionEvent::SessionInfoChanged {
            name: Some(name.to_string()),
        });
        Ok(())
    }

    /// `navigateTree(targetId, options)`.
    pub async fn navigate_tree(
        self: &Arc<Self>,
        target_id: &str,
        summarize: Option<bool>,
        editor_text: Option<&str>,
    ) -> Result<(), String> {
        self.navigate_tree_under_pause(target_id, summarize, editor_text)
            .await
    }

    /// `_navigateTree(targetId, options)`.
    pub(super) async fn navigate_tree_inner(
        self: &Arc<Self>,
        target_id: &str,
        summarize: Option<bool>,
        editor_text: Option<&str>,
    ) -> Result<(), String> {
        self.abort_branch_summary();
        // REPAIR CURSOR: the TypeScript returns
        // `{ editorText, cancelled, aborted, summaryEntry }` (agent-session.ts:12648-12653)
        // and reads/writes it from `_navigateTreeUnderPause`; this member keeps
        // `Result<(), String>` because every caller in the crate (the in-process
        // adapter at `core/agent_session_runtime/in_process_adapter.rs:338`) already
        // documents that shape as blocked. Add a Rust result struct to
        // `core/agent_session.rs` and return it here to unblock that seam.
        //
        // REPAIR CURSOR: `session_before_tree` / `session_tree` extension handoff
        // (agent-session.ts:12737-12762, 12851-12857) needs the `TreePreparation`
        // projection of `collect_entries_for_branch_summary`; it is not wired yet.
        let target_entry = self
            .session_manager
            .lock()
            .unwrap()
            .get_entry(target_id)
            .ok_or_else(|| format!("Entry {target_id} not found"))?;
        let old_leaf_id = self.session_manager.lock().unwrap().get_leaf_id();
        if old_leaf_id.as_deref() == Some(target_id) {
            return Ok(());
        }
        let entries: Vec<CompactionSessionEntry> = self
            .session_manager
            .lock()
            .unwrap()
            .get_branch(None)
            .iter()
            .filter_map(compaction_session_entry_from)
            .collect();
        let prepared = prepare_branch_entries(&entries, f64::INFINITY);
        if summarize.unwrap_or(false) && !prepared.messages.is_empty() {
            let Some(model) = self.model() else {
                return Err("No model available for summarization".to_string());
            };
            let auth = self.get_required_request_auth(&model).await?;
            let settings = self
                .settings_manager
                .lock()
                .unwrap()
                .get_branch_summary_settings();
            let controller = CancellationToken::new();
            *self.branch_summary_abort_controller.lock().unwrap() = Some(controller.clone());
            let result = generate_branch_summary(
                &entries,
                GenerateBranchSummaryOptions {
                    model,
                    api_key: auth.api_key.clone(),
                    headers: Some(
                        auth.headers
                            .clone()
                            .into_iter()
                            .map(|(name, value)| (name, Value::String(value)))
                            .collect(),
                    ),
                    signal: Some(controller.clone()),
                    custom_instructions: editor_text.map(|value| value.to_string()),
                    replace_instructions: None,
                    // `generateBranchSummary` takes the compaction module's
                    // `ProviderRetryPolicy`; the resolver reports the same fields.
                    retry: Some({
                        let policy = provider_retry_policy(&self.settings_manager.lock().unwrap());
                        crate::core::compaction::compaction::ProviderRetryPolicy {
                            enabled: policy.enabled,
                            max_retries: policy.max_retries.max(0.0) as u32,
                            base_delay_ms: policy.base_delay_ms,
                            max_retry_delay_ms: policy.max_retry_delay_ms,
                        }
                    }),
                    reserve_tokens: Some(settings.reserve_tokens),
                },
            )
            .await;
            *self.branch_summary_abort_controller.lock().unwrap() = None;
            if result.aborted == Some(true) {
                return Ok(());
            }
            if let Some(error) = result.error.clone() {
                return Err(error);
            }
            if let Some(summary) = result.summary.clone() {
                let details = Some(serde_json::json!({
                    "readFiles": result.read_files.clone().unwrap_or_default(),
                    "modifiedFiles": result.modified_files.clone().unwrap_or_default(),
                }));
                let new_leaf_id: Option<&str> = if target_entry.get("type").and_then(Value::as_str)
                    == Some("message")
                    && target_entry
                        .get("message")
                        .and_then(|message| message.get("role"))
                        .and_then(Value::as_str)
                        == Some("user")
                {
                    target_entry.get("parentId").and_then(Value::as_str)
                } else if target_entry.get("type").and_then(Value::as_str) == Some("custom_message")
                {
                    target_entry.get("parentId").and_then(Value::as_str)
                } else {
                    Some(target_id)
                };
                let _ = self.session_manager.lock().unwrap().branch_with_summary(
                    new_leaf_id,
                    &summary,
                    details,
                    None,
                    result.usage.as_ref(),
                );
            }
        } else {
            let new_leaf_id: Option<&str> = if target_entry.get("type").and_then(Value::as_str)
                == Some("message")
                && target_entry
                    .get("message")
                    .and_then(|message| message.get("role"))
                    .and_then(Value::as_str)
                    == Some("user")
            {
                target_entry.get("parentId").and_then(Value::as_str)
            } else if target_entry.get("type").and_then(Value::as_str) == Some("custom_message") {
                target_entry.get("parentId").and_then(Value::as_str)
            } else {
                Some(target_id)
            };
            match new_leaf_id {
                Some(leaf) => {
                    let _ = self.session_manager.lock().unwrap().branch(leaf);
                }
                None => self.session_manager.lock().unwrap().reset_leaf(),
            }
        }
        if let Some(editor_text) = editor_text {
            let _ = editor_text;
        }
        self.reload_goal_state_from_branch();
        self.reload_rlm_max_depth_from_branch();
        self.emit(AgentSessionEvent::TreeNavigated {
            target_id: target_id.to_string(),
        });
        Ok(())
    }

    /// `_navigateTreeUnderPause(targetId, options)`.
    pub(super) async fn navigate_tree_under_pause(
        self: &Arc<Self>,
        target_id: &str,
        summarize: Option<bool>,
        editor_text: Option<&str>,
    ) -> Result<(), String> {
        let pause = self.acquire_queued_work_pause();
        // The queued future lives inside the mutex and is not `Clone`, so the
        // previous tail is taken out and awaited while the new tail is installed.
        let previous = {
            let mut slot = self.branch_navigation_queue.lock().unwrap();
            std::mem::replace(&mut *slot, Box::pin(async { Ok(()) }))
        };
        let previous = previous;
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let tail: BoxFuture<Result<(), String>> = Box::pin(async move {
            let _ = rx.await;
            Ok(())
        });
        *self.branch_navigation_queue.lock().unwrap() = tail;
        let _ = previous.await;
        let result = self
            .navigate_tree_inner(target_id, summarize, editor_text)
            .await;
        pause.release();
        let _ = tx.send(());
        result
    }

    /// `getUserMessagesForForking()`.
    pub fn get_user_messages_for_forking(&self) -> Vec<UserMessageForkEntry> {
        let entries = self.session_manager.lock().unwrap().get_branch(None);
        entries
            .iter()
            .filter_map(|entry| {
                if entry.get("type").and_then(Value::as_str) != Some("message") {
                    return None;
                }
                let message = entry.get("message")?;
                if message.get("role").and_then(Value::as_str) != Some("user") {
                    return None;
                }
                let entry_id = entry.get("id").and_then(Value::as_str)?.to_string();
                let text = self.extract_user_message_text(message.get("content")?);
                Some(UserMessageForkEntry { entry_id, text })
            })
            .collect()
    }

    /// `_extractUserMessageText(content)`.
    pub(super) fn extract_user_message_text(&self, content: &Value) -> String {
        match content {
            Value::String(text) => text.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|part| {
                    if part.get("type").and_then(Value::as_str) == Some("text") {
                        part.get("text")
                            .and_then(Value::as_str)
                            .map(|text| text.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }

    /// `getSessionStats()` (agent-session.ts:12898-12941).
    pub fn get_session_stats(&self) -> SessionStats {
        let entries = self.session_manager.lock().unwrap().get_branch(None);
        let messages = self.messages();
        let user_messages = messages.iter().filter(|m| m.role() == "user").count() as i64;
        let assistant_messages = messages.iter().filter(|m| m.role() == "assistant").count() as i64;
        let tool_results = messages.iter().filter(|m| m.role() == "toolResult").count() as i64;
        let mut tool_calls = 0i64;
        let mut total_input = 0.0;
        let mut total_output = 0.0;
        let mut total_cache_read = 0.0;
        let mut total_cache_write = 0.0;
        let mut total_cost = 0.0;
        for message in &messages {
            if let AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) = message {
                tool_calls += assistant
                    .content
                    .iter()
                    .filter(|block| matches!(block, pi_ai::types::ContentBlock::ToolCall(_)))
                    .count() as i64;
                total_input += assistant.usage.input;
                total_output += assistant.usage.output;
                total_cache_read += assistant.usage.cache_read;
                total_cache_write += assistant.usage.cache_write;
                total_cost += assistant.usage.cost.total;
            }
        }
        SessionStats {
            session_file: self.session_file(),
            session_id: self.session_id(),
            user_messages,
            assistant_messages,
            tool_calls,
            tool_results,
            total_messages: messages.len() as i64,
            tokens: SessionStatsTokens {
                input: total_input,
                output: total_output,
                cache_read: total_cache_read,
                cache_write: total_cache_write,
                total: total_input + total_output + total_cache_read + total_cache_write,
            },
            cost: total_cost,
            context_usage: self.get_context_usage().map(|usage| {
                crate::core::session_stats::ContextUsage {
                    tokens: usage.tokens,
                    context_window: usage.context_window,
                    percent: usage.percent,
                }
            }),
        }
    }

    /// `getContextUsage()` (agent-session.ts:12943-12987).
    pub fn get_context_usage(&self) -> Option<ContextUsage> {
        let model = self.model()?;
        let context_window = model.context_window;
        if context_window <= 0.0 {
            return None;
        }
        // After compaction, the last assistant usage reflects pre-compaction context
        // size. Only usage from an assistant that answered after the latest
        // compaction can be trusted; otherwise the count stays unknown.
        let branch_entries = self.session_manager.lock().unwrap().get_branch(None);
        if let Some(latest_compaction) = get_latest_compaction_entry(&branch_entries) {
            let compaction_index = branch_entries
                .iter()
                .rposition(|entry| entry == &latest_compaction);
            let mut has_post_compaction_usage = false;
            if let Some(compaction_index) = compaction_index {
                for index in (compaction_index + 1..branch_entries.len()).rev() {
                    let entry = &branch_entries[index];
                    if entry.get("type").and_then(Value::as_str) == Some("message") {
                        if let Some(message) = entry.get("message") {
                            match agent_message_from_value(message) {
                                AgentMessage::Message(pi_ai::types::Message::Assistant(
                                    assistant,
                                )) => {
                                    if assistant.stop_reason != STOP_REASON_ABORTED
                                        && assistant.stop_reason != STOP_REASON_ERROR
                                    {
                                        if calculate_context_tokens(&assistant.usage) > 0.0 {
                                            has_post_compaction_usage = true;
                                        }
                                    }
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            if !has_post_compaction_usage {
                return Some(ContextUsage {
                    tokens: None,
                    context_window,
                    percent: None,
                });
            }
        }
        let estimate = estimate_context_tokens(&self.messages());
        Some(ContextUsage {
            tokens: Some(estimate.tokens),
            context_window,
            percent: Some((estimate.tokens / context_window) * 100.0),
        })
    }

    /// `_rlmSessionDirForReading()`.
    pub(super) fn rlm_session_dir_for_reading(&self) -> Option<String> {
        self.rlm_session_dir.clone().or_else(|| {
            self.session_manager
                .lock()
                .unwrap()
                .get_session_artifact_dir()
        })
    }

    /// `_contextWindowResolver()`.
    pub(super) fn context_window_resolver(&self) -> ContextWindowResolver {
        let registry = self.model_registry.clone();
        Arc::new(move |provider: &str, model_id: &str| {
            registry
                .lock()
                .unwrap()
                .find(provider, model_id)
                .map(|model| model.context_window)
        })
    }

    /// `_subtractUnindexedChildUsage(ownUsage, entries)`.
    pub(super) fn subtract_unindexed_child_usage(
        &self,
        own_usage: Usage,
        entries: &[SessionEntry],
    ) -> Usage {
        let unindexed = self.rlm_unindexed_child_usage.lock().unwrap();
        if unindexed.is_empty() {
            return own_usage;
        }
        let indexed: HashSet<i64> = entries
            .iter()
            .filter_map(|entry| entry.get("timestamp").and_then(Value::as_i64))
            .collect();
        let mut usage = own_usage;
        for (timestamp, child_usage) in unindexed.iter() {
            if indexed.contains(timestamp) {
                continue;
            }
            subtract_assistant_usage(&mut usage, child_usage);
        }
        usage
    }

    /// `_ownUsageMemo` accessor.
    pub(super) fn own_usage_memo(&self) -> Option<OwnUsageMemo> {
        self.own_usage_memo.lock().unwrap().clone()
    }

    /// `_setOwnUsageMemo(memo)`.
    pub(super) fn set_own_usage_memo(&self, memo: Option<OwnUsageMemo>) {
        *self.own_usage_memo.lock().unwrap() = memo;
    }

    /// `createReplacedSessionContext()`.
    pub fn create_replaced_session_context(&self) -> ReplacedSessionContext {
        let _ = &self.extension_runner_ref;
        ReplacedSessionContext {
            send_message: None,
            send_user_message: None,
        }
    }

    /// `hasExtensionHandlers(eventType)`.
    pub fn has_extension_handlers(&self, event_type: &str) -> bool {
        self.extension_runner()
            .map(|runner| runner.has_handlers(event_type))
            .unwrap_or(false)
    }

    /// `get extensionRunner()`.
    pub fn extension_runner(&self) -> Option<ExtensionRunner> {
        self.extension_runner_ref.current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extensions::types::BashOperations as ExtensionBashOperationsTrait;
    use std::sync::atomic::{AtomicI64, AtomicUsize};

    /// Records what the session forwarded to an extension-supplied `user_bash`
    /// `operations` value and streams one chunk back.
    struct RecordingExtensionOperations {
        command: Mutex<String>,
        cwd: Mutex<String>,
        exit_code: AtomicI64,
        exec_calls: Arc<AtomicUsize>,
    }

    impl ExtensionBashOperationsTrait for RecordingExtensionOperations {
        fn exec(
            &self,
            command: String,
            cwd: String,
            on_data: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
            signal: Option<CancellationToken>,
            timeout: Option<f64>,
            env: Option<Map<String, Value>>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Option<i64>, String>> + Send>>
        {
            self.exec_calls.fetch_add(1, Ordering::SeqCst);
            *self.command.lock().unwrap() = command;
            *self.cwd.lock().unwrap() = cwd;
            assert!(signal.is_none());
            assert!(timeout.is_none());
            assert!(env.is_none());
            let exit_code = self.exit_code.load(Ordering::SeqCst);
            Box::pin(async move {
                on_data(b"streamed".to_vec());
                Ok(if exit_code < 0 { None } else { Some(exit_code) })
            })
        }
    }

    fn recording_operations(
        exit_code: i64,
    ) -> (Arc<RecordingExtensionOperations>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let operations = Arc::new(RecordingExtensionOperations {
            command: Mutex::new(String::new()),
            cwd: Mutex::new(String::new()),
            exit_code: AtomicI64::new(exit_code),
            exec_calls: calls.clone(),
        });
        (operations, calls)
    }

    /// `operations: eventResult?.operations` (agent-session.ts:12464) must reach the
    /// executor unchanged: command, cwd, streamed output, and exit code.
    #[tokio::test]
    async fn extension_bash_operations_forward_command_cwd_and_exit_code() {
        let (recording, calls) = recording_operations(3);
        let adapter = ExtensionBashOperations {
            operations: recording.clone(),
        };
        let streamed: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let sink = streamed.clone();
        let result = crate::core::bash_executor::execute_bash_with_operations(
            "echo hi",
            "/tmp",
            Arc::new(adapter),
            Some(crate::core::bash_executor::BashExecutorOptions {
                on_chunk: Some(Arc::new(move |chunk| sink.lock().unwrap().push_str(chunk))),
                signal: None,
            }),
        )
        .await
        .expect("operations-backed bash succeeds");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(*recording.command.lock().unwrap(), "echo hi");
        assert_eq!(*recording.cwd.lock().unwrap(), "/tmp");
        assert_eq!(result.exit_code, Some(3));
        assert_eq!(result.output, "streamed");
        // The extension's stream is forwarded to the caller's chunk sink, not just returned.
        assert_eq!(*streamed.lock().unwrap(), "streamed");
    }

    /// A killed (`undefined`) exit code stays `None` instead of becoming `0`.
    #[tokio::test]
    async fn extension_bash_operations_preserve_absent_exit_code() {
        let (recording, _calls) = recording_operations(-1);
        let adapter = ExtensionBashOperations {
            operations: recording,
        };
        let result = crate::core::bash_executor::execute_bash_with_operations(
            "kill me",
            "/tmp",
            Arc::new(adapter),
            None,
        )
        .await
        .expect("operations-backed bash succeeds");
        assert_eq!(result.exit_code, None);
    }

    /// JS exit codes are numbers, not `i32`; an out-of-range value fails loudly rather
    /// than wrapping into a plausible-looking exit status.
    #[tokio::test]
    async fn extension_bash_operations_reject_out_of_range_exit_code() {
        let (recording, _calls) = recording_operations(i64::from(i32::MAX) + 1);
        let adapter = ExtensionBashOperations {
            operations: recording,
        };
        let error = crate::core::bash_executor::execute_bash_with_operations(
            "huge",
            "/tmp",
            Arc::new(adapter),
            None,
        )
        .await
        .expect_err("out-of-range exit code must be reported");
        assert!(error.contains("out-of-range exit code"), "{error}");
    }
}
