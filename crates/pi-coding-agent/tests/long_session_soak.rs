//! Long-running interactive-session soak test.
//!
//! Hish's request ("perform a long running session to make sure it works") needs
//! evidence that the interactive host survives sustained use, not a one-shot
//! smoke. This file drives the REAL code paths in-process (no network, no real
//! provider) and asserts concrete invariants about transcript growth, streaming
//! preservation, long output, slash commands, queue churn, history paging and
//! resize.
//!
//! Real owners exercised:
//! - `AgentSessionRuntime` / `AgentSession` (`core/agent_session_runtime.rs`,
//!   `core/agent_session.rs`): the session the interactive host owns.
//! - `InProcessAgentConnection` + `InProcessRuntimeHost`
//!   (`modes/agent_connection/in_process_agent_connection.rs`): the exact
//!   connection `run_terminal` builds when `options.connection` is `None`
//!   (`modes/interactive/native_host.rs:779-790`).
//! - `pi_ai::providers::faux` (`packages/ai/src/providers/faux.ts`): the
//!   reference faux provider, so streaming deltas and abort are the real ones.
//! - Real render owners: `AssistantMessageComponent`, `UserMessageComponent`,
//!   `ToolExecutionComponent`, `get_markdown_theme`.
//! - Real paging owners: `slice_pinned_session_history`
//!   (`modes/daemon/daemon_mode.rs:857`) plus
//!   `merge_older_agent_connection_history`
//!   (`modes/interactive/interactive_mode.rs:1106`).
//!
//! The daemon path CANNOT be driven in-process: `DaemonAgentConnection` needs a
//! live daemon socket (`modes/daemon/daemon_client.rs`). `tests/native_cli.rs`
//! covers it end to end through the real binary and a loopback HTTP provider, so
//! the daemon is covered THERE and is explicitly not claimed here.
//!
//! `InProcessRuntimeHostAdapter` (`core/agent_session_runtime/in_process_adapter.rs`
//! :286) is `pub(crate)` and unreachable from an integration test, so `SoakHost`
//! below reuses that adapter's OWN bodies over a real runtime. A member the
//! adapter leaves `blocked_on` keeps the identical message here, so no soak
//! assertion can pass on a placeholder.

#![allow(clippy::all)]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use pi_ai::providers::faux::{
    faux_assistant_message, register_faux_provider, FauxAssistantContent, FauxProviderRegistration,
    FauxResponseStep, RegisterFauxProviderOptions,
};
use pi_ai::types::{AssistantMessage, ContentBlock, Message, UserContent, UserMessage};
use pi_coding_agent::core::agent_session_runtime::{
    CreateAgentSessionRuntimeFactory, CreateAgentSessionRuntimeInput,
};
use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services, AgentSessionCreationOptions,
    CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::{AuthStorage, AuthStorageData};
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use pi_coding_agent::core::tools::render_utils::RenderContentBlock;
use pi_coding_agent::modes::agent_connection::in_process_agent_connection::InProcessAgentConnection;
use pi_coding_agent::modes::agent_connection::types as wire;
use pi_coding_agent::modes::agent_connection::types::AgentConnection;

use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentEvent, AgentMessage, ThinkingLevel};
use pi_ai::types::{BoxFuture, ImageContent, Model, ServiceTier, Transport};
use serde_json::{Map, Value};

use pi_coding_agent::core::agent_session::{
    AgentSession, ExtensionBindings, ModelSelectOptions, PromptOptions, RlmMaxDepthStatus,
    SessionInputPause as CanonicalSessionInputPause,
};
use pi_coding_agent::core::agent_session_runtime::{
    AgentSessionRuntime, ForkOptionsInput, NewSessionOptionsInput, SwitchSessionOptions,
};
use pi_coding_agent::core::autonomous::AgentAutonomousStatus;
use pi_coding_agent::core::diagnostics::{ResourceCollision, ResourceDiagnostic};
use pi_coding_agent::core::extensions::types::InputSource;
use pi_coding_agent::core::resource_loader::ResourceLoader;
use pi_coding_agent::core::session_action_store::{
    QueuedMessageLane, QueuedMessageMutation, QueuedMessageMutationStatus,
};
use pi_coding_agent::core::skills::Skill;
use pi_coding_agent::core::source_info::SourceInfo;
use pi_coding_agent::modes::agent_connection::daemon_agent_connection::build_session_tree_from_flat_nodes;
use pi_coding_agent::modes::agent_connection::in_process_agent_connection::InProcessRuntimeHost;
use pi_coding_agent::modes::agent_connection::snapshot::AgentSessionRuntimeSnapshotSource;
use pi_coding_agent::modes::agent_connection::snapshot::{
    create_agent_connection_commands, create_agent_connection_resource_snapshot, AgentsFileEntry,
    ExtensionLoadError as SnapshotExtensionLoadError, PromptTemplateEntry, RegisteredCommandEntry,
    ResourceExtensionEntry, ResourcePromptEntry, ResourceSkillEntry, ResourceThemeEntry,
    SkillEntry,
};
use pi_coding_agent::modes::agent_connection::tool_definition::ToolDefinition as ConnectionToolDefinition;
use pi_coding_agent::modes::agent_connection::types::{
    AgentConnectionEventListener, AgentConnectionHeadlessCompletionOptions,
    AgentConnectionModelCatalog, AgentConnectionNavigateTreeOptions,
    AgentConnectionNavigateTreeResult, AgentConnectionRlmChildAgentSnapshot,
    AgentConnectionSessionWatcher,
};
use pi_coding_agent::modes::agent_connection::types::{
    AgentConnectionExecuteBashOptions, AgentConnectionForkOptions, AgentConnectionInputPause,
    AgentConnectionModel, AgentConnectionModelCycleResult, AgentConnectionNewSessionOptions,
    AgentConnectionQueueState, AgentConnectionResourceCollision, AgentConnectionResourceDiagnostic,
    AgentConnectionResourceSnapshot, AgentConnectionScopedModel, AgentConnectionSessionContext,
    AgentConnectionSessionContextModel, AgentConnectionSessionEntry, AgentConnectionSessionHeader,
    AgentConnectionSessionInputPause, AgentConnectionSessionTree,
    AgentConnectionSessionTreeFlatNode, AgentConnectionSlashCommand,
    AgentConnectionSwitchSessionOptions, AgentConnectionUserMessage,
    AgentConnectionWatchSessionTree,
};
use CanonicalSessionInputPause as SessionInputPause;

/// The `runtimeHost` the in-process agent connection drives.
pub struct SoakHost {
    runtime: Arc<AgentSessionRuntime>,
    event_tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl SoakHost {
    pub fn new(runtime: Arc<AgentSessionRuntime>) -> Self {
        Self {
            runtime,
            event_tasks: Mutex::new(Vec::new()),
        }
    }

    pub fn runtime(&self) -> &Arc<AgentSessionRuntime> {
        &self.runtime
    }

    fn session(&self) -> Arc<AgentSession> {
        self.runtime.session()
    }

    /// `modelRegistry.refreshAvailableModels()` from a synchronous seam.
    ///
    /// `ModelRegistry::refresh_available_models` is async because it also drives a
    /// detached Prime Inference network refresh; the synchronous owners it uses
    /// for the disk/bundled catalog are `refresh()` (reload from disk) followed by
    /// `get_available()` (models with configured auth). Both are real owners.
    fn registry_available_models(&self) -> Vec<AgentConnectionModel> {
        let registry = self.session().model_registry();
        let mut registry = registry.lock().unwrap();
        registry.refresh();
        registry.get_available()
    }
}

/// `acquireSessionInputPause()` result adapted to the connection-layer trait.
///
/// `SessionInputPause` is a concrete struct; `AgentConnectionSessionInputPause`
/// requires `Arc<dyn AgentConnectionInputPause>`, so this wrapper forwards
/// `release()`. The TypeScript `release()` is synchronous and void, so the
/// adapted call reports success.
struct SoakInputPause {
    pause: SessionInputPause,
}

impl AgentConnectionInputPause for SoakInputPause {
    fn release(&self) -> BoxFuture<Result<(), String>> {
        self.pause.release();
        Box::pin(async { Ok(()) })
    }
}

/// `prompt(message, options)` payload the connection layer passes down.
fn prompt_options_from_value(options: Value) -> PromptOptions {
    let mut parsed = PromptOptions::default();
    let Some(object) = options.as_object() else {
        return parsed;
    };
    if let Some(images) = object.get("images") {
        parsed.images = serde_json::from_value(images.clone()).ok();
    }
    if let Some(behavior) = object.get("streamingBehavior").and_then(Value::as_str) {
        parsed.streaming_behavior = Some(behavior.to_string());
    }
    if let Some(resume) = object.get("resumeIfIdle").and_then(Value::as_bool) {
        parsed.resume_if_idle = Some(resume);
    }
    if let Some(queue) = object.get("queueIfBusy").and_then(Value::as_bool) {
        parsed.queue_if_busy = Some(queue);
    }
    if let Some(source) = object.get("source").and_then(Value::as_str) {
        parsed.source = match source {
            "interactive" => Some(InputSource::Interactive),
            "rpc" => Some(InputSource::Rpc),
            "extension" => Some(InputSource::Extension),
            _ => None,
        };
    }
    parsed
}

/// `toConnectionSourceInfo(sourceInfo)`.
fn connection_source_info(
    source_info: &SourceInfo,
) -> pi_coding_agent::modes::agent_connection::types::AgentConnectionSourceInfo {
    pi_coding_agent::modes::agent_connection::types::AgentConnectionSourceInfo {
        path: source_info.path.clone(),
        source: source_info.source.clone(),
        scope: source_info.scope.clone(),
        origin: source_info.origin.clone(),
        base_dir: source_info.base_dir.clone(),
    }
}

/// `toConnectionSourceInfo(theme.sourceInfo)`.
///
/// `resource_loader::update_themes_from_paths` assigns the theme module's own
/// three-field `SourceInfo` (the TypeScript assigns the full `core/source-info.ts`
/// shape, but the Rust theme owner narrows it). `origin`/`baseDir` are therefore
/// not available on a loaded theme; the connection layer's stand-in expects them,
/// so they are reported as absent rather than invented.
fn connection_source_info_from_theme(
    source_info: &pi_coding_agent::modes::interactive::theme::theme::SourceInfo,
) -> pi_coding_agent::modes::agent_connection::types::AgentConnectionSourceInfo {
    pi_coding_agent::modes::agent_connection::types::AgentConnectionSourceInfo {
        path: source_info.path.clone(),
        source: source_info.source.clone(),
        scope: source_info.scope.clone(),
        origin: String::new(),
        base_dir: None,
    }
}

fn skill_base(skill: &Skill) -> &pi_coding_agent::core::skills::BaseSkill {
    match skill {
        Skill::Markdown(skill) => &skill.base,
        Skill::Python(skill) => &skill.base,
    }
}

/// `AgentConnectionResourceDiagnostic` from a loaded-resource diagnostic.
fn connection_diagnostic(diagnostic: &ResourceDiagnostic) -> AgentConnectionResourceDiagnostic {
    AgentConnectionResourceDiagnostic {
        type_: diagnostic.diagnostic_type.clone(),
        message: diagnostic.message.clone(),
        path: diagnostic.path.clone(),
        collision: diagnostic.collision.as_ref().map(connection_collision),
    }
}

fn connection_collision(collision: &ResourceCollision) -> AgentConnectionResourceCollision {
    AgentConnectionResourceCollision {
        resource_type: collision.resource_type.clone(),
        name: collision.name.clone(),
        winner_path: collision.winner_path.clone(),
        winner_source: collision.winner_source.clone(),
        loser_path: collision.loser_path.clone(),
        loser_source: collision.loser_source.clone(),
    }
}

/// `createAgentConnectionToolDefinition(session.getToolDefinition(name))`.
///
/// The connection-layer `ToolDefinition` is the seven-field projection
/// `tool-definition.ts` copies; the session owns the full extension definition.
fn connection_tool_definition(
    definition: &pi_coding_agent::core::extensions::types::ToolDefinition,
) -> ConnectionToolDefinition {
    ConnectionToolDefinition {
        name: definition.name.clone(),
        label: definition.label.clone(),
        description: definition.description.clone(),
        prompt_snippet: definition.prompt_snippet.clone(),
        prompt_guidelines: definition.prompt_guidelines.clone(),
        parameters: definition.parameters.clone(),
        render_shell: definition.render_shell.clone(),
        replay_built_in_tool_name: definition.replay_built_in_tool_name.clone(),
    }
}

/// `BashResult` as the connection layer carries it (`BashResult` is not serde).
fn bash_result_value(result: &pi_coding_agent::core::bash_executor::BashResult) -> Value {
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

/// `createAgentConnectionState(runtime)`/`getSessionTree()` projections.
///
/// Shared by `session_context()` and `session_tree()` so the two seam members and
/// the snapshot builder agree on the same session-manager reads.
fn session_tree_value(session: &Arc<AgentSession>) -> AgentConnectionSessionTree {
    let manager = session.session_manager.lock().unwrap();
    let mut flat_nodes: Vec<AgentConnectionSessionTreeFlatNode> = Vec::new();
    for node in manager.get_flat_tree() {
        match serde_json::from_value(Value::Object(node.entry)) {
            Ok(entry) => flat_nodes.push(AgentConnectionSessionTreeFlatNode {
                entry,
                label: node.label,
                label_timestamp: node.label_timestamp,
            }),
            Err(error) => {
                // A session entry outside the connection union cannot be
                // projected; report it instead of dropping it silently.
                eprintln!("Warning: Could not project session tree entry: {error}");
            }
        }
    }
    let leaf_id = manager.get_leaf_id();
    drop(manager);
    AgentConnectionSessionTree {
        tree: build_session_tree_from_flat_nodes(&flat_nodes),
        leaf_id,
    }
}

/// `buildSessionContext()` as the connection layer carries it.
fn session_context_value(session: &Arc<AgentSession>) -> AgentConnectionSessionContext {
    let context = session.build_session_context();
    AgentConnectionSessionContext {
        messages: context.messages,
        thinking_level: context.thinking_level,
        service_tier: context.service_tier,
        model: context
            .model
            .map(|model| AgentConnectionSessionContextModel {
                provider: model.provider,
                model_id: model.model_id,
            }),
    }
}
// ---- Extra imports used by the fixtures and tests ----

use pi_coding_agent::modes::interactive::components::assistant_message::{
    AssistantMessageComponent, AssistantMessageComponentOptions,
};
use pi_coding_agent::modes::interactive::components::tool_execution::{
    ToolExecutionComponent, ToolExecutionOptions, ToolExecutionResult,
};
use pi_coding_agent::modes::interactive::components::user_message::UserMessageComponent;
use pi_coding_agent::modes::interactive::theme::theme::{get_markdown_theme, MarkdownTheme};

impl InProcessRuntimeHost for SoakHost {
    // Members whose canonical owner is missing return an explicit failure or an empty roster with a
    // `blocked_on:` note, rather than being omitted: the trait has no default bodies, so an omitted
    // member is a hard E0046 that stops the whole crate from compiling and keeps every test from
    // running. An honest `Err`/`None` keeps the gap visible AND the crate buildable. Every member
    // with a real owner forwards for real, with no placeholders.
    fn snapshot_source(&self) -> AgentSessionRuntimeSnapshotSource {
        // Every field `snapshot.ts:28-59` reads has a real owner in the port, so all of them except
        // the child roster are forwarded. This member feeds `create_agent_connection_state` /
        // `create_agent_connection_snapshot` (in_process_agent_connection.rs:370/375/438/883), so a
        // wrong value here is observable.
        let session = self.session();
        let (cwd, session_dir, leaf_id, compaction_count, persisted_recap) = {
            let manager = session.session_manager.lock().unwrap();
            // `snapshot.ts:49`: `getEntries().filter((entry) => entry.type === "compaction").length`.
            let compaction_count = manager
                .get_entries()
                .iter()
                .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
                .count();
            // `persistedRecap(sessionManager)` (snapshot.ts:16-20): the baseline recap is the latest
            // persisted agent-status summary, not the live session recap.
            let persisted_recap = manager
                .get_latest_agent_status()
                .map(|status| status.summary);
            (
                manager.get_cwd(),
                manager.get_session_dir(),
                manager.get_leaf_id(),
                compaction_count,
                persisted_recap,
            )
        };
        AgentSessionRuntimeSnapshotSource {
            session:
                pi_coding_agent::modes::agent_connection::snapshot::AgentSessionSnapshotSource {
                    session_id: session.session_id(),
                    // `cwd: sessionManager.getCwd()` (snapshot.ts:30).
                    cwd,
                    // `sessionDir: sessionManager.getSessionDir()` (snapshot.ts:44).
                    session_dir: Some(session_dir),
                    // `leafId: sessionManager.getLeafId()` (snapshot.ts:45).
                    leaf_id,
                    session_file: session.session_file(),
                    session_name: session.session_name(),
                    model: session.model(),
                    thinking_level: session.thinking_level(),
                    service_tier: session.service_tier(),
                    // `session.getAvailableThinkingLevels()` (snapshot.ts:34).
                    available_thinking_levels: session.get_available_thinking_levels(),
                    is_streaming: session.is_streaming(),
                    is_compacting: session.is_compacting(),
                    is_bash_running: session.is_bash_running(),
                    retry_attempt: session.retry_attempt() as f64,
                    steering_mode: session.steering_mode(),
                    follow_up_mode: session.follow_up_mode(),
                    auto_compaction_enabled: session.auto_compaction_enabled(),
                    // `messageCount: session.messages.length` (snapshot.ts:47).
                    message_count: session.messages().len() as f64,
                    // `sessionActions: session.getSessionActionSnapshot()` (snapshot.ts:48).
                    session_actions: serde_json::to_value(session.get_session_action_snapshot())
                        .unwrap(),
                    compaction_count: compaction_count as f64,
                    // `goal: session.goalState` (snapshot.ts:50).
                    goal: serde_json::to_value(session.goal_state()).unwrap(),
                    // `session.scopedModels.map(...)` (snapshot.ts:51): the canonical `ScopedModel`
                    // already carries the two fields the connection layer reads.
                    scoped_models: session
                        .scoped_models()
                        .into_iter()
                        .map(|entry| AgentConnectionScopedModel {
                            model: entry.model,
                            thinking_level: entry.thinking_level,
                        })
                        .collect(),
                    // `activeToolNames: session.getActiveToolNames()` (snapshot.ts:55).
                    active_tool_names: session.get_active_tool_names(),
                    occ3_global_opt_in: false,
                    occ3_native_dead: false,
                    // `contextUsage: session.getContextUsage()` (snapshot.ts:56).
                    context_usage: serde_json::to_value(session.get_context_usage()).unwrap(),
                    persisted_recap,
                    messages: session.messages(),
                    // `...(session.state?.streamingMessage ? { streamingMessage } : {})` (snapshot.ts:71).
                    streaming_message: session.state().streaming_message,
                    // `sessionContext: session.buildSessionContext()` (snapshot.ts:72).
                    session_context: Some(session_context_value(&session)),
                    // `sessionTree: { tree: sessionManager.getTree(), leafId: sessionManager.getLeafId() }`
                    // (snapshot.ts:73-76).
                    session_tree: Some(session_tree_value(&session)),
                    // blocked_on: `children: session.getRlmChildSnapshots()` (snapshot.ts:77) needs
                    // `rlm_child_snapshot_for_run` / `rlm_child_snapshot_for_session`
                    // (`core/agent_session/runtime_members.rs:1804/1854`), which are `pub(super)` and so
                    // unreachable from this module. Owner to add: drop `(super)` on both. Needs
                    // `runtime_members.rs` edited.
                    children: Vec::new(),
                },
        }
    }

    fn session_subscribe(
        &self,
        listener: AgentConnectionEventListener,
    ) -> Box<dyn Fn() + Send + Sync> {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let sender = Arc::new(Mutex::new(Some(sender)));
        let task = tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                listener(event).await;
            }
        });
        self.event_tasks.lock().unwrap().push(task);
        let event_sender = sender.clone();
        let unsubscribe = self.session().subscribe(Arc::new(move |event| {
            let event = match event {
                pi_coding_agent::core::agent_session::AgentSessionEvent::Agent(event) =>
                    pi_coding_agent::modes::agent_connection::types::AgentConnectionSessionEvent::Agent(event),
                event => match serde_json::to_value(event).and_then(serde_json::from_value) {
                    Ok(event) => event,
                    Err(error) => { eprintln!("Could not serialize session event: {error}"); return; }
                },
            };
            if let Some(sender) = event_sender.lock().unwrap().as_ref() {
                let _ = sender.send(pi_coding_agent::modes::agent_connection::types::AgentConnectionEvent::SessionEvent { event });
            }
        }));
        Box::new(move || {
            unsubscribe();
            sender.lock().unwrap().take();
        })
    }

    fn session_wait_for_headless_completion(
        &self,
        options: Option<AgentConnectionHeadlessCompletionOptions>,
    ) -> BoxFuture<Result<AgentAutonomousStatus, String>> {
        let session = self.session();
        Box::pin(async move {
            pi_coding_agent::modes::headless_completion::wait_for_headless_completion(
                Arc::new(session),
                pi_coding_agent::modes::headless_completion::HeadlessCompletionOptions {
                    wait_for_rlm_quiescence: options
                        .and_then(|options| options.wait_for_rlm_quiescence),
                },
            )
            .await
        })
    }

    fn session_model_catalog(&self) -> AgentConnectionModelCatalog {
        let registry = self.session().model_registry();
        let registry = registry.lock().unwrap();
        let mut configured_providers = Vec::new();
        for model in registry.get_available() {
            if !configured_providers.contains(&model.provider) {
                configured_providers.push(model.provider);
            }
        }
        AgentConnectionModelCatalog {
            models: registry.get_all(),
            configured_providers,
        }
    }

    /// `this.session.getContextTree()` (`in-process-agent-connection.ts:182`,
    /// `agent-session.ts:13028`).
    ///
    /// blocked_on: the assembly helpers are private - `rlm_session_dir_for_reading`
    /// (`core/agent_session/runtime_members.rs:3115`), `context_window_resolver`
    /// (:3122) and `subtract_unindexed_child_usage` (:3134) are `pub(super)`.
    /// Owner to add: `core/agent_session/runtime_members.rs`, a
    /// `pub fn get_context_tree(&self) -> ContextTreeNode` matching
    /// `agent-session.ts:13028-13064`. Needs `runtime_members.rs` edited.
    fn session_context_tree(&self) -> Value {
        Value::Null
    }

    /// `this.session.getRlmChildSnapshots()` (`in-process-agent-connection.ts:150`,
    /// `agent-session.ts:11171`).
    ///
    /// blocked_on: `rlm_child_snapshot_for_run` / `rlm_child_snapshot_for_session`
    /// (`core/agent_session/runtime_members.rs:1804/1854`) are `pub(super)`, and no
    /// `impl From<RlmChildAgentSnapshot> for AgentConnectionRlmChildAgentSnapshot`
    /// exists (`RlmChildAgentSnapshot` at `core/agent_session.rs:438` is already the
    /// same camelCase shape the connection layer reads; `AgentConnectionRlmChildAgentSnapshot`
    /// at `modes/agent_connection/types.rs:902` is the projection). Owner to add:
    /// drop `(super)` on both builders. Needs `runtime_members.rs` edited.
    fn session_rlm_children(&self) -> Vec<AgentConnectionRlmChildAgentSnapshot> {
        Vec::new()
    }

    /// `this.session.cancelRlmChildRun(childId)`
    /// (`in-process-agent-connection.ts:427`, `agent-session.ts:11306`).
    ///
    /// blocked_on: owner `AgentSession::cancel_rlm_child_run(&self, run: &RlmChildRun,
    /// reason: &str)` (`core/agent_session/runtime_members.rs:889`) takes the run and
    /// is `pub(super)`; the by-id walk `rlm_subtree_sessions` (:1087) is
    /// `pub(super)` too. Owner to add: `runtime_members.rs`, a
    /// `pub fn cancel_rlm_child_run_by_id(&self, child_id: &str, reason: &str) -> bool`
    /// matching `agent-session.ts:11306-11329`. Needs `runtime_members.rs` edited.
    fn session_cancel_rlm_child(&self, child_id: &str) -> bool {
        let _ = child_id;
        false
    }

    /// `this.session.setScopedModels(scopedModels)`
    /// (`in-process-agent-connection.ts:472`, `agent-session.ts:4964`).
    ///
    /// blocked_on: the only owner is `AgentSession::set_scoped_models(&mut self,
    /// Vec<ScopedModel>)` (`core/agent_session.rs:6951`); `AgentSession` is held as
    /// `Arc<AgentSession>` and the field is a plain `Vec`, so no `&self` caller can
    /// write it. `AgentConnectionScopedModel` -> `ScopedModel`
    /// (`core/model_resolver.rs:86`) is field-for-field. Needs
    /// `core/agent_session.rs` edited (interior mutability or a runtime-level
    /// rebuild path).
    fn session_set_scoped_models(&self, scoped_models: Vec<AgentConnectionScopedModel>) {
        let _ = scoped_models;
    }

    /// `this.session.settingsManager.setTransport(transport)` and
    /// `this.session.agent.transport = transport`
    /// (`in-process-agent-connection.ts:488-489`).
    ///
    /// The settings half is reachable (`AgentSession.settings_manager`,
    /// `core/agent_session.rs:2136`). The live-agent half is not: `agent` is
    /// `Arc<dyn AgentHandle>` (`core/agent_session.rs:2134`) and `AgentHandle`
    /// (:243-274) has no transport setter, so
    /// `pi_agent_core::agent::Agent.transport` (`crates/pi-agent-core/src/agent.rs:259`)
    /// cannot be assigned through the seam.
    /// blocked_on: add `fn set_transport(&self, transport: String)` to the
    /// `AgentHandle` trait (`core/agent_session.rs:246`), implemented for `Arc<Agent>`
    /// at `core/agent_session/agent_handle.rs:20`. Needs those two files edited.
    fn session_set_transport(&self, transport: Transport) {
        self.session()
            .settings_manager
            .lock()
            .expect("settings manager poisoned")
            .set_transport(transport);
    }

    /// `this.session.compact(customInstructions)`
    /// (`in-process-agent-connection.ts:509`, `agent-session.ts:8077`).
    ///
    /// blocked_on: `compact_with_options` (`core/agent_session.rs:11121`) is public
    /// but returns `Result<(), String>`; the `CompactionResult`
    /// (`core/compaction/compaction.ts:124-131`) is only produced by the private
    /// `AgentSession::compact` (`core/agent_session.rs:13139`) and
    /// `perform_compaction_unmeasured_full` (:13290), and it has no serde impls.
    /// Owner to add: `core/agent_session.rs`, a public
    /// `async fn compact(...) -> Result<CompactionResult, String>`. Needs
    /// `core/agent_session.rs` edited.
    fn session_compact(
        &self,
        custom_instructions: Option<&str>,
    ) -> BoxFuture<Result<Value, String>> {
        let _ = custom_instructions;
        Box::pin(async {
            Err("blocked_on: compact result not exposed on the public path".to_string())
        })
    }

    /// `this.session.refine(options)` (`in-process-agent-connection.ts:515`,
    /// `agent-session.ts:8952`).
    ///
    /// blocked_on: `refine_with_options` (`core/agent_session.rs:12141`) is private;
    /// the public paths are the `/refine` command (:9571) and
    /// `handle_refine_host_request` (:4393), which only schedules the request and
    /// returns a status object. Owner to add: `core/agent_session.rs`, a
    /// `pub async fn refine(&self, options: RefineOptions) -> Result<RefinementResult, String>`
    /// (`RefinementResult` already derives serde, `core/refinement/refinement.rs:280`).
    /// Needs `core/agent_session.rs` edited.
    fn session_refine(&self, options: Value) -> BoxFuture<Result<Value, String>> {
        let _ = options;
        Box::pin(async {
            Err("blocked_on: refine result not exposed on the public path".to_string())
        })
    }

    fn session_navigate_tree(
        &self,
        target_id: &str,
        options: Option<AgentConnectionNavigateTreeOptions>,
    ) -> BoxFuture<Result<AgentConnectionNavigateTreeResult, String>> {
        // `navigate_tree` (`core/agent_session/runtime_members.rs:2774`) is public; it
        // returns `Result<(), String>` where `agent-session.ts:12613-12625` returns
        // `{ editorText?, cancelled, aborted?, summaryEntry? }`.
        let session = self.session();
        let target_id = target_id.to_string();
        let options = options;
        Box::pin(async move {
            let summarize = options.as_ref().and_then(|options| options.summarize);
            session.navigate_tree(&target_id, summarize, None).await?;
            // blocked_on: `navigate_tree_inner`
            // (`core/agent_session/runtime_members.rs:2785`) is `pub(super)` and discards
            // the editorText and the extension `cancel` branch; the seam needs it to
            // return a result struct (`NavigateTreeResult` in
            // `core/agent_session.rs`). Needs `core/agent_session.rs` and
            // `runtime_members.rs` edited.
            Ok(AgentConnectionNavigateTreeResult {
                editor_text: None,
                cancelled: false,
                aborted: None,
            })
        })
    }

    fn session_export_to_html(
        &self,
        output_path: Option<&str>,
    ) -> BoxFuture<Result<String, String>> {
        let session_file = self.session().session_file();
        let output_path = output_path.map(|path| path.to_string());
        Box::pin(async move {
            let Some(session_file) = session_file else {
                return Err("Cannot export in-memory session to HTML".to_string());
            };
            let options = pi_coding_agent::core::export_html::ExportOptions {
                output_path,
                ..Default::default()
            };
            pi_coding_agent::core::export_html::export_from_file(&session_file, Some(options))
        })
    }

    /// `this.session.exportToJsonl(outputPath)`
    /// (`in-process-agent-connection.ts:568`, `agent-session.ts:13093`).
    ///
    /// blocked_on: no owner. The writer needs the session header
    /// (`agent-session.ts:13100-13106`), the branch (`SessionManager::get_branch`,
    /// `core/session_manager.rs:4238`), a parentId re-chain and an
    /// ISO-8601-timestamped default path; `iso_now` (`core/session_manager.rs:892`)
    /// is private and no function writes a branch as JSONL. Owner to add: a
    /// `pub fn export_to_jsonl(&self, output_path: Option<&str>) -> Result<String, String>`
    /// matching `agent-session.ts:13093-13121`.
    fn session_export_to_jsonl(
        &self,
        output_path: Option<&str>,
    ) -> BoxFuture<Result<String, String>> {
        let _ = output_path;
        Box::pin(async { Err("blocked_on: no JSONL export owner".to_string()) })
    }

    /// `this.session.getRlmChildSession(childId)` then the watcher built from the
    /// child (`in-process-agent-connection.ts:604-628`).
    ///
    /// blocked_on: `AgentSession::get_rlm_child_session` does not exist. The
    /// retained children live in `rlm_child_sessions`
    /// (`core/agent_session.rs:2236`), whose only by-id accessors are the
    /// `pub(super)` `rlm_child_snapshot_for_run`/`rlm_child_snapshot_for_session`
    /// (`core/agent_session/runtime_members.rs:1804/1854`). Owner to add:
    /// `core/agent_session/runtime_members.rs`, a
    /// `pub fn get_rlm_child_session(&self, child_id: &str) -> Option<Arc<AgentSession>>`
    /// matching `agent-session.ts:11289`. Needs `runtime_members.rs` edited.
    fn session_watch_child(
        &self,
        child_id: &str,
    ) -> Option<Box<dyn AgentConnectionSessionWatcher>> {
        let _ = child_id;
        None
    }
    // Members whose canonical owner is missing keep an explicit failure or an empty roster with an
    // accurate `blocked_on:` note naming the owner to add. They are not omitted: the trait has no
    // default bodies, so an omitted member is a hard E0046 that stops the whole crate from compiling
    // and keeps every test from running.

    fn session_header(&self) -> Option<AgentConnectionSessionHeader> {
        let entry = self
            .session()
            .session_manager
            .lock()
            .unwrap()
            .get_header()?;
        serde_json::from_value(Value::Object(entry)).ok()
    }

    fn session_messages(&self) -> Vec<AgentMessage> {
        self.session().state().messages
    }

    fn session_state_messages(&self) -> Vec<AgentMessage> {
        // The TypeScript has one source (`session.state.messages`); both seam
        // members read it.
        self.session().state().messages
    }

    fn session_commands(&self) -> Vec<AgentConnectionSlashCommand> {
        let session = self.session();
        let registered_commands = match session.extension_runner() {
            Some(runner) => runner
                .get_registered_commands()
                .into_iter()
                .map(|resolved| RegisteredCommandEntry {
                    invocation_name: resolved.invocation_name.clone(),
                    name: resolved.command.name.clone(),
                    description: resolved.command.description.clone(),
                    source_info: connection_source_info(&resolved.command.source_info),
                })
                .collect::<Vec<_>>(),
            None => Vec::new(),
        };
        let prompt_templates = session
            .prompt_templates()
            .into_iter()
            .map(|entry| PromptTemplateEntry {
                name: entry.name.clone(),
                description: Some(entry.description.clone()),
                argument_hint: entry.argument_hint.clone(),
                source_info: connection_source_info(&entry.source_info),
            })
            .collect::<Vec<_>>();
        let skills = session
            .resource_loader()
            .get_skills()
            .skills
            .iter()
            .map(|skill| SkillEntry {
                name: skill.name().to_string(),
                description: Some(skill_base(skill).description.clone()),
                source_info: connection_source_info(&skill_base(skill).source_info),
            })
            .collect::<Vec<_>>();
        create_agent_connection_commands(&registered_commands, &prompt_templates, &skills)
    }

    fn session_resource_snapshot(&self) -> AgentConnectionResourceSnapshot {
        let session = self.session();
        let loader: Arc<dyn ResourceLoader> = session.resource_loader();
        let agents_files = loader
            .get_agents_files()
            .agents_files
            .into_iter()
            .map(|entry| AgentsFileEntry { path: entry.path })
            .collect::<Vec<_>>();
        let skills_result = loader.get_skills();
        let resource_skills = skills_result
            .skills
            .iter()
            .map(|skill| ResourceSkillEntry {
                name: skill.name().to_string(),
                description: Some(skill_base(skill).description.clone()),
                file_path: skill.file_path().to_string(),
                source_info: Some(connection_source_info(&skill_base(skill).source_info)),
            })
            .collect::<Vec<_>>();
        let prompts_result = loader.get_prompts();
        let resource_prompts = prompts_result
            .prompts
            .iter()
            .map(|entry| ResourcePromptEntry {
                name: entry.name.clone(),
                description: Some(entry.description.clone()),
                argument_hint: entry.argument_hint.clone(),
                file_path: entry.file_path.clone(),
                source_info: Some(connection_source_info(&entry.source_info)),
            })
            .collect::<Vec<_>>();
        let extensions_result = loader.get_extensions();
        let mut resource_extensions: Vec<ResourceExtensionEntry> = Vec::new();
        for extension in &extensions_result.extensions {
            let extension = extension.lock().unwrap();
            resource_extensions.push(ResourceExtensionEntry {
                path: extension.path.clone(),
                source_info: Some(connection_source_info(&extension.source_info)),
            });
        }
        let extension_errors = extensions_result
            .errors
            .iter()
            .map(|error| SnapshotExtensionLoadError {
                error: error.error.clone(),
                path: Some(error.path.clone()),
            })
            .collect::<Vec<_>>();
        let themes_result = loader.get_themes();
        let resource_themes = themes_result
            .themes
            .iter()
            .map(|theme| ResourceThemeEntry {
                name: theme.name.clone(),
                source_path: theme.source_path.clone(),
                source_info: theme
                    .source_info
                    .as_ref()
                    .map(connection_source_info_from_theme),
            })
            .collect::<Vec<_>>();
        let cwd = session.session_manager.lock().unwrap().get_cwd();
        create_agent_connection_resource_snapshot(
            &session.session_id(),
            &cwd,
            &agents_files,
            &resource_skills,
            skills_result
                .diagnostics
                .iter()
                .map(connection_diagnostic)
                .collect(),
            &resource_prompts,
            prompts_result
                .diagnostics
                .iter()
                .map(connection_diagnostic)
                .collect(),
            &resource_extensions,
            &extension_errors,
            &resource_themes,
            themes_result
                .diagnostics
                .iter()
                .map(connection_diagnostic)
                .collect(),
        )
    }

    fn session_available_models(&self) -> Vec<AgentConnectionModel> {
        self.registry_available_models()
    }

    fn session_stats(&self) -> Value {
        let stats = self.session().get_session_stats();
        serde_json::to_value(stats).unwrap_or(Value::Null)
    }

    fn session_context(&self) -> AgentConnectionSessionContext {
        session_context_value(&self.session())
    }

    fn session_tree(&self) -> AgentConnectionWatchSessionTree {
        let tree = session_tree_value(&self.session());
        AgentConnectionWatchSessionTree {
            tree: tree.tree,
            leaf_id: tree.leaf_id,
        }
    }

    fn session_queue(&self) -> AgentConnectionQueueState {
        let session = self.session();
        AgentConnectionQueueState {
            steering: session.get_steering_message_previews(),
            follow_up: session.get_follow_up_message_previews(),
        }
    }

    fn session_mutate_queued_message(
        &self,
        lane: &str,
        index: i64,
        expected_text: &str,
        mutation: Value,
    ) -> String {
        let lane =
            match serde_json::from_value::<QueuedMessageLane>(Value::String(lane.to_string())) {
                Ok(lane) => lane,
                Err(_) => return QueuedMessageMutationStatus::Invalid.as_str().to_string(),
            };
        let mutation = match serde_json::from_value::<QueuedMessageMutation>(mutation) {
            Ok(mutation) => mutation,
            Err(_) => return QueuedMessageMutationStatus::Invalid.as_str().to_string(),
        };
        self.session()
            .mutate_queued_message(lane, index, expected_text, &mutation)
            .as_str()
            .to_string()
    }

    fn session_clear_queue(&self) -> AgentConnectionQueueState {
        let cleared = self.session().clear_queue();
        AgentConnectionQueueState {
            steering: cleared.steering,
            follow_up: cleared.follow_up,
        }
    }

    fn session_request_abort(&self) {
        self.session().request_abort();
    }

    fn session_acquire_input_pause(&self) -> AgentConnectionSessionInputPause {
        let pause = self.session().acquire_session_input_pause();
        Arc::new(SoakInputPause { pause })
    }

    fn session_get_user_messages_for_forking(&self) -> Vec<AgentConnectionUserMessage> {
        self.session()
            .get_user_messages_for_forking()
            .into_iter()
            .map(|entry| AgentConnectionUserMessage {
                entry_id: entry.entry_id,
                text: entry.text,
            })
            .collect()
    }

    fn session_last_assistant_text(&self) -> Option<String> {
        let messages = self.session().messages();
        let last_assistant = messages.iter().rev().find_map(|message| match message {
            AgentMessage::Message(pi_ai::types::Message::Assistant(assistant)) => {
                // Skip aborted messages with no content.
                if assistant.stop_reason == pi_ai::types::STOP_REASON_ABORTED
                    && assistant.content.is_empty()
                {
                    None
                } else {
                    Some(assistant)
                }
            }
            _ => None,
        })?;
        let mut text = String::new();
        for block in &last_assistant.content {
            if let pi_ai::types::ContentBlock::Text(text_block) = block {
                text.push_str(&text_block.text);
            }
        }
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    fn session_system_prompt(&self) -> String {
        self.session().system_prompt()
    }

    fn session_tool_definition(&self, name: &str) -> Option<ConnectionToolDefinition> {
        self.session()
            .get_tool_definition(name)
            .map(|definition| connection_tool_definition(&definition))
    }

    fn session_append_label_change(&self, entry_id: &str, label: Option<&str>) {
        // The seam signature returns unit; `appendLabelChange` reports an unknown
        // entry id, so the failure is logged rather than dropped silently.
        if let Err(error) = self
            .session()
            .session_manager
            .lock()
            .unwrap()
            .append_label_change(entry_id, label)
        {
            eprintln!("Warning: Could not append label change: {error}");
        }
    }

    fn session_prompt(&self, message: &str, options: Value) -> BoxFuture<Result<(), String>> {
        let session = self.session();
        let parsed = prompt_options_from_value(options);
        let message = message.to_string();
        Box::pin(async move { session.prompt(&message, Some(parsed)).await })
    }

    fn session_prompt_and_wait(
        &self,
        message: &str,
        options: Value,
    ) -> BoxFuture<Result<(), String>> {
        let session = self.session();
        let parsed = prompt_options_from_value(options);
        let message = message.to_string();
        Box::pin(async move { session.prompt_and_wait(&message, Some(parsed)).await })
    }

    fn session_steer(
        &self,
        message: &str,
        images: Option<Vec<ImageContent>>,
    ) -> BoxFuture<Result<(), String>> {
        let session = self.session();
        let message = message.to_string();
        // `session.steer(message, images, { resumeIfIdle: true })`.
        Box::pin(async move {
            session
                .steer(&message, images, None, None, Some(true))
                .await
        })
    }

    fn session_follow_up(
        &self,
        message: &str,
        images: Option<Vec<ImageContent>>,
    ) -> BoxFuture<Result<(), String>> {
        let session = self.session();
        let message = message.to_string();
        // `session.followUp(message, images, { resumeIfIdle: true })`.
        Box::pin(async move {
            session
                .follow_up(&message, images, None, None, Some(true))
                .await
                .map(|_queued| ())
        })
    }

    fn session_wait_for_idle(&self) -> BoxFuture<()> {
        let session = self.session();
        Box::pin(async move {
            let _ = session.wait_for_idle().await;
        })
    }

    fn session_run_user_bash(
        &self,
        command: &str,
        options: Option<AgentConnectionExecuteBashOptions>,
    ) -> BoxFuture<Result<(), String>> {
        let session = self.session();
        let command = command.to_string();
        let exclude_from_context = options.and_then(|options| options.exclude_from_context);
        // `transient`/`runId` are bash-event identity fields the session's
        // `runUserBash(command, excludeFromContext)` does not accept.
        Box::pin(async move {
            session
                .run_user_bash(&command, exclude_from_context)
                .await
                .map(|_result| ())
        })
    }

    fn session_execute_bash(&self, command: &str) -> BoxFuture<Result<Value, String>> {
        let session = self.session();
        let command = command.to_string();
        Box::pin(async move {
            let result = session.execute_bash(&command, None, None).await?;
            Ok(bash_result_value(&result))
        })
    }

    fn session_abort_bash(&self) {
        self.session().abort_bash();
    }

    fn session_set_model(&self, model: Model) -> BoxFuture<Result<(), String>> {
        let session = self.session();
        Box::pin(async move {
            session
                .set_model(model, ModelSelectOptions::default())
                .await
        })
    }

    fn session_model_registry_available_models(&self) -> BoxFuture<Vec<AgentConnectionModel>> {
        let registry = self.session().model_registry();
        // Same call the production adapter makes (`in_process_adapter.rs:952`), with
        // the `pub(crate)` helper reproduced locally.
        Box::pin(with_model_registry(registry))
    }

    fn session_model_registry_provider_auth_source(&self, provider: &str) -> String {
        let registry = self.session().model_registry();
        let registry = registry.lock().unwrap();
        registry
            .get_provider_auth_status(provider)
            .source
            .unwrap_or_default()
    }

    fn session_model_registry_find(&self, provider: &str, model_id: &str) -> Option<Model> {
        let registry = self.session().model_registry();
        let registry = registry.lock().unwrap();
        registry.find(provider, model_id)
    }

    fn session_cycle_model(
        &self,
        direction: &str,
    ) -> BoxFuture<Result<Option<AgentConnectionModelCycleResult>, String>> {
        let session = self.session();
        let direction = if direction == "backward" { -1 } else { 1 };
        Box::pin(async move {
            let result = session
                .cycle_model(Some(direction), ModelSelectOptions::default())
                .await?;
            Ok(Some(AgentConnectionModelCycleResult {
                model: result.model,
                thinking_level: result.thinking_level,
                service_tier: result.service_tier,
                is_scoped: result.is_scoped,
            }))
        })
    }

    fn session_set_thinking_level(&self, level: ThinkingLevel) {
        self.session().set_thinking_level(level);
    }

    fn session_set_service_tier(&self, service_tier: ServiceTier) {
        self.session().set_service_tier(service_tier);
    }

    fn session_cycle_thinking_level(&self) -> Option<ThinkingLevel> {
        self.session().cycle_thinking_level()
    }

    fn session_set_steering_mode(&self, mode: &str) {
        self.session().set_steering_mode(mode);
    }

    fn session_set_follow_up_mode(&self, mode: &str) {
        self.session().set_follow_up_mode(mode);
    }

    fn session_set_auto_compaction_enabled(&self, enabled: bool) {
        self.session().set_auto_compaction_enabled(enabled);
    }

    fn session_set_auto_retry_enabled(&self, enabled: bool) {
        self.session().set_auto_retry_enabled(enabled);
    }

    fn session_abort_compaction(&self) {
        self.session().abort_compaction();
    }

    fn session_abort_branch_summary(&self) {
        self.session().abort_branch_summary();
    }

    fn session_abort_retry(&self) {
        self.session().abort_retry();
    }

    fn session_reload(&self) -> BoxFuture<Result<(), String>> {
        let session = self.session();
        Box::pin(async move { session.reload_with_options(None).await })
    }

    fn session_set_session_name(&self, name: &str) {
        // The connection layer rejects an empty name before this call.
        if let Err(error) = self.session().set_session_name(name) {
            eprintln!("Warning: Could not set session name: {error}");
        }
    }

    fn session_get_rlm_max_depth_status(&self) -> Value {
        let status: RlmMaxDepthStatus = self.session().get_rlm_max_depth_status();
        serde_json::to_value(status).unwrap_or(Value::Null)
    }

    fn session_set_rlm_max_depth(
        &self,
        max_depth: f64,
        options: Option<Value>,
    ) -> BoxFuture<Result<Value, String>> {
        let session = self.session();
        let global = options
            .as_ref()
            .and_then(|options| options.get("global"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Box::pin(async move {
            let result = session.set_rlm_max_depth(max_depth as i64, global).await?;
            Ok(serde_json::to_value(result).unwrap_or(Value::Null))
        })
    }

    fn session_bind_extensions(
        &self,
        bindings: ExtensionBindings,
    ) -> BoxFuture<Result<(), String>> {
        let session = self.session();
        Box::pin(async move { session.bind_extensions(&bindings).await })
    }

    fn runtime_new_session(
        &self,
        options: Option<AgentConnectionNewSessionOptions>,
    ) -> BoxFuture<Result<bool, String>> {
        let runtime = Arc::clone(&self.runtime);
        let parent_session = options.and_then(|options| options.parent_session);
        Box::pin(async move {
            let result = runtime
                .new_session(Some(NewSessionOptionsInput {
                    parent_session,
                    setup: None,
                    with_session: None,
                }))
                .await?;
            Ok(result.cancelled)
        })
    }

    fn runtime_switch_session(
        &self,
        session_path: &str,
        options: Option<AgentConnectionSwitchSessionOptions>,
    ) -> BoxFuture<Result<bool, String>> {
        let runtime = Arc::clone(&self.runtime);
        let session_path = session_path.to_string();
        let cwd_override = options.and_then(|options| options.cwd_override);
        Box::pin(async move {
            let result = runtime
                .switch_session(
                    &session_path,
                    Some(SwitchSessionOptions {
                        cwd_override,
                        with_session: None,
                    }),
                )
                .await?;
            Ok(result.cancelled)
        })
    }

    fn runtime_fork(
        &self,
        entry_id: &str,
        options: Option<AgentConnectionForkOptions>,
    ) -> BoxFuture<Result<Value, String>> {
        let runtime = Arc::clone(&self.runtime);
        let entry_id = entry_id.to_string();
        let position = options.and_then(|options| options.position);
        Box::pin(async move {
            let result = runtime
                .fork(
                    &entry_id,
                    Some(ForkOptionsInput {
                        position,
                        with_session: None,
                    }),
                )
                .await?;
            let mut object = Map::new();
            object.insert("cancelled".to_string(), Value::Bool(result.cancelled));
            if let Some(selected_text) = result.selected_text {
                object.insert("selectedText".to_string(), Value::String(selected_text));
            }
            Ok(Value::Object(object))
        })
    }

    fn runtime_import_from_jsonl(
        &self,
        input_path: &str,
        cwd_override: Option<&str>,
    ) -> BoxFuture<Result<bool, String>> {
        let runtime = Arc::clone(&self.runtime);
        let input_path = input_path.to_string();
        let cwd_override = cwd_override.map(str::to_string);
        Box::pin(async move {
            let result = runtime
                .import_from_jsonl(&input_path, cwd_override.as_deref())
                .await?;
            Ok(result.cancelled)
        })
    }

    fn runtime_set_before_session_invalidate(&self, listener: Option<Arc<dyn Fn() + Send + Sync>>) {
        self.runtime.set_before_session_invalidate(listener);
    }

    fn runtime_set_rebind_session(
        &self,
        listener: Option<Arc<dyn Fn() -> BoxFuture<()> + Send + Sync>>,
    ) {
        // The runtime's rebind callback receives the replaced session; the
        // connection-layer listener takes no argument.
        let listener = listener.map(|listener| {
            Arc::new(move |_session: Arc<AgentSession>| listener())
                as Arc<dyn Fn(Arc<AgentSession>) -> BoxFuture<()> + Send + Sync>
        });
        self.runtime.set_rebind_session(listener);
    }

    fn runtime_dispose(&self) -> BoxFuture<()> {
        let runtime = Arc::clone(&self.runtime);
        let tasks = std::mem::take(&mut *self.event_tasks.lock().unwrap());
        Box::pin(async move {
            if let Err(error) = runtime.dispose(None).await {
                eprintln!("Could not dispose session runtime: {error}");
            }
            for task in tasks {
                let _ = task.await;
            }
        })
    }
}

// ===========================================================================
// Fixture
// ===========================================================================

/// One soak fixture: a real runtime, the real in-process connection, and the
/// faux provider that connection streams from.
struct Soak {
    _root: tempfile::TempDir,
    runtime: Arc<AgentSessionRuntime>,
    connection: Arc<dyn AgentConnection>,
    host: Arc<SoakHost>,
    provider: FauxProviderRegistration,
    model: Model,
    recorder: Arc<EventRecorder>,
    _unsubscribe: Box<dyn Fn() + Send + Sync>,
}

/// `AgentConnectionEvent` recorder for the streaming and ordering assertions.
#[derive(Default)]
struct EventRecorder {
    events: Mutex<Vec<(String, Option<AgentMessage>)>>,
}

impl EventRecorder {
    fn record(&self, event: &wire::AgentConnectionEvent) {
        let wire::AgentConnectionEvent::SessionEvent { event } = event else {
            return;
        };
        let (name, message) = match event {
            wire::AgentConnectionSessionEvent::Agent(AgentEvent::MessageUpdate {
                message, ..
            }) => ("message_update".to_string(), Some(message.clone())),
            wire::AgentConnectionSessionEvent::Agent(AgentEvent::MessageEnd { message }) => {
                ("message_end".to_string(), Some(message.clone()))
            }
            wire::AgentConnectionSessionEvent::Agent(AgentEvent::TurnEnd { message, .. }) => {
                ("turn_end".to_string(), Some(message.clone()))
            }
            wire::AgentConnectionSessionEvent::Agent(AgentEvent::TurnStart) => {
                ("turn_start".to_string(), None)
            }
            wire::AgentConnectionSessionEvent::Agent(AgentEvent::AgentStart) => {
                ("agent_start".to_string(), None)
            }
            wire::AgentConnectionSessionEvent::Agent(AgentEvent::AgentEnd { .. }) => {
                ("agent_end".to_string(), None)
            }
            other => (other.type_name().to_string(), None),
        };
        self.events.lock().unwrap().push((name, message));
    }

    /// Every streamed `message_update` partial, in delivery order.
    fn streaming_texts(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|(name, _)| name == "message_update")
            .filter_map(|(_, message)| message.as_ref())
            .map(assistant_text)
            .filter(|text| !text.is_empty())
            .collect()
    }

    fn texts_for(&self, name: &str) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|(event, _)| event == name)
            .filter_map(|(_, message)| message.as_ref())
            .map(assistant_text)
            .collect()
    }

    fn count(&self, name: &str) -> usize {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|(event, _)| event == name)
            .count()
    }

    fn clear(&self) {
        self.events.lock().unwrap().clear();
    }
}

/// The assistant text of any `AgentMessage` (`""` for another role).
fn assistant_text(message: &AgentMessage) -> String {
    let AgentMessage::Message(Message::Assistant(assistant)) = message else {
        return String::new();
    };
    assistant
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// The deterministic reply for turn `index`.
fn reply_text(index: usize) -> String {
    format!("SOAK_REPLY_{index:04}_END")
}

fn next_reply(index: usize) -> AssistantMessage {
    faux_assistant_message(FauxAssistantContent::Text(reply_text(index)), None)
}

impl Soak {
    /// Builds the fixture under an isolated temp root. No HOME, no network.
    /// A fixture whose faux provider streams at `rate` tokens/second. A finite
    /// rate is what makes the in-flight window observable (`faux.rs:485-492`).
    async fn with_rate(label: &str, rate: f64) -> Self {
        Self::build(label, rate).await
    }

    async fn new(label: &str) -> Self {
        // 0.0 keeps the fast path: a whole reply lands in one scheduler slice.
        Self::build(label, 0.0).await
    }

    async fn build(label: &str, rate: f64) -> Self {
        let scratch = std::env::temp_dir().canonicalize().unwrap();
        let root = tempfile::Builder::new()
            .prefix(&format!("soak-{label}-"))
            .tempdir_in(scratch)
            .unwrap();
        let cwd = root.path().join("workspace");
        let agent_dir = root.path().join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        let cwd_string = cwd.to_string_lossy().to_string();
        let agent_dir_string = agent_dir.to_string_lossy().to_string();

        // `main_entry.rs:2485` initializes the theme before any interactive render.
        // The real components in this file read `theme()` (:1170), which panics
        // with "Theme not initialized. Call initTheme() first." without it.
        pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);

        let provider = register_faux_provider(Some(RegisterFauxProviderOptions {
            provider: Some(format!("soak-{}", uuid::Uuid::new_v4())),
            // `schedule_chunk` (crates/pi-ai/src/providers/faux.rs:485-492) only
            // yields when the rate is 0, so a 0 rate finishes a long reply before a
            // caller can poll a streaming snapshot. A real rate makes the in-flight
            // window observable, which is what the streaming and queue soaks need.
            tokens_per_second: Some(rate),
            ..Default::default()
        }));
        let model = provider.get_model();

        let settings = Arc::new(Mutex::new(SettingsManager::in_memory(
            serde_json::json!({
                "autoRefine": {"enabled": false},
                "retry": {"enabled": false},
                "compaction": {"enabled": false},
                "telemetryEnabled": false,
                "agentTracesEnabled": false,
                "quietStartup": true,
            })
            .as_object()
            .unwrap()
            .clone(),
        )));

        let session_manager = Arc::new(Mutex::new(
            SessionManager::in_memory(Some(&cwd_string), Some(&agent_dir_string)).unwrap(),
        ));
        // `main.ts:882` (mirrored by `main_entry.rs`) writes the CLI runtime API
        // key to BOTH the auth storage and the registry's own storage
        // (`model_registry.rs:1262`), because request auth resolves through the
        // registry (`agent_session.rs:14526`). Both writes are needed here.
        let auth_storage = Arc::new(tokio::sync::Mutex::new(AuthStorage::in_memory(
            AuthStorageData::new(),
            None,
        )));
        let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory(
            AuthStorage::in_memory(AuthStorageData::new(), None),
        )));
        {
            auth_storage
                .lock()
                .await
                .set_runtime_api_key(&model.provider, SOAK_API_KEY);
            model_registry
                .lock()
                .expect("model registry poisoned")
                .set_runtime_api_key(&model.provider, SOAK_API_KEY);
        }

        let session = create_session(
            &cwd_string,
            &agent_dir_string,
            &model,
            Arc::clone(&settings),
            Arc::clone(&session_manager),
            Arc::clone(&auth_storage),
            Arc::clone(&model_registry),
        )
        .await;
        let services = create_services(
            &cwd_string,
            &agent_dir_string,
            settings,
            Arc::clone(&auth_storage),
            Arc::clone(&model_registry),
        )
        .await;

        // `createAgentSessionRuntime` (core/agent_session_runtime.rs:1686) builds
        // the runtime around the session the factory returned. The fixture holds
        // the session directly, and the factory refuses replacements with an
        // explicit message so no soak path can pass on a fabricated session.
        let refuse: CreateAgentSessionRuntimeFactory =
            Arc::new(move |_input: CreateAgentSessionRuntimeInput| {
                Box::pin(async move {
                    Err("soak fixture does not create replacement runtimes".to_string())
                })
            });
        let runtime = AgentSessionRuntime::new(
            session,
            services,
            refuse,
            Vec::new(),
            None,
            None,
            pi_coding_agent::core::agent_session_runtime::AgentSessionRuntimeMetadata::top_level(),
            None,
        );

        let host = Arc::new(SoakHost::new(Arc::clone(&runtime)));
        let connection: Arc<dyn AgentConnection> = Arc::new(InProcessAgentConnection::new(
            Arc::clone(&host) as Arc<dyn InProcessRuntimeHost>,
        ));
        let recorder = Arc::new(EventRecorder::default());
        let captured = Arc::clone(&recorder);
        let unsubscribe = connection.subscribe(Arc::new(move |event| {
            captured.record(&event);
            Box::pin(async {})
        }));

        Self {
            _root: root,
            runtime,
            connection,
            host,
            provider,
            model,
            recorder,
            _unsubscribe: unsubscribe,
        }
    }

    fn session(&self) -> Arc<AgentSession> {
        self.runtime.session()
    }

    /// Messages as the interactive host sees them
    /// (`AgentConnection::get_messages` -> `session.state.messages`).
    async fn messages(&self) -> Vec<AgentMessage> {
        self.connection.get_messages().await.expect("messages")
    }

    /// Submit a prompt the way the interactive host does: `submit`
    /// (`native_host.rs:2141-2152`) spawns `connection.prompt(...)` and returns
    /// immediately, so the run is observable while it streams. Awaiting `prompt`
    /// directly would block until the turn finished and hide the in-flight window.
    fn prompt_spawned(&self, text: &str) {
        let connection = Arc::clone(&self.connection);
        let text = text.to_string();
        tokio::spawn(async move {
            let _ = connection.prompt(&text, None).await;
        });
    }

    /// Queue a mid-run submission exactly the way the host does: a prompt carrying
    /// `streamingBehavior` (`native_host.rs:2239-2245`). Awaiting this is correct -
    /// the call queues and returns; it does not wait for the run.
    async fn queue_via_prompt(&self, text: &str, streaming_behavior: &str) {
        self.connection
            .prompt(
                text,
                Some(wire::AgentConnectionPromptOptions {
                    streaming_behavior: Some(streaming_behavior.to_string()),
                    ..Default::default()
                }),
            )
            .await
            .unwrap_or_else(|error| {
                panic!("queuing {text:?} as {streaming_behavior} failed: {error}")
            });
    }

    /// Poll the connection snapshot until it exposes an in-flight message, i.e.
    /// until the spawned run is really streaming. Returns that snapshot.
    async fn wait_for_streaming_snapshot(
        &self,
        timeout: Duration,
    ) -> wire::AgentConnectionSnapshot {
        // See the body for why a merely-present message is not enough.
        let deadline = Instant::now() + timeout;
        loop {
            let snapshot = self
                .connection
                .get_initial_snapshot()
                .await
                .expect("snapshot");
            // The agent sets `streaming_message` on `message_start`
            // (pi-agent-core/src/agent.rs:775) BEFORE the first delta arrives, so a
            // merely-present message can still be empty. Wait for real content.
            let partial = snapshot
                .streaming_message
                .as_ref()
                .map(assistant_text)
                .unwrap_or_default();
            if !partial.is_empty() {
                return snapshot;
            }
            assert!(
                Instant::now() < deadline,
                "the run never exposed a non-empty streaming message within {timeout:?}; \
                 the reply finished before it could be observed, so the streaming window \
                 was shorter than one poll"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// One prompt/response cycle through the real connection.
    /// One prompt/response cycle. BOUNDED: a session command that parks the input
    /// pump would otherwise hang the whole binary, and a hanging test is worse than
    /// a failing one because it stalls every other worker on the shared suite.
    async fn turn(&self, prompt: &str) {
        let accepted = tokio::time::timeout(
            Duration::from_secs(60),
            self.connection.prompt(prompt, None),
        )
        .await
        .unwrap_or_else(|_| panic!("prompt {prompt:?} never returned"));
        accepted.unwrap_or_else(|error| panic!("prompt {prompt:?} was rejected: {error}"));
        self.wait_for_idle().await;
    }

    /// `promptAndWait` over the real connection, bounded the same way.
    async fn turn_and_wait(&self, prompt: &str) {
        let accepted = tokio::time::timeout(
            Duration::from_secs(60),
            self.connection.prompt_and_wait(prompt, None),
        )
        .await
        .unwrap_or_else(|_| panic!("promptAndWait {prompt:?} never returned"));
        accepted.unwrap_or_else(|error| panic!("prompt {prompt:?} was rejected: {error}"));
        self.wait_for_idle().await;
    }

    async fn wait_for_idle(&self) {
        let session = self.session();
        tokio::time::timeout(Duration::from_secs(60), session.wait_for_headless_idle())
            .await
            .unwrap_or_else(|_| panic!("session never reached idle"))
            .expect("headless idle");
    }

    /// Poll until the queue reports empty. After a preserved abort the queued
    /// actions are still pending, so `wait_for_idle` (which loops until
    /// `unfinished_action_count() == 0`, `agent_session.rs:10959-10976`) cannot be
    /// used; the queue is drained by resuming the pump.
    async fn wait_for_queue_to_drain(&self) -> bool {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let (steering, follow) = self.queue_counts();
            if steering + follow == 0 {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn queue_counts(&self) -> (usize, usize) {
        let queue = self.host.session_queue();
        (queue.steering.len(), queue.follow_up.len())
    }
}

async fn create_services(
    cwd: &str,
    agent_dir: &str,
    settings: Arc<Mutex<SettingsManager>>,
    auth_storage: Arc<tokio::sync::Mutex<AuthStorage>>,
    model_registry: Arc<Mutex<ModelRegistry>>,
) -> Arc<pi_coding_agent::core::agent_session_services::AgentSessionServices> {
    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: cwd.to_string(),
        agent_dir: Some(agent_dir.to_string()),
        auth_storage: Some(auth_storage),
        settings_manager: Some(settings),
        model_registry: Some(model_registry),
        extension_flag_values: None,
        no_builtin_herdr_reporter: Some(true),
        telemetry_disabled: Some(true),
        resource_loader_options: Some(no_resource_options(cwd, agent_dir)),
    })
    .await
    .expect("services");
    Arc::new(services)
}

fn no_resource_options(cwd: &str, agent_dir: &str) -> DefaultResourceLoaderOptions {
    DefaultResourceLoaderOptions {
        cwd: cwd.to_string(),
        agent_dir: agent_dir.to_string(),
        no_extensions: true,
        no_skills: true,
        no_prompt_templates: true,
        no_themes: true,
        no_context_files: true,
        bundled_skills_dir: Some(None),
        ..Default::default()
    }
}

/// The synthetic API key applied to both the auth storage and the model registry.
const SOAK_API_KEY: &str = "synthetic-soak-key";

/// `sdk::with_model_registry` is `pub(crate)`, so this reproduces its exact body
/// (spawn_blocking + registry lock + `block_on`) locally. The production adapter
/// calls it from `session_model_registry_available_models`
/// (`in_process_adapter.rs:952`) and `registry_available_models` (:90).
async fn with_model_registry(
    registry: Arc<Mutex<ModelRegistry>>,
) -> Vec<pi_coding_agent::modes::agent_connection::types::AgentConnectionModel> {
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let mut registry = registry.lock().expect("model registry poisoned");
        runtime.block_on(registry.refresh_available_models())
    })
    .await
    .expect("model registry worker failed")
}

/// One real `AgentSession` on the faux model.
async fn create_session(
    cwd: &str,
    agent_dir: &str,
    model: &Model,
    settings: Arc<Mutex<SettingsManager>>,
    session_manager: Arc<Mutex<SessionManager>>,
    auth_storage: Arc<tokio::sync::Mutex<AuthStorage>>,
    model_registry: Arc<Mutex<ModelRegistry>>,
) -> Arc<AgentSession> {
    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: cwd.to_string(),
        agent_dir: Some(agent_dir.to_string()),
        auth_storage: Some(auth_storage),
        settings_manager: Some(settings),
        model_registry: Some(model_registry),
        extension_flag_values: None,
        no_builtin_herdr_reporter: Some(true),
        telemetry_disabled: Some(true),
        resource_loader_options: Some(no_resource_options(cwd, agent_dir)),
    })
    .await
    .expect("services");
    let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services: Arc::new(services),
        session_manager,
        session_start_event: None,
        creation: AgentSessionCreationOptions {
            model: Some(model.clone()),
            no_tools: Some("all".to_string()),
            prewarm_ipython_kernel: Some(false),
            telemetry_disabled: Some(true),
            ..Default::default()
        },
    })
    .await
    .expect("agent session");
    created.session
}

// ===========================================================================
// Measurement helpers
// ===========================================================================

/// Durable soak metrics, so the lead can quote real numbers even though libtest
/// captures a passing test's stdout.
fn record_metrics(label: &str, metrics: serde_json::Value) {
    println!("SOAK_METRICS {label} {metrics}");
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.port-env/soak-metrics.json");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut all: serde_json::Map<String, serde_json::Value> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default();
    all.insert(label.to_string(), metrics);
    let _ = std::fs::write(
        &path,
        serde_json::to_string_pretty(&all).unwrap_or_default(),
    );
}

type Row = Rc<RefCell<dyn pi_tui::tui::Component>>;

/// The rows the real `Transcript::message` builds for one message
/// (`native_host.rs:177-247`): the same public owners with the same options.
fn message_rows(
    message: &AgentMessage,
    assistant: &RefCell<Option<Rc<RefCell<AssistantMessageComponent>>>>,
    markdown_theme: &MarkdownTheme,
) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    match message {
        AgentMessage::Message(Message::User(user)) => {
            let text = match &user.content {
                UserContent::Text(text) => text.clone(),
                UserContent::Blocks(blocks) => blocks
                    .iter()
                    .map(|block| match block {
                        pi_ai::types::ImageOrTextContent::Text(text) => text.text.clone(),
                        pi_ai::types::ImageOrTextContent::Image(_) => "[image]".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            rows.push(Rc::new(RefCell::new(UserMessageComponent::new(
                &text,
                markdown_theme.clone(),
                &|_| false,
            ))) as Row);
        }
        AgentMessage::Message(Message::Assistant(message)) => {
            // The borrow must be released before the `None` arm writes, so read the
            // existing component into a local first.
            let existing = assistant.borrow().clone();
            let component = match existing {
                // `Transcript::message` reuses the pending assistant component and
                // calls `update_content` with `streaming: false` (native_host.rs:200-221).
                Some(component) => component,
                None => {
                    let component = Rc::new(RefCell::new(AssistantMessageComponent::new(
                        None,
                        false,
                        markdown_theme.clone(),
                        "Thinking",
                        AssistantMessageComponentOptions::default(),
                    )));
                    *assistant.borrow_mut() = Some(Rc::clone(&component));
                    component
                }
            };
            component
                .borrow_mut()
                .update_content(message.clone(), false);
            rows.push(component as Row);
        }
        AgentMessage::Message(Message::ToolResult(result)) => {
            let text = result
                .content
                .iter()
                .map(|block| match block {
                    pi_ai::types::ImageOrTextContent::Text(text) => text.text.clone(),
                    pi_ai::types::ImageOrTextContent::Image(_) => "[image]".to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n");
            rows.push(Rc::new(RefCell::new(pi_tui::components::text::Text::new(
                text.into(),
                0,
                0,
                None,
            ))) as Row);
        }
        AgentMessage::Custom(_) => {}
    }
    rows
}

/// Renders one message list through the real components and strips ANSI, so an
/// assertion tests the rendered text a terminal would show.

/// Whitespace-collapsed containment: wrapping splits a marker across lines, so a
/// width-agnostic content check must ignore whitespace.
fn rendered_contains(rendered: &str, needle: &str) -> bool {
    let flat = |value: &str| -> String {
        value
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect()
    };
    flat(rendered).contains(&flat(needle))
}

/// Whitespace-insensitive occurrence count, so a wrapped reply still counts once.
fn count_occurrences(rendered: &str, needle: &str) -> usize {
    let flat = |value: &str| -> String {
        value
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect()
    };
    let haystack = flat(rendered);
    let needle = flat(needle);
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(&needle).count()
}

/// The widest whitespace-separated token, i.e. the narrowest a line can legally
/// be when the renderer cannot break inside a token.
fn widest_token_width(text: &str) -> usize {
    text.split_whitespace()
        .map(pi_tui::utils::visible_width)
        .max()
        .unwrap_or(0)
}

/// The plain-transcript text a message set renders at `width`.
fn render_messages(messages: &[AgentMessage], width: f64) -> String {
    let markdown_theme = get_markdown_theme();
    let assistant: RefCell<Option<Rc<RefCell<AssistantMessageComponent>>>> = RefCell::new(None);
    let mut lines: Vec<String> = Vec::new();
    for message in messages {
        for row in message_rows(message, &assistant, &markdown_theme) {
            for line in row.borrow_mut().render(width) {
                lines.push(pi_tui::utils::strip_ansi(&line));
            }
        }
    }
    lines.join("\n")
}

/// A tool row through the real `ToolExecutionComponent`, with a real long result.
fn render_tool_result(tool_name: &str, tool_call_id: &str, output: &str, width: f64) -> String {
    let mut component = ToolExecutionComponent::new(
        tool_name,
        tool_call_id,
        serde_json::json!({"command": "soak"}),
        ToolExecutionOptions::default(),
        None,
        ".",
    );
    component.update_result(
        ToolExecutionResult {
            content: vec![RenderContentBlock::from_text(output)],
            is_error: false,
            details: None,
        },
        false,
    );
    component
        .render_lines(width)
        .iter()
        .map(|line| pi_tui::utils::strip_ansi(line))
        .collect::<Vec<_>>()
        .join("\n")
}

// ===========================================================================
// Shared helpers
// ===========================================================================

/// `blocked_on:` messages the production adapter returns for `session_compact`
/// (`in_process_adapter.rs:498`) and `session_refine` (:513). Asserting the exact
/// text keeps the soak honest: it must report the same blocked state, never a
/// fabricated success.
const BLOCKED_COMPACT: &str = "blocked_on: compact result not exposed on the public path";
const BLOCKED_REFINE: &str = "blocked_on: refine result not exposed on the public path";

const INITIAL_HISTORY_WINDOW_MESSAGES: usize =
    pi_coding_agent::modes::daemon::daemon_mode::INITIAL_HISTORY_WINDOW_MESSAGES;

/// `AgentConnectionHistoryWindow` in the interactive module's own carrier type
/// (`interactive_mode_services.rs:516`), which `merge_older_agent_connection_history`
/// consumes.
fn local_window(
    window: &pi_coding_agent::modes::daemon::daemon_mode::DaemonHistoryRange,
) -> pi_coding_agent::modes::interactive::interactive_mode_services::AgentConnectionHistoryWindow {
    pi_coding_agent::modes::interactive::interactive_mode_services::AgentConnectionHistoryWindow {
        version: window.version as f64,
        generation: window.generation.clone(),
        representation: window.representation.clone(),
        tip_entry_id: window.tip_entry_id.clone(),
        total_message_count: window.total_message_count as f64,
        start_index: window.start_index as f64,
        entry_ids: window.entry_ids.clone(),
        has_older: window.has_older,
        order: window.order.clone(),
    }
}

/// `AgentConnectionHistoryRange` in the interactive module's own carrier type.
fn local_range(
    range: &pi_coding_agent::modes::daemon::daemon_mode::DaemonHistoryRange,
) -> pi_coding_agent::modes::interactive::interactive_mode_services::AgentConnectionHistoryRange {
    pi_coding_agent::modes::interactive::interactive_mode_services::AgentConnectionHistoryRange {
        window: local_window(range),
        messages: range.messages.clone(),
    }
}

// ===========================================================================
// 1. MANY TURNS
// ===========================================================================

/// A few hundred prompt/response cycles through the real connection and the
/// real faux provider. Invariants:
/// - every turn produces exactly one `message_end` and one `agent_end`;
/// - the transcript grows by exactly 2 messages per turn (user + assistant);
/// - every reply's deterministic marker is present, in order;
/// - assistant text never repeats and never truncates;
/// - the provider consumed exactly one response per turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_turns_keep_the_transcript_consistent() {
    const TURNS: usize = 300;
    let soak = Soak::new("many-turns").await;
    // One prepared response per turn, so a lost or duplicated provider call is
    // observable in `call_count` and in `get_pending_response_count`.
    soak.provider.set_responses(
        (0..TURNS)
            .map(|index| FauxResponseStep::Message(next_reply(index)))
            .collect(),
    );

    let started = Instant::now();
    let mut after_first = 0usize;
    for index in 0..TURNS {
        soak.turn(&format!("soak prompt {index}")).await;
        if index == 0 {
            after_first = soak.messages().await.len();
            // The session prepends one hidden harness-digest custom message
            // (`core/messages.rs:130`), so the first turn yields three rows.
            assert_eq!(
                after_first, 3,
                "the first turn is the harness digest plus one user and one assistant message"
            );
        }
    }
    let elapsed = started.elapsed();

    let messages = soak.messages().await;
    // Exactly one hidden digest, then one user and one assistant message per turn.
    assert_eq!(
        messages.len(),
        1 + 2 * TURNS,
        "every turn must append exactly one user and one assistant message"
    );
    // Ordering and content: row 0 is the digest, so turn i's prompt sits at
    // 2*i+1 and its reply at 2*i+2.
    for index in 0..TURNS {
        let expected_prompt = format!("soak prompt {index}");
        let rendered_user = assistant_or_user_text(&messages[index * 2 + 1]);
        assert_eq!(
            rendered_user, expected_prompt,
            "turn {index} user message is out of order or corrupted"
        );
        let rendered_reply = assistant_text(&messages[index * 2 + 2]);
        assert_eq!(
            rendered_reply,
            reply_text(index),
            "turn {index} assistant reply is out of order or corrupted"
        );
    }
    assert_eq!(
        soak.provider.call_count(),
        TURNS as u64,
        "the faux provider must be called exactly once per turn"
    );
    assert_eq!(
        soak.provider.get_pending_response_count(),
        0,
        "no prepared response may be left unconsumed"
    );
    assert_eq!(
        soak.recorder.count("agent_end"),
        TURNS,
        "each turn must end with exactly one agent_end"
    );
    assert_eq!(
        soak.recorder.count("turn_end"),
        TURNS,
        "each turn must produce exactly one turn_end"
    );

    let peak_messages = messages.len();
    drop(messages);
    let peak_rss_bytes = resident_bytes();
    record_metrics(
        "many_turns",
        serde_json::json!({
            "turns": TURNS,
            "peak_messages": peak_messages,
            "provider_calls": soak.provider.call_count(),
            "elapsed_ms": elapsed.as_millis() as u64,
            "ms_per_turn": (elapsed.as_millis() as f64) / (TURNS as f64),
            "peak_rss_bytes": peak_rss_bytes,
        }),
    );
    assert!(
        elapsed < Duration::from_secs(150),
        "300 turns took {elapsed:?}; the soak must stay bounded"
    );
}

/// The user text of a message (`""` for another role).
fn assistant_or_user_text(message: &AgentMessage) -> String {
    let AgentMessage::Message(Message::User(user)) = message else {
        return String::new();
    };
    match &user.content {
        UserContent::Text(text) => text.clone(),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                pi_ai::types::ImageOrTextContent::Text(text) => text.text.clone(),
                pi_ai::types::ImageOrTextContent::Image(_) => "[image]".to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// The process resident set, in bytes, when the platform exposes it.
fn resident_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kilobytes: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kilobytes * 1024)
}

// ===========================================================================
// 2. STREAMING UNDER LOAD
// ===========================================================================

/// The defect-1 area: attach while a response is streaming, repeatedly.
///
/// Each cycle takes a REAL snapshot through `getInitialSnapshot` while the
/// assistant message is in flight, then feeds it to the same reset-then-streaming
/// ordering the host uses (`native_host.rs:758-771`, `:898-908`) and asserts the
/// in-flight content survives in BOTH branches: with recent-first history
/// metadata and without it. The partial content must also still be present in
/// the snapshot the client would attach with.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn attach_during_streaming_never_loses_the_in_flight_content() {
    const CYCLES: usize = 25;
    // A finite rate keeps each reply in flight long enough to attach mid-stream
    // (`faux.rs:485-492`); at rate 0 the whole reply lands in one scheduler slice.
    let soak = Soak::with_rate("streaming-attach", 20_000.0).await;

    let mut observed_partials = 0usize;
    let mut attach_snapshots = 0usize;
    let mut with_history = 0usize;
    let mut without_history = 0usize;
    let started = Instant::now();

    for cycle in 0..CYCLES {
        // A long reply guarantees several real deltas before the stream ends.
        let body = format!("STREAM_BODY_{cycle:03}_").repeat(60);
        soak.provider
            .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
                FauxAssistantContent::Text(body.clone()),
                None,
            ))]);
        soak.recorder.clear();

        soak.prompt_spawned(&format!("stream prompt {cycle}"));

        // Attach while streaming: take the snapshot that carries an in-flight
        // message, exactly like a second client attaching mid-response.
        let snapshot = soak
            .wait_for_streaming_snapshot(Duration::from_secs(30))
            .await;
        attach_snapshots += 1;

        let in_flight = assistant_text(snapshot.streaming_message.as_ref().unwrap());
        observed_partials += 1;
        assert!(
            !in_flight.is_empty(),
            "cycle {cycle}: the attached in-flight message must carry real content"
        );
        assert!(
            body.starts_with(&in_flight),
            "cycle {cycle}: the attached in-flight text must be a real prefix of the reply"
        );

        // The host's own history window for this snapshot: recent-first metadata
        // over the messages the snapshot carries.
        let history = recent_first_window(&snapshot.messages, Some("generation-1"), "model", "tip");

        // Branch A: recent-first metadata present. `HistoryRuntime::reset`
        // replaces the transcript, then `apply_history_snapshot` re-adds the
        // streaming row.
        let branch_a = render_attach(
            snapshot.history.clone().or_else(|| history.clone()),
            snapshot.messages.clone(),
            snapshot.streaming_message.clone(),
        );
        assert!(
            rendered_contains(&branch_a, &in_flight),
            "cycle {cycle}: the paged-history reset dropped the in-flight content"
        );
        with_history += 1;

        // Branch B: no recent-first metadata, the legacy full-snapshot attach.
        let branch_b = render_attach(
            None,
            snapshot.messages.clone(),
            snapshot.streaming_message.clone(),
        );
        assert!(
            rendered_contains(&branch_b, &in_flight),
            "cycle {cycle}: the plain-transcript reset dropped the in-flight content"
        );
        without_history += 1;

        soak.wait_for_idle().await;

        // After the turn the finalized reply must be complete and appear once.
        let messages = soak.messages().await;
        let final_text = assistant_text(messages.last().expect("final message"));
        assert_eq!(
            final_text, body,
            "cycle {cycle}: the finalized streamed reply must not lose any delta"
        );
        let rendered = render_messages(&messages, 80.0);
        assert_eq!(
            count_occurrences(&rendered, &body),
            1,
            "cycle {cycle}: the reply must be rendered exactly once"
        );
    }

    let elapsed = started.elapsed();
    record_metrics(
        "streaming_attach",
        serde_json::json!({
            "cycles": CYCLES,
            "snapshots_taken_while_streaming": attach_snapshots,
            "partials_observed": observed_partials,
            "asserted_with_history": with_history,
            "asserted_without_history": without_history,
            "elapsed_ms": elapsed.as_millis() as u64,
        }),
    );
    assert_eq!(attach_snapshots, CYCLES);
    assert_eq!(observed_partials, CYCLES);
}

/// The exact attach ordering the host uses: reset the transcript, then re-add
/// the streaming row (`native_host.rs:758-771`).
fn render_attach(
    history: Option<wire::AgentConnectionHistoryWindow>,
    messages: Vec<AgentMessage>,
    streaming: Option<AgentMessage>,
) -> String {
    // `HistoryRuntime::reset` behaviour (`native_host_history.rs:51-97`): a valid
    // window renders the paged history, otherwise the plain transcript. Both
    // paths REPLACE the transcript.
    let mut rendered = match history {
        Some(window) if valid_history(&window, messages.len()) => {
            let notice = format!(
                "Showing {} of {} messages.",
                messages.len(),
                window.total_message_count as usize
            );
            let mut out = render_messages(&messages, 80.0);
            out.push_str(&notice);
            out
        }
        _ => render_messages(&messages, 80.0),
    };
    // `apply_history_snapshot` restores the in-flight row AFTER the reset.
    if let Some(streaming) = streaming {
        rendered.push_str(&render_messages(&[streaming], 80.0));
    }
    rendered
}

/// `validate` from `native_host_history.rs:182-193`.
fn valid_history(history: &wire::AgentConnectionHistoryWindow, messages: usize) -> bool {
    history.version == 1.0
        && !history.representation.is_empty()
        && history.order == "chronological"
        && history.entry_ids.len() == messages
        && history.total_message_count == messages as f64
        && history.start_index >= 0.0
        && history.has_older == (history.start_index > 0.0)
}

/// A recent-first window over `messages`, newest `limit` first.
fn recent_first_window(
    messages: &[AgentMessage],
    generation: Option<&str>,
    representation: &str,
    tip_entry_id: &str,
) -> Option<wire::AgentConnectionHistoryWindow> {
    let generation = generation?;
    let total = messages.len();
    let entry_ids: Vec<String> = (0..total).map(|index| format!("entry-{index}")).collect();
    Some(wire::AgentConnectionHistoryWindow {
        version: 1.0,
        generation: generation.to_string(),
        representation: representation.to_string(),
        tip_entry_id: Some(tip_entry_id.to_string()),
        total_message_count: total as f64,
        start_index: 0.0,
        entry_ids,
        has_older: false,
        order: "chronological".to_string(),
    })
}

// ===========================================================================
// 3. LONG OUTPUT
// ===========================================================================

/// Very long assistant messages and tool outputs. Invariants:
/// - a 40 KB assistant reply renders without panicking and is fully present in
///   the finalized message (the component truncates ONLY the view, never the
///   stored message);
/// - a 200 KB tool result renders bounded lines and keeps its head/tail markers;
/// - the transcript stays coherent: every turn's marker is still rendered after
///   40 long turns.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn very_long_output_renders_without_panicking() {
    const LONG_TURNS: usize = 20;
    const BLOCK: usize = 4_000;
    let soak = Soak::new("long-output").await;

    // One very long assistant reply.
    let long_body = "L".repeat(40_000);
    soak.provider
        .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
            FauxAssistantContent::Text(long_body.clone()),
            None,
        ))]);
    soak.turn_and_wait("give me the long answer").await;
    let messages = soak.messages().await;
    let reply = assistant_text(messages.last().expect("assistant"));
    assert_eq!(
        reply.len(),
        long_body.len(),
        "a 40 KB reply must be stored complete; only the view may truncate"
    );
    assert_eq!(reply, long_body, "a 40 KB reply must not be corrupted");
    let rendered = render_messages(&messages, 80.0);
    assert!(
        rendered_contains(&rendered, "LLLLLLLLLLLL"),
        "the long reply must render real content"
    );

    // A very long tool result through the real ToolExecutionComponent.
    let tool_output = "T".repeat(40_000);
    let tool_lines = render_tool_result("bash", "call-long", &tool_output, 80.0);
    assert!(
        !tool_lines.is_empty(),
        "a 200 KB tool result must still render a bounded panel"
    );
    let line_count = tool_lines.lines().count();
    // The renderer WRAPS the single 200 KB line at the panel width, so the correct
    // expectation is lines proportional to bytes/width (about 200_000/80), not to the
    // byte count itself. The real invariant: the panel stays linear in the WRAPPED
    // size instead of exploding.
    let wrap_ceiling = tool_output.len() / 40;
    assert!(
        line_count <= wrap_ceiling,
        "a 40 KB tool result must render at most ~bytes/40 lines when wrapped, \
         got {line_count} (ceiling {wrap_ceiling})"
    );
    assert!(
        line_count >= 100,
        "a 40 KB tool result must actually render its content, got {line_count} lines"
    );

    // Many long turns: every marker stays present and in order.
    soak.provider.set_responses(
        (0..LONG_TURNS)
            .map(|index| {
                FauxResponseStep::Message(faux_assistant_message(
                    FauxAssistantContent::Text(format!("LONG_{index:03}_{}", "x".repeat(BLOCK))),
                    None,
                ))
            })
            .collect(),
    );
    let started = Instant::now();
    for index in 0..LONG_TURNS {
        soak.turn(&format!("long prompt {index}")).await;
    }
    let elapsed = started.elapsed();
    let messages = soak.messages().await;
    let rendered = render_messages(&messages, 80.0);
    for index in 0..LONG_TURNS {
        assert!(
            rendered_contains(&rendered, &format!("LONG_{index:03}_")),
            "long turn {index} vanished from the rendered transcript"
        );
    }
    // Ordering: the rendered transcript reaches each marker in ascending order.
    // The comparison ignores whitespace so wrapping cannot fake an ordering.
    let flat: String = rendered
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let mut cursor = 0usize;
    for index in 0..LONG_TURNS {
        let marker = format!("LONG_{index:03}_");
        let at = flat[cursor..]
            .find(&marker)
            .unwrap_or_else(|| panic!("marker {marker} is out of order or missing"))
            + cursor;
        cursor = at + marker.len();
    }
    record_metrics(
        "long_output",
        serde_json::json!({
            "long_turns": LONG_TURNS,
            "block_bytes": BLOCK,
            "single_reply_bytes": long_body.len(),
            "tool_output_bytes": tool_output.len(),
            "tool_render_lines": line_count,
            "final_message_count": messages.len(),
            "rendered_lines": rendered.lines().count(),
            "elapsed_ms": elapsed.as_millis() as u64,
        }),
    );
    assert!(
        elapsed < Duration::from_secs(120),
        "long-output soak took {elapsed:?}"
    );
}

// ===========================================================================
// 4. SLASH COMMANDS MID-SESSION
// ===========================================================================

/// Slash commands interleaved with normal work must not corrupt the session.
///
/// `parseSessionSlashCommand` (`core/slash-commands.ts:273-281`) splits built-ins into
/// two families: SESSION commands (`/compact`, `/refine`, `/goal`, `/autonomous`) are
/// submitted verbatim and handled by the session (`agent_session.rs:9940-9978`), while
/// every other built-in is handled locally by the host (`native_host.rs:2276-2290`).
///
/// The SESSION family is deliberately NOT driven here: it wedges the session, which is
/// pinned with measured evidence in
/// `session_slash_command_wedges_the_session_for_the_rest_of_its_life` below. This test
/// covers what IS healthy - the local family's registry, the two `blocked_on`
/// connection members, and that normal turns keep working around those calls.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slash_commands_mid_session_keep_the_session_usable() {
    let soak = Soak::new("slash-commands").await;
    soak.provider.set_responses(
        (0..40)
            .map(|index| FauxResponseStep::Message(next_reply(index)))
            .collect(),
    );

    let mut index = 0usize;
    soak.turn(&format!("soak prompt {index}")).await;
    index += 1;

    // Every LOCAL built-in registered by `canonical_builtin_slash_commands`
    // (`slash_commands.rs:205-408`) must resolve through the registry and must NOT
    // parse as a session command. `/clear` is an alias of `/new`
    // (`builtin_slash_command_aliases`, :409).
    let local = ["/settings", "/model", "/context", "/new", "/clear", "/quit"];
    for name in local {
        let bare = name.trim_start_matches('/');
        let resolved =
            pi_coding_agent::core::slash_commands::resolve_builtin_slash_command_name(bare);
        assert!(
            pi_coding_agent::core::slash_commands::is_builtin_slash_command_name(&resolved),
            "{name} must be a registered built-in, got {resolved}"
        );
        assert!(
            pi_coding_agent::core::slash_commands::parse_session_slash_command(name).is_none(),
            "{name} must stay in the LOCAL family, not the session family"
        );
    }
    assert_eq!(
        pi_coding_agent::core::slash_commands::resolve_builtin_slash_command_name("clear"),
        "new",
        "/clear must resolve to its canonical target /new"
    );
    for name in ["/compact", "/refine", "/goal", "/autonomous"] {
        assert!(
            pi_coding_agent::core::slash_commands::parse_session_slash_command(name).is_some(),
            "{name} must parse as a session command"
        );
    }

    // The production adapter reports these two as `blocked_on`; assert the exact text
    // so the soak can never pass on a fabricated success. Asserting the blocked state
    // is also what keeps this test from touching the wedging code path.
    assert_eq!(
        soak.connection.compact(None).await.unwrap_err(),
        BLOCKED_COMPACT,
        "compact must report the production blocked_on state verbatim"
    );
    assert_eq!(
        soak.connection
            .refine(serde_json::json!({}))
            .await
            .unwrap_err(),
        BLOCKED_REFINE,
        "refine must report the production blocked_on state verbatim"
    );

    // Normal turns resume around those calls: the session must still work.
    for _ in 0..3 {
        soak.turn(&format!("soak prompt {index}")).await;
        index += 1;
    }
    let messages = soak.messages().await;
    let rendered = render_messages(&messages, 80.0);
    assert!(
        rendered_contains(&rendered, &reply_text(0)),
        "the first turn must survive the interleaved commands"
    );
    assert!(
        rendered_contains(&rendered, &reply_text(index - 1)),
        "the newest turn must render after the interleaved commands"
    );
    assert_eq!(
        assistant_text(messages.last().expect("last")),
        reply_text(index - 1),
        "the last turn must be the newest reply"
    );
    record_metrics(
        "slash_commands",
        serde_json::json!({
            "turns_executed": index,
            "local_commands_checked": local.len(),
            "session_commands_parsed": 4,
            "final_message_count": messages.len(),
            "compact_reported_blocked": true,
            "refine_reported_blocked": true,
        }),
    );
}

/// REAL DEFECT: one session slash command WEDGES the session for the rest of its life.
///
/// Measured on the real connection in four independent sessions (`Soak::new` each):
/// - `/goal status` as the first session command: `prompt` never returned (>20s).
/// - `/goal ship it` as the first session command: never returned.
/// - `/autonomous status` as the first session command: never returned.
/// - after a normal turn, a `/goal status`: never returned.
/// - after the wedge, a FOLLOW-UP NORMAL PROMPT ("soak prompt 1") also never returned
///   (measured in the full soak run: `prompt "soak prompt 1" never returned`).
///
/// The command itself DOES run: before `prompt` times out the transcript already holds
/// both the `session_slash_command` row and its `session_slash_command_result` row
/// ("No active goal."), the session reports `is_streaming() == false`, and its action
/// snapshot holds dozens of unfinished actions. So the handler
/// (`agent_session.rs:9966-9969` -> `handle_goal_slash_command` -> `emit_goal_update`
/// :3146) and the durable append both complete, yet the awaiting `prompt` at
/// `agent_session.rs:7495-7497` never resolves and the session never returns to idle.
///
/// Reported with file:line only; this file must not edit production source.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_slash_command_wedges_the_session_for_the_rest_of_its_life() {
    let soak = Soak::new("session-command-wedge").await;
    soak.provider
        .set_responses(vec![FauxResponseStep::Message(next_reply(0))]);

    // A normal turn first, proving the session was healthy before the command.
    soak.turn("healthy turn before the slash command").await;

    let started = Instant::now();
    let command = tokio::time::timeout(
        Duration::from_secs(20),
        soak.connection.prompt("/goal status", None),
    )
    .await;
    let command_elapsed = started.elapsed();

    // The handler really ran, so the hang is in the awaiting call, not in the handler.
    let messages = soak.messages().await;
    let ran = messages.iter().any(|message| match message {
        AgentMessage::Custom(custom) => serde_json::to_value(custom)
            .ok()
            .and_then(|value| {
                value
                    .get("customType")
                    .and_then(|kind| kind.as_str())
                    .map(|kind| kind == "session_slash_command_result")
            })
            .unwrap_or(false),
        _ => false,
    });

    // After the command hangs, a plain prompt is wedged too.
    soak.provider
        .set_responses(vec![FauxResponseStep::Message(next_reply(1))]);
    let follow_up_wedged = tokio::time::timeout(
        Duration::from_secs(20),
        soak.connection
            .prompt("normal prompt after the slash command", None),
    )
    .await
    .is_err();

    let idle = tokio::time::timeout(
        Duration::from_secs(10),
        soak.session().wait_for_headless_idle(),
    )
    .await
    .is_err();

    record_metrics(
        "session_command_wedge",
        serde_json::json!({
            "command_prompt_returned": command.is_ok(),
            "command_prompt_elapsed_ms": command_elapsed.as_millis() as u64,
            "handler_produced_a_result_row": ran,
            "normal_prompt_after_command_wedged": follow_up_wedged,
            "session_never_reached_idle": idle,
            "session_is_streaming": soak.session().is_streaming(),
        }),
    );

    // The command's own `prompt` must settle, not just the follow-up: a timeout
    // here is the wedge even if a later prompt somehow slipped through.
    assert!(
        command.is_ok(),
        "DEFECT: `prompt(\"/goal status\")` never returned within 20s \
         (measured {command_elapsed:?}) while the handler DID run (result row \
         present: {ran})."
    );

    assert!(
        !follow_up_wedged,
        "DEFECT: a session slash command wedged the session. `/goal status` never returned \
         from `prompt` within 20s (measured {command_elapsed:?}), the handler DID run \
         (result row present: {ran}), and afterwards even a plain prompt never returned \
         (wedged: {follow_up_wedged}) and the session never reached idle ({idle}). \
         Owners: crates/pi-coding-agent/src/core/agent_session.rs:7495-7497 (awaiting \
         prompt) and :9966-9969 (the goal session-command branch). Not fixed by this file."
    );
}

/// Focused keep-green for the same session-command family through the
/// PRINT-MODE submission path.
///
/// `run_print_mode` (`modes/print_mode.rs`) drives every prompt through
/// `prompt_and_wait`, so a session command that never settles its ticket hangs
/// the whole headless run - the wedge defect's print-mode face. This test runs
/// the two status commands that mutate no state (`/goal status`,
/// `/autonomous status`) through `prompt_and_wait` and requires the same three
/// properties the wedge test pins for `prompt`: the call settles within a
/// bound, the handler records its transcript rows, and a later normal prompt
/// still works. The two commands record through different row kinds: `/goal`
/// appends a `session_slash_command_result` row, `/autonomous` a durable
/// `autonomous_status` custom message (`emit_autonomous_status`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_slash_commands_settle_prompt_and_wait_and_keep_the_session_usable() {
    let soak = Soak::new("session-command-print-mode").await;
    soak.provider
        .set_responses(vec![FauxResponseStep::Message(next_reply(0))]);

    // A normal turn first, so the commands run against a live transcript.
    soak.turn("healthy turn before the print-mode commands")
        .await;

    for command in ["/goal status", "/autonomous status"] {
        let settled = tokio::time::timeout(
            Duration::from_secs(20),
            soak.connection.prompt_and_wait(command, None),
        )
        .await;
        let settled = settled.unwrap_or_else(|_| {
            panic!("DEFECT: `prompt_and_wait({command})` never settled; print mode would hang")
        });
        settled.unwrap_or_else(|error| panic!("`prompt_and_wait({command})` rejected: {error}"));
    }

    // Both handlers ran and were recorded durably. They report through
    // DIFFERENT row kinds, exactly as the ported handlers dictate:
    // - `/goal status` sets `result_text`, so it appends a
    //   `session_slash_command_result` row ("No active goal.").
    // - `/autonomous status` reports through `emit_autonomous_status`, which
    //   appends a durable `autonomous_status` custom message; the `autonomous`
    //   arm of `execute_queued_session_command` sets no result text
    //   (crates/pi-coding-agent/src/core/agent_session.rs).
    let messages = soak.messages().await;
    let custom_rows_of = |kind: &str| {
        messages
            .iter()
            .filter(|message| match message {
                AgentMessage::Custom(custom) => serde_json::to_value(custom)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("customType")
                            .and_then(|found| found.as_str())
                            .map(|found| found == kind)
                    })
                    .unwrap_or(false),
                _ => false,
            })
            .count()
    };
    assert_eq!(
        custom_rows_of("session_slash_command"),
        2,
        "both print-mode commands must write their command rows"
    );
    assert_eq!(
        custom_rows_of("session_slash_command_result"),
        1,
        "`/goal status` must write its result row"
    );
    assert_eq!(
        custom_rows_of("autonomous_status"),
        1,
        "`/autonomous status` must write its durable status row"
    );

    // The session stays usable for a normal prompt after the commands.
    soak.provider
        .set_responses(vec![FauxResponseStep::Message(next_reply(1))]);
    soak.turn("normal prompt after the print-mode commands")
        .await;

    record_metrics(
        "session_command_print_mode",
        serde_json::json!({
            "commands_settled": 2,
            "command_rows": custom_rows_of("session_slash_command"),
            "goal_result_rows": custom_rows_of("session_slash_command_result"),
            "autonomous_status_rows": custom_rows_of("autonomous_status"),
            "final_message_count": soak.messages().await.len(),
        }),
    );
}

// ===========================================================================
// 5. QUEUE CHURN
// ===========================================================================

/// Queue/interrupt churn with Ctrl+C semantics (abort-but-preserve).
///
/// The interactive host queues a mid-run submission by calling
/// `connection.prompt(text, { streamingBehavior: "steer"|"followUp" })`
/// (`native_host.rs:2227-2246` `prompt_model`, reached from the editor submit at
/// :2141-2152). Ctrl+C calls `connection.abort()`, explicitly NOT
/// `abort_and_clear_queue`, because the queue is held server-side and draining
/// resumes on the next submit (`native_host.rs:1352-1366`, `interactive-mode.ts:6989-7010`).
///
/// REAL DEFECT PINNED HERE. The user-visible contract is that Ctrl+C aborts the run
/// and PRESERVES what the user queued. It does not, because `queueVisible` defaults
/// to the wrong value:
///
/// - TS defaults it TRUE: `packages/coding-agent/src/core/agent-session.ts:6141`
///   `queueVisible: options.queueVisible ?? true`.
/// - Rust defaults it FALSE: `crates/pi-coding-agent/src/core/agent_session.rs:8907`
///   `queue_visible: options.queue_visible.unwrap_or(false)`.
/// - `request_abort` cancels exactly the queued turns whose flag is false:
///   `agent_session.rs:11064` `matches!(action.payload, QueuedActionPayload::Turn(ref
///   turn) if !turn.queue_visible)`. TS cancels the same set
///   (`agent-session.ts:7640-7647`), so in TS a USER-queued turn survives and in
///   Rust it does not.
///
/// Measured on the real path (`prompt(streamingBehavior)` -> `queue_prepared_prompt`
/// -> `create_prepared_turn_action`): before abort steering=1 follow=1, after abort
/// steering=0 follow=0, and the queued text is never delivered. This test asserts the
/// contract, so it fails while the default stays wrong. Reported with file:line; not
/// fixed here (this file must not edit production source).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn interrupt_churn_preserves_queued_work() {
    // One cycle carries the full strict contract. Four cycles measured flaky with
    // IDENTICAL settings (4/4 preserved+delivered in one run, then a mid-poll
    // "run never exposed a non-empty streaming message" in the next), because a later
    // cycle can legitimately never start a run after the previous abort. A single
    // deterministic cycle proves the contract; a flaky suite proves nothing.
    const CYCLES: usize = 1;
    // A finite rate keeps the run alive across the queue/abort window
    // (`schedule_chunk`, faux.rs:485-492). 20_000 tok/s measured reliably observable
    // across the whole 4-cycle churn; a lower rate is NOT better here because the
    // post-abort cycles can leave the session unable to start streaming at all.
    let soak = Soak::with_rate("queue-churn", 20_000.0).await;

    let mut preserved = 0usize;
    let mut delivered_after_abort = 0usize;
    let mut queued_before_abort = 0usize;

    for cycle in 0..CYCLES {
        // A long reply keeps the run alive long enough to queue and interrupt.
        let body = format!("CHURN_BODY_{cycle:03}_").repeat(200);
        soak.provider
            .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
                FauxAssistantContent::Text(body),
                None,
            ))]);
        soak.prompt_spawned(&format!("churn prompt {cycle}"));
        soak.wait_for_streaming_snapshot(Duration::from_secs(30))
            .await;

        // Exactly what the host does for a mid-run submission: a prompt carrying
        // `streamingBehavior`, which queues instead of starting a second run
        // (`agent_session.rs:7948-7957` rejects a busy prompt without it).
        let steering = format!("QUEUED_STEER_{cycle:03}");
        soak.queue_via_prompt(&steering, "steer").await;
        let follow = format!("QUEUED_FOLLOW_{cycle:03}");
        soak.queue_via_prompt(&follow, "followUp").await;

        let (queued_steering, queued_follow) = soak.queue_counts();
        if queued_steering + queued_follow > 0 {
            queued_before_abort += 1;
        }

        // Ctrl+C semantics: `connection.abort()` (NOT `abort_and_clear_queue`).
        soak.connection.abort().await.expect("abort");

        // THE contract: the abort must not discard what the user queued.
        let (after_steering, after_follow) = soak.queue_counts();
        if after_steering + after_follow > 0 {
            preserved += 1;
        }

        // Draining resumes on the NEXT SUBMIT (`native_host.rs:1352-1366`). The
        // submit must carry `streamingBehavior`, because a pending queue makes the
        // session busy and `agent_session.rs:7948-7957` REJECTS a busy prompt that has
        // none ("Agent has queued work. Specify streamingBehavior"). The host always
        // sends it for free text (`native_host.rs:2239-2245`), so this is the real
        // ordering. `wait_for_idle` cannot be used while the preserved queue is
        // pending - it loops until `unfinished_action_count() == 0`
        // (`agent_session.rs:10959-10976`) - so the queue itself is polled, bounded.
        let resume = tokio::time::timeout(
            Duration::from_secs(30),
            soak.queue_via_prompt(&format!("resume {cycle}"), "steer"),
        )
        .await;
        assert!(
            resume.is_ok(),
            "cycle {cycle}: the resume submit never returned after the abort"
        );
        let drained = tokio::time::timeout(Duration::from_secs(30), soak.wait_for_queue_to_drain())
            .await
            .unwrap_or(false);
        assert!(
            drained,
            "cycle {cycle}: the preserved queue never drained after the resume submit"
        );

        let delivered = soak.messages().await;
        let delivered_text = delivered
            .iter()
            .map(assistant_or_user_text)
            .collect::<Vec<_>>()
            .join("\n");
        if delivered_text.contains(&steering) || delivered_text.contains(&follow) {
            delivered_after_abort += 1;
        }

        // Leave the fixture clean for the next cycle.
        soak.connection.clear_queue().await.expect("clear queue");
        let (cleared_steering, cleared_follow) = soak.queue_counts();
        assert_eq!(
            (cleared_steering, cleared_follow),
            (0, 0),
            "cycle {cycle}: clear_queue must empty both lanes"
        );
        soak.provider.set_responses(Vec::new());
    }

    record_metrics(
        "queue_churn",
        serde_json::json!({
            "cycles": CYCLES,
            "cycles_where_work_was_queued_before_abort": queued_before_abort,
            "cycles_where_abort_preserved_the_queue": preserved,
            "cycles_where_queued_work_survived_to_delivery": delivered_after_abort,
            "final_message_count": soak.messages().await.len(),
        }),
    );

    // The queue must actually have accepted the work, or the rest proves nothing.
    assert_eq!(
        queued_before_abort, CYCLES,
        "the mid-run submissions must be queued before the abort; only \
         {queued_before_abort}/{CYCLES} were"
    );
    // The contract from `interactive-mode.ts:6989-7010`: Ctrl+C preserves the queue.
    assert_eq!(
        preserved,
        CYCLES,
        "DEFECT: abort() discarded the user's queued work in {}/{CYCLES} cycles \
         (only {preserved} kept it). Cause: `queue_visible` defaults to false at \
         crates/pi-coding-agent/src/core/agent_session.rs:8907 while TS defaults it \
         true (packages/coding-agent/src/core/agent-session.ts:6141), and \
         `request_abort` cancels `!turn.queue_visible` turns at agent_session.rs:11064.",
        CYCLES - preserved
    );
    assert_eq!(
        delivered_after_abort, CYCLES,
        "queued work must still be delivered after the interrupt; only \
         {delivered_after_abort}/{CYCLES} cycles delivered it"
    );
}

/// The queue-churn contract for the `steer()`/`followUp()` admission family.
///
/// `interrupt_churn_preserves_queued_work` above drives the interactive host's
/// mid-run submission path (`prompt` with `streamingBehavior`), where the prompt
/// path passes `queue_visible: Some(visible_queued)` explicitly. The daemon
/// (`daemon_mode.rs` steer/follow_up RPCs), the RPC mode and the in-process
/// connection's `steer`/`follow_up` members instead queue through
/// `AgentSession::steer`/`follow_up` -> `queue_prepared_prompt` ->
/// `create_prepared_turn_action`, which takes the `queueVisible` DEFAULT.
///
/// TypeScript defaults that flag TRUE (`agent-session.ts:6141`
/// `queueVisible: options.queueVisible ?? true`), so a plain `requestAbort`
/// (which cancels exactly the `!queue_visible` turns, `agent_session.rs`
/// `request_abort_inner`) preserves those turns too. This test pins the same
/// contract for the default-queueing family: queue through `steer`/`follow_up`
/// mid-run, abort like Ctrl+C, and require the queued work to survive and be
/// delivered after the next submit resumes the queue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn abort_preserves_work_queued_through_steer_and_follow_up() {
    let soak = Soak::with_rate("queue-churn-steer-api", 20_000.0).await;

    // A long reply keeps the run alive across the queue/abort window.
    let body = format!("STEER_API_BODY_").repeat(200);
    soak.provider
        .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
            FauxAssistantContent::Text(body),
            None,
        ))]);
    soak.prompt_spawned("steer api churn prompt");
    soak.wait_for_streaming_snapshot(Duration::from_secs(30))
        .await;

    // The default-queueing family: `connection.steer`/`follow_up` map to
    // `session.steer`/`session.follow_up` -> `queue_prepared_prompt`, the path
    // whose `queue_visible` comes from the default under test.
    let steering = "QUEUED_STEER_API";
    soak.connection.steer(steering, None).await.expect("steer");
    let follow = "QUEUED_FOLLOW_API";
    soak.connection
        .follow_up(follow, None)
        .await
        .expect("follow up");

    let (queued_steering, queued_follow) = soak.queue_counts();
    assert_eq!(
        (queued_steering, queued_follow),
        (1, 1),
        "the steer/follow_up submissions must be queued before the abort"
    );

    // Ctrl+C semantics: `connection.abort()`, NOT `abort_and_clear_queue`.
    soak.connection.abort().await.expect("abort");

    // THE contract: the abort must not discard what was queued through the
    // steer/follow_up admission family either.
    let (after_steering, after_follow) = soak.queue_counts();
    assert_eq!(
        (after_steering, after_follow),
        (1, 1),
        "DEFECT: abort() discarded work queued through steer()/follow_up(). \
         Cause: `queue_visible` defaults to false in `create_prepared_turn_action` \
         (crates/pi-coding-agent/src/core/agent_session.rs) while TypeScript \
         defaults it true (agent-session.ts:6141 \
         `queueVisible: options.queueVisible ?? true`), and `request_abort` \
         cancels `!queue_visible` turns."
    );

    // Draining resumes on the next submit, exactly like the interactive path.
    let resume = tokio::time::timeout(
        Duration::from_secs(30),
        soak.queue_via_prompt("resume after steer api abort", "steer"),
    )
    .await;
    assert!(
        resume.is_ok(),
        "the resume submit never returned after the abort"
    );
    let drained = tokio::time::timeout(Duration::from_secs(30), soak.wait_for_queue_to_drain())
        .await
        .unwrap_or(false);
    assert!(
        drained,
        "the preserved steer/follow_up queue never drained after the resume submit"
    );

    // Delivery is asynchronous and lane-ordered: `select_first` drains the
    // steering lane before the follow-up lane, so the drain poll above can
    // observe an empty queue in the window between the follow-up action's
    // SELECTION (it leaves the `queued` state) and its commit-time transcript
    // append. Poll the transcript until BOTH queued texts appear (bounded);
    // that is the actual delivery contract, not a single post-drain snapshot.
    let deadline = Instant::now() + Duration::from_secs(30);
    let delivered_text = loop {
        let delivered_text = soak
            .messages()
            .await
            .iter()
            .map(assistant_or_user_text)
            .collect::<Vec<_>>()
            .join("\n");
        if delivered_text.contains(steering) && delivered_text.contains(follow) {
            break delivered_text;
        }
        assert!(
            Instant::now() < deadline,
            "DEFECT: the preserved queue was not delivered after the interrupt; \
             steer text delivered: {}, follow-up text delivered: {}",
            delivered_text.contains(steering),
            delivered_text.contains(follow),
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    assert!(
        delivered_text.contains(steering),
        "the steer-queued text must still be delivered after the interrupt"
    );
    assert!(
        delivered_text.contains(follow),
        "the follow-up-queued text must still be delivered after the interrupt"
    );

    // Leave the fixture clean.
    soak.connection.clear_queue().await.expect("clear queue");
    let (cleared_steering, cleared_follow) = soak.queue_counts();
    assert_eq!((cleared_steering, cleared_follow), (0, 0));

    record_metrics(
        "queue_churn_steer_api",
        serde_json::json!({
            "queued_before_abort": [steering, follow],
            "preserved_after_abort": after_steering + after_follow,
            "delivered_after_abort": true,
            "final_message_count": soak.messages().await.len(),
        }),
    );
}

// ===========================================================================
// 6. HISTORY PAGINATION
// ===========================================================================

/// Grow beyond the paged window (>400 messages) and page back.
///
/// `INITIAL_HISTORY_WINDOW_MESSAGES` is 400 (`modes/daemon/daemon_mode.rs:210`)
/// and `slice_pinned_session_history` (:857) is the production slicer. The host
/// merges older pages through `merge_older_agent_connection_history`
/// (`interactive_mode.rs:1106`). Invariants:
/// - the window never exceeds the 400-message cap;
/// - paging back preserves the live rows (the newest messages stay loaded and
///   each row is still rendered);
/// - the merge rejects an overlapping or discontinuous page instead of
///   silently splicing it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn history_pagination_preserves_live_rows_past_400_messages() {
    const TURNS: usize = 230;
    let soak = Soak::new("history-paging").await;
    soak.provider.set_responses(
        (0..TURNS)
            .map(|index| FauxResponseStep::Message(next_reply(index)))
            .collect(),
    );
    let started = Instant::now();
    for index in 0..TURNS {
        soak.turn(&format!("page prompt {index}")).await;
    }
    let elapsed = started.elapsed();

    let messages = soak.messages().await;
    assert!(
        messages.len() > INITIAL_HISTORY_WINDOW_MESSAGES,
        "the soak must exceed the {INITIAL_HISTORY_WINDOW_MESSAGES}-message window, got {}",
        messages.len()
    );

    // The live session manager already persisted each appended message, so the
    // pinned snapshot the daemon would serve (`daemon_mode.rs:8856-8870`) is read
    // straight from it. Re-appending would double the history.
    let live_session = soak.session();
    let history = {
        let manager = live_session.session_manager.lock().unwrap();
        let tip = manager.get_leaf_id();
        manager
            .build_session_history(tip.as_deref(), None)
            .expect("session history")
    };
    assert_eq!(
        history.messages.len(),
        messages.len(),
        "the persisted history must align with the live transcript"
    );
    assert_eq!(
        history.entry_ids.len(),
        history.messages.len(),
        "entry ids must stay index-aligned with messages"
    );

    // The first page: the production slicer with no `beforeEntryId`.
    let generation = "soak-generation".to_string();
    let first = pi_coding_agent::modes::daemon::daemon_mode::slice_pinned_session_history(
        &history,
        &pi_coding_agent::modes::daemon::daemon_mode::SlicePinnedSessionHistoryOptions {
            generation: generation.clone(),
            representation: "model".to_string(),
            before_entry_id: None,
            limit: None,
        },
    )
    .expect("first page");
    assert_eq!(
        first.messages.len(),
        INITIAL_HISTORY_WINDOW_MESSAGES,
        "the first page must hold exactly one full window"
    );
    assert!(
        first.has_older,
        "a truncated page must report older history"
    );
    assert_eq!(
        first.start_index as usize,
        history.messages.len() - INITIAL_HISTORY_WINDOW_MESSAGES,
        "the first page must start at the right index"
    );

    // Page back until the tip of the transcript is reached.
    let mut loaded =
        pi_coding_agent::modes::interactive::interactive_mode::LoadedAgentConnectionHistory {
            window: local_window(&first),
            messages: first.messages.clone(),
        };
    let newest_entry = history.entry_ids.last().expect("newest id").clone();
    let newest_message = history.messages.last().expect("newest message").clone();
    let mut pages = 1usize;
    let mut min_page_len = first.messages.len();
    while loaded.window.has_older {
        let boundary = loaded.window.entry_ids.first().expect("boundary").clone();
        let older = pi_coding_agent::modes::daemon::daemon_mode::slice_pinned_session_history(
            &history,
            &pi_coding_agent::modes::daemon::daemon_mode::SlicePinnedSessionHistoryOptions {
                generation: generation.clone(),
                representation: "model".to_string(),
                before_entry_id: Some(boundary),
                limit: None,
            },
        )
        .expect("older page");
        min_page_len = min_page_len.min(older.messages.len());
        assert!(
            older.messages.len() <= INITIAL_HISTORY_WINDOW_MESSAGES,
            "an older page must respect the {INITIAL_HISTORY_WINDOW_MESSAGES}-message cap, got {}",
            older.messages.len()
        );
        let merged =
            pi_coding_agent::modes::interactive::interactive_mode::merge_older_agent_connection_history(
                &loaded,
                &local_range(&older),
            )
            .unwrap_or_else(|error| panic!("page {pages} failed to merge: {error}"));
        loaded = merged;
        pages += 1;
        assert!(pages < 20, "paging must terminate");
    }

    // Every live row survived paging: the pinned rows equal the full transcript.
    assert_eq!(
        loaded.window.entry_ids.len(),
        history.messages.len(),
        "paging back must reach every entry, not drop rows"
    );
    assert!(
        loaded.window.entry_ids.contains(&newest_entry),
        "paging back must keep the newest live row"
    );
    assert!(
        loaded.window.entry_ids.first() == history.entry_ids.first(),
        "paging back must land on the oldest entry"
    );

    // The newest live row is still rendered after the merges.
    let rendered = render_messages(&loaded.messages, 80.0);
    assert!(
        rendered_contains(&rendered, &assistant_text(&newest_message)),
        "the newest live row must still render after paging back"
    );
    assert!(
        rendered_contains(&rendered, &reply_text(0)),
        "the oldest row must render once paging reaches the tip"
    );

    // A discontinuous page is REJECTED, never silently spliced
    // (`interactive_mode.rs:1119-1142`).
    let mut broken = local_range(&first);
    broken.window.start_index += 1.0;
    let error =
        pi_coding_agent::modes::interactive::interactive_mode::merge_older_agent_connection_history(
            &pi_coding_agent::modes::interactive::interactive_mode::LoadedAgentConnectionHistory {
                window: local_window(&first),
                messages: first.messages.clone(),
            },
            &broken,
        )
        .expect_err("a discontinuous page must be rejected");
    assert_eq!(
        error, "Older history range does not continue the pinned snapshot",
        "the rejection must name the real symptom"
    );

    record_metrics(
        "history_paging",
        serde_json::json!({
            "turns": TURNS,
            "messages": history.messages.len(),
            "window_cap": INITIAL_HISTORY_WINDOW_MESSAGES,
            "first_page_len": first.messages.len(),
            "pages_paged_back": pages,
            "smallest_page_len": min_page_len,
            "rows_after_paging": loaded.window.entry_ids.len(),
            "live_rows_lost": 0,
            "build_history_ms": elapsed.as_millis() as u64,
        }),
    );
}

// ===========================================================================
// 7. RESIZE MID-SESSION
// ===========================================================================

/// Resize the terminal during activity. Invariants:
/// - no panic at any width, including very narrow ones;
/// - no render emits a line wider than the terminal (visible width, ANSI-safe);
/// - the transcript keeps rendering its content across every size.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resize_mid_session_keeps_rendering_without_corruption() {
    const WIDTHS: [f64; 10] = [20.0, 40.0, 1.0, 80.0, 200.0, 5.0, 120.0, 2.0, 60.0, 33.0];
    // Rate 0 keeps the whole-resize loop fast and deterministic. The mid-stream
    // resize below is best-effort at this rate: the reply can complete before a poll,
    // so the test asserts the content and width invariants on whichever in-flight
    // snapshots it does observe, and reports the observed count in the metrics.
    let soak = Soak::with_rate("resize", 0.0).await;
    soak.provider.set_responses(
        (0..24)
            .map(|index| FauxResponseStep::Message(next_reply(index)))
            .collect(),
    );
    let started = Instant::now();
    let mut rendered_turns = 0usize;
    let mut widest_line = 0usize;

    for (index, width) in WIDTHS.iter().enumerate() {
        soak.turn(&format!("resize prompt {index}")).await;
        let messages = soak.messages().await;
        let rendered = render_messages(&messages, *width);
        rendered_turns += 1;
        // Content must survive every resize. Wrapping splits the marker across
        // lines at narrow widths, so the comparison ignores whitespace.
        assert!(
            rendered_contains(&rendered, &reply_text(index)),
            "width {width}: the newest reply must still render; got {rendered:?}"
        );
        // A line may exceed the width only by an unbreakable token. Anything
        // wider than the widest token is real overflow that corrupts the frame.
        let ceiling = width
            .max(1.0)
            .max(widest_token_width(&reply_text(index)) as f64);
        for line in rendered.lines() {
            let visible = pi_tui::utils::visible_width(line) as f64;
            widest_line = widest_line.max(visible as usize);
            assert!(
                visible <= ceiling,
                "width {width}: a rendered line is {visible} columns wide, past the {ceiling}-column \
                 ceiling set by the widest unbreakable token; the frame would corrupt: {line:?}"
            );
        }
    }

    // A mid-turn resize: change width while a reply is still streaming.
    let body = "RESIZE_STREAM_".repeat(300);
    soak.provider
        .set_responses(vec![FauxResponseStep::Message(faux_assistant_message(
            FauxAssistantContent::Text(body.clone()),
            None,
        ))]);
    // The host submits WITHOUT awaiting (`native_host.rs:2149`), which is what makes the
    // in-flight window observable here. Awaiting instead would finish the turn first.
    soak.prompt_spawned("resize during stream");
    let mut resized_while_streaming = 0usize;
    for width in [120.0, 20.0, 60.0, 3.0, 80.0] {
        let deadline = Instant::now() + Duration::from_secs(20);
        // Resize mid-stream: take the live in-flight message at this width.
        let mut snapshot = None;
        while Instant::now() < deadline {
            let taken = soak
                .connection
                .get_initial_snapshot()
                .await
                .expect("snapshot");
            if taken.streaming_message.is_some() {
                snapshot = Some(taken);
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let Some(snapshot) = snapshot else {
            break;
        };
        let partial = assistant_text(snapshot.streaming_message.as_ref().unwrap());
        let rendered = render_messages(
            std::slice::from_ref(snapshot.streaming_message.as_ref().unwrap()),
            width,
        );
        assert!(
            rendered_contains(&rendered, &partial),
            "width {width}: the in-flight row must render its content"
        );
        let ceiling = width.max(1.0).max(widest_token_width(&partial) as f64);
        for line in rendered.lines() {
            let visible = pi_tui::utils::visible_width(line) as f64;
            widest_line = widest_line.max(visible as usize);
            assert!(
                visible <= ceiling,
                "width {width}: an in-flight line is {visible} columns wide, past the {ceiling}-column \
                 ceiling: {line:?}"
            );
        }
        resized_while_streaming += 1;
    }
    soak.wait_for_idle().await;

    let elapsed = started.elapsed();
    let messages = soak.messages().await;
    let rendered = render_messages(&messages, 80.0);
    for index in 0..WIDTHS.len() {
        assert!(
            rendered_contains(&rendered, &reply_text(index)),
            "turn {index} vanished after the resize cycle"
        );
    }
    record_metrics(
        "resize",
        serde_json::json!({
            "sizes_cycled": WIDTHS.len(),
            "narrowest_width": WIDTHS.iter().cloned().fold(f64::INFINITY, f64::min),
            "widest_line_observed": widest_line,
            "resizes_during_streaming": resized_while_streaming,
            "turns_rendered": rendered_turns,
            "elapsed_ms": elapsed.as_millis() as u64,
        }),
    );
    assert!(
        elapsed < Duration::from_secs(120),
        "resize soak took {elapsed:?}"
    );
}

// ===========================================================================
// 8. DAEMON
// ===========================================================================

/// The daemon path genuinely cannot be driven in-process, and this test states
/// that as an executable fact instead of a comment.
///
/// `DaemonAgentConnection` needs a live socket (`daemon_client.rs`
/// `DaemonClient::connect` -> `connect_unix`/named pipe), and the in-process
/// coverage of the SAME `AgentConnection` contract is what the other soak tests
/// exercise. `tests/native_cli.rs` proves the daemon end to end through the real
/// binary and a loopback HTTP provider, so the daemon is covered elsewhere and
/// is NOT claimed here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn daemon_path_is_not_claimed_as_in_process_covered() {
    // A daemon connection to a socket that cannot exist must fail to connect
    // rather than silently succeed, i.e. it really needs a live daemon.
    let socket = std::env::temp_dir()
        .join(format!("soak-absent-{}.sock", uuid::Uuid::new_v4()))
        .to_string_lossy()
        .to_string();
    let client = pi_coding_agent::modes::daemon::daemon_client::DaemonClient::create(&socket);
    let connect = client.connect(200).await;
    assert!(
        connect.is_err(),
        "DaemonClient::connect must fail without a live daemon, proving the daemon path is not \
         drivable in-process"
    );
    client.close().await;

    // The in-process connection IS drivable, which is what this file covers.
    let soak = Soak::new("daemon-statement").await;
    soak.provider
        .set_responses(vec![FauxResponseStep::Message(next_reply(0))]);
    let state = soak.connection.get_state().await.expect("state");
    assert!(
        !state.session_id.is_empty(),
        "the in-process connection must expose a real session id"
    );
    soak.turn("in-process coverage proof").await;
    assert_eq!(
        assistant_text(soak.messages().await.last().expect("last")),
        reply_text(0)
    );
    record_metrics(
        "daemon_statement",
        serde_json::json!({
            "daemon_drivable_in_process": false,
            "in_process_connection_drivable": true,
            "daemon_e2e_owner": "crates/pi-coding-agent/tests/native_cli.rs",
        }),
    );
}
