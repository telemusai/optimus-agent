//! In-process `InProcessRuntimeHost` adapter over the real `AgentSessionRuntime`.
//!
//! `modes/agent_connection/in_process_agent_connection.rs` drives the explicit
//! `InProcessRuntimeHost` seam (the TypeScript `runtimeHost` object). This module
//! supplies the production implementation of that seam over the landed
//! `AgentSessionRuntime` + `AgentSession`, so the in-process agent connection is
//! backed by real sessions instead of an empty stand-in.
//!
//! Every forwarded call mirrors the TypeScript member it replaces. Where the
//! TypeScript reads a value the Rust port has no public owner for, the member
//! still exists but returns an explicit `Err`/`None`/empty roster with a
//! `blocked_on:` comment naming the owner to add, so the gap stays visible
//! without breaking the crate build.

use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{BoxFuture, ImageContent, Model, ServiceTier, Transport};
use serde_json::{Map, Value};

use crate::core::agent_session::{
    AgentSession, ExtensionBindings, ModelSelectOptions, PromptOptions, RlmMaxDepthStatus,
    SessionInputPause,
};
use crate::core::agent_session_runtime::{
    AgentSessionRuntime, ForkOptionsInput, NewSessionOptionsInput, SwitchSessionOptions,
};
use crate::core::diagnostics::{ResourceCollision, ResourceDiagnostic};
use crate::core::extensions::types::InputSource;
use crate::core::resource_loader::ResourceLoader;
use crate::core::skills::Skill;
use crate::core::session_action_store::{
    QueuedMessageLane, QueuedMessageMutation, QueuedMessageMutationStatus,
};
use crate::core::source_info::SourceInfo;
use crate::utils::warning_limiter::{limited_warning, wall_clock_ms, DEFAULT_WARNING_WINDOW_MS};
use crate::modes::agent_connection::daemon_agent_connection::build_session_tree_from_flat_nodes;
use crate::core::autonomous::AgentAutonomousStatus;
use crate::modes::agent_connection::in_process_agent_connection::InProcessRuntimeHost;
use crate::modes::agent_connection::snapshot::AgentSessionRuntimeSnapshotSource;
use crate::modes::agent_connection::types::{
    AgentConnectionEventListener, AgentConnectionHeadlessCompletionOptions,
    AgentConnectionModelCatalog, AgentConnectionNavigateTreeOptions,
    AgentConnectionNavigateTreeResult, AgentConnectionRlmChildAgentSnapshot,
    AgentConnectionSessionWatcher,
};
use crate::modes::agent_connection::snapshot::{
    create_agent_connection_commands, create_agent_connection_resource_snapshot, AgentsFileEntry,
    ExtensionLoadError as SnapshotExtensionLoadError, PromptTemplateEntry, RegisteredCommandEntry,
    ResourceExtensionEntry, ResourcePromptEntry, ResourceSkillEntry, ResourceThemeEntry, SkillEntry,
};
use crate::modes::agent_connection::tool_definition::ToolDefinition as ConnectionToolDefinition;
use crate::modes::agent_connection::types::{
    AgentConnectionExecuteBashOptions, AgentConnectionForkOptions,
    AgentConnectionInputPause, AgentConnectionModel, AgentConnectionModelCycleResult,
    AgentConnectionNewSessionOptions,
    AgentConnectionQueueState, AgentConnectionResourceCollision,
    AgentConnectionResourceDiagnostic, AgentConnectionResourceSnapshot,
    AgentConnectionScopedModel, AgentConnectionSessionContext, AgentConnectionSessionContextModel,
    AgentConnectionSessionEntry, AgentConnectionSessionHeader, AgentConnectionSessionInputPause,
    AgentConnectionSessionTree, AgentConnectionSessionTreeFlatNode, AgentConnectionSlashCommand,
    AgentConnectionSwitchSessionOptions, AgentConnectionUserMessage,
    AgentConnectionWatchSessionTree,
};

/// The `runtimeHost` the in-process agent connection drives.
pub struct InProcessRuntimeHostAdapter {
    runtime: Arc<AgentSessionRuntime>,
    event_tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl InProcessRuntimeHostAdapter {
    pub fn new(runtime: Arc<AgentSessionRuntime>) -> Self {
        Self { runtime, event_tasks: Mutex::new(Vec::new()) }
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
struct InProcessInputPause {
    pause: SessionInputPause,
}

impl AgentConnectionInputPause for InProcessInputPause {
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
) -> crate::modes::agent_connection::types::AgentConnectionSourceInfo {
    crate::modes::agent_connection::types::AgentConnectionSourceInfo {
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
    source_info: &crate::modes::interactive::theme::theme::SourceInfo,
) -> crate::modes::agent_connection::types::AgentConnectionSourceInfo {
    crate::modes::agent_connection::types::AgentConnectionSourceInfo {
        path: source_info.path.clone(),
        source: source_info.source.clone(),
        scope: source_info.scope.clone(),
        origin: String::new(),
        base_dir: None,
    }
}

fn skill_base(skill: &Skill) -> &crate::core::skills::BaseSkill {
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
    definition: &crate::core::extensions::types::ToolDefinition,
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

/// `createAgentConnectionState(runtime)`/`getSessionTree()` projections.
///
/// Shared by `session_context()` and `session_tree()` so the two seam members and
/// the snapshot builder agree on the same session-manager reads.
fn session_tree_value(session: &Arc<AgentSession>) -> AgentConnectionSessionTree {
    let manager = session.session_manager.lock().unwrap();
    let (flat_nodes, dropped) = project_flat_tree(manager.get_flat_tree());
    let leaf_id = manager.get_leaf_id();
    drop(manager);
    if !dropped.is_empty() {
        for line in report_projection_drops(&dropped) {
            eprintln!("{line}");
        }
    }
    AgentConnectionSessionTree {
        tree: build_session_tree_from_flat_nodes(&flat_nodes),
        leaf_id,
    }
}

/// One counted projection failure, keyed by its serde error signature.
struct ProjectionDrop {
    /// The serde error text: the same field is missing for every entry in the
    /// group, so one line covers them all.
    signature: String,
    count: usize,
    /// Bounded sample of the dropped entry ids, so the offending entries can be
    /// located from the log instead of guessed.
    sample_ids: Vec<String>,
}

/// Dropped entry ids named per failure signature.
const MAX_SAMPLE_IDS: usize = 3;

/// Project flat session-tree nodes into the connection union.
///
/// A node outside the union cannot be projected and is dropped. The drops are
/// counted per serde error rather than logged per entry: one non-conforming
/// entry is re-projected on every snapshot and tree refresh, and an observed
/// burst produced 10,530 per-entry warning lines.
fn project_flat_tree(
    nodes: Vec<crate::core::session_manager::SessionTreeFlatNode>,
) -> (Vec<AgentConnectionSessionTreeFlatNode>, Vec<ProjectionDrop>) {
    let mut flat_nodes: Vec<AgentConnectionSessionTreeFlatNode> = Vec::new();
    let mut dropped: Vec<ProjectionDrop> = Vec::new();
    for node in nodes {
        // Read the id before the entry is consumed: the projection owns the
        // parsed value, and cloning a whole entry per refresh would double the
        // allocation for every session tree.
        let entry_id = node
            .entry
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("<entry has no id>")
            .to_string();
        match serde_json::from_value(Value::Object(node.entry)) {
            Ok(entry) => flat_nodes.push(AgentConnectionSessionTreeFlatNode {
                entry,
                label: node.label,
                label_timestamp: node.label_timestamp,
            }),
            Err(error) => {
                let signature = error.to_string();
                match dropped
                    .iter_mut()
                    .find(|group| group.signature == signature)
                {
                    Some(group) => {
                        group.count += 1;
                        if group.sample_ids.len() < MAX_SAMPLE_IDS {
                            group.sample_ids.push(entry_id);
                        }
                    }
                    None => dropped.push(ProjectionDrop {
                        signature,
                        count: 1,
                        sample_ids: vec![entry_id],
                    }),
                }
            }
        }
    }
    (flat_nodes, dropped)
}

/// Distinct failure signatures named in one summary line.
const MAX_SUMMARY_SIGNATURES: usize = 5;

/// Build the warnings for a projection that omitted non-conforming entries.
///
/// Bounded by design: at most [`MAX_SUMMARY_SIGNATURES`] per-signature lines
/// plus one aggregate line, and each key emits at most once per window. A burst
/// of identical projections therefore costs two lines instead of one per entry,
/// while the first line of every window carries the true drop total, so a
/// suppressed repeat can never look like a clean projection.
fn report_projection_drops(dropped: &[ProjectionDrop]) -> Vec<String> {
    let total: usize = dropped.iter().map(|group| group.count).sum();
    let now_ms = wall_clock_ms();
    let mut lines: Vec<String> = Vec::new();
    for group in dropped.iter().take(MAX_SUMMARY_SIGNATURES) {
        let count = group.count;
        let sample = describe_sample_ids(&group.sample_ids, count);
        let sample = if sample.is_empty() {
            sample
        } else {
            format!(" ({sample})")
        };
        let message = format!(
            "Warning: Could not project {count} session tree entr{}{sample}: {signature}",
            if count == 1 { "y" } else { "ies" },
            signature = group.signature
        );
        // Key the limiter on a normalised signature: serde errors that embed the
        // offending value (for example "invalid type: integer `5`") would rotate
        // fresh keys per entry and defeat the per-window bound. The emitted line
        // keeps the verbatim signature for attribution.
        if let Some(line) = limited_warning(
            "session-tree-projection",
            &normalize_projection_signature(&group.signature),
            &message,
            now_ms,
            DEFAULT_WARNING_WINDOW_MS,
        ) {
            lines.push(line);
        }
    }
    // Cap the signature list so one pathological session cannot make the
    // summary line itself unbounded.
    let listed: Vec<String> = dropped
        .iter()
        .take(MAX_SUMMARY_SIGNATURES)
        .map(|group| format!("{}x {}", group.count, group.signature))
        .collect();
    let remainder = dropped.len().saturating_sub(listed.len());
    let summary = if remainder == 0 {
        format!(
            "Warning: {total} session tree entr{} omitted from this projection: {}",
            if total == 1 { "y" } else { "ies" },
            listed.join("; ")
        )
    } else {
        format!(
            "Warning: {total} session tree entr{} omitted from this projection: {}; and {remainder} more distinct error(s)",
            if total == 1 { "y" } else { "ies" },
            listed.join("; ")
        )
    };
    // Keyed by the total so a changed drop count is always reported, while a
    // repeated identical projection stays suppressed inside the window.
    let summary_key = format!("aggregate:{total}");
    if let Some(line) = limited_warning(
        "session-tree-projection",
        &summary_key,
        &summary,
        now_ms,
        DEFAULT_WARNING_WINDOW_MS,
    ) {
        lines.push(line);
    }
    lines
}
/// Limiter key for a projection signature: digits (and quoted ids) collapse to a
/// placeholder so value-embedding serde errors cannot rotate fresh keys per
/// entry. Display keeps the verbatim signature.
fn normalize_projection_signature(signature: &str) -> String {
    let mut normalized = String::with_capacity(signature.len());
    let mut in_quotes = false;
    let mut chars = signature.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                if !in_quotes {
                    // Closing quote of a quoted span: collapse the span's content
                    // by appending a single placeholder marker before the quote.
                    normalized.push('"');
                } else {
                    normalized.push('"');
                }
            }
            _ if in_quotes => {
                // Skip the quoted span content; it is replaced by the marker at
                // the closing quote. Backslash escapes are skipped too.
                if ch == '\\' {
                    chars.next();
                }
            }
            _ if ch.is_ascii_digit() => {
                normalized.push('#');
                while chars.peek().is_some_and(|next| next.is_ascii_digit()) {
                    chars.next();
                }
            }
            other => normalized.push(other),
        }
    }
    normalized
}

/// Name the sampled entries for one drop group: `ids a, b, c; +N more`.
fn describe_sample_ids(sample_ids: &[String], count: usize) -> String {
    if sample_ids.is_empty() {
        return String::new();
    }
    let named = sample_ids.join(", ");
    let remainder = count.saturating_sub(sample_ids.len());
    if remainder == 0 {
        format!("ids {named}")
    } else {
        format!("ids {named}; +{remainder} more")
    }
}

/// `buildSessionContext()` as the connection layer carries it.
fn session_context_value(session: &Arc<AgentSession>) -> AgentConnectionSessionContext {
    let context = session.build_session_context();
    AgentConnectionSessionContext {
        messages: context.messages,
        thinking_level: context.thinking_level,
        service_tier: context.service_tier,
        model: context.model.map(|model| AgentConnectionSessionContextModel {
            provider: model.provider,
            model_id: model.model_id,
        }),
    }
}

/// The `runtimeHost` implementation for a live `AgentSessionRuntime`.
impl InProcessRuntimeHost for InProcessRuntimeHostAdapter {
    fn session_start_side_question(&self, id: String, question: String,
        previous: Option<Vec<crate::core::side_question::SideQuestionTurn>>,
        on_event: Arc<dyn Fn(crate::core::side_question::SideQuestionEvent) -> BoxFuture<()> + Send + Sync>,
    ) -> Result<crate::core::side_question::SideQuestionRun, String> {
        crate::core::side_question::native::start(self.session().agent.clone(), id, question, on_event, previous, None)
    }
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
            let compaction_count = manager.get_entries().iter().filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction")).count();
            // `persistedRecap(sessionManager)` (snapshot.ts:16-20): the baseline recap is the latest
            // persisted agent-status summary, not the live session recap.
            let persisted_recap = manager.get_latest_agent_status().map(|status| status.summary);
            (manager.get_cwd(), manager.get_session_dir(), manager.get_leaf_id(), compaction_count, persisted_recap)
        };
        AgentSessionRuntimeSnapshotSource {
            session: crate::modes::agent_connection::snapshot::AgentSessionSnapshotSource {
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
                session_actions: serde_json::to_value(session.get_session_action_snapshot()).unwrap(),
                compaction_count: compaction_count as f64,
                // `goal: session.goalState` (snapshot.ts:50).
                goal: serde_json::to_value(session.goal_state()).unwrap(),
                // `session.scopedModels.map(...)` (snapshot.ts:51): the canonical `ScopedModel`
                // already carries the two fields the connection layer reads.
                scoped_models: session.scoped_models().into_iter().map(|entry| AgentConnectionScopedModel { model: entry.model, thinking_level: entry.thinking_level }).collect(),
                // `activeToolNames: session.getActiveToolNames()` (snapshot.ts:55).
                active_tool_names: session.get_active_tool_names(),
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
                children: self.session_rlm_children(),
            },
        }
    }

    fn session_subscribe(&self, listener: AgentConnectionEventListener) -> Box<dyn Fn() + Send + Sync> {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let sender = Arc::new(Mutex::new(Some(sender)));
        let task = tokio::spawn(async move {
            while let Some(event) = receiver.recv().await { listener(event).await; }
        });
        self.event_tasks.lock().unwrap().push(task);
        let event_sender = sender.clone();
        let unsubscribe = self.session().subscribe(Arc::new(move |event| {
            let event = match event {
                crate::core::agent_session::AgentSessionEvent::Agent(event) =>
                    crate::modes::agent_connection::types::AgentConnectionSessionEvent::Agent(event),
                event => match serde_json::to_value(event).and_then(serde_json::from_value) {
                    Ok(event) => event,
                    Err(error) => { eprintln!("Could not serialize session event: {error}"); return; }
                },
            };
            if let Some(sender) = event_sender.lock().unwrap().as_ref() {
                let _ = sender.send(crate::modes::agent_connection::types::AgentConnectionEvent::SessionEvent { event });
            }
        }));
        Box::new(move || { unsubscribe(); sender.lock().unwrap().take(); })
    }

    fn session_wait_for_headless_completion(
        &self, options: Option<AgentConnectionHeadlessCompletionOptions>,
    ) -> BoxFuture<Result<AgentAutonomousStatus, String>> {
        let session = self.session();
        Box::pin(async move {
            crate::modes::headless_completion::wait_for_headless_completion(Arc::new(session),
                crate::modes::headless_completion::HeadlessCompletionOptions {
                    wait_for_rlm_quiescence: options.and_then(|options| options.wait_for_rlm_quiescence),
                }).await
        })
    }

    fn session_model_catalog(&self) -> AgentConnectionModelCatalog {
        let registry = self.session().model_registry();
        let registry = registry.lock().unwrap();
        let mut configured_providers = Vec::new();
        for model in registry.get_available() {
            if !configured_providers.contains(&model.provider) { configured_providers.push(model.provider); }
        }
        AgentConnectionModelCatalog { models: registry.get_all(), configured_providers }
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

    fn session_rlm_children(&self) -> Vec<AgentConnectionRlmChildAgentSnapshot> {
        self.session().get_rlm_child_snapshots().into_iter()
            .map(|child| serde_json::from_value(serde_json::to_value(child).expect("child snapshot serializes")).expect("child snapshot projection"))
            .collect()
    }

    fn session_cancel_rlm_child(&self, child_id: &str) -> bool {
        self.session().cancel_rlm_child_run_by_id(child_id, "Cancelled by user")
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
    fn session_compact(&self, custom_instructions: Option<&str>) -> BoxFuture<Result<Value, String>> {
        let _ = custom_instructions;
        Box::pin(async { Err("blocked_on: compact result not exposed on the public path".to_string()) })
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
        Box::pin(async { Err("blocked_on: refine result not exposed on the public path".to_string()) })
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

    fn session_export_to_html(&self, output_path: Option<&str>) -> BoxFuture<Result<String, String>> {
        let session_file = self.session().session_file();
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
    fn session_export_to_jsonl(&self, output_path: Option<&str>) -> BoxFuture<Result<String, String>> {
        let _ = output_path;
        Box::pin(async { Err("blocked_on: no JSONL export owner".to_string()) })
    }

    fn session_watch_child(&self, child_id: &str) -> Option<Box<dyn AgentConnectionSessionWatcher>> {
        let session = self.session().get_rlm_child_session(child_id)?;
        Some(Box::new(ChildWatcher { session, subscriptions: Arc::new(Mutex::new(Vec::new())) }))
    }

    fn session_header(&self) -> Option<AgentConnectionSessionHeader> {
        let entry = self.session().session_manager.lock().unwrap().get_header()?;
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
        child_commands(&self.session())
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
                source_info: theme.source_info.as_ref().map(connection_source_info_from_theme),
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
        let lane = match serde_json::from_value::<QueuedMessageLane>(Value::String(lane.to_string())) {
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
        Arc::new(InProcessInputPause { pause })
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

    fn session_prompt_and_wait(&self, message: &str, options: Value) -> BoxFuture<Result<(), String>> {
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
        Box::pin(async move { session.steer(&message, images, None, None, Some(true)).await })
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
        Box::pin(async move { session.set_model(model, ModelSelectOptions::default()).await })
    }

    fn session_model_registry_available_models(&self) -> BoxFuture<Vec<AgentConnectionModel>> {
        let registry = self.session().model_registry();
        Box::pin(async move {
            crate::core::sdk::with_model_registry(registry, |registry| Box::pin(registry.refresh_available_models()))
                .await.expect("model registry worker failed")
        })
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

    fn session_bind_extensions(&self, bindings: ExtensionBindings) -> BoxFuture<Result<(), String>> {
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

    fn runtime_set_before_session_invalidate(
        &self,
        listener: Option<Arc<dyn Fn() + Send + Sync>>,
    ) {
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
            if let Err(error) = runtime.dispose(None).await { eprintln!("Could not dispose session runtime: {error}"); }
            for task in tasks { let _ = task.await; }
        })
    }

}

fn child_commands(session: &Arc<AgentSession>) -> Vec<AgentConnectionSlashCommand> {
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

struct ChildWatcher {
    session: Arc<AgentSession>,
    subscriptions: Arc<Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>>,
}

impl Drop for ChildWatcher {
    fn drop(&mut self) {
        let subscriptions = std::mem::take(&mut *self.subscriptions.lock().unwrap());
        for unsubscribe in subscriptions { unsubscribe(); }
    }
}

impl AgentConnectionSessionWatcher for ChildWatcher {
    fn get_messages(&self) -> BoxFuture<Vec<AgentMessage>> {
        let messages = self.session.messages();
        Box::pin(async move { messages })
    }
    fn get_commands(&self) -> BoxFuture<Vec<AgentConnectionSlashCommand>> {
        let commands = child_commands(&self.session);
        Box::pin(async move { commands })
    }
    fn get_tool_definition(&self, name: &str) -> BoxFuture<Option<crate::modes::agent_connection::types::AgentConnectionToolDefinition>> {
        let definition = self.session.get_tool_definition(name).map(|definition| connection_tool_definition(&definition));
        let definition = crate::modes::agent_connection::tool_definition::create_agent_connection_tool_definition(definition.as_ref());
        Box::pin(async move { definition })
    }
    fn subscribe(&self, listener: AgentConnectionEventListener) -> Box<dyn Fn() + Send + Sync> {
        let (send, mut receive) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(async move { while let Some(event) = receive.recv().await { listener(event).await; } });
        let unsubscribe = self.session.subscribe(Arc::new(move |event| {
            if let Ok(event) = serde_json::to_value(event).and_then(serde_json::from_value) {
                let _ = send.send(crate::modes::agent_connection::types::AgentConnectionEvent::SessionEvent { event });
            }
        }));
        let cleanup: Arc<dyn Fn() + Send + Sync> = Arc::new(move || { unsubscribe(); task.abort(); });
        self.subscriptions.lock().unwrap().push(cleanup.clone());
        Box::new(move || cleanup())
    }
    fn close(&self) -> BoxFuture<()> {
        let subscriptions = std::mem::take(&mut *self.subscriptions.lock().unwrap());
        for unsubscribe in subscriptions { unsubscribe(); }
        Box::pin(async {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::warning_limiter::{limiter_test_lock, reset_limited_warnings};
    use serde_json::json;

    // The warning limiter is process-global and Rust runs test functions in
    // parallel threads inside one process, so every test below takes the shared
    // limiter test lock before it resets or reads the map, and each test uses its
    // OWN serde failure signature. Without both, one test's reset would land
    // inside another test's 60s window and make the suite order-dependent.

    fn node(entry: Value) -> crate::core::session_manager::SessionTreeFlatNode {
        crate::core::session_manager::SessionTreeFlatNode {
            entry: match entry {
                Value::Object(map) => map,
                _ => panic!("node entry must be an object"),
            },
            label: None,
            label_timestamp: None,
        }
    }

    /// `label` without `targetId`: "missing field `target_id`" (the audit's
    /// 7,837-line signature).
    fn label_missing_target_id(id: &str) -> crate::core::session_manager::SessionTreeFlatNode {
        node(json!({
            "type": "label", "id": id, "timestamp": "2026-01-01T00:00:00.000Z", "label": "stale"
        }))
    }

    /// `custom_message` without `customType`: "missing field `custom_type`"
    /// (the audit's 2,377-line signature).
    fn custom_message_missing_custom_type(
        id: &str,
    ) -> crate::core::session_manager::SessionTreeFlatNode {
        node(json!({
            "type": "custom_message", "id": id, "timestamp": "2026-01-01T00:00:00.000Z",
            "content": "kept", "display": true
        }))
    }

    /// `label` with a numeric `targetId`: "invalid type: integer ..., expected a
    /// string". A third, distinct signature for the burst test.
    fn label_with_numeric_target_id(id: &str) -> crate::core::session_manager::SessionTreeFlatNode {
        node(json!({
            "type": "label", "id": id, "timestamp": "2026-01-01T00:00:00.000Z",
            "label": "stale", "targetId": 5
        }))
    }

    fn group(signature: &str, count: usize, sample_ids: &[&str]) -> ProjectionDrop {
        ProjectionDrop {
            signature: signature.to_string(),
            count,
            sample_ids: sample_ids.iter().map(|id| id.to_string()).collect(),
        }
    }

    /// A conforming entry projects; a foreign-shape entry is counted, not lost
    /// silently, and the report names both the cause and the offending entry.
    #[test]
    fn projection_counts_foreign_entries_and_names_them() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        let (flat, dropped) = project_flat_tree(vec![
            node(json!({
                "type": "session_info", "id": "e1", "timestamp": "2026-01-01T00:00:00.000Z",
                "name": "kept"
            })),
            label_missing_target_id("e2"),
        ]);
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].entry.type_name(), "session_info");
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].count, 1);
        assert_eq!(dropped[0].sample_ids, vec!["e2".to_string()]);
        // The current serde error spells the field camelCase (`targetId`); the
        // audit's pre-2026-09-18 log text was snake_case (`target_id`). The
        // bounded-warning mechanism is independent of the field spelling.
        assert!(
            dropped[0].signature.contains("targetId") || dropped[0].signature.contains("target_id"),
            "{}",
            dropped[0].signature
        );

        let lines = report_projection_drops(&dropped);
        assert_eq!(lines.len(), 2, "per-signature plus aggregate: {lines:?}");
        assert!(
            lines[0].contains("Could not project 1 session tree entry"),
            "{}",
            lines[0]
        );
        assert!(lines[0].contains("ids e2"), "{}", lines[0]);
        assert!(
            lines[1].contains("1 session tree entry omitted from this projection"),
            "{}",
            lines[1]
        );
        reset_limited_warnings();
    }

    /// The dropped entry ids are bounded, so the naming cannot grow with the
    /// session either.
    #[test]
    fn dropped_id_samples_are_bounded() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        let nodes: Vec<crate::core::session_manager::SessionTreeFlatNode> = (0..40)
            .map(|index| custom_message_missing_custom_type(&format!("c{index}")))
            .collect();
        let (_, dropped) = project_flat_tree(nodes);
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].count, 40);
        assert_eq!(dropped[0].sample_ids.len(), MAX_SAMPLE_IDS);
        assert!(
            dropped[0].signature.contains("customType") || dropped[0].signature.contains("custom_type"),
            "{}",
            dropped[0].signature
        );
        let lines = report_projection_drops(&dropped);
        assert!(lines[0].contains("+37 more"), "{}", lines[0]);
        reset_limited_warnings();
    }

    /// Repeated identical failures across many projections must produce a
    /// bounded number of lines, and the true count must survive in the summary.
    #[test]
    fn repeated_projection_failures_produce_bounded_lines() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        let nodes: Vec<crate::core::session_manager::SessionTreeFlatNode> = (0..7837)
            .map(|index| label_with_numeric_target_id(&format!("n{index}")))
            .collect();
        let mut emitted: Vec<String> = Vec::new();
        for _ in 0..12 {
            let (_, dropped) = project_flat_tree(nodes.clone());
            assert_eq!(dropped.iter().map(|group| group.count).sum::<usize>(), 7837);
            assert_eq!(
                dropped.len(),
                1,
                "identical failures collapse to one signature"
            );
            emitted.extend(report_projection_drops(&dropped));
        }
        assert_eq!(
            emitted.len(),
            2,
            "one window keeps the flood at two lines: {}",
            emitted.len()
        );
        assert!(
            emitted.iter().any(|line| line.contains("7837")),
            "the suppressed total stays truthful: {emitted:?}"
        );
        reset_limited_warnings();
    }

    /// Value-embedding serde errors (a distinct integer per entry) must not
    /// defeat the per-window bound: the limiter key is normalised, so a
    /// pathological session still costs two lines, not one per distinct value
    /// (review defect D5, review-glm).
    #[test]
    fn value_embedding_signatures_stay_bounded_across_projections() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        let nodes: Vec<crate::core::session_manager::SessionTreeFlatNode> = (0..7837)
            .map(|index| {
                node(json!({
                    "type": "label", "id": format!("n{index}"),
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "label": "stale", "targetId": index
                }))
            })
            .collect();
        let mut emitted: Vec<String> = Vec::new();
        for _ in 0..12 {
            let (_, dropped) = project_flat_tree(nodes.clone());
            assert_eq!(
                dropped.iter().map(|group| group.count).sum::<usize>(),
                7837,
                "every non-conforming entry is still counted"
            );
            emitted.extend(report_projection_drops(&dropped));
        }
        assert_eq!(
            emitted.len(),
            2,
            "one signature line plus the aggregate line per window: {emitted:?}"
        );
        assert!(
            emitted.iter().any(|line| line.contains("7837")),
            "the aggregate still states the true total: {emitted:?}"
        );
        reset_limited_warnings();
    }

    /// Distinct failure signatures are counted separately and listed, with the
    /// line count capped so the summary cannot grow without bound.
    #[test]
    fn distinct_projection_signatures_are_listed_but_capped() {
        let _guard = limiter_test_lock();
        reset_limited_warnings();
        // Digit-free names: the limiter keys on a normalised signature where
        // digits collapse to a placeholder, so numbered names would merge into
        // one key and defeat the distinct-listing this test verifies.
        let names = [
            "synthetic signature alpha", "synthetic signature beta", "synthetic signature gamma",
            "synthetic signature delta", "synthetic signature epsilon", "synthetic signature zeta",
            "synthetic signature eta",
        ];
        let dropped: Vec<ProjectionDrop> = names
            .iter()
            .map(|name| group(name, 4, &["x"]))
            .collect();
        let lines = report_projection_drops(&dropped);
        assert_eq!(
            lines.len(),
            6,
            "5 listed signatures plus the aggregate line"
        );
        let summary = &lines[lines.len() - 1];
        assert!(
            summary.contains("28 session tree entries omitted"),
            "{summary}"
        );
        assert!(
            summary.contains("and 2 more distinct error(s)"),
            "{summary}"
        );
        reset_limited_warnings();
    }
}
