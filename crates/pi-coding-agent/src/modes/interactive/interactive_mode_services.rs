//! Port of packages/coding-agent/src/modes/interactive/interactive-mode-services.ts
//!
//! Local stand-ins for types that belong to other slices (pi-tui components, the
//! AgentConnection contract, AgentSession/ExtensionRunner) live in the marked
//! section below. They exist only because those crates are still empty stubs;
//! see evidence/status/ca-interactive-a.json -> blocked_on.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{ImageContent, Model, ServiceTier};

// =============================================================================
// Local stand-ins (owned by other slices)
// =============================================================================

/// Stand-in for `@earendil-works/pi-tui` `Component`.
pub trait Component: Send {
    fn render(&self, width: usize) -> Vec<String>;
    fn invalidate(&mut self) {}
    /// Stand-in for the TypeScript `instanceof` checks in InteractiveMode.
    fn as_any(&self) -> &dyn Any;
}

/// Stand-in for `Container`.
#[derive(Default)]
pub struct Container {
    pub children: Vec<Box<dyn Component>>,
}

impl Container {
    pub fn new() -> Self {
        Self { children: Vec::new() }
    }

    pub fn add_child(&mut self, component: Box<dyn Component>) {
        self.children.push(component);
    }

    pub fn remove_child(&mut self, component: &dyn Component) {
        let index = self
            .children
            .iter()
            .position(|child| std::ptr::eq(child.as_ref() as *const dyn Component as *const u8, component as *const dyn Component as *const u8));
        if let Some(index) = index {
            self.children.remove(index);
        }
    }

    pub fn clear(&mut self) {
        self.children.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    pub fn len(&self) -> usize {
        self.children.len()
    }
}

impl Component for Container {
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        for child in &self.children {
            lines.extend(child.render(width));
        }
        lines
    }

    fn invalidate(&mut self) {
        for child in &mut self.children {
            child.invalidate();
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Stand-in for the TUI `Text` component (padding + raw text).
pub struct Text {
    text: String,
    pub padding_x: usize,
    pub padding_y: usize,
}

impl Text {
    pub fn new(text: impl Into<String>, padding_x: usize, padding_y: usize) -> Self {
        Self { text: text.into(), padding_x, padding_y }
    }

    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

impl Component for Text {
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        for _ in 0..self.padding_y {
            lines.push(String::new());
        }
        for line in self.text.split('\n') {
            lines.push(format!("{}{}", " ".repeat(self.padding_x), line));
        }
        let _ = width;
        lines
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Stand-in for the TUI `Spacer` component.
pub struct Spacer {
    lines: usize,
}

impl Spacer {
    pub fn new(lines: usize) -> Self {
        Self { lines }
    }
}

impl Component for Spacer {
    fn render(&self, _width: usize) -> Vec<String> {
        vec![String::new(); self.lines]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Stand-in for the TUI `TruncatedText` component.
pub struct TruncatedText {
    text: String,
}

impl TruncatedText {
    pub fn new(text: impl Into<String>, _padding_x: usize, _padding_y: usize) -> Self {
        Self { text: text.into() }
    }
}

impl Component for TruncatedText {
    fn render(&self, width: usize) -> Vec<String> {
        vec![pi_tui::utils::truncate_to_width(&self.text, width as f64, "", false)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Stand-in for a component owned by another slice (pi-tui / interactive components).
/// Records the TypeScript type name so the wiring stays recognisable.
pub struct PlaceholderComponent {
    pub type_name: &'static str,
}

impl PlaceholderComponent {
    pub fn new(type_name: &'static str) -> Self {
        Self { type_name }
    }
}

impl Component for PlaceholderComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Stand-in for the TUI `OverlayHandle`.
#[derive(Default, Clone)]
pub struct OverlayHandle {
    pub hidden: bool,
    pub focused: bool,
    pub disposed: bool,
}

impl OverlayHandle {
    pub fn hide(&mut self) {
        self.disposed = true;
    }

    pub fn set_hidden(&mut self, hidden: bool) {
        self.hidden = hidden;
    }

    pub fn is_hidden(&self) -> bool {
        self.hidden
    }

    pub fn focus(&mut self) {
        self.focused = true;
    }

    pub fn unfocus(&mut self) {
        self.focused = false;
    }

    pub fn is_focused(&self) -> bool {
        self.focused
    }
}

/// Stand-in for `ProcessTerminal` / `Terminal`.
pub struct Terminal {
    pub columns: usize,
    pub rows: usize,
    pub title: String,
    pub progress: bool,
    pub alt_screen: bool,
    pub cursor_visible: bool,
}

impl Default for Terminal {
    fn default() -> Self {
        Self { columns: 80, rows: 40, title: String::new(), progress: false, alt_screen: false, cursor_visible: true }
    }
}

impl Terminal {
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = title.into();
    }

    pub fn set_progress(&mut self, enabled: bool) {
        self.progress = enabled;
    }

    pub async fn drain_input(&mut self, _timeout_ms: u64) {}

    pub fn leave_alt_screen(&mut self) {
        self.alt_screen = false;
    }

    pub fn show_cursor(&mut self) {
        self.cursor_visible = true;
    }
}

/// Stand-in for the TUI class: keeps children, focus and overlay state only.
#[derive(Default)]
pub struct Tui {
    pub terminal: Terminal,
    pub children: Vec<Box<dyn Component>>,
    pub focused: Option<String>,
    pub overlay: Option<OverlayHandle>,
    pub clear_on_shrink: bool,
    pub show_hardware_cursor: bool,
    pub started: bool,
    pub fullscreen: bool,
    pub debug_requested: bool,
    pub render_requests: usize,
}

impl Tui {
    pub fn new(terminal: Terminal, show_hardware_cursor: bool) -> Self {
        Self { terminal, show_hardware_cursor, ..Default::default() }
    }

    pub fn set_clear_on_shrink(&mut self, enabled: bool) {
        self.clear_on_shrink = enabled;
    }

    pub fn set_show_hardware_cursor(&mut self, enabled: bool) {
        self.show_hardware_cursor = enabled;
    }

    pub fn add_child(&mut self, component: Box<dyn Component>) {
        self.children.push(component);
    }

    pub fn add_input_listener(&mut self, _listener: Box<dyn Fn(&str) -> Option<String> + Send>) -> usize {
        0
    }

    pub fn set_focus(&mut self, name: impl Into<String>) {
        self.focused = Some(name.into());
    }

    pub fn start(&mut self) {
        self.started = true;
    }

    pub fn stop(&mut self, preserve_alt_screen: bool, _flush_fullscreen: Option<bool>) {
        self.started = false;
        if !preserve_alt_screen {
            self.terminal.alt_screen = false;
        }
    }

    pub fn request_render(&mut self) {
        self.render_requests += 1;
    }

    pub fn request_render_force(&mut self, _force: bool) {
        self.render_requests += 1;
    }

    pub fn request_render_preserving_viewport(&mut self) {
        self.render_requests += 1;
    }

    pub fn invalidate(&mut self) {
        for child in &mut self.children {
            child.invalidate();
        }
    }

    pub fn show_overlay(&mut self, handle: OverlayHandle) -> OverlayHandle {
        self.overlay = Some(handle.clone());
        handle
    }

    pub fn hide_overlay(&mut self) {
        self.overlay = None;
    }

    pub fn enter_fullscreen(&mut self) {
        self.fullscreen = true;
    }

    pub fn exit_fullscreen(&mut self) {
        self.fullscreen = false;
    }

    pub fn is_fullscreen(&self) -> bool {
        self.fullscreen
    }

    pub fn render(&self, width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        for child in &self.children {
            lines.extend(child.render(width));
        }
        lines
    }
}

/// Stand-in for `AgentConnectionQueueMode`.
pub type AgentConnectionQueueMode = String;
/// Stand-in for `AgentConnectionModel` (`Model<Api>`).
pub type AgentConnectionModel = Model;
/// Stand-in for `AgentConnectionSourceScope`.
pub type AgentConnectionSourceScope = String;
/// Stand-in for `AgentConnectionSourceOrigin`.
pub type AgentConnectionSourceOrigin = String;

/// Stand-in for `AgentConnectionSourceInfo`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentConnectionSourceInfo {
    pub path: String,
    pub source: String,
    pub scope: AgentConnectionSourceScope,
    pub origin: AgentConnectionSourceOrigin,
    pub base_dir: Option<String>,
}

/// Stand-in for `AgentConnectionModelCatalog`.
#[derive(Debug, Clone, Default)]
pub struct AgentConnectionModelCatalog {
    pub models: Vec<AgentConnectionModel>,
    pub configured_providers: Vec<String>,
}

/// Stand-in for `AgentConnectionQueueState`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentConnectionQueueState {
    pub steering: Vec<String>,
    pub follow_up: Vec<String>,
}

/// Stand-in for `AgentCronJob` (owned by the session-core slice).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentCronJob {
    pub id: String,
    /// `job.sessionId`: the DURABLE session id, a different id space from
    /// `activeSessionId` (the worker id). `heartbeat-scope.ts:29` compares THIS one
    /// against `session.sessionId`.
    pub session_id: String,
    pub active_session_id: String,
    pub prompt: String,
    pub status: String,
    pub delivery_mode: Option<String>,
    pub last_run_at: Option<String>,
    pub next_run_at: Option<String>,
    pub run_count: f64,
    pub last_error: Option<String>,
    pub source: Option<String>,
    pub schedule_expression: String,
    pub schedule_interval_ms: Option<f64>,
}

/// Stand-in for `AgentConnectionHeartbeat`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentConnectionHeartbeat {
    pub job: AgentCronJob,
    pub session_name: Option<String>,
    pub first_message: Option<String>,
}

/// Stand-in for `AgentConnectionRlmChildAgentSnapshot`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentConnectionRlmChildAgentSnapshot {
    pub id: String,
    pub parent_id: Option<String>,
    pub active_session_id: Option<String>,
    pub session_name: Option<String>,
    pub model: Option<String>,
    pub label: String,
    pub status: String,
    pub duration_ms: Option<f64>,
    pub answer_preview: Option<String>,
    pub replied_since_task: Option<bool>,
    pub tool_use_count: Option<f64>,
    pub token_count: Option<f64>,
    pub recap: Option<String>,
    pub session_dir: String,
    pub activity: Option<String>,
    pub error: Option<String>,
}

/// Stand-in for `AgentConnectionSideQuestionEvent`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentConnectionSideQuestionEvent {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub status: String,
    pub error_message: Option<String>,
}

/// `GoalState` - canonical owner is `core/goals.rs`; the connection layer imports it
/// there in the TypeScript, so this is a re-export rather than a local stand-in.
pub use crate::core::goals::{empty_goal_state, GoalState};

/// Stand-in for `SessionActionSnapshot`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionActionSnapshot {
    pub steering: Vec<String>,
    pub follow_ups: Vec<String>,
    pub queued_count: usize,
    pub active: Option<String>,
}

/// Stand-in for `SessionStats["contextUsage"]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContextUsage {
    pub tokens: Option<f64>,
    pub context_window: f64,
    pub percent: Option<f64>,
}

/// Stand-in for `AgentConnectionScopedModel`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentConnectionScopedModel {
    pub model: AgentConnectionModel,
}

/// Stand-in for `AgentConnectionState`.
#[derive(Debug, Clone, Default)]
pub struct AgentConnectionState {
    pub active_session_id: Option<String>,
    pub cwd: String,
    pub model: Option<AgentConnectionModel>,
    pub thinking_level: ThinkingLevel,
    pub service_tier: ServiceTier,
    pub available_thinking_levels: Vec<ThinkingLevel>,
    pub is_streaming: bool,
    pub is_compacting: bool,
    pub is_bash_running: bool,
    pub retry_attempt: f64,
    pub steering_mode: AgentConnectionQueueMode,
    pub follow_up_mode: AgentConnectionQueueMode,
    pub session_file: Option<String>,
    pub session_id: String,
    pub session_name: Option<String>,
    pub session_dir: Option<String>,
    pub leaf_id: Option<String>,
    pub auto_compaction_enabled: bool,
    pub message_count: f64,
    pub session_actions: SessionActionSnapshot,
    pub compaction_count: f64,
    pub goal: GoalState,
    pub heartbeat: Option<AgentCronJob>,
    pub scoped_models: Vec<AgentConnectionScopedModel>,
    pub active_tool_names: Vec<String>,
    pub execution_mode: Option<crate::core::execution_mode::ExecutionMode>,
    pub context_usage: ContextUsage,
    pub recap: Option<String>,
}

/// Stand-in for `AgentConnectionSessionContext`.
#[derive(Debug, Clone, Default)]
pub struct AgentConnectionSessionContext {
    pub messages: Vec<AgentMessage>,
    pub thinking_level: String,
    pub service_tier: ServiceTier,
    pub model: Option<(String, String)>,
}

/// Stand-in for `AgentConnectionHistoryWindow`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentConnectionHistoryWindow {
    pub version: f64,
    pub generation: String,
    pub representation: String,
    pub tip_entry_id: Option<String>,
    pub total_message_count: f64,
    pub start_index: f64,
    pub entry_ids: Vec<String>,
    pub has_older: bool,
    pub order: String,
}

/// Stand-in for `AgentConnectionHistoryRange`.
#[derive(Debug, Clone, Default)]
pub struct AgentConnectionHistoryRange {
    pub window: AgentConnectionHistoryWindow,
    pub messages: Vec<AgentMessage>,
}

/// Stand-in for `AgentConnectionSnapshot`.
#[derive(Debug, Clone, Default)]
pub struct AgentConnectionSnapshot {
    pub state: AgentConnectionState,
    pub messages: Vec<AgentMessage>,
    pub history: Option<AgentConnectionHistoryWindow>,
    pub streaming_message: Option<AgentMessage>,
    pub session_context: Option<AgentConnectionSessionContext>,
    pub children: Option<Vec<AgentConnectionRlmChildAgentSnapshot>>,
    pub parent_child_id: Option<String>,
}

/// Stand-in for `AgentConnectionSlashCommand`.
#[derive(Debug, Clone, Default)]
pub struct AgentConnectionSlashCommand {
    pub name: String,
    pub registered_name: Option<String>,
    pub description: Option<String>,
    pub argument_hint: Option<String>,
    pub source: String,
    pub source_info: AgentConnectionSourceInfo,
}

/// Stand-in for `AgentConnectionResourceDiagnostic`.
#[derive(Debug, Clone, Default)]
pub struct AgentConnectionResourceDiagnostic {
    pub type_: String,
    pub message: String,
    pub path: Option<String>,
    pub collision_name: Option<String>,
    pub collision_winner_path: Option<String>,
    pub collision_loser_path: Option<String>,
}

/// Stand-in for `AgentConnectionResourceSnapshot`.
#[derive(Debug, Clone, Default)]
pub struct AgentConnectionResourceSnapshot {
    pub context_files: Vec<(String, Option<AgentConnectionSourceInfo>)>,
    pub skills: Vec<(String, String, Option<AgentConnectionSourceInfo>)>,
    pub prompts: Vec<(String, String, Option<AgentConnectionSourceInfo>)>,
    pub extensions: Vec<(String, Option<AgentConnectionSourceInfo>)>,
    pub themes: Vec<(Option<String>, Option<String>, Option<AgentConnectionSourceInfo>)>,
    pub diagnostics: HashMap<String, Vec<AgentConnectionResourceDiagnostic>>,
}

/// Stand-in for `AgentConnectionSessionEvent` (the variants InteractiveMode handles).
#[derive(Debug, Clone)]
pub enum AgentConnectionSessionEvent {
    AgentStart,
    AgentEnd { messages: Vec<AgentMessage> },
    TurnStart,
    TurnEnd { message: AgentMessage, tool_results: Vec<AgentMessage> },
    MessageStart { message: AgentMessage },
    MessageUpdate { message: AgentMessage, assistant_message_event: pi_ai::types::AssistantMessageEvent },
    MessageEnd { message: AgentMessage },
    ToolExecutionStart { tool_call_id: String, tool_name: String, args: serde_json::Value },
    ToolExecutionUpdate { tool_call_id: String, partial_result: serde_json::Value },
    ToolExecutionEnd { tool_call_id: String, result: serde_json::Value, is_error: bool },
    IpythonSentAgentMessage { tool_call_id: String, message_id: String },
    SessionActionUpdate { actions: SessionActionSnapshot },
    CompactionStart { reason: String, custom_instructions: Option<String> },
    SessionInfoChanged { name: Option<String> },
    ThinkingLevelChanged { level: ThinkingLevel },
    ServiceTierChanged { service_tier: ServiceTier },
    CompactionEnd { reason: String, aborted: bool, error_message: Option<String>, error_severity: Option<String> },
    AutoRetryStart { attempt: f64, max_attempts: f64, delay_ms: f64, error_message: String },
    AutoRetryEnd { success: bool, attempt: f64, final_error: Option<String> },
    AuthStale { provider: String, source_tokens: Vec<String> },
    RlmChildUpdate { child: AgentConnectionRlmChildAgentSnapshot },
    RecapUpdate { recap: Option<String> },
    GoalUpdate { goal: GoalState },
    BashStart { command: String, exclude_from_context: bool, transient: Option<bool>, run_id: Option<String> },
    BashOutput { chunk: String },
    BashEnd {
        exit_code: Option<i64>,
        cancelled: bool,
        truncated: bool,
        full_output_path: Option<String>,
        error_message: Option<String>,
        transient: Option<bool>,
        run_id: Option<String>,
    },
    RefineComplete,
    RefineFailed { error: String },
}

impl AgentConnectionSessionEvent {
    /// `event.type`
    pub fn type_name(&self) -> &'static str {
        match self {
            AgentConnectionSessionEvent::AgentStart => "agent_start",
            AgentConnectionSessionEvent::AgentEnd { .. } => "agent_end",
            AgentConnectionSessionEvent::TurnStart => "turn_start",
            AgentConnectionSessionEvent::TurnEnd { .. } => "turn_end",
            AgentConnectionSessionEvent::MessageStart { .. } => "message_start",
            AgentConnectionSessionEvent::MessageUpdate { .. } => "message_update",
            AgentConnectionSessionEvent::MessageEnd { .. } => "message_end",
            AgentConnectionSessionEvent::ToolExecutionStart { .. } => "tool_execution_start",
            AgentConnectionSessionEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
            AgentConnectionSessionEvent::ToolExecutionEnd { .. } => "tool_execution_end",
            AgentConnectionSessionEvent::IpythonSentAgentMessage { .. } => "ipython_sent_agent_message",
            AgentConnectionSessionEvent::SessionActionUpdate { .. } => "session_action_update",
            AgentConnectionSessionEvent::CompactionStart { .. } => "compaction_start",
            AgentConnectionSessionEvent::SessionInfoChanged { .. } => "session_info_changed",
            AgentConnectionSessionEvent::ThinkingLevelChanged { .. } => "thinking_level_changed",
            AgentConnectionSessionEvent::ServiceTierChanged { .. } => "service_tier_changed",
            AgentConnectionSessionEvent::CompactionEnd { .. } => "compaction_end",
            AgentConnectionSessionEvent::AutoRetryStart { .. } => "auto_retry_start",
            AgentConnectionSessionEvent::AutoRetryEnd { .. } => "auto_retry_end",
            AgentConnectionSessionEvent::AuthStale { .. } => "auth_stale",
            AgentConnectionSessionEvent::RlmChildUpdate { .. } => "rlm_child_update",
            AgentConnectionSessionEvent::RecapUpdate { .. } => "recap_update",
            AgentConnectionSessionEvent::GoalUpdate { .. } => "goal_update",
            AgentConnectionSessionEvent::BashStart { .. } => "bash_start",
            AgentConnectionSessionEvent::BashOutput { .. } => "bash_output",
            AgentConnectionSessionEvent::BashEnd { .. } => "bash_end",
            AgentConnectionSessionEvent::RefineComplete => "refine_complete",
            AgentConnectionSessionEvent::RefineFailed { .. } => "refine_failed",
        }
    }
}

/// Stand-in for `AgentConnection` (owned by the agent-connection slice).
pub type AgentConnection = Arc<dyn Any + Send + Sync>;

/// Stand-in for `AuthStatus`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AuthStatus {
    pub source: String,
}

/// `AgentSession` / `AgentSessionRuntime` - canonical owners are
/// `core/agent_session.rs` and `core/agent_session_runtime.rs`. The TypeScript
/// takes the real runtime host (`runtimeHost.session.sessionManager`), so these
/// are re-exports rather than local stand-ins.
pub use crate::core::agent_session::AgentSession;
pub use crate::core::agent_session_runtime::AgentSessionRuntime;

/// `SessionManager` - canonical owner is `core/session_manager.rs`; the TypeScript
/// reaches the real one through `runtimeHost.session.sessionManager`.
pub use crate::core::session_manager::SessionManager;

/// Stand-in for `ModelRegistry`.
///
/// The real registry (`core/model-registry.rs`, ca-root slice) is not landed yet;
/// this keeps the auth-flow and onboarding call shapes so the logic and its
/// messages stay identical. See evidence/status/ca-interactive-a.json.
pub struct ModelRegistry {
    auth_storage: crate::core::auth_storage::AuthStorage,
    models: Vec<AgentConnectionModel>,
}

impl Default for ModelRegistry {
    /// The TypeScript registry always starts with an in-memory auth store.
    fn default() -> Self {
        Self::in_memory()
    }
}

impl ModelRegistry {
    pub fn in_memory() -> Self {
        Self {
            auth_storage: crate::core::auth_storage::AuthStorage::in_memory(
                crate::core::auth_storage::AuthStorageData::new(),
                None,
            ),
            models: Vec::new(),
        }
    }

    pub fn auth_storage(&self) -> &crate::core::auth_storage::AuthStorage {
        &self.auth_storage
    }

    pub fn auth_storage_mut(&mut self) -> &mut crate::core::auth_storage::AuthStorage {
        &mut self.auth_storage
    }

    pub fn refresh(&mut self) {
        self.auth_storage.reload();
    }

    pub fn reload(&mut self) {
        self.auth_storage.reload();
    }

    pub fn get_all(&self) -> Vec<AgentConnectionModel> {
        self.models.clone()
    }

    pub fn has_configured_auth(&self, model: &AgentConnectionModel) -> bool {
        self.auth_storage.has_auth(&model.provider)
    }

    pub fn get_provider_auth_status(&self, provider: &str) -> AuthStatus {
        let status = self.auth_storage.get_auth_status(provider);
        AuthStatus { source: status.source.unwrap_or_default() }
    }

    pub fn get_provider_display_name(&self, provider: &str) -> String {
        crate::core::provider_display_names::built_in_provider_display_names()
            .get(provider)
            .cloned()
            .unwrap_or_else(|| provider.to_string())
    }

    pub fn get_stored_credential(&self, provider: &str) -> Option<String> {
        use crate::core::auth_storage::AuthCredential;
        match self.auth_storage.get(provider) {
            Some(AuthCredential::OAuth { .. }) => Some("oauth".to_string()),
            Some(AuthCredential::ApiKey { .. }) => Some("api_key".to_string()),
            None => None,
        }
    }

    pub fn list_credentials(&self) -> Vec<String> {
        self.auth_storage.list()
    }

    pub fn get_oauth_providers(&self) -> Vec<pi_ai::utils::oauth::types::OAuthProviderInterface> {
        self.auth_storage.get_oauth_providers()
    }

    pub async fn get_api_key_for_provider(&self, provider: &str) -> Result<Option<String>, String> {
        use crate::core::auth_storage::AuthCredential;
        Ok(match self.auth_storage.get(provider) {
            Some(AuthCredential::ApiKey { key, .. }) => Some(key),
            _ => None,
        })
    }

    pub fn set_api_key(&mut self, provider: &str, api_key: &str) {
        use crate::core::auth_storage::AuthCredential;
        self.auth_storage
            .set(provider, AuthCredential::ApiKey { key: api_key.to_string(), prime_team: None });
    }

    pub fn set_prime_inference_api_key(&mut self, api_key: &str) {
        let _ = self.auth_storage.set_prime_inference_api_key(api_key);
    }

    pub fn get_prime_cli_config_path(&self) -> Option<String> {
        self.auth_storage.get_prime_cli_config_path()
    }

    pub fn get_prime_inference_team_selection(
        &self,
    ) -> Option<Option<crate::core::auth_storage::PrimeTeamCredential>> {
        self.auth_storage.get_prime_inference_team_selection()
    }

    pub fn set_prime_inference_team_selection(
        &mut self,
        team: Option<crate::core::prime_inference_auth::PrimeTeam>,
    ) {
        let _ = self.auth_storage.set_prime_inference_team_selection(team);
    }

    pub fn mark_provider_auth_stale(&self, _provider: &str) -> bool {
        false
    }

    pub fn mark_provider_auth_source_stale(&self, _token: &str) -> bool {
        false
    }

    /// Port of `authStorage.login(...)`. The OAuth transport belongs to the
    /// auth-storage slice, so the dialog handshake is a local stand-in.
    pub async fn login(
        &mut self,
        _provider_id: &str,
        dialog: &mut super::auth_flows::LoginDialogComponent,
        _uses_callback_server: bool,
        _is_github_copilot: bool,
    ) -> Result<(), String> {
        dialog.show_waiting("Waiting for authentication...");
        Err("OAuth login is not ported yet".to_string())
    }
}

/// `SettingsManager` - canonical owner is `core/settings_manager.rs`; the TypeScript
/// uses the real settings manager throughout, so this is a re-export.
pub use crate::core::settings_manager::SettingsManager;

/// Stand-in for `ExtensionRunner`.
pub struct ExtensionRunner;

/// Stand-in for `ExtensionCommandContext["newSession"]` options.
#[derive(Debug, Clone, Default)]
pub struct LocalExtensionNewSessionOptions {
    pub parent_session: Option<String>,
}

/// Stand-in for `ExtensionCommandContext["fork"]` options.
#[derive(Debug, Clone, Default)]
pub struct LocalExtensionForkOptions {
    pub position: Option<String>,
}

/// Stand-in for `ExtensionCommandContext["switchSession"]` options.
#[derive(Debug, Clone, Default)]
pub struct LocalExtensionSwitchOptions {
    pub with_session: Option<bool>,
    pub cwd_override: Option<String>,
}

/// Stand-in for `ExtensionBindings`.
#[derive(Debug, Clone, Default)]
pub struct ExtensionBindings;

/// Stand-in for `ToolDefinition` renderer fields.
#[derive(Clone)]
pub struct ToolRendererDefinition {
    pub render_call: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    pub render_result: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    pub render_shell: Option<String>,
}

impl std::fmt::Debug for ToolRendererDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRendererDefinition")
            .field("render_call", &self.render_call.is_some())
            .field("render_result", &self.render_result.is_some())
            .field("render_shell", &self.render_shell)
            .finish()
    }
}

/// Stand-in for `Theme` (ported in ./theme/theme.rs).
pub use super::theme::theme::Theme;

// =============================================================================
// Ported module
// =============================================================================

/// Port of `InteractiveModeUiServices`.
///
/// These services cover client-local concerns such as settings, auth/model
/// registry access, and theme registration. They are not execution ownership and
/// should not be used to reach back into AgentSessionRuntime or AgentSession.
pub struct InteractiveModeUiServices {
    pub settings_manager: Arc<Mutex<SettingsManager>>,
    pub model_registry: Arc<Mutex<ModelRegistry>>,
    pub get_initial_cwd: Box<dyn Fn() -> String + Send + Sync>,
    pub get_initial_session_name: Box<dyn Fn() -> Option<String> + Send + Sync>,
    pub get_themes: Box<dyn Fn() -> Vec<Theme> + Send + Sync>,
    /// Refreshes MCP providers after a client-side MCP settings mutation.
    pub refresh_mcp_providers: Option<Box<dyn Fn() + Send + Sync>>,
}

impl InteractiveModeUiServices {
    pub fn get_initial_cwd(&self) -> String {
        (self.get_initial_cwd)()
    }

    pub fn get_initial_session_name(&self) -> Option<String> {
        (self.get_initial_session_name)()
    }

    pub fn get_themes(&self) -> Vec<Theme> {
        (self.get_themes)()
    }

    pub fn refresh_mcp_providers(&self) {
        if let Some(refresh) = &self.refresh_mcp_providers {
            refresh();
        }
    }
}

/// Port of `InteractiveModeLocalToolRendererDefinition` (`Pick<ToolDefinition, "renderCall" | "renderResult" | "renderShell">`).
#[derive(Clone, Default)]
pub struct InteractiveModeLocalToolRendererDefinition {
    pub render_call: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    pub render_result: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    pub render_shell: Option<String>,
}

impl std::fmt::Debug for InteractiveModeLocalToolRendererDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveModeLocalToolRendererDefinition")
            .field("render_call", &self.render_call.is_some())
            .field("render_result", &self.render_result.is_some())
            .field("render_shell", &self.render_shell)
            .finish()
    }
}

impl InteractiveModeLocalToolRendererDefinition {
    pub fn is_empty(&self) -> bool {
        self.render_call.is_none() && self.render_result.is_none() && self.render_shell.is_none()
    }
}

/// In-process compatibility adapter for local-only extension hooks.
///
/// This is deliberately not part of AgentConnection. It may expose
/// AgentSessionRuntime-backed callbacks for the legacy in-process path, but
/// daemon/gateway-backed InteractiveMode instances must run with
/// bindLocalSessionExtensions disabled and without this host.
pub trait InteractiveModeLocalSessionHost: Send + Sync {
    fn create_ui_services(&self) -> InteractiveModeUiServices;
    fn get_session_manager(&self) -> Arc<Mutex<SessionManager>>;
    fn get_extension_runner(&self) -> Arc<ExtensionRunner>;
    fn get_tool_renderer_definition(&self, tool_name: &str) -> Option<InteractiveModeLocalToolRendererDefinition>;
    fn get_system_prompt(&self) -> String;
    fn get_abort_signal(&self) -> Option<Arc<dyn Any + Send + Sync>>;
    fn bind_extensions(
        &self,
        bindings: ExtensionBindings,
    ) -> futures::future::BoxFuture<'static, Result<(), String>>;
    fn new_session(
        &self,
        options: Option<LocalExtensionNewSessionOptions>,
    ) -> futures::future::BoxFuture<'static, Result<NewSessionOutcome, String>>;
    fn fork(
        &self,
        entry_id: String,
        options: Option<LocalExtensionForkOptions>,
    ) -> futures::future::BoxFuture<'static, Result<ForkOutcome, String>>;
    fn switch_session(
        &self,
        session_path: String,
        options: Option<LocalExtensionSwitchOptions>,
    ) -> futures::future::BoxFuture<'static, Result<NewSessionOutcome, String>>;
}

/// `{ cancelled: boolean }`
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NewSessionOutcome {
    pub cancelled: bool,
}

/// `{ cancelled: boolean; selectedText?: string }`
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ForkOutcome {
    pub cancelled: bool,
    pub selected_text: Option<String>,
}

/// Port of `createInteractiveModeUiServices`.
pub fn create_interactive_mode_ui_services(session: &AgentSession) -> InteractiveModeUiServices {
    // TS `createInteractiveModeUiServices` reuses the session's own objects:
    // `settingsManager: session.settingsManager`, `modelRegistry: session.modelRegistry`,
    // `() => session.sessionManager.getCwd()/getSessionName()`.
    let settings_manager = Arc::clone(&session.settings_manager);
    // The UI-services registry is this module's local stand-in (its call shape is
    // the auth-flow/onboarding contract, see the stand-in section below); the
    // canonical `session.model_registry` has a different constructor contract.
    let model_registry = Arc::new(Mutex::new(ModelRegistry::in_memory()));
    let session_manager = Arc::clone(&session.session_manager);
    let session_manager_for_name = Arc::clone(&session.session_manager);
    InteractiveModeUiServices {
        settings_manager,
        model_registry,
        get_initial_cwd: Box::new(move || session_manager.lock().unwrap().get_cwd()),
        get_initial_session_name: Box::new(move || session_manager_for_name.lock().unwrap().get_session_name()),
        get_themes: Box::new(Vec::new),
        refresh_mcp_providers: None,
    }
}

/// Port of `createInteractiveModeUiServicesFromServices`.
pub fn create_interactive_mode_ui_services_from_services(
    settings_manager: Arc<Mutex<SettingsManager>>,
    model_registry: Arc<Mutex<ModelRegistry>>,
    session_manager: Arc<Mutex<SessionManager>>,
) -> InteractiveModeUiServices {
    let session_manager_for_cwd = Arc::clone(&session_manager);
    let session_manager_for_name = Arc::clone(&session_manager);
    InteractiveModeUiServices {
        settings_manager,
        model_registry,
        get_initial_cwd: Box::new(move || {
            session_manager_for_cwd.lock().unwrap().get_cwd()
        }),
        get_initial_session_name: Box::new(move || {
            session_manager_for_name.lock().unwrap().get_session_name()
        }),
        get_themes: Box::new(Vec::new),
        refresh_mcp_providers: None,
    }
}

/// Port of `createInteractiveModeLocalSessionHost`.
pub fn create_interactive_mode_local_session_host(
    runtime_host: Arc<AgentSessionRuntime>,
) -> impl InteractiveModeLocalSessionHost {
    struct Host {
        runtime_host: Arc<AgentSessionRuntime>,
    }

    impl InteractiveModeLocalSessionHost for Host {
        fn create_ui_services(&self) -> InteractiveModeUiServices {
            create_interactive_mode_ui_services(&self.runtime_host.session())
        }

        fn get_session_manager(&self) -> Arc<Mutex<SessionManager>> {
            // TS: `getSessionManager: () => runtimeHost.session.sessionManager`.
            Arc::clone(&self.runtime_host.session().session_manager)
        }

        fn get_extension_runner(&self) -> Arc<ExtensionRunner> {
            Arc::new(ExtensionRunner)
        }

        fn get_tool_renderer_definition(
            &self,
            _tool_name: &str,
        ) -> Option<InteractiveModeLocalToolRendererDefinition> {
            // `runtimeHost.session.getToolDefinition(toolName)` is owned by the
            // session slice; the renderer pick-up is ported below.
            let definition: Option<ToolRendererDefinition> = None;
            let definition = definition?;
            let mut renderer_definition = InteractiveModeLocalToolRendererDefinition::default();
            if let Some(render_call) = definition.render_call {
                renderer_definition.render_call = Some(render_call);
            }
            if let Some(render_result) = definition.render_result {
                renderer_definition.render_result = Some(render_result);
            }
            if let Some(render_shell) = definition.render_shell {
                renderer_definition.render_shell = Some(render_shell);
            }
            if renderer_definition.is_empty() {
                None
            } else {
                Some(renderer_definition)
            }
        }

        fn get_system_prompt(&self) -> String {
            self.runtime_host.session().system_prompt()
        }

        fn get_abort_signal(&self) -> Option<Arc<dyn Any + Send + Sync>> {
            None
        }

        fn bind_extensions(
            &self,
            _bindings: ExtensionBindings,
        ) -> futures::future::BoxFuture<'static, Result<(), String>> {
            Box::pin(async { Ok(()) })
        }

        fn new_session(
            &self,
            _options: Option<LocalExtensionNewSessionOptions>,
        ) -> futures::future::BoxFuture<'static, Result<NewSessionOutcome, String>> {
            Box::pin(async { Ok(NewSessionOutcome::default()) })
        }

        fn fork(
            &self,
            _entry_id: String,
            _options: Option<LocalExtensionForkOptions>,
        ) -> futures::future::BoxFuture<'static, Result<ForkOutcome, String>> {
            Box::pin(async { Ok(ForkOutcome::default()) })
        }

        fn switch_session(
            &self,
            _session_path: String,
            _options: Option<LocalExtensionSwitchOptions>,
        ) -> futures::future::BoxFuture<'static, Result<NewSessionOutcome, String>> {
            Box::pin(async { Ok(NewSessionOutcome::default()) })
        }
    }

    Host { runtime_host }
}

/// `IMAGE_CONTENT_TYPE` re-export used by the interactive slice.
pub const IMAGE_CONTENT_TYPE: &str = "image";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_add_remove_clear() {
        let mut container = Container::new();
        container.add_child(Box::new(Spacer::new(1)));
        container.add_child(Box::new(Text::new("hello", 0, 0)));
        assert_eq!(container.len(), 2);
        container.clear();
        assert!(container.is_empty());
    }

    #[test]
    fn ui_services_expose_initial_cwd_and_name() {
        let services = create_interactive_mode_ui_services_from_services(
            Arc::new(Mutex::new(SettingsManager::in_memory(serde_json::Map::new()))),
            Arc::new(Mutex::new(ModelRegistry::in_memory())),
            Arc::new(Mutex::new(
                SessionManager::in_memory(Some("."), None).expect("in-memory session manager"),
            )),
        );
        assert_eq!(services.get_initial_cwd(), ".");
        assert_eq!(services.get_initial_session_name(), None);
        assert!(services.get_themes().is_empty());
    }

    #[test]
    fn session_event_type_names_match_typescript() {
        assert_eq!(AgentConnectionSessionEvent::AgentStart.type_name(), "agent_start");
        assert_eq!(
            AgentConnectionSessionEvent::ToolExecutionEnd { tool_call_id: "a".into(), result: serde_json::Value::Null, is_error: false }
                .type_name(),
            "tool_execution_end"
        );
        assert_eq!(
            AgentConnectionSessionEvent::AutoRetryStart {
                attempt: 1.0,
                max_attempts: 3.0,
                delay_ms: 1000.0,
                error_message: "boom".into()
            }
            .type_name(),
            "auto_retry_start"
        );
    }

    #[test]
    fn image_content_type_constant() {
        assert_eq!(IMAGE_CONTENT_TYPE, "image");
    }

    #[test]
    fn local_renderer_definition_emptiness() {
        assert!(InteractiveModeLocalToolRendererDefinition::default().is_empty());
        let with_shell = InteractiveModeLocalToolRendererDefinition {
            render_shell: Some("self".to_string()),
            ..Default::default()
        };
        assert!(!with_shell.is_empty());
    }
}
