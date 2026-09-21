//! Port of packages/coding-agent/src/modes/agents-view/agents-view-mode.ts
//!
//! Platform adapters in native_host connect the controller to pi-tui, the
//! daemon client, and interactive chat while keeping model behavior testable.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::agents_view_state::{
    build_unified_session_index, collapse_whitespace, compute_recursive_rollups,
    create_unattachable_child_open_result, filter_unified_sessions, format_heartbeat_badge,
    get_agents_view_selection_key, get_agents_view_session_title, get_agents_view_summary_identity,
    get_unified_session_ancestor_session_ids, has_unified_session_children, migrate_agents_view_identity_set,
    parse_js_timestamp, reconcile_unified_sessions, resolve_agents_view_left_result,
    resolve_agents_view_scope_frames, resolve_agents_view_selection_state, scope_to_session_subtree,
    section_title, should_apply_scope_resolution, should_show_agents_view_session, summary_for_unified_record,
    to_iso_string, transition_agents_view_scope, AgentsViewRecursiveRollup, AgentsViewRow, AgentsViewRowInput,
    AgentsViewRowKind, AgentsViewScopeBackResult, AgentsViewScopeFrame, AgentsViewScopeKey,
    AgentsViewSection, AgentsViewSelectionKey, AgentConnectionHeartbeat, AgentConnectionSavedSessionInfo,
    SessionLifecycle, SessionSummary, UnifiedSessionIndex, UnifiedSessionRecord,
    build_agents_view_rows,
};
use super::roster_store::{
    AgentsViewRosterStore, DaemonClientRequestOptions, DaemonResponse, DaemonTransportClient,
    STALE_ROSTER_DAEMON_MESSAGE,
};
use super::session_view_search::matches_search_text;

pub const HEARTBEAT_POLL_INTERVAL_MS: u64 = 15000;
pub const RECONNECT_TIMEOUT_MS: u64 = 120000;
pub const RECONNECT_RETRY_MS: u64 = 1000;
pub const EXIT_HINT_DURATION_MS: u64 = 2000;
pub const DELETE_CONFIRM_DURATION_MS: u64 = 2000;
pub const STATUS_MESSAGE_DURATION_MS: u64 = 4500;
pub const SEARCH_PROMPT_PLACEHOLDER: &str = "Search sessions";
pub const REPLY_PROMPT_FALLBACK_PLACEHOLDER: &str = "Write a reply to this agent";
pub const RESUME_PROMPT_PLACEHOLDER: &str = "Write a prompt to resume this session";
pub const COMPLETED_ROW_ICON: &str = "✓";
pub const NEEDS_INPUT_ROW_ICON: &str = "●";
pub const SELECTED_ROW_MARKER: &str = "\u{0}agents-view-selected-row\u{0}";
pub const CODE_ROW_MARKER: &str = "\u{0}agents-view-code-row\u{0}";

/// `WORKING_ICON_INTERVAL_MS` from modes/interactive/theme/working-icon.ts.
pub const WORKING_ICON_INTERVAL_MS: u64 = 250;
pub const WORKING_ICON_FRAMES: [&str; 4] = ["◇", "◈", "◆", "◈"];

/// The port split the TypeScript `{ sessionId, activeSessionId }` key into a
/// selection key and a scope key with the same shape; `hasUnifiedSessionChildren`
/// takes the scope one.
fn scope_key_from_selection(selection: &AgentsViewSelectionKey) -> AgentsViewScopeKey {
    AgentsViewScopeKey {
        session_id: selection.session_id.clone(),
        active_session_id: selection.active_session_id.clone(),
    }
}

pub fn working_icon_frame(frame: i64) -> &'static str {
    let len = WORKING_ICON_FRAMES.len() as i64;
    let index = ((frame % len) + len) % len;
    WORKING_ICON_FRAMES[index as usize]
}

/// Settings/theme/cwd surface of `InteractiveModeUiServices`.
pub trait AgentsViewUiServices: Send + Sync {
    fn get_initial_cwd(&self) -> String;
    fn get_theme(&self) -> String;
    fn get_themes(&self) -> Vec<String>;
    fn get_show_hardware_cursor(&self) -> bool;
    fn get_clear_on_shrink(&self) -> bool;
    fn get_editor_padding_x(&self) -> usize;
    fn get_autocomplete_max_visible(&self) -> usize;
}

/// `AgentSessionRuntimeConfig` fields this mode reads or copies.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AgentsViewRuntimeConfig {
    pub cwd: Option<String>,
    pub session_dir: Option<String>,
    pub telemetry_disabled: Option<bool>,
}

#[derive(Clone)]
pub struct AgentsViewModeOptions {
    pub socket_path: Option<String>,
    pub config: AgentsViewRuntimeConfig,
    pub ui_services: Arc<dyn AgentsViewUiServices>,
    pub migrated_providers: Option<Vec<String>>,
    pub model_fallback_message: Option<String>,
    pub startup_model_id: Option<String>,
    pub verbose: Option<bool>,
    pub reconnect_timeout_ms: Option<u64>,
    pub initial_session: Option<SessionSummary>,
    /// When set, the first view is rooted at this session's direct children.
    pub initial_scope_key: Option<AgentsViewScopeKey>,
}

/// `StartupNotices` (modes/shared/startup-notices.ts).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StartupNotices {
    pub new_version: Option<String>,
    pub package_updates: Vec<String>,
    pub tmux_warning: Option<String>,
}

pub const PACKAGE_UPDATE_NOTICE_PREFIX: &str = "Package updates available: ";
pub const UPDATE_AVAILABLE_NOTICE_PREFIX: &str = "Update available: ";
pub const TMUX_WARNING_NOTICE_PREFIX: &str = "tmux: ";

pub fn format_update_available_notice(version: &str) -> String {
    format!("{UPDATE_AVAILABLE_NOTICE_PREFIX}{version}")
}

pub fn format_package_update_notice(packages: &[String]) -> String {
    format!("{PACKAGE_UPDATE_NOTICE_PREFIX}{}", packages.join(", "))
}

pub fn format_tmux_warning_notice(warning: &str) -> String {
    format!("{TMUX_WARNING_NOTICE_PREFIX}{warning}")
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentsViewRunResult {
    Exit,
    ScopeBack {
        selection: SessionSummary,
        expanded_ancestor_session_ids: Vec<String>,
        return_chat: Option<SessionSummary>,
        has_children: bool,
    },
    Open {
        summary: SessionSummary,
        /// Row restored after chat closes; differs from summary only for an
        /// unattachable-child fallback.
        selection: Option<SessionSummary>,
        expanded_ancestor_session_ids: Option<Vec<String>>,
        has_children: Option<bool>,
        status_message: Option<String>,
    },
}

/// `AgentsViewPersistentState`.
#[derive(Clone, Debug, Default)]
pub struct AgentsViewPersistentState {
    pub selected_row_identity: Option<String>,
    pub back_session: Option<SessionSummary>,
    pub scope_frames: Option<Vec<AgentsViewScopeFrame>>,
    pub scope_root_summary: Option<SessionSummary>,
    pub selected_session_key: Option<AgentsViewSelectionKey>,
    /// Ancestor chain to re-expand on return to a nested agent. Kept by sessionId,
    /// not row identity, so it survives an active→persisted identity flip.
    pub pending_expanded_ancestor_session_ids: Option<Vec<String>>,
    pub expanded_subagent_parents: Option<HashSet<String>>,
    pub program_shown_parents: Option<HashSet<String>>,
    pub inactive_expanded: Option<bool>,
    /// Distinguish a deliberate collapse from the legacy collapsed-by-default state.
    pub inactive_visibility_explicit: Option<bool>,
    pub status_message: Option<String>,
    pub startup_notices: Option<StartupNotices>,
    pub query: Option<String>,
    /// Reused across agents-view instances (`persistentState.rosterClient`).
    pub roster_client: Option<DaemonTransportClient>,
    pub saved_sessions: Option<Vec<AgentConnectionSavedSessionInfo>>,
    pub last_successful_saved_sessions: Option<Vec<AgentConnectionSavedSessionInfo>>,
    pub saved_catalog_loaded: Option<bool>,
    pub last_successful_live_summaries: Option<Vec<SessionSummary>>,
    pub saved_catalog_generation: Option<i64>,
    pub heartbeats: Option<Vec<AgentConnectionHeartbeat>>,
}

/// `PromptCommand` = `Extract<DaemonCommand, { type: "prompt" }>`.
pub fn create_prompt_command(
    active_session_id: &str,
    message: &str,
    streaming_behavior: Option<&str>,
) -> Value {
    let mut command = serde_json::json!({
        "type": "prompt",
        "activeSessionId": active_session_id,
        "message": message,
    });
    if let Some(behavior) = streaming_behavior {
        command["streamingBehavior"] = Value::String(behavior.to_string());
    }
    command
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingDeleteAgent {
    pub identity: String,
    pub active_session_id: Option<String>,
    pub session_file: Option<String>,
    pub summary: SessionSummary,
    pub stopped: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingKillSubagent {
    pub identity: String,
    pub root_active_session_id: String,
    pub child_id: String,
}

pub const AGENTS_VIEW_COMMAND_NAMES: [&str; 2] = ["name", "kill"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentsViewCommandName {
    Name,
    Kill,
}

impl AgentsViewCommandName {
    pub fn as_str(self) -> &'static str {
        match self {
            AgentsViewCommandName::Name => "name",
            AgentsViewCommandName::Kill => "kill",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentsViewCommand {
    pub name: AgentsViewCommandName,
    pub args: String,
}

/// Built-in slash commands this mode maps onto existing RPCs.
pub const BUILTIN_SLASH_COMMAND_NAMES: [&str; 37] = [
    "settings", "model", "effort", "fast", "scoped-models", "export", "import", "share", "copy", "btw",
    "name", "session", "system-prompt", "logs", "traces", "context", "changelog", "update", "hotkeys",
    "fork", "clone", "tree", "login", "logout", "mcp", "new", "compact", "refine", "goal", "autonomous",
    "rlm-max-depth", "heartbeat", "heartbeats", "resume", "reload", "fullscreen", "quit",
];

pub const BUILTIN_SLASH_COMMAND_ALIASES: [(&str, &str); 5] = [
    ("clear", "new"),
    ("usage", "context"),
    ("thinking", "effort"),
    ("rename", "name"),
    ("side", "btw"),
];

pub const SESSION_SLASH_COMMAND_NAMES: [&str; 4] = ["compact", "refine", "goal", "autonomous"];

pub fn is_session_slash_command_name(value: &str) -> bool {
    SESSION_SLASH_COMMAND_NAMES.contains(&value)
}

pub fn resolve_builtin_slash_command_name(name: &str) -> String {
    BUILTIN_SLASH_COMMAND_ALIASES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map(|(_, target)| (*target).to_string())
        .unwrap_or_else(|| name.to_string())
}

pub fn is_builtin_slash_command_name(name: &str) -> bool {
    BUILTIN_SLASH_COMMAND_NAMES.contains(&name)
        || BUILTIN_SLASH_COMMAND_ALIASES.iter().any(|(alias, _)| *alias == name)
}

/// `parseSlashCommand` (core/slash-commands.ts): `^\/(\S+)(?:\s+([\s\S]*))?$`.
pub fn parse_slash_command(text: &str) -> Option<(String, String)> {
    let rest = text.strip_prefix('/')?;
    if rest.is_empty() {
        return None;
    }
    let mut name = String::new();
    let mut args_start = None;
    for (index, ch) in rest.char_indices() {
        if ch.is_whitespace() {
            args_start = Some(index);
            break;
        }
        name.push(ch);
    }
    if name.is_empty() {
        return None;
    }
    let args = match args_start {
        None => String::new(),
        Some(index) => rest[index..].trim().to_string(),
    };
    Some((name, args))
}


/// `parseAgentsViewCommand`.
pub fn parse_agents_view_command(text: &str) -> Option<AgentsViewCommand> {
    let (name, args) = parse_slash_command(text)?;
    let resolved = resolve_builtin_slash_command_name(&name);
    match resolved.as_str() {
        "name" => Some(AgentsViewCommand { name: AgentsViewCommandName::Name, args }),
        "kill" => Some(AgentsViewCommand { name: AgentsViewCommandName::Kill, args }),
        _ => None,
    }
}

/// Reject recognized built-ins that are neither session-owned nor view
/// commands, so they are never sent to the model as plain prompt text.
pub fn get_reply_composer_command_rejection(text: &str) -> Option<String> {
    let (name, _args) = parse_slash_command(text)?;
    let resolved = resolve_builtin_slash_command_name(&name);
    if is_session_slash_command_name(&resolved) {
        return None;
    }
    if AGENTS_VIEW_COMMAND_NAMES.contains(&resolved.as_str()) {
        return None;
    }
    if !is_builtin_slash_command_name(&name) {
        return None;
    }
    Some(format!("/{name} is not available here; open the session to run it"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentsViewSlashCommand {
    pub name: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub argument_hint: Option<String>,
    pub takes_argument: Option<bool>,
}

pub const AGENTS_VIEW_COMMAND_DESCRIPTIONS: [(&str, &str, Option<&str>); 2] = [
    ("name", "Set session display name", Some("<name>")),
    ("kill", "Stop this agent's runtime (session stays resumable)", None),
];

/// `agentsViewSlashCommands`.
pub fn agents_view_slash_commands() -> Vec<AgentsViewSlashCommand> {
    AGENTS_VIEW_COMMAND_NAMES
        .iter()
        .map(|name| {
            let display = AGENTS_VIEW_COMMAND_DESCRIPTIONS
                .iter()
                .find(|(command, _, _)| command == name)
                .expect("description table covers every view command");
            AgentsViewSlashCommand {
                name: (*name).to_string(),
                aliases: Vec::new(),
                description: display.1.to_string(),
                argument_hint: display.2.map(|hint| hint.to_string()),
                takes_argument: if *name == "name" { Some(true) } else { None },
            }
        })
        .collect()
}

/// Session-owned built-ins offered by the reply composer autocomplete.
pub fn session_slash_commands() -> Vec<AgentsViewSlashCommand> {
    SESSION_SLASH_COMMAND_NAMES
        .iter()
        .map(|name| AgentsViewSlashCommand {
            name: (*name).to_string(),
            aliases: Vec::new(),
            description: String::new(),
            argument_hint: None,
            takes_argument: None,
        })
        .collect()
}

/// Autocomplete for the reply composer: session-owned plus view commands.
pub fn create_reply_composer_commands() -> Vec<AgentsViewSlashCommand> {
    let mut commands = session_slash_commands();
    commands.extend(agents_view_slash_commands());
    commands
}

pub fn resolve_current_reply_target_summary(
    records: &[UnifiedSessionRecord],
    target: &(String, SessionSummary),
    find_live: &dyn Fn(&str) -> Option<SessionSummary>,
) -> SessionSummary {
    let identity = get_agents_view_summary_identity(&target.1);
    let current = records
        .iter()
        .find(|record| record.identity == identity || record.identity_aliases.iter().any(|alias| alias == &identity));
    if let Some(record) = current {
        return summary_for_unified_record(record);
    }
    if let Some(active_session_id) = target.1.active_session_id.as_deref() {
        if let Some(live) = find_live(active_session_id) {
            return live;
        }
    }
    // A persisted target missing from the current live catalog can still be
    // resumed from its captured file, but its captured runtime id is stale.
    if target.1.session_file.is_some() && target.1.active_session_id.is_some() {
        let mut summary = target.1.clone();
        summary.active_session_id = None;
        summary.lifecycle = SessionLifecycle::Archived;
        summary.activity = super::agents_view_state::SessionActivity::Idle;
        return summary;
    }
    target.1.clone()
}

pub async fn resolve_agents_view_session_ui_services(
    options: &AgentsViewModeOptions,
    create_ui_services_for_session: Option<&(dyn Fn(&SessionSummary) -> super::agents_view_state::SessionSummary)>,
    summary: &SessionSummary,
) -> Arc<dyn AgentsViewUiServices> {
    match create_ui_services_for_session {
        Some(_) => options.ui_services.clone(),
        None => {
            let _ = summary;
            options.ui_services.clone()
        }
    }
}

/// Stripping cwd opens the session in its own stored directory; overrideCwd is
/// sent when that directory no longer exists so the daemon doesn't reject it.
pub fn create_agents_view_resume_config(
    config: &AgentsViewRuntimeConfig,
    override_cwd: Option<&str>,
) -> AgentsViewRuntimeConfig {
    let mut resume_config = config.clone();
    match override_cwd {
        Some(cwd) => resume_config.cwd = Some(cwd.to_string()),
        None => resume_config.cwd = None,
    }
    resume_config
}

pub fn create_agents_view_list_command() -> Value {
    // Omitting `all` returns daemon-resident sessions only; on-disk ones come back
    // through the view's saved-session catalog.
    serde_json::json!({ "type": "list" })
}

pub fn resolve_agents_view_active_summary_for_path(
    session_path: &str,
    summaries: &[SessionSummary],
) -> Option<SessionSummary> {
    let selected_path = crate::utils::paths::resolve_path(&crate::utils::paths::canonicalize_path(session_path));
    summaries
        .iter()
        .find(|summary| {
            summary.active_session_id.is_some()
                && summary
                    .session_file
                    .as_deref()
                    .map(|file| {
                        crate::utils::paths::resolve_path(&crate::utils::paths::canonicalize_path(file))
                            == selected_path
                    })
                    .unwrap_or(false)
        })
        .cloned()
}

// Status messages render in a single-row hint slot below the editor; embedded
// newlines would make that row taller than the layout accounts for and overlap
// the input, so flatten all whitespace runs to single spaces.
pub fn format_agents_view_status_line(text: &str) -> String {
    collapse_whitespace(text)
}

pub fn combine_agents_view_startup_notices(notices: &[Option<&str>]) -> Option<String> {
    let formatted: Vec<String> = notices
        .iter()
        .flatten()
        .map(|notice| format_agents_view_status_line(notice))
        .filter(|notice| !notice.is_empty())
        .collect();
    if formatted.is_empty() {
        None
    } else {
        Some(formatted.join(" · "))
    }
}

pub fn should_reconnect_agents_view_daemon(reason: Option<&str>) -> bool {
    reason != Some("shutdown")
}

pub fn create_agents_view_reply_headline(text: Option<&str>) -> Option<String> {
    text?.split('\n')
        .map(collapse_whitespace)
        .find(|line| !line.is_empty())
}

pub fn get_agents_view_depth(scope_root: Option<&SessionSummary>) -> i64 {
    match scope_root {
        Some(root) => root.rlm_depth.unwrap_or(0) + 1,
        None => 0,
    }
}

pub fn create_initial_agents_view_scope_frames(
    initial_scope_key: Option<&AgentsViewScopeKey>,
    return_chat: Option<&SessionSummary>,
) -> Vec<AgentsViewScopeFrame> {
    let Some(initial_scope_key) = initial_scope_key else {
        return Vec::new();
    };
    vec![AgentsViewScopeFrame {
        scope: initial_scope_key.clone(),
        return_chat: match return_chat {
            Some(chat) if chat.session_id == initial_scope_key.session_id => Some(chat.clone()),
            _ => None,
        },
    }]
}

pub fn create_initial_agents_view_persistent_state(
    initial_scope_key: Option<&AgentsViewScopeKey>,
    initial_session: Option<&SessionSummary>,
) -> AgentsViewPersistentState {
    // A scoped view excludes its root from its own rows, so anchoring the
    // selection on the entered-from chat could never resolve there and would
    // only arm the pending-anchor state for the whole catalog scan.
    let seed_selection = initial_session.is_some() && initial_scope_key.is_none();
    let mut state = AgentsViewPersistentState {
        back_session: initial_session.cloned(),
        ..AgentsViewPersistentState::default()
    };
    if seed_selection {
        let session = initial_session.expect("seedSelection implies initialSession");
        state.selected_row_identity = Some(get_agents_view_summary_identity(session));
        state.selected_session_key = Some(get_agents_view_selection_key(session));
    }
    if let Some(scope_key) = initial_scope_key {
        state.scope_frames = Some(create_initial_agents_view_scope_frames(Some(scope_key), initial_session));
        if let Some(session) = initial_session {
            state.last_successful_live_summaries = Some(vec![session.clone()]);
        }
    }
    state
}

pub fn create_scope_back_return_chat_open_result(
    result: &AgentsViewRunResult,
) -> Option<AgentsViewRunResult> {
    let AgentsViewRunResult::ScopeBack {
        return_chat,
        expanded_ancestor_session_ids,
        has_children,
        ..
    } = result
    else {
        return None;
    };
    let return_chat = return_chat.clone()?;
    Some(AgentsViewRunResult::Open {
        summary: return_chat,
        selection: None,
        expanded_ancestor_session_ids: Some(expanded_ancestor_session_ids.clone()),
        has_children: Some(*has_children),
        status_message: None,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpenedAgentsViewSession {
    pub summary: SessionSummary,
    pub cwd_fallback_notice: Option<String>,
}

pub fn resolve_agents_view_open_cwd(
    summary: &SessionSummary,
    fallback_cwd: Option<&str>,
) -> (Option<String>, Option<String>) {
    if summary.cwd.is_empty() || std::path::Path::new(&summary.cwd).exists() || fallback_cwd.is_none() {
        return (None, None);
    }
    let fallback = fallback_cwd.unwrap();
    (
        Some(fallback.to_string()),
        Some(format!(
            "Original directory is missing ({}); opened in {} instead.",
            summary.cwd, fallback
        )),
    )
}

/// `expectSessionSummary` / `isSessionSummary`.
pub fn expect_session_summary(value: &Value) -> Result<SessionSummary, String> {
    if !is_session_summary(value) {
        return Err("Daemon returned an invalid session summary".to_string());
    }
    serde_json::from_value(value.clone()).map_err(|_| "Daemon returned an invalid session summary".to_string())
}

pub fn is_session_summary(value: &Value) -> bool {
    is_record(value)
        && value.get("id").map(|id| id.is_string()).unwrap_or(false)
        && value.get("sessionId").map(|id| id.is_string()).unwrap_or(false)
}

pub fn expect_session_list(value: &Value) -> Result<Vec<SessionSummary>, String> {
    let Some(sessions) = value.get("sessions").filter(|sessions| sessions.is_array()) else {
        return Err("Daemon returned an invalid session list response".to_string());
    };
    if !is_record(value) {
        return Err("Daemon returned an invalid session list response".to_string());
    }
    let mut result = Vec::new();
    for session in sessions.as_array().unwrap() {
        if !is_session_summary(session) {
            return Err("Daemon returned an invalid session summary".to_string());
        }
        result.push(serde_json::from_value(session.clone()).map_err(|_| {
            "Daemon returned an invalid session summary".to_string()
        })?);
    }
    Ok(result)
}

pub fn is_record(value: &Value) -> bool {
    value.is_object()
}

pub fn require_daemon_data(response: &DaemonResponse) -> Result<Value, String> {
    if !response.success {
        return Err(response.error.clone().unwrap_or_else(|| "unknown error".to_string()));
    }
    Ok(response.data.clone().unwrap_or(Value::Null))
}

pub fn format_error(prefix: &str, message: &str) -> String {
    format_agents_view_status_line(&format!("{prefix}: {message}"))
}

pub fn is_unknown_active_session_error(message: &str) -> bool {
    message.starts_with("Unknown active session:")
}

/// `isUnknownDaemonCommandError(error, command)` (modes/daemon/daemon-protocol.ts).
pub fn is_unknown_daemon_command_error(message: &str, command: &str) -> bool {
    message.contains("Unknown daemon command") && message.contains(command)
}


/// Editor surface used by the agents view (`CustomEditor` in the reference).
/// The TUI slice owns the real component; this trait captures the calls the mode
/// makes so the behaviour stays testable. TODO(port): bind pi-tui's editor.
pub trait AgentsViewEditor: Send {
    fn set_text(&mut self, text: &str);
    fn get_text(&self) -> String;
    fn get_expanded_text(&self) -> String;
    fn set_placeholder(&mut self, text: &str);
    fn render(&mut self, width: usize) -> Vec<String>;
    fn invalidate(&mut self);
    fn get_lines(&self) -> Vec<String>;
    fn get_cursor(&self) -> (usize, usize);
    /// `handleInput(data)`; returns true when the editor consumed the key.
    fn handle_input(&mut self, data: &str) -> bool;
    fn focus(&mut self);
    fn is_focused(&self) -> bool;
    fn take_submissions(&mut self) -> Vec<String>;
}

/// `Component` from pi-tui.
pub trait AgentsViewComponent: Send {
    fn render(&mut self, width: usize) -> Vec<String>;
    fn invalidate(&mut self) {}
}

/// Terminal surface (`TUI` + `ProcessTerminal`).
pub trait AgentsViewTerminal: Send + Sync {
    fn rows(&self) -> usize;
    fn request_render(&self, force: bool);
    fn set_title(&self, title: &str);
    fn columns(&self) -> usize;
    fn poll_input(&self) -> Result<Option<Vec<String>>, String>;
    fn present(&self, lines: Vec<String>, dock: Vec<String>) -> Result<(), String>;
}

/// `clippedFullscreenDockHeight(renderedRows, terminalRows)` from pi-tui.
pub fn clipped_fullscreen_dock_height(rendered_rows: usize, terminal_rows: usize) -> usize {
    // The dock never takes the whole screen: keep at least one content row.
    rendered_rows.min(terminal_rows.saturating_sub(1))
}

/// `truncateToWidth` from pi-tui; ANSI-aware in the reference. This port keeps the
/// plain-text path and leaves escape-aware truncation to the TUI slice.
pub fn truncate_to_width(value: &str, width: usize) -> String {
    pi_tui::utils::truncate_to_width(value, width as f64, "", false)
}

/// `visibleWidth` from pi-tui (ANSI sequences count as zero width).
pub fn visible_width(value: &str) -> usize {
    pi_tui::utils::visible_width(value)
}

fn unicode_width_of(ch: char) -> usize {
    match ch {
        '\u{200b}' | '\u{feff}' => 0,
        _ => {
            if is_wide(ch) {
                2
            } else {
                1
            }
        }
    }
}

fn is_wide(ch: char) -> bool {
    matches!(ch as u32,
        0x1100..=0x115F | 0x2E80..=0x303E | 0x3041..=0x33FF | 0x3400..=0x4DBF |
        0x4E00..=0x9FFF | 0xA000..=0xA4CF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF |
        0xFE30..=0xFE6F | 0xFF00..=0xFF60 | 0xFFE0..=0xFFE6 | 0x1F300..=0x1F64F |
        0x1F900..=0x1F9FF | 0x20000..=0x2FFFD | 0x30000..=0x3FFFD)
}

/// `wrapTextWithAnsi(line, width)` reduced to plain text wrapping.
pub fn wrap_text_with_ansi(text: &str, width: usize) -> Vec<String> {
    pi_tui::utils::wrap_text_with_ansi(text, width)
}

/// `padLine(line, width)`.
pub fn pad_line(line: &str, width: usize) -> String {
    let padding = width.saturating_sub(visible_width(line));
    format!("{line}{}", " ".repeat(padding))
}

/// `formatTableCell(value, width)`: truncate, then right-pad to the column.
pub fn format_table_cell(value: &str, width: usize) -> String {
    let truncated = truncate_to_width(value, width);
    let padding = width.saturating_sub(visible_width(&truncated));
    format!("{truncated}{}", " ".repeat(padding))
}

pub fn pad_cell_start(value: &str, width: usize) -> String {
    let padding = width.saturating_sub(visible_width(value));
    format!("{}{value}", " ".repeat(padding))
}

/// `theme.fg(color, text)` / `theme.bold` / `theme.italic` / `theme.bg`.
/// The interactive theme slice owns the real ANSI palette; the agents view only
/// needs the composition points, so colour is applied through a small trait.
pub trait AgentsViewTheme: Send + Sync {
    fn fg(&self, color: &str, text: &str) -> String;
    fn bg(&self, color: &str, text: &str) -> String;
    fn bold(&self, text: &str) -> String;
    fn italic(&self, text: &str) -> String;
    fn selection_background_color(&self) -> Box<dyn Fn(&str) -> String + Send + Sync>;
}

/// Passthrough theme used when the interactive slice has not landed: it returns
/// the text unchanged so row geometry and ordering stay testable.
#[derive(Debug, Default)]
pub struct PlainAgentsViewTheme;

impl AgentsViewTheme for PlainAgentsViewTheme {
    fn fg(&self, _color: &str, text: &str) -> String {
        text.to_string()
    }
    fn bg(&self, _color: &str, text: &str) -> String {
        text.to_string()
    }
    fn bold(&self, text: &str) -> String {
        text.to_string()
    }
    fn italic(&self, text: &str) -> String {
        text.to_string()
    }
    fn selection_background_color(&self) -> Box<dyn Fn(&str) -> String + Send + Sync> {
        Box::new(|text: &str| text.to_string())
    }
}

/// `keyText(keybinding)` from modes/interactive/components/keybinding-hints.ts.
pub fn key_text(keybinding: &str) -> String {
    keybinding
        .split('/')
        .map(|binding| {
            binding
                .split('+')
                .map(|part| {
                    let normalized = if part == "escape" { "esc" } else { part };
                    match normalized {
                        "up" => "↑".to_string(),
                        "down" => "↓".to_string(),
                        "left" => "←".to_string(),
                        "right" => "→".to_string(),
                        "alt" => {
                            if cfg!(target_os = "macos") {
                                "Option".to_string()
                            } else {
                                "Alt".to_string()
                            }
                        }
                        other => {
                            let mut chars = other.chars();
                            match chars.next() {
                                None => String::new(),
                                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                            }
                        }
                    }
                })
                .collect::<Vec<String>>()
                .join("+")
        })
        .collect::<Vec<String>>()
        .join("/")
}

/// Default keybindings used by the view (`KeybindingsManager.create()`).
pub fn default_keybinding(keybinding: &str) -> &'static str {
    match keybinding {
        "app.clear" => "ctrl+c",
        "app.shortcuts" => "ctrl+o",
        "app.agents.rename" => "ctrl+r",
        "app.agents.delete" => "ctrl+x",
        "app.agents.reply" => "ctrl+p",
        "app.agents.new" => "ctrl+n",
        "app.agents.program" => "ctrl+g",
        "app.agents.inactiveCollapse" => "ctrl+h",
        "app.agents.expand" => "ctrl+e",
        "app.agents.open" => "ctrl+a",
        "app.message.followUp" => "alt+enter",
        "tui.select.up" => "up",
        "tui.select.down" => "down",
        "tui.select.pageUp" => "pageup",
        "tui.select.pageDown" => "pagedown",
        "tui.select.confirm" => "enter",
        "tui.select.cancel" => "escape",
        _ => "",
    }
}

pub fn matches_key(data: &str, keybinding: &str) -> bool {
    pi_tui::keybindings::get_keybindings().matches(data, keybinding)
}

#[derive(Clone, Debug, PartialEq)]
pub enum DisplayItem {
    Spacer,
    Heading(AgentsViewSection),
    RunningSubagents(usize),
    Row(usize),
}


#[derive(Clone, Debug, PartialEq)]
pub struct AgentsViewUsageLayout {
    pub legend: String,
    pub details: HashMap<String, String>,
    pub name_width: usize,
    pub model_width: usize,
    pub activity_width: usize,
}

pub fn build_compact_agents_view_layout(
    rows: &[AgentsViewRow],
    width: usize,
    theme: &dyn AgentsViewTheme,
) -> AgentsViewUsageLayout {
    let _ = theme;
    let sessions: Vec<&AgentsViewRow> = rows
        .iter()
        .filter(|row| row.kind == AgentsViewRowKind::Agent || row.kind == AgentsViewRowKind::Subagent)
        .collect();
    let entries: Vec<(String, String, String)> = sessions
        .iter()
        .map(|row| {
            (
                row.identity.clone(),
                format!("${:.2}", row.recursive_cost),
                format_session_duration(&row.summary),
            )
        })
        .collect();
    let cost_width = entries.iter().fold(4usize, |size, entry| size.max(visible_width(&entry.1)));
    let age_width = entries.iter().fold(3usize, |size, entry| size.max(visible_width(&entry.2)));
    let details_width = cost_width + 2 + age_width;
    let available = width.saturating_sub(details_width + 4);
    let desired_model_width = sessions
        .iter()
        .fold(12usize, |size, row| size.max(visible_width(&format_session_model(&row.summary))));
    let model_width = desired_model_width.min(32).min(available.saturating_sub(12));
    let name_width = 28usize.min(available.saturating_sub(model_width));
    let activity_width = available.saturating_sub(model_width + name_width + 2);
    let detail_line = |cost: &str, age: &str| format!("{}  {}", pad_cell_start(cost, cost_width), pad_cell_start(age, age_width));
    let mut headings = vec![
        format_table_cell("Session", name_width),
        format_table_cell("Model", model_width),
    ];
    if activity_width > 0 {
        headings.push(format_table_cell("Activity", activity_width));
    }
    headings.push(detail_line("Cost", "Age"));
    AgentsViewUsageLayout {
        legend: format_table_cell(&headings.join("  "), width),
        details: entries
            .into_iter()
            .map(|(identity, cost, age)| (identity, detail_line(&cost, &age)))
            .collect(),
        name_width,
        model_width,
        activity_width,
    }
}

pub fn format_session_model(summary: &SessionSummary) -> String {
    summary.model.as_ref().map(|model| model.id.clone()).unwrap_or_else(|| "-".to_string())
}

pub fn format_session_duration(summary: &SessionSummary) -> String {
    let value = if summary.active_session_id.is_some() {
        summary.created.as_deref().or(summary.modified.as_deref())
    } else {
        summary.modified.as_deref().or(summary.created.as_deref())
    };
    format_agents_view_relative_time(value, super::agents_view_state::now_ms())
}

pub fn format_agents_view_relative_time(value: Option<&str>, now: i64) -> String {
    let Some(timestamp) = parse_session_timestamp(value) else {
        return String::new();
    };
    let seconds = ((now - timestamp) as f64 / 1000.0).floor().max(0.0) as i64;
    if seconds < 60 {
        return format!("{seconds}s");
    }
    let minutes = (seconds / 60) as i64;
    if minutes < 60 {
        return format!("{minutes}m");
    }
    let hours = minutes / 60;
    if hours < 24 {
        return format!("{hours}h");
    }
    format!("{}d", hours / 24)
}

pub fn parse_session_timestamp(value: Option<&str>) -> Option<i64> {
    let value = value?;
    if value.is_empty() {
        return None;
    }
    parse_js_timestamp(value)
}

// Explicit session names read bold so they stand out from fallback titles
// (first prompt, cwd, ids); the "(no messages)" placeholder reads italic.
pub fn style_row_title(row: &AgentsViewRow, theme: &dyn AgentsViewTheme) -> String {
    if row
        .summary
        .session_name
        .as_deref()
        .map(|name| !collapse_whitespace(name).is_empty())
        .unwrap_or(false)
    {
        return theme.bold(&row.title);
    }
    if row.title == "(no messages)" {
        return theme.italic(&row.title);
    }
    row.title.clone()
}

/// `isInactiveExpanded(state)`.
pub fn is_inactive_expanded(state: &AgentsViewPersistentState) -> bool {
    // Older clients persisted false even when the user never chose to hide saved chats.
    state.inactive_visibility_explicit != Some(true) || state.inactive_expanded != Some(false)
}

/// `compactSessionRows(rows, showInactive)`.
pub fn compact_session_rows(rows: &[AgentsViewRow], show_inactive: bool) -> Vec<AgentsViewRow> {
    let mut visible = true;
    rows.iter()
        .filter(|row| {
            if row.depth == 0 {
                visible = show_inactive || row.section != AgentsViewSection::Inactive;
            }
            visible && row.kind != AgentsViewRowKind::SubagentSummary
        })
        .cloned()
        .collect()
}

// Nested rows (subagent summaries and expanded subagents) always render in
// their top-level agent's section block, regardless of their own section.
pub fn get_display_rows_for_section(rows: &[AgentsViewRow], section: AgentsViewSection) -> Vec<AgentsViewRow> {
    let mut result: Vec<AgentsViewRow> = Vec::new();
    let mut include = false;
    for row in rows {
        if row.depth == 0 {
            include = row.section == section;
        }
        if include {
            result.push(row.clone());
        }
    }
    result
}

pub fn count_rows_by_section(rows: &[AgentsViewRow]) -> HashMap<AgentsViewSection, i64> {
    let agents: Vec<&AgentsViewRow> = rows.iter().filter(|row| row.kind == AgentsViewRowKind::Agent).collect();
    let mut counts = HashMap::new();
    for section in [AgentsViewSection::Running, AgentsViewSection::Idle, AgentsViewSection::Inactive] {
        counts.insert(
            section,
            agents.iter().filter(|row| row.section == section).count() as i64,
        );
    }
    counts
}

pub fn get_selected_row_identity(row: Option<&AgentsViewRow>) -> Option<String> {
    row.map(|row| row.identity.clone())
}

pub fn row_has_spawn_code(row: &AgentsViewRow) -> bool {
    row.summary.spawn_code.as_deref().map(|code| !code.trim().is_empty()).unwrap_or(false)
}

// Destructive actions gate on live work anywhere in the subtree, never on the display section.
pub fn has_live_work(row: &AgentsViewRow) -> bool {
    row.section == AgentsViewSection::Running
        || row.running_subagent_count > 0
        || row.summary.has_running_rlm_children == Some(true)
}

/// `formatAgentsViewRelativeTime` for the reply header.
pub fn format_agents_view_relative_time_now(value: Option<&str>) -> String {
    format_agents_view_relative_time(value, super::agents_view_state::now_ms())
}

/// `new Date().toISOString()` used by `logClientError`.
pub fn now_iso_string() -> String {
    to_iso_string(&chrono::Utc::now())
}


/// `listDaemonHeartbeats` (modes/daemon/heartbeat-catalog.ts).
pub async fn list_daemon_heartbeats(
    client: &DaemonTransportClient,
    active_session_id: Option<&str>,
) -> Result<Vec<AgentConnectionHeartbeat>, String> {
    if client.hello().is_none() {
        let _ = client.wait_for_hello(3000).await;
    }
    if !client.supports_server_capability("heartbeat_catalog") {
        return Ok(Vec::new());
    }
    let mut command = serde_json::json!({ "type": "heartbeats_list" });
    if let Some(active_session_id) = active_session_id {
        command["activeSessionId"] = Value::String(active_session_id.to_string());
    }
    match client.request(command, 30000).await {
        Ok(response) => {
            if !response.success {
                return Err(deserialize_daemon_error(&response));
            }
            let heartbeats = response
                .data
                .as_ref()
                .and_then(|data| data.get("heartbeats"))
                .and_then(|value| value.as_array())
                .map(|values| {
                    values
                        .iter()
                        .filter_map(|value| serde_json::from_value(value.clone()).ok())
                        .collect()
                })
                .unwrap_or_default();
            Ok(heartbeats)
        }
        Err(error) => {
            if is_unknown_daemon_command_error(&error, "heartbeats_list") {
                return Ok(Vec::new());
            }
            Err(error)
        }
    }
}

/// `deserializeDaemonError`: the specialized error classes are declared by the
/// daemon slice, so this returns the same message the reference surfaces.
pub fn deserialize_daemon_error(response: &DaemonResponse) -> String {
    response.error.clone().unwrap_or_else(|| "unknown error".to_string())
}

pub const SAVED_SESSION_LIST_TIMEOUT_MS: u64 = 30000;

/// `DaemonSavedSessionCatalogContext`.
#[derive(Clone, Debug, PartialEq)]
pub enum DaemonSavedSessionCatalogContext {
    ActiveSessionId(String),
    Cwd { cwd: String, session_dir: Option<String> },
}

/// `deserializeSavedSessionInfo` (modes/daemon/saved-session-info.ts).
pub fn deserialize_saved_session_info(value: &Value) -> Result<AgentConnectionSavedSessionInfo, String> {
    let mut session: AgentConnectionSavedSessionInfo =
        serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
    // The wire carries ISO strings; `Date` conversion is already done by serde.
    let _ = &mut session;
    Ok(session)
}

/// `listDaemonSavedSessions` (modes/daemon/saved-session-catalog.ts).
pub async fn list_daemon_saved_sessions(
    client: &DaemonTransportClient,
    context: &DaemonSavedSessionCatalogContext,
    scope: &str,
    // `AgentConnectionSessionListCallbacks` stores these as shared closures
    // (agent_connection/types.rs), and the progress listener is `'static`.
    on_session: Option<Arc<dyn Fn() + Send + Sync>>,
    on_progress: Option<Arc<dyn Fn(i64, i64) + Send + Sync>>,
) -> Result<Vec<AgentConnectionSavedSessionInfo>, String> {
    let command = match context {
        DaemonSavedSessionCatalogContext::ActiveSessionId(active_session_id) => serde_json::json!({
            "type": "list_saved_sessions",
            "activeSessionId": active_session_id,
            "scope": scope,
        }),
        DaemonSavedSessionCatalogContext::Cwd { cwd, session_dir } => {
            let mut command = serde_json::json!({
                "type": "list_saved_sessions",
                "cwd": cwd,
                "scope": scope,
            });
            if let Some(session_dir) = session_dir {
                command["sessionDir"] = Value::String(session_dir.clone());
            }
            command
        }
    };
    let on_session_callback = on_session.clone();
    let on_progress_callback = on_progress.clone();
    let response = client
        .request_with_options(
            command,
            SAVED_SESSION_LIST_TIMEOUT_MS,
            DaemonClientRequestOptions::new()
                .recoverable(true)
                .with_progress(Box::new(move |progress: &Value| {
                if progress.get("type").and_then(|value| value.as_str()) == Some("session_list_progress") {
                    if let Some(callback) = on_progress_callback.as_ref() {
                        callback(
                            progress.get("loaded").and_then(|value| value.as_i64()).unwrap_or(0),
                            progress.get("total").and_then(|value| value.as_i64()).unwrap_or(0),
                        );
                    }
                } else if let Some(callback) = on_session_callback.as_ref() {
                    callback();
                }
            })),
        )
        .await?;
    if !response.success {
        return Err(deserialize_daemon_error(&response));
    }
    let sessions = response
        .data
        .as_ref()
        .and_then(|data| data.get("sessions"))
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| deserialize_saved_session_info(value).ok())
                .collect()
        })
        .unwrap_or_default();
    Ok(sessions)
}

/// `renameDaemonSavedSession`.
pub async fn rename_daemon_saved_session(
    client: &DaemonTransportClient,
    context: &DaemonSavedSessionCatalogContext,
    session_path: &str,
    name: &str,
) -> Result<(), String> {
    let command = match context {
        DaemonSavedSessionCatalogContext::ActiveSessionId(active_session_id) => serde_json::json!({
            "type": "rename_saved_session",
            "activeSessionId": active_session_id,
            "sessionPath": session_path,
            "name": name,
        }),
        DaemonSavedSessionCatalogContext::Cwd { .. } => serde_json::json!({
            "type": "rename_saved_session",
            "sessionPath": session_path,
            "name": name,
        }),
    };
    let response = client.request(command, 30000).await?;
    if !response.success {
        return Err(deserialize_daemon_error(&response));
    }
    Ok(())
}

/// `DeleteSessionFileResult` (core/session-file-actions.ts).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeleteSessionFileResult {
    pub ok: bool,
    pub method: Option<String>,
    pub error: Option<String>,
}

/// `deleteDaemonSavedSession`.
pub async fn delete_daemon_saved_session(
    client: &DaemonTransportClient,
    context: &DaemonSavedSessionCatalogContext,
    session_path: &str,
) -> Result<DeleteSessionFileResult, String> {
    let command = match context {
        DaemonSavedSessionCatalogContext::ActiveSessionId(active_session_id) => serde_json::json!({
            "type": "delete_saved_session",
            "activeSessionId": active_session_id,
            "sessionPath": session_path,
        }),
        DaemonSavedSessionCatalogContext::Cwd { .. } => serde_json::json!({
            "type": "delete_saved_session",
            "sessionPath": session_path,
        }),
    };
    let response = client.request(command, 30000).await?;
    if !response.success {
        return Err(deserialize_daemon_error(&response));
    }
    let data = response.data.clone().unwrap_or(Value::Null);
    Ok(DeleteSessionFileResult {
        ok: data.get("ok").and_then(|value| value.as_bool()).unwrap_or(false),
        method: data.get("method").and_then(|value| value.as_str()).map(|value| value.to_string()),
        error: data.get("error").and_then(|value| value.as_str()).map(|value| value.to_string()),
    })
}

struct PendingClient(Option<DaemonTransportClient>);
impl Drop for PendingClient {
    fn drop(&mut self) {
        if let Some(client) = self.0.take() { client.close(); }
    }
}

/// `connectAgentsViewDaemonClient(socketPath)`.
pub async fn connect_agents_view_daemon_client(
    socket_path: &str,
    transport: Arc<dyn super::roster_store::DaemonTransport>,
) -> Result<DaemonTransportClient, String> {
    let transport = transport.fresh_transport().unwrap_or(transport);
    let client = DaemonTransportClient::new(transport);
    let mut pending = PendingClient(Some(client.clone()));
    match client.connect(3000).await {
        Ok(()) => { pending.0 = None; Ok(client) },
        Err(error) => {
            client.close();
            Err(error)
        }
    }
}


/// `DaemonAgentConnection` surface used by the agents view.
/// TODO(port): modes/agent-connection/daemon-agent-connection.ts owns the real
/// type; this trait keeps the attach/prompt/dispose sequence in place.
pub trait DaemonAgentConnectionHandle: Send + Sync {
    fn prompt(&self, message: &str, streaming_behavior: Option<&str>) -> TransportFuture<Result<(), String>>;
    fn dispose(&self) -> TransportFuture<Result<(), String>>;
}

pub type TransportFuture<T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send>>;

/// `InteractiveMode` surface used by the run loop.
/// TODO(port): modes/interactive/interactive-mode.ts owns the real type.
pub trait InteractiveModeHandle: Send {
    fn run(&mut self) -> TransportFuture<Result<InteractiveRunResult, String>>;
    fn teardown_session_ui(&mut self, preserve_alt_screen: bool) -> TransportFuture<Result<(), String>>;
}

/// The subset of `InteractiveModeRunResult` the run loop reads.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InteractiveRunResult {
    pub kind: String,
    pub source: SessionSummary,
}

pub type InteractiveModeFactory = Box<
    dyn Fn(InteractiveModeOptions) -> Box<dyn InteractiveModeHandle> + Send + Sync,
>;

#[derive(Clone)]
pub struct InteractiveModeOptions {
    pub daemon_socket_path: Option<String>,
    pub ui_services: Arc<dyn AgentsViewUiServices>,
    pub prompt_stash_session_id: String,
    pub bind_local_session_extensions: bool,
    pub migrated_providers: Option<Vec<String>>,
    pub model_fallback_message: Option<String>,
    pub startup_notice: Option<String>,
    pub verbose: Option<bool>,
    pub return_to_agents_view: bool,
    pub force_fullscreen: bool,
    /// The agents view renders the global notices itself, so suppress them in-session.
    pub agents_view_owns_startup_notices: bool,
    pub session_depth: Option<i64>,
    pub session_has_children: Option<bool>,
    /// Metadata already obtained by attach; do not fetch the full chat again before mounting its UI.
    pub source_summary: SessionSummary,
}

pub trait DaemonAgentConnectionFactory: Send + Sync {
    /// `DaemonAgentConnection.attach(client, activeSessionId, options)`.
    fn attach(
        &self,
        client: DaemonTransportClient,
        active_session_id: &str,
        options: AttachOptions,
    ) -> TransportFuture<Result<Arc<dyn DaemonAgentConnectionHandle>, String>>;
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttachOptions {
    pub close_client_on_dispose: Option<bool>,
    pub supports_extension_ui: Option<bool>,
    pub reconnect_timeout_ms: Option<u64>,
    pub telemetry_disabled: Option<bool>,
}

/// `openAgentsViewSession(options, summary)`.
pub async fn open_agents_view_session(
    options: &AgentsViewModeOptions,
    summary: &SessionSummary,
    transport: Arc<dyn super::roster_store::DaemonTransport>,
    factory: &dyn DaemonAgentConnectionFactory,
) -> Result<(Arc<dyn DaemonAgentConnectionHandle>, SessionSummary, Option<String>), String> {
    let socket_path = options
        .socket_path
        .clone()
        .ok_or_else(|| "Agents view daemon socket is not configured".to_string())?;
    let mut client = connect_agents_view_daemon_client(&socket_path, transport.clone()).await?;
    let mut pending = PendingClient(Some(client.clone()));
    if let Some(active_session_id) = summary.active_session_id.clone() {
        let attached = factory
            .attach(
                client.clone(),
                &active_session_id,
                AttachOptions {
                    close_client_on_dispose: Some(true),
                    supports_extension_ui: Some(true),
                    reconnect_timeout_ms: options.reconnect_timeout_ms,
                    telemetry_disabled: options.config.telemetry_disabled,
                },
            )
            .await;
        match attached {
            Ok(connection) => { pending.0 = None; return Ok((connection, summary.clone(), None)); },
            Err(error) => {
                client.close();
                // Recovering takes the saved-session path too; its create/open route
                // retries the recovery.
                if summary.session_file.is_none()
                    || !(is_unknown_active_session_error(&error)
                        || error.starts_with("Session is recovering"))
                {
                    return Err(error);
                }
                client = connect_agents_view_daemon_client(&socket_path, transport.clone()).await?;
                pending.0 = Some(client.clone());
            }
        }
    }

    if summary.session_file.is_none() {
        client.close();
        return Err("Cannot open agent without an active runtime or saved session file".to_string());
    }

    match resume_saved_agents_view_session(&client, &options.config, summary).await {
        Ok((resumed, active_session_id, cwd_fallback_notice)) => {
            let attached = factory
                .attach(
                    client.clone(),
                    &active_session_id,
                    AttachOptions {
                        close_client_on_dispose: Some(true),
                        supports_extension_ui: Some(true),
                        reconnect_timeout_ms: options.reconnect_timeout_ms,
                        telemetry_disabled: options.config.telemetry_disabled,
                    },
                )
                .await;
            match attached {
                Ok(connection) => { pending.0 = None; Ok((connection, resumed, cwd_fallback_notice)) },
                Err(error) => {
                    client.close();
                    Err(error)
                }
            }
        }
        Err(error) => {
            client.close();
            Err(error)
        }
    }
}

/// Resume a saved session file into the daemon and return the live summary.
/// The daemon's create-with-sessionPath is idempotent for this client: an
/// already-resident session is reused instead of resumed twice.
pub async fn resume_saved_agents_view_session(
    client: &DaemonTransportClient,
    config: &AgentsViewRuntimeConfig,
    summary: &SessionSummary,
) -> Result<(SessionSummary, String, Option<String>), String> {
    let Some(session_file) = summary.session_file.clone() else {
        return Err("Cannot resume a session without a saved session file".to_string());
    };
    let (override_cwd, notice) = resolve_agents_view_open_cwd(summary, config.cwd.as_deref());
    let resume_config = create_agents_view_resume_config(config, override_cwd.as_deref());
    let mut command = serde_json::json!({
        "type": "create",
        "config": {
            "cwd": resume_config.cwd,
            "sessionDir": resume_config.session_dir,
            "telemetryDisabled": resume_config.telemetry_disabled,
        },
        "sessionPath": session_file,
    });
    if resume_config.cwd.is_none() {
        command["config"].as_object_mut().unwrap().remove("cwd");
    }
    if resume_config.session_dir.is_none() {
        command["config"].as_object_mut().unwrap().remove("sessionDir");
    }
    if resume_config.telemetry_disabled.is_none() {
        command["config"].as_object_mut().unwrap().remove("telemetryDisabled");
    }
    let response = client.request(command, 30000).await?;
    let data = require_daemon_data(&response)?;
    let created_summary = expect_session_summary(&data)?;
    let active_session_id = get_required_active_session_id(&created_summary)?;
    Ok((created_summary, active_session_id, notice))
}

pub fn get_required_active_session_id(summary: &SessionSummary) -> Result<String, String> {
    summary
        .active_session_id
        .clone()
        .ok_or_else(|| "Daemon returned a session without an active session id".to_string())
}


/// `runAgentsViewMode(options)`.
pub async fn run_agents_view_mode(
    options: AgentsViewModeOptions,
    terminal: Arc<dyn AgentsViewTerminal>,
    editor: Box<dyn AgentsViewEditor>,
    theme: Arc<dyn AgentsViewTheme>,
    transport: Arc<dyn super::roster_store::DaemonTransport>,
    factory: &dyn DaemonAgentConnectionFactory,
    interactive_factory: InteractiveModeFactory,
    recover_daemon: Option<Arc<dyn Fn() -> TransportFuture<Result<(), String>> + Send + Sync>>,
    prompt_stash_store: Option<Arc<()>>,
) -> Result<(), String> {
    let persistent_state =
        create_initial_agents_view_persistent_state(options.initial_scope_key.as_ref(), options.initial_session.as_ref());
    let mut runner = AgentsViewRunner {
        options,
        persistent_state,
        terminal,
        editor: Some(editor),
        theme,
        transport,
        #[cfg(test)]
        scripted_transport: None,
        factory,
        interactive_factory,
        recover_daemon,
        prompt_stash_store,
        roster_store: None,
        roster_client: None,
    };
    runner.run_loop().await
}

struct AgentsViewRunner<'a> {
    options: AgentsViewModeOptions,
    persistent_state: AgentsViewPersistentState,
    terminal: Arc<dyn AgentsViewTerminal>,
    /// `AgentsViewMode` owns the editor for the duration of one view, so the
    /// runner parks it here between iterations.
    editor: Option<Box<dyn AgentsViewEditor>>,
    theme: Arc<dyn AgentsViewTheme>,
    transport: Arc<dyn super::roster_store::DaemonTransport>,
    /// Concrete handle to the scripted transport, for test hooks only.
    #[cfg(test)]
    scripted_transport: Option<Arc<tests::FakeTransport>>,
    factory: &'a dyn DaemonAgentConnectionFactory,
    interactive_factory: InteractiveModeFactory,
    recover_daemon: Option<Arc<dyn Fn() -> TransportFuture<Result<(), String>> + Send + Sync>>,
    prompt_stash_store: Option<Arc<()>>,
    roster_store: Option<AgentsViewRosterStore>,
    roster_client: Option<DaemonTransportClient>,
}

impl AgentsViewRunner<'_> {
    async fn run_loop(&mut self) -> Result<(), String> {
        let result = self.run_loop_inner().await;
        // Close first: the supervisor drops the subscription with the socket.
        if let Some(client) = self.persistent_state.roster_client.take() {
            client.close();
        }
        if let Some(client) = self.roster_client.take() {
            client.close();
        }
        if let Some(store) = self.roster_store.take() {
            store.dispose().await;
        }
        result
    }

    async fn run_loop_inner(&mut self) -> Result<(), String> {
        loop {
            // The view borrows the editor and the interactive factory, so its
            // borrows are scoped to this block; `take_persistent_state` carries the
            // mutable state back out, matching the reference's shared object.
            let editor = self
                .editor
                .take()
                .ok_or_else(|| "Agents view editor is not available".to_string())?;
            let (view_result, persistent_state, editor) = {
                let mut view = AgentsViewMode::new(
                    self.options.clone(),
                    self.persistent_state.clone(),
                    self.terminal.clone(),
                    editor,
                    self.theme.clone(),
                    self.transport.clone(),
                    self.factory,
                    &mut self.interactive_factory,
                    self.recover_daemon.clone(),
                );
                let view_result = view.run().await;
                let persistent_state = view.take_persistent_state();
                (view_result, persistent_state, view.editor)
            };
            self.editor = Some(editor);
            self.persistent_state = persistent_state;
            let view_result = view_result?;
            let result = match view_result {
                AgentsViewRunResult::ScopeBack {
                    ref selection,
                    ref expanded_ancestor_session_ids,
                    ..
                } => {
                    let frames = self.persistent_state.scope_frames.clone().unwrap_or_default();
                    self.persistent_state.scope_frames =
                        Some(transition_agents_view_scope(&frames, &super::agents_view_state::AgentsViewScopeAction::Back));
                    self.persistent_state.scope_root_summary = None;
                    self.persistent_state.selected_row_identity = Some(get_agents_view_summary_identity(selection));
                    self.persistent_state.selected_session_key = Some(get_agents_view_selection_key(selection));
                    self.persistent_state.pending_expanded_ancestor_session_ids =
                        Some(expanded_ancestor_session_ids.clone());
                    self.persistent_state.query = Some(String::new());
                    match create_scope_back_return_chat_open_result(&view_result) {
                        Some(result) => result,
                        None => continue,
                    }
                }
                other => other,
            };

            let (summary, selection, expanded_ancestors, status_message) = match &result {
                AgentsViewRunResult::Open {
                    summary,
                    selection,
                    expanded_ancestor_session_ids,
                    status_message,
                    ..
                } => (
                    summary.clone(),
                    selection.clone().unwrap_or_else(|| summary.clone()),
                    expanded_ancestor_session_ids.clone(),
                    status_message.clone(),
                ),
                AgentsViewRunResult::Exit => return Ok(()),
                _ => continue,
            };
            self.persistent_state.selected_row_identity = Some(get_agents_view_summary_identity(&selection));
            self.persistent_state.selected_session_key = Some(get_agents_view_selection_key(&selection));
            self.persistent_state.pending_expanded_ancestor_session_ids = expanded_ancestors;
            if let Some(status_message) = status_message {
                self.persistent_state.status_message = Some(status_message);
            }

            let opened = wait_for_session_open(
                self.terminal.as_ref(),
                &summary,
                open_agents_view_session(&self.options, &summary, self.transport.clone(), self.factory),
            ).await?;
            let Some(opened) = opened else {
                self.persistent_state.status_message = Some("Opening cancelled; the agent was left running".into());
                continue;
            };
            match opened {
                Ok((connection, opened_summary, cwd_fallback_notice)) => {
                    self.persistent_state.back_session = Some(opened_summary.clone());
                    if let Some(notice) = &cwd_fallback_notice {
                        self.persistent_state.status_message = combine_agents_view_startup_notices(&[
                            result_status_message(&result).as_deref(),
                            Some(notice.as_str()),
                        ]);
                    }
                    let mut interactive = (self.interactive_factory)(InteractiveModeOptions {
                        daemon_socket_path: self.options.socket_path.clone(),
                        ui_services: self.options.ui_services.clone(),
                        prompt_stash_session_id: opened_summary.session_id.clone(),
                        bind_local_session_extensions: false,
                        migrated_providers: self.options.migrated_providers.clone(),
                        model_fallback_message: resolve_attach_model_fallback_message(
                            &opened_summary,
                            self.options.model_fallback_message.as_deref(),
                        ),
                        startup_notice: combine_agents_view_startup_notices(&[
                            result_status_message(&result).as_deref(),
                            cwd_fallback_notice.as_deref(),
                        ]),
                        verbose: self.options.verbose,
                        return_to_agents_view: true,
                        force_fullscreen: true,
                        agents_view_owns_startup_notices: true,
                        session_depth: opened_summary.rlm_depth,
                        session_has_children: result_has_children(&result),
                        source_summary: opened_summary.clone(),
                    });
                    match interactive.run().await {
                        Ok(interactive_result) => {
                            if interactive_result.kind == "exit" { return Ok(()); }
                            let mut returned_session = opened_summary.clone();
                            let source = interactive_result.source;
                            returned_session.active_session_id = source.active_session_id.clone();
                            if let Some(active) = &source.active_session_id {
                                returned_session.id = active.clone();
                            }
                            returned_session.session_id = source.session_id.clone();
                            returned_session.modified = source.modified.clone();
                            returned_session.created = source.created.clone();
                            returned_session.model = source.model.clone();
                            returned_session.cwd = source.cwd.clone();
                            returned_session.is_streaming = source.is_streaming;
                            returned_session.is_compacting = source.is_compacting;
                            returned_session.is_bash_running = source.is_bash_running;
                            returned_session.is_running_tools = source.is_running_tools;
                            returned_session.message_count = source.message_count;
                            returned_session.session_file = source.session_file.clone();
                            returned_session.rlm_depth = source.rlm_depth;
                            // Preserve an unattachable child's selection while its parent chat was open.
                            if selection.session_id == summary.session_id {
                                self.persistent_state.selected_row_identity =
                                    Some(get_agents_view_summary_identity(&returned_session));
                                self.persistent_state.selected_session_key =
                                    Some(get_agents_view_selection_key(&returned_session));
                            }
                            if interactive_result.kind == "scoped_agents_view" {
                                let next_scope = AgentsViewScopeKey {
                                    session_id: source.session_id.clone(),
                                    active_session_id: source.active_session_id.clone(),
                                };
                                let frames = self.persistent_state.scope_frames.clone().unwrap_or_default();
                                self.persistent_state.scope_frames = Some(transition_agents_view_scope(
                                    &frames,
                                    &super::agents_view_state::AgentsViewScopeAction::Push {
                                        scope: next_scope,
                                        return_chat: Some(returned_session.clone()),
                                    },
                                ));
                                let cached = self
                                    .persistent_state
                                    .last_successful_live_summaries
                                    .clone()
                                    .unwrap_or_default();
                                let cached_index =
                                    cached.iter().position(|entry| entry.session_id == returned_session.session_id);
                                self.persistent_state.last_successful_live_summaries = Some(match cached_index {
                                    None => {
                                        let mut next = cached.clone();
                                        next.push(returned_session.clone());
                                        next
                                    }
                                    Some(index) => cached
                                        .iter()
                                        .enumerate()
                                        .map(|(position, entry)| {
                                            if position == index { returned_session.clone() } else { entry.clone() }
                                        })
                                        .collect(),
                                });
                                self.persistent_state.scope_root_summary = None;
                                self.persistent_state.query = Some(String::new());
                            }
                            self.persistent_state.back_session = Some(returned_session);
                        }
                        Err(error) => {
                            // The session opened fine and then threw while running; label it as a
                            // runtime crash so it isn't mixed in with true open failures.
                            log_client_error("Agent session crashed", &error);
                            self.persistent_state.status_message = Some(format_error("Agent session crashed", &error));
                            // Tear down the session TUI exactly as a normal back-navigation would
                            // (drain input, stop renderer + theme watcher) so it doesn't fight the
                            // agents-view UI for the terminal, then drop the daemon connection.
                            let _ = interactive.teardown_session_ui(true).await;
                            let _ = connection.dispose().await;
                        }
                    }
                }
                Err(error) => {
                    log_client_error("Failed to open agent", &error);
                    self.persistent_state.status_message = Some(format_error("Failed to open agent", &error));
                }
            }
        }
    }
}

async fn wait_for_session_open<T>(
    terminal: &dyn AgentsViewTerminal,
    summary: &SessionSummary,
    pending: impl std::future::Future<Output = T>,
) -> Result<Option<T>, String> {
    tokio::pin!(pending);
    let mut ticker = tokio::time::interval(std::time::Duration::from_millis(16));
    let started = std::time::Instant::now();
    loop {
        tokio::select! {
            result = &mut pending => return Ok(Some(result)),
            _ = ticker.tick() => {
                let Some(input) = terminal.poll_input()? else { return Ok(None); };
                if input.iter().any(|key| matches_key(key, "tui.select.cancel") || matches_key(key, "app.clear")) {
                    return Ok(None);
                }
                let name = summary.session_name.as_deref().unwrap_or(&summary.session_id);
                terminal.present(
                    vec![format!("Opening {name}… ({:.1}s)", started.elapsed().as_secs_f64())],
                    vec![format!("{} cancel opening · agent work is not stopped", key_text(default_keybinding("tui.select.cancel")))],
                )?;
            }
        }
    }
}

fn result_status_message(result: &AgentsViewRunResult) -> Option<String> {
    match result {
        AgentsViewRunResult::Open { status_message, .. } => status_message.clone(),
        _ => None,
    }
}

fn result_has_children(result: &AgentsViewRunResult) -> Option<bool> {
    match result {
        AgentsViewRunResult::Open { has_children, .. } => *has_children,
        _ => None,
    }
}

/// `resolveAttachModelFallbackMessage` (modes/daemon/daemon-session-list.ts).
pub fn resolve_attach_model_fallback_message(
    summary: &SessionSummary,
    startup_model_fallback_message: Option<&str>,
) -> Option<String> {
    if let Some(message) = &summary.model_fallback_message {
        return Some(message.clone());
    }
    if summary.model.is_some() {
        return None;
    }
    startup_model_fallback_message.map(|message| message.to_string())
}

/// `logClientError`: the TUI owns stdout/stderr, so a log file is the only safe sink.
pub fn log_client_error(prefix: &str, error: &str) {
    let line = format!("[{}] {prefix}: {error}", now_iso_string());
    append_rotating_log(&client_error_log_path(), &line);
}

pub const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

/// `appendRotatingLog` (config.ts). Private plumbing for `logClientError`.
pub fn append_rotating_log(log_path: &str, message: &str) {
    let path = std::path::Path::new(log_path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(metadata) = std::fs::metadata(path) {
        if metadata.len() > MAX_LOG_BYTES {
            // Drop any prior .old first: rename fails on Windows if it exists.
            let old = format!("{log_path}.old");
            let _ = std::fs::remove_file(&old);
            let _ = std::fs::rename(path, &old);
        }
    }
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{message}");
    }
}

/// `getClientErrorLogPath()` = `<agentDir>/logs/client-errors.log`.
pub fn client_error_log_path() -> String {
    format!("{}{}logs{}client-errors.log", agent_dir(), std::path::MAIN_SEPARATOR, std::path::MAIN_SEPARATOR)
}

/// `getAgentDir()`: `$PI_CONFIG_DIR_AGENT_DIR`-style override, else `~/.prime`.
pub const ENV_AGENT_DIR: &str = "PRIME_AGENT_DIR";

pub fn agent_dir() -> String {
    if let Ok(value) = std::env::var(ENV_AGENT_DIR) {
        if !value.is_empty() {
            return value;
        }
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    format!("{home}{}.prime", std::path::MAIN_SEPARATOR)
}


/// `class AgentsViewMode implements Component, Focusable`.
pub struct AgentsViewMode<'a> {
    options: AgentsViewModeOptions,
    persistent_state: AgentsViewPersistentState,
    terminal: Arc<dyn AgentsViewTerminal>,
    editor: Box<dyn AgentsViewEditor>,
    theme: Arc<dyn AgentsViewTheme>,
    transport: Arc<dyn super::roster_store::DaemonTransport>,
    /// Concrete handle to the scripted transport, for test hooks only.
    #[cfg(test)]
    scripted_transport: Option<Arc<tests::FakeTransport>>,
    factory: &'a dyn DaemonAgentConnectionFactory,
    interactive_factory: &'a mut InteractiveModeFactory,
    recover_daemon: Option<Arc<dyn Fn() -> TransportFuture<Result<(), String>> + Send + Sync>>,
    roster_store: Arc<AgentsViewRosterStore>,
    roster_listener: Option<usize>,
    client: Option<DaemonTransportClient>,
    unsubscribe_client_close: Option<Box<dyn Fn() + Send + Sync>>,
    unsubscribe_client_message: Option<Box<dyn Fn() + Send + Sync>>,
    /// Last close reason reported by the transport (`getDaemonSocketCloseReason`).
    last_close_reason: Arc<Mutex<Option<String>>>,
    reconnect_last_error: Option<String>,
    reconnect_started: bool,
    reconnect_timed_out: bool,
    daemon_shutdown_received: bool,
    resolve_run: Option<tokio::sync::oneshot::Sender<AgentsViewRunResult>>,
    ctrl_c_exit_hint_expires_at: i64,
    delete_confirm_expires_at: i64,
    working_icon_frame: i64,
    rows: Vec<AgentsViewRow>,
    all_rows: Vec<AgentsViewRow>,
    show_actions: bool,
    last_listed_summaries: Vec<SessionSummary>,
    last_visible_summaries: Vec<SessionSummary>,
    saved_sessions: Vec<AgentConnectionSavedSessionInfo>,
    last_successful_saved_sessions: Vec<AgentConnectionSavedSessionInfo>,
    heartbeats: Vec<AgentConnectionHeartbeat>,
    unified_records: Vec<UnifiedSessionRecord>,
    unified_index: UnifiedSessionIndex,
    scoped_records: Vec<UnifiedSessionRecord>,
    scope_key: Option<AgentsViewScopeKey>,
    scope_root_summary: Option<SessionSummary>,
    saved_catalog_ready: bool,
    saved_catalog_generation: i64,
    heartbeat_catalog_generation: i64,
    saved_catalog_refresh_pending: bool,
    saved_catalog_progress: i64,
    expanded_subagent_parents: HashSet<String>,
    /// Agent row identities whose full spawn program is currently shown.
    program_shown_parents: HashSet<String>,
    selected_index: usize,
    selected_row_identity: Option<String>,
    selected_active_session_id: Option<String>,
    selected_session_key: Option<AgentsViewSelectionKey>,
    selection_anchor_pending: bool,
    /// Armed reply composer target: a live agent or a saved session to resume on send.
    reply_target: Option<(String, SessionSummary)>,
    fd_path: Option<String>,
    creating_new_session: bool,
    reply_last_assistant_text: Option<String>,
    reply_last_assistant_text_loading: bool,
    reply_header_time: String,
    pending_delete_agent: Option<PendingDeleteAgent>,
    pending_kill_subagent: Option<PendingKillSubagent>,
    rename_target: Option<(Option<String>, Option<String>, SessionSummary)>,
    action_mode_search_query: Option<String>,
    /// Session the view was entered from; exempt from the empty-session sort demotion.
    anchor_session_id: Option<String>,
    inactive_agent_identities: HashSet<String>,
    saved_search_fetch_started: bool,
    /// `{ duringReconnect }` carried to the next `refreshSavedSessions` call.
    saved_search_fetch_pending: Option<bool>,
    /// `submit(text, delivery)` queued from a synchronous key handler.
    pending_submit: Option<(String, String)>,
    /// Mirrors the placeholder handed to the editor, for assertions.
    editor_placeholder_for_test: Option<String>,
    /// Deferred async actions: Rust cannot spawn a task that borrows `&mut self`,
    /// so `handleInput` records the intent and the run loop dispatches it.
    delete_selected_requested: bool,
    reply_toggle_requested: bool,
    create_session_requested: bool,
    status_message: Option<String>,
    status_message_tone: StatusTone,
    status_message_sticky: bool,
    stopped: bool,
    focused: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusTone {
    Muted,
    Error,
    Warning,
}

impl StatusTone {
    fn as_str(self) -> &'static str {
        match self {
            StatusTone::Muted => "muted",
            StatusTone::Error => "error",
            StatusTone::Warning => "warning",
        }
    }
}

impl<'a> AgentsViewMode<'a> {
    pub fn new(
        options: AgentsViewModeOptions,
        persistent_state: AgentsViewPersistentState,
        terminal: Arc<dyn AgentsViewTerminal>,
        editor: Box<dyn AgentsViewEditor>,
        theme: Arc<dyn AgentsViewTheme>,
        transport: Arc<dyn super::roster_store::DaemonTransport>,
        factory: &'a dyn DaemonAgentConnectionFactory,
        interactive_factory: &'a mut InteractiveModeFactory,
        recover_daemon: Option<Arc<dyn Fn() -> TransportFuture<Result<(), String>> + Send + Sync>>,
    ) -> Self {
        let initial_frames = persistent_state.scope_frames.clone().unwrap_or_else(|| {
            create_initial_agents_view_scope_frames(
                options.initial_scope_key.as_ref(),
                persistent_state
                    .back_session
                    .as_ref()
                    .or(options.initial_session.as_ref()),
            )
        });
        let anchor_session_id = persistent_state
            .back_session
            .as_ref()
            .or(options.initial_session.as_ref())
            .map(|session| session.session_id.clone());
        let scope_key = initial_frames.last().map(|frame| frame.scope.clone());
        let mut persistent_state = persistent_state;
        persistent_state.scope_frames = Some(initial_frames);
        let expanded_subagent_parents = persistent_state.expanded_subagent_parents.take().unwrap_or_default();
        let program_shown_parents = persistent_state.program_shown_parents.take().unwrap_or_default();
        let last_listed_summaries = persistent_state.last_successful_live_summaries.clone().unwrap_or_default();
        let saved_sessions = persistent_state.saved_sessions.clone().unwrap_or_default();
        let last_successful_saved_sessions = persistent_state
            .last_successful_saved_sessions
            .clone()
            .unwrap_or_else(|| saved_sessions.clone());
        let saved_catalog_ready = persistent_state.saved_catalog_loaded == Some(true);
        let heartbeats = persistent_state.heartbeats.clone().unwrap_or_default();
        let saved_catalog_generation = persistent_state.saved_catalog_generation.unwrap_or(0);
        let mut editor = editor;
        editor.set_text(persistent_state.query.clone().unwrap_or_default().as_str());
        editor.set_placeholder(SEARCH_PROMPT_PLACEHOLDER);
        let editor_placeholder_for_test = Some(SEARCH_PROMPT_PLACEHOLDER.to_string());
        terminal.set_title(crate::config::app_display_title());
        let mut mode = Self {
            selected_row_identity: persistent_state.selected_row_identity.clone(),
            selected_session_key: persistent_state.selected_session_key.clone(),
            selected_active_session_id: persistent_state
                .selected_session_key
                .as_ref()
                .and_then(|key| key.active_session_id.clone()),
            options,
            persistent_state,
            terminal,
            editor,
            theme,
            transport,
            #[cfg(test)]
            scripted_transport: None,
            factory,
            interactive_factory,
            recover_daemon,
            roster_store: Arc::new(AgentsViewRosterStore::new()),
            roster_listener: None,
            client: None,
            unsubscribe_client_close: None,
            unsubscribe_client_message: None,
            last_close_reason: Arc::new(Mutex::new(None)),
            reconnect_last_error: None,
            reconnect_started: false,
            reconnect_timed_out: false,
            daemon_shutdown_received: false,
            resolve_run: None,
            ctrl_c_exit_hint_expires_at: 0,
            delete_confirm_expires_at: 0,
            working_icon_frame: 0,
            rows: Vec::new(),
            all_rows: Vec::new(),
            show_actions: false,
            last_listed_summaries,
            last_visible_summaries: Vec::new(),
            saved_sessions,
            last_successful_saved_sessions,
            heartbeats,
            unified_records: Vec::new(),
            unified_index: super::agents_view_state::build_unified_session_index(&[]),
            scoped_records: Vec::new(),
            scope_key,
            scope_root_summary: None,
            saved_catalog_ready,
            saved_catalog_generation,
            heartbeat_catalog_generation: 0,
            saved_catalog_refresh_pending: false,
            saved_catalog_progress: 0,
            expanded_subagent_parents,
            program_shown_parents,
            selected_index: 0,
            selection_anchor_pending: false,
            reply_target: None,
            fd_path: None,
            creating_new_session: false,
            reply_last_assistant_text: None,
            reply_last_assistant_text_loading: false,
            reply_header_time: String::new(),
            pending_delete_agent: None,
            pending_kill_subagent: None,
            rename_target: None,
            action_mode_search_query: None,
            anchor_session_id,
            inactive_agent_identities: HashSet::new(),
            saved_search_fetch_started: false,
            saved_search_fetch_pending: None,
            pending_submit: None,
            editor_placeholder_for_test,
            delete_selected_requested: false,
            reply_toggle_requested: false,
            create_session_requested: false,
            status_message: None,
            status_message_tone: StatusTone::Muted,
            status_message_sticky: false,
            stopped: false,
            focused: false,
        };
        mode.scope_root_summary = mode.persistent_state.scope_root_summary.clone();
        mode
    }

    pub fn take_persistent_state(&mut self) -> AgentsViewPersistentState {
        self.persistent_state.expanded_subagent_parents = Some(self.expanded_subagent_parents.clone());
        self.persistent_state.program_shown_parents = Some(self.program_shown_parents.clone());
        self.persistent_state.saved_sessions = Some(self.saved_sessions.clone());
        self.persistent_state.last_successful_saved_sessions = Some(self.last_successful_saved_sessions.clone());
        self.persistent_state.last_successful_live_summaries = Some(self.last_listed_summaries.clone());
        self.persistent_state.heartbeats = Some(self.heartbeats.clone());
        self.persistent_state.saved_catalog_generation = Some(self.saved_catalog_generation);
        self.persistent_state.selected_row_identity = self.selected_row_identity.clone();
        self.persistent_state.selected_session_key = self.selected_session_key.clone();
        self.persistent_state.query = Some(self.editor.get_text());
        self.persistent_state.back_session = self.persistent_state.back_session.clone();
        self.persistent_state.status_message = self.status_message.clone();
        self.persistent_state.scope_root_summary = self.scope_root_summary.clone();
        std::mem::take(&mut self.persistent_state)
    }

    pub fn is_focused(&self) -> bool {
        self.focused
    }

    /// Test/driver hook: attach the roster store to the scripted transport and
    /// load its first snapshot, as `run()` does.
    #[cfg(test)]
    pub async fn attach_roster_for_test(&mut self) {
        self.roster_store = Arc::new(AgentsViewRosterStore::new());
        let socket_path = self.require_socket_path().expect("socket path");
        let client = connect_agents_view_daemon_client(&socket_path, self.transport.clone())
            .await
            .expect("transport connects");
        self.roster_store
            .attach(client.clone())
            .await
            .expect("roster attaches");
        self.client = Some(client);
        let summaries = self.roster_store.summaries().await;
        self.apply_session_list(summaries, true);
    }

    #[cfg(test)]
    pub fn transport_for_test(&self) -> Arc<dyn super::roster_store::DaemonTransport> {
        self.transport.clone()
    }

    /// Test/driver hook: script the next daemon response.
    #[cfg(test)]
    pub fn set_scripted_transport(&mut self, transport: Arc<tests::FakeTransport>) {
        self.scripted_transport = Some(transport);
    }

    #[cfg(test)]
    pub fn push_response(&mut self, response: DaemonResponse) {
        if let Some(fake) = &self.scripted_transport {
            fake.push(Ok(response));
        }
    }

    /// Test/driver hook: requests the mode has sent.
    #[cfg(test)]
    pub fn recorded_requests(&self) -> Vec<Value> {
        self.scripted_transport
            .as_ref()
            .map(|fake| fake.requests())
            .unwrap_or_default()
    }

    /// `onAgentsBack`: the result left-navigation produces for the current scope.
    /// The reference builds this inline in the editor callback; a named method
    /// keeps the same behaviour without an editor dependency.
    pub fn build_agents_back_result(&self) -> AgentsViewRunResult {
        let ancestors = match &self.scope_key {
            Some(scope) => get_unified_session_ancestor_session_ids(
                &self.unified_records,
                scope,
                Some(&self.unified_index),
            ),
            None => Vec::new(),
        };
        let return_chat = self
            .persistent_state
            .scope_frames
            .as_ref()
            .and_then(|frames| frames.last())
            .and_then(|frame| frame.return_chat.clone());
        match resolve_agents_view_left_result(
            self.scope_root_summary.as_ref(),
            ancestors,
            return_chat.as_ref(),
        ) {
            Some(result) => {
                let has_children = has_unified_session_children(
                    &self.unified_records,
                    &scope_key_from_selection(&get_agents_view_selection_key(&result.selection)),
                    Some(&self.unified_index),
                );
                AgentsViewRunResult::ScopeBack {
                    selection: result.selection,
                    expanded_ancestor_session_ids: result.expanded_ancestor_session_ids,
                    return_chat: result.return_chat,
                    has_children,
                }
            }
            None => AgentsViewRunResult::Exit,
        }
    }

    /// Test/driver hook: enter a scope with a return chat, exactly as the
    /// reference enters one at startup (`this.persistentState.scopeFrames`
    /// pushed by `resolveAgentsViewScopeFrames`, agents-view-mode.ts:2188-2194).
    ///
    /// It must not finish: in the reference the scope is entered long before any
    /// key is handled, and `finish()` runs only from the editor callbacks
    /// (`onAgentsBack` -> `this.finish(...)`, agents-view-mode.ts:807-815;
    /// `onEscape`, :820-841). Finishing here left the mode stopped, so a later
    /// keybinding could never reach `handleInput`.
    #[cfg(test)]
    pub fn enter_scope_for_test(&mut self, scope: AgentsViewScopeKey, return_chat: SessionSummary) {
        let frames = self.persistent_state.scope_frames.clone().unwrap_or_default();
        self.persistent_state.scope_frames = Some(transition_agents_view_scope(
            &frames,
            &super::agents_view_state::AgentsViewScopeAction::Push {
                scope: scope.clone(),
                return_chat: Some(return_chat),
            },
        ));
        self.scope_key = Some(scope);
        self.reconcile_catalogs();
    }

    #[cfg(test)]
    pub fn left_result_available(&self) -> bool {
        !matches!(self.build_agents_back_result(), AgentsViewRunResult::Open { .. })
    }


    #[cfg(test)]
    pub fn set_search_query_for_test(&mut self, query: &str) {
        self.set_search_query(query);
    }

    #[cfg(test)]
    pub fn select_row_for_test(&mut self, index: usize) {
        self.selected_index = index;
        self.sync_selected_row_state();
    }

    #[cfg(test)]
    pub async fn toggle_reply_target_for_test(&mut self) {
        self.toggle_reply_target().await;
    }

    #[cfg(test)]
    pub async fn press_delete(&mut self) {
        self.handle_delete_selected().await;
    }

    #[cfg(test)]
    pub fn reply_target_armed(&self) -> bool {
        self.reply_target.is_some()
    }

    #[cfg(test)]
    pub fn editor_placeholder(&self) -> String {
        self.editor_placeholder_for_test.clone().unwrap_or_default()
    }

    #[cfg(test)]
    pub fn editor_text(&self) -> String {
        self.editor.get_text()
    }

    #[cfg(test)]
    pub fn delete_confirmation_armed(&self) -> bool {
        self.is_delete_confirmation_visible()
    }

    #[cfg(test)]
    pub fn rename_mode_active(&self) -> bool {
        self.rename_target.is_some()
    }

    #[cfg(test)]
    pub fn exit_hint_visible(&self) -> bool {
        self.is_ctrl_c_exit_hint_visible()
    }

    #[cfg(test)]
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    #[cfg(test)]
    pub fn new_session_requested(&self) -> bool {
        self.create_session_requested
    }

    #[cfg(test)]
    pub fn heartbeat_count(&self) -> usize {
        self.heartbeats.len()
    }

    #[cfg(test)]
    pub fn daemon_shutdown_received(&self) -> bool {
        self.daemon_shutdown_received
    }

    #[cfg(test)]
    pub fn status_message_is_sticky(&self) -> bool {
        self.status_message_sticky
    }

    /// Test/driver hook: run the deferred async action queue once, as the run
    /// loop's pending ticker does.
    pub async fn flush_pending_actions(&mut self) {
        self.dispatch_pending_actions().await;
    }

    /// Test/driver hook: render the full view for a width.
    pub fn render_view(&mut self, width: usize) -> Vec<String> {
        self.render(width)
    }

    /// Test/driver hook: current visible rows.
    pub fn rows(&self) -> &[AgentsViewRow] {
        &self.rows
    }

    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    pub fn status_message(&self) -> Option<&str> {
        self.status_message.as_deref()
    }

    /// `run(): Promise<AgentsViewRunResult>`.
    ///
    /// The reference arms two `setInterval` timers (`refreshHeartbeats` every
    /// HEARTBEAT_POLL_INTERVAL_MS, the working-icon/animation tick every
    /// WORKING_ICON_INTERVAL_MS) and resolves a promise from `finish`. Rust cannot
    /// spawn a task that borrows `&mut self`, so the same cadence is driven by a
    /// `select!` loop over the completion channel and the two tickers: identical
    /// ordering, no background mutation.
    pub async fn run(&mut self) -> Result<AgentsViewRunResult, String> {
        let socket_path = self.require_socket_path()?;
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        self.resolve_run = Some(tx);
        // Paint the cached list before any daemon/catalog round trip. Keep one
        // terminal owner polling keys while read-only refreshes are outstanding.
        self.reconcile_catalogs();
        self.load_startup_notices();
        self.editor.focus();
        self.focused = true;
        if let Some(message) = self.persistent_state.status_message.take() {
            self.set_status_message(Some(&message), false, None, false);
        }
        self.present_current_view()?;
        let client = match self.persistent_state.roster_client.clone() {
            Some(client) => client,
            None => {
                let transport = self.transport.clone();
                let Some(client) = self.wait_with_input(connect_agents_view_daemon_client(&socket_path, transport)).await? else {
                    return rx.await.map_err(|_| "agents view ended during connection".to_string());
                };
                let client = client?;
                self.persistent_state.roster_client = Some(client.clone());
                client
            }
        };
        if !client.is_connected() {
            if self.wait_with_input(client.reconnect(1000)).await?.is_none() {
                return rx.await.map_err(|_| "agents view ended during reconnect".to_string());
            }
        }
        let heartbeats_changed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let heartbeats_changed_listener = heartbeats_changed.clone();
        let unsubscribe_message = client.on_message(Box::new(move |message| {
            if matches!(message, super::roster_store::DaemonOutbound::HeartbeatsChanged) {
                heartbeats_changed_listener.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }));
        self.unsubscribe_client_message = Some(unsubscribe_message);
        self.client = Some(client.clone());

        let store = self.roster_store.clone();
        let Some(attached) = self.wait_with_input(store.attach(client.clone())).await? else {
            store.dispose().await;
            return rx.await.map_err(|_| "agents view ended during roster refresh".to_string());
        };
        if !attached? {
            return Err(STALE_ROSTER_DAEMON_MESSAGE.to_string());
        }
        self.subscribe_to_client_close(client.clone());

        let roster_changed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let roster_changed_listener = roster_changed.clone();
        let listener_index = self.roster_store.on_update(Arc::new(move || {
            roster_changed_listener.store(true, std::sync::atomic::Ordering::SeqCst);
        })).await;
        self.roster_listener = Some(listener_index);
        let summaries = self.roster_store.summaries().await;
        self.apply_session_list(summaries, true);
        self.arm_saved_search_fetch(false);
        self.resolve_missing_selection_anchor();
        let _ = self.refresh_heartbeats(false).await;
        self.terminal.request_render(true);
        let width = self.terminal.columns();
        let lines = self.render(width);
        self.terminal.present(lines, self.render_dock(width))?;

        let mut heartbeat_ticker =
            tokio::time::interval(std::time::Duration::from_millis(HEARTBEAT_POLL_INTERVAL_MS));
        heartbeat_ticker.tick().await;
        let mut animation_ticker =
            tokio::time::interval(std::time::Duration::from_millis(WORKING_ICON_INTERVAL_MS));
        animation_ticker.tick().await;
        // The reference starts these as fire-and-forget promises from synchronous
        // key handlers (`void this.handleDeleteSelected()`). A Rust key handler
        // cannot await, so it records the intent and this ticker drains it.
        let mut pending_ticker = tokio::time::interval(std::time::Duration::from_millis(10));
        pending_ticker.tick().await;

        loop {
            tokio::select! {
                result = &mut rx => {
                    self.roster_store.dispose().await;
                    return result.map_err(|_| "agents view run loop ended unexpectedly".to_string());
                }
                _ = heartbeat_ticker.tick() => {
                    let _ = heartbeats_changed.swap(false, std::sync::atomic::Ordering::SeqCst);
                    let _ = self.refresh_heartbeats(false).await;
                }
                _ = animation_ticker.tick() => {
                    self.tick_animation();
                }
                _ = pending_ticker.tick() => {
                    match self.terminal.poll_input()? {
                        None => self.finish(AgentsViewRunResult::Exit),
                        Some(input) => for data in input {
                            self.handle_input(&data);
                            if self.stopped { break; }
                        },
                    }
                    if self.stopped { continue; }
                    for text in self.editor.take_submissions() {
                        self.submit(&text, "steer").await;
                    }
                    if self.stopped { continue; }
                    if roster_changed.swap(false, std::sync::atomic::Ordering::SeqCst) {
                        self.on_roster_update().await;
                    }
                    if heartbeats_changed.swap(false, std::sync::atomic::Ordering::SeqCst) {
                        let _ = self.refresh_heartbeats(false).await;
                    }
                    let close_reason = self.last_close_reason.lock().expect("close reason poisoned").take();
                    if let Some(reason) = close_reason {
                        if reason.contains("shutdown") { self.handle_daemon_shutdown(&reason); }
                        else { self.start_client_reconnect(&reason); }
                    }
                    if self.reconnect_started {
                        let error = self.reconnect_last_error.clone().unwrap_or_default();
                        self.reconnect_client(&error).await?;
                    }
                    self.dispatch_pending_actions().await;
                    if self.stopped { continue; }
                    let width = self.terminal.columns();
                    let lines = self.render(width);
                    self.terminal.present(lines, self.render_dock(width))?;
                }
            }
        }
    }

    fn present_current_view(&mut self) -> Result<(), String> {
        let width = self.terminal.columns();
        let lines = self.render(width);
        self.terminal.present(lines, self.render_dock(width))
    }

    async fn wait_with_input<T>(&mut self, pending: impl std::future::Future<Output = T>) -> Result<Option<T>, String> {
        self.wait_with_input_progress(pending, None).await
    }

    async fn wait_with_input_progress<T>(
        &mut self,
        pending: impl std::future::Future<Output = T>,
        progress: Option<Arc<std::sync::atomic::AtomicI64>>,
    ) -> Result<Option<T>, String> {
        tokio::pin!(pending);
        let mut ticker = tokio::time::interval(std::time::Duration::from_millis(16));
        let mut animation_at = tokio::time::Instant::now();
        let mut progress_at = animation_at;
        loop {
            tokio::select! {
                result = &mut pending => return Ok(Some(result)),
                _ = ticker.tick() => {
                    match self.terminal.poll_input()? {
                        None => self.finish(AgentsViewRunResult::Exit),
                        Some(input) => for data in input {
                            self.handle_input(&data);
                            if self.stopped { break; }
                        },
                    }
                    if self.stopped { return Ok(None); }
                    if animation_at.elapsed().as_millis() >= WORKING_ICON_INTERVAL_MS as u128 {
                        self.tick_animation();
                        animation_at = tokio::time::Instant::now();
                    }
                    if progress_at.elapsed().as_millis() >= 100 {
                        if let Some(progress) = progress.as_ref() {
                            self.saved_catalog_progress = progress.load(std::sync::atomic::Ordering::Relaxed);
                        }
                        progress_at = tokio::time::Instant::now();
                    }
                    self.present_current_view()?;
                }
            }
        }
    }

    /// The animation interval body: age labels are baked into rows at build time,
    /// so ticking them needs a rebuild.
    pub fn tick_animation(&mut self) {
        let has_running = self.rows.iter().any(|row| row.section == AgentsViewSection::Running);
        let has_stale_age = self.rows.iter().any(|row| row.summary.last_heard_from_at.is_some());
        if !has_running && !has_stale_age {
            return;
        }
        if has_stale_age {
            self.rebuild_rows();
        }
        if has_running {
            self.working_icon_frame += 1;
        }
        self.terminal.request_render(false);
    }

    fn persistent_state_client(&self) -> Option<DaemonTransportClient> {
        None
    }

    /// `finish(result)`.
    pub fn finish(&mut self, result: AgentsViewRunResult) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        self.saved_catalog_generation += 1;
        self.heartbeat_catalog_generation += 1;
        // The `select!` loop in run() owns the timers; finish() only stops the view.
        self.clear_ctrl_c_exit_hint(false);
        self.clear_delete_confirmation(false);
        self.set_status_message(None, false, None, false);
        if let Some(unsubscribe) = self.unsubscribe_client_close.take() {
            unsubscribe();
        }
        if let Some(unsubscribe) = self.unsubscribe_client_message.take() {
            unsubscribe();
        }
        self.client = None;
        if let Some(resolve) = self.resolve_run.take() {
            let _ = resolve.send(result);
        }
    }

    fn require_socket_path(&self) -> Result<String, String> {
        self.options
            .socket_path
            .clone()
            .ok_or_else(|| "Session view daemon socket is not configured".to_string())
    }

    fn require_client(&self) -> Result<DaemonTransportClient, String> {
        self.client
            .clone()
            .ok_or_else(|| "Agents view daemon client is not connected".to_string())
    }

    /// `subscribeToClientClose(client)`.
    fn subscribe_to_client_close(&mut self, client: DaemonTransportClient) {
        if let Some(unsubscribe) = self.unsubscribe_client_close.take() {
            unsubscribe();
        }
        let reason_holder = Arc::new(Mutex::new(None::<String>));
        let holder = reason_holder.clone();
        let unsubscribe = client.on_close(Box::new(move |reason| {
            *holder.lock().expect("close reason poisoned") = Some(reason.to_string());
        }));
        self.unsubscribe_client_close = Some(unsubscribe);
        // The reference dispatches on `getDaemonSocketCloseReason(error)`; the
        // close listener here records the reason so the run loop can read it.
        self.last_close_reason = reason_holder;
    }

    /// `handleDaemonShutdown(client, error)`.
    pub fn handle_daemon_shutdown(&mut self, message: &str) {
        if self.stopped {
            return;
        }
        self.daemon_shutdown_received = true;
        self.reconnect_timed_out = false;
        self.set_status_message(
            Some(&format!(
                "Prime Agent daemon shut down. Restart Prime Agent to reconnect. {message}"
            )),
            false,
            Some(StatusTone::Error),
            true,
        );
        self.apply_session_list(Vec::new(), false);
    }

    /// `startClientReconnect(client, error)`.
    pub fn start_client_reconnect(&mut self, error: &str) {
        if self.stopped || self.reconnect_started || self.daemon_shutdown_received {
            return;
        }
        if !self.reconnect_timed_out {
            self.set_status_message(Some("Daemon connection lost; reconnecting…"), false, Some(StatusTone::Warning), true);
        }
        self.reconnect_started = true;
        self.reconnect_last_error = Some(error.to_string());
    }

    /// `reconnectClient(client, initialError)`: bounded retry loop.
    pub async fn reconnect_client(&mut self, initial_error: &str) -> Result<(), String> {
        let deadline = super::agents_view_state::now_ms()
            + (self.options.reconnect_timeout_ms.unwrap_or(RECONNECT_TIMEOUT_MS) as i64);
        let client = self.require_client()?;
        let mut last_error = initial_error.to_string();
        while !self.stopped
            && !self.daemon_shutdown_received
            && super::agents_view_state::now_ms() < deadline
        {
            let recover = self.recover_daemon.clone();
            let reconnect = async {
                if let Some(recover) = recover { recover().await?; }
                client.reconnect(1000).await
            };
            let Some(reconnected) = self.wait_with_input(reconnect).await? else { return Ok(()); };
            let attempt = match reconnected {
                Ok(()) => self.finish_reconnect_attempt(&client).await,
                Err(error) => Err(error),
            };
            match attempt {
                Ok(()) => return Ok(()),
                Err(error) => last_error = error,
            }
            if self.wait_with_input(tokio::time::sleep(std::time::Duration::from_millis(RECONNECT_RETRY_MS))).await?.is_none() {
                return Ok(());
            }
        }
        if !self.stopped && !self.daemon_shutdown_received {
            self.reconnect_timed_out = true;
            self.set_status_message(
                Some(&format_error("Daemon unavailable; retrying", &last_error)),
                false,
                Some(StatusTone::Error),
                true,
            );
        }
        self.reconnect_started = false;
        Ok(())
    }

    async fn finish_reconnect_attempt(&mut self, client: &DaemonTransportClient) -> Result<(), String> {
        let store = self.roster_store.clone();
        let Some(attached) = self.wait_with_input(store.attach(client.clone())).await? else { return Ok(()); };
        if !attached? {
            return Err("Daemon lost the agent_roster capability during reconnect".to_string());
        }
        let heartbeats_refreshed = self.refresh_heartbeats(true).await;
        if !heartbeats_refreshed && !client.is_connected() {
            return Err("Heartbeat catalog did not refresh during reconnect".to_string());
        }
        let sessions = self.roster_store.summaries().await;
        self.daemon_shutdown_received = false;
        self.reconnect_timed_out = false;
        self.reconnect_started = false;
        if heartbeats_refreshed {
            self.set_status_message(Some("Daemon reconnected"), false, None, false);
        } else {
            self.set_status_message(Some("Daemon reconnected; scheduled-task coverage is incomplete, retrying"), false, Some(StatusTone::Warning), false);
        }
        self.apply_session_list(sessions, true);
        self.arm_saved_search_fetch(true);
        Ok(())
    }

    /// `applySessionList(sessions, successful)`.
    pub fn apply_session_list(&mut self, sessions: Vec<SessionSummary>, successful: bool) {
        self.last_listed_summaries = sessions.clone();
        if successful {
            self.persistent_state.last_successful_live_summaries = Some(sessions);
        }
        self.reconcile_catalogs();
    }

    /// `reconcileCatalogs()`.
    pub fn reconcile_catalogs(&mut self) {
        let manually_inactive = self.inactive_agent_identities.clone();
        let visible_sessions: Vec<SessionSummary> = self
            .last_listed_summaries
            .iter()
            .filter(|summary| {
                should_show_agents_view_session(
                    summary,
                    manually_inactive.contains(&get_agents_view_summary_identity(summary)),
                )
            })
            .cloned()
            .collect();
        self.last_visible_summaries = self.with_pending_delete_session(&visible_sessions);
        self.unified_records = reconcile_unified_sessions(
            &self.last_visible_summaries,
            &self.saved_sessions,
            &self.heartbeats,
        );
        self.unified_index = super::agents_view_state::build_unified_session_index(&self.unified_records);
        migrate_agents_view_identity_set(&mut self.expanded_subagent_parents, &self.unified_index.by_key);
        migrate_agents_view_identity_set(&mut self.program_shown_parents, &self.unified_index.by_key);

        let frames = self.persistent_state.scope_frames.clone().unwrap_or_default();
        let resolution = resolve_agents_view_scope_frames(&self.unified_records, &frames, Some(&self.unified_index));
        if should_apply_scope_resolution(resolution.dropped_frames, self.saved_catalog_ready) {
            self.persistent_state.scope_frames = Some(resolution.frames.clone());
            self.scope_key = resolution.frames.last().map(|frame| frame.scope.clone());
            self.scope_root_summary = resolution.root.as_ref().map(summary_for_unified_record);
            self.persistent_state.scope_root_summary = self.scope_root_summary.clone();
            if resolution.dropped_frames > 0 {
                let destination = if resolution.root.is_some() {
                    "the nearest available parent"
                } else {
                    "the global view"
                };
                self.set_status_message(
                    Some(&format!("Scope is no longer available; returned to {destination}")),
                    false,
                    None,
                    false,
                );
            }
        }
        self.scoped_records =
            scope_to_session_subtree(&self.unified_records, self.scope_key.as_ref(), Some(&self.unified_index));
        self.rebuild_rows();
        self.apply_pending_ancestor_expansion();
        self.restore_selection();
        self.terminal.request_render(false);
    }

    /// `rebuildRows()`: rebuild from the last fetched summaries, keeping
    /// selection on the same row.
    pub fn rebuild_rows(&mut self) {
        let selected_identity = self.rows.get(self.selected_index).map(|row| row.identity.clone());
        let filtered = self.get_filtered_records();
        let rollups = compute_recursive_rollups(&self.unified_records, Some(&self.unified_index));
        let inputs: Vec<AgentsViewRowInput> = filtered.into_iter().map(AgentsViewRowInput::Record).collect();
        self.all_rows = build_agents_view_rows(
            &inputs,
            &self.expanded_subagent_parents,
            &self.program_shown_parents,
            self.scope_key.as_ref(),
            Some(&rollups),
            self.anchor_session_id.as_deref(),
        );
        let show_inactive = is_inactive_expanded(&self.persistent_state) || self.action_search_text().trim().len() > 0;
        self.rows = compact_session_rows(&self.all_rows, show_inactive);
        match selected_identity {
            Some(identity) => {
                let index = self.rows.iter().position(|row| row.identity == identity);
                match index {
                    Some(index) => self.selected_index = index,
                    None => self.restore_selection(),
                }
            }
            None => self.restore_selection(),
        }
    }

    /// The text the list filter uses: the search query, or the preserved query
    /// while the reply composer or rename mode owns the editor.
    fn action_search_text(&self) -> String {
        if self.reply_target.is_some() || self.rename_target.is_some() {
            self.action_mode_search_query.clone().unwrap_or_default()
        } else {
            self.editor.get_text()
        }
    }

    /// `getFilteredRecords()`.
    pub fn get_filtered_records(&self) -> Vec<UnifiedSessionRecord> {
        let query = self.action_search_text();
        filter_unified_sessions(&self.scoped_records, &|text| matches_search_text(text, &query))
    }

    /// `refreshHeartbeats(options)`.
    pub async fn refresh_heartbeats(&mut self, during_reconnect: bool) -> bool {
        if (!during_reconnect && self.reconnect_started) || self.daemon_shutdown_received {
            return false;
        }
        let generation = self.heartbeat_catalog_generation + 1;
        self.heartbeat_catalog_generation = generation;
        let client = match self.require_client() {
            Ok(client) => client,
            Err(_) => return false,
        };
        let pending = list_daemon_heartbeats(&client, None);
        let result = match self.wait_with_input(pending).await {
            Ok(Some(result)) => result,
            Ok(None) => return false,
            Err(error) => Err(error),
        };
        match result {
            Ok(heartbeats) => {
                if generation != self.heartbeat_catalog_generation {
                    return false;
                }
                self.heartbeats = heartbeats.clone();
                self.persistent_state.heartbeats = Some(heartbeats);
                self.reconcile_catalogs();
                if self.status_message.as_deref().is_some_and(|message|
                    message.starts_with("Failed to refresh heartbeats:")
                        || message == "Scheduled tasks are still loading; coverage is incomplete, retrying"
                        || message == "Daemon reconnected; scheduled-task coverage is incomplete, retrying") {
                    self.set_status_message(None, true, None, false);
                }
                true
            }
            Err(error) => {
                if generation == self.heartbeat_catalog_generation && !self.reconnect_started {
                    let connected = self.client.as_ref().map(|client| client.is_connected()).unwrap_or(false);
                    if !connected {
                        self.start_client_reconnect(&error);
                    } else if !self.status_message_sticky {
                        let pending = error == "Cannot list heartbeats while session worker is starting"
                            || error == "Cannot list heartbeats while session worker is recovering";
                        let message = if pending {
                            "Scheduled tasks are still loading; coverage is incomplete, retrying".to_string()
                        } else {
                            format_error("Failed to refresh heartbeats", &error)
                        };
                        self.set_status_message(Some(&message), true, pending.then_some(StatusTone::Warning), false);
                    }
                }
                false
            }
        }
    }

    /// `armSavedSearchFetch(options)`.
    pub fn arm_saved_search_fetch(&mut self, during_reconnect: bool) {
        // The inactive section is catalog-fed, so no query gate: load on view open.
        if self.saved_search_fetch_started || self.persistent_state.saved_catalog_loaded == Some(true) {
            return;
        }
        self.saved_search_fetch_started = true;
        self.saved_search_fetch_pending = Some(during_reconnect);
    }

    /// `refreshSavedSessions(options)`.
    pub async fn refresh_saved_sessions(&mut self, during_reconnect: bool, preserve_status_on_error: bool) -> bool {
        if self.stopped || (!during_reconnect && self.reconnect_started) || self.daemon_shutdown_received {
            self.rearm_saved_search_fetch();
            return false;
        }
        let generation = self.saved_catalog_generation + 1;
        self.saved_catalog_generation = generation;
        self.persistent_state.saved_catalog_generation = Some(generation);
        self.saved_catalog_refresh_pending = true;
        self.saved_catalog_ready = false;
        let successful_sessions = self.last_successful_saved_sessions.clone();
        self.saved_catalog_progress = 0;
        self.terminal.request_render(false);
        let client = match self.require_client() {
            Ok(client) => client,
            Err(error) => {
                if generation == self.saved_catalog_generation {
                    self.saved_catalog_refresh_pending = false;
                }
                self.set_status_message(Some(&format_error("Failed to load saved sessions", &error)), true, None, false);
                return false;
            }
        };
        let context = self.get_saved_session_catalog_context();
        let progress_cell = Arc::new(std::sync::atomic::AtomicI64::new(0));
        let progress_for_callback = progress_cell.clone();
        let on_session: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            // Rebuilding/canonicalizing the full tree per streamed record is
            // quadratic, particularly expensive on Windows. Keep the last
            // complete catalog visible.
            progress_for_callback.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        let pending = list_daemon_saved_sessions(&client, &context, "all", Some(on_session), None);
        let result = match self.wait_with_input_progress(pending, Some(progress_cell)).await {
            Ok(Some(result)) => result,
            Ok(None) => { self.saved_catalog_refresh_pending = false; return false; }
            Err(error) => Err(error),
        };
        match result {
            Ok(sessions) => {
                if generation != self.saved_catalog_generation || self.stopped || self.daemon_shutdown_received {
                    return false;
                }
                self.saved_sessions = sessions.clone();
                self.last_successful_saved_sessions = sessions.clone();
                self.saved_catalog_ready = true;
                self.persistent_state.last_successful_saved_sessions = Some(sessions.clone());
                self.persistent_state.saved_sessions = Some(sessions);
                self.persistent_state.saved_catalog_loaded = Some(true);
                self.reconcile_catalogs();
                self.saved_catalog_refresh_pending = false;
                true
            }
            Err(error) => {
                if generation == self.saved_catalog_generation && !self.stopped && !self.daemon_shutdown_received {
                    self.saved_sessions = successful_sessions.clone();
                    self.persistent_state.saved_sessions = Some(successful_sessions);
                    // Treat a terminal failure as settled so scope fallback cannot soft-lock.
                    self.saved_catalog_ready = true;
                    self.rearm_saved_search_fetch();
                    self.reconcile_catalogs();
                    if (!preserve_status_on_error || self.persistent_state.saved_catalog_loaded != Some(true))
                        && !self.reconnect_started
                    {
                        self.set_status_message(
                            Some(&format_error("Failed to load saved sessions", &error)),
                            true,
                            None,
                            false,
                        );
                    }
                }
                self.saved_catalog_refresh_pending = false;
                false
            }
        }
    }

    fn rearm_saved_search_fetch(&mut self) {
        if self.persistent_state.saved_catalog_loaded != Some(true) {
            self.saved_search_fetch_started = false;
        }
    }

    fn refresh_saved_sessions_if_loaded(&mut self) {
        if self.persistent_state.saved_catalog_loaded == Some(true) {
            self.saved_search_fetch_pending = Some(false);
        }
    }

    fn get_saved_session_cwd(&self) -> String {
        self.options
            .config
            .cwd
            .clone()
            .unwrap_or_else(|| self.options.ui_services.get_initial_cwd())
    }

    fn get_saved_session_catalog_context(&self) -> DaemonSavedSessionCatalogContext {
        DaemonSavedSessionCatalogContext::Cwd {
            cwd: self.get_saved_session_cwd(),
            session_dir: self.options.config.session_dir.clone(),
        }
    }

    /// `withPendingDeleteSession(sessions)`.
    fn with_pending_delete_session(&self, sessions: &[SessionSummary]) -> Vec<SessionSummary> {
        let Some(pending) = &self.pending_delete_agent else {
            return sessions.to_vec();
        };
        // Saved-only rows already come from the durable catalog. Injecting their
        // synthetic archived summary as a daemon record would move confirmation
        // from Inactive to Idle.
        if pending.summary.lifecycle != SessionLifecycle::Live || !self.is_delete_confirmation_visible() {
            return sessions.to_vec();
        }
        let mut replaced = false;
        let merged: Vec<SessionSummary> = sessions
            .iter()
            .map(|summary| {
                if get_agents_view_summary_identity(summary) != pending.identity {
                    return summary.clone();
                }
                replaced = true;
                pending.summary.clone()
            })
            .collect();
        if replaced {
            merged
        } else {
            let mut merged = merged;
            merged.push(pending.summary.clone());
            merged
        }
    }

    /// `resolveMissingSelectionAnchor()`.
    fn resolve_missing_selection_anchor(&mut self) {
        if !self.selection_anchor_pending || self.saved_catalog_refresh_pending {
            return;
        }
        self.selection_anchor_pending = false;
        let row = self.rows.get(self.selected_index);
        self.selected_active_session_id = row
            .filter(|row| row.selectable)
            .map(|row| row.summary.active_session_id.clone().unwrap_or_else(|| row.summary.id.clone()));
    }

    /// `restoreSelection()`.
    fn restore_selection(&mut self) {
        if self.rows.is_empty() {
            self.selected_index = 0;
            self.selected_active_session_id = None;
            return;
        }
        let identity = self
            .selected_row_identity
            .clone()
            .or_else(|| self.persistent_state.selected_row_identity.clone());
        let key = self
            .selected_session_key
            .clone()
            .or_else(|| self.persistent_state.selected_session_key.clone());
        let resolution =
            resolve_agents_view_selection_state(&self.rows, self.selected_index, identity.as_deref(), key.as_ref());
        self.selected_index = resolution.index;
        if resolution.resolved {
            self.sync_selected_row_state();
            return;
        }
        self.selection_anchor_pending = self.selected_row_identity.is_some()
            || self.persistent_state.selected_row_identity.is_some()
            || self.selected_session_key.is_some()
            || self.persistent_state.selected_session_key.is_some();
        // Catalogs stream independently. Show a temporary fallback row without
        // replacing the source-session anchor before its daemon row arrives.
        let fallback = self.rows.get(self.selected_index);
        self.selected_active_session_id = fallback
            .filter(|row| row.selectable)
            .map(|row| row.summary.active_session_id.clone().unwrap_or_else(|| row.summary.id.clone()));
    }

    /// `syncSelectedRowState()`.
    fn sync_selected_row_state(&mut self) {
        self.selection_anchor_pending = false;
        let row = self.rows.get(self.selected_index);
        self.selected_active_session_id = row
            .filter(|row| row.selectable)
            .map(|row| row.summary.active_session_id.clone().unwrap_or_else(|| row.summary.id.clone()));
        self.selected_row_identity = get_selected_row_identity(row);
        self.selected_session_key = row.filter(|row| row.selectable).map(|row| get_agents_view_selection_key(&row.summary));
        self.persistent_state.selected_row_identity = self.selected_row_identity.clone();
        self.persistent_state.selected_session_key = self.selected_session_key.clone();
    }

    /// `applyPendingAncestorExpansion()`.
    fn apply_pending_ancestor_expansion(&mut self) {
        let Some(session_ids) = self.persistent_state.pending_expanded_ancestor_session_ids.take() else {
            return;
        };
        if session_ids.is_empty() {
            return;
        }
        let wanted: HashSet<String> = session_ids.into_iter().collect();
        // A nested ancestor's row only appears once its own parent is expanded, so
        // expand-and-rebuild until a pass reveals nothing new.
        let mut added = true;
        while added {
            added = false;
            for row in self.rows.clone() {
                if wanted.contains(&row.summary.session_id) && !self.expanded_subagent_parents.contains(&row.identity)
                {
                    self.expanded_subagent_parents.insert(row.identity.clone());
                    added = true;
                }
            }
            if added {
                self.rebuild_rows();
            }
        }
    }

    /// `moveSelection(delta)`.
    pub fn move_selection(&mut self, delta: i64) {
        let selectable_indexes = self.get_selectable_row_indexes();
        if selectable_indexes.is_empty() {
            return;
        }
        let current_position = selectable_indexes
            .iter()
            .position(|index| *index == self.selected_index)
            .unwrap_or(0) as i64;
        let next_position = (current_position + delta).clamp(0, selectable_indexes.len() as i64 - 1);
        self.selected_index = selectable_indexes[next_position as usize];
        self.sync_selected_row_state();
        self.clear_delete_confirmation(false);
        // Reply stays armed only while the selection sits on the agent row it
        // targets; nested rows share the parent's session id but are read-only.
        let selected_row = self.rows.get(self.selected_index).cloned();
        if self.reply_target.is_some()
            && (selected_row.as_ref().map(|row| row.kind) != Some(AgentsViewRowKind::Agent)
                || self.reply_target.as_ref().map(|(key, _)| key.as_str()) != self.selected_active_session_id.as_deref())
        {
            self.set_reply_target(None);
        }
        self.terminal.request_render(false);
    }

    fn get_selectable_row_indexes(&self) -> Vec<usize> {
        self.rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.selectable)
            .map(|(index, _)| index)
            .collect()
    }

    /// `handleInput(data)`.
    pub fn handle_input(&mut self, data: &str) {
        self.clear_sticky_status_message();
        if pi_tui::keybindings::get_keybindings().matches(data, "app.exit") && self.editor.get_text().is_empty() {
            self.finish(AgentsViewRunResult::Exit);
            return;
        }
        if matches_key(data, "app.agents.back") {
            if self.reply_target.is_some() { self.set_reply_target(None); return; }
            if self.editor.get_text().is_empty() {
                // The global view has no hierarchy parent: consume Left.
                if self.scope_root_summary.is_some() { self.finish(self.build_agents_back_result()); }
                return;
            }
        }
        if matches_key(data, "app.input.clear") && self.rename_target.is_none() && !self.show_actions {
            if self.reply_target.is_some() { self.set_reply_target(None); }
            else if !self.editor.get_text().is_empty() { self.set_search_query(""); }
            else if let Some(summary) = self.persistent_state.back_session.clone() {
                let has_children = has_unified_session_children(&self.unified_records,
                    &scope_key_from_selection(&get_agents_view_selection_key(&summary)), Some(&self.unified_index));
                self.finish(AgentsViewRunResult::Open { summary, selection: None,
                    expanded_ancestor_session_ids: None, has_children: Some(has_children), status_message: None });
            } else { self.finish(AgentsViewRunResult::Exit); }
            return;
        }
        if self.show_actions {
            self.show_actions = false;
            self.terminal.request_render(false);
            if matches_key(data, "app.shortcuts") || matches_key(data, "tui.select.cancel") {
                return;
            }
        }
        if self.rename_target.is_some() {
            if matches_key(data, "tui.select.cancel") {
                self.exit_rename_mode();
                return;
            }
            self.editor.handle_input(data);
            return;
        }
        if matches_key(data, "app.clear") {
            // The composer hints advertise ctrl+c as cancel; it must not start the exit flow.
            if self.reply_target.is_some() {
                self.set_reply_target(None);
                return;
            }
            self.handle_ctrl_c();
            return;
        }
        if self.editor.get_text().is_empty() && matches_key(data, "app.agents.rename") {
            self.enter_rename_mode();
            return;
        }
        if self.editor.get_text().is_empty() && matches_key(data, "app.agents.delete") {
            self.clear_ctrl_c_exit_hint(false);
            self.delete_selected_requested = true;
            return;
        }
        self.clear_ctrl_c_exit_hint(false);
        self.clear_delete_confirmation(false);
        if matches_key(data, "app.agents.reply") && self.editor.get_text().is_empty() {
            self.reply_toggle_requested = true;
            return;
        }
        if self.reply_target.is_none() && matches_key(data, "app.agents.new") {
            self.create_session_requested = true;
            return;
        }
        if self.reply_target.is_some() && matches_key(data, "app.message.followUp") {
            self.handle_reply_follow_up();
            return;
        }
        if self.editor.get_text().is_empty() && matches_key(data, "app.agents.program") {
            self.cycle_program_for_selected();
            return;
        }
        if self.reply_target.is_none() && self.editor.get_text().is_empty() {
            if matches_key(data, "app.shortcuts") {
                self.show_actions = !self.show_actions;
                self.terminal.request_render(false);
                return;
            }
            if matches_key(data, "app.agents.inactiveCollapse") {
                self.persistent_state.inactive_expanded = Some(!is_inactive_expanded(&self.persistent_state));
                self.persistent_state.inactive_visibility_explicit = Some(true);
                self.rebuild_rows();
                self.sync_selected_row_state();
                self.terminal.request_render(false);
                return;
            }
            if matches_key(data, "app.agents.expand") {
                let row = self.rows.get(self.selected_index).cloned();
                if let Some(row) = row {
                    if row.descendant_count > 0 {
                        self.toggle_subagent_list(&row);
                    }
                }
                return;
            }
        }
        if self.reply_target.is_none() && matches_key(data, "app.agents.open") {
            if self.editor.get_text().is_empty() || self.is_search_cursor_at_end() {
                self.open_selected();
                return;
            }
        }
        if self.reply_target.is_none() && self.handle_list_navigation(data) {
            return;
        }
        let previous = self.editor.get_text();
        self.editor.handle_input(data);
        if self.reply_target.is_none() && self.editor.get_text() != previous {
            self.query_changed();
        }
    }

    fn is_search_cursor_at_end(&self) -> bool {
        let lines = self.editor.get_lines();
        let (line, col) = self.editor.get_cursor();
        line == lines.len().saturating_sub(1)
            && col == lines.get(line).map(|value| value.chars().count()).unwrap_or(0)
    }

    /// `handleListNavigation(data)`.
    fn handle_list_navigation(&mut self, data: &str) -> bool {
        if matches_key(data, "tui.select.up") {
            self.move_selection(-1);
            return true;
        }
        if matches_key(data, "tui.select.down") {
            self.move_selection(1);
            return true;
        }
        if matches_key(data, "tui.select.pageUp") {
            self.move_selection(-(self.visible_list_rows().max(1) as i64));
            return true;
        }
        if matches_key(data, "tui.select.pageDown") {
            self.move_selection(self.visible_list_rows().max(1) as i64);
            return true;
        }
        false
    }

    fn handle_ctrl_c(&mut self) {
        if self.is_ctrl_c_exit_hint_visible() {
            self.finish(AgentsViewRunResult::Exit);
            return;
        }
        self.show_ctrl_c_exit_hint();
    }

    fn show_ctrl_c_exit_hint(&mut self) {
        self.ctrl_c_exit_hint_expires_at = super::agents_view_state::now_ms() + EXIT_HINT_DURATION_MS as i64;
        self.terminal.request_render(false);
    }

    fn clear_ctrl_c_exit_hint(&mut self, render: bool) {
        if self.ctrl_c_exit_hint_expires_at == 0 {
            return;
        }
        self.ctrl_c_exit_hint_expires_at = 0;
        if render {
            self.terminal.request_render(false);
        }
    }

    fn is_ctrl_c_exit_hint_visible(&self) -> bool {
        self.ctrl_c_exit_hint_expires_at > super::agents_view_state::now_ms()
    }

    fn show_delete_confirmation(&mut self) {
        self.delete_confirm_expires_at = super::agents_view_state::now_ms() + DELETE_CONFIRM_DURATION_MS as i64;
        self.terminal.request_render(false);
    }

    fn clear_delete_confirmation(&mut self, render: bool) {
        self.pending_kill_subagent = None;
        if self.delete_confirm_expires_at == 0 {
            return;
        }
        self.delete_confirm_expires_at = 0;
        if render {
            self.terminal.request_render(false);
        }
    }

    fn is_delete_confirmation_visible(&self) -> bool {
        self.delete_confirm_expires_at > super::agents_view_state::now_ms()
    }

    /// `setStatusMessage(message, options)`.
    pub fn set_status_message(
        &mut self,
        message: Option<&str>,
        render: bool,
        tone: Option<StatusTone>,
        sticky: bool,
    ) {
        let status_line = message.map(format_agents_view_status_line);
        self.status_message = status_line.clone();
        // Errors come both from explicit tones and from formatError-style messages.
        self.status_message_tone = tone.unwrap_or_else(|| {
            if status_line.as_deref().map(|line| line.starts_with("Failed")).unwrap_or(false) {
                StatusTone::Error
            } else {
                StatusTone::Muted
            }
        });
        // Sticky messages stay up until the next keypress instead of a timer.
        self.status_message_sticky = sticky && status_line.is_some();
        if render {
            self.terminal.request_render(false);
        }
    }

    /// Sticky messages (e.g. billing warnings) stay until the user acknowledges
    /// them with any keypress.
    fn clear_sticky_status_message(&mut self) {
        if !self.status_message_sticky || self.daemon_shutdown_received || self.reconnect_started {
            return;
        }
        self.status_message_sticky = false;
        self.status_message = None;
        self.terminal.request_render(false);
    }

    /// `queryChanged()`.
    fn query_changed(&mut self) {
        self.persistent_state.query = Some(self.editor.get_text());
        self.arm_saved_search_fetch(false);
        self.rebuild_rows();
        // Searching is explicit user intent: claim the visible row as the new
        // anchor even if a remembered one is still waiting for its catalog row.
        self.sync_selected_row_state();
        self.terminal.request_render(false);
    }

    fn set_search_query(&mut self, query: &str) {
        self.editor.set_text(query);
        self.query_changed();
    }

    /// `toggleSubagentList(row)`.
    pub fn toggle_subagent_list(&mut self, row: &AgentsViewRow) {
        let target = row.identity.clone();
        if self.expanded_subagent_parents.contains(&target) {
            self.expanded_subagent_parents.remove(&target);
            self.program_shown_parents.remove(&target);
        } else {
            self.expanded_subagent_parents.insert(target);
        }
        self.rebuild_rows();
        self.sync_selected_row_state();
        self.terminal.request_render(false);
    }

    /// Toggle the full spawn program for the agent owning the selected row: one
    /// press shows it, another hides it.
    pub fn cycle_program_for_selected(&mut self) {
        let Some(row) = self.rows.get(self.selected_index).cloned() else {
            return;
        };
        let target = if row.kind == AgentsViewRowKind::Agent {
            Some(row.identity.clone())
        } else {
            row.parent_identity.clone()
        };
        let Some(target) = target else {
            return;
        };
        if !self.target_has_spawn_code(&target) {
            self.set_status_message(Some("No program recorded for these subagents"), true, None, false);
            return;
        }
        // Code only renders inside an expanded subagent list, so reveal it too.
        self.expanded_subagent_parents.insert(target.clone());
        if self.program_shown_parents.contains(&target) {
            self.program_shown_parents.remove(&target);
        } else {
            self.program_shown_parents.insert(target);
        }
        self.rebuild_rows();
        self.sync_selected_row_state();
        self.terminal.request_render(false);
    }

    /// Whether any subagent under the given agent identity carries spawn code.
    fn target_has_spawn_code(&self, target: &str) -> bool {
        for row in &self.all_rows {
            if row.parent_identity.as_deref() != Some(target) {
                continue;
            }
            if row.kind == AgentsViewRowKind::SubagentSummary {
                return row.has_spawn_code == Some(true);
            }
            if row.kind == AgentsViewRowKind::Subagent && row_has_spawn_code(row) {
                return true;
            }
        }
        false
    }

    /// `openSelected()`.
    pub fn open_selected(&mut self) {
        let Some(row) = self.rows.get(self.selected_index).cloned() else {
            return;
        };
        if !row.selectable || self.is_pending_delete_row(&row) {
            return;
        }
        if row.kind == AgentsViewRowKind::Subagent {
            self.open_selected_subagent(&row);
            return;
        }
        if row.summary.active_session_id.is_none() && row.summary.session_file.is_none() {
            self.set_status_message(
                Some("Cannot open agent without an active runtime or saved session file"),
                true,
                None,
                false,
            );
            return;
        }
        let has_children = has_unified_session_children(
            &self.unified_records,
            &scope_key_from_selection(&get_agents_view_selection_key(&row.summary)),
            Some(&self.unified_index),
        );
        self.finish(AgentsViewRunResult::Open {
            summary: row.summary.clone(),
            selection: None,
            expanded_ancestor_session_ids: None,
            has_children: Some(has_children),
            status_message: None,
        });
    }

    /// `openSelectedSubagent(row)`.
    pub fn open_selected_subagent(&mut self, row: &AgentsViewRow) {
        let expanded_ancestor_session_ids = self.collect_subagent_ancestor_session_ids(row);
        if row.summary.active_session_id.is_some() || row.summary.session_file.is_some() {
            let has_children = has_unified_session_children(
                &self.unified_records,
                &scope_key_from_selection(&get_agents_view_selection_key(&row.summary)),
                Some(&self.unified_index),
            );
            self.finish(AgentsViewRunResult::Open {
                summary: row.summary.clone(),
                selection: None,
                expanded_ancestor_session_ids: Some(expanded_ancestor_session_ids),
                has_children: Some(has_children),
                status_message: None,
            });
            return;
        }
        let root = self.find_subagent_root_row(row);
        let Some(root) = root else {
            self.set_status_message(
                Some("Cannot open agent without an active runtime or saved session file"),
                true,
                None,
                false,
            );
            return;
        };
        if root.summary.active_session_id.is_none() && root.summary.session_file.is_none() {
            self.set_status_message(
                Some("Cannot open agent without an active runtime or saved session file"),
                true,
                None,
                false,
            );
            return;
        }
        let has_children = has_unified_session_children(
            &self.unified_records,
            &scope_key_from_selection(&get_agents_view_selection_key(&root.summary)),
            Some(&self.unified_index),
        );
        let result = create_unattachable_child_open_result(
            &row.summary,
            &root.summary,
            &expanded_ancestor_session_ids,
            has_children,
        );
        self.finish(AgentsViewRunResult::Open {
            summary: result.summary,
            selection: Some(result.selection),
            expanded_ancestor_session_ids: Some(result.expanded_ancestor_session_ids),
            has_children: Some(result.has_children),
            status_message: Some(result.status_message),
        });
    }

    /// Session ids of every ancestor of a subagent row, root-most first.
    fn collect_subagent_ancestor_session_ids(&self, row: &AgentsViewRow) -> Vec<String> {
        let mut ancestors: Vec<String> = Vec::new();
        let mut parent_identity = row.parent_identity.clone();
        while let Some(identity) = parent_identity {
            let Some(parent) = self.rows.iter().find(|candidate| candidate.identity == identity) else {
                break;
            };
            ancestors.insert(0, parent.summary.session_id.clone());
            parent_identity = parent.parent_identity.clone();
        }
        ancestors
    }

    /// The whole subagent tree belongs to the root agent's session, so nested
    /// subagents also resolve to their top-level ancestor.
    fn find_subagent_root_row(&self, row: &AgentsViewRow) -> Option<AgentsViewRow> {
        let mut root = self
            .rows
            .iter()
            .find(|candidate| Some(&candidate.identity) == row.parent_identity.as_ref())
            .cloned();
        while let Some(current) = root.clone() {
            if current.kind == AgentsViewRowKind::Agent {
                break;
            }
            let parent_identity = current.parent_identity.clone();
            root = self
                .rows
                .iter()
                .find(|candidate| Some(&candidate.identity) == parent_identity.as_ref())
                .cloned();
        }
        root
    }

    fn find_summary_by_active_session_id(&self, active_session_id: &str) -> Option<SessionSummary> {
        self.rows
            .iter()
            .find(|row| row.summary.active_session_id.as_deref().unwrap_or(row.summary.id.as_str()) == active_session_id)
            .map(|row| row.summary.clone())
    }

    fn is_pending_delete_row(&self, row: &AgentsViewRow) -> bool {
        get_agents_view_summary_identity(&row.summary)
            == self.pending_delete_agent.as_ref().map(|pending| pending.identity.clone()).unwrap_or_default()
            && self.is_delete_confirmation_visible()
    }

    fn is_pending_kill_subagent_row(&self, row: &AgentsViewRow) -> bool {
        get_agents_view_summary_identity(&row.summary)
            == self.pending_kill_subagent.as_ref().map(|pending| pending.identity.clone()).unwrap_or_default()
            && self.is_delete_confirmation_visible()
    }

    /// Drain the async actions a synchronous key handler recorded. Equivalent to
    /// the reference's `void this.<action>()` fire-and-forget calls.
    pub async fn dispatch_pending_actions(&mut self) {
        if let Some((text, delivery)) = self.pending_submit.take() {
            self.submit(&text, &delivery).await;
        }
        if self.delete_selected_requested {
            self.delete_selected_requested = false;
            self.handle_delete_selected().await;
        }
        if self.reply_toggle_requested {
            self.reply_toggle_requested = false;
            self.toggle_reply_target().await;
        }
        if self.create_session_requested {
            self.create_session_requested = false;
            self.create_new_session().await;
        }
        if let Some(during_reconnect) = self.saved_search_fetch_pending.take() {
            if self.persistent_state.saved_catalog_loaded != Some(true) {
                let _ = self.refresh_saved_sessions(during_reconnect, true).await;
            }
        }
    }

    /// `toggleReplyTarget()`.
    pub async fn toggle_reply_target(&mut self) {
        let Some(selected_row) = self.rows.get(self.selected_index).cloned() else {
            return;
        };
        // Subagents are read-only; replying is reserved for top-level agents.
        if selected_row.kind != AgentsViewRowKind::Agent {
            return;
        }
        let summary = selected_row.summary.clone();
        // Live agents reply directly; saved sessions are resumed when the reply is
        // sent. Rows with neither runtime nor file have nothing to receive a prompt.
        if summary.active_session_id.is_none() && summary.session_file.is_none() {
            return;
        }
        if self
            .pending_delete_agent
            .as_ref()
            .map(|pending| pending.identity == selected_row.identity)
            .unwrap_or(false)
        {
            return;
        }
        let key = summary
            .active_session_id
            .clone()
            .unwrap_or_else(|| summary.id.clone());
        if self.reply_target.as_ref().map(|(armed, _)| armed == &key).unwrap_or(false) {
            self.set_reply_target(None);
            return;
        }
        self.set_reply_target(Some((key.clone(), summary.clone())));
        let Some(active_session_id) = summary.active_session_id.clone() else {
            // Inactive sessions have no live transcript endpoint; the persisted recap
            // (or opener) is the best preview and needs no daemon round-trip.
            self.reply_last_assistant_text = summary.summary.clone().or(summary.first_message.clone());
            self.terminal.request_render(false);
            return;
        };
        self.reply_last_assistant_text_loading = true;
        match self.get_last_assistant_text(&active_session_id).await {
            Ok(text) => {
                if self.reply_target.as_ref().map(|(armed, _)| armed == &key).unwrap_or(false) {
                    self.reply_last_assistant_text = text;
                    self.reply_last_assistant_text_loading = false;
                    self.terminal.request_render(false);
                }
            }
            Err(error) => {
                if self.reply_target.as_ref().map(|(armed, _)| armed == &key).unwrap_or(false) {
                    self.reply_last_assistant_text_loading = false;
                    self.set_status_message(
                        Some(&format_error("Failed to load latest response", &error)),
                        true,
                        None,
                        false,
                    );
                }
            }
        }
    }

    /// `setReplyTarget(target)`.
    pub fn set_reply_target(&mut self, target: Option<(String, SessionSummary)>) {
        if target.is_some() && self.reply_target.is_none() {
            self.action_mode_search_query = Some(self.editor.get_text());
            self.editor.set_text("");
        } else if target.is_none() && self.reply_target.is_some() {
            let restore = self
                .action_mode_search_query
                .clone()
                .or_else(|| self.persistent_state.query.clone())
                .unwrap_or_default();
            self.editor.set_text(&restore);
            self.action_mode_search_query = None;
        }
        self.reply_target = target.clone();
        self.reply_last_assistant_text = None;
        self.reply_last_assistant_text_loading = false;
        self.reply_header_time = match &target {
            Some((_, summary)) => format_agents_view_relative_time_now(
                summary.modified.as_deref().or(summary.created.as_deref()),
            ),
            None => String::new(),
        };
        let placeholder = match &target {
            Some((_, summary)) => {
                if summary.active_session_id.is_some() {
                    REPLY_PROMPT_FALLBACK_PLACEHOLDER
                } else {
                    RESUME_PROMPT_PLACEHOLDER
                }
            }
            None => SEARCH_PROMPT_PLACEHOLDER,
        };
        self.editor_placeholder_for_test = Some(placeholder.to_string());
        self.editor.set_placeholder(placeholder);
        if target.is_none() {
            self.rebuild_rows();
        }
        self.terminal.request_render(false);
    }

    /// `renderReplyHeaderLine()`.
    pub fn render_reply_header_line(&self) -> Option<String> {
        if self.rename_target.is_some() {
            return Some(self.theme.fg("warning", "Rename agent session"));
        }
        self.reply_target.as_ref()?;
        let headline = create_agents_view_reply_headline(self.reply_last_assistant_text.as_deref()).unwrap_or_else(|| {
            self.theme.fg(
                "dim",
                if self.reply_last_assistant_text_loading {
                    "Loading last response..."
                } else {
                    "No response yet"
                },
            )
        });
        Some(if self.reply_header_time.is_empty() {
            headline
        } else {
            format!("{} {headline}", self.theme.fg("warning", &self.reply_header_time))
        })
    }

    /// `getLastAssistantText(activeSessionId)`.
    pub async fn get_last_assistant_text(&self, active_session_id: &str) -> Result<Option<String>, String> {
        let client = self.require_client()?;
        let response = client
            .request(
                serde_json::json!({ "type": "get_last_assistant_text", "activeSessionId": active_session_id }),
                30000,
            )
            .await?;
        let data = require_daemon_data(&response)?;
        if !is_record(&data) {
            return Err("Daemon returned an invalid last assistant response".to_string());
        }
        match data.get("text") {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(text)) => Ok(Some(text.clone())),
            Some(_) => Err("Daemon returned an invalid last assistant response".to_string()),
        }
    }

    /// `enterRenameMode()`.
    pub fn enter_rename_mode(&mut self) {
        let Some(row) = self.rows.get(self.selected_index).cloned() else {
            return;
        };
        // Only top-level agents carry a renameable session; subagents do not.
        if row.kind != AgentsViewRowKind::Agent || !row.selectable {
            return;
        }
        let active_session_id = row.summary.active_session_id.clone();
        let session_file = row.summary.session_file.clone();
        if active_session_id.is_none() && session_file.is_none() {
            self.set_status_message(Some("This session cannot be renamed"), true, None, false);
            return;
        }
        self.set_reply_target(None);
        self.action_mode_search_query = Some(self.editor.get_text());
        self.pending_delete_agent = None;
        self.pending_kill_subagent = None;
        self.rename_target = Some((active_session_id, session_file, row.summary.clone()));
        self.editor_placeholder_for_test = Some("Name this agent session".to_string());
        self.editor.set_placeholder("Name this agent session");
        self.editor.set_text(row.summary.session_name.clone().unwrap_or_default().as_str());
        self.terminal.request_render(false);
    }

    /// `exitRenameMode()`.
    pub fn exit_rename_mode(&mut self) {
        self.rename_target = None;
        let restore = self
            .action_mode_search_query
            .clone()
            .or_else(|| self.persistent_state.query.clone())
            .unwrap_or_default();
        self.editor.set_text(&restore);
        self.action_mode_search_query = None;
        self.editor_placeholder_for_test = Some(SEARCH_PROMPT_PLACEHOLDER.to_string());
        self.editor.set_placeholder(SEARCH_PROMPT_PLACEHOLDER);
        self.rebuild_rows();
        self.terminal.request_render(false);
    }

    /// `confirmRename(value)`.
    pub async fn confirm_rename(&mut self, value: &str) {
        let Some((_, _, summary)) = self.rename_target.clone() else {
            return;
        };
        let name = value.trim().to_string();
        if name.is_empty() {
            self.exit_rename_mode();
            return;
        }
        self.exit_rename_mode();
        let _ = self.rename_session(&summary, &name).await;
    }

    /// Shared by rename mode and /name: rename, refresh both catalogs, report.
    pub async fn rename_session(&mut self, summary: &SessionSummary, name: &str) -> bool {
        self.set_status_message(Some("Renaming agent..."), true, None, false);
        let result = async {
            let client = self.require_client()?;
            if let Some(active_session_id) = &summary.active_session_id {
                let response = client
                    .request(
                        serde_json::json!({
                            "type": "rename",
                            "activeSessionId": active_session_id,
                            "name": name,
                        }),
                        30000,
                    )
                    .await?;
                require_daemon_data(&response)?;
            } else if let Some(session_file) = &summary.session_file {
                let context = self.get_saved_session_catalog_context();
                rename_daemon_saved_session(&client, &context, session_file, name).await?;
            } else {
                return Err("__cannot_rename__".to_string());
            }
            Ok::<(), String>(())
        }
        .await;
        match result {
            Ok(()) => {
                let _ = self.refresh_sessions().await;
                self.refresh_saved_sessions_if_loaded();
                self.set_status_message(Some(&format!("Renamed to {name}")), true, None, false);
                true
            }
            Err(error) if error == "__cannot_rename__" => {
                self.set_status_message(Some("This session cannot be renamed"), true, Some(StatusTone::Warning), false);
                false
            }
            Err(error) => {
                self.set_status_message(
                    Some(&if is_unknown_daemon_command_error(&error, "rename") {
                        "Failed to rename: the daemon is running an older build; restart the daemon and try again"
                            .to_string()
                    } else {
                        format_error("Failed to rename agent", &error)
                    }),
                    true,
                    None,
                    false,
                );
                false
            }
        }
    }

    /// `refreshSessions()`.
    pub async fn refresh_sessions(&mut self) {
        if self.reconnect_started || self.daemon_shutdown_received {
            return;
        }
        let summaries = self.roster_store.summaries().await;
        self.apply_session_list(summaries, true);
        self.resolve_missing_selection_anchor();
    }

    /// `onRosterUpdate()`.
    pub async fn on_roster_update(&mut self) {
        if self.stopped {
            return;
        }
        let summaries = self.roster_store.summaries().await;
        self.apply_session_list(summaries, true);
        self.resolve_missing_selection_anchor();
    }

    /// `selectSummary(summary)`: point selection (and its persisted key) at a
    /// freshly resumed session row.
    pub fn select_summary(&mut self, summary: &SessionSummary) {
        self.selected_row_identity = Some(get_agents_view_summary_identity(summary));
        self.selected_active_session_id = Some(
            summary
                .active_session_id
                .clone()
                .unwrap_or_else(|| summary.id.clone()),
        );
        self.selected_session_key = Some(get_agents_view_selection_key(summary));
        self.persistent_state.selected_row_identity = self.selected_row_identity.clone();
        self.persistent_state.selected_session_key = self.selected_session_key.clone();
    }

    /// `submit(value, delivery)`.
    pub async fn submit(&mut self, value: &str, delivery: &str) {
        if self.rename_target.is_some() {
            self.confirm_rename(value).await;
            return;
        }
        if let Some(target) = self.reply_target.clone() {
            let text = value.trim().to_string();
            if let Some(view_command) = parse_agents_view_command(&text) {
                // Stale summaries mis-route the RPCs after a runtime replacement.
                let records = self.unified_records.clone();
                let current_summary = {
                    let rows = self.rows.clone();
                    resolve_current_reply_target_summary(&records, &target, &|active_session_id| {
                        rows.iter()
                            .find(|row| {
                                row.summary.active_session_id.as_deref().unwrap_or(row.summary.id.as_str())
                                    == active_session_id
                            })
                            .map(|row| row.summary.clone())
                    })
                };
                let succeeded = self.run_agents_view_command(&view_command, &current_summary).await;
                if !succeeded && self.reply_target.as_ref() == Some(&target) && self.editor.get_text().is_empty() {
                    self.editor.set_text(value);
                }
                return;
            }
            if let Some(rejection) = get_reply_composer_command_rejection(&text) {
                // submitValue cleared the buffer before onSubmit; keep the draft.
                if self.editor.get_text().is_empty() {
                    self.editor.set_text(value);
                }
                self.set_status_message(Some(&rejection), true, Some(StatusTone::Warning), false);
                return;
            }
            if !text.is_empty() {
                self.editor.set_text("");
                let sent = self.send_reply(&target, &text, delivery).await;
                if sent {
                    if self.reply_target.as_ref() == Some(&target) && self.editor.get_text().is_empty() {
                        self.set_reply_target(None);
                    }
                    // Keep the send outcome (or sticky cwd notice) that sendReply just surfaced.
                    self.refresh_sessions().await;
                } else if self.reply_target.as_ref() == Some(&target) && self.editor.get_text().is_empty() {
                    self.editor.set_text(value);
                }
            }
            return;
        }
        // Search text is never a prompt or a command; Enter opens the selection.
        self.open_selected();
    }

    /// Alt+Enter in the reply composer queues the reply as a follow-up.
    pub fn handle_reply_follow_up(&mut self) {
        if self.reply_target.is_none() {
            return;
        }
        // Unlike Enter, this path skips submitValue, so expand paste markers here.
        let text = self.editor.get_expanded_text();
        if text.trim().is_empty() {
            return;
        }
        self.pending_submit = Some((text, "followUp".to_string()));
    }

    /// `sendReply(target, text, delivery)`.
    pub async fn send_reply(&mut self, target: &(String, SessionSummary), text: &str, delivery: &str) -> bool {
        let records = self.unified_records.clone();
        let rows = self.rows.clone();
        let current_summary = resolve_current_reply_target_summary(&records, target, &|active_session_id| {
            rows.iter()
                .find(|row| {
                    row.summary.active_session_id.as_deref().unwrap_or(row.summary.id.as_str()) == active_session_id
                })
                .map(|row| row.summary.clone())
        });
        let mut active_session_id = current_summary.active_session_id.clone();
        let mut live_summary = if active_session_id.is_some() {
            Some(current_summary.clone())
        } else {
            None
        };
        let mut cwd_fallback_notice = None;
        let mut did_resume = false;
        let outcome = async {
            if active_session_id.is_none() {
                // Saved session: resume it into the daemon first, then deliver the
                // prompt through the same path as a live reply.
                self.set_status_message(Some("Resuming session..."), true, None, false);
                let client = self.require_client()?;
                let (resumed, resumed_active_session_id, notice) =
                    resume_saved_agents_view_session(&client, &self.options.config, &current_summary).await?;
                active_session_id = Some(resumed_active_session_id);
                did_resume = true;
                // The rows are still pre-resume; the fresh summary is the authoritative
                // streaming state for scheduling the prompt.
                live_summary = Some(resumed.clone());
                cwd_fallback_notice = notice;
                self.inactive_agent_identities
                    .remove(&get_agents_view_summary_identity(&target.1));
                // The resume and delivery still belong to this submission, but selection
                // belongs to the current composer. Do not steal it after cancellation.
                if self.reply_target.as_ref() == Some(target) {
                    self.select_summary(&resumed);
                }
            }
            let behavior = if delivery == "followUp" {
                Some("followUp".to_string())
            } else if live_summary.as_ref().map(|summary| summary.is_streaming).unwrap_or(false) {
                Some("steer".to_string())
            } else {
                None
            };
            self.set_status_message(Some("Sending reply..."), true, None, false);
            let active_session_id = active_session_id.clone().unwrap_or_default();
            self.send_prompt(&active_session_id, text, behavior.as_deref()).await?;
            // The fallback-directory notice must outlive the transient send statuses.
            match &cwd_fallback_notice {
                Some(notice) => self.set_status_message(Some(notice), true, None, true),
                None => self.set_status_message(Some("Reply sent"), true, None, false),
            }
            Ok::<(), String>(())
        }
        .await;
        match outcome {
            Ok(()) => true,
            Err(error) => {
                self.set_status_message(Some(&format_error("Failed to send reply", &error)), true, None, false);
                if did_resume {
                    self.refresh_sessions().await;
                }
                false
            }
        }
    }

    /// `sendPrompt(activeSessionId, message, streamingBehavior)`.
    pub async fn send_prompt(
        &mut self,
        active_session_id: &str,
        message: &str,
        streaming_behavior: Option<&str>,
    ) -> Result<(), String> {
        if self.options.config.telemetry_disabled == Some(true) {
            let socket_path = self.require_socket_path()?;
            let client = connect_agents_view_daemon_client(&socket_path, self.transport.clone()).await?;
            let connection = self
                .factory
                .attach(
                    client,
                    active_session_id,
                    AttachOptions {
                        close_client_on_dispose: Some(true),
                        supports_extension_ui: Some(false),
                        reconnect_timeout_ms: self.options.reconnect_timeout_ms,
                        telemetry_disabled: Some(true),
                    },
                )
                .await?;
            let result = connection.prompt(message, streaming_behavior).await;
            let _ = connection.dispose().await;
            return result;
        }
        let command = create_prompt_command(active_session_id, message, streaming_behavior);
        let client = self.require_client()?;
        let response = client.request(command, 30000).await?;
        require_daemon_data(&response)?;
        Ok(())
    }

    /// `runAgentsViewCommand(command, target)`.
    pub async fn run_agents_view_command(
        &mut self,
        command: &AgentsViewCommand,
        target: &SessionSummary,
    ) -> bool {
        let armed_at_start = self.reply_target.clone();
        let mut disarm_if_unchanged = |mode: &mut Self| {
            if armed_at_start.is_some() && mode.reply_target == armed_at_start {
                mode.set_reply_target(None);
            }
        };
        match command.name {
            AgentsViewCommandName::Name => {
                let name = command.args.trim().to_string();
                if name.is_empty() {
                    self.set_status_message(Some("Usage: /name <session name>"), true, Some(StatusTone::Warning), false);
                    return false;
                }
                let renamed = self.rename_session(target, &name).await;
                if renamed {
                    disarm_if_unchanged(self);
                }
                renamed
            }
            AgentsViewCommandName::Kill => {
                let Some(active_session_id) = target.active_session_id.clone() else {
                    self.set_status_message(
                        Some("/kill needs a running agent; this session is inactive"),
                        true,
                        Some(StatusTone::Warning),
                        false,
                    );
                    return false;
                };
                let result = async {
                    let client = self.require_client()?;
                    let response = client
                        .request(
                            serde_json::json!({ "type": "kill", "activeSessionId": active_session_id }),
                            30000,
                        )
                        .await?;
                    require_daemon_data(&response)?;
                    Ok::<(), String>(())
                }
                .await;
                match result {
                    Ok(()) => {}
                    Err(error) => {
                        // As in deactivatePendingAgent: an agent that already finished counts as stopped.
                        if !is_unknown_active_session_error(&error) {
                            self.set_status_message(
                                Some(&format_error(&format!("Failed to run /{}", command.name.as_str()), &error)),
                                true,
                                None,
                                false,
                            );
                            return false;
                        }
                    }
                }
                disarm_if_unchanged(self);
                self.set_status_message(Some("Agent stopped"), true, None, false);
                self.refresh_sessions().await;
                true
            }
        }
    }

    /// `createNewSession()`: create a fresh daemon session and open it in the chat view.
    pub async fn create_new_session(&mut self) -> bool {
        if self.creating_new_session || self.stopped {
            return false;
        }
        self.creating_new_session = true;
        let result = async {
            let socket_path = self.require_socket_path()?;
            let client = connect_agents_view_daemon_client(&socket_path, self.transport.clone()).await?;
            let outcome = async {
                self.set_status_message(Some("Creating session..."), true, None, false);
                let mut command = serde_json::json!({
                    "type": "create",
                    "config": {
                        "cwd": self.options.config.cwd,
                        "sessionDir": self.options.config.session_dir,
                        "telemetryDisabled": self.options.config.telemetry_disabled,
                    },
                });
                if let Some(config) = command["config"].as_object_mut() {
                    if config.get("cwd").map(|value| value.is_null()).unwrap_or(false) {
                        config.remove("cwd");
                    }
                    if config.get("sessionDir").map(|value| value.is_null()).unwrap_or(false) {
                        config.remove("sessionDir");
                    }
                    if config.get("telemetryDisabled").map(|value| value.is_null()).unwrap_or(false) {
                        config.remove("telemetryDisabled");
                    }
                }
                let response = client.request(command, 120000).await?;
                let created = expect_session_summary(&require_daemon_data(&response)?)?;
                // The view can finish mid-create; kill the fresh session instead of orphaning it.
                if self.stopped {
                    if let Some(active_session_id) = created.active_session_id.clone() {
                        let _ = client
                            .request(
                                serde_json::json!({ "type": "kill", "activeSessionId": active_session_id }),
                                30000,
                            )
                            .await;
                    }
                    return Ok::<bool, String>(false);
                }
                self.select_summary(&created);
                self.finish(AgentsViewRunResult::Open {
                    summary: created,
                    selection: None,
                    expanded_ancestor_session_ids: None,
                    has_children: None,
                    status_message: None,
                });
                Ok(true)
            }
            .await;
            client.close();
            outcome
        }
        .await;
        self.creating_new_session = false;
        match result {
            Ok(value) => value,
            Err(error) => {
                if !self.stopped {
                    self.set_status_message(Some(&format_error("Failed to create session", &error)), true, None, false);
                }
                false
            }
        }
    }

    /// `handleDeleteSelected()`.
    pub async fn handle_delete_selected(&mut self) {
        let Some(row) = self.rows.get(self.selected_index).cloned() else {
            return;
        };
        if !row.selectable {
            return;
        }
        if row.kind == AgentsViewRowKind::Subagent {
            self.pending_delete_agent = None;
            self.handle_kill_subagent_selected(&row).await;
            return;
        }
        if row.kind != AgentsViewRowKind::Agent {
            return;
        }
        self.pending_kill_subagent = None;
        let identity = get_agents_view_summary_identity(&row.summary);
        if row.summary.active_session_id.is_none() && row.summary.session_file.is_some() {
            let session_file = row.summary.session_file.clone().unwrap_or_default();
            if self
                .pending_delete_agent
                .as_ref()
                .map(|pending| pending.identity == identity)
                .unwrap_or(false)
                && self.is_delete_confirmation_visible()
            {
                self.clear_delete_confirmation(false);
                let outcome = async {
                    // Authoritative liveness check; its narrower plain-list verdict stays local.
                    let client = self.require_client()?;
                    let latest = expect_session_list(&require_daemon_data(
                        &client.request(create_agents_view_list_command(), 30000).await?,
                    )?)?;
                    let active = resolve_agents_view_active_summary_for_path(&session_file, &latest);
                    if active.is_some() {
                        return Ok::<Option<DeleteSessionFileResult>, String>(None);
                    }
                    let context = self.get_saved_session_catalog_context();
                    delete_daemon_saved_session(&client, &context, &session_file).await.map(Some)
                }
                .await;
                match outcome {
                    Ok(None) => {
                        self.pending_delete_agent = None;
                        self.set_status_message(
                            Some("Session became active; stop it before deleting"),
                            true,
                            Some(StatusTone::Warning),
                            false,
                        );
                        self.refresh_sessions().await;
                    }
                    Ok(Some(result)) => {
                        if !result.ok {
                            self.set_status_message(
                                Some(&format!(
                                    "Failed to delete session: {}",
                                    result.error.clone().unwrap_or_else(|| "Unknown error".to_string())
                                )),
                                true,
                                Some(StatusTone::Error),
                                false,
                            );
                            return;
                        }
                        self.pending_delete_agent = None;
                        let refreshed = if self.persistent_state.saved_catalog_loaded != Some(true) {
                            true
                        } else {
                            self.refresh_saved_sessions(false, true).await
                        };
                        let success = if result.method.as_deref() == Some("trash") {
                            "Session moved to trash"
                        } else {
                            "Session deleted"
                        };
                        self.set_status_message(
                            Some(&if refreshed {
                                success.to_string()
                            } else {
                                format!("{success}; refresh failed")
                            }),
                            true,
                            None,
                            false,
                        );
                    }
                    Err(error) => {
                        self.set_status_message(
                            Some(&format_error("Failed to delete session", &error)),
                            true,
                            None,
                            false,
                        );
                    }
                }
                return;
            }
            self.pending_delete_agent = Some(PendingDeleteAgent {
                identity,
                active_session_id: None,
                session_file: Some(session_file),
                summary: row.summary.clone(),
                stopped: false,
            });
            self.show_delete_confirmation();
            return;
        }
        if self
            .pending_delete_agent
            .as_ref()
            .map(|pending| pending.identity == identity)
            .unwrap_or(false)
        {
            if self.is_delete_confirmation_visible() {
                self.deactivate_pending_agent().await;
                return;
            }
            self.show_delete_confirmation();
            return;
        }
        self.stop_agent_for_deletion(&row).await;
    }

    /// `handleKillSubagentSelected(row)`.
    pub async fn handle_kill_subagent_selected(&mut self, row: &AgentsViewRow) {
        let identity = get_agents_view_summary_identity(&row.summary);
        if self
            .pending_kill_subagent
            .as_ref()
            .map(|pending| pending.identity == identity)
            .unwrap_or(false)
            && self.is_delete_confirmation_visible()
        {
            let Some(pending) = self.pending_kill_subagent.clone() else {
                return;
            };
            self.clear_delete_confirmation(false);
            self.kill_subagent(&pending, row).await;
            return;
        }
        let child_id = row.summary.rlm_child_id.clone();
        let root_active_session_id = self
            .find_subagent_root_row(row)
            .and_then(|root| root.summary.active_session_id.clone());
        let (Some(child_id), Some(root_active_session_id)) = (child_id, root_active_session_id) else {
            self.set_status_message(Some("Cannot stop subagent without its parent agent"), true, None, false);
            return;
        };
        self.pending_kill_subagent = Some(PendingKillSubagent { identity, root_active_session_id, child_id });
        self.show_delete_confirmation();
    }

    /// `killSubagent(pending, currentRow)`.
    pub async fn kill_subagent(&mut self, pending: &PendingKillSubagent, current_row: &AgentsViewRow) {
        let running = has_live_work(current_row);
        self.set_status_message(
            Some(if running { "Stopping subagent..." } else { "Deleting subagent..." }),
            true,
            None,
            false,
        );
        let outcome = async {
            let client = self.require_client()?;
            if !running && client.supports_server_capability("delete_rlm_subagent") {
                let response = client
                    .request(
                        serde_json::json!({
                            "type": "delete_rlm_subagent",
                            "activeSessionId": pending.root_active_session_id,
                            "childId": pending.child_id,
                        }),
                        30000,
                    )
                    .await?;
                let data = require_daemon_data(&response)?;
                let deleted = data.get("deleted").and_then(|value| value.as_bool()) == Some(true);
                let still_running = data.get("reason").and_then(|value| value.as_str()) == Some("running");
                Ok::<(&'static str, Option<StatusTone>), String>((
                    if deleted {
                        "Subagent deleted"
                    } else if still_running {
                        "Subagent is running; stop it first"
                    } else {
                        "Subagent already removed"
                    },
                    None,
                ))
            } else {
                let response = client
                    .request(
                        serde_json::json!({
                            "type": "cancel_rlm_child",
                            "activeSessionId": pending.root_active_session_id,
                            "childId": pending.child_id,
                        }),
                        30000,
                    )
                    .await?;
                let data = require_daemon_data(&response)?;
                let cancelled = data.get("cancelled").and_then(|value| value.as_bool()) == Some(true);
                Ok((
                    if running {
                        if cancelled {
                            "Subagent stopped"
                        } else {
                            "Subagent already finished"
                        }
                    } else {
                        "The daemon cannot delete subagents; it was left unchanged"
                    },
                    if running { None } else { Some(StatusTone::Warning) },
                ))
            }
        }
        .await;
        match outcome {
            Ok((message, tone)) => {
                self.set_status_message(Some(message), false, tone, false);
                self.refresh_sessions().await;
            }
            Err(error) => {
                let command = if running { "cancel_rlm_child" } else { "delete_rlm_subagent" };
                self.set_status_message(
                    Some(&if is_unknown_daemon_command_error(&error, command) {
                        "Failed to update subagent: the daemon is running an older build; restart the daemon and try again"
                            .to_string()
                    } else {
                        format_error("Failed to update subagent", &error)
                    }),
                    true,
                    None,
                    false,
                );
            }
        }
    }

    /// `stopAgentForDeletion(row)`.
    pub async fn stop_agent_for_deletion(&mut self, row: &AgentsViewRow) {
        let identity = get_agents_view_summary_identity(&row.summary);
        let Some(active_session_id) = row.summary.active_session_id.clone() else {
            self.pending_delete_agent = Some(PendingDeleteAgent {
                identity,
                active_session_id: None,
                session_file: row.summary.session_file.clone(),
                summary: row.summary.clone(),
                stopped: false,
            });
            self.set_status_message(None, false, None, false);
            self.set_reply_target(None);
            self.show_delete_confirmation();
            return;
        };
        if !has_live_work(row) {
            self.pending_delete_agent = Some(PendingDeleteAgent {
                identity,
                active_session_id: Some(active_session_id),
                session_file: row.summary.session_file.clone(),
                summary: row.summary.clone(),
                stopped: false,
            });
            self.set_status_message(None, false, None, false);
            self.set_reply_target(None);
            self.show_delete_confirmation();
            return;
        }
        self.set_status_message(Some("Stopping agent..."), true, None, false);
        let result = async {
            let client = self.require_client()?;
            let response = client
                .request(serde_json::json!({ "type": "kill", "activeSessionId": active_session_id }), 30000)
                .await?;
            require_daemon_data(&response)?;
            Ok::<(), String>(())
        }
        .await;
        match result {
            Ok(()) => {
                self.pending_delete_agent = Some(PendingDeleteAgent {
                    identity,
                    active_session_id: Some(active_session_id.clone()),
                    session_file: row.summary.session_file.clone(),
                    summary: row.summary.clone(),
                    stopped: true,
                });
                self.selected_active_session_id = Some(active_session_id);
                self.set_reply_target(None);
                self.set_status_message(None, false, None, false);
                self.show_delete_confirmation();
                self.refresh_sessions().await;
            }
            Err(error) => {
                self.set_status_message(Some(&format_error("Failed to stop agent", &error)), true, None, false);
            }
        }
    }

    /// `deactivatePendingAgent()`.
    pub async fn deactivate_pending_agent(&mut self) {
        let Some(pending) = self.pending_delete_agent.clone() else {
            return;
        };
        self.set_status_message(Some("Deactivating agent..."), true, None, false);
        let result = async {
            let client = self.require_client()?;
            if let Some(active_session_id) = pending.active_session_id.clone() {
                let response = client
                    .request(serde_json::json!({ "type": "kill", "activeSessionId": active_session_id }), 30000)
                    .await;
                match response {
                    Ok(response) => {
                        if let Err(error) = require_daemon_data(&response) {
                            if !is_unknown_active_session_error(&error) {
                                return Err(error);
                            }
                        }
                    }
                    Err(error) => {
                        if !is_unknown_active_session_error(&error) {
                            return Err(error);
                        }
                    }
                }
            }
            // Skip a file deleted between listing and now: SessionManager.open would
            // recreate a stub at the old path instead of loading it.
            if let Some(session_file) = pending.session_file.clone() {
                if std::path::Path::new(&session_file).exists() {
                    // Persist archived unless it already is: sessions with no prior
                    // session_state entry would otherwise resurface on the next scan.
                    if !self.session_state_is_archived(&session_file) {
                        self.append_archived_session_state(&session_file);
                    }
                }
            }
            Ok::<(), String>(())
        }
        .await;
        match result {
            Ok(()) => {
                self.inactive_agent_identities.insert(pending.identity);
                self.pending_delete_agent = None;
                self.clear_delete_confirmation(false);
                self.selected_active_session_id = None;
                self.set_status_message(Some("Agent inactive"), false, None, false);
                self.refresh_sessions().await;
                self.refresh_saved_sessions_if_loaded();
            }
            Err(error) => {
                self.set_status_message(
                    Some(&format_error("Failed to deactivate agent", &error)),
                    true,
                    None,
                    false,
                );
            }
        }
    }

    /// `SessionManager.open(path).getSessionState()?.status === "archived"`.
    /// TODO(port): core/session-manager.ts owns SessionManager; this reads the
    /// persisted `session_state` entry directly.
    fn session_state_is_archived(&self, session_file: &str) -> bool {
        let Ok(contents) = std::fs::read_to_string(session_file) else {
            return false;
        };
        contents.lines().any(|line| {
            let Ok(value) = serde_json::from_str::<Value>(line) else {
                return false;
            };
            value.get("type").and_then(|value| value.as_str()) == Some("session_state")
                && value.get("status").and_then(|value| value.as_str()) == Some("archived")
        })
    }

    /// `SessionManager.open(path).appendSessionState({ status: "archived" })`.
    /// TODO(port): replace with SessionManager once core/session-manager.ts lands.
    fn append_archived_session_state(&self, session_file: &str) {
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(session_file) {
            let _ = writeln!(file, "{}", serde_json::json!({ "type": "session_state", "status": "archived" }));
        }
    }

    /// `loadStartupNotices()`.
    pub fn load_startup_notices(&mut self) {
        // Notices live on persistentState (read directly in renderStartupNotices),
        // so they survive leaving and re-entering the agents view regardless of
        // which instance's gather resolved.
        if self.persistent_state.startup_notices.is_some() {
            return;
        }
        // TODO(port): gatherStartupNotices (modes/shared/startup-notices.ts) does
        // the version/package/tmux checks; that slice owns the real work. The
        // placeholder keeps the "notices resolved once" lifecycle intact.
        let notices = StartupNotices::default();
        self.persistent_state.startup_notices = Some(notices);
    }

    /// `renderStartupNotices(width)`.
    pub fn render_startup_notices(&self, width: usize) -> Vec<String> {
        let Some(notices) = &self.persistent_state.startup_notices else {
            return Vec::new();
        };
        let mut formatted: Vec<String> = Vec::new();
        if let Some(new_version) = &notices.new_version {
            formatted.push(format_update_available_notice(new_version));
        }
        if !notices.package_updates.is_empty() {
            formatted.push(format_package_update_notice(&notices.package_updates));
        }
        if let Some(tmux_warning) = &notices.tmux_warning {
            formatted.push(format_tmux_warning_notice(tmux_warning));
        }
        // Match the splash header's one-column gutter and wrap so long notices
        // (e.g. the tmux fix instructions) stay readable instead of truncating.
        let wrap_width = width.saturating_sub(1).max(1);
        formatted
            .into_iter()
            .flat_map(|line| {
                wrap_text_with_ansi(&line, wrap_width)
                    .into_iter()
                    .map(|wrapped| format!(" {wrapped}"))
                    .collect::<Vec<String>>()
            })
            .collect()
    }

    /// `contentHeight(width)`.
    pub fn content_height(&self, width: usize) -> usize {
        let rows = self.terminal.rows();
        let dock_height = clipped_fullscreen_dock_height(self.render_dock(width).len(), rows);
        rows.saturating_sub(dock_height)
    }

    /// `render(width)`.
    pub fn render(&mut self, width: usize) -> Vec<String> {
        let safe_width = width.max(1);
        let height = self.content_height(safe_width);
        let mut lines = self.render_content(safe_width, height);
        lines.truncate(height);
        while lines.len() < height {
            lines.push(String::new());
        }
        lines
            .into_iter()
            .map(|line| self.finalize_rendered_line(&line, safe_width))
            .collect()
    }

    fn render_content(&mut self, width: usize, height: usize) -> Vec<String> {
        if height == 0 {
            return Vec::new();
        }
        let prompt_lines = self.editor.render(width);
        let list_gap = if height >= prompt_lines.len() + 2 { 1 } else { 0 };
        let reserved_list_rows = 3usize.min(1.max(height.saturating_sub(prompt_lines.len() + list_gap)));
        let header_rows = height.saturating_sub(prompt_lines.len() + list_gap + reserved_list_rows);
        let notice_lines = self.render_startup_notices(width);
        let notice_rows = if notice_lines.is_empty() { 0 } else { notice_lines.len() + 1 };
        // Select a complete smaller portrait before rendering, never crop it to
        // reclaim the search prompt or useful session rows on a short terminal.
        let mut header_lines: Vec<String> = Vec::new();
        let splash_height = header_rows.saturating_sub(notice_rows + 1);
        header_lines.extend(self.render_splash(width, splash_height));
        if !notice_lines.is_empty() {
            header_lines.push(String::new());
            header_lines.extend(notice_lines);
        }
        header_lines.push(String::new());

        let mut lines: Vec<String> = header_lines.into_iter().take(header_rows).collect();
        lines.extend(prompt_lines);
        if list_gap > 0 {
            lines.push(String::new());
        }
        let list_rows = height.saturating_sub(lines.len());
        lines.extend(self.render_session_rows(width, list_rows));
        lines
    }

    /// The same adaptive portrait and metadata component used by interactive chat.
    pub fn render_splash(&self, width: usize, height: usize) -> Vec<String> {
        use crate::modes::interactive::interactive_mode::{BrandSplashHeader, BrandSplashHeaderOptions, BrandSplashMetadataLine};
        let model = self.get_splash_model_id();
        let cwd = self.get_splash_cwd();
        let rows = self.terminal.rows() as f64;
        let metadata = vec![
            BrandSplashMetadataLine { label: "agents".into(), value: self.get_agent_counts_text() },
            BrandSplashMetadataLine { label: "scope".into(), value: self.scope_root_summary.as_ref().map(get_agents_view_session_title).unwrap_or_else(|| "global".into()) },
            BrandSplashMetadataLine { label: "depth".into(), value: get_agents_view_depth(self.scope_root_summary.as_ref()).to_string() },
        ];
        BrandSplashHeader::new(crate::config::VERSION.into(), Box::new(move || model.clone()),
            Box::new(move || cwd.clone()), None, BrandSplashHeaderOptions {
                get_rows: Some(Box::new(move || rows)),
                get_extra_metadata: Some(Box::new(move || metadata.clone())),
                get_hide_start_hint: Some(Box::new(|| true)),
                ..Default::default()
            }).render(width as f64, Some(height as f64))
    }

    /// `getExtraMetadata` for the splash header.
    pub fn splash_metadata(&self) -> Vec<String> {
        let root = self.scope_root_summary.as_ref();
        vec![
            format!("agents   {}", self.get_agent_counts_text()),
            format!(
                "scope    {}",
                root.map(get_agents_view_session_title).unwrap_or_else(|| "global".to_string())
            ),
            format!("depth    {}", get_agents_view_depth(root)),
        ]
    }

    fn get_agent_counts_text(&self) -> String {
        let counts = count_rows_by_section(&self.all_rows);
        let loading = if self.saved_catalog_refresh_pending {
            format!(" · loading saved chats ({})...", self.saved_catalog_progress)
        } else {
            String::new()
        };
        format!(
            "{} running, {} idle, {} inactive{loading}",
            counts.get(&AgentsViewSection::Running).copied().unwrap_or(0),
            counts.get(&AgentsViewSection::Idle).copied().unwrap_or(0),
            counts.get(&AgentsViewSection::Inactive).copied().unwrap_or(0),
        )
    }

    /// `renderSessionRows(width, maxRows)`.
    pub fn render_session_rows(&mut self, width: usize, max_rows: usize) -> Vec<String> {
        if max_rows == 0 {
            return Vec::new();
        }
        if self.show_actions {
            let mut lines = self.render_actions(width);
            lines.truncate(max_rows);
            return lines;
        }
        let layout = build_compact_agents_view_layout(&self.rows, width, self.theme.as_ref());
        let mut display_items: Vec<DisplayItem> = Vec::new();
        let count_source = if self.all_rows.is_empty() { &self.rows } else { &self.all_rows };
        let counts = count_rows_by_section(count_source);
        for section in [AgentsViewSection::Running, AgentsViewSection::Idle, AgentsViewSection::Inactive] {
            if counts.get(&section).copied().unwrap_or(0) == 0 {
                continue;
            }
            if !display_items.is_empty() {
                display_items.push(DisplayItem::Spacer);
            }
            display_items.push(DisplayItem::Heading(section));
            for (index, row) in get_display_rows_for_section(&self.rows, section).iter().enumerate() {
                let _ = index;
                let row_index = self
                    .rows
                    .iter()
                    .position(|candidate| candidate.identity == row.identity)
                    .unwrap_or(0);
                display_items.push(DisplayItem::Row(row_index));
                if (row.kind == AgentsViewRowKind::Agent || row.kind == AgentsViewRowKind::Subagent)
                    && row.running_subagent_count > 0
                    && !self.expanded_subagent_parents.contains(&row.identity)
                {
                    display_items.push(DisplayItem::RunningSubagents(row_index));
                }
            }
        }
        if display_items.is_empty() {
            let message = if self.saved_catalog_refresh_pending {
                "Loading saved chats..."
            } else {
                "No sessions match your search."
            };
            return vec![self.theme.fg("dim", message)];
        }
        // Reserve the shared column header before calculating the selection viewport.
        let header_rows = if max_rows > 1 { 1 } else { 0 };
        let visible_rows = max_rows - header_rows;
        let selected_identity = self.rows.get(self.selected_index).map(|row| row.identity.clone());
        let selected_display_index = display_items
            .iter()
            .position(|item| match item {
                DisplayItem::Row(index) => {
                    Some(&self.rows[*index].identity) == selected_identity.as_ref()
                }
                _ => false,
            })
            .unwrap_or(0) as i64;
        let start = (selected_display_index - (visible_rows as i64 / 2))
            .clamp(0, (display_items.len() as i64 - visible_rows as i64).max(0)) as usize;
        let show_leading_ellipsis = start > 0 && visible_rows > 1;
        let show_trailing_ellipsis = start + visible_rows < display_items.len() && visible_rows > 2;
        let content_rows =
            visible_rows.saturating_sub(usize::from(show_leading_ellipsis) + usize::from(show_trailing_ellipsis));
        let slice_start = if selected_display_index as usize >= start + content_rows {
            (selected_display_index as usize).saturating_sub(content_rows.saturating_sub(1))
        } else {
            start
        };
        let mut lines: Vec<String> = display_items
            .iter()
            .skip(slice_start)
            .take(content_rows)
            .map(|item| match item {
                DisplayItem::Spacer => String::new(),
                DisplayItem::RunningSubagents(index) => {
                    let row = &self.rows[*index];
                    let count = row.running_subagent_count;
                    let indent = "  ".repeat(row.depth + 1);
                    self.theme.fg(
                        "success",
                        &truncate_to_width(
                            &format!("{indent}{count} subagent{} running", if count == 1 { "" } else { "s" }),
                            width,
                        ),
                    )
                }
                DisplayItem::Heading(section) => {
                    let collapsed = *section == AgentsViewSection::Inactive
                        && !self
                            .rows
                            .iter()
                            .any(|row| row.depth == 0 && row.section == AgentsViewSection::Inactive);
                    let prefix = if *section == AgentsViewSection::Inactive {
                        format!("{} ", if collapsed { "▸" } else { "▾" })
                    } else {
                        String::new()
                    };
                    let hint = if *section == AgentsViewSection::Inactive {
                        format!(" · {}", key_text(default_keybinding("app.agents.inactiveCollapse")))
                    } else {
                        String::new()
                    };
                    self.theme.bold(&truncate_to_width(
                        &format!(
                            "{prefix}{} ({}){hint}",
                            section_title(*section),
                            counts.get(section).copied().unwrap_or(0)
                        ),
                        width,
                    ))
                }
                DisplayItem::Row(index) => {
                    let row = self.rows[*index].clone();
                    self.render_row(&row, width, &layout)
                }
            })
            .collect();
        if show_leading_ellipsis {
            lines.insert(0, self.theme.fg("dim", "  ..."));
        }
        if show_trailing_ellipsis {
            lines.push(self.theme.fg("dim", "  ..."));
        }
        if header_rows > 0 {
            lines.insert(0, self.theme.fg("muted", &layout.legend));
        }
        lines
    }

    /// `renderRow(row, width, layout)`.
    pub fn render_row(&self, row: &AgentsViewRow, width: usize, layout: &AgentsViewUsageLayout) -> String {
        let selected = row.selectable
            && Some(&row.identity) == self.rows.get(self.selected_index).map(|row| &row.identity);
        let mark_row = |line: String| -> String {
            if selected {
                format!("{SELECTED_ROW_MARKER}{line}")
            } else {
                line
            }
        };
        if row.kind == AgentsViewRowKind::SubagentCode {
            return self.render_code_row(row);
        }
        let pending_delete = row.kind == AgentsViewRowKind::Agent && self.is_pending_delete_row(row);
        let pending_kill = row.kind == AgentsViewRowKind::Subagent && self.is_pending_kill_subagent_row(row);
        let details = layout.details.get(&row.identity).cloned().unwrap_or_default();
        if pending_delete || pending_kill {
            let armed = row.summary.has_active_heartbeat == Some(true)
                || row.heartbeat.as_ref().map(|heartbeat| heartbeat.active_count).unwrap_or(0) > 0;
            let title = format!(
                "{}{}",
                if armed { "has an armed heartbeat — " } else { "" },
                if pending_delete {
                    self.get_pending_delete_title()
                } else {
                    format!(
                        "{} again to {}",
                        key_text(default_keybinding("app.agents.delete")),
                        if has_live_work(row) { "stop" } else { "delete" }
                    )
                }
            );
            return mark_row(format_table_cell(&self.theme.fg("error", &title), width));
        }
        let icon = self.format_row_icon(row.section, self.get_row_icon(row.section));
        let expand = if row.descendant_count > 0 {
            if self.expanded_subagent_parents.contains(&row.identity) {
                "▾"
            } else {
                "▸"
            }
        } else {
            " "
        };
        let badge = format_heartbeat_badge(row.heartbeat.as_ref(), super::agents_view_state::now_ms());
        let heartbeat = if badge.is_empty() {
            String::new()
        } else {
            format!(
                "{} ",
                self.theme.fg(
                    if row.heartbeat.as_ref().map(|heartbeat| heartbeat.active_count).unwrap_or(0) > 0 {
                        "error"
                    } else {
                        "dim"
                    },
                    &badge,
                )
            )
        };
        let title = format!(
            "{}{icon}{expand} {heartbeat}{}",
            "  ".repeat(row.depth),
            style_row_title(row, self.theme.as_ref())
        );
        let status = if row.summary.status_label.is_some() || row.summary.last_heard_from_at.is_some() {
            Some(row.status_label.clone())
        } else {
            None
        };
        let activity = [status, row.summary.summary.clone()]
            .into_iter()
            .flatten()
            .filter(|value| !value.is_empty())
            .collect::<Vec<String>>()
            .join(" · ");
        let mut cells = vec![
            format_table_cell(&title, layout.name_width),
            format_table_cell(&self.theme.fg("muted", &format_session_model(&row.summary)), layout.model_width),
        ];
        if layout.activity_width > 0 {
            cells.push(format_table_cell(&self.theme.fg("dim", &activity), layout.activity_width));
        }
        cells.push(details);
        mark_row(format_table_cell(&cells.join("  "), width))
    }

    /// `renderActions(width)`.
    pub fn render_actions(&self, width: usize) -> Vec<String> {
        let row = self.rows.get(self.selected_index);
        let mut actions = vec![
            format!(
                "{} open   {} open   {} new",
                key_text(default_keybinding("tui.select.confirm")),
                key_text(default_keybinding("app.agents.open")),
                key_text(default_keybinding("app.agents.new"))
            ),
            format!(
                "{} expand/collapse subagents   {} program",
                key_text(default_keybinding("app.agents.expand")),
                key_text(default_keybinding("app.agents.program"))
            ),
            format!(
                "{} show/hide inactive   {} close actions",
                key_text(default_keybinding("app.agents.inactiveCollapse")),
                key_text(default_keybinding("app.shortcuts"))
            ),
            format!(
                "{} reply/resume   {} rename   {} stop/delete",
                key_text(default_keybinding("app.agents.reply")),
                key_text(default_keybinding("app.agents.rename")),
                key_text(default_keybinding("app.agents.delete"))
            ),
        ];
        if let Some(row) = row {
            let model = row.summary.model.as_ref();
            let usage = row.summary.usage.as_ref();
            actions.push(String::new());
            actions.push(row.title.clone());
            actions.push(format!(
                "Model: {}{}",
                model.map(|model| format!("{}/{}", model.provider, model.id)).unwrap_or_else(|| "unknown".to_string()),
                row.summary
                    .thinking_level
                    .as_ref()
                    .map(|level| format!(" · {level}"))
                    .unwrap_or_default()
            ));
            actions.push(format!("Directory: {}", row.summary.cwd));
            actions.push(format!(
                "Tokens: {} in · {} out",
                usage.map(|usage| usage.input_tokens).unwrap_or(0),
                usage.map(|usage| usage.output_tokens).unwrap_or(0)
            ));
            actions.push(format!(
                "Cost: ${:.2} session · ${:.2} including subagents",
                usage.map(|usage| usage.cost).unwrap_or(0.0),
                row.recursive_cost
            ));
        }
        actions
            .into_iter()
            .flat_map(|line| wrap_text_with_ansi(&self.theme.fg("muted", &line), width))
            .collect()
    }

    // Spawn-code rows are read-only context. They render deemphasized — muted
    // text on a panel background (applied in finalizeRenderedLine) so the program
    // reads as one quiet segmented block rather than competing with agent rows.
    fn render_code_row(&self, row: &AgentsViewRow) -> String {
        let indent = "  ".repeat(row.depth);
        let body = self.theme.fg("muted", row.code.as_deref().unwrap_or(" "));
        format!("{CODE_ROW_MARKER}{indent}  {body}")
    }

    fn finalize_rendered_line(&self, line: &str, width: usize) -> String {
        let code = line.starts_with(CODE_ROW_MARKER);
        let selected = !code && line.starts_with(SELECTED_ROW_MARKER);
        let mut content = if code {
            line[CODE_ROW_MARKER.len()..].to_string()
        } else if selected {
            line[SELECTED_ROW_MARKER.len()..].to_string()
        } else {
            line.to_string()
        };
        // Each rendered line must occupy exactly one terminal row; a stray
        // newline would shift every line below it and overlap the editor.
        if content.contains('\n') || content.contains('\r') {
            content = content
                .split(|ch| ch == '\n' || ch == '\r')
                .filter(|part| !part.is_empty())
                .collect::<Vec<&str>>()
                .join(" ");
        }
        let padded = pad_line(&truncate_to_width(&content, width), width);
        if code {
            return self.theme.bg("toolPanelBg", &padded);
        }
        if !selected {
            return padded;
        }
        // Truncating styled cells embeds full \x1b[0m resets; re-open the
        // selection background after each so the highlight spans the whole row.
        let apply_selection_bg = self.theme.selection_background_color();
        padded
            .split("\u{1b}[0m")
            .map(|part| apply_selection_bg(part))
            .collect::<Vec<String>>()
            .join("\u{1b}[0m")
    }

    /// `renderDock(width)`.
    pub fn render_dock(&self, width: usize) -> Vec<String> {
        let safe_width = width.max(1);
        vec![self.finalize_rendered_line(&self.render_hints(safe_width), safe_width)]
    }

    /// `renderHints(width)`.
    pub fn render_hints(&self, width: usize) -> String {
        if self.is_ctrl_c_exit_hint_visible() {
            let clear_key = key_text(default_keybinding("app.clear"));
            let hint = if clear_key.is_empty() {
                "Press again to exit".to_string()
            } else {
                format!("Press {clear_key} again to exit")
            };
            return truncate_to_width(&self.theme.fg("muted", &hint), width);
        }
        if let Some(status_message) = &self.status_message {
            return truncate_to_width(&self.theme.fg(self.status_message_tone.as_str(), status_message), width);
        }
        if self.rename_target.is_some() {
            let hint = format!(
                "{} save   {} cancel",
                key_text(default_keybinding("tui.select.confirm")),
                key_text(default_keybinding("tui.select.cancel"))
            );
            return truncate_to_width(&self.theme.fg("muted", &hint), width);
        }
        if self.reply_target.is_some() {
            return truncate_to_width(&self.theme.fg("muted", &self.render_reply_composer_hints()), width);
        }
        let hints = format!(
            "{}/{} navigate   {} open   {} new   {} actions",
            key_text(default_keybinding("tui.select.up")),
            key_text(default_keybinding("tui.select.down")),
            key_text(default_keybinding("tui.select.confirm")),
            key_text(default_keybinding("app.agents.new")),
            key_text(default_keybinding("app.shortcuts"))
        );
        truncate_to_width(&self.theme.fg("muted", &hints), width)
    }

    /// `renderReplyComposerHints()`.
    pub fn render_reply_composer_hints(&self) -> String {
        let Some(target) = self.reply_target.clone() else {
            return String::new();
        };
        let records = self.unified_records.clone();
        let rows = self.rows.clone();
        let current = resolve_current_reply_target_summary(&records, &target, &|active_session_id| {
            rows.iter()
                .find(|row| {
                    row.summary.active_session_id.as_deref().unwrap_or(row.summary.id.as_str()) == active_session_id
                })
                .map(|row| row.summary.clone())
        });
        let streaming = current.active_session_id.is_some() && current.is_streaming;
        let has_text = !self.editor.get_text().trim().is_empty();
        [
            Some(format!(
                "{} {}",
                key_text(default_keybinding("tui.select.confirm")),
                if streaming {
                    "steer"
                } else if current.active_session_id.is_some() {
                    "send"
                } else {
                    "resume & send"
                }
            )),
            if has_text {
                Some(format!("{} queue", key_text(default_keybinding("app.message.followUp"))))
            } else {
                None
            },
            Some(format!("{} cancel", key_text(default_keybinding("tui.select.cancel")))),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<String>>()
        .join("   ")
    }

    fn visible_list_rows(&self) -> usize {
        self.terminal.rows().saturating_sub(9).max(4)
    }

    pub fn get_splash_model_id(&self) -> Option<String> {
        self.rows
            .get(self.selected_index)
            .and_then(|row| row.summary.model.as_ref().map(|model| model.id.clone()))
            .or_else(|| self.options.startup_model_id.clone())
    }

    pub fn get_splash_cwd(&self) -> String {
        self.rows
            .get(self.selected_index)
            .map(|row| row.summary.cwd.clone())
            .unwrap_or_else(|| self.options.ui_services.get_initial_cwd())
    }

    fn get_row_icon(&self, section: AgentsViewSection) -> String {
        match section {
            AgentsViewSection::Running => working_icon_frame(self.working_icon_frame).to_string(),
            AgentsViewSection::Idle => NEEDS_INPUT_ROW_ICON.to_string(),
            AgentsViewSection::Inactive => COMPLETED_ROW_ICON.to_string(),
        }
    }

    fn format_row_icon(&self, section: AgentsViewSection, icon: String) -> String {
        match section {
            AgentsViewSection::Running => self.theme.bold(&icon),
            AgentsViewSection::Idle => self.theme.fg("warning", &icon),
            AgentsViewSection::Inactive => self.theme.fg("dim", &icon),
        }
    }

    fn get_pending_delete_title(&self) -> String {
        let delete_key = key_text(default_keybinding("app.agents.delete"));
        if self.pending_delete_agent.as_ref().map(|pending| pending.stopped).unwrap_or(false) {
            format!("stopped - {delete_key} again to remove")
        } else {
            format!("{delete_key} again to remove")
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::agents_view::agents_view_state::{AgentRosterStatus, RuntimeKind, SessionUsageSummary};
    use crate::modes::agents_view::roster_store::{
        AgentRosterEntry, DaemonHello, DaemonOutbound, DaemonResponse, DaemonTransport, MessageListener,
        TransportFuture,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex as StdMutex;

    struct TestUiServices;

    impl AgentsViewUiServices for TestUiServices {
        fn get_initial_cwd(&self) -> String {
            "C:/work".to_string()
        }
        fn get_theme(&self) -> String {
            "optimus".to_string()
        }
        fn get_themes(&self) -> Vec<String> {
            vec!["optimus".to_string()]
        }
        fn get_show_hardware_cursor(&self) -> bool {
            false
        }
        fn get_clear_on_shrink(&self) -> bool {
            true
        }
        fn get_editor_padding_x(&self) -> usize {
            1
        }
        fn get_autocomplete_max_visible(&self) -> usize {
            8
        }
    }

    struct TestTerminal {
        rows: usize,
        renders: StdMutex<usize>,
        titles: StdMutex<Vec<String>>,
    }

    impl AgentsViewTerminal for TestTerminal {
        fn rows(&self) -> usize {
            self.rows
        }
        fn request_render(&self, _force: bool) {
            *self.renders.lock().unwrap() += 1;
        }
        fn set_title(&self, title: &str) {
            self.titles.lock().unwrap().push(title.to_string());
        }
        fn columns(&self) -> usize { 80 }
        fn poll_input(&self) -> Result<Option<Vec<String>>, String> { Ok(Some(Vec::new())) }
        fn present(&self, _lines: Vec<String>, _dock: Vec<String>) -> Result<(), String> { Ok(()) }
    }

    #[derive(Default)]
    struct TestEditor {
        text: String,
        placeholder: String,
    }

    impl AgentsViewEditor for TestEditor {
        fn set_text(&mut self, text: &str) {
            self.text = text.to_string();
        }
        fn get_text(&self) -> String {
            self.text.clone()
        }
        fn get_expanded_text(&self) -> String {
            self.text.clone()
        }
        fn set_placeholder(&mut self, text: &str) {
            self.placeholder = text.to_string();
        }
        fn render(&mut self, width: usize) -> Vec<String> {
            vec![format_table_cell(&format!("> {}", self.text), width)]
        }
        fn invalidate(&mut self) {}
        fn get_lines(&self) -> Vec<String> {
            vec![self.text.clone()]
        }
        fn get_cursor(&self) -> (usize, usize) {
            (0, self.text.chars().count())
        }
        fn handle_input(&mut self, data: &str) -> bool {
            if data.chars().count() == 1 && data.chars().all(|character| !character.is_control()) {
                self.text.push_str(data);
                return true;
            }
            false
        }
        fn focus(&mut self) {}
        fn is_focused(&self) -> bool {
            true
        }
        fn take_submissions(&mut self) -> Vec<String> { Vec::new() }
    }

    /// Fake daemon transport with scripted responses.
    pub struct FakeTransport {
        hello: StdMutex<Option<DaemonHello>>,
        connected: AtomicBool,
        requests: StdMutex<Vec<Value>>,
        responses: StdMutex<Vec<Result<DaemonResponse, String>>>,
    }

    impl FakeTransport {
        pub fn new(capabilities: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                hello: StdMutex::new(Some(DaemonHello {
                    socket_path: "pipe".into(),
                    server_capabilities: capabilities.iter().map(|value| value.to_string()).collect(),
                    client_id: "client-1".into(),
                    schema_revision: None,
                })),
                connected: AtomicBool::new(true),
                requests: StdMutex::new(Vec::new()),
                responses: StdMutex::new(Vec::new()),
            })
        }

        pub fn push(&self, response: Result<DaemonResponse, String>) {
            self.responses.lock().unwrap().push(response);
        }

        pub fn requests(&self) -> Vec<Value> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl DaemonTransport for FakeTransport {
        fn hello(&self) -> Option<DaemonHello> {
            self.hello.lock().unwrap().clone()
        }
        fn is_connected(&self) -> bool {
            self.connected.load(Ordering::SeqCst)
        }
        fn supports_server_capability(&self, capability: &str) -> bool {
            self.hello
                .lock()
                .unwrap()
                .as_ref()
                .map(|hello| hello.server_capabilities.iter().any(|value| value == capability))
                .unwrap_or(false)
        }
        fn wait_for_hello(&self, _timeout_ms: u64) -> TransportFuture<Result<DaemonHello, String>> {
            let hello = self.hello();
            Box::pin(async move { hello.ok_or_else(|| "no hello".to_string()) })
        }
        fn request(
            &self,
            command: Value,
            _timeout_ms: u64,
            _options: DaemonClientRequestOptions,
        ) -> TransportFuture<Result<DaemonResponse, String>> {
            self.requests.lock().unwrap().push(command);
            let mut queue = self.responses.lock().unwrap();
            let response = if queue.is_empty() {
                Err("no scripted response".to_string())
            } else {
                queue.remove(0)
            };
            Box::pin(async move { response })
        }
        fn on_message(&self, _listener: MessageListener) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }
        fn on_close(&self, _listener: Box<dyn Fn(&str) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }
        fn connect(&self, _timeout_ms: u64) -> TransportFuture<Result<(), String>> {
            Box::pin(async { Ok(()) })
        }
        fn reconnect(&self, _timeout_ms: u64) -> TransportFuture<Result<(), String>> {
            Box::pin(async { Ok(()) })
        }
        fn close(&self) {
            self.connected.store(false, Ordering::SeqCst);
        }
        fn socket_path(&self) -> String {
            "pipe".to_string()
        }
    }

    struct FakeAgentConnection;

    impl DaemonAgentConnectionHandle for FakeAgentConnection {
        fn prompt(
            &self,
            _message: &str,
            _streaming_behavior: Option<&str>,
        ) -> TransportFuture<Result<(), String>> {
            Box::pin(async { Ok(()) })
        }
        fn dispose(&self) -> TransportFuture<Result<(), String>> {
            Box::pin(async { Ok(()) })
        }
    }

    struct FakeConnectionFactory;

    struct InputDriver {
        input: StdMutex<std::collections::VecDeque<Vec<String>>>,
        frames: StdMutex<Vec<Vec<String>>>,
    }
    impl AgentsViewTerminal for InputDriver {
        fn rows(&self) -> usize { 30 }
        fn columns(&self) -> usize { 80 }
        fn request_render(&self, _force: bool) {}
        fn set_title(&self, _title: &str) {}
        fn poll_input(&self) -> Result<Option<Vec<String>>, String> {
            Ok(Some(self.input.lock().unwrap().pop_front().unwrap_or_default()))
        }
        fn present(&self, lines: Vec<String>, _dock: Vec<String>) -> Result<(), String> {
            self.frames.lock().unwrap().push(lines); Ok(())
        }
    }

    /// Stateless, so the tests borrow one `'static` instance like the module-level
    /// factory the reference passes into the mode.
    static FAKE_CONNECTION_FACTORY: FakeConnectionFactory = FakeConnectionFactory;

    impl DaemonAgentConnectionFactory for FakeConnectionFactory {
        fn attach(
            &self,
            _client: DaemonTransportClient,
            _active_session_id: &str,
            _options: AttachOptions,
        ) -> TransportFuture<Result<Arc<dyn DaemonAgentConnectionHandle>, String>> {
            Box::pin(async { Ok(Arc::new(FakeAgentConnection) as Arc<dyn DaemonAgentConnectionHandle>) })
        }
    }

    struct FakeInteractiveMode;

    impl InteractiveModeHandle for FakeInteractiveMode {
        fn run(&mut self) -> TransportFuture<Result<InteractiveRunResult, String>> {
            Box::pin(async {
                Ok(InteractiveRunResult {
                    kind: "back".to_string(),
                    source: SessionSummary::new("a-1", "s-1", "C:/work"),
                })
            })
        }
        fn teardown_session_ui(&mut self, _preserve_alt_screen: bool) -> TransportFuture<Result<(), String>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn interactive_factory() -> InteractiveModeFactory {
        Box::new(|_options: InteractiveModeOptions| Box::new(FakeInteractiveMode) as Box<dyn InteractiveModeHandle>)
    }

    fn summary(agent_id: &str, name: &str) -> SessionSummary {
        let mut summary = SessionSummary::new(agent_id, agent_id, "C:/work");
        summary.session_name = Some(name.to_string());
        summary.active_session_id = Some(agent_id.to_string());
        summary.message_count = 2;
        summary.usage = Some(SessionUsageSummary { input_tokens: 10, output_tokens: 5, cost: 1.25 });
        summary
    }

    fn roster_entry(agent_id: &str, name: &str) -> AgentRosterEntry {
        AgentRosterEntry {
            agent_id: agent_id.to_string(),
            summary: summary(agent_id, name),
            status: AgentRosterStatus::Idle,
            ..AgentRosterEntry::default()
        }
    }

    fn ok_response(data: Value) -> DaemonResponse {
        DaemonResponse { command: "x".into(), success: true, data: Some(data), error: None }
    }

    /// Build a mode with a roster snapshot already applied, without running the loop.
    async fn build_mode(entries: Vec<AgentRosterEntry>) -> AgentsViewMode<'static> {
        build_mode_with(entries, None, AgentsViewPersistentState::default()).await
    }

    async fn build_mode_with(
        entries: Vec<AgentRosterEntry>,
        initial_session: Option<SessionSummary>,
        persistent_state: AgentsViewPersistentState,
    ) -> AgentsViewMode<'static> {
        crate::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        let options = AgentsViewModeOptions {
            socket_path: Some("pipe".to_string()),
            config: AgentsViewRuntimeConfig::default(),
            ui_services: Arc::new(TestUiServices),
            migrated_providers: None,
            model_fallback_message: None,
            startup_model_id: Some("model-1".to_string()),
            verbose: None,
            reconnect_timeout_ms: None,
            initial_session,
            initial_scope_key: None,
        };
        let transport = FakeTransport::new(&["agent_roster", "heartbeat_catalog"]);
        transport.push(Ok(ok_response(serde_json::json!({
            "roster": entries
                .iter()
                .map(|entry| serde_json::to_value(entry).unwrap())
                .collect::<Vec<Value>>()
        }))));
        // The mode stores the factory by reference for its whole run, so the
        // helper leaks one stateless factory per test.
        let interactive: &'static mut InteractiveModeFactory =
            Box::leak(Box::new(interactive_factory()));
        let mut mode = AgentsViewMode::new(
            options,
            persistent_state,
            Arc::new(TestTerminal { rows: 30, renders: StdMutex::new(0), titles: StdMutex::new(Vec::new()) }),
            Box::new(TestEditor::default()),
            Arc::new(PlainAgentsViewTheme),
            transport.clone(),
            &FAKE_CONNECTION_FACTORY,
            interactive,
            None,
        );
        mode.set_scripted_transport(transport);
        mode.attach_roster_for_test().await;
        mode
    }

    #[test]
    fn constants_match_the_typescript() {
        assert_eq!(HEARTBEAT_POLL_INTERVAL_MS, 15000);
        assert_eq!(RECONNECT_TIMEOUT_MS, 120000);
        assert_eq!(RECONNECT_RETRY_MS, 1000);
        assert_eq!(EXIT_HINT_DURATION_MS, 2000);
        assert_eq!(DELETE_CONFIRM_DURATION_MS, 2000);
        assert_eq!(STATUS_MESSAGE_DURATION_MS, 4500);
        assert_eq!(SEARCH_PROMPT_PLACEHOLDER, "Search sessions");
        assert_eq!(REPLY_PROMPT_FALLBACK_PLACEHOLDER, "Write a reply to this agent");
        assert_eq!(RESUME_PROMPT_PLACEHOLDER, "Write a prompt to resume this session");
        assert_eq!(COMPLETED_ROW_ICON, "✓");
        assert_eq!(NEEDS_INPUT_ROW_ICON, "●");
        assert_eq!(SELECTED_ROW_MARKER, "\u{0}agents-view-selected-row\u{0}");
        assert_eq!(CODE_ROW_MARKER, "\u{0}agents-view-code-row\u{0}");
        assert_eq!(WORKING_ICON_INTERVAL_MS, 250);
    }

    #[test]
    fn working_icon_cycles_through_the_frames() {
        assert_eq!(working_icon_frame(0), "◇");
        assert_eq!(working_icon_frame(1), "◈");
        assert_eq!(working_icon_frame(2), "◆");
        assert_eq!(working_icon_frame(3), "◈");
        assert_eq!(working_icon_frame(4), "◇");
        assert_eq!(working_icon_frame(-1), "◈");
    }

    #[test]
    fn slash_command_parsing_matches_the_reference() {
        assert_eq!(parse_slash_command("/name My agent"), Some(("name".to_string(), "My agent".to_string())));
        assert_eq!(parse_slash_command("/name"), Some(("name".to_string(), String::new())));
        assert_eq!(parse_slash_command("/name   spaced  "), Some(("name".to_string(), "spaced".to_string())));
        assert_eq!(parse_slash_command("name"), None);
        assert_eq!(parse_slash_command("/"), None);
        assert_eq!(resolve_builtin_slash_command_name("clear"), "new");
        assert_eq!(resolve_builtin_slash_command_name("model"), "model");
        assert!(is_builtin_slash_command_name("clear"));
        assert!(!is_builtin_slash_command_name("nope"));
    }

    #[test]
    fn view_commands_and_rejections_match_the_reference() {
        let command = parse_agents_view_command("/name Foo").unwrap();
        assert_eq!(command.name, AgentsViewCommandName::Name);
        assert_eq!(command.args, "Foo");
        assert_eq!(parse_agents_view_command("/rename Foo").unwrap().name, AgentsViewCommandName::Name);
        assert_eq!(parse_agents_view_command("/kill").unwrap().name, AgentsViewCommandName::Kill);
        assert!(parse_agents_view_command("/model").is_none());
        assert!(parse_agents_view_command("plain text").is_none());

        assert_eq!(get_reply_composer_command_rejection("/compact"), None);
        assert_eq!(get_reply_composer_command_rejection("/name x"), None);
        assert_eq!(
            get_reply_composer_command_rejection("/model"),
            Some("/model is not available here; open the session to run it".to_string())
        );
        assert_eq!(get_reply_composer_command_rejection("/notacommand"), None);
    }

    #[test]
    fn status_line_and_notices_flatten_whitespace() {
        assert_eq!(format_agents_view_status_line("  a\n b\t c "), "a b c");
        assert_eq!(combine_agents_view_startup_notices(&[None, Some(""), None]), None);
        assert_eq!(
            combine_agents_view_startup_notices(&[Some("one\n two"), Some("three")]),
            Some("one two · three".to_string())
        );
        assert_eq!(create_agents_view_reply_headline(Some("  first\nsecond")), Some("first".to_string()));
        assert_eq!(create_agents_view_reply_headline(Some("\n\n")), None);
        assert_eq!(create_agents_view_reply_headline(None), None);
    }

    #[test]
    fn daemon_reconnect_and_depth_helpers() {
        assert!(should_reconnect_agents_view_daemon(None));
        assert!(should_reconnect_agents_view_daemon(Some("update")));
        assert!(!should_reconnect_agents_view_daemon(Some("shutdown")));

        let mut root = summary("a", "Root");
        root.rlm_depth = Some(2);
        assert_eq!(get_agents_view_depth(None), 0);
        assert_eq!(get_agents_view_depth(Some(&root)), 3);
    }

    #[test]
    fn resume_config_strips_or_overrides_cwd() {
        let config = AgentsViewRuntimeConfig {
            cwd: Some("C:/original".to_string()),
            session_dir: Some("C:/sessions".to_string()),
            telemetry_disabled: Some(true),
        };
        let stripped = create_agents_view_resume_config(&config, None);
        assert_eq!(stripped.cwd, None);
        assert_eq!(stripped.session_dir, Some("C:/sessions".to_string()));
        let overridden = create_agents_view_resume_config(&config, Some("C:/fallback"));
        assert_eq!(overridden.cwd, Some("C:/fallback".to_string()));
    }

    #[test]
    fn initial_scope_frames_carry_the_return_chat_only_for_the_same_session() {
        let scope = AgentsViewScopeKey { session_id: "s-1".into(), active_session_id: Some("a-1".into()) };
        let mut chat = summary("a-1", "Chat");
        chat.session_id = "s-1".to_string();
        let frames = create_initial_agents_view_scope_frames(Some(&scope), Some(&chat));
        assert_eq!(frames.len(), 1);
        assert!(frames[0].return_chat.is_some());

        let other = summary("a-2", "Other");
        let frames = create_initial_agents_view_scope_frames(Some(&scope), Some(&other));
        assert!(frames[0].return_chat.is_none());
        assert!(create_initial_agents_view_scope_frames(None, Some(&chat)).is_empty());
    }

    #[test]
    fn initial_persistent_state_seeds_selection_only_without_a_scope() {
        let chat = summary("a-1", "Chat");
        let unscoped = create_initial_agents_view_persistent_state(None, Some(&chat));
        assert!(unscoped.back_session.is_some());
        assert!(unscoped.selected_row_identity.is_some());
        assert!(unscoped.selected_session_key.is_some());
        assert!(unscoped.scope_frames.is_none());

        let scope = AgentsViewScopeKey { session_id: "s-1".into(), active_session_id: None };
        let scoped = create_initial_agents_view_persistent_state(Some(&scope), Some(&chat));
        assert!(scoped.selected_row_identity.is_none());
        assert_eq!(scoped.scope_frames.map(|frames| frames.len()), Some(1));
        assert_eq!(scoped.last_successful_live_summaries.map(|value| value.len()), Some(1));
    }

    #[test]
    fn scope_back_return_chat_open_result_only_fires_with_a_return_chat() {
        let chat = summary("a-1", "Chat");
        let result = AgentsViewRunResult::ScopeBack {
            selection: summary("a-2", "Sel"),
            expanded_ancestor_session_ids: vec!["root".to_string()],
            return_chat: Some(chat.clone()),
            has_children: true,
        };
        let open = create_scope_back_return_chat_open_result(&result).unwrap();
        match open {
            AgentsViewRunResult::Open { summary, expanded_ancestor_session_ids, has_children, .. } => {
                assert_eq!(summary.session_id, chat.session_id);
                assert_eq!(expanded_ancestor_session_ids, Some(vec!["root".to_string()]));
                assert_eq!(has_children, Some(true));
            }
            _ => panic!("expected open"),
        }
        let without_chat = AgentsViewRunResult::ScopeBack {
            selection: summary("a-2", "Sel"),
            expanded_ancestor_session_ids: Vec::new(),
            return_chat: None,
            has_children: false,
        };
        assert!(create_scope_back_return_chat_open_result(&without_chat).is_none());
        assert!(create_scope_back_return_chat_open_result(&AgentsViewRunResult::Exit).is_none());
    }

    #[test]
    fn open_cwd_falls_back_only_when_the_directory_is_missing() {
        let mut summary = summary("a-1", "Chat");
        summary.cwd = "definitely-not-a-real-directory-xyz".to_string();
        let (override_cwd, notice) = resolve_agents_view_open_cwd(&summary, Some("C:/fallback"));
        assert_eq!(override_cwd, Some("C:/fallback".to_string()));
        assert_eq!(
            notice,
            Some("Original directory is missing (definitely-not-a-real-directory-xyz); opened in C:/fallback instead.".to_string())
        );

        let (override_cwd, notice) = resolve_agents_view_open_cwd(&summary, None);
        assert_eq!(override_cwd, None);
        assert_eq!(notice, None);

        summary.cwd = std::env::current_dir().unwrap().to_string_lossy().to_string();
        let (override_cwd, notice) = resolve_agents_view_open_cwd(&summary, Some("C:/fallback"));
        assert_eq!(override_cwd, None);
        assert_eq!(notice, None);
    }

    #[test]
    fn daemon_payload_validation_matches_the_reference_errors() {
        let bad = serde_json::json!({ "sessions": [{}] });
        assert_eq!(
            expect_session_list(&bad),
            Err("Daemon returned an invalid session summary".to_string())
        );
        assert_eq!(
            expect_session_list(&serde_json::json!({})),
            Err("Daemon returned an invalid session list response".to_string())
        );
        let good = serde_json::json!({ "sessions": [serde_json::to_value(summary("a-1", "Chat")).unwrap()] });
        assert_eq!(expect_session_list(&good).unwrap().len(), 1);
        assert_eq!(
            expect_session_summary(&serde_json::json!({})),
            Err("Daemon returned an invalid session summary".to_string())
        );
        assert!(expect_session_summary(&serde_json::to_value(summary("a-1", "Chat")).unwrap()).is_ok());
        assert_eq!(
            require_daemon_data(&DaemonResponse {
                command: "x".into(),
                success: false,
                data: None,
                error: Some("boom".to_string())
            }),
            Err("boom".to_string())
        );
    }

    #[test]
    fn error_and_command_helpers_match_the_reference_strings() {
        assert_eq!(format_error("Failed to open agent", "boom"), "Failed to open agent: boom");
        assert!(is_unknown_active_session_error("Unknown active session: a-1"));
        assert!(!is_unknown_active_session_error("other"));
        assert!(is_unknown_daemon_command_error("Unknown daemon command: rename", "rename"));
        assert!(!is_unknown_daemon_command_error("Unknown daemon command: rename", "kill"));
    }

    #[test]
    fn prompt_command_shape_matches_the_wire() {
        assert_eq!(
            create_prompt_command("a-1", "hi", None),
            serde_json::json!({ "type": "prompt", "activeSessionId": "a-1", "message": "hi" })
        );
        assert_eq!(
            create_prompt_command("a-1", "hi", Some("followUp"))["streamingBehavior"],
            "followUp"
        );
        assert_eq!(create_agents_view_list_command()["type"], "list");
    }

    #[test]
    fn layout_and_row_helpers_match_the_reference() {
        let rows = vec![
            AgentsViewRow {
                kind: AgentsViewRowKind::Agent,
                section: AgentsViewSection::Running,
                identity: "a".into(),
                title: "Alpha".into(),
                recursive_cost: 1.5,
                summary: summary("a-1", "Alpha"),
                ..AgentsViewRow::default()
            },
            AgentsViewRow {
                kind: AgentsViewRowKind::Agent,
                section: AgentsViewSection::Inactive,
                identity: "b".into(),
                title: "Beta".into(),
                summary: summary("b-1", "Beta"),
                ..AgentsViewRow::default()
            },
        ];
        let layout = build_compact_agents_view_layout(&rows, 120, &PlainAgentsViewTheme);
        assert!(layout.legend.contains("Session"));
        assert!(layout.legend.contains("Cost"));
        assert!(layout.details.get("a").unwrap().contains("$1.50"));

        let counts = count_rows_by_section(&rows);
        assert_eq!(counts[&AgentsViewSection::Running], 1);
        assert_eq!(counts[&AgentsViewSection::Inactive], 1);
        assert_eq!(counts[&AgentsViewSection::Idle], 0);

        assert_eq!(compact_session_rows(&rows, false).len(), 1);
        assert_eq!(compact_session_rows(&rows, true).len(), 2);
        assert_eq!(get_display_rows_for_section(&rows, AgentsViewSection::Inactive).len(), 1);
    }

    #[test]
    fn inactive_visibility_defaults_to_expanded() {
        let mut state = AgentsViewPersistentState::default();
        assert!(is_inactive_expanded(&state));
        state.inactive_visibility_explicit = Some(true);
        state.inactive_expanded = Some(false);
        assert!(!is_inactive_expanded(&state));
        state.inactive_expanded = Some(true);
        assert!(is_inactive_expanded(&state));
    }

    #[test]
    fn live_work_detection_covers_the_subtree() {
        let mut row = AgentsViewRow {
            section: AgentsViewSection::Idle,
            running_subagent_count: 0,
            summary: summary("a-1", "Alpha"),
            ..AgentsViewRow::default()
        };
        assert!(!has_live_work(&row));
        row.running_subagent_count = 2;
        assert!(has_live_work(&row));
        row.running_subagent_count = 0;
        row.section = AgentsViewSection::Running;
        assert!(has_live_work(&row));
        row.section = AgentsViewSection::Idle;
        row.summary.has_running_rlm_children = Some(true);
        assert!(has_live_work(&row));
    }

    #[test]
    fn key_text_formats_bindings_like_the_reference() {
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        assert_eq!(key_text("escape"), "Esc");
        assert_eq!(key_text("up"), "↑");
        assert_eq!(key_text("ctrl+o"), "Ctrl+O");
        assert_eq!(key_text("alt+enter"), "Alt+Enter");
        assert_eq!(key_text("up/down"), "↑/↓");
        assert_eq!(default_keybinding("app.clear"), "ctrl+c");
        assert!(matches_key("\x03", "app.clear"));
        assert!(!matches_key("\x18", "app.clear"));
    }

    #[test]
    fn width_helpers_are_ansi_and_wide_char_aware() {
        assert_eq!(visible_width("abc"), 3);
        assert_eq!(visible_width("\u{1b}[31mred\u{1b}[0m"), 3);
        assert_eq!(visible_width("日本"), 4);
        assert_eq!(truncate_to_width("abcdef", 3), "abc");
        assert_eq!(truncate_to_width("ab", 5), "ab");
        assert_eq!(pad_line("ab", 4), "ab  ");
        assert_eq!(format_table_cell("abcd", 2), "ab");
        assert_eq!(format_table_cell("ab", 4), "ab  ");
        assert_eq!(pad_cell_start("1.50", 6), "  1.50");
    }

    #[test]
    fn relative_time_matches_the_reference_buckets() {
        let now = 1_700_000_000_000i64;
        assert_eq!(format_agents_view_relative_time(None, now), "");
        assert_eq!(
            format_agents_view_relative_time(Some(&to_iso_string(&chrono::DateTime::from_timestamp_millis(now + 1000).unwrap())), now),
            "0s"
        );
        let thirty_seconds = now - 30_000;
        let stamp = to_iso_string(&chrono::DateTime::from_timestamp_millis(thirty_seconds).unwrap());
        assert_eq!(format_agents_view_relative_time(Some(&stamp), now), "30s");
        let two_hours = now - 2 * 60 * 60 * 1000;
        let stamp = to_iso_string(&chrono::DateTime::from_timestamp_millis(two_hours).unwrap());
        assert_eq!(format_agents_view_relative_time(Some(&stamp), now), "2h");
        let three_days = now - 3 * 24 * 60 * 60 * 1000;
        let stamp = to_iso_string(&chrono::DateTime::from_timestamp_millis(three_days).unwrap());
        assert_eq!(format_agents_view_relative_time(Some(&stamp), now), "3d");
        assert!(parse_session_timestamp(Some("")).is_none());
        assert!(parse_session_timestamp(Some("nonsense")).is_none());
    }

    #[tokio::test]
    async fn run_pumps_terminal_input_renders_rows_and_exits() {
        struct Driver {
            input: StdMutex<std::collections::VecDeque<Vec<String>>>,
            frames: StdMutex<Vec<Vec<String>>>,
        }
        impl AgentsViewTerminal for Driver {
            fn rows(&self) -> usize { 30 }
            fn columns(&self) -> usize { 80 }
            fn request_render(&self, _force: bool) {}
            fn set_title(&self, _title: &str) {}
            fn poll_input(&self) -> Result<Option<Vec<String>>, String> {
                Ok(self.input.lock().unwrap().pop_front())
            }
            fn present(&self, lines: Vec<String>, _dock: Vec<String>) -> Result<(), String> {
                self.frames.lock().unwrap().push(lines); Ok(())
            }
        }
        crate::core::keybindings::KeybindingsManager::new(Default::default(), None).install();
        let terminal = Arc::new(Driver {
            input: StdMutex::new(std::collections::VecDeque::from([
                vec!["\x1b[B".into()], vec!["\x03".into(), "\x03".into()],
            ])),
            frames: StdMutex::new(Vec::new()),
        });
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha"), roster_entry("b-1", "Beta")]).await;
        mode.terminal = terminal.clone();
        mode.push_response(ok_response(serde_json::json!({"heartbeats":[]})));
        mode.push_response(ok_response(serde_json::json!({"sessions":[]})));
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), mode.run()).await.unwrap().unwrap();
        assert_eq!(result, AgentsViewRunResult::Exit);
        assert_eq!(mode.selected_index(), 1);
        let frames = terminal.frames.lock().unwrap();
        assert!(frames.len() >= 2);
        assert!(frames.iter().all(|lines| lines.len() == 29));
        assert!(frames[0].iter().any(|line| line.contains("Alpha")));
        assert!(frames[1].iter().any(|line| line.contains("Beta")));
    }

    #[tokio::test]
    async fn mode_lists_roster_rows_and_sorts_by_section() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha"), roster_entry("b-1", "Beta")]).await;
        let rows = mode.rows().to_vec();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].title, "Alpha");
        assert!(rows.iter().all(|row| row.section == AgentsViewSection::Idle));
        let rendered = mode.render_view(80);
        assert!(rendered.iter().any(|line| line.contains("Alpha")));
        assert!(rendered.iter().any(|line| line.contains("Session")));
    }

    #[tokio::test]
    async fn mode_render_never_returns_more_rows_than_the_height() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        let rendered = mode.render_view(60);
        assert!(rendered.len() <= mode.content_height(60) + 1);
        assert!(rendered.iter().all(|line| !line.contains('\n')));
    }

    #[tokio::test]
    async fn mode_search_filters_rows_and_restores_on_clear() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha"), roster_entry("b-1", "Beta")]).await;
        mode.set_search_query_for_test("Alpha");
        assert_eq!(mode.rows().len(), 1);
        assert_eq!(mode.rows()[0].title, "Alpha");
        mode.set_search_query_for_test("");
        assert_eq!(mode.rows().len(), 2);
        mode.set_search_query_for_test("re:Beta");
        assert_eq!(mode.rows().len(), 1);
        mode.set_search_query_for_test("re:[");
        assert!(mode.rows().is_empty());
    }

    #[tokio::test]
    async fn mode_selection_moves_between_selectable_rows() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha"), roster_entry("b-1", "Beta")]).await;
        assert_eq!(mode.selected_index(), 0);
        mode.move_selection(1);
        assert_eq!(mode.selected_index(), 1);
        mode.move_selection(5);
        assert_eq!(mode.selected_index(), 1);
        mode.move_selection(-5);
        assert_eq!(mode.selected_index(), 0);
    }

    #[tokio::test]
    async fn reply_target_swaps_the_composer_placeholder_and_hints() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.select_row_for_test(0);
        mode.toggle_reply_target_for_test().await;
        assert!(mode.reply_target_armed());
        assert!(mode.render_reply_composer_hints().contains("send"));
        assert_eq!(mode.editor_placeholder(), REPLY_PROMPT_FALLBACK_PLACEHOLDER);
        mode.set_reply_target(None);
        assert!(!mode.reply_target_armed());
        assert_eq!(mode.editor_placeholder(), SEARCH_PROMPT_PLACEHOLDER);
    }

    #[tokio::test]
    async fn submit_rejects_builtin_commands_and_keeps_the_draft() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.select_row_for_test(0);
        mode.toggle_reply_target_for_test().await;
        mode.submit("/model", "steer").await;
        assert_eq!(
            mode.status_message(),
            Some("/model is not available here; open the session to run it")
        );
        assert_eq!(mode.editor_text(), "/model");
    }

    #[tokio::test]
    async fn name_command_renames_the_armed_target() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.select_row_for_test(0);
        mode.toggle_reply_target_for_test().await;
        mode.push_response(ok_response(serde_json::json!({ "ok": true })));
        mode.submit("/name Renamed", "steer").await;
        assert_eq!(mode.status_message(), Some("Renamed to Renamed"));
        let requests = mode.recorded_requests();
        assert!(requests.iter().any(|request| request["type"] == "rename"
            && request["name"] == "Renamed"
            && request["activeSessionId"] == "a-1"));
    }

    #[tokio::test]
    async fn name_command_without_an_argument_reports_usage() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.select_row_for_test(0);
        mode.toggle_reply_target_for_test().await;
        mode.submit("/name", "steer").await;
        assert_eq!(mode.status_message(), Some("Usage: /name <session name>"));
    }

    #[tokio::test]
    async fn kill_command_stops_the_running_agent() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.select_row_for_test(0);
        mode.toggle_reply_target_for_test().await;
        mode.push_response(ok_response(serde_json::json!({ "ok": true })));
        mode.submit("/kill", "steer").await;
        assert_eq!(mode.status_message(), Some("Agent stopped"));
        assert!(mode.recorded_requests().iter().any(|request| request["type"] == "kill"));
    }

    #[tokio::test]
    async fn delete_arms_confirmation_then_deactivates_on_second_press() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.select_row_for_test(0);
        mode.push_response(ok_response(serde_json::json!({ "ok": true })));
        mode.press_delete().await;
        assert!(mode.delete_confirmation_armed());
        assert!(mode.rows()[0].title.contains("Alpha"));
        mode.push_response(ok_response(serde_json::json!({ "ok": true })));
        mode.press_delete().await;
        assert_eq!(mode.status_message(), Some("Agent inactive"));
        assert!(mode
            .recorded_requests()
            .iter()
            .filter(|request| request["type"] == "kill")
            .count()
            >= 1);
    }

    #[tokio::test]
    async fn rename_mode_restores_the_search_query_on_cancel() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.set_search_query_for_test("Alpha");
        mode.select_row_for_test(0);
        mode.enter_rename_mode();
        assert!(mode.rename_mode_active());
        assert_eq!(mode.editor_text(), "Alpha");
        mode.exit_rename_mode();
        assert!(!mode.rename_mode_active());
        assert_eq!(mode.editor_text(), "Alpha");
        assert_eq!(mode.editor_placeholder(), SEARCH_PROMPT_PLACEHOLDER);
    }

    #[tokio::test]
    async fn scope_back_with_a_return_chat_finishes_with_an_open() {
        let mut chat = summary("a-1", "Chat");
        chat.session_id = "s-1".to_string();
        let scope = AgentsViewScopeKey { session_id: "s-1".into(), active_session_id: Some("a-1".into()) };
        let mut mode = build_mode(vec![roster_entry("a-1", "Chat")]).await;
        mode.enter_scope_for_test(scope, chat);
        assert!(mode.left_result_available());
    }

    #[tokio::test]
    async fn ctrl_c_needs_two_presses_to_exit() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.handle_input("\x03");
        assert!(mode.exit_hint_visible());
        assert!(!mode.is_stopped());
        mode.handle_input("\x03");
        assert!(mode.is_stopped());
    }

    #[tokio::test]
    async fn handle_input_records_the_new_session_intent() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.handle_input("\x0e");
        assert!(mode.new_session_requested());
    }

    #[tokio::test]
    async fn back_and_clear_use_configured_application_bindings() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        let previous = pi_tui::keybindings::get_keybindings();
        let mut configured = previous.clone();
        let mut bindings = configured.get_user_bindings();
        bindings.insert("app.agents.back".into(), vec!["ctrl+l".into()]);
        bindings.insert("app.input.clear".into(), vec!["ctrl+k".into()]);
        configured.set_user_bindings(bindings);
        pi_tui::keybindings::set_keybindings(configured);

        mode.set_search_query_for_test("Alpha");
        mode.handle_input("\x1b");
        assert_eq!(mode.editor_text(), "Alpha");
        mode.handle_input("\x0b");
        assert_eq!(mode.editor_text(), "");
        let chat = summary("a-1", "Alpha");
        mode.enter_scope_for_test(
            AgentsViewScopeKey { session_id: chat.session_id.clone(), active_session_id: chat.active_session_id.clone() },
            chat,
        );
        mode.handle_input("\x1b[D");
        assert!(!mode.is_stopped());
        mode.handle_input("\x0c");
        assert!(mode.is_stopped());
        pi_tui::keybindings::set_keybindings(previous);
    }

    #[tokio::test]
    async fn create_new_session_opens_the_created_summary() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.push_response(ok_response(serde_json::to_value(summary("c-1", "Fresh")).unwrap()));
        let created = mode.create_new_session().await;
        assert!(created);
        assert!(mode.is_stopped());
    }

    #[tokio::test]
    async fn refresh_saved_sessions_populates_the_inactive_section() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        let saved = serde_json::json!({
            "path": "C:/sessions/old.jsonl",
            "id": "old-1",
            "cwd": "C:/work",
            "created": "2026-01-01T00:00:00.000Z",
            "modified": "2026-01-02T00:00:00.000Z",
            "messageCount": 4,
            "firstMessage": "old chat",
            "allMessagesText": "old chat text",
        });
        mode.push_response(ok_response(serde_json::json!({ "sessions": [saved] })));
        assert!(mode.refresh_saved_sessions(false, true).await);
        let rows = mode.rows().to_vec();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|row| row.section == AgentsViewSection::Inactive));
    }

    #[tokio::test]
    async fn refresh_heartbeats_records_the_catalog() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.push_response(ok_response(serde_json::json!({
            "heartbeats": [{
                "job": {
                    "id": "job-1",
                    "status": "active",
                    "activeSessionId": "a-1",
                    "sessionId": "a-1",
                    "sessionFile": "C:/sessions/a.jsonl",
                    "prompt": "tick",
                    "schedule": { "kind": "interval", "expression": "1h" },
                    "createdAt": "2026-01-01T00:00:00.000Z",
                    "updatedAt": "2026-01-01T00:00:00.000Z",
                    "runCount": 0
                }
            }]
        })));
        assert!(mode.refresh_heartbeats(false).await);
        assert_eq!(mode.heartbeat_count(), 1);
    }

    #[tokio::test]
    async fn heartbeat_recovery_clears_only_its_own_status() {
        let mut mode = build_mode(Vec::new()).await;
        mode.set_status_message(Some("Failed to refresh heartbeats: disconnected"), false, None, false);
        mode.push_response(ok_response(serde_json::json!({"heartbeats":[]})));
        assert!(mode.refresh_heartbeats(false).await);
        assert!(mode.status_message().is_none());
        mode.set_status_message(Some("Failed to rename session: denied"), false, None, false);
        mode.push_response(ok_response(serde_json::json!({"heartbeats":[]})));
        assert!(mode.refresh_heartbeats(false).await);
        assert_eq!(mode.status_message(), Some("Failed to rename session: denied"));
    }

    #[tokio::test]
    async fn starting_heartbeat_warning_retains_last_known_catalog() {
        let mut mode = build_mode(Vec::new()).await;
        mode.push_response(ok_response(serde_json::json!({"heartbeats":[{"job":{
            "id":"retained-job", "status":"active", "activeSessionId":"a-1", "sessionId":"a-1",
            "sessionFile":"C:/isolated/a.jsonl", "prompt":"tick", "schedule":{"kind":"interval","expression":"1h"},
            "createdAt":"2026-01-01T00:00:00Z", "updatedAt":"2026-01-01T00:00:00Z", "runCount":0
        }}]})));
        assert!(mode.refresh_heartbeats(false).await);
        assert_eq!(mode.heartbeat_count(), 1);
        mode.push_response(DaemonResponse {command: "heartbeats_list".into(), success:false, data:None,
            error:Some("Cannot list heartbeats while session worker is starting".into())});
        assert!(!mode.refresh_heartbeats(false).await);
        assert_eq!(mode.heartbeat_count(), 1);
        assert_eq!(mode.status_message(), Some("Scheduled tasks are still loading; coverage is incomplete, retrying"));
        mode.push_response(ok_response(serde_json::json!({"heartbeats":[]})));
        assert!(mode.refresh_heartbeats(false).await);
        assert!(mode.status_message().is_none());
    }

    #[tokio::test]
    async fn heartbeat_startup_does_not_block_roster_reconnect() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.reconnect_started = true;
        // The same live client/hello keeps its roster subscription; only the
        // heartbeat request is sent on this reconnect completion path.
        mode.push_response(DaemonResponse {command:"heartbeats_list".into(), success:false, data:None,
            error:Some("Cannot list heartbeats while session worker is starting".into())});
        let client = mode.require_client().unwrap();
        mode.finish_reconnect_attempt(&client).await.unwrap();
        assert!(!mode.reconnect_started);
        assert!(mode.rows().iter().any(|row| row.title.contains("Alpha")));
        assert_eq!(mode.status_message(), Some("Daemon reconnected; scheduled-task coverage is incomplete, retrying"));
    }

    #[tokio::test]
    async fn pending_catalog_displays_progress_without_rebuilding_cached_rows() {
        struct ProgressDriver {
            frames: StdMutex<Vec<Vec<String>>>,
            first_paint: tokio_util::sync::CancellationToken,
            second_paint: tokio_util::sync::CancellationToken,
        }
        impl AgentsViewTerminal for ProgressDriver {
            fn rows(&self) -> usize { 50 }
            fn columns(&self) -> usize { 180 }
            fn request_render(&self, _force: bool) {}
            fn set_title(&self, _title: &str) {}
            fn poll_input(&self) -> Result<Option<Vec<String>>, String> { Ok(Some(Vec::new())) }
            fn present(&self, lines: Vec<String>, _dock: Vec<String>) -> Result<(), String> {
                if lines.iter().any(|line| line.contains("loading saved chats (8)")) { self.first_paint.cancel(); }
                if lines.iter().any(|line| line.contains("loading saved chats (16)")) { self.second_paint.cancel(); }
                self.frames.lock().unwrap().push(lines);
                Ok(())
            }
        }
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.saved_catalog_refresh_pending = true;
        let terminal = Arc::new(ProgressDriver {
            frames: StdMutex::new(Vec::new()),
            first_paint: tokio_util::sync::CancellationToken::new(),
            second_paint: tokio_util::sync::CancellationToken::new(),
        });
        mode.terminal = terminal.clone();
        let progress = Arc::new(std::sync::atomic::AtomicI64::new(0));
        let producer = progress.clone();
        let first_paint = terminal.first_paint.clone();
        let second_paint = terminal.second_paint.clone();
        let pending = async move {
            producer.store(8, std::sync::atomic::Ordering::Relaxed);
            first_paint.cancelled().await;
            producer.store(16, std::sync::atomic::Ordering::Relaxed);
            second_paint.cancelled().await;
        };
        assert!(tokio::time::timeout(std::time::Duration::from_secs(3),
            mode.wait_with_input_progress(pending, Some(progress))).await.unwrap().unwrap().is_some());
        assert_eq!(mode.saved_catalog_progress, 16);
        let frames = terminal.frames.lock().unwrap();
        assert!(frames.iter().flatten().any(|line| line.contains("loading saved chats (8)")));
        assert!(frames.iter().flatten().any(|line| line.contains("loading saved chats (16)")));
        assert!(mode.rows().iter().any(|row| row.title.contains("Alpha")));
    }

    #[tokio::test]
    async fn get_last_assistant_text_handles_null_and_string() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.push_response(ok_response(serde_json::json!({ "text": Value::Null })));
        assert_eq!(mode.get_last_assistant_text("a-1").await, Ok(None));
        mode.push_response(ok_response(serde_json::json!({ "text": "hello" })));
        assert_eq!(mode.get_last_assistant_text("a-1").await, Ok(Some("hello".to_string())));
        mode.push_response(ok_response(serde_json::json!({ "text": 5 })));
        assert_eq!(
            mode.get_last_assistant_text("a-1").await,
            Err("Daemon returned an invalid last assistant response".to_string())
        );
        mode.push_response(ok_response(serde_json::json!([])));
        assert_eq!(
            mode.get_last_assistant_text("a-1").await,
            Err("Daemon returned an invalid last assistant response".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_attach_model_fallback_prefers_the_summary() {
        let mut summary = summary("a-1", "Alpha");
        summary.model = Some(super::super::agents_view_state::ModelRef { provider: "faux".into(), id: "test".into() });
        assert_eq!(resolve_attach_model_fallback_message(&summary, Some("startup")), None);
        summary.model = None;
        assert_eq!(
            resolve_attach_model_fallback_message(&summary, Some("startup")),
            Some("startup".to_string())
        );
        summary.model_fallback_message = Some("from summary".to_string());
        assert_eq!(
            resolve_attach_model_fallback_message(&summary, Some("startup")),
            Some("from summary".to_string())
        );
    }

    #[tokio::test]
    async fn both_browser_open_paths_enable_extension_dialogs() {
        struct RecordingFactory(StdMutex<Vec<AttachOptions>>);
        impl DaemonAgentConnectionFactory for RecordingFactory {
            fn attach(&self, _client: DaemonTransportClient, _active: &str, options: AttachOptions)
                -> TransportFuture<Result<Arc<dyn DaemonAgentConnectionHandle>, String>> {
                self.0.lock().unwrap().push(options);
                Box::pin(async { Ok(Arc::new(FakeAgentConnection) as Arc<dyn DaemonAgentConnectionHandle>) })
            }
        }
        let mode = build_mode(Vec::new()).await;
        let factory = RecordingFactory(StdMutex::new(Vec::new()));
        let transport = FakeTransport::new(&["agent_roster"]);
        let live = summary("a-1", "Live");
        open_agents_view_session(&mode.options, &live, transport.clone(), &factory).await.unwrap();
        let mut saved = live.clone();
        saved.active_session_id = None;
        saved.cwd = std::env::current_dir().unwrap().to_string_lossy().into_owned();
        saved.session_file = Some("C:/isolated/saved.jsonl".into());
        transport.push(Ok(ok_response(serde_json::to_value(&live).unwrap())));
        open_agents_view_session(&mode.options, &saved, transport, &factory).await.unwrap();
        let options = factory.0.lock().unwrap();
        assert_eq!(options.len(), 2);
        assert!(options.iter().all(|options| options.supports_extension_ui == Some(true)));
    }

    #[tokio::test]
    async fn cancelled_open_closes_only_its_pending_client() {
        struct PendingFactory;
        impl DaemonAgentConnectionFactory for PendingFactory {
            fn attach(&self, _client: DaemonTransportClient, _active: &str, _options: AttachOptions)
                -> TransportFuture<Result<Arc<dyn DaemonAgentConnectionHandle>, String>> {
                Box::pin(std::future::pending())
            }
        }
        let mode = build_mode(Vec::new()).await;
        let transport = FakeTransport::new(&["agent_roster"]);
        let terminal = InputDriver {
            input: StdMutex::new(std::collections::VecDeque::from([Vec::new(), vec!["\x1b".into()]])),
            frames: StdMutex::new(Vec::new()),
        };
        let chat = summary("a-1", "Live");
        let opened = tokio::time::timeout(std::time::Duration::from_secs(2), wait_for_session_open(
            &terminal, &chat, open_agents_view_session(&mode.options, &chat, transport.clone(), &PendingFactory),
        )).await.unwrap().unwrap();
        assert!(opened.is_none());
        assert!(!transport.is_connected());
        assert!(transport.requests().is_empty(), "cancel must not send abort, stop or shutdown");
        assert!(terminal.frames.lock().unwrap().iter().flatten().any(|line| line.contains("Opening Live")));
    }

    #[tokio::test]
    async fn pending_catalog_keeps_cached_rows_keyboard_navigation_and_exit_live() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha"), roster_entry("b-1", "Beta")]).await;
        let terminal = Arc::new(InputDriver {
            input: StdMutex::new(std::collections::VecDeque::from([
                vec!["\x1b[B".into()], vec!["\x03".into(), "\x03".into()],
            ])),
            frames: StdMutex::new(Vec::new()),
        });
        mode.terminal = terminal.clone();
        let (done, receive) = tokio::sync::oneshot::channel();
        mode.resolve_run = Some(done);
        let result = tokio::time::timeout(std::time::Duration::from_secs(2),
            mode.wait_with_input(std::future::pending::<()>())).await.unwrap().unwrap();
        assert!(result.is_none());
        assert_eq!(receive.await.unwrap(), AgentsViewRunResult::Exit);
        assert_eq!(mode.selected_index(), 1);
        assert!(terminal.frames.lock().unwrap().iter().flatten().any(|line| line.contains("Beta")));
    }

    #[tokio::test]
    async fn unavailable_daemon_reconnect_does_not_trap_the_browser() {
        let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
        mode.terminal = Arc::new(InputDriver {
            input: StdMutex::new(std::collections::VecDeque::from([vec!["\x03".into(), "\x03".into()]])),
            frames: StdMutex::new(Vec::new()),
        });
        mode.recover_daemon = Some(Arc::new(|| Box::pin(std::future::pending())));
        let (done, receive) = tokio::sync::oneshot::channel();
        mode.resolve_run = Some(done);
        mode.start_client_reconnect("offline");
        tokio::time::timeout(std::time::Duration::from_secs(2), mode.reconnect_client("offline")).await.unwrap().unwrap();
        assert_eq!(receive.await.unwrap(), AgentsViewRunResult::Exit);
        assert!(mode.stopped);
    }

    #[tokio::test]
    async fn resume_saved_session_sends_the_session_path() {
        let transport = FakeTransport::new(&["agent_roster"]);
        transport.push(Ok(ok_response(serde_json::to_value(summary("a-9", "Resumed")).unwrap())));
        let client = DaemonTransportClient::new(transport.clone());
        let mut chat = summary("a-9", "Resumed");
        chat.active_session_id = None;
        chat.cwd = std::env::current_dir().unwrap().to_string_lossy().into_owned();
        chat.session_file = Some("C:/sessions/a.jsonl".to_string());
        let config = AgentsViewRuntimeConfig {
            cwd: Some("C:/work".to_string()),
            ..AgentsViewRuntimeConfig::default()
        };
        let (resumed, active, notice) = resume_saved_agents_view_session(&client, &config, &chat).await.unwrap();
        assert_eq!(resumed.session_id, "a-9");
        assert_eq!(active, "a-9");
        assert_eq!(notice, None);
        let requests = transport.requests();
        assert_eq!(requests[0]["type"], "create");
        assert_eq!(requests[0]["sessionPath"], "C:/sessions/a.jsonl");
        assert!(requests[0]["config"].get("cwd").is_none());
    }

    #[tokio::test]
    async fn resume_without_a_session_file_is_rejected() {
        let transport = FakeTransport::new(&["agent_roster"]);
        let client = DaemonTransportClient::new(transport);
        let mut chat = summary("a-9", "Resumed");
        chat.session_file = None;
        let error = resume_saved_agents_view_session(&client, &AgentsViewRuntimeConfig::default(), &chat)
            .await
            .unwrap_err();
        assert_eq!(error, "Cannot resume a session without a saved session file");
    }

    #[tokio::test]
    async fn stale_roster_daemon_raises_the_exact_message() {
        let options = AgentsViewModeOptions {
            socket_path: Some("pipe".to_string()),
            config: AgentsViewRuntimeConfig::default(),
            ui_services: Arc::new(TestUiServices),
            migrated_providers: None,
            model_fallback_message: None,
            startup_model_id: None,
            verbose: None,
            reconnect_timeout_ms: None,
            initial_session: None,
            initial_scope_key: None,
        };
        let transport = FakeTransport::new(&[]);
        let mut interactive = interactive_factory();
        let mut mode = AgentsViewMode::new(
            options,
            AgentsViewPersistentState::default(),
            Arc::new(TestTerminal { rows: 30, renders: StdMutex::new(0), titles: StdMutex::new(Vec::new()) }),
            Box::new(TestEditor::default()),
            Arc::new(PlainAgentsViewTheme),
            transport,
            &FAKE_CONNECTION_FACTORY,
            &mut interactive,
            None,
        );
        let error = mode.run().await.unwrap_err();
        assert_eq!(error, STALE_ROSTER_DAEMON_MESSAGE);
    }

    #[tokio::test]
    async fn startup_titles_the_terminal_with_the_app_identity() {
        let options = AgentsViewModeOptions {
            socket_path: Some("pipe".to_string()),
            config: AgentsViewRuntimeConfig::default(),
            ui_services: Arc::new(TestUiServices),
            migrated_providers: None,
            model_fallback_message: None,
            startup_model_id: None,
            verbose: None,
            reconnect_timeout_ms: None,
            initial_session: None,
            initial_scope_key: None,
        };
        let terminal = Arc::new(TestTerminal { rows: 30, renders: StdMutex::new(0), titles: StdMutex::new(Vec::new()) });
        let transport = FakeTransport::new(&[]);
        let mut interactive = interactive_factory();
        let _mode = AgentsViewMode::new(
            options,
            AgentsViewPersistentState::default(),
            terminal.clone(),
            Box::new(TestEditor::default()),
            Arc::new(PlainAgentsViewTheme),
            transport,
            &FAKE_CONNECTION_FACTORY,
            &mut interactive,
            None,
        );
        let titles = terminal.titles.lock().unwrap().clone();
        assert_eq!(titles, vec![crate::config::app_display_title()]);
        assert!(
            titles[0].contains("Optimus"),
            "agents view title must be Optimus-branded, got {titles:?}"
        );
        assert_ne!(titles[0], "π - Agents");
    }

    #[test]
    fn daemon_shutdown_marks_the_view_and_clears_rows() {
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(async {
            let mut mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
            mode.handle_daemon_shutdown("socket closed");
            assert!(mode.daemon_shutdown_received());
            assert!(mode.rows().is_empty());
            assert!(mode
                .status_message()
                .unwrap()
                .starts_with("Prime Agent daemon shut down. Restart Prime Agent to reconnect."));
            assert!(mode.status_message_is_sticky());
        });
    }

    #[test]
    fn log_client_error_rotates_and_appends() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("client-errors.log");
        let log_path = log_path.to_string_lossy().to_string();
        append_rotating_log(&log_path, "first");
        append_rotating_log(&log_path, "second");
        let contents = std::fs::read_to_string(&log_path).unwrap();
        assert_eq!(contents, "first\nsecond\n");
    }

    #[test]
    fn splash_metadata_reports_scope_and_depth() {
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(async {
            let mode = build_mode(vec![roster_entry("a-1", "Alpha")]).await;
            let metadata = mode.splash_metadata();
            assert!(metadata[0].contains("0 running, 1 idle, 0 inactive"));
            assert!(metadata[1].contains("global"));
            assert!(metadata[2].contains('0'));
        });
    }

    #[test]
    fn outbound_variants_are_classified() {
        assert!(matches!(
            DaemonOutbound::HeartbeatsChanged,
            DaemonOutbound::HeartbeatsChanged
        ));
        assert!(matches!(DaemonOutbound::Other, DaemonOutbound::Other));
        let update = DaemonOutbound::RosterUpdate {
            changed: vec![roster_entry("a-1", "Alpha")],
            removed: Some(vec!["b-1".to_string()]),
            resync: Some(true),
        };
        let parsed = crate::modes::agents_view::roster_store::RosterUpdate::from_outbound(&update).unwrap();
        assert_eq!(parsed.changed.len(), 1);
        assert_eq!(parsed.removed, Some(vec!["b-1".to_string()]));
        assert_eq!(parsed.resync, Some(true));
        assert!(crate::modes::agents_view::roster_store::RosterUpdate::from_outbound(&DaemonOutbound::Other).is_none());
    }

    #[test]
    fn subagent_rows_carry_the_root_linkage() {
        let mut child = summary("c-1", "Child");
        child.runtime_kind = Some(RuntimeKind::Subagent);
        child.rlm_child_id = Some("child-1".to_string());
        child.parent_active_session_id = Some("a-1".to_string());
        assert!(crate::modes::agents_view::agents_view_state::is_subagent_summary(&child));
        let rows = build_agents_view_rows(
            &[
                AgentsViewRowInput::Summary(summary("a-1", "Alpha")),
                AgentsViewRowInput::Summary(child),
            ],
            &HashSet::new(),
            &HashSet::new(),
            None,
            None,
            None,
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].kind, AgentsViewRowKind::SubagentSummary);
    }

    #[test]
    fn count_rows_ignores_nested_rows() {
        let rows = vec![
            AgentsViewRow {
                kind: AgentsViewRowKind::Agent,
                section: AgentsViewSection::Running,
                ..AgentsViewRow::default()
            },
            AgentsViewRow {
                kind: AgentsViewRowKind::SubagentSummary,
                section: AgentsViewSection::Running,
                ..AgentsViewRow::default()
            },
            AgentsViewRow {
                kind: AgentsViewRowKind::SubagentCode,
                section: AgentsViewSection::Running,
                ..AgentsViewRow::default()
            },
        ];
        let counts = count_rows_by_section(&rows);
        assert_eq!(counts[&AgentsViewSection::Running], 1);
    }
}
