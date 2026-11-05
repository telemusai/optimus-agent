//! Port of packages/coding-agent/src/modes/agents-view/agents-view-state.ts
//!
//! Data model notes (PORT-RULES "absent vs null"):
//! - TypeScript optional fields (`?:`) become `Option<T>`: absent == `None`, and a
//!   JSON `null` also deserializes to `None` because the reference code never
//!   distinguishes the two for these fields (`??` / `=== true` tests only).
//! - `Map<UnifiedSessionRecord, _>` / `Set<UnifiedSessionRecord>` key on the record
//!   value in TypeScript. Rust has no reference identity for owned values, so the
//!   record's stable `identity` string is used as the map key instead. Two records
//!   with the same identity cannot exist: `reconcileUnifiedSessions` folds aliases
//!   onto one record.

use std::collections::{HashMap, HashSet};
use chrono::{DateTime, SecondsFormat, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Node's `Date#toISOString()` (millisecond precision, `Z` suffix).
pub fn to_iso_string(value: &DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// Approximation of `Date.parse`: RFC 3339 first, then a few common shapes that
/// the reference code passes in (JSONL headers, `statSync().mtime`).
pub fn parse_js_timestamp(value: &str) -> Option<i64> {
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return Some(parsed.timestamp_millis());
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M", "%Y-%m-%d"] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(value, format) {
            return Some(Utc.from_utc_datetime(&naive).timestamp_millis());
        }
        if let Ok(date) = chrono::NaiveDate::parse_from_str(value, format) {
            return Some(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0)?).timestamp_millis());
        }
    }
    None
}

/// Milliseconds since the Unix epoch, `Date.now()`.
pub fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentsViewSection {
    Running,
    Idle,
    Inactive,
}

impl Default for AgentsViewSection {
    /// `classifyUnifiedSession` returns `"inactive"` when the record has no daemon
    /// session, so that is the section a defaulted record gets.
    fn default() -> Self {
        AgentsViewSection::Inactive
    }
}

/// Durable lifecycle; decides agents-view visibility. Only `Live` is shown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionLifecycle {
    Draft,
    #[default]
    Live,
    Archived,
}

/// Heuristic activity of a live session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionActivity {
    Working,
    #[default]
    Idle,
}

/// One status formula shared by every agent surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentRosterStatus {
    Running,
    Idle,
    Inactive,
}
impl Default for AgentRosterStatus {
    /// `classifyAgentStatus` returns `"inactive"` for a non-resident agent, so a
    /// defaulted roster entry is inactive.
    fn default() -> Self {
        AgentRosterStatus::Inactive
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeKind {
    TopLevel,
    Subagent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentTaskState {
    NeedsInput,
    Completed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentRosterStatusLabel {
    Queued,
    Recovering,
    Failed,
    #[serde(rename = "background helper")]
    BackgroundHelper,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    Starting,
    Ready,
    Recovering,
    Stopping,
    Failed,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionUsageSummary {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef {
    pub provider: String,
    pub id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionActionActive {
    pub kind: String,
    pub phase: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionActionSnapshot {
    pub queued_count: i64,
    #[serde(default)]
    pub steering: Vec<String>,
    #[serde(default)]
    pub follow_ups: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<SessionActionActive>,
}

/// Lightweight daemon session shape (SessionSummary in daemon-session-list.ts).
///
/// `#[serde(default)]` mirrors the reference's structural typing: `isSessionSummary`
/// only checks `id` and `sessionId`, so any extra/absent field must not fail.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SessionSummary {
    pub id: String,
    pub lifecycle: SessionLifecycle,
    pub activity: SessionActivity,
    pub is_session_active: bool,
    pub has_active_heartbeat: Option<bool>,
    pub has_registered_heartbeat: Option<bool>,
    pub has_registered_cron_job: Option<bool>,
    pub last_activity_at: Option<String>,
    pub runtime_kind: Option<RuntimeKind>,
    /// RLM spawn depth (0 for roots); fork edges preserve the source depth.
    pub rlm_depth: Option<i64>,
    pub active_session_id: Option<String>,
    pub session_id: String,
    pub session_file: Option<String>,
    pub session_name: Option<String>,
    pub cwd: String,
    pub model: Option<ModelRef>,
    /// ThinkingLevel union; kept as the wire string until pi-agent-core lands one.
    pub thinking_level: Option<String>,
    pub is_streaming: bool,
    pub is_compacting: bool,
    pub is_bash_running: Option<bool>,
    pub has_running_rlm_children: Option<bool>,
    pub usage: Option<SessionUsageSummary>,
    pub is_running_tools: Option<bool>,
    pub attached_clients: i64,
    pub direct_attached_clients: Option<i64>,
    pub message_count: i64,
    pub unfinished_action_count: Option<i64>,
    pub session_actions: SessionActionSnapshot,
    pub streaming_message: Option<Value>,
    pub created: Option<String>,
    pub modified: Option<String>,
    pub first_message: Option<String>,
    pub parent_active_session_id: Option<String>,
    pub parent_session_id: Option<String>,
    pub parent_session_path: Option<String>,
    pub rlm_child_id: Option<String>,
    pub replied_since_task: Option<bool>,
    pub rlm_parent_node_id: Option<String>,
    /// Source of the Python cell that spawned this subagent, for display.
    pub spawn_code: Option<String>,
    pub model_fallback_message: Option<String>,
    pub diagnostics: Option<Vec<Value>>,
    pub summary: Option<String>,
    pub task_state: Option<AgentTaskState>,
    pub roster_status: Option<AgentRosterStatus>,
    pub status_label: Option<AgentRosterStatusLabel>,
    /// Set while the owning worker has been silent past the staleness threshold.
    pub last_heard_from_at: Option<String>,
    pub worker_state: Option<WorkerState>,
    pub worker_pid: Option<i64>,
    /// Optional schema-39 metadata. Unknown activity versions must not break the roster.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub communication_activity: Option<Value>,
    /// Preserve streaming-only throughput; legacy request-inclusive metadata is ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming_throughput: Option<Value>,
    /// Optional schema-45 advisory queue metadata; unknown versions must not break the roster.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub communication_queue: Option<Value>,
}

impl SessionSummary {
    /// Minimal constructor for the fields the reference always sets.
    pub fn new(id: impl Into<String>, session_id: impl Into<String>, cwd: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            id: id.clone(),
            session_id: session_id.into(),
            cwd: cwd.into(),
            lifecycle: SessionLifecycle::Live,
            activity: SessionActivity::Idle,
            is_session_active: false,
            is_streaming: false,
            is_compacting: false,
            attached_clients: 0,
            message_count: 0,
            session_actions: SessionActionSnapshot::default(),
            ..Self::default()
        }
    }

    pub fn with_session_file(mut self, path: impl Into<String>) -> Self {
        self.session_file = Some(path.into());
        self
    }

    pub fn with_active_session_id(mut self, active: impl Into<String>) -> Self {
        self.active_session_id = Some(active.into());
        self
    }
}


#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct UnifiedSessionHeartbeat {
    pub active_count: i64,
    pub paused_count: Option<i64>,
    pub next_run_at: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentConnectionSavedSessionState {
    pub status: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentConnectionAgentStatus {
    pub summary: String,
    pub task_state: Option<AgentTaskState>,
    pub based_on_message_count: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentConnectionSavedSessionInfo {
    pub path: String,
    pub id: String,
    pub cwd: String,
    pub name: Option<String>,
    pub state: Option<AgentConnectionSavedSessionState>,
    pub parent_session_path: Option<String>,
    pub rlm_depth: Option<i64>,
    pub created: DateTime<Utc>,
    pub modified: DateTime<Utc>,
    pub message_count: i64,
    pub first_message: String,
    pub all_messages_text: String,
    pub agent_status: Option<AgentConnectionAgentStatus>,
    pub usage: Option<SessionUsageSummary>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentCronSchedule {
    pub kind: String,
    pub expression: String,
    pub interval_ms: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentCronJob {
    pub id: String,
    pub status: String,
    pub source: Option<String>,
    pub runtime_kind: Option<String>,
    pub delivery_mode: Option<String>,
    pub active_session_id: String,
    pub session_id: String,
    pub session_file: String,
    pub cwd: String,
    pub label: Option<String>,
    pub prompt: String,
    pub schedule: AgentCronSchedule,
    pub created_at: String,
    pub updated_at: String,
    pub next_run_at: Option<String>,
    pub last_run_at: Option<String>,
    pub last_skipped_at: Option<String>,
    pub last_error: Option<String>,
    pub run_count: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentConnectionHeartbeat {
    pub job: AgentCronJob,
    pub session_name: Option<String>,
    pub first_message: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct UnifiedSessionRecord {
    pub daemon: Option<SessionSummary>,
    pub saved: Option<AgentConnectionSavedSessionInfo>,
    /// Stable UI key, chosen using canonical path, session id, then active id.
    pub identity: String,
    /// Alternate keys used to restore selection while a session is persisted or reattached.
    pub identity_aliases: Vec<String>,
    pub section: AgentsViewSection,
    pub searchable_text: String,
    pub heartbeat: Option<UnifiedSessionHeartbeat>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentsViewScopeKey {
    pub session_id: String,
    pub active_session_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentsViewScopeFrame {
    pub scope: AgentsViewScopeKey,
    /// Chat to revisit before returning to the parent agents view.
    pub return_chat: Option<SessionSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AgentsViewScopeAction {
    Push {
        scope: AgentsViewScopeKey,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        return_chat: Option<SessionSummary>,
    },
    Back,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentsViewScopeResolution {
    pub frames: Vec<AgentsViewScopeFrame>,
    pub root: Option<UnifiedSessionRecord>,
    pub dropped_frames: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentsViewScopeBackResult {
    #[serde(rename = "type")]
    pub kind: String,
    pub selection: SessionSummary,
    pub expanded_ancestor_session_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_chat: Option<SessionSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnattachableChildOpenResult {
    #[serde(rename = "type")]
    pub kind: String,
    pub summary: SessionSummary,
    pub selection: SessionSummary,
    pub expanded_ancestor_session_ids: Vec<String>,
    pub has_children: bool,
    pub status_message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AgentsViewRowKind {
    Agent,
    SubagentSummary,
    Subagent,
    SubagentCode,
}

/// Hard cap on spawn-code lines shown so a large program never floods the view.
pub const MAX_SPAWN_CODE_LINES: usize = 10;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentsViewRow {
    pub kind: AgentsViewRowKind,
    pub section: AgentsViewSection,
    pub summary: SessionSummary,
    pub title: String,
    pub subtitle: String,
    pub status_label: String,
    pub depth: usize,
    pub selectable: bool,
    pub running_subagent_count: i64,
    pub recursive_cost: f64,
    /// Total descendant sessions (resident + passive) under this row.
    pub descendant_count: i64,
    /// Unique selection identity for this row.
    pub identity: String,
    /// Identity of the agent row this row is nested under.
    pub parent_identity: Option<String>,
    /// True when this row's subagents carry spawn code that can be revealed.
    pub has_spawn_code: Option<bool>,
    /// True when this subagent-summary row's list is expanded.
    pub expanded: Option<bool>,
    /// One source line of the spawn cell, for "subagent-code" rows.
    pub code: Option<String>,
    /// Merged durable/live source data for unified rows.
    pub record: Option<UnifiedSessionRecord>,
    pub heartbeat: Option<UnifiedSessionHeartbeat>,
}

impl Default for AgentsViewRow {
    fn default() -> Self {
        Self {
            kind: AgentsViewRowKind::Agent,
            section: AgentsViewSection::Idle,
            summary: SessionSummary::default(),
            title: String::new(),
            subtitle: String::new(),
            status_label: String::new(),
            depth: 0,
            selectable: true,
            running_subagent_count: 0,
            recursive_cost: 0.0,
            descendant_count: 0,
            identity: String::new(),
            parent_identity: None,
            has_spawn_code: None,
            expanded: None,
            code: None,
            record: None,
            heartbeat: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentsViewRecursiveRollup {
    /// Own cost plus every descendant's cost.
    pub cost: f64,
    /// Total descendant sessions (resident + passive) under this record.
    pub descendant_count: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentsViewSelectionKey {
    pub session_id: String,
    pub active_session_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentsViewSelectionResolution {
    pub index: usize,
    pub resolved: bool,
}


/// Key used for record-keyed maps and sets: TypeScript keys on the record value,
/// Rust on its stable `identity` (see module docs).
pub type RecordKey = String;

pub type UnifiedSessionRollups = HashMap<RecordKey, AgentsViewRecursiveRollup>;

#[derive(Clone, Debug, Default)]
pub struct UnifiedSessionIndex {
    pub by_key: HashMap<String, RecordKey>,
    pub children_by_parent: HashMap<RecordKey, Vec<RecordKey>>,
    /// Records in catalog order, keyed by identity.
    pub records_by_key: HashMap<RecordKey, UnifiedSessionRecord>,
}

pub fn classify_agent_status(resident: bool, queued_child: bool, busy: bool) -> AgentRosterStatus {
    if queued_child {
        return AgentRosterStatus::Running;
    }
    if !resident {
        return AgentRosterStatus::Inactive;
    }
    if busy {
        AgentRosterStatus::Running
    } else {
        AgentRosterStatus::Idle
    }
}

pub fn classify_session_roster_status(summary: &SessionSummary, queued_child: bool) -> AgentRosterStatus {
    classify_agent_status(
        summary.active_session_id.is_some(),
        queued_child,
        summary.status_label != Some(AgentRosterStatusLabel::BackgroundHelper)
            && (summary.activity == SessionActivity::Working || summary.is_session_active),
    )
}

pub fn classify_agents_view_session(summary: &SessionSummary) -> AgentsViewSection {
    match summary.roster_status {
        Some(status) => match status {
            AgentRosterStatus::Running => AgentsViewSection::Running,
            AgentRosterStatus::Idle => AgentsViewSection::Idle,
            AgentRosterStatus::Inactive => AgentsViewSection::Inactive,
        },
        None => match classify_session_roster_status(summary, false) {
            AgentRosterStatus::Running => AgentsViewSection::Running,
            AgentRosterStatus::Idle => AgentsViewSection::Idle,
            AgentRosterStatus::Inactive => AgentsViewSection::Inactive,
        },
    }
}

pub fn classify_unified_session(record: &UnifiedSessionRecord) -> AgentsViewSection {
    match &record.daemon {
        None => AgentsViewSection::Inactive,
        Some(summary) => classify_agents_view_session(summary),
    }
}

pub fn should_show_agents_view_session(summary: &SessionSummary, manually_inactive: bool) -> bool {
    if manually_inactive {
        return false;
    }
    summary.lifecycle == SessionLifecycle::Live
}

// TODO(unify: #2055): replace with the shared user-content rule once it lands;
// session summaries only carry message counts today.
pub fn is_empty_agents_view_session(summary: &SessionSummary) -> bool {
    summary.message_count == 0
}

pub fn section_title(section: AgentsViewSection) -> &'static str {
    match section {
        AgentsViewSection::Running => "Running",
        AgentsViewSection::Idle => "Idle",
        AgentsViewSection::Inactive => "Inactive",
    }
}

fn format_age_label(timestamp: &str) -> String {
    let parsed = parse_js_timestamp(timestamp).unwrap_or(0);
    let seconds = (((now_ms() - parsed) as f64) / 1000.0).round().max(0.0) as i64;
    if seconds < 120 {
        return format!("{seconds}s ago");
    }
    let minutes = ((seconds as f64) / 60.0).round() as i64;
    if minutes < 120 {
        format!("{minutes}m ago")
    } else {
        format!("{}h ago", ((minutes as f64) / 60.0).round() as i64)
    }
}

/// `resolve(canonicalizePath(path))`: realpath where possible, lexical resolve after.
/// `canonicalize_path` (utils/paths.ts) already falls back to the raw path when
/// realpath fails, so this is `resolve(realpath(path) ?? path)` - the reference's
/// fallback chain. Private plumbing: `path.resolve` is a Node builtin, not a
/// ported function of this slice.
fn canonical_session_path(path: &str) -> String {
    resolve_path(&crate::utils::paths::canonicalize_path(path))
}

/// Node's `path.resolve` (lexical, no filesystem access). Private helper.
fn resolve_path(value: &str) -> String {
    let path = std::path::Path::new(value);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join(path)
    };
    let mut parts: Vec<String> = Vec::new();
    let mut prefix = String::new();
    for component in absolute.components() {
        match component {
            std::path::Component::Prefix(prefix_component) => {
                prefix.push_str(&prefix_component.as_os_str().to_string_lossy());
            }
            std::path::Component::RootDir => prefix.push(std::path::MAIN_SEPARATOR),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                parts.pop();
            }
            std::path::Component::Normal(part) => parts.push(part.to_string_lossy().to_string()),
        }
    }
    let joined = parts.join(&std::path::MAIN_SEPARATOR.to_string());
    if prefix.is_empty() {
        joined
    } else {
        format!("{prefix}{joined}")
    }
}

/// Node's `path.basename` for the cwd fallback title. Private helper.
fn basename(value: &str) -> String {
    let trimmed = value.trim_end_matches(['/', '\\']);
    match trimmed.rsplit(['/', '\\']).next() {
        Some(part) if !part.is_empty() => part.to_string(),
        _ => {
            if value == "/" || value == "\\" {
                value.to_string()
            } else {
                String::new()
            }
        }
    }
}

fn file_identity(path: &str) -> String {
    format!("file:{}", canonical_session_path(path))
}

fn summary_identity_aliases(summary: &SessionSummary) -> Vec<String> {
    let roster = if summary.runtime_kind == Some(RuntimeKind::Subagent) && summary.rlm_child_id.is_some() {
        Some(format!("agent:{}", roster_agent_id_for_summary(summary)))
    } else {
        None
    };
    [
        roster,
        summary.session_file.as_deref().map(file_identity),
        Some(format!("session:{}", summary.session_id)),
        summary.active_session_id.as_ref().map(|id| format!("active:{id}")),
        Some(format!("active:{}", summary.id)),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn saved_identity_aliases(saved: &AgentConnectionSavedSessionInfo) -> Vec<String> {
    vec![file_identity(&saved.path), format!("session:{}", saved.id)]
}

fn create_unified_searchable_text(
    daemon: Option<&SessionSummary>,
    saved: Option<&AgentConnectionSavedSessionInfo>,
) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if let Some(d) = daemon {
        parts.extend(
            [
                d.session_id.as_str(),
                d.active_session_id.as_deref().unwrap_or(""),
                d.session_name.as_deref().unwrap_or(""),
                d.first_message.as_deref().unwrap_or(""),
                d.cwd.as_str(),
                d.session_file.as_deref().unwrap_or(""),
                d.summary.as_deref().unwrap_or(""),
            ]
            .into_iter(),
        );
    }
    if let Some(s) = saved {
        parts.extend(
            [
                s.id.as_str(),
                s.name.as_deref().unwrap_or(""),
                s.first_message.as_str(),
                s.all_messages_text.as_str(),
                s.agent_status.as_ref().map(|status| status.summary.as_str()).unwrap_or(""),
                s.cwd.as_str(),
                s.path.as_str(),
                s.parent_session_path.as_deref().unwrap_or(""),
            ]
            .into_iter(),
        );
    }
    parts
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Child ids are only unique per parent (32-bit, mkdir-checked); the parent path
/// qualifies them daemon-wide.
pub fn roster_agent_id_for_summary(summary: &SessionSummary) -> String {
    if summary.runtime_kind == Some(RuntimeKind::Subagent) {
        if let Some(child_id) = summary.rlm_child_id.as_deref() {
            // No-session parents have no path (and no ledger edge); their live
            // parent id still disambiguates.
            let parent_key = summary
                .parent_session_path
                .as_deref()
                .map(canonical_session_path)
                .or_else(|| summary.parent_active_session_id.clone());
            return match parent_key {
                Some(key) => format!("{key}#{child_id}"),
                None => child_id.to_string(),
            };
        }
    }
    summary.session_id.clone()
}

/// Convert a merged row to the existing live-row rendering/action shape.
pub fn summary_for_unified_record(record: &UnifiedSessionRecord) -> SessionSummary {
    if let Some(daemon) = &record.daemon {
        let Some(saved) = &record.saved else {
            return daemon.clone();
        };
        let mut merged = daemon.clone();
        if merged.session_name.is_none() {
            merged.session_name = saved.name.clone();
        }
        if merged.first_message.is_none() {
            merged.first_message = Some(saved.first_message.clone());
        }
        if merged.usage.is_none() {
            merged.usage = saved.usage.clone();
        }
        if merged.session_file.is_none() {
            merged.session_file = Some(canonical_session_path(&saved.path));
        }
        if merged.parent_session_path.is_none() {
            merged.parent_session_path = saved.parent_session_path.clone();
        }
        if merged.rlm_depth.is_none() {
            merged.rlm_depth = saved.rlm_depth;
        }
        if merged.created.is_none() {
            merged.created = Some(to_iso_string(&saved.created));
        }
        if merged.modified.is_none() {
            merged.modified = Some(to_iso_string(&saved.modified));
        }
        if merged.last_activity_at.is_none() {
            merged.last_activity_at = Some(to_iso_string(&saved.modified));
        }
        return merged;
    }
    let saved = record
        .saved
        .as_ref()
        .expect("Unified session record has no daemon or saved source");
    let depth = saved.rlm_depth.unwrap_or(if saved.parent_session_path.is_some() { 1 } else { 0 });
    SessionSummary {
        id: saved.id.clone(),
        lifecycle: SessionLifecycle::Archived,
        activity: SessionActivity::Idle,
        is_session_active: false,
        runtime_kind: Some(if depth > 0 { RuntimeKind::Subagent } else { RuntimeKind::TopLevel }),
        rlm_depth: saved.rlm_depth,
        session_id: saved.id.clone(),
        session_file: Some(canonical_session_path(&saved.path)),
        parent_session_path: saved.parent_session_path.clone(),
        session_name: saved.name.clone(),
        cwd: saved.cwd.clone(),
        is_streaming: false,
        is_compacting: false,
        attached_clients: 0,
        message_count: saved.message_count,
        session_actions: SessionActionSnapshot::default(),
        created: Some(to_iso_string(&saved.created)),
        modified: Some(to_iso_string(&saved.modified)),
        last_activity_at: Some(to_iso_string(&saved.modified)),
        first_message: Some(saved.first_message.clone()),
        summary: saved.agent_status.as_ref().map(|status| status.summary.clone()),
        task_state: saved.agent_status.as_ref().and_then(|status| status.task_state),
        usage: saved.usage.clone(),
        ..SessionSummary::default()
    }
}

/// Reconcile daemon-resident and saved catalog rows without inventing runtime
/// ancestry from persisted fork metadata. Daemon data remains authoritative;
/// saved data only enriches durable/search fields.
pub fn reconcile_unified_sessions(
    daemon_summaries: &[SessionSummary],
    saved_sessions: &[AgentConnectionSavedSessionInfo],
    heartbeats: &[AgentConnectionHeartbeat],
) -> Vec<UnifiedSessionRecord> {
    let heartbeat_by_active_id = aggregate_session_heartbeats(daemon_summaries, heartbeats);
    let mut records: Vec<UnifiedSessionRecord> = Vec::new();
    let mut record_by_alias: HashMap<String, RecordKey> = HashMap::new();
    let mut by_identity: HashMap<RecordKey, UnifiedSessionRecord> = HashMap::new();

    for daemon in daemon_summaries {
        let aliases = summary_identity_aliases(daemon);
        let lookup_key = daemon.active_session_id.clone().unwrap_or_else(|| daemon.id.clone());
        let heartbeat = heartbeat_by_active_id.get(&lookup_key).cloned().or_else(|| {
            if daemon.has_active_heartbeat == Some(true) {
                Some(UnifiedSessionHeartbeat { active_count: 1, paused_count: None, next_run_at: None })
            } else {
                None
            }
        });
        let daemon_value = match &heartbeat {
            Some(hb) if hb.active_count > 0 && daemon.has_active_heartbeat != Some(true) => {
                let mut cloned = daemon.clone();
                cloned.has_active_heartbeat = Some(true);
                cloned
            }
            _ => daemon.clone(),
        };
        let mut record = UnifiedSessionRecord {
            daemon: Some(daemon_value),
            saved: None,
            identity: aliases.first().cloned().unwrap_or_default(),
            identity_aliases: aliases.clone(),
            section: AgentsViewSection::Idle,
            searchable_text: String::new(),
            heartbeat,
        };
        record.section = classify_unified_session(&record);
        record.searchable_text = create_unified_searchable_text(Some(daemon), None);
        for alias in &aliases {
            record_by_alias.insert(alias.clone(), record.identity.clone());
        }
        by_identity.insert(record.identity.clone(), record.clone());
        records.push(record);
    }

    for saved in saved_sessions {
        let aliases = saved_identity_aliases(saved);
        let existing = aliases.iter().find_map(|alias| record_by_alias.get(alias)).cloned();
        if let Some(key) = existing {
            if let Some(record) = by_identity.get_mut(&key) {
                record.saved = Some(saved.clone());
                for alias in &aliases {
                    if !record.identity_aliases.contains(alias) {
                        record.identity_aliases.push(alias.clone());
                    }
                }
                let daemon = record.daemon.clone();
                record.searchable_text = create_unified_searchable_text(daemon.as_ref(), Some(saved));
            }
            for alias in &aliases {
                record_by_alias.insert(alias.clone(), key.clone());
            }
            continue;
        }
        let inactive = UnifiedSessionRecord {
            daemon: None,
            saved: Some(saved.clone()),
            identity: aliases.first().cloned().unwrap_or_default(),
            identity_aliases: aliases.clone(),
            section: AgentsViewSection::Inactive,
            searchable_text: create_unified_searchable_text(None, Some(saved)),
            heartbeat: None,
        };
        for alias in &aliases {
            record_by_alias.insert(alias.clone(), inactive.identity.clone());
        }
        by_identity.insert(inactive.identity.clone(), inactive.clone());
        records.push(inactive);
    }

    // Mirror the reference's mutation order: records are patched in place, so
    // return the updated values from the identity map.
    records
        .into_iter()
        .map(|record| by_identity.remove(&record.identity).unwrap_or(record))
        .collect()
}

pub fn transition_agents_view_scope(
    frames: &[AgentsViewScopeFrame],
    action: &AgentsViewScopeAction,
) -> Vec<AgentsViewScopeFrame> {
    match action {
        AgentsViewScopeAction::Back => {
            let mut next = frames.to_vec();
            next.pop();
            next
        }
        AgentsViewScopeAction::Push { scope, return_chat } => {
            let next_frame = AgentsViewScopeFrame {
                scope: scope.clone(),
                return_chat: return_chat.clone(),
            };
            let same = frames.last().map(|frame| frame.scope.session_id == scope.session_id).unwrap_or(false);
            if !same {
                let mut next = frames.to_vec();
                next.push(next_frame);
                return next;
            }
            let mut next = frames.to_vec();
            next.pop();
            next.push(next_frame);
            next
        }
    }
}

pub fn resolve_agents_view_left_result(
    scope_root: Option<&SessionSummary>,
    expanded_ancestor_session_ids: Vec<String>,
    return_chat: Option<&SessionSummary>,
) -> Option<AgentsViewScopeBackResult> {
    let scope_root = scope_root?;
    let return_chat_value = match return_chat {
        Some(chat) if chat.session_id == scope_root.session_id => Some(scope_root.clone()),
        _ => None,
    };
    Some(AgentsViewScopeBackResult {
        kind: "scope_back".to_string(),
        selection: scope_root.clone(),
        expanded_ancestor_session_ids,
        return_chat: return_chat_value,
    })
}

pub fn should_apply_scope_resolution(dropped_frames: usize, saved_catalog_ready: bool) -> bool {
    dropped_frames == 0 || saved_catalog_ready
}

pub fn create_unattachable_child_open_result(
    child: &SessionSummary,
    parent: &SessionSummary,
    expanded_ancestor_session_ids: &[String],
    has_children: bool,
) -> UnattachableChildOpenResult {
    UnattachableChildOpenResult {
        kind: "open".to_string(),
        summary: parent.clone(),
        selection: child.clone(),
        expanded_ancestor_session_ids: expanded_ancestor_session_ids.to_vec(),
        has_children,
        status_message: "Child session is unavailable; opened its parent instead".to_string(),
    }
}

pub fn resolve_agents_view_scope_frames(
    records: &[UnifiedSessionRecord],
    frames: &[AgentsViewScopeFrame],
    index: Option<&UnifiedSessionIndex>,
) -> AgentsViewScopeResolution {
    let owned_index;
    let index = match index {
        Some(index) => index,
        None => {
            owned_index = build_unified_session_index(records);
            &owned_index
        }
    };
    if frames.is_empty() {
        return AgentsViewScopeResolution { frames: Vec::new(), root: None, dropped_frames: 0 };
    }
    for frame_index in (0..frames.len()).rev() {
        let frame = &frames[frame_index];
        if let Some(root) = find_scope_record(&frame.scope, index) {
            let kept = frames[..frame_index + 1].to_vec();
            let root_value = index.records_by_key.get(&root).cloned();
            if root_value.is_none() {
                continue;
            }
            return AgentsViewScopeResolution {
                frames: kept,
                root: root_value,
                dropped_frames: frames.len() - frame_index - 1,
            };
        }
    }
    AgentsViewScopeResolution { frames: Vec::new(), root: None, dropped_frames: frames.len() }
}

/// Restrict records to the scoped root and every descendant of that root.
pub fn scope_to_session_subtree(
    records: &[UnifiedSessionRecord],
    scope: Option<&AgentsViewScopeKey>,
    index: Option<&UnifiedSessionIndex>,
) -> Vec<UnifiedSessionRecord> {
    let owned_index;
    let index = match index {
        Some(index) => index,
        None => {
            owned_index = build_unified_session_index(records);
            &owned_index
        }
    };
    let Some(scope) = scope else {
        return records.to_vec();
    };
    let Some(root) = find_scope_record(scope, index) else {
        return Vec::new();
    };
    let retained = collect_subtree_keys(&root, index);
    records
        .iter()
        .filter(|record| retained.contains(&record.identity))
        .cloned()
        .collect()
}

pub fn has_unified_session_children(
    records: &[UnifiedSessionRecord],
    scope: &AgentsViewScopeKey,
    index: Option<&UnifiedSessionIndex>,
) -> bool {
    let owned_index;
    let index = match index {
        Some(index) => index,
        None => {
            owned_index = build_unified_session_index(records);
            &owned_index
        }
    };
    match find_scope_record(scope, index) {
        Some(root) => index.children_by_parent.get(&root).map(|c| c.len()).unwrap_or(0) > 0,
        None => false,
    }
}

pub fn get_unified_session_ancestor_session_ids(
    records: &[UnifiedSessionRecord],
    scope: &AgentsViewScopeKey,
    index: Option<&UnifiedSessionIndex>,
) -> Vec<String> {
    let owned_index;
    let index = match index {
        Some(index) => index,
        None => {
            owned_index = build_unified_session_index(records);
            &owned_index
        }
    };
    let Some(root) = find_scope_record(scope, index) else {
        return Vec::new();
    };
    let mut ancestors: Vec<String> = Vec::new();
    let mut visited: HashSet<RecordKey> = HashSet::new();
    visited.insert(root.clone());
    let mut current = find_parent_record(&root, index);
    while let Some(record_key) = current {
        if visited.contains(&record_key) {
            break;
        }
        visited.insert(record_key.clone());
        if let Some(record) = index.records_by_key.get(&record_key) {
            ancestors.insert(0, summary_for_unified_record(record).session_id);
        }
        current = find_parent_record(&record_key, index);
    }
    ancestors
}

pub fn filter_unified_sessions(
    records: &[UnifiedSessionRecord],
    matches: &dyn Fn(&str) -> bool,
) -> Vec<UnifiedSessionRecord> {
    let index = build_unified_session_index(records);
    let mut retained: HashSet<RecordKey> = HashSet::new();
    for record in records {
        if !matches(&record.searchable_text) {
            continue;
        }
        let mut current: Option<RecordKey> = Some(record.identity.clone());
        while let Some(key) = current {
            if retained.contains(&key) {
                break;
            }
            retained.insert(key.clone());
            current = find_parent_record(&key, &index);
        }
    }
    // Keep catalog order and the original records so row ranking and sections
    // remain authoritative while ancestors provide the hierarchy for matches.
    records
        .iter()
        .filter(|record| retained.contains(&record.identity))
        .cloned()
        .collect()
}


// Rolls costs and descendant counts over the UNFILTERED hierarchy: filters must
// never change a row's totals.
pub fn compute_recursive_rollups(
    records: &[UnifiedSessionRecord],
    index: Option<&UnifiedSessionIndex>,
) -> UnifiedSessionRollups {
    let owned_index;
    let index = match index {
        Some(index) => index,
        None => {
            owned_index = build_unified_session_index(records);
            &owned_index
        }
    };
    let mut order: Vec<RecordKey> = records
        .iter()
        .filter(|record| {
            let parent = find_parent_record(&record.identity, index);
            match parent {
                None => true,
                Some(key) => key == record.identity,
            }
        })
        .map(|record| record.identity.clone())
        .collect();
    let mut position = 0usize;
    while position < order.len() {
        if let Some(children) = index.children_by_parent.get(&order[position]) {
            order.extend(children.iter().cloned());
        }
        position += 1;
    }
    let mut rollups: UnifiedSessionRollups = HashMap::new();
    for position in (0..order.len()).rev() {
        let key = order[position].clone();
        let Some(record) = index.records_by_key.get(&key) else {
            continue;
        };
        let mut cost = record
            .daemon
            .as_ref()
            .and_then(|d| d.usage.as_ref().map(|u| u.cost))
            .or_else(|| record.saved.as_ref().and_then(|s| s.usage.as_ref().map(|u| u.cost)))
            .unwrap_or(0.0);
        let mut descendant_count = 0i64;
        if let Some(children) = index.children_by_parent.get(&key) {
            for child_key in children.clone() {
                let Some(child) = index.records_by_key.get(&child_key) else {
                    continue;
                };
                if !is_subagent_descendant_record(child, record) {
                    continue;
                }
                let child_rollup = rollups.get(&child_key);
                cost += child_rollup.map(|r| r.cost).unwrap_or(0.0);
                descendant_count += 1 + child_rollup.map(|r| r.descendant_count).unwrap_or(0);
            }
        }
        rollups.insert(key, AgentsViewRecursiveRollup { cost, descendant_count });
    }
    rollups
}

/// Rollups follow agent lineage only. A branched/forked session links to its
/// source through parentSession but keeps the source's rlmDepth: it is a sibling
/// chat, not a descendant, and its copied transcript would double-book the
/// source's totals. Spawned subagents carry runtimeKind (resident) or a deeper
/// rlmDepth (saved) and do roll up.
fn is_subagent_descendant_record(child: &UnifiedSessionRecord, parent: &UnifiedSessionRecord) -> bool {
    if child.daemon.is_some() {
        return is_subagent_summary(child.daemon.as_ref().unwrap());
    }
    let child_depth = child.saved.as_ref().and_then(|s| s.rlm_depth).unwrap_or(0);
    let parent_depth = parent
        .daemon
        .as_ref()
        .and_then(|d| d.rlm_depth)
        .or_else(|| parent.saved.as_ref().and_then(|s| s.rlm_depth))
        .unwrap_or(0);
    child_depth > parent_depth
}

pub fn build_unified_session_index(records: &[UnifiedSessionRecord]) -> UnifiedSessionIndex {
    let mut by_key: HashMap<String, RecordKey> = HashMap::new();
    let mut records_by_key: HashMap<RecordKey, UnifiedSessionRecord> = HashMap::new();
    for record in records {
        records_by_key.insert(record.identity.clone(), record.clone());
        for key in &record.identity_aliases {
            by_key.insert(key.clone(), record.identity.clone());
        }
    }
    let index = UnifiedSessionIndex {
        by_key,
        children_by_parent: HashMap::new(),
        records_by_key,
    };
    let mut children_by_parent: HashMap<RecordKey, Vec<RecordKey>> = HashMap::new();
    for record in records {
        let Some(parent) = find_parent_record(&record.identity, &index) else {
            continue;
        };
        if parent == record.identity {
            continue;
        }
        children_by_parent.entry(parent).or_default().push(record.identity.clone());
    }
    UnifiedSessionIndex { children_by_parent, ..index }
}

/// Row identities flip when a session gains a sessionFile (active→persisted) or
/// is re-attached; the old identity survives as an alias. Rewrite stale entries
/// in a persisted identity set to the current record identity. Entries with no
/// alias match are kept: their record may not have streamed in yet.
pub fn migrate_agents_view_identity_set(
    identities: &mut HashSet<String>,
    by_key: &HashMap<String, RecordKey>,
) {
    for identity in identities.clone() {
        let Some(record_key) = by_key.get(&identity) else {
            continue;
        };
        if record_key == &identity {
            continue;
        }
        identities.remove(&identity);
        identities.insert(record_key.clone());
    }
}

/// Node's `path.resolve` for an absolute-or-relative path. Private helper.
fn resolve_absolute(value: &str) -> String {
    resolve_path(value)
}

fn find_scope_record(
    scope: &AgentsViewScopeKey,
    index: &UnifiedSessionIndex,
) -> Option<RecordKey> {
    if let Some(active_session_id) = &scope.active_session_id {
        if let Some(active) = index.by_key.get(&format!("active:{active_session_id}")) {
            return Some(active.clone());
        }
    }
    index.by_key.get(&format!("session:{}", scope.session_id)).cloned()
}

fn find_parent_record(record_key: &str, index: &UnifiedSessionIndex) -> Option<RecordKey> {
    let record = index.records_by_key.get(record_key)?;
    let mut keys = match &record.daemon {
        Some(daemon) => get_parent_keys(daemon),
        None => Vec::new(),
    };
    if let Some(saved) = &record.saved {
        if let Some(path) = &saved.parent_session_path {
            keys.push(file_identity(path));
        }
    }
    for key in keys {
        if let Some(parent) = index.by_key.get(&key) {
            return Some(parent.clone());
        }
    }
    None
}

fn collect_subtree_keys(root: &str, index: &UnifiedSessionIndex) -> HashSet<RecordKey> {
    let mut retained: HashSet<RecordKey> = HashSet::new();
    let mut queue: Vec<RecordKey> = vec![root.to_string()];
    let mut queue_index = 0usize;
    while queue_index < queue.len() {
        let current = queue[queue_index].clone();
        queue_index += 1;
        if retained.contains(&current) {
            continue;
        }
        retained.insert(current.clone());
        if let Some(children) = index.children_by_parent.get(&current) {
            queue.extend(children.iter().cloned());
        }
    }
    retained
}

pub fn aggregate_session_heartbeats(
    summaries: &[SessionSummary],
    heartbeats: &[AgentConnectionHeartbeat],
) -> HashMap<String, UnifiedSessionHeartbeat> {
    let mut summary_by_key: HashMap<String, SessionSummary> = HashMap::new();
    for summary in summaries {
        for key in get_summary_keys(summary) {
            summary_by_key.insert(key, summary.clone());
        }
    }
    let mut active_job_ids_by_owner: HashMap<String, Vec<String>> = HashMap::new();
    let mut paused_job_ids_by_owner: HashMap<String, Vec<String>> = HashMap::new();
    let mut next_run_by_job: HashMap<String, String> = HashMap::new();
    for heartbeat in heartbeats {
        let job = &heartbeat.job;
        if job.status != "active" && job.status != "paused" {
            continue;
        }
        if job.status == "active" {
            if let Some(next_run_at) = &job.next_run_at {
                if parse_js_timestamp(next_run_at).is_some() {
                    next_run_by_job.insert(job.id.clone(), next_run_at.clone());
                }
            }
        }
        // Passivation stales the job's active id; session id and file still find the owning row.
        let mut summary: Option<SessionSummary> = None;
        for key in [
            format!("active:{}", job.active_session_id),
            format!("session:{}", job.session_id),
            file_identity(&job.session_file),
        ] {
            if let Some(found) = summary_by_key.get(&key) {
                summary = Some(found.clone());
                break;
            }
        }
        let mut visited: HashSet<String> = HashSet::new();
        if summary.is_none() {
            push_unique(
                if job.status == "active" { &mut active_job_ids_by_owner } else { &mut paused_job_ids_by_owner },
                &job.active_session_id,
                &job.id,
            );
        }
        while let Some(current) = summary.clone() {
            let owner = current
                .active_session_id
                .clone()
                .unwrap_or_else(|| current.id.clone());
            if visited.contains(&owner) {
                break;
            }
            visited.insert(owner.clone());
            push_unique(
                if job.status == "active" { &mut active_job_ids_by_owner } else { &mut paused_job_ids_by_owner },
                &owner,
                &job.id,
            );
            summary = find_parent_summary(&current, &summary_by_key);
        }
    }
    let mut owners: Vec<String> = active_job_ids_by_owner.keys().cloned().collect();
    for owner in paused_job_ids_by_owner.keys() {
        if !owners.contains(owner) {
            owners.push(owner.clone());
        }
    }
    let mut result: HashMap<String, UnifiedSessionHeartbeat> = HashMap::new();
    for owner in owners {
        let job_ids = active_job_ids_by_owner.get(&owner).cloned().unwrap_or_default();
        let paused_count = paused_job_ids_by_owner.get(&owner).map(|ids| ids.len()).unwrap_or(0) as i64;
        let mut next_runs: Vec<String> = job_ids
            .iter()
            .filter_map(|job_id| next_run_by_job.get(job_id).cloned())
            .collect();
        next_runs.sort_by_key(|value| parse_js_timestamp(value).unwrap_or(0));
        let next_run_at = next_runs.first().cloned();
        result.insert(
            owner,
            UnifiedSessionHeartbeat {
                active_count: job_ids.len() as i64,
                paused_count: if paused_count > 0 { Some(paused_count) } else { None },
                next_run_at,
            },
        );
    }
    result
}

fn push_unique(map: &mut HashMap<String, Vec<String>>, owner: &str, job_id: &str) {
    let ids = map.entry(owner.to_string()).or_default();
    if !ids.iter().any(|existing| existing == job_id) {
        ids.push(job_id.to_string());
    }
}

pub fn format_heartbeat_badge(heartbeat: Option<&UnifiedSessionHeartbeat>, now: i64) -> String {
    let Some(heartbeat) = heartbeat else {
        return String::new();
    };
    if heartbeat.active_count < 1 {
        let paused = heartbeat.paused_count.unwrap_or(0);
        return if paused > 0 { format!("♥ {paused}") } else { String::new() };
    }
    let next = heartbeat.next_run_at.as_deref().and_then(parse_js_timestamp);
    let countdown = next.map(|value| format_heartbeat_countdown(value - now));
    format!(
        "♥ {}{}",
        heartbeat.active_count,
        countdown.map(|c| format!("·{c}")).unwrap_or_default()
    )
}

fn format_heartbeat_countdown(duration_ms: i64) -> String {
    let seconds = ((duration_ms.max(0) as f64) / 1000.0).round().max(1.0) as i64;
    if seconds < 60 {
        return format!("{seconds}s");
    }
    let minutes = ((seconds as f64) / 60.0).round() as i64;
    if minutes < 60 {
        return format!("{minutes}m");
    }
    let hours = ((minutes as f64) / 60.0).round() as i64;
    if hours < 24 {
        return format!("{hours}h");
    }
    format!("{}d", ((hours as f64) / 24.0).round() as i64)
}

fn find_parent_summary(
    summary: &SessionSummary,
    by_key: &HashMap<String, SessionSummary>,
) -> Option<SessionSummary> {
    for key in get_parent_keys(summary) {
        if let Some(parent) = by_key.get(&key) {
            return Some(parent.clone());
        }
    }
    None
}

pub fn get_parent_keys(summary: &SessionSummary) -> Vec<String> {
    [
        summary.parent_active_session_id.as_ref().map(|id| format!("active:{id}")),
        summary.parent_session_id.as_ref().map(|id| format!("session:{id}")),
        summary.parent_session_path.as_deref().map(file_identity),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Direct-child linkage over getParentKeys, shared by the view tree and the chat subagents bar.
pub fn is_direct_agent_child(
    child: &SessionSummary,
    parent_active_session_id: Option<&str>,
    parent_session_id: Option<&str>,
    parent_session_file: Option<&str>,
) -> bool {
    let parent_keys: HashSet<String> = get_parent_keys(child).into_iter().collect();
    if let Some(active) = parent_active_session_id {
        if parent_keys.contains(&format!("active:{active}")) {
            return true;
        }
    }
    if let Some(session) = parent_session_id {
        if parent_keys.contains(&format!("session:{session}")) {
            return true;
        }
    }
    match parent_session_file {
        Some(file) => parent_keys.contains(&file_identity(file)),
        None => false,
    }
}

pub fn get_agents_view_summary_identity(summary: &SessionSummary) -> String {
    if summary.runtime_kind == Some(RuntimeKind::Subagent) && summary.rlm_child_id.is_some() {
        return format!("agent:{}", roster_agent_id_for_summary(summary));
    }
    if let Some(session_file) = &summary.session_file {
        return file_identity(session_file);
    }
    if let Some(active) = &summary.active_session_id {
        return format!("active:{active}");
    }
    format!("session:{}", summary.session_id)
}

pub fn get_agents_view_selection_key(summary: &SessionSummary) -> AgentsViewSelectionKey {
    AgentsViewSelectionKey {
        session_id: summary.session_id.clone(),
        active_session_id: summary.active_session_id.clone(),
    }
}


// Matches by identity, then activeSessionId, then sessionId: a row's identity
// changes when a session is persisted or re-attached, so the latter two keys
// re-find the same session across those transitions. Returns -1 when gone.
pub fn resolve_agents_view_selection_index(
    rows: &[AgentsViewRow],
    identity: Option<&str>,
    key: Option<&AgentsViewSelectionKey>,
) -> i64 {
    let find_selectable = |predicate: &dyn Fn(&AgentsViewRow) -> bool| -> i64 {
        rows.iter()
            .position(|row| row.selectable && predicate(row))
            .map(|index| index as i64)
            .unwrap_or(-1)
    };

    if let Some(identity) = identity {
        let index = find_selectable(&|row| row.identity == identity);
        // Synthetic nested rows deliberately reuse their parent's session key, so
        // their exact row identity must win over the active-runtime fallback.
        if index >= 0 && rows[index as usize].kind != AgentsViewRowKind::Agent {
            return index;
        }
    }
    if let Some(active_session_id) = key.and_then(|k| k.active_session_id.as_deref()) {
        let index = find_selectable(&|row| {
            row.summary.active_session_id.as_deref().unwrap_or(row.summary.id.as_str()) == active_session_id
        });
        if index >= 0 {
            return index;
        }
    }
    if let Some(identity) = identity {
        let index = find_selectable(&|row| row.identity == identity);
        if index >= 0 {
            return index;
        }
    }
    if let Some(session_id) = key.map(|k| k.session_id.as_str()) {
        return find_selectable(&|row| row.summary.session_id == session_id);
    }
    -1
}

pub fn resolve_agents_view_selection_state(
    rows: &[AgentsViewRow],
    current_index: usize,
    identity: Option<&str>,
    key: Option<&AgentsViewSelectionKey>,
) -> AgentsViewSelectionResolution {
    if rows.is_empty() {
        return AgentsViewSelectionResolution { index: 0, resolved: false };
    }
    let resolved_index = resolve_agents_view_selection_index(rows, identity, key);
    if resolved_index >= 0 {
        return AgentsViewSelectionResolution { index: resolved_index as usize, resolved: true };
    }
    let bounded_index = current_index.min(rows.len() - 1);
    if rows[bounded_index].selectable {
        return AgentsViewSelectionResolution { index: bounded_index, resolved: false };
    }
    let first_selectable = rows.iter().position(|row| row.selectable);
    AgentsViewSelectionResolution {
        index: first_selectable.unwrap_or(0),
        resolved: false,
    }
}

/// `SessionSummary | UnifiedSessionRecord` input union for buildAgentsViewRows.
#[derive(Clone, Debug)]
pub enum AgentsViewRowInput {
    Summary(SessionSummary),
    Record(UnifiedSessionRecord),
}

fn is_unified_session_record(value: &AgentsViewRowInput) -> bool {
    matches!(value, AgentsViewRowInput::Record(_))
}

pub fn build_agents_view_rows(
    summaries_or_records: &[AgentsViewRowInput],
    expanded_subagent_parents: &HashSet<String>,
    program_shown_parents: &HashSet<String>,
    scope: Option<&AgentsViewScopeKey>,
    recursive_rollups: Option<&UnifiedSessionRollups>,
    anchor_session_id: Option<&str>,
) -> Vec<AgentsViewRow> {
    let inputs: Vec<(SessionSummary, Option<UnifiedSessionRecord>)> = summaries_or_records
        .iter()
        .map(|input| match input {
            AgentsViewRowInput::Record(record) => (summary_for_unified_record(record), Some(record.clone())),
            AgentsViewRowInput::Summary(summary) => (summary.clone(), None),
        })
        .collect();
    let scope_root_index = scope.and_then(|scope| {
        inputs.iter().position(|(summary, _)| {
            summary.session_id == scope.session_id
                || (scope.active_session_id.is_some() && summary.active_session_id == scope.active_session_id)
        })
    });
    let scope_root_keys: HashSet<String> = match scope_root_index {
        None => HashSet::new(),
        Some(index) => match &inputs[index].1 {
            Some(record) => record.identity_aliases.iter().cloned().collect(),
            None => get_summary_keys(&inputs[index].0).into_iter().collect(),
        },
    };
    let is_direct_scope_child = |summary: &SessionSummary| -> bool {
        scope_root_index.is_some() && get_parent_keys(summary).iter().any(|key| scope_root_keys.contains(key))
    };
    let mut rows: Vec<AgentsViewRow> = inputs
        .iter()
        .map(|(summary, record)| {
            let mut row = AgentsViewRow {
                kind: if is_subagent_summary(summary) && !is_direct_scope_child(summary) {
                    AgentsViewRowKind::Subagent
                } else {
                    AgentsViewRowKind::Agent
                },
                section: record.as_ref().map(classify_unified_session).unwrap_or_else(|| {
                    classify_agents_view_session(summary)
                }),
                summary: summary.clone(),
                title: get_agents_view_session_title(summary),
                subtitle: get_session_subtitle(summary),
                status_label: get_session_status_label(summary, record.as_ref().and_then(|r| r.heartbeat.as_ref())),
                depth: 0,
                selectable: true,
                running_subagent_count: 0,
                recursive_cost: summary.usage.as_ref().map(|u| u.cost).unwrap_or(0.0),
                descendant_count: 0,
                identity: record
                    .as_ref()
                    .map(|r| r.identity.clone())
                    .unwrap_or_else(|| get_agents_view_summary_identity(summary)),
                ..AgentsViewRow::default()
            };
            if let Some(record) = record {
                row.heartbeat = record.heartbeat.clone();
                row.record = Some(record.clone());
            }
            row
        })
        .collect();
    let rows_by_key = build_row_key_map(&rows);
    let mut children_by_parent: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut nested_rows: HashSet<usize> = HashSet::new();

    for index in 0..rows.len() {
        if rows[index].kind != AgentsViewRowKind::Subagent {
            continue;
        }
        let parent = find_parent_row(&rows[index].summary, &rows_by_key);
        let Some(parent) = parent else {
            // Saved catalogs stream progressively, so a child can arrive before its
            // parent. Keep it reachable as a root until the parent record appears.
            rows[index].kind = AgentsViewRowKind::Agent;
            continue;
        };
        if parent == index {
            rows[index].kind = AgentsViewRowKind::Agent;
            continue;
        }
        // One definition of "child" with the rollup walk: a branched/forked
        // session links to its source but is a top-level chat in its own right,
        // so it must not nest (nor count in the expander) while #sub excludes it.
        let record = rows[index].record.clone();
        let parent_record = rows[parent].record.clone();
        if let (Some(child_record), Some(parent_record)) = (record, parent_record) {
            if !is_subagent_descendant_record(&child_record, &parent_record) {
                rows[index].kind = AgentsViewRowKind::Agent;
                continue;
            }
        }
        nested_rows.insert(index);
        children_by_parent.entry(parent).or_default().push(index);
    }

    // Busy-descendant tally from the live rows: iterative over the parent forest so
    // deep chains cannot overflow.
    let mut tally_order: Vec<usize> = (0..rows.len()).filter(|index| !nested_rows.contains(index)).collect();
    let mut position = 0usize;
    while position < tally_order.len() {
        if let Some(children) = children_by_parent.get(&tally_order[position]) {
            tally_order.extend(children.iter().cloned());
        }
        position += 1;
    }
    for position in (0..tally_order.len()).rev() {
        let index = tally_order[position];
        let mut count = 0i64;
        let mut descendants_cost = 0.0f64;
        let mut descendants = 0i64;
        for child in children_by_parent.get(&index).cloned().unwrap_or_default() {
            count += if rows[child].section == AgentsViewSection::Running { 1 } else { 0 }
                + rows[child].running_subagent_count;
            descendants_cost += rows[child].recursive_cost;
            descendants += 1 + rows[child].descendant_count;
        }
        rows[index].running_subagent_count = count;
        let rollup = rows[index]
            .record
            .as_ref()
            .and_then(|record| recursive_rollups.and_then(|map| map.get(&record.identity)));
        rows[index].recursive_cost =
            rollup.map(|r| r.cost).unwrap_or_else(|| rows[index].summary.usage.as_ref().map(|u| u.cost).unwrap_or(0.0) + descendants_cost);
        rows[index].descendant_count = rollup.map(|r| r.descendant_count).unwrap_or(descendants);
    }

    let roots: Vec<usize> = (0..rows.len()).filter(|index| !nested_rows.contains(index)).collect();
    let mut visible_roots: Vec<usize> = match scope_root_index {
        Some(scope_root_index) => roots.iter().copied().filter(|index| *index != scope_root_index).collect(),
        None => roots
            .iter()
            .copied()
            .filter(|index| rows[*index].section != AgentsViewSection::Inactive || !is_subagent_summary(&rows[*index].summary))
            .collect(),
    };
    visible_roots.sort_by(|a, b| compare_agents_view_rows(&rows[*a], &rows[*b], anchor_session_id));
    let mut flattened: Vec<AgentsViewRow> = Vec::new();
    for root in visible_roots {
        emit_rows(
            root,
            0,
            &mut rows,
            &children_by_parent,
            expanded_subagent_parents,
            program_shown_parents,
            anchor_session_id,
            &mut flattened,
        );
    }
    flattened
}

#[allow(clippy::too_many_arguments)]
fn emit_rows(
    row_index: usize,
    depth: usize,
    rows: &mut Vec<AgentsViewRow>,
    children_by_parent: &HashMap<usize, Vec<usize>>,
    expanded_subagent_parents: &HashSet<String>,
    program_shown_parents: &HashSet<String>,
    anchor_session_id: Option<&str>,
    flattened: &mut Vec<AgentsViewRow>,
) {
    rows[row_index].depth = depth;
    flattened.push(rows[row_index].clone());
    let children = children_by_parent.get(&row_index).cloned().unwrap_or_default();
    if children.is_empty() {
        return;
    }
    let child_has_spawn_code = children.iter().any(|index| has_spawn_code(&rows[*index].summary));
    let expanded = expanded_subagent_parents.contains(&rows[row_index].identity);
    let child_rows: Vec<AgentsViewRow> = children.iter().map(|index| rows[*index].clone()).collect();
    let summary_row =
        create_subagent_summary_row(&rows[row_index], &child_rows, depth + 1, child_has_spawn_code, expanded);
    flattened.push(summary_row);
    if !expanded {
        return;
    }
    let show_program = program_shown_parents.contains(&rows[row_index].identity);
    let mut sorted_children = children.clone();
    sorted_children.sort_by(|a, b| compare_agents_view_rows(&rows[*a], &rows[*b], anchor_session_id));
    let groups = group_children_by_spawn_code(&sorted_children, rows);
    for (group_index, group) in groups.iter().enumerate() {
        if show_program {
            if let Some(spawn_code) = &group.spawn_code {
                for code_row in build_spawn_code_rows(&rows[row_index], spawn_code, depth + 1, group_index) {
                    flattened.push(code_row);
                }
            }
        }
        for child in &group.children {
            let parent_identity = rows[row_index].identity.clone();
            rows[*child].parent_identity = Some(parent_identity);
            emit_rows(
                *child,
                depth + 1,
                rows,
                children_by_parent,
                expanded_subagent_parents,
                program_shown_parents,
                anchor_session_id,
                flattened,
            );
        }
    }
}

fn create_subagent_summary_row(
    parent: &AgentsViewRow,
    children: &[AgentsViewRow],
    depth: usize,
    has_spawn_code: bool,
    expanded: bool,
) -> AgentsViewRow {
    let total_count = children.len();
    let running = parent.running_subagent_count;
    let heartbeat_count = children
        .iter()
        .filter(|child| {
            child.summary.has_active_heartbeat == Some(true)
                || child.heartbeat.as_ref().map(|h| h.active_count).unwrap_or(0) > 0
        })
        .count();
    // Finished subagents stay reachable through the summary row even when
    // nothing is running anymore.
    let subagent_title = if running > 0 {
        format!("{running} {} running", if running == 1 { "subagent" } else { "subagents" })
    } else {
        format!("{total_count} {}", if total_count == 1 { "subagent" } else { "subagents" })
    };
    let title = if heartbeat_count > 0 {
        format!(
            "{subagent_title} · {heartbeat_count} {} active",
            if heartbeat_count == 1 { "heartbeat" } else { "heartbeats" }
        )
    } else {
        subagent_title
    };
    AgentsViewRow {
        kind: AgentsViewRowKind::SubagentSummary,
        section: parent.section,
        summary: parent.summary.clone(),
        title,
        subtitle: String::new(),
        status_label: String::new(),
        depth,
        selectable: true,
        running_subagent_count: running,
        recursive_cost: 0.0,
        descendant_count: 0,
        identity: format!("subagents:{}", parent.identity),
        parent_identity: Some(parent.identity.clone()),
        has_spawn_code: Some(has_spawn_code),
        expanded: Some(expanded),
        code: None,
        record: None,
        heartbeat: None,
    }
}

fn has_spawn_code(summary: &SessionSummary) -> bool {
    summary.spawn_code.as_deref().map(|code| !code.trim().is_empty()).unwrap_or(false)
}

#[derive(Clone, Debug)]
struct SpawnCodeGroup {
    /// Shared spawn-cell source for this group, or None when unavailable.
    spawn_code: Option<String>,
    children: Vec<usize>,
}

// Subagents spawned by the same Python cell share its source; group them so
// each spawn cell renders once, above the subagents it launched. Different turns
// produce different cells and therefore distinct groups. Insertion order follows
// each cell's first subagent so groups read top-to-bottom in spawn order.
fn group_children_by_spawn_code(children: &[usize], rows: &[AgentsViewRow]) -> Vec<SpawnCodeGroup> {
    const NO_CODE_KEY: &str = " no-spawn-code";
    let mut groups: Vec<SpawnCodeGroup> = Vec::new();
    let mut index_by_key: HashMap<String, usize> = HashMap::new();
    for child in children {
        let code = if has_spawn_code(&rows[*child].summary) {
            rows[*child].summary.spawn_code.clone()
        } else {
            None
        };
        let key = code.clone().unwrap_or_else(|| NO_CODE_KEY.to_string());
        match index_by_key.get(&key) {
            Some(index) => groups[*index].children.push(*child),
            None => {
                index_by_key.insert(key, groups.len());
                groups.push(SpawnCodeGroup { spawn_code: code, children: vec![*child] });
            }
        }
    }
    groups
}

fn build_spawn_code_rows(
    parent: &AgentsViewRow,
    spawn_code: &str,
    depth: usize,
    group_index: usize,
) -> Vec<AgentsViewRow> {
    let make_row = |code: &str, line_index: &str| -> AgentsViewRow {
        AgentsViewRow {
            kind: AgentsViewRowKind::SubagentCode,
            section: parent.section,
            summary: parent.summary.clone(),
            title: String::new(),
            subtitle: String::new(),
            status_label: String::new(),
            depth,
            // Code rows are read-only context; selection skips over them.
            selectable: false,
            running_subagent_count: 0,
            recursive_cost: 0.0,
            descendant_count: 0,
            identity: format!("code:{}:{group_index}:{line_index}", parent.identity),
            parent_identity: Some(parent.identity.clone()),
            has_spawn_code: None,
            expanded: None,
            code: Some(code.to_string()),
            record: None,
            heartbeat: None,
        }
    };
    let trimmed = spawn_code.trim_end();
    let all_lines: Vec<&str> = trimmed.split('\n').collect();
    // Cap the body so a long program can't flood the view; note the remainder.
    let mut lines: Vec<AgentsViewRow> = all_lines
        .iter()
        .take(MAX_SPAWN_CODE_LINES)
        .enumerate()
        .map(|(index, line)| make_row(line, &index.to_string()))
        .collect();
    let hidden = all_lines.len().saturating_sub(lines.len());
    if hidden > 0 {
        lines.push(make_row(
            &format!("… +{hidden} more {}", if hidden == 1 { "line" } else { "lines" }),
            "more",
        ));
    }
    // A blank panel line above and below pads the program into a clean block.
    let mut result = vec![make_row("", "pad-top")];
    result.extend(lines);
    result.push(make_row("", "pad-bottom"));
    result
}


fn compare_agents_view_rows(a: &AgentsViewRow, b: &AgentsViewRow, anchor_session_id: Option<&str>) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let section_diff = section_rank(a.section) - section_rank(b.section);
    if section_diff != 0 {
        return section_diff.cmp(&0);
    }
    let empty_diff = empty_session_rank(a, anchor_session_id) - empty_session_rank(b, anchor_session_id);
    if empty_diff != 0 {
        return empty_diff.cmp(&0);
    }
    if a.section == AgentsViewSection::Inactive {
        let a_flag = if a.summary.has_active_heartbeat.unwrap_or(false) { 1 } else { 0 };
        let b_flag = if b.summary.has_active_heartbeat.unwrap_or(false) { 1 } else { 0 };
        let heartbeat_diff = b_flag - a_flag;
        if heartbeat_diff != 0 {
            return heartbeat_diff.cmp(&0);
        }
    }
    if a.section != AgentsViewSection::Running {
        let a_busy = if a.running_subagent_count > 0 { 1 } else { 0 };
        let b_busy = if b.running_subagent_count > 0 { 1 } else { 0 };
        let busy_descendants_diff = b_busy - a_busy;
        if busy_descendants_diff != 0 {
            return busy_descendants_diff.cmp(&0);
        }
        let activity_diff = get_timestamp(b.summary.last_activity_at.as_deref())
            - get_timestamp(a.summary.last_activity_at.as_deref());
        if activity_diff != 0 {
            return activity_diff.cmp(&0);
        }
    }
    let created_diff = get_timestamp(b.summary.created.as_deref()) - get_timestamp(a.summary.created.as_deref());
    if created_diff != 0 {
        return created_diff.cmp(&0);
    }
    let title_diff = compare_locale(&a.title, &b.title);
    if title_diff != Ordering::Equal {
        return title_diff;
    }
    compare_locale(&a.summary.session_id, &b.summary.session_id)
}

/// `String#localeCompare` for the code-point order the reference relies on
/// (ASCII titles and ids; no locale-specific tailoring is observable here).
fn compare_locale(a: &str, b: &str) -> std::cmp::Ordering {
    a.cmp(b)
}

// Message-less sessions sink to the bottom of their section, except the session
// the view was entered from: it keeps its recency slot so opening the agents
// view from a fresh chat doesn't catapult that chat to the bottom.
fn empty_session_rank(row: &AgentsViewRow, anchor_session_id: Option<&str>) -> i32 {
    if !is_empty_agents_view_session(&row.summary) || Some(row.summary.session_id.as_str()) == anchor_session_id {
        return 0;
    }
    1
}

fn build_row_key_map(rows: &[AgentsViewRow]) -> HashMap<String, usize> {
    let mut rows_by_key: HashMap<String, usize> = HashMap::new();
    for (index, row) in rows.iter().enumerate() {
        for key in get_summary_keys(&row.summary) {
            rows_by_key.insert(key, index);
        }
    }
    rows_by_key
}

pub fn get_summary_keys(summary: &SessionSummary) -> Vec<String> {
    [
        Some(format!(
            "active:{}",
            summary.active_session_id.as_deref().unwrap_or(summary.id.as_str())
        )),
        Some(format!("session:{}", summary.session_id)),
        summary.session_file.as_deref().map(file_identity),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn find_parent_row(summary: &SessionSummary, rows_by_key: &HashMap<String, usize>) -> Option<usize> {
    for key in get_parent_keys(summary) {
        if let Some(index) = rows_by_key.get(&key) {
            return Some(*index);
        }
    }
    None
}

pub fn is_subagent_summary(summary: &SessionSummary) -> bool {
    if summary.runtime_kind == Some(RuntimeKind::Subagent)
        || summary.rlm_depth.map(|depth| depth > 0).unwrap_or(false)
    {
        return true;
    }
    if summary.rlm_depth == Some(0) {
        return false;
    }
    if summary.runtime_kind.is_some() {
        return false;
    }
    // Summaries from daemons that predate runtimeKind still carry subagent
    // linkage; never surface those as top-level agents.
    summary.rlm_child_id.is_some()
        || summary.rlm_parent_node_id.is_some()
        || summary.parent_active_session_id.is_some()
        || summary.parent_session_id.is_some()
        || summary.parent_session_path.is_some()
}

fn section_rank(section: AgentsViewSection) -> i32 {
    match section {
        AgentsViewSection::Running => 0,
        AgentsViewSection::Idle => 1,
        AgentsViewSection::Inactive => 2,
    }
}

fn get_timestamp(value: Option<&str>) -> i64 {
    match value {
        None => 0,
        Some(value) => parse_js_timestamp(value).unwrap_or(0),
    }
}

pub fn get_agents_view_session_title(summary: &SessionSummary) -> String {
    // `basename` allocates, so the cwd fallback is owned here.
    let cwd_basename = basename(&summary.cwd);
    let candidates: [Option<&str>; 5] = [
        summary.session_name.as_deref(),
        summary.first_message.as_deref(),
        Some(cwd_basename.as_str()),
        Some(summary.session_id.as_str()),
        Some(summary.id.as_str()),
    ];
    for candidate in candidates {
        let normalized = candidate.map(collapse_whitespace).unwrap_or_default();
        if !normalized.is_empty() {
            return normalized;
        }
    }
    "Untitled agent".to_string()
}

/// `value.replace(/\s+/g, " ").trim()`.
pub fn collapse_whitespace(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut in_ws = false;
    for ch in value.chars() {
        if ch.is_whitespace() {
            in_ws = true;
        } else {
            if in_ws && !out.is_empty() {
                out.push(' ');
            }
            in_ws = false;
            out.push(ch);
        }
    }
    out
}

fn get_session_subtitle(summary: &SessionSummary) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(model) = &summary.model {
        parts.push(format!("{}/{}", model.provider, model.id));
    }
    if !summary.cwd.is_empty() {
        parts.push(summary.cwd.clone());
    }
    let session_key = summary.active_session_id.clone().unwrap_or_else(|| summary.id.clone());
    if !session_key.is_empty() {
        parts.push(session_key);
    }
    parts.join("  ")
}

fn get_session_status_label(summary: &SessionSummary, heartbeat: Option<&UnifiedSessionHeartbeat>) -> String {
    if let Some(label) = &summary.status_label {
        return match label {
            AgentRosterStatusLabel::Queued => "queued".to_string(),
            AgentRosterStatusLabel::Recovering => "recovering".to_string(),
            AgentRosterStatusLabel::Failed => "failed".to_string(),
            AgentRosterStatusLabel::BackgroundHelper => "background helper".to_string(),
        };
    }
    if let Some(last_heard) = &summary.last_heard_from_at {
        return format!("last heard {}", format_age_label(last_heard));
    }
    // A non-ready worker cannot report fresh runtime flags; its state is the row's story.
    if let Some(worker_state) = summary.worker_state {
        if worker_state != WorkerState::Ready {
            return match worker_state {
                WorkerState::Starting => "starting",
                WorkerState::Ready => "ready",
                WorkerState::Recovering => "recovering",
                WorkerState::Stopping => "stopping",
                WorkerState::Failed => "failed",
            }
            .to_string();
        }
    }
    if summary.is_compacting {
        return "compacting".to_string();
    }
    if summary.is_streaming {
        return if summary.is_running_tools == Some(true) { "running tools" } else { "thinking" }.to_string();
    }
    // These classify the session as Running (isAgentsViewSessionBusy); the label
    // must agree with the section instead of claiming the session needs input.
    if summary.is_running_tools == Some(true) {
        return "running tools".to_string();
    }
    if summary.is_bash_running == Some(true) {
        return "running bash".to_string();
    }
    if let Some(active) = &summary.session_actions.active {
        return active
            .label
            .clone()
            .unwrap_or_else(|| active.kind.replacen('_', " ", 1));
    }
    if summary.session_actions.queued_count > 0 {
        return format!("{} queued", summary.session_actions.queued_count);
    }
    if summary.lifecycle == SessionLifecycle::Archived {
        return "archived".to_string();
    }
    if summary.has_active_heartbeat == Some(true) {
        let next = heartbeat.and_then(|h| h.next_run_at.as_deref()).and_then(parse_js_timestamp);
        return match next {
            Some(value) => format!("heartbeat · next {}", format_heartbeat_countdown(value - now_ms())),
            None => "heartbeat active".to_string(),
        };
    }
    if summary.runtime_kind == Some(RuntimeKind::Subagent) && summary.replied_since_task == Some(true) {
        return "replied".to_string();
    }
    if summary.activity == SessionActivity::Working {
        return "classifying".to_string();
    }
    if summary.task_state == Some(AgentTaskState::Completed) {
        "completed".to_string()
    } else {
        "needs input".to_string()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn daemon_summary(id: &str, active: Option<&str>, file: Option<&str>) -> SessionSummary {
        let mut summary = SessionSummary::new(id, id, "C:/work");
        summary.active_session_id = active.map(|value| value.to_string());
        summary.session_file = file.map(|value| value.to_string());
        summary.message_count = 3;
        summary
    }

    fn saved(id: &str, path: &str) -> AgentConnectionSavedSessionInfo {
        AgentConnectionSavedSessionInfo {
            path: path.to_string(),
            id: id.to_string(),
            cwd: "C:/work".to_string(),
            name: None,
            state: None,
            parent_session_path: None,
            rlm_depth: None,
            created: Utc.timestamp_millis_opt(0).unwrap(),
            modified: Utc.timestamp_millis_opt(1000).unwrap(),
            message_count: 2,
            first_message: "hello".to_string(),
            all_messages_text: "hello world".to_string(),
            agent_status: None,
            usage: None,
        }
    }

    #[test]
    fn session_summary_round_trips_the_wire_field_names() {
        let summary = SessionSummary::new("a-1", "s-1", "C:/work");
        let value = serde_json::to_value(&summary).unwrap();
        assert_eq!(value["sessionId"], json!("s-1"));
        assert_eq!(value["activeSessionId"], Value::Null);
        assert_eq!(value["sessionActions"]["queuedCount"], json!(0));
        let back: SessionSummary = serde_json::from_value(value).unwrap();
        assert_eq!(back.session_id, "s-1");
        assert_eq!(back.active_session_id, None);
    }

    #[test]
    fn session_summary_preserves_communication_queue_and_ignores_unknown_versions() {
        let mut row = serde_json::json!({
            "id":"p", "sessionId":"p", "lifecycle":"live", "activity":"idle", "isSessionActive":false,
            "cwd":"/tmp", "isStreaming":false, "isCompacting":false, "attachedClients":0, "messageCount":0,
            "communicationQueue": {
                "generation":"g", "version": 99,
                "pendingTotal": 2, "omitted": 1,
                "entries":[{"queueId":"q1","messageId":"m1","position":1,
                             "receiver":"sibling/a","state":"pending","attempts":0,"timestamp":1000}]
            }
        });
        let parsed: SessionSummary = serde_json::from_value(row.clone()).unwrap();
        assert_eq!(parsed.communication_queue.as_ref().unwrap()["entries"][0]["queueId"], "q1");
        assert_eq!(parsed.communication_queue.as_ref().unwrap()["pendingTotal"], 2);
        assert_eq!(parsed.communication_queue.as_ref().unwrap()["omitted"], 1);
        // Round-trip keeps the field; legacy rows without it still parse.
        let serialized = serde_json::to_value(&parsed).unwrap();
        assert!(serialized["communicationQueue"].is_object());
        row.as_object_mut().unwrap().remove("communicationQueue");
        let legacy: SessionSummary = serde_json::from_value(row).unwrap();
        assert!(legacy.communication_queue.is_none());
    }

    #[test]
    fn session_summary_preserves_streaming_throughput_without_reinterpreting_legacy() {
        let metadata = json!({"generation":"g", "outputTokens":300, "streamingMs":1000, "responses":1});
        let summary: SessionSummary = serde_json::from_value(json!({
            "sessionId":"a", "streamingThroughput":metadata,
            "generationThroughput":{"generation":"old", "outputTokens":999, "generationMs":1},
        })).unwrap();
        let projected = serde_json::to_value(&summary).unwrap();
        assert_eq!(projected["streamingThroughput"], metadata);
        assert!(projected.get("generationThroughput").is_none());
        let legacy: SessionSummary = serde_json::from_value(json!({
            "sessionId":"a", "generationThroughput":{"generation":"old", "outputTokens":999, "generationMs":1},
        })).unwrap();
        assert!(legacy.streaming_throughput.is_none());
        assert!(serde_json::to_value(legacy).unwrap().get("streamingThroughput").is_none());
    }

    #[test]
    fn classify_agent_status_matches_the_reference_table() {
        assert_eq!(classify_agent_status(true, true, false), AgentRosterStatus::Running);
        assert_eq!(classify_agent_status(false, false, true), AgentRosterStatus::Inactive);
        assert_eq!(classify_agent_status(true, false, true), AgentRosterStatus::Running);
        assert_eq!(classify_agent_status(true, false, false), AgentRosterStatus::Idle);
    }

    #[test]
    fn background_helper_fallback_is_idle_without_hiding_queued_recovery_status() {
        let mut summary = daemon_summary("a", Some("a"), None);
        summary.is_session_active = true;
        summary.status_label = Some(AgentRosterStatusLabel::BackgroundHelper);
        assert_eq!(classify_agents_view_session(&summary), AgentsViewSection::Idle);
        summary.roster_status = Some(AgentRosterStatus::Running);
        summary.status_label = Some(AgentRosterStatusLabel::Queued);
        assert_eq!(classify_agents_view_session(&summary), AgentsViewSection::Running);
        summary.status_label = Some(AgentRosterStatusLabel::Recovering);
        assert_eq!(classify_agents_view_session(&summary), AgentsViewSection::Running);
    }

    #[test]
    fn section_titles_are_exact() {
        assert_eq!(section_title(AgentsViewSection::Running), "Running");
        assert_eq!(section_title(AgentsViewSection::Idle), "Idle");
        assert_eq!(section_title(AgentsViewSection::Inactive), "Inactive");
    }

    #[test]
    fn scope_transitions_push_replace_and_back() {
        let a = AgentsViewScopeKey { session_id: "a".into(), active_session_id: None };
        let b = AgentsViewScopeKey { session_id: "b".into(), active_session_id: None };
        let frames = transition_agents_view_scope(&[], &AgentsViewScopeAction::Push { scope: a.clone(), return_chat: None });
        assert_eq!(frames.len(), 1);
        let frames = transition_agents_view_scope(&frames, &AgentsViewScopeAction::Push { scope: b.clone(), return_chat: None });
        assert_eq!(frames.len(), 2);
        // Re-pushing the same session id replaces the top frame instead of stacking.
        let frames = transition_agents_view_scope(&frames, &AgentsViewScopeAction::Push { scope: b, return_chat: None });
        assert_eq!(frames.len(), 2);
        let frames = transition_agents_view_scope(&frames, &AgentsViewScopeAction::Back);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].scope.session_id, "a");
        assert!(transition_agents_view_scope(&frames, &AgentsViewScopeAction::Back).is_empty());
    }

    #[test]
    fn reconcile_merges_saved_rows_onto_daemon_rows_by_file() {
        let daemon = vec![daemon_summary("a-1", Some("a-1"), Some("C:/sessions/a.jsonl"))];
        let saved_rows = vec![saved("a-1", "C:/sessions/a.jsonl")];
        let records = reconcile_unified_sessions(&daemon, &saved_rows, &[]);
        assert_eq!(records.len(), 1);
        assert!(records[0].daemon.is_some());
        assert!(records[0].saved.is_some());
        assert!(records[0].searchable_text.contains("hello world"));
        assert_eq!(records[0].section, AgentsViewSection::Idle);
    }

    #[test]
    fn reconcile_keeps_unmatched_saved_rows_inactive() {
        let records = reconcile_unified_sessions(&[], &[saved("z-9", "C:/sessions/z.jsonl")], &[]);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].section, AgentsViewSection::Inactive);
        let summary = summary_for_unified_record(&records[0]);
        assert_eq!(summary.lifecycle, SessionLifecycle::Archived);
        assert_eq!(summary.session_id, "z-9");
    }

    #[test]
    fn heartbeat_aggregation_rolls_up_to_parent_owners() {
        let parent = daemon_summary("p", Some("p"), None);
        let child = {
            let mut child = daemon_summary("c", Some("c"), None);
            child.parent_active_session_id = Some("p".to_string());
            child
        };
        let heartbeat = AgentConnectionHeartbeat {
            job: AgentCronJob {
                id: "job-1".into(),
                status: "active".into(),
                active_session_id: "c".into(),
                session_id: "c".into(),
                session_file: "C:/sessions/c.jsonl".into(),
                next_run_at: Some("2030-01-01T00:00:00.000Z".into()),
                ..AgentCronJob::default()
            },
            session_name: None,
            first_message: None,
        };
        let map = aggregate_session_heartbeats(&[parent, child], &[heartbeat]);
        assert_eq!(map.get("c").unwrap().active_count, 1);
        assert_eq!(map.get("p").unwrap().active_count, 1);
        assert!(map.get("c").unwrap().next_run_at.is_some());
    }

    #[test]
    fn heartbeat_badge_matches_the_reference_strings() {
        assert_eq!(format_heartbeat_badge(None, 0), "");
        assert_eq!(
            format_heartbeat_badge(Some(&UnifiedSessionHeartbeat { active_count: 0, paused_count: Some(2), next_run_at: None }), 0),
            "♥ 2"
        );
        assert_eq!(
            format_heartbeat_badge(Some(&UnifiedSessionHeartbeat { active_count: 2, paused_count: None, next_run_at: None }), 0),
            "♥ 2"
        );
        let now = 1_000_000i64;
        assert_eq!(
            format_heartbeat_badge(
                Some(&UnifiedSessionHeartbeat {
                    active_count: 1,
                    paused_count: None,
                    next_run_at: Some(to_iso_string(&Utc.timestamp_millis_opt(now + 30_000).unwrap())),
                }),
                now
            ),
            "♥ 1·30s"
        );
    }

    #[test]
    fn rows_nest_subagents_and_flatten_with_summary_and_code_rows() {
        let parent = {
            let mut summary = daemon_summary("p", Some("p"), None);
            summary.session_name = Some("Parent".into());
            summary
        };
        let child = {
            let mut summary = daemon_summary("c", Some("c"), None);
            summary.runtime_kind = Some(RuntimeKind::Subagent);
            summary.rlm_child_id = Some("child-1".into());
            summary.parent_active_session_id = Some("p".into());
            summary.spawn_code = Some("line1\nline2".into());
            summary
        };
        let inputs = vec![
            AgentsViewRowInput::Summary(parent),
            AgentsViewRowInput::Summary(child),
        ];
        let rows = build_agents_view_rows(&inputs, &HashSet::new(), &HashSet::new(), None, None, None);
        // Parent row + collapsed summary row (children hidden).
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].kind, AgentsViewRowKind::Agent);
        assert_eq!(rows[1].kind, AgentsViewRowKind::SubagentSummary);
        assert_eq!(rows[1].title, "1 subagent");
        assert_eq!(rows[1].has_spawn_code, Some(true));

        let expanded: HashSet<String> = [rows[0].identity.clone()].into_iter().collect();
        let program: HashSet<String> = [rows[0].identity.clone()].into_iter().collect();
        let rows = build_agents_view_rows(&inputs, &expanded, &program, None, None, None);
        let kinds: Vec<AgentsViewRowKind> = rows.iter().map(|row| row.kind).collect();
        assert_eq!(
            kinds,
            vec![
                AgentsViewRowKind::Agent,
                AgentsViewRowKind::SubagentSummary,
                AgentsViewRowKind::SubagentCode,
                AgentsViewRowKind::SubagentCode,
                AgentsViewRowKind::SubagentCode,
                AgentsViewRowKind::SubagentCode,
                AgentsViewRowKind::Subagent,
            ]
        );
        assert_eq!(rows[2].code.as_deref(), Some(""));
        assert_eq!(rows[3].code.as_deref(), Some("line1"));
        assert_eq!(rows[4].code.as_deref(), Some("line2"));
        assert_eq!(rows[5].code.as_deref(), Some(""));
        assert!(!rows[3].selectable);
        assert_eq!(rows[1].running_subagent_count, 0);
    }

    #[test]
    fn spawn_code_rows_cap_the_body_and_note_the_remainder() {
        let parent = AgentsViewRow {
            identity: "agent:x".into(),
            summary: SessionSummary::new("x", "x", "C:/w"),
            ..AgentsViewRow::default()
        };
        let code = (0..12).map(|i| format!("l{i}")).collect::<Vec<String>>().join("\n");
        let rows = build_spawn_code_rows(&parent, &code, 1, 0);
        // pad-top + 10 lines + "more" note + pad-bottom
        assert_eq!(rows.len(), 13);
        assert_eq!(rows[11].code.as_deref(), Some("… +2 more lines"));
        assert_eq!(rows[12].code.as_deref(), Some(""));
        assert_eq!(rows[0].identity, "code:agent:x:0:pad-top");
    }

    #[test]
    fn recursive_rollups_sum_costs_and_descendants() {
        let parent = {
            let mut summary = daemon_summary("p", Some("p"), None);
            summary.usage = Some(SessionUsageSummary { input_tokens: 1, output_tokens: 1, cost: 1.5 });
            summary
        };
        let child = {
            let mut summary = daemon_summary("c", Some("c"), None);
            summary.runtime_kind = Some(RuntimeKind::Subagent);
            summary.rlm_child_id = Some("child-1".into());
            summary.parent_active_session_id = Some("p".into());
            summary.usage = Some(SessionUsageSummary { input_tokens: 1, output_tokens: 1, cost: 2.0 });
            summary
        };
        let records = reconcile_unified_sessions(&[parent, child], &[], &[]);
        let index = build_unified_session_index(&records);
        let rollups = compute_recursive_rollups(&records, Some(&index));
        let parent_key = records[0].identity.clone();
        assert_eq!(rollups.get(&parent_key).unwrap().cost, 3.5);
        assert_eq!(rollups.get(&parent_key).unwrap().descendant_count, 1);
        assert_eq!(rollups.get(&records[1].identity).unwrap().cost, 2.0);
    }

    #[test]
    fn filters_keep_ancestors_of_matching_descendants() {
        let parent = {
            let mut summary = daemon_summary("p", Some("p"), None);
            summary.session_name = Some("Parent".into());
            summary
        };
        let child = {
            let mut summary = daemon_summary("c", Some("c"), None);
            summary.runtime_kind = Some(RuntimeKind::Subagent);
            summary.rlm_child_id = Some("child-1".into());
            summary.parent_active_session_id = Some("p".into());
            summary.session_name = Some("Needle".into());
            summary
        };
        let records = reconcile_unified_sessions(&[parent, child], &[], &[]);
        let filtered = filter_unified_sessions(&records, &|text| text.contains("Needle"));
        assert_eq!(filtered.len(), 2);
        let none = filter_unified_sessions(&records, &|_| false);
        assert!(none.is_empty());
    }

    #[test]
    fn identity_migration_rewrites_stale_keys() {
        let daemon = vec![daemon_summary("a-1", Some("a-1"), Some("C:/sessions/a.jsonl"))];
        let records = reconcile_unified_sessions(&daemon, &[saved("a-1", "C:/sessions/a.jsonl")], &[]);
        let index = build_unified_session_index(&records);
        let mut identities: HashSet<String> = ["session:a-1".to_string(), "unknown".to_string()].into_iter().collect();
        migrate_agents_view_identity_set(&mut identities, &index.by_key);
        assert!(identities.contains(&records[0].identity));
        assert!(identities.contains("unknown"));
    }

    #[test]
    fn titles_and_status_labels_match_the_reference() {
        let mut summary = SessionSummary::new("id-1", "sess-1", "C:/work/project");
        assert_eq!(get_agents_view_session_title(&summary), "project");
        summary.first_message = Some("  spaced   prompt ".into());
        assert_eq!(get_agents_view_session_title(&summary), "spaced prompt");
        summary.session_name = Some("  Named  ".into());
        assert_eq!(get_agents_view_session_title(&summary), "Named");

        summary.session_name = None;
        summary.first_message = None;
        summary.message_count = 4;
        assert_eq!(get_session_status_label(&summary, None), "needs input");
        summary.task_state = Some(AgentTaskState::Completed);
        assert_eq!(get_session_status_label(&summary, None), "completed");
        summary.status_label = Some(AgentRosterStatusLabel::Queued);
        assert_eq!(get_session_status_label(&summary, None), "queued");
        summary.status_label = None;
        summary.is_streaming = true;
        assert_eq!(get_session_status_label(&summary, None), "thinking");
        summary.is_running_tools = Some(true);
        assert_eq!(get_session_status_label(&summary, None), "running tools");
    }

    #[test]
    fn subagent_summary_detection_covers_legacy_summaries() {
        let mut summary = SessionSummary::new("id", "sess", "C:/w");
        assert!(!is_subagent_summary(&summary));
        summary.rlm_depth = Some(0);
        assert!(!is_subagent_summary(&summary));
        summary.parent_session_path = Some("C:/p.jsonl".into());
        assert!(!is_subagent_summary(&summary));
        summary.rlm_depth = None;
        assert!(is_subagent_summary(&summary));
        summary.parent_session_path = None;
        summary.runtime_kind = Some(RuntimeKind::TopLevel);
        assert!(!is_subagent_summary(&summary));
    }

    #[test]
    fn selection_resolution_prefers_exact_row_identity() {
        let rows = vec![
            AgentsViewRow { identity: "a".into(), kind: AgentsViewRowKind::Agent, selectable: true, ..AgentsViewRow::default() },
            AgentsViewRow {
                identity: "subagents:a".into(),
                kind: AgentsViewRowKind::SubagentSummary,
                selectable: true,
                ..AgentsViewRow::default()
            },
        ];
        assert_eq!(resolve_agents_view_selection_index(&rows, Some("subagents:a"), None), 1);
        assert_eq!(resolve_agents_view_selection_index(&rows, Some("a"), None), 0);
        assert_eq!(resolve_agents_view_selection_index(&rows, Some("gone"), None), -1);
        let empty: Vec<AgentsViewRow> = Vec::new();
        let resolution = resolve_agents_view_selection_state(&empty, 0, None, None);
        assert_eq!(resolution.index, 0);
        assert!(!resolution.resolved);
    }

    #[test]
    fn unattachable_child_open_result_is_exact() {
        let child = SessionSummary::new("c", "c", "C:/w");
        let parent = SessionSummary::new("p", "p", "C:/w");
        let result = create_unattachable_child_open_result(&child, &parent, &["root".to_string()], true);
        assert_eq!(result.kind, "open");
        assert_eq!(result.summary.session_id, "p");
        assert_eq!(result.selection.session_id, "c");
        assert_eq!(result.status_message, "Child session is unavailable; opened its parent instead");
        assert!(result.has_children);
    }
}
