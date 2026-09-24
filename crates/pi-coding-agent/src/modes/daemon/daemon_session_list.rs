//! Port of packages/coding-agent/src/modes/daemon/daemon-session-list.ts
//!
//! `SessionInfo` and `AgentCronJob` are the canonical `core` types, re-exported
//! here rather than restated: the daemon reads the same objects the session and
//! cron owners produce, so a second definition could only drift.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::active_session_state::ActiveSessionState;
use crate::core::cron_jobs::is_heartbeat_cron_job;

use super::agent_roster::{is_session_summary_busy, AgentRosterStatus};
use super::agent_roster::RosterSessionSummary;

/// Durable lifecycle; decides agents-view visibility. Only "live" is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionLifecycle {
    Draft,
    Live,
    Archived,
}

impl SessionLifecycle {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionLifecycle::Draft => "draft",
            SessionLifecycle::Live => "live",
            SessionLifecycle::Archived => "archived",
        }
    }
}

/// Heuristic activity of a live session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionActivity {
    Working,
    Idle,
}

impl SessionActivity {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionActivity::Working => "working",
            SessionActivity::Idle => "idle",
        }
    }
}

/// Upper bound on the spawn-code source carried in a session summary.
pub const SPAWN_CODE_MAX_CHARS: usize = 4000;
const MAX_DATE_TIMESTAMP_MS: f64 = 8.64e15;

// `SessionState` / `AgentStatus` / `SessionInfo` are the canonical session-manager
// types; `AgentCronJob` is the canonical cron-jobs type. The daemon previously kept
// lossy copies (`status: Option<String>` where TS requires `SessionStateStatus`,
// `AgentStatus` for `AgentStatus`, `Date` fields as `*_ms`, and an invented
// `heartbeat: bool` standing in for `isHeartbeatCronJob`). A duplicate only drifts.
pub use crate::core::cron_jobs::AgentCronJob;
pub use crate::core::session_manager::{AgentStatus, SessionInfo, SessionState};
use crate::core::session_manager::{AgentTaskState, SessionStateStatus};

/// The `SessionSummary` projection used by the daemon wire.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub lifecycle: String,
    pub activity: String,
    #[serde(rename = "isSessionActive")]
    pub is_session_active: bool,
    #[serde(rename = "hasActiveHeartbeat", skip_serializing_if = "Option::is_none", default)]
    pub has_active_heartbeat: Option<bool>,
    #[serde(rename = "hasRegisteredHeartbeat", skip_serializing_if = "Option::is_none", default)]
    pub has_registered_heartbeat: Option<bool>,
    #[serde(rename = "hasRegisteredCronJob", skip_serializing_if = "Option::is_none", default)]
    pub has_registered_cron_job: Option<bool>,
    #[serde(rename = "lastActivityAt", skip_serializing_if = "Option::is_none", default)]
    pub last_activity_at: Option<String>,
    #[serde(rename = "runtimeKind", skip_serializing_if = "Option::is_none", default)]
    pub runtime_kind: Option<String>,
    #[serde(rename = "rlmDepth", skip_serializing_if = "Option::is_none", default)]
    pub rlm_depth: Option<i64>,
    #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none", default)]
    pub active_session_id: Option<String>,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    #[serde(rename = "sessionFile", skip_serializing_if = "Option::is_none", default)]
    pub session_file: Option<String>,
    #[serde(rename = "sessionName", skip_serializing_if = "Option::is_none", default)]
    pub session_name: Option<String>,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model: Option<Value>,
    #[serde(rename = "thinkingLevel", skip_serializing_if = "Option::is_none", default)]
    pub thinking_level: Option<String>,
    #[serde(rename = "isStreaming")]
    pub is_streaming: bool,
    #[serde(rename = "isCompacting")]
    pub is_compacting: bool,
    #[serde(rename = "isBashRunning", skip_serializing_if = "Option::is_none", default)]
    pub is_bash_running: Option<bool>,
    #[serde(rename = "hasRunningRlmChildren", skip_serializing_if = "Option::is_none", default)]
    pub has_running_rlm_children: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub usage: Option<Value>,
    #[serde(rename = "isRunningTools", skip_serializing_if = "Option::is_none", default)]
    pub is_running_tools: Option<bool>,
    #[serde(rename = "attachedClients")]
    pub attached_clients: i64,
    #[serde(rename = "directAttachedClients", skip_serializing_if = "Option::is_none", default)]
    pub direct_attached_clients: Option<i64>,
    #[serde(rename = "messageCount")]
    pub message_count: i64,
    #[serde(rename = "unfinishedActionCount", skip_serializing_if = "Option::is_none", default)]
    pub unfinished_action_count: Option<i64>,
    #[serde(rename = "sessionActions", skip_serializing_if = "Option::is_none", default)]
    pub session_actions: Option<Value>,
    #[serde(rename = "streamingMessage", skip_serializing_if = "Option::is_none", default)]
    pub streaming_message: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub created: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub modified: Option<String>,
    #[serde(rename = "firstMessage", skip_serializing_if = "Option::is_none", default)]
    pub first_message: Option<String>,
    #[serde(rename = "parentActiveSessionId", skip_serializing_if = "Option::is_none", default)]
    pub parent_active_session_id: Option<String>,
    #[serde(rename = "parentSessionId", skip_serializing_if = "Option::is_none", default)]
    pub parent_session_id: Option<String>,
    #[serde(rename = "parentSessionPath", skip_serializing_if = "Option::is_none", default)]
    pub parent_session_path: Option<String>,
    #[serde(rename = "rlmChildId", skip_serializing_if = "Option::is_none", default)]
    pub rlm_child_id: Option<String>,
    #[serde(rename = "repliedSinceTask", skip_serializing_if = "Option::is_none", default)]
    pub replied_since_task: Option<bool>,
    #[serde(rename = "rlmParentNodeId", skip_serializing_if = "Option::is_none", default)]
    pub rlm_parent_node_id: Option<String>,
    #[serde(rename = "spawnCode", skip_serializing_if = "Option::is_none", default)]
    pub spawn_code: Option<String>,
    #[serde(rename = "modelFallbackMessage", skip_serializing_if = "Option::is_none", default)]
    pub model_fallback_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub diagnostics: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub summary: Option<String>,
    #[serde(rename = "taskState", skip_serializing_if = "Option::is_none", default)]
    pub task_state: Option<String>,
    #[serde(rename = "rosterStatus", skip_serializing_if = "Option::is_none", default)]
    pub roster_status: Option<AgentRosterStatus>,
    #[serde(rename = "statusLabel", skip_serializing_if = "Option::is_none", default)]
    pub status_label: Option<String>,
    #[serde(rename = "lastHeardFromAt", skip_serializing_if = "Option::is_none", default)]
    pub last_heard_from_at: Option<String>,
    #[serde(rename = "workerState", skip_serializing_if = "Option::is_none", default)]
    pub worker_state: Option<String>,
    #[serde(rename = "workerPid", skip_serializing_if = "Option::is_none", default)]
    pub worker_pid: Option<i64>,
    /// Summary fields this slice does not model yet; kept so nothing is lost.
    #[serde(flatten, default)]
    pub extra: HashMap<String, Value>,
}

impl SessionSummary {
    pub fn roster_view(&self) -> RosterSessionSummary {
        RosterSessionSummary {
            id: self.id.clone(),
            lifecycle: self.lifecycle.clone(),
            activity: self.activity.clone(),
            is_session_active: self.is_session_active,
            has_active_heartbeat: self.has_active_heartbeat,
            has_registered_heartbeat: self.has_registered_heartbeat,
            has_registered_cron_job: self.has_registered_cron_job,
            last_activity_at: self.last_activity_at.clone(),
            runtime_kind: self.runtime_kind.clone(),
            rlm_depth: self.rlm_depth,
            active_session_id: self.active_session_id.clone(),
            session_id: self.session_id.clone(),
            session_file: self.session_file.clone(),
            session_name: self.session_name.clone(),
            cwd: self.cwd.clone(),
            model: self.model.clone(),
            thinking_level: self.thinking_level.clone(),
            is_streaming: self.is_streaming,
            is_compacting: self.is_compacting,
            is_bash_running: self.is_bash_running,
            has_running_rlm_children: self.has_running_rlm_children,
            usage: self.usage.clone(),
            is_running_tools: self.is_running_tools,
            attached_clients: self.attached_clients,
            direct_attached_clients: self.direct_attached_clients,
            message_count: self.message_count,
            unfinished_action_count: self.unfinished_action_count,
            session_actions: self.session_actions.clone(),
            streaming_message: self.streaming_message.clone(),
            created: self.created.clone(),
            modified: self.modified.clone(),
            first_message: self.first_message.clone(),
            parent_active_session_id: self.parent_active_session_id.clone(),
            parent_session_id: self.parent_session_id.clone(),
            parent_session_path: self.parent_session_path.clone(),
            rlm_child_id: self.rlm_child_id.clone(),
            replied_since_task: self.replied_since_task,
            rlm_parent_node_id: self.rlm_parent_node_id.clone(),
            spawn_code: self.spawn_code.clone(),
            model_fallback_message: self.model_fallback_message.clone(),
            diagnostics: self.diagnostics.clone(),
            summary: self.summary.clone(),
            task_state: self.task_state.clone(),
            roster_status: self.roster_status,
            status_label: self.status_label.clone(),
            last_heard_from_at: self.last_heard_from_at.clone(),
            worker_state: self.worker_state.clone(),
            worker_pid: self.worker_pid,
            extra: self.extra.clone(),
        }
    }
}

/// Pick the model fallback message to show when attaching to a daemon session.
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
    startup_model_fallback_message.map(str::to_string)
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScheduledJobRegistrations {
    pub active_heartbeat_session_ids: HashSet<String>,
    pub heartbeat_session_ids: HashSet<String>,
    pub cron_session_ids: HashSet<String>,
    pub heartbeat_session_files: HashSet<String>,
    pub cron_session_files: HashSet<String>,
}

pub fn scheduled_job_registrations(scheduled_jobs: &[AgentCronJob]) -> ScheduledJobRegistrations {
    let mut registrations = ScheduledJobRegistrations::default();
    for job in scheduled_jobs {
        // TS daemon-session-list.ts:127 calls `isHeartbeatCronJob(job)`, which reads
        // the job's `source` ("heartbeat"/"rlm_heartbeat"); there is no `heartbeat` flag.
        let heartbeat = is_heartbeat_cron_job(job);
        if heartbeat && job.status == "active" {
            registrations
                .active_heartbeat_session_ids
                .insert(job.active_session_id.clone());
        }
        // A paused heartbeat cannot fire, so unlike a live heartbeat (or a
        // registered cron job) it must not silently pin a worker forever.
        let registered = if heartbeat {
            job.status == "active"
        } else {
            job.status == "active" || job.status == "paused"
        };
        if !registered {
            continue;
        }
        let resolved_file = resolve_path(&job.session_file);
        if heartbeat {
            registrations.heartbeat_session_ids.insert(job.active_session_id.clone());
            registrations.heartbeat_session_files.insert(resolved_file);
        } else {
            registrations.cron_session_ids.insert(job.active_session_id.clone());
            registrations.cron_session_files.insert(resolved_file);
        }
    }
    registrations
}

/// Naming signals intent to return, so named sessions are exempt even when empty.
pub fn is_evictable_empty_session_summary(summary: &SessionSummary) -> bool {
    summary.message_count == 0
        && summary.session_name.is_none()
        && !is_session_summary_busy(summary.is_session_active, summary.has_running_rlm_children)
        && summary.has_registered_cron_job != Some(true)
}

pub fn build_session_list(
    active_sessions: &[Arc<StdMutex<ActiveSessionState>>],
    saved_sessions: &[SessionInfo],
    scheduled_jobs: &[AgentCronJob],
) -> Vec<SessionSummary> {
    let mut active_by_session_file: HashMap<String, Arc<StdMutex<ActiveSessionState>>> = HashMap::new();
    let registrations = scheduled_job_registrations(scheduled_jobs);

    for active_session in active_sessions {
        let session_file = active_session
            .lock()
            .expect("active session poisoned")
            .runtime
            .session
            .session_file
            .clone();
        if let Some(session_file) = session_file {
            active_by_session_file.insert(resolve_path(&session_file), Arc::clone(active_session));
        }
    }

    let mut entries: Vec<SessionSummary> = Vec::new();
    let mut seen_active_session_ids: HashSet<String> = HashSet::new();
    for saved_session in saved_sessions {
        let session_file = resolve_path(&saved_session.path);
        match active_by_session_file.get(&session_file) {
            Some(active_session) => {
                let active_session_id = active_session
                    .lock()
                    .expect("active session poisoned")
                    .active_session_id
                    .clone();
                entries.push(summary_for_active_session(
                    active_session,
                    Some(saved_session),
                    registrations.active_heartbeat_session_ids.contains(&active_session_id),
                    registrations.heartbeat_session_ids.contains(&active_session_id)
                        || registrations.heartbeat_session_files.contains(&session_file),
                    registrations.cron_session_ids.contains(&active_session_id)
                        || registrations.cron_session_files.contains(&session_file),
                ));
                seen_active_session_ids.insert(active_session_id);
            }
            None => entries.push(summary_for_inactive_session(
                saved_session,
                registrations.heartbeat_session_files.contains(&session_file),
                registrations.cron_session_files.contains(&session_file),
            )),
        }
    }

    for active_session in active_sessions {
        let active_session_id = active_session
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        if seen_active_session_ids.contains(&active_session_id) {
            continue;
        }
        let session_file = active_session
            .lock()
            .expect("active session poisoned")
            .runtime
            .session
            .session_file
            .clone();
        let resolved_session_file = session_file.map(|file| resolve_path(&file));
        entries.push(summary_for_active_session(
            active_session,
            None,
            registrations.active_heartbeat_session_ids.contains(&active_session_id),
            registrations.heartbeat_session_ids.contains(&active_session_id)
                || resolved_session_file
                    .as_ref()
                    .is_some_and(|file| registrations.heartbeat_session_files.contains(file)),
            registrations.cron_session_ids.contains(&active_session_id)
                || resolved_session_file
                    .as_ref()
                    .is_some_and(|file| registrations.cron_session_files.contains(file)),
        ));
    }
    entries
}

pub fn summary_for_active_session(
    active_session: &Arc<StdMutex<ActiveSessionState>>,
    saved_session: Option<&SessionInfo>,
    has_active_heartbeat: bool,
    has_registered_heartbeat: bool,
    has_registered_cron_job: bool,
) -> SessionSummary {
    let state = active_session.lock().expect("active session poisoned");
    let session = &state.runtime.session;
    let mut modified = saved_session.map(|saved| iso_from_ms(saved.modified));
    if modified.is_none() {
        if let Some(session_file) = &session.session_file {
            if let Ok(metadata) = std::fs::metadata(session_file) {
                if let Ok(modified_time) = metadata.modified() {
                    modified = Some(iso_from_system_time(modified_time));
                }
            }
        }
    }

    let direct_attached_clients = state
        .clients
        .iter()
        .filter(|client| {
            client
                .lock()
                .expect("daemon client poisoned")
                .authentication_role
                .as_deref()
                == Some("session_client")
        })
        .count();

    let metadata = state.runtime.metadata.clone().unwrap_or_default();
    let is_running_tools = session.is_streaming;
    SessionSummary {
        id: state.active_session_id.clone(),
        lifecycle: active_lifecycle_for_session(&state).as_str().to_string(),
        activity: active_activity_for_session(&state).as_str().to_string(),
        is_session_active: session.is_session_active,
        has_active_heartbeat: has_active_heartbeat.then_some(true),
        has_registered_heartbeat: has_registered_heartbeat.then_some(true),
        has_registered_cron_job: has_registered_cron_job.then_some(true),
        last_activity_at: modified.clone(),
        runtime_kind: metadata.kind.clone(),
        rlm_depth: session.rlm_depth,
        active_session_id: Some(state.active_session_id.clone()),
        session_id: session.session_id.clone(),
        session_file: session.session_file.clone(),
        session_name: session.session_name.clone(),
        // TS reads the live session cwd (`session.sessionManager.getCwd()`,
        // daemon-session-list.ts:256). This builder only sees the bind-time
        // view, so the saved twin's recorded repo cwd is the faithful value;
        // empty stays reserved for genuinely unknown (UI008/UI011 grouping).
        cwd: saved_session.map(|saved| saved.cwd.clone()).unwrap_or_default(),
        // `daemon-session-list.ts:257-258` copies the live model and thinking level onto
        // every summary; the `--print`/`--json` clients gate on `summary.model`
        // (`main.ts:1628`), so leaving it unset made every attach fail with
        // "No models available" even though the worker had resolved a model.
        model: session
            .model_identity
            .as_ref()
            .and_then(|model| serde_json::to_value(model).ok()),
        thinking_level: session.thinking_level.clone(),
        is_streaming: session.is_streaming,
        is_compacting: session.is_compacting,
        is_bash_running: None,
        has_running_rlm_children: Some(session.has_running_rlm_children),
        usage: None,
        is_running_tools: Some(is_running_tools),
        attached_clients: state.clients.len() as i64,
        direct_attached_clients: (direct_attached_clients > 0).then_some(direct_attached_clients as i64),
        message_count: session.messages_len as i64,
        unfinished_action_count: None,
        session_actions: None,
        streaming_message: None,
        created: saved_session.map(|saved| iso_from_ms(saved.created)),
        modified,
        // Subagent sessions live in artifact dirs the saved-session scan never
        // sees; their spawn prompt is the most identifying title we have.
        first_message: saved_session
            .map(|saved| saved.first_message.clone())
            .filter(|first| !first.is_empty())
            .or_else(|| metadata.prompt.as_ref().map(|prompt| compact_rlm_text(prompt, 120))),
        parent_active_session_id: metadata.parent_active_session_id.clone(),
        parent_session_id: metadata.parent_session_id.clone(),
        parent_session_path: saved_session
            .and_then(|saved| saved.parent_session_path.clone())
            .or_else(|| metadata.parent_session_file.clone()),
        rlm_child_id: metadata.rlm_child_id.clone(),
        replied_since_task: None,
        rlm_parent_node_id: metadata.rlm_parent_node_id.clone(),
        // Cap the cell source so the summary stays small on the daemon wire.
        spawn_code: metadata
            .spawn_code
            .as_ref()
            .map(|code| code.chars().take(SPAWN_CODE_MAX_CHARS).collect()),
        model_fallback_message: state.runtime.model_fallback_message.clone(),
        diagnostics: None,
        summary: state.summary_state.as_ref().map(|status| status.summary.clone()),
        task_state: if is_summary_current(&state) {
            state
                .summary_state
                .as_ref()
                .and_then(|status| status.task_state.clone())
        } else {
            None
        },
        roster_status: None,
        status_label: (session.is_session_active && session.is_foreground_active == Some(false))
            .then(|| "background helper".to_string()),
        last_heard_from_at: None,
        worker_state: None,
        worker_pid: None,
        extra: HashMap::new(),
    }
}

/// Latest message activity from message timestamps (milliseconds).
pub fn latest_message_activity_at(timestamps: &[f64]) -> Option<String> {
    let mut latest: Option<f64> = None;
    for timestamp in timestamps {
        if timestamp.is_finite() && timestamp.abs() <= MAX_DATE_TIMESTAMP_MS {
            latest = Some(match latest {
                Some(current) => current.max(*timestamp),
                None => *timestamp,
            });
        }
    }
    latest.map(iso_from_ms)
}

pub fn is_summary_current(state: &ActiveSessionState) -> bool {
    state
        .summary_state
        .as_ref()
        .is_some_and(|status| status.based_on_message_count == state.runtime.session.messages_len)
}

pub fn summary_for_inactive_session(
    session: &SessionInfo,
    has_registered_heartbeat: bool,
    has_registered_cron_job: bool,
) -> SessionSummary {
    let currency = session
        .agent_status
        .as_ref()
        .is_some_and(|status| status.based_on_message_count == session.message_count);
    SessionSummary {
        id: session.id.clone(),
        lifecycle: inactive_lifecycle_for_session(session).as_str().to_string(),
        activity: "idle".to_string(),
        is_session_active: false,
        has_active_heartbeat: None,
        has_registered_heartbeat: has_registered_heartbeat.then_some(true),
        has_registered_cron_job: has_registered_cron_job.then_some(true),
        last_activity_at: Some(iso_from_ms(session.modified)),
        runtime_kind: None,
        rlm_depth: None,
        active_session_id: None,
        session_id: session.id.clone(),
        session_file: Some(session.path.clone()),
        session_name: session.name.clone(),
        cwd: session.cwd.clone(),
        model: None,
        thinking_level: None,
        is_streaming: false,
        is_compacting: false,
        is_bash_running: None,
        has_running_rlm_children: None,
        usage: session.usage.as_ref().and_then(|usage| serde_json::to_value(usage).ok()),
        is_running_tools: None,
        attached_clients: 0,
        direct_attached_clients: None,
        message_count: session.message_count,
        unfinished_action_count: Some(0),
        session_actions: Some(serde_json::json!({
            "queuedCount": 0,
            "steering": [],
            "followUps": [],
        })),
        streaming_message: None,
        created: Some(iso_from_ms(session.created)),
        modified: Some(iso_from_ms(session.modified)),
        first_message: Some(session.first_message.clone()),
        parent_active_session_id: None,
        parent_session_id: None,
        parent_session_path: session.parent_session_path.clone(),
        rlm_child_id: None,
        replied_since_task: None,
        rlm_parent_node_id: None,
        spawn_code: None,
        model_fallback_message: None,
        diagnostics: None,
        summary: currency.then(|| {
            session
                .agent_status
                .as_ref()
                .map(|status| status.summary.clone())
                .unwrap_or_default()
        }),
        // TS daemon-session-list.ts:358-360 copies `agentStatus.taskState`, a string union;
        // the port's verdict is the `AgentTaskState` enum, mapped to its TS string here.
        task_state: currency
            .then(|| {
                session.agent_status.as_ref().and_then(|status| {
                    status.task_state.map(|task_state| match task_state {
                        AgentTaskState::NeedsInput => "needs_input".to_string(),
                        AgentTaskState::Completed => "completed".to_string(),
                    })
                })
            })
            .flatten(),
        roster_status: None,
        status_label: None,
        last_heard_from_at: None,
        worker_state: None,
        worker_pid: None,
        extra: HashMap::new(),
    }
}

/// Live work that dies with the worker; the display activity axis excludes delegated work.
pub fn has_live_session_work(state: &ActiveSessionState) -> bool {
    let session = &state.runtime.session;
    session.is_session_active || session.has_running_rlm_children
}

pub fn active_activity_for_session(state: &ActiveSessionState) -> SessionActivity {
    // The session's own work only, ignoring the classification verdict.
    if state.runtime.session.is_foreground_active.unwrap_or(state.runtime.session.is_session_active) {
        return SessionActivity::Working;
    }
    // A resident worker or an outstanding status classifier is not agent work.
    // Delegated work remains represented separately by has_running_rlm_children.
    SessionActivity::Idle
}

/// Lifecycle for an on-disk session not resident in the daemon.
pub fn inactive_lifecycle_for_session(session: &SessionInfo) -> SessionLifecycle {
    // TS daemon-session-list.ts:448 reads `session.state?.status`; the port's status is
    // the `SessionStateStatus` enum, so compare the enum rather than a string.
    let status = session.state.as_ref().map(|state| state.status);
    if matches!(status, Some(SessionStateStatus::Archived) | Some(SessionStateStatus::Crash)) {
        return SessionLifecycle::Archived;
    }
    if session.message_count > 0 {
        SessionLifecycle::Live
    } else {
        SessionLifecycle::Draft
    }
}

pub fn active_lifecycle_for_session(state: &ActiveSessionState) -> SessionLifecycle {
    // A resident subagent is a spawned worker, not a user draft.
    if state.runtime.metadata.as_ref().and_then(|metadata| metadata.kind.as_deref()) == Some("subagent") {
        return SessionLifecycle::Live;
    }
    if state.runtime.session.messages_len == 0 {
        SessionLifecycle::Draft
    } else {
        SessionLifecycle::Live
    }
}

/// `compactRlmText` from core/agent-session.ts (owned by another slice).
pub fn compact_rlm_text(text: &str, max: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<&str>>().join(" ");
    if normalized.chars().count() > max {
        let truncated: String = normalized.chars().take(max).collect();
        format!("{truncated}…")
    } else {
        normalized
    }
}

fn resolve_path(path: &str) -> String {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        candidate.to_string_lossy().to_string()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| Path::new(".").to_path_buf())
            .join(candidate)
            .to_string_lossy()
            .to_string()
    }
}

fn iso_from_ms(ms: f64) -> String {
    let millis = ms as i64;
    chrono::DateTime::from_timestamp_millis(millis)
        .map(|time| time.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_default()
}

fn iso_from_system_time(time: SystemTime) -> String {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => iso_from_ms(duration.as_millis() as f64),
        Err(_) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cron_jobs::{SOURCE_CRON, SOURCE_HEARTBEAT};

    fn summary(message_count: i64, name: Option<&str>) -> SessionSummary {
        SessionSummary {
            id: "id".to_string(),
            lifecycle: "live".to_string(),
            activity: "idle".to_string(),
            session_id: "session".to_string(),
            cwd: "/tmp".to_string(),
            session_name: name.map(str::to_string),
            message_count,
            ..Default::default()
        }
    }

    /// `core::session_manager::SessionInfo` has no `Default`; build the whole value.
    fn saved_session() -> SessionInfo {
        SessionInfo {
            path: "/tmp/s.jsonl".to_string(),
            id: "s".to_string(),
            cwd: "/tmp".to_string(),
            name: None,
            state: None,
            parent_session_path: None,
            rlm_depth: 0,
            created: 0.0,
            modified: 0.0,
            message_count: 0,
            first_message: String::new(),
            all_messages_text: String::new(),
            agent_status: None,
            usage: None,
        }
    }

    #[test]
    fn attach_fallback_prefers_the_daemon_summary() {
        let mut with_message = summary(1, None);
        with_message.model_fallback_message = Some("daemon says no models".to_string());
        assert_eq!(
            resolve_attach_model_fallback_message(&with_message, Some("startup says no models")).as_deref(),
            Some("daemon says no models")
        );
        let with_model = summary(1, None);
        let mut with_model = with_model;
        with_model.model = Some(serde_json::json!({ "id": "m" }));
        assert_eq!(
            resolve_attach_model_fallback_message(&with_model, Some("startup says no models")),
            None
        );
        let without_model = summary(1, None);
        assert_eq!(
            resolve_attach_model_fallback_message(&without_model, Some("startup says no models")).as_deref(),
            Some("startup says no models")
        );
    }

    #[test]
    fn scheduled_registrations_classify_heartbeats_and_cron() {
        let jobs = vec![
            AgentCronJob {
                active_session_id: "a".to_string(),
                session_file: "/tmp/a.jsonl".to_string(),
                status: "active".to_string(),
                source: Some(SOURCE_HEARTBEAT.to_string()),
                ..AgentCronJob::default()
            },
            AgentCronJob {
                active_session_id: "b".to_string(),
                session_file: "/tmp/b.jsonl".to_string(),
                status: "paused".to_string(),
                source: Some(SOURCE_HEARTBEAT.to_string()),
                ..AgentCronJob::default()
            },
            AgentCronJob {
                active_session_id: "c".to_string(),
                session_file: "/tmp/c.jsonl".to_string(),
                status: "paused".to_string(),
                source: Some(SOURCE_CRON.to_string()),
                ..AgentCronJob::default()
            },
        ];
        let registrations = scheduled_job_registrations(&jobs);
        assert!(registrations.active_heartbeat_session_ids.contains("a"));
        assert!(!registrations.active_heartbeat_session_ids.contains("b"));
        assert!(registrations.heartbeat_session_ids.contains("a"));
        assert!(!registrations.heartbeat_session_ids.contains("b"));
        assert!(registrations.cron_session_ids.contains("c"));
        assert_eq!(registrations.heartbeat_session_ids.len(), 1);
    }

    #[test]
    fn evictable_empty_sessions_are_unnamed_and_idle() {
        assert!(is_evictable_empty_session_summary(&summary(0, None)));
        assert!(!is_evictable_empty_session_summary(&summary(1, None)));
        assert!(!is_evictable_empty_session_summary(&summary(0, Some("named"))));
        let mut busy = summary(0, None);
        busy.has_running_rlm_children = Some(true);
        assert!(!is_evictable_empty_session_summary(&busy));
        let mut cron = summary(0, None);
        cron.has_registered_cron_job = Some(true);
        assert!(!is_evictable_empty_session_summary(&cron));
    }

    #[test]
    fn inactive_lifecycle_follows_message_count_and_state() {
        let mut session = saved_session();
        assert_eq!(inactive_lifecycle_for_session(&session), SessionLifecycle::Draft);
        session.message_count = 3;
        assert_eq!(inactive_lifecycle_for_session(&session), SessionLifecycle::Live);
        session.state = Some(SessionState {
            status: SessionStateStatus::Archived,
        });
        assert_eq!(inactive_lifecycle_for_session(&session), SessionLifecycle::Archived);
    }

    #[test]
    fn inactive_summary_carries_a_current_verdict_only() {
        let session = SessionInfo {
            message_count: 2,
            agent_status: Some(AgentStatus {
                summary: "Doing work".to_string(),
                task_state: Some(AgentTaskState::Completed),
                based_on_message_count: 2,
            }),
            ..saved_session()
        };
        let current = summary_for_inactive_session(&session, false, false);
        assert_eq!(current.summary.as_deref(), Some("Doing work"));
        assert_eq!(current.task_state.as_deref(), Some("completed"));
        assert_eq!(current.activity, "idle");

        let stale = SessionInfo {
            agent_status: Some(AgentStatus {
                summary: "Old".to_string(),
                task_state: Some(AgentTaskState::Completed),
                based_on_message_count: 1,
            }),
            ..session.clone()
        };
        let stale_summary = summary_for_inactive_session(&stale, false, false);
        assert!(stale_summary.summary.is_none());
        assert!(stale_summary.task_state.is_none());
    }

    #[test]
    fn latest_message_activity_uses_the_max_finite_timestamp() {
        assert_eq!(latest_message_activity_at(&[]), None);
        assert_eq!(
            latest_message_activity_at(&[1000.0, 5000.0, f64::INFINITY]),
            Some(iso_from_ms(5000.0))
        );
    }

    #[test]
    fn activity_and_lifecycle_follow_the_live_session() {
        let mut runtime = super::super::active_session_state::AgentSessionRuntime::default();
        runtime.session.session_id = "s".to_string();
        let mut state = super::super::active_session_state::ActiveSessionState::new("a", runtime);
        assert_eq!(active_lifecycle_for_session(&state), SessionLifecycle::Draft);
        assert_eq!(active_activity_for_session(&state), SessionActivity::Idle);

        state.runtime.session.messages_len = 2;
        assert_eq!(active_lifecycle_for_session(&state), SessionLifecycle::Live);
        assert_eq!(active_activity_for_session(&state), SessionActivity::Idle);
        state.summary_state = Some(super::super::active_session_state::AgentStatus {
            summary: "done".to_string(),
            task_state: Some("completed".to_string()),
            based_on_message_count: 2,
        });
        assert_eq!(active_activity_for_session(&state), SessionActivity::Idle);
        assert!(is_summary_current(&state));
        state.runtime.session.is_session_active = true;
        assert_eq!(active_activity_for_session(&state), SessionActivity::Working);
        assert!(has_live_session_work(&state));
        state.runtime.session.is_foreground_active = Some(false);
        assert_eq!(active_activity_for_session(&state), SessionActivity::Idle);
        assert!(has_live_session_work(&state), "background helper must still prevent unsafe passivation");
        state.runtime.session.is_foreground_active = Some(true);
        assert_eq!(active_activity_for_session(&state), SessionActivity::Working);
        state.runtime.session.is_foreground_active = None;
        state.runtime.session.is_session_active = false;
        state.runtime.session.has_running_rlm_children = true;
        assert!(has_live_session_work(&state));
        assert_eq!(active_activity_for_session(&state), SessionActivity::Idle);
        state.summary_state.as_mut().unwrap().based_on_message_count = 1;
        assert!(!is_summary_current(&state));
        assert_eq!(active_activity_for_session(&state), SessionActivity::Idle);
    }

    #[test]
    fn background_only_summary_keeps_liveness_and_labels_the_helper() {
        let mut runtime = super::super::active_session_state::AgentSessionRuntime::default();
        runtime.session.session_id = "helper-owner".into();
        runtime.session.is_session_active = true;
        runtime.session.is_foreground_active = Some(false);
        let state = Arc::new(StdMutex::new(ActiveSessionState::new("helper-owner", runtime)));
        let summary = summary_for_active_session(&state, None, false, false, false);
        assert!(summary.is_session_active);
        assert_eq!(summary.activity, "idle");
        assert_eq!(summary.status_label.as_deref(), Some("background helper"));
        state.lock().unwrap().runtime.session.is_foreground_active = Some(true);
        let summary = summary_for_active_session(&state, None, false, false, false);
        assert_eq!(summary.activity, "working");
        assert!(summary.status_label.is_none());
    }

    #[test]
    fn sidebar_followup_active_summary_keeps_saved_cwd_without_new_wire_fields() {
        let mut runtime = super::super::active_session_state::AgentSessionRuntime::default();
        runtime.session.session_id = "s".into();
        let state = Arc::new(StdMutex::new(ActiveSessionState::new("a", runtime)));
        let mut saved = saved_session();
        saved.cwd = r"C:\work\distinct-project\source".into();
        let summary = summary_for_active_session(&state, Some(&saved), false, false, false);
        assert_eq!(summary.cwd, saved.cwd);
        assert_eq!(serde_json::to_value(&summary).unwrap()["cwd"], saved.cwd);
        assert!(summary_for_active_session(&state, None, false, false, false).cwd.is_empty());
    }

    #[test]
    fn subagent_sessions_are_live_even_when_empty() {
        let mut runtime = super::super::active_session_state::AgentSessionRuntime::default();
        runtime.session.session_id = "s".to_string();
        runtime.metadata = Some(super::super::active_session_state::AgentSessionRuntimeMetadata {
            kind: Some("subagent".to_string()),
            prompt: Some("spawn prompt".to_string()),
            ..Default::default()
        });
        let state = super::super::active_session_state::ActiveSessionState::new("a", runtime);
        assert_eq!(active_lifecycle_for_session(&state), SessionLifecycle::Live);
        assert_eq!(active_activity_for_session(&state), SessionActivity::Idle);
    }

    #[test]
    fn compact_rlm_text_collapses_whitespace_and_truncates() {
        assert_eq!(compact_rlm_text("  hello   world ", 120), "hello world");
        assert_eq!(compact_rlm_text("abcdef", 3), "abc…");
    }
}
