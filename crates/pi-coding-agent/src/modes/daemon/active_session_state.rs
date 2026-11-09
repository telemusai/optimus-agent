//! Port of packages/coding-agent/src/modes/daemon/active-session-state.ts
//!
//! `AgentSessionRuntime` (core/agent-session-runtime.ts) and `AgentStatus`
//! (core/session-manager.ts) belong to other slices. The daemon keeps the
//! minimal structural views it needs here; they carry the same field names as
//! the TypeScript so the full types can drop in later.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::Value;

use super::daemon_session_id::{format_session_display_id, matches_session_id_suffix};

/// Minimal view of `core/session-manager.ts`'s `AgentStatus`.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentStatus {
    pub summary: String,
    pub task_state: Option<String>,
    pub based_on_message_count: usize,
}

/// `sessionManager.appendAgentStatus` as the summarizer consumes it.
pub trait AgentStatusWriter: Send + Sync {
    fn append_agent_status(&self, status: &AgentStatus) -> Result<String, String>;
}

/// Minimal view of the live session held by an `AgentSessionRuntime`.
#[derive(Clone, Default)]
pub struct ActiveSessionRuntimeSession {
    pub session_id: String,
    pub session_name: Option<String>,
    pub session_file: Option<String>,
    /// The session's own work only (delegated child work is separate).
    pub is_session_active: bool,
    /// Optional for older/test adapters; lifecycle safety still uses is_session_active.
    pub is_foreground_active: Option<bool>,
    pub is_streaming: bool,
    pub is_compacting: bool,
    pub messages_len: usize,
    pub has_running_rlm_children: bool,
    pub rlm_depth: Option<i64>,
    /// Transcript visible to the model (daemon-session-summarizer reads it).
    pub messages: Vec<pi_agent_core::types::AgentMessage>,
    /// The in-progress assistant message of a streaming turn.
    pub streaming_message: Option<pi_agent_core::types::AgentMessage>,
    pub model_registry: Option<Arc<tokio::sync::Mutex<crate::core::model_registry::ModelRegistry>>>,
    pub settings_manager: Option<Arc<crate::core::settings_manager::SettingsManager>>,
    /// `session.model` (`daemon-session-list.ts:257`). Captured from the live
    /// `DaemonSession` at bind time because the summary builder only sees this view.
    pub model_identity: Option<crate::modes::daemon::daemon_mode::ModelIdentity>,
    /// `session.thinkingLevel` (`daemon-session-list.ts:258`).
    pub thinking_level: Option<String>,
    /// `sessionManager.getLeafId()`.
    pub leaf_id: Option<String>,
    /// `sessionManager.getLatestAgentStatus()`.
    pub latest_agent_status: Option<AgentStatus>,
    /// `sessionManager.appendAgentStatus(...)`.
    pub agent_status_writer: Option<Arc<dyn AgentStatusWriter>>,
}

impl std::fmt::Debug for ActiveSessionRuntimeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveSessionRuntimeSession")
            .field("session_id", &self.session_id)
            .field("session_name", &self.session_name)
            .field("session_file", &self.session_file)
            .field("is_session_active", &self.is_session_active)
            .field("is_foreground_active", &self.is_foreground_active)
            .field("is_streaming", &self.is_streaming)
            .field("is_compacting", &self.is_compacting)
            .field("messages_len", &self.messages_len)
            .field("has_running_rlm_children", &self.has_running_rlm_children)
            .field("rlm_depth", &self.rlm_depth)
            .field("messages", &self.messages.len())
            .field("streaming_message", &self.streaming_message.is_some())
            .field("model_registry", &self.model_registry.is_some())
            .field("settings_manager", &self.settings_manager.is_some())
            .field("leaf_id", &self.leaf_id)
            .field("latest_agent_status", &self.latest_agent_status)
            .field("agent_status_writer", &self.agent_status_writer.is_some())
            .finish()
    }
}

impl PartialEq for ActiveSessionRuntimeSession {
    fn eq(&self, other: &Self) -> bool {
        self.session_id == other.session_id
            && self.session_name == other.session_name
            && self.session_file == other.session_file
            && self.is_session_active == other.is_session_active
            && self.is_foreground_active == other.is_foreground_active
            && self.is_streaming == other.is_streaming
            && self.is_compacting == other.is_compacting
            && self.messages_len == other.messages_len
            && self.has_running_rlm_children == other.has_running_rlm_children
            && self.rlm_depth == other.rlm_depth
            && self.messages == other.messages
            && self.streaming_message == other.streaming_message
            && self.leaf_id == other.leaf_id
            && self.latest_agent_status == other.latest_agent_status
    }
}

/// Runtime metadata (`AgentSessionRuntimeMetadata` in core/agent-session-runtime.ts).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentSessionRuntimeMetadata {
    pub kind: Option<String>,
    pub prompt: Option<String>,
    pub parent_active_session_id: Option<String>,
    pub parent_session_id: Option<String>,
    pub parent_session_file: Option<String>,
    pub rlm_child_id: Option<String>,
    pub rlm_parent_node_id: Option<String>,
    pub spawn_code: Option<String>,
}

/// Minimal view of `core/agent-session-runtime.ts`'s `AgentSessionRuntime`.
#[derive(Debug, Clone, Default)]
pub struct AgentSessionRuntime {
    pub session: ActiveSessionRuntimeSession,
    pub metadata: Option<AgentSessionRuntimeMetadata>,
    pub model_fallback_message: Option<String>,
}

/// The extension-UI response shapes a dialog can resolve with.
#[derive(Debug, Clone, PartialEq)]
pub enum DaemonExtensionUIResponse {
    Value(String),
    Confirmed(bool),
    Cancelled,
}

/// One attached socket client, as far as the daemon state machine needs it.
pub struct DaemonSocketClient {
    pub id: String,
    pub attached_active_session_ids: HashSet<String>,
    /// Session events are dropped while the socket is blocked and replaced with
    /// one catch-up snapshot on drain.
    pub catchup_active_session_ids: Option<HashSet<String>>,
    /// A real runtime replacement takes precedence over an ordinary resync.
    pub catchup_purposes: Option<HashMap<String, String>>,
    pub backpressured: Option<bool>,
    pub roster_subscribed: Option<bool>,
    /// A push hit backpressure; one full-roster resync goes out on drain.
    pub roster_resync_pending: Option<bool>,
    pub authenticated: Option<bool>,
    pub authentication_role: Option<String>,
    pub transport: Option<String>,
    pub snapshot_streaming: Option<bool>,
    pub snapshot_active_session_ids: Option<HashSet<String>>,
    pub snapshot_active_session_counts: Option<HashMap<String, u32>>,
    pub detach_input: Option<Arc<dyn Fn() + Send + Sync>>,
    pub supports_extension_ui: bool,
    pub capabilities: HashSet<String>,
    pub capabilities_by_active_session_id: Option<HashMap<String, HashSet<String>>>,
}

impl DaemonSocketClient {
    pub fn new(id: impl Into<String>, supports_extension_ui: bool) -> Self {
        Self {
            id: id.into(),
            attached_active_session_ids: HashSet::new(),
            catchup_active_session_ids: None,
            catchup_purposes: None,
            backpressured: None,
            roster_subscribed: None,
            roster_resync_pending: None,
            authenticated: None,
            authentication_role: None,
            transport: None,
            snapshot_streaming: None,
            snapshot_active_session_ids: None,
            snapshot_active_session_counts: None,
            detach_input: None,
            supports_extension_ui,
            capabilities: HashSet::new(),
            capabilities_by_active_session_id: None,
        }
    }
}

pub struct ActiveSessionExtensionUiRequest {
    pub resolve: Box<dyn FnOnce(DaemonExtensionUIResponse) + Send>,
}

pub struct ActiveSessionState {
    pub active_session_id: String,
    pub runtime: AgentSessionRuntime,
    pub clients: Vec<Arc<StdMutex<DaemonSocketClient>>>,

    /// Attach snapshots in flight: reserved for passivation busyness, but not yet event recipients.
    pub pending_attaches: u64,
    pub extension_ui_requests: HashMap<String, ActiveSessionExtensionUiRequest>,
    pub event_generation: String,
    pub last_event_sequence: u64,
    /// Latest background status summary, surfaced in the agents view.
    pub summary_state: Option<AgentStatus>,
    pub unsubscribe: Option<Box<dyn FnOnce() + Send>>,
    /// Client env (e.g. herdr pane identity) bound when the runtime was created.
    pub client_env: Option<HashMap<String, String>>,
    /// True when this session was restored from a passivated RLM subagent
    /// (hydration) rather than being live from creation. Agent-message
    /// receipts consult it to distinguish delivery into a restored
    /// transcript from delivery to a live consumer; see
    /// `agent_message_requires_hydration_marker`.
    pub hydrated_from_passive: bool,
    /// True when a model change to an OCC-3-capable model arrived while the
    /// session was streaming and its activation retry is still owed: the
    /// pre-prompt path (`maybe_activate_pending_occ3`) consumes it at the
    /// next idle prompt. Cleared after that single attempt.
    pub occ3_activation_pending: bool,
}

impl ActiveSessionState {
    pub fn new(active_session_id: impl Into<String>, runtime: AgentSessionRuntime) -> Self {
        let active_session_id = active_session_id.into();
        Self {
            event_generation: active_session_id.clone(),
            active_session_id,
            runtime,
            clients: Vec::new(),
            pending_attaches: 0,
            extension_ui_requests: HashMap::new(),
            last_event_sequence: 0,
            summary_state: None,
            unsubscribe: None,
            client_env: None,
            hydrated_from_passive: false,
            occ3_activation_pending: false,
        }
    }
}

/// A monotonically increasing unique id source for `createActiveSessionId`.
static ACTIVE_SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Minimal `ActiveSessionIdIndex`: the ids the caller already holds.
pub trait ActiveSessionIdIndex {
    fn has(&self, active_session_id: &str) -> bool;
}

impl ActiveSessionIdIndex for HashSet<String> {
    fn has(&self, active_session_id: &str) -> bool {
        HashSet::contains(self, active_session_id)
    }
}

impl<F: Fn(&str) -> bool> ActiveSessionIdIndex for F {
    fn has(&self, active_session_id: &str) -> bool {
        self(active_session_id)
    }
}

/// Random UUID derived display id; retried until the index does not hold it.
pub fn create_active_session_id(existing_ids: Option<&dyn ActiveSessionIdIndex>) -> String {
    loop {
        ACTIVE_SESSION_COUNTER.fetch_add(1, Ordering::SeqCst);
        let candidate = format_session_display_id(&uuid::Uuid::new_v4().to_string());
        if !existing_ids.is_some_and(|index| index.has(&candidate)) {
            return candidate;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmbiguousActiveSessionError {
    pub message: String,
}

impl std::fmt::Display for AmbiguousActiveSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AmbiguousActiveSessionError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveActiveSessionError {
    Ambiguous(AmbiguousActiveSessionError),
    Unknown(String),
}

impl std::fmt::Display for ResolveActiveSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveActiveSessionError::Ambiguous(error) => error.fmt(f),
            ResolveActiveSessionError::Unknown(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ResolveActiveSessionError {}

/// Resolve a selector against the live active sessions: exact id, exact session
/// id/name, then unique id suffix.
pub fn resolve_active_session_state(
    sessions: &HashMap<String, Arc<StdMutex<ActiveSessionState>>>,
    selector: &str,
) -> Result<Arc<StdMutex<ActiveSessionState>>, ResolveActiveSessionError> {
    if let Some(direct) = sessions.get(selector) {
        return Ok(Arc::clone(direct));
    }

    let exact_matches = unique_states(
        sessions
            .values()
            .filter(|state| {
                let state = state.lock().expect("active session poisoned");
                state.runtime.session.session_id == selector
                    || state.runtime.session.session_name.as_deref() == Some(selector)
            })
            .cloned()
            .collect(),
    );
    if exact_matches.len() == 1 {
        return Ok(Arc::clone(&exact_matches[0]));
    }
    if exact_matches.len() > 1 {
        return Err(ResolveActiveSessionError::Ambiguous(
            AmbiguousActiveSessionError {
                message: format_ambiguous_session_error(selector, &exact_matches),
            },
        ));
    }

    let suffix_matches = unique_states(
        sessions
            .values()
            .filter(|state| {
                let state = state.lock().expect("active session poisoned");
                matches_session_id_suffix(&state.active_session_id, selector)
                    || matches_session_id_suffix(&state.runtime.session.session_id, selector)
            })
            .cloned()
            .collect(),
    );
    if suffix_matches.len() == 1 {
        return Ok(Arc::clone(&suffix_matches[0]));
    }
    if suffix_matches.len() > 1 {
        return Err(ResolveActiveSessionError::Ambiguous(
            AmbiguousActiveSessionError {
                message: format_ambiguous_session_error(selector, &suffix_matches),
            },
        ));
    }

    Err(ResolveActiveSessionError::Unknown(format!(
        "Unknown active session: {selector}"
    )))
}

fn unique_states(states: Vec<Arc<StdMutex<ActiveSessionState>>>) -> Vec<Arc<StdMutex<ActiveSessionState>>> {
    let mut unique: Vec<Arc<StdMutex<ActiveSessionState>>> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for state in states {
        let active_session_id = state
            .lock()
            .expect("active session poisoned")
            .active_session_id
            .clone();
        if seen.insert(active_session_id) {
            unique.push(state);
        }
    }
    unique
}

fn format_ambiguous_session_error(selector: &str, states: &[Arc<StdMutex<ActiveSessionState>>]) -> String {
    let matches: Vec<String> = states.iter().map(|state| format_session_match(state)).collect();
    format!("Ambiguous active session \"{selector}\": matches {}", matches.join(", "))
}

fn format_session_match(state: &Arc<StdMutex<ActiveSessionState>>) -> String {
    let state = state.lock().expect("active session poisoned");
    let name = state
        .runtime
        .session
        .session_name
        .as_ref()
        .map(|name| format!(" ({name})"))
        .unwrap_or_default();
    format!(
        "{}/{}{}",
        state.active_session_id, state.runtime.session.session_id, name
    )
}

/// The extension-UI payload field names the daemon puts on the wire.
pub fn extension_ui_payload_fields(payload: &Value) -> Vec<String> {
    payload
        .as_object()
        .map(|object| object.keys().cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(active_session_id: &str, session_id: &str, name: Option<&str>) -> Arc<StdMutex<ActiveSessionState>> {
        let mut runtime = AgentSessionRuntime::default();
        runtime.session.session_id = session_id.to_string();
        runtime.session.session_name = name.map(str::to_string);
        Arc::new(StdMutex::new(ActiveSessionState::new(active_session_id, runtime)))
    }

    #[test]
    fn resolves_by_active_id_session_id_and_suffix() {
        let mut sessions: HashMap<String, Arc<StdMutex<ActiveSessionState>>> = HashMap::new();
        let one = state("aabbccddeeff", "001122334455", Some("alpha"));
        sessions.insert("aabbccddeeff".to_string(), Arc::clone(&one));

        assert!(Arc::ptr_eq(
            &resolve_active_session_state(&sessions, "aabbccddeeff").expect("direct"),
            &one
        ));
        assert!(Arc::ptr_eq(
            &resolve_active_session_state(&sessions, "001122334455").expect("session id"),
            &one
        ));
        assert!(Arc::ptr_eq(
            &resolve_active_session_state(&sessions, "alpha").expect("session name"),
            &one
        ));
        assert!(Arc::ptr_eq(
            &resolve_active_session_state(&sessions, "eeff").expect("suffix"),
            &one
        ));
        assert!(resolve_active_session_state(&sessions, "aabb").is_err());
        // `expect_err` would require `ActiveSessionState: Debug`; the TypeScript
        // asserts on the thrown message only.
        let error = resolve_active_session_state(&sessions, "zzz")
            .err()
            .expect("unknown");
        assert_eq!(error.to_string(), "Unknown active session: zzz");
    }

    #[test]
    fn ambiguous_suffixes_are_reported() {
        let mut sessions: HashMap<String, Arc<StdMutex<ActiveSessionState>>> = HashMap::new();
        sessions.insert("11aa".to_string(), state("11aa", "s1", None));
        sessions.insert("22aa".to_string(), state("22aa", "s2", None));
        let error = resolve_active_session_state(&sessions, "aa")
            .err()
            .expect("ambiguous");
        assert!(error
            .to_string()
            .starts_with("Ambiguous active session \"aa\": matches "));
    }

    #[test]
    fn generated_active_session_ids_avoid_existing_ids() {
        let id = create_active_session_id(None);
        assert_eq!(id.len(), 12);
        let index: HashSet<String> = [id.clone()].into_iter().collect();
        let next = create_active_session_id(Some(&index));
        assert_ne!(id, next);
    }

    #[test]
    fn active_session_state_starts_with_the_session_generation() {
        let session = state("abc", "def", None);
        let session = session.lock().expect("poisoned");
        assert_eq!(session.event_generation, "abc");
        assert_eq!(session.last_event_sequence, 0);
        assert!(session.clients.is_empty());
    }
}
