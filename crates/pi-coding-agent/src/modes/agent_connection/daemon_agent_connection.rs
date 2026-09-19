//! Port of packages/coding-agent/src/modes/agent-connection/daemon-agent-connection.ts
//!
//! AgentConnection adapter for the local daemon JSONL socket transport.
//! InteractiveMode depends only on AgentConnection; local socket ownership and
//! daemon command details stay inside this adapter.
//!
//! The daemon transport slice (`DaemonTransportClient`, `DaemonRoutedClient`,
//! `DaemonCommand` bodies, the roster store, and the saved-session/heartbeat
//! catalogs) has not landed yet. This module therefore defines the exact
//! transport seam it consumes as a trait, keeps every constant, error string,
//! cursor rule, snapshot-assembly rule and capability check from the
//! TypeScript, and records the cross-slice needs in `blocked_on`.

#[path = "roster_subscription.rs"]
mod roster_subscription;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{BoxFuture, ImageContent, ServiceTier, Transport};

use crate::modes::agent_connection::types::*;
use crate::modes::daemon::daemon_protocol::DaemonClosingReason;

/// Extended request timeout for refine requests, which run an LLM pass.
pub const DAEMON_REFINE_REQUEST_TIMEOUT_MS: u64 = 10 * 60 * 1000;
const DAEMON_LONG_RUNNING_REQUEST_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
pub const DAEMON_RECONNECT_TIMEOUT_MS: u64 = 60_000;
pub const DAEMON_SNAPSHOT_TIMEOUT_MS: u64 = 30_000;
const MAX_IGNORED_SNAPSHOT_IDS: usize = 128;
const UPDATE_RECONNECT_TIMEOUT_MS: u64 = 120000;
const UPDATE_RECONNECT_RETRY_MS: u64 = 100;
const MAX_COMPLETED_SNAPSHOTS: usize = 128;
const OWNED_SESSION_DISPOSE_RECONNECT_WAIT_MS: u64 = 10_000;

/// `DaemonCapabilityUnavailableError` (modes/daemon/daemon-client.ts).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct DaemonCapabilityUnavailableError {
    pub message: String,
    pub command: String,
    pub capability: String,
    pub after_reconnect: bool,
}

impl DaemonCapabilityUnavailableError {
    pub fn new(command: impl Into<String>, capability: impl Into<String>) -> Self {
        let command = command.into();
        let capability = capability.into();
        Self {
            message: format!("The daemon does not support {command} (missing capability: {capability})"),
            command,
            capability,
            after_reconnect: false,
        }
    }
}

/// `DaemonEventCursor` / `DaemonReplayInfo` / snapshot + attach result shapes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DaemonEventCursor {
    pub generation: String,
    pub sequence: i64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DaemonReplayInfo {
    pub status: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DaemonSessionSummary {
    pub session_id: String,
    pub session_file: Option<String>,
    pub active_session_id: Option<String>,
    pub id: Option<String>,
    pub streaming_message: Option<AgentMessage>,
    pub last_event_sequence: Option<i64>,
    pub last_event_cursor: Option<DaemonEventCursor>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DaemonSessionSnapshot {
    pub state: AgentConnectionState,
    pub messages: Vec<AgentMessage>,
    pub summary: DaemonSessionSummary,
    pub history: Option<AgentConnectionHistoryWindow>,
    pub session_context: Option<AgentConnectionSessionContext>,
    pub session_tree: Option<AgentConnectionSessionTree>,
    pub parent: Option<AgentConnectionParentMetadata>,
    pub children: Option<Vec<AgentConnectionRlmChildAgentSnapshot>>,
    pub last_event_sequence: Option<i64>,
    pub last_event_cursor: Option<DaemonEventCursor>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DaemonAttachResult {
    pub active_session_id: String,
    pub snapshot: DaemonSessionSnapshot,
    pub snapshot_stream: Option<DaemonSnapshotStream>,
    pub replay: Option<DaemonReplayInfo>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DaemonSnapshotStream {
    pub id: String,
}

/// `DaemonOutbound` messages this adapter dispatches.
#[derive(Debug, Clone, PartialEq)]
pub enum DaemonOutbound {
    HeartbeatsChanged {
        active_session_id: Option<String>,
        meta: Option<DaemonEventMeta>,
    },
    SessionEvent {
        active_session_id: String,
        event: AgentConnectionSessionEvent,
        meta: Option<DaemonEventMeta>,
    },
    SideQuestionEvent {
        active_session_id: String,
        event: AgentConnectionSideQuestionEvent,
        meta: Option<DaemonEventMeta>,
    },
    SessionStatus {
        active_session_id: String,
        recap: Option<String>,
        meta: Option<DaemonEventMeta>,
    },
    SessionResynced {
        active_session_id: String,
        snapshot: DaemonSessionSnapshot,
        meta: Option<DaemonEventMeta>,
    },
    SessionReplaced {
        active_session_id: String,
        state: AgentConnectionState,
        messages: Vec<AgentMessage>,
        snapshot_follows: Option<bool>,
        meta: Option<DaemonEventMeta>,
    },
    ExtensionUiRequest {
        active_session_id: String,
        id: String,
        method: String,
        payload: Value,
        meta: Option<DaemonEventMeta>,
    },
    ExtensionError {
        active_session_id: String,
        extension_path: String,
        event: String,
        error: String,
        meta: Option<DaemonEventMeta>,
    },
    SessionClosed {
        active_session_id: String,
        reason: String,
        meta: Option<DaemonEventMeta>,
    },
    SessionSnapshotBegin {
        active_session_id: String,
        snapshot_id: String,
        snapshot: DaemonSessionSnapshot,
        message_count: usize,
        purpose: Option<String>,
    },
    SessionSnapshotChunk {
        active_session_id: String,
        snapshot_id: String,
        index: usize,
        messages: Vec<AgentMessage>,
    },
    SessionSnapshotEnd {
        active_session_id: String,
        snapshot_id: String,
        chunk_count: usize,
        last_event_sequence: i64,
        last_event_cursor: Option<DaemonEventCursor>,
    },
    SessionSnapshotFailed {
        active_session_id: String,
        snapshot_id: String,
        error: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DaemonEventMeta {
    pub sequence: Option<i64>,
    pub cursor: Option<DaemonEventCursor>,
}

impl DaemonOutbound {
    pub fn type_name(&self) -> &'static str {
        match self {
            DaemonOutbound::HeartbeatsChanged { .. } => "heartbeats_changed",
            DaemonOutbound::SessionEvent { .. } => "session_event",
            DaemonOutbound::SideQuestionEvent { .. } => "side_question_event",
            DaemonOutbound::SessionStatus { .. } => "session_status",
            DaemonOutbound::SessionResynced { .. } => "session_resynced",
            DaemonOutbound::SessionReplaced { .. } => "session_replaced",
            DaemonOutbound::ExtensionUiRequest { .. } => "extension_ui_request",
            DaemonOutbound::ExtensionError { .. } => "extension_error",
            DaemonOutbound::SessionClosed { .. } => "session_closed",
            DaemonOutbound::SessionSnapshotBegin { .. } => "session_snapshot_begin",
            DaemonOutbound::SessionSnapshotChunk { .. } => "session_snapshot_chunk",
            DaemonOutbound::SessionSnapshotEnd { .. } => "session_snapshot_end",
            DaemonOutbound::SessionSnapshotFailed { .. } => "session_snapshot_failed",
        }
    }

    fn active_session_id(&self) -> Option<&str> {
        match self {
            DaemonOutbound::HeartbeatsChanged { active_session_id, .. } => active_session_id.as_deref(),
            DaemonOutbound::SessionEvent { active_session_id, .. }
            | DaemonOutbound::SideQuestionEvent { active_session_id, .. }
            | DaemonOutbound::SessionStatus { active_session_id, .. }
            | DaemonOutbound::SessionResynced { active_session_id, .. }
            | DaemonOutbound::SessionReplaced { active_session_id, .. }
            | DaemonOutbound::ExtensionUiRequest { active_session_id, .. }
            | DaemonOutbound::ExtensionError { active_session_id, .. }
            | DaemonOutbound::SessionClosed { active_session_id, .. }
            | DaemonOutbound::SessionSnapshotBegin { active_session_id, .. }
            | DaemonOutbound::SessionSnapshotChunk { active_session_id, .. }
            | DaemonOutbound::SessionSnapshotEnd { active_session_id, .. }
            | DaemonOutbound::SessionSnapshotFailed { active_session_id, .. } => Some(active_session_id),
        }
    }

    fn meta(&self) -> Option<&DaemonEventMeta> {
        match self {
            DaemonOutbound::HeartbeatsChanged { meta, .. }
            | DaemonOutbound::SessionEvent { meta, .. }
            | DaemonOutbound::SideQuestionEvent { meta, .. }
            | DaemonOutbound::SessionStatus { meta, .. }
            | DaemonOutbound::SessionResynced { meta, .. }
            | DaemonOutbound::SessionReplaced { meta, .. }
            | DaemonOutbound::ExtensionUiRequest { meta, .. }
            | DaemonOutbound::ExtensionError { meta, .. }
            | DaemonOutbound::SessionClosed { meta, .. } => meta.as_ref(),
            _ => None,
        }
    }

    fn snapshot_id(&self) -> Option<&str> {
        match self {
            DaemonOutbound::SessionSnapshotBegin { snapshot_id, .. }
            | DaemonOutbound::SessionSnapshotChunk { snapshot_id, .. }
            | DaemonOutbound::SessionSnapshotEnd { snapshot_id, .. }
            | DaemonOutbound::SessionSnapshotFailed { snapshot_id, .. } => Some(snapshot_id),
            _ => None,
        }
    }
}

/// Response envelope returned by `DaemonTransportClient.request`.
#[derive(Debug, Clone, PartialEq)]
pub struct DaemonResponse {
    pub success: bool,
    pub data: Value,
    pub error: Option<String>,
    pub error_code: Option<String>,
}

impl DaemonResponse {
    pub fn ok(data: Value) -> Self {
        Self {
            success: true,
            data,
            error: None,
            error_code: None,
        }
    }

    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            success: false,
            data: Value::Null,
            error: Some(error.into()),
            error_code: None,
        }
    }
}

/// `DaemonTransportClient` seam consumed by this adapter.
pub trait DaemonTransportClient: Send + Sync {
    fn request(&self, command: Value, timeout_ms: Option<u64>) -> BoxFuture<Result<DaemonResponse, String>>;
    /// `client.request(command, timeoutMs, { recoverable })` - daemon-client.ts:130-134
    /// with the third argument this slice supplies.
    ///
    /// `recoverable: false` opts out of reconnect parking (daemon-client.ts:39-45):
    /// a socket close must reject the request so the caller's own bounded retry
    /// loop stays live, and the request must never be parked and replayed behind a
    /// hello. The connection slice's `requestData` passes that third argument
    /// (daemon-agent-connection.ts:1735-1740), so this seam has to carry it.
    ///
    /// GAP, owed by the two bridge impls outside this file
    /// (`main_entry.rs:1803` `MainEntryDaemonTransport`,
    /// `modes/telegram/worker.rs:438` `TelegramDaemonTransport`): their `request`
    /// forwards only `command`/`timeout_ms` and hardcodes
    /// `DaemonClientRequestOptions::default()`, whose `recoverable: None` resolves
    /// to `true` in `daemon_client.rs:1037` (park). Those impls therefore do not
    /// override this method yet, and the default below keeps their existing parked
    /// behaviour instead of inventing a new one. The override they owe is
    /// `DaemonClientRequestOptions { recoverable: Some(recoverable), .. }` handed to
    /// `DaemonClient::request` (`daemon_client.rs:903-908`).
    fn request_with_recoverable(
        &self,
        command: Value,
        timeout_ms: Option<u64>,
        _recoverable: bool,
    ) -> BoxFuture<Result<DaemonResponse, String>> {
        self.request(command, timeout_ms)
    }
    fn on_message(&self, listener: Arc<dyn Fn(DaemonOutbound) + Send + Sync>) -> Box<dyn Fn() + Send + Sync>;
    fn on_close(&self, listener: Arc<dyn Fn(String) + Send + Sync>) -> Box<dyn Fn() + Send + Sync>;
    fn supports_server_capability(&self, capability: &str) -> bool;
    fn hello_socket_path(&self) -> Option<String>;
    fn is_connected(&self) -> bool;
    fn enable_request_recovery(&self);
    fn close(&self);
    fn connect(&self, timeout_ms: u64) -> BoxFuture<Result<(), String>>;
    fn wait_for_hello(&self, timeout_ms: u64) -> BoxFuture<Result<(), String>>;
    fn reconnect(&self, timeout_ms: u64) -> BoxFuture<Result<(), String>>;
    fn disconnect_for_reconnect(&self, reason: &str);
    fn reset_transport_for_reconnect(&self);
    fn has_direct_transport(&self) -> bool {
        false
    }
    fn is_control_plane_ready(&self) -> bool {
        true
    }
    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn DaemonTransportClient>;
}

/// `getDaemonSocketCloseReason(error)` (`daemon-client.ts:103-105`).
///
/// TS returns the reason from a *typed* carrier: `error instanceof DaemonSocketClosedError ?
/// error.daemonClosingReason : undefined`. The one wire-visible form of that field is
/// `DaemonSocketClosedError`'s message template (`daemon-client.ts:73-86`):
/// `Connection to the Prime Agent daemon closed.{ Reason: <reason>.}{ Cause: <cause>.} Socket: ...`,
/// which is what `getDaemonSocketCloseReason`'s consumers actually receive, because
/// `daemon-agent-connection.ts` and `daemon-routed-client.ts:27` hand it the close *listener's*
/// `Error`. The only producer of this string on the Rust side is `DaemonSocketClosedError::message`
/// (`daemon_client.rs:235-250`), which formats the reason through `format!("{reason}")` as well.
///
/// The port previously matched the bare `"daemon socket closed: <reason>"` form, which no producer
/// on either side ever writes; every real close therefore classified as `None`. For
/// `daemon-agent-connection.ts:316` that means an authoritative shutdown was treated as transient
/// (it reconnects instead of emitting the terminal closed event), and for
/// `daemon-routed-client.ts:27` the closing reason never reached `DaemonDirectTransportClosedError`.
///
/// The reason is matched in its template position (`closed. Reason: <reason>.`), which is the text
/// analogue of reading the typed field; a `Cause:` body cannot inject a reason because the template
/// writes `closed.` immediately before the reason segment and `Cause:` before the cause.
pub fn get_daemon_socket_close_reason(error: &str) -> Option<DaemonClosingReason> {
    [DaemonClosingReason::Shutdown, DaemonClosingReason::Update]
        .into_iter()
        .find(|reason| error.contains(&format!("closed. Reason: {}.", reason.as_str())))
}

/// `getDaemonLogPath(socketPath)`.
pub fn get_daemon_log_path(socket_path: &str) -> String {
    format!("{socket_path}.log")
}

/// `getAgentLogPath()`.
pub fn get_agent_log_path() -> String {
    std::env::var("PRIME_AGENT_LOG_PATH").unwrap_or_else(|_| "agent.log".to_string())
}

/// `appendRotatingLog(path, line)`.
pub fn append_rotating_log(path: &str, line: &str) {
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

pub fn now_iso() -> String {
    let millis = now_ms();
    let seconds = millis.div_euclid(1000);
    let sub = millis.rem_euclid(1000);
    let days = seconds.div_euclid(86_400);
    let time_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{sub:03}Z",
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60
    )
}

/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn format_error_sentence(error: &str) -> String {
    let message = error.trim();
    if message.is_empty() {
        return "Unknown daemon error.".to_string();
    }
    let last = message.chars().last().unwrap_or('.');
    if last == '.' || last == '!' || last == '?' {
        message.to_string()
    } else {
        format!("{message}.")
    }
}

/// `updateTransportReconnects` weak map, kept as a per-client single-flight slot.
/// Clears `reconnect_in_flight` and wakes `dispose` when the recovery attempt ends,
/// mirroring the TypeScript promise settling (`daemon-agent-connection.ts:1643-1646`).
struct ReconnectAttempt {
    result: tokio::sync::watch::Sender<Option<Result<(), String>>>,
    cancel: tokio_util::sync::CancellationToken,
}

struct ReconnectScope {
    slot: Arc<Mutex<Option<Arc<ReconnectAttempt>>>>,
    attempt: Arc<ReconnectAttempt>,
}
impl Drop for ReconnectScope {
    fn drop(&mut self) {
        if self.attempt.result.borrow().is_none() {
            self.attempt.result.send_replace(Some(Err("Daemon reconnect cancelled".to_string())));
        }
        let mut slot = self.slot.lock().unwrap();
        if slot.as_ref().is_some_and(|current| Arc::ptr_eq(current, &self.attempt)) {
            slot.take();
        }
    }
}

struct AttachScope(Option<Arc<DaemonAgentConnection>>);
impl Drop for AttachScope {
    fn drop(&mut self) {
        if let Some(connection) = self.0.take() {
            // Cancelling navigation must release only this viewer, never its session.
            tokio::spawn(async move { connection.dispose_inner().await; });
        }
    }
}

fn reconnect_daemon_transport_after_update(client: Arc<dyn DaemonTransportClient>) -> BoxFuture<Result<(), String>> {
    Box::pin(async move {
        client.disconnect_for_reconnect("update");
        let deadline = now_ms() + UPDATE_RECONNECT_TIMEOUT_MS as i64;
        let mut last_error = "the updated daemon did not become available".to_string();
        while now_ms() < deadline {
            match client.reconnect(1000).await {
                Ok(()) => return Ok(()),
                Err(error) => last_error = error,
            }
            tokio::time::sleep(Duration::from_millis(UPDATE_RECONNECT_RETRY_MS)).await;
        }
        Err(last_error)
    })
}

#[derive(Debug, Clone, Default)]
pub struct DaemonAgentConnectionOptions {
    /// Interactive viewers release buffered updates when their first listener is installed.
    pub defer_session_events: bool,
    pub close_client_on_dispose: bool,
    /// Secondary watchers pass false to stay on the shared control-plane socket.
    pub direct_transport: bool,
    /// Restart/probe the detached supervisor after a transient socket loss.
    pub recover_daemon: bool,
    /// Bound supervisor recovery before surfacing a fatal connection error.
    pub reconnect_timeout_ms: Option<u64>,
    /// Bound an incomplete streamed snapshot before failing the attach or resync.
    pub snapshot_timeout_ms: Option<u64>,
    /// Send this client's allowlisted env with attach.
    pub send_client_env: bool,
    /// Advertise support for interactive extension dialogs.
    pub supports_extension_ui: bool,
    /// Dispose the connection by stopping its hidden worker instead of detaching.
    pub owned_session: bool,
    /// Fresh runtime context used only if the owned worker must be relaunched.
    pub owned_session_recovery_config: Option<Value>,
    /// Require the target worker to have been created with telemetry disabled.
    pub telemetry_disabled: bool,
}

/// `buildSessionTreeFromFlatNodes(flatNodes)`.
pub fn build_session_tree_from_flat_nodes(
    flat_nodes: &[AgentConnectionSessionTreeFlatNode],
) -> Vec<AgentConnectionSessionTreeNode> {
    let mut by_id = IndexMap::new();
    for flat_node in flat_nodes {
        by_id.insert(flat_node.entry.id(), flat_node);
    }
    let mut children = vec![Vec::new(); by_id.len()];
    let mut roots = Vec::new();
    for flat_node in flat_nodes {
        let entry = &flat_node.entry;
        let index = by_id.get_index_of(entry.id()).expect("flat node was indexed");
        let parent = entry.parent_id()
            .filter(|parent| *parent != entry.id())
            .and_then(|parent| by_id.get_index_of(parent));
        match parent {
            Some(parent) => children[parent].push(index),
            None => roots.push(index),
        }
    }
    for siblings in &mut children {
        siblings.sort_by_key(|index| {
            timestamp_millis(by_id.get_index(*index).unwrap().1.entry.timestamp())
        });
    }
    let make_node = |index| {
        let flat_node = by_id.get_index(index).unwrap().1;
        AgentConnectionSessionTreeNode {
            entry: flat_node.entry.clone(),
            label: flat_node.label.clone(),
            label_timestamp: flat_node.label_timestamp.clone(),
            children: Vec::new(),
        }
    };
    // JS links shared objects; Rust must finish each owned child before moving it
    // into its parent. Use explicit frames to preserve deep-chain behavior.
    let mut tree = Vec::with_capacity(roots.len());
    for root in roots {
        let mut stack = vec![(root, 0, make_node(root))];
        while let Some((index, next_child, _)) = stack.last_mut() {
            if let Some(child) = children[*index].get(*next_child).copied() {
                *next_child += 1;
                stack.push((child, 0, make_node(child)));
            } else {
                let (_, _, node) = stack.pop().unwrap();
                match stack.last_mut() {
                    Some((_, _, parent)) => parent.children.push(node),
                    None => tree.push(node),
                }
            }
        }
    }
    tree
}

/// `new Date(timestamp).getTime()` for an ISO-8601 timestamp string.
fn timestamp_millis(timestamp: &str) -> i64 {
    match chrono::DateTime::parse_from_rfc3339(timestamp) {
        Ok(parsed) => parsed.timestamp_millis(),
        Err(_) => 0,
    }
}

fn parse_session_tree_response(data: Value) -> Result<AgentConnectionWatchSessionTree, String> {
    let nodes = data.get("flatNodes").cloned()
        .ok_or_else(|| "Daemon returned an invalid session tree: missing flatNodes".to_string())?;
    let flat_nodes: Vec<AgentConnectionSessionTreeFlatNode> = serde_json::from_value(nodes)
        .map_err(|error| format!("Daemon returned an invalid session tree: {error}"))?;
    Ok(AgentConnectionWatchSessionTree {
        tree: build_session_tree_from_flat_nodes(&flat_nodes),
        leaf_id: data.get("leafId").and_then(Value::as_str).map(str::to_string),
    })
}

/// `readSessionSummaries(value)`.
pub fn read_session_summaries(value: &Value) -> Result<Vec<DaemonSessionSummary>, String> {
    let sessions = value.get("sessions").and_then(Value::as_array);
    match sessions {
        Some(sessions) => Ok(sessions
            .iter()
            .map(|item| DaemonSessionSummary {
                session_id: item
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                session_file: item.get("sessionFile").and_then(Value::as_str).map(str::to_string),
                active_session_id: item.get("activeSessionId").and_then(Value::as_str).map(str::to_string),
                id: item.get("id").and_then(Value::as_str).map(str::to_string),
                streaming_message: None,
                last_event_sequence: item.get("lastEventSequence").and_then(Value::as_i64),
                last_event_cursor: None,
            })
            .collect()),
        None => Err("Daemon returned an invalid session list response".to_string()),
    }
}

/// `getAttachActiveSessionId(result)`.
pub fn get_attach_active_session_id(result: &DaemonAttachResult) -> String {
    result.active_session_id.clone()
}

/// `maxEventSequence(current, observed)`.
pub fn max_event_sequence(current: Option<i64>, observed: Option<i64>) -> Option<i64> {
    match (current, observed) {
        (None, observed) => observed,
        (current, None) => current,
        (Some(current), Some(observed)) => Some(current.max(observed)),
    }
}

/// `mapDaemonSessionSnapshot(snapshot, replay?)`.
pub fn map_daemon_session_snapshot(
    snapshot: &DaemonSessionSnapshot,
    replay: Option<&DaemonReplayInfo>,
) -> Result<AgentConnectionSnapshot, String> {
    if let Some(history) = &snapshot.history {
        let entry_ids_unique = {
            let mut seen: HashSet<&String> = HashSet::new();
            history.entry_ids.iter().all(|entry_id| seen.insert(entry_id))
        };
        let invalid = history.version != 1.0
            || history.order != "chronological"
            || history.generation.is_empty() && false
            || history.representation.is_empty()
            || (snapshot.last_event_cursor.is_some()
                && Some(&history.generation) != snapshot.last_event_cursor.as_ref().map(|cursor| &cursor.generation))
            || history.entry_ids.len() != snapshot.messages.len()
            || !entry_ids_unique
            || !is_safe_integer(history.start_index)
            || history.start_index < 0.0
            || !is_safe_integer(history.total_message_count)
            || history.total_message_count < 0.0
            || history.start_index + snapshot.messages.len() as f64 != history.total_message_count
            || history.has_older != (history.start_index > 0.0);
        if invalid {
            return Err("Daemon returned an invalid recent-first history snapshot".to_string());
        }
    }
    let mut connection_snapshot = AgentConnectionSnapshot {
        state: snapshot.state.clone(),
        messages: snapshot.messages.clone(),
        history: snapshot.history.clone(),
        streaming_message: snapshot.summary.streaming_message.clone(),
        session_context: snapshot.session_context.clone(),
        session_tree: snapshot.session_tree.clone(),
        parent: snapshot.parent.clone(),
        children: snapshot.children.clone(),
        last_event_sequence: snapshot.last_event_sequence.map(|value| value as f64),
        last_event_cursor: snapshot.last_event_cursor.as_ref().map(|cursor| AgentConnectionEventCursor {
            generation: cursor.generation.clone(),
            sequence: cursor.sequence as f64,
        }),
        replay: None,
    };
    if snapshot.session_context.is_none() {
        connection_snapshot.session_context = None;
    }
    if snapshot.session_tree.is_none() {
        connection_snapshot.session_tree = None;
    }
    if snapshot.parent.is_none() {
        connection_snapshot.parent = None;
    }
    if snapshot.children.is_none() {
        connection_snapshot.children = None;
    }
    if let Some(replay) = replay {
        connection_snapshot.replay = Some(AgentConnectionReplayInfo {
            status: replay.status.clone(),
            from_sequence: None,
            to_sequence: snapshot.last_event_sequence.unwrap_or(0) as f64,
            from_cursor: None,
            to_cursor: None,
            reason: None,
        });
    }
    Ok(connection_snapshot)
}

fn is_safe_integer(value: f64) -> bool {
    value.is_finite() && value.fract() == 0.0 && value.abs() <= 9_007_199_254_740_991.0
}

fn get_daemon_message_sequence(message: &DaemonOutbound) -> Option<i64> {
    message.meta().and_then(|meta| meta.sequence)
}

fn get_daemon_message_cursor(message: &DaemonOutbound) -> Option<DaemonEventCursor> {
    message.meta().and_then(|meta| meta.cursor.clone())
}

/// `invalidatesCachedSnapshot(commandType)`.
pub fn invalidates_cached_snapshot(command_type: &str) -> bool {
    !matches!(
        command_type,
        "attach"
            | "reattach"
            | "detach"
            | "list"
            | "list_saved_sessions"
            | "wait_for_idle"
            | "get_state"
            | "jev_get_status"
            | "get_connection_state"
            | "get_messages"
            | "get_history_range"
            | "get_session_stats"
            | "get_commands"
            | "get_resource_snapshot"
            | "get_model_catalog"
            | "get_available_models"
            | "get_queue"
            | "cron_list"
            | "heartbeats_list"
            | "get_session_context"
            | "get_session_tree"
            | "get_user_messages_for_forking"
            | "get_last_assistant_text"
            | "get_system_prompt"
            | "get_tool_definition"
            | "start_side_question"
            | "abort_side_question"
            | "export_html"
            | "export_jsonl"
    )
}

/// `collectDaemonClientEnv()` - only the herdr pane identity is allowlisted.
pub fn collect_daemon_client_env() -> IndexMap<String, String> {
    let mut env = IndexMap::new();
    for key in ["HERDR_PANE_ID", "HERDR_SESSION_ID", "HERDR_TAB_ID"] {
        if let Ok(value) = std::env::var(key) {
            env.insert(key.to_string(), value);
        }
    }
    env
}

/// `collectDaemonLaunchEnv()`.
pub fn collect_daemon_launch_env() -> IndexMap<String, String> {
    let mut env = IndexMap::new();
    for key in ["PRIME_AGENT_KERNEL_PYTHON", "PRIME_AGENT_KERNEL_VENV"] {
        if let Ok(value) = std::env::var(key) {
            env.insert(key.to_string(), value);
        }
    }
    env
}

/// `isUnknownDaemonCommandError(error, commandType)`.
pub fn is_unknown_daemon_command_error(error: &str, command_type: &str) -> bool {
    // The daemon's own message is `Unknown daemon command: {command}`
    // (`daemon_mode.rs`, `daemon-protocol.ts:isUnknownDaemonCommandError`), while older
    // transports surface the shorter `unknown command: {command}`. Accept both, exactly
    // like the agents-view matcher (`agents_view_mode.rs`) does for its own call sites.
    error.contains(&format!("unknown command: {command_type}"))
        || (error.contains("Unknown daemon command") && error.contains(command_type))
}

fn command_body(type_: &str, fields: Vec<(&str, Value)>) -> Value {
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String(type_.to_string()));
    for (key, value) in fields {
        if !value.is_null() {
            object.insert(key.to_string(), value);
        }
    }
    Value::Object(object)
}

fn optional_string(value: Option<&str>) -> Value {
    match value {
        Some(value) => Value::String(value.to_string()),
        None => Value::Null,
    }
}

fn optional_bool(value: Option<bool>) -> Value {
    match value {
        Some(value) => Value::Bool(value),
        None => Value::Null,
    }
}


/// `DaemonSnapshotAssembly`.
#[derive(Default)]
struct DaemonSnapshotAssembly {
    begin: Option<SnapshotBegin>,
    chunks: HashMap<usize, Vec<AgentMessage>>,
    timeout_armed: bool,
    completed: bool,
    failed: Option<String>,
}

#[derive(Default)]
struct DeferredSessionEvents {
    queue: VecDeque<(usize, DaemonOutbound)>,
    bytes: usize,
    draining: bool,
    failure: Option<String>,
}

impl DeferredSessionEvents {
    fn push(&mut self, message: DaemonOutbound) -> bool {
        let size = match &message {
            DaemonOutbound::SessionEvent { event, .. } => serde_json::to_vec(event).map(|bytes| bytes.len()).unwrap_or(0),
            DaemonOutbound::ExtensionUiRequest { payload, .. } => serde_json::to_vec(payload).map(|bytes| bytes.len()).unwrap_or(0),
            DaemonOutbound::SessionResynced { snapshot, .. } => snapshot.messages.iter().map(|message| serde_json::to_vec(message).map(|bytes| bytes.len()).unwrap_or(0)).sum(),
            DaemonOutbound::SessionReplaced { messages, .. } => messages.iter().map(|message| serde_json::to_vec(message).map(|bytes| bytes.len()).unwrap_or(0)).sum(),
            _ => 1024,
        };
        let update = matches!(&message, DaemonOutbound::SessionEvent { event: AgentConnectionSessionEvent::MessageUpdate { .. }, .. });
        if update && self.queue.back().is_some_and(|(_, previous)| matches!(previous, DaemonOutbound::SessionEvent { event: AgentConnectionSessionEvent::MessageUpdate { .. }, .. })) {
            if let Some((previous_size, _)) = self.queue.pop_back() { self.bytes = self.bytes.saturating_sub(previous_size); }
        }
        if self.queue.len() >= 4096 || self.bytes.saturating_add(size) > 8 * 1024 * 1024 { return false; }
        self.bytes += size;
        self.queue.push_back((size, message));
        true
    }
    fn pop(&mut self) -> Option<DaemonOutbound> {
        let (size, message) = self.queue.pop_front()?;
        self.bytes = self.bytes.saturating_sub(size);
        Some(message)
    }
    fn clear(&mut self) { self.queue.clear(); self.bytes = 0; }
}

#[derive(Clone)]
struct SnapshotBegin {
    snapshot: DaemonSessionSnapshot,
    message_count: usize,
    purpose: String,
}

impl Default for SnapshotBegin {
    fn default() -> Self {
        Self {
            snapshot: DaemonSessionSnapshot::default(),
            message_count: 0,
            purpose: "attach".to_string(),
        }
    }
}

/// `DaemonAgentConnection`.
#[derive(Clone)]
pub struct DaemonAgentConnection {
    client: Arc<dyn DaemonTransportClient>,
    active_session_id: Arc<Mutex<String>>,
    options: Arc<Mutex<DaemonAgentConnectionOptions>>,
    listeners: Arc<Mutex<Vec<AgentConnectionEventListener>>>,
    before_session_invalidate_listeners: Arc<Mutex<Vec<AgentConnectionBeforeSessionInvalidateListener>>>,
    unsubscribe_daemon_messages: Arc<Mutex<Option<Box<dyn Fn() + Send + Sync>>>>,
    unsubscribe_daemon_close: Arc<Mutex<Option<Box<dyn Fn() + Send + Sync>>>>,
    client_id: String,
    session_input_pauses: Arc<Mutex<HashMap<String, AgentConnectionSessionInputPause>>>,
    session_input_pause_generation: Arc<Mutex<u64>>,
    owned_session_promotion_tail: Arc<tokio::sync::Mutex<()>>,
    /// `this.reconnectPromise` (`daemon-agent-connection.ts:255`): the in-flight recovery
    /// attempt that `dispose()` races against `OWNED_SESSION_DISPOSE_RECONNECT_WAIT_MS`
    /// (`daemon-agent-connection.ts:1594-1598`).
    reconnect_in_flight: Arc<Mutex<Option<Arc<ReconnectAttempt>>>>,
    last_event_cursor: Arc<Mutex<Option<DaemonEventCursor>>>,
    retired_event_generations: Arc<Mutex<HashSet<String>>>,
    last_event_sequence: Arc<Mutex<Option<i64>>>,
    child_roster_sequence: Arc<Mutex<Option<i64>>>,
    latest_snapshot: Arc<Mutex<Option<AgentConnectionSnapshot>>>,
    latest_snapshot_is_fresh: Arc<Mutex<bool>>,
    attached_session_id: Arc<Mutex<Option<String>>>,
    attached_session_file: Arc<Mutex<Option<String>>>,
    daemon_log_path: Arc<Mutex<Option<String>>>,
    update_restart_pending: Arc<Mutex<bool>>,
    update_reconnect_failed: Arc<Mutex<bool>>,
    terminal_close_emitted: Arc<Mutex<bool>>,
    active_side_question_ids: Arc<Mutex<HashSet<String>>>,
    snapshot_assemblies: Arc<Mutex<HashMap<String, Arc<Mutex<DaemonSnapshotAssembly>>>>>,
    completed_snapshots: Arc<Mutex<VecDeque<(String, DaemonSessionSnapshot)>>>,
    pending_reattach_active_session_ids: Arc<Mutex<HashSet<String>>>,
    ignored_snapshot_ids: Arc<Mutex<VecDeque<String>>>,
    snapshot_in_progress: Arc<Mutex<Option<String>>>,
    deferred_session_events: Arc<Mutex<DeferredSessionEvents>>,
    defer_session_events: Arc<Mutex<bool>>,
    attach_snapshot_pending: Arc<Mutex<bool>>,
    roster_store: Arc<Mutex<Option<Arc<dyn AgentConnectionRosterStore>>>>,
    initial_attach_pending: Arc<Mutex<bool>>,
    initial_control_plane_close: Arc<Mutex<Option<String>>>,
    disposing: Arc<Mutex<bool>>,
    disposed: Arc<Mutex<bool>>,
}

/// `AgentsViewRosterStore` seam (modes/agents-view/roster-store.ts, another slice).
pub trait AgentConnectionRosterStore: Send + Sync {
    fn attach(&self, client: Arc<dyn DaemonTransportClient>) -> BoxFuture<Result<bool, String>>;
    fn on_update(&self, listener: Arc<dyn Fn() + Send + Sync>) -> Box<dyn Fn() + Send + Sync>;
    fn summaries(&self) -> Vec<Value>;
    fn dispose(&self) -> BoxFuture<()>;
}

/// `STALE_ROSTER_DAEMON_MESSAGE`.
pub const STALE_ROSTER_DAEMON_MESSAGE: &str =
    "The running Prime Agent daemon is older than this client. Restart Prime Agent to use Agents View.";

impl DaemonAgentConnection {
    pub fn new(
        client: Arc<dyn DaemonTransportClient>,
        active_session_id: String,
        options: DaemonAgentConnectionOptions,
    ) -> Self {
        if options.recover_daemon {
            client.enable_request_recovery();
        }
        let defer_session_events = options.defer_session_events;
        Self {
            client,
            active_session_id: Arc::new(Mutex::new(active_session_id)),
            options: Arc::new(Mutex::new(options)),
            listeners: Arc::new(Mutex::new(Vec::new())),
            before_session_invalidate_listeners: Arc::new(Mutex::new(Vec::new())),
            unsubscribe_daemon_messages: Arc::new(Mutex::new(None)),
            unsubscribe_daemon_close: Arc::new(Mutex::new(None)),
            client_id: format!("daemon-agent-connection:{}", uuid::Uuid::new_v4()),
            session_input_pauses: Arc::new(Mutex::new(HashMap::new())),
            session_input_pause_generation: Arc::new(Mutex::new(0)),
            owned_session_promotion_tail: Arc::new(tokio::sync::Mutex::new(())),
            reconnect_in_flight: Arc::new(Mutex::new(None)),
            last_event_cursor: Arc::new(Mutex::new(None)),
            retired_event_generations: Arc::new(Mutex::new(HashSet::new())),
            last_event_sequence: Arc::new(Mutex::new(None)),
            child_roster_sequence: Arc::new(Mutex::new(None)),
            latest_snapshot: Arc::new(Mutex::new(None)),
            latest_snapshot_is_fresh: Arc::new(Mutex::new(false)),
            attached_session_id: Arc::new(Mutex::new(None)),
            attached_session_file: Arc::new(Mutex::new(None)),
            daemon_log_path: Arc::new(Mutex::new(None)),
            update_restart_pending: Arc::new(Mutex::new(false)),
            update_reconnect_failed: Arc::new(Mutex::new(false)),
            terminal_close_emitted: Arc::new(Mutex::new(false)),
            active_side_question_ids: Arc::new(Mutex::new(HashSet::new())),
            snapshot_assemblies: Arc::new(Mutex::new(HashMap::new())),
            completed_snapshots: Arc::new(Mutex::new(VecDeque::new())),
            pending_reattach_active_session_ids: Arc::new(Mutex::new(HashSet::new())),
            ignored_snapshot_ids: Arc::new(Mutex::new(VecDeque::new())),
            snapshot_in_progress: Arc::new(Mutex::new(None)),
            deferred_session_events: Arc::new(Mutex::new(DeferredSessionEvents::default())),
            defer_session_events: Arc::new(Mutex::new(defer_session_events)),
            attach_snapshot_pending: Arc::new(Mutex::new(false)),
            roster_store: Arc::new(Mutex::new(None)),
            initial_attach_pending: Arc::new(Mutex::new(false)),
            initial_control_plane_close: Arc::new(Mutex::new(None)),
            disposing: Arc::new(Mutex::new(false)),
            disposed: Arc::new(Mutex::new(false)),
        }
    }

    pub fn active_session_id(&self) -> String {
        self.active_session_id.lock().unwrap().clone()
    }

    fn set_active_session_id(&self, value: String) {
        *self.active_session_id.lock().unwrap() = value;
    }

    /// Transport message/close handlers are registered by `attach`, matching the
    /// TypeScript constructor, and detached by `dispose`.
    fn bind_transport(&self, this: &Arc<Self>) {
        let weak: Weak<DaemonAgentConnection> = Arc::downgrade(this);
        // Advance the event cursor in transport order. Per-message tasks can
        // otherwise advance past agent_end and discard it as stale.
        let (send, mut receive) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(message) = receive.recv().await {
                let Some(connection) = weak.upgrade() else { break };
                if *connection.disposed.lock().unwrap() { break; }
                let mut delivery = Box::pin(async move {
                    if let Err(error) = connection.handle_daemon_message(message).await {
                        append_rotating_log(
                            &get_agent_log_path(),
                            &format!("[{}] daemon-message: ignored failure: {error}", now_iso()),
                        );
                    }
                });
                // JS runs each callback through its first suspension in arrival
                // order. Poll that prefix here; let async listeners/recovery
                // settle independently so they can await later socket messages.
                let first_poll = futures::future::poll_fn(|cx| {
                    std::task::Poll::Ready(std::future::Future::poll(delivery.as_mut(), cx))
                }).await;
                if first_poll.is_pending() {
                    tokio::spawn(delivery);
                }
            }
        });
        let message_handle = self.client.on_message(Arc::new(move |message| {
            let _ = send.send(message);
        }));
        *self.unsubscribe_daemon_messages.lock().unwrap() = Some(message_handle);
        let weak: Weak<DaemonAgentConnection> = Arc::downgrade(this);
        let close_handle = self.client.on_close(Arc::new(move |error| {
            if let Some(connection) = weak.upgrade() {
                let connection_clone = connection.clone();
                tokio::spawn(async move {
                    connection_clone.handle_transport_close(error).await;
                });
            }
        }));
        *self.unsubscribe_daemon_close.lock().unwrap() = Some(close_handle);
        self.capture_daemon_log_path();
    }

    fn capture_daemon_log_path(&self) {
        if let Some(socket_path) = self.client.hello_socket_path() {
            *self.daemon_log_path.lock().unwrap() = Some(get_daemon_log_path(&socket_path));
        }
    }

    fn format_daemon_session_closed_error(&self, reason: &str) -> String {
        let explanation = match reason {
            "killed" => "The daemon stopped this agent session. Its transcript remains saved and can be reopened from Agents View.",
            "shutdown" => "The Prime Agent daemon shut down while this window was attached. The session transcript remains saved; restart Prime Agent and reopen it from Agents View.",
            "completed" => "The daemon closed this agent session after it completed. Its transcript remains available from Agents View.",
            "replaced" => "The daemon replaced this agent session with another session. Reopen the current session from Agents View.",
            "update" => "The Prime Agent daemon restarted for an update, but this window did not restore automatically. The session transcript remains saved; restart Prime Agent and reopen it from Agents View.",
            _ => "The daemon closed this agent session.",
        };
        format!("{explanation} {}", self.format_daemon_diagnostic_context())
    }

    fn format_daemon_connection_closed_error(&self, error: &str) -> String {
        format!(
            "Lost connection to the Prime Agent daemon. Cause: {} The session transcript remains saved; restart Prime Agent or reopen the session from Agents View. {}",
            format_error_sentence(error),
            self.format_daemon_diagnostic_context()
        )
    }

    fn format_update_reconnect_error(&self, error: &str) -> String {
        format!(
            "The Prime Agent daemon restarted for an update, but this window could not reconnect to its restored session before the recovery timeout expired. Last error: {} The session transcript remains saved; restart Prime Agent and reopen it from Agents View. {}",
            format_error_sentence(error),
            self.format_daemon_diagnostic_context()
        )
    }

    fn format_daemon_diagnostic_context(&self) -> String {
        let mut details: Vec<String> = Vec::new();
        if let Some(session_id) = self.attached_session_id.lock().unwrap().clone() {
            details.push(format!("Session ID: {session_id}."));
        }
        if let Some(session_file) = self.attached_session_file.lock().unwrap().clone() {
            details.push(format!("Session file: {session_file}."));
        }
        let log_path = self
            .daemon_log_path
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(get_agent_log_path);
        details.push(format!("Diagnostic log: {log_path}."));
        details.join(" ")
    }

    fn notify_before_session_invalidate(&self) {
        for listener in self.before_session_invalidate_listeners.lock().unwrap().clone() {
            listener();
        }
    }

    fn is_message_for_active_session(&self, message: &DaemonOutbound) -> bool {
        match message.active_session_id() {
            Some(active_session_id) => active_session_id == self.active_session_id(),
            None => false,
        }
    }

    fn is_stale_sequenced_message(&self, message: &DaemonOutbound) -> bool {
        if let Some(cursor) = get_daemon_message_cursor(message) {
            if self.retired_event_generations.lock().unwrap().contains(&cursor.generation) {
                return true;
            }
            let last = self.last_event_cursor.lock().unwrap().clone();
            return match last {
                Some(last) => last.generation == cursor.generation && cursor.sequence <= last.sequence,
                None => false,
            };
        }
        let sequence = get_daemon_message_sequence(message);
        match (sequence, *self.last_event_sequence.lock().unwrap()) {
            (Some(sequence), Some(last)) => sequence <= last,
            _ => false,
        }
    }

    fn observe_event_cursor(&self, cursor: DaemonEventCursor) {
        let current = self.last_event_cursor.lock().unwrap().clone();
        if let Some(current) = &current {
            if current.generation != cursor.generation {
                self.retired_event_generations
                    .lock()
                    .unwrap()
                    .insert(current.generation.clone());
            }
        }
        let should_replace = match &current {
            None => true,
            Some(current) => current.generation != cursor.generation || cursor.sequence > current.sequence,
        };
        if should_replace {
            *self.last_event_cursor.lock().unwrap() = Some(cursor);
        }
    }

    fn observe_daemon_event_sequence(&self, message: &DaemonOutbound) {
        if let Some(cursor) = get_daemon_message_cursor(message) {
            self.observe_event_cursor(cursor.clone());
            *self.last_event_sequence.lock().unwrap() = Some(cursor.sequence);
            return;
        }
        let sequence = match get_daemon_message_sequence(message) {
            Some(sequence) => sequence,
            None => return,
        };
        {
            let mut current = self.last_event_sequence.lock().unwrap();
            *current = max_event_sequence(*current, Some(sequence));
        }
        if let Some(cursor) = self.last_event_cursor.lock().unwrap().as_mut() {
            cursor.sequence = cursor.sequence.max(sequence);
        }
    }

    fn ignore_snapshot_id(&self, snapshot_id: &str) {
        let mut ignored = self.ignored_snapshot_ids.lock().unwrap();
        if !ignored.iter().any(|entry| entry == snapshot_id) {
            ignored.push_back(snapshot_id.to_string());
        }
        while ignored.len() > MAX_IGNORED_SNAPSHOT_IDS {
            ignored.pop_front();
        }
    }

    fn is_ignored_snapshot_id(&self, snapshot_id: &str) -> bool {
        self.ignored_snapshot_ids
            .lock()
            .unwrap()
            .iter()
            .any(|entry| entry == snapshot_id)
    }

    fn reject_snapshot_assemblies(&self, error: String) {
        let assemblies = std::mem::take(&mut *self.snapshot_assemblies.lock().unwrap());
        for (id, assembly) in assemblies {
            self.ignore_snapshot_id(&id);
            let mut guard = assembly.lock().unwrap();
            guard.timeout_armed = false;
            guard.failed = Some(error.clone());
        }
        self.completed_snapshots.lock().unwrap().clear();
        self.snapshot_in_progress.lock().unwrap().take();
        self.deferred_session_events.lock().unwrap().clear();
    }

    fn observe_rlm_child_update(&self, child: AgentConnectionRlmChildAgentSnapshot) {
        let mut snapshot_guard = self.latest_snapshot.lock().unwrap();
        if let Some(snapshot) = snapshot_guard.as_mut() {
            let children = snapshot.children.get_or_insert_with(Vec::new);
            match children.iter().position(|candidate| candidate.id == child.id) {
                Some(index) => children[index] = child,
                None => children.push(child),
            }
        }
    }

    fn observe_streaming_message(&self, event: &AgentConnectionSessionEvent, sequenced: bool) {
        let mut snapshot_guard = self.latest_snapshot.lock().unwrap();
        if snapshot_guard.is_none() {
            return;
        }
        let role = event.message_role();
        match event {
            AgentConnectionSessionEvent::MessageStart { message }
            | AgentConnectionSessionEvent::MessageUpdate { message, .. }
                if role == Some("assistant") =>
            {
                if let Some(snapshot) = snapshot_guard.as_mut() {
                    snapshot.streaming_message = Some(message.clone());
                }
            }
            AgentConnectionSessionEvent::MessageEnd { message } => {
                if let Some(snapshot) = snapshot_guard.as_mut() {
                    if role == Some("assistant") { snapshot.streaming_message = None; }
                    if sequenced || snapshot.messages.last() != Some(message) {
                        snapshot.messages.push(message.clone());
                        snapshot.state.message_count += 1.0;
                        if let Some(history) = snapshot.history.as_mut() { history.total_message_count += 1.0; }
                    }
                }
            }
            _ => {}
        }
        if let Some(snapshot) = snapshot_guard.as_mut() {
            match event.type_name() {
                "agent_start" => { snapshot.state.is_streaming = true; }
                "agent_end" => { snapshot.state.is_streaming = false; snapshot.streaming_message = None; }
                "compaction_start" => snapshot.state.is_compacting = true,
                "compaction_end" => snapshot.state.is_compacting = false,
                "bash_start" => snapshot.state.is_bash_running = true,
                "bash_end" => snapshot.state.is_bash_running = false,
                _ => {}
            }
            snapshot.last_event_sequence = self.last_event_sequence.lock().unwrap().map(|sequence| sequence as f64);
            snapshot.last_event_cursor = self.last_event_cursor.lock().unwrap().as_ref().map(|cursor| AgentConnectionEventCursor { generation: cursor.generation.clone(), sequence: cursor.sequence as f64 });
        }
    }

    fn observe_side_question_event(&self, event: &AgentConnectionSideQuestionEvent) {
        if event.status != "running" {
            self.active_side_question_ids.lock().unwrap().remove(&event.id);
        }
    }

    fn apply_replacement_snapshot(&self, snapshot: &DaemonSessionSnapshot, replay: Option<&DaemonReplayInfo>) {
        if let Some(cursor) = &snapshot.last_event_cursor {
            self.observe_event_cursor(cursor.clone());
        }
        {
            let mut sequence = self.last_event_sequence.lock().unwrap();
            *sequence = snapshot.last_event_sequence;
        }
        *self.attached_session_id.lock().unwrap() = Some(snapshot.state.session_id.clone());
        *self.attached_session_file.lock().unwrap() = snapshot.state.session_file.clone();
        if let Ok(mapped) = map_daemon_session_snapshot(snapshot, replay) {
            *self.latest_snapshot.lock().unwrap() = Some(mapped);
        }
        *self.child_roster_sequence.lock().unwrap() = if snapshot.children.is_some() {
            snapshot.last_event_sequence
        } else {
            None
        };
        *self.latest_snapshot_is_fresh.lock().unwrap() = true;
    }

    fn emit(&self, event: AgentConnectionEvent) -> BoxFuture<()> {
        let listeners = self.listeners.lock().unwrap().clone();
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

    async fn request_data(&self, command: Value, timeout_ms: Option<u64>) -> Result<Value, String> {
        self.request_data_with_recovery(command, timeout_ms, true).await
    }

    async fn request_data_with_recovery(
        &self,
        command: Value,
        timeout_ms: Option<u64>,
        recoverable: bool,
    ) -> Result<Value, String> {
        let response = self
            .client
            .request_with_recoverable(command.clone(), timeout_ms, recoverable)
            .await?;
        if !response.success {
            return Err(response
                .error
                .clone()
                .unwrap_or_else(|| "Daemon request failed".to_string()));
        }
        if let Some(type_) = command.get("type").and_then(Value::as_str) {
            if invalidates_cached_snapshot(type_) {
                *self.latest_snapshot_is_fresh.lock().unwrap() = false;
            }
        }
        Ok(response.data)
    }

    async fn request_ok(&self, command: Value) -> Result<(), String> {
        self.request_data(command, None).await.map(|_| ())
    }
}

impl DaemonAgentConnection {
    /// `DaemonAgentConnection.attach(client, activeSessionId, options)`.
    pub fn attach(
        client: Arc<dyn DaemonTransportClient>,
        active_session_id: String,
        options: DaemonAgentConnectionOptions,
    ) -> BoxFuture<Result<Arc<DaemonAgentConnection>, String>> {
        Box::pin(async move {
            let connection = Arc::new(DaemonAgentConnection::new(client, active_session_id, options));
            self_arc_registry()
                .lock()
                .unwrap()
                .insert(Arc::as_ptr(&connection) as usize, connection.clone());
            connection.bind_transport(&connection);
            let mut attach_scope = AttachScope(Some(connection.clone()));
            *connection.initial_attach_pending.lock().unwrap() = true;
            let initial = connection.attach_once(true).await;
            if let Err(error) = initial {
                // The routed transport falls back to the supervisor and retries once;
                // this retry owns its failure.
                if let Err(retry_error) = connection.attach_once(false).await {
                    // A control-plane close saved during the window is the authoritative cause.
                    let authoritative = connection.initial_control_plane_close.lock().unwrap().clone();
                    *connection.initial_attach_pending.lock().unwrap() = false;
                    connection.dispose_inner().await;
                    let _ = error;
                    return Err(authoritative.unwrap_or(retry_error));
                }
            }
            *connection.initial_attach_pending.lock().unwrap() = false;
            let initial_close = connection.initial_control_plane_close.lock().unwrap().take();
            if let Some(initial_close) = initial_close {
                if get_daemon_socket_close_reason(&initial_close) == Some(DaemonClosingReason::Shutdown) {
                    connection.dispose_inner().await;
                    return Err(initial_close);
                }
                connection.handle_transport_close(initial_close).await;
            }
            attach_scope.0 = None;
            Ok(connection)
        })
    }

    async fn attach_once(&self, recoverable: bool) -> Result<(), String> {
        self.reject_snapshot_assemblies("Superseded by a new attachment".to_string());
        self.deferred_session_events.lock().unwrap().failure = None;
        *self.attach_snapshot_pending.lock().unwrap() = true;
        let supports_extension_ui = self.options.lock().unwrap().supports_extension_ui;
        let owned_session = self.options.lock().unwrap().owned_session;
        let mut capabilities: Vec<Value> = vec![
            json!("attach_snapshot"),
            json!("event_sequence"),
            json!("slim_attach"),
            json!("chunked_snapshot"),
            json!("history_ranges"),
        ];
        if supports_extension_ui {
            capabilities.push(json!("extension_ui"));
        }
        if owned_session {
            capabilities.push(json!("client_owned_sessions"));
        }
        let mut fields: Vec<(&str, Value)> = vec![
            ("activeSessionId", Value::String(self.active_session_id())),
            ("supportsExtensionUi", Value::Bool(supports_extension_ui)),
            ("clientId", Value::String(self.client_id.clone())),
            ("capabilities", Value::Array(capabilities)),
        ];
        if self.options.lock().unwrap().send_client_env {
            fields.push((
                "env",
                serde_json::to_value(collect_daemon_client_env()).unwrap_or(Value::Null),
            ));
        }
        if owned_session {
            fields.push((
                "launchEnv",
                serde_json::to_value(collect_daemon_launch_env()).unwrap_or(Value::Null),
            ));
            let recovery_config = self.options.lock().unwrap().owned_session_recovery_config.clone();
            if let Some(config) = recovery_config {
                if self.client.supports_server_capability("owned_session_recovery_context") {
                    fields.push(("recoveryConfig", config));
                }
            }
        }
        if self.options.lock().unwrap().telemetry_disabled {
            fields.push(("telemetryDisabled", Value::Bool(true)));
        }
        if let Some(cursor) = self.last_event_cursor.lock().unwrap().clone() {
            fields.push((
                "resumeCursor",
                json!({
                    "activeSessionId": self.active_session_id(),
                    "generation": cursor.generation,
                    "sequence": cursor.sequence,
                }),
            ));
        }
        let command = command_body("attach", fields);
        let result = match self.request_data_with_recovery(command, None, recoverable).await {
            Ok(data) => self.apply_attach_result(&data).await,
            Err(error) => Err(error),
        };
        *self.attach_snapshot_pending.lock().unwrap() = false;
        if result.is_ok() { self.drain_deferred_session_events().await?; }
        result
    }

    async fn apply_attach_result(&self, data: &Value) -> Result<(), String> {
        let attach_result = parse_attach_result(data)?;
        self.set_active_session_id(get_attach_active_session_id(&attach_result));
        let summary = attach_result.snapshot.summary.clone();
        *self.attached_session_id.lock().unwrap() = Some(summary.session_id.clone());
        *self.attached_session_file.lock().unwrap() = summary
            .session_file
            .clone()
            .or_else(|| attach_result.snapshot.state.session_file.clone());
        self.capture_daemon_log_path();
        *self.update_reconnect_failed.lock().unwrap() = false;
        *self.terminal_close_emitted.lock().unwrap() = false;
        let snapshot = match &attach_result.snapshot_stream {
            Some(stream) => self.wait_for_snapshot(&stream.id).await?,
            None => attach_result.snapshot.clone(),
        };
        if *self.disposed.lock().unwrap() { return Err("Daemon connection disposed during attach".to_string()); }
        if let Some(error) = self.deferred_session_events.lock().unwrap().failure.clone() { return Err(error); }
        let mapped = map_daemon_session_snapshot(&snapshot, attach_result.replay.as_ref())?;
        if snapshot.children.is_some() {
            *self.child_roster_sequence.lock().unwrap() = snapshot.last_event_sequence;
        }
        *self.last_event_sequence.lock().unwrap() = snapshot.last_event_sequence;
        if let Some(cursor) = &snapshot.last_event_cursor { self.observe_event_cursor(cursor.clone()); }
        *self.last_event_cursor.lock().unwrap() = snapshot.last_event_cursor.clone();
        *self.latest_snapshot.lock().unwrap() = Some(mapped);
        *self.latest_snapshot_is_fresh.lock().unwrap() = true;
        // The roster bar is an accessory: its subscribe failure must never fail an
        // otherwise-recovered session. The bar degrades; the next reconnect or rebind
        // re-attaches through this same seam.
        let roster = self.roster_store.lock().unwrap().clone();
        if let Some(store) = roster {
            let _ = store.attach(self.client.clone()).await;
        }
        if let Some(error) = self.deferred_session_events.lock().unwrap().failure.clone() { return Err(error); }
        Ok(())
    }

    async fn handle_transport_close(&self, error: String) {
        let direct_session_survives = self.client.has_direct_transport();
        let invalidated_input_pause =
            !direct_session_survives && !self.session_input_pauses.lock().unwrap().is_empty();
        if !direct_session_survives {
            self.session_input_pauses.lock().unwrap().clear();
            *self.session_input_pause_generation.lock().unwrap() += 1;
            self.reject_snapshot_assemblies(error.clone());
        }
        if *self.initial_attach_pending.lock().unwrap() {
            // attach() owns failure handling until the initial attach settles.
            if direct_session_survives {
                *self.initial_control_plane_close.lock().unwrap() = Some(error);
            }
            return;
        }
        if *self.disposed.lock().unwrap() || *self.terminal_close_emitted.lock().unwrap() {
            return;
        }
        // A lost direct link invalidates the fence (holders learn via the generation bump) yet the session falls back.
        if invalidated_input_pause {
            *self.terminal_close_emitted.lock().unwrap() = true;
            let message = "Daemon connection closed while session input was paused; the fence was invalidated.";
            self.emit(AgentConnectionEvent::Closed {
                error: Some(message.to_string()),
            })
            .await;
            return;
        }
        // An authoritative shutdown/update reason outranks the surviving direct link.
        if get_daemon_socket_close_reason(&error) == Some(DaemonClosingReason::Shutdown) {
            *self.terminal_close_emitted.lock().unwrap() = true;
            let message = self.format_daemon_session_closed_error("shutdown");
            self.emit(AgentConnectionEvent::Closed { error: Some(message) }).await;
            return;
        }
        let update_pending = *self.update_restart_pending.lock().unwrap();
        if (update_pending || get_daemon_socket_close_reason(&error) == Some(DaemonClosingReason::Update))
            && !*self.update_reconnect_failed.lock().unwrap()
        {
            *self.update_restart_pending.lock().unwrap() = true;
            self.reconnect_after_update();
            return;
        }
        // A direct-transport loss is never itself a session loss: fall back through a supervisor re-attach.
        if direct_session_survives || self.options.lock().unwrap().recover_daemon {
            let _ = self.reconnect(error).await;
            return;
        }
        *self.terminal_close_emitted.lock().unwrap() = true;
        let message = self.format_daemon_connection_closed_error(&error);
        self.emit(AgentConnectionEvent::Closed { error: Some(message) }).await;
    }

    /// `reconnectAfterUpdate()`: fire-and-forget, single-flight per connection.
    fn reconnect_after_update(&self) {
        // The TypeScript keeps a promise per client; the port spawns one task and
        // guards it with `update_restart_pending`, which the caller already set.
        let Some(connection) = self.self_arc() else {
            return;
        };
        tokio::spawn(async move {
            let _ = connection.reconnect("The Prime Agent daemon is restarting for an update.".to_string()).await;
        });
    }

    async fn reconnect_update_owner(&self) -> Result<(), String> {
            let connection = self;
            let client = connection.client.clone();
            let result = reconnect_daemon_transport_after_update(client).await;
            let restore = match result {
                Ok(()) => connection.restore_connection_after_update().await,
                Err(error) => Err(error),
            };
            match restore {
                Ok(()) => {
                    if !*connection.disposed.lock().unwrap() {
                        connection
                            .emit(AgentConnectionEvent::ConnectionStatus {
                                status: "connected".to_string(),
                                error: None,
                            })
                            .await;
                    }
                }
                Err(error) => {
                    *connection.update_restart_pending.lock().unwrap() = false;
                    *connection.update_reconnect_failed.lock().unwrap() = true;
                    if !*connection.disposed.lock().unwrap() {
                        *connection.terminal_close_emitted.lock().unwrap() = true;
                        let message = connection.format_update_reconnect_error(&error);
                        connection
                            .emit(AgentConnectionEvent::Closed { error: Some(message) })
                            .await;
                    }
                }
            }
            Ok(())
    }

    /// The connection hands its own `Arc` to spawned recovery tasks.
    fn self_arc(&self) -> Option<Arc<DaemonAgentConnection>> {
        self_arc_registry()
            .lock()
            .unwrap()
            .values()
            .find(|connection| connection.client_id == self.client_id)
            .cloned()
            .or_else(|| Some(Arc::new(self.clone())))
    }
}

/// Registry of the `Arc` a `DaemonAgentConnection` was created with.
///
/// The TypeScript keeps recovery state on the instance and starts promises from
/// methods, which works without self-references. Rust needs the `Arc` for
/// `tokio::spawn`, so `attach` registers it here and `dispose` removes it.
static SELF_ARCS: once_cell::sync::Lazy<Mutex<HashMap<usize, Arc<DaemonAgentConnection>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

fn self_arc_registry() -> &'static Mutex<HashMap<usize, Arc<DaemonAgentConnection>>> {
    &SELF_ARCS
}

fn parse_attach_result(data: &Value) -> Result<DaemonAttachResult, String> {
    let object = data
        .as_object()
        .ok_or_else(|| "Daemon returned an invalid attach result".to_string())?;
    let active_session_id = object
        .get("activeSessionId")
        .and_then(Value::as_str)
        .ok_or_else(|| "Daemon returned an invalid attach result".to_string())?
        .to_string();
    let snapshot_value = object.get("snapshot").cloned().unwrap_or(Value::Null);
    let snapshot = parse_session_snapshot(&snapshot_value)?;
    let snapshot_stream = object
        .get("snapshotStream")
        .and_then(Value::as_object)
        .and_then(|stream| stream.get("id"))
        .and_then(Value::as_str)
        .map(|id| DaemonSnapshotStream { id: id.to_string() });
    Ok(DaemonAttachResult {
        active_session_id,
        snapshot,
        snapshot_stream,
        replay: None,
    })
}

fn parse_session_snapshot(value: &Value) -> Result<DaemonSessionSnapshot, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Daemon returned an invalid session snapshot".to_string())?;
    let state = object
        .get("state")
        .cloned()
        .and_then(|state| serde_json::from_value(state).ok())
        .unwrap_or_default();
    let messages: Vec<AgentMessage> = object
        .get("messages")
        .cloned()
        .and_then(|messages| serde_json::from_value(messages).ok())
        .unwrap_or_default();
    let summary_value = object.get("summary").cloned().unwrap_or(Value::Null);
    let summary = DaemonSessionSummary {
        session_id: summary_value
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        session_file: summary_value
            .get("sessionFile")
            .and_then(Value::as_str)
            .map(str::to_string),
        active_session_id: summary_value
            .get("activeSessionId")
            .and_then(Value::as_str)
            .map(str::to_string),
        id: summary_value.get("id").and_then(Value::as_str).map(str::to_string),
        streaming_message: summary_value
            .get("streamingMessage")
            .cloned()
            .and_then(|message| serde_json::from_value(message).ok()),
        last_event_sequence: summary_value.get("lastEventSequence").and_then(Value::as_i64),
        last_event_cursor: None,
    };
    Ok(DaemonSessionSnapshot {
        state,
        messages,
        summary,
        history: object
            .get("history")
            .cloned()
            .and_then(|history| serde_json::from_value(history).ok()),
        session_context: object
            .get("sessionContext")
            .cloned()
            .and_then(|context| serde_json::from_value(context).ok()),
        session_tree: object
            .get("sessionTree")
            .cloned()
            .and_then(|tree| serde_json::from_value(tree).ok()),
        parent: object
            .get("parent")
            .cloned()
            .and_then(|parent| serde_json::from_value(parent).ok()),
        children: object
            .get("children")
            .cloned()
            .and_then(|children| serde_json::from_value(children).ok()),
        last_event_sequence: object.get("lastEventSequence").and_then(Value::as_i64),
        last_event_cursor: object
            .get("lastEventCursor")
            .and_then(Value::as_object)
            .map(|cursor| DaemonEventCursor {
                generation: cursor
                    .get("generation")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                sequence: cursor.get("sequence").and_then(Value::as_i64).unwrap_or(0),
            }),
    })
}

impl DaemonAgentConnection {
    async fn restore_connection_after_update(&self) -> Result<(), String> {
        let session_id = self.attached_session_id.lock().unwrap().clone();
        let session_file = self.attached_session_file.lock().unwrap().clone();
        if session_id.is_none() && session_file.is_none() {
            return Err("the previous session identity is unavailable".to_string());
        }
        let deadline = now_ms() + UPDATE_RECONNECT_TIMEOUT_MS as i64;
        let mut last_error: Option<String> = None;
        while !*self.disposed.lock().unwrap() && now_ms() < deadline {
            let attempt: Result<bool, String> = async {
                self.client.reconnect(1000).await?;
                if *self.disposed.lock().unwrap() {
                    return Ok(true);
                }
                // This loop owns the retry: a socket close must reject these instead of
                // parking them behind a hello it can never produce.
                let response = self
                    .client
                    .request_with_recoverable(command_body("list", vec![]), Some(30000), false)
                    .await?;
                if *self.disposed.lock().unwrap() {
                    return Ok(true);
                }
                if !response.success {
                    return Err(response
                        .error
                        .clone()
                        .unwrap_or_else(|| "Daemon request failed".to_string()));
                }
                let sessions = read_session_summaries(&response.data)?;
                let restored = sessions.into_iter().find(|summary| {
                    summary.active_session_id.is_some()
                        && ((session_file.is_some() && summary.session_file == session_file)
                            || (session_id.is_some() && Some(summary.session_id.clone()) == session_id))
                });
                match restored.and_then(|summary| summary.active_session_id) {
                    Some(active_session_id) => {
                        if *self.disposed.lock().unwrap() {
                            return Ok(true);
                        }
                        self.set_active_session_id(active_session_id);
                        *self.last_event_sequence.lock().unwrap() = None;
                        *self.last_event_cursor.lock().unwrap() = None;
                        self.retired_event_generations.lock().unwrap().clear();
                        self.attach_once(false).await?;
                        if *self.disposed.lock().unwrap() {
                            return Ok(true);
                        }
                        let snapshot = self.get_initial_snapshot_inner(false).await?;
                        if *self.disposed.lock().unwrap() {
                            return Ok(true);
                        }
                        *self.update_restart_pending.lock().unwrap() = false;
                        self.emit(AgentConnectionEvent::SessionResynced { snapshot }).await;
                        Ok(true)
                    }
                    None => Ok(false),
                }
            }
            .await;
            match attempt {
                Ok(true) => return Ok(()),
                Ok(false) => {}
                Err(error) => last_error = Some(error),
            }
            tokio::time::sleep(Duration::from_millis(UPDATE_RECONNECT_RETRY_MS)).await;
        }
        if *self.disposed.lock().unwrap() {
            return Ok(());
        }
        Err(last_error.unwrap_or_else(|| "the restored session did not become available".to_string()))
    }

    /// `reconnect(cause)`: single-flight through `reconnectPromise`.
    async fn reconnect(&self, cause: String) -> Result<(), String> {
        let (attempt, owner) = {
            let mut slot = self.reconnect_in_flight.lock().unwrap();
            match slot.as_ref() {
                Some(attempt) => (attempt.clone(), false),
                None => {
                    let (result, _) = tokio::sync::watch::channel(None);
                    let attempt = Arc::new(ReconnectAttempt { result, cancel: tokio_util::sync::CancellationToken::new() });
                    *slot = Some(attempt.clone());
                    (attempt, true)
                }
            }
        };
        if !owner {
            let mut result = attempt.result.subscribe();
            loop {
                if let Some(result) = result.borrow().clone() { return result; }
                if result.changed().await.is_err() { return Err("Daemon reconnect cancelled".to_string()); }
            }
        }
        let _scope = ReconnectScope { slot: self.reconnect_in_flight.clone(), attempt: attempt.clone() };
        let result = tokio::select! {
            _ = attempt.cancel.cancelled() => Err("Daemon reconnect cancelled".to_string()),
            result = async {
                if !*self.update_restart_pending.lock().unwrap() {
                    self.reconnect_owner(cause).await?;
                }
                if !*self.disposed.lock().unwrap() && *self.update_restart_pending.lock().unwrap() {
                    self.reconnect_update_owner().await?;
                }
                Ok(())
            } => result,
        };
        attempt.result.send_replace(Some(result.clone()));
        result
    }

    async fn reconnect_owner(&self, cause: String) -> Result<(), String> {
        // Publish `this.reconnectPromise`'s completion signal so `dispose` can await it
        // (`daemon-agent-connection.ts:1643-1646` stores the promise; `:1594` races it).
        self.emit(AgentConnectionEvent::ConnectionStatus {
            status: "reconnecting".to_string(),
            error: Some(cause.clone()),
        })
        .await;
        let timeout_ms = self
            .options
            .lock()
            .unwrap()
            .reconnect_timeout_ms
            .unwrap_or(DAEMON_RECONNECT_TIMEOUT_MS);
        let mut deadline: Option<i64> = None;
        let mut attempt = 0u32;
        let mut last_error = cause;
        while !*self.disposed.lock().unwrap() {
            if *self.update_restart_pending.lock().unwrap() { return Ok(()); }
            // A held direct link owns session liveness: control-plane recovery retries unbounded,
            // and the bounded session-plane deadline arms only once the direct link is gone.
            let direct_session_held = self.client.has_direct_transport();
            if direct_session_held {
                deadline = None;
            } else {
                let current = *deadline.get_or_insert_with(|| now_ms() + timeout_ms as i64);
                if now_ms() >= current {
                    break;
                }
            }
            let mut control_plane_handshake_complete = false;
            let mut attempt_error: Option<String> = None;
            if let Err(error) = self.client.connect(1000).await {
                attempt_error = Some(error);
            } else if *self.disposed.lock().unwrap() {
                return Ok(());
            } else if let Err(error) = self.client.wait_for_hello(3000).await {
                attempt_error = Some(error);
            } else {
                control_plane_handshake_complete = true;
                if direct_session_held {
                    // The roster subscription is a control-plane accessory; its usual rebind
                    // seam (attach) is skipped while held.
                    let roster = self.roster_store.lock().unwrap().clone();
                    if let Some(store) = roster {
                        let _ = store.attach(self.client.clone()).await;
                    }
                    if *self.disposed.lock().unwrap()
                        || *self.terminal_close_emitted.lock().unwrap()
                        || *self.update_restart_pending.lock().unwrap()
                    {
                        return Ok(());
                    }
                    if self.client.has_direct_transport() {
                        self.emit(AgentConnectionEvent::ConnectionStatus {
                            status: "connected".to_string(),
                            error: None,
                        })
                        .await;
                        return Ok(());
                    }
                } else {
                    match self.attach_once(false).await {
                        Ok(()) => match self.get_initial_snapshot_inner(false).await {
                            Ok(snapshot) => {
                                if !*self.disposed.lock().unwrap() {
                                    self.emit(AgentConnectionEvent::SessionResynced { snapshot }).await;
                                    self.emit(AgentConnectionEvent::ConnectionStatus {
                                        status: "connected".to_string(),
                                        error: None,
                                    })
                                    .await;
                                }
                                return Ok(());
                            }
                            Err(error) => attempt_error = Some(error),
                        },
                        Err(error) => attempt_error = Some(error),
                    }
                }
            }
            if let Some(error) = attempt_error {
                last_error = error;
                if *self.disposed.lock().unwrap() {
                    return Ok(());
                }
                // A direct-half failure must not tear down a control-plane socket with a
                // completed handshake.
                let should_reset_control_plane =
                    !control_plane_handshake_complete || !self.client.is_control_plane_ready();
                if should_reset_control_plane {
                    self.client.reset_transport_for_reconnect();
                }
                if let Some(deadline) = deadline {
                    if deadline - now_ms() <= 0 {
                        break;
                    }
                }
                let backoff = 100i64 * 2i64.pow(attempt.min(5));
                let delay_ms = match deadline {
                    Some(deadline) => backoff.min(2000).min((deadline - now_ms()).max(0)),
                    None => backoff.min(2000),
                };
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(delay_ms as u64)).await;
            }
        }
        if !*self.disposed.lock().unwrap() {
            self.session_input_pauses.lock().unwrap().clear();
            *self.session_input_pause_generation.lock().unwrap() += 1;
            self.client.close();
            let message = format!("Daemon reconnection failed: {last_error}");
            self.emit(AgentConnectionEvent::Closed { error: Some(message) }).await;
        }
        Ok(())
    }

    async fn get_initial_snapshot_inner(&self, recoverable: bool) -> Result<AgentConnectionSnapshot, String> {
        if let Some(error) = self.deferred_session_events.lock().unwrap().failure.clone() { return Err(error); }
        if *self.latest_snapshot_is_fresh.lock().unwrap() {
            if let Some(snapshot) = self.latest_snapshot.lock().unwrap().clone() {
                return Ok(snapshot);
            }
        }
        // The session tree is intentionally not fetched here: it is large on long
        // sessions and only needed when the user opens the tree/branch selector.
        let snapshot_cursor = self.last_event_cursor.lock().unwrap().clone();
        let snapshot_sequence = *self.last_event_sequence.lock().unwrap();
        let active_session_id = self.active_session_id();
        let state_future = self.request_data_with_recovery(
            command_body(
                "get_connection_state",
                vec![("activeSessionId", Value::String(active_session_id.clone()))],
            ),
            None,
            recoverable,
        );
        let messages_future = self.request_data_with_recovery(
            command_body(
                "get_messages",
                vec![("activeSessionId", Value::String(active_session_id.clone()))],
            ),
            None,
            recoverable,
        );
        let context_future = self.request_data_with_recovery(
            command_body(
                "get_session_context",
                vec![("activeSessionId", Value::String(active_session_id))],
            ),
            None,
            recoverable,
        );
        let (state_data, messages_data, context_data) =
            tokio::join!(state_future, messages_future, context_future);
        let state: AgentConnectionState = serde_json::from_value(state_data?)
            .map_err(|error| format!("Daemon returned an invalid connection state: {error}"))?;
        let messages: Vec<AgentMessage> = messages_data?
            .get("messages")
            .cloned()
            .and_then(|messages| serde_json::from_value(messages).ok())
            .unwrap_or_default();
        let context: Option<AgentConnectionSessionContext> = context_data?
            .get("context")
            .cloned()
            .and_then(|context| serde_json::from_value(context).ok());
        let previous = self.latest_snapshot.lock().unwrap().clone();
        let children = previous.as_ref().and_then(|snapshot| snapshot.children.clone());
        let streaming_message = previous.as_ref().and_then(|snapshot| snapshot.streaming_message.clone());
        let mut snapshot = AgentConnectionSnapshot {
            state,
            messages,
            history: None,
            streaming_message,
            session_context: context,
            session_tree: None,
            parent: None,
            children,
            last_event_sequence: snapshot_sequence.map(|value| value as f64),
            last_event_cursor: snapshot_cursor.as_ref().map(|cursor| AgentConnectionEventCursor {
                generation: cursor.generation.clone(),
                sequence: cursor.sequence as f64,
            }),
            replay: None,
        };
        let latest_cursor = self.last_event_cursor.lock().unwrap().clone();
        let fresh = snapshot_sequence == *self.last_event_sequence.lock().unwrap()
            && snapshot_cursor.as_ref().map(|cursor| &cursor.generation)
                == latest_cursor.as_ref().map(|cursor| &cursor.generation)
            && snapshot_cursor.as_ref().map(|cursor| cursor.sequence)
                == latest_cursor.as_ref().map(|cursor| cursor.sequence);
        if snapshot.history.is_none() {
            snapshot.history = None;
        }
        *self.latest_snapshot.lock().unwrap() = Some(snapshot.clone());
        *self.latest_snapshot_is_fresh.lock().unwrap() = fresh;
        Ok(snapshot)
    }

    /// `waitForSnapshot(snapshotId)`.
    async fn wait_for_snapshot(&self, snapshot_id: &str) -> Result<DaemonSessionSnapshot, String> {
        if let Some(error) = self.deferred_session_events.lock().unwrap().failure.clone() { return Err(error); }
        if *self.disposed.lock().unwrap() || self.is_ignored_snapshot_id(snapshot_id) {
            return Err(format!("Snapshot {snapshot_id} is no longer available"));
        }
        let completed = {
            let mut snapshots = self.completed_snapshots.lock().unwrap();
            snapshots.iter().position(|(id, _)| id == snapshot_id)
                .and_then(|index| snapshots.remove(index))
        };
        if let Some((_, snapshot)) = completed {
            return Ok(snapshot);
        }
        let assembly = self.get_snapshot_assembly(snapshot_id);
        let timeout_ms = self
            .options
            .lock()
            .unwrap()
            .snapshot_timeout_ms
            .unwrap_or(DAEMON_SNAPSHOT_TIMEOUT_MS);
        let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            if *self.disposed.lock().unwrap() { return Err("Daemon connection disposed during snapshot transfer".to_string()); }
            if let Some(error) = self.deferred_session_events.lock().unwrap().failure.clone() { return Err(error); }
            {
                let guard = assembly.lock().unwrap();
                if let Some(error) = &guard.failed {
                    let error = error.clone();
                    drop(guard);
                    self.snapshot_assemblies.lock().unwrap().remove(snapshot_id);
                    self.ignore_snapshot_id(snapshot_id);
                    return Err(error);
                }
                if guard.completed {
                    break;
                }
            }
            if tokio::time::Instant::now() >= deadline {
                let error = format!("Timed out waiting for snapshot {snapshot_id}");
                {
                    let mut guard = assembly.lock().unwrap();
                    guard.failed = Some(error.clone());
                }
                self.snapshot_assemblies.lock().unwrap().remove(snapshot_id);
                self.ignore_snapshot_id(snapshot_id);
                return Err(error);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let snapshot = self.snapshot_assemblies.lock().unwrap().remove(snapshot_id);
        drop(snapshot);
        let assembled = {
            let guard = assembly.lock().unwrap();
            guard.begin.clone()
        };
        match assembled {
            Some(begin) => Ok(begin.snapshot),
            None => Err(format!("Snapshot {snapshot_id} ended before it began")),
        }
    }

    fn get_snapshot_assembly(&self, snapshot_id: &str) -> Arc<Mutex<DaemonSnapshotAssembly>> {
        let mut assemblies = self.snapshot_assemblies.lock().unwrap();
        if let Some(existing) = assemblies.get(snapshot_id) {
            return existing.clone();
        }
        let assembly = Arc::new(Mutex::new(DaemonSnapshotAssembly::default()));
        assemblies.insert(snapshot_id.to_string(), assembly.clone());
        assembly
    }

    fn reject_snapshot_assembly(
        &self,
        snapshot_id: &str,
        assembly: &Arc<Mutex<DaemonSnapshotAssembly>>,
        error: String,
    ) {
        {
            let mut guard = assembly.lock().unwrap();
            guard.timeout_armed = false;
            guard.failed = Some(error);
        }
        self.snapshot_assemblies.lock().unwrap().remove(snapshot_id);
        self.ignore_snapshot_id(snapshot_id);
        let mut current = self.snapshot_in_progress.lock().unwrap();
        if current.as_deref() == Some(snapshot_id) { current.take(); }
    }

    async fn recover_failed_snapshot(&self, purpose: &str, snapshot_error: String) {
        *self.latest_snapshot_is_fresh.lock().unwrap() = false;
        if purpose == "replacement" {
            *self.latest_snapshot.lock().unwrap() = None;
        }
        match self.get_initial_snapshot_inner(true).await {
            Ok(snapshot) => {
                if *self.disposed.lock().unwrap() {
                    return;
                }
                *self.attached_session_id.lock().unwrap() = Some(snapshot.state.session_id.clone());
                *self.attached_session_file.lock().unwrap() = snapshot.state.session_file.clone();
                if purpose == "replacement" {
                    self.emit(AgentConnectionEvent::SessionReplaced {
                        state: snapshot.state.clone(),
                        messages: snapshot.messages.clone(),
                    })
                    .await;
                } else {
                    self.emit(AgentConnectionEvent::SessionResynced { snapshot }).await;
                }
            }
            Err(recovery_error) => {
                if *self.disposed.lock().unwrap() {
                    return;
                }
                *self.terminal_close_emitted.lock().unwrap() = true;
                let message = format!(
                    "Failed to recover from a {purpose} snapshot transfer. Snapshot error: {} Recovery error: {} {}",
                    format_error_sentence(&snapshot_error),
                    format_error_sentence(&recovery_error),
                    self.format_daemon_diagnostic_context()
                );
                self.emit(AgentConnectionEvent::Closed { error: Some(message) }).await;
            }
        }
    }

    async fn complete_snapshot_assembly(&self, snapshot_id: &str, chunk_count: usize, last_event_sequence: i64, last_event_cursor: Option<DaemonEventCursor>) {
        if *self.disposed.lock().unwrap() || self.is_ignored_snapshot_id(snapshot_id) { return; }
        let assembly = { self.snapshot_assemblies.lock().unwrap().get(snapshot_id).cloned() };
        let Some(assembly) = assembly else { self.ignore_snapshot_id(snapshot_id); return; };
        let begin = { assembly.lock().unwrap().begin.clone() };
        let Some(begin) = begin else {
            self.reject_snapshot_assembly(
                snapshot_id,
                &assembly,
                format!("Snapshot {snapshot_id} ended before it began"),
            );
            return;
        };
        if self.snapshot_in_progress.lock().unwrap().as_deref() != Some(snapshot_id)
            || begin.snapshot.state.active_session_id.as_deref().is_some_and(|id| id != self.active_session_id())
            || begin.snapshot.last_event_sequence.is_some_and(|sequence| sequence != last_event_sequence)
            || begin.snapshot.last_event_cursor.as_ref().is_some_and(|cursor| Some(cursor) != last_event_cursor.as_ref())
        {
            self.reject_snapshot_assembly(snapshot_id, &assembly, format!("Snapshot {snapshot_id} was superseded or changed its captured cursor"));
            return;
        }
        let chunk_len = { assembly.lock().unwrap().chunks.len() };
        if chunk_len != chunk_count {
            self.reject_snapshot_assembly(
                snapshot_id,
                &assembly,
                format!("Snapshot {snapshot_id} ended with {chunk_len} of {chunk_count} chunks"),
            );
            return;
        }
        let mut messages: Vec<AgentMessage> = Vec::new();
        for index in 0..chunk_count {
            let chunk = { assembly.lock().unwrap().chunks.get(&index).cloned() };
            match chunk {
                Some(chunk) => messages.extend(chunk),
                None => {
                    self.reject_snapshot_assembly(
                        snapshot_id,
                        &assembly,
                        format!("Snapshot {snapshot_id} is missing chunk {index}"),
                    );
                    return;
                }
            }
        }
        if messages.len() != begin.message_count {
            self.reject_snapshot_assembly(
                snapshot_id,
                &assembly,
                format!(
                    "Snapshot {snapshot_id} contained {} of {} messages",
                    messages.len(),
                    begin.message_count
                ),
            );
            return;
        }
        let mut snapshot = begin.snapshot.clone();
        snapshot.messages = messages.clone();
        snapshot.last_event_sequence = Some(last_event_sequence);
        snapshot.last_event_cursor = last_event_cursor.clone();
        let purpose = begin.purpose.clone();
        if purpose != "attach" {
            self.apply_replacement_snapshot(&snapshot, None);
        }
        self.snapshot_in_progress.lock().unwrap().take();
        {
            let mut guard = assembly.lock().unwrap();
            // wait_for_snapshot returns the saved begin snapshot. Publish the
            // assembled transcript there before marking complete, otherwise an
            // attach replaces the full conversation with its empty chunk header.
            if let Some(begin) = guard.begin.as_mut() {
                begin.snapshot = snapshot.clone();
            }
            guard.completed = true;
        }
        if purpose != "attach" {
            self.snapshot_assemblies.lock().unwrap().remove(snapshot_id);
            if self
                .pending_reattach_active_session_ids
                .lock()
                .unwrap()
                .contains(&snapshot.state.active_session_id.clone().unwrap_or_default())
            {
                let mut completed = self.completed_snapshots.lock().unwrap();
                completed.push_back((snapshot_id.to_string(), snapshot.clone()));
                while completed.len() > MAX_COMPLETED_SNAPSHOTS {
                    completed.pop_front();
                }
            }
        }
        if purpose == "replacement" {
            self.emit(AgentConnectionEvent::SessionReplaced {
                state: snapshot.state.clone(),
                messages,
            })
            .await;
        } else if purpose == "resync" {
            self.notify_before_session_invalidate();
            let mapped = self.latest_snapshot.lock().unwrap().clone().unwrap_or_default();
            self.emit(AgentConnectionEvent::SessionResynced { snapshot: mapped }).await;
        }
        if purpose != "attach" { let _ = self.drain_deferred_session_events().await; }
    }

    async fn handle_daemon_message(&self, message: DaemonOutbound) -> Result<(), String> {
        self.handle_daemon_message_inner(message, false).await
    }

    async fn handle_daemon_message_inner(&self, message: DaemonOutbound, replaying: bool) -> Result<(), String> {
        if *self.disposed.lock().unwrap() { return Ok(()); }
        // A bounded opening buffer must fail visibly, never continue beyond a
        // missing prefix. A fresh attachment explicitly resets this failure.
        if self.deferred_session_events.lock().unwrap().failure.is_some() { return Ok(()); }
        if matches!(message, DaemonOutbound::HeartbeatsChanged { .. }) {
            self.emit(AgentConnectionEvent::HeartbeatsChanged).await;
            return Ok(());
        }
        if !self.is_message_for_active_session(&message) {
            return Ok(());
        }
        if let Some(snapshot_id) = message.snapshot_id() {
            if self.is_ignored_snapshot_id(snapshot_id) {
                return Ok(());
            }
        }
        match &message {
            DaemonOutbound::SessionSnapshotBegin {
                snapshot_id,
                snapshot,
                message_count,
                purpose,
                ..
            } => {
                if snapshot.last_event_cursor.as_ref().is_some_and(|cursor| self.retired_event_generations.lock().unwrap().contains(&cursor.generation)) {
                    self.ignore_snapshot_id(snapshot_id);
                    return Ok(());
                }
                let stale = match (&snapshot.last_event_cursor, &*self.last_event_cursor.lock().unwrap()) {
                    (Some(incoming), Some(current)) => incoming.generation == current.generation && incoming.sequence < current.sequence,
                    _ => false,
                };
                if stale && !*self.attach_snapshot_pending.lock().unwrap() {
                    self.ignore_snapshot_id(snapshot_id);
                    return Ok(());
                }
                let previous = self.snapshot_in_progress.lock().unwrap().replace(snapshot_id.clone());
                if let Some(previous) = previous.filter(|previous| previous != snapshot_id) {
                    let old = { self.snapshot_assemblies.lock().unwrap().get(&previous).cloned() };
                    if let Some(old) = old { self.reject_snapshot_assembly(&previous, &old, "Snapshot superseded".to_string()); }
                }
                let assembly = self.get_snapshot_assembly(snapshot_id);
                let mut guard = assembly.lock().unwrap();
                guard.begin = Some(SnapshotBegin {
                    snapshot: snapshot.clone(),
                    message_count: *message_count,
                    purpose: purpose.clone().unwrap_or_else(|| "attach".to_string()),
                });
                guard.timeout_armed = true;
                return Ok(());
            }
            DaemonOutbound::SessionSnapshotChunk {
                snapshot_id,
                index,
                messages,
                ..
            } => {
                let assembly = { self.snapshot_assemblies.lock().unwrap().get(snapshot_id).cloned() };
                if let Some(assembly) = assembly { assembly.lock().unwrap().chunks.insert(*index, messages.clone()); }
                return Ok(());
            }
            DaemonOutbound::SessionSnapshotEnd {
                snapshot_id,
                chunk_count,
                last_event_sequence,
                last_event_cursor,
                ..
            } => {
                self.complete_snapshot_assembly(
                    snapshot_id,
                    *chunk_count,
                    *last_event_sequence,
                    last_event_cursor.clone(),
                )
                .await;
                return Ok(());
            }
            DaemonOutbound::SessionSnapshotFailed {
                snapshot_id, error, ..
            } => {
                let assembly = self.get_snapshot_assembly(snapshot_id);
                let purpose = {
                    let guard = assembly.lock().unwrap();
                    guard
                        .begin
                        .as_ref()
                        .map(|begin| begin.purpose.clone())
                        .unwrap_or_else(|| "attach".to_string())
                };
                let snapshot_error = error.clone();
                let recovery = if purpose == "replacement" || purpose == "resync" {
                    Some((purpose.clone(), snapshot_error.clone()))
                } else {
                    None
                };
                self.reject_snapshot_assembly(snapshot_id, &assembly, snapshot_error);
                self.ignore_snapshot_id(snapshot_id);
                if let Some((purpose, error)) = recovery {
                    self.recover_failed_snapshot(&purpose, error).await;
                }
                return Ok(());
            }
            _ => {}
        }
        let deferred = {
            let mut pending = self.deferred_session_events.lock().unwrap();
            if !replaying && !matches!(message, DaemonOutbound::SessionClosed { .. })
            && (pending.draining || *self.defer_session_events.lock().unwrap()
                || *self.attach_snapshot_pending.lock().unwrap()
                || self.snapshot_in_progress.lock().unwrap().is_some())
            {
                if pending.push(message.clone()) { 1 } else {
                    pending.clear();
                    pending.failure = Some("Opening this busy conversation exceeded the bounded update buffer. Reopen it to load a fresh snapshot; the running session is preserved.".to_string());
                    2
                }
            } else { 0 }
        };
        if deferred == 2 {
            *self.terminal_close_emitted.lock().unwrap() = true;
            let error = self.deferred_session_events.lock().unwrap().failure.clone();
            self.emit(AgentConnectionEvent::Closed { error }).await;
            return Ok(());
        }
        if deferred == 1 { return Ok(()); }
        if self.is_stale_sequenced_message(&message) {
            return Ok(());
        }
        self.observe_daemon_event_sequence(&message);

        let message_sequence = get_daemon_message_sequence(&message);
        match message {
            DaemonOutbound::SessionEvent { event, .. } => {
                if event.type_name() != "refine_complete" && event.type_name() != "refine_failed" {
                    self.observe_streaming_message(&event, message_sequence.is_some());
                }
                if let AgentConnectionSessionEvent::RlmChildUpdate { child } = &event {
                    {
                        let mut sequence = self.child_roster_sequence.lock().unwrap();
                        *sequence = max_event_sequence(*sequence, message_sequence);
                    }
                    self.observe_rlm_child_update(child.clone());
                }
                // Only locally represented events retain freshness. Metadata
                // transitions (retry, model, effort, etc.) need an authoritative
                // read, but per-token deltas never cause a history refetch.
                let cached = matches!(event.type_name(),
                    "message_start" | "message_update" | "message_end" |
                    "tool_execution_start" | "tool_execution_update" |
                    "agent_start" | "agent_end" | "compaction_start" |
                    "bash_start" | "bash_end" | "rlm_child_update");
                let paged_message_end = event.type_name() == "message_end"
                    && self.latest_snapshot.lock().unwrap().as_ref().is_some_and(|snapshot| snapshot.history.is_some());
                if !cached || paged_message_end {
                    *self.latest_snapshot_is_fresh.lock().unwrap() = false;
                }
                self.emit(AgentConnectionEvent::SessionEvent { event }).await;
                Ok(())
            }
            DaemonOutbound::SideQuestionEvent { event, .. } => {
                self.observe_side_question_event(&event);
                self.emit(AgentConnectionEvent::SideQuestionEvent { event }).await;
                Ok(())
            }
            DaemonOutbound::SessionStatus { recap, .. } => {
                // Keep a cached snapshot's recap current so a later re-attach seeds it.
                {
                    let mut guard = self.latest_snapshot.lock().unwrap();
                    if let Some(snapshot) = guard.as_mut() {
                        snapshot.state.recap = recap.clone();
                    }
                }
                self.emit(AgentConnectionEvent::SessionStatus { recap }).await;
                Ok(())
            }
            DaemonOutbound::SessionResynced { snapshot, .. } => {
                self.notify_before_session_invalidate();
                *self.attached_session_id.lock().unwrap() = Some(snapshot.state.session_id.clone());
                *self.attached_session_file.lock().unwrap() = snapshot.state.session_file.clone();
                let mut mapped = map_daemon_session_snapshot(&snapshot, None)?;
                if snapshot.children.is_some() {
                    *self.child_roster_sequence.lock().unwrap() = snapshot.last_event_sequence;
                }
                if let Some(sequence) = *self.last_event_sequence.lock().unwrap() {
                    mapped.last_event_sequence = Some(sequence as f64);
                }
                if let Some(cursor) = self.last_event_cursor.lock().unwrap().clone() {
                    mapped.last_event_cursor = Some(AgentConnectionEventCursor {
                        generation: cursor.generation,
                        sequence: cursor.sequence as f64,
                    });
                }
                *self.latest_snapshot.lock().unwrap() = Some(mapped.clone());
                *self.latest_snapshot_is_fresh.lock().unwrap() = true;
                self.emit(AgentConnectionEvent::SessionResynced { snapshot: mapped })
                    .await;
                Ok(())
            }
            DaemonOutbound::SessionReplaced {
                state,
                messages,
                snapshot_follows,
                ..
            } => {
                self.notify_before_session_invalidate();
                *self.attached_session_id.lock().unwrap() = Some(state.session_id.clone());
                *self.attached_session_file.lock().unwrap() = state.session_file.clone();
                if snapshot_follows.unwrap_or(false) {
                    *self.latest_snapshot_is_fresh.lock().unwrap() = false;
                    return Ok(());
                }
                let mut latest = AgentConnectionSnapshot {
                    state: state.clone(),
                    messages: messages.clone(),
                    ..Default::default()
                };
                if let Some(sequence) = *self.last_event_sequence.lock().unwrap() {
                    latest.last_event_sequence = Some(sequence as f64);
                }
                if let Some(cursor) = self.last_event_cursor.lock().unwrap().clone() {
                    latest.last_event_cursor = Some(AgentConnectionEventCursor {
                        generation: cursor.generation,
                        sequence: cursor.sequence as f64,
                    });
                }
                *self.latest_snapshot.lock().unwrap() = Some(latest);
                *self.child_roster_sequence.lock().unwrap() = None;
                *self.latest_snapshot_is_fresh.lock().unwrap() = true;
                self.emit(AgentConnectionEvent::SessionReplaced { state, messages })
                    .await;
                Ok(())
            }
            DaemonOutbound::ExtensionUiRequest {
                id, method, payload, ..
            } => {
                self.emit(AgentConnectionEvent::ExtensionUiRequest {
                    request: AgentConnectionExtensionUiRequest { id, method, payload },
                })
                .await;
                Ok(())
            }
            DaemonOutbound::ExtensionError {
                extension_path,
                event,
                error,
                ..
            } => {
                self.emit(AgentConnectionEvent::ExtensionError {
                    extension_path,
                    event,
                    error,
                })
                .await;
                Ok(())
            }
            DaemonOutbound::SessionClosed { reason, .. } => {
                if reason == "update" {
                    self.capture_daemon_log_path();
                    *self.update_restart_pending.lock().unwrap() = true;
                    self.reconnect_after_update();
                    return Ok(());
                }
                *self.terminal_close_emitted.lock().unwrap() = true;
                let message = self.format_daemon_session_closed_error(&reason);
                self.emit(AgentConnectionEvent::Closed { error: Some(message) }).await;
                Ok(())
            }
            DaemonOutbound::HeartbeatsChanged { .. }
            | DaemonOutbound::SessionSnapshotBegin { .. }
            | DaemonOutbound::SessionSnapshotChunk { .. }
            | DaemonOutbound::SessionSnapshotEnd { .. }
            | DaemonOutbound::SessionSnapshotFailed { .. } => Ok(()),
        }
    }

    async fn dispose_inner(&self) {
        if *self.disposed.lock().unwrap() || *self.disposing.lock().unwrap() {
            return;
        }
        *self.disposing.lock().unwrap() = true;
        if let Some(attempt) = self.reconnect_in_flight.lock().unwrap().as_ref() {
            attempt.cancel.cancel();
        }
        // `if (this.options.ownedSession && !this.client.isConnected && this.reconnectPromise)
        //      await Promise.race([this.reconnectPromise, delay(...)])`
        // (`daemon-agent-connection.ts:1594-1598`). Dispose must not tear the connection
        // down while a recovery attempt is still running.
        if self.options.lock().unwrap().owned_session && !self.client.is_connected() {
            let in_flight = self
                .reconnect_in_flight
                .lock()
                .unwrap()
                .as_ref()
                .map(|attempt| attempt.result.subscribe());
            if let Some(mut receiver) = in_flight {
                let _ = tokio::time::timeout(
                    Duration::from_millis(OWNED_SESSION_DISPOSE_RECONNECT_WAIT_MS),
                    async {
                        while receiver.borrow().is_none() {
                            if receiver.changed().await.is_err() { break; }
                        }
                    },
                )
                .await;
            }
        }
        *self.disposed.lock().unwrap() = true;
        *self.update_restart_pending.lock().unwrap() = false;
        let side_questions: Vec<String> = self.active_side_question_ids.lock().unwrap().iter().cloned().collect();
        for id in side_questions {
            let _ = self.abort_side_question_inner(&id).await;
        }
        let roster = self.roster_store.lock().unwrap().take();
        if let Some(store) = roster {
            store.dispose().await;
        }
        if let Some(unsubscribe) = self.unsubscribe_daemon_messages.lock().unwrap().take() {
            unsubscribe();
        }
        if let Some(unsubscribe) = self.unsubscribe_daemon_close.lock().unwrap().take() {
            unsubscribe();
        }
        let active_session_id = self.active_session_id();
        if self.options.lock().unwrap().owned_session {
            let _ = self
                .request_ok(command_body(
                    "complete_owned_session",
                    vec![("activeSessionId", Value::String(active_session_id))],
                ))
                .await;
        } else {
            let _ = self
                .request_ok(command_body(
                    "detach",
                    vec![("activeSessionId", Value::String(active_session_id))],
                ))
                .await;
        }
        if self.options.lock().unwrap().close_client_on_dispose {
            self.client.close();
        }
        self.reject_snapshot_assemblies("Daemon connection disposed during snapshot transfer".to_string());
        self_arc_registry()
            .lock()
            .unwrap()
            .retain(|_, connection| connection.client_id != self.client_id);
    }

    async fn drain_deferred_session_events(&self) -> Result<(), String> {
        self.drain_deferred_session_events_inner(false).await
    }

    async fn drain_deferred_session_events_inner(&self, release_initial: bool) -> Result<(), String> {
        let failure = {
            let pending = self.deferred_session_events.lock().unwrap();
            if release_initial { *self.defer_session_events.lock().unwrap() = false; }
            pending.failure.clone()
        };
        if let Some(error) = failure {
            self.emit(AgentConnectionEvent::Closed { error: Some(error.clone()) }).await;
            return Err(error);
        }
        {
            let mut pending = self.deferred_session_events.lock().unwrap();
            if release_initial { *self.defer_session_events.lock().unwrap() = false; }
            if pending.draining || *self.defer_session_events.lock().unwrap() { return Ok(()); }
            pending.draining = true;
        }
        loop {
            if *self.disposed.lock().unwrap() || *self.defer_session_events.lock().unwrap()
                || *self.attach_snapshot_pending.lock().unwrap() || self.snapshot_in_progress.lock().unwrap().is_some() {
                self.deferred_session_events.lock().unwrap().draining = false;
                return Ok(());
            }
            let event = {
                let mut pending = self.deferred_session_events.lock().unwrap();
                let event = pending.pop();
                if event.is_none() { pending.draining = false; }
                event
            };
            let Some(event) = event else { return Ok(()); };
            if let Err(error) = Box::pin(self.handle_daemon_message_inner(event, true)).await {
                self.deferred_session_events.lock().unwrap().draining = false;
                return Err(error);
            }
        }
    }

    async fn abort_side_question_inner(&self, id: &str) -> Result<bool, String> {
        let data = self
            .request_data(
                command_body(
                    "abort_side_question",
                    vec![
                        ("activeSessionId", Value::String(self.active_session_id())),
                        ("sideQuestionId", Value::String(id.to_string())),
                    ],
                ),
                None,
            )
            .await?;
        self.active_side_question_ids.lock().unwrap().remove(id);
        Ok(data.get("aborted").and_then(Value::as_bool).unwrap_or(false))
    }

    /// `withOwnedSessionPromotion(operation)`.
    async fn with_owned_session_promotion<T, F>(&self, operation: F) -> Result<T, String>
    where
        F: FnOnce(bool) -> BoxFuture<Result<T, String>>,
    {
        let _guard = self.owned_session_promotion_tail.lock().await;
        let promote_owned_session = self.options.lock().unwrap().owned_session;
        let result = operation(promote_owned_session).await?;
        if promote_owned_session {
            self.options.lock().unwrap().owned_session = false;
        }
        Ok(result)
    }
}

impl DaemonAgentConnection {
    /// `promptWithAdmissionCancellation` shared by `prompt` and `prompt_and_wait`.
    fn prompt_with_admission_cancellation(
        &self,
        type_: &'static str,
        message: &str,
        options: Option<AgentConnectionPromptOptions>,
    ) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let active_session_id = this.active_session_id();
        let message = message.to_string();
        Box::pin(async move {
            let mut fields: Vec<(&str, Value)> = vec![
                ("activeSessionId", Value::String(active_session_id)),
                ("message", Value::String(message)),
            ];
            if let Some(options) = &options {
                if let Some(images) = &options.images {
                    fields.push(("images", serde_json::to_value(images).unwrap_or(Value::Null)));
                }
                if let Some(streaming_behavior) = &options.streaming_behavior {
                    fields.push(("streamingBehavior", Value::String(streaming_behavior.clone())));
                }
                if let Some(queue_if_busy) = options.queue_if_busy {
                    fields.push(("queueIfBusy", Value::Bool(queue_if_busy)));
                }
                if let Some(source) = &options.source {
                    fields.push(("source", Value::String(source.clone())));
                }
            }
            let command = command_body(type_, fields);
            this.request_data(command, Some(DAEMON_LONG_RUNNING_REQUEST_TIMEOUT_MS))
                .await
                .map(|_| ())
        })
    }
}

impl AgentConnection for DaemonAgentConnection {
    fn subscribe(&self, listener: AgentConnectionEventListener) -> Box<dyn Fn() + Send + Sync> {
        let this = self.clone();
        this.listeners.lock().unwrap().push(listener.clone());
        let was_deferred = *this.defer_session_events.lock().unwrap();
        if was_deferred {
            let connection = this.clone();
            tokio::spawn(async move {
                // Keep the queue locked until deferral is released and its drain
                // takes ownership; newer transport frames cannot overtake it.
                let _ = connection.drain_deferred_session_events_inner(true).await;
            });
        }
        let listeners = this.listeners.clone();
        Box::new(move || {
            let mut guard = listeners.lock().unwrap();
            if let Some(index) = guard.iter().position(|entry| Arc::ptr_eq(entry, &listener)) {
                guard.remove(index);
            }
        })
    }

    fn on_before_session_invalidate(
        &self,
        listener: AgentConnectionBeforeSessionInvalidateListener,
    ) -> Box<dyn Fn() + Send + Sync> {
        let this = self.clone();
        this.before_session_invalidate_listeners
            .lock()
            .unwrap()
            .push(listener.clone());
        let listeners = this.before_session_invalidate_listeners.clone();
        Box::new(move || {
            let mut guard = listeners.lock().unwrap();
            if let Some(index) = guard.iter().position(|entry| Arc::ptr_eq(entry, &listener)) {
                guard.remove(index);
            }
        })
    }

    fn get_state(&self) -> BoxFuture<Result<AgentConnectionState, String>> {
        let this = self.clone();
        let opening_error = this.deferred_session_events.lock().unwrap().failure.clone();
        if let Some(error) = opening_error { return Box::pin(async move { Err(error) }); }
        if *this.latest_snapshot_is_fresh.lock().unwrap() {
            if let Some(snapshot) = this.latest_snapshot.lock().unwrap().clone() {
                return Box::pin(async move { Ok(snapshot.state) });
            }
        }
        let command = command_body(
            "get_connection_state",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            serde_json::from_value(data).map_err(|error| format!("Daemon returned an invalid connection state: {error}"))
        })
    }

    fn supports_jev_features(&self) -> bool {
        self.client.supports_server_capability("jev_features")
    }

    fn get_jev_status(&self) -> BoxFuture<Result<Option<Value>, String>> {
        let this = self.clone();
        Box::pin(async move {
            if !this.client.supports_server_capability("jev_control") {
                return Ok(None);
            }
            let active_session_id = this.active_session_id();
            let command = command_body(
                "jev_get_status",
                vec![("activeSessionId", Value::String(active_session_id.clone()))],
            );
            // Status must not be replayed against a different worker/session
            // after reconnect, or force a fresh (potentially large) transcript.
            let data = this.request_data_with_recovery(command, Some(2_000), false).await?;
            if this.active_session_id() != active_session_id {
                return Err("Session changed while reading Jev status".to_string());
            }
            match data.get("pipeline") {
                Some(pipeline) if pipeline.is_object() || pipeline.is_null() => {
                    Ok(Some(json!({ "pipeline": pipeline })))
                }
                _ => Err("Daemon returned an invalid Jev status response".to_string()),
            }
        })
    }

    fn get_initial_snapshot(&self) -> BoxFuture<Result<AgentConnectionSnapshot, String>> {
        let this = self.clone();
        Box::pin(async move { this.get_initial_snapshot_inner(true).await })
    }

    fn get_rlm_child_snapshots(&self) -> BoxFuture<Result<Vec<AgentConnectionRlmChildAgentSnapshot>, String>> {
        let this = self.clone();
        if !this.client.supports_server_capability("authoritative_child_roster") {
            return Box::pin(async {
                Err(DaemonCapabilityUnavailableError::new("get_rlm_children", "authoritative_child_roster").to_string())
            });
        }
        let command = command_body(
            "get_rlm_children",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            let children: Vec<AgentConnectionRlmChildAgentSnapshot> = data
                .get("children")
                .cloned()
                .and_then(|children| serde_json::from_value(children).ok())
                .unwrap_or_default();
            let event_sequence = match data.get("eventSequence").and_then(Value::as_i64) {
                Some(event_sequence) => event_sequence,
                None => return Err("Daemon returned an invalid child roster".to_string()),
            };
            if this.child_roster_sequence.lock().unwrap().unwrap_or(-1) > event_sequence {
                return Ok(this
                    .latest_snapshot
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|snapshot| snapshot.children.clone())
                    .unwrap_or(children));
            }
            *this.child_roster_sequence.lock().unwrap() = Some(event_sequence);
            {
                let mut guard = this.latest_snapshot.lock().unwrap();
                if let Some(snapshot) = guard.as_mut() {
                    snapshot.children = Some(children.clone());
                }
            }
            Ok(children)
        })
    }

    fn get_messages(&self) -> BoxFuture<Result<Vec<AgentMessage>, String>> {
        let this = self.clone();
        if *this.latest_snapshot_is_fresh.lock().unwrap() {
            if let Some(snapshot) = this.latest_snapshot.lock().unwrap().clone() {
                if snapshot.history.is_none() {
                    return Box::pin(async move { Ok(snapshot.messages) });
                }
            }
        }
        let command = command_body(
            "get_messages",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data
                .get("messages")
                .cloned()
                .and_then(|messages| serde_json::from_value(messages).ok())
                .unwrap_or_default())
        })
    }

    fn get_history_range(
        &self,
        request: AgentConnectionHistoryRangeRequest,
    ) -> BoxFuture<Result<AgentConnectionHistoryRange, String>> {
        let this = self.clone();
        if !this.client.supports_server_capability("history_ranges") {
            return Box::pin(async {
                Err(DaemonCapabilityUnavailableError::new("get_history_range", "history_ranges").to_string())
            });
        }
        let mut fields = vec![
            ("activeSessionId", Value::String(this.active_session_id())),
            ("generation", Value::String(request.generation.clone())),
            ("representation", Value::String(request.representation.clone())),
            (
                "tipEntryId",
                match &request.tip_entry_id {
                    Some(tip_entry_id) => Value::String(tip_entry_id.clone()),
                    None => Value::Null,
                },
            ),
        ];
        if let Some(before_entry_id) = &request.before_entry_id {
            fields.push(("beforeEntryId", Value::String(before_entry_id.clone())));
        }
        if let Some(limit) = request.limit {
            fields.push(("limit", json!(limit)));
        }
        let command = command_body("get_history_range", fields);
        Box::pin(async move {
            let data = this.request_data_with_recovery(command, None, false).await?;
            let range: AgentConnectionHistoryRange = serde_json::from_value(data)
                .map_err(|_| "Daemon returned an invalid session history range".to_string())?;
            let entry_ids_unique = {
                let mut seen: HashSet<&String> = HashSet::new();
                range.window.entry_ids.iter().all(|entry_id| seen.insert(entry_id))
            };
            let valid = range.window.version == 1.0
                && range.window.generation == request.generation
                && range.window.representation == request.representation
                && range.window.tip_entry_id == request.tip_entry_id
                && range.window.order == "chronological"
                && range.messages.len() == range.window.entry_ids.len()
                && entry_ids_unique
                && is_safe_integer(range.window.start_index)
                && range.window.start_index >= 0.0
                && is_safe_integer(range.window.total_message_count)
                && range.window.total_message_count >= 0.0
                && range.window.start_index + range.messages.len() as f64 <= range.window.total_message_count
                && range.window.has_older == (range.window.start_index > 0.0);
            if !valid {
                return Err("Daemon returned an invalid session history range".to_string());
            }
            Ok(range)
        })
    }

    fn get_session_header(&self) -> BoxFuture<Result<Option<AgentConnectionSessionHeader>, String>> {
        let this = self.clone();
        let command = command_body(
            "get_session_header",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data
                .get("header")
                .cloned()
                .and_then(|header| serde_json::from_value(header).ok()))
        })
    }

    fn get_commands(&self) -> BoxFuture<Result<Vec<AgentConnectionSlashCommand>, String>> {
        let this = self.clone();
        let command = command_body(
            "get_commands",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data
                .get("commands")
                .cloned()
                .and_then(|commands| serde_json::from_value(commands).ok())
                .unwrap_or_default())
        })
    }

    fn get_resource_snapshot(&self) -> BoxFuture<Result<AgentConnectionResourceSnapshot, String>> {
        let this = self.clone();
        let command = command_body(
            "get_resource_snapshot",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            serde_json::from_value(data).map_err(|error| format!("Daemon returned an invalid resource snapshot: {error}"))
        })
    }

    fn supports_acp_mcp_servers(&self) -> bool {
        let this = self.clone();
        this.client.supports_server_capability("acp_mcp_servers")
    }

    fn replace_acp_mcp_servers(
        &self,
        servers: Vec<Value>,
        owner_id: &str,
    ) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        if !this.supports_acp_mcp_servers() {
            return Box::pin(async {
                Err(DaemonCapabilityUnavailableError::new("replace_acp_mcp_servers", "acp_mcp_servers").to_string())
            });
        }
        let command = command_body(
            "replace_acp_mcp_servers",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("ownerId", Value::String(owner_id.to_string())),
                ("servers", Value::Array(servers)),
            ],
        );
        Box::pin(async move { this.request_ok(command).await })
    }

    fn release_acp_mcp_servers(
        &self,
        owner_id: &str,
        _server_names: Vec<String>,
    ) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let owner_id = owner_id.to_string();
        Box::pin(async move { this.replace_acp_mcp_servers(Vec::new(), &owner_id).await })
    }

    fn get_available_models(&self) -> BoxFuture<Result<Vec<AgentConnectionModel>, String>> {
        let this = self.clone();
        let command = command_body(
            "get_available_models",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data
                .get("models")
                .cloned()
                .and_then(|models| serde_json::from_value(models).ok())
                .unwrap_or_default())
        })
    }

    fn get_model_catalog(&self) -> BoxFuture<Result<AgentConnectionModelCatalog, String>> {
        let this = self.clone();
        if !this.client.supports_server_capability("model_catalog") {
            let models_future = this.get_available_models();
            return Box::pin(async move {
                let models = models_future.await?;
                let mut configured: IndexMap<String, ()> = IndexMap::new();
                for model in &models {
                    configured.insert(model.provider.clone(), ());
                }
                Ok(AgentConnectionModelCatalog {
                    models,
                    configured_providers: configured.keys().cloned().collect(),
                })
            });
        }
        let command = command_body(
            "get_model_catalog",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            serde_json::from_value(data).map_err(|error| format!("Daemon returned an invalid model catalog: {error}"))
        })
    }

    fn get_session_stats(&self) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let command = command_body(
            "get_session_stats",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_data(command, None).await })
    }

    fn get_context_tree(&self) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let command = command_body(
            "get_context_tree",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_data(command, None).await })
    }

    fn get_session_context(&self) -> BoxFuture<Result<AgentConnectionSessionContext, String>> {
        let this = self.clone();
        if *this.latest_snapshot_is_fresh.lock().unwrap() {
            if let Some(snapshot) = this.latest_snapshot.lock().unwrap().clone() {
                if let Some(context) = snapshot.session_context {
                    return Box::pin(async move { Ok(context) });
                }
            }
        }
        let command = command_body(
            "get_session_context",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            data.get("context")
                .cloned()
                .and_then(|context| serde_json::from_value(context).ok())
                .ok_or_else(|| "Daemon returned an invalid session context".to_string())
        })
    }

    fn get_session_tree(&self) -> BoxFuture<Result<AgentConnectionWatchSessionTree, String>> {
        let this = self.clone();
        if *this.latest_snapshot_is_fresh.lock().unwrap() {
            if let Some(snapshot) = this.latest_snapshot.lock().unwrap().clone() {
                if let Some(tree) = snapshot.session_tree {
                    return Box::pin(async move {
                        Ok(AgentConnectionWatchSessionTree {
                            tree: tree.tree,
                            leaf_id: tree.leaf_id,
                        })
                    });
                }
            }
        }
        let command = command_body(
            "get_session_tree",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            parse_session_tree_response(data)
        })
    }

    fn list_saved_sessions(
        &self,
        scope: &str,
    ) -> BoxFuture<Result<Vec<AgentConnectionSavedSessionInfo>, String>> {
        let this = self.clone();
        // `listDaemonSavedSessions(client, target, scope, callbacks)` lives in
        // modes/daemon/saved-session-catalog.ts (another slice).
        let scope = scope.to_string();
        let _ = scope;
        Box::pin(async {
            Err("listSavedSessions requires modes/daemon/saved-session-catalog.ts (not ported in this slice)".to_string())
        })
    }

    fn get_queue(&self) -> BoxFuture<Result<AgentConnectionQueueState, String>> {
        let this = self.clone();
        let command = command_body(
            "get_queue",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            serde_json::from_value(data).map_err(|error| format!("Daemon returned an invalid queue state: {error}"))
        })
    }

    fn mutate_queued_message(
        &self,
        lane: &str,
        index: i64,
        expected_text: &str,
        mutation: Value,
    ) -> BoxFuture<Result<String, String>> {
        let this = self.clone();
        if !this.client.supports_server_capability("queue_message_mutation") {
            return Box::pin(async { Ok("unsupported".to_string()) });
        }
        let command = command_body(
            "mutate_queued_message",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("lane", Value::String(lane.to_string())),
                ("index", json!(index)),
                ("expectedText", Value::String(expected_text.to_string())),
                ("mutation", mutation),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string())
        })
    }

    fn clear_queue(&self) -> BoxFuture<Result<AgentConnectionQueueState, String>> {
        let this = self.clone();
        let command = command_body(
            "clear_queue",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            serde_json::from_value(data).map_err(|error| format!("Daemon returned an invalid queue state: {error}"))
        })
    }

    fn abort_and_clear_queue(&self) -> BoxFuture<Result<AgentConnectionQueueState, String>> {
        let this = self.clone();
        let command = command_body(
            "abort_and_clear_queue",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            match this.request_data(command, None).await {
                Ok(data) => serde_json::from_value(data)
                    .map_err(|error| format!("Daemon returned an invalid queue state: {error}")),
                Err(error) => {
                    if is_unknown_daemon_command_error(&error, "abort_and_clear_queue") {
                        Err("the daemon is running an older build; restart the daemon and try again".to_string())
                    } else {
                        Err(error)
                    }
                }
            }
        })
    }

    fn acquire_session_input_pause(
        &self,
        lease_key: &str,
    ) -> BoxFuture<Result<AgentConnectionSessionInputPause, String>> {
        let this = self.clone();
        if *this.terminal_close_emitted.lock().unwrap() {
            return Box::pin(async { Err("Daemon connection is closed; cannot acquire an input pause.".to_string()) });
        }
        let active_session_id = this.active_session_id();
        let generation = *this.session_input_pause_generation.lock().unwrap();
        let acquisition_key = format!("{active_session_id}\u{1}{lease_key}");
        if let Some(existing) = this.session_input_pauses.lock().unwrap().get(&acquisition_key).cloned() {
            return Box::pin(async move { Ok(existing) });
        }
        let command = command_body(
            "acquire_session_input_pause",
            vec![
                ("activeSessionId", Value::String(active_session_id.clone())),
                ("leaseKey", Value::String(lease_key.to_string())),
            ],
        );
        let release_command = move |pause_id: String| {
            command_body(
                "release_session_input_pause",
                vec![
                    ("activeSessionId", Value::String(active_session_id.clone())),
                    ("pauseId", Value::String(pause_id)),
                ],
            )
        };
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            let pause_id = data
                .get("pauseId")
                .and_then(Value::as_str)
                .ok_or_else(|| "Daemon returned an invalid input pause".to_string())?
                .to_string();
            if generation != *this.session_input_pause_generation.lock().unwrap()
                || *this.terminal_close_emitted.lock().unwrap()
            {
                let _ = this.request_data(release_command(pause_id), None).await;
                return Err("Session input pause acquisition was invalidated by a daemon reconnect.".to_string());
            }
            let connection = this.self_arc();
            let key = acquisition_key.clone();
            let generation_check = generation;
            struct DaemonInputPause {
                connection: Option<Arc<DaemonAgentConnection>>,
                pause_id: String,
                generation: u64,
                key: String,
                released: Arc<Mutex<bool>>,
            }
            impl AgentConnectionInputPause for DaemonInputPause {
                fn release(&self) -> BoxFuture<Result<(), String>> {
                    if *self.released.lock().unwrap() {
                        return Box::pin(async { Ok(()) });
                    }
                    let connection = self.connection.clone();
                    let pause_id = self.pause_id.clone();
                    let generation = self.generation;
                    let key = self.key.clone();
                    let released = self.released.clone();
                    Box::pin(async move {
                        let Some(connection) = connection else {
                            return Ok(());
                        };
                        if generation != *connection.session_input_pause_generation.lock().unwrap() {
                            return Err("Session input pause was invalidated by a daemon reconnect.".to_string());
                        }
                        connection
                            .request_data(
                                command_body(
                                    "release_session_input_pause",
                                    vec![
                                        ("activeSessionId", Value::String(connection.active_session_id())),
                                        ("pauseId", Value::String(pause_id)),
                                    ],
                                ),
                                None,
                            )
                            .await?;
                        *released.lock().unwrap() = true;
                        connection.session_input_pauses.lock().unwrap().remove(&key);
                        Ok(())
                    })
                }
            }
            let pause: AgentConnectionSessionInputPause = Arc::new(DaemonInputPause {
                connection,
                pause_id,
                generation: generation_check,
                key: key.clone(),
                released: Arc::new(Mutex::new(false)),
            });
            this.session_input_pauses
                .lock()
                .unwrap()
                .insert(key, pause.clone());
            Ok(pause)
        })
    }

    fn list_cron_jobs(&self, include_inactive: bool) -> BoxFuture<Result<Vec<Value>, String>> {
        let this = self.clone();
        let mut fields = vec![("activeSessionId", Value::String(this.active_session_id()))];
        if include_inactive {
            fields.push(("includeInactive", Value::Bool(true)));
        }
        let command = command_body("cron_list", fields);
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data
                .get("jobs")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default())
        })
    }

    fn list_heartbeats(&self) -> BoxFuture<Result<Vec<AgentConnectionHeartbeat>, String>> {
        let this = self.clone();
        // `listDaemonHeartbeats(client, this.options.ownedSession ? this.activeSessionId : undefined)`
        // (`daemon-agent-connection.ts:843`): only an OWNED session scopes the catalog to
        // its own active session id; a shared control-plane connection sends none.
        let active_session_id = this
            .options
            .lock()
            .unwrap()
            .owned_session
            .then(|| Value::String(this.active_session_id()));
        let command = command_body(
            "heartbeats_list",
            vec![("activeSessionId", active_session_id.unwrap_or(Value::Null))],
        );
        Box::pin(async move {
            // `await client.waitForHello(...)` first (`heartbeat-catalog.ts:12`): a
            // connection whose hello has not arrived yet must wait and then decide,
            // not deterministically report an empty catalog.
            if this.client.hello_socket_path().is_none() {
                this.client.wait_for_hello(3000).await?;
            }
            // `if (!client.supportsServerCapability("heartbeat_catalog")) return []`
            // (`heartbeat-catalog.ts:13`): an older daemon is an empty catalog, not an
            // error every caller has to swallow.
            if !this.client.supports_server_capability("heartbeat_catalog") {
                return Ok(Vec::new());
            }
            match this.request_data(command, None).await {
                Ok(data) => Ok(data
                    .get("heartbeats")
                    .and_then(Value::as_array)
                    .map(|entries| {
                        entries
                            .iter()
                            .filter_map(|entry| serde_json::from_value(entry.clone()).ok())
                            .collect::<Vec<AgentConnectionHeartbeat>>()
                    })
                    .unwrap_or_default()),
                // `catch (error) { if (isUnknownDaemonCommandError(error, "heartbeats_list")) return []; throw error; }`
                // (`heartbeat-catalog.ts:21-25`).
                Err(error) => {
                    if is_unknown_daemon_command_error(&error, "heartbeats_list") {
                        Ok(Vec::new())
                    } else {
                        Err(error)
                    }
                }
            }
        })
    }

    fn manage_heartbeat(
        &self,
        active_session_id: &str,
        job_id: &str,
        action: Value,
    ) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        if !this.client.supports_server_capability("heartbeat_management") {
            return Box::pin(async {
                Err("Heartbeat management requires a newer Prime Agent daemon.".to_string())
            });
        }
        let command = command_body(
            "heartbeat_manage",
            vec![
                ("activeSessionId", Value::String(active_session_id.to_string())),
                ("jobId", Value::String(job_id.to_string())),
                ("action", action),
            ],
        );
        Box::pin(async move {
            match this.request_data(command, None).await {
                Ok(data) => data
                    .get("heartbeat")
                    .cloned()
                    .ok_or_else(|| "Daemon returned an invalid heartbeat".to_string()),
                Err(error) => {
                    if is_unknown_daemon_command_error(&error, "heartbeat_manage") {
                        Err("Heartbeat management requires a newer Prime Agent daemon.".to_string())
                    } else {
                        Err(error)
                    }
                }
            }
        })
    }

    fn add_cron_job(&self, schedule: &str, prompt: &str) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let command = command_body(
            "cron_add",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("schedule", Value::String(schedule.to_string())),
                ("prompt", Value::String(prompt.to_string())),
            ],
        );
        Box::pin(async move {
            this.with_owned_session_promotion(|promote_owned_session| {
                let this = this.clone();
                let mut command = command.clone();
                if promote_owned_session {
                    if let Some(object) = command.as_object_mut() {
                        object.insert("promoteOwnedSession".to_string(), Value::Bool(true));
                    }
                }
                Box::pin(async move {
                    let data = this.request_data(command, None).await?;
                    data.get("job")
                        .cloned()
                        .ok_or_else(|| "Daemon returned an invalid cron job".to_string())
                })
            })
            .await
        })
    }

    fn cancel_cron_job(&self, job_id: &str) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let command = command_body(
            "cron_cancel",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("jobId", Value::String(job_id.to_string())),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            data.get("job")
                .cloned()
                .ok_or_else(|| "Daemon returned an invalid cron job".to_string())
        })
    }

    fn get_heartbeat(&self) -> BoxFuture<Result<Option<Value>, String>> {
        let this = self.clone();
        let command = command_body(
            "heartbeat_get",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data.get("heartbeat").cloned().filter(|value| !value.is_null()))
        })
    }

    fn set_heartbeat(
        &self,
        schedule: &str,
        instruction: &str,
        delivery_mode: Option<&str>,
    ) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let mut fields = vec![
            ("activeSessionId", Value::String(this.active_session_id())),
            ("schedule", Value::String(schedule.to_string())),
            ("prompt", Value::String(instruction.to_string())),
        ];
        if let Some(delivery_mode) = delivery_mode {
            fields.push(("deliveryMode", Value::String(delivery_mode.to_string())));
        }
        let command = command_body("heartbeat_set", fields);
        Box::pin(async move {
            this.with_owned_session_promotion(|promote_owned_session| {
                let this = this.clone();
                let mut command = command.clone();
                if promote_owned_session {
                    if let Some(object) = command.as_object_mut() {
                        object.insert("promoteOwnedSession".to_string(), Value::Bool(true));
                    }
                }
                Box::pin(async move {
                    let data = this.request_data(command, None).await?;
                    data.get("heartbeat")
                        .cloned()
                        .ok_or_else(|| "Daemon returned an invalid heartbeat".to_string())
                })
            })
            .await
        })
    }

    fn update_heartbeat(&self, action: Value) -> BoxFuture<Result<Option<Value>, String>> {
        let this = self.clone();
        let command = command_body(
            "heartbeat_update",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("action", action),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data.get("heartbeat").cloned().filter(|value| !value.is_null()))
        })
    }

    fn send_agent_message(&self, target_active_session_id: &str, message: &str) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let command = command_body(
            "send_message",
            vec![
                ("targetActiveSessionId", Value::String(target_active_session_id.to_string())),
                ("message", Value::String(message.to_string())),
                ("fromActiveSessionId", Value::String(this.active_session_id())),
            ],
        );
        Box::pin(async move { this.request_data(command, None).await })
    }

    fn get_agent_message_status(&self) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let command = command_body(
            "agent_messages_status",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_data(command, None).await })
    }

    fn pause_agent_messages(&self) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let command = command_body(
            "agent_messages_pause",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_data(command, None).await })
    }

    fn resume_agent_messages(&self) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let command = command_body(
            "agent_messages_resume",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_data(command, None).await })
    }

    fn clear_agent_messages(&self) -> BoxFuture<Result<f64, String>> {
        let this = self.clone();
        let command = command_body(
            "agent_messages_clear",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data.as_f64().unwrap_or(0.0))
        })
    }

    fn get_user_messages_for_forking(&self) -> BoxFuture<Result<Vec<AgentConnectionUserMessage>, String>> {
        let this = self.clone();
        let command = command_body(
            "get_user_messages_for_forking",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data
                .get("messages")
                .cloned()
                .and_then(|messages| serde_json::from_value(messages).ok())
                .unwrap_or_default())
        })
    }

    fn get_last_assistant_text(&self) -> BoxFuture<Result<Option<String>, String>> {
        let this = self.clone();
        let command = command_body(
            "get_last_assistant_text",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data.get("text").and_then(Value::as_str).map(str::to_string))
        })
    }

    fn get_system_prompt(&self) -> BoxFuture<Result<String, String>> {
        let this = self.clone();
        let command = command_body(
            "get_system_prompt",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data
                .get("systemPrompt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string())
        })
    }

    fn get_tool_definition(&self, name: &str) -> BoxFuture<Result<Option<AgentConnectionToolDefinition>, String>> {
        let this = self.clone();
        let command = command_body(
            "get_tool_definition",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("name", Value::String(name.to_string())),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(command, None).await?;
            Ok(data
                .get("toolDefinition")
                .cloned()
                .and_then(|definition| serde_json::from_value(definition).ok()))
        })
    }

    fn set_session_entry_label(&self, entry_id: &str, label: Option<&str>) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let command = command_body(
            "set_session_entry_label",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("entryId", Value::String(entry_id.to_string())),
                ("label", optional_string(label)),
            ],
        );
        Box::pin(async move { this.request_ok(command).await })
    }

    fn respond_to_extension_ui_request(
        &self,
        request_id: &str,
        response: AgentConnectionExtensionUiResponse,
    ) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let command = command_body(
            "extension_ui_response",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("requestId", Value::String(request_id.to_string())),
                (
                    "response",
                    serde_json::to_value(response).unwrap_or(Value::Null),
                ),
            ],
        );
        Box::pin(async move { this.request_ok(command).await })
    }

    fn subscribe_agent_roster(
        &self,
        listener: Arc<dyn Fn() + Send + Sync>,
    ) -> BoxFuture<Result<Arc<dyn AgentConnectionRosterStore>, String>> {
        let this = self.clone();
        let store = {
            let mut slot = this.roster_store.lock().unwrap();
            slot.get_or_insert_with(|| Arc::new(roster_subscription::RosterSubscription::new())).clone()
        };
        Box::pin(async move {
            if !store.attach(this.client.clone()).await? { return Err(STALE_ROSTER_DAEMON_MESSAGE.to_string()); }
            let _ = store.on_update(listener);
            Ok(store)
        })
    }
    fn prompt(&self, message: &str, options: Option<AgentConnectionPromptOptions>) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        this.prompt_with_admission_cancellation("prompt", message, options)
    }

    fn prompt_and_wait(
        &self,
        message: &str,
        options: Option<AgentConnectionPromptOptions>,
    ) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        this.prompt_with_admission_cancellation("prompt_and_wait", message, options)
    }

    fn start_side_question(
        &self,
        id: &str,
        question: &str,
        previous_turns: Option<Vec<AgentConnectionSideQuestionTurn>>,
    ) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let has_turns = previous_turns.as_ref().map(|turns| !turns.is_empty()).unwrap_or(false);
        if has_turns && !this.client.supports_server_capability("side_question_transcript") {
            // An older daemon would silently ignore previousTurns and answer the
            // follow-up without the side-conversation context; fail loudly instead.
            return Box::pin(async {
                Err("the daemon is running an older build without side-conversation follow-ups; restart the daemon and try again".to_string())
            });
        }
        let command = command_body(
            "start_side_question",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("sideQuestionId", Value::String(id.to_string())),
                ("question", Value::String(question.to_string())),
                (
                    "previousTurns",
                    match previous_turns {
                        Some(turns) => serde_json::to_value(turns).unwrap_or(Value::Null),
                        None => Value::Null,
                    },
                ),
            ],
        );
        let id = id.to_string();
        Box::pin(async move {
            this.active_side_question_ids.lock().unwrap().insert(id.clone());
            match this.request_ok(command).await {
                Ok(()) => Ok(()),
                Err(error) => {
                    this.active_side_question_ids.lock().unwrap().remove(&id);
                    if is_unknown_daemon_command_error(&error, "start_side_question") {
                        Err("the daemon is running an older build; restart the daemon and try again".to_string())
                    } else {
                        Err(error)
                    }
                }
            }
        })
    }

    fn abort_side_question(&self, id: &str) -> BoxFuture<Result<bool, String>> {
        let this = self.clone();
        let id = id.to_string();
        Box::pin(async move { this.abort_side_question_inner(&id).await })
    }

    fn steer(&self, message: &str, images: Option<Vec<ImageContent>>) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let command = command_body(
            "steer",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("message", Value::String(message.to_string())),
                (
                    "images",
                    match images {
                        Some(images) => serde_json::to_value(images).unwrap_or(Value::Null),
                        None => Value::Null,
                    },
                ),
            ],
        );
        Box::pin(async move { this.request_ok(command).await })
    }

    fn follow_up(&self, message: &str, images: Option<Vec<ImageContent>>) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let command = command_body(
            "follow_up",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("message", Value::String(message.to_string())),
                (
                    "images",
                    match images {
                        Some(images) => serde_json::to_value(images).unwrap_or(Value::Null),
                        None => Value::Null,
                    },
                ),
            ],
        );
        Box::pin(async move { this.request_ok(command).await })
    }

    fn abort(&self) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let command = command_body(
            "abort",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_ok(command).await })
    }

    fn cancel_rlm_child(&self, child_id: &str) -> BoxFuture<Result<bool, String>> {
        let this = self.clone();
        let command = command_body(
            "cancel_rlm_child",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("childId", Value::String(child_id.to_string())),
            ],
        );
        Box::pin(async move {
            match this.request_data(command, None).await {
                Ok(data) => Ok(data.get("cancelled").and_then(Value::as_bool).unwrap_or(false)),
                Err(error) => {
                    if is_unknown_daemon_command_error(&error, "cancel_rlm_child") {
                        Err("the daemon is running an older build; restart the daemon and try again".to_string())
                    } else {
                        Err(error)
                    }
                }
            }
        })
    }

    fn wait_for_idle(&self) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let command = command_body(
            "wait_for_idle",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            this.request_data(command, Some(DAEMON_LONG_RUNNING_REQUEST_TIMEOUT_MS))
                .await
                .map(|_| ())
        })
    }

    fn wait_for_headless_completion(
        &self,
        options: Option<AgentConnectionHeadlessCompletionOptions>,
    ) -> BoxFuture<Result<AgentAutonomousStatus, String>> {
        let this = self.clone();
        let wait_for_rlm_quiescence = options
            .as_ref()
            .and_then(|options| options.wait_for_rlm_quiescence)
            .unwrap_or(false);
        if wait_for_rlm_quiescence && !this.client.supports_server_capability("rlm_quiescence_barrier") {
            return Box::pin(async {
                Err("the daemon is running an older build without RLM quiescence barriers; restart the daemon and try again".to_string())
            });
        }
        let mut fields = vec![("activeSessionId", Value::String(this.active_session_id()))];
        if wait_for_rlm_quiescence {
            fields.push(("waitForRlmQuiescence", Value::Bool(true)));
        }
        let command = command_body("wait_for_headless_completion", fields);
        Box::pin(async move {
            let data = this
                .request_data(command, Some(DAEMON_LONG_RUNNING_REQUEST_TIMEOUT_MS))
                .await?;
            serde_json::from_value(data)
                .map_err(|error| format!("Daemon returned an invalid autonomous status: {error}"))
        })
    }

    fn execute_bash(
        &self,
        command: &str,
        options: Option<AgentConnectionExecuteBashOptions>,
    ) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let transient = options.as_ref().and_then(|options| options.transient).unwrap_or(false);
        if transient && !this.client.supports_server_capability("transient_bash") {
            // An older daemon would record the run into the session, leaking the
            // side conversation into the main transcript; fail loudly instead.
            return Box::pin(async {
                Err("the daemon is running an older build without side-conversation bash; restart the daemon and try again".to_string())
            });
        }
        let mut fields = vec![
            ("activeSessionId", Value::String(this.active_session_id())),
            ("command", Value::String(command.to_string())),
        ];
        if let Some(options) = &options {
            if let Some(exclude_from_context) = options.exclude_from_context {
                fields.push(("excludeFromContext", Value::Bool(exclude_from_context)));
            }
            if let Some(transient) = options.transient {
                fields.push(("transient", Value::Bool(transient)));
            }
            if let Some(run_id) = &options.run_id {
                fields.push(("runId", Value::String(run_id.clone())));
            }
        }
        let request = command_body("execute_bash", fields);
        Box::pin(async move {
            match this.request_ok(request).await {
                Ok(()) => Ok(()),
                Err(error) => {
                    if is_unknown_daemon_command_error(&error, "execute_bash") {
                        Err("the daemon is running an older build; restart the daemon and try again".to_string())
                    } else {
                        Err(error)
                    }
                }
            }
        })
    }

    fn execute_bash_and_wait(&self, command: &str) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let request = command_body(
            "execute_bash_and_wait",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("command", Value::String(command.to_string())),
            ],
        );
        Box::pin(async move {
            this.request_data(request, Some(DAEMON_LONG_RUNNING_REQUEST_TIMEOUT_MS))
                .await
        })
    }

    fn abort_bash(&self) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "abort_bash",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            match this.request_ok(request).await {
                Ok(()) => Ok(()),
                Err(error) => {
                    if is_unknown_daemon_command_error(&error, "abort_bash") {
                        Err("the daemon is running an older build; restart the daemon and try again".to_string())
                    } else {
                        Err(error)
                    }
                }
            }
        })
    }

    fn set_model(&self, provider: &str, model_id: &str) -> BoxFuture<Result<AgentConnectionModel, String>> {
        let this = self.clone();
        let request = command_body(
            "set_model",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("provider", Value::String(provider.to_string())),
                ("modelId", Value::String(model_id.to_string())),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(request, None).await?;
            serde_json::from_value(data).map_err(|error| format!("Daemon returned an invalid model: {error}"))
        })
    }

    fn cycle_model(
        &self,
        direction: Option<&str>,
    ) -> BoxFuture<Result<Option<AgentConnectionModelCycleResult>, String>> {
        let this = self.clone();
        let request = command_body(
            "cycle_model",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("direction", optional_string(direction)),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(request, None).await?;
            if data.is_null() {
                return Ok(None);
            }
            serde_json::from_value(data)
                .map(Some)
                .map_err(|error| format!("Daemon returned an invalid model cycle result: {error}"))
        })
    }

    fn set_scoped_models(&self, scoped_models: Vec<AgentConnectionScopedModel>) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "set_scoped_models",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                (
                    "scopedModels",
                    serde_json::to_value(scoped_models).unwrap_or(Value::Null),
                ),
            ],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn set_thinking_level(&self, level: ThinkingLevel) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "set_thinking_level",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("level", Value::String(level.as_str().to_string())),
            ],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn set_service_tier(&self, service_tier: ServiceTier) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "set_service_tier",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("serviceTier", json!(service_tier)),
            ],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn cycle_thinking_level(&self) -> BoxFuture<Result<Option<ThinkingLevel>, String>> {
        let this = self.clone();
        let request = command_body(
            "cycle_thinking_level",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move {
            let data = this.request_data(request, None).await?;
            if data.is_null() {
                return Ok(None);
            }
            let level = data
                .get("level")
                .and_then(Value::as_str)
                .and_then(thinking_level_from_str);
            Ok(level)
        })
    }

    fn set_transport(&self, transport: Transport) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "set_transport",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("transport", Value::String(transport)),
            ],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn set_steering_mode(&self, mode: &str) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "set_steering_mode",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("mode", Value::String(mode.to_string())),
            ],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn set_follow_up_mode(&self, mode: &str) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "set_follow_up_mode",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("mode", Value::String(mode.to_string())),
            ],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn set_auto_compaction_enabled(&self, enabled: bool) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "set_auto_compaction",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("enabled", Value::Bool(enabled)),
            ],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn set_auto_retry_enabled(&self, enabled: bool) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "set_auto_retry",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("enabled", Value::Bool(enabled)),
            ],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn compact(&self, custom_instructions: Option<&str>) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let request = command_body(
            "compact",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("customInstructions", optional_string(custom_instructions)),
            ],
        );
        Box::pin(async move { this.request_data(request, None).await })
    }

    fn refine(&self, options: Value) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let mut fields = vec![("activeSessionId", Value::String(this.active_session_id()))];
        if let Some(instructions) = options.get("instructions").and_then(Value::as_str) {
            fields.push(("instructions", Value::String(instructions.to_string())));
        }
        if let Some(rollback_id) = options.get("rollbackId").and_then(Value::as_str) {
            fields.push(("rollbackId", Value::String(rollback_id.to_string())));
        }
        if let Some(global) = options.get("global").and_then(Value::as_bool) {
            fields.push(("global", Value::Bool(global)));
        }
        let request = command_body("refine", fields);
        Box::pin(async move { this.request_data(request, Some(DAEMON_REFINE_REQUEST_TIMEOUT_MS)).await })
    }

    fn abort_compaction(&self) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "abort_compaction",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn abort_branch_summary(&self) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "abort_branch_summary",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn abort_retry(&self) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "abort_retry",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn reload(&self) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "reload",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn new_session(&self, options: Option<AgentConnectionNewSessionOptions>) -> BoxFuture<Result<bool, String>> {
        let this = self.clone();
        let parent_session = options.and_then(|options| options.parent_session);
        let request = command_body(
            "new_session",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                (
                    "parentSession",
                    match parent_session {
                        Some(parent) => Value::String(parent),
                        None => Value::Null,
                    },
                ),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(request, None).await?;
            Ok(data.get("cancelled").and_then(Value::as_bool).unwrap_or(false))
        })
    }

    fn switch_session(
        &self,
        session_path: &str,
        options: Option<AgentConnectionSwitchSessionOptions>,
    ) -> BoxFuture<Result<bool, String>> {
        let this = self.clone();
        let source_active_session_id = this.active_session_id();
        let cwd_override = options.and_then(|options| options.cwd_override);
        let request = command_body(
            "switch_session",
            vec![
                ("activeSessionId", Value::String(source_active_session_id.clone())),
                ("sessionPath", Value::String(session_path.to_string())),
                ("cwdOverride", optional_string(cwd_override.as_deref())),
            ],
        );
        Box::pin(async move {
            match this.request_data(request, None).await {
                Ok(data) => Ok(data.get("cancelled").and_then(Value::as_bool).unwrap_or(false)),
                Err(error) => {
                    // `SessionAlreadyActiveError` (core/session-lease.ts) carries the
                    // active session id; without that slice the port surfaces the error.
                    Err(error)
                }
            }
        })
    }

    fn fork(&self, entry_id: &str, options: Option<AgentConnectionForkOptions>) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let position = options.and_then(|options| options.position);
        let request = command_body(
            "fork",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("entryId", Value::String(entry_id.to_string())),
                ("position", optional_string(position.as_deref())),
            ],
        );
        Box::pin(async move { this.request_data(request, None).await })
    }

    fn navigate_tree(
        &self,
        target_id: &str,
        options: Option<AgentConnectionNavigateTreeOptions>,
    ) -> BoxFuture<Result<AgentConnectionNavigateTreeResult, String>> {
        let this = self.clone();
        let mut fields = vec![
            ("activeSessionId", Value::String(this.active_session_id())),
            ("targetId", Value::String(target_id.to_string())),
        ];
        if let Some(options) = &options {
            if let Some(summarize) = options.summarize {
                fields.push(("summarize", Value::Bool(summarize)));
            }
            if let Some(custom_instructions) = &options.custom_instructions {
                fields.push(("customInstructions", Value::String(custom_instructions.clone())));
            }
            if let Some(replace_instructions) = options.replace_instructions {
                fields.push(("replaceInstructions", Value::Bool(replace_instructions)));
            }
            if let Some(label) = &options.label {
                fields.push(("label", Value::String(label.clone())));
            }
        }
        let request = command_body("navigate_tree", fields);
        Box::pin(async move {
            let data = this.request_data(request, None).await?;
            serde_json::from_value(data)
                .map_err(|error| format!("Daemon returned an invalid navigate-tree result: {error}"))
        })
    }

    fn import_from_jsonl(&self, input_path: &str, cwd_override: Option<&str>) -> BoxFuture<Result<bool, String>> {
        let this = self.clone();
        let request = command_body(
            "import_jsonl",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("inputPath", Value::String(input_path.to_string())),
                ("cwdOverride", optional_string(cwd_override)),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(request, None).await?;
            Ok(data.get("cancelled").and_then(Value::as_bool).unwrap_or(false))
        })
    }

    fn export_to_html(&self, output_path: Option<&str>) -> BoxFuture<Result<String, String>> {
        let this = self.clone();
        let request = command_body(
            "export_html",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("outputPath", optional_string(output_path)),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(request, None).await?;
            Ok(data.get("path").and_then(Value::as_str).unwrap_or_default().to_string())
        })
    }

    fn export_to_jsonl(&self, output_path: Option<&str>) -> BoxFuture<Result<String, String>> {
        let this = self.clone();
        let request = command_body(
            "export_jsonl",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("outputPath", optional_string(output_path)),
            ],
        );
        Box::pin(async move {
            let data = this.request_data(request, None).await?;
            Ok(data.get("path").and_then(Value::as_str).unwrap_or_default().to_string())
        })
    }

    fn set_session_name(&self, name: &str) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        let request = command_body(
            "set_session_name",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("name", Value::String(name.to_string())),
            ],
        );
        Box::pin(async move { this.request_ok(request).await })
    }

    fn get_rlm_max_depth_status(&self) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let request = command_body(
            "get_rlm_max_depth_status",
            vec![("activeSessionId", Value::String(this.active_session_id()))],
        );
        Box::pin(async move { this.request_data(request, None).await })
    }

    fn set_rlm_max_depth(&self, max_depth: f64, options: Option<Value>) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        let global = options.as_ref().and_then(|options| options.get("global")).and_then(Value::as_bool);
        let request = command_body(
            "set_rlm_max_depth",
            vec![
                ("activeSessionId", Value::String(this.active_session_id())),
                ("maxDepth", json!(max_depth)),
                ("global", optional_bool(global)),
            ],
        );
        Box::pin(async move { this.request_data(request, None).await })
    }

    fn rename_saved_session(&self, session_path: &str, name: &str) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        // `renameDaemonSavedSession(...)` lives in
        // modes/daemon/saved-session-catalog.ts (another slice).
        let _ = (session_path, name);
        Box::pin(async {
            Err("renameSavedSession requires modes/daemon/saved-session-catalog.ts (not ported in this slice)".to_string())
        })
    }

    fn delete_saved_session(&self, session_path: &str) -> BoxFuture<Result<Value, String>> {
        let this = self.clone();
        // `deleteDaemonSavedSession(...)` lives in
        // modes/daemon/saved-session-catalog.ts (another slice).
        let _ = session_path;
        Box::pin(async {
            Err("deleteSavedSession requires modes/daemon/saved-session-catalog.ts (not ported in this slice)".to_string())
        })
    }

    fn watch_session(
        &self,
        active_session_id: &str,
    ) -> BoxFuture<Result<Option<Box<dyn AgentConnectionSessionWatcher>>, String>> {
        let this = self.clone();
        // A second connection on the shared client; each one filters to its own
        // session id. attach() rejects for an unknown/exited session - treat that
        // as unreachable.
        let client = this.client.clone().control_plane_transport();
        let options = DaemonAgentConnectionOptions {
            close_client_on_dispose: false,
            direct_transport: false,
            ..Default::default()
        };
        let active_session_id = active_session_id.to_string();
        Box::pin(async move {
            match DaemonAgentConnection::attach(client, active_session_id, options).await {
                Ok(connection) => Ok(Some(Box::new(DaemonWatcherConnection { connection })
                    as Box<dyn AgentConnectionSessionWatcher>)),
                Err(_) => Ok(None),
            }
        })
    }

    fn dispose(&self) -> BoxFuture<Result<(), String>> {
        let this = self.clone();
        Box::pin(async move {
            this.dispose_inner().await;
            Ok(())
        })
    }
}

/// `AgentConnectionSessionWatcher` backed by a second `DaemonAgentConnection`.
struct DaemonWatcherConnection {
    connection: Arc<DaemonAgentConnection>,
}

impl AgentConnectionSessionWatcher for DaemonWatcherConnection {
    fn get_messages(&self) -> BoxFuture<Vec<AgentMessage>> {
        let connection = self.connection.clone();
        Box::pin(async move {
            connection
                .get_messages()
                .await
                .unwrap_or_default()
        })
    }

    fn get_commands(&self) -> BoxFuture<Vec<AgentConnectionSlashCommand>> {
        let connection = self.connection.clone();
        Box::pin(async move {
            connection
                .get_commands()
                .await
                .unwrap_or_default()
        })
    }

    fn subscribe(&self, listener: AgentConnectionEventListener) -> Box<dyn Fn() + Send + Sync> {
        self.connection.subscribe(listener)
    }

    fn get_tool_definition(&self, name: &str) -> BoxFuture<Option<AgentConnectionToolDefinition>> {
        let connection = self.connection.clone();
        let name = name.to_string();
        Box::pin(async move { connection.get_tool_definition(&name).await.unwrap_or(None) })
    }

    fn close(&self) -> BoxFuture<()> {
        let connection = self.connection.clone();
        Box::pin(async move {
            let _ = connection.dispose().await;
        })
    }
}

/// `thinkingLevel` narrowing used by `cycleThinkingLevel`.
fn thinking_level_from_str(value: &str) -> Option<ThinkingLevel> {
    match value {
        "off" => Some(ThinkingLevel::Off),
        "minimal" => Some(ThinkingLevel::Minimal),
        "low" => Some(ThinkingLevel::Low),
        "medium" => Some(ThinkingLevel::Medium),
        "high" => Some(ThinkingLevel::High),
        "xhigh" => Some(ThinkingLevel::Xhigh),
        "max" => Some(ThinkingLevel::Max),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) async fn test_decode_cached_attach_frames(frames: &[Value]) -> Vec<AgentMessage> {
    let response = &frames[0]["data"];
    let client = crate::modes::daemon::daemon_client::DaemonClient::create("unused-cache-wire-test-socket");
    let transport = Arc::new(crate::main_entry::MainEntryDaemonTransport::new(client));
    let connection = DaemonAgentConnection::new(transport, response["activeSessionId"].as_str().unwrap().into(), Default::default());
    for frame in &frames[1..] {
        let outbound = crate::main_entry::test_outbound_from_wire(frame).expect("cached attach frame must decode through the production wire adapter");
        connection.handle_daemon_message(outbound).await.unwrap();
    }
    tokio::time::timeout(Duration::from_secs(2), connection.apply_attach_result(response)).await
        .expect("completed cached transcript must not hang the UI attach").unwrap();
    let snapshot = connection.latest_snapshot.lock().unwrap().clone().unwrap();
    snapshot.messages
}

#[cfg(test)]
#[path = "daemon_backlog_tests.rs"]
mod daemon_backlog_tests;

#[cfg(test)]
#[path = "jev_status_tests.rs"]
mod jev_status_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::daemon::daemon_client::{DaemonClientError, DaemonSocketClosedError};

    #[tokio::test]
    async fn nine_supervisor_new_client_accepts_old_server_full_snapshot() {
        let client = crate::modes::daemon::daemon_client::DaemonClient::create("unused-legacy-server-test-socket");
        let transport = Arc::new(crate::main_entry::MainEntryDaemonTransport::new(client));
        let connection = DaemonAgentConnection::new(transport, "active".into(), Default::default());
        let legacy = json!({"activeSessionId":"active","snapshot":{"summary":{"sessionId":"saved","activeSessionId":"active"},"state":{},"messages":[{"role":"user","content":"legacy transcript","timestamp":1}],"lastEventSequence":4}});
        tokio::time::timeout(Duration::from_secs(2), connection.apply_attach_result(&legacy)).await
            .expect("old server response must not wait for unadvertised chunks").unwrap();
        let snapshot = connection.latest_snapshot.lock().unwrap();
        assert_eq!(snapshot.as_ref().unwrap().messages.len(), 1);
        assert_eq!(snapshot.as_ref().unwrap().last_event_sequence, Some(4.0));
        assert!(connection.snapshot_assemblies.lock().unwrap().is_empty());
    }

    fn flat_node(id: &str, parent: Option<&str>, timestamp: &str) -> AgentConnectionSessionTreeFlatNode {
        AgentConnectionSessionTreeFlatNode {
            entry: AgentConnectionSessionEntry::Custom {
                id: id.to_string(),
                parent_id: parent.map(str::to_string),
                timestamp: timestamp.to_string(),
                custom_type: "x".to_string(),
                data: None,
            },
            label: None,
            label_timestamp: None,
        }
    }

    #[test]
    fn builds_a_tree_and_sorts_siblings_by_timestamp() {
        let nodes = vec![
            flat_node("b", Some("a"), "2026-01-01T00:00:02.000Z"),
            flat_node("a", None, "2026-01-01T00:00:01.000Z"),
            flat_node("c", Some("a"), "2026-01-01T00:00:01.500Z"),
        ];
        let tree = build_session_tree_from_flat_nodes(&nodes);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].entry.id(), "a");
        let children: Vec<&str> = tree[0].children.iter().map(|child| child.entry.id()).collect();
        assert_eq!(children, vec!["c", "b"]);
    }

    #[test]
    fn canonical_session_tree_response_preserves_branches_and_compaction() {
        let mut nodes = vec![
            json!({"type":"model_change", "id":"model", "parentId":null, "provider":"fixture", "modelId":"fixture"}),
            json!({"type":"thinking_level_change", "id":"effort", "parentId":"model", "thinkingLevel":"xhigh"}),
            json!({"type":"message", "id":"message", "parentId":"effort", "message":{"role":"user", "content":"hello", "timestamp":1}}),
            json!({"type":"compaction", "id":"compact", "parentId":"message", "summary":"summary", "firstKeptEntryId":"message", "tokensBefore":250000}),
            json!({"type":"branch_summary", "id":"branch", "parentId":"message", "fromId":"compact", "summary":"branch"}),
        ];
        for node in &mut nodes { node["timestamp"] = json!("2026-09-16T00:00:00Z"); }
        let response = json!({"flatNodes": nodes.iter().map(|entry| json!({"entry":entry})).collect::<Vec<_>>(), "leafId":"branch"});
        let parsed = parse_session_tree_response(response).unwrap();
        assert_eq!(parsed.tree.len(), 1);
        assert_eq!(parsed.tree[0].entry.id(), "model");
        let message = &parsed.tree[0].children[0].children[0];
        assert_eq!(message.entry.id(), "message");
        assert_eq!(message.children.iter().map(|node| node.entry.id()).collect::<Vec<_>>(), vec!["compact", "branch"]);
        assert_eq!(parsed.leaf_id.as_deref(), Some("branch"));
        let snapshot = parse_session_snapshot(&json!({"sessionTree": {"tree":parsed.tree, "leafId":"branch"}})).unwrap();
        assert_eq!(snapshot.session_tree.unwrap().tree[0].children[0].entry.parent_id(), Some("model"));
    }

    #[test]
    fn malformed_session_tree_is_an_explicit_error_not_an_empty_tree() {
        for response in [json!({}), json!({"flatNodes":null}), json!({"flatNodes":[{"entry":{"type":"model_change", "id":"bad"}}]})] {
            assert!(parse_session_tree_response(response).unwrap_err().starts_with("Daemon returned an invalid session tree:"));
        }
        assert!(parse_session_tree_response(json!({"flatNodes":[], "leafId":null})).unwrap().tree.is_empty());
    }

    #[test]
    #[ignore = "requires a fresh private daemon response captured from the candidate executable"]
    fn candidate_daemon_tree_wire_reaches_the_ui_decoder() {
        let path = std::env::var("OPTIMUS_SAFETY_TREE_WIRE").expect("private wire artifact path");
        assert!(path.contains("optimus-safety-fixes-20260916") || path.contains("optimus-nine-fixes-20260916"));
        let wire: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let expected: Vec<&str> = wire["flatNodes"].as_array().unwrap().iter()
            .map(|node| node["entry"]["id"].as_str().unwrap()).collect();
        let parsed = parse_session_tree_response(wire.clone()).unwrap();
        assert!(!expected.is_empty());
        let mut stack: Vec<_> = parsed.tree.iter().collect();
        let mut observed = Vec::new();
        while let Some(node) = stack.pop() {
            observed.push(node.entry.id());
            let source = wire["flatNodes"].as_array().unwrap().iter().find(|item| item["entry"]["id"].as_str() == Some(node.entry.id())).unwrap();
            assert_eq!(node.entry.parent_id(), source["entry"]["parentId"].as_str());
            stack.extend(node.children.iter());
        }
        assert_eq!(observed.len(), expected.len());
        assert!(expected.iter().all(|id| observed.contains(id)));
    }

    #[test]
    fn self_parent_entries_become_roots() {
        let nodes = vec![flat_node("root", Some("root"), "2026-01-01T00:00:01.000Z")];
        let tree = build_session_tree_from_flat_nodes(&nodes);
        assert_eq!(tree.len(), 1);
        assert!(tree[0].children.is_empty());
    }

    #[test]
    fn builds_descendants_after_parents_and_preserves_root_order() {
        let nodes = vec![
            flat_node("root", None, "2026-01-01T00:00:03.000Z"),
            flat_node("child", Some("root"), "2026-01-01T00:00:02.000Z"),
            flat_node("grandchild", Some("child"), "2026-01-01T00:00:01.000Z"),
            flat_node("orphan", Some("missing"), "2026-01-01T00:00:00.000Z"),
        ];
        let tree = build_session_tree_from_flat_nodes(&nodes);
        assert_eq!(tree.iter().map(|node| node.entry.id()).collect::<Vec<_>>(), vec!["root", "orphan"]);
        assert_eq!(tree[0].children[0].entry.id(), "child");
        assert_eq!(tree[0].children[0].children[0].entry.id(), "grandchild");
    }

    #[test]
    fn max_event_sequence_keeps_the_larger_value() {
        assert_eq!(max_event_sequence(None, Some(3)), Some(3));
        assert_eq!(max_event_sequence(Some(5), None), Some(5));
        assert_eq!(max_event_sequence(Some(5), Some(3)), Some(5));
        assert_eq!(max_event_sequence(Some(5), Some(9)), Some(9));
    }

    #[test]
    fn snapshot_and_cursor_updates_do_not_relock_their_state() {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            runtime.block_on(async move {
                let client = crate::modes::daemon::daemon_client::DaemonClient::create("unused-test-socket");
                let transport = Arc::new(crate::main_entry::MainEntryDaemonTransport::new(client));
                let connection = DaemonAgentConnection::new(transport, "active".to_string(), Default::default());
                connection.observe_event_cursor(DaemonEventCursor { generation: "generation".to_string(), sequence: 2 });
                connection.observe_daemon_event_sequence(&DaemonOutbound::HeartbeatsChanged {
                    active_session_id: Some("active".to_string()),
                    meta: Some(DaemonEventMeta { sequence: Some(5), cursor: None }),
                });
                assert_eq!(connection.last_event_cursor.lock().unwrap().as_ref().unwrap().sequence, 5);
                let snapshot = DaemonSessionSnapshot { last_event_sequence: Some(8), ..Default::default() };
                connection.apply_replacement_snapshot(&snapshot, None);
                assert_eq!(*connection.last_event_sequence.lock().unwrap(), Some(8));
                connection.completed_snapshots.lock().unwrap().push_back(("cached".to_string(), snapshot.clone()));
                assert_eq!(connection.wait_for_snapshot("cached").await.unwrap(), snapshot);
                assert!(connection.completed_snapshots.lock().unwrap().is_empty());
                sender.send(()).unwrap();
            });
        });
        receiver.recv_timeout(Duration::from_secs(5)).expect("snapshot/cursor update deadlocked");
    }

    #[test]
    fn invalidates_cached_snapshot_matches_the_switch() {
        assert!(!invalidates_cached_snapshot("get_messages"));
        assert!(!invalidates_cached_snapshot("jev_get_status"));
        assert!(!invalidates_cached_snapshot("attach"));
        assert!(invalidates_cached_snapshot("set_model"));
        assert!(invalidates_cached_snapshot("abort"));
    }

    #[test]
    fn format_error_sentence_adds_a_period() {
        assert_eq!(format_error_sentence("boom"), "boom.");
        assert_eq!(format_error_sentence("boom!"), "boom!");
        assert_eq!(format_error_sentence("   "), "Unknown daemon error.");
    }

    #[test]
    fn map_daemon_snapshot_rejects_invalid_history() {
        let mut snapshot = DaemonSessionSnapshot::default();
        snapshot.history = Some(AgentConnectionHistoryWindow {
            version: 1.0,
            generation: "g".to_string(),
            representation: "r".to_string(),
            tip_entry_id: None,
            total_message_count: 3.0,
            start_index: 0.0,
            entry_ids: vec!["a".to_string()],
            has_older: false,
            order: "chronological".to_string(),
        });
        let error = map_daemon_session_snapshot(&snapshot, None).unwrap_err();
        assert_eq!(error, "Daemon returned an invalid recent-first history snapshot");
    }

    #[test]
    fn map_daemon_snapshot_accepts_valid_history() {
        let mut snapshot = DaemonSessionSnapshot::default();
        snapshot.history = Some(AgentConnectionHistoryWindow {
            version: 1.0,
            generation: "g".to_string(),
            representation: "r".to_string(),
            tip_entry_id: None,
            total_message_count: 0.0,
            start_index: 0.0,
            entry_ids: Vec::new(),
            has_older: false,
            order: "chronological".to_string(),
        });
        assert!(map_daemon_session_snapshot(&snapshot, None).is_ok());
    }

    #[test]
    fn read_session_summaries_validates_shape() {
        assert!(read_session_summaries(&json!({"sessions": []})).is_ok());
        assert_eq!(
            read_session_summaries(&json!({})).unwrap_err(),
            "Daemon returned an invalid session list response"
        );
    }

    #[test]
    fn unknown_command_detection_is_exact() {
        assert!(is_unknown_daemon_command_error(
            "unknown command: abort_bash",
            "abort_bash"
        ));
        assert!(!is_unknown_daemon_command_error(
            "unknown command: abort_bash",
            "execute_bash"
        ));
    }

    /// `getDaemonSocketCloseReason` (`daemon-client.ts:103-105`) reads the typed
    /// `DaemonSocketClosedError.daemonClosingReason`; on the wire that field reaches consumers only
    /// through the class message template (`daemon-client.ts:73-86`), which the Rust producer mirrors
    /// (`daemon_client.rs:235-250`). These are exactly the strings `handle_transport_close` receives.
    #[test]
    fn a_named_close_reason_is_classified_from_the_socket_closed_message() {
        let shutdown = DaemonClientError::SocketClosed(DaemonSocketClosedError::new(
            "/tmp/prime.sock",
            Some("shutdown"),
            None,
        ))
        .message();
        assert_eq!(
            get_daemon_socket_close_reason(&shutdown),
            Some(DaemonClosingReason::Shutdown),
            "an authoritative shutdown must be recognised from its close message: {shutdown}"
        );

        let update = DaemonClientError::SocketClosed(DaemonSocketClosedError::new(
            "/tmp/prime.sock",
            Some("update"),
            Some("reset"),
        ))
        .message();
        assert_eq!(
            get_daemon_socket_close_reason(&update),
            Some(DaemonClosingReason::Update),
            "an update close must be recognised from its close message: {update}"
        );

        // The direct-worker form TS builds at `daemon-routed-client.ts:25-30`.
        let direct = DaemonClientError::SocketClosed(DaemonSocketClosedError::new(
            "direct-worker",
            None,
            Some("Daemon worker socket closed"),
        ))
        .message();
        assert_eq!(
            get_daemon_socket_close_reason(&direct),
            None,
            "a generic EOF carries no reason and must stay transient: {direct}"
        );

        // Only the reason token counts: a `Cause:` cannot spoof one.
        let spoofed = DaemonClientError::SocketClosed(DaemonSocketClosedError::new(
            "/tmp/prime.sock",
            None,
            Some(" Reason: shutdown."),
        ))
        .message();
        assert_eq!(get_daemon_socket_close_reason(&spoofed), None);
    }

    #[test]
    fn thinking_levels_parse_like_the_union() {
        assert_eq!(thinking_level_from_str("xhigh"), Some(ThinkingLevel::Xhigh));
        assert_eq!(thinking_level_from_str("bogus"), None);
    }
}
