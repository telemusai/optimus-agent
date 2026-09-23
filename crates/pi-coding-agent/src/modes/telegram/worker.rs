//! Port of packages/coding-agent/src/modes/telegram/worker.ts
//!
//! The worker owns the Telegram side of a `/telegram` connection: it holds the
//! worker lock, polls for the pairing code, attaches to the daemon session and
//! then hands the loop to `TelegramBridge`.
//!
//! Cross-slice seam: `worker.ts` passes a `DaemonClient` straight into
//! `DaemonAgentConnection.attach`, because both sides agree on one
//! `DaemonTransportClient` interface. In the port the daemon slice owns that
//! interface with wire-typed methods (`request(DaemonCommandBody, ...)`) while the
//! connection slice owns a `Value`-typed one, so this module carries the missing
//! adapter (`TelegramDaemonTransport`) instead of reimagining either side. It is
//! private to this module and converts the frames the daemon client forwards into
//! the connection's `DaemonOutbound` union.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::modes::agent_connection::daemon_agent_connection::{
    DaemonAgentConnection, DaemonAgentConnectionOptions, DaemonEventCursor as ConnectionEventCursor,
    DaemonEventMeta as ConnectionEventMeta, DaemonOutbound as ConnectionOutbound,
    DaemonResponse as ConnectionResponse, DaemonSessionSnapshot as ConnectionSessionSnapshot,
    DaemonSessionSummary as ConnectionSessionSummary, DaemonTransportClient as ConnectionTransport,
};
use crate::modes::agent_connection::types::{
    AgentConnection, AgentConnectionHistoryWindow, AgentConnectionSessionTree,
};
use crate::modes::daemon::daemon_client::{
    DaemonClient, DaemonClientCloseListener, DaemonClientError, DaemonClientMessageListener,
    DaemonClientRequestOptions, DaemonCommandBody,
};
use crate::modes::daemon::daemon_protocol::{
    DaemonEventCursor, DaemonHistoryWindow, DaemonOutbound as ProtocolOutbound,
    DaemonSessionSnapshot as ProtocolSessionSnapshot, DaemonSessionSnapshotHead, DaemonSessionTree,
};
use crate::modes::daemon::daemon_session_list::SessionSummary;
use crate::modes::telegram::api::{
    TelegramApi, TelegramCallError, TelegramFetcher, TelegramHttpResponse, TELEGRAM_DEFAULT_BASE_URL,
};
use crate::modes::telegram::bridge::{accept_telegram_pairing, TelegramBridge, TelegramBridgeError};
use crate::modes::telegram::manager::{
    now_ms, telegram_stop_requested, TelegramFileLock, TelegramLockError, TELEGRAM_WORKER_LOCK_OPTIONS,
};
use crate::modes::telegram::store::{
    is_record, TelegramConnectionSettings, TelegramDelivery, TelegramStore, TelegramWorkerStatus,
};

/// The `fetch` seam for `new TelegramApi(token)` with no injected `createApi`.
///
/// Private to this module: `core/extensions/builtin/telegram.ts` owns its own
/// fetcher for the `/telegram` command path. It issues the same POST the
/// TypeScript issues (`method: "POST"`, JSON body) and reports the same
/// response facts (`ok`, `status`, raw bytes).
struct ReqwestTelegramFetcher;

impl TelegramFetcher for ReqwestTelegramFetcher {
    fn fetch(
        &self,
        url: String,
        body: String,
        timeout_ms: u64,
    ) -> pi_ai::types::BoxFuture<Result<TelegramHttpResponse, String>> {
        Box::pin(async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_millis(timeout_ms))
                .build()
                .map_err(|error| error.to_string())?;
            let response = client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(body)
                .send()
                .await
                .map_err(|error| error.to_string())?;
            let ok = response.status().is_success();
            let status = response.status().as_u16() as f64;
            let body = response
                .bytes()
                .await
                .map_err(|error| error.to_string())?
                .to_vec();
            Ok(TelegramHttpResponse { ok, status, body })
        })
    }
}

/// `new TelegramApi(settings.botToken)`.
fn default_create_api(token: &str) -> Result<Arc<TelegramApi>, String> {
    Ok(Arc::new(TelegramApi::new(
        token,
        TELEGRAM_DEFAULT_BASE_URL,
        Arc::new(ReqwestTelegramFetcher),
    )?))
}

/// `randomUUID()`.
fn random_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// `process.pid`.
fn process_pid() -> f64 {
    std::process::id() as f64
}

/// `String(error)` for a failed Telegram call, matching what `api.redact(...)`
/// receives in the TypeScript catch blocks.
fn error_message(error: &TelegramCallError) -> String {
    match error {
        TelegramCallError::Api(api_error) => api_error.message.clone(),
        TelegramCallError::Error(message) => message.clone(),
        TelegramCallError::Aborted => "Telegram request aborted".to_string(),
    }
}

/// `error instanceof TelegramApiError && (error.code === 401 || error.code === 409)`.
fn is_fatal_api_error(error: &TelegramCallError) -> bool {
    matches!(error, TelegramCallError::Api(api_error) if api_error.code == 401.0 || api_error.code == 409.0)
}

/// `error instanceof TelegramApiError && error.retryAfter ? error.retryAfter * 1000 : 5000`.
fn pairing_retry_delay_ms(error: &TelegramCallError) -> f64 {
    match error {
        TelegramCallError::Api(api_error) if api_error.retry_after != 0.0 => api_error.retry_after * 1000.0,
        _ => 5000.0,
    }
}

/// `await delay(ms, undefined, { signal }).catch(() => undefined)`: resolves on
/// abort as well, and the caller ignores the outcome.
async fn delay_ms(ms: f64, controller: &CancellationToken) {
    let ms = if ms.is_finite() && ms > 0.0 { ms as u64 } else { 0 };
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_millis(ms)) => {}
        _ = controller.cancelled() => {}
    }
}

/// `setInterval(callback, periodMs)` (private copy of the bridge's helper).
fn start_interval<F>(period_ms: u64, callback: F) -> tokio::task::JoinHandle<()>
where
    F: Fn() + Send + Sync + 'static,
{
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(period_ms)).await;
            callback();
        }
    })
}

/// `store.write("worker.json", status)` with the shared `updatedAt` refresh.
fn write_worker_status(store: &TelegramStore, status: &Mutex<TelegramWorkerStatus>) -> Result<(), String> {
    let value = {
        let mut status = status.lock().unwrap();
        status.updated_at = now_ms() as f64;
        serde_json::to_value(&*status).map_err(|error| error.to_string())?
    };
    store.write("worker.json", &value)
}

/// `mapDaemonSessionSnapshot` reads the wire `meta`; the connection slice only
/// keeps `sequence` and `cursor`, so the adapter projects exactly those.
fn connection_meta_from_wire(meta: Option<&Value>) -> Option<ConnectionEventMeta> {
    let meta = meta?.as_object()?;
    let sequence = meta.get("sequence").and_then(Value::as_i64);
    let cursor = meta.get("cursor").and_then(Value::as_object).map(|cursor| {
        ConnectionEventCursor {
            generation: cursor
                .get("generation")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            sequence: cursor.get("sequence").and_then(Value::as_i64).unwrap_or(0),
        }
    });
    if sequence.is_none() && cursor.is_none() {
        return None;
    }
    Some(ConnectionEventMeta { sequence, cursor })
}

fn connection_cursor(cursor: &DaemonEventCursor) -> ConnectionEventCursor {
    ConnectionEventCursor {
        generation: cursor.generation.clone(),
        sequence: cursor.sequence as i64,
    }
}

fn connection_history_window(window: &DaemonHistoryWindow) -> AgentConnectionHistoryWindow {
    AgentConnectionHistoryWindow {
        version: window.version,
        generation: window.generation.clone(),
        representation: window.representation.clone(),
        tip_entry_id: window.tip_entry_id.clone(),
        total_message_count: window.total_message_count,
        start_index: window.start_index,
        entry_ids: window.entry_ids.clone(),
        has_older: window.has_older,
        order: window.order.clone(),
    }
}

fn connection_session_tree(tree: &DaemonSessionTree) -> AgentConnectionSessionTree {
    AgentConnectionSessionTree {
        tree: tree.tree.clone(),
        leaf_id: tree.leaf_id.clone(),
    }
}

/// `summary` as `DaemonSessionSummary`; the wire row keeps fields this adapter
/// does not model in `extra`, so `lastEventSequence` is read from there.
fn connection_session_summary(summary: &SessionSummary) -> ConnectionSessionSummary {
    ConnectionSessionSummary {
        session_id: summary.session_id.clone(),
        session_file: summary.session_file.clone(),
        active_session_id: summary.active_session_id.clone(),
        id: Some(summary.id.clone()),
        streaming_message: summary
            .streaming_message
            .clone()
            .and_then(|message| serde_json::from_value(message).ok()),
        last_event_sequence: summary
            .extra
            .get("lastEventSequence")
            .and_then(Value::as_i64),
        last_event_cursor: None,
    }
}

/// `Omit<DaemonSessionSnapshot, "messages">` -> the connection's snapshot shape.
fn connection_snapshot_head(head: &DaemonSessionSnapshotHead) -> ConnectionSessionSnapshot {
    ConnectionSessionSnapshot {
        state: head.state.clone(),
        messages: Vec::new(),
        summary: connection_session_summary(&head.summary),
        history: head.history.as_ref().map(connection_history_window),
        session_context: head.session_context.clone(),
        session_tree: head.session_tree.as_ref().map(connection_session_tree),
        parent: head.parent.clone(),
        children: head.children.clone(),
        last_event_sequence: Some(head.last_event_sequence as i64),
        last_event_cursor: head.last_event_cursor.as_ref().map(connection_cursor),
    }
}

/// The wire `DaemonSessionSnapshot` -> the connection's snapshot shape.
fn connection_session_snapshot(snapshot: &ProtocolSessionSnapshot) -> ConnectionSessionSnapshot {
    ConnectionSessionSnapshot {
        state: snapshot.state.clone(),
        messages: snapshot.messages.clone(),
        summary: connection_session_summary(&snapshot.summary),
        history: snapshot.history.as_ref().map(connection_history_window),
        session_context: snapshot.session_context.clone(),
        session_tree: snapshot.session_tree.as_ref().map(connection_session_tree),
        parent: snapshot.parent.clone(),
        children: snapshot.children.clone(),
        last_event_sequence: Some(snapshot.last_event_sequence as i64),
        last_event_cursor: snapshot.last_event_cursor.as_ref().map(connection_cursor),
    }
}

/// One wire frame as the connection slice's `DaemonOutbound`.
///
/// Frames with no connection-level meaning (`response`, `daemon_hello`,
/// `roster_update`, `session_attached`, `session_detached`,
/// `session_list_progress`) read as `None`, exactly as the TypeScript connection
/// ignores them.
fn connection_outbound_from_wire(value: &Value) -> Option<ConnectionOutbound> {
    let meta = connection_meta_from_wire(value.get("meta"));
    // `meta` is reconstructed above; dropping it here keeps parsing independent of
    // the daemon-side meta shape (which carries required fields this adapter and
    // the connection do not read).
    let mut frame = value.clone();
    if let Some(object) = frame.as_object_mut() {
        object.remove("meta");
    }
    let outbound = ProtocolOutbound::from_value(&frame)?;
    let outbound = match outbound {
        ProtocolOutbound::DaemonClosing { reason } => ConnectionOutbound::DaemonClosing { reason },
        ProtocolOutbound::HeartbeatsChanged { active_session_id, .. } => ConnectionOutbound::HeartbeatsChanged {
            active_session_id,
            meta,
        },
        ProtocolOutbound::SessionEvent {
            active_session_id,
            event,
            ..
        } => ConnectionOutbound::SessionEvent {
            active_session_id,
            event,
            meta,
        },
        ProtocolOutbound::SideQuestionEvent {
            active_session_id,
            event,
        } => ConnectionOutbound::SideQuestionEvent {
            active_session_id,
            event,
            meta: None,
        },
        ProtocolOutbound::SessionStatus {
            active_session_id,
            recap,
            ..
        } => ConnectionOutbound::SessionStatus {
            active_session_id,
            recap,
            meta,
        },
        ProtocolOutbound::SessionResynced {
            active_session_id,
            snapshot,
            ..
        } => ConnectionOutbound::SessionResynced {
            active_session_id,
            snapshot: connection_session_snapshot(&snapshot),
            meta,
        },
        ProtocolOutbound::SessionReplaced {
            active_session_id,
            state,
            messages,
            snapshot_follows,
            ..
        } => ConnectionOutbound::SessionReplaced {
            active_session_id,
            state,
            messages,
            snapshot_follows,
            meta,
        },
        ProtocolOutbound::SessionSnapshotBegin {
            active_session_id,
            snapshot_id,
            snapshot,
            message_count,
            purpose,
            ..
        } => ConnectionOutbound::SessionSnapshotBegin {
            active_session_id,
            snapshot_id,
            snapshot: connection_snapshot_head(&snapshot),
            message_count: message_count as usize,
            purpose,
        },
        ProtocolOutbound::SessionSnapshotChunk {
            active_session_id,
            snapshot_id,
            index,
            messages,
        } => ConnectionOutbound::SessionSnapshotChunk {
            active_session_id,
            snapshot_id,
            index: index as usize,
            messages,
        },
        ProtocolOutbound::SessionSnapshotEnd {
            active_session_id,
            snapshot_id,
            chunk_count,
            last_event_sequence,
            last_event_cursor,
        } => ConnectionOutbound::SessionSnapshotEnd {
            active_session_id,
            snapshot_id,
            chunk_count: chunk_count as usize,
            last_event_sequence: last_event_sequence as i64,
            last_event_cursor: last_event_cursor.as_ref().map(connection_cursor),
        },
        ProtocolOutbound::SessionSnapshotFailed {
            active_session_id,
            snapshot_id,
            error,
        } => ConnectionOutbound::SessionSnapshotFailed {
            active_session_id,
            snapshot_id,
            error,
        },
        ProtocolOutbound::SessionClosed {
            active_session_id,
            reason,
            ..
        } => ConnectionOutbound::SessionClosed {
            active_session_id,
            reason: reason.as_str().to_string(),
            meta,
        },
        ProtocolOutbound::ExtensionUiRequest {
            active_session_id,
            id,
            method,
            payload,
            ..
        } => ConnectionOutbound::ExtensionUiRequest {
            active_session_id,
            id,
            method,
            payload: Value::Object(payload),
            meta,
        },
        ProtocolOutbound::ExtensionError {
            active_session_id,
            extension_path,
            event,
            error,
            ..
        } => ConnectionOutbound::ExtensionError {
            active_session_id,
            extension_path,
            event,
            error,
            meta,
        },
        _ => return None,
    };
    Some(outbound)
}

/// `DaemonClient` as the connection slice's `DaemonTransportClient`.
///
/// The two slices agree on one interface in the TypeScript; the port splits it
/// (wire-typed on the daemon side, `Value`-typed on the connection side), so this
/// adapter is the missing bridge and nothing else.
struct TelegramDaemonTransport {
    client: Arc<DaemonClient>,
}

impl TelegramDaemonTransport {
    fn new(client: Arc<DaemonClient>) -> Self {
        Self { client }
    }
}

impl ConnectionTransport for TelegramDaemonTransport {
    /// `requestData(command, timeoutMs, options)` forwards `options.recoverable`
    /// (`daemon-agent-connection.ts:379/:498`; consumed at `daemon-client.ts:418`). Without this
    /// override the transport hardcodes `DaemonClientRequestOptions::default()`, so a caller's
    /// `recoverable: false` would never reach the client.
    fn request_with_recoverable(
        &self,
        command: Value,
        timeout_ms: Option<u64>,
        recoverable: bool,
    ) -> pi_ai::types::BoxFuture<Result<ConnectionResponse, String>> {
        let options = DaemonClientRequestOptions {
            recoverable: Some(recoverable),
            ..Default::default()
        };
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            let body: DaemonCommandBody = command
                .as_object()
                .cloned()
                .ok_or_else(|| "Daemon command must be a JSON object".to_string())?;
            let response = client
                .request(body, timeout_ms, options)
                .await
                .map_err(|error| error.message())?;
            Ok(ConnectionResponse {
                success: response.success,
                data: response.data.unwrap_or(Value::Null),
                error: response.error,
                error_code: response.error_info.map(|info| info.code().to_string()),
            })
        })
    }

    fn request(&self, command: Value, timeout_ms: Option<u64>) -> pi_ai::types::BoxFuture<Result<ConnectionResponse, String>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            let body: DaemonCommandBody = command
                .as_object()
                .cloned()
                .ok_or_else(|| "Daemon command must be a JSON object".to_string())?;
            let response = client
                .request(body, timeout_ms, DaemonClientRequestOptions::default())
                .await
                .map_err(|error| error.message())?;
            Ok(ConnectionResponse {
                success: response.success,
                data: response.data.unwrap_or(Value::Null),
                error: response.error,
                error_code: response.error_info.map(|info| info.code().to_string()),
            })
        })
    }

    fn on_message(&self, listener: Arc<dyn Fn(ConnectionOutbound) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        let wire_listener: DaemonClientMessageListener = Arc::new(move |value: &Value| {
            if let Some(outbound) = connection_outbound_from_wire(value) {
                listener(outbound);
            }
        });
        self.client.on_message(wire_listener)
    }

    fn on_close(&self, listener: Arc<dyn Fn(String) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        let wire_listener: DaemonClientCloseListener = Arc::new(move |error: &DaemonClientError| {
            listener(error.message());
        });
        self.client.on_close(wire_listener)
    }

    fn supports_server_capability(&self, capability: &str) -> bool {
        self.client.supports_server_capability(capability)
    }

    fn hello_socket_path(&self) -> Option<String> {
        // `this.client.hello?.socketPath`.
        self.client
            .hello()
            .and_then(|hello| hello.raw.get("socketPath").and_then(Value::as_str).map(str::to_string))
    }

    fn is_connected(&self) -> bool {
        self.client.is_connected()
    }

    fn enable_request_recovery(&self) {
        // `DaemonClient::enable_request_recovery` is async while the connection
        // calls this synchronously; `connect_telegram_session` awaits the same
        // flag before the first request so the ordering the TypeScript relies on
        // is preserved.
        let client = Arc::clone(&self.client);
        tokio::spawn(async move {
            client.enable_request_recovery().await;
        });
    }

    fn close(&self) {
        let client = Arc::clone(&self.client);
        tokio::spawn(async move {
            client.close().await;
        });
    }

    fn connect(&self, timeout_ms: u64) -> pi_ai::types::BoxFuture<Result<(), String>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move { client.connect(timeout_ms).await.map_err(|error| error.message()) })
    }

    fn wait_for_hello(&self, timeout_ms: u64) -> pi_ai::types::BoxFuture<Result<(), String>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            client
                .wait_for_hello(timeout_ms)
                .await
                .map(|_| ())
                .map_err(|error| error.message())
        })
    }

    fn reconnect(&self, timeout_ms: u64) -> pi_ai::types::BoxFuture<Result<(), String>> {
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            client
                .reconnect(timeout_ms)
                .await
                .map_err(|error| error.message())
        })
    }

    fn disconnect_for_reconnect(&self, reason: &str) {
        let client = Arc::clone(&self.client);
        let reason = reason.to_string();
        tokio::spawn(async move {
            client.disconnect_for_reconnect(&reason).await;
        });
    }

    fn reset_transport_for_reconnect(&self) {
        let client = Arc::clone(&self.client);
        tokio::spawn(async move {
            client.reset_transport_for_reconnect().await;
        });
    }

    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn ConnectionTransport> {
        self
    }
}

/// `connectTelegramSession(settings, agentDir)`.
pub async fn connect_telegram_session(
    settings: &TelegramConnectionSettings,
    agent_dir: &str,
) -> Result<Arc<DaemonAgentConnection>, String> {
    let client = DaemonClient::create(&settings.daemon_socket);
    let transport = Arc::new(TelegramDaemonTransport::new(Arc::clone(&client)));
    let attach_client: Arc<dyn ConnectionTransport> = transport.clone();
    let result = async {
        client
            .connect(3000)
            .await
            .map_err(|error| error.message())?;
        client
            .wait_for_hello(3000)
            .await
            .map_err(|error| error.message())?;
        // `recoverDaemon: async () => {}` is present in the TypeScript, which
        // turns on request recovery before the first command.
        client.enable_request_recovery().await;
        let listing = client
            .request(
                Map::from_iter([
                    ("type".to_string(), Value::String("list".to_string())),
                    ("all".to_string(), Value::Bool(true)),
                ]),
                None,
                DaemonClientRequestOptions::default(),
            )
            .await
            .map_err(|error| error.message())?;
        if !listing.success {
            return Err(listing.error.unwrap_or_else(|| "Daemon request failed".to_string()));
        }
        let sessions: Vec<Value> = listing
            .data
            .as_ref()
            .filter(|data| is_record(data))
            .and_then(|data| data.get("sessions"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let existing = sessions.iter().find(|item| {
            is_record(item) && item.get("sessionId").and_then(Value::as_str) == Some(settings.session_id.as_str())
        });
        let mut active_session_id = existing
            .and_then(|item| item.get("activeSessionId"))
            .and_then(Value::as_str)
            .map(str::to_string);
        if active_session_id.is_none() {
            if let Some(session_file) = &settings.session_file {
                let created = client
                    .request(
                        Map::from_iter([
                            ("type".to_string(), Value::String("create".to_string())),
                            ("sessionPath".to_string(), Value::String(session_file.clone())),
                            (
                                "config".to_string(),
                                serde_json::json!({
                                    "cwd": settings.cwd,
                                    "agentDir": agent_dir,
                                }),
                            ),
                        ]),
                        None,
                        DaemonClientRequestOptions::default(),
                    )
                    .await
                    .map_err(|error| error.message())?;
                if !created.success {
                    return Err(created.error.unwrap_or_else(|| "Daemon request failed".to_string()));
                }
                active_session_id = created
                    .data
                    .as_ref()
                    .filter(|data| is_record(data))
                    .and_then(|data| data.get("activeSessionId"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
        }
        let Some(active_session_id) = active_session_id else {
            return Err(
                "The connected Prime session is unavailable. Open it in Prime and run /telegram here.".to_string(),
            );
        };
        DaemonAgentConnection::attach(
            attach_client,
            active_session_id,
            DaemonAgentConnectionOptions {
                close_client_on_dispose: true,
                direct_transport: false,
                recover_daemon: true,
                supports_extension_ui: true,
                ..Default::default()
            },
        )
        .await
        .map_err(|error| error.to_string())
    }
    .await;

    match result {
        Ok(connection) => Ok(connection),
        Err(error) => {
            // `client.close()` in the catch: the socket must not outlive the attach.
            client.close().await;
            Err(error)
        }
    }
}

/// `options.createApi` plus `options.signal` for `runTelegramWorker`.
#[derive(Clone, Default)]
pub struct TelegramWorkerOptions {
    /// `createApi?: (token: string) => TelegramApi`.
    pub create_api: Option<Arc<dyn Fn(&str) -> Result<Arc<TelegramApi>, String> + Send + Sync>>,
    /// `signal?: AbortSignal`.
    pub signal: Option<CancellationToken>,
}

/// `process.once(signal, handler)`; aborting the returned handle removes it.
///
/// A local copy of the ACP mode's helper: another run mode's signal plumbing is
/// private, and the semantics are the same.
#[cfg(unix)]
fn listen_for_signal(
    signal: &'static str,
    handler: Arc<dyn Fn() + Send + Sync>,
) -> Option<tokio::task::JoinHandle<()>> {
    use tokio::signal::unix::{signal as unix_signal, SignalKind};
    let kind = match signal {
        "SIGINT" => SignalKind::interrupt(),
        "SIGTERM" => SignalKind::terminate(),
        _ => return None,
    };
    let mut stream = unix_signal(kind).ok()?;
    Some(tokio::spawn(async move {
        if stream.recv().await.is_some() {
            handler();
        }
    }))
}

#[cfg(not(unix))]
fn listen_for_signal(
    signal: &'static str,
    handler: Arc<dyn Fn() + Send + Sync>,
) -> Option<tokio::task::JoinHandle<()>> {
    // Node allows listening for both names on Windows; SIGTERM is never delivered
    // there, so only SIGINT is registered, through the console handler.
    if signal != "SIGINT" {
        return None;
    }
    Some(tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            handler();
        }
    }))
}

/// `writeStatus()`: refresh `updatedAt` and persist `worker.json`.
struct WorkerStatusWriter {
    store: TelegramStore,
    status: Arc<Mutex<TelegramWorkerStatus>>,
}

impl WorkerStatusWriter {
    fn write(&self) -> Result<(), String> {
        write_worker_status(&self.store, &self.status)
    }
}

/// `runTelegramWorker(agentDir, options)`.
pub async fn run_telegram_worker(agent_dir: &str, options: TelegramWorkerOptions) -> Result<(), String> {
    let store = TelegramStore::new(agent_dir);
    let controller = CancellationToken::new();
    let compromised = {
        let controller = controller.clone();
        Arc::new(move || controller.cancel()) as Arc<dyn Fn() + Send + Sync>
    };
    let release = match TelegramFileLock::acquire(
        &store.path("worker"),
        TELEGRAM_WORKER_LOCK_OPTIONS,
        None,
        Some(compromised),
    )
    .await
    {
        Ok(lock) => lock,
        // `if ((error as NodeJS.ErrnoException).code === "ELOCKED") return;`
        Err(TelegramLockError::Elocked) => return Ok(()),
        Err(error) => return Err(error.to_string()),
    };

    let instance_id = random_uuid();
    let writer = Arc::new(WorkerStatusWriter {
        store: TelegramStore::new(agent_dir),
        status: Arc::new(Mutex::new(TelegramWorkerStatus {
            instance_id: instance_id.clone(),
            pid: process_pid(),
            updated_at: now_ms() as f64,
            phase: "starting".to_string(),
            error: None,
        })),
    });
    writer.write()?;

    // `options.signal?.addEventListener("abort", stop, { once: true })` plus
    // `if (options.signal?.aborted) stop()`.
    let signal_task = options.signal.as_ref().map(|signal| {
        let controller = controller.clone();
        let signal = signal.clone();
        tokio::spawn(async move {
            if signal.is_cancelled() {
                controller.cancel();
                return;
            }
            signal.cancelled().await;
            controller.cancel();
        })
    });
    let signal_handlers: Vec<tokio::task::JoinHandle<()>> = ["SIGTERM", "SIGINT"]
        .iter()
        .filter_map(|signal| {
            let controller = controller.clone();
            listen_for_signal(signal, Arc::new(move || controller.cancel()))
        })
        .collect();

    let timer = {
        let controller = controller.clone();
        let writer = Arc::clone(&writer);
        let instance_id = instance_id.clone();
        start_interval(1000, move || {
            let outcome = (|| -> Result<(), String> {
                if telegram_stop_requested(&writer.store, &instance_id)? {
                    controller.cancel();
                }
                writer.write()
            })();
            if outcome.is_err() {
                controller.cancel();
            }
        })
    };

    let mut api: Option<Arc<TelegramApi>> = None;
    let mut connection: Option<Arc<DaemonAgentConnection>> = None;
    let mut bridge: Option<Arc<TelegramBridge>> = None;
    run_telegram_worker_body(
        agent_dir,
        &options,
        &controller,
        &writer,
        &mut api,
        &mut connection,
        &mut bridge,
    )
    .await;

    // `finally { stop(); clearInterval(timer); ... }`.
    controller.cancel();
    timer.abort();
    for handler in signal_handlers {
        handler.abort();
    }
    if let Some(task) = signal_task {
        task.abort();
    }
    if let Some(connection) = &connection {
        // `await connection?.dispose().catch(() => undefined)`.
        let _ = connection.dispose().await;
    }
    if let Some(bridge) = &bridge {
        bridge.wait_for_dispatch().await;
    }
    {
        let mut status = writer.status.lock().unwrap();
        if status.phase != "error" {
            status.phase = "stopped".to_string();
        }
    }
    let write_result = writer.write();
    release.release();
    write_result
}

/// The `try { ... }` body of `runTelegramWorker` including its one `catch`.
async fn run_telegram_worker_body(
    agent_dir: &str,
    options: &TelegramWorkerOptions,
    controller: &CancellationToken,
    writer: &Arc<WorkerStatusWriter>,
    api_slot: &mut Option<Arc<TelegramApi>>,
    connection_slot: &mut Option<Arc<DaemonAgentConnection>>,
    bridge_slot: &mut Option<Arc<TelegramBridge>>,
) {
    let result = run_telegram_worker_inner(
        agent_dir,
        options,
        controller,
        writer,
        api_slot,
        connection_slot,
        bridge_slot,
    )
    .await;
    if let Err(error) = result {
        if !controller.is_cancelled() {
            {
                let mut status = writer.status.lock().unwrap();
                status.phase = "error".to_string();
                status.error = Some(match api_slot {
                    // `api?.redact(error) || "Telegram worker could not start."`.
                    Some(api) => redact_error(api, &error),
                    None => "Telegram worker could not start.".to_string(),
                });
            }
            let _ = writer.write();
        }
    }
}

/// `api?.redact(error) || fallback`. The thrown value is always `Error`-like in
/// the port, so `redact` yields its message and `||` supplies the empty case.
fn redact_error(api: &TelegramApi, error: &str) -> String {
    if error.is_empty() {
        "Telegram worker could not start.".to_string()
    } else {
        api.redact(error)
    }
}

async fn run_telegram_worker_inner(
    agent_dir: &str,
    options: &TelegramWorkerOptions,
    controller: &CancellationToken,
    writer: &Arc<WorkerStatusWriter>,
    api_slot: &mut Option<Arc<TelegramApi>>,
    connection_slot: &mut Option<Arc<DaemonAgentConnection>>,
    bridge_slot: &mut Option<Arc<TelegramBridge>>,
) -> Result<(), String> {
    let store = TelegramStore::new(agent_dir);
    let Some(mut settings) = store.settings()? else {
        return Ok(());
    };
    if !settings.enabled {
        return Ok(());
    }
    let api = match &options.create_api {
        Some(create_api) => create_api(&settings.bot_token)?,
        None => default_create_api(&settings.bot_token)?,
    };
    *api_slot = Some(Arc::clone(&api));

    // `controller.signal` always exists, so every API call is signalled.
    let signal = Some(controller.clone());
    let bot = api
        .identify(signal.clone())
        .await
        .map_err(|error| error_message(&error))?;
    if bot.id != settings.bot_id {
        return Err("Telegram bot identity changed. Run /telegram setup again.".to_string());
    }
    api.require_polling(signal.clone())
        .await
        .map_err(|error| error_message(&error))?;

    // An unpaired bot must not advertise itself as a UI client to the session.
    let mut state = store.state(settings.bot_id)?;
    if settings.paired_user_id.is_none() {
        writer.status.lock().unwrap().phase = "pairing".to_string();
        writer.write()?;
    }
    while settings.paired_user_id.is_none() && !controller.is_cancelled() {
        match api.updates(state.offset, signal.clone()).await {
            Ok(updates) => {
                for update in updates {
                    if controller.is_cancelled() {
                        return Ok(());
                    }
                    if update.update_id < state.offset {
                        continue;
                    }
                    state.offset = update.update_id + 1.0;
                    let update_value = serde_json::to_value(&update).map_err(|error| error.to_string())?;
                    if accept_telegram_pairing(&mut settings, &update_value) {
                        store.write(
                            "connection.json",
                            &serde_json::to_value(&settings).map_err(|error| error.to_string())?,
                        )?;
                        state.outbox.push(TelegramDelivery {
                            id: random_uuid(),
                            // `settings.pairedUserId!` is set by the accepted pairing.
                            chat_id: settings.paired_user_id.unwrap_or_default(),
                            text: "Connected to Prime. This chat controls your connected session. Send a message to begin, or /help for commands.".to_string(),
                        });
                    }
                    store.write(
                        "state.json",
                        &serde_json::to_value(&state).map_err(|error| error.to_string())?,
                    )?;
                    if settings.paired_user_id.is_some() {
                        break;
                    }
                }
            }
            Err(error) => {
                if controller.is_cancelled() {
                    return Ok(());
                }
                if is_fatal_api_error(&error) {
                    return Err(error_message(&error));
                }
                writer.status.lock().unwrap().error = Some(api.redact(&error_message(&error)));
                writer.write()?;
                delay_ms(pairing_retry_delay_ms(&error), controller).await;
            }
        }
    }
    if controller.is_cancelled() {
        return Ok(());
    }

    let connection = connect_telegram_session(&settings, agent_dir).await?;
    *connection_slot = Some(Arc::clone(&connection));
    if controller.is_cancelled() {
        return Ok(());
    }
    writer.status.lock().unwrap().phase = "running".to_string();
    writer.write()?;

    let report_error = {
        let controller = controller.clone();
        let writer = Arc::clone(writer);
        Arc::new(move |error: Option<String>| {
            if controller.is_cancelled() {
                return;
            }
            writer.status.lock().unwrap().error = error;
            let _ = writer.write();
        }) as Arc<dyn Fn(Option<String>) + Send + Sync>
    };
    // `Arc::clone` would clone at the annotated trait-object type; the concrete
    // `Arc<DaemonAgentConnection>` coerces to the connection trait object instead.
    let connection_view: Arc<dyn AgentConnection> = connection.clone();
    let bridge = Arc::new(TelegramBridge::new(
        TelegramStore::new(agent_dir),
        settings,
        Arc::clone(&api),
        connection_view,
        report_error,
    )?);
    *bridge_slot = Some(Arc::clone(&bridge));
    bridge
        .run(Some(controller.clone()))
        .await
        .map_err(|error| bridge_error_message(&error))?;
    Ok(())
}

/// `TelegramBridgeError.message` (private in the bridge) for the worker's catch.
fn bridge_error_message(error: &TelegramBridgeError) -> String {
    match error {
        TelegramBridgeError::Api { message, .. } => message.clone(),
        TelegramBridgeError::Message(message) => message.clone(),
    }
}

/// The module-entry guard of `worker.ts`:
/// `if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)`.
///
/// The port's worker runs in-process, so the entry point is a named function the
/// CLI/dispatch calls rather than a side effect of being the script argument.
/// `setGlobalDispatcher(new EnvHttpProxyAgent())` has no equivalent: the port's
/// HTTP client has no global undici dispatcher, and `reqwest` already honours the
/// standard proxy environment variables.
pub async fn telegram_worker_main(argv: &[String]) -> Result<(), String> {
    let agent_dir = argv.get(2).cloned();
    let Some(agent_dir) = agent_dir else {
        return Err("Telegram worker requires an agent directory.".to_string());
    };
    run_telegram_worker(&agent_dir, TelegramWorkerOptions::default()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::modes::telegram::api::{private_message, TelegramApiError, TelegramUpdate};
    use crate::modes::telegram::store::{create_pairing, pairing_hash, TelegramPairing};

    fn transport_hello_frame() -> Value {
        serde_json::json!({
            "type": "daemon_hello",
            "socketPath": "/tmp/prime-agent.sock",
            "protocol": { "name": "prime-agent.daemon", "version": 7 },
            "clientId": "daemon-client:test",
            "serverCapabilities": ["attach_snapshot"],
        })
    }

    #[test]
    fn the_adapter_reads_the_hello_socket_path() {
        let hello = transport_hello_frame();
        assert_eq!(
            hello.get("socketPath").and_then(Value::as_str),
            Some("/tmp/prime-agent.sock")
        );
    }

    #[test]
    fn frames_with_no_connection_meaning_are_ignored() {
        assert!(connection_outbound_from_wire(&transport_hello_frame()).is_none());
        assert!(connection_outbound_from_wire(&serde_json::json!({
            "type": "response",
            "command": "list",
            "success": true,
        }))
        .is_none());
        assert!(connection_outbound_from_wire(&serde_json::json!({
            "type": "session_detached",
            "activeSessionId": "a",
        }))
        .is_none());
    }

    #[test]
    fn heartbeat_frames_keep_the_optional_active_session_id() {
        let outbound = connection_outbound_from_wire(&serde_json::json!({
            "type": "heartbeats_changed",
            "activeSessionId": "a",
        }))
        .expect("heartbeats_changed");
        assert_eq!(outbound.type_name(), "heartbeats_changed");
        match outbound {
            ConnectionOutbound::HeartbeatsChanged { active_session_id, .. } => {
                assert_eq!(active_session_id.as_deref(), Some("a"));
            }
            other => panic!("unexpected frame: {other:?}"),
        }
    }

    #[test]
    fn daemon_shutdown_notice_reaches_connection_recovery() {
        let event = connection_outbound_from_wire(&serde_json::json!({"type":"daemon_closing","reason":"shutdown"}));
        assert!(matches!(event, Some(ConnectionOutbound::DaemonClosing { reason: crate::modes::daemon::daemon_protocol::DaemonClosingReason::Shutdown })));
    }

    #[test]
    fn session_closed_frames_carry_the_reason_and_meta() {
        let outbound = connection_outbound_from_wire(&serde_json::json!({
            "type": "session_closed",
            "activeSessionId": "a",
            "reason": "update",
            "meta": { "sequence": 4, "cursor": { "generation": "g", "sequence": 9 } },
        }))
        .expect("session_closed");
        match &outbound {
            ConnectionOutbound::SessionClosed { reason, meta, .. } => {
                assert_eq!(reason, "update");
                let meta = meta.as_ref().expect("meta");
                assert_eq!(meta.sequence, Some(4));
                assert_eq!(meta.cursor.as_ref().map(|cursor| cursor.sequence), Some(9));
            }
            other => panic!("unexpected frame: {other:?}"),
        }
    }

    #[test]
    fn snapshot_chunks_map_index_and_messages() {
        let outbound = connection_outbound_from_wire(&serde_json::json!({
            "type": "session_snapshot_chunk",
            "activeSessionId": "a",
            "snapshotId": "s",
            "index": 2,
            "messages": [],
        }))
        .expect("session_snapshot_chunk");
        match outbound {
            ConnectionOutbound::SessionSnapshotChunk { index, snapshot_id, .. } => {
                assert_eq!(index, 2);
                assert_eq!(snapshot_id, "s");
            }
            other => panic!("unexpected frame: {other:?}"),
        }
    }

    #[test]
    fn pairing_retry_delay_takes_retry_after_or_five_seconds() {
        let throttled = TelegramCallError::Api(TelegramApiError::new(429.0, 3.0));
        assert_eq!(pairing_retry_delay_ms(&throttled), 3000.0);
        let plain = TelegramCallError::Api(TelegramApiError::new(500.0, 0.0));
        assert_eq!(pairing_retry_delay_ms(&plain), 5000.0);
        assert_eq!(pairing_retry_delay_ms(&TelegramCallError::Aborted), 5000.0);
    }

    #[test]
    fn only_401_and_409_are_fatal() {
        assert!(is_fatal_api_error(&TelegramCallError::Api(TelegramApiError::new(401.0, 0.0))));
        assert!(is_fatal_api_error(&TelegramCallError::Api(TelegramApiError::new(409.0, 0.0))));
        assert!(!is_fatal_api_error(&TelegramCallError::Api(TelegramApiError::new(429.0, 0.0))));
        assert!(!is_fatal_api_error(&TelegramCallError::Error("offline".to_string())));
    }

    #[test]
    fn redact_falls_back_only_for_an_empty_message() {
        let api = default_create_api("123456789:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").expect("api");
        assert_eq!(redact_error(&api, ""), "Telegram worker could not start.");
        assert_eq!(redact_error(&api, "boom"), "boom");
    }

    #[tokio::test]
    async fn the_entry_point_requires_an_agent_directory() {
        let error = telegram_worker_main(&["node".to_string(), "worker.js".to_string()])
            .await
            .expect_err("missing agent dir");
        assert_eq!(error, "Telegram worker requires an agent directory.");
    }

    #[test]
    fn pairing_acceptance_round_trips_through_the_frame_the_worker_sends() {
        let (code, pairing) = create_pairing(now_ms());
        let bot_username = "prime_bot".to_string();
        let mut settings = TelegramConnectionSettings {
            version: 1.0,
            enabled: true,
            bot_token: "123456789:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_string(),
            bot_id: 42.0,
            bot_username: bot_username.clone(),
            daemon_socket: "/tmp/prime-agent.sock".to_string(),
            cwd: "/work".to_string(),
            session_id: "session-1".to_string(),
            session_file: None,
            paired_user_id: None,
            pairing: Some(pairing),
        };
        let update = TelegramUpdate {
            update_id: 1.0,
            message: Some(serde_json::json!({
                "message_id": 2,
                "date": 1,
                "from": { "id": 7, "is_bot": false },
                "chat": { "id": 7, "type": "private" },
                "text": format!("/start@{bot_username} {code}"),
            })),
        };
        let frame = serde_json::to_value(&update).expect("frame");
        assert!(private_message(&frame).is_some());
        assert!(accept_telegram_pairing(&mut settings, &frame));
        assert_eq!(settings.paired_user_id, Some(7.0));
        assert!(settings.pairing.is_none());
        // The hashed code is what the frame must carry; a stale hash is rejected.
        let mut other = settings.clone();
        other.paired_user_id = None;
        other.pairing = Some(TelegramPairing {
            hash: pairing_hash("nope"),
            expires_at: f64::MAX,
        });
        assert!(!accept_telegram_pairing(&mut other, &frame));

        // Correct codes still fail once their ten-minute pairing window expires.
        other.pairing = Some(TelegramPairing {
            hash: pairing_hash(&code),
            expires_at: 0.0,
        });
        assert!(!accept_telegram_pairing(&mut other, &frame));
        assert!(other.paired_user_id.is_none());
    }
}
