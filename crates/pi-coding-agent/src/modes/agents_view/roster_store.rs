//! Port of packages/coding-agent/src/modes/agents-view/roster-store.ts
//!
//! Types owned by other slices (daemon-protocol.ts, daemon-client.ts,
//! agent-roster.ts) are declared locally here with only the fields this store
//! observes. TODO(port): re-point them at the daemon slice types when that
//! slice lands (see blocked_on in evidence/status/ca-agents-view.json).

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;

use super::agents_view_state::{
    AgentRosterStatus, AgentRosterStatusLabel, SessionActionSnapshot, SessionSummary,
};

pub const STALE_ROSTER_DAEMON_MESSAGE: &str =
    "Daemon is stale: it does not advertise the agent_roster capability; restart the daemon";

/// Local stand-in for `WorkerRosterEntry`/`AgentRosterEntry`
/// (modes/daemon/agent-roster.ts).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentRosterEntry {
    pub agent_id: String,
    pub queued_child: Option<bool>,
    pub seeded_cwd: Option<bool>,
    pub summary: SessionSummary,
    pub status: AgentRosterStatus,
    pub status_label: Option<AgentRosterStatusLabel>,
    pub last_heard_from_at: Option<String>,
    pub worker_id: Option<String>,
}

/// Port of `sessionSummaryFromRosterEntry` (modes/daemon/agent-roster.ts).
pub fn session_summary_from_roster_entry(entry: &AgentRosterEntry) -> SessionSummary {
    // statusLabel/lastHeardFromAt are set only for exceptional states; viewers
    // key label display on their presence.
    let mut summary = entry.summary.clone();
    summary.session_actions = SessionActionSnapshot::default();
    summary.roster_status = Some(entry.status);
    if let Some(label) = entry.status_label {
        summary.status_label = Some(label);
    }
    if let Some(last_heard) = &entry.last_heard_from_at {
        summary.last_heard_from_at = Some(last_heard.clone());
    }
    summary
}

/// Local stand-in for `DaemonResponse` (modes/daemon/daemon-protocol.ts).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DaemonResponse {
    pub command: String,
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// Local stand-in for the `roster_update` variant of `DaemonOutbound`.
#[derive(Clone, Debug, PartialEq)]
pub struct RosterUpdate {
    pub changed: Vec<AgentRosterEntry>,
    pub removed: Option<Vec<String>>,
    pub resync: Option<bool>,
}

/// Local stand-in for `DaemonOutbound`; only the roster push is modelled.
#[derive(Clone, Debug, PartialEq)]
pub enum DaemonOutbound {
    RosterUpdate { changed: Vec<AgentRosterEntry>, removed: Option<Vec<String>>, resync: Option<bool> },
    HeartbeatsChanged,
    Other,
}

impl RosterUpdate {
    pub fn from_outbound(message: &DaemonOutbound) -> Option<Self> {
        match message {
            DaemonOutbound::RosterUpdate { changed, removed, resync } => Some(Self {
                changed: changed.clone(),
                removed: removed.clone(),
                resync: *resync,
            }),
            _ => None,
        }
    }
}

/// Local stand-in for `DaemonHello` (modes/daemon/daemon-client.ts).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DaemonHello {
    pub socket_path: String,
    #[serde(default)]
    pub server_capabilities: Vec<String>,
    #[serde(default)]
    pub client_id: String,
    #[serde(default)]
    pub schema_revision: Option<i64>,
}

pub type TransportFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
pub type MessageListener = Box<dyn Fn(&DaemonOutbound) + Send + Sync>;
pub type ProgressListener = Box<dyn Fn(&serde_json::Value) + Send + Sync>;

/// `DaemonClientRequestOptions` (modes/daemon/daemon-client.ts).
#[derive(Default)]
pub struct DaemonClientRequestOptions {
    pub signal: Option<tokio_util::sync::CancellationToken>,
    pub on_progress: Option<ProgressListener>,
    /// False opts out of reconnect parking: a close rejects so the caller's own
    /// retry loop stays live. Any caller that owns its own bounded retry MUST
    /// pass false; a parked request waits for a hello that only the caller's
    /// stuck loop could produce.
    pub recoverable: Option<bool>,
}

impl DaemonClientRequestOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_progress(mut self, on_progress: ProgressListener) -> Self {
        self.on_progress = Some(on_progress);
        self
    }

    pub fn recoverable(mut self, recoverable: bool) -> Self {
        self.recoverable = Some(recoverable);
        self
    }
}

/// Local port of the `DaemonTransportClient` interface (daemon-client.ts). The
/// daemon slice owns the socket implementation; this store only calls these.
pub trait DaemonTransport: Send + Sync {
    /// A separate connection for one-shot operations that close their socket.
    fn fresh_transport(&self) -> Option<Arc<dyn DaemonTransport>> { None }
    /// Adapt this connected socket for the session viewer without dialing again.
    fn session_transport(&self) -> Option<Arc<dyn crate::modes::agent_connection::daemon_agent_connection::DaemonTransportClient>> { None }
    fn hello(&self) -> Option<DaemonHello>;
    fn is_connected(&self) -> bool;
    fn supports_server_capability(&self, capability: &str) -> bool;
    fn wait_for_hello(&self, timeout_ms: u64) -> TransportFuture<Result<DaemonHello, String>>;
    fn request(
        &self,
        command: serde_json::Value,
        timeout_ms: u64,
        options: DaemonClientRequestOptions,
    ) -> TransportFuture<Result<DaemonResponse, String>>;
    fn on_message(&self, listener: MessageListener) -> Box<dyn Fn() + Send + Sync>;
    fn on_close(&self, listener: Box<dyn Fn(&str) + Send + Sync>) -> Box<dyn Fn() + Send + Sync>;
    fn connect(&self, timeout_ms: u64) -> TransportFuture<Result<(), String>>;
    fn reconnect(&self, timeout_ms: u64) -> TransportFuture<Result<(), String>>;
    fn close(&self);
    /// Stable socket identity so a stale attach cannot detach a newer subscription.
    fn socket_path(&self) -> String;
}

#[derive(Clone)]
pub struct DaemonTransportClient {
    inner: Arc<dyn DaemonTransport>,
}

impl DaemonTransportClient {
    pub(super) fn session_transport(&self) -> Option<Arc<dyn crate::modes::agent_connection::daemon_agent_connection::DaemonTransportClient>> {
        self.inner.session_transport()
    }

    pub fn new(inner: Arc<dyn DaemonTransport>) -> Self {
        Self { inner }
    }

    pub fn hello(&self) -> Option<DaemonHello> {
        self.inner.hello()
    }

    pub fn is_connected(&self) -> bool {
        self.inner.is_connected()
    }

    pub fn supports_server_capability(&self, capability: &str) -> bool {
        self.inner.supports_server_capability(capability)
    }

    pub async fn wait_for_hello(&self, timeout_ms: u64) -> Result<DaemonHello, String> {
        self.inner.wait_for_hello(timeout_ms).await
    }

    /// `client.request(command, timeoutMs?, options?)` with the reference defaults.
    pub async fn request(
        &self,
        command: serde_json::Value,
        timeout_ms: u64,
    ) -> Result<DaemonResponse, String> {
        self.inner
            .request(command, timeout_ms, DaemonClientRequestOptions::new())
            .await
    }

    pub async fn request_with_options(
        &self,
        command: serde_json::Value,
        timeout_ms: u64,
        options: DaemonClientRequestOptions,
    ) -> Result<DaemonResponse, String> {
        self.inner.request(command, timeout_ms, options).await
    }

    pub fn on_message(&self, listener: MessageListener) -> Box<dyn Fn() + Send + Sync> {
        self.inner.on_message(listener)
    }

    pub fn on_close(&self, listener: Box<dyn Fn(&str) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        self.inner.on_close(listener)
    }

    pub async fn connect(&self, timeout_ms: u64) -> Result<(), String> {
        self.inner.connect(timeout_ms).await
    }

    pub async fn reconnect(&self, timeout_ms: u64) -> Result<(), String> {
        self.inner.reconnect(timeout_ms).await
    }

    pub fn close(&self) {
        self.inner.close()
    }

    pub fn same_socket(&self, other: &Self) -> bool {
        self.inner.socket_path() == other.inner.socket_path()
    }
}

impl std::fmt::Debug for DaemonTransportClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonTransportClient")
            .field("socket_path", &self.inner.socket_path())
            .finish()
    }
}

type Listener = Arc<dyn Fn() + Send + Sync>;

/// Mirrors `AgentsViewRosterStore`: one subscription per attached client, with
/// pushes buffered until the subscribe reply lands.
#[derive(Clone)]
pub struct AgentsViewRosterStore {
    entries: Arc<AsyncMutex<Vec<AgentRosterEntry>>>,
    listeners: Arc<AsyncMutex<Vec<Listener>>>,
    client: Arc<AsyncMutex<Option<DaemonTransportClient>>>,
    unsubscribe_message: Arc<AsyncMutex<Option<Box<dyn Fn() + Send + Sync>>>>,
    emit_scheduled: Arc<AsyncMutex<bool>>,
    subscribed: Arc<AsyncMutex<bool>>,
    subscribed_hello: Arc<AsyncMutex<Option<DaemonHello>>>,
    attach_chain: Arc<AsyncMutex<()>>,
}

impl Default for AgentsViewRosterStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentsViewRosterStore {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(AsyncMutex::new(Vec::new())),
            listeners: Arc::new(AsyncMutex::new(Vec::new())),
            client: Arc::new(AsyncMutex::new(None)),
            unsubscribe_message: Arc::new(AsyncMutex::new(None)),
            emit_scheduled: Arc::new(AsyncMutex::new(false)),
            subscribed: Arc::new(AsyncMutex::new(false)),
            subscribed_hello: Arc::new(AsyncMutex::new(None)),
            attach_chain: Arc::new(AsyncMutex::new(())),
        }
    }


    /// Serialized: a stale attempt settling late must not detach a newer
    /// subscription's listener. The `Err` mirrors the reference's `throw`.
    pub async fn attach(&self, client: DaemonTransportClient) -> Result<bool, String> {
        let _guard = self.attach_chain.lock().await;
        self.attach_to_client(client).await
    }

    async fn attach_to_client(&self, client: DaemonTransportClient) -> Result<bool, String> {
        if client.is_connected() && client.hello().is_none() {
            let _ = client.wait_for_hello(3000).await;
        }
        if !client.supports_server_capability("agent_roster") {
            self.detach_from_client().await;
            return Ok(false);
        }
        let hello = client.hello();
        {
            let subscribed = *self.subscribed.lock().await;
            let same_client = self
                .client
                .lock()
                .await
                .as_ref()
                .map(|current| current.same_socket(&client))
                .unwrap_or(false);
            let same_hello = *self.subscribed_hello.lock().await == hello;
            if subscribed && same_client && client.is_connected() && same_hello {
                return Ok(true);
            }
        }
        self.detach_from_client().await;
        *self.client.lock().await = Some(client.clone());
        let (updates, mut incoming) = tokio::sync::mpsc::unbounded_channel::<RosterUpdate>();
        let ready = tokio_util::sync::CancellationToken::new();
        let ready_consumer = ready.clone();
        let store = self.clone();
        let consumer = tokio::spawn(async move {
            ready_consumer.cancelled().await;
            while let Some(update) = incoming.recv().await {
                store.apply_update(update.changed, update.removed, update.resync).await;
            }
        });
        let unsubscribe = client.on_message(Box::new(move |message| {
            if let Some(update) = RosterUpdate::from_outbound(message) {
                let _ = updates.send(update);
            }
        }));
        *self.unsubscribe_message.lock().await = Some(Box::new(move || {
            unsubscribe();
            consumer.abort();
        }));

        // Not parkable: the awaiting reconnect loop must see a close as a rejection.
        let response = match client
            .request_with_options(roster_subscribe_command(), 30000, DaemonClientRequestOptions::new().recoverable(false))
            .await
        {
            Ok(response) => response,
            Err(error) => {
                self.detach_from_client().await;
                return Err(error);
            }
        };
        if !response.success {
            let error = response.error.clone().unwrap_or_else(|| "unknown error".to_string());
            self.detach_from_client().await;
            return Err(format!("roster_subscribe failed: {error}"));
        }
        let Some(data) = response.data.clone() else {
            self.detach_from_client().await;
            return Err("roster_subscribe failed: invalid roster payload".to_string());
        };
        if !data.is_object() {
            self.detach_from_client().await;
            return Err("roster_subscribe failed: invalid roster payload".to_string());
        }
        let roster: Vec<AgentRosterEntry> = data
            .get("roster")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| serde_json::from_value::<AgentRosterEntry>(value.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();
        self.apply_update(roster, None, Some(true)).await;
        ready.cancel();
        *self.subscribed.lock().await = true;
        *self.subscribed_hello.lock().await = hello;
        Ok(true)
    }

    pub async fn apply_update(
        &self,
        changed: Vec<AgentRosterEntry>,
        removed: Option<Vec<String>>,
        resync: Option<bool>,
    ) {
        {
            let mut entries = self.entries.lock().await;
            if resync == Some(true) {
                entries.clear();
            }
            for entry in changed {
                match entries.iter_mut().find(|existing| existing.agent_id == entry.agent_id) {
                    Some(existing) => *existing = entry,
                    None => entries.push(entry),
                }
            }
            if let Some(removed) = removed {
                entries.retain(|entry| !removed.contains(&entry.agent_id));
            }
        }
        self.schedule_emit().await;
    }

    pub async fn summaries(&self) -> Vec<SessionSummary> {
        self.entries
            .lock()
            .await
            .iter()
            .map(session_summary_from_roster_entry)
            .collect()
    }

    pub async fn on_update(&self, listener: Listener) -> usize {
        let mut listeners = self.listeners.lock().await;
        listeners.push(listener);
        listeners.len() - 1
    }

    pub async fn remove_listener(&self, index: usize) {
        let mut listeners = self.listeners.lock().await;
        if index < listeners.len() {
            listeners.remove(index);
        }
    }

    /// `queueMicrotask`: the flag makes several updates in one task coalesce
    /// into a single listener pass.
    async fn schedule_emit(&self) {
        {
            let mut scheduled = self.emit_scheduled.lock().await;
            if *scheduled {
                return;
            }
            *scheduled = true;
        }
        tokio::task::yield_now().await;
        let listeners: Vec<Listener> = self.listeners.lock().await.clone();
        *self.emit_scheduled.lock().await = false;
        for listener in listeners {
            // One consumer must not interrupt delivery to the others.
            listener();
        }
    }

    async fn detach_from_client(&self) {
        if let Some(unsubscribe) = self.unsubscribe_message.lock().await.take() {
            unsubscribe();
        }
        *self.subscribed.lock().await = false;
        *self.subscribed_hello.lock().await = None;
        *self.client.lock().await = None;
    }

    pub async fn dispose(&self) {
        let _guard = self.attach_chain.lock().await;
        self.dispose_now().await;
    }

    async fn dispose_now(&self) {
        let client = self.client.lock().await.clone();
        self.detach_from_client().await;
        self.listeners.lock().await.clear();
        // Fire-and-forget: nobody needs the ack, and a closed socket already unsubscribed.
        if let Some(client) = client {
            if client.is_connected() {
                tokio::spawn(async move {
                    let _ = client.request(roster_unsubscribe_command(), 30000).await;
                });
            }
        }
    }
}

pub fn roster_subscribe_command() -> serde_json::Value {
    serde_json::json!({ "type": "roster_subscribe" })
}

pub fn roster_unsubscribe_command() -> serde_json::Value {
    serde_json::json!({ "type": "roster_unsubscribe" })
}

#[cfg(test)]
pub mod test_support {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Fake transport: scripted hello/capabilities plus recorded requests.
    pub struct FakeDaemonTransport {
        pub socket_path: String,
        pub hello: Option<DaemonHello>,
        pub connected: AtomicBool,
        pub requests: Mutex<Vec<serde_json::Value>>,
        pub responses: Mutex<Vec<Result<DaemonResponse, String>>>,
        pub listeners: Arc<Mutex<Vec<Arc<dyn Fn(&DaemonOutbound) + Send + Sync>>>>,
    }

    impl FakeDaemonTransport {
        pub fn new(socket_path: &str, capabilities: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                socket_path: socket_path.to_string(),
                hello: Some(DaemonHello {
                    socket_path: socket_path.to_string(),
                    server_capabilities: capabilities.iter().map(|value| value.to_string()).collect(),
                    client_id: "client-1".to_string(),
                    schema_revision: None,
                }),
                connected: AtomicBool::new(true),
                requests: Mutex::new(Vec::new()),
                responses: Mutex::new(Vec::new()),
                listeners: Arc::new(Mutex::new(Vec::new())),
            })
        }

        pub fn push(&self, response: Result<DaemonResponse, String>) {
            self.responses.lock().unwrap().push(response);
        }

        pub fn emit(&self, message: DaemonOutbound) {
            let listeners = self.listeners.lock().unwrap().clone();
            for listener in listeners {
                listener(&message);
            }
        }

        pub fn recorded_requests(&self) -> Vec<serde_json::Value> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl DaemonTransport for FakeDaemonTransport {
        fn hello(&self) -> Option<DaemonHello> {
            self.hello.clone()
        }
        fn is_connected(&self) -> bool {
            self.connected.load(Ordering::SeqCst)
        }
        fn supports_server_capability(&self, capability: &str) -> bool {
            self.hello
                .as_ref()
                .map(|hello| hello.server_capabilities.iter().any(|value| value == capability))
                .unwrap_or(false)
        }
        fn wait_for_hello(&self, _timeout_ms: u64) -> TransportFuture<Result<DaemonHello, String>> {
            let hello = self.hello.clone();
            Box::pin(async move { hello.ok_or_else(|| "no hello".to_string()) })
        }
        fn request(
            &self,
            command: serde_json::Value,
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
        fn on_message(&self, listener: MessageListener) -> Box<dyn Fn() + Send + Sync> {
            let shared: Arc<dyn Fn(&DaemonOutbound) + Send + Sync> = Arc::from(listener);
            let listeners = Arc::clone(&self.listeners);
            listeners.lock().unwrap().push(Arc::clone(&shared));
            Box::new(move || {
                // `onMessage` returns `() => listeners.delete(listener)`.
                listeners
                    .lock()
                    .unwrap()
                    .retain(|candidate| !Arc::ptr_eq(candidate, &shared));
            })
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
            self.socket_path.clone()
        }
    }

    pub fn ok_response(data: serde_json::Value) -> DaemonResponse {
        DaemonResponse {
            command: "roster_subscribe".into(),
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn err_response(error: &str) -> DaemonResponse {
        DaemonResponse {
            command: "roster_subscribe".into(),
            success: false,
            data: None,
            error: Some(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{err_response, ok_response, FakeDaemonTransport};
    use super::*;
    use crate::modes::agents_view::agents_view_state::SessionUsageSummary;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn entry(agent_id: &str, session_id: &str) -> AgentRosterEntry {
        let mut summary = SessionSummary::new(agent_id, session_id, "C:/w");
        summary.usage = Some(SessionUsageSummary { input_tokens: 1, output_tokens: 2, cost: 0.5 });
        AgentRosterEntry {
            agent_id: agent_id.to_string(),
            summary,
            status: AgentRosterStatus::Idle,
            ..AgentRosterEntry::default()
        }
    }

    #[test]
    fn stale_daemon_message_is_exact() {
        assert_eq!(
            STALE_ROSTER_DAEMON_MESSAGE,
            "Daemon is stale: it does not advertise the agent_roster capability; restart the daemon"
        );
    }

    #[tokio::test]
    async fn updates_upsert_and_resync_clear_entries() {
        let store = AgentsViewRosterStore::new();
        store.apply_update(vec![entry("a", "s-a")], None, None).await;
        store.apply_update(vec![entry("b", "s-b")], None, None).await;
        assert_eq!(store.summaries().await.len(), 2);

        let mut updated = entry("a", "s-a");
        updated.summary.session_name = Some("Renamed".into());
        store.apply_update(vec![updated], None, None).await;
        let summaries = store.summaries().await;
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].session_name.as_deref(), Some("Renamed"));

        store.apply_update(Vec::new(), Some(vec!["a".to_string()]), None).await;
        assert_eq!(store.summaries().await.len(), 1);

        store.apply_update(vec![entry("c", "s-c")], None, Some(true)).await;
        let summaries = store.summaries().await;
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].session_id, "s-c");
    }

    #[tokio::test]
    async fn summaries_carry_roster_status_and_reset_session_actions() {
        let store = AgentsViewRosterStore::new();
        let mut record = entry("a", "s-a");
        record.status = AgentRosterStatus::Running;
        record.status_label = Some(AgentRosterStatusLabel::Queued);
        record.last_heard_from_at = Some("2026-01-01T00:00:00.000Z".into());
        store.apply_update(vec![record], None, None).await;
        let summary = store.summaries().await.remove(0);
        assert_eq!(summary.roster_status, Some(AgentRosterStatus::Running));
        assert_eq!(summary.status_label, Some(AgentRosterStatusLabel::Queued));
        assert_eq!(summary.last_heard_from_at.as_deref(), Some("2026-01-01T00:00:00.000Z"));
        assert_eq!(summary.session_actions.queued_count, 0);
        assert!(summary.session_actions.steering.is_empty());
    }

    #[tokio::test]
    async fn attach_rejects_a_daemon_without_the_roster_capability() {
        let store = AgentsViewRosterStore::new();
        let transport = FakeDaemonTransport::new("pipe-no-cap", &[]);
        let client = DaemonTransportClient::new(transport);
        assert_eq!(store.attach(client).await, Ok(false));
    }

    #[tokio::test]
    async fn attach_applies_the_snapshot_and_records_the_command() {
        let store = AgentsViewRosterStore::new();
        let transport = FakeDaemonTransport::new("pipe-ok", &["agent_roster"]);
        transport.push(Ok(ok_response(serde_json::json!({
            "roster": [serde_json::to_value(entry("a", "s-a")).unwrap()]
        }))));
        let client = DaemonTransportClient::new(transport.clone());
        assert_eq!(store.attach(client).await, Ok(true));
        assert_eq!(store.summaries().await.len(), 1);
        let requests = transport.recorded_requests();
        assert_eq!(requests[0]["type"], "roster_subscribe");
    }

    #[tokio::test]
    async fn attach_reports_a_failed_subscribe() {
        let store = AgentsViewRosterStore::new();
        let transport = FakeDaemonTransport::new("pipe-err", &["agent_roster"]);
        transport.push(Ok(err_response("boom")));
        let client = DaemonTransportClient::new(transport);
        assert_eq!(store.attach(client).await, Err("roster_subscribe failed: boom".to_string()));
    }

    #[tokio::test]
    async fn attached_store_applies_live_updates_in_order_and_detaches() {
        let store = AgentsViewRosterStore::new();
        let transport = FakeDaemonTransport::new("pipe-live", &["agent_roster"]);
        transport.push(Ok(ok_response(serde_json::json!({"roster": []}))));
        store.attach(DaemonTransportClient::new(transport.clone())).await.unwrap();
        transport.emit(DaemonOutbound::RosterUpdate { changed: vec![entry("a", "first")], removed: None, resync: None });
        transport.emit(DaemonOutbound::RosterUpdate { changed: vec![entry("a", "last")], removed: None, resync: None });
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if store.summaries().await.first().is_some_and(|row| row.session_id == "last") { break; }
                tokio::task::yield_now().await;
            }
        }).await.unwrap();
        store.dispose().await;
        assert!(transport.listeners.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn attach_reports_an_invalid_payload() {
        let store = AgentsViewRosterStore::new();
        let transport = FakeDaemonTransport::new("pipe-invalid", &["agent_roster"]);
        transport.push(Ok(ok_response(serde_json::json!("nope"))));
        let client = DaemonTransportClient::new(transport);
        assert_eq!(
            store.attach(client).await,
            Err("roster_subscribe failed: invalid roster payload".to_string())
        );
    }

    #[tokio::test]
    async fn update_listeners_fire_once_per_update() {
        let store = AgentsViewRosterStore::new();
        let count = Arc::new(AtomicUsize::new(0));
        let counter = count.clone();
        store
            .on_update(Arc::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }))
            .await;
        store.apply_update(vec![entry("a", "s-a")], None, None).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn roster_commands_use_the_wire_type_names() {
        assert_eq!(roster_subscribe_command()["type"], "roster_subscribe");
        assert_eq!(roster_unsubscribe_command()["type"], "roster_unsubscribe");
    }
}
