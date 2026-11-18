//! Port of packages/coding-agent/src/modes/daemon/daemon-routed-client.ts

use indexmap::IndexMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::Value;

use super::daemon_client::{
    get_daemon_socket_close_reason, DaemonClientCloseListener, DaemonClientError, DaemonClientMessageListener,
    DaemonClientRequestOptions, DaemonClientResult, DaemonCommandBody, DaemonHello, DaemonSocketClosedError,
    command_compatibilities, compatibility_hello,
    DaemonTransportClient,
};
use super::daemon_protocol::{
    is_session_plane_daemon_command, meets_daemon_command_compatibility, DaemonResponse,
};
use super::daemon_socket::get_daemon_socket_identity;
use super::daemon_worker_client::DaemonWorkerClient;
use super::daemon_worker_protocol::DaemonPeerTransportTicket;

/// `DaemonDirectTransportClosedError extends DaemonSocketClosedError`.
#[derive(Debug, Clone, PartialEq)]
pub struct DaemonDirectTransportClosedError {
    pub socket_path: String,
    pub daemon_closing_reason: Option<String>,
    pub cause: Option<String>,
}

impl DaemonDirectTransportClosedError {
    pub fn new(cause: &DaemonClientError) -> Self {
        Self {
            socket_path: "direct-worker".to_string(),
            daemon_closing_reason: get_daemon_socket_close_reason(cause),
            cause: Some(cause.message()),
        }
    }

    pub fn message(&self) -> String {
        DaemonSocketClosedError::new(
            &self.socket_path,
            self.daemon_closing_reason.as_deref(),
            self.cause.as_deref(),
        )
        .message()
    }

    pub fn to_client_error(&self) -> DaemonClientError {
        DaemonClientError::SocketClosed(DaemonSocketClosedError::new(
            &self.socket_path,
            self.daemon_closing_reason.as_deref(),
            self.cause.as_deref(),
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DaemonControlPlaneTransportError {
    pub message: String,
}

impl DaemonControlPlaneTransportError {
    pub fn new(cause: &str) -> Self {
        Self {
            message: format!("Daemon control-plane transport failed: {cause}"),
        }
    }
}

impl std::fmt::Display for DaemonControlPlaneTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DaemonControlPlaneTransportError {}

/// One logical daemon connection over two sockets: session-plane commands go
/// direct to the worker, the rest to the supervisor. Any direct-path loss
/// degrades silently to supervisor routing.
pub struct DaemonRoutedClient {
    supervisor: Arc<dyn DaemonTransportClient>,
    direct: StdMutex<Option<Arc<DaemonWorkerClient>>>,
    message_listeners: Arc<StdMutex<IndexMap<u64, DaemonClientMessageListener>>>,
    close_listeners: Arc<StdMutex<IndexMap<u64, DaemonClientCloseListener>>>,
    next_listener_id: AtomicU64,
    unsubscribe_supervisor_message: StdMutex<Option<Box<dyn Fn() + Send + Sync>>>,
    unsubscribe_supervisor_close: StdMutex<Option<Box<dyn Fn() + Send + Sync>>>,
    unsubscribe_direct_message: StdMutex<Option<Box<dyn Fn() + Send + Sync>>>,
    unsubscribe_direct_close: StdMutex<Option<Box<dyn Fn() + Send + Sync>>>,
    closed: AtomicBool,
    self_ref: std::sync::Weak<Self>,
}

impl DaemonRoutedClient {
    pub fn new(supervisor: Arc<dyn DaemonTransportClient>, direct: Arc<DaemonWorkerClient>) -> Arc<Self> {
        let client = Arc::new_cyclic(|self_ref| Self {
            supervisor: Arc::clone(&supervisor),
            direct: StdMutex::new(Some(Arc::clone(&direct))),
            message_listeners: Arc::new(StdMutex::new(IndexMap::new())),
            close_listeners: Arc::new(StdMutex::new(IndexMap::new())),
            next_listener_id: AtomicU64::new(0),
            unsubscribe_supervisor_message: StdMutex::new(None),
            unsubscribe_supervisor_close: StdMutex::new(None),
            unsubscribe_direct_message: StdMutex::new(None),
            unsubscribe_direct_close: StdMutex::new(None),
            closed: AtomicBool::new(false),
            self_ref: self_ref.clone(),
        });
        let weak = Arc::downgrade(&client);
        let unsubscribe_message = supervisor.on_message(Arc::new(move |message: &Value| {
            if let Some(client) = weak.upgrade() {
                client.emit_message(message);
            }
        }));
        *client
            .unsubscribe_supervisor_message
            .lock()
            .expect("supervisor message slot poisoned") = Some(unsubscribe_message);
        let weak = Arc::downgrade(&client);
        let unsubscribe_close = supervisor.on_close(Arc::new(move |error: &DaemonClientError| {
            if let Some(client) = weak.upgrade() {
                if !client.closed.load(Ordering::SeqCst) {
                    client.emit_close(error.clone());
                }
            }
        }));
        *client
            .unsubscribe_supervisor_close
            .lock()
            .expect("supervisor close slot poisoned") = Some(unsubscribe_close);
        client.bind_direct(&direct);
        client
    }

    pub fn hello(&self) -> Option<DaemonHello> {
        self.supervisor.hello().or_else(|| self.direct_client().and_then(|direct| direct.hello()))
    }

    pub fn is_connected(&self) -> bool {
        self.supervisor.is_connected()
            || self
                .direct_client()
                .map(|direct| direct.is_connected())
                .unwrap_or(false)
    }

    pub fn has_direct_transport(&self) -> bool {
        self.direct_client()
            .map(|direct| direct.is_connected())
            .unwrap_or(false)
    }

    pub fn control_plane_transport(&self) -> Arc<dyn DaemonTransportClient> {
        Arc::clone(&self.supervisor)
    }

    pub fn is_control_plane_ready(&self) -> bool {
        self.supervisor.is_connected() && self.supervisor.hello().is_some()
    }

    pub fn supports_server_capability(&self, capability: &str) -> bool {
        routed_server_capability_supported(
            capability,
            self.supervisor.is_connected(),
            self.supervisor.supports_server_capability(capability),
            self.direct_client().is_some_and(|direct| direct.supports_server_capability(capability)),
        )
    }

    pub async fn wait_for_hello(&self, timeout_ms: u64) -> DaemonClientResult<DaemonHello> {
        self.supervisor.wait_for_hello_boxed(timeout_ms).await
    }

    pub async fn connect(self: &Arc<Self>, timeout_ms: u64) -> DaemonClientResult<()> {
        if !self.supervisor.is_connected() {
            self.supervisor.connect_boxed(timeout_ms).await?;
        }
        Ok(())
    }

    pub async fn reconnect(self: &Arc<Self>, timeout_ms: u64) -> DaemonClientResult<()> {
        if !self.supervisor.is_connected() {
            self.supervisor.reconnect_boxed(timeout_ms).await?;
        }
        Ok(())
    }

    pub async fn disconnect_for_reconnect(self: &Arc<Self>, reason: &str) {
        self.fallback_to_supervisor();
        self.supervisor.disconnect_for_reconnect_boxed(reason.to_string()).await;
    }

    pub async fn reset_transport_for_reconnect(self: &Arc<Self>) {
        self.supervisor.reset_transport_for_reconnect_boxed().await;
    }

    pub fn on_message(&self, listener: DaemonClientMessageListener) -> Box<dyn Fn() + Send + Sync> {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.message_listeners
            .lock()
            .expect("message listeners poisoned")
            .insert(id, listener);
        let registry = Arc::clone(&self.message_listeners);
        Box::new(move || {
            registry.lock().expect("message listeners poisoned").shift_remove(&id);
        })
    }

    pub fn on_close(&self, listener: DaemonClientCloseListener) -> Box<dyn Fn() + Send + Sync> {
        let id = self.next_listener_id.fetch_add(1, Ordering::SeqCst);
        self.close_listeners
            .lock()
            .expect("close listeners poisoned")
            .insert(id, listener);
        let registry = Arc::clone(&self.close_listeners);
        Box::new(move || {
            registry.lock().expect("close listeners poisoned").shift_remove(&id);
        })
    }

    pub fn enable_request_recovery(&self) {
        self.supervisor.enable_request_recovery();
    }

    pub async fn request(
        self: &Arc<Self>,
        command: DaemonCommandBody,
        timeout_ms: u64,
        options: DaemonClientRequestOptions,
    ) -> DaemonClientResult<DaemonResponse> {
        if command.get("type").and_then(Value::as_str) == Some("reattach") {
            // Reattach may land on a different worker; the direct link to the old
            // one is stale on success.
            let response = self.request_control_plane(command, timeout_ms, options).await?;
            if response.success {
                self.fallback_to_supervisor();
            }
            return Ok(response);
        }
        let direct = self.direct_client();
        if let Some(direct) = direct {
            if direct.is_connected() && self.serves_direct(&direct, &command) {
                return direct.request(command, timeout_ms, options).await;
            }
        }
        self.request_control_plane(command, timeout_ms, options).await
    }

    fn serves_direct(&self, direct: &DaemonWorkerClient, command: &DaemonCommandBody) -> bool {
        if !is_session_plane_daemon_command(command.get("type").and_then(Value::as_str).unwrap_or_default()) {
            return false;
        }
        let Some(hello) = direct.hello() else {
            return false;
        };
        command_compatibilities(command)
            .iter()
            .all(|compatibility| meets_daemon_command_compatibility(&compatibility_hello(&hello), compatibility))
    }

    async fn request_control_plane(
        self: &Arc<Self>,
        command: DaemonCommandBody,
        timeout_ms: u64,
        options: DaemonClientRequestOptions,
    ) -> DaemonClientResult<DaemonResponse> {
        match self
            .supervisor
            .request_boxed(command, Some(timeout_ms), options)
            .await
        {
            Ok(response) => Ok(response),
            Err(error) => {
                if error.is_capability_unavailable() {
                    return Err(error);
                }
                Err(DaemonClientError::Message(
                    DaemonControlPlaneTransportError::new(&error.message()).message,
                ))
            }
        }
    }

    pub fn fallback_to_supervisor(&self) {
        let direct = self
            .direct
            .lock()
            .expect("direct slot poisoned")
            .take();
        let Some(direct) = direct else {
            return;
        };
        if let Some(unsubscribe) = self
            .unsubscribe_direct_message
            .lock()
            .expect("direct message slot poisoned")
            .take()
        {
            unsubscribe();
        }
        if let Some(unsubscribe) = self
            .unsubscribe_direct_close
            .lock()
            .expect("direct close slot poisoned")
            .take()
        {
            unsubscribe();
        }
        direct.close_now();
    }

    pub fn close(&self) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        self.fallback_to_supervisor();
        if let Some(unsubscribe) = self
            .unsubscribe_supervisor_message
            .lock()
            .expect("supervisor message slot poisoned")
            .take()
        {
            unsubscribe();
        }
        if let Some(unsubscribe) = self
            .unsubscribe_supervisor_close
            .lock()
            .expect("supervisor close slot poisoned")
            .take()
        {
            unsubscribe();
        }
        self.supervisor.close();
        self.message_listeners
            .lock()
            .expect("message listeners poisoned")
            .clear();
        self.close_listeners
            .lock()
            .expect("close listeners poisoned")
            .clear();
    }

    fn direct_client(&self) -> Option<Arc<DaemonWorkerClient>> {
        self.direct.lock().expect("direct slot poisoned").clone()
    }

    fn bind_direct(self: &Arc<Self>, direct: &Arc<DaemonWorkerClient>) {
        let weak = Arc::downgrade(self);
        let unsubscribe_message = direct.on_message(Arc::new(move |message: &Value| {
            if let Some(client) = weak.upgrade() {
                client.emit_message(message);
            }
        }));
        *self
            .unsubscribe_direct_message
            .lock()
            .expect("direct message slot poisoned") = Some(unsubscribe_message);
        let weak = Arc::downgrade(self);
        let bound = Arc::clone(direct);
        let unsubscribe_close = direct.on_close(Arc::new(move |error: &DaemonClientError| {
            let Some(client) = weak.upgrade() else {
                return;
            };
            if client.closed.load(Ordering::SeqCst) {
                return;
            }
            {
                let mut slot = client.direct.lock().expect("direct slot poisoned");
                match slot.as_ref() {
                    Some(current) if Arc::ptr_eq(current, &bound) => {
                        *slot = None;
                    }
                    _ => return,
                }
            }
            if let Some(unsubscribe) = client
                .unsubscribe_direct_message
                .lock()
                .expect("direct message slot poisoned")
                .take()
            {
                unsubscribe();
            }
            client
                .unsubscribe_direct_close
                .lock()
                .expect("direct close slot poisoned")
                .take();
            client.emit_close(DaemonDirectTransportClosedError::new(error).to_client_error());
        }));
        *self
            .unsubscribe_direct_close
            .lock()
            .expect("direct close slot poisoned") = Some(unsubscribe_close);
    }

    fn emit_message(&self, message: &Value) {
        let listeners: Vec<DaemonClientMessageListener> = self
            .message_listeners
            .lock()
            .expect("message listeners poisoned")
            .values()
            .cloned()
            .collect();
        for listener in listeners {
            listener(message);
        }
    }

    fn emit_close(&self, error: DaemonClientError) {
        let listeners: Vec<DaemonClientCloseListener> = self
            .close_listeners
            .lock()
            .expect("close listeners poisoned")
            .values()
            .cloned()
            .collect();
        for listener in listeners {
            listener(&error);
        }
    }
}

fn routed_server_capability_supported(
    capability: &str,
    supervisor_connected: bool,
    supervisor_supported: bool,
    direct_supported: bool,
) -> bool {
    // Roster metadata comes from the control plane, never the direct worker.
    if capability == "streaming_throughput" || capability == "agent_communication_queue" {
        supervisor_connected && supervisor_supported
    } else {
        supervisor_supported || direct_supported
    }
}

impl DaemonTransportClient for DaemonRoutedClient {
    fn hello(&self) -> Option<DaemonHello> { DaemonRoutedClient::hello(self) }
    fn is_connected(&self) -> bool { DaemonRoutedClient::is_connected(self) }
    fn supports_server_capability(&self, capability: &str) -> bool {
        DaemonRoutedClient::supports_server_capability(self, capability)
    }
    fn on_message(&self, listener: DaemonClientMessageListener) -> Box<dyn Fn() + Send + Sync> {
        DaemonRoutedClient::on_message(self, listener)
    }
    fn on_close(&self, listener: DaemonClientCloseListener) -> Box<dyn Fn() + Send + Sync> {
        DaemonRoutedClient::on_close(self, listener)
    }
    fn enable_request_recovery(&self) { DaemonRoutedClient::enable_request_recovery(self); }
    fn request_boxed(&self, command: DaemonCommandBody, timeout_ms: Option<u64>, options: DaemonClientRequestOptions)
        -> futures::future::BoxFuture<'static, DaemonClientResult<DaemonResponse>> {
        let client = self.self_ref.upgrade();
        Box::pin(async move {
            let client = client.ok_or_else(|| DaemonClientError::Message("Daemon routed client is closed".to_string()))?;
            client.request(command, timeout_ms.unwrap_or(30_000), options).await
        })
    }
    fn wait_for_hello_boxed(&self, timeout_ms: u64)
        -> futures::future::BoxFuture<'static, DaemonClientResult<DaemonHello>> {
        self.supervisor.wait_for_hello_boxed(timeout_ms)
    }
    fn connect_boxed(&self, timeout_ms: u64) -> futures::future::BoxFuture<'static, DaemonClientResult<()>> {
        if self.supervisor.is_connected() { Box::pin(async { Ok(()) }) }
        else { self.supervisor.connect_boxed(timeout_ms) }
    }
    fn reconnect_boxed(&self, timeout_ms: u64) -> futures::future::BoxFuture<'static, DaemonClientResult<()>> {
        if self.supervisor.is_connected() { Box::pin(async { Ok(()) }) }
        else { self.supervisor.reconnect_boxed(timeout_ms) }
    }
    fn disconnect_for_reconnect_boxed(&self, reason: String) -> futures::future::BoxFuture<'static, ()> {
        self.fallback_to_supervisor();
        self.supervisor.disconnect_for_reconnect_boxed(reason)
    }
    fn reset_transport_for_reconnect_boxed(&self) -> futures::future::BoxFuture<'static, ()> {
        self.supervisor.reset_transport_for_reconnect_boxed()
    }
    fn is_routed_client(&self) -> bool { true }
    fn close(&self) { DaemonRoutedClient::close(self); }
}

/// Upgrade a supervisor connection with a direct worker link; every failure
/// returns the supervisor unchanged.
pub async fn create_daemon_session_transport(
    supervisor: Arc<dyn DaemonTransportClient>,
    active_session_id: &str,
    direct_disabled: bool,
) -> Arc<dyn DaemonTransportClient> {
    if direct_disabled
        || supervisor.is_routed_client()
        || !supervisor.supports_server_capability("direct_peer_transport")
    {
        return supervisor;
    }
    let mut direct: Option<Arc<DaemonWorkerClient>> = None;
    let result: DaemonClientResult<Option<Arc<DaemonWorkerClient>>> = async {
        // recoverable:false - this caller owns the fallback; a parked ticket
        // request would pend the attach forever.
        let command = serde_json::Map::from_iter([
            ("type".to_string(), Value::String("get_direct_worker_transport".to_string())),
            ("activeSessionId".to_string(), Value::String(active_session_id.to_string())),
        ]);
        let response = supervisor
            .request_boxed(
                command,
                Some(5000),
                DaemonClientRequestOptions {
                    recoverable: Some(false),
                    ..Default::default()
                },
            )
            .await?;
        if !response.success {
            return Ok(None);
        }
        let Some(ticket) = read_session_transport_ticket(response.data.as_ref().unwrap_or(&Value::Null)) else {
            return Ok(None);
        };
        if ticket.active_session_id != active_session_id {
            return Ok(None);
        }
        let expires_at = chrono::DateTime::parse_from_rfc3339(&ticket.expires_at)
            .map(|time| time.timestamp_millis() as f64)
            .unwrap_or(f64::NAN);
        if expires_at <= chrono::Utc::now().timestamp_millis() as f64 {
            return Ok(None);
        }
        let current_identity = get_daemon_socket_identity(&ticket.socket_path);
        let identity_matches = current_identity
            .map(|identity| identity.dev == ticket.socket_identity.dev && identity.ino == ticket.socket_identity.ino)
            .unwrap_or(false);
        if !identity_matches {
            return Ok(None);
        }
        let client = Arc::new(DaemonWorkerClient::new(&ticket.socket_path));
        direct = Some(Arc::clone(&client));
        client.connect(1000).await?;
        client.wait_for_hello(3000).await?;
        client.authenticate_peer(&ticket, 3000).await?;
        Ok(Some(client))
    }
    .await;
    match result {
        Ok(Some(client)) => DaemonRoutedClient::new(supervisor, client),
        Ok(None) => supervisor,
        Err(_) => {
            if let Some(direct) = direct {
                direct.close_now();
            }
            supervisor
        }
    }
}

/// `readSessionTransportTicket`.
pub fn read_session_transport_ticket(value: &Value) -> Option<DaemonPeerTransportTicket> {
    let candidate = value.as_object()?;
    if candidate.get("purpose").and_then(Value::as_str) != Some("session_client") {
        return None;
    }
    for key in [
        "socketPath",
        "workerInstanceId",
        "activeSessionId",
        "grantId",
        "token",
        "expiresAt",
    ] {
        if candidate.get(key).and_then(Value::as_str).is_none() {
            return None;
        }
    }
    let identity = candidate.get("socketIdentity")?.as_object()?;
    if identity.get("dev").and_then(Value::as_u64).is_none()
        || identity.get("ino").and_then(Value::as_u64).is_none()
    {
        return None;
    }
    serde_json::from_value(value.clone()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::daemon_protocol::{DaemonProtocolInfo, DAEMON_PROTOCOL_NAME, DAEMON_PROTOCOL_VERSION, DAEMON_SCHEMA_REVISION};
    use std::sync::atomic::AtomicUsize;

    fn command(type_: &str) -> DaemonCommandBody {
        serde_json::Map::from_iter([("type".to_string(), Value::String(type_.to_string()))])
    }

    struct FakeSupervisor {
        connected: AtomicBool,
        hello: StdMutex<Option<DaemonHello>>,
        capability: bool,
        requests: StdMutex<Vec<DaemonCommandBody>>,
        response: StdMutex<Option<DaemonResponse>>,
        closed: AtomicUsize,
    }

    impl FakeSupervisor {
        fn new(connected: bool, capability: bool) -> Arc<Self> {
            Arc::new(Self {
                connected: AtomicBool::new(connected),
                hello: StdMutex::new(Some(DaemonHello {
                    protocol: DaemonProtocolInfo {
                        name: DAEMON_PROTOCOL_NAME.to_string(),
                        version: DAEMON_PROTOCOL_VERSION,
                    },
                    schema_revision: Some(DAEMON_SCHEMA_REVISION),
                    server_capabilities: Vec::new(),
                    raw: Value::Null,
                })),
                capability,
                requests: StdMutex::new(Vec::new()),
                response: StdMutex::new(None),
                closed: AtomicUsize::new(0),
            })
        }
    }

    impl DaemonTransportClient for FakeSupervisor {
        fn hello(&self) -> Option<DaemonHello> {
            self.hello.lock().expect("hello poisoned").clone()
        }

        fn is_connected(&self) -> bool {
            self.connected.load(Ordering::SeqCst)
        }

        fn supports_server_capability(&self, capability: &str) -> bool {
            (self.capability && capability == "direct_peer_transport")
                || self.hello.lock().expect("hello poisoned").as_ref()
                    .is_some_and(|hello| hello.server_capabilities.iter().any(|value| value == capability))
        }

        fn on_message(&self, _listener: DaemonClientMessageListener) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }

        fn on_close(&self, _listener: DaemonClientCloseListener) -> Box<dyn Fn() + Send + Sync> {
            Box::new(|| {})
        }

        fn enable_request_recovery(&self) {}

        fn request_boxed(
            &self,
            command: DaemonCommandBody,
            _timeout_ms: Option<u64>,
            _options: DaemonClientRequestOptions,
        ) -> futures::future::BoxFuture<'static, DaemonClientResult<DaemonResponse>> {
            self.requests.lock().expect("requests poisoned").push(command);
            let response = self.response.lock().expect("response poisoned").clone();
            Box::pin(async move {
                response.ok_or_else(|| DaemonClientError::Message("no fake response".to_string()))
            })
        }

        fn close(&self) {
            self.closed.fetch_add(1, Ordering::SeqCst);
        }

        fn is_routed_client(&self) -> bool {
            false
        }
    }

    fn ticket_value() -> Value {
        serde_json::json!({
            "purpose": "session_client",
            "socketPath": "/tmp/worker.sock",
            "socketIdentity": { "dev": 1, "ino": 2 },
            "workerInstanceId": "worker-1",
            "activeSessionId": "active-1",
            "grantId": "grant-1",
            "token": "token-1",
            "expiresAt": "2999-01-01T00:00:00.000Z"
        })
    }

    #[test]
    fn the_ticket_reader_requires_every_field() {
        assert!(read_session_transport_ticket(&ticket_value()).is_some());
        let mut missing_purpose = ticket_value();
        missing_purpose["purpose"] = serde_json::json!("other");
        assert!(read_session_transport_ticket(&missing_purpose).is_none());
        let mut missing_identity = ticket_value();
        missing_identity["socketIdentity"] = serde_json::json!({ "dev": 1 });
        assert!(read_session_transport_ticket(&missing_identity).is_none());
        let mut missing_token = ticket_value();
        missing_token.as_object_mut().expect("object").remove("token");
        assert!(read_session_transport_ticket(&missing_token).is_none());
    }

    #[test]
    fn direct_transport_close_errors_use_the_direct_worker_socket() {
        let cause = DaemonClientError::SocketClosed(DaemonSocketClosedError::new(
            "/tmp/worker.sock",
            Some("shutdown"),
            Some("socket hangup"),
        ));
        let error = DaemonDirectTransportClosedError::new(&cause);
        assert_eq!(error.socket_path, "direct-worker");
        assert_eq!(error.daemon_closing_reason.as_deref(), Some("shutdown"));
        assert_eq!(error.cause.as_deref(), Some(cause.message().as_str()));
        assert!(error.message().contains("Socket: direct-worker"));
        let control = DaemonControlPlaneTransportError::new("boom");
        assert_eq!(
            control.message,
            "Daemon control-plane transport failed: boom"
        );
    }

    #[test]
    fn communication_queue_metadata_never_uses_the_direct_worker_capability() {
        // Queue metadata is roster metadata from the control plane, exactly
        // like streaming throughput: a newer direct worker cannot satisfy
        // an older supervisor, and a connected supervisor must advertise it.
        assert!(!routed_server_capability_supported("agent_communication_queue", true, false, true));
        assert!(!routed_server_capability_supported("agent_communication_queue", false, true, true));
        assert!(routed_server_capability_supported("agent_communication_queue", true, true, false));
    }

    #[test]
    fn streaming_throughput_never_uses_new_worker_capability_for_old_supervisor() {
        assert!(!routed_server_capability_supported("streaming_throughput", true, false, true));
        assert!(!routed_server_capability_supported("streaming_throughput", false, true, true));
        assert!(routed_server_capability_supported("streaming_throughput", true, true, false));
        assert!(routed_server_capability_supported("model_catalog", true, false, true));
    }

    #[tokio::test]
    async fn streaming_throughput_depends_on_connected_control_plane() {
        let supervisor = FakeSupervisor::new(true, true);
        let direct = Arc::new(DaemonWorkerClient::new("/definitely/missing/worker.sock"));
        let routed = DaemonRoutedClient::new(supervisor.clone(), direct);
        assert!(!routed.supports_server_capability("streaming_throughput"));
        supervisor.hello.lock().unwrap().as_mut().unwrap().server_capabilities = vec!["generation_throughput".into()];
        assert!(!routed.supports_server_capability("streaming_throughput"));
        supervisor.hello.lock().unwrap().as_mut().unwrap().server_capabilities = vec!["streaming_throughput".into()];
        assert!(routed.supports_server_capability("streaming_throughput"));
        supervisor.connected.store(false, Ordering::SeqCst);
        assert!(!routed.supports_server_capability("streaming_throughput"));
        assert!(routed.supports_server_capability("direct_peer_transport"), "unrelated capability routing is unchanged");
    }

    #[tokio::test]
    async fn session_plane_commands_go_to_the_supervisor_when_direct_is_down() {
        let supervisor = FakeSupervisor::new(true, true);
        let response = serde_json::from_value::<DaemonResponse>(serde_json::json!({
            "type": "response",
            "command": "prompt",
            "success": true
        }))
        .expect("response");
        *supervisor.response.lock().expect("response poisoned") = Some(response);
        let direct = Arc::new(DaemonWorkerClient::new("/definitely/missing/worker.sock"));
        let routed = DaemonRoutedClient::new(supervisor.clone(), direct);
        assert!(routed.is_connected());
        assert!(!routed.has_direct_transport());
        assert!(routed.is_control_plane_ready());
        let response = routed
            .request(command("prompt"), 1000, DaemonClientRequestOptions::default())
            .await
            .expect("ok");
        assert!(response.success);
        assert_eq!(
            supervisor.requests.lock().expect("requests poisoned").len(),
            1
        );
        routed.close();
        assert_eq!(supervisor.closed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn reattach_falls_back_to_the_supervisor_on_success() {
        let supervisor = FakeSupervisor::new(true, true);
        let response = serde_json::from_value::<DaemonResponse>(serde_json::json!({
            "type": "response",
            "command": "reattach",
            "success": true
        }))
        .expect("response");
        *supervisor.response.lock().expect("response poisoned") = Some(response);
        let direct = Arc::new(DaemonWorkerClient::new("/definitely/missing/worker.sock"));
        let routed = DaemonRoutedClient::new(supervisor.clone(), direct);
        assert!(routed.direct_client().is_some());
        routed
            .request(command("reattach"), 1000, DaemonClientRequestOptions::default())
            .await
            .expect("ok");
        assert!(routed.direct_client().is_none());
    }

    #[tokio::test]
    async fn control_plane_errors_are_wrapped() {
        let supervisor = FakeSupervisor::new(true, true);
        let direct = Arc::new(DaemonWorkerClient::new("/definitely/missing/worker.sock"));
        let routed = DaemonRoutedClient::new(supervisor.clone(), direct);
        let error = routed
            .request(command("prompt"), 1000, DaemonClientRequestOptions::default())
            .await
            .expect_err("fails");
        assert_eq!(error.message(), "Daemon control-plane transport failed: no fake response");
    }

    #[tokio::test]
    async fn direct_disabled_returns_the_supervisor_unchanged() {
        let supervisor = FakeSupervisor::new(true, true);
        let transport: Arc<dyn DaemonTransportClient> = supervisor.clone();
        let result = create_daemon_session_transport(Arc::clone(&transport), "active-1", true).await;
        assert!(Arc::ptr_eq(&result, &transport));
        let result = create_daemon_session_transport(Arc::clone(&transport), "active-1", false).await;
        // The fake supervisor reports the capability, so a ticket request is
        // attempted and its failure returns the supervisor unchanged.
        assert!(Arc::ptr_eq(&result, &transport));
        let without_capability = FakeSupervisor::new(true, false);
        let transport: Arc<dyn DaemonTransportClient> = without_capability;
        let result = create_daemon_session_transport(Arc::clone(&transport), "active-1", false).await;
        assert!(Arc::ptr_eq(&result, &transport));
    }
}
