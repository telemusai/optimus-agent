use super::*;

// Pure in-memory transport: these checks never open a socket, launch a worker,
// read a profile, invoke a credential command, or contact Jev/the provider.
struct StatusTransport {
    supported: bool,
    features_supported: bool,
    dynamic_supported: bool,
    response: Result<DaemonResponse, String>,
    requests: Mutex<Vec<(Value, Option<u64>, bool)>>,
    entered: Arc<tokio::sync::Semaphore>,
    release: Option<Arc<tokio::sync::Semaphore>>,
}

impl StatusTransport {
    fn new(supported: bool, data: Value) -> Arc<Self> {
        Arc::new(Self {
            supported,
            features_supported: false,
            dynamic_supported: false,
            response: Ok(DaemonResponse::ok(data)),
            requests: Mutex::new(Vec::new()),
            entered: Arc::new(tokio::sync::Semaphore::new(0)),
            release: None,
        })
    }
}

impl DaemonTransportClient for StatusTransport {
    fn request(&self, command: Value, timeout_ms: Option<u64>) -> BoxFuture<Result<DaemonResponse, String>> {
        self.request_with_recoverable(command, timeout_ms, true)
    }
    fn request_with_recoverable(&self, command: Value, timeout_ms: Option<u64>, recoverable: bool) -> BoxFuture<Result<DaemonResponse, String>> {
        self.requests.lock().unwrap().push((command, timeout_ms, recoverable));
        let response = self.response.clone();
        let entered = self.entered.clone();
        let release = self.release.clone();
        Box::pin(async move {
            entered.add_permits(1);
            if let Some(release) = release { release.acquire().await.unwrap().forget(); }
            response
        })
    }
    fn on_message(&self, _: Arc<dyn Fn(DaemonOutbound) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> { Box::new(|| {}) }
    fn on_close(&self, _: Arc<dyn Fn(String) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> { Box::new(|| {}) }
    fn supports_server_capability(&self, capability: &str) -> bool {
        match capability {
            "jev_control" => self.supported,
            "jev_features" => self.features_supported,
            "jev_dynamic" => self.dynamic_supported,
            _ => false,
        }
    }
    fn hello_socket_path(&self) -> Option<String> { None }
    fn is_connected(&self) -> bool { true }
    fn enable_request_recovery(&self) {}
    fn close(&self) {}
    fn connect(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { panic!("status must not connect") }) }
    fn wait_for_hello(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { panic!("status must not await a handshake") }) }
    fn reconnect(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { panic!("status must not reconnect") }) }
    fn disconnect_for_reconnect(&self, _: &str) { panic!("status must not disconnect") }
    fn reset_transport_for_reconnect(&self) { panic!("status must not reset transport") }
    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn DaemonTransportClient> { self }
}

#[tokio::test]
async fn jev_status_unsupported_connection_makes_no_request() {
    let transport = StatusTransport::new(false, json!({"pipeline": {"completed": 0}}));
    let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
    assert_eq!(connection.get_jev_status().await.unwrap(), None);
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn jev_status_preserves_observed_or_unknown_pipeline_without_snapshot_invalidation() {
    for pipeline in [Value::Null, json!({"completed": 3, "failed": 1}),
        json!({"usage": {"requests": 2, "input_tokens": 120, "output_tokens": 9, "in_flight": 1}})] {
        let transport = StatusTransport::new(true, json!({"pipeline": pipeline, "settings": {"mode": "comparison"}}));
        let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
        *connection.latest_snapshot_is_fresh.lock().unwrap() = true;
        assert_eq!(connection.get_jev_status().await.unwrap(), Some(json!({"pipeline": pipeline})));
        assert_eq!(*transport.requests.lock().unwrap(), vec![(
            json!({"type": "jev_get_status", "activeSessionId": "active-a"}), Some(2_000), false,
        )]);
        assert!(*connection.latest_snapshot_is_fresh.lock().unwrap());
    }
}

#[tokio::test]
async fn jev_status_rejects_malformed_envelopes_instead_of_inventing_zero_counts() {
    for data in [json!({}), json!({"pipeline": []}), json!({"pipeline": false})] {
        let transport = StatusTransport::new(true, data);
        let connection = DaemonAgentConnection::new(transport, "active-a".into(), Default::default());
        assert_eq!(connection.get_jev_status().await.unwrap_err(), "Daemon returned an invalid Jev status response");
    }
}

#[tokio::test]
async fn jev_status_propagates_worker_failure_without_fallback_or_replay() {
    let mut transport = StatusTransport::new(true, Value::Null);
    Arc::get_mut(&mut transport).unwrap().response = Ok(DaemonResponse::failed("Worker unavailable"));
    let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
    assert_eq!(connection.get_jev_status().await.unwrap_err(), "Worker unavailable");
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
    assert!(!transport.requests.lock().unwrap()[0].2);
}

#[tokio::test]
async fn jev_status_discards_reply_if_active_session_changed_during_request() {
    let mut transport = StatusTransport::new(true, json!({"pipeline": {"completed": 7}}));
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    Arc::get_mut(&mut transport).unwrap().release = Some(release.clone());
    let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
    let request = tokio::spawn(connection.get_jev_status());
    tokio::time::timeout(Duration::from_secs(1), transport.entered.acquire()).await.unwrap().unwrap().forget();
    connection.set_active_session_id("active-b".into());
    release.add_permits(1);
    let result = tokio::time::timeout(Duration::from_secs(1), request).await.unwrap().unwrap();
    assert_eq!(result.unwrap_err(), "Session changed while reading Jev status");
}

#[test]
fn jev_feature_settings_gate_uses_the_execution_host_capability_without_requests() {
    for (legacy_supported, features_supported) in [(false, false), (true, false), (true, true)] {
        let mut transport = StatusTransport::new(legacy_supported, Value::Null);
        Arc::get_mut(&mut transport).unwrap().features_supported = features_supported;
        let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
        assert_eq!(connection.supports_jev_features(), features_supported);
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}

#[test]
fn jev_dynamic_requires_its_own_execution_host_capability_without_requests() {
    for dynamic_supported in [false, true] {
        let mut transport = StatusTransport::new(true, Value::Null);
        let fixture = Arc::get_mut(&mut transport).unwrap();
        fixture.features_supported = true;
        fixture.dynamic_supported = dynamic_supported;
        let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
        assert_eq!(connection.supports_jev_dynamic(), dynamic_supported);
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}
