use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct InitialAttachTransport {
    attach_requests: AtomicUsize,
    direct: AtomicBool,
    error: Option<String>,
    fall_back_on_error: bool,
}

impl InitialAttachTransport {
    fn new(error: Option<&str>, direct: bool, fall_back_on_error: bool) -> Self {
        Self {
            attach_requests: AtomicUsize::new(0),
            direct: AtomicBool::new(direct),
            error: error.map(str::to_string),
            fall_back_on_error,
        }
    }
}

impl DaemonTransportClient for InitialAttachTransport {
    fn request(
        &self,
        _command: Value,
        _timeout_ms: Option<u64>,
    ) -> BoxFuture<Result<DaemonResponse, String>> {
        Box::pin(async { Ok(DaemonResponse::ok(json!({}))) })
    }

    fn request_with_recoverable(
        &self,
        command: Value,
        _timeout_ms: Option<u64>,
        recoverable: bool,
    ) -> BoxFuture<Result<DaemonResponse, String>> {
        let attempt = if command["type"] == "attach" {
            self.attach_requests.fetch_add(1, Ordering::SeqCst) + 1
        } else {
            self.attach_requests.load(Ordering::SeqCst)
        };
        let first_error = self.error.clone();
        let fall_back = self.fall_back_on_error;
        if attempt == 1 && fall_back {
            self.direct.store(false, Ordering::SeqCst);
        }
        Box::pin(async move {
            match command["type"].as_str().unwrap() {
                "attach" => {
                    assert_eq!(recoverable, attempt == 1);
                    if attempt == 1 {
                        if let Some(error) = first_error {
                            return Err(error);
                        }
                    }
                    Ok(DaemonResponse::ok(json!({
                        "activeSessionId": "active",
                        "snapshot": {
                            "summary": {"sessionId": "saved", "activeSessionId": "active"},
                            "state": {"sessionId": "saved", "activeSessionId": "active"},
                            "messages": [],
                            "lastEventSequence": 0
                        }
                    })))
                }
                "detach" => Ok(DaemonResponse::ok(json!({}))),
                other => panic!("unexpected request: {other}"),
            }
        })
    }

    fn on_message(
        &self,
        _listener: Arc<dyn Fn(DaemonOutbound) + Send + Sync>,
    ) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }

    fn on_close(
        &self,
        _listener: Arc<dyn Fn(String) + Send + Sync>,
    ) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }

    fn supports_server_capability(&self, _capability: &str) -> bool {
        false
    }
    fn hello_socket_path(&self) -> Option<String> {
        Some("test-socket".into())
    }
    fn is_connected(&self) -> bool {
        true
    }
    fn enable_request_recovery(&self) {}
    fn close(&self) {}
    fn connect(&self, _timeout_ms: u64) -> BoxFuture<Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
    fn wait_for_hello(&self, _timeout_ms: u64) -> BoxFuture<Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
    fn reconnect(&self, _timeout_ms: u64) -> BoxFuture<Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
    fn disconnect_for_reconnect(&self, _reason: &str) {}
    fn reset_transport_for_reconnect(&self) {}

    fn has_direct_transport(&self) -> bool {
        self.direct.load(Ordering::SeqCst)
    }

    fn is_control_plane_ready(&self) -> bool {
        true
    }

    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn DaemonTransportClient> {
        self
    }
}

fn socket_error() -> String {
    "Connection to the Optimus Agent daemon closed. Socket: direct-worker".into()
}

#[tokio::test]
async fn direct_transport_fallback_retries_only_the_eligible_socket_close() {
    let transport = Arc::new(InitialAttachTransport::new(
        Some(&socket_error()),
        true,
        true,
    ));
    let connection =
        DaemonAgentConnection::attach(transport.clone(), "active".into(), Default::default())
            .await
            .expect("supervisor retry must attach");
    assert_eq!(transport.attach_requests.load(Ordering::SeqCst), 2);
    assert_eq!(connection.active_session_id(), "active");
    connection.dispose_inner().await;
}

#[tokio::test]
async fn permanent_attach_response_is_not_retried() {
    let transport = Arc::new(InitialAttachTransport::new(
        Some("Unknown active session"),
        true,
        false,
    ));
    let error =
        DaemonAgentConnection::attach(transport.clone(), "active".into(), Default::default())
            .await
            .err()
            .unwrap();
    assert_eq!(error, "Unknown active session");
    assert_eq!(transport.attach_requests.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn bounded_buffer_failure_is_not_retried_even_after_transport_fallback() {
    let error = "Opening this busy conversation exceeded the bounded update buffer.";
    let transport = Arc::new(InitialAttachTransport::new(Some(error), true, true));
    let actual =
        DaemonAgentConnection::attach(transport.clone(), "active".into(), Default::default())
            .await
            .err()
            .unwrap();
    assert_eq!(actual, error);
    assert_eq!(transport.attach_requests.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn control_plane_socket_failure_has_no_direct_fallback_retry() {
    let transport = Arc::new(InitialAttachTransport::new(
        Some(&socket_error()),
        false,
        false,
    ));
    let actual =
        DaemonAgentConnection::attach(transport.clone(), "active".into(), Default::default())
            .await
            .err()
            .unwrap();
    assert_eq!(actual, socket_error());
    assert_eq!(transport.attach_requests.load(Ordering::SeqCst), 1);
}
