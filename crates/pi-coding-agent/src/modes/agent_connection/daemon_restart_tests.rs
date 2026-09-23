use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct RestartTransport {
    requests: Arc<Mutex<Vec<Value>>>,
    reconnects: AtomicUsize,
    hang_list: bool,
}
impl DaemonTransportClient for RestartTransport {
    fn request(&self, command: Value, _: Option<u64>) -> BoxFuture<Result<DaemonResponse, String>> {
        self.requests.lock().unwrap().push(command.clone());
        let hang_list = self.hang_list;
        Box::pin(async move {
            match command["type"].as_str().unwrap() {
                "detach" => Ok(DaemonResponse::ok(json!({}))),
                "list" if hang_list => std::future::pending().await,
                "list" => Ok(DaemonResponse::ok(json!({"sessions":[
                    {"sessionId":"unrelated","sessionFile":"/elsewhere","activeSessionId":"wrong"},
                    {"sessionId":"saved","sessionFile":"/saved-session.jsonl","activeSessionId":"new-active"}
                ]}))),
                "attach" => {
                    assert_eq!(command["activeSessionId"], "new-active");
                    Ok(DaemonResponse::ok(
                        json!({"activeSessionId":"new-active", "snapshot":{
                            "summary":{"sessionId":"saved","sessionFile":"/saved-session.jsonl","activeSessionId":"new-active"},
                            "state":{"sessionId":"saved","activeSessionId":"new-active"},
                            "messages":[{"role":"user","content":"saved transcript","timestamp":1}],
                            "lastEventSequence":0,"lastEventCursor":{"generation":"new","sequence":0}
                        }}),
                    ))
                }
                other => panic!("recovery must only discover and attach, got {other}"),
            }
        })
    }
    fn request_with_recoverable(
        &self,
        command: Value,
        timeout: Option<u64>,
        recoverable: bool,
    ) -> BoxFuture<Result<DaemonResponse, String>> {
        if command["type"] != "detach" {
            assert!(!recoverable, "recovery owns retries; do not park requests");
        }
        self.request(command, timeout)
    }
    fn on_message(
        &self,
        _: Arc<dyn Fn(DaemonOutbound) + Send + Sync>,
    ) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }
    fn on_close(&self, _: Arc<dyn Fn(String) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }
    fn supports_server_capability(&self, _: &str) -> bool {
        false
    }
    fn hello_socket_path(&self) -> Option<String> {
        Some("same-fixture-socket".into())
    }
    fn is_connected(&self) -> bool {
        true
    }
    fn enable_request_recovery(&self) {}
    fn close(&self) {}
    fn connect(&self, _: u64) -> BoxFuture<Result<(), String>> {
        panic!("restart owns reconnect")
    }
    fn wait_for_hello(&self, _: u64) -> BoxFuture<Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
    fn reconnect(&self, _: u64) -> BoxFuture<Result<(), String>> {
        self.reconnects.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }
    fn disconnect_for_reconnect(&self, _: &str) {}
    fn reset_transport_for_reconnect(&self) {}
    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn DaemonTransportClient> {
        self
    }
}

fn fixture(
    hang_list: bool,
) -> (
    Arc<DaemonAgentConnection>,
    Arc<RestartTransport>,
    Arc<Mutex<Vec<AgentConnectionEvent>>>,
) {
    let transport = Arc::new(RestartTransport {
        requests: Default::default(),
        reconnects: AtomicUsize::new(0),
        hang_list,
    });
    let connection = Arc::new(DaemonAgentConnection::new(
        transport.clone(),
        "old-active".into(),
        DaemonAgentConnectionOptions {
            reconnect_timeout_ms: Some(75),
            ..Default::default()
        },
    ));
    *connection.attached_session_id.lock().unwrap() = Some("saved".into());
    *connection.attached_session_file.lock().unwrap() = Some("/saved-session.jsonl".into());
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    let _unsubscribe = connection.subscribe(Arc::new(move |event| {
        captured.lock().unwrap().push(event);
        Box::pin(async {})
    }));
    (connection, transport, events)
}

#[tokio::test]
async fn announced_shutdown_reattaches_the_saved_session_and_resyncs_once() {
    for close in ["socket", "shutdown", "killed"] {
        let (connection, transport, events) = fixture(false);
        connection
            .handle_daemon_message(DaemonOutbound::DaemonClosing {
                reason: DaemonClosingReason::Shutdown,
            })
            .await
            .unwrap();
        if close == "socket" {
            connection.handle_transport_close("Connection to the Prime Agent daemon closed. Reason: shutdown. Socket: same-fixture-socket".into()).await;
        } else {
            connection
                .handle_daemon_message(DaemonOutbound::SessionClosed {
                    active_session_id: "old-active".into(),
                    reason: close.into(),
                    meta: None,
                })
                .await
                .unwrap();
        }
        assert_eq!(connection.active_session_id(), "new-active", "{close}");
        assert!(!*connection.shutdown_restart_pending.lock().unwrap());
        assert!(!*connection.terminal_close_emitted.lock().unwrap());
        assert_eq!(transport.reconnects.load(Ordering::SeqCst), 1);
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentConnectionEvent::SessionResynced { .. }))
                .count(),
            1
        );
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentConnectionEvent::Closed { .. })));
        assert_eq!(
            connection
                .latest_snapshot
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .messages
                .len(),
            1
        );
    }
}

#[tokio::test]
async fn bare_session_stop_is_terminal_including_after_a_reconnect() {
    for recovered in [false, true] {
        let (connection, transport, events) = fixture(false);
        if recovered {
            *connection.shutdown_restart_pending.lock().unwrap() = true;
            connection.reconnect("restart".into()).await.unwrap();
        }
        connection
            .handle_daemon_message(DaemonOutbound::SessionClosed {
                active_session_id: connection.active_session_id(),
                reason: "shutdown".into(),
                meta: None,
            })
            .await
            .unwrap();
        assert!(*connection.terminal_close_emitted.lock().unwrap());
        assert_eq!(
            transport.reconnects.load(Ordering::SeqCst),
            usize::from(recovered)
        );
        assert_eq!(
            events
                .lock()
                .unwrap()
                .iter()
                .filter(|event| matches!(event, AgentConnectionEvent::Closed { .. }))
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn restart_deadline_bounds_even_stalled_discovery() {
    let (connection, _, events) = fixture(true);
    *connection.shutdown_restart_pending.lock().unwrap() = true;
    tokio::time::timeout(
        Duration::from_millis(500),
        connection.reconnect("restart".into()),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(*connection.terminal_close_emitted.lock().unwrap());
    assert!(events.lock().unwrap().iter().any(|event| matches!(event,
        AgentConnectionEvent::Closed { error: Some(error) } if error.contains("transcript remains saved"))));
}

#[tokio::test]
async fn disposal_cancels_restart_without_terminal_error() {
    let (connection, _, events) = fixture(true);
    *connection.shutdown_restart_pending.lock().unwrap() = true;
    let task = {
        let connection = connection.clone();
        tokio::spawn(async move { connection.reconnect("restart".into()).await })
    };
    while connection.reconnect_in_flight.lock().unwrap().is_none() {
        tokio::task::yield_now().await;
    }
    connection.dispose_inner().await;
    assert!(task.await.unwrap().is_err());
    assert!(!events
        .lock()
        .unwrap()
        .iter()
        .any(|event| matches!(event, AgentConnectionEvent::Closed { .. })));
}

#[tokio::test]
async fn update_recovery_takes_over_shutdown_poll_without_duplicate_connected_event() {
    let (connection, _, events) = fixture(true);
    *connection.shutdown_restart_pending.lock().unwrap() = true;
    let task = {
        let connection = connection.clone();
        tokio::spawn(async move { connection.reconnect_shutdown_owner("restart".into()).await })
    };
    tokio::task::yield_now().await;
    *connection.update_restart_pending.lock().unwrap() = true;
    task.await.unwrap().unwrap();
    assert!(!events.lock().unwrap().iter().any(|event| matches!(event,
        AgentConnectionEvent::ConnectionStatus { status, .. } if status == "connected")));
    assert!(!*connection.terminal_close_emitted.lock().unwrap());
}

#[tokio::test]
async fn update_announcement_recovers_killed_worker_close() {
    let (connection, _, events) = fixture(false);
    connection
        .handle_daemon_message(DaemonOutbound::DaemonClosing {
            reason: DaemonClosingReason::Update,
        })
        .await
        .unwrap();
    connection
        .handle_daemon_message(DaemonOutbound::SessionClosed {
            active_session_id: "old-active".into(),
            reason: "killed".into(),
            meta: None,
        })
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if events.lock().unwrap().iter().any(|event| {
                matches!(event,
                AgentConnectionEvent::ConnectionStatus { status, .. } if status == "connected")
            }) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(connection.active_session_id(), "new-active");
    assert!(!*connection.terminal_close_emitted.lock().unwrap());
}
