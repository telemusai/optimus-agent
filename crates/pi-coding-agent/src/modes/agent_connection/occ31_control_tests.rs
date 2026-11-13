use super::*;

// Pure in-memory transport: these checks never open a socket, launch a worker,
// read a profile, invoke a credential command, or contact a provider.
struct OccModesTransport {
    supported: bool,
    response: Result<DaemonResponse, String>,
    requests: Mutex<Vec<(Value, Option<u64>, bool)>>,
    entered: Arc<tokio::sync::Semaphore>,
    release: Option<Arc<tokio::sync::Semaphore>>,
}

impl OccModesTransport {
    fn new(supported: bool, data: Value) -> Arc<Self> {
        Arc::new(Self {
            supported,
            response: Ok(DaemonResponse::ok(data)),
            requests: Mutex::new(Vec::new()),
            entered: Arc::new(tokio::sync::Semaphore::new(0)),
            release: None,
        })
    }
}

impl DaemonTransportClient for OccModesTransport {
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
        capability == "occ31_textmas" && self.supported
    }
    fn hello_socket_path(&self) -> Option<String> { None }
    fn is_connected(&self) -> bool { true }
    fn enable_request_recovery(&self) {}
    fn close(&self) {}
    fn connect(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { panic!("controls must not connect") }) }
    fn wait_for_hello(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { panic!("controls must not await a handshake") }) }
    fn reconnect(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { panic!("controls must not reconnect") }) }
    fn disconnect_for_reconnect(&self, _: &str) { panic!("controls must not disconnect") }
    fn reset_transport_for_reconnect(&self) { panic!("controls must not reset transport") }
    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn DaemonTransportClient> { self }
}


#[tokio::test]
async fn occ31_unsupported_connection_never_sends_or_claims_configuration() {
    let transport = OccModesTransport::new(false, Value::Null);
    let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
    assert!(!connection.supports_occ31_textmas());
    assert!(connection.get_occ31_preferences().await.is_err());
    assert!(connection.set_occ31_mode(crate::core::occ::textmas::TextMasMode::Sequential).await.is_err());
    assert!(connection.set_occ32_enabled(true).await.is_err());
    *connection.latest_snapshot.lock().unwrap() = Some(AgentConnectionSnapshot {
        state: AgentConnectionState { occ3_modes: Some(Default::default()), ..Default::default() },
        ..Default::default()
    });
    *connection.latest_snapshot_is_fresh.lock().unwrap() = true;
    assert!(connection.get_state().await.unwrap().occ3_modes.is_none());
    assert!(connection.get_initial_snapshot().await.unwrap().state.occ3_modes.is_none());
    assert!(transport.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn occ31_host_requests_are_typed_targeted_and_invalidate_cached_modes() {
    use crate::core::occ::textmas::TextMasMode;
    let transport = OccModesTransport::new(true, json!({"occ31Mode":"hierarchical","applied":true}));
    let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
    assert!(connection.supports_occ31_textmas());
    connection.get_occ31_preferences().await.unwrap();
    *connection.latest_snapshot_is_fresh.lock().unwrap() = true;
    connection.set_occ31_mode(TextMasMode::Hierarchical).await.unwrap();
    assert!(!*connection.latest_snapshot_is_fresh.lock().unwrap());
    *connection.latest_snapshot_is_fresh.lock().unwrap() = true;
    connection.set_occ32_enabled(false).await.unwrap();
    assert!(!*connection.latest_snapshot_is_fresh.lock().unwrap());
    let sent: Vec<Value> = transport.requests.lock().unwrap().iter().map(|entry| entry.0.clone()).collect();
    assert_eq!(sent, vec![
        json!({"type":"occ31_get_preferences","activeSessionId":"active-a"}),
        json!({"type":"occ31_set_mode","activeSessionId":"active-a","mode":"hierarchical"}),
        json!({"type":"occ32_set_enabled","activeSessionId":"active-a","enabled":false}),
    ]);
}

#[tokio::test]
async fn occ31_reply_is_discarded_when_the_attached_session_changes() {
    let mut transport = OccModesTransport::new(true, json!({"occ31Mode":"off"}));
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    Arc::get_mut(&mut transport).unwrap().release = Some(release.clone());
    let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
    let request = tokio::spawn(connection.get_occ31_preferences());
    tokio::time::timeout(Duration::from_secs(1), transport.entered.acquire()).await.unwrap().unwrap().forget();
    connection.set_active_session_id("active-b".into());
    release.add_permits(1);
    assert!(request.await.unwrap().unwrap_err().contains("Session changed"));
}

#[test]
fn occ31_optional_metadata_unknown_values_never_break_legacy_attachment() {
    use crate::core::occ::textmas::TextMasMode;
    use crate::modes::agent_connection::types::Occ3Modes;
    let mut value = serde_json::to_value(AgentConnectionState::default()).unwrap();
    assert!(serde_json::from_value::<AgentConnectionState>(value.clone()).unwrap().occ3_modes.is_none());
    for malformed in [json!(false), json!({"textmasMode":"future"}), json!({"textmasMode":"off","latentEnabled":"yes"})] {
        value["occ3Modes"] = malformed;
        assert!(serde_json::from_value::<AgentConnectionState>(value.clone()).unwrap().occ3_modes.is_none());
    }
    let modes = Occ3Modes { textmas_mode: TextMasMode::Sequential, textmas_active: false, latent_enabled:true, latent_active:false, latent_available:false };
    value["occ3Modes"] = serde_json::to_value(&modes).unwrap();
    assert_eq!(serde_json::from_value::<AgentConnectionState>(value.clone()).unwrap().occ3_modes, Some(modes));
    #[derive(serde::Deserialize)]
    struct OldState { #[serde(rename="sessionId")] session_id: String }
    let old: OldState = serde_json::from_value(value).unwrap();
    assert_eq!(old.session_id, "");
}


#[tokio::test]
async fn occ31_get_state_reconciles_stale_active_snapshot_with_daemon_idle() {
    use crate::core::occ::textmas::TextMasMode;
    use crate::modes::agent_connection::types::Occ3Modes;

    let cached_modes = Occ3Modes {
        textmas_mode: TextMasMode::Sequential,
        textmas_active: true,
        latent_enabled: true,
        latent_active: false,
        latent_available: true,
    };
    let daemon_modes = Occ3Modes { textmas_active: false, ..cached_modes.clone() };
    let cached_state = AgentConnectionState {
        active_session_id: Some("active-a".into()),
        session_id: "session-a".into(),
        session_file: Some("/tmp/session-a.jsonl".into()),
        session_name: Some("preserve me".into()),
        message_count: 1.0,
        occ3_modes: Some(cached_modes),
        ..Default::default()
    };
    let daemon_state = AgentConnectionState {
        occ3_modes: Some(daemon_modes.clone()),
        ..cached_state.clone()
    };
    let transport = OccModesTransport::new(true, serde_json::to_value(&daemon_state).unwrap());
    let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
    let message: AgentMessage = serde_json::from_value(json!({
        "role": "user", "content": "preserve transcript", "timestamp": 7
    })).unwrap();
    *connection.latest_snapshot.lock().unwrap() = Some(AgentConnectionSnapshot {
        state: cached_state,
        messages: vec![message.clone()],
        ..Default::default()
    });
    *connection.latest_snapshot_is_fresh.lock().unwrap() = true;

    let reconciled = connection.get_state().await.unwrap();
    assert_eq!(
        reconciled.occ3_modes.as_ref().map(|modes| modes.textmas_active),
        Some(false),
        "the authoritative daemon idle state must supersede the stale cached TextMAS activity",
    );
    assert_eq!(reconciled.session_id, "session-a");
    assert_eq!(reconciled.session_file.as_deref(), Some("/tmp/session-a.jsonl"));

    let cached_read = connection.get_state().await.unwrap();
    assert_eq!(cached_read.occ3_modes, Some(daemon_modes.clone()));
    let requests = transport.requests.lock().unwrap();
    let commands: Vec<Value> = requests.iter().map(|entry| entry.0.clone()).collect();
    assert_eq!(commands, vec![json!({"type":"get_connection_state","activeSessionId":"active-a"})]);
    drop(requests);

    let snapshot = connection.latest_snapshot.lock().unwrap();
    let snapshot = snapshot.as_ref().unwrap();
    assert_eq!(snapshot.state.occ3_modes, Some(daemon_modes));
    assert_eq!(snapshot.state.session_id, "session-a");
    assert_eq!(snapshot.state.session_file.as_deref(), Some("/tmp/session-a.jsonl"));
    assert_eq!(snapshot.messages, vec![message]);
}

#[tokio::test]
async fn occ31_get_state_does_not_force_authoritative_textmas_activity_false() {
    use crate::core::occ::textmas::TextMasMode;
    use crate::modes::agent_connection::types::Occ3Modes;

    let cached_modes = Occ3Modes {
        textmas_mode: TextMasMode::Sequential,
        textmas_active: true,
        latent_enabled: false,
        latent_active: false,
        latent_available: false,
    };
    let daemon_modes = Occ3Modes { textmas_mode: TextMasMode::Hierarchical, ..cached_modes.clone() };
    let cached_state = AgentConnectionState {
        session_id: "session-a".into(),
        occ3_modes: Some(cached_modes),
        ..Default::default()
    };
    let daemon_state = AgentConnectionState {
        occ3_modes: Some(daemon_modes.clone()),
        ..cached_state.clone()
    };
    let transport = OccModesTransport::new(true, serde_json::to_value(&daemon_state).unwrap());
    let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
    *connection.latest_snapshot.lock().unwrap() = Some(AgentConnectionSnapshot {
        state: cached_state,
        ..Default::default()
    });
    *connection.latest_snapshot_is_fresh.lock().unwrap() = true;

    let reconciled = connection.get_state().await.unwrap();
    assert_eq!(reconciled.occ3_modes, Some(daemon_modes.clone()));
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
    assert_eq!(
        connection.latest_snapshot.lock().unwrap().as_ref().unwrap().state.occ3_modes,
        Some(daemon_modes),
    );
}

#[tokio::test]
async fn occ31_get_state_preserves_fast_path_for_fresh_inactive_modes() {
    use crate::core::occ::textmas::TextMasMode;
    use crate::modes::agent_connection::types::Occ3Modes;

    let inactive = |textmas_mode| Occ3Modes {
        textmas_mode,
        textmas_active: false,
        latent_enabled: false,
        latent_active: false,
        latent_available: false,
    };
    for cached_modes in [
        None,
        Some(inactive(TextMasMode::Off)),
        Some(inactive(TextMasMode::Sequential)),
    ] {
        let transport = OccModesTransport::new(true, Value::Null);
        let connection = DaemonAgentConnection::new(transport.clone(), "active-a".into(), Default::default());
        *connection.latest_snapshot.lock().unwrap() = Some(AgentConnectionSnapshot {
            state: AgentConnectionState {
                occ3_modes: cached_modes.clone(),
                ..Default::default()
            },
            ..Default::default()
        });
        *connection.latest_snapshot_is_fresh.lock().unwrap() = true;

        assert_eq!(connection.get_state().await.unwrap().occ3_modes, cached_modes);
        assert!(transport.requests.lock().unwrap().is_empty());
    }
}
