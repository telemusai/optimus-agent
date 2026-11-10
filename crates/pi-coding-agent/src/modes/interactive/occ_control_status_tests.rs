//! In-memory host regressions: optional config failures never mask a confirmed write.
use super::*;
use std::sync::{Mutex, atomic::{AtomicBool, Ordering}};
use serde_json::{json, Value};
use pi_ai::types::BoxFuture;
use crate::modes::agent_connection::daemon_agent_connection::{
    DaemonAgentConnection, DaemonOutbound, DaemonResponse, DaemonTransportClient,
};

struct OptionalFailureHost {
    supported: bool,
    invalid_echo: bool,
    occ2: AtomicBool,
    latent: AtomicBool,
    requests: Mutex<Vec<Value>>,
}

impl OptionalFailureHost {
    fn new(supported: bool, invalid_echo: bool) -> Arc<Self> {
        Arc::new(Self { supported, invalid_echo, occ2: AtomicBool::new(true), latent: AtomicBool::new(true), requests: Mutex::new(Vec::new()) })
    }
}

impl DaemonTransportClient for OptionalFailureHost {
    fn request(&self, command: Value, _: Option<u64>) -> BoxFuture<Result<DaemonResponse, String>> {
        self.requests.lock().unwrap().push(command.clone());
        let response = match command["type"].as_str().unwrap() {
            "occ_get_preferences" => DaemonResponse::ok(json!({"occ2Enabled":self.occ2.load(Ordering::SeqCst)})),
            "occ_set_preferences" => {
                let value = command["occ2Enabled"].as_bool().unwrap();
                self.occ2.store(value, Ordering::SeqCst);
                DaemonResponse::ok(json!({"occ2Enabled":if self.invalid_echo { !value } else { value },"applied":true}))
            }
            "occ32_set_enabled" => {
                let value = command["enabled"].as_bool().unwrap();
                self.latent.store(value, Ordering::SeqCst);
                DaemonResponse::ok(json!({"occ32Enabled":if self.invalid_echo { !value } else { value },"applied":true}))
            }
            // A corrupt occ31.json makes the optional snapshot metadata unavailable;
            // the authoritative own-lane getter/setter above still works normally.
            "get_connection_state" => DaemonResponse::ok(serde_json::to_value(wire::AgentConnectionState::default()).unwrap()),
            "occ31_get_preferences" => DaemonResponse::failed("OCC3.1 preference contains invalid JSON"),
            "get_messages" => DaemonResponse::ok(json!({"messages":[]})),
            _ => panic!("unexpected request"),
        };
        Box::pin(async move { Ok(response) })
    }
    fn on_message(&self, _: Arc<dyn Fn(DaemonOutbound) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> { Box::new(|| {}) }
    fn on_close(&self, _: Arc<dyn Fn(String) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> { Box::new(|| {}) }
    fn supports_server_capability(&self, capability: &str) -> bool { self.supported && matches!(capability, "occ31_textmas" | "occ_session_preferences") }
    fn hello_socket_path(&self) -> Option<String> { None }
    fn is_connected(&self) -> bool { true }
    fn enable_request_recovery(&self) {}
    fn close(&self) {}
    fn connect(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { panic!("no network") }) }
    fn wait_for_hello(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { panic!("no handshake") }) }
    fn reconnect(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { panic!("no reconnect") }) }
    fn disconnect_for_reconnect(&self, _: &str) { panic!("no disconnect") }
    fn reset_transport_for_reconnect(&self) { panic!("no reset") }
    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn DaemonTransportClient> { self }
}

#[tokio::test]
async fn confirmed_occ2_and_latent_writes_survive_unavailable_textmas_metadata() {
    for (args, saved) in [("2 off", "OCC-2 preference saved: OFF"), ("3.2 off", "OCC3.2 preference saved: OFF")] {
        let host = OptionalFailureHost::new(true, false);
        let connection: Arc<dyn wire::AgentConnection> = Arc::new(DaemonAgentConnection::new(host.clone(), "active-a".into(), Default::default()));
        let (send, _receive) = mpsc::channel();
        let output = run(&connection, &send, args).await.unwrap();
        let CommandOutput::TransientStatus(panel) = output else { panic!("successful mutation must remain acknowledged") };
        assert!(panel.contains(saved));
        assert!(panel.contains("OCC3.1: unavailable"));
        assert!(panel.contains("OCC3.2: unavailable"));
        assert!(!panel.contains("OCC3.2: OFF"), "unavailable observation is not OFF");
        if args == "2 off" { assert!(!host.occ2.load(Ordering::SeqCst)); assert!(host.latent.load(Ordering::SeqCst)); }
        else { assert!(!host.latent.load(Ordering::SeqCst)); assert!(host.occ2.load(Ordering::SeqCst)); }
    }
}

#[tokio::test]
async fn occ_status_preserves_available_lanes_and_own_echo_validation_stays_strict() {
    let host = OptionalFailureHost::new(true, false);
    let connection: Arc<dyn wire::AgentConnection> = Arc::new(DaemonAgentConnection::new(host, "active-a".into(), Default::default()));
    let (send, _receive) = mpsc::channel();
    let CommandOutput::Panel(panel) = run(&connection, &send, "status").await.unwrap() else { panic!("partial status panel expected") };
    assert!(panel.contains("OCC-2: ON"));
    assert!(panel.contains("OCC3.1: unavailable"));
    assert!(panel.contains("OCC3.2: unavailable"));
    for args in ["2 off", "3.2 off"] {
        let host = OptionalFailureHost::new(true, true);
        let connection: Arc<dyn wire::AgentConnection> = Arc::new(DaemonAgentConnection::new(host, "active-a".into(), Default::default()));
        let result = run(&connection, &send, args).await;
        assert!(matches!(result, Err(ref error) if error.contains("did not confirm")));
    }
}

#[tokio::test]
async fn unsupported_textmas_commands_do_not_probe_or_write_any_host() {
    for args in ["3", "3.1", "3 off", "3 sequential", "3.1 hierarchical"] {
        let host = OptionalFailureHost::new(false, false);
        let connection: Arc<dyn wire::AgentConnection> = Arc::new(DaemonAgentConnection::new(host.clone(), "active-a".into(), Default::default()));
        let (send, _receive) = mpsc::channel();
        assert!(matches!(run(&connection, &send, args).await.unwrap(), CommandOutput::Warning(_)));
        assert!(host.requests.lock().unwrap().is_empty());
    }
}
