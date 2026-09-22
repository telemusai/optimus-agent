//! In-memory wire checks only. No daemon service, credentials, or provider calls.

use super::super::daemon_protocol::{
    daemon_command_plane, daemon_outbound_compatibility, DAEMON_DEFAULT_SERVER_CAPABILITIES,
    DAEMON_SCHEMA_REVISION,
};
use super::*;
use serde::Deserialize;
use serde_json::json;

fn hello(revision: u32, capabilities: &[&str]) -> Value {
    json!({
        "type": "daemon_hello",
        "protocol": {"name": DAEMON_PROTOCOL_NAME, "version": 7},
        "schemaRevision": revision,
        "serverCapabilities": capabilities,
    })
}

fn mode_command(mode: &str) -> DaemonCommandBody {
    json!({"type": "jev_set_session_mode", "activeSessionId": "session-a", "mode": mode})
        .as_object()
        .unwrap()
        .clone()
}

fn supported(frame: &Value, command: &DaemonCommandBody) -> bool {
    let peer = compatibility_hello(&DaemonHello::from_value(frame).unwrap());
    command_compatibilities(command)
        .iter()
        .all(|gate| meets_daemon_command_compatibility(&peer, gate))
}

#[test]
fn jev_new_client_old_daemon_keeps_legacy_commands_without_enabling_combined_mode() {
    let legacy = hello(29, &["jev_control"]);
    for mode in ["off", "on", "compare", "active", "comparison", "enable"] {
        assert!(
            supported(&legacy, &mode_command(mode)),
            "legacy mode {mode}"
        );
    }
    for mode in [
        "compare-active",
        "compare-and-active",
        "compare_and_active",
        "both",
        " BOTH ",
    ] {
        assert!(
            !supported(&legacy, &mode_command(mode)),
            "combined alias {mode}"
        );
    }
    for command in ["jev_get_settings", "jev_get_status"] {
        let body = json!({"type": command, "activeSessionId": "session-a"})
            .as_object()
            .unwrap()
            .clone();
        assert!(supported(&legacy, &body));
        assert!(!supported(&hello(29, &[]), &body));
        assert!(!is_daemon_mutating_command(command));
        assert_eq!(daemon_command_plane(command), Some("session"));
    }
    assert!(is_daemon_mutating_command("jev_set_session_mode"));
    assert_eq!(
        daemon_command_plane("jev_set_session_mode"),
        Some("session")
    );
}

#[test]
fn jev_expansion_requires_both_capabilities_and_schema_revision() {
    let command = mode_command("compare-active");
    assert_eq!(DAEMON_PROTOCOL_VERSION, 7);
    assert_eq!(DAEMON_SCHEMA_REVISION, 31);
    assert!(DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::JevControl));
    assert!(DAEMON_DEFAULT_SERVER_CAPABILITIES.contains(&DaemonServerCapability::JevFeatures));
    assert!(!supported(
        &hello(29, &["jev_control", "jev_features"]),
        &command
    ));
    assert!(!supported(&hello(30, &["jev_control"]), &command));
    assert!(!supported(&hello(30, &["jev_features"]), &command));
    assert!(supported(
        &hello(30, &["jev_control", "jev_features"]),
        &command
    ));
}

#[test]
fn jev_old_client_new_daemon_accepts_legacy_commands_and_additive_metadata() {
    let current = hello(
        31,
        &["jev_control", "jev_features", "future_optional_capability"],
    );
    for mode in ["off", "on", "compare", "active"] {
        assert!(supported(&current, &mode_command(mode)));
    }
    // The pre-expansion reader treats mode as a string and ignores additive fields.
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct LegacySettings {
        mode: String,
        active_mode: bool,
        applied: bool,
    }
    let frame = json!({
        "type": "response", "command": "jev_get_settings", "success": true,
        "data": {"mode": "compare-active", "activeMode": true, "applied": false,
            "compareMode": true, "features": {"tool_requirement": true}, "compactionEnabled": false}
    });
    let response = daemon_response_from_value(&frame);
    let legacy: LegacySettings = serde_json::from_value(response.data.unwrap()).unwrap();
    assert_eq!(legacy.mode, "compare-active");
    assert!(legacy.active_mode);
    assert!(!legacy.applied);
    for event in [
        "response",
        "extension_ui_request",
        "session_attached",
        "session_event",
    ] {
        assert_eq!(
            daemon_outbound_compatibility(event),
            DaemonCommandCompatibility::legacy()
        );
    }
    // Missing optional metadata from an older daemon is not a parsing/startup error.
    let old = json!({"type": "response", "command": "jev_get_status", "success": true,
        "data": {"mode": "compare", "pipeline": null}});
    assert!(daemon_response_from_value(&old).success);
}

#[cfg(unix)]
async fn connected_client(frame: Value) -> (Arc<DaemonClient>, tokio::net::UnixStream) {
    let (stream, peer) = tokio::net::UnixStream::pair().unwrap();
    let (writer, _reader) = Framed::new(stream, LinesCodec::new()).split();
    let client = DaemonClient::create("unused-in-memory-socket");
    client.state.lock().await.socket = Some(Arc::new(ClientSocket {
        writer: Mutex::new(Some(writer)),
        reader_task: StdMutex::new(None),
    }));
    client.quick_connected.store(true, Ordering::SeqCst);
    client.handle_line(&frame.to_string()).await;
    (client, peer)
}

#[cfg(unix)]
fn assert_no_wire_bytes(peer: &tokio::net::UnixStream) {
    let mut buffer = [0u8; 1];
    assert_eq!(
        peer.try_read(&mut buffer).unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[cfg(unix)]
#[tokio::test]
async fn jev_new_client_old_daemon_rejects_before_any_wire_write() {
    let (client, peer) = connected_client(hello(29, &["jev_control"])).await;
    let result = client
        .request(
            mode_command("compare-active"),
            Some(100),
            Default::default(),
        )
        .await;
    match result.unwrap_err() {
        DaemonClientError::CapabilityUnavailable(error) => {
            assert_eq!(error.capability.as_deref(), Some("jev_features"));
            assert!(!error.after_reconnect);
        }
        error => panic!("unexpected rejection: {error}"),
    }
    assert_no_wire_bytes(&peer);
    assert!(client.state.lock().await.pending.is_empty());
    client.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn jev_negotiated_combined_mode_reaches_the_wire_unchanged() {
    let (client, peer) = connected_client(hello(30, &["jev_control", "jev_features"])).await;
    let mut peer = Framed::new(peer, LinesCodec::new());
    let receive = async {
        let line = tokio::time::timeout(Duration::from_secs(1), peer.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let envelope: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(envelope["command"]["mode"], "compare-active");
        client.handle_line(&json!({
            "type": "response", "id": envelope["id"], "command": "jev_set_session_mode", "success": true,
            "data": {"mode": "compare-active", "applied": true}
        }).to_string()).await;
    };
    let (response, ()) = tokio::join!(
        client.request(
            mode_command("compare-active"),
            Some(1000),
            Default::default()
        ),
        receive
    );
    assert!(response.unwrap().success);
    client.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn jev_combined_mode_is_not_replayed_to_an_older_daemon() {
    let (client, peer) = connected_client(hello(30, &["jev_control", "jev_features"])).await;
    let result = Arc::new(StdMutex::new(None));
    client.state.lock().await.pending.insert(
        "pending-mode".into(),
        PendingDaemonRequest {
            command_type: "jev_set_session_mode".into(),
            timeout_ms: 1000,
            on_progress: None,
            wire_data: "must-not-replay\n".into(),
            awaiting_reconnect: true,
            acknowledge_result: true,
            recoverable: true,
            compatibilities: command_compatibilities(&mode_command("both")),
            deadline: Instant::now() + Duration::from_secs(1),
            result: result.clone(),
            wake: Arc::new(Notify::new()),
        },
    );
    client
        .handle_line(&hello(29, &["jev_control"]).to_string())
        .await;
    match result.lock().unwrap().take().unwrap().unwrap_err() {
        DaemonClientError::CapabilityUnavailable(error) => {
            assert_eq!(error.capability.as_deref(), Some("jev_features"));
            assert!(error.after_reconnect);
        }
        error => panic!("unexpected replay rejection: {error}"),
    }
    assert_no_wire_bytes(&peer);
    assert!(client.state.lock().await.pending.is_empty());
    client.close().await;
}

#[test]
fn jev_usage_metadata_is_optional_for_both_peer_generations() {
    let command = json!({"type":"jev_get_status", "activeSessionId":"session-a"})
        .as_object().unwrap().clone();
    for revision in [29, 30, 31] {
        assert!(supported(&hello(revision, &["jev_control"]), &command));
        assert!(!supported(&hello(revision, &[]), &command));
    }
    #[derive(Deserialize)]
    struct LegacyPipeline { success_count: u64 }
    let frame = json!({"type":"response", "command":"jev_get_status", "success":true,
        "data":{"pipeline":{"success_count":2, "usage":{"requests":2, "input_tokens":120}}}});
    let response = daemon_response_from_value(&frame);
    let legacy: LegacyPipeline = serde_json::from_value(response.data.unwrap()["pipeline"].clone()).unwrap();
    assert_eq!(legacy.success_count, 2);
    assert_eq!(daemon_outbound_compatibility("response"), DaemonCommandCompatibility::legacy());
}
