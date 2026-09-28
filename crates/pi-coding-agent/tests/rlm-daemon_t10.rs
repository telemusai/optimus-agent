//! T10 lane `rlm-daemon`: heartbeat catalog through the real agent connection.
//!
//! Finding D-10: `DaemonAgentConnection::list_heartbeats` is a constant-error
//! stub whose comment claims `modes/daemon/heartbeat-catalog.ts` "is not ported
//! in this slice". The daemon side already serves `heartbeats_list`
//! (`daemon_mode.rs:5678-5682`, `:10899-10932`) and the TypeScript call site is
//! `listDaemonHeartbeats(this.client, this.options.ownedSession ? this.activeSessionId : undefined)`
//! (`daemon-agent-connection.ts:842-843`) over `heartbeat-catalog.ts:6-27`.
//!
//! These tests drive the REAL `DaemonAgentConnection::list_heartbeats` (the
//! `AgentConnection` trait method production calls) over a fake
//! `DaemonTransportClient` shaped like `tests/telegram_steering.rs`
//! `RecordingTransport`. No socket and no production daemon are involved.

use std::sync::{Arc, Mutex};

use pi_ai::types::BoxFuture;
use pi_coding_agent::modes::agent_connection::daemon_agent_connection::{
    DaemonAgentConnection, DaemonAgentConnectionOptions, DaemonOutbound, DaemonResponse,
    DaemonTransportClient,
};
use pi_coding_agent::modes::agent_connection::types::AgentConnection;
use serde_json::{json, Value};

/// What one `request` call must answer, scripted per test.
#[derive(Clone)]
enum Reply {
    Ok(Value),
    /// `success: false` with this error string.
    Failed(String),
    /// The transport itself rejects (socket/abort class failure).
    Transport(String),
}

struct FakeTransport {
    capabilities: Vec<String>,
    /// `hello` is present: `hello_socket_path()` returns Some.
    hello: bool,
    reply: Reply,
    requests: Mutex<Vec<Value>>,
    wait_for_hello_calls: Mutex<u64>,
}

impl FakeTransport {
    fn new(capabilities: &[&str], hello: bool, reply: Reply) -> Arc<Self> {
        Arc::new(Self {
            capabilities: capabilities.iter().map(|value| value.to_string()).collect(),
            hello,
            reply,
            requests: Mutex::new(Vec::new()),
            wait_for_hello_calls: Mutex::new(0),
        })
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }

    fn wait_for_hello_calls(&self) -> u64 {
        *self.wait_for_hello_calls.lock().unwrap()
    }
}

impl DaemonTransportClient for FakeTransport {
    fn request(&self, command: Value, _: Option<u64>) -> BoxFuture<Result<DaemonResponse, String>> {
        self.requests.lock().unwrap().push(command.clone());
        let reply = self.reply.clone();
        Box::pin(async move {
            match reply {
                Reply::Ok(data) => Ok(DaemonResponse::ok(data)),
                Reply::Failed(error) => Ok(DaemonResponse::failed(error)),
                Reply::Transport(error) => Err(error),
            }
        })
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

    fn supports_server_capability(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|entry| entry == capability)
    }

    fn hello_socket_path(&self) -> Option<String> {
        self.hello.then(|| "fake-socket".to_string())
    }

    fn is_connected(&self) -> bool {
        true
    }

    fn enable_request_recovery(&self) {}

    fn close(&self) {}

    fn connect(&self, _: u64) -> BoxFuture<Result<(), String>> {
        panic!("no socket connections in this test")
    }

    fn wait_for_hello(&self, _: u64) -> BoxFuture<Result<(), String>> {
        *self.wait_for_hello_calls.lock().unwrap() += 1;
        Box::pin(async { Ok(()) })
    }

    fn reconnect(&self, _: u64) -> BoxFuture<Result<(), String>> {
        panic!("no socket connections in this test")
    }

    fn disconnect_for_reconnect(&self, _: &str) {}

    fn reset_transport_for_reconnect(&self) {}

    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn DaemonTransportClient> {
        self
    }
}

fn connect(transport: Arc<FakeTransport>, owned_session: bool) -> DaemonAgentConnection {
    DaemonAgentConnection::new(
        transport as Arc<dyn DaemonTransportClient>,
        "active-rlm-daemon-t10".to_string(),
        DaemonAgentConnectionOptions {
            owned_session,
            ..DaemonAgentConnectionOptions::default()
        },
    )
}

/// The `data` payload of a `heartbeats_list` response. `DaemonResponse::ok(data)` takes
/// the response's INNER `data` (`daemon_client.rs:daemon_response_from_value` reads
/// `candidate.get("data")` off the envelope), exactly like
/// `tests/telegram_steering.rs`'s `RecordingTransport` passes one.
fn catalog_response(rows: Value) -> Value {
    json!({ "heartbeats": rows })
}

/// D-10 core: a daemon that supports `heartbeat_catalog` and answers
/// `heartbeats_list` must give the connection's callers real rows. Today the
/// method returns a constant `Err`, so every `/heartbeats` and panel refresh
/// (which swallow the error with `if let Ok(..)`) renders an empty list.
#[tokio::test]
async fn list_heartbeats_returns_the_catalog_rows() {
    let transport = FakeTransport::new(
        &["heartbeat_catalog"],
        true,
        Reply::Ok(catalog_response(json!([
            { "job": { "id": "h1", "sessionId": "s1" }, "sessionName": "one" },
            { "job": { "id": "h2", "sessionId": "s2" } }
        ]))),
    );
    let connection = connect(Arc::clone(&transport), true);

    let heartbeats = connection
        .list_heartbeats()
        .await
        .expect("list_heartbeats must return the catalog, not a hard error");

    assert_eq!(heartbeats.len(), 2, "both catalog rows must reach the caller");
    assert_eq!(
        heartbeats[0].job.get("id").and_then(Value::as_str),
        Some("h1")
    );
    assert_eq!(
        heartbeats[0].session_name.as_deref(),
        Some("one"),
        "the row's sessionName must survive the projection"
    );

    let requests = transport.requests();
    assert_eq!(requests.len(), 1, "exactly one heartbeat request");
    assert_eq!(requests[0].get("type").and_then(Value::as_str), Some("heartbeats_list"));
    assert_eq!(
        requests[0].get("activeSessionId").and_then(Value::as_str),
        Some("active-rlm-daemon-t10"),
        "an owned session scopes the catalog to its own active session id"
    );
}

/// `listDaemonHeartbeats(client, this.options.ownedSession ? this.activeSessionId : undefined)`
/// (`daemon-agent-connection.ts:843`): a shared control-plane connection sends no
/// `activeSessionId` at all.
#[tokio::test]
async fn list_heartbeats_omits_the_session_id_for_a_shared_connection() {
    let transport = FakeTransport::new(&["heartbeat_catalog"], true, Reply::Ok(catalog_response(json!([]))));
    let connection = connect(Arc::clone(&transport), false);

    connection.list_heartbeats().await.expect("empty catalog is Ok");
    let requests = transport.requests();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0].get("activeSessionId").is_none(),
        "a non-owned connection must not scope the request, got {:?}",
        requests[0]
    );
}

/// `if (!client.supportsServerCapability("heartbeat_catalog")) return []`
/// (`heartbeat-catalog.ts:13`). An older daemon is a clean empty list, not an
/// error the callers have to swallow.
#[tokio::test]
async fn list_heartbeats_is_empty_without_the_capability() {
    let transport = FakeTransport::new(&[], true, Reply::Failed("must not be called".into()));
    let connection = connect(Arc::clone(&transport), true);

    let heartbeats = connection
        .list_heartbeats()
        .await
        .expect("a missing capability degrades to an empty catalog");
    assert!(heartbeats.is_empty(), "no rows without the capability");
    assert!(
        transport.requests().is_empty(),
        "the capability gate must short-circuit before any request"
    );
}

/// `catch (error) { if (isUnknownDaemonCommandError(error, "heartbeats_list")) return []; throw error; }`
/// (`heartbeat-catalog.ts:21-25`): an unknown-command rejection is the older
/// daemon, everything else must still surface.
#[tokio::test]
async fn list_heartbeats_degrades_only_on_unknown_command() {
    let unknown = FakeTransport::new(
        &["heartbeat_catalog"],
        true,
        Reply::Failed("Unknown daemon command: heartbeats_list".to_string()),
    );
    let connection = connect(Arc::clone(&unknown), true);
    let heartbeats = connection
        .list_heartbeats()
        .await
        .expect("an unknown-command rejection degrades to an empty list");
    assert!(heartbeats.is_empty());

    let other = FakeTransport::new(
        &["heartbeat_catalog"],
        true,
        Reply::Failed("cron store is unavailable".to_string()),
    );
    let connection = connect(Arc::clone(&other), true);
    let error = connection
        .list_heartbeats()
        .await
        .expect_err("a real daemon failure must not be hidden as an empty list");
    assert!(
        error.contains("cron store is unavailable"),
        "the daemon error must survive, got {error:?}"
    );

    let transport_failure = FakeTransport::new(
        &["heartbeat_catalog"],
        true,
        Reply::Transport("Connection to the Optimus Agent daemon closed.".to_string()),
    );
    let connection = connect(Arc::clone(&transport_failure), true);
    assert!(
        connection.list_heartbeats().await.is_err(),
        "a transport failure is not an unknown command and must stay an error"
    );
}

/// The daemon really serves the command the connection must send
/// (`daemon-mode.ts` `heartbeats_list`; `daemon_mode.rs:5678-5682`).
#[test]
fn the_daemon_protocol_serves_heartbeats_list() {
    use pi_coding_agent::modes::daemon::daemon_mode::DAEMON_COMMAND_TYPES;
    assert!(
        DAEMON_COMMAND_TYPES.contains(&"heartbeats_list"),
        "the daemon command table must carry heartbeats_list"
    );
    assert!(DAEMON_COMMAND_TYPES.contains(&"heartbeat_manage"));
}
