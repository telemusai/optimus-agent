use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct DisposeTransport {
    requests: Mutex<Vec<Value>>,
    failures: HashMap<String, String>,
    close_calls: AtomicUsize,
}

impl DisposeTransport {
    fn new(failures: &[(&str, &str)]) -> Self {
        Self {
            failures: failures
                .iter()
                .map(|(command, error)| ((*command).to_string(), (*error).to_string()))
                .collect(),
            ..Default::default()
        }
    }

    fn command_types(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|command| command["type"].as_str().unwrap().to_string())
            .collect()
    }
}

impl DaemonTransportClient for DisposeTransport {
    fn request(&self, command: Value, _: Option<u64>) -> BoxFuture<Result<DaemonResponse, String>> {
        self.requests.lock().unwrap().push(command.clone());
        let command_type = command["type"].as_str().unwrap().to_string();
        let failure = self.failures.get(&command_type).cloned();
        Box::pin(async move {
            if let Some(error) = failure {
                return Err(error);
            }
            match command_type.as_str() {
                "abort_side_question" => Ok(DaemonResponse::ok(json!({"aborted": true}))),
                "detach" | "complete_owned_session" => Ok(DaemonResponse::ok(json!({}))),
                other => panic!("unexpected command during disposal: {other}"),
            }
        })
    }

    fn on_message(&self, _: Arc<dyn Fn(DaemonOutbound) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }

    fn on_close(&self, _: Arc<dyn Fn(String) + Send + Sync>) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {})
    }

    fn supports_server_capability(&self, _: &str) -> bool { false }
    fn hello_socket_path(&self) -> Option<String> { None }
    fn is_connected(&self) -> bool { true }
    fn enable_request_recovery(&self) {}
    fn close(&self) { self.close_calls.fetch_add(1, Ordering::SeqCst); }
    fn connect(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { Ok(()) }) }
    fn wait_for_hello(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { Ok(()) }) }
    fn reconnect(&self, _: u64) -> BoxFuture<Result<(), String>> { Box::pin(async { Ok(()) }) }
    fn disconnect_for_reconnect(&self, _: &str) {}
    fn reset_transport_for_reconnect(&self) {}
    fn control_plane_transport(self: Arc<Self>) -> Arc<dyn DaemonTransportClient> { self }
}

struct Fixture {
    connection: Arc<DaemonAgentConnection>,
    transport: Arc<DisposeTransport>,
    message_unsubscribes: Arc<AtomicUsize>,
    close_unsubscribes: Arc<AtomicUsize>,
}

fn fixture(owned_session: bool, failures: &[(&str, &str)]) -> Fixture {
    let transport = Arc::new(DisposeTransport::new(failures));
    let connection = Arc::new(DaemonAgentConnection::new(
        transport.clone(),
        "active".into(),
        DaemonAgentConnectionOptions {
            close_client_on_dispose: true,
            owned_session,
            ..Default::default()
        },
    ));
    let message_unsubscribes = Arc::new(AtomicUsize::new(0));
    let close_unsubscribes = Arc::new(AtomicUsize::new(0));
    let messages = message_unsubscribes.clone();
    *connection.unsubscribe_daemon_messages.lock().unwrap() = Some(Box::new(move || {
        messages.fetch_add(1, Ordering::SeqCst);
    }));
    let closes = close_unsubscribes.clone();
    *connection.unsubscribe_daemon_close.lock().unwrap() = Some(Box::new(move || {
        closes.fetch_add(1, Ordering::SeqCst);
    }));
    Fixture { connection, transport, message_unsubscribes, close_unsubscribes }
}

fn assert_remaining_cleanup_completed(fixture: &Fixture) {
    assert!(*fixture.connection.disposed.lock().unwrap());
    assert_eq!(fixture.message_unsubscribes.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.close_unsubscribes.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.transport.close_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn non_owned_dispose_detaches_once_and_second_dispose_is_a_noop() {
    let fixture = fixture(false, &[]);
    fixture.connection.dispose().await.unwrap();
    assert_eq!(fixture.transport.command_types(), vec!["detach"]);
    assert_remaining_cleanup_completed(&fixture);

    fixture.connection.dispose().await.unwrap();
    assert_eq!(fixture.transport.command_types(), vec!["detach"]);
    assert_eq!(fixture.transport.close_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dispose_surfaces_detach_failure_without_replaying_or_completing_owned_session() {
    let fixture = fixture(false, &[("detach", "synthetic detach failure")]);
    let error = fixture.connection.dispose().await.unwrap_err();
    assert_eq!(error, "Daemon session view detach failed.");
    assert!(!error.contains("synthetic detach failure"));
    assert_eq!(fixture.transport.command_types(), vec!["detach"]);
    assert_remaining_cleanup_completed(&fixture);

    fixture.connection.dispose().await.unwrap();
    assert_eq!(fixture.transport.command_types(), vec!["detach"]);
}

#[tokio::test]
async fn owned_dispose_completes_owned_session_without_detaching() {
    let fixture = fixture(true, &[]);
    fixture.connection.dispose().await.unwrap();
    assert_eq!(fixture.transport.command_types(), vec!["complete_owned_session"]);
    assert_remaining_cleanup_completed(&fixture);

    fixture.connection.dispose().await.unwrap();
    assert_eq!(fixture.transport.command_types(), vec!["complete_owned_session"]);
}

#[tokio::test]
async fn dispose_preserves_first_failure_and_finishes_later_cleanup() {
    let fixture = fixture(
        false,
        &[
            ("abort_side_question", "synthetic first cleanup failure"),
            ("detach", "synthetic later detach failure"),
        ],
    );
    fixture.connection.active_side_question_ids.lock().unwrap().insert("side-question".into());
    let error = fixture.connection.dispose().await.unwrap_err();
    assert_eq!(
        error,
        "Daemon session view side-question cleanup failed."
    );
    assert!(!error.contains("synthetic first cleanup failure"));
    assert!(!error.contains("synthetic later detach failure"));
    assert_eq!(fixture.transport.command_types(), vec!["abort_side_question", "detach"]);
    assert_remaining_cleanup_completed(&fixture);
}

#[tokio::test]
async fn owned_dispose_completion_failure_is_operation_only_and_does_not_detach() {
    let fixture = fixture(true, &[("complete_owned_session", "synthetic owned completion secret")]);
    let error = fixture.connection.dispose().await.unwrap_err();
    assert_eq!(error, "Daemon owned session completion failed.");
    assert!(!error.contains("synthetic owned completion secret"));
    assert_eq!(fixture.transport.command_types(), vec!["complete_owned_session"]);
    assert_remaining_cleanup_completed(&fixture);
    fixture.connection.dispose().await.unwrap();
    assert_eq!(fixture.transport.command_types(), vec!["complete_owned_session"]);
}

#[tokio::test]
async fn already_disposed_is_a_successful_noop_without_a_lifecycle_command() {
    let fixture = fixture(false, &[("detach", "must not be requested")]);
    *fixture.connection.disposed.lock().unwrap() = true;
    fixture.connection.dispose().await.unwrap();
    assert!(fixture.transport.command_types().is_empty());
    assert_eq!(fixture.transport.close_calls.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.message_unsubscribes.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.close_unsubscribes.load(Ordering::SeqCst), 0);
}
