//! Cross-worker protocol tests with private in-process worker sockets.

use super::*;
use super::daemon_supervisor_parity_tests::{add_descriptor_only_worker, SupervisorFixture};
use crate::modes::daemon::daemon_worker_client::{encode_private_frame, PrivateFrameDecoder};

struct ScriptedWorker {
    worker: Arc<Worker>,
    client: Arc<DaemonWorkerClient>,
    commands: Arc<Mutex<Vec<Value>>>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for ScriptedWorker {
    fn drop(&mut self) {
        self.client.close_now();
        self.task.abort();
    }
}

async fn serve<S: AsyncRead + AsyncWrite + Unpin>(
    mut socket: S,
    summary: SessionSummary,
    commands: Arc<Mutex<Vec<Value>>>,
) {
    let mut decoder = PrivateFrameDecoder::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let count = socket.read(&mut buffer).await.unwrap();
        if count == 0 { return; }
        for frame in decoder.push(&buffer[..count]).unwrap() {
            let command: Value = serde_json::from_slice(&frame.payload).unwrap();
            commands.lock().unwrap().push(command.clone());
            let data = match command["type"].as_str() {
                Some("list") => json!({"sessions":[summary]}),
                Some("attach") => json!({"activeSessionId":summary.id,"snapshot":{"summary":summary,"state":{},"messages":[{"role":"user","content":"saved transcript","timestamp":1}],"lastEventSequence":2}}),
                Some("worker_deliver_message") => json!({
                    "id":format!("agentmsg_{}", summary.session_id), "source":"agent_message",
                    "message":command["message"], "from":command["sender"],
                    "target":{"activeSessionId":summary.active_session_id,"sessionId":summary.session_id},
                    "deliveryStatus":match command["message"].as_str() {
                        Some("queued") => "queued",
                        // An unrecognized status is not proof of delivery (D-04).
                        Some("unknown-status") => "in-progress",
                        _ => "delivered",
                    },
                }),
                _ => Value::Null,
            };
            let response = if command["message"] == "reject" {
                json!({"type":"response","id":command["id"],"command":command["type"],"success":false,"error":"Agent message was not accepted"})
            } else {
                json!({"type":"response","id":command["id"],"command":command["type"],"success":true,"data":data})
            };
            let frame = encode_private_frame(
                &json!({"kind":"outbound","outboundType":"response","requestId":command["id"],"payloadEncoding":"jsonl"}),
                response.to_string().as_bytes(),
            ).unwrap();
            socket.write_all(&frame).await.unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nine_supervisor_snapshot_negotiation_preserves_legacy_clients_and_workers() {
    let mut fixture = SupervisorFixture::new("nine-snapshot-negotiation").await;
    let worker = ScriptedWorker::new(&fixture, "snapshot-worker", 0, None).await;
    let legacy = fixture.send(json!({"type":"attach","id":"legacy","activeSessionId":"snapshot-worker","capabilities":["attach_snapshot","event_sequence"]})).await;
    assert_eq!(legacy["success"], true, "{legacy}");
    assert_eq!(legacy["data"]["snapshot"]["messages"][0]["content"], "saved transcript");
    assert!(legacy["data"].get("snapshotStream").is_none());

    let supervisor = fixture.supervisor.clone(); let public = fixture.client.clone();
    let line = serde_json::to_vec(&json!({"type":"attach","id":"modern","activeSessionId":"snapshot-worker","capabilities":["attach_snapshot","event_sequence","chunked_snapshot"]})).unwrap();
    let task = tokio::spawn(async move { supervisor.handle_line(public, line).await });
    let response = fixture.next_frame(Duration::from_secs(5)).await.unwrap();
    assert_eq!(response["id"], "modern"); assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["data"]["snapshot"]["messages"], json!([]));
    assert!(response["data"]["snapshotStream"]["id"].is_string());
    assert_eq!(fixture.next_frame(Duration::from_secs(5)).await.unwrap()["type"], "session_snapshot_begin");
    let chunk = fixture.next_frame(Duration::from_secs(5)).await.unwrap();
    assert_eq!(chunk["messages"][0]["content"], "saved transcript");
    assert_eq!(fixture.next_frame(Duration::from_secs(5)).await.unwrap()["type"], "session_snapshot_end");
    task.await.unwrap();
    // A pre-upgrade worker receives only its existing full-frame contract.
    let commands = worker.commands.lock().unwrap();
    let attaches: Vec<_> = commands.iter().filter(|command| command["type"] == "attach").collect();
    assert_eq!(attaches.len(), 2);
    assert!(attaches.iter().all(|command| command["capabilities"] == json!(["attach_snapshot","event_sequence"])));
}

impl ScriptedWorker {
    async fn new(fixture: &SupervisorFixture, name: &str, depth: i64, parent: Option<&str>) -> Self {
        let worker = add_descriptor_only_worker(fixture, name, name, &format!("token-{name}"), DAEMON_WORKER_LIFECYCLE_READY);
        let summary = SessionSummary {
            id: name.to_string(), active_session_id: Some(name.to_string()),
            session_id: format!("session-{name}"), session_name: Some(format!("Name {name}")),
            cwd: fixture.root.join("workspace").to_string_lossy().into_owned(),
            rlm_depth: Some(depth), parent_session_id: parent.map(str::to_string),
            runtime_kind: Some(if depth == 0 { "top-level" } else { "subagent" }.to_string()),
            lifecycle: "resident".to_string(), activity: "idle".to_string(),
            unfinished_action_count: Some(0), ..Default::default()
        };
        fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&summary.roster_view()), Some(&worker), None);
        let commands = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&commands);
        #[cfg(windows)]
        let (socket, task) = {
            let socket = format!(r"\\.\pipe\optimus-message-worker-{}", uuid::Uuid::new_v4());
            let server = tokio::net::windows::named_pipe::ServerOptions::new().first_pipe_instance(true).create(&socket).unwrap();
            let task = tokio::spawn(async move {
                server.connect().await.unwrap();
                serve(server, summary, captured).await;
            });
            (socket, task)
        };
        #[cfg(unix)]
        let (socket, task) = {
            let socket = fixture.root.join(format!("{name}.sock")).to_string_lossy().into_owned();
            let server = tokio::net::UnixListener::bind(&socket).unwrap();
            let task = tokio::spawn(async move {
                let (stream, _) = server.accept().await.unwrap();
                serve(stream, summary, captured).await;
            });
            (socket, task)
        };
        let client = Arc::new(DaemonWorkerClient::new(&socket));
        client.connect(1_000).await.unwrap();
        worker.descriptor.lock().unwrap().socket_path = socket;
        *worker.client.lock().unwrap() = Some(Arc::clone(&client));
        Self { worker, client, commands, task }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn messaging_safety_supervisor_peer_catalog_authenticates_and_filters() {
    let mut fixture = SupervisorFixture::new("messaging-safety-catalog").await;
    let _source = ScriptedWorker::new(&fixture, "source", 0, None).await;
    let _peer = ScriptedWorker::new(&fixture, "peer", 0, None).await;
    let private = ScriptedWorker::new(&fixture, "private", 0, None).await;
    private.worker.descriptor.lock().unwrap().owner_client_id = Some("private-client".to_string());
    let stopped = ScriptedWorker::new(&fixture, "stopped", 0, None).await;
    stopped.worker.descriptor.lock().unwrap().stop_requested_at = Some("now".to_string());
    let disconnected = ScriptedWorker::new(&fixture, "disconnected", 0, None).await;
    disconnected.client.close_now();
    let response = fixture.send(json!({"type":"list_agent_peers","id":"peers","workerToken":"token-source"})).await;
    assert_eq!(response["success"], true, "{response}");
    let peers = response["data"]["peers"].as_array().unwrap();
    assert_eq!(peers.len(), 1, "{peers:?}");
    assert_eq!(peers[0]["activeSessionId"], "peer");
    assert_eq!(peers[0]["sessionId"], "session-peer");
    for token in ["", "unknown-token"] {
        let response = fixture.send(json!({"type":"list_agent_peers","id":"invalid-peers","workerToken":token})).await;
        assert_eq!(response["success"], false, "{response}");
        assert_eq!(response["error"], "Worker authentication failed");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn messaging_safety_supervisor_uses_trusted_sender_and_truthful_receipts() {
    let mut fixture = SupervisorFixture::new("messaging-safety-delivery").await;
    let _source = ScriptedWorker::new(&fixture, "source", 0, None).await;
    let target = ScriptedWorker::new(&fixture, "target", 0, None).await;
    for message in ["delivered", "queued", "reject"] {
        let response = fixture.send(json!({
            "type":"send_message","id":message,"fromActiveSessionId":"source",
            "targetActiveSessionId":"target","message":message,"agentOrigin":true,
            "sender":{"activeSessionId":"spoof","sessionName":"spoof"},
        })).await;
        assert_eq!(response["id"], message);
        assert_eq!(response["command"], "send_message");
        if message == "reject" {
            assert_eq!(response["success"], false, "{response}");
            assert_eq!(response["error"], "Agent message was not accepted");
        } else {
            assert_eq!(response["success"], true, "{response}");
            assert_eq!(response["data"]["deliveryStatus"], message);
            assert_eq!(response["data"]["from"]["activeSessionId"], "source");
            assert_eq!(response["data"]["from"]["sessionName"], "Name source");
        }
    }
    let commands = target.commands.lock().unwrap();
    assert_eq!(commands.len(), 3);
    assert!(commands.iter().all(|command| command["type"] == "worker_deliver_message"));
    assert!(commands.iter().all(|command| command.get("fromActiveSessionId").is_none()));
}

/// Register a worker that is addressable but no longer running: its recorded process
/// identity no longer matches, so a forward fails before the message reaches the target.
async fn unreachable_worker(fixture: &SupervisorFixture, name: &str) -> Arc<Worker> {
    let worker = add_descriptor_only_worker(fixture, name, name, &format!("token-{name}"), DAEMON_WORKER_LIFECYCLE_READY);
    let summary = SessionSummary {
        id: name.to_string(), active_session_id: Some(name.to_string()),
        session_id: format!("session-{name}"), session_name: Some(format!("Name {name}")),
        cwd: fixture.root.join("workspace").to_string_lossy().into_owned(),
        rlm_depth: Some(0), runtime_kind: Some("top-level".to_string()),
        lifecycle: "resident".to_string(), activity: "idle".to_string(),
        unfinished_action_count: Some(0), ..Default::default()
    };
    fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&summary.roster_view()), Some(&worker), None);
    worker.descriptor.lock().unwrap().process_start_id = Some("exited-worker-process".to_string());
    worker
}

/// D-04: every attempted cross-worker forward leaves one durable, content-free delivery
/// record, telemetry never changes the delivery result, and nothing is replayed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn messaging_safety_supervisor_records_durable_delivery_outcomes() {
    use crate::modes::daemon::agent_message_delivery_journal::{
        AgentMessageDeliveryJournal, AgentMessageDeliveryOutcome,
    };
    // A fresh case root per run, so the durable journal starts empty and the record
    // count in this test is the count this test produced.
    let case = format!("messaging-safety-delivery-journal-{}", uuid::Uuid::new_v4());
    let mut fixture = SupervisorFixture::new(&case).await;
    let _source = ScriptedWorker::new(&fixture, "source", 0, None).await;
    let target = ScriptedWorker::new(&fixture, "target", 0, None).await;
    let _unreachable = unreachable_worker(&fixture, "unreachable").await;
    let journal_path = fixture
        .root
        .join("daemon-workers")
        .join(AGENT_MESSAGE_DELIVERY_JOURNAL_FILE)
        .to_string_lossy()
        .into_owned();

    for message in ["delivered", "queued", "unknown-status"] {
        let response = fixture.send(json!({
            "type":"send_message","id":message,"fromActiveSessionId":"source",
            "targetActiveSessionId":"target","message":message,"agentOrigin":true,
        })).await;
        // Telemetry does not change what the caller sees: the receipt decides the result.
        assert_eq!(response["success"], true, "{response}");
        assert_eq!(response["data"]["from"]["activeSessionId"], "source");
    }

    // A self-send is rejected without a forward.
    let rejected = fixture.send(json!({
        "type":"send_message","id":"delivery-two","fromActiveSessionId":"source",
        "targetActiveSessionId":"source","message":"never delivered","agentOrigin":true,
    })).await;
    assert_eq!(rejected["success"], false, "{rejected}");
    assert_eq!(rejected["error"], "Agent messaging cannot target the sending session");

    // An unreachable target fails before any delivery, and the send is never replayed.
    let before = target.commands.lock().unwrap().len();
    let lost = fixture.send(json!({
        "type":"send_message","id":"delivery-three","fromActiveSessionId":"source",
        "targetActiveSessionId":"unreachable","message":"never attempted","agentOrigin":true,
    })).await;
    assert_eq!(lost["success"], false, "{lost}");
    assert_eq!(
        target.commands.lock().unwrap().len(),
        before,
        "a failed forward must not be delivered or replayed"
    );

    let records = AgentMessageDeliveryJournal::read(&journal_path);
    let outcomes: Vec<AgentMessageDeliveryOutcome> = records.iter().map(|record| record.outcome).collect();
    assert!(outcomes.contains(&AgentMessageDeliveryOutcome::Delivered), "{records:?}");
    assert!(outcomes.contains(&AgentMessageDeliveryOutcome::Queued), "{records:?}");
    assert!(outcomes.contains(&AgentMessageDeliveryOutcome::Rejected), "{records:?}");
    // Exactly one record per attempted send: three accepted forwards, a rejected self-send
    // and a rejected unreachable forward.
    assert_eq!(records.len(), 5, "{records:?}");
    assert_eq!(
        records.iter().filter(|record| matches!(record.outcome,
            AgentMessageDeliveryOutcome::Delivered | AgentMessageDeliveryOutcome::Queued)).count(),
        2,
        "one record per accepted forward, and no duplicate record per send: {records:?}"
    );

    let rejected_record = records
        .iter()
        .find(|record| record.outcome == AgentMessageDeliveryOutcome::Rejected)
        .expect("a rejected forward leaves a durable rejected record");
    assert_eq!(rejected_record.reason_code.as_deref(), Some("self_target"));
    assert_eq!(rejected_record.target_active_session_id, "source");
    assert_eq!(rejected_record.source_active_session_id.as_deref(), Some("source"));

    // An unrecognized receipt status is recorded as uncertain, not as delivered.
    let uncertain: Vec<&_> = records
        .iter()
        .filter(|record| record.outcome == AgentMessageDeliveryOutcome::Uncertain)
        .collect();
    assert_eq!(uncertain.len(), 1, "{records:?}");
    assert_eq!(uncertain[0].reason_code.as_deref(), Some("invalid_receipt"));
    assert_eq!(uncertain[0].target_active_session_id, "target");

    let unreachable_record = records
        .iter()
        .find(|record| record.target_active_session_id == "unreachable")
        .expect("an unreachable target leaves a durable record");
    assert_eq!(unreachable_record.outcome, AgentMessageDeliveryOutcome::Rejected);
    assert_eq!(unreachable_record.reason_code.as_deref(), Some("worker_unavailable"));
    assert!(unreachable_record.message_id.is_none(), "no receipt means no message id");
    // The reachable target got one record per send it received, and no duplicates.
    assert_eq!(
        records.iter().filter(|record| record.target_active_session_id == "target").count(),
        3,
        "{records:?}"
    );

    // The journal is telemetry, so no record may carry message text or a session name.
    let raw = std::fs::read_to_string(&journal_path).unwrap();
    for secret in ["never delivered", "never attempted", "Name source", "Name target", "Name unreachable"] {
        assert!(!raw.contains(secret), "delivery telemetry must not persist {secret:?}");
    }
    // One bounded JSON line per record, and nothing else.
    assert_eq!(raw.lines().count(), records.len(), "{raw}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn messaging_safety_supervisor_rejects_self_unknown_and_unrelated() {
    let mut fixture = SupervisorFixture::new("messaging-safety-reach").await;
    let _source = ScriptedWorker::new(&fixture, "source", 0, None).await;
    let target = ScriptedWorker::new(&fixture, "other-child", 2, Some("unrelated-parent")).await;
    let missing_source = fixture.send(json!({"type":"send_message","id":"missing-source","targetActiveSessionId":"other-child","message":"no","agentOrigin":true})).await;
    assert_eq!(missing_source["success"], false, "{missing_source}");
    assert_eq!(missing_source["error"], "Agent messaging requires fromActiveSessionId");
    for (source, destination, expected) in [
        ("source", "source", "cannot target the sending session"),
        ("unknown", "source", "Unknown active session"),
        ("source", "unknown", "Unknown active session"),
        ("source", "other-child", "Agent reach is limited"),
    ] {
        let response = fixture.send(json!({"type":"send_message","id":"rejected","fromActiveSessionId":source,"targetActiveSessionId":destination,"message":"no","agentOrigin":true})).await;
        assert_eq!(response["success"], false, "{response}");
        assert!(response["error"].as_str().unwrap().contains(expected), "{response}");
    }
    assert!(target.commands.lock().unwrap().iter().all(|command| command["type"] != "worker_deliver_message"));
}
