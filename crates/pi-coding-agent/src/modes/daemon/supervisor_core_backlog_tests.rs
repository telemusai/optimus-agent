//! Offline regressions for CORE-003 and CORE-005. No real worker is launched.
use super::*;
use super::daemon_supervisor_parity_tests::{add_descriptor_only_worker, SupervisorFixture};
use crate::modes::daemon::daemon_worker_client::{encode_private_frame, PrivateFrameDecoder};
use crate::modes::daemon::worker_recovery_journal::{WorkerRecoveryJournal, WorkerRecoveryRecordInput};

fn stopped_worker(fixture: &SupervisorFixture, id: &str) -> Arc<Worker> {
    let worker = add_descriptor_only_worker(fixture, id, id, "isolated-token", DAEMON_WORKER_LIFECYCLE_STOPPING);
    {
        let mut descriptor = worker.descriptor.lock().unwrap();
        descriptor.pid = i32::MAX;
        descriptor.process_start_id = Some("missing-process".into());
        descriptor.stop_requested_at = Some("2026-09-17T00:00:00Z".into());
        descriptor.archive_on_stop = Some(false);
        fixture.supervisor.persist_worker(&descriptor).unwrap();
    }
    worker
}

#[tokio::test]
async fn core_supervisor_unrelated_opens_progress_and_same_session_create_retry_join() {
    let fixture = SupervisorFixture::new("core-independent-opens").await;
    let worker = stopped_worker(&fixture, "same-chat");
    let path = fixture.root.join("saved.jsonl").to_string_lossy().into_owned();
    {
        let mut descriptor = worker.descriptor.lock().unwrap();
        descriptor.session_file = Some(path.clone());
        descriptor.stop_requested_at = None;
    }
    let key = canonical_session_path(&path);
    let entered = CancellationToken::new();
    let released = CancellationToken::new();
    let opener = {
        let supervisor = fixture.supervisor.clone();
        let entered = entered.clone(); let released = released.clone();
        tokio::spawn(async move {
            supervisor.with_worker_open(key, async {
                entered.cancel(); released.cancelled().await;
                Ok(json!({"activeSessionId":"shared-open"}))
            }).await
        })
    };
    entered.cancelled().await;
    let create_body = json!({"type":"create", "sessionPath":path}).as_object().unwrap().clone();
    let mut joined_create = Box::pin(fixture.supervisor.create_for_owner("owner".into(), &create_body));
    let mut joined_retry = Box::pin(fixture.supervisor.retry_worker("owner".into(), "same-chat"));
    assert!(tokio::time::timeout(Duration::from_millis(20), &mut joined_create).await.is_err());
    assert!(tokio::time::timeout(Duration::from_millis(20), &mut joined_retry).await.is_err());

    // This real create reaches config validation, not process launch. It must not
    // wait behind the unrelated worker's deliberately stalled create operation.
    let unrelated = json!({"type":"create", "config":"invalid-config"}).as_object().unwrap().clone();
    let error = tokio::time::timeout(Duration::from_secs(1), fixture.supervisor.create_for_owner("owner".into(), &unrelated)).await.expect("unrelated open blocked").unwrap_err();
    assert!(error.contains("invalid type"), "{error}");
    assert!(fixture.supervisor.opening.try_write().is_err(), "eviction must still exclude an open");
    released.cancel();
    let expected = opener.await.unwrap().unwrap();
    assert_eq!(joined_create.await.unwrap(), expected);
    assert_eq!(joined_retry.await.unwrap(), expected);
    assert!(fixture.supervisor.opening_workers.lock().unwrap().is_empty());
    assert!(fixture.supervisor.opening.try_write().is_ok());
    fixture.supervisor.stopped.cancel();
}

#[tokio::test]
async fn core_supervisor_cancelled_open_releases_key_and_joiners() {
    let fixture = SupervisorFixture::new("core-cancelled-open").await;
    let entered = CancellationToken::new();
    let opener = {
        let supervisor = fixture.supervisor.clone(); let entered = entered.clone();
        tokio::spawn(async move {
            supervisor.with_worker_open("same".into(), async {
                entered.cancel(); std::future::pending::<Result<Value, String>>().await
            }).await
        })
    };
    entered.cancelled().await;
    let mut joiner = Box::pin(fixture.supervisor.with_worker_open("same".into(), async { panic!("joiner must not launch") }));
    assert!(tokio::time::timeout(Duration::from_millis(20), &mut joiner).await.is_err());
    opener.abort(); assert!(opener.await.unwrap_err().is_cancelled());
    assert!(tokio::time::timeout(Duration::from_secs(1), joiner).await.unwrap().unwrap_err().contains("interrupted"));
    assert!(fixture.supervisor.opening_workers.lock().unwrap().is_empty());
    assert_eq!(fixture.supervisor.with_worker_open("same".into(), async { Ok(json!("retry")) }).await.unwrap(), json!("retry"));
    fixture.supervisor.stopped.cancel();
}

#[tokio::test]
async fn core_supervisor_late_create_never_revives_stopped_or_replaced_generation() {
    let fixture = SupervisorFixture::new("core-late-create").await;
    let worker = stopped_worker(&fixture, "generation");
    let expected = worker.descriptor.lock().unwrap().clone();
    let descriptor_file = fixture.supervisor.descriptor_dir.join("generation.json");
    let stopped_bytes = std::fs::read(&descriptor_file).unwrap();
    assert!(fixture.supervisor.mark_created_worker_ready(&worker, &expected, &json!({"sessionId":"late"})).is_err());
    fixture.supervisor.fail_created_worker(&worker, &expected, "late reply");
    assert_eq!(worker.descriptor.lock().unwrap().lifecycle, DAEMON_WORKER_LIFECYCLE_STOPPING);
    assert_eq!(std::fs::read(&descriptor_file).unwrap(), stopped_bytes);
    let replacement = stopped_worker(&fixture, "generation");
    let replacement_bytes = std::fs::read(&descriptor_file).unwrap();
    assert!(fixture.supervisor.mark_created_worker_ready(&worker, &expected, &json!({"sessionId":"late"})).is_err());
    fixture.supervisor.fail_created_worker(&worker, &expected, "retired open");
    assert_eq!(replacement.descriptor.lock().unwrap().lifecycle, DAEMON_WORKER_LIFECYCLE_STOPPING);
    assert_eq!(std::fs::read(&descriptor_file).unwrap(), replacement_bytes);
    assert!(Arc::ptr_eq(fixture.supervisor.workers.lock().unwrap().get("generation").unwrap(), &replacement));
    fixture.supervisor.stopped.cancel();
}

#[tokio::test]
async fn core_supervisor_permanent_stop_failure_parks_and_survives_adoption() {
    let fixture = SupervisorFixture::new("core-permanent-cleanup").await;
    let worker = stopped_worker(&fixture, "broken");
    let journal = worker.descriptor.lock().unwrap().recovery_journal_path.clone();
    WorkerRecoveryJournal::new(&journal).record(WorkerRecoveryRecordInput {
        active_session_id: "missing-child".into(), session_id: "missing-child".into(),
        session_file: None, busy: true, operation: "tool_call".into(),
    }).unwrap();
    let before = std::fs::read(&journal).unwrap();
    let descriptor = worker.descriptor.lock().unwrap().clone();
    stop_finalizations().lock().unwrap().insert(descriptor.worker_id.clone());
    tokio::time::timeout(Duration::from_secs(2), finalize_timed_out_worker_stop(fixture.supervisor.clone(), worker.clone(), descriptor)).await.expect("permanent cleanup spun");
    assert!(stop_cleanup_is_parked(&worker.descriptor.lock().unwrap()));
    assert!(!stop_finalizations().lock().unwrap().contains("broken"));
    assert_eq!(std::fs::read(&journal).unwrap(), before);
    let persisted: DaemonWorkerDescriptor = serde_json::from_slice(&std::fs::read(fixture.supervisor.descriptor_dir.join("broken.json")).unwrap()).unwrap();
    assert!(stop_cleanup_is_parked(&persisted));
    assert!(fixture.supervisor.reclaim_stale_worker_registration(&worker).await.unwrap_err().contains("parked"));
    fixture.supervisor.workers.lock().unwrap().clear();
    fixture.supervisor.adopt_workers().await.unwrap();
    let adopted = fixture.supervisor.workers.lock().unwrap().get("broken").unwrap().clone();
    assert!(stop_cleanup_is_parked(&adopted.descriptor.lock().unwrap()));
    assert!(!stop_finalizations().lock().unwrap().contains("broken"));
    assert_eq!(std::fs::read(&journal).unwrap(), before);
    fixture.supervisor.stopped.cancel();
}

#[tokio::test]
async fn core_supervisor_transient_stop_failure_has_finite_retry_budget() {
    let fixture = SupervisorFixture::new("core-transient-cleanup").await;
    let worker = stopped_worker(&fixture, "transient");
    let unreadable = fixture.root.join("orphan-journal-is-directory");
    std::fs::create_dir(&unreadable).unwrap();
    worker.descriptor.lock().unwrap().orphan_process_journal_path = Some(unreadable.to_string_lossy().into_owned());
    let descriptor = worker.descriptor.lock().unwrap().clone();
    stop_finalizations().lock().unwrap().insert(descriptor.worker_id.clone());
    tokio::time::timeout(Duration::from_secs(20), finalize_timed_out_worker_stop(fixture.supervisor.clone(), worker.clone(), descriptor)).await.expect("transient cleanup retries were unbounded");
    assert!(stop_cleanup_is_parked(&worker.descriptor.lock().unwrap()));
    assert!(fixture.supervisor.descriptor_dir.join("transient.json").exists());
    assert!(unreadable.exists());
    assert!(!stop_finalizations().lock().unwrap().contains("transient"));
    fixture.supervisor.stopped.cancel();
}

#[test]
fn core_supervisor_stop_retry_policy_distinguishes_invariants_from_io_errors() {
    for error in ["Uncertain operation has no saved transcript; recovery journal retained", "Malformed orphan record; recovery journal retained", "Unverified orphan record; recovery journal retained"] {
        assert!(!stop_cleanup_should_retry(error, 1));
    }
    for attempts in 1..STOP_CLEANUP_MAX_ATTEMPTS { assert!(stop_cleanup_should_retry("sharing violation", attempts)); }
    assert!(!stop_cleanup_should_retry("sharing violation", STOP_CLEANUP_MAX_ATTEMPTS));
}

#[tokio::test]
async fn core_supervisor_recovery_parking_preserves_replacement_descriptor() {
    let fixture = SupervisorFixture::new("core-recovery-park-generation").await;
    let retired = stopped_worker(&fixture, "park-generation");
    let replacement = stopped_worker(&fixture, "park-generation");
    let path = fixture.supervisor.descriptor_dir.join("park-generation.json");
    let before = std::fs::read(&path).unwrap();
    fixture.supervisor.park_worker_recovery_failure(&retired, "late recovery");
    fixture.supervisor.park_worker_stop_cleanup_failure(&retired, "late stop");
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(Arc::ptr_eq(fixture.supervisor.workers.lock().unwrap().get("park-generation").unwrap(), &replacement));
    fixture.supervisor.stopped.cancel();
}

#[tokio::test]
async fn core_supervisor_get_state_serves_roster_view_when_worker_cannot_answer() {
    let mut fixture = SupervisorFixture::new("core-get-state-fallback").await;
    // A READY worker with no connected client: the bounded worker forward is
    // skipped exactly like the saturated case, where the client's 30s deadline
    // used to wedge chat-open behind a request that would eventually succeed.
    let worker = add_descriptor_only_worker(&fixture, "saturated", "saturated-active", "fixture-token", DAEMON_WORKER_LIFECYCLE_READY);
    let summary = SessionSummary {
        active_session_id: Some("saturated-active".into()),
        id: "saturated-active".into(),
        session_id: "saturated-root".into(),
        session_name: Some("Saturated Chat".into()),
        cwd: fixture.root.join("workspace").to_string_lossy().into_owned(),
        lifecycle: "live".into(),
        activity: "streaming".into(),
        is_streaming: true,
        ..Default::default()
    };
    fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&summary.roster_view()), Some(&worker), None);
    let response = fixture.send(json!({"type":"get_state","id":"gs-fallback","activeSessionId":"saturated-active"})).await;
    assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["command"], "get_state");
    assert_eq!(response["data"]["sessionId"], "saturated-root");
    assert_eq!(response["data"]["sessionName"], "Saturated Chat");
    assert_eq!(response["data"]["isStreaming"], true);
    assert_eq!(response["data"]["workerState"], "recovering");
    assert_eq!(response["data"]["workerPid"], std::process::id() as i64);
    fixture.supervisor.stopped.cancel();
}

#[tokio::test]
async fn core_supervisor_get_state_unknown_session_still_fails() {
    let mut fixture = SupervisorFixture::new("core-get-state-unknown").await;
    let response = fixture.send(json!({"type":"get_state","id":"gs-unknown","activeSessionId":"missing"})).await;
    assert_eq!(response["success"], false, "{response}");
}

#[tokio::test]
async fn core_supervisor_get_state_falls_back_when_worker_connection_is_dead() {
    let mut fixture = SupervisorFixture::new("core-get-state-dead-client").await;
    let worker = add_descriptor_only_worker(&fixture, "wedged", "wedged-active", "fixture-token", DAEMON_WORKER_LIFECYCLE_READY);
    // A client object that never connected mirrors a worker whose socket died
    // mid-flight: the stored client must not be trusted for the forward.
    *worker.client.lock().unwrap() = Some(Arc::new(DaemonWorkerClient::new("unused-private-test-socket")));
    let summary = SessionSummary {
        active_session_id: Some("wedged-active".into()),
        id: "wedged-active".into(),
        session_id: "wedged-root".into(),
        session_name: Some("Wedged Chat".into()),
        cwd: fixture.root.join("workspace").to_string_lossy().into_owned(),
        lifecycle: "live".into(),
        activity: "idle".into(),
        ..Default::default()
    };
    fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&summary.roster_view()), Some(&worker), None);
    let response = fixture.send(json!({"type":"get_state","id":"gs-dead-client","activeSessionId":"wedged-active"})).await;
    assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["data"]["sessionId"], "wedged-root");
    fixture.supervisor.stopped.cancel();
}

// ---------------------------------------------------------------------------
// Audit S5/G2 (PR #168): get_state roster-fallback fidelity. The fallback
// fires exactly when the worker cannot answer (saturation, dead socket), so
// its queue placeholder must never read as an authoritative "no queued
// work", and a worker that DOES answer must keep its authoritative
// sessionActions through the supervisor's remap.
// ---------------------------------------------------------------------------

/// A scripted worker owning a real private socket: the supervisor sees a
/// connected client it can forward `get_state` to, and the script decides
/// whether that forward is answered or left to time out (saturation).
struct GetStateWorker {
    worker: Arc<Worker>,
    _client: Arc<DaemonWorkerClient>,
    commands: Arc<Mutex<Vec<Value>>>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for GetStateWorker {
    fn drop(&mut self) {
        if let Some(client) = self.worker.client.lock().unwrap().as_ref() {
            client.close_now();
        }
        self.task.abort();
    }
}

async fn serve_get_state<S: AsyncRead + AsyncWrite + Unpin>(
    mut socket: S,
    answer: Value,
    saturate: bool,
    commands: Arc<Mutex<Vec<Value>>>,
) {
    let mut decoder = PrivateFrameDecoder::new();
    let mut buffer = [0u8; 8192];
    loop {
        let count = match socket.read(&mut buffer).await {
            Ok(0) => return,
            Ok(count) => count,
            Err(_) => return,
        };
        for frame in decoder.push(&buffer[..count]).unwrap() {
            let command: Value = serde_json::from_slice(&frame.payload).unwrap();
            commands.lock().unwrap().push(command.clone());
            if command["type"].as_str() != Some("get_state") {
                let response = json!({"type":"response","id":command["id"],"command":command["type"],"success":true,"data":null});
                let frame = encode_private_frame(
                    &json!({"kind":"outbound","outboundType":"response","requestId":command["id"],"payloadEncoding":"jsonl"}),
                    response.to_string().as_bytes(),
                ).unwrap();
                socket.write_all(&frame).await.unwrap();
                continue;
            }
            if saturate {
                // The saturated worker: the request is accepted and never
                // answered, so the supervisor's bounded 5s forward fails.
                continue;
            }
            let response = json!({"type":"response","id":command["id"],"command":"get_state","success":true,"data":answer});
            let frame = encode_private_frame(
                &json!({"kind":"outbound","outboundType":"response","requestId":command["id"],"payloadEncoding":"jsonl"}),
                response.to_string().as_bytes(),
            ).unwrap();
            socket.write_all(&frame).await.unwrap();
        }
    }
}

async fn get_state_worker(fixture: &SupervisorFixture, name: &str, answer: Value, saturate: bool) -> GetStateWorker {
    let worker = add_descriptor_only_worker(fixture, name, name, &format!("token-{name}"), DAEMON_WORKER_LIFECYCLE_READY);
    let summary = SessionSummary {
        active_session_id: Some(name.to_string()),
        id: name.to_string(),
        session_id: format!("session-{name}"),
        session_name: Some(format!("Name {name}")),
        cwd: fixture.root.join("workspace").to_string_lossy().into_owned(),
        lifecycle: "resident".to_string(),
        activity: "streaming".to_string(),
        is_streaming: true,
        message_count: 4,
        ..Default::default()
    };
    fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&summary.roster_view()), Some(&worker), None);
    let commands = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&commands);
    let answer_for_task = answer.clone();
    #[cfg(windows)]
    let (socket, task) = {
        let socket = format!(r"\\.\pipe\optimus-getstate-worker-{}", uuid::Uuid::new_v4());
        let server = tokio::net::windows::named_pipe::ServerOptions::new().first_pipe_instance(true).create(&socket).unwrap();
        let task = tokio::spawn(async move {
            server.connect().await.unwrap();
            serve_get_state(server, answer_for_task, saturate, captured).await;
        });
        (socket, task)
    };
    #[cfg(unix)]
    let (socket, task) = {
        let socket = fixture.root.join(format!("{name}.sock")).to_string_lossy().into_owned();
        let server = tokio::net::UnixListener::bind(&socket).unwrap();
        let task = tokio::spawn(async move {
            let (stream, _) = server.accept().await.unwrap();
            serve_get_state(stream, answer_for_task, saturate, captured).await;
        });
        (socket, task)
    };
    let client = Arc::new(DaemonWorkerClient::new(&socket));
    client.connect(1_000).await.unwrap();
    worker.descriptor.lock().unwrap().socket_path = socket;
    *worker.client.lock().unwrap() = Some(Arc::clone(&client));
    GetStateWorker { worker, _client: client, commands, task }
}

/// The worker answered with real queued steering: the supervisor's remap must
/// keep the authoritative sessionActions instead of zeroing them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn core_supervisor_get_state_forward_preserves_worker_session_actions() {
    let mut fixture = SupervisorFixture::new("core-get-state-authoritative").await;
    let answer = json!({
        "id": "queued-chat",
        "lifecycle": "resident",
        "activity": "streaming",
        "isSessionActive": true,
        "activeSessionId": "queued-chat",
        "sessionId": "session-queued-chat",
        "sessionName": "Queued Chat",
        "cwd": fixture.root.join("workspace").to_string_lossy(),
        "isStreaming": true,
        "isCompacting": false,
        "attachedClients": 1,
        "messageCount": 9,
        "sessionActions": {
            "queuedCount": 2,
            "steering": ["turn left"],
            "followUps": [{"text": "then summarize"}],
        },
    });
    let worker = get_state_worker(&fixture, "queued-chat", answer, false).await;
    let response = fixture.send(json!({"type":"get_state","id":"gs-authoritative","activeSessionId":"queued-chat"})).await;
    assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["data"]["sessionId"], "session-queued-chat");
    assert_eq!(response["data"]["sessionActions"]["queuedCount"], 2, "{response}");
    assert_eq!(response["data"]["sessionActions"]["steering"], json!(["turn left"]), "{response}");
    assert_eq!(response["data"]["sessionActions"]["followUps"], json!([{"text": "then summarize"}]), "{response}");
    assert!(response["data"].get("stale").is_none(), "an authoritative answer is never marked stale: {response}");
    assert!(response["data"].get("fallback").is_none(), "an authoritative answer carries no fallback marker: {response}");
    assert_eq!(worker.commands.lock().unwrap().len(), 1, "the worker answered exactly one get_state");
    fixture.supervisor.stopped.cancel();
}

/// The saturated worker accepted the forward but never answered: after the
/// bounded 5s wait the supervisor serves the roster view, explicitly marked
/// stale, with the zeroed queue placeholder kept for shape parity.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn core_supervisor_get_state_saturated_fallback_is_marked_stale() {
    let mut fixture = SupervisorFixture::new("core-get-state-saturated").await;
    let answer = json!({"sessionId": "session-saturated"});
    let worker = get_state_worker(&fixture, "saturated-chat", answer, true).await;
    let started = std::time::Instant::now();
    let response = fixture.send(json!({"type":"get_state","id":"gs-saturated","activeSessionId":"saturated-chat"})).await;
    // The bounded forward must actually expire (5s) before the fallback runs.
    assert!(started.elapsed() >= Duration::from_millis(4_500), "the saturated forward did not wait its bound: {:?}", started.elapsed());
    assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["data"]["sessionId"], "session-saturated-chat");
    assert_eq!(response["data"]["stale"], true, "the roster fallback must be explicitly stale: {response}");
    assert_eq!(response["data"]["fallback"], "roster", "{response}");
    assert_eq!(response["data"]["sessionActions"]["queuedCount"], 0, "the placeholder keeps the summary shape: {response}");
    assert_eq!(response["data"]["sessionActions"]["steering"], json!([]), "{response}");
    assert_eq!(response["data"]["sessionActions"]["followUps"], json!([]), "{response}");
    let commands = worker.commands.lock().unwrap();
    assert_eq!(commands.len(), 1, "the saturated worker received the forward: {commands:?}");
    assert_eq!(commands[0]["type"], "get_state");
    fixture.supervisor.stopped.cancel();
}

/// A worker with no live client skips the forward entirely; its roster
/// fallback carries the same explicit stale marking.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn core_supervisor_get_state_disconnected_fallback_is_marked_stale() {
    let mut fixture = SupervisorFixture::new("core-get-state-disconnected").await;
    let worker = add_descriptor_only_worker(&fixture, "offline", "offline", "offline-token", DAEMON_WORKER_LIFECYCLE_READY);
    let summary = SessionSummary {
        active_session_id: Some("offline".into()),
        id: "offline".into(),
        session_id: "session-offline".into(),
        session_name: Some("Offline Chat".into()),
        cwd: fixture.root.join("workspace").to_string_lossy().into_owned(),
        lifecycle: "live".into(),
        activity: "idle".into(),
        ..Default::default()
    };
    fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&summary.roster_view()), Some(&worker), None);
    let response = fixture.send(json!({"type":"get_state","id":"gs-offline","activeSessionId":"offline"})).await;
    assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["data"]["sessionId"], "session-offline");
    assert_eq!(response["data"]["stale"], true, "the roster fallback must be explicitly stale: {response}");
    assert_eq!(response["data"]["fallback"], "roster", "{response}");
    fixture.supervisor.stopped.cancel();
}
