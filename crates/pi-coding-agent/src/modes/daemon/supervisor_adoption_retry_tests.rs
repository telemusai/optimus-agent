//! Isolated startup recovery probes; no worker processes or provider calls.
use super::*;
use super::daemon_supervisor_parity_tests::{SupervisorFixture, add_descriptor_only_worker};
use crate::modes::daemon::daemon_worker_client::{encode_private_frame, PrivateFrameDecoder};

struct AdoptionListFailure {
    socket: String,
    observed: Arc<Mutex<Option<Arc<DaemonWorkerClient>>>>,
    replacement: Arc<DaemonWorkerClient>,
    task: tokio::task::JoinHandle<()>,
    _socket_root: Option<tempfile::TempDir>,
}

async fn serve_adoption_list_failure<S>(
    mut socket: S,
    hello: Value,
    supervisor: Arc<Supervisor>,
    worker_id: String,
    observed: Arc<Mutex<Option<Arc<DaemonWorkerClient>>>>,
    replacement: Arc<DaemonWorkerClient>,
    replace_during_list: bool,
) where S: AsyncRead + AsyncWrite + Unpin {
    let frame = encode_private_frame(
        &json!({"kind":"outbound","outboundType":"daemon_hello","payloadEncoding":"jsonl"}),
        hello.to_string().as_bytes(),
    ).unwrap();
    socket.write_all(&frame).await.unwrap();
    let mut decoder = PrivateFrameDecoder::new();
    let mut bytes = [0; 8192];
    loop {
        let count = socket.read(&mut bytes).await.unwrap();
        if count == 0 { break; }
        for frame in decoder.push(&bytes[..count]).unwrap() {
            let command: Value = serde_json::from_slice(&frame.payload).unwrap();
            let kind = command["type"].as_str().unwrap().to_string();
            assert!(["worker_auth", "worker_subscribe", "list"].contains(&kind.as_str()));
            if kind == "list" {
                // This is the real adoption sequence: authentication and subscribe
                // have published readiness before the failing list request.
                let worker = supervisor.workers.lock().unwrap().get(&worker_id).cloned().expect("adopting worker");
                let client = worker.client.lock().unwrap().clone().expect("authenticated client");
                let ready = worker.subscribed_client.lock().unwrap().clone().expect("subscribe published readiness");
                assert!(Arc::ptr_eq(&client, &ready));
                *observed.lock().unwrap() = Some(Arc::clone(&client));
                if replace_during_list {
                    *worker.client.lock().unwrap() = Some(Arc::clone(&replacement));
                    *worker.subscribed_client.lock().unwrap() = Some(Arc::clone(&replacement));
                    let mut descriptor = worker.descriptor.lock().unwrap();
                    descriptor.lifecycle = DAEMON_WORKER_LIFECYCLE_READY.into();
                    descriptor.last_error = None;
                }
            }
            let success = kind != "list";
            let data = if kind == "worker_auth" {
                json!({"capabilities":[DAEMON_WORKER_ROSTER_CAPABILITY]})
            } else if kind == "list" {
                json!({"sessions":[]})
            } else {
                json!({"ok":true})
            };
            let response = json!({
                "type":"response", "id":command["id"], "command":kind,
                "success":success,
                "error":if success {Value::Null} else {json!("synthetic list failure")},
                "data":data,
            });
            let response = encode_private_frame(
                &json!({"kind":"outbound","outboundType":"response","requestId":command["id"],"payloadEncoding":"jsonl"}),
                response.to_string().as_bytes(),
            ).unwrap();
            socket.write_all(&response).await.unwrap();
        }
    }
}

fn adoption_list_failure(
    fixture: &SupervisorFixture,
    worker_id: &str,
    replace_during_list: bool,
) -> AdoptionListFailure {
    let hello = fixture.supervisor.hello(&fixture.client);
    let observed = Arc::new(Mutex::new(None));
    let replacement = Arc::new(DaemonWorkerClient::new("unused-adoption-replacement"));
    #[cfg(unix)]
    let (socket, task, socket_root) = {
        let socket_root = tempfile::Builder::new().prefix("oa-").tempdir_in("/tmp").unwrap();
        let socket = socket_root.path().join("adoption-list.sock").to_string_lossy().into_owned();
        let server = tokio::net::UnixListener::bind(&socket).unwrap();
        let supervisor = Arc::clone(&fixture.supervisor);
        let worker_id = worker_id.to_string();
        let observed_server = Arc::clone(&observed);
        let replacement_server = Arc::clone(&replacement);
        let task = tokio::spawn(async move {
            let (socket, _) = server.accept().await.unwrap();
            serve_adoption_list_failure(
                socket, hello, supervisor, worker_id, observed_server,
                replacement_server, replace_during_list,
            ).await;
        });
        (socket, task, Some(socket_root))
    };
    #[cfg(windows)]
    let (socket, task, socket_root) = {
        let socket = format!(r"\\.\pipe\optimus-adoption-list-{}", uuid::Uuid::new_v4());
        let server = tokio::net::windows::named_pipe::ServerOptions::new()
            .first_pipe_instance(true).create(&socket).unwrap();
        let supervisor = Arc::clone(&fixture.supervisor);
        let worker_id = worker_id.to_string();
        let observed_server = Arc::clone(&observed);
        let replacement_server = Arc::clone(&replacement);
        let task = tokio::spawn(async move {
            server.connect().await.unwrap();
            serve_adoption_list_failure(
                server, hello, supervisor, worker_id, observed_server,
                replacement_server, replace_during_list,
            ).await;
        });
        (socket, task, None)
    };
    AdoptionListFailure { socket, observed, replacement, task, _socket_root: socket_root }
}

#[tokio::test]
async fn adoption_list_failure_clears_both_caches_and_closes_failed_client() {
    let fixture = SupervisorFixture::new("adoption-list-failure-cache-cleanup").await;
    let worker = add_descriptor_only_worker(&fixture, "adopt-list", "root", "synthetic", DAEMON_WORKER_LIFECYCLE_READY);
    let scripted = adoption_list_failure(&fixture, "adopt-list", false);
    worker.descriptor.lock().unwrap().socket_path = scripted.socket.clone();
    fixture.supervisor.persist_worker(&worker.descriptor.lock().unwrap()).unwrap();

    fixture.supervisor.adopt_workers().await.unwrap();

    let worker = fixture.supervisor.workers.lock().unwrap()["adopt-list"].clone();
    let failed = scripted.observed.lock().unwrap().clone().expect("list observed the subscribed client");
    assert!(worker.client.lock().unwrap().is_none(), "failed current client is detached");
    assert!(worker.subscribed_client.lock().unwrap().is_none(), "failed ready client is detached");
    assert!(Supervisor::ready_connected_client(&worker).is_none(), "the discarded transport cannot be resurrected by the ready fast path");
    assert_eq!(failed.close_call_count(), 1, "the failed transport is closed exactly once");
    assert_eq!(worker.descriptor.lock().unwrap().lifecycle, DAEMON_WORKER_LIFECYCLE_RECOVERING);
    assert!(worker.deferred_recovery.load(Ordering::SeqCst));
    fixture.supervisor.stopped.cancel();
    tokio::time::timeout(Duration::from_secs(2), scripted.task).await.expect("failed client closes its socket").unwrap();
    fixture.supervisor.ownership.release().await.unwrap();
}

#[tokio::test]
async fn adoption_list_failure_preserves_replacement_and_still_closes_failed_client() {
    let fixture = SupervisorFixture::new("adoption-list-failure-replacement").await;
    let worker = add_descriptor_only_worker(&fixture, "adopt-race", "root", "synthetic", DAEMON_WORKER_LIFECYCLE_READY);
    let scripted = adoption_list_failure(&fixture, "adopt-race", true);
    worker.descriptor.lock().unwrap().socket_path = scripted.socket.clone();
    fixture.supervisor.persist_worker(&worker.descriptor.lock().unwrap()).unwrap();

    fixture.supervisor.adopt_workers().await.unwrap();

    let worker = fixture.supervisor.workers.lock().unwrap()["adopt-race"].clone();
    let failed = scripted.observed.lock().unwrap().clone().expect("list observed the subscribed client");
    let current = worker.client.lock().unwrap().clone().expect("replacement remains current");
    let ready = worker.subscribed_client.lock().unwrap().clone().expect("replacement remains ready");
    assert!(Arc::ptr_eq(&current, &scripted.replacement));
    assert!(Arc::ptr_eq(&ready, &scripted.replacement));
    assert!(!Arc::ptr_eq(&ready, &failed), "the discarded client is not restored as readiness");
    assert_eq!(failed.close_call_count(), 1, "stale failure closes only its own transport");
    assert_eq!(scripted.replacement.close_call_count(), 0, "replacement is never closed");
    assert_eq!(worker.descriptor.lock().unwrap().lifecycle, DAEMON_WORKER_LIFECYCLE_READY);
    assert!(worker.descriptor.lock().unwrap().last_error.is_none());
    assert!(!worker.deferred_recovery.load(Ordering::SeqCst));
    fixture.supervisor.stopped.cancel();
    tokio::time::timeout(Duration::from_secs(2), scripted.task).await.expect("failed client closes its socket").unwrap();
    fixture.supervisor.ownership.release().await.unwrap();
}

#[test]
fn adoption_identity_observation_distinguishes_unknown_from_replaced_and_dead() {
    assert!(process_identity_observation(true, Some("start"), None) == ProcessIdentityVerdict::Unknown);
    assert!(process_identity_observation(true, None, Some("start")) == ProcessIdentityVerdict::Unknown);
    assert!(process_identity_observation(true, Some("old"), Some("new")) == ProcessIdentityVerdict::Gone);
    assert!(process_identity_observation(false, Some("start"), None) == ProcessIdentityVerdict::Gone);
    assert!(process_identity_observation(true, Some("start"), Some("start")) == ProcessIdentityVerdict::Current);
}

#[tokio::test]
async fn adoption_unknown_identity_is_retained_for_bounded_recovery() {
    let fixture = SupervisorFixture::new("adoption-unknown-retry").await;
    let old = add_descriptor_only_worker(&fixture, "unknown", "root", "synthetic", DAEMON_WORKER_LIFECYCLE_READY);
    let mut descriptor = old.descriptor.lock().unwrap().clone();
    descriptor.process_start_id = None;
    fixture.supervisor.persist_worker(&descriptor).unwrap();
    fixture.supervisor.adopt_workers().await.unwrap();
    let worker = fixture.supervisor.workers.lock().unwrap()["unknown"].clone();
    assert!(worker.client.lock().unwrap().is_none());
    assert!(worker.deferred_recovery.load(Ordering::SeqCst));
    let after = worker.descriptor.lock().unwrap().clone();
    assert_eq!(after.lifecycle, DAEMON_WORKER_LIFECYCLE_RECOVERING);
    assert_eq!(after.pid, descriptor.pid);
    assert_eq!(after.process_start_id, None, "do not invent ownership from the observed PID");
    assert!(after.stop_requested_at.is_none());
    assert_eq!(after.authentication_token, descriptor.authentication_token);
    fixture.supervisor.stopped.cancel();
    fixture.supervisor.ownership.release().await.unwrap();
}

#[tokio::test]
async fn adoption_failure_cannot_revive_explicit_stop_or_replace_new_client() {
    let fixture = SupervisorFixture::new("adoption-stop-client-fences").await;
    let worker = add_descriptor_only_worker(&fixture, "fenced", "root", "synthetic", DAEMON_WORKER_LIFECYCLE_STOPPING);
    let failed = Arc::new(DaemonWorkerClient::new("unused-failed"));
    *worker.client.lock().unwrap() = Some(failed.clone());
    worker.descriptor.lock().unwrap().stop_requested_at = Some(chrono::Utc::now().to_rfc3339());
    fixture.supervisor.defer_worker_adoption(&worker, &failed, "test failure");
    assert_eq!(worker.descriptor.lock().unwrap().lifecycle, DAEMON_WORKER_LIFECYCLE_STOPPING);
    assert!(!worker.deferred_recovery.load(Ordering::SeqCst));
    let replacement = Arc::new(DaemonWorkerClient::new("unused-replacement"));
    *worker.client.lock().unwrap() = Some(replacement.clone());
    Supervisor::discard_failed_worker_client(&worker, &failed);
    assert!(Arc::ptr_eq(worker.client.lock().unwrap().as_ref().unwrap(), &replacement));
    fixture.supervisor.defer_worker_adoption(&worker, &failed, "stale failure");
    assert!(!worker.deferred_recovery.load(Ordering::SeqCst));
    fixture.supervisor.stopped.cancel();
    fixture.supervisor.ownership.release().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn adoption_failed_first_probe_reconnects_without_replay_or_identity_change() {
    use crate::modes::daemon::daemon_worker_client::{encode_private_frame, PrivateFrameDecoder};
    let fixture = SupervisorFixture::new("adoption-live-probe-retry").await;
    let old = add_descriptor_only_worker(&fixture, "live", "root", "synthetic", DAEMON_WORKER_LIFECYCLE_READY);
    let mut descriptor = old.descriptor.lock().unwrap().clone();
    // Unix socket paths must stay below SUN_LEN even under a long harness TMPDIR.
    let socket_root = tempfile::Builder::new().prefix("oa-").tempdir_in("/tmp").unwrap();
    descriptor.socket_path = socket_root.path().join("probe.sock").to_string_lossy().into_owned();
    fixture.supervisor.persist_worker(&descriptor).unwrap();
    let server = tokio::net::UnixListener::bind(&descriptor.socket_path).unwrap();
    let hello = fixture.supervisor.hello(&fixture.client);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let captured = seen.clone();
    let task = tokio::spawn(async move {
        for attempt in 0..2 {
            let (mut socket, _) = server.accept().await.unwrap();
            let frame = encode_private_frame(&json!({"kind":"outbound","outboundType":"daemon_hello","payloadEncoding":"jsonl"}), hello.to_string().as_bytes()).unwrap();
            socket.write_all(&frame).await.unwrap();
            let mut decoder = PrivateFrameDecoder::new();
            let mut bytes = [0; 8192];
            loop {
                let count = socket.read(&mut bytes).await.unwrap();
                if count == 0 { break; }
                for frame in decoder.push(&bytes[..count]).unwrap() {
                    let command: Value = serde_json::from_slice(&frame.payload).unwrap();
                    let kind = command["type"].as_str().unwrap().to_string();
                    captured.lock().unwrap().push(kind.clone());
                    assert!(["worker_auth", "worker_subscribe", "list"].contains(&kind.as_str()), "no prompt replay or new worker launch");
                    let success = attempt != 0 || kind != "worker_auth";
                    let data = if kind == "worker_auth" { json!({"capabilities":[DAEMON_WORKER_ROSTER_CAPABILITY]}) } else { json!({"sessions":[]}) };
                    let response = json!({"type":"response","id":command["id"],"command":kind,"success":success,"error":if success {Value::Null} else {json!("temporarily unavailable")},"data":data});
                    let frame = encode_private_frame(&json!({"kind":"outbound","outboundType":"response","requestId":command["id"],"payloadEncoding":"jsonl"}), response.to_string().as_bytes()).unwrap();
                    socket.write_all(&frame).await.unwrap();
                }
            }
        }
    });
    fixture.supervisor.adopt_workers().await.unwrap();
    let worker = fixture.supervisor.workers.lock().unwrap()["live"].clone();
    assert_eq!(worker.descriptor.lock().unwrap().lifecycle, DAEMON_WORKER_LIFECYCLE_RECOVERING);
    assert!(worker.deferred_recovery.load(Ordering::SeqCst));
    assert!(worker.client.lock().unwrap().is_none());
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if worker.descriptor.lock().unwrap().lifecycle == DAEMON_WORKER_LIFECYCLE_READY { break; }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }).await.expect("existing deferred ladder must reconnect");
    let after = worker.descriptor.lock().unwrap().clone();
    assert_eq!(after.pid, descriptor.pid);
    assert_eq!(after.process_start_id, descriptor.process_start_id);
    assert_eq!(after.authentication_token, descriptor.authentication_token);
    assert_eq!(seen.lock().unwrap().iter().filter(|kind| kind.as_str() == "worker_auth").count(), 2);
    fixture.supervisor.stopped.cancel();
    if let Some(client) = worker.client.lock().unwrap().take() { client.close_now(); }
    task.abort();
    fixture.supervisor.ownership.release().await.unwrap();
}

async fn connected_client_subscribe_attempt<S>(
    mut socket: S,
    attempt: usize,
    hello: Value,
    captured: Arc<Mutex<Vec<String>>>,
) where S: AsyncRead + AsyncWrite + Unpin {
    let frame = encode_private_frame(&json!({"kind":"outbound","outboundType":"daemon_hello","payloadEncoding":"jsonl"}), hello.to_string().as_bytes()).unwrap();
    socket.write_all(&frame).await.unwrap();
    let mut decoder = PrivateFrameDecoder::new();
    let mut bytes = [0; 8192];
    loop {
        let count = socket.read(&mut bytes).await.unwrap();
        if count == 0 { break; }
        for frame in decoder.push(&bytes[..count]).unwrap() {
            let command: Value = serde_json::from_slice(&frame.payload).unwrap();
            let kind = command["type"].as_str().unwrap().to_string();
            captured.lock().unwrap().push(kind.clone());
            let success = attempt != 0 || kind != "worker_subscribe";
            let data = if kind == "worker_auth" { json!({"capabilities":[DAEMON_WORKER_ROSTER_CAPABILITY]}) } else { json!({"ok":true}) };
            let response = json!({"type":"response","id":command["id"],"command":kind.clone(),"success":success,"error":if success {Value::Null} else {json!("first subscribe failed")},"data":data});
            let response = encode_private_frame(&json!({"kind":"outbound","outboundType":"response","requestId":command["id"],"payloadEncoding":"jsonl"}), response.to_string().as_bytes()).unwrap();
            socket.write_all(&response).await.unwrap();
            if attempt == 1 && kind == "worker_subscribe" {
                // Prove readiness by reaching the real worker listener and
                // public event forwarding path, not a subscription helper mock.
                let event = json!({"type":"session_event","activeSessionId":"root","marker":"delivered-after-retry"});
                let event = encode_private_frame(&json!({"kind":"outbound","outboundType":"session_event","activeSessionId":"root","payloadEncoding":"jsonl"}), event.to_string().as_bytes()).unwrap();
                socket.write_all(&event).await.unwrap();
            }
        }
    }
}

#[tokio::test]
async fn connected_client_retries_failed_subscribe_and_delivers_events() {
    let mut fixture = SupervisorFixture::new("connected-client-subscribe-transaction").await;
    let worker = add_descriptor_only_worker(&fixture, "live", "root", "synthetic", DAEMON_WORKER_LIFECYCLE_READY);
    fixture.client.subscriptions.lock().unwrap().insert("root".into());
    let hello = fixture.supervisor.hello(&fixture.client);
    let seen = Arc::new(Mutex::new(Vec::new()));
    #[cfg(unix)]
    let (_socket_root, socket, task) = {
        // Unix socket paths must stay below SUN_LEN even under a long harness TMPDIR.
        let socket_root = tempfile::Builder::new().prefix("oa-").tempdir_in("/tmp").unwrap();
        let socket = socket_root.path().join("subscribe.sock").to_string_lossy().into_owned();
        let server = tokio::net::UnixListener::bind(&socket).unwrap();
        let captured = seen.clone();
        let task = tokio::spawn(async move {
            for attempt in 0..2 {
                let (socket, _) = server.accept().await.unwrap();
                connected_client_subscribe_attempt(socket, attempt, hello.clone(), captured.clone()).await;
            }
        });
        (socket_root, socket, task)
    };
    #[cfg(windows)]
    let (socket, task) = {
        let socket = format!(r"\\.\pipe\optimus-connected-client-{}", uuid::Uuid::new_v4());
        let captured = seen.clone();
        let server_socket = socket.clone();
        let task = tokio::spawn(async move {
            for attempt in 0..2 {
                let server = tokio::net::windows::named_pipe::ServerOptions::new()
                    .first_pipe_instance(attempt == 0).create(&server_socket).unwrap();
                server.connect().await.unwrap();
                connected_client_subscribe_attempt(server, attempt, hello.clone(), captured.clone()).await;
            }
        });
        (socket, task)
    };
    worker.descriptor.lock().unwrap().socket_path = socket;
    let first = tokio::time::timeout(Duration::from_secs(5), fixture.supervisor.connected_client(&worker)).await.unwrap();
    let first_error = match first { Ok(_) => panic!("first subscribe unexpectedly succeeded"), Err(error) => error };
    assert!(first_error.contains("first subscribe failed"));
    assert!(worker.client.lock().unwrap().is_none(), "failed client is not cached");
    assert!(worker.subscribed_client.lock().unwrap().is_none());
    let client = tokio::time::timeout(Duration::from_secs(5), fixture.supervisor.connected_client(&worker)).await.unwrap().expect("retry resubscribes");
    assert!(client.is_connected());
    let event = fixture.next_frame(Duration::from_secs(2)).await.expect("event after successful resubscribe");
    assert_eq!(event["marker"], "delivered-after-retry");
    assert_eq!(seen.lock().unwrap().iter().filter(|kind| kind.as_str() == "worker_subscribe").count(), 2);
    assert!(Arc::ptr_eq(worker.subscribed_client.lock().unwrap().as_ref().unwrap(), &client));
    fixture.supervisor.stopped.cancel();
    client.close_now();
    task.abort();
    fixture.supervisor.ownership.release().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn adoption_failed_reconnect_keeps_client_replaced_during_subscribe() {
    use crate::modes::daemon::daemon_worker_client::{encode_private_frame, PrivateFrameDecoder};
    let fixture = SupervisorFixture::new("adoption-reconnect-client-race").await;
    let worker = add_descriptor_only_worker(&fixture, "race", "root", "synthetic", DAEMON_WORKER_LIFECYCLE_RECOVERING);
    let socket_root = tempfile::Builder::new().prefix("oa-").tempdir_in("/tmp").unwrap();
    let socket = socket_root.path().join("race.sock").to_string_lossy().into_owned();
    worker.descriptor.lock().unwrap().socket_path = socket.clone();
    let server = tokio::net::UnixListener::bind(&socket).unwrap();
    let hello = fixture.supervisor.hello(&fixture.client);
    let replacement = Arc::new(DaemonWorkerClient::new("unused-replacement"));
    let newer = replacement.clone();
    let raced = worker.clone();
    let task = tokio::spawn(async move {
        let (mut socket, _) = server.accept().await.unwrap();
        let frame = encode_private_frame(&json!({"kind":"outbound","outboundType":"daemon_hello","payloadEncoding":"jsonl"}), hello.to_string().as_bytes()).unwrap();
        socket.write_all(&frame).await.unwrap();
        let mut decoder = PrivateFrameDecoder::new();
        let mut bytes = [0; 8192];
        loop {
            let count = socket.read(&mut bytes).await.unwrap();
            if count == 0 { return; }
            for frame in decoder.push(&bytes[..count]).unwrap() {
                let command: Value = serde_json::from_slice(&frame.payload).unwrap();
                let subscribing = command["type"] == "worker_subscribe";
                if subscribing { *raced.client.lock().unwrap() = Some(newer.clone()); }
                let response = json!({"type":"response","id":command["id"],"command":command["type"],"success":!subscribing,"error":if subscribing {json!("subscribe failed")} else {Value::Null},"data":{"capabilities":[DAEMON_WORKER_ROSTER_CAPABILITY]}});
                let frame = encode_private_frame(&json!({"kind":"outbound","outboundType":"response","requestId":command["id"],"payloadEncoding":"jsonl"}), response.to_string().as_bytes()).unwrap();
                socket.write_all(&frame).await.unwrap();
            }
        }
    });
    let result = tokio::time::timeout(Duration::from_secs(2), fixture.supervisor.reconnect_worker(&worker)).await.unwrap();
    assert!(result.is_err());
    assert!(Arc::ptr_eq(worker.client.lock().unwrap().as_ref().unwrap(), &replacement));
    fixture.supervisor.stopped.cancel();
    task.abort();
    fixture.supervisor.ownership.release().await.unwrap();
}


#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn adoption_stale_probe_cannot_downgrade_concurrently_reconnected_ready_worker() {
    let fixture = SupervisorFixture::new("adoption-transition-race").await;
    let worker = add_descriptor_only_worker(&fixture, "transition", "root", "synthetic", DAEMON_WORKER_LIFECYCLE_RECOVERING);
    let failed = Arc::new(DaemonWorkerClient::new("unused-failed"));
    *worker.client.lock().unwrap() = Some(failed.clone());
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let supervisor = fixture.supervisor.clone();
    let raced = worker.clone();
    let gate = barrier.clone();
    let deferred = tokio::task::spawn_blocking(move || {
        // The failed probe captured the old client while it was still current.
        assert!(Arc::ptr_eq(raced.client.lock().unwrap().as_ref().unwrap(), &failed));
        let id = raced.descriptor.lock().unwrap().worker_id.clone();
        gate.wait();
        gate.wait();
        supervisor.defer_worker_adoption_if_current(&raced, &failed, &id, "stale probe failure");
    });
    barrier.wait();
    let replacement = Arc::new(DaemonWorkerClient::new("unused-reconnected"));
    *worker.client.lock().unwrap() = Some(replacement.clone());
    {
        let mut descriptor = worker.descriptor.lock().unwrap();
        descriptor.lifecycle = DAEMON_WORKER_LIFECYCLE_READY.into();
        descriptor.last_error = None;
        fixture.supervisor.persist_worker(&descriptor).unwrap();
    }
    barrier.wait();
    deferred.await.unwrap();
    assert!(Arc::ptr_eq(worker.client.lock().unwrap().as_ref().unwrap(), &replacement));
    assert_eq!(worker.descriptor.lock().unwrap().lifecycle, DAEMON_WORKER_LIFECYCLE_READY);
    assert!(worker.descriptor.lock().unwrap().last_error.is_none());
    assert!(!worker.deferred_recovery.load(Ordering::SeqCst));
    let persisted: DaemonWorkerDescriptor = serde_json::from_slice(&std::fs::read(fixture.supervisor.descriptor_dir.join("transition.json")).unwrap()).unwrap();
    assert_eq!(persisted.lifecycle, DAEMON_WORKER_LIFECYCLE_READY);
    fixture.supervisor.stopped.cancel();
    fixture.supervisor.ownership.release().await.unwrap();
}
