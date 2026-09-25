//! Isolated startup recovery probes; no worker processes or provider calls.
use super::*;
use super::daemon_supervisor_parity_tests::{SupervisorFixture, add_descriptor_only_worker};

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
