//! CR-03: exercise the real stop path against private framed transports and
//! expendable child processes. No installed daemon, profile or gateway is used.

use super::daemon_supervisor_parity_tests::{add_descriptor_only_worker, SupervisorFixture};
use super::*;
use crate::modes::daemon::daemon_worker_client::{encode_private_frame, PrivateFrameDecoder};

#[derive(Clone, Copy)]
enum Reply {
    Timeout,
    Disconnect,
    Reject,
    Success,
    ReplaceRegistration,
}

async fn child_worker(fixture: &SupervisorFixture) -> (tokio::process::Child, Arc<Worker>) {
    let powershell = PathBuf::from(std::env::var_os("SystemRoot").expect("SystemRoot"))
        .join("System32/WindowsPowerShell/v1.0/powershell.exe");
    let child = tokio::process::Command::new(powershell)
        .args(["-NoProfile", "-NonInteractive", "-Command", "Start-Sleep -Seconds 60"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .creation_flags(0x08000000)
        .kill_on_drop(true)
        .spawn()
        .expect("private sleeper process");
    let pid = child.id().expect("child pid") as i32;
    let start = get_process_start_id(pid as i64).expect("test-owned process identity");
    let worker_id = format!("cr03-{}", uuid::Uuid::new_v4());
    let worker = add_descriptor_only_worker(fixture, &worker_id, "cr03-active", "cr03-token", "ready");
    {
        let mut descriptor = worker.descriptor.lock().unwrap();
        descriptor.pid = pid;
        descriptor.process_start_id = Some(start);
        descriptor.worker_instance_id = Some(uuid::Uuid::new_v4().to_string());
        descriptor.socket_path = format!(r"\\.\pipe\optimus-cr03-{}", uuid::Uuid::new_v4());
        fixture.supervisor.persist_worker(&descriptor).expect("private worker descriptor");
    }
    (child, worker)
}

fn descriptor_path(fixture: &SupervisorFixture, worker: &Worker) -> PathBuf {
    fixture.supervisor.descriptor_dir.join(format!("{}.json", worker.descriptor.lock().unwrap().worker_id))
}

async fn scripted_client(worker: &Arc<Worker>, reply: Reply) -> (tokio::task::JoinHandle<()>, Arc<DaemonWorkerClient>) {
    let socket = worker.descriptor.lock().unwrap().socket_path.clone();
    let mut pipe = tokio::net::windows::named_pipe::ServerOptions::new()
        .first_pipe_instance(true).create(&socket).expect("private named pipe");
    let client = Arc::new(DaemonWorkerClient::new(&socket));
    let server_worker = Arc::clone(worker);
    let server = tokio::spawn(async move {
        pipe.connect().await.expect("private client connects");
        let mut decoder = PrivateFrameDecoder::new();
        let mut buffer = [0u8; 4096];
        let request = loop {
            let count = pipe.read(&mut buffer).await.expect("read stop request");
            assert!(count > 0, "stop request must reach the private transport");
            if let Some(frame) = decoder.push(&buffer[..count]).expect("valid stop frame").pop() {
                break frame;
            }
        };
        assert!(matches!(request.header["commandType"].as_str(), Some("shutdown" | "worker_archive_and_shutdown")));
        match reply {
            Reply::Disconnect => return,
            Reply::Timeout => std::future::pending::<()>().await,
            Reply::ReplaceRegistration => {
                server_worker.descriptor.lock().unwrap().worker_instance_id = Some("replacement-generation".into());
                let replacement = Arc::new(DaemonWorkerClient::new("unused-stop-replacement"));
                *server_worker.client.lock().unwrap() = Some(Arc::clone(&replacement));
                *server_worker.subscribed_client.lock().unwrap() = Some(replacement);
            }
            Reply::Reject | Reply::Success => {}
        }
        let success = matches!(reply, Reply::Success);
        let response = json!({
            "type": "response", "id": request.header["requestId"],
            "command": request.header["commandType"], "success": success,
            "error": if success { Value::Null } else { json!("synthetic shutdown rejection") },
            "data": if success { json!({"ok":true}) } else { Value::Null }
        });
        let header = json!({"kind": "outbound", "outboundType": "response", "requestId": request.header["requestId"]});
        let frame = encode_private_frame(&header, &serde_json::to_vec(&response).unwrap()).unwrap();
        pipe.write_all(&frame).await.expect("scripted acknowledgement");
        std::future::pending::<()>().await;
    });
    client.connect(2_000).await.expect("private worker client");
    *worker.client.lock().unwrap() = Some(Arc::clone(&client));
    (server, client)
}

async fn forced_transport_outcome(case: &str, reply: Reply, archive: bool) {
    let fixture = SupervisorFixture::new(case).await;
    let (mut child, worker) = child_worker(&fixture).await;
    let (server, client) = scripted_client(&worker, reply).await;
    let stale = if matches!(reply, Reply::Success) {
        Arc::clone(&client)
    } else {
        Arc::new(DaemonWorkerClient::new("stale-ready-stop-transport"))
    };
    *worker.subscribed_client.lock().unwrap() = Some(Arc::clone(&stale));
    let result = tokio::time::timeout(Duration::from_secs(8), fixture.supervisor.stop_worker(&worker, archive, true)).await;
    server.abort();
    let _ = server.await;
    assert!(matches!(result, Ok(Ok(()))), "force shutdown must continue beyond the RPC outcome: {result:?}");
    assert!(worker.client.lock().unwrap().is_none());
    assert!(worker.subscribed_client.lock().unwrap().is_none());
    assert_eq!(client.close_call_count(), 1, "request transport closes exactly once");
    assert_eq!(stale.close_call_count(), 1, "a shared transport is deduplicated; a stale ready transport closes once");
    tokio::time::timeout(Duration::from_secs(2), child.wait()).await.expect("test-owned worker exited").unwrap();
    assert!(fixture.supervisor.workers.lock().unwrap().is_empty());
    assert!(!descriptor_path(&fixture, &worker).exists(), "completed stop removes its tombstone");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_force_stops_after_rpc_timeout() {
    forced_transport_outcome("cr03-timeout", Reply::Timeout, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_force_stops_after_rpc_disconnect() {
    forced_transport_outcome("cr03-disconnect", Reply::Disconnect, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_force_stops_after_negative_archive_acknowledgement() {
    forced_transport_outcome("cr03-rejected", Reply::Reject, true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_graceful_stop_closes_shared_transport_and_clears_both_caches_once() {
    forced_transport_outcome("cr03-graceful-cache-cleanup", Reply::Success, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_force_preserves_replacement_process_generation() {
    let fixture = SupervisorFixture::new("cr03-replaced-pid").await;
    let (mut child, worker) = child_worker(&fixture).await;
    worker.descriptor.lock().unwrap().process_start_id = Some("not-this-process-generation".into());
    fixture.supervisor.stop_worker(&worker, false, true).await.expect("stale registration removed");
    assert!(child.try_wait().unwrap().is_none(), "a recycled PID must never be signalled");
    child.kill().await.unwrap();
    child.wait().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_force_keeps_unknown_identity_tombstoned() {
    let fixture = SupervisorFixture::new("cr03-unknown-pid").await;
    let (mut child, worker) = child_worker(&fixture).await;
    worker.descriptor.lock().unwrap().process_start_id = None;
    let current = Arc::new(DaemonWorkerClient::new("retained-current-stop-transport"));
    let stale = Arc::new(DaemonWorkerClient::new("retained-stale-ready-stop-transport"));
    *worker.client.lock().unwrap() = Some(Arc::clone(&current));
    *worker.subscribed_client.lock().unwrap() = Some(Arc::clone(&stale));
    let result = fixture.supervisor.stop_worker(&worker, false, true).await;
    assert!(result.as_ref().is_err_and(|error| error.contains("did not stop")), "unknown identity must not be reported dead: {result:?}");
    assert!(worker.client.lock().unwrap().is_none(), "retained stop clears the current cache");
    assert!(worker.subscribed_client.lock().unwrap().is_none(), "retained stop clears the stale ready cache");
    assert_eq!(current.close_call_count(), 1);
    assert_eq!(stale.close_call_count(), 1);
    // Leave the finalizer running past its escalation deadline: it must also
    // preserve a live process whose generation could not be authenticated.
    tokio::time::sleep(Duration::from_millis(super::super::STOP_FINALIZATION_SIGKILL_GRACE_MS + 250)).await;
    assert!(child.try_wait().unwrap().is_none(), "unknown PID identity must not be killed by finalization");
    assert!(descriptor_path(&fixture, &worker).exists());
    fixture.supervisor.stopped.cancel();
    child.kill().await.unwrap();
    child.wait().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_stop_does_not_follow_relaunched_registration() {
    let fixture = SupervisorFixture::new("cr03-relaunched").await;
    let (mut child, worker) = child_worker(&fixture).await;
    let (server, failed) = scripted_client(&worker, Reply::ReplaceRegistration).await;
    let stale = Arc::new(DaemonWorkerClient::new("old-ready-before-stop-replacement"));
    *worker.subscribed_client.lock().unwrap() = Some(Arc::clone(&stale));
    let result = fixture.supervisor.stop_worker(&worker, false, true).await;
    server.abort();
    let _ = server.await;
    assert!(result.as_ref().is_err_and(|error| error.contains("replaced during stop")), "stale stop must abort: {result:?}");
    let current = worker.client.lock().unwrap().clone().expect("newer current client survives old stop cleanup");
    let ready = worker.subscribed_client.lock().unwrap().clone().expect("newer ready client survives old stop cleanup");
    assert!(Arc::ptr_eq(&current, &ready));
    assert_eq!(current.close_call_count(), 0, "old stop cleanup never closes the replacement");
    assert_eq!(failed.close_call_count(), 1, "old request transport closes once");
    assert_eq!(stale.close_call_count(), 1, "old stale ready transport closes once");
    assert!(child.try_wait().unwrap().is_none(), "a replacement registration must not be followed");
    assert!(descriptor_path(&fixture, &worker).exists());
    child.kill().await.unwrap();
    child.wait().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_shutdown_requires_current_supervisor_ownership() {
    let fixture = SupervisorFixture::new("cr03-owner").await;
    let (mut child, worker) = child_worker(&fixture).await;
    fixture.supervisor.ownership.release().await.unwrap();
    let result = fixture.supervisor.stop_worker(&worker, false, true).await;
    assert!(result.is_err(), "an obsolete supervisor must refuse the stop");
    assert!(child.try_wait().unwrap().is_none());
    assert!(worker.descriptor.lock().unwrap().stop_requested_at.is_none());
    child.kill().await.unwrap();
    child.wait().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_delayed_finalizer_keeps_its_original_registration() {
    let fixture = SupervisorFixture::new("cr03-delayed-finalizer").await;
    let (mut child, worker) = child_worker(&fixture).await;
    let stopped = {
        let mut descriptor = worker.descriptor.lock().unwrap();
        descriptor.stop_requested_at = Some("original-stop".into());
        descriptor.clone()
    };
    let finalization = finalize_timed_out_worker_stop(Arc::clone(&fixture.supervisor), Arc::clone(&worker), stopped);
    worker.descriptor.lock().unwrap().worker_instance_id = Some("successor-before-first-poll".into());
    tokio::time::timeout(Duration::from_secs(1), finalization).await.expect("stale finalizer exits immediately");
    assert!(child.try_wait().unwrap().is_none());
    assert!(descriptor_path(&fixture, &worker).exists());
    child.kill().await.unwrap();
    child.wait().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cr03_ordinary_shutdown_does_not_force_on_negative_acknowledgement() {
    let fixture = SupervisorFixture::new("cr03-ordinary").await;
    let (mut child, worker) = child_worker(&fixture).await;
    let (server, client) = scripted_client(&worker, Reply::Reject).await;
    let stale = Arc::new(DaemonWorkerClient::new("ordinary-stale-ready-stop-transport"));
    *worker.subscribed_client.lock().unwrap() = Some(Arc::clone(&stale));
    let result = fixture.supervisor.stop_worker(&worker, false, false).await;
    fixture.supervisor.stopped.cancel();
    server.abort();
    let _ = server.await;
    assert!(result.as_ref().is_err_and(|error| error.contains("did not stop")), "ordinary shutdown must report its timeout: {result:?}");
    assert!(worker.client.lock().unwrap().is_none());
    assert!(worker.subscribed_client.lock().unwrap().is_none());
    assert_eq!(client.close_call_count(), 1);
    assert_eq!(stale.close_call_count(), 1);
    assert!(child.try_wait().unwrap().is_none(), "ordinary shutdown must not force termination before finalization");
    assert!(descriptor_path(&fixture, &worker).exists());
    child.kill().await.unwrap();
    child.wait().await.unwrap();
}
