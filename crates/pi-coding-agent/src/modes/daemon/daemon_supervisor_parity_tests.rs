//! Owner parity-validation tests for T09 (supervisor, recovery and protocol
//! contracts): findings C-03, C-04, C-05, C-06, C-07, C-08.
//!
//! These run inside the crate so they can construct the real `Supervisor` and drive
//! its real `dispatch`/`handle_line` entry points. All writable state lives under
//! `PARITY_DAEMON_STATE_ROOT`; nothing here touches the live daemon, its pipe,
//! `~/.prime/supervisor-owners`, or the user profile.

use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use super::*;

/// Isolated state root for this owner's fixtures (V00).
pub(super) fn state_root(case: &str) -> std::path::PathBuf {
    static ROOT: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let base = std::env::var_os("PARITY_DAEMON_STATE_ROOT")
        .map(|raw| {
            assert!(!raw.is_empty(), "PARITY_DAEMON_STATE_ROOT must not be empty");
            std::path::absolute(raw).expect("absolute state root")
        })
        .unwrap_or_else(|| {
            ROOT.get_or_init(|| tempfile::Builder::new().prefix("optimus-daemon-t09-").tempdir().expect("private state root"))
                .path().to_path_buf()
        });
    assert!(!base.to_string_lossy().to_lowercase().contains(".prime"), "state root must not touch .prime");
    std::fs::create_dir_all(&base).expect("state root");
    assert!(!std::fs::canonicalize(&base).expect("canonical state root").to_string_lossy().to_lowercase().contains(".prime"), "state root must not resolve into .prime");
    let root = base.join(case);
    std::fs::create_dir_all(root.join("agent")).expect("case root");
    std::fs::create_dir_all(root.join("workspace")).expect("case workspace");
    std::fs::create_dir_all(root.join("tmp")).expect("case temp");
    root
}

/// A public client attached to a scripted supervisor, with the frames the
/// supervisor wrote to it.
pub(super) struct SupervisorFixture {
    pub root: std::path::PathBuf,
    pub supervisor: Arc<Supervisor>,
    pub client: Arc<PublicClient>,
    pub outbound: mpsc::Receiver<Vec<u8>>,
}

impl SupervisorFixture {
    pub async fn new(case: &str) -> Self {
        let root = state_root(case);
        // Production normalizes its socket before owning/persisting descriptors.
        // Keep this invariant in the fixture too (Windows paths fold to lowercase).
        let socket_path = normalize_socket_path_for_daemon(&root.join("supervisor-parity.sock").to_string_lossy(), None);
        let descriptor_dir = root.join("daemon-workers").to_string_lossy().into_owned();
        let registry_dir = root.join("registry").to_string_lossy().into_owned();
        std::fs::create_dir_all(&descriptor_dir).expect("descriptor dir");
        let ownership = acquire_daemon_supervisor_ownership(
            AcquireDaemonSupervisorOwnershipOptions {
                socket_path: socket_path.clone(),
                descriptor_dir: descriptor_dir.clone(),
                agent_dir: root.join("agent").to_string_lossy().into_owned(),
                generation: uuid::Uuid::new_v4().to_string(),
                app_version: crate::config::VERSION.to_string(),
                registry_dir: Some(registry_dir.clone()),
            },
        )
        .await
        .expect("supervisor ownership");
        let journal = CommandRecoveryJournal::new(
            &Path::new(&descriptor_dir).join("command-journal.jsonl").to_string_lossy(),
        )
        .expect("command journal");
        let supervisor = Arc::new(Supervisor {
            eviction_fence: tokio::sync::RwLock::new(()),
            idle_eviction_task: Mutex::new(None),
            socket_path: socket_path.clone(),
            journal: Mutex::new(journal),
            agent_message_delivery_journal: Mutex::new(AgentMessageDeliveryJournal::new(
                &Path::new(&descriptor_dir)
                    .join(AGENT_MESSAGE_DELIVERY_JOURNAL_FILE)
                    .to_string_lossy(),
            )),
            descriptor_dir: PathBuf::from(&descriptor_dir),
            config: AgentSessionRuntimeConfig {
                cwd: Some(root.join("workspace").to_string_lossy().into_owned()),
                agent_dir: Some(root.join("agent").to_string_lossy().into_owned()),
                session_dir: Some(root.join("sessions").to_string_lossy().into_owned()),
                ..AgentSessionRuntimeConfig::default()
            },
            ownership,
            workers: Mutex::new(HashMap::new()),
            clients: Mutex::new(HashMap::new()),
            opening: tokio::sync::RwLock::new(()),
            pauses: Mutex::new(HashMap::new()),
            catalog: Arc::new(DaemonCatalogClient::new(Arc::new(|_| {}))),
            stopped: CancellationToken::new(),
            roster: Mutex::new(None),
            pending_roster_changed: Mutex::new(HashSet::new()),
            pending_roster_removed: Mutex::new(HashSet::new()),
            published_roster_ids: Mutex::new(HashSet::new()),
            roster_push_scheduled: AtomicBool::new(false),
            ledger: Arc::new(tokio::sync::OnceCell::new()),
            scheduled_wake_timer: Mutex::new(None),
            scheduled_wake_recompute: AtomicBool::new(false),
            scheduled_wake_recompute_queued: AtomicBool::new(false),
            scheduled_wake_failures: Mutex::new(HashMap::new()),
            prompt_admissions: Mutex::new(HashMap::new()),
            opening_workers: Mutex::new(HashMap::new()),
            pending_session_names: Mutex::new(HashSet::new()),
            pending_command_journal_log: Mutex::new(None),
        });
        supervisor.init_roster();
        let (sender, outbound) = mpsc::channel::<Vec<u8>>(1024);
        let identity = create_active_session_id(None);
        let client = Arc::new(PublicClient {
            connection_id: identity.clone(),
            id: Mutex::new(identity.clone()),
            protocol_id: Mutex::new(None),
            subscriptions: Mutex::new(HashSet::new()),
            supports_extension_ui: AtomicBool::new(false),
            pause_epoch: AtomicU64::new(0),
            output: sender,
            stopped: CancellationToken::new(),
            roster_subscribed: AtomicBool::new(false),
            roster_resync_pending: AtomicBool::new(false),
            backpressured: AtomicBool::new(false),
        });
        supervisor
            .clients
            .lock()
            .unwrap()
            .insert(identity, Arc::clone(&client));
        Self { root, supervisor, client, outbound }
    }

    /// Drive one raw command line through the real supervisor `handle_line`.
    pub async fn send(&mut self, value: Value) -> Value {
        let line = serde_json::to_vec(&value).expect("json line");
        self.supervisor.clone().handle_line(Arc::clone(&self.client), line).await;
        self.last_frame()
    }

    /// Drive one command and return every frame it produced, in order.
    pub async fn send_collect(&mut self, value: Value) -> Vec<Value> {
        let line = serde_json::to_vec(&value).expect("json line");
        self.supervisor.clone().handle_line(Arc::clone(&self.client), line).await;
        let mut frames = Vec::new();
        while let Ok(frame) = self.outbound.try_recv() {
            if let Ok(value) = serde_json::from_slice::<Value>(&frame) {
                frames.push(value);
            }
        }
        frames
    }

    /// The last parsed frame the supervisor wrote, if any.
    pub fn last_frame(&mut self) -> Value {
        let mut last = Value::Null;
        while let Ok(frame) = self.outbound.try_recv() {
            if let Ok(value) = serde_json::from_slice::<Value>(&frame) {
                last = value;
            }
        }
        last
    }

    /// Wait for the next frame and parse it.
    pub async fn next_frame(&mut self, timeout: Duration) -> Option<Value> {
        match tokio::time::timeout(timeout, self.outbound.recv()).await {
            Ok(Some(frame)) => serde_json::from_slice::<Value>(&frame).ok(),
            _ => None,
        }
    }

    /// The supervisor's hello frame (its first write to the client).
    pub fn hello(&self) -> Value {
        self.supervisor.hello(&self.client)
    }
}

impl Drop for SupervisorFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Register one scripted worker in the supervisor's map with no live client.
pub(super) fn add_descriptor_only_worker(
    fixture: &SupervisorFixture,
    worker_id: &str,
    root_active_session_id: &str,
    token: &str,
    lifecycle: &str,
) -> Arc<Worker> {
    let socket_path = fixture.supervisor.socket_path.clone();
    let descriptor = DaemonWorkerDescriptor {
        version: 2,
        worker_id: worker_id.to_string(),
        pid: std::process::id() as i32,
        process_start_id: get_process_start_id(std::process::id() as i64),
        socket_path: worker_socket(&socket_path, worker_id),
        recovery_journal_path: Path::new(&fixture.supervisor.descriptor_dir)
            .join(format!("{worker_id}-journal.jsonl"))
            .to_string_lossy()
            .into_owned(),
        orphan_process_journal_path: None,
        supervisor_socket_path: socket_path,
        authentication_token: token.to_string(),
        worker_instance_id: None,
        root_active_session_id: root_active_session_id.to_string(),
        owner_client_id: None,
        root_session_id: Some(root_active_session_id.to_string()),
        session_file: None,
        session_dir: None,
        telemetry_disabled: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        updated_at: chrono::Utc::now().to_rfc3339(),
        lifecycle: lifecycle.to_string(),
        create_command: DurableDaemonCreateCommand { type_: "create".to_string(), session_path: None, no_session: Some(true), extra: Default::default() },
        consecutive_failures: 0,
        stop_requested_at: None,
        archive_on_stop: None,
        last_failure_at: None,
        last_error: None,
    };
    fixture.supervisor.persist_worker(&descriptor).expect("persist worker");
    let worker = Arc::new(Worker {
        descriptor: Mutex::new(descriptor),
        client: Mutex::new(None),
        roster_epoch: AtomicU64::new(0),
        roster_stale: AtomicBool::new(false),
        last_frame_at: Mutex::new(None),
        pending_client: Mutex::new(None),
        connection: AsyncMutex::new(()),
        stream: Mutex::new(CompactAssistantStreamReconstructor::new()),
        recovery: AtomicBool::new(false),
        deferred_recovery: AtomicBool::new(false),
        deferred_recovery_rounds: AtomicU64::new(0),
        promoted_owner_client_id: Mutex::new(None),
        heartbeat_snapshot: Mutex::new(HeartbeatSnapshot::default()),
    });
    fixture
        .supervisor
        .workers
        .lock()
        .unwrap()
        .insert(worker_id.to_string(), Arc::clone(&worker));
    worker
}



// ---------------------------------------------------------------------------
// C-05: capabilities must match what the supervisor actually serves, and the
// unknown-command / protocol gates must exist.
// ---------------------------------------------------------------------------

/// C-05 (a): `daemon_hello.serverCapabilities` must advertise what the supervisor
/// serves. TS `SUPERVISOR_SERVER_CAPABILITIES` (daemon-supervisor.ts:196-200) is the
/// DEFAULT list plus `agent_roster` plus `direct_peer_transport`; the port filters SIX
/// names out (native_supervisor.rs:2709-2730), so a client that gates on a withheld
/// capability never even asks for a path the supervisor serves.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t09_capabilities_match_service() {
    let mut fixture = SupervisorFixture::new("t09-caps").await;
    let hello = fixture.hello();
    let advertised: Vec<String> = hello["serverCapabilities"]
        .as_array()
        .expect("serverCapabilities array")
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    let defaults: Vec<String> = daemon_protocol::DAEMON_DEFAULT_SERVER_CAPABILITIES
        .iter()
        .map(crate::modes::daemon::daemon_client::capability_name)
        .collect();

    // Every default the supervisor genuinely serves must still be advertised.
    let withheld_by_the_port = [
        "slim_attach",
        "history_ranges",
        "heartbeat_catalog",
        "authoritative_child_roster",
        "owned_session_recovery_context",
    ];
    for capability in &defaults {
        if withheld_by_the_port.contains(&capability.as_str()) {
            continue;
        }
        assert!(
            advertised.contains(capability),
            "the supervisor does not advertise {capability}, which it serves: {advertised:?}"
        );
    }
    // `get_rlm_children` is gated on `authoritative_child_roster`
    // (daemon_command_compatibility), and the supervisor answers it for a resident
    // worker; withholding it makes RLM children invisible to supervisor clients.
    assert!(
        advertised.contains(&"authoritative_child_roster".to_string()),
        "authoritative_child_roster must be advertised while get_rlm_children is served: {advertised:?}"
    );
    assert!(
        advertised.contains(&"heartbeat_catalog".to_string()),
        "heartbeat_catalog must be advertised while the generic forward serves heartbeats_list: {advertised:?}"
    );

    // C-05 (b): an unlisted command type must get the unknown-command error.
    let response = fixture
        .send(serde_json::json!({"type":"no_such_command","id":"unknown-1"}))
        .await;
    assert_eq!(
        response["success"], false,
        "an unlisted command was not refused: {response}"
    );
    let error = response["error"].as_str().unwrap_or("");
    assert!(
        error.contains("Unknown daemon command") && error.contains("no_such_command"),
        "the unknown-command contract must be reported verbatim, got: {error}"
    );

    // A bare body cannot claim an envelope protocol version.
    let response = fixture
        .send(serde_json::json!({
            "type":"get_session_tree",
            "id":"tree-old",
            "activeSessionId":"missing-session",
            "protocol":daemon_protocol::daemon_protocol_info()
        }))
        .await;
    let error = response["error"].as_str().unwrap_or("");
    assert!(
        error.contains("get_session_tree requires client protocol"),
        "a bare get_session_tree must be refused by the protocol gate, got: {error}"
    );

    let response = fixture
        .send(serde_json::json!({
            "type":"command",
            "id":"tree-current",
            "clientId":"parity-current-client",
            "protocol":daemon_protocol::daemon_protocol_info(),
            "command":{"type":"get_session_tree","activeSessionId":"missing-session"}
        }))
        .await;
    let error = response["error"].as_str().unwrap_or("");
    assert!(
        error.contains("Unknown active session") && !error.contains("requires client protocol"),
        "a current protocol envelope must reach session lookup, got: {response}"
    );
    assert_eq!(response["id"], "tree-current");
}




// ---------------------------------------------------------------------------
// C-08: id-less fan-outs must aggregate, not refuse
// ---------------------------------------------------------------------------

/// C-08: `agent_messages_pause` / `agent_messages_resume` (and `heartbeats_list`)
/// without an `activeSessionId` must fan out across the live workers, like TS
/// `daemon-supervisor.ts:2436-2448` / `:2481-2535`. The port refuses them outright
/// (native_supervisor.rs:2139-2141 / :2170-2172), so `optimus daemon agent-messages
/// pause` (no session arg) errors while the CLI and the TS supervisor both serve it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t09_idless_commands_have_defined_scope() {
    let mut fixture = SupervisorFixture::new("t09-idless").await;

    // No resident workers at all: the id-less fan-out is a no-op success
    // (`responses.find(success)?.data` is undefined, daemon-supervisor.ts:2448).
    for kind in ["agent_messages_pause", "agent_messages_resume"] {
        let response = fixture
            .send(serde_json::json!({"type": kind, "id": format!("{kind}-empty")}))
            .await;
        assert_ne!(
            response["success"], false,
            "the id-less {kind} fan-out must fan out (and no-op on an empty worker set), not refuse: {}",
            response["error"]
        );
    }
    let response = fixture
        .send(serde_json::json!({"type":"heartbeats_list","id":"heartbeats-empty"}))
        .await;
    assert_ne!(
        response["success"], false,
        "the id-less heartbeats_list must union the workers (or report the first worker error), not refuse: {}",
        response["error"]
    );

    // Two resident workers that are `ready` but have no live client: the TS answer is
    // the worker-state failure (`Cannot list heartbeats while session worker is
    // disconnected`, daemon-supervisor.ts:2511-2514), never an id-requirement refusal.
    add_descriptor_only_worker(&fixture, "worker-a", "active-a", "token-a", DAEMON_WORKER_LIFECYCLE_READY);
    add_descriptor_only_worker(&fixture, "worker-b", "active-b", "token-b", DAEMON_WORKER_LIFECYCLE_READY);
    for kind in ["agent_messages_pause", "agent_messages_resume"] {
        let response = fixture
            .send(serde_json::json!({"type": kind, "id": format!("{kind}-idless")}))
            .await;
        let error = response["error"].as_str().unwrap_or("");
        assert_ne!(
            response["success"], false,
            "the id-less {kind} fan-out must aggregate the live workers, not refuse: {error}"
        );
    }
    let response = fixture
        .send(serde_json::json!({"type":"heartbeats_list","id":"heartbeats-idless"}))
        .await;
    let error = response["error"].as_str().unwrap_or("");
    assert!(
        !error.contains("requires activeSessionId"),
        "the id-less heartbeats_list must not be refused for the missing id: {error}"
    );
    assert!(
        response["success"] == false && error.contains("Cannot list heartbeats while session worker is"),
        "a ready worker without a live client must produce the documented worker-state error, got: {response}"
    );
}

// ---------------------------------------------------------------------------
// C-07: a timed-out stop must still finish its cleanup
#[tokio::test]
async fn adoption_parks_dead_incomplete_create_without_poisoning_heartbeats() {
    let mut fixture = SupervisorFixture::new("heartbeat-dead-incomplete-create").await;
    let stale = add_descriptor_only_worker(&fixture, "dead-start", "dead-root", "test-token", DAEMON_WORKER_LIFECYCLE_STARTING);
    let descriptor = {
        let mut descriptor = stale.descriptor.lock().unwrap();
        descriptor.pid = i32::MAX;
        descriptor.process_start_id = None;
        descriptor.root_session_id = None;
        descriptor.clone()
    };
    fixture.supervisor.persist_worker(&descriptor).unwrap();
    fixture.supervisor.adopt_workers().await.unwrap();
    let adopted = fixture.supervisor.workers.lock().unwrap().get("dead-start").cloned().unwrap();
    assert_eq!(adopted.descriptor.lock().unwrap().lifecycle, DAEMON_WORKER_LIFECYCLE_FAILED);
    assert!(adopted.descriptor.lock().unwrap().last_error.as_deref().unwrap().contains("before creation completed"));
    let persisted: DaemonWorkerDescriptor = serde_json::from_slice(
        &std::fs::read(fixture.supervisor.descriptor_dir.join("dead-start.json")).unwrap()).unwrap();
    assert_eq!(persisted.lifecycle, DAEMON_WORKER_LIFECYCLE_FAILED);
    // Durable descriptors deliberately redact raw failures; keep that privacy boundary.
    assert_eq!(persisted.last_error.as_deref(), Some("Waiting for a client with fresh runtime context"));

    let healthy = add_descriptor_only_worker(&fixture, "healthy", "healthy-root", "test-token", DAEMON_WORKER_LIFECYCLE_READY);
    healthy.heartbeat_snapshot.lock().unwrap().store_if_current(0, vec![serde_json::json!({"job":{"id":"healthy-job"}})]);
    let response = fixture.send(serde_json::json!({"type":"heartbeats_list","id":"healthy-with-dead"})).await;
    assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["data"]["heartbeats"][0]["job"]["id"], "healthy-job");
}

#[tokio::test]
async fn starting_worker_keeps_heartbeat_coverage_incomplete_until_ready() {
    let mut fixture = SupervisorFixture::new("heartbeat-genuine-starting").await;
    let worker = add_descriptor_only_worker(&fixture, "starting", "starting-root", "test-token", DAEMON_WORKER_LIFECYCLE_STARTING);
    let response = fixture.send(serde_json::json!({"type":"heartbeats_list","id":"starting-check"})).await;
    assert_eq!(response["success"], false);
    assert!(response["error"].as_str().unwrap().contains("starting"));
    worker.descriptor.lock().unwrap().lifecycle = DAEMON_WORKER_LIFECYCLE_READY.into();
    worker.heartbeat_snapshot.lock().unwrap().store_if_current(0, Vec::new());
    let response = fixture.send(serde_json::json!({"type":"heartbeats_list","id":"ready-check"})).await;
    assert_eq!(response["success"], true, "{response}");
}

// ---------------------------------------------------------------------------

/// C-07: TS `scheduleWorkerStopFinalization` (daemon-supervisor.ts:6889-6996) keeps
/// escalating after a stop timeout and then finishes the interrupted cleanup. The port
/// returns `Err` from `stop_worker` and leaves the descriptor file behind forever
/// (native_supervisor.rs:2056-2059), so the next boot recovers an unreachable worker.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t09_timed_out_stop_finishes_cleanup() {
    let fixture = SupervisorFixture::new("t09-stop").await;
    // A test-owned live process that ignores the shutdown request on purpose: the
    // worker socket does not exist, so `stop_worker` takes its timeout path.
    let mut child = tokio::process::Command::new("cmd.exe")
        .args(["/c", "ping -n 60 127.0.0.1 > nul"])
        .stdout(std::process::Stdio::null())
        .spawn()
        .expect("spawn a test-owned unresponsive process");
    let child_pid = child.id().expect("child pid") as i32;

    let socket_path = fixture.supervisor.socket_path.clone();
    let descriptor_dir = fixture.supervisor.descriptor_dir.clone();
    let descriptor = DaemonWorkerDescriptor {
        version: 2,
        worker_id: "worker-wedged".to_string(),
        pid: child_pid,
        process_start_id: get_process_start_id(child_pid as i64),
        socket_path: worker_socket(&socket_path, "worker-wedged"),
        recovery_journal_path: Path::new(&descriptor_dir)
            .join("worker-wedged-journal.jsonl")
            .to_string_lossy()
            .into_owned(),
        orphan_process_journal_path: None,
        supervisor_socket_path: socket_path,
        authentication_token: "token-wedged".to_string(),
        worker_instance_id: None,
        root_active_session_id: "active-wedged".to_string(),
        owner_client_id: None,
        root_session_id: Some("active-wedged".to_string()),
        session_file: None,
        session_dir: None,
        telemetry_disabled: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        updated_at: chrono::Utc::now().to_rfc3339(),
        lifecycle: DAEMON_WORKER_LIFECYCLE_READY.to_string(),
        create_command: DurableDaemonCreateCommand {
            type_: "create".to_string(),
            session_path: None,
            no_session: Some(true),
            extra: Default::default(),
        },
        consecutive_failures: 0,
        stop_requested_at: None,
        archive_on_stop: None,
        last_failure_at: None,
        last_error: None,
    };
    fixture.supervisor.persist_worker(&descriptor).expect("persist wedged worker");
    let worker = Arc::new(Worker {
        descriptor: Mutex::new(descriptor.clone()),
        client: Mutex::new(None),
        roster_epoch: AtomicU64::new(0),
        roster_stale: AtomicBool::new(false),
        last_frame_at: Mutex::new(None),
        pending_client: Mutex::new(None),
        connection: AsyncMutex::new(()),
        stream: Mutex::new(CompactAssistantStreamReconstructor::new()),
        recovery: AtomicBool::new(false),
        deferred_recovery: AtomicBool::new(false),
        deferred_recovery_rounds: AtomicU64::new(0),
        promoted_owner_client_id: Mutex::new(None),
        heartbeat_snapshot: Mutex::new(HeartbeatSnapshot::default()),
    });
    fixture
        .supervisor
        .workers
        .lock()
        .unwrap()
        .insert("worker-wedged".to_string(), Arc::clone(&worker));

    let stop_result = fixture
        .supervisor
        .stop_worker(&worker, true, false)
        .await;
    let descriptor_file = Path::new(&descriptor_dir).join("worker-wedged.json");
    assert!(
        stop_result.is_err(),
        "a wedged worker must report a stop timeout, got: {stop_result:?}"
    );
    assert!(
        descriptor_file.exists(),
        "the tombstone is expected to survive the failed stop"
    );

    // The worker process finally exits (the operator or the OS ends it).
    let _ = child.kill().await;
    let _ = child.wait().await;

    // TS finishes the interrupted cleanup from the finalizer; the port has none.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while tokio::time::Instant::now() < deadline && descriptor_file.exists() {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        !descriptor_file.exists(),
        "the descriptor of a timed-out stop was left behind: the next boot would recover an          unreachable worker instead of finishing the interrupted cleanup          (daemon-supervisor.ts:6889-6996 scheduleWorkerStopFinalization)"
    );
}

// ---------------------------------------------------------------------------
// C-06: the update-restart commands must either serve the protocol or degrade
// through the documented CLI fallback.
// ---------------------------------------------------------------------------

/// C-06: `prepare_update_restart` (daemon-supervisor.ts:2421-2424, 6450-6505) and
/// `restart` (:2415-2417) are answered with
/// `"Daemon supervisor command is not implemented: <kind>"` (native_supervisor.rs:2362).
/// That text is NOT classified by `is_unknown_daemon_command_error`
/// (daemon-protocol.ts:1096-1098 demands `Unknown daemon command: <command>`), so the CLI
/// fallback at package-manager-cli.ts:1264-1277 never fires and the update aborts with
/// "Could not prepare daemon sessions for automatic resume" instead of degrading.
///
/// Contract proven here (TEST-SPEC T09 `prepare_restart_retains_resume_manifest`,
/// "if protocol is intentionally unsupported, prove the safe fallback contract and
/// label the limitation"): the supervisor writes NO resume manifest, so the prepare
/// answer must be a failure the CLI classifies as an unknown command, and `restart`
/// must serve its shutdown half (success + `daemon_closing{reason:"update"}`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t09_prepare_restart_retains_resume_manifest() {
    let mut fixture = SupervisorFixture::new("t09-update").await;

    let prepared = fixture
        .send(serde_json::json!({"type":"prepare_update_restart","id":"update-1"}))
        .await;
    assert_eq!(
        prepared["success"], false,
        "this supervisor does not implement the update manifest; the answer must therefore be a \
         classified failure, not a success: {prepared}"
    );
    let error = prepared["error"].as_str().unwrap_or("");
    assert!(
        daemon_protocol::is_unknown_daemon_command_error(error, "prepare_update_restart"),
        "the unimplemented-update error must be classified by is_unknown_daemon_command_error so \
         the CLI fallback runs (package-manager-cli.ts:1264-1277); today it is not: {error:?}"
    );
    // No manifest is written, so nothing may claim auto-resume: the documented limitation.
    let manifest_path = crate::config::get_daemon_update_restart_manifest_path(
        &fixture.supervisor.socket_path,
        Some(&fixture.root.join("agent").to_string_lossy()),
    );
    assert!(
        !Path::new(&manifest_path).exists(),
        "no resume manifest may be written when the prepare path degrades: {manifest_path}"
    );

    // `restart` needs no manifest: `setImmediate(() => void this.shutdown(0, false, true,
    // false, "update")); return success(...)` (daemon-supervisor.ts:2415-2417).
    let frames = fixture
        .send_collect(serde_json::json!({"type":"restart","id":"update-2"}))
        .await;
    let restarted = frames
        .iter()
        .find(|frame| frame["type"] == "response")
        .cloned()
        .unwrap_or(Value::Null);
    assert_ne!(
        restarted["success"], false,
        "restart must be served (daemon-supervisor.ts:2415-2417), got: {restarted}"
    );
    let closing_reason = frames
        .iter()
        .find(|frame| frame["type"] == "daemon_closing")
        .and_then(|frame| frame["reason"].as_str())
        .unwrap_or("")
        .to_string();
    assert_eq!(
        closing_reason, "update",
        "restart must broadcast daemon_closing with reason \"update\" before stopping \
         (daemon-supervisor.ts:2415-2417 + :7314-7318): {frames:?}"
    );
    // The supervisor stops serving so the workers can be replaced.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline && !fixture.supervisor.stopped.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        fixture.supervisor.stopped.is_cancelled(),
        "restart must stop the supervisor so the workers can be updated"
    );

    // A direct-transport request must also degrade by classification: no ticket is issued
    // and `direct_peer_transport` stays unadvertised, so the routed client keeps the
    // supervisor link (daemon-routed-client.ts:219-232).
    let transport = fixture
        .send(serde_json::json!({"type":"get_direct_worker_transport","id":"update-3","activeSessionId":"active-a"}))
        .await;
    assert_eq!(
        transport["success"], false,
        "no peer-transport ticket may be promised: {transport}"
    );
    let transport_error = transport["error"].as_str().unwrap_or("");
    assert!(
        daemon_protocol::is_unknown_daemon_command_error(transport_error, "get_direct_worker_transport"),
        "the direct-transport failure must classify as an unknown command so callers fall back \
         instead of aborting: {transport_error:?}"
    );
}

// ---------------------------------------------------------------------------
// C-04: the supervisor's saved-session list must stream rows and keep
// passivated ledger descendants
// ---------------------------------------------------------------------------

/// C-04: `list_saved_sessions` on the supervisor path calls `catalog.list(..., None)`
/// (native_supervisor.rs:2339-2360) instead of passing `onProgress`/`onSession`
/// (daemon-supervisor.ts:3003-3023), so `session_list_progress` / `session_list_item`
/// frames never reach the requesting client; and it never merges the spawn ledger
/// (`withPassiveRlmDescendantInfos`, :3025-3029), so a passivated RLM descendant's row
/// disappears from the saved-chat list even though the catalog scan cannot visit it.
///
/// The fixture replaces the catalog CHILD with a tiny scripted process that speaks the
/// real catalog protocol (ready -> progress*/session*/response) over the real
/// `DaemonCatalogClient`, so the production handlers run unmodified.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t09_saved_catalog_streams_and_preserves_descendants() {
    saved_catalog_streams_and_preserves_descendants(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t09_saved_catalog_uses_requested_session_directory_ledger() {
    saved_catalog_streams_and_preserves_descendants(true).await;
}

async fn saved_catalog_streams_and_preserves_descendants(non_default_dir: bool) {
    let fixture = SupervisorFixture::new(if non_default_dir { "t09-catalog-other-dir" } else { "t09-catalog" }).await;

    // A session that the catalog reports, and an RLM child that only the ledger knows.
    let requested_dir = fixture.root.join(if non_default_dir { "other-sessions" } else { "sessions" });
    let root_session = requested_dir.join("root.jsonl");
    let child_session = requested_dir.join("child.jsonl");
    std::fs::create_dir_all(root_session.parent().unwrap()).expect("sessions dir");
    std::fs::write(
        &root_session,
        format!(
            "{{\"type\":\"session\",\"id\":\"root\",\"version\":3,\"timestamp\":\"1970-01-01T00:00:00.000Z\",\"cwd\":{}}}\n",
            serde_json::to_string(&fixture.root.to_string_lossy()).unwrap()
        ),
    )
    .expect("root session");
    std::fs::write(
        &child_session,
        format!(
            "{{\"type\":\"session\",\"id\":\"child\",\"version\":3,\"timestamp\":\"1970-01-01T00:00:00.000Z\",\"cwd\":{}}}\n",
            serde_json::to_string(&fixture.root.to_string_lossy()).unwrap()
        ),
    )
    .expect("child session");
    let root_path = root_session.to_string_lossy().into_owned();
    let child_path = child_session.to_string_lossy().into_owned();

    // Scripted catalog child: one `progress` frame, one `session` frame, then the response.
    let script_dir = fixture.root.join("catalog");
    std::fs::create_dir_all(&script_dir).expect("script dir");
    let script = script_dir.join("fake-catalog.py");
    std::fs::write(
        &script,
        format!(
            "import json,sys\n\
             def send(v):\n\
             \x20   sys.stdout.write(json.dumps(v)+chr(10)); sys.stdout.flush()\n\
             send({{'type':'ready'}})\n\
             for line in sys.stdin:\n\
             \x20   try: req=json.loads(line)\n\
             \x20   except Exception: continue\n\
             \x20   if req.get('command')!='list': continue\n\
             \x20   rid=req.get('id')\n\
             \x20   send({{'type':'progress','id':rid,'loaded':1,'total':2}})\n\
             \x20   send({{'type':'session','id':rid,'session':{{'path':r'{root}','id':'root','cwd':r'{cwd}','created':'1970-01-01T00:00:00.000Z','modified':'1970-01-01T00:00:05.000Z','messageCount':1,'firstMessage':'root','allMessagesText':'root'}}}})\n\
             \x20   send({{'type':'response','id':rid,'success':True,'data':{{'sessions':[{{'path':r'{root}','id':'root','cwd':r'{cwd}','created':'1970-01-01T00:00:00.000Z','modified':'1970-01-01T00:00:05.000Z','messageCount':1,'firstMessage':'root','allMessagesText':'root'}}]}}}})\n",
            root = root_path.replace("\\", "\\\\"),
            cwd = fixture.root.to_string_lossy().replace("\\", "\\\\"),
        ),
    )
    .expect("fake catalog script");

    // Point the supervisor's catalog at the scripted child and start it.
    let executable = if cfg!(windows) { "python" } else { "python3" };
    fixture
        .supervisor
        .catalog
        .start(executable, vec![script.to_string_lossy().into_owned()], Vec::new())
        .await
        .expect("catalog child");

    // A ledger edge whose child transcript the catalog scan never reports.
    let ledger = RlmSpawnLedger::new(
        &fixture.root.join("agent").to_string_lossy(),
        &requested_dir.to_string_lossy(),
        None,
        None,
    );
    ledger
        .append_spawn(crate::modes::daemon::rlm_ledger::RlmSpawnInput {
            child_id: "child-1".to_string(),
            parent: root_path.clone(),
            child: child_path.clone(),
            depth: 1,
            name: "parity-child".to_string(),
        })
        .await
        .expect("ledger spawn");

    let mut fixture = fixture;
    let frames = fixture
        .send_collect(serde_json::json!({
            "type":"list_saved_sessions","id":"saved-1","cwd":fixture.root.to_string_lossy(),"scope":"all",
            "sessionDir":requested_dir.to_string_lossy()
        }))
        .await;

    let progress: Vec<&Value> = frames.iter().filter(|frame| frame["type"] == "session_list_progress").collect();
    let items: Vec<&Value> = frames.iter().filter(|frame| frame["type"] == "session_list_item").collect();
    assert!(
        !progress.is_empty(),
        "the requesting client must receive session_list_progress frames (daemon-supervisor.ts:3003-3013): {frames:?}"
    );
    assert!(
        !items.is_empty(),
        "the requesting client must receive session_list_item frames (daemon-supervisor.ts:3014-3021): {frames:?}"
    );
    assert_eq!(
        progress[0]["command"], "list_saved_sessions",
        "the progress frame carries its command: {progress:?}"
    );

    let response = frames
        .iter()
        .find(|frame| frame["type"] == "response")
        .cloned()
        .unwrap_or(Value::Null);
    assert_eq!(response["success"], true, "list_saved_sessions failed: {response}");
    let sessions = response["data"]["sessions"].as_array().cloned().unwrap_or_default();
    let ids: Vec<String> = sessions.iter().filter_map(|session| session["id"].as_str().map(str::to_string)).collect();
    assert!(
        ids.contains(&"root".to_string()),
        "the catalog row must be in the answer: {ids:?}"
    );
    // Windows canonicalisation prefixes a verbatim `\\?\`; compare paths loosely.
    let normalize = |value: &str| {
        value
            .replace('\\', "/")
            .trim_start_matches("//?/")
            .trim_start_matches("//.")
            .to_lowercase()
    };
    let child_row = sessions
        .iter()
        .find(|session| session["path"].as_str().is_some_and(|path| normalize(path) == normalize(&child_path)));
    assert!(
        child_row.is_some(),
        "the passivated RLM descendant must survive in the saved-session list \
         (daemon-supervisor.ts:3025-3029 withPassiveRlmDescendantInfos): {ids:?} / {sessions:?}"
    );
    assert_eq!(
        child_row.unwrap()["rlmDepth"].as_f64(),
        Some(1.0),
        "the merged row keeps its ledger depth: {sessions:?}"
    );
}


fn sidebar_saved_fixture(fixture: &SupervisorFixture, name: &str, child: bool) -> SessionSummary {
    let dir = fixture.root.join("sessions"); std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.jsonl"));
    std::fs::write(&path, format!("{}\n", json!({"type":"session","version":3,"id":name,
        "timestamp":"2026-09-24T00:00:00Z","cwd":fixture.root.join("workspace"),
        "parentSessionPath":child.then(|| dir.join("parent.jsonl")),"rlmDepth":if child {1} else {0}}))).unwrap();
    SessionSummary { id:name.into(), session_id:name.into(), session_file:Some(path.to_string_lossy().into_owned()),
        cwd:fixture.root.join("workspace").to_string_lossy().into_owned(), lifecycle:"live".into(), activity:"idle".into(),
        runtime_kind:Some(if child {"subagent"} else {"top-level"}.into()), ..Default::default() }
}

async fn sidebar_catalog_fixture(fixture: &SupervisorFixture, target: &str, ok: bool, ledger_path: &str) {
    // Only this explicit, synthetic file may be removed by the scripted catalog.
    let script = fixture.root.join("sidebar-catalog.py");
    std::fs::write(&script, r#"import json,sys,pathlib
allowed=pathlib.Path(sys.argv[1]).resolve()
ledger=pathlib.Path(sys.argv[2])
ok=sys.argv[3]=='true'
def send(value):
    print(json.dumps(value),flush=True)
send({'type':'ready'})
for line in sys.stdin:
    req=json.loads(line)
    if req['command']=='shutdown':
        send({'type':'response','id':req['id'],'success':True,'data':{}})
        break
    assert req['command']=='delete', req
    assert pathlib.Path(req['sessionPath']).resolve()==allowed, req
    records=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    tombstone=any(record.get('op')=='delete' for record in records)
    if ok: allowed.unlink()
    send({'type':'response','id':req['id'],'success':True,'data':{'ok':ok,'method':'unlink','error':'fixture refused','tombstoneSeen':tombstone,'sessionPath':req['sessionPath']}})
"#).unwrap();
    fixture.supervisor.catalog.start(if cfg!(windows) {"python"} else {"python3"}, vec![script.to_string_lossy().into_owned(), target.into(), ledger_path.into(), ok.to_string()], Vec::new()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sidebar_followup_idless_saved_delete_tombstones_catalog_and_removes_only_selected_roster() {
    let mut fixture = SupervisorFixture::new("sidebar-idless-delete").await;
    let target = sidebar_saved_fixture(&fixture, "selected-child", true);
    let other = sidebar_saved_fixture(&fixture, "unrelated", false);
    let marker = fixture.root.join("workspace/repo.txt"); std::fs::write(&marker, "repository unchanged").unwrap();
    let selected_entry = fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&target.roster_view()), None, None);
    let other_entry = fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&other.roster_view()), None, None);
    let ledger = fixture.supervisor.rlm_spawn_ledger().await.unwrap();
    ledger.append_spawn(crate::modes::daemon::rlm_ledger::RlmSpawnInput {
        child_id:"selected-child".into(), parent:fixture.root.join("sessions/parent.jsonl").to_string_lossy().into_owned(),
        child:target.session_file.clone().unwrap(), depth:1, name:"selected child".into(),
    }).await.unwrap();
    sidebar_catalog_fixture(&fixture, target.session_file.as_deref().unwrap(), true, ledger.ledger_path()).await;
    let response = fixture.send(json!({"type":"delete_saved_session","id":"selected-delete","sessionPath":target.session_file})).await;
    assert_eq!(response["success"], true, "{response}"); assert_eq!(response["data"]["ok"], true);
    assert_eq!(response["data"]["tombstoneSeen"], true, "ledger tombstone precedes catalog deletion");
    assert!(!Path::new(target.session_file.as_deref().unwrap()).exists());
    assert!(Path::new(other.session_file.as_deref().unwrap()).exists());
    assert_eq!(std::fs::read_to_string(marker).unwrap(), "repository unchanged");
    assert!(fixture.supervisor.roster().lock().unwrap().get(&selected_entry.agent_id).is_none());
    assert!(fixture.supervisor.roster().lock().unwrap().get(&other_entry.agent_id).is_some());
    assert!(ledger.edges(false).await.is_empty());
    fixture.supervisor.catalog.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sidebar_followup_saved_delete_refuses_active_foreign_and_uncertain_owners() {
    let mut fixture = SupervisorFixture::new("sidebar-delete-guards").await;
    let target = sidebar_saved_fixture(&fixture, "selected", false);
    let active = SessionSummary { active_session_id:Some("currently-active".into()), ..target.clone() };
    let entry = fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&active.roster_view()), None, None);
    let command = json!({"type":"delete_saved_session","id":"guarded","sessionPath":target.session_file});
    let response = fixture.send(command.clone()).await;
    assert_eq!(response["success"], false); assert!(response["error"].as_str().unwrap().contains("currently active"));
    fixture.supervisor.roster().lock().unwrap().delete(&entry.agent_id);
    let owner = add_descriptor_only_worker(&fixture, "private-owner", "private-active", "fixture-token", DAEMON_WORKER_LIFECYCLE_READY);
    owner.descriptor.lock().unwrap().session_file = target.session_file.clone();
    owner.descriptor.lock().unwrap().owner_client_id = Some("different-client".into());
    let response = fixture.send(command.clone()).await;
    assert_eq!(response["success"], false); assert!(response["error"].as_str().unwrap().contains("Unknown active session"));
    assert!(Path::new(target.session_file.as_deref().unwrap()).exists());
    // A conflicting descriptor must never turn an owned path into an unowned delete.
    owner.descriptor.lock().unwrap().owner_client_id = None;
    owner.descriptor.lock().unwrap().create_command.session_path = Some(fixture.root.join("different.jsonl").to_string_lossy().into_owned());
    let response = fixture.send(command).await;
    assert_eq!(response["success"], false); assert!(response["error"].as_str().unwrap().contains("registered worker"));
    assert!(Path::new(target.session_file.as_deref().unwrap()).exists());
    let missing = fixture.send(json!({"type":"delete_saved_session","id":"missing-path"})).await;
    assert!(missing["error"].as_str().unwrap().contains("sessionPath is required"));
    let explicit = fixture.send(json!({"type":"delete_saved_session","id":"bad-active","activeSessionId":"does-not-exist","sessionPath":target.session_file})).await;
    assert_eq!(explicit["success"], false); assert!(explicit["error"].as_str().unwrap().contains("Unknown active session"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sidebar_followup_failed_catalog_delete_keeps_saved_file_and_roster() {
    let mut fixture = SupervisorFixture::new("sidebar-delete-failure").await;
    let target = sidebar_saved_fixture(&fixture, "selected", false);
    let entry = fixture.supervisor.write_roster_entry(worker_roster_entry_from_summary(&target.roster_view()), None, None);
    let ledger = fixture.supervisor.rlm_spawn_ledger().await.unwrap();
    sidebar_catalog_fixture(&fixture, target.session_file.as_deref().unwrap(), false, ledger.ledger_path()).await;
    let response = fixture.send(json!({"type":"delete_saved_session","id":"refused","sessionPath":target.session_file})).await;
    assert_eq!(response["success"], true, "existing response contract returns data.ok=false");
    assert_eq!(response["data"]["ok"], false);
    assert!(Path::new(target.session_file.as_deref().unwrap()).exists());
    assert_eq!(fixture.supervisor.roster().lock().unwrap().get(&entry.agent_id), Some(entry));
    fixture.supervisor.catalog.stop().await;
}


async fn sidebar_owner_reply<S: AsyncRead + AsyncWrite + Unpin>(mut socket: S, captured: tokio::sync::oneshot::Sender<Value>) {
    use crate::modes::daemon::daemon_worker_client::{encode_private_frame, PrivateFrameDecoder};
    let mut decoder = PrivateFrameDecoder::new(); let mut buffer = [0; 8192];
    loop {
        let count = socket.read(&mut buffer).await.unwrap(); if count == 0 { return; }
        if let Some(frame) = decoder.push(&buffer[..count]).unwrap().into_iter().next() {
            let command: Value = serde_json::from_slice(&frame.payload).unwrap();
            let reply = json!({"type":"response","id":command["id"],"command":command["type"],"success":true,"data":{"ok":true,"ownerHandled":true}});
            let frame = encode_private_frame(&json!({"kind":"outbound","outboundType":"response","requestId":command["id"],"payloadEncoding":"jsonl"}), reply.to_string().as_bytes()).unwrap();
            socket.write_all(&frame).await.unwrap(); let _ = captured.send(command);
            // Keep the peer alive until the test explicitly drops it.
            let _ = socket.read(&mut buffer).await;
            return;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sidebar_followup_saved_delete_forwards_live_owner_then_reclaims_only_dead_failed_owner() {
    let mut fixture = SupervisorFixture::new("sidebar-delete-owner-route").await;
    let target = sidebar_saved_fixture(&fixture, "selected", false);
    let worker = add_descriptor_only_worker(&fixture, "owner", "owner-active", "synthetic-token", DAEMON_WORKER_LIFECYCLE_READY);
    worker.descriptor.lock().unwrap().session_file = target.session_file.clone();
    let (send, received) = tokio::sync::oneshot::channel();
    #[cfg(windows)]
    let (socket, task) = {
        let socket = format!(r"\\.\pipe\optimus-sidebar-owner-{}", uuid::Uuid::new_v4());
        let server = tokio::net::windows::named_pipe::ServerOptions::new().first_pipe_instance(true).create(&socket).unwrap();
        let task = tokio::spawn(async move { server.connect().await.unwrap(); sidebar_owner_reply(server, send).await; });
        (socket, task)
    };
    #[cfg(unix)]
    let (socket, task) = {
        let socket = fixture.root.join("owner.sock").to_string_lossy().into_owned();
        let server = tokio::net::UnixListener::bind(&socket).unwrap();
        let task = tokio::spawn(async move { let (stream, _) = server.accept().await.unwrap(); sidebar_owner_reply(stream, send).await; });
        (socket, task)
    };
    let client = Arc::new(DaemonWorkerClient::new(&socket)); client.connect(1000).await.unwrap();
    worker.descriptor.lock().unwrap().socket_path = socket; *worker.client.lock().unwrap() = Some(client.clone());
    let command = json!({"type":"delete_saved_session","id":"owner-forward","sessionPath":target.session_file});
    let response = fixture.send(command.clone()).await;
    assert_eq!(response["id"], "owner-forward"); assert_eq!(response["data"]["ownerHandled"], true, "{response}");
    assert_eq!(received.await.unwrap()["sessionPath"], target.session_file.as_deref().unwrap());
    assert!(Path::new(target.session_file.as_deref().unwrap()).exists(), "supervisor did not bypass live owner to delete itself");
    client.close_now(); task.abort(); *worker.client.lock().unwrap() = None;
    let response = fixture.send(command.clone()).await;
    assert_eq!(response["success"], false); assert!(response["error"].as_str().unwrap().contains("retry the delete"));
    {
        let mut descriptor = worker.descriptor.lock().unwrap();
        descriptor.lifecycle = DAEMON_WORKER_LIFECYCLE_FAILED.into(); descriptor.pid = i32::MAX; descriptor.process_start_id = None;
    }
    let ledger = fixture.supervisor.rlm_spawn_ledger().await.unwrap();
    sidebar_catalog_fixture(&fixture, target.session_file.as_deref().unwrap(), true, ledger.ledger_path()).await;
    let response = fixture.send(command).await;
    assert_eq!(response["success"], true, "{response}"); assert_eq!(response["data"]["ok"], true);
    assert!(!fixture.supervisor.workers.lock().unwrap().contains_key("owner"));
    assert!(!Path::new(target.session_file.as_deref().unwrap()).exists());
    fixture.supervisor.catalog.stop().await;
}
