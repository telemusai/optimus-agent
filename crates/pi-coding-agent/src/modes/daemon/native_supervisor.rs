//! Native transport/process adapter for daemon-supervisor.ts.
//!
//! Each resident root runs in a separately authenticated worker process. Public
//! clients share the supervisor connection, never the worker's secret or socket.
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio_util::sync::CancellationToken;
use crate::core::agent_session_config::{AgentSessionRuntimeConfig, durable_agent_session_runtime_config, merge_agent_session_runtime_config};
use crate::core::session_lease::{canonical_session_path, get_process_start_id};
use crate::utils::child_process::is_process_alive;
use crate::core::session_resolver::looks_like_session_path;
use crate::utils::atomic_file::{write_file_atomic_sync, remove_file_durably, RemoveFileDurablyOptions, WriteFileAtomicOptions};
use crate::utils::child_process::{signal_process_group_or_process, Signal};
use super::super::active_session_state::create_active_session_id;
use super::super::command_recovery_journal::{create_command_idempotency_key, CommandRecoveryJournal, CommandJournalBeginResult};
use crate::modes::daemon::agent_message_delivery_journal::{
    delivery_reason_code, AgentMessageDeliveryJournal, AgentMessageDeliveryOutcome,
    AgentMessageDeliveryRecord, AGENT_MESSAGE_DELIVERY_JOURNAL_FILE,
};
use super::super::compact_session_stream::{CompactAssistantStreamReconstructor, CompactAssistantDelta};
use super::super::daemon_catalog_process::{CatalogListCallbacks, DaemonCatalogClient, DAEMON_CATALOG_ROLE_ENV};
use super::super::agent_roster::{
    agent_roster_entry_to_value, classify_session_roster_status, is_session_summary_busy, AgentRosterStatus,
    passivated_worker_roster_entry, roster_agent_id_for_entry,
    worker_roster_entry_from_summary, AgentRoster, AgentRosterEntry, AgentRosterMutation,
    RegisteredHeartbeatFlags, RosterEntryMarks, RosterSessionSummary, RosterSummaryView, WorkerRosterEntry,
};
use super::super::daemon_client::DaemonClientRequestOptions;
use super::super::daemon_errors::DaemonSessionRecoveringError;
use super::super::daemon_protocol::{self, DaemonResponse};
use super::super::daemon_session_id::matches_session_id_suffix;
use super::super::daemon_session_list::{summary_for_inactive_session, SessionSummary};
use crate::core::session_manager::SessionInfo;
use crate::core::session_action_store::{can_evict_worker, IdleEvictionMinutes, SessionEvictionSnapshot, WorkerEvictionSnapshot, WorkerLifecycle};
use crate::core::agent_messages::{
    assert_agent_session_name_available, format_agent_session_name_unavailable,
    session_name_reservation_key, AgentFamilyCatalogEntry, AgentSessionNameAvailabilityInput,
    AgentSessionNameScope, AgentSessionMessageAgentSummary,
};
use super::super::rlm_ledger::{create_rlm_ledger_registry_seed_source, RlmLedgerEdge, RlmSpawnLedger};
use super::super::daemon_socket::*;
use super::super::daemon_supervisor_ownership::*;
use super::super::daemon_worker_client::{DaemonWorkerClient, PrivateFrame};
use super::super::daemon_worker_protocol::*;
use super::super::saved_session_info::serialize_saved_session_info;

#[path = "supervisor_maintenance.rs"]
mod supervisor_maintenance;
use supervisor_maintenance::HeartbeatSnapshot;

#[cfg(test)]
#[path = "daemon_supervisor_parity_tests.rs"]
mod daemon_supervisor_parity_tests;

#[cfg(test)]
#[path = "supervisor_messaging_safety_tests.rs"]
mod supervisor_messaging_safety_tests;

#[cfg(test)]
#[path = "supervisor_maintenance_tests.rs"]
mod supervisor_maintenance_tests;

#[cfg(test)]
#[path = "supervisor_core_backlog_tests.rs"]
mod supervisor_core_backlog_tests;

#[cfg(all(test, windows))]
#[path = "worker_stop_safety_tests.rs"]
mod worker_stop_safety_tests;

const REQUEST_TIMEOUT: u64 = 24 * 60 * 60 * 1000;
const STOP_CLEANUP_MAX_ATTEMPTS: usize = 3;
const MAX_PUBLIC_LINE: usize = super::super::daemon_client::DAEMON_MAX_LINE_LENGTH;

/// `ROSTER_WATCHDOG_INTERVAL_MS` / `ROSTER_STALE_AFTER_MS` (daemon-supervisor.ts:194-195).
const ROSTER_WATCHDOG_INTERVAL_MS: u64 = 15_000;
/// Maximum rate for the aged repeat of the command-journal pending report
/// (audit D-07): a backlog that persists unchanged is re-reported at most
/// this often, so a never-resolved command cannot hide behind "no change".
const PENDING_COMMAND_JOURNAL_REPORT_INTERVAL_MS: u64 = 30 * 60 * 1000;
const ROSTER_STALE_AFTER_MS: u64 = 3 * ROSTER_HEARTBEAT_INTERVAL_MS;

/// `SCHEDULED_WAKE_RETRY_MS` / `SCHEDULED_WAKE_MAX_TIMEOUT_MS` / `SCHEDULED_WAKE_CLIENT_ID`
/// (daemon-supervisor.ts:198-199 and daemon_supervisor.rs:140-142, which owns the values).
const SCHEDULED_WAKE_RETRY_MS: f64 = 60_000.0;
const SCHEDULED_WAKE_MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;
const SCHEDULED_WAKE_CLIENT_ID: &str = "scheduled-wake";

/// `const WORKER_RETRY_DELAYS_MS = [250, 1000, 5000] as const` (daemon-supervisor.ts:210).
/// daemon_supervisor.rs also declares this ladder, but keeps it private; the close-time
/// recovery ladder in this file needs its own copy.
const WORKER_RETRY_DELAYS_MS: [u64; 3] = [250, 1000, 5000];

struct PublicClient {
    connection_id: String,
    id: Mutex<String>,
    protocol_id: Mutex<Option<String>>,
    subscriptions: Mutex<HashSet<String>>,
    supports_extension_ui: AtomicBool,
    pause_epoch: AtomicU64,
    output: mpsc::Sender<Vec<u8>>,
    stopped: CancellationToken,
    /// `client.rosterSubscribed` / `client.rosterResyncPending`.
    roster_subscribed: AtomicBool,
    roster_resync_pending: AtomicBool,
    /// `client.backpressured`: a resync is deferred while the writer is behind.
    backpressured: AtomicBool,
}
impl PublicClient {
    fn write(&self, value: &Value) -> bool {
        let Ok(mut bytes) = serde_json::to_vec(value) else { return false; };
        bytes.push(b'\n');
        match self.output.try_send(bytes) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => { self.backpressured.store(true, Ordering::SeqCst); false }
            Err(mpsc::error::TrySendError::Closed(_)) => { self.stopped.cancel(); false }
        }
    }
    fn identity(&self) -> String { self.protocol_id.lock().unwrap().clone().unwrap_or_else(|| self.id.lock().unwrap().clone()) }
}
struct Worker {
    descriptor: Mutex<DaemonWorkerDescriptor>,
    /// `ResidentWorker.client`: replaced by a reconnection, so it is not `Arc`-pinned.
    client: Mutex<Option<Arc<DaemonWorkerClient>>>,
    /// Bumped per applied roster frame; a summaries pull that straddles one must not gap-fill.
    roster_epoch: AtomicU64,
    /// `worker.rosterStale`: the watchdog marks rows whose worker stopped talking.
    roster_stale: AtomicBool,
    last_frame_at: Mutex<Option<u64>>,
    /// In-flight reconnection; an allowed frame source alongside `client`.
    pending_client: Mutex<Option<Arc<DaemonWorkerClient>>>,
    connection: AsyncMutex<()>,
    stream: Mutex<CompactAssistantStreamReconstructor>,
    /// `worker.recovery`: one recovery ladder per worker (daemon-supervisor.ts:3912, 4226-4228).
    recovery: AtomicBool,
    /// `worker.deferredRecovery` / `worker.deferredRecoveryRounds` (daemon-supervisor.ts:3960-3966).
    deferred_recovery: AtomicBool,
    deferred_recovery_rounds: AtomicU64,
    /// `worker.promotedOwnerClientId` (daemon-supervisor.ts:3243, 3266): set once the promotion
    /// of this registration completed, so a repeated `promoteOwnedWorker` is a no-op (:3244).
    promoted_owner_client_id: Mutex<Option<String>>,
    heartbeat_snapshot: Mutex<HeartbeatSnapshot>,
}
#[derive(Clone)]
struct InputPause { connection_id: String, worker: Arc<Worker>, active: String, requested: String }
/// `interface SupervisorPromptAdmission` (daemon-supervisor.ts:438-447) with
/// `status: "waiting" | "owned" | "cancelled"` (:443).
#[derive(Clone, Copy, PartialEq, Eq)]
enum PromptAdmissionStatus { Waiting, Owned, Cancelled }
/// `this.promptAdmissions` (daemon-supervisor.ts:749) is a map per client; the port keys one
/// flat map by `promptAdmissionKey(client, activeSessionId, admissionId)`, which is the TS outer
/// key (`DaemonSocketClient`) plus the TS inner key `\`${activeSessionId}\0${publicAdmissionId}\``
/// (:1822-1823). Record field names mirror the TS interface (:438-447).
#[derive(Clone)]
struct PromptAdmissionRecord {
    /// `admission.client`.
    connection_id: String,
    /// `admission.activeSessionId`.
    active_session_id: String,
    /// `admission.publicAdmissionId`.
    public_admission_id: String,
    /// `admission.workerAdmissionId: \`supervisor-admission:${randomUUID()}\`` (:1914).
    worker_admission_id: String,
    /// `admission.status`.
    status: PromptAdmissionStatus,
    /// `admission.controller` (an `AbortController`; `CancellationToken` per PORT-RULES.md).
    controller: CancellationToken,
    /// `admission.worker` (:445).
    worker: Option<Arc<Worker>>,
    /// `admission.workerActiveSessionId` (:446).
    worker_active_session_id: Option<String>,
}
/// `promptAdmissionKey(activeSessionId, publicAdmissionId)` (daemon-supervisor.ts:1822-1823)
/// scoped by the owning socket, which is the TS map's outer key (:749).
fn prompt_admission_key(connection_id: &str, active_session_id: &str, public_admission_id: &str) -> String {
    format!("{connection_id}\u{0}{active_session_id}\u{0}{public_admission_id}")
}
/// One in-flight worker open, the analogue of a `this.openingWorkers` entry
/// (daemon-supervisor.ts:747, 3101). A joiner awaits `done` and reads the same outcome the
/// original caller received, which is what `await pending` does at :3153.
struct OpeningWorker { done: CancellationToken, result: Mutex<Option<Result<Value, String>>> }
struct OpeningWorkerGuard { supervisor: Arc<Supervisor>, key: String, opening: Arc<OpeningWorker> }
impl Drop for OpeningWorkerGuard {
    fn drop(&mut self) {
        let mut openings = self.supervisor.opening_workers.lock().unwrap();
        if openings.get(&self.key).is_some_and(|current| Arc::ptr_eq(current, &self.opening)) {
            openings.remove(&self.key);
        }
        self.opening.done.cancel();
    }
}
/// The `finally { if (admission) this.deletePromptAdmission(admission); }` (daemon-supervisor.ts:2842)
/// of `handleLine`: every exit path of a prompt command removes its registration.
///
/// TS re-checks object identity in `deletePromptAdmission` (:1846); the port cannot, because a
/// duplicate `(activeSessionId, admissionId)` is rejected while the first is registered (:1907),
/// so the key can only ever hold the registration this guard was created for.
struct PromptAdmissionGuard { supervisor: Arc<Supervisor>, admission: PromptAdmissionRecord }
impl Drop for PromptAdmissionGuard {
    fn drop(&mut self) { self.supervisor.delete_prompt_admission(&self.admission); }
}
struct Supervisor {
    // Lock order: admission read/write fence, then `opening`. A sweep drains
    // public commands and excludes recovery/opening before its final decision.
    eviction_fence: tokio::sync::RwLock<()>,
    idle_eviction_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    socket_path: String,
    descriptor_dir: PathBuf,
    config: AgentSessionRuntimeConfig,
    ownership: DaemonSupervisorOwnership,
    workers: Mutex<HashMap<String, Arc<Worker>>>,
    clients: Mutex<HashMap<String, Arc<PublicClient>>>,
    /// Opens share this fence; eviction takes it exclusively. Per-session joins below
    /// prevent duplicate creates/retries without serializing unrelated workers.
    opening: tokio::sync::RwLock<()>,
    /// `this.openingWorkers` (daemon-supervisor.ts:747) keyed as at :3060-3062: the canonical
    /// session path, or `new:<createCommandIdempotencyKey(clientId, command.id)>`. A second
    /// identical create joins the in-flight one instead of double-launching a worker (:3063-3066).
    opening_workers: Mutex<HashMap<String, Arc<OpeningWorker>>>,
    pauses: Mutex<HashMap<String, InputPause>>,
    journal: Mutex<CommandRecoveryJournal>,
    /// D-04: bounded, content-free delivery telemetry for cross-worker agent messages.
    /// Telemetry only: it never changes delivery semantics and never replays a send.
    agent_message_delivery_journal: Mutex<AgentMessageDeliveryJournal>,
    catalog: Arc<DaemonCatalogClient>,
    stopped: CancellationToken,
    /// `this.rosterStore`: the one supervisor-owned roster, lazily created.
    roster: Mutex<Option<Arc<Mutex<AgentRoster>>>>,
    /// `pendingRosterChanged` / `pendingRosterRemoved` / `publishedRosterIds`.
    pending_roster_changed: Mutex<HashSet<String>>,
    pending_roster_removed: Mutex<HashSet<String>>,
    published_roster_ids: Mutex<HashSet<String>>,
    roster_push_scheduled: AtomicBool,
    /// `this.rlmSpawnLedgerInstance`, a process-wide OnceLock so every clone shares one.
    ledger: Arc<tokio::sync::OnceCell<Arc<RlmSpawnLedger>>>,
    /// `this.scheduledWakeTimer` / `this.scheduledWakeRecompute` / `scheduledWakeRecomputeQueued`
    /// (daemon-supervisor.ts:775-778, 942-1055).
    scheduled_wake_timer: Mutex<Option<tokio::task::JoinHandle<()>>>,
    scheduled_wake_recompute: AtomicBool,
    scheduled_wake_recompute_queued: AtomicBool,
    /// `this.scheduledWakeFailures`: the failure floor per passive root.
    scheduled_wake_failures: Mutex<HashMap<String, f64>>,
    /// `this.promptAdmissions` (daemon-supervisor.ts:749), keyed by
    /// `prompt_admission_key(connection, activeSessionId, admissionId)`.
    prompt_admissions: Mutex<HashMap<String, PromptAdmissionRecord>>,
    /// `this.pendingSessionNames` (daemon-supervisor.ts:4545-4552): reservation keys held while a
    /// create with a name is in flight, so two concurrent creates cannot both pass the check.
    pending_session_names: Mutex<HashSet<String>>,
    /// Rate-limit state for the bounded command-journal pending report (audit D-07).
    pending_command_journal_log: Mutex<Option<(usize, u64)>>,
}

pub(crate) async fn run_daemon_supervisor_mode(socket_path: Option<String>, mut config: AgentSessionRuntimeConfig) -> Result<(), String> {
    let socket_path = normalize_socket_path_for_daemon(&socket_path.unwrap_or_else(default_daemon_socket_path), None);
    let agent_dir = config.agent_dir.clone().ok_or("Daemon supervisor config is missing agentDir")?;
    let descriptor_dir = Path::new(&agent_dir).join("daemon-workers").join(descriptor_key(&socket_path));
    if let Ok(bytes) = std::fs::read(descriptor_dir.join("supervisor-config")) {
        let persisted: Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        if persisted.get("version").and_then(Value::as_u64) == Some(1) && persisted.get("socketPath").and_then(Value::as_str) == Some(&socket_path) {
            let durable = serde_json::from_value::<crate::core::agent_session_config::DurableAgentSessionRuntimeConfig>(persisted.get("defaultSessionConfig").cloned().unwrap_or(Value::Null)).map_err(|error| error.to_string())?;
            let previous: AgentSessionRuntimeConfig = serde_json::from_value(serde_json::to_value(durable).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
            config = merge_agent_session_runtime_config(&config, Some(&previous));
        }
    }
    let lease = acquire_daemon_socket_path_lease(&socket_path).await;
    #[cfg(unix)]
    if lease.is_none() { return Err(format!("Could not acquire daemon socket lease: {socket_path}")); }
    let ownership = match async {
        wait_for_daemon_startup_fence(&socket_path, 120_000, None).await?;
        acquire_daemon_supervisor_ownership(AcquireDaemonSupervisorOwnershipOptions {
            socket_path: socket_path.clone(), descriptor_dir: descriptor_dir.to_string_lossy().into_owned(),
            agent_dir: agent_dir.clone(), generation: uuid::Uuid::new_v4().to_string(), app_version: crate::config::VERSION.to_string(), registry_dir: None,
        }).await
    }.await {
        Ok(owner) => owner,
        Err(error) => { if let Some(lease) = lease { lease.release().await; } return Err(error); }
    };
    let journal = match CommandRecoveryJournal::new(&descriptor_dir.join("command-journal.jsonl").to_string_lossy()) {
        Ok(journal) => journal,
        Err(error) => { let _ = ownership.release().await; if let Some(lease) = lease { lease.release().await; } return Err(error); }
    };
    let delivery_journal = AgentMessageDeliveryJournal::new(
        &descriptor_dir.join(AGENT_MESSAGE_DELIVERY_JOURNAL_FILE).to_string_lossy(),
    );
    let supervisor = Arc::new(Supervisor {
        eviction_fence: tokio::sync::RwLock::new(()), idle_eviction_task: Mutex::new(None),
        socket_path: socket_path.clone(), journal: Mutex::new(journal),
        agent_message_delivery_journal: Mutex::new(delivery_journal),
        descriptor_dir, config, ownership, workers: Mutex::new(HashMap::new()), clients: Mutex::new(HashMap::new()), opening: tokio::sync::RwLock::new(()), pauses: Mutex::new(HashMap::new()),
        catalog: Arc::new(DaemonCatalogClient::new(Arc::new(|message| eprintln!("Daemon catalog: {message}")))), stopped: CancellationToken::new(),
        roster: Mutex::new(None), pending_roster_changed: Mutex::new(HashSet::new()), pending_roster_removed: Mutex::new(HashSet::new()),
        published_roster_ids: Mutex::new(HashSet::new()), roster_push_scheduled: AtomicBool::new(false),
        ledger: Arc::new(tokio::sync::OnceCell::new()),
        scheduled_wake_timer: Mutex::new(None), scheduled_wake_recompute: AtomicBool::new(false),
        scheduled_wake_recompute_queued: AtomicBool::new(false), scheduled_wake_failures: Mutex::new(HashMap::new()),
        prompt_admissions: Mutex::new(HashMap::new()), opening_workers: Mutex::new(HashMap::new()),
        pending_session_names: Mutex::new(HashSet::new()),
        pending_command_journal_log: Mutex::new(None),
    });
    // The TS store is installed lazily by `roster()`; the port installs it here
    // because its mutation sink needs a `Weak` to the finished `Arc`.
    supervisor.init_roster();
    let mut identity = None;
    let result = async {
        prepare_daemon_socket_path(&socket_path, lease.clone()).await.map_err(|error| error.to_string())?;
        std::fs::create_dir_all(&supervisor.descriptor_dir).map_err(|error| error.to_string())?;
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&supervisor.descriptor_dir, std::fs::Permissions::from_mode(0o700)).map_err(|error| error.to_string())?;
        }
        persist_json(&supervisor.descriptor_dir.join("supervisor-config"), &json!({
            "version": 1, "socketPath": socket_path, "defaultSessionConfig": durable_agent_session_runtime_config(&supervisor.config),
        }))?;
        let executable = std::env::current_exe().map_err(|error| error.to_string())?;
        let mut environment: Vec<(String, String)> = std::env::vars().collect();
        environment.retain(|(key, _)| key != DAEMON_WORKER_ROLE_ENV && key != DAEMON_WORKER_TOKEN_ENV);
        environment.push((DAEMON_CATALOG_ROLE_ENV.to_string(), "1".to_string()));
        supervisor.catalog.start(&executable.to_string_lossy(), vec![], environment).await?;
        // Bounded dead-owner lease cleanup at supervisor start (audit D-08): a lease
        // whose recorded owner process is verifiably dead is reclaimed. Live owners
        // are never touched and nothing here kills a process.
        let lease_sweep = crate::core::session_lease::sweep_dead_owner_leases(&agent_dir);
        eprintln!(
            "[{}] Session lease sweep: scanned {}, reclaimed {} dead-owner lease(s), {} unreadable owner(s) left for manual review",
            iso_from_ms(supervisor_now_ms() as f64), lease_sweep.scanned, lease_sweep.reclaimed.len(), lease_sweep.unreadable_owners
        );
        supervisor.adopt_workers().await?;
        supervisor.seed_roster_ledger().await;
        supervisor.start_roster_watchdog();
        supervisor.start_idle_eviction();
        // Startup view of the command-recovery journal backlog (audit D-07).
        supervisor.report_pending_command_journal();
        // `this.scheduleScheduledSessionWakeRecompute()` on startup (daemon-supervisor.ts:883).
        supervisor.schedule_scheduled_session_wake_recompute();
        #[cfg(unix)]
        let listener = tokio::net::UnixListener::bind(&socket_path).map_err(|error| error.to_string())?;
        #[cfg(windows)]
        let mut listener = tokio::net::windows::named_pipe::ServerOptions::new().first_pipe_instance(true).create(&socket_path).map_err(|error| error.to_string())?;
        identity = get_daemon_socket_identity(&socket_path);
        restrict_daemon_socket_path(&socket_path);
        supervisor.ownership.update_phase("owner").await?;
        supervisor.register_signals();
        eprintln!(
            "[{}] Prime Agent daemon supervisor {} listening on {}",
            iso_from_ms(supervisor_now_ms() as f64),
            supervisor.ownership.snapshot().generation,
            socket_path
        );
        loop {
            #[cfg(unix)] {
                let accepted = tokio::select! { _ = supervisor.stopped.cancelled() => break, result = listener.accept() => result };
                let (stream, _) = accepted.map_err(|error| error.to_string())?;
                spawn_connection(supervisor.clone(), stream);
            }
            #[cfg(windows)] {
                tokio::select! { _ = supervisor.stopped.cancelled() => break, result = listener.connect() => result.map_err(|error| error.to_string())? };
                let next = tokio::net::windows::named_pipe::ServerOptions::new().create(&socket_path).map_err(|error| error.to_string())?;
                spawn_connection(supervisor.clone(), listener);
                listener = next;
            }
        }
        Ok(())
    }.await;
    supervisor.stopped.cancel();
    let idle_task = supervisor.idle_eviction_task.lock().unwrap().take();
    if let Some(task) = idle_task { let _ = task.await; }
    for client in supervisor.clients.lock().unwrap().values() { client.stopped.cancel(); }
    let workers: Vec<_> = supervisor.workers.lock().unwrap().values().cloned().collect();
    for worker in workers {
        let client = { worker.client.lock().unwrap().clone() };
        if let Some(client) = client { client.close().await; }
    }
    supervisor.catalog.stop().await;
    if identity.is_some() { cleanup_daemon_socket_path(&socket_path, identity, lease.as_deref()); }
    let release = supervisor.ownership.release().await;
    if let Some(lease) = lease { lease.release().await; }
    result.and(release)
}

impl Supervisor {
    fn register_signals(self: &Arc<Self>) {
        #[cfg(unix)]
        for kind in [tokio::signal::unix::SignalKind::interrupt(), tokio::signal::unix::SignalKind::terminate(), tokio::signal::unix::SignalKind::hangup()] {
            if let Ok(mut signal) = tokio::signal::unix::signal(kind) {
                let stopped = self.stopped.clone();
                tokio::spawn(async move { tokio::select! { _ = stopped.cancelled() => {}, _ = signal.recv() => stopped.cancel() } });
            }
        }
        #[cfg(windows)] {
            let stopped = self.stopped.clone();
            tokio::spawn(async move { tokio::select! { _ = stopped.cancelled() => {}, _ = tokio::signal::ctrl_c() => stopped.cancel() } });
        }
    }
    fn hello(&self, client: &PublicClient) -> Value {
        let owner = self.ownership.snapshot();
        json!({"type":"daemon_hello", "socketPath":self.socket_path,
            "protocol":daemon_protocol::daemon_protocol_info(), "schemaId":daemon_protocol::DAEMON_SCHEMA_ID,
            "schemaRevision":daemon_protocol::DAEMON_SCHEMA_REVISION, "appVersion":crate::config::VERSION,
            "runtime":super::super::daemon_runtime_identity::get_daemon_runtime_identity_from_process(),
            "supervisorGeneration":owner.generation, "supervisorOwnerToken":owner.token, "supervisorPid":owner.pid,
            "supervisorProcessStartId":owner.process_start_id, "supervisorSocketPath":owner.socket_path,
            "clientId":client.identity(), "serverCapabilities":server_capabilities()})
    }
    fn persist_worker(&self, descriptor: &DaemonWorkerDescriptor) -> Result<(), String> {
        persist_json(&self.descriptor_dir.join(format!("{}.json", descriptor.worker_id)), &serde_json::to_value(durable_daemon_worker_descriptor(descriptor)).map_err(|error| error.to_string())?)
    }
    async fn authenticate(&self, client: &Arc<DaemonWorkerClient>, descriptor: &DaemonWorkerDescriptor) -> Result<(), String> {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(super::WORKER_CONNECT_TIMEOUT_MS);
        loop {
            self.ownership.assert_current().await.map_err(|error| error.to_string())?;
            match client.connect(super::WORKER_CONNECT_PROBE_MS).await {
                Ok(()) => break,
                Err(error) if tokio::time::Instant::now() >= deadline => return Err(error.to_string()),
                Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
            }
        }
        client.wait_for_hello(super::WORKER_CONNECT_TIMEOUT_MS).await.map_err(|error| error.to_string())?;
        let owner = self.ownership.snapshot();
        let mut claim = vec![("supervisorGeneration", json!(owner.generation)), ("supervisorPid", json!(owner.pid)), ("supervisorSocketPath", json!(owner.socket_path))];
        if let Some(start) = owner.process_start_id { claim.push(("supervisorProcessStartId", json!(start))); }
        let response = client.authenticate_worker(&descriptor.authentication_token, &claim, super::WORKER_CONNECT_TIMEOUT_MS).await.map_err(|error| error.to_string())?;
        // `workerAuthAdvertisesRoster`: a worker that predates the roster protocol
        // must be restarted, never silently treated as roster-resident.
        if !worker_auth_advertises_roster(response.data.as_ref()) {
            return Err(format!("Session worker {} predates the roster protocol and must be restarted", descriptor.worker_id));
        }
        Ok(())
    }
    /// `installWorker(worker)` plus `connectWorker`'s listeners, which are
    /// registered BEFORE authentication so the worker's first roster snapshot
    /// cannot race them (daemon-supervisor.ts:3610).
    fn install_worker(self: &Arc<Self>, descriptor: DaemonWorkerDescriptor, client: Arc<DaemonWorkerClient>) -> Arc<Worker> {
        let worker = Arc::new(Worker {
            descriptor: Mutex::new(descriptor), client: Mutex::new(Some(Arc::clone(&client))), roster_epoch: AtomicU64::new(0),
            roster_stale: AtomicBool::new(false), last_frame_at: Mutex::new(None), pending_client: Mutex::new(None),
            connection: AsyncMutex::new(()), stream: Mutex::new(CompactAssistantStreamReconstructor::new()),
            recovery: AtomicBool::new(false), deferred_recovery: AtomicBool::new(false), deferred_recovery_rounds: AtomicU64::new(0),
            promoted_owner_client_id: Mutex::new(None),
            heartbeat_snapshot: Mutex::new(HeartbeatSnapshot::default()),
        });
        self.attach_worker_listeners(&worker, &client);
        self.workers.lock().unwrap().insert(worker.descriptor.lock().unwrap().worker_id.clone(), Arc::clone(&worker));
        worker
    }
    fn attach_worker_listeners(self: &Arc<Self>, worker: &Arc<Worker>, client: &Arc<DaemonWorkerClient>) {
        let weak = Arc::downgrade(self);
        let weak_worker = Arc::downgrade(worker);
        let weak_source = Arc::downgrade(client);
        let _unsubscribe = client.on_frame(Arc::new(move |frame| {
            if let (Some(supervisor), Some(worker), Some(source)) = (weak.upgrade(), weak_worker.upgrade(), weak_source.upgrade()) {
                supervisor.handle_worker_frame(&worker, frame, Some(&source));
            }
        }));
        let weak = Arc::downgrade(self);
        let weak_worker = Arc::downgrade(worker);
        let weak_source = Arc::downgrade(client);
        let _close = client.on_close(Arc::new(move |error| {
            if let (Some(supervisor), Some(worker), Some(source)) = (weak.upgrade(), weak_worker.upgrade(), weak_source.upgrade()) {
                let message = error.message();
                tokio::spawn(async move { supervisor.handle_worker_close(&worker, &source, &message).await; });
            }
        }));
        let _ = (_unsubscribe, _close);
    }
    /// `handleWorkerFrame(worker, frame, source)`.
    fn handle_worker_frame(self: &Arc<Self>, worker: &Arc<Worker>, frame: &PrivateFrame, source: Option<&Arc<DaemonWorkerClient>>) {
        if frame.header.get("kind").and_then(Value::as_str) != Some("outbound") { return; }
        if let Some(source) = source {
            let current = worker.client.lock().unwrap().as_ref().map(Arc::as_ptr);
            let pending = worker.pending_client.lock().unwrap().as_ref().map(Arc::as_ptr);
            if current != Some(Arc::as_ptr(source)) && pending != Some(Arc::as_ptr(source)) { return; }
        }
        *worker.last_frame_at.lock().unwrap() = Some(supervisor_now_ms());
        self.clear_roster_staleness(worker);
        match frame.header.get("outboundType").and_then(Value::as_str).unwrap_or("") {
            "roster_delta" => { self.consume_worker_roster_delta(worker, &frame.payload); return; }
            "roster_heartbeat" => return,
            "heartbeats_changed" => {
                worker.heartbeat_snapshot.lock().unwrap().invalidate();
                self.broadcast_heartbeats_changed();
                return;
            }
            _ => {}
        }
        self.forward_frame(worker, frame);
    }
    /// `handleWorkerClose(worker, client, error)` (daemon-supervisor.ts:3835-3888).
    ///
    /// A dead connection makes the worker's own rows non-live, and then the close
    /// itself drives recovery: mark `recovering` (3872), persist the descriptor
    /// transition with the close error (3884-3887) and hand off to the retry
    /// ladder. Without that hand-off every session of a crashed worker stays stuck
    /// on the typed "is recovering; retry shortly" error forever, because
    /// `recovery_command` was reachable only from `adopt_workers` and `retry_worker`.
    async fn handle_worker_close(self: &Arc<Self>, worker: &Arc<Worker>, client: &Arc<DaemonWorkerClient>, error: &str) {
        {
            let mut current = worker.client.lock().unwrap();
            if current.as_ref().map(Arc::as_ptr) != Some(Arc::as_ptr(client)) { return; }
            *current = None;
        }
        if self.stopped.is_cancelled() { return; }
        self.invalidate_worker_input_pauses(worker);
        self.mark_worker_roster_entries(worker, Some(DAEMON_WORKER_LIFECYCLE_RECOVERING));
        // `isWorkerRecoveryEligible` (3881, 3923-3925): no concurrent ladder may be
        // admitted for the same worker.
        if !self.is_worker_recovery_eligible(worker) { return; }
        {
            let mut descriptor = worker.descriptor.lock().unwrap();
            descriptor.lifecycle = DAEMON_WORKER_LIFECYCLE_RECOVERING.into();
            descriptor.last_error = Some(error.to_string());
            let persisted = descriptor.clone();
            drop(descriptor);
            if let Err(persist_error) = self.persist_worker(&persisted) { eprintln!("Could not persist worker descriptor {}: {persist_error}", persisted.worker_id); }
        }
        // `persistAndRecoverWorker(worker, recoveryTransition, false)` (3887): the
        // ladder is admitted (and its flag published) before the await returns.
        let _ = self.persist_and_recover_worker(worker, false).await;
    }
    /// `isWorkerRecoveryCandidate(worker)` (daemon-supervisor.ts:3949-3957).
    fn is_worker_recovery_candidate(&self, worker: &Arc<Worker>) -> bool {
        // Never nest the registry lock beneath a descriptor guard: roster/cleanup
        // readers also visit descriptors obtained from the registry.
        let descriptor = worker.descriptor.lock().unwrap().clone();
        !self.stopped.is_cancelled()
            && descriptor.lifecycle != DAEMON_WORKER_LIFECYCLE_STOPPING
            && descriptor.stop_requested_at.is_none()
            && self.workers.lock().unwrap().get(&descriptor.worker_id).is_some_and(|resident| Arc::ptr_eq(resident, worker))
            && worker.client.lock().unwrap().is_none()
    }
    /// `isWorkerRecoveryEligible(worker)` (daemon-supervisor.ts:3923-3925).
    fn is_worker_recovery_eligible(&self, worker: &Arc<Worker>) -> bool {
        self.is_worker_recovery_candidate(worker) && !worker.recovery.load(Ordering::SeqCst)
    }
    /// `persistAndRecoverWorker(worker, transition, waitForRecovery)` (daemon-supervisor.ts:3890-3921).
    /// The ladder flag is published before any await, so concurrent touches join it.
    async fn persist_and_recover_worker(self: &Arc<Self>, worker: &Arc<Worker>, wait_for_recovery: bool) -> Result<(), String> {
        // Joining an admitted ladder (`await worker.recovery`, daemon-supervisor.ts:3913-3914)
        // needs a shared in-flight handle; the flag below is the admission gate, so a second
        // caller returns instead of starting a competing ladder.
        if worker.recovery.swap(true, Ordering::SeqCst) {
            let _ = wait_for_recovery;
            return Ok(());
        }
        let result = self.recover_worker(worker).await;
        worker.recovery.store(false, Ordering::SeqCst);
        result
    }
    /// `recoverWorker(worker)` (daemon-supervisor.ts:4222-4361): the close-time
    /// retry ladder over `WORKER_RETRY_DELAYS_MS = [250, 1000, 5000]` (:210).
    async fn recover_worker(self: &Arc<Self>, worker: &Arc<Worker>) -> Result<(), String> {
        if self.is_worker_recovery_cancelled(worker) { return Ok(()); }
        if self.ownership.assert_current().await.is_err() { return Ok(()); }
        let descriptor_worker_id = worker.descriptor.lock().unwrap().worker_id.clone();
        // `let keepProbingLiveWorker = false` (daemon-supervisor.ts:4237). TS re-initialises it at
        // the top of every delay round (:4240); here every path that does not return sets it, so
        // only the last round's verdict reaches the post-loop decision, exactly as in TS.
        let mut keep_probing_live_worker = false;
        for retry_delay in WORKER_RETRY_DELAYS_MS {
            tokio::time::sleep(Duration::from_millis(retry_delay)).await;
            if self.is_worker_recovery_cancelled(worker) { return Ok(()); }
            if self.ownership.assert_current().await.is_err() { return Ok(()); }
            let descriptor = worker.descriptor.lock().unwrap().clone();
            let identity = ProcessIdentity { pid: descriptor.pid as i64, process_start_id: descriptor.process_start_id.clone() };
            // `processIdentity(pid, processStartId)` (daemon-supervisor.ts:4246, 4287-4292): the
            // three verdicts are distinguished so "unknown" keeps probing and "replaced" does not.
            let verdict = process_identity_verdict(&identity);
            let identity_compatible = verdict != ProcessIdentityVerdict::Gone && matches_exact_process_identity(&identity);
            if identity_compatible {
                match self.reconnect_worker(worker).await {
                    Ok(()) => {
                        if self.is_worker_recovery_cancelled(worker) { return Ok(()); }
                        {
                            let mut current = worker.descriptor.lock().unwrap();
                            current.lifecycle = DAEMON_WORKER_LIFECYCLE_READY.into();
                            current.consecutive_failures = 0;
                            let ready = current.clone();
                            drop(current);
                            if let Err(error) = self.persist_worker(&ready) { eprintln!("Could not persist worker descriptor {}: {error}", ready.worker_id); }
                        }
                        worker.deferred_recovery_rounds.store(0, Ordering::SeqCst);
                        return Ok(());
                    }
                    Err(error) => {
                        // daemon-supervisor.ts:4279-4284: a worker that keeps the same
                        // durable identity may still be load-slow, so keep probing it
                        // instead of replacing live work after a timeout.
                        keep_probing_live_worker = true;
                        {
                            let mut current = worker.descriptor.lock().unwrap();
                            current.consecutive_failures += 1;
                            current.last_failure_at = Some(chrono::Utc::now().to_rfc3339());
                            current.last_error = Some(error.clone());
                            let failed = current.clone();
                            drop(current);
                            if let Err(persist_error) = self.persist_worker(&failed) { eprintln!("Could not persist worker descriptor {}: {persist_error}", failed.worker_id); }
                        }
                    }
                }
                continue;
            }
            // daemon-supervisor.ts:4230-4236: a client-owned worker whose process is gone and
            // that has no stored launch environment waits for its owner, never for a relaunch.
            if verdict == ProcessIdentityVerdict::Gone && descriptor.owner_client_id.is_some() {
                if let Err(error) = self.recover_uncertain_worker_operations(worker).await {
                    self.park_worker_recovery_failure(worker, &error);
                    return Err(error);
                }
                let mut current = worker.descriptor.lock().unwrap();
                current.lifecycle = DAEMON_WORKER_LIFECYCLE_FAILED.into();
                current.last_error = Some("Waiting for the owning client to reconnect".to_string());
                let failed = current.clone();
                drop(current);
                if let Err(error) = self.persist_worker(&failed) { eprintln!("Could not persist worker descriptor {}: {error}", failed.worker_id); }
                return Err("Waiting for the owning client to reconnect".to_string());
            }
            // daemon-supervisor.ts:4287-4292: a live pid whose identity cannot be verified
            // ("unknown") keeps probing instead of being replaced.
            if verdict == ProcessIdentityVerdict::Unknown {
                keep_probing_live_worker = true;
                continue;
            }
            self.recover_uncertain_worker_operations(worker).await?;
            // daemon-supervisor.ts:4293-4302: the identity is gone and no client-owned launch
            // context is stored here, so the worker parks failed with the TS message. Recovery
            // from there is the shipped reclaim path (`create` / `retry_worker`), never a
            // silent relaunch from the close handler.
            //
            // blocked_on: the TS line above relaunches only a client-owned worker that carries
            // BOTH `transientCreateCommand` and `launchEnv`. Neither exists in
            // `DaemonWorkerDescriptor`, and the capability that would advertise that flow
            // (`owned_session_recovery_context`, daemon-supervisor.ts:1370-1375) is deliberately
            // unadvertised by this supervisor, so the launch half stays unported on purpose.
            {
                let mut current = worker.descriptor.lock().unwrap();
                current.lifecycle = DAEMON_WORKER_LIFECYCLE_FAILED.into();
                current.last_error = Some("Waiting for a client with fresh runtime context".to_string());
                let failed = current.clone();
                drop(current);
                if let Err(error) = self.persist_worker(&failed) { eprintln!("Could not persist worker descriptor {}: {error}", failed.worker_id); }
            }
            self.mark_worker_roster_entries(worker, Some(DAEMON_WORKER_LIFECYCLE_FAILED));
            return Ok(());
        }
        if self.is_worker_recovery_cancelled(worker) { return Ok(()); }
        if keep_probing_live_worker {
            // daemon-supervisor.ts:4329-4343 -> `deferWorkerRecovery` (3959-3981): stay in
            // `recovering` and re-probe every `DEFERRED_RECOVERY_RECHECK_MS`, parking failed
            // only after `MAX_DEFERRED_RECOVERY_ROUNDS` so a silent worker is never killed.
            {
                let mut current = worker.descriptor.lock().unwrap();
                current.lifecycle = DAEMON_WORKER_LIFECYCLE_RECOVERING.into();
                let recovering = current.clone();
                drop(current);
                if let Err(error) = self.persist_worker(&recovering) { eprintln!("Could not persist worker descriptor {}: {error}", recovering.worker_id); }
            }
            let rounds = worker.deferred_recovery_rounds.fetch_add(1, Ordering::SeqCst) + 1;
            if rounds > MAX_DEFERRED_RECOVERY_ROUNDS {
                let mut current = worker.descriptor.lock().unwrap();
                current.lifecycle = DAEMON_WORKER_LIFECYCLE_FAILED.into();
                current.last_error = Some(format!("Live session worker did not answer recovery probes for {MAX_DEFERRED_RECOVERY_ROUNDS} rounds"));
                let failed = current.clone();
                drop(current);
                if let Err(error) = self.persist_worker(&failed) { eprintln!("Could not persist worker descriptor {}: {error}", failed.worker_id); }
                self.mark_worker_roster_entries(worker, Some(DAEMON_WORKER_LIFECYCLE_FAILED));
                eprintln!("Worker {} is unresponsive; parked failed after {MAX_DEFERRED_RECOVERY_ROUNDS} probe rounds", descriptor_worker_id);
                return Ok(());
            }
            self.schedule_deferred_worker_recovery(worker);
            return Ok(());
        }
        // daemon-supervisor.ts:4351-4358: the worker's process is gone and no client
        // launch context exists, so it parks failed; `create` / `retry_worker` reclaim it.
        {
            let mut current = worker.descriptor.lock().unwrap();
            current.lifecycle = DAEMON_WORKER_LIFECYCLE_FAILED.into();
            let failed = current.clone();
            drop(current);
            if let Err(error) = self.persist_worker(&failed) { eprintln!("Could not persist worker descriptor {}: {error}", failed.worker_id); }
        }
        self.mark_worker_roster_entries(worker, Some(DAEMON_WORKER_LIFECYCLE_FAILED));
        eprintln!("Worker {descriptor_worker_id} failed after three recovery attempts");
        Ok(())
    }
    /// `resumeDeferredWorkerRecovery(worker, error)` (daemon-supervisor.ts:3983-4020): probe
    /// again after `DEFERRED_RECOVERY_RECHECK_MS` while the worker stays a candidate.
    fn schedule_deferred_worker_recovery(self: &Arc<Self>, worker: &Arc<Worker>) {
        if worker.deferred_recovery.swap(true, Ordering::SeqCst) { return; }
        let supervisor = Arc::clone(self);
        let worker = Arc::clone(worker);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(DEFERRED_RECOVERY_RECHECK_MS)).await;
            worker.deferred_recovery.store(false, Ordering::SeqCst);
            if !supervisor.is_worker_recovery_candidate(&worker) { return; }
            if !supervisor.is_worker_recovery_eligible(&worker) { return; }
            if supervisor.ownership.assert_current().await.is_err() { return; }
            let _ = supervisor.persist_and_recover_worker(&worker, false).await;
        });
    }
    /// `isWorkerRecoveryCancelled(worker)` (daemon-supervisor.ts:4365-4372).
    fn is_worker_recovery_cancelled(&self, worker: &Arc<Worker>) -> bool {
        if self.stopped.is_cancelled() { return true; }
        let descriptor = worker.descriptor.lock().unwrap().clone();
        descriptor.lifecycle == DAEMON_WORKER_LIFECYCLE_STOPPING
            || descriptor.stop_requested_at.is_some()
            || !self.workers.lock().unwrap().get(&descriptor.worker_id).is_some_and(|resident| Arc::ptr_eq(resident, worker))
    }
    /// `connectWorker(worker, WORKER_CONNECT_TIMEOUT_MS)` + `subscribeWorker` +
    /// `refreshWorkerSummaries(worker, true)` (daemon-supervisor.ts:4252-4254).
    async fn reconnect_worker(self: &Arc<Self>, worker: &Arc<Worker>) -> Result<(), String> {
        worker.heartbeat_snapshot.lock().unwrap().invalidate();
        let descriptor = worker.descriptor.lock().unwrap().clone();
        let client = Arc::new(DaemonWorkerClient::new(&descriptor.socket_path));
        self.attach_worker_listeners(worker, &client);
        *worker.pending_client.lock().unwrap() = Some(Arc::clone(&client));
        let result = self.authenticate(&client, &descriptor).await;
        *worker.pending_client.lock().unwrap() = None;
        if let Err(error) = result { client.close_now(); return Err(error); }
        *worker.client.lock().unwrap() = Some(Arc::clone(&client));
        self.subscribe(worker, &descriptor.root_active_session_id).await?;
        self.refresh(worker).await?;
        Ok(())
    }
    fn forward_frame(&self, worker: &Worker, frame: &PrivateFrame) {
        let kind = frame.header.get("outboundType").and_then(Value::as_str).unwrap_or("");
        if matches!(kind, "daemon_hello" | "response") { return; }
        let Ok(mut value) = serde_json::from_slice::<Value>(&frame.payload) else { return; };
        if frame.header.get("payloadEncoding").and_then(Value::as_str) == Some("assistant-delta") {
            let Ok(delta) = serde_json::from_value::<CompactAssistantDelta>(value) else { return; };
            let Some(reconstructed) = worker.stream.lock().unwrap().reconstruct(&delta) else { return; };
            value = reconstructed;
        } else { worker.stream.lock().unwrap().observe(&value); }
        if let Some(summary) = value.get("state").filter(|value| value.get("sessionId").is_some()).and_then(|value| serde_json::from_value::<RosterSessionSummary>(value.clone()).ok()) {
            value["state"] = serde_json::to_value(self.public_summary(worker, session_summary_from_roster_row(&summary, None, None, None))).unwrap_or(Value::Null);
        }
        let active = frame.header.get("activeSessionId").and_then(Value::as_str).or_else(|| value.get("activeSessionId").and_then(Value::as_str));
        let owner = worker.descriptor.lock().unwrap().owner_client_id.clone();
        for client in self.clients.lock().unwrap().values() {
            if owner.as_ref().is_some_and(|owner| owner != &client.identity()) { continue; }
            if active.is_some_and(|id| client.subscriptions.lock().unwrap().contains(id)) { client.write(&value); }
        }
    }
    fn idle_eviction_minutes(&self) -> IdleEvictionMinutes {
        let value = crate::core::settings_manager::SettingsManager::create(
            self.config.cwd.as_deref().unwrap_or("."), self.config.agent_dir.as_deref(),
        ).get_idle_eviction_minutes();
        match value {
            crate::core::settings_manager::IdleEvictionMinutes::Off => IdleEvictionMinutes::Off,
            crate::core::settings_manager::IdleEvictionMinutes::Minutes(value) => IdleEvictionMinutes::Minutes(value),
        }
    }

    fn start_idle_eviction(self: &Arc<Self>) {
        let supervisor = Arc::clone(self);
        let task = tokio::spawn(async move {
            loop {
                let minutes = supervisor.idle_eviction_minutes();
                let delay = match minutes {
                    IdleEvictionMinutes::Minutes(minutes) => (minutes * 60_000.0 / 3.0).clamp(60_000.0, 300_000.0) as u64,
                    IdleEvictionMinutes::Off => 300_000,
                };
                tokio::select! {
                    _ = supervisor.stopped.cancelled() => break,
                    _ = tokio::time::sleep(Duration::from_millis(delay)) => {},
                }
                if let Err(error) = supervisor.sweep_idle_workers().await {
                    eprintln!("Daemon idle eviction: {error}");
                }
            }
        });
        *self.idle_eviction_task.lock().unwrap() = Some(task);
    }

    fn worker_eviction_snapshot(&self, worker: &Worker) -> WorkerEvictionSnapshot {
        let descriptor = worker.descriptor.lock().unwrap().clone();
        let rows = self.worker_roster_entries(worker);
        let root = PathBuf::from(canonical_session_path(&Path::new(self.config.agent_dir.as_deref().unwrap_or("."))
            .join("sessions").to_string_lossy()));
        let has_wake_blind_schedule = rows.iter().any(|entry| {
            let row = &entry.summary;
            if row.has_registered_heartbeat != Some(true) && row.has_registered_cron_job != Some(true) { return false; }
            row.session_file.as_ref().is_none_or(|file| {
                let file = PathBuf::from(canonical_session_path(file));
                !file.starts_with(&root)
            })
        });
        WorkerEvictionSnapshot {
            lifecycle: match descriptor.lifecycle.as_str() {
                "ready" => WorkerLifecycle::Ready, "starting" => WorkerLifecycle::Starting,
                "recovering" => WorkerLifecycle::Recovering, "stopping" => WorkerLifecycle::Stopping,
                _ => WorkerLifecycle::Failed,
            },
            is_connected: worker.client.lock().unwrap().as_ref().is_some_and(|c| c.is_connected()),
            is_stopping: self.stopped.is_cancelled() || descriptor.stop_requested_at.is_some(),
            has_owner_client: descriptor.owner_client_id.is_some(),
            is_preparing_update_restart: self.ownership.snapshot().phase != "owner",
            has_wake_blind_schedule,
            sessions: rows.into_iter().filter(|entry| entry.queued_child != Some(true)).map(|entry| {
                let summary = summary_from_entry(&entry);
                let active = summary.active_session_id.as_deref().unwrap_or(&summary.id);
                SessionEvictionSnapshot {
                    is_session_active: is_session_summary_busy(summary.is_session_active, summary.has_running_rlm_children),
                    attached_clients: self.attached_client_count(&summary, active),
                    has_registered_cron_job: summary.has_registered_cron_job == Some(true),
                    last_activity_at: summary.last_activity_at.as_deref().and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                        .map(|d| d.timestamp_millis() as f64).unwrap_or(f64::NAN),
                }
            }).collect(),
        }
    }

    async fn sweep_idle_workers(self: &Arc<Self>) -> Result<(), String> {
        let minutes = self.idle_eviction_minutes();
        if minutes == IdleEvictionMinutes::Off || self.stopped.is_cancelled() { return Ok(()); }
        let workers: Vec<_> = self.workers.lock().unwrap().values().cloned().collect();
        let mut candidates = Vec::new();
        let now = chrono::Utc::now().timestamp_millis() as f64;
        for worker in workers {
            if self.stopped.is_cancelled() { return Ok(()); }
            // A failed or hung refresh must never turn stale roster data into a
            // destructive eviction decision.
            if !matches!(tokio::time::timeout(Duration::from_secs(30), self.refresh(&worker)).await, Ok(Ok(_))) { continue; }
            if can_evict_worker(&self.worker_eviction_snapshot(&worker), minutes, now) {
                candidates.push(worker);
            } else {
                let client = worker.client.lock().unwrap().clone();
                if let Some(client) = client {
                    let mut request = command("worker_passivate_idle_children");
                    if let IdleEvictionMinutes::Minutes(value) = minutes { request.insert("idleEvictionMinutes".into(), json!(value)); }
                    request.insert("now".into(), json!(now)); request.insert("limit".into(), json!(2));
                    match client.request_worker(request, 30_000).await.map_err(|e| e.to_string()).and_then(response_data) {
                        Ok(_) => {}, Err(error) => eprintln!("Child passivation sweep failed: {error}"),
                    }
                }
            }
        }
        if candidates.is_empty() { return Ok(()); }
        let _fence = tokio::time::timeout(Duration::from_secs(5), self.eviction_fence.write()).await
            .map_err(|_| "Timed out draining daemon commands for idle eviction".to_string())?;
        let _opening = tokio::time::timeout(Duration::from_secs(5), self.opening.write()).await
            .map_err(|_| "Timed out draining worker opens for idle eviction".to_string())?;
        for worker in candidates {
            if self.stopped.is_cancelled() { break; }
            let id = worker.descriptor.lock().unwrap().worker_id.clone();
            if !self.workers.lock().unwrap().get(&id).is_some_and(|w| Arc::ptr_eq(w, &worker)) { continue; }
            if !matches!(tokio::time::timeout(Duration::from_secs(30), self.refresh(&worker)).await, Ok(Ok(_))) { continue; }
            if can_evict_worker(&self.worker_eviction_snapshot(&worker), minutes, now) {
                // Archive the durable resident tree before graceful shutdown,
                // matching stopWorker(worker, true). Never force-kill a sweep.
                self.stop_worker(&worker, true, false).await?;
                eprintln!("Evicted idle worker {id}");
            }
        }
        self.schedule_scheduled_session_wake_recompute();
        Ok(())
    }

    /// `refreshWorkerSummaries(worker, recovery, fillGaps)`.
    async fn refresh(self: &Arc<Self>, worker: &Arc<Worker>) -> Result<Vec<Value>, String> {
        let client = self.connected_client(worker).await?;
        let response = client.request_worker(command("list"), REQUEST_TIMEOUT).await.map_err(|error| error.to_string())?;
        let sessions = session_summaries_from_response(response)?;
        for summary in &sessions {
            self.write_roster_entry(worker_roster_entry_from_summary(summary), Some(worker), None);
        }
        Ok(sessions.iter().map(|summary| serde_json::to_value(summary).unwrap_or(Value::Null)).collect())
    }
    /// The `Promise.all` fan-out used by the id-less supervisor commands
    /// (daemon-supervisor.ts:2442-2448): every live worker **that already holds a client** is
    /// asked (`filter((worker) => this.isLiveWorker(worker) && worker.client)`), and each entry
    /// keeps its own success/failure so the caller can pick the first failure or the first
    /// success. A worker without a client is skipped rather than failed.
    async fn forward_to_live_workers(self: &Arc<Self>, forwarded: Map<String, Value>) -> Vec<DaemonResponse> {
        let workers: Vec<Arc<Worker>> = self.workers.lock().unwrap().values().cloned().collect();
        let mut responses: Vec<DaemonResponse> = Vec::new();
        for worker in workers {
            if !self.is_live_worker(&worker) { continue; }
            let client = { worker.client.lock().unwrap().as_ref().cloned() };
            let Some(client) = client.filter(|client| client.is_connected()) else { continue };
            match client.request_worker(forwarded.clone(), REQUEST_TIMEOUT).await {
                Ok(response) => responses.push(response),
                Err(error) => responses.push(DaemonResponse::failure(None, forwarded.get("type").and_then(Value::as_str).unwrap_or(""), &error.to_string(), None)),
            }
        }
        responses
    }
    /// `this.requireWorkerClient(worker)`: reconnects a worker whose connection
    /// died, after proving the pid is still this worker's own process.
    async fn connected_client(self: &Arc<Self>, worker: &Arc<Worker>) -> Result<Arc<DaemonWorkerClient>, String> {
        if let Some(client) = worker.client.lock().unwrap().as_ref().cloned() {
            if client.is_connected() { return Ok(client); }
        }
        let _connection = worker.connection.lock().await;
        if let Some(client) = worker.client.lock().unwrap().as_ref().cloned() {
            if client.is_connected() { return Ok(client); }
        }
        let descriptor = worker.descriptor.lock().unwrap().clone();
        if !matches_exact_process_identity(&ProcessIdentity { pid: descriptor.pid as i64, process_start_id: descriptor.process_start_id.clone() }) {
            return Err(format!("Session worker {} exited; retry_worker is required", descriptor.worker_id));
        }
        let client = Arc::new(DaemonWorkerClient::new(&descriptor.socket_path));
        self.attach_worker_listeners(worker, &client);
        *worker.pending_client.lock().unwrap() = Some(Arc::clone(&client));
        let result = self.authenticate(&client, &descriptor).await;
        *worker.pending_client.lock().unwrap() = None;
        match result {
            Ok(()) => {
                *worker.client.lock().unwrap() = Some(Arc::clone(&client));
                self.subscribe(worker, &descriptor.root_active_session_id).await?;
                Ok(client)
            }
            Err(error) => { client.close_now(); Err(error) }
        }
    }
    async fn subscribe(&self, worker: &Arc<Worker>, active: &str) -> Result<(), String> {
        let client = { worker.client.lock().unwrap().as_ref().cloned().ok_or("Session worker is not connected")? };
        let mut body = command("worker_subscribe");
        body.insert("activeSessionId".into(), json!(active));
        // Full snapshots avoid a private chunk cache at the public boundary.
        let supports_ui = self.clients.lock().unwrap().values().any(|client| client.subscriptions.lock().unwrap().contains(active) && client.supports_extension_ui.load(Ordering::SeqCst));
        let capabilities = if supports_ui { vec!["attach_snapshot", "event_sequence", "extension_ui"] } else { vec!["attach_snapshot", "event_sequence"] };
        body.insert("capabilities".into(), json!(capabilities));
        body.insert("supportsExtensionUi".into(), json!(supports_ui));
        response_data(client.request_worker(body, REQUEST_TIMEOUT).await.map_err(|error| error.to_string())?).map(|_| ())
    }
    /// `consumeWorkerRosterDelta(worker, payload, source)`.
    fn consume_worker_roster_delta(self: &Arc<Self>, worker: &Arc<Worker>, payload: &[u8]) {
        let Ok(value) = serde_json::from_slice::<Value>(payload) else { return; };
        let outbound = match serde_json::from_value::<DaemonWorkerRosterOutbound>(value) {
            Ok(outbound) => outbound,
            Err(_) => return,
        };
        let DaemonWorkerRosterOutbound::RosterDelta { entries, removed_agent_ids, snapshot } = outbound else {
            // `roster_heartbeat` reaches the same consumer; it carries no rows.
            worker.roster_epoch.fetch_add(1, Ordering::SeqCst);
            return;
        };
        worker.roster_epoch.fetch_add(1, Ordering::SeqCst);
        if snapshot == Some(true) { self.apply_worker_roster_snapshot(worker, entries, removed_agent_ids); }
        else { self.apply_worker_roster_delta(worker, entries, removed_agent_ids); }
    }
    /// `applyWorkerRosterDelta`.
    fn apply_worker_roster_delta(self: &Arc<Self>, worker: &Arc<Worker>, entries: Vec<WorkerRosterEntry>, removed_agent_ids: Option<Vec<String>>) {
        for entry in entries {
            self.write_roster_entry(entry.clone(), Some(worker), None);
            self.sync_root_descriptor_from_roster_entry(worker, &entry);
        }
        for agent_id in removed_agent_ids.unwrap_or_default() {
            self.roster().lock().unwrap().delete(&agent_id);
        }
    }
    /// `applyWorkerRosterSnapshot`: the sweep of rows this worker did not claim,
    /// then the snapshot's own rows.
    ///
    /// blocked_on: the TS body is async and reads the spawn ledger twice (once to
    /// decide `edgesFailed` and once to reseed this worker's family). This port
    /// applies frames synchronously from the worker listener, so the ledger read
    /// and the reseed are not on the frame path; the boot seed
    /// (`seed_roster_ledger`) and `list(all=true)` supply those rows instead.
    /// The `edgesFailed` repair pull (`scheduleRosterRepairPull`) is therefore
    /// unreachable here.
    fn apply_worker_roster_snapshot(self: &Arc<Self>, worker: &Arc<Worker>, entries: Vec<WorkerRosterEntry>, removed_agent_ids: Option<Vec<String>>) {
        let sent: HashSet<String> = entries.iter().map(|entry| entry.agent_id.clone()).collect();
        let removed: HashSet<String> = removed_agent_ids.clone().unwrap_or_default().into_iter().collect();
        let unclaimed: Vec<AgentRosterEntry> = self.worker_roster_entries(worker)
            .into_iter().filter(|entry| !sent.contains(&entry.agent_id)).collect();
        // Unreadable edges skip the absentee sweep: it cannot tell registry children
        // from stale rows. The ledger read is async, so the sweep is conservative here.
        for entry in &unclaimed {
            self.roster().lock().unwrap().delete(&entry.agent_id);
        }
        for entry in entries {
            self.write_roster_entry(entry.clone(), Some(worker), None);
            self.sync_root_descriptor_from_roster_entry(worker, &entry);
        }
        for agent_id in removed { self.roster().lock().unwrap().delete(&agent_id); }
    }
    /// `handleList(client, command)`: live roster rows, then the saved-session
    /// catalog merge and the spawn ledger's dead families when `all` is set.
    async fn handle_list(&self, public: &PublicClient, body: &Map<String, Value>) -> Result<Value, String> {
        let include_client_owned = body.get("includeClientOwned").and_then(Value::as_bool) == Some(true);
        let all = body.get("all").and_then(Value::as_bool) == Some(true);
        let workers: HashMap<String, Arc<Worker>> = self.workers.lock().unwrap().clone();
        let mut active: Vec<SessionSummary> = Vec::new();
        let mut active_by_file: HashMap<String, SessionSummary> = HashMap::new();
        let mut busy_client_owned_session_count: i64 = 0;
        for entry in self.roster().lock().unwrap().values() {
            if entry.queued_child == Some(true) { continue; }
            let Some(worker) = entry.worker_id.as_ref().and_then(|worker_id| workers.get(worker_id)).cloned() else { continue; };
            let summary = self.public_summary(&worker, summary_from_entry(&entry));
            if self.is_visible_worker(&worker) {
                if let Some(file) = summary.session_file.as_ref() { active_by_file.insert(canonical_session_path(file), summary.clone()); }
                active.push(summary);
                continue;
            }
            if let Some(file) = summary.session_file.as_ref() { active_by_file.insert(canonical_session_path(file), summary.clone()); }
            // Busy counts come from the row's REAL state, never a synthetic flag.
            if is_session_summary_busy(summary.is_session_active, summary.has_running_rlm_children) { busy_client_owned_session_count += 1; }
            if include_client_owned && visible(&worker, &public.identity()) { active.push(summary); }
        }
        let mut data = json!({ "sessions": active });
        if include_client_owned { data["busyClientOwnedSessionCount"] = json!(busy_client_owned_session_count); }
        if !all { return Ok(data); }
        let session_dir = body.get("sessionDir").and_then(Value::as_str).map(str::to_string).or_else(|| self.config.session_dir.clone());
        let cwd = body.get("cwd").and_then(Value::as_str).map(resolve_path);
        let scanned = self.catalog.list(cwd.as_deref(), session_dir.as_deref(), None).await?;
        let mut merged: Vec<SessionSummary> = Vec::new();
        let mut served_rows: HashSet<String> = HashSet::new();
        for summary in &active { served_rows.insert(summary.served_row_key()); }
        let mut merged_active_files: HashSet<String> = HashSet::new();
        let mut scanned_files: HashSet<String> = HashSet::new();
        for info in &scanned {
            let file = canonical_session_path(&info.path);
            scanned_files.insert(file.clone());
            let worker_row = active_by_file.get(&file);
            // The on-disk scan is public: an unserved (client-owned) worker row hides its live metadata only.
            match worker_row.filter(|row| served_rows.contains(&row.served_row_key())) {
                Some(row) => { merged.push(row.clone()); merged_active_files.insert(file); }
                None => merged.push(summary_for_inactive_session(info, false, false)),
            }
        }
        let spawn_edges = match self.rlm_spawn_ledger().await { Ok(ledger) => ledger.live_edges().await, Err(error) => { eprintln!("Could not list spawn-ledger sessions: {error}"); Vec::new() } };
        let spawn_parents: HashMap<String, String> = spawn_edges.iter()
            .map(|edge| (canonical_session_path(&edge.child), edge.parent.clone())).collect();
        let offline_rows: Vec<AgentRosterEntry> = self.roster().lock().unwrap().values().into_iter()
            .filter(|entry| {
                if entry.queued_child == Some(true) || entry.summary.active_session_id.is_some() { return false; }
                if entry.worker_id.as_ref().is_some_and(|worker_id| workers.contains_key(worker_id)) { return false; }
                let Some(file) = entry.summary.session_file.as_ref().map(|file| canonical_session_path(file)) else { return false; };
                !scanned_files.contains(&file) && !active_by_file.contains_key(&file)
            })
            .collect();
        for entry in offline_rows {
            let hydrated = hydrate_seeded_roster_entry(entry, self).await;
            let summary = summary_from_entry(&hydrated);
            if cwd.as_ref().is_some_and(|cwd| resolve_path(&summary.cwd) != *cwd) { continue; }
            if !matches_list_session_dir(&summary, session_dir.as_deref(), &spawn_parents) { continue; }
            merged.push(summary);
        }
        let mut unseeded_files: HashSet<String> = HashSet::new();
        for edge in &spawn_edges {
            let child_path = canonical_session_path(&edge.child);
            if scanned_files.contains(&child_path) || active_by_file.contains_key(&child_path) || unseeded_files.contains(&child_path) { continue; }
            if self.roster().lock().unwrap().has_session_file(&child_path) { continue; }
            let entry = roster_entry_for_spawn_ledger_edge(edge);
            if self.roster().lock().unwrap().has(&entry.agent_id) { continue; }
            unseeded_files.insert(child_path);
            // Hydrated one at a time, like the boot seed: a large dead-family ledger must not fan
            // out into one concurrent transcript read per child.
            let hydrated = hydrated_seed_entry(&entry).await;
            // The same classification a roster write would have applied: these rows read "inactive".
            let status = classify_session_roster_status(
                &RosterSummaryView { active_session_id: hydrated.summary.active_session_id.clone(), activity: Some(hydrated.summary.activity.clone()), is_session_active: Some(hydrated.summary.is_session_active) },
                hydrated.queued_child == Some(true),
            );
            let summary = session_summary_from_roster_row(&hydrated.summary, Some(status), None, None);
            if cwd.as_ref().is_some_and(|cwd| resolve_path(&summary.cwd) != *cwd) { continue; }
            if !matches_list_session_dir(&summary, session_dir.as_deref(), &spawn_parents) { continue; }
            merged.push(summary);
        }
        for summary in active {
            let file = summary.session_file.as_ref().map(|file| canonical_session_path(file));
            if file.is_some_and(|file| merged_active_files.contains(&file)) { continue; }
            merged.push(summary);
        }
        data["sessions"] = json!(merged);
        Ok(data)
    }
    /// `hydrateSeededEntry(entry)`: refresh a seeded row's cwd from its transcript.
    /// `syncRootDescriptorFromRosterEntry(worker, entry)`.
    fn sync_root_descriptor_from_roster_entry(&self, worker: &Arc<Worker>, entry: &WorkerRosterEntry) {
        let summary = &entry.summary;
        if summary.active_session_id.as_deref() != Some(worker.descriptor.lock().unwrap().root_active_session_id.as_str()) { return; }
        {
            let mut descriptor = worker.descriptor.lock().unwrap();
            if descriptor.root_session_id == Some(summary.session_id.clone()) && descriptor.session_file == summary.session_file { return; }
            descriptor.root_session_id = Some(summary.session_id.clone());
            descriptor.session_file = summary.session_file.clone();
            descriptor.create_command = DurableDaemonCreateCommand {
                type_: "create".into(),
                session_path: summary.session_file.clone(),
                no_session: descriptor.create_command.no_session,
                extra: Map::new(),
            };
        }
        let descriptor = worker.descriptor.lock().unwrap().clone();
        if let Err(error) = self.persist_worker(&descriptor) { eprintln!("Could not persist worker descriptor {}: {error}", descriptor.worker_id); }
    }
    /// `markWorkerRosterEntries(worker, statusLabel)`.
    fn mark_worker_roster_entries(&self, worker: &Arc<Worker>, status_label: Option<&str>) {
        for entry in self.worker_roster_entries(worker) {
            if entry.queued_child != Some(true) && entry.summary.active_session_id.is_none() { continue; }
            self.roster().lock().unwrap().amend(&entry.agent_id, RosterEntryMarks { status_label: Some(status_label.map(str::to_string)), last_heard_from_at: None });
        }
    }
    /// `flipWorkerRosterEntriesInactive(worker)`: an owned worker's rows are
    /// ephemeral and die with the registration; a resident worker's rows are
    /// passivated so a scheduled wake can still find them.
    fn flip_worker_roster_entries_inactive(&self, worker: &Arc<Worker>) {
        let ephemeral = worker.descriptor.lock().unwrap().owner_client_id.is_some();
        for entry in self.worker_roster_entries(worker) {
            if ephemeral || entry.queued_child == Some(true) {
                self.roster().lock().unwrap().delete(&entry.agent_id);
                continue;
            }
            let registrations = RegisteredHeartbeatFlags {
                has_registered_heartbeat: entry.summary.has_registered_heartbeat == Some(true),
                has_registered_cron_job: entry.summary.has_registered_cron_job == Some(true),
            };
            let passivated = passivated_worker_roster_entry(
                &WorkerRosterEntry { agent_id: entry.agent_id.clone(), queued_child: entry.queued_child, seeded_cwd: entry.seeded_cwd, summary: entry.summary.clone() },
                Some(registrations),
            );
            self.write_roster_entry(passivated, None, None);
        }
    }
    /// `clearRosterStaleness(worker)`: a live frame clears the mark at once, not
    /// at the next watchdog tick, so `lastHeardFromAt` never outlives the report.
    fn clear_roster_staleness(&self, worker: &Arc<Worker>) {
        if !worker.roster_stale.swap(false, Ordering::SeqCst) { return; }
        for entry in self.worker_roster_entries(worker) {
            self.roster().lock().unwrap().amend(&entry.agent_id, RosterEntryMarks { status_label: None, last_heard_from_at: Some(None) });
        }
    }
    /// `sweepRosterStaleness(now)`.
    fn sweep_roster_staleness(&self) {
        let workers: Vec<Arc<Worker>> = self.workers.lock().unwrap().values().cloned().collect();
        for worker in workers {
            let Some(last_frame_at) = *worker.last_frame_at.lock().unwrap() else { continue; };
            if worker.client.lock().unwrap().is_none() { continue; }
            if supervisor_now_ms().saturating_sub(last_frame_at) > ROSTER_STALE_AFTER_MS {
                let last_heard_from_at = iso_from_ms(last_frame_at as f64);
                for entry in self.worker_roster_entries(&worker) {
                    if entry.last_heard_from_at.as_deref() != Some(last_heard_from_at.as_str()) {
                        self.roster().lock().unwrap().amend(&entry.agent_id, RosterEntryMarks { status_label: None, last_heard_from_at: Some(Some(last_heard_from_at.clone())) });
                    }
                }
                worker.roster_stale.store(true, Ordering::SeqCst);
            } else {
                self.clear_roster_staleness(&worker);
            }
        }
    }
    /// `scheduleScheduledSessionWakeRecompute()` (daemon-supervisor.ts:942-957): one recompute
    /// at a time, with a queued re-run so a change during a recompute is never lost.
    fn schedule_scheduled_session_wake_recompute(self: &Arc<Self>) {
        if self.stopped.is_cancelled() { return; }
        if self.scheduled_wake_recompute.swap(true, Ordering::SeqCst) {
            self.scheduled_wake_recompute_queued.store(true, Ordering::SeqCst);
            return;
        }
        let supervisor = Arc::clone(self);
        tokio::spawn(async move {
            let mut requeue = true;
            while requeue {
                requeue = false;
                if let Err(error) = supervisor.recompute_scheduled_session_wake().await {
                    eprintln!("Scheduled-session wake recompute failed: {error}");
                }
                supervisor.scheduled_wake_recompute.store(false, Ordering::SeqCst);
                if supervisor.scheduled_wake_recompute_queued.swap(false, Ordering::SeqCst) { requeue = true; }
            }
        });
    }
    /// `recomputeScheduledSessionWake()` (daemon-supervisor.ts:1028-1056): arm a timer for the
    /// earliest due passive scheduled job, with the failure floor that keeps an overdue job
    /// off a hot retry loop.
    async fn recompute_scheduled_session_wake(self: &Arc<Self>) -> Result<(), String> {
        if self.stopped.is_cancelled() { return Ok(()); }
        let candidates = self.collect_passive_scheduled_jobs(false).await;
        if self.stopped.is_cancelled() { return Ok(()); }
        {
            let mut timer = self.scheduled_wake_timer.lock().unwrap();
            if let Some(handle) = timer.take() { handle.abort(); }
        }
        let now = supervisor_now_ms() as f64;
        let candidate_roots: HashSet<String> = candidates.iter().map(|(root, _)| canonical_session_path(root)).collect();
        {
            let mut failures = self.scheduled_wake_failures.lock().unwrap();
            failures.retain(|root, _| candidate_roots.contains(root));
        }
        let mut wake_times: Vec<f64> = Vec::new();
        for (root_session_file, job) in &candidates {
            if job.get("status").and_then(Value::as_str) != Some(crate::core::cron_jobs::STATUS_ACTIVE) { continue; }
            let Some(next_run_at) = job.get("nextRunAt").and_then(Value::as_str) else { continue; };
            let Some(run_at) = iso_to_ms(next_run_at) else { continue; };
            let failed_at = self.scheduled_wake_failures.lock().unwrap().get(&canonical_session_path(root_session_file)).copied();
            wake_times.push(match failed_at { Some(failed_at) => run_at.max(failed_at + SCHEDULED_WAKE_RETRY_MS), None => run_at });
        }
        if wake_times.is_empty() { return Ok(()); }
        let earliest = wake_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let delay = (earliest - now).max(0.0).min(SCHEDULED_WAKE_MAX_TIMEOUT_MS);
        let supervisor = Arc::clone(self);
        let handle = tokio::spawn(async move {
            tokio::select! {
                _ = supervisor.stopped.cancelled() => {}
                _ = tokio::time::sleep(Duration::from_millis(delay as u64)) => {
                    if let Err(error) = supervisor.wake_due_scheduled_sessions().await {
                        eprintln!("Scheduled-session wake failed: {error}");
                    }
                }
            }
        });
        // The TS unrefs this timer; aborting it on shutdown matches that (see `start_roster_watchdog`).
        let replaced = self.scheduled_wake_timer.lock().unwrap().replace(handle);
        if let Some(replaced) = replaced { replaced.abort(); }
        Ok(())
    }
    /// `wakeDueScheduledSessions(now)` (daemon-supervisor.ts:1058-1086): create-or-reuse a
    /// worker for every passive root whose active scheduled job is due, then re-arm.
    async fn wake_due_scheduled_sessions(self: &Arc<Self>) -> Result<(), String> {
        self.scheduled_wake_timer.lock().unwrap().take();
        if self.stopped.is_cancelled() { return Ok(()); }
        let now = supervisor_now_ms() as f64;
        let mut due: Vec<(String, String)> = Vec::new();
        {
            let mut seen: HashSet<String> = HashSet::new();
            for (root_session_file, job) in self.collect_passive_scheduled_jobs(false).await {
                if job.get("status").and_then(Value::as_str) != Some(crate::core::cron_jobs::STATUS_ACTIVE) { continue; }
                let Some(run_at) = job.get("nextRunAt").and_then(Value::as_str).and_then(iso_to_ms) else { continue; };
                if run_at > now { continue; }
                let key = canonical_session_path(&root_session_file);
                if seen.insert(key) { due.push((canonical_session_path(&root_session_file), root_session_file)); }
            }
        }
        for (root_key, session_path) in due {
            if self.stopped.is_cancelled() { return Ok(()); }
            let body = json!({"type":"create", "sessionPath": session_path}).as_object().cloned().unwrap_or_default();
            // The supervisor only wakes non-resident trees; firing and delivery stay worker-owned
            // (daemon-supervisor.ts:941), so the wake uses the `SCHEDULED_WAKE_CLIENT_ID` owner.
            match self.create_for_owner(SCHEDULED_WAKE_CLIENT_ID.to_string(), &body).await {
                Ok(_) => { self.scheduled_wake_failures.lock().unwrap().remove(&root_key); eprintln!("Woke session worker for a due scheduled job: {session_path}"); }
                Err(error) => { self.scheduled_wake_failures.lock().unwrap().insert(root_key, supervisor_now_ms() as f64); eprintln!("Scheduled wake failed for {session_path}: {error}"); }
            }
        }
        self.schedule_scheduled_session_wake_recompute();
        Ok(())
    }
    /// The roster watchdog (`setInterval(() => this.sweepRosterStaleness(), ROSTER_WATCHDOG_INTERVAL_MS)`).
    fn start_roster_watchdog(self: &Arc<Self>) {
        let supervisor = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(ROSTER_WATCHDOG_INTERVAL_MS));
            loop {
                tokio::select! { _ = supervisor.stopped.cancelled() => break, _ = interval.tick() => { supervisor.sweep_roster_staleness(); supervisor.report_pending_command_journal(); } }
            }
        });
    }

    /// Bounded command-journal delivery observability (audit D-07): entries that
    /// were received but never resolved are uncertain commands that are never
    /// replayed; this only makes the backlog observable. It logs at startup, on
    /// count changes, and at most once per aged interval; it never resends and
    /// adds no wire shape.
    fn report_pending_command_journal(&self) {
        let summary = self.journal.lock().unwrap().pending_summary();
        if summary.total == 0 {
            *self.pending_command_journal_log.lock().unwrap() = None;
            return;
        }
        let now = supervisor_now_ms();
        let mut state = self.pending_command_journal_log.lock().unwrap();
        let should_log = match *state {
            None => true,
            Some((count, at)) => count != summary.total || now.saturating_sub(at) >= PENDING_COMMAND_JOURNAL_REPORT_INTERVAL_MS,
        };
        if !should_log { return; }
        *state = Some((summary.total, now));
        let oldest = summary.oldest_recorded_at.as_deref().unwrap_or("unknown");
        let oldest_type = summary.oldest_command_type.as_deref().unwrap_or("unknown");
        eprintln!(
            "[{}] Command recovery journal: {} unacknowledged command(s), {} still without a result; oldest is a {oldest_type} recorded {oldest}",
            iso_from_ms(now as f64), summary.total, summary.without_result
        );
    }
    /// `seedRosterLedger()`: registered workers' families become roster rows.
    async fn seed_roster_ledger(self: &Arc<Self>) {
        let roots: HashSet<String> = self.workers.lock().unwrap().values()
            .filter_map(|worker| {
                let descriptor = worker.descriptor.lock().unwrap();
                descriptor.session_file.clone().or_else(|| descriptor.create_command.session_path.clone())
            })
            .map(|root| canonical_session_path(&root))
            .collect();
        if roots.is_empty() { return; }
        let ledger = match self.rlm_spawn_ledger().await {
            Ok(ledger) => ledger,
            Err(error) => { eprintln!("Could not seed the agent roster from the spawn ledger: {error}"); return; }
        };
        let edges = ledger.live_edges().await;
        // The parent map is owned, so the walk does not borrow `edges` across the loop.
        let parent_by_child = roster_parent_by_child(&edges);
        for edge in &edges {
            if !roster_path_descends_from(&parent_by_child, &canonical_session_path(&edge.parent), &roots) { continue; }
            let entry = roster_entry_for_spawn_ledger_edge(&edge);
            if self.roster().lock().unwrap().has(&entry.agent_id) { continue; }
            if self.roster().lock().unwrap().has_session_file(&canonical_session_path(&edge.child)) { continue; }
            let hydrated = hydrated_seed_entry(&entry).await;
            self.roster().lock().unwrap().write(hydrated, None, None);
        }
    }
    async fn adopt_workers(self: &Arc<Self>) -> Result<(), String> {
        for entry in std::fs::read_dir(&self.descriptor_dir).map_err(|error| error.to_string())? {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") { continue; }
            let descriptor: DaemonWorkerDescriptor = match std::fs::read(&path).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()) { Some(value) => value, None => continue };
            if normalize_socket_path_for_daemon(&descriptor.supervisor_socket_path, None) != self.socket_path { continue; }
            let identity = ProcessIdentity { pid: descriptor.pid as i64, process_start_id: descriptor.process_start_id.clone() };
            if !is_stopping_process_alive(&identity) {
                let worker = self.install_worker(descriptor.clone(), Arc::new(DaemonWorkerClient::new(&descriptor.socket_path)));
                worker.client.lock().unwrap().take();
                if stop_cleanup_is_parked(&descriptor) {
                    self.mark_worker_roster_entries(&worker, Some(DAEMON_WORKER_LIFECYCLE_FAILED));
                    continue;
                }
                if descriptor.stop_requested_at.is_some() || descriptor.owner_client_id.is_some() {
                    if let Err(error) = self.stop_worker(&worker, descriptor.archive_on_stop == Some(true), false).await {
                        eprintln!("Retaining stopped worker cleanup record {}: {error}", descriptor.worker_id);
                        self.schedule_worker_stop_finalization(&worker);
                    }
                    continue;
                }
                if let Err(error) = self.recover_uncertain_worker_operations(&worker).await {
                    self.park_worker_recovery_failure(&worker, &error);
                    continue;
                }
                if descriptor.owner_client_id.is_none() && descriptor.session_file.is_some() {
                    let recovered = async {
                        let body = recovery_command(&descriptor)?;
                        self.launch_worker(&body, String::new(), Some(descriptor)).await
                    }.await;
                    if let Err(error) = recovered { self.park_worker_recovery_failure(&worker, &error); }
                } else {
                    // A process lost before create completed has no durable session
                    // to recover. Retain its record, but never leave it starting forever.
                    self.park_worker_recovery_failure(&worker, "Session worker exited before creation completed; open a new session to retry");
                }
                continue;
            }
            if descriptor.stop_requested_at.is_some() {
                let worker = self.install_worker(descriptor.clone(), Arc::new(DaemonWorkerClient::new(&descriptor.socket_path)));
                worker.client.lock().unwrap().take();
                self.schedule_worker_stop_finalization(&worker);
                continue;
            }
            if !matches_exact_process_identity(&identity) { continue; }
            let client = Arc::new(DaemonWorkerClient::new(&descriptor.socket_path));
            let root = descriptor.root_active_session_id.clone();
            // Listeners are installed before authentication so the roster snapshot
            // the worker flushes right after auth cannot race them.
            let worker = self.install_worker(descriptor, client);
            let client = { worker.client.lock().unwrap().clone().expect("installed worker client") };
            let descriptor = { worker.descriptor.lock().unwrap().clone() };
            let adopted = async {
                self.authenticate(&client, &descriptor).await?;
                self.refresh(&worker).await?;
                self.subscribe(&worker, &root).await
            }.await;
            if let Err(error) = adopted { self.park_worker_recovery_failure(&worker, &error); }
        }
        Ok(())
    }
    async fn create(self: &Arc<Self>, public: &PublicClient, body: &Map<String, Value>) -> Result<Value, String> {
        self.create_for_owner(public.identity(), body).await
    }
    /// `createOrReuseWorker(clientId, command)` (daemon-supervisor.ts:3033): the owner is a
    /// client identity string, so the scheduled wake can reuse the same path with
    /// `SCHEDULED_WAKE_CLIENT_ID` (:1075) instead of a live client connection.
    async fn create_for_owner(self: &Arc<Self>, owner: String, body: &Map<String, Value>) -> Result<Value, String> {
        // `if (command.name !== undefined) { const normalizedName = command.name.trim(); if
        // (!normalizedName) throw new Error("Session name cannot be empty"); createCommand = {...command,
        // name: normalizedName}; }` (daemon-supervisor.ts:3035-3041). The normalization happens before
        // the open key is derived, so a whitespace-padded name still collides with its trimmed form.
        let mut create_command = body.clone();
        if let Some(name) = body.get("name").and_then(Value::as_str) {
            let normalized = name.trim();
            if normalized.is_empty() { return Err("Session name cannot be empty".into()); }
            create_command.insert("name".into(), json!(normalized));
        }
        let body = &create_command;
        // `const key = createCommand.sessionPath ? canonicalSessionPath(createCommand.sessionPath)
        // : \`new:${command.id ? createCommandIdempotencyKey(clientId, command.id) :
        // createActiveSessionId()}\`;` (daemon-supervisor.ts:3060-3062).
        let opening_key = match body.get("sessionPath").and_then(Value::as_str) {
            Some(path) => canonical_session_path(path),
            None => format!("new:{}", body.get("id").and_then(Value::as_str)
                .map(|id| create_command_idempotency_key(&owner, id))
                .unwrap_or_else(|| create_active_session_id(None))),
        };
        self.with_worker_open(opening_key, self.create_named_worker(owner, body)).await
    }

    async fn with_worker_open<F>(self: &Arc<Self>, opening_key: String, operation: F) -> Result<Value, String>
    where F: std::future::Future<Output = Result<Value, String>> {
        // `const pending = this.openingWorkers.get(key); const opened = this.openingWorkers.get(key);`
        // (:3063, 3073): the map is the only gate, so the check and the registration are one
        // critical section and a concurrent opener either joins or becomes the single opener.
        let pending = {
            let mut openings = self.opening_workers.lock().unwrap();
            match openings.get(&opening_key).cloned() {
                Some(pending) => Some(pending),
                None => {
                    let opening = Arc::new(OpeningWorker { done: CancellationToken::new(), result: Mutex::new(None) });
                    openings.insert(opening_key.clone(), Arc::clone(&opening));
                    None
                }
            }
        };
        if let Some(pending) = pending {
            // `return this.joinOpeningWorker(pending, ownerClientId, createCommand.sessionPath ?? key);`
            // (:3065): `await pending` (:3153) then reuse its outcome.
            pending.done.cancelled().await;
            let result = pending.result.lock().unwrap().clone();
            return match result {
                Some(Ok(value)) => Ok(value),
                Some(Err(error)) => Err(error),
                None => Err("Session worker open was interrupted; retry opening the session".into()),
            };
        }
        let opening = self.opening_workers.lock().unwrap().get(&opening_key).cloned()
            .ok_or("Session worker open was interrupted; retry opening the session")?;
        let _guard = OpeningWorkerGuard { supervisor: Arc::clone(self), key: opening_key, opening: Arc::clone(&opening) };
        let _opening = self.opening.read().await;
        let result = if self.stopped.is_cancelled() { Err("Daemon is shutting down".to_string()) } else { operation.await };
        *opening.result.lock().unwrap() = Some(result.clone());
        result
    }

    async fn create_named_worker(self: &Arc<Self>, owner: String, body: &Map<String, Value>) -> Result<Value, String> {
        // `if (!createCommand.name) return this.launchWorker(createCommand, undefined, ownerClientId);`
        // (daemon-supervisor.ts:3085): the name check only guards a named create. The target is the
        // saved sibling when the path is known (:3086-3090), else the synthetic new-root summary.
        match body.get("name").and_then(Value::as_str) {
            None => self.create_for_owner_inner(owner, body).await,
            Some(name) => {
                // `const savedSiblings = createCommand.sessionPath ? await
                // this.rlmLedgerSiblings(createCommand.sessionPath) : [];` (:3086) and the
                // `canonicalSessionPath(session.path) === canonicalSessionPath(...)` match (:3087-3089).
                let siblings = match body.get("sessionPath").and_then(Value::as_str) {
                    Some(path) => match self.rlm_spawn_ledger().await {
                        Ok(ledger) => ledger.siblings(path).await,
                        Err(error) => { eprintln!("Could not read spawn-ledger siblings for the session name check: {error}"); Vec::new() }
                    },
                    None => Vec::new(),
                };
                let target = body.get("sessionPath").and_then(Value::as_str)
                    .and_then(|path| siblings.iter().find(|info| canonical_session_path(&info.path) == canonical_session_path(path)).cloned());
                // `const targetSummary = target ? summaryForInactiveSession(target) : { sessionId:
                // "new-root", rlmDepth: 0 };` (:3090).
                let target_summary = target.as_ref()
                    .map(|info| summary_for_inactive_session(info, false, false))
                    .unwrap_or_else(|| SessionSummary { session_id: "new-root".to_string(), rlm_depth: Some(0), ..SessionSummary::default() });
                let (scope, name) = Self::summary_name_reservation_input(&target_summary, name);
                let supervisor = Arc::clone(self);
                let reserved_name = name.clone();
                self.with_session_name_reservation(&scope, &reserved_name, async move {
                    // `if (target?.parentSessionPath && (target.rlmDepth ?? 0) > 0)
                    // this.assertSavedSiblingNameAvailable(savedSiblings, target, createCommand.name!);
                    // else await this.assertSupervisorSessionNameAvailable(targetSummary, ...)`
                    // (:3093-3097), then `return this.launchWorker(...)` (:3098).
                    match target.as_ref().filter(|info| info.parent_session_path.is_some() && info.rlm_depth > 0) {
                        Some(target) => supervisor.assert_saved_sibling_name_available(&siblings, target, &name).await?,
                        None => supervisor.assert_supervisor_session_name_available(&target_summary, &name).await?,
                    }
                    supervisor.create_for_owner_inner(owner, body).await
                }).await
            }
        }
    }
    /// The guarded body of `createOrReuseWorker` (daemon-supervisor.ts:3033) behind the
    /// `openingWorkers` join.
    async fn create_for_owner_inner(self: &Arc<Self>, owner: String, body: &Map<String, Value>) -> Result<Value, String> {
        if let Some(path) = body.get("sessionPath").and_then(Value::as_str) {
            // `matchWorkers(command.sessionPath)` (daemon-supervisor.ts:3044) is counted before
            // any reuse: when more than one worker claims the file the TS throws
            // `Ambiguous active session "<path>"` (:3051-3053) instead of attaching to an
            // arbitrary one.
            let workers: Vec<_> = self.workers.lock().unwrap().values().cloned().collect();
            let mut matches: Vec<(Arc<Worker>, SessionSummary)> = Vec::new();
            for worker in workers {
                if !visible(&worker, &owner) { continue; }
                let descriptor = worker.descriptor.lock().unwrap().clone();
                let target = canonical_session_path(path);
                let matches_path = descriptor.session_file.as_ref().or(descriptor.create_command.session_path.as_ref())
                    .is_some_and(|file| canonical_session_path(file) == target)
                    || self.find_summary_in_worker(&worker, path).is_some();
                if !matches_path { continue; }
                if self.reclaim_stale_worker_registration(&worker).await? { continue; }
                // The refresh republishes the worker's rows, so the summary that
                // answers a reuse is the roster's, matched on the canonical path.
                self.refresh(&worker).await?;
                if let Some(summary) = self.find_summary_in_worker(&worker, path) {
                    matches.push((worker, summary));
                }
            }
            if matches.len() > 1 { return Err(format!("Ambiguous active session \"{path}\"")); }
            if let Some((worker, summary)) = matches.into_iter().next() {
                return Ok(serde_json::to_value(self.public_summary(&worker, summary)).unwrap_or(Value::Null));
            }
        }
        self.launch_worker(body, owner, None).await
    }

    fn retry_worker_descriptor(&self, requested: &str) -> Result<(Option<Arc<Worker>>, DaemonWorkerDescriptor), String> {
        let workers: Vec<_> = self.workers.lock().unwrap().values().cloned().collect();
        for worker in workers {
            let descriptor = worker.descriptor.lock().unwrap().clone();
            if descriptor.root_active_session_id == requested || descriptor.root_session_id.as_deref() == Some(requested) {
                return Ok((Some(worker), descriptor));
            }
        }
        for entry in std::fs::read_dir(&self.descriptor_dir).map_err(|error| error.to_string())? {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") { continue; }
            if let Some(descriptor) = std::fs::read(path).ok().and_then(|bytes| serde_json::from_slice::<DaemonWorkerDescriptor>(&bytes).ok()) {
                if descriptor.root_active_session_id == requested || descriptor.root_session_id.as_deref() == Some(requested) {
                    return Ok((None, descriptor));
                }
            }
        }
        Err(format!("Unknown active session: {requested}"))
    }

    async fn retry_worker(self: &Arc<Self>, owner: String, requested: &str) -> Result<Value, String> {
        let (_, descriptor) = self.retry_worker_descriptor(requested)?;
        if descriptor.owner_client_id.as_deref().is_some_and(|client| client != owner) { return Err(format!("Unknown active session: {requested}")); }
        let key = descriptor.session_file.as_ref().or(descriptor.create_command.session_path.as_ref())
            .map(|path| canonical_session_path(path)).unwrap_or_else(|| format!("worker:{}", descriptor.worker_id));
        self.with_worker_open(key, async {
            // Another create/retry may have completed before we became the opener.
            let (worker, descriptor) = self.retry_worker_descriptor(requested)?;
            if descriptor.stop_requested_at.is_some() { return Err("Session worker is stopping".into()); }
            if descriptor.owner_client_id.as_deref().is_some_and(|client| client != owner) { return Err(format!("Unknown active session: {requested}")); }
            if let Some(worker) = worker {
                if matches_exact_process_identity(&ProcessIdentity { pid: descriptor.pid as i64, process_start_id: descriptor.process_start_id.clone() }) {
                    return Ok(self.refresh(&worker).await?.into_iter().find(|summary| summary.get("id").and_then(Value::as_str) == Some(&descriptor.root_active_session_id)).unwrap_or(Value::Null));
                }
            }
            if descriptor.owner_client_id.is_some() { return Err("Client-owned session recovery requires the owning client environment".into()); }
            let recovery = recovery_command(&descriptor)?;
            self.launch_worker(&recovery, owner, Some(descriptor)).await
        }).await
    }

    fn mark_created_worker_ready(&self, worker: &Arc<Worker>, expected: &DaemonWorkerDescriptor, summary: &Value) -> Result<(), String> {
        let workers = self.workers.lock().unwrap();
        if !workers.get(&expected.worker_id).is_some_and(|current| Arc::ptr_eq(current, worker)) {
            return Err("Session worker registration changed while opening".into());
        }
        let mut current = worker.descriptor.lock().unwrap();
        if self.stopped.is_cancelled() || current.stop_requested_at.is_some()
            || current.worker_instance_id != expected.worker_instance_id || current.pid != expected.pid
            || current.process_start_id != expected.process_start_id {
            return Err("Session worker stopped or changed generation while opening".into());
        }
        current.root_session_id = summary.get("sessionId").and_then(Value::as_str).map(str::to_string);
        current.session_file = summary.get("sessionFile").and_then(Value::as_str).map(str::to_string);
        current.lifecycle = DAEMON_WORKER_LIFECYCLE_READY.into();
        self.persist_worker(&current)
    }

    fn fail_created_worker(&self, worker: &Arc<Worker>, expected: &DaemonWorkerDescriptor, error: &str) {
        let mut workers = self.workers.lock().unwrap();
        if !workers.get(&expected.worker_id).is_some_and(|current| Arc::ptr_eq(current, worker)) { return; }
        let mut descriptor = worker.descriptor.lock().unwrap().clone();
        if descriptor.stop_requested_at.is_some() || descriptor.worker_instance_id != expected.worker_instance_id
            || descriptor.pid != expected.pid || descriptor.process_start_id != expected.process_start_id { return; }
        workers.remove(&descriptor.worker_id);
        descriptor.lifecycle = DAEMON_WORKER_LIFECYCLE_FAILED.into();
        descriptor.last_error = Some(error.into());
        descriptor.consecutive_failures += 1;
        let _ = self.persist_worker(&descriptor);
    }

    async fn launch_worker(self: &Arc<Self>, body: &Map<String, Value>, owner: String, existing: Option<DaemonWorkerDescriptor>) -> Result<Value, String> {
        self.ownership.assert_current().await.map_err(|error| error.to_string())?;
        let override_config = body.get("config").map(|value| serde_json::from_value::<AgentSessionRuntimeConfig>(value.clone())).transpose().map_err(|error| error.to_string())?;
        let config = merge_agent_session_runtime_config(&self.config, override_config.as_ref());
        let worker_id = existing.as_ref().map(|descriptor| descriptor.worker_id.clone()).unwrap_or_else(|| create_active_session_id(None));
        let root = existing.as_ref().map(|descriptor| descriptor.root_active_session_id.clone()).unwrap_or_else(|| create_active_session_id(None));
        let socket = existing.as_ref().map(|descriptor| descriptor.socket_path.clone()).unwrap_or_else(|| worker_socket(&self.socket_path, &worker_id));
        let mut token_bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut token_bytes);
        use base64::Engine;
        let token = existing.as_ref().map(|descriptor| descriptor.authentication_token.clone()).unwrap_or_else(|| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(token_bytes));
        let instance = uuid::Uuid::new_v4().to_string();
        let recovery = self.descriptor_dir.join(format!("{worker_id}.recovery.jsonl")).to_string_lossy().into_owned();
        let orphan = self.descriptor_dir.join(format!("{worker_id}.orphans.jsonl")).to_string_lossy().into_owned();
        let mut environment: HashMap<String, String> = std::env::vars().collect();
        if let Some(launch) = body.get("launchEnv").and_then(Value::as_object) {
            for (key, value) in launch { if let Some(value) = value.as_str() { environment.insert(key.clone(), value.to_string()); } }
        }
        for (key, value) in [(DAEMON_WORKER_ROLE_ENV, "1".to_string()), (DAEMON_WORKER_TOKEN_ENV, token.clone()),
            (DAEMON_WORKER_INSTANCE_ID_ENV, instance.clone()), (DAEMON_WORKER_ACTIVE_SESSION_ID_ENV, root.clone()),
            (DAEMON_WORKER_SUPERVISOR_SOCKET_ENV, self.socket_path.clone()), (DAEMON_WORKER_RECOVERY_JOURNAL_ENV, recovery.clone()),
            (SESSION_LEASES_ENABLED_ENV, "1".to_string()), (SESSION_LEASE_OWNER_ID_ENV, root.clone()),
            (crate::core::orphan_process_journal::ORPHAN_PROCESS_JOURNAL_ENV, orphan.clone())] { environment.insert(key.to_string(), value); }
        environment.remove("RLM_DEPTH"); environment.remove(DAEMON_CATALOG_ROLE_ENV);
        let (mut child, gate) = spawn_worker_process(
            &std::env::current_exe().map_err(|error| error.to_string())?.to_string_lossy(),
            vec![
                "--mode".to_string(),
                "daemon".to_string(),
                "--daemon-socket".to_string(),
                socket.clone(),
            ],
            &socket,
            config.cwd.as_deref(),
            environment,
        )?;
        let pid = child.id().ok_or("Failed to obtain daemon worker pid")?;
        // TS daemon-supervisor.ts:3339-3345: the worker stderr is piped and a
        // jsonl line reader forwards every line into the supervisor log
        // (bounded at 64KB per line). Inheriting stderr left it unobservable.
        if let Some(stderr) = child.stderr.take() {
            let worker_id = worker_id.clone();
            tokio::spawn(async move {
                use tokio::io::AsyncBufReadExt;
                let mut lines = tokio::io::BufReader::new(stderr);
                let mut line = String::new();
                loop {
                    line.clear();
                    match lines.read_line(&mut line).await {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            let text = line.trim_end_matches(['\r', '\n']);
                            if text.len() > 64 * 1024 {
                                let mut end = 64 * 1024;
                                while end > 0 && !text.is_char_boundary(end) {
                                    end -= 1;
                                }
                                eprintln!("Session worker {worker_id} stderr: {} [truncated]", &text[..end]);
                            } else {
                                eprintln!("Session worker {worker_id} stderr: {text}");
                            }
                        }
                    }
                }
            });
        }
        let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let descriptor = DaemonWorkerDescriptor {
            version: 2, worker_id, pid: pid as i32, process_start_id: get_process_start_id(pid as i64), socket_path: socket.clone(),
            recovery_journal_path: recovery, orphan_process_journal_path: Some(orphan), supervisor_socket_path: self.socket_path.clone(),
            authentication_token: token, worker_instance_id: Some(instance), root_active_session_id: root.clone(),
            owner_client_id: existing.as_ref().and_then(|descriptor| descriptor.owner_client_id.clone()).or_else(|| (body.get("lifecycle").and_then(Value::as_str) == Some("client_owned")).then_some(owner)),
            root_session_id: None, session_file: None, session_dir: config.session_dir.clone(), telemetry_disabled: config.telemetry_disabled,
            created_at: existing.as_ref().map(|descriptor| descriptor.created_at.clone()).unwrap_or_else(|| now.clone()), updated_at: now, lifecycle: "starting".into(),
            create_command: DurableDaemonCreateCommand { type_: "create".into(), session_path: body.get("sessionPath").and_then(Value::as_str).map(str::to_string), no_session: body.get("noSession").and_then(Value::as_bool), extra: Map::new() },
            consecutive_failures: 0, stop_requested_at: None, archive_on_stop: None, last_failure_at: None, last_error: None,
        };
        let mut launched_worker = None;
        let result: Result<Value, String> = async {
            self.persist_worker(&descriptor)?;
            let client = Arc::new(DaemonWorkerClient::new(&socket));
            // Listeners before authentication: the worker flushes its roster
            // snapshot immediately after auth succeeds.
            let worker = self.install_worker(descriptor.clone(), client);
            launched_worker = Some(Arc::clone(&worker));
            commit_gate(gate).await?;
            let client = { worker.client.lock().unwrap().clone().ok_or("Session worker is not connected")? };
            self.authenticate(&client, &descriptor).await?;
            let mut forwarded = body.clone(); forwarded.remove("id"); forwarded.remove("launchEnv"); forwarded.remove("lifecycle");
            forwarded.insert("config".into(), serde_json::to_value(config).map_err(|error| error.to_string())?);
            let response = client.request_worker(forwarded, REQUEST_TIMEOUT).await.map_err(|error| error.to_string())?;
            let summary = response_data(response)?;
            if summary.get("activeSessionId").or_else(|| summary.get("id")).and_then(Value::as_str) != Some(&root) { return Err("Session worker did not preserve its assigned active session id".into()); }
            self.ownership.assert_current().await.map_err(|error| error.to_string())?;
            self.mark_created_worker_ready(&worker, &descriptor, &summary)?;
            let summary_row = worker_roster_entry_from_value(&summary).ok_or("Session worker returned an invalid create response")?;
            self.write_roster_entry(summary_row, Some(&worker), None);
            self.subscribe(&worker, &root).await?;
            self.refresh(&worker).await?;
            let entry = self.roster().lock().unwrap().by_active_session_id(&root);
            let Some(entry) = entry else { return Err("Session worker started without a root session".into()); };
            Ok(serde_json::to_value(self.public_summary(&worker, summary_from_entry(&entry))).unwrap_or(Value::Null))
        }.await;
        if let Err(error) = &result {
            let _ = child.kill().await;
            if self.ownership.assert_current().await.is_ok() {
                if let Some(worker) = launched_worker.as_ref() { self.fail_created_worker(worker, &descriptor, error); }
            }
        }
        tokio::spawn(async move { let _ = child.wait().await; });
        result
    }
    /// The registration half of `parseCommandAndRegisterPromptAdmission` (daemon-supervisor.ts:1899-1919):
    /// validates the admission and stores a `waiting` record before dispatch can await anything.
    fn register_prompt_admission(&self, public: &PublicClient, body: &Map<String, Value>) -> Result<(), String> {
        if !matches!(body.get("type").and_then(Value::as_str), Some("prompt" | "prompt_and_wait")) {
            return Ok(());
        }
        let Some(admission_id) = body.get("admissionId") else { return Ok(()); };
        // `if (typeof command.activeSessionId !== "string" || typeof command.admissionId !== "string")
        // throw new Error("Prompt admission requires string activeSessionId and admissionId");`
        // (:1901-1903): a JSON `null` admissionId is present-but-not-a-string, so it fails here.
        let (Some(active_session_id), Some(public_admission_id)) =
            (body.get("activeSessionId").and_then(Value::as_str), admission_id.as_str())
        else {
            return Err("Prompt admission requires string activeSessionId and admissionId".into());
        };
        // `if (command.admissionId === "") throw new Error("admissionId must not be empty");` (:1904).
        if public_admission_id.is_empty() { return Err("admissionId must not be empty".into()); }
        let key = prompt_admission_key(&public.connection_id, active_session_id, public_admission_id);
        let mut admissions = self.prompt_admissions.lock().unwrap();
        // `if (admissions.has(key)) throw new Error(\`Prompt admission id is already in use:
        // ${command.admissionId}\`);` (:1907-1909).
        if admissions.contains_key(&key) {
            return Err(format!("Prompt admission id is already in use: {public_admission_id}"));
        }
        admissions.insert(key, PromptAdmissionRecord {
            connection_id: public.connection_id.clone(),
            active_session_id: active_session_id.to_string(),
            public_admission_id: public_admission_id.to_string(),
            // `workerAdmissionId: \`supervisor-admission:${randomUUID()}\`` (:1914): the id the
            // worker registers for this prompt, minted at supervisor parse time.
            worker_admission_id: format!("supervisor-admission:{}:{public_admission_id}", public.connection_id),
            status: PromptAdmissionStatus::Waiting,
            controller: CancellationToken::new(),
            worker: None,
            worker_active_session_id: None,
        });
        Ok(())
    }
    /// `familyCatalogEntries()` (daemon-supervisor.ts:4523-4538): every roster row, then the
    /// on-disk root sessions that no row covers.
    async fn family_catalog_entries(self: &Arc<Self>) -> Vec<AgentFamilyCatalogEntry> {
        let roster_rows = self.roster().lock().unwrap().values();
        let mut entries: Vec<AgentFamilyCatalogEntry> = roster_rows.iter()
            .map(|entry| self.family_catalog_entry(&summary_from_entry(entry))).collect();
        let known_files: HashSet<String> = roster_rows.iter()
            .filter_map(|entry| entry.summary.session_file.as_ref()).map(|file| canonical_session_path(file)).collect();
        let scanned = match self.catalog.list(None, self.config.session_dir.as_deref(), None).await {
            Ok(scanned) => scanned,
            // `const scanned = await this.catalog.list(undefined, ...)` (:4531) is awaited without a
            // catch, so a catalog failure fails the check rather than silently passing it.
            Err(error) => { eprintln!("Could not scan saved sessions for name availability: {error}"); Vec::new() }
        };
        for info in scanned {
            if known_files.contains(&canonical_session_path(&info.path)) { continue; }
            // `if ((info.rlmDepth ?? (info.parentSessionPath ? -1 : 0)) !== 0) continue;` (:4534).
            let depth = if info.rlm_depth != 0 { info.rlm_depth } else if info.parent_session_path.is_some() { -1 } else { 0 };
            if depth != 0 { continue; }
            entries.push(self.family_catalog_entry(&summary_for_inactive_session(&info, false, false)));
        }
        entries
    }
    /// `summaryNameReservationInput(target, name)` (daemon-supervisor.ts:4991-5002).
    fn summary_name_reservation_input(target: &SessionSummary, name: &str) -> (AgentSessionNameScope, String) {
        let depth = target.rlm_depth.map(|depth| depth as f64).unwrap_or(if target.parent_session_path.is_some() { 1.0 } else { 0.0 });
        let scope = AgentSessionNameScope {
            parent_session_id: (depth > 0.0).then(|| target.parent_session_id.clone()).flatten(),
            parent_session_path: (depth > 0.0).then(|| target.parent_session_path.clone()).flatten(),
            depth,
        };
        (scope, name.to_string())
    }
    /// `withSessionNameReservation(input, action)` (daemon-supervisor.ts:4540-4554): hold the key
    /// for the duration of the action and always release it.
    async fn with_session_name_reservation<T, F>(self: &Arc<Self>, scope: &AgentSessionNameScope, name: &str, action: F) -> Result<T, String>
    where F: std::future::Future<Output = Result<T, String>> {
        let key = session_name_reservation_key(scope, name);
        {
            let mut pending = self.pending_session_names.lock().unwrap();
            if pending.contains(&key) { return Err(format_agent_session_name_unavailable(name, scope.depth)); }
            pending.insert(key.clone());
        }
        let result = action.await;
        self.pending_session_names.lock().unwrap().remove(&key);
        result
    }
    /// `assertSavedSiblingNameAvailable(siblings, target, name)` (daemon-supervisor.ts:5018-5041):
    /// the sibling list is the catalog, all rows at the target's depth, and the roster row for each
    /// path supplies its status and name when it has one (:5023, 5028).
    async fn assert_saved_sibling_name_available(self: &Arc<Self>, siblings: &[SessionInfo], target: &SessionInfo, name: &str) -> Result<(), String> {
        let set_depth = target.rlm_depth as f64;
        let catalog: Vec<AgentFamilyCatalogEntry> = siblings.iter().map(|info| {
            let summary = summary_for_inactive_session(info, false, false);
            let path = canonical_session_path(&info.path);
            let ledger_row = self.roster().lock().unwrap().by_session_file(&path);
            AgentFamilyCatalogEntry {
                id: summary.session_id.clone(),
                name: summary.session_name.clone().or_else(|| ledger_row.as_ref().and_then(|row| row.summary.session_name.clone())),
                depth: set_depth,
                status: ledger_row.as_ref().map(|row| row.status.as_str().to_string())
                    .unwrap_or_else(|| classify_session_roster_status(
                        &RosterSummaryView { active_session_id: summary.active_session_id.clone(), activity: Some(summary.activity.clone()), is_session_active: Some(summary.is_session_active) }, false).as_str().to_string()),
                replied_since_task: None,
                parent_session_id: None,
                parent_session_path: summary.parent_session_path.as_ref().map(|path| canonical_session_path(path)),
                session_path: summary.session_file.as_ref().map(|file| canonical_session_path(file)),
            }
        }).collect();
        assert_agent_session_name_available(&catalog, &AgentSessionNameAvailabilityInput {
            parent_session_id: None,
            // `parentSessionPath: target.parentSessionPath ? canonicalSessionPath(target.parentSessionPath)
            // : undefined` (:5037).
            parent_session_path: target.parent_session_path.as_ref().map(|path| canonical_session_path(path)),
            depth: set_depth,
            name: name.to_string(),
            // `ignoreSessionId: target.id` (:5038).
            ignore_session_id: Some(target.id.clone()),
        })
    }
    /// `assertSupervisorSessionNameAvailable(target, name)` (daemon-supervisor.ts:4556-4567).
    async fn assert_supervisor_session_name_available(self: &Arc<Self>, target: &SessionSummary, name: &str) -> Result<(), String> {
        let catalog = self.family_catalog_entries().await;
        assert_agent_session_name_available(&catalog, &AgentSessionNameAvailabilityInput {
            parent_session_id: target.parent_session_id.clone(),
            parent_session_path: target.parent_session_path.as_ref().map(|path| canonical_session_path(path)),
            depth: target.rlm_depth.map(|depth| depth as f64).unwrap_or(0.0),
            name: name.to_string(),
            ignore_session_id: Some(target.session_id.clone()),
        })
    }
    /// `getPromptAdmission(client, activeSessionId, publicAdmissionId)` (daemon-supervisor.ts:1835-1841).
    fn get_prompt_admission(
        &self,
        connection_id: &str,
        active_session_id: &str,
        public_admission_id: &str,
    ) -> Option<PromptAdmissionRecord> {
        self.prompt_admissions
            .lock()
            .unwrap()
            .get(&prompt_admission_key(connection_id, active_session_id, public_admission_id))
            .cloned()
    }
    /// `deletePromptAdmission(admission)` (daemon-supervisor.ts:1843-1849).
    fn delete_prompt_admission(&self, admission: &PromptAdmissionRecord) {
        self.prompt_admissions.lock().unwrap().remove(&prompt_admission_key(
            &admission.connection_id,
            &admission.active_session_id,
            &admission.public_admission_id,
        ));
    }
    /// `cancelWaitingPromptAdmissionsForClient(client)` (daemon-supervisor.ts:1851-1881), called
    /// from the socket-close cleanup (:1681). A still-queued admission is cancelled locally
    /// (:1854-1857); one already attached to a worker is cancelled through that worker and its
    /// status is only downgraded while it is still `waiting` (:1866-1874).
    fn cancel_waiting_prompt_admissions_for_client(self: &Arc<Self>, public: &Arc<PublicClient>) {
        let waiting: Vec<PromptAdmissionRecord> = self.prompt_admissions.lock().unwrap().values()
            .filter(|admission| admission.connection_id == public.connection_id && admission.status == PromptAdmissionStatus::Waiting)
            .cloned().collect();
        for admission in waiting {
            let (Some(worker), Some(active)) = (admission.worker.clone(), admission.worker_active_session_id.clone()) else {
                // `if (!admission.worker || !admission.workerActiveSessionId) { admission.status =
                // "cancelled"; admission.controller.abort(); }` (:1854-1857).
                self.set_prompt_admission_status(&admission, PromptAdmissionStatus::Cancelled, true);
                continue;
            };
            let supervisor = Arc::clone(self);
            tokio::spawn(async move {
                let mut forwarded = command("cancel_prompt_admission");
                forwarded.insert("activeSessionId".into(), json!(active));
                forwarded.insert("admissionId".into(), json!(admission.worker_admission_id));
                let Ok(client) = supervisor.connected_client(&worker).await else { return; };
                let Ok(response) = client.request_worker(forwarded, REQUEST_TIMEOUT).await else { return; };
                // `if (admission.status !== "waiting") return;` (:1867).
                if supervisor.prompt_admission_status(&admission) != Some(PromptAdmissionStatus::Waiting) { return; }
                match response.data.as_ref().and_then(|data| data.get("status")).and_then(Value::as_str) {
                    Some("owned") => supervisor.set_prompt_admission_status(&admission, PromptAdmissionStatus::Owned, false),
                    Some("cancelled") => supervisor.set_prompt_admission_status(&admission, PromptAdmissionStatus::Cancelled, false),
                    _ => {}
                }
            });
        }
    }
    /// `admission.worker = match.worker; admission.workerActiveSessionId =
    /// match.summary.activeSessionId ?? match.summary.id;` (daemon-supervisor.ts:2806-2808).
    fn attach_prompt_admission_worker(&self, admission: &PromptAdmissionRecord, worker: &Arc<Worker>, active: &str) {
        let key = prompt_admission_key(&admission.connection_id, &admission.active_session_id, &admission.public_admission_id);
        if let Some(current) = self.prompt_admissions.lock().unwrap().get_mut(&key) {
            current.worker = Some(Arc::clone(worker));
            current.worker_active_session_id = Some(active.to_string());
        }
    }
    /// The status of a record, matched by its own key.
    fn prompt_admission_status(&self, admission: &PromptAdmissionRecord) -> Option<PromptAdmissionStatus> {
        self.prompt_admissions.lock().unwrap()
            .get(&prompt_admission_key(&admission.connection_id, &admission.active_session_id, &admission.public_admission_id))
            .map(|current| current.status)
    }
    fn set_prompt_admission_status(&self, admission: &PromptAdmissionRecord, status: PromptAdmissionStatus, abort: bool) {
        let key = prompt_admission_key(&admission.connection_id, &admission.active_session_id, &admission.public_admission_id);
        if let Some(current) = self.prompt_admissions.lock().unwrap().get_mut(&key) { current.status = status; }
        if abort { admission.controller.cancel(); }
    }
    /// `promoteOwnedWorker(client, worker)` (daemon-supervisor.ts:3241-3276).
    ///
    /// Clears `descriptor.ownerClientId` (:3250-3253) and persists the descriptor (:3255), then
    /// records the promotion (:3266) and re-amends the worker's roster rows (:3267-3269) so the
    /// previously private rows become visible to every client.
    fn promote_owned_worker(&self, worker: &Arc<Worker>, client_id: &str) -> Result<(), String> {
        // `if (worker.descriptor.ownerClientId === undefined && worker.promotedOwnerClientId ===
        // clientId) return;` (:3243-3245).
        if worker.descriptor.lock().unwrap().owner_client_id.is_none()
            && worker.promoted_owner_client_id.lock().unwrap().as_deref() == Some(client_id)
        {
            return Ok(());
        }
        // `if (worker.descriptor.ownerClientId !== clientId) throw new Error("Session is not owned
        // by this client");` (:3246-3248).
        if worker.descriptor.lock().unwrap().owner_client_id.as_deref() != Some(client_id) {
            return Err("Session is not owned by this client".into());
        }
        {
            let mut descriptor = worker.descriptor.lock().unwrap();
            descriptor.owner_client_id = None;
            self.persist_worker(&descriptor)?;
        }
        *worker.promoted_owner_client_id.lock().unwrap() = Some(client_id.to_string());
        // `for (const entry of this.workerRosterEntries(worker)) this.roster().amend(entry.agentId, {});`
        // (:3267-3269): the empty marks still announce the row, so clients re-read it.
        for entry in self.worker_roster_entries(worker) {
            self.roster().lock().unwrap().amend(&entry.agent_id, RosterEntryMarks::default());
        }
        Ok(())
    }
    /// `matchWorkers(selector, includeWorker)` over the roster, then
    /// `collectedPassiveScheduledJob { rootSessionFile, job, info }` (daemon-supervisor.ts:962).
    ///
    /// `collectPassiveScheduledJobs(includeInactive)` (daemon-supervisor.ts:960-1026): the
    /// durable scheduled-job truth for sessions with no resident worker, so a wake timer and
    /// `cron_list` can see jobs of passivated trees. The ledger family supplies the sessions
    /// and their parents; a session is "uncovered" when no worker holds its path.
    ///
    async fn collect_passive_scheduled_jobs(self: &Arc<Self>, include_inactive: bool) -> Vec<(String, Value)> {
        let cancelled_roots = self.settle_ephemeral_cancel_intents().await;
        let ledger = match self.rlm_spawn_ledger().await { Ok(ledger) => ledger, Err(_) => return Vec::new() };
        let infos = ledger.family().await;
        let info_by_path: HashMap<String, SessionInfo> = infos.iter()
            .map(|info| (canonical_session_path(&info.path), info.clone()))
            .collect();
        let mut results: Vec<(String, Value)> = Vec::new();
        let parent_map: HashMap<String, String> = infos.iter().filter_map(|info| info.parent_session_path.as_ref().map(|parent| (canonical_session_path(&info.path), canonical_session_path(parent)))).collect();
        for info in &infos {
            if roster_path_descends_from(&parent_map, &canonical_session_path(&info.path), &cancelled_roots) { continue; }
            if info.state.as_ref().is_some_and(|state| state.status != crate::core::session_manager::SessionStateStatus::Active) { continue; }
            let artifact_dir = crate::core::session_manager::get_session_artifact_path_for_file(&resolve_path(&info.path), Some(&info.id));
            if !Path::new(&artifact_dir).join(crate::core::cron_jobs::SESSION_SCHEDULED_JOBS_FILENAME).exists() { continue; }
            let store = crate::core::cron_jobs::AgentCronJobStore::for_session_artifacts();
            if !store.register_session_artifact(&info.id, &artifact_dir) { continue; }
            // TS wraps this in `try/catch` and skips unreadable jobs (1009-1014); the ported
            // `AgentCronJobStore::list()` returns just the jobs and has no failure channel, so
            // there is nothing to catch here.
            let jobs = store.list();
            for job in jobs {
                if !include_inactive && job.status != crate::core::cron_jobs::STATUS_ACTIVE && job.status != crate::core::cron_jobs::STATUS_PAUSED { continue; }
                // `uncoveredRootFor(info)`: walk up while no worker holds the current path.
                let root = {
                    let mut cursor = info.clone();
                    let mut visited: HashSet<String> = HashSet::from([canonical_session_path(&cursor.path)]);
                    loop {
                        if self.find_worker_by_session_file(&cursor.path, None).is_some() { break; }
                        let Some(parent_path) = cursor.parent_session_path.clone() else { break; };
                        let Some(parent) = info_by_path.get(&canonical_session_path(&parent_path)) else { break; };
                        if !visited.insert(canonical_session_path(&parent.path)) { break; }
                        cursor = parent.clone();
                    }
                    cursor
                };
                let current = root;
                if self.find_worker_by_session_file(&current.path, None).is_some() { continue; }
                results.push((current.path.clone(), serde_json::to_value(&job).unwrap_or(Value::Null)));
            }
        }
        results
    }
    /// `findWorkerBySessionFile(sessionFile, exclude)` (daemon-supervisor.ts:5316-5340).
    /// Conflicting resident paths are reported as an error, exactly like the TS throw.
    fn find_worker_by_session_file(&self, session_file: &str, exclude: Option<&Arc<Worker>>) -> Option<Arc<Worker>> {
        let target = canonical_session_path(session_file);
        let target_entry_worker = self.roster().lock().unwrap().by_session_file(&target).and_then(|entry| entry.worker_id);
        let mut matches: Vec<Arc<Worker>> = Vec::new();
        for worker in self.workers.lock().unwrap().values() {
            if exclude.is_some_and(|exclude| Arc::ptr_eq(exclude, worker)) { continue; }
            let descriptor = worker.descriptor.lock().unwrap();
            let summary_match = target_entry_worker.as_deref() == Some(descriptor.worker_id.as_str());
            let descriptor_path = descriptor.session_file.as_ref().map(|file| canonical_session_path(file));
            let configured_path = descriptor.create_command.session_path.as_ref().map(|file| canonical_session_path(file));
            if !summary_match && descriptor_path.as_deref() != Some(target.as_str()) && configured_path.as_deref() != Some(target.as_str()) { continue; }
            if descriptor_path.is_some() && configured_path.is_some() && descriptor_path != configured_path {
                return None;
            }
            matches.push(Arc::clone(worker));
        }
        if matches.len() > 1 { return None; }
        matches.into_iter().next()
    }
    /// `cancelPassiveScheduledJob`: `AgentCronJobStore.forSessionArtifacts().registerSessionArtifact(...).cancel(id)`
    /// (daemon-supervisor.ts:2618-2630).
    async fn cancel_passive_scheduled_job(self: &Arc<Self>, job_id: &str) -> Option<Value> {
        let jobs = self.collect_passive_scheduled_jobs(true).await;
        let job = jobs.into_iter().map(|(_, job)| job).find(|job| job.get("id").and_then(Value::as_str) == Some(job_id))?;
        let session_file = job.get("sessionFile").and_then(Value::as_str)?.to_string();
        let session_id = job.get("sessionId").and_then(Value::as_str)?.to_string();
        let store = crate::core::cron_jobs::AgentCronJobStore::for_session_artifacts();
        store.register_session_artifact(&session_id, &crate::core::session_manager::get_session_artifact_path_for_file(&resolve_path(&session_file), Some(&session_id)));
        store.cancel(job_id, supervisor_now_ms() as f64).and_then(|job| serde_json::to_value(job).ok())
    }
    /// `isLiveWorker(worker)` (daemon-supervisor.ts:5053-5055).
    fn is_live_worker(&self, worker: &Arc<Worker>) -> bool {
        self.is_visible_worker(worker) && worker.descriptor.lock().unwrap().stop_requested_at.is_none()
    }
    /// `findWorker`'s recovery fallback and its typed recovering error.
    async fn find(self: &Arc<Self>, identity: &str, requested: &str) -> Result<(Arc<Worker>, String), String> {
        let mut matches = self.match_workers(requested, Some(&|worker: &Arc<Worker>| visible(worker, identity)));
        if matches.is_empty() {
            let workers: Vec<Arc<Worker>> = self.workers.lock().unwrap().values().cloned().collect();
            for worker in workers { let _ = self.refresh(&worker).await; }
            matches = self.match_workers(requested, Some(&|worker: &Arc<Worker>| visible(worker, identity)));
        }
        match matches.len() {
            1 => return Ok(matches.remove(0)),
            0 => {}
            _ => return Err(format!("Ambiguous active session \"{requested}\"")),
        }
        // Descriptors are the durable half of addressability: an unhydrated root is
        // recovering, not unknown; failed workers stay unknown so clients take the
        // create fallback, which reclaims or retries them.
        let recovering: Vec<Arc<Worker>> = self.workers.lock().unwrap().values()
            .filter(|worker| visible(worker, identity))
            .filter(|worker| {
                let descriptor = worker.descriptor.lock().unwrap();
                descriptor.lifecycle != DAEMON_WORKER_LIFECYCLE_FAILED
                    && descriptor.stop_requested_at.is_none()
                    && worker.client.lock().unwrap().is_none()
                    && (descriptor.root_active_session_id == requested || descriptor.root_session_id.as_deref() == Some(requested))
            })
            .cloned().collect();
        let recovering = if recovering.is_empty() {
            self.workers.lock().unwrap().values().filter(|worker| visible(worker, identity)).filter(|worker| {
                let descriptor = worker.descriptor.lock().unwrap();
                descriptor.lifecycle != DAEMON_WORKER_LIFECYCLE_FAILED
                    && descriptor.stop_requested_at.is_none()
                    && worker.client.lock().unwrap().is_none()
                    && (matches_session_id_suffix(&descriptor.root_active_session_id, requested)
                        || descriptor.root_session_id.as_deref().is_some_and(|root| matches_session_id_suffix(root, requested)))
            }).cloned().collect()
        } else { recovering };
        if let [worker] = recovering.as_slice() {
            return Err(DaemonSessionRecoveringError::new(worker.descriptor.lock().unwrap().root_active_session_id.clone()).message());
        }
        Err(format!("Unknown active session: {requested}"))
    }
    /// `familyCatalogEntry(summary)` (daemon-supervisor.ts:5148-5161).
    fn family_catalog_entry(&self, summary: &SessionSummary) -> AgentFamilyCatalogEntry {
        let depth = summary.rlm_depth.unwrap_or(if summary.parent_session_path.is_some() { 1 } else { 0 });
        let status = summary.roster_status.unwrap_or_else(|| classify_session_roster_status(
            &RosterSummaryView {
                active_session_id: summary.active_session_id.clone(),
                activity: Some(summary.activity.clone()),
                is_session_active: Some(summary.is_session_active),
            },
            false,
        ));
        AgentFamilyCatalogEntry {
            id: summary.session_id.clone(),
            name: summary.session_name.clone(),
            depth: depth as f64,
            status: status.as_str().to_string(),
            replied_since_task: None,
            parent_session_id: (depth > 0).then(|| summary.parent_session_id.clone()).flatten(),
            parent_session_path: (depth > 0).then(|| summary.parent_session_path.as_ref().map(|path| canonical_session_path(path))).flatten(),
            session_path: summary.session_file.as_ref().map(|file| canonical_session_path(file)),
        }
    }
    /// The existing peer-discovery wire shape, projected only from supervisor-owned roster data.
    fn agent_peer_summary(&self, summary: &SessionSummary) -> AgentSessionMessageAgentSummary {
        let actions = summary.session_actions.as_ref();
        let unfinished = summary.unfinished_action_count.unwrap_or_else(|| {
            actions.and_then(|value| value.get("queuedCount")).and_then(Value::as_i64).unwrap_or(0)
                + i64::from(actions.and_then(|value| value.get("active")).is_some_and(|value| !value.is_null()))
        });
        AgentSessionMessageAgentSummary {
            active_session_id: summary.active_session_id.clone().unwrap_or_else(|| summary.id.clone()),
            session_id: summary.session_id.clone(),
            session_name: summary.session_name.clone(),
            runtime_kind: Some(summary.runtime_kind.clone().unwrap_or_else(|| "top-level".into())),
            cwd: summary.cwd.clone(),
            is_streaming: summary.is_streaming,
            unfinished_action_count: unfinished as f64,
            parent_active_session_id: summary.parent_active_session_id.clone(),
            parent_session_id: summary.parent_session_id.clone(),
            parent_session_path: summary.parent_session_path.clone(),
            session_path: summary.session_file.clone(),
            rlm_depth: summary.rlm_depth.map(|depth| depth as f64),
            status: Some(self.family_catalog_entry(summary).status),
            rlm_child_id: summary.rlm_child_id.clone(),
            ..Default::default()
        }
    }
    /// The roster summary for a resolved active session id, as the a2a send path needs it
    /// (`target.summary`, daemon-supervisor.ts:2746-2753).
    fn summary_for_active(&self, worker: &Arc<Worker>, active: &str) -> Option<SessionSummary> {
        let entry = self.roster().lock().unwrap().by_active_session_id(active)?;
        Some(self.public_summary(worker, summary_from_entry(&entry)))
    }
    /// `matchWorkers(selector, includeWorker)`: exact ids win, hex suffixes second.
    fn match_workers(&self, selector: &str, include_worker: Option<&dyn Fn(&Arc<Worker>) -> bool>) -> Vec<(Arc<Worker>, String)> {
        let mut exact: Vec<(Arc<Worker>, String)> = Vec::new();
        let mut suffix: Vec<(Arc<Worker>, String)> = Vec::new();
        for entry in self.roster().lock().unwrap().values() {
            if entry.queued_child == Some(true) { continue; }
            let worker = entry.worker_id.as_ref().and_then(|worker_id| self.workers.lock().unwrap().get(worker_id).cloned());
            let Some(worker) = worker else { continue; };
            if include_worker.is_some_and(|include| !include(&worker)) { continue; }
            let summary = self.public_summary(&worker, summary_from_entry(&entry));
            let active_session_id = summary.active_session_id.clone().unwrap_or_else(|| summary.id.clone());
            if active_session_id == selector || summary.session_id == selector || summary.session_name.as_deref() == Some(selector) {
                exact.push((Arc::clone(&worker), active_session_id));
            } else if matches_session_id_suffix(&active_session_id, selector) || matches_session_id_suffix(&summary.session_id, selector) {
                suffix.push((Arc::clone(&worker), active_session_id));
            }
        }
        if exact.is_empty() { suffix } else { exact }
    }
    /// `findSummaryInWorker(worker, selector)`.
    fn find_summary_in_worker(&self, worker: &Arc<Worker>, selector: &str) -> Option<SessionSummary> {
        let path_selector = looks_like_session_path(selector).then(|| canonical_session_path(selector));
        let summaries: Vec<SessionSummary> = self.worker_roster_entries(worker).into_iter()
            .filter(|entry| entry.queued_child != Some(true))
            .map(|entry| self.public_summary(worker, summary_from_entry(&entry)))
            .collect();
        let exact = summaries.iter().find(|summary| {
            let active_session_id = summary.active_session_id.clone().unwrap_or_else(|| summary.id.clone());
            active_session_id == selector
                || summary.session_id == selector
                || summary.session_name.as_deref() == Some(selector)
                || (path_selector.is_some() && summary.session_file.as_ref().map(|file| canonical_session_path(file)) == path_selector)
        });
        if let Some(exact) = exact { return Some(exact.clone()); }
        summaries.into_iter().find(|summary| {
            let active_session_id = summary.active_session_id.clone().unwrap_or_else(|| summary.id.clone());
            matches_session_id_suffix(&active_session_id, selector) || matches_session_id_suffix(&summary.session_id, selector)
        })
    }
    async fn release_client_pauses(&self, public: &PublicClient, active: Option<&str>) -> Result<(), String> {
        public.pause_epoch.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let entries: Vec<_> = self.pauses.lock().unwrap().iter().filter(|(_, pause)| pause.connection_id == public.connection_id && active.map_or(true, |active| active == pause.active || active == pause.requested)).map(|(id, pause)| (id.clone(), pause.clone())).collect();
        for (id, pause) in entries {
            let mut body = command("release_session_input_pause");
            body.insert("activeSessionId".into(), json!(pause.active)); body.insert("pauseId".into(), json!(id));
            let client = { pause.worker.client.lock().unwrap().clone().ok_or("Session worker is not connected")? };
            response_data(client.request_worker(body, 5_000).await.map_err(|error| error.to_string())?)?;
            self.pauses.lock().unwrap().remove(&id);
        }
        Ok(())
    }
        /// `publicSummary(worker, summary)`: attached clients plus the worker's
    /// effective lifecycle, which never reports a disconnected worker as ready.
    fn public_summary(&self, worker: &Worker, summary: SessionSummary) -> SessionSummary {
        let active = summary.active_session_id.clone().unwrap_or_else(|| summary.id.clone());
        let mut summary = summary;
        summary.attached_clients = self.attached_client_count(&summary, &active);
        summary.worker_state = Some(self.effective_worker_state(worker));
        summary.worker_pid = Some(worker.descriptor.lock().unwrap().pid as i64);
        summary
    }
    /// `attachedClientCount(summary, activeSessionId)`.
    fn attached_client_count(&self, summary: &SessionSummary, active_session_id: &str) -> i64 {
        let direct = summary.direct_attached_clients.unwrap_or(0);
        let supervisor = self.clients.lock().unwrap().values()
            .filter(|client| client.subscriptions.lock().unwrap().contains(active_session_id))
            .count() as i64;
        direct + supervisor
    }
    /// `effectiveWorkerState(worker)`.
    fn effective_worker_state(&self, worker: &Worker) -> String {
        let descriptor = worker.descriptor.lock().unwrap().clone();
        if descriptor.stop_requested_at.is_some() { return DAEMON_WORKER_LIFECYCLE_STOPPING.to_string(); }
        let connected = worker.client.lock().unwrap().is_some();
        if descriptor.lifecycle == DAEMON_WORKER_LIFECYCLE_READY && !connected { return DAEMON_WORKER_LIFECYCLE_RECOVERING.to_string(); }
        descriptor.lifecycle
    }
    /// `this.roster()`: the one store per supervisor, created at startup so its
    /// mutation sink can hold a real `Weak` to this supervisor.
    fn roster(&self) -> Arc<Mutex<AgentRoster>> {
        self.roster.lock().unwrap().clone().expect("supervisor roster is created at startup")
    }
    fn init_roster(self: &Arc<Self>) {
        let mut slot = self.roster.lock().unwrap();
        if slot.is_some() { return; }
        let weak = Arc::downgrade(self);
        *slot = Some(Arc::new(Mutex::new(AgentRoster::new(
            Box::new(canonical_session_path),
            Box::new(move |mutation| {
                if let Some(supervisor) = weak.upgrade() { supervisor.on_roster_mutation(mutation); }
            }),
        ))));
    }
    /// `onRosterMutation(mutation)`.
    fn on_roster_mutation(self: &Arc<Self>, mutation: AgentRosterMutation) {
        match mutation {
            AgentRosterMutation::Delete { agent_id } => {
                self.pending_roster_changed.lock().unwrap().remove(&agent_id);
                self.pending_roster_removed.lock().unwrap().insert(agent_id);
            }
            AgentRosterMutation::Write { agent_id } => {
                self.pending_roster_removed.lock().unwrap().remove(&agent_id);
                self.pending_roster_changed.lock().unwrap().insert(agent_id);
            }
        }
        self.schedule_roster_push();
    }
    /// `scheduleRosterPush()`.
    fn schedule_roster_push(self: &Arc<Self>) {
        if self.roster_push_scheduled.swap(true, Ordering::SeqCst) { return; }
        let supervisor = Arc::clone(self);
        tokio::spawn(async move {
            supervisor.roster_push_scheduled.store(false, Ordering::SeqCst);
            if supervisor.stopped.is_cancelled() { return; }
            supervisor.flush_roster_updates();
        });
    }
    /// `flushRosterUpdates()`.
    fn flush_roster_updates(&self) {
        let mut changed: Vec<AgentRosterEntry> = Vec::new();
        let mut removed: Vec<String> = Vec::new();
        let removed_ids: Vec<String> = self.pending_roster_removed.lock().unwrap().iter().cloned().collect();
        for agent_id in removed_ids {
            if self.published_roster_ids.lock().unwrap().remove(&agent_id) { removed.push(agent_id); }
        }
        let changed_ids: Vec<String> = self.pending_roster_changed.lock().unwrap().iter().cloned().collect();
        for agent_id in changed_ids {
            let entry = self.roster().lock().unwrap().get(&agent_id);
            let Some(entry) = entry else { continue; };
            if self.is_roster_entry_visible_to_clients(&entry) {
                changed.push(entry);
                self.published_roster_ids.lock().unwrap().insert(agent_id);
            } else if self.published_roster_ids.lock().unwrap().remove(&agent_id) {
                removed.push(agent_id);
            }
        }
        self.pending_roster_changed.lock().unwrap().clear();
        self.pending_roster_removed.lock().unwrap().clear();
        if changed.is_empty() && removed.is_empty() { return; }
        for client in self.clients.lock().unwrap().values() {
            if !client.roster_subscribed.load(Ordering::SeqCst) { continue; }
            if client.backpressured.load(Ordering::SeqCst) {
                client.roster_resync_pending.store(true, Ordering::SeqCst);
                continue;
            }
            client.write(&roster_update_value(&changed, &removed, None));
        }
    }
    /// `rosterEntriesForClient()`.
    fn roster_entries_for_client(&self) -> Vec<AgentRosterEntry> {
        let entries: Vec<AgentRosterEntry> = self.roster().lock().unwrap().values()
            .into_iter().filter(|entry| self.is_roster_entry_visible_to_clients(entry)).collect();
        for entry in &entries { self.published_roster_ids.lock().unwrap().insert(entry.agent_id.clone()); }
        entries
    }
    /// `isRosterEntryVisibleToClients(entry)`.
    fn is_roster_entry_visible_to_clients(&self, entry: &AgentRosterEntry) -> bool {
        let worker = entry.worker_id.as_ref().and_then(|worker_id| self.workers.lock().unwrap().get(worker_id).cloned());
        worker.is_none_or(|worker| self.is_visible_worker(&worker))
    }
    /// `isVisibleWorker(worker)`: client-owned workers are private to their owner.
    fn is_visible_worker(&self, worker: &Worker) -> bool {
        worker.descriptor.lock().unwrap().owner_client_id.is_none()
    }
    /// `workerRosterEntries(worker)`.
    fn worker_roster_entries(&self, worker: &Worker) -> Vec<AgentRosterEntry> {
        let worker_id = worker.descriptor.lock().unwrap().worker_id.clone();
        self.roster().lock().unwrap().entries_for_worker(&worker_id)
    }
    /// `writeRosterEntry(entry, worker?, statusLabel?)`.
    fn write_roster_entry(
        &self,
        entry: WorkerRosterEntry,
        worker: Option<&Arc<Worker>>,
        status_label: Option<&str>,
    ) -> AgentRosterEntry {
        let previous_direct = self.roster().lock().unwrap().get(&entry.agent_id)
            .and_then(|existing| existing.summary.direct_attached_clients).unwrap_or(0);
        let worker_id = worker.map(|worker| worker.descriptor.lock().unwrap().worker_id.clone());
        let stored = self.roster().lock().unwrap().write(entry.clone(), worker_id.as_deref(), status_label);
        // Direct peers attach and detach on the worker socket, so their last detach
        // arrives here as roster truth instead of through a supervisor-socket close.
        //
        // blocked_on: the TS follow-up for that case is
        // `evictEmptySessionOnLastDetach`, which needs the idle-eviction fence and
        // the passivate chain (daemon_mode.rs owns both). The row transition itself
        // is applied; only the eviction is deferred.
        let _ = (previous_direct, worker, &stored);
        stored
    }
    /// `rlmSpawnLedger()`: one shared instance per supervisor.
    async fn rlm_spawn_ledger(&self) -> Result<Arc<RlmSpawnLedger>, String> {
        let agent_dir = self.config.agent_dir.clone().ok_or("Daemon supervisor config is missing agentDir")?;
        let session_dir = self.config.session_dir.clone().unwrap_or_else(|| crate::config::get_sessions_dir(Some(&agent_dir)));
        self.ledger.get_or_try_init(|| async {
            Ok(Arc::new(RlmSpawnLedger::new(
                &agent_dir,
                &session_dir,
                Some(create_rlm_ledger_registry_seed_source()),
                Some(Arc::new(|message: &str| eprintln!("{message}"))),
            )))
        }).await.cloned()
    }
    async fn rlm_spawn_ledger_for(&self, session_dir: Option<&str>) -> Result<Arc<RlmSpawnLedger>, String> {
        let Some(session_dir) = session_dir else { return self.rlm_spawn_ledger().await; };
        let agent_dir = self.config.agent_dir.clone().ok_or("Daemon supervisor config is missing agentDir")?;
        let default_dir = self.config.session_dir.clone().unwrap_or_else(|| crate::config::get_sessions_dir(Some(&agent_dir)));
        if resolve_path(session_dir) == resolve_path(&default_dir) {
            return self.rlm_spawn_ledger().await;
        }
        Ok(Arc::new(RlmSpawnLedger::new(
            &agent_dir,
            session_dir,
            Some(create_rlm_ledger_registry_seed_source()),
            Some(Arc::new(|message: &str| eprintln!("{message}"))),
        )))
    }
    /// `scheduleWorkerStopFinalization(worker)` (daemon-supervisor.ts:6890-6897): one finalizer
    /// per worker. TS keeps the in-flight promise on the worker registration; this port keeps the
    /// same guard in `stop_finalizations()` (a process-wide set, one supervisor per process), so a
    /// repeated timeout does not stack escalation loops and no registration field is needed.
    fn schedule_worker_stop_finalization(self: &Arc<Self>, worker: &Arc<Worker>) {
        let descriptor = worker.descriptor.lock().unwrap().clone();
        if stop_cleanup_is_parked(&descriptor) { return; }
        let worker_id = descriptor.worker_id.clone();
        {
            let mut scheduled = stop_finalizations().lock().unwrap();
            if !scheduled.insert(worker_id.clone()) { return; }
        }
        tokio::spawn(finalize_timed_out_worker_stop(Arc::clone(self), Arc::clone(worker), descriptor));
    }
    /// `stopWorker(worker, removeDescriptor, force, archiveSession)` (daemon-supervisor.ts:6689).
    /// Only the `shutdown` command's `force` flag reaches here (`forceWorkers`, 2419/7326); every
    /// other caller keeps the non-forcing default of the port.
    async fn stop_worker(self: &Arc<Self>, worker: &Arc<Worker>, archive: bool, force: bool) -> Result<(), String> {
        self.ownership.assert_current().await.map_err(|error| error.to_string())?;
        let worker_id = worker.descriptor.lock().unwrap().worker_id.clone();
        if !self.workers.lock().unwrap().get(&worker_id).is_some_and(|registered| Arc::ptr_eq(registered, worker)) {
            return Err(format!("Session worker {worker_id} was replaced during stop"));
        }
        let descriptor = {
            let mut descriptor = worker.descriptor.lock().unwrap();
            descriptor.lifecycle = DAEMON_WORKER_LIFECYCLE_STOPPING.into();
            descriptor.stop_requested_at.get_or_insert_with(|| chrono::Utc::now().to_rfc3339());
            descriptor.archive_on_stop = Some(archive);
            self.persist_worker(&descriptor)?;
            descriptor.clone()
        };
        let assert_stop_current = || {
            let current = worker.descriptor.lock().unwrap().clone();
            if !self.workers.lock().unwrap().get(&descriptor.worker_id).is_some_and(|registered| Arc::ptr_eq(registered, worker))
                || current.pid != descriptor.pid
                || current.process_start_id != descriptor.process_start_id
                || current.worker_instance_id != descriptor.worker_instance_id
                || current.stop_requested_at != descriptor.stop_requested_at
            {
                return Err(format!("Session worker {} was replaced during stop", descriptor.worker_id));
            }
            Ok(())
        };
        let kind = if archive { "worker_archive_and_shutdown" } else { "shutdown" };
        let client = { worker.client.lock().unwrap().clone() };
        if let Some(client) = client {
            let request_timeout = if force { 1_000 } else { 30_000 };
            // A stuck write, disconnected channel or rejected request cannot bypass
            // process cleanup. The outer deadline also bounds the transport write.
            let requested = tokio::time::timeout(Duration::from_millis(request_timeout), async {
                client.request_worker(command(kind), request_timeout).await
                    .map_err(|error| error.to_string()).and_then(response_data)
            }).await.unwrap_or_else(|_| Err(format!("Timed out requesting {kind}")));
            if let Err(error) = requested { eprintln!("Worker {} shutdown request failed: {error}", descriptor.worker_id); }
            assert_stop_current()?;
            client.close().await;
            let mut current = worker.client.lock().unwrap();
            if current.as_ref().is_some_and(|current| Arc::ptr_eq(current, &client)) { *current = None; }
        }
        self.ownership.assert_current().await.map_err(|error| error.to_string())?;
        assert_stop_current()?;
        let identity = ProcessIdentity { pid: descriptor.pid as i64, process_start_id: descriptor.process_start_id.clone() };
        // `const gracefulDeadline = Date.now() + (force ? 500 : process.platform === "win32" ? 10_000 : 2000)`
        // (daemon-supervisor.ts:6824); this port only serves Windows workers.
        let deadline = tokio::time::Instant::now() + Duration::from_millis(if force { 500 } else { 10_000 });
        let mut alive = is_stopping_process_alive(&identity);
        let mut sigkill_sent = false;
        while alive {
            assert_stop_current()?;
            if tokio::time::Instant::now() >= deadline {
                // daemon-supervisor.ts:6830-6843: `force` escalates to SIGKILL, then waits 1s
                // before declaring the stop a timeout.
                self.ownership.assert_current().await.map_err(|error| error.to_string())?;
                assert_stop_current()?;
                if force && identity.process_start_id.is_some() && matches_exact_process_identity(&identity) {
                    sigkill_sent = signal_process_group_or_process(descriptor.pid, Signal::Kill);
                    let force_deadline = tokio::time::Instant::now() + Duration::from_millis(1000);
                    while is_stopping_process_alive(&identity) && tokio::time::Instant::now() < force_deadline {
                        assert_stop_current()?;
                        tokio::time::sleep(Duration::from_millis(25)).await;
                    }
                }
                assert_stop_current()?;
                if is_stopping_process_alive(&identity) {
                    // `if (removeDescriptor) { this.scheduleWorkerStopFinalization(worker); }` then
                    // `throw new WorkerStopTimeoutError(...)` (daemon-supervisor.ts:6845-6852): a
                    // stop that timed out keeps escalating in the background and finishes the
                    // interrupted cleanup once the process is gone, so a dead worker is never left
                    // registered for the next boot to recover.
                    self.schedule_worker_stop_finalization(worker);
                    return Err(format!("Session worker {} did not stop{}", descriptor.worker_id, if sigkill_sent { " after SIGKILL" } else { "" }));
                }
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
            alive = is_stopping_process_alive(&identity);
        }
        self.ownership.assert_current().await.map_err(|error| error.to_string())?;
        assert_stop_current()?;
        let client = { worker.client.lock().unwrap().take() };
        if let Some(client) = client { client.close().await; }
        if let Err(error) = self.recover_uncertain_worker_operations(worker).await {
            if supervisor_maintenance::permanent_stop_cleanup_error(&error) {
                self.park_worker_stop_cleanup_failure(worker, &error);
            } else {
                self.schedule_worker_stop_finalization(worker);
            }
            return Err(error);
        }
        self.invalidate_worker_input_pauses(worker);
        if let Err(error) = self.finish_worker_schedules(worker, archive).await {
            self.schedule_worker_stop_finalization(worker);
            return Err(error);
        }
        assert_stop_current()?;
        remove_file_durably(&self.descriptor_dir.join(format!("{}.json", descriptor.worker_id)).to_string_lossy(), RemoveFileDurablyOptions { fsync_dir: true, platform: None }).await.map_err(|error| error.to_string())?;
        self.retire_worker_journals(&descriptor).await;
        assert_stop_current()?;
        self.workers.lock().unwrap().remove(&descriptor.worker_id);
        // The registration is gone, so its rows stop being live: owned rows die
        // with it, resident rows are passivated.
        self.flip_worker_roster_entries_inactive(worker);
        self.broadcast_heartbeats_changed();
        Ok(())
    }
    async fn dispatch(self: &Arc<Self>, public: &Arc<PublicClient>, mut body: Map<String, Value>) -> Result<Option<DaemonResponse>, String> {
        let id = body.get("id").and_then(Value::as_str).map(str::to_string);
        let kind = body.get("type").and_then(Value::as_str).ok_or("Daemon command requires a type")?.to_string();
        let success = |data| Some(DaemonResponse::success(id.as_deref(), &kind, data));
        match kind.as_str() {
            "ack_result" => { if let Some(command_id) = body.get("commandId").and_then(Value::as_str) { self.journal.lock().unwrap().acknowledge(&public.identity(), command_id)?; } return Ok(None); }
            "create" => return Ok(success(Some(self.create(public, &body).await?))),
            "list" => {
                let data = self.handle_list(public, &body).await?;
                return Ok(success(Some(data)));
            }
            "list_agent_peers" => {
                let token = body.get("workerToken").and_then(Value::as_str).filter(|token| !token.is_empty())
                    .ok_or("Worker authentication failed")?;
                let workers: Vec<_> = self.workers.lock().unwrap().values().cloned().collect();
                let requester = workers.iter().find(|worker| worker.descriptor.lock().unwrap().authentication_token == token)
                    .ok_or("Worker authentication failed")?;
                let peers: Vec<_> = workers.iter().filter(|worker| {
                    !Arc::ptr_eq(worker, requester) && self.is_live_worker(worker)
                        && worker.descriptor.lock().unwrap().lifecycle == DAEMON_WORKER_LIFECYCLE_READY
                        && worker.client.lock().unwrap().as_ref().is_some_and(|client| client.is_connected())
                }).filter_map(|worker| {
                    let active = worker.descriptor.lock().unwrap().root_active_session_id.clone();
                    self.summary_for_active(worker, &active).map(|summary| self.agent_peer_summary(&summary))
                }).collect();
                return Ok(success(Some(json!({"peers": peers}))));
            }
            "roster_subscribe" => {
                public.roster_subscribed.store(true, Ordering::SeqCst);
                let roster: Vec<Value> = self.roster_entries_for_client().iter().map(agent_roster_entry_to_value).collect();
                return Ok(success(Some(json!({"roster": roster}))));
            }
            "roster_unsubscribe" => {
                public.roster_subscribed.store(false, Ordering::SeqCst);
                public.roster_resync_pending.store(false, Ordering::SeqCst);
                return Ok(success(None));
            }
            "retry_worker" => {
                let requested = body.get("activeSessionId").and_then(Value::as_str).ok_or("retry_worker requires activeSessionId")?;
                let summary = self.retry_worker(public.identity(), requested).await?;
                return Ok(success((!summary.is_null()).then_some(summary)));
            }
            // `case "agent_messages_status"` (daemon-supervisor.ts:2425-2435): without an
            // `activeSessionId` the id-less form is served, not rejected. The port is a
            // supervisor-only slice with no per-worker pause state, so the id-less answer is
            // the TS "no live worker" fallback `{ paused: false, limits: {} }` (2432), and the
            // id-less pause/resume fan-out (2436-2448) is refused explicitly because its
            // durable counterpart (`agent_messages_paused`) lives in daemon_mode.rs.
            "agent_messages_status" if body.get("activeSessionId").and_then(Value::as_str).is_none() => {
                return Ok(success(Some(json!({"paused": false, "limits": {}}))));
            }
            // `case "agent_messages_pause": case "agent_messages_resume"` (daemon-supervisor.ts:2436-2448):
            // without an `activeSessionId` the command is fanned out to every live worker that has a
            // client, and the first failure (else the first success) is the answer.
            "agent_messages_pause" | "agent_messages_resume" if body.get("activeSessionId").and_then(Value::as_str).is_none() => {
                let forwarded = command(&kind);
                let responses = self.forward_to_live_workers(forwarded).await;
                let failed = responses.iter().find(|response| !response.success).cloned();
                return Ok(Some(match failed {
                    Some(response) => { let mut response = response; response.id = id; response.command = kind.clone(); response }
                    None => DaemonResponse::success(id.as_deref(), &kind, responses.iter().find(|response| response.success).and_then(|response| response.data.clone())),
                }));
            }
            // `case "cron_list"` (daemon-supervisor.ts:2450-2480): the id-less form aggregates
            // the live workers' jobs and then adds every passive scheduled job
            // (`collectPassiveScheduledJobs`, 2476-2478).
            "cron_list" if body.get("activeSessionId").and_then(Value::as_str).is_none() => {
                let mut jobs: Vec<Value> = Vec::new();
                let mut seen: HashSet<String> = HashSet::new();
                let workers: Vec<Arc<Worker>> = self.workers.lock().unwrap().values().cloned().collect();
                for worker in workers {
                    if !self.is_live_worker(&worker) { continue; }
                    if worker.descriptor.lock().unwrap().lifecycle != DAEMON_WORKER_LIFECYCLE_READY { continue; }
                    let Ok(client) = self.connected_client(&worker).await else { continue; };
                    let Ok(response) = client.request_worker(command("cron_list"), 5_000).await else { continue; };
                    for job in cron_jobs_from_response(&response) {
                        let job_id = job.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                        if seen.insert(job_id) { jobs.push(job); }
                    }
                }
                let include_inactive = body.get("includeInactive").and_then(Value::as_bool) == Some(true);
                for (_, passive) in self.collect_passive_scheduled_jobs(include_inactive).await {
                    let job_id = passive.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                    if seen.insert(job_id) { jobs.push(passive); }
                }
                jobs.sort_by(|left, right| compare_cron_jobs_by_next_run(left, right));
                return Ok(success(Some(json!({"jobs": jobs}))));
            }
            // Refresh connected workers; a temporarily unreachable worker may use a
            // snapshot only while no change notification has invalidated it.
            "heartbeats_list" if body.get("activeSessionId").and_then(Value::as_str).is_none() => {
                let workers: Vec<Arc<Worker>> = self.workers.lock().unwrap().values().cloned().collect();
                let mut heartbeats: Vec<Value> = Vec::new();
                let mut seen: HashSet<String> = HashSet::new();
                let mut failure: Option<DaemonResponse> = None;
                for worker in workers {
                    if !self.is_live_worker(&worker) { continue; }
                    let lifecycle = worker.descriptor.lock().unwrap().lifecycle.clone();
                    if lifecycle == DAEMON_WORKER_LIFECYCLE_FAILED { continue; }
                    // `if (worker.client && worker.descriptor.lifecycle === "ready")` (:2493):
                    // only an already-connected ready worker is asked; this path never dials.
                    let client = { worker.client.lock().unwrap().as_ref().cloned() };
                    let snapshot_epoch = worker.heartbeat_snapshot.lock().unwrap().epoch;
                    let forwarded = match client.filter(|client| client.is_connected() && lifecycle == DAEMON_WORKER_LIFECYCLE_READY) {
                        Some(client) => match client.request_worker(command("heartbeats_list"), 5_000).await {
                            Ok(response) if response.success => {
                                let rows = response.data.as_ref().and_then(|data| data.get("heartbeats")).and_then(Value::as_array).cloned().unwrap_or_default();
                                worker.heartbeat_snapshot.lock().unwrap().store_if_current(snapshot_epoch, rows.clone());
                                Some(rows)
                            },
                            Ok(_) => None,
                            Err(_) => None,
                        },
                        None => None,
                    };
                    let rows = forwarded.or_else(|| worker.heartbeat_snapshot.lock().unwrap().fresh());
                    let Some(rows) = rows else {
                        if failure.is_none() {
                            let state = if lifecycle == DAEMON_WORKER_LIFECYCLE_READY { "disconnected".to_string() } else { lifecycle.clone() };
                            failure = Some(DaemonResponse::failure(id.as_deref(), &kind, &format!("Cannot list heartbeats while session worker is {state}"), None));
                        }
                        continue;
                    };
                    for heartbeat in rows {
                        let job_id = heartbeat.get("job").and_then(|job| job.get("id")).and_then(Value::as_str).unwrap_or_default().to_string();
                        if seen.insert(job_id) { heartbeats.push(heartbeat); }
                    }
                }
                if let Some(failure) = failure { let mut failure = failure; failure.id = id; failure.command = kind.clone(); return Ok(Some(failure)); }
                // Passivated sessions keep their armed heartbeats; no worker can list them
                // (daemon-supervisor.ts:2526-2534).
                for (_, job) in self.collect_passive_scheduled_jobs(false).await {
                    let is_heartbeat = job.get("source").and_then(Value::as_str).is_some_and(|source| source == crate::core::cron_jobs::SOURCE_HEARTBEAT || source == crate::core::cron_jobs::SOURCE_RLM_HEARTBEAT);
                    if !is_heartbeat { continue; }
                    let job_id = job.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                    if !seen.insert(job_id) { continue; }
                    heartbeats.push(json!({"job": job}));
                }
                return Ok(success(Some(json!({"heartbeats": heartbeats}))));
            }
            // `case "cron_cancel"` (daemon-supervisor.ts:2591-2632): the id-less form searches
            // the live workers for the job, then the passive scheduled stores, then throws.
            "cron_cancel" if body.get("activeSessionId").and_then(Value::as_str).is_none() => {
                let job_id = body.get("jobId").and_then(Value::as_str).ok_or("cron_cancel requires jobId")?.to_string();
                let workers: Vec<Arc<Worker>> = self.workers.lock().unwrap().values().cloned().collect();
                for worker in workers {
                    if !self.is_live_worker(&worker) { continue; }
                    if worker.descriptor.lock().unwrap().lifecycle != DAEMON_WORKER_LIFECYCLE_READY { continue; }
                    let Ok(client) = self.connected_client(&worker).await else { continue; };
                    let mut list = command("cron_list");
                    list.insert("includeInactive".into(), json!(true));
                    let Ok(response) = client.request_worker(list, 5_000).await else { continue; };
                    if !cron_jobs_from_response(&response).iter().any(|job| job.get("id").and_then(Value::as_str) == Some(job_id.as_str())) { continue; }
                    let mut forwarded = body.clone();
                    forwarded.insert("type".into(), json!("cron_cancel"));
                    let mut response = client.request_worker(forwarded, REQUEST_TIMEOUT).await.map_err(|error| error.to_string())?;
                    response.id = id; response.command = kind.clone();
                    return Ok(Some(response));
                }
                if let Some(job) = self.cancel_passive_scheduled_job(&job_id).await {
                    // `broadcastHeartbeatsChanged()` (2627) re-arms the wake timer (:7110).
                    self.broadcast_heartbeats_changed();
                    return Ok(success(Some(json!({"job": job}))));
                }
                return Err(format!("No cron job found: {job_id}"));
            }
            // Cross-worker messages use the authenticated private delivery command: the
            // target worker cannot resolve a source session hosted by another worker.
            "send_message" if body.get("activeSessionId").and_then(Value::as_str).is_none() => {
                let target_selector = body.get("targetActiveSessionId").and_then(Value::as_str).ok_or("send_message requires targetActiveSessionId")?.to_string();
                // D-04: the journal records one outcome per attempted forward. Every
                // rejection is recorded with a fixed reason code, never with the raw error
                // text or the message body, and no failure path retries the send.
                let rejected = |source: Option<&str>, target: &str, reason: &'static str| {
                    self.record_agent_message_delivery(AgentMessageDeliveryRecord::new(
                        source, target, None, AgentMessageDeliveryOutcome::Rejected, Some(reason),
                    ));
                };
                let source = match body.get("fromActiveSessionId").and_then(Value::as_str) {
                    Some(from) => match self.find(&public.identity(), from).await {
                        Ok(found) => Some(found),
                        Err(error) => {
                            rejected(None, &target_selector, delivery_reason_code(&error));
                            return Err(error);
                        }
                    },
                    None => None,
                };
                // The resolved source id is known before the summary lookup, so a source that
                // resolves but has no roster summary is still attributable in the journal.
                let source_active_session_id = source.as_ref().map(|(_, active)| active.clone());
                let source_summary = match source.as_ref().map(|(worker, active)| {
                    self.summary_for_active(worker, active).ok_or("Source session worker has no summary")
                }).transpose() {
                    Ok(summary) => summary,
                    Err(error) => {
                        rejected(source_active_session_id.as_deref(), &target_selector, delivery_reason_code(&error));
                        return Err(error.to_string());
                    }
                };
                let source_active_session_id = source_summary
                    .as_ref()
                    .map(|summary| summary.active_session_id.clone().unwrap_or_else(|| summary.id.clone()))
                    .or(source_active_session_id);
                let agent_origin = body.get("agentOrigin").and_then(Value::as_bool) == Some(true);
                if agent_origin && source_summary.is_none() {
                    rejected(None, &target_selector, "missing_source");
                    return Err("Agent messaging requires fromActiveSessionId".into());
                }
                // Rejections after the source is known record it, so an operator can join
                // the outcome to the sender without the journal storing any session name.
                let rejected_with_source = |target: &str, reason: &'static str| {
                    rejected(source_active_session_id.as_deref(), target, reason);
                };
                let target = match self.find(&public.identity(), &target_selector).await {
                    Ok(found) => found,
                    Err(error) if error.starts_with("Unknown active session:") => {
                        let cwd = source_summary.as_ref().map(|summary| summary.cwd.clone())
                            .or_else(|| self.config.cwd.clone())
                            .unwrap_or_else(|| std::env::current_dir().map(|dir| dir.to_string_lossy().into_owned()).unwrap_or_default());
                        let session_dir = source.as_ref().and_then(|(worker, _)| worker.descriptor.lock().unwrap().session_dir.clone()).or_else(|| self.config.session_dir.clone());
                        let session_path = match self.catalog.resolve(&target_selector, &cwd, session_dir.as_deref()).await {
                            Ok(session_path) => session_path,
                            Err(catalog_error) => {
                                // `Ambiguous session selector` is preserved so a2a senders can tell
                                // it apart from the original lookup failure (:2725-2732).
                                let catalog_error = if catalog_error.starts_with("Ambiguous session selector") { catalog_error } else { error.clone() };
                                rejected_with_source(&target_selector, delivery_reason_code(&catalog_error));
                                return Err(catalog_error);
                            }
                        };
                        if let (Some(source_summary), true) = (source_summary.as_ref(), agent_origin) {
                            let Some(target_info) = crate::core::session_manager::read_session_info(&session_path).await else {
                                let message = format!("Unknown active session: {target_selector}");
                                rejected_with_source(&target_selector, delivery_reason_code(&message));
                                return Err(message);
                            };
                            if let Err(error) = crate::core::agent_messages::assert_agent_family_reach(
                                &self.family_catalog_entry(source_summary),
                                &self.family_catalog_entry(&summary_for_inactive_session(&target_info, false, false)),
                            ) {
                                rejected_with_source(&target_selector, delivery_reason_code(&error));
                                return Err(error);
                            }
                        }
                        let create = json!({"type":"create", "sessionPath": session_path, "continueRecent": false});
                        if let Err(error) = self.create_for_owner(public.identity(), create.as_object().expect("create body is an object")).await {
                            // A session that was never created cannot have received the message.
                            rejected_with_source(&target_selector, delivery_reason_code(&error));
                            return Err(error);
                        }
                        // `findSummaryInWorker(worker, sessionPath) ?? sessionSummaryFromRosterEntry(root)`
                        // (:2746-2750): the created root is now addressable, so the ordinary
                        // lookup yields the target the wake was for.
                        match self.find(&public.identity(), &target_selector).await {
                            Ok(found) => found,
                            Err(error) => {
                                rejected_with_source(&target_selector, delivery_reason_code(&error));
                                return Err(error);
                            }
                        }
                    }
                    Err(error) => {
                        rejected_with_source(&target_selector, delivery_reason_code(&error));
                        return Err(error);
                    }
                };
                let (target_worker, target_active) = target;
                if source.as_ref().is_some_and(|(_, source_active)| source_active == &target_active) {
                    rejected_with_source(&target_active, "self_target");
                    return Err("Agent messaging cannot target the sending session".into());
                }
                if let (Some(source_summary), true) = (source_summary.as_ref(), agent_origin) {
                    let target_summary = match self.summary_for_active(&target_worker, &target_active) {
                        Some(summary) => summary,
                        None => {
                            rejected_with_source(&target_active, "worker_error");
                            return Err("Target session worker has no summary".to_string());
                        }
                    };
                    if let Err(error) = crate::core::agent_messages::assert_agent_family_reach(
                        &self.family_catalog_entry(source_summary), &self.family_catalog_entry(&target_summary),
                    ) {
                        rejected_with_source(&target_active, delivery_reason_code(&error));
                        return Err(error);
                    }
                }
                let client = match self.connected_client(&target_worker).await {
                    Ok(client) => client,
                    Err(error) => {
                        // The connection failed before the forward, so the message provably
                        // never reached the target: that is a rejection, not an unknown result.
                        self.record_agent_message_delivery(AgentMessageDeliveryRecord::new(
                            source_active_session_id.as_deref(),
                            &target_active,
                            None,
                            AgentMessageDeliveryOutcome::Rejected,
                            Some(delivery_reason_code(&error.to_string())),
                        ));
                        return Err(error);
                    }
                };
                let forwarded = if let Some(source_summary) = source_summary {
                    // Never accept sender attribution supplied by the public caller.
                    let sender = crate::core::agent_messages::AgentSessionMessageSender {
                        active_session_id: Some(source_summary.active_session_id.unwrap_or(source_summary.id)),
                        session_id: Some(source_summary.session_id),
                        session_name: source_summary.session_name,
                        runtime_kind: Some(source_summary.runtime_kind.unwrap_or_else(|| "top-level".into())),
                        client_id: Some(public.id.lock().unwrap().clone()),
                    };
                    json!({"type":"worker_deliver_message", "targetActiveSessionId":target_active,
                        "message":body.get("message"), "sender":sender}).as_object().unwrap().clone()
                } else {
                    let mut forwarded = body.clone();
                    forwarded.insert("targetActiveSessionId".into(), json!(target_active));
                    forwarded
                };
                let mut response = match client.request_worker(forwarded, REQUEST_TIMEOUT).await {
                    Ok(response) => response,
                    Err(error) => {
                        // A lost response leaves delivery unknown and is never replayed;
                        // the journal records that uncertainty instead of guessing.
                        self.record_agent_message_delivery(AgentMessageDeliveryRecord::new(
                            source_active_session_id.as_deref(),
                            &target_active,
                            None,
                            AgentMessageDeliveryOutcome::Uncertain,
                            Some(delivery_reason_code(&error.to_string())),
                        ));
                        return Err(error.to_string());
                    }
                };
                self.record_agent_message_delivery(agent_message_delivery_record_from_response(
                    source_active_session_id.as_deref(),
                    &target_active,
                    &response,
                ));
                response.id = id; response.command = kind.clone();
                return Ok(Some(response));
            }
            "list_saved_sessions" => {
                let (cwd, session_dir) = if let Some(active) = body.get("activeSessionId").and_then(Value::as_str) {
                    let (worker, active) = self.find(&public.identity(), active).await?;
                    let summary = self.summary_for_active(&worker, &active).ok_or("Session worker has no summary")?;
                    (Some(summary.cwd), self.config.session_dir.clone())
                } else {
                    (
                        body.get("cwd").and_then(Value::as_str).map(resolve_path).or_else(|| self.config.cwd.clone()),
                        body.get("sessionDir").and_then(Value::as_str).map(str::to_string).or_else(|| self.config.session_dir.clone()),
                    )
                };
                // The browser supplies its cwd even for the global ("all") view.
                let filter_cwd = if body.get("scope").and_then(Value::as_str) == Some("current") { cwd.as_deref() } else { None };
                let active_session_id = body.get("activeSessionId").and_then(Value::as_str).map(str::to_string);
                // `const callbacks = command.id ? { onProgress: ..., onSession: ... } : undefined`
                // (daemon-supervisor.ts:3003-3023): the catalog's progress/session frames are
                // rewritten on the requesting client's connection, so the browser can render rows
                // while the scan is still running. Without them the loading counter stays frozen.
                let callbacks = body.get("id").and_then(Value::as_str).map(|command_id| {
                    let progress_id = command_id.to_string();
                    let session_id = command_id.to_string();
                    let progress_active = active_session_id.clone();
                    let session_active = active_session_id.clone();
                    let progress_public = Arc::clone(public);
                    let session_public = Arc::clone(public);
                    CatalogListCallbacks {
                        on_progress: Some(Arc::new(move |loaded: u64, total: u64| {
                            let mut frame = json!({
                                "id": progress_id,
                                "type": "session_list_progress",
                                "command": "list_saved_sessions",
                                "loaded": loaded,
                                "total": total,
                            });
                            if let Some(active) = &progress_active { frame["activeSessionId"] = json!(active); }
                            progress_public.write(&frame);
                        })),
                        on_session: Some(Arc::new(move |session: SessionInfo| {
                            let mut frame = json!({
                                "id": session_id,
                                "type": "session_list_item",
                                "command": "list_saved_sessions",
                                "session": serialize_saved_session_info(&session),
                            });
                            if let Some(active) = &session_active { frame["activeSessionId"] = json!(active); }
                            session_public.write(&frame);
                        })),
                    }
                });
                let saved = self.catalog.list(filter_cwd, session_dir.as_deref(), callbacks).await?;
                // `await withPassiveRlmDescendantInfos(saved, this.rlmSpawnLedgerFor(sessionDir),
                // { cwd, onSession: callbacks?.onSession, log })` (:3025-3029): the catalog scan
                // never visits session-artifacts, so without this merge a passivated RLM
                // descendant's row (and its spend) disappears from the saved-chat list.
                let daemon = Arc::clone(self);
                let list_callbacks = body.get("id").and_then(Value::as_str).map(|command_id| {
                    let item_id = command_id.to_string();
                    let item_active = body.get("activeSessionId").and_then(Value::as_str).map(str::to_string);
                    let item_public = Arc::clone(public);
                    Arc::new(move |session: &SessionInfo| {
                        let mut frame = json!({
                            "id": item_id,
                            "type": "session_list_item",
                            "command": "list_saved_sessions",
                            "session": serialize_saved_session_info(session),
                        });
                        if let Some(active) = &item_active { frame["activeSessionId"] = json!(active); }
                        item_public.write(&frame);
                    }) as Arc<dyn Fn(&SessionInfo) + Send + Sync>
                });
                let ledger = self.rlm_spawn_ledger_for(session_dir.as_deref()).await?;
                let sessions = crate::modes::daemon::rlm_ledger::with_passive_rlm_descendant_infos(
                    saved,
                    &ledger,
                    crate::modes::daemon::rlm_ledger::WithPassiveRlmDescendantInfosOptions {
                        cwd: if body.get("scope").and_then(Value::as_str) == Some("current") { cwd.clone() } else { None },
                        on_session: list_callbacks,
                        // The supervisor has no client log sink (`DaemonSupervisor.log` writes to the
                        // supervisor log); TS's `log: (message) => this.log(message)` maps to stderr here.
                        log: Some(Arc::new(move |message: &str| { let _ = &daemon; eprintln!("{message}"); })),
                    },
                )
                .await;
                return Ok(success(Some(json!({"sessions":sessions.iter().map(serialize_saved_session_info).collect::<Vec<_>>()}))));
            }
            // `case "shutdown"` (daemon-supervisor.ts:2418-2420): the success reply is returned
            // immediately through `setImmediate`, the `daemon_closing` broadcast goes out BEFORE
            // any worker stop (7314-7318), `force` (2419) is honoured by the stop, and a worker
            // stop timeout is swallowed (7327-7334) instead of aborting the shutdown.
            "shutdown" => {
                let force = body.get("force").and_then(Value::as_bool) == Some(true);
                let reply = DaemonResponse::success(id.as_deref(), &kind, None);
                // `setImmediate(() => void this.shutdown(...))` + `return success(...)`
                // (daemon-supervisor.ts:2418-2420): the reply is written before the deferred
                // shutdown body runs, and that body writes `daemon_closing` to every client
                // before it stops any worker (7314-7318).
                public.write(&json!(reply));
                for client in self.clients.lock().unwrap().values() { client.write(&json!({"type":"daemon_closing","reason":"shutdown"})); }
                let supervisor = Arc::clone(self);
                tokio::spawn(async move {
                    let workers: Vec<_> = supervisor.workers.lock().unwrap().values().cloned().collect();
                    for worker in workers {
                        // `stopWorker(worker, true, forceWorkers, true)` in a `Promise.all` whose
                        // `WorkerStopTimeoutError` is logged, not rethrown (7324-7334).
                        if let Err(error) = supervisor.stop_worker(&worker, false, force).await {
                            eprintln!("Worker {} remains tombstoned for recovery after shutdown: {error}", worker.descriptor.lock().unwrap().worker_id);
                        }
                    }
                    supervisor.stopped.cancel();
                });
                return Ok(None);
            }
            "detach" => {
                let active = body.get("activeSessionId").and_then(Value::as_str);
                let targets = if let Some(active) = active { vec![active.to_string()] } else { public.subscriptions.lock().unwrap().iter().cloned().collect() };
                for active in targets { public.subscriptions.lock().unwrap().remove(&active); public.write(&json!({"type":"session_detached","activeSessionId":active})); }
                self.release_client_pauses(public, active).await?;
                return Ok(success(None));
            }
            "release_session_input_pause" => {
                let pause_id = body.get("pauseId").and_then(Value::as_str).ok_or("Session input pause requires pauseId")?.to_string();
                let pause = self.pauses.lock().unwrap().get(&pause_id).cloned();
                let Some(pause) = pause else { return Ok(success(None)); };
                if pause.connection_id != public.connection_id { return Err(format!("Session input pause is owned by another client: {pause_id}")); }
                let requested = body.get("activeSessionId").and_then(Value::as_str).ok_or("Session input pause requires activeSessionId")?;
                if requested != pause.active && requested != pause.requested { return Err(format!("Session input pause belongs to another session: {pause_id}")); }
                body.insert("activeSessionId".into(), json!(pause.active));
                let client = { pause.worker.client.lock().unwrap().clone().ok_or("Session worker is not connected")? };
                let mut response = client.request_worker(body, REQUEST_TIMEOUT).await.map_err(|error| error.to_string())?;
                if response.success { self.pauses.lock().unwrap().remove(&pause_id); }
                response.id = id; return Ok(Some(response));
            }
            // `get_direct_worker_transport` (daemon-supervisor.ts:2159-2165) is refused because
            // `issuePeerTransport` (5082-5146) is not ported: it needs the worker's
            // `worker_register_peer_transport` grant (5120) answered by the worker side
            // (daemon_mode.rs:9201-9231), `worker.peerTransportCapable` from the
            // `peer_transport` capability in the worker_auth response (3629,
            // worker_auth_advertises_roster's sibling `workerAuthAdvertisesPeerTransport` at
            // 612-616), and a `get_daemon_socket_identity` read of the live worker socket
            // (5105-5113). Advertising `direct_peer_transport` without them would hand clients
            // a ticket this supervisor cannot mint, so the capability stays unadvertised and
            // the rejection stays explicit. `prepare_update_restart` / `restart` need the
            // update-restart transaction (2421-2423, 2415-2417).
            // `case "cancel_prompt_admission"` (daemon-supervisor.ts:2098-2127): served entirely from
            // the supervisor admission registry, so a cancel issued while the addressed worker is
            // still being created fences the wait locally instead of failing a worker lookup.
            "cancel_prompt_admission" => {
                let active_session_id = body.get("activeSessionId").and_then(Value::as_str).unwrap_or("");
                let admission_id = body.get("admissionId").and_then(Value::as_str).unwrap_or("");
                let admission = self.get_prompt_admission(&public.connection_id, active_session_id, admission_id);
                // `if (!admission) return success(command.id, command.type, { status: "unknown" });` (:2101).
                let Some(admission) = admission else { return Ok(success(Some(json!({"status": "unknown"})))); };
                // `if (admission.status === "owned") return success(..., { status: "owned" });` (:2102)
                // and the definitive-cancellation rule at :2103-2106.
                if admission.status == PromptAdmissionStatus::Owned { return Ok(success(Some(json!({"status": "owned"})))); }
                if admission.status == PromptAdmissionStatus::Cancelled { return Ok(success(Some(json!({"status": "cancelled"})))); }
                let (Some(worker), Some(worker_active)) = (admission.worker.clone(), admission.worker_active_session_id.clone()) else {
                    // `if (!admission.worker || !admission.workerActiveSessionId) { admission.status =
                    // "cancelled"; admission.controller.abort(); return success(..., "cancelled"); }`
                    // (:2107-2111).
                    self.set_prompt_admission_status(&admission, PromptAdmissionStatus::Cancelled, true);
                    return Ok(success(Some(json!({"status": "cancelled"}))));
                };
                let mut forwarded = body.clone();
                forwarded.insert("type".into(), json!("cancel_prompt_admission"));
                forwarded.insert("activeSessionId".into(), json!(worker_active));
                forwarded.insert("admissionId".into(), json!(admission.worker_admission_id));
                let client = self.connected_client(&worker).await?;
                let mut response = client.request(forwarded, REQUEST_TIMEOUT, DaemonClientRequestOptions::default()).await.map_err(|error| error.to_string())?;
                let status = response.data.as_ref().and_then(|data| data.get("status")).and_then(Value::as_str).unwrap_or("unknown").to_string();
                // `const current = (admission as SupervisorPromptAdmission).status;` (:2122):
                // re-read after the round-trip, then `owned`/`cancelled` win and anything else
                // returns a still-running admission to `waiting` unless it is already cancelled (:2123-2125).
                let current = self.prompt_admission_status(&admission);
                match status.as_str() {
                    "owned" => self.set_prompt_admission_status(&admission, PromptAdmissionStatus::Owned, false),
                    "cancelled" => self.set_prompt_admission_status(&admission, PromptAdmissionStatus::Cancelled, false),
                    _ => if current != Some(PromptAdmissionStatus::Cancelled) { self.set_prompt_admission_status(&admission, PromptAdmissionStatus::Waiting, false); },
                }
                response.id = id; response.command = kind.clone();
                return Ok(Some(response));
            }
            // `case "restart"` (daemon-supervisor.ts:2415-2417): the shutdown half is served —
            // success plus a `daemon_closing{reason:"update"}` broadcast — so the updater can
            // replace the processes. `prepare_update_restart` (:2421-2424) and
            // `get_direct_worker_transport` (:2159-2165) have no ported implementation; they are
            // refused through the unknown-command contract so the CLI fallback
            // (`isUnknownDaemonCommandError`, package-manager-cli.ts:1264-1277) fires instead of
            // aborting the update with "Could not prepare daemon sessions for automatic resume".
            "restart" => {
                public.write(&json!(DaemonResponse::success(id.as_deref(), &kind, None)));
                for client in self.clients.lock().unwrap().values() { client.write(&json!({"type":"daemon_closing","reason":"update"})); }
                let supervisor = Arc::clone(self);
                tokio::spawn(async move {
                    supervisor.stopped.cancel();
                });
                return Ok(None);
            }
            "prepare_update_restart" | "get_direct_worker_transport" => {
                return Err(format!("Unknown daemon command: {kind}"));
            }
            _ => {}
        }
        let previous_active = body.get("activeSessionId").and_then(Value::as_str).map(str::to_string);
        let requested = if kind == "reattach" { body.get("targetActiveSessionId") } else { body.get("activeSessionId") }.and_then(Value::as_str).ok_or("Daemon command requires activeSessionId")?.to_string();
        // `const admission = (command.type === "prompt" || command.type ===
        // "prompt_and_wait") && command.admissionId ? this.getPromptAdmission(...) : undefined;`
        // (daemon-supervisor.ts:2785-2788) and `throwIfAdmissionCancelled(admission)` (:2790).
        let prompt_admission = if matches!(kind.as_str(), "prompt" | "prompt_and_wait") {
            body.get("admissionId").and_then(Value::as_str)
                .and_then(|admission| self.get_prompt_admission(&public.connection_id, &requested, admission))
        } else {
            None
        };
        if prompt_admission.as_ref().is_some_and(|admission| admission.status == PromptAdmissionStatus::Cancelled) {
            // `throw new PromptAdmissionCancelledError()` (:459-460, 2790).
            return Err(crate::core::prompt_admission::PROMPT_ADMISSION_CANCELLED_MESSAGE.to_string());
        }
        let (worker, active) = self.find(&public.identity(), &requested).await?;
        // `if (admission) { admission.worker = match.worker; admission.workerActiveSessionId =
        // match.summary.activeSessionId ?? match.summary.id; }` (daemon-supervisor.ts:2806-2808):
        // a cancel that lands while this prompt is in flight is routed to the same worker.
        if let Some(admission) = prompt_admission.as_ref() { self.attach_prompt_admission_worker(admission, &worker, &active); }
        // `throwIfAdmissionCancelled(admission);` again after the lookup (:2800): a cancel that
        // landed while this prompt was still resolving the worker stops it before the forward.
        // `fenced_prompt_admission` re-reads the live record, exactly as TS reads the live
        // `admission` object; the `cancellationAdmission` fence at :1952-1955 mutates the registry.
        let fenced_prompt_admission = prompt_admission.as_ref().map(|admission| {
            self.get_prompt_admission(&admission.connection_id, &admission.active_session_id, &admission.public_admission_id)
                .unwrap_or_else(|| admission.clone())
        });
        if fenced_prompt_admission.as_ref().is_some_and(|admission| admission.status == PromptAdmissionStatus::Cancelled) {
            return Err(crate::core::prompt_admission::PROMPT_ADMISSION_CANCELLED_MESSAGE.to_string());
        }
        body.insert("activeSessionId".into(), json!(active));
        if matches!(kind.as_str(), "complete_owned_session" | "promote_owned_session") {
            if worker.descriptor.lock().unwrap().owner_client_id.as_deref() != Some(&public.identity()) { return Err("Session is not owned by this client".into()); }
            if kind == "promote_owned_session" {
                // `await this.promoteOwnedWorker(client, match.worker)` (daemon-supervisor.ts:2394);
                // the summary is re-read after the promotion, as TS's `this.publicSummary(...)`
                // is evaluated after the await (:2395).
                self.promote_owned_worker(&worker, &public.identity())?;
                let summary = self.refresh(&worker).await?.into_iter().find(|summary| summary.get("activeSessionId").or_else(|| summary.get("id")).and_then(Value::as_str) == Some(&active));
                return Ok(success(summary));
            }
            self.stop_worker(&worker, true, false).await?;
            return Ok(success(None));
        }
        // `const isRootKill = command.type === "kill" && (match.summary.activeSessionId ??
        // match.summary.id) === match.worker.descriptor.rootActiveSessionId` (daemon-supervisor.ts:2810-2812).
        // A root kill still forwards into the worker and answers with the worker's response
        // (:2832, 2840); only the stop is supervisor-side, and it runs in the `finally` (:2833-2839).
        let root_kill = kind == "kill" && worker.descriptor.lock().unwrap().root_active_session_id == active;
        if root_kill {
            // `await this.persistWorkerStopTombstone(match.worker, true);` (:2828) persists
            // `stopRequestedAt` / `archiveOnStop`, which `stop_worker` writes before it signals
            // the worker (native_supervisor.rs `stop_worker` = daemon-supervisor.ts:6689).
            let forwarded_body = body.clone();
            let forwarded = async {
                let client = self.connected_client(&worker).await?;
                client.request(forwarded_body, REQUEST_TIMEOUT, DaemonClientRequestOptions::default()).await.map_err(|error| error.to_string())
            }.await;
            // `finally { await this.stopWorker(match.worker, true, false, true); }` (:2835): the
            // stop runs even when the forward failed, and `archiveSession: true` selects the
            // archive path; a stop failure never replaces the worker's response (:2840).
            if let Err(error) = self.stop_worker(&worker, true, false).await {
                eprintln!("Session worker {} root kill cleanup failed: {error}", worker.descriptor.lock().unwrap().worker_id);
            }
            let mut response = forwarded?;
            response.id = id;
            response.command = kind.clone();
            return Ok(Some(response));
        }
        let pause_epoch = public.pause_epoch.load(std::sync::atomic::Ordering::SeqCst);
        if kind == "acquire_session_input_pause" {
            let lease_key = body.get("leaseKey").and_then(Value::as_str).ok_or("Session input pause requires leaseKey")?;
            body.insert("leaseKey".into(), json!(serde_json::to_string(&[public.connection_id.as_str(), public.identity().as_str(), lease_key]).map_err(|error| error.to_string())?));
        }
        if matches!(kind.as_str(), "prompt" | "prompt_and_wait" | "cancel_prompt_admission") {
            if let Some(admission) = body.get("admissionId").and_then(Value::as_str) {
                body.insert("admissionId".into(), json!(format!("supervisor-admission:{}:{admission}", public.connection_id)));
            }
        }
        let attaching = matches!(kind.as_str(), "attach" | "reattach");
        let chunked_attach = attaching && body.get("capabilities").and_then(Value::as_array)
            .is_some_and(|capabilities| capabilities.iter().any(|capability| capability == "chunked_snapshot"));
        if attaching {
            let supports_ui = body.get("supportsExtensionUi").and_then(Value::as_bool) == Some(true) || body.get("capabilities").and_then(Value::as_array).is_some_and(|capabilities| capabilities.iter().any(|value| value.as_str() == Some("extension_ui")));
            public.supports_extension_ui.store(supports_ui, std::sync::atomic::Ordering::SeqCst);
            body.insert("type".into(), json!("attach"));
            // daemon-supervisor.ts:5451-5458 derives the forwarded list from the attaching
            // client's own capabilities (`normalizeCapabilities`, 704-713) and ALWAYS sends
            // `supportsExtensionUi: false`; the worker-side UI decision is made by the
            // supervisor, not by the worker: TS re-asks the worker through `subscribeWorker`
            // with the real flag (3660-3669).
            body.insert("supportsExtensionUi".into(), json!(false));
            // `slim_attach` / `chunked_snapshot` / `history_ranges` stay out of the list
            // although TS:5451-5457 can add them: this port pins full snapshots (see
            // `subscribe`, "Full snapshots avoid a private chunk cache at the public
            // boundary") and reads `response.data.snapshot` directly, so a slim or chunked
            // reply would be a snapshot the supervisor cannot consume.
            body.insert("capabilities".into(), json!(["attach_snapshot", "event_sequence"]));
            if let Some(client_id) = body.get("clientId").and_then(Value::as_str) { *public.id.lock().unwrap() = client_id.to_string(); }
            body.remove("clientId");
            public.subscriptions.lock().unwrap().insert(active.clone());
        }
        // `command.promoteOwnedSession` (:2586, :2640) must be read before `body` is forwarded.
        let promote_owned_session = body.get("promoteOwnedSession").and_then(Value::as_bool) == Some(true);
        let client = self.connected_client(&worker).await?;
        let mut response = client.request(body, REQUEST_TIMEOUT, DaemonClientRequestOptions::default()).await.map_err(|error| error.to_string())?;
        response.id = id; response.command = kind.clone();
        if response.success {
            if let Some(data) = response.data.as_mut() {
                if data.get("sessionId").is_some() {
                    if let Some(summary) = serde_json::from_value::<RosterSessionSummary>(data.clone()).ok() {
                        *data = serde_json::to_value(self.public_summary(&worker, session_summary_from_roster_row(&summary, None, None, None))).unwrap_or(Value::Null);
                    }
                }
                if let Some(summary) = data.get("state").cloned().and_then(|value| serde_json::from_value::<RosterSessionSummary>(value).ok()) {
                    data["state"] = serde_json::to_value(self.public_summary(&worker, session_summary_from_roster_row(&summary, None, None, None))).unwrap_or(Value::Null);
                }
                if let Some(summary) = data.get("snapshot").and_then(|snapshot| snapshot.get("summary")).cloned().and_then(|value| serde_json::from_value::<RosterSessionSummary>(value).ok()) {
                    data["snapshot"]["summary"] = serde_json::to_value(self.public_summary(&worker, session_summary_from_roster_row(&summary, None, None, None))).unwrap_or(Value::Null);
                }
            }
        }
        // `const forward = async () => { const response = await this.forwardToWorker(match.worker,
        // resolvedCommand); if (admission && response.success) admission.status = "owned"; return
        // response; };` (daemon-supervisor.ts:2814-2818): a successful forward owns the admission.
        if response.success {
            if let Some(admission) = fenced_prompt_admission.as_ref() {
                self.set_prompt_admission_status(admission, PromptAdmissionStatus::Owned, false);
            }
        }
        // `case "cron_add"` (daemon-supervisor.ts:2583-2589) and `case "heartbeat_set"` (:2637-2644):
        // the forward is unchanged, but `if (response.success && command.promoteOwnedSession)
        // await this.promoteOwnedWorker(client, match.worker);` runs before the response is
        // returned, so a schedule created from a client-owned session survives the owner's
        // disconnect (`disconnected()` stops owner-bound workers after the 30s grace).
        if response.success && promote_owned_session && matches!(kind.as_str(), "cron_add" | "heartbeat_set") {
            self.promote_owned_worker(&worker, &public.identity())?;
        }
        if response.success && matches!(kind.as_str(), "heartbeat_set" | "heartbeat_pause" | "heartbeat_clear" | "cron_add" | "cron_cancel") {
            worker.heartbeat_snapshot.lock().unwrap().invalidate();
            self.broadcast_heartbeats_changed();
        }
        if kind == "acquire_session_input_pause" && response.success {
            let pause_id = response.data.as_ref().and_then(|data| data.get("pauseId")).and_then(Value::as_str).ok_or("Worker returned an invalid session input pause id")?.to_string();
            let pause = InputPause { connection_id: public.connection_id.clone(), worker: worker.clone(), active: active.clone(), requested: requested.clone() };
            self.pauses.lock().unwrap().insert(pause_id, pause);
            if public.stopped.is_cancelled() || public.pause_epoch.load(std::sync::atomic::Ordering::SeqCst) != pause_epoch {
                self.release_client_pauses(public, Some(&active)).await?;
                return Err("Session input pause acquisition was invalidated before completion".into());
            }
        }
        if attaching {
            if !response.success { public.subscriptions.lock().unwrap().remove(&active); }
            else {
                if let Some(data) = response.data.as_mut().and_then(Value::as_object_mut) {
                    if let Some(client) = data.get_mut("client").and_then(Value::as_object_mut) { client.insert("id".into(), json!(public.id.lock().unwrap().clone())); }
                }
                if let Some(snapshot) = response.data.as_ref().and_then(|data| data.get("snapshot")) {
                    let message = snapshot.get("summary").and_then(|summary| summary.get("streamingMessage")).filter(|value| value.get("role").and_then(Value::as_str) == Some("assistant"))
                        .or_else(|| snapshot.get("messages").and_then(Value::as_array).and_then(|messages| messages.iter().rev().find(|value| value.get("role").and_then(Value::as_str) == Some("assistant"))));
                    if let Some(message) = message.and_then(|value| serde_json::from_value(value.clone()).ok()) { worker.stream.lock().unwrap().seed(&active, message); }
                }
                self.subscribe(&worker, &active).await?;
                if kind == "reattach" { if let Some(previous) = previous_active.filter(|previous| previous != &active) { public.subscriptions.lock().unwrap().remove(&previous); public.write(&json!({"type":"session_detached","activeSessionId":previous})); } }
            }
        }
        if chunked_attach && response.success {
            self.stream_cached_attach(public, &active, &mut response).await?;
            return Ok(None);
        }
        Ok(Some(response))
    }
    async fn handle_line(self: Arc<Self>, public: Arc<PublicClient>, line: Vec<u8>) {
        let parsed = parse_command(&line);
        let (body, envelope_client, protocol_version) = match parsed {
            Ok(parsed) => parsed,
            // `this.write(client, failure(salvageDaemonCommandId(line), "parse", error));`
            // (daemon-supervisor.ts:1939): the id survives the parse failure, so the sender can
            // settle its pending request instead of waiting out the 30s request timeout.
            Err(error) => {
                let salvaged = daemon_protocol::salvage_daemon_command_id(&String::from_utf8_lossy(&line));
                public.write(&json!(DaemonResponse::failure(salvaged.as_deref(), "parse", &error, None)));
                return;
            }
        };
        let id = body.get("id").and_then(Value::as_str).map(str::to_string);
        let kind = body.get("type").and_then(Value::as_str).unwrap_or("dispatch").to_string();
        // `parseCommandAndRegisterPromptAdmission(client, line)` (daemon-supervisor.ts:1884-1919):
        // the admission is registered before `handleLine`'s first await, so a `cancel_prompt_admission`
        // that arrives later in the same read can fence it (:1952-1955). Registration failures are
        // parse-phase failures and are answered with the salvaged id (:1939).
        // Bound to `_prompt_admission_guard` (not `_`): a named binding drops at the end of the
        // function, which is exactly the `finally` the TypeScript runs (daemon-supervisor.ts:2842).
        let mut _prompt_admission_guard: Option<PromptAdmissionGuard> = None;
        if matches!(kind.as_str(), "prompt" | "prompt_and_wait") && body.get("admissionId").is_some() {
            if let Err(error) = self.register_prompt_admission(&public, &body) {
                let salvaged = daemon_protocol::salvage_daemon_command_id(&String::from_utf8_lossy(&line));
                public.write(&json!(DaemonResponse::failure(salvaged.as_deref(), &kind, &error, None)));
                return;
            }
            _prompt_admission_guard = self.get_prompt_admission(
                &public.connection_id,
                body.get("activeSessionId").and_then(Value::as_str).unwrap_or(""),
                body.get("admissionId").and_then(Value::as_str).unwrap_or(""),
            ).map(|admission| PromptAdmissionGuard { supervisor: Arc::clone(&self), admission });
        }
        // `const cancellationAdmission = command.type === "cancel_prompt_admission" ?
        // this.getPromptAdmission(client, command.activeSessionId, command.admissionId) : undefined;`
        // (daemon-supervisor.ts:1948-1951) and the still-queued fence at :1952-1955.
        if kind == "cancel_prompt_admission" {
            if let (Some(active_session_id), Some(admission_id)) =
                (body.get("activeSessionId").and_then(Value::as_str), body.get("admissionId").and_then(Value::as_str))
            {
                let cancellation = self.get_prompt_admission(&public.connection_id, active_session_id, admission_id);
                if let Some(admission) = cancellation.filter(|admission| {
                    admission.status == PromptAdmissionStatus::Waiting && admission.worker.is_none()
                }) {
                    self.set_prompt_admission_status(&admission, PromptAdmissionStatus::Cancelled, true);
                }
            }
        }
        // `if (!DAEMON_COMMAND_TYPES.has(command.type)) { this.write(client, failure(command.id,
        // command.type, `Unknown daemon command: ${command.type}`)); return; }`
        // (daemon-supervisor.ts:1969-1972): the unlisted-command contract is what lets a client
        // detect a supervisor build that predates a command (`isUnknownDaemonCommandError`).
        if !super::DAEMON_COMMAND_TYPES.contains(&kind.as_str()) {
            public.write(&json!(DaemonResponse::failure(id.as_deref(), &kind, &format!("Unknown daemon command: {kind}"), None)));
            return;
        }
        // The negotiated protocol belongs to the envelope, never the command body.
        if kind == "get_session_tree" {
            let min_protocol = daemon_protocol::daemon_command_compatibility("get_session_tree").min_protocol;
            if protocol_version.unwrap_or(0) < u64::from(min_protocol) {
                public.write(&json!(DaemonResponse::failure(id.as_deref(), &kind, &format!("get_session_tree requires client protocol {min_protocol} or newer"), None)));
                return;
            }
        }
        if let Some(identity) = envelope_client.as_ref().filter(|identity| !identity.is_empty()) { *public.protocol_id.lock().unwrap() = Some(identity.clone()); }
        let journal_identity = envelope_client.map(|identity| if identity.is_empty() { public.identity() } else { identity }).filter(|_| daemon_protocol::is_daemon_mutating_command(&kind) && kind != "ack_result").zip(id.clone());
        // Prompt admissions above must be registered before the first await.
        // Keep this guard through dispatch so a fresh attachment or mutation
        // cannot race the eviction snapshot and graceful worker shutdown.
        let _admission = self.eviction_fence.read().await;
        if let Err(error) = self.ownership.assert_current().await {
            public.write(&json!(DaemonResponse::failure(id.as_deref(), &kind, &error.to_string(), None))); return;
        }
        if let Some((client, id)) = &journal_identity {
            match self.journal.lock().unwrap().begin(client, id, &kind) {
                Ok(CommandJournalBeginResult::Complete(response)) => { public.write(&json!(response)); return; }
                Ok(CommandJournalBeginResult::Pending) => { public.write(&json!({"type":"response", "command":kind, "id":id, "success":false, "error":"The previous command result is uncertain and was not replayed", "errorInfo":{"code":"command_result_uncertain","clientId":client,"commandId":id}})); return; }
                Ok(CommandJournalBeginResult::New) => {},
                Err(error) => { public.write(&json!(DaemonResponse::failure(id.as_str().into(), &kind, &error, None))); return; }
            }
        }
        let response = match self.dispatch(&public, body).await { Ok(response) => response, Err(error) => Some(DaemonResponse::failure(id.as_deref(), &kind, &error, None)) };
        if let Some(response) = response {
            if let Some((client, id)) = journal_identity {
                let durable = match self.ownership.assert_current().await {
                    Ok(()) => self.journal.lock().unwrap().record_result(&client, &id, response.clone()),
                    Err(error) => Err(error.to_string()),
                };
                if let Err(error) = durable { public.write(&json!(DaemonResponse::failure(Some(&id), &kind, &error, None))); return; }
            }
            public.write(&json!(response));
        }
    }
    /// D-04: append one bounded, content-free delivery record. Telemetry only: a failed
    /// append never changes the delivery result or triggers a replay.
    fn record_agent_message_delivery(&self, record: AgentMessageDeliveryRecord) {
        if let Ok(mut journal) = self.agent_message_delivery_journal.lock() {
            journal.record(record);
        }
    }

    fn owned_worker_candidates(&self, owner: &str) -> Vec<Arc<Worker>> {
        let candidates: Vec<_> = self.workers.lock().unwrap().values().cloned().collect();
        candidates.into_iter().filter(|worker| worker.descriptor.lock().unwrap().owner_client_id.as_deref() == Some(owner)).collect()
    }

    fn disconnected(self: &Arc<Self>, public: &Arc<PublicClient>) {
        self.clients.lock().unwrap().remove(&public.connection_id);
        public.roster_subscribed.store(false, Ordering::SeqCst);
        public.roster_resync_pending.store(false, Ordering::SeqCst);
        let owner = public.identity();
        // `this.cancelWaitingPromptAdmissionsForClient(client);` (daemon-supervisor.ts:1681): a
        // disconnected sender's still-queued admissions are cancelled so their in-flight prompts
        // fail with `PromptAdmissionCancelledError` instead of running on behalf of a gone client.
        self.cancel_waiting_prompt_admissions_for_client(public);
        let supervisor = self.clone();
        let public = public.clone();
        tokio::spawn(async move {
            if let Err(error) = supervisor.release_client_pauses(&public, None).await { eprintln!("Disconnected client pause cleanup failed: {error}"); }
            tokio::time::sleep(Duration::from_secs(30)).await;
            if supervisor.clients.lock().unwrap().values().any(|client| client.identity() == owner) { return; }
            let workers = supervisor.owned_worker_candidates(&owner);
            for worker in workers {
                // stop_worker revalidates registration and process generation before
                // every destructive step, including after asynchronous requests.
                if let Err(error) = supervisor.stop_worker(&worker, true, false).await { eprintln!("Owned worker cleanup failed: {error}"); }
            }
        });
    }
}

/// The TS `servedRows` set holds summary object identities; the port has no
/// such identity, so a row's own JSON is its key (`activeByFile` rows are the
/// same objects, so equal JSON means the same served row).
trait ServedRowKey {
    fn served_row_key(&self) -> String;
}

impl ServedRowKey for SessionSummary {
    fn served_row_key(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// `sessionSummaryFromRosterEntry(entry)`: the slim roster row widened to the
/// daemon `SessionSummary`. `statusLabel`/`lastHeardFromAt` are written only when
/// the row carries them, exactly like the TypeScript's conditional spreads.
fn session_summary_from_roster_row(
    summary: &RosterSessionSummary,
    status: Option<AgentRosterStatus>,
    status_label: Option<&str>,
    last_heard_from_at: Option<&str>,
) -> SessionSummary {
    let mut value = serde_json::to_value(summary).unwrap_or(Value::Null);
    if let Some(object) = value.as_object_mut() {
        object.insert("sessionActions".into(), json!({"queuedCount": 0, "steering": [], "followUps": []}));
        if let Some(status) = status { object.insert("rosterStatus".into(), json!(status.as_str())); }
        if let Some(label) = status_label { object.insert("statusLabel".into(), json!(label)); }
        if let Some(last_heard_from_at) = last_heard_from_at { object.insert("lastHeardFromAt".into(), json!(last_heard_from_at)); }
    }
    serde_json::from_value(value).unwrap_or_default()
}

/// `sessionSummaryFromRosterEntry(entry)` for an entry that already has its
/// ledger marks; `workerId` is not part of the summary shape.
fn summary_from_entry(entry: &AgentRosterEntry) -> SessionSummary {
    session_summary_from_roster_row(&entry.summary, Some(entry.status), entry.status_label.as_deref(), entry.last_heard_from_at.as_deref())
}

fn recovery_command(descriptor: &DaemonWorkerDescriptor) -> Result<Map<String, Value>, String> {
    let session_file = descriptor.session_file.as_ref().or(descriptor.create_command.session_path.as_ref()).ok_or("Worker has no saved session to recover")?;
    let mut command = command("create");
    command.insert("sessionPath".into(), json!(session_file));
    command.insert("config".into(), json!({"sessionDir":descriptor.session_dir, "telemetryDisabled":descriptor.telemetry_disabled}));
    Ok(command)
}
fn server_capabilities() -> Vec<String> {
    // Capability honesty: advertise every default the supervisor really serves, and withhold
    // only the ones it does not. `heartbeat_catalog` and `authoritative_child_roster` are
    // withheld by the filter below in the ported slice even though both paths ARE served:
    //   - `heartbeat_catalog` gates `heartbeats_list` on the CLIENT side
    //     (heartbeat_catalog.rs:34, daemon-agent-connection.ts), so withholding it made
    //     supervisor clients never ask for a command this supervisor forwards.
    //   - `authoritative_child_roster` gates `get_rlm_children` on the client side
    //     (daemon_agent_connection.rs:2499-2501), which the supervisor's generic forward
    //     already routes; withholding it made RLM children invisible to those clients.
    // Still withheld, because the path is genuinely absent here:
    //   - `slim_attach` - private worker snapshots still arrive in full. Public
    //     `chunked_snapshot` attaches are served through the disk-backed cache,
    //     but that is not a bound on the initial decoded worker response.
    //   - `history_ranges` - the attach branch pins `capabilities` to
    //     `["attach_snapshot", "event_sequence"]`, so the snapshot never carries the `history`
    //     field the capability promises.
    //   - `owned_session_recovery_context` - needs `command.recoveryConfig` handling in the
    //     owned-worker attach branch (daemon-supervisor.ts:5385-5396), which the port lacks.
    let mut capabilities: Vec<String> = daemon_protocol::DAEMON_DEFAULT_SERVER_CAPABILITIES
        .iter()
        .map(super::super::daemon_client::capability_name)
        .filter(|capability| {
            !matches!(
                capability.as_str(),
                "slim_attach" | "history_ranges" | "owned_session_recovery_context"
            )
        })
        .collect();
    capabilities.push("agent_roster".to_string());
    capabilities
}
/// `cronJobsFromResponse(response)` (daemon-supervisor.ts:640-646).
fn cron_jobs_from_response(response: &DaemonResponse) -> Vec<Value> {
    if !response.success { return Vec::new(); }
    response.data.as_ref().and_then(|data| data.get("jobs")).and_then(Value::as_array).cloned().unwrap_or_default()
}

/// `sortCronJobs(jobs)` (daemon-supervisor.ts:656-669): ascending `nextRunAt`, with
/// jobs that have no `nextRunAt` last.
fn compare_cron_jobs_by_next_run(left: &Value, right: &Value) -> std::cmp::Ordering {
    let next_run = |job: &Value| job.get("nextRunAt").and_then(Value::as_str).map(str::to_string);
    match (next_run(left), next_run(right)) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (Some(_), None) => std::cmp::Ordering::Less,
        (Some(left), Some(right)) => left.cmp(&right),
    }
}

/// `DEFERRED_RECOVERY_RECHECK_MS` / `MAX_DEFERRED_RECOVERY_ROUNDS`
/// (daemon-supervisor.ts:211-213, values shared with daemon_supervisor.rs:121-123).
const DEFERRED_RECOVERY_RECHECK_MS: u64 = 5000;
const MAX_DEFERRED_RECOVERY_ROUNDS: u64 = 10;

/// The three verdicts of `processIdentity(pid, processStartId)`
/// (daemon-supervisor.ts:4246, 4287-4292): `Current`, `Unknown` (alive but not provably ours)
/// and `Gone`/`Replaced`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProcessIdentityVerdict { Current, Unknown, Gone }

fn process_identity_verdict(identity: &ProcessIdentity) -> ProcessIdentityVerdict {
    if !is_process_alive(identity.pid as i32) { return ProcessIdentityVerdict::Gone; }
    if identity.process_start_id.is_none() { return ProcessIdentityVerdict::Unknown; }
    if get_process_start_id(identity.pid).as_deref() == identity.process_start_id.as_deref() { ProcessIdentityVerdict::Current } else { ProcessIdentityVerdict::Gone }
}

/// `Date.parse(value)`, `undefined` when unparseable (the TS guards with `Number.isFinite`).
fn iso_to_ms(value: &str) -> Option<f64> {
    let millis = crate::core::cron_jobs::parse_iso_date(value);
    millis.is_finite().then_some(millis)
}

fn descriptor_key(socket: &str) -> String { format!("{:x}", Sha256::digest(socket.as_bytes()))[..12].to_string() }
/// `finalizeTimedOutWorkerStop(worker)` (daemon-supervisor.ts:6899-6980).
///
/// A stop that timed out left a tombstoned registration behind. Keep escalating until the exact
/// process generation is gone (a replaced pid counts as gone, so a recycled pid is never
/// signalled; an unobservable identity counts as alive), then finish the interrupted cleanup by
/// re-running the stop with a bounded cleanup budget. Permanent failures retain their
/// journals and a failed stop record. The loop also stops on shutdown or replacement.
/// `worker.stopFinalization` (daemon-supervisor.ts:6891-6896): the worker ids whose timed-out
/// stop finalizer is already running. A process-wide set is equivalent to the per-registration
/// field because one supervisor owns the process; the entry is released when the finalizer ends.
fn stop_finalizations() -> &'static Mutex<HashSet<String>> {
    static SCHEDULED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SCHEDULED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn stop_cleanup_is_parked(descriptor: &DaemonWorkerDescriptor) -> bool {
    descriptor.stop_requested_at.is_some() && descriptor.lifecycle == DAEMON_WORKER_LIFECYCLE_FAILED
}

fn stop_cleanup_should_retry(error: &str, attempts: usize) -> bool {
    !supervisor_maintenance::permanent_stop_cleanup_error(error) && attempts < STOP_CLEANUP_MAX_ATTEMPTS
}

fn is_stopping_process_alive(identity: &ProcessIdentity) -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};
        // Windows keeps an exited process object addressable while another handle
        // is open. OpenProcess alone therefore does not establish liveness.
        unsafe {
            let handle = OpenProcess(PROCESS_SYNCHRONIZE, 0, identity.pid as u32);
            if !handle.is_null() {
                let exited = WaitForSingleObject(handle, 0) == WAIT_OBJECT_0;
                CloseHandle(handle);
                if exited { return false; }
            }
        }
    }
    is_process_identity_alive(identity)
}

async fn finalize_timed_out_worker_stop(supervisor: Arc<Supervisor>, worker: Arc<Worker>, descriptor: DaemonWorkerDescriptor) {
    let (pid, process_start_id, worker_id, worker_instance_id, stop_requested_at, archive_on_stop) =
        (descriptor.pid as i64, descriptor.process_start_id, descriptor.worker_id, descriptor.worker_instance_id, descriptor.stop_requested_at, descriptor.archive_on_stop == Some(true));
    let _guard = StopFinalizationGuard { worker_id: worker_id.clone() };
    let identity = ProcessIdentity { pid, process_start_id: process_start_id.clone() };
    let is_stop_generation_current = || {
        let descriptor = worker.descriptor.lock().unwrap().clone();
        supervisor.workers.lock().unwrap().get(&worker_id).is_some_and(|registered| Arc::ptr_eq(registered, &worker))
            && descriptor.stop_requested_at.is_some()
            && descriptor.stop_requested_at == stop_requested_at
            && descriptor.worker_instance_id == worker_instance_id
            && descriptor.pid as i64 == pid
            && descriptor.process_start_id == process_start_id
    };
    let sigkill_deadline = tokio::time::Instant::now() + Duration::from_millis(super::STOP_FINALIZATION_SIGKILL_GRACE_MS);
    let mut killed = false;
    while !supervisor.stopped.is_cancelled() {
        if !is_stop_generation_current() { return; }
        if !is_stopping_process_alive(&identity) { break; }
        if !killed && tokio::time::Instant::now() >= sigkill_deadline {
            if supervisor.ownership.assert_current().await.is_err() || !is_stop_generation_current() { return; }
            if identity.process_start_id.is_some() && matches_exact_process_identity(&identity) {
                killed = signal_process_group_or_process(identity.pid as i32, Signal::Kill);
            }
        }
        tokio::time::sleep(Duration::from_millis(super::STOP_FINALIZATION_RECHECK_MS)).await;
    }
    // Retry transient failures within a fixed budget. Permanent failures retain the
    // tombstone and journals for repair rather than spinning after every restart.
    let mut attempts = 0;
    while !supervisor.stopped.is_cancelled() {
        if !is_stop_generation_current() { return; }
        if stop_cleanup_is_parked(&worker.descriptor.lock().unwrap()) { return; }
        attempts += 1;
        match supervisor.stop_worker(&worker, archive_on_stop, true).await {
            Ok(()) => return,
            Err(error) => {
                eprintln!("Failed to finalize timed-out worker stop {worker_id}: {error}");
                if supervisor.ownership.assert_current().await.is_err() || !is_stop_generation_current() { return; }
                if stop_cleanup_is_parked(&worker.descriptor.lock().unwrap()) { return; }
                if !stop_cleanup_should_retry(&error, attempts) {
                    supervisor.park_worker_stop_cleanup_failure(&worker, &error);
                    return;
                }
                tokio::time::sleep(Duration::from_millis(super::STOP_FINALIZATION_RETRY_MS)).await;
            }
        }
    }
}
/// Releases the `stop_finalizations()` entry when the finalizer returns, like TS's
/// `worker.stopFinalization = undefined` in the `finally` (daemon-supervisor.ts:6894-6896).
struct StopFinalizationGuard { worker_id: String }
impl Drop for StopFinalizationGuard {
    fn drop(&mut self) { stop_finalizations().lock().unwrap().remove(&self.worker_id); }
}

fn worker_socket(supervisor: &str, worker: &str) -> String {
    let key = descriptor_key(supervisor);
    #[cfg(windows)] { format!(r"\\.\pipe\prime-agent-worker-{key}-{}", &worker[..worker.len().min(12)]) }
    #[cfg(unix)] { Path::new(&default_daemon_socket_dir()).join(format!("worker-{key}-{}.sock", &worker[..worker.len().min(12)])).to_string_lossy().into_owned() }
}
fn visible(worker: &Worker, client: &str) -> bool { worker.descriptor.lock().unwrap().owner_client_id.as_deref().map_or(true, |owner| owner == client) }

/// The public `roster_update` outbound (`{ type, changed, removed?, resync? }`).
fn roster_update_value(changed: &[AgentRosterEntry], removed: &[String], resync: Option<bool>) -> Value {
    let mut value = json!({
        "type": "roster_update",
        "changed": changed.iter().map(agent_roster_entry_to_value).collect::<Vec<_>>(),
    });
    if let Some(object) = value.as_object_mut() {
        if !removed.is_empty() { object.insert("removed".into(), json!(removed)); }
        if let Some(resync) = resync { object.insert("resync".into(), json!(resync)); }
    }
    value
}

/// `handleWorkerFrame`'s `worker.lastFrameAt = Date.now()`.
fn supervisor_now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|elapsed| elapsed.as_millis() as u64).unwrap_or(0)
}

/// `new Date(ms).toISOString()` with the daemon's millisecond precision.
fn iso_from_ms(ms: f64) -> String {
    let millis = ms as i64;
    chrono::DateTime::from_timestamp_millis(millis)
        .map(|time| time.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_default()
}

/// `workerAuthAdvertisesRoster(data)`.
fn worker_auth_advertises_roster(data: Option<&Value>) -> bool {
    data.and_then(|value| value.get("capabilities")).and_then(Value::as_array)
        .is_some_and(|capabilities| capabilities.iter().any(|value| value.as_str() == Some(DAEMON_WORKER_ROSTER_CAPABILITY)))
}

/// `rosterParentByChild(edges)` plus `rosterFamilyDescendsFrom`'s walk:
/// membership at any step of the parent chain, never just the ultimate root.
fn roster_parent_by_child(edges: &[RlmLedgerEdge]) -> HashMap<String, String> {
    edges.iter()
        .map(|edge| (canonical_session_path(&edge.child), canonical_session_path(&edge.parent)))
        .collect()
}

fn roster_path_descends_from(parent_by_child: &HashMap<String, String>, path: &str, roots: &HashSet<String>) -> bool {
    let mut visited: HashSet<String> = HashSet::new();
    let mut current = path.to_string();
    while !visited.contains(&current) {
        if roots.contains(&current) { return true; }
        visited.insert(current.clone());
        match parent_by_child.get(&current) {
            Some(parent) => current = parent.clone(),
            None => return false,
        }
    }
    false
}

/// `rosterEntryForSpawnLedgerEdge(edge)`.
fn roster_entry_for_spawn_ledger_edge(edge: &RlmLedgerEdge) -> WorkerRosterEntry {
    let persisted_session_id = Path::new(&edge.child).file_stem().map(|name| name.to_string_lossy().to_string()).unwrap_or_default();
    let summary = RosterSessionSummary {
        id: persisted_session_id.clone(),
        lifecycle: "live".to_string(),
        activity: "idle".to_string(),
        is_session_active: false,
        runtime_kind: Some("subagent".to_string()),
        rlm_depth: Some(edge.depth),
        session_id: persisted_session_id,
        session_file: Some(edge.child.clone()),
        session_name: (!edge.name.is_empty()).then(|| edge.name.clone()),
        cwd: Path::new(&edge.child).parent().map(|parent| parent.to_string_lossy().to_string()).unwrap_or_default(),
        is_streaming: false,
        is_compacting: false,
        attached_clients: 0,
        message_count: 0,
        parent_session_path: Some(edge.parent.clone()),
        rlm_child_id: Some(edge.child_id.clone()),
        ..RosterSessionSummary::default()
    };
    WorkerRosterEntry { agent_id: roster_agent_id_for_entry(&summary), queued_child: None, seeded_cwd: None, summary }
}

/// `hydrateSeededEntry(entry)`: only a row still marked `seededCwd` is re-read,
/// and only while it is still the store's current row.
async fn hydrate_seeded_roster_entry(entry: AgentRosterEntry, supervisor: &Supervisor) -> AgentRosterEntry {
    if entry.seeded_cwd != Some(true) || entry.summary.session_file.is_none() { return entry; }
    let worker_entry = WorkerRosterEntry { agent_id: entry.agent_id.clone(), queued_child: entry.queued_child, seeded_cwd: entry.seeded_cwd, summary: entry.summary.clone() };
    let hydrated = hydrated_seed_entry(&worker_entry).await;
    if hydrated.seeded_cwd == Some(true) { return entry; }
    let current = supervisor.roster().lock().unwrap().get(&entry.agent_id);
    match current {
        Some(current) if current.summary != entry.summary || current.status != entry.status => current,
        None => entry,
        Some(_) => supervisor.write_roster_entry(hydrated, None, entry.status_label.as_deref()),
    }
}

/// `hydratedSeedEntry(entry)`: fill `cwd` from the saved transcript when it exists.
async fn hydrated_seed_entry(entry: &WorkerRosterEntry) -> WorkerRosterEntry {
    let Some(session_file) = entry.summary.session_file.clone() else { return with_seeded_cwd(entry); };
    let Some(info) = crate::core::session_manager::read_session_info(&session_file).await else { return with_seeded_cwd(entry); };
    let mut hydrated = entry.clone();
    hydrated.seeded_cwd = None;
    hydrated.summary.cwd = info.cwd;
    hydrated
}

fn with_seeded_cwd(entry: &WorkerRosterEntry) -> WorkerRosterEntry {
    let mut seeded = entry.clone();
    seeded.seeded_cwd = Some(true);
    seeded
}

/// `matchesListSessionDir(summary, sessionDir, spawnParents)`.
fn matches_list_session_dir(summary: &SessionSummary, session_dir: Option<&str>, spawn_parents: &HashMap<String, String>) -> bool {
    let Some(session_dir) = session_dir else { return true; };
    let Some(session_file) = summary.session_file.as_ref() else { return false; };
    let mut file = resolve_path(session_file);
    let mut parent_session_path = summary.parent_session_path.clone();
    let mut visited: HashSet<String> = HashSet::new();
    while let Some(parent) = parent_session_path {
        let canonical = canonical_session_path(&parent);
        if visited.contains(&canonical) { break; }
        visited.insert(canonical.clone());
        file = resolve_path(&parent);
        parent_session_path = spawn_parents.get(&canonical).cloned();
    }
    Path::new(&file).parent().map(|parent| parent.to_string_lossy().to_string()).unwrap_or_default()
        == resolve_path(session_dir)
}

fn resolve_path(path: &str) -> String {
    let candidate = Path::new(path);
    if candidate.is_absolute() { candidate.to_string_lossy().to_string() }
    else {
        std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf())
            .join(candidate).to_string_lossy().to_string()
    }
}

/// `isSessionSummary(value)`: the daemon's summary guard for worker list payloads.
fn is_session_summary_value(value: &Value) -> bool {
    matches!(
        (value.get("id").and_then(Value::as_str), value.get("sessionId").and_then(Value::as_str), value.get("cwd").and_then(Value::as_str)),
        (Some(_), Some(_), Some(_))
    )
}

/// `sessionSummariesFromResponse(response)`: the worker `list` payload, validated.
fn session_summaries_from_response(response: DaemonResponse) -> Result<Vec<RosterSessionSummary>, String> {
    let data = response_data(response)?;
    let sessions = data.get("sessions").and_then(Value::as_array).ok_or("Session worker returned an invalid list response")?;
    if !sessions.iter().all(is_session_summary_value) { return Err("Session worker returned an invalid list response".to_string()); }
    Ok(sessions.iter().filter_map(|value| serde_json::from_value::<RosterSessionSummary>(value.clone()).ok()).collect())
}

/// `workerRosterEntryFromSummary` over the worker's wire summary.
fn worker_roster_entry_from_value(summary: &Value) -> Option<WorkerRosterEntry> {
    let summary: RosterSessionSummary = serde_json::from_value(summary.clone()).ok()?;
    Some(worker_roster_entry_from_summary(&summary))
}
/// D-04: convert a forwarded delivery response into a content-free journal record.
/// A receipt with an unexpected status is recorded as `uncertain`, never as delivered.
fn agent_message_delivery_record_from_response(
    source_active_session_id: Option<&str>,
    target_active_session_id: &str,
    response: &DaemonResponse,
) -> AgentMessageDeliveryRecord {
    if !response.success {
        // A failed forward provably did not deliver. Prefer the fixed reason code derived
        // from the message; the structured error code is already a fixed identifier.
        let reason = response
            .error
            .as_deref()
            .map(delivery_reason_code)
            .or_else(|| response.error_info.as_ref().map(|info| info.code()))
            .unwrap_or("worker_error");
        return AgentMessageDeliveryRecord::new(
            source_active_session_id,
            target_active_session_id,
            None,
            AgentMessageDeliveryOutcome::Rejected,
            Some(reason),
        );
    }
    let data = response.data.clone().unwrap_or(Value::Null);
    let message_id = data.get("id").and_then(Value::as_str);
    let receipt_target = data
        .get("target")
        .and_then(|target| target.get("activeSessionId"))
        .and_then(Value::as_str)
        .unwrap_or(target_active_session_id);
    let outcome = data
        .get("deliveryStatus")
        .and_then(Value::as_str)
        .and_then(AgentMessageDeliveryOutcome::from_receipt_status);
    let outcome = match outcome {
        Some(outcome) => outcome,
        None => {
            // A success response without a recognized status is not proof of delivery.
            return AgentMessageDeliveryRecord::new(
                source_active_session_id,
                target_active_session_id,
                None,
                AgentMessageDeliveryOutcome::Uncertain,
                Some("invalid_receipt"),
            );
        }
    };
    AgentMessageDeliveryRecord::new(
        source_active_session_id,
        receipt_target,
        message_id,
        outcome,
        None,
    )
}

fn command(kind: &str) -> Map<String, Value> { json!({"type":kind}).as_object().unwrap().clone() }
fn response_data(response: DaemonResponse) -> Result<Value, String> { if response.success { Ok(response.data.unwrap_or(Value::Null)) } else { Err(response.error.unwrap_or_else(|| "Session worker request failed".to_string())) } }
fn persist_json(path: &Path, value: &Value) -> Result<(), String> {
    write_file_atomic_sync(&path.to_string_lossy(), &serde_json::to_string(value).map_err(|error| error.to_string())?, WriteFileAtomicOptions { mode: Some(0o600), fsync: true, fsync_dir: true, ..Default::default() }).map_err(|error| error.to_string())
}
fn parse_command(line: &[u8]) -> Result<(Map<String, Value>, Option<String>, Option<u64>), String> {
    let value: Value = serde_json::from_slice(line).map_err(|error| error.to_string())?;
    let object = value.as_object().ok_or("Invalid daemon command")?;
    if object.get("type").and_then(Value::as_str) == Some("command") {
        if !daemon_protocol::is_daemon_command_envelope(&value) { return Err("Invalid daemon command envelope".into()); }
        let mut command = object.get("command").and_then(Value::as_object).ok_or("Invalid daemon command envelope")?.clone();
        command.insert("id".into(), object.get("id").cloned().ok_or("Daemon command requires id")?);
        let protocol_version = object.get("protocol").and_then(|protocol| protocol.get("version")).and_then(Value::as_u64);
        Ok((command, Some(object.get("clientId").and_then(Value::as_str).unwrap_or("").to_string()), protocol_version))
    } else { Ok((object.clone(), None, None)) }
}
fn spawn_connection<S>(supervisor: Arc<Supervisor>, stream: S) where S: AsyncRead + AsyncWrite + Unpin + Send + 'static {
    tokio::spawn(async move {
        let (mut input, mut output) = tokio::io::split(stream);
        let (sender, mut receiver) = mpsc::channel::<Vec<u8>>(1024);
        let identity = create_active_session_id(None);
        let public = Arc::new(PublicClient {
            connection_id: identity.clone(), id: Mutex::new(identity.clone()), protocol_id: Mutex::new(None),
            subscriptions: Mutex::new(HashSet::new()), supports_extension_ui: AtomicBool::new(false), pause_epoch: AtomicU64::new(0),
            output: sender, stopped: CancellationToken::new(), roster_subscribed: AtomicBool::new(false),
            roster_resync_pending: AtomicBool::new(false), backpressured: AtomicBool::new(false),
        });
        supervisor.clients.lock().unwrap().insert(identity, public.clone());
        public.write(&supervisor.hello(&public));
        let cancelled = public.stopped.clone();
        let writer_public = Arc::clone(&public);
        let writer_supervisor = Arc::clone(&supervisor);
        let writer = tokio::spawn(async move {
            loop {
                tokio::select! { biased;
                    bytes = receiver.recv() => match bytes {
                        Some(bytes) => {
                            if output.write_all(&bytes).await.is_err() { break; }
                            // The socket drained: a resync deferred under backpressure
                            // is written once, covering the whole loss gap.
                            if writer_public.backpressured.swap(false, Ordering::SeqCst)
                                && writer_public.roster_subscribed.load(Ordering::SeqCst)
                                && writer_public.roster_resync_pending.swap(false, Ordering::SeqCst)
                            {
                                let changed = writer_supervisor.roster_entries_for_client();
                                writer_public.write(&roster_update_value(&changed, &[], Some(true)));
                            }
                        },
                        None => break,
                    },
                    _ = cancelled.cancelled() => break,
                }
            }
            cancelled.cancel(); let _ = output.shutdown().await;
        });
        let mut pending = Vec::new(); let mut buffer = [0u8; 8192];
        loop {
            let read = tokio::select! { _ = supervisor.stopped.cancelled() => break, _ = public.stopped.cancelled() => break, read = input.read(&mut buffer) => read };
            let size = match read { Ok(0) | Err(_) => break, Ok(size) => size };
            for &byte in &buffer[..size] {
                if byte == b'\n' {
                    if !pending.is_empty() { tokio::spawn(supervisor.clone().handle_line(public.clone(), std::mem::take(&mut pending))); }
                } else { pending.push(byte); if pending.len() > MAX_PUBLIC_LINE { public.stopped.cancel(); break; } }
            }
        }
        public.stopped.cancel(); let _ = writer.await;
        supervisor.disconnected(&public);
    });
}

#[cfg(unix)]
fn spawn_worker_process(
    program: &str,
    args: Vec<String>,
    _socket: &str,
    cwd: Option<&str>,
    mut environment: HashMap<String, String>,
) -> Result<(tokio::process::Child, std::fs::File), String> {
    use std::os::fd::{AsRawFd, OwnedFd};
    // std creates close-on-exec handles, so concurrent process launches cannot
    // inherit the parent gate and keep the worker waiting for EOF.
    let (read, write) = std::os::unix::net::UnixStream::pair().map_err(|error| error.to_string())?;
    let read = std::fs::File::from(OwnedFd::from(read));
    let write = std::fs::File::from(OwnedFd::from(write));
    environment.insert(DAEMON_WORKER_STARTUP_GATE_FD_ENV.to_string(), "3".to_string());
    let mut process = tokio::process::Command::new(program);
    process.args(&args).env_clear().envs(environment)
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::piped());
    if let Some(cwd) = cwd { process.current_dir(cwd); }
    let fd = read.as_raw_fd();
    unsafe { process.pre_exec(move || {
        if libc::setsid() == -1 { return Err(std::io::Error::last_os_error()); }
        if fd != 3 && libc::dup2(fd, 3) == -1 { return Err(std::io::Error::last_os_error()); }
        if libc::fcntl(3, libc::F_SETFD, 0) == -1 { return Err(std::io::Error::last_os_error()); }
        Ok(())
    }); }
    let child = process.spawn().map_err(|error| error.to_string())?;
    drop(read);
    Ok((child, write))
}
#[cfg(windows)]
fn spawn_worker_process(
    program: &str,
    _args: Vec<String>,
    _socket: &str,
    _cwd: Option<&str>,
    _environment: HashMap<String, String>,
) -> Result<(tokio::process::Child, tokio::process::ChildStdin), String> {
    let mut environment = _environment;
    environment.insert(DAEMON_WORKER_STARTUP_GATE_FD_ENV.to_string(), "stdin".to_string());
    let mut process = tokio::process::Command::new(program);
    process.args(_args).env_clear().envs(environment)
        // TS spawnHidden detached (daemon-supervisor.ts:3333-3337): DETACHED_PROCESS
        // | CREATE_NEW_PROCESS_GROUP plus windowsHide; stderr piped so the jsonl
        // reader can forward worker stderr to the supervisor log.
        .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::piped());
    if let Some(cwd) = _cwd { process.current_dir(cwd); }
    process.creation_flags(0x00000208 | 0x08000000);
    let mut child = process.spawn().map_err(|error| error.to_string())?;
    let gate = child.stdin.take().ok_or("Failed to create daemon worker startup gate")?;
    Ok((child, gate))
}
#[cfg(unix)]
async fn commit_gate(mut gate: std::fs::File) -> Result<(), String> { std::io::Write::write_all(&mut gate, DAEMON_WORKER_STARTUP_GATE_COMMIT.as_bytes()).map_err(|error| error.to_string()) }
#[cfg(windows)]
async fn commit_gate(mut gate: tokio::process::ChildStdin) -> Result<(), String> {
    gate.write_all(DAEMON_WORKER_STARTUP_GATE_COMMIT.as_bytes()).await.map_err(|error| error.to_string())?;
    gate.shutdown().await.map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn backlog_recovery_and_disconnect_registry_lock_order() {
        use super::daemon_supervisor_parity_tests::{SupervisorFixture, add_descriptor_only_worker};
        let fixture = SupervisorFixture::new("backlog-lock-order").await;
        let worker = add_descriptor_only_worker(&fixture, "worker", "root", "token", DAEMON_WORKER_LIFECYCLE_FAILED);
        worker.descriptor.lock().unwrap().owner_client_id = Some("owner".into());
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let recovery = {
            let supervisor = fixture.supervisor.clone(); let worker = worker.clone(); let barrier = barrier.clone();
            std::thread::spawn(move || {
                for _ in 0..1000 { barrier.wait(); assert!(supervisor.is_worker_recovery_candidate(&worker)); assert!(!supervisor.is_worker_recovery_cancelled(&worker)); }
            })
        };
        for _ in 0..1000 { barrier.wait(); assert_eq!(fixture.supervisor.owned_worker_candidates("owner").len(), 1); }
        recovery.join().unwrap();
        let replacement = add_descriptor_only_worker(&fixture, "worker", "other-root", "other-token", DAEMON_WORKER_LIFECYCLE_FAILED);
        assert!(!fixture.supervisor.is_worker_recovery_candidate(&worker));
        assert!(fixture.supervisor.is_worker_recovery_cancelled(&worker));
        assert!(fixture.supervisor.stop_worker(&worker, true, false).await.unwrap_err().contains("replaced during stop"));
        assert!(Arc::ptr_eq(fixture.supervisor.workers.lock().unwrap().get("worker").unwrap(), &replacement));
    }
    #[test]
    fn descriptor_and_socket_paths_match_the_typescript_hash_contract() {
        assert_eq!(descriptor_key("/tmp/daemon.sock"), format!("{:x}", Sha256::digest(b"/tmp/daemon.sock"))[..12]);
        assert!(worker_socket("/tmp/daemon.sock", "abcdef1234567890").contains("abcdef123456"));
    }
    #[test]
    fn command_envelopes_preserve_public_request_and_client_identity() {
        let wire = json!({"type":"command", "id":"request-1", "clientId":"client-1", "protocol":daemon_protocol::daemon_protocol_info(), "command":{"type":"list"}});
        let (body, client, protocol) = parse_command(&serde_json::to_vec(&wire).unwrap()).unwrap();
        assert_eq!(protocol, wire["protocol"]["version"].as_u64());
        assert_eq!(body["id"], "request-1"); assert_eq!(client.as_deref(), Some("client-1"));
    }
    /// `failure(salvageDaemonCommandId(line), "parse", error)` (daemon-supervisor.ts:1939): a
    /// malformed line still yields the sender's id, and a line with no string id yields none.
    #[test]
    fn parse_failures_salvage_the_command_id() {
        assert_eq!(daemon_protocol::salvage_daemon_command_id(r#"{"id":"request-9","command":{}}"#).as_deref(), Some("request-9"));
        assert_eq!(daemon_protocol::salvage_daemon_command_id(r#"{"id":7}"#), None);
        assert_eq!(daemon_protocol::salvage_daemon_command_id("not json"), None);
    }
    /// `promptAdmissionKey(activeSessionId, publicAdmissionId)` (daemon-supervisor.ts:1822-1823) is
    /// `\`${activeSessionId}\0${publicAdmissionId}\``, scoped by the owning socket (:749).
    #[test]
    fn prompt_admission_keys_are_socket_and_session_scoped() {
        let first = prompt_admission_key("conn-1", "active-1", "admission-1");
        let second = prompt_admission_key("conn-2", "active-1", "admission-1");
        let third = prompt_admission_key("conn-1", "active-2", "admission-1");
        assert_eq!(first, "conn-1\u{0}active-1\u{0}admission-1");
        assert_ne!(first, second);
        assert_ne!(first, third);
    }

    /// G-30: the daemon worker spawn must keep the worker fully detached when
    /// the launcher closes AND its stderr must be observable by the supervisor
    /// (daemon-supervisor.ts:3333-3345: `spawnHidden(..., detached: true,
    /// stdio: ["ignore", "ignore", "pipe", "pipe"])` plus the jsonl stderr
    /// reader). The port inherited stderr and skipped the detach flag.
    #[cfg(windows)]
    #[tokio::test]
    async fn worker_stderr_is_observable_and_worker_survives_launcher_close() {
        use std::time::Duration;

        // cmd.exe needs a minimal viable environment (the spawn clears env).
        let mut environment: HashMap<String, String> = HashMap::new();
        environment.insert("PARITY_SENTINEL".to_string(), "1".to_string());
        for (name, value) in [
            ("SystemRoot", std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string())),
            ("SystemDrive", std::env::var("SystemDrive").unwrap_or_else(|_| "C:".to_string())),
            ("PATH", std::env::var("PATH").unwrap_or_default()),
            ("PATHEXT", std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())),
            ("COMSPEC", std::env::var("COMSPEC").unwrap_or_else(|_| "C:\\Windows\\system32\\cmd.exe".to_string())),
            ("TEMP", std::env::var("TEMP").unwrap_or_else(|_| "C:\\Windows\\Temp".to_string())),
            ("TMP", std::env::var("TMP").unwrap_or_else(|_| "C:\\Windows\\Temp".to_string())),
        ] {
            environment.insert(name.to_string(), value);
        }
        let (mut child, gate) = spawn_worker_process(
            "cmd",
            vec![
                "/c".to_string(),
                "echo PARITY-SENTINEL-STDERR 1>&2 & ping -n 12 127.0.0.1 > nul".to_string(),
            ],
            "socket-not-used",
            None,
            environment,
        )
        .expect("worker spawn");
        let stderr = child
            .stderr
            .take()
            .expect("worker stderr must be piped to the supervisor, not inherited");
        let pid = child.id().expect("worker pid") as i32;
        let reader = tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(stderr);
            let mut first = String::new();
            let _ = lines.read_line(&mut first).await;
            first
        });
        // Launcher close: drop the gate and the child handle without killing.
        drop(gate);
        drop(child);
        tokio::time::sleep(Duration::from_millis(800)).await;
        assert!(
            crate::utils::child_process::process_id_exists(pid),
            "detached worker died when the launcher dropped its handles"
        );
        let sentinel = tokio::time::timeout(Duration::from_secs(5), reader)
            .await
            .unwrap_or(Ok(String::new()))
            .unwrap_or_default();
        assert!(
            sentinel.contains("PARITY-SENTINEL-STDERR"),
            "supervisor never received the worker stderr sentinel: {sentinel:?}"
        );
        // Cleanup: kill only this test-owned worker.
        crate::utils::child_process::signal_process_group_or_process(pid, Signal::Kill);
        for _ in 0..20 {
            if !crate::utils::child_process::process_id_exists(pid) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}
