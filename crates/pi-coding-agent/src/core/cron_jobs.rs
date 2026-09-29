//! Port of packages/coding-agent/src/core/cron-jobs.ts

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value};

use crate::core::session_manager::get_session_artifact_path_for_file;
use crate::utils::atomic_file::{write_file_atomic_sync, WriteFileAtomicOptions};

pub type AgentCronJobStatus = String;
pub type AgentCronScheduleKind = String;
pub type AgentCronJobSource = String;
pub type AgentCronJobRuntimeKind = String;
pub type AgentHeartbeatUpdateAction = String;
pub type AgentHeartbeatManagementAction = String;
pub type AgentRlmHeartbeatStatusUpdate = String;
pub type AgentHeartbeatDeliveryMode = String;

pub const STATUS_ACTIVE: &str = "active";
pub const STATUS_PAUSED: &str = "paused";
pub const STATUS_COMPLETED: &str = "completed";
pub const STATUS_CANCELLED: &str = "cancelled";

pub const SCHEDULE_ONCE: &str = "once";
pub const SCHEDULE_CRON: &str = "cron";
pub const SCHEDULE_INTERVAL: &str = "interval";

pub const SOURCE_CRON: &str = "cron";
pub const SOURCE_HEARTBEAT: &str = "heartbeat";
pub const SOURCE_RLM_HEARTBEAT: &str = "rlm_heartbeat";

pub const RUNTIME_KIND_TOP_LEVEL: &str = "top-level";
pub const RUNTIME_KIND_SUBAGENT: &str = "subagent";

pub const DELIVERY_MODE_STEER: &str = "steer";
pub const DELIVERY_MODE_FOLLOW_UP: &str = "follow_up";

/// `interface AgentCronSchedule`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentCronSchedule {
    pub kind: AgentCronScheduleKind,
    pub expression: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval_ms: Option<f64>,
}

/// `interface AgentCronJob`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentCronJob {
    pub id: String,
    pub status: AgentCronJobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<AgentCronJobSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_kind: Option<AgentCronJobRuntimeKind>,
    /// Delivery mode for heartbeat/rlm_heartbeat jobs when the session is busy.
    /// Defaults to "steer".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delivery_mode: Option<AgentHeartbeatDeliveryMode>,
    pub active_session_id: String,
    pub session_id: String,
    pub session_file: String,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub prompt: String,
    pub schedule: AgentCronSchedule,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_run_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_run_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_skipped_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub run_count: f64,
}

/// `interface CreateAgentCronJobInput`.
#[derive(Debug, Clone, Default)]
pub struct CreateAgentCronJobInput {
    pub active_session_id: String,
    pub session_id: String,
    pub session_file: String,
    pub cwd: String,
    pub label: Option<String>,
    pub prompt: String,
    pub schedule_text: String,
    pub source: Option<AgentCronJobSource>,
    pub runtime_kind: Option<AgentCronJobRuntimeKind>,
    pub delivery_mode: Option<AgentHeartbeatDeliveryMode>,
    /// `now?: Date`, in milliseconds.
    pub now: Option<f64>,
}

pub type AgentCronJobRunResult = String;
pub const RUN_RESULT_RAN: &str = "ran";
pub const RUN_RESULT_SKIPPED: &str = "skipped";

/// `interface AgentCronDispatch`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentCronDispatch {
    pub id: String,
    pub job: AgentCronJob,
}

/// `interface AgentCronSchedulerHooks`.
///
/// TS `runJob` may reject, and `catch (runError)` records the error and calls `onError`
/// (cron-jobs.ts:1010-1015). `run_job` cannot reject, so the throwing form is supplied through
/// `AgentCronScheduler::enable_run_job_errors` instead of a new field here: this struct's single
/// non-test literal lives in another slice (`AgentDaemon::start_cron_scheduler`,
/// modes/daemon/daemon_mode.rs:7035). Blocked owner symbol: `AgentDaemon::run_cron_job`
/// (modes/daemon/daemon_mode.rs:10441) returns `Option<AgentCronJobRunResult>` and must return
/// `Result<Option<AgentCronJobRunResult>, String>` (it drops the queued-prompt error at
/// daemon_mode.rs:10482-10491, where daemon-mode.ts:2100-2105 rethrows) before the daemon can
/// supply that closure.
pub struct AgentCronSchedulerHooks {
    pub run_job: Arc<dyn Fn(AgentCronJob) -> BoxFuture<Option<AgentCronJobRunResult>> + Send + Sync>,
    /// `beginDispatch?: (dispatch) => (() => void) | undefined`.
    pub begin_dispatch: Option<Arc<dyn Fn(&AgentCronDispatch) -> Option<Arc<dyn Fn() + Send + Sync>> + Send + Sync>>,
    pub now: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
    pub on_error: Option<Arc<dyn Fn(&AgentCronJob, String) + Send + Sync>>,
}

pub type BoxFuture<T> = pi_ai::types::BoxFuture<T>;

/// `interface HeartbeatCronSessionActivity`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct HeartbeatCronSessionActivity {
    pub is_streaming: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_compacting: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_retrying: Option<bool>,
    pub is_bash_running: bool,
    pub has_pending_session_work: bool,
    pub unfinished_action_count: f64,
}

/// `interface CronJobsFile`.
#[derive(Debug, Clone, Default)]
struct CronJobsFile {
    jobs: Option<Value>,
    dispatches: Option<Value>,
}

/// `interface AgentCronDispatchRecord`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentCronDispatchRecord {
    id: String,
    job_id: String,
    claimed_at: String,
    scheduled_for: String,
}

/// `interface CronJobsState`.
#[derive(Debug, Clone, Default, PartialEq)]
struct CronJobsState {
    jobs: Vec<AgentCronJob>,
    dispatches: Vec<AgentCronDispatchRecord>,
}

pub const SESSION_SCHEDULED_JOBS_FILENAME: &str = "scheduled-jobs.json";

const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;
const ONE_SECOND_MS: f64 = 1000.0;
const ONE_MINUTE_MS: f64 = 60_000.0;
pub const DEFAULT_HEARTBEAT_SCHEDULE: &str = "every 5m";
pub const DEFAULT_HEARTBEAT_DELIVERY_MODE: &str = "steer";

/// `type ParsedHeartbeatCommand`.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedHeartbeatCommand {
    Status,
    Pause,
    Resume,
    Clear,
    Set {
        schedule: String,
        instruction: String,
        delivery_mode: Option<AgentHeartbeatDeliveryMode>,
    },
}

/// `interface AgentRlmHeartbeatController`.
pub trait AgentRlmHeartbeatController: Send + Sync {
    fn list_rlm_heartbeats(&self, options: Option<RlmHeartbeatListOptions>) -> Vec<AgentCronJob>;
    fn create_rlm_heartbeat(&self, input: RlmHeartbeatCreateInput) -> AgentCronJob;
    fn update_rlm_heartbeat(&self, input: RlmHeartbeatUpdateInput) -> Option<AgentCronJob>;
    fn delete_rlm_heartbeat(&self, id: &str) -> Option<AgentCronJob>;
}

/// `listRlmHeartbeats(options?: { includeInactive?: boolean })`.
#[derive(Debug, Clone, Default)]
pub struct RlmHeartbeatListOptions {
    pub include_inactive: Option<bool>,
}

/// `createRlmHeartbeat(input)`.
#[derive(Debug, Clone, Default)]
pub struct RlmHeartbeatCreateInput {
    pub instruction: String,
    pub interval: Option<String>,
    pub label: Option<String>,
    pub delivery_mode: Option<AgentHeartbeatDeliveryMode>,
}

/// `updateRlmHeartbeat(input)`.
#[derive(Debug, Clone, Default)]
pub struct RlmHeartbeatUpdateInput {
    pub id: String,
    pub instruction: Option<String>,
    pub interval: Option<String>,
    pub label: Option<String>,
    pub status: Option<AgentRlmHeartbeatStatusUpdate>,
    pub delivery_mode: Option<AgentHeartbeatDeliveryMode>,
}

/// `heartbeatCatalogSignature(jobs)`.
fn heartbeat_catalog_signature(jobs: &[AgentCronJob]) -> String {
    let mut filtered: Vec<&AgentCronJob> = jobs
        .iter()
        .filter(|job| {
            is_heartbeat_cron_job(job) && (job.status == STATUS_ACTIVE || job.status == STATUS_PAUSED)
        })
        .collect();
    filtered.sort_by(|left, right| left.id.cmp(&right.id));
    let projected: Vec<Value> = filtered
        .into_iter()
        .map(|job| {
            let mut object = Map::new();
            object.insert("id".to_string(), Value::String(job.id.clone()));
            object.insert("status".to_string(), Value::String(job.status.clone()));
            insert_optional(&mut object, "source", &job.source);
            insert_optional(&mut object, "runtimeKind", &job.runtime_kind);
            insert_optional(&mut object, "deliveryMode", &job.delivery_mode);
            object.insert("activeSessionId".to_string(), Value::String(job.active_session_id.clone()));
            object.insert("sessionId".to_string(), Value::String(job.session_id.clone()));
            object.insert("sessionFile".to_string(), Value::String(job.session_file.clone()));
            object.insert("cwd".to_string(), Value::String(job.cwd.clone()));
            insert_optional(&mut object, "label", &job.label);
            object.insert("prompt".to_string(), Value::String(job.prompt.clone()));
            object.insert(
                "schedule".to_string(),
                serde_json::to_value(&job.schedule).unwrap_or(Value::Null),
            );
            object.insert("createdAt".to_string(), Value::String(job.created_at.clone()));
            Value::Object(object)
        })
        .collect();
    serde_json::to_string(&projected).unwrap_or_else(|_| "[]".to_string())
}

/// `JSON.stringify` drops `undefined` members; an absent optional key must not
/// become JSON `null` in the catalog signature.
fn insert_optional(object: &mut Map<String, Value>, key: &str, value: &Option<String>) {
    if let Some(value) = value {
        object.insert(key.to_string(), Value::String(value.clone()));
    }
}

// ---------------------------------------------------------------------------
// Filesystem helpers (node:fs equivalents)
// ---------------------------------------------------------------------------

fn mkdir_recursive_mode_700(path: &Path) {
    let _ = std::fs::create_dir_all(path);
}

/// `lockSync(path, { realpath: false, lockfilePath: `${path}.lock`, stale: 30_000 })`.
///
/// The TypeScript retries `ELOCKED` up to 100 times, waiting ~10ms using
/// `Atomics.wait` on a shared buffer (a synchronous sleep). The Rust port keeps
/// the same `<path>.lock` file, the same 100-attempt/10ms retry budget and the
/// same 30s stale takeover, and adds an in-process coordinator so two tokio
/// tasks do not corrupt one state file.
struct CronJobsLockGuard {
    lock_path: PathBuf,
}

impl Drop for CronJobsLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

fn process_coordinator() -> &'static Mutex<()> {
    static COORDINATOR: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();
    COORDINATOR.get_or_init(|| Mutex::new(()))
}

thread_local! {
    /// Paths this thread already locked; `lockSync` is re-entrant here so an
    /// inner call cannot deadlock against the coordinator.
    static THREAD_HELD_PATHS: std::cell::RefCell<BTreeSet<String>> =
        std::cell::RefCell::new(BTreeSet::new());
}

/// `withCronJobsStateLocks(paths, action)`.
fn with_cron_jobs_state_locks<T>(paths: &[String], action: impl FnOnce() -> T) -> Result<T, String> {
    let mut unique: Vec<String> = paths.to_vec();
    unique.sort();
    unique.dedup();
    let already_held = THREAD_HELD_PATHS.with(|held| held.borrow().iter().cloned().collect::<BTreeSet<String>>());
    let needed: Vec<String> = unique
        .iter()
        .filter(|path| !already_held.contains(*path))
        .cloned()
        .collect();
    if needed.is_empty() {
        return Ok(action());
    }

    // 100 attempts x 10ms, matching the TypeScript `ELOCKED` retry budget.
    let mut coordinator_guard = None;
    for attempt in 0..100 {
        match process_coordinator().try_lock() {
            Ok(guard) => {
                coordinator_guard = Some(guard);
                break;
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                if attempt == 99 {
                    return Err(format!("Could not coordinate scheduled jobs: {}", needed[0]));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(format!("Could not coordinate scheduled jobs: {}", needed[0]));
            }
        }
    }
    let coordinator_guard = coordinator_guard.expect("loop returns when it cannot acquire");

    let mut guards: Vec<CronJobsLockGuard> = Vec::new();
    for path in &needed {
        if let Some(parent) = Path::new(path).parent() {
            mkdir_recursive_mode_700(parent);
        }
        let lock_path = PathBuf::from(format!("{path}.lock"));
        let mut acquired = false;
        for attempt in 0..100 {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock_path) {
                Ok(_) => {
                    guards.push(CronJobsLockGuard { lock_path });
                    acquired = true;
                    break;
                }
                Err(error) => {
                    let stale = std::fs::metadata(&lock_path)
                        .and_then(|metadata| metadata.modified())
                        .ok()
                        .and_then(|modified| modified.elapsed().ok())
                        .map(|elapsed| elapsed.as_millis() > 30_000)
                        .unwrap_or(false);
                    if stale {
                        let _ = std::fs::remove_file(&lock_path);
                        continue;
                    }
                    if error.kind() != std::io::ErrorKind::AlreadyExists {
                        return Err(error.to_string());
                    }
                    if attempt == 99 {
                        return Err(format!("Could not coordinate scheduled jobs: {path}"));
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
        if !acquired {
            return Err(format!("Could not coordinate scheduled jobs: {path}"));
        }
    }

    THREAD_HELD_PATHS.with(|held| {
        let mut held = held.borrow_mut();
        for path in &needed {
            held.insert(path.clone());
        }
    });
    let result = action();
    THREAD_HELD_PATHS.with(|held| {
        let mut held = held.borrow_mut();
        for path in &needed {
            held.remove(path);
        }
    });
    // `releases.reverse()` in the TypeScript; dropping in reverse keeps that order.
    while let Some(guard) = guards.pop() {
        drop(guard);
    }
    drop(coordinator_guard);
    Ok(result)
}

/// `readJobsState(path)`.
///
/// The parse error is returned to the caller, which records a `SettingsError`-style diagnostic and
/// treats the state as empty; a corrupt file is copied to `<path>.corrupt` first so the next
/// `writeJobsState` (cron-jobs.ts:1552-1555) can never be the thing that deletes every cron job.
fn read_jobs_state(path: &str) -> Result<CronJobsState, String> {
    if !Path::new(path).exists() {
        return Ok(CronJobsState::default());
    }
    // `JSON.parse(readFileSync(path, "utf-8"))` throws (cron-jobs.ts:1537); the empty state is
    // returned only when the file does not exist at all (1534-1536).
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: Value = serde_json::from_str(&text).map_err(|error| error.to_string())?;
    let parsed: CronJobsFile = CronJobsFile {
        jobs: value.get("jobs").cloned(),
        dispatches: value.get("dispatches").cloned(),
    };
    Ok(CronJobsState {
        jobs: parsed
            .jobs
            .as_ref()
            .and_then(|value| value.as_array())
            .map(|array| array.iter().filter_map(is_agent_cron_job).collect())
            .unwrap_or_default(),
        dispatches: parsed
            .dispatches
            .as_ref()
            .and_then(|value| value.as_array())
            .map(|array| array.iter().filter_map(is_agent_cron_dispatch_record).collect())
            .unwrap_or_default(),
    })
}

/// State for reading callers: a corrupt file is preserved and reported, then read as empty so the
/// command that reads it still answers (`JSON.parse` would have thrown, cron-jobs.ts:1537).
fn read_jobs_state_lossy(path: &str) -> CronJobsState {
    match read_jobs_state(path) {
        Ok(state) => state,
        Err(error) => {
            preserve_corrupt_jobs_file(path, Some(&error));
            CronJobsState::default()
        }
    }
}

/// Keep the unparsable jobs file next to the original (`<path>.corrupt`) so the jobs it still
/// holds survive the empty-state fallback that `JSON.parse` would have prevented (cron-jobs.ts:1537).
fn preserve_corrupt_jobs_file(path: &str, reason: Option<&str>) {
    let backup = format!("{path}.corrupt");
    if Path::new(&backup).exists() {
        return;
    }
    if std::fs::copy(path, &backup).is_err() {
        return;
    }
    if let Some(reason) = reason {
        let note = format!("Cron jobs file could not be parsed; kept at {backup}: {reason}\n");
        let _ = std::fs::write(format!("{backup}.reason"), note);
    }
}

/// `writeJobsState(path, state)`.
///
/// `writeFileAtomicSync` throws, so a failed write must not look like a successful one
/// (cron-jobs.ts:1552-1555): the claim/recover cycle would otherwise proceed as if the
/// dispatch had been persisted.
fn write_jobs_state(path: &str, state: &CronJobsState) -> Result<(), String> {
    if let Some(parent) = Path::new(path).parent() {
        mkdir_recursive_mode_700(parent);
    }
    let serialized = serde_json::to_string_pretty(&state_to_value(state)).unwrap_or_else(|_| "{}".to_string());
    write_file_atomic_sync(
        path,
        &format!("{serialized}\n"),
        WriteFileAtomicOptions {
            mode: Some(0o600),
            fsync: true,
            fsync_dir: true,
            before_rename: None,
        },
    )
    .map_err(|error| error.to_string())
}

fn state_to_value(state: &CronJobsState) -> Value {
    let mut object = Map::new();
    object.insert(
        "jobs".to_string(),
        Value::Array(
            state
                .jobs
                .iter()
                .map(|job| serde_json::to_value(job).unwrap_or(Value::Null))
                .collect(),
        ),
    );
    object.insert(
        "dispatches".to_string(),
        Value::Array(
            state
                .dispatches
                .iter()
                .map(|dispatch| serde_json::to_value(dispatch).unwrap_or(Value::Null))
                .collect(),
        ),
    );
    Value::Object(object)
}

/// `writeJobsFile(path, jobs, mergeCurrent)`.
fn write_jobs_file(path: &str, jobs: &[AgentCronJob], merge_current: bool) -> Result<(), String> {
    let current = read_jobs_state(path)?;
    write_jobs_state(
        path,
        &CronJobsState {
            jobs: if merge_current {
                merge_fresh_jobs(&current.jobs, jobs)
            } else {
                jobs.to_vec()
            },
            dispatches: current.dispatches,
        },
    )
}

fn is_agent_cron_job(value: &Value) -> Option<AgentCronJob> {
    let job: AgentCronJob = serde_json::from_value(value.clone()).ok()?;
    let schedule = value.get("schedule");
    let kind = schedule.and_then(|schedule| schedule.get("kind")).and_then(|kind| kind.as_str());
    if !matches!(kind, Some(SCHEDULE_ONCE) | Some(SCHEDULE_CRON) | Some(SCHEDULE_INTERVAL)) {
        return None;
    }
    if kind == Some(SCHEDULE_INTERVAL) {
        let interval = schedule.and_then(|schedule| schedule.get("intervalMs"));
        if !interval.map(|value| value.as_f64().map(|number| number > 0.0).unwrap_or(false)).unwrap_or(false) {
            return None;
        }
    }
    if !matches!(
        job.status.as_str(),
        STATUS_ACTIVE | STATUS_PAUSED | STATUS_COMPLETED | STATUS_CANCELLED
    ) {
        return None;
    }
    if let Some(source) = &job.source {
        if !matches!(source.as_str(), SOURCE_CRON | SOURCE_HEARTBEAT | SOURCE_RLM_HEARTBEAT) {
            return None;
        }
    }
    if let Some(runtime_kind) = &job.runtime_kind {
        if !matches!(runtime_kind.as_str(), RUNTIME_KIND_TOP_LEVEL | RUNTIME_KIND_SUBAGENT) {
            return None;
        }
    }
    if let Some(delivery_mode) = &job.delivery_mode {
        if !matches!(delivery_mode.as_str(), DELIVERY_MODE_STEER | DELIVERY_MODE_FOLLOW_UP) {
            return None;
        }
    }
    Some(job)
}

fn is_agent_cron_dispatch_record(value: &Value) -> Option<AgentCronDispatchRecord> {
    serde_json::from_value(value.clone()).ok()
}

/// `randomUUID()`.
fn random_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

// ---------------------------------------------------------------------------
// AgentCronJobStore
// ---------------------------------------------------------------------------

/// `class AgentCronJobStore`.
pub struct AgentCronJobStore {
    file_path: Option<String>,
    session_artifact_mode: bool,
    session_artifact_files: Mutex<HashMap<String, String>>,
    heartbeat_change_listeners: Arc<Mutex<Vec<Option<Arc<dyn Fn() + Send + Sync>>>>>,
}

impl std::fmt::Debug for AgentCronJobStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentCronJobStore")
            .field("file_path", &self.file_path)
            .field("session_artifact_mode", &self.session_artifact_mode)
            .finish_non_exhaustive()
    }
}

/// `updateRlmHeartbeat(activeSessionId, id, update)`.
#[derive(Debug, Clone, Default)]
pub struct RlmHeartbeatUpdate {
    pub label: Option<String>,
    pub prompt: Option<String>,
    pub schedule_text: Option<String>,
    pub status: Option<AgentRlmHeartbeatStatusUpdate>,
    pub delivery_mode: Option<AgentHeartbeatDeliveryMode>,
    /// `now?: Date`, in milliseconds.
    pub now: Option<f64>,
}

/// `cancelJobsForSession(input, now)`.
#[derive(Debug, Clone, Default)]
pub struct CancelJobsForSessionInput {
    pub active_session_id: Option<String>,
    pub session_id: Option<String>,
    pub session_file: Option<String>,
}

impl AgentCronJobStore {
    /// `new AgentCronJobStore(filePath?, sessionArtifactMode = false)`.
    pub fn new(file_path: Option<String>, session_artifact_mode: bool) -> Result<Self, String> {
        if file_path.is_none() && !session_artifact_mode {
            return Err("Cron job store requires a file path".to_string());
        }
        Ok(Self {
            file_path,
            session_artifact_mode,
            session_artifact_files: Mutex::new(HashMap::new()),
            heartbeat_change_listeners: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// `AgentCronJobStore.forSessionArtifacts()`.
    pub fn for_session_artifacts() -> Self {
        Self::new(None, true).expect("session artifact mode has no file path requirement")
    }

    /// `onHeartbeatChange(listener)`; the returned closure unsubscribes.
    pub fn on_heartbeat_change(&self, listener: Arc<dyn Fn() + Send + Sync>) -> Arc<dyn Fn() + Send + Sync> {
        let listeners = Arc::clone(&self.heartbeat_change_listeners);
        let index = {
            let mut guard = listeners.lock().expect("listeners poisoned");
            guard.push(Some(listener));
            guard.len() - 1
        };
        Arc::new(move || {
            let mut guard = listeners.lock().expect("listeners poisoned");
            if index < guard.len() {
                guard[index] = None;
            }
        })
    }

    /// `registerSessionArtifact(sessionId, artifactDir)`.
    pub fn register_session_artifact(&self, session_id: &str, artifact_dir: &str) -> bool {
        if !self.session_artifact_mode {
            return false;
        }
        let path = join_path(artifact_dir, SESSION_SCHEDULED_JOBS_FILENAME);
        let mut files = self.session_artifact_files.lock().expect("artifact files poisoned");
        if files.get(session_id) == Some(&path) {
            return false;
        }
        files.insert(session_id.to_string(), path);
        true
    }

    /// `recoverSessionArtifact(sessionId, now = new Date())`.
    pub fn recover_session_artifact(&self, session_id: &str, now_ms: f64) -> Vec<AgentCronJob> {
        let path = self
            .session_artifact_files
            .lock()
            .expect("artifact files poisoned")
            .get(session_id)
            .cloned();
        let Some(path) = path else {
            return Vec::new();
        };
        let mut state = read_jobs_state_lossy(&path);
        let mut recovered: Vec<AgentCronJob> = Vec::new();
        if !state.dispatches.is_empty() {
            recover_interrupted_in_state(&mut state, now_ms, &mut recovered, None);
            if let Err(error) = write_jobs_state(&path, &state) {
            Self::report_cron_write_failure(error);
        }
        }
        recovered
    }

    /// `list()`.
    pub fn list(&self) -> Vec<AgentCronJob> {
        let mut jobs = self.read_jobs();
        jobs.sort_by(|a, b| compare_optional_iso(a.next_run_at.as_deref(), b.next_run_at.as_deref()));
        jobs
    }

    /// `create(input)`.
    pub fn create(&self, input: &CreateAgentCronJobInput) -> Result<AgentCronJob, String> {
        let now = input.now.unwrap_or_else(now_millis);
        let prompt = input.prompt.trim().to_string();
        if prompt.is_empty() {
            return Err("Cron job prompt cannot be empty".to_string());
        }
        let parsed = parse_agent_cron_schedule(&input.schedule_text, now)?;
        let now_iso = iso_string(now);
        let job = AgentCronJob {
            id: random_uuid(),
            status: STATUS_ACTIVE.to_string(),
            source: Some(input.source.clone().unwrap_or_else(|| SOURCE_CRON.to_string())),
            runtime_kind: input.runtime_kind.clone(),
            active_session_id: input.active_session_id.clone(),
            session_id: input.session_id.clone(),
            session_file: input.session_file.clone(),
            cwd: input.cwd.clone(),
            label: normalize_optional_label(input.label.as_deref()),
            prompt,
            schedule: parsed.schedule,
            created_at: now_iso.clone(),
            updated_at: now_iso,
            next_run_at: Some(iso_string(parsed.next_run_at_ms)),
            last_run_at: None,
            last_skipped_at: None,
            last_error: None,
            run_count: 0.0,
            delivery_mode: None,
        };
        let mut jobs = self.read_jobs();
        jobs.push(job.clone());
        self.write_jobs(&jobs)?;
        Ok(job)
    }

    /// `rebindSessionJobs(input)`.
    pub fn rebind_session_jobs(&self, input: &CreateAgentCronJobInput) -> Vec<AgentCronJob> {
        let target_session_file = resolve_path(&input.session_file);
        let mut rebound: Vec<AgentCronJob> = Vec::new();
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.active_session_id != input.active_session_id
                    && resolve_path(&job.session_file) != target_session_file
                {
                    return job;
                }
                if job.active_session_id == input.active_session_id
                    && job.session_id == input.session_id
                    && resolve_path(&job.session_file) == target_session_file
                    && job.cwd == input.cwd
                {
                    return job;
                }
                let mut rebound_job = job.clone();
                rebound_job.active_session_id = input.active_session_id.clone();
                rebound_job.session_id = input.session_id.clone();
                rebound_job.session_file = input.session_file.clone();
                rebound_job.cwd = input.cwd.clone();
                rebound.push(rebound_job.clone());
                rebound_job
            })
            .collect();
        if !rebound.is_empty() {
            if let Err(error) = self.write_jobs(&jobs) {
                Self::report_cron_write_failure(error);
            }
        }
        rebound
    }

    /// `getHeartbeat(activeSessionId)`.
    pub fn get_heartbeat(&self, active_session_id: &str) -> Option<AgentCronJob> {
        let mut jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .filter(|job| {
                job.active_session_id == active_session_id
                    && job.source.as_deref() == Some(SOURCE_HEARTBEAT)
                    && (job.status == STATUS_ACTIVE || job.status == STATUS_PAUSED)
            })
            .collect();
        jobs.sort_by(|a, b| {
            parse_iso_date(&b.updated_at)
                .partial_cmp(&parse_iso_date(&a.updated_at))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        jobs.into_iter().next()
    }

    /// `getLatestHeartbeat(activeSessionId)`.
    pub fn get_latest_heartbeat(&self, active_session_id: &str) -> Option<AgentCronJob> {
        let mut jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .filter(|job| {
                job.active_session_id == active_session_id && job.source.as_deref() == Some(SOURCE_HEARTBEAT)
            })
            .collect();
        jobs.sort_by(|a, b| {
            parse_iso_date(&b.updated_at)
                .partial_cmp(&parse_iso_date(&a.updated_at))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        jobs.into_iter().next()
    }

    /// `createHeartbeat(input)`.
    pub fn create_heartbeat(&self, input: &CreateAgentCronJobInput) -> Result<AgentCronJob, String> {
        let now = input.now.unwrap_or_else(now_millis);
        let parsed = parse_agent_cron_schedule(&input.schedule_text, now)?;
        if parsed.schedule.kind == SCHEDULE_ONCE {
            return Err("Heartbeat schedule must be recurring".to_string());
        }
        let existing: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.active_session_id == input.active_session_id
                    && job.source.as_deref() == Some(SOURCE_HEARTBEAT)
                    && (job.status == STATUS_ACTIVE || job.status == STATUS_PAUSED)
                {
                    let mut cancelled = job.clone();
                    cancelled.status = STATUS_CANCELLED.to_string();
                    cancelled.next_run_at = None;
                    cancelled.updated_at = iso_string(now);
                    cancelled
                } else {
                    job
                }
            })
            .collect();
        let prompt = input.prompt.trim().to_string();
        if prompt.is_empty() {
            return Err("Heartbeat instruction cannot be empty".to_string());
        }
        let now_iso = iso_string(now);
        let job = AgentCronJob {
            id: random_uuid(),
            status: STATUS_ACTIVE.to_string(),
            source: Some(SOURCE_HEARTBEAT.to_string()),
            runtime_kind: input.runtime_kind.clone(),
            delivery_mode: Some(
                input
                    .delivery_mode
                    .clone()
                    .unwrap_or_else(|| DEFAULT_HEARTBEAT_DELIVERY_MODE.to_string()),
            ),
            active_session_id: input.active_session_id.clone(),
            session_id: input.session_id.clone(),
            session_file: input.session_file.clone(),
            cwd: input.cwd.clone(),
            label: normalize_optional_label(input.label.as_deref()),
            prompt,
            schedule: parsed.schedule,
            created_at: now_iso.clone(),
            updated_at: now_iso,
            next_run_at: Some(iso_string(parsed.next_run_at_ms)),
            last_run_at: None,
            last_skipped_at: None,
            last_error: None,
            run_count: 0.0,
        };
        let mut jobs = existing;
        jobs.push(job.clone());
        self.write_jobs(&jobs)?;
        Ok(job)
    }

    /// `listRlmHeartbeats(activeSessionId, options)`.
    pub fn list_rlm_heartbeats(
        &self,
        active_session_id: &str,
        options: Option<RlmHeartbeatListOptions>,
    ) -> Vec<AgentCronJob> {
        let include_inactive = options.and_then(|options| options.include_inactive).unwrap_or(false);
        let mut jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .filter(|job| {
                if job.active_session_id != active_session_id
                    || job.source.as_deref() != Some(SOURCE_RLM_HEARTBEAT)
                {
                    return false;
                }
                if include_inactive {
                    return true;
                }
                job.status == STATUS_ACTIVE || job.status == STATUS_PAUSED
            })
            .collect();
        jobs.sort_by(|a, b| compare_optional_iso(a.next_run_at.as_deref(), b.next_run_at.as_deref()));
        jobs
    }

    /// `createRlmHeartbeat(input)`.
    pub fn create_rlm_heartbeat(&self, input: &CreateAgentCronJobInput) -> Result<AgentCronJob, String> {
        let now = input.now.unwrap_or_else(now_millis);
        let parsed = parse_agent_cron_schedule(&input.schedule_text, now)?;
        if parsed.schedule.kind == SCHEDULE_ONCE {
            return Err("RLM heartbeat schedule must be recurring".to_string());
        }
        let prompt = input.prompt.trim().to_string();
        if prompt.is_empty() {
            return Err("RLM heartbeat instruction cannot be empty".to_string());
        }
        let now_iso = iso_string(now);
        let job = AgentCronJob {
            id: random_uuid(),
            status: STATUS_ACTIVE.to_string(),
            source: Some(SOURCE_RLM_HEARTBEAT.to_string()),
            runtime_kind: input.runtime_kind.clone(),
            delivery_mode: Some(
                input
                    .delivery_mode
                    .clone()
                    .unwrap_or_else(|| DEFAULT_HEARTBEAT_DELIVERY_MODE.to_string()),
            ),
            active_session_id: input.active_session_id.clone(),
            session_id: input.session_id.clone(),
            session_file: input.session_file.clone(),
            cwd: input.cwd.clone(),
            label: normalize_optional_label(input.label.as_deref()),
            prompt,
            schedule: parsed.schedule,
            created_at: now_iso.clone(),
            updated_at: now_iso,
            next_run_at: Some(iso_string(parsed.next_run_at_ms)),
            last_run_at: None,
            last_skipped_at: None,
            last_error: None,
            run_count: 0.0,
        };
        let mut jobs = self.read_jobs();
        jobs.push(job.clone());
        self.write_jobs(&jobs)?;
        Ok(job)
    }

    /// `updateRlmHeartbeat(activeSessionId, id, update)`.
    pub fn update_rlm_heartbeat(
        &self,
        active_session_id: &str,
        id: &str,
        update: &RlmHeartbeatUpdate,
    ) -> Result<Option<AgentCronJob>, String> {
        let now = update.now.unwrap_or_else(now_millis);
        let mut updated: Option<AgentCronJob> = None;
        let mut matched = false;
        let mut failure: Option<String> = None;
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.id != id
                    || job.active_session_id != active_session_id
                    || job.source.as_deref() != Some(SOURCE_RLM_HEARTBEAT)
                {
                    return job;
                }
                matched = true;
                if job.status == STATUS_CANCELLED || job.status == STATUS_COMPLETED {
                    return job;
                }
                let mut next_job = job.clone();
                if let Some(label) = &update.label {
                    next_job.label = normalize_optional_label(Some(label));
                }
                if let Some(delivery_mode) = &update.delivery_mode {
                    next_job.delivery_mode = Some(delivery_mode.clone());
                }
                if let Some(prompt) = &update.prompt {
                    let trimmed = prompt.trim();
                    if trimmed.is_empty() {
                        failure = Some("RLM heartbeat instruction cannot be empty".to_string());
                        return next_job;
                    }
                    next_job.prompt = trimmed.to_string();
                }
                if let Some(schedule_text) = &update.schedule_text {
                    match parse_agent_cron_schedule(schedule_text, now) {
                        Ok(parsed) => {
                            if parsed.schedule.kind == SCHEDULE_ONCE {
                                failure = Some("RLM heartbeat schedule must be recurring".to_string());
                                return next_job;
                            }
                            if next_job.status == STATUS_PAUSED {
                                next_job = without_next_run_at(AgentCronJob {
                                    schedule: parsed.schedule,
                                    ..next_job
                                });
                            } else {
                                next_job.schedule = parsed.schedule;
                                next_job.next_run_at = Some(iso_string(parsed.next_run_at_ms));
                            }
                        }
                        Err(error) => {
                            failure = Some(error);
                            return next_job;
                        }
                    }
                }
                if update.status.as_deref() == Some("pause") {
                    next_job = without_next_run_at(AgentCronJob {
                        status: STATUS_PAUSED.to_string(),
                        ..next_job
                    });
                } else if update.status.as_deref() == Some("resume") {
                    match next_run_at_for_schedule(&next_job.schedule, now) {
                        Ok(Some(next_run_at)) => {
                            next_job.status = STATUS_ACTIVE.to_string();
                            next_job.next_run_at = Some(iso_string(next_run_at));
                        }
                        _ => {
                            failure = Some("RLM heartbeat schedule must be recurring".to_string());
                            return next_job;
                        }
                    }
                }
                let stamped = AgentCronJob {
                    updated_at: iso_string(now),
                    ..next_job
                };
                updated = Some(stamped.clone());
                stamped
            })
            .collect();
        if let Some(failure) = failure {
            return Err(failure);
        }
        if matched && updated.is_some() {
            self.write_jobs(&jobs)?;
        }
        Ok(updated)
    }

    /// `deleteRlmHeartbeat(activeSessionId, id, now = new Date())`.
    pub fn delete_rlm_heartbeat(&self, active_session_id: &str, id: &str, now_ms: f64) -> Option<AgentCronJob> {
        let mut deleted: Option<AgentCronJob> = None;
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.id != id
                    || job.active_session_id != active_session_id
                    || job.source.as_deref() != Some(SOURCE_RLM_HEARTBEAT)
                {
                    return job;
                }
                let cancelled = without_next_run_at(AgentCronJob {
                    status: STATUS_CANCELLED.to_string(),
                    updated_at: iso_string(now_ms),
                    ..job
                });
                deleted = Some(cancelled.clone());
                cancelled
            })
            .collect();
        if deleted.is_some() {
            if let Err(error) = self.write_jobs(&jobs) {
                Self::report_cron_write_failure(error);
            }
        }
        deleted
    }

    /// `cancelRlmHeartbeatsForSession(activeSessionId, now = new Date())`.
    pub fn cancel_rlm_heartbeats_for_session(&self, active_session_id: &str, now_ms: f64) -> Vec<AgentCronJob> {
        let mut cancelled: Vec<AgentCronJob> = Vec::new();
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.active_session_id != active_session_id
                    || job.source.as_deref() != Some(SOURCE_RLM_HEARTBEAT)
                    || (job.status != STATUS_ACTIVE && job.status != STATUS_PAUSED)
                {
                    return job;
                }
                let cancelled_job = without_next_run_at(AgentCronJob {
                    status: STATUS_CANCELLED.to_string(),
                    updated_at: iso_string(now_ms),
                    ..job
                });
                cancelled.push(cancelled_job.clone());
                cancelled_job
            })
            .collect();
        if !cancelled.is_empty() {
            if let Err(error) = self.write_jobs(&jobs) {
                Self::report_cron_write_failure(error);
            }
        }
        cancelled
    }

    /// `cancelJobsForSession(input, now = new Date())`.
    pub fn cancel_jobs_for_session(&self, input: &CancelJobsForSessionInput, now_ms: f64) -> Vec<AgentCronJob> {
        let target_session_file = input.session_file.as_deref().map(resolve_path);
        let mut cancelled: Vec<AgentCronJob> = Vec::new();
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                let matches = input
                    .active_session_id
                    .as_deref()
                    .map(|value| job.active_session_id == value)
                    .unwrap_or(false)
                    || input
                        .session_id
                        .as_deref()
                        .map(|value| job.session_id == value)
                        .unwrap_or(false)
                    || target_session_file
                        .as_deref()
                        .map(|value| resolve_path(&job.session_file) == value)
                        .unwrap_or(false);
                if !matches || (job.status != STATUS_ACTIVE && job.status != STATUS_PAUSED) {
                    return job;
                }
                let cancelled_job = without_next_run_at(AgentCronJob {
                    status: STATUS_CANCELLED.to_string(),
                    updated_at: iso_string(now_ms),
                    ..job
                });
                cancelled.push(cancelled_job.clone());
                cancelled_job
            })
            .collect();
        if !cancelled.is_empty() {
            if let Err(error) = self.write_jobs(&jobs) {
                Self::report_cron_write_failure(error);
            }
        }
        cancelled
    }

    /// Cancel matching active/paused jobs, reporting lock/read/write failures so
    /// supervisor cleanup retains its durable retry record until cancellation commits.
    pub fn try_cancel_jobs_for_session(&self, input: &CancelJobsForSessionInput, now_ms: f64) -> Result<Vec<AgentCronJob>, String> {
        let target = input.session_file.as_deref().map(crate::core::session_lease::canonical_session_path);
        let mut cancelled = Vec::new();
        self.mutate_states(|state| {
            for job in &mut state.jobs {
                let matches = input.active_session_id.as_deref() == Some(job.active_session_id.as_str())
                    || input.session_id.as_deref() == Some(job.session_id.as_str())
                    || target.as_ref().is_some_and(|target| *target == crate::core::session_lease::canonical_session_path(&job.session_file));
                if matches && matches!(job.status.as_str(), STATUS_ACTIVE | STATUS_PAUSED) {
                    job.status = STATUS_CANCELLED.into();
                    job.next_run_at = None;
                    job.updated_at = iso_string(now_ms);
                    cancelled.push(job.clone());
                }
            }
            Vec::new()
        })?;
        Ok(cancelled)
    }

    /// `pauseHeartbeat(activeSessionId, now = new Date())`.
    pub fn pause_heartbeat(&self, active_session_id: &str, now_ms: f64) -> Option<AgentCronJob> {
        let current = self.get_heartbeat(active_session_id)?;
        let mut paused: Option<AgentCronJob> = None;
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.id != current.id {
                    return job;
                }
                let next = AgentCronJob {
                    status: STATUS_PAUSED.to_string(),
                    next_run_at: None,
                    updated_at: iso_string(now_ms),
                    ..job
                };
                paused = Some(next.clone());
                next
            })
            .collect();
        if let Err(error) = self.write_jobs(&jobs) {
            Self::report_cron_write_failure(error);
        }
        paused
    }

    /// `resumeHeartbeat(activeSessionId, now = new Date())`.
    pub fn resume_heartbeat(&self, active_session_id: &str, now_ms: f64) -> Result<Option<AgentCronJob>, String> {
        let Some(current) = self.get_heartbeat(active_session_id) else {
            return Ok(None);
        };
        let Some(next_run_at) = next_run_at_for_schedule(&current.schedule, now_ms)? else {
            return Err("Heartbeat schedule must be recurring".to_string());
        };
        let mut resumed: Option<AgentCronJob> = None;
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.id != current.id {
                    return job;
                }
                let next = AgentCronJob {
                    status: STATUS_ACTIVE.to_string(),
                    next_run_at: Some(iso_string(next_run_at)),
                    updated_at: iso_string(now_ms),
                    ..job
                };
                resumed = Some(next.clone());
                next
            })
            .collect();
        self.write_jobs(&jobs)?;
        Ok(resumed)
    }

    /// `clearHeartbeat(activeSessionId, now = new Date())`.
    pub fn clear_heartbeat(&self, active_session_id: &str, now_ms: f64) -> Option<AgentCronJob> {
        let current = self.get_heartbeat(active_session_id)?;
        let mut cleared: Option<AgentCronJob> = None;
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.id != current.id {
                    return job;
                }
                let next = AgentCronJob {
                    status: STATUS_CANCELLED.to_string(),
                    next_run_at: None,
                    updated_at: iso_string(now_ms),
                    ..job
                };
                cleared = Some(next.clone());
                next
            })
            .collect();
        if let Err(error) = self.write_jobs(&jobs) {
            Self::report_cron_write_failure(error);
        }
        cleared
    }

    /// `manageHeartbeat(activeSessionId, id, action, now = new Date())`.
    pub fn manage_heartbeat(
        &self,
        active_session_id: &str,
        id: &str,
        action: &str,
        now_ms: f64,
    ) -> Result<Option<AgentCronJob>, String> {
        let mut updated: Option<AgentCronJob> = None;
        let mut failure: Option<String> = None;
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.id != id || job.active_session_id != active_session_id || !is_heartbeat_cron_job(&job) {
                    return job;
                }
                if job.status == STATUS_CANCELLED || job.status == STATUS_COMPLETED {
                    return job;
                }
                if action == "pause" {
                    let next = without_next_run_at(AgentCronJob {
                        status: STATUS_PAUSED.to_string(),
                        updated_at: iso_string(now_ms),
                        ..job
                    });
                    updated = Some(next.clone());
                    return next;
                }
                if action == "stop" {
                    let next = without_next_run_at(AgentCronJob {
                        status: STATUS_CANCELLED.to_string(),
                        updated_at: iso_string(now_ms),
                        ..job
                    });
                    updated = Some(next.clone());
                    return next;
                }
                match next_run_at_for_schedule(&job.schedule, now_ms) {
                    Ok(Some(next_run_at)) => {
                        let next = AgentCronJob {
                            status: STATUS_ACTIVE.to_string(),
                            next_run_at: Some(iso_string(next_run_at)),
                            updated_at: iso_string(now_ms),
                            ..job
                        };
                        updated = Some(next.clone());
                        next
                    }
                    _ => {
                        failure = Some("Heartbeat schedule must be recurring".to_string());
                        job
                    }
                }
            })
            .collect();
        if let Some(failure) = failure {
            return Err(failure);
        }
        if updated.is_some() {
            self.write_jobs(&jobs)?;
        }
        Ok(updated)
    }

    /// `cancel(id, now = new Date())`.
    pub fn cancel(&self, id: &str, now_ms: f64) -> Option<AgentCronJob> {
        let mut cancelled: Option<AgentCronJob> = None;
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.id != id || job.status == STATUS_CANCELLED {
                    return job;
                }
                let next = AgentCronJob {
                    status: STATUS_CANCELLED.to_string(),
                    next_run_at: None,
                    updated_at: iso_string(now_ms),
                    ..job
                };
                cancelled = Some(next.clone());
                next
            })
            .collect();
        if cancelled.is_some() {
            if let Err(error) = self.write_jobs(&jobs) {
                Self::report_cron_write_failure(error);
            }
        }
        cancelled
    }

    /// `recordRunResult(id, result)`.
    pub fn record_run_result(&self, id: &str, now_ms: f64, error: Option<String>) -> Option<AgentCronJob> {
        let mut updated: Option<AgentCronJob> = None;
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.id != id {
                    return job;
                }
                if job.status != STATUS_ACTIVE {
                    updated = Some(job.clone());
                    return job;
                }
                let last_error = error.clone();
                let next_run_at = match job.schedule.kind.as_str() {
                    SCHEDULE_CRON => next_run_at_for_schedule(&job.schedule, now_ms + 1.0)
                        .ok()
                        .flatten()
                        .map(iso_string),
                    SCHEDULE_INTERVAL => next_run_at_for_schedule(&job.schedule, now_ms)
                        .ok()
                        .flatten()
                        .map(iso_string),
                    _ => None,
                };
                let next = AgentCronJob {
                    status: if job.schedule.kind == SCHEDULE_ONCE {
                        STATUS_COMPLETED.to_string()
                    } else {
                        STATUS_ACTIVE.to_string()
                    },
                    next_run_at,
                    last_run_at: Some(iso_string(now_ms)),
                    last_error,
                    run_count: job.run_count + 1.0,
                    updated_at: iso_string(now_ms),
                    ..job
                };
                updated = Some(next.clone());
                next
            })
            .collect();
        if updated.is_some() {
            if let Err(error) = self.write_jobs(&jobs) {
                Self::report_cron_write_failure(error);
            }
        }
        updated
    }

    /// `recordSkipResult(id, result)`.
    pub fn record_skip_result(&self, id: &str, now_ms: f64) -> Option<AgentCronJob> {
        let mut updated: Option<AgentCronJob> = None;
        let jobs: Vec<AgentCronJob> = self
            .read_jobs()
            .into_iter()
            .map(|job| {
                if job.id != id {
                    return job;
                }
                if job.status != STATUS_ACTIVE {
                    updated = Some(job.clone());
                    return job;
                }
                let next_run_at = next_run_at_for_schedule(&job.schedule, now_ms)
                    .ok()
                    .flatten()
                    .map(iso_string);
                let next = AgentCronJob {
                    next_run_at,
                    last_skipped_at: Some(iso_string(now_ms)),
                    updated_at: iso_string(now_ms),
                    ..job
                };
                updated = Some(next.clone());
                next
            })
            .collect();
        if updated.is_some() {
            if let Err(error) = self.write_jobs(&jobs) {
                Self::report_cron_write_failure(error);
            }
        }
        updated
    }

    /// `due(now = new Date())`.
    pub fn due(&self, now_ms: f64) -> Vec<AgentCronJob> {
        self.read_jobs()
            .into_iter()
            .filter(|job| is_due_job(job, now_ms))
            .collect()
    }

    /// `claimDue(dueAt = new Date(), claimedAt = dueAt)`.
    pub fn claim_due(&self, due_at_ms: f64, claimed_at_ms: f64) -> Result<Vec<AgentCronDispatch>, String> {
        self.mutate_states(|state| claim_due_in_state(state, due_at_ms, claimed_at_ms))
    }

    /// `getClaimedJob(id)`.
    pub fn get_claimed_job(&self, id: &str) -> Option<AgentCronJob> {
        for state in self.read_states() {
            if !state.dispatches.iter().any(|dispatch| dispatch.job_id == id) {
                continue;
            }
            return state
                .jobs
                .into_iter()
                .find(|job| job.id == id && job.status == STATUS_ACTIVE);
        }
        None
    }

    /// `recordDispatchResult(dispatchId, result)`.
    pub fn record_dispatch_result(
        &self,
        dispatch_id: &str,
        now_ms: f64,
        outcome: &str,
        error: Option<String>,
    ) -> Result<Option<AgentCronJob>, String> {
        let mut updated: Option<AgentCronJob> = None;
        self.mutate_states(|state| {
            let Some(index) = state
                .dispatches
                .iter()
                .position(|candidate| candidate.id == dispatch_id)
            else {
                return Vec::new();
            };
            let job_id = state.dispatches[index].job_id.clone();
            state.dispatches.retain(|candidate| candidate.id != dispatch_id);
            state.jobs = state
                .jobs
                .iter()
                .map(|job| {
                    if job.id != job_id || job.status != STATUS_ACTIVE {
                        return job.clone();
                    }
                    if outcome == RUN_RESULT_SKIPPED && error.is_none() {
                        let next_run_at = next_run_at_for_schedule(&job.schedule, now_ms)
                            .ok()
                            .flatten()
                            .map(iso_string);
                        let next = AgentCronJob {
                            status: if job.schedule.kind == SCHEDULE_ONCE {
                                STATUS_COMPLETED.to_string()
                            } else {
                                job.status.clone()
                            },
                            next_run_at,
                            last_skipped_at: Some(iso_string(now_ms)),
                            updated_at: iso_string(now_ms),
                            ..job.clone()
                        };
                        updated = Some(next.clone());
                        return next;
                    }
                    let next = AgentCronJob {
                        status: if job.schedule.kind == SCHEDULE_ONCE {
                            STATUS_COMPLETED.to_string()
                        } else {
                            job.status.clone()
                        },
                        last_run_at: Some(iso_string(now_ms)),
                        last_error: error.clone(),
                        run_count: job.run_count + 1.0,
                        updated_at: iso_string(now_ms),
                        ..job.clone()
                    };
                    updated = Some(next.clone());
                    next
                })
                .collect();
            Vec::new()
        })?;
        Ok(updated)
    }

    /// `recoverInterruptedDispatches(now = new Date())`.
    pub fn recover_interrupted_dispatches(&self, now_ms: f64) -> Result<Vec<AgentCronJob>, String> {
        let mut recovered: Vec<AgentCronJob> = Vec::new();
        self.mutate_states(|state| {
            recover_interrupted_in_state(state, now_ms, &mut recovered, None);
            Vec::new()
        })?;
        Ok(recovered)
    }

    /// `recoverInterruptedDispatchesById(dispatchIds, now = new Date())`.
    pub fn recover_interrupted_dispatches_by_id(
        &self,
        dispatch_ids: &[String],
        now_ms: f64,
    ) -> Result<Vec<AgentCronJob>, String> {
        let mut recovered: Vec<AgentCronJob> = Vec::new();
        let interrupted: BTreeSet<String> = dispatch_ids.iter().cloned().collect();
        self.mutate_states(|state| {
            recover_interrupted_in_state(state, now_ms, &mut recovered, Some(&interrupted));
            Vec::new()
        })?;
        Ok(recovered)
    }

    /// `getDueJob(id, now = new Date())`.
    pub fn get_due_job(&self, id: &str, now_ms: f64) -> Option<AgentCronJob> {
        self.read_jobs()
            .into_iter()
            .find(|job| job.id == id && is_due_job(job, now_ms))
    }

    /// `nextActiveRunAt()`.
    pub fn next_active_run_at(&self) -> Option<f64> {
        let mut times: Vec<f64> = self
            .read_jobs()
            .into_iter()
            .filter(|job| job.status == STATUS_ACTIVE && job.next_run_at.is_some())
            .map(|job| parse_iso_date(job.next_run_at.as_deref().expect("checked")))
            .filter(|value| value.is_finite())
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        times.into_iter().next()
    }

    fn read_jobs(&self) -> Vec<AgentCronJob> {
        self.read_states()
            .into_iter()
            .flat_map(|state| state.jobs)
            .collect()
    }

    fn read_states(&self) -> Vec<CronJobsState> {
        if self.session_artifact_mode {
            let paths: Vec<String> = self
                .session_artifact_files
                .lock()
                .expect("artifact files poisoned")
                .values()
                .cloned()
                .collect();
            return paths.into_iter().map(|path| read_jobs_state_lossy(&path)).collect();
        }
        vec![read_jobs_state_lossy(&self.require_file_path())]
    }

    fn mutate_states(
        &self,
        mut mutator: impl FnMut(&mut CronJobsState) -> Vec<AgentCronDispatch>,
    ) -> Result<Vec<AgentCronDispatch>, String> {
        let paths: Vec<String> = if self.session_artifact_mode {
            self.session_artifact_files
                .lock()
                .expect("artifact files poisoned")
                .values()
                .cloned()
                .collect()
        } else {
            vec![self.require_file_path()]
        };
        let previous_heartbeats = heartbeat_catalog_signature(&self.read_jobs());
        let mut changed = false;
        let dispatches = with_cron_jobs_state_locks(&paths, || {
            let mut dispatches: Vec<AgentCronDispatch> = Vec::new();
            for path in &paths {
                let mut state = read_jobs_state(path)?;
                let before = serde_json::to_string(&state_to_value(&state)).unwrap_or_default();
                dispatches.extend(mutator(&mut state));
                if serde_json::to_string(&state_to_value(&state)).unwrap_or_default() != before {
                    // `writeFileAtomicSync` throws (cron-jobs.ts:1552-1555).
                    write_jobs_state(path, &state)?;
                    changed = true;
                }
            }
            Ok::<_, String>(dispatches)
        })??;
        if changed && heartbeat_catalog_signature(&self.read_jobs()) != previous_heartbeats {
            self.notify_heartbeat_change();
        }
        Ok(dispatches)
    }

    /// `writeJobsState` throws, so the store write reports failure instead of pretending the
    /// jobs were persisted (cron-jobs.ts:1552-1555). Callers whose signature is not `Result`
    /// (`rebind_session_jobs`, `delete_rlm_heartbeat`, `cancel_rlm_heartbeats_for_session`,
    /// `cancel_jobs_for_session`, `pause_heartbeat`, `clear_heartbeat`, `cancel`,
    /// `record_run_result`, `record_skip_result`, `recover_session_artifact`) report through
    /// `report_cron_write_failure` until their daemon/rpc callers accept a `Result`.
    fn write_jobs(&self, jobs: &[AgentCronJob]) -> Result<(), String> {
        let previous_heartbeats = heartbeat_catalog_signature(&self.read_jobs());
        if self.session_artifact_mode {
            let registered: BTreeSet<String> = self
                .session_artifact_files
                .lock()
                .expect("artifact files poisoned")
                .keys()
                .cloned()
                .collect();
            if let Some(unregistered) = jobs.iter().find(|job| !registered.contains(&job.session_id)) {
                // The TypeScript throws here; a store write cannot return a
                // Result without changing every caller, so the store panics with
                // the identical message. Callers register artifacts first.
                panic!("Cron job {} targets an unregistered session artifact", unregistered.id);
            }
            let pairs: Vec<(String, String)> = self
                .session_artifact_files
                .lock()
                .expect("artifact files poisoned")
                .iter()
                .map(|(session_id, path)| (session_id.clone(), path.clone()))
                .collect();
            let paths: Vec<String> = pairs.iter().map(|(_, path)| path.clone()).collect();
            with_cron_jobs_state_locks(&paths, || {
                let current_by_session: Vec<(String, CronJobsState)> = pairs
                    .iter()
                    .map(|(session_id, path)| (session_id.clone(), read_jobs_state_lossy(path)))
                    .collect();
                let mut merged_by_session: Vec<(String, Vec<AgentCronJob>)> = Vec::new();
                for (session_id, current) in &current_by_session {
                    let retained: Vec<AgentCronJob> = current
                        .jobs
                        .iter()
                        .filter(|job| {
                            jobs.iter()
                                .find(|incoming| incoming.id == job.id)
                                .map(|incoming| incoming.session_id == *session_id)
                                .unwrap_or(true)
                        })
                        .cloned()
                        .collect();
                    let session_jobs: Vec<AgentCronJob> = jobs
                        .iter()
                        .filter(|job| job.session_id == *session_id)
                        .cloned()
                        .collect();
                    merged_by_session.push((session_id.clone(), merge_fresh_jobs(&retained, &session_jobs)));
                }
                let mut session_id_by_job_id: HashMap<String, String> = HashMap::new();
                for (session_id, session_jobs) in &merged_by_session {
                    for job in session_jobs {
                        session_id_by_job_id.insert(job.id.clone(), session_id.clone());
                    }
                }
                let dispatches: Vec<AgentCronDispatchRecord> = current_by_session
                    .iter()
                    .flat_map(|(_, state)| state.dispatches.clone())
                    .collect();
                for (session_id, path) in &pairs {
                    let current = current_by_session
                        .iter()
                        .find(|(candidate, _)| candidate == session_id)
                        .map(|(_, state)| state.clone())
                        .unwrap_or_default();
                    let next_state = CronJobsState {
                        jobs: merged_by_session
                            .iter()
                            .find(|(candidate, _)| candidate == session_id)
                            .map(|(_, session_jobs)| session_jobs.clone())
                            .unwrap_or_default(),
                        dispatches: dispatches
                            .iter()
                            .filter(|dispatch| {
                                session_id_by_job_id.get(&dispatch.job_id) == Some(session_id)
                            })
                            .cloned()
                            .collect(),
                    };
                    let current_text = serde_json::to_string(&state_to_value(&current)).unwrap_or_default();
                    let next_text = serde_json::to_string(&state_to_value(&next_state)).unwrap_or_default();
                    if current_text != next_text {
                        write_jobs_state(path, &next_state)?;
                    }
                }
                Ok::<(), String>(())
            })
            .expect("artifact write locks")?;
            if heartbeat_catalog_signature(&self.read_jobs()) != previous_heartbeats {
                self.notify_heartbeat_change();
            }
            return Ok(());
        }
        let path = self.require_file_path();
        with_cron_jobs_state_locks(std::slice::from_ref(&path), || {
            write_jobs_file(&path, jobs, true)
        })
        .expect("cron jobs write locks")?;
        if heartbeat_catalog_signature(&self.read_jobs()) != previous_heartbeats {
            self.notify_heartbeat_change();
        }
        Ok(())
    }

    /// A failed store write for a caller whose signature cannot carry a `Result` yet.
    fn report_cron_write_failure(error: String) {
        eprintln!("Cron jobs write failed: {error}");
    }

    fn notify_heartbeat_change(&self) {
        let listeners: Vec<Arc<dyn Fn() + Send + Sync>> = self
            .heartbeat_change_listeners
            .lock()
            .expect("listeners poisoned")
            .iter()
            .flatten()
            .cloned()
            .collect();
        for listener in listeners {
            listener();
        }
    }

    fn require_file_path(&self) -> String {
        self.file_path
            .clone()
            .expect("Cron job store does not have a legacy file path")
    }
}

/// `migrateLegacyCronJobsToSessionArtifacts(filePath, options)`.
pub fn migrate_legacy_cron_jobs_to_session_artifacts(
    file_path: &str,
    is_session_owned: Option<Arc<dyn Fn(&AgentCronJob) -> bool + Send + Sync>>,
    now_ms: f64,
) -> Result<f64, String> {
    let mut legacy_state = read_jobs_state(file_path)?;
    recover_interrupted_in_state(&mut legacy_state, now_ms, &mut Vec::new(), None);
    let jobs: Vec<AgentCronJob> = legacy_state
        .jobs
        .into_iter()
        .map(|job| {
            if is_session_owned.is_none()
                || is_session_owned.as_ref().map(|owned| owned(&job)).unwrap_or(false)
                || (job.status != STATUS_ACTIVE && job.status != STATUS_PAUSED)
            {
                return job;
            }
            without_next_run_at(AgentCronJob {
                status: STATUS_CANCELLED.to_string(),
                updated_at: iso_string(now_ms),
                ..job
            })
        })
        .collect();
    if jobs.is_empty() {
        return Ok(0.0);
    }
    let mut jobs_by_artifact: Vec<(String, Vec<AgentCronJob>)> = Vec::new();
    for job in jobs.iter() {
        let artifact_path = join_path(
            &get_session_artifact_path_for_file(&resolve_path(&job.session_file), Some(&job.session_id)),
            SESSION_SCHEDULED_JOBS_FILENAME,
        );
        match jobs_by_artifact.iter_mut().find(|(path, _)| *path == artifact_path) {
            Some((_, grouped)) => grouped.push(job.clone()),
            None => jobs_by_artifact.push((artifact_path, vec![job.clone()])),
        }
    }
    for (artifact_path, artifact_jobs) in &jobs_by_artifact {
        write_jobs_file(artifact_path, artifact_jobs, true)?;
    }
    let migrated = format!("{file_path}.migrated-{}", now_millis() as i64);
    std::fs::rename(file_path, &migrated).map_err(|error| error.to_string())?;
    Ok(jobs.len() as f64)
}

/// `class AgentCronScheduler`.
pub struct AgentCronScheduler {
    store: Arc<AgentCronJobStore>,
    hooks: Arc<AgentCronSchedulerHooks>,
    /// Set by `enable_run_job_errors`; see `AgentCronSchedulerHooks`. Shared so the timer's
    /// rebuilt scheduler keeps the same throwing `runJob`.
    run_job_error: Arc<Mutex<Option<CronRunJob>>>,
    state: Arc<Mutex<AgentCronSchedulerState>>,
}

pub type CronRunJob = Arc<
    dyn Fn(AgentCronJob) -> BoxFuture<Result<Option<AgentCronJobRunResult>, String>> + Send + Sync,
>;

#[derive(Default)]
struct AgentCronSchedulerState {
    running: bool,
    stopped: bool,
    has_started: bool,
    timer: Option<tokio::task::JoinHandle<()>>,
    timer_generation: u64,
    dispatch_lanes: HashMap<String, Arc<tokio::sync::Mutex<()>>>,
}

impl AgentCronScheduler {
    /// `new AgentCronScheduler(store, hooks)`.
    pub fn new(store: Arc<AgentCronJobStore>, hooks: Arc<AgentCronSchedulerHooks>) -> Self {
        Self {
            store,
            hooks,
            run_job_error: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(AgentCronSchedulerState {
                stopped: true,
                ..Default::default()
            })),
        }
    }

    /// Supply the throwing `runJob(job)`: its `Err` is the TS `runError` that
    /// `onError` receives and that `recordDispatchResult` stores as `lastError`
    /// (cron-jobs.ts:1010-1020, 739).
    pub fn enable_run_job_errors(&self, run_job: CronRunJob) {
        *self.run_job_error.lock().expect("run job poisoned") = Some(run_job);
    }

    /// `start()`.
    pub fn start(&self) {
        {
            let mut state = self.state.lock().expect("scheduler poisoned");
            state.stopped = false;
            if !state.has_started {
                state.has_started = true;
            } else {
                drop(state);
                self.schedule_next(None);
                return;
            }
        }
        let _ = self.store.recover_interrupted_dispatches(self.now());
        self.schedule_next(None);
    }

    /// `stop()`.
    pub fn stop(&self) {
        let mut state = self.state.lock().expect("scheduler poisoned");
        state.stopped = true;
        if let Some(timer) = state.timer.take() {
            timer.abort();
        }
    }

    /// `wake()`.
    pub fn wake(&self) {
        if self.state.lock().expect("scheduler poisoned").stopped {
            return;
        }
        self.schedule_next(Some(0.0));
    }

    /// `runDue(now = this.now())`.
    pub async fn run_due(&self, now_ms: Option<f64>) -> Result<usize, String> {
        let now = now_ms.unwrap_or_else(|| self.now());
        {
            let mut state = self.state.lock().expect("scheduler poisoned");
            if state.running || (state.stopped && state.has_started) {
                return Ok(0);
            }
            state.running = true;
        }
        let mut dispatches: Vec<(AgentCronDispatch, Option<Arc<dyn Fn() + Send + Sync>>)> = Vec::new();
        let mut claimed_dispatches: Option<Vec<AgentCronDispatch>> = None;
        let claim = self.store.claim_due(now, self.now());
        match claim {
            Ok(claimed) => {
                for dispatch in claimed {
                    let end_dispatch = self
                        .hooks
                        .begin_dispatch
                        .as_ref()
                        .and_then(|begin| begin(&dispatch));
                    dispatches.push((dispatch, end_dispatch));
                }
            }
            Err(error) => {
                for (_, end_dispatch) in &dispatches {
                    if let Some(end_dispatch) = end_dispatch {
                        end_dispatch();
                    }
                }
                if let Some(claimed) = &claimed_dispatches {
                    let ids: Vec<String> = claimed.iter().map(|dispatch| dispatch.id.clone()).collect();
                    let _ = self.store.recover_interrupted_dispatches_by_id(&ids, self.now());
                }
                let mut state = self.state.lock().expect("scheduler poisoned");
                state.running = false;
                let stopped = state.stopped;
                drop(state);
                if !stopped {
                    self.schedule_next(None);
                }
                return Err(error);
            }
        }
        claimed_dispatches = Some(dispatches.iter().map(|(dispatch, _)| dispatch.clone()).collect());
        {
            let mut state = self.state.lock().expect("scheduler poisoned");
            state.running = false;
            let stopped = state.stopped;
            drop(state);
            if !stopped {
                self.schedule_next(None);
            }
        }
        let mut results: Vec<Option<AgentCronJobRunResult>> = Vec::new();
        for (dispatch, end_dispatch) in dispatches {
            results.push(self.queue_dispatch(dispatch, end_dispatch).await);
        }
        Ok(results.into_iter().filter(|result| result.as_deref() != Some(RUN_RESULT_SKIPPED)).count())
    }

    /// `queueDispatch(dispatch, endDispatch)`.
    async fn queue_dispatch(
        &self,
        dispatch: AgentCronDispatch,
        end_dispatch: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Option<AgentCronJobRunResult> {
        let lane_key = dispatch.job.active_session_id.clone();
        let lane = {
            let mut state = self.state.lock().expect("scheduler poisoned");
            state
                .dispatch_lanes
                .entry(lane_key.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        let _lane_guard = lane.lock().await;
        let result = self.run_dispatch(dispatch, end_dispatch).await;
        let mut state = self.state.lock().expect("scheduler poisoned");
        if state
            .dispatch_lanes
            .get(&lane_key)
            .map(|current| Arc::ptr_eq(current, &lane))
            .unwrap_or(false)
        {
            state.dispatch_lanes.remove(&lane_key);
        }
        drop(state);
        result
    }

    async fn run_dispatch(
        &self,
        dispatch: AgentCronDispatch,
        end_dispatch: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Option<AgentCronJobRunResult> {
        let job = match self.store.get_claimed_job(&dispatch.job.id) {
            Some(job) => job,
            None => {
                let _ = self.store.record_dispatch_result(
                    &dispatch.id,
                    self.now(),
                    RUN_RESULT_SKIPPED,
                    None,
                );
                if let Some(end_dispatch) = end_dispatch {
                    end_dispatch();
                }
                return Some(RUN_RESULT_SKIPPED.to_string());
            }
        };
        // `let runResult; let error; try { runResult = await this.hooks.runJob(job) }
        // catch (runError) { error = runError; this.hooks.onError?.(job, runError) }`
        // (cron-jobs.ts:1008-1015).
        let mut run_result: Option<AgentCronJobRunResult> = None;
        let mut error: Option<String> = None;
        let throwing = self.run_job_error.lock().expect("run job poisoned").clone();
        let outcome = match throwing {
            Some(run_job) => run_job(job.clone()).await,
            None => Ok((self.hooks.run_job)(job.clone()).await),
        };
        match outcome {
            Ok(result) => run_result = result,
            Err(run_error) => {
                if let Some(on_error) = &self.hooks.on_error {
                    on_error(&job, run_error.clone());
                }
                error = Some(run_error);
            }
        }
        // `recordDispatchResult(..., { outcome: runResult === "skipped" && error === undefined
        // ? "skipped" : "ran", error })` (cron-jobs.ts:1016-1020); `error` becomes `lastError`
        // through `recordDispatchResult` (cron-jobs.ts:739).
        let _ = self.store.record_dispatch_result(
            &dispatch.id,
            self.now(),
            if run_result.as_deref() == Some(RUN_RESULT_SKIPPED) && error.is_none() {
                RUN_RESULT_SKIPPED
            } else {
                RUN_RESULT_RAN
            },
            error,
        );
        if let Some(end_dispatch) = end_dispatch {
            end_dispatch();
        }
        run_result
    }

    /// `scheduleNext(delayMs?)`.
    fn schedule_next(&self, delay_ms: Option<f64>) {
        let mut state = self.state.lock().expect("scheduler poisoned");
        if let Some(timer) = state.timer.take() {
            timer.abort();
        }
        let now = self.now();
        state.timer_generation = state.timer_generation.wrapping_add(1);
        let generation = state.timer_generation;
        let next_delay = match delay_ms {
            Some(delay) => Some(delay),
            None => self.store.next_active_run_at().map(|next| (next - now).max(0.0)),
        };
        let Some(next_delay) = next_delay else {
            return;
        };
        let delay = next_delay.min(MAX_TIMEOUT_MS);
        let store = self.store.clone();
        let hooks = self.hooks.clone();
        let run_job_error = Arc::clone(&self.run_job_error);
        let state_handle = self.state_handle();
        let timer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(delay.max(0.0) as u64)).await;
            {
                let mut state = state_handle.lock().expect("scheduler poisoned");
                if state.stopped || state.timer_generation != generation {
                    return;
                }
                // The fired task now owns dispatch work. Rearming, waking or
                // stopping the next timer must not abort it at an admission await.
                state.timer.take();
            }
            let scheduler = AgentCronScheduler {
                store,
                hooks,
                run_job_error,
                state: state_handle,
            };
            let _ = scheduler.run_due(None).await;
        });
        state.timer = Some(timer);
    }

    fn state_handle(&self) -> Arc<Mutex<AgentCronSchedulerState>> {
        self.state.clone()
    }

    fn now(&self) -> f64 {
        self.hooks.now.as_ref().map(|now| now()).unwrap_or_else(now_millis)
    }
}

/// `parseAgentCronSchedule(input, now = new Date())`.
///
/// Returns the schedule and the next run time in milliseconds.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAgentCronSchedule {
    pub schedule: AgentCronSchedule,
    pub next_run_at_ms: f64,
}

pub fn parse_agent_cron_schedule(input: &str, now_ms: f64) -> Result<ParsedAgentCronSchedule, String> {
    let text = strip_matching_quotes(input.trim());
    if text.is_empty() {
        return Err("Cron schedule cannot be empty".to_string());
    }

    if let Some(captures) = in_schedule_regex().captures(&text) {
        let amount: f64 = captures[1].parse().map_err(|_| "Invalid cron amount".to_string())?;
        let unit = captures[2].to_lowercase();
        let multiplier = if unit.starts_with('m') {
            ONE_MINUTE_MS
        } else if unit.starts_with('h') {
            60.0 * ONE_MINUTE_MS
        } else {
            24.0 * 60.0 * ONE_MINUTE_MS
        };
        return Ok(ParsedAgentCronSchedule {
            schedule: AgentCronSchedule {
                kind: SCHEDULE_ONCE.to_string(),
                expression: text,
                interval_ms: None,
            },
            next_run_at_ms: now_ms + amount * multiplier,
        });
    }

    if let Some(captures) = every_schedule_regex().captures(&text) {
        let amount: f64 = captures[1].parse().map_err(|_| "Invalid cron amount".to_string())?;
        let unit = captures[2].to_lowercase();
        let multiplier = if unit.starts_with('s') {
            ONE_SECOND_MS
        } else if unit.starts_with('m') {
            ONE_MINUTE_MS
        } else {
            60.0 * ONE_MINUTE_MS
        };
        let interval_ms = amount * multiplier;
        if interval_ms < 10.0 * ONE_SECOND_MS {
            return Err("Recurring interval must be at least 10 seconds".to_string());
        }
        return Ok(ParsedAgentCronSchedule {
            schedule: AgentCronSchedule {
                kind: SCHEDULE_INTERVAL.to_string(),
                expression: text,
                interval_ms: Some(interval_ms),
            },
            next_run_at_ms: now_ms + interval_ms,
        });
    }

    if text.to_lowercase().starts_with("at ") {
        let when = parse_iso_date(text[3..].trim());
        if !when.is_finite() {
            return Err("Invalid one-shot schedule. Use: at <ISO date>".to_string());
        }
        if when <= now_ms {
            return Err("One-shot schedule must be in the future".to_string());
        }
        return Ok(ParsedAgentCronSchedule {
            schedule: AgentCronSchedule {
                kind: SCHEDULE_ONCE.to_string(),
                expression: text,
                interval_ms: None,
            },
            next_run_at_ms: when,
        });
    }

    let expression = normalize_cron_alias(&text);
    let next_run_at = next_cron_run_after(&expression, now_ms)?;
    Ok(ParsedAgentCronSchedule {
        schedule: AgentCronSchedule {
            kind: SCHEDULE_CRON.to_string(),
            expression,
            interval_ms: None,
        },
        next_run_at_ms: next_run_at,
    })
}

/// `normalizeHeartbeatSchedule(input)`.
pub fn normalize_heartbeat_schedule(input: Option<&str>) -> String {
    let text = input.map(|value| value.trim()).unwrap_or("");
    if text.is_empty() {
        return DEFAULT_HEARTBEAT_SCHEDULE.to_string();
    }
    if heartbeat_schedule_prefix_regex().is_match(text) {
        return format!("every {text}");
    }
    text.to_string()
}

/// `normalizeHeartbeatDeliveryMode(value)`.
pub fn normalize_heartbeat_delivery_mode(value: Option<&Value>) -> Result<Option<AgentHeartbeatDeliveryMode>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    if let Some(text) = value.as_str() {
        if text == DELIVERY_MODE_STEER || text == DELIVERY_MODE_FOLLOW_UP {
            return Ok(Some(text.to_string()));
        }
    }
    Err("Heartbeat delivery mode must be \"steer\" or \"follow_up\"".to_string())
}

/// `resolveHeartbeatStreamingBehavior(deliveryMode)`.
pub fn resolve_heartbeat_streaming_behavior(delivery_mode: Option<&str>) -> String {
    if delivery_mode.unwrap_or(DEFAULT_HEARTBEAT_DELIVERY_MODE) == DELIVERY_MODE_FOLLOW_UP {
        "followUp".to_string()
    } else {
        "steer".to_string()
    }
}

/// `parseHeartbeatCommand(input)`.
pub fn parse_heartbeat_command(input: &str) -> Result<ParsedHeartbeatCommand, String> {
    let text = heartbeat_command_prefix_regex().replace(input, "").trim().to_string();
    if text.is_empty() || text == "status" {
        return Ok(ParsedHeartbeatCommand::Status);
    }
    if text == "pause" {
        return Ok(ParsedHeartbeatCommand::Pause);
    }
    if text == "resume" {
        return Ok(ParsedHeartbeatCommand::Resume);
    }
    if text == "clear" || text == "stop" {
        return Ok(ParsedHeartbeatCommand::Clear);
    }

    let leading_delivery = consume_delivery_option(&text)?;
    let mut delivery_mode = leading_delivery.delivery_mode.clone();
    let mut remaining = leading_delivery.rest.clone();

    if let Some(option) = consume_every_option(&remaining) {
        let trailing_delivery = consume_delivery_option(&option.rest)?;
        delivery_mode = trailing_delivery.delivery_mode.clone().or(delivery_mode);
        if trailing_delivery.rest.is_empty() {
            return Err("Usage: /heartbeat [--every <interval>] [--steer|--follow-up] <instruction>".to_string());
        }
        return Ok(ParsedHeartbeatCommand::Set {
            schedule: normalize_heartbeat_schedule(Some(&option.interval)),
            instruction: trailing_delivery.rest,
            delivery_mode,
        });
    }

    if let Some(leading_schedule) = consume_leading_every_schedule(&remaining) {
        let trailing_delivery = consume_delivery_option(&leading_schedule.rest)?;
        delivery_mode = trailing_delivery.delivery_mode.clone().or(delivery_mode);
        if trailing_delivery.rest.is_empty() {
            return Err("Usage: /heartbeat [--every <interval>] [--steer|--follow-up] <instruction>".to_string());
        }
        return Ok(ParsedHeartbeatCommand::Set {
            schedule: normalize_heartbeat_schedule(Some(&leading_schedule.interval)),
            instruction: trailing_delivery.rest,
            delivery_mode,
        });
    }

    remaining = remaining.trim().to_string();
    if remaining.is_empty() {
        return Err("Usage: /heartbeat [--every <interval>] [--steer|--follow-up] <instruction>".to_string());
    }
    Ok(ParsedHeartbeatCommand::Set {
        schedule: DEFAULT_HEARTBEAT_SCHEDULE.to_string(),
        instruction: remaining,
        delivery_mode,
    })
}

/// `nextRunAtForSchedule(schedule, after)`.
pub fn next_run_at_for_schedule(schedule: &AgentCronSchedule, after_ms: f64) -> Result<Option<f64>, String> {
    if schedule.kind == SCHEDULE_ONCE {
        return Ok(None);
    }
    if schedule.kind == SCHEDULE_INTERVAL {
        match schedule.interval_ms {
            Some(interval_ms) if interval_ms > 0.0 => return Ok(Some(after_ms + interval_ms)),
            _ => {
                return Err(format!("Invalid interval schedule: {}", schedule.expression));
            }
        }
    }
    Ok(Some(next_cron_run_after(&schedule.expression, after_ms)?))
}

/// `formatAgentCronJob(job)`.
///
/// `toLocaleString()` has no Rust equivalent; the port uses the same
/// `en-US`-style local rendering the TUI shows, falling back to the raw value.
pub fn format_agent_cron_job(job: &AgentCronJob) -> String {
    let next = job
        .next_run_at
        .as_deref()
        .map(format_locale_string)
        .unwrap_or_else(|| "-".to_string());
    let last = job
        .last_run_at
        .as_deref()
        .map(format_locale_string)
        .unwrap_or_else(|| "-".to_string());
    let preview: String = {
        let collapsed = whitespace_regex().replace_all(&job.prompt, " ").to_string();
        collapsed.chars().take(80).collect()
    };
    let error = job
        .last_error
        .as_ref()
        .map(|error| format!(" error={error}"))
        .unwrap_or_default();
    let label = job
        .label
        .as_ref()
        .map(|label| format!(" label=\"{label}\""))
        .unwrap_or_default();
    let skipped = job
        .last_skipped_at
        .as_deref()
        .map(|skipped| format!(" skipped={}", format_locale_string(skipped)))
        .unwrap_or_default();
    format!(
        "{} {}{} next={} last={}{} runs={} schedule=\"{}\" prompt=\"{}\"{}",
        job.id,
        job.status,
        label,
        next,
        last,
        skipped,
        format_f64(job.run_count),
        job.schedule.expression,
        preview,
        error
    )
}

/// `isHeartbeatCronJob(job)`.
pub fn is_heartbeat_cron_job(job: &AgentCronJob) -> bool {
    matches!(job.source.as_deref(), Some(SOURCE_HEARTBEAT) | Some(SOURCE_RLM_HEARTBEAT))
}

/// `shouldDeferHeartbeatCronJob(job, activity)`.
pub fn should_defer_heartbeat_cron_job(job: &AgentCronJob, activity: &HeartbeatCronSessionActivity) -> bool {
    if !is_heartbeat_cron_job(job) {
        return false;
    }
    // States where delivering a heartbeat is unsafe or would stack redundant work,
    // regardless of delivery mode.
    let busy_besides_streaming = activity.is_compacting == Some(true)
        || activity.is_retrying == Some(true)
        || activity.is_bash_running
        || activity.has_pending_session_work
        || (!activity.is_streaming && activity.unfinished_action_count > 0.0);
    if busy_besides_streaming {
        return true;
    }
    // "steer" heartbeats interrupt the current turn, so a plain streaming turn must
    // not defer them; "follow_up" heartbeats wait, so streaming still defers.
    if resolve_heartbeat_streaming_behavior(job.delivery_mode.as_deref()) == "steer" {
        return false;
    }
    activity.is_streaming
}

// ---------------------------------------------------------------------------
// Heartbeat option parsing
// ---------------------------------------------------------------------------

struct DeliveryOption {
    delivery_mode: Option<AgentHeartbeatDeliveryMode>,
    rest: String,
}

/// `consumeDeliveryOption(text)`.
fn consume_delivery_option(text: &str) -> Result<DeliveryOption, String> {
    let mut rest = text.trim().to_string();
    if deliver_flag_dangling_regex().is_match(&rest) {
        return Err("Heartbeat delivery mode must be \"steer\" or \"follow_up\"".to_string());
    }
    let mut delivery_mode: Option<AgentHeartbeatDeliveryMode> = None;
    while let Some(leading) = consume_leading_delivery_flag(&rest)? {
        delivery_mode = Some(leading.delivery_mode);
        rest = leading.rest.trim().to_string();
    }

    let mut trailing = consume_trailing_delivery_flag(&rest)?;
    let mut trailing_delivery_mode: Option<AgentHeartbeatDeliveryMode> = None;
    while let Some(found) = trailing {
        // Consume all trailing flags, but keep the rightmost flag's mode because it
        // is textually latest and should win over earlier flags.
        if trailing_delivery_mode.is_none() {
            trailing_delivery_mode = Some(found.delivery_mode.clone());
        }
        rest = found.rest.trim().to_string();
        trailing = consume_trailing_delivery_flag(&rest)?;
    }
    Ok(DeliveryOption {
        delivery_mode: trailing_delivery_mode.or(delivery_mode),
        rest,
    })
}

struct DeliveryFlag {
    delivery_mode: AgentHeartbeatDeliveryMode,
    rest: String,
}

/// `/^--(?:deliver(?:=|\s+)(\S+)|(steer)|(follow[-_]up))(?:\s+|$)([\s\S]*)$/i`.
fn consume_leading_delivery_flag(text: &str) -> Result<Option<DeliveryFlag>, String> {
    let Some(captures) = leading_delivery_regex().captures(text) else {
        return Ok(None);
    };
    let token = captures
        .get(1)
        .or_else(|| captures.get(2))
        .or_else(|| captures.get(3))
        .map(|value| value.as_str().to_string())
        .unwrap_or_default();
    Ok(Some(DeliveryFlag {
        delivery_mode: parse_delivery_mode_token(&token)?,
        rest: captures
            .get(4)
            .map(|value| value.as_str().trim().to_string())
            .unwrap_or_default(),
    }))
}

/// `/^([\s\S]*?)\s+--deliver\s+(\S+)$/` and the `--steer`/`--follow-up` shorthands.
fn consume_trailing_delivery_flag(text: &str) -> Result<Option<DeliveryFlag>, String> {
    if let Some(captures) = trailing_deliver_with_space_regex().captures(text) {
        return Ok(Some(DeliveryFlag {
            delivery_mode: parse_delivery_mode_token(captures.get(2).map(|v| v.as_str()).unwrap_or(""))?,
            rest: captures.get(1).map(|v| v.as_str().to_string()).unwrap_or_default(),
        }));
    }
    if let Some(captures) = trailing_deliver_with_equals_regex().captures(text) {
        return Ok(Some(DeliveryFlag {
            delivery_mode: parse_delivery_mode_token(captures.get(2).map(|v| v.as_str()).unwrap_or(""))?,
            rest: captures.get(1).map(|v| v.as_str().to_string()).unwrap_or_default(),
        }));
    }
    if let Some(captures) = trailing_shorthand_regex().captures(text) {
        return Ok(Some(DeliveryFlag {
            delivery_mode: parse_delivery_mode_token(captures.get(2).map(|v| v.as_str()).unwrap_or(""))?,
            rest: captures.get(1).map(|v| v.as_str().to_string()).unwrap_or_default(),
        }));
    }
    Ok(None)
}

/// `parseDeliveryModeToken(token)`.
fn parse_delivery_mode_token(token: &str) -> Result<AgentHeartbeatDeliveryMode, String> {
    let normalized = token.to_lowercase().replacen('-', "_", 1);
    if normalized == DELIVERY_MODE_STEER || normalized == DELIVERY_MODE_FOLLOW_UP {
        return Ok(normalized);
    }
    Err("Heartbeat delivery mode must be \"steer\" or \"follow_up\"".to_string())
}

struct EveryOption {
    interval: String,
    rest: String,
}

/// `consumeEveryOption(text)`.
fn consume_every_option(text: &str) -> Option<EveryOption> {
    let captures = every_option_regex().captures(text)?;
    Some(EveryOption {
        interval: captures
            .get(1)
            .or_else(|| captures.get(2))
            .or_else(|| captures.get(3))
            .or_else(|| captures.get(4))
            .map(|value| value.as_str().to_string())
            .unwrap_or_default(),
        rest: captures
            .get(5)
            .map(|value| value.as_str().trim().to_string())
            .unwrap_or_default(),
    })
}

struct LeadingEverySchedule {
    interval: String,
    rest: String,
}

/// `consumeLeadingEverySchedule(text)`.
fn consume_leading_every_schedule(text: &str) -> Option<LeadingEverySchedule> {
    let found = leading_every_regex().find(text)?;
    let interval = found.as_str().to_string();
    let rest = text[found.end()..]
        .trim()
        // Strip only a standalone "--" separator, never a flag like "--follow-up".
        .to_string();
    let rest = if let Some(stripped) = rest.strip_prefix("--") {
        if stripped.is_empty() || stripped.starts_with(char::is_whitespace) {
            stripped.trim().to_string()
        } else {
            rest
        }
    } else {
        rest
    };
    Some(LeadingEverySchedule { interval, rest })
}

// ---------------------------------------------------------------------------
// Cron expression evaluation
// ---------------------------------------------------------------------------

/// `interface CronFields`.
struct CronFields {
    minute: BTreeSet<i64>,
    hour: BTreeSet<i64>,
    day_of_month: BTreeSet<i64>,
    month: BTreeSet<i64>,
    day_of_week: BTreeSet<i64>,
}

/// `nextCronRunAfter(expression, after)`.
///
/// The walk happens in LOCAL wall time: `new Date(after.getTime())`, `setSeconds(0, 0)`,
/// `setMinutes(getMinutes() + 1)` and the `getMinutes`/`getHours`/`getDate`/`getMonth`/`getDay`
/// test (cron-jobs.ts:1367-1377, 1447-1457) all read and write the machine's local zone, so
/// `0 9 * * *` fires at 09:00 local time. `millis_from_civil` returns UTC, so the local offset is
/// applied when the wall time is converted back to an instant.
fn next_cron_run_after(expression: &str, after_ms: f64) -> Result<f64, String> {
    let fields = parse_cron_expression(expression)?;
    let mut components = local_civil_from_millis(after_ms);
    components.second = 0;
    components.millisecond = 0;
    components.minute += 1;
    normalize_civil(&mut components);
    let mut candidate = millis_from_local_civil(&components);
    let deadline = candidate + 366.0 * 24.0 * 60.0 * ONE_MINUTE_MS;
    while candidate <= deadline {
        if matches_cron_fields(&components, &fields) {
            return Ok(candidate);
        }
        components.minute += 1;
        normalize_civil(&mut components);
        candidate = millis_from_local_civil(&components);
    }
    Err(format!("Cron schedule did not match within one year: {expression}"))
}

/// `parseCronExpression(expression)`.
fn parse_cron_expression(expression: &str) -> Result<CronFields, String> {
    let parts: Vec<&str> = expression.trim().split_whitespace().collect();
    if parts.len() != 5 {
        return Err(
            "Unsupported cron schedule. Use 'in 10m', 'at <ISO date>', @hourly, or five fields: minute hour day month weekday"
                .to_string(),
        );
    }
    Ok(CronFields {
        minute: parse_cron_field(parts[0], 0, 59)?,
        hour: parse_cron_field(parts[1], 0, 23)?,
        day_of_month: parse_cron_field(parts[2], 1, 31)?,
        month: parse_cron_field(parts[3], 1, 12)?,
        day_of_week: parse_cron_field(parts[4], 0, 7)?,
    })
}

/// `parseCronField(field, min, max)`.
fn parse_cron_field(field: &str, min: i64, max: i64) -> Result<BTreeSet<i64>, String> {
    let mut values: BTreeSet<i64> = BTreeSet::new();
    for part in field.split(',') {
        if part.is_empty() {
            return Err(format!("Invalid cron field: {field}"));
        }
        let mut pieces = part.splitn(2, '/');
        let range_text = pieces.next().unwrap_or("");
        let step_text = pieces.next();
        let step = match step_text {
            None => 1,
            Some(step_text) => parse_cron_number(step_text, 1, max)?,
        };
        let (start, end) = if range_text == "*" {
            (min, max)
        } else if range_text.contains('-') {
            let mut bounds = range_text.splitn(2, '-');
            let start_text = bounds.next().unwrap_or("");
            let end_text = bounds.next().unwrap_or("");
            let start = parse_cron_number(start_text, min, max)?;
            let end = parse_cron_number(end_text, min, max)?;
            if start > end {
                return Err(format!("Invalid cron range: {range_text}"));
            }
            (start, end)
        } else {
            let start = parse_cron_number(range_text, min, max)?;
            (start, start)
        };
        let mut value = start;
        while value <= end {
            values.insert(value);
            value += step;
        }
    }
    Ok(values)
}

/// `parseCronNumber(value, min, max)`.
fn parse_cron_number(value: &str, min: i64, max: i64) -> Result<i64, String> {
    if value.is_empty() || !value.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(format!("Invalid cron number: {value}"));
    }
    let parsed: i64 = value.parse().map_err(|_| format!("Invalid cron number: {value}"))?;
    if parsed < min || parsed > max {
        return Err(format!("Cron number out of range: {value}"));
    }
    Ok(parsed)
}

/// `matchesCronFields(date, fields)`.
fn matches_cron_fields(components: &CivilComponents, fields: &CronFields) -> bool {
    let day = weekday_of(components);
    let day_matches = fields.day_of_week.contains(&day) || (day == 0 && fields.day_of_week.contains(&7));
    fields.minute.contains(&components.minute)
        && fields.hour.contains(&components.hour)
        && fields.day_of_month.contains(&components.day)
        && fields.month.contains(&components.month)
        && day_matches
}

/// `normalizeCronAlias(text)`.
fn normalize_cron_alias(text: &str) -> String {
    match text {
        "@hourly" => "0 * * * *".to_string(),
        "@daily" => "0 0 * * *".to_string(),
        "@weekly" => "0 0 * * 0".to_string(),
        "@monthly" => "0 0 1 * *".to_string(),
        other => other.to_string(),
    }
}

/// `stripMatchingQuotes(value)`.
fn strip_matching_quotes(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() >= 2 {
        let first = chars[0];
        let last = chars[chars.len() - 1];
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return chars[1..chars.len() - 1].iter().collect();
        }
    }
    value.to_string()
}

/// `isDueJob(job, now)`.
fn is_due_job(job: &AgentCronJob, now_ms: f64) -> bool {
    job.status == STATUS_ACTIVE
        && job
            .next_run_at
            .as_deref()
            .map(|next| parse_iso_date(next) <= now_ms)
            .unwrap_or(false)
}

/// `claimDueInState(state, dueAt, claimedAt)`.
fn claim_due_in_state(state: &mut CronJobsState, due_at_ms: f64, claimed_at_ms: f64) -> Vec<AgentCronDispatch> {
    let mut dispatches: Vec<AgentCronDispatch> = Vec::new();
    let claimed_job_ids: BTreeSet<String> = state
        .dispatches
        .iter()
        .map(|dispatch| dispatch.job_id.clone())
        .collect();
    state.jobs = state
        .jobs
        .iter()
        .map(|job| {
            if !is_due_job(job, due_at_ms) {
                return job.clone();
            }
            let scheduled_for = job.next_run_at.clone().unwrap_or_default();
            let next_run_at = next_run_at_for_schedule(&job.schedule, claimed_at_ms)
                .ok()
                .flatten()
                .map(iso_string);
            let advanced = match next_run_at {
                Some(next_run_at) => AgentCronJob {
                    next_run_at: Some(next_run_at),
                    updated_at: iso_string(claimed_at_ms),
                    ..job.clone()
                },
                None => without_next_run_at(AgentCronJob {
                    updated_at: iso_string(claimed_at_ms),
                    ..job.clone()
                }),
            };
            if claimed_job_ids.contains(&job.id) {
                return AgentCronJob {
                    last_skipped_at: Some(iso_string(claimed_at_ms)),
                    ..advanced
                };
            }
            let dispatch = AgentCronDispatchRecord {
                id: random_uuid(),
                job_id: job.id.clone(),
                claimed_at: iso_string(claimed_at_ms),
                scheduled_for,
            };
            state.dispatches.push(dispatch.clone());
            dispatches.push(AgentCronDispatch {
                id: dispatch.id,
                job: advanced.clone(),
            });
            advanced
        })
        .collect();
    dispatches
}

/// `recoverInterruptedInState(state, now, recovered, dispatchIds?)`.
fn recover_interrupted_in_state(
    state: &mut CronJobsState,
    now_ms: f64,
    recovered: &mut Vec<AgentCronJob>,
    dispatch_ids: Option<&BTreeSet<String>>,
) {
    let interrupted: Vec<AgentCronDispatchRecord> = match dispatch_ids {
        Some(ids) => state
            .dispatches
            .iter()
            .filter(|dispatch| ids.contains(&dispatch.id))
            .cloned()
            .collect(),
        None => state.dispatches.clone(),
    };
    if interrupted.is_empty() {
        return;
    }
    let interrupted_ids: BTreeSet<String> = interrupted
        .iter()
        .map(|dispatch| dispatch.job_id.clone())
        .collect();
    state.dispatches = match dispatch_ids {
        Some(ids) => state
            .dispatches
            .iter()
            .filter(|dispatch| !ids.contains(&dispatch.id))
            .cloned()
            .collect(),
        None => Vec::new(),
    };
    state.jobs = state
        .jobs
        .iter()
        .map(|job| {
            if !interrupted_ids.contains(&job.id) || job.status != STATUS_ACTIVE {
                return job.clone();
            }
            let next = AgentCronJob {
                status: if job.schedule.kind == SCHEDULE_ONCE {
                    STATUS_COMPLETED.to_string()
                } else {
                    job.status.clone()
                },
                last_error: Some("Interrupted before scheduled operation completion".to_string()),
                updated_at: iso_string(now_ms),
                ..job.clone()
            };
            recovered.push(next.clone());
            next
        })
        .collect();
}

/// `mergeFreshJobs(currentJobs, nextJobs)`.
fn merge_fresh_jobs(current_jobs: &[AgentCronJob], next_jobs: &[AgentCronJob]) -> Vec<AgentCronJob> {
    let mut merged: Vec<AgentCronJob> = Vec::new();
    let mut index_by_id: HashMap<String, usize> = HashMap::new();
    for job in current_jobs {
        index_by_id.insert(job.id.clone(), merged.len());
        merged.push(job.clone());
    }
    for job in next_jobs {
        match index_by_id.get(&job.id).copied() {
            Some(index) => {
                if is_at_least_as_fresh(job, &merged[index]) {
                    merged[index] = job.clone();
                }
            }
            None => {
                index_by_id.insert(job.id.clone(), merged.len());
                merged.push(job.clone());
            }
        }
    }
    merged
}

/// `isAtLeastAsFresh(candidate, current)`.
fn is_at_least_as_fresh(candidate: &AgentCronJob, current: &AgentCronJob) -> bool {
    let candidate_time = parse_iso_date(&candidate.updated_at);
    let current_time = parse_iso_date(&current.updated_at);
    if !current_time.is_finite() {
        return true;
    }
    if !candidate_time.is_finite() {
        return false;
    }
    candidate_time >= current_time
}

/// `compareOptionalIso(left, right)`.
fn compare_optional_iso(left: Option<&str>, right: Option<&str>) -> std::cmp::Ordering {
    if left == right {
        return std::cmp::Ordering::Equal;
    }
    let (Some(left), Some(right)) = (left, right) else {
        return if left.is_none() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        };
    };
    parse_iso_date(left)
        .partial_cmp(&parse_iso_date(right))
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// `normalizeOptionalLabel(label)`.
fn normalize_optional_label(label: Option<&str>) -> Option<String> {
    let trimmed = label.map(|label| label.trim()).unwrap_or("");
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// `withoutNextRunAt(job)` - drops the key entirely (absent, not null).
fn without_next_run_at(job: AgentCronJob) -> AgentCronJob {
    AgentCronJob {
        next_run_at: None,
        ..job
    }
}

/// `errorMessage(error)`.
pub fn error_message(error: &str) -> String {
    error.to_string()
}

/// `Date.now()` in milliseconds.
pub fn now_millis() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Regexes
// ---------------------------------------------------------------------------

fn in_schedule_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| {
        regex::Regex::new(r"(?i)^in\s+(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$")
            .expect("static pattern")
    })
}

fn every_schedule_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| {
        regex::Regex::new(
            r"(?i)^(?:every|each)\s+(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)$",
        )
        .expect("static pattern")
    })
}

fn heartbeat_schedule_prefix_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| {
        regex::Regex::new(
            r"(?i)^\d+\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)$",
        )
        .expect("static pattern")
    })
}

fn heartbeat_command_prefix_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| regex::Regex::new(r"^/heartbeat\b").expect("static pattern"))
}

fn deliver_flag_dangling_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| regex::Regex::new(r"(?i)(?:^|\s)--deliver=?$").expect("static pattern"))
}

fn leading_delivery_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| {
        regex::Regex::new(r"(?i)^--(?:deliver(?:=|\s+)(\S+)|(steer)|(follow[-_]up))(?:\s+|$)([\s\S]*)$")
            .expect("static pattern")
    })
}

fn trailing_deliver_with_space_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| regex::Regex::new(r"(?i)^([\s\S]*?)\s+--deliver\s+(\S+)$").expect("static pattern"))
}

fn trailing_deliver_with_equals_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| regex::Regex::new(r"(?i)^([\s\S]*?)\s+--deliver=(\S+)$").expect("static pattern"))
}

fn trailing_shorthand_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| regex::Regex::new(r"(?i)^([\s\S]*?)\s+--(steer|follow[-_]up)$").expect("static pattern"))
}

fn every_option_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| {
        regex::Regex::new(
            r#"(?i)^--every(?:=|\s+)(?:"([^"]+)"|'([^']+)'|(\d+\s*(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours))|(\S+))(?:\s+|$)([\s\S]*)$"#,
        )
        .expect("static pattern")
    })
}

fn leading_every_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| {
        regex::Regex::new(
            r"(?i)^(every|each)\s+\d+\s*(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)\b",
        )
        .expect("static pattern")
    })
}

fn whitespace_regex() -> &'static regex::Regex {
    static REGEX: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGEX.get_or_init(|| regex::Regex::new(r"\s+").expect("static pattern"))
}

// ---------------------------------------------------------------------------
// Date helpers (JavaScript Date semantics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct CivilComponents {
    year: i64,
    /// 1-based month, matching `Date.getMonth() + 1`.
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
    millisecond: i64,
}

/// Carries minute/hour/day overflow forward the way `Date.setMinutes` does.
fn normalize_civil(components: &mut CivilComponents) {
    let carry_days = components.minute.div_euclid(60);
    components.minute = components.minute.rem_euclid(60);
    components.hour += carry_days;
    let carry_days = components.hour.div_euclid(24);
    components.hour = components.hour.rem_euclid(24);
    components.day += carry_days;
    loop {
        let days_in_month = days_in_month(components.year, components.month);
        if components.day > days_in_month {
            components.day -= days_in_month;
            components.month += 1;
            if components.month > 12 {
                components.month = 1;
                components.year += 1;
            }
        } else {
            break;
        }
    }
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

/// `date.getDay()` - 0 = Sunday.
fn weekday_of(components: &CivilComponents) -> i64 {
    let (year, month, day) = (components.year, components.month, components.day);
    let adjusted_month = if month < 3 { month + 12 } else { month };
    let adjusted_year = if month < 3 { year - 1 } else { year };
    let k = adjusted_year % 100;
    let j = adjusted_year / 100;
    let h = (day + (13 * (adjusted_month + 1)) / 5 + k + k / 4 + j / 4 + 5 * j) % 7;
    (h + 6) % 7
}

/// `new Date(ms)` broken into local-time-free UTC components.
fn civil_from_millis(millis: f64) -> CivilComponents {
    let total_ms = millis.floor() as i64;
    let total_seconds = total_ms.div_euclid(1000);
    let millisecond = total_ms.rem_euclid(1000);
    let total_minutes = total_seconds.div_euclid(60);
    let second = total_seconds.rem_euclid(60);
    let total_hours = total_minutes.div_euclid(60);
    let minute = total_minutes.rem_euclid(60);
    let days = total_hours.div_euclid(24);
    let hour = total_hours.rem_euclid(24);
    let (year, month, day) = civil_from_days(days);
    CivilComponents {
        year,
        month,
        day,
        hour,
        minute,
        second,
        millisecond,
    }
}

/// The local UTC offset at an instant, in milliseconds (the negation of
/// `Date.getTimezoneOffset()`, which the TypeScript never calls because it reads local getters).
fn local_offset_millis(millis: f64) -> f64 {
    use chrono::TimeZone as _;
    match chrono::Local.timestamp_millis_opt(millis.floor() as i64) {
        chrono::LocalResult::Single(value) | chrono::LocalResult::Ambiguous(value, _) => {
            chrono::Offset::fix(value.offset()).local_minus_utc() as f64 * 1000.0
        }
        chrono::LocalResult::None => 0.0,
    }
}

/// `new Date(ms)` broken into LOCAL components, the space `matchesCronFields` tests.
fn local_civil_from_millis(millis: f64) -> CivilComponents {
    civil_from_millis(millis + local_offset_millis(millis))
}

/// `new Date(localComponents).getTime()`: the local wall time back as an instant. The offset is
/// resolved twice so a candidate that crosses a DST change still lands on the intended wall time.
fn millis_from_local_civil(components: &CivilComponents) -> f64 {
    let as_utc = millis_from_civil(components);
    let once = as_utc - local_offset_millis(as_utc);
    as_utc - local_offset_millis(once)
}

/// `date.toLocaleString()` in local wall time (cron-jobs.ts `toLocaleString()` call sites).
fn local_locale_string(value: &str) -> String {
    match chrono::DateTime::parse_from_rfc3339(value) {
        Ok(parsed) => parsed
            .with_timezone(&chrono::Local)
            .format("%m/%d/%Y, %I:%M:%S %p")
            .to_string(),
        Err(_) => value.to_string(),
    }
}

/// `Date.getTime()` from UTC components.
fn millis_from_civil(components: &CivilComponents) -> f64 {
    let days = days_from_civil(components.year, components.month, components.day);
    ((days * 24 + components.hour) * 60 + components.minute) as f64 * 60_000.0
        + components.second as f64 * 1000.0
        + components.millisecond as f64
}

/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Howard Hinnant's `days_from_civil`.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `Date.parse(iso)` in milliseconds; `NaN` becomes `f64::NAN`.
pub fn parse_iso_date(value: &str) -> f64 {
    match chrono::DateTime::parse_from_rfc3339(value) {
        Ok(parsed) => parsed.timestamp_millis() as f64,
        Err(_) => f64::NAN,
    }
}

/// `date.toISOString()`.
pub fn iso_string(millis: f64) -> String {
    let components = civil_from_millis(millis);
    let weekday = weekday_of(&components);
    let offset = chrono::FixedOffset::east_opt(0).expect("utc offset");
    match chrono::NaiveDate::from_ymd_opt(
        components.year as i32,
        components.month as u32,
        components.day as u32,
    )
    .and_then(|date| {
        date.and_hms_milli_opt(
            components.hour as u32,
            components.minute as u32,
            components.second as u32,
            components.millisecond as u32,
        )
    }) {
        Some(datetime) => {
            let _ = (weekday, offset);
            datetime.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
        }
        None => "Invalid Date".to_string(),
    }
}

/// `date.toLocaleString()`.
///
/// `toLocaleString()` renders the machine's local zone, so the UTC instant parsed from the ISO
/// string is converted before formatting (C2-15).
fn format_locale_string(value: &str) -> String {
    local_locale_string(value)
}

/// `String(number)` for integers, matching the TypeScript template output.
fn format_f64(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e21 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// `path.join(a, b)` for the two-segment case used here.
fn join_path(base: &str, segment: &str) -> String {
    let base = base.trim_end_matches(['/', '\\']);
    if base.is_empty() {
        return segment.to_string();
    }
    let separator = if base.contains('\\') && !base.contains('/') { '\\' } else { '/' };
    format!("{base}{separator}{segment}")
}

/// `path.resolve(path)`.
fn resolve_path(path: &str) -> String {
    let candidate = PathBuf::from(path);
    let absolute = if candidate.is_absolute() {
        candidate
    } else {
        std::env::current_dir().unwrap_or_default().join(candidate)
    };
    absolute
        .components()
        .collect::<PathBuf>()
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job_at(id: &str, next_run_at: Option<&str>) -> AgentCronJob {
        AgentCronJob {
            id: id.to_string(),
            status: STATUS_ACTIVE.to_string(),
            active_session_id: "s1".to_string(),
            session_id: "sess-1".to_string(),
            session_file: "/tmp/sess-1.jsonl".to_string(),
            cwd: "/tmp".to_string(),
            prompt: "keep going".to_string(),
            schedule: AgentCronSchedule {
                kind: SCHEDULE_INTERVAL.to_string(),
                expression: "every 10m".to_string(),
                interval_ms: Some(600_000.0),
            },
            created_at: iso_string(0.0),
            updated_at: iso_string(0.0),
            next_run_at: next_run_at.map(str::to_string),
            run_count: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn schedule_parser_matches_the_typescript_forms() {
        let now = 1_700_000_000_000.0;
        let parsed = parse_agent_cron_schedule("in 10m", now).unwrap();
        assert_eq!(parsed.schedule.kind, SCHEDULE_ONCE);
        assert_eq!(parsed.next_run_at_ms, now + 600_000.0);

        let parsed = parse_agent_cron_schedule("every 5m", now).unwrap();
        assert_eq!(parsed.schedule.kind, SCHEDULE_INTERVAL);
        assert_eq!(parsed.schedule.interval_ms, Some(300_000.0));

        assert_eq!(
            parse_agent_cron_schedule("every 5s", now).unwrap_err(),
            "Recurring interval must be at least 10 seconds"
        );
        assert_eq!(
            parse_agent_cron_schedule("", now).unwrap_err(),
            "Cron schedule cannot be empty"
        );
        assert_eq!(
            parse_agent_cron_schedule("at 2020-01-01T00:00:00Z", now).unwrap_err(),
            "One-shot schedule must be in the future"
        );

        let parsed = parse_agent_cron_schedule("@hourly", now).unwrap();
        assert_eq!(parsed.schedule.expression, "0 * * * *");
        assert_eq!(parsed.schedule.kind, SCHEDULE_CRON);
    }

    #[test]
    fn cron_field_parsing_rejects_out_of_range_and_reversed_ranges() {
        assert_eq!(parse_cron_number("60", 0, 59).unwrap_err(), "Cron number out of range: 60");
        assert_eq!(parse_cron_field("9-1", 0, 59).unwrap_err(), "Invalid cron range: 9-1");
        assert_eq!(parse_cron_field("", 0, 59).unwrap_err(), "Invalid cron field: ");
        let values = parse_cron_field("*/15", 0, 59).unwrap();
        assert_eq!(values.iter().copied().collect::<Vec<i64>>(), vec![0, 15, 30, 45]);
    }

    #[test]
    fn next_cron_run_advances_to_the_next_matching_minute() {
        // 2024-01-01T00:00:00Z
        let start = 1_704_067_200_000.0;
        let next = next_cron_run_after("0 * * * *", start).unwrap();
        assert_eq!(iso_string(next), "2024-01-01T01:00:00.000Z");
    }

    #[test]
    fn heartbeat_command_parsing_covers_control_and_set_forms() {
        assert_eq!(parse_heartbeat_command("/heartbeat").unwrap(), ParsedHeartbeatCommand::Status);
        assert_eq!(parse_heartbeat_command("/heartbeat status").unwrap(), ParsedHeartbeatCommand::Status);
        assert_eq!(parse_heartbeat_command("/heartbeat pause").unwrap(), ParsedHeartbeatCommand::Pause);
        assert_eq!(parse_heartbeat_command("/heartbeat resume").unwrap(), ParsedHeartbeatCommand::Resume);
        assert_eq!(parse_heartbeat_command("/heartbeat stop").unwrap(), ParsedHeartbeatCommand::Clear);

        assert_eq!(
            parse_heartbeat_command("/heartbeat check the disk").unwrap(),
            ParsedHeartbeatCommand::Set {
                schedule: DEFAULT_HEARTBEAT_SCHEDULE.to_string(),
                instruction: "check the disk".to_string(),
                delivery_mode: None,
            }
        );
        assert_eq!(
            parse_heartbeat_command("/heartbeat --every 10m --follow-up check").unwrap(),
            ParsedHeartbeatCommand::Set {
                schedule: "every 10m".to_string(),
                instruction: "check".to_string(),
                delivery_mode: Some(DELIVERY_MODE_FOLLOW_UP.to_string()),
            }
        );
        assert_eq!(
            parse_heartbeat_command("/heartbeat every 5m check").unwrap(),
            ParsedHeartbeatCommand::Set {
                schedule: "every 5m".to_string(),
                instruction: "check".to_string(),
                delivery_mode: None,
            }
        );
        assert_eq!(
            parse_heartbeat_command("/heartbeat --deliver=sideways hi").unwrap_err(),
            "Heartbeat delivery mode must be \"steer\" or \"follow_up\""
        );
    }

    #[test]
    fn heartbeat_schedule_normalization_prefixes_bare_intervals() {
        assert_eq!(normalize_heartbeat_schedule(None), DEFAULT_HEARTBEAT_SCHEDULE);
        assert_eq!(normalize_heartbeat_schedule(Some("  ")), DEFAULT_HEARTBEAT_SCHEDULE);
        assert_eq!(normalize_heartbeat_schedule(Some("15m")), "every 15m");
        assert_eq!(normalize_heartbeat_schedule(Some("every 15m")), "every 15m");
    }

    #[test]
    fn heartbeat_deferral_follows_delivery_mode() {
        let mut job = job_at("h1", None);
        job.source = Some(SOURCE_HEARTBEAT.to_string());
        job.delivery_mode = Some(DELIVERY_MODE_STEER.to_string());
        let streaming = HeartbeatCronSessionActivity {
            is_streaming: true,
            ..Default::default()
        };
        assert!(!should_defer_heartbeat_cron_job(&job, &streaming));

        job.delivery_mode = Some(DELIVERY_MODE_FOLLOW_UP.to_string());
        assert!(should_defer_heartbeat_cron_job(&job, &streaming));

        let busy = HeartbeatCronSessionActivity {
            is_streaming: false,
            unfinished_action_count: 2.0,
            ..Default::default()
        };
        assert!(should_defer_heartbeat_cron_job(&job, &busy));

        let mut cron_job = job_at("c1", None);
        cron_job.source = Some(SOURCE_CRON.to_string());
        assert!(!should_defer_heartbeat_cron_job(&cron_job, &streaming));
    }

    #[test]
    fn store_create_and_claim_advance_the_schedule() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("scheduled-jobs.json").to_string_lossy().to_string();
        let store = AgentCronJobStore::new(Some(path.clone()), false).unwrap();
        let now = 1_700_000_000_000.0;
        let job = store
            .create(&CreateAgentCronJobInput {
                active_session_id: "s1".to_string(),
                session_id: "sess-1".to_string(),
                session_file: "/tmp/sess-1.jsonl".to_string(),
                cwd: "/tmp".to_string(),
                prompt: "  keep going  ".to_string(),
                schedule_text: "every 10m".to_string(),
                now: Some(now),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(job.prompt, "keep going");
        assert_eq!(job.next_run_at.as_deref(), Some(iso_string(now + 600_000.0).as_str()));

        let due = store.due(now + 600_000.0);
        assert_eq!(due.len(), 1);
        let dispatches = store.claim_due(now + 600_000.0, now + 600_000.0).unwrap();
        assert_eq!(dispatches.len(), 1);
        assert_eq!(
            dispatches[0].job.next_run_at.as_deref(),
            Some(iso_string(now + 1_200_000.0).as_str())
        );
        assert!(store.get_claimed_job(&job.id).is_some());
        let updated = store
            .record_dispatch_result(&dispatches[0].id, now + 600_001.0, RUN_RESULT_RAN, None)
            .unwrap()
            .unwrap();
        assert_eq!(updated.run_count, 1.0);
        assert!(store.get_claimed_job(&job.id).is_none());
    }

    #[test]
    fn create_rejects_an_empty_prompt() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("jobs.json").to_string_lossy().to_string();
        let store = AgentCronJobStore::new(Some(path), false).unwrap();
        assert_eq!(
            store
                .create(&CreateAgentCronJobInput {
                    prompt: "   ".to_string(),
                    schedule_text: "every 10m".to_string(),
                    ..Default::default()
                })
                .unwrap_err(),
            "Cron job prompt cannot be empty"
        );
    }

    #[test]
    fn heartbeat_jobs_cancel_the_previous_active_one() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("jobs.json").to_string_lossy().to_string();
        let store = AgentCronJobStore::new(Some(path), false).unwrap();
        let now = 1_700_000_000_000.0;
        let mut input = CreateAgentCronJobInput {
            active_session_id: "s1".to_string(),
            session_id: "sess-1".to_string(),
            session_file: "/tmp/sess-1.jsonl".to_string(),
            cwd: "/tmp".to_string(),
            prompt: "first".to_string(),
            schedule_text: "every 10m".to_string(),
            now: Some(now),
            ..Default::default()
        };
        let first = store.create_heartbeat(&input).unwrap();
        input.prompt = "second".to_string();
        let second = store.create_heartbeat(&input).unwrap();
        let jobs = store.list();
        assert_eq!(jobs.len(), 2);
        assert_eq!(
            jobs.iter().find(|job| job.id == first.id).unwrap().status,
            STATUS_CANCELLED
        );
        assert_eq!(store.get_heartbeat("s1").unwrap().id, second.id);
        assert_eq!(second.delivery_mode.as_deref(), Some(DEFAULT_HEARTBEAT_DELIVERY_MODE));
    }

    #[test]
    fn create_heartbeat_rejects_one_shot_schedules() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("jobs.json").to_string_lossy().to_string();
        let store = AgentCronJobStore::new(Some(path), false).unwrap();
        assert_eq!(
            store
                .create_heartbeat(&CreateAgentCronJobInput {
                    prompt: "x".to_string(),
                    schedule_text: "in 10m".to_string(),
                    ..Default::default()
                })
                .unwrap_err(),
            "Heartbeat schedule must be recurring"
        );
    }

    #[test]
    fn store_requires_a_file_path_outside_session_artifact_mode() {
        assert_eq!(
            AgentCronJobStore::new(None, false).unwrap_err(),
            "Cron job store requires a file path"
        );
    }

    #[test]
    fn session_artifact_registration_is_idempotent() {
        let store = AgentCronJobStore::for_session_artifacts();
        assert!(store.register_session_artifact("sess-1", "/tmp/artifacts"));
        assert!(!store.register_session_artifact("sess-1", "/tmp/artifacts"));
        assert!(!AgentCronJobStore::new(Some("/tmp/x.json".to_string()), false)
            .unwrap()
            .register_session_artifact("sess-1", "/tmp/artifacts"));
    }

    #[test]
    fn merge_fresh_jobs_keeps_the_newer_updated_at() {
        let older = job_at("a", None);
        let mut newer = older.clone();
        newer.updated_at = iso_string(5_000.0);
        newer.prompt = "newer".to_string();
        let merged = merge_fresh_jobs(&[older.clone()], &[newer.clone()]);
        assert_eq!(merged[0].prompt, "newer");
        let merged = merge_fresh_jobs(&[newer], &[older.clone()]);
        assert_eq!(merged[0].prompt, "newer");
        assert!(is_at_least_as_fresh(&older, &job_at("a", None)));
    }

    #[test]
    fn interrupted_dispatches_recover_once() {
        let mut state = CronJobsState::default();
        state.jobs.push(job_at("a", Some(&iso_string(0.0))));
        state.dispatches.push(AgentCronDispatchRecord {
            id: "d1".to_string(),
            job_id: "a".to_string(),
            claimed_at: iso_string(0.0),
            scheduled_for: iso_string(0.0),
        });
        let mut recovered = Vec::new();
        recover_interrupted_in_state(&mut state, 10.0, &mut recovered, None);
        assert_eq!(recovered.len(), 1);
        assert_eq!(state.dispatches.len(), 0);
        assert_eq!(
            state.jobs[0].last_error.as_deref(),
            Some("Interrupted before scheduled operation completion")
        );
    }

    #[test]
    fn format_agent_cron_job_matches_the_typescript_shape() {
        let job = job_at("job-1", Some(&iso_string(0.0)));
        let text = format_agent_cron_job(&job);
        assert!(text.starts_with("job-1 active next="));
        assert!(text.contains("runs=0 schedule=\"every 10m\" prompt=\"keep going\""));
        assert!(text.ends_with("prompt=\"keep going\""));
    }

    #[test]
    fn heartbeat_signature_tracks_heartbeat_fields_only() {
        let mut heartbeat = job_at("h1", None);
        heartbeat.source = Some(SOURCE_HEARTBEAT.to_string());
        heartbeat.status = STATUS_ACTIVE.to_string();
        let signature = heartbeat_catalog_signature(std::slice::from_ref(&heartbeat));
        assert!(signature.contains("\"id\":\"h1\""));

        let mut cron = job_at("c1", None);
        cron.source = Some(SOURCE_CRON.to_string());
        assert_eq!(heartbeat_catalog_signature(std::slice::from_ref(&cron)), "[]");

        let mut cancelled = heartbeat.clone();
        cancelled.status = STATUS_CANCELLED.to_string();
        assert_eq!(heartbeat_catalog_signature(std::slice::from_ref(&cancelled)), "[]");
    }

    #[test]
    fn corrupt_jobs_file_is_preserved_before_the_state_falls_back() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("jobs.json").to_string_lossy().to_string();
        let store = AgentCronJobStore::new(Some(path.clone()), false).unwrap();
        let job = store
            .create(&CreateAgentCronJobInput {
                prompt: "keep going".to_string(),
                schedule_text: "every 10m".to_string(),
                ..Default::default()
            })
            .unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains(&job.id));

        // A corrupt file must be kept, not destroyed by the next write (cron-jobs.ts:1537).
        std::fs::write(&path, "{not json").unwrap();
        let corrupt = format!("{path}.corrupt");
        let listed = store.list();
        assert_eq!(listed.len(), 0);
        assert!(Path::new(&corrupt).exists(), "the unparsable jobs file was not preserved");
        assert_eq!(std::fs::read_to_string(&corrupt).unwrap(), "{not json");
        assert!(read_jobs_state(&path).is_err(), "a corrupt jobs file must not read as valid state");
    }

    #[test]
    fn scheduler_records_a_run_error_and_calls_on_error() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("jobs.json").to_string_lossy().to_string();
        let store = Arc::new(AgentCronJobStore::new(Some(path), false).unwrap());
        let job = store
            .create(&CreateAgentCronJobInput {
                prompt: "keep going".to_string(),
                schedule_text: "every 10m".to_string(),
                now: Some(0.0),
                ..Default::default()
            })
            .unwrap();
        let failures: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&failures);
        let hooks = Arc::new(AgentCronSchedulerHooks {
            run_job: Arc::new(|_job: AgentCronJob| Box::pin(async { Some(RUN_RESULT_RAN.to_string()) })),
            begin_dispatch: None,
            now: None,
            on_error: Some(Arc::new(move |_job: &AgentCronJob, error: String| {
                sink.lock().expect("sink").push(error);
            })),
        });
        let scheduler = AgentCronScheduler::new(Arc::clone(&store), hooks);
        scheduler.enable_run_job_errors(Arc::new(|_job: AgentCronJob| {
            Box::pin(async { Err("cron run exploded".to_string()) })
        }));

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let dispatches = store.claim_due(600_000.0, 600_000.0).unwrap();
        assert_eq!(dispatches.len(), 1);
        let _ = runtime.block_on(scheduler.run_dispatch(dispatches[0].clone(), None));

        // `catch (runError) { error = runError; this.hooks.onError?.(job, runError) }`
        // (cron-jobs.ts:1012-1015) and `lastError: errorMessage(result.error)` (739).
        assert_eq!(failures.lock().expect("sink").as_slice(), ["cron run exploded"]);
        let recorded = store.list().into_iter().find(|candidate| candidate.id == job.id).unwrap();
        assert_eq!(recorded.last_error.as_deref(), Some("cron run exploded"));
    }

    #[test]
    fn cron_matches_the_local_clock_not_utc() {
        // `0 9 * * *` at local 09:00 (cron-jobs.ts:1369-1373 walks local Date getters).
        use chrono::TimeZone as _;
        for month in [3, 7] {
            let start = chrono::Local
                .with_ymd_and_hms(2026, month, 5, 8, 30, 0)
                .single()
                .expect("local 08:30 exists");
            let next = next_cron_run_after("0 9 * * *", start.timestamp_millis() as f64).unwrap();
            let next_local = chrono::Local
                .timestamp_millis_opt(next as i64)
                .single()
                .expect("local timestamp");
            assert_eq!(
                next_local.format("%Y-%m-%d %H:%M").to_string(),
                format!("2026-{month:02}-05 09:00"),
                "the cron schedule fired at the wrong wall-clock time"
            );
        }
    }

    #[test]
    fn write_failures_are_not_silent() {
        // `writeJobsState` throws when the store cannot write (cron-jobs.ts:1552-1555).
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("jobs.json").to_string_lossy().to_string();
        let store = AgentCronJobStore::new(Some(path.clone()), false).unwrap();
        crate::utils::atomic_file::write_file_atomic_sync(
            &path,
            "{}",
            WriteFileAtomicOptions {
                mode: Some(0o600),
                fsync: true,
                fsync_dir: true,
                before_rename: None,
            },
        )
        .unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(store.create(&CreateAgentCronJobInput {
            prompt: "x".to_string(),
            schedule_text: "every 10m".to_string(),
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn next_run_at_for_schedule_rejects_invalid_intervals() {
        let schedule = AgentCronSchedule {
            kind: SCHEDULE_INTERVAL.to_string(),
            expression: "every 0m".to_string(),
            interval_ms: Some(0.0),
        };
        assert_eq!(
            next_run_at_for_schedule(&schedule, 0.0).unwrap_err(),
            "Invalid interval schedule: every 0m"
        );
        assert_eq!(
            next_run_at_for_schedule(&job_at("a", None).schedule, 1000.0).unwrap(),
            Some(601_000.0)
        );
    }
}
