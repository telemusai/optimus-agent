//! Port of packages/coding-agent/src/cli/daemon-update-restart.ts
//!
//! TODO(slice): `proper-lockfile` (dir lock), `getProcessStartId`/session-lease
//! (ca-session), daemon socket helpers (ca-daemon-b), worker-protocol env names
//! (ca-daemon-b), orphan-process journal (ca-misc), child-process helpers
//! (ca-utils) and config helpers (ca-root) are not landed. Private local
//! stand-ins live at the bottom of this module and are listed in the slice
//! status file.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::subprocess_launch::create_cli_subprocess_launch_spec;

pub const DAEMON_UPDATE_RESTART_COORDINATOR_FLAG: &str = "--internal-update-restart-coordinator";
pub const DAEMON_UPDATE_RESTART_STATUS_FLAG: &str = "--internal-update-restart-status";
pub const DAEMON_UPDATE_RESTART_ORIGIN_FLAG: &str = "--internal-update-restart-origin";

// TODO(slice): ca-daemon-b slice, modes/daemon/daemon-worker-protocol.ts.
pub const DAEMON_WORKER_ROLE_ENV: &str = "PRIME_AGENT_INTERNAL_DAEMON_WORKER";
pub const DAEMON_WORKER_TOKEN_ENV: &str = "PRIME_AGENT_INTERNAL_DAEMON_WORKER_TOKEN";
pub const DAEMON_WORKER_ACTIVE_SESSION_ID_ENV: &str = "PRIME_AGENT_INTERNAL_DAEMON_WORKER_ACTIVE_SESSION_ID";
pub const DAEMON_WORKER_RECOVERY_JOURNAL_ENV: &str = "PRIME_AGENT_INTERNAL_DAEMON_WORKER_RECOVERY_JOURNAL";
pub const DAEMON_WORKER_SUPERVISOR_SOCKET_ENV: &str = "PRIME_AGENT_INTERNAL_DAEMON_SUPERVISOR_SOCKET";
// TODO(slice): ca-misc slice, core/orphan-process-journal.ts.
pub const ORPHAN_PROCESS_JOURNAL_ENV: &str = "PRIME_AGENT_INTERNAL_ORPHAN_PROCESS_JOURNAL";
// TODO(slice): ca-session slice, core/session-lease.ts.
pub const SESSION_LEASES_ENABLED_ENV: &str = "PRIME_AGENT_INTERNAL_SESSION_LEASES";
pub const SESSION_LEASE_OWNER_ID_ENV: &str = "PRIME_AGENT_INTERNAL_SESSION_LEASE_OWNER_ID";
// TODO(slice): ca-root slice, config.ts.
pub const ENV_AGENT_DIR: &str = "PRIME_AGENT_CODING_AGENT_DIR";
pub const SELF_UPDATE_INTERACTIVE_CHILD_ENV: &str = "PRIME_AGENT_INTERACTIVE_SELF_UPDATE";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonUpdateRestartPhase {
    Starting,
    Preparing,
    Stopping,
    StartingDaemon,
    Restoring,
    Complete,
    Skipped,
    Failed,
}

impl DaemonUpdateRestartPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            DaemonUpdateRestartPhase::Starting => "starting",
            DaemonUpdateRestartPhase::Preparing => "preparing",
            DaemonUpdateRestartPhase::Stopping => "stopping",
            DaemonUpdateRestartPhase::StartingDaemon => "starting_daemon",
            DaemonUpdateRestartPhase::Restoring => "restoring",
            DaemonUpdateRestartPhase::Complete => "complete",
            DaemonUpdateRestartPhase::Skipped => "skipped",
            DaemonUpdateRestartPhase::Failed => "failed",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "starting" => Some(DaemonUpdateRestartPhase::Starting),
            "preparing" => Some(DaemonUpdateRestartPhase::Preparing),
            "stopping" => Some(DaemonUpdateRestartPhase::Stopping),
            "starting_daemon" => Some(DaemonUpdateRestartPhase::StartingDaemon),
            "restoring" => Some(DaemonUpdateRestartPhase::Restoring),
            "complete" => Some(DaemonUpdateRestartPhase::Complete),
            "skipped" => Some(DaemonUpdateRestartPhase::Skipped),
            "failed" => Some(DaemonUpdateRestartPhase::Failed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DaemonUpdateRestartCounts {
    pub total: i64,
    pub restored: i64,
    pub resumed: i64,
    pub failed: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DaemonUpdateRestartFailure {
    #[serde(rename = "sessionFile")]
    pub session_file: String,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DaemonUpdateRestartProcessIdentity {
    pub pid: i64,
    #[serde(rename = "processStartId", skip_serializing_if = "Option::is_none")]
    pub process_start_id: Option<String>,
    #[serde(rename = "supervisorGeneration", skip_serializing_if = "Option::is_none")]
    pub supervisor_generation: Option<String>,
    #[serde(rename = "supervisorOwnerToken", skip_serializing_if = "Option::is_none")]
    pub supervisor_owner_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DaemonUpdateRestartStatus {
    pub version: i64,
    #[serde(rename = "requestId")]
    pub request_id: String,
    #[serde(rename = "socketPath")]
    pub socket_path: String,
    pub phase: DaemonUpdateRestartPhase,
    pub coordinator: DaemonUpdateRestartProcessIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predecessor: Option<DaemonUpdateRestartProcessIdentity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub successor: Option<DaemonUpdateRestartProcessIdentity>,
    pub counts: DaemonUpdateRestartCounts,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failures: Option<Vec<DaemonUpdateRestartFailure>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(rename = "startedAt")]
    pub started_at: String,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    #[serde(rename = "heartbeatAt", skip_serializing_if = "Option::is_none")]
    pub heartbeat_at: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DaemonUpdateRestartReport {
    pub info: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DaemonUpdateRestartCoordinatorRecord {
    pub version: i64,
    pub token: String,
    #[serde(rename = "requestId")]
    pub request_id: String,
    pub pid: i64,
    #[serde(rename = "processStartId", skip_serializing_if = "Option::is_none")]
    pub process_start_id: Option<String>,
    #[serde(rename = "supervisorGeneration", skip_serializing_if = "Option::is_none")]
    pub supervisor_generation: Option<String>,
    #[serde(rename = "supervisorOwnerToken", skip_serializing_if = "Option::is_none")]
    pub supervisor_owner_token: Option<String>,
    #[serde(rename = "socketPath")]
    pub socket_path: String,
    #[serde(rename = "statusPath")]
    pub status_path: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

pub struct LaunchDaemonUpdateRestartCoordinatorOptions {
    pub socket_path: String,
    pub agent_dir: String,
    pub cwd: Option<String>,
    pub origin_active_session_id: Option<String>,
    pub timeout_ms: Option<f64>,
}

pub struct AcquireDaemonUpdateRestartCoordinatorOptions {
    pub request_id: String,
    pub socket_path: String,
    pub status_path: String,
    pub registry_dir: Option<String>,
}

pub fn resolve_daemon_update_restart_socket_path(socket_path: Option<&str>) -> String {
    match socket_path {
        Some(socket_path) => normalize_socket_path(socket_path, None),
        None => normalize_socket_path(&default_daemon_socket_path(), None),
    }
}

fn terminal_phases() -> [DaemonUpdateRestartPhase; 3] {
    [
        DaemonUpdateRestartPhase::Complete,
        DaemonUpdateRestartPhase::Skipped,
        DaemonUpdateRestartPhase::Failed,
    ]
}

fn all_phases() -> [DaemonUpdateRestartPhase; 8] {
    [
        DaemonUpdateRestartPhase::Starting,
        DaemonUpdateRestartPhase::Preparing,
        DaemonUpdateRestartPhase::Stopping,
        DaemonUpdateRestartPhase::StartingDaemon,
        DaemonUpdateRestartPhase::Restoring,
        DaemonUpdateRestartPhase::Complete,
        DaemonUpdateRestartPhase::Skipped,
        DaemonUpdateRestartPhase::Failed,
    ]
}

pub const DEFAULT_COORDINATOR_PROGRESS_TIMEOUT_MS: f64 = 30.0 * 60_000.0;
pub const COORDINATOR_LIVENESS_TIMEOUT_MS: f64 = 180_000.0;
pub const COORDINATOR_STATUS_HEARTBEAT_MS: u64 = 5000;
pub const COORDINATOR_REGISTRY_LOCK_STALE_MS: u64 = 5000;
pub const COORDINATOR_REGISTRY_LOCK_UPDATE_MS: u64 = 1000;
pub const COORDINATOR_REGISTRY_LOCK_RETRIES: u32 = 500;
pub const COORDINATOR_REGISTRY_LOCK_RETRY_MS: u64 = 10;
pub const PROCESS_START_ID_RECHECK_MS: f64 = 1000.0;

pub fn build_daemon_update_restart_report(status: &DaemonUpdateRestartStatus) -> DaemonUpdateRestartReport {
    let mut report = DaemonUpdateRestartReport::default();
    if status.phase == DaemonUpdateRestartPhase::Failed {
        report.warnings.push(format!(
            "Updated, but could not restart the daemon ({}).",
            status.message.as_deref().unwrap_or("unknown error")
        ));
    }
    if status.phase != DaemonUpdateRestartPhase::Complete && status.phase != DaemonUpdateRestartPhase::Failed {
        return report;
    }
    if status.counts.total > 0 {
        report.info.push(format!(
            "Restored {} daemon session{}",
            status.counts.restored,
            if status.counts.restored == 1 { "" } else { "s" }
        ));
    }
    if status.counts.resumed > 0 {
        report.info.push(format!(
            "Resumed {} interrupted session{}",
            status.counts.resumed,
            if status.counts.resumed == 1 { "" } else { "s" }
        ));
    }
    if status.counts.failed > 0 {
        report.warnings.push(format!(
            "{} daemon session{} could not be restored.",
            status.counts.failed,
            if status.counts.failed == 1 { "" } else { "s" }
        ));
    }
    for failure in status.failures.iter().flatten() {
        report
            .warnings
            .push(format!("Could not restore {}: {}", failure.session_file, failure.message));
    }
    report
}

fn delay(ms: u64) -> tokio::time::Sleep {
    tokio::time::sleep(std::time::Duration::from_millis(ms))
}

fn status_liveness_id(status: &DaemonUpdateRestartStatus) -> String {
    status
        .heartbeat_at
        .clone()
        .unwrap_or_else(|| status.updated_at.clone())
}

fn socket_key(socket_path: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(normalize_socket_path(socket_path, None).as_bytes());
    format!("{:x}", hasher.finalize())
}

fn write_json_atomically(path: &str, value: &serde_json::Value) {
    let temp_path = format!(
        "{}.{}.{}.tmp",
        path,
        std::process::id(),
        uuid::Uuid::new_v4()
    );
    let contents = format!(
        "{}\n",
        serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string())
    );
    match std::fs::write(&temp_path, contents) {
        Ok(()) => {
            if std::fs::rename(&temp_path, path).is_err() {
                let _ = std::fs::remove_file(&temp_path);
            }
        }
        Err(_) => {
            let _ = std::fs::remove_file(&temp_path);
        }
    }
}

fn is_process_identity(value: &serde_json::Value) -> Option<DaemonUpdateRestartProcessIdentity> {
    let object = value.as_object()?;
    let pid = object.get("pid").and_then(serde_json::Value::as_i64)?;
    if pid <= 0 {
        return None;
    }
    let process_start_id = match object.get("processStartId") {
        None => None,
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(_) => return None,
    };
    let supervisor_generation = match object.get("supervisorGeneration") {
        None => None,
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(_) => return None,
    };
    let supervisor_owner_token = match object.get("supervisorOwnerToken") {
        None => None,
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(_) => return None,
    };
    Some(DaemonUpdateRestartProcessIdentity {
        pid,
        process_start_id,
        supervisor_generation,
        supervisor_owner_token,
    })
}

fn is_counts(value: &serde_json::Value) -> Option<DaemonUpdateRestartCounts> {
    let object = value.as_object()?;
    let mut counts = DaemonUpdateRestartCounts::default();
    for (key, slot) in [
        ("total", &mut counts.total),
        ("restored", &mut counts.restored),
        ("resumed", &mut counts.resumed),
        ("failed", &mut counts.failed),
    ] {
        let entry = object.get(key).and_then(serde_json::Value::as_i64)?;
        if entry < 0 {
            return None;
        }
        *slot = entry;
    }
    Some(counts)
}

fn is_failures(value: &serde_json::Value) -> Option<Vec<DaemonUpdateRestartFailure>> {
    let array = value.as_array()?;
    let mut failures: Vec<DaemonUpdateRestartFailure> = Vec::new();
    for entry in array {
        let object = entry.as_object()?;
        let session_file = object.get("sessionFile").and_then(serde_json::Value::as_str)?;
        let message = object.get("message").and_then(serde_json::Value::as_str)?;
        failures.push(DaemonUpdateRestartFailure {
            session_file: session_file.to_string(),
            message: message.to_string(),
        });
    }
    Some(failures)
}

fn is_daemon_update_restart_status(value: &serde_json::Value) -> Option<DaemonUpdateRestartStatus> {
    let object = value.as_object()?;
    if object.get("version").and_then(serde_json::Value::as_i64) != Some(1) {
        return None;
    }
    let request_id = object.get("requestId").and_then(serde_json::Value::as_str)?.to_string();
    let socket_path = object.get("socketPath").and_then(serde_json::Value::as_str)?.to_string();
    let phase = DaemonUpdateRestartPhase::from_str(object.get("phase").and_then(serde_json::Value::as_str)?)?;
    if !all_phases().contains(&phase) {
        return None;
    }
    let coordinator = is_process_identity(object.get("coordinator")?)?;
    let predecessor = match object.get("predecessor") {
        None => None,
        Some(value) => Some(is_process_identity(value)?),
    };
    let successor = match object.get("successor") {
        None => None,
        Some(value) => Some(is_process_identity(value)?),
    };
    let counts = is_counts(object.get("counts")?)?;
    let failures = match object.get("failures") {
        None => None,
        Some(value) => Some(is_failures(value)?),
    };
    let message = match object.get("message") {
        None => None,
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(_) => return None,
    };
    let started_at = object.get("startedAt").and_then(serde_json::Value::as_str)?.to_string();
    let updated_at = object.get("updatedAt").and_then(serde_json::Value::as_str)?.to_string();
    let heartbeat_at = match object.get("heartbeatAt") {
        None => None,
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(_) => return None,
    };
    Some(DaemonUpdateRestartStatus {
        version: 1,
        request_id,
        socket_path,
        phase,
        coordinator,
        predecessor,
        successor,
        counts,
        failures,
        message,
        started_at,
        updated_at,
        heartbeat_at,
    })
}

pub fn read_daemon_update_restart_status(path: &str) -> Result<Option<DaemonUpdateRestartStatus>, String> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    let value: serde_json::Value = match serde_json::from_str(&contents) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    Ok(is_daemon_update_restart_status(&value))
}

fn read_terminal_daemon_update_restart_status(path: &str) -> Option<DaemonUpdateRestartStatus> {
    read_daemon_update_restart_status(path)
        .ok()
        .flatten()
        .filter(|status| terminal_phases().contains(&status.phase))
}

/// Port of the TypeScript `DaemonUpdateRestartStatusWriter` class.
pub struct DaemonUpdateRestartStatusWriter {
    path: String,
    status: std::sync::Mutex<DaemonUpdateRestartStatus>,
}

impl DaemonUpdateRestartStatusWriter {
    pub fn new(path: &str, request_id: &str, socket_path: &str) -> Self {
        let now = now_iso8601();
        let process_start_id = get_process_start_id(std::process::id() as i64);
        let writer = Self {
            path: path.to_string(),
            status: std::sync::Mutex::new(DaemonUpdateRestartStatus {
                version: 1,
                request_id: request_id.to_string(),
                socket_path: socket_path.to_string(),
                phase: DaemonUpdateRestartPhase::Starting,
                coordinator: DaemonUpdateRestartProcessIdentity {
                    pid: std::process::id() as i64,
                    process_start_id,
                    supervisor_generation: None,
                    supervisor_owner_token: None,
                },
                predecessor: None,
                successor: None,
                counts: DaemonUpdateRestartCounts::default(),
                failures: None,
                message: None,
                started_at: now.clone(),
                updated_at: now.clone(),
                heartbeat_at: Some(now),
            }),
        };
        writer.persist();
        writer
    }

    /// `update(update: Partial<...>)`: only the fields listed in `update` change.
    pub fn update(&self, update: DaemonUpdateRestartUpdate) {
        let now = now_iso8601();
        let mut status = self.status.lock().unwrap();
        if let Some(phase) = update.phase {
            status.phase = phase;
        }
        if let Some(predecessor) = update.predecessor {
            status.predecessor = predecessor;
        }
        if let Some(successor) = update.successor {
            status.successor = successor;
        }
        if let Some(counts) = update.counts {
            status.counts = counts;
        }
        if let Some(failures) = update.failures {
            status.failures = failures;
        }
        if let Some(message) = update.message {
            status.message = message;
        }
        status.updated_at = now.clone();
        status.heartbeat_at = Some(now);
        let snapshot = status.clone();
        drop(status);
        self.persist_status(&snapshot);
    }

    pub fn current(&self) -> DaemonUpdateRestartStatus {
        self.status.lock().unwrap().clone()
    }

    pub fn touch(&self) {
        let mut status = self.status.lock().unwrap();
        status.heartbeat_at = Some(now_iso8601());
        let snapshot = status.clone();
        drop(status);
        self.persist_status(&snapshot);
    }

    pub fn start_heartbeat(self: &std::sync::Arc<Self>) -> tokio::task::JoinHandle<()> {
        let writer = self.clone();
        tokio::spawn(async move {
            loop {
                delay(COORDINATOR_STATUS_HEARTBEAT_MS).await;
                // A later phase write will retry; otherwise the parent detects the stale heartbeat.
                writer.touch();
            }
        })
    }

    fn persist(&self) {
        let snapshot = self.current();
        self.persist_status(&snapshot);
    }

    fn persist_status(&self, status: &DaemonUpdateRestartStatus) {
        let value = serde_json::to_value(status).unwrap_or(serde_json::Value::Null);
        write_json_atomically(&self.path, &value);
    }
}

/// `Partial<Omit<DaemonUpdateRestartStatus, "version" | "requestId" | "socketPath" | "coordinator">>`.
#[derive(Default)]
pub struct DaemonUpdateRestartUpdate {
    pub phase: Option<DaemonUpdateRestartPhase>,
    pub predecessor: Option<Option<DaemonUpdateRestartProcessIdentity>>,
    pub successor: Option<Option<DaemonUpdateRestartProcessIdentity>>,
    pub counts: Option<DaemonUpdateRestartCounts>,
    pub failures: Option<Option<Vec<DaemonUpdateRestartFailure>>>,
    pub message: Option<Option<String>>,
}

fn default_coordinator_registry_dir() -> String {
    resolve_path(&Path::new(&default_daemon_socket_dir()).join("update-restart-coordinators"))
}

fn coordinator_record_path(registry_dir: &str, socket_path: &str) -> String {
    resolve_path(&Path::new(registry_dir).join(format!("{}.json", socket_key(socket_path))))
}

/// Local stand-in for `proper-lockfile`'s synchronous lock used by
/// `withCoordinatorRegistryGuard`: a `.guard` lock file with the same stale and
/// retry semantics.
async fn with_coordinator_registry_guard<T>(
    registry_dir: &str,
    action: impl FnOnce() -> T,
) -> Result<T, String> {
    let _ = std::fs::create_dir_all(registry_dir);
    let guard_path = resolve_path(&Path::new(registry_dir).join(".guard"));
    let mut acquired = false;
    for _ in 0..COORDINATOR_REGISTRY_LOCK_RETRIES {
        if try_acquire_lock(&guard_path, COORDINATOR_REGISTRY_LOCK_STALE_MS)? {
            acquired = true;
            break;
        }
        delay(COORDINATOR_REGISTRY_LOCK_RETRY_MS).await;
    }
    if !acquired {
        return Err(format!("Could not coordinate the daemon update restart registry: {}", registry_dir));
    }
    let result = action();
    let _ = std::fs::remove_file(&guard_path);
    Ok(result)
}

fn try_acquire_lock(guard_path: &str, stale_ms: u64) -> Result<bool, String> {
    if let Ok(metadata) = std::fs::metadata(guard_path) {
        if let Ok(modified) = metadata.modified() {
            if let Ok(age) = SystemTime::now().duration_since(modified) {
                if age.as_millis() as u64 > stale_ms {
                    let _ = std::fs::remove_file(guard_path);
                }
            }
        }
    }
    match std::fs::OpenOptions::new().write(true).create_new(true).open(guard_path) {
        Ok(mut file) => {
            use std::io::Write;
            let _ = writeln!(file, "{}", std::process::id());
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error.to_string()),
    }
}

/// daemon-update-restart.ts:357-363:
///   `function matchesProcessStartId(identity) {`
///   `  if (!identity.processStartId) return true;`
///   `  const observed = getProcessStartId(identity.pid);`
///   `  return observed === undefined || observed === identity.processStartId;`
///   `}`
/// A missing start id is permissive. An OBSERVABLE but different start id means
/// a recycled pid, so the identity is stale. Observation failing (undefined)
/// must stay permissive too: the local `get_process_start_id` falls back to `ps`,
/// which does not exist on Windows, so treating "unavailable" as "mismatched"
/// would make every recorded identity look dead on this host.
fn matches_process_start_id(identity: &DaemonUpdateRestartProcessIdentity) -> bool {
    let process_start_id = match &identity.process_start_id {
        Some(process_start_id) => process_start_id,
        None => return true,
    };
    let observed = get_process_start_id(identity.pid);
    observed.is_none() || observed.as_deref() == Some(process_start_id.as_str())
}

fn is_process_identity_alive(identity: &DaemonUpdateRestartProcessIdentity) -> bool {
    is_process_alive(identity.pid) && matches_process_start_id(identity)
}

/// Rate-limited process-start-id recheck, mirroring `createProcessIdentityLivenessCheck`.
pub struct ProcessIdentityLivenessCheck {
    identity: DaemonUpdateRestartProcessIdentity,
    last_start_id_check_at: std::cell::Cell<Option<f64>>,
}

impl ProcessIdentityLivenessCheck {
    pub fn new(identity: DaemonUpdateRestartProcessIdentity) -> Self {
        Self { identity, last_start_id_check_at: std::cell::Cell::new(None) }
    }

    pub fn is_alive(&self) -> bool {
        if !is_process_alive(self.identity.pid) {
            return false;
        }
        if self.identity.process_start_id.is_none() {
            return true;
        }
        let now = now_ms();
        if let Some(last) = self.last_start_id_check_at.get() {
            if now - last < PROCESS_START_ID_RECHECK_MS {
                return true;
            }
        }
        self.last_start_id_check_at.set(Some(now));
        matches_process_start_id(&self.identity)
    }
}

fn read_coordinator_record(path: &str) -> Result<Option<DaemonUpdateRestartCoordinatorRecord>, String> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    let value: serde_json::Value = match serde_json::from_str(&contents) {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let object = match value.as_object() {
        Some(object) => object,
        None => return Ok(None),
    };
    if object.get("version").and_then(serde_json::Value::as_i64) != Some(1) {
        return Ok(None);
    }
    let token = match object.get("token").and_then(serde_json::Value::as_str) {
        Some(token) => token.to_string(),
        None => return Ok(None),
    };
    let request_id = match object.get("requestId").and_then(serde_json::Value::as_str) {
        Some(request_id) => request_id.to_string(),
        None => return Ok(None),
    };
    let socket_path = match object.get("socketPath").and_then(serde_json::Value::as_str) {
        Some(socket_path) => socket_path.to_string(),
        None => return Ok(None),
    };
    let status_path = match object.get("statusPath").and_then(serde_json::Value::as_str) {
        Some(status_path) => status_path.to_string(),
        None => return Ok(None),
    };
    let created_at = match object.get("createdAt").and_then(serde_json::Value::as_str) {
        Some(created_at) => created_at.to_string(),
        None => return Ok(None),
    };
    let identity = match is_process_identity(&value) {
        Some(identity) => identity,
        None => return Ok(None),
    };
    Ok(Some(DaemonUpdateRestartCoordinatorRecord {
        version: 1,
        token,
        request_id,
        pid: identity.pid,
        process_start_id: identity.process_start_id,
        supervisor_generation: identity.supervisor_generation,
        supervisor_owner_token: identity.supervisor_owner_token,
        socket_path,
        status_path,
        created_at,
    }))
}

#[derive(Debug)]
pub struct DaemonUpdateRestartCoordinatorLease {
    pub record: DaemonUpdateRestartCoordinatorRecord,
    registry_dir: String,
    path: String,
    released: std::sync::Mutex<bool>,
}

impl DaemonUpdateRestartCoordinatorLease {
    pub async fn release(&self) -> Result<(), String> {
        {
            let mut released = self.released.lock().unwrap();
            if *released {
                return Ok(());
            }
            *released = true;
        }
        let path = self.path.clone();
        let token = self.record.token.clone();
        with_coordinator_registry_guard(&self.registry_dir, move || {
            if let Ok(Some(current)) = read_coordinator_record(&path) {
                if current.token == token {
                    let _ = std::fs::remove_file(&path);
                }
            }
        })
        .await?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DaemonUpdateRestartCoordinatorAlreadyRunningError {
    pub record: DaemonUpdateRestartCoordinatorRecord,
    pub message: String,
}

impl DaemonUpdateRestartCoordinatorAlreadyRunningError {
    pub fn new(record: DaemonUpdateRestartCoordinatorRecord) -> Self {
        let message = format!(
            "Another daemon update restart is already running for {}",
            record.socket_path
        );
        Self { record, message }
    }
}

pub async fn acquire_daemon_update_restart_coordinator(
    options: AcquireDaemonUpdateRestartCoordinatorOptions,
) -> Result<DaemonUpdateRestartCoordinatorLease, String> {
    let registry_dir = options
        .registry_dir
        .clone()
        .unwrap_or_else(default_coordinator_registry_dir);
    let path = coordinator_record_path(&registry_dir, &options.socket_path);
    let token = uuid::Uuid::new_v4().to_string();
    let process_start_id = get_process_start_id(std::process::id() as i64);
    let record = DaemonUpdateRestartCoordinatorRecord {
        version: 1,
        token,
        request_id: options.request_id.clone(),
        pid: std::process::id() as i64,
        process_start_id,
        supervisor_generation: None,
        supervisor_owner_token: None,
        socket_path: options.socket_path.clone(),
        status_path: options.status_path.clone(),
        created_at: now_iso8601(),
    };

    let path_for_guard = path.clone();
    let record_for_guard = record.clone();
    let already_running = with_coordinator_registry_guard(&registry_dir, move || {
        if let Ok(Some(current)) = read_coordinator_record(&path_for_guard) {
            // `isProcessIdentityAlive(current)` reads the coordinator record's own
            // identity fields, so the identity is projected out of the record first.
            let identity = DaemonUpdateRestartProcessIdentity {
                pid: current.pid,
                process_start_id: current.process_start_id.clone(),
                supervisor_generation: current.supervisor_generation.clone(),
                supervisor_owner_token: current.supervisor_owner_token.clone(),
            };
            if is_process_identity_alive(&identity) {
                return Some(DaemonUpdateRestartCoordinatorAlreadyRunningError::new(current));
            }
        }
        let _ = std::fs::remove_file(&path_for_guard);
        let value = serde_json::to_value(&record_for_guard).unwrap_or(serde_json::Value::Null);
        write_json_atomically(&path_for_guard, &value);
        None
    })
    .await?;

    if let Some(error) = already_running {
        return Err(error.message);
    }
    Ok(DaemonUpdateRestartCoordinatorLease {
        record,
        registry_dir,
        path,
        released: std::sync::Mutex::new(false),
    })
}

pub async fn wait_for_active_daemon_update_restart_coordinator(
    record: &DaemonUpdateRestartCoordinatorRecord,
    progress_timeout_ms: f64,
) -> Result<DaemonUpdateRestartStatus, String> {
    let coordinator_is_alive = ProcessIdentityLivenessCheck::new(DaemonUpdateRestartProcessIdentity {
        pid: record.pid,
        process_start_id: record.process_start_id.clone(),
        supervisor_generation: record.supervisor_generation.clone(),
        supervisor_owner_token: record.supervisor_owner_token.clone(),
    });
    let mut observed_updated_at: Option<String> = None;
    let mut observed_liveness_id: Option<String> = None;
    let mut last_progress_at = now_ms();
    let mut last_liveness_at = now_ms();
    loop {
        let status = read_daemon_update_restart_status(&record.status_path)?;
        if let Some(status) = &status {
            if Some(&status.updated_at) != observed_updated_at.as_ref() {
                observed_updated_at = Some(status.updated_at.clone());
                last_progress_at = now_ms();
            }
            let liveness_id = status_liveness_id(status);
            if Some(&liveness_id) != observed_liveness_id.as_ref() {
                observed_liveness_id = Some(liveness_id);
                last_liveness_at = now_ms();
            }
            if terminal_phases().contains(&status.phase) {
                return Ok(status.clone());
            }
        }
        if !coordinator_is_alive.is_alive() {
            if let Some(terminal_status) = read_terminal_daemon_update_restart_status(&record.status_path) {
                return Ok(terminal_status);
            }
            return Err(format!(
                "Active daemon update restart coordinator exited for {}",
                record.socket_path
            ));
        }
        if now_ms() - last_liveness_at >= COORDINATOR_LIVENESS_TIMEOUT_MS {
            return Err(format!(
                "Active daemon update restart coordinator stopped reporting liveness on {}",
                record.socket_path
            ));
        }
        if now_ms() - last_progress_at >= progress_timeout_ms {
            return Err(format!(
                "Timed out waiting for active daemon update restart progress on {}",
                record.socket_path
            ));
        }
        delay(50).await;
    }
}

fn create_status_path(agent_dir: &str, socket_path: &str, request_id: &str) -> String {
    let directory = resolve_path(&Path::new(agent_dir).join("update-restarts"));
    let _ = std::fs::create_dir_all(&directory);
    resolve_path(
        &Path::new(&directory).join(format!("{}-{}.json", &socket_key(socket_path)[..16], request_id)),
    )
}

fn coordinator_environment(agent_dir: &str) -> super::subprocess_launch::ProcessEnv {
    let mut environment = current_process_env();
    environment.insert(ENV_AGENT_DIR.to_string(), agent_dir.to_string());
    environment.shift_remove(SELF_UPDATE_INTERACTIVE_CHILD_ENV);
    environment.shift_remove(DAEMON_WORKER_ROLE_ENV);
    environment.shift_remove(DAEMON_WORKER_TOKEN_ENV);
    environment.shift_remove(DAEMON_WORKER_ACTIVE_SESSION_ID_ENV);
    environment.shift_remove(DAEMON_WORKER_RECOVERY_JOURNAL_ENV);
    environment.shift_remove(DAEMON_WORKER_SUPERVISOR_SOCKET_ENV);
    environment.shift_remove(ORPHAN_PROCESS_JOURNAL_ENV);
    environment.shift_remove(SESSION_LEASES_ENABLED_ENV);
    environment.shift_remove(SESSION_LEASE_OWNER_ID_ENV);
    environment
}

pub async fn launch_daemon_update_restart_coordinator(
    options: LaunchDaemonUpdateRestartCoordinatorOptions,
) -> Result<DaemonUpdateRestartStatus, String> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let agent_dir = resolve_path(Path::new(&options.agent_dir));
    let socket_path = resolve_daemon_update_restart_socket_path(Some(&options.socket_path));
    let status_path = create_status_path(&agent_dir, &socket_path, &request_id);
    let inherited_origin = std::env::var(DAEMON_WORKER_ACTIVE_SESSION_ID_ENV).ok();
    let origin_active_session_id = options.origin_active_session_id.clone().or(inherited_origin);
    let mut launch_args: Vec<String> = vec![
        "update".to_string(),
        DAEMON_UPDATE_RESTART_COORDINATOR_FLAG.to_string(),
        "--daemon-socket".to_string(),
        socket_path.clone(),
        DAEMON_UPDATE_RESTART_STATUS_FLAG.to_string(),
        status_path.clone(),
    ];
    if let Some(origin_active_session_id) = &origin_active_session_id {
        launch_args.push(DAEMON_UPDATE_RESTART_ORIGIN_FLAG.to_string());
        launch_args.push(origin_active_session_id.clone());
    }
    let launch = create_cli_subprocess_launch_spec(&launch_args, None, &[], None);
    let cwd = options.cwd.clone().unwrap_or_else(current_cwd);
    let env = coordinator_environment(&agent_dir);
    let child = spawn_hidden_detached(&launch.command, &launch.args, &cwd, &env);

    let progress_timeout_ms = options.timeout_ms.unwrap_or(DEFAULT_COORDINATOR_PROGRESS_TIMEOUT_MS);
    let mut observed_updated_at: Option<String> = None;
    let mut observed_liveness_id: Option<String> = None;
    let mut last_progress_at = now_ms();
    let mut last_liveness_at = now_ms();
    loop {
        let status = read_daemon_update_restart_status(&status_path)?;
        if let Some(status) = &status {
            if Some(&status.updated_at) != observed_updated_at.as_ref() {
                observed_updated_at = Some(status.updated_at.clone());
                last_progress_at = now_ms();
            }
            let liveness_id = status_liveness_id(status);
            if Some(&liveness_id) != observed_liveness_id.as_ref() {
                observed_liveness_id = Some(liveness_id);
                last_liveness_at = now_ms();
            }
            if terminal_phases().contains(&status.phase) {
                return Ok(status.clone());
            }
        }
        if let Some(child) = &child {
            if let Some(failure) = child.failure.lock().unwrap().clone() {
                match failure {
                    ChildFailure::Error(message) => return Err(message),
                    ChildFailure::Exit { code, signal } => {
                        if let Some(terminal_status) = read_terminal_daemon_update_restart_status(&status_path) {
                            return Ok(terminal_status);
                        }
                        let description = match signal {
                            Some(signal) => format!("signal {}", signal),
                            None => format!("code {}", code.map(|code| code.to_string()).unwrap_or_else(|| "unknown".to_string())),
                        };
                        return Err(format!(
                            "Daemon update restart coordinator exited with {}",
                            description
                        ));
                    }
                }
            }
        }
        if now_ms() - last_liveness_at >= COORDINATOR_LIVENESS_TIMEOUT_MS {
            return Err(format!(
                "Daemon update restart coordinator stopped reporting liveness on {}",
                socket_path
            ));
        }
        if now_ms() - last_progress_at >= progress_timeout_ms {
            return Err(format!(
                "Timed out waiting for daemon update restart progress on {}",
                socket_path
            ));
        }
        delay(50).await;
    }
}

// ---------------------------------------------------------------------------
// Private local stand-ins for not-yet-landed slices.
// ---------------------------------------------------------------------------

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn current_cwd() -> String {
    std::env::current_dir()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn process_platform() -> &'static str {
    crate::utils::pi_user_agent::process_platform()
}

fn normalize_socket_path(socket_path: &str, base_dir: Option<&str>) -> String {
    crate::utils::daemon_socket_path::normalize_socket_path(socket_path, base_dir)
}

fn default_daemon_socket_dir() -> String {
    crate::modes::daemon::daemon_socket::default_daemon_socket_dir()
}

fn default_daemon_socket_path() -> String {
    std::env::var(DAEMON_WORKER_SUPERVISOR_SOCKET_ENV)
        .ok()
        .filter(|path| !path.trim().is_empty())
        .unwrap_or_else(crate::modes::daemon::daemon_socket::default_daemon_socket_path)
}

fn resolve_path(path: &Path) -> String {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(path)
    };
    normalize_lexically(&joined)
}

fn normalize_lexically(path: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut prefix = String::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Prefix(prefix_component) => prefix.push_str(&prefix_component.as_os_str().to_string_lossy()),
            Component::RootDir => {
                // `Component::Prefix("C:")` followed by `Component::RootDir`
                // is the single drive root "C:\". Pushing the root separator
                // unconditionally keeps that root; the earlier `if prefix
                // .is_empty()` test dropped it and produced "C:Users/...",
                // a drive-relative path, so every caller failed with
                // os error 3 (and "/tmp/registry" became "C:tmp/registry").
                prefix.push('/');
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !parts.is_empty() && parts.last().map(|part| part != "..").unwrap_or(false) {
                    parts.pop();
                } else if !path.is_absolute() {
                    parts.push("..".to_string());
                }
            }
            Component::Normal(part) => parts.push(part.to_string_lossy().to_string()),
        }
    }
    if prefix.is_empty() {
        parts.join("/")
    } else if prefix == "/" {
        format!("/{}", parts.join("/"))
    } else {
        format!("{}{}", prefix, parts.join("/"))
    }
}

fn current_process_env() -> super::subprocess_launch::ProcessEnv {
    let mut environment = super::subprocess_launch::ProcessEnv::new();
    for (key, value) in std::env::vars() {
        environment.insert(key, value);
    }
    environment
}

/// session-lease.ts:162-168 dispatches to `getWindowsProcessStartId` on win32
/// (session-lease.ts:129-145), which queries
/// `([System.Diagnostics.Process]::GetProcessById($pid)).StartTime.ToUniversalTime().Ticks`
/// and formats `win:<ticks>`. Returning `None` here instead made the port report
/// "identity unavailable" for every pid on Windows, so process-identity checks
/// could never distinguish a live process from a recycled pid.
fn get_process_start_id(pid: i64) -> Option<String> {
    if pid <= 0 {
        return None;
    }
    if process_platform() == "win32" {
        return crate::core::session_lease::get_windows_process_start_id(pid, None);
    }
    if let Ok(stat) = std::fs::read_to_string(format!("/proc/{}/stat", pid)) {
        if let Some(command_end) = stat.rfind(')') {
            let fields: Vec<&str> = stat[command_end + 2..].split(' ').collect();
            if let Some(start_time) = fields.get(19) {
                if !start_time.is_empty() {
                    return Some(format!("proc:{}", start_time));
                }
            }
        }
    }
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "lstart="])
        .env("LC_ALL", "C")
        .env("LC_TIME", "C")
        .env("LANG", "C")
        .env("TZ", "UTC")
        .output()
        .ok()?;
    let start_time = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if start_time.is_empty() {
        None
    } else {
        Some(format!("ps:{}", start_time))
    }
}

fn is_process_alive(pid: i64) -> bool {
    kill_process(pid, 0) && !is_zombie_process(pid)
}

fn is_zombie_process(pid: i64) -> bool {
    if process_platform() == "win32" {
        return false;
    }
    if let Ok(stat) = std::fs::read_to_string(format!("/proc/{}/stat", pid)) {
        if let Some(command_end) = stat.rfind(')') {
            return stat[command_end + 2..].trim_start().starts_with('Z');
        }
    }
    std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "stat="])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().starts_with('Z'))
        .unwrap_or(false)
}

#[cfg(unix)]
fn kill_process(pid: i64, signal_number: i32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, signal_number) == 0 }
}

#[cfg(windows)]
fn kill_process(pid: i64, signal_number: i32) -> bool {
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess};
    let handle = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid as u32) };
    if handle.is_null() {
        return false;
    }
    if signal_number == 0 {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(handle) };
        return true;
    }
    let result = unsafe {
        let result = TerminateProcess(handle, 1);
        windows_sys::Win32::Foundation::CloseHandle(handle);
        result
    };
    result != 0
}

#[derive(Debug, Clone)]
enum ChildFailure {
    Error(String),
    Exit { code: Option<i32>, signal: Option<String> },
}

struct SpawnedChild {
    failure: std::sync::Arc<std::sync::Mutex<Option<ChildFailure>>>,
}

/// Local stand-in for `spawnHidden(command, args, { cwd, detached: true, env, stdio: "ignore" })`.
fn spawn_hidden_detached(
    command: &str,
    args: &[String],
    cwd: &str,
    env: &super::subprocess_launch::ProcessEnv,
) -> Option<SpawnedChild> {
    let mut process = std::process::Command::new(command);
    process
        .args(args)
        .current_dir(cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .env_clear();
    for (key, value) in env {
        process.env(key, value);
    }
    let failure: std::sync::Arc<std::sync::Mutex<Option<ChildFailure>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    let failure_slot = failure.clone();
    crate::utils::daemon_process::detach_daemon(&mut process);
    match process.spawn() {
        Ok(mut child) => {
            std::thread::spawn(move || {
                let status = child.wait();
                let mut slot = failure_slot.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(match status {
                        Ok(status) => ChildFailure::Exit {
                            code: status.code(),
                            signal: status_signal(&status),
                        },
                        Err(error) => ChildFailure::Error(error.to_string()),
                    });
                }
            });
            Some(SpawnedChild { failure })
        }
        Err(error) => {
            *failure.lock().unwrap() = Some(ChildFailure::Error(error.to_string()));
            None
        }
    }
}

#[cfg(unix)]
fn status_signal(status: &std::process::ExitStatus) -> Option<String> {
    use std::os::unix::process::ExitStatusExt;
    status.signal().map(|signal| signal_name(signal).to_string())
}

#[cfg(windows)]
fn status_signal(_status: &std::process::ExitStatus) -> Option<String> {
    None
}

fn signal_name(signal: i32) -> &'static str {
    match signal {
        1 => "SIGHUP",
        2 => "SIGINT",
        15 => "SIGTERM",
        9 => "SIGKILL",
        _ => "SIGUNKNOWN",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(phase: DaemonUpdateRestartPhase, counts: DaemonUpdateRestartCounts) -> DaemonUpdateRestartStatus {
        DaemonUpdateRestartStatus {
            version: 1,
            request_id: "req".to_string(),
            socket_path: "/tmp/daemon.sock".to_string(),
            phase,
            coordinator: DaemonUpdateRestartProcessIdentity {
                pid: 1,
                process_start_id: None,
                supervisor_generation: None,
                supervisor_owner_token: None,
            },
            predecessor: None,
            successor: None,
            counts,
            failures: None,
            message: None,
            started_at: "2026-01-01T00:00:00.000Z".to_string(),
            updated_at: "2026-01-01T00:00:00.000Z".to_string(),
            heartbeat_at: None,
        }
    }

    #[test]
    fn phases_round_trip_through_their_wire_names() {
        for phase in all_phases() {
            assert_eq!(DaemonUpdateRestartPhase::from_str(phase.as_str()), Some(phase));
        }
        assert_eq!(DaemonUpdateRestartPhase::from_str("nope"), None);
    }

    #[test]
    fn report_warns_when_the_restart_failed() {
        let mut failed = status(DaemonUpdateRestartPhase::Failed, DaemonUpdateRestartCounts::default());
        failed.message = None;
        let report = build_daemon_update_restart_report(&failed);
        assert_eq!(
            report.warnings,
            vec!["Updated, but could not restart the daemon (unknown error).".to_string()]
        );
        assert!(report.info.is_empty());
    }

    #[test]
    fn report_is_empty_for_non_terminal_phases() {
        let report = build_daemon_update_restart_report(&status(
            DaemonUpdateRestartPhase::Restoring,
            DaemonUpdateRestartCounts { total: 5, restored: 5, resumed: 1, failed: 0 },
        ));
        assert!(report.info.is_empty());
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn report_counts_sessions_and_pluralizes() {
        let complete = status(
            DaemonUpdateRestartPhase::Complete,
            DaemonUpdateRestartCounts { total: 1, restored: 1, resumed: 2, failed: 1 },
        );
        let report = build_daemon_update_restart_report(&complete);
        assert_eq!(report.info, vec!["Restored 1 daemon session", "Resumed 2 interrupted sessions"]);
        assert_eq!(report.warnings, vec!["1 daemon session could not be restored."]);
    }

    #[test]
    fn report_lists_individual_failures() {
        let mut complete = status(
            DaemonUpdateRestartPhase::Complete,
            DaemonUpdateRestartCounts { total: 1, restored: 0, resumed: 0, failed: 0 },
        );
        complete.failures = Some(vec![DaemonUpdateRestartFailure {
            session_file: "/s/1.jsonl".to_string(),
            message: "boom".to_string(),
        }]);
        let report = build_daemon_update_restart_report(&complete);
        assert_eq!(report.warnings, vec!["Could not restore /s/1.jsonl: boom"]);
    }

    #[test]
    fn status_writer_persists_and_updates_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("status.json").to_string_lossy().to_string();
        let writer = DaemonUpdateRestartStatusWriter::new(&path, "req-1", "/tmp/daemon.sock");
        let initial = writer.current();
        assert_eq!(initial.phase, DaemonUpdateRestartPhase::Starting);
        assert_eq!(initial.request_id, "req-1");
        assert_eq!(initial.version, 1);
        assert!(initial.heartbeat_at.is_some());

        writer.update(DaemonUpdateRestartUpdate {
            phase: Some(DaemonUpdateRestartPhase::Restoring),
            counts: Some(DaemonUpdateRestartCounts { total: 2, restored: 1, resumed: 0, failed: 1 }),
            ..Default::default()
        });
        let updated = read_daemon_update_restart_status(&path).unwrap().unwrap();
        assert_eq!(updated.phase, DaemonUpdateRestartPhase::Restoring);
        assert_eq!(updated.counts.total, 2);
        assert_eq!(updated.coordinator.pid, std::process::id() as i64);
        assert_eq!(updated.request_id, "req-1");
        assert_eq!(updated.socket_path, "/tmp/daemon.sock");
    }

    #[test]
    fn status_validation_rejects_malformed_documents() {
        assert!(read_daemon_update_restart_status("/definitely/not/here.json").unwrap().is_none());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "{\"version\": 2}").unwrap();
        assert!(read_daemon_update_restart_status(path.to_str().unwrap()).unwrap().is_none());
        std::fs::write(&path, "not json").unwrap();
        assert!(read_daemon_update_restart_status(path.to_str().unwrap()).unwrap().is_none());
    }

    #[test]
    fn socket_keys_and_registry_paths_are_stable() {
        let first = socket_key("/tmp/daemon.sock");
        let second = socket_key("/tmp/daemon.sock");
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        assert_ne!(first, socket_key("/tmp/other.sock"));
        let record_path = coordinator_record_path("/tmp/registry", "/tmp/daemon.sock");
        assert!(record_path.ends_with(&format!("{}.json", first)));
    }

    #[test]
    fn process_identity_liveness_tracks_the_current_process() {
        let pid = std::process::id() as i64;
        let check = ProcessIdentityLivenessCheck::new(DaemonUpdateRestartProcessIdentity {
            pid,
            process_start_id: None,
            supervisor_generation: None,
            supervisor_owner_token: None,
        });
        assert!(check.is_alive());
        let dead = ProcessIdentityLivenessCheck::new(DaemonUpdateRestartProcessIdentity {
            pid: 2_000_000_000,
            process_start_id: None,
            supervisor_generation: None,
            supervisor_owner_token: None,
        });
        assert!(!dead.is_alive());
    }

    #[test]
    fn start_id_matching_is_permissive_when_unavailable() {
        let pid = std::process::id() as i64;
        assert!(matches_process_start_id(&DaemonUpdateRestartProcessIdentity {
            pid,
            process_start_id: None,
            supervisor_generation: None,
            supervisor_owner_token: None,
        }));
        assert!(!matches_process_start_id(&DaemonUpdateRestartProcessIdentity {
            pid,
            process_start_id: Some("other".to_string()),
            supervisor_generation: None,
            supervisor_owner_token: None,
        }));
    }

    #[test]
    fn resolve_socket_path_normalizes_the_default() {
        let resolved = resolve_daemon_update_restart_socket_path(None);
        assert_eq!(resolved, normalize_socket_path(&default_daemon_socket_path(), None));
    }

    #[tokio::test]
    async fn coordinator_acquisition_blocks_a_second_live_coordinator() {
        let dir = tempfile::tempdir().unwrap();
        let registry_dir = dir.path().join("registry").to_string_lossy().to_string();
        let status_path = dir.path().join("status.json").to_string_lossy().to_string();
        let first = acquire_daemon_update_restart_coordinator(AcquireDaemonUpdateRestartCoordinatorOptions {
            request_id: "one".to_string(),
            socket_path: "/tmp/daemon.sock".to_string(),
            status_path: status_path.clone(),
            registry_dir: Some(registry_dir.clone()),
        })
        .await
        .unwrap();
        let second = acquire_daemon_update_restart_coordinator(AcquireDaemonUpdateRestartCoordinatorOptions {
            request_id: "two".to_string(),
            socket_path: "/tmp/daemon.sock".to_string(),
            status_path: status_path.clone(),
            registry_dir: Some(registry_dir.clone()),
        })
        .await;
        assert!(second.is_err());
        assert!(second.unwrap_err().contains("Another daemon update restart is already running"));

        first.release().await.unwrap();
        let third = acquire_daemon_update_restart_coordinator(AcquireDaemonUpdateRestartCoordinatorOptions {
            request_id: "three".to_string(),
            socket_path: "/tmp/daemon.sock".to_string(),
            status_path,
            registry_dir: Some(registry_dir),
        })
        .await;
        assert!(third.is_ok());
    }

    #[tokio::test]
    async fn waiting_for_a_coordinator_returns_a_terminal_status() {
        let dir = tempfile::tempdir().unwrap();
        let status_path = dir.path().join("status.json").to_string_lossy().to_string();
        let writer = DaemonUpdateRestartStatusWriter::new(&status_path, "req", "/tmp/daemon.sock");
        writer.update(DaemonUpdateRestartUpdate {
            phase: Some(DaemonUpdateRestartPhase::Complete),
            ..Default::default()
        });
        let record = DaemonUpdateRestartCoordinatorRecord {
            version: 1,
            token: "t".to_string(),
            request_id: "req".to_string(),
            pid: std::process::id() as i64,
            process_start_id: None,
            supervisor_generation: None,
            supervisor_owner_token: None,
            socket_path: "/tmp/daemon.sock".to_string(),
            status_path,
            created_at: "2026-01-01T00:00:00.000Z".to_string(),
        };
        let status = wait_for_active_daemon_update_restart_coordinator(&record, 1000.0).await.unwrap();
        assert_eq!(status.phase, DaemonUpdateRestartPhase::Complete);
    }

    #[tokio::test]
    async fn waiting_fails_when_the_coordinator_is_gone() {
        let dir = tempfile::tempdir().unwrap();
        let status_path = dir.path().join("status.json").to_string_lossy().to_string();
        let record = DaemonUpdateRestartCoordinatorRecord {
            version: 1,
            token: "t".to_string(),
            request_id: "req".to_string(),
            pid: 2_000_000_000,
            process_start_id: None,
            supervisor_generation: None,
            supervisor_owner_token: None,
            socket_path: "/tmp/daemon.sock".to_string(),
            status_path,
            created_at: "2026-01-01T00:00:00.000Z".to_string(),
        };
        let error = wait_for_active_daemon_update_restart_coordinator(&record, 1000.0).await.unwrap_err();
        assert_eq!(error, "Active daemon update restart coordinator exited for /tmp/daemon.sock");
    }

    #[test]
    fn coordinator_environment_strips_worker_variables() {
        std::env::set_var(DAEMON_WORKER_ROLE_ENV, "1");
        let environment = coordinator_environment("/tmp/agent");
        std::env::remove_var(DAEMON_WORKER_ROLE_ENV);
        assert_eq!(environment.get(ENV_AGENT_DIR).map(String::as_str), Some("/tmp/agent"));
        assert!(!environment.contains_key(DAEMON_WORKER_ROLE_ENV));
        assert!(!environment.contains_key(SESSION_LEASES_ENABLED_ENV));
    }
}

#[cfg(all(test, unix))]
#[test]
fn detached_daemon_survives_launcher_exit() {
    crate::utils::daemon_process::tests::assert_detached_spawn(|command, args, cwd, env| {
        assert!(spawn_hidden_detached(command, args, cwd, env).is_some());
    });
}
