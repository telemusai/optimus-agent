//! Port of packages/coding-agent/src/cli/daemon-launch.ts
//!
//! Daemon launch/readiness helpers.
//!
//! This module stays light on imports so clients can start a cold daemon before
//! the heavy main module graph loads. main.ts reuses the same memoized promise.
//!
//! TODO(slice): `DaemonClient`/daemon protocol (ca-daemon-b), config helpers
//! (ca-root), session-lease process identities (ca-session), orphan-process
//! journal (ca-misc), daemon socket helpers (ca-daemon-b) and child-process
//! helpers (ca-utils) are not landed. Private local stand-ins live at the bottom
//! of this module and are listed in the slice status file.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use super::command_registry::{
    is_help_command_request, public_command_names, removed_command_names,
};
use super::subprocess_launch::{create_cli_subprocess_env, format_current_cli_command, ProcessEnv};

pub const DAEMON_STARTUP_TIMEOUT_MS: f64 = 30_000.0;
pub const DAEMON_STARTUP_LOG_TAIL_BYTES: usize = 4 * 1024;
pub const DAEMON_STARTUP_EXIT_GRACE_MS: f64 = 2_000.0;

// TODO(slice): ca-daemon-b slice, modes/daemon/daemon-worker-protocol.ts.
pub const DAEMON_WORKER_ROLE_ENV: &str = "PRIME_AGENT_INTERNAL_DAEMON_WORKER";
pub const DAEMON_WORKER_TOKEN_ENV: &str = "PRIME_AGENT_INTERNAL_DAEMON_WORKER_TOKEN";
pub const DAEMON_WORKER_ACTIVE_SESSION_ID_ENV: &str =
    "PRIME_AGENT_INTERNAL_DAEMON_WORKER_ACTIVE_SESSION_ID";
pub const DAEMON_WORKER_RECOVERY_JOURNAL_ENV: &str =
    "PRIME_AGENT_INTERNAL_DAEMON_WORKER_RECOVERY_JOURNAL";
pub const DAEMON_WORKER_SUPERVISOR_SOCKET_ENV: &str =
    "PRIME_AGENT_INTERNAL_DAEMON_SUPERVISOR_SOCKET";
// TODO(slice): ca-misc slice, core/orphan-process-journal.ts.
pub const ORPHAN_PROCESS_JOURNAL_ENV: &str = "PRIME_AGENT_INTERNAL_ORPHAN_PROCESS_JOURNAL";
// TODO(slice): ca-session slice, core/session-lease.ts.
pub const SESSION_LEASES_ENABLED_ENV: &str = "PRIME_AGENT_INTERNAL_SESSION_LEASES";
pub const SESSION_LEASE_OWNER_ID_ENV: &str = "PRIME_AGENT_INTERNAL_SESSION_LEASE_OWNER_ID";
// TODO(slice): ca-daemon-b slice, modes/daemon/daemon-protocol.ts.
const DAEMON_PROTOCOL_VERSION: f64 = crate::modes::daemon::daemon_protocol::DAEMON_PROTOCOL_VERSION as f64;
const DAEMON_SCHEMA_ID: &str = crate::modes::daemon::daemon_protocol::DAEMON_SCHEMA_ID;
// TODO(slice): ca-root slice, config.ts.
const VERSION: &str = env!("CARGO_PKG_VERSION");
const CLIENT_ERROR_LOG_ENV: &str = "PRIME_AGENT_CLIENT_ERROR_LOG";

/// Narrowed `SessionSummary` surface used by this module. The full type lives in
/// the ca-daemon-b slice (modes/daemon/daemon-session-list.ts).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSummaryStub {
    pub is_session_active: bool,
    pub has_running_rlm_children: Option<bool>,
}

pub fn is_daemon_session_summary(value: &serde_json::Value) -> bool {
    let object = match value.as_object() {
        Some(object) => object,
        None => return false,
    };
    object
        .get("activeSessionId")
        .and_then(serde_json::Value::as_str)
        .is_some()
        || object
            .get("id")
            .and_then(serde_json::Value::as_str)
            .is_some()
}

fn delay(ms: u64) -> tokio::time::Sleep {
    tokio::time::sleep(std::time::Duration::from_millis(ms))
}

/// Local stand-in for `isSessionSummaryBusy` from daemon-session-list.js.
pub fn is_session_summary_busy(summary: &SessionSummaryStub) -> bool {
    summary.is_session_active || summary.has_running_rlm_children == Some(true)
}

pub fn is_session_busy(summary: &SessionSummaryStub) -> bool {
    is_session_summary_busy(summary)
}

/// `DaemonHello` as observed by this module.
#[derive(Debug, Clone, PartialEq)]
pub struct DaemonHello {
    pub app_version: Option<String>,
    pub protocol_version: f64,
    pub schema_id: Option<String>,
    pub build_id: Option<String>,
    pub launcher_path: Option<String>,
    pub entrypoint_path: Option<String>,
    pub executable_path: Option<String>,
    pub supervisor_pid: Option<i64>,
    pub supervisor_process_start_id: Option<String>,
}

impl DaemonHello {
    /// Parses a `daemon_hello` line, or returns `None` when the shape is wrong.
    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        let object = value.as_object()?;
        let protocol_version = object
            .get("protocol")
            .and_then(|protocol| protocol.get("version"))
            .and_then(serde_json::Value::as_f64)?;
        let runtime = object.get("runtime");
        Some(DaemonHello {
            app_version: object
                .get("appVersion")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            protocol_version,
            schema_id: object
                .get("schemaId")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            build_id: runtime
                .and_then(|runtime| runtime.get("buildId"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            launcher_path: runtime
                .and_then(|runtime| runtime.get("launcherPath"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            entrypoint_path: runtime
                .and_then(|runtime| runtime.get("entrypointPath"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            executable_path: runtime
                .and_then(|runtime| runtime.get("executablePath"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            supervisor_pid: object
                .get("supervisorPid")
                .and_then(serde_json::Value::as_i64),
            supervisor_process_start_id: object
                .get("supervisorProcessStartId")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DaemonVersionProbe {
    Absent,
    Current(DaemonHello),
    Stale(DaemonHello),
    Unresponsive,
}

pub fn is_current_daemon_hello(hello: &DaemonHello) -> bool {
    hello.protocol_version == DAEMON_PROTOCOL_VERSION
        && hello.schema_id.as_deref() == Some(DAEMON_SCHEMA_ID)
        && hello.app_version.as_deref() == Some(VERSION)
}

/// Connect to a running daemon and check whether it matches this client's protocol and app version.
pub async fn probe_daemon_version(socket_path: &str, hello_timeout_ms: f64) -> DaemonVersionProbe {
    let mut connected = None;
    for timeout_ms in [250.0, 2000.0] {
        if daemon_connect(socket_path, timeout_ms).await.is_ok() {
            connected = Some(());
            break;
        }
    }
    if connected.is_none() {
        return DaemonVersionProbe::Absent;
    }
    match daemon_wait_for_hello(socket_path, hello_timeout_ms).await {
        Err(_) => {
            // The supervisor accepts connections before startup and worker adoption finish.
            log_daemon_launch(&format!(
                "running daemon on {} sent no recognizable hello; waiting for startup",
                socket_path
            ));
            DaemonVersionProbe::Unresponsive
        }
        Ok(hello) => {
            let current = is_current_daemon_hello(&hello);
            if !current {
                log_daemon_launch(&format!(
                    "running daemon on {} is stale: daemon v{}/proto{}/schema {}/build {} vs client v{}/proto{}/schema {}/build {}",
                    socket_path,
                    hello.app_version.as_deref().unwrap_or("unknown"),
                    hello.protocol_version,
                    hello.schema_id.as_deref().unwrap_or("legacy"),
                    hello.build_id.as_deref().unwrap_or("unknown"),
                    VERSION,
                    DAEMON_PROTOCOL_VERSION,
                    DAEMON_SCHEMA_ID,
                    get_daemon_runtime_identity().build_id,
                ));
            }
            if current {
                DaemonVersionProbe::Current(hello)
            } else {
                DaemonVersionProbe::Stale(hello)
            }
        }
    }
}

pub struct ActiveDaemonSessions {
    pub sessions: Vec<serde_json::Value>,
    pub busy_client_owned_session_count: i64,
}

pub async fn list_active_daemon_session_summaries(
    socket_path: &str,
    include_client_owned: bool,
) -> Result<Vec<serde_json::Value>, String> {
    Ok(
        query_active_daemon_sessions(socket_path, include_client_owned)
            .await?
            .sessions,
    )
}

async fn query_active_daemon_sessions(
    socket_path: &str,
    include_client_owned: bool,
) -> Result<ActiveDaemonSessions, String> {
    let response = daemon_request(
        socket_path,
        serde_json::json!({ "type": "list", "includeClientOwned": include_client_owned }),
        None,
    )
    .await?;
    if response.get("success").and_then(serde_json::Value::as_bool) != Some(true) {
        return Err(response
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("Daemon request failed")
            .to_string());
    }
    let data = response.get("data").filter(|data| data.is_object());
    let data = match data {
        Some(data) => data,
        None => return Err("Daemon returned an invalid session list response".to_string()),
    };
    let sessions = match data.get("sessions") {
        Some(serde_json::Value::Array(sessions)) => sessions.clone(),
        _ => return Err("Daemon returned an invalid session list response".to_string()),
    };
    if !sessions.iter().all(is_daemon_session_summary) {
        return Err("Daemon returned an invalid session list response".to_string());
    }
    let busy_client_owned_session_count = match data.get("busyClientOwnedSessionCount") {
        None => 0,
        Some(value) => match value.as_i64() {
            Some(count) if count >= 0 => count,
            _ => return Err("Daemon returned an invalid client-owned session count".to_string()),
        },
    };
    Ok(ActiveDaemonSessions {
        sessions,
        busy_client_owned_session_count,
    })
}

/// Thrown when a stale-version daemon can't be replaced. The message is user-facing.
#[derive(Debug, Clone)]
pub struct StaleDaemonError {
    pub socket_path: String,
    pub message: String,
}

impl StaleDaemonError {
    pub fn new(socket_path: &str, hello: Option<&DaemonHello>) -> Self {
        let daemon_identity = match hello {
            Some(hello) => format!(
                "Daemon: v{}, protocol {}, schema {}, build {}, PID {}, executable {}",
                hello.app_version.as_deref().unwrap_or("unknown"),
                hello.protocol_version,
                hello.schema_id.as_deref().unwrap_or("legacy"),
                hello.build_id.as_deref().unwrap_or("unknown"),
                hello
                    .supervisor_pid
                    .map(|pid| pid.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                hello
                    .launcher_path
                    .as_deref()
                    .or(hello.entrypoint_path.as_deref())
                    .or(hello.executable_path.as_deref())
                    .unwrap_or("unknown"),
            ),
            None => format!("Daemon: unknown build on {}", socket_path),
        };
        let client = get_daemon_runtime_identity();
        Self {
            socket_path: socket_path.to_string(),
            message: format!(
                "An incompatible Prime Agent daemon is running.\n\n{}\nClient: v{}, protocol {}, schema {}, build {}, executable {}\n\nRun:\n{}\n\nThen retry the original command.",
                daemon_identity,
                VERSION,
                DAEMON_PROTOCOL_VERSION,
                DAEMON_SCHEMA_ID,
                client.build_id,
                client
                    .launcher_path
                    .as_deref()
                    .or(client.entrypoint_path.as_deref())
                    .unwrap_or(client.executable_path.as_str()),
                format_current_cli_command(
                    &["shutdown".to_string(), "--force".to_string()],
                    &current_process_env(),
                ),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DaemonProcessIdentity {
    pub pid: i64,
    pub process_start_id: Option<String>,
}

pub const PROCESS_START_ID_POLL_INTERVAL_MS: f64 = 1000.0;

fn has_process_identity_exited(
    identity: Option<&DaemonProcessIdentity>,
    verify_process_start_id: bool,
) -> bool {
    let identity = match identity {
        Some(identity) => identity,
        None => return true,
    };
    if !process_exists(identity.pid) {
        return true;
    }
    if identity.process_start_id.is_none() || !verify_process_start_id {
        return false;
    }
    let current_start_id = get_process_start_id(identity.pid);
    current_start_id.is_some()
        && current_start_id.as_deref() != identity.process_start_id.as_deref()
}

async fn wait_for_daemon_gone(
    socket_path: &str,
    timeout_ms: f64,
    require_socket_cleanup: bool,
    expected_identity: Option<&DaemonProcessIdentity>,
) -> bool {
    let deadline = now_ms() + timeout_ms;
    let mut next_process_start_id_poll_at = 0.0f64;
    let mut has_expected_process_exited = |force_start_id_poll: bool| {
        let now = now_ms();
        let verify_process_start_id = force_start_id_poll || now >= next_process_start_id_poll_at;
        if verify_process_start_id {
            next_process_start_id_poll_at = now + PROCESS_START_ID_POLL_INTERVAL_MS;
        }
        has_process_identity_exited(expected_identity, verify_process_start_id)
    };
    while now_ms() < deadline {
        if !can_connect_to_daemon(socket_path, 250.0).await
            && (!require_socket_cleanup
                || process_platform() == "win32"
                || !Path::new(socket_path).exists())
            && has_expected_process_exited(false)
        {
            return true;
        }
        delay(25).await;
    }
    // A daemon can exit without removing its Unix socket (for example, after a crash
    // during shutdown). Once the cleanup grace has elapsed, a non-listening socket
    // is safe for the replacement daemon's guarded startup path to reclaim.
    require_socket_cleanup
        && !can_connect_to_daemon(socket_path, 250.0).await
        && has_expected_process_exited(true)
}

fn process_identity_from_daemon_hello(
    hello: Option<&DaemonHello>,
) -> Option<DaemonProcessIdentity> {
    let pid = hello.and_then(|hello| hello.supervisor_pid)?;
    if pid <= 0 {
        return None;
    }
    let process_start_id = hello
        .and_then(|hello| hello.supervisor_process_start_id.clone())
        .or_else(|| get_process_start_id(pid));
    Some(DaemonProcessIdentity {
        pid,
        process_start_id,
    })
}

pub async fn shutdown_connected_daemon_and_wait(
    socket_path: &str,
    timeout_ms: f64,
    hello: Option<&DaemonHello>,
) -> bool {
    let mut shutdown_accepted = false;
    let expected_identity = process_identity_from_daemon_hello(hello);
    // A connect failure isn't treated as "gone"; waitForDaemonGone is the source of truth.
    match daemon_request(socket_path, serde_json::json!({ "type": "shutdown" }), None).await {
        Ok(response) => {
            shutdown_accepted =
                response.get("success").and_then(serde_json::Value::as_bool) == Some(true)
        }
        Err(_) => {}
    }
    wait_for_daemon_gone(
        socket_path,
        timeout_ms,
        shutdown_accepted,
        expected_identity.as_ref(),
    )
    .await
}

pub async fn shutdown_daemon_and_wait(socket_path: &str, timeout_ms: f64) -> bool {
    if daemon_connect(socket_path, 1000.0).await.is_err() {
        return wait_for_daemon_gone(socket_path, timeout_ms, false, None).await;
    }
    let hello = daemon_wait_for_hello(socket_path, 2000.0).await.ok();
    shutdown_connected_daemon_and_wait(socket_path, timeout_ms, hello.as_ref()).await
}

// activeSessions is None when the daemon is reachable but its sessions couldn't
// be listed — callers must treat that as "possibly busy", not idle.
#[derive(Debug, Clone, PartialEq)]
pub struct RunningDaemonProbe {
    pub reachable: bool,
    pub active_sessions: Option<Vec<SessionSummaryStub>>,
    pub busy_client_owned_session_count: Option<i64>,
}

pub async fn probe_running_daemon_sessions(socket_path: &str) -> RunningDaemonProbe {
    if daemon_connect(socket_path, 1000.0).await.is_err() {
        return RunningDaemonProbe {
            reachable: false,
            active_sessions: None,
            busy_client_owned_session_count: None,
        };
    }
    match query_active_daemon_sessions(socket_path, true).await {
        Ok(result) => {
            let active_sessions: Vec<SessionSummaryStub> = result
                .sessions
                .iter()
                .filter(|summary| {
                    summary
                        .get("activeSessionId")
                        .and_then(serde_json::Value::as_str)
                        .is_some()
                })
                .map(session_summary_stub)
                .collect();
            RunningDaemonProbe {
                reachable: true,
                active_sessions: Some(active_sessions),
                busy_client_owned_session_count: if result.busy_client_owned_session_count > 0 {
                    Some(result.busy_client_owned_session_count)
                } else {
                    None
                },
            }
        }
        Err(_) => RunningDaemonProbe {
            reachable: true,
            active_sessions: None,
            busy_client_owned_session_count: None,
        },
    }
}

/// `isSessionBusy` reads only two fields of the summary.
fn session_summary_stub(summary: &serde_json::Value) -> SessionSummaryStub {
    SessionSummaryStub {
        is_session_active: summary
            .get("isSessionActive")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        has_running_rlm_children: summary
            .get("hasRunningRlmChildren")
            .and_then(serde_json::Value::as_bool),
    }
}

// A reachable stale peer requires an explicit shutdown: a client-side idle
// probe cannot fence new admissions or identify the peer on a later connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleDaemonDisposition {
    Current,
    Stopped,
    Busy,
}

async fn shutdown_stale_daemon_if_not_busy(socket_path: &str) -> StaleDaemonDisposition {
    if daemon_connect(socket_path, 1000.0).await.is_err() {
        return if wait_for_daemon_gone(socket_path, 5000.0, false, None).await {
            StaleDaemonDisposition::Stopped
        } else {
            StaleDaemonDisposition::Busy
        };
    }

    let hello = daemon_wait_for_hello(socket_path, 2000.0).await.ok();
    if let Some(hello) = &hello {
        if is_current_daemon_hello(hello) {
            log_daemon_launch(&format!(
                "daemon on {} finished starting while staleness was being checked; reusing it",
                socket_path
            ));
            return StaleDaemonDisposition::Current;
        }
    }
    log_daemon_launch(&format!(
        "refusing automatic replacement of reachable stale daemon on {}: explicit shutdown is required to preserve work admitted after a probe",
        socket_path
    ));
    StaleDaemonDisposition::Busy
}

async fn ensure_daemon_running(socket_path: &str, spawn_cwd: Option<&str>) -> Result<(), String> {
    let probe_started_at = now_ms();
    let mut probe = probe_daemon_version(socket_path, 2000.0).await;
    if probe == DaemonVersionProbe::Unresponsive {
        let remaining_startup_ms =
            (DAEMON_STARTUP_TIMEOUT_MS - (now_ms() - probe_started_at)).max(1.0);
        probe = probe_daemon_version(socket_path, remaining_startup_ms).await;
    }
    match probe {
        DaemonVersionProbe::Current(_) => return Ok(()),
        DaemonVersionProbe::Unresponsive => {
            return Err(format!(
                "Prime Agent daemon on {} accepted connections but did not finish startup within {} seconds. It was left running to avoid interrupting active work.\n\nRun:\n{}\n\nThen retry the original command.",
                socket_path,
                DAEMON_STARTUP_TIMEOUT_MS / 1000.0,
                format_current_cli_command(
                    &["shutdown".to_string(), "--force".to_string()],
                    &current_process_env()
                ),
            ));
        }
        DaemonVersionProbe::Stale(hello) => {
            let disposition = shutdown_stale_daemon_if_not_busy(socket_path).await;
            if disposition == StaleDaemonDisposition::Current {
                return Ok(());
            }
            if disposition == StaleDaemonDisposition::Busy {
                return Err(StaleDaemonError::new(socket_path, Some(&hello)).message);
            }
        }
        DaemonVersionProbe::Absent => {}
    }

    let entrypoint = current_entrypoint();
    if entrypoint.is_empty() {
        return Err("Cannot determine current CLI entrypoint for daemon launch".to_string());
    }

    // Strip inherited daemon worker/supervisor role env vars so the spawned
    // daemon supervisor does not inherit worker-mode behavior. Without this,
    // a CLI running inside a daemon worker (e.g. a test spawned by the Prime
    // Agent daemon) would launch the supervisor in worker mode, which listens
    // on the socket but never sends the daemon_hello handshake.
    let mut env = create_cli_subprocess_env(
        &current_process_env(),
        Some(&entrypoint),
        &current_exec_args(),
    );
    env.shift_remove(DAEMON_WORKER_ROLE_ENV);
    env.shift_remove(DAEMON_WORKER_TOKEN_ENV);
    env.shift_remove(DAEMON_WORKER_ACTIVE_SESSION_ID_ENV);
    env.shift_remove(DAEMON_WORKER_RECOVERY_JOURNAL_ENV);
    env.shift_remove(DAEMON_WORKER_SUPERVISOR_SOCKET_ENV);
    env.shift_remove(ORPHAN_PROCESS_JOURNAL_ENV);
    env.shift_remove(SESSION_LEASES_ENABLED_ENV);
    env.shift_remove(SESSION_LEASE_OWNER_ID_ENV);

    let log_offset = current_daemon_log_size(socket_path);
    let spawn_cwd = spawn_cwd
        .map(str::to_string)
        .unwrap_or_else(|| current_cwd());
    let launch = super::subprocess_launch::create_cli_subprocess_launch_spec(
        &[
            "--mode".to_string(),
            "daemon".to_string(),
            "--daemon-socket".to_string(),
            socket_path.to_string(),
        ],
        None,
        &[],
        None,
    );
    // A pipe would tie the daemon's stderr to this short-lived CLI
    // (EPIPE once it exits); crash details come from the daemon log,
    // which the supervisor writes to before rethrowing startup errors.
    let child = spawn_hidden_detached(&launch.command, &launch.args, &spawn_cwd, &env, socket_path);

    let mut child_failure: Option<ChildFailure> = None;
    if let Some(child) = &child {
        child_failure = child
            .failure
            .lock()
            .ok()
            .and_then(|failure| failure.clone());
    }

    // A child exit is not immediately fatal: it may have lost the socket to a
    // concurrent launcher whose daemon is still booting. Keep probing for a
    // short grace window before attributing the failure to the exit.
    let deadline = now_ms() + DAEMON_STARTUP_TIMEOUT_MS;
    let mut exit_deadline: Option<f64> = None;
    while now_ms() < deadline.min(exit_deadline.unwrap_or(f64::INFINITY)) {
        let started = probe_daemon_version(socket_path, 2000.0).await;
        if matches!(started, DaemonVersionProbe::Current(_)) {
            return Ok(());
        }
        if child_failure.is_none() {
            if let Some(child) = &child {
                child_failure = child
                    .failure
                    .lock()
                    .ok()
                    .and_then(|failure| failure.clone());
            }
        }
        if child_failure.is_some() && exit_deadline.is_none() {
            exit_deadline = Some(now_ms() + DAEMON_STARTUP_EXIT_GRACE_MS);
        }
        delay(25).await;
    }

    if let Some(failure) = child_failure {
        let log_tail = read_daemon_log_tail(socket_path, log_offset);
        return Err(match failure {
            ChildFailure::Error(message) => {
                format!(
                    "Failed to spawn Prime Agent daemon: {}.{}",
                    message, log_tail
                )
            }
            ChildFailure::Exit { code, signal } => {
                let signal = signal
                    .map(|signal| format!(", signal {}", signal))
                    .unwrap_or_default();
                format!(
                    "Prime Agent daemon exited during startup (code {}{}).{}",
                    code.map(|code| code.to_string())
                        .unwrap_or_else(|| "unknown".to_string()),
                    signal,
                    log_tail
                )
            }
        });
    }
    Err(format!(
        "Timed out waiting for daemon to start on {}.{}",
        socket_path,
        read_daemon_log_tail(socket_path, log_offset)
    ))
}

fn current_daemon_log_size(socket_path: &str) -> u64 {
    std::fs::metadata(get_daemon_log_path(socket_path))
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

/// Reads only log content written after `offset`, so stale content from earlier
/// daemon runs is not misattributed to this startup attempt.
fn read_daemon_log_tail(socket_path: &str, offset: u64) -> String {
    let log_path = get_daemon_log_path(socket_path);
    let mut tail = String::new();
    if let Ok(content) = std::fs::read(&log_path) {
        // A rotation may have shrunk the file below the pre-spawn byte offset.
        let start = if (content.len() as u64) < offset {
            0
        } else {
            offset as usize
        };
        let sliced = &content[start.min(content.len())..];
        let tail_start = sliced.len().saturating_sub(DAEMON_STARTUP_LOG_TAIL_BYTES);
        tail = String::from_utf8_lossy(&sliced[tail_start..])
            .trim()
            .to_string();
    }
    if tail.is_empty() {
        // Missing log means the daemon crashed before logging was set up.
        format!(" The daemon wrote nothing to its log ({}).", log_path)
    } else {
        format!(" Recent daemon log ({}):\n{}", log_path, tail)
    }
}

/// One in-flight `ensureInteractiveDaemonRunning` per socket, shared by every
/// caller: the first call starts the work, later calls await the same result.
struct SharedEnsure {
    result: Mutex<Option<Result<(), String>>>,
    notify: tokio::sync::Notify,
}

fn ensure_promises() -> &'static Mutex<HashMap<String, std::sync::Arc<SharedEnsure>>> {
    static PROMISES: OnceLock<Mutex<HashMap<String, std::sync::Arc<SharedEnsure>>>> =
        OnceLock::new();
    PROMISES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Ensure a current-version daemon is listening on socketPath, spawning one if
/// needed. Memoized per socket so the early kick from cli.ts and the await in
/// main.ts share one probe/spawn; failed attempts are forgotten so a later call
/// retries (and surfaces the real error at its await site).
pub async fn ensure_interactive_daemon_running(
    socket_path: &str,
    spawn_cwd: Option<&str>,
) -> Result<(), String> {
    let (shared, is_first) = {
        let mut promises = ensure_promises().lock().unwrap();
        match promises.get(socket_path) {
            Some(shared) => (shared.clone(), false),
            None => {
                let shared = std::sync::Arc::new(SharedEnsure {
                    result: Mutex::new(None),
                    notify: tokio::sync::Notify::new(),
                });
                promises.insert(socket_path.to_string(), shared.clone());
                (shared, true)
            }
        }
    };

    if is_first {
        let result = ensure_daemon_running(socket_path, spawn_cwd).await;
        *shared.result.lock().unwrap() = Some(result.clone());
        shared.notify.notify_waiters();
        let mut promises = ensure_promises().lock().unwrap();
        if promises
            .get(socket_path)
            .map(|entry| std::sync::Arc::ptr_eq(entry, &shared))
            .unwrap_or(false)
        {
            promises.remove(socket_path);
        }
        return result;
    }

    loop {
        if let Some(result) = shared.result.lock().unwrap().clone() {
            return result;
        }
        let notified = shared.notify.notified();
        if let Some(result) = shared.result.lock().unwrap().clone() {
            return result;
        }
        notified.await;
    }
}

pub const EARLY_LAUNCH_EXCLUDED_FLAGS: [&str; 6] = [
    "--help",
    "-h",
    "--version",
    "-v",
    "--list-models",
    "--export",
];
pub const EARLY_LAUNCH_VALUE_FLAGS: [&str; 28] = [
    "--mode",
    "--daemon-socket",
    "--provider",
    "--model",
    "--api-key",
    "--cwd",
    "--system-prompt",
    "--append-system-prompt",
    "--fork",
    "--session-dir",
    "--models",
    "--tools",
    "-t",
    "--thinking",
    "--extension",
    "-e",
    "--skill",
    "--prompt-template",
    "--theme",
    "--autonomous-gate",
    "--autonomous-gate-retries",
    "--autonomous-gate-timeout-ms",
    "--autonomous-max-continuations",
    "--autonomous-max-turns",
    "--autonomous-max-tokens",
    "--autonomous-timeout-ms",
    "--goal",
    "--goal-token-budget",
];

fn find_first_early_launch_positional(args: &[String]) -> Option<(usize, String)> {
    let mut index = 0usize;
    while index < args.len() {
        let arg = args[index].clone();
        if arg == "--" {
            return args.get(index + 1).map(|value| (index + 1, value.clone()));
        }
        if EARLY_LAUNCH_VALUE_FLAGS.contains(&arg.as_str()) {
            index += 2;
            continue;
        }
        if arg == "--resume" || arg == "-r" {
            if args
                .get(index + 1)
                .map(|next| !next.starts_with('-'))
                .unwrap_or(false)
            {
                index += 2;
                continue;
            }
            index += 1;
            continue;
        }
        if !arg.starts_with('-') {
            return Some((index, arg));
        }
        index += 1;
    }
    None
}

pub fn should_start_daemon_early(args: &[String], startup_benchmark: bool) -> bool {
    if startup_benchmark {
        return false;
    }
    if let Some(mode_index) = args.iter().position(|arg| arg == "--mode") {
        if args.get(mode_index + 1).map(String::as_str) == Some("daemon") {
            return false;
        }
    }
    if args
        .iter()
        .any(|arg| EARLY_LAUNCH_EXCLUDED_FLAGS.contains(&arg.as_str()))
    {
        return false;
    }
    if args.iter().any(|arg| arg == "--print" || arg == "-p") {
        return true;
    }
    let first_positional = find_first_early_launch_positional(args);
    let is_help_command = match &first_positional {
        Some((index, value)) => {
            value == "help" && {
                let rest: Vec<&str> = args[index + 1..].iter().map(String::as_str).collect();
                is_help_command_request(&rest)
            }
        }
        None => false,
    };
    if let Some((_, value)) = &first_positional {
        if removed_command_names().contains(value.as_str())
            || (public_command_names().contains(value.as_str())
                && value != "agents"
                && (value != "help" || is_help_command))
        {
            return false;
        }
    }
    true
}

pub fn maybe_start_daemon_early(args: &[String]) {
    let benchmark_flag = std::env::var("PI_STARTUP_BENCHMARK")
        .unwrap_or_default()
        .to_lowercase();
    let startup_benchmark =
        benchmark_flag == "1" || benchmark_flag == "true" || benchmark_flag == "yes";
    if !should_start_daemon_early(args, startup_benchmark) {
        return;
    }
    let socket_index = args.iter().position(|arg| arg == "--daemon-socket");
    let raw_socket_path = match socket_index.and_then(|index| args.get(index + 1)) {
        Some(socket_path) => socket_path.clone(),
        None => default_daemon_socket_path(),
    };
    let cwd_index = args.iter().position(|arg| arg == "--cwd");
    let cwd_arg = cwd_index.and_then(|index| args.get(index + 1));
    let spawn_cwd = cwd_arg.map(|cwd_arg| resolve_path(&expand_tilde_path(cwd_arg)));
    if let Some(spawn_cwd) = &spawn_cwd {
        if !Path::new(spawn_cwd).exists() {
            return;
        }
    }
    let socket_path = normalize_socket_path(&raw_socket_path, spawn_cwd.as_deref());
    if let Ok(runtime) = tokio::runtime::Handle::try_current() {
        runtime.spawn(async move {
            if let Err(error) =
                ensure_interactive_daemon_running(&socket_path, spawn_cwd.as_deref()).await
            {
                log_daemon_launch(&error);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Private local stand-ins for not-yet-landed slices.
// ---------------------------------------------------------------------------

fn process_platform() -> &'static str {
    crate::utils::pi_user_agent::process_platform()
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as f64)
        .unwrap_or(0.0)
}

fn current_cwd() -> String {
    std::env::current_dir()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn current_exec_path() -> String {
    std::env::current_exe()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn current_entrypoint() -> String {
    current_exec_path()
}

fn current_exec_args() -> Vec<String> {
    Vec::new()
}

/// Local stand-in for the `NodeJS.ProcessEnv` snapshot `process.env` gives.
fn current_process_env() -> ProcessEnv {
    let mut environment = ProcessEnv::new();
    for (key, value) in std::env::vars() {
        environment.insert(key, value);
    }
    environment
}

fn expand_tilde_path(path: &str) -> String {
    if path == "~" {
        return home_dir();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return Path::new(&home_dir())
            .join(rest)
            .to_string_lossy()
            .to_string();
    }
    if process_platform() == "win32" {
        if let Some(rest) = path.strip_prefix("~\\") {
            return Path::new(&home_dir())
                .join(rest)
                .to_string_lossy()
                .to_string();
        }
    }
    path.to_string()
}

fn home_dir() -> String {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default()
}

fn resolve_path(path: &str) -> String {
    let candidate = PathBuf::from(path);
    let joined = if candidate.is_absolute() {
        candidate
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(candidate)
    };
    normalize_lexically(&joined)
}

fn normalize_lexically(path: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut prefix = String::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::Prefix(prefix_component) => {
                prefix.push_str(&prefix_component.as_os_str().to_string_lossy())
            }
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

/// Local stand-in for `normalizeSocketPath` from daemon-socket.js.
fn normalize_socket_path(socket_path: &str, base_dir: Option<&str>) -> String {
    crate::utils::daemon_socket_path::normalize_socket_path(socket_path, base_dir)
}

/// Local stand-in for `defaultDaemonSocketDir`.
fn default_daemon_socket_dir() -> String {
    crate::modes::daemon::daemon_socket::default_daemon_socket_dir()
}

/// Local stand-in for `defaultDaemonSocketPath`.
fn default_daemon_socket_path() -> String {
    crate::modes::daemon::daemon_socket::default_daemon_socket_path()
}

/// Local stand-in for `getDaemonLogPath` from ../config.js.
fn get_daemon_log_path(socket_path: &str) -> String {
    use sha2::{Digest, Sha256};
    let normalized = normalize_socket_path(socket_path, None);
    let hash = format!("{:x}", Sha256::digest(normalized.as_bytes()));
    let base_name = Path::new(&normalized)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    Path::new(&get_logs_dir())
        .join(format!("{}.{}.log", base_name, &hash[..8]))
        .to_string_lossy()
        .to_string()
}

/// Local stand-in for `getLogsDir` from ../config.js.
fn get_logs_dir() -> String {
    Path::new(&get_agent_dir())
        .join("logs")
        .to_string_lossy()
        .to_string()
}

/// Local stand-in for `getAgentDir` from ../config.js.
fn get_agent_dir() -> String {
    if let Ok(env_dir) = std::env::var("PRIME_AGENT_CODING_AGENT_DIR") {
        return expand_tilde_path(&env_dir);
    }
    if let Ok(env_dir) = std::env::var("PI_CODING_AGENT_DIR") {
        return expand_tilde_path(&env_dir);
    }
    Path::new(&home_dir())
        .join(".prime/agent")
        .to_string_lossy()
        .to_string()
}

/// Local stand-in for `appendRotatingLog(getClientErrorLogPath(), ...)`.
fn log_daemon_launch(message: &str) {
    let log_path = std::env::var(CLIENT_ERROR_LOG_ENV).unwrap_or_else(|_| {
        Path::new(&get_logs_dir())
            .join("client-errors.log")
            .to_string_lossy()
            .to_string()
    });
    append_rotating_log(
        &log_path,
        &format!("[{}] daemon-launch: {}", now_iso8601(), message),
    );
}

const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

/// Local stand-in for `appendRotatingLog`: single-generation rotation, best effort.
fn append_rotating_log(log_path: &str, message: &str) {
    let path = Path::new(log_path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(metadata) = std::fs::metadata(path) {
        if metadata.len() > MAX_LOG_BYTES {
            let old_path = format!("{}.old", log_path);
            let _ = std::fs::remove_file(&old_path);
            let _ = std::fs::rename(path, &old_path);
        }
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        use std::io::Write;
        let _ = writeln!(file, "{}", message);
    }
}

fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Local stand-in for `DaemonRuntimeIdentity` from daemon-runtime-identity.js.
#[derive(Debug, Clone)]
struct DaemonRuntimeIdentity {
    build_id: String,
    launcher_path: Option<String>,
    entrypoint_path: Option<String>,
    executable_path: String,
}

fn get_daemon_runtime_identity() -> DaemonRuntimeIdentity {
    DaemonRuntimeIdentity {
        build_id: format!("rust-port-{}", VERSION),
        launcher_path: std::env::var("PRIME_AGENT_LAUNCHER_PATH").ok(),
        entrypoint_path: None,
        executable_path: current_exec_path(),
    }
}

/// Local stand-in for `getProcessStartId` from session-lease.js.
fn get_process_start_id(pid: i64) -> Option<String> {
    if pid <= 0 {
        return None;
    }
    if process_platform() == "win32" {
        return None;
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

fn process_exists(pid: i64) -> bool {
    kill_process(pid, 0)
}

#[cfg(unix)]
fn kill_process(pid: i64, signal_number: i32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, signal_number) == 0 }
}

#[cfg(windows)]
fn kill_process(pid: i64, signal_number: i32) -> bool {
    use windows_sys::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE};
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

// ---------------------------------------------------------------------------
// Startup probes share the native client transport, including Windows pipe retry.
// ---------------------------------------------------------------------------

async fn daemon_connect(
    socket_path: &str,
    timeout_ms: f64,
) -> Result<crate::modes::daemon::daemon_client::DaemonSocketStream, String> {
    match tokio::time::timeout(
        std::time::Duration::from_millis(timeout_ms.max(1.0) as u64),
        crate::modes::daemon::daemon_client::connect_daemon_socket(socket_path),
    )
    .await
    {
        Ok(Ok(stream)) => Ok(stream),
        Ok(Err(error)) => Err(error.to_string()),
        Err(_) => Err(format!(
            "Timed out after {}ms connecting to the Prime Agent daemon. {}",
            timeout_ms,
            daemon_endpoint_details(socket_path)
        )),
    }
}

fn daemon_endpoint_details(socket_path: &str) -> String {
    format!("(daemon socket: {})", socket_path)
}

async fn can_connect_to_daemon(socket_path: &str, timeout_ms: f64) -> bool {
    daemon_connect(socket_path, timeout_ms).await.is_ok()
}

async fn daemon_exchange(
    socket_path: &str,
    command: Option<serde_json::Value>,
    response_timeout_ms: f64,
) -> Result<Vec<serde_json::Value>, String> {
    use tokio::io::{AsyncWriteExt, BufReader};

    let stream = daemon_connect(socket_path, 1000.0).await?;
    let (read_half, mut write_half) = tokio::io::split(stream);
    let mut reader = BufReader::new(read_half);
    let mut lines: Vec<serde_json::Value> = Vec::new();

    lines.push(read_daemon_record(&mut reader, response_timeout_ms, |value| value["type"] == "daemon_hello").await?);

    if let Some(mut command) = command {
        let id = command.get("id").and_then(serde_json::Value::as_str)
            .map(str::to_string).unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        command["id"] = serde_json::Value::String(id.clone());
        let line = format!(
            "{}\n",
            serde_json::to_string(&command).map_err(|error| error.to_string())?
        );
        write_half
            .write_all(line.as_bytes())
            .await
            .map_err(|error| error.to_string())?;
        lines.push(read_daemon_record(&mut reader, response_timeout_ms, |value| value["type"] == "response" && value["id"] == id).await?);
    }

    Ok(lines)
}

async fn read_daemon_record(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
    timeout_ms: f64,
    accept: impl Fn(&serde_json::Value) -> bool,
) -> Result<serde_json::Value, String> {
    use tokio::io::AsyncBufReadExt;
    // Keep one deadline across blank keepalives and unrelated broadcasts.
    tokio::time::timeout(std::time::Duration::from_millis(timeout_ms.max(1.0) as u64), async {
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).await.map_err(|error| error.to_string())? == 0 {
                return Err("Daemon connection closed before the expected record".to_string());
            }
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                if accept(&value) { return Ok(value); }
            }
        }
    }).await.map_err(|_| format!("Timed out after {timeout_ms}ms waiting for a daemon record"))?
}

async fn daemon_request(
    socket_path: &str,
    command: serde_json::Value,
    timeout_ms: Option<f64>,
) -> Result<serde_json::Value, String> {
    let lines = daemon_exchange(socket_path, Some(command), timeout_ms.unwrap_or(30_000.0)).await?;
    let response = lines
        .iter()
        .rev()
        .find(|line| line.get("type").and_then(serde_json::Value::as_str) == Some("response"));
    response
        .cloned()
        .ok_or_else(|| format!("Daemon on {} sent no response", socket_path))
}

async fn daemon_wait_for_hello(socket_path: &str, timeout_ms: f64) -> Result<DaemonHello, String> {
    let lines = daemon_exchange(socket_path, None, timeout_ms).await?;
    for line in &lines {
        if line.get("type").and_then(serde_json::Value::as_str) == Some("daemon_hello") {
            if let Some(hello) = DaemonHello::from_json(line) {
                return Ok(hello);
            }
        }
    }
    Err(format!(
        "Timed out after {}ms waiting for the Prime Agent daemon handshake. {}",
        timeout_ms,
        daemon_endpoint_details(socket_path)
    ))
}

#[derive(Debug, Clone)]
enum ChildFailure {
    Error(String),
    Exit {
        code: Option<i32>,
        signal: Option<String>,
    },
}

struct SpawnedChild {
    pid: u32,
    failure: std::sync::Arc<Mutex<Option<ChildFailure>>>,
}

/// Local stand-in for `spawnHidden(command, args, { cwd, detached: true, env, stdio: "ignore" })`.
fn spawn_hidden_detached(
    command: &str,
    args: &[String],
    cwd: &str,
    env: &ProcessEnv,
    socket_path: &str,
) -> Option<SpawnedChild> {
    let mut process = std::process::Command::new(command);
    process
        .args(args)
        .current_dir(cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .env_clear();
    let log_path = get_daemon_log_path(socket_path);
    if let Some(parent) = Path::new(&log_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut log_options = std::fs::OpenOptions::new();
    log_options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::{fs::OpenOptionsExt, process::CommandExt};
        log_options.mode(0o600);
        process.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        process.creation_flags(0x0800_0000 | 0x0000_0200);
    }
    if let Ok(log) = log_options.open(&log_path) {
        process.stderr(std::process::Stdio::from(log));
    }
    for (key, value) in env {
        process.env(key, value);
    }
    let failure: std::sync::Arc<Mutex<Option<ChildFailure>>> =
        std::sync::Arc::new(Mutex::new(None));
    let failure_slot = failure.clone();
    match process.spawn() {
        Ok(mut child) => {
            let pid = child.id();
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
            Some(SpawnedChild { pid, failure })
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
    status
        .signal()
        .map(|signal| signal_name(signal).to_string())
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

    #[cfg(windows)]
    #[tokio::test]
    async fn backlog_stale_or_replaced_peer_never_receives_automatic_shutdown() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::windows::named_pipe::ServerOptions;
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        let socket = format!(r"\\.\pipe\optimus-backlog-stale-{}", uuid::Uuid::new_v4());
        let mut server = ServerOptions::new().first_pipe_instance(true).create(&socket).unwrap();
        let commands = Arc::new(AtomicUsize::new(0));
        let observed = commands.clone();
        let listen_path = socket.clone();
        let task = tokio::spawn(async move {
            let mut identity = 100;
            loop {
                server.connect().await.unwrap();
                let connected = server;
                server = ServerOptions::new().create(&listen_path).unwrap();
                identity += 1;
                let observed = observed.clone();
                tokio::spawn(async move {
                    let (reader, mut writer) = tokio::io::split(connected);
                    let hello = serde_json::json!({"type":"daemon_hello","protocol":{"version":DAEMON_PROTOCOL_VERSION},"schemaId":"stale","appVersion":"0.0.0","supervisorPid":identity});
                    if writer.write_all(format!("{hello}\n").as_bytes()).await.is_err() { return; }
                    let mut line = String::new();
                    if BufReader::new(reader).read_line(&mut line).await.unwrap_or(0) > 0 { observed.fetch_add(1, Ordering::SeqCst); }
                });
            }
        });
        assert_eq!(shutdown_stale_daemon_if_not_busy(&socket).await, StaleDaemonDisposition::Busy);
        assert_eq!(commands.load(Ordering::SeqCst), 0, "automatic stale replacement must dispatch no mutation to either identity");
        assert!(can_connect_to_daemon(&socket, 500.0).await);
        task.abort();
        let _ = task.await;
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn startup_windows_probe_and_request_use_named_pipe() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::windows::named_pipe::ServerOptions;
        let socket = format!(r"\\.\pipe\optimus-startup-test-{}", uuid::Uuid::new_v4());
        let mut server = ServerOptions::new().first_pipe_instance(true).create(&socket).unwrap();
        let listen_path = socket.clone();
        let task = tokio::spawn(async move {
            loop {
                server.connect().await.unwrap();
                let connected = server;
                server = ServerOptions::new().create(&listen_path).unwrap();
                tokio::spawn(async move {
                    let (reader, mut writer) = tokio::io::split(connected);
                    let hello = serde_json::json!({"type":"daemon_hello","protocol":{"version":DAEMON_PROTOCOL_VERSION},"schemaId":DAEMON_SCHEMA_ID,"appVersion":VERSION});
                    if writer.write_all(format!("\n{hello}\n").as_bytes()).await.is_err() { return; }
                    let mut line = String::new();
                    if BufReader::new(reader).read_line(&mut line).await.unwrap_or(0) == 0 { return; }
                    let request: serde_json::Value = serde_json::from_str(&line).unwrap();
                    let response = serde_json::json!({"type":"response","id":request["id"],"command":request["type"],"success":true,"data":{"sessions":[]}});
                    let _ = writer.write_all(format!("\n{{\"type\":\"heartbeats_changed\"}}\n{response}\n").as_bytes()).await;
                });
            }
        });
        let version = probe_daemon_version(&socket, 500.0).await;
        assert!(matches!(version, DaemonVersionProbe::Current(_)), "{version:?}");
        let response = daemon_request(&socket, serde_json::json!({"type":"list","id":"probe"}), Some(500.0)).await.unwrap();
        assert_eq!(response["id"], "probe");
        assert_eq!(response["success"], true);
        task.abort();
    }

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn hello_requires_a_protocol_object() {
        let value = serde_json::json!({"type": "daemon_hello", "protocol": {"version": 7}});
        let hello = DaemonHello::from_json(&value).unwrap();
        assert_eq!(hello.protocol_version, 7.0);
        assert_eq!(hello.app_version, None);
        assert!(DaemonHello::from_json(&serde_json::json!({"type": "daemon_hello"})).is_none());
    }

    #[test]
    fn current_hello_matches_version_schema_and_protocol() {
        let mut hello = DaemonHello {
            app_version: Some(VERSION.to_string()),
            protocol_version: crate::modes::daemon::daemon_protocol::DAEMON_PROTOCOL_VERSION as f64,
            schema_id: Some(crate::modes::daemon::daemon_protocol::DAEMON_SCHEMA_ID.to_string()),
            build_id: None,
            launcher_path: None,
            entrypoint_path: None,
            executable_path: None,
            supervisor_pid: None,
            supervisor_process_start_id: None,
        };
        assert!(is_current_daemon_hello(&hello));
        hello.schema_id = Some("legacy".to_string());
        assert!(!is_current_daemon_hello(&hello));
        hello.schema_id = Some(DAEMON_SCHEMA_ID.to_string());
        hello.protocol_version = 6.0;
        assert!(!is_current_daemon_hello(&hello));
        hello.protocol_version = DAEMON_PROTOCOL_VERSION;
        hello.app_version = Some("0.0.1".to_string());
        assert!(!is_current_daemon_hello(&hello));
    }

    #[test]
    fn session_summaries_are_detected_by_either_identity_field() {
        assert!(is_daemon_session_summary(
            &serde_json::json!({"activeSessionId": "a"})
        ));
        assert!(is_daemon_session_summary(&serde_json::json!({"id": "a"})));
        assert!(!is_daemon_session_summary(&serde_json::json!({"id": 5})));
        assert!(!is_daemon_session_summary(&serde_json::json!(null)));
    }

    #[test]
    fn busy_sessions_follow_the_roster_predicate() {
        assert!(is_session_busy(&SessionSummaryStub {
            is_session_active: true,
            has_running_rlm_children: None
        }));
        assert!(is_session_busy(&SessionSummaryStub {
            is_session_active: false,
            has_running_rlm_children: Some(true)
        }));
        assert!(!is_session_busy(&SessionSummaryStub {
            is_session_active: false,
            has_running_rlm_children: Some(false)
        }));
        assert!(!is_session_busy(&SessionSummaryStub {
            is_session_active: false,
            has_running_rlm_children: None
        }));
    }

    #[test]
    fn process_identity_exit_detection_handles_missing_and_live_pids() {
        assert!(has_process_identity_exited(None, true));
        assert!(has_process_identity_exited(
            Some(&DaemonProcessIdentity {
                pid: 2_000_000_000,
                process_start_id: None
            }),
            true
        ));
        let pid = std::process::id() as i64;
        assert!(!has_process_identity_exited(
            Some(&DaemonProcessIdentity {
                pid,
                process_start_id: None
            }),
            true
        ));
        if let Some(start_id) = get_process_start_id(pid) {
            assert!(!has_process_identity_exited(
                Some(&DaemonProcessIdentity {
                    pid,
                    process_start_id: Some(start_id.clone())
                }),
                true
            ));
            assert!(has_process_identity_exited(
                Some(&DaemonProcessIdentity {
                    pid,
                    process_start_id: Some("other".to_string())
                }),
                true
            ));
            // Start-id verification is skipped when the poll interval has not elapsed.
            assert!(!has_process_identity_exited(
                Some(&DaemonProcessIdentity {
                    pid,
                    process_start_id: Some("other".to_string())
                }),
                false
            ));
        }
    }

    #[test]
    fn stale_daemon_error_reports_both_identities() {
        let hello = DaemonHello {
            app_version: Some("0.1.0".to_string()),
            protocol_version: 6.0,
            schema_id: None,
            build_id: None,
            launcher_path: Some("/usr/local/bin/prime-agent".to_string()),
            entrypoint_path: None,
            executable_path: None,
            supervisor_pid: Some(4242),
            supervisor_process_start_id: None,
        };
        let error = StaleDaemonError::new("/tmp/daemon.sock", Some(&hello));
        assert!(error
            .message
            .starts_with("An incompatible Prime Agent daemon is running.\n\n"));
        assert!(error.message.contains("Daemon: v0.1.0, protocol 6, schema legacy, build unknown, PID 4242, executable /usr/local/bin/prime-agent"));
        assert!(error.message.contains(&format!("Client: v{}", VERSION)));
        assert!(error.message.ends_with("Then retry the original command."));
        let error = StaleDaemonError::new("/tmp/daemon.sock", None);
        assert!(error
            .message
            .contains("Daemon: unknown build on /tmp/daemon.sock"));
    }

    #[test]
    fn early_launch_is_skipped_for_excluded_flags_and_daemon_mode() {
        assert!(!should_start_daemon_early(
            &args(&["--mode", "daemon"]),
            false
        ));
        assert!(!should_start_daemon_early(&args(&["--help"]), false));
        assert!(!should_start_daemon_early(&args(&["--version"]), false));
        assert!(!should_start_daemon_early(&args(&["--list-models"]), false));
        assert!(!should_start_daemon_early(&args(&["--export"]), false));
        assert!(!should_start_daemon_early(&args(&["anything"]), true));
    }

    #[test]
    fn early_launch_runs_for_print_mode() {
        assert!(should_start_daemon_early(&args(&["-p", "hi"]), false));
        assert!(should_start_daemon_early(&args(&["--print"]), false));
    }

    #[test]
    fn early_launch_skips_public_and_removed_commands_but_keeps_agents() {
        assert!(!should_start_daemon_early(&args(&["list"]), false));
        assert!(!should_start_daemon_early(&args(&["daemon"]), false));
        assert!(!should_start_daemon_early(&args(&["help", "list"]), false));
        assert!(should_start_daemon_early(&args(&["agents"]), false));
        assert!(!should_start_daemon_early(&args(&["help"]), false));
        assert!(should_start_daemon_early(&args(&["hello world"]), false));
    }

    #[test]
    fn first_positional_skips_value_flags_and_resume() {
        assert_eq!(
            find_first_early_launch_positional(&args(&["--model", "m", "hello"])),
            Some((2, "hello".to_string()))
        );
        assert_eq!(
            find_first_early_launch_positional(&args(&["--resume", "abc"])),
            None
        );
        assert_eq!(
            find_first_early_launch_positional(&args(&["--resume", "abc", "hello"])),
            Some((2, "hello".to_string()))
        );
        assert_eq!(
            find_first_early_launch_positional(&args(&["--", "--dash-value"])),
            Some((1, "--dash-value".to_string()))
        );
        assert_eq!(find_first_early_launch_positional(&args(&["--"])), None);
    }

    #[test]
    fn daemon_log_tail_reads_only_new_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("daemon.sock").to_string_lossy().to_string();
        let log_path = get_daemon_log_path(&socket_path);
        if let Some(parent) = Path::new(&log_path).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&log_path, "old content").unwrap();
        let offset = "old content".len() as u64;
        std::fs::write(&log_path, "old content\nnew crash line").unwrap();
        let tail = read_daemon_log_tail(&socket_path, offset);
        assert!(tail.contains("new crash line"));
        assert!(!tail.contains("old content"));
        assert!(tail.starts_with(" Recent daemon log ("));
    }

    #[test]
    fn daemon_log_tail_reports_a_missing_log() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir
            .path()
            .join("missing.sock")
            .to_string_lossy()
            .to_string();
        let tail = read_daemon_log_tail(&socket_path, 0);
        assert!(tail.starts_with(" The daemon wrote nothing to its log ("));
    }

    #[test]
    fn daemon_log_tail_handles_a_shrunk_file() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir
            .path()
            .join("rotated.sock")
            .to_string_lossy()
            .to_string();
        let log_path = get_daemon_log_path(&socket_path);
        if let Some(parent) = Path::new(&log_path).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&log_path, "rotated").unwrap();
        let tail = read_daemon_log_tail(&socket_path, 4096);
        assert!(tail.contains("rotated"));
    }

    #[tokio::test]
    async fn probe_reports_absent_for_a_missing_socket() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("nope.sock").to_string_lossy().to_string();
        assert_eq!(
            probe_daemon_version(&socket_path, 200.0).await,
            DaemonVersionProbe::Absent
        );
        assert_eq!(
            probe_running_daemon_sessions(&socket_path).await,
            RunningDaemonProbe {
                reachable: false,
                active_sessions: None,
                busy_client_owned_session_count: None
            }
        );
    }

    #[tokio::test]
    async fn wait_for_daemon_gone_returns_true_without_a_socket() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("nope.sock").to_string_lossy().to_string();
        assert!(wait_for_daemon_gone(&socket_path, 200.0, false, None).await);
    }

    #[tokio::test]
    async fn wait_for_daemon_gone_waits_for_a_live_expected_process() {
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("nope.sock").to_string_lossy().to_string();
        let identity = DaemonProcessIdentity {
            pid: std::process::id() as i64,
            process_start_id: None,
        };
        assert!(!wait_for_daemon_gone(&socket_path, 100.0, false, Some(&identity)).await);
    }
    /// `Component::Prefix("C:")` followed by `Component::RootDir` is the single
    /// drive root "C:\". Returns the path once the host has produced that shape
    /// (None on a POSIX host, where "C:\..." has no Prefix component at all), so
    /// the assertions below cannot silently pass on a path that never reaches the
    /// RootDir arm.
    fn drive_absolute_input(label: &str, raw: &str) -> Option<PathBuf> {
        use std::path::Component;
        let path = Path::new(raw);
        if !matches!(path.components().next(), Some(Component::Prefix(_))) {
            return None;
        }
        assert!(
            path.components().any(|component| component == Component::RootDir),
            "{raw:?} has a drive prefix but no RootDir component ({label})"
        );
        Some(path.to_path_buf())
    }

    /// The defect this pins: the old `if prefix.is_empty()` guard skipped the
    /// root separator because `Prefix("C:")` had already filled `prefix`, so a
    /// drive-absolute path collapsed to a DRIVE-RELATIVE one - "C:Users/x/registry"
    /// instead of "C:/Users/x/registry", and a drive-rooted "/tmp/x" became
    /// "C:tmp/x". Every downstream create_dir_all / open / lockfile then failed
    /// with os error 3 ("The system cannot find the path specified").
    #[test]
    fn drive_roots_survive_the_lexical_collapse() {
        // Checked on every host: a rooted path keeps its root separator.
        let rooted = normalize_lexically(Path::new("/tmp/x"));
        assert!(rooted.starts_with('/'), "root separator dropped: {rooted}");

        let registry = match drive_absolute_input("registry", "C:\\Users\\x\\registry") {
            Some(path) => path,
            None => return,
        };
        let normalised = normalize_lexically(&registry);
        assert!(normalised.starts_with("C:/"), "drive root dropped: {normalised}");
        assert_eq!(normalised, "C:/Users/x/registry");

        // `Path::join` keeps the drive when it appends the rooted POSIX spelling,
        // which is exactly how a caller's resolve("/tmp/x") reaches this function.
        let joined = drive_absolute_input("joined /tmp/x", "C:\\base")
            .expect("C:\\base is drive-absolute")
            .join("/tmp/x");
        let normalised = normalize_lexically(&joined);
        assert!(!normalised.starts_with("C:tmp"), "/tmp/x became drive-relative: {normalised}");
        assert_eq!(normalised, "C:/tmp/x");

        let dotted = drive_absolute_input("parent collapse", "C:\\Users\\x\\..\\y")
            .expect("C:\\Users\\x\\..\\y is drive-absolute");
        assert_eq!(normalize_lexically(&dotted), "C:/Users/y");

        // Only the RootDir arm changed: a drive-RELATIVE input has no RootDir, so
        // it must still come back rootless instead of gaining a "C:/" root.
        let relative = Path::new("C:registry");
        if !relative.components().any(|component| component == std::path::Component::RootDir) {
            assert_eq!(normalize_lexically(relative), "C:registry");
        }
    }
}
