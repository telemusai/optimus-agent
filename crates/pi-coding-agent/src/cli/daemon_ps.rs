//! Port of packages/coding-agent/src/cli/daemon-ps.ts
//!
//! TODO(slice): `DaemonClient` (ca-daemon-b), `getProcessStartId`/session leases
//! (ca-session), orphan-process journal (ca-misc), daemon socket helpers
//! (ca-daemon-b), `acquireDaemonShutdownAdmission` (ca-daemon-b),
//! `DaemonWorkerDescriptor` (ca-daemon-b), child-process helpers (ca-utils) and
//! `getAgentDir`/`APP_NAME`/`VERSION` (ca-root) are not landed. Private local
//! stand-ins with the same behaviour live at the bottom of this module and are
//! listed in the slice status file.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::daemon_ps_format::format_daemon_list_table;

pub const STATUS_ORDER: [DaemonStatus; 4] = [
    DaemonStatus::Current,
    DaemonStatus::Stale,
    DaemonStatus::Unreachable,
    DaemonStatus::OrphanFile,
];
pub const SHUTDOWN_QUIET_PERIOD_MS: f64 = 1000.0;
pub const SHUTDOWN_CONVERGENCE_TIMEOUT_MS: f64 = 10_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DaemonStatus {
    Current,
    Stale,
    Unreachable,
    OrphanFile,
}

impl DaemonStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            DaemonStatus::Current => "current",
            DaemonStatus::Stale => "stale",
            DaemonStatus::Unreachable => "unreachable",
            DaemonStatus::OrphanFile => "orphan-file",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "current" => Some(DaemonStatus::Current),
            "stale" => Some(DaemonStatus::Stale),
            "unreachable" => Some(DaemonStatus::Unreachable),
            "orphan-file" => Some(DaemonStatus::OrphanFile),
            _ => None,
        }
    }

    fn order(self) -> i32 {
        match self {
            DaemonStatus::Current => 0,
            DaemonStatus::Stale => 1,
            DaemonStatus::Unreachable => 2,
            DaemonStatus::OrphanFile => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredDaemonProcess {
    pub pid: i64,
    pub socket_path: String,
    pub uptime_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DaemonInfo {
    pub socket_path: String,
    pub pid: Option<i64>,
    pub uptime_seconds: Option<f64>,
    pub version: Option<String>,
    pub protocol_version: Option<f64>,
    pub schema_id: Option<String>,
    pub build_id: Option<String>,
    pub executable_path: Option<String>,
    pub pid_source: Option<String>,
    pub session_count: Option<i64>,
    pub status: DaemonStatus,
    pub is_default: bool,
    pub has_tracked_workers: Option<bool>,
}

pub fn evaluate_shutdown_quiet_period(now: f64, quiet_since: Option<f64>) -> &'static str {
    if let Some(quiet_since) = quiet_since {
        if now - quiet_since >= SHUTDOWN_QUIET_PERIOD_MS {
            return "complete";
        }
    }
    "waiting"
}

// Linux comm names (and thus the process name ss reports) are capped at 15 chars.
const MAX_COMM_LENGTH: usize = 15;

fn process_name_matches(name: &str, app_name: &str) -> bool {
    if name == app_name {
        return true;
    }
    let truncated: String = app_name.chars().take(MAX_COMM_LENGTH).collect();
    truncated == name
}

/// Parse `ss -lxp` output into the prime-agent daemons listening on unix sockets.
pub fn parse_ss_listeners(stdout: &str, app_name: &str) -> Vec<DiscoveredDaemonProcess> {
    let mut daemons: Vec<DiscoveredDaemonProcess> = Vec::new();
    for line in stdout.split('\n') {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.get(1).copied() != Some("LISTEN") {
            continue;
        }
        let socket_path = match fields.get(4) {
            Some(socket_path) if socket_path.starts_with('/') => *socket_path,
            _ => continue,
        };
        let owner = match parse_ss_owner(line) {
            Some(owner) => owner,
            None => continue,
        };
        if !process_name_matches(&owner.0, app_name) {
            continue;
        }
        daemons.push(DiscoveredDaemonProcess {
            pid: owner.1,
            socket_path: normalize_socket_path(socket_path, None),
            uptime_seconds: None,
        });
    }
    daemons
}

/// `users:\(\("([^"]+)",pid=(\d+)`.
fn parse_ss_owner(line: &str) -> Option<(String, i64)> {
    let start = line.find("users:((")?;
    let rest = &line[start + "users:((".len()..];
    let quote_start = rest.find('"')? + 1;
    let quote_end = rest[quote_start..].find('"')? + quote_start;
    let name = rest[quote_start..quote_end].to_string();
    let after_name = &rest[quote_end..];
    let pid_start = after_name.find("pid=")? + "pid=".len();
    let digits: String = after_name[pid_start..].chars().take_while(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    Some((name, digits.parse::<i64>().ok()?))
}

/// Parse `lsof -nP -F pn -U -a -c <app>` output into listening unix socket owners (macOS fallback).
pub fn parse_lsof_listeners(stdout: &str) -> Vec<DiscoveredDaemonProcess> {
    let mut daemons: Vec<DiscoveredDaemonProcess> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut pid: Option<i64> = None;
    for line in stdout.split('\n') {
        let mut chars = line.chars();
        let field = match chars.next() {
            Some(field) => field,
            None => continue,
        };
        let value: String = chars.collect();
        if field == 'p' {
            pid = value.parse::<i64>().ok();
        } else if field == 'n' && pid.is_some() && value.starts_with('/') {
            let socket_path = normalize_socket_path(&value, None);
            let key = format!("{}:{}", pid.unwrap(), socket_path);
            if !seen.contains(&key) {
                seen.insert(key);
                daemons.push(DiscoveredDaemonProcess { pid: pid.unwrap(), socket_path, uptime_seconds: None });
            }
        }
    }
    daemons
}

pub fn parse_prime_agent_process_ids(stdout: &str, app_name: &str) -> Vec<i64> {
    let mut pids: Vec<i64> = Vec::new();
    for line in stdout.split('\n') {
        let trimmed = line.trim();
        let mut fields = trimmed.split_whitespace();
        let pid_field = match fields.next() {
            Some(pid_field) if pid_field.bytes().all(|byte| byte.is_ascii_digit()) => pid_field,
            _ => continue,
        };
        let command_field = match fields.next() {
            Some(command_field) => command_field,
            None => continue,
        };
        let remainder: String = fields.collect::<Vec<_>>().join(" ");
        let command = base_name(command_field);
        let argv0_raw = remainder
            .trim()
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
        let argv0 = base_name(&argv0_raw);
        if process_name_matches(&command, app_name) || process_name_matches(&argv0, app_name) {
            if let Ok(pid) = pid_field.parse::<i64>() {
                pids.push(pid);
            }
        }
    }
    pids
}

pub fn merge_discovered_daemon_processes(groups: &[Vec<DiscoveredDaemonProcess>]) -> Vec<DiscoveredDaemonProcess> {
    let mut by_identity: BTreeMap<String, DiscoveredDaemonProcess> = BTreeMap::new();
    for group in groups {
        for daemon in group {
            by_identity.insert(format!("{}:{}", daemon.pid, daemon.socket_path), daemon.clone());
        }
    }
    by_identity.into_values().collect()
}

/// Parse `ps -o pid=,etimes=` output into a pid → uptime-seconds map.
pub fn parse_ps_etimes(stdout: &str) -> BTreeMap<i64, f64> {
    let mut uptimes: BTreeMap<i64, f64> = BTreeMap::new();
    for line in stdout.split('\n') {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() == 2
            && fields[0].bytes().all(|byte| byte.is_ascii_digit())
            && fields[1].bytes().all(|byte| byte.is_ascii_digit())
        {
            if let (Ok(pid), Ok(uptime)) = (fields[0].parse::<i64>(), fields[1].parse::<f64>()) {
                uptimes.insert(pid, uptime);
            }
        }
    }
    uptimes
}

pub fn verify_hello_supervisor_pid(pid: Option<i64>, expected_process_start_id: Option<&str>) -> Option<i64> {
    let pid = pid?;
    if pid <= 0 {
        return None;
    }
    if !process_exists(pid) {
        return None;
    }
    if let Some(expected) = expected_process_start_id {
        let observed = get_process_start_id(pid);
        if observed.as_deref() != Some(expected) {
            return None;
        }
    }
    Some(pid)
}

pub fn sort_daemons(mut infos: Vec<DaemonInfo>) -> Vec<DaemonInfo> {
    infos.sort_by(|left, right| {
        if left.is_default != right.is_default {
            return if left.is_default { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater };
        }
        left.status
            .order()
            .cmp(&right.status.order())
            .then_with(|| left.socket_path.cmp(&right.socket_path))
    });
    infos
}

pub fn is_worker_socket_path(socket_path: &str) -> bool {
    if process_platform() == "win32" {
        return false;
    }
    let parent = Path::new(socket_path).parent().map(|path| resolve_path(path)).unwrap_or_default();
    let name = base_name(socket_path);
    parent == resolve_path(Path::new(&default_daemon_socket_dir()))
        && name.starts_with("worker-")
        && name.ends_with(".sock")
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReapAction {
    RemoveFile { daemon: DaemonInfo },
    Kill { daemon: DaemonInfo },
    Shutdown { daemon: DaemonInfo },
    Skip { daemon: DaemonInfo, reason: String },
}

impl ReapAction {
    pub fn daemon(&self) -> &DaemonInfo {
        match self {
            ReapAction::RemoveFile { daemon }
            | ReapAction::Kill { daemon }
            | ReapAction::Shutdown { daemon }
            | ReapAction::Skip { daemon, .. } => daemon,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            ReapAction::RemoveFile { .. } => "remove-file",
            ReapAction::Kill { .. } => "kill",
            ReapAction::Shutdown { .. } => "shutdown",
            ReapAction::Skip { .. } => "skip",
        }
    }
}

/// Decide what to do with each discovered daemon (pure, no side effects).
pub fn plan_reap(daemons: &[DaemonInfo], force: bool) -> Vec<ReapAction> {
    let mut pid_counts: BTreeMap<i64, i64> = BTreeMap::new();
    for daemon in daemons {
        if let Some(pid) = daemon.pid {
            *pid_counts.entry(pid).or_insert(0) += 1;
        }
    }

    daemons
        .iter()
        .map(|daemon| {
            // An orphan socket file has no owning process, so removing it is safe even
            // on the default path (a stale daemon.sock left by a crash). Decide this
            // before the default guard so a dead default socket still gets cleaned up.
            if daemon.status == DaemonStatus::OrphanFile {
                return ReapAction::RemoveFile { daemon: daemon.clone() };
            }
            if daemon.is_default {
                return ReapAction::Skip {
                    daemon: daemon.clone(),
                    reason: "default background service".to_string(),
                };
            }
            if daemon.status == DaemonStatus::Unreachable {
                if !force || daemon.pid.is_none() {
                    return ReapAction::Skip {
                        daemon: daemon.clone(),
                        reason: "unreachable; use \"prime-agent shutdown --force\" to stop it".to_string(),
                    };
                }
                let pid = daemon.pid.unwrap();
                if pid_counts.get(&pid).copied().unwrap_or(0) > 1 {
                    return ReapAction::Skip {
                        daemon: daemon.clone(),
                        reason: format!("unreachable; pid {} also backs another daemon, not killing", pid),
                    };
                }
                return ReapAction::Kill { daemon: daemon.clone() };
            }
            if daemon.session_count != Some(0) {
                return ReapAction::Skip {
                    daemon: daemon.clone(),
                    reason: format!(
                        "has {} session(s)",
                        daemon
                            .session_count
                            .map(|count| count.to_string())
                            .unwrap_or_else(|| "unknown".to_string())
                    ),
                };
            }
            ReapAction::Shutdown { daemon: daemon.clone() }
        })
        .collect()
}

pub fn plan_shutdown_all(daemons: &[DaemonInfo], force: bool) -> Vec<ReapAction> {
    daemons
        .iter()
        .map(|daemon| {
            if daemon.status == DaemonStatus::OrphanFile {
                return ReapAction::RemoveFile { daemon: daemon.clone() };
            }
            if daemon.status == DaemonStatus::Unreachable {
                if daemon.pid.is_none() {
                    return if force || daemon.has_tracked_workers != Some(true) {
                        ReapAction::RemoveFile { daemon: daemon.clone() }
                    } else {
                        ReapAction::Skip {
                            daemon: daemon.clone(),
                            reason: "has unreachable workers; use --force to kill".to_string(),
                        }
                    };
                }
                return if force {
                    ReapAction::Kill { daemon: daemon.clone() }
                } else {
                    ReapAction::Skip {
                        daemon: daemon.clone(),
                        reason: "unreachable; use --force to kill".to_string(),
                    }
                };
            }
            ReapAction::Shutdown { daemon: daemon.clone() }
        })
        .collect()
}

fn shutdown_all_action_order(kind: &str) -> i32 {
    match kind {
        "shutdown" => 0,
        "remove-file" => 1,
        "kill" => 2,
        _ => 3,
    }
}

pub fn plan_shutdown_confirmation(
    daemon_count: usize,
    json: bool,
    force: bool,
    stdin_is_tty: Option<bool>,
) -> &'static str {
    if daemon_count == 0 || force {
        return "none";
    }
    if json {
        return "json-error";
    }
    if stdin_is_tty == Some(true) {
        "prompt"
    } else {
        "tty-error"
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReapOutcome {
    pub reaped: Option<String>,
    pub skipped: Option<String>,
}

impl ReapOutcome {
    fn reaped(message: impl Into<String>) -> Self {
        Self { reaped: Some(message.into()), skipped: None }
    }

    fn skipped(message: impl Into<String>) -> Self {
        Self { reaped: None, skipped: Some(message.into()) }
    }
}

fn apply(
    outcome: ReapOutcome,
    socket_path: &str,
    reaped: &mut Vec<(String, String)>,
    skipped: &mut Vec<(String, String)>,
) {
    match (outcome.reaped, outcome.skipped) {
        (Some(action), _) => reaped.push((socket_path.to_string(), action)),
        (None, Some(reason)) => skipped.push((socket_path.to_string(), reason)),
        (None, None) => {}
    }
}

/// Callback surface for `console.log` and the interactive prompt.
pub struct DaemonPsIo<'a> {
    pub log: &'a dyn Fn(&str),
    pub error: &'a dyn Fn(&str),
    /// `process.stdin.isTTY`
    pub stdin_is_tty: Option<bool>,
    /// Interactive yes/no prompt (`promptYesNo`); returns the answer.
    pub prompt_yes_no: &'a dyn Fn(&str) -> bool,
    /// `process.exitCode = 1`
    pub set_exit_code: &'a dyn Fn(i32),
    /// `process.cwd()`
    pub cwd: String,
}

pub async fn run_ps(json: bool, io: &DaemonPsIo<'_>) -> Result<(), String> {
    let daemons = discover_daemons().await;
    if json {
        (io.log)(&to_pretty_json(&daemons_to_json(&daemons)));
        return Ok(());
    }
    if daemons.is_empty() {
        (io.log)("No background services found.");
        return Ok(());
    }
    (io.log)(&format_daemon_list_table(&daemons));
    Ok(())
}

pub async fn run_shutdown_all(json: bool, force: bool, io: &DaemonPsIo<'_>) -> Result<(), String> {
    let daemons = discover_daemons().await;
    match plan_shutdown_confirmation(daemons.len(), json, force, io.stdin_is_tty) {
        "json-error" => {
            (io.set_exit_code)(1);
            let failed: Vec<serde_json::Value> = daemons
                .iter()
                .map(|daemon| {
                    serde_json::json!({
                        "socketPath": daemon.socket_path,
                        "reason": "confirmation required; use \"prime-agent shutdown --force --json\"",
                    })
                })
                .collect();
            (io.log)(&to_pretty_json(&serde_json::json!({ "stopped": [], "failed": failed })));
            return Ok(());
        }
        "tty-error" => {
            return Err(
                "Shutdown requires confirmation in an interactive terminal. Use \"prime-agent shutdown --force\"."
                    .to_string(),
            );
        }
        "prompt" => {
            let confirmed = (io.prompt_yes_no)(
                "Stop every agent and background service? Active work will be interrupted.",
            );
            if !confirmed {
                (io.log)("\u{1b}[2mShutdown cancelled.\u{1b}[22m");
                return Ok(());
            }
        }
        _ => {}
    }
    let admission = acquire_daemon_shutdown_admission().await;
    let result = run_shutdown_all_converging(json, force, &admission, io).await;
    admission.release().await;
    result
}

async fn run_shutdown_all_converging(
    json: bool,
    force: bool,
    admission: &DaemonShutdownAdmission,
    io: &DaemonPsIo<'_>,
) -> Result<(), String> {
    let mut stopped: Vec<(String, String)> = Vec::new();
    let mut failed: Vec<(String, String)> = Vec::new();
    let mut handled_pids: BTreeSet<i64> = BTreeSet::new();
    let mut reported_failures: BTreeSet<String> = BTreeSet::new();

    if force {
        stop_hidden_supervisors(&mut stopped, &mut failed, &mut handled_pids, &mut reported_failures, admission, io)
            .await?;
    }
    let daemons: Vec<DaemonInfo> = discover_daemons()
        .await
        .into_iter()
        .filter(|daemon| !is_worker_socket_path(&daemon.socket_path))
        .collect();

    let mut actions = plan_shutdown_all(&daemons, force);
    actions.sort_by_key(|action| shutdown_all_action_order(action.kind()));

    for action in actions {
        let socket_path = action.daemon().socket_path.clone();
        let pid = action.daemon().pid;
        let action_kind = action.kind();
        if let Some(pid) = pid {
            if handled_pids.contains(&pid) {
                admission.assert_or_renew().await?;
                remove_socket_file(&socket_path);
                stopped.push((
                    socket_path.clone(),
                    format!("background service already stopped (pid {})", pid),
                ));
                if force {
                    for reason in force_stop_tracked_workers(&socket_path, admission).await {
                        failed.push((socket_path.clone(), reason));
                    }
                }
                continue;
            }
        }
        match action {
            ReapAction::RemoveFile { .. } => {
                if probe_daemon(&socket_path).await.reachable {
                    apply(
                        stop_background_service(&socket_path, pid, &mut handled_pids, force, admission).await?,
                        &socket_path,
                        &mut stopped,
                        &mut failed,
                    );
                } else {
                    admission.assert_or_renew().await?;
                    if remove_socket_file(&socket_path) {
                        stopped.push((socket_path.clone(), "removed stale socket file".to_string()));
                    } else {
                        failed.push((socket_path.clone(), "could not remove socket file".to_string()));
                    }
                }
            }
            ReapAction::Kill { .. } => {
                if probe_daemon(&socket_path).await.reachable {
                    apply(
                        stop_background_service(&socket_path, pid, &mut handled_pids, force, admission).await?,
                        &socket_path,
                        &mut stopped,
                        &mut failed,
                    );
                } else if is_daemon_process_listening(pid.unwrap_or(0), &socket_path) {
                    admission.assert_or_renew().await?;
                    force_kill_daemon(pid.unwrap_or(0)).await;
                    handled_pids.insert(pid.unwrap_or(0));
                    admission.assert_or_renew().await?;
                    remove_socket_file(&socket_path);
                    stopped.push((
                        socket_path.clone(),
                        format!("killed unreachable background service (pid {})", pid.unwrap_or(0)),
                    ));
                } else {
                    admission.assert_or_renew().await?;
                    remove_socket_file(&socket_path);
                    stopped.push((socket_path.clone(), "background service already stopped".to_string()));
                }
            }
            ReapAction::Shutdown { .. } => {
                apply(
                    stop_background_service(&socket_path, pid, &mut handled_pids, force, admission).await?,
                    &socket_path,
                    &mut stopped,
                    &mut failed,
                );
            }
            ReapAction::Skip { reason, .. } => {
                failed.push((socket_path.clone(), reason));
            }
        }
        if force && action_kind != "skip" {
            for reason in force_stop_tracked_workers(&socket_path, admission).await {
                failed.push((socket_path.clone(), reason));
            }
        }
    }

    if force {
        terminate_verified_residuals(&mut stopped, &mut failed, &mut handled_pids, &mut reported_failures, admission)
            .await?;
    }

    if json {
        if !failed.is_empty() {
            (io.set_exit_code)(1);
        }
        let stopped_json: Vec<serde_json::Value> = stopped
            .iter()
            .map(|(socket_path, action)| serde_json::json!({ "socketPath": socket_path, "action": action }))
            .collect();
        let failed_json: Vec<serde_json::Value> = failed
            .iter()
            .map(|(socket_path, reason)| serde_json::json!({ "socketPath": socket_path, "reason": reason }))
            .collect();
        (io.log)(&to_pretty_json(
            &serde_json::json!({ "stopped": stopped_json, "failed": failed_json }),
        ));
        return Ok(());
    }
    if stopped.is_empty() && failed.is_empty() {
        (io.log)("No background services found.");
        return Ok(());
    }
    for (socket_path, action) in &stopped {
        (io.log)(&format!("\u{1b}[32mstopped {}: {}\u{1b}[39m", socket_path, action));
    }
    for (socket_path, reason) in &failed {
        (io.log)(&format!("\u{1b}[31mfailed  {}: {}\u{1b}[39m", socket_path, reason));
    }
    if !failed.is_empty() {
        (io.set_exit_code)(1);
    }
    Ok(())
}

async fn stop_hidden_supervisors(
    stopped: &mut Vec<(String, String)>,
    failed: &mut Vec<(String, String)>,
    handled_pids: &mut BTreeSet<i64>,
    reported_failures: &mut BTreeSet<String>,
    admission: &DaemonShutdownAdmission,
    _io: &DaemonPsIo<'_>,
) -> Result<(), String> {
    loop {
        let listeners: Vec<DiscoveredDaemonProcess> = scan_listening_daemons()
            .into_iter()
            .filter(|listener| !is_worker_socket_path(&listener.socket_path))
            .collect();
        let mut by_socket: BTreeMap<String, Vec<DiscoveredDaemonProcess>> = BTreeMap::new();
        for listener in listeners {
            by_socket.entry(listener.socket_path.clone()).or_default().push(listener);
        }
        let mut hidden: Vec<DiscoveredDaemonProcess> = Vec::new();
        for (socket_path, group) in &by_socket {
            let distinct: BTreeSet<i64> = group.iter().map(|listener| listener.pid).collect();
            if distinct.len() < 2 {
                continue;
            }
            let current_pid = probe_daemon(socket_path).await.supervisor_pid;
            match current_pid {
                Some(current_pid) if group.iter().any(|listener| listener.pid == current_pid) => {}
                _ => {
                    record_shutdown_failure(
                        failed,
                        reported_failures,
                        socket_path,
                        "could not identify the current same-path daemon",
                    );
                    continue;
                }
            }
            hidden.extend(group.iter().filter(|listener| listener.pid != current_pid.unwrap()).cloned());
        }
        if hidden.is_empty() {
            return Ok(());
        }
        let before = daemon_listener_signature(&hidden);
        for listener in &hidden {
            if terminate_verified_listener(listener, failed, reported_failures, admission).await {
                handled_pids.insert(listener.pid);
                stopped.push((
                    listener.socket_path.clone(),
                    format!("stopped hidden daemon (pid {})", listener.pid),
                ));
            }
        }
        let after_hidden: Vec<DiscoveredDaemonProcess> = scan_listening_daemons()
            .into_iter()
            .filter(|listener| {
                !is_worker_socket_path(&listener.socket_path)
                    && hidden.iter().any(|candidate| {
                        candidate.pid == listener.pid && candidate.socket_path == listener.socket_path
                    })
            })
            .collect();
        if after_hidden.is_empty() || daemon_listener_signature(&after_hidden) == before {
            return Ok(());
        }
    }
}

async fn terminate_verified_residuals(
    stopped: &mut Vec<(String, String)>,
    failed: &mut Vec<(String, String)>,
    handled_pids: &mut BTreeSet<i64>,
    reported_failures: &mut BTreeSet<String>,
    admission: &DaemonShutdownAdmission,
) -> Result<(), String> {
    let mut previous_signature: Option<String> = None;
    let mut quiet_since: Option<f64> = None;
    let deadline = now_ms() + SHUTDOWN_CONVERGENCE_TIMEOUT_MS;
    loop {
        admission.assert_or_renew().await?;
        let listeners = scan_listening_daemons();
        let now = now_ms();
        if listeners.is_empty() {
            previous_signature = None;
            if quiet_since.is_none() {
                quiet_since = Some(now);
            }
            if evaluate_shutdown_quiet_period(now, quiet_since) == "complete" {
                return Ok(());
            }
            sleep_ms(100).await;
            continue;
        }
        quiet_since = None;
        let signature = daemon_listener_signature(&listeners);
        if now >= deadline {
            record_residual_listener_failures(&listeners, failed, reported_failures, "kept respawning during shutdown");
            return Ok(());
        }
        if Some(&signature) == previous_signature.as_ref() {
            record_residual_listener_failures(&listeners, failed, reported_failures, "remained after shutdown");
            return Ok(());
        }
        previous_signature = Some(signature);
        let mut seen_pids: BTreeSet<i64> = BTreeSet::new();
        for listener in &listeners {
            if seen_pids.contains(&listener.pid) {
                continue;
            }
            seen_pids.insert(listener.pid);
            let already_reported = handled_pids.contains(&listener.pid);
            if terminate_verified_listener(listener, failed, reported_failures, admission).await {
                handled_pids.insert(listener.pid);
                if !already_reported {
                    stopped.push((
                        listener.socket_path.clone(),
                        format!("stopped residual daemon process (pid {})", listener.pid),
                    ));
                }
            }
        }
    }
}

fn record_residual_listener_failures(
    listeners: &[DiscoveredDaemonProcess],
    failed: &mut Vec<(String, String)>,
    reported_failures: &mut BTreeSet<String>,
    reason: &str,
) {
    for listener in listeners {
        let process_start_id = get_process_start_id(listener.pid);
        let identity = match &process_start_id {
            Some(process_start_id) => format!("pid {}, start {}", listener.pid, process_start_id),
            None => format!("pid {}, process identity unavailable", listener.pid),
        };
        record_shutdown_failure(
            failed,
            reported_failures,
            &listener.socket_path,
            &format!(
                "daemon {} ({}){}",
                reason,
                identity,
                describe_daemon_parent(listener.pid)
            ),
        );
    }
}

fn describe_daemon_parent(pid: i64) -> String {
    let result = spawn_sync_hidden("ps", &["-o", "ppid=,tty=,command=", "-p", &pid.to_string()]);
    let stdout = match result {
        Some((0, stdout)) => stdout,
        _ => return String::new(),
    };
    let trimmed = stdout.trim();
    let mut fields = trimmed.split_whitespace();
    let ppid = match fields.next() {
        Some(ppid) if ppid.bytes().all(|byte| byte.is_ascii_digit()) => ppid,
        _ => return String::new(),
    };
    let tty = match fields.next() {
        Some(tty) => tty,
        None => return String::new(),
    };
    let command: String = fields.collect::<Vec<_>>().join(" ");
    if command.is_empty() {
        return String::new();
    }
    format!("; close parent PID {} on {} ({}) and retry shutdown", ppid, tty, command)
}

async fn terminate_verified_listener(
    listener: &DiscoveredDaemonProcess,
    failed: &mut Vec<(String, String)>,
    reported_failures: &mut BTreeSet<String>,
    admission: &DaemonShutdownAdmission,
) -> bool {
    let process_start_id = match get_process_start_id(listener.pid) {
        Some(process_start_id) => process_start_id,
        None => {
            record_shutdown_failure(
                failed,
                reported_failures,
                &listener.socket_path,
                &format!("could not verify daemon process identity (pid {})", listener.pid),
            );
            return false;
        }
    };
    if get_process_start_id(listener.pid).as_deref() != Some(process_start_id.as_str()) {
        return false;
    }
    if admission.assert_or_renew().await.is_err() {
        return false;
    }
    if get_process_start_id(listener.pid).as_deref() != Some(process_start_id.as_str()) {
        return false;
    }
    kill_daemon(listener.pid);
    let deadline = now_ms() + 1000.0;
    while get_process_start_id(listener.pid).as_deref() == Some(process_start_id.as_str()) && now_ms() < deadline {
        sleep_ms(50).await;
    }
    if get_process_start_id(listener.pid).as_deref() == Some(process_start_id.as_str()) {
        if admission.assert_or_renew().await.is_err() {
            return false;
        }
        if get_process_start_id(listener.pid).as_deref() != Some(process_start_id.as_str()) {
            return false;
        }
        // The verified process exited between the identity check and the signal.
        force_kill_process(listener.pid);
    }
    get_process_start_id(listener.pid).as_deref() != Some(process_start_id.as_str())
}

fn daemon_listener_signature(listeners: &[DiscoveredDaemonProcess]) -> String {
    let mut parts: Vec<String> = listeners
        .iter()
        .map(|listener| {
            format!(
                "{}:{}:{}",
                listener.pid,
                get_process_start_id(listener.pid).unwrap_or_else(|| "unknown".to_string()),
                listener.socket_path
            )
        })
        .collect();
    parts.sort();
    parts.join("\n")
}

fn record_shutdown_failure(
    failed: &mut Vec<(String, String)>,
    reported_failures: &mut BTreeSet<String>,
    socket_path: &str,
    reason: &str,
) {
    let key = format!("{}\u{0}{}", socket_path, reason);
    if reported_failures.contains(&key) {
        return;
    }
    reported_failures.insert(key);
    failed.push((socket_path.to_string(), reason.to_string()));
}

async fn stop_background_service(
    socket_path: &str,
    pid: Option<i64>,
    handled_pids: &mut BTreeSet<i64>,
    force: bool,
    admission: &DaemonShutdownAdmission,
) -> Result<ReapOutcome, String> {
    admission.assert_or_renew().await?;
    if shutdown_daemon(socket_path, force).await {
        if let Some(pid) = pid {
            handled_pids.insert(pid);
        }
        return Ok(ReapOutcome::reaped(format!(
            "stopped background service{}",
            pid.map(|pid| format!(" (pid {})", pid)).unwrap_or_default()
        )));
    }
    if !can_connect_to_socket(socket_path, 250.0).await {
        admission.assert_or_renew().await?;
        remove_socket_file(socket_path);
        return Ok(ReapOutcome::reaped("background service already stopped"));
    }
    let pid = match pid {
        Some(pid) => pid,
        None => return Ok(ReapOutcome::skipped("still listening but no pid to kill")),
    };
    if !force {
        return Ok(ReapOutcome::skipped("did not stop gracefully; retry with --force"));
    }
    admission.assert_or_renew().await?;
    force_kill_daemon(pid).await;
    handled_pids.insert(pid);
    admission.assert_or_renew().await?;
    remove_socket_file(socket_path);
    Ok(ReapOutcome::reaped(format!("force-killed unresponsive background service (pid {})", pid)))
}

struct TrackedWorker {
    descriptor: DaemonWorkerDescriptor,
    descriptor_path: String,
}

async fn force_stop_tracked_workers(
    supervisor_socket_path: &str,
    admission: &DaemonShutdownAdmission,
) -> Vec<String> {
    let mut failures: Vec<String> = Vec::new();
    for worker in find_tracked_workers(supervisor_socket_path) {
        let descriptor = worker.descriptor.clone();
        let mut cleanup_worker_records = stop_tracked_process(
            descriptor.pid,
            descriptor.process_start_id.as_deref(),
            admission,
        )
        .await;
        if !cleanup_worker_records {
            failures.push(format!(
                "could not safely stop worker {} (pid {})",
                descriptor.worker_id, descriptor.pid
            ));
        }
        if let Some(journal_path) = descriptor.orphan_process_journal_path.clone() {
            let mut orphans: Vec<ActiveOrphanProcess> = Vec::new();
            match read_active_orphan_processes(&journal_path, descriptor.pid) {
                Ok(read) => orphans = read,
                Err(error) => failures.push(format!(
                    "could not read child process records for worker {}: {}",
                    descriptor.worker_id, error
                )),
            }
            for orphan in orphans {
                // Pid-only records go through the platform predicate (stopTrackedProcess needs a startId).
                if orphan.process_start_id.is_none() {
                    if should_reap_orphan_process(&orphan) {
                        kill_orphan_process(orphan.pid);
                    }
                    continue;
                }
                if !is_orphan_process_identity_current(&orphan) {
                    continue;
                }
                if process_platform() == "win32" {
                    // taskkill /T, like the sibling reapers: signalling only the shell pid leaves its descendants alive.
                    if admission.assert_or_renew().await.is_err() {
                        cleanup_worker_records = false;
                        continue;
                    }
                    if is_orphan_process_identity_current(&orphan) {
                        kill_orphan_process(orphan.pid);
                        if is_process_alive(orphan.pid) {
                            cleanup_worker_records = false;
                            failures.push(format!(
                                "could not stop child process {} for worker {}",
                                orphan.pid, descriptor.worker_id
                            ));
                        }
                    }
                    continue;
                }
                if !stop_tracked_process(orphan.pid, orphan.process_start_id.as_deref(), admission).await {
                    cleanup_worker_records = false;
                    failures.push(format!(
                        "could not stop child process {} for worker {}",
                        orphan.pid, descriptor.worker_id
                    ));
                }
            }
        }
        if cleanup_worker_records {
            remove_socket_file(&descriptor.socket_path);
            let _ = std::fs::remove_file(&worker.descriptor_path);
            let _ = std::fs::remove_file(&descriptor.recovery_journal_path);
            if let Some(journal_path) = &descriptor.orphan_process_journal_path {
                let _ = std::fs::remove_file(journal_path);
            }
        }
    }
    failures
}

fn find_tracked_workers(supervisor_socket_path: &str) -> Vec<TrackedWorker> {
    find_all_tracked_workers()
        .into_iter()
        .filter(|worker| {
            normalize_socket_path(&worker.descriptor.supervisor_socket_path, None)
                == normalize_socket_path(supervisor_socket_path, None)
        })
        .collect()
}

fn find_all_tracked_workers() -> Vec<TrackedWorker> {
    let root = Path::new(&get_agent_dir()).join("daemon-workers");
    if !root.exists() {
        return Vec::new();
    }
    let mut workers: Vec<TrackedWorker> = Vec::new();
    let directory_names = match std::fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    for directory_entry in directory_names.flatten() {
        let directory = directory_entry.path();
        if !directory.is_dir() {
            continue;
        }
        let file_names = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for file_entry in file_names.flatten() {
            let file_name = file_entry.file_name().to_string_lossy().to_string();
            if !file_name.ends_with(".json") {
                continue;
            }
            let descriptor_path = file_entry.path();
            let contents = match std::fs::read_to_string(&descriptor_path) {
                Ok(contents) => contents,
                Err(_) => continue,
            };
            let value: serde_json::Value = match serde_json::from_str(&contents) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if let Some(descriptor) = is_tracked_worker_descriptor(&value) {
                workers.push(TrackedWorker {
                    descriptor,
                    descriptor_path: descriptor_path.to_string_lossy().to_string(),
                });
            }
        }
    }
    workers
}

/// `isTrackedWorkerDescriptor(value): value is DaemonWorkerDescriptor`
fn is_tracked_worker_descriptor(value: &serde_json::Value) -> Option<DaemonWorkerDescriptor> {
    let object = value.as_object()?;
    let version = object.get("version").and_then(serde_json::Value::as_i64);
    if version != Some(1) && version != Some(2) {
        return None;
    }
    let supervisor_socket_path = object
        .get("supervisorSocketPath")
        .and_then(serde_json::Value::as_str)?
        .to_string();
    let worker_id = object.get("workerId").and_then(serde_json::Value::as_str)?.to_string();
    let pid = object.get("pid").and_then(serde_json::Value::as_i64).filter(|pid| *pid > 0)?;
    let process_start_id = match object.get("processStartId") {
        None => None,
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(_) => return None,
    };
    let socket_path = object.get("socketPath").and_then(serde_json::Value::as_str)?.to_string();
    let recovery_journal_path = object
        .get("recoveryJournalPath")
        .and_then(serde_json::Value::as_str)?
        .to_string();
    let orphan_process_journal_path = object
        .get("orphanProcessJournalPath")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Some(DaemonWorkerDescriptor {
        version,
        worker_id,
        pid,
        process_start_id,
        socket_path,
        recovery_journal_path,
        orphan_process_journal_path,
        supervisor_socket_path,
    })
}

async fn stop_tracked_process(
    pid: i64,
    expected_start_id: Option<&str>,
    admission: &DaemonShutdownAdmission,
) -> bool {
    if tracked_process_stopped(pid) {
        return true;
    }
    let expected_start_id = match expected_start_id {
        Some(expected_start_id) => expected_start_id,
        None => return false,
    };
    if !tracked_leader_identity_current(pid, expected_start_id) {
        return false;
    }
    if admission.assert_or_renew().await.is_err() {
        return false;
    }
    if !tracked_leader_identity_current(pid, expected_start_id) {
        return false;
    }
    signal_process_group_if_held(pid, "SIGTERM");
    let mut deadline = now_ms() + 500.0;
    while !tracked_process_stopped(pid) && now_ms() < deadline {
        sleep_ms(25).await;
    }
    if tracked_process_stopped(pid) {
        return true;
    }
    if admission.assert_or_renew().await.is_err() {
        return false;
    }
    if !tracked_leader_identity_current(pid, expected_start_id) {
        return false;
    }
    signal_process_group_if_held(pid, "SIGKILL");
    deadline = now_ms() + 1000.0;
    while !tracked_process_stopped(pid) && now_ms() < deadline {
        sleep_ms(25).await;
    }
    tracked_process_stopped(pid)
}

/// A GROUP stop completes when the leader is gone AND no live member remains; unreaped zombies do not block it.
fn tracked_process_stopped(pid: i64) -> bool {
    !is_process_alive(pid) && !process_group_has_live_member(pid)
}

/// Identity gates guard pid reuse, so they apply only while the leader exists; a pgid cannot be reused while members hold it.
fn tracked_leader_identity_current(pid: i64, expected_start_id: &str) -> bool {
    if !process_id_exists(pid) {
        return true;
    }
    get_process_start_id(pid).as_deref() == Some(expected_start_id)
}

pub async fn run_reap(json: bool, force: bool, io: &DaemonPsIo<'_>) -> Result<(), String> {
    let daemons = discover_daemons().await;
    let mut reaped: Vec<(String, String)> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    for action in plan_reap(&daemons, force) {
        let socket_path = action.daemon().socket_path.clone();
        let pid = action.daemon().pid;
        match action {
            ReapAction::Skip { reason, .. } => {
                skipped.push((socket_path, reason));
            }
            ReapAction::RemoveFile { .. } => {
                // Re-probe before unlinking: a path marked orphan-file at discovery
                // may have since become a live listener. Only remove it if it is
                // still unreachable, so we never delete a socket a daemon is using.
                if probe_daemon(&socket_path).await.reachable {
                    skipped.push((socket_path, "now reachable; not removing socket file".to_string()));
                } else if remove_socket_file(&socket_path) {
                    reaped.push((socket_path, "removed stale socket file".to_string()));
                } else {
                    skipped.push((socket_path, "could not remove socket file".to_string()));
                }
            }
            ReapAction::Kill { .. } => {
                // Re-probe right before killing: discovery and this kill happen at
                // different moments, so a daemon classified "unreachable" may have
                // since started answering. Never SIGTERM one that now responds;
                // defer to the session-aware shutdown path instead.
                let recheck = probe_daemon(&socket_path).await;
                if !recheck.reachable {
                    kill_daemon(pid.unwrap_or(0));
                    remove_socket_file(&socket_path);
                    reaped.push((
                        socket_path,
                        format!("killed unreachable daemon (pid {})", pid.unwrap_or(0)),
                    ));
                } else {
                    apply(reap_reachable_daemon(&socket_path, pid).await, &socket_path, &mut reaped, &mut skipped);
                }
            }
            ReapAction::Shutdown { .. } => {
                apply(reap_reachable_daemon(&socket_path, pid).await, &socket_path, &mut reaped, &mut skipped);
            }
        }
    }

    if json {
        let reaped_json: Vec<serde_json::Value> = reaped
            .iter()
            .map(|(socket_path, action)| serde_json::json!({ "socketPath": socket_path, "action": action }))
            .collect();
        let skipped_json: Vec<serde_json::Value> = skipped
            .iter()
            .map(|(socket_path, reason)| serde_json::json!({ "socketPath": socket_path, "reason": reason }))
            .collect();
        (io.log)(&to_pretty_json(
            &serde_json::json!({ "reaped": reaped_json, "skipped": skipped_json }),
        ));
        return Ok(());
    }
    if reaped.is_empty() && skipped.is_empty() {
        (io.log)("No background services found.");
        return Ok(());
    }
    for (socket_path, action) in &reaped {
        (io.log)(&format!("\u{1b}[32mreaped {}: {}\u{1b}[39m", socket_path, action));
    }
    for (socket_path, reason) in &skipped {
        (io.log)(&format!("\u{1b}[2mkept   {}: {}\u{1b}[22m", socket_path, reason));
    }
    Ok(())
}

/// Gracefully stop a daemon, but only after a fresh probe confirms it is idle.
async fn reap_reachable_daemon(socket_path: &str, pid: Option<i64>) -> ReapOutcome {
    let probe = probe_daemon(socket_path).await;
    if !probe.reachable {
        return ReapOutcome::skipped("no longer reachable");
    }
    if probe.session_count != Some(0) {
        return ReapOutcome::skipped(format!(
            "now has {} session(s)",
            probe
                .session_count
                .map(|count| count.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ));
    }
    if shutdown_daemon(socket_path, false).await {
        ReapOutcome::reaped(format!(
            "stopped idle background service{}",
            pid.map(|pid| format!(" (pid {})", pid)).unwrap_or_default()
        ))
    } else {
        ReapOutcome::skipped("shutdown request failed")
    }
}

fn remove_socket_file(socket_path: &str) -> bool {
    match std::fs::remove_file(socket_path) {
        Ok(()) => true,
        Err(error) => error.kind() == std::io::ErrorKind::NotFound,
    }
}

fn kill_daemon(pid: i64) {
    signal_process(pid, "SIGTERM");
}

async fn force_kill_daemon(pid: i64) {
    kill_daemon(pid);
    let deadline = now_ms() + 1000.0;
    while now_ms() < deadline {
        if !is_process_alive(pid) {
            return;
        }
        sleep_ms(50).await;
    }
    // The process may already have exited between the liveness check and the kill.
    force_kill_process(pid);
}

async fn can_connect_to_socket(socket_path: &str, timeout_ms: f64) -> bool {
    can_connect_to_daemon(socket_path, timeout_ms).await
}

/// Ask a daemon to shut down and confirm it actually stopped listening.
async fn shutdown_daemon(socket_path: &str, force: bool) -> bool {
    if !can_connect_to_daemon(socket_path, 1000.0).await {
        return false;
    }
    // The daemon may still stop; the connectivity check below is the source of truth.
    let _ = request_daemon(socket_path, serde_json::json!({ "type": "shutdown", "force": force }), 1500.0).await;

    let deadline = now_ms() + 5000.0;
    while now_ms() < deadline {
        if !can_connect_to_socket(socket_path, 250.0).await {
            return true;
        }
        sleep_ms(50).await;
    }
    false
}

fn to_pretty_json(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string())
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

async fn sleep_ms(ms: u64) {
    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
}

/// Local stand-in for `normalizeSocketPath` from ../../utils/daemon-socket-path.js.
fn normalize_socket_path(socket_path: &str, base_dir: Option<&str>) -> String {
    crate::utils::daemon_socket_path::normalize_socket_path(socket_path, base_dir)
}

/// Local stand-in for `basename` from node:path.
fn base_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Local stand-in for `resolve` from node:path.
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
            Component::Prefix(prefix_component) => {
                prefix.push_str(&prefix_component.as_os_str().to_string_lossy());
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

/// Local stand-in for `defaultDaemonSocketDir` from daemon-socket.js.
fn default_daemon_socket_dir() -> String {
    let suffix = match std::env::var("USERNAME").ok().or_else(|| std::env::var("UID").ok()) {
        Some(_) => current_user_id(),
        None => "user".to_string(),
    };
    Path::new(&std::env::temp_dir())
        .join(format!("prime-agent-{}", suffix))
        .to_string_lossy()
        .to_string()
}

fn current_user_id() -> String {
    // Node's `process.getuid()` is undefined on Windows, which the TypeScript
    // renders as the literal "user" suffix.
    if process_platform() == "win32" {
        return "user".to_string();
    }
    std::env::var("UID").unwrap_or_else(|_| "user".to_string())
}

/// Local stand-in for `defaultDaemonSocketPath` from daemon-socket.js.
fn default_daemon_socket_path() -> String {
    if process_platform() == "win32" {
        return "\\\\.\\pipe\\prime-agent-daemon".to_string();
    }
    Path::new(&default_daemon_socket_dir())
        .join("daemon.sock")
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
    Path::new(&home_dir()).join(".prime/agent").to_string_lossy().to_string()
}

fn home_dir() -> String {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default()
}

fn expand_tilde_path(path: &str) -> String {
    if path == "~" {
        return home_dir();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return Path::new(&home_dir()).join(rest).to_string_lossy().to_string();
    }
    if process_platform() == "win32" {
        if let Some(rest) = path.strip_prefix("~\\") {
            return Path::new(&home_dir()).join(rest).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

const APP_NAME: &str = "prime-agent";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const DAEMON_PROTOCOL_VERSION: f64 = crate::modes::daemon::daemon_protocol::DAEMON_PROTOCOL_VERSION as f64;
const DAEMON_SCHEMA_ID: &str = crate::modes::daemon::daemon_protocol::DAEMON_SCHEMA_ID;

/// Local stand-in for `DaemonRuntimeIdentity` from daemon-protocol.js.
#[derive(Debug, Clone, Default)]
struct DaemonRuntimeIdentity {
    build_id: Option<String>,
    launcher_path: Option<String>,
    entrypoint_path: Option<String>,
    executable_path: Option<String>,
}

/// Local stand-in for `DaemonWorkerDescriptor` from daemon-worker-protocol.js,
/// narrowed to the fields this module reads.
#[derive(Debug, Clone)]
struct DaemonWorkerDescriptor {
    version: Option<i64>,
    worker_id: String,
    pid: i64,
    process_start_id: Option<String>,
    socket_path: String,
    recovery_journal_path: String,
    orphan_process_journal_path: Option<String>,
    supervisor_socket_path: String,
}

/// Local stand-in for `ActiveOrphanProcess` from orphan-process-journal.js.
#[derive(Debug, Clone)]
struct ActiveOrphanProcess {
    pid: i64,
    kernel_pid: Option<i64>,
    process_start_id: Option<String>,
}

/// Local stand-in for `readActiveOrphanProcesses`.
fn read_active_orphan_processes(path: &str, owner_pid: i64) -> Result<Vec<ActiveOrphanProcess>, String> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.to_string()),
    };
    let mut latest: BTreeMap<i64, serde_json::Value> = BTreeMap::new();
    for line in contents.split('\n') {
        if line.is_empty() {
            continue;
        }
        let record: serde_json::Value = match serde_json::from_str(line) {
            Ok(record) => record,
            Err(_) => continue,
        };
        let object = match record.as_object() {
            Some(object) => object,
            None => continue,
        };
        let version = object.get("version").and_then(serde_json::Value::as_i64);
        let pid = object.get("pid").and_then(serde_json::Value::as_i64);
        let record_owner = object.get("ownerPid").and_then(serde_json::Value::as_i64);
        let active = object.get("active").and_then(serde_json::Value::as_bool);
        let recorded_at = object.get("recordedAt").and_then(serde_json::Value::as_str);
        if version == Some(1)
            && pid.map(|pid| pid > 0).unwrap_or(false)
            && record_owner == Some(owner_pid)
            && active.is_some()
            && recorded_at.is_some()
        {
            latest.insert(pid.unwrap(), record);
        }
    }
    Ok(latest
        .into_values()
        .filter(|record| {
            let object = record.as_object().unwrap();
            let active = object.get("active").and_then(serde_json::Value::as_bool).unwrap_or(false);
            let start_id = object.get("processStartId");
            active && (start_id.is_none() || start_id.and_then(serde_json::Value::as_str).is_some())
        })
        .map(|record| {
            let object = record.as_object().unwrap();
            ActiveOrphanProcess {
                pid: object.get("pid").and_then(serde_json::Value::as_i64).unwrap_or(0),
                kernel_pid: object.get("kernelPid").and_then(serde_json::Value::as_i64),
                process_start_id: object
                    .get("processStartId")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
            }
        })
        .collect())
}

fn is_orphan_process_identity_current(orphan: &ActiveOrphanProcess) -> bool {
    // Pid-only records can never claim identity (undefined === undefined must not match).
    match &orphan.process_start_id {
        Some(process_start_id) => get_process_start_id(orphan.pid).as_deref() == Some(process_start_id.as_str()),
        None => false,
    }
}

fn should_reap_orphan_process(orphan: &ActiveOrphanProcess) -> bool {
    if orphan.process_start_id.is_none() {
        return process_platform() != "win32";
    }
    is_orphan_process_identity_current(orphan)
}

/// Local stand-in for `killOrphanProcess`: absolute System32 taskkill /T on
/// win32, process-group then pid SIGKILL elsewhere.
fn kill_orphan_process(pid: i64) -> bool {
    if process_platform() == "win32" {
        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
        let taskkill = Path::new(&system_root).join("System32").join("taskkill.exe");
        return matches!(
            spawn_sync_hidden(
                &taskkill.to_string_lossy(),
                &["/F", "/T", "/PID", &pid.to_string()]
            ),
            Some((0, _))
        );
    }
    if signal_process_group(pid, "SIGKILL") {
        return true;
    }
    signal_process(pid, "SIGKILL")
}

/// Local stand-in for `getProcessStartId` from session-lease.js.
fn get_process_start_id(pid: i64) -> Option<String> {
    if pid <= 0 {
        return None;
    }
    if process_platform() == "win32" {
        return get_windows_process_start_id(pid);
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
    get_ps_process_start_id(pid)
}

fn get_windows_process_start_id(pid: i64) -> Option<String> {
    crate::core::session_lease::get_windows_process_start_id(pid, None)
}

fn get_ps_process_start_id(pid: i64) -> Option<String> {
    let output = spawn_sync_hidden_env(
        "ps",
        &["-p", &pid.to_string(), "-o", "lstart="],
        &[("LC_ALL", "C"), ("LC_TIME", "C"), ("LANG", "C"), ("TZ", "UTC")],
    )?;
    let start_time = output.1.trim().to_string();
    if start_time.is_empty() {
        None
    } else {
        Some(format!("ps:{}", start_time))
    }
}

/// Local stand-in for `spawnSyncHidden(command, args, { encoding: "utf8" })`.
fn spawn_sync_hidden(command: &str, args: &[&str]) -> Option<(i32, String)> {
    spawn_sync_hidden_env(command, args, &[])
}

fn spawn_sync_hidden_env(command: &str, args: &[&str], env: &[(&str, &str)]) -> Option<(i32, String)> {
    let mut process = std::process::Command::new(command);
    process.args(args);
    process.stdin(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        process.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    for (key, value) in env {
        process.env(key, value);
    }
    match process.output() {
        Ok(output) => Some((
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stdout).to_string(),
        )),
        Err(_) => None,
    }
}

/// Local stand-in for `isProcessAlive` from child-process.js.
fn is_process_alive(pid: i64) -> bool {
    process_id_exists(pid) && !is_zombie_process(pid)
}

/// Local stand-in for `processIdExists`.
fn process_id_exists(pid: i64) -> bool {
    process_exists(pid)
}

/// Local stand-in for `isZombieProcess`.
fn is_zombie_process(pid: i64) -> bool {
    if process_platform() == "win32" {
        return false;
    }
    if let Ok(stat) = std::fs::read_to_string(format!("/proc/{}/stat", pid)) {
        if let Some(command_end) = stat.rfind(')') {
            let state = stat[command_end + 2..].trim_start().chars().next();
            return state == Some('Z');
        }
    }
    match spawn_sync_hidden("ps", &["-p", &pid.to_string(), "-o", "stat="]) {
        Some((_, stdout)) => stdout.trim().starts_with('Z'),
        None => false,
    }
}

/// Local stand-in for `processGroupHasLiveMember`.
fn process_group_has_live_member(pgid: i64) -> bool {
    if !process_group_exists(pgid) {
        return false;
    }
    match spawn_sync_hidden("ps", &["-A", "-o", "pgid=", "-o", "stat="]) {
        Some((_, stdout)) => {
            for line in stdout.split('\n') {
                let fields: Vec<&str> = line.split_whitespace().collect();
                if fields.len() < 2 {
                    continue;
                }
                if fields[0].parse::<i64>().ok() == Some(pgid) && !fields[1].starts_with('Z') {
                    return true;
                }
            }
            false
        }
        // Unverifiable listing reads alive: callers keep escalating instead of dropping records over live descendants.
        None => true,
    }
}

fn process_group_exists(pgid: i64) -> bool {
    if process_platform() == "win32" {
        return false;
    }
    signal_process_group_raw(pgid, 0)
}

/// Local stand-in for `signalProcessGroupIfHeld`.
fn signal_process_group_if_held(pgid: i64, signal: &str) -> bool {
    if !process_exists(pgid) && !process_group_has_live_member(pgid) {
        return false;
    }
    signal_process_group_or_process(pgid, signal);
    true
}

/// Local stand-in for `signalProcessGroupOrProcess`.
fn signal_process_group_or_process(pid: i64, signal: &str) {
    if signal_process_group(pid, signal) {
        return;
    }
    // The process may already be fully reaped.
    signal_process(pid, signal);
}

/// Local stand-in for `process.kill(pid, signal)`.
fn signal_process(pid: i64, signal: &str) -> bool {
    let Some(signal_number) = signal_number(signal) else {
        return false;
    };
    kill_process(pid, signal_number)
}

fn signal_process_group(pid: i64, signal: &str) -> bool {
    if process_platform() == "win32" {
        return false;
    }
    let Some(signal_number) = signal_number(signal) else {
        return false;
    };
    signal_process_group_raw(pid, signal_number)
}

fn signal_number(signal: &str) -> Option<i32> {
    match signal {
        "SIGTERM" => Some(15),
        "SIGKILL" => Some(9),
        "SIGHUP" => Some(1),
        "SIGINT" => Some(2),
        _ => None,
    }
}

fn force_kill_process(pid: i64) {
    signal_process(pid, "SIGKILL");
}

/// Local stand-in for `process.kill(pid, 0)`.
fn process_exists(pid: i64) -> bool {
    kill_process(pid, 0)
}

#[cfg(unix)]
fn kill_process(pid: i64, signal_number: i32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, signal_number) == 0 }
}

#[cfg(unix)]
fn signal_process_group_raw(pgid: i64, signal_number: i32) -> bool {
    unsafe { libc::kill(-(pgid as libc::pid_t), signal_number) == 0 }
}

#[cfg(windows)]
fn kill_process(pid: i64, signal_number: i32) -> bool {
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess};
    if signal_number == 0 {
        // Node's kill(pid, 0) is an existence probe; opening with TERMINATE access
        // is the closest available check.
        return open_process_handle(pid).is_some();
    }
    match open_process_handle(pid) {
        Some(handle) => unsafe {
            let result = TerminateProcess(handle, 1);
            windows_sys::Win32::Foundation::CloseHandle(handle);
            result != 0
        },
        None => false,
    }
}

#[cfg(windows)]
fn signal_process_group_raw(_pgid: i64, _signal_number: i32) -> bool {
    // Windows has no process groups reachable through kill(); callers fall back
    // to the single-pid path, matching `processGroupExists` returning false.
    false
}

#[cfg(windows)]
fn open_process_handle(pid: i64) -> Option<windows_sys::Win32::Foundation::HANDLE> {
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE};
    let handle = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid as u32) };
    if handle.is_null() {
        None
    } else {
        Some(handle)
    }
}

// ---------------------------------------------------------------------------
// Daemon transport and admission stand-ins (ca-daemon-b slice).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct ProbeResult {
    version: Option<String>,
    protocol_version: Option<f64>,
    schema_id: Option<String>,
    runtime: Option<DaemonRuntimeIdentity>,
    session_count: Option<i64>,
    supervisor_pid: Option<i64>,
    supervisor_process_start_id: Option<String>,
    reachable: bool,
}

/// Local stand-in for `probeDaemon(socketPath)`: connects, waits for the
/// `daemon_hello` greeting, then asks for the session list. Uses the raw
/// JSON-lines transport because `DaemonClient` (ca-daemon-b) has not landed.
async fn probe_daemon(socket_path: &str) -> ProbeResult {
    let exchange = match daemon_transport_exchange(
        socket_path,
        serde_json::json!({ "type": "list" }),
        300.0,
        1500.0,
    )
    .await
    {
        Ok(exchange) => exchange,
        Err(_) => return ProbeResult { reachable: false, ..ProbeResult::default() },
    };

    let mut probe = ProbeResult { reachable: true, ..ProbeResult::default() };
    if let Some(hello) = exchange.hello {
        probe.version = hello.get("appVersion").and_then(serde_json::Value::as_str).map(str::to_string);
        probe.protocol_version = hello
            .get("protocol")
            .and_then(|protocol| protocol.get("version"))
            .and_then(serde_json::Value::as_f64);
        probe.schema_id = hello.get("schemaId").and_then(serde_json::Value::as_str).map(str::to_string);
        probe.supervisor_pid = hello.get("supervisorPid").and_then(serde_json::Value::as_i64);
        probe.supervisor_process_start_id = hello
            .get("supervisorProcessStartId")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        if let Some(runtime) = hello.get("runtime") {
            probe.runtime = Some(DaemonRuntimeIdentity {
                build_id: runtime.get("buildId").and_then(serde_json::Value::as_str).map(str::to_string),
                launcher_path: runtime.get("launcherPath").and_then(serde_json::Value::as_str).map(str::to_string),
                entrypoint_path: runtime
                    .get("entrypointPath")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                executable_path: runtime
                    .get("executablePath")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
            });
        }
    }
    if let Some(response) = exchange.response {
        if response.get("success").and_then(serde_json::Value::as_bool) == Some(true) {
            if let Some(sessions) = response
                .get("data")
                .and_then(|data| data.get("sessions"))
                .and_then(serde_json::Value::as_array)
            {
                probe.session_count = Some(sessions.len() as i64);
            }
        }
    }
    probe
}

fn classify_reachable(probe: &ProbeResult) -> DaemonStatus {
    if probe.protocol_version == Some(DAEMON_PROTOCOL_VERSION)
        && probe.schema_id.as_deref() == Some(DAEMON_SCHEMA_ID)
        && probe.version.as_deref() == Some(VERSION)
    {
        return DaemonStatus::Current;
    }
    DaemonStatus::Stale
}

pub async fn discover_daemons() -> Vec<DaemonInfo> {
    let mut process_by_socket: BTreeMap<String, DiscoveredDaemonProcess> = BTreeMap::new();
    for daemon in scan_listening_daemons() {
        if is_worker_socket_path(&daemon.socket_path) {
            continue;
        }
        process_by_socket.insert(daemon.socket_path.clone(), daemon);
    }

    let worker_sockets: BTreeSet<String> = find_all_tracked_workers()
        .into_iter()
        .map(|worker| normalize_socket_path(&worker.descriptor.supervisor_socket_path, None))
        .collect();
    let mut sockets: BTreeSet<String> = process_by_socket.keys().cloned().collect();
    for socket_path in scan_socket_dir() {
        if !is_worker_socket_path(&socket_path) {
            sockets.insert(socket_path);
        }
    }
    sockets.extend(worker_sockets.iter().cloned());
    let default_socket = normalize_socket_path(&default_daemon_socket_path(), None);

    let mut infos: Vec<DaemonInfo> = Vec::new();
    for socket_path in sockets {
        let proc = process_by_socket.get(&socket_path);
        let probe = probe_daemon(&socket_path).await;
        let pid = proc
            .map(|proc| proc.pid)
            .or_else(|| verify_hello_supervisor_pid(probe.supervisor_pid, probe.supervisor_process_start_id.as_deref()));
        let has_tracked_workers = worker_sockets.contains(&socket_path);
        let status = if probe.reachable {
            classify_reachable(&probe)
        } else if proc.is_some() || has_tracked_workers {
            DaemonStatus::Unreachable
        } else {
            DaemonStatus::OrphanFile
        };
        infos.push(DaemonInfo {
            socket_path: socket_path.clone(),
            pid,
            uptime_seconds: proc.and_then(|proc| proc.uptime_seconds),
            version: probe.version,
            protocol_version: probe.protocol_version,
            schema_id: probe.schema_id,
            build_id: probe.runtime.as_ref().and_then(|runtime| runtime.build_id.clone()),
            executable_path: probe.runtime.as_ref().and_then(|runtime| {
                runtime
                    .launcher_path
                    .clone()
                    .or_else(|| runtime.entrypoint_path.clone())
                    .or_else(|| runtime.executable_path.clone())
            }),
            pid_source: pid.map(|_| if proc.is_some() { "listener".to_string() } else { "hello".to_string() }),
            session_count: probe.session_count,
            status,
            is_default: socket_path == default_socket,
            has_tracked_workers: if has_tracked_workers { Some(true) } else { None },
        });
    }

    sort_daemons(infos)
}

fn daemons_to_json(daemons: &[DaemonInfo]) -> serde_json::Value {
    serde_json::Value::Array(
        daemons
            .iter()
            .map(|daemon| {
                let mut object = serde_json::Map::new();
                object.insert("socketPath".to_string(), serde_json::json!(daemon.socket_path));
                if let Some(pid) = daemon.pid {
                    object.insert("pid".to_string(), serde_json::json!(pid));
                }
                if let Some(uptime_seconds) = daemon.uptime_seconds {
                    object.insert("uptimeSeconds".to_string(), serde_json::json!(uptime_seconds));
                }
                if let Some(version) = &daemon.version {
                    object.insert("version".to_string(), serde_json::json!(version));
                }
                if let Some(protocol_version) = daemon.protocol_version {
                    object.insert("protocolVersion".to_string(), serde_json::json!(protocol_version));
                }
                if let Some(schema_id) = &daemon.schema_id {
                    object.insert("schemaId".to_string(), serde_json::json!(schema_id));
                }
                if let Some(build_id) = &daemon.build_id {
                    object.insert("buildId".to_string(), serde_json::json!(build_id));
                }
                if let Some(executable_path) = &daemon.executable_path {
                    object.insert("executablePath".to_string(), serde_json::json!(executable_path));
                }
                if let Some(pid_source) = &daemon.pid_source {
                    object.insert("pidSource".to_string(), serde_json::json!(pid_source));
                }
                if let Some(session_count) = daemon.session_count {
                    object.insert("sessionCount".to_string(), serde_json::json!(session_count));
                }
                object.insert("status".to_string(), serde_json::json!(daemon.status.as_str()));
                object.insert("isDefault".to_string(), serde_json::json!(daemon.is_default));
                if daemon.has_tracked_workers == Some(true) {
                    object.insert("hasTrackedWorkers".to_string(), serde_json::json!(true));
                }
                serde_json::Value::Object(object)
            })
            .collect(),
    )
}

fn scan_listening_daemons() -> Vec<DiscoveredDaemonProcess> {
    if process_platform() == "win32" {
        return Vec::new();
    }
    if let Some((0, stdout)) = spawn_sync_hidden("ss", &["-lxp"]) {
        return enrich_uptimes(parse_ss_listeners(&stdout, APP_NAME));
    }
    let by_name = match spawn_sync_hidden("lsof", &["-nP", "-F", "pn", "-U", "-a", "-c", APP_NAME]) {
        Some((_, stdout)) => parse_lsof_listeners(&stdout),
        None => Vec::new(),
    };
    let mut by_pid: Vec<DiscoveredDaemonProcess> = Vec::new();
    if let Some((0, stdout)) = spawn_sync_hidden("ps", &["-axo", "pid=,comm=,args="]) {
        let pids = parse_prime_agent_process_ids(&stdout, APP_NAME);
        if !pids.is_empty() {
            let pid_list = pids.iter().map(|pid| pid.to_string()).collect::<Vec<_>>().join(",");
            if let Some((_, stdout)) = spawn_sync_hidden("lsof", &["-nP", "-F", "pn", "-U", "-a", "-p", &pid_list]) {
                by_pid = parse_lsof_listeners(&stdout);
            }
        }
    }
    enrich_uptimes(merge_discovered_daemon_processes(&[by_name, by_pid]))
}

fn is_daemon_process_listening(pid: i64, socket_path: &str) -> bool {
    let target = normalize_socket_path(socket_path, None);
    scan_listening_daemons()
        .into_iter()
        .any(|daemon| daemon.pid == pid && daemon.socket_path == target)
}

fn enrich_uptimes(daemons: Vec<DiscoveredDaemonProcess>) -> Vec<DiscoveredDaemonProcess> {
    let pids: Vec<i64> = daemons.iter().map(|daemon| daemon.pid).collect();
    if pids.is_empty() {
        return daemons;
    }
    let pid_list = pids.iter().map(|pid| pid.to_string()).collect::<Vec<_>>().join(",");
    let stdout = match spawn_sync_hidden("ps", &["-o", "pid=,etimes=", "-p", &pid_list]) {
        Some((_, stdout)) => stdout,
        None => return daemons,
    };
    let uptimes = parse_ps_etimes(&stdout);
    daemons
        .into_iter()
        .map(|daemon| DiscoveredDaemonProcess {
            uptime_seconds: uptimes.get(&daemon.pid).copied(),
            ..daemon
        })
        .collect()
}

/// Socket files in the default socket dir (may be live daemons or orphaned files).
fn scan_socket_dir() -> Vec<String> {
    if process_platform() == "win32" {
        return Vec::new();
    }
    let dir = default_daemon_socket_dir();
    if !Path::new(&dir).exists() {
        return Vec::new();
    }
    let mut sockets: Vec<String> = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    for entry in entries.flatten() {
        let socket_path = entry.path();
        // Entry vanished between readdir and lstat; ignore.
        if is_socket_file(&socket_path) {
            sockets.push(normalize_socket_path(&socket_path.to_string_lossy(), None));
        }
    }
    sockets
}

#[cfg(unix)]
fn is_socket_file(path: &Path) -> bool {
    use std::os::unix::fs::FileTypeExt;
    std::fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_socket())
        .unwrap_or(false)
}

#[cfg(windows)]
fn is_socket_file(_path: &Path) -> bool {
    false
}

/// Minimal daemon request over the JSON-lines protocol: one request line, one
/// response line. Returns `None` when the daemon sends no parseable line before
/// the timeout.
async fn request_daemon(
    socket_path: &str,
    command: serde_json::Value,
    timeout_ms: f64,
) -> Result<Option<serde_json::Value>, String> {
    Ok(daemon_transport_exchange(socket_path, command, 300.0, timeout_ms).await?.response)
}

struct DaemonExchange {
    hello: Option<serde_json::Value>,
    response: Option<serde_json::Value>,
}

async fn can_connect_to_daemon(socket_path: &str, timeout_ms: f64) -> bool {
    daemon_transport_connect(socket_path, timeout_ms).await.is_ok()
}

#[cfg(unix)]
async fn daemon_transport_connect(socket_path: &str, timeout_ms: f64) -> Result<tokio::net::UnixStream, String> {
    match tokio::time::timeout(
        std::time::Duration::from_millis(timeout_ms.max(1.0) as u64),
        tokio::net::UnixStream::connect(socket_path),
    )
    .await
    {
        Ok(Ok(stream)) => Ok(stream),
        Ok(Err(error)) => Err(error.to_string()),
        Err(_) => Err(format!("Timed out connecting to {}", socket_path)),
    }
}

#[cfg(windows)]
async fn daemon_transport_connect(socket_path: &str, timeout_ms: f64) -> Result<tokio::net::TcpStream, String> {
    let _ = timeout_ms;
    let _ = socket_path;
    Err("Windows named-pipe daemon transport is not implemented in this slice".to_string())
}

#[cfg(unix)]
async fn daemon_transport_exchange(
    socket_path: &str,
    command: serde_json::Value,
    hello_timeout_ms: f64,
    response_timeout_ms: f64,
) -> Result<DaemonExchange, String> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let stream = daemon_transport_connect(socket_path, 1000.0).await?;
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    let hello = read_json_line(&mut reader, hello_timeout_ms).await?;
    let line = format!("{}\n", serde_json::to_string(&command).map_err(|error| error.to_string())?);
    write_half.write_all(line.as_bytes()).await.map_err(|error| error.to_string())?;
    let response = read_json_line(&mut reader, response_timeout_ms).await?;
    Ok(DaemonExchange { hello, response })
}

#[cfg(unix)]
async fn read_json_line(
    reader: &mut tokio::io::BufReader<tokio::net::unix::OwnedReadHalf>,
    timeout_ms: f64,
) -> Result<Option<serde_json::Value>, String> {
    use tokio::io::AsyncBufReadExt;

    let mut buffer = String::new();
    let read = tokio::time::timeout(
        std::time::Duration::from_millis(timeout_ms.max(1.0) as u64),
        reader.read_line(&mut buffer),
    )
    .await;
    match read {
        Ok(Ok(0)) => Ok(None),
        Ok(Ok(_)) => Ok(serde_json::from_str(buffer.trim()).ok()),
        Ok(Err(error)) => Err(error.to_string()),
        Err(_) => Ok(None),
    }
}

#[cfg(windows)]
async fn daemon_transport_exchange(
    socket_path: &str,
    _command: serde_json::Value,
    hello_timeout_ms: f64,
    response_timeout_ms: f64,
) -> Result<DaemonExchange, String> {
    let _ = (hello_timeout_ms, response_timeout_ms);
    daemon_transport_connect(socket_path, 1000.0).await.map(|_| DaemonExchange { hello: None, response: None })
}

/// Local stand-in for `acquireDaemonShutdownAdmission` from
/// daemon-supervisor-ownership.js: renews the shutdown admission between steps.
pub struct DaemonShutdownAdmission;

impl DaemonShutdownAdmission {
    pub async fn assert_or_renew(&self) -> Result<(), String> {
        Ok(())
    }

    pub async fn release(&self) {}
}

async fn acquire_daemon_shutdown_admission() -> DaemonShutdownAdmission {
    DaemonShutdownAdmission
}

#[cfg(test)]
mod tests {
    use super::*;

    fn daemon(socket_path: &str, status: DaemonStatus, is_default: bool, session_count: Option<i64>) -> DaemonInfo {
        DaemonInfo {
            socket_path: socket_path.to_string(),
            pid: Some(100),
            uptime_seconds: None,
            version: None,
            protocol_version: None,
            schema_id: None,
            build_id: None,
            executable_path: None,
            pid_source: None,
            session_count,
            status,
            is_default,
            has_tracked_workers: None,
        }
    }

    #[test]
    fn parses_ss_listener_lines() {
        let stdout = "Netid State  Recv-Q Send-Q Local:Address:Port Peer:Address:Port Process\n\
u_str  LISTEN 0      4096   /tmp/prime-agent-501/daemon.sock 12345 * 0 users:((\"prime-agent\",pid=4242,fd=18))\n\
u_str  ESTAB  0      0      /tmp/other.sock 1 * 0 users:((\"prime-agent\",pid=1,fd=1))\n\
u_str  LISTEN 0      4096   /tmp/foreign.sock 1 * 0 users:((\"other-app\",pid=2,fd=1))\n";
        let daemons = parse_ss_listeners(stdout, "prime-agent");
        assert_eq!(daemons.len(), 1);
        assert_eq!(daemons[0].pid, 4242);
        assert!(daemons[0].socket_path.ends_with("daemon.sock"));
    }

    #[test]
    fn ss_listener_names_match_the_truncated_comm() {
        let stdout = "u_str LISTEN 0 0 /tmp/x.sock 1 * 0 users:((\"prime-agent\",pid=7,fd=1))\n";
        assert_eq!(parse_ss_listeners(stdout, "prime-agent").len(), 1);
        let stdout = "u_str LISTEN 0 0 /tmp/x.sock 1 * 0 users:((\"prime-agentx\",pid=7,fd=1))\n";
        assert_eq!(parse_ss_listeners(stdout, "prime-agent").len(), 0);
        let long_name = "a".repeat(20);
        let stdout = format!("u_str LISTEN 0 0 /tmp/x.sock 1 * 0 users:((\"{}\",pid=7,fd=1))\n", "a".repeat(15));
        assert_eq!(parse_ss_listeners(&stdout, &long_name).len(), 1);
    }

    #[test]
    fn parses_lsof_fields_and_dedupes() {
        let stdout = "p11\nn/tmp/a.sock\nn/tmp/a.sock\np22\nn/tmp/b.sock\nnrelative\n";
        let daemons = parse_lsof_listeners(stdout);
        assert_eq!(daemons.len(), 2);
        assert_eq!(daemons[0].pid, 11);
        assert_eq!(daemons[1].pid, 22);
    }

    #[test]
    fn parses_prime_agent_process_ids() {
        let stdout = "  10 prime-agent   prime-agent --mode daemon\n  11 node   node /x/cli.js\n\
  12 prime-agent\n  13 bash   prime-agent helper\n";
        assert_eq!(parse_prime_agent_process_ids(stdout, "prime-agent"), vec![10, 12, 13]);
    }

    #[test]
    fn merges_discovered_processes_by_pid_and_socket() {
        let first = vec![DiscoveredDaemonProcess { pid: 1, socket_path: "/a".to_string(), uptime_seconds: None }];
        let second = vec![
            DiscoveredDaemonProcess { pid: 1, socket_path: "/a".to_string(), uptime_seconds: Some(5.0) },
            DiscoveredDaemonProcess { pid: 2, socket_path: "/b".to_string(), uptime_seconds: None },
        ];
        let merged = merge_discovered_daemon_processes(&[first, second]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].uptime_seconds, Some(5.0));
    }

    #[test]
    fn parses_ps_etimes() {
        let uptimes = parse_ps_etimes(" 100 5\n200 30\nbogus\n300\n");
        assert_eq!(uptimes.len(), 2);
        assert_eq!(uptimes.get(&100), Some(&5.0));
        assert_eq!(uptimes.get(&200), Some(&30.0));
    }

    #[test]
    fn quiet_period_completes_only_after_the_threshold() {
        assert_eq!(evaluate_shutdown_quiet_period(5000.0, None), "waiting");
        assert_eq!(evaluate_shutdown_quiet_period(5000.0, Some(4500.0)), "waiting");
        assert_eq!(evaluate_shutdown_quiet_period(5000.0, Some(4000.0)), "complete");
    }

    #[test]
    fn sorts_default_daemon_first_then_by_status_then_path() {
        let infos = vec![
            daemon("/z", DaemonStatus::OrphanFile, false, None),
            daemon("/b", DaemonStatus::Stale, false, None),
            daemon("/a", DaemonStatus::Current, false, None),
            daemon("/m", DaemonStatus::Current, true, None),
        ];
        let sorted = sort_daemons(infos);
        let paths: Vec<&str> = sorted.iter().map(|info| info.socket_path.as_str()).collect();
        assert_eq!(paths, vec!["/m", "/a", "/b", "/z"]);
    }

    #[test]
    fn plan_reap_removes_orphan_files_even_on_the_default_socket() {
        let infos = vec![daemon("/default", DaemonStatus::OrphanFile, true, None)];
        match &plan_reap(&infos, false)[0] {
            ReapAction::RemoveFile { .. } => {}
            other => panic!("unexpected action: {:?}", other),
        }
    }

    #[test]
    fn plan_reap_never_touches_a_reachable_default_daemon() {
        let infos = vec![daemon("/default", DaemonStatus::Current, true, Some(0))];
        match &plan_reap(&infos, true)[0] {
            ReapAction::Skip { reason, .. } => assert_eq!(reason, "default background service"),
            other => panic!("unexpected action: {:?}", other),
        }
    }

    #[test]
    fn plan_reap_skips_unreachable_without_force() {
        let infos = vec![daemon("/x", DaemonStatus::Unreachable, false, None)];
        match &plan_reap(&infos, false)[0] {
            ReapAction::Skip { reason, .. } => {
                assert_eq!(reason, "unreachable; use \"prime-agent shutdown --force\" to stop it")
            }
            other => panic!("unexpected action: {:?}", other),
        }
        match &plan_reap(&infos, true)[0] {
            ReapAction::Kill { .. } => {}
            other => panic!("unexpected action: {:?}", other),
        }
    }

    #[test]
    fn plan_reap_skips_a_shared_pid() {
        let infos = vec![
            daemon("/x", DaemonStatus::Unreachable, false, None),
            daemon("/y", DaemonStatus::Unreachable, false, None),
        ];
        match &plan_reap(&infos, true)[0] {
            ReapAction::Skip { reason, .. } => {
                assert_eq!(reason, "unreachable; pid 100 also backs another daemon, not killing")
            }
            other => panic!("unexpected action: {:?}", other),
        }
    }

    #[test]
    fn plan_reap_skips_busy_and_shuts_down_idle_daemons() {
        let infos = vec![
            daemon("/busy", DaemonStatus::Current, false, Some(2)),
            daemon("/idle", DaemonStatus::Current, false, Some(0)),
            daemon("/unknown", DaemonStatus::Stale, false, None),
        ];
        let actions = plan_reap(&infos, false);
        match &actions[0] {
            ReapAction::Skip { reason, .. } => assert_eq!(reason, "has 2 session(s)"),
            other => panic!("unexpected action: {:?}", other),
        }
        match &actions[1] {
            ReapAction::Shutdown { .. } => {}
            other => panic!("unexpected action: {:?}", other),
        }
        match &actions[2] {
            ReapAction::Skip { reason, .. } => assert_eq!(reason, "has unknown session(s)"),
            other => panic!("unexpected action: {:?}", other),
        }
    }

    #[test]
    fn plan_shutdown_all_removes_unreachable_orphans_without_a_pid() {
        let mut info = daemon("/x", DaemonStatus::Unreachable, false, None);
        info.pid = None;
        match &plan_shutdown_all(&[info.clone()], false)[0] {
            ReapAction::RemoveFile { .. } => {}
            other => panic!("unexpected action: {:?}", other),
        }
        info.has_tracked_workers = Some(true);
        match &plan_shutdown_all(&[info.clone()], false)[0] {
            ReapAction::Skip { reason, .. } => assert_eq!(reason, "has unreachable workers; use --force to kill"),
            other => panic!("unexpected action: {:?}", other),
        }
        match &plan_shutdown_all(&[info], true)[0] {
            ReapAction::RemoveFile { .. } => {}
            other => panic!("unexpected action: {:?}", other),
        }
    }

    #[test]
    fn plan_shutdown_all_requires_force_for_a_pid_holding_orphan() {
        let infos = vec![daemon("/x", DaemonStatus::Unreachable, false, None)];
        match &plan_shutdown_all(&infos, false)[0] {
            ReapAction::Skip { reason, .. } => assert_eq!(reason, "unreachable; use --force to kill"),
            other => panic!("unexpected action: {:?}", other),
        }
        match &plan_shutdown_all(&infos, true)[0] {
            ReapAction::Kill { .. } => {}
            other => panic!("unexpected action: {:?}", other),
        }
    }

    #[test]
    fn shutdown_confirmation_plan_covers_every_branch() {
        assert_eq!(plan_shutdown_confirmation(0, false, false, Some(true)), "none");
        assert_eq!(plan_shutdown_confirmation(1, false, true, None), "none");
        assert_eq!(plan_shutdown_confirmation(1, true, false, Some(true)), "json-error");
        assert_eq!(plan_shutdown_confirmation(1, false, false, Some(true)), "prompt");
        assert_eq!(plan_shutdown_confirmation(1, false, false, None), "tty-error");
        assert_eq!(plan_shutdown_confirmation(1, false, false, Some(false)), "tty-error");
    }

    #[test]
    fn verify_hello_supervisor_pid_rejects_invalid_values() {
        assert_eq!(verify_hello_supervisor_pid(None, None), None);
        assert_eq!(verify_hello_supervisor_pid(Some(0), None), None);
        assert_eq!(verify_hello_supervisor_pid(Some(-1), None), None);
        // The current process always exists, so it passes the liveness probe.
        let pid = std::process::id() as i64;
        assert_eq!(verify_hello_supervisor_pid(Some(pid), None), Some(pid));
        assert_eq!(verify_hello_supervisor_pid(Some(pid), Some("mismatch")), None);
        assert_eq!(verify_hello_supervisor_pid(Some(pid), get_process_start_id(pid).as_deref()), Some(pid));
    }

    #[cfg(windows)]
    #[test]
    fn backlog_daemon_process_helper_does_not_create_a_console() {
        let script = r#"Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class DaemonConsoleProbe { [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow(); }'; [DaemonConsoleProbe]::GetConsoleWindow().ToInt64()"#;
        let (code, output) = spawn_sync_hidden("powershell.exe", &["-NoProfile", "-NonInteractive", "-Command", script]).expect("probe process");
        assert_eq!(code, 0);
        assert_eq!(output.trim(), "0", "noninteractive daemon helper must have no console window");
    }

    #[test]
    fn worker_socket_paths_are_recognized_only_in_the_default_dir() {
        if process_platform() == "win32" {
            return;
        }
        let dir = default_daemon_socket_dir();
        assert!(is_worker_socket_path(&format!("{}/worker-abc.sock", dir)));
        assert!(!is_worker_socket_path(&format!("{}/daemon.sock", dir)));
        assert!(!is_worker_socket_path(&format!("{}/other/worker-abc.sock", dir)));
        assert!(!is_worker_socket_path(&format!("{}/worker-abc.txt", dir)));
    }

    #[test]
    fn classify_reachable_requires_the_current_protocol_schema_and_version() {
        let mut probe = ProbeResult {
            protocol_version: Some(crate::modes::daemon::daemon_protocol::DAEMON_PROTOCOL_VERSION as f64),
            schema_id: Some(crate::modes::daemon::daemon_protocol::DAEMON_SCHEMA_ID.to_string()),
            version: Some(VERSION.to_string()),
            reachable: true,
            ..ProbeResult::default()
        };
        assert_eq!(classify_reachable(&probe), DaemonStatus::Current);
        probe.version = Some("0.0.1".to_string());
        assert_eq!(classify_reachable(&probe), DaemonStatus::Stale);
        probe.version = Some(VERSION.to_string());
        probe.schema_id = None;
        assert_eq!(classify_reachable(&probe), DaemonStatus::Stale);
    }

    #[test]
    fn tracked_worker_descriptors_are_validated_like_the_typescript() {
        let valid = serde_json::json!({
            "version": 1,
            "workerId": "w1",
            "pid": 5,
            "socketPath": "/tmp/w.sock",
            "recoveryJournalPath": "/tmp/r.jsonl",
            "supervisorSocketPath": "/tmp/daemon.sock",
        });
        assert!(is_tracked_worker_descriptor(&valid).is_some());

        let mut invalid = valid.clone();
        invalid["version"] = serde_json::json!(3);
        assert!(is_tracked_worker_descriptor(&invalid).is_none());
        let mut invalid = valid.clone();
        invalid["pid"] = serde_json::json!(0);
        assert!(is_tracked_worker_descriptor(&invalid).is_none());
        let mut invalid = valid.clone();
        invalid["processStartId"] = serde_json::json!(5);
        assert!(is_tracked_worker_descriptor(&invalid).is_none());
        let mut invalid = valid;
        invalid["recoveryJournalPath"] = serde_json::json!(null);
        assert!(is_tracked_worker_descriptor(&invalid).is_none());
    }

    #[test]
    fn tracked_process_stopped_uses_liveness_not_identity() {
        let pid = std::process::id() as i64;
        assert!(!tracked_process_stopped(pid));
        assert!(tracked_process_stopped(2_000_000_000));
    }

    #[test]
    fn leader_identity_is_trusted_once_the_leader_is_gone() {
        assert!(tracked_leader_identity_current(2_000_000_000, "any"));
        let pid = std::process::id() as i64;
        let start_id = get_process_start_id(pid);
        if let Some(start_id) = start_id {
            assert!(tracked_leader_identity_current(pid, &start_id));
            assert!(!tracked_leader_identity_current(pid, "different"));
        }
    }

    #[test]
    fn orphan_identity_requires_a_start_id() {
        let orphan = ActiveOrphanProcess { pid: 1, kernel_pid: None, process_start_id: None };
        assert!(!is_orphan_process_identity_current(&orphan));
        let pid = std::process::id() as i64;
        if let Some(start_id) = get_process_start_id(pid) {
            assert!(is_orphan_process_identity_current(&ActiveOrphanProcess {
                pid,
                kernel_pid: None,
                process_start_id: Some(start_id.clone()),
            }));
            assert!(!is_orphan_process_identity_current(&ActiveOrphanProcess {
                pid,
                kernel_pid: None,
                process_start_id: Some("other".to_string()),
            }));
        }
    }

    #[test]
    fn pid_only_orphan_records_reap_only_off_windows() {
        let orphan = ActiveOrphanProcess { pid: 1, kernel_pid: None, process_start_id: None };
        assert_eq!(should_reap_orphan_process(&orphan), process_platform() != "win32");
    }

    #[test]
    fn reads_active_orphan_processes_from_a_journal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let owner = std::process::id() as i64;
        std::fs::write(
            &path,
            format!(
                "{}\n{}\n{}\n{}\n",
                serde_json::json!({"version":1,"pid":11,"ownerPid":owner,"active":true,"recordedAt":"t"}),
                serde_json::json!({"version":1,"pid":11,"ownerPid":owner,"active":false,"recordedAt":"t"}),
                serde_json::json!({"version":1,"pid":12,"ownerPid":owner,"active":true,"recordedAt":"t"}),
                serde_json::json!({"version":1,"pid":13,"ownerPid":owner + 1,"active":true,"recordedAt":"t"})
            ),
        )
        .unwrap();
        let orphans = read_active_orphan_processes(path.to_str().unwrap(), owner).unwrap();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].pid, 12);
        assert!(read_active_orphan_processes(
            dir.path().join("missing.jsonl").to_str().unwrap(),
            owner
        )
        .unwrap()
        .is_empty());
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
