//! Port of packages/coding-agent/src/utils/shell.ts

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[cfg(not(windows))]
use super::child_process::spawn_sync_hidden;
use super::child_process::SpawnOptions;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellConfig {
    pub shell: String,
    pub args: Vec<String>,
}

/// System32\\bash.exe is the WSL launcher (runs Linux-side), so %SystemRoot% matches are only a last resort.
pub fn order_windows_bash_candidates(matches: &[String], system_root: Option<&str>) -> Vec<String> {
    let Some(system_root) = system_root else {
        return matches.to_vec();
    };
    let prefix = format!("{}\\", system_root.trim_end_matches('\\')).to_lowercase();
    let under_system_root = |candidate: &String| candidate.to_lowercase().starts_with(&prefix);
    let mut ordered: Vec<String> = matches
        .iter()
        .filter(|candidate| !under_system_root(candidate))
        .cloned()
        .collect();
    ordered.extend(matches.iter().filter(|candidate| under_system_root(candidate)).cloned());
    ordered
}

/// Find bash executable on PATH (cross-platform)
fn find_bash_on_path() -> Option<String> {
    if super::pi_user_agent::process_platform() == "win32" {
        // Windows: Use 'where' and verify file exists (where can return non-existent paths)
        // TS shell.ts:27: spawnSyncHidden(..., { timeout: 5000 }).
        let result = crate::utils::child_process::spawn_sync_hidden_with_timeout(
            "where",
            &["bash.exe".to_string()],
            SpawnOptions {
                capture_stdout: true,
                capture_stderr: true,
                ..Default::default()
            },
            5000,
        );
        if let Ok(output) = result {
            if output.status.success() && !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let matches: Vec<String> = stdout
                    .trim()
                    .split('\n')
                    .map(|line| line.trim_end_matches('\r').to_string())
                    .filter(|line| !line.is_empty())
                    .collect();
                let system_root = std::env::var("SystemRoot").ok();
                for candidate in order_windows_bash_candidates(&matches, system_root.as_deref()) {
                    if Path::new(&candidate).exists() {
                        return Some(candidate);
                    }
                }
            }
        }
        return None;
    }

    // Unix: Use 'which' and trust its output (handles Termux and special filesystems)
    // TS shell.ts:44: spawnSyncHidden(..., { timeout: 5000 }).
    let result = crate::utils::child_process::spawn_sync_hidden_with_timeout(
        "which",
        &["bash".to_string()],
        SpawnOptions {
            capture_stdout: true,
            capture_stderr: true,
            ..Default::default()
        },
        5000,
    );
    if let Ok(output) = result {
        if output.status.success() && !output.stdout.is_empty() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(first_match) = stdout.trim().split('\n').next() {
                if !first_match.is_empty() {
                    return Some(first_match.trim_end_matches('\r').to_string());
                }
            }
        }
    }
    None
}

/// Resolve shell configuration based on platform and an optional explicit shell path.
/// Resolution order:
/// 1. User-specified shellPath
/// 2. On Windows: Git Bash in known locations, then bash on PATH
/// 3. On Unix: /bin/bash, then bash on PATH, then fallback to sh
pub fn get_shell_config(custom_shell_path: Option<&str>) -> Result<ShellConfig, String> {
    // 1. Check user-specified shell path
    if let Some(custom_shell_path) = custom_shell_path {
        if !custom_shell_path.is_empty() {
            if Path::new(custom_shell_path).exists() {
                return Ok(ShellConfig {
                    shell: custom_shell_path.to_string(),
                    args: vec!["-c".to_string()],
                });
            }
            return Err(format!("Custom shell path not found: {}", custom_shell_path));
        }
    }

    if super::pi_user_agent::process_platform() == "win32" {
        // 2. Try Git Bash in known locations
        let mut paths: Vec<String> = Vec::new();
        if let Ok(program_files) = std::env::var("ProgramFiles") {
            if !program_files.is_empty() {
                paths.push(format!("{}\\Git\\bin\\bash.exe", program_files));
            }
        }
        if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
            if !program_files_x86.is_empty() {
                paths.push(format!("{}\\Git\\bin\\bash.exe", program_files_x86));
            }
        }

        for path in &paths {
            if Path::new(path).exists() {
                return Ok(ShellConfig {
                    shell: path.clone(),
                    args: vec!["-c".to_string()],
                });
            }
        }

        // 3. Fallback: search bash.exe on PATH (Cygwin, MSYS2, WSL, etc.)
        if let Some(bash_on_path) = find_bash_on_path() {
            return Ok(ShellConfig {
                shell: bash_on_path,
                args: vec!["-c".to_string()],
            });
        }

        let searched = paths
            .iter()
            .map(|path| format!("  {}", path))
            .collect::<Vec<String>>()
            .join("\n");
        return Err(format!(
            "No bash shell found. Options:\n  1. Install Git for Windows: https://git-scm.com/download/win\n  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n  3. Set shellPath in settings.json\n\nSearched Git Bash in:\n{}",
            searched
        ));
    }

    // Unix: try /bin/bash, then bash on PATH, then fallback to sh
    if Path::new("/bin/bash").exists() {
        return Ok(ShellConfig {
            shell: "/bin/bash".to_string(),
            args: vec!["-c".to_string()],
        });
    }

    if let Some(bash_on_path) = find_bash_on_path() {
        return Ok(ShellConfig {
            shell: bash_on_path,
            args: vec!["-c".to_string()],
        });
    }

    Ok(ShellConfig {
        shell: "sh".to_string(),
        args: vec!["-c".to_string()],
    })
}

// Hardcoded literals: ProgramFiles env vars are ambient attacker-influenceable
// input, the same trust-laundering class as PATH.
const WINDOWS_GIT_BASH_PATHS: [&str; 2] = [
    "C:\\Program Files\\Git\\bin\\bash.exe",
    "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
];

/// Absolute default shell for the kernel's bash(): explicit shellPath wins; POSIX
/// uses /bin/bash else /bin/sh (absolute, never PATH); win32 uses only the
/// canonical Git Bash install paths, never PATH.
/// None = no shell found: kernel startup must not fail, bash() raises its
/// teaching error.
pub fn resolve_kernel_bash_shell(custom_shell_path: Option<&str>) -> Option<String> {
    if let Some(explicit) = custom_shell_path.map(|value| value.trim()).filter(|value| !value.is_empty()) {
        return Some(explicit.to_string());
    }
    if super::pi_user_agent::process_platform() != "win32" {
        return Some(if Path::new("/bin/bash").exists() {
            "/bin/bash".to_string()
        } else {
            "/bin/sh".to_string()
        });
    }
    for path in WINDOWS_GIT_BASH_PATHS {
        if Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    None
}

pub fn get_shell_env() -> Vec<(String, String)> {
    let bin_dir = super::tools_manager::get_bin_dir().to_string_lossy().to_string();
    shell_env_with_bin_dir(std::env::vars().collect(), &bin_dir, cfg!(windows))
}

fn shell_env_with_bin_dir(mut env: Vec<(String, String)>, bin_dir: &str, windows: bool) -> Vec<(String, String)> {
    let is_path = |key: &str| if windows { key.eq_ignore_ascii_case("PATH") } else { key == "PATH" };
    let path_index = env.iter().position(|(key, _)| is_path(key));
    let current_path = path_index.map(|index| env[index].1.as_str()).unwrap_or_default();
    let separator = if windows { ';' } else { ':' };
    // The managed bin dir must lead PATH so managed tools win command
    // resolution, and it must appear exactly once so repeated calls stay
    // idempotent. An inherited PATH that already lists it mid-list (for
    // example a launcher that exports its own PATH prefixes ahead of it)
    // would otherwise leave it shadowed, so existing copies are dropped
    // instead of skipping the prepend. Empty entries are dropped too: they
    // resolve to the current directory and only enable binary planting.
    let updated_path = std::iter::once(bin_dir)
        .chain(
            current_path
                .split(separator)
                .filter(|entry| !entry.is_empty() && *entry != bin_dir),
        )
        .collect::<Vec<_>>()
        .join(&separator.to_string());
    if let Some(index) = path_index {
        env[index].1 = updated_path;
    } else {
        env.push(("PATH".to_string(), updated_path));
    }
    env
}

/// Sanitize binary output for display/storage.
/// Removes characters that crash string-width or cause display issues:
/// - Control characters (except tab, newline, carriage return)
/// - Lone surrogates
/// - Unicode Format characters (crash string-width due to a bug)
/// - Characters with undefined code points
pub fn sanitize_binary_output(value: &str) -> String {
    value
        .chars()
        .filter(|character| {
            let code = *character as u32;

            // Allow tab, newline, carriage return
            if code == 0x09 || code == 0x0a || code == 0x0d {
                return true;
            }

            // Filter out control characters (0x00-0x1F, except 0x09, 0x0a, 0x0d)
            if code <= 0x1f {
                return false;
            }

            // Filter out Unicode format characters
            if (0xfff9..=0xfffb).contains(&code) {
                return false;
            }

            true
        })
        .collect()
}

/// A numeric PID alone is not durable process ownership. Every handle carries
/// its own generation and immutable OS start token through registration,
/// detached tracking, deferred cleanup, and removal.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProcessStartIdentity(String);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct OwnedChildProcess {
    pid: i32,
    generation: u64,
    start: Option<ProcessStartIdentity>,
}

impl OwnedChildProcess {
    pub(crate) fn pid(&self) -> i32 {
        self.pid
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProcessGeneration {
    pid: i32,
    generation: u64,
}

impl From<&OwnedChildProcess> for ProcessGeneration {
    fn from(process: &OwnedChildProcess) -> Self {
        Self { pid: process.pid, generation: process.generation }
    }
}

#[derive(Debug)]
struct DetachedProcess {
    process: OwnedChildProcess,
    pending_kill: bool,
}

#[derive(Default, Debug)]
struct OwnedProcessRegistry {
    owned: HashMap<ProcessGeneration, OwnedChildProcess>,
    detached: HashMap<ProcessGeneration, DetachedProcess>,
}

impl OwnedProcessRegistry {
    fn record(&mut self, process: OwnedChildProcess) {
        // Generation collision must never replace the original token. Normal
        // registrations use a fresh atomic generation; this also fails closed
        // for synthetic collisions.
        self.owned.entry(ProcessGeneration::from(&process)).or_insert(process);
    }

    fn forget(&mut self, process: &OwnedChildProcess) {
        let key = ProcessGeneration::from(process);
        if self.owned.get(&key) == Some(process) {
            self.owned.remove(&key);
        }
    }

    fn track(&mut self, process: &OwnedChildProcess) {
        let key = ProcessGeneration::from(process);
        let process = self.owned.get(&key).cloned().unwrap_or_else(|| process.clone());
        self.detached.entry(key).or_insert(DetachedProcess { process, pending_kill: false });
    }

    fn release(&mut self, process: &OwnedChildProcess) {
        // A tracked or drained entry owns the token after the OwnedChild value
        // is gone. Another generation at the same numeric PID is irrelevant.
        if !self.detached.contains_key(&ProcessGeneration::from(process)) {
            self.forget(process);
        }
    }

    fn untrack(&mut self, process: &OwnedChildProcess) -> bool {
        let key = ProcessGeneration::from(process);
        let tracked = self.detached.get(&key).is_some_and(|entry| {
            !entry.pending_kill && entry.process == *process
        });
        if !tracked {
            // Completion racing a drained shutdown must not erase the token
            // that the pending cleanup still carries.
            return false;
        }
        self.detached.remove(&key);
        self.forget(process);
        true
    }

    fn drain_tracked(&mut self) -> Vec<OwnedChildProcess> {
        let keys: Vec<ProcessGeneration> = self.detached
            .iter()
            .filter(|(_, entry)| !entry.pending_kill)
            .map(|(key, _)| *key)
            .collect();
        let mut drained = Vec::with_capacity(keys.len());
        for key in keys {
            let Some(entry) = self.detached.get_mut(&key) else { continue };
            if entry.pending_kill {
                continue;
            }
            entry.pending_kill = true;
            drained.push(entry.process.clone());
        }
        drained.sort_by_key(|process| (process.pid, process.generation));
        drained
    }

    fn complete_kill(&mut self, process: &OwnedChildProcess, killed: bool) -> Option<bool> {
        let key = ProcessGeneration::from(process);
        let pending = self.detached.get(&key).is_some_and(|entry| {
            entry.pending_kill && entry.process == *process
        });
        if !pending {
            return None;
        }
        if killed {
            self.detached.remove(&key);
            self.forget(process);
        } else if let Some(entry) = self.detached.get_mut(&key) {
            entry.pending_kill = false;
        }
        Some(self.detached.keys().any(|key| key.pid == process.pid))
    }

    fn owned_at_pid_for_bare_kill(&self, pid: i32) -> Result<Option<OwnedChildProcess>, ()> {
        let mut matches = self.owned.values().filter(|process| process.pid == pid);
        let Some(process) = matches.next() else {
            return Ok(None);
        };
        if matches.next().is_some() {
            return Err(());
        }
        Ok(Some(process.clone()))
    }
}

static NEXT_OWNED_CHILD_GENERATION: AtomicU64 = AtomicU64::new(1);
static OWNED_PROCESS_REGISTRY: Mutex<Option<OwnedProcessRegistry>> = Mutex::new(None);

fn with_process_registry<R>(
    registry: &Mutex<Option<OwnedProcessRegistry>>,
    operation: impl FnOnce(&mut OwnedProcessRegistry) -> R,
) -> R {
    let mut guard = registry.lock().expect("owned process registry");
    operation(guard.get_or_insert_with(OwnedProcessRegistry::default))
}

#[cfg(all(not(windows), target_os = "linux"))]
fn probe_process_start_identity(pid: i32) -> Option<ProcessStartIdentity> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The command name is parenthesized and may contain spaces or parentheses.
    // Fields after its final ')' start at stat(5) field 3 (state), so field 22
    // (starttime) is zero-based offset 19 in this iterator.
    let start_ticks = stat
        .get(stat.rfind(')')? + 1..)?
        .split_whitespace()
        .nth(19)?;
    Some(ProcessStartIdentity(format!("linux:{start_ticks}")))
}

#[cfg(all(not(windows), not(target_os = "linux")))]
fn probe_process_start_identity(pid: i32) -> Option<ProcessStartIdentity> {
    let output = super::child_process::spawn_sync_hidden_with_timeout(
        "ps",
        &[
            "-p".to_string(),
            pid.to_string(),
            "-o".to_string(),
            "lstart=".to_string(),
        ],
        SpawnOptions {
            capture_stdout: true,
            capture_stderr: true,
            ..Default::default()
        },
        1_000,
    )
    .ok()?;
    if !output.status.success() {
        return None;
    }
    let start = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (!start.is_empty()).then_some(ProcessStartIdentity(format!("lstart:{start}")))
}

#[cfg(windows)]
fn probe_process_start_identity(pid: i32) -> Option<ProcessStartIdentity> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
        if handle.is_null() {
            return None;
        }
        let mut creation = std::mem::zeroed();
        let mut exit = std::mem::zeroed();
        let mut kernel = std::mem::zeroed();
        let mut user = std::mem::zeroed();
        let queried = GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user);
        CloseHandle(handle);
        if queried == 0 {
            return None;
        }
        Some(ProcessStartIdentity(format!(
            "windows:{:08x}{:08x}",
            creation.dwHighDateTime, creation.dwLowDateTime
        )))
    }
}

/// Capture the OS identity outside the registry lock, then attach a fresh
/// generation. A missing token is retained as missing, so live kills fail
/// closed instead of guessing from the current process listing.
pub(crate) fn register_owned_child_process(pid: i32) -> Option<OwnedChildProcess> {
    if pid <= 0 {
        return None;
    }
    let process = OwnedChildProcess {
        pid,
        generation: NEXT_OWNED_CHILD_GENERATION.fetch_add(1, Ordering::Relaxed),
        start: probe_process_start_identity(pid),
    };
    with_process_registry(&OWNED_PROCESS_REGISTRY, |registry| registry.record(process.clone()));
    Some(process)
}

fn registered_process_for_bare_kill(pid: i32) -> Result<Option<OwnedChildProcess>, ()> {
    with_process_registry(&OWNED_PROCESS_REGISTRY, |registry| registry.owned_at_pid_for_bare_kill(pid))
}

/// Detached entries are keyed by generation, not by their reusable numeric PID.
pub(crate) fn track_detached_child(process: &OwnedChildProcess) {
    with_process_registry(&OWNED_PROCESS_REGISTRY, |registry| registry.track(process));
    record_orphan_process_state(process.pid, true);
}

pub(crate) fn release_owned_child_process(process: &OwnedChildProcess) {
    with_process_registry(&OWNED_PROCESS_REGISTRY, |registry| registry.release(process));
}

pub(crate) fn untrack_detached_child(process: &OwnedChildProcess) {
    let active = with_process_registry(&OWNED_PROCESS_REGISTRY, |registry| {
        if registry.untrack(process) {
            Some(registry.detached.keys().any(|key| key.pid == process.pid))
        } else {
            None
        }
    });
    if let Some(active) = active {
        // Do not let generation A's deferred completion clear generation B's
        // active journal evidence at the same numeric PID.
        record_orphan_process_state(process.pid, active);
    }
}

pub fn kill_tracked_detached_children() {
    let completions = kill_tracked_detached_children_in_with(
        &OWNED_PROCESS_REGISTRY,
        kill_owned_child_process_tree,
    );
    for (pid, active) in completions {
        record_orphan_process_state(pid, active);
    }
}

fn kill_tracked_detached_children_in_with(
    registry: &Mutex<Option<OwnedProcessRegistry>>,
    mut kill: impl FnMut(&OwnedChildProcess) -> bool,
) -> Vec<(i32, bool)> {
    let mut completions = Vec::new();
    // One initial pending snapshot and one settling pass. Continuous
    // registration cannot keep process shutdown in an unbounded loop; entries
    // arriving after the second snapshot remain pending for external cleanup.
    for _ in 0..2 {
        let drained = with_process_registry(registry, OwnedProcessRegistry::drain_tracked);
        completions.extend(settle_drained_detached_children_in_with(registry, drained, &mut kill));
    }
    completions
}

fn settle_drained_detached_children_in_with(
    registry: &Mutex<Option<OwnedProcessRegistry>>,
    drained: Vec<OwnedChildProcess>,
    mut kill: impl FnMut(&OwnedChildProcess) -> bool,
) -> Vec<(i32, bool)> {
    let mut completions = Vec::new();
    for process in drained {
        // No registry lock is held while the callback probes or signals the
        // OS. Its identity is the immutable drained value, never a fresh
        // bare-PID registry lookup.
        let killed = kill(&process);
        if let Some(active) = with_process_registry(registry, |registry| {
            registry.complete_kill(&process, killed)
        }) {
            completions.push((process.pid, active));
        }
    }
    completions
}

/// `recordOrphanProcessState` from core/orphan-process-journal.ts.
///
/// Delegates to the journal module (same crate) so detached bash() children
/// and autonomous spawns surface to the recovered supervisor's reaper, and a
/// failed kill keeps its truthful active evidence.
fn record_orphan_process_state(pid: i32, tracked: bool) {
    crate::core::orphan_process_journal::record_orphan_process_state(pid as i64, tracked);
}

/// Absolute System32 taskkill for Windows tree kills. A bare "taskkill" name
/// would resolve through PATH and could pick up a planted CWD taskkill.exe.
pub fn windows_taskkill_program() -> String {
    windows_taskkill_program_for(std::env::var("SystemRoot").ok().as_deref())
}

fn windows_taskkill_program_for(system_root: Option<&str>) -> String {
    let system_root = system_root.unwrap_or("C:\\Windows");
    format!("{}\\System32\\taskkill.exe", system_root.trim_end_matches('\\'))
}

const PROCESS_EXIT_VERIFY_MS: u64 = 1_000;
const WINDOWS_TASKKILL_TIMEOUT_MS: u64 = 3_000;

fn wait_for_process_exit_with(mut alive: impl FnMut() -> bool, timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if !alive() {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(10).min(
            deadline.saturating_duration_since(std::time::Instant::now()),
        ));
    }
}

#[cfg(windows)]
const STILL_ACTIVE: u32 = 259;

#[cfg(windows)]
fn windows_exit_code_is_running(exit_code: u32) -> bool {
    exit_code == STILL_ACTIVE
}

/// Windows OpenProcess success alone is not liveness: another handle can keep
/// an exited process object open. Query the real status and fail conservative.
#[cfg(windows)]
fn process_target_running_status(pid: i32) -> Option<bool> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
        if handle.is_null() {
            return match std::io::Error::last_os_error().raw_os_error() {
                // Windows reports a nonexistent PID as ERROR_INVALID_PARAMETER.
                Some(87) => Some(false),
                _ => None,
            };
        }
        let mut exit_code = 0;
        let queried = GetExitCodeProcess(handle, &mut exit_code);
        CloseHandle(handle);
        (queried != 0).then(|| windows_exit_code_is_running(exit_code))
    }
}

#[cfg(not(windows))]
fn process_target_running_status(pid: i32) -> Option<bool> {
    Some(super::child_process::is_process_alive(pid))
}

fn process_target_running(pid: i32) -> bool {
    process_target_running_status(pid).unwrap_or(true)
}

fn process_exited(pid: i32) -> bool {
    process_exited_with(
        || process_target_running_status(pid),
        std::time::Duration::from_millis(PROCESS_EXIT_VERIFY_MS),
    )
}

fn process_exited_with(
    mut running: impl FnMut() -> Option<bool>,
    timeout: std::time::Duration,
) -> bool {
    wait_for_process_exit_with(|| running().unwrap_or(true), timeout)
}

fn taskkill_delivered(result: std::io::Result<std::process::Output>) -> bool {
    result.is_ok_and(|output| output.status.success())
}

/// Injectable POSIX operations keep process-group regressions synthetic. In
/// particular, tests never send a real negative-pid signal to the host.
trait PosixProcessControl {
    fn process_id(&mut self) -> i32;
    fn process_group_id(&mut self, pid: i32) -> Option<i32>;
    fn process_start_identity(&mut self, pid: i32) -> Option<ProcessStartIdentity>;
    fn process_running(&mut self, pid: i32) -> bool;
    fn group_has_live_member(&mut self, pgid: i32) -> bool;
    fn signal_process(&mut self, pid: i32) -> bool;
    fn signal_group(&mut self, pgid: i32) -> bool;
}

struct SystemPosixProcessControl;

impl PosixProcessControl for SystemPosixProcessControl {
    fn process_id(&mut self) -> i32 {
        std::process::id() as i32
    }

    fn process_group_id(&mut self, pid: i32) -> Option<i32> {
        #[cfg(not(windows))]
        {
            let pgid = unsafe { libc::getpgid(pid) };
            (pgid >= 0).then_some(pgid)
        }
        #[cfg(windows)]
        {
            let _ = pid;
            None
        }
    }

    fn process_start_identity(&mut self, pid: i32) -> Option<ProcessStartIdentity> {
        #[cfg(not(windows))]
        {
            probe_process_start_identity(pid)
        }
        #[cfg(windows)]
        {
            let _ = pid;
            None
        }
    }

    fn process_running(&mut self, pid: i32) -> bool {
        process_target_running(pid)
    }

    fn group_has_live_member(&mut self, pgid: i32) -> bool {
        super::child_process::process_group_has_live_member(pgid)
    }

    fn signal_process(&mut self, pid: i32) -> bool {
        #[cfg(not(windows))]
        {
            unsafe { libc::kill(pid, libc::SIGKILL) == 0 }
        }
        #[cfg(windows)]
        {
            let _ = pid;
            false
        }
    }

    fn signal_group(&mut self, pgid: i32) -> bool {
        // This is deliberately not signal_process_group_or_process: its
        // positive-pid fallback could hit a reused leader after the real
        // process group has gone away.
        #[cfg(not(windows))]
        {
            unsafe { libc::kill(-pgid, libc::SIGKILL) == 0 }
        }
        #[cfg(windows)]
        {
            let _ = pgid;
            false
        }
    }
}

fn kill_posix_process_with(
    control: &mut impl PosixProcessControl,
    pid: i32,
    timeout: std::time::Duration,
) -> bool {
    let signalled = control.signal_process(pid);
    if signalled {
        wait_for_process_exit_with(|| control.process_running(pid), timeout)
    } else {
        // A racing natural exit is still a completed cleanup. Any surviving
        // target keeps the tracking and journal evidence active.
        !control.process_running(pid)
    }
}

fn kill_posix_group_with(
    control: &mut impl PosixProcessControl,
    pgid: i32,
    timeout: std::time::Duration,
) -> bool {
    let signalled = control.signal_group(pgid);
    let mut survivors = || {
        control.process_running(pgid) || control.group_has_live_member(pgid)
    };
    if signalled {
        wait_for_process_exit_with(survivors, timeout)
    } else {
        !survivors()
    }
}

fn kill_posix_process_tree_with(
    control: &mut impl PosixProcessControl,
    pid: i32,
    expected_start: Option<&ProcessStartIdentity>,
    timeout: std::time::Duration,
) -> bool {
    let own_pid = control.process_id();
    if pid <= 0 || pid == own_pid {
        return false;
    }
    let own_pgid = control.process_group_id(own_pid);
    let target_pgid = control.process_group_id(pid);
    let target_running = control.process_running(pid);

    // Every live numeric anchor must still be the exact process registered at
    // spawn. pgid == pid only describes its current group; it says nothing
    // about whether the numeric pid was recycled after the original exited.
    if target_running {
        let Some(expected_start) = expected_start else {
            return false;
        };
        if control.process_start_identity(pid).as_ref() != Some(expected_start) {
            return false;
        }
    }

    match target_pgid {
        Some(pgid) if pgid <= 0 => false,
        Some(pgid) if Some(pgid) == own_pgid => {
            // A non-detached child can share the agent's group. Kill only the
            // requested member; signalling that group would kill the agent.
            kill_posix_process_with(control, pid, timeout)
        }
        Some(pgid) if pgid != pid => {
            // Owned detached targets lead their own group. A live leader in a
            // different, non-self group is not a safe anchor for the original
            // group and may be a reused pid. Do not signal either identity.
            false
        }
        Some(pgid) => {
            // getpgid also succeeds for a zombie. Verify its start token too:
            // a dead or zombie new owner is not proof that this is our group.
            let Some(expected_start) = expected_start else {
                return false;
            };
            if control.process_start_identity(pid).as_ref() != Some(expected_start) {
                return false;
            }
            kill_posix_group_with(control, pgid, timeout)
        }
        None if target_running => {
            // The start token has already authenticated this live process.
            // getpgid failure cannot justify guessing or using a group number.
            kill_posix_process_with(control, pid, timeout)
        }
        None => {
            // A group can outlive its leader. There is no positive-pid anchor
            // to signal in this branch: preserve the orphan cleanup contract
            // for targets spawned with process_group(0) when a member remains.
            if !control.group_has_live_member(pid) {
                return true;
            }
            if Some(pid) == own_pgid {
                return false;
            }
            kill_posix_group_with(control, pid, timeout)
        }
    }
}

fn kill_posix_process_tree(process: Option<&OwnedChildProcess>, pid: i32) -> bool {
    let expected_start = process.and_then(|process| process.start.as_ref());
    kill_posix_process_tree_with(
        &mut SystemPosixProcessControl,
        pid,
        expected_start,
        std::time::Duration::from_millis(PROCESS_EXIT_VERIFY_MS),
    )
}

pub(crate) fn kill_owned_child_process_tree(process: &OwnedChildProcess) -> bool {
    if super::pi_user_agent::process_platform() != "win32" {
        return kill_posix_process_tree(Some(process), process.pid);
    }
    if !process_target_running(process.pid) {
        return true;
    }
    let Some(expected_start) = process.start.as_ref() else {
        return false;
    };
    if probe_process_start_identity(process.pid).as_ref() != Some(expected_start) {
        return false;
    }
    kill_windows_process_tree(process.pid)
}

#[cfg(test)]
fn checked_signal_kill(pid: i32, signalled: bool) -> bool {
    signalled && process_exited(pid)
}

/// Kill a process and all its children. `true` means the target was already
/// gone or was observed exited inside the verification bound. False keeps
/// detached tracking and journal evidence active for a later retry.
pub fn kill_process_tree(pid: i32) -> bool {
    if super::pi_user_agent::process_platform() != "win32" {
        // Compatibility for callers that do not hold an OwnedChildProcess.
        // Ambiguity between generations fails closed; all owned cleanup paths
        // pass their exact generation directly instead of using this helper.
        let Ok(process) = registered_process_for_bare_kill(pid) else {
            return false;
        };
        return kill_posix_process_tree(process.as_ref(), pid);
    }
    kill_windows_process_tree(pid)
}

fn kill_windows_process_tree(pid: i32) -> bool {
    if !process_target_running(pid) {
        return true;
    }
    // Capture and drain taskkill under the same bounded sync helper as
    // other children; a failed spawn/status or a live target fails closed.
    let result = super::child_process::spawn_sync_hidden_with_timeout(
        &windows_taskkill_program(),
        &[
            "/F".to_string(),
            "/T".to_string(),
            "/PID".to_string(),
            pid.to_string(),
        ],
        SpawnOptions {
            env: Some(vec![("NoDefaultCurrentDirectoryInExePath".to_string(), "1".to_string())]),
            capture_stdout: true,
            capture_stderr: true,
            ..Default::default()
        },
        WINDOWS_TASKKILL_TIMEOUT_MS,
    );
    taskkill_delivered(result) && process_exited(pid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_binary_output_keeps_printable_characters() {
        assert_eq!(sanitize_binary_output("hello\tworld\n"), "hello\tworld\n");
        assert_eq!(sanitize_binary_output("a\u{0}b"), "ab");
        assert_eq!(sanitize_binary_output("a\u{1f}b"), "ab");
        assert_eq!(sanitize_binary_output("a\u{fff9}b"), "ab");
        assert_eq!(sanitize_binary_output("a\u{fffb}b"), "ab");
        assert_eq!(sanitize_binary_output("emoji \u{1f600}"), "emoji \u{1f600}");
    }

    #[test]
    fn windows_bash_candidates_put_system_root_last() {
        let matches = vec![
            "C:\\Windows\\System32\\bash.exe".to_string(),
            "C:\\Program Files\\Git\\bin\\bash.exe".to_string(),
        ];
        let ordered = order_windows_bash_candidates(&matches, Some("C:\\Windows"));
        assert_eq!(ordered[0], "C:\\Program Files\\Git\\bin\\bash.exe");
        assert_eq!(ordered[1], "C:\\Windows\\System32\\bash.exe");
        assert_eq!(order_windows_bash_candidates(&matches, None), matches);
    }

    #[test]
    fn explicit_shell_path_wins_and_missing_paths_fail() {
        let existing = std::env::current_exe().unwrap().to_string_lossy().to_string();
        let config = get_shell_config(Some(&existing)).unwrap();
        assert_eq!(config.shell, existing);
        assert_eq!(config.args, vec!["-c".to_string()]);

        let error = get_shell_config(Some("no-such-shell-xyz")).unwrap_err();
        assert_eq!(error, "Custom shell path not found: no-such-shell-xyz");
    }

    #[test]
    fn windows_taskkill_program_is_the_absolute_system32_path() {
        assert_eq!(
            windows_taskkill_program_for(None),
            "C:\\Windows\\System32\\taskkill.exe"
        );
        assert_eq!(
            windows_taskkill_program_for(Some("D:\\WinTest")),
            "D:\\WinTest\\System32\\taskkill.exe"
        );
        assert_eq!(
            windows_taskkill_program_for(Some("D:\\WinTest\\")),
            "D:\\WinTest\\System32\\taskkill.exe"
        );
    }

    #[test]
    fn kernel_bash_shell_prefers_the_explicit_path() {
        assert_eq!(
            resolve_kernel_bash_shell(Some("  /custom/bash  ")),
            Some("/custom/bash".to_string())
        );
        if cfg!(windows) {
            let resolved = resolve_kernel_bash_shell(None);
            if let Some(resolved) = resolved {
                assert!(WINDOWS_GIT_BASH_PATHS.contains(&resolved.as_str()));
            }
        } else {
            assert!(resolve_kernel_bash_shell(None).is_some());
        }
    }

    #[test]
    fn managed_path_uses_platform_keys_and_preserves_existing_entries() {
        let env = vec![("Path".into(), "unrelated".into()), ("PATH".into(), ":/usr/bin:".into())];
        let posix = shell_env_with_bin_dir(env, "/managed", false);
        assert_eq!(posix[0].1, "unrelated");
        assert_eq!(posix[1].1, "/managed:/usr/bin");
        assert_eq!(shell_env_with_bin_dir(posix.clone(), "/managed", false), posix);
        let windows = shell_env_with_bin_dir(vec![("Path".into(), r"C:\System;D:\Tools".into())], r"C:\Managed", true);
        assert_eq!(windows, vec![("Path".into(), r"C:\Managed;C:\System;D:\Tools".into())]);
        assert_eq!(shell_env_with_bin_dir(windows.clone(), r"C:\Managed", true), windows);
        for env in [vec![], vec![("PATH".into(), String::new())]] {
            assert_eq!(shell_env_with_bin_dir(env, "/managed", false), vec![("PATH".into(), "/managed".into())]);
        }
    }

    #[test]
    fn shell_env_prepends_the_bin_dir_once() {
        let bin_dir = super::super::tools_manager::get_bin_dir().to_string_lossy().to_string();
        let env = get_shell_env();
        let path_entry = env
            .iter()
            .find(|(key, _)| key.to_lowercase() == "path")
            .map(|(_, value)| value.clone())
            .unwrap();
        // The managed bin dir must lead PATH exactly once, even when the
        // inherited PATH already lists it mid-list behind launcher-exported
        // prefixes.
        let separator = if cfg!(windows) { ';' } else { ':' };
        let entries: Vec<&str> = path_entry.split(separator).collect();
        assert_eq!(
            entries.first(),
            Some(&bin_dir.as_str()),
            "PATH must start with the managed bin dir: {path_entry}"
        );
        assert_eq!(
            entries.iter().filter(|entry| **entry == bin_dir).count(),
            1,
            "managed bin dir must appear exactly once: {path_entry}"
        );

        let second = get_shell_env();
        let second_path = second
            .iter()
            .find(|(key, _)| key.to_lowercase() == "path")
            .map(|(_, value)| value.clone())
            .unwrap();
        assert_eq!(path_entry, second_path);
    }

    // Regression (product): an inherited PATH that already contains the
    // managed bin dir mid-list must get it moved to the front exactly once,
    // not left in place. Host launchers export their own PATH prefixes ahead
    // of the agent bin dir, which used to leave the managed tools shadowed
    // (observed on Windows where the test harness prepends toolchain dirs).
    #[test]
    fn shell_env_moves_an_inherited_mid_list_bin_dir_to_the_front() {
        let posix = shell_env_with_bin_dir(
            vec![("PATH".into(), "/usr/bin:/managed:/extra".into())],
            "/managed",
            false,
        );
        assert_eq!(posix, vec![("PATH".into(), "/managed:/usr/bin:/extra".into())]);

        let windows = shell_env_with_bin_dir(
            vec![("Path".into(), r"C:\Toolchain;C:\Managed;C:\Tools".into())],
            r"C:\Managed",
            true,
        );
        assert_eq!(
            windows,
            vec![("Path".into(), r"C:\Managed;C:\Toolchain;C:\Tools".into())]
        );
    }

    fn process_at(pid: i32, generation: u64, start: &ProcessStartIdentity) -> OwnedChildProcess {
        OwnedChildProcess { pid, generation, start: Some(start.clone()) }
    }

    fn registry_with(processes: &[OwnedChildProcess]) -> Mutex<Option<OwnedProcessRegistry>> {
        let registry = Mutex::new(Some(OwnedProcessRegistry::default()));
        for process in processes {
            registry.lock().unwrap().as_mut().unwrap().record(process.clone());
        }
        registry
    }

    fn is_owned(registry: &Mutex<Option<OwnedProcessRegistry>>, process: &OwnedChildProcess) -> bool {
        registry.lock().unwrap().as_ref().unwrap().owned.get(&ProcessGeneration::from(process)) == Some(process)
    }

    fn detached_entry(
        registry: &Mutex<Option<OwnedProcessRegistry>>,
        process: &OwnedChildProcess,
    ) -> (bool, bool) {
        let guard = registry.lock().unwrap();
        let entry = guard.as_ref().unwrap().detached.get(&ProcessGeneration::from(process));
        (entry.is_some(), entry.is_some_and(|entry| entry.pending_kill))
    }

    fn owned_start() -> ProcessStartIdentity {
        ProcessStartIdentity("owned-start".to_string())
    }

    fn unrelated_start() -> ProcessStartIdentity {
        ProcessStartIdentity("unrelated-start".to_string())
    }

    #[test]
    fn detached_child_tracking_is_idempotent_per_generation() {
        let process = process_at(999_991, 1, &owned_start());
        let registry = registry_with(&[process.clone()]);
        registry.lock().unwrap().as_mut().unwrap().track(&process);
        registry.lock().unwrap().as_mut().unwrap().track(&process);
        assert_eq!(detached_entry(&registry, &process), (true, false));
        assert!(registry.lock().unwrap().as_mut().unwrap().untrack(&process));
        assert_eq!(detached_entry(&registry, &process), (false, false));
        assert!(!is_owned(&registry, &process));
    }

    #[test]
    fn failed_kill_and_spawn_results_are_not_delivered() {
        assert!(!checked_signal_kill(999_990, false));
        let spawn_failure: std::io::Result<std::process::Output> = Err(
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "injected spawn failure"),
        );
        assert!(!taskkill_delivered(spawn_failure));
    }

    #[test]
    fn process_exit_verification_is_bounded_and_observes_late_exit() {
        let mut probes = 0;
        assert!(wait_for_process_exit_with(
            || {
                probes += 1;
                probes < 3
            },
            std::time::Duration::from_secs(1),
        ));
        assert_eq!(probes, 3);
        let started = std::time::Instant::now();
        assert!(!wait_for_process_exit_with(
            || true,
            std::time::Duration::from_millis(15),
        ));
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn failed_detached_kill_stays_registered_for_retry() {
        let first = process_at(999_970, 1, &owned_start());
        let retry = process_at(999_971, 2, &owned_start());
        let registry = registry_with(&[first.clone(), retry.clone()]);
        {
            let mut guard = registry.lock().unwrap();
            let registry = guard.as_mut().unwrap();
            registry.track(&first);
            registry.track(&retry);
        }
        kill_tracked_detached_children_in_with(&registry, |process| process.pid != retry.pid);
        assert_eq!(detached_entry(&registry, &first), (false, false));
        assert_eq!(detached_entry(&registry, &retry), (true, false));
        assert!(is_owned(&registry, &retry));
    }

    #[test]
    fn registration_during_shutdown_is_processed_in_the_next_round() {
        let first = process_at(999_960, 1, &owned_start());
        let second = process_at(999_961, 2, &unrelated_start());
        let registry = registry_with(&[first.clone()]);
        registry.lock().unwrap().as_mut().unwrap().track(&first);
        let mut killed = Vec::new();
        kill_tracked_detached_children_in_with(&registry, |process| {
            killed.push(process.pid);
            if process == &first {
                let mut guard = registry.lock().unwrap();
                let registry = guard.as_mut().unwrap();
                registry.record(second.clone());
                registry.track(&second);
            }
            true
        });
        assert_eq!(killed, vec![first.pid, second.pid]);
        assert_eq!(detached_entry(&registry, &first), (false, false));
        assert_eq!(detached_entry(&registry, &second), (false, false));
    }

    #[test]
    fn continuous_registration_is_left_pending_after_two_bounded_passes() {
        let first = process_at(999_950, 1, &owned_start());
        let registry = registry_with(&[first.clone()]);
        registry.lock().unwrap().as_mut().unwrap().track(&first);
        let mut killed = Vec::new();
        let mut successor = None;
        kill_tracked_detached_children_in_with(&registry, |process| {
            killed.push(process.pid);
            // Register exactly one successor per pass, forever in principle.
            // The second registration must survive the bounded shutdown call.
            let next = process_at(process.pid + 1, process.generation + 1, &unrelated_start());
            successor = Some(next.clone());
            let mut guard = registry.lock().unwrap();
            let registry = guard.as_mut().unwrap();
            registry.record(next.clone());
            registry.track(&next);
            true
        });
        assert_eq!(killed, vec![first.pid, first.pid + 1]);
        assert_eq!(detached_entry(&registry, &first), (false, false));
        let successor = successor.unwrap();
        assert_eq!(successor.pid, first.pid + 2);
        assert_eq!(detached_entry(&registry, &successor), (true, false));
    }

    #[test]
    fn process_exit_probe_honors_status_and_conserves_query_errors() {
        assert!(process_exited_with(
            || Some(false),
            std::time::Duration::ZERO,
        ));
        assert!(!process_exited_with(
            || None,
            std::time::Duration::ZERO,
        ));
    }

    #[test]
    fn same_pid_generations_keep_distinct_tokens_without_overwrite() {
        let first = process_at(4_242, 1, &owned_start());
        let second = process_at(4_242, 2, &unrelated_start());
        let registry = registry_with(&[first.clone(), second.clone()]);
        assert!(is_owned(&registry, &first));
        assert!(is_owned(&registry, &second));
        assert!(registry.lock().unwrap().as_ref().unwrap().owned_at_pid_for_bare_kill(first.pid).is_err());

        // Even a synthetic generation collision cannot replace A's token.
        registry.lock().unwrap().as_mut().unwrap().record(process_at(4_242, 1, &unrelated_start()));
        assert!(is_owned(&registry, &first));
    }

    #[test]
    fn stale_untrack_after_same_pid_b_registration_only_removes_a() {
        let first = process_at(4_242, 1, &owned_start());
        let second = process_at(4_242, 2, &unrelated_start());
        let registry = registry_with(&[first.clone(), second.clone()]);
        {
            let mut guard = registry.lock().unwrap();
            let registry = guard.as_mut().unwrap();
            registry.track(&first);
            registry.track(&second);
        }
        assert!(registry.lock().unwrap().as_mut().unwrap().untrack(&first));
        assert!(!is_owned(&registry, &first));
        assert!(is_owned(&registry, &second));
        assert_eq!(detached_entry(&registry, &second), (true, false));
        assert!(!registry.lock().unwrap().as_mut().unwrap().untrack(&first));
        assert!(is_owned(&registry, &second));
    }

    #[test]
    fn deferred_release_and_untrack_cannot_delete_pending_a_or_new_b() {
        let first = process_at(4_242, 1, &owned_start());
        let second = process_at(4_242, 2, &unrelated_start());
        let registry = registry_with(&[first.clone()]);
        registry.lock().unwrap().as_mut().unwrap().track(&first);
        assert_eq!(registry.lock().unwrap().as_mut().unwrap().drain_tracked(), vec![first.clone()]);

        {
            let mut guard = registry.lock().unwrap();
            let registry = guard.as_mut().unwrap();
            registry.record(second.clone());
            registry.track(&second);
            registry.release(&first);
        }
        assert!(is_owned(&registry, &first), "release must retain the drained token");
        assert!(!registry.lock().unwrap().as_mut().unwrap().untrack(&first));
        assert!(is_owned(&registry, &first), "completion must retain the pending token");
        assert!(is_owned(&registry, &second));

        assert_eq!(registry.lock().unwrap().as_mut().unwrap().complete_kill(&first, true), Some(true));
        assert!(!is_owned(&registry, &first));
        assert!(is_owned(&registry, &second), "A completion deleted B's token");
        assert_eq!(detached_entry(&registry, &second), (true, false));
    }

    struct FakePosixProcessControl {
        own_pid: i32,
        own_pgid: i32,
        target_pid: i32,
        target_pgid: Option<i32>,
        target_running: bool,
        recorded_start: ProcessStartIdentity,
        observed_start: Option<ProcessStartIdentity>,
        group_live: bool,
        group_signal_delivered: bool,
        process_signal_delivered: bool,
        group_survives_signal: bool,
        group_probes: usize,
        process_signals: Vec<i32>,
        group_signals: Vec<i32>,
    }

    impl FakePosixProcessControl {
        fn new(target_pgid: Option<i32>, target_running: bool, group_live: bool) -> Self {
            Self {
                own_pid: 1_000,
                own_pgid: 1_000,
                target_pid: 4_242,
                target_pgid,
                target_running,
                recorded_start: owned_start(),
                observed_start: Some(owned_start()),
                group_live,
                group_signal_delivered: true,
                process_signal_delivered: true,
                group_survives_signal: false,
                group_probes: 0,
                process_signals: Vec::new(),
                group_signals: Vec::new(),
            }
        }

        fn kill(&mut self) -> bool {
            let pid = self.target_pid;
            let expected_start = self.recorded_start.clone();
            kill_posix_process_tree_with(
                self,
                pid,
                Some(&expected_start),
                std::time::Duration::ZERO,
            )
        }
    }

    impl PosixProcessControl for FakePosixProcessControl {
        fn process_id(&mut self) -> i32 {
            self.own_pid
        }

        fn process_group_id(&mut self, pid: i32) -> Option<i32> {
            if pid == self.own_pid {
                Some(self.own_pgid)
            } else if pid == self.target_pid {
                self.target_pgid
            } else {
                None
            }
        }

        fn process_start_identity(&mut self, pid: i32) -> Option<ProcessStartIdentity> {
            (pid == self.target_pid)
                .then(|| self.observed_start.clone())
                .flatten()
        }

        fn process_running(&mut self, pid: i32) -> bool {
            if pid == self.own_pid {
                true
            } else {
                pid == self.target_pid && self.target_running
            }
        }

        fn group_has_live_member(&mut self, _pgid: i32) -> bool {
            self.group_probes += 1;
            self.group_live
        }

        fn signal_process(&mut self, pid: i32) -> bool {
            self.process_signals.push(pid);
            if self.process_signal_delivered && pid == self.target_pid {
                self.target_running = false;
            }
            self.process_signal_delivered
        }

        fn signal_group(&mut self, pgid: i32) -> bool {
            self.group_signals.push(pgid);
            if self.group_signal_delivered && !self.group_survives_signal {
                self.group_live = false;
                self.target_running = false;
            }
            self.group_signal_delivered
        }
    }

    #[test]
    fn pending_drained_a_after_b_registration_fails_closed_and_preserves_b() {
        let first = process_at(4_242, 1, &owned_start());
        let second = process_at(4_242, 2, &unrelated_start());
        let registry = registry_with(&[first.clone()]);
        registry.lock().unwrap().as_mut().unwrap().track(&first);
        let drained = registry.lock().unwrap().as_mut().unwrap().drain_tracked();
        assert_eq!(drained, vec![first.clone()]);

        // A exits in its owning runtime after shutdown drained A. Pending
        // cleanup still owns A's token; B then acquires the same numeric PID.
        assert!(!registry.lock().unwrap().as_mut().unwrap().untrack(&first));
        registry.lock().unwrap().as_mut().unwrap().record(second.clone());
        assert!(is_owned(&registry, &first));
        assert!(is_owned(&registry, &second));

        let mut control = FakePosixProcessControl::new(Some(first.pid), true, true);
        control.observed_start = Some(unrelated_start());
        let mut kill_attempts = 0;
        settle_drained_detached_children_in_with(&registry, drained, |pending| {
            kill_attempts += 1;
            assert_eq!(pending, &first, "cleanup resolved a replacement by bare PID");
            assert_eq!(pending.start.as_ref(), Some(&owned_start()));
            assert!(!kill_posix_process_tree_with(
                &mut control,
                pending.pid,
                pending.start.as_ref(),
                std::time::Duration::ZERO,
            ));
            false
        });
        assert_eq!(kill_attempts, 1);
        assert!(control.process_signals.is_empty());
        assert!(control.group_signals.is_empty());
        assert!(control.target_running, "generation B was signalled");
        assert!(is_owned(&registry, &second));
        assert_eq!(detached_entry(&registry, &first), (true, false));
    }

    #[test]
    fn b_registration_inside_pending_kill_cannot_replace_drained_a_identity() {
        let first = process_at(4_242, 1, &owned_start());
        let second = process_at(4_242, 2, &unrelated_start());
        let registry = registry_with(&[first.clone()]);
        registry.lock().unwrap().as_mut().unwrap().track(&first);
        let drained = registry.lock().unwrap().as_mut().unwrap().drain_tracked();
        assert_eq!(drained, vec![first.clone()]);

        let mut control = FakePosixProcessControl::new(Some(first.pid), true, true);
        control.observed_start = Some(unrelated_start());
        settle_drained_detached_children_in_with(&registry, drained, |pending| {
            // Interleave reuse after the drain but immediately before the
            // current-start probe. The expected token was already copied.
            registry.lock().unwrap().as_mut().unwrap().record(second.clone());
            assert_eq!(pending.generation, first.generation);
            assert_eq!(pending.start.as_ref(), Some(&owned_start()));
            assert!(!kill_posix_process_tree_with(
                &mut control,
                pending.pid,
                pending.start.as_ref(),
                std::time::Duration::ZERO,
            ));
            false
        });
        assert!(control.process_signals.is_empty());
        assert!(control.group_signals.is_empty());
        assert!(control.target_running);
        assert!(is_owned(&registry, &second));
    }

    #[test]
    fn live_drained_a_with_its_exact_token_is_still_killed() {
        let first = process_at(4_242, 1, &owned_start());
        let second = process_at(4_242, 2, &unrelated_start());
        let registry = registry_with(&[first.clone(), second.clone()]);
        registry.lock().unwrap().as_mut().unwrap().track(&first);

        let mut control = FakePosixProcessControl::new(Some(first.pid), true, true);
        control.observed_start = Some(owned_start());
        kill_tracked_detached_children_in_with(&registry, |pending| {
            assert_eq!(pending, &first);
            kill_posix_process_tree_with(
                &mut control,
                pending.pid,
                pending.start.as_ref(),
                std::time::Duration::ZERO,
            )
        });
        assert_eq!(control.group_signals, vec![first.pid]);
        assert!(!control.target_running);
        assert!(!control.group_live);
        assert_eq!(detached_entry(&registry, &first), (false, false));
        assert!(is_owned(&registry, &second), "successful A cleanup removed B");
    }

    #[test]
    fn posix_dead_or_zombie_leader_still_kills_the_live_owned_group() {
        for target_pgid in [None, Some(4_242)] {
            let mut control = FakePosixProcessControl::new(target_pgid, false, true);
            assert!(control.kill());
            assert_eq!(control.group_signals, vec![4_242]);
            assert!(control.process_signals.is_empty());
            assert!(control.group_probes >= 1);
            assert!(!control.group_live);
        }
    }

    #[test]
    fn posix_reused_setsid_leader_with_same_pid_and_pgid_is_not_signalled() {
        let mut control = FakePosixProcessControl::new(Some(4_242), true, true);
        control.observed_start = Some(unrelated_start());
        assert!(!control.kill());
        assert!(control.group_signals.is_empty());
        assert!(control.process_signals.is_empty());
    }

    #[test]
    fn posix_live_process_with_getpgid_failure_and_reused_pid_is_not_signalled() {
        let mut control = FakePosixProcessControl::new(None, true, false);
        control.observed_start = Some(unrelated_start());
        assert!(!control.kill());
        assert!(control.group_signals.is_empty());
        assert!(control.process_signals.is_empty());
    }

    #[test]
    fn posix_zombie_new_owner_does_not_authenticate_a_live_group() {
        let mut control = FakePosixProcessControl::new(Some(4_242), false, true);
        control.observed_start = Some(unrelated_start());
        assert!(!control.kill());
        assert!(control.group_signals.is_empty());
        assert!(control.process_signals.is_empty());
    }

    #[test]
    fn posix_live_pid_without_a_recorded_start_fails_closed() {
        let mut control = FakePosixProcessControl::new(Some(4_242), true, true);
        assert!(!kill_posix_process_tree_with(
            &mut control,
            4_242,
            None,
            std::time::Duration::ZERO,
        ));
        assert!(control.group_signals.is_empty());
        assert!(control.process_signals.is_empty());
    }

    #[test]
    fn posix_live_registered_setsid_leader_kills_the_owned_group() {
        let mut control = FakePosixProcessControl::new(Some(4_242), true, true);
        assert!(control.kill());
        assert_eq!(control.group_signals, vec![4_242]);
        assert!(control.process_signals.is_empty());
        assert!(!control.group_live);
    }

    #[test]
    fn posix_live_registered_process_survives_getpgid_failure_as_single_target() {
        let mut control = FakePosixProcessControl::new(None, true, false);
        assert!(control.kill());
        assert!(control.group_signals.is_empty());
        assert_eq!(control.process_signals, vec![4_242]);
        assert!(!control.target_running);
    }

    #[test]
    fn posix_reused_leader_in_an_unrelated_group_is_not_signalled() {
        let mut control = FakePosixProcessControl::new(Some(5_150), true, false);
        assert!(!control.kill());
        assert!(control.group_signals.is_empty());
        assert!(control.process_signals.is_empty());
    }

    #[test]
    fn posix_failed_group_signal_never_falls_back_to_a_reused_leader() {
        let mut control = FakePosixProcessControl::new(None, false, true);
        control.group_signal_delivered = false;
        assert!(!control.kill());
        assert_eq!(control.group_signals, vec![4_242]);
        assert!(control.process_signals.is_empty());
        assert!(control.group_live);
    }

    #[test]
    fn posix_surviving_group_member_keeps_cleanup_unsuccessful() {
        let mut control = FakePosixProcessControl::new(None, false, true);
        control.group_survives_signal = true;
        assert!(!control.kill());
        assert_eq!(control.group_signals, vec![4_242]);
        assert!(control.group_live);
    }

    #[test]
    fn posix_absent_leader_and_group_is_already_cleaned_up() {
        let mut control = FakePosixProcessControl::new(None, false, false);
        assert!(control.kill());
        assert!(control.group_signals.is_empty());
        assert!(control.process_signals.is_empty());
        assert_eq!(control.group_probes, 1);
    }

    #[test]
    fn posix_self_group_is_never_group_signalled() {
        let mut control = FakePosixProcessControl::new(Some(1_000), true, true);
        assert!(control.kill());
        assert!(control.group_signals.is_empty());
        assert_eq!(control.process_signals, vec![4_242]);
        assert!(!control.target_running);
    }

    #[test]
    fn posix_nonpositive_identity_is_never_signalled() {
        for pid in [0, -1, -4_242] {
            let mut control = FakePosixProcessControl::new(Some(4_242), true, true);
            assert!(!kill_posix_process_tree_with(
                &mut control,
                pid,
                Some(&owned_start()),
                std::time::Duration::ZERO,
            ));
            assert!(control.group_signals.is_empty());
            assert!(control.process_signals.is_empty());
        }
    }

    #[test]
    fn posix_own_process_is_never_signalled() {
        let mut control = FakePosixProcessControl::new(Some(1_000), true, true);
        control.target_pid = control.own_pid;
        assert!(!control.kill());
        assert!(control.group_signals.is_empty());
        assert!(control.process_signals.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn windows_exit_status_distinguishes_running_from_exited_handles() {
        assert!(windows_exit_code_is_running(STILL_ACTIVE));
        assert!(!windows_exit_code_is_running(0));
        assert!(!windows_exit_code_is_running(1));
    }

}
