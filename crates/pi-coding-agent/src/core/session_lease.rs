//! Port of packages/coding-agent/src/core/session-lease.ts
//!
//! Local stand-ins are used for two helpers that live in other slices:
//! `isProcessAlive` (utils/child-process.ts) and `execFileSyncHidden`
//! (utils/child-process.ts). They are private plumbing inside this module and are
//! recorded in blocked_on.

#![allow(clippy::too_many_arguments)]

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const SESSION_LEASES_ENABLED_ENV: &str = "PRIME_AGENT_INTERNAL_SESSION_LEASES";
pub const SESSION_LEASE_OWNER_ID_ENV: &str = "PRIME_AGENT_INTERNAL_SESSION_LEASE_OWNER_ID";

const LEASE_OWNER_FILE: &str = "owner.json";
const LEASE_GUARD_STALE_MS: u64 = 5000;
const LEASE_GUARD_ATTEMPTS: usize = 100;
const LEASE_GUARD_SLEEP_MS: u64 = 10;
const LEASE_ACQUIRE_ATTEMPTS: usize = 3;

/// `interface SessionLeaseOwner` - `version: 1` is the only accepted value.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionLeaseOwner {
    pub version: i64,
    pub token: String,
    pub pid: i64,
    /// `processStartId?` - absent when the platform probe failed.
    #[serde(rename = "processStartId", skip_serializing_if = "Option::is_none")]
    pub process_start_id: Option<String>,
    /// `activeSessionId?`
    #[serde(rename = "activeSessionId", skip_serializing_if = "Option::is_none")]
    pub active_session_id: Option<String>,
    #[serde(rename = "sessionPath")]
    pub session_path: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

/// `SessionAlreadyActiveError extends Error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionAlreadyActiveError {
    pub session_path: String,
    pub active_session_id: Option<String>,
}

impl SessionAlreadyActiveError {
    pub const CODE: &'static str = "session_already_active";

    pub fn new(session_path: impl Into<String>, active_session_id: Option<String>) -> Self {
        Self {
            session_path: session_path.into(),
            active_session_id,
        }
    }
}

impl std::fmt::Display for SessionAlreadyActiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.active_session_id {
            Some(active_session_id) => write!(
                f,
                "Session is already active in {active_session_id}: {}",
                self.session_path
            ),
            None => write!(
                f,
                "Session is already active in another process: {}",
                self.session_path
            ),
        }
    }
}

impl std::error::Error for SessionAlreadyActiveError {}

#[derive(Debug)]
pub struct SessionLease {
    pub session_path: String,
    directory: String,
    token: String,
    released: bool,
}

impl SessionLease {
    fn new(session_path: String, directory: String, token: String) -> Self {
        Self {
            session_path,
            directory,
            token,
            released: false,
        }
    }

    pub fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        let directory = self.directory.clone();
        let token = self.token.clone();
        let result = with_lease_guard(&directory, || {
            if let Ok(LeaseOwnerState::Owner(owner)) = read_lease_owner(&directory) {
                if owner.token == token {
                    reclaim_stale_lease(&directory);
                }
            }
        });
        // Lease cleanup is best-effort. A stale owner is reclaimed by the next process.
        let _ = result;
    }
}

fn leases_enabled(lookup: &dyn Fn(&str) -> Option<String>) -> bool {
    let value = lookup(SESSION_LEASES_ENABLED_ENV)
        .unwrap_or_default()
        .to_lowercase();
    value == "1" || value == "true" || value == "yes"
}

fn env_lookup(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn lease_directory(agent_dir: &str, session_path: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(session_path.as_bytes());
    let digest = hasher.finalize();
    let mut key = String::with_capacity(digest.len() * 2);
    for byte in digest {
        key.push_str(&format!("{byte:02x}"));
    }
    Path::new(agent_dir)
        .join("session-leases")
        .join(format!("{key}.lock"))
        .to_string_lossy()
        .to_string()
}

pub fn canonical_session_path(session_path: &str) -> String {
    let resolved_path = resolve_path(session_path);
    if let Ok(canonical) = plain_realpath(Path::new(&resolved_path)) {
        return canonical;
    }
    let parent = dirname(&resolved_path);
    if let Ok(canonical_parent) = plain_realpath(Path::new(&parent)) {
        return Path::new(&canonical_parent)
            .join(basename(&resolved_path))
            .to_string_lossy()
            .to_string();
    }
    resolved_path
}

/// TS `realpathIfPresentSync` resolves through `realpathSync`, which returns
/// plain win32 paths; std canonicalize emits verbatim `\\?\` paths on Windows.
/// Strip the verbatim prefix so both runtimes share one cross-tool identity.
fn plain_realpath(path: &Path) -> std::io::Result<String> {
    let canonical = std::fs::canonicalize(path)?;
    let text = canonical.to_string_lossy().to_string();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return Ok(format!(r"\\{rest}"));
    }
    if let Some(rest) = text.strip_prefix(r"\\?\") {
        return Ok(rest.to_string());
    }
    Ok(text)
}

fn resolve_path(path: &str) -> String {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate.to_string_lossy().to_string();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(candidate).to_string_lossy().to_string(),
        Err(_) => candidate.to_string_lossy().to_string(),
    }
}

fn basename(value: &str) -> String {
    Path::new(value)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn dirname(value: &str) -> String {
    Path::new(value)
        .parent()
        .map(|parent| {
            let text = parent.to_string_lossy().to_string();
            if text.is_empty() {
                ".".to_string()
            } else {
                text
            }
        })
        .unwrap_or_else(|| ".".to_string())
}

/// `readLeaseOwner` result: an owner, or the two non-owner sentinels.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LeaseOwnerState {
    Owner(SessionLeaseOwner),
    Absent,
    Unreadable,
}

/// An unreadable owner may hold a live lease. Only a missing owner is safely absent.
fn read_lease_owner(directory: &str) -> Result<LeaseOwnerState, String> {
    let owner_path = Path::new(directory).join(LEASE_OWNER_FILE);
    let owner_path_text = owner_path.to_string_lossy().to_string();
    let raw = match std::fs::read_to_string(&owner_path) {
        Ok(raw) => raw,
        Err(error) => {
            return Ok(if error.kind() == io::ErrorKind::NotFound {
                LeaseOwnerState::Absent
            } else {
                LeaseOwnerState::Unreadable
            })
        }
    };
    let parsed: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(parsed) => parsed,
        Err(error) => {
            return Err(format!(
                "Corrupt session lease owner file: {owner_path_text} - {error}"
            ))
        }
    };
    let valid = parsed.get("version").and_then(|v| v.as_i64()) == Some(1)
        && parsed.get("token").map(|v| v.is_string()).unwrap_or(false)
        && parsed.get("pid").map(|v| v.is_i64()).unwrap_or(false)
        && parsed
            .get("sessionPath")
            .map(|v| v.is_string())
            .unwrap_or(false)
        && parsed
            .get("createdAt")
            .map(|v| v.is_string())
            .unwrap_or(false);
    if !valid {
        return Err(format!(
            "Corrupt session lease owner file: {owner_path_text} - missing or invalid required fields",
        ));
    }
    match serde_json::from_value::<SessionLeaseOwner>(parsed) {
        Ok(owner) => Ok(LeaseOwnerState::Owner(owner)),
        Err(error) => Err(format!(
            "Corrupt session lease owner file: {owner_path_text} - {error}"
        )),
    }
}



// ---------------------------------------------------------------------------
// Process start identity
// ---------------------------------------------------------------------------

type ProcessQuery = dyn Fn(&str, &[String], Option<&[(String, String)]>) -> Result<String, String>;

fn run_process_query(
    command: &str,
    args: &[String],
    env: Option<&[(String, String)]>,
) -> Result<String, String> {
    let mut process = Command::new(command);
    process.args(args);
    process.stdin(std::process::Stdio::null());
    process.stdout(std::process::Stdio::piped());
    process.stderr(std::process::Stdio::null());
    if let Some(env) = env {
        for (key, value) in env {
            process.env(key, value);
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        process.creation_flags(CREATE_NO_WINDOW);
    }
    match process.output() {
        Ok(output) => Ok(String::from_utf8_lossy(&output.stdout).to_string()),
        Err(error) => Err(error.to_string()),
    }
}

pub fn get_windows_process_start_id(pid: i64, query: Option<&ProcessQuery>) -> Option<String> {
    if pid <= 0 || pid > u32::MAX as i64 {
        return None;
    }
    #[cfg(windows)]
    if query.is_none() {
        return windows_process_start_id_native(pid as u32);
    }
    let script = format!(
        "([System.Diagnostics.Process]::GetProcessById({pid})).StartTime.ToUniversalTime().Ticks"
    );
    let args = vec![
        "-NoLogo".to_string(),
        "-NoProfile".to_string(),
        "-NonInteractive".to_string(),
        "-Command".to_string(),
        script,
    ];
    let default_query = run_process_query;
    let query = query.unwrap_or(&default_query);
    let start_ticks = query("powershell.exe", &args, None)
        .ok()?
        .trim()
        .to_string();
    if !start_ticks.is_empty() && start_ticks.bytes().all(|b| b.is_ascii_digit()) {
        Some(format!("win:{start_ticks}"))
    } else {
        None
    }
}

#[cfg(windows)]
fn windows_process_start_id_native(pid: u32) -> Option<String> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    // FILETIME counts 100ns since 1601; existing persisted identities use .NET
    // ticks since 0001. Keep that exact format so leases survive this upgrade.
    const DOTNET_FILETIME_OFFSET: u64 = 504_911_232_000_000_000;
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() { return None; }
        let mut created = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let mut exited = created;
        let mut kernel = created;
        let mut user = created;
        let ok = GetProcessTimes(handle, &mut created, &mut exited, &mut kernel, &mut user);
        CloseHandle(handle);
        if ok == 0 { return None; }
        let ticks = ((created.dwHighDateTime as u64) << 32) | created.dwLowDateTime as u64;
        ticks.checked_add(DOTNET_FILETIME_OFFSET).map(|ticks| format!("win:{ticks}"))
    }
}

pub fn get_ps_process_start_id(pid: i64, query: Option<&ProcessQuery>) -> Option<String> {
    if pid <= 0 {
        return None;
    }
    // `lstart` is rendered in the subprocess timezone and locale, so pin both for a durable identity.
    let env: Vec<(String, String)> = vec![
        ("LC_ALL".to_string(), "C".to_string()),
        ("LC_TIME".to_string(), "C".to_string()),
        ("LANG".to_string(), "C".to_string()),
        ("TZ".to_string(), "UTC".to_string()),
    ];
    let args = vec![
        "-p".to_string(),
        pid.to_string(),
        "-o".to_string(),
        "lstart=".to_string(),
    ];
    let default_query = run_process_query;
    let query = query.unwrap_or(&default_query);
    let start_time = query("ps", &args, Some(&env)).ok()?.trim().to_string();
    if start_time.is_empty() {
        None
    } else {
        Some(format!("ps:{start_time}"))
    }
}

pub fn get_process_start_id(pid: i64) -> Option<String> {
    if pid <= 0 {
        return None;
    }
    if cfg!(windows) {
        return get_windows_process_start_id(pid, None);
    }
    if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        if let Some(command_end) = stat.rfind(')') {
            let fields: Vec<&str> = stat[command_end + 2..].split(' ').collect();
            if let Some(start_time) = fields.get(19) {
                if !start_time.is_empty() {
                    return Some(format!("proc:{start_time}"));
                }
            }
        }
    }
    // Fall through to the portable process listing used on macOS and BSD.
    get_ps_process_start_id(pid, None)
}

fn get_current_process_start_id() -> Option<String> {
    static CURRENT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    CURRENT
        .get_or_init(|| get_process_start_id(std::process::id() as i64))
        .clone()
}

/// Local stand-in for `isProcessAlive` from utils/child-process.ts (other slice).
fn is_process_alive(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
            return true;
        }
        return io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
    }
    #[cfg(windows)]
    {
        u32::try_from(pid).is_ok_and(windows_process_id_exists)
    }
}

#[cfg(windows)]
fn windows_process_id_exists(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    const STILL_ACTIVE: u32 = 259;
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            // Access denied (or another probe failure) is not proof of death.
            // Only the documented nonexistent-PID result permits reclamation.
            return windows_process_probe_may_be_alive(Some(GetLastError()), None);
        }
        let mut exit_code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut exit_code) != 0;
        CloseHandle(handle);
        windows_process_probe_may_be_alive(None, ok.then_some(exit_code == STILL_ACTIVE))
    }
}

#[cfg(windows)]
fn windows_process_probe_may_be_alive(open_error: Option<u32>, active: Option<bool>) -> bool {
    use windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER;
    if let Some(error) = open_error {
        return error != ERROR_INVALID_PARAMETER;
    }
    active.unwrap_or(true)
}

fn is_lease_owner_alive(owner: &SessionLeaseOwner) -> bool {
    if !is_process_alive(owner.pid) {
        return false;
    }
    let Some(owner_start_id) = owner.process_start_id.as_ref() else {
        return true;
    };
    match get_process_start_id(owner.pid) {
        None => true,
        Some(current_start_id) => &current_start_id == owner_start_id,
    }
}

// ---------------------------------------------------------------------------
// Guard lock (port of proper-lockfile lockSync with a fixed lockfilePath)
// ---------------------------------------------------------------------------

fn sleep_ms(ms: u64) {
    std::thread::sleep(Duration::from_millis(ms));
}

fn modified_millis(path: &Path) -> Option<u64> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    modified
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Release handle for the guard directory; mirrors proper-lockfile's release().
struct LeaseGuard {
    path: PathBuf,
    compromised: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl LeaseGuard {
    fn release(&self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn acquire_lease_guard(directory: &str) -> Result<LeaseGuard, String> {
    let guard_path = PathBuf::from(format!("{directory}.guard"));
    let compromised = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    for attempt in 0..LEASE_GUARD_ATTEMPTS {
        match std::fs::create_dir(&guard_path) {
            Ok(()) => {
                return Ok(LeaseGuard {
                    path: guard_path,
                    compromised,
                })
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                // Stale guard: proper-lockfile removes a lock older than `stale` ms.
                if let Some(modified) = modified_millis(&guard_path) {
                    if now_millis().saturating_sub(modified) > LEASE_GUARD_STALE_MS {
                        let _ = std::fs::remove_dir_all(&guard_path);
                        continue;
                    }
                }
                if attempt == LEASE_GUARD_ATTEMPTS - 1 {
                    return Err(format!("Could not coordinate session lease: {directory}"));
                }
                sleep_ms(LEASE_GUARD_SLEEP_MS);
            }
            Err(error) => return Err(error.to_string()),
        }
    }
    Err(format!("Could not coordinate session lease: {directory}"))
}

fn with_lease_guard<T>(directory: &str, action: impl FnOnce() -> T) -> Result<T, String> {
    let guard = acquire_lease_guard(directory)?;
    with_acquired_lease_guard(directory, guard, action)
}

/// Opportunistic cleanup must not wait behind or evict a contended guard. The
/// normal acquire/release paths retain their bounded coordination retries.
fn try_with_lease_guard<T>(directory: &str, action: impl FnOnce() -> T) -> Result<T, String> {
    let path = PathBuf::from(format!("{directory}.guard"));
    std::fs::create_dir(&path).map_err(|error| error.to_string())?;
    let guard = LeaseGuard {
        path,
        compromised: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    with_acquired_lease_guard(directory, guard, action)
}

fn with_acquired_lease_guard<T>(
    directory: &str,
    guard: LeaseGuard,
    action: impl FnOnce() -> T,
) -> Result<T, String> {
    let assert_guard_held = || -> Result<(), String> {
        if guard.compromised.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(format!("Session lease guard was compromised: {directory}"));
        }
        Ok(())
    };
    let result = (|| -> Result<T, String> {
        assert_guard_held()?;
        let value = action();
        assert_guard_held()?;
        Ok(value)
    })();
    if guard.compromised.load(std::sync::atomic::Ordering::SeqCst) {
        // The compromised guard no longer owns a lock that can be safely released.
    } else {
        guard.release();
    }
    result
}

// ---------------------------------------------------------------------------
// Rename contention and stale reclamation
// ---------------------------------------------------------------------------

/// Map an io error to the POSIX-ish code string Node reports (`error.code`).
fn errno_code(error: &io::Error) -> Option<String> {
    if error.kind() == io::ErrorKind::AlreadyExists {
        return Some("EEXIST".to_string());
    }
    if error.kind() == io::ErrorKind::NotFound {
        return Some("ENOENT".to_string());
    }
    if error.kind() == io::ErrorKind::PermissionDenied {
        return Some("EPERM".to_string());
    }
    let raw = error.raw_os_error()?;
    #[cfg(windows)]
    {
        // libuv's Windows error translation for the codes this port observes.
        return Some(
            match raw {
                2 | 3 => "ENOENT",
                5 => "EACCES",
                32 | 33 => "EBUSY",
                183 => "EEXIST",
                145 => "ENOTEMPTY",
                _ => return None,
            }
            .to_string(),
        );
    }
    #[cfg(unix)]
    {
        match raw {
            libc::ENOTEMPTY => Some("ENOTEMPTY".to_string()),
            libc::EBUSY => Some("EBUSY".to_string()),
            _ => None,
        }
    }
}

pub fn is_rename_target_contention(directory: &str, code: Option<&str>, platform: &str) -> bool {
    // POSIX: renameSync into an existing directory raises EEXIST or ENOTEMPTY.
    if code == Some("EEXIST") || code == Some("ENOTEMPTY") {
        return true;
    }
    // Windows: renameSync into an existing directory raises EPERM or EACCES
    // instead of EEXIST.  Only treat them as contention when the target
    // actually exists so real permission errors still propagate.
    if (code == Some("EPERM") || code == Some("EACCES")) && platform == "win32" {
        return Path::new(directory).exists();
    }
    false
}

fn reclaim_stale_lease(directory: &str) -> bool {
    let stale_path = format!(
        "{directory}.stale-{}-{}",
        std::process::id(),
        Uuid::new_v4()
    );
    let mut attempt = 1usize;
    loop {
        match std::fs::rename(directory, &stale_path) {
            Ok(()) => break,
            Err(error) => {
                let code = errno_code(&error);
                if code.as_deref() == Some("ENOENT") {
                    return true;
                }
                let transient = cfg!(windows)
                    && matches!(
                        code.as_deref(),
                        Some("EBUSY") | Some("EPERM") | Some("EACCES")
                    );
                if !transient || attempt >= 8 {
                    return false;
                }
                sleep_ms(10 * attempt as u64);
                attempt += 1;
            }
        }
    }
    // The quarantined directory no longer owns the lease path.
    let _ = std::fs::remove_dir_all(&stale_path);
    true
}

// ---------------------------------------------------------------------------
// Acquisition
// ---------------------------------------------------------------------------

pub fn acquire_session_lease(
    session_path: Option<&str>,
    agent_dir: &str,
    environment: Option<&[(String, String)]>,
) -> Result<Option<SessionLease>, AcquireSessionLeaseError> {
    let session_path = match session_path {
        Some(session_path) if !session_path.is_empty() => session_path,
        _ => return Ok(None),
    };
    let enabled = match environment {
        Some(environment) => {
            let lookup = |name: &str| {
                environment
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|(_, value)| value.clone())
            };
            leases_enabled(&lookup)
        }
        None => leases_enabled(&env_lookup),
    };
    if !enabled {
        return Ok(None);
    }

    // TS realpathIfPresentSync rethrows every non-ENOENT canonicalization error;
    // only ENOENT falls back (atomic-file.ts:271-278).
    let resolved_path = resolve_path(session_path);
    if let Err(error) = std::fs::canonicalize(&resolved_path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            return Err(AcquireSessionLeaseError::Other(error.to_string()));
        }
    }
    let canonical_path = canonical_session_path(session_path);
    let root = Path::new(agent_dir).join("session-leases");
    std::fs::create_dir_all(&root).map_err(|error| AcquireSessionLeaseError::Other(error.to_string()))?;
    let directory = lease_directory(agent_dir, &canonical_path);
    let owner_id = match environment {
        Some(environment) => environment
            .iter()
            .find(|(key, _)| key == SESSION_LEASE_OWNER_ID_ENV)
            .map(|(_, value)| value.clone()),
        None => std::env::var(SESSION_LEASE_OWNER_ID_ENV).ok(),
    };

    let acquired = with_lease_guard(&directory, || {
        let read_owner_error = |message: String| AcquireSessionLeaseError::Other(message);
        for _attempt in 0..LEASE_ACQUIRE_ATTEMPTS {
            let token = Uuid::new_v4().to_string();
            let candidate_directory =
                format!("{directory}.candidate-{}-{token}", std::process::id());
            let owner = SessionLeaseOwner {
                version: 1,
                token: token.clone(),
                pid: std::process::id() as i64,
                process_start_id: get_current_process_start_id(),
                active_session_id: owner_id.clone(),
                session_path: canonical_path.clone(),
                created_at: iso_now(),
            };
            std::fs::create_dir_all(&candidate_directory)
                .map_err(|error| AcquireSessionLeaseError::Other(error.to_string()))?;
            let owner_path = Path::new(&candidate_directory).join(LEASE_OWNER_FILE);
            let body = format!(
                "{}\n",
                serde_json::to_string_pretty(&owner)
                    .map_err(|error| AcquireSessionLeaseError::Other(error.to_string()))?
            );
            std::fs::write(&owner_path, body)
                .map_err(|error| AcquireSessionLeaseError::Other(error.to_string()))?;
            match std::fs::rename(&candidate_directory, &directory) {
                Ok(()) => {
                    return Ok(Some(SessionLease::new(
                        canonical_path.clone(),
                        directory.clone(),
                        token,
                    )));
                }
                Err(error) => {
                    let _ = std::fs::remove_dir_all(&candidate_directory);
                    let code = errno_code(&error);
                    if code.as_deref() == Some("ENOENT") {
                        // Candidate vanished - treat as retryable race.
                        continue;
                    }
                    if is_rename_target_contention(&directory, code.as_deref(), platform_name()) {
                        let existing_owner =
                            read_lease_owner(&directory).map_err(read_owner_error)?;
                        if existing_owner == LeaseOwnerState::Unreadable {
                            continue;
                        }
                        if let LeaseOwnerState::Owner(existing_owner) = &existing_owner {
                            if is_lease_owner_alive(existing_owner) {
                                return Err(AcquireSessionLeaseError::AlreadyActive(
                                    SessionAlreadyActiveError::new(
                                        canonical_path.clone(),
                                        existing_owner.active_session_id.clone(),
                                    ),
                                ));
                            }
                        }
                        reclaim_stale_lease(&directory);
                        continue;
                    }
                    return Err(AcquireSessionLeaseError::Other(error.to_string()));
                }
            }
        }

        let owner = if Path::new(&directory).exists() {
            read_lease_owner(&directory).map_err(read_owner_error)?
        } else {
            LeaseOwnerState::Absent
        };
        if let LeaseOwnerState::Owner(owner) = &owner {
            if is_lease_owner_alive(owner) {
                return Err(AcquireSessionLeaseError::AlreadyActive(
                    SessionAlreadyActiveError::new(
                        canonical_path.clone(),
                        owner.active_session_id.clone(),
                    ),
                ));
            }
        }
        Err(AcquireSessionLeaseError::Other(format!(
            "Could not acquire session lease: {canonical_path}"
        )))
    })
    .map_err(AcquireSessionLeaseError::Other)?;
    acquired
}

/// Result of a bounded dead-owner lease sweep (audit D-08).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SessionLeaseSweepResult {
    pub scanned: usize,
    pub reclaimed: Vec<String>,
    /// Directories the sweep could not resolve (unreadable or corrupt owner
    /// files, guard coordination failures). Never reclaimed; only counted.
    pub unreadable_owners: usize,
}

/// Upper bound on lease directories examined per sweep, so a large directory
/// cannot turn startup cleanup into an unbounded pass.
const SESSION_LEASE_SWEEP_MAX_DIRECTORIES: usize = 256;
const SESSION_LEASE_SWEEP_MAX_DURATION: Duration = Duration::from_millis(250);
/// A lease younger than this is left for a later sweep, so a concurrently
/// starting owner is never raced by the reclamation window.
const SESSION_LEASE_SWEEP_MIN_AGE_MS: u64 = 30_000;

/// Bounded reclamation of dead-owner session leases: a `session-leases/*.lock`
/// directory whose recorded owner process is verifiably dead (or that has no
/// owner file at all) is reclaimed; the sweep never kills a process, never
/// touches directories it could not prove dead, and leaves unreadable owner
/// files for a manual pass while only counting them.
pub fn sweep_dead_owner_leases(agent_dir: &str) -> SessionLeaseSweepResult {
    let mut result = SessionLeaseSweepResult::default();
    let root = Path::new(agent_dir).join("session-leases");
    let Ok(entries) = std::fs::read_dir(&root) else {
        return result;
    };
    let started = std::time::Instant::now();
    // Count every directory-entry attempt, not only successfully parsed owners:
    // an all-corrupt or all-contended directory must have the same work bound.
    for entry in entries.take(SESSION_LEASE_SWEEP_MAX_DIRECTORIES) {
        if started.elapsed() >= SESSION_LEASE_SWEEP_MAX_DURATION {
            break;
        }
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("lock") {
            continue;
        }
        if !path.is_dir() {
            continue;
        }
        let directory = path.to_string_lossy().to_string();
        let mut unreadable = false;
        let claimed = try_with_lease_guard(&directory, || {
            let owner = match read_lease_owner(&directory) {
                Ok(LeaseOwnerState::Owner(owner)) => owner,
                Ok(LeaseOwnerState::Absent) => return Ok(reclaim_stale_lease(&directory)),
                Ok(LeaseOwnerState::Unreadable) | Err(_) => {
                    unreadable = true;
                    return Err("unreadable session lease owner".to_string());
                }
            };
            // Age floor: a fresh lease is left for its owner, dead or not, so a
            // restarting process that reused the pid is never evicted mid-start.
            if parse_lease_created_at(&owner.created_at)
                .is_some_and(|created_at| now_millis().saturating_sub(created_at) < SESSION_LEASE_SWEEP_MIN_AGE_MS)
            {
                return Ok(false);
            }
            if is_lease_owner_alive(&owner) {
                return Ok(false);
            }
            Ok(reclaim_stale_lease(&directory))
        });
        // Only directories the sweep could actually examine count as scanned;
        // unreadable/corrupt owners and guard-coordination failures may hold a
        // live lease: separately counted, never reclaimed.
        if unreadable || claimed.is_err() {
            result.unreadable_owners += 1;
        } else {
            result.scanned += 1;
        }
        match claimed {
            Ok(Ok(true)) => result.reclaimed.push(basename(&directory)),
            Ok(Ok(false)) => {}
            Ok(Err(_)) | Err(_) => {}
        }
    }
    result
}

fn parse_lease_created_at(value: &str) -> Option<u64> {
    let millis = crate::core::cron_jobs::parse_iso_date(value);
    millis.is_finite().then_some(millis as u64)
}

fn platform_name() -> &'static str {
    if cfg!(windows) {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    }
}

/// `new Date().toISOString()`.
fn iso_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    let days = secs / 86_400;
    let seconds_of_day = secs % 86_400;
    let (year, month, day) = civil_from_days(days as i64);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        seconds_of_day / 3600,
        (seconds_of_day % 3600) / 60,
        seconds_of_day % 60
    )
}

/// Howard Hinnant's civil_from_days.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Errors surfaced by `acquireSessionLease`: the typed already-active error plus
/// the plain `Error` cases the TypeScript throws for coordination failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcquireSessionLeaseError {
    AlreadyActive(SessionAlreadyActiveError),
    Other(String),
}

impl std::fmt::Display for AcquireSessionLeaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquireSessionLeaseError::AlreadyActive(error) => error.fmt(f),
            AcquireSessionLeaseError::Other(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for AcquireSessionLeaseError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn leases_are_disabled_by_default() {
        let environment = env(&[]);
        let lookup = |name: &str| {
            environment
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        };
        assert!(!leases_enabled(&lookup));
        for value in ["1", "true", "TRUE", "yes", "Yes"] {
            let environment = env(&[(SESSION_LEASES_ENABLED_ENV, value)]);
            let lookup = |name: &str| {
                environment
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|(_, value)| value.clone())
            };
            assert!(leases_enabled(&lookup), "{value}");
        }
        let environment = env(&[(SESSION_LEASES_ENABLED_ENV, "0")]);
        let lookup = |name: &str| {
            environment
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        };
        assert!(!leases_enabled(&lookup));
    }

    #[test]
    fn lease_directory_is_a_sha256_lock_name() {
        let dir = lease_directory("/agent", "/sessions/a.jsonl");
        assert!(dir.starts_with(
            Path::new("/agent")
                .join("session-leases")
                .to_string_lossy()
                .as_ref()
        ));
        assert!(dir.ends_with(".lock"));
        let other = lease_directory("/agent", "/sessions/b.jsonl");
        assert_ne!(dir, other);
        assert_eq!(dir, lease_directory("/agent", "/sessions/a.jsonl"));
    }

    #[test]
    fn acquires_then_detects_a_live_owner_and_releases() {
        let temp = tempfile::tempdir().unwrap();
        let agent_dir = temp.path().to_string_lossy().to_string();
        let session = temp.path().join("sessions").join("a.jsonl");
        std::fs::create_dir_all(session.parent().unwrap()).unwrap();
        std::fs::write(&session, b"").unwrap();
        let session_path = session.to_string_lossy().to_string();
        let environment = env(&[(SESSION_LEASES_ENABLED_ENV, "1")]);

        let mut lease = acquire_session_lease(Some(&session_path), &agent_dir, Some(&environment))
            .unwrap()
            .expect("lease acquired");
        assert_eq!(lease.session_path, canonical_session_path(&session_path));

        let second = acquire_session_lease(Some(&session_path), &agent_dir, Some(&environment));
        let error = second.err().expect("second acquisition must fail");
        match error {
            AcquireSessionLeaseError::AlreadyActive(error) => {
                assert_eq!(error.session_path, canonical_session_path(&session_path));
                assert_eq!(error.to_string(), format!("Session is already active in another process: {}", error.session_path));
            }
            other => panic!("expected already-active, got {other:?}"),
        }

        lease.release();
        let third = acquire_session_lease(Some(&session_path), &agent_dir, Some(&environment))
            .unwrap()
            .expect("lease reacquired after release");
        drop(third);
    }

    #[test]
    fn disabled_or_missing_paths_return_none() {
        let temp = tempfile::tempdir().unwrap();
        let agent_dir = temp.path().to_string_lossy().to_string();
        assert!(acquire_session_lease(None, &agent_dir, Some(&env(&[])))
            .unwrap()
            .is_none());
        assert!(
            acquire_session_lease(Some("/sessions/a.jsonl"), &agent_dir, Some(&env(&[])))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn rename_contention_matches_platform_codes() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().to_string_lossy().to_string();
        assert!(is_rename_target_contention(&dir, Some("EEXIST"), "linux"));
        assert!(is_rename_target_contention(
            &dir,
            Some("ENOTEMPTY"),
            "linux"
        ));
        assert!(is_rename_target_contention(&dir, Some("EPERM"), "win32"));
        assert!(!is_rename_target_contention(&dir, Some("EPERM"), "linux"));
        assert!(!is_rename_target_contention(
            "/definitely/missing",
            Some("EPERM"),
            "win32"
        ));
        assert!(!is_rename_target_contention(&dir, None, "win32"));
        #[cfg(unix)]
        assert_eq!(errno_code(&io::Error::from_raw_os_error(libc::ENOTEMPTY)).as_deref(), Some("ENOTEMPTY"));
    }

    #[test]
    fn reads_owner_files_with_the_three_states() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().to_string_lossy().to_string();
        assert_eq!(read_lease_owner(&dir).unwrap(), LeaseOwnerState::Absent);

        let owner = SessionLeaseOwner {
            version: 1,
            token: "t".to_string(),
            pid: std::process::id() as i64,
            process_start_id: None,
            active_session_id: Some("sid".to_string()),
            session_path: "/sessions/a.jsonl".to_string(),
            created_at: "2026-01-01T00:00:00.000Z".to_string(),
        };
        std::fs::write(
            Path::new(&dir).join(LEASE_OWNER_FILE),
            format!("{}\n", serde_json::to_string_pretty(&owner).unwrap()),
        )
        .unwrap();
        match read_lease_owner(&dir).unwrap() {
            LeaseOwnerState::Owner(read) => assert_eq!(read, owner),
            other => panic!("expected owner, got {other:?}"),
        }
        assert!(is_lease_owner_alive(&owner));

        std::fs::write(Path::new(&dir).join(LEASE_OWNER_FILE), "{not json").unwrap();
        assert!(read_lease_owner(&dir).is_err());
        std::fs::write(Path::new(&dir).join(LEASE_OWNER_FILE), "{\"version\":2}").unwrap();
        assert!(read_lease_owner(&dir).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn backlog_native_process_identity_matches_persisted_powershell_ticks() {
        let pid = std::process::id() as i64;
        let legacy_started = std::time::Instant::now();
        let legacy = get_windows_process_start_id(pid, Some(&run_process_query)).expect("PowerShell identity");
        let legacy_elapsed = legacy_started.elapsed();
        let native_started = std::time::Instant::now();
        for _ in 0..100 {
            assert_eq!(get_windows_process_start_id(pid, None).as_deref(), Some(legacy.as_str()));
        }
        eprintln!("identity lookup: legacy one={legacy_elapsed:?}; native hundred={:?}", native_started.elapsed());
        assert_eq!(get_windows_process_start_id(0, None), None);
        assert_eq!(get_windows_process_start_id(-1, None), None);
        assert_eq!(get_windows_process_start_id(u32::MAX as i64 + 1, None), None);
        assert_eq!(get_windows_process_start_id(u32::MAX as i64, None), None);
    }

    #[test]
    fn sweep_reclaims_only_dead_owner_leases() {
        let temp = tempfile::tempdir().unwrap();
        let agent_dir = temp.path().to_string_lossy().to_string();
        let root = Path::new(&agent_dir).join("session-leases");
        std::fs::create_dir_all(&root).unwrap();

        let dead_session = temp.path().join("dead.jsonl");
        std::fs::write(&dead_session, b"").unwrap();
        let dead_owner = SessionLeaseOwner {
            version: 1,
            token: "dead".to_string(),
            pid: 999_999_999,
            process_start_id: None,
            active_session_id: Some("dead".to_string()),
            session_path: canonical_session_path(&dead_session.to_string_lossy()),
            // Old enough to pass the sweep's age floor.
            created_at: "2026-01-01T00:00:00.000Z".to_string(),
        };
        let dead_dir = lease_directory(&agent_dir, &canonical_session_path(&dead_session.to_string_lossy()));
        std::fs::create_dir_all(&dead_dir).unwrap();
        std::fs::write(
            Path::new(&dead_dir).join(LEASE_OWNER_FILE),
            format!("{}\n", serde_json::to_string_pretty(&dead_owner).unwrap()),
        )
        .unwrap();

        let live_session = temp.path().join("live.jsonl");
        std::fs::write(&live_session, b"").unwrap();
        let live_owner = SessionLeaseOwner {
            version: 1,
            token: "live".to_string(),
            pid: std::process::id() as i64,
            process_start_id: get_current_process_start_id(),
            active_session_id: Some("live".to_string()),
            session_path: canonical_session_path(&live_session.to_string_lossy()),
            created_at: "2026-01-01T00:00:00.000Z".to_string(),
        };
        let live_dir = lease_directory(&agent_dir, &canonical_session_path(&live_session.to_string_lossy()));
        std::fs::create_dir_all(&live_dir).unwrap();
        std::fs::write(
            Path::new(&live_dir).join(LEASE_OWNER_FILE),
            format!("{}\n", serde_json::to_string_pretty(&live_owner).unwrap()),
        )
        .unwrap();

        // A lease with a fresh (recent) dead owner is left for a later sweep.
        let fresh_dir = lease_directory(&agent_dir, "Z:\\fresh-never-held.jsonl");
        std::fs::create_dir_all(&fresh_dir).unwrap();
        let fresh_owner = SessionLeaseOwner { created_at: iso_now(), ..dead_owner.clone() };
        std::fs::write(
            Path::new(&fresh_dir).join(LEASE_OWNER_FILE),
            format!("{}\n", serde_json::to_string_pretty(&fresh_owner).unwrap()),
        )
        .unwrap();

        // An abandoned directory (no owner file) is reclaimable.
        let absent_dir = lease_directory(&agent_dir, "Z:\\abandoned.jsonl");
        std::fs::create_dir_all(&absent_dir).unwrap();
        // An unreadable owner is counted and left alone.
        let corrupt_dir = lease_directory(&agent_dir, "Z:\\corrupt.jsonl");
        std::fs::create_dir_all(&corrupt_dir).unwrap();
        std::fs::write(Path::new(&corrupt_dir).join(LEASE_OWNER_FILE), "{not json").unwrap();

        let result = sweep_dead_owner_leases(&agent_dir);
        assert_eq!(result.scanned, 4, "{result:?}");
        assert_eq!(result.reclaimed.len(), 2, "{result:?}");
        assert!(result.reclaimed.contains(&basename(&dead_dir)));
        assert!(result.reclaimed.contains(&basename(&absent_dir)));
        assert_eq!(result.unreadable_owners, 1, "{result:?}");
        let exists = |dir: &str| Path::new(dir).exists();
        assert!(!exists(&dead_dir));
        assert!(!exists(&absent_dir));
        assert!(exists(&live_dir), "a live owner's lease must survive the sweep");
        assert!(exists(&fresh_dir), "a fresh dead-owner lease is left for a later sweep");
        assert!(exists(&corrupt_dir), "an unreadable owner is never reclaimed");
        std::fs::remove_dir_all(&live_dir).unwrap();
        std::fs::remove_dir_all(&fresh_dir).unwrap();
        std::fs::remove_dir_all(&corrupt_dir).unwrap();
    }

    #[test]
    fn sweep_bounds_unreadable_entries_instead_of_only_successful_scans() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("session-leases");
        std::fs::create_dir_all(&root).unwrap();
        for index in 0..SESSION_LEASE_SWEEP_MAX_DIRECTORIES + 20 {
            let dir = root.join(format!("corrupt-{index}.lock"));
            std::fs::create_dir(&dir).unwrap();
            std::fs::write(dir.join(LEASE_OWNER_FILE), b"{unreadable owner").unwrap();
        }
        let result = sweep_dead_owner_leases(&temp.path().to_string_lossy());
        assert_eq!(result.scanned, 0);
        assert!(result.unreadable_owners > 0);
        assert!(result.unreadable_owners <= SESSION_LEASE_SWEEP_MAX_DIRECTORIES, "{result:?}");
        assert!(result.reclaimed.is_empty());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), SESSION_LEASE_SWEEP_MAX_DIRECTORIES + 20);
    }

    #[test]
    fn sweep_never_waits_on_or_reclaims_a_contended_guard() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("session-leases");
        std::fs::create_dir_all(&root).unwrap();
        let mut guards = Vec::new();
        for index in 0..12 {
            let dir = root.join(format!("contended-{index}.lock"));
            std::fs::create_dir(&dir).unwrap();
            // Even ownerless leases are protected by another writer's guard.
            let guard = PathBuf::from(format!("{}.guard", dir.display()));
            std::fs::create_dir(&guard).unwrap();
            guards.push((dir, guard));
        }
        let started = std::time::Instant::now();
        let result = sweep_dead_owner_leases(&temp.path().to_string_lossy());
        assert!(started.elapsed() < Duration::from_secs(2), "cleanup waited on a contended guard");
        assert_eq!(result.scanned, 0);
        assert!(result.unreadable_owners > 0);
        assert!(result.reclaimed.is_empty());
        for (directory, guard) in guards {
            assert!(directory.is_dir());
            assert!(guard.is_dir(), "another writer's guard must not be removed");
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_lease_cleanup_requires_proof_of_death_not_a_failed_probe() {
        use windows_sys::Win32::Foundation::{ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER};
        assert!(windows_process_probe_may_be_alive(Some(ERROR_ACCESS_DENIED), None));
        assert!(windows_process_probe_may_be_alive(Some(1), None));
        assert!(windows_process_probe_may_be_alive(None, None));
        assert!(windows_process_probe_may_be_alive(None, Some(true)));
        assert!(!windows_process_probe_may_be_alive(None, Some(false)));
        assert!(!windows_process_probe_may_be_alive(Some(ERROR_INVALID_PARAMETER), None));
        assert!(is_process_alive(std::process::id() as i64));
        assert!(!is_process_alive(i64::MAX));
    }

    #[test]
    fn iso_now_has_the_javascript_shape() {
        let text = iso_now();
        assert_eq!(text.len(), 24);
        assert!(text.ends_with('Z'));
        assert_eq!(&text[4..5], "-");
        assert_eq!(&text[10..11], "T");
        assert_eq!(&text[19..20], ".");
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
    }
}
