//! Port of packages/coding-agent/src/modes/daemon/daemon-socket.ts
//!
//! Unix socket path ownership: the per-path lock lease (proper-lockfile in the
//! TypeScript) is ported with the same stale/update windows and retry budget,
//! using a lock directory next to the socket path.

use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use crate::utils::daemon_socket_path::normalize_socket_path;

const DAEMON_SOCKET_MODE: u32 = 0o600;
const DAEMON_SOCKET_DIR_MODE: u32 = 0o700;
const DAEMON_SOCKET_RELEASE_GRACE_MS: u64 = 1000;
const DAEMON_SOCKET_RELEASE_POLL_MS: u64 = 25;
const DAEMON_SOCKET_LOCK_STALE_MS: u64 = 5000;
const DAEMON_SOCKET_LOCK_UPDATE_MS: u64 = 1000;
const DAEMON_SOCKET_LOCK_RETRIES: u32 = 600;

pub type DaemonSocketCompromiseListener = std::sync::Arc<dyn Fn(&DaemonSocketError) + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonSocketError {
    pub message: String,
}

impl std::fmt::Display for DaemonSocketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DaemonSocketError {}

pub struct DaemonSocketPathLease {
    pub socket_path: String,
    released: std::sync::atomic::AtomicBool,
    compromised_error: std::sync::Mutex<Option<DaemonSocketError>>,
    compromise_listeners: std::sync::Mutex<Vec<DaemonSocketCompromiseListener>>,
    lock_path: Option<String>,
    lock_identity: Option<DaemonSocketIdentity>,
}

impl DaemonSocketPathLease {
    pub fn new(socket_path: &str, lock_path: Option<String>) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            released: std::sync::atomic::AtomicBool::new(false),
            compromised_error: std::sync::Mutex::new(None),
            compromise_listeners: std::sync::Mutex::new(Vec::new()),
            lock_identity: lock_path.as_deref().and_then(current_lock_identity),
            lock_path,
        }
    }

    fn owns_lock(&self) -> bool {
        match (&self.lock_path, self.lock_identity) {
            (Some(lock_path), Some(identity)) => current_lock_identity(lock_path) == Some(identity),
            (None, None) => true,
            _ => false,
        }
    }

    fn lock_compromise_error(&self) -> DaemonSocketError {
        DaemonSocketError {
            message: format!(
                "Daemon socket lock was removed or replaced: {}",
                self.lock_path.as_deref().unwrap_or("unknown")
            ),
        }
    }

    fn start_heartbeat(self: &std::sync::Arc<Self>) {
        let lease = std::sync::Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(DAEMON_SOCKET_LOCK_UPDATE_MS)).await;
                let Some(lease) = lease.upgrade() else {
                    return;
                };
                if !lease.refresh_lock() {
                    return;
                }
            }
        });
    }

    fn refresh_lock(&self) -> bool {
        let mut compromised = self.compromised_error.lock().expect("compromise poisoned");
        if self.released.load(std::sync::atomic::Ordering::SeqCst) || compromised.is_some() {
            return false;
        }
        let (Some(lock_path), Some(identity)) = (&self.lock_path, self.lock_identity) else {
            return false;
        };
        let error = match refresh_lock(lock_path, identity) {
            Ok(true) => return true,
            Ok(false) => self.lock_compromise_error(),
            Err(error) => DaemonSocketError {
                message: format!("Could not refresh daemon socket lock {lock_path}: {error}"),
            },
        };
        *compromised = Some(error.clone());
        drop(compromised);
        self.notify_compromise(error);
        false
    }

    fn notify_compromise(&self, error: DaemonSocketError) {
        let listeners: Vec<DaemonSocketCompromiseListener> = self
            .compromise_listeners
            .lock()
            .expect("listeners poisoned")
            .drain(..)
            .collect();
        for listener in listeners {
            notify_compromise_listener(&listener, &error);
        }
    }

    pub fn compromise(&self) -> Option<DaemonSocketError> {
        self.compromised_error
            .lock()
            .expect("compromise poisoned")
            .clone()
    }

    /// Returns the unsubscribe handle (a no-op when the lease is already compromised).
    pub fn on_compromised(
        self: &std::sync::Arc<Self>,
        listener: DaemonSocketCompromiseListener,
    ) -> Box<dyn FnOnce() + Send> {
        let compromised = self.compromised_error.lock().expect("compromise poisoned");
        if let Some(error) = compromised.as_ref() {
            let error = error.clone();
            drop(compromised);
            notify_compromise_listener(&listener, &error);
            return Box::new(|| {});
        }
        let lease = std::sync::Arc::clone(self);
        let retained = std::sync::Arc::clone(&listener);
        self.compromise_listeners
            .lock()
            .expect("listeners poisoned")
            .push(listener);
        drop(compromised);
        Box::new(move || {
            lease
                .compromise_listeners
                .lock()
                .expect("listeners poisoned")
                .retain(|candidate| !std::sync::Arc::ptr_eq(candidate, &retained));
        })
    }

    pub fn record_compromise(&self, error: DaemonSocketError) {
        {
            let mut compromised = self.compromised_error.lock().expect("compromise poisoned");
            if compromised.is_some() {
                return;
            }
            *compromised = Some(error.clone());
        }
        self.notify_compromise(error);
    }

    pub async fn release(&self) {
        self.release_inner(None).await;
    }

    #[cfg(test)]
    async fn release_with_hook(&self, before_quarantine: &(dyn Fn() + Sync)) {
        self.release_inner(Some(before_quarantine)).await;
    }

    async fn release_inner(&self, before_quarantine: Option<&(dyn Fn() + Sync)>) {
        let mut compromised = self.compromised_error.lock().expect("compromise poisoned");
        if self
            .released
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        self.compromise_listeners
            .lock()
            .expect("listeners poisoned")
            .clear();
        // Keep release serialized with heartbeat through the identity-fenced
        // quarantine. Never perform a check-then-remove against the live path.
        let removed = if compromised.is_none() {
            match (self.lock_path.as_deref(), self.lock_identity) {
                (Some(lock_path), Some(identity)) => {
                    remove_lock_if_identity_with_hook(lock_path, identity, before_quarantine)
                }
                (None, None) => true,
                _ => false,
            }
        } else {
            true
        };
        if !removed && compromised.is_none() && self.lock_path.is_some() {
            *compromised = Some(self.lock_compromise_error());
        }
    }
}

fn notify_compromise_listener(
    listener: &DaemonSocketCompromiseListener,
    error: &DaemonSocketError,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listener(error)));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DaemonSocketIdentity {
    pub dev: u64,
    pub ino: u64,
}

pub fn default_daemon_socket_path() -> String {
    if crate::utils::pi_user_agent::process_platform() == "win32" {
        return "\\\\.\\pipe\\prime-agent-daemon".to_string();
    }
    Path::new(&default_daemon_socket_dir())
        .join("daemon.sock")
        .to_string_lossy()
        .to_string()
}

/// Acquire the per-path lease. `None` on Windows and on Unix when the lock is
/// held elsewhere past the retry budget.
pub async fn acquire_daemon_socket_path_lease(
    socket_path: &str,
) -> Option<std::sync::Arc<DaemonSocketPathLease>> {
    if let Err(error) = ensure_default_daemon_socket_dir(socket_path) {
        let _ = error;
        return None;
    }
    if crate::utils::pi_user_agent::process_platform() == "win32" {
        return None;
    }
    let lock_path = lock_directory_path(socket_path);
    let deadline = Instant::now()
        + Duration::from_millis(
            DAEMON_SOCKET_RELEASE_POLL_MS as u64 * DAEMON_SOCKET_LOCK_RETRIES as u64,
        );
    loop {
        match std::fs::create_dir(&lock_path) {
            Ok(()) => {
                let lease =
                    std::sync::Arc::new(DaemonSocketPathLease::new(socket_path, Some(lock_path)));
                if lease.lock_identity.is_none() {
                    let _ = std::fs::remove_dir(lease.lock_path.as_deref().expect("lock path"));
                    return None;
                }
                lease.start_heartbeat();
                return Some(lease);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if reclaim_stale_lock(&lock_path) {
                    continue;
                }
                if Instant::now() >= deadline {
                    return None;
                }
                tokio::time::sleep(Duration::from_millis(DAEMON_SOCKET_RELEASE_POLL_MS)).await;
            }
            Err(_) => return None,
        }
    }
}

fn lock_directory_path(socket_path: &str) -> String {
    format!("{socket_path}.lock")
}

fn current_lock_identity(lock_path: &str) -> Option<DaemonSocketIdentity> {
    std::fs::symlink_metadata(lock_path)
        .ok()
        .map(|metadata| identity_from_metadata(&metadata))
}

fn reclaim_stale_lock(lock_path: &str) -> bool {
    reclaim_stale_lock_with_hook(lock_path, None)
}

fn reclaim_stale_lock_with_hook(
    lock_path: &str,
    before_quarantine: Option<&(dyn Fn() + Sync)>,
) -> bool {
    let metadata = match std::fs::symlink_metadata(lock_path) {
        Ok(metadata) if metadata.is_dir() => metadata,
        _ => return false,
    };
    let stale = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .map(|age| age.as_millis() as u64 > DAEMON_SOCKET_LOCK_STALE_MS)
        .unwrap_or(false);
    if !stale {
        return false;
    }
    let identity = identity_from_metadata(&metadata);
    let current = match std::fs::symlink_metadata(lock_path) {
        Ok(current) => current,
        Err(error) => return error.kind() == std::io::ErrorKind::NotFound,
    };
    let still_stale = current
        .modified()
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .map(|age| age.as_millis() as u64 > DAEMON_SOCKET_LOCK_STALE_MS)
        .unwrap_or(false);
    if identity_from_metadata(&current) != identity || !still_stale {
        return false;
    }
    remove_lock_if_identity_with_hook(lock_path, identity, before_quarantine)
}

/// Atomically move the candidate out of the live path before checking its
/// identity again. A foreign inode caught by the rename is never deleted.
fn remove_lock_if_identity(lock_path: &str, expected: DaemonSocketIdentity) -> bool {
    remove_lock_if_identity_with_hook(lock_path, expected, None)
}

fn remove_lock_if_identity_with_hook(
    lock_path: &str,
    expected: DaemonSocketIdentity,
    before_quarantine: Option<&(dyn Fn() + Sync)>,
) -> bool {
    static NEXT_QUARANTINE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let quarantine = format!(
        "{lock_path}.quarantine-{}-{}",
        std::process::id(),
        NEXT_QUARANTINE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    if let Some(hook) = before_quarantine {
        hook();
    }
    if let Err(error) = std::fs::rename(lock_path, &quarantine) {
        return error.kind() == std::io::ErrorKind::NotFound;
    }
    if current_lock_identity(&quarantine) == Some(expected) {
        return match std::fs::remove_dir(&quarantine) {
            Ok(()) => true,
            Err(error) => error.kind() == std::io::ErrorKind::NotFound,
        };
    }
    // A replacement was moved into quarantine between validation and rename.
    // Put it back only if the live path is still vacant. If another owner has
    // already appeared, retain it quarantined; deleting either owner is worse.
    let _ = restore_quarantined_lock(&quarantine, lock_path);
    false
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn restore_quarantined_lock(quarantine: &str, lock_path: &str) -> std::io::Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let source = std::ffi::CString::new(Path::new(quarantine).as_os_str().as_bytes())?;
    let target = std::ffi::CString::new(Path::new(lock_path).as_os_str().as_bytes())?;
    // RENAME_NOREPLACE makes restoration safe if a new owner won the vacant
    // live path after the foreign inode was quarantined.
    if unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            target.as_ptr(),
            libc::RENAME_NOREPLACE as libc::c_uint,
        )
    } == 0
    {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos"
))]
fn restore_quarantined_lock(quarantine: &str, lock_path: &str) -> std::io::Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let source = std::ffi::CString::new(Path::new(quarantine).as_os_str().as_bytes())?;
    let target = std::ffi::CString::new(Path::new(lock_path).as_os_str().as_bytes())?;
    if unsafe { libc::renamex_np(source.as_ptr(), target.as_ptr(), libc::RENAME_EXCL) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos"
)))]
fn restore_quarantined_lock(_quarantine: &str, _lock_path: &str) -> std::io::Result<()> {
    // No portable atomic no-replace rename exists. Never fall back to an
    // overwriting rename or another check-then-remove race.
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "atomic lock restoration is not supported on this platform",
    ))
}

fn refresh_lock(lock_path: &str, expected: DaemonSocketIdentity) -> std::io::Result<bool> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let lock = options.open(lock_path)?;
    if identity_from_metadata(&lock.metadata()?) != expected
        || current_lock_identity(lock_path) != Some(expected)
    {
        return Ok(false);
    }
    lock.set_times(std::fs::FileTimes::new().set_modified(SystemTime::now()))?;
    Ok(current_lock_identity(lock_path) == Some(expected))
}

pub async fn prepare_daemon_socket_path(
    socket_path: &str,
    lease: Option<std::sync::Arc<DaemonSocketPathLease>>,
) -> Result<(), DaemonSocketError> {
    ensure_default_daemon_socket_dir(socket_path)?;

    if crate::utils::pi_user_agent::process_platform() == "win32" {
        return Ok(());
    }
    if let Some(lease) = lease {
        assert_socket_lease(socket_path, &lease)?;
        assert_socket_lease_held(socket_path, &lease)?;
        return prepare_unix_daemon_socket_path(socket_path, Some(&lease)).await;
    }
    if !Path::new(socket_path).exists() {
        return Ok(());
    }
    if can_connect_to_unix_socket(socket_path).await {
        return Err(DaemonSocketError {
            message: format!("Daemon socket already in use: {socket_path}"),
        });
    }
    let owned_lease = acquire_daemon_socket_path_lease(socket_path).await;
    let result = prepare_unix_daemon_socket_path(socket_path, owned_lease.as_deref()).await;
    if let Some(lease) = owned_lease {
        lease.release().await;
    }
    result
}

async fn prepare_unix_daemon_socket_path(
    socket_path: &str,
    lease: Option<&DaemonSocketPathLease>,
) -> Result<(), DaemonSocketError> {
    if !Path::new(socket_path).exists() {
        return Ok(());
    }

    let metadata = match std::fs::symlink_metadata(socket_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(DaemonSocketError {
                message: error.to_string(),
            })
        }
    };
    if !is_socket(&metadata) {
        return Err(DaemonSocketError {
            message: format!("Daemon socket path exists and is not a socket: {socket_path}"),
        });
    }

    let stale_identity = match get_daemon_socket_identity(socket_path) {
        Some(identity) => identity,
        None => return Ok(()),
    };
    if can_connect_to_unix_socket(socket_path).await {
        return Err(DaemonSocketError {
            message: format!("Daemon socket already in use: {socket_path}"),
        });
    }
    let deadline = Instant::now() + Duration::from_millis(DAEMON_SOCKET_RELEASE_GRACE_MS);
    while Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(DAEMON_SOCKET_RELEASE_POLL_MS)).await;
        if !Path::new(socket_path).exists() {
            return Ok(());
        }
        let current_identity = match get_daemon_socket_identity(socket_path) {
            Some(identity) => identity,
            None => return Ok(()),
        };
        if current_identity != stale_identity {
            return Err(DaemonSocketError {
                message: format!(
                    "Daemon socket changed ownership while waiting for cleanup: {socket_path}"
                ),
            });
        }
        if can_connect_to_unix_socket(socket_path).await {
            return Err(DaemonSocketError {
                message: format!("Daemon socket already in use: {socket_path}"),
            });
        }
    }

    if let Some(lease) = lease {
        assert_socket_lease_held(socket_path, lease)?;
    }
    std::fs::remove_file(socket_path).map_err(|error| DaemonSocketError {
        message: error.to_string(),
    })
}

pub fn restrict_daemon_socket_path(socket_path: &str) {
    if crate::utils::pi_user_agent::process_platform() == "win32" {
        return;
    }
    let _ = set_file_mode(socket_path, DAEMON_SOCKET_MODE);
}

pub fn get_daemon_socket_identity(socket_path: &str) -> Option<DaemonSocketIdentity> {
    if crate::utils::pi_user_agent::process_platform() == "win32" {
        return None;
    }
    let metadata = std::fs::symlink_metadata(socket_path).ok()?;
    Some(identity_from_metadata(&metadata))
}

pub fn cleanup_daemon_socket_path(
    socket_path: &str,
    expected_identity: Option<DaemonSocketIdentity>,
    lease: Option<&DaemonSocketPathLease>,
) {
    if crate::utils::pi_user_agent::process_platform() == "win32" {
        return;
    }
    if let Some(lease) = lease {
        if assert_socket_lease(socket_path, lease).is_err() {
            return;
        }
        if assert_socket_lease_held(socket_path, lease).is_err() {
            return;
        }
        let _ = cleanup_unix_daemon_socket_path(socket_path, expected_identity);
        return;
    }
    let lock_path = lock_directory_path(socket_path);
    if std::fs::create_dir(&lock_path).is_err() {
        return;
    }
    let _ = cleanup_unix_daemon_socket_path(socket_path, expected_identity);
    let _ = std::fs::remove_dir(&lock_path);
}

fn cleanup_unix_daemon_socket_path(
    socket_path: &str,
    expected_identity: Option<DaemonSocketIdentity>,
) -> Option<()> {
    if !Path::new(socket_path).exists() {
        return None;
    }
    if let Some(expected_identity) = expected_identity {
        let current_identity = get_daemon_socket_identity(socket_path)?;
        if current_identity != expected_identity {
            return None;
        }
    }
    std::fs::remove_file(socket_path).ok()
}

fn assert_socket_lease(
    socket_path: &str,
    lease: &DaemonSocketPathLease,
) -> Result<(), DaemonSocketError> {
    if lease.socket_path != socket_path {
        return Err(DaemonSocketError {
            message: format!("Daemon socket lease does not match {socket_path}"),
        });
    }
    Ok(())
}

fn assert_socket_lease_held(
    socket_path: &str,
    lease: &DaemonSocketPathLease,
) -> Result<(), DaemonSocketError> {
    if lease.released.load(std::sync::atomic::Ordering::SeqCst) {
        return Err(DaemonSocketError {
            message: format!("Daemon socket lease for {socket_path} was released"),
        });
    }
    if !lease.owns_lock() {
        lease.record_compromise(lease.lock_compromise_error());
    }
    if let Some(compromise) = lease.compromise() {
        return Err(DaemonSocketError {
            message: format!(
                "Daemon socket lease for {socket_path} was compromised: {}",
                compromise.message
            ),
        });
    }
    Ok(())
}

pub fn default_daemon_socket_dir() -> String {
    let suffix = current_uid()
        .map(|uid| uid.to_string())
        .unwrap_or_else(|| "user".to_string());
    Path::new(&std::env::temp_dir())
        .join(format!("prime-agent-{suffix}"))
        .to_string_lossy()
        .to_string()
}

fn ensure_default_daemon_socket_dir(socket_path: &str) -> Result<(), DaemonSocketError> {
    let default_dir = default_daemon_socket_dir();
    if crate::utils::pi_user_agent::process_platform() == "win32" {
        return Ok(());
    }
    if parent_dir(socket_path) != default_dir {
        return Ok(());
    }

    if !Path::new(&default_dir).exists() {
        create_dir_mode(&default_dir, DAEMON_SOCKET_DIR_MODE).map_err(|error| {
            DaemonSocketError {
                message: error.to_string(),
            }
        })?;
    }

    let metadata = std::fs::symlink_metadata(&default_dir).map_err(|error| DaemonSocketError {
        message: error.to_string(),
    })?;
    if !metadata.is_dir() {
        return Err(DaemonSocketError {
            message: format!(
                "Daemon socket directory exists and is not a directory: {default_dir}"
            ),
        });
    }

    if let (Some(uid), Some(owner)) = (current_uid(), file_owner(&metadata)) {
        if owner != uid {
            return Err(DaemonSocketError {
                message: format!(
                    "Daemon socket directory is not owned by the current user: {default_dir}"
                ),
            });
        }
    }

    let _ = set_file_mode(&default_dir, DAEMON_SOCKET_DIR_MODE);
    Ok(())
}

#[cfg(unix)]
async fn can_connect_to_unix_socket(socket_path: &str) -> bool {
    let connect = tokio::net::UnixStream::connect(socket_path);
    match tokio::time::timeout(Duration::from_millis(250), connect).await {
        Ok(Ok(_stream)) => true,
        _ => false,
    }
}

/// Windows uses named pipes, so the Unix probe is never reached there
/// (every caller returns early when `process.platform === "win32"`).
#[cfg(not(unix))]
async fn can_connect_to_unix_socket(_socket_path: &str) -> bool {
    false
}

fn parent_dir(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(|parent| parent.to_string_lossy().to_string())
        .unwrap_or_default()
}

#[cfg(unix)]
fn is_socket(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::FileTypeExt;
    metadata.file_type().is_socket()
}

#[cfg(not(unix))]
fn is_socket(_metadata: &std::fs::Metadata) -> bool {
    false
}

#[cfg(unix)]
fn identity_from_metadata(metadata: &std::fs::Metadata) -> DaemonSocketIdentity {
    use std::os::unix::fs::MetadataExt;
    DaemonSocketIdentity {
        dev: metadata.dev(),
        ino: metadata.ino(),
    }
}

#[cfg(not(unix))]
fn identity_from_metadata(_metadata: &std::fs::Metadata) -> DaemonSocketIdentity {
    DaemonSocketIdentity { dev: 0, ino: 0 }
}

#[cfg(unix)]
fn current_uid() -> Option<u32> {
    Some(unsafe { libc::getuid() })
}

#[cfg(not(unix))]
fn current_uid() -> Option<u32> {
    None
}

#[cfg(unix)]
fn file_owner(metadata: &std::fs::Metadata) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    Some(metadata.uid())
}

#[cfg(not(unix))]
fn file_owner(_metadata: &std::fs::Metadata) -> Option<u32> {
    None
}

#[cfg(unix)]
fn create_dir_mode(path: &str, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true).mode(mode);
    builder.create(path)
}

#[cfg(not(unix))]
fn create_dir_mode(path: &str, _mode: u32) -> std::io::Result<()> {
    std::fs::create_dir_all(path)
}

#[cfg(unix)]
fn set_file_mode(path: &str, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn set_file_mode(_path: &str, _mode: u32) -> std::io::Result<()> {
    Ok(())
}

/// `export { normalizeSocketPath } from "../../utils/daemon-socket-path.js"`.
pub fn normalize_socket_path_for_daemon(path: &str, base_dir: Option<&str>) -> String {
    normalize_socket_path(path, base_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "visionos"
    ))]
    #[test]
    fn two_reclaimers_cannot_delete_a_new_owner_created_after_the_first_quarantine() {
        let root = tempfile::tempdir().expect("test directory");
        let lock_path = root.path().join("daemon.sock.lock");
        std::fs::create_dir(&lock_path).expect("stale lock");
        std::fs::File::open(&lock_path)
            .expect("open stale lock")
            .set_times(
                std::fs::FileTimes::new().set_modified(SystemTime::now() - Duration::from_secs(10)),
            )
            .expect("age stale lock");
        let lock_path = lock_path.to_string_lossy().into_owned();
        let stale_identity = current_lock_identity(&lock_path).expect("stale identity");
        let replacement = root.path().join("replacement.lock");
        std::fs::create_dir(&replacement).expect("replacement lock");
        let replacement_identity =
            current_lock_identity(&replacement.to_string_lossy()).expect("replacement identity");

        // The outer reclaimer is paused after its stale identity/staleness checks.
        // The inner call is the second reclaimer, which already validated that
        // same identity. A new owner then takes the live path before the outer
        // reclaimer performs its atomic quarantine rename.
        let reclaimed = reclaim_stale_lock_with_hook(
            &lock_path,
            Some(&|| {
                assert!(remove_lock_if_identity(&lock_path, stale_identity));
                std::fs::rename(&replacement, &lock_path).expect("new owner takes lock path");
            }),
        );
        assert!(!reclaimed, "the late reclaimer saw a replacement");
        assert_eq!(
            current_lock_identity(&lock_path),
            Some(replacement_identity)
        );
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "visionos"
    ))]
    #[tokio::test]
    async fn release_revalidates_quarantined_identity_when_replacement_races() {
        let root = tempfile::tempdir().expect("test directory");
        let lock_path = root.path().join("daemon.sock.lock");
        std::fs::create_dir(&lock_path).expect("owned lock");
        let lock_path = lock_path.to_string_lossy().into_owned();
        let lease = DaemonSocketPathLease::new("/tmp/daemon.sock", Some(lock_path.clone()));
        let replacement = root.path().join("replacement.lock");
        std::fs::create_dir(&replacement).expect("replacement lock");
        let replacement_identity =
            current_lock_identity(&replacement.to_string_lossy()).expect("replacement identity");

        lease
            .release_with_hook(&|| {
                // This is the old owns_lock -> remove_dir race window. The release
                // path must now catch and preserve the replacement via quarantine.
                std::fs::remove_dir(&lock_path).expect("external owner removal");
                std::fs::rename(&replacement, &lock_path).expect("replacement takes lock path");
            })
            .await;

        assert_eq!(
            current_lock_identity(&lock_path),
            Some(replacement_identity)
        );
        assert!(lease.compromise().is_some());
        assert!(lease.released.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn default_paths_match_the_platform() {
        let path = default_daemon_socket_path();
        if crate::utils::pi_user_agent::process_platform() == "win32" {
            assert_eq!(path, "\\\\.\\pipe\\prime-agent-daemon");
        } else {
            assert!(path.ends_with("daemon.sock"));
        }
        assert!(default_daemon_socket_dir().contains("prime-agent-"));
    }

    #[tokio::test]
    async fn a_lease_blocks_a_second_acquire_until_release() {
        if crate::utils::pi_user_agent::process_platform() == "win32" {
            assert!(acquire_daemon_socket_path_lease("/tmp/whatever.sock")
                .await
                .is_none());
            return;
        }
        let root = std::env::temp_dir().join(format!("daemon-socket-lease-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("root");
        let socket_path = root.join("daemon.sock").to_string_lossy().to_string();
        let lease = acquire_daemon_socket_path_lease(&socket_path)
            .await
            .expect("lease");
        assert!(Path::new(&lock_directory_path(&socket_path)).exists());
        lease.release().await;
        assert!(!Path::new(&lock_directory_path(&socket_path)).exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn lease_compromise_notifies_listeners_once() {
        let lease = std::sync::Arc::new(DaemonSocketPathLease::new("/tmp/s.sock", None));
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = std::sync::Arc::clone(&seen);
        let listener: DaemonSocketCompromiseListener =
            std::sync::Arc::new(move |error: &DaemonSocketError| {
                captured.lock().expect("seen").push(error.message.clone());
            });
        let _unsubscribe = lease.on_compromised(listener);
        lease.record_compromise(DaemonSocketError {
            message: "lock lost".to_string(),
        });
        lease.record_compromise(DaemonSocketError {
            message: "second".to_string(),
        });
        assert_eq!(
            seen.lock().expect("seen").clone(),
            vec!["lock lost".to_string()]
        );
        assert_eq!(
            lease.compromise().map(|error| error.message),
            Some("lock lost".to_string())
        );
        assert_eq!(
            assert_socket_lease("/tmp/other.sock", &lease)
                .expect_err("mismatch")
                .message,
            "Daemon socket lease does not match /tmp/other.sock"
        );
        assert_eq!(
            assert_socket_lease_held("/tmp/s.sock", &lease)
                .expect_err("compromised")
                .message,
            "Daemon socket lease for /tmp/s.sock was compromised: lock lost"
        );
    }

    #[tokio::test]
    async fn cleanup_removes_the_socket_when_the_identity_matches() {
        if crate::utils::pi_user_agent::process_platform() == "win32" {
            return;
        }
        let root =
            std::env::temp_dir().join(format!("daemon-socket-cleanup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("root");
        let socket_path = root.join("daemon.sock").to_string_lossy().to_string();
        std::fs::write(&socket_path, b"stale").expect("stale file");
        let identity = get_daemon_socket_identity(&socket_path).expect("identity");
        cleanup_daemon_socket_path(&socket_path, Some(identity), None);
        assert!(!Path::new(&socket_path).exists());

        std::fs::write(&socket_path, b"stale").expect("stale file");
        cleanup_daemon_socket_path(
            &socket_path,
            Some(DaemonSocketIdentity { dev: 0, ino: 0 }),
            None,
        );
        assert!(Path::new(&socket_path).exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn prepare_rejects_a_non_socket_path_and_ignores_missing_paths() {
        if crate::utils::pi_user_agent::process_platform() == "win32" {
            return;
        }
        let root =
            std::env::temp_dir().join(format!("daemon-socket-prepare-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("root");
        let socket_path = root.join("daemon.sock").to_string_lossy().to_string();
        prepare_daemon_socket_path(&socket_path, None)
            .await
            .expect("missing path is fine");
        std::fs::write(&socket_path, b"not a socket").expect("file");
        assert_eq!(
            prepare_daemon_socket_path(&socket_path, None)
                .await
                .expect_err("not a socket")
                .message,
            format!("Daemon socket path exists and is not a socket: {socket_path}")
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
