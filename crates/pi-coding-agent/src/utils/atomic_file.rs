//! Port of packages/coding-agent/src/utils/atomic-file.ts

use std::collections::HashMap;
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::pi_user_agent::process_platform;

const WIN32_RENAME_ATTEMPTS: u32 = 5;

fn sleep_sync(ms: u64) {
    std::thread::sleep(Duration::from_millis(ms));
}

/// Node maps EPERM/EACCES/EBUSY from these platform error numbers; on Windows
/// the raw codes are ERROR_ACCESS_DENIED (5), ERROR_SHARING_VIOLATION (32) and
/// ERROR_LOCK_VIOLATION (33).
fn is_transient_windows_rename_error(error: &std::io::Error, platform: &str) -> bool {
    if platform != "win32" {
        return false;
    }
    const TRANSIENT: [i32; 6] = [1, 5, 13, 16, 32, 33];
    match error.raw_os_error() {
        Some(code) => TRANSIENT.contains(&code),
        None => matches!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::WouldBlock
        ),
    }
}

// Windows raises transient EPERM/EACCES when the destination is held open (antivirus, indexer).
fn rename_onto_sync(from: &str, to: &str, platform: &str) -> std::io::Result<()> {
    let mut attempt: u32 = 1;
    loop {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(error) => {
                if !is_transient_windows_rename_error(&error, platform) || attempt >= WIN32_RENAME_ATTEMPTS {
                    return Err(error);
                }
                sleep_sync(10 * attempt as u64);
                attempt += 1;
            }
        }
    }
}

#[derive(Debug)]
pub struct AtomicRenameRetryEvent {
    pub attempt: u32,
    pub delay_ms: u64,
    pub error: std::io::Error,
}

pub type AsyncSleepFn = Arc<dyn Fn(u64) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

#[derive(Clone, Default)]
pub struct AtomicRenameRetryOptions {
    pub attempts: Option<u32>,
    pub platform: Option<String>,
    pub sleep: Option<AsyncSleepFn>,
    /// Best-effort observer. Observer failures never fail the durable write.
    pub on_retry: Option<Arc<dyn Fn(&AtomicRenameRetryEvent) + Send + Sync>>,
}

fn async_delay(delay_ms: u64) -> Pin<Box<dyn Future<Output = ()> + Send>> {
    Box::pin(tokio::time::sleep(Duration::from_millis(delay_ms)))
}

async fn rename_onto(from: &str, to: &str, options: &AtomicRenameRetryOptions) -> std::io::Result<()> {
    let attempts = options.attempts.unwrap_or(WIN32_RENAME_ATTEMPTS).max(1);
    let platform = options
        .platform
        .clone()
        .unwrap_or_else(|| process_platform().to_string());
    let mut attempt: u32 = 1;
    loop {
        match tokio::fs::rename(from, to).await {
            Ok(()) => return Ok(()),
            Err(error) => {
                if !is_transient_windows_rename_error(&error, &platform) || attempt >= attempts {
                    return Err(error);
                }
                let delay_ms = 10 * attempt as u64;
                if let Some(on_retry) = options.on_retry.as_ref() {
                    // Retry observation is disposable; persistence is not.
                    let event = AtomicRenameRetryEvent {
                        attempt,
                        delay_ms,
                        error: std::io::Error::new(error.kind(), error.to_string()),
                    };
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_retry(&event)));
                }
                let sleep = options.sleep.clone().unwrap_or_else(|| Arc::new(async_delay));
                sleep(delay_ms).await;
                attempt += 1;
            }
        }
    }
}

#[derive(Default)]
pub struct WriteFileAtomicOptions {
    pub mode: Option<u32>,
    /// fsync the temp file before the rename.
    pub fsync: bool,
    /// Directory fsync after the rename; tolerates only unsupported Windows directory fsync.
    pub fsync_dir: bool,
    /// Runs on the written temp file before it replaces the destination (validation, ownership).
    pub before_rename: Option<Box<dyn FnOnce(&str)>>,
}

#[derive(Default)]
pub struct WriteFileAtomicAsyncOptions {
    pub mode: Option<u32>,
    pub fsync: bool,
    pub fsync_dir: bool,
    /// Runs on the written temp file before it replaces the destination (validation, ownership).
    pub before_rename: Option<Box<dyn FnOnce(&str)>>,
    pub rename_retry: Option<AtomicRenameRetryOptions>,
}

fn should_ignore_directory_fsync_error(error: &std::io::Error, platform: &str) -> bool {
    // Node checks `code === "EPERM" && syscall === "fsync"`; Windows reports the
    // same access-denied class for unsupported directory sync.
    platform == "win32" && error.kind() == std::io::ErrorKind::PermissionDenied
}

fn fsync_directory_sync(path: &str, platform: &str) -> std::io::Result<()> {
    let result = (|| -> std::io::Result<()> {
        let file = std::fs::File::open(path)?;
        file.sync_all()
    })();
    match result {
        Ok(()) => Ok(()),
        Err(error) => {
            if should_ignore_directory_fsync_error(&error, platform) {
                Ok(())
            } else {
                Err(error)
            }
        }
    }
}

async fn fsync_directory(path: &str, platform: &str) -> std::io::Result<()> {
    let result = (|| -> std::io::Result<()> {
        let file = std::fs::File::open(path)?;
        file.sync_all()
    })();
    match result {
        Ok(()) => Ok(()),
        Err(error) => {
            if should_ignore_directory_fsync_error(&error, platform) {
                Ok(())
            } else {
                Err(error)
            }
        }
    }
}

fn temp_path_for(path: &str) -> String {
    format!("{}.{}.{}.tmp", path, std::process::id(), uuid::Uuid::new_v4())
}

fn parent_dir(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(|parent| parent.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string())
}

/// Durable-write owner: temp file beside the destination, then an atomic rename.
pub fn write_file_atomic_sync(path: &str, data: &str, options: WriteFileAtomicOptions) -> std::io::Result<()> {
    write_bytes_atomic_sync(path, data.as_bytes(), options)
}

/// Byte-capable twin of `write_file_atomic_sync` for payloads that are not valid
/// UTF-8 (corrupt-state recovery copies must preserve the original bytes exactly).
pub fn write_bytes_atomic_sync(path: &str, data: &[u8], mut options: WriteFileAtomicOptions) -> std::io::Result<()> {
    let temp_path = temp_path_for(path);
    let platform = process_platform().to_string();

    let outcome = (|| -> std::io::Result<()> {
        let mut open_options = std::fs::OpenOptions::new();
        open_options.write(true).create_new(true);
        #[cfg(unix)]
        if let Some(mode) = options.mode {
            use std::os::unix::fs::OpenOptionsExt;
            open_options.mode(mode);
        }
        let mut descriptor = open_options.open(&temp_path)?;

        // writeSync may return a short count without throwing; a partial temp must never be renamed in.
        let bytes = data;
        let mut offset = 0usize;
        while offset < bytes.len() {
            let written = descriptor.write(&bytes[offset..])?;
            if written == 0 {
                return Err(std::io::Error::other(format!("Short write persisting {}", path)));
            }
            offset += written;
        }
        if options.fsync {
            descriptor.sync_all()?;
        }
        drop(descriptor);

        // openSync's mode is masked by the umask; enforce the requested bits exactly.
        #[cfg(unix)]
        if let Some(mode) = options.mode {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&temp_path, std::fs::Permissions::from_mode(mode))?;
        }
        if let Some(before_rename) = options.before_rename.take() {
            before_rename(&temp_path);
        }
        rename_onto_sync(&temp_path, path, &platform)
    })();

    let _ = std::fs::remove_file(&temp_path);
    outcome?;

    if options.fsync_dir {
        fsync_directory_sync(&parent_dir(path), &platform)?;
    }
    Ok(())
}

/// Async atomic replacement for daemon-owned writes that must not block the event loop during rename contention.
pub async fn write_file_atomic(
    path: &str,
    data: &str,
    mut options: WriteFileAtomicAsyncOptions,
) -> std::io::Result<()> {
    let temp_path = temp_path_for(path);
    let platform = process_platform().to_string();

    let outcome = async {
        let mut open_options = tokio::fs::OpenOptions::new();
        open_options.write(true).create_new(true);
        #[cfg(unix)]
        if let Some(mode) = options.mode {
            use std::os::unix::fs::OpenOptionsExt;
            open_options.mode(mode);
        }
        let mut handle = open_options.open(&temp_path).await?;

        let bytes = data.as_bytes();
        let mut offset = 0usize;
        while offset < bytes.len() {
            let written = tokio::io::AsyncWriteExt::write(&mut handle, &bytes[offset..]).await?;
            if written == 0 {
                return Err(std::io::Error::other(format!("Short write persisting {}", path)));
            }
            offset += written;
        }
        if options.fsync {
            handle.sync_all().await?;
        }
        drop(handle);

        #[cfg(unix)]
        if let Some(mode) = options.mode {
            use std::os::unix::fs::PermissionsExt;
            tokio::fs::set_permissions(&temp_path, std::fs::Permissions::from_mode(mode)).await?;
        }
        if let Some(before_rename) = options.before_rename.take() {
            before_rename(&temp_path);
        }
        let retry = options.rename_retry.clone().unwrap_or_default();
        rename_onto(&temp_path, path, &retry).await
    }
    .await;

    let _ = tokio::fs::remove_file(&temp_path).await;
    outcome?;

    if options.fsync_dir {
        let platform = options
            .rename_retry
            .as_ref()
            .and_then(|retry| retry.platform.clone())
            .unwrap_or(platform);
        fsync_directory(&parent_dir(path), &platform).await?;
    }
    Ok(())
}

pub struct RemoveFileDurablyOptions {
    pub fsync_dir: bool,
    pub platform: Option<String>,
}

/// Remove a lifecycle file and, when requested, durably persist the directory entry change.
pub async fn remove_file_durably(path: &str, options: RemoveFileDurablyOptions) -> std::io::Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    if options.fsync_dir {
        let platform = options.platform.unwrap_or_else(|| process_platform().to_string());
        fsync_directory(&parent_dir(path), &platform).await?;
    }
    Ok(())
}

pub struct AtomicFileWriteResult {
    pub generation: u64,
}

struct AtomicFileTargetState {
    next_generation: u64,
    lock: Arc<tokio::sync::Mutex<()>>,
}

/// Serializes atomic operations per target. Different targets proceed concurrently.
/// Every write gets a monotonic generation and no lifecycle-critical write is coalesced.
pub struct AtomicFileWriteCoordinator {
    targets: Mutex<HashMap<String, AtomicFileTargetState>>,
    platform: String,
}

impl Default for AtomicFileWriteCoordinator {
    fn default() -> Self {
        Self::new(process_platform())
    }
}

impl AtomicFileWriteCoordinator {
    pub fn new(platform: &str) -> Self {
        Self {
            targets: Mutex::new(HashMap::new()),
            platform: platform.to_string(),
        }
    }

    fn target_key(&self, path: &str) -> String {
        let absolute = absolute_lexical(path);
        // Windows descriptor paths live inside a trusted lease directory, where lexical case
        // aliases name the same file.
        if self.platform == "win32" {
            absolute.to_lowercase()
        } else {
            absolute
        }
    }

    pub async fn write(
        &self,
        path: &str,
        data: &str,
        options: WriteFileAtomicAsyncOptions,
    ) -> std::io::Result<AtomicFileWriteResult> {
        let path = path.to_string();
        let data = data.to_string();
        let key = path.clone();
        self.run(&key, move || async move {
            write_file_atomic(&path, &data, options).await
        })
        .await
    }

    pub async fn run<F, Fut>(&self, path: &str, operation: F) -> std::io::Result<AtomicFileWriteResult>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = std::io::Result<()>>,
    {
        let target = self.target_key(path);
        let (generation, lock) = {
            let mut targets = self.targets.lock().expect("coordinator targets");
            let state = targets
                .entry(target)
                .or_insert_with(|| AtomicFileTargetState {
                    next_generation: 0,
                    lock: Arc::new(tokio::sync::Mutex::new(())),
                });
            state.next_generation += 1;
            (state.next_generation, state.lock.clone())
        };
        let guard = lock.lock().await;
        let result = operation().await;
        drop(guard);
        result.map(|_| AtomicFileWriteResult { generation })
    }

    pub async fn drain(&self, timeout_ms: u64) -> std::io::Result<()> {
        let started_at = std::time::Instant::now();
        loop {
            let snapshot: Vec<(String, Arc<tokio::sync::Mutex<()>>)> = {
                let targets = self.targets.lock().expect("coordinator targets");
                if targets.is_empty() {
                    return Ok(());
                }
                targets
                    .iter()
                    .map(|(target, state)| (target.clone(), state.lock.clone()))
                    .collect()
            };

            let elapsed = started_at.elapsed().as_millis() as u64;
            let remaining = timeout_ms.saturating_sub(elapsed);
            let acquired = tokio::time::timeout(Duration::from_millis(remaining), async {
                for (_, lock) in &snapshot {
                    let guard = lock.lock().await;
                    drop(guard);
                }
            })
            .await;
            if acquired.is_err() {
                return Err(std::io::Error::other(format!(
                    "Timed out after {}ms draining atomic file writes",
                    timeout_ms
                )));
            }

            // A write may have joined an existing target (or created a new one) while
            // this drain was waiting. Observe tails again so shutdown cannot report a
            // successful drain while a late, already-admitted durable write is pending.
            let targets = self.targets.lock().expect("coordinator targets");
            let unchanged = targets.len() == snapshot.len()
                && snapshot.iter().all(|(target, lock)| match targets.get(target) {
                    Some(state) => Arc::ptr_eq(&state.lock, lock),
                    None => false,
                });
            if unchanged {
                return Ok(());
            }
        }
    }
}

fn absolute_lexical(path: &str) -> String {
    let path_ref = Path::new(path);
    if path_ref.is_absolute() {
        path.to_string()
    } else {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        cwd.join(path_ref).to_string_lossy().to_string()
    }
}

/// Resolve symlink aliases so a replace lands on the real file (in-place-write parity).
pub fn realpath_if_present_sync(path: &str) -> std::io::Result<String> {
    match std::fs::canonicalize(path) {
        Ok(real) => return Ok(real.to_string_lossy().to_string()),
        Err(error) => {
            if error.kind() != std::io::ErrorKind::NotFound {
                return Err(error);
            }
        }
    }
    // ENOENT also means a DANGLING symlink chain: follow it like in-place writes did.
    let mut current = PathBuf::from(path);
    for _ in 0..32 {
        let target = match std::fs::read_link(&current) {
            Ok(target) => target,
            Err(_) => return Ok(current.to_string_lossy().to_string()),
        };
        // A relative target resolves against the link's PHYSICAL parent directory.
        let parent = current
            .parent()
            .map(|parent| parent.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let parent = std::fs::canonicalize(&parent).unwrap_or(parent);
        current = if target.is_absolute() {
            target
        } else {
            parent.join(target)
        };
    }
    // A loud failure beats silently replacing an intermediate link (or looping on a cycle).
    Err(std::io::Error::other(format!(
        "Too many symlink hops resolving {}",
        path
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        std::mem::forget(dir);
        path
    }

    #[test]
    fn sync_write_replaces_the_destination_and_leaves_no_temp() {
        let dir = temp_dir();
        let path = dir.join("state.json");
        std::fs::write(&path, "old").unwrap();

        write_file_atomic_sync(
            path.to_str().unwrap(),
            "new",
            WriteFileAtomicOptions {
                mode: Some(0o600),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["state.json".to_string()]);
    }

    #[test]
    fn sync_write_runs_before_rename_before_the_replacement() {
        let dir = temp_dir();
        let path = dir.join("state.json");
        std::fs::write(&path, "old").unwrap();

        let seen = Arc::new(Mutex::new(String::new()));
        let seen_clone = seen.clone();
        let destination = path.clone();
        write_file_atomic_sync(
            path.to_str().unwrap(),
            "new",
            WriteFileAtomicOptions {
                before_rename: Some(Box::new(move |temp_path| {
                    let content = std::fs::read_to_string(temp_path).unwrap();
                    let destination_content = std::fs::read_to_string(&destination).unwrap();
                    *seen_clone.lock().unwrap() = format!("{}|{}", content, destination_content);
                })),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(*seen.lock().unwrap(), "new|old");
    }

    #[test]
    fn sync_write_failure_keeps_the_old_destination() {
        let dir = temp_dir();
        let path = dir.join("nested").join("state.json");
        let result = write_file_atomic_sync(path.to_str().unwrap(), "new", WriteFileAtomicOptions::default());
        assert!(result.is_err());
        assert!(!path.exists());
    }

    #[test]
    fn windows_rename_retry_matches_the_typescript_predicate() {
        let denied = std::io::Error::from_raw_os_error(5);
        assert!(is_transient_windows_rename_error(&denied, "win32"));
        assert!(!is_transient_windows_rename_error(&denied, "linux"));
        let other = std::io::Error::from_raw_os_error(9);
        assert!(!is_transient_windows_rename_error(&other, "win32"));
    }

    #[tokio::test]
    async fn async_write_writes_the_destination() {
        let dir = temp_dir();
        let path = dir.join("state.json");
        std::fs::write(&path, "old").unwrap();

        let delays = Arc::new(Mutex::new(Vec::<u64>::new()));
        let delays_clone = delays.clone();
        let sleeps = Arc::new(move |delay_ms: u64| {
            delays_clone.lock().unwrap().push(delay_ms);
            async_delay(0)
        });

        write_file_atomic(
            path.to_str().unwrap(),
            "new",
            WriteFileAtomicAsyncOptions {
                rename_retry: Some(AtomicRenameRetryOptions {
                    platform: Some("win32".to_string()),
                    sleep: Some(sleeps),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        assert!(delays.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn async_write_propagates_unexpected_rename_errors() {
        let dir = temp_dir();
        let path = dir.join("state.json");
        std::fs::write(&path, "old").unwrap();
        // Renaming a temp path that was never created is a hard failure.
        let result = write_file_atomic(
            path.to_str().unwrap(),
            "new",
            WriteFileAtomicAsyncOptions {
                before_rename: Some(Box::new(|temp_path| {
                    let _ = std::fs::remove_file(temp_path);
                })),
                ..Default::default()
            },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "old");
    }

    #[tokio::test]
    async fn coordinator_keeps_generations_monotonic() {
        let dir = temp_dir();
        let path = dir.join("journal.jsonl");
        let path_string = path.to_string_lossy().to_string();
        let coordinator = AtomicFileWriteCoordinator::default();

        let mut order: Vec<u64> = Vec::new();
        for _ in 0..3u64 {
            let result = coordinator
                .run(&path_string, || async move {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    Ok(())
                })
                .await
                .unwrap();
            order.push(result.generation);
        }
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn coordinator_drain_returns_when_idle_and_times_out_on_a_held_target() {
        let coordinator = std::sync::Arc::new(AtomicFileWriteCoordinator::default());
        coordinator.drain(50).await.unwrap();

        let path = "drain-target.json";
        let (gate_sender, gate_receiver) = tokio::sync::oneshot::channel::<()>();
        let holder = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move {
                coordinator
                    .run(path, || async move {
                        let _ = gate_receiver.await;
                        Ok(())
                    })
                    .await
            })
        };
        tokio::time::sleep(Duration::from_millis(10)).await;
        let drain_result = coordinator.drain(20).await;
        assert!(drain_result.is_err());
        assert!(drain_result
            .unwrap_err()
            .to_string()
            .contains("draining atomic file writes"));
        let _ = gate_sender.send(());
        assert!(holder.await.unwrap().is_ok());
    }

    #[test]
    fn realpath_follows_symlinks_and_tolerates_missing_paths() {
        let dir = temp_dir();
        let real = dir.join("real.txt");
        std::fs::write(&real, "x").unwrap();
        let resolved = realpath_if_present_sync(real.to_str().unwrap()).unwrap();
        assert_eq!(resolved, std::fs::canonicalize(&real).unwrap().to_string_lossy());

        let missing = dir.join("missing.txt");
        assert_eq!(
            realpath_if_present_sync(missing.to_str().unwrap()).unwrap(),
            missing.to_string_lossy()
        );
    }

    #[cfg(unix)]
    #[test]
    fn realpath_follows_a_dangling_symlink_chain() {
        let dir = temp_dir();
        let first = dir.join("first");
        let second = dir.join("second");
        std::os::unix::fs::symlink("second", &first).unwrap();
        std::os::unix::fs::symlink("final.txt", &second).unwrap();
        assert_eq!(
            realpath_if_present_sync(first.to_str().unwrap()).unwrap(),
            dir.join("final.txt").to_string_lossy()
        );
    }

    #[test]
    fn remove_file_durably_ignores_missing_files() {
        let dir = temp_dir();
        let missing = dir.join("gone.json");
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(remove_file_durably(
                missing.to_str().unwrap(),
                RemoveFileDurablyOptions {
                    fsync_dir: true,
                    platform: None,
                },
            ))
            .unwrap();
    }
}
