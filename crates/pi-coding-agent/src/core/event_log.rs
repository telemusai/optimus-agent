//! Port of packages/coding-agent/src/core/event-log.ts
//!
//! Append-only JSONL event log: the shared crash-safety substrate under the
//! RLM spawn ledger and the ACP semantic-edge ledger.
//!
//! Appends are single O_APPEND writes (PIPE_BUF-scale atomicity), fsynced only
//! when the caller needs durability. Tail rule (union of every consumer's
//! safety): an unterminated final line is an uncommitted append — skipped on
//! read even when it parses, truncated at its byte offset on the next append,
//! never newline-completed (completion turns a line a strict parser rejects
//! into permanent fail-closed interior poison). Interior malformed lines fail
//! closed. Repair runs only on append, never on read: a viewer may replay a
//! live writer's log.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use crate::utils::timed_flush::{register_timed_flush, write_flush_period, TimedBytes, TimedFlush};

pub type EventLogLogger = std::sync::Arc<dyn Fn(String) + Send + Sync>;

/// `EventLogOptions`.
#[derive(Clone, Default)]
pub struct EventLogOptions {
    /// Fail closed beyond these bounds on every full read, including the repair path.
    pub max_bytes: Option<u64>,
    pub max_records: Option<usize>,
    pub log: Option<EventLogLogger>,
}

/// `replaySync` options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayOptions {
    pub missing_file_throws: bool,
}

/// Bounded read through the descriptor: the size check and the allocation see the same fd, so a concurrent grow cannot bypass the bound.
fn read_all_sync(
    file: &mut std::fs::File,
    max_bytes: Option<u64>,
    path: &str,
) -> std::io::Result<Vec<u8>> {
    let size = file.metadata()?.len();
    if let Some(max_bytes) = max_bytes {
        if size > max_bytes {
            return Err(std::io::Error::other(format!(
                "event log {path} exceeds {max_bytes} bytes ({size}); refusing to read"
            )));
        }
    }
    file.seek(SeekFrom::Start(0))?;
    let mut buffer = vec![0u8; size as usize];
    let mut offset = 0usize;
    while offset < buffer.len() {
        let bytes_read = file.read(&mut buffer[offset..])?;
        if bytes_read == 0 {
            break;
        }
        offset += bytes_read;
    }
    buffer.truncate(offset);
    Ok(buffer)
}

fn serialize_line(event: &serde_json::Value) -> Result<String, std::io::Error> {
    match serde_json::to_string(event) {
        Ok(serialized) => Ok(format!("{serialized}\n")),
        Err(_) => Err(std::io::Error::other("event is not JSON-serializable")),
    }
}

/// Per-instance append fast-path state, held under one lock so repair, open,
/// and write serialize per `EventLog`.
struct EventLogAppendState {
    /// File length recorded after this instance's last fully successful append.
    /// `Some(len)` means the tail byte is this instance's own terminating
    /// newline as long as the file still ends at `len`; any other length
    /// (or `None`) makes the next append re-run the repair probe.
    clean_tail_len: Option<u64>,
    /// Open-once append handle. Dropped whenever the log is (re)created so
    /// deletion-then-append keeps its `on_create` lead-record semantics.
    append_handle: Option<std::fs::File>,
}

impl Default for EventLogAppendState {
    fn default() -> Self {
        Self {
            clean_tail_len: None,
            append_handle: None,
        }
    }
}

/// Early flush threshold for buffered appends, mirroring the session JSONL
/// buffer cap: one oversized ledger line (or a burst) must not wait a full
/// period.
const EVENT_LOG_WRITE_BUFFER_CAP_BYTES: usize = 128 * 1024;

struct EventLogBufferInner {
    timed: TimedBytes,
    /// Whether the target file existed on disk at the last accept (or after
    /// the last successful drain). An accept that finds the file missing
    /// drops pending bytes ONLY when they were accepted against an existing
    /// file — the write-through path would have lost those bytes to that
    /// external deletion. Bytes accepted while the file was absent are the
    /// creation still in flight and are kept.
    file_was_present: bool,
    /// First drain failure is retained (with the failed bytes dropped): a
    /// partially-landed batch must never be retried whole, because the retry
    /// would duplicate committed records into the log interior. Every later
    /// append or replay surfaces the error until the process restarts.
    pending_error: Option<String>,
}

/// Pending append bytes for ONE event-log path, shared by every `EventLog`
/// instance open on that path: a second recorder on the same ledger must
/// observe the first instance's buffered records, and a replay from a fresh
/// instance must see bytes another instance accepted.
///
/// Reader contract (write-buffering audit): `replay_sync` and `Drop` drain
/// before reading, so every full-file reader sees a flush-consistent tail.
/// Stat-only observers (size/mtime cursors) simply see the file grow at flush
/// boundaries instead of per append; no content reader may bypass
/// `replay_sync` without first calling `flush_sync`.
struct EventLogSharedBuffer {
    path: String,
    max_bytes: Option<u64>,
    log: Option<EventLogLogger>,
    /// Flush period; zero never happens here (write-through instances never
    /// acquire a shared buffer). Captured from the first instance on the path.
    period: Duration,
    inner: Mutex<EventLogBufferInner>,
}

impl EventLogSharedBuffer {
    /// Land every pending byte. The torn-tail repair and the missing-file
    /// create run here, exactly where the write-through path runs them.
    fn drain_locked(&self, inner: &mut EventLogBufferInner, durable: bool) -> Result<(), String> {
        if inner.timed.bytes.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::take(&mut inner.timed.bytes);
        inner.timed.deadline = None;
        match self.write_pending(&bytes, durable) {
            Ok(()) => {
                inner.file_was_present = true;
                Ok(())
            }
            Err(error) => {
                inner.file_was_present = false;
                inner.pending_error = Some(error.clone());
                Err(error)
            }
        }
    }

    /// One flush's write: repair any torn tail, then append the batch through
    /// an append-mode handle so concurrent appenders never interleave. The
    /// repair needs a read-write handle (truncation is not guaranteed on
    /// append-only access), hence the reopen only on the torn path.
    fn write_pending(&self, bytes: &[u8], durable: bool) -> Result<(), String> {
        if std::fs::metadata(&self.path).is_err() {
            if let Some(parent) = std::path::Path::new(&self.path).parent() {
                std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
        }
        if let Some(parent) = std::path::Path::new(&self.path).parent() {
            set_dir_mode_700(parent);
        }
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&self.path)
            .map_err(|error| error.to_string())?;
        let size = file.metadata().map_err(|error| error.to_string())?.len();
        if let Some(max_bytes) = self.max_bytes {
            if size > max_bytes {
                return Err(format!(
                    "event log {} exceeds {} bytes ({}); refusing to read",
                    self.path, max_bytes, size
                ));
            }
        }
        if size > 0 {
            let mut last_byte = [0u8; 1];
            file.seek(SeekFrom::Start(size - 1))
                .map_err(|error| error.to_string())?;
            let read = file
                .read(&mut last_byte)
                .map_err(|error| error.to_string())?;
            if read == 1 && last_byte[0] != 0x0a {
                drop(file);
                let mut repair_handle = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.path)
                    .map_err(|error| error.to_string())?;
                repair_torn_tail(&mut repair_handle, &self.path, self.max_bytes, &self.log)?;
                drop(repair_handle);
                file = std::fs::OpenOptions::new()
                    .read(true)
                    .append(true)
                    .open(&self.path)
                    .map_err(|error| error.to_string())?;
            }
        }
        set_file_mode_600(&self.path);
        file.write_all(bytes).map_err(|error| error.to_string())?;
        if durable {
            file.sync_all().map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

impl TimedFlush for EventLogSharedBuffer {
    fn poll_flush(&self, now: Instant) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if inner.timed.deadline.is_some_and(|deadline| deadline <= now) {
            let _ = self.drain_locked(&mut inner, false);
        }
    }
}

/// One shared buffer per event-log path (Weak so unused entries vanish with
/// their last instance; the timed-flusher registry holds its own Weaks).
type EventLogBufferRegistry = std::collections::HashMap<String, Weak<EventLogSharedBuffer>>;

fn event_log_buffers() -> &'static Mutex<EventLogBufferRegistry> {
    static BUFFERS: OnceLock<Mutex<EventLogBufferRegistry>> = OnceLock::new();
    BUFFERS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn acquire_or_create_shared_buffer(
    path: &str,
    max_bytes: Option<u64>,
    log: Option<EventLogLogger>,
    period: Duration,
) -> Arc<EventLogSharedBuffer> {
    let mut registry = event_log_buffers()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.retain(|_, weak| weak.strong_count() > 0);
    if let Some(existing) = registry.get(path).and_then(|weak| weak.upgrade()) {
        return existing;
    }
    let buffer = Arc::new(EventLogSharedBuffer {
        path: path.to_string(),
        max_bytes,
        log,
        period,
        inner: Mutex::new(EventLogBufferInner {
            timed: TimedBytes::new(),
            file_was_present: false,
            pending_error: None,
        }),
    });
    register_timed_flush(&(buffer.clone() as Arc<dyn TimedFlush>));
    registry.insert(path.to_string(), Arc::downgrade(&buffer));
    buffer
}

/// Drain any shared buffer registered for `path`. Readers that must observe
/// every accepted append call this first; a drain failure fails the read
/// (data was lost; failing closed is the recovery contract).
fn drain_shared_event_log_buffer(path: &str) -> Result<(), String> {
    let buffer = {
        let registry = event_log_buffers()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.get(path).and_then(|weak| weak.upgrade())
    };
    match buffer {
        Some(buffer) => {
            let mut inner = buffer
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            buffer.drain_locked(&mut inner, false)
        }
        None => Ok(()),
    }
}

pub struct EventLog {
    pub path: String,
    options: EventLogOptions,
    append_state: std::sync::Mutex<EventLogAppendState>,
    /// Flush period captured at construction; zero keeps the write-through
    /// path (probe + open-once append handle) exactly as before buffering.
    write_period: Duration,
    /// Lazily acquired on the first buffered append: read-only instances
    /// (a fresh `EventLog` per `replay_sync`) must not grow the registry.
    shared_buffer: OnceLock<Arc<EventLogSharedBuffer>>,
}

impl EventLog {
    pub fn new(path: impl Into<String>, options: EventLogOptions) -> Self {
        Self {
            path: path.into(),
            options,
            append_state: std::sync::Mutex::new(EventLogAppendState::default()),
            write_period: write_flush_period(),
            shared_buffer: OnceLock::new(),
        }
    }

    /// Test-only period override (the env var is process-global and races
    /// parallel tests); huge periods pin bytes for deterministic assertions.
    #[cfg(test)]
    pub(crate) fn with_write_period(mut self, period: Duration) -> Self {
        self.write_period = period;
        self
    }

    /// Land any bytes this instance's path still holds in its shared write
    /// buffer. Public for callers (benches, tools) that read the file with
    /// something other than `replay_sync`.
    pub fn flush_sync(&self) -> Result<(), String> {
        drain_shared_event_log_buffer(&self.path)
    }

    fn acquire_shared_buffer(&self) -> Option<&Arc<EventLogSharedBuffer>> {
        if self.write_period.is_zero() {
            return None;
        }
        Some(self.shared_buffer.get_or_init(|| {
            acquire_or_create_shared_buffer(
                &self.path,
                self.options.max_bytes,
                self.options.log.clone(),
                self.write_period,
            )
        }))
    }

    /// Replay every terminated line through `parse`: return `Err` to reject a
    /// line, `Ok(None)` to skip one. The missing-file decision is made at the
    /// open, so no check-then-read window exists.
    pub fn replay_sync<T>(
        &self,
        parse: impl Fn(&str, usize) -> Result<Option<T>, String>,
        options: ReplayOptions,
    ) -> Result<Vec<T>, String> {
        // Write buffering defers disk visibility by up to one flush period;
        // a replay must observe every accepted append, so drain first. The
        // drain looks the shared buffer up by path: another instance may be
        // the one holding pending bytes.
        drain_shared_event_log_buffer(&self.path)?;
        let mut file = match std::fs::File::open(&self.path) {
            Ok(file) => file,
            Err(error) => {
                if !options.missing_file_throws && error.kind() == std::io::ErrorKind::NotFound {
                    return Ok(Vec::new());
                }
                return Err(error.to_string());
            }
        };
        let bytes = read_all_sync(&mut file, self.options.max_bytes, &self.path)
            .map_err(|error| error.to_string())?;
        let contents = String::from_utf8_lossy(&bytes).to_string();
        let ends_with_newline = contents.ends_with('\n');
        let raw_lines: Vec<&str> = contents.split('\n').collect();
        let mut events: Vec<T> = Vec::new();
        let mut record_count = 0usize;
        for (index, raw_line) in raw_lines.iter().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if index == raw_lines.len() - 1 && !ends_with_newline {
                if let Some(log) = &self.options.log {
                    log("ignored torn final line".to_string());
                }
                continue;
            }
            if let Some(max_records) = self.options.max_records {
                record_count += 1;
                if record_count > max_records {
                    return Err(format!(
                        "event log {} exceeds {} records; refusing to read",
                        self.path, max_records
                    ));
                }
            }
            if let Some(event) = parse(line, index)? {
                events.push(event);
            }
        }
        Ok(events)
    }

    /// Append events as one write; `durable` fsyncs before returning. When the
    /// file is created by this append, `onCreate`'s records lead the payload.
    /// An unserializable event throws before any byte (including repair) is
    /// written.
    ///
    /// Fast path: after this instance's own clean append, the repair probe is
    /// skipped while the file still ends at the length that append left (the
    /// tail byte is then this instance's terminating newline). The probe reruns
    /// on the first append per instance, after any failed append, and whenever
    /// the length differs — so a torn tail left by any other writer is still
    /// truncated before this append lands. The append handle is likewise held
    /// open across appends and reopened only when the log is (re)created.
    pub fn append_sync(
        &self,
        events: &[serde_json::Value],
        durable: bool,
        on_create: Option<&dyn Fn() -> Vec<serde_json::Value>>,
    ) -> Result<(), String> {
        let mut lines: Vec<String> = Vec::with_capacity(events.len());
        for event in events {
            lines.push(serialize_line(event).map_err(|error| error.to_string())?);
        }
        if let Some(buffer) = self.acquire_shared_buffer() {
            return self.buffered_append(buffer, lines, durable, on_create);
        }
        let mut state = self
            .append_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outcome = self.append_with_state(&mut state, lines, durable, on_create);
        if outcome.is_err() {
            // Any failure leaves the tail unverified and the handle suspect:
            // drop both so the next append re-runs the probe.
            state.clean_tail_len = None;
            state.append_handle = None;
        }
        outcome
    }

    fn append_with_state(
        &self,
        state: &mut EventLogAppendState,
        lines: Vec<String>,
        durable: bool,
        on_create: Option<&dyn Fn() -> Vec<serde_json::Value>>,
    ) -> Result<(), String> {
        let parent = std::path::Path::new(&self.path).parent();
        let mut lead_lines: Vec<String> = Vec::new();
        // Same existence predicate the per-append probe used (`Path::exists`):
        // any metadata failure counts as missing, so an externally deleted
        // file (possibly still delete-pending behind this instance's held
        // handle) recreates with `on_create` lead records exactly as before.
        let mut pre_append_len = std::fs::metadata(&self.path)
            .ok()
            .map(|metadata| metadata.len());
        if let Some(len) = pre_append_len {
            if let Some(max_bytes) = self.options.max_bytes {
                if len > max_bytes {
                    return Err(format!(
                        "event log {} exceeds {} bytes ({}); refusing to read",
                        self.path, max_bytes, len
                    ));
                }
            }
            if state.clean_tail_len != Some(len) {
                self.repair_tail_sync()?;
                // The tail was externally modified since this instance's last
                // clean append: the held handle may point at a file that is no
                // longer the one at this path, so drop it and let the append
                // reopen whatever the probe just validated.
                state.append_handle = None;
                // The repair may have truncated a torn tail: re-read the length
                // the append lands on (a vanished file recreates below).
                pre_append_len = std::fs::metadata(&self.path)
                    .ok()
                    .map(|metadata| metadata.len());
            }
        } else {
            // Creating the log: the parent directory may not exist yet, and a
            // handle from a previous life of the file must not survive.
            drop(state.append_handle.take());
            if let Some(parent) = parent {
                std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            if let Some(on_create) = on_create {
                for event in on_create() {
                    lead_lines.push(serialize_line(&event).map_err(|error| error.to_string())?);
                }
            }
        }
        if let Some(parent) = parent {
            set_dir_mode_700(parent);
        }
        let payload = [lead_lines, lines].concat().join("");
        let mut handle = match state.append_handle.take() {
            Some(handle) => handle,
            None => std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&self.path)
                .map_err(|error| error.to_string())?,
        };
        set_file_mode_600(&self.path);
        let buffer = payload.as_bytes();
        let written = handle.write(buffer).map_err(|error| error.to_string())?;
        if written < buffer.len() {
            // A short write must fail, not complete or reclaim: a second write could weld
            // into a rival's append, and reclaiming could destroy a rival's committed record.
            return Err(format!(
                "event log {}: short write ({} of {} bytes)",
                self.path,
                written,
                buffer.len()
            ));
        }
        if durable {
            handle.sync_all().map_err(|error| error.to_string())?;
        }
        state.clean_tail_len = Some(pre_append_len.unwrap_or(0) + buffer.len() as u64);
        state.append_handle = Some(handle);
        Ok(())
    }

    /// Buffered append: the payload lands in the path's shared buffer and one
    /// drain writes it (with any still-pending bytes) as a single batch.
    /// `on_create`'s records are decided at append time against the on-disk
    /// stat (the same predicate the write-through path uses), so a file that
    /// is missing now recreates with its lead records exactly as before.
    fn buffered_append(
        &self,
        buffer: &Arc<EventLogSharedBuffer>,
        lines: Vec<String>,
        durable: bool,
        on_create: Option<&dyn Fn() -> Vec<serde_json::Value>>,
    ) -> Result<(), String> {
        let file_missing = std::fs::metadata(&self.path).is_err();
        if file_missing {
            // The write-through create branch makes the ledger's first append
            // the de-facto creator of the artifact directory, and sibling
            // writers (kernel snapshots, tool outputs) rely on that side
            // effect. Keep it synchronous: creation decisions happen at
            // accept time; only the bytes are deferred to the drain.
            if let Some(parent) = std::path::Path::new(&self.path).parent() {
                std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
        }
        let mut inner = buffer
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(error) = inner.pending_error.clone() {
            return Err(error);
        }
        let mut payload = String::new();
        if file_missing {
            // An external actor removed the target file while earlier
            // accepted bytes were still in flight. The write-through path
            // would have lost those same bytes to the deletion, and
            // `on_create`'s records must lead whatever this drain creates —
            // so the orphaned bytes are dropped, not welded after the lead
            // records. Bytes accepted against a missing file are the
            // creation still in flight: they stay, and no lead records are
            // taken (the write-through append would have found the file its
            // own earlier append created).
            if !inner.timed.bytes.is_empty() && inner.file_was_present {
                inner.timed.bytes.clear();
                inner.timed.deadline = None;
                inner.file_was_present = false;
            }
            if inner.timed.bytes.is_empty() {
                if let Some(on_create) = on_create {
                    for event in on_create() {
                        payload
                            .push_str(&serialize_line(&event).map_err(|error| error.to_string())?);
                    }
                }
            }
        }
        for line in lines {
            payload.push_str(&line);
        }
        inner.file_was_present = !file_missing;
        inner
            .timed
            .push(payload.as_bytes(), buffer.period, Instant::now());
        if durable || inner.timed.bytes.len() >= EVENT_LOG_WRITE_BUFFER_CAP_BYTES {
            return buffer.drain_locked(&mut inner, durable);
        }
        Ok(())
    }

    /// Truncate an unterminated tail before appending (module-doc tail rule); a failure gates the append.
    fn repair_tail_sync(&self) -> Result<(), String> {
        let mut file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.to_string()),
        };
        repair_torn_tail(
            &mut file,
            &self.path,
            self.options.max_bytes,
            &self.options.log,
        )
    }
}

/// Truncate an unterminated tail on an open read-write handle: unstable bytes
/// mean a live rival writer whose own append terminates the tail.
fn repair_torn_tail(
    file: &mut std::fs::File,
    path: &str,
    max_bytes: Option<u64>,
    log: &Option<EventLogLogger>,
) -> Result<(), String> {
    let size = file.metadata().map_err(|error| error.to_string())?.len();
    if size == 0 {
        return Ok(());
    }
    if let Some(max_bytes) = max_bytes {
        if size > max_bytes {
            return Err(format!(
                "event log {} exceeds {} bytes ({}); refusing to read",
                path, max_bytes, size
            ));
        }
    }
    // All offsets are BYTE offsets on raw buffers: string indices diverge
    // from byte offsets as soon as any record carries multi-byte UTF-8,
    // and ftruncate takes bytes.
    {
        let mut last_byte = [0u8; 1];
        file.seek(SeekFrom::Start(size - 1))
            .map_err(|error| error.to_string())?;
        let read = file
            .read(&mut last_byte)
            .map_err(|error| error.to_string())?;
        if read != 1 || last_byte[0] == 0x0a {
            return Ok(());
        }
    }
    // Double-read stability: unstable bytes mean a live rival writer whose own append terminates the tail.
    let first = read_all_sync(file, max_bytes, path).map_err(|error| error.to_string())?;
    let second = read_all_sync(file, max_bytes, path).map_err(|error| error.to_string())?;
    if second.len() != first.len() || second != first {
        return Ok(());
    }
    if file.metadata().map_err(|error| error.to_string())?.len() != first.len() as u64 {
        return Ok(());
    }
    let keep = first
        .iter()
        .rposition(|byte| *byte == 0x0a)
        .map(|index| index + 1)
        .unwrap_or(0) as u64;
    file.set_len(keep).map_err(|error| error.to_string())?;
    if let Some(log) = log {
        log(format!(
            "truncated torn final line ({} bytes)",
            first.len() as u64 - keep
        ));
    }
    Ok(())
}

/// Best-effort final drain: a dropped instance that buffered appends lands
/// them, mirroring the session manager's Drop drain. Instances that only
/// read never acquired a shared buffer, so readers do not force flushes.
impl Drop for EventLog {
    fn drop(&mut self) {
        if self.shared_buffer.get().is_some() {
            let _ = drain_shared_event_log_buffer(&self.path);
        }
    }
}

#[cfg(unix)]
fn set_file_mode_600(path: &str) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn set_file_mode_600(_path: &str) {}

#[cfg(unix)]
fn set_dir_mode_700(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn set_dir_mode_700(_path: &std::path::Path) {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse_line(line: &str, index: usize) -> Result<Option<serde_json::Value>, String> {
        serde_json::from_str::<serde_json::Value>(line)
            .map(Some)
            .map_err(|error| format!("corrupt semantic-edge ledger line {}: {}", index + 1, error))
    }

    #[test]
    fn missing_file_replays_empty_unless_it_must_throw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.jsonl");
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
        assert!(log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap()
            .is_empty());
        let error = log
            .replay_sync(
                parse_line,
                ReplayOptions {
                    missing_file_throws: true,
                },
            )
            .unwrap_err();
        assert!(!error.is_empty());
    }

    #[test]
    fn append_then_replay_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.jsonl");
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
        log.append_sync(&[json!({"a": 1}), json!({"b": 2})], false, None)
            .unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events, vec![json!({"a": 1}), json!({"b": 2})]);
    }

    #[test]
    fn torn_final_line_is_skipped_on_read_and_truncated_on_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("torn.jsonl");
        let path_str = path.to_string_lossy().to_string();
        std::fs::write(&path, b"{\"a\":1}\n{\"b\":2}").unwrap();
        let log = EventLog::new(path_str.clone(), EventLogOptions::default());
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events, vec![json!({"a": 1})]);
        log.append_sync(&[json!({"c": 3})], false, None).unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events, vec![json!({"a": 1}), json!({"c": 3})]);
    }

    #[test]
    fn interior_malformed_lines_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        std::fs::write(&path, b"{\"a\":1}\nnot json\n{\"b\":2}\n").unwrap();
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
        let error = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap_err();
        assert!(error.starts_with("corrupt semantic-edge ledger line 2:"));
    }

    #[test]
    fn bounds_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bounded.jsonl");
        std::fs::write(&path, b"{\"a\":1}\n{\"b\":2}\n").unwrap();
        let path_str = path.to_string_lossy().to_string();
        let bytes_log = EventLog::new(
            path_str.clone(),
            EventLogOptions {
                max_bytes: Some(4),
                ..Default::default()
            },
        );
        let error = bytes_log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap_err();
        assert!(error.contains("exceeds 4 bytes"));
        let records_log = EventLog::new(
            path_str,
            EventLogOptions {
                max_records: Some(1),
                ..Default::default()
            },
        );
        let error = records_log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap_err();
        assert!(error.contains("exceeds 1 records"));
    }

    #[test]
    fn create_hook_records_lead_the_first_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("created.jsonl");
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
        log.append_sync(
            &[json!({"second": true})],
            false,
            Some(&|| vec![json!({"first": true})]),
        )
        .unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(
            events,
            vec![json!({"first": true}), json!({"second": true})]
        );
        // A second append does not re-run the create hook.
        log.append_sync(
            &[json!({"third": true})],
            false,
            Some(&|| vec![json!({"again": true})]),
        )
        .unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn fast_path_appends_after_external_torn_tail_still_repair() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("guarded.jsonl");
        // These tests pin the write-through interleavings (probe, held
        // handle, on-file-replacement races) against the physical file, so
        // they run with buffering disabled.
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::ZERO);
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        // An external writer leaves a torn tail between appends: the length no
        // longer matches this instance's last clean append, so the probe must
        // run and truncate it before the next append lands.
        let mut rival = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        rival.write_all(b"{\"torn\":1").unwrap();
        drop(rival);
        log.append_sync(&[json!({"b": 2})], false, None).unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events, vec![json!({"a": 1}), json!({"b": 2})]);
    }

    #[test]
    fn fast_path_reopens_after_an_external_file_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("replaced.jsonl");
        // These tests pin the write-through interleavings (probe, held
        // handle, on-file-replacement races) against the physical file, so
        // they run with buffering disabled.
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::ZERO);
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        // An external actor replaces the ledger with a different file at the
        // same path: the next append must land on the file at the path, not on
        // the detached pre-replacement handle.
        let replacement = dir.path().join("replacement.jsonl");
        std::fs::write(&replacement, b"{\"rival\":1}\n{\"other\":2}\n").unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        log.append_sync(&[json!({"b": 2})], false, None).unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(
            events,
            vec![json!({"rival": 1}), json!({"other": 2}), json!({"b": 2})]
        );
    }

    #[test]
    fn fast_path_tolerates_external_complete_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.jsonl");
        // These tests pin the write-through interleavings (probe, held
        // handle, on-file-replacement races) against the physical file, so
        // they run with buffering disabled.
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::ZERO);
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        let mut rival = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        rival.write_all(b"{\"rival\":true}\n").unwrap();
        drop(rival);
        log.append_sync(&[json!({"b": 2})], false, None).unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(
            events,
            vec![json!({"a": 1}), json!({"rival": true}), json!({"b": 2})]
        );
    }

    #[test]
    fn failed_append_leaves_the_tail_unverified_for_the_next_probe() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("failed.jsonl");
        // These tests pin the write-through interleavings (probe, held
        // handle, on-file-replacement races) against the physical file, so
        // they run with buffering disabled.
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::ZERO);
        // A failing append (the path is a directory, so the open fails) must
        // leave the tail unverified.
        std::fs::create_dir(&path).unwrap();
        assert!(log.append_sync(&[json!({"b": 2})], false, None).is_err());
        std::fs::remove_dir(&path).unwrap();
        std::fs::write(&path, b"{\"a\":1}\n{\"torn\":1").unwrap();
        // The torn tail must be truncated before the append lands.
        log.append_sync(&[json!({"c": 3})], false, None).unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events, vec![json!({"a": 1}), json!({"c": 3})]);
    }

    #[test]
    fn recreated_log_leads_with_on_create_records_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("recreated.jsonl");
        // These tests pin the write-through interleavings (probe, held
        // handle, on-file-replacement races) against the physical file, so
        // they run with buffering disabled.
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::ZERO);
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        std::fs::remove_file(&path).unwrap();
        log.append_sync(
            &[json!({"b": 2})],
            false,
            Some(&|| vec![json!({"first": true})]),
        )
        .unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events, vec![json!({"first": true}), json!({"b": 2})]);
    }

    #[test]
    fn fast_path_enforces_bounds_without_the_probe() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bounded-append.jsonl");
        // The bound is enforced at append time on the write-through path.
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions {
                max_bytes: Some(7),
                ..Default::default()
            },
        )
        .with_write_period(Duration::ZERO);
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        // The file (8 bytes) is beyond the bound; the append must refuse even
        // on the probe-skipping fast path.
        let error = log
            .append_sync(&[json!({"b": 2})], false, None)
            .unwrap_err();
        assert!(error.contains("exceeds 7 bytes"), "{error}");
    }

    #[test]
    fn buffered_appends_land_only_on_drain() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("buffered.jsonl");
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::from_secs(3600));
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        log.append_sync(&[json!({"b": 2})], false, None).unwrap();
        assert!(
            !path.exists(),
            "nothing may hit the disk before a drain or the period"
        );
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events, vec![json!({"a": 1}), json!({"b": 2})]);
    }

    #[test]
    fn cross_instance_replay_sees_another_instances_buffered_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cross.jsonl");
        let writer = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::from_secs(3600));
        writer.append_sync(&[json!({"a": 1})], false, None).unwrap();
        // A fresh reader instance never appended; it must still observe the
        // writer's pending bytes (the buffer is shared per path).
        let reader = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
        let events = reader
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events, vec![json!({"a": 1})]);
    }

    #[test]
    fn durable_append_drains_pending_bytes_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("durable.jsonl");
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::from_secs(3600));
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        // The durable append must land the earlier buffered line first, then
        // itself, then fsync the whole file before returning.
        log.append_sync(&[json!({"d": 1})], true, None).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        let events: Vec<serde_json::Value> = contents
            .trim_end()
            .split('\n')
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events, vec![json!({"a": 1}), json!({"d": 1})]);
    }

    #[test]
    fn oversized_buffered_append_flushes_at_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("capped.jsonl");
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::from_secs(3600));
        // One line past the 128KiB cap flushes immediately, no period wait.
        let big = "x".repeat(200 * 1024);
        log.append_sync(&[json!({"big": big})], false, None)
            .unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains(&"x".repeat(1024)));
        assert!(contents.ends_with('\n'));
    }

    #[test]
    fn externally_deleted_file_drops_in_flight_bytes_and_releads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deleted.jsonl");
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::from_secs(3600));
        // A durable append lands immediately, so the file exists.
        log.append_sync(&[json!({"a": 1})], true, None).unwrap();
        // Then a buffered line goes in flight and the file is deleted by an
        // external actor before it lands.
        log.append_sync(&[json!({"b": 2})], false, None).unwrap();
        std::fs::remove_file(&path).unwrap();
        // The recreate append: the in-flight line belonged to the deleted
        // file and is dropped, and `on_create`'s records lead the new file.
        log.append_sync(
            &[json!({"c": 3})],
            false,
            Some(&|| vec![json!({"first": true})]),
        )
        .unwrap();
        let events = log
            .replay_sync(parse_line, ReplayOptions::default())
            .unwrap();
        assert_eq!(events, vec![json!({"first": true}), json!({"c": 3})]);
    }

    #[test]
    fn sticky_drain_error_fails_later_appends() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sticky.jsonl");
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        )
        .with_write_period(Duration::from_secs(3600));
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        // Make the drain fail: the target path is now a directory.
        std::fs::create_dir(&path).unwrap();
        assert!(log.flush_sync().is_err());
        // The failure is sticky: the next append fails closed without a write.
        assert!(log.append_sync(&[json!({"b": 2})], false, None).is_err());
        std::fs::remove_dir(&path).unwrap();
    }

    #[test]
    fn drop_drains_pending_event_log_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dropped.jsonl");
        {
            let log = EventLog::new(
                path.to_string_lossy().to_string(),
                EventLogOptions::default(),
            )
            .with_write_period(Duration::from_secs(3600));
            log.append_sync(&[json!({"a": 1})], false, None).unwrap();
            log.append_sync(&[json!({"b": 2})], false, None).unwrap();
        }
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.matches('\n').count(), 2);
    }

    #[test]
    fn timed_flusher_lands_event_log_bytes_within_the_period() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("timed.jsonl");
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        // No explicit drain: the background flusher must land the line within
        // the flush period plus poll jitter.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let contents = std::fs::read_to_string(&path).unwrap_or_default();
            if contents.matches("a").count() >= 1 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the timed flusher did not land the buffered line"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn torn_tail_repair_is_logged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("logged.jsonl");
        std::fs::write(&path, b"{\"a\":1}\n{\"b\":2}").unwrap();
        let messages: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = messages.clone();
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions {
                log: Some(std::sync::Arc::new(move |message| {
                    // Poison-tolerant: a test-thread panic must not turn the
                    // Drop drain's logging into a double panic (abort).
                    recorder
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(message);
                })),
                ..Default::default()
            },
        )
        // The repair log line is emitted where the repair runs; buffered
        // appends repair at drain time, so pin write-through here.
        .with_write_period(Duration::ZERO);
        log.append_sync(&[json!({"c": 3})], false, None).unwrap();
        let messages = messages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(messages
            .iter()
            .any(|message| message.starts_with("truncated torn final line")));
    }
}
