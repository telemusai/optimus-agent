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

pub struct EventLog {
    pub path: String,
    options: EventLogOptions,
    append_state: std::sync::Mutex<EventLogAppendState>,
}

impl EventLog {
    pub fn new(path: impl Into<String>, options: EventLogOptions) -> Self {
        Self {
            path: path.into(),
            options,
            append_state: std::sync::Mutex::new(EventLogAppendState::default()),
        }
    }

    /// Replay every terminated line through `parse`: return `Err` to reject a
    /// line, `Ok(None)` to skip one. The missing-file decision is made at the
    /// open, so no check-then-read window exists.
    pub fn replay_sync<T>(
        &self,
        parse: impl Fn(&str, usize) -> Result<Option<T>, String>,
        options: ReplayOptions,
    ) -> Result<Vec<T>, String> {
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

    /// Truncate an unterminated tail before appending (module-doc tail rule); a failure gates the append.
    fn repair_tail_sync(&self) -> Result<(), String> {
        let size = match std::fs::metadata(&self.path) {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.to_string()),
        };
        if size == 0 {
            return Ok(());
        }
        if let Some(max_bytes) = self.options.max_bytes {
            if size > max_bytes {
                return Err(format!(
                    "event log {} exceeds {} bytes ({}); refusing to read",
                    self.path, max_bytes, size
                ));
            }
        }
        // All offsets are BYTE offsets on raw buffers: string indices diverge
        // from byte offsets as soon as any record carries multi-byte UTF-8,
        // and ftruncate takes bytes.
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|error| error.to_string())?;
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
        let first = read_all_sync(&mut file, self.options.max_bytes, &self.path)
            .map_err(|error| error.to_string())?;
        let second = read_all_sync(&mut file, self.options.max_bytes, &self.path)
            .map_err(|error| error.to_string())?;
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
        if let Some(log) = &self.options.log {
            log(format!(
                "truncated torn final line ({} bytes)",
                first.len() as u64 - keep
            ));
        }
        Ok(())
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
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
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
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
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
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
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
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
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
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions::default(),
        );
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
        let log = EventLog::new(
            path.to_string_lossy().to_string(),
            EventLogOptions {
                max_bytes: Some(7),
                ..Default::default()
            },
        );
        log.append_sync(&[json!({"a": 1})], false, None).unwrap();
        // The file (8 bytes) is beyond the bound; the append must refuse even
        // on the probe-skipping fast path.
        let error = log
            .append_sync(&[json!({"b": 2})], false, None)
            .unwrap_err();
        assert!(error.contains("exceeds 7 bytes"), "{error}");
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
                    recorder.lock().unwrap().push(message);
                })),
                ..Default::default()
            },
        );
        log.append_sync(&[json!({"c": 3})], false, None).unwrap();
        let messages = messages.lock().unwrap();
        assert!(messages
            .iter()
            .any(|message| message.starts_with("truncated torn final line")));
    }
}
