//! Port of packages/coding-agent/src/modes/daemon/command-recovery-journal.ts

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// UNKNOWN: ['::{DaemonResponse', 'DaemonSavedSessionInfo}']
use super::daemon_protocol::{DaemonResponse, DaemonSavedSessionInfo};

use crate::utils::atomic_file::{write_file_atomic_sync, WriteFileAtomicOptions};

const COMPACT_AFTER_RECORDS: usize = 4096;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceivedRecord {
    pub version: u32,
    #[serde(rename = "type")]
    pub type_: String,
    pub key: String,
    #[serde(rename = "clientId")]
    pub client_id: String,
    #[serde(rename = "commandId")]
    pub command_id: String,
    #[serde(rename = "commandType")]
    pub command_type: String,
    #[serde(rename = "recordedAt")]
    pub recorded_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultRecord {
    pub version: u32,
    #[serde(rename = "type")]
    pub type_: String,
    pub key: String,
    pub response: DaemonResponse,
    #[serde(rename = "recordedAt")]
    pub recorded_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcknowledgedRecord {
    pub version: u32,
    #[serde(rename = "type")]
    pub type_: String,
    pub key: String,
    #[serde(rename = "recordedAt")]
    pub recorded_at: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JournalRecord {
    Received(ReceivedRecord),
    Result(ResultRecord),
    Acknowledged(AcknowledgedRecord),
}

impl JournalRecord {
    pub fn version(&self) -> u32 {
        match self {
            JournalRecord::Received(record) => record.version,
            JournalRecord::Result(record) => record.version,
            JournalRecord::Acknowledged(record) => record.version,
        }
    }

    pub fn key(&self) -> &str {
        match self {
            JournalRecord::Received(record) => &record.key,
            JournalRecord::Result(record) => &record.key,
            JournalRecord::Acknowledged(record) => &record.key,
        }
    }

    pub fn type_(&self) -> &str {
        match self {
            JournalRecord::Received(record) => &record.type_,
            JournalRecord::Result(record) => &record.type_,
            JournalRecord::Acknowledged(record) => &record.type_,
        }
    }

    pub fn to_value(&self) -> Value {
        match self {
            JournalRecord::Received(record) => serde_json::to_value(record),
            JournalRecord::Result(record) => serde_json::to_value(record),
            JournalRecord::Acknowledged(record) => serde_json::to_value(record),
        }
        .unwrap_or(Value::Null)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct JournalEntry {
    received: ReceivedRecord,
    response: Option<DaemonResponse>,
}

/// Bounded observability of unresolved command-journal entries (audit D-07).
/// `total` counts received commands without an acknowledgement; `without_result`
/// is the uncertain subset whose result never arrived.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandJournalPendingSummary {
    pub total: usize,
    pub without_result: usize,
    pub oldest_recorded_at: Option<String>,
    pub oldest_command_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CommandJournalBeginResult {
    New,
    Pending,
    Complete(DaemonResponse),
}

pub fn create_command_idempotency_key(client_id: &str, command_id: &str) -> String {
    serde_json::to_string(&[client_id, command_id]).unwrap_or_else(|_| format!("[\"{client_id}\",\"{command_id}\"]"))
}

/// Append-only command journal used at the supervisor boundary. A received
/// record is durable before a mutating command is dispatched; a missing result
/// after a crash is therefore treated as uncertain and is never replayed.
pub struct CommandRecoveryJournal {
    path: String,
    entries: HashMap<String, JournalEntry>,
    record_count: usize,
}

impl CommandRecoveryJournal {
    pub fn new(path: &str) -> Result<Self, String> {
        if let Some(parent) = Path::new(path).parent() {
            create_private_dir(parent).map_err(|error| error.to_string())?;
        }
        let mut journal = Self {
            path: path.to_string(),
            entries: HashMap::new(),
            record_count: 0,
        };
        journal.load()?;
        Ok(journal)
    }

    pub fn lookup(&self, client_id: &str, command_id: &str) -> Option<CommandJournalBeginResult> {
        let existing = self.entries.get(&create_command_idempotency_key(client_id, command_id));
        match existing {
            Some(entry) => match &entry.response {
                Some(response) => Some(CommandJournalBeginResult::Complete(response.clone())),
                None => Some(CommandJournalBeginResult::Pending),
            },
            None => None,
        }
    }

    pub fn begin(&mut self, client_id: &str, command_id: &str, command_type: &str) -> Result<CommandJournalBeginResult, String> {
        let key = create_command_idempotency_key(client_id, command_id);
        if let Some(existing) = self.lookup(client_id, command_id) {
            return Ok(existing);
        }
        let received = ReceivedRecord {
            version: 1,
            type_: "received".to_string(),
            key: key.clone(),
            client_id: client_id.to_string(),
            command_id: command_id.to_string(),
            command_type: command_type.to_string(),
            recorded_at: now_iso(),
        };
        self.append(&JournalRecord::Received(received.clone()))?;
        self.entries.insert(
            key,
            JournalEntry {
                received,
                response: None,
            },
        );
        Ok(CommandJournalBeginResult::New)
    }

    pub fn record_result(&mut self, client_id: &str, command_id: &str, response: DaemonResponse) -> Result<(), String> {
        let key = create_command_idempotency_key(client_id, command_id);
        if !self.entries.contains_key(&key) {
            return Err(format!("Cannot record a result before command receipt: {key}"));
        }
        let record = ResultRecord {
            version: 1,
            type_: "result".to_string(),
            key: key.clone(),
            response: response.clone(),
            recorded_at: now_iso(),
        };
        self.append(&JournalRecord::Result(record))?;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.response = Some(response);
        }
        if self.record_count >= COMPACT_AFTER_RECORDS {
            self.compact()?;
        }
        Ok(())
    }

    /// Same as `record_result`, but reports the TypeScript error condition.
    pub fn try_record_result(
        &mut self,
        client_id: &str,
        command_id: &str,
        response: DaemonResponse,
    ) -> Result<(), String> {
        let key = create_command_idempotency_key(client_id, command_id);
        if !self.entries.contains_key(&key) {
            return Err(format!("Cannot record a result before command receipt: {key}"));
        }
        self.record_result(client_id, command_id, response)
    }

    /// Snapshot of unresolved entries, oldest first by `recordedAt`.
    pub fn pending_summary(&self) -> CommandJournalPendingSummary {
        let mut summary = CommandJournalPendingSummary::default();
        let mut oldest: Option<(String, &str, &str)> = None;
        for entry in self.entries.values() {
            summary.total += 1;
            if entry.response.is_none() {
                summary.without_result += 1;
            }
            let candidate = (
                entry.received.recorded_at.clone(),
                entry.received.command_type.as_str(),
                entry.received.command_id.as_str(),
            );
            let older = match &oldest {
                None => true,
                Some((at, _, _)) => candidate.0 < *at,
            };
            if candidate.0.is_empty() { continue; }
            if older {
                oldest = Some(candidate);
            }
        }
        if let Some((recorded_at, command_type, _)) = oldest {
            summary.oldest_recorded_at = Some(recorded_at);
            summary.oldest_command_type = Some(command_type.to_string());
        }
        summary
    }

    pub fn acknowledge(&mut self, client_id: &str, command_id: &str) -> Result<(), String> {
        let key = create_command_idempotency_key(client_id, command_id);
        if !self.entries.contains_key(&key) {
            return Ok(());
        }
        self.append(&JournalRecord::Acknowledged(AcknowledgedRecord {
            version: 1,
            type_: "acknowledged".to_string(),
            key: key.clone(),
            recorded_at: now_iso(),
        }))?;
        self.entries.remove(&key);
        if self.entries.is_empty() || self.record_count >= COMPACT_AFTER_RECORDS {
            self.compact()?;
        }
        Ok(())
    }

    fn load(&mut self) -> Result<(), String> {
        let contents = match std::fs::read_to_string(&self.path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.to_string()),
        };
        for line in contents.split('\n') {
            if line.is_empty() {
                continue;
            }
            // A crash may leave only the final append truncated.
            let Ok(record) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let Some(candidate) = record.as_object() else {
                continue;
            };
            if candidate.get("version").and_then(Value::as_u64) != Some(1) {
                continue;
            }
            let Some(key) = candidate.get("key").and_then(Value::as_str) else {
                continue;
            };
            let key = key.to_string();
            self.record_count += 1;
            match candidate.get("type").and_then(Value::as_str) {
                Some("received") => {
                    if let Ok(received) = serde_json::from_value::<ReceivedRecord>(record.clone()) {
                        if !received.client_id.is_empty()
                            && !received.command_id.is_empty()
                            && !received.command_type.is_empty()
                        {
                            self.entries.insert(
                                key,
                                JournalEntry {
                                    received,
                                    response: None,
                                },
                            );
                        }
                    }
                }
                Some("acknowledged") => {
                    self.entries.remove(&key);
                }
                Some("result") => {
                    let Ok(result) = serde_json::from_value::<ResultRecord>(record.clone()) else {
                        continue;
                    };
                    if result.response.type_ != "response" {
                        continue;
                    }
                    if let Some(entry) = self.entries.get_mut(&key) {
                        entry.response = Some(result.response);
                    }
                }
                _ => {}
            }
        }
            Ok(())
    }

    fn append(&mut self, record: &JournalRecord) -> Result<(), String> {
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        { use std::os::unix::fs::OpenOptionsExt; options.mode(0o600); }
        let mut file = options.open(&self.path).map_err(|error| error.to_string())?;
        let line = serde_json::to_string(&record.to_value()).map_err(|error| error.to_string())?;
        file.write_all(format!("{line}\n").as_bytes()).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        set_private_mode(&self.path).map_err(|error| error.to_string())?;
        self.record_count += 1;
        Ok(())
    }

    fn compact(&mut self) -> Result<(), String> {
        let mut records: Vec<JournalRecord> = Vec::new();
        for (key, entry) in &self.entries {
            records.push(JournalRecord::Received(entry.received.clone()));
            if let Some(response) = &entry.response {
                records.push(JournalRecord::Result(ResultRecord {
                    version: 1,
                    type_: "result".to_string(),
                    key: key.clone(),
                    response: response.clone(),
                    recorded_at: now_iso(),
                }));
            }
        }
        let payload: String = records
            .iter()
            .map(|record| serde_json::to_string(&record.to_value()).unwrap_or_default())
            .collect::<Vec<String>>()
            .join("\n");
        let payload = format!("{payload}\n");
        write_file_atomic_sync(
            &self.path,
            &payload,
            WriteFileAtomicOptions {
                mode: Some(0o600),
                fsync: true,
                fsync_dir: true,
                before_rename: None,
            },
        ).map_err(|error| error.to_string())?;
        self.record_count = records.len();
        Ok(())
    }
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn create_private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn set_private_mode(path: &str) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

/// A saved-session row stored inside a journal result record.
pub fn journal_result_saved_session(value: &Value) -> Option<DaemonSavedSessionInfo> {
    serde_json::from_value(value.clone()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("command-recovery-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join(name).to_string_lossy().to_string()
    }

    fn response(id: &str) -> DaemonResponse {
        DaemonResponse::success(Some(id), "prompt", None)
    }

    #[test]
    fn begin_lookup_and_complete_cycle() {
        let path = temp_path("journal.jsonl");
        let mut journal = CommandRecoveryJournal::new(&path).unwrap();
        assert_eq!(journal.begin("client", "c1", "prompt").unwrap(), CommandJournalBeginResult::New);
        assert_eq!(journal.lookup("client", "c1"), Some(CommandJournalBeginResult::Pending));
        assert_eq!(journal.begin("client", "c1", "prompt").unwrap(), CommandJournalBeginResult::Pending);
        journal.record_result("client", "c1", response("c1")).unwrap();
        assert_eq!(
            journal.lookup("client", "c1"),
            Some(CommandJournalBeginResult::Complete(response("c1")))
        );
        journal.acknowledge("client", "c1").unwrap();
        assert_eq!(journal.lookup("client", "c1"), None);
        assert_eq!(std::fs::read_to_string(&path).expect("journal").trim(), "");
    }

    #[test]
    fn records_survive_a_reload() {
        let path = temp_path("journal.jsonl");
        {
            let mut journal = CommandRecoveryJournal::new(&path).unwrap();
            journal.begin("client", "c2", "prompt").unwrap();
            journal.record_result("client", "c2", response("c2")).unwrap();
        }
        let journal = CommandRecoveryJournal::new(&path).unwrap();
        assert_eq!(
            journal.lookup("client", "c2"),
            Some(CommandJournalBeginResult::Complete(response("c2")))
        );
    }

    #[test]
    fn recording_before_receipt_is_an_error() {
        let path = temp_path("journal.jsonl");
        let mut journal = CommandRecoveryJournal::new(&path).unwrap();
        let error = journal
            .try_record_result("client", "missing", response("missing"))
            .expect_err("must fail");
        assert!(error.starts_with("Cannot record a result before command receipt:"));
    }

    #[test]
    fn truncated_final_line_is_ignored() {
        let path = temp_path("journal.jsonl");
        let mut journal = CommandRecoveryJournal::new(&path).unwrap();
        journal.begin("client", "c3", "prompt").unwrap();
        let mut contents = std::fs::read_to_string(&path).expect("journal");
        contents.push_str("{\"version\":1,\"type\":\"resu");
        std::fs::write(&path, contents).expect("seed truncation");
        let journal = CommandRecoveryJournal::new(&path).unwrap();
        assert_eq!(journal.lookup("client", "c3"), Some(CommandJournalBeginResult::Pending));
    }

    #[test]
    fn idempotency_key_is_a_two_element_json_array() {
        assert_eq!(create_command_idempotency_key("c", "1"), "[\"c\",\"1\"]");
    }

    #[test]
    fn pending_summary_reports_the_uncertain_backlog() {
        let path = temp_path("journal.jsonl");
        let mut journal = CommandRecoveryJournal::new(&path).unwrap();
        assert_eq!(journal.pending_summary(), CommandJournalPendingSummary::default());
        journal.begin("client", "c-1", "prompt").unwrap();
        journal.begin("client", "c-2", "create").unwrap();
        journal.record_result("client", "c-1", response("c-1")).unwrap();
        let summary = journal.pending_summary();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.without_result, 1);
        assert_eq!(summary.oldest_command_type.as_deref(), Some("prompt"));
        assert!(summary.oldest_recorded_at.is_some());
        // Acknowledged entries leave the backlog; a recorded result stays until acked.
        journal.acknowledge("client", "c-1").unwrap();
        let summary = journal.pending_summary();
        assert_eq!(summary.total, 1);
        assert_eq!(summary.without_result, 1);
        assert_eq!(summary.oldest_command_type.as_deref(), Some("create"));
        journal.acknowledge("client", "c-2").unwrap();
        assert_eq!(journal.pending_summary().total, 0);
    }

    #[test]
    fn failed_append_does_not_admit_or_complete_a_command() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("journal.jsonl");
        let mut journal = CommandRecoveryJournal::new(path.to_str().unwrap()).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(journal.begin("client", "new", "prompt").is_err());
        assert_eq!(journal.lookup("client", "new"), None);
        std::fs::remove_dir(&path).unwrap();
        journal.begin("client", "pending", "prompt").unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(journal.record_result("client", "pending", response("pending")).is_err());
        assert!(journal.acknowledge("client", "pending").is_err());
        assert_eq!(journal.lookup("client", "pending"), Some(CommandJournalBeginResult::Pending));
    }
}
