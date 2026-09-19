//! Durable, content-free delivery telemetry for cross-worker agent messages (D-04).
//!
//! Cross-worker `agent_message.send` is at-most-once by design: a sent message is never
//! replayed after a lost response, because it may already have been accepted by the other
//! worker (`agent_message_transport.rs`). Before this journal the only durable delivery
//! evidence was transcript metadata inside the target session, so an operator could not
//! answer "was this child-to-child answer delivered?" from daemon state.
//!
//! The journal is telemetry only. It never changes delivery semantics, never retries, and
//! never carries message text, sender names, session names or error strings: only the
//! message id minted by the target worker, both active session ids, the outcome and a
//! fixed reason code. The file is append-only and self-compacting so it stays bounded.

use std::collections::VecDeque;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Keep at most this many records on disk and in memory.
pub const AGENT_MESSAGE_DELIVERY_JOURNAL_LIMIT: usize = 512;
/// Rewrite the file once this many records have been appended.
const COMPACT_AFTER_RECORDS: usize = 4096;

/// Terminal outcome of one supervisor-forwarded agent message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMessageDeliveryOutcome {
    Delivered,
    Queued,
    Rejected,
    /// The forward was attempted but the result is unknown (timeout, lost response,
    /// worker failure). The message is never replayed in that case.
    Uncertain,
}

impl AgentMessageDeliveryOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            AgentMessageDeliveryOutcome::Delivered => "delivered",
            AgentMessageDeliveryOutcome::Queued => "queued",
            AgentMessageDeliveryOutcome::Rejected => "rejected",
            AgentMessageDeliveryOutcome::Uncertain => "uncertain",
        }
    }

    /// Map a receipt's delivery status onto a journal outcome. Unknown status strings are
    /// rejected rather than recorded as a trusted outcome.
    pub fn from_receipt_status(status: &str) -> Option<Self> {
        match status {
            "delivered" => Some(AgentMessageDeliveryOutcome::Delivered),
            "queued" => Some(AgentMessageDeliveryOutcome::Queued),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentMessageDeliveryRecord {
    pub version: u32,
    /// Source session of the send, when the supervisor could resolve it.
    #[serde(rename = "sourceActiveSessionId", skip_serializing_if = "Option::is_none", default)]
    pub source_active_session_id: Option<String>,
    #[serde(rename = "targetActiveSessionId")]
    pub target_active_session_id: String,
    /// Message id minted by the target worker (`agentmsg_<uuid>`). Absent when the send
    /// never produced a receipt.
    #[serde(rename = "messageId", skip_serializing_if = "Option::is_none", default)]
    pub message_id: Option<String>,
    pub outcome: AgentMessageDeliveryOutcome,
    /// Fixed reason code for a non-delivery: `worker_error`, `worker_unavailable`,
    /// `unknown_active_session`, `agent_reach_denied`, `self_target`, `missing_source`,
    /// or `invalid_receipt`. Never the raw error text.
    #[serde(rename = "reasonCode", skip_serializing_if = "Option::is_none", default)]
    pub reason_code: Option<String>,
    #[serde(rename = "recordedAt")]
    pub recorded_at: String,
}

impl AgentMessageDeliveryRecord {
    pub fn new(
        source_active_session_id: Option<&str>,
        target_active_session_id: &str,
        message_id: Option<&str>,
        outcome: AgentMessageDeliveryOutcome,
        reason_code: Option<&str>,
    ) -> Self {
        Self {
            version: 1,
            // Identifiers are sanitized the same way as the metric sidecar: control
            // characters are stripped and length is bounded, so no caller can use the
            // journal to persist arbitrary text.
            source_active_session_id: source_active_session_id.and_then(|value| sanitize_id(value)),
            target_active_session_id: sanitize_id(target_active_session_id).unwrap_or_default(),
            message_id: message_id
                .filter(|value| is_agent_message_id(value))
                .map(str::to_string),
            outcome,
            reason_code: reason_code.map(str::to_string),
            recorded_at: now_iso(),
        }
    }
}

/// Only the target worker's own id shape is accepted, so a spoofed or oversized id is
/// never written to durable state.
fn is_agent_message_id(value: &str) -> bool {
    value.starts_with("agentmsg_")
        && value.len() <= 128
        && !value.chars().any(|character| character.is_control())
}

fn sanitize_id(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let bounded: String = trimmed
        .chars()
        .filter(|character| !character.is_control())
        .take(128)
        .collect();
    if bounded.is_empty() {
        None
    } else {
        Some(bounded)
    }
}

/// Append-only bounded journal of agent-message delivery outcomes.
pub struct AgentMessageDeliveryJournal {
    path: String,
    records: VecDeque<AgentMessageDeliveryRecord>,
    appended: usize,
}

impl AgentMessageDeliveryJournal {
    pub fn new(path: &str) -> Self {
        if let Some(parent) = Path::new(path).parent() {
            let _ = create_private_dir(&parent.to_string_lossy());
        }
        let records = parse_records(path);
        Self {
            path: path.to_string(),
            records,
            appended: 0,
        }
    }

    /// Append one outcome record. Failure to persist is not allowed to change delivery.
    pub fn record(&mut self, record: AgentMessageDeliveryRecord) {
        if self.append(&record).is_err() {
            return;
        }
        self.records.push_back(record);
        while self.records.len() > AGENT_MESSAGE_DELIVERY_JOURNAL_LIMIT {
            self.records.pop_front();
        }
        self.appended += 1;
        if self.appended >= COMPACT_AFTER_RECORDS {
            let _ = self.compact();
        }
    }

    pub fn records(&self) -> Vec<AgentMessageDeliveryRecord> {
        self.records.iter().cloned().collect()
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// Read the durable records, newest last. Bounded to the documented limit.
    pub fn read(path: &str) -> Vec<AgentMessageDeliveryRecord> {
        parse_records(path).into_iter().collect()
    }

    fn append(&mut self, record: &AgentMessageDeliveryRecord) -> Result<(), String> {
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&self.path).map_err(|error| error.to_string())?;
        let line = serde_json::to_string(record).map_err(|error| error.to_string())?;
        file.write_all(format!("{line}\n").as_bytes())
            .map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        set_private_mode(&self.path);
        Ok(())
    }

    fn compact(&mut self) -> Result<(), String> {
        let payload: String = self
            .records
            .iter()
            .map(|record| serde_json::to_string(record).unwrap_or_default())
            .collect::<Vec<String>>()
            .join("\n");
        let payload = format!("{payload}\n");
        let temp_path = format!("{}.{}.tmp", self.path, std::process::id());
        std::fs::write(&temp_path, payload).map_err(|error| error.to_string())?;
        set_private_mode(&temp_path);
        std::fs::rename(&temp_path, &self.path).map_err(|error| error.to_string())?;
        self.appended = 0;
        Ok(())
    }
}

fn parse_records(path: &str) -> VecDeque<AgentMessageDeliveryRecord> {
    let mut records: VecDeque<AgentMessageDeliveryRecord> = VecDeque::new();
    let Ok(contents) = std::fs::read_to_string(path) else {
        return records;
    };
    for line in contents.split('\n') {
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(record) = delivery_record_from_value(&value) else {
            continue;
        };
        records.push_back(record);
    }
    while records.len() > AGENT_MESSAGE_DELIVERY_JOURNAL_LIMIT {
        records.pop_front();
    }
    records
}

/// True when a parsed JSON value is a usable delivery record.
pub fn delivery_record_from_value(value: &Value) -> Option<AgentMessageDeliveryRecord> {
    let candidate = value.as_object()?;
    if candidate.get("version").and_then(Value::as_u64) != Some(1) {
        return None;
    }
    let target = candidate.get("targetActiveSessionId").and_then(Value::as_str)?;
    if target.is_empty() {
        return None;
    }
    let outcome = candidate.get("outcome").and_then(Value::as_str)?;
    let outcome = match outcome {
        "delivered" => AgentMessageDeliveryOutcome::Delivered,
        "queued" => AgentMessageDeliveryOutcome::Queued,
        "rejected" => AgentMessageDeliveryOutcome::Rejected,
        "uncertain" => AgentMessageDeliveryOutcome::Uncertain,
        _ => return None,
    };
    Some(AgentMessageDeliveryRecord {
        version: 1,
        source_active_session_id: candidate
            .get("sourceActiveSessionId")
            .and_then(Value::as_str)
            .map(str::to_string),
        target_active_session_id: target.to_string(),
        message_id: candidate
            .get("messageId")
            .and_then(Value::as_str)
            .filter(|value| is_agent_message_id(value))
            .map(str::to_string),
        outcome,
        reason_code: candidate
            .get("reasonCode")
            .and_then(Value::as_str)
            .map(str::to_string),
        recorded_at: candidate
            .get("recordedAt")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    })
}

/// File name used under the supervisor descriptor directory.
pub const AGENT_MESSAGE_DELIVERY_JOURNAL_FILE: &str = "agent-message-delivery-journal.jsonl";

/// Fixed, content-free reason code for a failed forward. Never the raw error text.
pub fn delivery_reason_code(error: &str) -> &'static str {
    if error.starts_with("Unknown active session:") {
        "unknown_active_session"
    } else if error.starts_with("Agent reach is limited") {
        "agent_reach_denied"
    } else if error.starts_with("Agent messaging cannot target the sending session") {
        "self_target"
    } else if error.starts_with("Agent messaging requires fromActiveSessionId") {
        "missing_source"
    } else if error.starts_with("Session worker ") && error.contains("exited") {
        // The target worker is gone, so the message never left the supervisor.
        "worker_unavailable"
    } else if error.contains("is not connected") || error.contains("predates the roster protocol") {
        "worker_unavailable"
    } else {
        "worker_error"
    }
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn create_private_dir(path: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn set_private_mode(path: &str) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_journal() -> (AgentMessageDeliveryJournal, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("agent-message-delivery-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(AGENT_MESSAGE_DELIVERY_JOURNAL_FILE);
        (AgentMessageDeliveryJournal::new(&path.to_string_lossy()), path)
    }

    #[test]
    fn records_survive_a_restart_and_carry_no_message_text() {
        let (mut journal, path) = temp_journal();
        journal.record(AgentMessageDeliveryRecord::new(
            Some("source-active"),
            "target-active",
            Some("agentmsg_00000000-0000-0000-0000-000000000001"),
            AgentMessageDeliveryOutcome::Delivered,
            None,
        ));
        journal.record(AgentMessageDeliveryRecord::new(
            Some("source-active"),
            "missing-active",
            None,
            AgentMessageDeliveryOutcome::Uncertain,
            Some(delivery_reason_code("Unknown active session: missing-active")),
        ));

        let durable = AgentMessageDeliveryJournal::read(&path.to_string_lossy());
        assert_eq!(durable.len(), 2);
        assert_eq!(durable[0].outcome, AgentMessageDeliveryOutcome::Delivered);
        assert_eq!(
            durable[0].message_id.as_deref(),
            Some("agentmsg_00000000-0000-0000-0000-000000000001")
        );
        assert_eq!(durable[1].outcome, AgentMessageDeliveryOutcome::Uncertain);
        assert_eq!(durable[1].reason_code.as_deref(), Some("unknown_active_session"));
        assert!(durable[1].message_id.is_none());

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("hello peer"), "no message text may reach the journal");
        assert!(!raw.contains("Unknown active session"), "no raw error text may reach the journal");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn only_agent_message_ids_and_bounded_identifiers_are_accepted() {
        let record = AgentMessageDeliveryRecord::new(
            Some(" \u{7}source\n "),
            "target",
            Some("not-a-message-id"),
            AgentMessageDeliveryOutcome::Queued,
            None,
        );
        assert_eq!(record.source_active_session_id.as_deref(), Some("source"));
        assert!(record.message_id.is_none(), "an unexpected id shape is dropped");
        let long = "a".repeat(500);
        let record = AgentMessageDeliveryRecord::new(
            Some(&long),
            &long,
            Some(&format!("agentmsg_{long}")),
            AgentMessageDeliveryOutcome::Delivered,
            None,
        );
        assert_eq!(record.source_active_session_id.as_deref().unwrap().len(), 128);
        assert_eq!(record.target_active_session_id.len(), 128);
        assert!(record.message_id.is_none());
    }

    #[test]
    fn the_journal_stays_bounded_across_many_appends() {
        let (mut journal, path) = temp_journal();
        for index in 0..(AGENT_MESSAGE_DELIVERY_JOURNAL_LIMIT + 50) {
            journal.record(AgentMessageDeliveryRecord::new(
                Some("source"),
                &format!("target-{index}"),
                None,
                AgentMessageDeliveryOutcome::Delivered,
                None,
            ));
        }
        assert_eq!(journal.records().len(), AGENT_MESSAGE_DELIVERY_JOURNAL_LIMIT);
        let durable = AgentMessageDeliveryJournal::read(&path.to_string_lossy());
        assert_eq!(durable.len(), AGENT_MESSAGE_DELIVERY_JOURNAL_LIMIT);
        // The newest record is retained, the oldest is dropped.
        assert_eq!(
            durable.last().unwrap().target_active_session_id,
            format!("target-{}", AGENT_MESSAGE_DELIVERY_JOURNAL_LIMIT + 49)
        );
        let reread = AgentMessageDeliveryJournal::new(&path.to_string_lossy());
        assert_eq!(reread.records().len(), AGENT_MESSAGE_DELIVERY_JOURNAL_LIMIT);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn corrupt_lines_are_ignored_and_receipt_statuses_map_safely() {
        let dir = std::env::temp_dir().join(format!("agent-message-delivery-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(AGENT_MESSAGE_DELIVERY_JOURNAL_FILE);
        std::fs::write(
            &path,
            concat!(
                "not json\n",
                "{\"version\":2,\"targetActiveSessionId\":\"x\",\"outcome\":\"delivered\"}\n",
                "{\"version\":1,\"targetActiveSessionId\":\"\",\"outcome\":\"delivered\"}\n",
                "{\"version\":1,\"targetActiveSessionId\":\"y\",\"outcome\":\"invented\"}\n",
                "{\"version\":1,\"targetActiveSessionId\":\"y\",\"outcome\":\"queued\",\"messageId\":\"forged\"}\n",
            ),
        )
        .unwrap();
        let records = AgentMessageDeliveryJournal::read(&path.to_string_lossy());
        assert_eq!(records.len(), 1, "only the valid record survives");
        assert_eq!(records[0].target_active_session_id, "y");
        assert_eq!(records[0].outcome, AgentMessageDeliveryOutcome::Queued);
        assert!(records[0].message_id.is_none());

        assert_eq!(
            AgentMessageDeliveryOutcome::from_receipt_status("delivered"),
            Some(AgentMessageDeliveryOutcome::Delivered)
        );
        assert_eq!(
            AgentMessageDeliveryOutcome::from_receipt_status("queued"),
            Some(AgentMessageDeliveryOutcome::Queued)
        );
        assert_eq!(AgentMessageDeliveryOutcome::from_receipt_status("rejected"), None);
        std::fs::remove_dir_all(dir).ok();
    }
}
