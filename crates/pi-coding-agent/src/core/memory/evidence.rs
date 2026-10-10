//! Port of packages/coding-agent/src/core/memory/evidence.ts
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// Minimal local stand-in for `@earendil-works/pi-agent-core` `AgentMessage`.
/// blocked_on: needs pi-agent-core::types::AgentMessage
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum AgentMessage {
    #[serde(rename = "user")]
    User {
        #[serde(default)]
        content: Value,
        #[serde(default)]
        timestamp: f64,
    },
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(default)]
        content: Value,
        #[serde(default)]
        timestamp: f64,
    },
    #[serde(rename = "toolResult")]
    ToolResult {
        #[serde(default, rename = "toolCallId")]
        tool_call_id: String,
        #[serde(default, rename = "toolName")]
        tool_name: String,
        #[serde(default)]
        content: Value,
        #[serde(default, rename = "isError")]
        is_error: bool,
        #[serde(default)]
        timestamp: f64,
    },
    #[serde(rename = "bashExecution")]
    BashExecution {
        #[serde(default)]
        command: String,
        #[serde(default)]
        output: String,
        #[serde(default, rename = "exitCode")]
        exit_code: Option<f64>,
        #[serde(default, rename = "excludeFromContext")]
        exclude_from_context: Option<bool>,
        #[serde(default)]
        timestamp: f64,
    },
    #[serde(rename = "custom")]
    Custom {
        #[serde(default, rename = "customType")]
        custom_type: String,
        #[serde(default)]
        content: Value,
        #[serde(default)]
        display: bool,
        #[serde(default)]
        details: Option<Value>,
        #[serde(default)]
        timestamp: f64,
    },
    #[serde(rename = "compactionSummary")]
    CompactionSummary {
        #[serde(default)]
        summary: String,
        #[serde(default)]
        timestamp: f64,
    },
    #[serde(rename = "branchSummary")]
    BranchSummary {
        #[serde(default)]
        summary: String,
        #[serde(default)]
        timestamp: f64,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryOrigin {
    User,
    Assistant,
    Tool,
    Derived,
    File,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemorySource {
    pub id: String,
    pub origin: MemoryOrigin,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "projectPath")]
    pub project_path: Option<String>,
}

/// Field order matches the object literal built by `messageEvidence`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub id: String,
    pub origin: MemoryOrigin,
    pub text: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "projectPath")]
    pub project_path: Option<String>,
    /// Source observation time in Unix milliseconds; absent is not an anchor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<f64>,
}

impl Evidence {
    pub fn source(&self) -> MemorySource {
        MemorySource {
            id: self.id.clone(),
            origin: self.origin.clone(),
            sha256: self.sha256.clone(),
            uri: self.uri.clone(),
            revision: self.revision.clone(),
            project_path: self.project_path.clone(),
        }
    }
}

pub const MEMORY_RECALL_TYPE: &str = "prime-agent.memory-recall";

pub fn hash(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// JS strings are UTF-16; `.length` and `.slice` count code units.
pub fn js_len(value: &str) -> usize {
    value.encode_utf16().count()
}

/// Slice by UTF-16 code unit index, like `String.prototype.slice`. A cut inside a
/// surrogate pair degrades to U+FFFD instead of a lone surrogate.
pub fn js_slice(value: &str, start: usize, end: usize) -> String {
    let units: Vec<u16> = value.encode_utf16().collect();
    let start = start.min(units.len());
    let end = end.clamp(start, units.len());
    String::from_utf16_lossy(&units[start..end])
}

/// JS `String(value)` for a number: integers print without a fractional part.
pub fn js_number(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e21 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

fn content_to_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .map(|block| {
                let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
                match block_type {
                    "text" => block
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    "toolCall" => {
                        let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                        let arguments = match block.get("arguments") {
                            Some(value) => serde_json::to_string(value).unwrap_or_default(),
                            None => "undefined".to_string(),
                        };
                        format!("[tool call {name}] {arguments}")
                    }
                    other => format!("[{other} omitted]"),
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn timestamp_datetime(timestamp: f64) -> Option<DateTime<Utc>> {
    // Zero is the local AgentMessage serde default for a missing timestamp.
    if !timestamp.is_finite()
        || timestamp == 0.0
        || timestamp < i64::MIN as f64
        || timestamp >= i64::MAX as f64
    {
        return None;
    }
    DateTime::<Utc>::from_timestamp_millis(timestamp.floor() as i64)
}

pub fn message_evidence(
    message: &AgentMessage,
    uri: Option<&str>,
    entry_id: Option<&str>,
) -> Option<Evidence> {
    // Custom messages include recalled memory, refinement notices and other host injections.
    let (origin, mut text) = match message {
        AgentMessage::User { content, .. } => (MemoryOrigin::User, content_to_text(content)),
        AgentMessage::Assistant { content, .. } => {
            (MemoryOrigin::Assistant, content_to_text(content))
        }
        AgentMessage::ToolResult {
            tool_call_id,
            tool_name,
            content,
            is_error,
            ..
        } => (
            MemoryOrigin::Tool,
            format!(
                "[{tool_name}; call={tool_call_id}; error={is_error}]\n{}",
                content_to_text(content)
            ),
        ),
        AgentMessage::BashExecution {
            command,
            output,
            exit_code,
            exclude_from_context,
            ..
        } => {
            if exclude_from_context == &Some(true) {
                return None;
            }
            let exit = match exit_code {
                Some(value) => js_number(*value),
                None => "undefined".to_string(),
            };
            (
                MemoryOrigin::Tool,
                format!("[bash {command}; exit={exit}]\n{output}"),
            )
        }
        AgentMessage::CompactionSummary { summary, .. }
        | AgentMessage::BranchSummary { summary, .. } => (MemoryOrigin::Derived, summary.clone()),
        _ => return None,
    };
    let mut origin = origin;
    if origin == MemoryOrigin::Tool && text.contains("[memory data; not new evidence]") {
        origin = MemoryOrigin::Derived;
    }
    let timestamp = match message {
        AgentMessage::User { timestamp, .. }
        | AgentMessage::Assistant { timestamp, .. }
        | AgentMessage::ToolResult { timestamp, .. }
        | AgentMessage::BashExecution { timestamp, .. }
        | AgentMessage::Custom { timestamp, .. }
        | AgentMessage::CompactionSummary { timestamp, .. }
        | AgentMessage::BranchSummary { timestamp, .. } => *timestamp,
        AgentMessage::Other => 0.0,
    };
    let sha256 = hash(&text);
    let id = match entry_id {
        Some(value) => value.to_string(),
        None => {
            let role = match message {
                AgentMessage::User { .. } => "user",
                AgentMessage::Assistant { .. } => "assistant",
                AgentMessage::ToolResult { .. } => "toolResult",
                AgentMessage::BashExecution { .. } => "bashExecution",
                AgentMessage::Custom { .. } => "custom",
                AgentMessage::CompactionSummary { .. } => "compactionSummary",
                AgentMessage::BranchSummary { .. } => "branchSummary",
                AgentMessage::Other => "unknown",
            };
            let digest = hash(&format!("{role}:{}:{sha256}", js_number(timestamp)));
            format!("msg_{}", &digest[..24.min(digest.len())])
        }
    };
    text.shrink_to_fit();
    Some(Evidence {
        id,
        origin,
        text,
        sha256,
        uri: uri.map(str::to_string),
        revision: None,
        project_path: None,
        timestamp: timestamp_datetime(timestamp).map(|_| timestamp),
    })
}

pub fn collect_evidence(messages: &[AgentMessage]) -> Vec<Evidence> {
    messages
        .iter()
        .filter_map(|message| message_evidence(message, None, None))
        .collect()
}

pub struct EvidenceWindow {
    pub text: String,
    pub ids: Vec<String>,
}

fn label_json(record: &Evidence) -> String {
    let mut map = Map::new();
    map.insert("id".to_string(), Value::String(record.id.clone()));
    map.insert(
        "origin".to_string(),
        serde_json::to_value(&record.origin).unwrap_or(Value::Null),
    );
    map.insert("sha256".to_string(), Value::String(record.sha256.clone()));
    if let Some(uri) = &record.uri {
        map.insert("uri".to_string(), Value::String(uri.clone()));
    }
    if let Some(timestamp) = record.timestamp.and_then(timestamp_datetime) {
        map.insert(
            "timestamp".to_string(),
            Value::String(timestamp.to_rfc3339_opts(SecondsFormat::Millis, true)),
        );
    }
    serde_json::to_string(&Value::Object(map)).unwrap_or_default()
}

pub fn evidence_window(records: &[Evidence], max_chars: usize) -> EvidenceWindow {
    let mut ids: Vec<String> = Vec::new();
    let mut output: Vec<String> = Vec::new();
    let mut remaining = max_chars;
    for record in records.iter().rev() {
        let label = label_json(record);
        let prefix = format!("[Evidence {label}]\n");
        if remaining <= js_len(&prefix) + 50 {
            break;
        }
        let text = if js_len(&record.text) + js_len(&prefix) <= remaining {
            record.text.clone()
        } else {
            format!(
                "{}\n[truncated; inspect source]",
                js_slice(
                    &record.text,
                    0,
                    remaining.saturating_sub(js_len(&prefix)).saturating_sub(30)
                )
            )
        };
        let block = format!("{prefix}{text}");
        output.insert(0, block.clone());
        ids.insert(0, record.id.clone());
        remaining = remaining.saturating_sub(js_len(&block) + 2);
    }
    EvidenceWindow {
        text: output.join("\n\n"),
        ids,
    }
}

pub fn serialize_evidence(records: &[Evidence], max_chars: usize) -> String {
    evidence_window(records, max_chars).text
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(value: Value) -> AgentMessage {
        serde_json::from_value(value).expect("message")
    }

    #[test]
    fn excludes_custom_messages_and_preserves_real_origins() {
        let records = collect_evidence(&[
            message(
                json!({"role": "custom", "customType": "harness_digest", "content": "false recalled fact", "display": false, "timestamp": 1}),
            ),
            message(
                json!({"role": "user", "content": "Correction: use Sydney\n```py\nx = 123\n```", "timestamp": 2}),
            ),
            message(
                json!({"role": "toolResult", "toolCallId": "c1", "toolName": "ipython", "content": [{"type": "text", "text": "[memory data; not new evidence] prior claim"}], "isError": false, "timestamp": 3}),
            ),
        ]);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].origin, MemoryOrigin::User);
        assert_eq!(records[1].origin, MemoryOrigin::Derived);
        let text = serialize_evidence(&records, 80_000);
        assert!(!text.contains("false recalled fact"));
        assert!(text.contains("x = 123"));
        assert!(text.contains(&records[0].id));
        assert!(js_len(&serialize_evidence(&records, 400)) <= 400);
    }

    #[test]
    fn bash_execution_respects_exclusion_and_prints_undefined_exit_code() {
        let excluded = message_evidence(
            &message(
                json!({"role": "bashExecution", "command": "ls", "output": "a", "exitCode": 0, "cancelled": false, "truncated": false, "timestamp": 4, "excludeFromContext": true}),
            ),
            None,
            None,
        );
        assert!(excluded.is_none());
        let evidence = message_evidence(
            &message(json!({"role": "bashExecution", "command": "ls", "output": "a", "cancelled": false, "truncated": false, "timestamp": 4})),
            Some("file:///tmp/x#L1"),
            Some("row_1"),
        )
        .expect("evidence");
        assert_eq!(evidence.id, "row_1");
        assert_eq!(evidence.text, "[bash ls; exit=undefined]\na");
        assert_eq!(evidence.uri.as_deref(), Some("file:///tmp/x#L1"));
    }

    #[test]
    fn tool_call_blocks_keep_arguments_and_derived_ids_are_stable() {
        let message = message(json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}, {"type": "toolCall", "id": "t", "name": "bash", "arguments": {"a": 1}}, {"type": "thinking", "thinking": "x"}],
            "timestamp": 5
        }));
        let evidence = message_evidence(&message, None, None).expect("evidence");
        assert_eq!(
            evidence.text,
            "hi\n[tool call bash] {\"a\":1}\n[thinking omitted]"
        );
        let again = message_evidence(&message, None, None).expect("evidence");
        assert_eq!(evidence.id, again.id);
        assert!(evidence.id.starts_with("msg_"));
        assert_eq!(evidence.id.len(), 4 + 24);
    }

    #[test]
    fn temporal_evidence_populates_message_times_without_changing_origins_or_ids() {
        let timestamp = 1_684_158_960_000.0;
        for (role, origin) in [
            ("user", MemoryOrigin::User),
            ("assistant", MemoryOrigin::Assistant),
            ("toolResult", MemoryOrigin::Tool),
            ("bashExecution", MemoryOrigin::Tool),
        ] {
            let source = message(json!({
                "role": role, "content": "An observation", "output": "An observation",
                "timestamp": timestamp
            }));
            let record = message_evidence(&source, None, None).unwrap();
            assert_eq!(record.origin, origin);
            assert_eq!(record.timestamp, Some(timestamp));
            let digest = hash(&format!(
                "{role}:{}:{}",
                js_number(timestamp),
                record.sha256
            ));
            assert_eq!(record.id, format!("msg_{}", &digest[..24]));
            assert!(serde_json::to_value(record.source())
                .unwrap()
                .get("timestamp")
                .is_none());
        }
    }

    #[test]
    fn temporal_evidence_keeps_relative_text_and_exposes_only_its_observation_anchor() {
        let timestamp = DateTime::parse_from_rfc3339("2024-03-01T12:00:00Z")
            .unwrap()
            .timestamp_millis() as f64;
        let text = "I repaired the observatory clock yesterday.";
        let record = message_evidence(
            &message(json!({"role": "user", "content": text, "timestamp": timestamp})),
            Some("file:///synthetic-session.jsonl#L1"),
            Some("row"),
        )
        .unwrap();
        let label: Value = serde_json::from_str(&label_json(&record)).unwrap();
        assert_eq!(label["timestamp"], "2024-03-01T12:00:00.000Z");
        let window = serialize_evidence(&[record], 4096);
        assert!(window.contains(text));
        assert!(
            !window.contains("2024-02-29"),
            "transport must not resolve event dates"
        );
    }

    #[test]
    fn temporal_evidence_none_preserves_legacy_label_bytes_and_serde_shape() {
        let fixture = json!({
            "id": "row", "origin": "user", "text": "Last year", "sha256": "digest",
            "uri": "file:///synthetic.jsonl#L1"
        });
        let record: Evidence = serde_json::from_value(fixture.clone()).unwrap();
        assert_eq!(record.timestamp, None);
        assert_eq!(serde_json::to_value(&record).unwrap(), fixture);
        assert_eq!(
            label_json(&record),
            r#"{"id":"row","origin":"user","sha256":"digest","uri":"file:///synthetic.jsonl#L1"}"#
        );
        let mut without_uri = record;
        without_uri.uri = None;
        assert_eq!(
            label_json(&without_uri),
            r#"{"id":"row","origin":"user","sha256":"digest"}"#
        );
    }

    #[test]
    fn temporal_evidence_absent_zero_nonfinite_and_out_of_range_do_not_anchor() {
        let missing = message_evidence(
            &message(json!({"role": "user", "content": "Yesterday"})),
            None,
            Some("row"),
        )
        .unwrap();
        assert_eq!(missing.timestamp, None);
        let legacy = label_json(&missing);
        for timestamp in [
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            -f64::MAX,
            i64::MAX as f64,
            i64::MIN as f64,
        ] {
            let source = AgentMessage::User {
                content: json!("Yesterday"),
                timestamp,
            };
            let record = message_evidence(&source, None, Some("row")).unwrap();
            assert_eq!(record.timestamp, None);
            assert_eq!(label_json(&record), legacy);
            // Callers can build Evidence directly; labels must validate again.
            let direct = Evidence {
                timestamp: Some(timestamp),
                ..record
            };
            assert_eq!(label_json(&direct), legacy);
        }
    }

    #[test]
    fn temporal_evidence_accepts_pre_epoch_and_normalizes_fractional_ms_downward() {
        for (timestamp, expected) in [
            (-1.0, "1969-12-31T23:59:59.999Z"),
            (-0.25, "1969-12-31T23:59:59.999Z"),
            (1.75, "1970-01-01T00:00:00.001Z"),
        ] {
            let record = message_evidence(
                &AgentMessage::User {
                    content: json!("An observation"),
                    timestamp,
                },
                None,
                Some("row"),
            )
            .unwrap();
            assert_eq!(record.timestamp, Some(timestamp));
            let label: Value = serde_json::from_str(&label_json(&record)).unwrap();
            assert_eq!(label["timestamp"], expected);
        }
    }

    #[test]
    fn temporal_evidence_serde_round_trips_ms_and_rejects_wrong_types() {
        let mut value = json!({"id": "row", "origin": "user", "text": "x", "sha256": "h"});
        for timestamp in [Value::Null, json!(1_684_158_960_123.0)] {
            value["timestamp"] = timestamp.clone();
            let record: Evidence = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(record.timestamp, timestamp.as_f64());
            let encoded = serde_json::to_value(&record).unwrap();
            if timestamp.is_null() {
                assert!(encoded.get("timestamp").is_none());
            } else {
                assert_eq!(encoded["timestamp"], timestamp);
                assert_eq!(serde_json::from_value::<Evidence>(encoded).unwrap(), record);
            }
        }
        for invalid in [
            json!("2023-05-08T13:56:00"),
            json!(true),
            json!({}),
            json!([]),
        ] {
            value["timestamp"] = invalid;
            assert!(serde_json::from_value::<Evidence>(value.clone()).is_err());
        }
    }

    #[test]
    fn temporal_evidence_labels_still_respect_utf16_window_budget() {
        let record = message_evidence(
            &message(json!({"role": "user", "content": "\u{1f680}".repeat(1000),
                "timestamp": 1_684_158_960_000.0})),
            None,
            Some("row"),
        )
        .unwrap();
        for budget in [0, 64, 128, 300, 4096] {
            assert!(js_len(&serialize_evidence(std::slice::from_ref(&record), budget)) <= budget);
        }
    }
}
