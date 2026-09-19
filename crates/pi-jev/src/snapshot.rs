//! Immutable, bounded state snapshots (DESIGN.md section 4).
//!
//! A snapshot is the single unit a bundle of Jev questions is asked against.
//! Snapshots are shared immutably, payload-capped, and fingerprinted with a
//! sha256 of their canonical JSON so records never need raw state copies.

use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Maximum serialized state bytes sent to SystemOne or kept in a snapshot.
pub const MAX_STATE_BYTES: usize = 8192;

/// Version of the bounded state schema.
pub const STATE_SCHEMA_VERSION: &str = "jev.state/1";

/// Maximum string length kept inside bounded state.
pub const MAX_TEXT_CHARS: usize = 400;

/// Maximum array/object width kept inside state.
pub const MAX_ITEMS: usize = 32;

/// Maximum JSON nesting depth kept inside state.
pub const MAX_DEPTH: usize = 6;

/// Lifecycle stage a snapshot was captured at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotStage {
    TurnStart,
    ToolCall,
    AgentEnd,
    ModelSelect,
}

impl SnapshotStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            SnapshotStage::TurnStart => "turn_start",
            SnapshotStage::ToolCall => "tool_call",
            SnapshotStage::AgentEnd => "agent_end",
            SnapshotStage::ModelSelect => "model_select",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("state payload exceeds {MAX_STATE_BYTES} bytes after bounding")]
    StateTooLarge,
    #[error("no questions eligible for this snapshot")]
    EmptyQuestions,
}

/// Immutable bounded snapshot. Shared via `Arc` once built.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub stage: SnapshotStage,
    /// Opaque local session identifier; never a human-readable prompt.
    pub session_id: String,
    pub turn: u64,
    /// Monotonic per-session sequence for ordering.
    pub seq: u32,
    /// sha256 hex of the canonical JSON of `state`.
    pub fingerprint: String,
    /// RFC 3339 UTC creation time.
    pub created_at: String,
    pub model: Option<String>,
    /// Bounded, prompt-free semantic state (excerpt-capped).
    pub state: Value,
    /// Question ids this snapshot's bundle carries.
    pub question_ids: Vec<String>,
}

impl StateSnapshot {
    /// Build a snapshot; `state` must already be bounded by the caller
    /// (`bound_json`), and is bounded again defensively here.
    pub fn new(
        stage: SnapshotStage,
        session_id: impl Into<String>,
        turn: u64,
        seq: u32,
        model: Option<String>,
        state: Value,
        question_ids: Vec<String>,
    ) -> Result<Self, SnapshotError> {
        let state = bound_json(state, 0);
        let bytes = serde_json::to_vec(&state).unwrap_or_default();
        if bytes.len() > MAX_STATE_BYTES {
            return Err(SnapshotError::StateTooLarge);
        }
        Ok(Self {
            stage,
            session_id: session_id.into(),
            turn,
            seq,
            fingerprint: fingerprint_of(&state),
            created_at: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
            model,
            state,
            question_ids,
        })
    }
}

/// sha256 hex of the canonical (key-sorted) JSON of `value`.
pub fn fingerprint_of(value: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(canonical_json(value));
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Canonical JSON: objects with recursively sorted keys (BTreeMap-backed).
pub fn canonical_json(value: &Value) -> Vec<u8> {
    let canonical = canonicalize(value);
    serde_json::to_vec(&canonical).unwrap_or_default()
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = serde_json::Map::new();
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                sorted.insert(key.clone(), canonicalize(&map[key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

/// Deep-bound a JSON value for snapshot/state use: cap strings, arrays,
/// object width and depth; replace oversized pieces with bounded markers.
/// This keeps raw prompt/tool-output dumps out of snapshots and records.
pub fn bound_json(value: Value, depth: usize) -> Value {
    if depth > MAX_DEPTH {
        return serde_json::json!({ "_truncated": "depth" });
    }
    match value {
        Value::String(text) => {
            let (bounded, truncated) = truncate_text(&text, MAX_TEXT_CHARS);
            if truncated {
                Value::String(format!("{bounded}...[truncated]"))
            } else {
                Value::String(bounded)
            }
        }
        Value::Array(items) => {
            let original_len = items.len();
            let bounded: Vec<Value> = items
                .into_iter()
                .take(MAX_ITEMS)
                .map(|item| bound_json(item, depth + 1))
                .collect();
            if original_len > bounded.len() {
                let mut with_marker = bounded;
                with_marker.push(serde_json::json!({ "_truncated": "items" }));
                return Value::Array(with_marker);
            }
            Value::Array(bounded)
        }
        Value::Object(map) => {
            let mut bounded = serde_json::Map::new();
            let mut count = 0usize;
            for (key, item) in map.into_iter() {
                count += 1;
                if count > MAX_ITEMS {
                    bounded.insert("_truncated".to_string(), serde_json::json!("items"));
                    break;
                }
                bounded.insert(truncate_key(&key), bound_json(item, depth + 1));
            }
            Value::Object(bounded)
        }
        Value::Number(number) => {
            if number.as_f64().map(|v| v.is_finite()).unwrap_or(true) {
                Value::Number(number)
            } else {
                // NaN / infinity must never enter snapshots.
                serde_json::json!("nonfinite")
            }
        }
        other => other,
    }
}

fn truncate_key(key: &str) -> String {
    let (bounded, truncated) = truncate_text(key, 64);
    if truncated {
        format!("{bounded}...")
    } else {
        bounded
    }
}

/// Truncate to `max` characters on char boundaries; reports whether truncated.
pub fn truncate_text(text: &str, max: usize) -> (String, bool) {
    if text.chars().count() <= max {
        return (text.to_string(), false);
    }
    let bounded: String = text.chars().take(max).collect();
    (bounded, true)
}
