//! Port of packages/coding-agent/src/modes/daemon/rlm-ledger.ts
//!
//! Daemon-owned RLM spawn ledger: one append-only JSONL file per sessions dir,
//! written at spawn admission, rename, and deletion, and replayed for topology.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::event_log::{EventLog, EventLogOptions, ReplayOptions};
use crate::core::session_lease::canonical_session_path;
use crate::core::session_manager::{
    get_session_artifact_path_for_file, read_session_info, SessionInfo,
};
use crate::utils::file_lines::read_first_line_sync;

pub const RLM_LEDGER_DIR: &str = "rlm-ledger";

/// Bounded read: a ledger beyond these limits fails closed loudly.
pub const RLM_LEDGER_MAX_BYTES: u64 = 32 * 1024 * 1024;
pub const RLM_LEDGER_MAX_RECORDS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RlmLedgerDeleteReason {
    User,
    ParentTeardown,
    Revoked,
    Gc,
}

impl RlmLedgerDeleteReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            RlmLedgerDeleteReason::User => "user",
            RlmLedgerDeleteReason::ParentTeardown => "parent-teardown",
            RlmLedgerDeleteReason::Revoked => "revoked",
            RlmLedgerDeleteReason::Gc => "gc",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "user" => Some(RlmLedgerDeleteReason::User),
            "parent-teardown" => Some(RlmLedgerDeleteReason::ParentTeardown),
            "revoked" => Some(RlmLedgerDeleteReason::Revoked),
            "gc" => Some(RlmLedgerDeleteReason::Gc),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlmLedgerMetaRecord {
    pub v: u32,
    pub op: String,
    pub at: String,
    #[serde(rename = "sessionsDir")]
    pub sessions_dir: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlmLedgerSpawnRecord {
    pub v: u32,
    pub op: String,
    pub at: String,
    #[serde(rename = "childId")]
    pub child_id: String,
    pub parent: String,
    pub child: String,
    pub depth: i64,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlmLedgerRenameRecord {
    pub v: u32,
    pub op: String,
    pub at: String,
    #[serde(rename = "childId")]
    pub child_id: String,
    pub child: String,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlmLedgerDeleteRecord {
    pub v: u32,
    pub op: String,
    pub at: String,
    #[serde(rename = "childId")]
    pub child_id: String,
    pub child: String,
    pub reason: RlmLedgerDeleteReason,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RlmLedgerRecord {
    Meta(RlmLedgerMetaRecord),
    Spawn(RlmLedgerSpawnRecord),
    Rename(RlmLedgerRenameRecord),
    Delete(RlmLedgerDeleteRecord),
}

impl RlmLedgerRecord {
    pub fn op(&self) -> &str {
        match self {
            RlmLedgerRecord::Meta(record) => &record.op,
            RlmLedgerRecord::Spawn(record) => &record.op,
            RlmLedgerRecord::Rename(record) => &record.op,
            RlmLedgerRecord::Delete(record) => &record.op,
        }
    }

    pub fn to_value(&self) -> Value {
        match self {
            RlmLedgerRecord::Meta(record) => serde_json::to_value(record).unwrap_or(Value::Null),
            RlmLedgerRecord::Spawn(record) => serde_json::to_value(record).unwrap_or(Value::Null),
            RlmLedgerRecord::Rename(record) => serde_json::to_value(record).unwrap_or(Value::Null),
            RlmLedgerRecord::Delete(record) => serde_json::to_value(record).unwrap_or(Value::Null),
        }
    }
}

/// A live edge after replaying the ledger (last-writer-wins per childId+child).
#[derive(Debug, Clone, PartialEq)]
pub struct RlmLedgerEdge {
    pub child_id: String,
    pub parent: String,
    pub child: String,
    pub depth: i64,
    pub name: String,
    pub deleted: Option<RlmLedgerDeleteReason>,
}

/// Minimal registry-entry shape the seeder consumes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlmLedgerSeedRegistryEntry {
    #[serde(rename = "childId")]
    pub child_id: String,
    #[serde(rename = "sessionName")]
    pub session_name: String,
    #[serde(rename = "sessionFile")]
    pub session_file: String,
    #[serde(rename = "rlmDepth", skip_serializing_if = "Option::is_none", default)]
    pub rlm_depth: Option<i64>,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyRlmSubagentRegistryEntry {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(rename = "childId")]
    pub child_id: String,
    #[serde(rename = "sessionName")]
    pub session_name: String,
    #[serde(rename = "sessionDir")]
    pub session_dir: String,
    #[serde(rename = "sessionFile")]
    pub session_file: String,
    #[serde(rename = "parentSessionId")]
    pub parent_session_id: String,
    #[serde(
        rename = "parentSessionFile",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub parent_session_file: Option<String>,
    #[serde(rename = "rlmDepth", skip_serializing_if = "Option::is_none", default)]
    pub rlm_depth: Option<i64>,
    #[serde(
        rename = "rlmMaxDepth",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub rlm_max_depth: Option<i64>,
    #[serde(
        rename = "rlmParentNodeId",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub rlm_parent_node_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub prompt: Option<String>,
    #[serde(rename = "spawnCode", skip_serializing_if = "Option::is_none", default)]
    pub spawn_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model: Option<Value>,
    #[serde(rename = "createdAt")]
    pub created_at: f64,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    pub status: String,
}

/// `readLegacyRlmSubagentRegistry` options.
#[derive(Clone, Default)]
pub struct ReadLegacyRlmSubagentRegistryOptions {
    pub throw_on_read_error: bool,
    pub log: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    pub on_read_error: Option<Arc<dyn Fn() + Send + Sync>>,
}

#[async_trait::async_trait]
pub trait RlmLedgerSeedSource: Send + Sync {
    async fn read_registry_for_session_file(
        &self,
        session_file: &str,
    ) -> Vec<RlmLedgerSeedRegistryEntry>;
}

/// Latest entry per childId from the legacy per-parent registry; tolerant reads.
pub async fn read_legacy_rlm_subagent_registry(
    path: &str,
    options: ReadLegacyRlmSubagentRegistryOptions,
) -> Result<Vec<LegacyRlmSubagentRegistryEntry>, std::io::Error> {
    let contents = match tokio::fs::read_to_string(path).await {
        Ok(contents) => contents,
        Err(error) => {
            if let Some(on_read_error) = &options.on_read_error {
                on_read_error();
            }
            if error.kind() != std::io::ErrorKind::NotFound {
                if let Some(log) = &options.log {
                    log(&format!("failed to read RLM subagent registry: {error}"));
                }
                if options.throw_on_read_error {
                    return Err(error);
                }
            }
            return Ok(Vec::new());
        }
    };
    let mut latest: indexmap::IndexMap<String, LegacyRlmSubagentRegistryEntry> =
        indexmap::IndexMap::new();
    for line in contents.split('\n') {
        let trimmed = line.trim_end_matches('\r').trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(parsed) = serde_json::from_str::<Value>(trimmed) else {
            if let Some(log) = &options.log {
                log("ignored malformed RLM subagent registry entry: invalid JSON");
            }
            continue;
        };
        let Some(entry) = parsed.as_object() else {
            if let Some(log) = &options.log {
                log("ignored malformed RLM subagent registry entry: not an object");
            }
            continue;
        };
        if !is_legacy_registry_entry(entry) {
            continue;
        }
        let mut parsed_entry: LegacyRlmSubagentRegistryEntry =
            serde_json::from_value(Value::Object(entry.clone())).unwrap_or_else(|_| {
                LegacyRlmSubagentRegistryEntry {
                    type_: "rlm_subagent".to_string(),
                    child_id: entry
                        .get("childId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    session_name: entry
                        .get("sessionName")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    session_dir: String::new(),
                    session_file: entry
                        .get("sessionFile")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    parent_session_id: entry
                        .get("parentSessionId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    parent_session_file: None,
                    rlm_depth: None,
                    rlm_max_depth: None,
                    rlm_parent_node_id: None,
                    prompt: None,
                    spawn_code: None,
                    model: None,
                    created_at: 0.0,
                    updated_at: String::new(),
                    status: entry
                        .get("status")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                }
            });
        if parsed_entry.session_dir.is_empty() {
            parsed_entry.session_dir = dirname_of(&parsed_entry.session_file);
        }
        // A damaged rlmMaxDepth is dropped instead of rejecting the whole entry.
        parsed_entry.rlm_max_depth = match parsed_entry.rlm_max_depth {
            Some(depth) if depth >= 0 => Some(depth),
            _ => None,
        };
        latest.insert(parsed_entry.child_id.clone(), parsed_entry);
    }
    Ok(latest.into_values().collect())
}

fn is_legacy_registry_entry(entry: &serde_json::Map<String, Value>) -> bool {
    if entry.get("type").and_then(Value::as_str) != Some("rlm_subagent") {
        return false;
    }
    if entry.get("childId").and_then(Value::as_str).is_none() {
        return false;
    }
    if entry.get("sessionName").and_then(Value::as_str).is_none() {
        return false;
    }
    if entry.get("sessionFile").and_then(Value::as_str).is_none() {
        return false;
    }
    match entry.get("status").and_then(Value::as_str) {
        Some("running") | Some("completed") | Some("deleted") => {}
        _ => return false,
    }
    if let Some(depth) = entry.get("rlmDepth") {
        match depth.as_i64() {
            Some(depth) if depth >= 0 => {}
            _ => return false,
        }
    }
    true
}

pub struct RegistrySeedSource;

#[async_trait::async_trait]
impl RlmLedgerSeedSource for RegistrySeedSource {
    async fn read_registry_for_session_file(
        &self,
        session_file: &str,
    ) -> Vec<RlmLedgerSeedRegistryEntry> {
        let first_line = read_first_line_sync(session_file, 1024 * 1024);
        let Some(first_line) = first_line else {
            return Vec::new();
        };
        let Ok(header) = serde_json::from_str::<Value>(&first_line) else {
            return Vec::new();
        };
        let Some(header_id) = header.get("id").and_then(Value::as_str) else {
            return Vec::new();
        };
        let path = Path::new(&get_session_artifact_path_for_file(
            session_file,
            Some(header_id),
        ))
        .join("rlm-subagents.jsonl")
        .to_string_lossy()
        .to_string();
        let entries = read_legacy_rlm_subagent_registry(
            &path,
            ReadLegacyRlmSubagentRegistryOptions::default(),
        )
        .await
        .unwrap_or_default();
        entries
            .into_iter()
            .map(|entry| RlmLedgerSeedRegistryEntry {
                child_id: entry.child_id,
                session_name: entry.session_name,
                session_file: entry.session_file,
                rlm_depth: entry.rlm_depth,
                status: entry.status,
            })
            .collect()
    }
}

pub fn create_rlm_ledger_registry_seed_source() -> Arc<dyn RlmLedgerSeedSource> {
    Arc::new(RegistrySeedSource)
}

/// Canonicalize a directory: realpath when it exists, plain resolve otherwise.
fn canonicalize_dir_path(dir: &str) -> String {
    let resolved = resolve_path(dir);
    std::fs::canonicalize(&resolved)
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or(resolved)
}

pub fn rlm_ledger_path(agent_dir: &str, sessions_dir: &str) -> String {
    let canonical = canonicalize_dir_path(sessions_dir);
    let hash = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let digest = hasher.finalize();
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        hex.chars().take(16).collect::<String>()
    };
    Path::new(agent_dir)
        .join(RLM_LEDGER_DIR)
        .join(format!("{hash}.jsonl"))
        .to_string_lossy()
        .to_string()
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn resolve_path(path: &str) -> String {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate.to_string_lossy().to_string();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(&candidate).to_string_lossy().to_string())
        .unwrap_or_else(|_| candidate.to_string_lossy().to_string())
}

fn dirname_of(path: &str) -> String {
    Path::new(path)
        .parent()
        .map(|parent| parent.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn basename_of(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Parse one ledger line. `Ok(None)` for a well-formed v:1 record with an
/// unknown op (forward compat); any other violation is an error.
pub fn parse_ledger_line(line: &str, index: usize) -> Result<Option<RlmLedgerRecord>, String> {
    let parsed: Value = serde_json::from_str(line)
        .map_err(|error| format!("Malformed RLM ledger line {}: {error}", index + 1))?;
    let Some(record) = parsed.as_object() else {
        return Err(format!(
            "Malformed RLM ledger line {}: missing v/at",
            index + 1
        ));
    };
    let version = record.get("v").and_then(Value::as_i64);
    let at = record.get("at").and_then(Value::as_str);
    if version != Some(1) || at.is_none() {
        return Err(format!(
            "Malformed RLM ledger line {}: missing v/at",
            index + 1
        ));
    }
    let at = at.unwrap_or_default().to_string();
    let field = |key: &str| record.get(key).and_then(Value::as_str).map(str::to_string);
    match record.get("op").and_then(Value::as_str) {
        Some("meta") => {
            let Some(sessions_dir) = field("sessionsDir") else {
                return Err(format!(
                    "Malformed RLM ledger line {}: meta without sessionsDir",
                    index + 1
                ));
            };
            Ok(Some(RlmLedgerRecord::Meta(RlmLedgerMetaRecord {
                v: 1,
                op: "meta".to_string(),
                at,
                sessions_dir,
            })))
        }
        Some("spawn") => {
            let depth = record.get("depth").and_then(Value::as_i64);
            let (Some(child_id), Some(parent), Some(child), Some(name)) = (
                field("childId"),
                field("parent"),
                field("child"),
                field("name"),
            ) else {
                return Err(format!(
                    "Malformed RLM ledger line {}: invalid spawn record",
                    index + 1
                ));
            };
            match depth {
                Some(depth) if depth >= 1 => {}
                _ => {
                    return Err(format!(
                        "Malformed RLM ledger line {}: invalid spawn record",
                        index + 1
                    ))
                }
            }
            Ok(Some(RlmLedgerRecord::Spawn(RlmLedgerSpawnRecord {
                v: 1,
                op: "spawn".to_string(),
                at,
                child_id,
                parent,
                child,
                depth: depth.unwrap_or(1),
                name,
            })))
        }
        Some("rename") => {
            let (Some(child_id), Some(child), Some(name)) =
                (field("childId"), field("child"), field("name"))
            else {
                return Err(format!(
                    "Malformed RLM ledger line {}: invalid rename record",
                    index + 1
                ));
            };
            Ok(Some(RlmLedgerRecord::Rename(RlmLedgerRenameRecord {
                v: 1,
                op: "rename".to_string(),
                at,
                child_id,
                child,
                name,
            })))
        }
        Some("delete") => {
            let reason = record
                .get("reason")
                .and_then(Value::as_str)
                .and_then(RlmLedgerDeleteReason::from_str);
            let (Some(child_id), Some(child), Some(reason)) =
                (field("childId"), field("child"), reason)
            else {
                return Err(format!(
                    "Malformed RLM ledger line {}: invalid delete record",
                    index + 1
                ));
            };
            Ok(Some(RlmLedgerRecord::Delete(RlmLedgerDeleteRecord {
                v: 1,
                op: "delete".to_string(),
                at,
                child_id,
                child,
                reason,
            })))
        }
        _ => Ok(None),
    }
}

fn edge_key(child_id: &str, child: &str) -> String {
    format!("{child_id}\u{0}{}", canonical_session_path(child))
}

/// Per-sessions-dir spawn ledger. Operations are serialized on an internal
/// lock; the first operation lazily seeds a missing ledger from the existing
/// per-parent registries (memoized; a seeding failure degrades to an empty
/// ledger and is never fail-closed).
pub struct RlmSpawnLedger {
    path: String,
    event_log: EventLog,
    canonical_sessions_dir: String,
    seed_source: Option<Arc<dyn RlmLedgerSeedSource>>,
    log: Arc<dyn Fn(&str) + Send + Sync>,
    seed_attempted: StdMutex<bool>,
    queue: tokio::sync::Mutex<()>,
}

impl RlmSpawnLedger {
    pub fn new(
        agent_dir: &str,
        sessions_dir: &str,
        seed_source: Option<Arc<dyn RlmLedgerSeedSource>>,
        log: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    ) -> Self {
        let canonical_sessions_dir = canonicalize_dir_path(sessions_dir);
        let path = rlm_ledger_path(agent_dir, sessions_dir);
        let logger = log.clone();
        let event_log = EventLog::new(
            path.clone(),
            EventLogOptions {
                max_bytes: Some(RLM_LEDGER_MAX_BYTES),
                max_records: Some(RLM_LEDGER_MAX_RECORDS),
                log: Some(Arc::new(move |message: String| {
                    if let Some(log) = &logger {
                        log(&format!("RLM ledger: {message}"));
                    }
                })),
            },
        );
        Self {
            path,
            event_log,
            canonical_sessions_dir,
            seed_source,
            log: log.unwrap_or_else(|| Arc::new(|_| {})),
            seed_attempted: StdMutex::new(false),
            queue: tokio::sync::Mutex::new(()),
        }
    }

    pub fn ledger_path(&self) -> &str {
        &self.path
    }

    async fn enqueue<T, F: std::future::Future<Output = T>>(&self, run: impl FnOnce() -> F) -> T {
        let _guard = self.queue.lock().await;
        let already_attempted = {
            let mut attempted = self.seed_attempted.lock().expect("seed flag poisoned");
            let previous = *attempted;
            *attempted = true;
            previous
        };
        if !already_attempted {
            if let Err(error) = self.seed().await {
                (self.log)(&format!("RLM ledger seeding failed: {error}"));
            }
        }
        run().await
    }

    pub async fn append_spawn(&self, input: RlmSpawnInput) -> Result<(), String> {
        self.enqueue(move || async move { self.append_spawn_unlocked(input) })
            .await
    }

    pub async fn append_rename(
        &self,
        child_id: &str,
        child: &str,
        name: &str,
    ) -> Result<(), String> {
        let record = RlmLedgerRecord::Rename(RlmLedgerRenameRecord {
            v: 1,
            op: "rename".to_string(),
            at: now_iso(),
            child_id: child_id.to_string(),
            child: canonical_session_path(child),
            name: name.to_string(),
        });
        self.enqueue(move || async move { self.append_record(&record) })
            .await
    }

    /// Rename by child session path alone (offline saved-session rename knows no childId).
    pub async fn append_rename_by_child_path(&self, child: &str, name: &str) -> Result<(), String> {
        let target = canonical_session_path(child);
        let name = name.to_string();
        self.enqueue(move || async move {
            let edges: Vec<RlmLedgerEdge> = self.replay_sync().into_values().collect();
            let mut first_error: Option<String> = None;
            for edge in edges {
                if edge.deleted.is_some() || canonical_session_path(&edge.child) != target {
                    continue;
                }
                let record = RlmLedgerRecord::Rename(RlmLedgerRenameRecord {
                    v: 1,
                    op: "rename".to_string(),
                    at: now_iso(),
                    child_id: edge.child_id,
                    child: target.clone(),
                    name: name.clone(),
                });
                if let Err(error) = self.append_record(&record) {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
            first_error.map_or(Ok(()), Err)
        })
        .await
    }

    pub async fn append_delete(
        &self,
        child_id: &str,
        child: &str,
        reason: RlmLedgerDeleteReason,
    ) -> Result<(), String> {
        let record = RlmLedgerRecord::Delete(RlmLedgerDeleteRecord {
            v: 1,
            op: "delete".to_string(),
            at: now_iso(),
            child_id: child_id.to_string(),
            child: canonical_session_path(child),
            reason,
        });
        self.enqueue(move || async move { self.append_record(&record) })
            .await
    }

    /// Resolves once every operation enqueued so far has completed.
    pub async fn flush(&self) {
        let _guard = self.queue.lock().await;
    }

    /// Replay edges without liveness reconciliation. Deleted edges are filtered
    /// by default; `include_deleted` keeps the tombstones.
    pub async fn edges(&self, include_deleted: bool) -> Vec<RlmLedgerEdge> {
        self.enqueue(move || async move {
            self.replay_sync()
                .into_values()
                .filter(|edge| include_deleted || edge.deleted.is_none())
                .collect()
        })
        .await
    }

    /// Family of every session rooted in this ledger's sessions dir.
    pub async fn family(&self) -> Vec<SessionInfo> {
        self.enqueue(|| self.family_unlocked()).await
    }

    /// Same-parent rows for a child session path, including the child itself.
    pub async fn siblings(&self, session_path: &str) -> Vec<SessionInfo> {
        let session_path = session_path.to_string();
        self.enqueue(move || async move { self.siblings_unlocked(&session_path).await })
            .await
    }

    async fn siblings_unlocked(&self, session_path: &str) -> Vec<SessionInfo> {
        let target = canonical_session_path(session_path);
        let family = self.family_unlocked().await;
        let edges: Vec<RlmLedgerEdge> = self
            .replay_sync()
            .into_values()
            .filter(|edge| edge.deleted.is_none())
            .collect();
        let parent_by_child: HashMap<String, String> = edges
            .iter()
            .map(|edge| {
                (
                    canonical_session_path(&edge.child),
                    canonical_session_path(&edge.parent),
                )
            })
            .collect();
        if let Some(parent) = parent_by_child.get(&target) {
            let rows: Vec<SessionInfo> = family
                .iter()
                .filter(|row| {
                    parent_by_child.get(&canonical_session_path(&row.path)) == Some(parent)
                })
                .cloned()
                .collect();
            if !rows
                .iter()
                .any(|row| canonical_session_path(&row.path) == target)
                && Path::new(&target).is_file()
            {
                return vec![self.session_row(&target, 0, None, None).await];
            }
            return rows;
        }
        // Roots are siblings of the other roots.
        let roots: Vec<SessionInfo> = family
            .iter()
            .filter(|row| row.rlm_depth == 0)
            .cloned()
            .collect();
        if roots
            .iter()
            .any(|row| canonical_session_path(&row.path) == target)
        {
            return roots;
        }
        if !Path::new(&target).is_file() {
            return Vec::new();
        }
        vec![self.session_row(&target, 0, None, None).await]
    }

    fn append_spawn_unlocked(&self, input: RlmSpawnInput) -> Result<(), String> {
        // Enforce the same invariants parse_ledger_line checks.
        if input.child_id.is_empty()
            || input.parent.is_empty()
            || input.child.is_empty()
            || input.depth < 1
        {
            return Err(format!(
                "RLM ledger: invalid spawn for {} (depth {})",
                if input.child_id.is_empty() {
                    "<missing childId>"
                } else {
                    &input.child_id
                },
                input.depth
            ));
        }
        let child_path = canonical_session_path(&input.child);
        // Advisory, per-process: catches double-admission mistakes inside this daemon.
        for edge in self.replay_sync().into_values() {
            if edge.deleted.is_none()
                && canonical_session_path(&edge.child) == child_path
                && edge.child_id != input.child_id
            {
                return Err(format!(
                    "RLM ledger: duplicate child session path {child_path} (already {})",
                    edge.child_id
                ));
            }
        }
        let record = RlmLedgerRecord::Spawn(RlmLedgerSpawnRecord {
            v: 1,
            op: "spawn".to_string(),
            at: now_iso(),
            child_id: input.child_id,
            parent: canonical_session_path(&input.parent),
            child: child_path,
            depth: input.depth,
            name: input.name,
        });
        self.append_record(&record)
    }

    /// Live edges reconciled by stat, exactly like `family()`.
    pub async fn live_edges(&self) -> Vec<RlmLedgerEdge> {
        self.enqueue(|| async {
            let edges: Vec<RlmLedgerEdge> = self
                .replay_sync()
                .into_values()
                .filter(|edge| edge.deleted.is_none())
                .collect();
            self.live_edges_unlocked(edges)
        })
        .await
    }

    fn live_edges_unlocked(&self, edges: Vec<RlmLedgerEdge>) -> Vec<RlmLedgerEdge> {
        let mut stat_cache: HashMap<String, bool> = HashMap::new();
        let mut alive: Vec<RlmLedgerEdge> = Vec::new();
        for edge in edges {
            let child_alive = cached_is_file(&mut stat_cache, &edge.child);
            let parent_alive = cached_is_file(&mut stat_cache, &edge.parent);
            if child_alive && parent_alive {
                alive.push(edge);
            }
        }
        alive
    }

    async fn family_unlocked(&self) -> Vec<SessionInfo> {
        let candidates: Vec<RlmLedgerEdge> = self
            .replay_sync()
            .into_values()
            .filter(|edge| edge.deleted.is_none())
            .collect();
        let mut alive = self.live_edges_unlocked(candidates);
        let by_child: HashSet<String> = alive
            .iter()
            .map(|edge| canonical_session_path(&edge.child))
            .collect();

        let mut root_entries: Vec<String> = std::fs::read_dir(&self.canonical_sessions_dir)
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().to_string())
                    .collect()
            })
            .unwrap_or_default();
        root_entries.retain(|name| name.ends_with(".jsonl"));
        root_entries.sort();
        let mut root_paths: Vec<String> = Vec::new();
        for entry in root_entries {
            let path = canonical_session_path(
                &Path::new(&self.canonical_sessions_dir)
                    .join(&entry)
                    .to_string_lossy()
                    .to_string(),
            );
            // Ledger children that live directly in the sessions dir are not roots.
            if by_child.contains(&path) {
                continue;
            }
            root_paths.push(path);
        }

        // Depth monotonicity is verified between ledger-known depths only: a
        // contradictory edge is dropped and logged, never fails the family.
        let depth_by_path: HashMap<String, i64> = alive
            .iter()
            .map(|edge| (canonical_session_path(&edge.child), edge.depth))
            .collect();
        alive.retain(|edge| {
            match depth_by_path.get(&canonical_session_path(&edge.parent)) {
                Some(parent_depth) if edge.depth != parent_depth + 1 => {
                    (self.log)(&format!(
                        "RLM ledger: dropped edge {} with contradictory depth (parent {}, child {})",
                        edge.child_id, parent_depth, edge.depth
                    ));
                    false
                }
                _ => true,
            }
        });

        let mut rows: Vec<SessionInfo> = Vec::new();
        for root_path in root_paths {
            rows.push(self.session_row(&root_path, 0, None, None).await);
        }
        for edge in alive {
            rows.push(
                self.session_row(
                    &canonical_session_path(&edge.child),
                    edge.depth,
                    Some(&canonical_session_path(&edge.parent)),
                    Some(&edge.name),
                )
                .await,
            );
        }
        rows
    }

    async fn session_row(
        &self,
        path: &str,
        depth: i64,
        parent_path: Option<&str>,
        name: Option<&str>,
    ) -> SessionInfo {
        // Display-grade fields are best-effort from the ordinary session-info
        // read; topology (path, depth, parent) comes EXCLUSIVELY from the ledger.
        match read_session_info(path).await {
            Some(mut info) => {
                info.parent_session_path = parent_path.map(str::to_string);
                info.rlm_depth = depth;
                if let Some(name) = name {
                    info.name = Some(name.to_string());
                }
                info
            }
            None => {
                let file_name = basename_of(path);
                SessionInfo {
                    id: file_name
                        .strip_suffix(".jsonl")
                        .unwrap_or(&file_name)
                        .to_string(),
                    path: path.to_string(),
                    cwd: String::new(),
                    name: name.map(str::to_string),
                    state: None,
                    parent_session_path: parent_path.map(str::to_string),
                    rlm_depth: depth,
                    created: 0.0,
                    modified: 0.0,
                    message_count: 0,
                    first_message: String::new(),
                    all_messages_text: String::new(),
                    agent_status: None,
                    usage: None,
                }
            }
        }
    }

    async fn seed(&self) -> Result<(), String> {
        let Some(seed_source) = self.seed_source.clone() else {
            return Ok(());
        };
        if Path::new(&self.path).exists() {
            return Ok(());
        }
        let Ok(read_dir) = std::fs::read_dir(&self.canonical_sessions_dir) else {
            return Ok(());
        };
        let mut root_names: Vec<String> = read_dir
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.ends_with(".jsonl"))
            .collect();
        root_names.sort();
        let mut queue: std::collections::VecDeque<(String, i64)> = root_names
            .into_iter()
            .map(|name| {
                (
                    Path::new(&self.canonical_sessions_dir)
                        .join(&name)
                        .to_string_lossy()
                        .to_string(),
                    0i64,
                )
            })
            .collect();
        let mut visited: HashSet<String> = queue
            .iter()
            .map(|(session_file, _)| canonical_session_path(session_file))
            .collect();
        let mut records: Vec<RlmLedgerSpawnRecord> = Vec::new();
        while let Some((session_file, depth)) = queue.pop_front() {
            for entry in seed_source
                .read_registry_for_session_file(&session_file)
                .await
            {
                if entry.status == "deleted" {
                    continue;
                }
                let child_path = canonical_session_path(&entry.session_file);
                if visited.contains(&child_path) {
                    continue;
                }
                visited.insert(child_path.clone());
                // A registry depth < 1 would be unwritable under the spawn
                // invariants; treat it as absent and derive parent depth + 1.
                let child_depth = entry
                    .rlm_depth
                    .filter(|depth| *depth >= 1)
                    .unwrap_or(depth + 1);
                if entry.child_id.is_empty() {
                    (self.log)("RLM ledger: skipped seeding a registry entry without a childId");
                    continue;
                }
                records.push(RlmLedgerSpawnRecord {
                    v: 1,
                    op: "spawn".to_string(),
                    at: now_iso(),
                    child_id: entry.child_id,
                    parent: canonical_session_path(&session_file),
                    child: child_path,
                    depth: child_depth,
                    name: entry.session_name,
                });
                queue.push_back((entry.session_file, child_depth));
            }
        }
        if records.is_empty() {
            return Ok(());
        }
        let meta = RlmLedgerMetaRecord {
            v: 1,
            op: "meta".to_string(),
            at: now_iso(),
            sessions_dir: self.canonical_sessions_dir.clone(),
        };
        let mut payload = format!(
            "{}\n",
            serde_json::to_string(&meta).map_err(|error| error.to_string())?
        );
        for record in &records {
            payload.push_str(&format!(
                "{}\n",
                serde_json::to_string(record).map_err(|error| error.to_string())?
            ));
        }
        // A seed beyond the read bounds would publish a ledger every replay
        // refuses to read; skip seeding entirely rather than publish partial
        // topology.
        if records.len() + 1 > RLM_LEDGER_MAX_RECORDS || payload.len() as u64 > RLM_LEDGER_MAX_BYTES
        {
            (self.log)(&format!(
                "RLM ledger: seed exceeds read bounds ({} records, {} bytes); skipping seeding",
                records.len(),
                payload.len()
            ));
            return Ok(());
        }
        let dir = dirname_of(&self.path);
        std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let temp_path = format!(
            "{}.seed-{}-{}",
            self.path,
            std::process::id(),
            chrono::Utc::now().timestamp_millis()
        );
        write_seed_file(&temp_path, &payload)?;
        let publish = self.publish_seed_file(&temp_path);
        let _ = std::fs::remove_file(&temp_path);
        publish
    }

    fn publish_seed_file(&self, temp_path: &str) -> Result<(), String> {
        // Atomic no-clobber publish: an existing file wins (its data is fresher
        // than the registries) and the seed is discarded.
        match std::fs::hard_link(temp_path, &self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(error) => {
                (self.log)(&format!(
                    "RLM ledger: link publish unavailable ({}); skipping seeding",
                    error.raw_os_error().unwrap_or(0)
                ));
                Ok(())
            }
        }
    }

    fn append_record(&self, record: &RlmLedgerRecord) -> Result<(), String> {
        let sessions_dir = self.canonical_sessions_dir.clone();
        self.event_log.append_sync(
            &[record.to_value()],
            true,
            Some(&move || {
                vec![serde_json::json!({
                    "v": 1,
                    "op": "meta",
                    "at": now_iso(),
                    "sessionsDir": sessions_dir,
                })]
            }),
        )
    }

    fn replay_sync(&self) -> indexmap::IndexMap<String, RlmLedgerEdge> {
        let mut edges: indexmap::IndexMap<String, RlmLedgerEdge> = indexmap::IndexMap::new();
        let records = self
            .event_log
            .replay_sync(
                |line, index| {
                    // One damaged record must not hide every other child (or
                    // its deletion tombstone). Keep the shared event log strict;
                    // only this reconstructible topology ledger skips bad rows.
                    let record = match parse_ledger_line(line, index) {
                        Ok(record) => record,
                        Err(error) => {
                            (self.log)(&format!("RLM ledger: skipped corrupt record: {error}"));
                            return Ok(None);
                        }
                    };
                    if record.is_none() {
                        (self.log)(&format!(
                            "RLM ledger: skipped record with unknown op on line {}",
                            index + 1
                        ));
                    }
                    Ok(record)
                },
                ReplayOptions::default(),
            )
            .unwrap_or_else(|error| {
                (self.log)(&format!("RLM ledger: replay failed: {error}"));
                Vec::new()
            });
        for record in records {
            match record {
                RlmLedgerRecord::Meta(_) => continue,
                RlmLedgerRecord::Spawn(record) => {
                    edges.insert(
                        edge_key(&record.child_id, &record.child),
                        RlmLedgerEdge {
                            child_id: record.child_id,
                            parent: record.parent,
                            child: record.child,
                            depth: record.depth,
                            name: record.name,
                            deleted: None,
                        },
                    );
                }
                RlmLedgerRecord::Rename(record) => {
                    if let Some(existing) =
                        edges.get_mut(&edge_key(&record.child_id, &record.child))
                    {
                        existing.name = record.name;
                    }
                }
                RlmLedgerRecord::Delete(record) => {
                    if let Some(existing) =
                        edges.get_mut(&edge_key(&record.child_id, &record.child))
                    {
                        existing.deleted = Some(record.reason);
                    }
                }
            }
        }
        edges
    }
}

#[derive(Debug, Clone)]
pub struct RlmSpawnInput {
    pub child_id: String,
    pub parent: String,
    pub child: String,
    pub depth: i64,
    pub name: String,
}

fn cached_is_file(stat_cache: &mut HashMap<String, bool>, path: &str) -> bool {
    let canonical = canonical_session_path(path);
    if let Some(cached) = stat_cache.get(&canonical) {
        return *cached;
    }
    let ok = Path::new(&canonical).is_file();
    stat_cache.insert(canonical, ok);
    ok
}

fn write_seed_file(path: &str, payload: &str) -> Result<(), String> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut handle = options.open(path).map_err(|error| error.to_string())?;
    handle
        .write_all(payload.as_bytes())
        .map_err(|error| error.to_string())?;
    handle.sync_all().map_err(|error| error.to_string())
}

// The catalog scan never visits session-artifacts, where RLM children persist:
// without this merge a passivated descendant's row (and its spend) survives only
// as long as some resident roster remembers it.
#[derive(Clone, Default)]
pub struct WithPassiveRlmDescendantInfosOptions {
    pub cwd: Option<String>,
    pub on_session: Option<Arc<dyn Fn(&SessionInfo) + Send + Sync>>,
    pub log: Option<Arc<dyn Fn(&str) + Send + Sync>>,
}

pub async fn with_passive_rlm_descendant_infos(
    saved_sessions: Vec<SessionInfo>,
    ledger: &RlmSpawnLedger,
    options: WithPassiveRlmDescendantInfosOptions,
) -> Vec<SessionInfo> {
    let mut sessions = saved_sessions.clone();
    let mut seen: HashSet<String> = saved_sessions
        .iter()
        .map(|info| canonical_session_path(&info.path))
        .collect();
    let edges = ledger.live_edges().await;
    for edge in edges {
        let child_path = canonical_session_path(&edge.child);
        if seen.contains(&child_path) {
            continue;
        }
        seen.insert(child_path.clone());
        let Some(mut info) = read_session_info(&child_path).await else {
            continue;
        };
        if let Some(cwd) = &options.cwd {
            if info.cwd.is_empty() || resolve_path(&info.cwd) != resolve_path(cwd) {
                continue;
            }
        }
        // The ledger edge is the authoritative topology; a fork can leave the
        // transcript header pointing at a dead ancestor path.
        info.parent_session_path = Some(edge.parent.clone());
        info.rlm_depth = edge.depth;
        if let Some(on_session) = &options.on_session {
            on_session(&info);
        }
        sessions.push(info);
    }
    sessions
}

/// Shared user-delete policy: only a readable no-parent transcript is
/// positively top-level; children and unknown targets tombstone via the ledger
/// BEFORE the file delete.
#[derive(Debug, Clone, PartialEq)]
pub struct TombstoneSavedSessionDeleteResult {
    pub deleted_info: Option<SessionInfo>,
    pub ledger_edge: Option<RlmLedgerEdge>,
}

pub async fn tombstone_saved_session_delete(
    ledger: &RlmSpawnLedger,
    session_path: &str,
    known_summary_runtime_kind: Option<&str>,
) -> TombstoneSavedSessionDeleteResult {
    let deleted_path = canonical_session_path(session_path);
    let deleted_info = read_session_info(session_path).await;
    let known_child = known_summary_runtime_kind == Some("subagent")
        || deleted_info
            .as_ref()
            .is_some_and(|info| info.parent_session_path.is_some())
        || deleted_info.as_ref().is_some_and(|info| info.rlm_depth > 0);
    let positively_top_level =
        !known_child && (known_summary_runtime_kind.is_some() || deleted_info.is_some());
    if positively_top_level {
        return TombstoneSavedSessionDeleteResult {
            deleted_info,
            ledger_edge: None,
        };
    }
    let edges = ledger.edges(false).await;
    // Tombstone every matching edge: a duplicate edge left live would resurrect
    // a later recreation at that path as a subagent.
    let matching: Vec<RlmLedgerEdge> = edges
        .into_iter()
        .filter(|edge| canonical_session_path(&edge.child) == deleted_path)
        .collect();
    for edge in &matching {
        let _ = ledger
            .append_delete(&edge.child_id, session_path, RlmLedgerDeleteReason::User)
            .await;
    }
    TombstoneSavedSessionDeleteResult {
        deleted_info,
        ledger_edge: matching.into_iter().next(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("rlm-ledger-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("temp root");
        root
    }

    fn write_session_file(dir: &Path, name: &str) -> String {
        let path = dir.join(name);
        std::fs::write(
            &path,
            format!(
                "{}\n",
                serde_json::json!({ "type": "session", "id": name.trim_end_matches(".jsonl") })
            ),
        )
        .expect("write session");
        path.to_string_lossy().to_string()
    }

    fn spawn(child_id: &str, parent: &str, child: &str, depth: i64, name: &str) -> RlmSpawnInput {
        RlmSpawnInput {
            child_id: child_id.to_string(),
            parent: parent.to_string(),
            child: child.to_string(),
            depth,
            name: name.to_string(),
        }
    }

    #[test]
    fn the_ledger_path_is_hashed_per_sessions_dir() {
        let agent_dir = temp_root("path");
        let sessions_dir = agent_dir.join("sessions");
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
        let path = rlm_ledger_path(
            &agent_dir.to_string_lossy(),
            &sessions_dir.to_string_lossy(),
        );
        assert!(path.contains(RLM_LEDGER_DIR));
        assert!(path.ends_with(".jsonl"));
        let file_name = basename_of(&path);
        assert_eq!(file_name.len(), ".jsonl".len() + 16);
        let _ = std::fs::remove_dir_all(&agent_dir);
    }

    #[test]
    fn ledger_lines_are_parsed_and_validated() {
        let spawn_line = serde_json::json!({
            "v": 1,
            "op": "spawn",
            "at": "2026-01-01T00:00:00.000Z",
            "childId": "c1",
            "parent": "/p.jsonl",
            "child": "/c.jsonl",
            "depth": 1,
            "name": "child"
        })
        .to_string();
        let record = parse_ledger_line(&spawn_line, 0)
            .expect("parses")
            .expect("record");
        match record {
            RlmLedgerRecord::Spawn(record) => {
                assert_eq!(record.child_id, "c1");
                assert_eq!(record.depth, 1);
            }
            other => panic!("unexpected record: {other:?}"),
        }
        let unknown_op = serde_json::json!({ "v": 1, "op": "future", "at": "x" }).to_string();
        assert!(parse_ledger_line(&unknown_op, 0).expect("ok").is_none());
        let bad_depth = serde_json::json!({
            "v": 1,
            "op": "spawn",
            "at": "x",
            "childId": "c",
            "parent": "p",
            "child": "c",
            "depth": 0,
            "name": "n"
        })
        .to_string();
        assert_eq!(
            parse_ledger_line(&bad_depth, 3).expect_err("invalid"),
            "Malformed RLM ledger line 4: invalid spawn record"
        );
        let missing_version =
            serde_json::json!({ "op": "meta", "at": "x", "sessionsDir": "/s" }).to_string();
        assert_eq!(
            parse_ledger_line(&missing_version, 0).expect_err("invalid"),
            "Malformed RLM ledger line 1: missing v/at"
        );
        assert!(parse_ledger_line("{not json", 1)
            .expect_err("invalid")
            .starts_with("Malformed RLM ledger line 2: "));
    }

    #[tokio::test]
    async fn appends_replay_into_edges_and_tombstones() {
        let root = temp_root("edges");
        let sessions = root.join("sessions");
        std::fs::create_dir_all(&sessions).expect("sessions dir");
        let agent_dir = root.to_string_lossy().to_string();
        let parent = write_session_file(&sessions, "parent.jsonl");
        let child = write_session_file(&sessions, "child.jsonl");
        let ledger = RlmSpawnLedger::new(&agent_dir, &sessions.to_string_lossy(), None, None);
        ledger
            .append_spawn(spawn("c1", &parent, &child, 1, "child"))
            .await
            .expect("spawn");
        assert_eq!(ledger.edges(false).await.len(), 1);
        ledger
            .append_rename("c1", &child, "renamed")
            .await
            .expect("rename");
        let edges = ledger.edges(false).await;
        assert_eq!(edges[0].name, "renamed");
        ledger
            .append_delete("c1", &child, RlmLedgerDeleteReason::User)
            .await
            .expect("delete");
        assert!(ledger.edges(false).await.is_empty());
        let with_deleted = ledger.edges(true).await;
        assert_eq!(with_deleted[0].deleted, Some(RlmLedgerDeleteReason::User));
        ledger
            .append_rename_by_child_path(&child, "again")
            .await
            .expect("rename by path");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn corrupt_records_preserve_children_tombstones_and_duplicate_guards() {
        use std::io::Write;
        let root = tempfile::tempdir().unwrap();
        let sessions = root.path().join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        let parent = write_session_file(&sessions, "parent.jsonl");
        let child = write_session_file(&sessions, "child.jsonl");
        let other = write_session_file(&sessions, "other.jsonl");
        let logs = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let captured = logs.clone();
        let ledger = RlmSpawnLedger::new(
            &root.path().to_string_lossy(), &sessions.to_string_lossy(), None,
            Some(Arc::new(move |message| captured.lock().unwrap().push(message.into()))),
        );
        ledger.append_spawn(spawn("c1", &parent, &child, 1, "first")).await.unwrap();
        let mut file = std::fs::OpenOptions::new().append(true).open(ledger.ledger_path()).unwrap();
        file.write_all(b"not json\n\xff\xfe\n{\"v\":1,\"op\":\"spawn\",\"at\":\"x\"}\n").unwrap();
        ledger.append_rename("c1", &child, "renamed").await.unwrap();
        ledger.append_delete("c1", &child, RlmLedgerDeleteReason::User).await.unwrap();
        ledger.append_spawn(spawn("c2", &parent, &other, 1, "second")).await.unwrap();
        assert_eq!(ledger.edges(false).await.len(), 1);
        let edges = ledger.edges(true).await;
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].name, "renamed");
        assert_eq!(edges[0].deleted, Some(RlmLedgerDeleteReason::User));
        assert!(ledger.append_spawn(spawn("duplicate", &parent, &other, 1, "bad")).await
            .unwrap_err().contains("duplicate child session path"));
        // The existing torn-tail repair still protects the next append.
        file.write_all(b"{\"v\":1,\"op\":\"spawn\"").unwrap();
        ledger.append_rename("c2", &other, "after torn tail").await.unwrap();
        assert_eq!(ledger.edges(false).await[0].name, "after torn tail");
        assert!(logs.lock().unwrap().iter().any(|line| line.contains("skipped corrupt record")));
    }

    #[tokio::test]
    async fn spawn_rejects_invalid_and_duplicate_inputs() {
        let root = temp_root("spawn-guard");
        let sessions = root.join("sessions");
        std::fs::create_dir_all(&sessions).expect("sessions dir");
        let agent_dir = root.to_string_lossy().to_string();
        let parent = write_session_file(&sessions, "parent.jsonl");
        let child = write_session_file(&sessions, "child.jsonl");
        let ledger = RlmSpawnLedger::new(&agent_dir, &sessions.to_string_lossy(), None, None);
        assert_eq!(
            ledger
                .append_spawn(spawn("", &parent, &child, 1, "n"))
                .await
                .expect_err("invalid"),
            "RLM ledger: invalid spawn for <missing childId> (depth 1)"
        );
        assert_eq!(
            ledger
                .append_spawn(spawn("c1", &parent, &child, 0, "n"))
                .await
                .expect_err("invalid"),
            "RLM ledger: invalid spawn for c1 (depth 0)"
        );
        ledger
            .append_spawn(spawn("c1", &parent, &child, 1, "n"))
            .await
            .expect("spawn");
        assert!(ledger
            .append_spawn(spawn("c2", &parent, &child, 1, "n"))
            .await
            .expect_err("duplicate")
            .starts_with("RLM ledger: duplicate child session path "));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn family_and_siblings_use_ledger_topology() {
        let root = temp_root("family");
        let sessions = root.join("sessions");
        std::fs::create_dir_all(&sessions).expect("sessions dir");
        let agent_dir = root.to_string_lossy().to_string();
        let parent = write_session_file(&sessions, "parent.jsonl");
        let child = write_session_file(&sessions, "child.jsonl");
        let other = write_session_file(&sessions, "other.jsonl");
        let ledger = RlmSpawnLedger::new(&agent_dir, &sessions.to_string_lossy(), None, None);
        ledger
            .append_spawn(spawn("c1", &parent, &child, 1, "child"))
            .await
            .expect("spawn");
        let family = ledger.family().await;
        assert_eq!(family.len(), 3);
        let child_row = family
            .iter()
            .find(|row| canonical_session_path(&row.path) == canonical_session_path(&child))
            .expect("child row");
        assert_eq!(child_row.rlm_depth, 1);
        assert_eq!(
            child_row.parent_session_path.as_deref(),
            Some(canonical_session_path(&parent).as_str())
        );
        let siblings = ledger.siblings(&parent).await;
        assert_eq!(siblings.len(), 2);
        assert!(siblings
            .iter()
            .any(|row| canonical_session_path(&row.path) == canonical_session_path(&other)));
        // TypeScript siblings() includes rows sharing this child's parent,
        // including itself; the two root sessions are a separate sibling set.
        assert_eq!(ledger.siblings(&child).await.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn dead_edges_are_dropped_and_delete_policy_tombstones_children() {
        let root = temp_root("dead");
        let sessions = root.join("sessions");
        std::fs::create_dir_all(&sessions).expect("sessions dir");
        let agent_dir = root.to_string_lossy().to_string();
        let parent = write_session_file(&sessions, "parent.jsonl");
        let child = write_session_file(&sessions, "child.jsonl");
        let ledger = RlmSpawnLedger::new(&agent_dir, &sessions.to_string_lossy(), None, None);
        ledger
            .append_spawn(spawn("c1", &parent, &child, 1, "child"))
            .await
            .expect("spawn");
        std::fs::remove_file(&parent).expect("remove parent");
        assert!(ledger.live_edges().await.is_empty());

        let result = tombstone_saved_session_delete(&ledger, &child, Some("subagent")).await;
        assert!(result.ledger_edge.is_some());
        assert!(result.deleted_info.is_some());
        let edges = ledger.edges(true).await;
        assert_eq!(edges[0].deleted, Some(RlmLedgerDeleteReason::User));

        let top_level = tombstone_saved_session_delete(&ledger, &parent, Some("top-level")).await;
        assert!(top_level.ledger_edge.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn seeding_publishes_registry_edges_once() {
        let root = temp_root("seed");
        let sessions = root.join("sessions");
        std::fs::create_dir_all(&sessions).expect("sessions dir");
        let agent_dir = root.to_string_lossy().to_string();
        let parent = write_session_file(&sessions, "parent.jsonl");
        // Registries seed descendants outside the root session directory;
        // a file already in the root queue is deliberately visited once.
        let children = sessions.join("children");
        std::fs::create_dir_all(&children).expect("children dir");
        let child = write_session_file(&children, "child.jsonl");

        struct Source {
            parent: String,
            child: String,
        }

        #[async_trait::async_trait]
        impl RlmLedgerSeedSource for Source {
            async fn read_registry_for_session_file(
                &self,
                session_file: &str,
            ) -> Vec<RlmLedgerSeedRegistryEntry> {
                if canonical_session_path(session_file) != canonical_session_path(&self.parent) {
                    return Vec::new();
                }
                vec![RlmLedgerSeedRegistryEntry {
                    child_id: "seed-child".to_string(),
                    session_name: "seeded".to_string(),
                    session_file: self.child.clone(),
                    rlm_depth: Some(0),
                    status: "running".to_string(),
                }]
            }
        }

        let ledger = RlmSpawnLedger::new(
            &agent_dir,
            &sessions.to_string_lossy(),
            Some(Arc::new(Source {
                parent: parent.clone(),
                child: child.clone(),
            })),
            None,
        );
        let edges = ledger.edges(false).await;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].child_id, "seed-child");
        assert_eq!(edges[0].depth, 1);
        assert!(Path::new(ledger.ledger_path()).exists());

        let second = RlmSpawnLedger::new(&agent_dir, &sessions.to_string_lossy(), None, None);
        assert_eq!(second.edges(false).await.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn passive_descendants_are_merged_and_cwd_filtered() {
        let root = temp_root("passive");
        let sessions = root.join("sessions");
        std::fs::create_dir_all(&sessions).expect("sessions dir");
        let agent_dir = root.to_string_lossy().to_string();
        let parent = write_session_file(&sessions, "parent.jsonl");
        let child = write_session_file(&sessions, "child.jsonl");
        let ledger = RlmSpawnLedger::new(&agent_dir, &sessions.to_string_lossy(), None, None);
        ledger
            .append_spawn(spawn("c1", &parent, &child, 1, "child"))
            .await
            .expect("spawn");
        let saved = vec![read_session_info(&parent).await.expect("parent info")];
        let merged =
            with_passive_rlm_descendant_infos(saved.clone(), &ledger, Default::default()).await;
        assert_eq!(merged.len(), 2);
        let filtered = with_passive_rlm_descendant_infos(
            saved,
            &ledger,
            WithPassiveRlmDescendantInfosOptions {
                cwd: Some("/definitely/not/the/cwd".to_string()),
                ..Default::default()
            },
        )
        .await;
        assert_eq!(filtered.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn the_legacy_registry_read_is_tolerant_and_latest_wins() {
        let root = temp_root("legacy");
        let path = root.join("rlm-subagents.jsonl");
        let entry = |name: &str, status: &str| {
            serde_json::json!({
                "type": "rlm_subagent",
                "childId": "c1",
                "sessionName": name,
                "sessionFile": "/tmp/c1.jsonl",
                "parentSessionId": "p",
                "createdAt": 1,
                "updatedAt": "x",
                "status": status
            })
            .to_string()
        };
        std::fs::write(
            &path,
            format!(
                "{}\nnot json\n{}\n",
                entry("first", "running"),
                entry("second", "deleted")
            ),
        )
        .expect("write registry");
        let entries = read_legacy_rlm_subagent_registry(
            &path.to_string_lossy(),
            ReadLegacyRlmSubagentRegistryOptions::default(),
        )
        .await
        .expect("reads");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].session_name, "second");
        assert_eq!(entries[0].session_dir, "/tmp");

        let missing = read_legacy_rlm_subagent_registry(
            &root.join("missing.jsonl").to_string_lossy(),
            ReadLegacyRlmSubagentRegistryOptions {
                throw_on_read_error: true,
                ..Default::default()
            },
        )
        .await
        .expect("missing is tolerated");
        assert!(missing.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
