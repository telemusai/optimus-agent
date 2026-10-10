//! Port of packages/coding-agent/src/core/memory/store.ts
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::core::refinement::refinement::{
    apply_refinement_proposal, ApplyRefinementOptions, HarnessRefinementEvent, HarnessScope,
    HarnessState, RefinementEdit, RefinementProposal, RefinementResult,
};
use crate::utils::atomic_file::{write_file_atomic_sync, WriteFileAtomicOptions};

use super::evidence::{hash, MemoryOrigin, MemorySource};
use super::project::ProjectIdentity;

pub type JsonMap = Map<String, Value>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedConfig {
    pub url: String,
    #[serde(rename = "tokenFile")]
    pub token_file: String,
}

/// `shared?: { url, tokenFile } | null` - `None` is an absent key, `Some(None)` is `null`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemorySettings {
    pub recall: bool,
    pub learning: bool,
    #[serde(rename = "maxRecallChars")]
    pub max_recall_chars: i64,
    #[serde(rename = "maxRecallEntries")]
    pub max_recall_entries: i64,
    #[serde(rename = "maxExtractionTokens")]
    pub max_extraction_tokens: i64,
    #[serde(rename = "maxImportBytes")]
    pub max_import_bytes: i64,
    #[serde(rename = "maxImportChunkChars")]
    pub max_import_chunk_chars: i64,
    #[serde(rename = "maxImportChunksPerRun")]
    pub max_import_chunks_per_run: i64,
    /// Optional override for the session-import extraction instruction.
    /// None keeps the built-in default instruction.
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "importInstructions")]
    pub import_instructions: Option<String>,
    /// Distill the recall query with one bounded LLM call before lexical search.
    /// Falls back to the raw message text if the call fails or returns nothing.
    #[serde(default, rename = "recallQueryDistillation")]
    pub recall_query_distillation: bool,
    /// Rerank lexical search hits with one bounded LLM call before recall rendering.
    /// Falls back to lexical order if the call fails or returns nothing parseable.
    #[serde(default, rename = "recallRerank")]
    pub recall_rerank: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared: Option<Option<SharedConfig>>,
}

impl MemorySettings {
    pub fn shared_config(&self) -> Option<&SharedConfig> {
        self.shared.as_ref().and_then(|value| value.as_ref())
    }
}

pub fn default_memory_settings() -> MemorySettings {
    MemorySettings {
        recall: true,
        learning: true,
        max_recall_chars: 6000,
        max_recall_entries: 6,
        max_extraction_tokens: 4096,
        max_import_bytes: 32 * 1024 * 1024,
        max_import_chunk_chars: 40000,
        max_import_chunks_per_run: 4,
        import_instructions: None,
        recall_query_distillation: false,
        recall_rerank: false,
        shared: None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub schema: f64,
    #[serde(rename = "projectId")]
    pub project_id: String,
    pub revision: i64,
    pub history: Vec<RefinementResult>,
    pub events: IndexMap<String, String>,
}

/// `MemoryDocument extends HarnessState`: the harness fields sit at the top level
/// next to the `memory` metadata object, in that serialization order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryDocument {
    pub schema: f64,
    pub entries:
        IndexMap<String, IndexMap<String, crate::core::refinement::refinement::HarnessEntry>>,
    pub refinements: Vec<HarnessRefinementEvent>,
    pub memory: MemoryMetadata,
}

impl MemoryDocument {
    pub fn harness(&self) -> HarnessState {
        HarnessState {
            schema: self.schema,
            entries: self.entries.clone(),
            refinements: self.refinements.clone(),
        }
    }

    pub fn set_harness(&mut self, state: &HarnessState) {
        self.schema = state.schema;
        self.entries = state.entries.clone();
        self.refinements = state.refinements.clone();
    }
}

pub fn empty_document(project_id: &str) -> MemoryDocument {
    let mut entries: IndexMap<
        String,
        IndexMap<String, crate::core::refinement::refinement::HarnessEntry>,
    > = IndexMap::new();
    entries.insert("memory".to_string(), IndexMap::new());
    entries.insert("prompt".to_string(), IndexMap::new());
    entries.insert("skill".to_string(), IndexMap::new());
    entries.insert("subagent".to_string(), IndexMap::new());
    MemoryDocument {
        schema: 1.0,
        entries,
        refinements: Vec::new(),
        memory: MemoryMetadata {
            schema: 1.0,
            project_id: project_id.to_string(),
            revision: 0,
            history: Vec::new(),
            events: IndexMap::new(),
        },
    }
}

pub fn record(value: &Value) -> Result<JsonMap, String> {
    match value {
        Value::Object(map) => Ok(map.clone()),
        _ => Err("Expected an object".to_string()),
    }
}

fn safe_integer(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Number(number)) => number
            .as_f64()
            .filter(|value| value.fract() == 0.0)
            .map(|value| value as i64),
        _ => None,
    }
}

pub fn validate_document(value: &Value, project_id: &str) -> Result<MemoryDocument, String> {
    let doc = record(value)?;
    let metadata = record(doc.get("memory").unwrap_or(&Value::Null))?;
    let revision = safe_integer(metadata.get("revision"));
    if doc.get("schema").and_then(Value::as_f64) != Some(1.0)
        || metadata.get("schema").and_then(Value::as_f64) != Some(1.0)
        || metadata.get("projectId").and_then(Value::as_str) != Some(project_id)
        || revision.map(|value| value < 0).unwrap_or(true)
        || !matches!(metadata.get("history"), Some(Value::Array(_)))
        || !matches!(doc.get("refinements"), Some(Value::Array(_)))
    {
        return Err("Invalid project memory snapshot".to_string());
    }
    record(metadata.get("events").unwrap_or(&Value::Null))?;
    let entries = record(doc.get("entries").unwrap_or(&Value::Null))?;
    for kind in ["memory", "prompt", "skill", "subagent"] {
        let bucket = record(entries.get(kind).unwrap_or(&Value::Null))?;
        for (id, raw) in &bucket {
            let entry = record(raw)?;
            if entry.get("id").and_then(Value::as_str) != Some(id.as_str())
                || entry.get("kind").and_then(Value::as_str) != Some(kind)
                || !matches!(entry.get("title"), Some(Value::String(_)))
                || !matches!(entry.get("content"), Some(Value::String(_)))
                || safe_integer(entry.get("version")).is_none()
            {
                return Err("Invalid memory entry".to_string());
            }
            let entry_metadata = record(entry.get("metadata").unwrap_or(&Value::Null))?;
            if entry_metadata.get("projectId").and_then(Value::as_str) != Some(project_id) {
                return Err("Entry belongs to another project".to_string());
            }
            record(entry.get("reference").unwrap_or(&Value::Null))?;
            record(entry.get("arguments").unwrap_or(&Value::Null))?;
        }
    }
    serde_json::from_value(value.clone()).map_err(|_| "Invalid project memory snapshot".to_string())
}

pub fn read_json(file: &str) -> Result<Value, String> {
    let raw = std::fs::read_to_string(file).map_err(|error| error.to_string())?;
    serde_json::from_str(&raw).map_err(|error| error.to_string())
}

pub fn write_json(file: &str, value: &Value) -> Result<(), String> {
    let body = format!(
        "{}\n",
        serde_json::to_string_pretty(value).unwrap_or_default()
    );
    write_file_atomic_sync(
        file,
        &body,
        WriteFileAtomicOptions {
            mode: Some(0o600),
            fsync: true,
            fsync_dir: true,
            ..Default::default()
        },
    )
    .map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Default)]
pub struct PartialMemorySettings {
    pub recall: Option<bool>,
    pub learning: Option<bool>,
    pub max_recall_chars: Option<i64>,
    pub max_recall_entries: Option<i64>,
    pub max_extraction_tokens: Option<i64>,
    pub max_import_bytes: Option<i64>,
    pub max_import_chunk_chars: Option<i64>,
    pub max_import_chunks_per_run: Option<i64>,
    pub import_instructions: Option<String>,
    pub recall_query_distillation: Option<bool>,
    pub recall_rerank: Option<bool>,
    pub shared: Option<Option<SharedConfig>>,
}

const LIMITS: [(&str, i64, i64); 6] = [
    ("maxRecallChars", 0, 32000),
    ("maxRecallEntries", 0, 50),
    ("maxExtractionTokens", 256, 32000),
    ("maxImportBytes", 1024, 128 * 1024 * 1024),
    ("maxImportChunkChars", 1000, 80000),
    ("maxImportChunksPerRun", 1, 64),
];

pub fn validate_settings(value: &Value) -> Result<PartialMemorySettings, String> {
    let source = record(value)?;
    let mut result = PartialMemorySettings::default();
    for key in [
        "recall",
        "learning",
        "recallQueryDistillation",
        "recallRerank",
    ] {
        if let Some(raw) = source.get(key) {
            let boolean = raw
                .as_bool()
                .ok_or_else(|| format!("{key} must be boolean"))?;
            match key {
                "recall" => result.recall = Some(boolean),
                "learning" => result.learning = Some(boolean),
                "recallQueryDistillation" => {
                    result.recall_query_distillation = Some(boolean)
                }
                "recallRerank" => result.recall_rerank = Some(boolean),
                _ => {}
            }
        }
    }
    for (key, low, high) in LIMITS {
        if let Some(raw) = source.get(key) {
            let number = safe_integer(Some(raw)).filter(|value| *value >= low && *value <= high);
            let number = number.ok_or_else(|| format!("Invalid {key}"))?;
            match key {
                "maxRecallChars" => result.max_recall_chars = Some(number),
                "maxRecallEntries" => result.max_recall_entries = Some(number),
                "maxExtractionTokens" => result.max_extraction_tokens = Some(number),
                "maxImportBytes" => result.max_import_bytes = Some(number),
                "maxImportChunkChars" => result.max_import_chunk_chars = Some(number),
                _ => result.max_import_chunks_per_run = Some(number),
            }
        }
    }
    if let Some(Value::String(instructions)) = source.get("importInstructions") {
        if !instructions.trim().is_empty() {
            result.import_instructions = Some(instructions.trim().to_string());
        }
    }
    match source.get("shared") {
        Some(Value::Null) => result.shared = Some(None),
        Some(Value::Object(_)) => {
            let shared = record(source.get("shared").unwrap())?;
            let url = shared
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| "Sharing requires url and tokenFile".to_string())?;
            let token_file = shared
                .get("tokenFile")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| "Sharing requires url and tokenFile".to_string())?;
            let parsed = url::Url::parse(url).map_err(|_| "Invalid URL".to_string())?;
            let loopback =
                ["127.0.0.1", "[::1]", "localhost"].contains(&parsed.host_str().unwrap_or(""));
            if !parsed.username().is_empty()
                || parsed.password().is_some()
                || parsed.query().is_some()
                || parsed.fragment().is_some()
                || (parsed.scheme() != "https" && !(parsed.scheme() == "http" && loopback))
            {
                return Err("Use HTTPS or a loopback SSH tunnel for memory sharing".to_string());
            }
            result.shared = Some(Some(SharedConfig {
                url: url.trim_end_matches('/').to_string(),
                token_file: token_file.to_string(),
            }));
        }
        Some(_) => {
            record(source.get("shared").unwrap())?;
        }
        None => {}
    }
    Ok(result)
}

#[derive(Debug, Clone)]
pub struct MemoryStore {
    pub agent_dir: String,
    pub project: ProjectIdentity,
    pub dir: String,
    pub path: String,
    pub host_id: String,
}

impl MemoryStore {
    pub fn new(agent_dir: &str, project: ProjectIdentity) -> Result<MemoryStore, String> {
        let id_re = regex::Regex::new(r"^project_[A-Za-z0-9_-]{1,80}$").unwrap();
        if !id_re.is_match(&project.id) {
            return Err("Invalid project ID".to_string());
        }
        let dir = Path::new(agent_dir)
            .join("memory")
            .join("projects")
            .join(&project.id);
        let path = dir.join("harness_state.json");
        let root = Path::new(agent_dir).join("memory");
        std::fs::create_dir_all(&root).map_err(|error| error.to_string())?;
        let host_path = root.join("host-id.json");
        if !host_path.exists() {
            let _lock = super::acquire_lock_sync(
                &root.to_string_lossy(),
                super::MemoryLockRetries {
                    stale_ms: 10_000,
                    retries: 0,
                    min_timeout_ms: 0,
                    max_timeout_ms: 0,
                },
            )?;
            if !host_path.exists() {
                write_json(
                    &host_path.to_string_lossy(),
                    &serde_json::json!({"id": uuid::Uuid::new_v4().to_string()}),
                )?;
            }
        }
        let host = record(&read_json(&host_path.to_string_lossy())?)?;
        let host_id = host
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| {
                regex::Regex::new(r"^[a-f0-9-]{36}$")
                    .unwrap()
                    .is_match(value)
            })
            .ok_or_else(|| "Invalid memory host identity".to_string())?
            .to_string();
        Ok(MemoryStore {
            agent_dir: agent_dir.to_string(),
            project,
            dir: dir.to_string_lossy().to_string(),
            path: path.to_string_lossy().to_string(),
            host_id,
        })
    }

    pub fn read(&self) -> Result<MemoryDocument, String> {
        if Path::new(&self.path).exists() {
            validate_document(&read_json(&self.path)?, &self.project.id)
        } else {
            Ok(empty_document(&self.project.id))
        }
    }

    pub fn settings(&self) -> MemorySettings {
        let global_file = Path::new(&self.agent_dir).join("settings.json");
        let global = if global_file.exists() {
            read_json(&global_file.to_string_lossy())
                .ok()
                .and_then(|value| record(&value).ok())
                .unwrap_or_default()
        } else {
            JsonMap::new()
        };
        let local_file = Path::new(&self.dir).join("settings.json");
        let mut settings = default_memory_settings();
        let global_memory = global
            .get("memory")
            .cloned()
            .unwrap_or(Value::Object(JsonMap::new()));
        apply_settings(
            &mut settings,
            &validate_settings(&global_memory).unwrap_or_default(),
        );
        if local_file.exists() {
            if let Ok(value) = read_json(&local_file.to_string_lossy()) {
                apply_settings(
                    &mut settings,
                    &validate_settings(&value).unwrap_or_default(),
                );
            }
        }
        settings
    }

    pub async fn exclusive<T, F>(&self, operation: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
    {
        std::fs::create_dir_all(&self.dir).map_err(|error| error.to_string())?;
        // The lock is held for the duration of the synchronous body: acquire,
        // run, then release, which is what `lockSync(dir)` wrapping gives the TS.
        let result = {
            let _lock = super::acquire_lock(
                &self.dir,
                super::MemoryLockRetries {
                    stale_ms: 30_000,
                    retries: 30,
                    min_timeout_ms: 20,
                    max_timeout_ms: 200,
                },
            )
            .await?;
            operation()
        };
        result
    }

    /// `exclusive` with an async body: the directory lock is held across awaits,
    /// matching the TypeScript where the lock wraps the awaited callback.
    pub async fn exclusive_async<T, F, Fut>(&self, operation: F) -> Result<T, String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
    {
        std::fs::create_dir_all(&self.dir).map_err(|error| error.to_string())?;
        let _lock = super::acquire_lock(
            &self.dir,
            super::MemoryLockRetries {
                stale_ms: 30_000,
                retries: 30,
                min_timeout_ms: 20,
                max_timeout_ms: 200,
            },
        )
        .await?;
        operation().await
    }

    pub async fn configure(&self, value: &Value) -> Result<MemorySettings, String> {
        let patch = validate_settings(value)?;
        self.exclusive(|| {
            let path = Path::new(&self.dir).join("settings.json");
            let mut merged = if path.exists() {
                read_json(&path.to_string_lossy())
                    .ok()
                    .and_then(|value| record(&value).ok())
                    .unwrap_or_default()
            } else {
                JsonMap::new()
            };
            merge_settings_patch(&mut merged, &patch);
            write_json(&path.to_string_lossy(), &Value::Object(merged))?;
            Ok(())
        })
        .await?;
        Ok(self.settings())
    }

    pub fn backup(&self, doc: Option<MemoryDocument>) -> Result<String, String> {
        let doc = match doc {
            Some(doc) => doc,
            None => self.read()?,
        };
        let body = serde_json::to_string(&doc).unwrap_or_default();
        let digest = hash(&body);
        let id = format!(
            "backup_{}_{}",
            doc.memory.revision,
            &digest[..16.min(digest.len())]
        );
        let dir = Path::new(&self.dir).join("backups");
        std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        write_json(
            &dir.join(format!("{id}.json")).to_string_lossy(),
            &serde_json::json!({"schema": 1, "sha256": digest, "state": doc}),
        )?;
        Ok(id)
    }

    pub async fn restore(&self, id: &str) -> Result<i64, String> {
        if !regex::Regex::new(r"^backup_\d+_[a-f0-9]{16}$")
            .unwrap()
            .is_match(id)
        {
            return Err("Invalid backup ID".to_string());
        }
        let id = id.to_string();
        self.exclusive(move || {
            let path = Path::new(&self.dir)
                .join("backups")
                .join(format!("{id}.json"));
            let staged = record(&read_json(&path.to_string_lossy())?)?;
            let state = staged.get("state").cloned().unwrap_or(Value::Null);
            if staged.get("schema").and_then(Value::as_f64) != Some(1.0)
                || hash(&serde_json::to_string(&state).unwrap_or_default())
                    != staged.get("sha256").and_then(Value::as_str).unwrap_or("")
            {
                return Err("Backup checksum mismatch".to_string());
            }
            let mut restored = validate_document(&state, &self.project.id)?;
            let current = self.read()?;
            self.backup(Some(current.clone()))?;
            restored.memory.revision = current.memory.revision + 1;
            // Preserve delivery receipts so restoring old content cannot replay already committed jobs.
            for (key, value) in &current.memory.events {
                restored.memory.events.insert(key.clone(), value.clone());
            }
            write_json(
                &self.path,
                &serde_json::to_value(&restored).unwrap_or(Value::Null),
            )?;
            Ok(restored.memory.revision)
        })
        .await
    }

    pub async fn apply(
        &self,
        proposal: &RefinementProposal,
        options: ApplyOptions,
    ) -> Result<RefinementResult, String> {
        let proposal = proposal.clone();
        let options = options.clone();
        self.exclusive(move || {
            let doc = self.read()?;
            let fingerprint = hash(
                &serde_json::to_string(&serde_json::json!({
                    "proposal": proposal,
                    "sources": options.sources.as_deref().unwrap_or(&[]),
                    "host": options.host,
                    "replaceMetadata": options.replace_metadata,
                }))
                .unwrap_or_default(),
            );
            if let Some(prior) = doc.memory.events.get(&options.event_id) {
                if prior != &fingerprint {
                    return Err("Event ID reused with different content".to_string());
                }
                let result = doc
                    .memory
                    .history
                    .iter()
                    .find(|item| item.id == options.event_id);
                return match result {
                    Some(result) => Ok(result.clone()),
                    None => Err(
                        "Event already committed before restore; inspect current memory before resubmitting"
                            .to_string(),
                    ),
                };
            }
            if options.automatic && !self.settings().learning {
                return Err("Automatic learning is paused".to_string());
            }
            if doc.memory.revision != options.expected_revision {
                return Err("Memory changed; review current revision before applying".to_string());
            }
            if !regex::Regex::new(r"^[A-Za-z0-9_-]{1,160}$")
                .unwrap()
                .is_match(&options.event_id)
            {
                return Err("Invalid event ID".to_string());
            }
            let mut edits = Vec::new();
            for edit in &proposal.edits {
                if let Some(id) = &edit.id {
                    if !regex::Regex::new(r"^[A-Za-z0-9_-]{1,160}$")
                        .unwrap()
                        .is_match(id)
                        || ["__proto__", "constructor", "prototype"].contains(&id.as_str())
                    {
                        return Err("Invalid memory ID".to_string());
                    }
                }
                if let Some(project_id) = edit
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("projectId"))
                    .and_then(Value::as_str)
                {
                    if project_id != self.project.id {
                        return Err("Memory belongs to another project".to_string());
                    }
                }
                let before = edit
                    .id
                    .as_ref()
                    .and_then(|id| {
                        doc.entries
                            .get(&edit.kind)
                            .and_then(|bucket| bucket.get(id))
                    })
                    .cloned();
                let before_host = before
                    .as_ref()
                    .and_then(|entry| entry.metadata.get("hostId"))
                    .and_then(Value::as_str)
                    .filter(|host| !host.is_empty())
                    .map(str::to_string);
                if let Some(host) = &before_host {
                    if host != &self.host_id {
                        return Err("Memory belongs to another host".to_string());
                    }
                }
                let mut metadata = if options.replace_metadata {
                    JsonMap::new()
                } else {
                    before
                        .as_ref()
                        .map(|entry| entry.metadata.clone())
                        .unwrap_or_default()
                };
                if let Some(extra) = &edit.metadata {
                    for (key, value) in extra {
                        metadata.insert(key.clone(), value.clone());
                    }
                }
                let metadata_host = metadata
                    .get("hostId")
                    .and_then(Value::as_str)
                    .filter(|host| !host.is_empty())
                    .map(str::to_string);
                if let Some(host) = &metadata_host {
                    if host != &self.host_id {
                        return Err("Memory belongs to another host".to_string());
                    }
                }
                let host_id = if options.host {
                    self.host_id.clone()
                } else {
                    metadata_host.clone().unwrap_or_default()
                };
                if edit.action == "create" {
                    let content = edit.content.clone().unwrap_or_default();
                    let duplicate = doc
                        .entries
                        .get(&edit.kind)
                        .map(|bucket| {
                            bucket.values().find(|entry| {
                                entry.content.trim() == content.trim()
                                    && entry
                                        .metadata
                                        .get("hostId")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        == host_id
                                    && !entry.metadata.contains_key("supersededBy")
                            })
                        })
                        .flatten();
                    if let Some(duplicate) = duplicate {
                        return Err(format!("Exact duplicate: {}", duplicate.id));
                    }
                }
                let superseded = metadata.contains_key("supersededBy");
                metadata.insert(
                    "projectId".to_string(),
                    Value::String(self.project.id.clone()),
                );
                // JSON.stringify omits the TS undefined hostId for project memories.
                if host_id.is_empty() {
                    metadata.remove("hostId");
                } else {
                    metadata.insert("hostId".to_string(), Value::String(host_id));
                }
                metadata.insert(
                    "sources".to_string(),
                    Value::Array(
                        options
                            .sources
                            .clone()
                            .or_else(|| {
                                metadata
                                    .get("sources")
                                    .and_then(Value::as_array)
                                    .map(|sources| {
                                        sources
                                            .iter()
                                            .filter_map(|value| {
                                                serde_json::from_value(value.clone()).ok()
                                            })
                                            .collect::<Vec<MemorySource>>()
                                    })
                            })
                            .unwrap_or_default()
                            .iter()
                            .map(|source| serde_json::to_value(source).unwrap_or(Value::Null))
                            .collect(),
                    ),
                );
                metadata.insert(
                    "status".to_string(),
                    Value::String(if superseded { "superseded" } else { "current" }.to_string()),
                );
                let mut next = edit.clone();
                next.metadata = Some(metadata);
                edits.push(next);
            }
            let mut candidate = doc.clone();
            let mut harness = candidate.harness();
            let mut applied = apply_refinement_proposal(
                &mut harness,
                &RefinementProposal {
                    summary: proposal.summary.clone(),
                    rationale: proposal.rationale.clone(),
                    expected_outcome: proposal.expected_outcome.clone(),
                    edits,
                },
                ApplyRefinementOptions {
                    id: options.event_id.clone(),
                    rollback_of: None,
                    scope: Some(HarnessScope::Local),
                    baseline_state: None,
                },
            );
            if applied.applied_edits.iter().any(|edit| !edit.applied) {
                let message = applied
                    .applied_edits
                    .iter()
                    .filter(|edit| !edit.applied)
                    .map(|edit| edit.error.clone().unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(message);
            }
            candidate.set_harness(&harness);
            applied.harness_state_path = self.path.clone();
            candidate.memory.revision += 1;
            candidate
                .memory
                .events
                .insert(options.event_id.clone(), fingerprint);
            candidate.memory.history.push(applied.clone());
            self.backup(Some(doc))?;
            write_json(
                &self.path,
                &serde_json::to_value(&candidate).unwrap_or(Value::Null),
            )?;
            Ok(applied)
        })
        .await
    }

    pub async fn rollback(
        &self,
        id: &str,
        expected_revision: i64,
    ) -> Result<RefinementResult, String> {
        let doc = self.read()?;
        let result = doc
            .memory
            .history
            .iter()
            .find(|item| item.id == id)
            .cloned()
            .ok_or_else(|| "Unknown refinement ID".to_string())?;
        let mut edits = Vec::new();
        for edit in result.applied_edits.iter().rev() {
            if !edit.applied {
                continue;
            }
            let current = doc
                .entries
                .get(&edit.edit.kind)
                .and_then(|bucket| bucket.get(&edit.id));
            if serde_json::to_string(&current).unwrap_or_default()
                != serde_json::to_string(&edit.after).unwrap_or_default()
            {
                return Err(format!("Entry {} has changed since this edit", edit.id));
            }
            match &edit.before {
                // `{ ...edit.before, action }`: the prior entry is replayed as an edit.
                Some(before) => edits.push(RefinementEdit {
                    action: if edit.after.is_some() {
                        "update"
                    } else {
                        "create"
                    }
                    .to_string(),
                    kind: before.kind.as_str().to_string(),
                    id: Some(before.id.clone()),
                    title: Some(before.title.clone()),
                    content: Some(before.content.clone()),
                    path: Some(before.path.clone()),
                    reference: Some(before.reference.clone()),
                    arguments: Some(before.arguments.clone()),
                    metadata: Some(before.metadata.clone()),
                    reason: None,
                }),
                None => edits.push(RefinementEdit {
                    action: "delete".to_string(),
                    kind: edit.edit.kind.clone(),
                    id: Some(edit.id.clone()),
                    title: None,
                    content: None,
                    path: None,
                    reference: None,
                    arguments: None,
                    metadata: None,
                    reason: None,
                }),
            }
        }
        self.apply(
            &RefinementProposal {
                summary: format!("Rollback {id}"),
                rationale: "Explicit rollback".to_string(),
                expected_outcome: "Restore prior memory".to_string(),
                edits,
            },
            ApplyOptions {
                event_id: format!("rollback_{id}_{expected_revision}"),
                expected_revision,
                sources: None,
                host: false,
                automatic: false,
                replace_metadata: true,
            },
        )
        .await
    }
}

#[derive(Debug, Clone, Default)]
pub struct ApplyOptions {
    pub event_id: String,
    pub expected_revision: i64,
    pub sources: Option<Vec<MemorySource>>,
    pub host: bool,
    pub automatic: bool,
    pub replace_metadata: bool,
}

fn apply_settings(settings: &mut MemorySettings, patch: &PartialMemorySettings) {
    if let Some(value) = patch.recall {
        settings.recall = value;
    }
    if let Some(value) = patch.learning {
        settings.learning = value;
    }
    if let Some(value) = patch.max_recall_chars {
        settings.max_recall_chars = value;
    }
    if let Some(value) = patch.max_recall_entries {
        settings.max_recall_entries = value;
    }
    if let Some(value) = patch.max_extraction_tokens {
        settings.max_extraction_tokens = value;
    }
    if let Some(value) = patch.max_import_bytes {
        settings.max_import_bytes = value;
    }
    if let Some(value) = patch.max_import_chunk_chars {
        settings.max_import_chunk_chars = value;
    }
    if let Some(value) = patch.max_import_chunks_per_run {
        settings.max_import_chunks_per_run = value;
    }
    if let Some(value) = &patch.import_instructions {
        settings.import_instructions = Some(value.clone());
    }
    if patch.shared.is_some() {
        settings.shared = patch.shared.clone();
    }
}

fn merge_settings_patch(target: &mut JsonMap, patch: &PartialMemorySettings) {
    let mut set = |key: &str, value: Option<Value>| {
        if let Some(value) = value {
            target.insert(key.to_string(), value);
        }
    };
    set("recall", patch.recall.map(Value::Bool));
    set("learning", patch.learning.map(Value::Bool));
    set(
        "recallQueryDistillation",
        patch.recall_query_distillation.map(Value::Bool),
    );
    set("recallRerank", patch.recall_rerank.map(Value::Bool));
    set(
        "maxRecallChars",
        patch.max_recall_chars.map(|value| Value::from(value)),
    );
    set(
        "maxRecallEntries",
        patch.max_recall_entries.map(|value| Value::from(value)),
    );
    set(
        "maxExtractionTokens",
        patch.max_extraction_tokens.map(|value| Value::from(value)),
    );
    set(
        "maxImportBytes",
        patch.max_import_bytes.map(|value| Value::from(value)),
    );
    set(
        "maxImportChunkChars",
        patch.max_import_chunk_chars.map(|value| Value::from(value)),
    );
    set(
        "maxImportChunksPerRun",
        patch
            .max_import_chunks_per_run
            .map(|value| Value::from(value)),
    );
    if let Some(instructions) = &patch.import_instructions {
        target.insert(
            "importInstructions".to_string(),
            Value::String(instructions.clone()),
        );
    }
    if let Some(shared) = &patch.shared {
        target.insert(
            "shared".to_string(),
            match shared {
                Some(config) => serde_json::to_value(config).unwrap_or(Value::Null),
                None => Value::Null,
            },
        );
    }
}

pub fn memory_source_from_value(value: &Value) -> Option<MemorySource> {
    let record = record(value).ok()?;
    let origin = record.get("origin").and_then(Value::as_str)?;
    let origin = match origin {
        "user" => MemoryOrigin::User,
        "assistant" => MemoryOrigin::Assistant,
        "tool" => MemoryOrigin::Tool,
        "derived" => MemoryOrigin::Derived,
        "file" => MemoryOrigin::File,
        _ => return None,
    };
    Some(MemorySource {
        id: record.get("id").and_then(Value::as_str)?.to_string(),
        origin,
        sha256: record.get("sha256").and_then(Value::as_str)?.to_string(),
        uri: record
            .get("uri")
            .and_then(Value::as_str)
            .map(str::to_string),
        revision: record
            .get("revision")
            .and_then(Value::as_str)
            .map(str::to_string),
        project_path: record
            .get("projectPath")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

pub fn path_join(base: &str, child: &str) -> String {
    PathBuf::from(base)
        .join(child)
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::refinement::refinement::normalize_refinement_proposal;

    fn fixture() -> (MemoryStore, PathBuf) {
        let root = std::env::temp_dir().join(format!("prime-store-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let cwd = root.join("repo");
        let agent_dir = root.join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        let project = ProjectIdentity {
            id: "project_test".to_string(),
            root: cwd.to_string_lossy().to_string(),
            aliases: Vec::new(),
        };
        let store = MemoryStore::new(&agent_dir.to_string_lossy(), project).unwrap();
        (store, root)
    }

    fn proposal(id: &str, content: &str) -> RefinementProposal {
        normalize_refinement_proposal(&serde_json::json!({
            "summary": id,
            "rationale": "test evidence",
            "expectedOutcome": "correct recall",
            "edits": [{"action": "create", "kind": "memory", "id": id, "title": id, "content": content}]
        }))
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn empty_document_matches_the_typescript_shape() {
        let document = empty_document("project_test");
        let value = serde_json::to_value(&document).unwrap();
        assert_eq!(value["schema"].as_f64(), Some(1.0));
        assert_eq!(value["memory"]["schema"].as_f64(), Some(1.0));
        assert_eq!(
            value["memory"]["projectId"],
            serde_json::json!("project_test")
        );
        assert_eq!(value["memory"]["revision"], serde_json::json!(0));
        assert_eq!(value["memory"]["history"], serde_json::json!([]));
        assert_eq!(value["memory"]["events"], serde_json::json!({}));
        assert_eq!(value["refinements"], serde_json::json!([]));
        let kinds: Vec<&String> = value["entries"].as_object().unwrap().keys().collect();
        assert_eq!(kinds, vec!["memory", "prompt", "skill", "subagent"]);
    }

    #[test]
    fn serializes_competing_revisions_and_deduplicates_retries() {
        let (store, root) = fixture();
        let runtime = runtime();
        let first = runtime
            .block_on(store.apply(
                &proposal("one", "Content one"),
                ApplyOptions {
                    event_id: "op_one".to_string(),
                    expected_revision: 0,
                    ..Default::default()
                },
            ))
            .expect("apply");
        assert_eq!(first.id, "op_one");
        // Same event id + same payload is a retry, not a conflict.
        let retried = runtime
            .block_on(store.apply(
                &proposal("one", "Content one"),
                ApplyOptions {
                    event_id: "op_one".to_string(),
                    expected_revision: 0,
                    ..Default::default()
                },
            ))
            .expect("retry");
        assert_eq!(retried.id, "op_one");
        let error = runtime
            .block_on(store.apply(
                &proposal("different", "Content different"),
                ApplyOptions {
                    event_id: "op_one".to_string(),
                    expected_revision: 1,
                    ..Default::default()
                },
            ))
            .expect_err("different content");
        assert!(error.contains("different content"));
        let duplicate = runtime
            .block_on(store.apply(
                &proposal("duplicate", "Content one"),
                ApplyOptions {
                    event_id: "duplicate".to_string(),
                    expected_revision: 1,
                    ..Default::default()
                },
            ))
            .expect_err("duplicate");
        assert!(duplicate.contains("duplicate"));
        let stale = runtime
            .block_on(store.apply(
                &proposal("stale", "x"),
                ApplyOptions {
                    event_id: "op_stale".to_string(),
                    expected_revision: 0,
                    ..Default::default()
                },
            ))
            .expect_err("stale revision");
        assert_eq!(
            stale,
            "Memory changed; review current revision before applying"
        );
        assert_eq!(store.read().unwrap().memory.revision, 1);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn fails_closed_on_corrupt_authoritative_state() {
        let (store, root) = fixture();
        std::fs::create_dir_all(&store.dir).unwrap();
        std::fs::write(&store.path, "{\"schema\": 2}").unwrap();
        let error = store.read().expect_err("corrupt state");
        assert_eq!(error, "Expected an object");
        let mut invalid = serde_json::to_value(empty_document(&store.project.id)).unwrap();
        invalid["schema"] = serde_json::json!(2);
        write_json(&store.path, &invalid).unwrap();
        assert_eq!(store.read().unwrap_err(), "Invalid project memory snapshot");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn backs_up_history_validates_restore_and_preserves_delivery_receipts() {
        let (store, root) = fixture();
        let runtime = runtime();
        runtime
            .block_on(store.apply(
                &proposal("one", "Content one"),
                ApplyOptions {
                    event_id: "op_one".to_string(),
                    expected_revision: 0,
                    ..Default::default()
                },
            ))
            .unwrap();
        let backup = store.backup(None).unwrap();
        assert!(backup.starts_with("backup_1_"));
        let revision = runtime.block_on(store.restore(&backup)).expect("restore");
        assert_eq!(revision, 2);
        let document = store.read().unwrap();
        assert_eq!(document.memory.revision, 2);
        assert_eq!(
            document.memory.events.get("op_one").map(String::as_str),
            Some(document.memory.events["op_one"].as_str())
        );
        let error = runtime
            .block_on(store.restore("nope"))
            .expect_err("invalid id");
        assert_eq!(error, "Invalid backup ID");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rolls_back_a_committed_refinement_and_restores_host_isolation() {
        let (store, root) = fixture();
        let runtime = runtime();
        let result = runtime
            .block_on(store.apply(
                &proposal("one", "Content one"),
                ApplyOptions {
                    event_id: "op_one".to_string(),
                    expected_revision: 0,
                    ..Default::default()
                },
            ))
            .unwrap();
        let rolled = runtime
            .block_on(store.rollback(&result.id, 1))
            .expect("rollback");
        assert_eq!(rolled.applied_edits[0].edit.action, "delete");
        let document = store.read().unwrap();
        assert!(document.entries.get("memory").unwrap().is_empty());
        assert_eq!(document.memory.revision, 2);
        let error = runtime
            .block_on(store.rollback("refine_missing", 2))
            .expect_err("unknown refinement");
        assert_eq!(error, "Unknown refinement ID");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn keeps_host_facts_private_and_remaps_sources() {
        let (store, root) = fixture();
        let runtime = runtime();
        let mut host_proposal = proposal("host_only", "Content host");
        host_proposal.edits[0].metadata = Some(
            serde_json::json!({"projectReusable": true})
                .as_object()
                .unwrap()
                .clone(),
        );
        runtime
            .block_on(store.apply(
                &host_proposal,
                ApplyOptions {
                    event_id: "op_host".to_string(),
                    expected_revision: 0,
                    host: true,
                    ..Default::default()
                },
            ))
            .unwrap();
        let document = store.read().unwrap();
        let entry = document
            .entries
            .get("memory")
            .unwrap()
            .get("host_only")
            .unwrap();
        assert_eq!(
            entry.metadata.get("hostId").and_then(Value::as_str),
            Some(store.host_id.as_str())
        );
        assert_eq!(
            entry.metadata.get("projectId").and_then(Value::as_str),
            Some("project_test")
        );
        assert_eq!(
            entry.metadata.get("status").and_then(Value::as_str),
            Some("current")
        );
        assert_eq!(entry.metadata.get("sources"), Some(&serde_json::json!([])));
        // Another host cannot see or modify it.
        let other_root = root.join("other-agent");
        std::fs::create_dir_all(&other_root).unwrap();
        let other = MemoryStore::new(
            &other_root.to_string_lossy(),
            ProjectIdentity {
                id: "project_test".to_string(),
                root: store.project.root.clone(),
                aliases: Vec::new(),
            },
        )
        .unwrap();
        // A second host reading the same authoritative project snapshot must
        // reject its host-scoped entry, rather than testing an empty store.
        std::fs::create_dir_all(&other.dir).unwrap();
        std::fs::copy(&store.path, &other.path).unwrap();
        let mut update = proposal("host_only", "Content host changed");
        update.edits[0].action = "update".to_string();
        let error = runtime
            .block_on(other.apply(
                &update,
                ApplyOptions {
                    event_id: "op_update".to_string(),
                    expected_revision: 1,
                    ..Default::default()
                },
            ))
            .expect_err("other host");
        assert_eq!(error, "Memory belongs to another host");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn validates_settings_limits_and_shared_endpoints() {
        assert_eq!(
            validate_settings(&serde_json::json!({})).unwrap().recall,
            None
        );
        assert_eq!(
            validate_settings(&serde_json::json!({"recall": 1})).unwrap_err(),
            "recall must be boolean"
        );
        assert_eq!(
            validate_settings(&serde_json::json!({"maxRecallChars": 32001})).unwrap_err(),
            "Invalid maxRecallChars"
        );
        assert_eq!(
            validate_settings(&serde_json::json!({"maxImportChunksPerRun": 0})).unwrap_err(),
            "Invalid maxImportChunksPerRun"
        );
        assert_eq!(
            validate_settings(&serde_json::json!({"shared": {}})).unwrap_err(),
            "Sharing requires url and tokenFile"
        );
        assert_eq!(
            validate_settings(
                &serde_json::json!({"shared": {"url": "http://example.com", "tokenFile": "t"}})
            )
            .unwrap_err(),
            "Use HTTPS or a loopback SSH tunnel for memory sharing"
        );
        let loopback = validate_settings(
            &serde_json::json!({"shared": {"url": "http://127.0.0.1:8799/", "tokenFile": "t"}}),
        )
        .unwrap();
        assert_eq!(
            loopback.shared.unwrap().unwrap(),
            SharedConfig {
                url: "http://127.0.0.1:8799".to_string(),
                token_file: "t".to_string()
            }
        );
        let cleared = validate_settings(&serde_json::json!({"shared": null})).unwrap();
        assert_eq!(cleared.shared, Some(None));
    }

    #[test]
    fn reads_settings_from_the_global_and_local_files() {
        let (store, root) = fixture();
        let runtime = runtime();
        assert_eq!(store.settings().max_recall_chars, 6000);
        std::fs::write(
            Path::new(&store.agent_dir).join("settings.json"),
            "{\"memory\": {\"maxRecallChars\": 7000, \"learning\": false}}",
        )
        .unwrap();
        let settings = store.settings();
        assert_eq!(settings.max_recall_chars, 7000);
        assert!(!settings.learning);
        runtime
            .block_on(store.configure(&serde_json::json!({"maxRecallEntries": 3})))
            .unwrap();
        let settings = store.settings();
        assert_eq!(settings.max_recall_entries, 3);
        assert_eq!(settings.max_recall_chars, 7000);
        let error = runtime
            .block_on(store.configure(&serde_json::json!({"maxRecallEntries": 100})))
            .expect_err("invalid limit");
        assert_eq!(error, "Invalid maxRecallEntries");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn refuses_paused_automatic_learning() {
        let (store, root) = fixture();
        let runtime = runtime();
        std::fs::write(
            Path::new(&store.agent_dir).join("settings.json"),
            "{\"memory\": {\"learning\": false}}",
        )
        .unwrap();
        let error = runtime
            .block_on(store.apply(
                &proposal("auto", "Content auto"),
                ApplyOptions {
                    event_id: "op_auto".to_string(),
                    expected_revision: 0,
                    automatic: true,
                    ..Default::default()
                },
            ))
            .expect_err("paused learning");
        assert_eq!(error, "Automatic learning is paused");
        std::fs::remove_dir_all(&root).ok();
    }
}
