//! Port of packages/coding-agent/src/core/memory/jobs.ts
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::refinement::refinement::RefinementProposal;
use crate::utils::atomic_file::{write_file_atomic_sync, WriteFileAtomicOptions};

use super::evidence::{hash, js_len, message_evidence, serialize_evidence, AgentMessage, Evidence};
use super::store::{read_json, record, write_json, MemoryStore};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportChunk {
    pub records: Vec<Evidence>,
    #[serde(rename = "startLine")]
    pub start_line: i64,
    #[serde(rename = "endLine")]
    pub end_line: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal: Option<RefinementProposal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImportJobStatus {
    Pending,
    Running,
    Preview,
    Applied,
    Failed,
}

impl ImportJobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ImportJobStatus::Pending => "pending",
            ImportJobStatus::Running => "running",
            ImportJobStatus::Preview => "preview",
            ImportJobStatus::Applied => "applied",
            ImportJobStatus::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportCoverage {
    pub lines: i64,
    #[serde(rename = "evidenceRecords")]
    pub evidence_records: i64,
    #[serde(rename = "excludedLines")]
    pub excluded_lines: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportUsage {
    pub input: f64,
    pub output: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportJob {
    pub schema: f64,
    pub id: String,
    pub source: String,
    #[serde(rename = "sourceHash")]
    pub source_hash: String,
    pub status: ImportJobStatus,
    pub chunks: Vec<ImportChunk>,
    #[serde(rename = "nextChunk")]
    pub next_chunk: i64,
    #[serde(rename = "expectedRevision")]
    pub expected_revision: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub coverage: ImportCoverage,
    pub usage: ImportUsage,
    #[serde(skip_serializing_if = "Option::is_none", rename = "acceptedProposal")]
    pub accepted_proposal: Option<RefinementProposal>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractionResult {
    pub proposal: RefinementProposal,
    pub input: f64,
    pub output: f64,
}

/// `(records: Evidence[]) => Promise<{ proposal, input, output }>`
pub type MemoryExtractor = Arc<
    dyn Fn(Vec<Evidence>) -> Pin<Box<dyn Future<Output = Result<ExtractionResult, String>> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct MemoryJobs {
    pub dir: String,
    pub store: MemoryStore,
}

impl MemoryJobs {
    pub fn new(store: MemoryStore) -> MemoryJobs {
        let dir = Path::new(&store.dir)
            .join("jobs")
            .to_string_lossy()
            .to_string();
        MemoryJobs { dir, store }
    }

    fn path(&self, id: &str) -> Result<String, String> {
        if !regex::Regex::new(r"^import_[a-f0-9]{24}$")
            .unwrap()
            .is_match(id)
        {
            return Err("Invalid import ID".to_string());
        }
        Ok(Path::new(&self.dir)
            .join(format!("{id}.json"))
            .to_string_lossy()
            .to_string())
    }

    pub fn get(&self, id: &str) -> Result<ImportJob, String> {
        let path = self.path(id)?;
        let value = read_json(&path)?;
        serde_json::from_value(value).map_err(|error| error.to_string())
    }

    pub fn list(&self) -> Vec<ImportJob> {
        if !Path::new(&self.dir).exists() {
            return Vec::new();
        }
        let names = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(_) => return Vec::new(),
        };
        let name_re = regex::Regex::new(r"^import_[a-f0-9]{24}\.json$").unwrap();
        let mut jobs = Vec::new();
        for entry in names.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name_re.is_match(&name) {
                continue;
            }
            let id = &name[..name.len() - 5];
            if let Ok(job) = self.get(id) {
                jobs.push(job);
            }
        }
        jobs
    }

    pub async fn prepare(&self, source: &str) -> Result<ImportJob, String> {
        let settings = self.store.settings();
        let metadata = std::fs::metadata(source).map_err(|_| {
            "Selected session exceeds import size limit or is not a file".to_string()
        })?;
        if !metadata.is_file() || metadata.len() as i64 > settings.max_import_bytes {
            return Err("Selected session exceeds import size limit or is not a file".to_string());
        }
        let raw = std::fs::read_to_string(source).map_err(|_| {
            "Selected session exceeds import size limit or is not a file".to_string()
        })?;
        let source_hash = hash(&raw);
        let digest = hash(&format!("{}:{source_hash}", self.store.project.id));
        let id = format!("import_{}", &digest[..24.min(digest.len())]);
        let store = self.store.clone();
        let source = source.to_string();
        let raw_for_job = raw.clone();
        let id_for_job = id.clone();
        self.store
            .exclusive(move || {
                std::fs::create_dir_all(&self.dir).map_err(|error| error.to_string())?;
                if Path::new(&self.path(&id_for_job)?).exists() {
                    return self.get(&id_for_job);
                }
                let input_path = Path::new(&self.dir)
                    .join(format!("{id_for_job}.source.jsonl"))
                    .to_string_lossy()
                    .to_string();
                // Preserve the byte-for-byte selected transcript before publishing pending work.
                write_file_atomic_sync(
                    &input_path,
                    &raw_for_job,
                    WriteFileAtomicOptions {
                        mode: Some(0o600),
                        fsync: true,
                        fsync_dir: false,
                        ..Default::default()
                    },
                )
                .map_err(|error| error.to_string())?;
                let lines: Vec<&str> = raw_for_job.split('\n').collect();
                let mut chunks: Vec<ImportChunk> = Vec::new();
                let mut excluded_lines: Vec<i64> = Vec::new();
                let mut evidence_records: i64 = 0;
                for (index, line) in lines.iter().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let row: Value = serde_json::from_str(line).map_err(|_| {
                        format!(
                            "Invalid session JSON at line {}; preserved input at {input_path}",
                            index + 1
                        )
                    })?;
                    let row = record(&row)?;
                    let message = if row.get("type").and_then(Value::as_str) == Some("message") {
                        row.get("message").cloned()
                    } else {
                        None
                    };
                    let evidence = match message {
                        Some(message) => {
                            let message: AgentMessage = serde_json::from_value(message)
                                .ok()
                                .unwrap_or(AgentMessage::Other);
                            let uri = format!(
                                "{}#L{}",
                                url::Url::from_file_path(Path::new(&input_path))
                                    .map(|url| url.to_string())
                                    .unwrap_or_else(|_| format!(
                                        "file://{}",
                                        input_path.replace('\\', "/")
                                    )),
                                index + 1
                            );
                            let entry_id =
                                row.get("id").and_then(Value::as_str).map(str::to_string);
                            message_evidence(&message, Some(&uri), entry_id.as_deref())
                        }
                        None => None,
                    };
                    let evidence = match evidence {
                        Some(evidence)
                            if evidence.origin != super::evidence::MemoryOrigin::Derived =>
                        {
                            evidence
                        }
                        _ => {
                            excluded_lines.push((index + 1) as i64);
                            continue;
                        }
                    };
                    evidence_records += 1;
                    // Oversized code/logs are split losslessly; source ID and character coverage remain explicit.
                    let chunk_budget = settings.max_import_chunk_chars.min(75000) as usize;
                    let reference = Evidence {
                        text: String::new(),
                        ..evidence.clone()
                    };
                    let text_budget = chunk_budget as i64
                        - js_len(&serialize_evidence(&[reference], usize::MAX / 4)) as i64
                        - 64;
                    if text_budget < 1 {
                        return Err(
                            "Source reference exceeds chunk budget; increase maxImportChunkChars"
                                .to_string(),
                        );
                    }
                    let text_budget = text_budget as usize;
                    let mut characters = evidence.text.chars().peekable();
                    let mut offset = 0usize;
                    while characters.peek().is_some() {
                        let mut part_text = String::new();
                        let mut part_units = 0usize;
                        while let Some(&character) = characters.peek() {
                            if part_units + character.len_utf16() > text_budget {
                                break;
                            }
                            part_text.push(character);
                            part_units += character.len_utf16();
                            characters.next();
                        }
                        if part_units == 0 {
                            return Err("Source reference exceeds chunk budget; increase maxImportChunkChars".to_string());
                        }
                        let part = Evidence {
                            id: format!("{}:{offset}", evidence.id),
                            text: part_text,
                            ..evidence.clone()
                        };
                        if js_len(&serialize_evidence(std::slice::from_ref(&part), usize::MAX / 4)) > chunk_budget {
                            return Err("Source reference exceeds chunk budget; increase maxImportChunkChars".to_string());
                        }
                        let fits = match chunks.last() {
                            Some(previous) => {
                                let mut combined = previous.records.clone();
                                combined.push(part.clone());
                                js_len(&serialize_evidence(&combined, usize::MAX / 4))
                                    <= chunk_budget
                            }
                            None => false,
                        };
                        if fits {
                            let previous = chunks.last_mut().unwrap();
                            previous.records.push(part);
                            previous.end_line = (index + 1) as i64;
                        } else {
                            chunks.push(ImportChunk {
                                records: vec![part],
                                start_line: (index + 1) as i64,
                                end_line: (index + 1) as i64,
                                proposal: None,
                            });
                        }
                        offset += part_units;
                    }
                }
                let job = ImportJob {
                    schema: 1.0,
                    id: id_for_job.clone(),
                    source: source.clone(),
                    source_hash: source_hash.clone(),
                    status: ImportJobStatus::Pending,
                    chunks,
                    next_chunk: 0,
                    expected_revision: store.read()?.memory.revision,
                    error: None,
                    coverage: ImportCoverage {
                        lines: lines.len() as i64,
                        evidence_records,
                        excluded_lines,
                    },
                    usage: ImportUsage {
                        input: 0.0,
                        output: 0.0,
                    },
                    accepted_proposal: None,
                };
                write_json(
                    &self.path(&id_for_job)?,
                    &serde_json::to_value(&job).map_err(|error| error.to_string())?,
                )?;
                Ok(job)
            })
            .await
    }

    pub async fn run(
        &self,
        id: &str,
        extract: MemoryExtractor,
        signal_aborted: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    ) -> Result<ImportJob, String> {
        std::fs::create_dir_all(&self.dir).map_err(|error| error.to_string())?;
        // One extraction worker per project across processes, including restart recovery.
        let _lock = super::acquire_lock(
            &self.dir,
            super::MemoryLockRetries {
                stale_ms: 30_000,
                retries: 0,
                min_timeout_ms: 0,
                max_timeout_ms: 0,
            },
        )
        .await?;
        let mut job = match self.get(id) {
            Ok(job) => job,
            Err(error) => return Err(error),
        };
        if matches!(
            job.status,
            ImportJobStatus::Applied | ImportJobStatus::Preview
        ) {
            return Ok(job);
        }
        let outcome = self.run_inner(id, &mut job, extract, signal_aborted).await;
        match outcome {
            Ok(()) => Ok(job),
            Err(error) => {
                let mut failed = job.clone();
                failed.status = ImportJobStatus::Failed;
                failed.error = Some(error.clone());
                let _ = write_json(
                    &self.path(id).unwrap_or_default(),
                    &serde_json::to_value(&failed).unwrap_or(Value::Null),
                );
                Err(error)
            }
        }
    }

    async fn run_inner(
        &self,
        id: &str,
        job: &mut ImportJob,
        extract: MemoryExtractor,
        signal_aborted: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    ) -> Result<(), String> {
        job.status = ImportJobStatus::Running;
        job.error = None;
        write_json(
            &self.path(id)?,
            &serde_json::to_value(&*job).map_err(|error| error.to_string())?,
        )?;
        let mut processed = 0i64;
        let max_per_run = self.store.settings().max_import_chunks_per_run;
        while (job.next_chunk as usize) < job.chunks.len() && processed < max_per_run {
            if let Some(aborted) = &signal_aborted {
                if aborted() {
                    return Err("The operation was aborted".to_string());
                }
            }
            let records = job.chunks[job.next_chunk as usize].records.clone();
            let result = extract(records).await?;
            job.chunks[job.next_chunk as usize].proposal = Some(result.proposal);
            job.usage.input += result.input;
            job.usage.output += result.output;
            job.next_chunk += 1;
            processed += 1;
            write_json(
                &self.path(id)?,
                &serde_json::to_value(&*job).map_err(|error| error.to_string())?,
            )?;
        }
        job.status = if job.next_chunk as usize == job.chunks.len() {
            ImportJobStatus::Preview
        } else {
            ImportJobStatus::Pending
        };
        write_json(
            &self.path(id)?,
            &serde_json::to_value(&*job).map_err(|error| error.to_string())?,
        )?;
        Ok(())
    }

    pub async fn apply(&self, id: &str, expected_revision: i64) -> Result<ImportJob, String> {
        let _lock = super::acquire_lock(
            &self.dir,
            super::MemoryLockRetries {
                stale_ms: 30_000,
                retries: 0,
                min_timeout_ms: 0,
                max_timeout_ms: 0,
            },
        )
        .await?;
        let mut job = self.get(id)?;
        if job.status == ImportJobStatus::Applied {
            return Ok(job);
        }
        if job.status != ImportJobStatus::Preview {
            return Err(
                "Import must finish extraction and be previewed before applying".to_string(),
            );
        }
        if job.accepted_proposal.is_none() {
            let edits: Vec<crate::core::refinement::refinement::RefinementEdit> = job
                .chunks
                .iter()
                .flat_map(|chunk| {
                    chunk
                        .proposal
                        .as_ref()
                        .map(|proposal| proposal.edits.clone())
                        .unwrap_or_default()
                })
                .collect();
            if edits
                .iter()
                .any(|edit| edit.action != "create" || edit.kind != "memory")
            {
                return Err("Imports only create memories".to_string());
            }
            let mut unique: Vec<crate::core::refinement::refinement::RefinementEdit> = Vec::new();
            let mut seen: Vec<String> = Vec::new();
            for edit in edits {
                let key = hash(&format!(
                    "{}:{}",
                    edit.kind,
                    edit.content.clone().unwrap_or_default().trim()
                ));
                if seen.contains(&key) {
                    continue;
                }
                // Chunks are extracted independently, so the model can re-emit
                // one id with different content across chunks of the same
                // session. Only one create per id can apply; a repeat would
                // fail the whole import with "entry already exists". Keep the
                // last extraction so the most recent chunk summary wins.
                if edit.action == "create" {
                    if let Some(id) = edit.id.as_deref() {
                        unique.retain(|kept| {
                            !(kept.action == "create" && kept.id.as_deref() == Some(id))
                        });
                    }
                }
                let mut edit = edit;
                // The extractor sometimes copies a display projectId into edit
                // metadata; the store rejects foreign project ids outright
                // ("Memory belongs to another project"), aborting the whole
                // import. Drop the field and let the store stamp its own.
                if let Some(metadata) = edit.metadata.as_mut() {
                    metadata.shift_remove("projectId");
                }
                if edit.action == "create" {
                    let invalid = edit
                        .id
                        .as_deref()
                        .map(|id| {
                            id.is_empty()
                                || id.len() > 160
                                || ["__proto__", "constructor", "prototype"].contains(&id)
                                || !id
                                    .chars()
                                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                        })
                        .unwrap_or(true);
                    if invalid {
                        // The store rejects invalid ids outright ("Invalid memory
                        // ID"), which would abort the whole import. Models
                        // sometimes emit an empty id on a create; derive a stable
                        // one from the edit content instead of failing.
                        let content_key = hash(&format!(
                            "{}:{}",
                            edit.kind,
                            edit.content.clone().unwrap_or_default().trim()
                        ));
                        edit.id = Some(format!("imported_{:.48}", &content_key[..24.min(content_key.len())]));
                    }
                }
                seen.push(key);
                unique.push(edit);
            }
            let current = self.store.read()?;
            let new_edits: Vec<crate::core::refinement::refinement::RefinementEdit> = unique
                .into_iter()
                .filter(|edit| {
                    // Idempotent skip: an existing entry with the same id makes the create
                    // edit a no-op (apply_refinement_proposal would mark it not-applied and
                    // fail the whole import). The id-space is model-chosen, so collisions
                    // across import jobs are legitimate, not conflicts.
                    if let Some(id) = edit.id.as_deref() {
                        if current
                            .entries
                            .get("memory")
                            .map(|bucket| bucket.contains_key(id))
                            .unwrap_or(false)
                        {
                            return false;
                        }
                    }
                    let content = edit.content.clone().unwrap_or_default().trim().to_string();
                    !current
                        .entries
                        .get("memory")
                        .map(|bucket| {
                            bucket.values().any(|entry| {
                                entry.content.trim() == content
                                    && !entry.metadata.contains_key("supersededBy")
                            })
                        })
                        .unwrap_or(false)
                })
                .collect();
            job.accepted_proposal = Some(RefinementProposal {
                summary: format!("Import {id}"),
                rationale: format!("Selected session {}", job.source_hash),
                expected_outcome: "Recover selected project knowledge with original evidence"
                    .to_string(),
                edits: new_edits,
            });
            // Persist the exact accepted operation before applying it; crash retries use the same fingerprint.
            write_json(
                &self.path(id)?,
                &serde_json::to_value(&job).map_err(|error| error.to_string())?,
            )?;
        }
        let proposal = job.accepted_proposal.clone().unwrap();
        self.store
            .apply(
                &proposal,
                super::store::ApplyOptions {
                    event_id: id.to_string(),
                    expected_revision,
                    sources: None,
                    host: false,
                    automatic: false,
                    replace_metadata: false,
                },
            )
            .await?;
        job.status = ImportJobStatus::Applied;
        write_json(
            &self.path(id)?,
            &serde_json::to_value(&job).map_err(|error| error.to_string())?,
        )?;
        Ok(job)
    }
}

pub fn import_overview(job: &ImportJob) -> Value {
    let chunks: Vec<Value> = job
        .chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| {
            serde_json::json!({
                "startLine": chunk.start_line,
                "endLine": chunk.end_line,
                "proposal": chunk.proposal,
                "index": index,
                "sourceIds": chunk.records.iter().map(|record| record.id.clone()).collect::<Vec<_>>(),
            })
        })
        .collect();
    let mut value = serde_json::to_value(job).unwrap_or(Value::Null);
    if let Some(map) = value.as_object_mut() {
        map.insert("chunks".to_string(), Value::Array(chunks));
    }
    value
}

pub fn job_path(dir: &str, id: &str) -> PathBuf {
    Path::new(dir).join(format!("{id}.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory::project::ProjectIdentity;
    use crate::core::refinement::refinement::normalize_refinement_proposal;

    fn fixture() -> (MemoryJobs, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("prime-jobs-{}", uuid::Uuid::new_v4()));
        let cwd = root.join("repo");
        let agent_dir = root.join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        let store = MemoryStore::new(
            &agent_dir.to_string_lossy(),
            ProjectIdentity {
                id: "project_test".to_string(),
                root: cwd.to_string_lossy().to_string(),
                aliases: Vec::new(),
            },
        )
        .unwrap();
        (MemoryJobs::new(store), root)
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn session_file(root: &Path) -> String {
        let path = root.join("session.jsonl");
        let lines = [
            serde_json::json!({"type": "message", "id": "row_1", "message": {"role": "user", "content": "Use Sydney", "timestamp": 1}}),
            serde_json::json!({"type": "message", "id": "row_2", "message": {"role": "custom", "customType": "harness_digest", "content": "synthetic", "display": false, "timestamp": 2}}),
            serde_json::json!({"type": "message", "id": "row_3", "message": {"role": "toolResult", "toolCallId": "c1", "toolName": "ipython", "content": [{"type": "text", "text": "print(1)\n".repeat(40)}], "isError": false, "timestamp": 3}}),
            serde_json::json!({"type": "message", "id": "row_4", "message": {"role": "compactionSummary", "summary": "derived", "tokensBefore": 1, "timestamp": 4}}),
        ];
        let body = lines
            .iter()
            .map(|line| serde_json::to_string(line).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, body).unwrap();
        path.to_string_lossy().to_string()
    }

    fn extractor() -> MemoryExtractor {
        Arc::new(|records: Vec<Evidence>| {
            Box::pin(async move {
                Ok(ExtractionResult {
                    proposal: normalize_refinement_proposal(&serde_json::json!({
                        "summary": "import",
                        "rationale": "r",
                        "expectedOutcome": "o",
                        "edits": [{"action": "create", "kind": "memory", "id": format!("imported_{}", records[0].id.replace(':', "_")), "title": "Imported", "content": records[0].text}]
                    })),
                    input: 10.0,
                    output: 5.0,
                })
            })
        })
    }

    #[test]
    fn import_chunks_respect_utf16_budget_and_preserve_all_text_and_offsets() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        runtime.block_on(jobs.store.configure(&serde_json::json!({"maxImportChunkChars": 4096}))).unwrap();
        for (index, text) in ["a".repeat(9000), "\u{1f600}".repeat(9000), "\u{6f22}\u{5b57}e\u{301}\u{1f680}".repeat(2000)].into_iter().enumerate() {
            let source = root.join(format!("unicode-{index}.jsonl"));
            std::fs::write(&source, serde_json::json!({
                "type": "message", "id": "source", "message": {"role": "user", "content": text, "timestamp": 1}
            }).to_string()).unwrap();
            let job = runtime.block_on(jobs.prepare(&source.to_string_lossy())).unwrap();
            assert!(job.chunks.len() > 1);
            let mut joined = String::new();
            let mut offset = 0usize;
            for chunk in &job.chunks {
                assert!(js_len(&serialize_evidence(&chunk.records, usize::MAX / 4)) <= 4096);
                for record in &chunk.records {
                    assert_eq!(record.id, format!("source:{offset}"));
                    assert!(!record.text.is_empty());
                    assert_eq!(record.sha256, hash(&text));
                    offset += js_len(&record.text);
                    joined.push_str(&record.text);
                }
            }
            assert_eq!(joined, text);
            assert_eq!(offset, js_len(&text));
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn import_rejects_reference_that_leaves_no_unicode_character_room() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        runtime.block_on(jobs.store.configure(&serde_json::json!({"maxImportChunkChars": 4096}))).unwrap();
        let source = root.join("oversized-reference.jsonl");
        std::fs::write(&source, serde_json::json!({
            "type": "message", "id": "x".repeat(90_000),
            "message": {"role": "user", "content": "\u{1f600}", "timestamp": 1}
        }).to_string()).unwrap();
        let error = runtime.block_on(jobs.prepare(&source.to_string_lossy())).unwrap_err();
        assert!(error.contains("Source reference exceeds chunk budget"));
        assert!(jobs.list().is_empty());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn long_import_remains_exclusive_past_the_old_thirty_second_lease() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        runtime.block_on(async {
            let job = jobs.prepare(&session_file(&root)).await.unwrap();
            let entered = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let started = entered.clone();
            let finish = release.clone();
            let extract: MemoryExtractor = Arc::new(move |records| {
                let started = started.clone();
                let finish = finish.clone();
                Box::pin(async move {
                    started.notify_one();
                    finish.notified().await;
                    extractor()(records).await
                })
            });
            let first_jobs = jobs.clone();
            let id = job.id.clone();
            let first = tokio::spawn(async move { first_jobs.run(&id, extract, None).await });
            tokio::time::timeout(std::time::Duration::from_secs(5), entered.notified()).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_secs(32)).await;
            for _ in 0..2 {
                let rejected = jobs.run(&job.id, extractor(), None).await.unwrap_err();
                assert!(rejected.starts_with("Lock file is already being held:"));
            }
            assert_eq!(jobs.get(&job.id).unwrap().status, ImportJobStatus::Running);
            release.notify_one();
            let complete = tokio::time::timeout(std::time::Duration::from_secs(5), first).await.unwrap().unwrap().unwrap();
            assert_eq!(complete.status, ImportJobStatus::Preview);
            assert_eq!(complete.usage.input, 10.0);
            assert_eq!(jobs.run(&job.id, extractor(), None).await.unwrap().usage.input, 10.0);
        });
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn preserves_raw_code_and_reports_coverage() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let source = session_file(&root);
        let job = runtime.block_on(jobs.prepare(&source)).expect("prepare");
        assert_eq!(job.schema, 1.0);
        assert_eq!(job.status, ImportJobStatus::Pending);
        assert!(job.id.starts_with("import_"));
        assert_eq!(job.id.len(), "import_".len() + 24);
        assert_eq!(job.coverage.lines, 4);
        assert_eq!(job.coverage.evidence_records, 2);
        // Synthetic custom messages and derived summaries are excluded by line number.
        assert_eq!(job.coverage.excluded_lines, vec![2, 4]);
        let all_text: String = job
            .chunks
            .iter()
            .flat_map(|chunk| chunk.records.iter())
            .map(|record| record.text.clone())
            .collect::<Vec<_>>()
            .join("");
        assert!(all_text.contains("print(1)"));
        assert!(!all_text.contains("synthetic"));
        // The byte-for-byte transcript is preserved next to the job.
        let input_path = Path::new(&jobs.dir).join(format!("{}.source.jsonl", job.id));
        assert_eq!(
            std::fs::read_to_string(&input_path).unwrap(),
            std::fs::read_to_string(&source).unwrap()
        );
        assert_eq!(jobs.list().len(), 1);
        assert_eq!(jobs.get(&job.id).unwrap().id, job.id);
        assert_eq!(jobs.get("nope").unwrap_err(), "Invalid import ID");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rejects_invalid_session_json_and_oversized_files() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let path = root.join("bad.jsonl");
        std::fs::write(&path, "{\"type\": \"message\"}\nnot json\n").unwrap();
        let error = runtime
            .block_on(jobs.prepare(&path.to_string_lossy()))
            .expect_err("invalid json");
        assert!(error.starts_with("Invalid session JSON at line 2; preserved input at "));
        let missing = runtime
            .block_on(jobs.prepare(&root.join("missing.jsonl").to_string_lossy()))
            .expect_err("missing");
        assert_eq!(
            missing,
            "Selected session exceeds import size limit or is not a file"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resumes_failed_chunks_and_requires_a_completed_preview() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let source = session_file(&root);
        let job = runtime.block_on(jobs.prepare(&source)).unwrap();
        let failing: MemoryExtractor =
            Arc::new(|_| Box::pin(async { Err("extractor down".to_string()) }));
        let error = runtime
            .block_on(jobs.run(&job.id, failing, None))
            .expect_err("failed");
        assert_eq!(error, "extractor down");
        let failed = jobs.get(&job.id).unwrap();
        assert_eq!(failed.status, ImportJobStatus::Failed);
        assert_eq!(failed.error.as_deref(), Some("extractor down"));
        let apply_error = runtime
            .block_on(jobs.apply(&job.id, 0))
            .expect_err("not previewed");
        assert_eq!(
            apply_error,
            "Import must finish extraction and be previewed before applying"
        );
        let done = runtime
            .block_on(jobs.run(&job.id, extractor(), None))
            .expect("run");
        assert_eq!(done.status, ImportJobStatus::Preview);
        assert_eq!(done.next_chunk as usize, done.chunks.len());
        assert_eq!(done.usage.input, 10.0);
        assert_eq!(done.usage.output, 5.0);
        // A second run is a no-op once previewed.
        let again = runtime
            .block_on(jobs.run(&job.id, extractor(), None))
            .unwrap();
        assert_eq!(again.usage.input, 10.0);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn limits_extraction_calls_per_invocation() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let mut lines = Vec::new();
        for index in 0..8 {
            lines.push(
                serde_json::to_string(&serde_json::json!({
                    "type": "message", "id": format!("row_{index}"),
                    "message": {"role": "user", "content": format!("note {index} {}", "x".repeat(40_000)), "timestamp": index}
                }))
                .unwrap(),
            );
        }
        let path = root.join("big.jsonl");
        std::fs::write(&path, lines.join("\n")).unwrap();
        let job = runtime
            .block_on(jobs.prepare(&path.to_string_lossy()))
            .unwrap();
        assert!(job.chunks.len() > 4);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_handle = calls.clone();
        let extract: MemoryExtractor = Arc::new(move |_| {
            let calls = calls_handle.clone();
            Box::pin(async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(ExtractionResult {
                    proposal: normalize_refinement_proposal(&serde_json::json!({
                        "summary": "s", "rationale": "r", "expectedOutcome": "o", "edits": []
                    })),
                    input: 1.0,
                    output: 1.0,
                })
            })
        });
        let first = runtime
            .block_on(jobs.run(&job.id, extract.clone(), None))
            .unwrap();
        assert_eq!(first.status, ImportJobStatus::Pending);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 4);
        let mut completed = first;
        while completed.status == ImportJobStatus::Pending {
            let previous = completed.next_chunk as usize;
            completed = runtime.block_on(jobs.run(&job.id, extract.clone(), None)).unwrap();
            assert_eq!(completed.next_chunk as usize, (previous + 4).min(job.chunks.len()));
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), completed.next_chunk as usize);
        }
        assert_eq!(completed.status, ImportJobStatus::Preview);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), job.chunks.len());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn applies_only_creates_and_skips_existing_content() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let source = session_file(&root);
        let job = runtime.block_on(jobs.prepare(&source)).unwrap();
        runtime
            .block_on(jobs.run(&job.id, extractor(), None))
            .unwrap();
        let applied = runtime.block_on(jobs.apply(&job.id, 0)).expect("apply");
        assert_eq!(applied.status, ImportJobStatus::Applied);
        assert!(applied.accepted_proposal.is_some());
        assert!(jobs.store.read().unwrap().memory.revision == 1);
        // Re-applying an applied job is a no-op.
        let again = runtime.block_on(jobs.apply(&job.id, 1)).unwrap();
        assert_eq!(again.status, ImportJobStatus::Applied);
        let overview = import_overview(&applied);
        assert!(overview["chunks"][0]["index"] == serde_json::json!(0));
        assert!(overview["chunks"][0]["sourceIds"].as_array().unwrap().len() >= 1);
        assert!(overview["chunks"][0].get("records").is_none());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn repeated_create_ids_across_chunks_keep_the_last_extraction() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let mut lines = Vec::new();
        for index in 0..4 {
            lines.push(
                serde_json::to_string(&serde_json::json!({
                    "type": "message", "id": format!("row_{index}"),
                    "message": {"role": "user", "content": format!("note {index} {}", "x".repeat(40_000)), "timestamp": index}
                }))
                .unwrap(),
            );
        }
        let path = root.join("repeated.jsonl");
        std::fs::write(&path, lines.join("\n")).unwrap();
        let job = runtime
            .block_on(jobs.prepare(&path.to_string_lossy()))
            .unwrap();
        assert!(job.chunks.len() > 1, "expected multiple chunks");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_handle = calls.clone();
        let extract: MemoryExtractor = Arc::new(move |_| {
            let calls = calls_handle.clone();
            Box::pin(async move {
                let index = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(ExtractionResult {
                    proposal: normalize_refinement_proposal(&serde_json::json!({
                        "summary": "s", "rationale": "r", "expectedOutcome": "o",
                        "edits": [{"action": "create", "kind": "memory", "id": "shared_topic", "title": "Shared", "content": format!("extraction_{index}")}]
                    })),
                    input: 1.0,
                    output: 1.0,
                })
            })
        });
        let mut completed = runtime
            .block_on(jobs.run(&job.id, extract.clone(), None))
            .unwrap();
        while completed.status == ImportJobStatus::Pending {
            completed = runtime
                .block_on(jobs.run(&job.id, extract.clone(), None))
                .unwrap();
        }
        assert_eq!(completed.status, ImportJobStatus::Preview);
        let total = calls.load(std::sync::atomic::Ordering::SeqCst);
        assert!(total > 1, "expected multiple chunk extractions");
        let applied = runtime.block_on(jobs.apply(&job.id, 0)).expect("apply");
        assert_eq!(applied.status, ImportJobStatus::Applied);
        let accepted = applied.accepted_proposal.expect("accepted proposal");
        let shared: Vec<_> = accepted
            .edits
            .iter()
            .filter(|edit| edit.id.as_deref() == Some("shared_topic"))
            .collect();
        assert_eq!(shared.len(), 1, "one create per id");
        let store = jobs.store.read().unwrap();
        let bucket = store.entries.get("memory").expect("memory bucket");
        let entry = bucket.get("shared_topic").expect("stored entry");
        assert_eq!(entry.content, shared[0].content.clone().unwrap_or_default());
        assert_eq!(
            entry.content,
            format!("extraction_{}", total - 1),
            "last chunk extraction wins"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rejects_imports_that_are_not_memory_creates() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let source = session_file(&root);
        let job = runtime.block_on(jobs.prepare(&source)).unwrap();
        let bad: MemoryExtractor = Arc::new(|_| {
            Box::pin(async {
                Ok(ExtractionResult {
                    proposal: normalize_refinement_proposal(&serde_json::json!({
                        "summary": "s", "rationale": "r", "expectedOutcome": "o",
                        "edits": [{"action": "update", "kind": "memory", "id": "x", "title": "x", "content": "c"}]
                    })),
                    input: 1.0,
                    output: 1.0,
                })
            })
        });
        runtime.block_on(jobs.run(&job.id, bad, None)).unwrap();
        let error = runtime
            .block_on(jobs.apply(&job.id, 0))
            .expect_err("only creates");
        assert_eq!(error, "Imports only create memories");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn reports_aborted_extraction() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let source = session_file(&root);
        let job = runtime.block_on(jobs.prepare(&source)).unwrap();
        let error = runtime
            .block_on(jobs.run(&job.id, extractor(), Some(Arc::new(|| true))))
            .expect_err("aborted");
        assert_eq!(error, "The operation was aborted");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn temporal_prepare_preserves_anchor_on_every_lossless_chunk_and_reload() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        runtime
            .block_on(
                jobs.store
                    .configure(&serde_json::json!({"maxImportChunkChars": 4096})),
            )
            .unwrap();
        let text = format!(
            "I repaired the observatory clock yesterday. {}",
            "\u{1f680}".repeat(9000)
        );
        let timestamp = 1_709_294_400_000.0;
        let source = root.join("temporal-chunks.jsonl");
        std::fs::write(
            &source,
            serde_json::json!({
                "type": "message", "id": "row", "timestamp": "2099-01-01T00:00:00Z",
                "message": {"role": "user", "content": text, "timestamp": timestamp}
            })
            .to_string(),
        )
        .unwrap();
        let job = runtime
            .block_on(jobs.prepare(&source.to_string_lossy()))
            .unwrap();
        assert!(job.chunks.len() > 1);
        assert_eq!(job.coverage.evidence_records, 1);
        let mut joined = String::new();
        let mut offset = 0;
        for chunk in &job.chunks {
            let label_text = serialize_evidence(&chunk.records, usize::MAX / 4);
            assert!(label_text.contains(r#""timestamp":"2024-03-01T12:00:00.000Z""#));
            assert!(!label_text.contains("2099-01-01"));
            assert!(js_len(&label_text) <= 4096);
            for record in &chunk.records {
                assert_eq!(record.timestamp, Some(timestamp));
                assert_eq!(record.id, format!("row:{offset}"));
                assert_eq!(record.sha256, hash(&text));
                joined.push_str(&record.text);
                offset += js_len(&record.text);
            }
        }
        assert_eq!(joined, text);
        assert_eq!(jobs.get(&job.id).unwrap(), job);
        assert_eq!(
            std::fs::read_to_string(&source).unwrap(),
            std::fs::read_to_string(Path::new(&jobs.dir).join(format!("{}.source.jsonl", job.id)))
                .unwrap()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn temporal_prepare_never_promotes_row_append_time_to_an_observation_date() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let source = root.join("undated.jsonl");
        let rows = [
            serde_json::json!({"type": "message", "id": "missing", "timestamp": "2024-03-01T12:00:00Z",
                "message": {"role": "assistant", "content": "I finished yesterday."}}),
            serde_json::json!({"type": "message", "id": "zero", "timestamp": 1_700_000_000_000i64,
                "message": {"role": "user", "content": "A later event", "timestamp": 0}}),
        ];
        std::fs::write(
            &source,
            rows.iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        let job = runtime
            .block_on(jobs.prepare(&source.to_string_lossy()))
            .unwrap();
        assert_eq!(job.coverage.evidence_records, 2);
        let records: Vec<_> = job
            .chunks
            .iter()
            .flat_map(|chunk| &chunk.records)
            .cloned()
            .collect();
        assert_eq!(
            records[0].origin,
            super::super::evidence::MemoryOrigin::Assistant
        );
        assert!(records.iter().all(|record| record.timestamp.is_none()));
        let text = serialize_evidence(&records, 80_000);
        assert!(!text.contains("\"timestamp\""));
        assert!(!text.contains("2023-11-14"));
        assert!(!text.contains("2024-03-01"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn temporal_old_import_job_loads_without_rewrite_or_timestamp_backfill() {
        let (jobs, root) = fixture();
        std::fs::create_dir_all(&jobs.dir).unwrap();
        let id = "import_0123456789abcdef01234567";
        let value = serde_json::json!({
            "schema": 1.0, "id": id, "source": "legacy.jsonl", "sourceHash": "legacy-source",
            "status": "pending", "chunks": [{"records": [{
                "id": "legacy:0", "origin": "user", "text": "Last year", "sha256": "legacy-hash",
                "uri": "file:///legacy.jsonl#L1"
            }], "startLine": 1, "endLine": 1}],
            "nextChunk": 0, "expectedRevision": 0,
            "coverage": {"lines": 1, "evidenceRecords": 1, "excludedLines": []},
            "usage": {"input": 0.0, "output": 0.0}
        });
        let path = jobs.path(id).unwrap();
        let raw = serde_json::to_string_pretty(&value).unwrap();
        std::fs::write(&path, &raw).unwrap();
        let job = jobs.get(id).unwrap();
        assert_eq!(job.schema, 1.0);
        assert_eq!(job.chunks[0].records[0].timestamp, None);
        assert_eq!(serde_json::to_value(&job).unwrap(), value);
        assert_eq!(jobs.list(), vec![job]);
        assert_eq!(std::fs::read_to_string(path).unwrap(), raw);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn temporal_invalid_message_timestamp_keeps_existing_exclusion_semantics() {
        let (jobs, root) = fixture();
        let runtime = runtime();
        let source = root.join("invalid-temporal.jsonl");
        let invalid = [
            serde_json::json!("not a date"),
            Value::Null,
            serde_json::json!({}),
            serde_json::json!(true),
        ];
        let rows: Vec<_> = invalid.into_iter().enumerate().map(|(index, timestamp)| {
            serde_json::json!({"type": "message", "id": format!("row-{index}"), "timestamp": 1_700_000_000_000i64,
                "message": {"role": "user", "content": "An observation", "timestamp": timestamp}}).to_string()
        }).collect();
        std::fs::write(&source, rows.join("\n")).unwrap();
        let job = runtime
            .block_on(jobs.prepare(&source.to_string_lossy()))
            .unwrap();
        assert_eq!(job.coverage.evidence_records, 0);
        assert_eq!(job.coverage.excluded_lines, vec![1, 2, 3, 4]);
        assert!(job.chunks.is_empty());
        let nonfinite = root.join("nonfinite.jsonl");
        std::fs::write(
            &nonfinite,
            r#"{"type":"message","message":{"role":"user","timestamp":NaN}}"#,
        )
        .unwrap();
        let error = runtime
            .block_on(jobs.prepare(&nonfinite.to_string_lossy()))
            .unwrap_err();
        assert!(error.starts_with("Invalid session JSON at line 1; preserved input at "));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn temporal_evidence_change_does_not_change_legacy_memory_document_schema() {
        let value = serde_json::json!({
            "schema": 1.0,
            "entries": {"memory": {"legacy": {
                "id": "legacy", "kind": "memory", "title": "Legacy", "content": "No event date",
                "path": "", "source": "legacy import", "created_at": "2020-01-01T00:00:00Z",
                "updated_at": "2020-01-01T00:00:00Z", "version": 1,
                "metadata": {"projectId": "legacy-project", "sources": [{
                    "id": "row", "origin": "user", "sha256": "legacy-hash", "uri": "file:///legacy#L1"
                }]}, "reference": {}, "arguments": {}
            }}, "prompt": {}, "skill": {}, "subagent": {}},
            "refinements": [],
            "memory": {"schema": 1.0, "projectId": "legacy-project", "revision": 0,
                "history": [], "events": {}}
        });
        let document = super::super::store::validate_document(&value, "legacy-project").unwrap();
        assert_eq!(serde_json::to_value(document).unwrap(), value);
    }
}
