//! Port of packages/coding-agent/src/core/memory/service.ts
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::core::kernel::shared::{
    HostRequestHandler, HostRequestHandlers, KernelError,
};
use crate::core::refinement::refinement::{
    get_global_harness_state_dir, get_local_harness_state_dir, load_harness_state,
    normalize_refinement_proposal, HarnessScope,
};

use super::evidence::{hash, MemoryOrigin, MemorySource};
use super::jobs::{import_overview, ImportJob, MemoryExtractor, MemoryJobs};
use super::project::{project_identity, ProjectIdentity};
use super::search::{
    freshness, recall_memory, search_memory, MemoryHit, MemoryScope, RecallResult, SearchCorpus,
};
use super::sharing::MemorySharing;
use super::store::{record, MemoryStore};

pub type JsonMap = Map<String, Value>;

/// Port of `getAgentDir()` from config.ts: PRIME_AGENT_CODING_AGENT_DIR wins,
/// otherwise `<home>/.prime/agent`.
/// blocked_on: needs crate::config::get_agent_dir
pub fn get_agent_dir() -> String {
    let env_name = "PRIME_AGENT_CODING_AGENT_DIR";
    if let Ok(value) = std::env::var(env_name) {
        if !value.is_empty() {
            return expand_tilde_path(&value);
        }
    }
    dirs::home_dir()
        .map(|home| {
            home.join(".prime")
                .join("agent")
                .to_string_lossy()
                .to_string()
        })
        .unwrap_or_else(|| ".prime/agent".to_string())
}

fn expand_tilde_path(path: &str) -> String {
    if path == "~" {
        return dirs::home_dir()
            .map(|home| home.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        return dirs::home_dir()
            .map(|home| home.join(rest).to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
    }
    path.to_string()
}

#[derive(Clone)]
pub struct MemoryService {
    pub store: MemoryStore,
    pub jobs: MemoryJobs,
    pub sharing: MemorySharing,
    pub session_artifact_dir: Option<String>,
}

impl MemoryService {
    pub fn new(
        cwd: &str,
        agent_dir: &str,
        session_artifact_dir: Option<String>,
    ) -> Result<MemoryService, String> {
        let store = MemoryStore::new(agent_dir, project_identity(cwd, agent_dir, None)?)?;
        let jobs = MemoryJobs::new(store.clone());
        let sharing = MemorySharing::new(store.clone());
        Ok(MemoryService {
            store,
            jobs,
            sharing,
            session_artifact_dir,
        })
    }

    pub fn search(&self, query: &str, include_inactive: bool) -> Vec<MemoryHit> {
        let mut extra: Vec<SearchCorpus> = vec![
            SearchCorpus {
                state: load_harness_state(
                    &get_global_harness_state_dir(&self.store.agent_dir),
                    HarnessScope::Global,
                ),
                scope: MemoryScope::Global,
            },
            SearchCorpus {
                state: self.sharing.cache().state.harness(),
                scope: MemoryScope::Shared,
            },
        ];
        if let Some(session_artifact_dir) = &self.session_artifact_dir {
            if let Some(dir) = get_local_harness_state_dir(Some(session_artifact_dir)) {
                extra.push(SearchCorpus {
                    state: load_harness_state(&dir, HarnessScope::Local),
                    scope: MemoryScope::Session,
                });
            }
        }
        // A selected shared copy of a local memory should not consume recall twice.
        let found = search_memory(&self.store, query, &extra, include_inactive);
        let local_ids: Vec<String> = found
            .iter()
            .filter(|hit| hit.scope == MemoryScope::Project)
            .map(|hit| format!("{}:{}", hit.entry.kind.as_str(), hit.entry.id))
            .collect();
        found
            .into_iter()
            .filter(|hit| {
                hit.scope != MemoryScope::Shared
                    || !local_ids.contains(&format!("{}:{}", hit.entry.kind.as_str(), hit.entry.id))
            })
            .collect()
    }

    pub fn recall(&self, query: &str) -> RecallResult {
        self.render_recall(&self.search(query, false))
    }

    /// Render a request-local candidate selection without changing stored memory.
    pub fn render_recall(&self, hits: &[MemoryHit]) -> RecallResult {
        recall_memory(hits, &self.store.settings(), Some(&self.store.project.root))
    }

    pub async fn request(
        &self,
        action: &str,
        payload: &JsonMap,
        extract: Option<MemoryExtractor>,
    ) -> Result<Value, String> {
        let string = |name: &str| -> Result<String, String> {
            payload
                .get(name)
                .and_then(Value::as_str)
                .map(str::to_string)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| format!("{name} is required"))
        };
        let revision = || -> Result<i64, String> {
            match payload.get("revision") {
                Some(Value::Number(number)) => number
                    .as_f64()
                    .filter(|value| {
                        value.fract() == 0.0 && *value >= 0.0 && *value <= 9007199254740991.0
                    })
                    .map(|value| value as i64)
                    .ok_or_else(|| "Supply the current revision from memory.status()".to_string()),
                _ => Err("Supply the current revision from memory.status()".to_string()),
            }
        };
        match action {
            "status" => {
                let settings = self.store.settings();
                let cache = self.sharing.cache();
                let jobs: Vec<Value> = self
                    .jobs
                    .list()
                    .iter()
                    .map(|job| {
                        serde_json::json!({
                            "id": job.id,
                            "status": job.status.as_str(),
                            "nextChunk": job.next_chunk,
                            "chunks": job.chunks.len(),
                            "error": job.error,
                            "usage": job.usage,
                        })
                    })
                    .collect();
                Ok(serde_json::json!({
                    "project": self.store.project,
                    "revision": self.store.read()?.memory.revision,
                    "settings": settings,
                    "hostId": self.store.host_id,
                    "sharing": {
                        "configured": settings.shared.is_some(),
                        "connected": cache.connected,
                        "pending": cache.pending.len(),
                        "error": cache.error,
                    },
                    "jobs": jobs,
                }))
            }
            "search" => {
                let hits = self.search(
                    payload.get("query").and_then(Value::as_str).unwrap_or(""),
                    payload.get("includeInactive") == Some(&Value::Bool(true)),
                );
                let scope_filter = payload.get("scope").and_then(Value::as_str);
                let values: Vec<Value> = hits
                    .iter()
                    .filter(|hit| scope_filter.is_none() || scope_filter == Some(hit.scope.as_str()))
                    .take(50)
                    .map(|hit| {
                        serde_json::json!({
                            "id": hit.id,
                            "scope": hit.scope.as_str(),
                            "title": hit.entry.title,
                            "preview": hit.entry.content.chars().take(600).collect::<String>(),
                            "truncated": hit.entry.content.chars().count() > 600,
                            "version": hit.entry.version,
                            "status": if hit.entry.metadata.contains_key("supersededBy") { "superseded" } else { "current" },
                            "score": hit.score,
                            "matched": hit.matched,
                            "freshness": freshness(&hit.sources, Some(&self.store.project.root)).as_str(),
                            "sources": hit.sources.iter().take(8).map(|source| serde_json::to_value(source).unwrap_or(Value::Null)).collect::<Vec<_>>(),
                        })
                    })
                    .collect();
                Ok(Value::Array(values))
            }
            "read" => {
                let id = string("id")?;
                let hit = self
                    .search("", true)
                    .into_iter()
                    .find(|entry| entry.id == id)
                    .ok_or_else(|| "Memory not found in this scope".to_string())?;
                let mut value = serde_json::to_value(&hit).unwrap_or(Value::Null);
                if let Some(map) = value.as_object_mut() {
                    map.insert(
                        "freshness".to_string(),
                        Value::String(
                            freshness(&hit.sources, Some(&self.store.project.root))
                                .as_str()
                                .to_string(),
                        ),
                    );
                }
                Ok(value)
            }
            "configure" => {
                let settings = self
                    .store
                    .configure(payload.get("settings").unwrap_or(&Value::Null))
                    .await?;
                Ok(serde_json::to_value(settings).unwrap_or(Value::Null))
            }
            "bind" => {
                let project_id = string("projectId")?;
                let identity = project_identity(
                    &self.store.project.root,
                    &self.store.agent_dir,
                    Some(&project_id),
                )?;
                Ok(serde_json::to_value(identity).unwrap_or(Value::Null))
            }
            "apply" => {
                let raw_proposal = payload.get("proposal").cloned().unwrap_or(Value::Null);
                let proposal = normalize_refinement_proposal(&raw_proposal);
                let raw_edits = record(&raw_proposal)?
                    .get("edits")
                    .and_then(Value::as_array)
                    .cloned();
                match raw_edits {
                    Some(edits) if edits.len() == proposal.edits.len() => {}
                    _ => return Err("Invalid proposal edits".to_string()),
                }
                let sources = self.validate_sources(payload.get("sources"))?;
                let result = self
                    .store
                    .apply(
                        &proposal,
                        super::store::ApplyOptions {
                            event_id: string("eventId")?,
                            expected_revision: revision()?,
                            host: payload.get("host") == Some(&Value::Bool(true)),
                            automatic: payload.get("automatic") == Some(&Value::Bool(true)),
                            sources: Some(sources),
                            replace_metadata: false,
                        },
                    )
                    .await?;
                Ok(serde_json::to_value(result).unwrap_or(Value::Null))
            }
            "handoff" => {
                let task = string("task")?;
                let digest = hash(&task);
                let id = format!("handoff_{}", &digest[..24.min(digest.len())]);
                let before = self
                    .store
                    .read()?
                    .entries
                    .get("memory")
                    .and_then(|bucket| bucket.get(&id))
                    .cloned();
                let content = [
                    "State".to_string(),
                    string("state")?,
                    "Decisions".to_string(),
                    string("decisions")?,
                    "Unresolved".to_string(),
                    string("unresolved")?,
                ]
                .join("\n\n");
                let proposal = normalize_refinement_proposal(&serde_json::json!({
                    "summary": format!("Handoff: {task}"),
                    "rationale": "Explicit project task checkpoint",
                    "expectedOutcome": "Resume work with source links",
                    "edits": [{
                        "action": if before.is_some() { "update" } else { "create" },
                        "kind": "memory",
                        "id": id,
                        "title": task,
                        "content": content,
                        "path": "handoffs",
                        "metadata": {"task": task},
                    }],
                }));
                let result = self
                    .store
                    .apply(
                        &proposal,
                        super::store::ApplyOptions {
                            event_id: string("eventId")?,
                            expected_revision: revision()?,
                            sources: Some(self.validate_sources(payload.get("sources"))?),
                            automatic: payload.get("automatic") == Some(&Value::Bool(true)),
                            host: false,
                            replace_metadata: false,
                        },
                    )
                    .await?;
                Ok(serde_json::to_value(result).unwrap_or(Value::Null))
            }
            "source" => {
                let path = string("path")?;
                // `service.ts:175` / `:183`: the reference resolves the input with
                // `resolve(path)` (process-cwd based, lexical) and then records
                // `pathToFileURL(path).href`, so a relative in-project path is a
                // project path and gains an absolute `file:` URI.
                let resolved = resolve_source_path(&path);
                let metadata = std::fs::metadata(&resolved)
                    .map_err(|_| "Source must be a file below 32 MiB".to_string())?;
                if !metadata.is_file() || metadata.len() > 32 * 1024 * 1024 {
                    return Err("Source must be a file below 32 MiB".to_string());
                }
                let content = std::fs::read_to_string(&resolved)
                    .map_err(|_| "Source must be a file below 32 MiB".to_string())?;
                let sha256 = hash(&content);
                let relative = path_relative(&self.store.project.root, &resolved);
                let project_path = relative.filter(|value| !value.starts_with(".."));
                Ok(serde_json::json!({
                    "id": format!("file_{}", &sha256[..24.min(sha256.len())]),
                    "origin": "file",
                    "uri": url::Url::from_file_path(&resolved).map(|url| url.to_string()).unwrap_or_default(),
                    "sha256": sha256,
                    "projectPath": project_path,
                }))
            }
            "backup" => {
                let store = self.store.clone();
                let backup_store = store.clone();
                store
                    .exclusive(move || Ok(serde_json::json!({"id": backup_store.backup(None)?})))
                    .await
            }
            "restore" => {
                let id = string("id")?;
                Ok(serde_json::json!({"revision": self.store.restore(&id).await?}))
            }
            "history" => {
                Ok(serde_json::to_value(self.store.read()?.memory.history).unwrap_or(Value::Null))
            }
            "rollback" => {
                let id = string("id")?;
                let result = self.store.rollback(&id, revision()?).await?;
                Ok(serde_json::to_value(result).unwrap_or(Value::Null))
            }
            "import_prepare" => {
                let path = string("path")?;
                let job = self.jobs.prepare(&path).await?;
                Ok(import_overview(&job))
            }
            "import_read" => {
                let id = string("id")?;
                let job = self.jobs.get(&id)?;
                Ok(import_overview(&job))
            }
            "import_chunk" => {
                let job = self.jobs.get(&string("id")?)?;
                let index = match payload.get("chunk") {
                    Some(Value::Number(number)) => number
                        .as_f64()
                        .filter(|value| {
                            value.fract() == 0.0 && *value >= 0.0 && *value <= 9007199254740991.0
                        })
                        .map(|value| value as i64),
                    _ => None,
                };
                let index = index.ok_or_else(|| "Invalid import chunk index".to_string())?;
                if index < 0 || index as usize >= job.chunks.len() {
                    return Err("Invalid import chunk index".to_string());
                }
                Ok(serde_json::to_value(&job.chunks[index as usize]).unwrap_or(Value::Null))
            }
            "import_run" => {
                let extract = extract.ok_or_else(|| {
                    "Use /memory import-run <id> to extract with the session's selected model"
                        .to_string()
                })?;
                let job = self.jobs.run(&string("id")?, extract, None).await?;
                Ok(import_overview(&job))
            }
            "import_apply" => {
                let job = self.jobs.apply(&string("id")?, revision()?).await?;
                Ok(import_overview(&job))
            }
            "share" => {
                let ids = self.string_array(payload.get("ids"))?;
                let remove = match payload.get("remove") {
                    None => Vec::new(),
                    Some(value) => self.string_array(Some(value))?,
                };
                let write = self.sharing.queue(&ids, remove).await?;
                Ok(serde_json::to_value(write).unwrap_or(Value::Null))
            }
            "sync" => {
                let cache = self.sharing.sync().await?;
                Ok(serde_json::to_value(cache).unwrap_or(Value::Null))
            }
            "discard_pending" => {
                self.sharing.discard_pending().await?;
                Ok(serde_json::json!({"discarded": true}))
            }
            _ => Err(format!("Unknown memory operation: {action}")),
        }
    }

    fn string_array(&self, value: Option<&Value>) -> Result<Vec<String>, String> {
        match value {
            Some(Value::Array(values)) if values.iter().all(Value::is_string) => Ok(values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()),
            _ => Err("Expected an array of strings".to_string()),
        }
    }

    fn validate_sources(&self, value: Option<&Value>) -> Result<Vec<MemorySource>, String> {
        let value = match value {
            None => return Ok(Vec::new()),
            Some(value) => value,
        };
        let values = match value {
            Value::Array(values) if values.len() <= 200 => values,
            _ => return Err("Expected at most 200 sources".to_string()),
        };
        let mut sources = Vec::new();
        for raw in values {
            let source = record(raw)?;
            let sha256 = source.get("sha256").and_then(Value::as_str).unwrap_or("");
            let origin = source.get("origin").and_then(Value::as_str).unwrap_or("");
            if !matches!(source.get("id"), Some(Value::String(_)))
                || !matches!(source.get("sha256"), Some(Value::String(_)))
                || !regex::Regex::new(r"^[a-f0-9]{64}$")
                    .unwrap()
                    .is_match(sha256)
                || !["user", "assistant", "tool", "derived", "file"].contains(&origin)
            {
                return Err("Invalid source reference".to_string());
            }
            let origin = match origin {
                "user" => MemoryOrigin::User,
                "assistant" => MemoryOrigin::Assistant,
                "tool" => MemoryOrigin::Tool,
                "derived" => MemoryOrigin::Derived,
                _ => MemoryOrigin::File,
            };
            sources.push(MemorySource {
                id: source
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                origin,
                sha256: sha256.to_string(),
                uri: source
                    .get("uri")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                revision: source
                    .get("revision")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                project_path: source
                    .get("projectPath")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            });
        }
        Ok(sources)
    }
}

/// `resolve(path)` from `service.ts:175`: Node resolves a relative input against
/// the process cwd and normalizes `.`/`..` lexically, without touching the disk.
fn resolve_source_path(path: &str) -> PathBuf {
    let cwd = std::env::current_dir()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_default();
    PathBuf::from(crate::core::tools::path_utils::resolve_path(&cwd, path))
}

/// `path.relative(root, target)` for the `source` decision (`service.ts:175-179`).
///
/// Measured Node win32 semantics (`work/logs/memoryscope/ts-relative-rule.out.json`):
/// the shared prefix is compared case-insensitively and the remainder keeps the
/// target's original spelling. The port's project root can additionally carry a
/// verbatim device prefix, which Node's `realpathSync` never returns
/// (`project.ts:41`), so that prefix is normalized away before the comparison.
fn path_relative(root: &str, target: &Path) -> Option<String> {
    let root_components = comparable_components(root);
    let target_components = comparable_components(&target.to_string_lossy());
    let mut common = 0;
    while common < root_components.len().min(target_components.len())
        && root_components[common].0 == target_components[common].0
    {
        common += 1;
    }
    let mut parts: Vec<String> = Vec::new();
    for _ in common..root_components.len() {
        parts.push("..".to_string());
    }
    for component in &target_components[common..] {
        parts.push(component.1.clone());
    }
    Some(parts.join("/"))
}

/// `(comparison key, original spelling)` per component. The key drops the verbatim
/// device prefix and lowercases on Windows so the shared prefix matches the way
/// Node's win32 `path.relative` matches it; the original spelling is what the
/// caller reports as `projectPath`.
fn comparable_components(value: &str) -> Vec<(String, String)> {
    let normalized = strip_verbatim_prefix(value);
    Path::new(&normalized)
        .components()
        .map(|component| {
            let original = component.as_os_str().to_string_lossy().to_string();
            let key = if cfg!(windows) {
                original.to_lowercase()
            } else {
                original.clone()
            };
            (key, original)
        })
        .collect()
}

/// Strip the verbatim device prefix: a `\\\\?\\C:\x` root becomes `C:\x`.
fn strip_verbatim_prefix(value: &str) -> String {
    if let Some(rest) = value.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{rest}");
    }
    if let Some(rest) = value.strip_prefix(r"\\?\") {
        return rest.to_string();
    }
    value.to_string()
}

pub fn import_overview_value(job: &ImportJob) -> Value {
    import_overview(job)
}

pub fn create_memory_host_handlers(
    cwd: String,
    agent_dir: Option<String>,
    session_artifact_dir: Option<String>,
    get_model_info: Option<HostRequestHandler>,
) -> HostRequestHandlers {
    let mut handlers: HostRequestHandlers = std::collections::HashMap::new();
    let handler: HostRequestHandler = Arc::new(move |payload: Value| {
        let cwd = cwd.clone();
        let agent_dir = agent_dir.clone();
        let session_artifact_dir = session_artifact_dir.clone();
        let get_model_info = get_model_info.clone();
        Box::pin(async move {
            let action = payload
                .get("action")
                .and_then(Value::as_str)
                .ok_or_else(|| KernelError::new("memory.request requires action"))?
                .to_string();
            let service = MemoryService::new(
                &cwd,
                agent_dir.as_deref().unwrap_or(&get_agent_dir()),
                session_artifact_dir,
            )
            .map_err(KernelError::new)?;
            let mut extract = None;
            if action == "import_run" {
                let get_model_info = get_model_info
                    .ok_or_else(|| KernelError::new("The host did not supply a selected model"))?;
                let info = get_model_info(serde_json::json!({})).await?;
                let (Some(provider), Some(id)) = (
                    info.get("provider").and_then(Value::as_str),
                    info.get("id").and_then(Value::as_str),
                ) else {
                    return Err(KernelError::new("The host did not supply a selected model"));
                };
                let directory = Path::new(&service.store.agent_dir);
                let mut registry = crate::core::model_registry::ModelRegistry::create(
                    crate::core::auth_storage::AuthStorage::create(
                        Some(directory.join("auth.json").to_string_lossy().into_owned()), None,
                    ),
                    Some(directory.join("models.json").to_string_lossy().into_owned()),
                );
                let model = registry.find(provider, id).ok_or_else(|| KernelError::new(
                    "Selected model is session-only; use /memory import-run in the owning session",
                ))?;
                let auth = registry.get_api_key_and_headers(&model).await;
                let api_key = auth.api_key.filter(|key| auth.ok && !key.is_empty())
                    .ok_or_else(|| KernelError::new("No credentials for the selected model"))?;
                let headers: Option<std::collections::HashMap<String, String>> =
                    auth.headers.map(|headers| headers.into_iter().collect());
                let memory = service.clone();
                let extraction_cwd = cwd.clone();
                let extractor: MemoryExtractor = Arc::new(move |records| {
                    let memory = memory.clone();
                    let model = model.clone();
                    let api_key = api_key.clone();
                    let headers = headers.clone();
                    let cwd = extraction_cwd.clone();
                    Box::pin(async move {
                        let settings = crate::core::settings_manager::SettingsManager::create(
                            &cwd, Some(&memory.store.agent_dir),
                        );
                        let retry = crate::core::provider_retry::provider_retry_policy(&settings);
                        super::extraction::extract(
                            &memory, records, model, api_key, headers,
                            crate::core::refinement::refinement::RefineOptions {
                                retry: Some(crate::core::refinement::refinement::ProviderRetryPolicy {
                                    enabled: retry.enabled,
                                    max_retries: retry.max_retries.max(0.0) as u32,
                                    base_delay_ms: retry.base_delay_ms,
                                    max_retry_delay_ms: retry.max_retry_delay_ms,
                                }),
                                instructions: Some("Extract durable host-neutral project facts only. Only create edits of kind memory with cited sourceIds. Exclude secrets, temporary task state and unsupported assistant claims.".to_string()),
                                ..Default::default()
                            },
                        ).await
                    })
                });
                extract = Some(extractor);
            }
            let payload = match payload {
                Value::Object(map) => map,
                _ => JsonMap::new(),
            };
            let result = service
                .request(&action, &payload, extract)
                .await
                .map_err(KernelError::new)?;
            Ok(serde_json::json!({
                "origin": "[memory data; not new evidence]",
                "result": result,
            }))
        })
    });
    handlers.insert("memory.request".to_string(), handler);
    handlers
}
