//! H2 — memory benchmark driver (campaign: optimus-memory-bench).
//!
//! Drives the REAL optimus-agent memory system in-process against a manifest of
//! benchmark environments and questions (design: reports/H1-harness-design.md).
//!
//! Usage:
//!   cargo run -p pi-coding-agent --example memory_bench -- <manifest.json>
//!
//! Per environment: fresh scratch agent_dir / cwd / session_artifact_dir under
//! <output_dir>/envs/<env_id>/, a copied DGX provider entry in models.json, and a
//! scratch settings.json mirroring tests/jev_memory_retrieval.rs (autoRefine,
//! retry, compaction off; telemetry off). Sessions are ingested with the real
//! import pipeline (import_prepare -> import_run* -> import_apply) using the real
//! model from the registry. Each question runs in a full AgentSession (memory
//! extension only, real model) and captures: the injected recall (memory
//! diagnostic), the final answer, usage, latency, plus direct MemoryService
//! search/recall results.
//!
//! Outputs (append-only, resumable by (run_id, env_id[, qid])):
//!   <output_dir>/run.jsonl  — one record per question
//!   <output_dir>/env.jsonl  — one record per environment (store snapshot)
//!
//! Isolation: PRIME_AGENT_CODING_AGENT_DIR is removed in-process; every path is
//! explicit under the manifest's output_dir. No secrets are printed; models.json
//! is copied programmatically from the production profile (env override
//! MEMORY_BENCH_MODELS_JSON).

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Map, Value};

use pi_coding_agent::core::agent_session_services::{
    create_agent_session_from_services, create_agent_session_services,
    AgentSessionCreationOptions, CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
};
use pi_coding_agent::core::auth_storage::AuthStorage;
use pi_coding_agent::core::extensions::builtin::memory::create_memory_extension;
use pi_coding_agent::core::kernel::shared::HostRequestHandler;
use pi_coding_agent::core::memory::service::{create_memory_host_handlers, MemoryService};
use pi_coding_agent::core::model_registry::ModelRegistry;
use pi_coding_agent::core::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::core::session_manager::SessionManager;
use pi_coding_agent::core::settings_manager::SettingsManager;
use pi_coding_agent::core::side_question::read_assistant_text;

// ---------------------------------------------------------------------------
// Manifest (v1)
// ---------------------------------------------------------------------------

fn default_true() -> bool {
    true
}

fn default_variant() -> String {
    "base".to_string()
}

fn default_question_timeout_s() -> u64 {
    180
}

fn default_ingest_timeout_s() -> u64 {
    3600
}

fn default_import_run_timeout_s() -> u64 {
    900
}

fn default_import_run_retries() -> u32 {
    3
}

#[derive(Debug, Deserialize)]
struct ModelSpec {
    provider: String,
    id: String,
}

#[derive(Debug, Deserialize)]
struct EventSpec {
    role: String,
    text: String,
    #[serde(default)]
    ext_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SessionSpec {
    session_id: String,
    #[serde(default)]
    events: Vec<EventSpec>,
}

#[derive(Debug, Deserialize)]
struct EnvSpec {
    env_id: String,
    #[serde(default)]
    sessions: Vec<SessionSpec>,
    #[serde(default)]
    ground_truth_refs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QuestionSpec {
    qid: String,
    env_id: String,
    question: String,
    #[serde(default)]
    answer: Option<Value>,
    #[serde(default)]
    evidence: Vec<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    eval: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManifestSettings {
    #[serde(default = "default_true", alias = "recall_on")]
    recall_on: bool,
    #[serde(default, alias = "max_recall_chars")]
    max_recall_chars: Option<i64>,
    #[serde(default, alias = "max_recall_entries")]
    max_recall_entries: Option<i64>,
    /// Override memory.maxExtractionTokens (None = production default 4096).
    #[serde(default, alias = "max_extraction_tokens")]
    max_extraction_tokens: Option<i64>,
    /// Override memory.maxImportChunkChars (None = production default 40000).
    #[serde(default, alias = "max_import_chunk_chars")]
    max_import_chunk_chars: Option<i64>,
    /// Extra driver-level retries for a failed import_run invocation (the job
    /// checkpoints per chunk, so a retry resumes where it stopped).
    #[serde(default = "default_import_run_retries", alias = "import_run_retries")]
    import_run_retries: u32,
    #[serde(default, alias = "auto_refine")]
    auto_refine: Option<Value>,
    #[serde(default = "default_question_timeout_s", alias = "question_timeout_s")]
    question_timeout_s: u64,
    #[serde(default = "default_ingest_timeout_s", alias = "ingest_timeout_s")]
    ingest_timeout_s: u64,
    #[serde(default = "default_import_run_timeout_s", alias = "import_run_timeout_s")]
    import_run_timeout_s: u64,
    /// Disable the session toolset for question answering (default true).
    #[serde(default = "default_true", alias = "disable_tools")]
    disable_tools: bool,
}

impl Default for ManifestSettings {
    fn default() -> Self {
        ManifestSettings {
            recall_on: true,
            max_recall_chars: None,
            max_recall_entries: None,
            max_extraction_tokens: None,
            max_import_chunk_chars: None,
            import_run_retries: default_import_run_retries(),
            auto_refine: None,
            question_timeout_s: default_question_timeout_s(),
            ingest_timeout_s: default_ingest_timeout_s(),
            import_run_timeout_s: default_import_run_timeout_s(),
            disable_tools: true,
        }
    }
}

#[derive(Debug, Deserialize)]
struct Manifest {
    run_id: String,
    #[serde(default)]
    benchmark: String,
    #[serde(default = "default_variant")]
    variant: String,
    output_dir: String,
    #[serde(default)]
    envs: Vec<EnvSpec>,
    #[serde(default)]
    questions: Vec<QuestionSpec>,
    model: ModelSpec,
    #[serde(default)]
    settings: ManifestSettings,
    /// Informational only (subset-selection seeds are applied upstream).
    #[serde(default)]
    #[allow(dead_code)]
    seed: Option<Value>,
}

// ---------------------------------------------------------------------------
// Scratch environment
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct EnvScratch {
    #[allow(dead_code)]
    env_id: String,
    #[allow(dead_code)]
    root: PathBuf,
    agent_dir: PathBuf,
    cwd: PathBuf,
    session_artifact_dir: PathBuf,
    sessions_dir: PathBuf,
}

impl EnvScratch {
    fn prepare(output_dir: &Path, env: &EnvSpec) -> Result<EnvScratch, String> {
        let root = output_dir.join("envs").join(&env.env_id);
        let agent_dir = root.join("agent");
        let cwd = root.join("workspace");
        let session_artifact_dir = root.join("session-artifacts");
        let sessions_dir = root.join("sessions");
        for dir in [&agent_dir, &cwd, &session_artifact_dir, &sessions_dir] {
            std::fs::create_dir_all(dir).map_err(|error| {
                format!("cannot create scratch dir {}: {error}", dir.display())
            })?;
        }
        Ok(EnvScratch {
            env_id: env.env_id.clone(),
            root,
            agent_dir,
            cwd,
            session_artifact_dir,
            sessions_dir,
        })
    }

    fn write_models_json(&self, models_source: &Path, provider: &str) -> Result<(), String> {
        let raw = std::fs::read_to_string(models_source).map_err(|error| {
            format!(
                "cannot read models.json source {}: {error}",
                models_source.display()
            )
        })?;
        let parsed: Value = serde_json::from_str(&raw)
            .map_err(|error| format!("models.json source is not valid JSON: {error}"))?;
        let entry = parsed
            .get("providers")
            .and_then(|providers| providers.get(provider))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "provider \"{provider}\" not found in {}",
                    models_source.display()
                )
            })?;
        let mut providers = Map::new();
        providers.insert(provider.to_string(), entry);
        let document = json!({ "providers": Value::Object(providers) });
        let path = self.agent_dir.join("models.json");
        std::fs::write(&path, serde_json::to_string_pretty(&document).unwrap())
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        Ok(())
    }

    /// Scratch settings.json: the jev_memory_retrieval fixture keys plus the
    /// `memory` block derived from the manifest.
    fn write_settings_json(&self, settings: &ManifestSettings) -> Result<(), String> {
        let mut auto_refine = json!({"enabled": false});
        if let Some(override_value) = &settings.auto_refine {
            if let (Some(target), Some(source)) = (auto_refine.as_object_mut(), override_value.as_object()) {
                for (key, value) in source {
                    target.insert(key.clone(), value.clone());
                }
            }
        }
        let mut memory = json!({"recall": settings.recall_on});
        if let Some(max_chars) = settings.max_recall_chars {
            memory["maxRecallChars"] = json!(max_chars);
        }
        if let Some(max_entries) = settings.max_recall_entries {
            memory["maxRecallEntries"] = json!(max_entries);
        }
        if let Some(max_extraction_tokens) = settings.max_extraction_tokens {
            memory["maxExtractionTokens"] = json!(max_extraction_tokens);
        }
        if let Some(max_import_chunk_chars) = settings.max_import_chunk_chars {
            memory["maxImportChunkChars"] = json!(max_import_chunk_chars);
        }
        let document = json!({
            "autoRefine": auto_refine,
            "retry": {"enabled": false},
            "compaction": {"enabled": false},
            "telemetryEnabled": false,
            "agentTracesEnabled": false,
            "quietStartup": true,
            "memory": memory,
        });
        let path = self.agent_dir.join("settings.json");
        std::fs::write(&path, serde_json::to_string_pretty(&document).unwrap())
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        Ok(())
    }

    /// Write one synthetic session JSONL per manifest session; returns the
    /// event index (line -> ext_id/role) for capture-metric joins.
    fn write_session_files(&self, env: &EnvSpec) -> Result<Vec<Value>, String> {
        let mut index: Vec<Value> = Vec::new();
        for session in &env.sessions {
            let path = self.sessions_dir.join(format!("{}.jsonl", session.session_id));
            let mut lines: Vec<String> = Vec::new();
            let mut events: Vec<Value> = Vec::new();
            let mut previous_id: Option<String> = None;
            let base_timestamp_ms = 1_700_000_000_000i64;
            for (position, event) in session.events.iter().enumerate() {
                let row_id = format!("m_{:04}", position + 1);
                let timestamp = base_timestamp_ms + (position as i64) * 60_000;
                let row = json!({
                    "type": "message",
                    "id": row_id,
                    "parentId": previous_id,
                    "timestamp": timestamp,
                    "message": {
                        "role": event.role,
                        "content": [{"type": "text", "text": event.text}],
                        "timestamp": timestamp as f64,
                    },
                });
                lines.push(serde_json::to_string(&row).unwrap());
                events.push(json!({
                    "line": position + 1,
                    "row_id": row_id,
                    "ext_id": event.ext_id,
                    "role": event.role,
                }));
                previous_id = Some(row_id);
            }
            std::fs::write(&path, lines.join("\n") + "\n").map_err(|error| {
                format!("cannot write {}: {error}", path.display())
            })?;
            index.push(json!({
                "session_id": session.session_id,
                "file": path.to_string_lossy(),
                "events": events,
            }));
        }
        Ok(index)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn iso_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn append_jsonl(path: &Path, record: &Value) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut line = serde_json::to_string(record).unwrap();
    line.push('\n');
    file.write_all(line.as_bytes())
        .and_then(|_| file.flush())
        .map_err(|error| format!("cannot append to {}: {error}", path.display()))?;
    Ok(())
}

fn read_jsonl_keys(path: &Path, key_fields: &[&str]) -> Result<HashSet<String>, String> {
    let mut keys = HashSet::new();
    if !path.exists() {
        return Ok(keys);
    }
    let raw = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let key: Vec<String> = key_fields
            .iter()
            .map(|field| {
                value
                    .get(*field)
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        keys.insert(key.join("\u{1}"));
    }
    Ok(keys)
}

fn record_key(fields: &[&str]) -> String {
    fields.join("\u{1}")
}

/// Resolve (api_key, headers) for a model through a registry without blocking
/// a Tokio worker on the synchronous registry mutex.
async fn registry_auth(
    registry: Arc<Mutex<ModelRegistry>>,
    model: pi_ai::types::Model,
) -> Result<(String, Option<indexmap::IndexMap<String, String>>), String> {
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let mut guard = registry
            .lock()
            .map_err(|_| "model registry poisoned".to_string())?;
        let auth = handle.block_on(guard.get_api_key_and_headers(&model));
        if !auth.ok {
            return Err(auth
                .error
                .unwrap_or_else(|| "no API key available for the model".to_string()));
        }
        let api_key = auth
            .api_key
            .filter(|key| !key.is_empty())
            .ok_or_else(|| "no API key available for the model".to_string())?;
        Ok((api_key, auth.headers))
    })
    .await
    .map_err(|error| error.to_string())?
}

// ---------------------------------------------------------------------------
// Preflight (fail fast when the real endpoint is unreachable)
// ---------------------------------------------------------------------------

async fn preflight(
    output_dir: &Path,
    models_source: &Path,
    model_spec: &ModelSpec,
    settings: &ManifestSettings,
) -> Result<(), String> {
    let scratch_root = output_dir.join("_preflight");
    std::fs::create_dir_all(&scratch_root)
        .map_err(|error| format!("cannot create {}: {error}", scratch_root.display()))?;
    let scratch = EnvScratch {
        env_id: "_preflight".to_string(),
        root: scratch_root.clone(),
        agent_dir: scratch_root.join("agent"),
        cwd: scratch_root.join("workspace"),
        session_artifact_dir: scratch_root.join("session-artifacts"),
        sessions_dir: scratch_root.join("sessions"),
    };
    for dir in [&scratch.agent_dir, &scratch.cwd] {
        std::fs::create_dir_all(dir)
            .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    }
    scratch.write_models_json(models_source, &model_spec.provider)?;
    scratch.write_settings_json(settings)?;

    let registry = Arc::new(Mutex::new(ModelRegistry::create(
        AuthStorage::create(Some(scratch.agent_dir.join("auth.json").to_string_lossy().into_owned()), None),
        Some(scratch.agent_dir.join("models.json").to_string_lossy().into_owned()),
    )));
    let model = registry
        .lock()
        .map_err(|_| "model registry poisoned".to_string())?
        .find(&model_spec.provider, &model_spec.id)
        .ok_or_else(|| {
            format!(
                "model {}/{} not found in the scratch registry (provider entry copy failed?)",
                model_spec.provider, model_spec.id
            )
        })?;
    let (api_key, headers) = registry_auth(registry.clone(), model.clone()).await?;
    let context = pi_ai::types::Context {
        system_prompt: Some("You are a connectivity probe. Reply with the single word OK.".to_string()),
        messages: vec![pi_ai::types::Message::User(pi_ai::types::UserMessage::new(
            pi_ai::types::UserContent::Text("ping".to_string()),
            1_700_000_000_000,
        ))],
        tools: None,
    };
    let options = pi_ai::types::SimpleStreamOptions {
        stream: pi_ai::types::StreamOptions {
            max_tokens: Some(16.0),
            api_key: Some(api_key),
            headers,
            ..Default::default()
        },
        ..Default::default()
    };
    let reply = tokio::time::timeout(
        Duration::from_secs(120),
        pi_ai::stream::complete_simple(&model, &context, Some(&options)),
    )
    .await
    .map_err(|_| "preflight timed out after 120s".to_string())?;
    if reply.stop_reason.as_str() == "error" {
        return Err(format!(
            "preflight completion failed: {}",
            reply.error_message.unwrap_or_else(|| "unknown provider error".to_string())
        ));
    }
    eprintln!("[preflight] {}/{} reachable", model_spec.provider, model_spec.id);
    Ok(())
}

// ---------------------------------------------------------------------------
// Ingest (import mode)
// ---------------------------------------------------------------------------

struct IngestOutcome {
    record: Value,
}

async fn ingest_env(
    scratch: &EnvScratch,
    env: &EnvSpec,
    manifest: &Manifest,
    event_index: &[Value],
) -> Result<IngestOutcome, String> {
    let started = Instant::now();
    let cwd_str = scratch.cwd.to_string_lossy().into_owned();
    let agent_dir_str = scratch.agent_dir.to_string_lossy().into_owned();
    let session_artifact_str = scratch
        .session_artifact_dir
        .to_string_lossy()
        .into_owned();

    let get_model_info: HostRequestHandler = {
        let provider = manifest.model.provider.clone();
        let id = manifest.model.id.clone();
        Arc::new(move |_payload| {
            let provider = provider.clone();
            let id = id.clone();
            Box::pin(async move {
                Ok(json!({"provider": provider, "id": id}))
            })
        })
    };
    let handlers = create_memory_host_handlers(
        cwd_str.clone(),
        Some(agent_dir_str.clone()),
        Some(session_artifact_str.clone()),
        Some(get_model_info),
    );
    let memory_request = handlers
        .get("memory.request")
        .ok_or("memory.request handler missing")?
        .clone();

    let mut job_ids: Vec<String> = Vec::new();
    let mut total_usage_input = 0.0f64;
    let mut total_usage_output = 0.0f64;
    let mut chunk_count = 0usize;

    for session in &env.sessions {
        if session.events.is_empty() {
            eprintln!(
                "[env {}] session {} has no events; skipping import",
                env.env_id, session.session_id
            );
            continue;
        }
        let source = scratch
            .sessions_dir
            .join(format!("{}.jsonl", session.session_id))
            .to_string_lossy()
            .into_owned();

        // import_prepare
        let prepared = tokio::time::timeout(
            Duration::from_secs(60),
            call_memory(&memory_request, json!({"action": "import_prepare", "path": source})),
        )
        .await
        .map_err(|_| format!("import_prepare timed out for {}", session.session_id))?
        .map_err(|error| format!("import_prepare failed: {error}"))?;
        let job = prepared
            .get("result")
            .cloned()
            .ok_or("import_prepare returned no result")?;
        let job_id = job
            .get("id")
            .and_then(Value::as_str)
            .ok_or("import_prepare returned no job id")?
            .to_string();
        job_ids.push(job_id.clone());

        // import_run until preview (each call processes at most
        // maxImportChunksPerRun chunks; the job checkpoints after every chunk,
        // so a failed invocation resumes from the last completed chunk).
        let mut invocations = 0usize;
        let max_invocations = 512usize;
        let mut consecutive_failures = 0u32;
        loop {
            invocations += 1;
            if invocations > max_invocations {
                return Err(format!(
                    "import_run for job {job_id} exceeded {max_invocations} invocations"
                ));
            }
            let outcome = tokio::time::timeout(
                Duration::from_secs(manifest.settings.import_run_timeout_s),
                call_memory(&memory_request, json!({"action": "import_run", "id": job_id})),
            )
            .await;
            let run = match outcome {
                Ok(Ok(value)) => value,
                Ok(Err(error)) => {
                    let reason = error;
                    consecutive_failures += 1;
                    eprintln!(
                        "[env {}] import_run invocation {invocations} failed ({consecutive_failures} consecutive): {reason}",
                        env.env_id
                    );
                    if consecutive_failures > manifest.settings.import_run_retries {
                        return Err(format!(
                            "import_run for job {job_id} failed {consecutive_failures} times in a row: {reason}"
                        ));
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
                Err(_) => {
                    let reason = format!(
                        "import_run timed out for job {job_id} ({}s)",
                        manifest.settings.import_run_timeout_s
                    );
                    consecutive_failures += 1;
                    eprintln!(
                        "[env {}] import_run invocation {invocations} failed ({consecutive_failures} consecutive): {reason}",
                        env.env_id
                    );
                    if consecutive_failures > manifest.settings.import_run_retries {
                        return Err(format!(
                            "import_run for job {job_id} failed {consecutive_failures} times in a row: {reason}"
                        ));
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
            };
            consecutive_failures = 0;
            let result = run.get("result").cloned().unwrap_or(Value::Null);
            let status = result.get("status").and_then(Value::as_str).unwrap_or("");
            if status == "preview" || status == "applied" {
                if let Some(usage) = result.get("usage") {
                    total_usage_input += usage.get("input").and_then(Value::as_f64).unwrap_or(0.0);
                    total_usage_output += usage.get("output").and_then(Value::as_f64).unwrap_or(0.0);
                }
                if let Some(chunks) = result.get("chunks").and_then(Value::as_array) {
                    chunk_count += chunks.len();
                }
                break;
            }
        }

        // import_apply (revision read fresh right before applying)
        let service = MemoryService::new(&cwd_str, &agent_dir_str, Some(session_artifact_str.clone()))?;
        let revision = service.store.read()?.memory.revision;
        let applied = tokio::time::timeout(
            Duration::from_secs(120),
            call_memory(
                &memory_request,
                json!({"action": "import_apply", "id": job_id, "revision": revision}),
            ),
        )
        .await
        .map_err(|_| format!("import_apply timed out for job {job_id}"))?
        .map_err(|error| format!("import_apply failed for job {job_id}: {error}"))?;
        let status = applied
            .get("result")
            .and_then(|result| result.get("status"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if status != "applied" {
            return Err(format!(
                "import_apply for job {job_id} returned status {status:?}"
            ));
        }
    }

    // Snapshot the store plus the harness state files.
    let service = MemoryService::new(&cwd_str, &agent_dir_str, Some(session_artifact_str.clone()))?;
    let document = service.store.read()?;
    let store_snapshot = serde_json::to_value(&document.entries)
        .map_err(|error| format!("cannot serialize store snapshot: {error}"))?;
    let global_harness = read_json_file(
        &scratch.agent_dir.join("harness").join("harness_state.json"),
    );
    let session_harness = read_json_file(
        &scratch
            .session_artifact_dir
            .join("harness")
            .join("harness_state.json"),
    );
    let entry_count: usize = document
        .entries
        .values()
        .map(|bucket| bucket.len())
        .sum();

    let record = json!({
        "run_id": manifest.run_id,
        "benchmark": manifest.benchmark,
        "variant": manifest.variant,
        "env_id": env.env_id,
        "ingest_mode": "import",
        "project_id": service.store.project.id,
        "model": {"provider": manifest.model.provider, "id": manifest.model.id},
        "revision": document.memory.revision,
        "entry_count": entry_count,
        "store_snapshot": store_snapshot,
        "global_harness": global_harness,
        "session_harness": session_harness,
        "ingest_latency_ms": started.elapsed().as_millis() as u64,
        "ingest_llm_calls": chunk_count,
        "ingest_usage": {"input": total_usage_input, "output": total_usage_output},
        "job_ids": job_ids,
        "ground_truth_refs": env.ground_truth_refs,
        "sessions": event_index,
        "ts": iso_now(),
    });
    Ok(IngestOutcome { record })
}

async fn call_memory(
    request: &HostRequestHandler,
    payload: Value,
) -> Result<Value, String> {
    request(payload)
        .await
        .map_err(|error| error.to_string())
}

fn read_json_file(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------------
// Question runner
// ---------------------------------------------------------------------------

struct QuestionOutcome {
    record: Value,
}

#[allow(clippy::too_many_arguments)]
async fn run_question(
    scratch: &EnvScratch,
    manifest: &Manifest,
    question: &QuestionSpec,
) -> Result<QuestionOutcome, String> {
    let started = Instant::now();
    let cwd_str = scratch.cwd.to_string_lossy().into_owned();
    let agent_dir_str = scratch.agent_dir.to_string_lossy().into_owned();

    let settings = Arc::new(Mutex::new(SettingsManager::create(
        &cwd_str,
        Some(&agent_dir_str),
    )));
    let registry = Arc::new(Mutex::new(ModelRegistry::create(
        AuthStorage::create(
            Some(scratch.agent_dir.join("auth.json").to_string_lossy().into_owned()),
            None,
        ),
        Some(scratch.agent_dir.join("models.json").to_string_lossy().into_owned()),
    )));
    let model = {
        let guard = registry
            .lock()
            .map_err(|_| "model registry poisoned".to_string())?;
        guard
            .find(&manifest.model.provider, &manifest.model.id)
            .ok_or_else(|| {
                format!(
                    "model {}/{} not found in the scratch registry",
                    manifest.model.provider, manifest.model.id
                )
            })?
    };

    let services = create_agent_session_services(CreateAgentSessionServicesOptions {
        cwd: cwd_str.clone(),
        agent_dir: Some(agent_dir_str.clone()),
        auth_storage: None,
        settings_manager: Some(settings.clone()),
        model_registry: Some(registry),
        extension_flag_values: None,
        no_builtin_herdr_reporter: Some(true),
        telemetry_disabled: Some(true),
        resource_loader_options: Some(DefaultResourceLoaderOptions {
            cwd: cwd_str.clone(),
            agent_dir: agent_dir_str.clone(),
            no_extensions: true,
            extension_factories: vec![create_memory_extension(
                agent_dir_str.clone(),
                settings.clone(),
            )],
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            bundled_skills_dir: Some(None),
            ..Default::default()
        }),
    })
    .await
    .map_err(|error| format!("create_agent_session_services failed: {error}"))?;

    let session_manager = Arc::new(Mutex::new(
        SessionManager::in_memory(Some(&cwd_str), Some(&agent_dir_str))
            .map_err(|error| format!("in-memory session manager failed: {error}"))?,
    ));
    let created = create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services: Arc::new(services),
        session_manager,
        session_start_event: None,
        creation: AgentSessionCreationOptions {
            model: Some(model),
            no_tools: if manifest.settings.disable_tools {
                Some("all".to_string())
            } else {
                None
            },
            prewarm_ipython_kernel: Some(false),
            telemetry_disabled: Some(true),
            ..Default::default()
        },
    })
    .await
    .map_err(|error| format!("create_agent_session_from_services failed: {error}"))?;
    let session = created.session;

    // Prompt + wait, bounded by the per-question timeout.
    // Optional debug: dump the exact system prompt the question session uses.
    if std::env::var("MEMORY_BENCH_DUMP_SYSTEM_PROMPT").is_ok() {
        let _ = std::fs::write(
            scratch.root.join("system_prompt_debug.txt"),
            session.system_prompt(),
        );
    }

    let prompt_result = tokio::time::timeout(
        Duration::from_secs(manifest.settings.question_timeout_s),
        async {
            session
                .prompt_and_wait(&question.question, None)
                .await
                .map_err(|error| format!("prompt failed: {error}"))?;
            session
                .wait_for_headless_idle()
                .await
                .map_err(|error| format!("wait_for_headless_idle failed: {error}"))?;
            Ok::<(), String>(())
        },
    )
    .await;

    let mut record = json!({
        "run_id": manifest.run_id,
        "benchmark": manifest.benchmark,
        "variant": manifest.variant,
        "env_id": question.env_id,
        "qid": question.qid,
        "question": question.question,
        "model": {"provider": manifest.model.provider, "id": manifest.model.id},
        "answer": question.answer,
        "evidence": question.evidence,
        "category": question.category,
        "eval": question.eval,
        "ts": iso_now(),
    });

    match prompt_result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            record["error"] = json!(error);
            record["latency_ms"] = json!(started.elapsed().as_millis() as u64);
            session.dispose();
            return Ok(QuestionOutcome { record });
        }
        Err(_) => {
            record["error"] = json!(format!(
                "question timed out after {}s",
                manifest.settings.question_timeout_s
            ));
            record["latency_ms"] = json!(started.elapsed().as_millis() as u64);
            session.dispose();
            return Ok(QuestionOutcome { record });
        }
    }

    // Final assistant answer + usage.
    let assistant = session.agent.find_last_assistant_message();
    let answer_text = session
        .messages()
        .iter()
        .rev()
        .find_map(|message| {
            let text = read_assistant_text(message);
            if text.trim().is_empty() {
                None
            } else {
                Some(text)
            }
        })
        .unwrap_or_default();
    if let Some(assistant) = &assistant {
        record["stop_reason"] = json!(assistant.stop_reason.as_str());
        if let Some(error_message) = &assistant.error_message {
            record["error"] = json!(error_message);
        }
        record["usage"] = json!({
            "input_tokens": assistant.usage.input,
            "output_tokens": assistant.usage.output,
            "cache_read": assistant.usage.cache_read,
            "cache_write": assistant.usage.cache_write,
            "total_tokens": assistant.usage.total_tokens,
        });
    }
    record["answer_text"] = json!(answer_text);

    // Injected recall: memory-diagnostic session entries.
    let branch = {
        let manager = session
            .session_manager
            .lock()
            .map_err(|_| "session manager poisoned".to_string())?;
        manager.get_branch(None)
    };
    let mut diagnostic: Option<&Map<String, Value>> = None;
    for entry in &branch {
        let is_diagnostic = entry.get("type").and_then(Value::as_str) == Some("custom")
            && entry.get("customType").and_then(Value::as_str)
                == Some("prime-agent.memory-diagnostic");
        if !is_diagnostic {
            continue;
        }
        let data = entry.get("data").and_then(Value::as_object);
        if let Some(data) = data {
            if data.get("operation").and_then(Value::as_str) == Some("recall") {
                diagnostic = Some(data);
            }
        }
    }
    match diagnostic {
        Some(data) => {
            record["recall_ids"] = data.get("ids").cloned().unwrap_or(Value::Null);
            record["recall_chars"] = data.get("chars").cloned().unwrap_or(Value::Null);
            record["recall_latency_ms"] = data.get("latencyMs").cloned().unwrap_or(Value::Null);
        }
        None => {
            record["recall_ids"] = json!([]);
            record["recall_chars"] = json!(0);
            record["recall_latency_ms"] = Value::Null;
        }
    }

    session.dispose();

    // Direct retrieval (no LLM): full ranked search + rendered recall.
    let service = MemoryService::new(&cwd_str, &agent_dir_str, None)?;
    let hits = service.search(&question.question, false);
    let ranked: Vec<Value> = hits
        .iter()
        .take(50)
        .map(|hit| {
            json!({
                "id": hit.id,
                "scope": hit.scope.as_str(),
                "title": hit.entry.title,
                "score": hit.score,
                "matched": hit.matched,
                "version": hit.entry.version,
            })
        })
        .collect();
    let direct = service.recall(&question.question);
    record["search_ranked"] = json!(ranked);
    record["direct_recall_ids"] = json!(direct.ids);
    record["direct_recall_chars"] = json!(direct.chars);
    record["latency_ms"] = json!(started.elapsed().as_millis() as u64);

    Ok(QuestionOutcome { record })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: memory_bench <manifest.json>");
        std::process::exit(2);
    }
    let manifest_path = PathBuf::from(&args[1]);
    let raw = std::fs::read_to_string(&manifest_path).unwrap_or_else(|error| {
        eprintln!(
            "error: cannot read manifest {}: {error}",
            manifest_path.display()
        );
        std::process::exit(2);
    });
    let manifest: Manifest = serde_json::from_str(&raw).unwrap_or_else(|error| {
        eprintln!("error: invalid manifest JSON: {error}");
        std::process::exit(2);
    });
    let models_source = PathBuf::from(
        std::env::var("MEMORY_BENCH_MODELS_JSON").unwrap_or_else(|_| {
            "C:/Users/openclawuser/Optimus-Assistant/profile/agent/models.json".to_string()
        }),
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime");

    if let Err(error) = runtime.block_on(run(manifest, &models_source)) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run(manifest: Manifest, models_source: &Path) -> Result<(), String> {
    // Isolation: never resolve the production agent dir through the ambient env.
    std::env::remove_var("PRIME_AGENT_CODING_AGENT_DIR");
    // The ambient RLM_DEPTH would make benchmark sessions read as child agents
    // (child doctrine + RLM_CHILD_STATUS lines in answers). Benchmark sessions
    // are always top-level.
    std::env::remove_var("RLM_DEPTH");

    let output_dir = PathBuf::from(&manifest.output_dir);
    std::fs::create_dir_all(&output_dir)
        .map_err(|error| format!("cannot create output dir: {error}"))?;
    let run_path = output_dir.join("run.jsonl");
    let env_path = output_dir.join("env.jsonl");

    let done_questions = read_jsonl_keys(&run_path, &["run_id", "env_id", "qid"])?;
    let done_envs = read_jsonl_keys(&env_path, &["run_id", "env_id"])?;

    eprintln!(
        "[run {}] benchmark={} variant={} envs={} questions={} (resume: {} questions, {} envs done)",
        manifest.run_id,
        manifest.benchmark,
        manifest.variant,
        manifest.envs.len(),
        manifest.questions.len(),
        done_questions.len(),
        done_envs.len()
    );

    // Fail fast if the real endpoint is unreachable.
    preflight(&output_dir, models_source, &manifest.model, &manifest.settings).await?;

    // Environments: scratch + ingest + snapshot.
    let mut scratch_by_env: std::collections::HashMap<String, EnvScratch> =
        std::collections::HashMap::new();
    for (position, env) in manifest.envs.iter().enumerate() {
        let scratch = EnvScratch::prepare(&output_dir, env)?;
        // Always refresh credentials/settings (cheap, keeps variant knobs current).
        scratch.write_models_json(models_source, &manifest.model.provider)?;
        scratch.write_settings_json(&manifest.settings)?;
        scratch_by_env.insert(env.env_id.clone(), scratch.clone());

        let key = record_key(&[&manifest.run_id, &env.env_id]);
        if done_envs.contains(&key) {
            eprintln!(
                "[env {}/{}] {} already ingested (env.jsonl); skipping ingest",
                position + 1,
                manifest.envs.len(),
                env.env_id
            );
            continue;
        }
        eprintln!(
            "[env {}/{}] {} ingesting {} session(s)",
            position + 1,
            manifest.envs.len(),
            env.env_id,
            env.sessions.len()
        );
        let event_index = scratch.write_session_files(env)?;
        let started = Instant::now();
        let outcome = tokio::time::timeout(
            Duration::from_secs(manifest.settings.ingest_timeout_s),
            ingest_env(&scratch, env, &manifest, &event_index),
        )
        .await
        .map_err(|_| {
            format!(
                "ingest for env {} timed out after {}s",
                env.env_id, manifest.settings.ingest_timeout_s
            )
        })??;
        append_jsonl(&env_path, &outcome.record)?;
        eprintln!(
            "[env {}/{}] {} ingested in {}ms ({} entries)",
            position + 1,
            manifest.envs.len(),
            env.env_id,
            started.elapsed().as_millis(),
            outcome
                .record
                .get("entry_count")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        );
    }

    // Questions.
    for (position, question) in manifest.questions.iter().enumerate() {
        let scratch = scratch_by_env.get(&question.env_id).ok_or_else(|| {
            format!(
                "question {} references unknown env {}",
                question.qid, question.env_id
            )
        })?;
        let key = record_key(&[&manifest.run_id, &question.env_id, &question.qid]);
        if done_questions.contains(&key) {
            eprintln!(
                "[q {}/{}] {} already recorded (run.jsonl); skipping",
                position + 1,
                manifest.questions.len(),
                question.qid
            );
            continue;
        }
        eprintln!(
            "[q {}/{}] {} asking ({} chars)",
            position + 1,
            manifest.questions.len(),
            question.qid,
            question.question.chars().count()
        );
        let started = Instant::now();
        let outcome = run_question(scratch, &manifest, question).await?;
        append_jsonl(&run_path, &outcome.record)?;
        let error = outcome.record.get("error").and_then(Value::as_str);
        eprintln!(
            "[q {}/{}] {} done in {}ms{}",
            position + 1,
            manifest.questions.len(),
            question.qid,
            started.elapsed().as_millis(),
            error.map(|error| format!(" ERROR: {error}")).unwrap_or_default()
        );
    }

    eprintln!("[run {}] complete", manifest.run_id);
    Ok(())
}
