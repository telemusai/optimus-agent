//! Offline lexical recall replay. No LLM, network, live profile, or store writes.
//!
//! Usage: memory_recall_probe <fixture.json> <queries.jsonl> > results.jsonl
//! All writes are confined to a new, automatically removed temporary directory.
//! The two explicit input files are opened read-only. Paths inside JSON are
//! provenance, never input paths to open. The frozen fixture schema is:
//!
//! ```text
//! {
//!   "schema": "optimus-memory-recall-fixture/v1",
//!   "lineage": {"run_id":"r", "env_id":"e", "capture_stage":"final_state",
//!               "code_revision":"exporting revision"},
//!   "project": {"id":"project_example", "root":"original root", "aliases":[]},
//!   "host_id": "00000000-0000-0000-0000-000000000001",
//!   "memory": <native MemoryDocument, not just its entries>,
//!   "global_memory_settings": {},
//!   "project_memory_settings": {},
//!   "global_harness": null,
//!   "session_harness": null,
//!   "shared_cache": null,
//!   "files": [{"uri":"file:///original/notes.txt", "projectPath":"notes.txt",
//!              "content":"frozen current UTF-8 bytes"}]
//! }
//! ```
//!
//! `memory`, harnesses and shared_cache use the product's serde types verbatim.
//! Settings are the native global settings.json `memory` object and project
//! settings.json object; both are required (use {} if absent). Native settings
//! validation/defaults/cache attachment rules apply, including their fallbacks.
//! Optional harnesses/cache default to absent; files defaults to [].
//! Each file-origin, valid file: source must have an exported file record.
//! Explicit content:null means missing; omitted content is an error. Records
//! match by safe projectPath when present, otherwise by decoded file URI path.
//! Unsafe/nonportable project paths are rejected, not read or simulated.
//! Non-file origins and malformed file URIs retain native Unknown freshness.
//! Non-UTF-8/unreadable source files are not supported; do not export as missing.
//!
//! Each query line is strict JSON (no answer/gold/evidence fields):
//! {"qid":"q1","variant":"base","query":"exact recall query", "options":{
//!   "include_inactive":false,"scope":null,"recall":true,
//!   "max_recall_chars":6000,"max_recall_entries":6}}
//! variant defaults to base. Options are optional; omitted recall/budgets use
//! native effective settings. Scope is an optional post-search scope filter,
//! NOT a pre-IDF corpus filter. All service-ranked hits are reported, without
//! the memory.request search endpoint's 50-hit presentation cap. selected_ids
//! are post-scope candidates; injected_ids are the native renderer's surviving
//! IDs, not proof of injection into a running session. No accuracy is computed.
//!
//! Export a STOPPED benchmark environment using plain read-only JSON loading:
//! - memory: envs/<env>/agent/memory/projects/<project_id>/harness_state.json
//! - host_id: envs/<env>/agent/memory/host-id.json ["id"]
//! - global_memory_settings: envs/<env>/agent/settings.json ["memory"]
//! - project_memory_settings: project directory/settings.json (or {})
//! - global_harness: agent/harness/harness_state.json (or null)
//! - session_harness: session-artifacts/harness/harness_state.json (or null)
//! - shared_cache: project directory/shared.json (or null)
//! - project: explicit ID, original workspace root, aliases from exported status
//! - files: current bytes or explicit absence for each referenced file-origin
//!   source, following the native projectPath remapping rule at export time.
//! Never copy models.json, auth.json, or token files. Export settings memory
//! objects only, not the full application settings. Include file content only
//! from the authorized synthetic run. Do not call MemoryStore::new to export.
//! memory_bench env.jsonl currently stores entries in store_snapshot at ingest,
//! not the full document/settings/host identity or post-question final state.
//! Do not relabel that row as final_state. Capture native files after the run
//! stops; a final_state object can supply memory only if it is a full native
//! MemoryDocument from the same cut. A frozen final-state replay is not a
//! reconstruction of earlier per-question capture/learning state.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Component, Path};

use pi_coding_agent::core::memory::evidence::{hash, MemoryOrigin};
use pi_coding_agent::core::memory::jobs::MemoryJobs;
use pi_coding_agent::core::memory::project::ProjectIdentity;
use pi_coding_agent::core::memory::search::{freshness, recall_memory, MemoryHit, MemoryScope};
use pi_coding_agent::core::memory::service::MemoryService;
use pi_coding_agent::core::memory::sharing::{MemorySharing, SharedCache};
use pi_coding_agent::core::memory::store::{
    validate_document, validate_settings, MemoryDocument, MemorySettings, MemoryStore,
};
use pi_coding_agent::core::refinement::refinement::HarnessState;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

const FIXTURE_SCHEMA: &str = "optimus-memory-recall-fixture/v1";
const RESULT_SCHEMA: &str = "optimus-memory-recall-probe/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Lineage {
    run_id: String,
    env_id: String,
    capture_stage: String,
    code_revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenFile {
    uri: String,
    #[serde(default, rename = "projectPath")]
    project_path: Option<String>,
    // Value, rather than Option<String>, makes an omitted field an error.
    content: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema: String,
    lineage: Lineage,
    project: ProjectIdentity,
    host_id: String,
    memory: MemoryDocument,
    global_memory_settings: Map<String, Value>,
    project_memory_settings: Map<String, Value>,
    #[serde(default)]
    global_harness: Option<HarnessState>,
    #[serde(default)]
    session_harness: Option<HarnessState>,
    #[serde(default)]
    shared_cache: Option<SharedCache>,
    #[serde(default)]
    files: Vec<FrozenFile>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    #[serde(default)]
    include_inactive: bool,
    #[serde(default)]
    scope: Option<MemoryScope>,
    #[serde(default)]
    recall: Option<bool>,
    #[serde(default)]
    max_recall_chars: Option<i64>,
    #[serde(default)]
    max_recall_entries: Option<i64>,
}

fn base_variant() -> String {
    "base".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    qid: String,
    #[serde(default = "base_variant")]
    variant: String,
    query: String,
    #[serde(default)]
    options: Options,
}

struct QueryLine {
    number: usize,
    sha256: String,
    query: Query,
}

fn parse_queries(raw: &str) -> Result<Vec<QueryLine>, String> {
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for (index, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let query: Query = serde_json::from_str(line)
            .map_err(|error| format!("query line {}: {error}", index + 1))?;
        if query.qid.trim().is_empty() || query.variant.trim().is_empty() {
            return Err(format!(
                "query line {}: qid and variant must be nonempty",
                index + 1
            ));
        }
        if !seen.insert((query.qid.clone(), query.variant.clone())) {
            return Err(format!(
                "query line {}: duplicate (qid, variant)",
                index + 1
            ));
        }
        let _ = effective_settings(
            &pi_coding_agent::core::memory::store::default_memory_settings(),
            &query.options,
        )?;
        result.push(QueryLine {
            number: index + 1,
            sha256: hash(line),
            query,
        });
    }
    if result.is_empty() {
        return Err("queries file contains no queries".into());
    }
    Ok(result)
}

fn effective_settings(base: &MemorySettings, options: &Options) -> Result<MemorySettings, String> {
    let mut patch = Map::new();
    if let Some(value) = options.recall {
        patch.insert("recall".into(), json!(value));
    }
    if let Some(value) = options.max_recall_chars {
        patch.insert("maxRecallChars".into(), json!(value));
    }
    if let Some(value) = options.max_recall_entries {
        patch.insert("maxRecallEntries".into(), json!(value));
    }
    let patch = validate_settings(&Value::Object(patch))?;
    let mut settings = base.clone();
    if let Some(value) = patch.recall {
        settings.recall = value;
    }
    if let Some(value) = patch.max_recall_chars {
        settings.max_recall_chars = value;
    }
    if let Some(value) = patch.max_recall_entries {
        settings.max_recall_entries = value;
    }
    Ok(settings)
}

// Return None only when native freshness cannot decode this URI and returns Unknown.
// Never resolve an input path on disk. Replay paths are generated numeric names.
fn file_key(uri: &str, project_path: Option<&str>) -> Result<Option<String>, String> {
    if !uri.starts_with("file:") {
        return Ok(None);
    }
    let Some(path) = url::Url::parse(uri)
        .ok()
        .and_then(|url| url.to_file_path().ok())
    else {
        return Ok(None);
    };
    if let Some(relative) = project_path {
        // Portable, unambiguous subset of native projectPath. In particular,
        // native lexical containment is not a sufficient sandbox boundary.
        if relative.is_empty()
            || relative.contains(['\\', ':', '\0'])
            || relative
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
            || !Path::new(relative)
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
        {
            return Err(format!("unsupported projectPath {relative:?}; export only safe slash-separated relative paths"));
        }
        return Ok(Some(format!("project:{relative}")));
    }
    Ok(Some(format!("uri:{}", path.to_string_lossy())))
}

fn write_value(path: &Path, value: &impl Serialize) -> Result<(), String> {
    std::fs::create_dir_all(path.parent().ok_or("scratch file has no parent")?)
        .map_err(|error| error.to_string())?;
    std::fs::write(
        path,
        serde_json::to_vec(value).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

struct Replay {
    _scratch: tempfile::TempDir,
    service: MemoryService,
    file_paths: BTreeMap<String, String>,
}

impl Replay {
    fn new(fixture: &Fixture) -> Result<Self, String> {
        if fixture.schema != FIXTURE_SCHEMA {
            return Err("unsupported fixture schema".into());
        }
        if fixture.lineage.run_id.trim().is_empty()
            || fixture.lineage.env_id.trim().is_empty()
            || fixture.lineage.capture_stage.trim().is_empty()
            || fixture.lineage.code_revision.trim().is_empty()
        {
            return Err("all lineage fields must be nonempty".into());
        }
        validate_document(
            &serde_json::to_value(&fixture.memory).map_err(|e| e.to_string())?,
            &fixture.project.id,
        )?;
        if !regex::Regex::new(r"^project_[A-Za-z0-9_-]{1,80}$")
            .unwrap()
            .is_match(&fixture.project.id)
            || !regex::Regex::new(r"^[a-f0-9-]{36}$")
                .unwrap()
                .is_match(&fixture.host_id)
        {
            return Err("invalid native project/host identity".into());
        }
        let scratch = tempfile::Builder::new()
            .prefix("memory-recall-probe-")
            .tempdir()
            .map_err(|e| e.to_string())?;
        let agent_dir = scratch.path().join("agent");
        let dir = agent_dir.join("project");
        let workspace = scratch.path().join("workspace");
        let session = scratch.path().join("session");
        std::fs::create_dir_all(&workspace).map_err(|e| e.to_string())?;
        write_value(&dir.join("harness_state.json"), &fixture.memory)?;
        write_value(
            &agent_dir.join("settings.json"),
            &json!({"memory":fixture.global_memory_settings}),
        )?;
        write_value(&dir.join("settings.json"), &fixture.project_memory_settings)?;
        if let Some(state) = &fixture.global_harness {
            write_value(&agent_dir.join("harness/harness_state.json"), state)?;
        }
        if let Some(state) = &fixture.session_harness {
            write_value(&session.join("harness/harness_state.json"), state)?;
        }
        if let Some(cache) = &fixture.shared_cache {
            write_value(&dir.join("shared.json"), cache)?;
        }
        // Public fields avoid MemoryStore::new's host identity persistence and
        // MemoryService::new's project binding. Every operational path is ours.
        let store = MemoryStore {
            agent_dir: agent_dir.to_string_lossy().into_owned(),
            project: ProjectIdentity {
                root: workspace.to_string_lossy().into_owned(),
                ..fixture.project.clone()
            },
            dir: dir.to_string_lossy().into_owned(),
            path: dir
                .join("harness_state.json")
                .to_string_lossy()
                .into_owned(),
            host_id: fixture.host_id.clone(),
        };
        let service = MemoryService {
            jobs: MemoryJobs::new(store.clone()),
            sharing: MemorySharing::new(store.clone()),
            session_artifact_dir: fixture
                .session_harness
                .as_ref()
                .map(|_| session.to_string_lossy().into_owned()),
            store,
        };
        let mut file_paths = BTreeMap::new();
        for (index, file) in fixture.files.iter().enumerate() {
            let key = file_key(&file.uri, file.project_path.as_deref())?
                .ok_or("exported file record must have a decodable file: URI")?;
            let relative = format!("source_{index}");
            if file_paths.insert(key, relative.clone()).is_some() {
                return Err("duplicate exported file location".into());
            }
            match &file.content {
                Value::Null => {}
                Value::String(content) => {
                    std::fs::write(workspace.join(&relative), content).map_err(|e| e.to_string())?
                }
                _ => {
                    return Err(
                        "exported file content must be UTF-8 string or explicit null".into(),
                    )
                }
            }
        }
        let replay = Self {
            _scratch: scratch,
            service,
            file_paths,
        };
        // Check currently visible sources up front. Query-specific shared matches
        // are checked again because native deduplication depends on the query.
        for hit in replay.service.search("", true) {
            let _ = replay.frozen_hit(&hit)?;
        }
        Ok(replay)
    }

    fn frozen_hit(&self, hit: &MemoryHit) -> Result<MemoryHit, String> {
        let mut frozen = hit.clone();
        for source in &mut frozen.sources {
            if source.origin != MemoryOrigin::File {
                continue;
            }
            let Some(uri) = &source.uri else {
                continue;
            };
            let Some(key) = file_key(uri, source.project_path.as_deref())? else {
                continue;
            };
            let relative = self.file_paths.get(&key)
                .ok_or_else(|| format!("missing frozen file export for source {} ({key}); use content:null only if missing at capture", source.id))?;
            // Native freshness uses projectPath; native recall labels contain
            // only source id/uri. Preserve those labels and their exact budget.
            source.project_path = Some(relative.clone());
        }
        Ok(frozen)
    }

    fn query(&self, query: &Query) -> Result<Value, String> {
        let base = self.service.store.settings();
        let settings = effective_settings(&base, &query.options)?;
        let mut hits = self
            .service
            .search(&query.query, query.options.include_inactive);
        let mut selected = Vec::new();
        let root = Some(self.service.store.project.root.as_str());
        for hit in &mut hits {
            let frozen = self.frozen_hit(hit)?;
            hit.freshness = freshness(&frozen.sources, root);
            if query.options.scope.is_none() || query.options.scope == Some(hit.scope) {
                selected.push(frozen);
            }
        }
        let recall = recall_memory(&selected, &settings, root);
        Ok(json!({
            "query":query,
            "base_effective_settings":base,
            "effective_settings":settings,
            "ordered_hits":hits,
            "selected_ids":selected.iter().map(|hit| &hit.id).collect::<Vec<_>>(),
            "injected_ids":recall.ids,
            "recall_chars":recall.chars,
            "recall_text":recall.text,
            "injection_mode":"render_only_no_session",
            "ranking":"native_lexical_no_distillation_no_rerank",
        }))
    }
}

fn implementation_hashes() -> Value {
    json!({
        "probe":hash(include_str!("memory_recall_probe.rs")),
        "search":hash(include_str!("../src/core/memory/search.rs")),
        "service":hash(include_str!("../src/core/memory/service.rs")),
        "store":hash(include_str!("../src/core/memory/store.rs")),
        "evidence":hash(include_str!("../src/core/memory/evidence.rs")),
        "sharing":hash(include_str!("../src/core/memory/sharing.rs")),
        "refinement":hash(include_str!("../src/core/refinement/refinement.rs")),
    })
}

fn run(fixture_path: &Path, queries_path: &Path, output: &mut impl Write) -> Result<(), String> {
    let fixture_raw = std::fs::read_to_string(fixture_path).map_err(|e| format!("fixture: {e}"))?;
    let queries_raw = std::fs::read_to_string(queries_path).map_err(|e| format!("queries: {e}"))?;
    let fixture: Fixture =
        serde_json::from_str(&fixture_raw).map_err(|e| format!("fixture JSON: {e}"))?;
    let queries = parse_queries(&queries_raw)?;
    let replay = Replay::new(&fixture)?;
    let hashes = implementation_hashes();
    let fixture_sha256 = hash(&fixture_raw);
    let queries_sha256 = hash(&queries_raw);
    for line in queries {
        let mut result = replay.query(&line.query)?;
        result["schema"] = json!(RESULT_SCHEMA);
        result["lineage"] = json!({
            "fixture":fixture.lineage,
            "original_project":fixture.project,
            "host_id":fixture.host_id,
            "memory_revision":fixture.memory.memory.revision,
            "fixture_path":fixture_path.to_string_lossy(),
            "fixture_sha256":fixture_sha256,
            "queries_path":queries_path.to_string_lossy(),
            "queries_sha256":queries_sha256,
            "query_line":line.number,
            "query_line_sha256":line.sha256,
            "implementation_sha256":hashes,
            "package_version":env!("CARGO_PKG_VERSION"),
            "platform":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH},
        });
        serde_json::to_writer(&mut *output, &result).map_err(|e| e.to_string())?;
        output.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("usage: memory_recall_probe <fixture.json> <queries.jsonl>\nOutputs JSONL to stdout. Offline native lexical search/recall; no live profile.\nSee the example's module documentation for the strict exported-fixture schema.");
        return;
    }
    if args.len() != 2 {
        eprintln!("usage: memory_recall_probe <fixture.json> <queries.jsonl>");
        std::process::exit(2);
    }
    if let Err(error) = run(
        Path::new(&args[0]),
        Path::new(&args[1]),
        &mut std::io::stdout().lock(),
    ) {
        eprintln!("memory_recall_probe: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_coding_agent::core::memory::evidence::MemorySource;
    use pi_coding_agent::core::memory::search::{search_memory, MemoryFreshness};
    use pi_coding_agent::core::memory::store::{default_memory_settings, empty_document};
    use pi_coding_agent::core::refinement::refinement::HarnessEntry;

    fn fixture() -> Fixture {
        Fixture {
            schema: FIXTURE_SCHEMA.into(),
            lineage: Lineage {
                run_id: "synthetic".into(),
                env_id: "env".into(),
                capture_stage: "synthetic".into(),
                code_revision: "fixture-only".into(),
            },
            project: ProjectIdentity {
                id: "project_probe".into(),
                root: "never-open-this-original-root".into(),
                aliases: vec![],
            },
            host_id: "00000000-0000-0000-0000-000000000001".into(),
            memory: empty_document("project_probe"),
            global_memory_settings: Map::new(),
            project_memory_settings: Map::new(),
            global_harness: None,
            session_harness: None,
            shared_cache: None,
            files: vec![],
        }
    }

    fn entry(id: &str, title: &str, content: &str) -> HarnessEntry {
        serde_json::from_value(json!({
            "id":id,"kind":"memory","title":title,"content":content,"path":"general",
            "metadata":{"projectId":"project_probe"},"reference":{},"arguments":{},"source":"refine",
            "created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z","version":1
        })).unwrap()
    }

    fn insert(document: &mut MemoryDocument, entry: HarnessEntry) {
        document
            .entries
            .get_mut("memory")
            .unwrap()
            .insert(entry.id.clone(), entry);
    }

    fn query(text: &str) -> Query {
        Query {
            qid: "q1".into(),
            variant: "base".into(),
            query: text.into(),
            options: Options::default(),
        }
    }

    fn ids(value: &Value, field: &str) -> Vec<String> {
        value[field]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn lexical_idf_title_weight_and_query_terms_are_native() {
        let mut data = fixture();
        insert(&mut data.memory, entry("a", "rare", "filler"));
        insert(&mut data.memory, entry("b", "filler", ""));
        insert(&mut data.memory, entry("c", "other", "filler"));
        insert(&mut data.memory, entry("d", "other", "filler"));
        let replay = Replay::new(&data).unwrap();
        let result = replay.query(&query("RARE filler rare")).unwrap();
        assert_eq!(
            ids(&result, "selected_ids"),
            [
                "project:memory:a",
                "project:memory:b",
                "project:memory:c",
                "project:memory:d"
            ]
        );
        let rare_idf = (5.0f64 / 2.0).ln() + 1.0;
        let expected = (3.0 * rare_idf + 1.0) / (rare_idf + 1.0);
        assert!((result["ordered_hits"][0]["score"].as_f64().unwrap() - expected).abs() < 1e-12);
        assert_eq!(
            result["ordered_hits"][0]["matched"],
            json!(["rare", "filler"])
        );
        let native = search_memory(&replay.service.store, "RARE filler rare", &[], false);
        assert_eq!(
            serde_json::to_value(native).unwrap(),
            result["ordered_hits"]
        );
        let empty_query = replay.query(&query("")).unwrap();
        assert_eq!(ids(&empty_query, "selected_ids").len(), 4);
        assert!(empty_query["ordered_hits"]
            .as_array()
            .unwrap()
            .iter()
            .all(|hit| hit["score"] == 0.0 && hit["matched"] == json!([])));
        let no_match = replay.query(&query("unmatched-token")).unwrap();
        assert!(ids(&no_match, "selected_ids").is_empty());
        assert!(ids(&no_match, "injected_ids").is_empty());
    }

    #[test]
    fn scope_order_host_project_isolation_and_shared_dedup_are_native() {
        let mut data = fixture();
        insert(&mut data.memory, entry("project", "needle", "local"));
        let mut host = entry("host", "needle", "host");
        host.metadata.insert("hostId".into(), json!(data.host_id));
        insert(&mut data.memory, host);
        let mut session = empty_document(&data.project.id);
        insert(&mut session, entry("session", "needle", "session"));
        data.session_harness = Some(session.harness());
        let mut global = empty_document(&data.project.id);
        let mut generic = entry("global", "needle", "global");
        generic.metadata.clear();
        insert(&mut global, generic);
        let mut other_project = entry("other-project", "needle", "hidden");
        other_project
            .metadata
            .insert("projectId".into(), json!("project_other"));
        insert(&mut global, other_project);
        let mut other_host = entry("other-host", "needle", "hidden");
        other_host
            .metadata
            .insert("hostId".into(), json!("other-host"));
        insert(&mut global, other_host);
        data.global_harness = Some(global.harness());
        let mut shared = empty_document(&data.project.id);
        insert(&mut shared, entry("project", "needle", "shared duplicate"));
        insert(&mut shared, entry("shared", "needle", "shared"));
        let endpoint = "https://frozen.invalid";
        data.global_memory_settings.insert(
            "shared".into(),
            json!({"url":endpoint,"tokenFile":"never-open-token"}),
        );
        data.shared_cache = Some(SharedCache {
            schema: 1.0,
            url: endpoint.into(),
            revision: 0,
            state: shared,
            pending: vec![],
            connected: false,
            error: None,
        });
        let replay = Replay::new(&data).unwrap();
        let result = replay.query(&query("needle")).unwrap();
        assert_eq!(
            ids(&result, "selected_ids"),
            [
                "session:memory:session",
                "project:memory:project",
                "host:memory:host",
                "shared:memory:shared",
                "global:memory:global"
            ]
        );
        let mut scoped = query("needle");
        scoped.options.scope = Some(MemoryScope::Host);
        let scoped_result = replay.query(&scoped).unwrap();
        assert_eq!(scoped_result["ordered_hits"], result["ordered_hits"]);
        assert_eq!(ids(&scoped_result, "selected_ids"), ["host:memory:host"]);
        data.global_memory_settings.clear();
        let detached = Replay::new(&data).unwrap().query(&query("needle")).unwrap();
        assert!(!ids(&detached, "selected_ids")
            .iter()
            .any(|id| id.starts_with("shared:")));
    }

    #[test]
    fn inactive_filter_and_settings_precedence_are_native() {
        let mut data = fixture();
        insert(&mut data.memory, entry("active", "needle", "live"));
        for (id, marker) in [
            ("a", json!({"supersededBy":null})),
            ("b", json!({"status":"superseded"})),
            ("c", json!({"detached":false})),
        ] {
            let mut inactive = entry(id, "needle", "inactive");
            inactive
                .metadata
                .extend(marker.as_object().unwrap().clone());
            insert(&mut data.memory, inactive);
        }
        data.global_memory_settings =
            json!({"recall":false,"maxRecallEntries":4,"maxRecallChars":1200})
                .as_object()
                .unwrap()
                .clone();
        data.project_memory_settings = json!({"recall":true,"maxRecallEntries":1})
            .as_object()
            .unwrap()
            .clone();
        let replay = Replay::new(&data).unwrap();
        let result = replay.query(&query("needle")).unwrap();
        assert_eq!(ids(&result, "selected_ids"), ["project:memory:active"]);
        assert_eq!(result["effective_settings"]["maxRecallChars"], 1200);
        assert_eq!(result["effective_settings"]["maxRecallEntries"], 1);
        assert_eq!(result["effective_settings"]["recall"], true);
        let mut all = query("needle");
        all.options.include_inactive = true;
        let result = replay.query(&all).unwrap();
        assert_eq!(ids(&result, "selected_ids").len(), 4);
        assert_eq!(ids(&result, "injected_ids").len(), 1);
    }

    #[test]
    fn native_budget_truncation_unicode_and_disabled_recall() {
        let mut data = fixture();
        insert(
            &mut data.memory,
            entry("a", "needle", &"東京é🙂 ".repeat(1500)),
        );
        insert(&mut data.memory, entry("b", "needle", "second"));
        let replay = Replay::new(&data).unwrap();
        let mut request = query("needle");
        request.options.max_recall_chars = Some(800);
        request.options.max_recall_entries = Some(1);
        let result = replay.query(&request).unwrap();
        assert_eq!(ids(&result, "selected_ids").len(), 2);
        assert_eq!(ids(&result, "injected_ids"), ["project:memory:a"]);
        let text = result["recall_text"].as_str().unwrap();
        assert!(text.contains("[read entry for full text]"));
        assert_eq!(
            text.chars().count(),
            result["recall_chars"].as_u64().unwrap() as usize
        );
        assert!(text.chars().count() <= 800);
        let native = replay
            .service
            .render_recall(&replay.service.search("needle", false));
        let baseline = replay.query(&query("needle")).unwrap();
        assert_eq!(baseline["recall_text"], native.text);
        assert_eq!(baseline["injected_ids"], json!(native.ids));
        for options in [
            Options {
                recall: Some(false),
                ..Default::default()
            },
            Options {
                max_recall_chars: Some(0),
                ..Default::default()
            },
            Options {
                max_recall_entries: Some(0),
                ..Default::default()
            },
            Options {
                max_recall_chars: Some(80),
                ..Default::default()
            },
        ] {
            request.options = options;
            let empty = replay.query(&request).unwrap();
            assert!(ids(&empty, "injected_ids").is_empty());
            assert_eq!(empty["recall_chars"], 0);
            assert_eq!(empty["recall_text"], "");
        }
    }

    #[test]
    fn frozen_freshness_ignores_live_files_preserves_labels_and_native_budget() {
        let original = tempfile::tempdir().unwrap();
        let file = original.path().join("live.txt");
        std::fs::write(&file, "live changed after capture").unwrap();
        let uri = url::Url::from_file_path(&file).unwrap().to_string();
        let mut data = fixture();
        let mut expected_sources = Vec::new();
        for (id, content, digest) in [
            ("a-current", Some("frozen"), hash("frozen")),
            ("b-stale", Some("changed"), hash("old")),
            ("c-missing", None, hash("absent")),
        ] {
            let source = MemorySource {
                id: id.into(),
                origin: MemoryOrigin::File,
                sha256: digest,
                uri: Some(uri.clone()),
                revision: Some("frozen-revision".into()),
                project_path: Some(format!("{id}.txt")),
            };
            let mut item = entry(id, "needle", "saved note");
            item.metadata.insert("sources".into(), json!([source]));
            insert(&mut data.memory, item);
            data.files.push(FrozenFile {
                uri: uri.clone(),
                project_path: source.project_path.clone(),
                content: json!(content),
            });
            expected_sources.push(source);
        }
        insert(
            &mut data.memory,
            entry("d-unknown", "needle", "no file evidence"),
        );
        let replay = Replay::new(&data).unwrap();
        let result = replay.query(&query("needle")).unwrap();
        let hits: Vec<MemoryHit> = serde_json::from_value(result["ordered_hits"].clone()).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.freshness).collect::<Vec<_>>(),
            [
                MemoryFreshness::Current,
                MemoryFreshness::Stale,
                MemoryFreshness::Missing,
                MemoryFreshness::Unknown
            ]
        );
        assert_eq!(
            ids(&result, "injected_ids"),
            ["project:memory:a-current", "project:memory:d-unknown"]
        );
        assert_eq!(hits[0].sources[0], expected_sources[0]);
        assert!(result["recall_text"]
            .as_str()
            .unwrap()
            .contains(&serde_json::to_string(&uri).unwrap()));
        // Compare exact rendering against an independent native file-root cut.
        std::fs::write(original.path().join("a-current.txt"), "frozen").unwrap();
        std::fs::write(original.path().join("b-stale.txt"), "changed").unwrap();
        let native = recall_memory(
            &hits,
            &default_memory_settings(),
            Some(&original.path().to_string_lossy()),
        );
        assert_eq!(result["recall_text"], native.text);
        assert_eq!(result["injected_ids"], json!(native.ids));
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "live changed after capture"
        );
        // No-projectPath file references are also frozen, never opened live.
        data.memory
            .entries
            .get_mut("memory")
            .unwrap()
            .get_mut("a-current")
            .unwrap()
            .metadata
            .get_mut("sources")
            .unwrap()[0]
            .as_object_mut()
            .unwrap()
            .remove("projectPath");
        data.files[0].project_path = None;
        let without_relative = Replay::new(&data).unwrap().query(&query("needle")).unwrap();
        assert_eq!(without_relative["ordered_hits"][0]["freshness"], "current");
    }

    #[test]
    fn incomplete_unsafe_and_ambiguous_file_exports_fail_closed() {
        let scratch = tempfile::tempdir().unwrap();
        let uri = url::Url::from_file_path(scratch.path().join("do-not-read"))
            .unwrap()
            .to_string();
        let mut data = fixture();
        let mut item = entry("a", "needle", "body");
        item.metadata.insert("sources".into(),json!([{"id":"source","origin":"file","sha256":hash("x"),"uri":uri,"projectPath":"safe.txt"}]));
        insert(&mut data.memory, item);
        assert!(Replay::new(&data)
            .err()
            .unwrap()
            .contains("missing frozen file export"));
        for path in [
            "../outside",
            "a/../../outside",
            "/absolute",
            "C:/outside",
            "a\\b",
            "./a",
            "",
        ] {
            assert!(file_key(&uri, Some(path)).is_err(), "{path}");
        }
        let exported = FrozenFile {
            uri: uri.clone(),
            project_path: Some("safe.txt".into()),
            content: Value::Null,
        };
        data.files = vec![exported.clone(), exported];
        assert!(Replay::new(&data).err().unwrap().contains("duplicate"));
        assert!(serde_json::from_value::<FrozenFile>(json!({"uri":uri})).is_err());
    }

    #[test]
    fn query_schema_rejects_gold_unknown_fields_duplicate_keys_and_invalid_limits() {
        for raw in [
            r#"{"qid":"q","query":"needle","answer":"gold"}"#,
            r#"{"qid":"q","query":"needle","options":{"rerank":true}}"#,
            r#"{"qid":"q","query":"needle","options":{"max_recall_chars":-1}}"#,
            r#"{"qid":"q","query":"needle","options":{"max_recall_entries":51}}"#,
            r#"{"qid":"q","query":"needle","options":{"max_recall_chars":1.5}}"#,
            r#"{"qid":"","query":"needle"}"#,
        ] {
            assert!(parse_queries(raw).is_err(), "{raw}");
        }
        let row = r#"{"qid":"q","query":"needle"}"#;
        assert!(parse_queries(&format!("{row}\n{row}")).is_err());
        assert!(parse_queries("\n\n").is_err());
        let rows = parse_queries(&format!(
            "\n{row}\n{{\"qid\":\"q\",\"variant\":\"budget\",\"query\":\"needle\"}}"
        ))
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].number, 2);
        assert_eq!(rows[0].sha256, hash(row));
    }

    fn tree(path: &Path) -> BTreeMap<String, Vec<u8>> {
        fn visit(root: &Path, dir: &Path, result: &mut BTreeMap<String, Vec<u8>>) {
            for item in std::fs::read_dir(dir).unwrap() {
                let path = item.unwrap().path();
                if path.is_dir() {
                    visit(root, &path, result);
                } else {
                    result.insert(
                        path.strip_prefix(root)
                            .unwrap()
                            .to_string_lossy()
                            .into_owned(),
                        std::fs::read(path).unwrap(),
                    );
                }
            }
        }
        let mut result = BTreeMap::new();
        visit(path, path, &mut result);
        result
    }

    #[test]
    fn cli_core_is_reproducible_and_does_not_mutate_inputs_or_native_scratch() {
        let input = tempfile::tempdir().unwrap();
        let mut data = fixture();
        insert(&mut data.memory, entry("a", "needle", "body"));
        data.project.root = input
            .path()
            .join("original-project-must-not-be-created")
            .to_string_lossy()
            .into_owned();
        let fixture_path = input.path().join("fixture.json");
        let queries_path = input.path().join("queries.jsonl");
        write_value(&fixture_path, &data).unwrap();
        std::fs::write(&queries_path, "{\"qid\":\"q\",\"query\":\"needle\"}\n").unwrap();
        let before = tree(input.path());
        let mut first = Vec::new();
        run(&fixture_path, &queries_path, &mut first).unwrap();
        let mut second = Vec::new();
        run(&fixture_path, &queries_path, &mut second).unwrap();
        assert_eq!(first, second);
        assert_eq!(before, tree(input.path()));
        assert!(!Path::new(&data.project.root).exists());
        let result: Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(result["schema"], RESULT_SCHEMA);
        assert_eq!(
            result["lineage"]["fixture_sha256"],
            hash(&std::fs::read_to_string(&fixture_path).unwrap())
        );
        assert_eq!(result["lineage"]["query_line"], 1);
        assert_eq!(
            result["lineage"]["implementation_sha256"]["search"],
            hash(include_str!("../src/core/memory/search.rs"))
        );
        let replay = Replay::new(&data).unwrap();
        let scratch_before = tree(replay._scratch.path());
        let scratch_path = replay._scratch.path().to_path_buf();
        let _ = replay.query(&query("needle")).unwrap();
        assert_eq!(scratch_before, tree(replay._scratch.path()));
        assert!(!Path::new(&replay.service.store.agent_dir)
            .join("memory/host-id.json")
            .exists());
        drop(replay);
        assert!(!scratch_path.exists());
    }
}
