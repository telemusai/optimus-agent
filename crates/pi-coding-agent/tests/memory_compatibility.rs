//! Legacy memory compatibility through public APIs, without a model or daemon.
//!
//! Schema evidence: pre-campaign release commit
//! `4baed27e709d67f20a8edb5d0bd09e40bd3a7678` (2026-10-06):
//! - core/memory/store.rs:28-90,144-180: settings and schema-1 snapshots.
//! - core/memory/evidence.rs:93-105: source IDs, digests and optional provenance.
//! - core/refinement/refinement.rs:220-321: entries and refinement history.
//! Paths above are under crates/pi-coding-agent/src. The fixtures reconstruct
//! those JSON shapes with synthetic values, not current struct serialization or
//! real user data. They intentionally omit the later import/recall settings.

use std::collections::BTreeSet;
use std::path::Path;

use pi_coding_agent::core::memory::evidence::{hash, MemoryOrigin, MemorySource};
use pi_coding_agent::core::memory::project::ProjectIdentity;
use pi_coding_agent::core::memory::search::{
    freshness, recall_memory, search_memory, sources, MemoryFreshness, MemoryHit, MemoryScope,
    SearchCorpus,
};
use pi_coding_agent::core::memory::service::MemoryService;
use pi_coding_agent::core::memory::store::{
    default_memory_settings, memory_source_from_value, validate_document, validate_settings,
    ApplyOptions, MemoryDocument, MemorySettings, MemoryStore,
};
use pi_coding_agent::core::refinement::refinement::{
    get_global_harness_state_dir, load_harness_state, normalize_refinement_proposal, HarnessScope,
};
use serde_json::{json, Value};
use tempfile::TempDir;

const LEGACY_SETTINGS: &str = r#"{
    "recall": true,
    "learning": true,
    "maxRecallChars": 6000,
    "maxRecallEntries": 6,
    "maxExtractionTokens": 4096,
    "maxImportBytes": 33554432,
    "maxImportChunkChars": 40000,
    "maxImportChunksPerRun": 4
}"#;
const LEGACY_TIME: &str = "2026-10-01T00:00:00.000Z";
const EVIDENCE_BODY: &str = "Synthetic legacy evidence.\n";

struct Fixture {
    root: TempDir,
}

impl Fixture {
    fn new() -> Self {
        Self {
            root: tempfile::tempdir().expect("isolated compatibility directory"),
        }
    }

    fn store(&self, name: &str) -> MemoryStore {
        let project_root = self.root.path().join(name);
        std::fs::create_dir_all(&project_root).unwrap();
        MemoryStore::new(
            &self.root.path().join("agent").to_string_lossy(),
            ProjectIdentity {
                id: format!("project_{name}"),
                root: project_root.to_string_lossy().into_owned(),
                aliases: Vec::new(),
            },
        )
        .expect("private memory store")
    }

    fn service(&self, name: &str) -> MemoryService {
        let project_root = self.root.path().join(name);
        std::fs::create_dir_all(&project_root).unwrap();
        MemoryService::new(
            &project_root.to_string_lossy(),
            &self.root.path().join("agent").to_string_lossy(),
            None,
        )
        .expect("private memory service")
    }
}

fn write_fixture(path: &Path, value: &Value) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn read_fixture(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn legacy_user_source() -> Value {
    json!({"id": "legacy_message", "origin": "user", "sha256": hash(EVIDENCE_BODY)})
}

fn legacy_file_source(store: &MemoryStore, name: &str, body: &str) -> Value {
    let path = Path::new(&store.project.root).join(name);
    std::fs::write(&path, body).unwrap();
    json!({
        "id": format!("legacy_file_{name}"),
        "origin": "file",
        "sha256": hash(body),
        "uri": url::Url::from_file_path(&path).unwrap().to_string(),
        "revision": "synthetic_revision_1",
        "projectPath": name
    })
}

fn legacy_entry(id: &str, content: &str, metadata: Value) -> Value {
    json!({
        "id": id, "kind": "memory", "title": "Legacy beacon", "content": content,
        "path": "general", "scope": "local", "reference": {}, "arguments": {},
        "metadata": metadata, "source": "refine",
        "created_at": LEGACY_TIME, "updated_at": LEGACY_TIME, "version": 1
    })
}

fn project_entry(store: &MemoryStore, id: &str, content: &str, refs: Vec<Value>) -> Value {
    legacy_entry(
        id,
        content,
        json!({
            "projectId": store.project.id, "status": "current", "sources": refs,
            "sourceIds": ["legacy_message"], "projectReusable": true
        }),
    )
}

fn legacy_snapshot(store: &MemoryStore, entries: &[Value]) -> Value {
    let mut bucket = serde_json::Map::new();
    let mut changes = Vec::new();
    let mut edits = Vec::new();
    for entry in entries {
        let id = entry["id"].as_str().unwrap();
        bucket.insert(id.to_string(), entry.clone());
        changes.push(format!("create memory:{id}"));
        edits.push(json!({
            "action": "create", "kind": "memory", "id": id,
            "title": entry["title"], "content": entry["content"], "path": entry["path"],
            "metadata": entry["metadata"], "after": entry, "applied": true
        }));
    }
    json!({
        "schema": 1.0,
        "entries": {"memory": bucket, "prompt": {}, "skill": {}, "subagent": {}},
        "refinements": [{
            "id": "legacy_create", "trigger": "Legacy fixture", "changes": changes,
            "evidence": "Synthetic evidence", "outcome": "Retain old state",
            "created_at": LEGACY_TIME
        }],
        "memory": {
            "schema": 1.0, "projectId": store.project.id, "revision": 1,
            "history": [{
                "id": "legacy_create", "summary": "Legacy fixture",
                "rationale": "Synthetic evidence", "expectedOutcome": "Retain old state",
                "appliedEdits": edits, "harnessStatePath": store.path, "scope": "local"
            }],
            "events": {"legacy_create": hash("synthetic legacy delivery receipt")}
        }
    })
}

fn seed(store: &MemoryStore, entries: &[Value]) -> Value {
    let raw = legacy_snapshot(store, entries);
    write_fixture(Path::new(&store.path), &raw);
    raw
}

fn hit_ids(hits: &[MemoryHit]) -> BTreeSet<String> {
    hits.iter().map(|hit| hit.id.clone()).collect()
}

#[test]
fn legacy_settings_deserialize_with_unchanged_defaults_and_round_trip() {
    let old_defaults: Value = serde_json::from_str(LEGACY_SETTINGS).unwrap();
    let defaults: MemorySettings = serde_json::from_value(old_defaults.clone()).unwrap();
    assert_eq!(defaults, default_memory_settings());
    assert!(!defaults.recall_query_distillation);
    assert!(!defaults.recall_rerank);
    assert_eq!(defaults.import_instructions, None);
    assert_eq!(defaults.shared, None);

    let mut configured = old_defaults.clone();
    configured["recall"] = json!(false);
    configured["learning"] = json!(false);
    configured["maxRecallChars"] = json!(7500);
    configured["maxRecallEntries"] = json!(9);
    configured["maxExtractionTokens"] = json!(2048);
    configured["maxImportBytes"] = json!(65536);
    configured["maxImportChunkChars"] = json!(12000);
    configured["maxImportChunksPerRun"] = json!(2);
    configured["shared"] = json!({
        "url": "https://memory.invalid", "tokenFile": "synthetic-token-not-read"
    });
    for legacy in [old_defaults, configured] {
        let settings: MemorySettings = serde_json::from_value(legacy.clone()).unwrap();
        let encoded = serde_json::to_value(&settings).unwrap();
        for (key, value) in legacy.as_object().unwrap() {
            assert_eq!(&encoded[key], value, "legacy field {key}");
        }
        assert_eq!(encoded["recallQueryDistillation"], false);
        assert_eq!(encoded["recallRerank"], false);
        assert!(encoded.get("importInstructions").is_none());
        assert_eq!(
            serde_json::from_value::<MemorySettings>(encoded).unwrap(),
            settings
        );
    }
}

#[tokio::test]
async fn legacy_partial_settings_merge_without_resetting_inherited_values() {
    let fixture = Fixture::new();
    let store = fixture.store("alpha");
    let sibling = fixture.store("beta");
    let global_path = Path::new(&store.agent_dir).join("settings.json");
    let global = json!({
        "theme": "synthetic-theme",
        "memory": {
            "recall": false, "learning": false, "maxRecallChars": 7300,
            "maxExtractionTokens": 3072, "maxImportBytes": 65536,
            "maxImportChunkChars": 12000, "maxImportChunksPerRun": 2,
            "shared": {"url": "https://memory.invalid", "tokenFile": "not-read"}
        }
    });
    write_fixture(&global_path, &global);
    let global_bytes = std::fs::read(&global_path).unwrap();
    let inherited = store.settings();
    assert!(!inherited.recall);
    assert!(!inherited.learning);
    assert_eq!(inherited.max_recall_chars, 7300);
    assert_eq!(inherited.max_recall_entries, 6);
    assert_eq!(inherited.max_extraction_tokens, 3072);
    assert_eq!(inherited.max_import_bytes, 65536);
    assert_eq!(inherited.max_import_chunk_chars, 12000);
    assert_eq!(inherited.max_import_chunks_per_run, 2);
    assert_eq!(inherited.shared_config().unwrap().token_file, "not-read");

    let expected = MemorySettings {
        recall: true,
        max_recall_entries: 3,
        ..inherited.clone()
    };
    assert_eq!(
        store
            .configure(&json!({"maxRecallEntries": 3}))
            .await
            .unwrap(),
        MemorySettings {
            recall: false,
            ..expected.clone()
        }
    );
    assert_eq!(
        store.configure(&json!({"recall": true})).await.unwrap(),
        expected
    );
    assert_eq!(store.configure(&json!({})).await.unwrap(), expected);
    assert_eq!(fixture.store("alpha").settings(), expected);
    assert_eq!(sibling.settings(), inherited);
    let local_path = Path::new(&store.dir).join("settings.json");
    assert_eq!(
        read_fixture(&local_path),
        json!({"maxRecallEntries": 3, "recall": true})
    );

    let cleared = store
        .configure(&json!({"shared": null, "maxRecallChars": 0}))
        .await
        .unwrap();
    assert_eq!(cleared.shared, Some(None));
    assert!(cleared.shared_config().is_none());
    assert_eq!(cleared.max_recall_chars, 0);
    assert!(!cleared.learning);
    assert_eq!(cleared.max_import_chunks_per_run, 2);
    assert_eq!(read_fixture(&local_path)["shared"], Value::Null);
    assert_eq!(fixture.store("alpha").settings(), cleared);
    assert_eq!(std::fs::read(global_path).unwrap(), global_bytes);
    assert_eq!(sibling.settings(), inherited);
}

#[tokio::test]
async fn legacy_invalid_settings_do_not_replace_a_valid_partial_file() {
    let fixture = Fixture::new();
    let store = fixture.store("alpha");
    let configured = store.configure(&json!({"learning": false})).await.unwrap();
    let path = Path::new(&store.dir).join("settings.json");
    let original = std::fs::read(&path).unwrap();
    for (key, low, high) in [
        ("maxRecallChars", 0_i64, 32000_i64),
        ("maxRecallEntries", 0, 50),
        ("maxExtractionTokens", 256, 32000),
        ("maxImportBytes", 1024, 134217728),
        ("maxImportChunkChars", 1000, 80000),
        ("maxImportChunksPerRun", 1, 64),
    ] {
        for valid in [low, high] {
            assert!(
                validate_settings(&json!({(key): valid})).is_ok(),
                "{key}={valid}"
            );
        }
        for invalid in [json!(low - 1), json!(high + 1), json!(1.5), json!("1")] {
            assert!(
                store.configure(&json!({(key): invalid})).await.is_err(),
                "{key}"
            );
            assert_eq!(std::fs::read(&path).unwrap(), original);
            assert_eq!(store.settings(), configured);
        }
    }
    assert!(store
        .configure(&json!({"learning": "false"}))
        .await
        .is_err());
    assert_eq!(std::fs::read(path).unwrap(), original);
}

#[test]
fn legacy_sources_hydrate_optional_provenance_without_changing_ids() {
    for (origin, expected) in [
        ("user", MemoryOrigin::User),
        ("assistant", MemoryOrigin::Assistant),
        ("tool", MemoryOrigin::Tool),
        ("derived", MemoryOrigin::Derived),
        ("file", MemoryOrigin::File),
    ] {
        let raw = json!({"id": "legacy_source", "origin": origin, "sha256": hash(EVIDENCE_BODY)});
        let source: MemorySource = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(source.origin, expected);
        assert_eq!(source.id, "legacy_source");
        assert_eq!(source.uri, None);
        assert_eq!(source.revision, None);
        assert_eq!(source.project_path, None);
        assert_eq!(memory_source_from_value(&raw), Some(source.clone()));
        assert_eq!(serde_json::to_value(source).unwrap(), raw);
    }
    let fixture = Fixture::new();
    let store = fixture.store("alpha");
    let raw = legacy_file_source(&store, "evidence.txt", EVIDENCE_BODY);
    let source: MemorySource = serde_json::from_value(raw.clone()).unwrap();
    assert_eq!(source.id, "legacy_file_evidence.txt");
    assert_eq!(source.revision.as_deref(), Some("synthetic_revision_1"));
    assert_eq!(source.project_path.as_deref(), Some("evidence.txt"));
    assert_eq!(source.sha256, hash(EVIDENCE_BODY));
    assert_eq!(memory_source_from_value(&raw), Some(source.clone()));
    assert_eq!(
        freshness(std::slice::from_ref(&source), Some(&store.project.root)),
        MemoryFreshness::Current
    );
    assert_eq!(serde_json::to_value(source).unwrap(), raw);
}

#[test]
fn legacy_snapshot_hydrates_entries_sources_revision_and_history() {
    let fixture = Fixture::new();
    let store = fixture.store("alpha");
    let refs = vec![
        legacy_user_source(),
        legacy_file_source(&store, "evidence.txt", EVIDENCE_BODY),
    ];
    let entry = project_entry(&store, "legacy_note", "Beacon legacy fact", refs.clone());
    let raw = seed(&store, &[entry.clone()]);
    let original = std::fs::read(&store.path).unwrap();
    let document = store.read().unwrap();
    assert_eq!(
        document,
        serde_json::from_value::<MemoryDocument>(raw.clone()).unwrap()
    );
    assert_eq!(document.memory.revision, 1);
    assert_eq!(document.memory.project_id, store.project.id);
    assert_eq!(
        serde_json::to_value(&document.entries["memory"]["legacy_note"]).unwrap(),
        entry
    );
    assert_eq!(
        serde_json::to_value(sources(&document.entries["memory"]["legacy_note"])).unwrap(),
        json!(refs)
    );
    let history = &document.memory.history[0];
    assert_eq!(history.id, "legacy_create");
    assert_eq!(history.scope, Some(HarnessScope::Local));
    assert_eq!(history.rollback_of, None);
    assert_eq!(history.harness_state_path, store.path);
    assert_eq!(history.applied_edits.len(), 1);
    assert_eq!(history.applied_edits[0].id, "legacy_note");
    assert_eq!(history.applied_edits[0].edit.action, "create");
    assert!(history.applied_edits[0].applied);
    assert!(history.applied_edits[0].before.is_none());
    assert_eq!(
        history.applied_edits[0].after.as_ref(),
        Some(&document.entries["memory"]["legacy_note"])
    );
    assert_eq!(
        document.refinements[0].changes,
        vec!["create memory:legacy_note"]
    );
    assert_eq!(
        document.memory.events["legacy_create"],
        hash("synthetic legacy delivery receipt")
    );
    let encoded = serde_json::to_value(&document).unwrap();
    assert_eq!(encoded, raw);
    assert_eq!(
        validate_document(&encoded, &store.project.id).unwrap(),
        document
    );
    assert_eq!(fixture.store("alpha").read().unwrap(), document);
    assert_eq!(
        std::fs::read(&store.path).unwrap(),
        original,
        "hydration must not rewrite old state"
    );

    let mut foreign = raw.clone();
    foreign["entries"]["memory"]["legacy_note"]["metadata"]["projectId"] = json!("project_other");
    assert!(validate_document(&foreign, &store.project.id).is_err());
    assert!(validate_document(&raw, "project_other").is_err());
    for invalid_revision in [json!(-1), json!(1.5), json!("1")] {
        let mut invalid = raw.clone();
        invalid["memory"]["revision"] = invalid_revision;
        assert!(validate_document(&invalid, &store.project.id).is_err());
    }
}

#[test]
fn legacy_project_and_global_scopes_keep_colliding_ids_separate() {
    let fixture = Fixture::new();
    let alpha = fixture.store("alpha");
    let beta = fixture.store("beta");
    seed(
        &alpha,
        &[project_entry(
            &alpha,
            "collision",
            "Beacon alpha only",
            vec![],
        )],
    );
    seed(
        &beta,
        &[project_entry(
            &beta,
            "collision",
            "Beacon beta only",
            vec![],
        )],
    );
    let mut global_entry = legacy_entry("collision", "Beacon global fact", json!({}));
    global_entry.as_object_mut().unwrap().remove("scope");
    let mut qualified = legacy_entry(
        "qualified",
        "Beacon alpha-qualified global fact",
        json!({"projectId": alpha.project.id}),
    );
    qualified.as_object_mut().unwrap().remove("scope");
    let dir = get_global_harness_state_dir(&alpha.agent_dir);
    let global_path = Path::new(&dir).join("harness_state.json");
    write_fixture(
        &global_path,
        &json!({
            "schema": 1.0, "entries": {"memory": {"collision": global_entry, "qualified": qualified}},
            "refinements": []
        }),
    );
    let state = load_harness_state(&dir, HarnessScope::Global);
    assert_eq!(
        state.entries["memory"]["collision"].scope,
        Some(HarnessScope::Global)
    );
    let additional = [
        SearchCorpus {
            state,
            scope: MemoryScope::Global,
        },
        SearchCorpus {
            state: beta.read().unwrap().harness(),
            scope: MemoryScope::Shared,
        },
    ];
    let alpha_hits = search_memory(&alpha, "beacon", &additional, false);
    assert_eq!(
        hit_ids(&alpha_hits),
        BTreeSet::from([
            "project:memory:collision".to_string(),
            "global:memory:collision".to_string(),
            "global:memory:qualified".to_string(),
        ])
    );
    assert!(alpha_hits
        .iter()
        .all(|hit| !hit.entry.content.contains("beta only")));
    let beta_hits = search_memory(&beta, "beacon", &additional[..1], false);
    assert_eq!(
        hit_ids(&beta_hits),
        BTreeSet::from([
            "project:memory:collision".to_string(),
            "global:memory:collision".to_string(),
        ])
    );
    assert!(beta_hits
        .iter()
        .all(|hit| !hit.entry.content.contains("alpha")));
    assert_eq!(
        alpha_hits
            .iter()
            .find(|hit| hit.id == "project:memory:collision")
            .unwrap()
            .entry
            .content,
        "Beacon alpha only"
    );
    assert_eq!(
        beta_hits
            .iter()
            .find(|hit| hit.id == "project:memory:collision")
            .unwrap()
            .entry
            .content,
        "Beacon beta only"
    );
    let reopened = fixture.store("alpha");
    assert_eq!(
        hit_ids(&search_memory(&reopened, "beacon", &additional, false)),
        hit_ids(&alpha_hits)
    );
}

#[test]
fn legacy_inactive_and_stale_notes_remain_inspectable_but_are_not_recalled() {
    let fixture = Fixture::new();
    let store = fixture.store("alpha");
    let current = legacy_file_source(&store, "current.txt", EVIDENCE_BODY);
    let stale = legacy_file_source(&store, "stale.txt", EVIDENCE_BODY);
    let missing = legacy_file_source(&store, "missing.txt", EVIDENCE_BODY);
    std::fs::write(
        Path::new(&store.project.root).join("stale.txt"),
        "Changed evidence",
    )
    .unwrap();
    std::fs::remove_file(Path::new(&store.project.root).join("missing.txt")).unwrap();
    let mut entries = vec![
        project_entry(&store, "current", "Beacon current fact", vec![current]),
        project_entry(
            &store,
            "user",
            "Beacon user report",
            vec![legacy_user_source()],
        ),
        project_entry(&store, "stale", "Beacon stale fact", vec![stale]),
        project_entry(&store, "missing", "Beacon missing fact", vec![missing]),
    ];
    for (id, key, value) in [
        ("superseded_link", "supersededBy", json!("current")),
        ("superseded_status", "status", json!("superseded")),
        ("detached", "detached", json!(true)),
    ] {
        let mut entry = project_entry(&store, id, &format!("Beacon inactive {id}"), vec![]);
        entry["metadata"][key] = value;
        entries.push(entry);
    }
    seed(&store, &entries);
    let normal = search_memory(&store, "beacon", &[], false);
    assert_eq!(
        hit_ids(&normal),
        BTreeSet::from([
            "project:memory:current".to_string(),
            "project:memory:user".to_string(),
            "project:memory:stale".to_string(),
            "project:memory:missing".to_string(),
        ])
    );
    for (id, expected) in [
        ("current", MemoryFreshness::Current),
        ("user", MemoryFreshness::Unknown),
        ("stale", MemoryFreshness::Stale),
        ("missing", MemoryFreshness::Missing),
    ] {
        let hit = normal.iter().find(|hit| hit.entry.id == id).unwrap();
        assert_eq!(freshness(&hit.sources, Some(&store.project.root)), expected);
    }
    let inspected = search_memory(&store, "beacon", &[], true);
    assert_eq!(
        hit_ids(&inspected),
        entries
            .iter()
            .map(|entry| format!("project:memory:{}", entry["id"].as_str().unwrap()))
            .collect()
    );
    let recall = recall_memory(
        &normal,
        &default_memory_settings(),
        Some(&store.project.root),
    );
    assert_eq!(
        recall.ids.iter().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "project:memory:current".to_string(),
            "project:memory:user".to_string(),
        ])
    );
    assert!(recall.text.starts_with("[memory data; not new evidence]"));
    for excluded in [
        "Beacon stale fact",
        "Beacon missing fact",
        "Beacon inactive",
    ] {
        assert!(
            !recall.text.contains(excluded),
            "excluded content: {excluded}"
        );
    }
    assert_eq!(recall.chars, recall.text.chars().count());
    assert!(recall.chars <= 6000);
    for disabled in [
        MemorySettings {
            recall: false,
            ..default_memory_settings()
        },
        MemorySettings {
            max_recall_chars: 0,
            ..default_memory_settings()
        },
        MemorySettings {
            max_recall_entries: 0,
            ..default_memory_settings()
        },
    ] {
        let result = recall_memory(&normal, &disabled, Some(&store.project.root));
        assert!(result.ids.is_empty());
        assert!(result.text.is_empty());
        assert_eq!(result.chars, 0);
    }
}

#[tokio::test]
async fn legacy_store_updates_preserve_stable_ids_sources_and_revision_history() {
    let fixture = Fixture::new();
    let store = fixture.store("alpha");
    let sibling = fixture.store("beta");
    let refs = vec![
        legacy_user_source(),
        legacy_file_source(&store, "evidence.txt", EVIDENCE_BODY),
    ];
    let entry = project_entry(&store, "legacy_note", "Beacon original fact", refs);
    let raw = seed(&store, &[entry]);
    let before = store.read().unwrap();
    let global_dir = get_global_harness_state_dir(&store.agent_dir);
    let global_path = Path::new(&global_dir).join("harness_state.json");
    write_fixture(
        &global_path,
        &json!({"schema": 1.0, "entries": {"memory": {}}, "refinements": []}),
    );
    let global_bytes = std::fs::read(&global_path).unwrap();
    let proposal = normalize_refinement_proposal(&json!({
        "summary": "Correct legacy fact", "rationale": "Synthetic correction",
        "expectedOutcome": "Keep original source provenance",
        "edits": [{"action": "update", "kind": "memory", "id": "legacy_note",
            "title": "Legacy beacon corrected", "content": "Beacon corrected fact",
            "metadata": {"reviewed": true}}]
    }));
    // Store-level omission means retain provenance, not an explicit empty list.
    let options = ApplyOptions {
        event_id: "compatibility_update".to_string(),
        expected_revision: 1,
        ..Default::default()
    };
    let applied = store.apply(&proposal, options.clone()).await.unwrap();
    assert_eq!(applied.applied_edits.len(), 1);
    assert!(applied.applied_edits[0].applied);
    let reopened = fixture.store("alpha");
    let after = reopened.read().unwrap();
    let old_entry = &before.entries["memory"]["legacy_note"];
    let updated = &after.entries["memory"]["legacy_note"];
    assert_eq!(after.entries["memory"].len(), 1);
    assert_eq!(updated.id, old_entry.id);
    assert_eq!(updated.path, old_entry.path);
    assert_eq!(updated.reference, old_entry.reference);
    assert_eq!(updated.arguments, old_entry.arguments);
    assert_eq!(updated.created_at, old_entry.created_at);
    assert_eq!(updated.version, 2);
    assert_eq!(updated.title, "Legacy beacon corrected");
    assert_eq!(updated.content, "Beacon corrected fact");
    assert_eq!(updated.metadata["sources"], old_entry.metadata["sources"]);
    assert_eq!(
        updated.metadata["sourceIds"],
        old_entry.metadata["sourceIds"]
    );
    assert_eq!(
        updated.metadata["projectId"],
        old_entry.metadata["projectId"]
    );
    assert_eq!(updated.metadata["projectReusable"], true);
    assert_eq!(updated.metadata["reviewed"], true);
    assert_eq!(updated.metadata["status"], "current");
    assert_eq!(sources(updated), sources(old_entry));
    assert_eq!(after.memory.revision, 2);
    assert_eq!(after.memory.history.len(), 2);
    assert_eq!(after.memory.history[0], before.memory.history[0]);
    assert_eq!(after.refinements[0], before.refinements[0]);
    assert_eq!(
        after.memory.events["legacy_create"],
        raw["memory"]["events"]["legacy_create"].as_str().unwrap()
    );
    assert_eq!(after.memory.events.len(), 2);
    let edit = &after.memory.history[1].applied_edits[0];
    assert_eq!(edit.before.as_ref(), Some(old_entry));
    assert_eq!(edit.after.as_ref(), Some(updated));
    assert_eq!(
        hit_ids(&search_memory(&reopened, "beacon", &[], false)),
        BTreeSet::from(["project:memory:legacy_note".to_string()])
    );
    assert!(search_memory(&sibling, "beacon", &[], true).is_empty());
    assert_eq!(std::fs::read(&global_path).unwrap(), global_bytes);

    let committed_bytes = std::fs::read(&store.path).unwrap();
    let retried = reopened.apply(&proposal, options).await.unwrap();
    assert_eq!(retried.id, applied.id);
    assert_eq!(reopened.read().unwrap().memory.revision, 2);
    assert_eq!(std::fs::read(&store.path).unwrap(), committed_bytes);
    let stale = reopened
        .apply(
            &proposal,
            ApplyOptions {
                event_id: "stale_update".to_string(),
                expected_revision: 1,
                ..Default::default()
            },
        )
        .await;
    assert!(stale.is_err());
    assert_eq!(std::fs::read(&store.path).unwrap(), committed_bytes);
}

async fn assert_recall_opt_in_round_trip(key: &str, other_key: &str) {
    let fixture = Fixture::new();
    let store = fixture.store("alpha");
    let global_path = Path::new(&store.agent_dir).join("settings.json");
    let mut legacy: Value = serde_json::from_str(LEGACY_SETTINGS).unwrap();
    legacy["learning"] = json!(false);
    write_fixture(&global_path, &json!({"memory": legacy}));
    assert_eq!(serde_json::to_value(store.settings()).unwrap()[key], false);

    let configured = store.configure(&json!({(key): true})).await.unwrap();
    let encoded = serde_json::to_value(&configured).unwrap();
    assert_eq!(encoded[key], true, "configure must return effective {key}");
    assert_eq!(encoded[other_key], false, "the other opt-in stays disabled");
    assert_eq!(
        encoded["learning"], false,
        "legacy settings remain in force"
    );
    assert_eq!(store.settings(), configured);
    assert_eq!(fixture.store("alpha").settings(), configured);
    let local_path = Path::new(&store.dir).join("settings.json");
    assert_eq!(read_fixture(&local_path), json!({(key): true}));

    let unrelated = store
        .configure(&json!({"maxRecallEntries": 3}))
        .await
        .unwrap();
    assert_eq!(serde_json::to_value(&unrelated).unwrap()[key], true);
    assert_eq!(unrelated.max_recall_entries, 3);
    let both = store.configure(&json!({(other_key): true})).await.unwrap();
    assert_eq!(serde_json::to_value(&both).unwrap()[key], true);
    assert_eq!(serde_json::to_value(&both).unwrap()[other_key], true);
    let disabled = store.configure(&json!({(key): false})).await.unwrap();
    assert_eq!(serde_json::to_value(&disabled).unwrap()[key], false);
    assert_eq!(serde_json::to_value(&disabled).unwrap()[other_key], true);
    assert_eq!(fixture.store("alpha").settings(), disabled);

    legacy[key] = json!(true);
    write_fixture(&global_path, &json!({"memory": legacy}));
    let inherited = fixture.store("beta").settings();
    assert_eq!(
        serde_json::to_value(inherited).unwrap()[key],
        true,
        "global opt-in must propagate"
    );
    assert_eq!(
        store.settings(),
        disabled,
        "explicit project false overrides global true"
    );
    let before = std::fs::read(&local_path).unwrap();
    assert!(store.configure(&json!({(key): "true"})).await.is_err());
    assert_eq!(std::fs::read(local_path).unwrap(), before);
    assert_eq!(store.settings(), disabled);
}

#[tokio::test]
async fn legacy_settings_recall_query_distillation_configure_and_read_propagate() {
    assert_recall_opt_in_round_trip("recallQueryDistillation", "recallRerank").await;
}

#[tokio::test]
async fn legacy_settings_recall_rerank_configure_and_read_propagate() {
    assert_recall_opt_in_round_trip("recallRerank", "recallQueryDistillation").await;
}

async fn assert_service_update_provenance(explicit_clear: bool) {
    let fixture = Fixture::new();
    let service = fixture.service("alpha");
    let refs = vec![
        legacy_user_source(),
        legacy_file_source(&service.store, "evidence.txt", EVIDENCE_BODY),
    ];
    seed(
        &service.store,
        &[project_entry(
            &service.store,
            "legacy_note",
            "Beacon original service fact",
            refs.clone(),
        )],
    );
    let old = service.store.read().unwrap();
    let mut payload = json!({
        "eventId": "service_update", "revision": 1,
        "proposal": {
            "summary": "Edit a legacy note", "rationale": "Synthetic correction",
            "expectedOutcome": "Only change the requested fields",
            "edits": [{
                "action": "update", "kind": "memory", "id": "legacy_note",
                "title": "Legacy beacon", "content": "Beacon corrected service fact"
            }]
        }
    });
    if explicit_clear {
        payload["sources"] = json!([]);
    }
    let applied = service
        .request("apply", payload.as_object().unwrap(), None)
        .await
        .unwrap();
    assert_eq!(applied["appliedEdits"][0]["applied"], true);
    let reopened = fixture.service("alpha");
    let after = reopened.store.read().unwrap();
    let updated = &after.entries["memory"]["legacy_note"];
    let expected = if explicit_clear {
        json!([])
    } else {
        json!(refs)
    };
    assert_eq!(
        updated.metadata["sources"], expected,
        "omitted sources preserve provenance; only explicit [] clears it"
    );
    assert_eq!(updated.id, "legacy_note");
    assert_eq!(updated.created_at, LEGACY_TIME);
    assert_eq!(updated.version, 2);
    assert_eq!(updated.content, "Beacon corrected service fact");
    assert_eq!(after.memory.revision, 2);
    assert_eq!(after.memory.history.len(), 2);
    assert_eq!(after.memory.history[0], old.memory.history[0]);
    let update = &after.memory.history[1].applied_edits[0];
    assert_eq!(
        update.before.as_ref(),
        Some(&old.entries["memory"]["legacy_note"])
    );
    assert_eq!(update.after.as_ref(), Some(updated));
    let read_payload = json!({"id": "project:memory:legacy_note"});
    let read = reopened
        .request("read", read_payload.as_object().unwrap(), None)
        .await
        .unwrap();
    assert_eq!(read["sources"], expected);
    assert_eq!(read["id"], "project:memory:legacy_note");
}

#[tokio::test]
async fn legacy_service_update_with_omitted_sources_preserves_provenance() {
    assert_service_update_provenance(false).await;
}

#[tokio::test]
async fn legacy_service_update_with_explicit_empty_sources_clears_provenance() {
    assert_service_update_provenance(true).await;
}
