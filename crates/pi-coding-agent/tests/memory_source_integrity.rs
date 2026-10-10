//! Source intent, schema-1 delivery receipts, and retry integrity without a model.
use pi_coding_agent::core::memory::evidence::hash;
use pi_coding_agent::core::memory::service::MemoryService;
use pi_coding_agent::core::memory::store::{empty_document, ApplyOptions};
use pi_coding_agent::core::refinement::refinement::normalize_refinement_proposal;
use serde_json::{json, Value};
use tempfile::TempDir;

const CONFLICT: &str = "Event ID reused with different content";
const RESTORED: &str =
    "Event already committed before restore; inspect current memory before resubmitting";
const LEGACY_TIME: &str = "2026-10-01T00:00:00.000Z";
// Pre-fix schema-1 receipt: sha256 of the original ordered fingerprint payload.
const LEGACY_DIGEST: &str = "abc9fd1c16daa9b5b02aa300f3d98c4f47c3a097a0aa65c4b269583a84eeb270";
const LEGACY_REPLACEMENT_DIGEST: &str =
    "5b81e19cff6e093cc1e6f998a4d88a40b05bc2aa52ce4827546fb7fcad130e7f";
const LEGACY_PROPOSAL: &str = r#"{
    "summary": "Legacy source update", "rationale": "Synthetic fixture",
    "edits": [{"action": "update", "kind": "memory", "id": "note",
        "title": "Synthetic note", "content": "Legacy updated content"}],
    "expectedOutcome": "Keep prior evidence"
}"#;

struct Fixture {
    _root: TempDir,
    service: MemoryService,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        std::fs::create_dir_all(&project).unwrap();
        let service = MemoryService::new(
            &project.to_string_lossy(),
            &root.path().join("agent").to_string_lossy(),
            None,
        )
        .unwrap();
        Self {
            _root: root,
            service,
        }
    }

    async fn apply(&self, payload: &Value) -> Result<Value, String> {
        self.service
            .request("apply", payload.as_object().unwrap(), None)
            .await
    }

    async fn seed(&self) -> Value {
        self.apply(&apply_payload(
            "seed",
            0,
            "create",
            "Original content",
            Some(json!([source("original")])),
        ))
        .await
        .unwrap()
    }

    fn bytes(&self) -> Vec<u8> {
        std::fs::read(&self.service.store.path).unwrap()
    }

    fn sources(&self) -> Value {
        self.service.store.read().unwrap().entries["memory"]["note"].metadata["sources"].clone()
    }
}

fn source(id: &str) -> Value {
    json!({
        "id": id, "origin": "user", "sha256": hash(&format!("Synthetic evidence: {id}")),
        "uri": format!("synthetic://evidence/{id}"), "revision": "synthetic_revision"
    })
}

fn apply_payload(
    event: &str,
    revision: i64,
    action: &str,
    content: &str,
    sources: Option<Value>,
) -> Value {
    let mut payload = json!({
        "eventId": event, "revision": revision,
        "proposal": {
            "summary": "Update synthetic evidence", "rationale": "Synthetic fixture",
            "edits": [{"action": action, "kind": "memory", "id": "note",
                "title": "Synthetic note", "content": content}],
            "expectedOutcome": "Keep source intent"
        }
    });
    if let Some(sources) = sources {
        payload["sources"] = sources;
    }
    payload
}

fn legacy_fingerprint(payload: &Value) -> String {
    let proposal = normalize_refinement_proposal(&payload["proposal"]);
    hash(
        &serde_json::to_string(&json!({
            "proposal": proposal,
            "sources": payload.get("sources").cloned().unwrap_or_else(|| json!([])),
            "host": payload.get("host") == Some(&Value::Bool(true)),
            "replaceMetadata": false
        }))
        .unwrap(),
    )
}

fn write_snapshot(fixture: &Fixture, snapshot: &Value) {
    std::fs::create_dir_all(&fixture.service.store.dir).unwrap();
    std::fs::write(
        &fixture.service.store.path,
        serde_json::to_vec_pretty(snapshot).unwrap(),
    )
    .unwrap();
}

fn legacy_payload(sources: Option<Value>) -> Value {
    let mut payload = json!({
        "eventId": "legacy_update", "revision": 1,
        "proposal": serde_json::from_str::<Value>(LEGACY_PROPOSAL).unwrap()
    });
    if let Some(sources) = sources {
        payload["sources"] = sources;
    }
    payload
}

// Reconstruct historical entry/history JSON rather than generating it by apply().
// The old store could retain sources with None; the old service supplied Some([]).
fn install_legacy_fixture(fixture: &Fixture, sources: Option<Value>) -> Value {
    let digest = if sources
        .as_ref()
        .is_some_and(|refs| !refs.as_array().unwrap().is_empty())
    {
        LEGACY_REPLACEMENT_DIGEST
    } else {
        LEGACY_DIGEST
    };
    let store = &fixture.service.store;
    let before = json!({
        "id": "note", "kind": "memory", "title": "Synthetic note",
        "content": "Original content", "path": "general", "scope": "local",
        "reference": {}, "arguments": {},
        "metadata": {"projectId": store.project.id, "sources": [source("original")], "status": "current"},
        "source": "refine", "created_at": LEGACY_TIME, "updated_at": LEGACY_TIME, "version": 1
    });
    let mut after = before.clone();
    after["content"] = json!("Legacy updated content");
    after["version"] = json!(2);
    after["metadata"]["sources"] = sources.unwrap_or_else(|| json!([source("original")]));
    let proposal: Value = serde_json::from_str(LEGACY_PROPOSAL).unwrap();
    let mut edit = proposal["edits"][0].clone();
    edit["metadata"] = after["metadata"].clone();
    edit["before"] = before;
    edit["after"] = after.clone();
    edit["applied"] = json!(true);
    let result = json!({
        "id": "legacy_update", "summary": proposal["summary"], "rationale": proposal["rationale"],
        "expectedOutcome": proposal["expectedOutcome"], "appliedEdits": [edit],
        "harnessStatePath": store.path, "scope": "local"
    });
    let snapshot = json!({
        "schema": 1.0, "entries": {"memory": {"note": after}, "prompt": {}, "skill": {}, "subagent": {}},
        "refinements": [{
            "id": "legacy_update", "trigger": proposal["summary"], "changes": ["update memory:note"],
            "evidence": proposal["rationale"], "outcome": proposal["expectedOutcome"], "created_at": LEGACY_TIME
        }],
        "memory": {
            "schema": 1.0, "projectId": store.project.id, "revision": 2,
            "history": [result.clone()], "events": {"legacy_update": digest}
        }
    });
    write_snapshot(fixture, &snapshot);
    result
}

#[tokio::test]
async fn source_modes_preserve_clear_or_replace_and_retry_the_original_result() {
    for refs in [None, Some(json!([])), Some(json!([source("replacement")]))] {
        let fixture = Fixture::new();
        fixture.seed().await;
        let payload = apply_payload("update", 1, "update", "Updated content", refs.clone());
        let applied = fixture.apply(&payload).await.unwrap();
        let expected = refs.unwrap_or_else(|| json!([source("original")]));
        assert_eq!(fixture.sources(), expected);
        assert_eq!(
            applied["appliedEdits"][0]["before"]["metadata"]["sources"],
            json!([source("original")])
        );
        assert_eq!(
            applied["appliedEdits"][0]["after"]["metadata"]["sources"],
            expected
        );

        fixture
            .apply(&apply_payload(
                "later",
                2,
                "update",
                "Later content",
                Some(json!([source("later")])),
            ))
            .await
            .unwrap();
        let committed = fixture.bytes();
        let reopened = MemoryService::new(
            &fixture.service.store.project.root,
            &fixture.service.store.agent_dir,
            None,
        )
        .unwrap();
        let retry = reopened
            .request("apply", payload.as_object().unwrap(), None)
            .await
            .unwrap();
        assert_eq!(
            retry, applied,
            "retry returns history, not current entry state"
        );
        assert_eq!(fixture.bytes(), committed);
        assert_eq!(fixture.sources(), json!([source("later")]));
        assert_eq!(reopened.store.read().unwrap().memory.revision, 3);
    }
}

#[tokio::test]
async fn new_receipts_bind_all_three_source_modes() {
    let modes = [None, Some(json!([])), Some(json!([source("replacement")]))];
    for (committed_index, sources) in modes.iter().enumerate() {
        let fixture = Fixture::new();
        fixture.seed().await;
        let payload = apply_payload("update", 1, "update", "Updated content", sources.clone());
        let applied = fixture.apply(&payload).await.unwrap();
        let doc = fixture.service.store.read().unwrap();
        if committed_index < 2 {
            assert_ne!(doc.memory.events["update"], legacy_fingerprint(&payload));
        } else {
            assert_eq!(doc.memory.events["update"], legacy_fingerprint(&payload));
        }
        assert_eq!(
            doc.memory.events["update"].len(),
            64,
            "schema-1 SHA-256 receipt shape"
        );
        let before = fixture.bytes();
        for (retry_index, sources) in modes.iter().enumerate() {
            let retry = apply_payload("update", 1, "update", "Updated content", sources.clone());
            if retry_index == committed_index {
                assert_eq!(fixture.apply(&retry).await.unwrap(), applied);
            } else {
                assert_eq!(fixture.apply(&retry).await.unwrap_err(), CONFLICT);
            }
            assert_eq!(fixture.bytes(), before);
        }
    }
}

#[tokio::test]
async fn nonempty_source_fingerprints_stay_legacy_compatible_and_bind_all_source_fields() {
    let fixture = Fixture::new();
    let payload = apply_payload(
        "create",
        0,
        "create",
        "Original content",
        Some(json!([source("original")])),
    );
    let applied = fixture.apply(&payload).await.unwrap();
    assert_eq!(
        fixture.service.store.read().unwrap().memory.events["create"],
        legacy_fingerprint(&payload)
    );
    let before = fixture.bytes();
    for (key, value) in [
        ("id", json!("different")),
        ("origin", json!("assistant")),
        ("sha256", json!(hash("different evidence"))),
        ("uri", json!("synthetic://different")),
        ("revision", json!("different_revision")),
        ("projectPath", json!("different.txt")),
    ] {
        let mut changed = payload.clone();
        changed["sources"][0][key] = value;
        assert_eq!(
            fixture.apply(&changed).await.unwrap_err(),
            CONFLICT,
            "{key}"
        );
    }
    assert_eq!(fixture.apply(&payload).await.unwrap(), applied);
    assert_eq!(fixture.bytes(), before);
}

#[tokio::test]
async fn legacy_receipts_return_history_without_rewriting_or_reapplying() {
    let modes = [None, Some(json!([])), Some(json!([source("replacement")]))];
    for (committed_index, sources) in modes.iter().enumerate() {
        let fixture = Fixture::new();
        let result = install_legacy_fixture(&fixture, sources.clone());
        let digest = if committed_index < 2 {
            LEGACY_DIGEST
        } else {
            LEGACY_REPLACEMENT_DIGEST
        };
        assert_eq!(legacy_fingerprint(&legacy_payload(sources.clone())), digest);
        let before = fixture.bytes();
        for (retry_index, sources) in modes.iter().enumerate() {
            let retry = legacy_payload(sources.clone());
            if retry_index == committed_index || (retry_index < 2 && committed_index < 2) {
                // Historical receipts irrecoverably collapsed None and Some([]).
                assert_eq!(fixture.apply(&retry).await.unwrap(), result);
            } else {
                assert_eq!(fixture.apply(&retry).await.unwrap_err(), CONFLICT);
            }
            assert_eq!(fixture.bytes(), before);
        }
        let mut changed_proposal = legacy_payload(sources.clone());
        changed_proposal["proposal"]["edits"][0]["content"] = json!("Different content");
        assert_eq!(
            fixture.apply(&changed_proposal).await.unwrap_err(),
            CONFLICT
        );
        let mut changed_host = legacy_payload(sources.clone());
        changed_host["host"] = json!(true);
        assert_eq!(fixture.apply(&changed_host).await.unwrap_err(), CONFLICT);
        assert_eq!(fixture.bytes(), before);
        assert_eq!(fixture.service.store.read().unwrap().memory.revision, 2);
    }
}

#[tokio::test]
async fn restore_keeps_new_source_receipts_when_update_history_is_absent() {
    let modes = [None, Some(json!([])), Some(json!([source("replacement")]))];
    for (committed_index, sources) in modes.iter().enumerate() {
        let fixture = Fixture::new();
        let seed = fixture.seed().await;
        let backup = fixture.service.store.backup(None).unwrap();
        let payload = apply_payload("update", 1, "update", "Updated content", sources.clone());
        fixture.apply(&payload).await.unwrap();
        let receipts = fixture.service.store.read().unwrap().memory.events;
        assert_eq!(fixture.service.store.restore(&backup).await.unwrap(), 3);
        let restored = fixture.service.store.read().unwrap();
        assert_eq!(restored.memory.events, receipts);
        assert_eq!(restored.memory.history.len(), 1);
        assert_eq!(fixture.sources(), json!([source("original")]));
        let before = fixture.bytes();
        for (retry_index, sources) in modes.iter().enumerate() {
            let retry = apply_payload("update", 1, "update", "Updated content", sources.clone());
            let expected = if retry_index == committed_index {
                RESTORED
            } else {
                CONFLICT
            };
            assert_eq!(fixture.apply(&retry).await.unwrap_err(), expected);
            assert_eq!(fixture.bytes(), before);
        }
        let seed_payload = apply_payload(
            "seed",
            0,
            "create",
            "Original content",
            Some(json!([source("original")])),
        );
        assert_eq!(fixture.apply(&seed_payload).await.unwrap(), seed);
        assert_eq!(fixture.bytes(), before);
    }
}

#[tokio::test]
async fn restored_legacy_receipts_cannot_replay_even_without_saved_history() {
    let modes = [None, Some(json!([])), Some(json!([source("replacement")]))];
    for (committed_index, sources) in modes.iter().enumerate() {
        let fixture = Fixture::new();
        let empty =
            serde_json::to_value(empty_document(&fixture.service.store.project.id)).unwrap();
        write_snapshot(&fixture, &empty);
        let backup = fixture.service.store.backup(None).unwrap();
        install_legacy_fixture(&fixture, sources.clone());
        let receipts = fixture.service.store.read().unwrap().memory.events;
        assert_eq!(fixture.service.store.restore(&backup).await.unwrap(), 3);
        let before = fixture.bytes();
        for (retry_index, sources) in modes.iter().enumerate() {
            let expected =
                if retry_index == committed_index || (retry_index < 2 && committed_index < 2) {
                    RESTORED
                } else {
                    CONFLICT
                };
            assert_eq!(
                fixture
                    .apply(&legacy_payload(sources.clone()))
                    .await
                    .unwrap_err(),
                expected
            );
            assert_eq!(fixture.bytes(), before);
        }
        let restored = fixture.service.store.read().unwrap();
        assert!(restored.entries["memory"].is_empty());
        assert!(restored.memory.history.is_empty());
        assert_eq!(restored.memory.events, receipts);
    }
}

#[tokio::test]
async fn malformed_sources_and_proposals_fail_without_erasing_provenance_or_receipts() {
    let fixture = Fixture::new();
    fixture.seed().await;
    let before = fixture.bytes();
    let mut missing_id = source("invalid");
    missing_id.as_object_mut().unwrap().remove("id");
    let mut bad_hash = source("invalid");
    bad_hash["sha256"] = json!("not-a-digest");
    let mut bad_origin = source("invalid");
    bad_origin["origin"] = json!("memory");
    for sources in [
        Value::Null,
        json!(false),
        json!({}),
        json!("not-an-array"),
        json!([null]),
        json!([missing_id]),
        json!([bad_hash]),
        json!([bad_origin]),
        Value::Array(vec![source("too_many"); 201]),
    ] {
        let payload = apply_payload("invalid", 1, "update", "Invalid update", Some(sources));
        assert!(fixture.apply(&payload).await.is_err());
        assert_eq!(fixture.bytes(), before);
    }
    for proposal in [
        Value::Null,
        json!({"edits": [null]}),
        json!({"edits": "bad"}),
    ] {
        let mut payload = apply_payload("invalid", 1, "update", "Invalid update", None);
        payload["proposal"] = proposal;
        assert!(fixture.apply(&payload).await.is_err());
        assert_eq!(fixture.bytes(), before);
    }
    assert_eq!(fixture.sources(), json!([source("original")]));
    assert_eq!(fixture.service.store.read().unwrap().memory.events.len(), 1);
}

#[tokio::test]
async fn rollback_restores_before_sources_instead_of_the_current_source_policy() {
    for refs in [None, Some(json!([])), Some(json!([source("replacement")]))] {
        let fixture = Fixture::new();
        fixture.seed().await;
        let original = fixture.service.store.read().unwrap().entries["memory"]["note"].clone();
        let mut payload = apply_payload("update", 1, "update", "Updated content", refs);
        payload["proposal"]["edits"][0]["metadata"] = json!({"reviewed": true});
        fixture.apply(&payload).await.unwrap();
        let rolled = fixture.service.store.rollback("update", 2).await.unwrap();
        let doc = fixture.service.store.read().unwrap();
        let entry = &doc.entries["memory"]["note"];
        assert_eq!(entry.metadata, original.metadata);
        assert_eq!(entry.content, original.content);
        assert_eq!(entry.created_at, original.created_at);
        assert_eq!(entry.version, 3);
        assert_eq!(doc.memory.revision, 3);
        assert_eq!(rolled.applied_edits[0].after.as_ref(), Some(entry));
    }
}

#[tokio::test]
async fn handoff_retains_its_existing_complete_checkpoint_source_semantics() {
    let fixture = Fixture::new();
    let base = json!({
        "task": "Synthetic task", "state": "Started", "decisions": "Keep fixture local",
        "unresolved": "Nothing", "eventId": "handoff_create", "revision": 0,
        "sources": [source("original")]
    });
    fixture
        .service
        .request("handoff", base.as_object().unwrap(), None)
        .await
        .unwrap();
    let mut update = base.clone();
    update["eventId"] = json!("handoff_update");
    update["revision"] = json!(1);
    update["state"] = json!("Checkpoint complete");
    update.as_object_mut().unwrap().remove("sources");
    let applied = fixture
        .service
        .request("handoff", update.as_object().unwrap(), None)
        .await
        .unwrap();
    assert_eq!(
        applied["appliedEdits"][0]["before"]["metadata"]["sources"],
        json!([source("original")])
    );
    assert_eq!(
        applied["appliedEdits"][0]["after"]["metadata"]["sources"],
        json!([])
    );
    let before = fixture.bytes();
    assert_eq!(
        fixture
            .service
            .request("handoff", update.as_object().unwrap(), None)
            .await
            .unwrap(),
        applied
    );
    assert_eq!(fixture.bytes(), before);
}

#[tokio::test]
async fn omitted_store_sources_keep_import_style_per_edit_evidence_and_legacy_retries() {
    let fixture = Fixture::new();
    let raw = json!({
        "summary": "Import synthetic evidence", "rationale": "Offline fixture",
        "expectedOutcome": "Keep separate citations",
        "edits": [
            {"action": "create", "kind": "memory", "id": "one", "title": "One", "content": "First fact",
                "metadata": {"sources": [source("one")], "sourceIds": ["one"]}},
            {"action": "create", "kind": "memory", "id": "two", "title": "Two", "content": "Second fact",
                "metadata": {"sources": [source("two")], "sourceIds": ["two"]}}
        ]
    });
    let proposal = normalize_refinement_proposal(&raw);
    let options = ApplyOptions {
        event_id: "synthetic_import".to_string(),
        expected_revision: 0,
        ..Default::default()
    };
    let applied = serde_json::to_value(
        fixture
            .service
            .store
            .apply(&proposal, options.clone())
            .await
            .unwrap(),
    )
    .unwrap();
    let document = fixture.service.store.read().unwrap();
    for id in ["one", "two"] {
        assert_eq!(
            document.entries["memory"][id].metadata["sources"],
            json!([source(id)])
        );
        assert_eq!(
            document.entries["memory"][id].metadata["sourceIds"],
            json!([id])
        );
    }
    let before = fixture.bytes();
    assert_eq!(
        serde_json::to_value(
            fixture
                .service
                .store
                .apply(&proposal, options.clone())
                .await
                .unwrap(),
        )
        .unwrap(),
        applied
    );
    assert_eq!(fixture.bytes(), before);

    // Emulate a pre-fix accepted import receipt after a crash before job status was saved.
    let mut legacy = serde_json::to_value(document).unwrap();
    legacy["memory"]["events"]["synthetic_import"] =
        json!(legacy_fingerprint(&json!({"proposal": raw})));
    write_snapshot(&fixture, &legacy);
    let before = fixture.bytes();
    assert_eq!(
        serde_json::to_value(
            fixture
                .service
                .store
                .apply(&proposal, options)
                .await
                .unwrap(),
        )
        .unwrap(),
        applied
    );
    assert_eq!(fixture.bytes(), before);
    assert_eq!(fixture.service.store.read().unwrap().memory.revision, 1);
}
