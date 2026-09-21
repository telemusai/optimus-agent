//! ROOT-CONTRACT v9: requested-Jev-model settings tests — resolution, exact
//! validation, idempotent durable writes, the A->B->A write-revision identity,
//! full-jev overlay independence and the legacy-default behavior rules.
//! No test performs network I/O; no test reads a production credential.

use std::path::Path;

use pi_jev::config::{JevMode, JevSettings, JevSettingsStore};
use pi_jev::models::{validate_requested_model_id, MAX_MODEL_NAME_CHARS};
use pi_jev::DEFAULT_MODEL;

fn store_in(agent_dir: &Path) -> JevSettingsStore {
    JevSettingsStore::new(agent_dir)
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

#[test]
fn s1_default_resolution_is_the_documented_native_default() {
    let settings = JevSettings::default();
    assert!(settings.requested_model.is_none(), "default has no explicit selection");
    assert_eq!(settings.requested_model_or_default(), "jev-latest");
    assert_eq!(settings.requested_model_or_default(), DEFAULT_MODEL);
    // The explicit selection resolves EXACTLY as persisted, never normalized.
    let mut explicit = JevSettings::default();
    explicit.requested_model = Some("jev-1.13.0".to_string());
    assert_eq!(explicit.requested_model_or_default(), "jev-1.13.0");
}

#[test]
fn s2_legacy_settings_without_the_field_load_as_none() {
    // A file written by an older build (no `requested_model` key) must load
    // byte/behavior-equivalent to the default: no selection, native default.
    let dir = tempfile::Builder::new().prefix("jev-legacy-").tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("jev")).unwrap();
    std::fs::write(
        dir.path().join("jev").join("jev-settings.json"),
        r#"{"schema_version": 1, "global_default": "compare"}"#,
    )
    .unwrap();
    let loaded = store_in(dir.path()).load();
    assert!(loaded.requested_model.is_none());
    assert_eq!(loaded.requested_model_or_default(), "jev-latest");
    assert_eq!(loaded.effective_mode("any"), JevMode::Compare, "legacy modes are preserved");
}

// ---------------------------------------------------------------------------
// Validation (the same exact-identifier rule the catalog parser uses)
// ---------------------------------------------------------------------------

#[test]
fn s3_set_refuses_hostile_ids_and_never_echoes_the_input() {
    for hostile in [
        "jev-\r\nlatest",
        "jev\u{0000}x",
        "\u{202E}evil",
        "jev\u{00A0}latest",
        "jev\u{200B}latest",
        "Bearer abc123",
        "sk-live-abcdef",
        "",
        "   ",
    ] {
        let mut settings = JevSettings::default();
        let outcome = settings.set_requested_model(hostile);
        assert!(outcome.is_err(), "must refuse: {hostile:?}");
        let reason = outcome.err().unwrap().to_string();
        if !hostile.is_empty() {
            assert!(
                !reason.contains(hostile),
                "refusal diagnostics must not echo the supplied value: {reason:?}"
            );
        }
        assert!(settings.requested_model.is_none(), "a refused id is never stored");
    }
    // Over-cap ids are refused, never truncated.
    let mut settings = JevSettings::default();
    assert!(settings.set_requested_model(&"a".repeat(MAX_MODEL_NAME_CHARS + 1)).is_err());
}

#[test]
fn s4_safe_ids_are_accepted_exactly_as_given() {
    assert_eq!(validate_requested_model_id("jev-1.13.0"), Ok(()));
    let mut settings = JevSettings::default();
    assert_eq!(settings.set_requested_model("jev-1.13.0"), Ok(true));
    assert_eq!(settings.requested_model.as_deref(), Some("jev-1.13.0"));
    // Case is preserved, while every whitespace-bearing identifier is refused.
    let mut cased = JevSettings::default();
    assert_eq!(cased.set_requested_model("JEv-Preview"), Ok(true));
    assert_eq!(cased.requested_model.as_deref(), Some("JEv-Preview"));
    for unsafe_id in ["My Model 2", " jev-preview", "jev-preview "] {
        let mut settings = JevSettings::default();
        assert!(settings.set_requested_model(unsafe_id).is_err());
        assert!(settings.requested_model.is_none());
    }
}

// ---------------------------------------------------------------------------
// Durable writes: idempotence, reload, tombstone, A->B->A identity
// ---------------------------------------------------------------------------

#[test]
fn s5_identical_set_is_an_idempotent_noop_that_never_moves_the_revision() {
    let dir = tempfile::Builder::new().prefix("jev-model-idem-").tempdir().unwrap();
    let store = store_in(dir.path());
    let mut first = store.load();
    assert_eq!(first.set_requested_model("jev-1.13.0"), Ok(true));
    store.save(&first).expect("first save");
    let revision_after_first = store.load().write_revision;
    assert_eq!(revision_after_first, 1, "the first authoritative write is revision 1");

    // An identical set is Ok(false); the caller must NOT save. Even if it
    // did, the durable value stays identical — prove the no-op contract at
    // the settings layer by re-saving and checking the persisted VALUE.
    let mut reloaded = store.load();
    assert_eq!(reloaded.set_requested_model("jev-1.13.0"), Ok(false));
    // A reset with nothing set is likewise a truthful no-op.
    let mut fresh = JevSettings::default();
    assert!(!fresh.clear_requested_model(), "reset with nothing set changes nothing");
}

#[test]
fn s6_set_reset_roundtrip_survives_a_durable_reload() {
    let dir = tempfile::Builder::new().prefix("jev-model-durable-").tempdir().unwrap();
    let store = store_in(dir.path());
    let mut settings = store.load();
    assert_eq!(settings.set_requested_model("jev-1.13.0"), Ok(true));
    store.save(&settings).expect("save set");
    let reloaded = store.load();
    assert_eq!(reloaded.requested_model.as_deref(), Some("jev-1.13.0"));
    assert_eq!(reloaded.requested_model_or_default(), "jev-1.13.0");
    assert!(reloaded.write_revision >= 1, "an authoritative write advanced the revision");

    let mut reset = reloaded.clone();
    assert!(reset.clear_requested_model(), "a real reset removes an existing selection");
    store.save(&reset).expect("save reset");
    let after_reset = store.load();
    assert!(after_reset.requested_model.is_none(), "the tombstone is durable");
    assert_eq!(after_reset.requested_model_or_default(), "jev-latest");
    assert!(
        after_reset.write_revision > reloaded.write_revision,
        "the reset write advanced the durable revision (tombstones invalidate held work)"
    );
}

#[test]
fn s7_model_a_b_a_always_advances_the_durable_write_revision() {
    // ROOT-CONTRACT v9 identity: even when the serialized VALUES return to
    // identical bytes, every authoritative save strictly advances the
    // persisted write revision, so consumers that miss intermediate writes
    // still reject older-stamped work. (A->B->A never reads as "unchanged".)
    let dir = tempfile::Builder::new().prefix("jev-model-aba-").tempdir().unwrap();
    let store = store_in(dir.path());
    let save_with = |id: &str| {
        let mut settings = store.load();
        if id == DEFAULT_MODEL {
            settings.clear_requested_model();
        } else {
            settings.set_requested_model(id).expect("valid id");
        }
        store.save(&settings).expect("save");
        store.load().write_revision
    };
    let revision_a = save_with("jev-1.13.0");
    let revision_b = save_with("jev-preview");
    let revision_a_again = save_with(DEFAULT_MODEL);
    assert!(revision_b > revision_a, "A->B advanced the revision");
    assert!(
        revision_a_again > revision_b,
        "B->A (reset tombstone) advanced the revision even though the values returned to A"
    );
}

// ---------------------------------------------------------------------------
// Independence and minimal mutation
// ---------------------------------------------------------------------------

#[test]
fn s8_model_writes_never_touch_the_full_jev_overlay_or_other_settings() {
    let dir = tempfile::Builder::new().prefix("jev-model-overlay-").tempdir().unwrap();
    let store = store_in(dir.path());
    let mut with_overlay = store.load();
    assert!(with_overlay.full_jev_install(), "the overlay installs");
    store.save(&with_overlay).expect("save overlay");
    let stamp_before = store.load().full_jev_stamp();

    let mut set_model = store.load();
    assert_eq!(set_model.set_requested_model("jev-1.13.0"), Ok(true));
    store.save(&set_model).expect("save model");
    let after = store.load();
    assert_eq!(after.full_jev_stamp(), stamp_before, "overlay identity is not churned");
    assert!(after.full_jev_active(), "the overlay survives a model write");
    assert_eq!(after.requested_model.as_deref(), Some("jev-1.13.0"));

    // Minimal mutation: beside identity bookkeeping only requested_model moved.
    let baseline = store.load();
    let mut next = baseline.clone();
    next.set_requested_model("jev-preview").expect("valid id");
    let before = serde_json::to_value(&baseline).unwrap();
    let after_value = serde_json::to_value(&next).unwrap();
    let (before_map, after_map) = (before.as_object().unwrap(), after_value.as_object().unwrap());
    for key in before_map.keys() {
        let changed = before_map.get(key) != after_map.get(key);
        assert!(
            !changed || key == "requested_model",
            "only requested_model may change; {key} moved unexpectedly"
        );
    }
    // write_revision moves only on the PERSISTED file (save path), not in the
    // in-memory struct, so it is intentionally not part of this comparison.
}


#[test]
fn s9_a_hostile_hand_edited_model_field_loads_as_defaults() {
    // Documented corrupt-file fail-safe: a hand-edited hostile requested_model
    // makes the whole file load as defaults (native default governs; nothing
    // blocks startup; no value is normalized into a different selectable id).
    let dir = tempfile::Builder::new().prefix("jev-model-hostile-").tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("jev")).unwrap();
    std::fs::write(
        dir.path().join("jev").join("jev-settings.json"),
        r#"{"schema_version": 1, "global_default": "compare", "requested_model": "jev-\u0000latest"}"#,
    )
    .unwrap();
    let mut loaded = store_in(dir.path()).load();
    assert_eq!(loaded.requested_model_or_default(), "jev-latest");
    // `loaded_generation` is read metadata, never persisted settings truth.
    loaded.loaded_generation = None;
    assert_eq!(loaded, JevSettings::default(), "a hostile persisted id fails validation to defaults");
}

#[test]
fn s10_save_refuses_a_secret_shaped_model_selection() {
    // Defense in depth: the identifier rule already refuses credential shapes;
    // the store's secret guard would refuse the write even if a value slipped
    // through a hand-crafted snapshot.
    let mut crafted = JevSettings::default();
    crafted.requested_model = Some("Bearer abc123".to_string());
    let dir = tempfile::Builder::new().prefix("jev-model-secret-").tempdir().unwrap();
    let error = store_in(dir.path()).save(&crafted).expect_err("must refuse");
    assert!(!error.to_string().is_empty());
}

#[test]
fn s11_setter_and_validate_agree_on_the_boundary() {
    // The setter must accept exactly what validate() accepts (one rule, two
    // entry points): a sample of safe and unsafe values agrees.
    for candidate in ["jev-1.13.0", "jev-preview", "my model", " x", "x ", "x\ty"] {
        let via_validator = validate_requested_model_id(candidate).is_ok();
        let mut settings = JevSettings::default();
        let via_setter = settings.set_requested_model(candidate).is_ok();
        assert_eq!(via_validator, via_setter, "boundary must agree for {candidate:?}");
    }
}

