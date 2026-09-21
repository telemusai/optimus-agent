//! ROOT-CONTRACT v9: native `/jev` model command tests.
//!
//! Same shape as `jev_ui_tests.rs` (the established lane-C pattern):
//! 1. The PURE logic is executed for real via `#[path]`-included
//!    `jev_menu.rs`: command parsing (exact-id preservation), the durable
//!    zero-network bridge setters, credential-overlap refusal, full-jev
//!    independence, persisted-bytes minimal mutation, and both model panels.
//! 2. The WIRING is audited from source: the dispatch arms, the single
//!    explicit catalog fetch call site, and the transport-free bridge surface.
//!
//! Mock/local fixtures only; no network. UNEXECUTED: the integrator owns
//! compilation and the coherent test run.

#[path = "../src/modes/interactive/jev_menu.rs"]
mod jev_menu_mod;

use std::fs;
use std::path::Path;

use jev_menu_mod::{
    is_on_shorthand, jev_usage, parse_jev_request, render_model_catalog, render_model_status,
    FullJevChange, JevModelStatusReport, JevModeBridge, JevRequest, JEV_ARGUMENT_HINT,
    JEV_BOUNDARY_NOTICE, JEV_COMMAND_DESCRIPTION, MAX_MODEL_CATALOG_DISPLAY,
};
use pi_jev::config::{JevMode, JevSettings, JevSettingsStore};
use pi_jev::credential::SecretString;
use pi_jev::models::parse_model_catalog;
use pi_jev::types::DEFAULT_MODEL;

const CRATE: &str = env!("CARGO_MANIFEST_DIR");

fn crate_file(relative: &str) -> String {
    let path = Path::new(CRATE).join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} must be readable: {error}", path.display()))
}

fn store_in(agent_dir: &Path) -> JevSettingsStore {
    JevSettingsStore::new(agent_dir)
}

// ---------------------------------------------------------------------------
// Command parsing
// ---------------------------------------------------------------------------

#[test]
fn c1_parse_models_and_model_commands() {
    assert_eq!(parse_jev_request("models"), JevRequest::Models);
    assert_eq!(parse_jev_request("  models  "), JevRequest::Models);
    assert_eq!(parse_jev_request("MODELS"), JevRequest::Models);
    assert_eq!(parse_jev_request("model"), JevRequest::ModelStatus);
    assert_eq!(parse_jev_request("model status"), JevRequest::ModelStatus);
    assert_eq!(parse_jev_request("Model STATUS"), JevRequest::ModelStatus);
    assert_eq!(parse_jev_request("model reset"), JevRequest::ModelReset);
    assert_eq!(parse_jev_request("MODEL RESET"), JevRequest::ModelReset);
}

#[test]
fn c1_parse_keeps_the_requested_model_id_exactly() {
    // The id is an EXACT identifier: case and unsafe spacing reach validation unchanged.
    assert_eq!(
        parse_jev_request("model set jev-1.13.0"),
        JevRequest::ModelSet("jev-1.13.0".to_string())
    );
    assert_eq!(
        parse_jev_request("model set JEv-1.13.0"),
        JevRequest::ModelSet("JEv-1.13.0".to_string())
    );
    assert_eq!(
        parse_jev_request("model set my model"),
        JevRequest::ModelSet("my model".to_string())
    );
    assert_eq!(
        parse_jev_request("model set my model "),
        JevRequest::ModelSet("my model ".to_string())
    );
    assert_eq!(
        parse_jev_request("MODEL SET jev-preview"),
        JevRequest::ModelSet("jev-preview".to_string())
    );
    // A bare `model set` yields the empty id (a usage error downstream).
    assert_eq!(parse_jev_request("model set"), JevRequest::ModelSet(String::new()));
    // Unrelated forms still parse as before.
    assert!(matches!(parse_jev_request("models bogus"), JevRequest::Unknown(_)));
    assert!(matches!(parse_jev_request("model status extra"), JevRequest::Unknown(_)));
    assert!(matches!(parse_jev_request("model settings"), JevRequest::Unknown(_)));
    assert_eq!(parse_jev_request("on"), JevRequest::SetMode(pi_jev::types::JevMode::Compare));
    assert_eq!(parse_jev_request("status"), JevRequest::Status);
    assert!(is_on_shorthand("on"));
    assert!(jev_usage().contains(JEV_ARGUMENT_HINT));
}

// ---------------------------------------------------------------------------
// Bridge setters: durable, idempotent, validating, zero-network
// ---------------------------------------------------------------------------

#[test]
fn c2_bridge_set_is_durable_and_reloadable() {
    let dir = tempfile::Builder::new().prefix("jev-cmd-set-").tempdir().unwrap();
    let bridge = JevModeBridge::new(dir.path());
    let outcome = bridge.set_requested_model("jev-1.13.0", None).expect("valid id");
    assert!(outcome.written);
    assert_eq!(outcome.requested, "jev-1.13.0");
    let reloaded = store_in(dir.path()).load();
    assert_eq!(reloaded.requested_model.as_deref(), Some("jev-1.13.0"));
    assert_eq!(reloaded.write_revision, 1, "the first authoritative write is revision 1");
}

#[test]
fn c3_identical_set_writes_nothing_and_moves_no_revision() {
    let dir = tempfile::Builder::new().prefix("jev-cmd-idem-").tempdir().unwrap();
    let bridge = JevModeBridge::new(dir.path());
    bridge.set_requested_model("jev-1.13.0", None).expect("first set");
    let bytes_before = fs::read(bridge.path()).unwrap();
    let outcome = bridge.set_requested_model("jev-1.13.0", None).expect("second set");
    assert!(!outcome.written, "an identical set is a truthful no-op");
    assert_eq!(bytes_before, fs::read(bridge.path()).unwrap(), "a no-op must not rewrite the file");
    assert_eq!(store_in(dir.path()).load().write_revision, 1, "the revision did not move");
}

#[test]
fn c4_bridge_reset_tombstone_is_durable_and_idempotent() {
    let dir = tempfile::Builder::new().prefix("jev-cmd-reset-").tempdir().unwrap();
    let bridge = JevModeBridge::new(dir.path());
    bridge.set_requested_model("jev-1.13.0", None).expect("set");
    let outcome = bridge.reset_requested_model().expect("reset");
    assert!(outcome.written);
    assert_eq!(outcome.requested, DEFAULT_MODEL);
    let reloaded = store_in(dir.path()).load();
    assert!(reloaded.requested_model.is_none());
    assert_eq!(reloaded.requested_model_or_default(), "jev-latest");
    assert!(reloaded.write_revision >= 2, "the reset write advanced the revision");
    // An already-default reset is a truthful no-op with no write.
    let bytes_before = fs::read(bridge.path()).unwrap();
    let second = bridge.reset_requested_model().expect("second reset");
    assert!(!second.written, "already default: nothing written");
    assert_eq!(bytes_before, fs::read(bridge.path()).unwrap(), "no rewrite happened");
}

#[test]
fn c5_bridge_validates_and_refuses_without_echo() {
    let dir = tempfile::Builder::new().prefix("jev-cmd-refuse-").tempdir().unwrap();
    let bridge = JevModeBridge::new(dir.path());
    for hostile in ["\u{202E}evil", "Bearer abc123", "jev\u{00A0}latest", "my model"] {
        let error = bridge.set_requested_model(hostile, None).expect_err("must refuse");
        assert!(!error.contains(hostile), "refusals never echo the supplied value: {error:?}");
        assert!(bridge.settings().requested_model.is_none());
    }
    // The empty id is refused too (no echo assertion needed for empty text).
    assert!(bridge.set_requested_model("", None).is_err());
    assert!(bridge.set_requested_model("   ", None).is_err());
    // Effective-credential overlap: refused GENERALLY, never echoed, never saved.
    let secret = SecretString::new("abc123456789");
    for overlapping in ["abc123456789", "xabc123456789y"] {
        let error = bridge
            .set_requested_model(overlapping, Some(&secret))
            .expect_err("credential overlap must refuse");
        assert!(
            error.contains("looks like a credential"),
            "the generic reason is surfaced: {error:?}"
        );
        assert!(!error.contains("abc123456789"), "no credential fragment is echoed");
        assert!(bridge.settings().requested_model.is_none(), "nothing was persisted");
    }
    // A safe id passes WITH a credential present.
    bridge
        .set_requested_model("jev-1.13.0", Some(&secret))
        .expect("an ordinary id never overlaps");
}

#[test]
fn c6_model_writes_are_independent_of_the_full_jev_overlay() {
    let dir = tempfile::Builder::new().prefix("jev-cmd-overlay-").tempdir().unwrap();
    let bridge = JevModeBridge::new(dir.path());
    let installed = bridge.set_full_jev(true).expect("overlay installs");
    assert!(matches!(
        installed,
        FullJevChange::Installed { already_active: false }
    ));
    let stamp_before = bridge.settings().full_jev_stamp();
    // The overlay does NOT mask a deliberate model selection.
    let outcome = bridge.set_requested_model("jev-1.13.0", None).expect("set under overlay");
    assert!(outcome.written);
    let settings = bridge.settings();
    assert_eq!(settings.requested_model.as_deref(), Some("jev-1.13.0"));
    assert!(settings.full_jev_active(), "the overlay survives");
    assert_eq!(settings.full_jev_stamp(), stamp_before, "overlay identity is not churned");
    // And the selection survives an overlay removal untouched.
    bridge.set_full_jev(false).expect("overlay removed");
    assert_eq!(bridge.settings().requested_model.as_deref(), Some("jev-1.13.0"));
}

#[test]
fn c7_a_model_write_moves_only_requested_model_in_memory() {
    let before = JevSettings::default();
    let mut after = before.clone();
    after.set_requested_model("jev-1.13.0").expect("valid id");
    let before_value = serde_json::to_value(&before).unwrap();
    let after_value = serde_json::to_value(&after).unwrap();
    let (before_map, after_map) = (
        before_value.as_object().unwrap(),
        after_value.as_object().unwrap(),
    );
    for (key, before_field) in before_map {
        assert!(
            after_map.get(key) == Some(before_field) || key == "requested_model",
            "only requested_model may change; {key} moved unexpectedly"
        );
    }
    assert_eq!(after_map.get("requested_model"), Some(&serde_json::json!("jev-1.13.0")));
}

#[test]
fn c7b_persisted_bytes_change_only_the_requested_model_field() {
    let dir = tempfile::Builder::new().prefix("jev-cmd-bytes-").tempdir().unwrap();
    let bridge = JevModeBridge::new(dir.path());
    let before_bytes = fs::read(bridge.path()).unwrap_or_default();
    bridge.set_requested_model("jev-1.13.0", None).expect("set");
    let before: serde_json::Value = if before_bytes.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_slice(&before_bytes).unwrap()
    };
    let after: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(bridge.path()).unwrap()).unwrap();
    for (key, before_field) in before.as_object().unwrap() {
        let changed = after.get(key) != Some(before_field);
        assert!(
            !changed || key == "requested_model" || key == "write_revision",
            "persisted key {key} moved unexpectedly"
        );
    }
    assert_eq!(after.get("requested_model"), Some(&serde_json::json!("jev-1.13.0")));
    assert!(after.get("write_revision").is_some(), "the persisted identity is present");
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

#[test]
fn c8_model_status_panel_reports_requested_default_and_drift_truthfully() {
    let default_report = JevModelStatusReport {
        requested: "jev-latest".to_string(),
        explicit: None,
        write_revision: 7,
        reported: None,
    };
    let default_text = render_model_status(&default_report);
    assert!(default_text.contains("native default; nothing explicitly set"));
    assert!(default_text.contains("unknown (no comparison observed in this worker)"));
    assert!(default_text.contains("Durable settings write revision: 7"));

    let explicit_report = JevModelStatusReport {
        requested: "jev-1.13.0".to_string(),
        explicit: Some("jev-1.13.0".to_string()),
        write_revision: 7,
        reported: Some("jev-1.12.0".to_string()),
    };
    let explicit_text = render_model_status(&explicit_report);
    assert!(explicit_text.contains("explicit /jev model set selection"));
    assert!(explicit_text.contains("Model reported by server: jev-1.12.0"));
    assert!(
        explicit_text.contains("differs from the requested id"),
        "drift is disclosed: {explicit_text}"
    );
    // Boundary statements are present and truthful.
    assert!(explicit_text.contains("never selects the primary chat model"));
    assert!(explicit_text.contains("no network call"));
}

#[test]
fn c9_catalog_panel_is_bounded_and_never_echoes_hostile_values() {
    // A clean catalog renders entries exactly.
    let clean = serde_json::json!({
        "models": [
            {"name": "jev-latest", "description": "The most recent stable, official release.", "release_date": "2026-08-01"}
        ]
    })
    .to_string();
    let catalog = parse_model_catalog(clean.as_bytes()).expect("parses");
    let text = render_model_catalog(&catalog);
    assert!(text.contains("- jev-latest"));
    assert!(text.contains("released 2026-08-01"));
    assert!(text.contains("nothing was selected"));

    // Rejected entries are disclosed with bounded reasons, never with the
    // hostile value itself.
    let hostile = serde_json::json!({
        "models": [
            {"name": "sk-live-abcdef", "description": "hostile", "release_date": "2026-08-01"},
            {"name": "jev-latest", "description": "stable", "release_date": "2026-08-01"}
        ]
    })
    .to_string();
    let catalog = parse_model_catalog(hostile.as_bytes()).expect("parses");
    let text = render_model_catalog(&catalog);
    assert!(!text.contains("sk-live-abcdef"), "rejected values are never echoed");
    assert!(text.contains("Rejected entries"));

    // The display cap is honored with a truthful overflow note.
    let mut entries = Vec::new();
    for i in 0..(pi_jev::models::MAX_MODEL_CATALOG_ENTRIES + 5) {
        entries.push(serde_json::json!({
            "name": format!("model-{i}"),
            "description": "d",
            "release_date": "2026-08-01"
        }));
    }
    let big = serde_json::json!({ "models": entries }).to_string();
    let catalog = parse_model_catalog(big.as_bytes()).expect("parses");
    assert_eq!(catalog.models.len(), pi_jev::models::MAX_MODEL_CATALOG_ENTRIES);
    let text = render_model_catalog(&catalog);
    assert!(text.contains("host display cap 32"));
}

// ---------------------------------------------------------------------------
// Dispatch wiring: the ONLY networked model command is the explicit catalog
// ---------------------------------------------------------------------------

#[test]
fn c10_only_the_models_arm_fetches_and_status_set_reset_stay_off_the_network() {
    let host = crate_file("src/modes/interactive/jev_host.rs");
    // The explicit catalog arm is present and is the ONLY fetch call site.
    assert!(host.contains("JevRequest::Models => {"), "the catalog arm exists");
    assert_eq!(
        host.matches("fetch_model_catalog").count(),
        1,
        "exactly one catalog fetch call site: the explicit /jev models arm"
    );
    // The zero-network arms exist and never construct a transport or fetch.
    assert!(host.contains("JevRequest::ModelStatus => {"));
    assert!(host.contains("JevRequest::ModelSet(id) => {"));
    assert!(host.contains("JevRequest::ModelReset => {"));
    // The bridge surface the zero-network arms use has no transport type.
    let menu = crate_file("src/modes/interactive/jev_menu.rs");
    let bridge_block = menu
        .split("impl JevModeBridge")
        .nth(1)
        .expect("the bridge impl exists")
        .to_string();
    for forbidden in ["Transport", "fetch_model_catalog", "get_models", "JevHttpTransport"] {
        assert!(
            !bridge_block.contains(forbidden),
            "the mode/model bridge must own no transport surface: found {forbidden}"
        );
    }
    // The catalog transport helper is command-scoped: named for the command,
    // built over the existing single build_transport selection path.
    let bridge_src = crate_file("src/core/jev_bridge.rs");
    assert!(bridge_src.contains("fn catalog_transport_for_command"));
    assert!(bridge_src.contains("fn credential_for_model_overlap"));
    // The metadata constants quote the model commands.
    assert!(JEV_ARGUMENT_HINT.contains("models"), "{JEV_ARGUMENT_HINT}");
    assert!(JEV_ARGUMENT_HINT.contains("model"), "{JEV_ARGUMENT_HINT}");
    assert!(JEV_COMMAND_DESCRIPTION.contains("model catalog"));
}

#[test]
fn c10_usage_and_boundary_text_stay_truthful() {
    let usage = jev_usage();
    assert!(usage.contains(JEV_ARGUMENT_HINT));
    // The boundary notice still says Jev never controls the PRIMARY model.
    assert!(JEV_BOUNDARY_NOTICE.contains("never controls the primary model"));
}

