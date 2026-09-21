//! Lane A (jev-models) tests: explicit model-catalog wire, sanitization, bounded fetch,
//! and explicit-only guarantees (ROOT-CONTRACT v9).
//!
//! No test performs network I/O, reads a production credential, or spawns a process.
//! Every transport is `MockJevTransport` (or a local minimal `Transport`) going through
//! the production code paths: `JevHttpTransport::get_models` / `MockJevTransport::
//! get_models` produce raw bytes, and `parse_model_catalog` is the same parser both use.

use std::sync::Arc;
use std::time::Duration;

use pi_jev::client::{decide_with, JevLimits};
use pi_jev::error::{sanitize_opaque_header_value, JevError};
use pi_jev::mock::{choice_question, noul_question, MockJevTransport, MockStep};
use pi_jev::models::{
    fetch_model_catalog, id_overlaps_credential, parse_model_catalog, validate_requested_model_id,
    ModelCard, MODEL_CATALOG_PATH, MAX_MODEL_CATALOG_ENTRIES, MAX_MODEL_DESCRIPTION_CHARS,
    MAX_MODEL_NAME_CHARS,
};
use pi_jev::types::{DecisionBundle, DecisionCategory, Transport};
use pi_jev::{DEFAULT_MODEL, JevMode, JevStats, SecretString, SystemOneRequest};
use serde_json::json;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// The documented example shape: aliases with name/description/release_date.
fn documented_catalog_body() -> String {
    json!({
        "models": [
            {
                "name": "jev-latest",
                "description": "The most recent stable, official release.",
                "release_date": "2026-08-01"
            },
            {
                "name": "jev-preview",
                "description": "The most recent release, whether or not it is an official one.",
                "release_date": "2026-08-01"
            }
        ]
    })
    .to_string()
}

fn catalog_limits() -> JevLimits {
    JevLimits {
        timeout: Duration::from_millis(100),
        ..JevLimits::default()
    }
}

async fn fetch(
    transport: &Arc<MockJevTransport>,
    limits: &JevLimits,
) -> Result<pi_jev::models::ModelCatalog, JevError> {
    fetch_model_catalog(transport.as_ref(), limits).await
}

fn bundle() -> DecisionBundle {
    let mut questions = BTreeMap::new();
    questions.insert(
        "task_classification.0".to_string(),
        choice_question("Which task type is this?", &[("coding", Some("code change"))]),
    );
    questions.insert(
        "result_sufficiency.0".to_string(),
        noul_question("Is the result sufficient?", "sufficient", "insufficient"),
    );
    let question_categories = questions
        .keys()
        .map(|id| {
            let prefix = id.split('.').next().unwrap_or(id);
            (
                id.clone(),
                DecisionCategory::parse(prefix).unwrap_or(DecisionCategory::TaskClassification),
            )
        })
        .collect();
    DecisionBundle {
        session_id: "local-session-1".to_string(),
        turn: 3,
        stage: "turn_start".to_string(),
        state: json!({"summary": "bounded state excerpt"}),
        model: "jev-latest".to_string(),
        questions,
        question_categories,
    }
}

/// A minimal transport that implements ONLY `post`, exercising the trait's provided
/// `get_models` default (an unsupported transport must not fabricate a catalog).
struct PostOnlyTransport;

impl Transport for PostOnlyTransport {
    fn post(
        &self,
        _request: &SystemOneRequest,
        _timeout: Duration,
    ) -> pi_jev::types::BoxFuture<Result<pi_jev::types::SystemOneResponse, JevError>> {
        Box::pin(async {
            Err(JevError::Internal {
                detail: "post-only transport".to_string(),
            })
        })
    }
}

// ---------------------------------------------------------------------------
// m1-m2: documented schema
// ---------------------------------------------------------------------------

#[tokio::test]
async fn m1_documented_schema_parses_exact_fields() {
    let catalog = parse_model_catalog(documented_catalog_body().as_bytes()).expect("documented body parses");
    assert_eq!(catalog.models.len(), 2, "two documented aliases");
    assert!(catalog.rejected.is_empty(), "no rejections for clean input");
    assert_eq!(
        catalog.models[0],
        ModelCard {
            name: "jev-latest".to_string(),
            description: "The most recent stable, official release.".to_string(),
            release_date: "2026-08-01".to_string(),
        }
    );
    assert_eq!(catalog.models[1].name, "jev-preview");
}

#[tokio::test]
async fn m2_extra_fields_are_ignored_nothing_is_invented() {
    let body = json!({
        "object": "list",
        "models": [
            {
                "name": "jev-latest",
                "description": "stable",
                "release_date": "2026-08-01",
                "context_length": "64k tokens",
                "price_per_btok": 42
            }
        ]
    });
    let catalog = parse_model_catalog(body.to_string().as_bytes()).expect("parses");
    assert_eq!(catalog.models.len(), 1);
    assert_eq!(catalog.models[0].name, "jev-latest");
    assert!(catalog.rejected.is_empty());
}

// ---------------------------------------------------------------------------
// m3-m6: sanitization and per-entry validation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn m3_names_reject_control_characters_prose_fields_are_stripped() {
    // NAME = identifier: control characters (CR/LF, NUL) REJECT the entry instead of
    // being normalized into a different selectable ID.
    let body = json!({
        "models": [
            {
                "name": " jev-\r\nlatest\u{0000} ",
                "description": "stable",
                "release_date": "2026-08-01"
            },
            {
                "name": "jev-latest",
                "description": "stable\r\nrelease\u{0007}",
                "release_date": "2026-08-01\n"
            }
        ]
    });
    let catalog = parse_model_catalog(body.to_string().as_bytes()).expect("envelope parses");
    assert_eq!(catalog.models.len(), 1, "only the clean entry survives");
    assert_eq!(catalog.models[0].name, "jev-latest", "safe name preserved exactly");
    assert_eq!(catalog.models[0].description, "stablerelease", "prose: presentation sanitizing");
    assert_eq!(catalog.models[0].release_date, "2026-08-01");
    assert_eq!(catalog.rejected.len(), 1, "the hostile name is disclosed");
    assert!(catalog.rejected[0].contains("entry 0"));
    assert!(catalog.rejected[0].contains("control character"));
}

#[tokio::test]
async fn m4_credential_echoes_and_bidi_are_rejected_individually() {
    let body = json!({
        "models": [
            {"name": "jev-latest", "description": "stable", "release_date": "2026-08-01"},
            {"name": "sk-live-abcdef", "description": "hostile", "release_date": "2026-08-01"},
            {"name": "jev-preview", "description": "Bearer abc123", "release_date": "2026-08-01"},
            {"name": "jev-1.13.0\u{202E}", "description": "bidi injection", "release_date": "2026-08-01"},
            {"name": "jev-1.12.0", "description": "older", "release_date": "authorization=x"}
        ]
    });
    let catalog = parse_model_catalog(body.to_string().as_bytes()).expect("well-formed envelope parses");
    // Exactly one clean entry survives; hostile entries are disclosed, never silent.
    assert_eq!(catalog.models.len(), 1, "only the clean entry is kept");
    assert_eq!(catalog.models[0].name, "jev-latest");
    assert_eq!(catalog.rejected.len(), 4, "every hostile entry is disclosed");
    let joined = catalog.rejected.join("\n");
    assert!(joined.contains("entry 1"), "credential echo in name");
    assert!(joined.contains("entry 2"), "credential echo in description");
    assert!(joined.contains("entry 3"), "bidi control in name");
    assert!(joined.contains("entry 4"), "credential echo in release_date");
    assert!(joined.contains("credential echo") && joined.contains("bidi"));
}

#[tokio::test]
async fn m5_structural_failures_are_malformed_response() {
    for body in [
        "not json".to_string(),
        json!(["jev-latest"]).to_string(),
        json!({"nope": true}).to_string(),
        json!({"models": "jev-latest"}).to_string(),
    ] {
        let error = parse_model_catalog(body.as_bytes()).expect_err("structural failure");
        assert_eq!(error.kind(), "malformed_response", "body: {body}");
        assert!(
            error.to_string().contains("model catalog payload"),
            "the detail names the catalog parser: {error}"
        );
    }
}

#[tokio::test]
async fn m6_entry_shape_rejections_and_caps() {
    // Missing / non-string / whitespace-bearing name; over-cap name is rejected, never truncated.
    let body = json!({
        "models": [
            {"description": "no name", "release_date": "2026-08-01"},
            {"name": 42, "description": "numeric name", "release_date": "2026-08-01"},
            {"name": "   ", "description": "blank name", "release_date": "2026-08-01"},
            {"name": "a".repeat(MAX_MODEL_NAME_CHARS + 1), "description": "too long", "release_date": "2026-08-01"},
            {"name": "jev-latest", "release_date": "2026-08-01"}
        ]
    });
    let catalog = parse_model_catalog(body.to_string().as_bytes()).expect("envelope parses");
    assert!(catalog.models.is_empty(), "every entry above is hostile");
    let joined = catalog.rejected.join("\n");
    assert!(joined.contains("name is missing"));
    assert!(joined.contains("name is not a string"));
    assert!(joined.contains("name carries whitespace or an invisible character"));
    assert!(joined.contains("never truncated"));
    assert!(joined.contains("description is missing"), "description is required by the docs");

    // Over-cap description / release_date are TRUNCATED to the cap (kept entry).
    let long_description = "d".repeat(MAX_MODEL_DESCRIPTION_CHARS + 50);
    let long_date = "x".repeat(80);
    let body = json!({
        "models": [
            {"name": "jev-latest", "description": long_description, "release_date": long_date}
        ]
    });
    let catalog = parse_model_catalog(body.to_string().as_bytes()).expect("parses");
    assert_eq!(catalog.models.len(), 1);
    assert_eq!(catalog.models[0].description.chars().count(), MAX_MODEL_DESCRIPTION_CHARS);
    assert_eq!(catalog.models[0].release_date.chars().count(), 64);
}

#[tokio::test]
async fn m7_entry_overflow_is_capped_and_disclosed() {
    let mut entries = Vec::new();
    for i in 0..(MAX_MODEL_CATALOG_ENTRIES + 1) {
        entries.push(json!({
            "name": format!("model-{i}"),
            "description": "d",
            "release_date": "2026-08-01"
        }));
    }
    let body = json!({ "models": entries });
    let catalog = parse_model_catalog(body.to_string().as_bytes()).expect("parses");
    assert_eq!(catalog.models.len(), MAX_MODEL_CATALOG_ENTRIES, "cap is enforced");
    assert_eq!(catalog.models[0].name, "model-0", "wire order preserved");
    assert_eq!(catalog.rejected.len(), 1, "one overflow disclosure");
    assert!(catalog.rejected[0].contains("256"), "disclosure names the cap");
}

// ---------------------------------------------------------------------------
// m8-m9: bounded fetch through the transport abstraction
// ---------------------------------------------------------------------------

#[tokio::test]
async fn m8_fetch_roundtrip_is_explicit_and_counts_one_call() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::ModelsBody(
        documented_catalog_body(),
    )]));
    let limits = catalog_limits();
    let catalog = fetch(&transport, &limits).await.expect("mock fetch succeeds");
    assert_eq!(catalog.models.len(), 2);
    assert_eq!(transport.models_call_count(), 1, "exactly one explicit fetch");
    assert_eq!(transport.call_count(), 0, "the decide path was never touched");
    // The production parser path is shared with the transport (raw body -> parse).
    assert_eq!(catalog.models[0].description, "The most recent stable, official release.");
}

#[tokio::test]
async fn m9_fetch_error_paths_are_honest_and_single_shot() {
    // 401 terminal with a sanitized server request id.
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::ModelsHttpStatus {
        status: 401,
        server_request_id: Some("3f2504e0-4f89-11d3-9a0c-0305e82c3301".to_string()),
    }]));
    let limits = catalog_limits();
    let error = fetch(&transport, &limits).await.expect_err("401 is terminal");
    assert_eq!(error.kind(), "http_status_401");
    assert_eq!(error.status_code(), Some(401));
    assert_eq!(error.server_request_id().unwrap(), "3f2504e0-4f89-11d3-9a0c-0305e82c3301");
    assert_eq!(transport.models_call_count(), 1, "single bounded attempt: no retry");

    // Credential-echo refusal happens in the PRODUCTION transport, which applies
    // `sanitize_opaque_header_value` to the raw header BEFORE constructing the error
    // (same sanitizer the decide path uses). The scripted mock passes ids through
    // verbatim by contract ("pre-sanitized by the test"), so the refusal is asserted
    // at the production sanitizer, and the mock then proves a clean id travels
    // verbatim into the error metadata.
    assert_eq!(
        sanitize_opaque_header_value("Bearer abc123"),
        None,
        "production refusal: credential echoes are never stored"
    );
    assert_eq!(
        sanitize_opaque_header_value("sk-live-abcdef"),
        None,
        "production refusal: sk- shaped echoes are refused"
    );
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::ModelsHttpStatus {
        status: 500,
        server_request_id: Some("3f2504e0-4f89-11d3-9a0c-0305e82c3301".to_string()),
    }]));
    let error = fetch(&transport, &limits).await.expect_err("500");
    assert_eq!(error.kind(), "http_status_500");
    assert_eq!(
        error.server_request_id().unwrap(),
        "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        "a pre-sanitized id travels verbatim through the mock error"
    );

    // Deadline: honest timeout, still exactly one attempt.
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::Timeout]));
    let error = fetch(&transport, &limits).await.expect_err("deadline");
    assert_eq!(error.kind(), "timeout");
    assert_eq!(transport.models_call_count(), 1, "no retry on timeout");

    // Connection dropped: honest unavailable.
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::DroppedConnection]));
    let error = fetch(&transport, &limits).await.expect_err("dropped");
    assert_eq!(error.kind(), "connection");
}

// ---------------------------------------------------------------------------
// m10-m12: explicit-only guarantees and compatibility
// ---------------------------------------------------------------------------

#[tokio::test]
async fn m10_decide_never_touches_the_catalog_and_lanes_stay_separate() {
    // Script: one decide step first, then one catalog step. The decide call must
    // consume ONLY the decide step; the catalog counter must stay zero.
    let transport = Arc::new(MockJevTransport::scripted(vec![
        MockStep::Valid,
        MockStep::ModelsBody(documented_catalog_body()),
    ]));
    let stats = Arc::new(JevStats::default());
    let limits = catalog_limits();
    let credential = SecretString::new("jev-test-key-0123456789abcdef");
    let outcome = decide_with(
        &(transport.clone() as Arc<dyn Transport>),
        &credential,
        JevMode::Compare,
        &limits,
        &stats,
        bundle(),
    )
    .await;
    assert!(outcome.applied || !outcome.records.is_empty(), "decide succeeded");
    assert_eq!(transport.models_call_count(), 0, "decide never fetches the catalog");
    assert_eq!(transport.call_count(), 1, "exactly one decide-path call");

    // The explicit fetch then consumes the catalog step; lanes never cross-feed.
    let catalog = fetch(&transport, &limits).await.expect("explicit fetch works");
    assert_eq!(catalog.models.len(), 2);
    assert_eq!(transport.models_call_count(), 1);
    assert_eq!(transport.call_count(), 1, "the decide counter did not move");
}

#[tokio::test]
async fn m11_default_model_and_catalog_path_are_unchanged() {
    assert_eq!(DEFAULT_MODEL, "jev-latest", "the native default is untouched");
    assert_eq!(MODEL_CATALOG_PATH, "/v1/models", "documented catalog path");
    let limits = JevLimits::default();
    assert_eq!(
        limits.models_endpoint(),
        "https://api.typesafe.ai/v1/models",
        "documented origin + documented path"
    );
    assert!(
        !limits.models_endpoint().contains('?'),
        "no query string, no credential in the URL"
    );
}

#[tokio::test]
async fn m12_unsupported_transport_reports_internal_never_a_fabricated_catalog() {
    let limits = catalog_limits();
    let error = fetch_model_catalog(&PostOnlyTransport, &limits)
        .await
        .expect_err("the provided default must refuse");
    assert_eq!(error.kind(), "internal");
    assert!(
        error.to_string().contains("not supported"),
        "the detail explains the refusal: {error}"
    );
}

// ---------------------------------------------------------------------------
// m13-m14: names are exact identifiers; requested-id validation never echoes input
// ---------------------------------------------------------------------------

#[tokio::test]
async fn m13_model_names_are_exact_identifiers_never_normalized() {
    // Any whitespace or invisible formatting in a NAME rejects the entry instead of
    // being normalized into a different selectable ID (ordinary spaces, NBSP,
    // zero-width space, soft hyphen, word joiner).
    for hostile in [
        "my model",
        " jev-latest",
        "jev-latest ",
        "jev\u{00A0}latest",
        "jev\u{200B}latest",
        "jev\u{00AD}latest",
        "jev\u{2060}latest",
    ] {
        let body = json!({
            "models": [{"name": hostile, "description": "d", "release_date": "2026-08-01"}]
        });
        let catalog = parse_model_catalog(body.to_string().as_bytes()).expect("envelope parses");
        assert!(catalog.models.is_empty(), "whitespace-bearing name must reject: {hostile:?}");
        assert_eq!(catalog.rejected.len(), 1);
        assert!(
            catalog.rejected[0].contains("whitespace")
                || catalog.rejected[0].contains("invisible"),
            "reason names the class: {}",
            catalog.rejected[0]
        );
    }
}

#[tokio::test]
async fn m14_requested_model_id_validation_never_echoes_the_input() {
    // Safe identifiers pass exactly as given.
    assert_eq!(validate_requested_model_id("jev-1.13.0"), Ok(()));
    assert!(validate_requested_model_id("my model").is_err());
    // Hostile inputs fail; the reason never contains the supplied text.
    for hostile in [
        "jev-\r\nlatest",
        "jev\u{00A0}x",
        "\u{202E}evil",
        "Bearer abc123",
        "sk-live-abcdef",
        "",
        "   ",
    ] {
        let reason = validate_requested_model_id(hostile).expect_err("must refuse");
        assert!(reason.chars().count() <= 120, "reason stays bounded");
        if !hostile.is_empty() {
            assert!(
                !reason.contains(hostile),
                "refusal diagnostics must not echo the supplied value: {hostile:?} vs {reason:?}"
            );
        }
    }
    // Effective-credential echo protection: full key, trimmed key, and a partial paste
    // of the key all overlap; a short unrelated id does not.
    let credential = SecretString::new("jev-test-key-0123456789abcdef");
    assert!(id_overlaps_credential("jev-test-key-0123456789abcdef", &credential));
    assert!(id_overlaps_credential(" jev-test-key-0123456789abcdef ", &credential));
    assert!(id_overlaps_credential("key-0123456789", &credential), "partial paste");
    assert!(
        id_overlaps_credential("jev-test-key-0123456789abcdefX", &credential),
        "paste with an extra character still contains the whole key"
    );
    assert!(!id_overlaps_credential("jev-latest", &credential));
    assert!(!id_overlaps_credential("", &credential), "empty id never overlaps");
    assert!(!id_overlaps_credential("   ", &credential), "blank id never overlaps");
}
