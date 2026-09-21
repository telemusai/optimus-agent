//! Live SystemOne connectivity probe.
//!
//! Ignored by default. It performs a real network request and needs a real
//! credential, so it never runs in the normal test suite:
//!
//! ```text
//! TYPESAFE_API_KEY=... cargo test -p pi-jev --test live_systemone -- --ignored --nocapture
//! ```
//!
//! The probe asserts only client-contract truth: one transport attempt, a
//! validated outcome, `applied == false`, and a record or an explicit skip for
//! the question it asked. It never prints the credential, the raw request or
//! the raw response body.
//!
//! Endpoint override (staging or a local replay server):
//! `JEV_BASE_URL=https://... cargo test ... -- --ignored`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use pi_jev::client::{bundle_with_questions, JevLimits, JevStats, JevSystemOne};
use pi_jev::config::{JevMode, ENV_JEV_API_KEY, ENV_TYPESAFE_API_KEY};
use pi_jev::credential::SecretString;
use pi_jev::mock::noul_question;
use pi_jev::types::SystemOne;

/// Credential from the documented environment names, primary first. The value
/// is wrapped immediately and never logged.
fn live_credential() -> Option<SecretString> {
    for name in [ENV_TYPESAFE_API_KEY, ENV_JEV_API_KEY] {
        if let Ok(value) = std::env::var(name) {
            let trimmed = value.trim().to_string();
            if !trimmed.is_empty() {
                return Some(SecretString::new(trimmed));
            }
        }
    }
    None
}

#[tokio::test]
#[ignore = "live network probe; set TYPESAFE_API_KEY (or JEV_API_KEY) and pass --ignored"]
async fn live_systemone_round_trip() {
    let Some(credential) = live_credential() else {
        panic!("set TYPESAFE_API_KEY or JEV_API_KEY before running this live probe");
    };
    let mut limits = JevLimits::default();
    if let Ok(base_url) = std::env::var("JEV_BASE_URL") {
        if !base_url.trim().is_empty() {
            limits.base_url = base_url.trim().trim_end_matches('/').to_string();
        }
    }
    limits.timeout = Duration::from_secs(30);
    limits.max_retries = 1;
    let endpoint = limits.endpoint();
    let stats = Arc::new(JevStats::default());
    let client = JevSystemOne::with_http(JevMode::Compare, credential, limits, Arc::clone(&stats))
        .expect("Compare client builds with a credential");

    let question_id = "task_classification.0".to_string();
    let mut questions = BTreeMap::new();
    questions.insert(
        question_id.clone(),
        noul_question(
            "Does this task require writing or editing repository files?",
            "the task writes or edits files",
            "the task only reads or inspects",
        ),
    );
    let bundle = bundle_with_questions(
        "live-probe-session",
        1,
        "turn_start",
        serde_json::json!({"user_text_excerpt": "Live SystemOne probe: read the repository README."}),
        "jev-latest",
        questions,
    );

    let outcome = client.decide(bundle).await;
    let snapshot = stats.snapshot();
    println!("endpoint={endpoint}");
    println!(
        "attempts={} successes={} failures={} retries={} rate_limited={} timeouts={} malformed={}",
        snapshot.attempts,
        snapshot.successes,
        snapshot.failures,
        snapshot.retries,
        snapshot.rate_limited,
        snapshot.timeouts,
        snapshot.malformed
    );
    println!(
        "response_model={:?} input_tokens={:?} output_tokens={:?}",
        outcome.response_model, outcome.usage.input_tokens, outcome.usage.output_tokens
    );
    for record in &outcome.records {
        println!(
            "record id={} category={:?} drift={:?} applied={}",
            record.question_id,
            record.category,
            record.drift(),
            record.applied
        );
    }
    for (id, reason) in &outcome.skips {
        println!("skip id={id} reason={reason}");
    }

    assert!(outcome.attempts >= 1, "a live probe must attempt a request");
    assert!(!outcome.applied, "Jev output is never applied in this build");
    assert!(
        outcome.records.iter().any(|record| record.question_id == question_id)
            || outcome.skips.iter().any(|(id, _)| id == &question_id),
        "a live round trip must record or explicitly skip the question it asked"
    );
    assert!(
        snapshot.attempts >= 1,
        "stats must count the attempt that was actually made"
    );
}
