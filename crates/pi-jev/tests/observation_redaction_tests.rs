//! Transport-boundary proof that a request built from shadow observations
//! carries no credential-shaped material.
//!
//! The observation values are built exactly as the pi-coding-agent bridge
//! builds them (redacted, bounded excerpts; no raw tool arguments), then pushed
//! through the production client path (`decide_with` -> injected transport).
//! The transport is the only difference from production, and it records the
//! serialized request it would have sent.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;

use pi_jev::client::{bundle_with_questions, decide_with, JevLimits, JevStats};
use pi_jev::config::JevMode;
use pi_jev::credential::SecretString;
use pi_jev::error::JevError;
use pi_jev::mock::noul_question;
use pi_jev::redact::{bounded_excerpt, REDACTED};
use pi_jev::types::{BoxFuture, QuestionSpec, SystemOneRequest, SystemOneResponse, Transport};

const SECRETS: [&str; 6] = [
    "sk-live-abcdefghijklmnopqrstuvwxyz",
    "ghp_abcdefghijklmnopqrstuvwxyz",
    "hunter2-the-password",
    "dXNlcjpwYXNzd29yZA==",
    "AKIAIOSFODNN7EXAMPLE",
    "MIIEowIBAAKCAQEAprivatekeymaterial",
];

#[derive(Default)]
struct CapturingTransport {
    bodies: Mutex<Vec<String>>,
}

impl CapturingTransport {
    fn bodies(&self) -> Vec<String> {
        self.bodies.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }
}

impl Transport for CapturingTransport {
    fn post(
        &self,
        request: &SystemOneRequest,
        _timeout: Duration,
    ) -> BoxFuture<Result<SystemOneResponse, JevError>> {
        let body = serde_json::to_string(request).unwrap_or_default();
        self.bodies
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(body);
        // Fail the call: nothing is fabricated from this capture, and the
        // request has already been recorded.
        Box::pin(async {
            Err(JevError::MalformedResponse {
                detail: "capture-only transport".to_string(),
            })
        })
    }
}

fn questions() -> BTreeMap<String, QuestionSpec> {
    let mut questions = BTreeMap::new();
    questions.insert(
        "task_classification.0".to_string(),
        noul_question("Is this a coding task?", "yes", "no"),
    );
    questions
}

/// The observation state the bridge produces for a sensitive turn: a task
/// excerpt, a result excerpt, and tool identity only.
fn sensitive_observation_state() -> serde_json::Value {
    let raw_task = "deploy with TYPESAFE_API_KEY=sk-live-abcdefghijklmnopqrstuvwxyz and \
                    Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz";
    let raw_result = "wrote ~/.netrc with password=hunter2-the-password and \
                      -----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAprivatekeymaterial";
    json!({
        "user_text_excerpt": bounded_excerpt(raw_task, 400),
        "result_excerpt": bounded_excerpt(raw_result, 400),
        "observed_tools": ["bash"],
        "tool_name": "bash",
        "args_omitted": true,
        "message_count": 4,
    })
}

#[tokio::test]
async fn a_request_built_from_observations_carries_no_credential_shapes() {
    let capture = Arc::new(CapturingTransport::default());
    let transport: Arc<dyn Transport> = Arc::clone(&capture) as Arc<dyn Transport>;
    let stats = Arc::new(JevStats::default());
    let bundle = bundle_with_questions(
        "synthetic-session",
        1,
        "agent_end",
        sensitive_observation_state(),
        "jev-1.13.0",
        questions(),
    );
    let outcome = decide_with(
        &transport,
        &SecretString::new("synthetic-test-credential"),
        JevMode::Compare,
        &JevLimits::default(),
        &stats,
        bundle,
    )
    .await;
    // The capture transport always fails; the outcome must be a skip, not a
    // fabricated answer.
    assert!(outcome.records.is_empty(), "{outcome:?}");

    let bodies = capture.bodies();
    assert!(!bodies.is_empty(), "no request reached the transport");
    for body in &bodies {
        for secret in SECRETS {
            assert!(!body.contains(secret), "{secret} reached the transport: {body}");
        }
        assert!(body.contains(REDACTED), "expected a redaction marker: {body}");
        assert!(!body.contains("args_excerpt"), "{body}");
    }
}

#[tokio::test]
async fn a_multi_megabyte_observation_stays_bounded_and_redacted() {
    let raw = format!(
        "{}{}",
        "x".repeat(4 * 1024 * 1024),
        "token=ghp_abcdefghijklmnopqrstuvwxyz"
    );
    let state = json!({ "result_excerpt": bounded_excerpt(&raw, 400) });
    let serialized = serde_json::to_string(&state).unwrap();
    assert!(serialized.len() < 2_048, "bounded state was {} bytes", serialized.len());
    assert!(!serialized.contains("ghp_abcdefghijklmnopqrstuvwxyz"));

    let capture = Arc::new(CapturingTransport::default());
    let transport: Arc<dyn Transport> = Arc::clone(&capture) as Arc<dyn Transport>;
    let stats = Arc::new(JevStats::default());
    let bundle = bundle_with_questions(
        "synthetic-session",
        2,
        "agent_end",
        state,
        "jev-1.13.0",
        questions(),
    );
    let _ = decide_with(
        &transport,
        &SecretString::new("synthetic-test-credential"),
        JevMode::Compare,
        &JevLimits::default(),
        &stats,
        bundle,
    )
    .await;
    for body in &capture.bodies() {
        assert!(!body.contains("ghp_abcdefghijklmnopqrstuvwxyz"), "{body}");
    }
}
