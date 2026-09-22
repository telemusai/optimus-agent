//! Synthetic transport only: no profile, credentials, processes, or network access.
use pi_jev::mock::{choice_question, MockJevTransport, MockStep};
use pi_jev::telemetry::session_usage;
use pi_jev::{
    decide_with, DecisionBundle, DecisionCategory, JevLimits, JevMode, JevStats, SecretString,
    Transport,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

fn bundle(session: &str) -> DecisionBundle {
    DecisionBundle {
        session_id: session.into(),
        turn: 1,
        stage: "tool_call".into(),
        state: serde_json::json!({"task":"synthetic"}),
        model: "jev-latest".into(),
        questions: BTreeMap::from([
            (
                "tool_requirement.0".into(),
                choice_question("Need tools?", &[("yes", None), ("no", None)]),
            ),
            (
                "tool_requirement.1".into(),
                choice_question("Need reading?", &[("yes", None), ("no", None)]),
            ),
        ]),
        question_categories: BTreeMap::from([
            (
                "tool_requirement.0".into(),
                DecisionCategory::ToolRequirement,
            ),
            (
                "tool_requirement.1".into(),
                DecisionCategory::ToolRequirement,
            ),
        ]),
    }
}

#[tokio::test]
async fn counts_requests_once_per_bundle_and_marks_retry_usage_partial() {
    let session = uuid::Uuid::new_v4().to_string();
    let mock = Arc::new(MockJevTransport::scripted(vec![
        MockStep::ServerError { status: 503 },
        MockStep::Valid,
    ]));
    let transport: Arc<dyn Transport> = mock.clone();
    let limits = JevLimits {
        max_retries: 1,
        backoff_initial: Duration::ZERO,
        backoff_max: Duration::ZERO,
        ..Default::default()
    };
    let result = decide_with(
        &transport,
        &SecretString::new("synthetic"),
        JevMode::CompareAndActive,
        &limits,
        &Arc::new(JevStats::default()),
        bundle(&session),
    )
    .await;
    assert_eq!(result.records.len(), 2);
    let usage = session_usage(&session).unwrap();
    assert_eq!(
        (
            usage.requests,
            usage.attempts,
            usage.completed,
            usage.in_flight
        ),
        (1, 2, 1, 0)
    );
    assert_eq!(
        (usage.input_tokens, usage.output_tokens),
        (result.usage.input_tokens, result.usage.output_tokens)
    );
    assert!(usage.input_tokens.is_some());
    assert!(usage.incomplete_usage);
    assert_eq!(usage.activity, "idle");
    assert!(usage.last_latency_ms.is_some());
}

#[tokio::test]
async fn disabled_calls_do_not_create_usage_and_cancelled_calls_clear_activity() {
    let session = uuid::Uuid::new_v4().to_string();
    let transport: Arc<dyn Transport> =
        Arc::new(MockJevTransport::scripted(vec![MockStep::SlowResponse {
            delay_ms: 5000,
        }]));
    let limits = JevLimits {
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let stats = Arc::new(JevStats::default());
    let key = SecretString::new("synthetic");
    decide_with(
        &transport,
        &key,
        JevMode::Off,
        &limits,
        &stats,
        bundle(&session),
    )
    .await;
    assert!(session_usage(&session).is_none());
    let pending = decide_with(
        &transport,
        &key,
        JevMode::Active,
        &limits,
        &stats,
        bundle(&session),
    );
    assert!(tokio::time::timeout(Duration::from_millis(50), pending)
        .await
        .is_err());
    let usage = session_usage(&session).unwrap();
    assert_eq!(
        (
            usage.requests,
            usage.attempts,
            usage.in_flight,
            usage.cancelled
        ),
        (1, 1, 0, 1)
    );
    assert_eq!(usage.activity, "idle");
    assert_eq!(usage.input_tokens, None);
    assert!(usage.incomplete_usage);
}

#[tokio::test]
async fn failed_request_is_not_reported_as_measured_zero() {
    let session = uuid::Uuid::new_v4().to_string();
    let transport: Arc<dyn Transport> =
        Arc::new(MockJevTransport::scripted(vec![MockStep::ServerError {
            status: 401,
        }]));
    decide_with(
        &transport,
        &SecretString::new("synthetic"),
        JevMode::Compare,
        &JevLimits::default(),
        &Arc::new(JevStats::default()),
        bundle(&session),
    )
    .await;
    let usage = session_usage(&session).unwrap();
    assert_eq!((usage.failed, usage.completed, usage.in_flight), (1, 0, 0));
    assert_eq!((usage.input_tokens, usage.output_tokens), (None, None));
    assert!(usage.incomplete_usage);
}
