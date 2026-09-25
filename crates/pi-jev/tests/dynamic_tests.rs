//! Offline dynamic decisions use the production parser, limits and lifecycle.
use pi_jev::config::JevMode;
use pi_jev::dynamic::decide_scoped;
use pi_jev::mock::{MockJevTransport, MockStep};
use pi_jev::scheduler::{request_session_retain_stop, session_retain_status, CancellationToken};
use pi_jev::types::{DecisionBundle, DecisionCategory, QuestionSpec};
use pi_jev::{JevLimits, JevStats, JevSystemOne, SecretString};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

fn bundle(session: &str) -> DecisionBundle {
    let questions: BTreeMap<String, QuestionSpec> = BTreeMap::from([
        (
            "coin".into(),
            QuestionSpec::choice(
                "Choose a coin side",
                [("heads", Some("Heads")), ("tails", Some("Tails"))],
            ),
        ),
        (
            "yes".into(),
            QuestionSpec::Noul {
                instructions: Some("Does this mention a coin?".into()),
                criteria: None,
            },
        ),
        (
            "rating".into(),
            QuestionSpec::score("Rate clarity", ["Unclear", "Clear"]),
        ),
    ]);
    DecisionBundle {
        session_id: session.into(),
        turn: 1,
        stage: "dynamic".into(),
        state: json!({"request":"Flip a coin"}),
        model: "jev-test-selected".into(),
        question_categories: questions
            .keys()
            .map(|id| (id.clone(), DecisionCategory::Dynamic))
            .collect(),
        questions,
    }
}
fn client(mode: JevMode, transport: Arc<MockJevTransport>) -> JevSystemOne {
    JevSystemOne::new(
        mode,
        SecretString::new("synthetic-only"),
        transport,
        JevLimits {
            max_retries: 1,
            backoff_initial: Duration::from_millis(1),
            backoff_max: Duration::from_millis(2),
            ..Default::default()
        },
        Arc::new(JevStats::default()),
    )
    .unwrap()
}
async fn wait_for_call(transport: &MockJevTransport) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while transport.call_count() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn dynamic_batches_all_primitives_and_records_session_usage() {
    let transport = Arc::new(MockJevTransport::all_valid());
    let client = client(JevMode::Active, transport.clone());
    let outcome = decide_scoped(
        &client,
        bundle("dynamic-batch"),
        || true,
        std::future::pending(),
    )
    .await
    .unwrap();
    assert_eq!(outcome.records.len(), 3);
    assert!(outcome.skips.is_empty());
    assert!(outcome
        .records
        .iter()
        .all(|record| record.category == DecisionCategory::Dynamic));
    assert_eq!(transport.call_count(), 1);
    assert_eq!(transport.calls()[0].model, "jev-test-selected");
    let usage = pi_jev::telemetry::session_usage("dynamic-batch").unwrap();
    assert_eq!(
        (usage.requests, usage.completed, usage.in_flight),
        (1, 1, 0)
    );
    assert_eq!(usage.input_tokens, Some(312));
    assert_eq!(session_retain_status("dynamic-batch").completed_work, 1);
}

#[tokio::test]
async fn dynamic_disabled_modes_and_pre_cancel_make_no_calls() {
    let off = pi_jev::client::DisabledSystemOne::new(JevMode::Off);
    assert_eq!(
        decide_scoped(&off, bundle("dynamic-off"), || true, std::future::pending())
            .await
            .unwrap_err(),
        "dynamic_disabled"
    );
    for (mode, allowed) in [(JevMode::Compare, true), (JevMode::Active, false)] {
        let transport = Arc::new(MockJevTransport::all_valid());
        let client = client(mode, transport.clone());
        assert_eq!(
            decide_scoped(
                &client,
                bundle("dynamic-disabled"),
                || allowed,
                std::future::pending()
            )
            .await
            .unwrap_err(),
            "dynamic_disabled"
        );
        assert_eq!(transport.call_count(), 0);
    }
    let transport = Arc::new(MockJevTransport::all_valid());
    let client = client(JevMode::Active, transport.clone());
    assert_eq!(
        decide_scoped(&client, bundle("dynamic-precancel"), || true, async {})
            .await
            .unwrap_err(),
        "cancelled"
    );
    assert_eq!(transport.call_count(), 0);
    assert_eq!(session_retain_status("dynamic-precancel").cancelled_work, 1);
}

#[tokio::test]
async fn dynamic_cancellation_drops_transport_before_acknowledgement() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::SlowResponse {
        delay_ms: 5000,
    }]));
    let client = client(JevMode::CompareAndActive, transport.clone());
    let cancel = CancellationToken::new();
    let request_cancel = cancel.clone();
    let task = tokio::spawn(async move {
        decide_scoped(
            &client,
            bundle("dynamic-cancel"),
            || true,
            request_cancel.cancelled(),
        )
        .await
    });
    wait_for_call(&transport).await;
    assert_eq!(
        pi_jev::telemetry::session_usage("dynamic-cancel")
            .unwrap()
            .activity,
        "answering dynamic questions"
    );
    cancel.cancel();
    assert_eq!(task.await.unwrap().unwrap_err(), "cancelled");
    let usage = pi_jev::telemetry::session_usage("dynamic-cancel").unwrap();
    assert_eq!((usage.in_flight, usage.cancelled), (0, 1));
    let status = session_retain_status("dynamic-cancel");
    assert_eq!(
        (
            status.pending_work,
            status.cancelled_work,
            status.failed_work
        ),
        (0, 1, 0)
    );
}

#[tokio::test]
async fn dynamic_settings_changes_discard_inflight_answers() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::SlowResponse {
        delay_ms: 5000,
    }]));
    let client = client(JevMode::Active, transport.clone());
    let allowed = Arc::new(AtomicBool::new(true));
    let gate = allowed.clone();
    let task = tokio::spawn(async move {
        decide_scoped(
            &client,
            bundle("dynamic-settings"),
            || gate.load(Ordering::SeqCst),
            std::future::pending(),
        )
        .await
    });
    wait_for_call(&transport).await;
    allowed.store(false, Ordering::SeqCst);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err(),
        "settings_changed"
    );
    assert_eq!(
        pi_jev::telemetry::session_usage("dynamic-settings")
            .unwrap()
            .in_flight,
        0
    );
}

#[tokio::test]
async fn dynamic_retention_stop_cancels_and_rejects_new_work() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::SlowResponse {
        delay_ms: 5000,
    }]));
    let client = Arc::new(client(JevMode::Active, transport.clone()));
    let task_client = client.clone();
    let task = tokio::spawn(async move {
        decide_scoped(
            task_client.as_ref(),
            bundle("dynamic-retain"),
            || true,
            std::future::pending(),
        )
        .await
    });
    wait_for_call(&transport).await;
    request_session_retain_stop("dynamic-retain");
    assert_eq!(task.await.unwrap().unwrap_err(), "session_stopped");
    assert!(session_retain_status("dynamic-retain").settled);
    assert_eq!(
        decide_scoped(
            client.as_ref(),
            bundle("dynamic-retain"),
            || true,
            std::future::pending()
        )
        .await
        .unwrap_err(),
        "session_stopped"
    );
    assert_eq!(transport.call_count(), 1);
}

#[tokio::test]
async fn dynamic_retries_through_existing_client_without_llm_comparison() {
    let transport = Arc::new(MockJevTransport::scripted(vec![
        MockStep::DroppedConnection,
        MockStep::LowConfidence,
    ]));
    let client = client(JevMode::CompareAndActive, transport.clone());
    let outcome = decide_scoped(
        &client,
        bundle("dynamic-retry"),
        || true,
        std::future::pending(),
    )
    .await
    .unwrap();
    assert_eq!(transport.call_count(), 2);
    assert_eq!(outcome.attempts, 2);
    assert_eq!(outcome.records.len(), 3);
    assert!(outcome.skips.is_empty());
}

#[tokio::test]
async fn dynamic_dropped_tool_future_settles_as_cancelled() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::SlowResponse {
        delay_ms: 5000,
    }]));
    let client = client(JevMode::Active, transport.clone());
    let task = tokio::spawn(async move {
        decide_scoped(
            &client,
            bundle("dynamic-dropped"),
            || true,
            std::future::pending(),
        )
        .await
    });
    wait_for_call(&transport).await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let status = session_retain_status("dynamic-dropped");
    assert_eq!(
        (
            status.pending_work,
            status.cancelled_work,
            status.failed_work
        ),
        (0, 1, 0)
    );
    let usage = pi_jev::telemetry::session_usage("dynamic-dropped").unwrap();
    assert_eq!((usage.in_flight, usage.cancelled), (0, 1));
}
