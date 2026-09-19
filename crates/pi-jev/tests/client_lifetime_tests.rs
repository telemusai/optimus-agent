use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use pi_jev::client::{bundle_with_questions, parse_retry_after};
use pi_jev::mock::{noul_question, MockJevTransport, MockStep};
use pi_jev::{BoxFuture, JevError, JevLimits, JevMode, JevStats, JevSystemOne,
    SecretString, SystemOne, SystemOneRequest, SystemOneResponse, Transport};

struct NeverReturns;

impl Transport for NeverReturns {
    fn post(&self, _: &SystemOneRequest, _: Duration) -> BoxFuture<Result<SystemOneResponse, JevError>> {
        Box::pin(std::future::pending())
    }
}

fn decision(client: &JevSystemOne) -> BoxFuture<pi_jev::DecisionOutcome> {
    client.decide(bundle_with_questions(
        "lifetime-test", 0, "turn_start", serde_json::json!({}), "jev-latest",
        BTreeMap::from([("task_classification.0".into(), noul_question("Read only?", "yes", "no"))]),
    ))
}

fn client(transport: Arc<dyn Transport>, stats: Arc<JevStats>, limits: JevLimits) -> JevSystemOne {
    JevSystemOne::new(JevMode::Compare, SecretString::new("synthetic-test-key"), transport, limits, stats).unwrap()
}

#[tokio::test]
async fn cancelling_a_pending_request_releases_in_flight_accounting() {
    let stats = Arc::new(JevStats::default());
    let client = client(Arc::new(NeverReturns), stats.clone(), JevLimits::default());
    let mut pending = decision(&client);
    assert!(tokio::time::timeout(Duration::from_millis(10), &mut pending).await.is_err());
    assert_eq!(stats.snapshot().in_flight, 1);
    drop(pending);
    assert_eq!(stats.snapshot().in_flight, 0);
    assert_eq!(stats.snapshot().successes, 0);
}

#[tokio::test]
async fn client_bounds_even_a_transport_that_ignores_the_deadline() {
    let stats = Arc::new(JevStats::default());
    let client = client(Arc::new(NeverReturns), stats.clone(), JevLimits {
        timeout: Duration::from_millis(20), max_retries: 0, ..JevLimits::default()
    });
    let result = tokio::time::timeout(Duration::from_secs(2), decision(&client)).await.unwrap();
    assert_eq!(result.attempts, 1);
    assert_eq!(result.skips[0].1, "timeout");
    assert_eq!(stats.snapshot().in_flight, 0);
    assert_eq!(stats.snapshot().timeouts, 1);
}

#[tokio::test]
async fn server_delay_outside_the_budget_stops_without_an_early_retry() {
    let transport = Arc::new(MockJevTransport::scripted(vec![
        MockStep::RateLimited { retry_after_secs: 10 }, MockStep::Valid,
    ]));
    let stats = Arc::new(JevStats::default());
    let client = client(transport.clone(), stats.clone(), JevLimits {
        timeout: Duration::from_millis(50), max_retries: 1,
        backoff_max: Duration::from_millis(1), ..JevLimits::default()
    });
    let result = tokio::time::timeout(Duration::from_secs(2), decision(&client)).await.unwrap();
    assert_eq!(result.attempts, 1);
    assert!(result.records.is_empty());
    assert_eq!(transport.call_count(), 1);
    assert_eq!(stats.snapshot().retries, 0);
    assert_eq!(stats.snapshot().in_flight, 0);
}

#[tokio::test]
async fn retry_attempt_only_receives_the_remaining_total_budget() {
    let transport = Arc::new(MockJevTransport::scripted(vec![MockStep::Timeout]));
    let stats = Arc::new(JevStats::default());
    let client = client(transport.clone(), stats.clone(), JevLimits {
        timeout: Duration::from_millis(100), max_retries: 1,
        backoff_initial: Duration::from_millis(50), backoff_max: Duration::from_millis(50),
        ..JevLimits::default()
    });
    let result = tokio::time::timeout(Duration::from_secs(2), decision(&client)).await.unwrap();
    assert_eq!(result.attempts, 2);
    let calls = transport.calls();
    assert_eq!(calls.len(), 2);
    assert!(calls[1].timeout < Duration::from_millis(60));
    assert_eq!(stats.snapshot().in_flight, 0);
}

#[test]
fn retry_after_http_date_preserves_the_server_delay() {
    let future = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc2822();
    let delay = parse_retry_after(&future).unwrap();
    assert!(delay > Duration::from_secs(58) && delay <= Duration::from_secs(60));
}
