//! Comparison-mode tests: all eleven categories, explicit skips, bundling,
//! malicious payload hardening, scheduler behavior and correlation records.
//! Uses the mock transport only; no network.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use pi_jev::client::{JevLimits, JevStats, JevSystemOne};
use pi_jev::config::JevMode;
use pi_jev::credential::SecretString;
use pi_jev::correlate::{read_records, RECORD_SCHEMA_VERSION};
use pi_jev::error::JevError;
use pi_jev::evaluators::{all_evaluators, for_boundary, EvaluatorOutput};
use pi_jev::hooks::{JevObserver, JevObserverConfig, PROMPT_VERSION};
use pi_jev::mock::{MockJevTransport, MockStep};
use pi_jev::scheduler::SchedulerConfig;
use pi_jev::snapshot::{
    bound_json, truncate_text, SnapshotStage, StateSnapshot, STATE_SCHEMA_VERSION,
};
use pi_jev::types::{
    Answer, DecisionCategory, QuestionSpec, SystemOneRequest, SystemOneResponse, Transport,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Synthetic credential for Compare clients in tests. Never a real key.
const SYNTHETIC_KEY: &str = "jev-test-key-not-real";

/// Test-only transport over a per-request closure (dynamic answers).
struct FnTransport<F>(F)
where
    F: Fn(&SystemOneRequest, Duration) -> Result<SystemOneResponse, JevError> + Send + Sync;

impl<F> Transport for FnTransport<F>
where
    F: Fn(&SystemOneRequest, Duration) -> Result<SystemOneResponse, JevError> + Send + Sync,
{
    fn post(
        &self,
        request: &SystemOneRequest,
        timeout: Duration,
    ) -> pi_jev::types::BoxFuture<Result<SystemOneResponse, JevError>> {
        let result = (self.0)(request, timeout);
        Box::pin(async move { result })
    }
}

fn snapshot(stage: SnapshotStage, state: Value) -> StateSnapshot {
    StateSnapshot::new(
        stage,
        "opaque-session",
        3,
        7,
        Some("model-a".to_string()),
        state,
        Vec::new(),
    )
    .unwrap()
}

fn turn_start_snapshot() -> StateSnapshot {
    snapshot(
        SnapshotStage::TurnStart,
        json!({
            "user_text_excerpt": "fix the failing parser test in pkg/foo",
            "message_count": 5,
            "observed_tools": ["bash", "read", "edit"],
            "model_allowlist": ["allow-a", "allow-b"],
            "memory_excerpt": "project uses vitest",
        }),
    )
}

fn agent_end_snapshot() -> StateSnapshot {
    snapshot(
        SnapshotStage::AgentEnd,
        json!({
            "result_excerpt": "changed two files; 1 test still failing",
            "tool_call_count": 3,
            "error_count": 0,
        }),
    )
}

struct ObserverFixture<T: Transport + 'static> {
    observer: Arc<JevObserver>,
    mock: Arc<T>,
    records_path: std::path::PathBuf,
    stats: Arc<JevStats>,
}

fn make_observer<T>(
    transport_impl: T,
    records_path: std::path::PathBuf,
    mode: JevMode,
    scheduler: SchedulerConfig,
) -> ObserverFixture<T>
where
    T: Transport + 'static,
{
    let mock = Arc::new(transport_impl);
    let transport: Arc<dyn Transport> = mock.clone();
    let stats: Arc<JevStats> = Arc::new(JevStats::default());
    let system_one: Arc<dyn pi_jev::types::SystemOne> = Arc::new(
        JevSystemOne::new(
            JevMode::Compare,
            SecretString::new(SYNTHETIC_KEY),
            transport,
            JevLimits::default(),
            Arc::clone(&stats),
        )
        .unwrap(),
    );
    let config = JevObserverConfig {
        mode_gate: Arc::new(move |_| mode),
        enabled_categories: HashSet::new(),
        scheduler,
        ..JevObserverConfig::default()
    };
    let observer = JevObserver::new(config, system_one, records_path.clone());
    ObserverFixture {
        observer,
        mock,
        records_path,
        stats,
    }
}

fn fast_scheduler() -> SchedulerConfig {
    SchedulerConfig {
        request_deadline: Duration::from_secs(2),
        min_interval: Duration::ZERO,
        ..SchedulerConfig::default()
    }
}

/// Poll until the records file gains a record matching `predicate` or time
/// runs out; bounded so a broken pipeline fails fast instead of hanging.
async fn wait_for_records(
    records_path: &std::path::Path,
    predicate: impl Fn(&pi_jev::correlate::CorrelationRecord) -> bool,
) -> Vec<pi_jev::correlate::CorrelationRecord> {
    for _ in 0..200 {
        let records = read_records(records_path, 1 << 20);
        if records.iter().any(|record| predicate(record)) {
            return records;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    read_records(records_path, 1 << 20)
}

fn turn_payload(session_id: &str) -> Value {
    json!({
        "session_id": session_id,
        "turn": 1,
        "model": "model-a",
        "state": {
            "user_text_excerpt": "build a parser",
            "message_count": 4,
            "observed_tools": ["read"],
            "model_allowlist": ["allow-a"],
            "memory_excerpt": "m",
            "result_excerpt": "r",
        },
    })
}

// ---------------------------------------------------------------------------
// 1. All eleven categories produce correct questions from synthetic snapshots
// ---------------------------------------------------------------------------

#[test]
fn all_eleven_categories_produce_questions() {
    let evaluators = all_evaluators();
    assert_eq!(evaluators.len(), 11);

    let ts = turn_start_snapshot();
    let mut produced: Vec<String> = Vec::new();
    for evaluator in for_boundary(SnapshotStage::TurnStart) {
        match evaluator.evaluate(&ts) {
            EvaluatorOutput::Questions(questions) => {
                for question in questions {
                    assert!(
                        question
                            .question_id
                            .starts_with(&format!("{}.", evaluator.category().as_str())),
                        "question id must be <category>.<n>: {}",
                        question.question_id
                    );
                    produced.push(question.question_id);
                }
            }
            EvaluatorOutput::Skipped(reason) => {
                // Only category 4 (ToolCall boundary too, but its TurnStart
                // branch needs observed tools) and 6 (ModelSelect only) may
                // skip in this synthetic state.
                assert!(
                    matches!(
                        evaluator.category(),
                        DecisionCategory::ToolCandidates | DecisionCategory::SubagentModelRouting
                    ),
                    "unexpected TurnStart skip for {}: {}",
                    evaluator.category().as_str(),
                    reason
                );
            }
        }
    }
    for category in [
        DecisionCategory::TaskClassification,
        DecisionCategory::Complexity,
        DecisionCategory::ToolRequirement,
        DecisionCategory::ToolCandidates,
        DecisionCategory::SubagentRequirement,
        DecisionCategory::ContextRelevance,
        DecisionCategory::MemoryRelevance,
    ] {
        let id = category.as_str();
        assert!(
            produced.iter().any(|q| q.starts_with(&format!("{id}."))),
            "category {id} produced no question at TurnStart"
        );
    }

    // ToolCall: category 4 produces the suitability question.
    let tool_call = snapshot(
        SnapshotStage::ToolCall,
        json!({ "tool_name": "bash", "tool_call_id": "tc1" }),
    );
    let tool_outputs: Vec<EvaluatorOutput> = for_boundary(SnapshotStage::ToolCall)
        .iter()
        .map(|evaluator| evaluator.evaluate(&tool_call))
        .collect();
    assert!(tool_outputs
        .iter()
        .any(|output| matches!(output, EvaluatorOutput::Questions(_))));

    // AgentEnd: categories 9-11 all ask.
    let end = agent_end_snapshot();
    let produced_end = for_boundary(SnapshotStage::AgentEnd)
        .iter()
        .filter_map(|evaluator| match evaluator.evaluate(&end) {
            EvaluatorOutput::Questions(questions) => Some(questions),
            EvaluatorOutput::Skipped(_) => None,
        })
        .flatten()
        .map(|question| question.question_id)
        .collect::<Vec<_>>();
    assert_eq!(produced_end.len(), 3, "AgentEnd must ask 9, 10 and 11");
    for id in [
        "continue_stop_escalate.0",
        "result_sufficiency.0",
        "first_pass_verification.0",
    ] {
        assert!(produced_end.contains(&id.to_string()), "missing {id}");
    }

    // ModelSelect: category 6 advisory-only inside the user-approved allowlist.
    let select = snapshot(
        SnapshotStage::ModelSelect,
        json!({ "model_allowlist": ["allow-a", "allow-b"], "selected_model": "allow-a" }),
    );
    let routing: Vec<pi_jev::evaluators::PreparedQuestion> = for_boundary(SnapshotStage::ModelSelect)
        .iter()
        .flat_map(|evaluator| match evaluator.evaluate(&select) {
            EvaluatorOutput::Questions(questions) => questions,
            EvaluatorOutput::Skipped(_) => Vec::new(),
        })
        .collect();
    assert_eq!(routing.len(), 1, "routing question at ModelSelect");
    let QuestionSpec::Choice { criteria, instructions } = &routing[0].spec else {
        panic!("routing must be a choice question");
    };
    assert!(criteria.contains_key("allow-a"));
    assert!(criteria.contains_key("none"));
    assert!(instructions.contains("never switches"), "advisory wording required");
}

#[test]
fn routing_without_allowlist_is_skipped() {
    let select = snapshot(
        SnapshotStage::ModelSelect,
        json!({ "model_allowlist": [], "selected_model": "anything" }),
    );
    let outputs: Vec<EvaluatorOutput> = for_boundary(SnapshotStage::ModelSelect)
        .iter()
        .map(|evaluator| evaluator.evaluate(&select))
        .collect();
    assert!(outputs.iter().any(
        |output| matches!(output, EvaluatorOutput::Skipped(reason) if reason == "no_model_allowlist")
    ));
}

// ---------------------------------------------------------------------------
// 2. Missing inputs give explicit skipped reasons
// ---------------------------------------------------------------------------

#[test]
fn missing_inputs_produce_explicit_skips() {
    let empty = snapshot(SnapshotStage::TurnStart, json!({}));
    for evaluator in for_boundary(SnapshotStage::TurnStart) {
        let expected = match evaluator.category() {
            DecisionCategory::ToolCandidates => "no_tool_catalog_observed",
            DecisionCategory::SubagentModelRouting => "no_model_allowlist",
            DecisionCategory::ContextRelevance => "no_context_messages",
            DecisionCategory::MemoryRelevance => "no_memory_state",
            _ => "no_task_text",
        };
        assert!(
            matches!(evaluator.evaluate(&empty), EvaluatorOutput::Skipped(reason) if reason == expected),
            "category {} must skip with {expected}",
            evaluator.category().as_str()
        );
    }
    let no_result = snapshot(SnapshotStage::AgentEnd, json!({}));
    for evaluator in for_boundary(SnapshotStage::AgentEnd) {
        assert!(matches!(evaluator.evaluate(&no_result), EvaluatorOutput::Skipped(_)));
    }
}

// ---------------------------------------------------------------------------
// 3. Bundling: independent questions, one request per stage, no dependencies
// ---------------------------------------------------------------------------

#[test]
fn bundling_is_independent_and_bounded() {
    let ts = turn_start_snapshot();
    let mut bundle: BTreeMap<String, QuestionSpec> = BTreeMap::new();
    for evaluator in for_boundary(SnapshotStage::TurnStart) {
        if let EvaluatorOutput::Questions(questions) = evaluator.evaluate(&ts) {
            for question in questions {
                bundle.insert(question.question_id, question.spec);
            }
        }
    }
    assert!(bundle.len() >= 5, "expected multiple TurnStart questions");
    for (id, spec) in &bundle {
        let text = serde_json::to_string(spec).unwrap();
        for other in bundle.keys() {
            if other != id {
                assert!(
                    !text.contains(other.as_str()),
                    "question {id} must not depend on {other}"
                );
            }
        }
    }
    for id in bundle.keys() {
        let (prefix, n) = id.rsplit_once('.').unwrap();
        assert!(
            DecisionCategory::parse(prefix).is_some(),
            "bad category prefix in {id}"
        );
        assert_eq!(n, "0");
    }
}

// ---------------------------------------------------------------------------
// 4. Snapshot fingerprinting and bounding
// ---------------------------------------------------------------------------

#[test]
fn snapshot_fingerprint_is_canonical_and_stable() {
    let a = snapshot(SnapshotStage::TurnStart, json!({ "b": 1, "a": 2 }));
    let b = snapshot(SnapshotStage::TurnStart, json!({ "a": 2, "b": 1 }));
    assert_eq!(
        a.fingerprint, b.fingerprint,
        "key order must not change the fingerprint"
    );
    assert_eq!(a.fingerprint.len(), 64);
    let c = snapshot(SnapshotStage::TurnStart, json!({ "a": 3 }));
    assert_ne!(a.fingerprint, c.fingerprint);
    assert_eq!(STATE_SCHEMA_VERSION, "jev.state/1");
}

#[test]
fn enforce_age_never_deletes_credential_or_settings_files() {
    let dir = tempfile::Builder::new()
        .prefix("jev-age-safe-")
        .tempdir()
        .unwrap();
    let jev_dir = dir.path().join("jev");
    std::fs::create_dir_all(&jev_dir).unwrap();
    let old_stamp = std::time::SystemTime::now() - Duration::from_secs(60 * 60 * 24 * 400);
    let protected = [
        ("jev-settings.json", "{}".as_bytes()),
        ("jev-credential.json", "not-a-real-key".as_bytes()),
    ];
    for (name, body) in protected {
        let path = jev_dir.join(name);
        let file = std::fs::File::create(&path).unwrap();
        file.set_times(std::fs::FileTimes::new()
            .set_modified(old_stamp)
            .set_accessed(old_stamp))
        .unwrap();
        drop(file);
        std::fs::write(&path, body).unwrap();
        // Re-apply the old stamp after the content write (write refreshes mtime).
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_times(std::fs::FileTimes::new()
            .set_modified(old_stamp)
            .set_accessed(old_stamp))
        .unwrap();
        drop(file);
    }
    let correlator = pi_jev::correlate::Correlator::with_retention(
        jev_dir.join("records.jsonl"),
        0.0,
        pi_jev::correlate::RetentionPolicy {
            max_bytes: 1,
            max_age_days: 14,
        },
    );
    // Two records: the first creates the file, the second triggers rotation
    // and the age pass. The age pass sees the pre-dated protected files.
    for _ in 0..2 {
        correlator.record_skipped_category(
            "sess-age",
            1,
            "turn_start",
            pi_jev::types::DecisionCategory::TaskClassification,
            "age_probe",
            "jev-compare-prompts/1",
            "compare",
        );
    }
    let survivors: Vec<String> = std::fs::read_dir(&jev_dir)
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.file_name().to_string_lossy().to_string()))
        .collect();
    assert!(
        survivors.iter().any(|name| name == "jev-settings.json"),
        "settings must survive retention: {survivors:?}"
    );
    assert!(
        survivors.iter().any(|name| name == "jev-credential.json"),
        "credential file must survive retention: {survivors:?}"
    );
    assert!(
        survivors
            .iter()
            .filter(|name| name.starts_with("records.jsonl"))
            .count()
            > 0,
        "record files are still rotated: {survivors:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn budget_exceeded_drops_are_counted() {
    let transport = MockJevTransport::all_valid();
    let fixture = make_observer(
        transport,
        temp_records(),
        JevMode::Compare,
        SchedulerConfig {
            max_requests_per_minute: 1,
            request_deadline: Duration::from_secs(5),
            min_interval: Duration::ZERO,
            ..SchedulerConfig::default()
        },
    );
    for session in ["sess-b1", "sess-b2", "sess-b3"] {
        fixture.observer.observe("turn_start", &turn_payload(session));
    }
    // Poll until the budget metric lands; the first request completes, the
    // rest are dropped with a truthful metric.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
    loop {
        let metrics = fixture.observer.scheduler_metrics();
        if metrics["dropped_budget"].as_u64().unwrap_or(0) >= 1 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "budget drops must be counted: {metrics:?}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    fixture.observer.shutdown();
}

#[test]
fn bounding_caps_strings_arrays_and_depth() {
    let huge = json!({
        "long": "x".repeat(5000),
        "wide": (0..100).collect::<Vec<u32>>(),
        "deep": { "a": { "b": { "c": { "d": { "e": { "f": { "g": 1 } } } } } } },
    });
    let bounded = bound_json(huge, 0);
    let serialized = serde_json::to_string(&bounded).unwrap();
    assert!(serialized.contains("[truncated]"));
    assert!(serialized.contains("_truncated"));
    assert!(bounded
        .get("long")
        .and_then(Value::as_str)
        .map(|s| s.len() < 5000)
        .unwrap_or(false));
}

#[test]
fn oversized_state_is_bounded_then_rejected_at_the_backstop() {
    // Bounding first: a 20k string is excerpt-capped with a marker, not
    // silently accepted verbatim and not a fabricated rejection.
    let huge = json!({ "blob": "y".repeat(20_000) });
    let built = StateSnapshot::new(
        SnapshotStage::TurnStart,
        "session",
        0,
        0,
        None,
        huge,
        Vec::new(),
    )
    .expect("bounded state must be accepted");
    let serialized = serde_json::to_string(&built.state).unwrap();
    assert!(serialized.contains("[truncated]"), "{serialized}");
    assert!(built.fingerprint.len() == 64);

    // Backstop: a state that is STILL over cap after bounding (many
    // excerpt-capped strings) is rejected, never persisted.
    let pathological = json!({
        "rows": (0..32)
            .map(|i| (format!("k{i}"), serde_json::Value::String("y".repeat(400))))
            .collect::<serde_json::Map<String, serde_json::Value>>(),
    });
    let built = StateSnapshot::new(
        SnapshotStage::TurnStart,
        "session",
        0,
        0,
        None,
        pathological,
        Vec::new(),
    );
    assert!(built.is_err(), "over-cap state must be rejected");
}

#[test]
fn truncate_text_respects_char_boundaries() {
    let (cut, truncated) = truncate_text("hello", 10);
    assert_eq!(cut, "hello");
    assert!(!truncated);
    let fifty = "x".repeat(50);
    let (cut, truncated) = truncate_text(&fifty, 10);
    assert_eq!(cut.chars().count(), 10);
    assert!(truncated);
}

// ---------------------------------------------------------------------------
// 5. Scheduler behavior
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scheduler_completes_a_bundle_and_records_it() {
    let fixture = make_observer(
        MockJevTransport::all_valid(),
        temp_records(),
        JevMode::Compare,
        fast_scheduler(),
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-1"));
    fixture.observer.observe("agent_end", &turn_payload("sess-1"));
    let records = wait_for_records(
        fixture.records_path(),
        |record| record.selected_value.is_some() && record.stage == "agent_end",
    )
    .await;
    assert!(!records.is_empty(), "completed requests must produce records");
    for record in &records {
        assert_eq!(record.schema_version, RECORD_SCHEMA_VERSION);
        assert!(!record.applied, "applied must ALWAYS be false");
        assert_eq!(record.mode, "compare");
        if record.selected_value.is_some() {
            assert!(record.response_model.is_some(), "response model recorded");
            assert!(record.duration_ms.is_some());
            assert!(record.request_start_ts.is_some() && record.terminal_ts.is_some());
        }
    }
    assert!(
        records
            .iter()
            .any(|record| record.response_model.as_deref() == Some("jev-1.13.0")),
        "actual response model recorded (drift visible)"
    );
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scheduler_honors_deadline() {
    let slow = MockJevTransport::scripted(vec![MockStep::SlowResponse {
        delay_ms: 5_000,
    }]);
    let fixture = make_observer(
        slow,
        temp_records(),
        JevMode::Compare,
        SchedulerConfig {
            request_deadline: Duration::from_millis(120),
            min_interval: Duration::ZERO,
            ..SchedulerConfig::default()
        },
    );
    fixture.observer.observe(
        "turn_start",
        &json!({
            "session_id": "sess-dl",
            "turn": 0,
            "state": { "user_text_excerpt": "t", "message_count": 1 },
        }),
    );
    let records = wait_for_records(fixture.records_path(), |record| {
        record
            .skipped_reason
            .as_deref()
            .map(|s| s.starts_with("request_failed"))
            .unwrap_or(false)
    })
    .await;
    assert!(
        records.iter().any(|record| record
            .skipped_reason
            .as_deref()
            .map(|s| s.contains("timeout"))
            .unwrap_or(false)),
        "deadline timeout must surface as a logged skip: {records:?}"
    );
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scheduler_drops_on_full_queue_with_metric() {
    let blocked = MockJevTransport::scripted(vec![MockStep::SlowResponse {
        delay_ms: 30_000,
    }]);
    let fixture = make_observer(
        blocked,
        temp_records(),
        JevMode::Compare,
        SchedulerConfig {
            queue_capacity: 1,
            concurrency: 1,
            request_deadline: Duration::from_secs(30),
            min_interval: Duration::ZERO,
            ..SchedulerConfig::default()
        },
    );
    // Job 1 occupies the single (blocked) worker.
    fixture.observer.observe("turn_start", &turn_payload("sess-q"));
    tokio::time::sleep(Duration::from_millis(120)).await;
    // Job 2 fills the one queue slot.
    fixture.observer.observe("agent_end", &turn_payload("sess-q"));
    tokio::time::sleep(Duration::from_millis(40)).await;
    // Job 3 must drop: the queue is full.
    fixture.observer.observe(
        "tool_call",
        &json!({
            "session_id": "sess-q",
            "turn": 1,
            "tool_name": "read",
            "state": { "tool_name": "read" },
        }),
    );
    tokio::time::sleep(Duration::from_millis(80)).await;
    let metrics = fixture.observer.scheduler_metrics();
    assert_eq!(
        metrics["dropped_queue_full"].as_u64().unwrap_or(0),
        1,
        "queue full must drop with metric: {metrics}"
    );
    let records = read_records(fixture.records_path(), 1 << 20);
    let dropped = records
        .iter()
        .find(|record| record.skipped_reason.as_deref() == Some("queue_full"));
    assert!(
        dropped.is_some(),
        "dropped requests must be recorded: {records:?}"
    );
    assert_eq!(
        dropped.unwrap().baseline_actual_choice.as_deref(),
        Some("read"),
        "baseline captured at the boundary is preserved on drop"
    );
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn circuit_breaker_opens_after_consecutive_failures() {
    let transport = MockJevTransport::scripted(vec![MockStep::ServerError { status: 500 }]);
    let fixture = make_observer(
        transport,
        temp_records(),
        JevMode::Compare,
        SchedulerConfig {
            max_consecutive_failures: 2,
            concurrency: 1,
            breaker_cooldown: Duration::from_secs(30),
            request_deadline: Duration::from_secs(5),
            min_interval: Duration::ZERO,
            ..SchedulerConfig::default()
        },
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-cb"));
    fixture.observer.observe("agent_end", &turn_payload("sess-cb"));
    fixture.observer.observe(
        "model_select",
        &json!({
            "session_id": "sess-cb",
            "turn": 1,
            "selected_model": "allow-a",
            "state": { "model_allowlist": ["allow-a"] },
        }),
    );
    // Client retries own their backoff; poll until the breaker metric lands
    // instead of assuming fixed timing.
    let mut metrics = serde_json::json!({});
    let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
    loop {
        metrics = fixture.observer.scheduler_metrics();
        if metrics["dropped_circuit_open"].as_u64().unwrap_or(0) >= 1 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "breaker must open after consecutive failures: {metrics}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let records = read_records(fixture.records_path(), 1 << 20);
    assert!(records
        .iter()
        .any(|record| record.skipped_reason.as_deref() == Some("circuit_open")));
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retry_after_is_honored_before_final_success() {
    let calls = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let calls_for_responder = Arc::clone(&calls);
    let transport = FnTransport(move |request: &SystemOneRequest, _: Duration| {
        let prior = calls_for_responder.fetch_add(1, Ordering::SeqCst);
        if prior == 0 {
            return Err(JevError::HttpStatus {
                status: 429,
                detail: "rate limited (test)".to_string(),
                retry_after: Some(Duration::from_millis(50)),
            });
        }
        Ok(pi_jev::mock::valid_response_for(request))
    });
    let fixture = make_observer(
        transport,
        temp_records(),
        JevMode::Compare,
        SchedulerConfig {
            request_deadline: Duration::from_secs(5),
            min_interval: Duration::ZERO,
            ..SchedulerConfig::default()
        },
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-ra"));
    let records = wait_for_records(fixture.records_path(), |record| {
        record.selected_value.is_some()
    })
    .await;
    assert!(
        records.iter().any(|record| record.selected_value.is_some()),
        "retry-after path must complete"
    );
    // Retries are owned by the client; the scheduler owns queueing only.
    assert!(
        fixture.stats.snapshot().retries >= 1,
        "retry recorded: {:?}",
        fixture.stats.snapshot()
    );
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_cleans_up_pending_work() {
    let blocked = MockJevTransport::scripted(vec![MockStep::SlowResponse {
        delay_ms: 30_000,
    }]);
    let fixture = make_observer(
        blocked,
        temp_records(),
        JevMode::Compare,
        SchedulerConfig {
            queue_capacity: 8,
            request_deadline: Duration::from_secs(30),
            min_interval: Duration::ZERO,
            ..SchedulerConfig::default()
        },
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-cancel"));
    tokio::time::sleep(Duration::from_millis(50)).await;
    // Dispose the session: in-flight and queued work must be dropped.
    fixture
        .observer
        .observe("session_shutdown", &json!({ "session_id": "sess-cancel" }));
    let records = wait_for_records(fixture.records_path(), |record| {
        record.skipped_reason.as_deref() == Some("cancelled")
    })
    .await;
    assert!(
        !records
            .iter()
            .any(|record| record.selected_value.is_some()),
        "cancelled requests must not complete"
    );
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_marker_does_not_blackhole_later_requests() {
    let blocked = MockJevTransport::scripted(vec![MockStep::SlowResponse {
        delay_ms: 30_000,
    }]);
    let fixture = make_observer(
        blocked,
        temp_records(),
        JevMode::Compare,
        SchedulerConfig {
            queue_capacity: 8,
            request_deadline: Duration::from_secs(30),
            min_interval: Duration::ZERO,
            ..SchedulerConfig::default()
        },
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-rearm"));
    tokio::time::sleep(Duration::from_millis(50)).await;
    // Compare -> Off: in-flight work cancelled, marker recorded.
    fixture
        .observer
        .observe("session_shutdown", &json!({ "session_id": "sess-rearm" }));
    let _ = wait_for_records(fixture.records_path(), |record| {
        record.skipped_reason.as_deref() == Some("cancelled")
    })
    .await;
    // A transport that can complete now: Compare -> Off -> Compare must run,
    // not black-hole on the stale marker.
    let fixture = make_observer(
        MockJevTransport::all_valid(),
        fixture.records_path().to_path_buf(),
        JevMode::Compare,
        SchedulerConfig {
            queue_capacity: 8,
            request_deadline: Duration::from_secs(5),
            min_interval: Duration::ZERO,
            ..SchedulerConfig::default()
        },
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-rearm"));
    // If the stale marker still black-holed the session, this wait times out:
    // a completed record proves the post-cancel request ran.
    let records = wait_for_records(fixture.records_path(), |record| {
        record.session_id == "sess-rearm" && record.selected_value.is_some()
    })
    .await;
    assert!(
        records
            .iter()
            .any(|record| record.session_id == "sess-rearm"
                && record.selected_value.is_some()
                && !record.applied),
        "post-cancel request must complete, not be dropped as cancelled"
    );
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn off_mode_never_schedules_anything() {
    let fixture = make_observer(
        MockJevTransport::all_valid(),
        temp_records(),
        JevMode::Off,
        fast_scheduler(),
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-off"));
    fixture.observer.observe(
        "agent_end",
        &json!({
            "session_id": "sess-off",
            "turn": 1,
            "state": { "result_excerpt": "r" },
        }),
    );
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert_eq!(fixture.mock.call_count(), 0, "Off mode must make zero SystemOne calls");
    let records = read_records(fixture.records_path(), 1 << 20);
    assert!(records.is_empty(), "Off mode must produce no records: {records:?}");
    let metrics = fixture.observer.scheduler_metrics();
    assert_eq!(metrics["enqueued"].as_u64().unwrap_or(1), 0, "no scheduler work in Off");
    fixture.observer.shutdown();
}

/// Active is an operative mode, and the mode resolution rules are unchanged:
/// an explicit per-session mode wins, and a mode is never silently rewritten.
/// What Active does NOT grant is unchanged too: this crate holds no host
/// authority, applies nothing itself, and the client-side refusals for every
/// control capability still hold.
#[test]
fn active_mode_is_operative_and_never_silently_rewritten() {
    use pi_jev::config::{resolve_effective_mode, JevSettings};

    // The mode is represented exactly as set: never silently turned into
    // Compare, and never inferred from a credential.
    assert_eq!(
        resolve_effective_mode(Some(JevMode::Active), None),
        JevMode::Active,
        "the reserved mode is represented, never silently turned into Compare"
    );
    assert_eq!(
        resolve_effective_mode(Some(JevMode::Compare), Some(JevMode::Active)),
        JevMode::Compare,
        "an explicit Compare wins over an Active default"
    );

    let settings = JevSettings {
        global_default: Some(JevMode::Active),
        sessions: std::collections::BTreeMap::from([(
            "sess-a".to_string(),
            pi_jev::config::PersistedSessionMode {
                mode: Some(JevMode::Active),
                inherited_from: None,
            },
        )]),
        ..JevSettings::default()
    };
    assert_eq!(settings.effective_mode("sess-a"), JevMode::Active);
    assert!(
        settings.wants_observer(),
        "an Active configuration needs the observer and its client"
    );

    // The settings file may say active; loading reports exactly that.
    let dir = tempfile::Builder::new()
        .prefix("jev-active-")
        .tempdir()
        .unwrap();
    std::fs::create_dir_all(dir.path().join("jev")).unwrap();
    std::fs::write(
        dir.path().join("jev").join("jev-settings.json"),
        r#"{"schema_version": 1, "global_default": "active"}"#,
    )
    .unwrap();
    let loaded = pi_jev::config::JevSettingsStore::new(dir.path()).load();
    assert_eq!(loaded.effective_mode("sess-any"), JevMode::Active);
    // Active is operative: it needs the observer and its client.
    assert!(loaded.wants_observer());

    // Active builds a real client. The crate still applies nothing itself:
    // an accepted answer is returned to the caller, which owns the effect.
    let client = JevSystemOne::new(
        JevMode::Active,
        SecretString::new(SYNTHETIC_KEY),
        Arc::new(MockJevTransport::all_valid()),
        JevLimits::default(),
        Arc::new(JevStats::default()),
    )
    .expect("Active must construct");
    assert_eq!(client.effective_mode(), JevMode::Active);
}

/// Delayed/stale SystemOne answers can only ever become records
/// (applied=false); they never re-enter or re-drive the agent loop. After
/// shutdown, cancellation prevents any accepted answer; terminal cancellation
/// records remain visible so every attempted observation has an honest outcome.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delayed_stale_results_cannot_act() {
    let delayed = MockJevTransport::scripted(vec![MockStep::SlowResponse {
        delay_ms: 350,
    }]);
    let fixture = make_observer(delayed, temp_records(), JevMode::Compare, fast_scheduler());
    // The turn boundary passes: nothing acts, nothing is recorded yet.
    fixture.observer.observe("turn_start", &turn_payload("sess-stale"));
    let early = read_records(fixture.records_path(), 1 << 20);
    assert!(
        early.is_empty(),
        "a delayed answer must not surface before it arrives: {early:?}"
    );
    // The result lands asynchronously as a record only.
    let records = wait_for_records(fixture.records_path(), |record| {
        record.selected_value.is_some()
    })
    .await;
    assert!(
        records
            .iter()
            .all(|record| !record.applied && record.mode == "compare"),
        "stale results are records only: {records:?}"
    );
    assert_eq!(
        fixture.observer.scheduler_metrics()["completed"].as_u64().unwrap_or(0),
        1,
    );
    let settled_count = records.len();
    let accepted_count = records.iter().filter(|record| record.selected_value.is_some()).count();
    let calls_before_shutdown = fixture.mock.call_count();
    // Disposal cancels remaining work; nothing further can act afterwards.
    fixture.observer.observe("session_shutdown", &json!({ "session_id": "sess-stale" }));
    fixture.observer.shutdown();
    fixture.observer.observe("agent_end", &turn_payload("sess-stale"));
    tokio::time::sleep(Duration::from_millis(80)).await;
    let after = read_records(fixture.records_path(), 1 << 20);
    assert_eq!(after.iter().filter(|record| record.selected_value.is_some()).count(),
        accepted_count, "post-shutdown work must not accept a late answer: {after:?}");
    assert_eq!(fixture.mock.call_count(), calls_before_shutdown,
        "post-shutdown observes must not call the transport");
    assert!(after[settled_count..].iter().all(|record| !record.applied
        && record.selected_value.is_none()
        && record.skipped_reason.as_deref() == Some("cancelled")),
        "only terminal cancellation records are allowed: {after:?}");
}

// ---------------------------------------------------------------------------
// 6. Malicious payloads only land as records
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malicious_answers_fail_to_logged_skips() {
    let malicious = FnTransport(|request: &SystemOneRequest, _: Duration| {
        let mut response = pi_jev::mock::valid_response_for(request);
        for answer in response.answers.values_mut() {
            if let Answer::Choice {
                choice,
                probabilities,
                confidence,
            } = answer
            {
                *choice = "<script>alert(1)</script>".to_string();
                *confidence = 1.0;
                probabilities.clear();
                probabilities.insert("coding".to_string(), f64::NAN);
            }
        }
        Ok(response)
    });
    let fixture = make_observer(
        malicious,
        temp_records(),
        JevMode::Compare,
        fast_scheduler(),
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-mal"));
    let records = wait_for_records(fixture.records_path(), |record| {
        record.skipped_reason.as_deref() == Some("choice_not_in_criteria")
    })
    .await;
    // Per-question validation: the tampered choice answers are logged skips
    // with the precise reason; nothing malicious ever lands in a record.
    assert!(
        !records
            .iter()
            .any(|record| record
                .selected_value
                .as_deref()
                .map(|value| value.contains("<script>"))
                .unwrap_or(false)),
        "malicious answers must never produce fabricated decisions: {records:?}"
    );
    assert!(records.iter().all(|record| !record.applied));
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_and_extra_answer_ids_are_skipped() {
    let partial = FnTransport(|request: &SystemOneRequest, _: Duration| {
        let mut response = pi_jev::mock::valid_response_for(request);
        if let Some(last_key) = response.answers.keys().next_back().cloned() {
            response.answers.remove(&last_key);
        }
        response.answers.insert(
            "unknown_question_id".to_string(),
            Answer::Choice {
                choice: "coding".to_string(),
                probabilities: BTreeMap::from([("coding".to_string(), 1.0)]),
                confidence: 0.9,
            },
        );
        Ok(response)
    });
    let fixture = make_observer(
        partial,
        temp_records(),
        JevMode::Compare,
        fast_scheduler(),
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-ids"));
    let records = wait_for_records(fixture.records_path(), |record| {
        record.skipped_reason.is_some()
    })
    .await;
    // Per-question validation (validate_response): the unknown id and the
    // missing id are logged as precise per-question skips, the valid answers
    // are still recorded. No fabrication, no whole-request loss.
    assert!(
        records
            .iter()
            .any(|record| record.skipped_reason.as_deref() == Some("missing_answer_id")),
        "missing id must become a per-question skip: {records:?}"
    );
    assert!(
        records
            .iter()
            .any(|record| record.skipped_reason.as_deref() == Some("answer_missing")
                || record.selected_value.is_some()),
        "valid answers are still recorded: {records:?}"
    );
    assert!(
        !records
            .iter()
            .any(|record| record.skipped_reason.as_deref().map(|s| s.contains("request_failed")).unwrap_or(false)),
        "an unknown id must not fail the whole request: {records:?}"
    );
    fixture.observer.shutdown();
}

// ---------------------------------------------------------------------------
// 7. Correlation records
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn correlation_records_have_baselines_and_applied_false() {
    let fixture = make_observer(
        MockJevTransport::all_valid(),
        temp_records(),
        JevMode::Compare,
        fast_scheduler(),
    );
    fixture.observer.observe(
        "tool_call",
        &json!({
            "session_id": "sess-corr",
            "turn": 2,
            "tool_name": "edit",
            "state": { "tool_name": "edit" },
        }),
    );
    let records = wait_for_records(fixture.records_path(), |record| {
        record.question_id == "tool_candidates.0"
    })
    .await;
    assert!(!records.is_empty());
    for record in &records {
        assert!(!record.applied, "applied=false ALWAYS");
        assert_eq!(record.mode, "compare");
        assert_eq!(record.session_id, "sess-corr");
        assert_eq!(record.turn, 2);
        assert_eq!(record.stage, "tool_call");
        assert_eq!(record.state_schema_version, STATE_SCHEMA_VERSION);
        assert_eq!(record.prompt_version, PROMPT_VERSION);
    }
    let question = records
        .iter()
        .find(|record| record.question_id == "tool_candidates.0")
        .expect("tool question record");
    assert_eq!(question.baseline_actual_choice.as_deref(), Some("edit"));
    // A noul question has no comparable baseline: noncomparable, not zero.
    assert_eq!(question.agreement.as_deref(), Some("noncomparable"));
    // No raw prompt or tool output text may appear anywhere in the file.
    let raw = std::fs::read_to_string(fixture.records_path()).unwrap();
    assert!(
        !raw.contains("user_text_excerpt"),
        "records must not carry prompt field dumps"
    );
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn agreement_classification_agrees_and_disagrees() {
    let fixture = make_observer(
        MockJevTransport::all_valid(),
        temp_records(),
        JevMode::Compare,
        fast_scheduler(),
    );
    fixture.observer.observe(
        "agent_end",
        &json!({
            "session_id": "sess-agree",
            "turn": 1,
            "state": { "result_excerpt": "done" },
        }),
    );
    let records = wait_for_records(fixture.records_path(), |record| {
        record.agreement.as_deref() == Some("disagree")
    })
    .await;
    let disagree = records
        .iter()
        .find(|record| record.agreement.as_deref() == Some("disagree"))
        .expect("baseline stop vs recommended continue must disagree");
    assert_eq!(disagree.baseline_actual_choice.as_deref(), Some("stop"));
    assert_eq!(disagree.category, "continue_stop_escalate");
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_and_terminal_written_exactly_once() {
    let fixture = make_observer(
        MockJevTransport::all_valid(),
        temp_records(),
        JevMode::Compare,
        fast_scheduler(),
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-once"));
    let _ = wait_for_records(fixture.records_path(), |record| {
        record.selected_value.is_some()
    })
    .await;
    tokio::time::sleep(Duration::from_millis(60)).await;
    let records = read_records(fixture.records_path(), 1 << 20);
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for record in &records {
        assert!(
            seen.insert((record.request_id.clone(), record.question_id.clone())),
            "each (request, question) record must be written exactly once: {}",
            record.question_id
        );
        if record.selected_value.is_some() {
            assert!(record.request_start_ts.is_some());
            assert!(record.terminal_ts.is_some());
        }
    }
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retention_cleanup_is_confined_to_jev_files() {
    let dir = tempfile::tempdir().unwrap();
    let records_path = dir.path().join("jev").join("records.jsonl");
    // A foreign (non-Jev) monitoring log next door must never be touched.
    let foreign_dir = dir.path().join("other-monitoring");
    std::fs::create_dir_all(&foreign_dir).unwrap();
    let foreign = foreign_dir.join("keep-me.log");
    std::fs::write(&foreign, "keep me\n").unwrap();
    let fixture = make_observer(
        MockJevTransport::all_valid(),
        records_path,
        JevMode::Compare,
        fast_scheduler(),
    );
    fixture.observer.observe("turn_start", &turn_payload("sess-ret"));
    tokio::time::sleep(Duration::from_millis(120)).await;
    let jev_dir = fixture.records_path().parent().unwrap().to_path_buf();
    let entries: Vec<String> = std::fs::read_dir(&jev_dir)
        .unwrap()
        .filter_map(|entry| {
            entry
                .ok()
                .map(|e| e.file_name().to_string_lossy().to_string())
        })
        .collect();
    assert!(
        entries.iter().all(|name| name.starts_with("records.jsonl")),
        "only jev-owned record files may exist under the jev dir: {entries:?}"
    );
    assert_eq!(std::fs::read_to_string(&foreign).unwrap(), "keep me\n");
    fixture.observer.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hypothetical_savings_never_reported_as_measured() {
    let fixture = make_observer(
        MockJevTransport::all_valid(),
        temp_records(),
        JevMode::Compare,
        fast_scheduler(),
    );
    fixture.observer.observe(
        "agent_end",
        &json!({
            "session_id": "sess-sav",
            "turn": 1,
            "state": { "result_excerpt": "r" },
        }),
    );
    let records = wait_for_records(fixture.records_path(), |record| {
        record.selected_value.is_some()
    })
    .await;
    for record in &records {
        if record.selected_value.is_some() {
            let acceptance = record.hypothetical_acceptance.clone().unwrap_or_default();
            assert!(
                acceptance == "accepted" || acceptance == "fallback",
                "acceptance is hypothetical, not measured: {acceptance}"
            );
        }
    }
    let report = pi_jev::report::build_report_from(&records, "2026-09-19T00:00:00Z");
    assert_eq!(report["actual_llm_calls_avoided"], json!(0));
    assert!(report["savings_note"].as_str().unwrap().contains("hypothetical"));
    fixture.observer.shutdown();
}

#[test]
fn sanitize_strips_urls_and_key_shaped_tokens() {
    let dirty = "GET https://api.typesafe.ai/v1/systemone?api_key=abc123 Bearer sk-123 failed";
    let clean = pi_jev::error::sanitize_detail(&dirty);
    assert!(!clean.contains("https://"));
    assert!(!clean.contains("abc123"));
    assert!(!clean.contains("sk-123"));
    assert!(clean.contains(pi_jev::error::REDACTED));
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

struct TempRecords {
    /// Held for the whole test process so the records path stays valid.
    #[allow(dead_code)]
    dir: tempfile::TempDir,
}

thread_local! {
    static TEMP_RECORDS: std::cell::RefCell<Vec<TempRecords>> = std::cell::RefCell::new(Vec::new());
}

/// Create a bounded per-test temp records path; the TempDir is kept alive for
/// the whole test process (fine for these short tests).
fn temp_records() -> std::path::PathBuf {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("jev").join("records.jsonl");
    std::fs::create_dir_all(dir.path().join("jev")).unwrap();
    TEMP_RECORDS.with(|slots| slots.borrow_mut().push(TempRecords { dir }));
    path
}

impl<T: Transport + 'static> ObserverFixture<T> {
    fn records_path(&self) -> &std::path::Path {
        &self.records_path
    }
}
