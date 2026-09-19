//! Shared decision/compare lifecycle tests. All transports are local mocks.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_jev::active::{ActivationPolicy, AppliedEffect};
use pi_jev::config::JevMode;
use pi_jev::correlate::{read_records, ACTIVE_RECORD_SCHEMA_VERSION, RECORD_SCHEMA_VERSION};
use pi_jev::evaluators::PreparedQuestion;
use pi_jev::hooks::{ActiveSettings, JevObserver, JevObserverConfig};
use pi_jev::mock::{valid_response_for, MockJevTransport};
use pi_jev::{
    BoxFuture, JevError, JevLimits, JevStats, JevSystemOne, SecretString, SystemOneRequest,
    SystemOneResponse, Transport,
};
use serde_json::{json, Value};

fn questions() -> Vec<PreparedQuestion> {
    vec![PreparedQuestion {
        question_id: "tool_requirement.0".to_string(),
        spec: pi_jev::mock::choice_question("Need tools?", &[("none", None), ("read", None)]),
    }]
}

fn payload() -> Value {
    json!({"session_id":"combined-test", "turn":3, "state":{"task":"bounded synthetic task"},
        "baseline_action":{"tools":"count:2", "reasoning_effort":"high"}, "compaction_enabled":false})
}

struct Fixture {
    observer: Arc<JevObserver>,
    mode: Arc<Mutex<JevMode>>,
    generation: Arc<Mutex<String>>,
    stats: Arc<JevStats>,
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new(mode: JevMode, transport: Arc<dyn Transport>, deadline: Duration) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mode = Arc::new(Mutex::new(mode));
        let gate = mode.clone();
        let generation = Arc::new(Mutex::new(String::new()));
        let generation_gate = generation.clone();
        let stats = Arc::new(JevStats::default());
        let client = JevSystemOne::new(
            JevMode::Compare,
            SecretString::new("synthetic-key"),
            transport,
            JevLimits {
                max_retries: 0,
                ..Default::default()
            },
            stats.clone(),
        )
        .unwrap();
        let observer = JevObserver::new(
            JevObserverConfig {
                mode_gate: Arc::new(move |_| *gate.lock().unwrap()),
                independent_gate: Arc::new(|_| true),
                policy_generation: Arc::new(move |_, independent| {
                    if independent {
                        "compaction".to_string()
                    } else {
                        generation_gate.lock().unwrap().clone()
                    }
                }),
                active: ActiveSettings {
                    deadline,
                    ..Default::default()
                },
                ..Default::default()
            },
            Arc::new(client),
            dir.path().join("records.jsonl"),
        );
        Self {
            observer,
            mode,
            generation,
            stats,
            dir,
        }
    }

    fn records(&self) -> Vec<pi_jev::correlate::CorrelationRecord> {
        read_records(&self.dir.path().join("records.jsonl"), 1024 * 1024)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.observer.shutdown();
    }
}

#[tokio::test]
async fn combined_shares_one_call_and_retains_premutation_baseline() {
    let transport = Arc::new(MockJevTransport::all_valid());
    let fixture = Fixture::new(
        JevMode::CompareAndActive,
        transport.clone(),
        Duration::from_secs(1),
    );
    let mut input = payload();
    let outcome = fixture
        .observer
        .decide_prepared(
            &input,
            "provider_request",
            questions(),
            &ActivationPolicy::default(),
        )
        .await;
    input["baseline_action"]["tools"] = json!("count:0");
    let effects = BTreeMap::from([(
        "tool_requirement".to_string(),
        vec![AppliedEffect::new(
            "tools",
            Some("count:2".to_string()),
            None,
        )],
    )]);
    fixture.observer.record_active(&outcome, &effects);
    assert_eq!(transport.call_count(), 1);
    let rows = fixture.records();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].schema_version, RECORD_SCHEMA_VERSION);
    assert_eq!(rows[1].schema_version, ACTIVE_RECORD_SCHEMA_VERSION);
    assert_eq!(rows[0].request_id, rows[1].request_id);
    assert_eq!(rows[0].mode, "compare-active");
    assert!(!rows[0].applied);
    assert!(rows[1].applied);
    assert_eq!(rows[1].baseline_action["tools"], "count:2");
    assert_eq!(rows[0].baseline_action["tools"], "count:2");
    assert_eq!(rows[1].actual_action["tools"], "absent");
    assert_eq!(rows[1].attempt, 1);
    for row in &rows {
        assert_eq!(row.observed_metrics["jev_input_tokens"], 312);
        assert_eq!(row.observed_metrics["jev_output_tokens"], 48);
    }
    assert!(rows[1].request_start_ts.is_some());
    assert_eq!(rows[1].agreement.as_deref(), Some("noncomparable"));
    let report = pi_jev::report::build_report_from(&rows, "synthetic-time");
    assert_eq!(report["logical_decisions"], json!(1));
    assert_eq!(report["timed_requests"], json!(1));
    assert_eq!(report["active"]["applied"], json!(1));
    let mut audit = rows[0].clone();
    audit.schema_version = "jev.compaction/1".to_string();
    audit.request_id = "compaction-request".to_string();
    audit.question_id = "compaction.audit".to_string();
    audit.category = "compaction".to_string();
    audit.selected_value = None;
    audit.applied = true;
    audit.compaction_enabled = Some(true);
    audit.outcome = Some("applied".to_string());
    let mixed = vec![
        rows[0].clone(),
        rows[1].clone(),
        rows[1].clone(),
        audit.clone(),
        audit,
    ];
    let mixed_report = pi_jev::report::build_report_from(&mixed, "synthetic-time");
    assert_eq!(mixed_report["logical_decisions"], json!(1));
    assert_eq!(mixed_report["active"]["applied"], json!(1));
    assert_eq!(mixed_report["compaction"]["requests"], json!(1));
    assert_eq!(mixed_report["compaction"]["applied"], json!(1));
    assert_eq!(mixed_report["compaction"]["fallback"], json!(0));
    assert_eq!(mixed_report["timed_requests"], json!(2));
    assert!(mixed_report["category_coverage"]
        .get("compaction")
        .is_none());
    assert!(mixed_report["actual_llm_calls_avoided"].is_null());
}

#[tokio::test]
async fn accepted_noop_is_not_applied_and_refused_answer_is_visible() {
    let fixture = Fixture::new(
        JevMode::Active,
        Arc::new(MockJevTransport::all_valid()),
        Duration::from_secs(1),
    );
    let first = fixture
        .observer
        .decide_prepared(
            &payload(),
            "provider_request",
            questions(),
            &ActivationPolicy::default(),
        )
        .await;
    fixture.observer.record_active(&first, &BTreeMap::new());
    let policy = ActivationPolicy {
        enabled_categories: BTreeSet::new(),
        ..Default::default()
    };
    let second = fixture
        .observer
        .decide_prepared(&payload(), "provider_request", questions(), &policy)
        .await;
    fixture.observer.record_active(&second, &BTreeMap::new());
    let rows = fixture.records();
    assert_eq!(rows.len(), 2);
    assert!(!rows[0].applied);
    assert_eq!(rows[0].acceptance.as_deref(), Some("accepted"));
    assert_eq!(rows[0].outcome.as_deref(), Some("accepted_no_effect"));
    assert!(rows[1].selected_value.is_some());
    assert!(rows[1].confidence.is_some());
    assert_eq!(
        rows[1].fallback_reason.as_deref(),
        Some("category_disabled")
    );
}

struct GatedTransport {
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Semaphore>,
}
impl Transport for GatedTransport {
    fn post(
        &self,
        request: &SystemOneRequest,
        _: Duration,
    ) -> BoxFuture<Result<SystemOneResponse, JevError>> {
        let entered = self.entered.clone();
        let release = self.release.clone();
        let response = valid_response_for(request);
        Box::pin(async move {
            entered.notify_one();
            let _permit = release.acquire().await.unwrap();
            Ok(response)
        })
    }
}

fn gated_fixture(
    mode: JevMode,
) -> (
    Fixture,
    Arc<tokio::sync::Notify>,
    Arc<tokio::sync::Semaphore>,
) {
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let fixture = Fixture::new(
        mode,
        Arc::new(GatedTransport {
            entered: entered.clone(),
            release: release.clone(),
        }),
        Duration::from_secs(1),
    );
    (fixture, entered, release)
}

#[tokio::test]
async fn mode_policy_and_disposal_changes_refuse_late_answers() {
    for change in 0..3 {
        let (fixture, entered, release) = gated_fixture(JevMode::CompareAndActive);
        let observer = fixture.observer.clone();
        let pending = tokio::spawn(async move {
            observer
                .decide_prepared(
                    &payload(),
                    "provider_request",
                    questions(),
                    &ActivationPolicy::default(),
                )
                .await
        });
        entered.notified().await;
        match change {
            0 => *fixture.mode.lock().unwrap() = JevMode::Off,
            1 => *fixture.generation.lock().unwrap() = "rotated-policy".to_string(),
            _ => fixture.observer.cancel_session("combined-test"),
        }
        release.add_permits(1);
        let outcome = tokio::time::timeout(Duration::from_secs(1), pending)
            .await
            .unwrap()
            .unwrap();
        assert!(outcome.decisions.is_empty());
        assert!(outcome.unavailable.is_some());
        assert!(!fixture.observer.can_apply(&outcome));
        assert_eq!(fixture.stats.snapshot().in_flight, 0);
        fixture.observer.record_active(&outcome, &BTreeMap::new());
        assert!(fixture.records().iter().all(|row| !row.applied));
    }
}

#[tokio::test]
async fn completed_decision_is_invalidated_before_application() {
    let fixture = Fixture::new(
        JevMode::Active,
        Arc::new(MockJevTransport::all_valid()),
        Duration::from_secs(1),
    );
    let outcome = fixture
        .observer
        .decide_prepared(
            &payload(),
            "provider_request",
            questions(),
            &ActivationPolicy::default(),
        )
        .await;
    assert!(fixture.observer.can_apply(&outcome));
    fixture.observer.cancel_session("combined-test");
    assert!(!fixture.observer.can_apply(&outcome));
}

#[tokio::test]
async fn independent_compaction_runs_with_decisions_off_and_survives_decision_cancel() {
    let (fixture, entered, release) = gated_fixture(JevMode::Off);
    let disabled = fixture
        .observer
        .decide_prepared(
            &payload(),
            "provider_request",
            questions(),
            &ActivationPolicy::default(),
        )
        .await;
    assert!(disabled.decisions.is_empty());
    assert_eq!(fixture.stats.snapshot().attempts, 0);
    let observer = fixture.observer.clone();
    let pending = tokio::spawn(async move {
        observer
            .decide_independent(
                &payload(),
                "compaction",
                vec![PreparedQuestion {
                    question_id: "compaction.call_t0".to_string(),
                    spec: pi_jev::mock::noul_question("Keep?", "yes", "no"),
                }],
            )
            .await
    });
    entered.notified().await;
    fixture.observer.cancel_decisions("combined-test");
    release.add_permits(1);
    let outcome = pending.await.unwrap();
    assert!(outcome.raw.is_some());
    assert!(fixture.observer.can_apply(&outcome));
    assert_eq!(fixture.stats.snapshot().attempts, 1);
    fixture.observer.cancel_session("combined-test");
    assert!(!fixture.observer.can_apply(&outcome));
}

#[tokio::test]
async fn caller_abort_and_deadline_release_transport_accounting() {
    let (fixture, entered, _release) = gated_fixture(JevMode::Active);
    let observer = fixture.observer.clone();
    let pending = tokio::spawn(async move {
        observer
            .decide_prepared(
                &payload(),
                "provider_request",
                questions(),
                &ActivationPolicy::default(),
            )
            .await
    });
    entered.notified().await;
    pending.abort();
    let _ = pending.await;
    assert_eq!(fixture.stats.snapshot().in_flight, 0);
    let transport = Arc::new(GatedTransport {
        entered: Arc::new(tokio::sync::Notify::new()),
        release: Arc::new(tokio::sync::Semaphore::new(0)),
    });
    let timeout = Fixture::new(JevMode::Active, transport, Duration::from_millis(5));
    for _ in 0..3 {
        let outcome = timeout
            .observer
            .decide_prepared(
                &payload(),
                "provider_request",
                questions(),
                &ActivationPolicy::default(),
            )
            .await;
        assert_eq!(outcome.terminal_reason.as_deref(), Some("timeout"));
    }
    let blocked = timeout
        .observer
        .decide_prepared(
            &payload(),
            "provider_request",
            questions(),
            &ActivationPolicy::default(),
        )
        .await;
    assert_eq!(blocked.terminal_reason.as_deref(), Some("circuit_open"));
    assert_eq!(timeout.stats.snapshot().attempts, 3);
    assert_eq!(timeout.stats.snapshot().in_flight, 0);
}

struct ContradictoryTransport;
impl Transport for ContradictoryTransport {
    fn post(
        &self,
        request: &SystemOneRequest,
        _: Duration,
    ) -> BoxFuture<Result<SystemOneResponse, JevError>> {
        let mut response = valid_response_for(request);
        for answer in response.answers.values_mut() {
            *answer = pi_jev::Answer::Choice {
                choice: "drop".to_string(),
                probabilities: BTreeMap::from([
                    ("keep".to_string(), 1.0),
                    ("drop".to_string(), 0.0),
                ]),
                confidence: 1.0,
            };
        }
        Box::pin(async move { Ok(response) })
    }
}

#[tokio::test]
async fn optional_drop_requires_selected_label_probability() {
    let fixture = Fixture::new(
        JevMode::CompareAndActive,
        Arc::new(ContradictoryTransport),
        Duration::from_secs(1),
    );
    let policy = ActivationPolicy {
        enabled_categories: [pi_jev::DecisionCategory::MemoryRelevance]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let prepared = vec![PreparedQuestion {
        question_id: "memory_relevance.0".to_string(),
        spec: pi_jev::mock::choice_question("Keep?", &[("keep", None), ("drop", None)]),
    }];
    let outcome = fixture
        .observer
        .decide_prepared(&payload(), "retrieval", prepared, &policy)
        .await;
    assert!(outcome.decisions.is_empty());
    fixture.observer.record_active(&outcome, &BTreeMap::new());
    let rows = fixture.records();
    assert_eq!(rows[1].selected_value.as_deref(), Some("drop"));
    assert_eq!(rows[1].confidence, Some(1.0));
    assert_eq!(rows[1].fallback_reason.as_deref(), Some("low_confidence"));
}

#[tokio::test]
async fn captured_policy_stamp_refuses_a_plan_built_before_feature_change() {
    let transport = Arc::new(MockJevTransport::all_valid());
    let fixture = Fixture::new(JevMode::Active, transport.clone(), Duration::from_secs(1));
    let mut input = payload();
    input["policy_generation"] = json!("old-policy");
    *fixture.generation.lock().unwrap() = "new-policy".to_string();
    let outcome = fixture
        .observer
        .decide_prepared(
            &input,
            "provider_request",
            questions(),
            &ActivationPolicy::default(),
        )
        .await;
    assert!(outcome.decisions.is_empty());
    assert_eq!(transport.call_count(), 0);
}

#[tokio::test]
async fn completed_answer_is_recorded_as_fallback_after_policy_change() {
    let fixture = Fixture::new(
        JevMode::Active,
        Arc::new(MockJevTransport::all_valid()),
        Duration::from_secs(1),
    );
    let outcome = fixture
        .observer
        .decide_prepared(
            &payload(),
            "provider_request",
            questions(),
            &ActivationPolicy::default(),
        )
        .await;
    *fixture.generation.lock().unwrap() = "changed".to_string();
    fixture.observer.record_active(
        &outcome,
        &BTreeMap::from([(
            "tool_requirement".to_string(),
            vec![AppliedEffect::new("tools", None, None)],
        )]),
    );
    let rows = fixture.records();
    assert_eq!(rows[0].acceptance.as_deref(), Some("fallback"));
    assert_eq!(
        rows[0].fallback_reason.as_deref(),
        Some("cancelled_or_policy_changed")
    );
    assert!(!rows[0].applied);
}
