//! ROOT-CONTRACT v9: all-lane requested-model capture tests. Every lane seam
//! (compare queue, bundled active decide, explicit decides for guidance /
//! control / search / evidence, and independent compaction) must stamp the
//! requested Jev model captured from the SAME gate snapshot as the mode,
//! BEFORE any await; queued/held work keeps its captured wire identity and is
//! rejected if the authoritative model/generation moves (never late-restamped).
//! Mock transport only; no network.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_jev::client::{JevLimits, JevStats, JevSystemOne};
use pi_jev::config::JevMode;
use pi_jev::credential::SecretString;
use pi_jev::evaluators::PreparedQuestion;
use pi_jev::hooks::{
    ActiveSettings, JevObserver, JevObserverConfig, JevRequestGate, SYSTEM_ONE_MODEL,
};
use pi_jev::mock::{choice_question, noul_question, MockJevTransport, MockStep};
use pi_jev::scheduler::SchedulerConfig;
use pi_jev::types::{DecisionCategory, Transport};
use serde_json::{json, Value};

const SYNTHETIC_KEY: &str = "jev-test-key-not-real";

/// Keeps per-test record tempdirs alive for the whole test (the established
/// comparison-tests pattern: a thread-local slot holder).
thread_local! {
    static TEMP_RECORDS: std::cell::RefCell<Vec<tempfile::TempDir>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn temp_records() -> std::path::PathBuf {
    let dir = tempfile::Builder::new().prefix("jev-model-lane-").tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("jev")).unwrap();
    let path = dir.path().join("jev").join("records.jsonl");
    TEMP_RECORDS.with(|slots| slots.borrow_mut().push(dir));
    path
}


/// Shared gate truth: `(mode, requested_model)`, flippable by a test.
type SharedGate = Arc<Mutex<(JevMode, String)>>;

fn explicit_questions() -> Vec<PreparedQuestion> {
    vec![
        PreparedQuestion {
            question_id: "task_classification.0".to_string(),
            spec: choice_question("Which task type is this?", &[("coding", Some("code change"))]),
        },
        PreparedQuestion {
            question_id: "result_sufficiency.0".to_string(),
            spec: noul_question("Is the result sufficient?", "sufficient", "insufficient"),
        },
    ]
}

fn payload(session_id: &str) -> Value {
    json!({
        "session_id": session_id,
        "turn": 3,
        "state": {"user_text_excerpt": "fix the parser test", "message_count": 4},
        "policy_generation": "gen-test",
    })
}

struct Fixture {
    observer: Arc<JevObserver>,
    mock: Arc<MockJevTransport>,
    gate: SharedGate,
}

fn make_fixture(steps: Vec<MockStep>, gate: SharedGate) -> Fixture {
    make_fixture_with_scheduler(steps, gate, SchedulerConfig::default())
}

fn make_fixture_with_scheduler(
    steps: Vec<MockStep>,
    gate: SharedGate,
    scheduler: SchedulerConfig,
) -> Fixture {
    let mock = Arc::new(MockJevTransport::scripted(steps));
    let transport: Arc<dyn Transport> = mock.clone();
    let stats = Arc::new(JevStats::default());
    let system_one: Arc<dyn pi_jev::types::SystemOne> = Arc::new(
        JevSystemOne::new(
            JevMode::CompareAndActive,
            SecretString::new(SYNTHETIC_KEY),
            transport,
            JevLimits::default(),
            stats,
        )
        .unwrap(),
    );
    let scheduler_gate = gate.clone();
    let config = JevObserverConfig {
        mode_gate: Arc::new(move |_| scheduler_gate.lock().unwrap().clone()),
        independent_gate: Arc::new(|_| true),
        policy_generation: Arc::new(|_, _| "gen-test".to_string()),
        scheduler,
        active: ActiveSettings {
            deadline: std::time::Duration::from_secs(5),
            ..ActiveSettings::default()
        },
        ..JevObserverConfig::default()
    };
    let observer = JevObserver::new(config, system_one, temp_records());
    Fixture { observer, mock, gate }
}

fn active_policy() -> pi_jev::active::ActivationPolicy {
    pi_jev::active::ActivationPolicy {
        enabled_categories: [DecisionCategory::TaskClassification].into_iter().collect(),
        ..pi_jev::active::ActivationPolicy::default()
    }
}

#[tokio::test]
async fn l1_active_decide_carries_the_requested_model_captured_with_the_gate() {
    let gate: SharedGate = Arc::new(Mutex::new((JevMode::CompareAndActive, "jev-1.13.0".to_string())));
    let fixture = make_fixture(vec![MockStep::Valid], gate);
    let outcome = fixture
        .observer
        .decide_prepared(&payload("sess-l1"), "model_select", explicit_questions(), &active_policy())
        .await;
    assert!(outcome.dispatched, "the decide must dispatch");
    assert_eq!(fixture.mock.call_count(), 1);
    assert_eq!(
        fixture.mock.calls()[0].model,
        "jev-1.13.0",
        "the explicit-questions request carries the requested model, not the built-in default"
    );
    assert_eq!(fixture.mock.models_call_count(), 0, "decide never fetches the catalog");
    fixture.observer.shutdown();
}

#[tokio::test]
async fn l2_independent_compaction_decide_carries_the_requested_model() {
    let gate: SharedGate = Arc::new(Mutex::new((JevMode::Off, "jev-1.13.0".to_string())));
    let fixture = make_fixture(vec![MockStep::Valid], gate);
    let outcome = fixture
        .observer
        .decide_independent(&payload("sess-l2"), "compaction", explicit_questions())
        .await;
    assert!(outcome.dispatched, "independent compaction must dispatch with its gate on");
    assert_eq!(fixture.mock.calls()[0].model, "jev-1.13.0");
    fixture.observer.shutdown();
}

#[tokio::test]
async fn l3_compare_observe_path_carries_the_requested_model_into_the_queue() {
    let gate: SharedGate = Arc::new(Mutex::new((JevMode::Compare, "jev-1.13.0".to_string())));
    let fixture = make_fixture(vec![MockStep::Valid], gate);
    fixture
        .observer
        .observe_prepared(&payload("sess-l3"), "retrieval", explicit_questions());
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    while fixture.mock.call_count() < 1 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the queued compare request must dispatch"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(fixture.mock.calls()[0].model, "jev-1.13.0");
    assert_eq!(fixture.mock.models_call_count(), 0);
    fixture.observer.shutdown();
}

#[tokio::test]
async fn l4_bundled_decide_active_carries_the_requested_model() {
    let gate: SharedGate = Arc::new(Mutex::new((JevMode::CompareAndActive, "jev-1.13.0".to_string())));
    let fixture = make_fixture(vec![MockStep::Valid], gate);
    let outcome = fixture
        .observer
        .decide_active(&payload("sess-l4"), pi_jev::snapshot::SnapshotStage::TurnStart, &active_policy())
        .await;
    assert!(outcome.dispatched, "the bundled decide must dispatch");
    assert_eq!(fixture.mock.calls()[0].model, "jev-1.13.0");
    fixture.observer.shutdown();
}

#[tokio::test]
async fn l5_held_active_model_change_keeps_wire_identity_but_refuses_the_old_result() {
    // One authoritative gate snapshot supplies model + durable generation.
    // Hold the mock response, move A/gen-test -> B/gen-b while it is in
    // flight, then prove the wire kept A while the old answer was refused.
    let mock = Arc::new(MockJevTransport::scripted(vec![MockStep::SlowResponse { delay_ms: 120 }]));
    let transport: Arc<dyn Transport> = mock.clone();
    let stats = Arc::new(JevStats::default());
    let system_one: Arc<dyn pi_jev::types::SystemOne> = Arc::new(
        JevSystemOne::new(
            JevMode::CompareAndActive,
            SecretString::new(SYNTHETIC_KEY),
            transport,
            JevLimits::default(),
            stats,
        )
        .unwrap(),
    );
    let truth = Arc::new(Mutex::new(("model-a".to_string(), "gen-test".to_string())));
    let truth_for_gate = Arc::clone(&truth);
    let config = JevObserverConfig {
        authoritative_gate: Some(Arc::new(move |_, _| {
            let (model, generation) = truth_for_gate.lock().unwrap().clone();
            JevRequestGate {
                mode: JevMode::CompareAndActive,
                requested_model: model,
                policy_generation: generation,
                allowed: true,
            }
        })),
        active: ActiveSettings {
            deadline: Duration::from_secs(5),
            ..ActiveSettings::default()
        },
        ..JevObserverConfig::default()
    };
    let observer = JevObserver::new(config, system_one, temp_records());
    let running_observer = Arc::clone(&observer);
    let task = tokio::spawn(async move {
        running_observer
            .decide_prepared(&payload("sess-l5"), "retrieval", explicit_questions(), &active_policy())
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while mock.call_count() < 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the held request dispatched");
    *truth.lock().unwrap() = ("model-b".to_string(), "gen-b".to_string());
    let outcome = task.await.expect("decide task");
    assert!(outcome.dispatched);
    assert_eq!(mock.calls()[0].model, "model-a", "wire identity never restamped");
    assert!(!observer.can_apply(&outcome), "the held A result is stale under B");
    assert!(outcome.raw.is_none(), "a stale result never becomes an accepted outcome");
    observer.shutdown();
}

#[tokio::test]
async fn l6_queued_compare_model_change_drops_old_work_before_second_dispatch() {
    // Seed one dispatch, then enqueue a second request inside the scheduler's
    // min-interval window. A -> B during that wait must drop the captured-A
    // job, not restamp it and not send it under either identity.
    let gate: SharedGate = Arc::new(Mutex::new((JevMode::Compare, "model-a".to_string())));
    let fixture = make_fixture_with_scheduler(
        vec![MockStep::Valid, MockStep::Valid],
        gate.clone(),
        SchedulerConfig {
            concurrency: 1,
            min_interval: Duration::from_millis(250),
            ..SchedulerConfig::default()
        },
    );
    fixture
        .observer
        .observe_prepared(&payload("sess-l6"), "retrieval", explicit_questions());
    tokio::time::timeout(Duration::from_secs(5), async {
        while fixture.mock.call_count() < 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the positive-control request dispatched");
    assert_eq!(fixture.mock.calls()[0].model, "model-a");

    fixture
        .observer
        .observe_prepared(&payload("sess-l6"), "retrieval", explicit_questions());
    *fixture.gate.lock().unwrap() = (JevMode::Compare, "model-b".to_string());
    tokio::time::timeout(Duration::from_secs(5), async {
        while fixture.observer.scheduler_metrics()["dropped_cancelled"]
            .as_u64()
            .unwrap_or(0)
            < 1
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the stale queued comparison was dropped");
    assert_eq!(
        fixture.mock.call_count(),
        1,
        "the queued A request never dispatched after the model changed to B"
    );
    fixture.observer.shutdown();
}

#[tokio::test]
async fn l7_default_requested_model_is_the_documented_native_default() {
    let gate: SharedGate = Arc::new(Mutex::new((JevMode::CompareAndActive, SYSTEM_ONE_MODEL.to_string())));
    let fixture = make_fixture(vec![MockStep::Valid], gate);
    let outcome = fixture
        .observer
        .decide_prepared(&payload("sess-l7"), "retrieval", explicit_questions(), &active_policy())
        .await;
    assert!(outcome.dispatched);
    assert_eq!(fixture.mock.calls()[0].model, "jev-latest");
    fixture.observer.shutdown();
}

#[tokio::test]
async fn l8_the_capture_seam_follows_a_b_a_at_the_request_boundary() {
    // A->B->A at the CAPTURE seam: each request carries the then-current
    // selection. (The durable identity proof lives in the settings tests.)
    let gate: SharedGate = Arc::new(Mutex::new((JevMode::CompareAndActive, "model-a".to_string())));
    let fixture = make_fixture(vec![MockStep::Valid, MockStep::Valid, MockStep::Valid], gate.clone());
    let mut recorded = Vec::new();
    for id in ["model-a", "model-b", "model-a"] {
        *fixture.gate.lock().unwrap() = (JevMode::CompareAndActive, id.to_string());
        let outcome = fixture
            .observer
            .decide_prepared(&payload("sess-l8"), "retrieval", explicit_questions(), &active_policy())
            .await;
        assert!(outcome.dispatched);
        recorded.push(fixture.mock.calls()[recorded.len()].model.clone());
    }
    assert_eq!(recorded, vec!["model-a", "model-b", "model-a"]);
    fixture.observer.shutdown();
}

// - (ADDENDUM v2.1 NEEDED-TEST-A — appends after l8; tests only)
//
// ROOT-CONTRACT v9 follow-up pin: the production single-load
// `authoritative_gate` (mode + requested model + durable generation +
// permission from ONE settings load) refuses a payload whose
// `policy_generation` was captured from an OLDER snapshot at the lane top,
// BEFORE prepare/dispatch — on decided AND queued lanes. The integrator's
// l5 pins the held-result refusal after the await and l6 pins the queued
// drop at dispatch; this test pins the remaining half: the decide path
// refuses BEFORE dispatch with ZERO wire calls, and the stale payload
// never even enqueues.

#[tokio::test]
async fn l9_a_stale_payload_generation_is_refused_before_any_wire_call() {
    let mock = Arc::new(MockJevTransport::scripted(vec![MockStep::Valid]));
    let transport: Arc<dyn Transport> = mock.clone();
    let stats = Arc::new(JevStats::default());
    let system_one: Arc<dyn pi_jev::types::SystemOne> = Arc::new(
        JevSystemOne::new(
            JevMode::CompareAndActive,
            SecretString::new(SYNTHETIC_KEY),
            transport,
            JevLimits::default(),
            stats,
        )
        .unwrap(),
    );
    let config = JevObserverConfig {
        authoritative_gate: Some(Arc::new(|_, _| JevRequestGate {
            mode: JevMode::CompareAndActive,
            requested_model: "jev-current".to_string(),
            policy_generation: "gen-current".to_string(),
            allowed: true,
        })),
        active: ActiveSettings {
            deadline: Duration::from_secs(5),
            ..ActiveSettings::default()
        },
        ..JevObserverConfig::default()
    };
    let observer = JevObserver::new(config, system_one, temp_records());
    // Decided lane: the stale payload generation is refused at the lane top
    // (payload_matches_gate), so nothing is prepared and nothing is sent.
    let mut stale = payload("sess-l9");
    stale["policy_generation"] = json!("gen-origin");
    let outcome = observer
        .decide_prepared(&stale, "retrieval", explicit_questions(), &active_policy())
        .await;
    assert!(
        !outcome.dispatched,
        "a stale payload generation must be refused before dispatch"
    );
    assert_eq!(
        outcome.terminal_reason.as_deref(),
        Some("mode_or_generation_changed"),
        "the refusal names the durable-identity mismatch"
    );
    assert_eq!(
        mock.call_count(),
        0,
        "no mixed-identity request may reach the wire"
    );
    assert_eq!(mock.models_call_count(), 0);
    // Positive control: the matching snapshot identity dispatches normally
    // and the wire carries the gate-captured model.
    let mut current = payload("sess-l9b");
    current["policy_generation"] = json!("gen-current");
    let outcome = observer
        .decide_prepared(&current, "retrieval", explicit_questions(), &active_policy())
        .await;
    assert!(outcome.dispatched);
    assert_eq!(mock.calls()[0].model, "jev-current");
    assert_eq!(mock.models_call_count(), 0);
    // Queued lane: a stale payload generation is refused BEFORE enqueueing.
    let before_calls = mock.call_count();
    let before_enqueued = observer.scheduler_metrics()["enqueued"]
        .as_u64()
        .unwrap_or(0);
    observer.observe_prepared(&stale, "retrieval", explicit_questions());
    assert_eq!(
        observer.scheduler_metrics()["enqueued"].as_u64().unwrap_or(0),
        before_enqueued,
        "the stale payload must not enqueue a compare request"
    );
    assert_eq!(mock.call_count(), before_calls);
    observer.shutdown();
}
