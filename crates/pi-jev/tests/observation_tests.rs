use std::collections::{BTreeMap, BTreeSet};
use std::time::SystemTime;

use pi_jev::active::{
    evaluate_answer, Acceptance, ActivationPolicy, AnswerCandidate, FallbackReason,
};
use pi_jev::config::JevMode;
use pi_jev::evaluators::continue_stop_escalate::ContinueStopEscalate;
use pi_jev::evaluators::first_pass_verification::FirstPassVerification;
use pi_jev::evaluators::result_sufficiency::ResultSufficiency;
use pi_jev::evaluators::retry_classification::RetryClassification;
use pi_jev::evaluators::subagent_model_routing::SubagentModelRouting;
use pi_jev::evaluators::trace_assessment::TraceAssessment;
use pi_jev::evaluators::{CategoryEvaluator, EvaluatorOutput, PreparedQuestion, StateView};
use pi_jev::observation::{
    ObservedStopReason, ResultAssessment, RetryFailureKind, RoutingMetrics,
    TraceAssessment as Assessment, TraceEvent, TraceObserver, VerificationEvidence,
    MAX_TRACE_EVENTS,
};
use pi_jev::snapshot::{SnapshotStage, StateSnapshot, MAX_STATE_BYTES};
use pi_jev::types::{validate_answer, Answer, DecisionCategory, QuestionSpec};
use serde_json::{json, Value};

fn snapshot(stage: SnapshotStage, state: Value) -> StateSnapshot {
    StateSnapshot::new(stage, "isolated-test", 1, 1, None, state, vec![]).unwrap()
}

fn questions(output: EvaluatorOutput) -> Vec<PreparedQuestion> {
    match output {
        EvaluatorOutput::Questions(questions) => questions,
        EvaluatorOutput::Skipped(reason) => panic!("unexpected skip: {reason}"),
    }
}

fn skip(output: EvaluatorOutput, expected: &str) {
    match output {
        EvaluatorOutput::Skipped(reason) => assert_eq!(reason, expected),
        EvaluatorOutput::Questions(_) => panic!("expected skip {expected}"),
    }
}

fn choice_options(question: &PreparedQuestion) -> Vec<String> {
    match &question.spec {
        QuestionSpec::Choice { criteria, .. } => criteria.keys().cloned().collect(),
        _ => panic!("expected choice"),
    }
}

fn entry_text(entry: &pi_jev::types::EntryValue) -> &str {
    match entry {
        pi_jev::types::EntryValue::Text(text) => text,
        other => panic!("expected text entry, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Tool-candidates TurnStart fallback: no-match escape, bounded subset scope,
// reserved-name collision handling (root skill-audit fix).
// ---------------------------------------------------------------------------

fn tool_candidates_turn_start(state: serde_json::Value) -> PreparedQuestion {
    use pi_jev::evaluators::tool_candidates::ToolCandidates;
    let state = snapshot(SnapshotStage::TurnStart, state);
    let prepared = questions(ToolCandidates.evaluate(&state));
    assert_eq!(prepared.len(), 1, "exactly one fallback question");
    prepared[0].clone()
}

#[test]
fn tool_candidates_fallback_carries_a_no_match_escape() {
    let prepared = tool_candidates_turn_start(json!({
        "user_text_excerpt": "fix the retry loop",
        "observed_tools": ["bash", "ipython", "grep"],
    }));
    let QuestionSpec::Choice { criteria, instructions } = &prepared.spec else {
        panic!("choice")
    };
    // The no-match escape exists and never shadows a genuine tool.
    assert!(criteria.contains_key("none"));
    assert!(criteria.contains_key("multiple"));
    assert!(criteria.contains_key("bash"));
    assert!(criteria.contains_key("ipython"));
    assert!(criteria.contains_key("grep"));
    let text = entry_text(instructions.as_ref().unwrap());
    assert!(text.contains("Choose \"none\""), "the escape is documented");
    assert!(text.contains("not that other tools cannot exist"), "none is scoped, not proof of absence");
    assert!(text.contains("never changes tool availability"), "advisory boundary stated");
}

#[test]
fn tool_candidates_fallback_discloses_the_bounded_subset() {
    let names: Vec<String> = (0..12).map(|index| format!("tool_{index}")).collect();
    let prepared = tool_candidates_turn_start(json!({
        "user_text_excerpt": "assess the task",
        "observed_tools": names,
    }));
    let QuestionSpec::Choice { criteria, instructions } = &prepared.spec else {
        panic!("choice")
    };
    // Exactly the first 8 observed names + the two escapes are offered.
    assert_eq!(criteria.len(), 10);
    for index in 0..8 {
        assert!(criteria.contains_key(&format!("tool_{index}")));
    }
    assert!(!criteria.contains_key("tool_8"));
    assert!(!criteria.contains_key("tool_11"));
    // Instructions describe ONLY the assessed subset and disclose the cap.
    let text = entry_text(instructions.as_ref().unwrap());
    assert!(text.contains("tool_7"));
    assert!(!text.contains("tool_8 "), "instructions never list unassessed names as assessable");
    assert!(text.contains("4 further observed tool name(s) are outside this bounded question"));
}

#[test]
fn tool_candidates_fallback_never_collides_with_reserved_outcomes() {
    // A tool literally named "none" would make the answer ambiguous: it is
    // excluded and the exclusion is disclosed; genuine options stay intact.
    let prepared = tool_candidates_turn_start(json!({
        "user_text_excerpt": "assess the task",
        "observed_tools": ["none", "bash", "ipython"],
    }));
    let QuestionSpec::Choice { criteria, instructions } = &prepared.spec else {
        panic!("choice")
    };
    assert!(criteria.contains_key("bash"));
    assert!(criteria.contains_key("ipython"));
    assert!(criteria.contains_key("none"));
    assert!(criteria.contains_key("multiple"));
    let text = entry_text(instructions.as_ref().unwrap());
    assert!(text.contains("1 observed tool name(s) named like the reserved outcomes"));
    // All-reserved catalog is skipped instead of asking an impossible question.
    use pi_jev::evaluators::EvaluatorOutput;
    let state = snapshot(SnapshotStage::TurnStart, json!({
        "user_text_excerpt": "assess the task",
        "observed_tools": ["none", "multiple"],
    }));
    match pi_jev::evaluators::tool_candidates::ToolCandidates.evaluate(&state) {
        EvaluatorOutput::Skipped(reason) => assert_eq!(reason, "no_assessable_tools"),
        EvaluatorOutput::Questions(_) => panic!("expected skip"),
    }
}

fn error_trace() -> TraceObserver {
    let mut observer = TraceObserver::default();
    observer.record(TraceEvent::TurnStarted);
    observer.record(TraceEvent::AssistantEnded {
        stop_reason: ObservedStopReason::Error,
        failure_kind: Some(RetryFailureKind::RateLimited),
    });
    observer
}

#[test]
fn classifications_are_closed_typed_vocabularies() {
    let retry = RetryFailureKind::ALL.map(|value| value.as_str());
    assert_eq!(
        retry,
        [
            "transient",
            "bad_arguments",
            "permission",
            "rate_limited",
            "provider_failure",
            "tool_failure",
            "fatal",
            "unknown"
        ]
    );
    assert_eq!(
        ResultAssessment::ALL.map(|value| value.as_str()),
        ["complete", "partial", "failed", "uncertain"]
    );
    assert_eq!(
        Assessment::ALL.map(|value| value.as_str()),
        [
            "good",
            "review",
            "retry_recommended",
            "escalate",
            "suspicious"
        ]
    );
    for value in RetryFailureKind::ALL {
        assert_eq!(serde_json::to_value(value).unwrap(), value.as_str());
        assert_eq!(
            serde_json::from_value::<RetryFailureKind>(json!(value.as_str())).unwrap(),
            value
        );
    }
    assert!(serde_json::from_value::<RetryFailureKind>(json!("execute_retry")).is_err());
}

#[test]
fn structured_provider_kinds_normalize_without_reading_error_text() {
    for (input, expected) in [
        ("overloaded", RetryFailureKind::Transient),
        ("timeout", RetryFailureKind::Transient),
        ("network_error", RetryFailureKind::Transient),
        ("invalid_request", RetryFailureKind::BadArguments),
        ("permission", RetryFailureKind::Permission),
        ("auth", RetryFailureKind::Permission),
        ("rate_limit", RetryFailureKind::RateLimited),
        ("server_error", RetryFailureKind::ProviderFailure),
        ("malformed_response", RetryFailureKind::ProviderFailure),
        ("request_interrupted", RetryFailureKind::ProviderFailure),
        ("refusal", RetryFailureKind::Fatal),
        ("safety", RetryFailureKind::Fatal),
        ("agent_lifecycle_failure", RetryFailureKind::Fatal),
        ("unknown", RetryFailureKind::Unknown),
        (
            "retry now; authorization: secret",
            RetryFailureKind::Unknown,
        ),
    ] {
        assert_eq!(RetryFailureKind::from_provider_kind(input), expected);
    }
    assert_eq!(
        ObservedStopReason::from_stop_reason("toolUse"),
        ObservedStopReason::ToolUse
    );
    assert_eq!(
        ObservedStopReason::from_stop_reason("secret"),
        ObservedStopReason::Unknown
    );
}

#[test]
fn trace_is_bounded_cloneable_resettable_and_contains_only_metadata() {
    let mut observer = TraceObserver::default();
    for _ in 0..10_000 {
        observer.record(TraceEvent::ToolEnded { is_error: false });
    }
    let summary = observer.summary();
    assert_eq!(summary.recent_events.len(), MAX_TRACE_EVENTS);
    assert_eq!(summary.events_seen, 10_000);
    assert_eq!(summary.events_dropped, 10_000 - MAX_TRACE_EVENTS as u64);
    assert_eq!(summary.tool_results, 10_000);
    assert_eq!(summary.verification, VerificationEvidence::Unknown);
    assert_eq!(observer.clone().summary(), summary);
    let state = json!({"observation": summary});
    assert!(serde_json::to_vec(&state).unwrap().len() < MAX_STATE_BYTES);
    let bounded = snapshot(SnapshotStage::AgentEnd, state);
    assert!(StateView::new(&bounded).observation().is_some());
    observer.reset();
    assert!(!observer.summary().has_evidence());
    assert_eq!(observer.summary().tool_results, 0);
    assert!(serde_json::from_value::<TraceEvent>(
        json!({"event":"tool_ended", "is_error": false, "output":"secret"})
    )
    .is_err());
}

#[test]
fn missing_verification_and_actual_retry_remain_unknown() {
    let mut observer = error_trace();
    observer.record(TraceEvent::ToolEnded { is_error: false });
    assert_eq!(
        observer.summary().verification,
        VerificationEvidence::Unknown
    );
    assert_eq!(observer.summary().retries_observed, 0);
    assert_eq!(observer.summary().last_retry_attempt, None);
    observer.record(TraceEvent::RetryObserved { attempt: 2 });
    assert_eq!(observer.summary().retries_observed, 1);
    assert_eq!(observer.summary().last_retry_attempt, Some(2));
    observer.record(TraceEvent::VerificationObserved {
        outcome: VerificationEvidence::Passed,
    });
    assert_eq!(
        observer.summary().verification,
        VerificationEvidence::Passed
    );
    observer.record(TraceEvent::TurnStarted);
    assert_eq!(observer.summary().failure_kind, None);
    assert_eq!(observer.summary().last_stop_reason, None);
    assert_eq!(
        observer.summary().verification,
        VerificationEvidence::Unknown
    );
}

#[test]
fn cancelled_and_successful_messages_do_not_fabricate_provider_failures() {
    let mut observer = TraceObserver::default();
    for reason in [ObservedStopReason::Aborted, ObservedStopReason::Stop] {
        observer.record(TraceEvent::AssistantEnded {
            stop_reason: reason,
            failure_kind: Some(RetryFailureKind::Fatal),
        });
    }
    assert_eq!(observer.summary().assistant_errors, 0);
    assert_eq!(observer.summary().failure_kind, None);
    observer.record(TraceEvent::ToolEnded { is_error: true });
    assert_eq!(observer.summary().tool_errors, 1);
    assert_eq!(
        observer.summary().failure_kind,
        Some(RetryFailureKind::ToolFailure)
    );
}

#[test]
fn new_categories_require_opt_in_and_real_evidence() {
    let off = snapshot(
        SnapshotStage::AgentEnd,
        json!({"observation": error_trace().summary()}),
    );
    skip(RetryClassification.evaluate(&off), "feature_disabled");
    skip(TraceAssessment.evaluate(&off), "feature_disabled");
    let missing = snapshot(
        SnapshotStage::AgentEnd,
        json!({"features":{"retry_classification":true,"trace_observer":true}}),
    );
    skip(
        RetryClassification.evaluate(&missing),
        "no_failure_observed",
    );
    skip(TraceAssessment.evaluate(&missing), "no_trace_observed");
    let enabled = snapshot(
        SnapshotStage::AgentEnd,
        json!({"features":{"retry_classification":true,"trace_observer":true},"observation":error_trace().summary()}),
    );
    assert_eq!(
        questions(RetryClassification.evaluate(&enabled))[0].question_id,
        "retry_classification.0"
    );
    assert_eq!(
        questions(TraceAssessment.evaluate(&enabled))[0].question_id,
        "trace_assessment.0"
    );
    let empty = snapshot(
        SnapshotStage::AgentEnd,
        json!({"features":{"trace_observer":true},"observation":TraceObserver::default().summary()}),
    );
    skip(TraceAssessment.evaluate(&empty), "no_trace_observed");
}

#[test]
fn result_enhancement_is_additive_and_legacy_choices_remain() {
    let legacy = snapshot(
        SnapshotStage::AgentEnd,
        json!({"result_excerpt":"finished"}),
    );
    let old = questions(ResultSufficiency.evaluate(&legacy));
    assert_eq!(old.len(), 1);
    assert_eq!(
        choice_options(&old[0]),
        ["insufficient", "sufficient", "unknown"]
    );
    let enhanced = snapshot(
        SnapshotStage::AgentEnd,
        json!({"features":{"result_sufficiency":true},"result_excerpt":"finished","user_text_excerpt":"fix bug"}),
    );
    let added = questions(ResultSufficiency.evaluate(&enhanced));
    assert_eq!(added.len(), 2);
    assert_eq!(added[0].spec, old[0].spec);
    assert_eq!(added[1].question_id, "result_sufficiency.1");
    assert_eq!(
        choice_options(&added[1]),
        ["complete", "failed", "partial", "uncertain"]
    );
    assert!(added[1]
        .spec
        .instructions()
        .contains("verification is unknown"));
    skip(
        ResultSufficiency.evaluate(&snapshot(
            SnapshotStage::AgentEnd,
            json!({"features":{"result_sufficiency":true}}),
        )),
        "no_result_observed",
    );
}

#[test]
fn loop_observation_adds_no_stop_authority_and_gates_turn_end() {
    let off = snapshot(SnapshotStage::TurnEnd, json!({"result_excerpt":"work"}));
    skip(ContinueStopEscalate.evaluate(&off), "feature_disabled");
    let enabled = snapshot(
        SnapshotStage::TurnEnd,
        json!({"features":{"loop_control":true},"observation":error_trace().summary()}),
    );
    let prepared = questions(ContinueStopEscalate.evaluate(&enabled));
    assert_eq!(
        choice_options(&prepared[0]),
        ["continue", "escalate", "stop"]
    );
    assert!(prepared[0]
        .spec
        .instructions()
        .contains("host keeps stopping"));
    assert!(prepared[0]
        .spec
        .instructions()
        .contains("does not establish task completion"));
    assert_eq!(
        questions(ContinueStopEscalate.evaluate(&snapshot(
            SnapshotStage::AgentEnd,
            json!({"result_excerpt":"done"})
        )))
        .len(),
        1
    );
}

#[test]
fn verification_never_calls_successful_tools_test_evidence() {
    let mut observer = TraceObserver::default();
    observer.record(TraceEvent::ToolEnded { is_error: false });
    let state = snapshot(
        SnapshotStage::AgentEnd,
        json!({"features":{"verification":true},"result_excerpt":"done","observation":observer.summary()}),
    );
    let prepared = questions(FirstPassVerification.evaluate(&state));
    assert_eq!(
        choice_options(&prepared[0]),
        ["escalate", "none", "rerun", "verify"]
    );
    assert!(prepared[0]
        .spec
        .instructions()
        .contains("verification=Unknown"));
    assert!(prepared[0]
        .spec
        .instructions()
        .contains("Successful tool execution is not verification"));
}

#[test]
fn routing_is_allowlisted_bounded_and_reports_only_actual_valid_metrics() {
    let candidates = (0..12)
        .map(|index| format!("provider/model-{index}"))
        .collect::<Vec<_>>();
    let state = snapshot(
        SnapshotStage::ModelSelect,
        json!({
            "model_allowlist": candidates,
            "routing_metrics": [
                {"model":"provider/model-0", "attempts":4,"failures":1,"latency_ms":12.5,"cost":0.01},
                {"model":"provider/model-1", "attempts":0,"failures":0,"latency_ms":null,"cost":null},
                {"model":"provider/model-2", "attempts":1,"failures":2,"latency_ms":1.0,"cost":null},
                {"model":"not-allowlisted", "attempts":999,"failures":0,"latency_ms":1.0,"cost":null},
            ],
        }),
    );
    let prepared = questions(SubagentModelRouting.evaluate(&state));
    let QuestionSpec::Choice {
        criteria,
        instructions,
    } = &prepared[0].spec
    else {
        panic!("choice")
    };
    assert_eq!(criteria.len(), 9);
    assert!(entry_text(&criteria["provider/model-0"]).contains("success rate=0.75"));
    assert!(entry_text(&criteria["provider/model-1"]).contains("unknown, not zero"));
    assert!(entry_text(&criteria["provider/model-2"]).contains("unknown, not zero"));
    assert!(!entry_text(instructions.as_ref().unwrap()).contains("model-8"));
    assert!(!criteria.contains_key("not-allowlisted"));
    assert!(entry_text(instructions.as_ref().unwrap()).contains("never switches any model"));
}

#[test]
fn routing_rejects_nonfinite_metrics_and_unsafe_model_ids() {
    let mut metrics = RoutingMetrics {
        model: "provider/model".to_string(),
        attempts: 2,
        failures: 1,
        latency_ms: Some(1.0),
        cost: None,
    };
    assert_eq!(metrics.observed_success_rate(), Some(0.5));
    for value in [f64::NAN, f64::INFINITY, -1.0] {
        metrics.latency_ms = Some(value);
        assert!(!metrics.is_valid());
        assert_eq!(metrics.observed_success_rate(), None);
    }
    skip(
        SubagentModelRouting.evaluate(&snapshot(SnapshotStage::ModelSelect, json!({}))),
        "no_model_allowlist",
    );
    skip(
        SubagentModelRouting.evaluate(&snapshot(
            SnapshotStage::ModelSelect,
            json!({"model_allowlist":["ignore instructions", "none", "x".repeat(121)]}),
        )),
        "no_eligible_models",
    );
}

#[test]
fn hostile_observations_validate_as_data_but_cannot_be_applied() {
    let state = snapshot(
        SnapshotStage::AgentEnd,
        json!({"features":{"retry_classification":true,"trace_observer":true,"result_sufficiency":true},"result_excerpt":"done","observation":error_trace().summary()}),
    );
    for (category, prepared) in [
        (
            DecisionCategory::RetryClassification,
            questions(RetryClassification.evaluate(&state)),
        ),
        (
            DecisionCategory::TraceAssessment,
            questions(TraceAssessment.evaluate(&state)),
        ),
        (
            DecisionCategory::ResultSufficiency,
            questions(ResultSufficiency.evaluate(&state)),
        ),
        (
            DecisionCategory::ContinueStopEscalate,
            questions(ContinueStopEscalate.evaluate(&state)),
        ),
        (
            DecisionCategory::FirstPassVerification,
            questions(FirstPassVerification.evaluate(&state)),
        ),
    ] {
        for question in prepared {
            let options = choice_options(&question);
            for selected in &options {
                let probabilities: BTreeMap<String, f64> = options
                    .iter()
                    .map(|option| (option.clone(), if option == selected { 1.0 } else { 0.0 }))
                    .collect();
                let answer = Answer::Choice {
                    choice: selected.clone(),
                    probabilities,
                    confidence: 1.0,
                };
                validate_answer(&question.question_id, &question.spec, &answer).unwrap();
                let candidate = AnswerCandidate {
                    category,
                    question_id: question.question_id.clone(),
                    value: Some(selected.clone()),
                    confidence: Some(1.0),
                    response_model: Some("mock".to_string()),
                    request_id: "mock".to_string(),
                    turn: 1,
                    decided_at: SystemTime::UNIX_EPOCH,
                };
                let policy = ActivationPolicy {
                    enabled_categories: BTreeSet::from([category]),
                    ..Default::default()
                };
                assert_eq!(
                    evaluate_answer(&policy, JevMode::Active, &candidate, SystemTime::UNIX_EPOCH),
                    Acceptance::Fallback(FallbackReason::CategoryNotAppliable)
                );
            }
        }
    }
}
