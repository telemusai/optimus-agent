//! CONTROL-lane tests for the pure control library (`pi_jev::control`).
//!
//! These are pure-logic tests: no transport, no SystemOne calls, no mock
//! provider. They pin the ROOT CONTRACT v1 semantics:
//! - thresholds are the approved starting constants and record provenance;
//! - budgets are maxima per real user task epoch, consumed at acceptance
//!   time, gating feedback and vetoes but never pausing itself;
//! - correlation and freshness are computed from host facts only;
//! - the veto set is {bad_arguments, fatal} at the high-impact floor;
//! - no verification state is derivable from tool transport success;
//! - nonprogress is content-signature based, never repeated tool names.

use std::time::{Duration, SystemTime};

use pi_jev::active::{AnswerCandidate, FallbackReason};
use pi_jev::config::JevMode;
use pi_jev::control::{
    combine_sufficiency, control_gates_open, evaluate_control_answer, nonprogress_verdict,
    verification_need, CorrelatedVerificationEvidence, ControlBoundary, ControlBudgetKind,
    ControlBudgetSnapshot, ControlBudgets, ControlEffectKind, ControlFeatures, ControlPolicy,
    ControlRefusal, ControlVerificationState, FeedbackKind, HostControlFacts,
    NonprogressVerdict, PauseReason, SufficiencyVerdict, TurnSignature, VerificationNeed,
    VerificationSourceKind, CONTROL_ACT_MIN_CONFIDENCE, CONTROL_HIGH_IMPACT_MIN_CONFIDENCE,
    CONTROL_MAX_DECISION_AGE, CONTROL_MAX_FEEDBACK, CONTROL_MAX_NONPROGRESS_CORRECTIONS,
    CONTROL_MAX_RETRY_VETOES, CONTROL_MAX_VERIFICATION_REQUESTS,
    CONTROL_NONPROGRESS_MIN_IDENTICAL_TURNS,
};
use pi_jev::observation::RetryFailureKind;
use pi_jev::types::DecisionCategory;

const SESSION: &str = "sess-a";
const EPOCH: &str = "sess-a:1";
const REQUEST: &str = "req-1";

fn features() -> ControlFeatures {
    ControlFeatures {
        result_sufficiency: true,
        loop_control: true,
        verification: true,
        retry_classification: true,
        full_jev_active: true,
    }
}

fn facts(turn: u64, question_ids: &[&str]) -> HostControlFacts {
    HostControlFacts {
        now: SystemTime::now(),
        session_id: SESSION.to_string(),
        turn,
        expected_question_ids: question_ids.iter().map(|id| id.to_string()).collect(),
        policy_generation: "gen-1".to_string(),
        full_jev_stamp: "stamp-1".to_string(),
        prompt_version: "jev-control-prompts/1".to_string(),
        epoch_id: EPOCH.to_string(),
        request_id: REQUEST.to_string(),
    }
}

fn candidate(
    category: DecisionCategory,
    question_id: &str,
    value: &str,
    confidence: f64,
    turn: u64,
) -> AnswerCandidate {
    AnswerCandidate {
        category,
        question_id: question_id.to_string(),
        value: Some(value.to_string()),
        confidence: Some(confidence),
        response_model: Some("test-model".to_string()),
        request_id: REQUEST.to_string(),
        turn,
        // Stamped strictly BEFORE the helper-built facts.now (facts() runs
        // first in every test): decided_at must be in the past of the host
        // clock or freshness would refuse with Stale by construction.
        decided_at: SystemTime::now() - Duration::from_millis(100),
    }
}

fn budget() -> ControlBudgetSnapshot {
    ControlBudgetSnapshot {
        epoch_id: EPOCH.to_string(),
        feedback_remaining: 2,
        verification_remaining: 1,
        nonprogress_remaining: 1,
        veto_remaining: 2,
        available: true,
    }
}

fn refused_reason(
    verdict: pi_jev::control::ControlVerdict,
) -> String {
    match verdict {
        pi_jev::control::ControlVerdict::Refused(reason) => reason.as_str().to_string(),
        pi_jev::control::ControlVerdict::Applied(_) => "applied".to_string(),
    }
}

#[test]
fn contract_constants_are_the_approved_maxima() {
    assert_eq!(CONTROL_ACT_MIN_CONFIDENCE, 0.70);
    assert_eq!(CONTROL_HIGH_IMPACT_MIN_CONFIDENCE, 0.85);
    assert_eq!(CONTROL_MAX_DECISION_AGE, Duration::from_secs(3));
    let maxima = ControlBudgets::maxima();
    assert_eq!(maxima.feedback, 2);
    assert_eq!(maxima.verification_requests, 1);
    assert_eq!(maxima.nonprogress_corrections, 1);
    assert_eq!(maxima.retry_vetoes, 2);
    assert_eq!(CONTROL_NONPROGRESS_MIN_IDENTICAL_TURNS, 2);
}

#[test]
fn gates_close_without_full_profile_or_active_mode() {
    // Local named `gates` so the `features()` helper stays callable below.
    let mut gates = features();
    assert!(control_gates_open(&gates, JevMode::Active));
    assert!(control_gates_open(&gates, JevMode::CompareAndActive));
    assert!(!control_gates_open(&gates, JevMode::Off));
    assert!(!control_gates_open(&gates, JevMode::Compare));
    gates.full_jev_active = false;
    assert!(!control_gates_open(&gates, JevMode::Active));

    let fact = facts(3, &["result_sufficiency.0"]);
    let mut answer = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.95,
        3,
    );
    // Gates closed by mode.
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Off,
        ControlBoundary::AgentEnd,
        &answer,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "gates_closed");

    // Gates closed by feature: the refusal names the disabled feature
    // (distinct from the mode/profile "gates_closed" tag).
    let mut feature_off = features();
    feature_off.result_sufficiency = false;
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &feature_off,
        JevMode::Active,
        ControlBoundary::AgentEnd,
        &answer,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "feature_disabled");

    // No session facts: nothing may apply.
    answer.category = DecisionCategory::ResultSufficiency;
    let mut empty_facts = facts(3, &["result_sufficiency.0"]);
    empty_facts.session_id = String::new();
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::AgentEnd,
        &answer,
        &empty_facts,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "no_answer");
}

#[test]
fn act_floor_applies_to_corrective_feedback() {
    let fact = facts(7, &["result_sufficiency.0"]);
    let at_floor = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        CONTROL_ACT_MIN_CONFIDENCE,
        7,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::AgentEnd,
        &at_floor,
        &fact,
        &budget(),
    );
    match &verdict {
        pi_jev::control::ControlVerdict::Applied(acceptance) => {
            assert_eq!(
                acceptance.effect,
                ControlEffectKind::Feedback(FeedbackKind::ResultGap)
            );
            assert_eq!(acceptance.category, DecisionCategory::ResultSufficiency);
            assert!(acceptance.complete && acceptance.correlated && acceptance.fresh);
            assert_eq!(acceptance.value, "insufficient");
            assert_eq!(
                acceptance.provenance.prompt_version,
                "jev-control-prompts/1"
            );
            assert_eq!(acceptance.provenance.full_jev_stamp, "stamp-1");
        }
        _ => panic!("expected applied result-gap feedback"),
    }

    let below = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        CONTROL_ACT_MIN_CONFIDENCE - 0.01,
        7,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::AgentEnd,
        &below,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "low_confidence");
}

#[test]
fn insufficient_values_map_to_result_gap_feedback() {
    let fact = facts(7, &["result_sufficiency.1"]);
    for value in ["insufficient", "partial", "failed"] {
        let answer = candidate(
            DecisionCategory::ResultSufficiency,
            "result_sufficiency.1",
            value,
            0.9,
            7,
        );
        let verdict = evaluate_control_answer(
            &ControlPolicy::default(),
            &features(),
            JevMode::Active,
            ControlBoundary::AgentEnd,
            &answer,
            &fact,
            &budget(),
        );
        match verdict {
            pi_jev::control::ControlVerdict::Applied(acceptance) => assert_eq!(
                acceptance.effect,
                ControlEffectKind::Feedback(FeedbackKind::ResultGap)
            ),
            _ => panic!("expected applied feedback for {value}"),
        }
    }
}

#[test]
fn high_impact_floor_governs_pause_escalate_veto_verification() {
    // Continue/stop/escalate at TurnEnd.
    let fact = facts(4, &["continue_stop_escalate.0"]);
    let stop_at_floor = candidate(
        DecisionCategory::ContinueStopEscalate,
        "continue_stop_escalate.0",
        "stop",
        CONTROL_HIGH_IMPACT_MIN_CONFIDENCE,
        4,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::TurnEnd,
        &stop_at_floor,
        &fact,
        &budget(),
    );
    match verdict {
        pi_jev::control::ControlVerdict::Applied(acceptance) => assert_eq!(
            acceptance.effect,
            ControlEffectKind::Pause(PauseReason::NonprogressUncorrected)
        ),
        _ => panic!("expected applied pause"),
    }

    let stop_below = candidate(
        DecisionCategory::ContinueStopEscalate,
        "continue_stop_escalate.0",
        "stop",
        CONTROL_HIGH_IMPACT_MIN_CONFIDENCE - 0.01,
        4,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::TurnEnd,
        &stop_below,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "low_confidence");

    // Verification request at AgentEnd.
    let fact = facts(9, &["first_pass_verification.0"]);
    let verify_below = candidate(
        DecisionCategory::FirstPassVerification,
        "first_pass_verification.0",
        "verify",
        CONTROL_HIGH_IMPACT_MIN_CONFIDENCE - 0.01,
        9,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::AgentEnd,
        &verify_below,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "low_confidence");

    // Retry veto.
    let fact = facts(2, &["retry_classification.0"]);
    let veto_below = candidate(
        DecisionCategory::RetryClassification,
        "retry_classification.0",
        "fatal",
        CONTROL_HIGH_IMPACT_MIN_CONFIDENCE - 0.01,
        2,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::RetryDecision,
        &veto_below,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "low_confidence");
}

#[test]
fn correlation_requires_host_owned_ids_turn_and_epoch() {
    let answer = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );
    let policy = ControlPolicy::default();
    let mode = JevMode::Active;
    let feature_set = features();

    // Wrong question id.
    let fact = facts(7, &["result_sufficiency.1"]);
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::AgentEnd,
        &answer,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "no_answer");

    // Wrong turn.
    let fact = facts(6, &["result_sufficiency.0"]);
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::AgentEnd,
        &answer,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "no_answer");

    // Wrong request id on the answer.
    let fact = facts(7, &["result_sufficiency.0"]);
    let mut foreign = answer.clone();
    foreign.request_id = "req-other".to_string();
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::AgentEnd,
        &foreign,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "no_answer");

    // Empty host request id: nothing may apply.
    let mut empty_request = facts(7, &["result_sufficiency.0"]);
    empty_request.request_id = String::new();
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::AgentEnd,
        &answer,
        &empty_request,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "no_answer");

    // Epoch mismatch between facts and budget snapshot.
    let mut other_epoch = budget();
    other_epoch.epoch_id = "sess-a:2".to_string();
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::AgentEnd,
        &answer,
        &fact,
        &other_epoch,
    );
    assert_eq!(refused_reason(verdict), "no_answer");
}

#[test]
fn freshness_bounds_decision_age() {
    let policy = ControlPolicy::default();
    let fact = facts(7, &["result_sufficiency.0"]);
    let mode = JevMode::Active;
    let feature_set = features();
    let within = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::AgentEnd,
        &within,
        &fact,
        &budget(),
    );
    assert!(matches!(
        verdict,
        pi_jev::control::ControlVerdict::Applied(_)
    ));

    let mut stale = within.clone();
    stale.decided_at = SystemTime::now() - (policy.max_decision_age + Duration::from_secs(1));
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::AgentEnd,
        &stale,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "stale");

    // A future-stamped decision is not usable evidence.
    let mut future = within.clone();
    future.decided_at = SystemTime::now() + Duration::from_secs(1);
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::AgentEnd,
        &future,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "stale");
}

#[test]
fn veto_set_is_bad_arguments_and_fatal_only() {
    let policy = ControlPolicy::default();
    let mode = JevMode::Active;
    let feature_set = features();
    let fact = facts(2, &["retry_classification.0"]);
    for (value, kind) in [
        ("transient", RetryFailureKind::Transient),
        ("bad_arguments", RetryFailureKind::BadArguments),
        ("permission", RetryFailureKind::Permission),
        ("rate_limited", RetryFailureKind::RateLimited),
        ("provider_failure", RetryFailureKind::ProviderFailure),
        ("tool_failure", RetryFailureKind::ToolFailure),
        ("fatal", RetryFailureKind::Fatal),
        ("unknown", RetryFailureKind::Unknown),
    ] {
        let answer = candidate(
            DecisionCategory::RetryClassification,
            "retry_classification.0",
            value,
            0.9,
            2,
        );
        let verdict = evaluate_control_answer(
            &policy,
            &feature_set,
            mode,
            ControlBoundary::RetryDecision,
            &answer,
            &fact,
            &budget(),
        );
        match verdict {
            pi_jev::control::ControlVerdict::Applied(acceptance) => match kind {
                RetryFailureKind::BadArguments | RetryFailureKind::Fatal => {
                    assert_eq!(
                        acceptance.effect,
                        ControlEffectKind::RetryVeto(kind),
                        "{value} must veto"
                    );
                }
                _ => assert_eq!(
                    acceptance.effect, ControlEffectKind::Continue,
                    "{value} must fall back to baseline retries"
                ),
            },
            _ => panic!("expected applied verdict for {value}"),
        }
    }

    // Outside the vocabulary: invalid value.
    let bogus = candidate(
        DecisionCategory::RetryClassification,
        "retry_classification.0",
        "vibes",
        0.9,
        2,
    );
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::RetryDecision,
        &bogus,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "invalid_value");
}

#[test]
fn retry_classification_is_record_only_at_agent_end() {
    let fact = facts(2, &["retry_classification.0"]);
    let answer = candidate(
        DecisionCategory::RetryClassification,
        "retry_classification.0",
        "fatal",
        0.9,
        2,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::AgentEnd,
        &answer,
        &fact,
        &budget(),
    );
    match verdict {
        pi_jev::control::ControlVerdict::Applied(acceptance) => {
            assert_eq!(acceptance.effect, ControlEffectKind::Continue);
        }
        _ => panic!("expected record-only continue at agent end"),
    }
}

#[test]
fn boundaries_own_their_categories() {
    let policy = ControlPolicy::default();
    let mode = JevMode::Active;
    let feature_set = features();
    let answer = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );
    let fact = facts(7, &["result_sufficiency.0"]);
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::TurnEnd,
        &answer,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "wrong_boundary");

    let retry = candidate(
        DecisionCategory::RetryClassification,
        "retry_classification.0",
        "fatal",
        0.9,
        7,
    );
    let fact = facts(7, &["retry_classification.0"]);
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::TurnEnd,
        &retry,
        &fact,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "wrong_boundary");
}

#[test]
fn budgets_gate_feedback_and_veto_but_never_pause() {
    let policy = ControlPolicy::default();
    let mode = JevMode::Active;
    let feature_set = features();

    // Spent feedback budget refuses corrective feedback.
    let fact = facts(7, &["result_sufficiency.0"]);
    let answer = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.9,
        7,
    );
    let mut spent_feedback = budget();
    spent_feedback.feedback_remaining = 0;
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::AgentEnd,
        &answer,
        &fact,
        &spent_feedback,
    );
    assert_eq!(refused_reason(verdict), "budget_exhausted_feedback");

    // Spent veto budget refuses the veto (baseline retries proceed).
    let fact = facts(2, &["retry_classification.0"]);
    let retry = candidate(
        DecisionCategory::RetryClassification,
        "retry_classification.0",
        "fatal",
        0.9,
        2,
    );
    let mut spent_veto = budget();
    spent_veto.veto_remaining = 0;
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::RetryDecision,
        &retry,
        &fact,
        &spent_veto,
    );
    assert_eq!(refused_reason(verdict), "budget_exhausted_veto");

    // Pause is never budget-gated: stopping is free.
    let fact = facts(4, &["continue_stop_escalate.0"]);
    let stop = candidate(
        DecisionCategory::ContinueStopEscalate,
        "continue_stop_escalate.0",
        "stop",
        0.9,
        4,
    );
    let mut spent_nonprogress = budget();
    spent_nonprogress.nonprogress_remaining = 0;
    let verdict = evaluate_control_answer(
        &policy,
        &feature_set,
        mode,
        ControlBoundary::TurnEnd,
        &stop,
        &fact,
        &spent_nonprogress,
    );
    assert!(matches!(
        verdict,
        pi_jev::control::ControlVerdict::Applied(_)
    ));
}

#[test]
fn verification_feedback_draws_both_budgets() {
    let start = ControlBudgetSnapshot {
        epoch_id: "sess-b:1".to_string(),
        feedback_remaining: CONTROL_MAX_FEEDBACK,
        verification_remaining: CONTROL_MAX_VERIFICATION_REQUESTS,
        nonprogress_remaining: CONTROL_MAX_NONPROGRESS_CORRECTIONS,
        veto_remaining: CONTROL_MAX_RETRY_VETOES,
        available: true,
    };
    assert_eq!(start.feedback_remaining, 2);
    assert_eq!(start.verification_remaining, 1);
    let next = start.consume(ControlBudgetKind::Feedback(FeedbackKind::VerificationMissing));
    assert_eq!(next.feedback_remaining, 1);
    assert_eq!(next.verification_remaining, 0);
    assert!(!next.allows(ControlBudgetKind::Feedback(FeedbackKind::VerificationMissing)));
    // A second verification request is impossible this epoch even with
    // feedback budget left.
    let again = next.consume(ControlBudgetKind::Feedback(FeedbackKind::VerificationMissing));
    assert_eq!(again.verification_remaining, 0);
    assert!(!again.allows(ControlBudgetKind::Feedback(FeedbackKind::VerificationMissing)));

    // Result-gap feedback draws only the shared feedback budget.
    let gap = start.consume(ControlBudgetKind::Feedback(FeedbackKind::ResultGap));
    assert_eq!(gap.feedback_remaining, 1);
    assert_eq!(gap.verification_remaining, 1);
}

#[test]
fn verification_evidence_requires_explicit_correlated_fields() {
    // The ONLY constructor of Verified/Failed states. It must reject empty
    // or oversized fields: transport success (a tool name plus is_error=false)
    // has no path into a verification state.
    assert!(CorrelatedVerificationEvidence::new(
        VerificationSourceKind::Test,
        "run-digest",
        "scope-digest",
        "outcome-digest",
        "2026-09-21T00:00:00Z",
        "sess-b:1",
    )
    .is_some());

    assert!(CorrelatedVerificationEvidence::new(
        VerificationSourceKind::Test,
        "",
        "scope-digest",
        "outcome-digest",
        "2026-09-21T00:00:00Z",
        "sess-b:1",
    )
    .is_none());
    assert!(CorrelatedVerificationEvidence::new(
        VerificationSourceKind::Build,
        "run-digest",
        "scope-digest",
        "outcome-digest",
        "",
        "sess-b:1",
    )
    .is_none());
    let oversized = "x".repeat(129);
    assert!(CorrelatedVerificationEvidence::new(
        VerificationSourceKind::Check,
        oversized.clone(),
        "scope-digest",
        "outcome-digest",
        "2026-09-21T00:00:00Z",
        "sess-b:1",
    )
    .is_none());
}

#[test]
fn applied_acceptances_never_carry_a_verification_outcome() {
    // Structural honesty check: an acceptance records an effect and provenance
    // only. There is no field by which is_error=false could become Verified;
    // ControlVerificationState::Verified is reachable only through
    // CorrelatedVerificationEvidence (previous test).
    let fact = facts(7, &["result_sufficiency.0"]);
    let answer = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "sufficient",
        0.9,
        7,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::AgentEnd,
        &answer,
        &fact,
        &budget(),
    );
    match verdict {
        pi_jev::control::ControlVerdict::Applied(acceptance) => {
            assert_eq!(acceptance.effect, ControlEffectKind::Continue);
            assert_eq!(acceptance.budget.epoch_id, EPOCH);
        }
        _ => panic!("expected applied continue"),
    }
    // Honest default states exist and stay distinct from Verified/Failed.
    assert_ne!(ControlVerificationState::Unknown, ControlVerificationState::NotApplicable);
    assert_ne!(ControlVerificationState::Unknown, ControlVerificationState::Unverified);
}

#[test]
fn nonprogress_needs_two_identical_no_evidence_turns() {
    let signature = TurnSignature {
        tool_calls_digest: "call-1".to_string(),
        text_digest: "text-1".to_string(),
        results_digest: "results-1".to_string(),
        has_new_evidence: false,
    };
    assert_eq!(nonprogress_verdict(&[]), NonprogressVerdict::None);
    assert_eq!(nonprogress_verdict(&[signature.clone()]), NonprogressVerdict::None);
    assert_eq!(
        nonprogress_verdict(&[signature.clone(), signature.clone()]),
        NonprogressVerdict::Candidate { identical_turns: 2 }
    );
    assert_eq!(
        nonprogress_verdict(&[
            signature.clone(),
            signature.clone(),
            signature.clone(),
        ]),
        NonprogressVerdict::Candidate { identical_turns: 3 }
    );

    // New evidence on the latest turn breaks the candidate.
    let mut fresh = signature.clone();
    fresh.has_new_evidence = true;
    assert_eq!(
        nonprogress_verdict(&[signature.clone(), fresh]),
        NonprogressVerdict::None
    );

    // A single reasoning-only/empty turn is never nonprogress.
    let empty = TurnSignature::default();
    assert_eq!(nonprogress_verdict(&[empty]), NonprogressVerdict::None);
}

#[test]
fn nonprogress_ignores_repeated_tool_name_with_new_content() {
    // Same tool NAME both turns, but the result content differs: not stuck.
    let first = TurnSignature {
        tool_calls_digest: "same-tool".to_string(),
        text_digest: "t1".to_string(),
        results_digest: "r1".to_string(),
        has_new_evidence: false,
    };
    let second = TurnSignature {
        tool_calls_digest: "same-tool".to_string(),
        text_digest: "t1".to_string(),
        results_digest: "r2".to_string(),
        has_new_evidence: false,
    };
    assert_eq!(
        nonprogress_verdict(&[first, second]),
        NonprogressVerdict::None
    );

    // Different tool NAME with identical arguments and results: the call
    // signature changed, so the predicate conservatively does NOT flag it
    // (identical digests across ALL components are required; a name-only
    // difference may be a genuinely different action).
    let renamed = TurnSignature {
        tool_calls_digest: "other-tool".to_string(),
        text_digest: "t1".to_string(),
        results_digest: "r1".to_string(),
        has_new_evidence: false,
    };
    let base = TurnSignature {
        tool_calls_digest: "same-tool".to_string(),
        text_digest: "t1".to_string(),
        results_digest: "r1".to_string(),
        has_new_evidence: false,
    };
    assert_eq!(
        nonprogress_verdict(&[base, renamed]),
        NonprogressVerdict::None
    );
}

#[test]
fn sufficiency_combines_primary_and_coverage() {
    assert_eq!(
        combine_sufficiency(Some("sufficient"), Some("complete")),
        SufficiencyVerdict::Sufficient
    );
    assert_eq!(
        combine_sufficiency(Some("Sufficient"), None),
        SufficiencyVerdict::Sufficient
    );
    assert_eq!(
        combine_sufficiency(Some("sufficient"), Some("partial")),
        SufficiencyVerdict::Insufficient
    );
    assert_eq!(
        combine_sufficiency(Some("complete"), Some("failed")),
        SufficiencyVerdict::Insufficient
    );
    assert_eq!(
        combine_sufficiency(Some("unknown"), Some("uncertain")),
        SufficiencyVerdict::Unknown
    );
    assert_eq!(combine_sufficiency(None, None), SufficiencyVerdict::Unknown);
    // Uncertain never escalates to insufficient on its own (no guesses).
    assert_eq!(
        combine_sufficiency(Some("uncertain"), None),
        SufficiencyVerdict::Unknown
    );
}

#[test]
fn verification_need_maps_recommendations() {
    assert_eq!(verification_need(Some("none")), VerificationNeed::NotRequired);
    assert_eq!(verification_need(Some("verify")), VerificationNeed::Request);
    assert_eq!(verification_need(Some("rerun")), VerificationNeed::Request);
    assert_eq!(verification_need(Some("escalate")), VerificationNeed::Escalate);
    assert_eq!(verification_need(Some("unknown")), VerificationNeed::Unknown);
    assert_eq!(verification_need(None), VerificationNeed::Unknown);
}

#[test]
fn feedback_kinds_and_pause_reasons_have_stable_ids() {
    assert_eq!(FeedbackKind::ResultGap.as_str(), "result_gap");
    assert_eq!(FeedbackKind::VerificationMissing.as_str(), "verification_missing");
    assert_eq!(FeedbackKind::Nonprogress.as_str(), "nonprogress");
    assert_eq!(PauseReason::NonprogressUncorrected.as_str(), "nonprogress_uncorrected");
    assert_eq!(PauseReason::EscalateRecommended.as_str(), "escalate_recommended");
    assert_eq!(PauseReason::VerificationUnconfirmed.as_str(), "verification_unconfirmed");
    assert_eq!(PauseReason::BudgetExhausted.as_str(), "budget_exhausted");
    assert_eq!(
        ControlBudgetKind::NonprogressCorrection.as_str(),
        "nonprogress_correction"
    );
    assert_eq!(ControlBudgetKind::RetryVeto.as_str(), "retry_veto");
    assert_eq!(
        ControlRefusal::ContinuationPending.as_str(),
        "continuation_pending"
    );
    assert_eq!(ControlRefusal::ExplicitStop.as_str(), "explicit_stop");
    assert_eq!(
        ControlRefusal::WrongBoundary.as_str(),
        "wrong_boundary"
    );
    let _ = FallbackReason::Unavailable.as_str();
}

#[test]
fn accounting_unavailable_refuses_every_effect() {
    let fact = facts(7, &["result_sufficiency.0"]);
    let insufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.95,
        7,
    );
    // Untrusted durable accounting (zero headroom) refuses effects BEFORE any
    // budget or boundary logic: no maxima are synthesized from it.
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::AgentEnd,
        &insufficient,
        &fact,
        &ControlBudgetSnapshot::zero_headroom(EPOCH),
    );
    assert_eq!(refused_reason(verdict), "accounting_unavailable");

    // The same refusal governs effects that never draw budget (pause).
    let stop = candidate(
        DecisionCategory::ContinueStopEscalate,
        "continue_stop_escalate.0",
        "stop",
        0.95,
        7,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::TurnEnd,
        &stop,
        &fact,
        &ControlBudgetSnapshot::zero_headroom(EPOCH),
    );
    assert_eq!(refused_reason(verdict), "accounting_unavailable");
}

#[test]
fn zero_headroom_and_fresh_maxima_shape() {
    let zero = ControlBudgetSnapshot::zero_headroom("sess:9");
    assert_eq!(zero.epoch_id, "sess:9");
    assert!(!zero.available);
    assert_eq!(zero.feedback_remaining, 0);
    assert_eq!(zero.verification_remaining, 0);
    assert_eq!(zero.nonprogress_remaining, 0);
    assert_eq!(zero.veto_remaining, 0);

    // Fresh maxima are the ONLY trusted construction path, and they carry the
    // contract maxima.
    let fresh = ControlBudgetSnapshot::fresh_maxima("sess:1");
    assert!(fresh.available);
    assert_eq!(fresh.feedback_remaining, CONTROL_MAX_FEEDBACK);
    assert_eq!(fresh.verification_remaining, CONTROL_MAX_VERIFICATION_REQUESTS);
    assert_eq!(fresh.nonprogress_remaining, CONTROL_MAX_NONPROGRESS_CORRECTIONS);
    assert_eq!(fresh.veto_remaining, CONTROL_MAX_RETRY_VETOES);
}

#[test]
fn correlation_refuses_epoch_mismatch() {
    // A decision captured under an older epoch can never spend the current
    // epoch's budget: facts epoch != budget epoch fails correlation.
    let fact = facts(7, &["result_sufficiency.0"]);
    let mut stale_facts = fact.clone();
    stale_facts.epoch_id = "sess-a:2".to_string();
    let insufficient = candidate(
        DecisionCategory::ResultSufficiency,
        "result_sufficiency.0",
        "insufficient",
        0.95,
        7,
    );
    let verdict = evaluate_control_answer(
        &ControlPolicy::default(),
        &features(),
        JevMode::Active,
        ControlBoundary::AgentEnd,
        &insufficient,
        &stale_facts,
        &budget(),
    );
    assert_eq!(refused_reason(verdict), "no_answer");
}
