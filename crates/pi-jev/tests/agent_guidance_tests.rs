//! Pure-core tests for the agent-guidance assessment module
//! (ROOT CONTRACT v7, AGENT-GUIDANCE lane).
//!
//! These tests run without a host and without a live SystemOne call. They pin
//! the truth rules: single-answer uncertainty only (no repeat probes), host
//! policy thresholds/weights (never model-settable), unmaskable critical
//! flags, deterministic no-hint reasons, and closed-set question shaping.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use pi_jev::agent_guidance::{CapturedSkillHintStamp, choice_abstention, compose_fixed_weighted, guardrail_battery_questions, guardrail_route, noul_band, skill_fit_question, skill_gate_question, skill_hint, skill_rank_question, verify_guidance_question, verify_guidance_request, verify_guidance_state, verify_rank_catalog, ChoiceAbstention, GuidanceRefusal, GuardrailInput, GuardrailPolicy, GuardrailRoute, GuidanceThresholds, NoHintReason, NoulBand, SkillAnswerSet, SkillCatalogEntry, SkillHintOutcome, TimingLabel, MAX_GUIDANCE_CATALOG, MAX_GUIDANCE_ID_CHARS, MAX_GUIDANCE_TEXT_CHARS, SkillHintStamp, skill_hint_is_current, guidance_stamp_hash};
use pi_jev::evaluators::PreparedQuestion;
use pi_jev::hooks::{ActiveSettings, JevObserver, JevObserverConfig};
use pi_jev::mock::{MockJevTransport, RecordedCall};
use pi_jev::snapshot::{fingerprint_of, bound_json, MAX_STATE_BYTES};
use pi_jev::types::estimate_request_tokens;
use pi_jev::types::{Answer, DecisionCategory, QuestionSpec, REQUEST_TOKEN_CEILING};
use pi_jev::{JevLimits, JevStats, JevSystemOne, SecretString, Transport};
use serde_json::json;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn thresholds() -> GuidanceThresholds {
    GuidanceThresholds {
        noul_low: 0.30,
        noul_high: 0.70,
        choice_top_min: 0.60,
        min_confidence: 0.60,
    }
}

/// Host stamp fixture: stable production facts for hint tests.
fn test_stamp() -> pi_jev::agent_guidance::SkillHintStamp {
    pi_jev::agent_guidance::SkillHintStamp {
        settings_revision: "rev-7".to_string(),
        mode: "compare_and_active".to_string(),
        feature_enabled: true,
        turn: 4,
        delivery_id: "delivery-1".to_string(),
        task_hash: guidance_stamp_hash(&["task excerpt A"]),
        catalog_hash: guidance_stamp_hash(&["alpha", "beta"]),
    }
}

fn noul(probability: f64) -> Answer {
    Answer::Noul { noul: probability }
}

fn choice(label: &str, probability: f64, confidence: f64) -> Answer {
    Answer::Choice {
        choice: label.to_string(),
        probabilities: BTreeMap::from([
            (label.to_string(), probability),
            ("none".to_string(), 1.0 - probability),
        ]),
        confidence,
    }
}

fn catalog(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|id| id.to_string()).collect()
}

fn entries(ids: &[&str]) -> Vec<SkillCatalogEntry> {
    ids.iter()
        .map(|id| SkillCatalogEntry {
            id: id.to_string(),
            description: format!("skill {id}"),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Threshold validation
// ---------------------------------------------------------------------------

#[test]
fn thresholds_accept_the_documented_shape() {
    assert!(thresholds().validate().is_ok());
}

#[test]
fn thresholds_reject_non_finite_and_out_of_range_values() {
    for bad in [
        GuidanceThresholds { noul_low: f64::NAN, ..thresholds() },
        GuidanceThresholds { noul_high: f64::INFINITY, ..thresholds() },
        GuidanceThresholds { choice_top_min: -0.1, ..thresholds() },
        GuidanceThresholds { min_confidence: 1.1, ..thresholds() },
        GuidanceThresholds { noul_low: 0.8, noul_high: 0.3, ..thresholds() },
    ] {
        assert!(bad.validate().is_err());
    }
}

// ---------------------------------------------------------------------------
// Single-answer band / abstention (no added model calls)
// ---------------------------------------------------------------------------

#[test]
fn band_boundaries_are_inclusive() {
    assert_eq!(noul_band(0.29, 0.30, 0.70), Some(NoulBand::Below));
    assert_eq!(noul_band(0.30, 0.30, 0.70), Some(NoulBand::Within));
    assert_eq!(noul_band(0.50, 0.30, 0.70), Some(NoulBand::Within));
    assert_eq!(noul_band(0.70, 0.30, 0.70), Some(NoulBand::Within));
    assert_eq!(noul_band(0.71, 0.30, 0.70), Some(NoulBand::Above));
}

#[test]
fn band_rejects_invalid_probability_or_band() {
    assert_eq!(noul_band(f64::NAN, 0.30, 0.70), None);
    assert_eq!(noul_band(1.1, 0.30, 0.70), None);
    assert_eq!(noul_band(0.5, 0.8, 0.2), None);
    assert_eq!(noul_band(0.5, f64::NAN, 0.70), None);
}

#[test]
fn choice_abstention_acts_at_exactly_the_threshold() {
    assert_eq!(choice_abstention(0.60, 0.60), Some(ChoiceAbstention::Select));
    assert_eq!(choice_abstention(0.59, 0.60), Some(ChoiceAbstention::Uncertain));
    assert_eq!(choice_abstention(f64::NAN, 0.60), None);
    assert_eq!(choice_abstention(0.5, 1.1), None);
}

// ---------------------------------------------------------------------------
// Fixed-weight composition (host weights only)
// ---------------------------------------------------------------------------

#[test]
fn composition_averages_with_host_weights() {
    let weights = BTreeMap::from([
        ("a".to_string(), 1.0),
        ("b".to_string(), 3.0),
    ]);
    let items = vec![
        ("a".to_string(), 0.8),
        ("b".to_string(), 0.4),
    ];
    let composed = compose_fixed_weighted(&items, &weights).expect("composed");
    assert!((composed.composed - (0.8 * 1.0 + 0.4 * 3.0) / 4.0).abs() < 1e-9);
    assert_eq!(composed.contributions.len(), 2);
}

#[test]
fn composition_requires_full_finite_non_negative_coverage() {
    let weights = BTreeMap::from([("a".to_string(), 1.0)]);
    // A hazard without a weight is a policy error, never a silent skip.
    assert!(compose_fixed_weighted(&[("a".to_string(), 0.5), ("b".to_string(), 0.5)], &weights).is_none());
    // Negative or non-finite weights are policy errors.
    let negative = BTreeMap::from([("a".to_string(), -1.0)]);
    assert!(compose_fixed_weighted(&[("a".to_string(), 0.5)], &negative).is_none());
    let nan = BTreeMap::from([("a".to_string(), f64::NAN)]);
    assert!(compose_fixed_weighted(&[("a".to_string(), 0.5)], &nan).is_none());
    // Zero weights are legal only while the sum stays positive.
    let zero = BTreeMap::from([("a".to_string(), 0.0), ("b".to_string(), 1.0)]);
    let composed = compose_fixed_weighted(&[("a".to_string(), 0.5), ("b".to_string(), 0.5)], &zero).expect("zero weight ok");
    assert!((composed.composed - 0.5).abs() < 1e-9);
    let all_zero = BTreeMap::from([("a".to_string(), 0.0)]);
    assert!(compose_fixed_weighted(&[("a".to_string(), 0.5)], &all_zero).is_none());
    // Probabilities stay in unit range.
    assert!(compose_fixed_weighted(&[("a".to_string(), 1.2)], &BTreeMap::from([("a".to_string(), 1.0)])).is_none());
}

// ---------------------------------------------------------------------------
// Skill hint derivation
// ---------------------------------------------------------------------------

struct SkillFixture {
    rank: Answer,
    need: Answer,
    inverse: Answer,
    act: Answer,
    fit: Option<Answer>,
    catalog: Vec<String>,
    fresh: bool,
    cancelled: bool,
    budget_ok: bool,
}

fn good_skill_fixture() -> SkillFixture {
    SkillFixture {
        rank: choice("alpha", 0.9, 0.8),
        need: noul(0.9),
        inverse: noul(0.1),
        act: noul(0.8),
        fit: None,
        catalog: catalog(&["alpha", "beta", "none"]),
        fresh: true,
        cancelled: false,
        budget_ok: true,
    }
}

fn hint_outcome(fixture: &SkillFixture, fit: Option<&Answer>) -> SkillHintOutcome {
    skill_hint(
        &SkillAnswerSet {
            rank: Some(&fixture.rank),
            need_gate: Some(&fixture.need),
            inverse_gate: Some(&fixture.inverse),
            act_gate: Some(&fixture.act),
            fit,
            catalog_ids: &fixture.catalog,
            fresh: fixture.fresh,
            cancelled: fixture.cancelled,
            budget_ok: fixture.budget_ok,
            stamp: CapturedSkillHintStamp::capture(test_stamp()),
        },
        &thresholds(),
    )
}

#[test]
fn lifecycle_failures_name_their_reason_and_never_hint() {
    let mut fixture = good_skill_fixture();
    fixture.cancelled = true;
    assert_eq!(hint_outcome(&fixture, None), SkillHintOutcome::NoHint(NoHintReason::Cancelled));

    let mut fixture = good_skill_fixture();
    fixture.fresh = false;
    assert_eq!(hint_outcome(&fixture, None), SkillHintOutcome::NoHint(NoHintReason::Stale));

    let mut fixture = good_skill_fixture();
    fixture.budget_ok = false;
    assert_eq!(
        hint_outcome(&fixture, None),
        SkillHintOutcome::NoHint(NoHintReason::BudgetExhausted)
    );
}

#[test]
fn empty_catalog_is_catalog_absent_not_a_claim() {
    let fixture = good_skill_fixture();
    let mut empty = fixture;
    empty.catalog = Vec::new();
    assert_eq!(hint_outcome(&empty, None), SkillHintOutcome::NoHint(NoHintReason::CatalogAbsent));
}

#[test]
fn invalid_host_policy_yields_no_hint_without_a_claim() {
    let fixture = good_skill_fixture();
    let invalid = GuidanceThresholds {
        noul_low: 0.9,
        noul_high: 0.3,
        ..thresholds()
    };
    let outcome = skill_hint(
        &SkillAnswerSet {
            rank: Some(&fixture.rank),
            need_gate: Some(&fixture.need),
            inverse_gate: Some(&fixture.inverse),
            act_gate: Some(&fixture.act),
            fit: None,
            catalog_ids: &fixture.catalog,
            fresh: true,
            cancelled: false,
            budget_ok: true,
            stamp: CapturedSkillHintStamp::capture(test_stamp()),
        },
        &invalid,
    );
    // An invalid host policy is an unavailable assessment: no hint, no claim.
    assert_eq!(outcome, SkillHintOutcome::NoHint(NoHintReason::BudgetExhausted));
}

#[test]
fn rank_below_threshold_is_uncertain_not_absent() {
    let mut fixture = good_skill_fixture();
    fixture.rank = choice("alpha", 0.55, 0.9);
    assert_eq!(hint_outcome(&fixture, None), SkillHintOutcome::NoHint(NoHintReason::Uncertain));
}

#[test]
fn weak_confidence_or_bad_rank_is_no_confident_candidate() {
    let mut fixture = good_skill_fixture();
    fixture.rank = choice("alpha", 0.9, 0.5);
    assert_eq!(
        hint_outcome(&fixture, None),
        SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate)
    );

    let mut fixture = good_skill_fixture();
    fixture.rank = Answer::Noul { noul: 0.9 }; // wrong type for the rank question
    assert_eq!(
        hint_outcome(&fixture, None),
        SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate)
    );
}

#[test]
fn ranked_id_outside_the_current_catalog_is_stale() {
    let mut fixture = good_skill_fixture();
    fixture.catalog = catalog(&["beta", "none"]);
    assert_eq!(hint_outcome(&fixture, None), SkillHintOutcome::NoHint(NoHintReason::Stale));
}

#[test]
fn gate_bands_are_uncertain_and_wrong_sides_are_unconfident() {
    for (need, inverse, act, expected) in [
        (0.9, 0.1, 0.8, None),                 // decisive hint
        (0.5, 0.1, 0.8, Some(NoHintReason::Uncertain)), // need inside band
        (0.9, 0.5, 0.8, Some(NoHintReason::Uncertain)), // inverse inside band
        (0.9, 0.1, 0.5, Some(NoHintReason::Uncertain)), // act inside band
        (0.2, 0.1, 0.8, Some(NoHintReason::NoConfidentCandidate)), // need decisive-low
        (0.9, 0.9, 0.8, Some(NoHintReason::NoConfidentCandidate)), // inverse decisive-high
        (0.9, 0.1, 0.2, Some(NoHintReason::NoConfidentCandidate)), // act decisive-low
    ] {
        let mut fixture = good_skill_fixture();
        fixture.need = noul(need);
        fixture.inverse = noul(inverse);
        fixture.act = noul(act);
        let outcome = hint_outcome(&fixture, None);
        match expected {
            None => assert!(matches!(outcome, SkillHintOutcome::Hint(_)), "{need}/{inverse}/{act}"),
            Some(reason) => assert_eq!(outcome, SkillHintOutcome::NoHint(reason)),
        }
    }
}

#[test]
fn missing_or_invalid_answers_are_no_confident_candidate() {
    let fixture = good_skill_fixture();
    // A missing gate answer is an incomplete assessment, not a confident one.
    let outcome = skill_hint(
        &SkillAnswerSet {
            rank: Some(&fixture.rank),
            need_gate: None,
            inverse_gate: Some(&fixture.inverse),
            act_gate: Some(&fixture.act),
            fit: None,
            catalog_ids: &fixture.catalog,
            fresh: true,
            cancelled: false,
            budget_ok: true,
            stamp: CapturedSkillHintStamp::capture(test_stamp()),
        },
        &thresholds(),
    );
    assert_eq!(outcome, SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate));

    let mut fixture = good_skill_fixture();
    fixture.need = noul(f64::NAN);
    assert_eq!(
        hint_outcome(&fixture, None),
        SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate)
    );
}

#[test]
fn fit_gates_the_second_pass() {
    let fixture = good_skill_fixture();
    assert!(matches!(hint_outcome(&fixture, Some(&noul(0.9))), SkillHintOutcome::Hint(_)));
    assert_eq!(hint_outcome(&fixture, Some(&noul(0.5))), SkillHintOutcome::NoHint(NoHintReason::Uncertain));
    assert_eq!(
        hint_outcome(&fixture, Some(&noul(0.1))),
        SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate)
    );
}

#[test]
fn a_hint_is_advisory_non_exclusive_and_host_worded() {
    let fixture = good_skill_fixture();
    match hint_outcome(&fixture, None) {
        SkillHintOutcome::Hint(hint) => {
            assert_eq!(hint.skill_id, "alpha");
            assert_eq!(hint.label, "advisory, non-exclusive");
            assert!(hint.why.chars().count() <= 120);
            // Host-derived rationale only: fixed labels plus returned numbers.
            assert!(hint.why.starts_with("rank p="));
            assert!(!hint.why.contains("secret"));
        }
        other => panic!("expected a hint, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Guardrail route derivation (deterministic; never enforcement)
// ---------------------------------------------------------------------------

fn policy() -> GuardrailPolicy {
    let mut weights = BTreeMap::new();
    for (id, weight) in [
        ("hidden", 0.30),
        ("exfil", 0.30),
        ("destructive", 0.25),
        ("safety", 0.15),
    ] {
        weights.insert(id.to_string(), weight);
    }
    GuardrailPolicy {
        thresholds: thresholds(),
        severity_review: 0.40,
        severity_block: 0.75,
        weights,
        critical_ids: &["exfil", "destructive"],
        support_ids: &["safety"],
        support_enabled: true,
    }
}

fn guardrail_input(hazards: Vec<(String, f64)>, severity: Option<f64>) -> GuardrailInput {
    GuardrailInput {
        hazards,
        severity,
        fresh: true,
        cancelled: false,
        budget_ok: true,
    }
}

#[test]
fn guardrail_route_is_none_when_the_run_did_not_happen() {
    let mut input = guardrail_input(vec![], None);
    input.cancelled = true;
    assert!(guardrail_route(&input, &policy(), TimingLabel::InputScreenPreventive).is_none());
    let mut input = guardrail_input(vec![], None);
    input.fresh = false;
    assert!(guardrail_route(&input, &policy(), TimingLabel::InputScreenPreventive).is_none());
    let mut input = guardrail_input(vec![], None);
    input.budget_ok = false;
    assert!(guardrail_route(&input, &policy(), TimingLabel::InputScreenPreventive).is_none());
}

#[test]
fn guardrail_route_escalates_deterministically() {
    let quiet = guardrail_input(
        vec![
            ("hidden".to_string(), 0.05),
            ("exfil".to_string(), 0.05),
            ("destructive".to_string(), 0.05),
            ("safety".to_string(), 0.05),
        ],
        Some(0.1),
    );
    let assessment = guardrail_route(&quiet, &policy(), TimingLabel::InputScreenPreventive).expect("assessment");
    assert_eq!(assessment.route, GuardrailRoute::Pass);
    assert!(assessment.enforced == false, "enforced is structurally false");
    assert_eq!(assessment.timing, TimingLabel::InputScreenPreventive);

    // The composed fixed-weight value alone crosses review: with the host
    // weights (0.30/0.30/0.25/0.15) it is 0.44 >= severity_review 0.40. Both
    // decisive hazards (hidden, safety) are NON-critical, so the composition
    // itself escalates — the critical mask-proof rule is not involved.
    let review = guardrail_input(
        vec![
            ("hidden".to_string(), 0.9),
            ("exfil".to_string(), 0.05),
            ("destructive".to_string(), 0.05),
            ("safety".to_string(), 0.95),
        ],
        Some(0.1),
    );
    assert_eq!(
        guardrail_route(&review, &policy(), TimingLabel::InputScreenPreventive).expect("assessment").route,
        GuardrailRoute::Review
    );

    // Composed crosses block: 0.9/0.9/0.9/0.1 with the host weights -> 0.78
    // >= severity_block 0.75.
    let block = guardrail_input(
        vec![
            ("hidden".to_string(), 0.9),
            ("exfil".to_string(), 0.9),
            ("destructive".to_string(), 0.9),
            ("safety".to_string(), 0.1),
        ],
        Some(0.9),
    );
    assert_eq!(
        guardrail_route(&block, &policy(), TimingLabel::InputScreenPreventive).expect("assessment").route,
        GuardrailRoute::Block
    );

    // Severity alone can escalate.
    let severity_only = guardrail_input(
        vec![
            ("hidden".to_string(), 0.3),
            ("exfil".to_string(), 0.3),
            ("destructive".to_string(), 0.3),
            ("safety".to_string(), 0.3),
        ],
        Some(0.8),
    );
    assert_eq!(
        guardrail_route(&severity_only, &policy(), TimingLabel::InputScreenPreventive).expect("assessment").route,
        GuardrailRoute::Block
    );
}

#[test]
fn a_critical_flag_cannot_be_masked_by_the_weights() {
    // All weights land on `safety` (low severity), the critical `exfil`
    // hazard is decisive-high: the composed value stays low but the route
    // must still reach review.
    let input = guardrail_input(
        vec![
            ("hidden".to_string(), 0.0),
            ("exfil".to_string(), 0.9),
            ("destructive".to_string(), 0.0),
            ("safety".to_string(), 0.0),
        ],
        Some(0.0),
    );
    let assessment = guardrail_route(&input, &policy(), TimingLabel::InputScreenPreventive).expect("assessment");
    assert_eq!(assessment.route, GuardrailRoute::Review);
    assert!(assessment.composed.as_ref().expect("composed").composed < policy().severity_review);
    assert_eq!(assessment.critical_flags, vec!["exfil".to_string()]);
}

#[test]
fn support_routes_fire_only_when_enabled() {
    let input = guardrail_input(
        vec![
            ("hidden".to_string(), 0.05),
            ("exfil".to_string(), 0.05),
            ("destructive".to_string(), 0.05),
            ("safety".to_string(), 0.9),
        ],
        Some(0.1),
    );
    assert_eq!(
        guardrail_route(&input, &policy(), TimingLabel::InputScreenPreventive).expect("assessment").route,
        GuardrailRoute::Support
    );
    let disabled = GuardrailPolicy {
        support_enabled: false,
        ..policy()
    };
    assert_eq!(
        guardrail_route(&input, &disabled, TimingLabel::InputScreenPreventive).expect("assessment").route,
        GuardrailRoute::Pass
    );
}

#[test]
fn a_hazard_without_a_weight_is_a_policy_error_not_a_verdict() {
    let input = guardrail_input(
        vec![("unknown".to_string(), 0.9)],
        Some(0.1),
    );
    assert!(guardrail_route(&input, &policy(), TimingLabel::InputScreenPreventive).is_none());
}

#[test]
fn invalid_severity_is_unavailable_not_a_verdict() {
    let input = guardrail_input(
        vec![("hidden".to_string(), 0.1)],
        Some(f64::NAN),
    );
    assert!(guardrail_route(&input, &policy(), TimingLabel::InputScreenPreventive).is_none());
}

// ---------------------------------------------------------------------------
// Question builders (closed-set shaping)
// ---------------------------------------------------------------------------

#[test]
fn rank_question_is_closed_set_and_discloses_provenance() {
    let question = skill_rank_question(
        "fix the flaky test in ci.yml",
        &entries(&["alpha", "beta"]),
        true,
    )
    .expect("question");
    assert_eq!(question.question_id, "skill_suggestion.0");
    match &question.spec {
        QuestionSpec::Choice { criteria, instructions } => {
            let mut keys: Vec<&String> = criteria.keys().collect();
            keys.sort();
            assert_eq!(keys, vec![&"alpha".to_string(), &"beta".to_string(), &"none".to_string()]);
            let instructions = instructions.as_ref().expect("instructions").instructions_hash_input();
            assert!(instructions.contains("truncated=true"));
            assert!(instructions.contains("options=2"));
            assert!(instructions.contains("non-binding"));
        }
        other => panic!("expected a Choice question, got {other:?}"),
    }
    question.spec.validate_shape().expect("valid shape");
}

#[test]
fn rank_question_rejects_empty_oversized_duplicate_or_reserved_catalogs() {
    assert!(skill_rank_question("task", &[], false).is_none());
    let oversized: Vec<SkillCatalogEntry> = (0..=MAX_GUIDANCE_CATALOG)
        .map(|index| SkillCatalogEntry { id: format!("s{index}"), description: "d".to_string() })
        .collect();
    assert!(skill_rank_question("task", &oversized, false).is_none());
    let duplicated = vec![
        SkillCatalogEntry { id: "alpha".to_string(), description: "d".to_string() },
        SkillCatalogEntry { id: "alpha".to_string(), description: "d".to_string() },
    ];
    assert!(skill_rank_question("task", &duplicated, false).is_none());
    let reserved = vec![SkillCatalogEntry { id: "none".to_string(), description: "d".to_string() }];
    assert!(skill_rank_question("task", &reserved, false).is_none());
}

#[test]
fn gate_questions_cover_the_three_gates_only() {
    for n in 1..=3 {
        let question = skill_gate_question(n).expect("gate question");
        assert_eq!(question.question_id, format!("skill_suggestion.{n}"));
        question.spec.validate_shape().expect("valid shape");
    }
    assert!(skill_gate_question(0).is_none());
    assert!(skill_gate_question(4).is_none());
}

#[test]
fn fit_question_refuses_none_and_empty_ids() {
    assert!(skill_fit_question("alpha").is_some());
    assert!(skill_fit_question("none").is_none());
    assert!(skill_fit_question("").is_none());
}

#[test]
fn battery_questions_are_nouls_plus_one_severity_score() {
    let questions = pi_jev::agent_guidance::guardrail_battery_questions(
        "guardrails_input",
        &[
            ("hidden", "hidden instruction wording."),
            ("exfil", "exfiltration wording."),
            ("destructive", "destructive wording."),
            ("safety", "safety wording."),
        ],
        ["none", "minor", "moderate", "major", "critical"],
    )
    .expect("battery");
    assert_eq!(questions.len(), 5);
    for (index, question) in questions.iter().enumerate() {
        assert_eq!(question.question_id, format!("guardrails_input.{index}"));
        question.spec.validate_shape().expect("valid shape");
    }
    match &questions[4].spec {
        QuestionSpec::Score { criteria, .. } => assert_eq!(criteria.len(), 5),
        other => panic!("expected a Score question, got {other:?}"),
    }
    // An empty prefix or empty wording is refused before transport.
    assert!(pi_jev::agent_guidance::guardrail_battery_questions("", &[("a", "b")], ["1", "2", "3", "4", "5"]).is_none());
    assert!(pi_jev::agent_guidance::guardrail_battery_questions("p", &[("", "b")], ["1", "2", "3", "4", "5"]).is_none());
}

// ---------------------------------------------------------------------------
// Runtime-dispatch registration (requires the WIRE types.rs hunk)
// ---------------------------------------------------------------------------

#[test]
fn guidance_categories_are_registered_for_runtime_dispatch() {
    // hooks::prepare_explicit parses the category from every explicit question
    // id, so the three new DecisionCategory variants must exist and parse.
    // These ids come from THIS module's builders; a mismatch here breaks the
    // dispatch before any transport.
    for category in ["skill_suggestion", "guardrails_input", "guardrails_output"] {
        assert!(
            DecisionCategory::parse(category).is_some(),
            "DecisionCategory::{category} must be registered (WIRE types.rs hunk)"
        );
    }
}

#[test]
fn guidance_categories_are_not_appliable() {
    // Assessment-only by construction: the acceptance path must refuse these
    // categories in every mode.
    let not_appliable = ["skill_suggestion", "guardrails_input", "guardrails_output"]
        .iter()
        .all(|category| {
            DecisionCategory::parse(category).map_or(true, |parsed| {
                !pi_jev::active::DEFAULT_APPLIABLE_CATEGORIES.contains(&parsed)
                    && !pi_jev::active::OPTIONAL_APPLIABLE_CATEGORIES.contains(&parsed)
            })
        });
    assert!(not_appliable);
}

#[test]
fn text_bounds_stay_small() {
    let long = "x".repeat(MAX_GUIDANCE_TEXT_CHARS + 100);
    let bounded = pi_jev::agent_guidance::bound_text(&long, MAX_GUIDANCE_TEXT_CHARS);
    assert!(bounded.chars().count() <= MAX_GUIDANCE_TEXT_CHARS + 3);
    assert!(bounded.ends_with("..."));
}

// ---------------------------------------------------------------------------
// Outgoing-payload verification against the ACTUAL native rules (v7)
// ---------------------------------------------------------------------------
//
// Builder completeness is not enough: the native observer bounds state
// (snapshot::bound_json: strings >400 chars, arrays/maps >32 entries, keys
// >64 chars) and silently drops requests whose state exceeds 8KiB. These
// tests capture ACTUAL native mock-transport requests and prove every
// selectable id carries its evidence post-bound; anything unprovable is
// refused (fail-open), never fixed by raising caps.


struct CaptureFixture {
    observer: Arc<JevObserver>,
    transport: Arc<MockJevTransport>,
    dir: tempfile::TempDir,
}

impl CaptureFixture {
    fn new(mode: pi_jev::config::JevMode) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let transport = Arc::new(MockJevTransport::all_valid());
        let stats = Arc::new(JevStats::default());
        let client = JevSystemOne::new(
            mode,
            SecretString::new("synthetic-key"),
            transport.clone(),
            JevLimits {
                max_retries: 0,
                ..Default::default()
            },
            stats.clone(),
        )
        .expect("client");
        let gate_mode = Arc::new(std::sync::Mutex::new(mode));
        let gate = gate_mode.clone();
        let observer = JevObserver::new(
            JevObserverConfig {
                mode_gate: Arc::new(move |_| {
                    (*gate.lock().unwrap(), pi_jev::hooks::SYSTEM_ONE_MODEL.to_string())
                }),
                ..Default::default()
            },
            Arc::new(client),
            dir.path().join("records.jsonl"),
        );
        Self {
            observer,
            transport,
            dir,
        }
    }
}

impl Drop for CaptureFixture {
    fn drop(&mut self) {
        self.observer.shutdown();
    }
}

async fn wait_for_calls(transport: &MockJevTransport, expected: usize) -> Vec<RecordedCall> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if transport.call_count() >= expected {
            return transport.calls();
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "expected {expected} recorded transport call(s), got {}",
            transport.call_count()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn guidance_catalog(size: usize) -> Vec<SkillCatalogEntry> {
    (0..size)
        .map(|index| SkillCatalogEntry {
            id: format!("skill{index:02}"),
            description: format!("documented procedure for skill{index:02}"),
        })
        .collect()
}

fn full_guidance_request(
    catalog_size: usize,
) -> (serde_json::Value, Vec<PreparedQuestion>, Vec<String>) {
    let catalog = guidance_catalog(catalog_size);
    let task = "fix the flaky ci test";
    let rank = skill_rank_question(task, &catalog, false).expect("rank question");
    let questions = vec![
        rank,
        skill_gate_question(1).expect("gate 1"),
        skill_gate_question(2).expect("gate 2"),
        skill_gate_question(3).expect("gate 3"),
    ];
    let state = json!({
        "user_text_excerpt": task,
        "skill_catalog_count": catalog.len(),
        "skill_catalog_truncated": false,
        "assessment": "skill_suggestion",
        "hint_authority": "none",
    });
    let ids: Vec<String> = catalog.iter().map(|entry| entry.id.clone()).collect();
    (state, questions, ids)
}

#[tokio::test]
async fn observe_prepared_reaches_the_transport_complete_and_unclipped() {
    // A FULL bounded catalog (31 skills + none) must survive end to end:
    // verifier proof first, then the ACTUAL native observer dispatch.
    let (state, questions, ids) = full_guidance_request(MAX_GUIDANCE_CATALOG);
    verify_guidance_request(&state, &questions, Some(("skill_suggestion.0", &ids)))
        .expect("request must be provably visible and in-bounds");

    let state_for_fingerprint = state.clone();
    let fixture = CaptureFixture::new(pi_jev::config::JevMode::Compare);
    // The handle under test is the fixture's OWN transport — the exact
    // instance the client writes to; a detached Arc would assert nothing.
    let transport = fixture.transport.clone();
    fixture
        .observer
        .observe_prepared(&json!({
            "session_id": "guidance-capture",
            "turn": 4,
            "state": state.clone(),
        }), "skill_suggestion", questions.clone());

    let calls = wait_for_calls(&transport, 1).await;
    assert_eq!(calls.len(), 1, "exactly one outgoing request, not dropped");
    let recorded = &calls[0];
    // Every guidance question id reached the transport: nothing silently
    // dropped by the category gate, the question cap, or duplicate ids.
    let mut expected_ids: Vec<String> = questions
        .iter()
        .map(|question| question.question_id.clone())
        .collect();
    expected_ids.sort();
    let mut recorded_ids = recorded.question_ids.clone();
    recorded_ids.sort();
    assert_eq!(recorded_ids, expected_ids);
    // The outgoing state is byte-identical to what the builder produced:
    // the native fingerprint of the recorded request equals the builder's.
    assert_eq!(recorded.state_fingerprint, fingerprint_of(&state));
    // Dynamic size proofs (no fixture-derived magic budget):
    assert!(serde_json::to_vec(&state).unwrap().len() <= MAX_STATE_BYTES);
    let request = pi_jev::types::SystemOneRequest {
        state: state.clone(),
        model: pi_jev::hooks::SYSTEM_ONE_MODEL.to_string(),
        questions: questions
            .iter()
            .map(|question| (question.question_id.clone(), question.spec.clone()))
            .collect(),
    };
    assert!(estimate_request_tokens(&request) <= REQUEST_TOKEN_CEILING);
    // The native defensive bounding is a verified NO-OP on this payload:
    // nothing can be silently clipped by any defensive bounding on the way out.
    assert_eq!(bound_json(state.clone(), 0), state);
}

#[tokio::test]
async fn decide_prepared_carries_every_guidance_answer_and_never_applies() {
    let (state, questions, _ids) = full_guidance_request(MAX_GUIDANCE_CATALOG);
    let state_for_fingerprint = state.clone();
    let fixture = CaptureFixture::new(pi_jev::config::JevMode::CompareAndActive);
    // Same-instance guarantee: assert against the transport the client uses.
    let transport = fixture.transport.clone();
    let outcome = fixture
        .observer
        .decide_prepared(
            &json!({
                "session_id": "guidance-active",
                "turn": 2,
                "state": state,
            }),
            "skill_suggestion",
            questions,
            &pi_jev::active::ActivationPolicy::default(),
        )
        .await;
    // The mock answers EVERY outgoing question: one record per guidance id.
    let raw = outcome.raw.as_ref().expect("raw decision outcome");
    let mut answered: Vec<String> = raw.records.iter().map(|r| r.question_id.clone()).collect();
    let mut expected: Vec<String> = vec![
        "skill_suggestion.0".into(),
        "skill_suggestion.1".into(),
        "skill_suggestion.2".into(),
        "skill_suggestion.3".into(),
    ];
    expected.sort();
    answered.sort();
    assert_eq!(answered, expected, "every asked question was answered");
    // Assessment-only by construction through the REAL acceptance path:
    // nothing is applied, every guidance answer is refused as not appliable.
    assert!(outcome.decisions.is_empty(), "no guidance answer may apply");
    assert_eq!(outcome.refusals.len(), raw.records.len());
    for refusal in &outcome.refusals {
        assert_eq!(refusal.reason.as_str(), "category_not_appliable");
    }
    // The recorded outgoing request again carries the exact builder state.
    assert_eq!(transport.calls().len(), 1);
    assert_eq!(transport.calls()[0].state_fingerprint, fingerprint_of(&state_for_fingerprint));
}

#[tokio::test]
async fn oversize_state_is_refused_by_the_verifier_and_dropped_by_the_native_gate() {
    let oversized_state = json!({
        "user_text_excerpt": "x".repeat(MAX_STATE_BYTES),
        "assessment": "skill_suggestion",
    });
    // The verifier refuses BEFORE transport (fail-open).
    assert_eq!(
        verify_guidance_state(&oversized_state),
        Err(GuidanceRefusal::StateTooLarge)
    );
    // The native explicit gate would silently drop the same request: proof
    // that refusing early is the only truthful behavior.
    let fixture = CaptureFixture::new(pi_jev::config::JevMode::Compare);
    // The handle under test is the fixture's OWN transport — the exact
    // instance the client writes to; a detached Arc would assert nothing.
    let transport = fixture.transport.clone();
    fixture.observer.observe_prepared(
        &json!({
            "session_id": "guidance-oversize",
            "turn": 1,
            "state": oversized_state,
        }),
        "skill_suggestion",
        vec![skill_gate_question(1).expect("gate question")],
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(transport.call_count(), 0, "the native gate silently drops oversize state");
}

#[test]
fn verify_refuses_any_evidence_bound_json_would_clip() {
    // A 401-char state string would be clipped by the native bounding.
    let clipping_state = json!({ "excerpt": "y".repeat(401) });
    assert!(matches!(
        verify_guidance_state(&clipping_state),
        Err(GuidanceRefusal::BoundWouldClip(_))
    ));
    // A 400-char string is exactly at the native bound: no clip.
    assert!(verify_guidance_state(&json!({ "excerpt": "y".repeat(400) })).is_ok());
    // An unbounded choice (33 options) would be clipped by the native
    // 32-entry width bound: refused, never asked with missing evidence.
    let options: Vec<(String, Option<String>)> = (0..33)
        .map(|index| (format!("s{index:02}"), Some("description".to_string())))
        .collect();
    let wide = PreparedQuestion {
        question_id: "guardrails_input.0".into(),
        spec: QuestionSpec::choice("screen", options),
    };
    assert!(matches!(
        verify_guidance_question(&wide),
        Err(GuidanceRefusal::BoundWouldClip(_))
    ));
    // A key above the native 64-char key bound is refused.
    let long_key = PreparedQuestion {
        question_id: "guardrails_input.0".into(),
        spec: QuestionSpec::choice(
            "screen",
            vec![("z".repeat(65), Some("description".to_string()))],
        ),
    };
    assert!(matches!(
        verify_guidance_question(&long_key),
        Err(GuidanceRefusal::BoundWouldClip(_))
    ));
    // A rank option without its description is a closed-set mismatch.
    let (state, mut questions, ids) = full_guidance_request(2);
    let _ = state;
    if let QuestionSpec::Choice { criteria, .. } = &mut questions[0].spec {
        criteria.remove(&ids[0]);
    }
    let rank_id = "skill_suggestion.0";
    assert!(matches!(
        verify_rank_catalog(&questions[0], &ids),
        Err(GuidanceRefusal::CatalogClosedSetMismatch(_))
    ));
    // An id above the native key cap is refused by the builder itself.
    let catalog = vec![SkillCatalogEntry {
        id: "k".repeat(MAX_GUIDANCE_ID_CHARS + 1),
        description: "description".to_string(),
    }];
    assert!(skill_rank_question("task", &catalog, false).is_none());
}

#[test]
fn verify_refuses_a_rank_set_that_does_not_match_the_catalog() {
    let catalog = guidance_catalog(3);
    let rank = skill_rank_question("task", &catalog, false).expect("rank");
    // Exact match (plus `none`): visible.
    assert!(verify_rank_catalog(&rank, &catalog.iter().map(|e| e.id.clone()).collect::<Vec<_>>()).is_ok());
    // One extra intended id that the question does not ask: mismatch.
    let mut ids: Vec<String> = catalog.iter().map(|e| e.id.clone()).collect();
    ids.push("ghost".to_string());
    assert!(matches!(
        verify_rank_catalog(&rank, &ids),
        Err(GuidanceRefusal::CatalogClosedSetMismatch(_))
    ));
}

#[test]
fn battery_questions_survive_the_outgoing_bounds() {
    let questions = guardrail_battery_questions(
        "guardrails_output",
        &[
            ("instruction_echo", "echo wording."),
            ("secret_disclosure", "disclosure wording."),
            ("destructive_rendering", "rendering wording."),
            ("harmful_content", "harmful wording."),
        ],
        ["none", "minor", "moderate", "major", "critical"],
    )
    .expect("battery");
    let state = json!({
        "screened_excerpt": "bounded excerpt",
        "screening": "assessment_only",
        "enforced": false,
        "timing": "output_post_hoc_assessment",
    });
    verify_guidance_request(&state, &questions, None).expect("battery must be provably visible");
    for question in &questions {
        verify_guidance_question(question).expect("question visibility");
    }
}

#[test]
fn token_ceiling_and_body_bounds_are_checked_dynamically() {
    // No fixture-derived magic overhead: the request is measured against the
    // native constants (estimated tokens and serialized body bytes).
    let (state, questions, ids) = full_guidance_request(MAX_GUIDANCE_CATALOG);
    let mut specs = std::collections::BTreeMap::new();
    for question in &questions {
        specs.insert(question.question_id.clone(), question.spec.clone());
    }
    let request = pi_jev::types::SystemOneRequest {
        state,
        model: pi_jev::hooks::SYSTEM_ONE_MODEL.to_string(),
        questions: specs,
    };
    assert!(estimate_request_tokens(&request) <= REQUEST_TOKEN_CEILING);
    assert!(
        serde_json::to_vec(&request).unwrap().len() <= pi_jev::client::DEFAULT_MAX_PAYLOAD_BYTES,
        "serialized body fits the transport payload cap"
    );
    let _ = ids;
}
