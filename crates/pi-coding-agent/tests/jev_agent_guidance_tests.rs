//! Adapter tests for the agent-guidance lane (ROOT CONTRACT v7).
//!
//! Pure, offline tests over the REAL `Skill` roster shape and the pure
//! pi-jev guidance core. No session, provider, filesystem, network or Cargo
//! execution happens here. These tests prove the adapter contract the
//! integrator's dispatch hunks rely on:
//!
//! * The suggestion pool is parity with the prompt roster
//!   (`format_skills_for_prompt` visibility filter).
//! * Questions are closed-set, bounded, shape-valid; provenance disclosed.
//! * Consumption is assessment-only: `enforced=false`, truthful timing labels,
//!   advisory non-exclusive hints, deterministic no-hint reasons.
//! * The hint renders as a separate appended block; the roster text itself is
//!   never spliced, so prompt prefix caching stays intact.

use std::collections::BTreeMap;

use pi_coding_agent::core::jev_agent_guidance::{
    format_guardrail_advisory, format_skill_hint, guardrail_assessment_from_raw, guardrail_policy, prepare_guardrail_battery, prepare_skill_suggestion, skill_catalog_ids, skill_fit_prepared, skill_guidance_thresholds, skill_hint_from_raw, skill_roster_view, PreparedGuidance, GUARDRAIL_INPUT_HAZARDS, GUARDRAIL_OUTPUT_HAZARDS, GUARDRAIL_SEVERITY_LEVELS, skill_hint_for_render, skill_hint_stamp,
};
use pi_coding_agent::core::skills::{
    format_skills_for_prompt, BaseSkill, MarkdownSkill, PythonSkill, Skill, SkillKind,
    SkillPythonMetadata,
};
use pi_coding_agent::core::source_info::{
    create_synthetic_source_info, SyntheticSourceInfoOptions,
};
use pi_jev::agent_guidance::{CapturedSkillHintStamp, NoHintReason, SkillHintOutcome, TimingLabel, MAX_GUIDANCE_CATALOG, SkillHintStamp, skill_hint_is_current, guidance_stamp_hash};
use pi_jev::types::{Answer, DecisionCategory, DecisionRecord, QuestionSpec};

// ---------------------------------------------------------------------------
// Real roster fixtures (same construction the system-prompt tests use)
// ---------------------------------------------------------------------------

fn markdown_skill(name: &str, disable_model_invocation: bool) -> Skill {
    Skill::Markdown(MarkdownSkill {
        base: BaseSkill {
            name: name.to_string(),
            description: format!("Description for {name}"),
            file_path: format!("/skills/{name}/SKILL.md"),
            base_dir: format!("/skills/{name}"),
            source_info: create_synthetic_source_info(
                &format!("/skills/{name}/SKILL.md"),
                &SyntheticSourceInfoOptions {
                    source: "local".to_string(),
                    ..Default::default()
                },
            ),
            disable_model_invocation,
        },
        kind: SkillKind::Markdown,
    })
}

fn python_skill(name: &str, disable_model_invocation: bool) -> Skill {
    Skill::Python(PythonSkill {
        base: BaseSkill {
            name: name.to_string(),
            description: format!("Python skill {name}"),
            file_path: format!("/skills/{name}/SKILL.py"),
            base_dir: format!("/skills/{name}"),
            source_info: create_synthetic_source_info(
                &format!("/skills/{name}/SKILL.py"),
                &SyntheticSourceInfoOptions {
                    source: "local".to_string(),
                    ..Default::default()
                },
            ),
            disable_model_invocation,
        },
        kind: SkillKind::Python,
        python: SkillPythonMetadata {
            import_name: format!("skill_{name}"),
            package_path: format!("/skills/{name}"),
            pyproject_path: format!("/skills/{name}/pyproject.toml"),
        },
    })
}

fn noul(probability: f64) -> Answer {
    Answer::Noul { noul: probability }
}

fn choice(label: &str, probability: f64, confidence: f64) -> Answer {
    Answer::Choice {
        choice: label.to_string(),
        probabilities: BTreeMap::from([
            (label.to_string(), probability),
            ("none".to_string(), (1.0 - probability).max(0.0)),
        ]),
        confidence,
    }
}

/// A raw record for one prepared question id. The category value is an
/// EXISTING DecisionCategory variant on purpose: consumption matches by
/// question id, so these adapter tests compile and pass independently of the
/// WIRE types.rs hunk that registers the three new guidance categories.
fn record(question_id: &str, answer: Answer) -> DecisionRecord {
    DecisionRecord {
        question_id: question_id.to_string(),
        category: DecisionCategory::TraceAssessment,
        answer,
        response_model: Some("jev-test".to_string()),
        requested_model: "jev-latest".to_string(),
        applied: false,
    }
}

fn severity_answer(value: f64) -> Answer {
    Answer::Score {
        score: value,
        legend: BTreeMap::from([
            ("0".to_string(), pi_jev::types::EntryValue::Null),
            ("1".to_string(), pi_jev::types::EntryValue::Null),
            ("2".to_string(), pi_jev::types::EntryValue::Null),
            ("3".to_string(), pi_jev::types::EntryValue::Null),
            ("4".to_string(), pi_jev::types::EntryValue::Null),
        ]),
        probabilities: BTreeMap::from([
            ("0".to_string(), 0.0),
            ("1".to_string(), 0.0),
            ("2".to_string(), 0.0),
            ("3".to_string(), 0.0),
            ("4".to_string(), 1.0),
        ]),
        confidence: 0.9,
    }
}

fn roster(names: &[&str]) -> Vec<Skill> {
    names.iter().map(|name| markdown_skill(name, false)).collect()
}

/// The CURRENT host stamp fixture: authoritative production/consumption facts.
fn test_stamp() -> SkillHintStamp {
    let skills = roster(&["alpha", "beta"]);
    skill_hint_stamp(
        "rev-7",
        "compare_and_active",
        true,
        4,
        "delivery-1",
        "task excerpt A",
        &skill_roster_view(&skills).0,
    )
}

fn test_captured() -> CapturedSkillHintStamp {
    CapturedSkillHintStamp::capture(test_stamp())
}

/// Rebuilds the CURRENT stamp with the same facts but a different catalog
/// view, for content-drift tests.
fn stamp_with_catalog(catalog: &[pi_jev::agent_guidance::SkillCatalogEntry]) -> SkillHintStamp {
    skill_hint_stamp("rev-7", "compare_and_active", true, 4, "delivery-1", "task excerpt A", catalog)
}

// ---------------------------------------------------------------------------
// Roster view: suggestion pool == prompt roster
// ---------------------------------------------------------------------------

#[test]
fn suggestion_pool_excludes_disabled_skills_like_the_prompt_roster() {
    let skills = vec![
        markdown_skill("alpha", false),
        markdown_skill("refine", true),
        python_skill("websearch", false),
    ];
    let (entries, truncated) = skill_roster_view(&skills);
    assert!(!truncated);
    let ids: Vec<&str> = entries.iter().map(|entry| entry.id.as_str()).collect();
    assert_eq!(ids, vec!["alpha", "websearch"]);
    // Parity with the prompt roster: the same visibility filter.
    let prompt_roster = format_skills_for_prompt(&skills);
    for entry in &entries {
        assert!(prompt_roster.contains(&format!("<name>{}</name>", entry.id)));
    }
    assert!(!prompt_roster.contains("<name>refine</name>"));
}

#[test]
fn suggestion_pool_is_bounded_and_discloses_truncation() {
    let names: Vec<String> = (0..MAX_GUIDANCE_CATALOG + 7)
        .map(|index| format!("skill{index:04}"))
        .collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let skills = roster(&refs);
    let (entries, truncated) = skill_roster_view(&skills);
    assert!(truncated);
    assert_eq!(entries.len(), MAX_GUIDANCE_CATALOG);
    // First-come order matches the loaded roster order.
    assert_eq!(entries[0].id, "skill0000");
}

#[test]
fn catalog_ids_match_the_roster_view() {
    let skills = roster(&["alpha", "beta"]);
    assert_eq!(
        skill_catalog_ids(&skills),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

// ---------------------------------------------------------------------------
// Skill suggestion questions
// ---------------------------------------------------------------------------

#[test]
fn skill_suggestion_prepares_rank_plus_three_gates() {
    let skills = roster(&["alpha", "beta"]);
    let prepared = prepare_skill_suggestion("fix the flaky ci test", &skills).expect("prepared");
    let ids: Vec<&str> = prepared
        .questions
        .iter()
        .map(|question| question.question_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec![
            "skill_suggestion.0",
            "skill_suggestion.1",
            "skill_suggestion.2",
            "skill_suggestion.3",
        ]
    );
    for question in &prepared.questions {
        question.spec.validate_shape().expect("valid shape");
    }
    // The rank question's closed set is exactly the visible roster plus none.
    match &prepared.questions[0].spec {
        QuestionSpec::Choice { criteria, .. } => {
            let mut keys: Vec<&String> = criteria.keys().collect();
            keys.sort();
            assert_eq!(
                keys,
                vec![&"alpha".to_string(), &"beta".to_string(), &"none".to_string()]
            );
        }
        other => panic!("expected a Choice rank question, got {other:?}"),
    }
    let state = &prepared.state;
    assert_eq!(state["skill_catalog_count"], 2);
    assert_eq!(state["skill_catalog_truncated"], false);
    assert_eq!(state["hint_authority"], "none");
    assert!(state["user_text_excerpt"].as_str().expect("excerpt").contains("flaky"));
}

#[test]
fn empty_roster_prepares_nothing() {
    let skills: Vec<Skill> = vec![markdown_skill("refine", true)];
    assert!(prepare_skill_suggestion("task", &skills).is_none());
}

#[test]
fn oversized_task_excerpts_stay_bounded() {
    let skills = roster(&["alpha"]);
    let long = "x".repeat(10_000);
    let prepared = prepare_skill_suggestion(&long, &skills).expect("prepared");
    let excerpt = prepared.state["user_text_excerpt"].as_str().expect("excerpt");
    assert!(excerpt.chars().count() <= pi_jev::agent_guidance::MAX_GUIDANCE_TEXT_CHARS + 3);
}

// ---------------------------------------------------------------------------
// Skill hint consumption
// ---------------------------------------------------------------------------

fn accepted_skill_records() -> Vec<DecisionRecord> {
    vec![
        record("skill_suggestion.0", choice("alpha", 0.9, 0.85)),
        record("skill_suggestion.1", noul(0.95)),
        record("skill_suggestion.2", noul(0.05)),
        record("skill_suggestion.3", noul(0.9)),
    ]
}

#[test]
fn accepted_records_yield_one_advisory_non_exclusive_hint() {
    let skills = roster(&["alpha", "beta"]);
    let outcome = skill_hint_from_raw(&accepted_skill_records(), true, true, &skills, test_captured());
    match outcome {
        SkillHintOutcome::Hint(hint) => {
            assert_eq!(hint.skill_id, "alpha");
            assert_eq!(hint.label, "advisory, non-exclusive");
            let text = format_skill_hint(&hint);
            assert!(text.contains("<jev_skill_hint>"));
            assert!(text.contains("NON-EXCLUSIVE"));
            assert!(text.contains("No permission, model, tool or execution change"));
            assert!(text.contains("<available_skills> roster above remains complete"));
        }
        other => panic!("expected a hint, got {other:?}"),
    }
}

#[test]
fn the_hint_never_splices_into_the_roster_block() {
    let skills = roster(&["alpha", "beta"]);
    let roster_text = format_skills_for_prompt(&skills);
    let outcome = skill_hint_from_raw(&accepted_skill_records(), true, true, &skills, test_captured());
    let hint = match outcome {
        SkillHintOutcome::Hint(hint) => hint,
        other => panic!("expected a hint, got {other:?}"),
    };
    let hint_text = format_skill_hint(&hint);
    // The hint is its own appended block: it never appears inside the roster
    // text, and the roster text is identical whether or not a hint exists.
    assert!(hint_text.starts_with("\n\n<jev_skill_hint>"));
    assert!(!roster_text.contains("<jev_skill_hint>"));
    assert_eq!(roster_text, format_skills_for_prompt(&skills));
}

#[test]
fn consumption_names_deterministic_no_hint_reasons() {
    let skills = roster(&["alpha", "beta"]);
    // Missing rank answer: no confident candidate, never an absence claim.
    let mut records = accepted_skill_records();
    records.remove(0);
    assert_eq!(
        skill_hint_from_raw(&records, true, true, &skills, test_captured()),
        SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate)
    );
    // Stale: the ranked id left the roster.
    let stale = vec![
        record("skill_suggestion.0", choice("ghost", 0.9, 0.9)),
        record("skill_suggestion.1", noul(0.95)),
        record("skill_suggestion.2", noul(0.05)),
        record("skill_suggestion.3", noul(0.9)),
    ];
    assert_eq!(
        skill_hint_from_raw(&stale, true, true, &skills, test_captured()),
        SkillHintOutcome::NoHint(NoHintReason::Stale)
    );
    // Budget: the assessment did not run.
    assert_eq!(
        skill_hint_from_raw(&accepted_skill_records(), true, false, &skills, test_captured()),
        SkillHintOutcome::NoHint(NoHintReason::BudgetExhausted)
    );
    // Not fresh: generation/turn correlation failed.
    assert_eq!(
        skill_hint_from_raw(&accepted_skill_records(), false, true, &skills, test_captured()),
        SkillHintOutcome::NoHint(NoHintReason::Stale)
    );
}

#[test]
fn uncertain_gates_never_become_catalog_absence_claims() {
    let skills = roster(&["alpha", "beta"]);
    let mut records = accepted_skill_records();
    records[1] = record("skill_suggestion.1", noul(0.5)); // need gate inside the band
    assert_eq!(
        skill_hint_from_raw(&records, true, true, &skills, test_captured()),
        SkillHintOutcome::NoHint(NoHintReason::Uncertain)
    );
}

#[test]
fn thresholds_and_policy_are_valid_host_constants() {
    assert!(skill_guidance_thresholds().validate().is_ok());
    let policy = guardrail_policy();
    assert!(policy.validate().is_ok());
    // Full coverage of both batteries' hazard ids.
    for (id, _) in GUARDRAIL_INPUT_HAZARDS.iter().chain(GUARDRAIL_OUTPUT_HAZARDS.iter()) {
        assert!(policy.weights.contains_key(*id), "weight coverage for {id}");
    }
    for critical in policy.critical_ids {
        assert!(policy.weights.contains_key(*critical));
    }
}

// ---------------------------------------------------------------------------
// Guardrail batteries: assessment only, truthful timing, enforced=false
// ---------------------------------------------------------------------------

#[test]
fn batteries_prepare_four_nouls_plus_one_severity_score() {
    for prefix in ["guardrails_input", "guardrails_output"] {
        let prepared = prepare_guardrail_battery(prefix, "user content excerpt").expect("battery");
        assert_eq!(prepared.questions.len(), 5);
        for (index, question) in prepared.questions.iter().enumerate() {
            assert_eq!(question.question_id, format!("{prefix}.{index}"));
            question.spec.validate_shape().expect("valid shape");
        }
        match &prepared.questions[4].spec {
            QuestionSpec::Score { criteria, .. } => {
                assert_eq!(criteria.len(), GUARDRAIL_SEVERITY_LEVELS.len());
            }
            other => panic!("expected a Score severity question, got {other:?}"),
        }
        assert_eq!(prepared.state["enforced"], false);
        assert_eq!(prepared.state["screening"], "assessment_only");
        let expected_timing = if prefix == "guardrails_input" {
            "input_screen_preventive"
        } else {
            "output_post_hoc_assessment"
        };
        assert_eq!(prepared.state["timing"], expected_timing);
    }
}

#[test]
fn unknown_battery_prefix_prepares_nothing() {
    assert!(prepare_guardrail_battery("guardrails_sideways", "excerpt").is_none());
}

fn guardrail_records(
    prefix: &str,
    first: f64,
    second: f64,
    third: f64,
    fourth: f64,
    severity: Option<f64>,
) -> Vec<DecisionRecord> {
    let mut records = vec![
        record(&format!("{prefix}.0"), noul(first)),
        record(&format!("{prefix}.1"), noul(second)),
        record(&format!("{prefix}.2"), noul(third)),
        record(&format!("{prefix}.3"), noul(fourth)),
    ];
    if let Some(severity_value) = severity {
        records.push(record(&format!("{prefix}.4"), severity_answer(severity_value)));
    }
    records
}

#[test]
fn output_assessment_is_labelled_post_hoc_and_never_enforced() {
    let records = guardrail_records("guardrails_output", 0.1, 0.1, 0.1, 0.1, Some(0.2));
    let assessment =
        guardrail_assessment_from_raw("guardrails_output", &records, true, true).expect("assessment");
    assert_eq!(assessment.timing, TimingLabel::OutputPostHocAssessment);
    assert_eq!(assessment.enforced, false, "structurally false, labelled truthfully");
    let text = format_guardrail_advisory(&assessment, "guardrails_output");
    assert!(text.contains("enforced=\"false\""));
    assert!(text.contains("output_post_hoc_assessment"));
    assert!(text.contains("POST-HOC"));
    assert!(text.contains("not a safety guarantee"));
    assert!(text.contains("No message, tool, model, permission or execution path was changed"));
}

#[test]
fn input_assessment_uses_preventive_timing() {
    let records = guardrail_records("guardrails_input", 0.9, 0.1, 0.1, 0.1, Some(0.2));
    let assessment = guardrail_assessment_from_raw("guardrails_input", &records, true, true).expect("assessment");
    assert_eq!(assessment.timing, TimingLabel::InputScreenPreventive);
    // The decisive hidden-instruction hazard alone stays under the composed
    // review threshold (0.30*0.9 + small tails < 0.40): pass, flags recorded.
    assert_eq!(assessment.decisive_flags, vec!["hidden_instructions".to_string()]);
    assert_eq!(assessment.route.as_str(), "pass");
}

#[test]
fn critical_hazards_force_review_even_when_the_composition_is_low() {
    let records = guardrail_records("guardrails_input", 0.05, 0.95, 0.05, 0.05, Some(0.1));
    let assessment = guardrail_assessment_from_raw("guardrails_input", &records, true, true).expect("assessment");
    assert_eq!(assessment.route.as_str(), "review");
    assert_eq!(assessment.critical_flags, vec!["credential_exfiltration".to_string()]);
}

#[test]
fn missing_or_invalid_answers_assess_nothing() {
    // A missing hazard record cannot produce a verdict.
    let mut records = guardrail_records("guardrails_output", 0.1, 0.1, 0.1, 0.1, Some(0.2));
    records.remove(2);
    assert!(guardrail_assessment_from_raw("guardrails_output", &records, true, true).is_none());
    // A malformed noul is unavailable, not a verdict.
    let mut records = guardrail_records("guardrails_output", 0.1, 0.1, 0.1, 0.1, Some(0.2));
    records[1] = record("guardrails_output.1", noul(f64::NAN));
    assert!(guardrail_assessment_from_raw("guardrails_output", &records, true, true).is_none());
    // Lifecycle failures (freshness/budget) are passed through by the caller
    // and the core refuses to derive a verdict from them.
    let records = guardrail_records("guardrails_output", 0.1, 0.1, 0.1, 0.1, Some(0.2));
    assert!(guardrail_assessment_from_raw("guardrails_output", &records, false, true).is_none());
    assert!(guardrail_assessment_from_raw("guardrails_output", &records, true, false).is_none());
}

#[test]
fn fit_question_is_reachable_for_the_second_pass() {
    let question = skill_fit_prepared("alpha").expect("fit question");
    assert_eq!(question.question_id, "skill_suggestion.4");
    question.spec.validate_shape().expect("valid shape");
}

// ---------------------------------------------------------------------------
// Outgoing-payload visibility (ROOT CONTRACT v7 cross-lane rule, adapter side)
// ---------------------------------------------------------------------------

#[test]
fn overlong_skill_ids_are_skipped_never_asked() {
    // An id above the native key bound could be silently clipped by
    // snapshot::bound_json: it is excluded from the pool entirely, so no
    // selectable choice can lose its evidence on the way out.
    let long_name = "z".repeat(pi_jev::agent_guidance::MAX_GUIDANCE_ID_CHARS + 1);
    let skills = vec![markdown_skill(&long_name, false), markdown_skill("alpha", false)];
    let (entries, truncated, skipped) = skill_roster_view_with_skips_public(&skills);
    assert_eq!(skipped, 1);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].id, "alpha");
    assert!(!truncated);
    // The prepared request's closed set never contains the clipped id.
    let prepared = prepare_skill_suggestion("task", &skills).expect("prepared");
    match &prepared.questions[0].spec {
        QuestionSpec::Choice { criteria, .. } => {
            assert!(!criteria.keys().any(|key| key.len() > pi_jev::agent_guidance::MAX_GUIDANCE_ID_CHARS));
            assert!(!criteria.contains_key(long_name.as_str()));
        }
        other => panic!("expected a Choice rank question, got {other:?}"),
    }
    // The skip is disclosed in the outgoing state (provenance truth).
    assert_eq!(prepared.state["skill_catalog_skipped_overlong_ids"], 1);
}

/// Public re-export of the adapter's roster-with-skips view for tests.
use pi_coding_agent::core::jev_agent_guidance::skill_roster_view_with_skips as skill_roster_view_with_skips_public;

#[test]
fn prepared_guidance_payloads_survive_the_native_bounds() {
    // The adapter verifies the ACTUAL outgoing payload before returning: the
    // same native rules (state bytes <= 8KiB, bound_json no-op, closed-set
    // options with present evidence, shape, token ceiling, body bytes).
    let skills = roster(&["alpha", "beta"]);
    let prepared: PreparedGuidance = prepare_skill_suggestion("bounded task", &skills).expect("prepared");
    // The intended catalog is exactly what the rank question asks (minus none).
    let catalog_ids: Vec<String> = match &prepared.questions[0].spec {
        QuestionSpec::Choice { criteria, .. } => criteria
            .keys()
            .filter(|key| key.as_str() != "none")
            .cloned()
            .collect(),
        other => panic!("expected a Choice rank question, got {other:?}"),
    };
    // Re-verify through the core verifier exactly as the adapter did.
    pi_jev::agent_guidance::verify_guidance_request(
        &prepared.state,
        &prepared.questions,
        Some(("skill_suggestion.0", &catalog_ids)),
    )
    .expect("skill request must remain provably visible");
    // Every state string survives the native text bound unchanged.
    assert_eq!(
        pi_jev::snapshot::bound_json(prepared.state.clone(), 0),
        prepared.state
    );
    // Batteries: same proof.
    let battery = prepare_guardrail_battery("guardrails_input", "user excerpt").expect("battery");
    pi_jev::agent_guidance::verify_guidance_request(&battery.state, &battery.questions, None)
        .expect("battery must remain provably visible");
}


#[test]
fn battery_excerpt_bounds_match_the_native_text_cap() {
    let long = "x".repeat(5_000);
    let battery = prepare_guardrail_battery("guardrails_output", &long).expect("battery");
    let excerpt = battery.state["screened_excerpt"].as_str().expect("excerpt");
    assert!(excerpt.chars().count() <= pi_jev::agent_guidance::MAX_GUIDANCE_TEXT_CHARS + 3);
    // Scope disclosure: a partial excerpt cannot claim full message coverage.
    assert!(battery.state["scope"]
        .as_str()
        .expect("scope")
        .contains("never the full transcript"));
}

// ---------------------------------------------------------------------------
// Runtime prompt wiring (render layer): REAL build_system_prompt capture
// ---------------------------------------------------------------------------

use pi_coding_agent::core::system_prompt::{build_system_prompt, BuildSystemPromptOptions};

fn prompt_options(skills: Vec<Skill>, hint: Option<String>) -> BuildSystemPromptOptions {
    BuildSystemPromptOptions {
        custom_prompt: None,
        selected_tools: Some(vec!["ipython".to_string(), "bash".to_string()]),
        tool_snippets: None,
        prompt_guidelines: None,
        append_system_prompt: None,
        cwd: "/tmp/jev-guidance-probe".to_string(),
        messages_path: None,
        context_files: Some(Vec::new()),
        skills: Some(skills),
        skill_hint: hint,
        allow_recursion: None,
        rlm_depth: None,
        rlm_parent_agent: None,
        harness_state: None,
        generic_mcp_servers: None,
    }
}

#[test]
fn hint_block_renders_once_after_a_byte_identical_roster() {
    let skills = roster(&["alpha", "beta"]);
    let hint = match skill_hint_from_raw(&accepted_skill_records(), true, true, &skills, test_captured()) {
        SkillHintOutcome::Hint(hint) => hint,
        other => panic!("expected a hint, got {other:?}"),
    };
    let hint_block = format_skill_hint(&hint);
    let without = build_system_prompt(&prompt_options(skills.clone(), None));
    let with = build_system_prompt(&prompt_options(skills.clone(), Some(hint_block.clone())));
    // Roster byte parity: the SAME full roster block appears unchanged in both.
    let roster_block = format_skills_for_prompt(&skills);
    assert!(!roster_block.is_empty(), "fixture renders a roster");
    assert!(without.contains(&roster_block), "baseline prompt carries the roster");
    assert!(with.contains(&roster_block), "hinted prompt carries the SAME roster");
    // Exactly one hint block.
    assert_eq!(with.matches("<jev_skill_hint>").count(), 1, "at most one hint block");
    assert!(!without.contains("<jev_skill_hint>"), "no hint renders when none is provided");
    // The hinted prompt equals the baseline prompt with the hint inserted
    // EXACTLY once, right after the roster block: everything before and after
    // the insertion point is byte-identical (prefix caching preserved).
    let insert_at = without.find(&roster_block).expect("roster offset") + roster_block.len();
    assert_eq!(&with[..insert_at], &without[..insert_at], "before-hint region byte-identical");
    assert!(with[insert_at..].starts_with(&hint_block), "hint appended immediately after the roster");
    assert_eq!(&with[insert_at + hint_block.len()..], &without[insert_at..], "after-hint region byte-identical");
    // A held hint can never appear above the roster: insertion is positional.
    assert!(with.rfind("<jev_skill_hint>").unwrap() > with.find(&roster_block).unwrap());
}

#[test]
fn no_hint_renders_no_hint_block() {
    // The None path is what NoHint / failure / full-off removal MUST end in:
    // nothing renders.
    let skills = roster(&["alpha", "beta"]);
    let prompt = build_system_prompt(&prompt_options(skills, None));
    assert!(!prompt.contains("<jev_skill_hint>"));
}

#[test]
fn hint_is_never_rendered_without_a_roster_above_it() {
    let skills = roster(&["alpha"]);
    let hint = match skill_hint_from_raw(&accepted_skill_records(), true, true, &skills, test_captured()) {
        SkillHintOutcome::Hint(hint) => hint,
        other => panic!("expected a hint, got {other:?}"),
    };
    let block = format_skill_hint(&hint);
    // Empty roster: the skills section (and with it the hint) never renders.
    let prompt = build_system_prompt(&prompt_options(Vec::new(), Some(block)));
    assert!(!prompt.contains("<jev_skill_hint>"), "a hint never renders without a roster above it");
    assert!(!prompt.contains("/skills/alpha/SKILL.md"), "no skill text leaks without the skills section");
}

// ---------------------------------------------------------------------------
// Hint identity stamp: consumption-side ABA / task / turn rejection
// ---------------------------------------------------------------------------

#[test]
fn hint_carries_the_production_stamp_and_the_gate_rejects_any_drift() {
    let skills = roster(&["alpha", "beta"]);
    let hint = match skill_hint_from_raw(&accepted_skill_records(), true, true, &skills, test_captured()) {
        SkillHintOutcome::Hint(hint) => hint,
        other => panic!("expected a hint, got {other:?}"),
    };
    // The hint carries the production facts verbatim.
    assert_eq!(hint.stamp, test_stamp());
    // Current facts unchanged: renders.
    assert!(skill_hint_is_current(&hint, &test_stamp()));

    // New task: hintA must never serve taskB.
    let catalog = skill_roster_view(&skills).0;
    let task_b = skill_hint_stamp("rev-7", "compare_and_active", true, 4, "delivery-1", "task excerpt B", &catalog);
    assert!(!skill_hint_is_current(&hint, &task_b), "task drift rejected");

    // Catalog change (a skill left/entered the roster): rejected.
    let other_catalog = stamp_with_catalog(&skill_roster_view(&roster(&["alpha"])).0);
    assert!(!skill_hint_is_current(&hint, &other_catalog), "catalog drift rejected");

    // Turn advance (prompt built after a new TurnStart): rejected.
    let next_turn = skill_hint_stamp("rev-7", "compare_and_active", true, 5, "delivery-1", "task excerpt A", &catalog);
    assert!(!skill_hint_is_current(&hint, &next_turn), "cross-turn leak rejected");

    // Mode change: rejected.
    let mode_changed = skill_hint_stamp("rev-7", "compare", true, 4, "delivery-1", "task excerpt A", &catalog);
    assert!(!skill_hint_is_current(&hint, &mode_changed), "mode drift rejected");

    // Feature disabled at consumption: always rejected (defense in depth).
    let feature_off = skill_hint_stamp("rev-7", "compare_and_active", false, 4, "delivery-1", "task excerpt A", &catalog);
    assert!(!skill_hint_is_current(&hint, &feature_off), "feature-off consumption rejected");

    // Off->On ABA: NO consumer visit during the off window, settings reloaded
    // twice while off (rev-7 -> rev-9). The old hint must be rejected because
    // the authoritative settings revision moved — clearing alone cannot do
    // this when nobody observed the off period.
    assert!(!skill_hint_is_current(&hint, &after_aba_stamp(&catalog)), "Off->On ABA rejected via settings revision");
    assert!(!skill_hint_is_current(&hint, &after_aba_stamp(&catalog)), "ABA rejection is stable");
}

fn after_aba_stamp(catalog: &[pi_jev::agent_guidance::SkillCatalogEntry]) -> SkillHintStamp {
    skill_hint_stamp("rev-9", "compare_and_active", true, 4, "delivery-1", "task excerpt A", catalog)
}

#[test]
fn render_gate_fails_open_to_no_block_on_stale_hints() {
    let skills = roster(&["alpha", "beta"]);
    let hint = match skill_hint_from_raw(&accepted_skill_records(), true, true, &skills, test_captured()) {
        SkillHintOutcome::Hint(hint) => hint,
        other => panic!("expected a hint, got {other:?}"),
    };
    // Current facts: the hint renders as the normal block.
    assert!(skill_hint_for_render(&hint, &test_stamp()).is_some());
    // Stale facts: NO block at all (advisory fail-open; the canonical base
    // prompt is untouched — request-local derivation, never regex-stripping).
    let stale = skill_hint_stamp("rev-9", "compare_and_active", true, 4, "delivery-1", "task excerpt A", &skill_roster_view(&skills).0);
    assert_eq!(skill_hint_for_render(&hint, &stale), None);
}

#[test]
fn stamp_hashes_separate_task_and_catalog_identity() {
    let a = guidance_stamp_hash(&["task excerpt A"]);
    let b = guidance_stamp_hash(&["task excerpt B"]);
    assert_ne!(a, b, "distinct tasks hash apart");
    let c1 = guidance_stamp_hash(&["alpha", "beta"]);
    let c2 = guidance_stamp_hash(&["alpha"]);
    assert_ne!(c1, c2, "catalog changes hash apart");
    assert_eq!(a, guidance_stamp_hash(&["task excerpt A"]), "deterministic");
}

#[test]
fn stamp_rejects_same_ids_with_changed_descriptions() {
    // Same catalog IDs, changed description MEANING: the ids-only identity of
    // the old design could not detect this; the content fingerprint does.
    let catalog_a = vec![catalog_entry("alpha", "documents the deploy procedure")];
    let catalog_b = vec![catalog_entry("alpha", "documents the rollback procedure")];
    let hint_stamp = skill_hint_stamp("rev-7", "compare_and_active", true, 4, "delivery-1", "task text", &catalog_a);
    let current = skill_hint_stamp("rev-7", "compare_and_active", true, 4, "delivery-1", "task text", &catalog_b);
    assert_ne!(hint_stamp.catalog_hash, current.catalog_hash, "same ids, changed meaning hash apart");
    let skills = roster(&["alpha"]);
    let hint = match skill_hint_from_raw(&accepted_skill_records(), true, true, &skills, CapturedSkillHintStamp::capture(hint_stamp)) {
        SkillHintOutcome::Hint(hint) => hint,
        other => panic!("expected a hint, got {other:?}"),
    };
    assert!(!skill_hint_is_current(&hint, &current), "changed-description catalog rejected at consumption");
    // Identical contents still match (deterministic fingerprint).
    let same = skill_hint_stamp("rev-7", "compare_and_active", true, 4, "delivery-1", "task text", &catalog_a);
    assert!(skill_hint_is_current(&hint, &same), "unchanged catalog accepted");
}

#[test]
fn stamp_rejects_tasks_that_share_a_bounded_prefix() {
    // Two DIFFERENT tasks whose first 400+ chars are identical: the old
    // bounded-400 presentation excerpt hashed them EQUAL; the full-text
    // fingerprint does not.
    let prefix = "step ".repeat(200); // > 400 chars, identical prefix
    let task_a = format!("{prefix}deploy now");
    let task_b = format!("{prefix}rollback now");
    let skills = roster(&["alpha"]);
    let catalog = skill_roster_view(&skills).0;
    let s_a = skill_hint_stamp("rev-7", "compare_and_active", true, 4, "delivery-1", &task_a, &catalog);
    let s_b = skill_hint_stamp("rev-7", "compare_and_active", true, 4, "delivery-1", &task_b, &catalog);
    assert_ne!(s_a.task_hash, s_b.task_hash, "same prefix, different task hashes apart");
    assert!(s_a.task_hash.len() >= 64, "fingerprint is a local digest; no raw task text");
}

fn catalog_entry(id: &str, description: &str) -> pi_jev::agent_guidance::SkillCatalogEntry {
    pi_jev::agent_guidance::SkillCatalogEntry {
        id: id.to_string(),
        description: description.to_string(),
    }
}

#[test]
fn identical_text_on_a_new_delivery_never_reuses_the_previous_hint() {
    // Root: a REAL NEW task with byte-identical text must not reuse the
    // previous request's hint — content hashing alone cannot distinguish the
    // two deliveries; the host per-delivery identity does.
    let skills = roster(&["alpha"]);
    let catalog = skill_roster_view(&skills).0;
    let task = "identical task text";
    let old = skill_hint_stamp("rev-7", "compare_and_active", true, 4, "delivery-1", task, &catalog);
    // Same turn, same settings, same catalog, IDENTICAL text — but a NEW
    // delivery: rejected.
    let new_delivery = skill_hint_stamp("rev-7", "compare_and_active", true, 4, "delivery-2", task, &catalog);
    assert_ne!(old.delivery_id, new_delivery.delivery_id);
    assert_ne!(old, new_delivery, "identical text on a new delivery is a different stamp");
    let hint = match skill_hint_from_raw(&accepted_skill_records(), true, true, &skills, CapturedSkillHintStamp::capture(old.clone())) {
        SkillHintOutcome::Hint(hint) => hint,
        other => panic!("expected a hint, got {other:?}"),
    };
    assert!(!skill_hint_is_current(&hint, &new_delivery), "identical-text new-task reuse rejected");
    // The SAME delivery still matches (positive path).
    assert!(skill_hint_is_current(&hint, &old), "same delivery accepted");
}
