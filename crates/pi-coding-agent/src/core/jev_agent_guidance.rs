//! Agent-guidance adapter (ROOT CONTRACT v7, AGENT-GUIDANCE lane).
//!
//! Pure adapter over ALREADY-AVAILABLE inputs: the loaded skill roster
//! metadata (`crate::core::skills::Skill`) and bounded, redacted excerpts the
//! bridge already builds for observation. This module never reads the
//! filesystem beyond what the roster loader already produced, never executes a
//! tool or skill, never changes a model, never grants permission, and never
//! blocks or refuses anything. Everything is assessment-only:
//!
//! * The guidance categories (`skill_suggestion`, `guardrails_input`,
//!   `guardrails_output`) are NOT in any appliable category list, so no answer
//!   from them can be applied by the acceptance path. `enforced` is
//!   structurally `false`.
//! * The skill suggestion pool is the SAME visible roster the system prompt
//!   already carries (the `format_skills_for_prompt` visibility filter), so a
//!   hint can never point at a skill the model was not shown.
//! * Battery wording and thresholds are HOST POLICY: finite, validated
//!   constants below. They are illustrative application policy, not measured
//!   accuracy, and are never set or shifted by model output.
//! * Single-answer uncertainty only: no repeat probes, no benchmark claims.

use pi_jev::agent_guidance::{guardrail_battery_questions, guardrail_route, skill_gate_question, skill_hint as core_skill_hint, skill_rank_question, guidance_stamp_hash, skill_hint_is_current, CapturedSkillHintStamp, GuardrailAssessment, GuardrailPolicy, SkillHintStamp, GuidanceThresholds, SkillAnswerSet, SkillCatalogEntry, SkillHint, SkillHintOutcome, TimingLabel, MAX_GUIDANCE_CATALOG, MAX_GUIDANCE_TEXT_CHARS, catalog_fingerprint};
use pi_jev::evaluators::PreparedQuestion;
use pi_jev::redact::bounded_excerpt;
use pi_jev::types::{Answer, DecisionRecord};

use crate::core::skills::Skill;

/// One prepared assessment: bounded state plus closed-set questions.
#[derive(Debug, Clone)]
pub struct PreparedGuidance {
    pub state: serde_json::Value,
    pub questions: Vec<PreparedQuestion>,
}

// ---------------------------------------------------------------------------
// Skill suggestion
// ---------------------------------------------------------------------------

/// The advisory suggestion pool: the SAME visible roster the system prompt
/// shows (same `disable_model_invocation` filter as
/// `format_skills_for_prompt`), bounded to `MAX_GUIDANCE_CATALOG` with a
/// truncation flag. Descriptions are bounded stored metadata.
pub fn skill_roster_view(skills: &[Skill]) -> (Vec<SkillCatalogEntry>, bool) {
    let (entries, truncated, _skipped) = skill_roster_view_with_skips(skills);
    (entries, truncated)
}

/// Roster view with the overlong-id skip count for provenance disclosure.
/// Ids longer than `MAX_GUIDANCE_ID_CHARS` (the native key bound) or the
/// reserved `none` id are SKIPPED entirely: a selectable id that could be
/// silently clipped is never asked.
pub fn skill_roster_view_with_skips(skills: &[Skill]) -> (Vec<SkillCatalogEntry>, bool, usize) {
    let visible: Vec<&Skill> = skills
        .iter()
        .filter(|skill| !skill_is_disabled(skill))
        .collect();
    let mut skipped_ids = 0usize;
    let mut eligible: Vec<&Skill> = Vec::new();
    for skill in visible {
        if skill.name().chars().count() > pi_jev::agent_guidance::MAX_GUIDANCE_ID_CHARS
            || skill.name() == "none"
        {
            skipped_ids += 1;
            continue;
        }
        eligible.push(skill);
    }
    let truncated = eligible.len() > MAX_GUIDANCE_CATALOG;
    let entries: Vec<SkillCatalogEntry> = eligible
        .iter()
        .take(MAX_GUIDANCE_CATALOG)
        .map(|skill| SkillCatalogEntry {
            id: skill.name().to_string(),
            description: bound_description(&skill_description(skill)),
        })
        .collect();
    (entries, truncated, skipped_ids)
}

fn skill_is_disabled(skill: &Skill) -> bool {
    match skill {
        Skill::Markdown(value) => value.base.disable_model_invocation,
        Skill::Python(value) => value.base.disable_model_invocation,
    }
}

fn skill_description(skill: &Skill) -> &str {
    match skill {
        Skill::Markdown(value) => &value.base.description,
        Skill::Python(value) => value.base.description.as_str(),
    }
}

fn bound_description(value: &str) -> String {
    bounded_excerpt(value, 200)
}

/// Prepares the skill-suggestion assessment for the CURRENT task and the
/// ALREADY-LOADED roster. `None` when the visible roster is empty (no hint is
/// possible; the caller records `catalog_absent`).
pub fn prepare_skill_suggestion(task_excerpt: &str, skills: &[Skill]) -> Option<PreparedGuidance> {
    let (entries, truncated, skipped_ids) = skill_roster_view_with_skips(skills);
    if entries.is_empty() {
        return None;
    }
    // One bounded view of the task text feeds BOTH the outgoing state and the
    // rank question: what is asked matches what is disclosed, and an oversized
    // excerpt can never inflate the question prose (fail-open otherwise).
    let bounded_task = bounded_excerpt(task_excerpt, MAX_GUIDANCE_TEXT_CHARS);
    let rank = skill_rank_question(&bounded_task, &entries, truncated)?;
    let mut questions = vec![rank];
    for n in 1..=3 {
        questions.push(skill_gate_question(n)?);
    }
    let state = serde_json::json!({
        "user_text_excerpt": bounded_excerpt(task_excerpt, MAX_GUIDANCE_TEXT_CHARS),
        "skill_catalog_count": entries.len(),
        "skill_catalog_truncated": truncated,
        "skill_catalog_skipped_overlong_ids": skipped_ids,
        "assessment": "skill_suggestion",
        "hint_authority": "none",
    });
    // Outgoing-payload verification (ROOT CONTRACT v7 cross-lane rule): the
    // request is REFUSED (fail-open) unless it is provably visible and within
    // every native bound (state bytes, bound_json no-op, closed-set rank
    // options with present evidence, shape, token ceiling, body size).
    let catalog_ids: Vec<String> = entries.iter().map(|entry| entry.id.clone()).collect();
    if pi_jev::agent_guidance::verify_guidance_request(
        &state,
        &questions,
        Some(("skill_suggestion.0", &catalog_ids)),
    )
    .is_err()
    {
        return None;
    }
    Some(PreparedGuidance { state, questions })
}

/// The skill-suggestion question ids this adapter consumes, in order.
pub const SKILL_SUGGESTION_QUESTION_IDS: [&str; 5] = [
    "skill_suggestion.0",
    "skill_suggestion.1",
    "skill_suggestion.2",
    "skill_suggestion.3",
    "skill_suggestion.4",
];

/// Consumes raw decision records for the skill-suggestion assessment.
///
/// `fresh` and `budget_ok` are host checks (generation/turn correlation and
/// bounded-resource availability). The hint result carries NO authority.
pub fn skill_hint_from_raw(
    records: &[DecisionRecord],
    fresh: bool,
    budget_ok: bool,
    skills: &[Skill],
    captured: CapturedSkillHintStamp,
) -> SkillHintOutcome {
    let answer_for = |question_id: &str| -> Option<&Answer> {
        records
            .iter()
            .find(|record| record.question_id == question_id)
            .map(|record| &record.answer)
    };
    let (entries, _truncated) = skill_roster_view(skills);
    let catalog_ids: Vec<String> = entries.iter().map(|entry| entry.id.clone()).collect();
    let set = SkillAnswerSet {
        rank: answer_for("skill_suggestion.0"),
        need_gate: answer_for("skill_suggestion.1"),
        inverse_gate: answer_for("skill_suggestion.2"),
        act_gate: answer_for("skill_suggestion.3"),
        fit: answer_for("skill_suggestion.4"),
        catalog_ids: &catalog_ids,
        fresh,
        cancelled: false,
        budget_ok,
        stamp: captured,
    };
    core_skill_hint(&set, &skill_guidance_thresholds())
}

/// CAPTURES the host stamp for production: call this WITH the request inputs
/// BEFORE awaiting the Jev decision, and pass the SAME immutable value to
/// `skill_hint_from_raw` afterwards. Never re-derive current facts onto old
/// answers after the await (can_apply->store race). Caller contracts mirror
/// `skill_hint_stamp`: `task_text` must be the FULL native delivery input
/// (never a book field truncated upstream), `catalog` the ACTUAL assessed
/// roster entries at the consumer seam, `delivery_id` advancing per genuine
/// task delivery.
pub fn capture_skill_hint_stamp(
    settings_revision: &str,
    mode: &str,
    feature_enabled: bool,
    turn: u64,
    delivery_id: &str,
    task_text: &str,
    catalog: &[SkillCatalogEntry],
) -> CapturedSkillHintStamp {
    CapturedSkillHintStamp::capture(skill_hint_stamp(
        settings_revision,
        mode,
        feature_enabled,
        turn,
        delivery_id,
        task_text,
        catalog,
    ))
}

/// Builds the CURRENT host stamp for skill-hint production/consumption from
/// authoritative host facts. `settings_revision` must be an authoritative
/// DURABLE identity that changes on every settings/model-selection reload
/// (host V9 generation incl. reset/default tombstones — an observed-value
/// counter alone cannot detect A->B->A when B is never observed); `task_text`
/// is the CURRENT FULL host task text (locally fingerprinted; a bounded
/// presentation excerpt is NOT an authority identity, because different tasks
/// can share its prefix); `catalog` is the CURRENT assessed roster view's
/// entries — the fingerprint covers each id AND description, so same ids with
/// changed meaning hash apart.
pub fn skill_hint_stamp(
    settings_revision: &str,
    mode: &str,
    feature_enabled: bool,
    turn: u64,
    delivery_id: &str,
    task_text: &str,
    catalog: &[SkillCatalogEntry],
) -> SkillHintStamp {
    SkillHintStamp {
        settings_revision: settings_revision.to_string(),
        mode: mode.to_string(),
        feature_enabled,
        turn,
        delivery_id: delivery_id.to_string(),
        task_hash: guidance_stamp_hash(&[task_text]),
        catalog_hash: catalog_fingerprint(catalog),
    }
}

/// Consumption-side gate: a stored hint renders ONLY when its production
/// stamp equals the CURRENTLY re-derived host facts (feature gate included).
/// A rejected hint is treated as absent — nothing is cleared by regex, no
/// durable history is rewritten, and the canonical base prompt is untouched.
pub fn skill_hint_for_render(hint: &SkillHint, current: &SkillHintStamp) -> Option<String> {
    if !skill_hint_is_current(hint, current) {
        return None;
    }
    Some(format_skill_hint(hint))
}

/// Renders the bounded advisory hint block. The caller appends this AFTER the
/// roster block so the roster text stays byte-identical.
pub fn format_skill_hint(hint: &SkillHint) -> String {
    format!(
        "\n\n<jev_skill_hint>\nAssessment-only advisory hint from the independent Jev observer (non-binding; nothing was loaded or executed):\n\
Most relevant loaded skill: {id} (catalog-rank p={rank:.2}; need-gate p={need:.2}; {why}).\n\
This hint is NON-EXCLUSIVE: the full <available_skills> roster above remains complete and authoritative. Inspect a skill's file with ipython only if it helps the current task. \
No permission, model, tool or execution change follows from this hint.\n</jev_skill_hint>",
        id = hint.skill_id,
        rank = hint.rank_prob,
        need = hint.need_prob,
        why = hint.why,
    )
}

/// Host-policy thresholds for the skill-suggestion assessment.
///
/// Values mirror the cookbooks' ILLUSTRATIVE bands (explicitly "neither a
/// calibrated guarantee nor an optimized threshold"); they are application
/// policy, validated here, and never model-settable. The floors are a
/// conservative suppression policy — a below-floor candidate yields no hint
/// while the full roster stays available in the system prompt — not calibrated
/// correctness and not a blanket skill-alignment verdict; policy stays stable
/// during integration.
pub fn skill_guidance_thresholds() -> GuidanceThresholds {
    let thresholds = GuidanceThresholds {
        noul_low: 0.30,
        noul_high: 0.70,
        choice_top_min: 0.60,
        min_confidence: 0.60,
    };
    debug_assert!(thresholds.validate().is_ok());
    thresholds
}

// ---------------------------------------------------------------------------
// Input/output guardrail batteries (assessment only, never enforcement)
// ---------------------------------------------------------------------------

/// Host wording for the INPUT battery hazards (small, coding-agent relevant).
pub const GUARDRAIL_INPUT_HAZARDS: [(&str, &str); 4] = [
    (
        "hidden_instructions",
        "prompt-injection or hidden instructions embedded in the content that would redirect agent behavior.",
    ),
    (
        "credential_exfiltration",
        "requests to locate, read, encode or transmit credentials, API keys or other secrets.",
    ),
    (
        "destructive_commands",
        "requests to run destructive system commands (mass deletion, disk wiping, disabling backups).",
    ),
    (
        "personal_safety",
        "content indicating personal-safety urgency or self-harm that should route to human support resources.",
    ),
];

/// Host wording for the OUTPUT battery (post-hoc: streaming already finished).
pub const GUARDRAIL_OUTPUT_HAZARDS: [(&str, &str); 4] = [
    (
        "instruction_echo",
        "hidden or injected instructions echoed into the agent's own output.",
    ),
    (
        "secret_disclosure",
        "credentials, API keys, tokens or other secrets disclosed in the output.",
    ),
    (
        "destructive_rendering",
        "destructive commands rendered in ready-to-run form without safety framing.",
    ),
    (
        "harmful_content",
        "harmful content generation that a coding assistant should not produce.",
    ),
];

/// Host severity-level wording (illustrative host policy, five levels).
pub const GUARDRAIL_SEVERITY_LEVELS: [&str; 5] =
    ["none", "minor", "moderate", "major", "critical"];

/// Prepares one guardrail battery over an ALREADY bounded+redacted excerpt.
/// The excerpt is bounded again defensively before it enters the state.
pub fn prepare_guardrail_battery(stage_prefix: &str, excerpt: &str) -> Option<PreparedGuidance> {
    let hazards: &[(&str, &str)] = match stage_prefix {
        "guardrails_input" => &GUARDRAIL_INPUT_HAZARDS,
        "guardrails_output" => &GUARDRAIL_OUTPUT_HAZARDS,
        _ => return None,
    };
    let timing = match stage_prefix {
        "guardrails_input" => TimingLabel::InputScreenPreventive,
        _ => TimingLabel::OutputPostHocAssessment,
    };
    let questions = guardrail_battery_questions(stage_prefix, hazards, GUARDRAIL_SEVERITY_LEVELS)?;
    let state = serde_json::json!({
        "screened_excerpt": bounded_excerpt(excerpt, MAX_GUIDANCE_TEXT_CHARS),
        "screening": "assessment_only",
        "enforced": false,
        "timing": timing.as_str(),
        "scope": "bounded excerpt only; never the full transcript; partial excerpts cannot claim full message coverage",
    });
    // Outgoing-payload verification (ROOT CONTRACT v7 cross-lane rule): the
    // battery is REFUSED (fail-open) unless it is provably visible and within
    // every native bound.
    if pi_jev::agent_guidance::verify_guidance_request(&state, &questions, None).is_err() {
        return None;
    }
    Some(PreparedGuidance { state, questions })
}

/// Host guardrail policy: fixed weights, thresholds and critical/support ids.
///
/// Full coverage of both batteries' hazard ids. A critical hazard
/// (`credential_exfiltration`, `destructive_commands`, `secret_disclosure`,
/// `destructive_rendering`) can never be masked by the composed value; the
/// personal-safety hazards route to support when decisive and support is
/// enabled.
pub fn guardrail_policy() -> GuardrailPolicy {
    let mut weights = std::collections::BTreeMap::new();
    for (id, weight) in [
        ("hidden_instructions", 0.30),
        ("credential_exfiltration", 0.30),
        ("destructive_commands", 0.25),
        ("personal_safety", 0.15),
        ("instruction_echo", 0.30),
        ("secret_disclosure", 0.30),
        ("destructive_rendering", 0.25),
        ("harmful_content", 0.15),
    ] {
        weights.insert(id.to_string(), weight);
    }
    let policy = GuardrailPolicy {
        thresholds: skill_guidance_thresholds(),
        severity_review: 0.40,
        severity_block: 0.75,
        weights,
        critical_ids: &[
            "credential_exfiltration",
            "destructive_commands",
            "secret_disclosure",
            "destructive_rendering",
        ],
        support_ids: &["personal_safety", "harmful_content"],
        support_enabled: true,
    };
    debug_assert!(policy.validate().is_ok());
    policy
}

/// Consumes raw decision records for one guardrail battery.
pub fn guardrail_assessment_from_raw(
    stage_prefix: &str,
    records: &[DecisionRecord],
    fresh: bool,
    budget_ok: bool,
) -> Option<GuardrailAssessment> {
    let hazards: &[(&str, &str)] = match stage_prefix {
        "guardrails_input" => &GUARDRAIL_INPUT_HAZARDS,
        "guardrails_output" => &GUARDRAIL_OUTPUT_HAZARDS,
        _ => return None,
    };
    let mut pairs: Vec<(String, f64)> = Vec::with_capacity(hazards.len());
    for (index, (hazard_id, _)) in hazards.iter().enumerate() {
        let question_id = format!("{stage_prefix}.{index}");
        let record = records
            .iter()
            .find(|record| record.question_id == question_id)?;
        match &record.answer {
            Answer::Noul { noul } if noul.is_finite() && (0.0..=1.0).contains(noul) => {
                pairs.push(((*hazard_id).to_string(), *noul));
            }
            _ => return None,
        }
    }
    let severity_question_id = format!("{stage_prefix}.{}", hazards.len());
    let severity = records
        .iter()
        .find(|record| record.question_id == severity_question_id)
        .and_then(|record| match &record.answer {
            Answer::Score { score, .. } if score.is_finite() && (0.0..=1.0).contains(score) => {
                Some(*score)
            }
            _ => None,
        });
    let timing = match stage_prefix {
        "guardrails_input" => TimingLabel::InputScreenPreventive,
        _ => TimingLabel::OutputPostHocAssessment,
    };
    guardrail_route(
        &pi_jev::agent_guidance::GuardrailInput {
            hazards: pairs,
            severity,
            fresh,
            cancelled: false,
            budget_ok,
        },
        &guardrail_policy(),
        timing,
    )
}

/// Renders the bounded, labelled advisory block for one guardrail assessment.
/// Truthful labels: assessment only, `enforced=false`, and for output the
/// explicit post-hoc timing. No safety guarantee, no verified-success claim.
pub fn format_guardrail_advisory(assessment: &GuardrailAssessment, stage_prefix: &str) -> String {
    let composed = assessment
        .composed
        .as_ref()
        .map(|weighted| format!("{:.2}", weighted.composed))
        .unwrap_or_else(|| "unavailable".to_string());
    let decisive = if assessment.decisive_flags.is_empty() {
        "none".to_string()
    } else {
        assessment.decisive_flags.join(", ")
    };
    let uncertain = assessment
        .uncertain_flags
        .iter()
        .chain(assessment.critical_flags.iter())
        .filter(|id| !assessment.decisive_flags.iter().any(|flag| flag == *id))
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let uncertain = if uncertain.is_empty() {
        "none".to_string()
    } else {
        uncertain
    };
    let timing_sentence = match assessment.timing {
        TimingLabel::OutputPostHocAssessment => {
            "This is a POST-HOC assessment: streaming already finished, so it is not preventive screening."
        }
        TimingLabel::InputScreenPreventive => {
            "This is an input-boundary screening assessment; it observes and records only."
        }
    };
    format!(
        "\n\n<jev_guardrail_assessment stage=\"{stage_prefix}\" route=\"{}\" enforced=\"false\" timing=\"{}\">\n\
Assessment only. No message, tool, model, permission or execution path was changed by this assessment. \
Composed severity: {composed} (host-fixed weights). Decisive hazards: {decisive}. Uncertain hazards: {uncertain}. \
This is not a safety guarantee and not a verified success claim. Timing label: {}. {timing_sentence}\n</jev_guardrail_assessment>",
        stage_prefix = stage_prefix,
        route = assessment.route.as_str(),
        timing = assessment.timing.as_str(),
        timing_sentence = timing_sentence,
        composed = composed,
        decisive = decisive,
        uncertain = uncertain,
    )
}

/// Bounded helper for hosts that want the roster ids a hint may name.
pub fn skill_catalog_ids(skills: &[Skill]) -> Vec<String> {
    skill_roster_view(skills)
        .0
        .into_iter()
        .map(|entry| entry.id)
        .collect()
}

/// Builds the optional second-pass fit question for a ranked skill id.
pub fn skill_fit_prepared(skill_id: &str) -> Option<PreparedQuestion> {
    pi_jev::agent_guidance::skill_fit_question(skill_id)
}

