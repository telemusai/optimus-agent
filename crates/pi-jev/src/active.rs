//! Jev Active acceptance policy and applied-change records.
//!
//! This module owns one question: may an answer that System One returned be
//! applied to the host application's own behavior, and what exactly was
//! applied. It is deliberately framework-neutral. It does not know how a
//! decision is applied (the host owns the target, for example a provider
//! request body), holds no credentials, performs no network work, and never
//! mutates host state itself.
//!
//! Compare mode does not use this module: a Compare answer is always recorded
//! with `applied: false` and never reaches an acceptance decision.

use std::collections::BTreeSet;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::types::DecisionCategory;

/// Categories an Active deployment may apply by default.
///
/// Both have a bounded, reversible effect at the provider-request boundary:
/// `ToolRequirement` can withdraw the tool catalog for one request, and
/// `Complexity` can move one step on the reasoning-effort ladder. Categories
/// with no reversible host-side effect stay record-only even in Active mode.
pub const DEFAULT_APPLIABLE_CATEGORIES: [DecisionCategory; 2] = [
    DecisionCategory::ToolRequirement,
    DecisionCategory::Complexity,
];

/// Longest accepted answer value. Longer values are refused rather than
/// truncated, because a partial value is not the answer System One returned.
pub const MAX_VALUE_CHARS: usize = 64;

/// Longest retained rendering of a changed value in a record.
pub const MAX_EFFECT_CHARS: usize = 120;

/// One field the host changed because of an accepted decision.
///
/// The field name is opaque to this crate: only the host knows its target
/// vocabulary. Values are bounded by [`MAX_EFFECT_CHARS`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedEffect {
    pub field: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

impl AppliedEffect {
    pub fn new(field: impl Into<String>, from: Option<String>, to: Option<String>) -> Self {
        Self {
            field: field.into(),
            from: from.map(|value| bound(value, MAX_EFFECT_CHARS)),
            to: to.map(|value| bound(value, MAX_EFFECT_CHARS)),
        }
    }
}

/// Bound a rendering to `max` characters, marking truncation.
fn bound(value: String, max: usize) -> String {
    if value.chars().count() <= max {
        return value;
    }
    let mut truncated: String = value.chars().take(max).collect();
    truncated.push_str("...");
    truncated
}

/// Active-mode acceptance policy.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationPolicy {
    /// Categories the operator allows to be applied. Empty means
    /// [`DEFAULT_APPLIABLE_CATEGORIES`].
    pub enabled_categories: BTreeSet<DecisionCategory>,
    /// Minimum answer confidence. An answer without a confidence is refused
    /// when this is set: "unknown" is not "high".
    pub min_confidence: f64,
    /// How long an accepted decision stays usable.
    pub max_decision_age: Duration,
}

impl Default for ActivationPolicy {
    fn default() -> Self {
        Self {
            enabled_categories: BTreeSet::new(),
            min_confidence: 0.7,
            max_decision_age: Duration::from_secs(3),
        }
    }
}

impl ActivationPolicy {
    /// Categories this policy may apply.
    pub fn appliable(&self) -> BTreeSet<DecisionCategory> {
        if self.enabled_categories.is_empty() {
            DEFAULT_APPLIABLE_CATEGORIES.into_iter().collect()
        } else {
            self.enabled_categories.clone()
        }
    }

    /// True when this category could be applied under this policy. A category
    /// outside the set stays record-only, which is not an error.
    pub fn allows(&self, category: DecisionCategory) -> bool {
        self.appliable().contains(&category)
    }
}

/// Why an answer was not applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// The effective mode was not Active.
    ModeNotActive,
    /// The category has no reversible host-side effect.
    CategoryNotAppliable,
    /// The operator disabled this category.
    CategoryDisabled,
    /// No answer came back for the question.
    NoAnswer,
    /// The answer carried no confidence.
    MissingConfidence,
    /// The answer confidence was below the policy threshold.
    LowConfidence,
    /// The answer value was empty or longer than [`MAX_VALUE_CHARS`].
    InvalidValue,
    /// The accepted decision was older than the policy allows.
    Stale,
    /// The transport failed, timed out or was cancelled.
    Unavailable,
}

impl FallbackReason {
    /// Stable snake_case id used in records.
    pub fn as_str(self) -> &'static str {
        match self {
            FallbackReason::ModeNotActive => "mode_not_active",
            FallbackReason::CategoryNotAppliable => "category_not_appliable",
            FallbackReason::CategoryDisabled => "category_disabled",
            FallbackReason::NoAnswer => "no_answer",
            FallbackReason::MissingConfidence => "missing_confidence",
            FallbackReason::LowConfidence => "low_confidence",
            FallbackReason::InvalidValue => "invalid_value",
            FallbackReason::Stale => "stale",
            FallbackReason::Unavailable => "unavailable",
        }
    }
}

/// One accepted decision, ready for the host to apply.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveDecision {
    pub category: DecisionCategory,
    pub question_id: String,
    pub value: String,
    pub confidence: f64,
    pub response_model: Option<String>,
    pub request_id: String,
    pub turn: u64,
    pub decided_at: SystemTime,
}

impl ActiveDecision {
    /// True when the decision is still within the policy's age bound.
    pub fn is_fresh(&self, now: SystemTime, policy: &ActivationPolicy) -> bool {
        match now.duration_since(self.decided_at) {
            Ok(age) => age <= policy.max_decision_age,
            // A decision stamped in the future is not usable evidence.
            Err(_) => false,
        }
    }
}

/// Result of the acceptance check for one answer.
#[derive(Debug, Clone, PartialEq)]
pub enum Acceptance {
    Accepted(Box<ActiveDecision>),
    Fallback(FallbackReason),
}

impl Acceptance {
    pub fn accepted(&self) -> Option<&ActiveDecision> {
        match self {
            Acceptance::Accepted(decision) => Some(decision),
            Acceptance::Fallback(_) => None,
        }
    }

    /// Reason code for a record: `None` when the answer was accepted.
    pub fn fallback_reason(&self) -> Option<FallbackReason> {
        match self {
            Acceptance::Accepted(_) => None,
            Acceptance::Fallback(reason) => Some(*reason),
        }
    }
}

/// Shape of one answer as it arrives from the transport.
#[derive(Debug, Clone, PartialEq)]
pub struct AnswerCandidate {
    pub category: DecisionCategory,
    pub question_id: String,
    pub value: Option<String>,
    pub confidence: Option<f64>,
    pub response_model: Option<String>,
    pub request_id: String,
    pub turn: u64,
    pub decided_at: SystemTime,
}

/// Decide whether one answer may be applied.
///
/// Order of checks is the contract: category eligibility, then value, then
/// confidence, then freshness. The first failing check is the reported reason,
/// so a record always names the single reason that stopped application.
pub fn evaluate_answer(
    policy: &ActivationPolicy,
    mode: crate::config::JevMode,
    candidate: &AnswerCandidate,
    now: SystemTime,
) -> Acceptance {
    if mode != crate::config::JevMode::Active {
        return Acceptance::Fallback(FallbackReason::ModeNotActive);
    }
    if !DEFAULT_APPLIABLE_CATEGORIES.contains(&candidate.category) {
        return Acceptance::Fallback(FallbackReason::CategoryNotAppliable);
    }
    if !policy.enabled_categories.is_empty() && !policy.allows(candidate.category) {
        return Acceptance::Fallback(FallbackReason::CategoryDisabled);
    }
    let Some(raw_value) = candidate.value.as_deref() else {
        return Acceptance::Fallback(FallbackReason::NoAnswer);
    };
    let value = raw_value.trim();
    if value.is_empty() || value.chars().count() > MAX_VALUE_CHARS {
        return Acceptance::Fallback(FallbackReason::InvalidValue);
    }
    let Some(confidence) = candidate.confidence else {
        return Acceptance::Fallback(FallbackReason::MissingConfidence);
    };
    if !confidence.is_finite() || confidence < policy.min_confidence {
        return Acceptance::Fallback(FallbackReason::LowConfidence);
    }
    let decision = ActiveDecision {
        category: candidate.category,
        question_id: candidate.question_id.clone(),
        value: value.to_string(),
        confidence,
        response_model: candidate.response_model.clone(),
        request_id: candidate.request_id.clone(),
        turn: candidate.turn,
        decided_at: candidate.decided_at,
    };
    if !decision.is_fresh(now, policy) {
        return Acceptance::Fallback(FallbackReason::Stale);
    }
    Acceptance::Accepted(Box::new(decision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::JevMode;

    fn candidate(category: DecisionCategory, value: Option<&str>, confidence: Option<f64>) -> AnswerCandidate {
        AnswerCandidate {
            category,
            question_id: format!("{}.0", category.as_str()),
            value: value.map(str::to_string),
            confidence,
            response_model: Some("jev-1.13.0".to_string()),
            request_id: "jev-test".to_string(),
            turn: 3,
            decided_at: SystemTime::UNIX_EPOCH,
        }
    }

    fn now() -> SystemTime {
        SystemTime::UNIX_EPOCH
    }

    #[test]
    fn accepted_when_category_value_and_confidence_pass() {
        let policy = ActivationPolicy::default();
        let acceptance = evaluate_answer(
            &policy,
            JevMode::Active,
            &candidate(DecisionCategory::ToolRequirement, Some("none"), Some(0.9)),
            now(),
        );
        let decision = acceptance.accepted().expect("accepted");
        assert_eq!(decision.value, "none");
        assert_eq!(decision.category, DecisionCategory::ToolRequirement);
        assert!(acceptance.fallback_reason().is_none());
    }

    #[test]
    fn compare_never_accepts() {
        let policy = ActivationPolicy::default();
        let acceptance = evaluate_answer(
            &policy,
            JevMode::Compare,
            &candidate(DecisionCategory::ToolRequirement, Some("none"), Some(1.0)),
            now(),
        );
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::ModeNotActive));
    }

    #[test]
    fn off_never_accepts() {
        let policy = ActivationPolicy::default();
        let acceptance = evaluate_answer(
            &policy,
            JevMode::Off,
            &candidate(DecisionCategory::Complexity, Some("low"), Some(1.0)),
            now(),
        );
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::ModeNotActive));
    }

    #[test]
    fn category_without_reversible_effect_is_refused() {
        let policy = ActivationPolicy::default();
        for category in [
            DecisionCategory::MemoryRelevance,
            DecisionCategory::ContextRelevance,
            DecisionCategory::SubagentRequirement,
            DecisionCategory::SubagentModelRouting,
            DecisionCategory::ContinueStopEscalate,
            DecisionCategory::ResultSufficiency,
            DecisionCategory::FirstPassVerification,
            DecisionCategory::TaskClassification,
            DecisionCategory::ToolCandidates,
        ] {
            let acceptance = evaluate_answer(
                &policy,
                JevMode::Active,
                &candidate(category, Some("1"), Some(1.0)),
                now(),
            );
            assert_eq!(
                acceptance,
                Acceptance::Fallback(FallbackReason::CategoryNotAppliable),
                "{category:?} must stay record-only"
            );
        }
    }

    #[test]
    fn operator_can_disable_a_default_category() {
        let policy = ActivationPolicy {
            enabled_categories: [DecisionCategory::ToolRequirement].into_iter().collect(),
            ..ActivationPolicy::default()
        };
        let acceptance = evaluate_answer(
            &policy,
            JevMode::Active,
            &candidate(DecisionCategory::Complexity, Some("low"), Some(1.0)),
            now(),
        );
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::CategoryDisabled));
    }

    #[test]
    fn missing_confidence_is_not_high_confidence() {
        let policy = ActivationPolicy::default();
        let acceptance = evaluate_answer(
            &policy,
            JevMode::Active,
            &candidate(DecisionCategory::Complexity, Some("low"), None),
            now(),
        );
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::MissingConfidence));
    }

    #[test]
    fn below_threshold_is_refused() {
        let policy = ActivationPolicy::default();
        let acceptance = evaluate_answer(
            &policy,
            JevMode::Active,
            &candidate(DecisionCategory::Complexity, Some("low"), Some(0.69)),
            now(),
        );
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::LowConfidence));
    }

    #[test]
    fn non_finite_confidence_is_refused() {
        let policy = ActivationPolicy::default();
        let acceptance = evaluate_answer(
            &policy,
            JevMode::Active,
            &candidate(DecisionCategory::Complexity, Some("low"), Some(f64::NAN)),
            now(),
        );
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::LowConfidence));
    }

    #[test]
    fn empty_values_are_refused() {
        let policy = ActivationPolicy::default();
        for value in [Some("   "), Some("")] {
            let acceptance = evaluate_answer(
                &policy,
                JevMode::Active,
                &candidate(DecisionCategory::ToolRequirement, value, Some(1.0)),
                now(),
            );
            assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::InvalidValue));
        }
    }

    #[test]
    fn absent_answer_is_refused() {
        let policy = ActivationPolicy::default();
        let acceptance = evaluate_answer(
            &policy,
            JevMode::Active,
            &candidate(DecisionCategory::ToolRequirement, None, Some(1.0)),
            now(),
        );
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::NoAnswer));
    }

    #[test]
    fn oversized_value_is_refused() {
        let policy = ActivationPolicy::default();
        let long = "x".repeat(MAX_VALUE_CHARS + 1);
        let acceptance = evaluate_answer(
            &policy,
            JevMode::Active,
            &candidate(DecisionCategory::ToolRequirement, Some(&long), Some(1.0)),
            now(),
        );
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::InvalidValue));
    }

    #[test]
    fn stale_decision_is_refused() {
        let policy = ActivationPolicy {
            max_decision_age: Duration::from_secs(1),
            ..ActivationPolicy::default()
        };
        let mut stale = candidate(DecisionCategory::Complexity, Some("low"), Some(1.0));
        stale.decided_at = SystemTime::UNIX_EPOCH - Duration::from_secs(60);
        let acceptance = evaluate_answer(&policy, JevMode::Active, &stale, now());
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::Stale));
    }

    #[test]
    fn future_stamp_is_refused() {
        let policy = ActivationPolicy::default();
        let mut future = candidate(DecisionCategory::Complexity, Some("low"), Some(1.0));
        future.decided_at = SystemTime::UNIX_EPOCH + Duration::from_secs(60);
        let acceptance = evaluate_answer(&policy, JevMode::Active, &future, now());
        assert_eq!(acceptance, Acceptance::Fallback(FallbackReason::Stale));
    }

    #[test]
    fn effects_are_bounded() {
        let long = "y".repeat(500);
        let effect = AppliedEffect::new("tools", Some(long.clone()), None);
        assert_eq!(effect.from.as_deref().map(|value| value.chars().count()), Some(MAX_EFFECT_CHARS + 3));
        assert!(effect.to.is_none());
    }
}
