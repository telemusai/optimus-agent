//! Applied bounded control policy for the opt-in full-jev profile.
//!
//! ROOT CONTRACT v1 (Applied control) + API-HANDOFF v1 (T2/B7/N-rules) govern this
//! module. It owns the TYPED acceptance of Jev control answers and the budget
//! policy for four categories that stay record-only in the legacy advisory API:
//! result sufficiency, first-pass verification, continue/stop/escalate (loop
//! control) and retry classification.
//!
//! Invariants (contract-backed):
//! - Legacy advisory behavior is untouched: this module never changes
//!   `active::evaluate_answer`, its categories, or its record-only guarantees.
//! - Applied effects are gated on full profile + feature + Active mode + a
//!   current accepted decision; any miss falls back to baseline.
//! - `complete`/`correlated`/`fresh` on [`ControlAcceptance`] are COMPUTED from
//!   trusted host facts (host request/turn/epoch/clock/policy generation), never
//!   copied from any server- or model-provided boolean, and stay internal.
//! - Verification truth never comes from tool transport success: no
//!   `VerificationObserved{Passed}` from a tool name plus `is_error == false`.
//!   `Verified`/`Failed` require an explicit [`CorrelatedVerificationEvidence`];
//!   without a reliable adapter the only paths are honest `Unknown`,
//!   `NotApplicable`, and the bounded verification-request behavior.
//! - Budgets are MAXIMA per real user task epoch (2 corrective feedback
//!   continuations including at most 1 verification request, 1 nonprogress
//!   correction, 2 retry vetoes) and are consumed at acceptance time.
//! - Retry veto is consult-only: it can SKIP a host-planned retry attempt within
//!   the existing ceilings, once per attempt, and only for
//!   {bad_arguments, fatal} at the high-impact floor. It can never add retries,
//!   never bypass host replay protection, and never touches credentials.
//! - Thresholds are proposed, unvalidated starting constants (contract wording),
//!   recorded in every acceptance for provenance; the model version is the
//!   server-reported response model and stays `unknown` when absent.

use std::time::{Duration, SystemTime};

use crate::active::{AnswerCandidate, FallbackReason};
use crate::config::JevMode;
use crate::observation::RetryFailureKind;
use crate::types::DecisionCategory;

/// Confidence floor for lower-impact applied effects (corrective feedback).
/// Proposed, unvalidated starting constant (ROOT CONTRACT v1).
pub const CONTROL_ACT_MIN_CONFIDENCE: f64 = 0.70;

/// Confidence floor for higher-impact applied effects: pause, escalate,
/// verification request, retry veto. Proposed, unvalidated starting constant.
pub const CONTROL_HIGH_IMPACT_MIN_CONFIDENCE: f64 = 0.85;

/// Maximum age of an accepted control answer at apply time.
pub const CONTROL_MAX_DECISION_AGE: Duration = Duration::from_secs(3);

/// Maximum corrective feedback continuations per real user task epoch
/// (shared budget; includes the verification request below).
pub const CONTROL_MAX_FEEDBACK: u8 = 2;

/// Maximum verification requests per epoch (must not exceed the shared budget).
pub const CONTROL_MAX_VERIFICATION_REQUESTS: u8 = 1;

/// Maximum nonprogress corrections per epoch (then truthful pause path).
pub const CONTROL_MAX_NONPROGRESS_CORRECTIONS: u8 = 1;

/// Maximum retry vetoes per epoch.
pub const CONTROL_MAX_RETRY_VETOES: u8 = 2;

/// Minimum consecutive identical turn signatures for a nonprogress candidate.
/// One empty/reasoning-only turn alone is never nonprogress (contract).
pub const CONTROL_NONPROGRESS_MIN_IDENTICAL_TURNS: u32 = 2;

/// Turn-signature window kept by the host for the predicate.
pub const CONTROL_NONPROGRESS_WINDOW: usize = 4;

/// The four categories with an applied host effect under full-jev.
pub const CONTROL_APPLIABLE_CATEGORIES: [DecisionCategory; 4] = [
    DecisionCategory::ResultSufficiency,
    DecisionCategory::FirstPassVerification,
    DecisionCategory::ContinueStopEscalate,
    DecisionCategory::RetryClassification,
];

/// Feature gates consumed by control acceptance (values come from resolved
/// settings truth, never from a decision payload).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlFeatures {
    pub result_sufficiency: bool,
    pub loop_control: bool,
    pub verification: bool,
    pub retry_classification: bool,
    /// Full-jev overlay active for this session (resolved settings truth).
    pub full_jev_active: bool,
}

/// Hard gate shared by every applied effect: full profile + Active mode.
pub fn control_gates_open(features: &ControlFeatures, mode: JevMode) -> bool {
    features.full_jev_active && mode.allows_active()
}

/// Boundary a control evaluation runs at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlBoundary {
    TurnEnd,
    AgentEnd,
    RetryDecision,
}

impl ControlBoundary {
    pub fn as_str(self) -> &'static str {
        match self {
            ControlBoundary::TurnEnd => "turn_end",
            ControlBoundary::AgentEnd => "agent_end",
            ControlBoundary::RetryDecision => "retry_decision",
        }
    }
}

/// Kind of applied host effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlEffectKind {
    /// Baseline continues; no effect (record-only).
    Continue,
    /// Queue ONE host-authored fixed corrective feedback continuation.
    Feedback(FeedbackKind),
    /// Stop further automatic work at the turn boundary. Never success.
    Pause(PauseReason),
    /// Pause plus an explicit user-attention flag.
    Escalate,
    /// Skip one host-planned retry attempt (within existing ceilings).
    RetryVeto(RetryFailureKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackKind {
    ResultGap,
    VerificationMissing,
    Nonprogress,
}

impl FeedbackKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FeedbackKind::ResultGap => "result_gap",
            FeedbackKind::VerificationMissing => "verification_missing",
            FeedbackKind::Nonprogress => "nonprogress",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseReason {
    NonprogressUncorrected,
    EscalateRecommended,
    VerificationUnconfirmed,
    BudgetExhausted,
}

impl PauseReason {
    pub fn as_str(self) -> &'static str {
        match self {
            PauseReason::NonprogressUncorrected => "nonprogress_uncorrected",
            PauseReason::EscalateRecommended => "escalate_recommended",
            PauseReason::VerificationUnconfirmed => "verification_unconfirmed",
            PauseReason::BudgetExhausted => "budget_exhausted",
        }
    }
}

/// Honest verification states. `Verified`/`Failed` can only be constructed
/// through [`CorrelatedVerificationEvidence`]; there is deliberately no
/// constructor from a tool name plus an error flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlVerificationState {
    /// No explicit evidence observed; ordinary answers stay here and are NOT
    /// forced into pause loops.
    Unknown,
    /// The task was assessed as not requiring tests/builds.
    NotApplicable,
    /// A verification request was delivered; outcome still unconfirmed.
    Unverified,
    /// Explicit correlated completed verification evidence (test/build/check
    /// outcome with source and scope). Not synthesizable in this batch.
    Verified(CorrelatedVerificationEvidence),
    /// Explicit correlated completed verification evidence that failed.
    Failed(CorrelatedVerificationEvidence),
}

/// Explicit correlated completed verification evidence. Every field must be
/// non-empty; the digests are host-computed fingerprints, never raw content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrelatedVerificationEvidence {
    pub source_kind: VerificationSourceKind,
    /// Fingerprint of the completed run that produced the outcome.
    pub run_digest: String,
    /// Fingerprint of the scope the outcome covers (bounded rendering).
    pub scope_digest: String,
    /// Fingerprint of the actual outcome text (bounded rendering).
    pub outcome_digest: String,
    /// RFC 3339 completion time of the verified run.
    pub completed_at: String,
    /// Task epoch the evidence correlates to.
    pub task_epoch_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationSourceKind {
    Test,
    Build,
    Check,
}

impl CorrelatedVerificationEvidence {
    /// Validated constructor; rejects empty fields so no caller can smuggle a
    /// `Passed` state from transport success alone.
    pub fn new(
        source_kind: VerificationSourceKind,
        run_digest: impl Into<String>,
        scope_digest: impl Into<String>,
        outcome_digest: impl Into<String>,
        completed_at: impl Into<String>,
        task_epoch_id: impl Into<String>,
    ) -> Option<Self> {
        let evidence = Self {
            source_kind,
            run_digest: run_digest.into(),
            scope_digest: scope_digest.into(),
            outcome_digest: outcome_digest.into(),
            completed_at: completed_at.into(),
            task_epoch_id: task_epoch_id.into(),
        };
        let sane = !evidence.run_digest.is_empty()
            && !evidence.scope_digest.is_empty()
            && !evidence.outcome_digest.is_empty()
            && !evidence.completed_at.is_empty()
            && !evidence.task_epoch_id.is_empty()
            && evidence.run_digest.len() <= 128
            && evidence.scope_digest.len() <= 128
            && evidence.outcome_digest.len() <= 128;
        if sane { Some(evidence) } else { None }
    }
}

/// Budget maxima per real user task epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlBudgets {
    pub feedback: u8,
    pub verification_requests: u8,
    pub nonprogress_corrections: u8,
    pub retry_vetoes: u8,
}

impl Default for ControlBudgets {
    fn default() -> Self {
        Self::maxima()
    }
}

impl ControlBudgets {
    /// Contract-approved maxima (ROOT CONTRACT v1).
    pub fn maxima() -> Self {
        Self {
            feedback: CONTROL_MAX_FEEDBACK,
            verification_requests: CONTROL_MAX_VERIFICATION_REQUESTS,
            nonprogress_corrections: CONTROL_MAX_NONPROGRESS_CORRECTIONS,
            retry_vetoes: CONTROL_MAX_RETRY_VETOES,
        }
    }
}

/// Which budget a consumption draws from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlBudgetKind {
    Feedback(FeedbackKind),
    NonprogressCorrection,
    RetryVeto,
}

impl ControlBudgetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ControlBudgetKind::Feedback(FeedbackKind::ResultGap) => "feedback_result_gap",
            ControlBudgetKind::Feedback(FeedbackKind::VerificationMissing) => {
                "feedback_verification_missing"
            }
            ControlBudgetKind::Feedback(FeedbackKind::Nonprogress) => "feedback_nonprogress",
            ControlBudgetKind::NonprogressCorrection => "nonprogress_correction",
            ControlBudgetKind::RetryVeto => "retry_veto",
        }
    }
}

/// Immutable budget snapshot carried on every acceptance and refusal record.
/// `available: false` marks accounting that could not be trusted (missing,
/// corrupt, oversized, or failed durable ledger). Such a snapshot carries
/// ZERO headroom: new control effects are refused, the ordinary baseline
/// continues, and no maxima are ever synthesized from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlBudgetSnapshot {
    pub epoch_id: String,
    pub feedback_remaining: u8,
    pub verification_remaining: u8,
    pub nonprogress_remaining: u8,
    pub veto_remaining: u8,
    /// Trusted durable accounting backs this snapshot. Only `true` snapshots
    /// may authorize effects; the pure gate refuses `false` snapshots.
    pub available: bool,
}

impl ControlBudgetSnapshot {
    /// Zero-headroom snapshot for untrusted accounting (fail-closed). Fresh
    /// maxima exist ONLY through a committed real-user-task initialization.
    pub fn zero_headroom(epoch_id: impl Into<String>) -> Self {
        Self {
            epoch_id: epoch_id.into(),
            feedback_remaining: 0,
            verification_remaining: 0,
            nonprogress_remaining: 0,
            veto_remaining: 0,
            available: false,
        }
    }

    /// Fresh maxima for a committed real-user-task epoch (trusted).
    pub fn fresh_maxima(epoch_id: impl Into<String>) -> Self {
        let maxima = ControlBudgets::maxima();
        Self {
            epoch_id: epoch_id.into(),
            feedback_remaining: maxima.feedback,
            verification_remaining: maxima.verification_requests,
            nonprogress_remaining: maxima.nonprogress_corrections,
            veto_remaining: maxima.retry_vetoes,
            available: true,
        }
    }
}

/// Provenance recorded with every applied control effect (truthful telemetry).
#[derive(Debug, Clone, PartialEq)]
pub struct ControlProvenance {
    /// Server-reported response model; `None` stays `unknown` in records.
    pub response_model: Option<String>,
    /// Prompt version of the question set (host constant).
    pub prompt_version: String,
    /// Exact thresholds used for this acceptance.
    pub act_min_confidence: f64,
    pub high_impact_min_confidence: f64,
    /// Full-jev overlay stamp at decision time (host truth).
    pub full_jev_stamp: String,
}

/// Trusted host facts used to COMPUTE the typed acceptance booleans. Every
/// value comes from host state, never from the decision payload.
#[derive(Debug, Clone)]
pub struct HostControlFacts {
    pub now: SystemTime,
    pub session_id: String,
    pub turn: u64,
    /// Question ids the host actually asked at this boundary.
    pub expected_question_ids: Vec<String>,
    /// Current decision policy generation (settings truth).
    pub policy_generation: String,
    /// Current full-jev overlay stamp (settings truth).
    pub full_jev_stamp: String,
    /// Host prompt version for the control question set.
    pub prompt_version: String,
    /// Real user task epoch id for this session.
    pub epoch_id: String,
    /// Host request id that carried the questions.
    pub request_id: String,
}

/// Typed applied acceptance (API-HANDOFF T2). The `complete`, `correlated`
/// and `fresh` fields are computed by [`evaluate_control_answer`] from
/// [`HostControlFacts`] versus the answer candidate; they are never accepted
/// from the wire.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlAcceptance {
    pub category: DecisionCategory,
    pub question_id: String,
    pub effect: ControlEffectKind,
    /// Answer is a well-formed in-criteria value with a finite confidence.
    pub complete: bool,
    /// Answer belongs to this session/turn/epoch and an expected question.
    pub correlated: bool,
    /// Within the decision age bound and the current policy generation.
    pub fresh: bool,
    pub confidence: f64,
    pub value: String,
    pub budget: ControlBudgetSnapshot,
    pub provenance: ControlProvenance,
}

/// Why an answer was not applied. Extends — never bypasses — the legacy
/// `FallbackReason` vocabulary for transport/acceptance misses.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlRefusal {
    /// Legacy reason (mode, no answer, missing/low confidence, invalid value,
    /// stale, unavailable, ...).
    Baseline(FallbackReason),
    /// Gates closed: full profile off, feature off, or mode not Active.
    GatesClosed(&'static str),
    /// The named budget for this epoch is exhausted.
    BudgetExhausted(ControlBudgetKind),
    /// Host prefilter said no decision was needed at this boundary.
    TriggerNotMet(&'static str),
    /// Another continuation was already pending; never double-continue.
    ContinuationPending,
    /// Explicit user stop wins; no control runs.
    ExplicitStop,
    /// Boundary does not own this category.
    WrongBoundary,
    /// Durable task accounting is missing, corrupt, oversized, or failed;
    /// new control effects are refused (fail-closed, never minted).
    AccountingUnavailable(&'static str),
}

impl ControlRefusal {
    pub fn as_str(&self) -> &'static str {
        match self {
            ControlRefusal::Baseline(reason) => reason.as_str(),
            ControlRefusal::GatesClosed(reason) => reason,
            ControlRefusal::BudgetExhausted(kind) => match kind {
                ControlBudgetKind::Feedback(_) => "budget_exhausted_feedback",
                ControlBudgetKind::NonprogressCorrection => "budget_exhausted_nonprogress",
                ControlBudgetKind::RetryVeto => "budget_exhausted_veto",
            },
            ControlRefusal::TriggerNotMet(reason) => reason,
            ControlRefusal::ContinuationPending => "continuation_pending",
            ControlRefusal::ExplicitStop => "explicit_stop",
            ControlRefusal::WrongBoundary => "wrong_boundary",
            ControlRefusal::AccountingUnavailable(_) => "accounting_unavailable",
        }
    }
}

/// Result of one control evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlVerdict {
    Applied(ControlAcceptance),
    Refused(ControlRefusal),
}

impl ControlVerdict {
    pub fn applied(&self) -> Option<&ControlAcceptance> {
        match self {
            ControlVerdict::Applied(acceptance) => Some(acceptance),
            ControlVerdict::Refused(_) => None,
        }
    }
}

/// Policy of starting constants for control acceptance.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlPolicy {
    pub act_min_confidence: f64,
    pub high_impact_min_confidence: f64,
    pub max_decision_age: Duration,
}

impl Default for ControlPolicy {
    fn default() -> Self {
        Self {
            act_min_confidence: CONTROL_ACT_MIN_CONFIDENCE,
            high_impact_min_confidence: CONTROL_HIGH_IMPACT_MIN_CONFIDENCE,
            max_decision_age: CONTROL_MAX_DECISION_AGE,
        }
    }
}

/// Bounded content signature of one turn, as computed by the host. Digests are
/// opaque fingerprints over bounded renderings; raw content never lands here.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TurnSignature {
    pub tool_calls_digest: String,
    pub text_digest: String,
    pub results_digest: String,
    /// Host marker: the turn produced evidence not seen in prior turns
    /// (new tool result content, new text, new files).
    pub has_new_evidence: bool,
}

/// Nonprogress verdict from content signatures (contract): repeated tool NAME
/// alone never counts; identical signatures over at least
/// [`CONTROL_NONPROGRESS_MIN_IDENTICAL_TURNS`] consecutive turns with no new
/// evidence do. A single empty/reasoning-only turn is never nonprogress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NonprogressVerdict {
    None,
    Candidate { identical_turns: u32 },
}

pub fn nonprogress_verdict(signatures: &[TurnSignature]) -> NonprogressVerdict {
    if signatures.len() < CONTROL_NONPROGRESS_MIN_IDENTICAL_TURNS as usize {
        return NonprogressVerdict::None;
    }
    // Last-N window over the bounded signature list. The list itself is
    // bounded by the host (CONTROL_NONPROGRESS_WINDOW); this slice is a
    // borrow, never a collected temporary.
    let start = signatures.len().saturating_sub(CONTROL_NONPROGRESS_WINDOW);
    let window: &[TurnSignature] = &signatures[start..];
    let last = match window.last() {
        Some(last) => last,
        None => return NonprogressVerdict::None,
    };
    if last.has_new_evidence {
        return NonprogressVerdict::None;
    }
    let mut identical: u32 = 1;
    for prior in window.iter().rev().skip(1) {
        if prior.tool_calls_digest == last.tool_calls_digest
            && prior.text_digest == last.text_digest
            && prior.results_digest == last.results_digest
            && !prior.has_new_evidence
        {
            identical += 1;
        } else {
            break;
        }
    }
    if identical >= CONTROL_NONPROGRESS_MIN_IDENTICAL_TURNS {
        NonprogressVerdict::Candidate { identical_turns: identical }
    } else {
        NonprogressVerdict::None
    }
}

/// Evaluate one control answer at one boundary.
///
/// Order of checks is the contract: gates, boundary ownership, completeness
/// (value in criteria + finite confidence), correlation (expected question,
/// session/turn/epoch), freshness (age + policy generation), then the typed
/// effect mapping and budget availability. The FIRST failing check names the
/// single recorded refusal reason.
pub fn evaluate_control_answer(
    policy: &ControlPolicy,
    features: &ControlFeatures,
    mode: JevMode,
    boundary: ControlBoundary,
    candidate: &AnswerCandidate,
    facts: &HostControlFacts,
    budget: &ControlBudgetSnapshot,
) -> ControlVerdict {
    use ControlEffectKind as Effect;
    use ControlRefusal as Refusal;

    if facts.session_id.is_empty() {
        return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::NoAnswer));
    }
    if !control_gates_open(features, mode) {
        return ControlVerdict::Refused(Refusal::GatesClosed("gates_closed"));
    }
    // Fail-closed accounting gate (ROOT CONTRACT: budgets are durable
    // per-real-task maxima). Untrusted accounting never authorizes effects;
    // the baseline continues and no maxima are synthesized.
    if !budget.available {
        return ControlVerdict::Refused(Refusal::AccountingUnavailable(
            "accounting_unavailable",
        ));
    }
    if !CONTROL_APPLIABLE_CATEGORIES.contains(&candidate.category) {
        return ControlVerdict::Refused(Refusal::WrongBoundary);
    }
    let feature_on = match candidate.category {
        DecisionCategory::ResultSufficiency => features.result_sufficiency,
        DecisionCategory::FirstPassVerification => features.verification,
        DecisionCategory::ContinueStopEscalate => features.loop_control,
        DecisionCategory::RetryClassification => features.retry_classification,
        _ => false,
    };
    if !feature_on {
        return ControlVerdict::Refused(Refusal::GatesClosed("feature_disabled"));
    }
    let boundary_owns = match candidate.category {
        DecisionCategory::ResultSufficiency => boundary == ControlBoundary::AgentEnd,
        DecisionCategory::FirstPassVerification => boundary == ControlBoundary::AgentEnd,
        DecisionCategory::ContinueStopEscalate => {
            boundary == ControlBoundary::TurnEnd || boundary == ControlBoundary::AgentEnd
        }
        DecisionCategory::RetryClassification => {
            boundary == ControlBoundary::RetryDecision || boundary == ControlBoundary::AgentEnd
        }
        _ => false,
    };
    if !boundary_owns {
        return ControlVerdict::Refused(Refusal::WrongBoundary);
    }
    let Some(raw_value) = candidate.value.as_deref() else {
        return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::NoAnswer));
    };
    let value = raw_value.trim();
    if value.is_empty() || value.chars().count() > crate::active::MAX_VALUE_CHARS {
        return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::InvalidValue));
    }
    let Some(confidence) = candidate.confidence else {
        return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::MissingConfidence));
    };
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::LowConfidence));
    }
    // Correlation is computed from trusted host facts, never from the answer:
    // the answer must belong to the question set THIS host asked, on the host's
    // own current request, at the current turn, for the current task epoch.
    // (Overlay/mode staleness stays enforced by the bridge `can_apply`
    // re-check at apply time; this pure gate adds the request-id binding.)
    let correlated = facts.expected_question_ids.iter().any(|id| id == &candidate.question_id)
        && candidate.turn == facts.turn
        && !facts.request_id.is_empty()
        && candidate.request_id == facts.request_id
        && facts.epoch_id == budget.epoch_id;
    if !correlated {
        return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::NoAnswer));
    }
    // Freshness: decided within the age bound; a future-stamped decision is
    // not usable evidence.
    let fresh = match facts.now.duration_since(candidate.decided_at) {
        Ok(age) => age <= policy.max_decision_age,
        Err(_) => false,
    };
    if !fresh {
        return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::Stale));
    }

    let (effect, floor, budget_kind) = match (candidate.category, boundary, value) {
        (DecisionCategory::ResultSufficiency, ControlBoundary::AgentEnd, value) => {
            let assessment = value.to_ascii_lowercase();
            match assessment.as_str() {
                "sufficient" | "complete" => (Effect::Continue, policy.act_min_confidence, None),
                "insufficient" | "partial" | "failed" => (
                    Effect::Feedback(FeedbackKind::ResultGap),
                    policy.act_min_confidence,
                    Some(ControlBudgetKind::Feedback(FeedbackKind::ResultGap)),
                ),
                _ => return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::InvalidValue)),
            }
        }
        (DecisionCategory::FirstPassVerification, ControlBoundary::AgentEnd, value) => {
            match value.to_ascii_lowercase().as_str() {
                "verify" | "rerun" => (
                    Effect::Feedback(FeedbackKind::VerificationMissing),
                    policy.high_impact_min_confidence,
                    Some(ControlBudgetKind::Feedback(FeedbackKind::VerificationMissing)),
                ),
                "none" => (Effect::Continue, policy.act_min_confidence, None),
                "escalate" => (Effect::Escalate, policy.high_impact_min_confidence, None),
                _ => return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::InvalidValue)),
            }
        }
        (DecisionCategory::ContinueStopEscalate, ControlBoundary::TurnEnd, value) => {
            match value.to_ascii_lowercase().as_str() {
                "continue" => (Effect::Continue, policy.act_min_confidence, None),
                "stop" => (
                    Effect::Pause(PauseReason::NonprogressUncorrected),
                    policy.high_impact_min_confidence,
                    None,
                ),
                "escalate" => (Effect::Escalate, policy.high_impact_min_confidence, None),
                _ => return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::InvalidValue)),
            }
        }
        (DecisionCategory::ContinueStopEscalate, ControlBoundary::AgentEnd, value) => {
            match value.to_ascii_lowercase().as_str() {
                "continue" => (Effect::Continue, policy.act_min_confidence, None),
                // At AgentEnd the loop already stopped; `stop` records only.
                "stop" => (Effect::Continue, policy.act_min_confidence, None),
                "escalate" => (Effect::Escalate, policy.high_impact_min_confidence, None),
                _ => return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::InvalidValue)),
            }
        }
        (DecisionCategory::RetryClassification, ControlBoundary::RetryDecision, value) => {
            let kind = RetryFailureKind::ALL
                .into_iter()
                .find(|kind| kind.as_str() == value.to_ascii_lowercase());
            match kind {
                Some(kind @ (RetryFailureKind::BadArguments | RetryFailureKind::Fatal)) => (
                    Effect::RetryVeto(kind),
                    policy.high_impact_min_confidence,
                    Some(ControlBudgetKind::RetryVeto),
                ),
                // transient/rate_limited/provider_failure/tool_failure/permission/unknown
                // fall back: no veto, baseline retry policy untouched.
                Some(_) => (Effect::Continue, policy.act_min_confidence, None),
                None => {
                    return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::InvalidValue))
                }
            }
        }
        (DecisionCategory::RetryClassification, ControlBoundary::AgentEnd, value) => {
            let kind = RetryFailureKind::ALL
                .into_iter()
                .find(|kind| kind.as_str() == value.to_ascii_lowercase());
            // At AgentEnd a classification is recorded only; the veto consult
            // happens at the retry decision boundary.
            match kind {
                Some(_) => (Effect::Continue, policy.act_min_confidence, None),
                None => {
                    return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::InvalidValue))
                }
            }
        }
        _ => return ControlVerdict::Refused(Refusal::WrongBoundary),
    };
    if confidence < floor {
        return ControlVerdict::Refused(Refusal::Baseline(FallbackReason::LowConfidence));
    }
    if let Some(kind) = budget_kind {
        if !budget.allows(kind) {
            return ControlVerdict::Refused(Refusal::BudgetExhausted(kind));
        }
    }
    ControlVerdict::Applied(ControlAcceptance {
        category: candidate.category,
        question_id: candidate.question_id.clone(),
        effect,
        complete: true,
        correlated,
        fresh,
        confidence,
        value: value.to_string(),
        budget: budget.clone(),
        provenance: ControlProvenance {
            response_model: candidate.response_model.clone(),
            prompt_version: facts.prompt_version.clone(),
            act_min_confidence: policy.act_min_confidence,
            high_impact_min_confidence: policy.high_impact_min_confidence,
            full_jev_stamp: facts.full_jev_stamp.clone(),
        },
    })
}

impl ControlBudgetSnapshot {
    /// Whether the named budget still has headroom.
    pub fn allows(&self, kind: ControlBudgetKind) -> bool {
        match kind {
            ControlBudgetKind::Feedback(FeedbackKind::VerificationMissing) => {
                self.feedback_remaining > 0 && self.verification_remaining > 0
            }
            ControlBudgetKind::Feedback(_) => self.feedback_remaining > 0,
            ControlBudgetKind::NonprogressCorrection => self.nonprogress_remaining > 0,
            ControlBudgetKind::RetryVeto => self.veto_remaining > 0,
        }
    }

    /// Saturating consumption of one budget unit; the caller persists the
    /// returned snapshot so consumption survives eviction and restarts.
    pub fn consume(&self, kind: ControlBudgetKind) -> ControlBudgetSnapshot {
        let mut next = self.clone();
        match kind {
            ControlBudgetKind::Feedback(FeedbackKind::VerificationMissing) => {
                next.feedback_remaining = next.feedback_remaining.saturating_sub(1);
                next.verification_remaining = next.verification_remaining.saturating_sub(1);
            }
            ControlBudgetKind::Feedback(_) => {
                next.feedback_remaining = next.feedback_remaining.saturating_sub(1);
            }
            ControlBudgetKind::NonprogressCorrection => {
                next.nonprogress_remaining = next.nonprogress_remaining.saturating_sub(1);
            }
            ControlBudgetKind::RetryVeto => {
                next.veto_remaining = next.veto_remaining.saturating_sub(1);
            }
        }
        next
    }
}

/// Honest result-sufficiency classification for the AgentEnd gate, combining
/// the two sufficiency questions when both are present. `uncertain`/`unknown`
/// never escalate to feedback on their own (no guesses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SufficiencyVerdict {
    Sufficient,
    Insufficient,
    Unknown,
}

pub fn combine_sufficiency(primary: Option<&str>, coverage: Option<&str>) -> SufficiencyVerdict {
    let insufficient = |value: &str| matches!(value, "insufficient" | "partial" | "failed");
    let sufficient = |value: &str| matches!(value, "sufficient" | "complete");
    match (primary, coverage) {
        (Some(primary), Some(coverage)) => {
            let primary = primary.to_ascii_lowercase();
            let coverage = coverage.to_ascii_lowercase();
            if insufficient(&primary) || insufficient(&coverage) {
                SufficiencyVerdict::Insufficient
            } else if sufficient(&primary) && sufficient(&coverage) {
                SufficiencyVerdict::Sufficient
            } else {
                SufficiencyVerdict::Unknown
            }
        }
        (Some(only), None) | (None, Some(only)) => {
            let only = only.to_ascii_lowercase();
            if insufficient(&only) {
                SufficiencyVerdict::Insufficient
            } else if sufficient(&only) {
                SufficiencyVerdict::Sufficient
            } else {
                SufficiencyVerdict::Unknown
            }
        }
        (None, None) => SufficiencyVerdict::Unknown,
    }
}

/// Verification-need classification for the AgentEnd gate. `NotApplicable`
/// requires an accepted `none` recommendation; `unknown` stays `unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationNeed {
    NotRequired,
    Request,
    Escalate,
    Unknown,
}

pub fn verification_need(recommendation: Option<&str>) -> VerificationNeed {
    match recommendation.map(|value| value.to_ascii_lowercase()) {
        Some(value) if value == "none" => VerificationNeed::NotRequired,
        Some(value) if value == "verify" || value == "rerun" => VerificationNeed::Request,
        Some(value) if value == "escalate" => VerificationNeed::Escalate,
        _ => VerificationNeed::Unknown,
    }
}
