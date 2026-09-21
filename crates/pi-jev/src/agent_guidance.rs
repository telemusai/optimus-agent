//! Pure agent-guidance assessment core (ROOT CONTRACT v7, AGENT-GUIDANCE lane).
//!
//! Framework-neutral building blocks consumed by the coding-agent adapter
//! (`crates/pi-coding-agent/src/core/jev_agent_guidance.rs`). This module never
//! performs I/O, never executes a tool or skill, never changes a model, and
//! never grants permission. The guidance categories are NOT in
//! `DEFAULT_APPLIABLE_CATEGORIES` or `OPTIONAL_APPLIABLE_CATEGORIES`, so no
//! answer from them can ever be applied by the acceptance path
//! (`evaluate_answer` refuses them with `CategoryNotAppliable`); `enforced` is
//! structurally `false`.
//!
//! Truth rules baked into this module:
//!
//! * Single-answer uncertainty handling only. The consistency cookbooks use
//!   15-repeat conditions as benchmark measurement methodology; that is NOT a
//!   runtime mandate. The band/abstention mapping below consumes ONE returned
//!   answer and adds no model calls (noul cookbook: "The escalation is
//!   application logic over the returned probability: no new question, no
//!   second API call"; choice cookbook: "adds no model calls").
//! * Thresholds and weights are HOST POLICY: finite, validated, passed in by
//!   the caller. They are never derived from, and never settable by, model
//!   output. The cookbooks' bands and thresholds are illustrative application
//!   policy, not measured accuracy and not global constants.
//! * A decisive critical guardrail hazard cannot be masked by the fixed-weight
//!   composition: it forces the route to at least review.
//! * Every no-hint outcome names a deterministic reason. Absence of a hint is
//!   never reported as a whole-catalog absence claim.

use std::collections::BTreeMap;

use crate::client::DEFAULT_MAX_PAYLOAD_BYTES;
use crate::evaluators::PreparedQuestion;
use crate::hooks::SYSTEM_ONE_MODEL;
use serde_json::Value;
use sha2::{Digest, Sha256};
use crate::snapshot::{bound_json, MAX_ITEMS, MAX_STATE_BYTES, MAX_TEXT_CHARS as MAX_TEXT_LIMIT};
use crate::types::{
    estimate_request_tokens, validate_request_shape, Answer, DecisionCategory, EntryValue,
    NoulCriteria, QuestionSpec, REQUEST_TOKEN_CEILING,
};

/// Bounded catalog size for one skill-suggestion rank question: 31 skill
/// ids + the reserved `none` option = 32 entries, exactly the native
/// `snapshot::bound_json` width cap (`snapshot::MAX_ITEMS`). The full loaded
/// roster stays in the system prompt; this bound applies only to the advisory
/// assessment's closed-set option list. Bounded roster subsets are explicit
/// and match the asked choices: `skill_rank_question` refuses more
/// (fail-open) and never silently drops an option.
pub const MAX_GUIDANCE_CATALOG: usize = 31;

/// Bounded text size for excerpts and descriptions inside guidance state and
/// questions: aligned with `snapshot::MAX_TEXT_CHARS`, so the native
/// `bound_json` (strings above 400 chars are clipped) is a verified no-op on
/// every guidance payload.
pub const MAX_GUIDANCE_TEXT_CHARS: usize = 400;

/// Maximum option/key length kept by `snapshot::bound_json` before key
/// truncation: guidance ids stay at or below it, so a selectable id can never
/// be silently clipped or collide after bounding.
pub const MAX_GUIDANCE_ID_CHARS: usize = 64;

/// Host-policy cap on one guidance question's instructions prose. The native
/// wire carries Text entries verbatim (no size cap; the structured-entry cap
/// is 2048 bytes for Json entries), so this bound is a conservative defensive
/// choice, measured dynamically by the builders (never fixture-derived), and
/// verified by `verify_entry_bounded`. The per-id EVIDENCE entries (choice
/// option descriptions) stay at or below `MAX_TEXT_LIMIT`, so a defensive
/// `bound_json` can never clip selectable evidence.
pub const MAX_GUIDANCE_INSTRUCTIONS_CHARS: usize = 1024;

/// Longest host-derived hint rationale (numbers and fixed labels only).
pub const MAX_HINT_WHY_CHARS: usize = 120;

/// Bounds a host string to `max` characters, marking a cut with `...`.
pub fn bound_text(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut truncated: String = value.chars().take(max).collect();
    truncated.push_str("...");
    truncated
}

// ---------------------------------------------------------------------------
// Host-policy thresholds (validated; never model-settable)
// ---------------------------------------------------------------------------

/// Host-policy thresholds for single-answer uncertainty handling. The values
/// are application policy; hosts validate before use. Nothing here is a global
/// constant, and no threshold claims measured accuracy. These floors are a
/// CONSERVATIVE suppression policy: a candidate below a floor yields no hint
/// (the full roster stays available in the system prompt regardless), they are
/// NOT calibrated correctness guarantees and NOT a blanket skill-alignment
/// verdict; policy stays stable during integration.
#[derive(Debug, Clone, PartialEq)]
pub struct GuidanceThresholds {
    /// A Noul strictly below this value is decisive-low.
    pub noul_low: f64,
    /// A Noul strictly above this value is decisive-high.
    pub noul_high: f64,
    /// Minimum top probability for a Choice answer to count as decisive.
    pub choice_top_min: f64,
    /// Minimum Choice confidence for an assessment to act on it.
    pub min_confidence: f64,
}

impl GuidanceThresholds {
    /// Finite, ordered validation. `Err` carries a stable reason string and
    /// callers must treat `Err` as "assessment unavailable" (no hint, no
    /// claim, no default-on).
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("noul_low", self.noul_low),
            ("noul_high", self.noul_high),
            ("choice_top_min", self.choice_top_min),
            ("min_confidence", self.min_confidence),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be a finite value in [0,1]"));
            }
        }
        if self.noul_low > self.noul_high {
            return Err("noul_low must not exceed noul_high".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Single-answer uncertainty band / abstention (no added model calls)
// ---------------------------------------------------------------------------

/// Where one returned Noul probability sits inside the uncertainty band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoulBand {
    /// Decisive-low (strictly below the band).
    Below,
    /// Inside the inclusive band: explicitly uncertain, not a side.
    Within,
    /// Decisive-high (strictly above the band).
    Above,
}

/// Maps one Noul probability through an inclusive uncertainty band.
/// `None` when the probability or the band is invalid.
pub fn noul_band(probability: f64, low: f64, high: f64) -> Option<NoulBand> {
    if !is_unit(probability) {
        return None;
    }
    if !is_unit(low) || !is_unit(high) || !(low <= high) {
        return None;
    }
    if probability < low {
        Some(NoulBand::Below)
    } else if probability > high {
        Some(NoulBand::Above)
    } else {
        Some(NoulBand::Within)
    }
}

/// Single-answer Choice abstention: act on the returned top label or abstain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceAbstention {
    Select,
    Uncertain,
}

/// `Select` when `top_probability >= threshold` (the choice cookbook acts "at
/// exactly" the threshold), `Uncertain` below it. `None` for invalid input.
pub fn choice_abstention(top_probability: f64, threshold: f64) -> Option<ChoiceAbstention> {
    if !is_unit(top_probability) || !is_unit(threshold) {
        return None;
    }
    if top_probability >= threshold {
        Some(ChoiceAbstention::Select)
    } else {
        Some(ChoiceAbstention::Uncertain)
    }
}

fn is_unit(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

// ---------------------------------------------------------------------------
// Bounded fixed-weight composition (host weights only)
// ---------------------------------------------------------------------------

/// Result of one fixed-weight composition over bounded inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedComposition {
    /// Composed value in [0,1].
    pub composed: f64,
    /// Per-input contributions `(input_id, weight * probability)`.
    pub contributions: Vec<(String, f64)>,
}

/// Composes `items` (`(id, probability)`) with HOST-FIXED weights.
///
/// Deterministic only under full valid coverage: every item needs a finite,
/// non-negative weight and the weights must sum to a positive value.
/// Otherwise `None` — a policy error is surfaced, never silently averaged.
pub fn compose_fixed_weighted(
    items: &[(String, f64)],
    weights: &BTreeMap<String, f64>,
) -> Option<WeightedComposition> {
    let mut weight_sum = 0.0f64;
    let mut contributions = Vec::with_capacity(items.len());
    for (id, probability) in items {
        if !is_unit(*probability) {
            return None;
        }
        let weight = weights.get(id).copied()?;
        if !weight.is_finite() || weight < 0.0 {
            return None;
        }
        weight_sum += weight;
        contributions.push((id.clone(), weight * probability));
    }
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return None;
    }
    let composed = contributions.iter().map(|(_, value)| value).sum::<f64>() / weight_sum;
    if !is_unit(composed) {
        return None;
    }
    Some(WeightedComposition {
        composed,
        contributions,
    })
}

// ---------------------------------------------------------------------------
// Skill suggestion: closed-set assessment, at most ONE non-exclusive hint
// ---------------------------------------------------------------------------

/// One already-loaded catalog entry (host metadata only; no filesystem
/// authority). The `id` doubles as the closed-set Choice option id.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillCatalogEntry {
    pub id: String,
    pub description: String,
}

/// The answers of one skill-suggestion assessment, normalized by the adapter.
/// Every reference is an already-validated `Answer` (the transport's
/// `validate_answer` has run); `None` means the question had no answer.
pub struct SkillAnswerSet<'a> {
    /// `skill_suggestion.0` — closed-set catalog rank Choice.
    pub rank: Option<&'a Answer>,
    /// `skill_suggestion.1` — need gate (task matches a documented procedure).
    pub need_gate: Option<&'a Answer>,
    /// `skill_suggestion.2` — inverse gate (prompt prose already suffices).
    pub inverse_gate: Option<&'a Answer>,
    /// `skill_suggestion.3` — act gate (task acts on the user's system).
    pub act_gate: Option<&'a Answer>,
    /// `skill_suggestion.4` — optional second-pass fit Noul for the ranked id.
    pub fit: Option<&'a Answer>,
    /// The CURRENT bounded catalog ids a hint may name.
    pub catalog_ids: &'a [String],
    /// The decision is fresh and generation-correlated (host check).
    pub fresh: bool,
    /// The decision was cancelled before it completed.
    pub cancelled: bool,
    /// Bounded budget (deadline, slots, breaker) allowed the run.
    pub budget_ok: bool,
    /// Host identity facts CAPTURED with the request inputs before the await;
    /// carried onto any produced hint and re-checked at consumption by
    /// `skill_hint_is_current` against freshly resolved current facts.
    pub stamp: CapturedSkillHintStamp,
}

/// Why no hint is emitted. Every outcome is truthful and specific; "no hint"
/// is NEVER a whole-catalog absence claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoHintReason {
    /// Answers missing/invalid, thresholds not met, or confidence too low:
    /// an incomplete or weak assessment is not a confident candidate.
    NoConfidentCandidate,
    /// A gate sits inside the uncertainty band: explicitly uncertain.
    Uncertain,
    /// The loaded catalog is empty.
    CatalogAbsent,
    /// The decision was cancelled before it completed.
    Cancelled,
    /// The decision is stale for the current context (freshness/generation
    /// check failed, or the ranked id left the current catalog).
    Stale,
    /// Bounded resources (deadline, slots, breaker) did not allow the run.
    BudgetExhausted,
}

impl NoHintReason {
    pub fn as_str(self) -> &'static str {
        match self {
            NoHintReason::NoConfidentCandidate => "no_confident_candidate",
            NoHintReason::Uncertain => "uncertain",
            NoHintReason::CatalogAbsent => "catalog_absent",
            NoHintReason::Cancelled => "cancelled",
            NoHintReason::Stale => "stale",
            NoHintReason::BudgetExhausted => "budget_exhausted",
        }
    }
}

/// Host-supplied identity facts stamped onto a hint at PRODUCTION time and
/// re-derived from CURRENT authoritative host state at CONSUMPTION time.
/// Caller contracts: `task_hash` must fingerprint the FULL native delivery
/// input (not a book field truncated upstream); `catalog_hash` must
/// fingerprint the ACTUAL assessed roster at the consumer seam (the same
/// list that renders the roster block), never a stale cache passed as
/// current; `delivery_id` must advance per genuine task delivery. The
/// consumer rejects any hint whose stamp does not equal the current facts, so
/// full-off / feature-off / Off->On ABA (no consumer visit during the off
/// window), new-task reuse, held old-activation installs and cross-turn leaks
/// are all impossible at the render layer — clearing alone cannot provide
/// this. `settings_revision` MUST be a host value that CHANGES on every
/// authoritative settings reload (for example the settings envelope
/// generation), so an Off->On transition is always observable at consumption
/// even when no prompt was built while off.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillHintStamp {
    /// Authoritative settings identity at the relevant time (host-owned,
    /// opaque; changes on every settings reload).
    pub settings_revision: String,
    /// Host decision mode at that time (e.g. the host's mode label).
    pub mode: String,
    /// Whether the skill-suggestion feature was enabled at that time.
    pub feature_enabled: bool,
    /// The turn the hint was produced for.
    pub turn: u64,
    /// Host-opaque per-DELIVERY identity (e.g. turn plus per-turn delivery
    /// sequence, or the host's request id). Two genuinely NEW task deliveries
    /// with byte-identical text must not reuse the previous request's hint:
    /// content hashing alone cannot distinguish them, this field does.
    pub delivery_id: String,
    /// Host hash of the FULL host task text the hint was produced for (the
    /// real host task/delivery identity, locally fingerprinted — a bounded
    /// presentation excerpt is deliberately NOT used, because different tasks
    /// can share its prefix). The caller must pass the FULL native delivery
    /// input, never an already-truncated book field.
    pub task_hash: String,
    /// Fingerprint of the ASSESSED catalog CONTENTS (id + description per
    /// entry) the hint was produced against — same ids with changed
    /// descriptions hash apart.
    pub catalog_hash: String,
}

/// Deterministic host-side identity hash for stamps (sha256 hex over the
/// joined parts). Hosts use it for `task_hash` / `catalog_hash`; the values
/// are compared for equality only, never shown to a model.
/// Fingerprints the ASSESSED CATALOG CONTENTS: every entry's id AND
/// description participate. Same ids with changed descriptions/meaning hash
/// apart, so a hint produced against one catalog meaning is rejected when the
/// catalog's content changed. Local fingerprint only — no raw catalog text
/// leaves the process.
pub fn catalog_fingerprint(entries: &[SkillCatalogEntry]) -> String {
    let parts: Vec<String> = entries
        .iter()
        .map(|entry| format!("{}\u{0}{}", entry.id, entry.description))
        .collect();
    let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
    guidance_stamp_hash(&refs)
}
pub fn guidance_stamp_hash(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update([0u8]); // part separator
    }
    format!("{:x}", hasher.finalize())
}

/// An IMMUTABLE host stamp CAPTURED WITH THE REQUEST INPUTS BEFORE awaiting a
/// Jev decision, and carried unchanged to store/render. Construction is
/// explicit so the capture-then-compare flow is visible at the type level:
/// the captured facts describe the request the answers belong to. Re-deriving
/// a CURRENT stamp onto OLD answers after the await would bless stale work in
/// a can_apply->store race and is exactly what this type's flow forbids —
/// store/render must compare the CAPTURED stamp against freshly resolved
/// current facts via `skill_hint_is_current`, never attach new facts to old
/// answers.
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedSkillHintStamp {
    stamp: SkillHintStamp,
}

impl CapturedSkillHintStamp {
    /// Captures the host facts as they stood when the request inputs were
    /// composed (BEFORE the Jev await). The value is immutable afterwards.
    pub fn capture(stamp: SkillHintStamp) -> Self {
        Self { stamp }
    }
    /// The captured production facts (read-only).
    pub fn stamp(&self) -> &SkillHintStamp {
        &self.stamp
    }
}

/// ONE non-exclusive advisory hint. The suggested skill is never exclusive,
/// never auto-loaded, never executed, and never granted permission.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillHint {
    /// The suggested catalog id (verified present in the CURRENT catalog).
    pub skill_id: String,
    /// Host identity facts at production time; the consumer re-derives them
    /// from current authoritative state and rejects any mismatch.
    pub stamp: SkillHintStamp,
    /// The ranked option's own probability.
    pub rank_prob: f64,
    /// Need-gate probability backing the hint.
    pub need_prob: f64,
    /// Fixed label: advisory and non-exclusive.
    pub label: &'static str,
    /// Bounded host-derived rationale (fixed labels and returned numbers only;
    /// model text is never copied).
    pub why: String,
}

/// Assessment outcome: at most one hint, or a named reason for no hint.
#[derive(Debug, Clone, PartialEq)]
pub enum SkillHintOutcome {
    Hint(SkillHint),
    NoHint(NoHintReason),
}

/// Noul probability of an answered Noul question, `None` otherwise.
fn noul_probability(answer: Option<&Answer>) -> Option<f64> {
    match answer {
        Some(Answer::Noul { noul }) if is_unit(*noul) => Some(*noul),
        _ => None,
    }
}

/// `(choice id, its own probability, confidence)` for a Choice answer.
///
/// The transport's `validate_answer` has already enforced the closed set, a
/// complete summing distribution, and the argmax constraint (with the
/// documented tie tolerance), so the CHOSEN label's probability is the value
/// to threshold; ties stay valid.
fn choice_top(answer: &Answer) -> Option<(String, f64, f64)> {
    match answer {
        Answer::Choice { choice, probabilities, confidence } => {
            if !is_unit(*confidence) {
                return None;
            }
            if probabilities.is_empty()
                || probabilities.values().any(|p| !is_unit(*p))
            {
                return None;
            }
            let probability = probabilities.get(choice).copied()?;
            Some((choice.clone(), probability, *confidence))
        }
        _ => None,
    }
}

fn bound_hint_why(value: String) -> String {
    let max = MAX_HINT_WHY_CHARS;
    if value.chars().count() <= max {
        return value;
    }
    let mut truncated: String = value.chars().take(max).collect();
    truncated.push_str("...");
    truncated
}

/// Derives the at-most-one non-exclusive advisory skill hint.
///
/// Deterministic order: cancellation, freshness, budget, catalog presence,
/// policy validity, then answer quality. The hint carries no authority: the
/// caller renders it without loading, executing or privileging any skill.
pub fn skill_hint(set: &SkillAnswerSet, thresholds: &GuidanceThresholds) -> SkillHintOutcome {
    if set.cancelled {
        return SkillHintOutcome::NoHint(NoHintReason::Cancelled);
    }
    if !set.fresh {
        return SkillHintOutcome::NoHint(NoHintReason::Stale);
    }
    if !set.budget_ok {
        return SkillHintOutcome::NoHint(NoHintReason::BudgetExhausted);
    }
    if set.catalog_ids.is_empty() {
        return SkillHintOutcome::NoHint(NoHintReason::CatalogAbsent);
    }
    if thresholds.validate().is_err() {
        // Invalid host policy: assessment unavailable. No hint, no claim.
        return SkillHintOutcome::NoHint(NoHintReason::BudgetExhausted);
    }
    let (rank_id, rank_prob, rank_confidence) = match set.rank.and_then(choice_top) {
        Some(top) => top,
        None => return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate),
    };
    if rank_confidence < thresholds.min_confidence {
        return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate);
    }
    match choice_abstention(rank_prob, thresholds.choice_top_min) {
        Some(ChoiceAbstention::Select) => {}
        Some(ChoiceAbstention::Uncertain) => {
            return SkillHintOutcome::NoHint(NoHintReason::Uncertain);
        }
        None => return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate),
    }
    // The ranked id must exist in the CURRENT catalog; a miss is a stale or
    // truncated-context decision, never a hint against a ghost entry.
    if !set.catalog_ids.iter().any(|id| *id == rank_id) {
        return SkillHintOutcome::NoHint(NoHintReason::Stale);
    }
    let Some(need) = noul_probability(set.need_gate) else {
        return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate);
    };
    let Some(inverse) = noul_probability(set.inverse_gate) else {
        return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate);
    };
    let Some(act) = noul_probability(set.act_gate) else {
        return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate);
    };
    let (Some(need_band), Some(inverse_band), Some(act_band)) = (
        noul_band(need, thresholds.noul_low, thresholds.noul_high),
        noul_band(inverse, thresholds.noul_low, thresholds.noul_high),
        noul_band(act, thresholds.noul_low, thresholds.noul_high),
    ) else {
        return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate);
    };
    // Any gate inside the band is explicitly uncertain: no hint, no absence
    // claim, and the uncertainty is recorded by the caller.
    if need_band == NoulBand::Within || inverse_band == NoulBand::Within || act_band == NoulBand::Within {
        return SkillHintOutcome::NoHint(NoHintReason::Uncertain);
    }
    // Hint requires need decisively HIGH, inverse gate decisively LOW (the
    // prompt prose does NOT already suffice) and act gate decisively HIGH.
    let gates_agree =
        need_band == NoulBand::Above && inverse_band == NoulBand::Below && act_band == NoulBand::Above;
    if !gates_agree {
        return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate);
    }
    if let Some(fit_answer) = set.fit {
        let Some(fit_prob) = noul_probability(Some(fit_answer)) else {
            return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate);
        };
        match noul_band(fit_prob, thresholds.noul_low, thresholds.noul_high) {
            Some(NoulBand::Above) => {}
            Some(NoulBand::Within) => return SkillHintOutcome::NoHint(NoHintReason::Uncertain),
            Some(NoulBand::Below) | None => {
                return SkillHintOutcome::NoHint(NoHintReason::NoConfidentCandidate);
            }
        }
    }
    SkillHintOutcome::Hint(SkillHint {
        skill_id: rank_id,
        stamp: set.stamp.stamp().clone(),
        rank_prob,
        need_prob: need,
        label: "advisory, non-exclusive",
        why: bound_hint_why(format!(
            "rank p={rank_prob:.2}; need={need:.2}; inverse={inverse:.2}; act={act:.2}; conf={rank_confidence:.2}"
        )),
    })
}

/// Consumption-side gate for a stored hint: the hint renders ONLY when its
/// production stamp equals the CURRENTLY re-derived host facts. Rejects held
/// old-activation installs (turn mismatch), new-task reuse (task hash),
/// catalog changes (catalog hash), cross-turn leaks, mode changes and — the
/// Off->On ABA case — any settings revision change, with no dependency on
/// somebody observing the off window. A mismatch consumer treats the hint as
/// absent (advisory fail-open); nothing is cleared by regex and no history is
/// rewritten.
pub fn skill_hint_is_current(hint: &SkillHint, current: &SkillHintStamp) -> bool {
    // Consumption-side feature gate: even with a perfectly matching stamp, a
    // hint is never rendered while the feature is disabled.
    if !current.feature_enabled {
        return false;
    }
    hint.stamp == *current
}

// ---------------------------------------------------------------------------
// Guardrail assessment: labelled route recommendation, never enforcement
// ---------------------------------------------------------------------------

/// Deterministic route recommendation for one guardrail assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailRoute {
    Pass,
    Review,
    Block,
    Support,
}

impl GuardrailRoute {
    pub fn as_str(self) -> &'static str {
        match self {
            GuardrailRoute::Pass => "pass",
            GuardrailRoute::Review => "review",
            GuardrailRoute::Block => "block",
            GuardrailRoute::Support => "support",
        }
    }
}

/// Truthful timing label. The output battery runs AFTER streaming: it is a
/// post-hoc assessment, NOT preventive screening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingLabel {
    /// Input battery at the input boundary.
    InputScreenPreventive,
    /// Output battery after streaming: post-hoc assessment only.
    OutputPostHocAssessment,
}

impl TimingLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            TimingLabel::InputScreenPreventive => "input_screen_preventive",
            TimingLabel::OutputPostHocAssessment => "output_post_hoc_assessment",
        }
    }
}

/// Host-policy weights and thresholds for one guardrail battery. Validated
/// before use; weights are host constants and can never be set or shifted by
/// model output.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardrailPolicy {
    pub thresholds: GuidanceThresholds,
    /// Composed severity at or above this routes to review.
    pub severity_review: f64,
    /// Composed severity at or above this routes to block.
    pub severity_block: f64,
    /// Fixed host weights keyed by hazard id; full coverage required.
    pub weights: BTreeMap<String, f64>,
    /// Hazard ids decisive when they score decisive-high. A critical flag can
    /// never be masked by the composed value.
    pub critical_ids: &'static [&'static str],
    /// Hazard ids that route to support resources when decisive.
    pub support_ids: &'static [&'static str],
    /// When false, the support route never fires.
    pub support_enabled: bool,
}

impl GuardrailPolicy {
    pub fn validate(&self) -> Result<(), String> {
        self.thresholds.validate()?;
        for (name, value) in [
            ("severity_review", self.severity_review),
            ("severity_block", self.severity_block),
        ] {
            if !is_unit(value) {
                return Err(format!("{name} must be a finite value in [0,1]"));
            }
        }
        if self.severity_review > self.severity_block {
            return Err("severity_review must not exceed severity_block".to_string());
        }
        if self.weights.is_empty() {
            return Err("guardrail weights are empty".to_string());
        }
        for (id, weight) in &self.weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(format!("weight for {id} must be finite and non-negative"));
            }
        }
        let sum: f64 = self.weights.values().copied().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err("guardrail weights must sum to a positive value".to_string());
        }
        for critical in self.critical_ids {
            if !self.weights.contains_key(*critical) {
                return Err(format!("critical hazard {critical} has no weight"));
            }
        }
        for support in self.support_ids {
            if !self.weights.contains_key(*support) {
                return Err(format!("support hazard {support} has no weight"));
            }
        }
        Ok(())
    }
}

/// One guardrail assessment input, normalized by the adapter from raw records.
pub struct GuardrailInput {
    /// `(hazard id, returned Noul probability)` pairs.
    pub hazards: Vec<(String, f64)>,
    /// Returned severity score in [0,1], when the severity question answered.
    pub severity: Option<f64>,
    pub fresh: bool,
    pub cancelled: bool,
    pub budget_ok: bool,
}

/// The full assessment result. `enforced` is ALWAYS `false`: nothing here
/// blocks, drops, refuses, switches a model, adds a tool, or starts a loop.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardrailAssessment {
    pub route: GuardrailRoute,
    pub timing: TimingLabel,
    pub enforced: bool,
    pub composed: Option<WeightedComposition>,
    /// Hazard ids that scored decisive-high.
    pub decisive_flags: Vec<String>,
    /// Hazard ids inside the uncertainty band.
    pub uncertain_flags: Vec<String>,
    /// Critical hazard ids that forced the route up (mask-proof).
    pub critical_flags: Vec<String>,
}

/// Derives the labelled route recommendation for one screened message.
///
/// `None` when the assessment could not run truthfully (cancelled, stale, over
/// budget, invalid policy, invalid answers, or a hazard without a host
/// weight) — the caller labels the no-op; nothing is inferred and nothing is
/// enforced.
pub fn guardrail_route(
    input: &GuardrailInput,
    policy: &GuardrailPolicy,
    timing: TimingLabel,
) -> Option<GuardrailAssessment> {
    if input.cancelled || !input.fresh || !input.budget_ok {
        return None;
    }
    if policy.validate().is_err() {
        return None;
    }
    let weighted = compose_fixed_weighted(&input.hazards, &policy.weights)?;
    let composed = weighted.composed;
    let mut decisive_flags = Vec::new();
    let mut uncertain_flags = Vec::new();
    let mut critical_flags = Vec::new();
    for (id, probability) in &input.hazards {
        let band = noul_band(*probability, policy.thresholds.noul_low, policy.thresholds.noul_high)?;
        match band {
            NoulBand::Above => {
                decisive_flags.push(id.clone());
                if policy.critical_ids.contains(&id.as_str()) {
                    critical_flags.push(id.clone());
                }
            }
            NoulBand::Within => uncertain_flags.push(id.clone()),
            NoulBand::Below => {}
        }
    }
    let severity = match input.severity {
        Some(value) if is_unit(value) => Some(value),
        Some(_) => return None,
        None => None,
    };
    // Deterministic escalation. Precedence: block > review > support > pass.
    let mut route = GuardrailRoute::Pass;
    if policy.support_enabled
        && decisive_flags
            .iter()
            .any(|id| policy.support_ids.contains(&id.as_str()))
    {
        route = GuardrailRoute::Support;
    }
    if composed >= policy.severity_review
        && matches!(route, GuardrailRoute::Pass | GuardrailRoute::Support)
    {
        route = GuardrailRoute::Review;
    }
    if let Some(severity_value) = severity {
        if severity_value >= policy.severity_review
            && matches!(route, GuardrailRoute::Pass | GuardrailRoute::Support)
        {
            route = GuardrailRoute::Review;
        }
    }
    if !critical_flags.is_empty() {
        // Mask-proof: a decisive critical hazard routes at least to review no
        // matter what the fixed-weight composition says.
        route = GuardrailRoute::Review;
    }
    if composed >= policy.severity_block {
        route = GuardrailRoute::Block;
    }
    if let Some(severity_value) = severity {
        if severity_value >= policy.severity_block {
            route = GuardrailRoute::Block;
        }
    }
    Some(GuardrailAssessment {
        route,
        timing,
        enforced: false,
        composed: Some(weighted),
        decisive_flags,
        uncertain_flags,
        critical_flags,
    })
}

// ---------------------------------------------------------------------------
// Question builders (generic shaping only; wording lives host-side)
// ---------------------------------------------------------------------------

/// Truncates `text` to `max` characters on char boundaries with a `...` mark.
fn bound_question_text(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut truncated: String = value.chars().take(max).collect();
    truncated.push_str("...");
    truncated
}

/// Builds the closed-set catalog rank question.
///
/// `catalog` is the CURRENT bounded catalog (the adapter truncates and
/// discloses via `truncated`). The `none` option keeps abstention inside the
/// closed set, so a no-skill verdict is a first-class answer, never a missing
/// one and never a whole-catalog absence claim.
pub fn skill_rank_question(
    task_excerpt: &str,
    catalog: &[SkillCatalogEntry],
    truncated: bool,
) -> Option<PreparedQuestion> {
    if catalog.is_empty() || catalog.len() > MAX_GUIDANCE_CATALOG {
        return None;
    }
    let mut ids: Vec<&str> = catalog.iter().map(|entry| entry.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != catalog.len()
        || ids.iter().any(|id| id.is_empty() || *id == "none" || id.chars().count() > MAX_GUIDANCE_ID_CHARS)
    {
        return None;
    }
    // Dynamic excerpt budget: the whole instructions string stays at or below
    // `snapshot::MAX_TEXT_CHARS`, so a defensive bounding can never clip the
    // provenance or the excerpt. No fixture-derived magic overhead: the
    // budget is measured from the composed provenance template itself.
    let prefix = format!(
        "Advisory catalog assessment only (non-binding, recorded assessment): which ONE skill from the loaded catalog below is most relevant to the current task? \
Choose `none` when none clearly helps. This never loads, executes or privileges a skill; the full roster remains in the system prompt. \
Catalog source: already-loaded roster metadata; options={}; truncated={truncated}. Current bounded task excerpt: ",
        catalog.len()
    );
    // Account for the ellipsis suffix `bound_question_text` appends when it
    // truncates, so the composed instructions stay within the cap even for an
    // excerpt at the full budget.
    let budget = MAX_GUIDANCE_INSTRUCTIONS_CHARS
        .saturating_sub(prefix.chars().count())
        .saturating_sub(3);
    if budget == 0 {
        // The provenance alone cannot fit the bound: refuse rather than clip.
        return None;
    }
    let excerpt = bound_question_text(task_excerpt, budget);
    let instructions = format!("{prefix}{excerpt}");
    let mut options: Vec<(String, Option<String>)> = catalog
        .iter()
        .map(|entry| {
            (
                entry.id.clone(),
                Some(bound_question_text(&entry.description, 200)),
            )
        })
        .collect();
    options.push((
        "none".to_string(),
        Some("no cataloged skill clearly helps this task".to_string()),
    ));
    Some(PreparedQuestion {
        question_id: "skill_suggestion.0".to_string(),
        spec: QuestionSpec::choice(instructions, options),
    })
}

/// Builds one of the three gate Noul questions (`n` in `1..=3`):
/// 1 need gate, 2 inverse gate (prompt prose suffices), 3 act gate.
pub fn skill_gate_question(n: usize) -> Option<PreparedQuestion> {
    let (id, question, yes, no) = match n {
        1 => (
            "skill_suggestion.1",
            "Does the current task match a documented procedure that one of the cataloged skills carries?",
            "the task matches a documented skill procedure",
            "no cataloged skill documents a matching procedure",
        ),
        2 => (
            "skill_suggestion.2",
            "Are the instructions already present in the system prompt sufficient for this task without a skill?",
            "the existing prompt instructions already suffice",
            "a skill's documented procedure would add real guidance",
        ),
        3 => (
            "skill_suggestion.3",
            "Does the current task ask the agent to act on the user's system (files, commands, network or configuration)?",
            "the task acts on the user's system",
            "the task does not act on the user's system",
        ),
        _ => return None,
    };
    Some(PreparedQuestion {
        question_id: id.to_string(),
        spec: QuestionSpec::noul(
            EntryValue::text(format!(
                "{question} Answer from the current bounded task context only. Assessment only; nothing is enforced."
            )),
            NoulCriteria::text(yes, no),
        ),
    })
}

/// Builds the optional second-pass fit Noul for one ranked skill id.
pub fn skill_fit_question(skill_id: &str) -> Option<PreparedQuestion> {
    let id = bound_question_text(skill_id, 64);
    if id.is_empty() || id == "none" {
        return None;
    }
    Some(PreparedQuestion {
        question_id: "skill_suggestion.4".to_string(),
        spec: QuestionSpec::noul(
            EntryValue::text(format!(
                "Would following the documented procedure of loaded skill `{id}` materially help with the current bounded task? Advisory assessment only; nothing is enforced."
            )),
            NoulCriteria::text(
                "the skill's documented procedure would materially help",
                "the skill's procedure would not materially help",
            ),
        ),
    })
}

/// Builds one guardrail battery: one Noul per hazard plus ONE severity Score
/// question. The hazard WORDING is host policy (passed in); this builder only
/// shapes the closed-set questions.
pub fn guardrail_battery_questions(
    prefix: &str,
    hazards: &[(&str, &str)],
    severity_levels: [&str; 5],
) -> Option<Vec<PreparedQuestion>> {
    if prefix.is_empty() || hazards.is_empty() || hazards.len() > 8 {
        return None;
    }
    if severity_levels.iter().any(|level| level.is_empty()) {
        return None;
    }
    let mut questions = Vec::with_capacity(hazards.len() + 1);
    for (index, (hazard_id, hazard_text)) in hazards.iter().enumerate() {
        let hazard_id = bound_question_text(hazard_id, 64);
        let hazard_text = bound_question_text(hazard_text, 200);
        if hazard_id.is_empty() {
            return None;
        }
        questions.push(PreparedQuestion {
            question_id: format!("{prefix}.{index}"),
            spec: QuestionSpec::noul(
                EntryValue::text(format!(
                    "Assess this bounded, redacted excerpt for `{hazard_id}`: {hazard_text} Assessment only; nothing is enforced."
                )),
                NoulCriteria::text(
                    format!("present: {hazard_id}"),
                    format!("absent: {hazard_id}"),
                ),
            ),
        });
    }
    questions.push(PreparedQuestion {
        question_id: format!("{prefix}.{}", hazards.len()),
        spec: QuestionSpec::score(
            EntryValue::text(
                "Overall severity of the screened excerpt for a coding-agent session. Assessment only; the level wording is host policy, illustrative, not measured accuracy; nothing is enforced.",
            ),
            severity_levels,
        ),
    });
    Some(questions)
}

// ---------------------------------------------------------------------------
// Outgoing-payload visibility verification (ROOT CONTRACT v7 cross-lane rule)
// ---------------------------------------------------------------------------
//
// Builder completeness is not enough: the native observer bounds its outgoing
// state (`snapshot::bound_json`: strings above 400 chars, arrays/objects above
// 32 entries, keys above 64 chars are clipped) and silently refuses requests
// whose state exceeds `MAX_STATE_BYTES` (hooks::prepare_explicit returns None;
// the scheduler drops state above its payload cap). Every guidance request is
// therefore verified against the ACTUAL outgoing payload rules below. Anything
// that cannot be proven visible and within bounds is REFUSED (fail-open): no
// question over silently dropped evidence, no raised caps, no annotations.

/// Why a guidance request was refused before transport. Every variant is a
/// fail-open refusal: the caller sends nothing and records the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuidanceRefusal {
    /// Raw state exceeds `snapshot::MAX_STATE_BYTES`: the native explicit
    /// path would silently drop the whole request.
    StateTooLarge,
    /// The defensive `bound_json` would change the payload: some string,
    /// width or key would be silently clipped, so selectable evidence could
    /// go missing. Carries the offending question/state detail.
    BoundWouldClip(String),
    /// The question's category id is not registered in `DecisionCategory`:
    /// the native dispatch would silently refuse it.
    CategoryUnregistered(String),
    /// The question violates a native shape cap (entry JSON bytes, choice
    /// width, score levels).
    ShapeInvalid(String),
    /// The rank question's closed set does not exactly match the intended
    /// bounded catalog (plus `none`), or an option's evidence is missing.
    CatalogClosedSetMismatch(String),
    /// The heuristic estimated request tokens exceed the host-policy ceiling.
    TokenCeilingExceeded,
    /// The serialized request body exceeds the transport payload cap.
    PayloadTooLarge,
}

impl GuidanceRefusal {
    pub fn as_str(&self) -> &'static str {
        match self {
            GuidanceRefusal::StateTooLarge => "state_too_large",
            GuidanceRefusal::BoundWouldClip(_) => "bound_would_clip",
            GuidanceRefusal::CategoryUnregistered(_) => "category_unregistered",
            GuidanceRefusal::ShapeInvalid(_) => "shape_invalid",
            GuidanceRefusal::CatalogClosedSetMismatch(_) => "catalog_closed_set_mismatch",
            GuidanceRefusal::TokenCeilingExceeded => "token_ceiling_exceeded",
            GuidanceRefusal::PayloadTooLarge => "payload_too_large",
        }
    }
}

/// Verifies the outgoing STATE of one guidance request against the native
/// rules:
///
/// 1. the raw serialized state fits `MAX_STATE_BYTES` (the explicit path's
///    actual gate — above it the request is silently dropped);
/// 2. applying the native `bound_json` is a byte-level NO-OP (nothing would
///    be silently clipped by any defensive bounding).
pub fn verify_guidance_state(state: &Value) -> Result<(), GuidanceRefusal> {
    let raw_len = serde_json::to_vec(state).unwrap_or_default().len();
    if raw_len > MAX_STATE_BYTES {
        return Err(GuidanceRefusal::StateTooLarge);
    }
    let bounded = bound_json(state.clone(), 0);
    if bounded != *state {
        return Err(GuidanceRefusal::BoundWouldClip(format!(
            "snapshot::bound_json would clip the state (raw {raw_len} bytes)"
        )));
    }
    Ok(())
}

/// Verifies ONE question's outgoing visibility: the category id must be
/// registered (the native dispatch gate), and every text entry must survive
/// `bound_json` unchanged (strings at or below `MAX_TEXT_LIMIT`, option maps
/// at or below `MAX_ITEMS`, ids at or below the key cap). Shape validity is
/// the native `QuestionSpec::validate_shape`.
pub fn verify_guidance_question(question: &PreparedQuestion) -> Result<(), GuidanceRefusal> {
    let category = question
        .question_id
        .rsplit_once('.')
        .map(|(prefix, _)| prefix)
        .unwrap_or("");
    let Some(_parsed) = DecisionCategory::parse(category) else {
        return Err(GuidanceRefusal::CategoryUnregistered(
            question.question_id.clone(),
        ));
    };
    question.spec.validate_shape().map_err(|error| {
        GuidanceRefusal::ShapeInvalid(format!("{}: {error}", question.question_id))
    })?;
    verify_entry_bounded(&question.spec, &question.question_id)?;
    Ok(())
}

fn verify_entry_bounded(spec: &QuestionSpec, question_id: &str) -> Result<(), GuidanceRefusal> {
    let clip = |name: &str| GuidanceRefusal::BoundWouldClip(format!("{question_id}: {name}"));
    // Instructions prose: host-policy cap, measured, comfortably within the
    // native structured-entry limit; goes out verbatim on the explicit path.
    let check_instructions = |text: &str| -> Result<(), GuidanceRefusal> {
        if text.chars().count() > MAX_GUIDANCE_INSTRUCTIONS_CHARS {
            return Err(GuidanceRefusal::BoundWouldClip(format!(
                "{question_id}: instructions exceed {MAX_GUIDANCE_INSTRUCTIONS_CHARS} chars"
            )));
        }
        Ok(())
    };
    // Per-id EVIDENCE (descriptions, criteria sides): at or below the native
    // state text cap, so a defensive bound_json can never clip evidence.
    let check_evidence = |name: &str, text: &str| -> Result<(), GuidanceRefusal> {
        if text.chars().count() > MAX_TEXT_LIMIT {
            return Err(GuidanceRefusal::BoundWouldClip(format!(
                "{question_id}: {name} exceeds the {MAX_TEXT_LIMIT}-char evidence bound"
            )));
        }
        Ok(())
    };
    match spec {
        QuestionSpec::Noul { instructions, criteria } => {
            if let Some(EntryValue::Text(text)) = instructions {
                check_instructions(text)?;
            }
            if let Some(criteria) = criteria {
                check_evidence("criteria true", &criteria.r#true.instructions_hash_input())?;
                check_evidence("criteria false", &criteria.r#false.instructions_hash_input())?;
            }
        }
        QuestionSpec::Choice { instructions, criteria } => {
            if criteria.len() > MAX_ITEMS {
                return Err(GuidanceRefusal::BoundWouldClip(format!(
                    "{question_id}: {} options exceed the {MAX_ITEMS}-entry bound",
                    criteria.len()
                )));
            }
            if let Some(EntryValue::Text(text)) = instructions {
                check_instructions(text)?;
            }
            for (key, entry) in criteria {
                if key.chars().count() > MAX_GUIDANCE_ID_CHARS {
                    return Err(GuidanceRefusal::BoundWouldClip(format!(
                        "{question_id}: option key exceeds {MAX_GUIDANCE_ID_CHARS} chars"
                    )));
                }
                if let EntryValue::Text(text) = entry {
                    check_evidence(&format!("option `{key}`"), text)?;
                }
            }
        }
        QuestionSpec::Score { instructions, criteria } => {
            if let Some(EntryValue::Text(text)) = instructions {
                check_instructions(text)?;
            }
            for (index, entry) in criteria.iter().enumerate() {
                if let EntryValue::Text(text) = entry {
                    check_evidence(&format!("level {index}"), text)?;
                }
            }
        }
    }
    Ok(())
}

/// Verifies the rank question's closed set EXACTLY matches the intended
/// bounded catalog (plus the reserved `none` option) and that every selectable
/// id has its redacted description PRESENT post-bounding. This is the
/// anti-defect rule: no question over silently dropped skills.
pub fn verify_rank_catalog(
    question: &PreparedQuestion,
    catalog_ids: &[String],
) -> Result<(), GuidanceRefusal> {
    let QuestionSpec::Choice { criteria, .. } = &question.spec else {
        return Err(GuidanceRefusal::CatalogClosedSetMismatch(format!(
            "{} is not a Choice question",
            question.question_id
        )));
    };
    let mut expected: Vec<&String> = catalog_ids.iter().collect();
    expected.sort();
    expected.dedup();
    if expected.len() != catalog_ids.len() || catalog_ids.is_empty() {
        return Err(GuidanceRefusal::CatalogClosedSetMismatch(
            "catalog ids are empty or contain duplicates".to_string(),
        ));
    }
    let mut actual: Vec<&String> = criteria.keys().collect();
    actual.sort();
    if actual.len() != expected.len() + 1 || !actual.contains(&&"none".to_string()) {
        return Err(GuidanceRefusal::CatalogClosedSetMismatch(format!(
            "rank options ({}) do not match the catalog plus `none`",
            actual.len()
        )));
    }
    for id in catalog_ids {
        let Some(EntryValue::Text(description)) = criteria.get(id) else {
            return Err(GuidanceRefusal::CatalogClosedSetMismatch(format!(
                "option `{id}` is missing its description"
            )));
        };
        if description.trim().is_empty() {
            return Err(GuidanceRefusal::CatalogClosedSetMismatch(format!(
                "option `{id}` has an empty description"
            )));
        }
    }
    Ok(())
}

/// Verifies a whole guidance request against the ACTUAL outgoing rules: state
/// visibility, per-question visibility and dispatch, the rank closed set, the
/// native request-shape gate, the estimated token ceiling, and the serialized
/// body size. `rank` is `Some((question_id, catalog_ids))` for the
/// skill-suggestion request and `None` for batteries.
pub fn verify_guidance_request(
    state: &Value,
    questions: &[PreparedQuestion],
    rank: Option<(&str, &[String])>,
) -> Result<(), GuidanceRefusal> {
    verify_guidance_state(state)?;
    let mut specs = std::collections::BTreeMap::new();
    for question in questions {
        verify_guidance_question(question)?;
        if specs.insert(question.question_id.clone(), question.spec.clone()).is_some() {
            return Err(GuidanceRefusal::ShapeInvalid(format!(
                "duplicate question id {}",
                question.question_id
            )));
        }
    }
    if let Some((rank_id, catalog_ids)) = rank {
        let rank_question = questions
            .iter()
            .find(|question| question.question_id == rank_id)
            .ok_or_else(|| {
                GuidanceRefusal::CatalogClosedSetMismatch(format!("{rank_id} missing"))
            })?;
        verify_rank_catalog(rank_question, catalog_ids)?;
    }
    let request = crate::types::SystemOneRequest {
        state: state.clone(),
        model: SYSTEM_ONE_MODEL.to_string(),
        questions: specs,
    };
    validate_request_shape(&request)
        .map_err(|error| GuidanceRefusal::ShapeInvalid(error.to_string()))?;
    if estimate_request_tokens(&request) > REQUEST_TOKEN_CEILING {
        return Err(GuidanceRefusal::TokenCeilingExceeded);
    }
    let body = serde_json::to_vec(&request).unwrap_or_default();
    if body.len() > DEFAULT_MAX_PAYLOAD_BYTES {
        return Err(GuidanceRefusal::PayloadTooLarge);
    }
    Ok(())
}
