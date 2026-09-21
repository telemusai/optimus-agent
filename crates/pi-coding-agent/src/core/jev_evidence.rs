//! ROOT-CONTRACT v6 (Evidence lane): host adapter for the two bounded,
//! advisory-only evidence assessments.
//!
//! - Retrieval safety: pure primitives live in `pi_jev::evidence`. This
//!   module supplies the veto/annotation application the bridge calls after
//!   the filter stage consumes the combined request (battery rides the
//!   EXISTING filter decision; no new timeout or round-trip stack).
//! - Citation check: one advisory Choice judgment over the ACTUAL ipython
//!   source-read path. The reachability scan below mirrors the deterministic
//!   `print(open(PATH).read())` / `print(Path(PATH).read_text())` idiom
//!   recognition line-find uses (self-owned copy; no private deps), with the
//!   SAME one-judgment-per-provider-request and single-annotation discipline.
//!
//! Contract conformance (fail-open everywhere):
//! - Unknown, missing, stale, cancelled, malformed or over-budget answers
//!   produce NO effect; originals are preserved; annotations attach to the
//!   request copy only (single text block on a single-block toolResult).
//! - Model classification is advisory telemetry. Nothing here verifies,
//!   grants authority or a permission, or claims document-wide absence.
//! - An absent quote is reported as
//!   `quote_not_found_in_supplied_source_span` (supplied-span scope only),
//!   computed NATIVELY, never by the model, and never as fabricated support.
//! - The contradiction veto only SHRINKS the planned removal set (un-drop of
//!   conflicting evidence). No candidate is ever newly dropped here.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use pi_jev::active::ActivationPolicy;
use pi_jev::config::JevMode;
use pi_jev::evidence::{
    battery_state, citation_annotation_json, citation_metadata, citation_questions,
    citation_verdict, extract_quoted_span, safety_annotation_json, safety_labels_from_decisions,
    safety_metadata, safety_questions, select_battery_candidates, vetoed_removals, BatteryScope,
    CITATION_RELATION_QUESTION_ID, EVIDENCE_PROMPT_VERSION, MAX_CITATION_SPAN_BYTES,
    SAFETY_BATTERY_CANDIDATE_CAP, SAFETY_STAGE,
};
use pi_jev::hooks::{ActiveDecideOutcome, JevObserver};
use pi_jev::redact::bounded_excerpt;
use pi_jev::search::SearchBudget;
use pi_jev::snapshot::fingerprint_of;
use pi_jev::types::DecisionCategory;
use serde_json::{json, Value};

use super::jev_code_search::{
    already_observed, remember_observation, SearchPresentation, MAX_PRESENTATION_BYTES,
};

/// Stage tag used in observation cache keys (never shadows the search or
/// line-find stages).
pub const CITATION_STAGE: &str = "code_citation_check";

/// Everything the citation stage needs, resolved once by the bridge wiring.
/// The adapter stays free of bridge-internal helpers.
pub struct CitationStageInputs<'a> {
    pub observer: Arc<JevObserver>,
    pub session_id: &'a str,
    pub turn: u64,
    pub mode: JevMode,
    pub policy_generation: &'a str,
    pub max_decision_age: Duration,
    pub budget: &'a SearchBudget,
    /// The claim: the same query/task excerpt the other search stages use.
    pub claim: &'a str,
    /// Production captures this from the strict, unannotated provider-context
    /// input before any Jev stage runs. Direct adapter tests may leave it empty
    /// and exercise the adapter's bounded reachability scan itself.
    pub pre_annotation_presentation: Option<CitationPresentation>,
    /// Fresh cancellation probe (the bridge closes over `ctx.signal()`).
    pub is_cancelled: &'a (dyn Fn() -> bool + Sync),
}

/// One citation judgment per provider request: the most recent eligible
/// source read only.
#[derive(Clone)]
pub struct CitationPresentation {
    pub message_index: usize,
    tool_call_id: String,
    pub path: String,
    /// The full supplied span (byte-capped): the verbatim check runs here.
    pub span: String,
    /// The span text embedded in the question (bounded excerpt of `span`).
    pub span_excerpt: String,
    pub span_bytes: usize,
    pub span_truncated: bool,
    pub fingerprint: String,
}

/// ROOT-CONTRACT v6 (Evidence lane): the citation stage. Advisory annotation
/// on the request copy only, or nothing. Compare observes without applying.
pub async fn annotate_citation_check(
    mut messages: Vec<Value>,
    inputs: &CitationStageInputs<'_>,
) -> Vec<Value> {
    let claim = inputs.claim;
    if claim.trim().is_empty() || (inputs.is_cancelled)() {
        return messages;
    }
    if !inputs.mode.is_enabled() {
        return messages;
    }
    if !inputs.mode.allows_active() {
        return observe_citation(messages, inputs, claim).await;
    }
    if inputs.budget.expired() {
        return messages;
    }
    let Some(presentation) = inputs
        .pre_annotation_presentation
        .clone()
        .or_else(|| prepare_citation(&messages, claim).into_iter().next())
    else {
        return messages;
    };
    let quote = extract_quoted_span(claim);
    let Some(questions) = citation_questions(claim, quote.as_deref(), &presentation.span_excerpt)
    else {
        return messages;
    };
    let state = json!({
        "user_text_excerpt": claim,
        "citation_source_path": presentation.path,
        "citation_span": presentation.span_excerpt,
    });
    // Mirror of the observer's explicit-request state precondition: a state
    // that prepare_explicit would silently refuse is refused here first —
    // truthful, transport-free, zero effects.
    if serde_json::to_vec(&state)
        .map(|bytes| bytes.len())
        .unwrap_or(usize::MAX)
        > 8192
    {
        return messages;
    }
    let mut payload = json!({
        "session_id": inputs.session_id,
        "turn": inputs.turn,
        "policy_generation": inputs.policy_generation,
        "state": state,
    });
    payload["decision_timeout_ms"] = json!(inputs.budget.remaining_ms());
    let policy = ActivationPolicy {
        enabled_categories: [DecisionCategory::CodeCitationCheck].into_iter().collect(),
        // Advisory: no confidence gate; the review flag is computed from the
        // confidence and the native verbatim check instead.
        min_confidence: 0.0,
        max_decision_age: inputs.max_decision_age,
    };
    let outcome = inputs
        .observer
        .decide_prepared(&payload, CITATION_STAGE, questions, &policy)
        .await;
    let usable = !(inputs.is_cancelled)()
        && inputs.observer.can_apply(&outcome)
        && outcome.unavailable.is_none()
        && outcome.raw.as_ref().is_some_and(|raw| raw.skips.is_empty())
        && outcome.turn == inputs.turn;
    if !usable {
        return messages;
    }
    let Some(decision) = outcome.decisions.iter().find(|decision| {
        decision.question_id == CITATION_RELATION_QUESTION_ID
            && decision.category == DecisionCategory::CodeCitationCheck
    }) else {
        return messages;
    };
    let Some(verdict) = citation_verdict(
        &decision.value,
        decision.confidence,
        quote.as_deref(),
        &presentation.span,
    ) else {
        return messages;
    };
    let Some(request_id) = outcome.request_id.as_deref() else {
        return messages;
    };
    let Some(annotation) = citation_annotation_json(
        &verdict,
        "ipython_source_read",
        &presentation.path,
        presentation.span_bytes,
        presentation.span_truncated,
        request_id,
        outcome.response_model.as_deref(),
    ) else {
        return messages;
    };
    let Ok(text) = serde_json::to_string(&annotation) else {
        return messages;
    };
    let block = json!({"type": "text", "text": text});
    // ROOT-CONTRACT v6 (Evidence lane): the eligible ORIGINAL source was
    // fingerprinted at presentation time; verify it is still the same
    // immutable original at attach time (prior advisory annotations are
    // stripped for this check, never merged). A mismatch fails open: the
    // assessment is recorded, nothing is attached.
    let mut targets = messages.iter().enumerate().filter(|(_, message)| {
        message["role"] == "toolResult"
            && message["toolCallId"].as_str() == Some(presentation.tool_call_id.as_str())
    });
    let target = targets.next();
    let unique_target = targets.next().is_none();
    drop(targets);
    let target_index = target.map(|(index, _)| index);
    let target_intact = unique_target
        && target
            .and_then(|(_, message)| message["content"].as_array())
            .and_then(|content| original_source_block(content))
            .and_then(|block| block["text"].as_str())
            .map(|text_now| {
                fingerprint_of(&json!([
                    presentation.tool_call_id.as_str(),
                    text_now,
                    claim
                ]))
            })
            .is_some_and(|fingerprint| fingerprint == presentation.fingerprint);
    let attached = if !target_intact {
        false
    } else {
        target_index.is_some_and(|index| attach_advisory_block(&mut messages, index, block))
    };
    let mut metadata = citation_metadata(&verdict, &presentation.path);
    // The assessment ran and was accepted; record it whether or not the
    // annotation could attach, so the model call is never spent silently.
    // The attach state is recorded distinctly — no false attached claims.
    metadata.insert(
        "jev_citation_attach".to_string(),
        if !target_intact {
            "target_changed".to_string()
        } else if attached {
            "attached".to_string()
        } else {
            "skipped_multi_block".to_string()
        },
    );
    inputs
        .observer
        .record_active_with_action(&outcome, &BTreeMap::new(), &metadata);
    messages
}

/// Compare mode: observe the citation questions for the most recent eligible
/// source read and remember the observation; nothing changes.
async fn observe_citation(
    messages: Vec<Value>,
    inputs: &CitationStageInputs<'_>,
    claim: &str,
) -> Vec<Value> {
    let Some(presentation) = inputs
        .pre_annotation_presentation
        .clone()
        .or_else(|| prepare_citation(&messages, claim).into_iter().next())
    else {
        return messages;
    };
    let cache_key = fingerprint_of(&json!([
        inputs.session_id,
        inputs.turn,
        inputs.mode.as_str(),
        inputs.policy_generation,
        CITATION_STAGE,
        presentation.fingerprint,
    ]));
    if already_observed(&cache_key) {
        return messages;
    }
    let quote = extract_quoted_span(claim);
    let Some(questions) = citation_questions(claim, quote.as_deref(), &presentation.span_excerpt)
    else {
        return messages;
    };
    let mut payload = json!({
        "session_id": inputs.session_id,
        "turn": inputs.turn,
        "policy_generation": inputs.policy_generation,
        "state": {
            "user_text_excerpt": claim,
            "citation_source_path": presentation.path,
            "citation_span": presentation.span_excerpt,
        },
    });
    payload["decision_timeout_ms"] = json!(inputs.budget.remaining_ms());
    inputs
        .observer
        .observe_prepared(&payload, CITATION_STAGE, questions);
    remember_observation(cache_key);
    messages
}

// ---------------------------------------------------------------------------
// Retrieval-safety battery (its OWN typed request under the scheduler cap)
// ---------------------------------------------------------------------------

/// Everything the battery round needs, resolved once by the bridge wiring.
/// The battery NEVER rides the filter request: the scheduler's real
/// per-request question cap (default 16) would silently refuse a combined
/// filter+battery request and kill the filter under the flag. The battery is
/// its own typed request; planned drops are covered first (the veto
/// targets), then retained candidates up to the explicit battery cap.
pub struct BatteryInputs<'a> {
    pub observer: Arc<JevObserver>,
    pub session_id: &'a str,
    pub turn: u64,
    pub mode: JevMode,
    pub policy_generation: &'a str,
    pub max_decision_age: Duration,
    pub budget: &'a SearchBudget,
    /// Fresh cancellation probe (the bridge closes over `ctx.signal()`).
    pub is_cancelled: &'a (dyn Fn() -> bool + Sync),
}

/// Mirror of the observer's explicit-request state precondition
/// (prepare_explicit refuses over-cap states silently; refusing here is
/// truthful and transport-free): raw serialized state <= 8 KiB.
fn battery_request_admissible(state: &Value) -> bool {
    serde_json::to_vec(state)
        .map(|bytes| bytes.len())
        .unwrap_or(usize::MAX)
        <= 8192
}

/// Run ONE battery request over the covered subset and return the outcome
/// only when it is fully usable. Fail-open: any refusal, skip, stale,
/// cancelled or over-cap condition yields `None` and the caller proceeds
/// exactly as if the battery had not run.
async fn decide_safety_battery(
    inputs: &BatteryInputs<'_>,
    state: &Value,
) -> Option<ActiveDecideOutcome> {
    if (inputs.is_cancelled)() || inputs.budget.expired() {
        return None;
    }
    if !battery_request_admissible(state) {
        return None;
    }
    let Some(questions) = safety_questions(state) else {
        return None;
    };
    if questions.is_empty() {
        return None;
    }
    let mut payload = json!({
        "session_id": inputs.session_id,
        "turn": inputs.turn,
        "policy_generation": inputs.policy_generation,
        "state": state,
    });
    payload["decision_timeout_ms"] = json!(inputs.budget.remaining_ms());
    let policy = ActivationPolicy {
        enabled_categories: [DecisionCategory::CodeRetrievalSafety]
            .into_iter()
            .collect(),
        // Typed acceptance; a Noul is never confidence-gated.
        min_confidence: 0.0,
        max_decision_age: inputs.max_decision_age,
    };
    let outcome = inputs
        .observer
        .decide_prepared(&payload, SAFETY_STAGE, questions, &policy)
        .await;
    let usable = !(inputs.is_cancelled)()
        && inputs.observer.can_apply(&outcome)
        && outcome.unavailable.is_none()
        && outcome.raw.as_ref().is_some_and(|raw| raw.skips.is_empty())
        && outcome.turn == inputs.turn;
    if !usable {
        None
    } else {
        Some(outcome)
    }
}

/// Collect (envelope index, excerpt) over ALL filter-stage candidates of the
/// presentation, plus the first batch's task excerpt.
fn presentation_candidates(presentation: &SearchPresentation) -> (String, Vec<(usize, String)>) {
    let mut task_excerpt = String::new();
    let mut envelope_excerpts: Vec<(usize, String)> = Vec::new();
    for batch in &presentation.batches {
        if task_excerpt.is_empty() {
            task_excerpt = batch
                .state
                .get("user_text_excerpt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
        }
        let Some(candidates) = batch
            .state
            .get("code_search_candidates")
            .and_then(Value::as_array)
        else {
            continue;
        };
        for (ordinal, candidate) in candidates.iter().enumerate() {
            let Some(envelope_index) = batch.candidate_indices.get(ordinal).copied() else {
                continue;
            };
            let Some(excerpt) = candidate.get("excerpt").and_then(Value::as_str) else {
                continue;
            };
            envelope_excerpts.push((envelope_index, excerpt.to_string()));
        }
    }
    (task_excerpt, envelope_excerpts)
}

/// The battery round: select the explicit covered subset (planned drops
/// first), run ONE typed battery request, apply the contradiction veto, and
/// build the bounded scope-disclosing annotation. Returns
/// `(removals_after_veto, Option<annotation block>, metadata)`. Fail-open:
/// a refused battery leaves the removals untouched and attaches nothing.
/// The accepted battery outcome is recorded truthfully with the scope
/// metadata; refused rounds record nothing (no request, no fabrication).
#[allow(clippy::too_many_arguments)]
pub async fn run_safety_veto(
    inputs: &BatteryInputs<'_>,
    presentation: &SearchPresentation,
    removals: Vec<usize>,
) -> (
    Vec<usize>,
    Option<Value>,
    BTreeMap<String, String>,
    Option<ActiveDecideOutcome>,
) {
    // Returns `(removals_after_veto, annotation_block, battery_metadata,
    // correlated_battery_outcome)`. The outcome is `None` exactly when the
    // round was refused or the flag did not fund it; the caller records it
    // with the attach state so assessment and annotation stay distinct.

    let (task_excerpt, envelope_excerpts) = presentation_candidates(presentation);
    let retained: Vec<usize> = envelope_excerpts
        .iter()
        .map(|(index, _)| *index)
        .filter(|index| !removals.contains(index))
        .collect();
    let covered = select_battery_candidates(&removals, &retained, SAFETY_BATTERY_CANDIDATE_CAP);
    let scope = BatteryScope {
        covered: covered.clone(),
        planned_removals_total: removals.len(),
        planned_removals_covered: covered
            .iter()
            .filter(|index| removals.contains(index))
            .count(),
    };
    if covered.is_empty() {
        return (removals, None, BTreeMap::new(), None);
    }
    let covered_excerpts: Vec<(usize, String)> = covered
        .iter()
        .filter_map(|index| {
            envelope_excerpts
                .iter()
                .find(|(candidate, _)| candidate == index)
                .cloned()
        })
        .collect();
    if covered_excerpts.len() != covered.len() {
        // Cannot prove every covered candidate carries its excerpt: refuse.
        return (removals, None, refused_metadata(&scope), None);
    }
    let state = battery_state(&task_excerpt, &covered_excerpts);
    let Some(outcome) = decide_safety_battery(inputs, &state).await else {
        return (removals, None, refused_metadata(&scope), None);
    };
    let labels = safety_labels_from_decisions(&outcome.decisions, covered.len());
    let mapping: Vec<(usize, usize)> = covered
        .iter()
        .enumerate()
        .map(|(ask, index)| (ask, *index))
        .collect();
    let (kept, vetoed) = vetoed_removals(&mapping, &labels, &removals);
    let metadata = safety_metadata(&labels, &vetoed, &scope);
    let annotation = safety_annotation_json(&labels, &vetoed, &scope);
    // The correlated outcome is RETURNED so the caller can record it with the
    // attach outcome (attached / skipped_multi_block / withheld_stale_gates)
    // and fold its freshness into the effect-site gate. Assessment and
    // annotation are recorded distinctly; refusal rounds return None.
    (kept, annotation, metadata, Some(outcome))
}

/// Compare mode: observe the battery questions for the explicit retained
/// subset and remember the observation; nothing changes, nothing is applied.
pub fn observe_safety_battery(inputs: &BatteryInputs<'_>, presentation: &SearchPresentation) {
    if (inputs.is_cancelled)() || !inputs.mode.allows_compare() {
        return;
    }
    let (task_excerpt, envelope_excerpts) = presentation_candidates(presentation);
    let retained: Vec<usize> = envelope_excerpts.iter().map(|(index, _)| *index).collect();
    let covered = select_battery_candidates(&[], &retained, SAFETY_BATTERY_CANDIDATE_CAP);
    if covered.is_empty() {
        return;
    }
    let covered_excerpts: Vec<(usize, String)> = covered
        .iter()
        .filter_map(|index| {
            envelope_excerpts
                .iter()
                .find(|(candidate, _)| candidate == index)
                .cloned()
        })
        .collect();
    if covered_excerpts.len() != covered.len() {
        return;
    }
    let state = battery_state(&task_excerpt, &covered_excerpts);
    let Some(questions) = safety_questions(&state) else {
        return;
    };
    if questions.is_empty() {
        return;
    }
    let cache_key = fingerprint_of(&json!([
        inputs.session_id,
        inputs.turn,
        inputs.mode.as_str(),
        inputs.policy_generation,
        SAFETY_STAGE,
        fingerprint_of(&state),
    ]));
    if already_observed(&cache_key) {
        return;
    }
    let mut payload = json!({
        "session_id": inputs.session_id,
        "turn": inputs.turn,
        "policy_generation": inputs.policy_generation,
        "state": state,
    });
    payload["decision_timeout_ms"] = json!(inputs.budget.remaining_ms());
    inputs
        .observer
        .observe_prepared(&payload, SAFETY_STAGE, questions);
    remember_observation(cache_key);
}

/// Additive advisory annotation attach, post-veto: the caller passes the
/// block built by `run_safety_veto`; the still-current gate decides whether
/// it lands (the bridge attaches inside its gate).
/// Attaches the advisory block and reports whether it landed. `false` means
/// the single-block guard refused (the target already carries more than one
/// content block) — the caller records that state instead of losing it.
pub fn attach_safety_block(messages: &mut [Value], message_index: usize, block: Value) -> bool {
    attach_block(messages, message_index, block).is_some()
}

/// True for the bounded advisory annotation texts this lane attaches to
/// request copies. Compaction's eligibility scan uses this so an annotated
/// tool result is left untouched: never truncated, never dropped. The check
/// is a cheap exact prefix on the serialized single-key envelope.
/// True for the bounded advisory annotation texts the Jev lanes attach to
/// request copies (retrieval-safety, citation, line-find). Compaction's
/// eligibility scan uses this so an annotated tool result is left untouched:
/// never truncated, never dropped. The citation reachability also uses it to
/// strip PRIOR advisory blocks from a source read while keeping the strict
/// single-ORIGINAL-text-block eligibility. The check is a cheap exact prefix
/// on the serialized single-key envelope — no regex, no user-data stripping.
pub fn is_evidence_annotation_text(text: &str) -> bool {
    text.starts_with("{\"jev_retrieval_safety\":")
        || text.starts_with("{\"jev_citation_check\":")
        || text.starts_with("{\"jev_line_find\":")
}

/// Splits a toolResult content array into the single ORIGINAL source text
/// block and prior advisory annotations. Strict: an image block, a second
/// original text block, or any unrecognised extra block makes the content
/// ineligible (`None`). Recognised advisory blocks are ignored for
/// eligibility and are never merged into the source text.
fn original_source_block<'a>(content: &'a [Value]) -> Option<&'a Value> {
    let mut original: Option<&Value> = None;
    for block in content {
        if block.get("type").and_then(Value::as_str) != Some("text") {
            return None;
        }
        let text = block.get("text").and_then(Value::as_str)?;
        if is_evidence_annotation_text(text) {
            continue;
        }
        if original.is_some() {
            return None;
        }
        original = Some(block);
    }
    original
}

/// Truthful telemetry for a refused battery round: the scope that WOULD have
/// been covered, explicitly marked refused. Zero effects either way.
fn refused_metadata(scope: &BatteryScope) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("jev_safety_battery".to_string(), "refused".to_string()),
        (
            "jev_safety_prompt".to_string(),
            EVIDENCE_PROMPT_VERSION.to_string(),
        ),
        (
            "jev_safety_covered".to_string(),
            scope
                .covered
                .iter()
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ),
        (
            "jev_safety_drops_covered".to_string(),
            format!(
                "{}/{}",
                scope.planned_removals_covered, scope.planned_removals_total
            ),
        ),
    ])
}

/// The single-annotation attach discipline, shared by both stages: only a
/// toolResult with exactly one content block accepts an additive block.
/// Composition attach: the target's ORIGINAL source must still be the single
/// original text block (`original_source_block`), a citation annotation must
/// not already be present, and no unrecognised extra block may exist. The new
/// block is APPENDED after the existing bounded advisory annotations; existing
/// content is untouched and session history keeps the original.
fn attach_advisory_block(messages: &mut [Value], message_index: usize, block: Value) -> bool {
    let Some(content) = messages
        .get_mut(message_index)
        .and_then(|message| message["content"].as_array_mut())
    else {
        return false;
    };
    if content.iter().any(|existing| {
        existing
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.starts_with("{\"jev_citation_check\":"))
    }) {
        return false;
    }
    if original_source_block(content).is_none() {
        return false;
    }
    content.push(block);
    true
}

fn attach_block(messages: &mut [Value], message_index: usize, block: Value) -> Option<()> {
    let message = messages.get_mut(message_index)?;
    if message["role"] != "toolResult" {
        return None;
    }
    let content = message.get_mut("content")?.as_array_mut()?;
    if content.len() != 1 {
        return None;
    }
    content.push(block);
    Some(())
}

// ---------------------------------------------------------------------------
// Reachability scan (self-owned copy of the deterministic source-read idiom)
// ---------------------------------------------------------------------------

/// Capture the production citation basis before any request-local Jev stage
/// can add an advisory block. The eligible native result must have exactly one
/// original text block at capture time. Prefix-shaped source text and every
/// pre-existing multi-block result fail closed; later composition can therefore
/// accept only the same fingerprinted original plus host-added annotations.
pub fn prepare_original_citation(messages: &[Value], claim: &str) -> Option<CitationPresentation> {
    let presentation = prepare_citation(messages, claim).into_iter().next()?;
    let content = messages
        .get(presentation.message_index)?
        .get("content")?
        .as_array()?;
    if content.len() != 1 {
        return None;
    }
    let text = content[0].get("text").and_then(Value::as_str)?;
    if content[0].get("type").and_then(Value::as_str) != Some("text")
        || is_evidence_annotation_text(text)
    {
        return None;
    }
    Some(presentation)
}

/// Recognize eligible citation inputs in the request window, most recent
/// first: the deterministic ipython source-read idiom only. Mirrors the
/// line-find reachability (message caps, provider-context start, excluded
/// and clean classes, paired-call uniqueness, single-text-block toolResult).
pub fn prepare_citation(messages: &[Value], claim: &str) -> Vec<CitationPresentation> {
    if messages.len() > 512 || claim.trim().is_empty() {
        return Vec::new();
    }
    if messages.iter().any(|message| {
        message["content"]
            .as_array()
            .is_some_and(|blocks| blocks.len() > 64)
    }) {
        return Vec::new();
    }
    let start = messages
        .iter()
        .rposition(|message| {
            message
                .get("providerContext")
                .is_some_and(|value| !value.is_null())
        })
        .map_or(0, |index| index + 1);
    let mut presentations = Vec::new();
    for (index, message) in messages.iter().enumerate().skip(start).rev() {
        if excluded_message(message) || !clean_details(message) {
            continue;
        }
        // Strict eligibility on the ORIGINAL source: exactly one original text
        // block; prior bounded advisory annotation blocks (line-find, safety,
        // citation) are ignored for eligibility and never merged into the
        // source text. Images, second originals, or unrecognised extra blocks
        // keep the message ineligible.
        let Some(original) = message["content"]
            .as_array()
            .and_then(|content| original_source_block(content))
        else {
            continue;
        };
        if original.get("textSignature").is_some() {
            continue;
        }
        let Some(text) = original["text"]
            .as_str()
            .filter(|text| text.len() <= MAX_PRESENTATION_BYTES)
        else {
            continue;
        };
        let Some(id) = message["toolCallId"].as_str().filter(|id| !id.is_empty()) else {
            continue;
        };
        let Some((call_at, call)) = paired_call(messages, id, index) else {
            continue;
        };
        if call_at < start {
            continue;
        }
        if messages
            .iter()
            .filter(|message| message["role"] == "toolResult" && message["toolCallId"] == id)
            .count()
            != 1
        {
            continue;
        }
        let code = call["arguments"]
            .get("code")
            .and_then(Value::as_str)
            .or_else(|| call["input"].get("code").and_then(Value::as_str));
        let Some(code) = code else {
            continue;
        };
        let Some(path) = parse_source_read(code) else {
            continue;
        };
        // Byte-cap the span; report the truncation truthfully.
        let truncated = text.len() > MAX_CITATION_SPAN_BYTES;
        let span = if truncated {
            String::from_utf8_lossy(&text.as_bytes()[..MAX_CITATION_SPAN_BYTES]).into_owned()
        } else {
            text.to_string()
        };
        let span_bytes = span.len();
        let span_excerpt = bounded_excerpt(&span, pi_jev::evidence::CITATION_SPAN_EXCERPT_CHARS);
        // Fingerprint BEFORE the struct literal: json! only borrows here.
        let fingerprint = fingerprint_of(&json!([id, text, claim]));
        presentations.push(CitationPresentation {
            message_index: index,
            tool_call_id: id.to_string(),
            path,
            span,
            span_excerpt,
            span_bytes,
            span_truncated: truncated,
            fingerprint,
        });
        break;
    }
    presentations
}

/// Excluded message classes, shared with the search presentation discipline
/// (self-owned copy; identical rules).
fn excluded_message(message: &Value) -> bool {
    message["role"] != "toolResult"
        || message["toolName"] != "ipython"
        || message["isError"] != false
        || message.get("textSignature").is_some()
        || ["pinned", "mandatory", "instructions", "edited"]
            .iter()
            .any(|key| {
                message
                    .get(*key)
                    .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
                    || message["details"]
                        .get(*key)
                        .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
            })
}

fn clean_details(message: &Value) -> bool {
    let details = &message["details"];
    details["status"] == "ok"
        && details["kernelRestarted"] != true
        && ["stderr", "backgroundOutput", "result"].iter().all(|key| {
            details
                .get(*key)
                .is_none_or(|value| value.is_null() || value == "")
        })
        && details
            .get("stdout")
            .and_then(Value::as_str)
            .is_some_and(|stdout| !stdout.is_empty())
}

/// Find the single paired assistant ipython call for a tool result id.
fn paired_call<'a>(messages: &'a [Value], id: &str, before: usize) -> Option<(usize, &'a Value)> {
    let mut found: Option<(usize, &Value)> = None;
    for (at, message) in messages.iter().enumerate().take(before) {
        for call in message["content"].as_array().into_iter().flatten() {
            if message["role"] == "assistant"
                && call["type"] == "toolCall"
                && call["id"] == id
                && call["name"] == "ipython"
            {
                if found.is_some() {
                    return None;
                }
                found = Some((at, call));
            }
        }
    }
    found
}

/// Parse a single-quoted or double-quoted string literal. Fails on escapes
/// (a path idiom never needs them), embedded quotes and newlines.
fn parse_string_literal(text: &str) -> Option<(&str, &str)> {
    let quote = text.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let rest = &text[1..];
    let end = rest.find(quote)?;
    let literal = &rest[..end];
    if literal.contains('\\') || literal.contains(quote) || literal.contains('\n') {
        return None;
    }
    Some((literal, &rest[end + 1..]))
}

/// Recognize the deterministic source-read idiom in a paired ipython cell:
/// `print(open(PATH).read())`, `print(open(PATH, "r").read())`,
/// `print(open(PATH, mode="r").read())`, `print(Path(PATH).read_text())`,
/// `print(Path(PATH).read_text(encoding="..."))`, optionally with a trailing
/// semicolon. The exact compaction-safe `from pathlib import Path\n` prefix is
/// also accepted only with its exact `print(Path(PATH).read_text())` spelling.
/// Everything else — aliases, write modes, readlines, bare expressions, or
/// extra statements — is rejected (fail closed).
fn parse_source_read(code: &str) -> Option<String> {
    let code = code.trim();
    let (code, imported_pathlib) = match code.strip_prefix("from pathlib import Path\n") {
        Some(expression) => (expression, true),
        None => (code.trim_end_matches(';').trim(), false),
    };
    if imported_pathlib {
        code.strip_prefix("print(Path(")?
            .strip_suffix(").read_text())")?;
    }
    let body = code.strip_prefix("print(")?.strip_suffix(')')?;
    let (path, rest) = if let Some(after) = body.strip_prefix("open(") {
        let (path, rest) = parse_string_literal(after)?;
        let rest = match rest.strip_prefix(',') {
            Some(rest) => {
                // Only the advertised read-only mode and optional encoding are
                // accepted inside open(...); update/write modes fail closed.
                let rest = rest.trim_start();
                let (mode, rest) = if rest.starts_with("mode") {
                    let rest = rest
                        .strip_prefix("mode")?
                        .trim_start()
                        .strip_prefix('=')?
                        .trim_start();
                    parse_string_literal(rest)?
                } else {
                    parse_string_literal(rest)?
                };
                if mode != "r" {
                    return None;
                }
                match rest.strip_prefix(',') {
                    Some(rest) => {
                        let rest = rest.trim_start();
                        let rest = rest
                            .strip_prefix("encoding")?
                            .trim_start()
                            .strip_prefix('=')?
                            .trim_start();
                        let (_encoding, rest) = parse_string_literal(rest)?;
                        rest
                    }
                    None => rest,
                }
            }
            None => rest,
        };
        let rest = rest.strip_prefix(')')?;
        if !rest.starts_with(".read()") {
            return None;
        }
        (path, &rest[".read()".len()..])
    } else if let Some(after) = body.strip_prefix("Path(") {
        let (path, rest) = parse_string_literal(after)?;
        let rest = rest.strip_prefix(')')?;
        let rest = rest.strip_prefix(".read_text(")?;
        let rest = match rest.strip_suffix(')') {
            Some(inner) => {
                let inner = inner.trim();
                if inner.is_empty() {
                    ""
                } else {
                    let inner = inner
                        .strip_prefix("encoding")?
                        .trim_start()
                        .strip_prefix('=')?
                        .trim_start();
                    let (_encoding, rest) = parse_string_literal(inner)?;
                    rest
                }
            }
            None => return None,
        };
        if !rest.is_empty() {
            return None;
        }
        (path, "")
    } else {
        return None;
    };
    if !rest.is_empty() {
        return None;
    }
    let path = path.trim();
    if path.is_empty() || path.chars().count() > 1024 {
        return None;
    }
    // Instruction-class paths are never evidence sources (same class as
    // pinned search candidates).
    let lowered = path.to_ascii_lowercase();
    if lowered.ends_with(".md")
        || lowered.ends_with(".mdx")
        || lowered.ends_with(".mdc")
        || lowered.contains("agents")
        || lowered.contains("instruction")
        || lowered.contains("prompt")
        || lowered.contains("claude")
    {
        return None;
    }
    Some(path.to_string())
}
