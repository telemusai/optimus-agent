//! Shared decision-domain types and the SystemOne wire contract (DESIGN.md sections 2 and 3).
//!
//! Exact struct/enum names follow DESIGN.md section 3.1; both lanes use these names.
//! Wire serde: externally tagged enums (`{"type":"noul",...}`) with
//! `#[serde(tag = "type", rename_all = "snake_case")]`.
//!
//! Validation is total: an answer is accepted only when the id set matches exactly and every
//! field obeys the documented contract. A violation produces an `AnswerIssue` (a logged skip);
//! the client never fabricates a value for a malformed answer.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::JevError;

/// SystemOne evaluation endpoint (official docs, verified 2026-09-19).
pub const SYSTEMONE_PATH: &str = "/v1/systemone";

/// Default model alias from the official docs; it resolves to `jev-1.13.0`.
pub const DEFAULT_MODEL: &str = "jev-latest";

/// Absolute probability-sum tolerance. The API returns floats that sum to 1;
/// a real distribution stays far inside this bound.
pub const PROBABILITY_TOLERANCE: f64 = 1e-6;

/// Host-policy argmax tolerance for FULL-PRECISION distributions (no 2-decimal
/// quantization detected). Float noise between the server's argmax computation and the
/// reported values is far below this bound. Quantized distributions use the wider,
/// rounding-derived `QUANTIZED_ARGMAX_TOLERANCE` instead.
pub const ARGMAX_TOLERANCE: f64 = 1e-9;

/// Maximum accepted sub-question count per request (speculative fan-out stays bounded).
pub const MAX_QUESTIONS_PER_REQUEST: usize = 64;

/// HOST POLICY, not an API guarantee: upper bound for one structured (object/array) entry
/// when it is serialized for hashing and size checks. The official docs define no wire size
/// limit for entries; this bound exists so a hostile builder cannot blow up hashing.
/// Bare-string entries are NOT size-capped here (legacy valid strings are never rejected).
pub const MAX_ENTRY_JSON_BYTES: usize = 2048;

/// HOST POLICY, not an API guarantee: nesting-depth cap for structured (object/array) entry
/// forms. Applies to Json variants only; bare strings are unaffected.
pub const MAX_ENTRY_JSON_DEPTH: usize = 6;

/// Documented API limit, enforced BEFORE the transport call (mirrors the official
/// contract): api.md "Request body", Choice criteria - "You can have a maximum of 255
/// options per Choice"; choice.md corroborates - "A Choice question accepts up to 255
/// options". The cap also keeps speculative fan-out bounded. Documented valid shapes
/// below the cap are untouched.
pub const MAX_CHOICES_PER_QUESTION: usize = 255;

/// Documented API limits, enforced BEFORE the transport call (mirrors the official
/// contract): api.md Score criteria - "A Score should have at least two levels; the API
/// accepts up to 10"; score.md corroborates - "Should have at least two levels; the API
/// accepts up to 10" and "Use as many levels as you can describe distinctly, up to 10".
/// The 10-level ceiling also bounds distribution size and the expectation-tolerance
/// math. Documented examples use 2-5 levels.
pub const MAX_SCORE_LEVELS: usize = 10;

/// HOST POLICY, not an API guarantee: default ceiling on the ESTIMATED token size of one
/// request (state + questions). The estimate is a heuristic (see `estimate_tokens`); the
/// client skips the transport call when the estimate exceeds the ceiling. `JevLimits`
/// may lower it (`max_request_tokens`); values above the default are ignored.
pub const REQUEST_TOKEN_CEILING: usize = 30_000;

/// Host-policy tolerance for the Choice argmax check on distributions whose values are
/// all exact to two decimal places (the documented examples all display 2-decimal values).
/// A displayed value rounds by at most 0.005, so the reported peak can differ from the
/// true argmax by at most 0.01.
pub const QUANTIZED_ARGMAX_TOLERANCE: f64 = 0.01;

/// Absolute float-noise slack for tolerance comparisons. Probability deltas that matter
/// are >= 0.01 (quantized) or >= 1e-6 (full precision), while f64 noise on values <= 1
/// stays around 1e-16 even after a handful of operations; 1e-12 sits far below any
/// meaningful delta and above the noise, so an exact-boundary case (gap == tolerance)
/// is accepted deterministically instead of flipping on one ULP.
pub const FLOAT_NOISE_SLACK: f64 = 1e-12;

/// Host-policy tolerance for the Score expectation check on 2-decimal-quantized
/// distributions: worst case `sum(|i * delta_i|) = 0.005 * (N(N-1)/2)` from rounding each
/// probability plus 0.005 from the score's own display rounding.
pub fn quantized_expectation_tolerance(levels: usize) -> f64 {
    0.005 * ((levels as f64) * ((levels as f64) - 1.0) / 2.0) + 0.005
}

/// True when every probability is within 1e-9 of an exact two-decimal value, i.e. the
/// server most likely displayed (and rounded) the values at 2 decimal places.
pub fn is_two_decimal_quantized(values: impl IntoIterator<Item = f64>) -> bool {
    values.into_iter().all(|value| {
        if !value.is_finite() {
            return false;
        }
        let rounded = (value * 100.0).round() / 100.0;
        (value - rounded).abs() <= 1e-9
    })
}

// ---------------------------------------------------------------------------
// Entry values (string | object | array | null)
// ---------------------------------------------------------------------------

/// One entry value as the wire accepts it: a bare string, `null`, or structured JSON.
///
/// Untagged with `Text` FIRST: a bare string serializes exactly as before (byte-identical
/// legacy wire), `null` serializes as `null`, and structured values serialize as JSON.
/// The `Null` variant is declared BEFORE `Json` so untagged deserialization maps JSON
/// `null` to `Null` instead of letting `Value::Null` be swallowed by the `Json` arm.
/// There is no implicit `String -> EntryValue` conversion on the wire; builders convert
/// explicitly (`EntryValue::from(...)`, `EntryValue::text(...)`, or the question helpers).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EntryValue {
    /// Bare string (legacy form; byte-identical serialization).
    Text(String),
    /// Documented `null` form.
    Null,
    /// Structured form: a JSON object or array (validate_shape rejects scalars here).
    Json(Value),
}

impl EntryValue {
    /// Convenience constructor for the bare-string form.
    pub fn text(value: impl Into<String>) -> Self {
        EntryValue::Text(value.into())
    }

    /// The bare string, when this entry is one.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            EntryValue::Text(text) => Some(text),
            _ => None,
        }
    }

    /// True for the documented `null` form.
    pub fn is_null(&self) -> bool {
        matches!(self, EntryValue::Null)
    }

    /// The semantic JSON value of this entry (`Text("x")` == `Json(Value::String("x"))`).
    pub fn canonical_value(&self) -> Value {
        match self {
            EntryValue::Text(text) => Value::String(text.clone()),
            EntryValue::Null => Value::Null,
            EntryValue::Json(value) => value.clone(),
        }
    }

    /// Semantic equality: two entries are equal when their canonical JSON values are equal,
    /// so a Text echo of an object-valued legend entry still compares equal.
    pub fn equivalent(&self, other: &EntryValue) -> bool {
        self.canonical_value() == other.canonical_value()
    }

    /// HOST POLICY depth of this value (strings count as depth 0).
    pub fn json_depth(&self) -> usize {
        fn depth(value: &Value) -> usize {
            match value {
                Value::Object(map) => 1 + map.values().map(depth).max().unwrap_or(0),
                Value::Array(items) => 1 + items.iter().map(depth).max().unwrap_or(0),
                _ => 0,
            }
        }
        match self {
            EntryValue::Json(value) => depth(value),
            _ => 0,
        }
    }

    /// HOST POLICY serialized size of this value.
    pub fn json_size(&self) -> usize {
        serde_json::to_string(&self.canonical_value())
            .map(|encoded| encoded.len())
            .unwrap_or(usize::MAX)
    }
}

impl Default for EntryValue {
    /// `null` is the neutral entry value (an absent or empty side is `Null`, not a guess).
    fn default() -> Self {
        EntryValue::Null
    }
}

impl From<String> for EntryValue {
    fn from(value: String) -> Self {
        EntryValue::Text(value)
    }
}

impl From<&str> for EntryValue {
    fn from(value: &str) -> Self {
        EntryValue::Text(value.to_string())
    }
}

/// Canonical JSON encoding for prompt hashing: object keys sorted, arrays preserved.
/// Reuses the mock's canonicalizer so hash input is stable regardless of key insertion order.
pub fn canonical_entry_json(value: &Value) -> String {
    serde_json::to_string(&crate::mock::canonicalize(value)).unwrap_or_default()
}

/// Validates one entry against the documented semantics plus the HOST POLICY bounds.
///
/// Documented semantics (applied to all forms): bare strings must be non-empty after
/// trimming (pre-existing meaningful-builder policy). `null` entries are a documented
/// form and pass. Structured entries must be an object or an array and non-empty
/// (a bare scalar is accepted by the docs' string form only as a bare string).
/// Size/depth caps apply to Json variants only (HOST POLICY; strings are untouched).
pub fn validate_entry_shape(entry: &EntryValue, what: &str) -> Result<(), JevError> {
    match entry {
        EntryValue::Null => Ok(()),
        EntryValue::Text(text) => {
            if text.trim().is_empty() {
                return Err(JevError::validation(format!("{what} is an empty string")));
            }
            Ok(())
        }
        EntryValue::Json(value) => match value {
            Value::Object(map) if map.is_empty() => {
                Err(JevError::validation(format!("{what} is an empty object")))
            }
            Value::Array(items) if items.is_empty() => {
                Err(JevError::validation(format!("{what} is an empty array")))
            }
            Value::Object(_) | Value::Array(_) => {
                if entry.json_size() > MAX_ENTRY_JSON_BYTES {
                    return Err(JevError::validation(format!(
                        "{what} exceeds the host-policy structured entry size limit ({MAX_ENTRY_JSON_BYTES} bytes)"
                    )));
                }
                if entry.json_depth() > MAX_ENTRY_JSON_DEPTH {
                    return Err(JevError::validation(format!(
                        "{what} exceeds the host-policy structured entry depth limit ({MAX_ENTRY_JSON_DEPTH})"
                    )));
                }
                Ok(())
            }
            _ => Err(JevError::validation(format!(
                "{what} must be a string, object, array, or null"
            ))),
        },
    }
}

/// Validates instructions: absent or `null` is a documented form and passes; a bare
/// string must be non-empty after trimming (pre-existing policy); structured entries
/// obey the documented object/array requirement plus the host-policy bounds.
pub fn validate_instructions_entry(instructions: Option<&EntryValue>) -> Result<(), JevError> {
    match instructions {
        None => Ok(()),
        Some(entry) => validate_entry_shape(entry, "instructions"),
    }
}

/// Deserializes a PRESENT instructions field: JSON `null` maps to `Some(EntryValue::Null)`
/// (the documented null form), NOT to `None`. Field absence is handled separately by
/// `#[serde(default)]` (-> `None`), so the null form and the absent form stay
/// distinguishable and both roundtrip faithfully.
fn deserialize_present_entry<'de, D>(deserializer: D) -> Result<Option<EntryValue>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Some(EntryValue::deserialize(deserializer)?))
}

/// `{"true": .., "false": ..}` descriptions for a Noul question.
///
/// Each side is an `EntryValue`: the documented per-side `null` form is `EntryValue::Null`
/// and an absent side deserializes to `Null` (`#[serde(default)]`). Both sides always
/// serialize (a `Null` side emits `null`), so legacy both-string criteria stay
/// byte-identical. At least one side must be non-Null (validated in `validate_shape`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(default, rename = "true")]
    pub r#true: EntryValue,
    #[serde(default, rename = "false")]
    pub r#false: EntryValue,
}

impl NoulCriteria {
    /// Convenience constructor for the legacy both-strings form.
    pub fn text(yes: impl Into<String>, no: impl Into<String>) -> Self {
        Self {
            r#true: EntryValue::text(yes),
            r#false: EntryValue::text(no),
        }
    }
}

/// One typed question. Serde writes `{"type": ..., "instructions": ..., "criteria": ...}`.
///
/// `instructions` accepts the documented string, object, array, and `null` forms plus
/// absence (`None` serializes as an omitted field). A bare string serializes exactly as
/// before, so every legacy builder stays byte-identical on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QuestionSpec {
    Noul {
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present_entry"
        )]
        instructions: Option<EntryValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present_entry"
        )]
        instructions: Option<EntryValue>,
        /// Option descriptions: `EntryValue::Null` is the documented null description.
        criteria: BTreeMap<String, EntryValue>,
    },
    Score {
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present_entry"
        )]
        instructions: Option<EntryValue>,
        /// Ordered level descriptions; a `Null` element is the documented null level.
        criteria: Vec<EntryValue>,
    },
}

impl QuestionSpec {
    /// Convenience constructor: Noul question with bare-string instructions and criteria.
    pub fn noul(instructions: impl Into<EntryValue>, criteria: NoulCriteria) -> Self {
        QuestionSpec::Noul {
            instructions: Some(instructions.into()),
            criteria: Some(criteria),
        }
    }

    /// Convenience constructor: Choice question with bare-string instructions and
    /// `(option, description)` pairs (a `None` description becomes `EntryValue::Null`).
    pub fn choice(
        instructions: impl Into<EntryValue>,
        options: impl IntoIterator<Item = (impl Into<String>, Option<impl Into<String>>)>,
    ) -> Self {
        QuestionSpec::Choice {
            instructions: Some(instructions.into()),
            criteria: options
                .into_iter()
                .map(|(option, description)| {
                    (
                        option.into(),
                        description
                            .map(|value| EntryValue::Text(value.into()))
                            .unwrap_or(EntryValue::Null),
                    )
                })
                .collect(),
        }
    }

    /// Convenience constructor: Score question with bare-string instructions and levels.
    pub fn score(instructions: impl Into<EntryValue>, levels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        QuestionSpec::Score {
            instructions: Some(instructions.into()),
            criteria: levels.into_iter().map(|level| EntryValue::Text(level.into())).collect(),
        }
    }

    /// Wire type tag of this question.
    pub fn question_type(&self) -> QuestionType {
        match self {
            QuestionSpec::Noul { .. } => QuestionType::Noul,
            QuestionSpec::Choice { .. } => QuestionType::Choice,
            QuestionSpec::Score { .. } => QuestionType::Score,
        }
    }

    /// Hash input for prompt-versioning: the bare string for `Text`, canonical JSON for
    /// `Json` and `Null` forms, and the empty string for absent instructions. The return
    /// type is owned because structured entries have no stable borrowed text form.
    pub fn instructions(&self) -> String {
        let entry = match self {
            QuestionSpec::Noul { instructions, .. }
            | QuestionSpec::Choice { instructions, .. }
            | QuestionSpec::Score { instructions, .. } => instructions,
        };
        match entry {
            Some(EntryValue::Text(text)) => text.clone(),
            Some(EntryValue::Null) => "null".to_string(),
            Some(EntryValue::Json(value)) => canonical_entry_json(value),
            None => String::new(),
        }
    }

    /// The bare instruction text, when the instructions are a bare string.
    pub fn instructions_text(&self) -> Option<&str> {
        match self {
            QuestionSpec::Noul { instructions, .. }
            | QuestionSpec::Choice { instructions, .. }
            | QuestionSpec::Score { instructions, .. } => {
                instructions.as_ref().and_then(EntryValue::as_text)
            }
        }
    }

    /// The option/level keys this question defines, in wire (sorted) order.
    pub fn criteria_keys(&self) -> Vec<String> {
        match self {
            QuestionSpec::Noul { .. } => Vec::new(),
            QuestionSpec::Choice { criteria, .. } => criteria.keys().cloned().collect(),
            QuestionSpec::Score { criteria, .. } => {
                (0..criteria.len()).map(|index| index.to_string()).collect()
            }
        }
    }

    /// Rejects shapes the API would reject (so the client never sends them).
    ///
    /// Documented semantics, enforced pre-transport: non-empty-trim instructions (bare
    /// strings stay non-empty-trim, pre-existing policy), `null`/absent forms pass,
    /// structured entries must be object/array and non-empty, Noul criteria keep at
    /// least one non-null side, choice options <= 255 and score levels 2..=10 (both
    /// documented limits, see the `MAX_CHOICES_PER_QUESTION` / `MAX_SCORE_LEVELS` doc
    /// comments). Only the structured-entry SIZE/DEPTH bounds are host policy
    /// (`MAX_ENTRY_JSON_BYTES` / `MAX_ENTRY_JSON_DEPTH`); all-null criteria are
    /// documented nowhere and are refused.
    pub fn validate_shape(&self) -> Result<(), JevError> {
        match self {
            QuestionSpec::Noul { instructions, criteria } => {
                validate_instructions_entry(instructions.as_ref())?;
                if let Some(criteria) = criteria {
                    let both_null = criteria.r#true.is_null() && criteria.r#false.is_null();
                    if both_null {
                        return Err(JevError::validation(
                            "noul criteria must keep at least one non-null side (documented: an object with true and false descriptions)",
                        ));
                    }
                    validate_entry_shape(&criteria.r#true, "noul criteria true side")?;
                    validate_entry_shape(&criteria.r#false, "noul criteria false side")?;
                }
                Ok(())
            }
            QuestionSpec::Choice { instructions, criteria } => {
                validate_instructions_entry(instructions.as_ref())?;
                if criteria.is_empty() {
                    return Err(JevError::validation("choice question has no criteria options"));
                }
                if criteria.len() > MAX_CHOICES_PER_QUESTION {
                    return Err(JevError::validation(format!(
                        "choice question has {} options, limit is {MAX_CHOICES_PER_QUESTION}",
                        criteria.len()
                    )));
                }
                if criteria.keys().any(|key| key.is_empty()) {
                    return Err(JevError::validation("choice question has an empty option key"));
                }
                for (key, entry) in criteria {
                    validate_entry_shape(entry, &format!("choice option `{key}` description"))?;
                }
                Ok(())
            }
            QuestionSpec::Score { instructions, criteria } => {
                validate_instructions_entry(instructions.as_ref())?;
                // The API requires at least two ordered levels.
                if criteria.len() < 2 {
                    return Err(JevError::validation("score question needs at least two levels"));
                }
                if criteria.len() > MAX_SCORE_LEVELS {
                    return Err(JevError::validation(format!(
                        "score question has {} levels, limit is {MAX_SCORE_LEVELS}",
                        criteria.len()
                    )));
                }
                for (index, entry) in criteria.iter().enumerate() {
                    validate_entry_shape(entry, &format!("score level {index} description"))?;
                }
                Ok(())
            }
        }
    }

    /// The level keys of a Score question, in order.
    pub fn score_levels(&self) -> Vec<String> {
        match self {
            QuestionSpec::Score { criteria, .. } => {
                (0..criteria.len()).map(|index| index.to_string()).collect()
            }
            _ => Vec::new(),
        }
    }
}

/// One SystemOne request: immutable state plus sibling questions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneRequest {
    pub state: Value,
    pub model: String,
    pub questions: BTreeMap<String, QuestionSpec>,
}

impl SystemOneRequest {
    /// Builds a request with the documented default model.
    pub fn new(state: Value, questions: BTreeMap<String, QuestionSpec>) -> Self {
        Self {
            state,
            model: DEFAULT_MODEL.to_string(),
            questions,
        }
    }

    /// Builds a request for one bundle (lane B's `DecisionBundle`).
    pub fn from_bundle(bundle: &DecisionBundle) -> Self {
        Self {
            state: bundle.state.clone(),
            model: bundle.model.clone(),
            questions: bundle.questions.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Wire response
// ---------------------------------------------------------------------------

/// Token usage for a request.
///
/// Knownness is explicit END-TO-END (root decision): a missing or `null` field
/// deserializes to `None` (UNKNOWN, never fabricated 0), and `Some(0)` is a real
/// measured zero. Fields are omitted from serialization when unknown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
}

impl Usage {
    /// UNKNOWN usage (absent or null on the wire): no token count is fabricated.
    pub fn unknown() -> Self {
        Self::default()
    }
}

/// Wire type tag of a question or an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionType {
    Noul,
    Choice,
    Score,
}

impl QuestionType {
    pub fn as_str(self) -> &'static str {
        match self {
            QuestionType::Noul => "noul",
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
        }
    }
}

/// Alias kept for callers that name the tag by answer side.
pub type AnswerType = QuestionType;

impl fmt::Display for QuestionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Typed acceptance for ONE rerank Noul score (ROOT CONTRACT v1, Search).
/// A Noul probability is NEVER converted into legacy Choice confidence and is
/// never gated by a Choice-confidence threshold; the legacy `evaluate_answer`
/// path stays untouched. An incomplete set retains the original order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankAssessment {
    /// Local ordinal inside the batch (`code_search_rerank.<id>`).
    pub candidate_id: usize,
    /// Raw Noul probability in [0,1]; not a confidence.
    pub noul: f64,
    /// The answer existed and parsed as a finite in-range Noul.
    pub complete: bool,
    /// Same request and turn as the decision context it was answered in.
    pub correlated: bool,
    /// Within the decision-age bound.
    pub fresh: bool,
}

/// Typed acceptance for ONE line-find pair (where Choice + existence Noul,
/// answered together in a single request). A partial pair never applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineFindAssessment {
    /// The selected where-line id (e.g. `L0123`), when the pair is complete.
    pub where_line: Option<String>,
    /// The independent existence Noul; `None` when the pair is incomplete.
    pub existence_noul: Option<f64>,
    /// Both answers of the pair arrived and parsed.
    pub complete: bool,
    /// Same request and turn as the decision context.
    pub correlated: bool,
    /// Within the decision-age bound.
    pub fresh: bool,
    /// The judgment covered one cascade window, not the whole text.
    pub windowed: bool,
    /// A supplied line was truncated before asking (disclosed).
    pub truncated: bool,
}

/// One typed answer. A Noul answer has NO confidence field (official docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        /// Level descriptions echoed by position. Structured entries are a documented
        /// form (score.md structured example); a `null` echo compares equal to a `Null`
        /// criteria entry.
        legend: BTreeMap<String, EntryValue>,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
}

impl Answer {
    /// Wire type tag of this answer.
    pub fn answer_type(&self) -> QuestionType {
        match self {
            Answer::Noul { .. } => QuestionType::Noul,
            Answer::Choice { .. } => QuestionType::Choice,
            Answer::Score { .. } => QuestionType::Score,
        }
    }

    /// Confidence for Choice/Score. Noul has no confidence field: `None`, not 0.0.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Answer::Noul { .. } => None,
            Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => Some(*confidence),
        }
    }

    /// The selected value as a display string (choice option, score value, noul probability).
    pub fn selected_value(&self) -> String {
        match self {
            Answer::Noul { noul } => format!("{noul}"),
            Answer::Choice { choice, .. } => choice.clone(),
            Answer::Score { score, .. } => format!("{score}"),
        }
    }

    /// The full probability distribution, when the answer type carries one.
    pub fn probabilities(&self) -> Option<&BTreeMap<String, f64>> {
        match self {
            Answer::Noul { .. } => None,
            Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => Some(probabilities),
        }
    }
}

/// One response from the SystemOne endpoint.
///
/// The two trailing fields are NOT part of the wire contract: `answer_parse_skips` is
/// filled by the production parser (lenient per-answer parsing) and `server_request_id`
/// by the HTTP transport from the untrusted `x-typesafe-request-id` header. Both are
/// `#[serde(skip)]` so the wire serialization is unchanged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub answers: BTreeMap<String, Answer>,
    #[serde(default)]
    pub usage: Usage,
    /// Per-answer parse failures recorded by the production parser (fail-open within
    /// the skip-not-fail rule: one malformed answer never invalidates the whole body).
    #[serde(skip)]
    pub answer_parse_skips: Vec<(String, AnswerIssue)>,
    /// Bounded, sanitized value of the untrusted `x-typesafe-request-id` response
    /// header, when the transport captured one. None when absent or refused.
    #[serde(skip)]
    pub server_request_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Why one answer (or the whole response) failed validation. Every variant is a skip reason.
///
/// Not `Eq`: several variants carry `f64` values, which are only `PartialEq`.
#[derive(Debug, Clone, PartialEq)]
pub enum AnswerIssue {
    /// A question id in the response was not requested.
    UnknownId { question_id: String },
    /// A requested question id had no answer.
    MissingId { question_id: String },
    /// The response type tag does not match the question type.
    TypeMismatch {
        question_id: String,
        expected: QuestionType,
        actual: QuestionType,
    },
    /// The selected Choice option is not one of the defined options.
    UnknownChoice { question_id: String, choice: String },
    /// The probability map keys do not match the criteria keys.
    DistributionKeysMismatch {
        question_id: String,
        missing: Vec<String>,
        extra: Vec<String>,
    },
    /// A probability is NaN, infinite, negative or greater than 1.
    ProbabilityOutOfRange {
        question_id: String,
        key: String,
        value: f64,
    },
    /// The probabilities do not sum to 1 within tolerance.
    ProbabilitiesNotSummingToOne { question_id: String, sum: f64 },
    /// Confidence is NaN, infinite or outside [0, 1].
    ConfidenceOutOfRange { question_id: String, value: f64 },
    /// A Score legend does not match the criteria levels.
    LegendMismatch {
        question_id: String,
        missing: Vec<String>,
        extra: Vec<String>,
    },
    /// The score value is NaN or infinite.
    ScoreNotFinite { question_id: String, value: f64 },
    /// The score value lies outside the defined level range.
    ScoreOutOfRange {
        question_id: String,
        value: f64,
        max: f64,
    },
    /// The noul value is NaN, infinite or outside [0, 1].
    NoulOutOfRange { question_id: String, value: f64 },
    /// The reported choice is not the distribution's argmax (within the documented
    /// rounding tolerance). The distribution itself was valid; the selection is not.
    ChoiceNotPeak {
        question_id: String,
        choice: String,
        peak: String,
    },
    /// The reported score is not the probability-weighted expected level (within the
    /// documented rounding tolerance).
    ScoreNotExpectation {
        question_id: String,
        value: f64,
        expected: f64,
    },
    /// A known answer `type` tag arrived with a payload that failed to parse
    /// (per-answer skip; the rest of the body is unaffected). `detail` is the
    /// bounded serde_json error class (one of io/syntax/data/eof).
    MalformedAnswer {
        question_id: String,
        detail: &'static str,
    },
    /// The answer's `type` tag is not a documented answer type (per-answer skip).
    UnknownAnswerType { question_id: String },
    /// The response carried no model identifier, so drift cannot be recorded.
    MissingResponseModel,
}

impl AnswerIssue {
    /// Stable machine-readable reason code for logs and records.
    pub fn reason(&self) -> &'static str {
        match self {
            AnswerIssue::UnknownId { .. } => "unknown_answer_id",
            AnswerIssue::MissingId { .. } => "missing_answer_id",
            AnswerIssue::TypeMismatch { .. } => "answer_type_mismatch",
            AnswerIssue::UnknownChoice { .. } => "choice_not_in_criteria",
            AnswerIssue::DistributionKeysMismatch { .. } => "distribution_keys_mismatch",
            AnswerIssue::ProbabilityOutOfRange { .. } => "probability_out_of_range",
            AnswerIssue::ProbabilitiesNotSummingToOne { .. } => "probabilities_not_summing_to_one",
            AnswerIssue::ConfidenceOutOfRange { .. } => "confidence_out_of_range",
            AnswerIssue::LegendMismatch { .. } => "score_legend_mismatch",
            AnswerIssue::ScoreNotFinite { .. } => "score_not_finite",
            AnswerIssue::ScoreOutOfRange { .. } => "score_out_of_range",
            AnswerIssue::NoulOutOfRange { .. } => "noul_out_of_range",
            AnswerIssue::ChoiceNotPeak { .. } => "choice_not_peak",
            AnswerIssue::ScoreNotExpectation { .. } => "score_not_expectation",
            AnswerIssue::MalformedAnswer { .. } => "malformed_answer",
            AnswerIssue::UnknownAnswerType { .. } => "unknown_answer_type",
            AnswerIssue::MissingResponseModel => "missing_response_model",
        }
    }

    /// Question id when the issue is per-answer.
    pub fn question_id(&self) -> Option<&str> {
        match self {
            AnswerIssue::MissingResponseModel => None,
            AnswerIssue::UnknownId { question_id }
            | AnswerIssue::MissingId { question_id }
            | AnswerIssue::TypeMismatch { question_id, .. }
            | AnswerIssue::UnknownChoice { question_id, .. }
            | AnswerIssue::DistributionKeysMismatch { question_id, .. }
            | AnswerIssue::ProbabilityOutOfRange { question_id, .. }
            | AnswerIssue::ProbabilitiesNotSummingToOne { question_id, .. }
            | AnswerIssue::ConfidenceOutOfRange { question_id, .. }
            | AnswerIssue::LegendMismatch { question_id, .. }
            | AnswerIssue::ScoreNotFinite { question_id, .. }
            | AnswerIssue::ScoreOutOfRange { question_id, .. }
            | AnswerIssue::NoulOutOfRange { question_id, .. }
            | AnswerIssue::ChoiceNotPeak { question_id, .. }
            | AnswerIssue::ScoreNotExpectation { question_id, .. }
            | AnswerIssue::MalformedAnswer { question_id, .. }
            | AnswerIssue::UnknownAnswerType { question_id } => Some(question_id),
        }
    }
}

impl fmt::Display for AnswerIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Ids are caller-chosen question ids, not user content; values are bounded numbers.
        match self {
            AnswerIssue::UnknownId { question_id } => write!(f, "unknown answer id `{question_id}`"),
            AnswerIssue::MissingId { question_id } => write!(f, "missing answer for `{question_id}`"),
            AnswerIssue::TypeMismatch {
                question_id,
                expected,
                actual,
            } => write!(
                f,
                "answer type mismatch for `{question_id}`: expected {expected}, got {actual}"
            ),
            AnswerIssue::UnknownChoice { question_id, choice } => {
                write!(f, "choice `{choice}` for `{question_id}` is not in the criteria")
            }
            AnswerIssue::DistributionKeysMismatch {
                question_id,
                missing,
                extra,
            } => write!(
                f,
                "distribution keys for `{question_id}` mismatch (missing {missing:?}, extra {extra:?})"
            ),
            AnswerIssue::ProbabilityOutOfRange {
                question_id,
                key,
                value,
            } => write!(f, "probability {value} for `{question_id}/{key}` is out of range"),
            AnswerIssue::ProbabilitiesNotSummingToOne { question_id, sum } => {
                write!(f, "probabilities for `{question_id}` sum to {sum}, not 1")
            }
            AnswerIssue::ConfidenceOutOfRange { question_id, value } => {
                write!(f, "confidence {value} for `{question_id}` is out of range")
            }
            AnswerIssue::LegendMismatch {
                question_id,
                missing,
                extra,
            } => write!(
                f,
                "score legend for `{question_id}` mismatch (missing {missing:?}, extra {extra:?})"
            ),
            AnswerIssue::ScoreNotFinite { question_id, value } => {
                write!(f, "score {value} for `{question_id}` is not finite")
            }
            AnswerIssue::ScoreOutOfRange {
                question_id,
                value,
                max,
            } => write!(f, "score {value} for `{question_id}` is outside 0..={max}"),
            AnswerIssue::NoulOutOfRange { question_id, value } => {
                write!(f, "noul {value} for `{question_id}` is out of range")
            }
            AnswerIssue::ChoiceNotPeak {
                question_id,
                choice,
                peak,
            } => write!(
                f,
                "choice `{choice}` for `{question_id}` is not the distribution peak (`{peak}`)"
            ),
            AnswerIssue::ScoreNotExpectation {
                question_id,
                value,
                expected,
            } => write!(
                f,
                "score {value} for `{question_id}` is not the weighted expected level {expected}"
            ),
            AnswerIssue::MalformedAnswer {
                question_id,
                detail,
            } => write!(
                f,
                "answer `{question_id}` failed to parse ({detail}); the rest of the body is unaffected"
            ),
            AnswerIssue::UnknownAnswerType { question_id } => {
                write!(f, "answer `{question_id}` carries an unknown answer type")
            }
            AnswerIssue::MissingResponseModel => write!(f, "response did not carry a model id"),
        }
    }
}

/// Result of validating a full response against its request.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseValidation {
    /// Answers that passed every check, keyed by question id.
    pub accepted: BTreeMap<String, Answer>,
    /// One entry per rejected or missing answer; each entry is a logged skip.
    pub skipped: Vec<(String, AnswerIssue)>,
    /// Response `model` field, when present and non-empty.
    pub response_model: Option<String>,
}

impl ResponseValidation {
    /// Number of answers that passed every check.
    pub fn accepted_count(&self) -> usize {
        self.accepted.len()
    }

    /// Number of answers rejected or missing.
    pub fn skipped_count(&self) -> usize {
        self.skipped.len()
    }

    /// True when at least one answer is usable.
    pub fn has_accepted(&self) -> bool {
        !self.accepted.is_empty()
    }

    /// Skip reasons in stable order (question id, reason code).
    pub fn skip_reasons(&self) -> Vec<(String, &'static str)> {
        self.skipped
            .iter()
            .map(|(id, issue)| (id.clone(), issue.reason()))
            .collect()
    }

    /// One bounded, secret-free log line per skip.
    pub fn log_lines(&self) -> Vec<String> {
        self.skipped
            .iter()
            .map(|(id, issue)| format!("jev skip [{id}] {}: {issue}", issue.reason()))
            .collect()
    }
}

fn is_finite_in_unit(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn key_sets(keys: impl IntoIterator<Item = String>) -> BTreeSet<String> {
    keys.into_iter().collect()
}

/// Validates every answer in `response` against `request`.
///
/// The accepted set contains ONLY answers that passed every documented check. Unknown,
/// extra, missing, mistyped and malformed answers become `AnswerIssue` skips.
pub fn validate_response(request: &SystemOneRequest, response: &SystemOneResponse) -> ResponseValidation {
    let mut accepted = BTreeMap::new();
    let mut skipped: Vec<(String, AnswerIssue)> = Vec::new();
    let response_model = Some(response.model.clone()).filter(|model| !model.trim().is_empty());

    for (question_id, answer) in &response.answers {
        let Some(question) = request.questions.get(question_id) else {
            skipped.push((
                question_id.clone(),
                AnswerIssue::UnknownId {
                    question_id: question_id.clone(),
                },
            ));
            continue;
        };
        match validate_answer(question_id, question, answer) {
            Ok(()) => {
                accepted.insert(question_id.clone(), answer.clone());
            }
            Err(issue) => skipped.push((question_id.clone(), issue)),
        }
    }

    // Per-answer parse failures recorded by the production parser become skips here, so
    // a single malformed or unknown-typed answer never invalidates the whole body.
    for (question_id, issue) in &response.answer_parse_skips {
        skipped.push((question_id.clone(), issue.clone()));
    }
    let parse_skipped: BTreeSet<String> = response
        .answer_parse_skips
        .iter()
        .map(|(question_id, _)| question_id.clone())
        .collect();
    for question_id in request.questions.keys() {
        if !response.answers.contains_key(question_id) && !parse_skipped.contains(question_id) {
            skipped.push((
                question_id.clone(),
                AnswerIssue::MissingId {
                    question_id: question_id.clone(),
                },
            ));
        }
    }

    if response_model.is_none() {
        skipped.push((String::new(), AnswerIssue::MissingResponseModel));
    }

    skipped.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.reason().cmp(right.1.reason())));
    ResponseValidation {
        accepted,
        skipped,
        response_model,
    }
}

/// Validates one answer against its question. Returns the specific issue on failure.
pub fn validate_answer(
    question_id: &str,
    question: &QuestionSpec,
    answer: &Answer,
) -> Result<(), AnswerIssue> {
    let expected = question.question_type();
    let actual = answer.answer_type();
    if expected != actual {
        return Err(AnswerIssue::TypeMismatch {
            question_id: question_id.to_string(),
            expected,
            actual,
        });
    }

    match (question, answer) {
        (QuestionSpec::Noul { .. }, Answer::Noul { noul }) => {
            if !is_finite_in_unit(*noul) {
                return Err(AnswerIssue::NoulOutOfRange {
                    question_id: question_id.to_string(),
                    value: *noul,
                });
            }
            Ok(())
        }
        (QuestionSpec::Choice { criteria, .. }, Answer::Choice { choice, probabilities, confidence }) => {
            if !criteria.contains_key(choice) {
                return Err(AnswerIssue::UnknownChoice {
                    question_id: question_id.to_string(),
                    choice: choice.clone(),
                });
            }
            validate_distribution(question_id, criteria.keys().cloned(), probabilities)?;
            if !is_finite_in_unit(*confidence) {
                return Err(AnswerIssue::ConfidenceOutOfRange {
                    question_id: question_id.to_string(),
                    value: *confidence,
                });
            }
            // Documented semantic constraint: the reported choice must be the argmax of the
            // reported distribution. Tolerance follows the docs' 2-decimal example rounding
            // (0.01 slack) when every probability is exactly 2-decimal-quantized, and a
            // tight bound for full-precision distributions. Ties stay valid.
            let tolerance = if is_two_decimal_quantized(probabilities.values().copied()) {
                QUANTIZED_ARGMAX_TOLERANCE
            } else {
                ARGMAX_TOLERANCE
            };
            // `validate_distribution` above guarantees a finite, summing-to-one,
            // key-matching distribution, so the peak is a real value here.
            let peak_value = probabilities
                .values()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let peak = probabilities
                .iter()
                .find(|(_, probability)| **probability == peak_value)
                .map(|(key, _)| key.clone())
                .unwrap_or_default();
            let chosen = probabilities.get(choice).copied().unwrap_or(0.0);
            if chosen < peak_value - tolerance - FLOAT_NOISE_SLACK {
                return Err(AnswerIssue::ChoiceNotPeak {
                    question_id: question_id.to_string(),
                    choice: choice.clone(),
                    peak,
                });
            }
            Ok(())
        }
        (
            QuestionSpec::Score { criteria, .. },
            Answer::Score {
                score,
                legend,
                probabilities,
                confidence,
            },
        ) => {
            let expected_keys = key_sets((0..criteria.len()).map(|index| index.to_string()));
            let legend_keys = key_sets(legend.keys().cloned());
            if legend_keys != expected_keys {
                let missing: Vec<String> = expected_keys.difference(&legend_keys).cloned().collect();
                let extra: Vec<String> = legend_keys.difference(&expected_keys).cloned().collect();
                return Err(AnswerIssue::LegendMismatch {
                    question_id: question_id.to_string(),
                    missing,
                    extra,
                });
            }
            for key in &expected_keys {
                let Some(description) = legend.get(key) else {
                    continue;
                };
                let index: usize = key.parse().unwrap_or(usize::MAX);
                // Semantic equality: a Text echo equals a Json string echo, and a `null`
                // echo equals a `Null` criteria entry.
                let matches = match criteria.get(index) {
                    Some(expected_entry) => expected_entry.equivalent(description),
                    None => false,
                };
                if !matches {
                    return Err(AnswerIssue::LegendMismatch {
                        question_id: question_id.to_string(),
                        missing: Vec::new(),
                        extra: vec![key.clone()],
                    });
                }
            }
            validate_distribution(question_id, expected_keys.iter().cloned(), probabilities)?;
            if !score.is_finite() {
                return Err(AnswerIssue::ScoreNotFinite {
                    question_id: question_id.to_string(),
                    value: *score,
                });
            }
            let max = (criteria.len() as f64) - 1.0;
            if *score < 0.0 || *score > max {
                return Err(AnswerIssue::ScoreOutOfRange {
                    question_id: question_id.to_string(),
                    value: *score,
                    max,
                });
            }
            if !is_finite_in_unit(*confidence) {
                return Err(AnswerIssue::ConfidenceOutOfRange {
                    question_id: question_id.to_string(),
                    value: *confidence,
                });
            }
            // Documented semantic constraint: the score is the probability-weighted expected
            // level, `sum(index * p_index)` over the 0-based positions. The tolerance follows
            // the docs' 2-decimal example rounding for quantized distributions and the strict
            // probability tolerance for full-precision ones.
            let expected: f64 = probabilities
                .iter()
                .map(|(key, probability)| {
                    key.parse::<usize>().map_or(0.0, |index| index as f64) * probability
                })
                .sum();
            let tolerance = if is_two_decimal_quantized(probabilities.values().copied()) {
                quantized_expectation_tolerance(criteria.len())
            } else {
                PROBABILITY_TOLERANCE
            };
            if (score - expected).abs() > tolerance + FLOAT_NOISE_SLACK {
                return Err(AnswerIssue::ScoreNotExpectation {
                    question_id: question_id.to_string(),
                    value: *score,
                    expected,
                });
            }
            Ok(())
        }
        // Type equality was checked above; this arm is unreachable but keeps the match total.
        _ => Err(AnswerIssue::TypeMismatch {
            question_id: question_id.to_string(),
            expected,
            actual,
        }),
    }
}

fn validate_distribution(
    question_id: &str,
    expected_keys: impl Iterator<Item = String>,
    probabilities: &BTreeMap<String, f64>,
) -> Result<(), AnswerIssue> {
    let expected = key_sets(expected_keys);
    let actual = key_sets(probabilities.keys().cloned());
    if expected != actual {
        let missing: Vec<String> = expected.difference(&actual).cloned().collect();
        let extra: Vec<String> = actual.difference(&expected).cloned().collect();
        return Err(AnswerIssue::DistributionKeysMismatch {
            question_id: question_id.to_string(),
            missing,
            extra,
        });
    }
    let mut sum = 0.0f64;
    for (key, value) in probabilities {
        if !value.is_finite() || !(0.0..=1.0).contains(value) {
            return Err(AnswerIssue::ProbabilityOutOfRange {
                question_id: question_id.to_string(),
                key: key.clone(),
                value: *value,
            });
        }
        sum += value;
    }
    if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(AnswerIssue::ProbabilitiesNotSummingToOne {
            question_id: question_id.to_string(),
            sum,
        });
    }
    Ok(())
}

/// Validates a whole request before it can reach the network.
pub fn validate_request_shape(request: &SystemOneRequest) -> Result<(), JevError> {
    if request.model.trim().is_empty() {
        return Err(JevError::validation("request model is empty"));
    }
    if request.questions.is_empty() {
        return Err(JevError::validation("request has no questions"));
    }
    if request.questions.len() > MAX_QUESTIONS_PER_REQUEST {
        return Err(JevError::validation(format!(
            "request has {} questions, limit is {MAX_QUESTIONS_PER_REQUEST}",
            request.questions.len()
        )));
    }
    for (id, question) in &request.questions {
        if id.trim().is_empty() {
            return Err(JevError::validation("request has an empty question id"));
        }
        question.validate_shape()?;
    }
    validate_state_shape(&request.state)?;
    Ok(())
}

/// Validates the `state` shape: a plain string, a JSON object, or a JSON array
/// (api.md: "A plain string for text, or structured data (object/array)"; the
/// SDK `JSONContent` schema is string | object | array). Bare scalars (`null`,
/// number, boolean) are documented nowhere and are refused before transport.
pub fn validate_state_shape(state: &Value) -> Result<(), JevError> {
    match state {
        Value::String(_) | Value::Object(_) | Value::Array(_) => Ok(()),
        _ => Err(JevError::validation(
            "state must be a string, object, or array (api.md Request body)",
        )),
    }
}

/// Convenience check used by the client's pre-transport gate.
pub fn state_shape_is_valid(state: &Value) -> bool {
    validate_state_shape(state).is_ok()
}

/// Estimates the prompt-token size of one request with the crate's documented heuristic
/// (`crate::compaction::estimate_tokens`, "Heuristic only; byte limits are enforced
/// separately on the serialized request").
///
/// The result is an ESTIMATE, never an exact count: it is used only for the host-policy
/// `request_token_limit` pre-transport skip, never to claim a server-side guarantee.
pub fn estimate_request_tokens(request: &SystemOneRequest) -> usize {
    let mut text = String::new();
    match &request.state {
        Value::String(state) => text.push_str(state),
        value @ (Value::Object(_) | Value::Array(_)) => text.push_str(&canonical_entry_json(value)),
        _ => {}
    }
    text.push_str(&request.model);
    for (id, question) in &request.questions {
        text.push_str(id);
        text.push_str(&question.instructions());
        match question {
            QuestionSpec::Noul { criteria: Some(criteria), .. } => {
                text.push_str(&criteria.r#true.instructions_hash_input());
                text.push_str(&criteria.r#false.instructions_hash_input());
            }
            QuestionSpec::Choice { criteria, .. } => {
                for (option, entry) in criteria {
                    text.push_str(option);
                    text.push_str(&entry.instructions_hash_input());
                }
            }
            QuestionSpec::Score { criteria, .. } => {
                for entry in criteria {
                    text.push_str(&entry.instructions_hash_input());
                }
            }
            _ => {}
        }
    }
    crate::compaction::estimate_tokens(&text)
}

impl EntryValue {
    /// Hash input for one entry: the bare string for `Text`, canonical JSON for `Json`
    /// and `Null` forms. Mirrors `QuestionSpec::instructions` semantics at entry level.
    pub fn instructions_hash_input(&self) -> String {
        match self {
            EntryValue::Text(text) => text.clone(),
            EntryValue::Null => "null".to_string(),
            EntryValue::Json(value) => canonical_entry_json(value),
        }
    }
}

/// Model configuration drift: what the caller asked for versus what answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDrift {
    pub requested: String,
    pub response: Option<String>,
    pub drifted: bool,
}

/// Compares the requested model alias with the versioned id that answered.
pub fn model_drift(requested: &str, response_model: Option<&str>) -> ModelDrift {
    let response = response_model.map(|model| model.to_string());
    let drifted = match response.as_deref() {
        Some(model) => model != requested,
        None => true,
    };
    ModelDrift {
        requested: requested.to_string(),
        response,
        drifted,
    }
}

// ---------------------------------------------------------------------------
// Decision domain (DESIGN.md section 3)
// ---------------------------------------------------------------------------

/// Authoritative mode control lives in config.rs (DESIGN.md 3.1) and is re-exported here so
/// callers can use one import path for the decision-domain types.
pub use crate::config::JevMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Coding,
    Research,
    Debugging,
    Planning,
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Complexity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRequirement {
    None,
    Read,
    Search,
    Shell,
    Python,
    Delegate,
    Multiple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContinueDecision {
    Continue,
    Stop,
    Escalate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sufficient {
    Sufficient,
    Insufficient,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationRecommendation {
    None,
    ReRun,
    Escalate,
    Verify,
}

/// Comparison categories. All are recommend/record only in Compare.
/// Retry classification and trace assessment require explicit observation flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionCategory {
    TaskClassification,
    Complexity,
    ToolRequirement,
    ToolCandidates,
    SubagentRequirement,
    SubagentModelRouting,
    ContextRelevance,
    CodeSearchRelevance,
    CodeSearchRerank,
    CodeLineFind,
    /// ROOT-CONTRACT v6 (Evidence lane): untrusted-retrieval safety battery
    /// (possible prompt injection / premise contradiction / evidence
    /// usefulness). Noul-only typed acceptance; advisory labels only — never
    /// a drop, never a verification, never an authority claim.
    CodeRetrievalSafety,
    /// ROOT-CONTRACT v6 (Evidence lane): citation relation over the ACTUAL
    /// supplied source span (supports / contradicts / unclear). Choice
    /// answer; advisory labels only — never verification, never a gate.
    CodeCitationCheck,
    /// Agent-guidance lane: advisory skill suggestion over bounded task
    /// metadata; recommendation only — never installs, executes or spawns.
    SkillSuggestion,
    /// Agent-guidance lane: advisory guardrail battery over bounded input
    /// excerpts; annotation only — never blocks, never modifies behavior.
    GuardrailsInput,
    /// Agent-guidance lane: advisory guardrail battery over bounded output
    /// excerpts (AgentEnd); annotation only — never blocks, never modifies.
    GuardrailsOutput,
    MemoryRelevance,
    ContinueStopEscalate,
    ResultSufficiency,
    FirstPassVerification,
    RetryClassification,
    TraceAssessment,
    /// Explicit agent-authored questions; never an automatic control recommendation.
    Dynamic,
}

impl DecisionCategory {
    /// Stable snake_case id used as the question-id prefix and in records.
    pub fn as_str(self) -> &'static str {
        match self {
            DecisionCategory::TaskClassification => "task_classification",
            DecisionCategory::Complexity => "complexity",
            DecisionCategory::ToolRequirement => "tool_requirement",
            DecisionCategory::ToolCandidates => "tool_candidates",
            DecisionCategory::SubagentRequirement => "subagent_requirement",
            DecisionCategory::SubagentModelRouting => "subagent_model_routing",
            DecisionCategory::ContextRelevance => "context_relevance",
            DecisionCategory::CodeSearchRelevance => "code_search_relevance",
            DecisionCategory::CodeSearchRerank => "code_search_rerank",
            DecisionCategory::CodeLineFind => "code_line_find",
            DecisionCategory::CodeRetrievalSafety => "code_retrieval_safety",
            DecisionCategory::CodeCitationCheck => "code_citation_check",
            DecisionCategory::SkillSuggestion => "skill_suggestion",
            DecisionCategory::GuardrailsInput => "guardrails_input",
            DecisionCategory::GuardrailsOutput => "guardrails_output",
            DecisionCategory::MemoryRelevance => "memory_relevance",
            DecisionCategory::ContinueStopEscalate => "continue_stop_escalate",
            DecisionCategory::ResultSufficiency => "result_sufficiency",
            DecisionCategory::FirstPassVerification => "first_pass_verification",
            DecisionCategory::RetryClassification => "retry_classification",
            DecisionCategory::TraceAssessment => "trace_assessment",
            DecisionCategory::Dynamic => "dynamic",
        }
    }

    /// All categories in canonical order.
    pub fn all() -> [DecisionCategory; 22] {
        [
            DecisionCategory::TaskClassification,
            DecisionCategory::Complexity,
            DecisionCategory::ToolRequirement,
            DecisionCategory::ToolCandidates,
            DecisionCategory::SubagentRequirement,
            DecisionCategory::SubagentModelRouting,
            DecisionCategory::ContextRelevance,
            DecisionCategory::CodeSearchRelevance,
            DecisionCategory::CodeSearchRerank,
            DecisionCategory::CodeLineFind,
            DecisionCategory::CodeRetrievalSafety,
            DecisionCategory::CodeCitationCheck,
            DecisionCategory::SkillSuggestion,
            DecisionCategory::GuardrailsInput,
            DecisionCategory::GuardrailsOutput,
            DecisionCategory::MemoryRelevance,
            DecisionCategory::ContinueStopEscalate,
            DecisionCategory::ResultSufficiency,
            DecisionCategory::FirstPassVerification,
            DecisionCategory::RetryClassification,
            DecisionCategory::TraceAssessment,
            DecisionCategory::Dynamic,
        ]
    }

    /// Parses a category id.
    pub fn parse(raw: &str) -> Option<Self> {
        DecisionCategory::all()
            .into_iter()
            .find(|category| category.as_str() == raw)
    }

    /// Builds the question id for sub-question `n` of this category.
    pub fn question_id(self, index: usize) -> String {
        format!("{}.{}", self.as_str(), index)
    }
}

impl fmt::Display for DecisionCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Automatic category switches; optional observers also require their feature flag.
/// Explicit Dynamic questions never enter the comparison lane.
pub fn compare_default_categories() -> BTreeMap<DecisionCategory, bool> {
    DecisionCategory::all()
        .into_iter()
        .map(|category| (category, category != DecisionCategory::Dynamic))
        .collect()
}

/// One immutable bundle of sibling questions against one state snapshot.
///
/// Different lifecycle stages use separate bundles; no question in a bundle depends on
/// another answer in the same bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionBundle {
    /// Opaque local session id (never a transcript).
    pub session_id: String,
    pub turn: u64,
    /// Lifecycle stage id, e.g. `turn_start`, `tool_call`, `agent_end`, `model_select`.
    pub stage: String,
    /// Bounded state snapshot as untrusted data.
    pub state: Value,
    pub model: String,
    pub questions: BTreeMap<String, QuestionSpec>,
    /// Category for each question id (categories map 1:N to sub-questions).
    pub question_categories: BTreeMap<String, DecisionCategory>,
}

impl DecisionBundle {
    /// Builds the wire request for this bundle.
    pub fn to_request(&self) -> SystemOneRequest {
        SystemOneRequest {
            state: self.state.clone(),
            model: self.model.clone(),
            questions: self.questions.clone(),
        }
    }

    /// Question ids whose category is enabled, in stable order.
    pub fn enabled_question_ids(&self, enabled: &BTreeMap<DecisionCategory, bool>) -> Vec<String> {
        self.questions
            .keys()
            .filter(|id| {
                self.question_categories
                    .get(*id)
                    .and_then(|category| enabled.get(category))
                    .copied()
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Permanent no-subagent-control boundary (DESIGN.md sections 11 and 12; binding)
// ---------------------------------------------------------------------------

/// Capabilities Jev must NEVER hold, in any mode, including any future Active mode.
///
/// This list is documentation plus test data: `refuse_subagent_control` returns
/// `JevError::SubagentControlForbidden` for every entry, so no caller can reach such an action
/// through this crate and no configuration flag can enable one.
pub const FORBIDDEN_SUBAGENT_CAPABILITIES: [&str; 12] = [
    "spawn_subagent",
    "delete_subagent",
    "cancel_subagent",
    "pause_subagent",
    "resume_subagent",
    "select_child_model_or_effort",
    "assign_agent_work_or_role",
    "send_agent_message",
    "steer_suppress_or_reorder_agent_messages",
    "change_depth_concurrency_or_budget",
    "decide_child_completion",
    "switch_primary_model_or_effort",
];

/// A request that would let Jev influence a child agent. Every variant is refused.
///
/// The type exists so a boundary violation is a compile-visible, test-visible call instead of an
/// implicit capability. Nothing in this crate can construct, accept or execute one successfully.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubagentControlRequest {
    /// Create a new child agent.
    Spawn { role: String },
    /// Remove a child agent.
    Delete { child: String },
    /// Cancel a child's work.
    Cancel { child: String },
    /// Model/provider/effort selection for a child or for the primary model.
    SelectModel { child: String, model: String, effort: String },
    /// Work or role assignment.
    AssignWork { child: String, summary: String },
    /// Any outbound agent message, including a steering or suppression request.
    SendMessage { target: String, body: String },
    /// Depth, concurrency or budget change.
    ChangeBudget { max_depth: u32, max_concurrency: u32 },
    /// A decision that a child has finished.
    DecideCompletion { child: String, done: bool },
}

impl SubagentControlRequest {
    /// The stable capability name from `FORBIDDEN_SUBAGENT_CAPABILITIES`.
    pub fn capability(&self) -> &'static str {
        match self {
            SubagentControlRequest::Spawn { .. } => "spawn_subagent",
            SubagentControlRequest::Delete { .. } => "delete_subagent",
            SubagentControlRequest::Cancel { .. } => "cancel_subagent",
            SubagentControlRequest::SelectModel { .. } => "select_child_model_or_effort",
            SubagentControlRequest::AssignWork { .. } => "assign_agent_work_or_role",
            SubagentControlRequest::SendMessage { .. } => "send_agent_message",
            SubagentControlRequest::ChangeBudget { .. } => "change_depth_concurrency_or_budget",
            SubagentControlRequest::DecideCompletion { .. } => "decide_child_completion",
        }
    }
}

/// Always refuses. There is no argument, mode, confidence or flag that makes this succeed.
pub fn refuse_subagent_control(request: &SubagentControlRequest) -> Result<(), JevError> {
    Err(JevError::SubagentControlForbidden {
        capability: request.capability(),
    })
}

/// Read-only observation of a child, as categories 5/6 may record it.
///
/// The struct has no method that mutates anything, holds no handle to a child, and carries
/// `applied = false` as a field with a getter that can only return false.
#[derive(Debug, Clone, PartialEq)]
pub struct SubagentObservation {
    /// Opaque local child id observed at the boundary.
    pub child_session_id: String,
    /// Advisory suitability text within a user-approved allowlist. Never a command.
    pub advisory: String,
    /// Category that produced this observation (5 or 6).
    pub category: DecisionCategory,
    /// ALWAYS false: no Jev output is applied.
    pub applied: bool,
}

impl SubagentObservation {
    /// Builds an observation. `applied` is forced to false.
    pub fn new(
        child_session_id: impl Into<String>,
        advisory: impl Into<String>,
        category: DecisionCategory,
    ) -> Self {
        Self {
            child_session_id: child_session_id.into(),
            advisory: advisory.into(),
            category,
            applied: false,
        }
    }

    /// True only for advisory categories whose output cannot act (5 and 6).
    pub fn is_advisory(&self) -> bool {
        matches!(
            self.category,
            DecisionCategory::SubagentRequirement | DecisionCategory::SubagentModelRouting
        )
    }

    /// Always false. A caller cannot convert an observation into an applied action.
    pub fn actionable(&self) -> bool {
        false
    }

    /// Always false, mirroring `applied` without exposing a setter.
    pub fn applied(&self) -> bool {
        false
    }
}

/// One decision produced by a SystemOne implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRecord {
    pub question_id: String,
    pub category: DecisionCategory,
    pub answer: Answer,
    /// Response model id (drift is visible here).
    pub response_model: Option<String>,
    /// Requested model alias.
    pub requested_model: String,
    /// ALWAYS false in Compare: nothing from Jev is applied to live behavior.
    pub applied: bool,
}

impl DecisionRecord {
    /// Model drift detail for this record.
    pub fn drift(&self) -> ModelDrift {
        model_drift(&self.requested_model, self.response_model.as_deref())
    }
}

/// Result of one SystemOne call: accepted records plus skips. Never a fabricated answer.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOutcome {
    pub records: Vec<DecisionRecord>,
    /// (question id, reason code) for each skipped category answer.
    pub skips: Vec<(String, &'static str)>,
    pub response_model: Option<String>,
    pub usage: Usage,
    /// ALWAYS false in this build.
    pub applied: bool,
    /// Transport attempts the client actually made for this outcome (1 +
    /// retries; 0 when no transport call happened, e.g. pre-transport
    /// validation skips). The client owns retries, so this is the truthful
    /// attempt number for correlation records.
    pub attempts: u32,
    /// Bounded, sanitized server-provided request id from the response
    /// (`x-typesafe-request-id`), when the transport captured one. Untrusted
    /// server data: it is bounded, control-char free and credential-echo safe.
    pub server_request_id: Option<String>,
}

impl DecisionOutcome {
    /// An outcome that ran no request (Off, missing credential, refusal).
    ///
    /// The skip carries no question id: no question was evaluated, and inventing a category
    /// result would be fabrication.
    pub fn skipped_all(reason: &'static str) -> Self {
        Self::skipped_questions(reason, std::iter::empty::<String>())
    }

    /// An outcome where each of `question_ids` was skipped for the same reason.
    pub fn skipped_questions(reason: &'static str, question_ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            records: Vec::new(),
            skips: question_ids.into_iter().map(|id| (id, reason)).collect(),
            response_model: None,
            usage: Usage::default(),
            applied: false,
            attempts: 0,
            server_request_id: None,
        }
    }

    /// True when no answer could be used.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SystemOne + Transport traits
// ---------------------------------------------------------------------------

/// Boxed future used by the object-safe transport trait.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// HTTP transport. Production uses `JevHttpTransport`; tests use `MockJevTransport`
/// through the exact same code path (no special-casing in the client).
pub trait Transport: Send + Sync {
    /// Sends one request and returns one validated-parsed response.
    fn post(&self, request: &SystemOneRequest, timeout: std::time::Duration) -> BoxFuture<Result<SystemOneResponse, JevError>>;

    /// Explicit read-only model-catalog fetch (`GET /v1/models`, ROOT-CONTRACT v9). Returns
    /// the RAW response body; parsing and validation happen in `crate::models`.
    ///
    /// Default: unsupported. A transport that cannot fetch the catalog reports `Internal`
    /// instead of fabricating an empty catalog, so unknowns are never invented.
    fn get_models(&self, _timeout: std::time::Duration) -> BoxFuture<Result<Vec<u8>, JevError>> {
        Box::pin(async {
            Err(JevError::Internal {
                detail: "model catalog fetch is not supported by this transport".to_string(),
            })
        })
    }
}

/// System One abstraction: `DisabledSystemOne` (Off) and `JevSystemOne` (Compare).
///
/// This trait is object-safe on purpose; callers hold `Arc<dyn SystemOne>`.
pub trait SystemOne: Send + Sync {
    /// Current effective mode.
    fn mode(&self) -> JevMode;

    /// Evaluates a bundle. Returns skips, never fabricated answers, on any failure.
    fn decide(&self, bundle: DecisionBundle) -> BoxFuture<DecisionOutcome>;
}
