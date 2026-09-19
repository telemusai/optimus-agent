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

/// Maximum accepted sub-question count per request (speculative fan-out stays bounded).
pub const MAX_QUESTIONS_PER_REQUEST: usize = 64;

// ---------------------------------------------------------------------------
// Wire request
// ---------------------------------------------------------------------------

/// `{"true": .., "false": ..}` descriptions for a Noul question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true")]
    pub r#true: String,
    #[serde(rename = "false")]
    pub r#false: String,
}

/// One typed question. Serde writes `{"type": ..., "instructions": ..., "criteria": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QuestionSpec {
    Noul {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        instructions: String,
        criteria: BTreeMap<String, Option<String>>,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
}

impl QuestionSpec {
    /// Wire type tag of this question.
    pub fn question_type(&self) -> QuestionType {
        match self {
            QuestionSpec::Noul { .. } => QuestionType::Noul,
            QuestionSpec::Choice { .. } => QuestionType::Choice,
            QuestionSpec::Score { .. } => QuestionType::Score,
        }
    }

    /// Instructions text (used for prompt-version hashing by lane B).
    pub fn instructions(&self) -> &str {
        match self {
            QuestionSpec::Noul { instructions, .. }
            | QuestionSpec::Choice { instructions, .. }
            | QuestionSpec::Score { instructions, .. } => instructions,
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
    pub fn validate_shape(&self) -> Result<(), JevError> {
        match self {
            QuestionSpec::Noul { instructions, .. } => {
                if instructions.trim().is_empty() {
                    return Err(JevError::validation("noul question has empty instructions"));
                }
                Ok(())
            }
            QuestionSpec::Choice { instructions, criteria } => {
                if instructions.trim().is_empty() {
                    return Err(JevError::validation("choice question has empty instructions"));
                }
                if criteria.is_empty() {
                    return Err(JevError::validation("choice question has no criteria options"));
                }
                if criteria.keys().any(|key| key.is_empty()) {
                    return Err(JevError::validation("choice question has an empty option key"));
                }
                Ok(())
            }
            QuestionSpec::Score { instructions, criteria } => {
                if instructions.trim().is_empty() {
                    return Err(JevError::validation("score question has empty instructions"));
                }
                // The API requires at least two ordered levels.
                if criteria.len() < 2 {
                    return Err(JevError::validation("score question needs at least two levels"));
                }
                if criteria.iter().any(|level| level.trim().is_empty()) {
                    return Err(JevError::validation("score question has an empty level description"));
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
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
        legend: BTreeMap<String, String>,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub answers: BTreeMap<String, Answer>,
    #[serde(default)]
    pub usage: Usage,
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
            | AnswerIssue::NoulOutOfRange { question_id, .. } => Some(question_id),
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

    for question_id in request.questions.keys() {
        if !response.answers.contains_key(question_id) {
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
                let expected_description = criteria.get(index);
                if expected_description != Some(description) {
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
    Ok(())
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
    MemoryRelevance,
    ContinueStopEscalate,
    ResultSufficiency,
    FirstPassVerification,
    RetryClassification,
    TraceAssessment,
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
            DecisionCategory::MemoryRelevance => "memory_relevance",
            DecisionCategory::ContinueStopEscalate => "continue_stop_escalate",
            DecisionCategory::ResultSufficiency => "result_sufficiency",
            DecisionCategory::FirstPassVerification => "first_pass_verification",
            DecisionCategory::RetryClassification => "retry_classification",
            DecisionCategory::TraceAssessment => "trace_assessment",
        }
    }

    /// All categories in canonical order.
    pub fn all() -> [DecisionCategory; 13] {
        [
            DecisionCategory::TaskClassification,
            DecisionCategory::Complexity,
            DecisionCategory::ToolRequirement,
            DecisionCategory::ToolCandidates,
            DecisionCategory::SubagentRequirement,
            DecisionCategory::SubagentModelRouting,
            DecisionCategory::ContextRelevance,
            DecisionCategory::MemoryRelevance,
            DecisionCategory::ContinueStopEscalate,
            DecisionCategory::ResultSufficiency,
            DecisionCategory::FirstPassVerification,
            DecisionCategory::RetryClassification,
            DecisionCategory::TraceAssessment,
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

/// Category switches enabled; optional observers also require their feature flag.
pub fn compare_default_categories() -> BTreeMap<DecisionCategory, bool> {
    DecisionCategory::all()
        .into_iter()
        .map(|category| (category, true))
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
