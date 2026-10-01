//! Versioned JSONL correlation records (DESIGN.md section 7).
//!
//! Every Compare record carries `applied=false` ALWAYS, classifies agreement
//! against the boundary-captured baseline, marks savings hypothetical, holds
//! no secrets, raw prompts, transcripts or tool outputs, and cleans up with
//! size/time retention strictly confined to Jev's own files.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};

use crate::scheduler::JobResult;
use crate::types::DecisionCategory;

/// Record schema version for Compare (shadow) records.
pub const RECORD_SCHEMA_VERSION: &str = "jev.compare/1";

/// Record schema version for Active records, where the answer may have been
/// applied. Separate from the Compare id so a reader can never mistake an
/// applied row for a shadow row.
pub const ACTIVE_RECORD_SCHEMA_VERSION: &str = "jev.active/1";

/// Maximum text length of any single field value kept in a record.
pub const MAX_FIELD_TEXT: usize = 120;

const MAX_BATCH_ROWS: usize = 16;
const MAX_BATCH_BYTES: usize = 128 * 1024;

/// One content-free pre-context assessment, not one row per question.
/// Dispatch means invocation of SystemOne, not proof of an HTTP attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentRecord {
    pub schema_version: String,
    pub assessment_id: String,
    pub request_id: Option<String>,
    pub session_id: String,
    pub turn: u64,
    pub task_epoch_id: Option<String>,
    pub delivery_id: Option<String>,
    pub stamp_fingerprint: String,
    pub policy_fingerprint: String,
    pub requested_model: String,
    pub response_model: Option<String>,
    pub mode: String,
    pub request_start_ts: Option<String>,
    pub terminal_ts: Option<String>,
    pub decision_dispatched: bool,
    pub transport_dispatched: Option<bool>,
    pub attempts: Option<u32>,
    pub attempt_count_known: bool,
    /// Full local assessment span; never a service-reported duration.
    pub elapsed_ms: u64,
    /// Local SystemOne invocation span, absent when not invoked.
    pub decision_duration_ms: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub terminal_reason: String,
    pub unavailable: Option<bool>,
    pub policy_refusal_count: Option<u64>,
    pub host_fresh: Option<bool>,
    pub hint_state: Option<String>,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agreement {
    Agree,
    Disagree,
    Noncomparable,
}

impl Agreement {
    pub fn as_str(&self) -> &'static str {
        match self {
            Agreement::Agree => "agree",
            Agreement::Disagree => "disagree",
            Agreement::Noncomparable => "noncomparable",
        }
    }
}

/// Classify agreement between Jev's selected value and the baseline actual
/// choice. A missing side is `noncomparable`: null is unknown, not zero.
/// Numeric (noul/score) selections are only comparable against baselines of
/// the same shape, never against enum choices.
pub fn classify_agreement(jev_value: Option<&str>, baseline: Option<&str>) -> Agreement {
    let Some(jev) = jev_value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Agreement::Noncomparable;
    };
    let Some(actual) = baseline.map(str::trim).filter(|value| !value.is_empty()) else {
        return Agreement::Noncomparable;
    };
    // Noul/Score answers render as bare numbers while baselines are enum-like
    // choices (tool names, model ids, stop reasons). Comparing a number to a
    // non-number is noncomparable, not a disagreement.
    let jev_number = jev.parse::<f64>().ok();
    let actual_number = actual.parse::<f64>().ok();
    match (jev_number, actual_number) {
        (Some(_), None) | (None, Some(_)) => return Agreement::Noncomparable,
        (Some(jev_value), Some(actual_value)) => {
            return if (jev_value - actual_value).abs() < f64::EPSILON {
                Agreement::Agree
            } else {
                Agreement::Disagree
            };
        }
        (None, None) => {}
    }
    if jev.eq_ignore_ascii_case(actual) {
        Agreement::Agree
    } else {
        Agreement::Disagree
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationRecord {
    pub schema_version: String,
    pub request_id: String,
    pub attempt: u32,
    /// False when cancellation left only a logical dispatch count, not exact transport attempts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_count_known: Option<bool>,
    /// Opaque local session id.
    pub session_id: String,
    pub turn: u64,
    pub stage: String,
    pub state_fingerprint: String,
    pub state_schema_version: String,
    pub category: String,
    pub question_id: String,
    pub prompt_version: String,
    /// Captured decision mode. Independent compaction may run with decisions Off.
    pub mode: String,
    /// True only when this record describes an answer that was applied to an
    /// outgoing provider request. Every Compare record is false.
    pub applied: bool,
    pub request_start_ts: Option<String>,
    pub terminal_ts: Option<String>,
    pub duration_ms: Option<u64>,
    /// Actual response model (drift visible).
    pub response_model: Option<String>,
    /// Confidence for choice/score answers only.
    pub confidence: Option<f64>,
    pub selected_value: Option<String>,
    pub baseline_actual_choice: Option<String>,
    /// agree | disagree | noncomparable.
    pub agreement: Option<String>,
    /// What WOULD have happened under an acceptance policy (hypothetical).
    pub hypothetical_acceptance: Option<String>,
    pub fallback_reason: Option<String>,
    pub skipped_reason: Option<String>,
    /// `accepted` | `fallback`. Present on Active records only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance: Option<String>,
    /// Fields the host changed because this decision was applied. Active only,
    /// and empty for a refused answer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub applied_effects: Vec<crate::active::AppliedEffect>,
    /// Bounded host configuration captured before the decision, not a semantic answer.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub baseline_action: BTreeMap<String, String>,
    /// Configuration after the host applied the accepted decision.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub actual_action: BTreeMap<String, String>,
    /// Application outcome only. This never claims downstream task success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub observed_metrics: BTreeMap<String, u64>,
    /// Bounded, sanitized server-provided request id (`x-typesafe-request-id`),
    /// when the transport captured one. Untrusted server data: control chars
    /// stripped, length-capped, credential echoes refused. Correlation only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_request_id: Option<String>,
}

/// Everything an Active record row needs that is shared across rows of one
/// decision boundary.
#[derive(Debug, Clone)]
pub struct ActiveRecordContext {
    pub request_id: String,
    pub session_id: String,
    pub turn: u64,
    pub stage: String,
    pub response_model: Option<String>,
    pub duration_ms: Option<u64>,
    pub state_fingerprint: String,
    pub prompt_version: String,
    pub mode: String,
    pub request_start_ts: Option<String>,
    pub attempts: u32,
    pub attempt_count_known: bool,
    pub baselines: BTreeMap<String, Option<String>>,
    pub baseline_action: BTreeMap<String, String>,
    pub compaction_enabled: Option<bool>,
    pub observed_metrics: BTreeMap<String, u64>,
}

/// One row of an Active decision boundary: an accepted answer with the fields
/// it changed, or a refused answer with the single reason that stopped it.
#[derive(Debug, Clone)]
pub struct ActiveRecordRow {
    pub category: String,
    pub question_id: String,
    pub accepted: bool,
    pub selected_value: Option<String>,
    pub confidence: Option<f64>,
    pub fallback_reason: Option<String>,
    pub applied_effects: Vec<crate::active::AppliedEffect>,
    pub skipped_reason: Option<String>,
    pub actual_action: BTreeMap<String, String>,
    pub outcome: String,
}

/// Retention caps for Jev-owned record files.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    pub max_bytes: u64,
    pub max_age_days: u32,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_bytes: 5 * 1024 * 1024,
            max_age_days: 14,
        }
    }
}

struct PendingRequest {
    ctx: crate::scheduler::RequestContext,
    baseline_action: BTreeMap<String, String>,
    compaction_enabled: Option<bool>,
}

/// JSONL correlator. One record line per question, written exactly once.
pub struct Correlator {
    records_path: PathBuf,
    retention: RetentionPolicy,
    min_confidence: f64,
    pending: Mutex<BTreeMap<String, PendingRequest>>,
    file_guard: Mutex<()>,
    append_failures: AtomicU64,
}

impl Correlator {
    pub fn new(records_path: PathBuf, min_confidence: f64) -> Self {
        Self {
            records_path,
            retention: RetentionPolicy::default(),
            min_confidence,
            pending: Mutex::new(BTreeMap::new()),
            file_guard: Mutex::new(()),
            append_failures: AtomicU64::new(0),
        }
    }

    /// Constructor with explicit retention caps (size/time, jev files only).
    pub fn with_retention(
        records_path: PathBuf,
        min_confidence: f64,
        retention: RetentionPolicy,
    ) -> Self {
        Self {
            records_path,
            retention,
            min_confidence,
            pending: Mutex::new(BTreeMap::new()),
            file_guard: Mutex::new(()),
            append_failures: AtomicU64::new(0),
        }
    }

    /// Directory that holds Jev-owned files (retention scope).
    pub fn jev_dir(&self) -> PathBuf {
        self.records_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn now_rfc3339() -> String {
        Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
    }

    fn base_record(&self, ctx: &crate::scheduler::RequestContext) -> CorrelationRecord {
        CorrelationRecord {
            schema_version: RECORD_SCHEMA_VERSION.to_string(),
            request_id: ctx.request_id.clone(),
            attempt: 0,
            attempt_count_known: None,
            session_id: ctx.session_id.clone(),
            turn: ctx.turn,
            stage: ctx.stage.clone(),
            state_fingerprint: ctx.state_fingerprint.clone(),
            state_schema_version: ctx.state_schema_version.clone(),
            category: String::new(),
            question_id: String::new(),
            prompt_version: ctx.prompt_version.clone(),
            mode: ctx.mode.clone(),
            applied: false,
            request_start_ts: Some(ctx.request_start_ts.clone()),
            terminal_ts: None,
            duration_ms: None,
            response_model: None,
            confidence: None,
            selected_value: None,
            baseline_actual_choice: None,
            agreement: None,
            hypothetical_acceptance: None,
            fallback_reason: None,
            skipped_reason: None,
            acceptance: None,
            applied_effects: Vec::new(),
            baseline_action: BTreeMap::new(),
            actual_action: BTreeMap::new(),
            outcome: None,
            compaction_enabled: None,
            observed_metrics: BTreeMap::new(),
            server_request_id: None,
        }
    }

    /// Register a dispatched request's baselines (captured at the boundary).
    /// The pending map is bounded: the oldest unsettled request is recorded
    /// as skipped when the map grows past the cap.
    pub fn track(&self, ctx: &crate::scheduler::RequestContext) {
        self.track_boundary(ctx, BTreeMap::new(), None);
    }

    pub fn track_boundary(&self, ctx: &crate::scheduler::RequestContext, baseline_action: BTreeMap<String, String>, compaction_enabled: Option<bool>) {
        const MAX_PENDING: usize = 256;
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while pending.len() >= MAX_PENDING {
            let Some(oldest_key) = pending.keys().next().cloned() else {
                break;
            };
            if let Some(oldest) = pending.remove(&oldest_key) {
                let terminal_ts = Self::now_rfc3339();
                let mut records = Vec::new();
                for meta in &oldest.ctx.questions {
                    let mut record = self.base_record(&oldest.ctx);
                    record.category = meta.category.clone();
                    record.question_id = meta.question_id.clone();
                    record.terminal_ts = Some(terminal_ts.clone());
                    record.skipped_reason = Some("unsettled".to_string());
                    records.push(record);
                }
                self.write_records(&records);
            }
        }
        pending.insert(ctx.request_id.clone(), PendingRequest { ctx: ctx.clone(), baseline_action, compaction_enabled });
    }

    /// Handle a scheduler outcome. Writes each question's record exactly once
    /// and forgets the pending entry.
    pub fn handle_result(&self, result: &JobResult) {
        match result {
            JobResult::Completed {
                ctx,
                outcome,
                duration_ms,
            } => self.complete(ctx, outcome, *duration_ms),
            JobResult::Failed {
                ctx,
                reason,
                attempts,
            } => {
                self.skip_request(ctx, reason, Some(*attempts));
            }
            JobResult::Dropped { ctx, reason } => {
                self.skip_request(ctx, reason, None);
            }
        }
    }

    fn complete(
        &self,
        ctx: &crate::scheduler::RequestContext,
        outcome: &crate::types::DecisionOutcome,
        duration_ms: u64,
    ) {
        let terminal_ts = Self::now_rfc3339();
        let records: Vec<CorrelationRecord> = {
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(entry) = pending.remove(&ctx.request_id) else {
                return;
            };
            let mut records = Vec::new();
            for meta in &entry.ctx.questions {
                let Some(decision) = outcome
                    .records
                    .iter()
                    .find(|decision| decision.question_id == meta.question_id)
                else {
                    // Missing answer: logged skip, never fabricated. The
                    // precise validation reason (unknown_answer_id,
                    // missing_answer_id, choice_not_in_criteria, ...) is
                    // carried through when the client reported one.
                    let skip_reason = outcome
                        .skips
                        .iter()
                        .find(|(question_id, _)| question_id == &meta.question_id)
                        .map(|(_, reason)| *reason)
                        .unwrap_or("answer_missing");
                    let mut record = self.base_record(ctx);
                    record.baseline_action = sanitize_action(&entry.baseline_action);
                    record.compaction_enabled = entry.compaction_enabled;
                    record.category = meta.category.clone();
                    record.question_id = meta.question_id.clone();
                    record.terminal_ts = Some(terminal_ts.clone());
                    record.skipped_reason = Some(skip_reason.to_string());
                    records.push(record);
                    continue;
                };
                let answer = &decision.answer;
                let mut record = self.base_record(ctx);
                    record.baseline_action = sanitize_action(&entry.baseline_action);
                    record.compaction_enabled = entry.compaction_enabled;
                record.observed_metrics = usage_metrics(outcome);
                record.server_request_id = outcome
                    .server_request_id
                    .as_deref()
                    .map(|value| sanitize_text(value, MAX_FIELD_TEXT));
                record.category = meta.category.clone();
                record.question_id = meta.question_id.clone();
                record.terminal_ts = Some(terminal_ts.clone());
                record.duration_ms = Some(duration_ms);
                record.response_model = decision
                    .response_model
                    .as_deref()
                    .map(|m| crate::correlate::sanitize_text(m, MAX_FIELD_TEXT));
                // Truthful attempt number: the client owns retries and
                // reports what it actually made for this outcome.
                record.attempt = outcome.attempts;
                record.confidence = answer.confidence();
                record.selected_value = Some(crate::correlate::sanitize_text(
                    &answer.selected_value(),
                    MAX_FIELD_TEXT,
                ));
                record.baseline_actual_choice = entry
                    .ctx
                    .baselines
                    .get(&meta.question_id)
                    .cloned()
                    .unwrap_or(None)
                    .map(|b| crate::correlate::sanitize_text(&b, MAX_FIELD_TEXT));
                record.agreement = Some(
                    classify_agreement(record.selected_value.as_deref(), record.baseline_actual_choice.as_deref())
                        .as_str()
                        .to_string(),
                );
                // Hypothetical acceptance only: nothing was applied.
                match record.confidence {
                    Some(confidence) if confidence >= self.min_confidence => {
                        record.hypothetical_acceptance = Some("accepted".to_string());
                    }
                    Some(_) => {
                        record.hypothetical_acceptance = Some("fallback".to_string());
                        record.fallback_reason = Some("low_confidence".to_string());
                    }
                    None => {
                        record.hypothetical_acceptance = Some("fallback".to_string());
                        record.fallback_reason = Some("no_confidence_signal".to_string());
                    }
                }
                records.push(record);
            }
            records
        };
        self.write_records(&records);
    }

    /// Record a request-level or category-level skip.
    pub fn skip_request(
        &self,
        ctx: &crate::scheduler::RequestContext,
        reason: &str,
        attempts: Option<u32>,
    ) {
        let terminal_ts = Self::now_rfc3339();
        let records: Vec<CorrelationRecord> = {
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(entry) = pending.remove(&ctx.request_id) else {
                return;
            };
            entry
                .ctx
                .questions
                .iter()
                .map(|meta| {
                    let mut record = self.base_record(ctx);
                    record.baseline_action = sanitize_action(&entry.baseline_action);
                    record.compaction_enabled = entry.compaction_enabled;
                    record.category = meta.category.clone();
                    record.question_id = meta.question_id.clone();
                    record.terminal_ts = Some(terminal_ts.clone());
                    record.attempt = attempts.unwrap_or(0);
                    record.skipped_reason = Some(crate::correlate::sanitize_text(reason, MAX_FIELD_TEXT));
                    record.baseline_actual_choice = entry
                        .ctx
                        .baselines
                        .get(&meta.question_id)
                        .cloned()
                        .unwrap_or(None)
                        .map(|b| crate::correlate::sanitize_text(&b, MAX_FIELD_TEXT));
                    record.agreement = Some(Agreement::Noncomparable.as_str().to_string());
                    record
                })
                .collect()
        };
        self.write_records(&records);
    }

    /// Record a category that was skipped before any request existed.
    pub fn record_skipped_category(
        &self,
        session_id: &str,
        turn: u64,
        stage: &str,
        category: DecisionCategory,
        reason: &str,
        prompt_version: &str,
        mode: &str,
    ) {
        let record = CorrelationRecord {
            schema_version: RECORD_SCHEMA_VERSION.to_string(),
            request_id: format!("skipped-{}", uuid::Uuid::new_v4()),
            attempt: 0,
            attempt_count_known: None,
            session_id: sanitize_text(session_id, MAX_FIELD_TEXT),
            turn,
            stage: stage.to_string(),
            state_fingerprint: String::new(),
            state_schema_version: crate::snapshot::STATE_SCHEMA_VERSION.to_string(),
            category: category.as_str().to_string(),
            question_id: format!("{}.0", category.as_str()),
            prompt_version: prompt_version.to_string(),
            mode: mode.to_string(),
            applied: false,
            request_start_ts: None,
            terminal_ts: Some(Self::now_rfc3339()),
            duration_ms: None,
            response_model: None,
            confidence: None,
            selected_value: None,
            baseline_actual_choice: None,
            agreement: Some(Agreement::Noncomparable.as_str().to_string()),
            hypothetical_acceptance: None,
            fallback_reason: None,
            skipped_reason: Some(sanitize_text(reason, MAX_FIELD_TEXT)),
            acceptance: None,
            applied_effects: Vec::new(),
            baseline_action: BTreeMap::new(),
            actual_action: BTreeMap::new(),
            outcome: None,
            compaction_enabled: None,
            observed_metrics: BTreeMap::new(),
            server_request_id: None,
        };
        self.write_record(&record);
    }

    /// Write one Active decision boundary. Returns the number of rows written.
    ///
    /// An accepted row carries `applied: true` only when the host changed fields.
    /// A refused or accepted-no-effect row carries `applied: false`. The single
    /// reason that stopped it. Nothing is written when there is nothing to
    /// report: an empty boundary is not an event.
    pub fn record_active_rows(&self, ctx: &ActiveRecordContext, rows: &[ActiveRecordRow]) -> usize {
        let terminal_ts = Self::now_rfc3339();
        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            let record = CorrelationRecord {
                schema_version: ACTIVE_RECORD_SCHEMA_VERSION.to_string(),
                request_id: ctx.request_id.clone(),
                attempt: ctx.attempts,
                attempt_count_known: Some(ctx.attempt_count_known),
                session_id: sanitize_text(&ctx.session_id, MAX_FIELD_TEXT),
                turn: ctx.turn,
                stage: ctx.stage.clone(),
                state_fingerprint: ctx.state_fingerprint.clone(),
                state_schema_version: crate::snapshot::STATE_SCHEMA_VERSION.to_string(),
                category: sanitize_text(&row.category, MAX_FIELD_TEXT),
                question_id: sanitize_text(&row.question_id, MAX_FIELD_TEXT),
                prompt_version: ctx.prompt_version.clone(),
                mode: ctx.mode.clone(),
                applied: row.accepted && !row.applied_effects.is_empty(),
                request_start_ts: ctx.request_start_ts.clone(),
                terminal_ts: Some(terminal_ts.clone()),
                duration_ms: ctx.duration_ms,
                response_model: ctx.response_model.as_deref().map(|value| sanitize_text(value, MAX_FIELD_TEXT)),
                confidence: row.confidence,
                selected_value: row
                    .selected_value
                    .as_deref()
                    .map(|value| sanitize_text(value, MAX_FIELD_TEXT)),
                baseline_actual_choice: ctx.baselines.get(&row.question_id).cloned().flatten()
                    .map(|value| sanitize_text(&value, MAX_FIELD_TEXT)),
                agreement: Some(classify_agreement(row.selected_value.as_deref(),
                    ctx.baselines.get(&row.question_id).and_then(|value| value.as_deref())).as_str().to_string()),
                hypothetical_acceptance: None,
                fallback_reason: row
                    .fallback_reason
                    .as_deref()
                    .map(|reason| sanitize_text(reason, MAX_FIELD_TEXT)),
                skipped_reason: row.skipped_reason.clone(),
                acceptance: Some(if row.accepted { "accepted".to_string() } else { "fallback".to_string() }),
                applied_effects: if row.accepted { row.applied_effects.iter().map(|effect| crate::active::AppliedEffect::new(
                    sanitize_text(&effect.field, 64),
                    effect.from.as_deref().map(sanitize_metadata),
                    effect.to.as_deref().map(sanitize_metadata),
                )).collect() } else { Vec::new() },
                baseline_action: sanitize_action(&ctx.baseline_action),
                actual_action: sanitize_action(&row.actual_action),
                outcome: Some(sanitize_text(&row.outcome, MAX_FIELD_TEXT)),
                compaction_enabled: ctx.compaction_enabled,
                observed_metrics: ctx.observed_metrics.clone(),
                // Active rows carry the sanitized server id through the context when
                // the host wires it; the field stays additive and optional here.
                server_request_id: None,
            };
            records.push(record);
        }
        self.write_records(&records)
    }

    /// Independent compaction audit. Statistics are local counts, never content.
    pub fn record_compaction(&self, ctx: &crate::scheduler::RequestContext, response_model: Option<&str>, attempts: u32, attempts_known: bool, duration_ms: Option<u64>, stats: &serde_json::Value, fallback: Option<&str>) {
        let mut record = self.base_record(ctx);
        record.schema_version = "jev.compaction/1".to_string();
        record.category = "compaction".to_string();
        record.question_id = "compaction.audit".to_string();
        record.compaction_enabled = Some(true);
        record.attempt = attempts;
        record.attempt_count_known = Some(attempts_known);
        record.response_model = response_model.map(|model| sanitize_text(model, MAX_FIELD_TEXT));
        record.duration_ms = duration_ms;
        record.terminal_ts = Some(Self::now_rfc3339());
        record.applied = fallback.is_none() && stats.get("applied").and_then(serde_json::Value::as_bool) == Some(true);
        record.outcome = Some(if record.applied { "applied" } else if fallback.is_some() { "fallback" } else { "no_effect" }.to_string());
        record.fallback_reason = fallback.map(|reason| sanitize_text(reason, MAX_FIELD_TEXT));
        record.observed_metrics = stats.as_object().map(|values| values.iter().take(32)
            .filter_map(|(key, value)| value.as_u64().map(|count| (sanitize_text(key, 64), count))).collect()).unwrap_or_default();
        record.observed_metrics.insert("question_count".to_string(), ctx.questions.len() as u64);
        self.write_record(&record);
    }

    /// Local diagnostic only. No append failure is counted as a written row.
    pub fn append_failure_count(&self) -> u64 {
        self.append_failures.load(Ordering::Relaxed)
    }

    pub fn record_assessment(&self, record: &AssessmentRecord) -> bool {
        let mut record = record.clone();
        record.schema_version = "jev.assessment/1".to_string();
        record.assessment_id = sanitize_text(&record.assessment_id, MAX_FIELD_TEXT);
        record.request_id = record.request_id.as_deref().map(|value| sanitize_text(value, MAX_FIELD_TEXT));
        record.session_id = sanitize_text(&record.session_id, MAX_FIELD_TEXT);
        record.task_epoch_id = record.task_epoch_id.as_deref().map(sanitize_metadata);
        record.delivery_id = record.delivery_id.as_deref().map(sanitize_metadata);
        record.stamp_fingerprint = sanitize_text(&record.stamp_fingerprint, MAX_FIELD_TEXT);
        record.policy_fingerprint = sanitize_text(&record.policy_fingerprint, MAX_FIELD_TEXT);
        record.requested_model = sanitize_metadata(&record.requested_model);
        record.response_model = record.response_model.as_deref().map(sanitize_metadata);
        record.mode = sanitize_text(&record.mode, MAX_FIELD_TEXT);
        record.terminal_reason = sanitize_text(&record.terminal_reason, MAX_FIELD_TEXT);
        record.hint_state = record.hint_state.as_deref().map(|value| sanitize_text(value, MAX_FIELD_TEXT));
        self.write_records(std::slice::from_ref(&record)) == 1
    }

    fn write_record(&self, record: &CorrelationRecord) {
        self.write_records(std::slice::from_ref(record));
    }

    /// One synchronous guard per existing phase. Buffer caps and rotation
    /// split writes, never transactions or callbacks. No detached work.
    fn write_records<T: Serialize>(&self, records: &[T]) -> usize {
        if records.is_empty() { return 0; }
        let _guard = self.file_guard.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = self.append_records(records);
        if result.1 { self.append_failures.fetch_add(1, Ordering::Relaxed); }
        result.0
    }

    fn append_records<T: Serialize>(&self, records: &[T]) -> (usize, bool) {
        if let Some(parent) = self.records_path.parent() {
            if std::fs::create_dir_all(parent).is_err() { return (0, true); }
        }
        let mut file: Option<std::fs::File> = None;
        let mut size = 0u64;
        let mut buffer = Vec::new();
        let mut rows = 0usize;
        let mut written = 0usize;
        for record in records {
            let mut line = match bounded_record_bytes(record) {
                Ok(line) => line,
                _ => {
                    if !buffer.is_empty() {
                        let (count, _) = append_counted(file.as_mut().expect("opened batch"), &buffer);
                        written += count;
                    }
                    return (written, true);
                }
            };
            line.push(b'\n');
            if !buffer.is_empty() && (rows == MAX_BATCH_ROWS || buffer.len() + line.len() > MAX_BATCH_BYTES) {
                let (count, failed) = append_counted(file.as_mut().expect("opened batch"), &buffer);
                written += count;
                if failed { return (written, true); }
                size += buffer.len() as u64;
                buffer.clear();
                rows = 0;
            }
            if file.is_none() {
                let opened = std::fs::OpenOptions::new().create(true).append(true).open(&self.records_path);
                let Ok(opened) = opened else { return (written, true); };
                let Ok(meta) = opened.metadata() else { return (written, true); };
                size = meta.len();
                file = Some(opened);
            }
            buffer.extend_from_slice(&line);
            rows += 1;
            // Preserve the old append-then-rotate row boundary exactly.
            if size.saturating_add(buffer.len() as u64) > self.retention.max_bytes {
                let (count, failed) = append_counted(file.as_mut().expect("opened batch"), &buffer);
                written += count;
                if failed { return (written, true); }
                buffer.clear();
                rows = 0;
                drop(file.take());
                if self.rotate_records().is_err() { return (written, true); }
                self.enforce_age();
            }
        }
        if !buffer.is_empty() {
            let (count, failed) = append_counted(file.as_mut().expect("opened batch"), &buffer);
            written += count;
            if failed { return (written, true); }
        }
        (written, false)
    }

    fn rotate_records(&self) -> std::io::Result<()> {
        let mut rotated = self.records_path.clone().into_os_string();
        rotated.push(".1");
        let rotated = PathBuf::from(rotated);
        // Preserve the previous generation if replacement fails.
        std::fs::rename(&self.records_path, &rotated)
    }

    fn enforce_age(&self) {
        // Time cap: delete only jev-owned record files older than the cap.
        let Some(dir) = self.records_path.parent() else {
            return;
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let cutoff = Utc::now() - chrono::Duration::days(self.retention.max_age_days as i64);
        for entry in entries_of(&entries_collect(entries)) {
            // Only record files are age-managed. The credential store and the
            // settings file live in this directory too and must never be
            // deleted by retention.
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if !file_name.starts_with("records.jsonl") {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            let modified = meta
                .modified()
                .ok()
                .map(DateTime::<Utc>::from)
                .unwrap_or(Utc::now());
            if modified < cutoff {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

fn bounded_record_bytes(record: &impl Serialize) -> Result<Vec<u8>, serde_json::Error> {
    struct Bounded(Vec<u8>);
    impl Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.0.len().saturating_add(bytes.len()) >= MAX_BATCH_BYTES {
                return Err(std::io::Error::other("audit_row_byte_limit"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let mut bytes = Bounded(Vec::new());
    serde_json::to_writer(&mut bytes, record)?;
    Ok(bytes.0)
}

/// Counts complete JSONL rows only, including on a short/failed write.
fn append_counted(writer: &mut impl Write, bytes: &[u8]) -> (usize, bool) {
    let mut offset = 0;
    let mut failed = false;
    while offset < bytes.len() {
        match writer.write(&bytes[offset..]) {
            Ok(0) => { failed = true; break; }
            Ok(count) => offset += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => { failed = true; break; }
        }
    }
    (bytes[..offset].iter().filter(|byte| **byte == b'\n').count(), failed)
}

pub(crate) fn usage_metrics(outcome: &crate::types::DecisionOutcome) -> BTreeMap<String, u64> {
    let mut metrics = BTreeMap::new();
    // Knownness is explicit END-TO-END (root decision): absent or null usage is UNKNOWN
    // and emits NO metric (it is never fabricated as 0). `Some(0)` is a real measured
    // zero and IS emitted, so records can distinguish measured-zero from unknown.
    if let Some(input_tokens) = outcome.usage.input_tokens {
        metrics.insert("jev_input_tokens".into(), input_tokens);
    }
    if let Some(output_tokens) = outcome.usage.output_tokens {
        metrics.insert("jev_output_tokens".into(), output_tokens);
    }
    metrics
}

fn entries_collect(entries: std::fs::ReadDir) -> Vec<std::fs::DirEntry> {
    entries.filter_map(|entry| entry.ok()).collect()
}

fn entries_of(entries: &[std::fs::DirEntry]) -> &[std::fs::DirEntry] {
    entries
}

fn sanitize_metadata(value: &str) -> String {
    sanitize_text(&crate::redact::bounded_excerpt(value, MAX_FIELD_TEXT), MAX_FIELD_TEXT)
}

fn sanitize_action(values: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    values.iter().take(16).map(|(key, value)|
        (sanitize_text(key, 64), sanitize_metadata(value))).collect()
}

/// Truncate and strip control characters from record field values.
pub fn sanitize_text(text: &str, max_len: usize) -> String {
    let cleaned: String = text
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect();
    let (bounded, truncated) = crate::snapshot::truncate_text(&cleaned, max_len);
    if truncated {
        format!("{bounded}...")
    } else {
        bounded
    }
}

/// Read records back, bounded to the newest `max_bytes` bytes of the file.
pub fn read_records(records_path: &Path, max_bytes: u64) -> Vec<CorrelationRecord> {
    let Ok(meta) = std::fs::metadata(records_path) else {
        return Vec::new();
    };
    let file_len = meta.len();
    let skip = file_len.saturating_sub(max_bytes);
    let Ok(mut file) = std::fs::File::open(records_path) else { return Vec::new(); };
    if file.seek(SeekFrom::Start(skip)).is_err() { return Vec::new(); }
    let mut bytes = Vec::new();
    if file.take(max_bytes).read_to_end(&mut bytes).is_err() { return Vec::new(); }
    let text = String::from_utf8_lossy(&bytes);
    let mut start = 0usize;
    if skip > 0 {
        if let Some(pos) = text.find('\n') {
            start = pos + 1;
        }
    }
    text[start..]
        .lines()
        .filter_map(|line| serde_json::from_str::<CorrelationRecord>(line).ok())
        .collect()
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::scheduler::{QuestionMeta, RequestContext};
    use crate::types::{Answer, DecisionOutcome, DecisionRecord, Usage};

    fn context(session: &str, count: usize) -> RequestContext {
        RequestContext {
            request_id: format!("request-{session}"), session_id: session.into(), turn: 7,
            stage: "skill_suggestion".into(), state_fingerprint: "immutable-hash".into(),
            state_schema_version: crate::snapshot::STATE_SCHEMA_VERSION.into(),
            prompt_version: "fixture/1".into(), mode: "compareAndActive".into(),
            requested_model: "synthetic-model".into(), policy_generation: "epoch-1".into(),
            questions: (0..count).map(|index| QuestionMeta {
                question_id: format!("skill_suggestion.{index}"), category: "skill_suggestion".into(),
            }).collect(),
            baselines: BTreeMap::new(), request_start_ts: "2026-10-01T00:00:00.000Z".into(),
        }
    }

    fn active_context(ctx: &RequestContext) -> ActiveRecordContext {
        ActiveRecordContext {
            request_id: ctx.request_id.clone(), session_id: ctx.session_id.clone(), turn: ctx.turn,
            stage: ctx.stage.clone(), state_fingerprint: ctx.state_fingerprint.clone(),
            response_model: Some("synthetic\nmodel".into()), duration_ms: Some(12),
            prompt_version: ctx.prompt_version.clone(), mode: ctx.mode.clone(),
            request_start_ts: Some(ctx.request_start_ts.clone()), attempts: 2, attempt_count_known: true,
            baselines: ctx.baselines.clone(), baseline_action: BTreeMap::new(),
            compaction_enabled: None, observed_metrics: BTreeMap::from([("jev_input_tokens".into(), 0)]),
        }
    }

    fn active_rows(ctx: &RequestContext) -> Vec<ActiveRecordRow> {
        ctx.questions.iter().map(|question| ActiveRecordRow {
            category: question.category.clone(), question_id: question.question_id.clone(),
            accepted: false, selected_value: Some("none\n".into()), confidence: Some(0.9),
            fallback_reason: Some("category_not_appliable".into()), applied_effects: Vec::new(),
            skipped_reason: None, actual_action: BTreeMap::new(), outcome: "refused".into(),
        }).collect()
    }

    #[test]
    fn per_phase_compare_and_active_rows_preserve_binding_sanitization_and_knownness() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let correlator = Correlator::new(path.clone(), 0.7);
        let ctx = context("phase", 5);
        correlator.track(&ctx);
        let outcome = DecisionOutcome {
            records: ctx.questions[..4].iter().map(|question| DecisionRecord {
                question_id: question.question_id.clone(), category: DecisionCategory::SkillSuggestion,
                answer: Answer::Noul { noul: 0.9 }, response_model: Some("synthetic\nmodel".into()),
                requested_model: "synthetic-model".into(), applied: false,
            }).collect(),
            skips: vec![(ctx.questions[4].question_id.clone(), "missing_answer_id")],
            response_model: Some("synthetic-model".into()),
            usage: Usage { input_tokens: Some(0), output_tokens: None }, applied: false,
            attempts: 2, server_request_id: Some("server\nid".into()),
        };
        correlator.handle_result(&JobResult::Completed { ctx: ctx.clone(), outcome, duration_ms: 12 });
        assert_eq!(correlator.record_active_rows(&active_context(&ctx), &active_rows(&ctx)), 5);
        let records = read_records(&path, 1 << 20);
        assert_eq!(records.len(), 10);
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.schema_version, if index < 5 { RECORD_SCHEMA_VERSION } else { ACTIVE_RECORD_SCHEMA_VERSION });
            assert_eq!(record.request_id, ctx.request_id);
            assert_eq!(record.session_id, ctx.session_id);
            assert_eq!(record.turn, ctx.turn);
            assert_eq!(record.stage, ctx.stage);
            assert_eq!(record.question_id, ctx.questions[index % 5].question_id);
            assert!(!record.applied);
            if index == 4 {
                assert_eq!(record.attempt, 0);
                assert!(record.duration_ms.is_none());
                assert!(record.observed_metrics.is_empty(), "host skip must not duplicate transport usage");
                assert_eq!(record.skipped_reason.as_deref(), Some("missing_answer_id"));
            } else {
                assert_eq!(record.attempt, 2);
                assert_eq!(record.duration_ms, Some(12));
                assert_eq!(record.response_model.as_deref(), Some("synthetic model"));
                assert_eq!(record.observed_metrics.get("jev_input_tokens"), Some(&0));
                assert!(!record.observed_metrics.contains_key("jev_output_tokens"));
            }
        }
        let unchanged = std::fs::read(&path).unwrap();
        correlator.handle_result(&JobResult::Dropped { ctx, reason: "duplicate".into() });
        assert_eq!(std::fs::read(&path).unwrap(), unchanged, "a phase is recorded once");
        assert_eq!(correlator.append_failure_count(), 0);
    }

    #[test]
    fn open_and_oversized_row_failure_never_count_attempted_rows_as_written() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = context("failure", 4);
        let blocked = dir.path().join("records.jsonl");
        std::fs::create_dir(&blocked).unwrap();
        let correlator = Correlator::new(blocked, 0.7);
        assert_eq!(correlator.record_active_rows(&active_context(&ctx), &active_rows(&ctx)), 0);
        assert_eq!(correlator.append_failure_count(), 1);
        let path = dir.path().join("valid.jsonl");
        let correlator = Correlator::new(path.clone(), 0.7);
        let mut active = active_context(&ctx);
        active.state_fingerprint = "x".repeat(MAX_BATCH_BYTES);
        assert_eq!(correlator.record_active_rows(&active, &active_rows(&ctx)), 0);
        assert_eq!(correlator.append_failure_count(), 1);
        assert!(!path.exists());
    }

    #[test]
    fn counted_append_handles_short_interrupted_zero_and_failed_writes() {
        struct Short { bytes: Vec<u8>, limit: usize, fail: bool, interrupted: bool }
        impl Write for Short {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.interrupted {
                    self.interrupted = false;
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                if self.bytes.len() >= self.limit {
                    return if self.fail { Err(std::io::ErrorKind::Other.into()) } else { Ok(0) };
                }
                let count = bytes.len().min(2).min(self.limit - self.bytes.len());
                self.bytes.extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        for fail in [false, true] {
            let mut writer = Short { bytes: Vec::new(), limit: 7, fail, interrupted: true };
            assert_eq!(append_counted(&mut writer, b"{}\n{}\n{}\n"), (2, true));
            assert_eq!(writer.bytes, b"{}\n{}\n{" );
        }
        let mut writer = Short { bytes: Vec::new(), limit: usize::MAX, fail: false, interrupted: true };
        assert_eq!(append_counted(&mut writer, b"{}\n{}\n{}\n"), (3, false));
    }

    #[test]
    fn rotation_splits_at_the_same_row_boundary_and_age_ignores_credentials() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let limit = 41;
        let correlator = Correlator::with_retention(path.clone(), 0.7,
            RetentionPolicy { max_bytes: limit, max_age_days: 14 });
        let lines: Vec<_> = (0..41).map(|index| serde_json::json!({"index": index})).collect();
        let mut expected = Vec::new();
        let mut rotated = Vec::new();
        for line in &lines {
            expected.extend_from_slice(&serde_json::to_vec(line).unwrap());
            expected.push(b'\n');
            if expected.len() as u64 > limit { rotated = std::mem::take(&mut expected); }
        }
        assert_eq!(correlator.write_records(&lines), lines.len());
        assert_eq!(std::fs::read(path.with_file_name("records.jsonl.1")).unwrap(), rotated);
        assert_eq!(std::fs::read(&path).unwrap_or_default(), expected);
        let old_record = dir.path().join("records.jsonl.old");
        let credential = dir.path().join("default.key");
        for old in [&old_record, &credential] {
            std::fs::write(old, b"keep-private").unwrap();
            std::fs::File::options().write(true).open(old).unwrap().set_times(
                std::fs::FileTimes::new().set_modified(std::time::UNIX_EPOCH)).unwrap();
        }
        correlator.enforce_age();
        assert!(!old_record.exists());
        assert_eq!(std::fs::read(credential).unwrap(), b"keep-private");
        assert_eq!(correlator.append_failure_count(), 0);
    }


    #[test]
    fn byte_cap_splits_preserve_all_serialized_rows_and_rotation_failure_is_visible() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let correlator = Correlator::new(path.clone(), 0.7);
        let rows: Vec<_> = (0..32).map(|index| serde_json::json!({"index": index, "bounded": "x".repeat(12_000)})).collect();
        assert_eq!(correlator.write_records(&rows), 32);
        let expected: Vec<u8> = rows.iter().flat_map(|row| {
            let mut line = serde_json::to_vec(row).unwrap();
            line.push(b'\n');
            line
        }).collect();
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        assert_eq!(correlator.append_failure_count(), 0);
        let blocked = dir.path().join("records.jsonl.1");
        std::fs::create_dir(&blocked).unwrap();
        let correlator = Correlator::with_retention(path.clone(), 0.7,
            RetentionPolicy { max_bytes: 1, max_age_days: 14 });
        assert_eq!(correlator.write_records(&[serde_json::json!({}), serde_json::json!({})]), 1);
        assert_eq!(correlator.append_failure_count(), 1, "completed append is distinct from failed rotation");
    }


    #[test]
    fn rotation_failure_preserves_the_previous_generation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let previous = dir.path().join("records.jsonl.1");
        std::fs::write(&previous, b"previous-generation\n").unwrap();
        let correlator = Correlator::new(path, 0.7);
        assert!(correlator.rotate_records().is_err(), "missing source makes replacement fail");
        assert_eq!(std::fs::read(previous).unwrap(), b"previous-generation\n");
    }

    #[test]
    fn simultaneous_sessions_never_interleave_rows_inside_a_phase() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let correlator = std::sync::Arc::new(Correlator::new(path.clone(), 0.7));
        let mut workers = Vec::new();
        for index in 0..8 {
            let captured = correlator.clone();
            workers.push(std::thread::spawn(move || {
                let ctx = context(&format!("session-{index}"), 16);
                assert_eq!(captured.record_active_rows(&active_context(&ctx), &active_rows(&ctx)), 16);
            }));
        }
        for worker in workers { worker.join().unwrap(); }
        let records = read_records(&path, 1 << 20);
        assert_eq!(records.len(), 128);
        for phase in records.chunks(16) {
            assert!(phase.iter().all(|row| row.session_id == phase[0].session_id && row.request_id == phase[0].request_id));
            for (index, row) in phase.iter().enumerate() {
                assert_eq!(row.question_id, format!("skill_suggestion.{index}"));
            }
        }
    }
}
