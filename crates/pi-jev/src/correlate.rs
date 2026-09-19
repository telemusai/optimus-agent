//! Versioned JSONL correlation records (DESIGN.md section 7).
//!
//! Every Compare record carries `applied=false` ALWAYS, classifies agreement
//! against the boundary-captured baseline, marks savings hypothetical, holds
//! no secrets, raw prompts, transcripts or tool outputs, and cleans up with
//! size/time retention strictly confined to Jev's own files.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};

use crate::scheduler::JobResult;
use crate::types::DecisionCategory;

/// Record schema version.
pub const RECORD_SCHEMA_VERSION: &str = "jev.compare/1";

/// Maximum text length of any single field value kept in a record.
pub const MAX_FIELD_TEXT: usize = 120;

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
    /// Opaque local session id.
    pub session_id: String,
    pub turn: u64,
    pub stage: String,
    pub state_fingerprint: String,
    pub state_schema_version: String,
    pub category: String,
    pub question_id: String,
    pub prompt_version: String,
    /// Comparison mode; records only exist in Compare.
    pub mode: String,
    /// ALWAYS false in this build. Test-enforced.
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
}

/// JSONL correlator. One record line per question, written exactly once.
pub struct Correlator {
    records_path: PathBuf,
    retention: RetentionPolicy,
    min_confidence: f64,
    pending: Mutex<BTreeMap<String, PendingRequest>>,
    file_guard: Mutex<()>,
}

impl Correlator {
    pub fn new(records_path: PathBuf, min_confidence: f64) -> Self {
        Self {
            records_path,
            retention: RetentionPolicy::default(),
            min_confidence,
            pending: Mutex::new(BTreeMap::new()),
            file_guard: Mutex::new(()),
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
        }
    }

    /// Register a dispatched request's baselines (captured at the boundary).
    /// The pending map is bounded: the oldest unsettled request is recorded
    /// as skipped when the map grows past the cap.
    pub fn track(&self, ctx: &crate::scheduler::RequestContext) {
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
                for meta in &oldest.ctx.questions {
                    let mut record = self.base_record(&oldest.ctx);
                    record.category = meta.category.clone();
                    record.question_id = meta.question_id.clone();
                    record.terminal_ts = Some(terminal_ts.clone());
                    record.skipped_reason = Some("unsettled".to_string());
                    self.write_record(&record);
                }
            }
        }
        pending.insert(ctx.request_id.clone(), PendingRequest { ctx: ctx.clone() });
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
                    record.category = meta.category.clone();
                    record.question_id = meta.question_id.clone();
                    record.terminal_ts = Some(terminal_ts.clone());
                    record.skipped_reason = Some(skip_reason.to_string());
                    records.push(record);
                    continue;
                };
                let answer = &decision.answer;
                let mut record = self.base_record(ctx);
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
        for record in records {
            self.write_record(&record);
        }
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
        for record in records {
            self.write_record(&record);
        }
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
    ) {
        let record = CorrelationRecord {
            schema_version: RECORD_SCHEMA_VERSION.to_string(),
            request_id: format!("skipped-{}", uuid::Uuid::new_v4()),
            attempt: 0,
            session_id: sanitize_text(session_id, MAX_FIELD_TEXT),
            turn,
            stage: stage.to_string(),
            state_fingerprint: String::new(),
            state_schema_version: crate::snapshot::STATE_SCHEMA_VERSION.to_string(),
            category: category.as_str().to_string(),
            question_id: format!("{}.0", category.as_str()),
            prompt_version: prompt_version.to_string(),
            mode: "compare".to_string(),
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
        };
        self.write_record(&record);
    }

    fn write_record(&self, record: &CorrelationRecord) {
        let _guard = self.file_guard.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(parent) = self.records_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let line = match serde_json::to_string(record) {
            Ok(line) => line,
            Err(_) => return,
        };
        let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.records_path)
        else {
            return;
        };
        let _ = writeln!(file, "{line}");
        self.enforce_retention();
    }

    fn enforce_retention(&self) {
        // Size cap: rotate records.jsonl -> records.jsonl.1 (at most one
        // generation), strictly inside the jev directory.
        let Ok(meta) = std::fs::metadata(&self.records_path) else {
            return;
        };
        if meta.len() <= self.retention.max_bytes {
            return;
        }
        let mut rotated = self.records_path.clone().into_os_string();
        rotated.push(".1");
        let rotated = PathBuf::from(rotated);
        let _ = std::fs::rename(&self.records_path, &rotated);
        self.enforce_age();
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

fn entries_collect(entries: std::fs::ReadDir) -> Vec<std::fs::DirEntry> {
    entries.filter_map(|entry| entry.ok()).collect()
}

fn entries_of(entries: &[std::fs::DirEntry]) -> &[std::fs::DirEntry] {
    entries
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
    let Ok(bytes) = std::fs::read(records_path) else {
        return Vec::new();
    };
    let slice = &bytes[skip.min(bytes.len() as u64) as usize..];
    let text = String::from_utf8_lossy(slice);
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
