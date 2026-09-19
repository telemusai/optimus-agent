//! Bounded aggregate report generator over correlation records.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::correlate::{read_records, CorrelationRecord, RECORD_SCHEMA_VERSION};

/// Report reads at most this many bytes of records.
pub const MAX_REPORT_INPUT_BYTES: u64 = 2 * 1024 * 1024;

/// Maximum disagreement samples embedded in a report.
pub const MAX_SAMPLES: usize = 10;

/// Build the bounded aggregate report. `actual_llm_calls_avoided` is ZERO in
/// Compare by construction; savings fields are hypothetical only.
pub fn build_report(records_path: &Path, generated_at: &str) -> Value {
    let records = read_records(records_path, MAX_REPORT_INPUT_BYTES);
    build_report_from(&records, generated_at)
}

pub fn build_report_from(records: &[CorrelationRecord], generated_at: &str) -> Value {
    let mut categories: BTreeMap<String, Value> = BTreeMap::new();
    let mut skip_reasons: BTreeMap<String, u64> = BTreeMap::new();
    let mut error_kinds: BTreeMap<String, u64> = BTreeMap::new();
    let mut durations: Vec<u64> = Vec::new();
    let mut samples: Vec<Value> = Vec::new();
    let mut modes: BTreeMap<String, u64> = BTreeMap::new();

    for record in records {
        let entry = categories
            .entry(record.category.clone())
            .or_insert_with(|| {
                json!({
                    "asked": 0u64, "skipped": 0u64, "agree": 0u64,
                    "disagree": 0u64, "noncomparable": 0u64,
                })
            });
        let asked = has_decision(record);
        let is_skip = record.skipped_reason.is_some();
        if is_skip {
            entry["skipped"] = json!(entry["skipped"].as_u64().unwrap_or(0) + 1);
            let reason = record.skipped_reason.clone().unwrap_or_default();
            // Coarsen request-level reasons to their prefix for the histogram.
            let prefix = reason.split(':').next().unwrap_or(&reason).to_string();
            *skip_reasons.entry(prefix).or_insert(0) += 1;
        } else if asked {
            entry["agree"] = json!(entry["agree"].as_u64().unwrap_or(0)
                + u64::from(record.agreement.as_deref() == Some("agree")));
            entry["disagree"] = json!(entry["disagree"].as_u64().unwrap_or(0)
                + u64::from(record.agreement.as_deref() == Some("disagree")));
            entry["noncomparable"] = json!(entry["noncomparable"].as_u64().unwrap_or(0)
                + u64::from(record.agreement.as_deref() == Some("noncomparable")));
        } else {
            entry["skipped"] = json!(entry["skipped"].as_u64().unwrap_or(0) + 1);
        }
        if asked {
            entry["asked"] = json!(entry["asked"].as_u64().unwrap_or(0) + 1);
        }
        if let Some(reason) = &record.skipped_reason {
            if reason.starts_with("request_failed") {
                *error_kinds.entry("request_failed".to_string()).or_insert(0) += 1;
            }
        }
        if let Some(duration) = record.duration_ms {
            durations.push(duration);
        }
        *modes.entry(record.mode.clone()).or_insert(0) += 1;
        if record.agreement.as_deref() == Some("disagree")
            && samples.len() < MAX_SAMPLES
        {
            samples.push(json!({
                "category": record.category,
                "question_id": record.question_id,
                "jev_value": record.selected_value,
                "baseline_actual": record.baseline_actual_choice,
                "session_id": record.session_id,
                "turn": record.turn,
            }));
        }
    }

    let durations_sorted_peek: Vec<u64> = {
        let mut bounded = durations.clone();
        bounded.sort_unstable();
        bounded
    };
    let avg = if durations.is_empty() {
        None
    } else {
        Some(durations.iter().sum::<u64>() / durations.len() as u64)
    };

    json!({
        "schema_version": RECORD_SCHEMA_VERSION,
        "generated_at": generated_at,
        "modes": modes,
        "total_records": records.len(),
        "category_coverage": categories,
        "skip_reasons": skip_reasons,
        "errors": { "by_kind": error_kinds },
        "latency_overhead": {
            "avg_duration_ms": avg,
            "max_duration_ms": durations_sorted_peek.last().copied(),
        },
        "disagreement_samples": samples,
        "actual_llm_calls_avoided": 0,
        "savings_note": "hypothetical only; counterfactual downstream success requires separate evals, never measured from logs",
    })
}

fn has_decision(record: &CorrelationRecord) -> bool {
    record.selected_value.is_some()
}

/// Write the report as pretty JSON.
pub fn write_report(report: &Value, output_path: &Path) -> Result<(), String> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
    std::fs::write(output_path, text).map_err(|e| e.to_string())
}
