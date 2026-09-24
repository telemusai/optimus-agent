//! Typed opt-in script status, independent of stdout/stderr and transport success.
//!
//! Task verification designation (CTRL-001): a report counts as verification
//! evidence only when its public receipt carries the strictly-shaped object
//! `verification: {"kind": "task_check", "label": ...}` (see
//! [`VERIFICATION_DESIGNATION_KEY`]). Callers emit it through the supported
//! script path, e.g. `rlm.execution.report_script_result(result, receipt=
//! {"verification": {"kind": "task_check", "label": "focused scope: ..."}})`.
//! A plain exit 0 is never verification; the outcome stays the host-measured
//! process result, and the designation never makes the task complete or
//! verified by itself.
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Details field of an ipython tool result carrying the structured reports
/// (serde rename of `IpythonToolDetails::execution_reports`).
pub const EXECUTION_REPORTS_DETAILS_KEY: &str = "executionReports";

/// Receipt key for the explicit task-verification designation (CTRL-001).
/// A report counts as verification evidence ONLY when the agent opts in with
/// this strictly-shaped object; a plain successful exit is never verification.
pub const VERIFICATION_DESIGNATION_KEY: &str = "verification";
/// Only supported designation kind in this version.
pub const VERIFICATION_KIND_TASK_CHECK: &str = "task_check";
/// Bound for the designation label (public, human-readable, non-authoritative).
pub const MAX_VERIFICATION_LABEL_CHARS: usize = 128;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScriptExecutionReport {
    pub schema: String,
    pub stage: String,
    pub script_id: Option<String>,
    pub exit_code: i64,
    pub duration_seconds: f64,
    pub expected_exit_codes: Vec<i64>,
    pub is_error: bool,
    pub receipt: Option<Value>,
}

impl ScriptExecutionReport {
    pub fn failed(&self) -> bool {
        self.exit_code < 0 || !self.expected_exit_codes.contains(&self.exit_code)
            || self.receipt.as_ref().is_some_and(|value| {
                value.get("isError").and_then(Value::as_bool) == Some(true)
                    || value.get("fatal").and_then(Value::as_bool) == Some(true)
                    || matches!(value.get("status").and_then(Value::as_str), Some("error" | "fatal" | "aborted"))
            })
    }

    /// The explicit task-verification designation, when the agent marked this
    /// report as the task's verification. The OUTCOME still comes only from
    /// the host-measured process result; the label is descriptive metadata.
    pub fn verification_label(&self) -> Option<&str> {
        self.receipt
            .as_ref()?
            .get(VERIFICATION_DESIGNATION_KEY)?
            .get("label")
            .and_then(Value::as_str)
            .filter(|label| !label.trim().is_empty())
    }

    fn validate(&self) -> Result<(), String> {
        let code_valid = |code: i64| (i32::MIN as i64..=u32::MAX as i64).contains(&code);
        if self.schema != "optimus.script-result.v1" || self.stage.trim().is_empty()
            || self.stage.chars().count() > 128
            || self.script_id.as_ref().is_some_and(|id| id.trim().is_empty() || id.chars().count() > 256)
            || !code_valid(self.exit_code) || self.expected_exit_codes.is_empty()
            || self.expected_exit_codes.len() > 32 || !self.expected_exit_codes.iter().all(|code| code_valid(*code))
            || !self.duration_seconds.is_finite() || self.duration_seconds < 0.0 {
            return Err("Invalid structured script execution report".into());
        }
        if let Some(receipt) = &self.receipt {
            if !receipt.is_object() || serde_json::to_vec(receipt).map_err(|e| e.to_string())?.len() > 16 * 1024 {
                return Err("Script receipt must be a public JSON object up to 16 KiB".into());
            }
            for flag in ["isError", "fatal"] {
                if receipt.get(flag).is_some_and(|value| !value.is_boolean()) {
                    return Err(format!("Invalid script receipt {flag}"));
                }
            }
            if receipt.get("status").is_some_and(|value| !matches!(value.as_str(), Some("ok" | "error" | "fatal" | "aborted"))) {
                return Err("Invalid script receipt status".into());
            }
            // CTRL-001: the verification designation must be an exact,
            // strictly-shaped object. Anything else fails the whole report:
            // verification evidence can never be manufactured from prose,
            // arbitrary keys, or a bare exit code.
            if let Some(designation) = receipt.get(VERIFICATION_DESIGNATION_KEY) {
                let object = designation.as_object()
                    .ok_or_else(|| "Invalid verification designation".to_string())?;
                if object.len() != 2
                    || object.get("kind").and_then(Value::as_str) != Some(VERIFICATION_KIND_TASK_CHECK)
                    || object.get("label").and_then(Value::as_str).is_none_or(|label| {
                        label.trim().is_empty() || label.chars().count() > MAX_VERIFICATION_LABEL_CHARS
                    })
                {
                    return Err("Invalid verification designation".into());
                }
            }
        }
        if self.is_error != self.failed() {
            return Err("Script report isError contradicts process/receipt outcome".into());
        }
        Ok(())
    }
}

pub fn parse_execution_reports(value: Option<&Value>) -> Result<Option<Vec<ScriptExecutionReport>>, String> {
    let Some(value) = value else { return Ok(None); };
    let array = value.as_array().ok_or("executionReports must be an array")?;
    if array.len() > 32 { return Err("Too many script execution reports".into()); }
    let reports: Vec<ScriptExecutionReport> = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    for report in &reports { report.validate()?; }
    Ok(Some(reports))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn report(code: i64, expected: Value, receipt: Value, error: bool) -> Value {
        json!([{ "schema":"optimus.script-result.v1", "stage":"process", "scriptId":"s1", "exitCode":code,
            "durationSeconds":0.1, "expectedExitCodes":expected, "isError":error, "receipt":receipt }])
    }
    #[test]
    fn structured_success_failure_and_handled_nonzero() {
        assert!(parse_execution_reports(None).unwrap().is_none());
        for (code, expected, error) in [(0, json!([0]), false), (1,json!([0]),true), (1,json!([0,1]),false)] {
            let parsed = parse_execution_reports(Some(&report(code,expected,Value::Null,error))).unwrap().unwrap();
            assert_eq!(parsed[0].failed(),error);
        }
    }
    #[test]
    fn fatal_signal_and_source_guards_cannot_be_expected_away() {
        for receipt in [json!({"status":"fatal","fatal":true,"errorCode":"SOURCE_CHANGED"}), json!({"status":"aborted"}), json!({"isError":true})] {
            assert!(parse_execution_reports(Some(&report(0,json!([0]),receipt,true))).unwrap().unwrap()[0].failed());
        }
        assert!(parse_execution_reports(Some(&report(-9,json!([-9]),Value::Null,true))).unwrap().unwrap()[0].failed());
    }
    #[test]
    fn never_scan_text_and_reject_forged_or_malformed_flags() {
        assert!(!parse_execution_reports(Some(&report(0,json!([0]),json!({"note":"error Traceback PermissionError"}),false))).unwrap().unwrap()[0].failed());
        assert!(parse_execution_reports(Some(&report(7,json!([0]),Value::Null,false))).is_err());
        assert!(parse_execution_reports(Some(&report(0,json!([0]),json!({"fatal":"false"}),false))).is_err());
        assert!(parse_execution_reports(Some(&json!({"isError":false}))).is_err());
    }
    #[test]
    fn verification_designation_is_strictly_shaped_and_optional() {
        // No designation: a plain successful exit is never verification.
        let plain = parse_execution_reports(Some(&report(0,json!([0]),Value::Null,false))).unwrap().unwrap();
        assert_eq!(plain[0].verification_label(), None);
        // Exact designation exposes the label; outcome stays process-driven.
        let designated = parse_execution_reports(Some(&report(0,json!([0]),
            json!({"verification":{"kind":"task_check","label":"focused scope: cargo test -- x"}}),false))).unwrap().unwrap();
        assert_eq!(designated[0].verification_label(), Some("focused scope: cargo test -- x"));
        assert!(!designated[0].failed());
        // A failed designated report stays failed (honest Failed evidence).
        let failed = parse_execution_reports(Some(&report(1,json!([0]),
            json!({"verification":{"kind":"task_check","label":"focused scope"}}),true))).unwrap().unwrap();
        assert_eq!(failed[0].verification_label(), Some("focused scope"));
        assert!(failed[0].failed());
        // Wrong kind, missing/extra keys, empty or oversized labels all reject.
        for receipt in [
            json!({"verification":{"kind":"assert","label":"x"}}),
            json!({"verification":{"kind":"task_check"}}),
            json!({"verification":{"kind":"task_check","label":"x","note":"y"}}),
            json!({"verification":{"kind":"task_check","label":""}}),
            json!({"verification":{"kind":"task_check","label":"x".repeat(129)}}),
            json!({"verification":"task_check"}),
        ] {
            assert!(parse_execution_reports(Some(&report(0,json!([0]),receipt.clone(),false))).is_err(), "{receipt}");
        }
    }
}
