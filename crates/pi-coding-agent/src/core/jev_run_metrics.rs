//! Opt-in local run measurements. No decision client, transcript text, or tool arguments.
//! `trace_observer` permits local metrics even in Off. Identities are SHA-256 hashes.

use std::collections::HashMap;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use pi_jev::config::{JevFeatures, JevMode};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::core::extensions::types::ExtensionEvent;

const MAX_TRACKED_RUNS: usize = 128;
const MAX_RUN_FILE_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Configuration {
    mode: JevMode,
    compaction_enabled: bool,
    features: JevFeatures,
}

/// Numeric fields describe observed events, not counterfactual savings.
#[derive(Debug, Clone, Serialize)]
pub struct JevRunRecord {
    pub schema_version: &'static str,
    pub request_id: String,
    /// SHA-256 of the native session id, never the raw identifier.
    pub session_id: String,
    pub session_id_encoding: &'static str,
    pub mode: String,
    pub compaction_enabled: Option<bool>,
    pub features: Option<JevFeatures>,
    pub mode_start: JevMode,
    pub mode_end: JevMode,
    pub compaction_start: bool,
    pub compaction_end: bool,
    pub mixed_configuration: bool,
    pub measurement_complete: bool,
    pub benchmark_eligible: bool,
    #[serde(rename = "agent_turns")]
    pub turns: Option<u64>,
    pub assistant_messages: Option<u64>,
    pub logical_primary_calls: Option<u64>,
    pub logical_primary_calls_basis: &'static str,
    /// Message completion events cannot establish physical provider attempts.
    pub primary_llm_calls: Option<u64>,
    pub transport_attempts: Option<u64>,
    #[serde(rename = "model_input_tokens")]
    pub input_tokens: Option<u64>,
    #[serde(rename = "model_output_tokens")]
    pub output_tokens: Option<u64>,
    #[serde(rename = "model_cache_read_tokens")]
    pub cache_read_tokens: Option<u64>,
    #[serde(rename = "model_cache_write_tokens")]
    pub cache_write_tokens: Option<u64>,
    #[serde(rename = "model_total_tokens")]
    pub total_tokens: Option<u64>,
    #[serde(rename = "elapsed_ms")]
    pub wall_clock_ms: u64,
    pub wall_clock_scope: &'static str,
}

#[derive(Default)]
struct UsageTotals {
    input: Option<u64>,
    output: Option<u64>,
    cache_read: Option<u64>,
    cache_write: Option<u64>,
    total: Option<u64>,
}

impl UsageTotals {
    fn zero() -> Self {
        Self {
            input: Some(0),
            output: Some(0),
            cache_read: Some(0),
            cache_write: Some(0),
            total: Some(0),
        }
    }

    fn add(&mut self, message: &Value) {
        let usage = message.get("usage");
        let value = |key: &str| usage.and_then(|usage| usage.get(key)).and_then(token_count);
        let counts = [
            value("input"),
            value("output"),
            value("cacheRead"),
            value("cacheWrite"),
            value("totalTokens"),
        ];
        // Native messages can carry default zero usage when the provider did not
        // report usage. Do not turn that absence into measured zero-token calls.
        if !counts
            .iter()
            .any(|count| count.is_some_and(|count| count > 0))
        {
            *self = Self::default();
            return;
        }
        add_count(&mut self.input, counts[0]);
        add_count(&mut self.output, counts[1]);
        add_count(&mut self.cache_read, counts[2]);
        add_count(&mut self.cache_write, counts[3]);
        add_count(&mut self.total, counts[4]);
    }
}

fn token_count(value: &Value) -> Option<u64> {
    let number = value.as_f64()?;
    (number.is_finite()
        && number >= 0.0
        && number.fract() == 0.0
        && number <= 9_007_199_254_740_991.0)
        .then_some(number as u64)
}

fn add_count(total: &mut Option<u64>, count: Option<u64>) {
    *total = total.and_then(|total| count.and_then(|count| total.checked_add(count)));
}

struct Run {
    request_id: String,
    started: Instant,
    initial: Configuration,
    current: Configuration,
    mixed: bool,
    complete: bool,
    turns: Option<u64>,
    assistant_messages: Option<u64>,
    logical_calls: Option<u64>,
    usage: UsageTotals,
}

impl Run {
    fn new(configuration: Configuration) -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            started: Instant::now(),
            initial: configuration,
            current: configuration,
            mixed: false,
            complete: true,
            turns: Some(0),
            assistant_messages: Some(0),
            logical_calls: Some(0),
            usage: UsageTotals::zero(),
        }
    }

    fn update_configuration(&mut self, configuration: Configuration) {
        self.mixed |= self.current != configuration;
        self.current = configuration;
    }

    fn message_end(&mut self, message: &Value) {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            return;
        }
        add_count(&mut self.assistant_messages, Some(1));
        let metadata_valid = ["api", "provider", "model"].into_iter().all(|key| {
            message
                .get(key)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty())
        }) && message.get("timestamp").and_then(token_count).is_some()
            && message.get("content").is_some_and(Value::is_array);
        let completed = matches!(
            message.get("stopReason").and_then(Value::as_str),
            Some("stop" | "length" | "toolUse")
        );
        if metadata_valid && completed {
            add_count(&mut self.logical_calls, Some(1));
        } else {
            // Error/abort messages can be synthetic without a provider request.
            self.logical_calls = None;
        }
        if metadata_valid {
            self.usage.add(message);
        } else {
            self.usage = UsageTotals::default();
        }
    }

    fn finish(self, session_id: &str) -> JevRunRecord {
        let complete = self.complete;
        let known = |value| if complete { value } else { None };
        JevRunRecord {
            schema_version: "jev.run/1",
            request_id: self.request_id,
            session_id: format!("{:x}", Sha256::digest(session_id.as_bytes())),
            session_id_encoding: "sha256",
            mode: if self.mixed {
                "mixed".into()
            } else {
                self.initial.mode.as_str().into()
            },
            compaction_enabled: (!self.mixed).then_some(self.initial.compaction_enabled),
            features: (!self.mixed).then_some(self.initial.features),
            mode_start: self.initial.mode,
            mode_end: self.current.mode,
            compaction_start: self.initial.compaction_enabled,
            compaction_end: self.current.compaction_enabled,
            mixed_configuration: self.mixed,
            measurement_complete: complete,
            benchmark_eligible: complete && !self.mixed && self.logical_calls.is_some(),
            turns: known(self.turns),
            assistant_messages: known(self.assistant_messages),
            logical_primary_calls: known(self.logical_calls),
            logical_primary_calls_basis:
                "non-error assistant_message_end; unknown after malformed/error/aborted completion",
            primary_llm_calls: None,
            transport_attempts: None,
            input_tokens: known(self.usage.input),
            output_tokens: known(self.usage.output),
            cache_read_tokens: known(self.usage.cache_read),
            cache_write_tokens: known(self.usage.cache_write),
            total_tokens: known(self.usage.total),
            wall_clock_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
            wall_clock_scope: "agent_start_to_agent_end_observer",
        }
    }
}

/// A worker-local bounded accumulator. Construction and disabled observations do no I/O.
pub struct JevRunMetrics {
    records_path: PathBuf,
    runs: Mutex<HashMap<String, Run>>,
    file_guard: Mutex<()>,
}

impl JevRunMetrics {
    pub fn new(agent_dir: PathBuf) -> Self {
        Self {
            records_path: agent_dir.join("jev").join("runs.jsonl"),
            runs: Mutex::new(HashMap::new()),
            file_guard: Mutex::new(()),
        }
    }

    /// Call once per native observer event, before the decision-mode gate.
    /// A disabled gap never counts events or reads the clock. If measurement
    /// resumes before AgentEnd, the partial row is excluded from benchmarks.
    pub fn observe(
        &self,
        session_id: &str,
        event: &ExtensionEvent,
        mode: JevMode,
        compaction_enabled: bool,
        features: JevFeatures,
    ) -> Option<JevRunRecord> {
        let mut runs = self.runs.lock().unwrap_or_else(|p| p.into_inner());
        if matches!(event, ExtensionEvent::SessionShutdown(_)) {
            runs.remove(session_id);
            return None;
        }
        let configuration = Configuration {
            mode,
            compaction_enabled,
            features,
        };
        if !features.trace_observer {
            if matches!(
                event,
                ExtensionEvent::AgentStart | ExtensionEvent::AgentEnd(_)
            ) {
                runs.remove(session_id);
            } else if let Some(run) = runs.get_mut(session_id) {
                run.update_configuration(configuration);
                run.complete = false;
            }
            return None;
        }
        if matches!(event, ExtensionEvent::AgentStart) {
            if runs.len() < MAX_TRACKED_RUNS || runs.contains_key(session_id) {
                runs.insert(session_id.to_string(), Run::new(configuration));
            }
            return None;
        }
        // Enabling metrics mid-run cannot recover the missing start/counters.
        let run = runs.get_mut(session_id)?;
        run.update_configuration(configuration);
        match event {
            ExtensionEvent::TurnStart(payload) => {
                let valid = payload.turn_index.is_finite()
                    && payload.turn_index >= 0.0
                    && payload.turn_index.fract() == 0.0;
                add_count(&mut run.turns, valid.then_some(1));
            }
            ExtensionEvent::MessageEnd(payload) => run.message_end(&payload.message),
            ExtensionEvent::AgentEnd(_) => {
                let record = runs.remove(session_id)?.finish(session_id);
                drop(runs);
                self.write_record(&record);
                return Some(record);
            }
            _ => {}
        }
        None
    }

    fn write_record(&self, record: &JevRunRecord) {
        let _guard = self.file_guard.lock().unwrap_or_else(|p| p.into_inner());
        let Some(parent) = self.records_path.parent() else {
            return;
        };
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
        if std::fs::metadata(&self.records_path)
            .is_ok_and(|metadata| metadata.len() >= MAX_RUN_FILE_BYTES)
        {
            let rotated = self.records_path.with_file_name("runs.jsonl.1");
            if rotated.exists() && std::fs::remove_file(&rotated).is_err() {
                return;
            }
            if std::fs::rename(&self.records_path, rotated).is_err() {
                return;
            }
        }
        let Ok(mut line) = serde_json::to_vec(record) else {
            return;
        };
        line.push(b'\n');
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        options.mode(0o600);
        if let Ok(mut file) = options.open(&self.records_path) {
            let _ = file.write_all(&line);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extensions::types::{AgentEndPayload, MessageEndPayload, TurnStartPayload};
    use serde_json::json;
    use std::time::Duration;

    fn enabled() -> JevFeatures {
        JevFeatures {
            trace_observer: true,
            ..Default::default()
        }
    }
    fn end() -> ExtensionEvent {
        ExtensionEvent::AgentEnd(AgentEndPayload { messages: vec![] })
    }
    fn turn() -> ExtensionEvent {
        ExtensionEvent::TurnStart(TurnStartPayload {
            turn_index: 0.0,
            timestamp: 0.0,
        })
    }
    fn assistant(usage: Value, reason: &str) -> ExtensionEvent {
        ExtensionEvent::MessageEnd(MessageEndPayload {
            message: json!({
                "role": "assistant", "api": "faux", "provider": "faux", "model": "faux",
                "timestamp": 1.0, "stopReason": reason, "usage": usage,
                "content": [{"type": "text", "text": "SECRET raw prompt args credential"}]
            }),
        })
    }
    fn usage() -> Value {
        json!({"input":10.0,"output":2.0,"cacheRead":3.0,"cacheWrite":4.0,"totalTokens":19.0})
    }

    #[test]
    fn disabled_off_never_creates_a_run_or_file() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = JevRunMetrics::new(dir.path().join("not-created"));
        for event in [
            ExtensionEvent::AgentStart,
            turn(),
            assistant(usage(), "stop"),
            end(),
        ] {
            assert!(metrics
                .observe(
                    "session",
                    &event,
                    JevMode::Off,
                    false,
                    JevFeatures::default()
                )
                .is_none());
        }
        assert!(metrics.runs.lock().unwrap().is_empty());
        assert!(!dir.path().join("not-created").exists());
    }

    #[test]
    fn measures_all_mode_and_compaction_axes_without_reading_content() {
        for mode in [
            JevMode::Off,
            JevMode::Compare,
            JevMode::Active,
            JevMode::CompareAndActive,
        ] {
            for compact in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let metrics = JevRunMetrics::new(dir.path().to_path_buf());
                let _ = metrics.observe(
                    "session-SECRET",
                    &ExtensionEvent::AgentStart,
                    mode,
                    compact,
                    enabled(),
                );
                metrics
                    .runs
                    .lock()
                    .unwrap()
                    .get_mut("session-SECRET")
                    .unwrap()
                    .started = Instant::now() - Duration::from_secs(2);
                let _ = metrics.observe("session-SECRET", &turn(), mode, compact, enabled());
                let _ = metrics.observe(
                    "session-SECRET",
                    &assistant(usage(), "toolUse"),
                    mode,
                    compact,
                    enabled(),
                );
                let _ = metrics.observe(
                    "session-SECRET",
                    &assistant(usage(), "stop"),
                    mode,
                    compact,
                    enabled(),
                );
                let record = metrics
                    .observe("session-SECRET", &end(), mode, compact, enabled())
                    .unwrap();
                assert_eq!(record.turns, Some(1));
                assert_eq!(record.assistant_messages, Some(2));
                assert_eq!(record.logical_primary_calls, Some(2));
                assert_eq!(record.primary_llm_calls, None);
                assert_eq!(record.transport_attempts, None);
                assert_eq!(record.input_tokens, Some(20));
                assert_eq!(record.output_tokens, Some(4));
                assert_eq!(record.cache_read_tokens, Some(6));
                assert_eq!(record.cache_write_tokens, Some(8));
                assert_eq!(record.total_tokens, Some(38));
                assert_eq!(record.mode, mode.as_str());
                assert_eq!(record.compaction_enabled, Some(compact));
                assert!(record.benchmark_eligible);
                assert!(record.wall_clock_ms >= 2000);
                let text = std::fs::read_to_string(&metrics.records_path).unwrap();
                assert!(!text.contains("SECRET"));
                assert!(!text.contains("credential"));
                let wire: Value = serde_json::from_str(text.trim()).unwrap();
                assert_eq!(wire["schema_version"], "jev.run/1");
                assert_eq!(wire["agent_turns"], 1);
                assert_eq!(wire["model_input_tokens"], 20);
                assert_eq!(wire["session_id_encoding"], "sha256");
                assert_eq!(wire["session_id"].as_str().unwrap().len(), 64);
                assert!(Uuid::parse_str(wire["request_id"].as_str().unwrap()).is_ok());
                assert!(metrics.runs.lock().unwrap().is_empty());
            }
        }
    }

    #[test]
    fn unknown_or_default_usage_stays_unknown_per_field() {
        for reported in [
            Value::Null,
            json!({}),
            json!({"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0}),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let metrics = JevRunMetrics::new(dir.path().to_path_buf());
            let _ = metrics.observe(
                "s",
                &ExtensionEvent::AgentStart,
                JevMode::Compare,
                false,
                enabled(),
            );
            let _ = metrics.observe(
                "s",
                &assistant(reported, "stop"),
                JevMode::Compare,
                false,
                enabled(),
            );
            let record = metrics
                .observe("s", &end(), JevMode::Compare, false, enabled())
                .unwrap();
            assert_eq!(record.input_tokens, None);
            assert_eq!(record.output_tokens, None);
            assert_eq!(record.cache_read_tokens, None);
            assert_eq!(record.total_tokens, None);
            assert_eq!(record.logical_primary_calls, Some(1));
        }
        let mut totals = UsageTotals::zero();
        totals.add(&json!({"usage":{"input":10,"output":-1,"cacheRead":1.5}}));
        assert_eq!(totals.input, Some(10));
        assert_eq!(totals.output, None);
        assert_eq!(totals.cache_read, None);
        assert_eq!(totals.cache_write, None);
        totals.add(&json!({"usage":usage()}));
        assert_eq!(totals.input, Some(20));
        assert_eq!(
            totals.output, None,
            "later complete usage cannot recover missing usage"
        );
    }

    #[test]
    fn errors_and_malformed_assistants_do_not_claim_primary_calls() {
        for reason in ["error", "aborted", "unknown"] {
            let dir = tempfile::tempdir().unwrap();
            let metrics = JevRunMetrics::new(dir.path().to_path_buf());
            let _ = metrics.observe(
                "s",
                &ExtensionEvent::AgentStart,
                JevMode::Active,
                false,
                enabled(),
            );
            let _ = metrics.observe(
                "s",
                &assistant(usage(), reason),
                JevMode::Active,
                false,
                enabled(),
            );
            let record = metrics
                .observe("s", &end(), JevMode::Active, false, enabled())
                .unwrap();
            assert_eq!(record.assistant_messages, Some(1));
            assert_eq!(record.logical_primary_calls, None);
            assert!(!record.benchmark_eligible);
        }
        let mut run = Run::new(Configuration {
            mode: JevMode::Off,
            compaction_enabled: false,
            features: enabled(),
        });
        run.message_end(&json!({"role":"assistant", "usage": usage()}));
        assert_eq!(run.logical_calls, None);
        assert_eq!(run.usage.input, None);
    }

    #[test]
    fn observed_configuration_changes_exclude_the_run_from_grouping() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = JevRunMetrics::new(dir.path().to_path_buf());
        let _ = metrics.observe(
            "s",
            &ExtensionEvent::AgentStart,
            JevMode::Off,
            false,
            enabled(),
        );
        let mut changed = enabled();
        changed.verification = true;
        let _ = metrics.observe("s", &turn(), JevMode::CompareAndActive, true, changed);
        let record = metrics
            .observe("s", &end(), JevMode::Off, false, enabled())
            .unwrap();
        assert_eq!(record.mode, "mixed");
        assert_eq!(record.compaction_enabled, None);
        assert_eq!(record.features, None);
        assert!(record.mixed_configuration);
        assert!(!record.benchmark_eligible);
        assert!(record.measurement_complete);
        // A feature-only transition is also mixed.
        let _ = metrics.observe(
            "s",
            &ExtensionEvent::AgentStart,
            JevMode::Off,
            false,
            enabled(),
        );
        let record = metrics
            .observe("s", &end(), JevMode::Off, false, changed)
            .unwrap();
        assert!(record.mixed_configuration);
    }

    #[test]
    fn trace_opt_out_gaps_do_not_count_or_write_and_resumed_rows_are_partial() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = JevRunMetrics::new(dir.path().to_path_buf());
        let _ = metrics.observe(
            "s",
            &ExtensionEvent::AgentStart,
            JevMode::Off,
            false,
            enabled(),
        );
        let _ = metrics.observe(
            "s",
            &assistant(usage(), "stop"),
            JevMode::Off,
            false,
            JevFeatures::default(),
        );
        assert!(!metrics.records_path.exists());
        assert_eq!(
            metrics.runs.lock().unwrap()["s"].assistant_messages,
            Some(0)
        );
        let record = metrics
            .observe("s", &end(), JevMode::Off, false, enabled())
            .unwrap();
        assert!(record.mixed_configuration);
        assert!(!record.measurement_complete);
        assert!(!record.benchmark_eligible);
        assert_eq!(record.turns, None);
        assert_eq!(record.logical_primary_calls, None);
        assert_eq!(record.input_tokens, None);
        let before = std::fs::read(&metrics.records_path).unwrap();
        let _ = metrics.observe(
            "s",
            &ExtensionEvent::AgentStart,
            JevMode::Off,
            false,
            enabled(),
        );
        assert!(metrics
            .observe("s", &end(), JevMode::Off, false, JevFeatures::default())
            .is_none());
        assert_eq!(std::fs::read(&metrics.records_path).unwrap(), before);
    }

    #[test]
    fn missing_starts_and_capacity_overflow_do_not_invent_runs() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = JevRunMetrics::new(dir.path().to_path_buf());
        assert!(metrics
            .observe("s", &turn(), JevMode::Off, false, enabled())
            .is_none());
        assert!(metrics
            .observe("s", &end(), JevMode::Off, false, enabled())
            .is_none());
        assert!(!metrics.records_path.exists());
        for index in 0..MAX_TRACKED_RUNS + 5 {
            let _ = metrics.observe(
                &format!("session-{index}"),
                &ExtensionEvent::AgentStart,
                JevMode::Off,
                false,
                enabled(),
            );
        }
        assert_eq!(metrics.runs.lock().unwrap().len(), MAX_TRACKED_RUNS);
        assert!(metrics
            .observe("session-128", &end(), JevMode::Off, false, enabled())
            .is_none());
    }

    #[test]
    fn local_writer_is_bounded_and_filesystem_failure_does_not_abort_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = JevRunMetrics::new(dir.path().to_path_buf());
        std::fs::create_dir_all(metrics.records_path.parent().unwrap()).unwrap();
        std::fs::write(
            &metrics.records_path,
            vec![b' '; MAX_RUN_FILE_BYTES as usize],
        )
        .unwrap();
        for _ in 0..2 {
            let _ = metrics.observe(
                "s",
                &ExtensionEvent::AgentStart,
                JevMode::Off,
                false,
                enabled(),
            );
            assert!(metrics
                .observe("s", &end(), JevMode::Off, false, enabled())
                .is_some());
        }
        assert_eq!(
            std::fs::metadata(metrics.records_path.with_file_name("runs.jsonl.1"))
                .unwrap()
                .len(),
            MAX_RUN_FILE_BYTES
        );
        assert!(std::fs::metadata(&metrics.records_path).unwrap().len() < MAX_RUN_FILE_BYTES);
        let occupied = dir.path().join("file-not-directory");
        std::fs::write(&occupied, "occupied").unwrap();
        let failed = JevRunMetrics::new(occupied);
        let _ = failed.observe(
            "s",
            &ExtensionEvent::AgentStart,
            JevMode::Off,
            false,
            enabled(),
        );
        assert!(failed
            .observe("s", &end(), JevMode::Off, false, enabled())
            .is_some());
        assert!(!failed.records_path.exists());
    }
}
